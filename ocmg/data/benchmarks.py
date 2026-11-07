"""Loaders for the two public benchmarks.

Both loaders produce (conversation_id, List[Turn], List[Question]), so the rest
of the pipeline never learns whether its input came from a benchmark or the
synthetic generator.

LoCoMo specifics handled here:

* Session dates. Every session carries a real date string such as
  "1:56 pm on 8 May, 2023". These are parsed, so temporal normalisation and
  temporal questions see the true chronology rather than invented dates.
* Evidence. Every question lists the dialogue ids of its supporting turns
  (e.g. "D1:3"). These are kept on the Question and each Turn keeps its own
  dia_id, so record-level utility labels can be validated against annotated
  evidence. A handful of ids in the release are malformed and are normalised.
* Categories. The release uses integers. Their meaning was checked against the
  data itself: category 1 questions average 3.1 evidence turns (98% have more
  than one), so 1 is multi-hop and 4, at 1.07, is single-hop.
* Adversarial questions (category 5) have no "answer" field, only an
  "adversarial_answer" that the question is designed to elicit. That value is
  the wrong answer, so it is stored separately and the reference is an
  abstention.
* Image turns. Some turns share a photo described by "blip_caption"; the
  caption is appended so the content is retrievable.

Download:
  LoCoMo         https://github.com/snap-research/locomo (data/locomo10.json)
  LongMemEval    https://github.com/xiaowu0162/LongMemEval
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

from ..types import Question, Turn

DAY = 86400.0

LOCOMO_CATEGORIES: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# Reference answer used for adversarial questions: the correct behaviour is
# to say the information is not in the conversation.
ABSTAIN_ANSWER = "Not mentioned in the conversation"

_LOCOMO_DATE_FMT = "%I:%M %p on %d %B, %Y"
_DIA_ID_RE = re.compile(r"D:?(\d+):0*(\d+)")


def parse_locomo_datetime(value: str) -> float:
    """Parse '1:56 pm on 8 May, 2023' into a POSIX timestamp.

    The naive datetime is interpreted in local time, matching the
    datetime.fromtimestamp call used by temporal normalisation, so the
    round trip never shifts a date.
    """
    return datetime.strptime(value.strip(), _LOCOMO_DATE_FMT).timestamp()


def normalise_evidence(raw_ids: List[str]) -> List[str]:
    """Clean evidence ids: split joined ids, drop separators and leading zeros.

    Examples from the release: 'D8:6; D9:17' -> ['D8:6', 'D9:17'],
    'D:11:26' -> ['D11:26'], 'D30:05' -> ['D30:5'], 'D' -> [].
    """
    out: List[str] = []
    for item in raw_ids or []:
        for m in _DIA_ID_RE.finditer(str(item)):
            dia = f"D{int(m.group(1))}:{int(m.group(2))}"
            if dia not in out:
                out.append(dia)
    return out


def _turn_text(utt: dict) -> str:
    text = str(utt.get("text", "")).strip()
    caption = utt.get("blip_caption")
    if caption:
        text = f"{text} [shares an image: {str(caption).strip()}]".strip()
    return text


def load_locomo(
    path: str, return_report: bool = False
) -> List[Tuple[str, List[Turn], List[Question]]]:
    """Parse the LoCoMo release format (locomo10.json)."""
    with open(path, "r", encoding="utf8") as fh:
        raw = json.load(fh)

    out = []
    report = {"conversations": 0, "sessions": 0, "turns": 0, "questions": 0,
              "evidence_ids": 0, "evidence_unresolved": [], "image_turns": 0,
              "per_category": {}}

    for ci, sample in enumerate(raw):
        cid = str(sample.get("sample_id", f"locomo{ci:03d}"))
        conv = sample.get("conversation", {})
        session_keys = sorted(
            [k for k in conv
             if re.fullmatch(r"session_\d+", k) and isinstance(conv[k], list)],
            key=lambda k: int(k.split("_")[1]),
        )

        turns: List[Turn] = []
        tix = 0
        last_session_time = 0.0
        for key in session_keys:
            session_no = int(key.split("_")[1])
            date_str = conv.get(f"{key}_date_time")
            if not date_str:
                raise ValueError(f"{cid}: {key} has no date_time field")
            session_time = parse_locomo_datetime(date_str)
            last_session_time = max(last_session_time, session_time)
            for pos, utt in enumerate(conv[key]):
                if utt.get("blip_caption"):
                    report["image_turns"] += 1
                turns.append(
                    Turn(
                        session_id=session_no,
                        turn_index=tix,
                        speaker=str(utt.get("speaker", "user")),
                        text=_turn_text(utt),
                        # Turns keep their order inside a session; one minute
                        # apart is a placeholder because the release has no
                        # per-turn clock.
                        timestamp=session_time + pos * 60.0,
                        dia_id=str(utt.get("dia_id")) if utt.get("dia_id") else None,
                    )
                )
                tix += 1
            report["sessions"] += 1

        known_ids = {t.dia_id for t in turns if t.dia_id}
        questions: List[Question] = []
        for qi, qa in enumerate(sample.get("qa", []) or []):
            cat_raw = qa.get("category")
            category = LOCOMO_CATEGORIES.get(int(cat_raw), str(cat_raw)) \
                if str(cat_raw).isdigit() else str(cat_raw)
            evidence = normalise_evidence(qa.get("evidence", []))
            unresolved = [e for e in evidence if e not in known_ids]
            for e in unresolved:
                report["evidence_unresolved"].append(f"{cid}:q{qi}:{e}")
            evidence = [e for e in evidence if e in known_ids]
            report["evidence_ids"] += len(evidence)

            if category == "adversarial":
                answer = ABSTAIN_ANSWER
                adversarial = qa.get("adversarial_answer")
            else:
                answer = str(qa.get("answer", ""))
                adversarial = None

            questions.append(
                Question(
                    conversation_id=cid,
                    question_id=f"{cid}:q{qi}",
                    text=str(qa.get("question", "")),
                    answer=answer,
                    category=category,
                    # Answerable questions are supported by their annotated
                    # evidence turns; adversarial ones have none by design.
                    supporting_fact_ids=[] if category == "adversarial" else list(evidence),
                    # Questions are posed after the final session.
                    asked_at=last_session_time + DAY,
                    evidence_ids=evidence,
                    adversarial_answer=str(adversarial) if adversarial else None,
                )
            )
            report["per_category"][category] = report["per_category"].get(category, 0) + 1

        if turns and questions:
            out.append((cid, turns, questions))
            report["conversations"] += 1
            report["turns"] += len(turns)
            report["questions"] += len(questions)

    if return_report:
        return out, report
    return out


def load_longmemeval(path: str) -> List[Tuple[str, List[Turn], List[Question]]]:
    """Parse the LongMemEval-S release format."""
    with open(path, "r", encoding="utf8") as fh:
        raw = json.load(fh)

    out = []
    for ci, item in enumerate(raw):
        cid = str(item.get("question_id", f"lme{ci:03d}"))
        turns: List[Turn] = []
        tix = 0
        for si, session in enumerate(item.get("haystack_sessions", []) or []):
            base = 1_700_000_000.0 + si * 3 * DAY
            for utt in session or []:
                turns.append(
                    Turn(
                        session_id=si,
                        turn_index=tix,
                        speaker=str(utt.get("role", "user")),
                        text=str(utt.get("content", "")),
                        timestamp=base + tix * 60.0,
                    )
                )
                tix += 1
        questions = [
            Question(
                conversation_id=cid,
                question_id=cid,
                text=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                category=str(item.get("question_type", "single_hop")),
                supporting_fact_ids=[],
                asked_at=1_700_000_000.0 + 400 * DAY,
            )
        ]
        if turns:
            out.append((cid, turns, questions))
    return out


def load_dataset(name: str, path: str | None = None, **kwargs):
    """Dispatch by name; falls back to the synthetic generator."""
    if name == "locomo":
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                "LoCoMo not found. Download data/locomo10.json from "
                "https://github.com/snap-research/locomo"
            )
        return load_locomo(path)
    if name in ("longmemeval", "longmemeval_s"):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                "LongMemEval not found. Download from "
                "https://github.com/xiaowu0162/LongMemEval"
            )
        return load_longmemeval(path)
    from .synthetic import generate_dataset

    return generate_dataset(**kwargs)
