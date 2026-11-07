"""Stage 1: ingestion and typing.

Three operations, in order:

1. Rule-based removal of structurally uninformative turns (greetings, bare
   acknowledgements). Passing these through segmentation and typing inflates
   cost without contributing information.
2. Topic-coherent segmentation. Boundaries are placed at local minima of the
   cosine similarity between consecutive utterance embeddings, rather than at
   fixed turn windows, because a session commonly spans unrelated topics.
3. Type routing into profile / episodic / factual.

The router here is a rule-based classifier over surface cues. This is a
deliberate design choice, not a shortcut: every comparable system routes by
prompting a language model, which costs one generative call per segment and is
not reproducible across runs. Keeping the router free is part of the cost
argument the study makes.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Sequence

import numpy as np

from .config import IngestConfig
from .embedding import tokenize
from .types import EPISODIC, FACTUAL, PROFILE, Candidate, Turn

# Turns that carry no retrievable content.
_FILLER = {
    "ok", "okay", "sure", "thanks", "thank you", "yeah", "yes", "no", "hi",
    "hello", "hey", "got it", "sounds good", "np", "cool", "nice", "great",
    "you're welcome", "see you", "bye", "haha", "lol", "mhm", "right",
}

_PROFILE_CUES = re.compile(
    r"\b(i am|i'm|my name|i live|i work|i prefer|i like|i love|i hate|i always|"
    r"i usually|my favourite|my favorite|i have a|i own|allergic|vegetarian|vegan|"
    r"my [a-z]+ (?:is|are)|my [a-z]+ [a-z]+ (?:is|are))\b",
    re.I,
)
_EPISODIC_CUES = re.compile(
    r"\b(yesterday|today|tomorrow|last week|last month|last year|this morning|"
    r"next week|on monday|on tuesday|on wednesday|on thursday|on friday|"
    r"on saturday|on sunday|in \d{4}|ago|just|finally|started|finished|moved|"
    r"bought|sold|visited|went|arrived|left|joined|quit|met)\b",
    re.I,
)
_RELATIVE_TIME = re.compile(
    r"\b(yesterday|today|tomorrow|last week|last month|next week|this morning|"
    r"an hour ago|two days ago|a week ago)\b",
    re.I,
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")


def is_uninformative(turn: Turn, cfg: IngestConfig) -> bool:
    text = turn.text.strip().lower().rstrip(".!?")
    if len(text) < cfg.drop_short_turns_below_chars and text in _FILLER:
        return True
    return text in _FILLER


def normalise_temporal(text: str, session_time: float) -> str:
    """Resolve relative expressions against the session date.

    'last week' resolves differently depending on when it was uttered, and
    leaving it unresolved forces every downstream component to reconstruct
    chronology from ambiguous phrasing.
    """
    base = datetime.fromtimestamp(session_time)
    offsets = {
        "yesterday": -1, "today": 0, "tomorrow": 1, "this morning": 0,
        "last week": -7, "next week": 7, "last month": -30,
        "a week ago": -7, "two days ago": -2, "an hour ago": 0,
    }

    def _sub(m: re.Match) -> str:
        key = m.group(0).lower()
        days = offsets.get(key)
        if days is None:
            return m.group(0)
        resolved = (base + timedelta(days=days)).strftime("%Y-%m-%d")
        return f"{m.group(0)} [{resolved}]"

    return _RELATIVE_TIME.sub(_sub, text)


def route_type(text: str) -> str:
    """Assign a segment to one of the three record types."""
    if _PROFILE_CUES.search(text):
        return PROFILE
    if _EPISODIC_CUES.search(text):
        return EPISODIC
    return FACTUAL


def extract_entities(text: str) -> List[str]:
    ents = set(_ENTITY_RE.findall(text))
    ents.update(_NUMBER_RE.findall(text))
    return sorted(ents)


def segment_session(
    turns: Sequence[Turn], embedder, cfg: IngestConfig
) -> List[List[Turn]]:
    """Split one session into topic-coherent runs of turns."""
    if not turns:
        return []
    if cfg.naive_chunk_turns > 0:
        # Fixed-window chunking: no boundary detection, no coherence test.
        # This is the naive top-k RAG baseline's ingest, and it has to be a
        # different segmentation to be a different baseline at all. Routing
        # every segment to one type is not enough on its own, because
        # record_type never enters retrieval ranking - it feeds three
        # estimator features and the prompt string - so with governance off a
        # merely untyped store is the same computation on the same records.
        n = cfg.naive_chunk_turns
        return [list(turns[i:i + n]) for i in range(0, len(turns), n)]
    if len(turns) == 1:
        return [list(turns)]

    vecs = embedder.encode([t.text for t in turns])
    sims = np.array(
        [float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(turns) - 1)],
        dtype=np.float32,
    )

    boundaries = set()
    for i in range(len(sims)):
        left = sims[i - 1] if i > 0 else sims[i]
        right = sims[i + 1] if i < len(sims) - 1 else sims[i]
        depth = max(left, right) - sims[i]
        if depth >= cfg.boundary_depth:
            boundaries.add(i)

    segments, current = [], [turns[0]]
    for i in range(1, len(turns)):
        too_long = len(current) >= cfg.max_segment_turns
        # A gap in the original turn indices means a turn was dropped by the
        # filter. Segments must stay contiguous in the source stream, otherwise
        # an informative turn is glued to unrelated chatter across the hole and
        # the resulting record is neither cleanly useful nor cleanly discardable.
        discontinuous = turns[i].turn_index != turns[i - 1].turn_index + 1
        if (i - 1) in boundaries or too_long or discontinuous:
            segments.append(current)
            current = []
        current.append(turns[i])
    if current:
        segments.append(current)
    return segments


def build_candidates(
    conversation_id: str, turns: Sequence[Turn], embedder, cfg: IngestConfig
) -> List[Candidate]:
    """Full Stage 1: filter, segment, normalise time, type, embed."""
    kept = [t for t in turns if not is_uninformative(t, cfg)]
    by_session: dict[int, List[Turn]] = {}
    for t in kept:
        by_session.setdefault(t.session_id, []).append(t)

    candidates: List[Candidate] = []
    for session_id in sorted(by_session):
        session_turns = sorted(by_session[session_id], key=lambda t: t.turn_index)
        for seg in segment_session(session_turns, embedder, cfg):
            raw = " ".join(t.text for t in seg).strip()
            if len(raw) < cfg.min_segment_chars:
                continue
            session_time = seg[0].timestamp
            text = normalise_temporal(raw, session_time)
            # A segment inherits a supporting fact if any of its turns carried
            # one; this is synthetic-data bookkeeping only.
            fact_ids = [t.fact_id for t in seg if t.fact_id]
            # Every source id in the segment (synthetic fact ids or benchmark
            # dialogue ids), so annotated evidence can be traced to records.
            source_ids = list(dict.fromkeys(
                i for t in seg for i in (t.fact_id, t.dia_id) if i
            ))
            candidates.append(
                Candidate(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    text=text,
                    record_type=FACTUAL if cfg.flat_store else route_type(text),
                    timestamp=session_time,
                    speaker=seg[0].speaker,
                    start_turn=seg[0].turn_index,
                    end_turn=seg[-1].turn_index,
                    fact_id=fact_ids[0] if fact_ids else None,
                    source_ids=source_ids,
                    entities=extract_entities(text),
                    has_temporal_anchor=bool(_RELATIVE_TIME.search(raw))
                    or bool(re.search(r"\b(19|20)\d{2}\b", raw)),
                )
            )

    if candidates:
        embs = embedder.encode([c.text for c in candidates])
        for c, e in zip(candidates, embs):
            c.embedding = e
    return candidates


def summarise(text: str, max_tokens: int = 28) -> str:
    """Tier 1 retrieval gist: a truncation, not a generated summary.

    Generating summaries would reintroduce the per-segment generative call the
    framework exists to avoid.
    """
    toks = text.split()
    if len(toks) <= max_tokens:
        return text
    return " ".join(toks[:max_tokens]) + " ..."


def count_tokens(text: str) -> int:
    return len(tokenize(text))
