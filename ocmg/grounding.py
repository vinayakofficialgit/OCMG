"""Grounding and faithfulness checks that need nothing but locomo10.json.

Accuracy says whether an answer matched the gold answer. It does not say why.
A correct answer can come from the right evidence, or from a lucky guess with
no evidence in the prompt. A wrong answer can come from evidence that was
never retrieved, or from evidence that was retrieved and then misread. These
are different failures with different fixes, and LoCoMo can tell them apart
because every answerable question lists the dialogue ids of its supporting
turns (`evidence`), and every record keeps the ids of the turns it was built
from (`source_ids`).

Two levels, both measured per question:

Retrieval level - against the gold evidence annotation (exact, no model):
    evidence_recall    |gold evidence ids in the prompt| / |gold evidence ids|
                       "did the prompt contain what the answer needs?"
    context_precision  records carrying >= 1 gold evidence id / records sent
                       "how much of what we paid for was relevant?"
    evidence_hit       1 if at least one gold evidence id reached the prompt

Answer level - against the context actually shown to the reader:
    answer_support     share of the answer's content words that appear in
                       the prompt (stemmed, LoCoMo normalisation). A lexical
                       proxy for faithfulness: an answer whose words are not
                       in its context did not come from its context.
    grounded           answer_support >= threshold (EvalConfig)
    hallucination      gave an answer (did not abstain) and is not grounded

The four-way split of correctness by grounding is the diagnostic that matters:

    correct + grounded      the system worked
    correct + ungrounded    right for the wrong reason (guess / prior knowledge)
    wrong   + grounded      evidence was there, the reader misused it
    wrong   + ungrounded    evidence missing, or the reader made it up

Limits, stated so nobody over-reads them:
* answer_support is lexical. A temporal answer the reader derives correctly
  ("yesterday" in a session dated 8 May -> "7 May 2023") shares only some
  words with its context, so temporal questions under-report support. A
  model-based faithfulness judge (claims entailed by the context, as in
  RAGAS) needs a generative model and is the Step 4 LLM-judge item.
* Adversarial questions have no evidence by design, so the retrieval-level
  metrics are undefined for them and they are excluded, not scored zero.
* Under the simulated reader a "correct" answer is the gold text itself, so
  answer-level numbers there only test the machinery.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .locomo_eval import _stem, normalize_answer
from .reader import format_session_date, is_abstention, record_source_ids
from .types import Question, Record

# Function words that carry no claim. Without this, "She went to the park"
# would count "she" and "to" as support and nearly every answer would look
# grounded.
_STOP = {
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "we", "our", "they", "them", "their", "is", "was", "are", "were",
    "be", "been", "being", "am", "do", "did", "does", "have", "has", "had",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "or",
    "that", "this", "these", "those", "there", "so", "but", "if", "then",
    "also", "about", "into", "over", "after", "before", "up", "out", "not",
}


def _content_tokens(text: str) -> List[str]:
    return [_stem(w) for w in normalize_answer(text).split() if w not in _STOP]


def context_tokens(context: Sequence[Record]) -> set:
    """Every stemmed token the reader was shown, session dates included."""
    toks: set = set()
    for r in context:
        toks.update(_content_tokens(f"{format_session_date(r.timestamp)} {r.text}"))
    return toks


def answer_support(prediction: str, context: Sequence[Record],
                   ctx_tokens: Optional[set] = None) -> Optional[float]:
    """Share of the answer's content words present in the context.

    None when there is nothing to check: an abstention makes no claim, and
    an answer made only of function words has no content to look for.
    """
    if is_abstention(prediction):
        return None
    words = _content_tokens(prediction)
    if not words:
        return None
    have = ctx_tokens if ctx_tokens is not None else context_tokens(context)
    return sum(1 for w in words if w in have) / len(words)


def retrieval_scores(question: Question, context: Sequence[Record]) -> Optional[dict]:
    """Recall, precision and hit against the gold evidence ids.

    None for a question with no evidence (adversarial), where neither
    quantity is defined.
    """
    # supporting_fact_ids, not evidence_ids. LoCoMo gives adversarial
    # questions an evidence id too - the turn the trap is built on (Melanie's
    # charity race, asked about Caroline) - but nothing supports an answer,
    # so the loader leaves supporting_fact_ids empty for them. Reading
    # evidence_ids here counted 446 unanswerable questions as needing
    # evidence the correct response is to ignore.
    gold = set(question.supporting_fact_ids)
    if not gold:
        return None
    present: set = set()
    relevant = 0
    for r in context:
        ids = record_source_ids(r)
        present |= ids
        relevant += int(bool(ids & gold))
    return {
        "evidence_recall": len(gold & present) / len(gold),
        "context_precision": relevant / len(context) if context else 0.0,
        "evidence_hit": float(bool(gold & present)),
    }


class GroundingTracker:
    """Accumulates the per-question checks for one condition."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.rows: List[dict] = []

    def add(self, question: Question, context: Sequence[Record],
            prediction: str, correct: int) -> dict:
        ret = retrieval_scores(question, context)
        sup = answer_support(prediction, context)
        abstained = is_abstention(prediction)
        grounded = None if sup is None else int(sup >= self.threshold)
        row = {
            "category": question.category,
            "correct": int(correct),
            "abstained": int(abstained),
            "support": sup,
            "grounded": grounded,
            "hallucination": int(not abstained and grounded == 0),
            **(ret or {}),
        }
        self.rows.append(row)
        return row

    @staticmethod
    def _mean(vals) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def summary(self) -> Dict[str, object]:
        rows = self.rows
        if not rows:
            return {}
        answerable = [r for r in rows if "evidence_recall" in r]
        claimed = [r for r in rows if r["grounded"] is not None]
        quad = {"correct_grounded": 0, "correct_ungrounded": 0,
                "wrong_grounded": 0, "wrong_ungrounded": 0}
        for r in claimed:
            key = ("correct" if r["correct"] else "wrong") + (
                "_grounded" if r["grounded"] else "_ungrounded")
            quad[key] += 1
        recall_by_cat: Dict[str, List[float]] = {}
        for r in answerable:
            recall_by_cat.setdefault(r["category"], []).append(r["evidence_recall"])
        return {
            "n_questions": len(rows),
            "n_answerable": len(answerable),
            "n_claims": len(claimed),
            "evidence_recall": self._mean(r["evidence_recall"] for r in answerable),
            "context_precision": self._mean(r["context_precision"] for r in answerable),
            "evidence_hit_rate": self._mean(r["evidence_hit"] for r in answerable),
            "answer_support": self._mean(r["support"] for r in claimed),
            "grounded_rate": self._mean(r["grounded"] for r in claimed),
            "hallucination_rate": float(np.mean([r["hallucination"] for r in rows])),
            "abstention_rate": float(np.mean([r["abstained"] for r in rows])),
            "support_threshold": self.threshold,
            "quadrants": quad,
            "evidence_recall_per_category": {
                c: float(np.mean(v)) for c, v in recall_by_cat.items()},
        }
