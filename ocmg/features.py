"""Feature extraction for the utility estimator (Section 7.3).

Thirty features across four families. Every one is computable at write time
from the candidate and the current store, and none requires a generative call —
that is the constraint the whole cost argument rests on.

The historical family is zero-filled for a candidate that has never been in the
store, which is the correct value: a record with no retrieval history has no
accumulated evidence, and the estimator learns what that absence implies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .embedding import cosine_matrix, tokenize
from .types import EPISODIC, FACTUAL, PROFILE, Candidate, LedgerRow, Record

FEATURE_NAMES: List[str] = [
    # --- structural (8) ---
    "type_profile", "type_episodic", "type_factual", "token_length",
    "n_entities", "has_temporal_anchor", "n_sentences", "digit_ratio",
    # --- distinctiveness (8) ---
    "max_similarity", "mean_top5_similarity", "novelty", "mean_idf",
    "max_idf", "unique_token_ratio", "rare_term_count", "near_duplicate",
    # --- provenance (7) ---
    "session_index_norm", "position_in_session", "is_user", "is_assistant",
    "turn_span", "modifies_existing", "contradicts_existing",
    # --- historical (7) ---
    "retrieval_count", "log_retrieval_count", "recency_gap",
    "running_utility", "credit_total", "weight_total", "ever_retrieved",
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 30, N_FEATURES

# Appended to the record features when the estimator is query-conditional.
# One-hot rather than an ordinal code: the three types are not ordered, and a
# tree given an ordinal would have to spend two splits to isolate the middle
# one. The order here is the contract between training and inference, so it is
# defined once and read from both.
QUERY_TYPE_NAMES: List[str] = ["q_simple", "q_temporal", "q_enumerative"]
_QUERY_TYPE_INDEX = {"simple": 0, "temporal": 1, "enumerative": 2}


def query_type_onehot(qtype: str) -> np.ndarray:
    """One-hot for a derived query type; all-zero for an unknown one."""
    v = np.zeros(len(QUERY_TYPE_NAMES), dtype=np.float32)
    i = _QUERY_TYPE_INDEX.get(qtype)
    if i is not None:
        v[i] = 1.0
    return v

_NEGATION = {"not", "no", "never", "isn't", "wasn't", "aren't", "doesn't",
             "didn't", "won't", "cannot", "can't", "stopped", "quit", "former"}


class IdfIndex:
    """Document frequencies over admitted records, used for term specificity."""

    def __init__(self) -> None:
        self.df: Dict[str, int] = {}
        self.n_docs: int = 0

    def add(self, text: str) -> None:
        self.n_docs += 1
        for tok in set(tokenize(text)):
            self.df[tok] = self.df.get(tok, 0) + 1

    def idf(self, token: str) -> float:
        # Smoothed idf; an unseen term gets the maximum value.
        return math.log((1 + self.n_docs) / (1 + self.df.get(token, 0))) + 1.0

    def profile(self, text: str) -> tuple[float, float, int]:
        toks = tokenize(text)
        if not toks:
            return 0.0, 0.0, 0
        vals = [self.idf(t) for t in toks]
        rare = sum(1 for v in vals if v > math.log(1 + self.n_docs) * 0.75)
        return float(np.mean(vals)), float(max(vals)), rare


def _structural(cand: Candidate) -> List[float]:
    text = cand.text
    toks = tokenize(text)
    n_tok = len(toks)
    digits = sum(1 for ch in text if ch.isdigit())
    return [
        1.0 if cand.record_type == PROFILE else 0.0,
        1.0 if cand.record_type == EPISODIC else 0.0,
        1.0 if cand.record_type == FACTUAL else 0.0,
        float(n_tok),
        float(len(cand.entities)),
        1.0 if cand.has_temporal_anchor else 0.0,
        float(text.count(".") + text.count("?") + text.count("!")),
        (digits / len(text)) if text else 0.0,
    ]


def _distinctiveness(
    cand: Candidate,
    store_matrix: Optional[np.ndarray],
    idf: IdfIndex,
    dedup_threshold: float,
) -> List[float]:
    if store_matrix is None or store_matrix.size == 0 or cand.embedding is None:
        max_sim, top5 = 0.0, 0.0
    else:
        sims = cosine_matrix(cand.embedding, store_matrix)
        max_sim = float(np.max(sims))
        k = min(5, sims.shape[0])
        top5 = float(np.mean(np.sort(sims)[-k:]))
    mean_idf, max_idf, rare = idf.profile(cand.text)
    toks = tokenize(cand.text)
    uniq = len(set(toks)) / len(toks) if toks else 0.0
    return [
        max_sim,
        top5,
        1.0 - max_sim,
        mean_idf,
        max_idf,
        uniq,
        float(rare),
        1.0 if max_sim >= dedup_threshold else 0.0,
    ]


def _provenance(
    cand: Candidate,
    n_sessions: int,
    session_position: float,
    neighbour_texts: Sequence[str],
) -> List[float]:
    """Provenance includes two cheap relational cues.

    'modifies_existing' fires when a candidate shares most of its entities with
    a near neighbour, and 'contradicts_existing' when it does so while carrying
    a negation the neighbour lacks. This is a lexical proxy for the graph-based
    conflict check used elsewhere in the literature, at no generative cost.
    """
    ent = set(cand.entities)
    toks = set(tokenize(cand.text))
    modifies = 0.0
    contradicts = 0.0
    for nt in neighbour_texts:
        n_ent = set(w for w in nt.split() if w[:1].isupper())
        if ent and len(ent & n_ent) / max(1, len(ent)) >= 0.5:
            modifies = 1.0
            n_toks = set(tokenize(nt))
            if (toks & _NEGATION) and not (n_toks & _NEGATION):
                contradicts = 1.0
    return [
        cand.session_id / max(1, n_sessions),
        session_position,
        1.0 if cand.speaker == "user" else 0.0,
        1.0 if cand.speaker == "assistant" else 0.0,
        float(cand.end_turn - cand.start_turn + 1),
        modifies,
        contradicts,
    ]


def _historical(row: Optional[LedgerRow], now: float) -> List[float]:
    if row is None or row.retrieval_count == 0:
        return [0.0] * 7
    gap = max(0.0, now - row.last_retrieved_at) / 86400.0  # days
    return [
        float(row.retrieval_count),
        math.log1p(row.retrieval_count),
        gap,
        row.running_utility,
        row.credit_total,
        row.weight_total,
        1.0,
    ]


def extract_features(
    cand: Candidate,
    store_matrix: Optional[np.ndarray],
    idf: IdfIndex,
    n_sessions: int,
    session_position: float,
    neighbour_texts: Sequence[str] = (),
    ledger_row: Optional[LedgerRow] = None,
    now: float = 0.0,
    dedup_threshold: float = 0.92,
) -> np.ndarray:
    vec = (
        _structural(cand)
        + _distinctiveness(cand, store_matrix, idf, dedup_threshold)
        + _provenance(cand, n_sessions, session_position, neighbour_texts)
        + _historical(ledger_row, now)
    )
    assert len(vec) == N_FEATURES
    return np.asarray(vec, dtype=np.float32)


def record_to_candidate(rec: Record) -> Candidate:
    """Re-derive a Candidate view of a stored record, for re-scoring."""
    return Candidate(
        conversation_id=rec.conversation_id,
        session_id=rec.session_id,
        text=rec.text,
        record_type=rec.record_type,
        timestamp=rec.timestamp,
        speaker=rec.speaker,
        start_turn=rec.start_turn,
        end_turn=rec.end_turn,
        fact_id=rec.fact_id,
        source_ids=list(rec.source_ids),
        entities=rec.entities,
        has_temporal_anchor=rec.has_temporal_anchor,
        embedding=rec.embedding,
    )
