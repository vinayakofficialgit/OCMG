"""Core data structures for the OCMG pipeline.

Everything that moves between stages is defined here so that the stages stay
decoupled: ingestion produces Candidates, the store holds Records, retrieval
produces RetrievedSets, and the reader produces QueryLogs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# The three record types of the type-aware store (Section 7.2 of the proposal).
PROFILE = "profile"
EPISODIC = "episodic"
FACTUAL = "factual"
RECORD_TYPES = (PROFILE, EPISODIC, FACTUAL)


@dataclass
class Turn:
    """One speaker-attributed, timestamped utterance."""

    session_id: int
    turn_index: int
    speaker: str  # "user" | "assistant"
    text: str
    timestamp: float
    # Ground-truth fact id, present only in synthetic data. Used to build the
    # supervision signal; never visible to the governance mechanism.
    fact_id: Optional[str] = None
    # Benchmark dialogue id (LoCoMo "D3:7"). Lets annotated evidence be
    # matched back to the records built from this turn.
    dia_id: Optional[str] = None


@dataclass
class Candidate:
    """A segment that has been typed but not yet admitted to the store."""

    conversation_id: str
    session_id: int
    text: str
    record_type: str
    timestamp: float
    speaker: str
    start_turn: int
    end_turn: int
    fact_id: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    has_temporal_anchor: bool = False
    embedding: Optional[np.ndarray] = None
    # Ids of every source turn in the segment: synthetic fact ids or benchmark
    # dialogue ids. Bookkeeping for evaluation only; never a model feature.
    source_ids: List[str] = field(default_factory=list)


@dataclass
class Record:
    """An admitted memory record.

    Tier 1 fields (summary, embedding, entities) are what retrieval scans.
    Tier 2 fields (text, provenance) are expanded only for survivors.
    """

    record_id: int
    conversation_id: str
    record_type: str
    summary: str  # Tier 1
    text: str  # Tier 2
    embedding: np.ndarray
    entities: List[str]
    timestamp: float
    session_id: int
    start_turn: int
    end_turn: int
    speaker: str
    has_temporal_anchor: bool
    fact_id: Optional[str] = None
    active: bool = True
    write_features: Optional[np.ndarray] = None
    merged_count: int = 1
    # True when the admission gate demoted this record to the cold tier at
    # write time, as opposed to it being evicted later. Kept apart so the
    # two kinds of governance decision stay separable in analysis.
    demoted_at_write: bool = False
    source_ids: List[str] = field(default_factory=list)


@dataclass
class LedgerRow:
    """Accumulated outcome evidence for one record (Section 7.4)."""

    record_id: int
    retrieval_count: int = 0
    credit_total: float = 0.0
    weight_total: float = 0.0
    last_retrieved_at: float = 0.0

    @property
    def running_utility(self) -> float:
        """Credit normalised by accumulated weight.

        Normalising by weight rather than by count keeps the estimate stable
        when records are retrieved at different ranks and frequencies.
        """
        if self.weight_total <= 0.0:
            return 0.0
        return self.credit_total / self.weight_total


@dataclass
class Question:
    """A benchmark question with its reference answer."""

    conversation_id: str
    question_id: str
    text: str
    answer: str
    category: str  # single_hop | multi_hop | temporal | open_domain | adversarial
    supporting_fact_ids: List[str] = field(default_factory=list)
    asked_at: float = 0.0
    # Benchmark evidence annotation: dialogue ids of the turns that support
    # the reference answer. Used to validate record-level utility labels.
    evidence_ids: List[str] = field(default_factory=list)
    # LoCoMo category 5 only: the misleading answer the question is built to
    # elicit. The correct response to these questions is to abstain.
    adversarial_answer: Optional[str] = None


@dataclass
class RetrievedSet:
    """The outcome of one retrieval call."""

    question: Question
    record_ids: List[int]
    ranks: List[int]
    scores: List[float]
    prompt_tokens: int
    latency_ms: float


@dataclass
class QueryLog:
    """One instrumented retrieval, with the downstream outcome attached.

    This is the unit of supervision. Persisted across a run, the collection of
    QueryLogs is the reusable dataset described in Section 5.
    """

    conversation_id: str
    question_id: str
    category: str
    record_ids: List[int]
    ranks: List[int]
    correct: int
    f1: float
    prompt_tokens: int
    latency_ms: float
    generative_calls: int = 0


@dataclass
class RunResult:
    """Aggregate metrics for one experimental condition."""

    condition: str
    accuracy: float
    token_f1: float
    prompt_tokens_per_query: float
    retrieval_latency_ms: float
    records_retained: int
    generative_calls: int
    n_questions: int
    per_question_correct: List[int] = field(default_factory=list)
    per_category: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
