"""Stage 2 and 5: the three governance decisions (Section 7.5).

Two calibration details here were arrived at empirically and are the parts most
worth defending in a viva, because the obvious implementation of each is wrong.

Quantile thresholds, not absolute ones. Calibrated utility probabilities on this
task cluster tightly around the dataset base rate — most candidates look
similarly useful ex ante. An absolute cut such as U >= 0.5 therefore admits
nearly everything or nearly nothing depending on which side of the base rate it
falls, and the operating point swings wildly between runs. Setting the threshold
at a quantile of the observed score distribution fixes the admitted fraction
directly, which is both more stable and easier to interpret.

Calibration on the contested population. Near-duplicates are merged before the
gate is applied, so they never reach it. If the threshold is computed over the
raw candidate stream it is fitted to a population that includes records the gate
will never actually judge, and the effective admission rate drifts. The
threshold must be computed over the post-deduplication candidates only — the
ones the gate genuinely contests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import GovernanceConfig
from .embedding import tokenize
from .store import MemoryStore
from .types import Candidate


def select_lowest_fraction(
    scored: Sequence[Tuple[int, float]], fraction: float
) -> List[int]:
    """The lowest-scoring `fraction` of records, selected by rank.

    Selecting by value — evict everything at or below the fraction's quantile —
    looks equivalent and is not, because it cannot split a tie. Realised
    utility has an enormous spike at exactly 0.0, since every record that has
    never been retrieved alongside a correct answer scores precisely zero. The
    quantile for any small fraction therefore lands *on* that spike, and the
    comparison takes the whole group with it: a requested 25% evicts 60%, and
    requesting 0% evicts exactly as much as requesting 25%. The configured
    fraction stops controlling anything.

    Ranking makes the realised fraction match the configured one. Ties break on
    record id, so the choice within a tie group is arbitrary but deterministic
    and reproducible across runs.
    """
    if fraction <= 0.0 or not scored:
        return []
    n_evict = int(len(scored) * fraction)
    if n_evict <= 0:
        return []
    ordered = sorted(scored, key=lambda t: (t[1], t[0]))
    return [rid for rid, _ in ordered[:n_evict]]


@dataclass
class AdmissionThresholds:
    write: float = 0.0
    forget: float = 0.0
    n_contested: int = 0
    fitted: bool = False


class GovernancePolicy:
    def __init__(self, cfg: GovernanceConfig, estimator=None):
        self.cfg = cfg
        self.estimator = estimator
        self.thresholds = AdmissionThresholds()
        self.stats: Dict[str, int] = {
            "admitted": 0, "discarded": 0, "merged": 0, "evicted": 0
        }

    # --------------------------------------------------------- threshold fit

    def calibrate(self, contested_scores: Sequence[float]) -> AdmissionThresholds:
        """Set thresholds as quantiles of the contested score distribution."""
        scores = np.asarray([s for s in contested_scores], dtype=float)
        if scores.size < 20:
            # Too few contested candidates to locate a stable quantile; admit
            # everything rather than cut on noise.
            self.thresholds = AdmissionThresholds(
                write=-np.inf, forget=-np.inf, n_contested=int(scores.size), fitted=False
            )
            return self.thresholds
        self.thresholds = AdmissionThresholds(
            write=float(np.quantile(scores, self.cfg.write_quantile)),
            forget=float(np.quantile(scores, self.cfg.forget_quantile)),
            n_contested=int(scores.size),
            fitted=True,
        )
        return self.thresholds

    # ------------------------------------------------------- write admission

    def check_duplicate(
        self, cand: Candidate, store: MemoryStore
    ) -> Optional[int]:
        """Return the id of a near-duplicate to merge into, if one exists."""
        if cand.embedding is None:
            return None
        for rid, sim in store.nearest(cand.embedding, k=3, conv=cand.conversation_id):
            if sim < self.cfg.dedup_similarity:
                break
            # Embedding similarity alone is not identity. Two records that
            # differ only in the value of a named entity ("my coffee is a flat
            # white" / "a cortado") sit very close in embedding space while
            # asserting different things, and merging them silently destroys
            # the fact. Require the entity sets to agree as well.
            other = store.records[rid]
            a, b = set(cand.entities), set(other.entities)
            if (a or b) and len(a & b) / max(1, len(a | b)) < 0.6:
                continue
            # Embedding proximity is necessary but not sufficient. A compact
            # encoder maps short sentences of similar shape close together
            # whatever they assert, so without a lexical confirmation the gate
            # merges unrelated chatter wholesale and takes real facts with it.
            ta, tb = set(tokenize(cand.text)), set(tokenize(other.text))
            if len(ta | tb) and len(ta & tb) / len(ta | tb) < 0.75:
                continue
            return rid
        return None

    def admit(self, utility: float) -> bool:
        if not self.thresholds.fitted:
            return True
        return utility >= self.thresholds.write

    # ------------------------------------------------------------ forgetting

    def select_for_eviction(self, store: MemoryStore, conv: Optional[str] = None) -> List[int]:
        """Evict records with enough evidence and too little demonstrated use.

        The minimum-retrieval guard is what separates this from a decay
        schedule: a record is only removed once it has had a fair chance to
        prove itself, and it is removed on demonstrated uselessness rather than
        on disuse.
        """
        if self.cfg.forget_quantile <= 0.0:
            # Forgetting off. A quantile of zero does not express this: the cut
            # would land on the lowest utility value and the tie-group rule
            # below would still evict every record sharing it.
            return []

        active = store.active_records(conv)
        # "Now" is the latest event the store has seen, derived the same way
        # the decay baseline derives it, so that the recency signal below is
        # the baseline's criterion and not an approximation of it.
        now = max(
            [r.timestamp for r in active]
            + [store.ledger[r.record_id].last_retrieved_at for r in active],
            default=0.0,
        )

        candidates: List[Tuple[int, float]] = []
        for rec in store.active_records(conv):
            row = store.ledger[rec.record_id]
            if row.retrieval_count < self.cfg.min_retrievals_before_forget:
                continue
            candidates.append((rec.record_id, self.forget_score(rec, row, now)))

        if len(candidates) < 10:
            return []

        return select_lowest_fraction(candidates, self.cfg.forget_quantile)

    def forget_score(self, rec, row, now: float = 0.0) -> float:
        """Rank a record for eviction under the configured signal.

        The eligibility guard is deliberately left identical across signals, so
        a comparison between them varies the ranking and nothing else.
        """
        signal = self.cfg.forget_signal
        if signal == "recency":
            # The decay baseline's criterion, evaluated inside this pipeline.
            #
            # DecayPolicy exists as a rival *system*: it runs on an ungoverned
            # store and keeps no eligibility guard, so sweeping both at a
            # matched forget_quantile matches the fraction and not the number
            # of records removed - measured at 103 against 587 evictions at
            # 0.10. Those two runs cannot separate a weak criterion from a
            # differently composed candidate pool. This signal can, because
            # admission, budget, guard and fraction are all held identical and
            # only the ranking changes.
            gap_days = max(0.0, now - row.last_retrieved_at) / 86400.0
            recency = 1.0 / (1.0 + gap_days)
            return float(np.log1p(row.retrieval_count) * 0.6 + recency * 0.4)

        usable = (
            signal != "ledger"
            and self.estimator is not None
            and getattr(self.estimator, "is_fitted", False)
            and rec.write_features is not None
        )
        if not usable:
            return row.running_utility

        # Forgetting also acts on a record with no question in hand, so a
        # conditional estimator is marginalised the same way the write gate
        # marginalises it. getattr keeps a bare estimator double - which the
        # tests pass - working without a predict_marginal of its own.
        scorer = getattr(self.estimator, "predict_marginal",
                         self.estimator.predict)
        prior = float(scorer(rec.write_features)[0])
        if signal == "estimator":
            return prior
        if signal == "blend":
            # Shrink the realised mean toward the prior by observation count:
            # with one retrieval the prior dominates, with many the evidence
            # does. This is what stops a record being evicted for an absence
            # of evidence rather than for evidence of uselessness.
            n = float(row.retrieval_count)
            m = float(self.cfg.forget_prior_weight)
            return (n * row.running_utility + m * prior) / (n + m)
        raise ValueError(f"unknown forget_signal: {signal}")

    def run_forgetting(
        self, store: MemoryStore, now: float, conv: Optional[str] = None
    ) -> int:
        victims = self.select_for_eviction(store, conv)
        for rid in victims:
            store.evict(rid, now, reason="low_utility")
        self.stats["evicted"] += len(victims)
        return len(victims)
