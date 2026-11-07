"""Stage 4: budgeted two-stage retrieval.

Dense and sparse paths run in parallel over Tier 1 and are score-fused. The
shortlist is then expanded into Tier 2 for the survivors only, which is the
coarse-to-fine principle: expensive full-record comparison stays confined to a
small candidate set.

The budget k is set per query rather than fixed. Where the predicted-utility
mass over candidates is concentrated, a small k suffices; where it is diffuse,
k rises to a ceiling. This conditions on properties of the candidate records,
not only on the query, which is the extension over query-conditioned budgeting.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RetrievalConfig
from .embedding import cosine_matrix, tokenize
from .ingest import count_tokens
from .store import MemoryStore
from .types import Question, Record, RetrievedSet


class BM25Index:
    """Sparse path. Kept alongside dense retrieval because exact matching is
    necessary for named entities that dense similarity dilutes."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: Dict[int, List[str]] = {}
        self.df: Dict[str, int] = {}
        self.avg_len = 1.0

    def rebuild(self, records: Sequence[Record]) -> None:
        self.docs = {r.record_id: tokenize(r.text) for r in records}
        self.df = {}
        for toks in self.docs.values():
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        lengths = [len(t) for t in self.docs.values()]
        self.avg_len = float(np.mean(lengths)) if lengths else 1.0

    def score(self, query: str) -> Dict[int, float]:
        q = tokenize(query)
        n = max(1, len(self.docs))
        out: Dict[int, float] = {}
        for rid, toks in self.docs.items():
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                s += idf * (f * (self.k1 + 1)) / denom
            if s > 0:
                out[rid] = s
        return out


def _minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


# Derived from the question text, never from the benchmark's own category
# label. LoCoMo labels every question multi-hop / temporal / single-hop /
# open-domain / adversarial, and reading that at retrieval time would be using
# a gold annotation the deployed system would not have - the same leakage the
# evidence annotations are held out to avoid.
_TEMPORAL_CUES = (
    "when ", "what year", "what month", "what day", "how long", "how many days",
    "how many years", " before ", " after ", "first time", "last time",
    " earlier", " later", " since ", " ago", "how often",
)
_ENUMERATIVE_CUES = (
    "how many", "list ", "what are", "which of", "all of the", "both ",
    " and what", "name the",
)


def query_type(text: str) -> str:
    """Cheap surface classification of what the question needs.

    Three types, because they differ in how much evidence an answer needs
    rather than in subject matter: a temporal question has to compare two
    points in time, an enumerative one has to collect several records, and
    everything else usually turns on one. Keyword cues rather than a model,
    so the budget rule stays interpretable and costs nothing.
    """
    t = f" {text.lower().strip()} "
    if any(c in t for c in _ENUMERATIVE_CUES):
        return "enumerative"
    if any(c in t for c in _TEMPORAL_CUES):
        return "temporal"
    return "simple"


def query_floor(text: str, cfg: RetrievalConfig, mass_target: float) -> float:
    """The relative-quality floor for this question, by derived type."""
    if not cfg.budget_by_query_type:
        return mass_target
    return mass_target * cfg.query_type_floor_scale.get(query_type(text), 1.0)


def adaptive_k(
    scores: Sequence[float], cfg: RetrievalConfig, mass_target: float
) -> int:
    """Set the retrieval depth from the shape of the ranked score distribution.

    A cumulative-mass rule on min-max normalised scores does not work: the
    normalisation forces the top score to 1 and the bottom to 0 on every query,
    so the distribution has the same shape whatever the evidence looks like and
    k pins to its ceiling. The rule below uses the raw ranked scores and keeps
    records that stay within a fixed fraction of the best one, so a query whose
    answer sits in a single record takes two or three records, and a diffuse
    one takes as many as the ceiling allows.
    """
    if not scores:
        return cfg.k_min
    u = np.asarray(sorted(scores, reverse=True), dtype=np.float64)
    top = float(u[0])
    if top <= 0:
        return cfg.k_min
    # mass_target doubles as the relative-quality floor: 0.55 keeps everything
    # scoring at least 55% of the best candidate.
    keep = int(np.sum(u >= mass_target * top))
    return int(np.clip(keep, cfg.k_min, cfg.k_max))


class Retriever:
    def __init__(self, store: MemoryStore, embedder, cfg: RetrievalConfig):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.bm25 = BM25Index()
        self._bm25_size = None

    def _ensure_sparse(self, conv: Optional[str]) -> None:
        active = self.store.active_records(conv)
        cold = (
            self.store.cold_records(conv)
            if self.cfg.cold_tier_penalty > 0.0 else []
        )
        key = (conv, len(active), len(cold))
        if key != self._bm25_size:
            # The sparse path must span both tiers, or a demoted record could
            # never be recovered by an exact entity match - which is the case
            # it is most likely to be needed for.
            self.bm25.rebuild(active + cold)
            self._bm25_size = key

    def retrieve(
        self,
        question: Question,
        k: Optional[int] = None,
        utility_fn=None,
        mass_target: float = 0.80,
        rng: Optional[np.random.Generator] = None,
        epsilon_random: float = 0.0,
    ) -> RetrievedSet:
        t0 = time.perf_counter()
        conv = question.conversation_id
        self._ensure_sparse(conv)

        mat, ids = self.store.matrix(conv)
        cold_ids: List[int] = []
        if self.cfg.cold_tier_penalty > 0.0:
            cmat, cold_ids = self.store.matrix_cold(conv)
            if cold_ids:
                mat = np.vstack([mat, cmat]) if len(ids) else cmat
                ids = list(ids) + list(cold_ids)
        if not ids:
            return RetrievedSet(question, [], [], [], 0, 0.0)
        cold = set(cold_ids)

        q_emb = self.embedder.encode_one(question.text)
        dense = {rid: float(s) for rid, s in zip(ids, cosine_matrix(q_emb, mat))}
        sparse = self.bm25.score(question.text)

        dn, sn = _minmax(dense), _minmax(sparse)
        # Cold records compete on the same ranking, discounted rather than
        # excluded: a demoted record enters the prompt only when its relevance
        # beats the hot tier despite the penalty. Because k is unchanged, this
        # changes what is retrieved and not how much.
        fused = {
            rid: (
                self.cfg.dense_weight * dn.get(rid, 0.0)
                + self.cfg.sparse_weight * sn.get(rid, 0.0)
            ) * (self.cfg.cold_tier_penalty if rid in cold else 1.0)
            for rid in ids
        }

        # Tier 1 shortlist -> Tier 2 expansion for survivors only.
        target_k = k if k is not None else self.cfg.k_fixed
        shortlist_n = min(len(ids), max(target_k * self.cfg.tier1_multiplier, target_k))
        shortlist = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:shortlist_n]

        if utility_fn is not None and shortlist:
            # The question goes with the ids: a conditional estimator needs
            # it to build its features, and a pooled one ignores it.
            priors = utility_fn([rid for rid, _ in shortlist], question)
            # Rank by fused relevance modulated by predicted utility. Utility
            # reorders within the relevant set; it does not replace relevance.
            # A record the estimator likes but the query does not match must
            # not outrank a record the query does match.
            #
            # The multiplier is (1-w) + 2*w*prior, so w=0 ignores utility
            # entirely, w=0.5 reproduces the original 0.5+prior, and w=1 lets
            # utility scale a score by up to 2x. Expressing it as one weight
            # makes the strength of the utility signal a measurable choice
            # rather than a constant nobody set deliberately.
            w = self.cfg.utility_weight
            blended = [
                (rid, score * ((1.0 - w) + 2.0 * w * prior))
                for (rid, score), prior in zip(shortlist, priors)
            ]
            blended.sort(key=lambda x: x[1], reverse=True)
            if k is None:
                # Budget from the blended score distribution, not the raw
                # utility priors: it is the ranked distribution actually being
                # consumed, and it is peaked when the answer is localised.
                # Stage 4 of the method sets k from the query type as well as
                # the utility distribution: a question needing several records
                # gets a lower bar for admitting the next one.
                floor = query_floor(question.text, self.cfg, mass_target)
                target_k = adaptive_k([s for _, s in blended], self.cfg, floor)
            shortlist = blended

        chosen = [rid for rid, _ in shortlist[:target_k]]
        scores = [s for _, s in shortlist[:target_k]]

        # Exploration only: force-include records the policy would not surface,
        # so that Phase A labels are not confined to the policy's own choices.
        if epsilon_random > 0.0 and rng is not None:
            pool = [r for r in ids if r not in chosen]
            n_extra = int(rng.binomial(len(chosen), epsilon_random)) if chosen else 0
            if pool and n_extra > 0:
                extra = list(rng.choice(pool, size=min(n_extra, len(pool)), replace=False))
                chosen.extend(int(e) for e in extra)
                scores.extend([0.0] * len(extra))

        prompt_tokens = sum(
            count_tokens(self.store.records[r].text) for r in chosen
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return RetrievedSet(
            question=question,
            record_ids=chosen,
            ranks=list(range(1, len(chosen) + 1)),
            scores=scores,
            prompt_tokens=prompt_tokens,
            latency_ms=latency,
        )
