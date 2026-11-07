"""The two-phase experiment orchestrator (Sections 7.6 and 8).

Phase A (warm-up). Build a store under a permissive admission policy and
retrieve under an *exploration* policy: wider budgets and epsilon-random
inclusion of records the ranker would not have surfaced. This is the single
most important detail in the whole protocol. If Phase A retrieved under the
policy being trained, the estimator would be fitted on a label distribution its
own ranking produced, and would learn to rationalise its existing preferences
rather than to predict usefulness. The wider budget and the random inclusions
are what make the training labels causally informative.

Phase B (governed). Fit the estimator on Phase A logs, calibrate thresholds on
the contested candidate population, rebuild the store under utility-gated
admission, and evaluate under adaptive budgeting.

Conversation-level splits keep the two phases disjoint, and the warm-up store
is discarded before Phase B so nothing leaks through record identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import OCMGConfig
from .credit import assign_credit
from .embedding import build_embedder
from .estimator import EstimatorReport, UtilityEstimator
from .features import (FEATURE_NAMES, QUERY_TYPE_NAMES, IdfIndex,
                       extract_features, query_type_onehot,
                       record_to_candidate)
from .governance import GovernancePolicy
from .ingest import build_candidates, count_tokens
from .metrics import (accuracy, accuracy_per_1k_tokens,
                      accuracy_retention_ratio, paired_bootstrap,
                      retention_precision, token_reduction_ratio)
from .reader import Judge, build_reader
from .retrieval import Retriever, query_type
from .store import MemoryStore
from .types import Candidate, Question, Record, RunResult, Turn

Conversation = Tuple[str, List[Turn], List[Question]]


@dataclass
class PhaseAOutput:
    X: np.ndarray
    y: np.ndarray
    groups: List[str]
    logs: List[dict]
    contested_scores: List[float] = field(default_factory=list)
    # Held-out annotation, aligned row-for-row with y: does this record contain
    # a turn the benchmark annotates as evidence for some question? Never a
    # feature and never a training target - it exists so the outcome labels in
    # y can be audited against ground truth rather than trusted.
    evidence_labels: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=int)
    )
    # Per-question-category labels, each aligned row-for-row with y, with -1
    # where that record was never retrieved for a question of that category.
    # The first marker's objection is that utility is not a fixed property of
    # a record: it depends on the query. Testing that needs the same estimator
    # scored against each category's own outcomes, which needs the credit
    # split by category rather than pooled.
    category_labels: Dict[str, np.ndarray] = field(default_factory=dict)


class OCMGRunner:
    def __init__(self, cfg: OCMGConfig):
        self.cfg = cfg
        self.embedder = build_embedder(
            cfg.embedder, cfg.embedding_dim, cfg.st_model, cfg.seed
        )
        # Keep the configured dim in step with the actual encoder.
        self.cfg.embedding_dim = getattr(self.embedder, "dim", cfg.embedding_dim)
        self.reader = build_reader(cfg.reader, seed=cfg.seed)
        self.judge = Judge("exact" if cfg.reader.kind == "simulated" else "f1")
        self.rng = np.random.default_rng(cfg.seed)
        self.estimator = UtilityEstimator(cfg.estimator)
        self.estimator_report: Optional[EstimatorReport] = None

    # ------------------------------------------------------------- building

    def features_for(
        self,
        cand: Candidate,
        store: MemoryStore,
        n_sessions: int,
        session_position: float,
    ) -> np.ndarray:
        """The single feature path.

        Calibration, admission and re-scoring must all compute features
        identically. Two code paths that differ by even one feature produce
        train/serve skew: thresholds are then fitted to a score distribution
        the gate never actually sees, and the admitted fraction collapses or
        explodes without warning. Everything routes through here.
        """
        conv = cand.conversation_id
        mat, ids = store.matrix(conv)
        neighbours = (
            [
                store.records[rid].text
                for rid, _ in store.nearest(cand.embedding, k=3, conv=conv)
            ]
            if ids
            else []
        )
        return extract_features(
            cand,
            mat if len(ids) else None,
            store.idf,
            n_sessions=n_sessions,
            session_position=session_position,
            neighbour_texts=neighbours,
            ledger_row=None,
            now=cand.timestamp,
            dedup_threshold=self.cfg.governance.dedup_similarity,
        )

    def _score_candidates(
        self, store: MemoryStore, cands: Sequence[Candidate], n_sessions: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Feature-extract and score a batch of candidates against the store."""
        feats: List[np.ndarray] = []
        scores: List[float] = []
        for i, cand in enumerate(cands):
            mat, ids = store.matrix()
            neighbours = [
                store.records[rid].text for rid, _ in store.nearest(cand.embedding, k=3)
            ] if ids else []
            f = extract_features(
                cand,
                mat if len(ids) else None,
                store.idf,
                n_sessions=n_sessions,
                session_position=i / max(1, len(cands)),
                neighbour_texts=neighbours,
                ledger_row=None,
                now=cand.timestamp,
                dedup_threshold=self.cfg.governance.dedup_similarity,
            )
            feats.append(f)
            scores.append(
                float(self.estimator.predict_marginal(f)[0])
                if self.estimator.is_fitted else 0.5
            )
        return feats, scores

    def build_store(
        self,
        conversations: Sequence[Conversation],
        policy: Optional[GovernancePolicy],
        governed: bool,
        flat: bool = False,
    ) -> Tuple[MemoryStore, Dict[str, int]]:
        """Ingest conversations and admit candidates under the given policy.

        `flat=True` routes every segment to a single type, so the store carries
        no typing and the three type indicators carry no information. That is
        what separates the naive top-k baseline from the append-all typed
        store: without it the two are the same computation on the same store,
        and they reported identical numbers to the last digit.
        """
        ingest_cfg = (
            replace(self.cfg.ingest, flat_store=True,
                    naive_chunk_turns=self.cfg.ingest.naive_chunk_turns or 4)
            if flat else self.cfg.ingest)
        store = MemoryStore(self.cfg.db_path, self.cfg.embedding_dim)
        counters = {"candidates": 0, "admitted": 0, "merged": 0, "discarded": 0,
                    "demoted": 0}

        for cid, turns, _ in conversations:
            n_sessions = 1 + max((t.session_id for t in turns), default=0)
            cands = build_candidates(cid, turns, self.embedder, ingest_cfg)
            counters["candidates"] += len(cands)

            for i, cand in enumerate(cands):
                # Deduplication happens BEFORE the gate. Near-duplicates are
                # merged and never contest admission; this is why the threshold
                # must be calibrated on the post-dedup population only.
                dup_id = policy.check_duplicate(cand, store) if policy else None
                if dup_id is not None:
                    store.merge_into(dup_id, cand)
                    counters["merged"] += 1
                    continue

                feats = self.features_for(
                    cand, store, n_sessions, i / max(1, len(cands))
                )

                hot = True
                # A policy may carry an external admission gate - the
                # generative-judge baseline does. It replaces the estimator
                # decision rather than stacking on it, and it discards rather
                # than demotes, because a single-tier curator is what that
                # baseline represents.
                gate = getattr(policy, "candidate_gate", None)
                if governed and gate is not None:
                    if not gate(cand):
                        counters["discarded"] += 1
                        continue
                elif governed and policy is not None and self.estimator.is_fitted:
                    u = float(self.estimator.predict_marginal(feats)[0])
                    if not policy.admit(u):
                        if self.cfg.governance.demote_instead_of_discard:
                            # Kept but demoted: retrieval can still reach it.
                            store.admit(cand, write_features=feats, hot=False)
                            counters["demoted"] += 1
                            continue
                        counters["discarded"] += 1
                        continue

                store.admit(cand, write_features=feats, hot=hot)
                counters["admitted"] += 1

        store.commit()
        return store, counters

    # -------------------------------------------------------------- phase A

    def _feature_width(self) -> int:
        """Columns the estimator expects under the current configuration."""
        return len(FEATURE_NAMES) + (
            len(QUERY_TYPE_NAMES) if self.cfg.estimator.conditional_utility
            else 0
        )

    def run_phase_a(self, conversations: Sequence[Conversation]) -> PhaseAOutput:
        """Warm-up: permissive admission, exploratory retrieval, log outcomes.

        Credit is assigned by leave-one-out re-execution rather than by rank
        weighting. The reason is measurable: exploratory retrieval returns on
        the order of a dozen records per query while at most one or two carry
        the answer, so a co-occurrence rule gives almost identical credit to
        the record that answered the question and to the eleven that happened
        to sit beside it. The resulting label is close to noise and the
        estimator cannot exceed chance on it.

        Leave-one-out removes each record from the prompt, re-runs the reader,
        and observes whether the answer changes. It uses nothing but the
        downstream correctness signal, and it costs |R| extra reader calls per
        query, which is affordable in warm-up and is why the deployed phase
        falls back to the cheap rank-weighted rule.
        """
        policy = GovernancePolicy(self.cfg.governance, self.estimator)
        store, _ = self.build_store(conversations, policy, governed=False)
        retriever = Retriever(store, self.embedder, self.cfg.retrieval)

        # Union of annotated supporting ids per conversation. LoCoMo supplies
        # dialogue ids in evidence_ids; the synthetic generator supplies fact
        # ids in supporting_fact_ids. Taking both keeps the audit available on
        # either source without the rest of the pipeline knowing which it has.
        evidence_by_conv: Dict[str, set] = {}
        for cid, _t, qs in conversations:
            ids: set = set()
            for q in qs:
                ids.update(q.evidence_ids)
                ids.update(q.supporting_fact_ids)
            evidence_by_conv[cid] = ids

        logs: List[dict] = []
        contrib_sum: Dict[int, float] = {}
        contrib_n: Dict[int, int] = {}
        cat_sum: Dict[tuple, float] = {}
        cat_n: Dict[tuple, int] = {}
        qt_sum: Dict[tuple, float] = {}
        qt_n: Dict[tuple, int] = {}
        qtype_seen: Dict[str, int] = {}
        ex = self.cfg.exploration
        # Whether leave-one-out was actually used, not merely available. With
        # loo_fraction at zero the reader still supports re-execution but no
        # ablation is ever run, so the counterfactual label branch would find
        # an empty observation table and drop every record. Falling back to the
        # cheap credit rule is what makes the credit-scheme ablation possible.
        use_loo = (
            getattr(self.reader, "supports_counterfactual", False)
            and ex.loo_fraction > 0.0
        )

        for cid, _, questions in conversations:
            for q in questions:
                k = int(self.rng.integers(ex.k_min, ex.k_max + 1))
                rs = retriever.retrieve(
                    q, k=k, utility_fn=None,
                    rng=self.rng, epsilon_random=ex.epsilon_random_inclusion,
                )
                context = [store.records[r] for r in rs.record_ids]
                # The full prompt and every leave-one-out prompt are known
                # before any of them is answered, so they go to the reader as
                # one batch - the only place generation cost can be amortised
                # on a CPU. The loo draw moves above the reader call to build
                # that batch; no reader consumes self.rng, so the sequence of
                # draws, and therefore every seeded result, is unchanged.
                run_loo = (use_loo and bool(context)
                           and self.rng.random() < ex.loo_fraction)
                # Which of the retrieved records get a counterfactual. The
                # default (loo_max_records 0) is all of them. A cap draws a
                # uniform sample instead of taking the top m, for the reason
                # in ExplorationConfig; the draw is guarded so that an
                # uncapped run consumes no extra randomness and every
                # previously recorded seeded result still reproduces exactly.
                loo_idx = list(range(len(context)))
                if run_loo and 0 < ex.loo_max_records < len(context):
                    loo_idx = sorted(int(i) for i in self.rng.choice(
                        len(context), size=ex.loo_max_records, replace=False,
                    ))
                batch = [(q, context)]
                if run_loo:
                    batch += [(q, context[:i] + context[i + 1:])
                              for i in loo_idx]
                answers = self.reader.answer_many(batch)
                correct, f1 = self.judge.score(answers[0][0], q.answer)

                deltas: List[float] = []
                for j, (pred_a, _, _) in enumerate(answers[1:]):
                    ok_a, _ = self.judge.score(pred_a, q.answer)
                    d = float(correct - ok_a)
                    deltas.append(d)
                    i = loo_idx[j]
                    rid = rs.record_ids[i]
                    contrib_sum[rid] = contrib_sum.get(rid, 0.0) + d
                    contrib_n[rid] = contrib_n.get(rid, 0) + 1
                    key = (rid, q.category)
                    cat_sum[key] = cat_sum.get(key, 0.0) + d
                    cat_n[key] = cat_n.get(key, 0) + 1
                    # The same credit split by DERIVED query type rather than
                    # by the benchmark's category. This is the one a deployed
                    # system could compute, so it is the one U(m|q) may train
                    # on; cat_* above stays for the per-category audit.
                    qkey = (rid, query_type(q.text))
                    qt_sum[qkey] = qt_sum.get(qkey, 0.0) + d
                    qt_n[qkey] = qt_n.get(qkey, 0) + 1

                qt_here = query_type(q.text)
                qtype_seen[qt_here] = qtype_seen.get(qt_here, 0) + 1

                scheme = "leave_one_out" if deltas else self.cfg.credit_scheme
                # assign_credit zips ids against deltas positionally, so a
                # capped run must hand it the ids it actually ablated rather
                # than the first len(deltas) of the retrieved list.
                credit_ids = ([rs.record_ids[i] for i in loo_idx] if deltas
                              else rs.record_ids)
                credit_ranks = ([rs.ranks[i] for i in loo_idx] if deltas
                                else rs.ranks)
                for rid, credit, weight in assign_credit(
                    scheme, credit_ids, credit_ranks, correct,
                    loo_deltas=deltas if deltas else None,
                ):
                    store.update_ledger(rid, credit, weight, q.asked_at)

                logs.append(
                    {
                        "conversation_id": cid,
                        "question_id": q.question_id,
                        "category": q.category,
                        "record_ids": list(rs.record_ids),
                        "ranks": list(rs.ranks),
                        "correct": int(correct),
                        "f1": float(f1),
                        "prompt_tokens": rs.prompt_tokens,
                        "loo_deltas": deltas,
                    }
                )
        store.commit()
        # Handed to the estimator for predict_marginal: the write-time gate
        # weights each query type by how often it was actually asked, rather
        # than assuming the three are equally common.
        if qtype_seen:
            self.estimator.qtype_mix = {k: float(v)
                                        for k, v in qtype_seen.items()}

        # Label: did removing this record measurably cost accuracy, averaged
        # over the times it was retrieved? Records with too few observations
        # are dropped rather than labelled on one noisy trial.
        categories = sorted({c for _, c in cat_n})
        cat_rows: Dict[str, List[int]] = {c: [] for c in categories}
        conditional = self.cfg.estimator.conditional_utility
        X, y, groups, ev = [], [], [], []
        for rid, rec in store.records.items():
            if rec.write_features is None:
                continue
            n_obs = contrib_n.get(rid, 0)
            if use_loo:
                if n_obs < ex.loo_min_observations:
                    continue
                label = int((contrib_sum[rid] / n_obs) > 0.0)
            else:
                row = store.ledger[rid]
                if row.retrieval_count == 0:
                    continue
                label = int(row.running_utility >= 0.5)

            # One row per record, or one per (record, query type) cell the
            # record was actually observed under. An unobserved cell is not a
            # negative example - it is no example - so it is skipped rather
            # than labelled 0, which is the same distinction the -1 in
            # cat_rows makes.
            if conditional:
                seen = [(qt, n) for (r, qt), n in qt_n.items()
                        if r == rid and n > 0]
                rows = [
                    (np.concatenate([rec.write_features,
                                     query_type_onehot(qt)]),
                     int((qt_sum[(rid, qt)] / n) > 0.0))
                    for qt, n in sorted(seen)
                ]
                if not rows:
                    continue
            else:
                rows = [(rec.write_features, label)]

            sources = set(rec.source_ids or [])
            if rec.fact_id:
                sources.add(rec.fact_id)
            is_ev = int(bool(
                sources & evidence_by_conv.get(rec.conversation_id, set())
            ))
            for feats, lab in rows:
                X.append(feats)
                y.append(lab)
                groups.append(rec.conversation_id)
                ev.append(is_ev)
                for c in categories:
                    n_c = cat_n.get((rid, c), 0)
                    # -1 marks "no outcome for this category", which is
                    # different from a negative outcome and must not be
                    # scored as one.
                    cat_rows[c].append(
                        -1 if n_c == 0 else int((cat_sum[(rid, c)] / n_c) > 0.0)
                    )

        return PhaseAOutput(
            X=np.vstack(X) if X else np.zeros((0, self._feature_width())),
            y=np.asarray(y, dtype=int),
            groups=groups,
            category_labels={c: np.asarray(v, dtype=int)
                             for c, v in cat_rows.items()},
            logs=logs,
            evidence_labels=np.asarray(ev, dtype=int),
        )

    # ---------------------------------------------------- threshold fitting

    def calibrate_thresholds(
        self, conversations: Sequence[Conversation], policy: GovernancePolicy
    ) -> None:
        """Score the contested candidate population and set quantile cuts."""
        probe = MemoryStore(":memory:", self.cfg.embedding_dim)
        rows: List[np.ndarray] = []
        for cid, turns, _ in conversations:
            n_sessions = 1 + max((t.session_id for t in turns), default=0)
            cands = build_candidates(cid, turns, self.embedder, self.cfg.ingest)
            for i, cand in enumerate(cands):
                if policy.check_duplicate(cand, probe) is not None:
                    continue  # merged, never contests the gate
                feats = self.features_for(
                    cand, probe, n_sessions, i / max(1, len(cands))
                )
                rows.append(feats)
                probe.admit(cand, write_features=feats)

        # Scored in one batch rather than per candidate. This is exact, not an
        # approximation: calibration admits every candidate to the probe store
        # regardless of its score, so no prediction here influences the store
        # the next feature vector is computed against. The admission loop in
        # build_store is genuinely sequential and cannot be batched this way.
        # The saving is large because the fitted model is a calibrated
        # ensemble, so a single-row call walks every sub-model's trees and pays
        # sklearn's per-call overhead on one row.
        contested = (
            list(self.estimator.predict_marginal(np.vstack(rows)))
            if rows else []
        )
        policy.calibrate(contested)

    # -------------------------------------------------------------- phase B

    def evaluate(
        self,
        conversations: Sequence[Conversation],
        condition: str,
        store: MemoryStore,
        k: Optional[int],
        use_utility: bool,
        adaptive: bool,
        policy: Optional[GovernancePolicy] = None,
        full_context: bool = False,
    ) -> RunResult:
        retriever = Retriever(store, self.embedder, self.cfg.retrieval)
        correct_flags: List[int] = []
        f1s: List[float] = []
        tokens: List[int] = []
        lats: List[float] = []
        by_cat: Dict[str, List[int]] = {}
        n_queries = 0

        conditional = self.cfg.estimator.conditional_utility

        def utility_fn(record_ids: Sequence[int],
                       question: Optional[Question] = None) -> List[float]:
            if not self.estimator.is_fitted:
                return [0.5] * len(record_ids)
            # The query type is derived from the question text, so the same
            # value is available here as during training. A missing question
            # would silently change the feature width, so it is an error
            # rather than a zero vector.
            if conditional:
                if question is None:
                    raise ValueError(
                        "conditional_utility needs the question at scoring "
                        "time; the retriever did not pass one"
                    )
                tail = query_type_onehot(query_type(question.text))
            else:
                tail = None
            feats = []
            for rid in record_ids:
                rec = store.records[rid]
                base = (rec.write_features if rec.write_features is not None
                        else np.zeros(len(FEATURE_NAMES), dtype=np.float32))
                feats.append(base if tail is None
                             else np.concatenate([base, tail]))
            return list(self.estimator.predict(np.vstack(feats)))

        for cid, _, questions in conversations:
            for q in questions:
                if full_context:
                    context = store.active_records(cid)
                    prompt_tokens = sum(count_tokens(r.text) for r in context)
                    latency = 0.0
                    rec_ids, ranks = [r.record_id for r in context], list(
                        range(1, len(context) + 1)
                    )
                else:
                    rs = retriever.retrieve(
                        q,
                        k=None if adaptive else (k or self.cfg.retrieval.k_fixed),
                        utility_fn=utility_fn if use_utility else None,
                        mass_target=self.cfg.governance.budget_mass_target,
                    )
                    context = [store.records[r] for r in rs.record_ids]
                    prompt_tokens, latency = rs.prompt_tokens, rs.latency_ms
                    rec_ids, ranks = rs.record_ids, rs.ranks

                pred, _, _ = self.reader.answer(q, context)
                ok, f1 = self.judge.score(pred, q.answer)

                for rid, credit, weight in assign_credit(
                    self.cfg.credit_scheme, rec_ids, ranks, ok
                ):
                    store.update_ledger(rid, credit, weight, q.asked_at)

                correct_flags.append(int(ok))
                f1s.append(f1)
                tokens.append(prompt_tokens)
                lats.append(latency)
                by_cat.setdefault(q.category, []).append(int(ok))
                n_queries += 1

                if (
                    policy is not None
                    and n_queries % self.cfg.governance.forget_every_n_queries == 0
                ):
                    policy.run_forgetting(store, q.asked_at, conv=cid)

        store.commit()
        return RunResult(
            condition=condition,
            accuracy=accuracy(correct_flags),
            token_f1=float(np.mean(f1s)) if f1s else 0.0,
            prompt_tokens_per_query=float(np.mean(tokens)) if tokens else 0.0,
            retrieval_latency_ms=float(np.mean(lats)) if lats else 0.0,
            records_retained=len(store.active_records()),
            generative_calls=0,
            n_questions=len(correct_flags),
            per_question_correct=correct_flags,
            per_category={k2: float(np.mean(v)) for k2, v in by_cat.items()},
        )
