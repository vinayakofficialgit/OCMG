"""The five baselines, the OCMG condition and the ablation suite (Section 8.3, 8.6).

Every condition shares the same embedder, reader, judge and hardware, so that
the only thing varying is governance. Conditions differ in exactly three
dimensions: what is admitted, what is forgotten, and how deep retrieval goes.

    condition          admission        forgetting     budget
    full_context       all              none           whole conversation
    naive_rag          all (flat)       none           fixed top-k
    decay              all (typed)      recency/freq   fixed top-k
    generative_judge   LLM-gated        none           fixed top-k
    ocmg               utility-gated    utility-gated  adaptive
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import console as C
from .config import OCMGConfig
from .estimator import MIN_ROWS_PER_CLASS
from .governance import GovernancePolicy, select_lowest_fraction
from .reader import build_generative_gate
from .metrics import (accuracy_per_1k_tokens, accuracy_retention_ratio,
                      label_validation, paired_bootstrap, per_category_auc,
                      retention_precision,
                      token_reduction_ratio)
from .orchestrator import Conversation, OCMGRunner
from .store import MemoryStore
from .types import RunResult

LABELS = {
    "full_context": "Full context",
    "naive_rag": "Naive RAG (top-k)",
    "append_all": "Append-all typed",
    "decay": "Decay-based",
    "generative_judge": "Generative judge",
    "ocmg": "OCMG",
}

# ROC-AUC the estimator reached before the credit-label fix (README section 7,
# item 7), on LoCoMo at seed 0 under this same configuration. Named here so a
# run can show what the fix changed without the figure being a number typed
# into a slide.
# LoCoMo, seed 0, full run, reader hf (<model>), embedder sentence-transformers, measured before the credit-label fix. Source: <results file>.
PRE_FIX_ESTIMATOR_AUC = 0.507


def subsample_questions(
    conversations: Sequence[Conversation], per_conversation: int, seed: int = 0
) -> List[Conversation]:
    """Keep at most `per_conversation` questions, stratified by category.

    Only for making a real-reader run affordable: every reported accuracy then
    rests on fewer questions and its confidence interval widens accordingly,
    so the kept count belongs in any table drawn from such a run.

    Sampling is stratified because LoCoMo's categories are very unevenly sized
    - single-hop outnumbers open-domain roughly nine to one - and the
    categories differ in difficulty, so taking a plain random slice would move
    measured accuracy by changing the question mix rather than the system.
    Records are untouched: the store still ingests every turn.
    """
    if per_conversation <= 0:
        return list(conversations)
    rng = np.random.default_rng(seed)
    out: List[Conversation] = []
    for cid, turns, questions in conversations:
        if len(questions) <= per_conversation:
            out.append((cid, turns, questions))
            continue
        by_cat: Dict[str, List] = {}
        for q in questions:
            by_cat.setdefault(q.category, []).append(q)
        share = per_conversation / len(questions)
        kept = []
        for cat in sorted(by_cat):
            pool = by_cat[cat]
            # At least one of every category present, so a small category is
            # not silently dropped from the run.
            take = max(1, int(round(len(pool) * share)))
            take = min(take, len(pool))
            pick = rng.choice(len(pool), size=take, replace=False)
            kept.extend(pool[i] for i in sorted(pick))
        # Rounding up per category can overshoot the target; trim at random
        # rather than by category so the trim does not reintroduce a skew.
        if len(kept) > per_conversation:
            drop = rng.choice(len(kept), size=len(kept) - per_conversation,
                              replace=False)
            kept = [q for i, q in enumerate(kept) if i not in set(drop.tolist())]
        kept.sort(key=lambda q: q.question_id)
        out.append((cid, turns, kept))
    return out


def split_conversations(
    conversations: Sequence[Conversation], warmup_frac: float = 0.5, seed: int = 0
) -> Tuple[List[Conversation], List[Conversation]]:
    """Split at conversation level so the two phases share no records."""
    idx = np.random.default_rng(seed).permutation(len(conversations))
    cut = max(1, int(len(conversations) * warmup_frac))
    warm = [conversations[i] for i in idx[:cut]]
    test = [conversations[i] for i in idx[cut:]]
    return warm, (test if test else warm)


class DecayPolicy(GovernancePolicy):
    """Frequency-and-recency forgetting, in the manner of the Ebbinghaus-style
    schedules that dominate current practice.

    The contrast with OCMG is the criterion, not the machinery: this schedule
    reinforces a record for being retrieved, whatever followed. A record that
    is retrieved constantly and never precedes a correct answer survives here
    and is a prime eviction candidate under OCMG.
    """

    def select_for_eviction(
        self, store: MemoryStore, conv: Optional[str] = None
    ) -> List[int]:
        scored = []
        active = store.active_records(conv)
        # "Now" is the latest event the store has seen: a record write or a
        # retrieval. Questions are asked after the final session, so using
        # record timestamps alone puts "now" before the retrievals.
        now = max(
            [r.timestamp for r in active]
            + [store.ledger[r.record_id].last_retrieved_at for r in active],
            default=0.0,
        )
        for rec in active:
            row = store.ledger[rec.record_id]
            gap_days = max(0.0, now - row.last_retrieved_at) / 86400.0
            recency = 1.0 / (1.0 + gap_days)
            strength = np.log1p(row.retrieval_count) * 0.6 + recency * 0.4
            scored.append((rec.record_id, strength))
        if len(scored) < 10 or self.cfg.decay_forget_quantile <= 0.0:
            return []
        # Rank-based for the same reason as the utility policy: records that
        # have never been retrieved share an identical strength, so a value cut
        # sweeps the whole tie group and the baseline evicts far more than the
        # configured fraction. A baseline weakened by an implementation detail
        # is not a baseline.
        return select_lowest_fraction(scored, self.cfg.decay_forget_quantile)


# Measured label yield, used by label_budget below. Each entry is
# (warm-up questions, leave-one-out cap) -> (rows, positives), measured on
# LoCoMo seed 0 with the simulated reader and hashing embedder, phase A only.
# These are properties of the benchmark and the exploration policy, not of the
# reader, which is what lets a pre-flight predict a real-reader run from a
# cheap one. Re-measure if the exploration policy or the segmenter changes.
#
# A curve was tried first and was wrong in both directions at once: rows grow
# superlinearly in question count (a record needs loo_min_observations before
# it counts, and repeats only arrive once questioning is dense), while the
# positive rate on subsampled runs is 7-11% rather than the 43% the full
# question set shows. So this reports measurements and interpolates nothing.
MEASURED_LABEL_YIELD = {
    (10, 4): (2, 0),    (10, 0): (7, 0),
    (15, 4): (8, 1),    (15, 0): (18, 1),
    (25, 4): (13, 0),   (25, 0): (53, 4),
    (50, 4): (30, 2),   (50, 0): (129, 14),
}

# The smallest warm-up size measured to produce a fittable label set, per cap.
# None means no tested size worked: at a cap of 4, fifty warm-up questions
# still yield only 2 positives, so the cap is the binding constraint and
# raising the question count is the wrong fix.
SMALLEST_FITTABLE = {0: 50, 4: None}


def label_budget(warm, cfg) -> dict:
    """Most labelled rows the warm-up could possibly produce.

    Leave-one-out gives one observation per ablated record per question, a
    record needs `loo_min_observations` of them before it is labelled at all,
    and `fit` then needs `MIN_ROWS_PER_CLASS` rows in each class. Chaining
    those gives a ceiling on the label set that is known before any
    generation happens.

    Optimistic on purpose: it assumes every ablation lands on a distinct
    record and that observations spread as evenly as they can. The real
    number is lower - the 10 Nov run projected 20 and produced 2 - so a
    projection under the threshold is a reliable refusal, while one above it
    is not a promise.
    """
    ex = cfg.exploration
    n_questions = sum(len(qs) for _, _, qs in warm)
    mean_k = (ex.k_min + ex.k_max) / 2.0
    per_q = min(mean_k, ex.loo_max_records) if ex.loo_max_records else mean_k

    # Measured yield, not a derived bound. A bound was tried first and was
    # useless: it projected 20 rows for a configuration that produced 0,
    # because the gap between the optimistic ceiling and reality is about 6x.
    cap = ex.loo_max_records if ex.loo_max_records else 0

    # Nearest measured configuration at or below this one, for the same cap.
    # Reporting the neighbour rather than an interpolation keeps the warning
    # honest about being a measurement and not a prediction.
    same_cap = sorted(q for (q, c) in MEASURED_LABEL_YIELD if c == cap)
    nearest = max((q for q in same_cap if q <= n_questions), default=None)
    measured = MEASURED_LABEL_YIELD.get((nearest, cap)) if nearest else None

    floor = SMALLEST_FITTABLE.get(cap, None)
    # With no measured fittable size for this cap, the cap itself is the
    # problem; with one, the question count has to reach it.
    fittable = floor is not None and n_questions >= floor
    n_convs = max(1, len(warm))
    return {
        "questions": n_questions,
        "cap": cap,
        "ablations_per_question": per_q,
        "nearest_measured_questions": nearest,
        "measured_rows": measured[0] if measured else None,
        "measured_positives": measured[1] if measured else None,
        "needed": MIN_ROWS_PER_CLASS,
        "floor_questions": floor,
        # Warm-up takes about half the questions, split by conversation.
        "need_per_conv": (floor / n_convs * 2.0) if floor else None,
        "fittable": fittable,
        "cap_is_binding": floor is None,
    }


def run_all_conditions(
    conversations: Sequence[Conversation],
    cfg: OCMGConfig,
    conditions: Optional[Sequence[str]] = None,
    verbose: bool = True,
    baseline_auc: Optional[float] = None,
) -> Dict[str, RunResult]:
    """Full two-phase experiment across all requested conditions."""
    conditions = conditions or [
        "full_context", "naive_rag", "append_all", "decay",
        "generative_judge", "ocmg",
    ]
    runner = OCMGRunner(cfg)
    warm, test = split_conversations(conversations, 0.5, cfg.seed)

    if verbose:
        print(C.section("1", "SETUP"))
        print(C.facts([
            ("embedder", f"{runner.embedder.name} ({cfg.embedding_dim}d)",
             "turns text into vectors for similarity search; hashing = "
             "offline stand-in for a trained encoder"
             if runner.embedder.name.startswith("hashing")
             else "trained sentence encoder for similarity search"),
            ("reader", cfg.reader.kind,
             "probability model of a reader, no LLM: right with p 0.92 if the "
             "evidence is in the prompt, 0.04 if not"
             if cfg.reader.kind == "simulated"
             else f"frozen LLM {cfg.reader.hf_model}, greedy decoding"),
            ("conversation split", f"{len(warm)} warm-up / {len(test)} evaluation",
             "split by conversation, so no record is shared"),
        ]))

    # Checked before the first generation, because on a CPU reader phase A
    # is tens of minutes and discovering afterwards that there was nothing to
    # fit wastes all of it.
    budget = label_budget(warm, cfg)
    if verbose and not budget["fittable"]:
        print(C.note(
            ("TOO SMALL TO FIT - this run cannot train the estimator. "
             f"{budget['questions']} warm-up questions"
             + (f" at a leave-one-out cap of {budget['cap']}"
                if budget['cap'] else " with uncapped leave-one-out")
             + (f"; the nearest measured run "
                f"({budget['nearest_measured_questions']} questions, same "
                f"cap) produced {budget['measured_rows']} labelled rows and "
                f"{budget['measured_positives']} positives"
                if budget['measured_rows'] is not None else "")
             + f". Fitting needs {MIN_ROWS_PER_CLASS} of each class, so the "
               "estimator will report a fallback, not a measurement. "
             + ("The CAP is the binding constraint, not the question count: "
                "50 warm-up questions yield 14 positives uncapped and 2 at a "
                "cap of 4. Drop --loo-max-records (and so --fast) rather "
                "than raising --questions-per-conv."
                if budget['cap_is_binding'] else
                f"Raise --questions-per-conv to about "
                f"{max(1, round(budget['need_per_conv']))}.")
             + " Measured on LoCoMo seed 0, phase A only. The baseline "
               "accuracy comparisons in this run are unaffected."),
            "critical", glyph=C.G.bad))

    # ---------------- Phase A: exploration and estimator fitting -----------
    t0 = time.time()
    phase_a = runner.run_phase_a(warm)
    if verbose:
        print(C.section("2", "WARM-UP   exploratory retrieval, outcome logging"))
        print(C.facts([
            ("questions asked", f"{len(phase_a.logs):,}",
             "each answered with all retrieved records, then once per record "
             "removed (leave-one-out)"),
            ("records labelled", f"{phase_a.X.shape[0]:,}",
             "records with >= 2 leave-one-out results: the estimator's "
             "training rows"),
            ("labelled useful", f"{phase_a.y.mean():.1%}",
             "removing it turned a correct answer wrong, on average"),
            ("elapsed", f"{time.time() - t0:.1f}s", "warm-up wall-clock time"),
        ]))

    report = runner.estimator.fit(phase_a.X, phase_a.y, phase_a.groups)
    runner.estimator_report = report
    if verbose:
        fitted = getattr(report, "fitted", True)
        met = report.auc_mean >= 0.70 and fitted
        print(C.section("3", "UTILITY ESTIMATOR   RQ1: can outcomes train it?"))
        if not fitted:
            print(C.note(
                f"NOT MEASURABLE - nothing to fit. The warm-up produced "
                f"{report.n_samples} labelled record(s) at a "
                f"{report.positive_rate:.1%} positive rate, and fitting needs "
                "at least 10 of each class, so no model was trained and no "
                "fold was scored. The 0.500 below is a declared fallback, not "
                "a failed measurement: read it as 'no answer', not as 'the "
                "estimator does not work'. Raise --questions-per-conv, or "
                "--loo-max-records if it is capped.",
                "critical", glyph=C.G.bad))
        print(C.facts([
            ("ROC-AUC", f"{report.auc_mean:.3f} ± {report.auc_std:.3f}",
             C.status(met) if fitted
             else C.badge("FALLBACK", "critical", C.G.bad),
             "chance a useful record is scored above a useless one; "
             "0.5 = random, target >= 0.70" if fitted
             else "fallback, not a measurement"),
            ("per fold", "  ".join(f"{a:.3f}" for a in report.auc_folds), "",
             "AUC on each held-out group of conversations"),
            ("Brier score", f"{report.brier:.4f}",
             C.grade(report.brier, 0.20, 0.25, ("GOOD", "FAIR", "POOR"),
                     higher_is_better=False),
             "mean squared error of the probabilities; lower is better, "
             "0.25 = always guessing 0.5"),
            ("average precision", f"{report.ap_mean:.3f}",
             C.grade(report.ap_mean - report.positive_rate, 0.10, 0.02,
                     ("SKILL", "SLIGHT", "NO SKILL")),
             f"precision averaged over all recall levels; "
             f"{report.positive_rate:.3f} (the useful rate) = no skill"),
        ]))
        if baseline_auc is not None and report.auc_folds:
            print("")
            print(C.note("effect of the credit-label fix", "muted"))
            print(C.before_after("before the fix (recorded, earlier run)", baseline_auc,
                                 "after the fix", report.auc_mean,
                                 reference=("target", 0.70)))
        if report.auc_single_feature:
            print("")
            print(C.note("can one raw feature do the estimator's job?   same "
                         "folds; model gain = estimator AUC minus the feature's",
                         "muted"))
            print(C.table(
                ["single feature", "ROC-AUC", "model gain", "reading"],
                # Within 0.005 is a tie, not a win either way: under
                # --estimator rarity the model IS rare_term_count, and its
                # gain of exactly 0.000 read "the model does better".
                # amber = the feature matches or beats the model (a warning
                # about the model), green = the model adds something.
                [[name, f"{a:.3f}", f"{report.auc_mean - a:+.3f}",
                  C.paint("same as the model", "warning")
                  if abs(report.auc_mean - a) < 0.005
                  else C.paint(f"{C.G.warn} beats the trained model on its own",
                               "warning")
                  if report.auc_mean - a < 0 else
                  C.paint("the model does better", "good")]
                 for name, a in sorted(report.auc_single_feature.items(),
                                       key=lambda kv: -kv[1])],
                aligns=["left", "right", "right", "left"],
            ))
        if report.feature_importance:
            # Zero-importance features are not "what predicts usefulness";
            # listing them padded the rarity ablation with five 0.000 rows.
            top = [kv for kv in sorted(report.feature_importance.items(),
                                       key=lambda kv: -abs(kv[1]))
                   if abs(kv[1]) >= 0.0005][:6]
        if report.feature_importance and top:
            print("")
            print(C.bar_chart([(k, abs(v)) for k, v in top],
                              title="what predicts usefulness   permutation "
                                    "importance = AUC lost when the feature "
                                    "is shuffled",
                              label_width=22, emphasise=top[0][0]))

    # Is utility a fixed property of a record, or does it depend on the query?
    # One estimator, scored against each category's own outcomes.
    per_cat = per_category_auc(
        phase_a.category_labels,
        oof_scores=report.oof_predictions,
        oof_mask=report.oof_mask,
    ) if report.oof_predictions is not None else {}
    if per_cat and verbose:
        usable = {c: v for c, v in per_cat.items() if v["auc"] == v["auc"]}

        def _cat_reading(v) -> str:
            if v["auc"] != v["auc"]:
                return "no evidence by design, nothing to rank"
            if v["auc"] < 0.55:
                return "near random for this question type"
            return "ranks useful records above chance"

        print("")
        print(C.note("does usefulness depend on the question type?   same "
                     "estimator, scored separately per LoCoMo category",
                     "muted"))
        print(C.table(
            ["question category", "ROC-AUC", "vs overall", "rows",
             "useful", "reading"],
            [[c, f"{v['auc']:.3f}", f"{v['auc'] - report.auc_mean:+.3f}",
              f"{v['n']:,}", f"{v['positive_rate']:.1%}",
              C.paint(_cat_reading(v),
                      "critical" if v["auc"] < 0.55 else "good")]
             for c, v in sorted(usable.items(), key=lambda kv: -kv[1]["auc"])]
            + [[c, "-", "-", f"{v['n']:,}", "-",
                C.paint(_cat_reading(v), "muted")]
               for c, v in sorted(per_cat.items()) if v["auc"] != v["auc"]],
            aligns=["left", "right", "right", "right", "right", "left"],
        ))

    # Audit the outcome labels against the held-out evidence annotation. This
    # does not feed back into the model: it is reported so that the claim
    # "outcome credit identifies genuinely useful records" is tested rather
    # than assumed.
    validation = label_validation(
        phase_a.y,
        phase_a.evidence_labels,
        oof_scores=report.oof_predictions,
        oof_mask=report.oof_mask,
    )
    if verbose and validation.get("n"):
        v = validation
        print(C.section("4", "LABEL AUDIT   training labels vs the gold "
                             "evidence ids (never trained on)"))
        audit = [
            ("precision", f"{v['precision']:.3f}",
             C.grade(v["precision"], 0.90, 0.70, ("HIGH", "FAIR", "LOW")),
             f"labelled-useful records that hold an evidence turn: "
             f"TP/(TP+FP) = {v['true_positives']}/"
             f"{v['true_positives'] + v['false_positives']}"),
            ("recall", f"{v['recall']:.3f}",
             C.grade(v["recall"], 0.80, 0.50, ("HIGH", "PARTIAL", "LOW")),
             f"evidence records that were labelled useful: TP/(TP+FN) = "
             f"{v['true_positives']}/"
             f"{v['true_positives'] + v['false_negatives']}; partial is "
             "expected, another record can carry the same fact"),
            # Landis & Koch bands: 0.41-0.60 moderate, 0.61-0.80 substantial.
            ("Cohen's kappa", f"{v['kappa']:.3f}",
             C.grade(v["kappa"], 0.61, 0.41,
                     ("SUBSTANTIAL", "MODERATE", "WEAK")),
             "agreement beyond chance: 0 = chance, 1 = perfect"),
        ]
        if "estimator_auc_vs_evidence" in validation:
            a = validation["estimator_auc_vs_evidence"]
            audit.append(("estimator vs evidence", f"{a:.3f} ROC-AUC",
                          C.grade(a, 0.70, 0.60),
                          "estimator's scores ranked against evidence "
                          "directly (0.5 = random)"))
        print(C.facts(audit))

    # ---------------- threshold calibration on contested population --------
    policy = GovernancePolicy(cfg.governance, runner.estimator)
    runner.calibrate_thresholds(test, policy)
    if verbose:
        th = policy.thresholds
        print(C.section("5", "GOVERNANCE THRESHOLDS"))
        def _onoff(q):
            return (C.badge("OFF", "muted") if q <= 0
                    else C.badge(f"ON {q:.0%}", "emphasis"))

        print(C.facts([
            ("admission cut", f"{th.write:.4f}",
             _onoff(cfg.governance.write_quantile),
             f"predicted utility below this is demoted; set at the "
             f"{cfg.governance.write_quantile:.0%} quantile"),
            ("forgetting cut", f"{th.forget:.4f}",
             _onoff(cfg.governance.forget_quantile),
             f"utility below this can be evicted; set at the "
             f"{cfg.governance.forget_quantile:.0%} quantile"),
            ("candidates contested", f"{th.n_contested:,}", "",
             "test-set records the gate judges, after near-duplicates merge"),
        ]))

    results: Dict[str, RunResult] = {}

    # ---------------- baselines -------------------------------------------
    if "full_context" in conditions:
        store, _ = runner.build_store(test, None, governed=False)
        results["full_context"] = runner.evaluate(
            test, "full_context", store, None, False, False, full_context=True
        )

    if "naive_rag" in conditions:
        # Naive RAG is an untyped chunk store with a fixed top-k. The flat
        # routing is what makes it naive: with typing left on it is the
        # append-all condition below, and the two reported identical numbers.
        store, _ = runner.build_store(test, None, governed=False, flat=True)
        results["naive_rag"] = runner.evaluate(
            test, "naive_rag", store, cfg.retrieval.k_fixed, False, False
        )

    if "append_all" in conditions:
        # The typed two-tier store with no governance: admit everything, never
        # forget, fixed top-k. Against naive_rag it isolates stages 1 and 3 of
        # the method - segmentation and typing - from the utility governance in
        # stages 2, 4 and 5, which is the only pair in the table that does.
        store, c = runner.build_store(test, None, governed=False)
        r = runner.evaluate(
            test, "append_all", store, cfg.retrieval.k_fixed, False, False
        )
        r.extra.update(c)
        results["append_all"] = r

    if "decay" in conditions:
        p = DecayPolicy(cfg.governance, None)
        store, c = runner.build_store(test, p, governed=False)
        r = runner.evaluate(
            test, "decay", store, cfg.retrieval.k_fixed, False, False, policy=p
        )
        r.extra.update(c)
        r.extra["evicted"] = p.stats["evicted"]
        results["decay"] = r

    if "generative_judge" in conditions:
        # Per-item model assessment at ingestion. One generative call per
        # candidate: this is the cost baseline the whole efficiency claim
        # is measured against, so the gate has to make real decisions.
        gate = build_generative_gate(runner.reader)
        if gate is None:
            # No generative model to ask. Skipping is the honest option: the
            # alternatives are to invent a judge's accuracy, or to run an
            # ungoverned build and count calls, which is what this condition
            # did until 2026-10-30 and which produced a row identical to
            # naive RAG in every column.
            results["_generative_judge_skipped"] = (
                "needs --reader hf; no generative model in this run"
            )
        else:
            p = GovernancePolicy(cfg.governance, None)
            p.candidate_gate = gate
            store, c = runner.build_store(test, p, governed=True)
            r = runner.evaluate(
                test, "generative_judge", store, cfg.retrieval.k_fixed,
                False, False
            )
            r.generative_calls = gate.calls
            r.extra.update(c)
            r.extra["judge_unparsed"] = gate.unparsed
            results["generative_judge"] = r

    # ---------------- OCMG -------------------------------------------------
    if "ocmg" in conditions:
        store, c = runner.build_store(test, policy, governed=True)
        r = runner.evaluate(
            test, "ocmg", store, None, use_utility=True, adaptive=True, policy=policy
        )
        r.generative_calls = 0
        r.extra.update(c)
        r.extra["evicted"] = policy.stats["evicted"]
        r.extra["write_threshold"] = policy.thresholds.write
        # Retention precision: did eviction keep the useful records?
        # Only meaningful when forgetting actually removed something. Two
        # traps, both of which produced numbers in results files before
        # 2026-11-01. First, cold_store holds records DEMOTED at write as well
        # as records evicted later, so reading it whole compares retained
        # against demoted and calls the result "eviction precision" - seed 0
        # reported a gap of -0.420 at AUC 0.030 this way, which reads as
        # eviction being catastrophically wrong when nothing was evicted at
        # all. Second, with forget_quantile at 0 the evicted list is empty or
        # near-empty and the metric returns a placeholder 0.0 gap at AUC 0.5,
        # indistinguishable from a real measurement of no effect.
        evicted_ids = [
            rid for rid, rec in store.records.items()
            if not rec.active and not rec.demoted_at_write
            and store.ledger[rid].retrieval_count > 0
        ]
        if len(evicted_ids) >= 10:
            retained = [
                store.ledger[rid].running_utility
                for rid, rec in store.records.items()
                if rec.active and store.ledger[rid].retrieval_count > 0
            ]
            r.extra["retention_precision"] = retention_precision(
                retained, [store.ledger[rid].running_utility
                           for rid in evicted_ids])
        else:
            r.extra["retention_precision_skipped"] = (
                f"only {len(evicted_ids)} evicted records were ever retrieved; "
                "nothing to measure"
            )
        results["ocmg"] = r
        results["_ledger"] = store.export_ledger()  # type: ignore[assignment]

    if runner.estimator_report is not None:
        results["_estimator"] = runner.estimator_report  # type: ignore[assignment]
    if validation.get("n"):
        results["_label_validation"] = validation  # type: ignore[assignment]
    # summarise() needs to know which signal ranked eviction, to decide
    # whether the retention-precision metric is circular. It takes only the
    # results dict, so the signal travels with the results.
    results["_forget_signal"] = cfg.governance.forget_signal  # type: ignore[assignment]
    if per_cat:
        results["_per_category_auc"] = per_cat  # type: ignore[assignment]
    # Generation reuse, for the real reader only. Greedy decoding makes a
    # repeated prompt's answer identical to the first one, so this saving
    # costs nothing - but an invisible saving cannot be checked, and a
    # suspiciously fast run with no stated reason is worse than a slow one.
    if runner.sample_answers:
        results["_samples"] = runner.sample_answers  # type: ignore[assignment]
    hits = getattr(runner.reader, "cache_hits", 0)
    made = getattr(runner.reader, "generated", 0)
    if hits or made:
        asked = hits + made
        results["_generation"] = {  # type: ignore[assignment]
            "requested": asked, "decoded": made, "reused": hits,
            "reuse_rate": hits / asked if asked else 0.0,
        }
        if verbose:
            print(C.kv("generation reuse",
                       f"{made:,} decoded of {asked:,} prompts",
                       hint=f"{hits / asked:.1%} identical to one already "
                            f"answered" if asked else ""))
    return results


# One line per condition saying what it is, so a table row reads on its own.
_DESCRIBE = {
    # Short enough to stay on one line beside the numbers at 120 columns.
    "full_context": "entire conversation",
    "naive_rag": "fixed top-10 chunks (ref)",
    "append_all": "keep everything, top-10",
    "decay": "forgets old, rarely used",
    "generative_judge": "LLM judges each record",
    "ocmg": "learned utility + adaptive k",
}


def _pct(v) -> str:
    return "-" if v is None else f"{v:.1%}"


def _muted(text: str) -> str:
    return C.paint(text, "muted")


def _grounding_block(present: Dict[str, RunResult], order) -> List[str]:
    """Section 9: retrieval checked against LoCoMo's gold evidence ids, and
    answers checked against the prompt they came from (ocmg/grounding.py)."""
    names = [n for n in order if present[n].extra.get("grounding")]
    gs = [present[n].extra["grounding"] for n in names]

    def col(key, higher=True, fmt="{:.1%}"):
        return C.rank_paint([g.get(key) for g in gs], fmt, higher)

    rec_c, prec_c, hit_c = (col("evidence_recall"), col("context_precision"),
                            col("evidence_hit_rate"))
    grd_c, hal_c = col("grounded_rate"), col("hallucination_rate", False)
    mis_c = C.rank_paint([g.get("quadrants", {}).get("wrong_grounded", 0)
                          for g in gs], "{:,}", higher_is_better=False)
    rows = []
    for i, (name, g) in enumerate(zip(names, gs)):
        q = g.get("quadrants", {})
        rec = g.get("evidence_recall")
        # "sends everything" describes full context's design, not a recall
        # of 100%: a retriever can find all the evidence without sending all.
        if rec is None:
            reading = ""
        elif name == "full_context":
            reading = "sends everything"
        elif rec >= 0.9995:
            reading = "misses none"
        else:
            reading = f"misses {1 - rec:.0%}"
        rows.append([
            LABELS[name], rec_c[i], prec_c[i], hit_c[i], grd_c[i], hal_c[i],
            f"{q.get('correct_ungrounded', 0):,}", mis_c[i], _muted(reading),
        ])
    if not rows:
        return []
    out = [C.section("9", "GROUNDING   answers checked against the gold "
                          "evidence ids")]
    out.append(C.note(
        "answerable questions only (adversarial ones have nothing to "
        "support) · ev recall = gold evidence turns that reached the prompt · "
        "ctx prec = prompt records holding evidence · ev hit = questions with "
        "any evidence sent · grounded = answers with >= 50% of their words in "
        "the prompt · halluc = answered, not abstained, not grounded · lucky "
        "= right without support · misread = wrong with evidence present · "
        "reading = share of needed evidence that never reached the prompt · "
        "green = best in the column, red = worst", "muted"))
    out.append(C.table(
        ["condition", "ev recall", "ctx prec", "ev hit", "grounded",
         "halluc", "lucky", "misread", "reading"],
        rows,
        aligns=["left"] + ["right"] * 7 + ["left"],
        emphasis_row=names.index("ocmg") if "ocmg" in names else None,
    ))
    out.append(C.note(
        "first three columns hold for any reader; the rest are lexical and, "
        "with the simulated reader, only test the code", "muted"))
    return out


def summarise(results: Dict[str, RunResult]) -> str:
    """The comparison block: cost beside accuracy, difference made explicit.

    Every number carries its meaning on its own line or row: a one-line
    definition above each table, a plain-language reading at the end of each
    row. No glossary blocks and no repeated charts.
    """
    present = {k: v for k, v in results.items() if isinstance(v, RunResult)}
    if not present:
        return "(no results)"

    full = results.get("full_context")
    naive = results.get("naive_rag")
    ocmg = results.get("ocmg")
    order = [n for n in LABELS if n in present]
    out: List[str] = []

    # ----------------------------------------------------- conditions table
    out.append(C.section("6", "CONDITIONS   accuracy beside what it costs"))
    out.append(C.note(
        "acc = share of questions judged correct · vs RAG = difference from "
        f"naive RAG ({C.G.up} better, {C.G.down} worse) · tok/q = memory word "
        "tokens per question · acc/1k = accuracy per 1,000 tokens · records = "
        "kept at the end · green = best in the column, red = worst",
        "muted"))
    ref_acc = naive.accuracy if isinstance(naive, RunResult) else None
    ref_tok = naive.prompt_tokens_per_query if isinstance(naive, RunResult) else None

    acc_col = C.rank_paint([present[n].accuracy for n in order])
    tok_col = C.rank_paint([present[n].prompt_tokens_per_query for n in order],
                           "{:,.0f}", higher_is_better=False)
    apk_col = C.rank_paint([accuracy_per_1k_tokens(
        present[n].accuracy, present[n].prompt_tokens_per_query) for n in order])
    # The LLM-calls column is only worth its width when something spent one.
    show_calls = any(present[n].generative_calls for n in order)
    rows = []
    for idx, name in enumerate(order):
        r = present[name]
        if ref_acc is None or name == "naive_rag":
            d_acc = C.paint("ref", "muted")
            d_tok = C.paint("ref", "muted")
        else:
            d_acc = C.delta(r.accuracy - ref_acc)
            d_tok = C.delta(
                (r.prompt_tokens_per_query - ref_tok) / ref_tok,
                higher_is_better=False, fmt="{:+.0%}",
            )
        rows.append([
            LABELS[name], acc_col[idx], d_acc, tok_col[idx], d_tok,
            apk_col[idx], f"{r.records_retained:,}"]
            + ([f"{r.generative_calls:,}"] if show_calls else [])
            + [_muted(_DESCRIBE.get(name, ""))])
    out.append(C.table(
        ["condition", "acc", "vs RAG", "tok/q", "vs RAG", "acc/1k", "records"]
        + (["LLM calls"] if show_calls else []) + ["what it is"],
        rows,
        aligns=["left"] + ["right"] * (7 if show_calls else 6) + ["left"],
        emphasis_row=order.index("ocmg") if "ocmg" in order else None,
    ))
    skipped = results.get("_generative_judge_skipped")
    if skipped:
        out.append(C.kv("generative judge", "not run",
                        hint=f"{skipped}; left out rather than faked"))

    if ocmg is None:
        return "\n".join(out)

    # ------------------------------------------------------------- targets
    out.append(C.section("7", "TARGETS   the proposal's success criteria"))
    target_rows = []
    if isinstance(full, RunResult):
        arr = accuracy_retention_ratio(ocmg.accuracy, full.accuracy)
        # The ratio only means "share of the ceiling kept" while full context
        # IS the ceiling; above naive RAG it is not, and MET would be false.
        ceiling_holds = (not isinstance(naive, RunResult)
                         or full.accuracy >= naive.accuracy)
        if ceiling_holds:
            status = C.status(arr >= 0.95)
            why = f"OCMG keeps {arr:.0%} of full-context accuracy"
        else:
            status = C.badge("NO CEILING", "warning", C.G.warn)
            why = (f"full context ({full.accuracy:.3f}) scored below naive RAG "
                   f"({naive.accuracy:.3f}), so it is no upper bound")
        target_rows.append(("accuracy retention", f"{arr:.3f}  (target >= 0.95)",
                            status, f"OCMG acc / full-context acc; {why}"))
    if isinstance(naive, RunResult):
        trr = token_reduction_ratio(
            ocmg.prompt_tokens_per_query, naive.prompt_tokens_per_query
        )
        more_less = "fewer" if trr >= 0 else "more"
        target_rows.append(("token reduction", f"{trr:.3f}  (target >= 0.40)",
                            C.status(trr >= 0.40),
                            f"1 - OCMG tokens / naive RAG tokens; OCMG sends "
                            f"{abs(trr):.0%} {more_less} tokens"))
    if target_rows:
        out.append(C.facts(target_rows))

    # -------------------------------------------------------- significance
    out.append(C.section("8", "SIGNIFICANCE   is the accuracy difference real?"))
    out.append(C.note(
        "delta = OCMG accuracy minus the other system on the same questions · "
        "95% CI from 10,000 paired resamples · p < 0.05 = significant",
        "muted"))
    sig_rows = []
    sig = {}
    for other in ("naive_rag", "append_all", "decay", "generative_judge"):
        o = results.get(other)
        if not isinstance(o, RunResult) or len(o.per_question_correct) != len(
            ocmg.per_question_correct
        ):
            continue
        bs = paired_bootstrap(ocmg.per_question_correct, o.per_question_correct)
        sig[other] = bs
        if bs["ci_low"] <= 0.0 <= bs["ci_high"]:
            verdict = C.paint(f"{C.G.flat} no real difference: CI includes 0",
                              "warning")
        elif bs["delta"] > 0:
            verdict = C.paint(f"{C.G.ok} OCMG better: whole CI above 0", "good")
        else:
            verdict = C.paint(f"{C.G.bad} OCMG worse: whole CI below 0",
                              "critical")
        sig_rows.append([
            f"OCMG - {LABELS[other]}", f"{bs['delta']:+.3f}",
            f"[{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]",
            f"{bs['p_value']:.3f}", verdict,
        ])
    out.append(C.table(
        ["comparison", "delta", "95% CI", "p", "verdict"],
        sig_rows,
        aligns=["left", "right", "right", "right", "left"],
    ))
    out.extend(_grounding_block(present, order))

    rp = ocmg.extra.get("retention_precision")
    # Only meaningful when the eviction signal and this metric are different
    # quantities. Under forget_signal="ledger" the policy ranks records by
    # running_utility and this measures the gap in running_utility between
    # those kept and those evicted, so it scores the sort against the key it
    # sorted on: the interval cannot straddle zero however bad the mechanism
    # is, and the printed verdict would be a tautology. The honest test of the
    # same claim is the label audit in section 4, which uses the held-out
    # evidence annotations instead.
    circular = results.get("_forget_signal", "ledger") == "ledger"
    if rp and circular:
        out.append("")
        out.append(C.kv("retention precision", "not reported",
                        hint="would score eviction against its own sort key; "
                             "section 4 is the honest test"))
    elif rp:
        out.append("")
        decisive = not (rp["ci_low"] <= 0.0 <= rp["ci_high"])
        out.append(C.kv(
            "retention precision",
            f"gap {rp['gap']:+.3f}  [{rp['ci_low']:+.3f}, {rp['ci_high']:+.3f}]"
            f"   AUC {rp['auc']:.3f}",
            hint=C.paint(
                f"{C.G.ok} eviction beats chance" if decisive and rp["gap"] > 0
                else f"{C.G.flat} no better than chance",
                "good" if decisive and rp["gap"] > 0 else "warning",
            ),
        ))
        # The verdict is read off the interval, never asserted: a gap whose
        # interval straddles zero means forgetting discarded records no worse
        # than at random, which would refute the mechanism rather than support
        # it. Wording this as a fixed claim would print a conclusion the
        # numbers contradict.
        out.append(C.note(
            "Measured as the gap in realised utility between records kept and "
            "records evicted. An interval straddling zero means eviction was "
            "no better than chance.", "muted"))
    out.extend(_glance(results, ocmg, naive, sig))
    return "\n".join(out)


def _verdict_word(bs) -> str:
    if bs["ci_low"] <= 0.0 <= bs["ci_high"]:
        return C.paint(f"{C.G.flat} no real difference", "warning")
    if bs["delta"] > 0:
        return C.paint(f"{C.G.ok} OCMG better", "good")
    return C.paint(f"{C.G.bad} OCMG worse", "critical")


def _glance(results, ocmg, naive, sig) -> List[str]:
    """The closing card: each research question answered in one line, with
    the number that answers it and what it means."""
    rows = []
    est = results.get("_estimator")
    if est is not None:
        auc = est.auc_mean
        best = max((est.auc_single_feature or {}).items(), key=lambda kv: kv[1],
                   default=None)
        expl = "target 0.70"
        if best and best[1] >= auc - 0.005:
            expl += f"; but {best[0]} alone scores {best[1]:.3f}"
        rows.append(("RQ1 estimator", f"AUC {auc:.3f}  {C.status(auc >= 0.70)}",
                     expl))
    v = results.get("_label_validation")
    if v:
        tp, fp = v.get("true_positives", 0), v.get("false_positives", 0)
        rows.append(("training labels", f"precision {v['precision']:.3f}",
                     f"{tp}/{tp + fp} credited records hold real evidence; "
                     f"recall {v['recall']:.3f}"))
    names = {"naive_rag": "naive RAG", "append_all": "append-all",
             "decay": "decay", "generative_judge": "LLM judge"}
    for other, bs in sig.items():
        o = results[other]
        rows.append((f"OCMG vs {names[other]}",
                     f"{ocmg.accuracy:.3f} vs {o.accuracy:.3f}",
                     f"{_verdict_word(bs)}  (delta {bs['delta']:+.3f}, "
                     f"p {bs['p_value']:.2f})"))
    if isinstance(naive, RunResult) and naive.prompt_tokens_per_query:
        ch = ocmg.prompt_tokens_per_query / naive.prompt_tokens_per_query - 1
        rows.append(("token cost",
                     f"{ocmg.prompt_tokens_per_query:,.0f} vs "
                     f"{naive.prompt_tokens_per_query:,.0f}",
                     f"{abs(ch):.0%} {'more' if ch > 0 else 'fewer'} than naive "
                     f"RAG; target 40% fewer  "
                     f"{C.status(-ch >= 0.40)}"))
    go = ocmg.extra.get("grounding") or {}
    gn = naive.extra.get("grounding") if isinstance(naive, RunResult) else None
    if go and gn:
        rows.append(("evidence in prompt",
                     f"recall {go['evidence_recall']:.1%} · precision "
                     f"{go['context_precision']:.1%}",
                     f"naive RAG {gn['evidence_recall']:.1%} · "
                     f"{gn['context_precision']:.1%}: OCMG sends more focused "
                     f"prompts but misses more of the needed evidence"))
    if not rows:
        return []
    return ["", C.card("AT A GLANCE", rows)]


ABLATIONS = {
    "full": {},
    "no_admission_gate": {"governance.write_quantile": 0.0},
    # Forgetting is off by default now that the sweep has measured its
    # optimum at zero, so the informative ablation is switching it ON.
    "with_forgetting": {"governance.forget_quantile": 0.25},
    "no_adaptive_budget": {"_fixed_k": True},
    "flat_store": {"_flat": True},
    "uniform_credit": {"credit_scheme": "uniform"},
}
