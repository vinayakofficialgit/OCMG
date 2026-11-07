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
        print(C.kv("embedder", f"{runner.embedder.name} ({cfg.embedding_dim}d)"))
        print(C.kv("reader", cfg.reader.kind))
        print(C.kv("conversation split",
                   f"{len(warm)} warm-up   {len(test)} evaluation",
                   hint="split by conversation, so no record is shared"))

    # ---------------- Phase A: exploration and estimator fitting -----------
    t0 = time.time()
    phase_a = runner.run_phase_a(warm)
    if verbose:
        print(C.section("2", "WARM-UP   exploratory retrieval, outcome logging"))
        print(C.kv("questions asked", f"{len(phase_a.logs):,}"))
        print(C.kv("records labelled", f"{phase_a.X.shape[0]:,}",
                   hint="retrieved at least twice"))
        print(C.kv("labelled useful", f"{phase_a.y.mean():.1%}"))
        print(C.kv("elapsed", f"{time.time() - t0:.1f}s"))

    report = runner.estimator.fit(phase_a.X, phase_a.y, phase_a.groups)
    runner.estimator_report = report
    if verbose:
        met = report.auc_mean >= 0.70
        print(C.section("3", "UTILITY ESTIMATOR   RQ1: can outcomes train it?"))
        print(C.kv("ROC-AUC", f"{report.auc_mean:.3f} +/- {report.auc_std:.3f}",
                   hint=f"target >= 0.70   {C.status(met)}"))
        print(C.kv("per fold", "  ".join(f"{a:.3f}" for a in report.auc_folds),
                   hint="one held-out conversation each"))
        print(C.kv("calibration",
                   f"Brier {report.brier:.4f}   avg precision {report.ap_mean:.3f}"))
        if baseline_auc is not None and report.auc_folds:
            print("")
            print(C.note("effect of the credit-label fix", "muted"))
            print(C.before_after("before the fix (recorded, earlier run)", baseline_auc,
                                 "after the fix", report.auc_mean,
                                 reference=("target", 0.70)))
        if report.auc_single_feature:
            best = max(report.auc_single_feature.items(), key=lambda kv: kv[1])
            print("")
            print(C.table(
                ["single-feature baseline", "ROC-AUC", "model gains"],
                [[name, f"{a:.3f}", f"{report.auc_mean - a:+.3f}"]
                 for name, a in sorted(report.auc_single_feature.items(),
                                       key=lambda kv: -kv[1])],
                aligns=["left", "right", "right"],
            ))
            margin = report.auc_mean - best[1]
            verdict = (f"adds {margin:+.3f} over" if margin >= 0
                       else f"is {abs(margin):.3f} BELOW")
            print(C.note(
                f"Ranking by {best[0]} alone reaches {best[1]:.3f}. The fitted "
                f"estimator {verdict} the best single feature, "
                "which is what the learned part of the signal is worth. "
                "Permutation importance favours lexical rarity, so this is "
                "the comparison that decides whether the model is more than "
                "an IDF heuristic. Scored on the same folds as the model, "
                "with each feature's direction fixed on the training fold.",
                "muted"))
        if report.feature_importance:
            top = sorted(report.feature_importance.items(),
                         key=lambda kv: -abs(kv[1]))[:6]
            print("")
            print(C.bar_chart([(k, abs(v)) for k, v in top],
                              title="what predicts usefulness (permutation importance)",
                              label_width=22, emphasise=top[0][0]))

    # Is utility a fixed property of a record, or does it depend on the query?
    # One estimator, scored against each category's own outcomes.
    per_cat = per_category_auc(
        phase_a.category_labels,
        oof_scores=report.oof_predictions,
        oof_mask=report.oof_mask,
    ) if report.oof_predictions is not None else {}
    if per_cat:
        usable = {c: v for c, v in per_cat.items() if v["auc"] == v["auc"]}
        print("")
        print(C.table(
            ["question category", "ROC-AUC", "vs pooled", "labelled rows",
             "useful rate"],
            [[c, f"{v['auc']:.3f}", f"{v['auc'] - report.auc_mean:+.3f}",
              f"{v['n']:,}", f"{v['positive_rate']:.1%}"]
             for c, v in sorted(usable.items(), key=lambda kv: -kv[1]["auc"])]
            + [[c, "-", "-", f"{v['n']:,}", "too few rows"]
               for c, v in sorted(per_cat.items()) if v["auc"] != v["auc"]],
            aligns=["left", "right", "right", "right", "right"],
        ))
        if len(usable) > 1:
            lo = min(usable.items(), key=lambda kv: kv[1]["auc"])
            hi = max(usable.items(), key=lambda kv: kv[1]["auc"])
            print(C.note(
                f"Utility is not a fixed property of a record: the same "
                f"estimator ranks {hi[0]} at {hi[1]['auc']:.3f} and {lo[0]} at "
                f"{lo[1]['auc']:.3f}, a spread of "
                f"{hi[1]['auc'] - lo[1]['auc']:.3f}. Rows with no outcome for "
                "a category are dropped, not counted as negatives.", "muted"))

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
        print(C.section("4", "LABEL AUDIT   are the credited records really useful?"))
        print(C.note("Benchmark evidence annotations, held out entirely: never a "
                     "feature, never a training target.", "muted"))
        print("")
        print(C.kv("precision", f"{validation['precision']:.3f}",
                   hint="credited records that do contain annotated evidence"))
        print(C.kv("recall", f"{validation['recall']:.3f}",
                   hint="evidence-bearing records that were credited"))
        print(C.kv("Cohen's kappa", f"{validation['kappa']:.3f}",
                   hint="agreement corrected for chance"))
        if "estimator_auc_vs_evidence" in validation:
            print(C.kv("estimator vs evidence",
                       f"{validation['estimator_auc_vs_evidence']:.3f} ROC-AUC",
                       hint="ranks evidence above non-evidence"))

    # ---------------- threshold calibration on contested population --------
    policy = GovernancePolicy(cfg.governance, runner.estimator)
    runner.calibrate_thresholds(test, policy)
    if verbose:
        th = policy.thresholds
        print(C.section("5", "GOVERNANCE THRESHOLDS"))
        print(C.kv("admission cut", f"{th.write:.4f}",
                   hint=f"quantile {cfg.governance.write_quantile:.0%} of contested utility"))
        print(C.kv("forgetting cut", f"{th.forget:.4f}",
                   hint=f"quantile {cfg.governance.forget_quantile:.0%}"))
        print(C.kv("candidates contested", f"{th.n_contested:,}"))

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


def summarise(results: Dict[str, RunResult]) -> str:
    """The comparison block: cost beside accuracy, difference made explicit.

    Every condition is reported against the fixed top-k baseline rather than in
    isolation, because the question is never "how accurate is this" but "how
    accurate for what it spends". The delta columns exist so that the reader
    does not have to do the subtraction, which is where a presentation loses
    its audience.
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
    out.append(C.section("6", "CONDITIONS   what each governance policy costs"))
    ref_acc = naive.accuracy if isinstance(naive, RunResult) else None
    ref_tok = naive.prompt_tokens_per_query if isinstance(naive, RunResult) else None

    rows = []
    for name in order:
        r = present[name]
        apk = accuracy_per_1k_tokens(r.accuracy, r.prompt_tokens_per_query)
        if ref_acc is None or name == "naive_rag":
            # The baseline is marked, not given a delta against itself, so the
            # neutral dot stays available to mean "no measurable change".
            d_acc = C.paint("ref", "muted")
            d_tok = C.paint("ref", "muted")
        else:
            d_acc = C.delta(r.accuracy - ref_acc)
            d_tok = C.delta(
                (r.prompt_tokens_per_query - ref_tok) / ref_tok,
                higher_is_better=False, fmt="{:+.0%}",
            )
        rows.append([
            LABELS[name], f"{r.accuracy:.3f}", d_acc,
            f"{r.prompt_tokens_per_query:,.0f}", d_tok,
            f"{apk:.3f}", f"{r.records_retained:,}", f"{r.generative_calls:,}",
        ])

    out.append(C.table(
        ["condition", "acc", "vs RAG", "tok/q", "vs RAG", "acc/1k",
         "records", "gen calls"],
        rows,
        emphasis_row=order.index("ocmg") if "ocmg" in order else None,
    ))
    out.append("")
    out.append(C.legend([
        (C.G.up, "better than top-k", "good"),
        (C.G.down, "worse", "critical"),
        (C.G.flat, "no change", "muted"),
        ("ref", "the baseline", "muted"),
    ]))
    skipped = results.get("_generative_judge_skipped")
    if skipped:
        out.append("")
        out.append(C.note(
            f"Generative judge not run: {skipped}. The row is absent rather "
            "than filled with an ungoverned build, which would duplicate "
            "naive RAG exactly and report a price for no decisions.",
            "warning", glyph=C.G.warn))

    # ------------------------------------------------- efficiency bar chart
    # Accuracy per thousand tokens is the only measure that can share one
    # axis across conditions, because accuracy and token cost have different
    # units and plotting both would invent a relationship.
    out.append("")
    out.append(C.bar_chart(
        [
            (LABELS[n],
             accuracy_per_1k_tokens(present[n].accuracy,
                                    present[n].prompt_tokens_per_query),
             f"{present[n].generative_calls:,} calls"
             if present[n].generative_calls else "")
            for n in order
        ],
        title="accuracy per 1,000 prompt tokens   (higher is better)",
        emphasise=LABELS.get("ocmg"),
    ))

    if ocmg is None:
        return "\n".join(out)

    # ------------------------------------------------------------- targets
    out.append(C.section("7", "TARGETS"))
    target_rows = []
    notes = []
    if isinstance(full, RunResult):
        arr = accuracy_retention_ratio(ocmg.accuracy, full.accuracy)
        # The ratio only means "how much of the ceiling was kept" while full
        # context IS the ceiling. Where it is not - a small or synthetic set
        # where the long prompt distracts more than it informs - the ratio
        # goes above 1 and the target passes without the system having
        # retained anything. Reporting that as MET would be a false pass, so
        # the status is withheld and the reason printed.
        ceiling_holds = (not isinstance(naive, RunResult)
                         or full.accuracy >= naive.accuracy)
        if ceiling_holds:
            status, margin = (C.status(arr >= 0.95),
                              C.delta(arr - 0.95, fmt="{:+.3f}"))
        else:
            status, margin = C.paint(f"{C.G.warn} NO CEILING", "warning"), "-"
            notes.append(
                f"Accuracy retention is not interpretable in this run: full "
                f"context scored {full.accuracy:.3f}, below naive top-k at "
                f"{naive.accuracy:.3f}, so it is not an upper bound and the "
                f"ratio of {arr:.3f} measures nothing. The long prompt is "
                "distracting the reader more than it informs it."
            )
        target_rows.append([
            "Accuracy retention vs full context", f"{arr:.3f}", ">= 0.95",
            status, margin,
        ])
    if isinstance(naive, RunResult):
        trr = token_reduction_ratio(
            ocmg.prompt_tokens_per_query, naive.prompt_tokens_per_query
        )
        target_rows.append([
            "Token reduction vs fixed top-k", f"{trr:.3f}", ">= 0.40",
            C.status(trr >= 0.40), C.delta(trr - 0.40, fmt="{:+.3f}"),
        ])
    out.append(C.table(
        ["measure", "value", "target", "status", "margin"],
        target_rows,
        aligns=["left", "right", "right", "left", "right"],
    ))
    for note in notes:
        out.append(C.note(note, "warning", glyph=C.G.warn))

    # -------------------------------------------------------- significance
    out.append(C.section("8", "SIGNIFICANCE   paired bootstrap over questions"))
    sig_rows = []
    for other in ("naive_rag", "append_all", "decay", "generative_judge"):
        o = results.get(other)
        if not isinstance(o, RunResult) or len(o.per_question_correct) != len(
            ocmg.per_question_correct
        ):
            continue
        bs = paired_bootstrap(ocmg.per_question_correct, o.per_question_correct)
        crosses_zero = bs["ci_low"] <= 0.0 <= bs["ci_high"]
        if crosses_zero:
            verdict = C.paint(f"{C.G.flat} indistinguishable", "warning")
        elif bs["delta"] > 0:
            verdict = C.paint(f"{C.G.ok} OCMG better", "good")
        else:
            verdict = C.paint(f"{C.G.bad} OCMG worse", "critical")
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
        out.append(C.note(
            "Retention precision not reported: the eviction signal is "
            "realised utility and this metric is the gap in realised utility, "
            "so it would score the sort against its own key. Section 4 tests "
            "the same claim against the held-out evidence annotations.",
            "muted"))
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
    return "\n".join(out)


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
