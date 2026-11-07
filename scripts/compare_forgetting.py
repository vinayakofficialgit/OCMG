#!/usr/bin/env python3
"""Which signal should rank records for eviction?

The estimator (ROC-AUC about 0.73) governed admission, where the iso-cost and
tier experiments showed it cannot help, while forgetting ranked records by
realised ledger utility - which the credit ablation measured at chance level,
because the deployed rank-weighted rule credits every retrieved record equally.
This compares putting each signal on the forgetting decision.

    python scripts/compare_forgetting.py --dataset locomo --path data/locomo10.json

Cost note. Warm-up and the fitted estimator depend only on the seed, not on the
forgetting signal, so they are computed once per seed and reused across all
arms. Recomputing them per arm - which an earlier version of this comparison
did - quadrupled the runtime and could not change any result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocmg.runtime import pin_threads  # noqa: E402
pin_threads()

import numpy as np  # noqa: E402

from ocmg import console as C                                  # noqa: E402
from ocmg.config import OCMGConfig                             # noqa: E402
from ocmg.data.benchmarks import load_dataset                  # noqa: E402
from ocmg.experiment import DecayPolicy, split_conversations   # noqa: E402
from ocmg.governance import GovernancePolicy                   # noqa: E402
from ocmg.metrics import evidence_retention, paired_bootstrap  # noqa: E402
from ocmg.orchestrator import OCMGRunner                       # noqa: E402

ARMS = {
    "ledger": "realised utility (current default)",
    "estimator": "estimator prediction",
    "blend": "realised, shrunk toward the prior",
    "recency": "the decay criterion, same pipeline",
    "none": "forgetting disabled",
}

DEFAULT_QUANTILE = 0.25


def arm_plan(args) -> list:
    """One (label, signal, quantile) triple per condition to run.

    Signal mode varies which score ranks records for eviction at a fixed
    fraction. Quantile mode varies the fraction at a fixed signal, which is
    what distinguishes "this signal is wrong" from "any forgetting is wrong":
    if accuracy rises monotonically as the fraction falls to zero, the optimum
    is no forgetting and no ranking rule can rescue it.
    """
    if not args.quantiles:
        return [
            (a, "ledger" if a == "none" else a,
             0.0 if a == "none" else args.forget_quantile, "ocmg")
            for a in args.arms
        ]
    plan = []
    for q in args.quantiles:
        plan.append((f"q={q:.2f}", args.signal if q > 0.0 else "ledger",
                     q, "ocmg"))
        # The baseline reads decay_forget_quantile, set alongside the
        # governed fraction below so both arms evict at the same rate. Without it the sweep cannot
        # distinguish "forgetting this much is harmful" from "OCMG's signal is
        # worse than recency at forgetting this much" - the two curves answer
        # different questions and only the pair identifies the cause.
        if args.with_decay and q > 0.0:
            plan.append((f"decay q={q:.2f}", "ledger", q, "decay"))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--path", default="data/locomo10.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--mass-target", type=float, default=0.80)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--forget-quantile", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--quantiles", type=float, nargs="+", default=None,
                    help="sweep the eviction fraction instead of the signal")
    ap.add_argument("--signal", default="ledger", choices=list(ARMS)[:4],
                    help="signal held fixed while sweeping --quantiles")
    ap.add_argument("--with-decay", action="store_true",
                    help="also run the recency baseline at each fraction")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    C.refresh_glyphs()
    if args.no_color:
        C.set_color(False)

    print(C.banner(
        "OCMG  -  how much forgetting?" if args.quantiles
        else "OCMG  -  which signal should govern forgetting?",
        f"{args.dataset}  |  embedder {args.embedder}  |  "
        f"seeds {', '.join(str(s) for s in args.seeds)}"
        + (f"  |  signal {args.signal}, sweeping the fraction"
           if args.quantiles else ""),
    ))
    print(C.note(
        "PIPELINE VALIDATION.  Simulated reader and/or hashing embedder: "
        "these numbers test the machinery, they are not findings.",
        "warning", glyph=C.G.warn))

    data = load_dataset(args.dataset, args.path)
    plan = arm_plan(args)
    labels = [label for label, _, _, _ in plan]
    # Every other condition is tested against this one.
    baseline = (f"q={args.forget_quantile:.2f}" if args.quantiles
                else "ledger")
    if baseline not in labels:
        baseline = labels[0]
    per_arm: dict = {label: [] for label in labels}

    for seed in args.seeds:
        print(C.section("", f"SEED {seed}"))

        # Warm-up and estimator once per seed: they are identical across arms.
        cfg = OCMGConfig(seed=seed)
        cfg.embedder = args.embedder
        cfg.governance.budget_mass_target = args.mass_target
        runner = OCMGRunner(cfg)
        warm, test = split_conversations(data, 0.5, seed)
        pa = runner.run_phase_a(warm)
        report = runner.estimator.fit(pa.X, pa.y, pa.groups)
        print(C.kv("estimator ROC-AUC", f"{report.auc_mean:.3f}",
                   hint=f"{report.n_samples:,} labelled records, reused by every arm"))
        print("")

        for label, signal, quantile, kind in plan:
            # Mutating the shared config is what lets the fitted estimator be
            # reused; the policy reads these fields by reference.
            runner.cfg.governance.forget_signal = signal
            runner.cfg.governance.forget_quantile = quantile
            runner.cfg.governance.decay_forget_quantile = quantile

            if kind == "decay":
                # Matched fraction, unmatched criterion, and evaluated the way
                # the main experiment evaluates it: ungoverned admission and a
                # fixed budget, so the only difference from that table is the
                # eviction fraction being swept.
                policy = DecayPolicy(runner.cfg.governance, None)
                store, _ = runner.build_store(test, policy, governed=False)
                r = runner.evaluate(
                    test, "decay", store, runner.cfg.retrieval.k_fixed,
                    False, False, policy=policy,
                )
            else:
                policy = GovernancePolicy(runner.cfg.governance,
                                          runner.estimator)
                runner.calibrate_thresholds(test, policy)
                store, _ = runner.build_store(test, policy, governed=True)
                r = runner.evaluate(test, "ocmg", store, k=None,
                                    use_utility=True, adaptive=True,
                                    policy=policy)
            ev = evidence_retention(store, test)
            tok = r.prompt_tokens_per_query
            per_arm[label].append({
                "seed": seed,
                "accuracy": r.accuracy,
                "tokens": tok,
                "acc_per_1k": r.accuracy / (tok / 1000.0) if tok else 0.0,
                "ceiling": ev["evidence_ceiling"],
                "hot": r.records_retained,
                "evicted": policy.stats["evicted"],
                "correct": list(r.per_question_correct),
            })
            print(C.kv(label, f"acc {r.accuracy:.3f}   {tok:,.0f} tok   "
                            f"ceiling {ev['evidence_ceiling']:.1%}"
                            f"   evicted {policy.stats['evicted']:,}", width=12))

    # ------------------------------------------------------------- summary
    print(C.section("", "MEAN OVER SEEDS"))
    rows = []
    for arm in labels:
        runs = per_arm[arm]
        acc = np.array([x["accuracy"] for x in runs])
        tok = np.array([x["tokens"] for x in runs])
        a1k = np.array([x["acc_per_1k"] for x in runs])
        ceil = np.array([x["ceiling"] for x in runs])
        rows.append([
            arm, f"{acc.mean():.3f}", f"{acc.std():.3f}", f"{tok.mean():,.0f}",
            f"{a1k.mean():.3f}", f"{ceil.mean():.1%}",
            f"{np.mean([x['evicted'] for x in runs]):,.0f}",
        ])
    print(C.table(
        ["signal", "acc mean", "acc sd", "tok/q", "acc/1k", "ceiling", "evicted"],
        rows,
    ))

    if len(labels) > 1:
        print("")
        print(C.note(f"paired bootstrap against {baseline}, per seed", "muted"))
        sig_rows = []
        for arm in labels:
            if arm == baseline:
                continue
            for i, run in enumerate(per_arm[arm]):
                base = per_arm[baseline][i]
                bs = paired_bootstrap(run["correct"], base["correct"])
                crosses = bs["ci_low"] <= 0.0 <= bs["ci_high"]
                verdict = (C.paint(f"{C.G.flat} indistinguishable", "warning")
                           if crosses else
                           C.paint(f"{C.G.ok} better", "good") if bs["delta"] > 0
                           else C.paint(f"{C.G.bad} worse", "critical"))
                sig_rows.append([
                    f"{arm} vs {baseline}", str(run["seed"]), f"{bs['delta']:+.3f}",
                    f"[{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]",
                    f"{bs['p_value']:.3f}", verdict,
                ])
        print(C.table(
            ["comparison", "seed", "delta", "95% CI", "p", "verdict"],
            sig_rows,
            aligns=["left", "right", "right", "right", "right", "left"],
        ))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            arm: [{k: v for k, v in run.items() if k != "correct"} for run in runs]
            for arm, runs in per_arm.items()
        }
        with open(args.out, "w", encoding="utf8") as fh:
            json.dump(payload, fh, indent=2)
        print(C.section("", "OUTPUT"))
        print(C.kv("written", args.out))
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
