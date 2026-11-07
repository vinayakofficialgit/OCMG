#!/usr/bin/env python3
"""Iso-cost accuracy frontier and ablation study.

Why this script exists. A single operating point cannot settle whether one
governance policy is better than another, because the policies spend different
numbers of tokens to reach their accuracy. A condition that retrieves twice as
much and scores one point higher has not won anything. The honest comparison
sweeps the retrieval budget for every condition, traces accuracy against token
cost, and reads off the accuracy each achieves at matched spend.

    python scripts/sweep_budget.py --dataset locomo --path data/locomo10.json
    python scripts/sweep_budget.py --dataset locomo --path data/locomo10.json --ablations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Thread pinning must happen before numpy and sklearn are imported, so it sits
# above the package imports rather than inside main().
from ocmg.runtime import pin_threads  # noqa: E402
pin_threads()

from ocmg import console as C                                     # noqa: E402
from ocmg.config import OCMGConfig                                # noqa: E402
from ocmg.data.benchmarks import load_dataset                     # noqa: E402
from ocmg.experiment import DecayPolicy, split_conversations      # noqa: E402
from ocmg.governance import GovernancePolicy                      # noqa: E402
from ocmg.metrics import (evidence_retention, interpolate_at_cost,  # noqa: E402
                          iso_cost_comparison, pareto_frontier)
from ocmg.orchestrator import OCMGRunner                          # noqa: E402

CURVE_LABELS = {
    "naive_rag": "Naive RAG (top-k)",
    "append_all": "Append-all typed",
    "decay": "Decay-based",
    "ocmg": "OCMG",
}


def sweep(args) -> Dict[str, List[Tuple[float, float]]]:
    """Trace (tokens/query, accuracy) for each condition across budgets."""
    data = load_dataset(
        args.dataset, args.path, n_conversations=args.conversations,
        seed=args.seed, n_sessions=args.sessions, turns_per_session=args.turns,
    )
    cfg = OCMGConfig(seed=args.seed)
    cfg.embedder = args.embedder
    if getattr(args, "max_segment_turns", None):
        cfg.ingest.max_segment_turns = args.max_segment_turns
    if getattr(args, "boundary_depth", None) is not None:
        cfg.ingest.boundary_depth = args.boundary_depth
    if getattr(args, "write_quantile", None) is not None:
        cfg.governance.write_quantile = args.write_quantile
    runner = OCMGRunner(cfg)
    warm, test = split_conversations(data, 0.5, cfg.seed)

    print(C.section("1", "WARM-UP AND ESTIMATOR"))
    phase_a = runner.run_phase_a(warm)
    report = runner.estimator.fit(phase_a.X, phase_a.y, phase_a.groups)
    print(C.kv("ROC-AUC", f"{report.auc_mean:.3f} +/- {report.auc_std:.3f}",
               hint=f"on {report.n_samples:,} labelled records"))

    policy = GovernancePolicy(cfg.governance, runner.estimator)
    runner.calibrate_thresholds(test, policy)

    curves: Dict[str, List[Tuple[float, float]]] = {}

    print(C.section("2", "BUDGET SWEEP"))
    print(C.note(
        "Each condition is run across a range of retrieval budgets, so the "
        "comparison can be made at matched token cost instead of at whichever "
        "operating point each policy happens to land on.", "muted"))
    print("")

    # Baselines: sweep fixed k directly.
    for name, gov in (("naive_rag", None), ("append_all", "plain"),
                      ("decay", "decay")):
        pts = []
        for k in args.budgets:
            if gov is None:
                p = None
            elif gov == "decay":
                p = DecayPolicy(cfg.governance, None)
            else:
                p = GovernancePolicy(cfg.governance, None)
            # naive_rag is the untyped store; the other two keep the typing.
            store, _ = runner.build_store(
                test, p, governed=False, flat=(name == "naive_rag"))
            r = runner.evaluate(
                test, name, store, k=k, use_utility=False, adaptive=False,
                policy=p if gov == "decay" else None,
            )
            pts.append((r.prompt_tokens_per_query, r.accuracy))
            print(C.kv(f"{CURVE_LABELS[name]}  k={k}",
                       f"{r.prompt_tokens_per_query:7.1f} tok   acc {r.accuracy:.3f}",
                       width=30))
        curves[name] = pts

    # OCMG: sweep the relative-quality floor together with the ceiling. The
    # floor alone cannot span the budget range the baselines occupy, because k
    # is clipped to k_max whatever the floor is; sweeping both traces a curve
    # over the same token range and makes the comparison iso-cost.
    pts = []
    for floor, kmax in zip(args.floors, args.kmax):
        cfg.governance.budget_mass_target = floor
        cfg.retrieval.k_max = kmax
        runner.cfg.retrieval.k_max = kmax
        store, _ = runner.build_store(test, policy, governed=True)
        r = runner.evaluate(
            test, "ocmg", store, k=None, use_utility=True, adaptive=True, policy=policy
        )
        pts.append((r.prompt_tokens_per_query, r.accuracy))
        print(C.kv(f"OCMG  floor={floor} kmax={kmax}",
                   f"{r.prompt_tokens_per_query:7.1f} tok   acc {r.accuracy:.3f}",
                   width=30))
    curves["ocmg"] = pts
    return curves


def _overlap_grid(
    curves: Dict[str, List[Tuple[float, float]]], n: int = 5
) -> List[float]:
    """Token budgets every curve actually covers.

    Derived from the curves rather than fixed, because the budget range
    depends entirely on the dataset: LoCoMo records run an order of magnitude
    longer than synthetic ones, and a hardcoded grid would fall outside every
    curve and print a table of dashes.
    """
    if not curves:
        return []
    lo = max(min(c for c, _ in curve) for curve in curves.values() if curve)
    hi = min(max(c for c, _ in curve) for curve in curves.values() if curve)
    if hi <= lo:
        return []
    return [float(x) for x in np.linspace(lo, hi, n)]


def report_frontier(
    curves: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, float]:
    """Print one seed's iso-cost comparison and return its gaps by baseline."""
    ocmg = curves.get("ocmg", [])
    gaps: Dict[str, float] = {}

    print(C.section("3", "ISO-COST COMPARISON   accuracy at matched token cost"))
    rows = []
    for other in ("naive_rag", "append_all", "decay"):
        if other not in curves:
            continue
        cmp = iso_cost_comparison(ocmg, curves[other])
        gap = cmp["mean_gap"]
        if cmp["n_points"]:
            gaps[other] = gap
        if cmp["n_points"] == 0:
            verdict = C.paint(f"{C.G.flat} budget ranges do not overlap", "warning")
        elif gap > 0:
            verdict = C.paint(f"{C.G.ok} OCMG ahead", "good")
        else:
            verdict = C.paint(f"{C.G.bad} baseline ahead", "critical")
        rows.append([
            f"OCMG vs {CURVE_LABELS[other]}",
            f"{gap:+.4f}",
            f"{cmp.get('min_cost', 0):.0f}-{cmp.get('max_cost', 0):.0f}",
            str(cmp["n_points"]),
            verdict,
        ])
    if rows:
        print(C.table(
            ["comparison", "mean acc gap", "tok range", "points", "verdict"],
            rows,
            aligns=["left", "right", "right", "right", "left"],
        ))

    grid = _overlap_grid(curves)
    if grid:
        print("")
        print(C.note("accuracy at matched token budgets", "muted"))
        header = ["condition"] + [f"{g:.0f}" for g in grid]
        table_rows = []
        order = [n for n in CURVE_LABELS if n in curves]
        for name in order:
            curve = curves[name]
            row = [CURVE_LABELS[name]]
            lo, hi = min(c for c, _ in curve), max(c for c, _ in curve)
            for g in grid:
                v = interpolate_at_cost(curve, g)
                row.append(f"{v:.3f}" if (v is not None and lo <= g <= hi) else "-")
            table_rows.append(row)
        print(C.table(
            header, table_rows,
            emphasis_row=order.index("ocmg") if "ocmg" in order else None,
        ))

        # The mean gap above averages the whole overlap range, which hides a
        # sign change: the deficit against the ungoverned store is largest at
        # small budgets and closes, or reverses, at large ones, because the
        # ungoverned store's accuracy FALLS as k grows while governance filters
        # some of the distractors that cause it. A single mean reports neither.
        ocmg_curve = curves.get("ocmg")
        if ocmg_curve:
            olo, ohi = (min(c for c, _ in ocmg_curve),
                        max(c for c, _ in ocmg_curve))
            gap_rows = []
            for name in order:
                if name == "ocmg":
                    continue
                curve = curves[name]
                blo, bhi = min(c for c, _ in curve), max(c for c, _ in curve)
                row = [f"vs {CURVE_LABELS[name]}"]
                for g in grid:
                    a = interpolate_at_cost(ocmg_curve, g)
                    b = interpolate_at_cost(curve, g)
                    if (a is None or b is None or not olo <= g <= ohi
                            or not blo <= g <= bhi):
                        row.append("-")
                    else:
                        row.append(C.delta(a - b, fmt="{:+.3f}"))
                gap_rows.append(row)
            if gap_rows:
                print("")
                print(C.note("OCMG minus each baseline, at the same budget",
                             "muted"))
                print(C.table(header[:1] + header[1:], gap_rows))
        print("")
        print(C.note(
            "Read down a column: at the same number of prompt tokens, which "
            "policy answers more questions correctly. This is the comparison a "
            "single operating point cannot make.", "muted"))

    print("")
    print(C.note("Pareto frontier   cheapest condition at each accuracy level", "muted"))
    allpts = [(c, a, n) for n, curve in curves.items() for c, a in curve]
    front = pareto_frontier([(c, a) for c, a, _ in allpts])
    for c, a in front:
        owner = next(n for cc, aa, n in allpts if cc == c and aa == a)
        print(C.kv(f"{c:,.0f} tokens", f"acc {a:.3f}",
                   width=18, hint=CURVE_LABELS.get(owner, owner)))

    return gaps

def sweep_admission(args) -> None:
    """Trace accuracy and evidence survival against the admission rate.

    The budget sweep varies how much is retrieved while holding admission
    fixed, which cannot find the binding constraint if the constraint is
    admission. This varies the write quantile at a fixed retrieval budget, so
    the only thing changing is how much the gate throws away, and reports the
    evidence ceiling beside the accuracy so the mechanism is visible rather
    than inferred.

    Quantile 0.0 is the append-all reference: the gate admits everything.
    """
    data = load_dataset(
        args.dataset, args.path, n_conversations=args.conversations,
        seed=args.seed, n_sessions=args.sessions, turns_per_session=args.turns,
    )
    print(C.section("5", "ADMISSION SWEEP   what the write gate costs"))
    print(C.note(
        "Retrieval budget held fixed; only the admission quantile varies. The "
        "evidence ceiling is the share of answerable questions whose "
        "supporting turn still exists in the store at all - a limit no "
        "retrieval budget can exceed.", "muted"))
    print("")

    cfg = OCMGConfig(seed=args.seed)
    cfg.embedder = args.embedder
    cfg.governance.budget_mass_target = args.mass_target
    runner = OCMGRunner(cfg)
    warm, test = split_conversations(data, 0.5, cfg.seed)
    pa = runner.run_phase_a(warm)
    report = runner.estimator.fit(pa.X, pa.y, pa.groups)
    print(C.kv("estimator ROC-AUC", f"{report.auc_mean:.3f}"))
    print("")

    rows = []
    for q in args.write_quantiles:
        cfg.governance.write_quantile = q
        policy = GovernancePolicy(cfg.governance, runner.estimator)
        runner.calibrate_thresholds(test, policy)
        store, counters = runner.build_store(test, policy, governed=True)
        r = runner.evaluate(
            test, "ocmg", store, k=None, use_utility=True, adaptive=True,
            policy=policy,
        )
        ev = evidence_retention(store, test)
        # Rejected by the gate, however that rejection is realised: discarded
        # outright, or demoted to the cold tier. Reporting only "discarded"
        # would read as 0% on every row once demotion is enabled, hiding the
        # very thing the sweep varies.
        gated = (
            counters["discarded"] + counters["demoted"]
        ) / max(1, counters["candidates"])
        rows.append([
            "append-all (0.00)" if q == 0.0 else f"{q:.2f}",
            f"{gated:.1%}",
            f"{ev['evidence_ceiling']:.1%}",
            f"{r.accuracy:.3f}",
            f"{r.prompt_tokens_per_query:,.0f}",
            f"{r.accuracy / (r.prompt_tokens_per_query / 1000.0):.3f}"
            if r.prompt_tokens_per_query else "-",
            f"{r.records_retained:,}",
        ])
        print(C.kv(f"write_quantile {q:.2f}",
                   f"acc {r.accuracy:.3f}   ceiling {ev['evidence_ceiling']:.1%}"
                   f"   {r.prompt_tokens_per_query:,.0f} tok",
                   width=24))

    print("")
    print(C.table(
        ["write quantile", "gated", "evidence ceiling", "acc", "tok/q",
         "acc/1k", "hot records"],
        rows,
    ))
    print("")
    # Read the ceiling ACROSS rows, not against 100%. Demoted records stay
    # retrievable, so a gate that only demotes cannot lower the ceiling, and a
    # flat column is the evidence for that. Whatever shortfall from 100% is
    # shared by every row - including the row where the gate admits everything
    # - was caused by something else, which on this configuration is the
    # forgetting schedule. Blaming the gate for it would attribute the loss to
    # the wrong mechanism.
    print(C.note(
        "Read the ceiling column across rows, not against 100%. A flat column "
        "means the gate is destroying no evidence, since demoted records stay "
        "retrievable. Any shortfall shared by every row - including the "
        "admit-everything row - comes from forgetting instead, and a column "
        "that falls as the quantile rises is the gate losing records that no "
        "retrieval budget can recover.", "muted"))


def sweep_tiers(args) -> None:
    """Admission strength against cold-tier reachability, as a grid.

    These two parameters are not independent and cannot be tuned separately.
    The write quantile decides how much the gate rejects; the cold-tier
    penalty decides whether a rejected record can still be retrieved. Varying
    either alone confounds the two questions "does the gate pick the right
    records" and "does rejecting a record cost anything", which is why the
    one-dimensional sweeps were inconclusive.

    At a penalty of 1.0 the tiers are indistinguishable to retrieval, so the
    admission decision cannot affect any prompt. Accuracy that is flat along
    the quantile axis at that penalty is therefore direct evidence that the
    gate contributes nothing of its own.

    A naive top-k curve is traced alongside so each cell can be read against
    the baseline at matched token cost, which is the only fair comparison.

    Every cell is averaged over whatever seeds were asked for. The grid used
    to run at args.seed alone, and args.seed is left holding the last seed of
    the frontier loop, so "--seeds 0 1 2 --tiers" silently ran the grid at
    seed 2 under a banner that said seed 0.
    """
    seeds = args.seeds or [args.seed]
    n_cells = len(args.cold_penalties) * len(args.write_quantiles)
    print(C.section("6", "TIER GRID   admission strength x reachability"))
    print(C.note(
        "The penalty is the price retrieval charges a demoted record. At 1.00 "
        "the tiers are identical to retrieval, so the admission gate cannot "
        "influence any prompt.", "muted"))
    if len(seeds) == 1:
        print(C.note(
            f"ONE SEED ({seeds[0]}). Every one of the {n_cells} cells is a "
            "full pipeline fit, so three seeds cost three times the wall "
            "clock. Read the ordering of nearby cells as noise until this is "
            "re-run with --seeds 0 1 2.", "warning", glyph=C.G.warn))
    print("")

    cells = {}
    for sd in seeds:
        if len(seeds) > 1:
            print(C.section("", f"SEED {sd}"))
        warm, test = split_conversations(
            load_dataset(args.dataset, args.path,
                         n_conversations=args.conversations, seed=sd,
                         n_sessions=args.sessions,
                         turns_per_session=args.turns),
            0.5, sd,
        )

        # Baseline curve for matched-cost reading, traced once per seed. It
        # has to be per seed: the conversation split changes with the seed, so
        # one seed's curve is the wrong yardstick for another seed's cells.
        base_cfg = OCMGConfig(seed=sd)
        base_cfg.embedder = args.embedder
        base_runner = OCMGRunner(base_cfg)
        curve = []
        for k in (3, 5, 8, 10, 14):
            store, _ = base_runner.build_store(test, None, governed=False)
            r = base_runner.evaluate(test, "naive_rag", store, k=k,
                                     use_utility=False, adaptive=False)
            curve.append((r.prompt_tokens_per_query, r.accuracy))
        print(C.kv("naive top-k curve",
                   "  ".join(f"{t:.0f}tok/{a:.3f}" for t, a in curve),
                   width=22))
        print("")

        for pen in args.cold_penalties:
            for q in args.write_quantiles:
                cfg = OCMGConfig(seed=sd)
                cfg.embedder = args.embedder
                cfg.governance.budget_mass_target = args.mass_target
                cfg.governance.write_quantile = q
                cfg.governance.demote_instead_of_discard = True
                cfg.retrieval.cold_tier_penalty = pen

                runner = OCMGRunner(cfg)
                pa = runner.run_phase_a(warm)
                runner.estimator.fit(pa.X, pa.y, pa.groups)
                policy = GovernancePolicy(cfg.governance, runner.estimator)
                runner.calibrate_thresholds(test, policy)
                store, _ = runner.build_store(test, policy, governed=True)
                r = runner.evaluate(test, "ocmg", store, k=None,
                                    use_utility=True, adaptive=True,
                                    policy=policy)
                ev = evidence_retention(store, test)
                tok = float(r.prompt_tokens_per_query)
                base = interpolate_at_cost(curve, tok) or 0.0
                cells.setdefault((pen, q), []).append(
                    (r.accuracy, tok, r.accuracy - base,
                     ev["evidence_ceiling"], r.records_retained))
                print(C.kv(f"penalty {pen:.2f}  quantile {q:.2f}",
                           f"acc {r.accuracy:.3f}   {tok:,.0f} tok   "
                           f"vs top-k {r.accuracy - base:+.3f}", width=30))
        print("")

    rows, gaps_mean = [], []
    for pen in args.cold_penalties:
        for q in args.write_quantiles:
            got = cells.get((pen, q))
            if not got:
                continue
            acc = [c[0] for c in got]
            tok = [c[1] for c in got]
            gap = [c[2] for c in got]
            gaps_mean.append((float(np.mean(gap)), pen, q))
            rows.append([
                f"{pen:.2f}", f"{q:.2f}",
                f"{float(np.mean([c[3] for c in got])):.1%}",
                f"{float(np.mean(acc)):.3f}",
                f"{float(np.std(acc)):.3f}" if len(seeds) > 1 else "-",
                f"{float(np.mean(tok)):,.0f}",
                f"{float(np.mean(acc)) / (float(np.mean(tok)) / 1000.0):.3f}"
                if float(np.mean(tok)) else "-",
                f"{float(np.mean([c[4] for c in got])):,.0f}",
                C.delta(float(np.mean(gap)), fmt="{:+.3f}"),
                "  ".join(f"{g:+.3f}" for g in gap) if len(seeds) > 1 else "-",
            ])

    print(C.table(
        ["cold penalty", "write quantile", "ceiling", "acc", "sd", "tok/q",
         "acc/1k", "hot records", "vs top-k", "gap per seed"],
        rows,
    ))
    print("")
    print(C.note(
        "Read along the quantile axis at each penalty. Flat means the gate is "
        "doing nothing; falling means it is doing harm. The 'vs top-k' column "
        "is the only comparison that controls for token cost.", "muted"))
    # The grid exists to answer one question, so it should answer it rather
    # than leaving the reader to scan eighteen rows for a positive number.
    if gaps_mean:
        best, bpen, bq = max(gaps_mean)
        if best <= 0:
            print(C.note(
                f"NO CELL BEATS THE BASELINE AT MATCHED COST. Best of "
                f"{len(gaps_mean)} cells is penalty {bpen:.2f} / quantile "
                f"{bq:.2f} at {best:+.3f}. The admission gate does not pay "
                "for itself at any reachability setting tested.",
                "critical", glyph=C.G.bad))
        else:
            print(C.note(
                f"Best cell: penalty {bpen:.2f} / quantile {bq:.2f} at "
                f"{best:+.3f} against naive top-k at matched cost.",
                "good", glyph=C.G.ok))


def run_ablations(args) -> None:
    """Configurations isolating each governance component.

    The credit variants are the methodologically important ones. Leave-one-out
    re-execution is the only rule that isolates a record's marginal
    contribution, and it costs |R| extra reader calls per warm-up query. The
    rank-weighted and uniform rules are what a deployed system can afford.
    Running all three answers what the expensive rule actually buys, rather
    than assuming the cheap approximation is adequate.
    """
    data = load_dataset(
        args.dataset, args.path, n_conversations=args.conversations,
        seed=args.seed, n_sessions=args.sessions, turns_per_session=args.turns,
    )
    print(C.section("4", "ABLATION STUDY"))

    variants = [
        ("full", {}),
        ("no admission gate", {"write_quantile": 0.0}),
        ("forgetting on (0.25)", {"forget_quantile": 0.25}),
        ("no adaptive budget", {"fixed_k": True}),
        ("flat store (no types)", {"flat": True}),
        ("rank-weighted credit", {"credit": "rank_weighted", "no_loo": True}),
        ("uniform credit", {"credit": "uniform", "no_loo": True}),
    ]

    rows = []
    for label, mods in variants:
        cfg = OCMGConfig(seed=args.seed)
        cfg.embedder = args.embedder
        if "write_quantile" in mods:
            cfg.governance.write_quantile = mods["write_quantile"]
        if "forget_quantile" in mods:
            cfg.governance.forget_quantile = mods["forget_quantile"]
        if "credit" in mods:
            cfg.credit_scheme = mods["credit"]
        if mods.get("no_loo"):
            # Labels now come from the cheap co-occurrence rule instead of
            # counterfactual re-execution.
            cfg.exploration.loo_fraction = 0.0
        if mods.get("flat"):
            cfg.ingest.flat_store = True

        runner = OCMGRunner(cfg)
        warm, test = split_conversations(data, 0.5, cfg.seed)
        pa = runner.run_phase_a(warm)
        report = runner.estimator.fit(pa.X, pa.y, pa.groups)
        policy = GovernancePolicy(cfg.governance, runner.estimator)
        runner.calibrate_thresholds(test, policy)
        store, _ = runner.build_store(test, policy, governed=True)
        r = runner.evaluate(
            test, label, store, k=10 if mods.get("fixed_k") else None,
            use_utility=True, adaptive=not mods.get("fixed_k"), policy=policy,
        )
        acc1k = (
            r.accuracy / (r.prompt_tokens_per_query / 1000.0)
            if r.prompt_tokens_per_query else 0.0
        )
        rows.append([
            label, f"{report.auc_mean:.3f}", f"{r.accuracy:.3f}",
            f"{r.prompt_tokens_per_query:,.0f}", f"{acc1k:.2f}",
            f"{r.records_retained:,}",
        ])
        print(C.kv(label, f"acc {r.accuracy:.3f}   AUC {report.auc_mean:.3f}",
                   width=24))

    print("")
    print(C.table(
        ["configuration", "est AUC", "acc", "tok/q", "acc/1k", "records"],
        rows,
        emphasis_row=0,
    ))
    print("")
    print(C.note(
        "Estimator AUC is shown per configuration because the credit variants "
        "change the training labels, not the governance rule: a drop there is "
        "the cost of replacing counterfactual re-execution with a cheap "
        "co-occurrence approximation.", "muted"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--path", default=None)
    ap.add_argument("--conversations", type=int, default=40)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run the iso-cost comparison on several seeds and "
                         "report the mean gap; overrides --seed")
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 5, 8, 10, 14])
    ap.add_argument("--floors", type=float, nargs="+",
                    default=[0.98, 0.90, 0.75, 0.55, 0.30, 0.10])
    ap.add_argument("--kmax", type=int, nargs="+",
                    default=[2, 3, 5, 8, 12, 16])
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--admission", action="store_true",
                    help="sweep the write quantile at a fixed retrieval budget")
    ap.add_argument("--tiers", action="store_true",
                    help="grid the write quantile against the cold-tier penalty")
    ap.add_argument("--cold-penalties", type=float, nargs="+",
                    default=[0.6, 0.8, 1.0])
    ap.add_argument("--write-quantiles", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.20, 0.35, 0.50])
    ap.add_argument("--write-quantile", type=float, default=None,
                    help="admission fraction for the OCMG curve; 0 removes the "
                         "gate and leaves utility reranking and adaptive k")
    ap.add_argument("--boundary-depth", type=float, default=None,
                    help="segmenter sensitivity; lower splits more. 0.05 gives "
                         "1.56 turns per record against the naive chunker's 3.75")
    ap.add_argument("--max-segment-turns", type=int, default=None,
                    help="segment granularity; the naive baseline chunks at 4")
    ap.add_argument("--mass-target", type=float, default=0.80,
                    help="adaptive-budget floor; matches run_experiment.py")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip the retrieval-budget sweep")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    C.refresh_glyphs()
    if args.no_color:
        C.set_color(False)

    print(C.banner(
        "OCMG  -  iso-cost frontier and ablations",
        f"{args.dataset}  |  embedder {args.embedder}  |  seed {args.seed}",
    ))
    if args.embedder != "sentence-transformers":
        print(C.note(
            "PIPELINE VALIDATION.  Simulated reader and/or hashing embedder: "
            "these numbers test the machinery, they are not findings.",
            "warning", glyph=C.G.warn))

    curves = {}
    if not args.skip_sweep:
        seeds = args.seeds or [args.seed]
        per_seed, per_seed_curves = {}, {}
        if len(seeds) == 1:
            args.seed = seeds[0]
            curves = sweep(args)
            per_seed[seeds[0]] = report_frontier(curves)
            per_seed_curves[seeds[0]] = curves
        else:
            # The iso-cost gap is the number the thesis rests on, and a single
            # seed of it has already reversed once in this project, so each
            # seed's frontier is reported and then the gaps are averaged.
            for sd in seeds:
                print(C.section("", f"SEED {sd}"))
                args.seed = sd
                curves = sweep(args)
                per_seed_curves[sd] = curves
                per_seed[sd] = report_frontier(curves)
            print(C.section("", f"ISO-COST GAP OVER {len(seeds)} SEEDS"))
            names = [n for n in per_seed[seeds[0]]] if per_seed else []
            rows = []
            for name in names:
                gaps = [per_seed[sd][name] for sd in seeds
                        if per_seed[sd].get(name) is not None]
                if not gaps:
                    continue
                mean = float(np.mean(gaps))
                rows.append([
                    CURVE_LABELS.get(name, name),
                    f"{mean:+.4f}", f"{float(np.std(gaps)):.4f}",
                    "  ".join(f"{g:+.3f}" for g in gaps),
                    C.paint(f"{C.G.ok} OCMG ahead", "good") if mean > 0
                    else C.paint(f"{C.G.bad} baseline ahead", "critical"),
                ])
            print(C.table(
                ["OCMG vs", "mean gap", "sd", "per seed", "verdict"], rows,
                aligns=["left", "right", "right", "right", "left"],
            ))
        # The loop above assigns args.seed on every iteration, so without this
        # every sweep that runs afterwards inherits the LAST seed while the
        # banner still names the first. The sweeps read args.seeds themselves
        # now, but a stale args.seed is still the wrong default for them.
        args.seed = seeds[0]
    if args.ablations:
        run_ablations(args)
    if args.admission or args.write_quantiles:
        sweep_admission(args)
    # --cold-penalties feeds sweep_tiers, so passing it alone used to parse
    # fine and run nothing: the script printed its banner and exited, which
    # looks like a sweep that found nothing rather than one that never ran.
    if args.tiers or args.cold_penalties:
        sweep_tiers(args)
    if (args.skip_sweep and not any((args.ablations, args.admission,
                                     args.write_quantiles, args.tiers,
                                     args.cold_penalties))):
        print(C.note(
            "--skip-sweep was given and no other sweep was selected, so "
            "there was nothing to run. Add --ablations, --admission or "
            "--tiers.", "warning", glyph=C.G.warn))
    if args.out and curves:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Every seed's curves, plus the settings that produced them. Writing
        # only the last seed's curves - which this did until 2026-11-01 - left
        # a file that looked like the whole run and was one seed, with nothing
        # recorded to say which seed or at what granularity.
        meta = {
            "dataset": args.dataset,
            "seeds": seeds,
            "boundary_depth": getattr(args, "boundary_depth", None),
            "max_segment_turns": getattr(args, "max_segment_turns", None),
            "write_quantile": getattr(args, "write_quantile", None),
            "mass_target": args.mass_target,
            "budgets": args.budgets,
            "floors": args.floors,
            "kmax": args.kmax,
            "embedder": args.embedder,
            "reportable": args.embedder == "sentence-transformers",
        }
        if len(seeds) > 1:
            payload = {
                "meta": meta,
                "per_seed": {str(sd): c for sd, c in per_seed_curves.items()},
                "iso_cost_gaps": {str(sd): g for sd, g in per_seed.items()},
            }
        else:
            payload = {"meta": meta, "per_seed": {str(seeds[0]): curves}}
        with open(args.out, "w", encoding="utf8") as fh:
            json.dump(payload, fh, indent=2)
        print(C.section("", "OUTPUT"))
        print(C.kv("curves written", args.out,
                   hint=f"{len(payload['per_seed'])} seed(s), with settings"))
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
