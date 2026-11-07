#!/usr/bin/env python3
"""Run the full OCMG experiment: baselines, OCMG, ablations.

    python scripts/run_experiment.py --conversations 24 --runs 3
    python scripts/run_experiment.py --dataset locomo --path data/locomo10.json
    python scripts/run_experiment.py --dataset locomo --path data/locomo10.json \\
        --embedder sentence-transformers --reader hf --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Thread pinning must happen before numpy and sklearn are imported, so it sits
# above the package imports rather than inside main().
from ocmg.runtime import pin_threads, threads_for_argv  # noqa: E402
pin_threads(threads_for_argv(sys.argv))

from ocmg import console as C                            # noqa: E402
from ocmg.config import OCMGConfig                       # noqa: E402
from ocmg.data.benchmarks import load_dataset            # noqa: E402
from ocmg.experiment import (PRE_FIX_ESTIMATOR_AUC, LABELS,  # noqa: E402
                             run_all_conditions, subsample_questions,
                             summarise)
from ocmg.reporting import aggregate_runs, run_to_dict  # noqa: E402


def build_config(args) -> OCMGConfig:
    cfg = OCMGConfig(seed=args.seed)
    cfg.embedder = args.embedder
    cfg.reader.kind = args.reader
    cfg.estimator.model = args.estimator
    cfg.retrieval.k_fixed = args.k
    cfg.governance.write_quantile = args.write_quantile
    cfg.governance.forget_quantile = args.forget_quantile
    cfg.governance.decay_forget_quantile = args.decay_forget_quantile
    cfg.governance.budget_mass_target = args.mass_target
    if args.utility_weight is not None:
        cfg.retrieval.utility_weight = args.utility_weight
    if args.conditional_utility:
        cfg.estimator.conditional_utility = True
    if args.max_segment_turns:
        cfg.ingest.max_segment_turns = args.max_segment_turns
    if args.boundary_depth is not None:
        cfg.ingest.boundary_depth = args.boundary_depth
    cfg.retrieval.budget_by_query_type = not args.no_query_type
    cfg.credit_scheme = args.credit
    cfg.reader.hf_model = args.hf_model
    cfg.reader.max_new_tokens = args.max_new_tokens
    cfg.reader.batch_size = args.batch_size
    cfg.reader.device = args.device
    cfg.reader.dtype = args.dtype
    cfg.exploration.loo_fraction = args.loo_fraction
    cfg.exploration.loo_max_records = args.loo_max_records
    if args.epsilon is not None:
        cfg.exploration.epsilon_random_inclusion = args.epsilon
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="OCMG experiment runner")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "locomo", "longmemeval"])
    ap.add_argument("--path", default=None, help="path to benchmark json")
    ap.add_argument("--conversations", type=int, default=24)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--facts", type=int, default=10)
    ap.add_argument("--runs", type=int, default=1, help="repeat with different seeds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default="hashing",
                    choices=["auto", "hashing", "sentence-transformers"])
    ap.add_argument("--reader", default="simulated", choices=["simulated", "hf"])
    ap.add_argument("--hf-model", default=OCMGConfig().reader.hf_model,
                    help="backbone for --reader hf")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts generated together; leave-one-out fills it")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--epsilon", type=float, default=None,
                    help="warm-up random-inclusion rate. Setting 0 removes the "
                         "exploration that breaks retrieval-selection bias, "
                         "which is the first marker's objection: only records "
                         "the retriever already favoured ever get an outcome")
    ap.add_argument("--loo-fraction", type=float, default=1.0,
                    help="share of questions given counterfactual labelling")
    ap.add_argument("--loo-max-records", type=int, default=0,
                    help="cap counterfactuals per question; 0 keeps all. The "
                         "sample is uniform over the retrieved set, not the "
                         "top m, so the cost is variance rather than the "
                         "retrieval-selection bias a top-m cap would add")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of full_context,naive_rag,"
                         "append_all,decay,generative_judge,ocmg. Default is "
                         "all six. On a CPU reader two of them dominate the "
                         "wall clock: full_context sends a ~14k-token prompt "
                         "per question, and generative_judge spends one "
                         "generation per candidate record")
    ap.add_argument("--fast", action="store_true",
                    help="preset for a quick real-reader run: drops the two "
                         "arms above, caps counterfactuals at 4, and cuts "
                         "max_new_tokens to 8. Changes what is measured - it "
                         "is for timing and smoke-testing, not for results")
    ap.add_argument("--questions-per-conv", type=int, default=0,
                    help="stratified subsample per conversation; 0 keeps all")
    ap.add_argument("--estimator", default="gbdt",
                    choices=["gbdt", "logistic", "rarity"],
                    help="rarity ranks by rare_term_count alone, as an ablation")
    ap.add_argument("--credit", default="rank_weighted",
                    choices=["rank_weighted", "uniform", "top1"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--write-quantile", type=float, default=0.35)
    ap.add_argument("--utility-weight", type=float, default=None,
                    help="strength of utility reranking; 0 disables it")
    ap.add_argument("--conditional-utility", action="store_true",
                    help="fit U(m|q) instead of U(m): one training row per "
                         "observed (record, derived query type) cell, with "
                         "the query type appended to the record features")
    ap.add_argument("--forget-quantile", type=float,
                    default=OCMGConfig().governance.forget_quantile,
                    help="governed eviction fraction; 0.0 is the measured optimum")
    ap.add_argument("--decay-forget-quantile", type=float,
                    default=OCMGConfig().governance.decay_forget_quantile,
                    help="the decay baseline's fraction, kept separate")
    ap.add_argument("--boundary-depth", type=float, default=None,
                    help="segmenter sensitivity; lower splits more. 0.05 gives "
                         "1.56 turns per record against the naive chunker's 3.75")
    ap.add_argument("--max-segment-turns", type=int, default=None,
                    help="segment granularity; the naive baseline chunks at 4")
    ap.add_argument("--mass-target", type=float, default=0.80)
    ap.add_argument("--no-query-type", action="store_true",
                    help="ablate stage 4's query-type term from the budget")
    ap.add_argument("--out", default=None,
                    help="results json (default: results/run_<dataset>.json)")
    ap.add_argument("--no-color", action="store_true",
                    help="plain output, for redirecting into a file or a log")
    args = ap.parse_args()

    # Box-drawing characters and status glyphs need a UTF-8 stream; the Windows
    # console still defaults to a legacy code page in some shells.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    C.refresh_glyphs()
    if args.no_color:
        C.set_color(False)

    # The preset only fills in values the user did not set, so an explicit
    # flag always wins over it.
    if args.fast:
        if args.conditions is None:
            args.conditions = "naive_rag,append_all,decay,ocmg"
        if args.loo_max_records == 0:
            args.loo_max_records = 4
        if args.max_new_tokens == 24:
            args.max_new_tokens = 8

    conditions = ([c.strip() for c in args.conditions.split(",") if c.strip()]
                  if args.conditions else None)
    if conditions:
        known = {"full_context", "naive_rag", "append_all", "decay",
                 "generative_judge", "ocmg"}
        unknown = [c for c in conditions if c not in known]
        if unknown:
            ap.error(f"unknown condition(s): {', '.join(unknown)}. "
                     f"Choose from {', '.join(sorted(known))}.")

    reportable = args.reader == "hf" and args.embedder == "sentence-transformers"
    print(C.banner(
        "OCMG  -  Outcome-Calibrated Memory Governance",
        f"{args.dataset}  |  embedder {args.embedder}  |  reader {args.reader}"
        f"  |  {args.runs} seed(s) from {args.seed}",
    ))
    if not reportable:
        print(C.note(
            "PIPELINE VALIDATION.  Simulated reader and/or hashing embedder: "
            "these numbers test the machinery, they are not findings.",
            "warning", glyph=C.G.warn,
        ))
    if args.fast or conditions or args.loo_max_records:
        missing = ([c for c in ("full_context", "naive_rag", "append_all",
                                "decay", "generative_judge", "ocmg")
                    if c not in conditions] if conditions else [])
        parts = []
        if missing:
            parts.append(f"arms not run: {', '.join(missing)}")
        if args.loo_max_records:
            parts.append(f"counterfactuals capped at "
                         f"{args.loo_max_records} per question")
        if args.max_new_tokens != 24:
            parts.append(f"max_new_tokens {args.max_new_tokens}")
        print(C.note(
            "REDUCED RUN.  " + "; ".join(parts) + ".  A dropped arm removes "
            "its comparison entirely, and a capped counterfactual labels "
            "fewer records, so the estimator sees less. Timing and smoke "
            "tests only - a reduced run is not a result even with a real "
            "reader and a real embedder.",
            "critical", glyph=C.G.bad,
        ))

    all_runs = []
    for run in range(args.runs):
        seed = args.seed + run
        if args.runs > 1:
            print(C.section("", f"RUN {run + 1} of {args.runs}   seed {seed}"))

        data = load_dataset(
            args.dataset, args.path,
            n_conversations=args.conversations, seed=seed,
            n_sessions=args.sessions, turns_per_session=args.turns,
            n_facts=args.facts,
        )
        if args.questions_per_conv > 0:
            before = sum(len(q) for _, _, q in data)
            data = subsample_questions(data, args.questions_per_conv, seed)
            after = sum(len(q) for _, _, q in data)
            print(C.note(
                f"SUBSAMPLED.  {after:,} of {before:,} questions kept, "
                f"stratified by category: every interval below is wider "
                f"than a full-dataset run would give.",
                "warning", glyph=C.G.warn))
        cfg = build_config(args)
        cfg.seed = seed
        # The pre-fix figure is a LoCoMo seed-0 measurement, so it is only an
        # honest comparison for that run.
        baseline = (
            PRE_FIX_ESTIMATOR_AUC
            if (args.dataset == "locomo" and seed == 0 and not args.fast
                and args.reader == "hf" and args.embedder == "sentence-transformers")
            else None
        )
        results = run_all_conditions(data, cfg, conditions=conditions,
                                     baseline_auc=baseline)
        print(summarise(results))
        all_runs.append(run_to_dict(results, seed))

    aggregate = aggregate_runs(all_runs)
    if args.runs > 1:
        print(C.section("9", f"MEAN OVER {args.runs} SEEDS"))
        order = [n for n in LABELS if n in aggregate["conditions"]]
        print(C.table(
            ["condition", "acc mean", "acc sd", "tok/q"],
            [
                [
                    LABELS[n],
                    f"{aggregate['conditions'][n]['accuracy']['mean']:.3f}",
                    f"{aggregate['conditions'][n]['accuracy']['sd']:.3f}",
                    f"{aggregate['conditions'][n]['tokens_per_query']['mean']:,.0f}",
                ]
                for n in order
            ],
            emphasis_row=order.index("ocmg") if "ocmg" in order else None,
        ))
        # The same two guards summarise() applies per seed. Without them this
        # footer printed MET for a retention ratio measured against something
        # that was not a ceiling, and for a token reduction measured against a
        # baseline spending twice as much - the mean-over-seeds block is what
        # gets read, so a guard only in the per-seed block is no guard at all.
        conds = aggregate["conditions"]
        acc = {n: conds[n]["accuracy"]["mean"] for n in conds}
        tok = {n: conds[n]["tokens_per_query"]["mean"] for n in conds}
        no_ceiling = ("full_context" in acc and "naive_rag" in acc
                      and acc["full_context"] < acc["naive_rag"])
        cost_gap = (tok.get("naive_rag", 0) / tok["ocmg"]
                    if tok.get("ocmg") else 1.0)

        for key, label, target in (
            ("accuracy_retention", "Accuracy retention vs full context", 0.95),
            ("token_reduction", "Token reduction vs fixed top-k", 0.40),
            ("estimator_auc", "Utility estimator ROC-AUC", 0.70),
        ):
            t = aggregate.get("targets", {}).get(key)
            if not t:
                continue
            status = C.status(bool(t["met"]))
            if key == "accuracy_retention" and no_ceiling:
                status = C.paint(f"{C.G.warn} NO CEILING", "warning")
            print(C.kv(label, f"{t['mean']:.3f} +/- {t['sd']:.3f}", width=36,
                       hint=f">= {target:.2f}   {status}"))

        if no_ceiling:
            print(C.note(
                f"Accuracy retention is not interpretable: full context scored "
                f"{acc['full_context']:.3f}, below naive top-k at "
                f"{acc['naive_rag']:.3f}, so it is not an upper bound.",
                "warning", glyph=C.G.warn))
        if cost_gap > 1.25:
            print(C.note(
                f"Token reduction is measured against a baseline spending "
                f"{cost_gap:.1f}x as much ({tok['naive_rag']:,.0f} against "
                f"{tok['ocmg']:,.0f} tokens per query), so a pass here says "
                "the comparator is expensive, not that OCMG is efficient. "
                "Read the iso-cost sweep instead: accuracy at matched spend "
                "is the frame this evaluation is built on.",
                "warning", glyph=C.G.warn))

    out_path = args.out or os.path.join("results", f"run_{args.dataset}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "path": args.path,
            "embedder": args.embedder,
            "reader": args.reader,
            "estimator": args.estimator,
            "credit": args.credit,
            "k": args.k,
            "write_quantile": args.write_quantile,
            "forget_quantile": args.forget_quantile,
            "decay_forget_quantile": args.decay_forget_quantile,
            "mass_target": args.mass_target,
            "utility_weight": (args.utility_weight if args.utility_weight
                               is not None else OCMGConfig().retrieval.utility_weight),
            "conditional_utility": bool(args.conditional_utility),
            "budget_by_query_type": not args.no_query_type,
            "hf_model": args.hf_model if args.reader == "hf" else None,
            "max_new_tokens": args.max_new_tokens,
            "loo_fraction": args.loo_fraction,
            "epsilon_random_inclusion": (
                args.epsilon if args.epsilon is not None
                else OCMGConfig().exploration.epsilon_random_inclusion),
            "questions_per_conv": args.questions_per_conv,
            # A reduced run must be identifiable from the file alone, or a
            # timing probe gets read back later as a measurement.
            "conditions": conditions or "all",
            "loo_max_records": args.loo_max_records,
            "max_new_tokens": args.max_new_tokens,
            "reduced_run": bool(conditions or args.loo_max_records
                                or args.max_new_tokens != 24),
            "runs": args.runs,
            "seed": args.seed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            # Results from the simulated reader or hashing embedder validate
            # the pipeline only; they are not findings.
            "reportable": (args.reader == "hf"
                           and args.embedder == "sentence-transformers"
                           and not conditions
                           and not args.loo_max_records
                           and args.max_new_tokens == 24),
        },
        "runs": all_runs,
        "aggregate": aggregate,
    }
    with open(out_path, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2)
    print(C.section("", "OUTPUT"))
    print(C.kv("results written", out_path))
    reduced = bool(conditions or args.loo_max_records
                   or args.max_new_tokens != 24)
    print(C.kv("reportable", "yes" if reportable and not reduced else "no",
               hint=("reduced run: arms dropped, counterfactuals capped or "
                     "tokens cut" if reduced and reportable
                     else "" if reportable
                     else "needs --reader hf --embedder sentence-transformers")))
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
