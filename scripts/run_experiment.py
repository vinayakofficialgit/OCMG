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
from ocmg.metrics import paired_bootstrap                # noqa: E402
from ocmg.reporting import aggregate_runs, run_to_dict  # noqa: E402
from ocmg.types import RunResult                         # noqa: E402


def _versus(results, other: str):
    """OCMG minus `other` on the same questions: (delta, p, glyph)."""
    a, b = results.get("ocmg"), results.get(other)
    if not (isinstance(a, RunResult) and isinstance(b, RunResult)) or len(
            a.per_question_correct) != len(b.per_question_correct):
        return None
    bs = paired_bootstrap(a.per_question_correct, b.per_question_correct)
    if bs["ci_low"] <= 0.0 <= bs["ci_high"]:
        mark = "same"
    elif bs["delta"] > 0:
        mark = "better"
    else:
        mark = "worse"
    return bs["delta"], bs["p_value"], mark


_MARK = {"better": "\u2714", "worse": "\u2718", "same": "\u00b7"}


def _cell(v) -> str:
    if v is None:
        return "-"
    d, p, mark = v
    return f"{d:+.3f} p{p:.2f} {_MARK[mark]}"


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
    ap.add_argument("--detail", choices=["auto", "full", "brief"],
                    default="auto",
                    help="full = sections 1-9 for every seed; brief = one row "
                         "per seed plus the mean. auto = full for one seed, "
                         "brief for several. Every number is in the JSON "
                         "either way; --runs 1 --seed N shows one seed in full")
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
    # Read from the config, as --forget-quantile is. This was a literal 0.35
    # until 21 Nov 2026, which silently re-enabled the admission gate on every
    # main run after config.py had set the measured optimum of 0.0.
    ap.add_argument("--write-quantile", type=float,
                    default=OCMGConfig().governance.write_quantile)
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
    ap.add_argument("--save", default=None, metavar="FILE",
                    help="also write everything printed to FILE as UTF-8 "
                         "text; use this instead of | Tee-Object")
    ap.add_argument("--save-append", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--no-color", action="store_true",
                    help="plain output, for redirecting into a file or a log")
    args = ap.parse_args()

    # Box-drawing characters and status glyphs need a UTF-8 stream; the Windows
    # console still defaults to a legacy code page in some shells.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    C.save_to(args.save, append=args.save_append)
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
        # No max_new_tokens cut. It was in this preset until it was
        # measured, on the strength of the claim in ReaderConfig that
        # generation is sequential and therefore the strongest cost lever. At
        # real LoCoMo prompt lengths that is false: cutting 24 tokens to 8
        # while the prompt grew from 298 to ~1,100 tokens made the run 4.2x
        # SLOWER per generation, so cost tracks prefill, not decode. The cut
        # bought almost nothing and it zeroed the measurement - every arm
        # scored 0.000 because 8 tokens are spent on preamble before the
        # answer appears. See CLAUDE.md section 8 result E.

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
            "PIPELINE VALIDATION: simulated reader / hashing embedder, so "
            "these numbers test the code and are not thesis findings",
            "warning", glyph=C.G.warn,
        ))
    key = C.color_key()
    if key:
        print(key)
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

    detail = (args.detail if args.detail != "auto"
              else "full" if args.runs == 1 else "brief")
    all_runs = []
    seed_rows = []
    verdicts = {"naive_rag": [], "append_all": [], "decay": []}
    for run in range(args.runs):
        seed = args.seed + run
        if args.runs > 1 and detail == "full":
            print(C.section("", f"RUN {run + 1} of {args.runs}   seed {seed}"))
        elif detail == "brief":
            print(C.kv(f"seed {seed}", f"running ({run + 1} of {args.runs})",
                       hint="full detail for this seed: --runs 1 --seed "
                            f"{seed}"))

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
                                     baseline_auc=baseline,
                                     verbose=(detail == "full"))
        if detail == "full":
            print(summarise(results))
        all_runs.append(run_to_dict(results, seed))
        vs = {o: _versus(results, o) for o in verdicts}
        for o, v in vs.items():
            if v:
                verdicts[o].append(v[2])
        est = results.get("_estimator")
        acc = {n: results[n].accuracy for n in LABELS
               if isinstance(results.get(n), RunResult)}
        seed_rows.append([
            str(seed),
            *[f"{acc[n]:.3f}" if n in acc else "-"
              for n in ("full_context", "naive_rag", "append_all", "decay",
                        "ocmg")],
            f"{est.auc_mean:.3f}" if est is not None else "-",
            _cell(vs["naive_rag"]), _cell(vs["append_all"]),
        ])

    aggregate = aggregate_runs(all_runs)
    if args.runs > 1:
        if detail == "brief":
            print(C.section("", f"PER SEED   accuracy of each condition, then "
                                "OCMG minus a baseline"))
            print(C.note(
                "acc = share of questions judged correct · AUC = estimator "
                "ROC-AUC · OCMG-RAG / OCMG-append = accuracy difference on the "
                "same questions, paired-bootstrap p, \u2714 better \u2718 worse "
                "\u00b7 no real difference", "muted"))
            print(C.table(
                ["seed", "full ctx", "naive RAG", "append", "decay", "OCMG",
                 "AUC", "OCMG-RAG", "OCMG-append"],
                seed_rows,
                aligns=["left"] + ["right"] * 8,
            ))

        print(C.section("10", f"MEAN OVER {args.runs} SEEDS   sd = spread "
                              "across seeds"))
        order = [n for n in LABELS if n in aggregate["conditions"]]

        def _g(n, key):
            vals = [r["conditions"][n].get("extra", {}).get("grounding", {})
                    .get(key) for r in all_runs if n in r["conditions"]]
            vals = [v for v in vals if v is not None]
            return f"{sum(vals) / len(vals):.1%}" if vals else "-"

        print(C.note(
            "acc = share correct · tok/q = memory word tokens per question · "
            "ev recall = gold evidence that reached the prompt · ctx prec = "
            "prompt records holding evidence", "muted"))
        print(C.table(
            ["condition", "acc mean", "acc sd", "tok/q mean", "ev recall",
             "ctx prec"],
            [
                [
                    LABELS[n],
                    f"{aggregate['conditions'][n]['accuracy']['mean']:.3f}",
                    f"{aggregate['conditions'][n]['accuracy']['sd']:.3f}",
                    f"{aggregate['conditions'][n]['tokens_per_query']['mean']:,.0f}",
                    _g(n, "evidence_recall"), _g(n, "context_precision"),
                ]
                for n in order
            ],
            emphasis_row=order.index("ocmg") if "ocmg" in order else None,
        ))
        # The same two guards summarise() applies per seed: the mean block is
        # what gets read, so a guard only per seed is no guard at all.
        conds = aggregate["conditions"]
        acc = {n: conds[n]["accuracy"]["mean"] for n in conds}
        tok = {n: conds[n]["tokens_per_query"]["mean"] for n in conds}
        no_ceiling = ("full_context" in acc and "naive_rag" in acc
                      and acc["full_context"] < acc["naive_rag"])
        cost_gap = (tok.get("naive_rag", 0) / tok["ocmg"]
                    if tok.get("ocmg") else 1.0)

        meaning = {
            "accuracy_retention": (
                f"OCMG acc / full-context acc; full context "
                f"({acc.get('full_context', 0):.3f}) is below naive RAG "
                f"({acc.get('naive_rag', 0):.3f}), so no upper bound"
                if no_ceiling else "share of full-context accuracy OCMG keeps"),
            "token_reduction": (
                "1 - OCMG tokens / naive RAG tokens; negative = OCMG is dearer"
                + (f"; naive RAG spends {cost_gap:.1f}x OCMG, so a pass would "
                   "flatter it - read the iso-cost sweep" if cost_gap > 1.25
                   else "")),
            "estimator_auc": ("chance a useful record is scored above a useless one; "
                              "rare_term_count alone scores about the same"),
        }
        for key, label, target in (
            ("accuracy_retention", "accuracy retention", 0.95),
            ("token_reduction", "token reduction", 0.40),
            ("estimator_auc", "estimator ROC-AUC", 0.70),
        ):
            t = aggregate.get("targets", {}).get(key)
            if not t:
                continue
            status = C.status(bool(t["met"]))
            if key == "accuracy_retention" and no_ceiling:
                status = C.paint(f"{C.G.warn} NO CEILING", "warning")
            print(C.kv(label, f"{t['mean']:.3f} +/- {t['sd']:.3f}", width=22,
                       hint=f"target >= {target:.2f}  {status}   "
                            f"{meaning[key]}"))

        names = {"naive_rag": "naive RAG", "append_all": "append-all",
                 "decay": "decay"}
        for o, marks in verdicts.items():
            if not marks:
                continue
            print(C.kv(f"OCMG vs {names[o]}",
                       f"better {marks.count('better')}  worse "
                       f"{marks.count('worse')}  no difference "
                       f"{marks.count('same')}",
                       width=22,
                       hint=f"seeds out of {len(marks)}, each by paired "
                            "bootstrap at 95%"))

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
    print(C.kv("results written", out_path,
               hint="every number above, per seed; make_report.py builds "
                    "tables from it"))
    reduced = bool(conditions or args.loo_max_records
                   or args.max_new_tokens != 24)
    print(C.kv("reportable", "yes" if reportable and not reduced else "no",
               hint=("reduced run: arms dropped, counterfactuals capped or "
                     "tokens cut" if reduced and reportable
                     else "" if reportable
                     else "thesis-quotable only with --reader hf "
                          "--embedder sentence-transformers")))
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
