#!/usr/bin/env python3
"""One command that runs every OCMG experiment in order.

    python scripts/run_all.py              # the full set, ~25 min
    python scripts/run_all.py --quick      # synthetic, ~2 min, for rehearsal
    python scripts/run_all.py --real       # also the sentence-transformer and
                                           # generative-reader steps

Each step is a separate process, the same invocation the README documents, so
that running the set and running one step by hand cannot diverge. The steps are
sequenced rather than parallelised: they share four cores, and two runs
competing for them take longer in total than one after the other.

Steps that need torch are skipped with a reason when it cannot be imported,
rather than failing the run. A skip is reported as a skip: this script never
reports a step as passed when it did not execute.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocmg import console as C  # noqa: E402

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
LOCOMO = ["--dataset", "locomo", "--path", "data/locomo10.json"]
SYNTH = ["--dataset", "synthetic", "--conversations", "8"]


def steps(args) -> list:
    """(label, what it answers, argv, needs_torch)."""
    data = SYNTH if args.quick else LOCOMO
    n = 1 if args.quick else args.seeds
    seeds = ["--runs", str(n)]
    # compare_forgetting.py takes explicit seed numbers, not a count.
    seed_list = ["--seeds", *[str(i) for i in range(n)]]
    out = os.path.join("results", "quick" if args.quick else "full")

    plan = []
    if not args.skip_tests:
        plan.append(("tests", "does the code still do what it claims",
                     [PY, "-m", "pytest", "-q"], False))
    plan += [
        ("main experiment", "RQ1-RQ4: every condition, cost beside accuracy",
         [PY, os.path.join(HERE, "run_experiment.py"), *data, *seeds,
          "--out", os.path.join(out, "run_main.json")], False),
        ("rarity ablation", "is the learned utility just an IDF heuristic",
         [PY, os.path.join(HERE, "run_experiment.py"), *data, *seeds,
          "--estimator", "rarity",
          "--out", os.path.join(out, "run_rarity.json")], False),
    ]
    if not args.quick:
        # Both of these sweep several configurations per seed, so they are the
        # expensive half of the set and the synthetic rehearsal omits them.
        plan += [
            ("forgetting fraction", "how much forgetting is optimal",
             [PY, os.path.join(HERE, "compare_forgetting.py"), *LOCOMO,
              *seed_list,
              "--quantiles", "0.00", "0.05", "0.10", "0.25", "0.40",
              "--signal", "estimator",
              "--out", os.path.join(out, "forget_quantiles.json")], False),
            ("forgetting criterion", "does the utility signal beat recency",
             [PY, os.path.join(HERE, "compare_forgetting.py"), *LOCOMO,
              *seed_list,
              "--arms", "ledger", "estimator", "recency", "none",
              "--out", os.path.join(out, "forget_criterion.json")], False),
        ]
    if args.real:
        plan += [
            ("real embedder", "does a real encoder change the retrieval result",
             [PY, os.path.join(HERE, "run_experiment.py"), *LOCOMO, *seeds,
              "--embedder", "sentence-transformers",
              "--out", os.path.join(out, "run_real_embed.json")], True),
            ("real reader probe", "what one generation costs on this machine",
             [PY, os.path.join(HERE, "run_experiment.py"), *LOCOMO,
              "--reader", "hf", "--embedder", "sentence-transformers",
              "--hf-model", args.hf_model, "--questions-per-conv", "5",
              "--runs", "1",
              "--out", os.path.join(out, "probe_hf.json")], True),
        ]
    return plan


def torch_status() -> str | None:
    """None when torch is usable, otherwise why it is not."""
    try:
        import torch  # noqa: F401, PLC0415
    except Exception as exc:
        # The common one here is WinError 126, and the message names the wrong
        # file: shm.dll is fine, but torch_cpu.dll needs
        # vcruntime140_threads.dll, which ships with the VC++ 2015-2022
        # redistributable from ~14.30 and not with the 2019 one (14.28).
        # scripts/fix_torch_dll.py repairs that without administrator rights.
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="run every OCMG experiment")
    ap.add_argument("--quick", action="store_true",
                    help="synthetic data, one seed: a rehearsal, not a result")
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per step. Cost is close to linear in this, so "
                         "1 runs in about a third of the time - but every "
                         "single-seed result in this project that mattered "
                         "has reversed at some point, so 3 is the default")
    ap.add_argument("--skip-tests", action="store_true",
                    help="omit the test step")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only steps whose label contains one of these")
    ap.add_argument("--real", action="store_true",
                    help="also the steps that need torch")
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    try:
        # Line buffering matters here, not just the encoding: this process
        # prints the plan and the step headers while child processes write
        # straight to the console, so a block-buffered parent prints its
        # banner after the steps it was supposed to introduce.
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    C.refresh_glyphs()
    if args.no_color:
        C.set_color(False)

    plan = steps(args)
    if args.only:
        want = [w.lower() for w in args.only]
        plan = [row for row in plan
                if any(w in row[0].lower() for w in want)]
        if not plan:
            print(C.note("--only matched no step.", "critical", glyph=C.G.bad))
            return 2
    outdir = os.path.join("results", "quick" if args.quick else "full")
    os.makedirs(outdir, exist_ok=True)

    print(C.banner(
        "OCMG  -  full experiment set",
        f"{len(plan)} steps  |  results in {outdir}"
        + ("  |  QUICK: synthetic data" if args.quick else ""),
    ))
    if args.quick:
        print(C.note(
            "QUICK MODE.  Synthetic data and one seed. This checks that every "
            "step runs; the numbers are not comparable to a LoCoMo run and "
            "are not results.", "warning", glyph=C.G.warn))

    blocked = torch_status() if args.real else None
    if args.real and blocked:
        print(C.note(
            f"torch cannot be imported ({blocked}). The two real-model steps "
            "will be skipped. On Windows this is usually one missing DLL, "
            "vcruntime140_threads.dll, which torch_cpu.dll needs and the "
            "VS2019 runtime does not carry. Fix it with "
            "'python scripts/fix_torch_dll.py' - no administrator rights "
            "needed - then re-run.", "warning", glyph=C.G.warn))

    print(C.section("", "PLAN"))
    for i, (label, question, _, needs) in enumerate(plan, 1):
        mark = "skip" if (needs and blocked) else f"{i}"
        print(C.kv(f"{mark}  {label}", question, width=26))

    results = []
    for i, (label, _, argv, needs) in enumerate(plan, 1):
        if needs and blocked:
            results.append((label, "skipped", 0.0))
            continue
        print(C.section(str(i), label.upper()))
        t0 = time.time()
        code = subprocess.run(argv, cwd=os.path.join(HERE, "..")).returncode
        dt = time.time() - t0
        results.append((label, "ok" if code == 0 else f"failed ({code})", dt))
        if code != 0:
            # Stop on failure: later steps read nothing from earlier ones, but
            # a wall of output after a broken step buries the cause.
            print(C.note(f"{label} failed; stopping here so the error above "
                         "is the last thing on screen.",
                         "critical", glyph=C.G.bad))
            break

    print(C.section("", "SUMMARY"))
    print(C.table(
        ["step", "outcome", "elapsed"],
        [[label, state, f"{dt:,.0f}s" if dt else "-"]
         for label, state, dt in results],
        aligns=["left", "left", "right"],
    ))
    total = sum(dt for _, _, dt in results)
    print(C.kv("total", f"{total / 60:.1f} min"))
    print(C.kv("results", outdir))
    failed = [l for l, s, _ in results if s.startswith("failed")]
    print("")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
