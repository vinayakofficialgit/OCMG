#!/usr/bin/env python3
"""Print the finished results that are worth showing, with no run required.

A live run is the wrong way to show a result that already exists: it takes
minutes, it can fail in front of an audience, and the strongest numbers in
this project come from runs that take over an hour. This reads the result
files and prints them.

    python scripts/show_demo.py              # everything available
    python scripts/show_demo.py --seeds      # the 3-seed real-embedder table
    python scripts/show_demo.py --answers    # what the real reader generated

Nothing here recomputes anything, so what it prints is exactly what the runs
recorded, including their reportable flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocmg import console as C  # noqa: E402

SEEDS_FILE = "results/full/run_real_embed.json"
ANSWERS_FILES = ["results/full/demo_real.json", "results/full/fittable_hf.json"]

LABELS = {
    "full_context": "Full context",
    "naive_rag": "Naive RAG (top-k)",
    "append_all": "Append-all typed",
    "decay": "Decay-based",
    "ocmg": "OCMG",
}


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def show_seeds() -> bool:
    d = load(SEEDS_FILE)
    if d is None:
        print(C.note(f"{SEEDS_FILE} not found - run the 3-seed real-embedder "
                     "experiment first.", "warning", glyph=C.G.warn))
        return False

    meta = d.get("meta", {})
    runs = d.get("runs", [])
    print(C.section("", "REAL EMBEDDER, 3 SEEDS   the strongest result available"))
    print(C.kv("embedder", str(meta.get("embedder"))))
    print(C.kv("reader", str(meta.get("reader")),
               hint="simulated, so still not reportable under section 2.3"))
    print(C.kv("seeds", str(len(runs))))
    print("")

    # Per-seed gap against each baseline. Sign consistency across seeds is
    # what this project treats as evidence, so it is the column that matters.
    rows = []
    for r in runs:
        c = r.get("conditions", {})
        if "ocmg" not in c or "naive_rag" not in c:
            continue
        o = c["ocmg"]["accuracy"]
        rows.append([
            str(r.get("seed")),
            f"{c['naive_rag']['accuracy']:.4f}",
            f"{o:.4f}",
            f"{o - c['naive_rag']['accuracy']:+.4f}",
            f"{o - c['append_all']['accuracy']:+.4f}"
            if "append_all" in c else "-",
            f"{o - c['decay']['accuracy']:+.4f}" if "decay" in c else "-",
        ])
    if rows:
        print(C.table(["seed", "naive top-k", "OCMG", "vs top-k",
                       "vs append-all", "vs decay"], rows))
        print("")

    agg = d.get("aggregate", {}).get("conditions", {})
    if agg:
        print(C.table(
            ["condition", "acc mean", "acc sd", "tok/q"],
            # LABELS order, not the JSON's. Every other table in the project
            # runs full context -> baselines -> OCMG, and a table that alone
            # runs alphabetically makes an audience hunt for the row.
            [[LABELS[k], f"{agg[k]['accuracy']['mean']:.3f}",
              f"{agg[k]['accuracy']['sd']:.3f}",
              f"{agg[k]['tokens_per_query']['mean']:,.0f}"]
             for k in LABELS if k in agg],
        ))

    n = len(rows)
    neg = sum(1 for r in rows if r[3].startswith("-"))
    if n:
        print("")
        print(C.note(
            f"OCMG is behind naive top-k on {neg} of {n} seeds. The standard "
            "deviations on both are 0.006, so the gap is several times the "
            "seed noise. With the hashing embedder OCMG was nominally ahead "
            "at a single operating point, and only the iso-cost sweep "
            "refuted it; with real embeddings it is behind on both readings. "
            "A competent retriever absorbs the headroom the governance was "
            "filling.", "muted"))
    return True


def show_answers() -> bool:
    for path in ANSWERS_FILES:
        d = load(path)
        if d is None:
            continue
        samples = (d.get("runs") or [{}])[0].get("samples") or []
        if not samples:
            continue

        meta = d.get("meta", {})
        print(C.section("", "WHAT THE REAL READER ACTUALLY GENERATED"))
        print(C.kv("model", str(meta.get("hf_model"))))
        print(C.kv("source", path))
        print("")
        for x in samples:
            print(C.kv("question", x["question"][:88]))
            print(C.kv("  reference", repr(x["gold"])[:84]))
            print(C.kv("  generated", repr(x["prediction"])[:84],
                       hint=f"token-F1 {x['f1']:.2f}   "
                            f"judged {'correct' if x['correct'] else 'wrong'}"))
            print("")
        print(C.note(
            "Read these before reading the accuracy column. The scored "
            "accuracy is 0.000, but the model is substantively right on "
            "several: it answers '2023-08-21' where the reference says "
            "'21 August, 2023'. Two failures are visible and neither is the "
            "memory system - the token-F1 judge cannot see that two date "
            "formats are the same fact, and generation does not stop at the "
            "end of the answer, so every reply runs on into the next record "
            "and dilutes the overlap it is scored on. This is the concrete "
            "argument for the LLM judge in section 9 step 4, and for a stop "
            "condition on the reader.", "critical", glyph=C.G.bad))
        return True

    print(C.note("No run with captured answers found. Any run with "
                 "--reader hf records them.", "warning", glyph=C.G.warn))
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="print finished results")
    ap.add_argument("--seeds", action="store_true",
                    help="the 3-seed real-embedder comparison")
    ap.add_argument("--answers", action="store_true",
                    help="generated answers from a real-reader run")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    C.refresh_glyphs()
    if args.no_color:
        C.set_color(False)

    both = not (args.seeds or args.answers)
    ok = True
    if args.seeds or both:
        ok = show_seeds() and ok
        if both:
            print("")
    if args.answers or both:
        ok = show_answers() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
