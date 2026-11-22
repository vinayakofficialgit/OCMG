#!/usr/bin/env python3
"""Build the demo pack: four PNG images that hold everything to show.

    python scripts/make_demo_pack.py

    demo/1_pipeline_flow.png      docs/PIPELINE_FLOW.md: aims, what we leverage,
                                  the 4 flow diagrams with step-by-step text,
                                  precision/recall, tokens, accuracy, checks,
                                  file map
    demo/2_results_3_seeds.png    results/full/RESULTS.md: 3-seed tables, targets,
                                  significance, label audit, rarity ablation,
                                  forgetting, 4 charts
    demo/3_run_seed0.png          the colour terminal run of one seed (LoCoMo,
                                  seed 0): sections 1-9 and AT A GLANCE
    demo/4_run_all_3_seeds.png    the colour terminal log of run_all.py: tests,
                                  main experiment, rarity ablation, forgetting
                                  fraction and criterion, 3 seeds each

Inputs it reads (each made by an earlier command):
    results/demo_seed0_color.txt  scripts/terminal_to_png.py --record ... run_experiment.py
    results/run_all_color.txt     scripts/terminal_to_png.py --record ... run_all.py
A missing input is reported and skipped; nothing is faked.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
OUT = os.path.join(ROOT, "demo")

STEPS = [
    ("1_pipeline_flow.png", "make_doc_png.py", "docs/PIPELINE_FLOW.md",
     "how the project works, end to end"),
    ("2_results_3_seeds.png", "make_doc_png.py", "results/full/RESULTS.md",
     "every 3-seed result, tables and charts"),
    ("3_run_seed0.png", "terminal_to_png.py", "results/demo_seed0_color.txt",
     "one real run, explained section by section"),
    ("4_run_all_3_seeds.png", "terminal_to_png.py", "results/run_all_color.txt",
     "the full experiment log behind image 2"),
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for name, tool, source, what in STEPS:
        src = os.path.join(ROOT, source)
        dst = os.path.join(OUT, name)
        if not os.path.exists(src):
            print(f"skip     {name:26} {source} not found")
            continue
        flag = "--doc" if tool == "make_doc_png.py" else None
        cmd = [PY, os.path.join(ROOT, "scripts", tool)]
        cmd += [flag, src] if flag else [src]
        cmd += ["--out", dst]
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        size = os.path.getsize(dst) / 1e6
        print(f"made     {name:26} {what}  ({size:.1f} MB)")
        made += 1
    print(f"\n{made} image(s) in {os.path.relpath(OUT, ROOT)}\\  "
          "open them in VS Code or any image viewer")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
