#!/usr/bin/env python3
"""Draw the pipeline diagrams used in docs/PIPELINE_FLOW.md.\n\n    python scripts/make_flow_diagrams.py\n\nWrites four PNGs to docs/figures/. They are images rather than Mermaid so they\nrender in VS Code's built-in Markdown preview (no extension) and can be pasted\ninto slides or the thesis unchanged. Re-run after changing the pipeline.\n"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

INK, MUTED, LINE = "#222222", "#666666", "#4a4a4a"
STYLE = {
    # fill, edge, linestyle
    "data": ("#eef1f5", "#33507a", "-"),
    "step": ("#ffffff", "#33507a", "-"),
    "learn": ("#fbe9cf", "#c77c02", "-"),   # the one thing that learns
    "out": ("#e7f0e7", "#3d7a3d", "-"),
    "audit": ("#ffffff", "#888888", "--"),
    "group": ("#f7f7f7", "#bbbbbb", "-"),
}


def _ax(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub="", kind="step", fs=10):
    fill, edge, ls = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.4, ls=ls, zorder=2))
    if sub:
        ax.text(x, y + h * 0.14, title, ha="center", va="center", fontsize=fs,
                weight="bold", color=INK, zorder=3)
        ax.text(x, y - h * 0.22, sub, ha="center", va="center",
                fontsize=fs - 2, color=MUTED, zorder=3, linespacing=1.3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs,
                weight="bold", color=INK, zorder=3)


def group(ax, x0, y0, x1, y1, label):
    fill, edge, _ = STYLE["group"]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.2,rounding_size=2",
                                fc=fill, ec=edge, lw=1, zorder=0))
    ax.text(x0 + 1.5, y1 - 1.8, label, ha="left", va="top", fontsize=10,
            weight="bold", color=MUTED, zorder=1)


def arrow(ax, pts, label="", dashed=False, lpos=0.5, loff=(1.2, 0), ha="left"):
    """Polyline through pts with an arrowhead on the last segment."""
    ls = (0, (4, 3)) if dashed else "-"
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=LINE, lw=1.3, ls=ls, zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                ls=ls, mutation_scale=14))
    if label:
        i = min(int(lpos * (len(pts) - 1)), len(pts) - 2)
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        ax.text((x0 + x1) / 2 + loff[0], (y0 + y1) / 2 + loff[1], label,
                fontsize=8, color=MUTED, ha=ha, va="center", style="italic",
                zorder=3)


def legend(ax, x, y):
    items = [("data", "dataset / data"), ("step", "processing step"),
             ("learn", "the learner (only part that trains)"),
             ("out", "outputs"), ("audit", "audit only, never trains")]
    for i, (k, t) in enumerate(items):
        fill, edge, ls = STYLE[k]
        ax.add_patch(FancyBboxPatch((x, y - i * 3.2), 3, 1.8,
                                    boxstyle="round,pad=0.1", fc=fill, ec=edge,
                                    ls=ls, lw=1.2))
        ax.text(x + 4, y - i * 3.2 + 0.9, t, fontsize=8, color=MUTED,
                va="center")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ diagrams

def pipeline():
    fig, ax = _ax(12, 13, 120, 130)
    ax.text(60, 128, "OCMG pipeline: one experiment run", ha="center",
            fontsize=14, weight="bold", color=INK)
    A, B = 29, 93          # centre of the Phase A and Phase B columns
    AW, BW = 42, 44        # box widths

    box(ax, 60, 118, 46, 7, "data/locomo10.json",
        "10 conversations · 5,882 turns · 1,986 questions", "data")
    box(ax, 60, 106, 46, 7, "Load & parse   (data/benchmarks.py)",
        "session dates, dia_ids, 5 categories, evidence ids")
    box(ax, 60, 94, 46, 7, "Split by conversation   (experiment.py)",
        "seeded; no record is shared between the halves")
    arrow(ax, [(60, 114.2), (60, 110)])
    arrow(ax, [(60, 102.2), (60, 98)])

    # Phase A
    group(ax, 4, 26, 54, 84, "PHASE A · warm-up")
    box(ax, A, 74, AW, 7, "Build store: admit everything",
        "ingest.py -> store.py (in-memory SQLite)")
    box(ax, A, 63, AW, 7, "Exploratory retrieval   (retrieval.py)",
        "k drawn from 4-12, plus 20% random records")
    box(ax, A, 52, AW, 7, "Frozen reader answers, Judge scores",
        "reader.py: correct 1 / 0 against the gold answer")
    box(ax, A, 41, AW, 7, "Leave-one-out credit   (credit.py)",
        "re-answer without each record; delta = lost correctness")
    box(ax, A, 31, AW, 6, "Labels: useful = mean delta > 0",
        "655 records labelled, 43% useful (seed 0)", "data", fs=9)
    for y0, y1 in ((70.2, 66.8), (59.2, 55.8), (48.2, 44.8), (37.2, 34.4)):
        arrow(ax, [(A, y0), (A, y1)])
    arrow(ax, [(36.6, 94), (A, 94), (A, 78)], "5 conversations", lpos=0.0,
          loff=(-6, 1.6), ha="right")

    box(ax, A, 16, AW, 8, "Utility estimator   (estimator.py)",
        "gradient-boosted trees + Platt calibration\n"
        "5-fold CV grouped by conversation", "learn")
    arrow(ax, [(A, 27.6), (A, 20.4)])
    box(ax, A, 4, AW, 6, "Thresholds   (governance.calibrate)",
        "quantiles of predicted utility U(m)", fs=9)
    arrow(ax, [(A, 11.6), (A, 7.4)])

    # audit: compares labels with LoCoMo evidence ids, never feeds training
    box(ax, 59.5, 16, 8.4, 8, "Label\naudit", "", "audit", fs=8.5)
    ax.text(59.5, 10.2, "vs evidence ids", ha="center", fontsize=7,
            color=MUTED)
    arrow(ax, [(50.4, 16), (54.8, 16)], dashed=True)

    # Phase B
    group(ax, 68, 22, 118, 84, "PHASE B · evaluation")
    box(ax, B, 74, BW, 8, "Build one store per condition",
        "Full context | Naive RAG | Append-all\nDecay | OCMG")
    box(ax, B, 61, BW, 7, "Every question: retrieve -> read",
        "orchestrator.evaluate()")
    box(ax, B, 49, BW, 7, "Score each answer",
        "Judge 0/1 · LoCoMo F1 · grounding vs evidence ids")
    box(ax, B, 37, BW, 7, "Aggregate   (metrics.py)",
        "accuracy · tokens/question · paired bootstrap")
    box(ax, B, 27, BW, 5, "repeat for each seed (3)", "", "group", fs=9)
    arrow(ax, [(83.4, 94), (B, 94), (B, 78.4)], "5 conversations", lpos=0.0,
          loff=(6, 1.6), ha="left")
    for y0, y1 in ((69.6, 64.8), (57.2, 52.8), (45.2, 40.8)):
        arrow(ax, [(B, y0), (B, y1)])

    # the estimator and thresholds feed the OCMG condition
    arrow(ax, [(50.4, 4), (65.5, 4), (65.5, 74), (70.6, 74)])
    ax.text(64.8, 30, "OCMG uses\nU(m) and\nthresholds", ha="right",
            va="center", fontsize=8, color=MUTED, style="italic")

    box(ax, B, 10, BW, 9, "Outputs",
        "results/*.json -> make_report.py (CSV, figures)\n"
        "make_results_page.py -> RESULTS.md", "out")
    arrow(ax, [(B, 24.4), (B, 14.6)])

    legend(ax, 4, 126)
    return save(fig, "flow_1_pipeline.png")


def learning_loop():
    fig, ax = _ax(12, 6.2, 120, 62)
    ax.text(60, 60, "How the system learns: leave-one-out feedback loop",
            ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(60, 55.5, "The reader LLM is frozen. Only the utility estimator "
            "trains, on labels produced by the loop below.",
            ha="center", fontsize=9, color=MUTED, style="italic")

    top = [(14, "Question", "from a warm-up\nconversation", "data"),
           (43, "Retrieve records", "A  B  C  D", "step"),
           (72, "Reader answers", "frozen LLM\n(or simulated)", "step"),
           (101, "Judge", "correct = 1 / 0\nvs gold answer", "step")]
    bot = [(101, "Remove each record", "answer again without\nA, then B, C, D", "step"),
           (72, "Credit per record", "delta = correct(all)\n- correct(without it)", "step"),
           (43, "Label", "useful if mean delta > 0\n(>= 2 observations)", "data"),
           (14, "Estimator learns", "30 write-time features\n-> P(useful) = U(m)", "learn")]
    for x, t, s, k in top:
        box(ax, x, 41, 24, 10, t, s, k)
    for x, t, s, k in bot:
        box(ax, x, 18, 24, 10, t, s, k)
    for (x0, *_), (x1, *_) in zip(top, top[1:]):
        arrow(ax, [(x0 + 12.6, 41), (x1 - 12.6, 41)])
    arrow(ax, [(101, 35.4), (101, 23.6)])
    for (x0, *_), (x1, *_) in zip(bot, bot[1:]):
        arrow(ax, [(x0 - 12.6, 18), (x1 + 12.6, 18)])
    # Back to retrieval, through the gap between the first two columns and
    # between the rows, so the line never crosses a box.
    arrow(ax, [(14, 12.4), (14, 7), (28.5, 7), (28.5, 29.5), (43, 29.5),
               (43, 35.4)])
    ax.text(31, 7, "U(m) then decides admission, forgetting and retrieval "
            "depth (OCMG)", fontsize=8.5, color=MUTED, style="italic",
            va="center")
    ax.text(31, 2.5, "Example: correct with A B C D, wrong without A  ->  "
            "delta(A) = 1, delta(B) = delta(C) = delta(D) = 0",
            fontsize=8.5, color=MUTED, va="center")
    return save(fig, "flow_3_learning_loop.png")


def record_journey():
    steps = [
        ("1  Raw turn", "locomo10.json", '"I went to a LGBTQ support group yesterday..."  (session 8 May 2023)', "data"),
        ("2  Turn", "benchmarks.py", "dia_id D1:3, timestamp 8 May 2023 13:58", "step"),
        ("3  Candidate", "ingest.py", "topic segment; 'yesterday' -> [2023-05-07]; type = profile", "step"),
        ("4  30 features", "features.py", "token_length 111, rare_term_count 111, has_temporal_anchor 1", "step"),
        ("5  Utility U(m)", "estimator.py", "predicted probability the record will be useful", "learn"),
        ("6  Admit", "governance.py + store.py", "U >= write threshold -> hot tier, else cold", "step"),
        ("7  Retrieve", "retrieval.py", "0.65 dense + 0.35 BM25; OCMG adaptive k (2-10)", "step"),
        ("8  Prompt + answer", "reader.py", "[1] (profile, 8 May 2023) ...  ->  '7 May 2023'", "step"),
        ("9  Score", "Judge + locomo_eval + grounding", "correct 1 · LoCoMo F1 1.000 · evidence D1:3 in prompt", "out"),
    ]
    n = len(steps)
    fig, ax = _ax(12, 0.95 * n + 1.2, 120, n * 10 + 8)
    ax.text(60, n * 10 + 5, "One record, from JSON to score   "
            "(question: 'When did Caroline go to the LGBTQ support group?')",
            ha="center", fontsize=12, weight="bold", color=INK)
    for i, (t, f, ex, k) in enumerate(steps):
        y = (n - i) * 10 - 4
        box(ax, 17, y, 28, 6.5, t, f, k, fs=9.5)
        ax.text(36, y, ex, fontsize=9, color=INK, va="center")
        if i < n - 1:
            arrow(ax, [(17, y - 3.6), (17, y - 6.4)])
    return save(fig, "flow_2_record_journey.png")


def verification():
    fig, ax = _ax(12, 6.6, 120, 66)
    ax.text(60, 64, "How results are verified with only locomo10.json",
            ha="center", fontsize=14, weight="bold", color=INK)
    box(ax, 14, 33, 22, 10, "locomo10.json", "the only ground truth", "data")
    box(ax, 44, 46, 24, 9, "Gold answers", "what is correct", "data")
    box(ax, 44, 20, 24, 9, "Evidence ids", "which turns support it", "data")
    arrow(ax, [(25.6, 36), (31.6, 44)])
    arrow(ax, [(25.6, 30), (31.6, 22)])

    checks_a = [(55, "Accuracy", "reader.Judge (0/1)"),
                (46, "Official LoCoMo F1", "locomo_eval.py"),
                (37, "Answer groundedness", "grounding.answer_support")]
    checks_e = [(27, "Label audit", "precision 1.000 · kappa 0.57"),
                (18, "Evidence recall / context precision", "grounding.retrieval_scores"),
                (9, "Evidence still in memory", "forgetting ceiling")]
    for y, t, s in checks_a:
        box(ax, 86, y, 34, 7, t, s, "out", fs=9.5)
        arrow(ax, [(56.6, 46), (68.6, y)])
    for y, t, s in checks_e:
        box(ax, 86, y, 34, 7, t, s, "out", fs=9.5)
        arrow(ax, [(56.6, 20), (68.6, y)])
    ax.text(60, 1.5, "Plus, independent of the data: paired bootstrap "
            "(10,000 resamples), 3 seeds, bit-identical re-runs, 90 unit tests",
            ha="center", fontsize=9, color=MUTED, style="italic")
    return save(fig, "flow_4_verification.png")


def simple_architecture():
    """The whole system as four steps in a loop: the version to remember."""
    fig, ax = _ax(12, 6.4, 120, 64)
    ax.text(60, 62, "OCMG in four steps", ha="center", fontsize=15,
            weight="bold", color=INK)
    ax.text(60, 57.5, "The system learns which memories are worth keeping by "
            "checking which ones actually helped answer correctly.",
            ha="center", fontsize=9.5, color=MUTED, style="italic")

    y_top = 38
    box(ax, 18, y_top, 28, 13, "1. STORE",
        "turn the conversation into\nsmall memory records\n(ingest.py, store.py)", "data")
    box(ax, 60, y_top, 28, 13, "2. RETRIEVE",
        "pick the most relevant\nmemories for a question\n(retrieval.py)")
    box(ax, 102, y_top, 28, 13, "3. ANSWER",
        "the LLM reads them and\nanswers - it never learns\n(reader.py)")
    box(ax, 60, 11, 34, 13, "4. LEARN",
        "remove each memory and ask again:\nanswer breaks = memory was useful\n"
        "(credit.py, estimator.py)", "learn")

    arrow(ax, [(32.6, y_top), (45.4, y_top)])
    arrow(ax, [(74.6, y_top), (87.4, y_top)])
    arrow(ax, [(102, 31.1), (102, 11), (77.6, 11)], "was the answer right?",
          lpos=0.0, loff=(1.5, 0))
    arrow(ax, [(42.4, 11), (18, 11), (18, 30.9)],
          "what it learned decides\nwhat to keep and retrieve", lpos=0.5,
          loff=(1.8, 1), ha="left")

    ax.text(60, 1.5, "Amber = the only part that learns.  The LLM is frozen: "
            "same weights before and after.", ha="center", fontsize=8.5,
            color=MUTED, style="italic")
    return save(fig, "flow_0_simple_architecture.png")


def main() -> int:
    for fn in (simple_architecture, pipeline, learning_loop,
               record_journey, verification):
        print(f"wrote {os.path.relpath(fn())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
