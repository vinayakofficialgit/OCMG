#!/usr/bin/env python3
"""Build one readable results page from a run_all.py result set.

    python scripts/make_results_page.py                 # results/full
    python scripts/make_results_page.py --dir results/quick

Reads the four JSON files run_all.py writes (run_main, run_rarity,
forget_quantiles, forget_criterion) and writes <dir>/RESULTS.md plus four
charts in <dir>/figures/. Open RESULTS.md in VS Code and press Ctrl+Shift+V
for the formatted preview: real tables and embedded charts, nothing wraps.

Every number comes from the JSON; nothing is re-run. The page is a view of
results, not thesis text.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LABELS = {
    "full_context": "Full context",
    "naive_rag": "Naive RAG (top-k)",
    "append_all": "Append-all typed",
    "decay": "Decay-based",
    "ocmg": "OCMG",
}
WHAT = {
    "full_context": "whole conversation in every prompt",
    "naive_rag": "fixed top-10 retrieval (the reference)",
    "append_all": "keep every record, top-10",
    "decay": "forgets by recency and use count",
    "ocmg": "learned utility gate + adaptive k",
}
# One hue for every condition and one emphasis colour for OCMG, the same pair
# make_report.py uses, so the page and the thesis figures read as one set.
BASE, EMPH, MUTED, INK = "#33507a", "#c77c02", "#9a9a9a", "#333333"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888", "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


# -------------------------------------------------------------------- helpers

def _load(d, name):
    p = os.path.join(d, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf8") as fh:
        return json.load(fh)


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    # Population sd, as ocmg/reporting.py and the console use, so the page and
    # the run log show the same spread.
    return statistics.mean(vals), statistics.pstdev(vals)


def _verdict(sig):
    if sig["ci_low"] <= 0 <= sig["ci_high"]:
        return "same"
    return "better" if sig["delta"] > 0 else "worse"


_WORD = {"better": "✔ better", "worse": "✘ worse", "same": "· no difference"}


def _grounding(run, cond, key):
    return (run["conditions"].get(cond, {}).get("extra", {})
            .get("grounding", {}).get(key))


def _table(head, rows, align=None):
    align = align or ["l"] + ["r"] * (len(head) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _stamp(fig, reportable):
    if not reportable:
        # Below the axes, so it never sits on an axis label.
        fig.text(0.99, -0.04, "pipeline validation - simulated reader",
                 ha="right", va="top", fontsize=7, color="#a33",
                 style="italic")


# --------------------------------------------------------------------- charts

def chart_cost(main, fdir, reportable):
    """Accuracy against tokens: one point per condition, mean +/- sd."""
    agg = main["aggregate"]["conditions"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for n in [c for c in LABELS if c in agg and c != "full_context"]:
        a, t = agg[n]["accuracy"], agg[n]["tokens_per_query"]["mean"]
        col = EMPH if n == "ocmg" else BASE
        ax.errorbar(t, a["mean"], yerr=a["sd"], fmt="o", ms=8, color=col,
                    ecolor=MUTED, elinewidth=1, capsize=3, zorder=3)
        # Naive RAG and OCMG sit close together; label one under its error bar,
        # one above-right so the names never overlap.
        off = {"naive_rag": (0, -44), "ocmg": (10, 8)}.get(n, (10, 4))
        ax.annotate(LABELS[n], (t, a["mean"]), xytext=off,
                    textcoords="offset points", fontsize=9, color=INK,
                    ha="center" if off[0] == 0 else "left")
    ax.set_xlabel("memory word tokens per question (lower = cheaper)")
    ax.set_ylabel("accuracy (share correct)")
    fc = agg.get("full_context")
    if fc:
        ax.set_title(f"Accuracy vs cost, mean of {main['aggregate']['n_runs']} "
                     f"seeds   (full context: {fc['accuracy']['mean']:.3f} at "
                     f"{fc['tokens_per_query']['mean']:,.0f} tokens, off chart)",
                     fontsize=9, loc="left", color=INK)
    _stamp(fig, reportable)
    path = os.path.join(fdir, "1_accuracy_vs_tokens.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_evidence(main, fdir, reportable):
    """Evidence recall and context precision side by side (both percentages)."""
    runs = main["runs"]
    conds = [c for c in LABELS if c in runs[0]["conditions"]]
    rec = [_ms([_grounding(r, c, "evidence_recall") for r in runs])[0] * 100
           for c in conds]
    pre = [_ms([_grounding(r, c, "context_precision") for r in runs])[0] * 100
           for c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, vals, title in ((axes[0], rec, "evidence recall %\n(needed evidence that reached the prompt)"),
                            (axes[1], pre, "context precision %\n(prompt records that hold evidence)")):
        cols = [EMPH if c == "ocmg" else BASE for c in conds]
        bars = ax.barh([LABELS[c] for c in conds], vals, color=cols, height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                    va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    _stamp(fig, reportable)
    path = os.path.join(fdir, "2_evidence_recall_precision.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_forget_fraction(fq, fdir, reportable):
    """Accuracy and evidence ceiling as forgetting grows: two panels, one unit each."""
    qs = list(fq)
    x = [float(q.split("=")[1]) * 100 for q in qs]
    acc = [_ms([r["accuracy"] for r in fq[q]]) for q in qs]
    ceil = [_ms([r["ceiling"] for r in fq[q]])[0] * 100 for q in qs]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    axes[0].errorbar(x, [a for a, _ in acc], yerr=[s for _, s in acc],
                     marker="o", ms=7, lw=2, color=BASE, ecolor=MUTED, capsize=3)
    axes[0].set_title("accuracy", fontsize=9, loc="left", color=INK)
    axes[1].plot(x, ceil, marker="o", ms=7, lw=2, color=BASE)
    axes[1].set_title("gold evidence still in memory %", fontsize=9,
                      loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("share of records forgotten (%)")
        ax.set_xticks(x)
    _stamp(fig, reportable)
    path = os.path.join(fdir, "3_forgetting_fraction.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_forget_criterion(fc, fdir, reportable):
    """Dot plot, not bars: the differences are a few points of accuracy, and a
    bar must start at zero, which would hide them; a dot may not."""
    arms = list(fc)
    acc = [_ms([r["accuracy"] for r in fc[a]]) for a in arms]
    names = {"ledger": "outcome ledger", "estimator": "learned estimator",
             "recency": "recency", "none": "no forgetting"}
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = list(range(len(arms)))
    for yi, a, (m, sd) in zip(y, arms, acc):
        col = EMPH if a == "none" else BASE
        ax.errorbar(m, yi, xerr=sd, fmt="o", ms=9, color=col, ecolor=MUTED,
                    elinewidth=1.2, capsize=3)
        ax.text(m + sd + 0.004, yi, f"{m:.3f}", va="center", fontsize=8,
                color=INK)
    ax.set_yticks(y, [names.get(a, a) for a in arms])
    ax.set_ylim(len(arms) - 0.5, -0.5)
    hi = max(m + sd for m, sd in acc)
    lo = min(m - sd for m, sd in acc)
    ax.set_xlim(lo - 0.01, hi + 0.025)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (mean ± sd over seeds)")
    ax.set_title("accuracy by forgetting criterion (about 25% forgotten)",
                 fontsize=9, loc="left", color=INK)
    _stamp(fig, reportable)
    path = os.path.join(fdir, "4_forgetting_criterion.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------- page

def section_main(main, rarity, lines, figs):
    runs, agg = main["runs"], main["aggregate"]
    n = agg["n_runs"]
    conds = [c for c in LABELS if c in agg["conditions"]]

    lines += ["## 1 · Main experiment: does OCMG beat the baselines?", ""]
    rows = []
    for c in conds:
        a = agg["conditions"][c]
        rec = _ms([_grounding(r, c, "evidence_recall") for r in runs])[0]
        pre = _ms([_grounding(r, c, "context_precision") for r in runs])[0]
        name = f"**{LABELS[c]}**" if c == "ocmg" else LABELS[c]
        rows.append([name, f"{a['accuracy']['mean']:.3f} ± {a['accuracy']['sd']:.3f}",
                     f"{a['tokens_per_query']['mean']:,.0f}",
                     f"{rec:.1%}", f"{pre:.1%}", WHAT[c]])
    lines += _table(["condition", "accuracy", "tokens / question",
                     "evidence recall", "context precision", "what it is"],
                    rows, ["l", "r", "r", "r", "r", "l"])
    lines += ["", f"*Mean ± sd over {n} seeds. Accuracy = share of questions "
              "judged correct. Evidence recall = gold evidence turns that "
              "reached the prompt. Context precision = prompt records that "
              "hold evidence.*", "",
              f"![accuracy vs tokens]({figs['cost']})", "",
              f"![evidence]({figs['evidence']})", ""]

    lines += ["**Is the difference real?** Paired bootstrap on the same "
              "questions, 10,000 resamples, 95% CI, per seed.", ""]
    head = ["seed", "OCMG acc"]
    others = [o for o in ("naive_rag", "append_all", "decay")
              if f"ocmg_vs_{o}" in runs[0]["significance"]]
    head += [f"OCMG − {LABELS[o]}" for o in others]
    rows = []
    tally = {o: [] for o in others}
    for r in runs:
        row = [r["seed"], f"{r['conditions']['ocmg']['accuracy']:.3f}"]
        for o in others:
            s = r["significance"][f"ocmg_vs_{o}"]
            v = _verdict(s)
            tally[o].append(v)
            row.append(f"{s['delta']:+.3f} (p {s['p_value']:.2f}) {_WORD[v]}")
        rows.append(row)
    rows.append(["**seeds**", ""] + [
        f"better {t.count('better')} · worse {t.count('worse')} · "
        f"same {t.count('same')}" for t in tally.values()])
    lines += _table(head, rows) + [""]

    lines += ["**Targets from the proposal**", ""]
    t = agg["targets"]
    fc = agg["conditions"].get("full_context", {}).get("accuracy", {}).get("mean")
    nr = agg["conditions"].get("naive_rag", {}).get("accuracy", {}).get("mean")
    rows = []
    if "estimator_auc" in t:
        e = t["estimator_auc"]
        rows.append(["estimator ROC-AUC", f"{e['mean']:.3f} ± {e['sd']:.3f}",
                     "≥ 0.70", "✔ met" if e["met"] else "✘ not met",
                     "how well the estimator ranks useful records above useless ones"])
    if "token_reduction" in t:
        e = t["token_reduction"]
        rows.append(["token reduction", f"{e['mean']:+.3f} ± {e['sd']:.3f}",
                     "≥ 0.40", "✔ met" if e["met"] else "✘ not met",
                     "1 − OCMG tokens / naive RAG tokens; negative = OCMG sends more"])
    if "accuracy_retention" in t:
        e = t["accuracy_retention"]
        no_ceiling = not e.get("full_context_is_ceiling", True)
        rows.append(["accuracy retention", f"{e['mean']:.3f} ± {e['sd']:.3f}",
                     "≥ 0.95", "⚠ not interpretable" if no_ceiling
                     else ("✔ met" if e["met"] else "✘ not met"),
                     (f"full context ({fc:.3f}) scores below naive RAG "
                      f"({nr:.3f}), so it is not an upper bound")
                     if no_ceiling and fc and nr else "OCMG acc / full-context acc"])
    lines += _table(["measure", "value", "target", "status", "meaning"], rows,
                    ["l", "r", "r", "l", "l"]) + [""]

    lv = agg.get("label_validation")
    if lv:
        lines += ["## 2 · Are the training labels right?", "",
                  "Labels come from leave-one-out outcomes; LoCoMo's evidence "
                  "ids are used only to check them, never to train.", ""]
        lines += _table(["measure", "value", "meaning"], [
            ["precision", f"{lv['precision']['mean']:.3f}",
             "records labelled useful that really hold evidence"],
            ["recall", f"{lv['recall']['mean']:.3f}",
             "evidence records that got labelled useful"],
            ["Cohen's kappa", f"{lv['kappa']['mean']:.3f}",
             "agreement beyond chance (0 = chance, 1 = perfect)"],
            ["estimator vs evidence", f"{lv['estimator_auc_vs_evidence']['mean']:.3f}",
             "ROC-AUC of the estimator's scores against evidence"],
        ], ["l", "r", "l"]) + [""]

    if rarity:
        ra = rarity["aggregate"]
        lines += ["## 3 · Rarity ablation: is the estimator just a word-rarity score?",
                  "", "Same pipeline, estimator replaced by a rarity heuristic.", ""]
        rows = []
        for name, res in (("learned estimator (main)", agg), ("rarity heuristic", ra)):
            o = res["conditions"]["ocmg"]
            rows.append([name, f"{res['targets']['estimator_auc']['mean']:.3f}",
                         f"{o['accuracy']['mean']:.3f} ± {o['accuracy']['sd']:.3f}",
                         f"{o['tokens_per_query']['mean']:,.0f}"])
        lines += _table(["OCMG with", "ROC-AUC", "OCMG accuracy", "tokens / question"],
                        rows) + ["", "*A higher AUC with lower accuracy means "
                                 "AUC alone does not predict system accuracy.*", ""]


def section_forget(fq, fc, lines, figs):
    if fq:
        lines += ["## 4 · Forgetting fraction: how much should be forgotten?", ""]
        rows = []
        for q, rs in fq.items():
            a = _ms([r["accuracy"] for r in rs])
            rows.append([f"{float(q.split('=')[1]):.0%}", f"{a[0]:.3f} ± {a[1]:.3f}",
                         f"{_ms([r['tokens'] for r in rs])[0]:,.0f}",
                         f"{_ms([r['ceiling'] for r in rs])[0]:.1%}",
                         f"{_ms([r['evicted'] for r in rs])[0]:.0f}"])
        lines += _table(["forgotten", "accuracy", "tokens / question",
                         "evidence still in memory", "records evicted"], rows)
        lines += ["", f"![forgetting fraction]({figs['ffrac']})", ""]
    if fc:
        lines += ["## 5 · Forgetting criterion: which signal should choose what to forget?", ""]
        rows = []
        base = {r["seed"]: r["accuracy"] for r in fc.get("ledger", [])}
        for arm, rs in fc.items():
            a = _ms([r["accuracy"] for r in rs])
            diff = (_ms([r["accuracy"] - base[r["seed"]] for r in rs])[0]
                    if base and arm != "ledger" else None)
            rows.append([arm, f"{a[0]:.3f} ± {a[1]:.3f}",
                         "-" if diff is None else f"{diff:+.3f}",
                         f"{_ms([r['ceiling'] for r in rs])[0]:.1%}",
                         f"{_ms([r['evicted'] for r in rs])[0]:.0f}"])
        lines += _table(["criterion", "accuracy", "vs ledger",
                         "evidence still in memory", "records evicted"], rows)
        lines += ["", f"![forgetting criterion]({figs['fcrit']})", ""]


def main() -> int:
    ap = argparse.ArgumentParser(description="one-page view of a run_all result set")
    ap.add_argument("--dir", default=os.path.join("results", "full"))
    args = ap.parse_args()

    main_r = _load(args.dir, "run_main")
    if main_r is None:
        print(f"no run_main.json in {args.dir}; run scripts/run_all.py first")
        return 1
    rarity = _load(args.dir, "run_rarity")
    fq = _load(args.dir, "forget_quantiles")
    fc = _load(args.dir, "forget_criterion")

    meta = main_r["meta"]
    reportable = meta.get("reader") != "simulated" and meta.get("embedder") != "hashing"
    fdir = os.path.join(args.dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    rel = lambda p: os.path.relpath(p, args.dir).replace(os.sep, "/")  # noqa: E731
    figs = {"cost": rel(chart_cost(main_r, fdir, reportable)),
            "evidence": rel(chart_evidence(main_r, fdir, reportable))}
    if fq:
        figs["ffrac"] = rel(chart_forget_fraction(fq, fdir, reportable))
    if fc:
        figs["fcrit"] = rel(chart_forget_criterion(fc, fdir, reportable))

    lines = ["# OCMG results", "",
             f"Dataset **{meta['dataset']}** · reader **{meta['reader']}** · "
             f"embedder **{meta['embedder']}** · estimator **{meta['estimator']}** · "
             f"**{main_r['aggregate']['n_runs']} seeds** · created {meta['created'][:16].replace('T', ' ')}",
             ""]
    if not reportable:
        lines += ["> ⚠ **Pipeline validation.** Simulated reader and hashing "
                  "embedder: these numbers test the code, they are not thesis "
                  "findings. Real numbers need `--reader hf --embedder "
                  "sentence-transformers` on a GPU.", ""]
    section_main(main_r, rarity, lines, figs)
    section_forget(fq, fc, lines, figs)
    lines += ["---", "",
              "*Generated by `scripts/make_results_page.py` from the JSON "
              f"files in `{args.dir}`. Full console log: "
              "`python scripts/run_all.py --save results/run_all_output.txt`.*"]

    out = os.path.join(args.dir, "RESULTS.md")
    with open(out, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"page     {out}   open in VS Code, then Ctrl+Shift+V for the formatted view")
    print(f"charts   {fdir}   {len(figs)} PNGs used by the page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
