#!/usr/bin/env python3
"""Build tables (CSV), figures (PNG, 300 dpi) and a Markdown summary from a run.

    python scripts/make_report.py
    python scripts/make_report.py --input results/run_locomo.json

Outputs go to results/tables, results/figures and results/summary.md.
If the run used the simulated reader or the hashing embedder, every figure
is stamped as pipeline validation, because those numbers are not findings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["full_context", "naive_rag", "append_all", "decay", "generative_judge", "ocmg"]
LABELS = {
    "full_context": "Full context",
    "naive_rag": "Naive RAG (top-k)",
    "append_all": "Append-all typed",
    "decay": "Decay-based",
    "generative_judge": "Generative judge",
    "ocmg": "OCMG",
}
CATEGORY_ORDER = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _mean_sd(runs: List[dict], name: str, key: str):
    vals = [r["conditions"][name][key] for r in runs if name in r["conditions"]]
    return float(np.mean(vals)), float(np.std(vals))


def _stamp(fig, reportable: bool) -> None:
    if not reportable:
        # Below the axes, outside the plot area; bbox_inches="tight" keeps it.
        fig.text(0.99, -0.04, "Pipeline validation (simulated reader) - not a finding",
                 ha="right", va="top", fontsize=7, color="#a33", style="italic")


def _write_csv(path: str, header: List[str], rows: List[list]) -> None:
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build tables and figures from a results file")
    ap.add_argument("--input", default=os.path.join("results", "run_locomo.json"))
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    with open(args.input, encoding="utf8") as fh:
        payload = json.load(fh)
    meta, runs = payload["meta"], payload["runs"]
    reportable = bool(meta.get("reportable"))
    names = [n for n in ORDER if n in runs[0]["conditions"]]

    tdir = os.path.join(args.outdir, "tables")
    fdir = os.path.join(args.outdir, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    # ------------------------------------------------------------ main table
    main_rows = []
    for n in names:
        acc, acc_sd = _mean_sd(runs, n, "accuracy")
        f1, _ = _mean_sd(runs, n, "token_f1")
        tok, _ = _mean_sd(runs, n, "tokens_per_query")
        apk, _ = _mean_sd(runs, n, "accuracy_per_1k_tokens")
        rec, _ = _mean_sd(runs, n, "records_retained")
        gen, _ = _mean_sd(runs, n, "generative_calls")
        main_rows.append([LABELS[n], f"{acc:.3f}", f"{acc_sd:.3f}", f"{f1:.3f}",
                          f"{tok:.1f}", f"{apk:.3f}", f"{rec:.0f}", f"{gen:.0f}"])
    _write_csv(os.path.join(tdir, "main_results.csv"),
               ["Condition", "Accuracy", "Accuracy SD", "Token F1", "Tokens per query",
                "Accuracy per 1k tokens", "Records retained", "Generative calls"], main_rows)

    # ------------------------------------------------------ per-category table
    cats = [c for c in CATEGORY_ORDER
            if any(c in runs[0]["conditions"][n]["per_category"] for n in names)]
    cat_rows = []
    for n in names:
        row = [LABELS[n]]
        for c in cats:
            vals = [r["conditions"][n]["per_category"].get(c) for r in runs]
            vals = [v for v in vals if v is not None]
            row.append(f"{np.mean(vals):.3f}" if vals else "")
        cat_rows.append(row)
    _write_csv(os.path.join(tdir, "per_category_accuracy.csv"),
               ["Condition"] + [c.replace("_", " ").title() for c in cats], cat_rows)

    # ------------------------------------------------------------ targets
    agg_t = payload.get("aggregate", {}).get("targets", {})
    target_rows = []
    for key, label in (("accuracy_retention", "Accuracy retention vs full context"),
                       ("token_reduction", "Token reduction vs fixed top-k"),
                       ("estimator_auc", "Utility estimator ROC-AUC")):
        if key in agg_t:
            t = agg_t[key]
            target_rows.append([label, f"{t['mean']:.3f}", f"{t['sd']:.3f}",
                                f">= {t['target']:.2f}", "Met" if t["met"] else "Not met"])
    _write_csv(os.path.join(tdir, "targets.csv"),
               ["Measure", "Value (mean)", "SD", "Target", "Status"], target_rows)

    # ------------------------------------------------------------ significance
    sig_rows = []
    for r in runs:
        for comp, s in r.get("significance", {}).items():
            sig_rows.append([r["seed"], comp, f"{s['delta']:+.3f}",
                             f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]", f"{s['p_value']:.3f}"])
    _write_csv(os.path.join(tdir, "significance.csv"),
               ["Seed", "Comparison", "Accuracy difference", "95% CI", "p-value"], sig_rows)

    # -------------------------------------------------- label validation
    # The audit of outcome labels against the held-out evidence annotation.
    # Reported separately from the targets because it validates the labels the
    # estimator was trained on, not the governed system's performance.
    val_rows = []
    agg_val = payload.get("aggregate", {}).get("label_validation", {})
    for key, label in (
        ("precision", "Precision of outcome labels vs evidence"),
        ("recall", "Recall of outcome labels vs evidence"),
        ("kappa", "Cohen's kappa (chance-corrected agreement)"),
        ("estimator_auc_vs_evidence", "Estimator ROC-AUC against evidence labels"),
        ("evidence_positive_rate", "Records containing annotated evidence"),
        ("outcome_positive_rate", "Records labelled useful by outcome credit"),
    ):
        if key in agg_val:
            val_rows.append([label, f"{agg_val[key]['mean']:.3f}",
                             f"{agg_val[key]['sd']:.3f}"])
    if val_rows:
        _write_csv(os.path.join(tdir, "label_validation.csv"),
                   ["Measure", "Value (mean)", "SD"], val_rows)

    # ------------------------------------------------------------ estimator
    est = runs[0].get("estimator")
    if est:
        fi = sorted(est["feature_importance"].items(), key=lambda kv: -abs(kv[1]))
        _write_csv(os.path.join(tdir, "feature_importance.csv"),
                   ["Feature", "Importance"], [[k, f"{v:.4f}"] for k, v in fi])

    # ============================================================== figures
    # Figure: accuracy against token cost
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    pts = []
    for n in names:
        acc, acc_sd = _mean_sd(runs, n, "accuracy")
        tok, _ = _mean_sd(runs, n, "tokens_per_query")
        pts.append((n, tok, acc, acc_sd))
        style = dict(s=80, zorder=3, color="#c77c02" if n == "ocmg" else "#33507a",
                     marker="*" if n == "ocmg" else "o")
        ax.scatter(tok, acc, **style)
        ax.errorbar(tok, acc, yerr=acc_sd, fmt="none", ecolor="#999", zorder=2)
    # Stack labels from the highest point down so neighbours never overlap.
    accs = [p[2] for p in pts]
    step = max(0.035 * (max(accs) - min(accs) + 0.05), 0.006)
    labelled: List[tuple] = []
    for n, tok, acc, _ in sorted(pts, key=lambda p: -p[2]):
        y = acc
        for t0, y0 in labelled:
            if abs(np.log10(max(tok, 1)) - np.log10(max(t0, 1))) < 0.15 and y > y0 - step:
                y = y0 - step
        labelled.append((tok, y))
        ax.annotate(LABELS[n], xy=(tok, acc), xytext=(tok * 1.12, y),
                    textcoords="data", fontsize=8, va="center",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6)
                    if abs(y - acc) > 1e-9 else None)
    ax.set_xscale("log")
    ax.set_xlabel("Prompt tokens per query (log scale)")
    ax.set_ylabel("Answer accuracy")
    ax.set_title("Accuracy against token cost")
    ax.grid(alpha=0.3, which="both")
    _stamp(fig, reportable)
    fig.savefig(os.path.join(fdir, "accuracy_vs_tokens.png"))
    plt.close(fig)

    # Figure: accuracy by condition
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    means = [_mean_sd(runs, n, "accuracy")[0] for n in names]
    sds = [_mean_sd(runs, n, "accuracy")[1] for n in names]
    colors = ["#c77c02" if n == "ocmg" else "#33507a" for n in names]
    ax.bar([LABELS[n] for n in names], means, yerr=sds, color=colors, capsize=3)
    ax.set_ylabel("Answer accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by condition")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _stamp(fig, reportable)
    fig.savefig(os.path.join(fdir, "accuracy_by_condition.png"))
    plt.close(fig)

    # Figure: per-category accuracy
    if cats:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        width = 0.8 / len(names)
        x = np.arange(len(cats))
        for i, n in enumerate(names):
            vals = []
            for c in cats:
                v = [r["conditions"][n]["per_category"].get(c) for r in runs]
                v = [a for a in v if a is not None]
                vals.append(np.mean(v) if v else 0.0)
            ax.bar(x + i * width, vals, width, label=LABELS[n],
                   color=plt.cm.viridis(i / max(1, len(names) - 1)))
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels([c.replace("_", " ").title() for c in cats])
        ax.set_ylabel("Answer accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy by question category")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        _stamp(fig, reportable)
        fig.savefig(os.path.join(fdir, "per_category_accuracy.png"))
        plt.close(fig)

    # Figure: estimator fold AUCs and feature importance
    if est:
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        folds = est["auc_folds"]
        ax.bar([f"Fold {i + 1}" for i in range(len(folds))], folds, color="#33507a")
        ax.axhline(0.5, color="#888", linestyle=":", label="Chance (0.50)")
        ax.axhline(0.70, color="#c77c02", linestyle="--", label="Target (0.70)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"Utility estimator ROC-AUC (mean {est['auc_mean']:.3f})")
        ax.legend(fontsize=8)
        _stamp(fig, reportable)
        fig.savefig(os.path.join(fdir, "estimator_auc_folds.png"))
        plt.close(fig)

        top = fi[:10][::-1]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.barh([k for k, _ in top], [v for _, v in top],
                color=["#c77c02" if v > 0 else "#33507a" for _, v in top])
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_xlabel("Importance (signed)")
        ax.set_title("Top 10 estimator features")
        _stamp(fig, reportable)
        fig.savefig(os.path.join(fdir, "feature_importance.png"))
        plt.close(fig)

    # ============================================================ summary.md
    lines = [
        "# OCMG results summary", "",
        f"- Created: {meta.get('created')}",
        f"- Dataset: {meta.get('dataset')} ({meta.get('path')})",
        f"- Embedder: {meta.get('embedder')}  |  Reader: {meta.get('reader')}  |  "
        f"Estimator: {meta.get('estimator')}  |  Credit: {meta.get('credit')}",
        f"- Runs (seeds): {meta.get('runs')}",
        "",
    ]
    if not reportable:
        lines += ["> **Pipeline validation only.** This run used the simulated reader and/or "
                  "the hashing embedder. Use these numbers to check the pipeline, not as "
                  "thesis findings.", ""]
    lines += ["## Main results", "",
              "| Condition | Accuracy | SD | Token F1 | Tokens/query | Acc per 1k tokens | Records | Gen. calls |",
              "|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in main_rows]
    lines += ["", "## Targets", "", "| Measure | Value | SD | Target | Status |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in target_rows]
    if val_rows:
        lines += ["", "## Label validation against annotated evidence", "",
                  "Outcome-derived utility labels scored against the benchmark "
                  "evidence annotations, which are held out and never used for "
                  "training. Low recall is expected: a record can support an "
                  "answer and still never be the record whose removal changed "
                  "it. Low precision would be the result that refutes the "
                  "method.", "",
                  "| Measure | Value | SD |", "|---|---|---|"]
        lines += ["| " + " | ".join(r) + " |" for r in val_rows]

    lines += ["", "## Figures", ""]
    for f in sorted(os.listdir(fdir)):
        lines.append(f"- figures/{f}")
    with open(os.path.join(args.outdir, "summary.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"tables  -> {tdir}")
    print(f"figures -> {fdir}")
    print(f"summary -> {os.path.join(args.outdir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
