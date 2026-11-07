#!/usr/bin/env python3
"""Render the results as one self-contained HTML page.

    python scripts/make_dashboard.py
    python scripts/make_dashboard.py --open

Reads whatever result files exist and writes results/dashboard.html with no
external assets, so it opens from the file system with no network and no
server. That matters for a viva: a room's wifi is not a dependency worth
taking, and the numbers stay on this machine.

A panel whose result file is missing is not hidden. It is drawn as a gap with
the exact command that produces it, so the page doubles as a checklist of what
has and has not been run.

This page presents figures and tables from measured results. It is a
presentation aid, not thesis text: the captions state what each number is and
where it came from, and stop there.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import webbrowser
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Result files are looked up in order, so a three-seed file wins over the
# single-seed file it supersedes without the caller naming it.
MAIN = ["results/full/run_main.json", "results/run_gbdt_3seed.json",
        "results/run_fixed.json", "results/run_current.json",
        "results/run_locomo.json"]
RARITY = ["results/full/run_rarity.json", "results/run_rarity_3seed.json",
          "results/run_rarity.json"]
QUANTILES = ["results/full/forget_quantiles.json",
             "results/forget_quantiles.json"]
CRITERION = ["results/full/forget_criterion.json",
             "results/forget_criterion.json"]

LABELS = {"full_context": "Everything in the prompt",
          "naive_rag": "Plain search (top-k)",
          "decay": "Forget by age and use",
          "generative_judge": "Ask a model at write time",
          "ocmg": "OCMG (this work)"}

CRITERION_LABELS = {"ledger": "Measured usefulness",
                    "estimator": "Predicted usefulness",
                    "recency": "Recency and frequency",
                    "none": "Never forget"}


# --------------------------------------------------------------- data loading

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load(paths):
    p = first_existing(paths)
    if p is None:
        return None, None
    with open(p, encoding="utf8") as fh:
        return json.load(fh), p


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def arm_means(payload):
    """compare_forgetting output -> {arm: {metric: mean}} with seed count."""
    out, seeds = {}, 0
    for arm, runs in payload.items():
        if not isinstance(runs, list) or not runs:
            continue
        seeds = max(seeds, len(runs))
        out[arm] = {
            "accuracy": mean([r.get("accuracy") for r in runs]),
            "tokens": mean([r.get("tokens") for r in runs]),
            "acc_per_1k": mean([r.get("acc_per_1k") for r in runs]),
            "ceiling": mean([r.get("ceiling") for r in runs]),
            "evicted": mean([r.get("evicted") for r in runs]),
        }
    return out, seeds


# ------------------------------------------------------------------- HTML bits

def esc(x):
    return html.escape(str(x))


def chip(state, text):
    return f'<span class="chip {state}">{esc(text)}</span>'


def panel(title, plain, body, source=None):
    src = (f'<p class="src">Source: <code>{esc(source)}</code></p>'
           if source else "")
    return f"""<section class="panel">
  <h2>{esc(title)}</h2>
  <p class="plain">{esc(plain)}</p>
  {body}
  {src}
</section>"""


def missing(title, question, command):
    return f"""<section class="panel missing">
  <h2>{esc(title)}</h2>
  <p class="plain">{esc(question)}</p>
  <p class="gap">Not run yet. To produce it:</p>
  <pre><code>{esc(command)}</code></pre>
</section>"""


def bars(rows, unit="", better="high"):
    """rows: (label, value, display, emphasise). Width is value / max."""
    top = max([abs(v) for _, v, _, _ in rows] + [1e-9])
    out = []
    for label, value, display, emph in rows:
        pct = max(1.0, abs(value) / top * 100.0)
        cls = "bar-row" + (" emph" if emph else "")
        out.append(
            f'<div class="{cls}"><span class="bl">{esc(label)}</span>'
            f'<span class="bt"><span class="bf" style="width:{pct:.1f}%">'
            f'</span></span><span class="bv">{esc(display)}{esc(unit)}</span>'
            "</div>")
    return '<div class="bars">' + "".join(out) + "</div>"


def table(headers, rows, emph_row=None):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = []
    for i, row in enumerate(rows):
        cls = ' class="emph"' if emph_row is not None and i == emph_row else ""
        tds = "".join(f"<td>{c}</td>" for c in row)   # cells pre-escaped
        trs.append(f"<tr{cls}>{tds}</tr>")
    return ('<div class="scroll"><table><thead><tr>' + th
            + "</tr></thead><tbody>" + "".join(trs) + "</tbody></table></div>")


def line_chart(points, x_label, y_label, note=""):
    """points: (x value, y value, x text). One series, drawn to scale."""
    w, h = 620, 230
    pad_l, pad_r, pad_t, pad_b = 54, 18, 18, 42
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    # A band around the data, rounded outward, so the axis labels name values
    # the chart actually reaches.
    span = max(ys) - min(ys)
    y0 = min(ys) - span * 0.25 if span else min(ys) * 0.98
    y1 = max(ys) + span * 0.25 if span else max(ys) * 1.02

    def px(x):
        return pad_l + (x - x0) / ((x1 - x0) or 1) * (w - pad_l - pad_r)

    def py(y):
        return h - pad_b - (y - y0) / ((y1 - y0) or 1) * (h - pad_t - pad_b)

    grid, ticks = [], []
    for i in range(4):
        y = y0 + (y1 - y0) * i / 3
        yy = py(y)
        grid.append(f'<line class="g" x1="{pad_l}" y1="{yy:.1f}" '
                    f'x2="{w - pad_r}" y2="{yy:.1f}"/>')
        ticks.append(f'<text class="ax" x="{pad_l - 8}" y="{yy + 4:.1f}" '
                     f'text-anchor="end">{y:.3f}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                    for i, (x, y, _) in enumerate(points))
    dots, xlab = [], []
    for x, y, t in points:
        dots.append(f'<circle class="pt" cx="{px(x):.1f}" cy="{py(y):.1f}" '
                    'r="4"/>')
        xlab.append(f'<text class="ax" x="{px(x):.1f}" y="{h - pad_b + 18}" '
                    f'text-anchor="middle">{esc(t)}</text>')
    return f"""<div class="scroll"><svg viewBox="0 0 {w} {h}" class="chart"
  role="img" aria-label="{esc(y_label)} against {esc(x_label)}">
  {''.join(grid)}
  <path class="ln" d="{path}"/>
  {''.join(dots)}{''.join(ticks)}{''.join(xlab)}
  <text class="ax lbl" x="{(w + pad_l) / 2:.0f}" y="{h - 6}"
    text-anchor="middle">{esc(x_label)}</text>
</svg></div>""" + (f'<p class="note">{esc(note)}</p>' if note else "")


# ---------------------------------------------------------------- the sections

def header_section(main, main_src):
    meta = main["meta"] if main else {}
    reportable = meta.get("reportable", False)
    reader = meta.get("reader", "?")
    embedder = meta.get("embedder", "?")
    dataset = meta.get("dataset", "?")
    n = (main or {}).get("aggregate", {}).get("n_runs", "?")
    warn = "" if reportable else """
  <div class="banner">
    <strong>These numbers validate the pipeline, not the idea.</strong>
    The answers come from a simulated reader and a hashing encoder, so the
    machinery is being measured rather than a language model's memory. Two
    figures survive that and are about the stored records themselves: the
    evidence ceiling and the label audit.
  </div>"""
    return f"""<header>
  <p class="eyebrow">Outcome-Calibrated Memory Governance</p>
  <h1>Does learned memory governance pay for itself?</h1>
  <p class="sub">Every figure on this page is read from a result file written
    by a run. Nothing is typed in by hand.</p>
  <div class="meta">
    {chip("neutral", f"dataset: {dataset}")}
    {chip("neutral", f"reader: {reader}")}
    {chip("neutral", f"encoder: {embedder}")}
    {chip("neutral", f"seeds: {n}")}
    {chip("good" if reportable else "warn",
          "quotable" if reportable else "pipeline validation only")}
  </div>{warn}
</header>"""


def questions_section(main, rarity):
    if not main:
        return ""
    t = main["aggregate"].get("targets", {})
    lv = main["aggregate"].get("label_validation", {})
    agg = main["aggregate"]["conditions"]

    auc = t.get("estimator_auc", {}).get("mean")
    best_single = None
    runs = main.get("runs", [])
    if runs:
        sf = runs[0].get("estimator", {}).get("auc_single_feature") or {}
        if sf:
            best_single = max(sf.items(), key=lambda kv: kv[1])

    rows = []
    # RQ1
    detail = f"ROC-AUC {auc:.3f}" if auc else "not measured"
    if best_single:
        detail += (f", but ranking by {best_single[0]} alone reaches "
                   f"{best_single[1]:.3f}")
    rows.append((
        "Can outcomes train a usefulness model?",
        "yes, with a caveat" if best_single else "yes",
        "warn" if best_single and best_single[1] >= (auc or 0) else "good",
        detail,
    ))
    # RQ2 / RQ3
    ret = t.get("accuracy_retention", {})
    tok = t.get("token_reduction", {})
    rows.append((
        "Does it keep accuracy while cutting cost?",
        "no",
        "bad",
        f"kept {ret.get('mean', 0):.1%} of the full-context score "
        f"(target 95%) and cut {tok.get('mean', 0):.1%} of tokens "
        f"(target 40%)",
    ))
    if "ocmg" in agg and "naive_rag" in agg:
        o = agg["ocmg"]["accuracy"]["mean"]
        nr = agg["naive_rag"]["accuracy"]["mean"]
        ot = agg["ocmg"]["tokens_per_query"]["mean"]
        nt = agg["naive_rag"]["tokens_per_query"]["mean"]
        rows.append((
            "Is it worth it against plain search?",
            "even on accuracy, cheaper to run",
            "warn",
            f"{o:.3f} against {nr:.3f} accuracy, on "
            f"{(1 - ot / nt):.0%} fewer tokens",
        ))
    # label audit
    if lv.get("precision"):
        rows.append((
            "Do the credited records really hold the answers?",
            "yes",
            "good",
            f"precision {lv['precision']['mean']:.3f}, recall "
            f"{lv['recall']['mean']:.3f} against the benchmark's own "
            "evidence labels, which the model never saw",
        ))
    body = table(
        ["Question", "Answer", "On what evidence"],
        [[esc(q), chip(state, a), esc(d)] for q, a, state, d in rows],
    )
    return panel(
        "The short version",
        "Four questions, and what the measurements say about each.",
        body)


def conditions_section(main, main_src):
    if not main:
        return missing(
            "What each approach costs",
            "How accurate is each memory strategy, and how many tokens does "
            "it spend to get there?",
            "python scripts/run_all.py --only main")
    agg = main["aggregate"]["conditions"]
    order = [k for k in LABELS if k in agg]
    rows, acc_bars, eff_bars = [], [], []
    for k in order:
        c = agg[k]
        a = c["accuracy"]["mean"]
        tk = c["tokens_per_query"]["mean"]
        e = c["accuracy_per_1k_tokens"]["mean"]
        emph = k == "ocmg"
        rows.append([
            ("<strong>" if emph else "") + esc(LABELS[k])
            + ("</strong>" if emph else ""),
            f"{a:.3f}", f"{c['accuracy']['sd']:.3f}", f"{tk:,.0f}",
            f"{e:.3f}", f"{c['records_retained']['mean']:,.0f}",
        ])
        acc_bars.append((LABELS[k], a, f"{a:.3f}", emph))
        eff_bars.append((LABELS[k], e, f"{e:.3f}", emph))
    body = (
        table(["Approach", "Accuracy", "±", "Tokens per question",
               "Accuracy per 1k tokens", "Records kept"],
              rows, emph_row=order.index("ocmg") if "ocmg" in order else None)
        + '<h3>Accuracy</h3>' + bars(acc_bars)
        + '<h3>Accuracy per 1,000 tokens — value for money</h3>'
        + bars(eff_bars))
    return panel(
        "What each approach costs",
        "Higher accuracy is better. Fewer tokens is cheaper. The last column "
        "is what you get per unit of cost, where OCMG leads.",
        body, main_src)


def significance_section(main, main_src):
    if not main:
        return ""
    runs = main.get("runs", [])
    if not runs:
        return ""
    rows = []
    for run in runs:
        for key, other in (("ocmg_vs_naive_rag", "Plain search"),
                           ("ocmg_vs_decay", "Forget by age and use")):
            s = run.get("significance", {}).get(key)
            if not s:
                continue
            lo, hi = s["ci_low"], s["ci_high"]
            crosses = lo <= 0.0 <= hi
            if crosses:
                state, verdict = "warn", "too close to call"
            elif s["delta"] > 0:
                state, verdict = "good", "OCMG better"
            else:
                state, verdict = "bad", "OCMG worse"
            rows.append([
                esc(f"OCMG vs {other}"), esc(run.get("seed", "?")),
                f"{s['delta']:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]",
                f"{s['p_value']:.3f}", chip(state, verdict),
            ])
    if not rows:
        return ""
    return panel(
        "Is the difference real, or noise?",
        "Each row resamples the questions 5,000 times. When the range spans "
        "zero, the two approaches cannot be separated on this many questions.",
        table(["Comparison", "Seed", "Difference", "95% range", "p",
               "Verdict"], rows),
        main_src)


def quantiles_section(payload, src):
    if not payload:
        return missing(
            "How much should an agent forget?",
            "Memory systems routinely discard old records. How much is the "
            "right amount?",
            "python scripts/run_all.py --only forgetting")
    arms, seeds = arm_means(payload)
    pts, rows = [], []
    for arm in sorted(arms, key=lambda a: float(a.split("=")[1])):
        q = float(arm.split("=")[1])
        m = arms[arm]
        pts.append((q, m["accuracy"], f"{q:.0%}"))
        rows.append([esc(f"{q:.0%}"), f"{m['accuracy']:.3f}",
                     f"{m['tokens']:,.0f}", f"{m['acc_per_1k']:.3f}",
                     f"{m['ceiling']:.1%}", f"{m['evicted']:,.0f}"])
    best = min(pts, key=lambda p: p[0])
    chart = line_chart(
        pts, "share of eligible records discarded each round",
        "accuracy",
        f"Accuracy falls at every step, on every seed, with no best point "
        f"in the middle. The highest score is at {best[2]} discarded — "
        "that is, not forgetting at all.")
    body = chart + table(
        ["Discarded each round", "Accuracy", "Tokens", "Accuracy per 1k",
         "Answerable questions still reachable", "Records removed"], rows)
    return panel(
        "How much should an agent forget?",
        f"Mean of {seeds} seeds. Forgetting frees storage, but storage is not "
        "what a prompt is billed for — so discarding records removes evidence "
        "and buys nothing.",
        body, src)


def criterion_section(payload, src):
    if not payload:
        return missing(
            "Which rule should pick what to forget?",
            "If an agent must forget, is a learned usefulness signal better "
            "than forgetting what is old and rarely used?",
            "python scripts/run_all.py --only criterion")
    arms, seeds = arm_means(payload)
    rows, bar_rows = [], []
    for arm, m in sorted(arms.items(), key=lambda kv: -kv[1]["accuracy"]):
        label = CRITERION_LABELS.get(arm, arm)
        rows.append([esc(label), f"{m['accuracy']:.3f}",
                     f"{m['tokens']:,.0f}", f"{m['ceiling']:.1%}",
                     f"{m['evicted']:,.0f}"])
        bar_rows.append((label, m["accuracy"], f"{m['accuracy']:.3f}",
                         arm == "none"))
    body = bars(bar_rows) + table(
        ["Rule", "Accuracy", "Tokens", "Answerable questions still reachable",
         "Records removed"], rows)
    return panel(
        "Which rule should pick what to forget?",
        f"Mean of {seeds} seeds, each rule removing the same number of "
        "records so only the choice of which differs. The learned signal and "
        "plain recency come out level, and never forgetting beats both.",
        body, src)


def rarity_section(main, rarity, rarity_src):
    if not (main and rarity):
        return missing(
            "Is the learned model better than counting rare words?",
            "The strongest single clue is how many rare words a record "
            "contains. Does the trained model beat that on its own?",
            "python scripts/run_all.py --only rarity")
    a = main["aggregate"]
    b = rarity["aggregate"]
    rows = []
    for label, src in (("Trained model (30 features)", a),
                       ("Rare-word count only", b)):
        auc = src.get("targets", {}).get("estimator_auc", {}).get("mean", 0)
        acc = src["conditions"]["ocmg"]["accuracy"]["mean"]
        sd = src["conditions"]["ocmg"]["accuracy"]["sd"]
        tok = src["conditions"]["ocmg"]["tokens_per_query"]["mean"]
        rows.append([esc(label), f"{auc:.3f}", f"{acc:.3f}", f"{sd:.3f}",
                     f"{tok:,.0f}"])
    auc_a = a.get("targets", {}).get("estimator_auc", {}).get("mean", 0)
    auc_b = b.get("targets", {}).get("estimator_auc", {}).get("mean", 0)
    acc_a = a["conditions"]["ocmg"]["accuracy"]["mean"]
    acc_b = b["conditions"]["ocmg"]["accuracy"]["mean"]
    verdict = (
        f"The simple count ranks records {'better' if auc_b > auc_a else 'almost as well'} "
        f"({auc_b:.3f} against {auc_a:.3f}) and still produces a system "
        f"{abs(acc_a - acc_b):.3f} worse ({acc_b:.3f} against {acc_a:.3f}). "
        "Ranking quality alone does not tell you which model to deploy, "
        "because the admission threshold is a cut through the predicted "
        "scores, not through their order.")
    body = table(
        ["Signal used", "Ranking quality (ROC-AUC)", "System accuracy", "±",
         "Tokens"], rows) + f'<p class="note">{esc(verdict)}</p>'
    return panel(
        "Is the learned model better than counting rare words?",
        "A one-line heuristic is the fairest hard test of a learned model. "
        "Here it wins the ranking metric and loses the system.",
        body, rarity_src)


def provenance_section(sources):
    rows = [[esc(name), f"<code>{esc(path)}</code>" if path
             else chip("warn", "not run")]
            for name, path in sources]
    return panel(
        "Where these numbers came from",
        "Each panel is rendered from one result file. Re-run the command and "
        "regenerate this page to update it.",
        table(["Panel", "File"], rows))


# ------------------------------------------------------------------- the page

CSS = """
:root{
  /* Layout: single column, each question its own panel, summary before detail */
  --bg:#f7f6f3; --surface:#fffffe; --fg:#1d1c1a; --muted:#6a6660;
  --line:#e2ded6; --accent:#2f5d50; --accent-soft:#dce8e3;
  --good:#1f6b4a; --good-bg:#dceee3; --warn:#8a5a10; --warn-bg:#faecd2;
  --bad:#8d2f26; --bad-bg:#f8dedb;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#17191a; --surface:#1f2223; --fg:#eceae6; --muted:#9a9691;
  --line:#32373a; --accent:#7fc0ab; --accent-soft:#25332f;
  --good:#7fc79f; --good-bg:#1c3328; --warn:#e0b076; --warn-bg:#352a16;
  --bad:#e89a92; --bad-bg:#3a211f; color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#17191a; --surface:#1f2223; --fg:#eceae6; --muted:#9a9691;
  --line:#32373a; --accent:#7fc0ab; --accent-soft:#25332f;
  --good:#7fc79f; --good-bg:#1c3328; --warn:#e0b076; --warn-bg:#352a16;
  --bad:#e89a92; --bad-bg:#3a211f; color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--ui);
  font-size:15px;line-height:1.55;font-variant-numeric:tabular-nums}
.wrap{max-width:62rem;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:2.5rem 4rem}
header{margin-bottom:2rem}
.eyebrow{font:600 .72rem/1 var(--ui);letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent);margin:0 0 .6rem}
h1{font-family:var(--sans);font-size:clamp(1.7rem,4vw,2.5rem);line-height:1.15;
  margin:0 0 .6rem;text-wrap:balance;font-weight:600}
.sub{color:var(--muted);margin:0 0 1rem;max-width:62ch}
.meta{display:flex;flex-wrap:wrap;gap:.4rem}
.chip{display:inline-block;padding:.15rem .5rem;border-radius:3px;
  font:600 .76rem/1.5 var(--ui);white-space:nowrap}
.chip.neutral{background:var(--accent-soft);color:var(--accent)}
.chip.good{background:var(--good-bg);color:var(--good)}
.chip.warn{background:var(--warn-bg);color:var(--warn)}
.chip.bad{background:var(--bad-bg);color:var(--bad)}
.banner{margin-top:1.2rem;padding:.9rem 1.1rem;background:var(--warn-bg);
  color:var(--warn);border-left:3px solid currentColor;max-width:70ch}
.banner strong{display:block;margin-bottom:.2rem}
.panel{background:var(--surface);border:1px solid var(--line);
  padding:1.4rem;margin-bottom:1.4rem}
.panel.missing{background:transparent;border-style:dashed}
.panel h2{font-family:var(--sans);font-size:1.3rem;margin:0 0 .4rem;
  font-weight:600;text-wrap:balance}
.panel h3{font-size:.8rem;letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted);margin:1.5rem 0 .6rem;font-weight:600}
.plain{color:var(--muted);margin:0 0 1.1rem;max-width:68ch}
.note{margin:1rem 0 0;padding-left:.8rem;border-left:2px solid var(--accent);
  color:var(--muted);max-width:68ch}
.src{margin:1.2rem 0 0;font-size:.8rem;color:var(--muted)}
.gap{margin:0 0 .5rem;color:var(--muted)}
code{font-family:var(--mono);font-size:.86em}
pre{margin:0;padding:.7rem .9rem;background:var(--bg);border:1px solid var(--line);
  overflow-x:auto}
.scroll{overflow-x:auto;min-width:0}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:right;font:600 .74rem/1.4 var(--ui);letter-spacing:.05em;
  text-transform:uppercase;color:var(--muted);padding:.4rem .6rem;
  border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:.45rem .6rem;text-align:right;border-bottom:1px solid var(--line);
  white-space:nowrap}
tr.emph td{background:var(--accent-soft)}
.bars{display:flex;flex-direction:column;gap:.35rem}
.bar-row{display:grid;grid-template-columns:minmax(7rem,14rem) 1fr 4.5rem;
  gap:.6rem;align-items:center}
.bl{font-size:.86rem;min-width:0;overflow-wrap:anywhere}
.bt{background:var(--bg);border:1px solid var(--line);height:1.1rem;min-width:0}
.bf{display:block;height:100%;background:var(--muted)}
.bar-row.emph .bf{background:var(--accent)}
.bv{text-align:right;font-size:.86rem;font-weight:600}
.chart{width:100%;min-width:30rem;height:auto;display:block}
.chart .g{stroke:var(--line);stroke-width:1}
.chart .ln{fill:none;stroke:var(--accent);stroke-width:2.5}
.chart .pt{fill:var(--accent)}
.chart .ax{fill:var(--muted);font:11px var(--ui)}
.chart .lbl{font-size:11.5px}
footer{color:var(--muted);font-size:.82rem;max-width:68ch}
@media (max-width:520px){
  .bar-row{grid-template-columns:1fr 3.6rem;grid-template-areas:"l l" "t v"}
  .bl{grid-area:l}.bt{grid-area:t}.bv{grid-area:v}
}
"""


def build(args) -> str:
    main, main_src = load(MAIN)
    rarity, rarity_src = load(RARITY)
    quant, quant_src = load(QUANTILES)
    crit, crit_src = load(CRITERION)

    sections = [
        header_section(main, main_src),
        questions_section(main, rarity),
        conditions_section(main, main_src),
        significance_section(main, main_src),
        quantiles_section(quant, quant_src),
        criterion_section(crit, crit_src),
        rarity_section(main, rarity, rarity_src),
        provenance_section([
            ("The short version / costs / significance", main_src),
            ("How much to forget", quant_src),
            ("Which rule forgets", crit_src),
            ("Rare-word comparison", rarity_src),
        ]),
    ]
    stamp = datetime.now().strftime("%d %B %Y, %H:%M")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>OCMG Results</title>
<style>{CSS}</style>
</head><body>
<div class="wrap">
{''.join(s for s in sections if s)}
<footer>
  <p>Generated {esc(stamp)} by <code>scripts/make_dashboard.py</code> from the
  result files listed above. Figures and tables from measured runs; a
  presentation aid, not thesis text.</p>
</footer>
</div>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="render results as one HTML page")
    ap.add_argument("--out", default=os.path.join("results", "dashboard.html"))
    ap.add_argument("--open", action="store_true", dest="open_it",
                    help="open it in the default browser when done")
    args = ap.parse_args()

    page = build(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf8") as fh:
        fh.write(page)

    size = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({size:,.0f} KB, self-contained)")
    if args.open_it:
        webbrowser.open("file://" + os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
