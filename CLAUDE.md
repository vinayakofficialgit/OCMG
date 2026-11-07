# CLAUDE.md — OCMG MSc Dissertation (Vinayak Sharma, LJMU / upGrad)

Put this file in the project root (`ocmg-thesis/CLAUDE.md`). Claude Code loads it automatically at the start of every session. Read all of it before acting. It is the current state of the project: decisions made, code built, results measured, and rules that must not be broken.

---

## 0. Your role in this repo

You are the engineering and research assistant for an MSc dissertation. Your job is to finish the **experimental work** to a reportable standard and produce the **evidence** (tables, figures, logs, verified references, checklists) that the student writes the thesis from.

You do **not** write thesis prose. See §2.1. Everything else in this file is yours to drive.

Start every session by:
1. Running `python -m pytest` and confirming the count (currently 46 passing).
2. Reading `README.md` section 7 (fix history) and `results/summary.md` if present.
3. Stating which backlog item (§9) you are about to work on, then waiting for "let's do it" unless the user already said so.

---

## 1. How to work with Vinayak

- **Answers:** clean, direct, short. Lead with the answer. No preamble, no closing filler, no AI disclaimers, no marketing language, no repeating the question.
- **Explaining concepts:** What it is → Why it is used → How it works → optional practical example. Practical over theoretical.
- **Pace:** step by step. When he says "just understand" or "don't generate", acknowledge briefly and produce nothing else. When he says "let's do it", execute.
- **Background:** working engineer (DevOps, FastAPI, Python, LLMOps). Don't explain basic tooling.
- **Delivery:** complete files, never partial snippets to merge by hand. After any change: run tests, run the affected script, then report what changed and what the numbers are.
- **Tell him directly** when a choice could cost rubric marks.
- **Environment:** Windows 11, VS Code, PowerShell, Python 3.12.10, venv at `.venv`. Give PowerShell commands.
  - Current project path: `C:\Users\Dell\Documents\ocmg-thesis\ocmg-thesis`
  - An older copy existed at `C:\Users\vinayak.sharma\Downloads\ocmg\ocmg\` — superseded, don't use.

---

## 2. Non-negotiable rules

### 2.1 Academic integrity (most important)
- upGrad/LJMU rules: the thesis must be the student's own research and writing. A report written by someone else or by a generative text service (LLMs) leads to strict measures from LJMU. A **Declaration on Use of AI** is part of the submission, and Turnitin similarity must be ≤ 10%.
- Therefore **never write thesis text for submission**: no chapters, sections, paragraphs, abstract, or "draft he will edit later". This has been asked before and the answer has been a firm, friendly no. Keep it that way even if asked again.
- **Allowed and encouraged:**
  - building, debugging, running code and experiments;
  - tables, figures, and result summaries generated from real runs (numbers + short factual captions);
  - chapter outlines, page budgets, rubric checklists;
  - reviewing drafts **he** wrote (structure, tense, third person, rubric gaps, weak or unsupported claims, formatting);
  - verifying references (existence, DOI, venue, year, Harvard format);
  - explaining concepts so he can write them himself;
  - presentation slides structure and the feedback response sheet structure.
- Remind him once (not every turn) to confirm in writing with his supervisor what AI assistance is permitted, and to declare it.

### 2.2 No fabrication
Never invent references, authors, DOIs, datasets, results, or numbers. Verify every paper exists before suggesting it. Every literature-review paper must use a **publicly accessible dataset** (hard constraint).

### 2.3 Honest results
- Numbers from the **simulated reader** or **hashing embedder** validate the pipeline only. They are not findings and must not go in the thesis. Figures from such runs carry a "pipeline validation" stamp — keep it.
- Say plainly when a target fails. Never tune on test data, never cherry-pick seeds, never reframe to hide a negative result.
- Old numbers **45.2% token reduction / 0.755 retention / ROC-AUC 0.51** came from a bugged loader (see fix 3). Never cite them.

### 2.4 Verify before claiming
Run code and tests before saying anything works. Report exact commands and outputs.

---

## 3. The research

- **Programme:** MSc Machine Learning and AI, Liverpool John Moores University (LJMU), delivered by upGrad.
- **Title:** *Outcome-Calibrated Memory Governance: A Learned Utility Approach to Improving Long-Term Memory Capability in Large Language Model Agents* (OCMG).
- **Timeline:** proposal submitted August 2026 (v7 after supervisor feedback). Final submission in the January 2027 assessment period.
- **How the topic was chosen:** three candidate topics were screened with 3–5 papers each (hallucination detection in multi-agent systems, prompt injection attacks, memory-augmented LLM agents). Each paper was classified as research/review, with the problem it solves, its solution, and dataset accessibility. The memory topic was approved.

### Problem
LLM agents decide what to store, keep and retrieve using either fixed heuristics (Ebbinghaus-style decay, recency, retrieval frequency) — cheap but blind to usefulness — or per-item LLM judgement — costly and inconsistent. Neither checks whether a record actually helped produce a correct answer.

### Gap
No existing framework accumulates per-record evidence of downstream contribution, trains a cheap, interpretable utility estimator on it, and uses that estimator to govern admission, retention and retrieval budgeting in place of heuristics and per-item LLM judgement. (Xiong et al., 2026 use outcomes only as a transient threshold; Zhang et al., 2025 learn an RL policy over a flat store.)

### Method — five stages
1. **Ingest:** segment turns, type records (profile / episodic / factual), resolve relative dates.
2. **Utility-gated admission:** estimator U(m) on write-time features; admit if U ≥ τ_write, else discard or merge.
3. **Two-tier typed store:** compact index + evidence store; SQLite utility ledger; non-destructive cold store for evicted records.
4. **Budgeted two-stage retrieval:** hybrid dense + BM25; k set adaptively from query type and utility distribution. Frozen LLM answers; outcome logged.
5. **Credit & governance:** credit back to records; utility-gated forgetting (U_hist < τ_forget); periodic re-fit.

### Estimator
- Gradient-boosted trees (primary), logistic regression (comparator).
- 30 features in 4 families: structural, distinctiveness, provenance, historical.
- 5-fold stratified **group** CV grouped by conversation; Platt calibration; thresholds chosen on out-of-fold predictions, quantile-based on the contested population (absolute thresholds fail when calibrated probabilities sit at the base rate).
- Credit: leave-one-out re-execution in warm-up; rank-weighted 1/log₂(r+1) when deployed.

### Baselines
Full context · naive fixed top-k RAG · append-all typed store · decay-based forgetting · generative judge (per-item LLM).

### Metrics and targets
| Metric | Target |
|---|---|
| Accuracy retention vs full context | ≥ 0.95 |
| Token reduction vs fixed top-k | ≥ 0.40 |
| Estimator ROC-AUC | ≥ 0.70 |
- Accuracy: LLM judge, with token F1 as lexical check. Cost: tokens/query, latency, records retained, generative calls. Retention precision. Paired bootstrap, 3 seeds, 6 ablation configs.
- **Evaluation frame:** iso-cost (accuracy at matched token spend). Raw accuracy at one operating point is confounded by token spend.

### Research questions
- **RQ1:** Can downstream outcomes train a useful utility estimator?
- **RQ2:** Does utility-gated admission and forgetting beat decay and append-all?
- **RQ3:** Does utility-based retrieval budgeting improve accuracy per token (95% / 40% targets)?
- **RQ4:** How much cost is saved by replacing per-item generative judgement?

### Models and data
- Backbone: Llama-3.1-8B-Instruct (primary), Qwen2.5-7B-Instruct (robustness). 24 GB GPU; full context on LongMemEval-S needs 4-bit AWQ.
- Embedders: bge-small-en-v1.5 (primary), all-mpnet-base-v2 (comparator).
- Datasets: **LoCoMo** (primary, official release in `data/locomo10.json`), **LongMemEval-S** (secondary, HF `xiaowu0162/longmemeval-cleaned`, not yet integrated).
- Why LoCoMo: 105–260 questions per conversation (enough for ledger accumulation), evidence annotations, real irregular timestamps, comparable with published work.

### Open framing decision
If accuracy retention stays below 0.95 with real models, he chooses between (a) improving the estimator further, or (b) reframing the contribution as **efficiency**: comparable accuracy at much lower token cost. **Ask him before producing anything that depends on this.**

### Marker feedback on the proposal (thesis must address it)
**First marker**
- Crediting every retrieved record when an answer is correct does not prove each was useful → validate against evidence annotations and stronger leave-one-out tests. *(Partly done: fix 8.)*
- Control for retrieval-selection bias (only retrieved records get outcomes).
- Utility is not fixed; it depends on query type, task, model, future needs.
- Support healthcare / finance / data-protection / EU AI Act claims with authoritative sources.
- Mark publication status of 2025–2026 sources; separate preprints from peer-reviewed.
- Justify numeric thresholds from literature or evidence.
- Prioritise the essential experiment; reserve time for validation and writing.

**Second marker**
- More figures and more cited papers (including in background).
- Restate the gap after the positioning table.
- Explain methods tested and the theoretical contribution.
- Tighter scope. More hardware detail (e.g., CPU speed). Better Gantt and figures.

---

## 4. Final thesis requirements

### Length and sources
~60–65 pages, ~45 references. Recent (2022+), peer-reviewed first, preprints clearly marked. LJMU Harvard style, alphabetical by surname.

### Structure (Thesis_Report_Template_Final.docx + upGrad "Effective Thesis Writing" deck)
- **Front matter:** cover (title, student name, "Final Thesis Report", month-year) → Table of Contents → Dedication → Acknowledgements → Abstract → List of Tables → List of Figures → List of Abbreviations.
- **Ch 1 Introduction:** Background, Problem Statement, Aim and Objectives, Research Questions, Scope, Significance, Structure.
- **Ch 2 Literature Review** (systematic: databases, search terms, inclusion criteria).
- **Ch 3 Research Methodology:** design, data, preprocessing, features, models, evaluation, tools.
- **Ch 4 Analysis and Design / Implementation:** EDA, hyperparameters, experiment details.
- **Ch 5 Results and Discussion:** final results, analysis, error analysis, justification.
- **Ch 6 Conclusions and Recommendations:** against aims/objectives, contribution, future work.
- **Back matter:** References, Appendix A (research proposal), Appendix B (ethics — public data, no human participants; confirm whether a form is needed).
- Implementation-style TOC (with Tools section, code appendices) was suggested — **not yet confirmed**.

### Formatting rules
- Times New Roman 12, 1.5 spacing, justified, one blank line between sections; headings numbered to 3 levels; each chapter on a new page.
- Chapters 2–5 each open with an Introduction and close with a Summary. Each chapter opens with a one-line statement of what it does.
- Table caption **above**, figure caption **below** ("Table 3.1 …", "Figure 4.2 …"). Never start a section with a figure/table. Refer by number, never "the above table".
- Expand acronyms on first use. Third person throughout.
- Abstract: no citations, abbreviations or graphics; covers intro, objective, method, results, conclusion; written last.
- Results: no references unless comparing, no first person, no qualifiers ("obviously", "just"), numbers under 10 spelled out except mathematical use (proposal used numerals — check with supervisor).
- ~3 objectives in order of importance; each objective has a method and a result.

### Tense (Univ. of Melbourne guide)
Abstract past · Intro present (present perfect for prior work) · Methods past, often passive · Results past (present for referring to tables/figures) · Discussion present, hedged where evidence is weak · Conclusion mixed.

### Writing order
Ch 4 → Ch 5 → Ch 1–3 → Ch 6 → Abstract.

### Final thesis rubric (100)
| Criterion | Wt | Distinction needs |
|---|---|---|
| Structure, formatting, grammar | 5 | Professional, no deficiencies |
| Introduction | 10 | Outstanding problem, RQs, aim, objectives, justification |
| Literature review | 15 | Wide, deep, systematic |
| Materials and methods | 20 | Creative, appropriate, clearly justified |
| Experiments / implementation / tools | 20 | Outstanding implementation, tools used professionally, fully explained |
| Results, analysis, discussion | 15 | Critical analysis, not description |
| Conclusion, implications, recommendations | 10 | All objectives addressed, strong implications |
| Citations and references | 5 | Full, accurate, recent, reliable |

Methods + Implementation = 40 marks. That is where your work has the most impact.

### Submission package (all PDF)
Thesis · Turnitin report (≤ 10%, default filters) · supervisor acknowledgement email · Declaration on Use of AI · presentation slides · PDF of links (code, dataset, **video presentation — mandatory, missing it = 0 marks**) · feedback response sheet. Thesis and Turnitin PDFs must **not** be uploaded to any cloud service; code, data and video may be shared by link.

### Proposal standards (Appendix A, already submitted — for consistency only)
Abstract → Background → Literature Review → RQs → Aim & Objectives → Significance → Scope → Methodology → Requirements/Resources → Research Plan → References. Proposal v7 has an 18-week Gantt, milestone table, 7-risk matrix, positioning table (Table 1). Thesis claims must stay consistent with it; ask for the v7 file if a conflict appears.

---

## 5. Codebase

```
ocmg-thesis/
  README.md  pyproject.toml  requirements.txt  requirements-full.txt  requirements-lock.txt  .gitignore
  .vscode/        settings, launch.json, tasks.json, extensions
  data/           locomo10.json + README (source, counts, known issues)
  ocmg/           types, config, data/benchmarks.py, data/synthetic.py, ingest, embedding,
                  features, estimator, store (SQLite :memory:), retrieval, reader, credit,
                  governance, orchestrator, experiment, metrics, reporting
  scripts/        verify_data.py  run_experiment.py  make_report.py  sweep_budget.py
                  run_all.py (one command for every experiment)
                  compare_forgetting.py (signal / fraction / criterion sweeps)
                  make_dashboard.py (self-contained results/dashboard.html)
  tests/          46 tests, all passing
  results/        written by scripts (overwritten unless --out / --outdir given)
  docs/archive/   superseded README, notes, invalid old results
```

- Nothing persists between runs: store lives in RAM. Each run splits 10 conversations into 5 warm-up / 5 test, retrains the estimator from scratch, evaluates all conditions. Seed 0 is deterministic on the same library versions.
- Single feature path: `features_for` (prevents train/serve skew). All index ops partitioned by `conversation_id` (prevents cross-conversation leakage).
- Editing tip: for batches of 5+ related edits across files, a Python script with an `edit(path, old, new)` helper that asserts the match is more reliable than many single replacements.

### Commands (PowerShell, project root)
```powershell
.\.venv\Scripts\Activate.ps1

# one command for the whole set (~15 min, 3 seeds). --seeds 1 to explore,
# --only main|rarity|forgetting|criterion for one step, --quick for a
# synthetic rehearsal, --real to add the torch steps.
python scripts/run_all.py

# browsable results page, no server and no network
python scripts/make_dashboard.py --open

python scripts/verify_data.py
python -m pytest
python scripts/run_experiment.py --dataset locomo --path data/locomo10.json --runs 1
python scripts/make_report.py

# thesis-style, 3 seeds, separate outputs
python scripts/run_experiment.py --dataset locomo --path data/locomo10.json --runs 3 --out results/run_3seeds.json
python scripts/make_report.py --input results/run_3seeds.json --outdir results/report_3seeds

# real models (GPU >= 24 GB; gated Llama: accept licence, then huggingface-cli login)
pip install -r requirements-full.txt
python scripts/run_experiment.py --dataset locomo --path data/locomo10.json --embedder sentence-transformers --reader hf --runs 3 --out results/run_locomo_real.json
```

### Report outputs → thesis use
| File | Use |
|---|---|
| `results/tables/main_results.csv` | Main comparison table, Ch 5 |
| `results/tables/per_category_accuracy.csv` | Accuracy by question type, error analysis |
| `results/tables/targets.csv` | 95% / 40% / 0.70 target check |
| `results/tables/significance.csv` | OCMG vs each baseline, 95% CI, p-values |
| `results/tables/feature_importance.csv` | Estimator feature reliance |
| `results/tables/label_validation.csv` | Outcome labels vs evidence annotations |
| `results/dataset_summary.json` | Dataset description table, Ch 4 |
| `results/figures/*.png` | 300 dpi; stamped when not reportable |
| `results/summary.md` | Human-readable run summary |

Figure tip: put the "pipeline validation" stamp outside the axes (negative y with `bbox_inches="tight"`); for scatter labels, sort by value and stack with a minimum vertical step to avoid overlap.

---

## 6. LoCoMo facts (verified from the file)
- 10 conversations, 272 sessions, 5,882 turns (1,226 with image captions), 1,986 questions, 2,820 evidence links.
- Date format `"%I:%M %p on %d %B, %Y"` — all parse.
- Categories: 1 multi-hop (282 q, mean 3.13 evidence turns, 98% multi-evidence) · 2 temporal (321) · 3 open-domain (96) · 4 single-hop (841, mean 1.06 evidence) · 5 adversarial (446).
- Adversarial questions have no `answer`, only a misleading `adversarial_answer`; the correct reference is an **abstention**.
- 9 malformed evidence ids normalised; 2 point to non-existent turns and are dropped.

---

## 7. Fix history (README §7) — do not regress these
1. Loader rewritten: real session dates, turn `dia_id` and question `evidence_ids` kept, category names mapped, image captions included, evidence ids normalised.
2. Adversarial reference = abstention; judge is abstention-aware; simulated reader fails these by giving the trap answer.
3. Records carry `source_ids` so the simulated reader checks whether annotated evidence was retrieved. **Before this, every LoCoMo question looked unanswerable — all earlier "passing" results were artefacts.**
4. Decay baseline divide-by-zero with real dates — fixed.
5. Reporting: rich JSON with metadata and `reportable` flag, CSVs, 300-dpi stamped figures, `summary.md`.
6. Tooling: pyproject, split/locked requirements, VS Code config, tests.
7. Simulated reader uses one fixed draw per question (common random numbers); previously leave-one-out labels were two independent coin flips → noise.
8. Label validation: each labelled record gets an evidence label (contains an annotated evidence turn), **validation only, never training**. Reports precision, recall, kappa, estimator AUC vs evidence.
9. **Windows torch import (WinError 126) — diagnosed and repaired without administrator rights, 3 Nov 2026.** The error names `shm.dll`, which is a red herring: walking the import tables shows `shm.dll` resolves fine and `torch_cpu.dll` is the one importing **`VCRUNTIME140_THREADS.dll`**, which ships with the VC++ 2015-2022 redistributable from ~14.30 while this machine carries the VS2019 runtime (14.28) and so has `vcruntime140.dll` and `vcruntime140_1.dll` but not that one. The same Microsoft-signed DLL (verified Authenticode: *Microsoft Windows Software Compatibility Publisher*, 14.51.36247.0) is published in the `msvc-runtime` wheel, so `scripts/fix_torch_dll.py` downloads it, extracts that single file into `torch/lib` (which torch already adds to the DLL search path) and re-checks the import. Scoped to the venv, reversible by deleting one file, **no admin needed**. Do not go back to telling the user to install the redistributable.
   - Corollary for anyone debugging this again: a dependency walker reports `api-ms-win-crt-*.dll` as missing on a *healthy* system too, because those are API sets resolved through the apiset schema rather than by finding a file. They are not evidence of anything.

Earlier-era fixes (keep): leave-one-out credit in warm-up replaced ~93%-noise co-occurrence credit; conversation-partitioned store; single `features_for`; format-aware date parser; quantile thresholds on contested population; exploration policy (wider budgets, ε-random inclusion) to break circular Phase A data collection.

---

## 8. Current results (simulated reader + hashing embedder -> pipeline validation only)

### Where this stands as of 2 November 2026 — read this before the tables below

Three more stages have been measured since the tables further down were
written, and all three came back negative. **The tables below are kept because
their mechanisms still hold, but their headline numbers predate the current
defaults** (`boundary_depth` 0.30, `write_quantile` 0.0, `forget_quantile`
0.0) and should not be quoted. Current numbers are in this subsection.

**A. Utility reranking is harmful, monotonically.** The blend was a
hard-coded `0.5 + prior` that nobody chose; it is now
`(1-w) + 2*w*prior` behind `retrieval.utility_weight`. LoCoMo, 3 seeds:

| w | accuracy | tokens/q | accuracy per 1k |
|---|---|---|---|
| **0.00** | **0.6151 +/- 0.0112** | **1,186** | **0.519** |
| 0.25 | 0.6081 +/- 0.0120 | 1,251 | 0.486 |
| 0.50 (the original rule) | 0.5943 +/- 0.0152 | 1,296 | 0.459 |
| 0.75 | 0.5744 +/- 0.0173 | 1,326 | 0.433 |
| 1.00 | 0.5513 +/- 0.0237 | 1,318 | 0.418 |

Monotone on all three seeds, 15 of 15 cells. Accuracy falls and cost *rises*
together, because the multiplier reshapes the score distribution the
adaptive-k floor is measured against, so a worse ranking is also a wider one.
The default is deliberately **left at 0.5**: see §10 item 8.

**B. The admission gate's whole effect is the cold-tier discount, not the
decision.** `sweep_budget.py --tiers`, 18 cells, replicated on two seeds
(0 and 2 — seed 2 arrived by accident, because the frontier loop was leaving
`args.seed` on the last seed it used; that bug is fixed and the grid now
honours `--seeds`). Along the quantile axis 0.00 / 0.05 / 0.10 / 0.20 / 0.35 /
0.50:

| penalty | seed 0 | seed 2 |
|---|---|---|
| 0.60 | 0.632 0.632 0.632 0.633 0.611 0.568 | 0.586 0.586 0.586 0.586 0.574 0.523 |
| 0.80 | 0.632 0.632 0.632 0.633 0.619 0.592 | 0.586 0.586 0.586 0.586 0.576 0.546 |
| 1.00 | 0.632 0.632 0.632 0.633 0.635 0.635 | 0.586 0.586 0.586 0.586 ... |

At penalty 1.00 the tiers are identical to retrieval, so the gate cannot
reach any prompt — and the row is flat. Below 1.00 the rows fall as the gate
rejects more. So the records the gate rejects are not the ones worth
rejecting. **No cell in either grid beats naive top-k at matched cost**: seed
0 ranges -0.001 to -0.067, seed 2 ranges -0.038 to -0.100.

The two-tier store is also **inert in the selected configuration**: with
`write_quantile` and `forget_quantile` both 0.0, nothing is demoted or
evicted, so the cold tier is empty and `cold_tier_penalty` has no effect.

**C. The efficiency framing fails at iso-cost. This is the decisive result.**
Gateless OCMG (`write_quantile` 0.0, `forget_quantile` 0.0), `sweep_budget.py
--seeds 0 1 2`, 12 matched points per seed:

| comparison | mean gap | sd | per seed | verdict |
|---|---|---|---|---|
| OCMG vs Naive RAG | **-0.0155** | 0.0098 | -0.003 / -0.018 / -0.026 | baseline ahead on all 3 |
| OCMG vs Append-all typed | **-0.0300** | 0.0096 | -0.016 / -0.038 / -0.036 | baseline ahead on all 3 |
| OCMG vs Decay-based | **+0.0459** | 0.0087 | +0.040 / +0.058 / +0.040 | **OCMG ahead on all 3** |

Removing the gate does **not** buy an efficiency win. The single-operating-
point reading (0.6151 at 1,186 tokens against naive RAG's 0.6131 at 1,224,
which looks like parity or better) is the *same* artifact that produced the
retracted 507-vs-600 claim: at matched cost it is -0.0155, behind on every
seed. Any sentence of the form "matched accuracy at fewer tokens" is
unsupported.

One honest nuance, not a claim: on seed 0 the per-budget gap against naive
RAG runs -0.021 / -0.017 / -0.007 / +0.010 / +0.019, so OCMG is ahead at the
*highest* budgets only. That is because naive RAG degrades past k=8 (0.624 ->
0.617 -> 0.604) while adaptive k holds up. "Degrades more gracefully when
over-retrieving" is true and is not a win, because that whole regime is worse
than either method's own optimum.

**Compound consequence, stated plainly.** Of the five stages, stage 2
(admission) is harmful, stage 5 (forgetting) is harmful, stage 4's utility
reranking is harmful, and stage 3 (the two-tier store) is inert once the
first two are off. At `write_quantile` 0.0, `forget_quantile` 0.0 and
`utility_weight` 0.0 — each the measured optimum of its own sweep — **the
learned estimator has no causal path to any prompt.** ROC-AUC 0.728 would be
an honest measurement of a component that cannot affect the system's
behaviour. What survives is stage 1 (segmentation and typing, where
`boundary_depth` genuinely matters), stage 4's adaptive budget, and the
leave-one-out credit machinery as a *method* result.

**D. Query-conditional U(m|q) does not rescue it. The estimator was never the
bottleneck — the blend is.** Implemented behind `--conditional-utility` /
`EstimatorConfig.conditional_utility` (default off): the leave-one-out table
already holds (record, question) pairs and the pooled estimator discarded the
question, so each observed (record, derived query type) cell becomes its own
training row with the query type one-hot appended. Write-time scoring
marginalises over query types weighted by observed frequency, because
admission and forgetting act before any question exists.

The model learned real query-conditional structure and used it heavily —
`q_enumerative` is its **second most important feature** (+0.084) and
`q_simple` its third (+0.059), behind only `rare_term_count`. It is not
ignoring the query. And the system did not move:

| w | pooled U(m) | conditional U(m\|q) | delta |
|---|---|---|---|
| 0.00 | **0.6151 +/- 0.0112** | same by construction (utility ignored) | — |
| 0.25 | 0.6081 +/- 0.0120 | 0.6093 +/- 0.0132 | +0.0011 |
| 0.50 | 0.5943 +/- 0.0152 | 0.5967 +/- 0.0143 | +0.0024 |
| 0.75 | 0.5744 +/- 0.0173 | 0.5637 +/- 0.0094 | -0.0107 |

Every delta is inside one standard deviation, the curve is still monotone
decreasing on all three seeds (12 of 12 cells), and w=0 is still the best
cell. **Do not read the AUC as an improvement**: it rises 0.728 +/- 0.005 to
0.753 +/- 0.006, but the row population changed with it (655 rows at a 0.432
positive rate to 1,467 at 0.243), so the two AUCs are measured on different
label distributions — which is the same base-rate confound that motivated the
experiment. The honest comparator is downstream accuracy, and it is flat.

**What this rules out, and what it leaves.** It rules out "the estimator is
too weak" as the explanation: a better-informed estimator that demonstrably
uses the query changes nothing. The bottleneck is the *combination rule* —
multiplying a relevance score by any utility estimate degrades the ranking
regardless of how good the estimate is. The one structurally untested option
is a learned reranker that takes relevance as an input *feature* rather than
as a multiplicand (standard learning-to-rank), which subsumes both the pooled
and conditional forms. That is a design change, not a tuning change, so it is
in §10 item 2 as an option rather than queued.

Third instance of the same pattern, now in both directions: the rarity
heuristic had *higher* AUC and much worse accuracy; the conditional model has
higher AUC and unchanged accuracy. **AUC moves independently of system
accuracy in this pipeline**, which is a methodological result worth reporting
in its own right and a direct problem for RQ1's "ROC-AUC >= 0.70" target as
written.

Measured 1 November 2026, LoCoMo, **3 seeds**, `results/full/run_main.json`,
after the §3 alignment pass (naive RAG is fixed-window chunking, append-all is
the typed ungoverned store, stage-4 budget reads a derived query type).

| condition | accuracy | sd | tokens/q | accuracy per 1k |
|---|---|---|---|---|
| Full context | 0.573 | 0.007 | 15,366 | 0.037 |
| Naive RAG (fixed 4-turn chunks, k=10) | **0.613** | 0.011 | 1,224 | 0.501 |
| Append-all typed (no governance) | 0.520 | 0.003 | 600 | 0.868 |
| Decay-based | 0.465 | 0.015 | 638 | 0.731 |
| **OCMG** | 0.507 | 0.009 | **507** | **1.003** |

| target | value | verdict |
|---|---|---|
| Accuracy retention >= 0.95 | 0.885 +/- 0.016 | **NO CEILING** - naive RAG (0.613) now scores above full context (0.573), so the ratio has no upper bound to measure against |
| Token reduction >= 0.40 | 0.586 +/- 0.019 | **passes, but do not quote it** - OCMG's own spend went UP (448 -> 507); the target flipped because the comparator got dearer. §2.3 forbids presenting that as a win |
| Estimator ROC-AUC >= 0.70 | 0.709 +/- 0.016 | MET, and beaten by one untrained feature (see result 3) |

### The iso-cost result (the frame §3 mandates)

`scripts/sweep_budget.py --seeds 0 1 2`, 12 matched points per seed over
249-1,279 tokens. The gap is a mean of interpolated differences along each
frontier, not a paired test, so read the consistency of sign across seeds
rather than treating the magnitude as significant:

| comparison | mean gap | sd | per seed | verdict |
|---|---|---|---|---|
| OCMG vs Naive RAG | **-0.1106** | 0.0092 | -0.123 / -0.108 / -0.101 | baseline ahead on all 3 |
| OCMG vs Append-all typed | **-0.0168** | 0.0132 | -0.035 / -0.003 / -0.012 | baseline ahead on all 3, by little |
| OCMG vs Decay-based | **+0.0373** | 0.0248 | +0.003 / +0.061 / +0.048 | **OCMG ahead on all 3** |

**So RQ2 splits.** It asks whether utility-gated admission and forgetting beat
decay *and* append-all. Against decay, yes: OCMG is ahead at matched cost on
every seed, mean +0.037. Against append-all, no: behind on every seed, though
by only 0.017 on average. Seed 0 alone gave -0.035, twice the three-seed mean,
so the single-seed table below is the pessimistic end of the range.

The 0.017 deficit against append-all is the real finding to report: governance
neither helps nor badly hurts relative to keeping everything, once cost is
matched. Against the published-practice baseline it is behind by 0.111.

Accuracy at matched token budgets, seed 0 (the worst of the three for OCMG):

| condition | 249 | 408 | 568 | 728 | 887 |
|---|---|---|---|---|---|
| Naive RAG | 0.558 | 0.585 | 0.608 | 0.618 | 0.622 |
| Append-all typed | 0.498 | 0.516 | 0.520 | 0.522 | 0.527 |
| Decay-based | 0.464 | 0.478 | 0.482 | 0.484 | 0.482 |
| OCMG | 0.458 | 0.481 | 0.490 | 0.493 | 0.489 |

**OCMG is below append-all at every budget, and appears nowhere on the Pareto
frontier** (append-all owns the two cheapest points, naive RAG everything
above). Its curve also tops out lower: 0.494 against append-all's 0.527 and
naive RAG's 0.624, so the admission gate is removing records that matter.

**This retracts the single-operating-point reading.** Comparing OCMG at 507
tokens to append-all at k=10 (600 tokens, 0.520) looked like 16% cheaper for
indistinguishable accuracy. At matched cost it is not: append-all at k=8 spends
496 tokens and scores 0.520, so it is both cheaper *and* better than OCMG's
507 tokens at 0.507. The apparent saving was the confound §3 warns about -
"raw accuracy at one operating point is confounded by token spend" - and the
earlier claim that governance buys a free 16% was wrong.

So on the mandated frame **RQ3 fails and RQ2 half-holds**: OCMG beats the
decay baseline at matched cost on every seed, and is behind the ungoverned
typed store on every seed. The efficiency reframing in §3's open decision is
not supported as "cheaper for the same accuracy", because there is no budget
at which OCMG is the best choice; what the data supports is the narrower claim
that governance roughly breaks even against keeping everything while beating
the decay schedules that dominate current practice.

### Four results beyond the target table

1. **The optimum amount of forgetting is zero, monotonically.**
   3 seeds: accuracy 0.507 / 0.505 / 0.502 / 0.487 / 0.468 at fractions 0.00 /
   0.05 / 0.10 / 0.25 / 0.40, falling on all 15 seed-by-fraction cells with no
   interior peak; accuracy per 1k falls with it (1.003 -> 0.891) because tokens
   *rise* (507 -> 527) as eviction widens the candidate pool and the adaptive
   budget grows. Evidence ceiling 99.9% -> 63.1%. **q=0.00 beats q=0.25 on all
   three seeds now** (+0.020 / +0.020 / +0.022, p = 0.001 / 0.002 / 0.001),
   where it was two of three before. Mechanism: under an adaptive budget the
   prompt is billed for k, not for store size, so eviction cannot buy tokens
   and can only destroy evidence. (`results/full/forget_quantiles.json`)
2. **No forgetting criterion beats recency at matched eviction volume.**
   `forget_signal="recency"` runs the decay baseline's exact formula inside the
   governed pipeline, so admission, budget, eligibility guard and fraction are
   identical and only the ranking differs (evictions 245 / 239 / 239). Means:
   ledger 0.481, recency 0.482, estimator 0.487, none 0.507. **Every criterion
   is indistinguishable from every other**: estimator vs ledger p = 0.124 /
   0.338 / 0.190, recency vs ledger p = 0.204 / 0.956 / 0.337. Only switching
   forgetting off separates (none vs ledger +0.032 / +0.012 / +0.032). This is
   the comparison the system-level OCMG-vs-decay table **cannot** make, because
   at a matched `forget_quantile` the two evict 103 vs 587 records.
   (`results/full/forget_criterion.json`)
3. **Ranking AUC is the wrong metric for choosing the estimator.**
   `--estimator rarity` ranks by `rare_term_count` alone through a
   one-dimensional logistic. Across 3 seeds its **AUC is higher** (0.719 +/-
   0.016 vs 0.709 +/- 0.016) and its system is far worse (OCMG 0.448 +/- 0.005
   vs 0.503 +/- 0.008, about 7 pooled sd; significantly worse than naive RAG on
   every seed, p=0.000). RQ1's 0.70 target is therefore cleared by one
   untrained integer that also beats the model - say this plainly rather than
   reporting 0.709 as a pass. Mechanism: the admission threshold is a quantile
   of the predicted probability, so calibration and score distribution decide
   what is cut, not pairwise order.

   **The sharpest version of what the estimator buys.** Against append-all -
   governance on versus off, same store - the learned estimator is
   indistinguishable (p = 0.088 / 0.695 / 0.104) while the rarity heuristic is
   **significantly worse on every seed** (-0.057 / -0.064 / -0.061, every
   p = 0.000). So the estimator is not what makes governance *win*; it is what
   keeps governance from actively *costing* accuracy relative to doing nothing.
   (`results/full/run_rarity.json`)
4. **Leave-one-out credit remains the strongest positive.** AUC 0.727 with
   counterfactual re-execution against 0.477 (rank-weighted) and 0.462
   (uniform); the cheap co-occurrence rules collapse the system to 0.345. This
   answers the first marker's objection with a measurement.

### Known limitations
- Real-reader judge is token F1 >= 0.6 + abstention check; the proposal promises an **LLM judge** - not built.
- **LongMemEval-S not integrated.**
- **Retrieval-selection bias not handled** (eps-exploration exists in warm-up config only).
- ~~Per-category estimator AUC not produced~~ — produced: single_hop 0.666, temporal 0.650, multi_hop 0.630, open_domain 0.493, pooled 0.723 (`metrics.per_category_auc`, -1 rows dropped, NaN for one-class categories). Feature importance now also names the query-type columns, which `zip(FEATURE_NAMES, ...)` had been silently truncating.
- Threshold sensitivity curves for tau_write exist via the admission sweep; tau_forget is covered by result 1 above. Neither is written up.
- Real-model runs not possible on the current machine: **no CUDA GPU** (Intel Iris Xe, 4-core i7-1185G7, 15.4 GB RAM), so Llama-3.1-8B is unreachable locally. A small CPU reader (Qwen2.5-1.5B-Instruct) is wired up behind `--reader hf`; torch is installed but needs the VC++ 2015-2022 redistributable for `vcruntime140_threads.dll`.
- `results/dashboard.html` and every results table are still stamped pipeline-validation, correctly.

### Fixed since the step-2 notes (do not re-list as limitations)
- Generative-judge baseline **now actually filters** (`GenerativeGate`, one model call per candidate, fails open, unparsed replies counted). It is skipped with a printed reason when no generative model is loaded, instead of silently duplicating naive RAG.
- Iso-cost comparison exists (`sweep_budget.py`, data-driven overlap grid).
- The old budget-sweep non-monotonicity is gone; token spend is monotonic in budget.
- `append_all` baseline **removed**: it built an ungoverned store and retrieved fixed top-k, so it was `naive_rag` exactly and reported identical numbers to the last digit.
- Retention precision is no longer printed under `forget_signal="ledger"`: the policy ranks on realised utility and the metric is the gap in realised utility, so it scored the sort against its own key. §4 label audit tests the same claim against held-out annotations.
- Accuracy retention now reports **NO CEILING** instead of a false MET when full context scores below naive top-k (it does on synthetic data: 0.342 vs 0.487, ratio 1.462).
- Runtime: a single LoCoMo run is ~71s, down from 128s, verified bit-identical. Two fixes - batched prediction in `calibrate_thresholds` (59s -> 0.1s) and an incremental dense index (`_rebuild` was running 10,206 times per run).

---

## 9. Backlog (in order)

Each item: implement → add/extend tests → run → regenerate report → summarise numbers and whether targets/criteria pass.

### Step 3 — Iso-cost sweep + ablations — **DONE** (31 Oct 2026)
- `scripts/sweep_budget.py` runs the budget grid for OCMG and every baseline, with a data-driven overlap grid (the old grid was hardcoded to the synthetic token scale of 25-150 while LoCoMo spends 380-700).
- Iso-cost result: **naive top-k leads at every matched budget**; the best OCMG configuration is -0.039. Token spend is monotonic in budget, so the old coupling is gone.
- Ablations implemented and run: admission off · forgetting **on** (0.25; off is now the default, so the informative switch is reversed) · fixed k · flat store (`IngestConfig.flat_store` routes every segment to FACTUAL - changing segment length would measure something else) · uniform credit · rank-weighted credit. Both credit variants set `loo_fraction=0`, without which the ablation was a silent no-op.
- Beyond the plan: forgetting-fraction sweep, forgetting-criterion ablation at matched eviction volume, and the rarity-only estimator ablation. All four are in §8.
- Two pre-existing bugs found and fixed here: eviction swept whole tie groups at utility 0.0, so `forget_quantile` 0.0 and 0.25 both evicted 60 of 100 and the decay baseline retained only 339 of 1,616 records (fixed with rank-based `select_lowest_fraction`; **this flipped the OCMG-minus-decay verdict, so the earlier +0.034 claim was retracted**); and the cold tier leaked evicted records into retrieval, turning every eviction into a demotion and contaminating decay.

### Step 4 — Close the methodology gaps (NEXT)
- **LLM judge:** configurable judge model; prompt asks correct/incorrect/abstained vs reference; validate against token-F1 on a sample and report agreement (Cohen's kappa). Cache judgements to disk.
- ~~**Real generative-judge baseline**~~ — built (`GenerativeGate`). Still needs a run with a real model loaded to produce RQ4's cost numbers.
- **Selection-bias analysis:** compare estimator trained with vs without ε-exploration; optionally inverse-propensity weighting of labels; report effect on AUC and downstream accuracy (first-marker concern).
- **Utility context-dependence:** per-category estimator AUC and per-category feature importance (first-marker concern: utility depends on query type).
- **Threshold justification:** sensitivity curves for τ_write and τ_forget (second-marker / first-marker concern).
- Optional: LongMemEval-S loader (`answer_session_ids` as evidence) + one run.

### Step 5 — Real-model runs (reportable)
- **Answered: there is no CUDA GPU on this machine** (Intel Iris Xe, i7-1185G7 4 cores, 15.4 GB RAM), so Llama-3.1-8B is unreachable locally.
- **UNBLOCKED 3 Nov 2026: torch now imports** (2.15.0+cpu, 4 threads) via `scripts/fix_torch_dll.py` — see §7 item 9. transformers 5.22.1, sentence-transformers 6.2.0, tokenizers 0.23.3 all import. The real embedder is live: `BAAI/bge-small-en-v1.5` loads in ~50 s and encodes 64 texts in 0.22 s (~290/s, dim 384), so embedding is not the bottleneck; the reader is. Agreed path: a small CPU reader (Qwen2.5-1.5B-Instruct) with `--questions-per-conv` subsampling and a measured timing probe before any overnight run. Rented GPU or HPC remains the alternative if an 8B backbone is required for the claim.
- sentence-transformers embedder + HF reader (Llama-3.1-8B-Instruct; Qwen2.5-7B-Instruct for robustness), 3 seeds.
- Record hardware (GPU model, VRAM, CPU model and clock, RAM), library versions, wall-clock per condition → Ch 4 tools/hardware table (second-marker concern).
- Set `reportable: true` only for these runs. Regenerate all tables and figures without the stamp.
- Re-run label validation with the real reader (this is where it becomes a real result).

### Step 6 — Thesis evidence pack (no prose)
- Final tables (CSV + a formatted .docx table file using TNR 12, caption above) and figures (PNG 300 dpi, caption text below) numbered by chapter (Table 5.1, Figure 4.2…).
- Architecture figure: hand-crafted look, night mode (dark navy background, serif type, rounded boxes, amber governance components, dashed feedback loop) — consistent with the proposal.
- Dataset description table and EDA figures for Ch 4 (sessions per conversation, turns, question categories, evidence counts).
- Error analysis data for Ch 5: failure cases by category, adversarial abstention rate, examples of wrongly evicted records (IDs + short factual description, no narrative).
- Objective → method → result traceability table (RQ1–RQ4 ↔ experiment ↔ table/figure).
- Reproducibility: pinned `requirements-lock.txt`, seeds, exact commands in README; code + data link ready for the submission links PDF.

### Step 7 — Writing support tools (he writes the text)
- `scripts/check_draft.py`: lint a .docx draft he wrote — font/size/spacing, justified text, heading numbering depth, caption position (table above / figure below), section starting with a figure/table, "the above table" phrasing, first-person pronouns, qualifiers in results, acronyms not expanded on first use, citations in abstract/conclusion, numbers < 10 as numerals in results. Output a report, never rewrite his text.
- `scripts/check_refs.py`: for his reference list, check every in-text citation has an entry and vice versa, duplicates, alphabetical order, DOI resolves (Crossref), year/venue match, preprint flagged; target ~45 refs, 2022+ bias.
- Rubric checklist (markdown or xlsx) mapping each rubric criterion to where it is evidenced in his draft.
- Outline + page budget per chapter for ~60–65 pages (structure only).
- Slide skeleton for the mandatory video presentation and a feedback-response-sheet template (marker point → where addressed), both as structure only.

---

## 10. Open questions — ask before depending on them
1. Proposal v7 file (for consistency of claims and numbers).
2. Framing: accuracy vs efficiency. **Both branches are now measured, and both fail.** Accuracy: behind append-all on every seed at matched cost (-0.030). Efficiency: behind naive top-k on every seed at matched cost (-0.0155), so "matched accuracy at fewer tokens" is refuted — the earlier recommendation of the efficiency framing in this file was based on a single operating point and has been withdrawn. §3's choice has therefore narrowed further, because (a) "improve the estimator" is now also measured and failed: query-conditional U(m|q) is implemented, demonstrably uses the query (§8 result D), and leaves system accuracy unchanged. So the live options are (c) report the negative result with its mechanism plus leave-one-out credit as the methodological contribution, or (a') one more attempt at the *combination rule* rather than the estimator - a learned reranker taking relevance as a feature instead of a multiplicand. **Ask before writing anything that depends on this.** Do not quote an efficiency claim in the meantime.

8. `retrieval.utility_weight` default. The measurement says 0.0 (§8 result A), and the project's practice has been to set defaults from measurement, as `write_quantile` and `forget_quantile` were. It is deliberately **not** set here, because at 0.0 — with the other two already at 0.0 — the learned estimator has no causal path to any prompt at all, which decides what the work claims rather than how a parameter is tuned. His call.
3. Confirm the implementation-style chapter structure.
4. Cover page: final title, full name, submission month.
5. ~~GPU ≥ 24 GB or HPC access~~ — answered: none on this machine. Still open whether to rent one, or accept a small CPU reader (see Step 5).
6. ~~Rubric spreadsheet~~ — the rubric with weightings is now in §4.
7. Supervisor's written position on permitted AI assistance.

---

## 11. Definition of done (experimental side)
- All tests pass; Steps 3–5 complete with 3 seeds on real models.
- Every target in `targets.csv` reported honestly as met / not met, with CIs.
- Iso-cost frontier, ablations, label validation (real reader), selection-bias analysis, threshold sensitivity, per-category analysis all produced.
- Every number the thesis will cite traces to a file in `results/` produced by a committed command.
- Evidence pack and checker scripts delivered. No thesis prose generated.
