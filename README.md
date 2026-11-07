# OCMG — Outcome-Calibrated Memory Governance

Code for the MSc dissertation *Outcome-Calibrated Memory Governance: A Learned Utility Approach to
Improving Long-Term Memory Capability in Large Language Model Agents* (Liverpool John Moores
University).

OCMG learns, from whether an agent answered correctly, which memory records are worth keeping. A
small classifier (the utility estimator) then decides what enters memory, what is evicted, and how
many records are retrieved per question. The framework is compared against five baselines on the
LoCoMo benchmark.

---

## 1. Current status (read before demonstrating)

| Part | Status |
|---|---|
| LoCoMo loading (dates, evidence, categories) | Working and tested |
| Full pipeline: 5 baselines + OCMG, metrics, significance tests | Working, about 2 minutes on a laptop CPU |
| Report: CSV tables, 300-dpi figures, Markdown summary | Working |
| Utility estimator | ROC-AUC about 0.73 (target 0.70), validated against the LoCoMo evidence annotations |
| Thesis-grade numbers | Need the real embedder and a real language model (section 6) |

The default run uses a **simulated reader** (no language model) and a **hashing embedder**. These
let the whole pipeline run on any laptop, and every figure from such a run is stamped
"Pipeline validation - not a finding". Do not quote these numbers as results.

---

## 2. Project structure

```
ocmg-thesis/
├── README.md                  this guide
├── pyproject.toml             package metadata and pytest settings
├── requirements.txt           core dependencies (CPU, no downloads)
├── requirements-full.txt      adds real models (sentence-transformers, transformers, torch)
├── requirements-lock.txt      exact versions the reference run used (for reproducibility)
├── .gitignore
├── .vscode/
│   ├── settings.json          interpreter, pytest integration
│   ├── launch.json            Run and Debug entries for each demo step
│   ├── tasks.json             one-click setup and full demo
│   └── extensions.json        recommended extensions
├── data/
│   ├── locomo10.json          LoCoMo benchmark (10 conversations, 1,986 questions)
│   └── README.md              source, contents, known issues in the release
├── ocmg/                      the framework (importable package)
│   ├── types.py               data structures: Turn, Candidate, Record, Question, results
│   ├── config.py              every tunable setting in one place
│   ├── data/
│   │   ├── benchmarks.py      LoCoMo and LongMemEval loaders
│   │   └── synthetic.py       synthetic data generator for tests
│   ├── ingest.py              Stage 1: filter, segment, resolve dates, route to type
│   ├── embedding.py           hashing embedder and sentence-transformers wrapper
│   ├── features.py            write-time features for the utility estimator
│   ├── estimator.py           Stage 2: gradient-boosted utility estimator, grouped CV, calibration
│   ├── store.py               Stage 3: two-tier store, SQLite utility ledger, cold store
│   ├── retrieval.py           Stage 4: hybrid dense + BM25 retrieval, adaptive budget
│   ├── reader.py              simulated reader, Hugging Face reader, answer judge
│   ├── credit.py              Stage 5: rank-weighted credit assignment
│   ├── governance.py          admission, merging and forgetting rules
│   ├── orchestrator.py        runs warm-up (phase A) and governed evaluation
│   ├── experiment.py          all experimental conditions, decay baseline
│   ├── metrics.py             accuracy ratios, paired bootstrap, iso-cost frontier
│   ├── reporting.py           converts results to JSON for tables and figures
│   └── console.py             terminal presentation: colour, tables, bar charts
├── scripts/
│   ├── verify_data.py         step 1: check the dataset and print its statistics
│   ├── run_experiment.py      step 3: run all conditions, save results/run_<dataset>.json
│   ├── make_report.py         step 4: build tables, figures and summary.md
│   └── sweep_budget.py        iso-cost frontier and ablations (later step)
├── tests/                     25 tests: loader, judge, credit, store, retrieval, pipeline
├── results/                   created by the scripts (tables/, figures/, *.json, summary.md)
└── docs/archive/              superseded README, notes and results from the earlier version
```

---

## 3. Setup in VS Code (Windows)

Prerequisites: Python 3.10 or newer (3.11 recommended) and VS Code.

1. Unzip the project, then in VS Code choose **File > Open Folder** and select `ocmg-thesis`.
2. When VS Code offers to install the recommended extensions, accept. You need at least the
   Python extension.
3. Open a terminal with **Terminal > New Terminal** and create the environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks `Activate.ps1`, run this once and try again:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

   Alternatively, run the task **Terminal > Run Task > Setup: create venv and install**.
4. Select the interpreter: press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick
   `.venv`.

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

For exactly reproducible numbers, install the pinned versions instead:
`pip install -r requirements-lock.txt` (created with Python 3.12). Different library versions shift results in the third
decimal place.

---

## 4. Demonstration, step by step

Each step is available in the **Run and Debug** panel (`Ctrl+Shift+D`) under the same name, or as a
terminal command.

### Step 1. Verify the dataset

```powershell
python scripts/verify_data.py
```

This prints 10 conversations, 272 sessions, 5,882 turns and 1,986 questions, the category
breakdown, and the date range of each conversation. It also writes `results/dataset_summary.json`.

### Step 2. Run the tests

```powershell
python -m pytest
```

Expect `25 passed`. Tests also appear in the VS Code **Testing** panel (flask icon).

### Step 3. Run the experiment

```powershell
python scripts/run_experiment.py --dataset locomo --path data/locomo10.json --runs 1
```

It takes about 2 minutes on a laptop CPU. Everything is saved to `results/run_locomo.json`.

Use `--runs 3` for mean and standard deviation over three seeds, which is what the thesis needs.

The console output is built for showing, not just for logging. It runs through nine numbered
sections — setup, warm-up, estimator, label audit, thresholds, conditions, targets, significance,
output — and uses colour to carry the comparison:

| Encoding | Meaning |
|---|---|
| `▲` green | better than the fixed top-k baseline |
| `▼` red | worse |
| `·` | no measurable change; `ref` marks the baseline row itself |
| `✔ MET` / `✘ NOT MET` | a target passed or failed |
| emphasised bar | the condition under discussion; the rest recede |

Three rules are deliberate and should not be "tidied up" later. The arrow encodes better-or-worse
rather than up-or-down, so on the token column a fall is an up-arrow — otherwise a reader who
cannot distinguish the red would read a red up-arrow as good news. Every colour is paired with a
glyph and a word, so the output survives a projector, a monochrome print and a colourblind reader.
And accuracy and token cost never share a chart, because they have different units and one axis
would imply a relationship that is not in the data; only the derived ratio is plotted.

Colour switches itself off when the output is redirected to a file, when `NO_COLOR` is set, or on a
dumb terminal. `--no-color` forces it off.

### Step 4. Build the report

```powershell
python scripts/make_report.py
```

| Output | Content |
|---|---|
| `results/summary.md` | Main results table and target checks (open with `Ctrl+Shift+V`) |
| `results/tables/main_results.csv` | Accuracy, token F1, tokens per query, records, generative calls |
| `results/tables/per_category_accuracy.csv` | Accuracy for each of the 5 question types |
| `results/tables/targets.csv` | The 95% retention, 40% token reduction and 0.70 AUC targets |
| `results/tables/significance.csv` | OCMG against each baseline, with 95% CI and p-value |
| `results/tables/feature_importance.csv` | Which record features the estimator relies on |
| `results/figures/*.png` | 300-dpi figures for the thesis |

### All four steps at once

Run **Terminal > Run Build Task** (`Ctrl+Shift+B`). This executes verify, test, run and report in
sequence.

---

## 5. What the conditions are

| Condition | What it does |
|---|---|
| Full context | Puts the whole conversation in the prompt; accuracy reference point |
| Naive RAG (top-k) | Fixed top-k retrieval over all records; token-cost reference point |
| Append-all | Same typed store as OCMG, keeps everything, never forgets |
| Decay-based | Forgets by retrieval frequency and recency (Ebbinghaus-style heuristic) |
| Generative judge | Counts one model call per candidate record (cost reference; see limitations) |
| OCMG | Utility-gated admission, forgetting and adaptive retrieval budget |

---

## 6. Running with real models (for thesis results)

The reported results must use the real embedder and a frozen language model.

```powershell
pip install -r requirements-full.txt
python scripts/run_experiment.py --dataset locomo --path data/locomo10.json `
    --embedder sentence-transformers --reader hf --runs 3 --out results/run_locomo_real.json
python scripts/make_report.py --input results/run_locomo_real.json --outdir results/real
```

Requirements:

- An NVIDIA GPU with at least 24 GB of memory for `meta-llama/Llama-3.1-8B-Instruct` in bfloat16,
  and a CUDA build of PyTorch (https://pytorch.org/get-started/locally/).
- Llama 3.1 is gated: accept the licence on its Hugging Face page, then run
  `huggingface-cli login`.
- The embedder `BAAI/bge-small-en-v1.5` downloads automatically and runs on CPU.

A run like this is stamped as reportable in the JSON (`"reportable": true`) and its figures carry
no validation stamp.

---

## 7. Changes in this version

1. **LoCoMo loader rewritten.**
   - It now uses the real session dates instead of invented weekly dates.
   - Each turn keeps its dialogue id (such as `D1:3`), and each question keeps its evidence ids.
   - The category codes map to names: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop and
     5 adversarial. The mapping was confirmed from the evidence counts in the data.
   - Image captions are included in the turn text.
   - Malformed evidence ids are normalised.
2. **Adversarial questions fixed.** The old loader used the misleading "trap" answer as the
   correct reference. The reference is now an abstention. The judge marks an adversarial question
   correct only when the reader abstains.
3. **Records are linked to their source turns**, so the simulated reader checks whether the
   annotated evidence was actually retrieved. Previously, with no evidence available, the
   simulated reader treated every question as unanswerable and rewarded short prompts. That made
   the earlier "passing" results artefacts. Those results are kept in `docs/archive/` for the
   record only.
4. **Decay baseline bug fixed.** With real dates it divided by zero.
5. **Reporting added.** `ocmg/reporting.py` and `scripts/make_report.py` produce the tables,
   figures and a run metadata block. `scripts/verify_data.py` checks the dataset.
6. **Project tooling added:** `pyproject.toml`, split requirements, VS Code configuration,
   `.gitignore`, and 11 new tests.
7. **Credit labels fixed (common random numbers).** The simulated reader drew a fresh random
   number on every call, so each leave-one-out comparison was a difference of two independent
   coin flips. The expectation was right but the variance swamped it, and the labels the
   estimator trained on were close to noise. The draw is now fixed per question and derived
   from the question id, so removing a record changes the outcome only when it changes
   `p_correct`. This alone moved the estimator from 0.507 to about 0.73.
8. **Label validation against annotated evidence.** Every labelled record now also carries an
   evidence label - does it contain a turn LoCoMo annotates as supporting some answer? It is
   held out completely: never a feature, never a training target. Each run reports the
   precision, recall and Cohen's kappa of the outcome labels against it, plus the estimator's
   ROC-AUC against evidence directly. This is what answers the first marker's objection that
   crediting every retrieved record does not prove each one was useful.

---

## 8. Known limitations

- The simulated reader is a stand-in for a language model, and its numbers are not findings.
- With the real reader, correctness is currently judged by token F1 of at least 0.6, plus an
  abstention check for adversarial questions. The proposal specifies a language-model judge,
  which is not yet implemented.
- The generative-judge baseline only counts the cost of one model call per candidate. It does not
  yet filter records with a model.
- Recall of the outcome labels against annotated evidence is about 0.48: fewer than half the
  evidence-bearing records are ever credited. Some of this is expected, because a record can
  support an answer without being the record whose removal changes it, but the gap has not been
  separated into redundancy and genuine misses.
- Label precision of 1.000 is partly by construction: the simulated reader decides correctness
  from evidence alone, so a record whose removal changed the answer must be evidence. The number
  becomes meaningful only under the real reader.
- OCMG is measured at a single operating point. The iso-cost comparison is still outstanding.

---

## 9. Next steps

1. Done: evidence-validated utility labels (section 7, items 7 and 8).
2. Re-run the credit-scheme comparison (`rank_weighted`, `uniform`, `top1`) against the
   leave-one-out labels now that those labels carry signal, to quantify what the cheap
   deployed rule gives up.
3. Re-run the budget sweep (`scripts/sweep_budget.py`) on LoCoMo for the iso-cost comparison.
4. Run with real models and three seeds, then build the final report.
