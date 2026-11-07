# OCMG — Outcome-Calibrated Memory Governance

Reference implementation for *Outcome-Calibrated Memory Governance: A Learned
Utility Approach to Improving Long-Term Memory Capability in Large Language
Model Agents*.

The idea in one sentence: **an agent that answers questions already knows which
of its memories were worth keeping — it just throws that information away.**
OCMG catches it, trains a cheap predictor on it, and uses the predictor to
decide what enters memory, what stays, and how much gets retrieved.

---

## Quick start

```bash
cd ocmg
pip install -r requirements.txt

# full experiment: 5 baselines + OCMG, three seeds
python scripts/run_experiment.py --conversations 40 --runs 3

# the analysis that actually settles the question
python scripts/sweep_budget.py --conversations 24 --ablations

# sanity checks
python -m pytest tests/ -q
```

Runs in about a minute on a laptop CPU with no downloads. Nothing is stubbed —
every stage in the paper is implemented and exercised.

---

## What runs by default, and what you must change for the dissertation

| Component | Default (runs anywhere) | For reported results |
|---|---|---|
| Embedder | 256-d hashing encoder | `--embedder sentence-transformers` (bge-small-en-v1.5) |
| Reader | `SimulatedReader` | `--reader hf` (Llama-3.1-8B-Instruct, frozen) |
| Judge | exact match | LLM-as-a-judge |
| Data | synthetic generator | `--dataset locomo --path ...` |

**Be clear about this with your supervisor.** `SimulatedReader` does not call a
language model. It models answer generation as a probabilistic function of
whether the retrieved prompt contains the supporting evidence, with a penalty
for irrelevant context. It exists so the governance machinery can be validated
end-to-end — the causal chain *evidence in prompt → answer correct → credit to
record* is preserved exactly — but **absolute accuracy numbers from it are not
findings.** They tell you the pipeline is wired correctly and the conditions
are ordered sensibly. Every number quoted in the dissertation must come from
`HFReader` on LoCoMo or LongMemEval-S.

---

## Pipeline

```
Turn stream
   │
   ├─ 1 INGEST     filter fillers → topic segmentation → temporal
   │                normalisation → type router {profile, episodic, factual}
   │
   ├─ 2 ADMIT      dedup-merge  →  U(m) ≥ τ_write ?  →  store or discard
   │                              (no generative call)
   │
   ├─ 3 STORE      Tier 1 compact index  |  Tier 2 evidence store
   │                          Utility Ledger (SQLite, transactional)
   │
   ├─ 4 RETRIEVE   dense + BM25 → Tier 1 shortlist → adaptive k →
   │                Tier 2 expansion → frozen reader → correct / incorrect
   │
   └─ 5 FEEDBACK   credit assignment → ledger update → periodic re-fit of U(m)
                   → evict where U_hist < τ_forget (non-destructive)
                          │
                          └──────────► back to stage 2
```

## Module map

| File | Responsibility |
|---|---|
| `ocmg/types.py` | Records, candidates, questions, ledger rows, results |
| `ocmg/config.py` | Every threshold, budget and seed in one object |
| `ocmg/embedding.py` | Hashing and sentence-transformers backends |
| `ocmg/ingest.py` | Stage 1 — segmentation, temporal normalisation, type routing |
| `ocmg/features.py` | 30 features across 4 families |
| `ocmg/store.py` | Stage 3 — two-tier store, SQLite ledger, cold store |
| `ocmg/retrieval.py` | Stage 4 — hybrid retrieval, adaptive budget |
| `ocmg/estimator.py` | U(m) — GBDT/logistic, grouped CV, calibration |
| `ocmg/credit.py` | Stage 5 — four credit-assignment schemes |
| `ocmg/governance.py` | Admission, forgetting, quantile thresholds |
| `ocmg/reader.py` | Frozen backbone (simulated / HF) and judge |
| `ocmg/orchestrator.py` | Two-phase protocol |
| `ocmg/experiment.py` | Baselines, ablations, reporting |
| `ocmg/metrics.py` | Bootstrap, retention precision, iso-cost frontier |

## The 30 features

| Family | Count | Features |
|---|---|---|
| Structural | 8 | type one-hot (3), token length, entity count, temporal anchor, sentence count, digit ratio |
| Distinctiveness | 8 | max/top-5 similarity, novelty, mean/max idf, unique-token ratio, rare-term count, near-duplicate flag |
| Provenance | 7 | session index, position in session, user/assistant, turn span, modifies-existing, contradicts-existing |
| Historical | 7 | retrieval count, log count, recency gap, running utility, credit, weight, ever-retrieved |

Historical features are zero at write time. That is correct, not a bug: a
record with no history has no accumulated evidence, and the estimator learns
what that absence implies.

## Experimental conditions

| Condition | Admission | Forgetting | Budget |
|---|---|---|---|
| `full_context` | all | none | whole conversation |
| `naive_rag` | all, flat | none | fixed top-k |
| `append_all` | all, typed | none | fixed top-k |
| `decay` | all, typed | frequency + recency | fixed top-k |
| `generative_judge` | LLM-gated | none | fixed top-k |
| `ocmg` | utility-gated | utility-gated | adaptive |

## Using the real benchmarks

```bash
# LoCoMo      https://snap-research.github.io/locomo
# LongMemEval https://github.com/xiaowu0162/LongMemEval
python scripts/run_experiment.py \
    --dataset locomo --path data/locomo10.json \
    --embedder sentence-transformers --reader hf --runs 3
```

Neither benchmark annotates which turn supports which answer, so
`supporting_fact_ids` is empty for real data and the outcome signal comes from
the judge rather than from evidence bookkeeping. The orchestrator handles both
paths; nothing else changes.

## Reproducibility

Every run is seeded. `OCMGConfig.to_json()` dumps the full configuration.
`MemoryStore.export_ledger()` writes the per-record outcome dataset described
in Section 5 of the proposal.

## Licence

MIT for the code. The benchmarks carry their own licences.
