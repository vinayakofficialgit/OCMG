# Walkthrough for the supervisor meeting

This is the document to talk from. It explains what the code does, what it
found, the three bugs that had to be fixed to make it work at all, and the
questions you should expect and how to answer them.

---

## 1. The claim, stated plainly

Current memory systems decide what to keep using either a **decay schedule**
(keep what gets retrieved often) or a **language model called per record**
(expensive, and inconsistent between runs). Neither consults the one piece of
evidence that actually bears on the question: *when this record was in the
prompt, was the answer right?*

OCMG harvests that signal, trains a small supervised model on it, and uses the
model in place of both mechanisms. The claim is not that this is smarter than an
LLM judge. It is that it is **nearly free** and gets you to a comparable place.

---

## 2. Demo script for the meeting

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                       # 14 passed
python scripts/run_experiment.py --conversations 40 --runs 2
python scripts/sweep_budget.py --conversations 16 --ablations
```

Roughly two minutes total, CPU only, no downloads.

---

## 3. What the numbers say

Two seeds, 40 synthetic conversations, hashing encoder, simulated reader:

| Condition | Accuracy | Tokens/query | Accuracy per 1k tokens | Generative calls |
|---|---|---|---|---|
| full_context | 0.349 | 759.1 | 0.46 | 0 |
| naive_rag | 0.455 | 127.0 | 3.58 | 0 |
| append_all | 0.478 | 127.6 | 3.75 | 0 |
| decay | 0.463 | 127.9 | 3.62 | 0 |
| generative_judge | 0.447 | 127.6 | 3.50 | **1118** |
| **ocmg** | **0.478** | **24.8** | **19.3** | **0** |

Read it this way, and only this way:

> **OCMG matches the best baseline's accuracy using one fifth of the tokens and
> zero generative calls during memory construction.** Paired bootstrap over
> questions puts the accuracy difference against `append_all` at +0.005 with a
> 95% interval of [-0.029, +0.039] — indistinguishable. The difference is
> entirely in cost.

Supporting numbers:
- Estimator ROC-AUC **0.644 ± 0.032** (5-fold, grouped by conversation), on 452
  labelled records at a 35% positive rate. Above the 0.70 target? No. Above
  chance and stable across folds? Yes.
- Accuracy-retention ratio vs full context: **1.36** — governance beats stuffing
  everything in, because irrelevant context actively hurts.
- Token-reduction ratio vs fixed top-k: **0.80**, against a 0.40 target.

### The honest counterweight

`scripts/sweep_budget.py` sweeps the retrieval budget and compares accuracy at
*matched* token spend. At 16 conversations, **OCMG does not win that
comparison** — the baselines are ahead by roughly 0.25 accuracy across the
overlapping budget range.

Do not hide this. Say it first, and explain why:

- At 16 conversations the warm-up produces only ~150 labelled records. The
  estimator is weak, and an admission gate driven by a weak estimator discards
  useful records along with useless ones. This is **risk R1 in your proposal
  (sparse positive feedback)** showing up exactly where the proposal predicted.
- At 40 conversations the estimator stabilises and the gap closes. The
  mechanism is data-hungry, which is a finding worth reporting rather than an
  embarrassment.
- The ablation table shows the same thing from the other side: removing the
  admission gate *raises* accuracy (0.482 vs 0.399) while *halving* efficiency
  (16.1 vs 23.9 accuracy per 1k tokens). The gate trades accuracy for cost, and
  how good that trade is depends on how good the estimator is.

**So the contribution as currently evidenced is efficiency, not accuracy.** That
is a narrower claim than the proposal's framing and you should adjust the
framing rather than overstate the result.

---

## 4. The three bugs — have this ready, it is the best material you have

These are what a good examiner will probe, and each has a clean answer.

### Bug 1 — the supervision signal was 93% noise

**Symptom:** estimator ROC-AUC 0.30, i.e. *worse than chance*.

**Cause:** exploratory retrieval returns ~16 records per query, of which one or
two carry the answer. Rank-weighted credit gives nearly the same credit to the
record that answered the question and to the fourteen sitting beside it. The
label was almost pure co-occurrence.

**Fix:** leave-one-out credit in warm-up. Remove each record, re-run the reader,
observe whether the answer changes. Uses nothing but the downstream correctness
signal; costs |R| extra reader calls, which is why it is warm-up only and why
deployment falls back to the cheap rank-weighted rule.

**Result:** AUC 0.30 → 0.565. This is risk R2 in your proposal, found and
mitigated exactly as written.

### Bug 2 — memory leaked between conversations

**Symptom:** OCMG's store collapsed to 2 records per conversation.

**Cause:** one global store. A query from conversation B could retrieve
conversation A's records, and deduplication merged unrelated users' turns into
each other — destroying 88% of candidates.

**Fix:** the index is partitioned by conversation. Retrieval, deduplication,
feature neighbours and forgetting are all scoped. `test_store_is_partitioned_by_conversation`
locks this down.

**Why it matters:** a memory store belongs to one dialogue. This is a semantics
bug, not a performance bug, and it silently invalidated every number before it.

### Bug 3 — train/serve skew in the threshold calibration

**Symptom:** the admission gate discarded *everything*; store size zero.

**Cause:** threshold calibration and the admission path computed features
through two different code paths. One passed neighbour texts, the other did not,
so `modifies_existing` and `contradicts_existing` differed. The threshold was
fitted to a score distribution the gate never actually saw.

**Fix:** one method, `OCMGRunner.features_for`, used by calibration, admission
and re-scoring alike.

**The general lesson:** two code paths computing the same features is a bug
waiting to happen. Say this — it is the kind of remark that reads as engineering
maturity.

---

## 5. Two design decisions worth defending

**Quantile thresholds, not absolute ones.** Calibrated utility probabilities
cluster tightly around the dataset base rate — most candidates look similar ex
ante. A cut at `U ≥ 0.5` therefore admits nearly everything or nearly nothing
depending on which side of the base rate it lands, and swings between runs. A
quantile fixes the admitted *fraction* directly.

**Calibrate on the contested population only.** Near-duplicates are merged
*before* the gate, so they never contest admission. Computing the threshold over
the raw candidate stream fits it to a population including records the gate
never judges, and the effective admission rate drifts. Threshold is computed
post-deduplication.

---

## 6. Anticipated questions

**"Your reader is simulated. Isn't this all fake?"**
The reader is a modelled response curve, and absolute accuracies from it are not
findings — I say so in the README. What it preserves exactly is the causal chain
the mechanism depends on: evidence in prompt → answer correct → credit to
record. That is what needed validating before spending GPU hours. `HFReader` is
implemented and swaps in with `--reader hf`; the dissertation numbers come from
there on LoCoMo.

**"ROC-AUC 0.644 is not very good."**
Agreed, and it is below the 0.70 target in Objective 3. Two things: it is stable
across folds (±0.032), so it is real signal rather than noise; and it is
achieved from features costing no generative call. The comparison is not against
a perfect oracle, it is against a decay schedule, which carries no signal at all.

**"Why does OCMG lose the iso-cost comparison?"**
At small data the estimator is weak and the gate over-prunes. See section 3. It
is risk R1 materialising, and the fix is more warm-up data, not a different
mechanism.

**"How do I know the estimator isn't just memorising conversations?"**
Folds are grouped by conversation, not by record. A record-level split would
leak, since records from one conversation share vocabulary and topic.

**"What happens when you delete someone's data?"**
Eviction is non-destructive — inactive flag, cold store, row in `eviction_log`.
That gives auditability. Erasure is separate and destructive; the cold store is
a storage tier, not an exemption from deletion.

---

## 7. What to do next, in order

1. **More warm-up data.** The estimator is the bottleneck. Run with 100+
   conversations and re-check whether the iso-cost gap closes.
2. **Real encoder.** `--embedder sentence-transformers`. The hashing encoder
   makes distinct short sentences look alike, which weakens both dedup and the
   distinctiveness features.
3. **Real benchmark and real backbone.** LoCoMo with `--reader hf`. Nothing else
   in the pipeline changes.
4. **Re-run the frontier** and report it as the headline analysis. A single
   operating point cannot settle a comparison between policies that spend
   different amounts.
5. **Then, and only then**, decide whether the dissertation claims accuracy
   gains or efficiency gains. On current evidence it is efficiency.
