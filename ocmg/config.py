"""Central configuration.

Every threshold, budget and seed that affects a reported number lives here, so
that a run can be reproduced from a single object and so that no magic constant
is buried in a module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import json


@dataclass
class IngestConfig:
    # Segment boundaries are placed at local minima of consecutive-utterance
    # cosine similarity; this is the depth a minimum must have to count.
    #
    # Set from measurement, not intuition. 0.05 split so aggressively that
    # records averaged 1.56 turns and the max_segment_turns cap almost never
    # bound - raising the cap from 2 to 4 moved records only to 1.81 turns. The
    # naive fixed chunker averages 3.75. Against LoCoMo's 3.13 evidence turns
    # per multi-hop question, 1.56-turn records fragment the evidence across
    # two or three records, so retrieval needs several hits where a coarser
    # chunk needs one.
    #
    # Iso-cost gap, 3 seeds (scripts/sweep_budget.py --boundary-depth):
    #              vs naive RAG   vs append-all   vs decay   turns/rec
    #   0.05           -0.1106        -0.0168     +0.0373        1.56
    #   0.30           -0.0318        -0.0480     +0.0278        4.83
    # So 0.30 closes 71% of the deficit against the published-practice
    # baseline and raises OCMG's own best accuracy from 0.494 to 0.598. It
    # also WIDENS the gap to the ungoverned typed store, because governance's
    # cost scales with granularity: a coarser record carries more evidence, so
    # a wrong demotion destroys more. Both directions are reported; this is
    # not a setting chosen to flatter the comparison.
    boundary_depth: float = 0.30
    max_segment_turns: int = 6
    min_segment_chars: int = 25
    # Structurally uninformative turns removed by rule before segmentation.
    drop_short_turns_below_chars: int = 12
    # Ablation switch: route every segment to one type, so the three type
    # indicator features carry no information and the typed store collapses to
    # a flat one. This is the only honest way to ablate typing - changing the
    # segment length instead would alter what a record *is* and measure
    # something else entirely.
    flat_store: bool = False
    # Fixed-window chunking for the naive top-k RAG baseline: that many turns
    # per chunk, no similarity boundary detection, no coherence test. Zero
    # keeps the method's own segmenter. Four turns is chosen as twice
    # max_segment_turns, so the naive chunk spans more than one topic on
    # average - which is the cost of not segmenting, and the thing the
    # baseline is there to show.
    naive_chunk_turns: int = 0


@dataclass
class RetrievalConfig:
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    # Tier 1 shortlist size before Tier 2 expansion (coarse-to-fine).
    tier1_multiplier: int = 4
    k_min: int = 2
    k_max: int = 10
    k_fixed: int = 10  # for the fixed top-k baseline
    # Relevance discount applied to cold-tier records. They compete with
    # the hot tier on the same ranking rather than being gated by a
    # threshold, so a demoted record surfaces only when it is clearly the
    # best match available, and a continuous price avoids the cliff a
    # threshold introduces.
    #
    # It does NOT leave the prompt size unchanged, which is worth stating
    # because the opposite is the intuitive assumption. The budget k is set
    # adaptively from how many records score within `budget_mass_target` of
    # the best one, so widening the candidate pool admits more records above
    # that floor and k rises with it. Measured on LoCoMo at seed 0: enabling
    # demotion moved accuracy 0.465 -> 0.471 and tokens 430 -> 481, so
    # accuracy per 1k tokens fell from 1.083 to 0.979. Demotion buys back
    # reachability and costs efficiency; it is not free.
    #
    # Measured as a grid against write_quantile (sweep_budget.py --tiers),
    # LoCoMo, seed 0, 18 cells. Along the quantile axis:
    #     penalty 0.60:  0.632 0.632 0.632 0.633 0.611 0.568
    #     penalty 0.80:  0.632 0.632 0.632 0.633 0.619 0.592
    #     penalty 1.00:  0.632 0.632 0.632 0.633 0.635 0.635
    # at quantiles 0.00 0.05 0.10 0.20 0.35 0.50. At 1.00 the tiers are
    # identical to retrieval, so the gate cannot reach a prompt, and the row
    # is flat; below 1.00 the rows fall as the gate rejects more. The gate's
    # whole effect is therefore the discount, not the decision - the records
    # it rejects are not the ones worth rejecting. No cell in the grid beats
    # naive top-k at matched token cost: the best is -0.001 and the worst
    # -0.067, so the two-tier store is not a gain at any setting tested.
    #
    # It is also inert in the selected configuration: with write_quantile and
    # forget_quantile both 0.0 nothing is demoted or evicted, so the cold
    # tier is empty and this number has no effect. Kept at 0.6 so the grid
    # stays reproducible, not because 0.6 was chosen.
    cold_tier_penalty: float = 0.6
    # Stage 4 of the method sets the retrieval budget from the query type as
    # well as the utility distribution. The type is derived from the question
    # text by keyword cues (retrieval.query_type); the benchmark's own category
    # label is deliberately not read, because a deployed system would not have
    # it and using it is the same class of leakage the held-out evidence
    # annotations exist to avoid.
    #
    # The scale multiplies the relative-quality floor: below 1.0 lowers the bar
    # so more records clear it. Temporal questions have to compare two points
    # in time and enumerative ones have to collect several records, so both
    # need a wider budget than a question that turns on a single fact.
    # How strongly the utility estimate reorders the relevant set. The
    # multiplier is (1-w) + 2*w*prior: 0 ignores utility, 0.5 is the original
    # hard-coded 0.5+prior rule, 1 lets utility double or erase a score.
    # Now that admission and forgetting are both off by measurement, this is
    # where the estimator does its remaining work, so its strength should be
    # measured rather than assumed.
    #
    # Measured on LoCoMo over 3 seeds:
    #     w     accuracy          tokens   acc/1k
    #     0.00  0.6151 +- 0.0112   1,186    0.519
    #     0.25  0.6081 +- 0.0120   1,251    0.486
    #     0.50  0.5943 +- 0.0152   1,296    0.459
    #     0.75  0.5744 +- 0.0173   1,326    0.433
    #     1.00  0.5513 +- 0.0237   1,318    0.418
    # Monotone on all three seeds, 15 of 15 cells: utility reranking lowers
    # accuracy and raises cost at the same time. Tokens rise because the
    # multiplier reshapes the score distribution the adaptive-k floor is
    # measured against, so a worse ranking is also a wider one.
    #
    # The default is NOT set from this measurement, unlike write_quantile and
    # forget_quantile. At w=0 with both of those at 0 the estimator has no
    # causal path to any prompt at all - ROC-AUC 0.728 would be a real
    # measurement of a component that cannot affect the system's behaviour.
    # That is a decision about what the work claims, not a parameter, so it is
    # left to be made deliberately.
    #
    # The likely cause is in the per-category AUCs: single_hop 0.666,
    # temporal 0.650, multi_hop 0.630, open_domain 0.493, every one below the
    # pooled 0.723. Utility as implemented is P(this record is ever useful),
    # a property of the record alone, and multiplying a query-conditional
    # relevance score by a marginal one fights the signal that is actually
    # doing the work. A query-conditional U(m|q) is the fix this points to.
    utility_weight: float = 0.5
    budget_by_query_type: bool = True
    query_type_floor_scale: Dict[str, float] = field(default_factory=lambda: {
        "temporal": 0.85, "enumerative": 0.80, "simple": 1.0,
    })
    tokens_per_record: int = 40  # nominal; the real count comes from the text


@dataclass
class EstimatorConfig:
    # "gbdt" | "logistic" | "rarity" (one-feature heuristic ablation)
    model: str = "gbdt"
    n_splits: int = 5
    calibration: str = "sigmoid"  # Platt scaling; "isotonic" where folds allow
    class_weight_balanced: bool = True
    random_state: int = 0
    gbdt_max_iter: int = 200
    gbdt_learning_rate: float = 0.06
    gbdt_max_depth: Optional[int] = 4
    # Train U(m|q) instead of U(m): one row per observed (record, derived
    # query type) cell, with the query type one-hot appended to the record
    # features. The leave-one-out table is already per (record, question), so
    # the pooled estimator is discarding this signal rather than lacking it.
    #
    # Off by default so existing results reproduce exactly. Turning it on
    # changes the feature width, so a model fitted one way cannot score the
    # other and the flag has to match between fit and predict - which it does,
    # because both read it from this config.
    conditional_utility: bool = False


@dataclass
class GovernanceConfig:
    """Thresholds are quantiles, not absolute probabilities.

    Calibrated probabilities on this task cluster tightly around the dataset
    base rate, so an absolute cut such as 0.5 either admits everything or
    nothing. Quantiles keep the admitted fraction under direct control and make
    the operating point comparable across runs.
    """

    # Fraction of contested candidates demoted at write. Zero is the measured
    # optimum, like forget_quantile below, and for a sharper reason: the gate
    # cannot pay. Admission sweep, LoCoMo seed 0, retrieval budget held fixed
    # (scripts/sweep_budget.py --admission):
    #
    #   quantile   gated   evidence ceiling    acc   tok/q
    #     0.00      0.0%            100.0%   0.632   1,279
    #     0.15     15.1%            100.0%   0.632   1,279
    #     0.35     35.3%            100.0%   0.611   1,291
    #     0.50     49.6%            100.0%   0.568   1,288
    #
    # Tokens are flat and the ceiling never moves, because demotion is
    # non-destructive - so the gate saves nothing and loses nothing
    # permanently, yet accuracy still falls. The loss is entirely the
    # cold-tier discount pushing demoted records down the ranking. Gating 15%
    # is exactly free because those records were never retrieved anyway.
    #
    # Removing it, 3 seeds: OCMG accuracy 0.594 -> 0.615, tokens 1,296 ->
    # 1,287, parity with naive top-k on all three seeds, and the margin over
    # the decay baseline grows from +0.045 to +0.065.
    write_quantile: float = 0.0
    # Fraction of eligible records evicted per forgetting round, for the
    # governed policy. Zero is the MEASURED optimum, not a disabled feature:
    # scripts/compare_forgetting.py --quantiles over 3 seeds gives mean
    # accuracy 0.503 / 0.501 / 0.499 / 0.488 / 0.468 at 0.00 / 0.05 / 0.10 /
    # 0.25 / 0.40, falling strictly on all 15 seed-by-fraction cells with no
    # interior peak, and accuracy per 1k tokens falls with it (1.128 -> 1.013)
    # because eviction widens the candidate pool so the adaptive budget grows.
    # Set it above zero to reproduce the sweep or to run the stage deliberately.
    forget_quantile: float = 0.0
    # The decay baseline's fraction, kept separate and non-zero. Both policies
    # read a single field until 2026-10-30, which meant the governed optimum
    # above could not be adopted without silently disabling the baseline's
    # forgetting and collapsing the decay condition into the ungoverned one.
    decay_forget_quantile: float = 0.25
    min_retrievals_before_forget: int = 3
    # Which signal ranks records for eviction.
    #
    #   ledger     realised utility from the deployed credit rule. This is the
    #              original design and it has a measured defect: the deployed
    #              rule is rank-weighted co-occurrence, which the credit
    #              ablation showed trains a chance-level estimator (AUC 0.477
    #              against 0.727 for leave-one-out). Every retrieved record
    #              gets equal credit, so a record that merely sat beside a
    #              wrong answer scores zero whatever its own relevance, and
    #              "demonstrably useless" is indistinguishable from "never
    #              found out".
    #   estimator  the fitted utility estimator's write-time prediction. A far
    #              better signal, but static: it cannot learn from what
    #              happened after the write, so forgetting becomes a delayed
    #              admission gate and inherits that gate's uselessness.
    #   blend      the realised mean shrunk toward the estimator prior in
    #              proportion to how many observations back it. Experience
    #              where there is experience, prior where there is not, which
    #              is the actual problem with the ledger signal.
    # Measured on LoCoMo, 3 seeds, estimator signal, by
    # scripts/compare_forgetting.py --quantiles: accuracy falls monotonically
    # as this rises, on every seed, with no interior optimum - 0.503, 0.501,
    # 0.499, 0.488, 0.468 at 0.00/0.05/0.10/0.25/0.40, and accuracy per 1k
    # tokens falls with it (1.128 -> 1.013) because evicting records widens
    # the candidate pool and the adaptive budget grows. The measured optimum
    # is therefore no forgetting at all.
    #
    # The value is deliberately NOT set to 0.0. DecayPolicy reads this same
    # field, so zeroing it would disable the baseline's forgetting too and
    # collapse the decay condition into the ungoverned one. Holding the
    # fraction equal across both is what makes the comparison fair, so it
    # stays a matched experimental variable and the sweep above is the result.
    # "ledger" | "estimator" | "blend" | "recency"
    forget_signal: str = "ledger"  # "ledger" | "estimator" | "blend"
    # Pseudo-count for the blend: how many observations the prior is worth.
    forget_prior_weight: float = 5.0
    dedup_similarity: float = 0.92  # near-duplicates merge rather than store twice
    # Adaptive budget: concentrated utility mass -> small k.
    budget_mass_target: float = 0.55
    forget_every_n_queries: int = 50
    # Demote rejected candidates to the cold tier instead of discarding
    # them. Eviction in this design is already non-destructive; admission
    # was not, which left the same estimator making a recoverable decision
    # in one place and an irreversible one in the other.
    demote_instead_of_discard: bool = True


@dataclass
class ExplorationConfig:
    """Phase A must not be governed by the policy it is meant to train.

    If Phase A retrieves with the same narrow budget the governed phase uses,
    records the policy would never surface are never labelled, and the
    estimator is fitted on a population it created. Widening the budget and
    injecting random inclusions breaks that circularity.
    """

    k_min: int = 4
    k_max: int = 12
    epsilon_random_inclusion: float = 0.20
    admit_all: bool = True  # permissive admission during warm-up
    # Leave-one-out re-execution is the only credit rule that isolates a
    # record's marginal contribution. It costs |R| extra reader calls per
    # query, so it is applied to a sampled fraction of warm-up queries.
    loo_fraction: float = 1.0
    loo_min_observations: int = 2
    # Cap on how many of the |R| retrieved records get a counterfactual on a
    # question that was selected for leave-one-out. 0 keeps all of them, which
    # is the measured default and the only setting that labels every retrieved
    # record. A cap is a pure speed lever and it is not free: the records left
    # unablated get no ledger entry from that question at all, so label
    # coverage falls and `loo_min_observations` rejects more records.
    #
    # The sample is uniform over the retrieved set, NOT the top m. Ablating
    # only high-ranked records would label exactly the population that
    # `epsilon_random_inclusion` exists to widen, reinstating the
    # retrieval-selection bias by the back door - which is the first marker's
    # objection. A uniform sample leaves every record equally likely to be
    # labelled across questions, so the cost is variance, not bias.
    loo_max_records: int = 0


@dataclass
class ReaderConfig:
    kind: str = "simulated"  # "simulated" | "hf"
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    # CPU execution settings. These change how long a real-reader run takes,
    # and `max_new_tokens` also changes what the reader can say, so it is
    # recorded with every result rather than left to a library default.
    #
    # LoCoMo reference answers are short - a name, a date, a place - and the
    # judge scores token F1, so a long continuation can only add tokens that
    # dilute the F1 it is scored on. Generation is sequential and dominates
    # cost on CPU, so this is the strongest cost lever in the whole run.
    max_new_tokens: int = 24
    # Leave-one-out issues one prompt per retrieved record, all of which are
    # known at once, so they are generated as a batch.
    batch_size: int = 8
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    dtype: str = "auto"   # "auto" | "float32" | "bfloat16" | "float16"
    # Simulated reader response curve. p_correct given supporting evidence is
    # present, and given it is absent (a nonzero floor models prior knowledge).
    p_correct_with_evidence: float = 0.92
    p_correct_without_evidence: float = 0.04
    # Each irrelevant record in the prompt costs a little accuracy: this is the
    # "distraction" effect reported throughout the long-context literature.
    distractor_penalty: float = 0.012
    max_penalty: float = 0.35


@dataclass
class OCMGConfig:
    seed: int = 0
    embedding_dim: int = 256
    embedder: str = "auto"  # "auto" | "hashing" | "sentence-transformers"
    st_model: str = "BAAI/bge-small-en-v1.5"
    db_path: str = ":memory:"
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    credit_scheme: str = "rank_weighted"  # rank_weighted|uniform|top1|leave_one_out

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
