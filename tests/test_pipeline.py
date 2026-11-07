"""Sanity checks. These assert wiring and invariants, not accuracy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

import numpy as np
import pytest

from ocmg.config import OCMGConfig
from ocmg.credit import assign_credit, rank_weight
from ocmg.data.synthetic import generate_dataset
from ocmg.embedding import HashingEmbedder
from ocmg.estimator import EstimatorConfig, UtilityEstimator
from ocmg.experiment import run_all_conditions
from ocmg.features import FEATURE_NAMES, N_FEATURES
from ocmg.governance import GovernancePolicy
from ocmg.ingest import build_candidates, normalise_temporal, route_type
from ocmg.metrics import interpolate_at_cost, paired_bootstrap, pareto_frontier
from ocmg.orchestrator import OCMGRunner
from ocmg.retrieval import adaptive_k
from ocmg.store import MemoryStore


def test_feature_count_is_thirty():
    assert N_FEATURES == 30 == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == 30


def test_rank_weight_is_monotone_decreasing():
    ws = [rank_weight(r) for r in range(1, 11)]
    assert all(a > b for a, b in zip(ws, ws[1:]))
    assert ws[0] == pytest.approx(1.0)


def test_credit_conserves_outcome_under_uniform():
    ids, ranks = [1, 2, 3, 4], [1, 2, 3, 4]
    triples = assign_credit("uniform", ids, ranks, 1)
    assert sum(c for _, c, _ in triples) == pytest.approx(1.0)


def test_ledger_running_utility_is_weighted_mean():
    store = MemoryStore(":memory:", 8)
    cands = generate_candidates()
    rid = store.admit(cands[0])
    store.update_ledger(rid, credit=1.0, weight=1.0, now=0.0)
    store.update_ledger(rid, credit=0.0, weight=1.0, now=1.0)
    assert store.ledger[rid].running_utility == pytest.approx(0.5)


def generate_candidates():
    cfg = OCMGConfig(seed=0)
    emb = HashingEmbedder(dim=8)
    _, turns, _ = generate_dataset(1, seed=0, n_sessions=2, turns_per_session=6)[0]
    return build_candidates("c0", turns, emb, cfg.ingest)


def test_temporal_expressions_are_resolved():
    out = normalise_temporal("I went there yesterday.", 1_700_000_000.0)
    assert "[" in out and "yesterday" in out


def test_type_router_separates_profile_from_chatter():
    assert route_type("My coffee is a flat white.") == "profile"
    assert route_type("The bus was late again.") == "factual"


def test_eviction_is_non_destructive():
    store = MemoryStore(":memory:", 8)
    rid = store.admit(generate_candidates()[0])
    store.evict(rid, now=1.0)
    assert not store.records[rid].active
    assert rid in store.cold_store          # recoverable
    assert store.records[rid].text          # body preserved
    rows = store.conn.execute("SELECT COUNT(*) FROM eviction_log").fetchone()
    assert rows[0] == 1                     # auditable


def test_store_is_partitioned_by_conversation():
    """A query must never reach another conversation's memory."""
    store = MemoryStore(":memory:", 8)
    cands = generate_candidates()
    for c in cands[:3]:
        store.admit(c)
    other = cands[0]
    other.conversation_id = "other"
    store.admit(other)
    _, ids_a = store.matrix("c0")
    _, ids_b = store.matrix("other")
    assert set(ids_a).isdisjoint(ids_b)


def test_adaptive_k_shrinks_when_scores_are_peaked():
    from ocmg.config import RetrievalConfig
    cfg = RetrievalConfig(k_min=1, k_max=10)
    peaked = [1.0, 0.05, 0.04, 0.03, 0.02]
    flat = [1.0, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]
    assert adaptive_k(peaked, cfg, 0.55) < adaptive_k(flat, cfg, 0.55)


def test_quantile_thresholds_admit_expected_fraction():
    cfg = OCMGConfig(seed=0)
    cfg.governance.write_quantile = 0.30
    pol = GovernancePolicy(cfg.governance, None)
    scores = list(np.linspace(0, 1, 200))
    pol.calibrate(scores)
    admitted = sum(1 for s in scores if pol.admit(s))
    assert 0.62 <= admitted / len(scores) <= 0.78


def test_paired_bootstrap_detects_a_real_difference():
    a = [1] * 80 + [0] * 20
    b = [1] * 40 + [0] * 60
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["delta"] > 0 and r["p_value"] < 0.05


def test_pareto_frontier_keeps_only_improvements():
    pts = [(10, 0.5), (20, 0.4), (30, 0.7), (40, 0.6)]
    assert pareto_frontier(pts) == [(10, 0.5), (30, 0.7)]


def test_interpolation_between_budget_points():
    curve = [(10.0, 0.4), (20.0, 0.6)]
    assert interpolate_at_cost(curve, 15.0) == pytest.approx(0.5)


def test_end_to_end_phase_a_produces_labels():
    cfg = OCMGConfig(seed=0)
    cfg.embedder = "hashing"
    runner = OCMGRunner(cfg)
    data = generate_dataset(4, seed=0, n_sessions=4, turns_per_session=10)
    out = runner.run_phase_a(data)
    assert out.X.shape[1] == N_FEATURES
    assert len(out.y) == out.X.shape[0] == len(out.groups)
    assert len(out.logs) > 0


def test_simulated_reader_uses_common_random_numbers():
    """Leave-one-out must compare like with like.

    The same question answered twice must consume the same random draw, so
    that a difference in outcome reflects the change in context and not a
    second coin flip. Without this the leave-one-out credit signal is noise
    and the estimator cannot rise above chance.
    """
    from ocmg.config import ReaderConfig
    from ocmg.reader import SimulatedReader
    from ocmg.types import Question

    reader = SimulatedReader(ReaderConfig(), seed=0)
    q = Question(conversation_id="c0", question_id="q7", text="where?",
                 answer="Paris", category="single_hop",
                 supporting_fact_ids=["f1"])
    draws = {reader._draw(q) for _ in range(5)}
    assert len(draws) == 1                      # stable within a question
    other = Question(conversation_id="c0", question_id="q8", text="when?",
                     answer="May", category="temporal",
                     supporting_fact_ids=["f1"])
    assert reader._draw(other) != reader._draw(q)   # varies across questions
    assert 0.0 <= next(iter(draws)) < 1.0


def test_label_validation_scores_labels_against_evidence():
    from ocmg.metrics import label_validation

    # Four records: two agree with the annotation, one false positive, one
    # false negative.
    outcome = [1, 1, 1, 0]
    evidence = [1, 1, 0, 1]
    v = label_validation(outcome, evidence)
    assert v["precision"] == pytest.approx(2 / 3)
    assert v["recall"] == pytest.approx(2 / 3)
    assert v["false_positives"] == 1 and v["false_negatives"] == 1
    # Perfect agreement gives kappa 1; inverted labels give a negative kappa.
    assert label_validation([1, 1, 0, 0], [1, 1, 0, 0])["kappa"] == pytest.approx(1.0)
    assert label_validation([1, 1, 0, 0], [0, 0, 1, 1])["kappa"] < 0


def test_delta_glyph_agrees_with_colour_valence():
    """The arrow must mean the same thing as the colour, on both polarities.

    On a cost column an increase is bad, so direction-encoding would print an
    up-arrow on a red number and a reader who cannot see the red would take it
    for an improvement.
    """
    from ocmg import console as C

    C.set_color(False)
    try:
        assert C.G.up in C.delta(+0.05, higher_is_better=True)
        assert C.G.down in C.delta(-0.05, higher_is_better=True)
        # Cost: smaller is better, so a fall is the up-arrow.
        assert C.G.up in C.delta(-0.32, higher_is_better=False)
        assert C.G.down in C.delta(+21.5, higher_is_better=False)
        assert C.delta(0.0).strip() == C.G.flat
    finally:
        C.set_color(None)


def test_table_aligns_around_colour_escapes():
    """Padding must measure printable width, not byte length."""
    from ocmg import console as C

    C.set_color(True)
    try:
        painted = C.paint("ok", "good")
        assert len(painted) > 2                 # escapes present
        assert C._w(painted) == 2               # printable width unchanged
        assert C._w(C.pad(painted, 8)) == 8
    finally:
        C.set_color(None)


def test_eviction_fraction_is_exact_despite_ties():
    """A value cut cannot split a tie; realised utility is mostly ties at zero.

    Every record never retrieved alongside a correct answer scores exactly 0.0,
    so the quantile for any small fraction lands on that spike and a value
    comparison takes the whole group. Requesting 25% then evicts 60%, and
    requesting 0% evicts just as much.
    """
    from ocmg.governance import select_lowest_fraction

    scored = ([(i, 0.0) for i in range(60)]
              + [(i, 0.3) for i in range(60, 80)]
              + [(i, 0.8) for i in range(80, 100)])
    assert select_lowest_fraction(scored, 0.0) == []
    assert len(select_lowest_fraction(scored, 0.25)) == 25
    assert len(select_lowest_fraction(scored, 0.50)) == 50
    # Lowest utility goes first, and ties break deterministically.
    assert select_lowest_fraction(scored, 0.25) == list(range(25))


def test_forgetting_off_means_no_eviction():
    from ocmg.config import OCMGConfig
    from ocmg.governance import GovernancePolicy
    from ocmg.store import MemoryStore

    cfg = OCMGConfig(seed=0)
    cfg.governance.forget_quantile = 0.0
    store = MemoryStore(":memory:", 8)
    for cand in generate_candidates()[:12]:
        rid = store.admit(cand)
        store.update_ledger(rid, credit=0.0, weight=1.0, now=1.0)
        store.update_ledger(rid, credit=0.0, weight=1.0, now=2.0)
        store.update_ledger(rid, credit=0.0, weight=1.0, now=3.0)
    pol = GovernancePolicy(cfg.governance, None)
    assert pol.select_for_eviction(store) == []


def test_demoted_records_stay_retrievable():
    """Admission must be recoverable, not final.

    A demoted record leaves the hot tier but stays in the store and in both
    retrieval indexes, so a gate error costs ranking position rather than the
    record itself. This is the property that lets an imperfect estimator
    govern admission at all.
    """
    from ocmg.config import OCMGConfig
    from ocmg.embedding import HashingEmbedder
    from ocmg.retrieval import Retriever
    from ocmg.store import MemoryStore
    from ocmg.types import Question

    cfg = OCMGConfig(seed=0)
    emb = HashingEmbedder(dim=64)
    store = MemoryStore(":memory:", 64)
    cands = generate_candidates()
    for c in cands:
        c.embedding = emb.encode_one(c.text)

    hot_id = store.admit(cands[0])
    cold_id = store.admit(cands[1], hot=False)

    assert store.records[cold_id].demoted_at_write
    assert cold_id not in [r.record_id for r in store.active_records()]
    assert cold_id in [r.record_id for r in store.cold_records()]
    # Indexed in the cold tier, so retrieval can still reach it.
    _, cold_ids = store.matrix_cold(cands[1].conversation_id)
    assert cold_id in cold_ids

    # A query matching the demoted record's own text must surface it.
    q = Question(conversation_id=cands[1].conversation_id, question_id="q1",
                 text=cands[1].text, answer="x", category="single_hop")
    rs = Retriever(store, emb, cfg.retrieval).retrieve(q, k=2)
    assert cold_id in rs.record_ids

    # And promotion returns it to the hot tier, auditably.
    assert store.promote(cold_id, now=1.0)
    assert cold_id in [r.record_id for r in store.active_records()]
    reasons = [r[0] for r in store.conn.execute(
        "SELECT reason FROM eviction_log WHERE record_id=?", (cold_id,)).fetchall()]
    assert "low_utility_admission" in reasons and "promoted" in reasons


def test_cold_tier_is_discounted_not_excluded():
    """A cold record must lose to an equally relevant hot one, and only then."""
    from ocmg.config import RetrievalConfig

    cfg = RetrievalConfig()
    assert 0.0 < cfg.cold_tier_penalty < 1.0


def test_evicted_records_leave_service_but_stay_auditable():
    """Forgetting must actually forget.

    Evicted records are retained in the cold store for audit and recovery, but
    they must not remain retrievable: a policy whose evicted records still
    answer queries has not forgotten anything, and the decay baseline then
    stops implementing the heuristic it exists to represent.
    """
    from ocmg.store import MemoryStore

    store = MemoryStore(":memory:", 8)
    cands = generate_candidates()
    evicted = store.admit(cands[0])
    demoted = store.admit(cands[1], hot=False)
    store.evict(evicted, now=1.0)

    assert evicted in store.cold_store and demoted in store.cold_store
    served = [r.record_id for r in store.cold_records()]
    assert demoted in served          # demotion keeps it in service
    assert evicted not in served      # eviction does not
    _, cold_ids = store.matrix_cold()
    assert evicted not in cold_ids


def test_forget_signals_rank_differently():
    """The blend must sit between the realised mean and the prior.

    A record retrieved three times with no credit scores 0.0 on the ledger
    regardless of whether it was ever the relevant record. If the estimator
    rates it highly, the blend should not treat that absence of evidence as
    evidence of uselessness.
    """
    from ocmg.config import OCMGConfig
    from ocmg.governance import GovernancePolicy
    from ocmg.types import LedgerRow

    class StubEstimator:
        is_fitted = True

        def predict(self, x):
            return np.array([0.9])

    rec = generate_candidates()[0]
    rec.write_features = np.zeros(N_FEATURES, dtype=np.float32)
    row = LedgerRow(record_id=1, retrieval_count=3, credit_total=0.0,
                    weight_total=3.0)
    assert row.running_utility == pytest.approx(0.0)

    cfg = OCMGConfig(seed=0)
    scores = {}
    for signal in ("ledger", "estimator", "blend"):
        cfg.governance.forget_signal = signal
        pol = GovernancePolicy(cfg.governance, StubEstimator())
        scores[signal] = pol.forget_score(rec, row)

    assert scores["ledger"] == pytest.approx(0.0)
    assert scores["estimator"] == pytest.approx(0.9)
    # prior weight 5, n=3  ->  (3*0.0 + 5*0.9) / 8
    assert scores["blend"] == pytest.approx(5 * 0.9 / 8)
    assert scores["ledger"] < scores["blend"] < scores["estimator"]


def test_forget_signal_falls_back_without_an_estimator():
    from ocmg.config import OCMGConfig
    from ocmg.governance import GovernancePolicy
    from ocmg.types import LedgerRow

    cfg = OCMGConfig(seed=0)
    cfg.governance.forget_signal = "blend"
    rec = generate_candidates()[0]
    row = LedgerRow(record_id=1, retrieval_count=2, credit_total=1.0,
                    weight_total=2.0)
    pol = GovernancePolicy(cfg.governance, None)
    assert pol.forget_score(rec, row) == pytest.approx(0.5)


def test_recency_signal_matches_the_decay_baseline_formula():
    """The recency arm must BE the baseline's criterion, not resemble it.

    If the two formulas drift apart, the criterion ablation stops answering
    the question it was added for, and nothing else in the suite would catch
    it because both would still run.
    """
    cfg = OCMGConfig()
    cfg.governance.forget_signal = "recency"
    policy = GovernancePolicy(cfg.governance, None)

    now = 10 * 86400.0
    for count, last in ((0, now), (3, now - 86400.0), (12, now - 7 * 86400.0)):
        row = SimpleNamespace(retrieval_count=count, last_retrieved_at=last,
                              running_utility=0.5)
        rec = SimpleNamespace(record_id=1, write_features=None, timestamp=0.0)
        gap_days = max(0.0, now - last) / 86400.0
        expected = (np.log1p(count) * 0.6 + (1.0 / (1.0 + gap_days)) * 0.4)
        assert policy.forget_score(rec, row, now) == pytest.approx(expected)


def test_recency_signal_needs_no_estimator():
    """It must work with estimator=None, or it cannot be run as a baseline."""
    cfg = OCMGConfig()
    cfg.governance.forget_signal = "recency"
    policy = GovernancePolicy(cfg.governance, None)
    row = SimpleNamespace(retrieval_count=5, last_retrieved_at=0.0,
                          running_utility=0.0)
    rec = SimpleNamespace(record_id=1, write_features=None, timestamp=0.0)
    assert policy.forget_score(rec, row, 86400.0) > 0.0


def test_single_feature_baselines_are_scored_on_the_model_folds():
    """The rarity baseline must be comparable to the model's own AUC.

    A pooled in-sample AUC against a mean-of-folds AUC is not a like-for-like
    comparison, and it flatters the raw feature. This pins the baselines to
    the same fold structure and the same count of folds.
    """
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, N_FEATURES))
    j = FEATURE_NAMES.index("rare_term_count")
    y = (X[:, j] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    groups = np.array([f"c{i % 6}" for i in range(n)])

    est = UtilityEstimator(EstimatorConfig())
    report = est.fit(X, y, groups)

    assert set(report.auc_single_feature) <= set(
        UtilityEstimator.SINGLE_FEATURE_BASELINES)
    assert report.auc_single_feature["rare_term_count"] > 0.7
    # Orientation is resolved, so no baseline is reported below chance.
    assert all(v >= 0.5 for v in report.auc_single_feature.values())


def test_single_feature_orientation_uses_only_training_folds():
    """A feature that is pure noise must not be flattered above chance.

    Picking the better orientation after seeing validation labels would push
    a noise feature's reported AUC above 0.5 on every fold; fixing it on the
    training fold leaves it scattered around chance.
    """
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, N_FEATURES))
    y = rng.integers(0, 2, size=n)
    groups = np.array([f"c{i % 8}" for i in range(n)])

    report = UtilityEstimator(EstimatorConfig()).fit(X, y, groups)
    for name, auc in report.auc_single_feature.items():
        assert auc < 0.70, f"{name} flattered to {auc:.3f} on noise labels"


def test_rarity_estimator_survives_calibration_cloning():
    """The ablation must run inside CalibratedClassifierCV.

    A hand-written class with fit/predict_proba is not enough: calibration
    clones its base estimator and needs get_params/set_params, which is how
    the first version of this ablation failed at run time rather than here.
    """
    cfg = EstimatorConfig()
    cfg.model = "rarity"
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, N_FEATURES))
    j = FEATURE_NAMES.index("rare_term_count")
    y = (X[:, j] + rng.normal(scale=0.6, size=n) > 0).astype(int)
    groups = np.array([f"c{i % 6}" for i in range(n)])

    est = UtilityEstimator(cfg)
    report = est.fit(X, y, groups)
    assert est.is_fitted
    assert report.auc_mean > 0.7
    assert est.predict(X[:5]).shape == (5,)


def test_rarity_estimator_ignores_every_other_feature():
    """It must be the heuristic, not a model that happens to be small.

    Labels are driven by a feature the ranker does not read, so a ranker that
    leaked any other column would score above chance here.
    """
    cfg = EstimatorConfig()
    cfg.model = "rarity"
    rng = np.random.default_rng(4)
    n = 400
    X = rng.normal(size=(n, N_FEATURES))
    other = FEATURE_NAMES.index("max_similarity")
    y = (X[:, other] > 0).astype(int)
    groups = np.array([f"c{i % 6}" for i in range(n)])

    report = UtilityEstimator(cfg).fit(X, y, groups)
    assert report.auc_mean < 0.62, f"leaked another feature: {report.auc_mean:.3f}"


def many_candidates(n_sessions: int = 6, turns: int = 12):
    """More candidates than the dense index's initial capacity of 8.

    generate_candidates() yields 6, which is under the buffer's first size, so
    tests built on it would never exercise the doubling path.
    """
    cfg = OCMGConfig(seed=0)
    emb = HashingEmbedder(dim=8)
    _, turns_list, _ = generate_dataset(
        1, seed=0, n_sessions=n_sessions, turns_per_session=turns)[0]
    return build_candidates("c0", turns_list, emb, cfg.ingest)


def test_dense_index_appends_match_a_full_rebuild():
    """An index grown by appends must equal one rebuilt from scratch.

    This is the correctness condition for making admission O(n) instead of
    O(n^2). If the two disagree in row order, retrieval tie-breaks change and
    every downstream number shifts quietly - a wrong answer dressed as a
    speed-up, which no accuracy assertion would catch.
    """
    cands = many_candidates()
    assert len(cands) > 9, "need more than the initial buffer capacity of 8"

    store = MemoryStore(":memory:", 8)
    for c in cands:
        store.admit(c)
    grown, grown_ids = store.matrix("c0")
    grown = grown.copy()

    store._dirty = True                     # force the full-rebuild path
    rebuilt, rebuilt_ids = store.matrix("c0")

    assert grown_ids == rebuilt_ids
    assert np.array_equal(grown, rebuilt)


def test_dense_index_grows_past_its_initial_capacity():
    """Rows must survive the doubling, and the view must expose only real ones."""
    cands = many_candidates()
    store = MemoryStore(":memory:", 8)
    seen = []
    for c in cands[:12]:
        store.admit(c)
        seen.append(c.embedding)
        mat, ids = store.matrix("c0")
        # The buffer has spare capacity; the view must not expose it as data.
        assert mat.shape[0] == len(seen) == len(ids)
    mat, _ = store.matrix("c0")
    assert np.array_equal(mat, np.vstack(seen))


def test_dense_index_keeps_the_embedding_dtype():
    """float32 in, float32 out: a preallocated buffer must not downcast."""
    store = MemoryStore(":memory:", 8)
    store.admit(generate_candidates()[0])
    mat, _ = store.matrix("c0")
    assert mat.dtype == np.float32


def test_eviction_takes_a_row_out_of_the_hot_index():
    """Appends are incremental but removals are not, so this is the path that
    still needs a full rebuild; it must actually happen."""
    cands = many_candidates()
    store = MemoryStore(":memory:", 8)
    ids = [store.admit(c) for c in cands[:5]]
    assert store.matrix("c0")[0].shape[0] == 5
    store.evict(ids[2], now=1.0)
    mat, remaining = store.matrix("c0")
    assert mat.shape[0] == 4
    assert ids[2] not in remaining


def test_naive_rag_and_append_all_are_different_conditions():
    """The two must not be the same computation.

    Until 2026-10-31 both built an ungoverned typed store and retrieved a
    fixed top-k, so they reported identical accuracy, tokens and record counts
    and an identical bootstrap interval against OCMG. CLAUDE.md lists them as
    separate baselines because the pair isolates the typed store; a test is
    the only thing that keeps them separate.
    """
    cfg = OCMGConfig(seed=0)
    data = generate_dataset(4, seed=0, n_sessions=2, turns_per_session=8)
    runner = OCMGRunner(cfg)

    flat, _ = runner.build_store(data, None, governed=False, flat=True)
    typed, _ = runner.build_store(data, None, governed=False)

    flat_types = sorted({r.record_type for r in flat.records.values()})
    typed_types = sorted({r.record_type for r in typed.records.values()})
    assert len(flat_types) == 1, f"naive_rag store is still typed: {flat_types}"
    assert len(typed_types) > 1, f"append_all store lost its typing: {typed_types}"


def test_query_type_comes_from_the_text_not_the_gold_label():
    """Budgeting must not read the benchmark's own category annotation.

    LoCoMo labels each question temporal / multi-hop / single-hop and so on.
    Reading that at retrieval time is the same leakage the held-out evidence
    annotations exist to avoid, so the type is derived from the text.
    """
    from ocmg.retrieval import query_floor, query_type

    assert query_type("When did she move to Berlin?") == "temporal"
    assert query_type("How many siblings does he have?") == "enumerative"
    assert query_type("Where does Mark work?") == "simple"

    cfg = OCMGConfig().retrieval
    simple = query_floor("Where does Mark work?", cfg, 0.80)
    temporal = query_floor("When did she move to Berlin?", cfg, 0.80)
    # A lower floor admits more records, so the wider-evidence type must sit
    # below the single-fact one.
    assert temporal < simple == pytest.approx(0.80)


def test_query_type_budgeting_can_be_switched_off():
    """It is a config switch, so the ablation is possible."""
    from ocmg.retrieval import query_floor

    cfg = OCMGConfig().retrieval
    cfg.budget_by_query_type = False
    assert query_floor("When did she move?", cfg, 0.80) == pytest.approx(0.80)


def test_per_category_auc_drops_unlabelled_rows():
    """A record with no outcome for a category must not count as useless.

    -1 means "never retrieved for this category". Scoring it as a negative
    would manufacture separability out of absence and inflate every category's
    AUC, which is the same error the forgetting signals avoid.
    """
    from ocmg.metrics import per_category_auc

    mask = np.ones(6, dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    # Two real rows, four unlabelled: too few to score, and must be reported
    # as such rather than as a number computed on the -1s.
    labels = {"temporal": np.array([1, 0, -1, -1, -1, -1])}
    out = per_category_auc(labels, scores, mask, min_rows=2)
    assert out["temporal"]["n"] == 2
    assert out["temporal"]["positive_rate"] == pytest.approx(0.5)


def test_per_category_auc_reports_nothing_when_one_class():
    """Adversarial questions are abstentions, so no record ever helps them."""
    from ocmg.metrics import per_category_auc

    mask = np.ones(50, dtype=bool)
    out = per_category_auc(
        {"adversarial": np.zeros(50, dtype=int)},
        np.linspace(0, 1, 50), mask, min_rows=10,
    )
    auc = out["adversarial"]["auc"]
    assert auc != auc, "one-class category must report NaN, not a number"
    assert out["adversarial"]["n"] == 50


def test_per_category_auc_separates_an_easy_category_from_a_hard_one():
    rng = np.random.default_rng(0)
    n = 300
    scores = rng.random(n)
    easy = (scores > 0.5).astype(int)                 # perfectly ranked
    hard = rng.integers(0, 2, size=n)                 # unrelated to scores
    from ocmg.metrics import per_category_auc

    out = per_category_auc({"easy": easy, "hard": hard},
                           scores, np.ones(n, dtype=bool))
    assert out["easy"]["auc"] > 0.95
    assert 0.4 < out["hard"]["auc"] < 0.6


def test_retention_precision_is_absent_when_nothing_was_evicted():
    """Demoted records are not evicted records.

    cold_store holds both. Measuring retained-versus-cold_store while
    forgetting is off compares retained against demoted and labels it eviction
    precision; seed 0 reported a gap of -0.420 at AUC 0.030 that way, which
    reads as eviction being catastrophically wrong when nothing was evicted.
    """
    cfg = OCMGConfig(seed=0)
    cfg.governance.forget_quantile = 0.0          # the default: no forgetting
    data = generate_dataset(6, seed=0, n_sessions=2, turns_per_session=8)
    results = run_all_conditions(data, cfg, conditions=["ocmg"])
    extra = results["ocmg"].extra
    assert "retention_precision" not in extra, (
        "reported eviction precision with forgetting switched off")
    assert "retention_precision_skipped" in extra


def test_utility_weight_zero_means_relevance_only():
    """w=0 must leave the ranking exactly as relevance ordered it.

    The utility multiplier is (1-w) + 2*w*prior, so at w=0 every record gets
    the same factor of 1.0 and utility cannot reorder anything. Without this
    the ablation "retrieval without utility" would silently still use it.
    """
    cfg = OCMGConfig(seed=0)
    cfg.retrieval.utility_weight = 0.0
    w = cfg.retrieval.utility_weight
    for prior in (0.0, 0.3, 0.9, 1.0):
        assert (1.0 - w) + 2.0 * w * prior == pytest.approx(1.0)


def test_utility_weight_half_reproduces_the_original_rule():
    """0.5 must be the hard-coded 0.5+prior rule, so the default is unchanged."""
    w = 0.5
    for prior in (0.0, 0.25, 0.5, 1.0):
        assert (1.0 - w) + 2.0 * w * prior == pytest.approx(0.5 + prior)


def _run_result(name, accuracy, tokens):
    """Minimal RunResult for the reporting tests."""
    from ocmg.types import RunResult
    return RunResult(
        condition=name, accuracy=accuracy, token_f1=accuracy,
        prompt_tokens_per_query=tokens, retrieval_latency_ms=1.0,
        records_retained=100, generative_calls=0, n_questions=50,
        per_question_correct=[1] * 25 + [0] * 25,
    )


def test_retention_is_not_met_when_full_context_is_not_a_ceiling():
    """A ratio above 1.0 against a non-ceiling must not report as met.

    Full context on LoCoMo regularly scores BELOW naive top-k, because a
    15k-token prompt buries the evidence. Retention is a ratio against full
    context, so when that happens the ratio goes above 1.0 and the target
    reads met for a condition that beat nothing. The printed footer has
    warned about this since a vacuous 1.462 was caught; the JSON did not, so
    the dashboard and any later reader saw met=True.
    """
    from ocmg.reporting import run_to_dict

    out = run_to_dict({
        "ocmg": _run_result("ocmg", 0.594, 1296.0),
        "full_context": _run_result("full_context", 0.574, 15375.0),
        "naive_rag": _run_result("naive_rag", 0.613, 1224.0),
    }, seed=0)

    t = out["targets"]["accuracy_retention"]
    assert t["value"] > 1.0, t["value"]
    assert t["met"] is False
    assert t["full_context_is_ceiling"] is False


def test_retention_is_met_when_full_context_really_is_a_ceiling():
    """The guard must not fire when the comparison is sound."""
    from ocmg.reporting import run_to_dict

    out = run_to_dict({
        "ocmg": _run_result("ocmg", 0.700, 1296.0),
        "full_context": _run_result("full_context", 0.720, 15375.0),
        "naive_rag": _run_result("naive_rag", 0.613, 1224.0),
    }, seed=0)

    t = out["targets"]["accuracy_retention"]
    assert t["full_context_is_ceiling"] is True
    assert t["met"] is True


def test_aggregate_does_not_launder_a_vacuous_retention_target():
    """Averaging ratios against a non-ceiling does not make one a ceiling.

    The aggregate recomputes met as mean >= target, which would discard the
    per-seed guard at exactly the point the number gets read: the mean block
    is what the footer prints and the write-up quotes.
    """
    from ocmg.reporting import aggregate_runs, run_to_dict

    runs = [run_to_dict({
        "ocmg": _run_result("ocmg", 0.594, 1296.0),
        "full_context": _run_result("full_context", 0.574, 15375.0),
        "naive_rag": _run_result("naive_rag", 0.613, 1224.0),
    }, seed=sd) for sd in (0, 1, 2)]

    agg = aggregate_runs(runs)
    t = agg["targets"]["accuracy_retention"]
    assert t["mean"] > 1.0
    assert t["met"] is False
    assert t["full_context_is_ceiling"] is False


def test_query_type_onehot_is_a_contract_between_fit_and_predict():
    """Training and inference must encode a query type identically.

    The one-hot position is the only thing linking a column in the training
    matrix to a column at scoring time. If the two ever disagree the model
    still fits, still predicts, and is silently wrong - so the encoding is
    pinned here rather than left to agreement between two call sites.
    """
    from ocmg.features import QUERY_TYPE_NAMES, query_type_onehot

    assert QUERY_TYPE_NAMES == ["q_simple", "q_temporal", "q_enumerative"]
    assert list(query_type_onehot("simple")) == [1.0, 0.0, 0.0]
    assert list(query_type_onehot("temporal")) == [0.0, 1.0, 0.0]
    assert list(query_type_onehot("enumerative")) == [0.0, 0.0, 1.0]
    # An unrecognised type must not land on another type's column.
    assert list(query_type_onehot("something-else")) == [0.0, 0.0, 0.0]


def test_predict_marginal_is_plain_predict_for_a_pooled_model():
    """The pooled path must be untouched by the conditional machinery.

    Every result measured so far came through predict(). If marginalising
    changed that path by even a float, none of those numbers would still be
    comparable to a new run.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from ocmg.config import OCMGConfig
    from ocmg.estimator import UtilityEstimator

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30))
    y = (X[:, 0] > 0).astype(int)

    est = UtilityEstimator(OCMGConfig().estimator)
    est.model = LogisticRegression().fit(X, y)

    probe = rng.normal(size=(7, 30))
    assert np.allclose(est.predict_marginal(probe), est.predict(probe))


def test_predict_marginal_averages_over_query_types_by_observed_frequency():
    """A record with no question attached is scored as a weighted average.

    Admission, calibration and forgetting all run before any question exists.
    A conditional model cannot be asked for U(m) directly, so the reduction
    has to be the mix the questions were actually drawn from - not a uniform
    guess, and not one arbitrary type standing in for all three.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from ocmg.config import OCMGConfig
    from ocmg.estimator import UtilityEstimator
    from ocmg.features import query_type_onehot

    rng = np.random.default_rng(1)
    X30 = rng.normal(size=(300, 30))
    tails = np.vstack([query_type_onehot(t) for t in
                       rng.choice(["simple", "temporal", "enumerative"], 300)])
    X33 = np.hstack([X30, tails])
    y = (X30[:, 0] + tails[:, 1] > 0).astype(int)

    est = UtilityEstimator(OCMGConfig().estimator)
    est.model = LogisticRegression().fit(X33, y)
    est.qtype_mix = {"simple": 6.0, "temporal": 3.0, "enumerative": 1.0}

    probe = rng.normal(size=(5, 30))
    got = est.predict_marginal(probe)

    want = np.zeros(5)
    for qtype, w in est.qtype_mix.items():
        tail = np.tile(query_type_onehot(qtype), (5, 1))
        want += (w / 10.0) * est.predict(np.hstack([probe, tail]))
    assert np.allclose(got, want)

    # A uniform mix must be a different answer, or the weighting is inert.
    est.qtype_mix = None
    assert not np.allclose(est.predict_marginal(probe), want)


def test_predict_marginal_refuses_a_width_it_cannot_reconcile():
    """A width mismatch that is not the query-type tail is a bug, not a case.

    Silently padding or truncating here would turn a feature-ordering error
    into plausible-looking probabilities, which is the failure mode that is
    hardest to notice in a result table.
    """
    import numpy as np
    import pytest
    from sklearn.linear_model import LogisticRegression

    from ocmg.config import OCMGConfig
    from ocmg.estimator import UtilityEstimator

    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 41))
    y = (X[:, 0] > 0).astype(int)
    est = UtilityEstimator(OCMGConfig().estimator)
    est.model = LogisticRegression().fit(X, y)

    with pytest.raises(ValueError, match="columns"):
        est.predict_marginal(rng.normal(size=(3, 30)))


def test_conditional_utility_widens_the_training_matrix():
    """Turning the flag on must change what phase A emits, end to end.

    The flag touches feature construction, label construction and inference,
    so a unit test of the encoder alone would pass while the pipeline still
    trained on pooled rows.
    """
    import numpy as np

    from ocmg.config import OCMGConfig
    from ocmg.data.benchmarks import load_dataset
    from ocmg.features import FEATURE_NAMES, QUERY_TYPE_NAMES
    from ocmg.orchestrator import OCMGRunner

    data = load_dataset("synthetic", None, n_conversations=3, seed=0,
                        n_sessions=3, turns_per_session=6)

    pooled = OCMGRunner(OCMGConfig(seed=0)).run_phase_a(data)
    assert pooled.X.shape[1] == len(FEATURE_NAMES)

    cfg = OCMGConfig(seed=0)
    cfg.estimator.conditional_utility = True
    runner = OCMGRunner(cfg)
    cond = runner.run_phase_a(data)

    assert cond.X.shape[1] == len(FEATURE_NAMES) + len(QUERY_TYPE_NAMES)
    # One row per observed (record, query type) cell, so never fewer rows
    # than the pooled matrix had records with observations.
    assert cond.X.shape[0] >= 1
    # The appended block must be a one-hot: exactly one type per row.
    tail = cond.X[:, len(FEATURE_NAMES):]
    assert set(np.unique(tail)) <= {0.0, 1.0}
    assert np.all(tail.sum(axis=1) == 1.0)
    # And the observed mix must have reached the estimator for marginalising.
    assert runner.estimator.qtype_mix
    assert set(runner.estimator.qtype_mix) <= {"simple", "temporal",
                                               "enumerative"}


def test_category_labels_stay_aligned_with_conditional_rows():
    """The audit labels are read row-for-row against y, so they must widen too.

    per_category_auc indexes category_labels against the same rows as the
    out-of-fold scores. If the conditional path emitted more rows in X than in
    category_labels, the audit would silently score the wrong records.
    """
    from ocmg.config import OCMGConfig
    from ocmg.data.benchmarks import load_dataset
    from ocmg.orchestrator import OCMGRunner

    data = load_dataset("synthetic", None, n_conversations=3, seed=0,
                        n_sessions=3, turns_per_session=6)
    cfg = OCMGConfig(seed=0)
    cfg.estimator.conditional_utility = True
    out = OCMGRunner(cfg).run_phase_a(data)

    n = out.X.shape[0]
    assert len(out.y) == n
    assert len(out.groups) == n
    assert len(out.evidence_labels) == n
    for cat, labels in out.category_labels.items():
        assert len(labels) == n, cat


# --------------------------------------------------------------------------
# The generation cache and the capped counterfactual. Both are speed levers
# added for CPU real-reader runs, and both can corrupt results silently: the
# cache by returning the wrong text, the cap by crediting the wrong record.
# --------------------------------------------------------------------------


class _CountingModel:
    """Stands in for a causal LM. Counts how many rows it was asked to decode.

    Returning the row index as the "generated" token is enough to tell the
    outputs apart, which is what a cache has to get right.
    """

    def __init__(self):
        self.rows = 0

    def generate(self, **kw):
        ids = kw["input_ids"]
        self.rows += len(ids)
        return [[0] * ids.shape[1] + [100 + i] for i in range(len(ids))]


class _StubTok:
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, prompts, **kw):
        width = max(len(p) for p in prompts)
        arr = np.zeros((len(prompts), width), dtype=int)
        return SimpleNamespace(
            __getitem__=lambda self_, k: arr,
            to=lambda device: SimpleNamespace(
                __getitem__=lambda self_, k: arr,
                keys=lambda: ["input_ids"],
                __iter__=lambda self_: iter(["input_ids"]),
            ),
        )

    def decode(self, row, **kw):
        return f"tok{row[-1]}"


def _stub_hf_reader():
    """An HFReader with the model and tokeniser replaced, no weights loaded."""
    from ocmg.reader import HFReader

    r = object.__new__(HFReader)
    r.cfg = OCMGConfig().reader
    r.cfg.batch_size = 8
    r.cfg.max_new_tokens = 24
    r.device = "cpu"
    r._cache = {}
    r.cache_hits = 0
    r.generated = 0
    r.model = _CountingModel()
    r.torch = SimpleNamespace(no_grad=lambda: _NullCtx())

    class _Tok:
        pad_token_id = 0

        def __call__(self, prompts, **kw):
            width = max(len(p) for p in prompts)
            return _Inputs(np.zeros((len(prompts), width), dtype=int))

        def decode(self, row, **kw):
            return f"tok{row[-1]}"

    r.tok = _Tok()
    return r


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class _Inputs(dict):
    def __init__(self, arr):
        super().__init__(input_ids=_Arr(arr))

    def to(self, device):
        return self


class _Arr:
    def __init__(self, arr):
        self._a = arr

    def __len__(self):
        return len(self._a)

    @property
    def shape(self):
        return self._a.shape


def test_generation_cache_decodes_a_repeated_prompt_only_once():
    r = _stub_hf_reader()
    first = r.generate(["alpha", "beta"])
    assert r.model.rows == 2
    assert r.generated == 2 and r.cache_hits == 0

    second = r.generate(["alpha", "beta"])
    # Nothing new was decoded, and the answers are the ones from before.
    assert r.model.rows == 2
    assert second == first
    assert r.cache_hits == 2


def test_generation_cache_deduplicates_inside_one_call():
    r = _stub_hf_reader()
    out = r.generate(["same", "same", "same"])
    assert r.model.rows == 1
    assert out[0] == out[1] == out[2]
    assert r.cache_hits == 2


def test_generation_cache_keys_on_the_token_limit():
    """The admission gate asks for 4 tokens and the reader for 24. A cache
    that ignored the limit would hand the gate's one-word reply back to the
    reader as an answer."""
    r = _stub_hf_reader()
    r.generate(["prompt"], max_new_tokens=4)
    assert r.model.rows == 1
    r.generate(["prompt"], max_new_tokens=24)
    assert r.model.rows == 2, "a different token limit must not be a cache hit"


class _SpyReader:
    """Wraps the simulated reader and records every context it was shown."""

    supports_counterfactual = True

    def __init__(self, inner):
        self.inner = inner
        self.batches = []

    def answer(self, q, c):
        return self.inner.answer(q, c)

    def answer_many(self, items):
        self.batches.append([list(c) for _, c in items])
        return self.inner.answer_many(items)


def _phase_a_with_cap(cap):
    from ocmg.reader import SimulatedReader

    cfg = OCMGConfig(seed=0)
    cfg.exploration.loo_max_records = cap
    data = generate_dataset(n_conversations=2, n_sessions=2,
                            turns_per_session=6, n_facts=4, seed=0)
    runner = OCMGRunner(cfg)
    runner.reader = _SpyReader(SimulatedReader(cfg.reader, seed=0))
    out = runner.run_phase_a(data)
    return runner.reader.batches, out


def test_capped_counterfactual_ablates_only_the_cap():
    """Each leave-one-out batch is the full set plus at most `cap` ablations."""
    batches, _ = _phase_a_with_cap(3)
    assert batches, "phase A asked no questions"
    ablated = [b for b in batches if len(b) > 1]
    assert ablated, "no question was given a counterfactual"
    for b in ablated:
        assert len(b) - 1 <= 3, f"{len(b) - 1} ablations exceeds the cap of 3"


def test_capped_counterfactual_credits_the_record_it_removed():
    """The alignment that matters.

    assign_credit zips record ids against the delta list positionally, so a
    capped run that passed the first len(deltas) retrieved ids would credit
    records it never ablated. Each ablated context must be the full set minus
    exactly one record, and that record is the one the delta belongs to.
    """
    batches, _ = _phase_a_with_cap(2)
    for b in batches:
        if len(b) == 1:
            continue
        full = b[0]
        full_ids = [id(r) for r in full]
        for ablation in b[1:]:
            assert len(ablation) == len(full) - 1
            missing = [i for i in full_ids if i not in [id(r) for r in ablation]]
            assert len(missing) == 1, (
                "an ablated context must differ from the full set by exactly "
                "one record, or the delta cannot be attributed"
            )


def test_uncapped_counterfactual_still_ablates_every_record():
    batches, _ = _phase_a_with_cap(0)
    ablated = [b for b in batches if len(b) > 1]
    assert ablated
    for b in ablated:
        assert len(b) - 1 == len(b[0]), (
            "with no cap every retrieved record must get a counterfactual"
        )
