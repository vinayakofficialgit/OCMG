"""Tests for the LoCoMo loader.

The unit tests run without the dataset. The integration tests run only when
data/locomo10.json is present.
"""

import os
from datetime import datetime

import pytest

from ocmg.data.benchmarks import (
    ABSTAIN_ANSWER,
    LOCOMO_CATEGORIES,
    load_locomo,
    normalise_evidence,
    parse_locomo_datetime,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "locomo10.json")
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="data/locomo10.json not present")


def test_date_string_parses_to_the_right_day():
    ts = parse_locomo_datetime("1:56 pm on 8 May, 2023")
    dt = datetime.fromtimestamp(ts)
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2023, 5, 8, 13, 56)


def test_malformed_evidence_ids_are_normalised():
    assert normalise_evidence(["D8:6; D9:17"]) == ["D8:6", "D9:17"]
    assert normalise_evidence(["D:11:26"]) == ["D11:26"]
    assert normalise_evidence(["D30:05"]) == ["D30:5"]
    assert normalise_evidence(["D"]) == []
    assert normalise_evidence(["D1:3", "D1:3"]) == ["D1:3"]


def test_category_map_covers_all_five_codes():
    assert set(LOCOMO_CATEGORIES) == {1, 2, 3, 4, 5}
    assert LOCOMO_CATEGORIES[5] == "adversarial"


@needs_data
def test_release_totals():
    data, report = load_locomo(DATA, return_report=True)
    assert report["conversations"] == 10
    assert report["turns"] == 5882
    assert report["questions"] == 1986


@needs_data
def test_adversarial_reference_is_abstention_not_trap_answer():
    data = load_locomo(DATA)
    adv = [q for _, _, qs in data for q in qs if q.category == "adversarial"]
    assert adv
    assert all(q.answer == ABSTAIN_ANSWER for q in adv)
    assert all(q.supporting_fact_ids == [] for q in adv)
    assert any(q.adversarial_answer for q in adv)


@needs_data
def test_every_kept_evidence_id_points_to_a_real_turn():
    for _, turns, questions in load_locomo(DATA):
        ids = {t.dia_id for t in turns}
        for q in questions:
            assert set(q.evidence_ids) <= ids


@needs_data
def test_sessions_are_in_chronological_order():
    for _, turns, _ in load_locomo(DATA):
        stamps = [t.timestamp for t in turns]
        assert stamps == sorted(stamps)
