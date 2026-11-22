"""The official LoCoMo scorer, and the reader prompt it depends on.

Expected values were produced by running the official
task_eval/evaluation.py (snap-research/locomo) on the same strings; the
replica agreed with it on all 15,888 pairs of a full cross-check over every
LoCoMo reference answer.
"""

from datetime import datetime

import numpy as np
import pytest

from ocmg.data.benchmarks import ABSTAIN_ANSWER, parse_locomo_datetime
from ocmg.locomo_eval import locomo_f1, mean_by_category, normalize_answer
from ocmg.reader import HFReader, format_session_date, is_abstention
from ocmg.types import Question, Record


def test_normalisation_follows_the_official_order():
    # Commas go first, then case, punctuation, and a/an/the/and.
    assert normalize_answer("The Cat, and a Dog!") == "cat dog"


def test_stemming_matches_inflected_forms():
    # Token F1 without stemming would score this 0; the official scorer 1.
    assert locomo_f1("painting", "paint", "single_hop") == pytest.approx(1.0)


def test_multi_hop_splits_into_sub_answers():
    # Two reference parts, one matched: half credit, not a token-overlap F1.
    assert locomo_f1("pottery", "pottery, camping", "multi_hop") == pytest.approx(0.5)
    assert locomo_f1("camping, pottery", "pottery, camping", "multi_hop") == pytest.approx(1.0)


def test_open_domain_uses_the_part_before_the_semicolon():
    assert locomo_f1("Likely yes", "Likely yes; she enjoys it", "open_domain") == pytest.approx(1.0)


def test_adversarial_follows_the_official_substring_rule():
    assert locomo_f1("Not mentioned in the conversation", ABSTAIN_ANSWER, "adversarial") == 1.0
    assert locomo_f1("No information available.", ABSTAIN_ANSWER, "adversarial") == 1.0
    # Our judge accepts this phrase; the official rule does not. The reader
    # prompt was changed so a real model never produces it.
    assert locomo_f1("unable to determine from memory", ABSTAIN_ANSWER, "adversarial") == 0.0
    assert locomo_f1("self-care is important", ABSTAIN_ANSWER, "adversarial") == 0.0


def test_mean_by_category():
    out = mean_by_category([1.0, 0.0, 0.5], ["temporal", "temporal", "multi_hop"])
    assert out == {"temporal": 0.5, "multi_hop": 0.5}


def test_session_date_is_written_like_locomo_answers():
    ts = parse_locomo_datetime("1:56 pm on 8 May, 2023")
    assert format_session_date(ts) == "8 May 2023"


def _record(text: str, ts: float) -> Record:
    return Record(
        record_id=1, conversation_id="c", record_type="episodic", summary=text,
        text=text, embedding=np.zeros(4, dtype=np.float32), entities=[],
        timestamp=ts, session_id=1, start_turn=0, end_turn=0, speaker="Caroline",
        has_temporal_anchor=True,
    )


def test_reader_prompt_carries_the_session_date():
    # The model is never loaded: _prompt needs only the class.
    reader = HFReader.__new__(HFReader)
    ts = datetime(2023, 5, 8, 13, 56).timestamp()
    q = Question(conversation_id="c", question_id="c:q0",
                 text="When did Caroline go to the LGBTQ support group?",
                 answer="7 May 2023", category="temporal")
    prompt = reader._prompt(q, [_record("I went to a support group yesterday.", ts)])
    assert "(episodic, 8 May 2023) I went to a support group yesterday." in prompt


def test_reader_abstention_reply_passes_both_scorers():
    reply = HFReader.ABSTAIN_REPLY
    assert is_abstention(reply)
    assert locomo_f1(reply, ABSTAIN_ANSWER, "adversarial") == 1.0
