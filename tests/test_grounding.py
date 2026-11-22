"""Grounding checks against gold evidence ids and against the prompt."""

from datetime import datetime

import numpy as np

from ocmg.config import OCMGConfig
from ocmg.grounding import GroundingTracker, answer_support, retrieval_scores
from ocmg.orchestrator import OCMGRunner
from ocmg.reader import Judge
from ocmg.types import Question, Record

TS = datetime(2023, 5, 8, 13, 56).timestamp()


def _rec(rid, text, sources):
    return Record(
        record_id=rid, conversation_id="c", record_type="episodic",
        summary=text, text=text, embedding=np.zeros(4, dtype=np.float32),
        entities=[], timestamp=TS, session_id=1, start_turn=0, end_turn=0,
        speaker="Caroline", has_temporal_anchor=False, source_ids=sources,
    )


def _q(evidence, category="single_hop", answer="pottery"):
    # Mirrors the loader: adversarial questions keep an evidence id but have
    # no supporting ids.
    return Question(conversation_id="c", question_id="c:q", text="What?",
                    answer=answer, category=category, evidence_ids=evidence,
                    supporting_fact_ids=[] if category == "adversarial"
                    else list(evidence))


def test_recall_and_precision_against_gold_evidence():
    ctx = [_rec(1, "a", ["D1:1", "D1:2"]), _rec(2, "b", ["D2:1"]),
           _rec(3, "c", ["D3:1"]), _rec(4, "d", ["D4:1"])]
    s = retrieval_scores(_q(["D1:2", "D3:1", "D9:9"]), ctx)
    assert s["evidence_recall"] == 2 / 3      # D9:9 never retrieved
    assert s["context_precision"] == 2 / 4    # records 1 and 3 carry evidence
    assert s["evidence_hit"] == 1.0


def test_adversarial_has_no_retrieval_score():
    # Even with the trap's evidence id retrieved, there is nothing to recall.
    assert retrieval_scores(_q(["D1:1"], "adversarial"),
                            [_rec(1, "a", ["D1:1"])]) is None


def test_answer_support_is_lexical_and_ignores_function_words():
    ctx = [_rec(1, "Melanie signed up for a pottery class last week.", ["D1:1"])]
    assert answer_support("pottery class", ctx) == 1.0
    # Content words: took, pottery, course ("she", "a" dropped); one present.
    assert answer_support("She took a pottery course", ctx) == 1 / 3
    assert answer_support("Not mentioned in the conversation", ctx) is None


def test_session_date_counts_as_context():
    ctx = [_rec(1, "I went there yesterday.", ["D1:1"])]
    assert answer_support("8 May 2023", ctx) == 1.0


def test_tracker_quadrants_and_hallucination():
    t = GroundingTracker(0.5)
    ctx = [_rec(1, "pottery class", ["D1:1"])]
    t.add(_q(["D1:1"]), ctx, "pottery", 1)                  # correct, grounded
    t.add(_q(["D1:1"]), ctx, "a painting", 0)                # wrong, ungrounded
    t.add(_q([], "adversarial"), ctx, "Not mentioned", 1)    # abstained
    s = t.summary()
    assert s["quadrants"] == {"correct_grounded": 1, "correct_ungrounded": 0,
                              "wrong_grounded": 0, "wrong_ungrounded": 1}
    assert s["hallucination_rate"] == 1 / 3
    assert s["abstention_rate"] == 1 / 3
    assert s["n_answerable"] == 2 and s["evidence_recall"] == 1.0


def test_judge_threshold_comes_from_config():
    cfg = OCMGConfig()
    cfg.embedder = "hashing"
    cfg.reader.kind = "simulated"
    assert OCMGRunner(cfg).judge.kind == "exact"
    # F1 of "May 2023" against "7 May 2023" is 0.8.
    assert Judge("f1", f1_threshold=0.6).score("May 2023", "7 May 2023")[0] == 1
    assert Judge("f1", f1_threshold=0.9).score("May 2023", "7 May 2023")[0] == 0
