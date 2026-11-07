"""Judge behaviour on answerable and unanswerable (adversarial) questions."""

from ocmg.data.benchmarks import ABSTAIN_ANSWER
from ocmg.reader import Judge, is_abstention


def test_abstention_is_correct_for_unanswerable_question():
    ok, _ = Judge("f1").score("unable to determine from memory", ABSTAIN_ANSWER)
    assert ok == 1


def test_taking_the_bait_is_wrong_for_unanswerable_question():
    ok, _ = Judge("f1").score("self-care is important", ABSTAIN_ANSWER)
    assert ok == 0


def test_answerable_question_uses_token_overlap():
    assert Judge("f1").score("adoption agencies", "Adoption agencies")[0] == 1
    assert Judge("f1").score("a painting", "Adoption agencies")[0] == 0


def test_abstention_phrases_are_detected():
    assert is_abstention("I don't know")
    assert is_abstention("That is not mentioned.")
    assert not is_abstention("She went to Paris")
