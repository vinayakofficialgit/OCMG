"""The official LoCoMo answer scorer, reproduced for comparability.

Why this exists. The pipeline's own judge (reader.Judge) turns each answer
into a binary correct/incorrect with token F1 >= 0.6. That is what accuracy,
the leave-one-out labels and every significance test are built on, and it is
not what LoCoMo's authors report. Published LoCoMo numbers (the original
paper and the memory systems that cite it) are the *mean F1* produced by
`task_eval/evaluation.py::eval_question_answering` in
https://github.com/snap-research/locomo. A number from one protocol cannot be
set beside a number from the other, so both are reported: the binary judge
drives the method, this one makes the table comparable with prior work.

What the official protocol does, per category (checked against the source):

    single_hop, temporal   F1 on normalised, Porter-stemmed tokens
    open_domain            same, against the part of the reference before ';'
    multi_hop              split both strings on ',', best F1 per reference
                           part, then the mean over reference parts
    adversarial            1 if the answer contains 'not mentioned' or
                           'no information available', else 0

Normalisation: drop commas, lowercase, strip punctuation, remove the words
a / an / the / and, collapse whitespace. This differs from reader.token_f1 in
three ways that move numbers: stemming ('painting' matches 'paint'), the
article list, and the multi-hop split.

Two departures from the official script, both forced by the release and both
deliberate:

* The official code reads qa['answer'] for adversarial questions, but 444 of
  the 446 store only 'adversarial_answer'; run unmodified it fails on this
  release. Here the reference for adversarial questions is the abstention the
  loader already stores, and scoring follows the official substring rule.
* The official prompt turns adversarial questions into a two-option choice
  using the gold category. That uses an annotation a deployed system would not
  have, so it is not reproduced; only the scoring rule is.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

try:  # the official scorer stems with NLTK's Porter stemmer
    from nltk.stem import PorterStemmer
except ImportError as exc:  # pragma: no cover - surfaced at first use
    PorterStemmer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCT = set(string.punctuation)
_STEMMER = None

# Substrings the official scorer accepts as a correct abstention. Narrower
# than reader.is_abstention, on purpose: this is their rule, not ours.
OFFICIAL_ABSTENTIONS = ("no information available", "not mentioned")


def _stem(word: str) -> str:
    global _STEMMER
    if _STEMMER is None:
        if PorterStemmer is None:
            raise ImportError(
                "the official LoCoMo scorer needs nltk (pip install nltk)"
            ) from _IMPORT_ERROR
        _STEMMER = PorterStemmer()
    return _STEMMER.stem(word)


def normalize_answer(s: str) -> str:
    """Official normalisation, in the official order."""
    s = str(s).replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token F1 after normalisation and stemming (single reference)."""
    p = [_stem(w) for w in normalize_answer(prediction).split()]
    g = [_stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """Multi-hop: each reference part takes its best-matching prediction part."""
    preds = [p.strip() for p in str(prediction).split(",")]
    golds = [g.strip() for g in str(ground_truth).split(",")]
    return float(np.mean([max(f1_score(p, g) for p in preds) for g in golds]))


def locomo_f1(prediction: str, reference: str, category: str) -> float:
    """Score one answer exactly as LoCoMo's evaluation script would."""
    prediction = str(prediction)
    if category == "adversarial":
        low = prediction.lower()
        return float(any(phrase in low for phrase in OFFICIAL_ABSTENTIONS))
    reference = str(reference)
    if category == "open_domain":
        reference = reference.split(";")[0].strip()
    if category == "multi_hop":
        return multi_answer_f1(prediction, reference)
    # single_hop, temporal, and any non-LoCoMo category (synthetic data).
    return f1_score(prediction, reference)


def mean_by_category(
    scores: Sequence[float], categories: Sequence[str]
) -> Dict[str, float]:
    """Per-category mean, the breakdown the LoCoMo paper tabulates."""
    out: Dict[str, List[float]] = {}
    for s, c in zip(scores, categories):
        out.setdefault(c, []).append(float(s))
    return {c: float(np.mean(v)) for c, v in out.items()}
