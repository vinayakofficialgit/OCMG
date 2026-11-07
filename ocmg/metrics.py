"""Evaluation metrics (Section 8.4).

Accuracy is never reported without cost. The two derived ratios that express
the operating targets — accuracy retention against full context, token
reduction against fixed top-k — are computed as a pair, because either is
trivially satisfiable at the other's expense.

The frontier utility is the one that matters most for interpreting results. A
single operating point cannot establish that one governance policy beats
another, because the two policies spend different numbers of tokens. Sweeping
the budget and comparing accuracy at matched cost is the honest comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def accuracy(correct: Sequence[int]) -> float:
    return float(np.mean(correct)) if len(correct) else 0.0


def paired_bootstrap(
    a: Sequence[int],
    b: Sequence[int],
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap over questions.

    Paired rather than independent because both conditions answer the same
    questions; ignoring the pairing inflates the variance and hides real
    differences.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

    rng = np.random.default_rng(seed)
    n = len(a)
    observed = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    # Two-sided p: proportion of resamples on the opposite side of zero.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": observed,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float(min(1.0, p)),
    }


def per_category_auc(
    category_labels: Dict[str, "np.ndarray"],
    oof_scores: "np.ndarray",
    oof_mask: "np.ndarray",
    min_rows: int = 40,
) -> Dict[str, Dict[str, float]]:
    """Score one fitted estimator against each question category's outcomes.

    The first marker's objection is that a record's utility is not fixed: it
    depends on the query asked. If that holds, a single estimator trained on
    pooled outcomes will rank well for some categories and poorly for others,
    and the spread across categories is the size of the problem.

    Rows labelled -1 had no outcome for that category and are dropped rather
    than counted as negatives - absence of evidence is not evidence of
    uselessness, the same distinction the forgetting signals turn on. A
    category with too few usable rows, or with only one class, reports no AUC
    rather than a number computed on nothing.
    """
    out: Dict[str, Dict[str, float]] = {}
    for cat, labels in sorted(category_labels.items()):
        y = np.asarray(labels)[oof_mask]
        keep = y >= 0
        n = int(keep.sum())
        if n < min_rows or len(set(y[keep].tolist())) < 2:
            out[cat] = {"auc": float("nan"), "n": n,
                        "positive_rate": float(np.mean(y[keep])) if n else 0.0}
            continue
        try:
            auc = float(roc_auc_score(y[keep], oof_scores[keep]))
        except ValueError:
            auc = float("nan")
        out[cat] = {"auc": auc, "n": n,
                    "positive_rate": float(np.mean(y[keep]))}
    return out


def retention_precision(
    retained_utilities: Sequence[float],
    evicted_utilities: Sequence[float],
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Does the governance mechanism keep the records that were worth keeping?

    Reported as the gap in mean empirical utility between retained and evicted
    records, with a bootstrap interval. A gap whose interval straddles zero
    means eviction is no better than chance, which would be a direct refutation
    of the mechanism.
    """
    r = np.asarray(retained_utilities, dtype=float)
    e = np.asarray(evicted_utilities, dtype=float)
    if len(r) == 0 or len(e) == 0:
        return {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "auc": 0.5}

    rng = np.random.default_rng(seed)
    gaps = [
        rng.choice(r, len(r), replace=True).mean()
        - rng.choice(e, len(e), replace=True).mean()
        for _ in range(n_boot)
    ]
    labels = np.concatenate([np.ones(len(r)), np.zeros(len(e))])
    scores = np.concatenate([r, e])
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.5
    return {
        "gap": float(r.mean() - e.mean()),
        "ci_low": float(np.percentile(gaps, 2.5)),
        "ci_high": float(np.percentile(gaps, 97.5)),
        "auc": auc,
    }


def label_validation(
    outcome_labels: Sequence[int],
    evidence_labels: Sequence[int],
    oof_scores: Optional[Sequence[float]] = None,
    oof_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Score outcome-derived utility labels against annotated evidence.

    This is the check that answers the central objection to the method: a
    correct answer does not by itself prove that every record in the prompt
    contributed to it, so a label built from outcomes might be measuring
    co-occurrence rather than usefulness. The benchmark annotations say which
    turns genuinely support each reference answer, which makes the question
    empirical. The annotations are held out entirely — they are never a
    feature and never a training target — and are used only here.

    Precision is P(record is annotated evidence | the outcome labelled it
    useful) and recall is the converse. Kappa corrects the agreement for the
    rate expected by chance, which matters because both label sets are
    imbalanced and raw agreement would look high on the negatives alone.

    Read precision and recall asymmetrically. Low recall is expected and not a
    fault: a record can carry annotated evidence and still never be the record
    that changed an answer, because another retrieved record carried the same
    fact. Low precision is the finding that would refute the method, because it
    would mean outcome credit is landing on records that demonstrably do not
    support the answer.

    `oof_scores` with `oof_mask` additionally scores the fitted estimator
    directly against evidence, which is the stronger claim: not that the labels
    agree with evidence, but that a model trained on them ranks genuinely
    supporting records above the rest.
    """
    y = np.asarray(outcome_labels, dtype=int)
    e = np.asarray(evidence_labels, dtype=int)
    if len(y) == 0 or len(y) != len(e):
        return {"n": 0}

    tp = int(np.sum((y == 1) & (e == 1)))
    fp = int(np.sum((y == 1) & (e == 0)))
    fn = int(np.sum((y == 0) & (e == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    out: Dict[str, Any] = {
        "n": int(len(y)),
        "outcome_positive_rate": float(y.mean()),
        "evidence_positive_rate": float(e.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "kappa": float(cohen_kappa_score(e, y)) if len(set(y.tolist())) > 1 else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

    if oof_scores is not None:
        scores = np.asarray(oof_scores, dtype=float)
        ev = e[np.asarray(oof_mask, dtype=bool)] if oof_mask is not None else e
        if len(scores) == len(ev) and len(set(ev.tolist())) > 1:
            out["estimator_auc_vs_evidence"] = float(roc_auc_score(ev, scores))
    return out


def evidence_retention(store, conversations) -> Dict[str, float]:
    """How much annotated evidence survived governance.

    Accuracy alone cannot distinguish the two ways a governed store loses a
    question: the evidence was discarded at write time, or it was kept and
    retrieval failed to surface it. Only the first is irreversible, and only
    the first puts a ceiling on what any retrieval budget can reach. This
    measures the store's contents directly, so the two causes separate.

    Note what this metric does *not* depend on: the reader. It asks whether a
    supporting turn is present in the store, which is a property of the
    governance policy alone. Results from it therefore carry over unchanged
    from the simulated reader to a real one, unlike every accuracy number.
    """
    evidence_by_conv: Dict[str, set] = {}
    for cid, _turns, questions in conversations:
        ids: set = set()
        for q in questions:
            ids.update(q.evidence_ids)
            ids.update(q.supporting_fact_ids)
        evidence_by_conv[cid] = ids

    # Reachability spans both tiers: a demoted record is still retrievable, so
    # it does not reduce the ceiling. Only a discarded record does. Hot-tier
    # counts are tracked separately because they are what sets token cost.
    present_by_conv: Dict[str, set] = {}
    n_records = 0
    n_evidence_records = 0
    n_cold = 0
    for rec in store.records.values():
        sources = set(rec.source_ids or [])
        if rec.fact_id:
            sources.add(rec.fact_id)
        is_evidence = bool(sources & evidence_by_conv.get(rec.conversation_id, set()))
        if rec.active:
            n_records += 1
            if is_evidence:
                n_evidence_records += 1
        elif rec.demoted_at_write:
            n_cold += 1
        else:
            continue  # evicted: retained for audit, but out of service
        present_by_conv.setdefault(rec.conversation_id, set()).update(sources)

    reachable = 0
    total = 0
    for cid, _turns, questions in conversations:
        present = present_by_conv.get(cid, set())
        for q in questions:
            needed = set(q.evidence_ids) | set(q.supporting_fact_ids)
            if not needed:
                continue  # unanswerable by construction; abstention is correct
            total += 1
            if needed & present:
                reachable += 1

    return {
        "records": n_records,
        "cold_records": n_cold,
        "evidence_records": n_evidence_records,
        "answerable_reachable": reachable,
        "answerable_total": total,
        # The hard ceiling governance imposes: no retrieval budget can exceed
        # it, because the supporting turn is not in the store in any tier.
        "evidence_ceiling": (reachable / total) if total else 0.0,
    }


def accuracy_retention_ratio(governed_acc: float, full_context_acc: float) -> float:
    if full_context_acc <= 0:
        return 0.0
    return governed_acc / full_context_acc


def token_reduction_ratio(governed_tokens: float, baseline_tokens: float) -> float:
    if baseline_tokens <= 0:
        return 0.0
    return 1.0 - (governed_tokens / baseline_tokens)


def accuracy_per_1k_tokens(acc: float, tokens_per_query: float) -> float:
    if tokens_per_query <= 0:
        return 0.0
    return acc / (tokens_per_query / 1000.0)


def pareto_frontier(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Upper-left frontier of (cost, accuracy) points: cheaper and better."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: List[Tuple[float, float]] = []
    best = -np.inf
    for cost, acc in pts:
        if acc > best:
            frontier.append((cost, acc))
            best = acc
    return frontier


def interpolate_at_cost(
    curve: Sequence[Tuple[float, float]], cost: float
) -> Optional[float]:
    """Accuracy of a condition at a given token budget, by linear interpolation.

    This is what makes an iso-cost comparison possible when two conditions do
    not happen to land on the same budget.
    """
    pts = sorted(curve, key=lambda p: p[0])
    if not pts:
        return None
    if cost <= pts[0][0]:
        return pts[0][1]
    if cost >= pts[-1][0]:
        return pts[-1][1]
    for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
        if c0 <= cost <= c1:
            if c1 - c0 < 1e-9:
                return a1
            t = (cost - c0) / (c1 - c0)
            return a0 + t * (a1 - a0)
    return pts[-1][1]


def iso_cost_comparison(
    curve_a: Sequence[Tuple[float, float]],
    curve_b: Sequence[Tuple[float, float]],
    n_points: int = 12,
) -> Dict[str, float]:
    """Mean accuracy gap between two conditions over the overlapping budget range."""
    if not curve_a or not curve_b:
        return {"mean_gap": 0.0, "n_points": 0}
    lo = max(min(c for c, _ in curve_a), min(c for c, _ in curve_b))
    hi = min(max(c for c, _ in curve_a), max(c for c, _ in curve_b))
    if hi <= lo:
        return {"mean_gap": 0.0, "n_points": 0}
    grid = np.linspace(lo, hi, n_points)
    gaps = []
    for c in grid:
        a = interpolate_at_cost(curve_a, float(c))
        b = interpolate_at_cost(curve_b, float(c))
        if a is not None and b is not None:
            gaps.append(a - b)
    return {
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "min_cost": float(lo),
        "max_cost": float(hi),
        "n_points": len(gaps),
    }
