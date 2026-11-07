"""Stage 5: outcome propagation and credit assignment (Section 7.4).

Attributing a correct answer to one record among several retrieved jointly is
under-determined without counterfactual re-execution. Four schemes are
implemented so that the sensitivity of every result to that approximation can
be measured rather than assumed:

    rank_weighted   c += y / log2(rank + 1)      -- the default
    uniform         c += y / |R|                 -- the ablation comparator
    top1            all credit to rank 1         -- a deliberately crude bound
    leave_one_out   exact attribution by re-running without each record

leave_one_out is the ground truth the other three approximate. It costs |R|
extra reader calls per query, so it is applied to a stratified subsample and
used to calibrate the cheap schemes, exactly as risk R2 in the proposal states.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def rank_weight(rank: int) -> float:
    """Logarithmic discount borrowed from ranked-retrieval evaluation."""
    return 1.0 / math.log2(rank + 1)


def assign_credit(
    scheme: str,
    record_ids: Sequence[int],
    ranks: Sequence[int],
    outcome: int,
    loo_deltas: Sequence[float] | None = None,
) -> List[Tuple[int, float, float]]:
    """Return (record_id, credit, weight) triples for one query.

    Weight is accumulated separately from credit so that running utility is a
    weighted mean rather than a raw sum; a record retrieved often at low ranks
    is then not penalised relative to one retrieved rarely at rank 1.
    """
    if not record_ids:
        return []

    if scheme == "uniform":
        w = 1.0 / len(record_ids)
        return [(rid, outcome * w, w) for rid in record_ids]

    if scheme == "top1":
        out = []
        for rid, rank in zip(record_ids, ranks):
            w = 1.0 if rank == 1 else 0.0
            out.append((rid, outcome * w, w))
        return out

    if scheme == "leave_one_out":
        if loo_deltas is None:
            raise ValueError("leave_one_out requires loo_deltas")
        # delta = P(correct | full set) - P(correct | set minus this record).
        # Positive delta means the record carried the answer.
        out = []
        for rid, d in zip(record_ids, loo_deltas):
            out.append((rid, float(max(0.0, d)), 1.0))
        return out

    # default: rank_weighted
    out = []
    for rid, rank in zip(record_ids, ranks):
        w = rank_weight(rank)
        out.append((rid, outcome * w, w))
    return out


def leave_one_out_deltas(
    record_ids: Sequence[int],
    full_outcome_prob: float,
    ablated_probs: Sequence[float],
) -> List[float]:
    """Marginal contribution of each record to the probability of correctness."""
    return [float(full_outcome_prob - p) for p in ablated_probs]


def compare_schemes(
    logs: Sequence[dict], schemes: Sequence[str] = ("rank_weighted", "uniform", "top1")
) -> Dict[str, Dict[int, float]]:
    """Recompute per-record utility under several schemes from the same logs.

    Because the logs record ranks and outcomes, the counterfactual 'what would
    the ledger look like under a different credit rule' is answerable offline,
    with no re-execution. This is what makes the credit ablation cheap.
    """
    out: Dict[str, Dict[int, float]] = {}
    for scheme in schemes:
        credit: Dict[int, float] = {}
        weight: Dict[int, float] = {}
        for log in logs:
            triples = assign_credit(
                scheme, log["record_ids"], log["ranks"], log["correct"]
            )
            for rid, c, w in triples:
                credit[rid] = credit.get(rid, 0.0) + c
                weight[rid] = weight.get(rid, 0.0) + w
        out[scheme] = {
            rid: (credit[rid] / weight[rid]) if weight.get(rid, 0.0) > 0 else 0.0
            for rid in credit
        }
    return out
