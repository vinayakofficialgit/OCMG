"""Turn experiment results into plain, JSON-serialisable records.

Every number the report scripts plot or tabulate comes from here, so the
console summary, the JSON file and the thesis tables can never disagree.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .metrics import (
    accuracy_per_1k_tokens,
    accuracy_retention_ratio,
    paired_bootstrap,
    token_reduction_ratio,
)
from .types import RunResult

TARGET_ACCURACY_RETENTION = 0.95
TARGET_TOKEN_REDUCTION = 0.40
TARGET_ESTIMATOR_AUC = 0.70


def _plain(value: Any) -> Any:
    """Convert numpy scalars and nested containers to JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def condition_to_dict(r: RunResult) -> Dict[str, Any]:
    return {
        "accuracy": r.accuracy,
        "token_f1": r.token_f1,
        "tokens_per_query": r.prompt_tokens_per_query,
        "accuracy_per_1k_tokens": accuracy_per_1k_tokens(
            r.accuracy, r.prompt_tokens_per_query
        ),
        "retrieval_latency_ms": r.retrieval_latency_ms,
        "records_retained": r.records_retained,
        "generative_calls": r.generative_calls,
        "n_questions": r.n_questions,
        "per_category": _plain(r.per_category),
        "extra": _plain(r.extra),
    }


def run_to_dict(results: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """One seed's results: conditions, estimator, targets and significance."""
    conditions = {
        k: condition_to_dict(v) for k, v in results.items() if isinstance(v, RunResult)
    }
    out: Dict[str, Any] = {"seed": seed, "conditions": conditions}

    est = results.get("_estimator")
    if est is not None:
        out["estimator"] = {
            "auc_mean": float(est.auc_mean),
            "auc_std": float(est.auc_std),
            "auc_folds": [float(a) for a in est.auc_folds],
            "average_precision": float(est.ap_mean),
            "brier": float(est.brier),
            "n_samples": int(est.n_samples),
            "positive_rate": float(est.positive_rate),
            "feature_importance": _plain(est.feature_importance),
            # What ranking by one raw feature achieves on the same folds. Kept
            # in the results file because it is the figure that answers "is
            # the learned utility just an IDF heuristic".
            "auc_single_feature": _plain(est.auc_single_feature),
        }

    val = results.get("_label_validation")
    if val:
        out["label_validation"] = _plain(val)

    ocmg = results.get("ocmg")
    full = results.get("full_context")
    naive = results.get("naive_rag")
    targets: Dict[str, Any] = {}
    if isinstance(ocmg, RunResult) and isinstance(full, RunResult):
        v = accuracy_retention_ratio(ocmg.accuracy, full.accuracy)
        # Retention is a ratio against full context, which is only meaningful
        # while full context is an upper bound. On this dataset it often is
        # not: a 15k-token prompt buries the evidence and scores below naive
        # top-k, and the ratio then reads above 1.0 and "met" reads True for a
        # condition that beat nothing. The printed footer has carried this
        # warning since the vacuous 1.462 was caught; the JSON did not, so
        # anything reading the file back - the dashboard, a later run, a table
        # in the write-up - saw met=True with no way to know it was vacuous.
        not_a_ceiling = bool(isinstance(naive, RunResult)
                             and full.accuracy < naive.accuracy)
        targets["accuracy_retention"] = {
            "value": v, "target": TARGET_ACCURACY_RETENTION,
            "met": bool(v >= TARGET_ACCURACY_RETENTION
                        and not not_a_ceiling),
            "full_context_is_ceiling": not not_a_ceiling,
            "full_context_accuracy": float(full.accuracy),
            "naive_rag_accuracy": (float(naive.accuracy)
                                   if isinstance(naive, RunResult) else None),
        }
    if isinstance(ocmg, RunResult) and isinstance(naive, RunResult):
        v = token_reduction_ratio(
            ocmg.prompt_tokens_per_query, naive.prompt_tokens_per_query
        )
        targets["token_reduction"] = {
            "value": v, "target": TARGET_TOKEN_REDUCTION,
            "met": bool(v >= TARGET_TOKEN_REDUCTION),
        }
    if "estimator" in out:
        v = out["estimator"]["auc_mean"]
        targets["estimator_auc"] = {
            "value": v, "target": TARGET_ESTIMATOR_AUC,
            "met": bool(v >= TARGET_ESTIMATOR_AUC),
        }
    out["targets"] = targets

    significance = {}
    if isinstance(ocmg, RunResult):
        for other in ("full_context", "naive_rag", "append_all", "decay", "generative_judge"):
            o = results.get(other)
            if isinstance(o, RunResult) and len(o.per_question_correct) == len(
                ocmg.per_question_correct
            ):
                significance[f"ocmg_vs_{other}"] = paired_bootstrap(
                    ocmg.per_question_correct, o.per_question_correct
                )
    out["significance"] = significance
    return _plain(out)


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and standard deviation of the headline numbers across seeds."""
    if not runs:
        return {}
    names = sorted(runs[0]["conditions"].keys())
    agg: Dict[str, Any] = {"n_runs": len(runs), "conditions": {}}
    for name in names:
        rows = [r["conditions"][name] for r in runs if name in r["conditions"]]
        entry = {}
        for metric in ("accuracy", "token_f1", "tokens_per_query",
                       "accuracy_per_1k_tokens", "records_retained"):
            vals = np.array([row[metric] for row in rows], dtype=float)
            entry[metric] = {"mean": float(vals.mean()), "sd": float(vals.std())}
        agg["conditions"][name] = entry
    val_keys = ("precision", "recall", "kappa", "estimator_auc_vs_evidence",
                "evidence_positive_rate", "outcome_positive_rate")
    val_runs = [r["label_validation"] for r in runs if "label_validation" in r]
    if val_runs:
        agg["label_validation"] = {
            k: {
                "mean": float(np.mean([v[k] for v in val_runs if k in v])),
                "sd": float(np.std([v[k] for v in val_runs if k in v])),
            }
            for k in val_keys
            if any(k in v for v in val_runs)
        }
        agg["label_validation"]["n_runs"] = len(val_runs)

    for key in ("accuracy_retention", "token_reduction", "estimator_auc"):
        vals = [r["targets"][key]["value"] for r in runs if key in r.get("targets", {})]
        if vals:
            entry = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                "target": runs[0]["targets"][key]["target"],
                "met": bool(np.mean(vals) >= runs[0]["targets"][key]["target"]),
            }
            # A target that is vacuous on any seed is vacuous for the mean:
            # averaging ratios against a non-ceiling does not make them a
            # ceiling. Without this the per-seed guard is discarded exactly
            # where the number gets read.
            flags = [r["targets"][key].get("full_context_is_ceiling")
                     for r in runs if key in r.get("targets", {})]
            if any(f is False for f in flags):
                entry["met"] = False
                entry["full_context_is_ceiling"] = False
            agg.setdefault("targets", {})[key] = entry
    return agg
