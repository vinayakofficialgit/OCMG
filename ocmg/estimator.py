"""The utility estimator U(m) (Section 7.3).

Estimates P(record contributes to a correct answer | features available at
write time). Gradient-boosted trees by default, regularised logistic regression
as the simpler comparator; both admit direct inspection of feature importance,
which matters because identifying which record properties predict usefulness is
itself a stated contribution.

Two protocol details carry most of the methodological weight:

Grouped folds. The grouping unit is the conversation, not the record. A
record-level split leaks: records from one conversation share vocabulary,
entities and topic, so a model split at record level can identify the
conversation and shortcut the task. Grouping by conversation removes that path.

Calibration. Raw GBDT scores are not probabilities. Since the governance
thresholds are quantiles over predicted utility, miscalibration would distort
the admitted fraction, so probabilities are calibrated inside each training
fold and thresholds are chosen on out-of-fold predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EstimatorConfig
from .features import FEATURE_NAMES, QUERY_TYPE_NAMES, query_type_onehot


@dataclass
class EstimatorReport:  # noqa: D101
    auc_mean: float
    auc_std: float
    auc_folds: List[float]
    ap_mean: float
    brier: float
    n_samples: int
    positive_rate: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    oof_predictions: Optional[np.ndarray] = None
    oof_labels: Optional[np.ndarray] = None
    # Which rows of the original X received an out-of-fold prediction. Needed
    # to align oof_predictions back to any row-aligned array held outside the
    # estimator, such as the held-out evidence annotation.
    oof_mask: Optional[np.ndarray] = None
    # Single-feature baselines over the identical folds. Permutation importance
    # puts lexical rarity on top, so the obvious objection is that the learned
    # utility is an IDF heuristic with extra steps. This answers it with a
    # number instead of an argument: each entry is the ROC-AUC of ranking by
    # that raw feature alone, no model and no fitting involved.
    auc_single_feature: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"samples={self.n_samples}  positive_rate={self.positive_rate:.3f}",
            f"ROC-AUC = {self.auc_mean:.3f} +/- {self.auc_std:.3f}  "
            f"(folds: {', '.join(f'{a:.3f}' for a in self.auc_folds)})",
            f"average precision = {self.ap_mean:.3f}   Brier = {self.brier:.4f}",
        ]
        if self.feature_importance:
            top = sorted(self.feature_importance.items(), key=lambda x: -abs(x[1]))[:10]
            lines.append("top features: " + ", ".join(f"{k}={v:+.3f}" for k, v in top))
        return "\n".join(lines)


def _rarity_ranker(cfg: EstimatorConfig, feature: str = "rare_term_count"):
    """Rank by one raw feature: the "is it just an IDF heuristic?" ablation.

    The single-feature ROC-AUCs reported beside the model say
    rare_term_count reaches 0.715 against the model's 0.727, so the learned
    part is worth +0.012 AUC. That is a statement about ranking only; running
    the whole pipeline on the heuristic measures the downstream effect instead
    of inferring it.

    A one-dimensional logistic rather than the raw count, because the
    governance thresholds are quantiles of a calibrated probability and
    feeding an unbounded integer through them would move the operating point
    as well as the signal. It is monotone in the feature, so the ranking is
    exactly the heuristic's.

    Built from sklearn parts rather than a custom class because
    CalibratedClassifierCV clones its base estimator, which requires
    get_params/set_params; a ColumnTransformer selects the one column and
    clones correctly.
    """
    j = FEATURE_NAMES.index(feature)
    return Pipeline([
        ("pick", ColumnTransformer([("f", "passthrough", [j])],
                                   remainder="drop")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if cfg.class_weight_balanced else None,
            random_state=cfg.random_state,
        )),
    ])


def _base_model(cfg: EstimatorConfig):
    if cfg.model == "rarity":
        return _rarity_ranker(cfg)
    if cfg.model == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        class_weight="balanced" if cfg.class_weight_balanced else None,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    return HistGradientBoostingClassifier(
        max_iter=cfg.gbdt_max_iter,
        learning_rate=cfg.gbdt_learning_rate,
        max_depth=cfg.gbdt_max_depth,
        class_weight="balanced" if cfg.class_weight_balanced else None,
        random_state=cfg.random_state,
    )


class UtilityEstimator:
    # Raw features scored as standalone rankers alongside the model. Chosen
    # because permutation importance puts lexical rarity at the top, which
    # invites the objection that the learned utility is an IDF heuristic;
    # max_similarity and token_length are included as non-rarity controls.
    SINGLE_FEATURE_BASELINES = (
        "rare_term_count", "mean_idf", "max_idf",
        "max_similarity", "token_length",
    )

    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.model = None
        self.report: Optional[EstimatorReport] = None
        self._fallback_rate: float = 0.5
        # Observed frequency of each derived query type, set by the runner
        # after warm-up and read only by predict_marginal. None means a
        # uniform mix, which is what holds before warm-up has measured it.
        self.qtype_mix: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------- fit

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence[str]
    ) -> EstimatorReport:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups)
        self._fallback_rate = float(y.mean()) if len(y) else 0.5

        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if n_pos < 10 or n_neg < 10:
            # Not enough signal to fit anything trustworthy. Degrade to the
            # base rate rather than pretending to a model.
            self.model = None
            self.report = EstimatorReport(
                auc_mean=0.5, auc_std=0.0, auc_folds=[], ap_mean=self._fallback_rate,
                brier=0.25, n_samples=len(y), positive_rate=self._fallback_rate,
            )
            return self.report

        n_groups = len(set(groups.tolist()))
        n_splits = int(np.clip(min(self.cfg.n_splits, n_groups), 2, 10))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.cfg.random_state
        )

        oof = np.full(len(y), np.nan, dtype=float)
        fold_aucs: List[float] = []
        single_folds: Dict[str, List[float]] = {}
        for tr, va in splitter.split(X, y, groups):
            if len(set(y[tr].tolist())) < 2 or len(set(y[va].tolist())) < 2:
                continue
            m = self._fit_calibrated(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            oof[va] = p
            fold_aucs.append(float(roc_auc_score(y[va], p)))

            # Single-feature baselines on the same folds, so the figures are
            # comparable to auc_mean above rather than to a pooled estimate.
            # Orientation is decided on the training fold: a raw feature has
            # nothing fitted, but choosing the direction after seeing the
            # validation labels is one bit of leakage and would flatter it.
            for name in self.SINGLE_FEATURE_BASELINES:
                j = FEATURE_NAMES.index(name)
                try:
                    a_tr = roc_auc_score(y[tr], X[tr][:, j])
                    a_va = float(roc_auc_score(y[va], X[va][:, j]))
                except ValueError:
                    continue
                single_folds.setdefault(name, []).append(
                    1.0 - a_va if a_tr < 0.5 else a_va
                )

        mask = ~np.isnan(oof)
        auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        auc_std = float(np.std(fold_aucs)) if fold_aucs else 0.0
        ap = float(average_precision_score(y[mask], oof[mask])) if mask.sum() > 2 else 0.0
        brier = float(brier_score_loss(y[mask], oof[mask])) if mask.sum() > 2 else 0.25

        singles = {n: float(np.mean(v)) for n, v in single_folds.items() if v}

        # Refit on everything for deployment.
        self.model = self._fit_calibrated(X, y)

        self.report = EstimatorReport(
            auc_mean=auc_mean,
            auc_std=auc_std,
            auc_folds=fold_aucs,
            ap_mean=ap,
            brier=brier,
            n_samples=len(y),
            positive_rate=self._fallback_rate,
            feature_importance=self._importance(X, y),
            oof_predictions=oof[mask],
            oof_labels=y[mask],
            oof_mask=mask,
            auc_single_feature=singles,
        )
        return self.report

    def _fit_calibrated(self, X: np.ndarray, y: np.ndarray):
        base = _base_model(self.cfg)
        n_min = int(min(y.sum(), (1 - y).sum()))
        if n_min < 6:
            base.fit(X, y)
            return base
        cv = int(np.clip(n_min // 3, 2, 5))
        method = self.cfg.calibration
        if method == "isotonic" and n_min < 50:
            method = "sigmoid"  # isotonic overfits on small folds
        cal = CalibratedClassifierCV(base, method=method, cv=cv)
        cal.fit(X, y)
        return cal

    def _importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Permutation importance on the fitted model.

        Preferred over impurity-based importance because it is model-agnostic
        and does not inflate the score of high-cardinality features.
        """
        try:
            from sklearn.inspection import permutation_importance

            n = min(len(y), 3000)
            idx = np.random.RandomState(0).choice(len(y), n, replace=False)
            r = permutation_importance(
                self.model, X[idx], y[idx], n_repeats=5,
                random_state=self.cfg.random_state, scoring="roc_auc",
            )
            # zip against FEATURE_NAMES alone truncates to 30, silently
            # discarding the query-type columns a conditional model is fitted
            # on - which are the ones worth looking at, because their
            # importance is the direct answer to whether the model uses the
            # query at all.
            names = list(FEATURE_NAMES)
            if X.shape[1] == len(FEATURE_NAMES) + len(QUERY_TYPE_NAMES):
                names += list(QUERY_TYPE_NAMES)
            return {
                name: float(v)
                for name, v in zip(names, r.importances_mean)
            }
        except Exception:
            return {}

    # --------------------------------------------------------------- predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.model is None:
            return np.full(X.shape[0], self._fallback_rate, dtype=float)
        return self.model.predict_proba(X)[:, 1]

    def predict_marginal(self, X: np.ndarray) -> np.ndarray:
        """Score records with no question attached to them.

        Admission, threshold calibration and forgetting all act on a record
        without a query in hand, so a conditional U(m|q) has nothing to
        condition on. This marginalises over the query types, weighted by how
        often each was observed in warm-up (uniform until that is known).

        The width mismatch is detected from the fitted model rather than from
        configuration: an estimator fitted on 30 columns and handed 30 columns
        is a plain predict, and the pooled path therefore returns exactly what
        it returned before this method existed.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        want = getattr(self.model, "n_features_in_", None)
        if self.model is None or want is None or want == X.shape[1]:
            return self.predict(X)
        if want != X.shape[1] + len(QUERY_TYPE_NAMES):
            raise ValueError(
                f"estimator was fitted on {want} columns and given "
                f"{X.shape[1]}; expected either a match or "
                f"{want - len(QUERY_TYPE_NAMES)} for marginalising"
            )
        mix = self.qtype_mix or {
            "simple": 1.0, "temporal": 1.0, "enumerative": 1.0,
        }
        total = sum(mix.values()) or 1.0
        acc = np.zeros(X.shape[0], dtype=float)
        for qtype, weight in mix.items():
            tail = np.tile(query_type_onehot(qtype), (X.shape[0], 1))
            acc += (weight / total) * self.predict(np.hstack([X, tail]))
        return acc

    @property
    def is_fitted(self) -> bool:
        return self.model is not None
