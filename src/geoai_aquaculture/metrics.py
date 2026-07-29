"""Competition metrics with the immutable 0.5 decision threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .constants import FIXED_THRESHOLD


@dataclass(frozen=True, slots=True)
class CompetitionMetrics:
    """Metric bundle used for model selection and reporting."""

    f1: float
    roc_auc: float
    competition_score: float


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Official and secondary metrics, including explicit undefined-slice state."""

    sample_count: int
    positive_count: int
    negative_count: int
    f1: float
    roc_auc: float | None
    combined_score: float | None
    auc_defined: bool
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    positive_prediction_rate: float
    log_loss: float
    brier_score: float
    threshold: float = FIXED_THRESHOLD

    def as_dict(self) -> dict[str, float | int | bool | None]:
        """Return JSON-safe metric values with no fabricated AUC."""

        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "combined_score": self.combined_score,
            "auc_defined": self.auc_defined,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_positive": self.true_positive,
            "positive_prediction_rate": self.positive_prediction_rate,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "threshold": self.threshold,
        }


def _validated_metric_inputs(
    y_true: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probabilities must be aligned one-dimensional arrays")
    if y.shape[0] == 0:
        raise ValueError("metric inputs must not be empty")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return y.astype(np.int8, copy=False), p


def metric_result(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    threshold: float = FIXED_THRESHOLD,
) -> MetricResult:
    """Compute exact official metrics and explicit diagnostics at fixed threshold 0.5."""

    if threshold != FIXED_THRESHOLD:
        raise ValueError(f"classification threshold must remain exactly {FIXED_THRESHOLD}")
    y, p = _validated_metric_inputs(y_true, probabilities)
    predictions = (p >= FIXED_THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    f1 = float(f1_score(y, predictions, zero_division=0))
    auc_defined = np.unique(y).size == 2
    auc = float(roc_auc_score(y, p)) if auc_defined else None
    combined = 0.60 * f1 + 0.40 * auc if auc is not None else None
    return MetricResult(
        sample_count=int(y.size),
        positive_count=int(y.sum()),
        negative_count=int(y.size - y.sum()),
        f1=f1,
        roc_auc=auc,
        combined_score=combined,
        auc_defined=auc_defined,
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        positive_prediction_rate=float(predictions.mean()),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, p)),
    )


def competition_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> CompetitionMetrics:
    """Compute F1, ROC-AUC and the official weighted score."""

    y, p = _validated_metric_inputs(y_true, probabilities)
    if np.unique(y).size != 2:
        raise ValueError("y_true must contain both binary classes 0 and 1")
    result = metric_result(y, p)
    assert result.roc_auc is not None and result.combined_score is not None
    return CompetitionMetrics(
        f1=result.f1,
        roc_auc=result.roc_auc,
        competition_score=result.combined_score,
    )
