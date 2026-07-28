"""Competition metrics with the immutable 0.5 decision threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score, roc_auc_score

from .constants import FIXED_THRESHOLD


@dataclass(frozen=True, slots=True)
class CompetitionMetrics:
    """Metric bundle used for model selection and reporting."""

    f1: float
    roc_auc: float
    competition_score: float


def competition_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> CompetitionMetrics:
    """Compute F1, ROC-AUC and the official weighted score."""

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probabilities must be aligned one-dimensional arrays")
    if y.shape[0] == 0:
        raise ValueError("metric inputs must not be empty")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError("y_true must contain both binary classes 0 and 1")

    predictions = (p >= FIXED_THRESHOLD).astype(np.int8)
    f1 = float(f1_score(y, predictions))
    roc_auc = float(roc_auc_score(y, p))
    score = 0.60 * f1 + 0.40 * roc_auc
    return CompetitionMetrics(f1=f1, roc_auc=roc_auc, competition_score=score)
