"""Fold-local window weighting with original-level and class-balance guarantees."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import WeightingPolicy


class WeightingError(ValueError):
    """Raised when fold-local sample weights cannot satisfy the declared policy."""


@dataclass(frozen=True, slots=True)
class SampleWeightResult:
    """Window weights plus the fold-local class weights used to derive them."""

    values: np.ndarray
    policy: WeightingPolicy
    class_weights: dict[int, float]
    original_weight_min: float
    original_weight_max: float

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.size == 0:
            raise WeightingError("sample weights must be a non-empty one-dimensional array")
        if not np.isfinite(self.values).all() or (self.values <= 0.0).any():
            raise WeightingError("sample weights must be finite and strictly positive")


def _fold_local_balanced_class_weights(labels: pd.Series) -> dict[int, float]:
    counts = labels.value_counts().sort_index()
    if set(counts.index.tolist()) != {0, 1}:
        raise WeightingError("fold-training originals must contain both binary classes")
    total = float(counts.sum())
    return {label: total / (2.0 * float(count)) for label, count in counts.items()}


def build_window_sample_weights(
    original_ids: np.ndarray,
    labels: np.ndarray,
    policy: WeightingPolicy,
) -> SampleWeightResult:
    """Compute weights from current-fold training rows, never validation prevalence."""

    ids = pd.Series(np.asarray(original_ids, dtype=str), dtype="string")
    target = pd.Series(np.asarray(labels, dtype=np.int8))
    if ids.size == 0 or ids.size != target.size:
        raise WeightingError("sample-weight IDs and labels must be aligned and non-empty")
    if ids.isna().any() or set(target.unique().tolist()) != {0, 1}:
        raise WeightingError("sample-weight inputs require non-missing IDs and both labels")
    attached = pd.DataFrame({"original_id": ids, "label": target})
    if attached.groupby("original_id", observed=True)["label"].nunique().ne(1).any():
        raise WeightingError("one original cannot carry multiple labels within a fold")

    if policy in {"equal_original", "equal_original_class_weighted"}:
        counts = attached.groupby("original_id", observed=True)["original_id"].transform("size")
        weights = 1.0 / counts.to_numpy(dtype=np.float64)
    elif policy in {"uniform", "class_weighted"}:
        weights = np.ones(attached.shape[0], dtype=np.float64)
    else:
        raise WeightingError(f"unsupported sample-weight policy: {policy}")

    class_weights = {0: 1.0, 1: 1.0}
    if policy in {"class_weighted", "equal_original_class_weighted"}:
        originals = attached.drop_duplicates("original_id", keep="first")
        class_weights = _fold_local_balanced_class_weights(originals["label"])
        weights *= attached["label"].map(class_weights).to_numpy(dtype=np.float64)

    totals = pd.Series(weights).groupby(attached["original_id"], observed=True).sum()
    return SampleWeightResult(
        values=weights,
        policy=policy,
        class_weights=class_weights,
        original_weight_min=float(totals.min()),
        original_weight_max=float(totals.max()),
    )
