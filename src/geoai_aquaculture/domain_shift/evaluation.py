"""Test-like holdout and importance-weight diagnostics using existing OOF evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoai_aquaculture.metrics import metric_result
from geoai_aquaculture.validation import FoldManifest, SimilarityHoldoutManifest


class DomainEvaluationError(ValueError):
    """Raised when retained-model OOF artifacts do not align with similarity scores."""


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """Per-repeat label metrics on the most test-like training originals."""

    model_name: str
    repeat_metrics: pd.DataFrame
    mean_combined_score: float
    worst_repeat_score: float
    selected_count: int


def load_original_oof(path: str | Path, folds: FoldManifest) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"original_id", "repeat", "fold", "label", "probability"}
    if missing := sorted(required - set(frame.columns)):
        raise DomainEvaluationError(f"label OOF is missing columns: {missing}")
    if frame.duplicated(["original_id", "repeat"]).any():
        raise DomainEvaluationError("label OOF contains duplicate original/repeat rows")
    expected = folds.n_originals * folds.n_repeats
    if frame.shape[0] != expected:
        raise DomainEvaluationError(f"label OOF rows changed: {frame.shape[0]} != {expected}")
    return frame


def evaluate_similarity_holdout(
    oof: pd.DataFrame,
    holdout: SimilarityHoldoutManifest,
    *,
    model_name: str,
) -> HoldoutEvaluation:
    selected = holdout.frame.loc[holdout.frame["selected"], ["original_id"]]
    subset = oof.merge(selected, on="original_id", how="inner", validate="many_to_one")
    records: list[dict[str, float | int]] = []
    for repeat, group in subset.groupby("repeat", sort=True, observed=True):
        metrics = metric_result(group["label"], group["probability"])
        records.append({"repeat": int(repeat), **metrics.as_dict()})
    repeat_metrics = pd.DataFrame.from_records(records)
    combined = repeat_metrics["combined_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(combined).all():
        raise DomainEvaluationError("test-like holdout metrics are undefined")
    return HoldoutEvaluation(
        model_name=model_name,
        repeat_metrics=repeat_metrics,
        mean_combined_score=float(np.mean(combined)),
        worst_repeat_score=float(np.min(combined)),
        selected_count=holdout.selected_count,
    )


def build_importance_weights(
    scores: pd.DataFrame,
    *,
    minimum: float,
    maximum: float,
) -> pd.DataFrame:
    """Convert balanced OOF domain probabilities to clipped density-ratio weights."""

    required = {"original_id", "similarity_score", "is_oof"}
    if missing := sorted(required - set(scores.columns)):
        raise DomainEvaluationError(f"domain score table is missing columns: {missing}")
    if scores["original_id"].duplicated().any() or not scores["is_oof"].all():
        raise DomainEvaluationError("importance weighting requires one OOF score per original")
    probability = scores["similarity_score"].to_numpy(dtype=np.float64)
    clipped_probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    ratio = clipped_probability / (1.0 - clipped_probability)
    positive = ratio[ratio > 0.0]
    center = float(np.median(positive)) if positive.size else 1.0
    centered_ratio = ratio / max(center, 1e-12)
    clipped = np.clip(centered_ratio, minimum, maximum)
    result = scores.loc[:, ["original_id", "similarity_score", "is_oof"]].copy()
    result["raw_density_ratio"] = ratio
    result["median_centered_ratio"] = centered_ratio
    result["importance_weight"] = clipped
    result["was_clipped_low"] = centered_ratio < minimum
    result["was_clipped_high"] = centered_ratio > maximum
    return result.sort_values("original_id", kind="stable", ignore_index=True)
