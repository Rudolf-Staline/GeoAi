"""Original-level metrics, temporal stress slices, stability, and robust scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.data import ValidationConfig
from geoai_aquaculture.metrics import MetricResult, metric_result

from .common import ValidationError
from .oof import OOFPredictions, aggregate_window_predictions


@dataclass(frozen=True, slots=True)
class SliceMetricResult:
    """One explicitly named original-level stress-slice metric result."""

    slice_group: str
    slice_value: str
    repeat: int
    metrics: MetricResult

    def as_dict(self) -> dict[str, Any]:
        """Flatten the slice and metric metadata for stable CSV/JSON storage."""

        return {
            "slice_group": self.slice_group,
            "slice_value": self.slice_value,
            "repeat": self.repeat,
            **self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Reusable official and robust evaluation over one complete OOF object."""

    summary: dict[str, Any]
    repeat_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    slice_metrics: pd.DataFrame
    prediction_stability: pd.DataFrame


def season_for_start_month(month: int, config: ValidationConfig) -> str:
    """Map a valid window start month to one configured season."""

    matches = [season.name for season in config.seasons if month in season.start_months]
    if len(matches) != 1:
        raise ValidationError(f"window start month {month} does not map to exactly one season")
    return matches[0]


def _metrics_record(group: pd.DataFrame, **metadata: Any) -> dict[str, Any]:
    metrics = metric_result(group["label"], group["probability"])
    return {**metadata, **metrics.as_dict()}


def evaluate_repeat_and_fold_metrics(
    original_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate repeats and folds on canonical originals, never duplicated windows."""

    repeat_records = [
        _metrics_record(group, repeat=int(repeat))
        for repeat, group in original_oof.groupby("repeat", sort=True, observed=True)
    ]
    fold_records = [
        _metrics_record(group, repeat=int(repeat), fold=int(fold))
        for (repeat, fold), group in original_oof.groupby(
            ["repeat", "fold"], sort=True, observed=True
        )
    ]
    return pd.DataFrame(repeat_records), pd.DataFrame(fold_records)


def _slice_selectors(
    frame: pd.DataFrame,
    config: ValidationConfig,
) -> list[tuple[str, str, np.ndarray]]:
    length = frame["window_length"].to_numpy(dtype=np.int8)
    start = frame["window_start"].to_numpy(dtype=np.int8)
    gaps = frame["internal_optical_gap_count"].to_numpy(dtype=np.int8)
    radar_months = frame["radar_months"].to_numpy(dtype=np.int8)
    optical_months = frame["optical_months"].to_numpy(dtype=np.int8)
    optical_proportion = optical_months / length
    selectors: list[tuple[str, str, np.ndarray]] = []
    for value in config.window_lengths:
        selectors.append(("window_length", str(value), length == value))
    for value in sorted(np.unique(start).tolist()):
        selectors.append(("start_month", str(value), start == value))
    for season in config.seasons:
        selectors.append(("season", season.name, np.isin(start, np.asarray(season.start_months))))
    selectors.extend(
        (
            ("optical_gaps", "none", gaps == 0),
            ("optical_gaps", "one", gaps == 1),
            ("optical_gaps", "two_or_more", gaps >= 2),
            (
                "optical_proportion",
                "severely_limited",
                optical_proportion < config.optical_severe_limit,
            ),
            (
                "optical_proportion",
                "moderate",
                (optical_proportion >= config.optical_severe_limit)
                & (optical_proportion < config.optical_high_completeness),
            ),
            (
                "optical_proportion",
                "high_incomplete",
                (optical_proportion >= config.optical_high_completeness)
                & (optical_proportion < 1.0),
            ),
            ("optical_proportion", "complete", optical_proportion == 1.0),
            ("availability", "radar_complete", radar_months == length),
            ("availability", "optical_complete", optical_months == length),
            ("availability", "radar_only", (radar_months > 0) & (optical_months == 0)),
            (
                "availability",
                "severely_optical_limited",
                (radar_months > 0) & (optical_proportion < config.optical_severe_limit),
            ),
        )
    )
    return selectors


def evaluate_temporal_slices(
    window_predictions: pd.DataFrame,
    config: ValidationConfig,
) -> pd.DataFrame:
    """Aggregate inside each stress view before computing any reported metric."""

    records: list[dict[str, Any]] = []
    for repeat, repeat_frame in window_predictions.groupby("repeat", sort=True, observed=True):
        for group_name, value, selector in _slice_selectors(repeat_frame, config):
            selected = repeat_frame.loc[selector]
            if selected.empty:
                continue
            aggregated = aggregate_window_predictions(
                selected,
                method=config.aggregation_method,
                trimmed_fraction=config.trimmed_mean_fraction,
            )
            result = SliceMetricResult(
                slice_group=group_name,
                slice_value=value,
                repeat=int(repeat),
                metrics=metric_result(aggregated["label"], aggregated["probability"]),
            )
            record = result.as_dict()
            record["window_prediction_count"] = int(selected.shape[0])
            record["original_prediction_count"] = int(aggregated.shape[0])
            records.append(record)
    return pd.DataFrame.from_records(records)


def prediction_stability(
    window_predictions: pd.DataFrame,
    *,
    aggregation_method: str = "mean",
    trimmed_fraction: float = 0.10,
) -> pd.DataFrame:
    """Summarize across-window probability variance for each original and repeat."""

    original = aggregate_window_predictions(
        window_predictions,
        method=aggregation_method,
        trimmed_fraction=trimmed_fraction,
    ).set_index(["repeat", "original_id"])
    records: list[dict[str, Any]] = []
    for (repeat, original_id), group in window_predictions.groupby(
        ["repeat", "original_id"], sort=False, observed=True
    ):
        p = group["probability"].to_numpy(dtype=np.float64)
        positive_proportion = float((p >= 0.5).mean())
        aggregate_row = original.loc[(repeat, original_id)]
        records.append(
            {
                "original_id": str(original_id),
                "repeat": int(repeat),
                "fold": int(group["fold"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "window_count": int(group.shape[0]),
                "mean_probability": float(np.mean(p)),
                "standard_deviation": float(np.std(p, ddof=0)),
                "minimum_probability": float(np.min(p)),
                "maximum_probability": float(np.max(p)),
                "probability_range": float(np.max(p) - np.min(p)),
                "positive_window_proportion": positive_proportion,
                "disagreement_rate": min(positive_proportion, 1.0 - positive_proportion),
                "aggregate_probability": float(aggregate_row["probability"]),
                "aggregate_prediction": int(aggregate_row["prediction"]),
                "is_error": bool(int(aggregate_row["prediction"]) != int(group["label"].iloc[0])),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["repeat", "fold", "original_id"], kind="stable", ignore_index=True
    )


def _finite_combined(frame: pd.DataFrame, label: str) -> np.ndarray:
    values = frame["combined_score"].dropna().to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValidationError(f"robust score component '{label}' is undefined")
    return values


def build_validation_report(
    oof: OOFPredictions,
    config: ValidationConfig,
) -> ValidationReport:
    """Compute the fixed official and robust Phase 4 model-comparison report."""

    repeat_metrics, fold_metrics = evaluate_repeat_and_fold_metrics(oof.original)
    slices = evaluate_temporal_slices(oof.windows, config)
    stability = prediction_stability(
        oof.windows,
        aggregation_method=config.aggregation_method,
        trimmed_fraction=config.trimmed_mean_fraction,
    )
    repeat_combined = _finite_combined(repeat_metrics, "mean_combined")
    fold_combined = _finite_combined(fold_metrics, "worst_fold")
    length_rows = slices.loc[slices["slice_group"].eq("window_length")]
    season_rows = slices.loc[slices["slice_group"].eq("season")]
    length_means = (
        length_rows.groupby("slice_value", observed=True)["combined_score"].mean().dropna()
    )
    season_means = (
        season_rows.groupby("slice_value", observed=True)["combined_score"].mean().dropna()
    )
    if set(length_means.index) != {"4", "5", "6"}:
        raise ValidationError("robust report requires defined 4-, 5-, and 6-month metrics")
    if season_means.empty:
        raise ValidationError("robust report requires at least one defined season metric")
    components = {
        "mean_combined_score": float(np.mean(repeat_combined)),
        "worst_fold_score": float(np.min(fold_combined)),
        "worst_window_length_score": float(length_means.min()),
        "worst_season_score": float(season_means.min()),
    }
    weights = config.robust_score_weights
    robust_score = (
        weights.mean_combined * components["mean_combined_score"]
        + weights.worst_fold * components["worst_fold_score"]
        + weights.worst_window_length * components["worst_window_length_score"]
        + weights.worst_season * components["worst_season_score"]
    )
    gap_rows = slices.loc[
        slices["slice_group"].eq("optical_gaps") & slices["slice_value"].eq("two_or_more")
    ]
    error_stability = stability.groupby("is_error").agg(
        originals=("original_id", "size"),
        mean_standard_deviation=("standard_deviation", "mean"),
        mean_probability_range=("probability_range", "mean"),
        mean_disagreement_rate=("disagreement_rate", "mean"),
    )
    summary: dict[str, Any] = {
        "official_metric": {
            "threshold": 0.5,
            "mean_f1": float(repeat_metrics["f1"].mean()),
            "mean_roc_auc": float(repeat_metrics["roc_auc"].mean()),
            "mean_combined_score": components["mean_combined_score"],
            "combined_score_standard_deviation": float(
                repeat_metrics["combined_score"].std(ddof=0)
            ),
            "worst_repeat_score": float(repeat_metrics["combined_score"].min()),
            "mean_fold_combined_score": float(fold_metrics["combined_score"].mean()),
            "fold_combined_score_standard_deviation": float(
                fold_metrics["combined_score"].std(ddof=0)
            ),
            "worst_fold_score": components["worst_fold_score"],
            "positive_prediction_rate": float(repeat_metrics["positive_prediction_rate"].mean()),
            "mean_log_loss": float(repeat_metrics["log_loss"].mean()),
            "mean_brier_score": float(repeat_metrics["brier_score"].mean()),
        },
        "robust_selection": {
            "score": float(robust_score),
            "components": components,
            "weights": {
                "mean_combined": weights.mean_combined,
                "worst_fold": weights.worst_fold,
                "worst_window_length": weights.worst_window_length,
                "worst_season": weights.worst_season,
            },
            "official_metric": False,
        },
        "window_length_scores": {key: float(value) for key, value in length_means.items()},
        "season_scores": {key: float(value) for key, value in season_means.items()},
        "optical_two_plus_gap_score": (
            float(gap_rows["combined_score"].mean()) if not gap_rows.empty else None
        ),
        "prediction_stability": {
            "mean_standard_deviation": float(stability["standard_deviation"].mean()),
            "mean_probability_range": float(stability["probability_range"].mean()),
            "mean_disagreement_rate": float(stability["disagreement_rate"].mean()),
            "errors_by_stability": {
                str(bool(index)): {
                    key: float(value) if key != "originals" else int(value)
                    for key, value in row.items()
                }
                for index, row in error_stability.to_dict(orient="index").items()
            },
        },
        "aggregation_method": config.aggregation_method,
        "fold_manifest_fingerprint": oof.fold_manifest_fingerprint,
        "validation_window_fingerprint": oof.validation_window_fingerprint,
        "oof_fingerprint": oof.fingerprint,
    }
    return ValidationReport(summary, repeat_metrics, fold_metrics, slices, stability)
