"""Temporal statistics over true valid relative positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indices import FeatureEngineeringError

AGGREGATION_NAMES = (
    "valid_count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "amplitude",
    "p25",
    "p75",
    "iqr",
    "first",
    "last",
    "first_to_last",
    "slope",
)


@dataclass(frozen=True, slots=True)
class TemporalAggregateResult:
    """Ordered aggregate values for one monthly series."""

    values: np.ndarray
    names: tuple[str, ...] = AGGREGATION_NAMES

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise FeatureEngineeringError("temporal aggregate output shape is invalid")
        if np.isinf(self.values).any():
            raise FeatureEngineeringError("temporal aggregates must never contain infinity")


def aggregate_temporal_series(
    values: np.ndarray,
    relative_positions: np.ndarray,
    validity: np.ndarray,
) -> TemporalAggregateResult:
    """Aggregate one series without compressing gaps or admitting padded positions."""

    series = np.asarray(values, dtype=np.float64)
    positions = np.asarray(relative_positions, dtype=np.float64)
    mask = np.asarray(validity, dtype=bool).copy()
    if series.ndim != 2 or positions.shape != series.shape or mask.shape != series.shape:
        raise FeatureEngineeringError(
            "values, relative positions, and validity must be aligned 2D arrays"
        )
    mask &= np.isfinite(series) & np.isfinite(positions) & (positions > 0)
    counts = mask.sum(axis=1)
    output = np.full((series.shape[0], len(AGGREGATION_NAMES)), np.nan, dtype=np.float64)
    output[:, 0] = counts.astype(np.float64)

    has_value = counts > 0
    safe_counts = np.maximum(counts, 1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        means = (np.where(mask, series / safe_counts[:, None], 0.0)).sum(axis=1)
    output[:, 1] = np.where(has_value & np.isfinite(means), means, np.nan)

    sorted_values = np.sort(np.where(mask, series, np.inf), axis=1)

    def linear_quantile(probability: float) -> np.ndarray:
        locations = (safe_counts - 1.0) * probability
        lower = np.floor(locations).astype(np.intp)
        upper = np.ceil(locations).astype(np.intp)
        rows = np.arange(series.shape[0])
        lower_values = sorted_values[rows, lower]
        upper_values = sorted_values[rows, upper]
        with np.errstate(invalid="ignore", over="ignore"):
            quantile = lower_values + (upper_values - lower_values) * (locations - lower)
        return np.where(has_value & np.isfinite(quantile), quantile, np.nan)

    output[:, 2] = linear_quantile(0.5)
    output[:, 4] = np.where(has_value, sorted_values[:, 0], np.nan)
    last_valid_indices = np.maximum(counts - 1, 0)
    output[:, 5] = np.where(
        has_value,
        sorted_values[np.arange(series.shape[0]), last_valid_indices],
        np.nan,
    )
    output[:, 7] = linear_quantile(0.25)
    output[:, 8] = linear_quantile(0.75)

    at_least_two = counts >= 2
    centered = np.where(mask, series - output[:, 1, None], 0.0)
    scale = np.max(np.abs(centered), axis=1)
    nonzero_scale = at_least_two & np.isfinite(scale) & (scale > 0.0)
    scaled = np.zeros_like(centered)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(centered, scale[:, None], out=scaled, where=nonzero_scale[:, None])
        variance = (scaled * scaled).sum(axis=1) / safe_counts * scale * scale
        standard_deviation = np.sqrt(variance)
    zero_variance = at_least_two & np.isfinite(scale) & (scale == 0.0)
    output[:, 3] = np.where(
        nonzero_scale & np.isfinite(standard_deviation),
        standard_deviation,
        np.where(zero_variance, 0.0, np.nan),
    )
    output[:, 3] = np.where(at_least_two, output[:, 3], np.nan)
    output[:, 6] = np.where(at_least_two, output[:, 5] - output[:, 4], np.nan)
    output[:, 9] = np.where(at_least_two, output[:, 8] - output[:, 7], np.nan)

    first_indices = np.argmax(mask, axis=1)
    last_indices = series.shape[1] - 1 - np.argmax(np.flip(mask, axis=1), axis=1)
    row_indices = np.arange(series.shape[0])
    output[:, 10] = np.where(has_value, series[row_indices, first_indices], np.nan)
    output[:, 11] = np.where(has_value, series[row_indices, last_indices], np.nan)
    output[:, 12] = np.where(at_least_two, output[:, 11] - output[:, 10], np.nan)

    x = np.where(mask, positions, 0.0)
    y = np.where(mask, series, 0.0)
    n = counts.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean_x = x.sum(axis=1) / n
        mean_y = y.sum(axis=1) / n
        centered_x = np.where(mask, x - mean_x[:, None], 0.0)
        centered_y = np.where(mask, y - mean_y[:, None], 0.0)
        denominator = (centered_x * centered_x).sum(axis=1)
        slopes = (centered_x * centered_y).sum(axis=1) / denominator
    slope_valid = at_least_two & np.isfinite(denominator) & (denominator > 0.0)
    output[:, 13] = np.where(slope_valid & np.isfinite(slopes), slopes, np.nan)

    output[~np.isfinite(output)] = np.nan
    return TemporalAggregateResult(values=output)
