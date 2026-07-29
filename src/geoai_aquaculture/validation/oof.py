"""Model-agnostic window prediction aggregation and original-level OOF contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from geoai_aquaculture.constants import FIXED_THRESHOLD

from .common import ValidationError, dataframe_fingerprint, validate_probabilities
from .folds import FoldManifest

AggregationMethod = Literal["mean", "median", "logit_mean", "trimmed_mean"]

WINDOW_PREDICTION_COLUMNS = (
    "ID",
    "original_id",
    "repeat",
    "fold",
    "y_true",
    "label",
    "probability",
    "prediction",
    "predicted_class",
    "window_id",
    "window_start",
    "window_end",
    "window_length",
    "radar_months",
    "optical_months",
    "internal_optical_gap_count",
    "experiment_id",
    "model_id",
    "fold_manifest_fingerprint",
    "validation_window_fingerprint",
)

ORIGINAL_OOF_COLUMNS = (
    "ID",
    "original_id",
    "repeat",
    "fold",
    "y_true",
    "label",
    "probability",
    "prediction",
    "predicted_class",
    "window_count",
    "window_start",
    "window_length",
    "optical_months",
    "aggregation_method",
    "experiment_id",
    "model_id",
    "fold_manifest_fingerprint",
    "validation_window_fingerprint",
)


def aggregate_probabilities(
    probabilities: np.ndarray,
    *,
    method: AggregationMethod = "mean",
    trimmed_fraction: float = 0.10,
    epsilon: float = 1e-6,
) -> float:
    """Aggregate window probabilities without learning from validation labels."""

    values = np.asarray(probabilities, dtype=np.float64)
    validate_probabilities(values)
    if method == "mean":
        result = np.mean(values)
    elif method == "median":
        result = np.median(values)
    elif method == "logit_mean":
        if not 0.0 < epsilon < 0.5:
            raise ValidationError("logit aggregation epsilon must be in (0, 0.5)")
        clipped = np.clip(values, epsilon, 1.0 - epsilon)
        logits = np.log(clipped) - np.log1p(-clipped)
        mean_logit = float(np.mean(logits))
        result = 1.0 / (1.0 + np.exp(-mean_logit))
    elif method == "trimmed_mean":
        if not 0.0 <= trimmed_fraction < 0.5:
            raise ValidationError("trimmed mean fraction must be in [0, 0.5)")
        ordered = np.sort(values)
        trim = int(np.floor(ordered.size * trimmed_fraction))
        retained = ordered[trim : ordered.size - trim] if trim else ordered
        result = np.mean(retained)
    else:
        raise ValidationError(f"unsupported original-level aggregation method: {method}")
    result = float(result)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValidationError("aggregated probability is not finite within [0, 1]")
    return result


def make_window_prediction_frame(
    window_manifest: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    experiment_id: str,
    model_id: str,
    fold_manifest_fingerprint: str,
    validation_window_fingerprint: str,
) -> pd.DataFrame:
    """Attach bounded probabilities to fixed validation-window metadata."""

    p = np.asarray(probabilities, dtype=np.float64)
    validate_probabilities(p)
    if len(p) != window_manifest.shape[0]:
        raise ValidationError("window probabilities do not align with validation metadata")
    required = {
        "original_id",
        "repeat",
        "fold",
        "label",
        "window_id",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
    }
    missing = sorted(required - set(window_manifest.columns))
    if missing:
        raise ValidationError(f"window prediction metadata is missing columns: {missing}")
    if window_manifest.duplicated(["repeat", "window_id"]).any():
        raise ValidationError("window prediction metadata contains duplicate repeat/window IDs")
    prediction = (p >= FIXED_THRESHOLD).astype(np.int8)
    frame = pd.DataFrame(
        {
            "ID": window_manifest["original_id"].astype("string").to_numpy(),
            "original_id": window_manifest["original_id"].astype("string").to_numpy(),
            "repeat": window_manifest["repeat"].to_numpy(dtype=np.int16),
            "fold": window_manifest["fold"].to_numpy(dtype=np.int16),
            "y_true": window_manifest["label"].to_numpy(dtype=np.int8),
            "label": window_manifest["label"].to_numpy(dtype=np.int8),
            "probability": p,
            "prediction": prediction,
            "predicted_class": prediction,
            "window_id": window_manifest["window_id"].astype("string").to_numpy(),
            "window_start": window_manifest["window_start"].to_numpy(dtype=np.int8),
            "window_end": window_manifest["window_end"].to_numpy(dtype=np.int8),
            "window_length": window_manifest["window_length"].to_numpy(dtype=np.int8),
            "radar_months": window_manifest["radar_months"].to_numpy(dtype=np.int8),
            "optical_months": window_manifest["optical_months"].to_numpy(dtype=np.int8),
            "internal_optical_gap_count": window_manifest["internal_optical_gap_count"].to_numpy(
                dtype=np.int8
            ),
            "experiment_id": experiment_id,
            "model_id": model_id,
            "fold_manifest_fingerprint": fold_manifest_fingerprint,
            "validation_window_fingerprint": validation_window_fingerprint,
        }
    )
    frame = frame.loc[:, list(WINDOW_PREDICTION_COLUMNS)]
    for column in (
        "ID",
        "original_id",
        "experiment_id",
        "model_id",
        "fold_manifest_fingerprint",
        "validation_window_fingerprint",
    ):
        frame[column] = frame[column].astype("category")
    for column in (
        "y_true",
        "label",
        "prediction",
        "predicted_class",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
    ):
        frame[column] = frame[column].astype(np.int8)
    frame["repeat"] = frame["repeat"].astype(np.int16)
    frame["fold"] = frame["fold"].astype(np.int16)
    return frame


def aggregate_window_predictions(
    window_predictions: pd.DataFrame,
    *,
    method: AggregationMethod = "mean",
    trimmed_fraction: float = 0.10,
) -> pd.DataFrame:
    """Return exactly one canonical OOF probability per original and repeat."""

    missing = sorted(set(WINDOW_PREDICTION_COLUMNS) - set(window_predictions.columns))
    if missing:
        raise ValidationError(f"window predictions are missing columns: {missing}")
    p = window_predictions["probability"].to_numpy(dtype=np.float64)
    validate_probabilities(p)
    if window_predictions.duplicated(["repeat", "window_id"]).any():
        raise ValidationError("window predictions contain duplicate repeat/window IDs")
    records: list[dict[str, object]] = []
    grouped = window_predictions.groupby(["repeat", "original_id"], sort=False, observed=True)
    for (repeat, original_id), group in grouped:
        for column in (
            "fold",
            "label",
            "y_true",
            "experiment_id",
            "model_id",
            "fold_manifest_fingerprint",
            "validation_window_fingerprint",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValidationError(f"window group has inconsistent {column}")
        probability = aggregate_probabilities(
            group["probability"].to_numpy(dtype=np.float64),
            method=method,
            trimmed_fraction=trimmed_fraction,
        )
        prediction = int(probability >= FIXED_THRESHOLD)
        records.append(
            {
                "ID": str(original_id),
                "original_id": str(original_id),
                "repeat": int(repeat),
                "fold": int(group["fold"].iloc[0]),
                "y_true": int(group["y_true"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "probability": probability,
                "prediction": prediction,
                "predicted_class": prediction,
                "window_count": int(group.shape[0]),
                # Mixed-window original aggregates use explicit nonphysical sentinels.
                "window_start": -1,
                "window_length": -1,
                "optical_months": float(group["optical_months"].mean()),
                "aggregation_method": method,
                "experiment_id": str(group["experiment_id"].iloc[0]),
                "model_id": str(group["model_id"].iloc[0]),
                "fold_manifest_fingerprint": str(group["fold_manifest_fingerprint"].iloc[0]),
                "validation_window_fingerprint": str(
                    group["validation_window_fingerprint"].iloc[0]
                ),
            }
        )
    frame = pd.DataFrame.from_records(records, columns=ORIGINAL_OOF_COLUMNS)
    frame = frame.sort_values(["repeat", "fold", "original_id"], kind="stable").reset_index(
        drop=True
    )
    for column in (
        "ID",
        "original_id",
        "aggregation_method",
        "experiment_id",
        "model_id",
        "fold_manifest_fingerprint",
        "validation_window_fingerprint",
    ):
        frame[column] = frame[column].astype("category")
    for column in (
        "y_true",
        "label",
        "prediction",
        "predicted_class",
        "window_start",
        "window_length",
    ):
        frame[column] = frame[column].astype(np.int8)
    frame["repeat"] = frame["repeat"].astype(np.int16)
    frame["fold"] = frame["fold"].astype(np.int16)
    frame["window_count"] = frame["window_count"].astype(np.int16)
    return frame


@dataclass(frozen=True, slots=True)
class OOFPredictions:
    """Validated original/window OOF tables linked to immutable manifests."""

    original: pd.DataFrame
    windows: pd.DataFrame
    fold_manifest_fingerprint: str
    validation_window_fingerprint: str
    aggregation_method: AggregationMethod
    trimmed_fraction: float
    fingerprint: str

    def __post_init__(self) -> None:
        missing_original = sorted(set(ORIGINAL_OOF_COLUMNS) - set(self.original.columns))
        missing_windows = sorted(set(WINDOW_PREDICTION_COLUMNS) - set(self.windows.columns))
        if missing_original or missing_windows:
            raise ValidationError(
                f"OOF schema mismatch: original={missing_original}, windows={missing_windows}"
            )
        if self.original.duplicated(["original_id", "repeat"]).any():
            raise ValidationError("OOF must contain one original-level row per repeat")
        if self.windows.duplicated(["repeat", "window_id"]).any():
            raise ValidationError("window-level OOF contains duplicate predictions")
        for frame in (self.original, self.windows):
            validate_probabilities(frame["probability"].to_numpy(dtype=np.float64))
            expected = (frame["probability"].to_numpy() >= FIXED_THRESHOLD).astype(np.int8)
            if not np.array_equal(frame["prediction"].to_numpy(dtype=np.int8), expected):
                raise ValidationError("OOF predictions do not use the immutable 0.5 threshold")
            if (
                frame["fold_manifest_fingerprint"].nunique() != 1
                or str(frame["fold_manifest_fingerprint"].iloc[0]) != self.fold_manifest_fingerprint
            ):
                raise ValidationError("OOF fold-manifest fingerprint mismatch")
            if (
                frame["validation_window_fingerprint"].nunique() != 1
                or str(frame["validation_window_fingerprint"].iloc[0])
                != self.validation_window_fingerprint
            ):
                raise ValidationError("OOF validation-window fingerprint mismatch")
        rebuilt = aggregate_window_predictions(
            self.windows,
            method=self.aggregation_method,
            trimmed_fraction=self.trimmed_fraction,
        )
        compare = [
            "original_id",
            "repeat",
            "fold",
            "label",
            "probability",
            "prediction",
            "window_count",
        ]
        try:
            pd.testing.assert_frame_equal(
                self.original.loc[:, compare].reset_index(drop=True),
                rebuilt.loc[:, compare].reset_index(drop=True),
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValidationError(
                "original-level OOF does not match deterministic window aggregation"
            ) from exc
        actual = dataframe_fingerprint(self.original, columns=ORIGINAL_OOF_COLUMNS)
        if actual != self.fingerprint:
            raise ValidationError("original-level OOF fingerprint mismatch")


def build_oof_predictions(
    window_predictions: pd.DataFrame,
    folds: FoldManifest,
    *,
    validation_window_fingerprint: str,
    method: AggregationMethod = "mean",
    trimmed_fraction: float = 0.10,
) -> OOFPredictions:
    """Aggregate, align, and validate complete repeated original-level OOF predictions."""

    original = aggregate_window_predictions(
        window_predictions,
        method=method,
        trimmed_fraction=trimmed_fraction,
    )
    expected = folds.frame.loc[:, ["original_id", "repeat", "fold", "label"]].copy()
    actual = original.loc[:, ["original_id", "repeat", "fold", "label"]].copy()
    expected = expected.sort_values(["repeat", "fold", "original_id"], kind="stable").reset_index(
        drop=True
    )
    actual = actual.sort_values(["repeat", "fold", "original_id"], kind="stable").reset_index(
        drop=True
    )
    if expected.shape != actual.shape or not np.array_equal(
        expected.astype(
            {"original_id": "string", "repeat": "int64", "fold": "int64", "label": "int64"}
        ).to_numpy(),
        actual.astype(
            {"original_id": "string", "repeat": "int64", "fold": "int64", "label": "int64"}
        ).to_numpy(),
    ):
        raise ValidationError("OOF predictions have duplicate, missing, or misaligned originals")
    fingerprint = dataframe_fingerprint(original, columns=ORIGINAL_OOF_COLUMNS)
    return OOFPredictions(
        original=original,
        windows=window_predictions.reset_index(drop=True),
        fold_manifest_fingerprint=folds.fingerprint,
        validation_window_fingerprint=validation_window_fingerprint,
        aggregation_method=method,
        trimmed_fraction=trimmed_fraction,
        fingerprint=fingerprint,
    )


def load_oof_predictions(
    original_path: str | Path,
    window_path: str | Path,
    folds: FoldManifest,
    *,
    validation_window_fingerprint: str,
    method: AggregationMethod = "mean",
    trimmed_fraction: float = 0.10,
    expected_fingerprint: str | None = None,
) -> OOFPredictions:
    """Load persisted OOF tables and re-derive originals from window predictions."""

    original_source = Path(original_path)
    window_source = Path(window_path)
    for source in (original_source, window_source):
        if not source.is_file():
            raise FileNotFoundError(f"OOF table not found: {source}")
    string_columns = {
        "ID": "string",
        "original_id": "string",
        "experiment_id": "string",
        "model_id": "string",
        "fold_manifest_fingerprint": "string",
        "validation_window_fingerprint": "string",
    }
    windows = pd.read_csv(
        window_source,
        dtype={**string_columns, "window_id": "string"},
        float_precision="round_trip",
    )
    original = pd.read_csv(
        original_source,
        dtype={**string_columns, "aggregation_method": "string"},
        float_precision="round_trip",
    )
    for frame in (windows, original):
        for column in string_columns:
            frame[column] = frame[column].astype(object).astype("category")
    windows["window_id"] = windows["window_id"].astype(object)
    for column in (
        "y_true",
        "label",
        "prediction",
        "predicted_class",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
    ):
        windows[column] = windows[column].astype(np.int8)
    windows["repeat"] = windows["repeat"].astype(np.int16)
    windows["fold"] = windows["fold"].astype(np.int16)
    original["aggregation_method"] = (
        original["aggregation_method"].astype(object).astype("category")
    )
    for column in (
        "y_true",
        "label",
        "prediction",
        "predicted_class",
        "window_start",
        "window_length",
    ):
        original[column] = original[column].astype(np.int8)
    original["repeat"] = original["repeat"].astype(np.int16)
    original["fold"] = original["fold"].astype(np.int16)
    original["window_count"] = original["window_count"].astype(np.int16)
    original["optical_months"] = original["optical_months"].astype(np.float64)
    rebuilt = build_oof_predictions(
        windows,
        folds,
        validation_window_fingerprint=validation_window_fingerprint,
        method=method,
        trimmed_fraction=trimmed_fraction,
    )
    try:
        pd.testing.assert_frame_equal(
            original.loc[:, list(ORIGINAL_OOF_COLUMNS)],
            rebuilt.original.loc[:, list(ORIGINAL_OOF_COLUMNS)],
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValidationError("persisted original OOF does not match window predictions") from exc
    if expected_fingerprint is not None and rebuilt.fingerprint != expected_fingerprint:
        raise ValidationError("loaded OOF fingerprint mismatch")
    return rebuilt
