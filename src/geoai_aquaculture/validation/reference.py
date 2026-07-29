"""One untuned fold-local estimator used only to test validation end to end."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from geoai_aquaculture.data import FeatureConfig, TemporalWindowDataset, ValidationConfig
from geoai_aquaculture.features import AGGREGATION_NAMES, aggregate_temporal_series, safe_divide

from .common import ValidationError, json_fingerprint, release_temporary_memory
from .evaluation import ValidationReport, build_validation_report
from .folds import FoldManifest
from .oof import OOFPredictions, build_oof_predictions, make_window_prediction_frame
from .views import ValidationWindowSet

REFERENCE_FEATURES = (
    "radar__vv__mean",
    "radar__vh__mean",
    "radar__vv_minus_vh__mean",
    "radar__vv__std",
    "radar__vh__std",
    "optical__ndvi__mean",
    "optical__ndwi__mean",
    "optical__mndwi__mean",
    "optical__ndmi__mean",
    "optical__nbr__mean",
    "metadata__window_length",
    "metadata__start_month_sin",
    "metadata__start_month_cos",
    "metadata__radar_valid_proportion",
    "metadata__optical_valid_proportion",
    "metadata__optical_gap_count",
)


class ProbabilisticEstimator(Protocol):
    """Minimum interface required by the fold-local validation orchestrator."""

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> object: ...

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


EstimatorFactory = Callable[[int], ProbabilisticEstimator]


@dataclass(frozen=True, slots=True)
class ReferenceRunResult:
    """Noncompetitive integration output following the full Phase 4 contracts."""

    oof: OOFPredictions
    report: ValidationReport
    fold_metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    runtime_seconds: float


def make_reference_estimator(seed: int) -> Pipeline:
    """Create a fresh untuned logistic pipeline with fold-local learned preprocessing."""

    return Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        )
    )


def _assert_unfitted(estimator: ProbabilisticEstimator) -> None:
    try:
        check_is_fitted(estimator)
    except (NotFittedError, TypeError):
        return
    raise ValidationError("validation orchestration rejects estimators fitted outside the fold")


def _aggregate_column(
    values: np.ndarray,
    mask: np.ndarray,
    windows: TemporalWindowDataset,
    statistic: str,
) -> np.ndarray:
    aggregate = aggregate_temporal_series(values, windows.relative_positions, mask)
    return aggregate.values[:, AGGREGATION_NAMES.index(statistic)]


def _reference_feature_frame(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Build only the declared integration features through Phase 3 numerical helpers."""

    raw_index = {band: index for index, band in enumerate(windows.band_names)}
    optical_index = {band: index for index, band in enumerate(windows.optical_bands)}
    roles = config.bands

    def raw(role_band: str, *, optical: bool) -> tuple[np.ndarray, np.ndarray]:
        values = windows.values[:, :, raw_index[role_band]]
        base_mask = (
            windows.optical_mask[:, :, optical_index[role_band]] if optical else windows.radar_mask
        )
        return values, base_mask & np.isfinite(values)

    vv, vv_mask = raw(roles.vv, optical=False)
    vh, vh_mask = raw(roles.vh, optical=False)
    columns: dict[str, np.ndarray] = {
        "radar__vv__mean": _aggregate_column(vv, vv_mask, windows, "mean"),
        "radar__vh__mean": _aggregate_column(vh, vh_mask, windows, "mean"),
        "radar__vv__std": _aggregate_column(vv, vv_mask, windows, "std"),
        "radar__vh__std": _aggregate_column(vh, vh_mask, windows, "std"),
    }
    radar_difference = vv - vh
    columns["radar__vv_minus_vh__mean"] = _aggregate_column(
        radar_difference,
        vv_mask & vh_mask & np.isfinite(radar_difference),
        windows,
        "mean",
    )

    def normalized_difference(name: str, left_band: str, right_band: str) -> None:
        left, left_mask = raw(left_band, optical=True)
        right, right_mask = raw(right_band, optical=True)
        divided = safe_divide(left - right, left + right, epsilon=config.epsilon)
        validity = divided.validity & left_mask & right_mask
        columns[f"optical__{name}__mean"] = _aggregate_column(
            divided.values,
            validity,
            windows,
            "mean",
        )

    normalized_difference("ndvi", roles.nir, roles.red)
    normalized_difference("ndwi", roles.green, roles.nir)
    normalized_difference("mndwi", roles.green, roles.swir1)
    normalized_difference("ndmi", roles.nir, roles.swir1)
    normalized_difference("nbr", roles.nir, roles.swir2)

    length = windows.position_mask.sum(axis=1).astype(np.float64)
    start = windows.manifest["window_start"].to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * (start - 1.0) / 12.0
    optical_observations = windows.optical_mask.sum(axis=(1, 2)).astype(np.float64)
    optical_month = windows.optical_mask.all(axis=2) & windows.position_mask
    columns.update(
        {
            "metadata__window_length": length,
            "metadata__start_month_sin": np.sin(angle),
            "metadata__start_month_cos": np.cos(angle),
            "metadata__radar_valid_proportion": windows.radar_mask.sum(axis=1) / length,
            "metadata__optical_valid_proportion": optical_observations
            / (length * len(windows.optical_bands)),
            "metadata__optical_gap_count": (
                windows.radar_mask & windows.position_mask & ~optical_month
            ).sum(axis=1),
        }
    )
    frame = pd.DataFrame(columns, columns=REFERENCE_FEATURES, dtype=np.float64)
    if np.isinf(frame.to_numpy()).any():
        raise ValidationError("reference features must contain no infinity")
    return frame


def run_reference_estimator(
    windows: ValidationWindowSet,
    folds: FoldManifest,
    validation_config: ValidationConfig,
    feature_config: FeatureConfig,
    *,
    estimator_factory: EstimatorFactory = make_reference_estimator,
    experiment_id: str = "PHASE4-REFERENCE",
    model_id: str = "logistic_integration_reference",
) -> ReferenceRunResult:
    """Exercise fixed folds, fold-local preprocessing, OOF aggregation, and slices."""

    if windows.manifest.fold_manifest_fingerprint != folds.fingerprint:
        raise ValidationError("reference windows and folds have incompatible fingerprints")
    started = perf_counter()
    all_probabilities = np.full(windows.manifest.frame.shape[0], np.nan, dtype=np.float64)
    metadata: list[dict[str, object]] = []
    used_estimators: list[weakref.ReferenceType[object]] = []
    expected_schema: str | None = None
    repeat_size = windows.manifest.frame.shape[0] // folds.n_repeats
    for repeat in range(folds.n_repeats):
        dataset = windows.for_repeat(repeat)
        selected = _reference_feature_frame(dataset, feature_config)
        release_temporary_memory()
        schema_fingerprint = json_fingerprint(
            {"features": list(REFERENCE_FEATURES), "version": feature_config.version}
        )
        if expected_schema is None:
            expected_schema = schema_fingerprint
        elif schema_fingerprint != expected_schema:
            raise ValidationError("reference feature schema changed between repeats")
        labels = dataset.manifest["label"].to_numpy(dtype=np.int8)
        folds_array = dataset.manifest["fold"].to_numpy(dtype=np.int16)
        original_ids = dataset.manifest["original_id"].astype("string").to_numpy()
        for fold in range(folds.n_splits):
            fold_started = perf_counter()
            valid_selector = folds_array == fold
            train_selector = ~valid_selector
            train_originals = set(original_ids[train_selector].astype(str))
            valid_originals = set(original_ids[valid_selector].astype(str))
            if train_originals & valid_originals:
                raise ValidationError("reference estimator detected original-row fold leakage")
            fold_seed = validation_config.seed + repeat * 10_007 + fold
            estimator = estimator_factory(fold_seed)
            if any(estimator is previous() for previous in used_estimators):
                raise ValidationError("estimator factory must return a fresh object for every fold")
            used_estimators.append(weakref.ref(estimator))
            _assert_unfitted(estimator)
            estimator.fit(selected.loc[train_selector], labels[train_selector])
            probability_matrix = np.asarray(
                estimator.predict_proba(selected.loc[valid_selector]), dtype=np.float64
            )
            if probability_matrix.ndim != 2 or probability_matrix.shape[1] != 2:
                raise ValidationError("reference estimator must return two-class probabilities")
            repeat_indices = np.flatnonzero(valid_selector) + repeat * repeat_size
            all_probabilities[repeat_indices] = probability_matrix[:, 1]
            metadata.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "fold_seed": fold_seed,
                    "train_original_count": len(train_originals),
                    "validation_original_count": len(valid_originals),
                    "train_window_count": int(train_selector.sum()),
                    "validation_window_count": int(valid_selector.sum()),
                    "feature_count": len(REFERENCE_FEATURES),
                    "feature_schema_fingerprint": schema_fingerprint,
                    "preprocessing_scope": "fit_only_on_current_fold_training_windows",
                    "runtime_seconds": perf_counter() - fold_started,
                }
            )
            del estimator, probability_matrix
            release_temporary_memory()
        del dataset, selected
        release_temporary_memory()
    if not np.isfinite(all_probabilities).all():
        raise ValidationError("reference fold execution left missing window predictions")
    window_frame = make_window_prediction_frame(
        windows.manifest.frame,
        all_probabilities,
        experiment_id=experiment_id,
        model_id=model_id,
        fold_manifest_fingerprint=folds.fingerprint,
        validation_window_fingerprint=windows.manifest.fingerprint,
    )
    oof = build_oof_predictions(
        window_frame,
        folds,
        validation_window_fingerprint=windows.manifest.fingerprint,
        method=validation_config.aggregation_method,
        trimmed_fraction=validation_config.trimmed_mean_fraction,
    )
    report = build_validation_report(oof, validation_config)
    return ReferenceRunResult(
        oof=oof,
        report=report,
        fold_metadata=pd.DataFrame.from_records(metadata),
        feature_names=REFERENCE_FEATURES,
        runtime_seconds=perf_counter() - started,
    )
