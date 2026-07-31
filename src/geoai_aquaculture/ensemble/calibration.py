"""Fold-local sigmoid and beta probability calibration for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from geoai_aquaculture.data import ProjectConfig
from geoai_aquaculture.validation import (
    OOFPredictions,
    ValidationReport,
    build_oof_predictions,
    build_validation_report,
    load_fold_manifest,
)

CalibrationMethod = Literal["none", "sigmoid", "beta"]


class CalibrationError(ValueError):
    """Raised when calibration would leak or produce invalid probabilities."""


@dataclass(frozen=True, slots=True)
class FittedCalibrator:
    """Serializable logistic calibrator operating on bounded probabilities."""

    method: CalibrationMethod
    coefficients: tuple[float, ...]
    intercept: float
    epsilon: float = 1e-6

    def _features(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
            raise CalibrationError("calibration probabilities must be finite in [0, 1]")
        clipped = np.clip(p, self.epsilon, 1.0 - self.epsilon)
        if self.method == "none":
            return clipped.reshape(-1, 1)
        if self.method == "sigmoid":
            return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)
        if self.method == "beta":
            return np.column_stack((np.log(clipped), -np.log1p(-clipped)))
        raise CalibrationError(f"unsupported calibration method: {self.method}")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return np.asarray(probabilities, dtype=np.float64).copy()
        features = self._features(probabilities)
        coefficient = np.asarray(self.coefficients, dtype=np.float64)
        logits = features @ coefficient + self.intercept
        result = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
            raise CalibrationError("calibration produced invalid probabilities")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    """Cross-fitted OOF calibration plus a full-data production calibrator."""

    method: CalibrationMethod
    oof: OOFPredictions
    report: ValidationReport
    production_calibrator: FittedCalibrator
    expected_calibration_error: float
    fold_parameters: pd.DataFrame


def _features(probabilities: np.ndarray, method: CalibrationMethod, epsilon: float) -> np.ndarray:
    calibrator = FittedCalibrator(method=method, coefficients=(), intercept=0.0, epsilon=epsilon)
    return calibrator._features(probabilities)


def _fit_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    method: CalibrationMethod,
    *,
    epsilon: float = 1e-6,
) -> FittedCalibrator:
    if method == "none":
        return FittedCalibrator(method="none", coefficients=(1.0,), intercept=0.0, epsilon=epsilon)
    y = np.asarray(labels, dtype=np.int8)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise CalibrationError("calibration training requires both classes")
    x = _features(np.asarray(probabilities, dtype=np.float64), method, epsilon)
    model = LogisticRegression(
        C=10.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=2026,
    )
    model.fit(x, y)
    coefficients = tuple(float(value) for value in model.coef_[0])
    intercept = float(model.intercept_[0])
    return FittedCalibrator(
        method=method,
        coefficients=coefficients,
        intercept=intercept,
        epsilon=epsilon,
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Compute fixed equal-width expected calibration error."""

    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        selector = indices == index
        if not selector.any():
            continue
        error += float(selector.mean()) * abs(float(p[selector].mean()) - float(y[selector].mean()))
    return float(error)


def _calibrated_windows(
    base: OOFPredictions,
    project: ProjectConfig,
    method: CalibrationMethod,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = base.original.copy()
    windows = base.windows.copy()
    records: list[dict[str, Any]] = []
    calibrated = np.empty(windows.shape[0], dtype=np.float64)
    for repeat in sorted(original["repeat"].unique().tolist()):
        repeat_original = original.loc[original["repeat"].eq(repeat)]
        for fold in sorted(repeat_original["fold"].unique().tolist()):
            train = repeat_original.loc[~repeat_original["fold"].eq(fold)]
            calibrator = _fit_calibrator(
                train["probability"].to_numpy(dtype=np.float64),
                train["label"].to_numpy(dtype=np.int8),
                method,
            )
            selector = windows["repeat"].eq(repeat) & windows["fold"].eq(fold)
            calibrated[selector.to_numpy()] = calibrator.transform(
                windows.loc[selector, "probability"].to_numpy(dtype=np.float64)
            )
            records.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "method": method,
                    "coefficients": list(calibrator.coefficients),
                    "intercept": calibrator.intercept,
                    "training_originals": int(train.shape[0]),
                }
            )
    if not np.isfinite(calibrated).all():
        raise CalibrationError("cross-fitted calibration left missing window predictions")
    windows["probability"] = calibrated
    prediction = (calibrated >= 0.5).astype(np.int8)
    windows["prediction"] = prediction
    windows["predicted_class"] = prediction
    windows["experiment_id"] = f"PHASE8-CAL-{method.upper()}"
    windows["model_id"] = f"calibration:{method}"
    return windows, pd.DataFrame.from_records(records)


def crossfit_calibration(
    base: OOFPredictions,
    project: ProjectConfig,
    method: CalibrationMethod,
) -> CalibrationEvaluation:
    """Calibrate each fold using distinct originals from the same repeat only."""

    if method == "none":
        report = build_validation_report(base, project.validation)
        consensus = (
            base.original.groupby("original_id", observed=True, as_index=False)
            .agg(label=("label", "first"), probability=("probability", "mean"))
        )
        calibrator = _fit_calibrator(
            consensus["probability"].to_numpy(dtype=np.float64),
            consensus["label"].to_numpy(dtype=np.int8),
            "none",
        )
        return CalibrationEvaluation(
            method="none",
            oof=base,
            report=report,
            production_calibrator=calibrator,
            expected_calibration_error=expected_calibration_error(
                base.original["label"].to_numpy(dtype=np.int8),
                base.original["probability"].to_numpy(dtype=np.float64),
            ),
            fold_parameters=pd.DataFrame(
                columns=[
                    "repeat",
                    "fold",
                    "method",
                    "coefficients",
                    "intercept",
                    "training_originals",
                ]
            ),
        )
    windows, fold_parameters = _calibrated_windows(base, project, method)
    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    oof = build_oof_predictions(
        windows,
        folds,
        validation_window_fingerprint=base.validation_window_fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )
    report = build_validation_report(oof, project.validation)
    consensus = (
        base.original.groupby("original_id", observed=True, as_index=False)
        .agg(label=("label", "first"), probability=("probability", "mean"))
    )
    production = _fit_calibrator(
        consensus["probability"].to_numpy(dtype=np.float64),
        consensus["label"].to_numpy(dtype=np.int8),
        method,
    )
    return CalibrationEvaluation(
        method=method,
        oof=oof,
        report=report,
        production_calibrator=production,
        expected_calibration_error=expected_calibration_error(
            oof.original["label"].to_numpy(dtype=np.int8),
            oof.original["probability"].to_numpy(dtype=np.float64),
        ),
        fold_parameters=fold_parameters,
    )
