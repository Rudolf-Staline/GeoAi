"""Deterministic CatBoost and LightGBM adapters behind one Phase 5 contract."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from geoai_aquaculture.training.config import ModelProfile


class ModelAdapterError(ValueError):
    """Raised when a tabular estimator violates the common probability contract."""


@dataclass(frozen=True, slots=True)
class ModelFitMetadata:
    """Fold-local early-stopping outcome normalized across model families."""

    best_iteration: int
    fitted_iterations: int
    validation_metric_name: str
    validation_metric_value: float
    seed: int

    def __post_init__(self) -> None:
        if self.best_iteration < 1 or self.fitted_iterations < 1:
            raise ModelAdapterError("fitted iteration metadata must be positive")
        if not self.validation_metric_name or not np.isfinite(self.validation_metric_value):
            raise ModelAdapterError("validation metric metadata must be finite and named")


def _validate_features(features: pd.DataFrame, expected: tuple[str, ...] | None = None) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ModelAdapterError("model features must be a non-empty pandas DataFrame")
    if expected is not None and tuple(features.columns) != expected:
        raise ModelAdapterError("prediction feature schema does not match fitted schema")
    if np.isinf(features.to_numpy(dtype=np.float64)).any():
        raise ModelAdapterError("model features cannot contain infinity")


def _validate_labels(labels: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8)
    if values.shape != (rows,) or set(np.unique(values).tolist()) != {0, 1}:
        raise ModelAdapterError("fit labels must align and contain both binary classes")
    return values


def _validate_weights(weights: np.ndarray | None, rows: int, name: str) -> np.ndarray | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ModelAdapterError(f"{name} must be aligned, finite, and strictly positive")
    return values


def _bounded_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ModelAdapterError("predicted probabilities must be finite and one-dimensional")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ModelAdapterError("predicted probabilities must lie within [0, 1]")
    return values


class TabularModelAdapter(ABC):
    """Common fit, probability, serialization, and importance contract."""

    family: str
    model_suffix: str

    def __init__(self, parameters: dict[str, Any], *, seed: int, cpu_threads: int = 1) -> None:
        if cpu_threads < 1:
            raise ModelAdapterError("cpu_threads must be positive")
        self.parameters = MappingProxyType(dict(parameters))
        self.seed = seed
        self.cpu_threads = cpu_threads
        self.feature_names: tuple[str, ...] = ()
        self.fit_metadata: ModelFitMetadata | None = None

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        validation_features: pd.DataFrame,
        validation_labels: np.ndarray,
        validation_weight: np.ndarray | None,
        early_stopping_rounds: int,
    ) -> ModelFitMetadata:
        """Fit on one fold and evaluate only on that fold's validation rows."""

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities within [0, 1]."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist one ignored fold model plus deterministic adapter metadata."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> TabularModelAdapter:
        """Restore an adapter saved through the common contract."""

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Return native importance types in long format."""

    def _metadata_payload(self) -> dict[str, Any]:
        if self.fit_metadata is None or not self.feature_names:
            raise ModelAdapterError("cannot save an unfitted adapter")
        return {
            "schema_version": 1,
            "family": self.family,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "cpu_threads": self.cpu_threads,
            "feature_names": list(self.feature_names),
            "fit_metadata": {
                "best_iteration": self.fit_metadata.best_iteration,
                "fitted_iterations": self.fit_metadata.fitted_iterations,
                "validation_metric_name": self.fit_metadata.validation_metric_name,
                "validation_metric_value": self.fit_metadata.validation_metric_value,
                "seed": self.fit_metadata.seed,
            },
        }

    @staticmethod
    def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"saved adapter or metadata not found: {path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _restore_common_metadata(self, payload: dict[str, Any]) -> None:
        self.feature_names = tuple(payload["feature_names"])
        metadata = payload["fit_metadata"]
        self.fit_metadata = ModelFitMetadata(
            best_iteration=int(metadata["best_iteration"]),
            fitted_iterations=int(metadata["fitted_iterations"]),
            validation_metric_name=str(metadata["validation_metric_name"]),
            validation_metric_value=float(metadata["validation_metric_value"]),
            seed=int(metadata["seed"]),
        )


class CatBoostAdapter(TabularModelAdapter):
    """CatBoost with native missing values and deterministic CPU settings."""

    family = "catboost"
    model_suffix = ".cbm"

    def __init__(self, parameters: dict[str, Any], *, seed: int, cpu_threads: int = 1) -> None:
        super().__init__(parameters, seed=seed, cpu_threads=cpu_threads)
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - dependency acceptance catches this
            raise ModelAdapterError("CatBoost is required for Phase 5") from exc
        params = {
            **dict(parameters),
            "random_seed": seed,
            "thread_count": cpu_threads,
            "allow_writing_files": False,
            "verbose": False,
        }
        self._model = CatBoostClassifier(**params)

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        validation_features: pd.DataFrame,
        validation_labels: np.ndarray,
        validation_weight: np.ndarray | None,
        early_stopping_rounds: int,
    ) -> ModelFitMetadata:
        from catboost import Pool

        _validate_features(features)
        _validate_features(validation_features, tuple(features.columns))
        y_train = _validate_labels(labels, features.shape[0])
        y_valid = _validate_labels(validation_labels, validation_features.shape[0])
        train_weight = _validate_weights(sample_weight, features.shape[0], "training weights")
        valid_weight = _validate_weights(
            validation_weight, validation_features.shape[0], "validation weights"
        )
        self.feature_names = tuple(features.columns)
        validation_pool = Pool(
            validation_features,
            y_valid,
            weight=valid_weight,
            feature_names=list(self.feature_names),
        )
        self._model.fit(
            features,
            y_train,
            sample_weight=train_weight,
            eval_set=validation_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        best_zero_based = int(self._model.get_best_iteration())
        fitted_iterations = int(self._model.tree_count_)
        best_iteration = best_zero_based + 1 if best_zero_based >= 0 else fitted_iterations
        score = self._model.get_best_score().get("validation", {})
        if not score:
            raise ModelAdapterError("CatBoost did not expose a validation metric")
        metric_name, metric_value = next(iter(score.items()))
        self.fit_metadata = ModelFitMetadata(
            best_iteration=best_iteration,
            fitted_iterations=fitted_iterations,
            validation_metric_name=str(metric_name),
            validation_metric_value=float(metric_value),
            seed=self.seed,
        )
        return self.fit_metadata

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        _validate_features(features, self.feature_names)
        matrix = np.asarray(self._model.predict_proba(features), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape != (features.shape[0], 2):
            raise ModelAdapterError("CatBoost predict_proba must return two columns")
        return _bounded_probabilities(matrix[:, 1])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(destination.as_posix())
        self._write_metadata(destination, self._metadata_payload())
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CatBoostAdapter:
        destination = Path(path)
        payload = cls._read_metadata(destination)
        if payload.get("family") != cls.family:
            raise ModelAdapterError("saved model family is not CatBoost")
        adapter = cls(
            dict(payload["parameters"]),
            seed=int(payload["seed"]),
            cpu_threads=int(payload.get("cpu_threads", 1)),
        )
        adapter._model.load_model(destination.as_posix())
        adapter._restore_common_metadata(payload)
        return adapter

    def get_feature_importance(self) -> pd.DataFrame:
        if self.fit_metadata is None:
            raise ModelAdapterError("feature importance requires a fitted CatBoost model")
        values = np.asarray(self._model.get_feature_importance(), dtype=np.float64)
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_type": "prediction_values_change",
                "importance": values,
            }
        )


class LightGBMAdapter(TabularModelAdapter):
    """LightGBM with deterministic CPU settings and native missing values."""

    family = "lightgbm"
    model_suffix = ".txt"

    def __init__(self, parameters: dict[str, Any], *, seed: int, cpu_threads: int = 1) -> None:
        super().__init__(parameters, seed=seed, cpu_threads=cpu_threads)
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - dependency acceptance catches this
            raise ModelAdapterError("LightGBM is required for Phase 5") from exc
        params = {
            **dict(parameters),
            "random_state": seed,
            "n_jobs": cpu_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        self._model = LGBMClassifier(**params)
        self._booster = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        validation_features: pd.DataFrame,
        validation_labels: np.ndarray,
        validation_weight: np.ndarray | None,
        early_stopping_rounds: int,
    ) -> ModelFitMetadata:
        from lightgbm import early_stopping, log_evaluation

        _validate_features(features)
        _validate_features(validation_features, tuple(features.columns))
        y_train = _validate_labels(labels, features.shape[0])
        y_valid = _validate_labels(validation_labels, validation_features.shape[0])
        train_weight = _validate_weights(sample_weight, features.shape[0], "training weights")
        valid_weight = _validate_weights(
            validation_weight, validation_features.shape[0], "validation weights"
        )
        self.feature_names = tuple(features.columns)
        self._model.fit(
            features,
            y_train,
            sample_weight=train_weight,
            eval_sample_weight=[valid_weight] if valid_weight is not None else None,
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
            eval_X=validation_features,
            eval_y=y_valid,
        )
        self._booster = self._model.booster_
        best_iteration = int(self._model.best_iteration_ or self._booster.current_iteration())
        fitted_iterations = int(self._booster.current_iteration())
        score = self._model.best_score_.get("valid_0", {})
        if not score:
            raise ModelAdapterError("LightGBM did not expose a validation metric")
        metric_name, metric_value = next(iter(score.items()))
        self.fit_metadata = ModelFitMetadata(
            best_iteration=best_iteration,
            fitted_iterations=fitted_iterations,
            validation_metric_name=str(metric_name),
            validation_metric_value=float(metric_value),
            seed=self.seed,
        )
        return self.fit_metadata

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        _validate_features(features, self.feature_names)
        if self._booster is not None and not hasattr(self._model, "classes_"):
            values = np.asarray(
                self._booster.predict(features, num_iteration=self.fit_metadata.best_iteration),
                dtype=np.float64,
            )
            return _bounded_probabilities(values)
        matrix = np.asarray(
            self._model.predict_proba(
                features,
                num_iteration=self.fit_metadata.best_iteration if self.fit_metadata else None,
            ),
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape != (features.shape[0], 2):
            raise ModelAdapterError("LightGBM predict_proba must return two columns")
        return _bounded_probabilities(matrix[:, 1])

    def save(self, path: str | Path) -> Path:
        if self._booster is None:
            raise ModelAdapterError("cannot save an unfitted LightGBM model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(
            destination.as_posix(),
            num_iteration=self.fit_metadata.best_iteration if self.fit_metadata else None,
        )
        self._write_metadata(destination, self._metadata_payload())
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LightGBMAdapter:
        from lightgbm import Booster

        destination = Path(path)
        payload = cls._read_metadata(destination)
        if payload.get("family") != cls.family:
            raise ModelAdapterError("saved model family is not LightGBM")
        adapter = cls(
            dict(payload["parameters"]),
            seed=int(payload["seed"]),
            cpu_threads=int(payload.get("cpu_threads", 1)),
        )
        adapter._booster = Booster(model_file=destination.as_posix())
        adapter._restore_common_metadata(payload)
        return adapter

    def get_feature_importance(self) -> pd.DataFrame:
        if self._booster is None or self.fit_metadata is None:
            raise ModelAdapterError("feature importance requires a fitted LightGBM model")
        records: list[dict[str, str | float]] = []
        for importance_type in ("gain", "split"):
            values = self._booster.feature_importance(importance_type=importance_type)
            records.extend(
                {
                    "feature": feature,
                    "importance_type": importance_type,
                    "importance": float(value),
                }
                for feature, value in zip(self.feature_names, values, strict=True)
            )
        return pd.DataFrame.from_records(records)


def create_tabular_model_adapter(
    profile: ModelProfile,
    *,
    seed: int,
    cpu_threads: int = 1,
    iteration_limit: int | None = None,
) -> TabularModelAdapter:
    """Instantiate one family without allowing the runner to mutate other parameters."""

    parameters = dict(profile.parameters)
    if iteration_limit is not None:
        key = "iterations" if profile.family == "catboost" else "n_estimators"
        parameters[key] = min(int(parameters[key]), iteration_limit)
    if profile.family == "catboost":
        return CatBoostAdapter(parameters, seed=seed, cpu_threads=cpu_threads)
    if profile.family == "lightgbm":
        return LightGBMAdapter(parameters, seed=seed, cpu_threads=cpu_threads)
    raise ModelAdapterError(f"unsupported model family: {profile.family}")
