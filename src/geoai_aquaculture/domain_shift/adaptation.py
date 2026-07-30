"""Controlled Phase 7 label-model adaptations on immutable Phase 4 folds."""

from __future__ import annotations

import gc
import hashlib
import json
import resource
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from geoai_aquaculture.features import FeatureRegistry
from geoai_aquaculture.models import create_tabular_model_adapter
from geoai_aquaculture.training import TabularExperimentConfig
from geoai_aquaculture.training.tabular import PreparedTabularData, prepare_tabular_experiment_data
from geoai_aquaculture.validation import (
    OOFPredictions,
    ValidationReport,
    build_oof_predictions,
    build_validation_report,
    make_window_prediction_frame,
)

from .evaluation import DomainEvaluationError


class AdaptationError(ValueError):
    """Raised when a controlled adaptation violates the fixed validation contract."""


@dataclass(frozen=True, slots=True)
class AdaptationFoldResult:
    """One fold's bounded training metadata."""

    repeat: int
    fold: int
    seed: int
    train_original_count: int
    validation_original_count: int
    feature_count: int
    best_iteration: int
    validation_metric_value: float
    training_seconds: float
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    """Complete full-OOF result for one adaptation method and seed."""

    method: str
    seed: int
    experiment_id: str
    removed_features: tuple[str, ...]
    oof: OOFPredictions
    report: ValidationReport
    folds: tuple[AdaptationFoldResult, ...]
    feature_importance: pd.DataFrame
    runtime_seconds: float
    peak_rss_megabytes: float
    feature_count: int
    fingerprint: str


def _subset_prepared(
    prepared: PreparedTabularData,
    removed_features: tuple[str, ...],
) -> PreparedTabularData:
    if not removed_features:
        return prepared
    available = set(prepared.feature_names)
    missing = sorted(set(removed_features) - available)
    if missing:
        raise AdaptationError(f"domain-sensitive removal features are absent: {missing}")
    retained = tuple(name for name in prepared.feature_names if name not in set(removed_features))
    if len(retained) < 10:
        raise AdaptationError("feature removal left an unusably small model matrix")
    definitions = tuple(
        definition
        for definition in prepared.selected_registry.definitions
        if definition.name in set(retained)
    )
    registry = FeatureRegistry(definitions)
    digest = hashlib.sha256()
    digest.update(prepared.selected_feature_schema_fingerprint.encode())
    digest.update("\x1f".join(retained).encode())
    return replace(
        prepared,
        features=prepared.features.loc[:, list(retained)].copy(),
        feature_names=retained,
        selected_registry=registry,
        selected_feature_schema_fingerprint=digest.hexdigest(),
    )


def _weight_lookup(weights: pd.DataFrame | None) -> dict[str, float]:
    if weights is None:
        return {}
    required = {"original_id", "importance_weight"}
    if missing := sorted(required - set(weights.columns)):
        raise DomainEvaluationError(f"importance weight table is missing columns: {missing}")
    if weights["original_id"].duplicated().any():
        raise DomainEvaluationError("importance weights must contain one row per original")
    values = weights["importance_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise DomainEvaluationError("importance weights must be finite and positive")
    return dict(zip(weights["original_id"].astype(str), values, strict=True))


def _train_weights(ids: np.ndarray, mapping: dict[str, float]) -> np.ndarray:
    if not mapping:
        return np.ones(ids.shape[0], dtype=np.float64)
    missing = sorted(set(ids.astype(str).tolist()) - set(mapping))
    if missing:
        raise AdaptationError(f"importance weights do not cover training originals: {missing[:5]}")
    return np.asarray([mapping[str(value)] for value in ids], dtype=np.float64)


def run_label_adaptation(
    project: object,
    baseline: TabularExperimentConfig,
    *,
    method: str,
    seed: int,
    removed_features: tuple[str, ...] = (),
    importance_weights: pd.DataFrame | None = None,
) -> AdaptationResult:
    """Run one full 5x3 LightGBM adaptation without changing Phase 4 contracts."""

    if baseline.model.family != "lightgbm" or baseline.feature_set != "full":
        raise AdaptationError("Phase 7 adaptations require the accepted full LightGBM baseline")
    if method not in {"feature_removal", "importance_weighting"}:
        raise AdaptationError("unsupported controlled adaptation method")
    if method == "feature_removal" and not removed_features:
        raise AdaptationError("feature-removal adaptation requires an explicit feature list")
    if method == "importance_weighting" and importance_weights is None:
        raise AdaptationError("importance-weighting adaptation requires OOF domain weights")
    prepared = prepare_tabular_experiment_data(project, baseline)
    prepared = _subset_prepared(prepared, removed_features if method == "feature_removal" else ())
    lookup = _weight_lookup(importance_weights if method == "importance_weighting" else None)
    experiment_id = f"EXP-DOM-{method.upper().replace('_', '-')}-SEED-{seed}"
    started = perf_counter()
    fold_results: list[AdaptationFoldResult] = []
    window_predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for repeat in range(prepared.folds.n_repeats):
        manifest = prepared.windows.frame.loc[
            prepared.windows.frame["repeat"].eq(repeat)
        ].reset_index(drop=True)
        for fold in range(prepared.folds.n_splits):
            valid_selector = manifest["fold"].to_numpy(dtype=np.int16) == fold
            train_selector = ~valid_selector
            train_ids = manifest.loc[train_selector, "original_id"].astype("string").to_numpy()
            valid_ids = manifest.loc[valid_selector, "original_id"].astype("string").to_numpy()
            if set(train_ids.astype(str)) & set(valid_ids.astype(str)):
                raise AdaptationError("adaptation fold leaked an original row")
            train_labels = manifest.loc[train_selector, "label"].to_numpy(dtype=np.int8)
            valid_labels = manifest.loc[valid_selector, "label"].to_numpy(dtype=np.int8)
            fold_seed = seed + repeat * 10_007 + fold
            adapter = create_tabular_model_adapter(
                baseline.model,
                seed=fold_seed,
                cpu_threads=project.tabular.cpu_threads,
            )
            train_features = prepared.features.loc[train_selector]
            valid_features = prepared.features.loc[valid_selector]
            fit_started = perf_counter()
            fit = adapter.fit(
                train_features,
                train_labels,
                sample_weight=_train_weights(train_ids, lookup),
                validation_features=valid_features,
                validation_labels=valid_labels,
                validation_weight=np.ones(valid_labels.shape[0], dtype=np.float64),
                early_stopping_rounds=baseline.early_stopping_rounds,
            )
            training_seconds = perf_counter() - fit_started
            infer_started = perf_counter()
            probability = adapter.predict_proba(valid_features)
            inference_seconds = perf_counter() - infer_started
            valid_manifest = manifest.loc[valid_selector].reset_index(drop=True)
            window_predictions.append(
                make_window_prediction_frame(
                    valid_manifest,
                    probability,
                    experiment_id=experiment_id,
                    model_id=f"lightgbm:{method}",
                    fold_manifest_fingerprint=prepared.folds.fingerprint,
                    validation_window_fingerprint=prepared.windows.fingerprint,
                )
            )
            importance = adapter.get_feature_importance()
            importance.insert(0, "fold", fold)
            importance.insert(0, "repeat", repeat)
            importances.append(importance)
            fold_results.append(
                AdaptationFoldResult(
                    repeat=repeat,
                    fold=fold,
                    seed=fold_seed,
                    train_original_count=int(pd.Series(train_ids).nunique()),
                    validation_original_count=int(pd.Series(valid_ids).nunique()),
                    feature_count=len(prepared.feature_names),
                    best_iteration=fit.best_iteration,
                    validation_metric_value=fit.validation_metric_value,
                    training_seconds=training_seconds,
                    inference_seconds=inference_seconds,
                )
            )
            del adapter, train_features, valid_features
            gc.collect()
    windows = pd.concat(window_predictions, ignore_index=True).sort_values(
        ["repeat", "fold", "original_id", "window_id"],
        kind="stable",
        ignore_index=True,
    )
    oof = build_oof_predictions(
        windows,
        prepared.folds,
        validation_window_fingerprint=prepared.windows.fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )
    if oof.original.shape[0] != project.tabular.expected_full_oof_rows:
        raise AdaptationError("adaptation OOF does not contain the authoritative 5,463 rows")
    report = build_validation_report(oof, project.validation)
    importance = pd.concat(importances, ignore_index=True)
    digest = hashlib.sha256()
    digest.update(method.encode())
    digest.update(str(seed).encode())
    digest.update(oof.fingerprint.encode())
    digest.update("\x1f".join(removed_features).encode())
    return AdaptationResult(
        method=method,
        seed=seed,
        experiment_id=experiment_id,
        removed_features=removed_features,
        oof=oof,
        report=report,
        folds=tuple(fold_results),
        feature_importance=importance,
        runtime_seconds=perf_counter() - started,
        peak_rss_megabytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        feature_count=len(prepared.feature_names),
        fingerprint=digest.hexdigest(),
    )


def write_adaptation_result(root: Path, result: AdaptationResult) -> Path:
    """Persist one ignored controlled-adaptation result."""

    output = root / "adaptations" / result.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    result.oof.original.to_csv(output / "oof_predictions.csv", index=False)
    result.oof.windows.to_csv(output / "window_predictions.csv", index=False)
    result.report.repeat_metrics.to_csv(output / "repeat_metrics.csv", index=False)
    result.report.fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    result.report.slice_metrics.to_csv(output / "slice_metrics.csv", index=False)
    result.feature_importance.to_csv(output / "feature_importance.csv", index=False)
    pd.DataFrame([asdict(fold) for fold in result.folds]).to_csv(
        output / "training_folds.csv", index=False
    )
    payload = {
        "method": result.method,
        "seed": result.seed,
        "experiment_id": result.experiment_id,
        "feature_count": result.feature_count,
        "removed_features": list(result.removed_features),
        "runtime_seconds": result.runtime_seconds,
        "peak_rss_megabytes": result.peak_rss_megabytes,
        "fingerprint": result.fingerprint,
        "summary": result.report.summary,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
