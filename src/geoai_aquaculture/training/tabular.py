"""Leakage-safe staged fold and experiment runner for Phase 5 tabular models."""

from __future__ import annotations

import gc
import resource
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from geoai_aquaculture.data import (
    ProjectConfig,
    extract_test_mask_library,
    git_provenance,
    load_competition_data,
)
from geoai_aquaculture.features import (
    FeatureRegistry,
    SelectedFeatureMatrix,
    build_tabular_features,
    select_tabular_features,
)
from geoai_aquaculture.models import create_tabular_model_adapter
from geoai_aquaculture.validation import (
    ORIGINAL_OOF_COLUMNS,
    FoldManifest,
    OOFPredictions,
    ValidationReport,
    ValidationWindowManifest,
    aggregate_window_predictions,
    build_oof_predictions,
    build_validation_report,
    build_validation_windows,
    dataframe_fingerprint,
    load_fold_manifest,
    load_validation_window_manifest,
    make_window_prediction_frame,
)

from .artifacts import (
    ExperimentArtifactError,
    assert_resume_compatible,
    experiment_artifact_dir,
    prepare_experiment_artifact_dir,
    sha256_file,
    write_tabular_experiment_artifacts,
)
from .config import ExperimentStage, TabularExperimentConfig
from .results import FoldTrainingResult, TabularTrainingResult
from .weights import SampleWeightResult, build_window_sample_weights


class TabularTrainingError(ValueError):
    """Raised when staged training violates immutable validation or OOF contracts."""


@dataclass(frozen=True, slots=True)
class PreparedTabularData:
    """One rebuilt Phase 3 matrix aligned to immutable Phase 4 manifests."""

    features: pd.DataFrame
    feature_names: tuple[str, ...]
    selected_registry: FeatureRegistry
    full_feature_schema_fingerprint: str
    selected_feature_schema_fingerprint: str
    folds: FoldManifest
    windows: ValidationWindowManifest
    rows_per_repeat: int

    def __post_init__(self) -> None:
        if self.features.shape != (self.rows_per_repeat, len(self.feature_names)):
            raise TabularTrainingError("prepared feature rows or columns are misaligned")
        if self.selected_registry.feature_names != self.feature_names:
            raise TabularTrainingError("prepared feature registry is misaligned")
        expected_window_rows = self.rows_per_repeat * self.folds.n_repeats
        if self.windows.frame.shape[0] != expected_window_rows:
            raise TabularTrainingError("prepared validation manifest has incomplete repeats")


@dataclass(frozen=True, slots=True)
class FoldRunOutput:
    """Internal output from one current-fold fit and prediction."""

    metadata: FoldTrainingResult
    window_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    permutation_importance: pd.DataFrame


def validate_phase3_feature_contract(
    *,
    feature_count: int,
    schema_fingerprint: str,
    project: ProjectConfig,
) -> None:
    """Reject a rebuilt matrix that differs from the authoritative Phase 3 artifact."""

    runtime = project.tabular
    if feature_count != runtime.expected_feature_count:
        raise TabularTrainingError(
            f"Phase 3 feature count changed: {feature_count} != {runtime.expected_feature_count}"
        )
    if schema_fingerprint != runtime.feature_schema_fingerprint:
        raise TabularTrainingError(
            "Phase 3 feature-schema fingerprint does not match the authoritative artifact"
        )


def _manifest_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "original_id",
        "generation_mode",
        "view_index",
        "augmentation_seed",
        "window_start",
        "window_end",
        "window_length",
        "radar_availability",
        "optical_month_availability",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
        "mask_id",
    ]
    result = frame.loc[:, columns].copy()
    for column in result.select_dtypes(include=["category", "string"]).columns:
        result[column] = result[column].astype("string")
    return result.reset_index(drop=True)


def _validate_shared_repeat_panel(windows: ValidationWindowManifest, n_repeats: int) -> int:
    counts = windows.frame.groupby("repeat", observed=True).size()
    if set(counts.index.tolist()) != set(range(n_repeats)) or counts.nunique() != 1:
        raise TabularTrainingError("validation-window repeats do not share a complete panel")
    rows_per_repeat = int(counts.iloc[0])
    reference = _manifest_semantics(
        windows.frame.loc[windows.frame["repeat"].eq(0)].reset_index(drop=True)
    )
    for repeat in range(1, n_repeats):
        candidate = _manifest_semantics(
            windows.frame.loc[windows.frame["repeat"].eq(repeat)].reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
        except AssertionError as exc:
            raise TabularTrainingError(
                "fixed validation masks differ between repeats; numeric features cannot be reused"
            ) from exc
    return rows_per_repeat


def prepare_tabular_experiment_data(
    project: ProjectConfig,
    experiment: TabularExperimentConfig,
) -> PreparedTabularData:
    """Load artifacts, regenerate exact tensors, and build/select Phase 3 features once."""

    runtime = project.tabular
    validation_dir = runtime.validation_artifacts_dir
    folds = load_fold_manifest(
        validation_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=runtime.fold_manifest_fingerprint,
    )
    if folds.n_originals != runtime.expected_original_count:
        raise TabularTrainingError(
            "authoritative fold count changed: "
            f"{folds.n_originals} != {runtime.expected_original_count}"
        )
    if folds.n_repeats != runtime.expected_repeat_count:
        raise TabularTrainingError("authoritative fold repeat count changed")
    persisted_windows = load_validation_window_manifest(
        validation_dir / "validation_window_manifest.csv",
        folds,
        project.validation,
        expected_fingerprint=runtime.validation_window_fingerprint,
    )
    data = load_competition_data(project)
    masks = extract_test_mask_library(data)
    rebuilt = build_validation_windows(
        data,
        folds,
        project.validation,
        mask_library=masks,
        expected_fingerprint=runtime.validation_window_fingerprint,
        retain_datasets=True,
    )
    if rebuilt.manifest.fingerprint != persisted_windows.fingerprint:
        raise TabularTrainingError(
            "regenerated and persisted validation-window fingerprints differ"
        )
    rows_per_repeat = _validate_shared_repeat_panel(persisted_windows, folds.n_repeats)
    repeat_zero = rebuilt.for_repeat(0)
    matrix = build_tabular_features(repeat_zero, project.features)
    validate_phase3_feature_contract(
        feature_count=matrix.features.shape[1],
        schema_fingerprint=matrix.schema_fingerprint,
        project=project,
    )
    selected: SelectedFeatureMatrix = select_tabular_features(
        matrix,
        experiment.feature_set,
        project.features.bands,
    )
    if selected.full_schema_fingerprint != runtime.feature_schema_fingerprint:
        raise TabularTrainingError(
            "selected features are not derived from the approved full schema"
        )
    del data, masks, rebuilt, repeat_zero, matrix
    gc.collect()
    return PreparedTabularData(
        features=selected.features,
        feature_names=selected.feature_names,
        selected_registry=selected.registry,
        full_feature_schema_fingerprint=selected.full_schema_fingerprint,
        selected_feature_schema_fingerprint=selected.schema_fingerprint,
        folds=folds,
        windows=persisted_windows,
        rows_per_repeat=rows_per_repeat,
    )


def stage_repeat_folds(
    stage: ExperimentStage,
    *,
    n_repeats: int,
    n_splits: int,
) -> tuple[tuple[int, int], ...]:
    """Implement the fixed A/B/C compute policy without adaptive fold selection."""

    if stage == "smoke":
        return ((0, 0),)
    if stage == "screen":
        return tuple((0, fold) for fold in range(n_splits))
    if stage == "full":
        return tuple((repeat, fold) for repeat in range(n_repeats) for fold in range(n_splits))
    raise TabularTrainingError(f"unsupported experiment stage: {stage}")


def _validation_weight_policy(training_policy: str) -> str:
    if training_policy in {"equal_original", "equal_original_class_weighted"}:
        return "equal_original"
    return "uniform"


def _top_native_features(importance: pd.DataFrame, limit: int) -> tuple[str, ...]:
    if limit <= 0 or importance.empty:
        return ()
    ranked = (
        importance.groupby("feature", observed=True)["importance"]
        .sum()
        .sort_values(ascending=False)
    )
    return tuple(ranked.head(limit).index.astype(str))


def _bounded_permutation_importance(
    adapter: object,
    features: pd.DataFrame,
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    native_importance: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    seed: int,
    feature_limit: int,
) -> pd.DataFrame:
    """Permute a bounded native top set on one held-out fold as a diagnostic only."""

    names = _top_native_features(native_importance, feature_limit)
    if not names:
        return pd.DataFrame(
            columns=(
                "repeat",
                "fold",
                "feature",
                "baseline_window_log_loss",
                "permuted_window_log_loss",
                "log_loss_increase",
            )
        )
    baseline = float(log_loss(labels, baseline_probabilities, labels=[0, 1]))
    rng = np.random.default_rng(seed)
    mutable = features.copy()
    records: list[dict[str, float | int | str]] = []
    for name in names:
        original = mutable[name].to_numpy(copy=True)
        mutable[name] = rng.permutation(original)
        probability = adapter.predict_proba(mutable)
        permuted = float(log_loss(labels, probability, labels=[0, 1]))
        records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "feature": name,
                "baseline_window_log_loss": baseline,
                "permuted_window_log_loss": permuted,
                "log_loss_increase": permuted - baseline,
            }
        )
        mutable[name] = original
    return pd.DataFrame.from_records(records)


def run_tabular_fold(
    prepared: PreparedTabularData,
    project: ProjectConfig,
    experiment: TabularExperimentConfig,
    *,
    stage: ExperimentStage,
    repeat: int,
    fold: int,
    output_dir: Path,
    permutation_enabled: bool,
) -> FoldRunOutput:
    """Fit exactly one fold with fold-local weights and current-fold early stopping."""

    repeat_manifest = prepared.windows.frame.loc[
        prepared.windows.frame["repeat"].eq(repeat)
    ].reset_index(drop=True)
    if repeat_manifest.shape[0] != prepared.rows_per_repeat:
        raise TabularTrainingError("repeat manifest and reusable feature matrix are misaligned")
    valid_selector = repeat_manifest["fold"].to_numpy(dtype=np.int16) == fold
    train_selector = ~valid_selector
    if not valid_selector.any() or not train_selector.any():
        raise TabularTrainingError("fold selectors must contain training and validation windows")
    train_ids = repeat_manifest.loc[train_selector, "original_id"].astype("string").to_numpy()
    valid_ids = repeat_manifest.loc[valid_selector, "original_id"].astype("string").to_numpy()
    if set(train_ids.astype(str)) & set(valid_ids.astype(str)):
        raise TabularTrainingError("original-row overlap detected before model fitting")
    train_labels = repeat_manifest.loc[train_selector, "label"].to_numpy(dtype=np.int8)
    valid_labels = repeat_manifest.loc[valid_selector, "label"].to_numpy(dtype=np.int8)
    train_weights: SampleWeightResult = build_window_sample_weights(
        train_ids,
        train_labels,
        experiment.weighting,
    )
    validation_weights = build_window_sample_weights(
        valid_ids,
        valid_labels,
        _validation_weight_policy(experiment.weighting),
    )
    train_features = prepared.features.loc[train_selector]
    valid_features = prepared.features.loc[valid_selector]
    fold_seed = experiment.seed + repeat * 10_007 + fold
    iteration_limit = experiment.smoke_iteration_limit if stage == "smoke" else None
    adapter = create_tabular_model_adapter(
        experiment.model,
        seed=fold_seed,
        cpu_threads=project.tabular.cpu_threads,
        iteration_limit=iteration_limit,
    )
    train_started = perf_counter()
    fit_metadata = adapter.fit(
        train_features,
        train_labels,
        sample_weight=train_weights.values,
        validation_features=valid_features,
        validation_labels=valid_labels,
        validation_weight=validation_weights.values,
        early_stopping_rounds=min(
            experiment.early_stopping_rounds,
            max(1, (iteration_limit or fit_metadata_limit(experiment)) // 2),
        ),
    )
    training_seconds = perf_counter() - train_started
    inference_started = perf_counter()
    probabilities = adapter.predict_proba(valid_features)
    inference_seconds = perf_counter() - inference_started
    model_name = f"repeat_{repeat:02d}_fold_{fold:02d}{adapter.model_suffix}"
    model_path = adapter.save(output_dir / "models" / model_name)
    importance = adapter.get_feature_importance()
    importance.insert(0, "fold", fold)
    importance.insert(0, "repeat", repeat)
    permutation = (
        _bounded_permutation_importance(
            adapter,
            valid_features,
            valid_labels,
            probabilities,
            importance,
            repeat=repeat,
            fold=fold,
            seed=fold_seed + 1_000_003,
            feature_limit=experiment.permutation_feature_count,
        )
        if permutation_enabled
        else pd.DataFrame(
            columns=(
                "repeat",
                "fold",
                "feature",
                "baseline_window_log_loss",
                "permuted_window_log_loss",
                "log_loss_increase",
            )
        )
    )
    valid_manifest = repeat_manifest.loc[valid_selector].reset_index(drop=True)
    predictions = make_window_prediction_frame(
        valid_manifest,
        probabilities,
        experiment_id=experiment.experiment_id,
        model_id=f"{experiment.model.family}:{experiment.model.name}",
        fold_manifest_fingerprint=prepared.folds.fingerprint,
        validation_window_fingerprint=prepared.windows.fingerprint,
    )
    train_original = repeat_manifest.loc[train_selector, ["original_id", "label"]].drop_duplicates(
        "original_id"
    )
    valid_original = repeat_manifest.loc[valid_selector, ["original_id", "label"]].drop_duplicates(
        "original_id"
    )
    metadata = FoldTrainingResult(
        repeat=repeat,
        fold=fold,
        seed=fold_seed,
        train_original_count=train_original.shape[0],
        validation_original_count=valid_original.shape[0],
        train_window_count=int(train_selector.sum()),
        validation_window_count=int(valid_selector.sum()),
        train_positive_rate=float(train_original["label"].mean()),
        validation_positive_rate=float(valid_original["label"].mean()),
        feature_count=len(prepared.feature_names),
        weighting_policy=experiment.weighting,
        class_weight_zero=float(train_weights.class_weights[0]),
        class_weight_one=float(train_weights.class_weights[1]),
        original_weight_min=train_weights.original_weight_min,
        original_weight_max=train_weights.original_weight_max,
        best_iteration=fit_metadata.best_iteration,
        fitted_iterations=fit_metadata.fitted_iterations,
        validation_metric_name=fit_metadata.validation_metric_name,
        validation_metric_value=fit_metadata.validation_metric_value,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        model_path=model_path.relative_to(output_dir).as_posix(),
        model_sha256=sha256_file(model_path),
    )
    del adapter, train_features, valid_features
    gc.collect()
    return FoldRunOutput(metadata, predictions, importance, permutation)


def fit_metadata_limit(experiment: TabularExperimentConfig) -> int:
    """Return the declared maximum iteration count for one model family."""

    key = "iterations" if experiment.model.family == "catboost" else "n_estimators"
    return int(experiment.model.parameters[key])


def _partial_oof(
    window_predictions: pd.DataFrame,
    project: ProjectConfig,
    prepared: PreparedTabularData,
) -> OOFPredictions:
    original = aggregate_window_predictions(
        window_predictions,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )
    fingerprint = dataframe_fingerprint(original, columns=ORIGINAL_OOF_COLUMNS)
    return OOFPredictions(
        original=original,
        windows=window_predictions,
        fold_manifest_fingerprint=prepared.folds.fingerprint,
        validation_window_fingerprint=prepared.windows.fingerprint,
        aggregation_method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
        fingerprint=fingerprint,
    )


def validate_full_oof_contract(
    oof: OOFPredictions,
    project: ProjectConfig,
) -> None:
    """Require exactly one authoritative OOF row per original and repeat."""

    expected = project.tabular.expected_full_oof_rows
    if oof.original.shape[0] != expected:
        raise TabularTrainingError(
            f"full Stage C OOF must contain exactly {expected} rows; found {oof.original.shape[0]}"
        )
    if oof.original.duplicated(["original_id", "repeat"]).any():
        raise TabularTrainingError("full Stage C OOF contains duplicate original/repeat rows")
    if oof.original["original_id"].nunique() != project.tabular.expected_original_count:
        raise TabularTrainingError("full Stage C OOF original coverage changed")
    if set(oof.original["repeat"].unique().tolist()) != set(
        range(project.tabular.expected_repeat_count)
    ):
        raise TabularTrainingError("full Stage C OOF repeat coverage changed")


def run_tabular_experiment(
    prepared: PreparedTabularData,
    project: ProjectConfig,
    experiment: TabularExperimentConfig,
    *,
    stage: ExperimentStage,
    output_dir: Path,
) -> TabularTrainingResult:
    """Execute fixed folds, aggregate original OOF, and compute Phase 4 metrics."""

    experiment.require_stage(stage)
    started = perf_counter()
    outputs: list[FoldRunOutput] = []
    splits = stage_repeat_folds(
        stage,
        n_repeats=prepared.folds.n_repeats,
        n_splits=prepared.folds.n_splits,
    )
    for split_index, (repeat, fold) in enumerate(splits):
        outputs.append(
            run_tabular_fold(
                prepared,
                project,
                experiment,
                stage=stage,
                repeat=repeat,
                fold=fold,
                output_dir=output_dir,
                permutation_enabled=(stage == "full" and split_index == 0),
            )
        )
    windows = pd.concat(
        [output.window_predictions for output in outputs], ignore_index=True
    ).sort_values(["repeat", "fold", "original_id", "window_id"], kind="stable", ignore_index=True)
    if stage == "full":
        oof = build_oof_predictions(
            windows,
            prepared.folds,
            validation_window_fingerprint=prepared.windows.fingerprint,
            method=project.validation.aggregation_method,
            trimmed_fraction=project.validation.trimmed_mean_fraction,
        )
        validate_full_oof_contract(oof, project)
    else:
        oof = _partial_oof(windows, project, prepared)
    report: ValidationReport = build_validation_report(oof, project.validation)
    importance = pd.concat([output.feature_importance for output in outputs], ignore_index=True)
    nonempty_permutations = [
        output.permutation_importance
        for output in outputs
        if not output.permutation_importance.empty
    ]
    permutations = (
        pd.concat(nonempty_permutations, ignore_index=True)
        if nonempty_permutations
        else outputs[0].permutation_importance.copy()
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return TabularTrainingResult(
        stage=stage,
        oof=oof,
        report=report,
        folds=tuple(output.metadata for output in outputs),
        feature_importance=importance,
        permutation_importance=permutations,
        feature_names=prepared.feature_names,
        selected_registry=prepared.selected_registry,
        full_feature_schema_fingerprint=prepared.full_feature_schema_fingerprint,
        selected_feature_schema_fingerprint=prepared.selected_feature_schema_fingerprint,
        runtime_seconds=perf_counter() - started,
        peak_rss_megabytes=peak_rss,
        artifact_dir=output_dir,
    )


def execute_tabular_experiment(
    project: ProjectConfig,
    experiment: TabularExperimentConfig,
    *,
    stage: ExperimentStage,
    resume: bool = False,
    overwrite: bool = False,
) -> TabularTrainingResult | None:
    """Prepare, compatibility-check, run, and persist one declared experiment."""

    experiment.require_stage(stage)
    root = project.tabular.experiments_artifacts_dir
    output = experiment_artifact_dir(root, experiment.experiment_id, stage)
    if output.exists() and any(output.iterdir()) and not resume and not overwrite:
        raise ExperimentArtifactError(
            f"experiment artifacts already exist at {output}; pass --resume or --overwrite"
        )
    prepared = prepare_tabular_experiment_data(project, experiment)
    provenance = git_provenance(project.project_root)
    base_sha = sha256_file(project.source_path)
    if resume:
        assert_resume_compatible(
            output,
            stage=stage,
            experiment=experiment,
            base_config_sha256=base_sha,
            provenance=provenance,
            fold_manifest_fingerprint=prepared.folds.fingerprint,
            validation_window_fingerprint=prepared.windows.fingerprint,
            full_feature_schema_fingerprint=prepared.full_feature_schema_fingerprint,
            selected_feature_schema_fingerprint=prepared.selected_feature_schema_fingerprint,
        )
        return None
    output = prepare_experiment_artifact_dir(
        root,
        experiment.experiment_id,
        stage,
        overwrite=overwrite,
    )
    result = run_tabular_experiment(
        prepared,
        project,
        experiment,
        stage=stage,
        output_dir=output,
    )
    write_tabular_experiment_artifacts(
        output,
        project=project,
        experiment=experiment,
        result=result,
    )
    return result
