from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from geoai_aquaculture.data import (
    TabularRuntimeConfig,
    git_provenance,
    load_project_config,
)
from geoai_aquaculture.features import FeatureDefinition, FeatureRegistry
from geoai_aquaculture.models import CatBoostAdapter, LightGBMAdapter
from geoai_aquaculture.training import (
    AcceptedCandidateRegistry,
    CandidateRecord,
    ExperimentArtifactError,
    ExperimentConfigError,
    PreparedTabularData,
    TabularTrainingError,
    analyze_oof_diversity,
    assert_resume_compatible,
    build_window_sample_weights,
    load_accepted_candidate_registry,
    load_experiment_artifact_manifest,
    load_tabular_experiment_config,
    run_tabular_experiment,
    stage_repeat_folds,
    validate_full_oof_contract,
    validate_phase3_feature_contract,
    write_tabular_experiment_artifacts,
)
from geoai_aquaculture.training.artifacts import sha256_file
from geoai_aquaculture.validation import (
    FoldManifest,
    ValidationWindowManifest,
    build_oof_predictions,
    build_repeated_fold_manifest,
    build_validation_report,
    dataframe_fingerprint,
    json_fingerprint,
    make_window_prediction_frame,
)


def _tiny_xy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(91)
    features = pd.DataFrame(
        rng.normal(size=(80, 6)),
        columns=[f"feature_{index}" for index in range(6)],
    )
    features.loc[::9, "feature_4"] = np.nan
    labels = (features["feature_0"].fillna(0.0) + 0.3 * features["feature_1"] > 0).astype(np.int8)
    return features, labels.to_numpy()


@pytest.mark.parametrize(
    ("adapter_class", "parameters", "suffix"),
    (
        (
            CatBoostAdapter,
            {
                "iterations": 18,
                "learning_rate": 0.1,
                "depth": 3,
                "l2_leaf_reg": 3.0,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
            },
            ".cbm",
        ),
        (
            LightGBMAdapter,
            {
                "objective": "binary",
                "n_estimators": 18,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 5,
            },
            ".txt",
        ),
    ),
)
def test_tree_adapters_are_deterministic_bounded_and_round_trip(
    tmp_path: Path,
    adapter_class,
    parameters: dict[str, object],
    suffix: str,
) -> None:
    features, labels = _tiny_xy()
    train = features.iloc[:60]
    valid = features.iloc[60:]
    train_labels = labels[:60]
    valid_labels = labels[60:]

    first = adapter_class(parameters, seed=37, cpu_threads=2)
    second = adapter_class(parameters, seed=37, cpu_threads=2)
    metadata = first.fit(
        train,
        train_labels,
        sample_weight=np.ones(60),
        validation_features=valid,
        validation_labels=valid_labels,
        validation_weight=np.ones(20),
        early_stopping_rounds=4,
    )
    second.fit(
        train,
        train_labels,
        sample_weight=np.ones(60),
        validation_features=valid,
        validation_labels=valid_labels,
        validation_weight=np.ones(20),
        early_stopping_rounds=4,
    )
    probability = first.predict_proba(valid)
    np.testing.assert_allclose(probability, second.predict_proba(valid), rtol=0.0, atol=0.0)
    assert np.isfinite(probability).all() and ((probability >= 0.0) & (probability <= 1.0)).all()
    assert metadata.best_iteration >= 1
    importance = first.get_feature_importance()
    assert set(importance["feature"]) == set(features.columns)

    path = tmp_path / f"model{suffix}"
    first.save(path)
    restored = adapter_class.load(path)
    assert restored.cpu_threads == 2
    np.testing.assert_allclose(probability, restored.predict_proba(valid), rtol=0.0, atol=0.0)


def test_window_weighting_is_original_balanced_and_fold_local() -> None:
    ids = np.asarray(["a", "a", "b", "c", "c", "c", "d", "d"])
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    equal = build_window_sample_weights(ids, labels, "equal_original")
    totals = pd.Series(equal.values).groupby(ids).sum()
    np.testing.assert_allclose(totals.to_numpy(), np.ones(4))
    uniform = build_window_sample_weights(ids, labels, "uniform")
    assert np.all(uniform.values == 1.0)
    class_weighted = build_window_sample_weights(ids, labels, "equal_original_class_weighted")
    assert class_weighted.class_weights == {0: 1.0, 1: 1.0}

    imbalanced_ids = np.asarray(["a", "a", "b", "c", "d", "e"])
    imbalanced_labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int8)
    weighted = build_window_sample_weights(
        imbalanced_ids, imbalanced_labels, "equal_original_class_weighted"
    )
    assert weighted.class_weights[1] > weighted.class_weights[0]
    weighted_totals = pd.Series(weighted.values).groupby(imbalanced_ids).sum()
    assert weighted_totals["d"] > weighted_totals["a"]


def _base_config(path: Path, *, originals: int = 12, repeats: int = 1) -> Path:
    rows = originals * repeats
    path.write_text(
        f"""\
project:
  name: phase5-synthetic
  seed: 11
data:
  train_path: Train.csv
  test_path: Test.csv
  sample_submission_path: SampleSubmission.csv
  id_column: ID
  target_column: label
  missing_sentinel: -9999.0
  months: 12
  radar_bands: [VH, VV]
  optical_bands: [blue, green, nir, nira, re1, re2, re3, red, swir1, swir2]
validation:
  seed: 19
  n_splits: 2
  n_repeats: {repeats}
  threshold: 0.5
  sampled_windows_per_original: 3
  validation_window_seed: 23
  aggregation_method: mean
augmentation:
  enabled: true
  use_test_missingness_masks: true
  windows_per_sample: 3
features:
  version: phase3_v1
tabular:
  validation_artifacts_dir: artifacts/validation
  experiments_artifacts_dir: artifacts/experiments
  fold_manifest_fingerprint: {"a" * 64}
  validation_window_fingerprint: {"b" * 64}
  feature_schema_fingerprint: {"c" * 64}
  expected_original_count: {originals}
  expected_repeat_count: {repeats}
  expected_full_oof_rows: {rows}
  expected_feature_count: 5
  permutation_feature_count: 2
reporting:
  artifacts_dir: artifacts
""",
        encoding="utf-8",
    )
    return path


def _experiment_config(
    path: Path,
    *,
    experiment_id: str,
    family: str,
) -> Path:
    if family == "catboost":
        parameters = """\
      iterations: 20
      learning_rate: 0.1
      depth: 3
      l2_leaf_reg: 3.0
      loss_function: Logloss
      eval_metric: Logloss
"""
    else:
        parameters = """\
      objective: binary
      n_estimators: 20
      learning_rate: 0.1
      num_leaves: 7
      min_child_samples: 3
"""
    path.write_text(
        f"""\
experiment:
  id: {experiment_id}
  hypothesis: Synthetic end-to-end contract verification.
  feature_set: full
  weighting: equal_original
  seed: 29
  early_stopping_rounds: 4
  smoke_iteration_limit: 8
  permutation_feature_count: 2
  allowed_stages: [smoke, screen, full]
  threshold: 0.5
  notes: Tiny deterministic fixture.
  model:
    family: {family}
    name: tiny_{family}
    hypothesis: Tiny trees verify the adapter without parameter search.
    parameters:
{parameters}""",
        encoding="utf-8",
    )
    return path


def test_experiment_config_rejects_threshold_and_stage_promotion(tmp_path: Path) -> None:
    path = _experiment_config(
        tmp_path / "experiment.yaml", experiment_id="TEST-CB", family="catboost"
    )
    experiment = load_tabular_experiment_config(path)
    assert experiment.model.family == "catboost"
    assert stage_repeat_folds("smoke", n_repeats=3, n_splits=5) == ((0, 0),)
    assert len(stage_repeat_folds("screen", n_repeats=3, n_splits=5)) == 5
    assert len(stage_repeat_folds("full", n_repeats=3, n_splits=5)) == 15
    restricted = path.read_text().replace(
        "allowed_stages: [smoke, screen, full]", "allowed_stages: [smoke, screen]"
    )
    path.write_text(restricted, encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="not approved"):
        load_tabular_experiment_config(path).require_stage("full")
    path.write_text(restricted.replace("threshold: 0.5", "threshold: 0.49"), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="exactly 0.5"):
        load_tabular_experiment_config(path)


def _synthetic_prepared(tmp_path: Path):
    project = load_project_config(_base_config(tmp_path / "base.yaml"))
    train = pd.DataFrame(
        {
            "ID": pd.Series([f"original-{index:02d}" for index in range(12)], dtype="string"),
            "label": np.asarray([index % 2 for index in range(12)], dtype=np.int8),
        }
    )
    folds: FoldManifest = build_repeated_fold_manifest(train, project.validation)
    templates = ((4, 1, 0), (5, 4, 1), (6, 7, 2))
    records: list[dict[str, object]] = []
    feature_rows: list[list[float]] = []
    for row in folds.frame.itertuples(index=False):
        for view, (length, start, gaps) in enumerate(templates):
            records.append(
                {
                    "repeat": int(row.repeat),
                    "window_id": f"r{row.repeat}-{row.original_id}-v{view}",
                    "original_id": str(row.original_id),
                    "fold": int(row.fold),
                    "label": int(row.label),
                    "generation_mode": "sampled",
                    "view_index": view,
                    "augmentation_seed": 23,
                    "window_start": start,
                    "window_end": start + length - 1,
                    "window_length": length,
                    "radar_availability": "1" * length + "0" * (6 - length),
                    "optical_month_availability": (
                        "1" * (length - gaps) + "0" * (6 - length + gaps)
                    ),
                    "radar_months": length,
                    "optical_months": length - gaps,
                    "internal_optical_gap_count": gaps,
                    "mask_id": f"mask-{view}",
                }
            )
            label = float(row.label)
            feature_rows.append(
                [
                    label + view * 0.03,
                    (1.0 - label) + view * 0.02,
                    float(length),
                    float(start),
                    float(gaps),
                ]
            )
    frame = pd.DataFrame.from_records(records)
    content = dataframe_fingerprint(frame)
    window_fingerprint = json_fingerprint(
        {
            "content": content,
            "fold_manifest": folds.fingerprint,
            "mode": "sampled",
            "seed": 23,
        }
    )
    windows = ValidationWindowManifest(
        frame=frame,
        mode="sampled",
        seed=23,
        fold_manifest_fingerprint=folds.fingerprint,
        fingerprint=window_fingerprint,
    )
    names = tuple(f"physical_feature_{index}" for index in range(5))
    registry = FeatureRegistry(
        tuple(
            FeatureDefinition(
                name=name,
                feature_group="synthetic_physical",
                source_bands=("VV",),
                formula=f"synthetic formula {index}",
                temporal_aggregation="mean",
                validity_rule="always finite fixture",
                expected_dtype="float64",
                feature_kind="aggregate",
                output_representation="tabular",
                version="phase3_v1",
            )
            for index, name in enumerate(names)
        )
    )
    prepared = PreparedTabularData(
        features=pd.DataFrame(feature_rows, columns=names, dtype=np.float64),
        feature_names=names,
        selected_registry=registry,
        full_feature_schema_fingerprint="c" * 64,
        selected_feature_schema_fingerprint="d" * 64,
        folds=folds,
        windows=windows,
        rows_per_repeat=36,
    )
    runtime = replace(
        project.tabular,
        fold_manifest_fingerprint=folds.fingerprint,
        validation_window_fingerprint=windows.fingerprint,
    )
    return replace(project, tabular=runtime), prepared


def _fit_and_write(tmp_path: Path, project, prepared, *, experiment_id: str, family: str):
    experiment = load_tabular_experiment_config(
        _experiment_config(
            tmp_path / f"{experiment_id}.yaml",
            experiment_id=experiment_id,
            family=family,
        )
    )
    output = tmp_path / experiment_id
    output.mkdir()
    result = run_tabular_experiment(
        prepared,
        project,
        experiment,
        stage="full",
        output_dir=output,
    )
    manifest = write_tabular_experiment_artifacts(
        output,
        project=project,
        experiment=experiment,
        result=result,
    )
    return experiment, result, manifest


def test_fold_runner_artifacts_resume_and_diversity_are_complete(tmp_path: Path) -> None:
    project, prepared = _synthetic_prepared(tmp_path)
    cb_experiment, cb_result, cb_manifest = _fit_and_write(
        tmp_path,
        project,
        prepared,
        experiment_id="TEST-CB-FULL",
        family="catboost",
    )
    lgb_experiment, lgb_result, _ = _fit_and_write(
        tmp_path,
        project,
        prepared,
        experiment_id="TEST-LGB-FULL",
        family="lightgbm",
    )
    assert cb_result.oof.original.shape[0] == 12
    assert cb_result.oof.windows.shape[0] == 36
    assert len(cb_result.folds) == 2
    assert all(fold.best_iteration >= 1 for fold in cb_result.folds)
    assert not cb_result.oof.original.duplicated(["original_id", "repeat"]).any()
    expected_files = {
        "resolved_config.yaml",
        "experiment_manifest.json",
        "fold_manifest_fingerprint.json",
        "validation_window_fingerprint.json",
        "feature_schema.json",
        "feature_list.txt",
        "metrics.json",
        "fold_metrics.csv",
        "repeat_metrics.csv",
        "slice_metrics.csv",
        "oof_predictions.csv",
        "window_predictions.csv",
        "feature_importance.csv",
        "fold_models_manifest.json",
        "runtime.json",
        "report.md",
    }
    assert expected_files.issubset(path.name for path in (tmp_path / "TEST-CB-FULL").iterdir())
    assert load_experiment_artifact_manifest(tmp_path / "TEST-CB-FULL") == cb_manifest
    grouped_importance = pd.read_csv(tmp_path / "TEST-CB-FULL/feature_group_importance.csv")
    np.testing.assert_allclose(
        grouped_importance.groupby("importance_type")["mean_normalized_importance"].sum(),
        np.ones(grouped_importance["importance_type"].nunique()),
    )

    provenance = git_provenance(project.project_root)
    resumed = assert_resume_compatible(
        tmp_path / "TEST-CB-FULL",
        stage="full",
        experiment=cb_experiment,
        base_config_sha256=sha256_file(project.source_path),
        provenance=provenance,
        fold_manifest_fingerprint=prepared.folds.fingerprint,
        validation_window_fingerprint=prepared.windows.fingerprint,
        full_feature_schema_fingerprint=prepared.full_feature_schema_fingerprint,
        selected_feature_schema_fingerprint=prepared.selected_feature_schema_fingerprint,
    )
    assert resumed.oof_fingerprint == cb_result.oof.fingerprint
    with pytest.raises(ExperimentArtifactError, match="compatibility mismatch"):
        assert_resume_compatible(
            tmp_path / "TEST-CB-FULL",
            stage="full",
            experiment=cb_experiment,
            base_config_sha256=sha256_file(project.source_path),
            provenance=provenance,
            fold_manifest_fingerprint=prepared.folds.fingerprint,
            validation_window_fingerprint=prepared.windows.fingerprint,
            full_feature_schema_fingerprint="e" * 64,
            selected_feature_schema_fingerprint=prepared.selected_feature_schema_fingerprint,
        )

    candidate_path = tmp_path / "candidates.yaml"
    candidate_path.write_text(
        f"""\
candidates:
  - experiment_id: {cb_experiment.experiment_id}
    artifact_dir: {tmp_path / cb_experiment.experiment_id}
    candidate_role: strongest_catboost
    retention_reason: Synthetic contract candidate.
  - experiment_id: {lgb_experiment.experiment_id}
    artifact_dir: {tmp_path / lgb_experiment.experiment_id}
    candidate_role: strongest_lightgbm
    retention_reason: Synthetic diversity candidate.
""",
        encoding="utf-8",
    )
    registry: AcceptedCandidateRegistry = load_accepted_candidate_registry(candidate_path, project)
    report = analyze_oof_diversity(registry, prepared.folds, project)
    assert report.pairwise.shape[0] == 1
    assert report.slice_error_overlap["slice_group"].nunique() == 3
    assert report.diagnostic_blends.shape[0] == 1
    keys = ["repeat", "window_id"]
    aligned_windows = cb_result.oof.windows.loc[:, [*keys, "probability"]].merge(
        lgb_result.oof.windows.loc[:, [*keys, "probability"]],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    slow_windows = make_window_prediction_frame(
        cb_result.oof.windows,
        (
            aligned_windows["probability_left"].to_numpy()
            + aligned_windows["probability_right"].to_numpy()
        )
        / 2.0,
        experiment_id="SLOW-EQUIVALENCE",
        model_id="slow_equal_blend",
        fold_manifest_fingerprint=prepared.folds.fingerprint,
        validation_window_fingerprint=prepared.windows.fingerprint,
    )
    slow_oof = build_oof_predictions(
        slow_windows,
        prepared.folds,
        validation_window_fingerprint=prepared.windows.fingerprint,
        method="mean",
    )
    slow_report = build_validation_report(slow_oof, project.validation)
    fast_blend = report.diagnostic_blends.iloc[0]
    assert fast_blend["mean_combined_score"] == pytest.approx(
        slow_report.summary["official_metric"]["mean_combined_score"]
    )
    assert fast_blend["robust_score"] == pytest.approx(
        slow_report.summary["robust_selection"]["score"]
    )
    assert lgb_result.oof.original.shape == cb_result.oof.original.shape


def test_exact_authoritative_oof_count_and_feature_schema_guards(tmp_path: Path) -> None:
    project = load_project_config(_base_config(tmp_path / "base.yaml"))
    authoritative_runtime = TabularRuntimeConfig(
        validation_artifacts_dir=tmp_path / "validation",
        experiments_artifacts_dir=tmp_path / "experiments",
        fold_manifest_fingerprint="a" * 64,
        validation_window_fingerprint="b" * 64,
        feature_schema_fingerprint="c" * 64,
        expected_original_count=1821,
        expected_repeat_count=3,
        expected_full_oof_rows=5463,
        expected_feature_count=688,
        permutation_feature_count=10,
    )
    authoritative = replace(project, tabular=authoritative_runtime)
    ids = np.tile([f"original-{index:04d}" for index in range(1821)], 3)
    repeats = np.repeat(np.arange(3), 1821)
    frame = pd.DataFrame({"original_id": ids, "repeat": repeats})
    validate_full_oof_contract(SimpleNamespace(original=frame), authoritative)
    with pytest.raises(TabularTrainingError, match="exactly 5463"):
        validate_full_oof_contract(SimpleNamespace(original=frame.iloc[:-1]), authoritative)
    validate_phase3_feature_contract(
        feature_count=688,
        schema_fingerprint="c" * 64,
        project=authoritative,
    )
    with pytest.raises(TabularTrainingError, match="feature-schema fingerprint"):
        validate_phase3_feature_contract(
            feature_count=688,
            schema_fingerprint="d" * 64,
            project=authoritative,
        )


def test_candidate_registry_rejects_incompatible_fingerprints(tmp_path: Path) -> None:
    first = CandidateRecord(
        experiment_id="A",
        model_family="catboost",
        model_profile="a",
        feature_set="full",
        artifact_dir=tmp_path / "a",
        candidate_role="first",
        retention_reason="test",
        official_score=0.9,
        robust_score=0.8,
        fold_manifest_fingerprint="a" * 64,
        validation_window_fingerprint="b" * 64,
        selected_feature_schema_fingerprint="c" * 64,
    )
    second = replace(first, experiment_id="B", fold_manifest_fingerprint="d" * 64)
    with pytest.raises(ValueError, match="fold fingerprints differ"):
        AcceptedCandidateRegistry((first, second), "a" * 64, "b" * 64)
