from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from geoai_aquaculture.constants import OPTICAL_BANDS, RADAR_BANDS
from geoai_aquaculture.data import (
    ConfigError,
    FeatureConfig,
    TemporalWindowDataset,
    ValidationConfig,
)
from geoai_aquaculture.metrics import metric_result
from geoai_aquaculture.validation import (
    OOFPredictions,
    ValidationError,
    ValidationWindowManifest,
    ValidationWindowSet,
    aggregate_probabilities,
    build_cluster_holdout_manifest,
    build_leave_season_out_manifest,
    build_oof_predictions,
    build_repeated_fold_manifest,
    build_similarity_holdout_manifest,
    build_validation_report,
    cluster_balance_summary,
    dataframe_fingerprint,
    fold_balance_summary,
    json_fingerprint,
    load_fold_manifest,
    load_oof_predictions,
    load_validation_window_manifest,
    make_reference_estimator,
    make_window_prediction_frame,
    materialize_leave_season_split,
    run_reference_estimator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANDS = RADAR_BANDS + OPTICAL_BANDS
FEATURE_COLUMNS = tuple(f"{band}_{month:02d}" for month in range(1, 13) for band in BANDS)


def _train_frame(n_originals: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": pd.Series(
                [f"original-{index:03d}" for index in range(n_originals)], dtype="string"
            ),
            "label": np.asarray([index % 2 for index in range(n_originals)], dtype=np.int64),
        }
    )


def _validation_config(*, seed: int = 2026, n_repeats: int = 2) -> ValidationConfig:
    return ValidationConfig(
        seed=seed,
        n_splits=3,
        n_repeats=n_repeats,
        sampled_windows_per_original=4,
        validation_window_seed=811,
        similarity_holdout_min_samples=6,
        cluster_n_clusters=2,
        cluster_min_size=4,
    )


def _folds(*, seed: int = 2026, n_repeats: int = 2):
    config = _validation_config(seed=seed, n_repeats=n_repeats)
    return build_repeated_fold_manifest(_train_frame(), config), config


def _fixed_windows(folds, config) -> ValidationWindowManifest:
    records: list[dict[str, object]] = []
    templates = ((4, 1, 0), (5, 4, 1), (6, 7, 2), (4, 2, 0))
    for row in folds.frame.itertuples(index=False):
        for view, (length, start, gaps) in enumerate(templates):
            optical_months = max(0, length - gaps)
            records.append(
                {
                    "repeat": int(row.repeat),
                    "window_id": f"r{row.repeat}-{row.original_id}-w{view}",
                    "original_id": str(row.original_id),
                    "fold": int(row.fold),
                    "label": int(row.label),
                    "generation_mode": "sampled",
                    "view_index": view,
                    "augmentation_seed": config.validation_window_seed,
                    "window_start": start,
                    "window_end": start + length - 1,
                    "window_length": length,
                    "radar_availability": "1" * length + "0" * (6 - length),
                    "optical_month_availability": (
                        "1" * optical_months + "0" * (6 - optical_months)
                    ),
                    "radar_months": length,
                    "optical_months": optical_months,
                    "internal_optical_gap_count": gaps,
                    "mask_id": f"mask-{view}",
                }
            )
    frame = pd.DataFrame.from_records(records)
    content = dataframe_fingerprint(frame)
    fingerprint = json_fingerprint(
        {
            "content": content,
            "fold_manifest": folds.fingerprint,
            "mode": "sampled",
            "seed": config.validation_window_seed,
        }
    )
    return ValidationWindowManifest(
        frame=frame,
        mode="sampled",
        seed=config.validation_window_seed,
        fold_manifest_fingerprint=folds.fingerprint,
        fingerprint=fingerprint,
    )


def _window_predictions(folds, config):
    windows = _fixed_windows(folds, config)
    labels = windows.frame["label"].to_numpy(dtype=np.float64)
    view = windows.frame["view_index"].to_numpy(dtype=np.float64)
    probability = np.where(labels == 1.0, 0.76, 0.24) + (view - 1.5) * 0.04
    frame = make_window_prediction_frame(
        windows.frame,
        probability,
        experiment_id="SYNTHETIC-REFERENCE",
        model_id="deterministic-fixture",
        fold_manifest_fingerprint=folds.fingerprint,
        validation_window_fingerprint=windows.fingerprint,
    )
    return windows, frame


def _temporal_dataset(windows: ValidationWindowManifest, *, repeat: int = 0):
    manifest = windows.frame.loc[windows.frame["repeat"].eq(repeat)].reset_index(drop=True)
    rows = manifest.shape[0]
    values = np.full((rows, 6, len(BANDS)), np.nan, dtype=np.float64)
    calendar = np.zeros((rows, 6), dtype=np.int8)
    relative = np.zeros((rows, 6), dtype=np.int8)
    positions = np.zeros((rows, 6), dtype=bool)
    radar = np.zeros((rows, 6), dtype=bool)
    optical = np.zeros((rows, 6, len(OPTICAL_BANDS)), dtype=bool)
    band_index = {band: index for index, band in enumerate(BANDS)}
    base = {
        "blue": 1.0,
        "green": 3.0,
        "nir": 6.0,
        "nira": 7.0,
        "re1": 2.0,
        "re2": 4.0,
        "re3": 5.0,
        "red": 2.0,
        "swir1": 2.0,
        "swir2": 3.0,
    }
    for row, metadata in enumerate(manifest.itertuples(index=False)):
        length = int(metadata.window_length)
        positions[row, :length] = True
        radar[row, :length] = True
        optical[row, :length, :] = True
        calendar[row, :length] = np.arange(metadata.window_start, metadata.window_end + 1)
        relative[row, :length] = np.arange(1, length + 1)
        for position in range(length):
            values[row, position, band_index["VH"]] = -5.0 - position
            values[row, position, band_index["VV"]] = -3.0 - position
            for band, value in base.items():
                values[row, position, band_index[band]] = value * (1.0 + position / 20.0)
    return TemporalWindowDataset(
        manifest=manifest,
        values=values,
        calendar_months=calendar,
        relative_positions=relative,
        position_mask=positions,
        radar_mask=radar,
        optical_mask=optical,
        band_names=BANDS,
        optical_bands=OPTICAL_BANDS,
    )


def test_validation_config_locks_threshold_and_robust_policy() -> None:
    config = _validation_config()
    assert config.threshold == 0.5
    assert sum(
        (
            config.robust_score_weights.mean_combined,
            config.robust_score_weights.worst_fold,
            config.robust_score_weights.worst_window_length,
            config.robust_score_weights.worst_season,
        )
    ) == pytest.approx(1.0)
    with pytest.raises(ConfigError, match=r"exactly 0\.5"):
        replace(config, fixed_threshold=0.49)
    with pytest.raises(ConfigError, match="precede"):
        replace(config, split_before_augmentation=False)


def test_repeated_folds_are_complete_stratified_deterministic_and_seeded() -> None:
    folds, config = _folds()
    repeated = build_repeated_fold_manifest(_train_frame(), config)
    alternate = build_repeated_fold_manifest(_train_frame(), replace(config, seed=config.seed + 1))

    assert folds.frame.shape == (60, 6)
    assert not folds.frame.duplicated(["original_id", "repeat"]).any()
    assert folds.frame.groupby("repeat")["original_id"].nunique().eq(30).all()
    assert folds.fingerprint == repeated.fingerprint
    assert folds.fingerprint != alternate.fingerprint
    balance = fold_balance_summary(folds)
    assert balance.shape[0] == config.n_splits * config.n_repeats
    assert (balance["validation_positive_rate"] - 0.5).abs().max() <= 0.1

    augmented = pd.concat([_train_frame(), _train_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError, match="before augmentation"):
        build_repeated_fold_manifest(augmented, config)


def test_fold_manifest_rejects_mutation_and_cross_fold_originals() -> None:
    folds, _ = _folds()
    malformed = folds.frame.copy()
    malformed.loc[0, "fold"] = (int(malformed.loc[0, "fold"]) + 1) % folds.n_splits
    with pytest.raises(ValidationError, match="fingerprint mismatch"):
        replace(folds, frame=malformed)


def test_probability_aggregation_methods_are_deterministic_and_safe() -> None:
    values = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert aggregate_probabilities(values, method="mean") == pytest.approx(0.5)
    assert aggregate_probabilities(values, method="median") == pytest.approx(0.5)
    assert aggregate_probabilities(values, method="logit_mean") == pytest.approx(0.5)
    assert aggregate_probabilities(
        np.asarray([0.0, 0.2, 0.3, 1.0]), method="trimmed_mean", trimmed_fraction=0.25
    ) == pytest.approx(0.25)
    with pytest.raises(ValidationError, match=r"within \[0, 1\]"):
        aggregate_probabilities(np.asarray([0.4, 1.1]))


def test_exact_metrics_use_point_five_and_explicit_single_class_auc() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.5, 0.5, 0.9])
    result = metric_result(labels, probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    assert result.f1 == pytest.approx(f1_score(labels, predictions))
    assert result.roc_auc == pytest.approx(roc_auc_score(labels, probabilities))
    assert result.combined_score == pytest.approx(0.6 * result.f1 + 0.4 * result.roc_auc)
    assert (
        result.true_negative,
        result.false_positive,
        result.false_negative,
        result.true_positive,
    ) == (
        1,
        1,
        0,
        2,
    )
    single = metric_result(np.ones(3), np.asarray([0.4, 0.6, 0.8]))
    assert single.roc_auc is None and single.combined_score is None
    assert single.auc_defined is False
    with pytest.raises(ValueError, match=r"exactly 0\.5"):
        metric_result(labels, probabilities, threshold=0.51)


def test_oof_contract_aggregates_originals_and_rejects_malformed_predictions(
    tmp_path: Path,
) -> None:
    folds, config = _folds()
    windows, predictions = _window_predictions(folds, config)
    oof = build_oof_predictions(
        predictions,
        folds,
        validation_window_fingerprint=windows.fingerprint,
    )

    assert oof.original.shape[0] == folds.n_originals * folds.n_repeats
    assert not oof.original.duplicated(["original_id", "repeat"]).any()
    assert oof.original["window_count"].eq(4).all()
    assert oof.original["prediction"].equals(oof.original["predicted_class"])
    first = predictions.loc[
        (predictions["repeat"] == 0) & (predictions["original_id"].astype(str) == "original-000")
    ]
    actual = oof.original.loc[
        (oof.original["repeat"] == 0) & (oof.original["original_id"].astype(str) == "original-000"),
        "probability",
    ].iloc[0]
    assert actual == pytest.approx(first["probability"].mean())
    original_path = tmp_path / "oof.csv"
    window_path = tmp_path / "window.csv"
    oof.original.to_csv(original_path, index=False, float_format="%.17g")
    oof.windows.to_csv(window_path, index=False, float_format="%.17g")
    loaded = load_oof_predictions(
        original_path,
        window_path,
        folds,
        validation_window_fingerprint=windows.fingerprint,
        expected_fingerprint=oof.fingerprint,
    )
    assert loaded.fingerprint == oof.fingerprint

    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError, match="duplicate"):
        build_oof_predictions(
            duplicate,
            folds,
            validation_window_fingerprint=windows.fingerprint,
        )
    missing_original = predictions.loc[
        ~((predictions["repeat"] == 0) & (predictions["original_id"].astype(str) == "original-000"))
    ]
    with pytest.raises(ValidationError, match="missing"):
        build_oof_predictions(
            missing_original,
            folds,
            validation_window_fingerprint=windows.fingerprint,
        )
    with pytest.raises(ValidationError, match="fold-manifest fingerprint mismatch"):
        replace(oof, fold_manifest_fingerprint="wrong")


def test_temporal_slices_are_original_level_and_stability_uses_all_windows() -> None:
    folds, config = _folds()
    windows, predictions = _window_predictions(folds, config)
    oof = build_oof_predictions(
        predictions,
        folds,
        validation_window_fingerprint=windows.fingerprint,
    )
    report = build_validation_report(oof, config)

    length_four = report.slice_metrics.loc[
        (report.slice_metrics["slice_group"] == "window_length")
        & (report.slice_metrics["slice_value"] == "4")
    ]
    assert length_four["window_prediction_count"].eq(60).all()
    assert length_four["original_prediction_count"].eq(30).all()
    assert report.prediction_stability["window_count"].eq(4).all()
    assert set(report.summary["window_length_scores"]) == {"4", "5", "6"}
    assert set(report.summary["season_scores"]) == {"early_year", "mid_year", "late_year"}
    assert 0.0 <= report.summary["robust_selection"]["score"] <= 1.0


def test_leave_season_out_keeps_validation_originals_out_of_training() -> None:
    folds, config = _folds()
    windows = _fixed_windows(folds, config)
    manifest = build_leave_season_out_manifest(windows, folds, config)
    assert manifest.frame.shape[0] == config.n_repeats * config.n_splits * len(config.seasons)
    train, validation = materialize_leave_season_split(
        windows,
        config,
        repeat=0,
        fold=0,
        season_name="early_year",
    )
    assert not set(train["original_id"]) & set(validation["original_id"])
    assert validation["window_start"].isin([1, 2, 3]).all()
    assert ~train["window_start"].isin([1, 2, 3]).any()


def test_cluster_and_similarity_holdout_interfaces_are_train_only_and_oof() -> None:
    folds, config = _folds(n_repeats=1)
    ids = _train_frame()["ID"]
    invariant = pd.DataFrame(
        {
            "original_id": ids,
            "aggregate_a": np.r_[np.linspace(-3, -2, 15), np.linspace(2, 3, 15)],
            "aggregate_b": np.r_[np.linspace(-1, 0, 15), np.linspace(1, 2, 15)],
        }
    )
    cluster = build_cluster_holdout_manifest(
        invariant,
        folds.frame,
        repeat=0,
        outer_fold=0,
        n_clusters=2,
        minimum_cluster_size=4,
        seed=config.seed,
    )
    summary = cluster_balance_summary(cluster)
    assert summary["size"].sum() == 20
    assert summary["size"].min() >= 4
    assert "label" not in cluster.feature_names

    scores = pd.DataFrame(
        {
            "original_id": ids,
            "similarity_score": np.linspace(0.0, 1.0, len(ids)),
            "is_oof": True,
        }
    )
    holdout = build_similarity_holdout_manifest(
        scores,
        folds,
        fraction=0.2,
        minimum_samples=6,
    )
    assert holdout.selected_count == 6
    assert holdout.frame.loc[holdout.frame["selected"], "label"].value_counts().to_dict() == {
        0: 3,
        1: 3,
    }
    scores.loc[0, "is_oof"] = False
    with pytest.raises(ValidationError, match="out-of-fold"):
        build_similarity_holdout_manifest(scores, folds, fraction=0.2, minimum_samples=6)


def test_reference_runner_is_deterministic_fold_local_and_rejects_prefit_factory() -> None:
    folds, config = _folds(n_repeats=1)
    windows = _fixed_windows(folds, config)
    template = _temporal_dataset(windows)
    window_set = ValidationWindowSet(template, windows, n_repeats=1)

    first = run_reference_estimator(window_set, folds, config, FeatureConfig())
    second = run_reference_estimator(window_set, folds, config, FeatureConfig())
    assert first.oof.fingerprint == second.oof.fingerprint
    assert first.oof.original.shape[0] == 30
    assert (
        first.fold_metadata["preprocessing_scope"]
        .eq("fit_only_on_current_fold_training_windows")
        .all()
    )
    assert first.fold_metadata.shape[0] == config.n_splits

    fitted = make_reference_estimator(1)
    fitted.fit(np.zeros((4, 16)), np.asarray([0, 1, 0, 1]))
    with pytest.raises(ValidationError, match="fitted outside"):
        run_reference_estimator(
            window_set,
            folds,
            config,
            FeatureConfig(),
            estimator_factory=lambda _seed: fitted,
        )


def _write_cli_fixture(root: Path) -> Path:
    train = pd.DataFrame(index=range(12), columns=FEATURE_COLUMNS, dtype=float)
    for row in range(train.shape[0]):
        for month in range(1, 13):
            for band_index, band in enumerate(BANDS):
                train.loc[row, f"{band}_{month:02d}"] = 10 + row + month + band_index / 10
    train.insert(0, "label", [0, 1] * 6)
    train.insert(0, "ID", [f"train-{row:03d}" for row in range(12)])
    test = pd.DataFrame(-9999.0, index=range(3), columns=FEATURE_COLUMNS)
    for row, (start, length) in enumerate(((1, 4), (4, 5), (7, 6))):
        for month in range(start, start + length):
            for band_index, band in enumerate(BANDS):
                test.loc[row, f"{band}_{month:02d}"] = 20 + row + month + band_index / 10
    test.loc[0, [f"{band}_02" for band in OPTICAL_BANDS]] = -9999.0
    test.insert(0, "ID", [f"test-{row:03d}" for row in range(3)])
    sample = pd.DataFrame({"ID": test["ID"], "TargetF1": [0] * 3, "TargetRAUC": [0.0] * 3})
    train.to_csv(root / "Train.csv", index=False)
    test.to_csv(root / "Test.csv", index=False)
    sample.to_csv(root / "SampleSubmission.csv", index=False)
    config = """\
project:
  name: synthetic-validation
  seed: 29
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
  seed: 17
  n_splits: 2
  n_repeats: 2
  threshold: 0.5
  sampled_windows_per_original: 2
  validation_window_seed: 31
  aggregation_method: mean
  similarity_holdout_min_samples: 4
  cluster_n_clusters: 2
  cluster_min_size: 2
augmentation:
  enabled: true
  use_test_missingness_masks: true
  windows_per_sample: 2
features:
  version: phase3_test
reporting:
  artifacts_dir: artifacts
"""
    path = root / "base.yaml"
    path.write_text(config, encoding="utf-8")
    return path


def _run_validation_cli(config: Path, output: Path, *, seed: int | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_validation.py"),
        "--config",
        str(config),
        "--skip-reference",
        "--output-dir",
        str(output),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_validation_cli_artifacts_are_reproducible_and_seed_sensitive(tmp_path: Path) -> None:
    config = _write_cli_fixture(tmp_path)
    first = tmp_path / "validation-a"
    repeated = tmp_path / "validation-b"
    alternate = tmp_path / "validation-c"
    _run_validation_cli(config, first)
    _run_validation_cli(config, repeated)
    _run_validation_cli(config, alternate, seed=18)

    first_fingerprints = json.loads((first / "fingerprints.json").read_text())
    repeated_fingerprints = json.loads((repeated / "fingerprints.json").read_text())
    alternate_fingerprints = json.loads((alternate / "fingerprints.json").read_text())
    assert first_fingerprints == repeated_fingerprints
    assert first_fingerprints["fold_manifest"] != alternate_fingerprints["fold_manifest"]
    assert pd.read_csv(first / "fold_manifest.csv").shape[0] == 24
    assert pd.read_csv(first / "validation_window_manifest.csv").shape[0] == 48
    loaded_folds = load_fold_manifest(
        first / "fold_manifest.csv",
        replace(_validation_config(seed=17), n_splits=2),
        expected_fingerprint=first_fingerprints["fold_manifest"],
    )
    loaded_windows = load_validation_window_manifest(
        first / "validation_window_manifest.csv",
        loaded_folds,
        replace(
            _validation_config(seed=17),
            n_splits=2,
            sampled_windows_per_original=2,
            validation_window_seed=31,
        ),
        expected_fingerprint=first_fingerprints["validation_window_manifest"],
    )
    assert loaded_folds.frame.shape[0] == 24
    assert loaded_windows.frame.shape[0] == 48
    expected = {
        "diagnostic_plans.json",
        "fingerprints.json",
        "fold_manifest.csv",
        "fold_summary.json",
        "leave_season_out_manifest.csv",
        "protocol.json",
        "run_metadata.json",
        "slice_definitions.json",
        "validation_report.md",
        "validation_window_manifest.csv",
    }
    assert {path.name for path in first.iterdir()} == expected


def test_oof_constructor_rejects_probability_or_manifest_tampering() -> None:
    folds, config = _folds()
    windows, predictions = _window_predictions(folds, config)
    oof = build_oof_predictions(
        predictions,
        folds,
        validation_window_fingerprint=windows.fingerprint,
    )
    bad_probability = oof.original.copy()
    bad_probability.loc[0, "probability"] = np.inf
    with pytest.raises(ValidationError, match="finite"):
        OOFPredictions(
            bad_probability,
            oof.windows,
            oof.fold_manifest_fingerprint,
            oof.validation_window_fingerprint,
            oof.aggregation_method,
            oof.trimmed_fraction,
            oof.fingerprint,
        )
    bad_windows = windows.frame.copy()
    bad_windows.loc[0, "window_start"] = 9
    with pytest.raises(ValidationError, match="fingerprint mismatch"):
        replace(windows, frame=bad_windows)
