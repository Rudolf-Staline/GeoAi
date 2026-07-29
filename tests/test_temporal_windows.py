from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoai_aquaculture.constants import OPTICAL_BANDS, RADAR_BANDS
from geoai_aquaculture.data import (
    FoldAssignmentError,
    MaskLibrary,
    MaskTemplateError,
    TemporalWindowError,
    WindowGenerationConfig,
    assert_no_fold_leakage,
    assign_original_folds,
    build_mask_template,
    enumerate_consecutive_windows,
    extract_test_mask_library,
    generate_temporal_windows,
    load_competition_data,
    load_project_config,
    materialize_test_windows,
    window_dataset_fingerprint,
    window_view_fingerprint,
)

BANDS = RADAR_BANDS + OPTICAL_BANDS
FEATURE_COLUMNS = tuple(f"{band}_{month:02d}" for month in range(1, 13) for band in BANDS)


def _configuration() -> str:
    return """\
project:
  name: synthetic-windows
  seed: 41
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
  strategy: stratified_group_kfold
  n_splits: 2
  n_repeats: 1
  fixed_threshold: 0.5
  window_lengths: [4, 5, 6]
  split_before_augmentation: true
augmentation:
  enabled: true
  use_test_missingness_masks: true
  exhaustive_windows: false
  windows_per_sample: 8
  temporal_dropout_enabled: false
  temporal_dropout_probability: 0.0
  optical_dropout_enabled: false
  optical_dropout_probability: 0.0
reporting:
  artifacts_dir: artifacts
"""


def _set_window(
    frame: pd.DataFrame,
    row: int,
    start: int,
    length: int,
    *,
    optical_gap_month: int | None = None,
    partial_optical_gap: bool = False,
) -> None:
    for month in range(start, start + length):
        for band_index, band in enumerate(BANDS):
            frame.loc[row, f"{band}_{month:02d}"] = (
                500_000 + row * 10_000 + month * 100 + band_index
            )
    if optical_gap_month is not None:
        bands = ("blue",) if partial_optical_gap else OPTICAL_BANDS
        for band in bands:
            frame.loc[row, f"{band}_{optical_gap_month:02d}"] = -9999.0


def _write_competition_fixture(root: Path) -> Path:
    train = pd.DataFrame(index=range(10), columns=FEATURE_COLUMNS, dtype=float)
    for row in range(train.shape[0]):
        for month in range(1, 13):
            for band_index, band in enumerate(BANDS):
                train.loc[row, f"{band}_{month:02d}"] = row * 10_000 + month * 100 + band_index
    train.insert(0, "label", [0, 1] * 5)
    train.insert(0, "ID", [f"train-{row:03d}" for row in range(10)])

    test = pd.DataFrame(-9999.0, index=range(4), columns=FEATURE_COLUMNS)
    _set_window(test, 0, 1, 4, optical_gap_month=2)
    _set_window(test, 1, 1, 4, optical_gap_month=2)
    _set_window(test, 2, 4, 5)
    _set_window(test, 3, 7, 6, optical_gap_month=9, partial_optical_gap=True)
    test.insert(0, "ID", [f"test-{row:03d}" for row in range(4)])

    sample = pd.DataFrame(
        {
            "ID": test["ID"],
            "TargetF1": [0] * len(test),
            "TargetRAUC": [0.0] * len(test),
        }
    )
    train.to_csv(root / "Train.csv", index=False)
    test.to_csv(root / "Test.csv", index=False)
    sample.to_csv(root / "SampleSubmission.csv", index=False)
    config_path = root / "base.yaml"
    config_path.write_text(_configuration(), encoding="utf-8")
    return config_path


def _loaded(tmp_path: Path):
    config = load_project_config(_write_competition_fixture(tmp_path))
    data = load_competition_data(config)
    folds = assign_original_folds(
        data.train,
        n_splits=config.validation.n_splits,
        seed=config.seed,
    )
    return config, data, folds


def _single_template_library(library: MaskLibrary, *, start: int, length: int) -> MaskLibrary:
    template = next(
        template
        for template in library.templates
        if template.window_start == start and template.window_length == length
    )
    return MaskLibrary(templates=(replace(template, frequency=1),))


def test_window_enumeration_has_exact_counts_and_edge_months() -> None:
    windows = enumerate_consecutive_windows()

    assert len(windows) == 24
    assert Counter(window.window_length for window in windows) == {4: 9, 5: 8, 6: 7}
    assert [(window.window_start, window.window_end) for window in windows[:2]] == [(1, 4), (2, 5)]
    assert any(
        window.window_length == 4 and window.window_start == 9 and window.window_end == 12
        for window in windows
    )


def test_fold_assignment_is_deterministic_and_precedes_augmentation(tmp_path: Path) -> None:
    config, data, folds = _loaded(tmp_path)
    train_before = data.train.copy(deep=True)

    repeated = assign_original_folds(
        data.train,
        n_splits=config.validation.n_splits,
        seed=config.seed,
    )

    pd.testing.assert_frame_equal(folds, repeated)
    assert folds["original_id"].is_unique
    assert folds["fold"].nunique() == 2
    pd.testing.assert_frame_equal(data.train, train_before)


def test_exhaustive_windows_have_relative_positions_and_no_leakage(tmp_path: Path) -> None:
    _config, data, folds = _loaded(tmp_path)
    train_before = data.train.copy(deep=True)
    test_before = data.test.copy(deep=True)
    folds_before = folds.copy(deep=True)
    generation = WindowGenerationConfig(
        enabled=True,
        exhaustive_windows=True,
        use_test_missingness_masks=False,
    )

    windows = generate_temporal_windows(data, folds, generation, seed=17)
    repeated = generate_temporal_windows(data, folds, generation, seed=17)

    assert windows.n_windows == 240
    assert windows.manifest["window_length"].value_counts().sort_index().to_dict() == {
        4: 90,
        5: 80,
        6: 70,
    }
    assert windows.manifest.groupby("original_id").size().eq(24).all()
    first = windows.manifest.iloc[0]
    assert (first["window_start"], first["window_end"]) == (1, 4)
    assert windows.calendar_months[0].tolist() == [1, 2, 3, 4, 0, 0]
    assert windows.relative_positions[0].tolist() == [1, 2, 3, 4, 0, 0]
    assert windows.position_mask[0].tolist() == [True, True, True, True, False, False]
    edge_index = windows.manifest.index[
        (windows.manifest["original_id"] == "train-000")
        & (windows.manifest["window_length"] == 4)
        & (windows.manifest["window_start"] == 9)
    ][0]
    assert windows.calendar_months[edge_index].tolist() == [9, 10, 11, 12, 0, 0]
    assert windows.values[0, 0, 0] == pytest.approx(100.0)
    assert np.isnan(windows.values[0, 4:]).all()
    assert windows.manifest.groupby("original_id")["fold"].nunique().eq(1).all()
    assert window_dataset_fingerprint(windows) == window_dataset_fingerprint(repeated)
    pd.testing.assert_frame_equal(data.train, train_before)
    pd.testing.assert_frame_equal(data.test, test_before)
    pd.testing.assert_frame_equal(folds, folds_before)


def test_sampled_windows_are_seeded_without_changing_folds(tmp_path: Path) -> None:
    config, data, folds = _loaded(tmp_path)
    library = extract_test_mask_library(data)

    first = generate_temporal_windows(
        data,
        folds,
        config.augmentation,
        seed=81,
        mask_library=library,
    )
    repeated = generate_temporal_windows(
        data,
        folds,
        config.augmentation,
        seed=81,
        mask_library=library,
    )
    alternate = generate_temporal_windows(
        data,
        folds,
        config.augmentation,
        seed=82,
        mask_library=library,
    )

    assert first.n_windows == 80
    assert window_dataset_fingerprint(first) == window_dataset_fingerprint(repeated)
    assert window_view_fingerprint(first) != window_view_fingerprint(alternate)
    first_folds = first.manifest[["original_id", "fold"]].drop_duplicates().reset_index(drop=True)
    alternate_folds = (
        alternate.manifest[["original_id", "fold"]].drop_duplicates().reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(first_folds, alternate_folds)
    assert set(first.manifest["mask_id"]) <= {template.mask_id for template in library.templates}


def test_mask_library_uses_only_availability_and_applies_optical_gap(tmp_path: Path) -> None:
    config, data, folds = _loaded(tmp_path)
    library = extract_test_mask_library(data)
    repeated_pattern = next(
        template
        for template in library.templates
        if template.window_start == 1 and template.window_length == 4
    )
    one_mask = _single_template_library(library, start=1, length=4)
    generation = replace(config.augmentation, windows_per_sample=1)

    changed_test = data.test.copy(deep=True)
    feature_columns = list(data.feature_columns)
    valid = changed_test.loc[:, feature_columns].notna()
    changed_test.loc[:, feature_columns] = changed_test.loc[:, feature_columns].where(
        ~valid, changed_test.loc[:, feature_columns] + 9_000_000
    )
    changed_data = replace(data, test=changed_test)
    changed_library = extract_test_mask_library(changed_data)

    pd.testing.assert_frame_equal(library.to_frame(), changed_library.to_frame())
    windows = generate_temporal_windows(
        data,
        folds,
        generation,
        seed=7,
        mask_library=one_mask,
    )
    changed_windows = generate_temporal_windows(
        changed_data,
        folds,
        generation,
        seed=7,
        mask_library=one_mask,
    )

    np.testing.assert_allclose(windows.values, changed_windows.values, equal_nan=True)
    assert windows.radar_mask[0].tolist() == [True, True, True, True, False, False]
    assert windows.optical_mask[0, 1].tolist() == [False] * len(OPTICAL_BANDS)
    assert not np.isnan(windows.values[0, 1, : len(RADAR_BANDS)]).any()
    assert np.isnan(windows.values[0, 1, len(RADAR_BANDS) :]).all()
    assert windows.manifest.loc[0, "internal_optical_gap_count"] == 1
    assert windows.manifest.loc[0, "test_mask_optical_gap_count"] == 1
    assert repeated_pattern.frequency == 2
    assert one_mask.templates[0].frequency == 1
    assert library.observation_count == 4
    assert len(library.templates) == 3


def test_per_band_optical_gap_is_preserved(tmp_path: Path) -> None:
    config, data, folds = _loaded(tmp_path)
    library = extract_test_mask_library(data)
    partial = _single_template_library(library, start=7, length=6)
    generation = replace(config.augmentation, windows_per_sample=1)

    windows = generate_temporal_windows(
        data,
        folds,
        generation,
        seed=5,
        mask_library=partial,
    )
    blue_index = windows.band_names.index("blue")
    green_index = windows.band_names.index("green")

    assert windows.calendar_months[0].tolist() == [7, 8, 9, 10, 11, 12]
    assert windows.optical_mask[0, 2, 0] == np.bool_(False)
    assert windows.optical_mask[0, 2, 1:].all()
    assert np.isnan(windows.values[0, 2, blue_index])
    assert not np.isnan(windows.values[0, 2, green_index])
    assert windows.manifest.loc[0, "optical_months"] == 5
    assert windows.manifest.loc[0, "internal_optical_gap_count"] == 1


def test_observed_test_windows_preserve_values_masks_and_input(tmp_path: Path) -> None:
    _config, data, _folds = _loaded(tmp_path)
    test_before = data.test.copy(deep=True)

    windows = materialize_test_windows(data)
    repeated = materialize_test_windows(data)

    assert windows.n_windows == data.test.shape[0]
    assert "label" not in windows.manifest
    assert windows.manifest["fold"].eq(-1).all()
    assert windows.manifest["original_id"].tolist() == data.test["ID"].tolist()
    assert windows.calendar_months[3].tolist() == [7, 8, 9, 10, 11, 12]
    assert windows.values[0, 0, windows.band_names.index("VH")] == pytest.approx(500_100.0)
    assert np.isnan(windows.values[0, 1, windows.band_names.index("blue")])
    assert windows.optical_mask[3, 2, 0] == np.bool_(False)
    assert windows.optical_mask[3, 2, 1:].all()
    assert window_dataset_fingerprint(windows) == window_dataset_fingerprint(repeated)
    pd.testing.assert_frame_equal(data.test, test_before)


def test_padding_and_configured_dropout_are_distinguishable(tmp_path: Path) -> None:
    _config, data, folds = _loaded(tmp_path)
    temporal_dropout = WindowGenerationConfig(
        enabled=True,
        windows_per_sample=1,
        temporal_dropout_enabled=True,
        temporal_dropout_probability=1.0,
    )
    optical_dropout = WindowGenerationConfig(
        enabled=True,
        windows_per_sample=1,
        optical_dropout_enabled=True,
        optical_dropout_probability=1.0,
    )

    temporal = generate_temporal_windows(data, folds, temporal_dropout, seed=3)
    optical = generate_temporal_windows(data, folds, optical_dropout, seed=3)

    assert temporal.position_mask[0].sum() in {4, 5, 6}
    assert not temporal.radar_mask[0].any()
    assert not temporal.optical_mask[0].any()
    assert temporal.manifest.loc[0, "temporal_dropout_mask"].count("1") in {4, 5, 6}
    assert optical.radar_mask[0].sum() == optical.position_mask[0].sum()
    assert not optical.optical_mask[0].any()
    assert np.isnan(optical.values[0, :, len(RADAR_BANDS) :]).all()
    assert optical.calendar_months[0, ~optical.position_mask[0]].tolist() == [0] * int(
        (~optical.position_mask[0]).sum()
    )


def test_window_leakage_validator_rejects_cross_fold_copy(tmp_path: Path) -> None:
    _config, data, folds = _loaded(tmp_path)
    generation = WindowGenerationConfig(enabled=True, windows_per_sample=2)
    windows = generate_temporal_windows(data, folds, generation, seed=13)
    corrupted = windows.manifest.copy(deep=True)
    first_id = corrupted.loc[0, "original_id"]
    duplicate_index = corrupted.index[corrupted["original_id"] == first_id][1]
    corrupted.loc[duplicate_index, "fold"] = 1 - int(corrupted.loc[0, "fold"])

    with pytest.raises(FoldAssignmentError, match="cross fold boundaries"):
        assert_no_fold_leakage(corrupted)


def test_window_generation_rejects_post_augmentation_fold_manifest(tmp_path: Path) -> None:
    _config, data, folds = _loaded(tmp_path)
    duplicated = pd.concat([folds, folds.iloc[[0]]], ignore_index=True)
    generation = WindowGenerationConfig(enabled=True, windows_per_sample=1)

    with pytest.raises(TemporalWindowError, match="before augmentation"):
        generate_temporal_windows(data, duplicated, generation, seed=1)


def test_mask_validation_rejects_malformed_and_nonconsecutive_patterns() -> None:
    no_optical = [[False] * len(OPTICAL_BANDS) for _ in range(12)]
    nonconsecutive = [True, True, False, True, True, *([False] * 7)]
    with pytest.raises(MaskTemplateError, match="consecutive"):
        build_mask_template(nonconsecutive, no_optical)

    radar = [True, True, True, True, *([False] * 8)]
    optical_outside = [[False] * len(OPTICAL_BANDS) for _ in range(12)]
    optical_outside[8] = [True] * len(OPTICAL_BANDS)
    with pytest.raises(MaskTemplateError, match="outside the radar window"):
        build_mask_template(radar, optical_outside)

    with pytest.raises(MaskTemplateError, match="exactly 12 months"):
        build_mask_template(radar[:-1], no_optical)


def test_mask_extraction_rejects_partial_or_nonconsecutive_radar(tmp_path: Path) -> None:
    _config, data, _folds = _loaded(tmp_path)
    partial_test = data.test.copy(deep=True)
    partial_test.loc[0, "VH_01"] = np.nan
    with pytest.raises(MaskTemplateError, match="partial radar"):
        extract_test_mask_library(replace(data, test=partial_test))

    broken_test = data.test.copy(deep=True)
    broken_test.loc[0, ["VH_02", "VV_02"]] = np.nan
    with pytest.raises(MaskTemplateError, match="consecutive"):
        extract_test_mask_library(replace(data, test=broken_test))
