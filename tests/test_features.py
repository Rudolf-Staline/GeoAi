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

from geoai_aquaculture.constants import OPTICAL_BANDS, RADAR_BANDS
from geoai_aquaculture.data import (
    BandSemanticMapping,
    ConfigError,
    FeatureConfig,
    TemporalWindowDataset,
    load_project_config,
)
from geoai_aquaculture.features import (
    AGGREGATION_NAMES,
    FeatureEngineeringError,
    aggregate_temporal_series,
    assert_feature_schema_alignment,
    build_feature_audit,
    build_feature_representations,
    build_monthly_features,
    safe_divide,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANDS = RADAR_BANDS + OPTICAL_BANDS
FEATURE_COLUMNS = tuple(f"{band}_{month:02d}" for month in range(1, 13) for band in BANDS)


def _feature_windows() -> TemporalWindowDataset:
    rows = 3
    values = np.full((rows, 6, len(BANDS)), np.nan, dtype=np.float64)
    lengths = (4, 5, 6)
    starts = (3, 5, 7)
    position_mask = np.zeros((rows, 6), dtype=bool)
    radar_mask = np.zeros((rows, 6), dtype=bool)
    optical_mask = np.zeros((rows, 6, len(OPTICAL_BANDS)), dtype=bool)
    calendar_months = np.zeros((rows, 6), dtype=np.int8)
    relative_positions = np.zeros((rows, 6), dtype=np.int8)
    band_index = {band: index for index, band in enumerate(BANDS)}
    optical_base = {
        "blue": 1.0,
        "green": 3.0,
        "nir": 6.0,
        "nira": 8.0,
        "re1": 2.0,
        "re2": 4.0,
        "re3": 5.0,
        "red": 2.0,
        "swir1": 2.0,
        "swir2": 3.0,
    }
    for row, (length, start) in enumerate(zip(lengths, starts, strict=True)):
        position_mask[row, :length] = True
        radar_mask[row, :length] = True
        optical_mask[row, :length, :] = True
        calendar_months[row, :length] = np.arange(start, start + length)
        relative_positions[row, :length] = np.arange(1, length + 1)
        for position in range(length):
            values[row, position, band_index["VH"]] = -4.0 - row - position
            values[row, position, band_index["VV"]] = -2.0 - row - position
            multiplier = 1.0 + 0.1 * position + 0.05 * row
            for band, base in optical_base.items():
                values[row, position, band_index[band]] = base * multiplier

    # A partial cloud anomaly: blue is absent but indices not requiring blue remain valid.
    optical_mask[0, 1, OPTICAL_BANDS.index("blue")] = False
    values[0, 1, band_index["blue"]] = np.nan
    # Three optical-gap months, including one partial-band gap, inside a radar-valid window.
    optical_mask[1, 1:3, :] = False
    values[1, 1:3, len(RADAR_BANDS) :] = np.nan
    radar_mask[1, 1] = False
    values[1, 1, : len(RADAR_BANDS)] = np.nan
    optical_mask[1, 3, OPTICAL_BANDS.index("blue")] = False
    values[1, 3, band_index["blue"]] = np.nan
    # All optical measurements unavailable, while radar remains available.
    optical_mask[2, :, :] = False
    values[2, :, len(RADAR_BANDS) :] = np.nan

    manifest = pd.DataFrame(
        {
            "ID": ["original-a", "original-a", "original-b"],
            "window_id": ["window-a-4", "window-a-5", "window-b-6"],
            "original_id": ["original-a", "original-a", "original-b"],
            "fold": np.asarray([0, 0, 1], dtype=np.int16),
            "label": np.asarray([1, 1, 0], dtype=np.int8),
            "window_start": starts,
            "window_end": tuple(
                start + length - 1 for start, length in zip(starts, lengths, strict=True)
            ),
            "window_length": lengths,
        }
    )
    return TemporalWindowDataset(
        manifest=manifest,
        values=values,
        calendar_months=calendar_months,
        relative_positions=relative_positions,
        position_mask=position_mask,
        radar_mask=radar_mask,
        optical_mask=optical_mask,
        band_names=BANDS,
        optical_bands=OPTICAL_BANDS,
    )


def _channel(collection, name: str) -> tuple[np.ndarray, np.ndarray]:
    index = next(index for index, spec in enumerate(collection.specs) if spec.name == name)
    return collection.values[:, :, index], collection.masks[:, :, index]


def test_safe_division_handles_zero_negative_missing_and_overflow() -> None:
    numerator = np.asarray([1.0, 1.0, 1.0, np.nan, 1.0e308, 4.0])
    denominator = np.asarray([0.0, 1.0e-7, -2.0, 1.0, 0.5, -2.0])

    result = safe_divide(numerator, denominator, epsilon=1.0e-6)

    assert result.validity.tolist() == [False, False, True, False, False, True]
    assert np.isnan(result.values[[0, 1, 3, 4]]).all()
    np.testing.assert_allclose(result.values[[2, 5]], [-0.5, -2.0])
    assert not np.isinf(result.values).any()
    with pytest.raises(FeatureEngineeringError, match="finite and positive"):
        safe_divide(numerator, denominator, epsilon=0.0)


def test_optical_index_formulas_and_validity_are_exact() -> None:
    monthly = build_monthly_features(_feature_windows(), FeatureConfig())
    expected = {
        "optical__ndvi": 0.5,
        "optical__ndwi": -1.0 / 3.0,
        "optical__mndwi": 0.2,
        "optical__ndmi": 0.5,
        "optical__nbr": 1.0 / 3.0,
    }

    for name, expected_value in expected.items():
        values, mask = _channel(monthly, name)
        assert mask[0, 0]
        assert values[0, 0] == pytest.approx(expected_value)

    ndvi, ndvi_mask = _channel(monthly, "optical__ndvi")
    blue_contrast, blue_contrast_mask = _channel(monthly, "optical__blue_green_contrast")
    assert ndvi_mask[0, 1]
    assert np.isfinite(ndvi[0, 1])
    assert not blue_contrast_mask[0, 1]
    assert np.isnan(blue_contrast[0, 1])
    assert not ndvi_mask[2].any()
    assert np.isnan(ndvi[2]).all()


def test_radar_formulas_ratios_and_adjacent_differences_are_exact() -> None:
    monthly = build_monthly_features(_feature_windows(), FeatureConfig())
    expected = {
        "radar__vv_minus_vh": 2.0,
        "radar__vv_plus_vh": -6.0,
        "radar__vv_over_abs_vh": -0.5,
        "radar__vh_over_abs_vv": -2.0,
    }
    for name, expected_value in expected.items():
        values, mask = _channel(monthly, name)
        assert mask[0, 0]
        assert values[0, 0] == pytest.approx(expected_value)

    vv_difference, vv_difference_mask = _channel(monthly, "radar__vv_first_difference")
    assert not vv_difference_mask[0, 0]
    assert np.isnan(vv_difference[0, 0])
    assert vv_difference_mask[0, 1]
    assert vv_difference[0, 1] == pytest.approx(-1.0)


def test_temporal_aggregation_uses_true_positions_and_ignores_padding() -> None:
    values = np.asarray(
        [
            [1.0, 100.0, 200.0, 7.0, 999.0, 999.0],
            [np.nan, 5.0, np.nan, np.nan, 999.0, 999.0],
            [np.nan, np.nan, np.nan, np.nan, 999.0, 999.0],
        ]
    )
    positions = np.asarray(
        [
            [1, 2, 3, 4, 0, 0],
            [1, 2, 3, 4, 0, 0],
            [1, 2, 3, 4, 0, 0],
        ],
        dtype=np.int8,
    )
    validity = np.asarray(
        [
            [True, False, False, True, True, True],
            [False, True, False, False, True, True],
            [False, False, False, False, True, True],
        ]
    )
    validity_before = validity.copy()

    aggregate = aggregate_temporal_series(values, positions, validity)
    columns = {name: index for index, name in enumerate(AGGREGATION_NAMES)}

    assert aggregate.values[0, columns["valid_count"]] == 2
    assert aggregate.values[0, columns["mean"]] == pytest.approx(4.0)
    assert aggregate.values[0, columns["std"]] == pytest.approx(3.0)
    assert aggregate.values[0, columns["amplitude"]] == pytest.approx(6.0)
    assert aggregate.values[0, columns["p25"]] == pytest.approx(2.5)
    assert aggregate.values[0, columns["p75"]] == pytest.approx(5.5)
    assert aggregate.values[0, columns["iqr"]] == pytest.approx(3.0)
    assert aggregate.values[0, columns["first"]] == pytest.approx(1.0)
    assert aggregate.values[0, columns["last"]] == pytest.approx(7.0)
    assert aggregate.values[0, columns["first_to_last"]] == pytest.approx(6.0)
    assert aggregate.values[0, columns["slope"]] == pytest.approx(2.0)
    assert aggregate.values[1, columns["valid_count"]] == 1
    assert aggregate.values[1, columns["mean"]] == pytest.approx(5.0)
    for name in ("std", "amplitude", "iqr", "first_to_last", "slope"):
        assert np.isnan(aggregate.values[1, columns[name]])
    assert aggregate.values[2, columns["valid_count"]] == 0
    assert np.isnan(aggregate.values[2, 1:]).all()
    assert not np.isinf(aggregate.values).any()
    np.testing.assert_array_equal(validity, validity_before)

    constant = aggregate_temporal_series(
        np.asarray([[5.0, 5.0, np.nan, np.nan, 1.0e308, 1.0e308]]),
        np.asarray([[1, 4, 0, 0, 0, 0]], dtype=np.int8),
        np.asarray([[True, True, False, False, True, True]]),
    ).values[0]
    assert constant[columns["std"]] == 0.0
    assert constant[columns["amplitude"]] == 0.0
    assert constant[columns["slope"]] == 0.0
    assert not np.isinf(constant).any()


def test_representations_preserve_masks_metadata_schema_and_folds() -> None:
    windows = _feature_windows()
    values_before = windows.values.copy()
    optical_before = windows.optical_mask.copy()
    manifest_before = windows.manifest.copy(deep=True)

    tabular, sequence = build_feature_representations(windows, FeatureConfig())
    repeated_tabular, repeated_sequence = build_feature_representations(windows, FeatureConfig())

    assert tabular.features.shape == (3, 688)
    assert len(tabular.feature_names) == 688
    assert sequence.radar_values.shape == (3, 6, 8)
    assert sequence.optical_values.shape == (3, 6, 10)
    assert sequence.monthly_indices.shape == (3, 6, 14)
    assert sequence.absolute_month_encoding.shape == (3, 6, 2)
    assert sequence.radar_feature_mask.shape == sequence.radar_values.shape
    assert sequence.optical_band_mask.shape == sequence.optical_values.shape
    assert sequence.raw_band_mask.shape == (3, 6, 12)
    assert sequence.raw_band_names == BANDS
    assert sequence.index_mask.shape == sequence.monthly_indices.shape
    assert sequence.padding_mask[0].tolist() == [False, False, False, False, True, True]
    assert sequence.relative_positions[0].tolist() == [1, 2, 3, 4, 0, 0]
    assert sequence.calendar_months[0].tolist() == [3, 4, 5, 6, 0, 0]
    np.testing.assert_allclose(
        sequence.absolute_month_encoding[0, 0],
        [np.sin(np.pi / 3.0), np.cos(np.pi / 3.0)],
    )
    assert not sequence.radar_mask[0, 4:].any()
    assert not sequence.optical_mask[0, 1]
    assert sequence.optical_band_mask[0, 1, OPTICAL_BANDS.index("green")]
    assert not sequence.optical_band_mask[0, 1, OPTICAL_BANDS.index("blue")]
    ndvi_index = sequence.index_feature_names.index("optical__ndvi")
    assert sequence.index_mask[0, 1, ndvi_index]
    assert np.isnan(sequence.radar_values[sequence.padding_mask]).all()
    assert np.isnan(sequence.optical_values[sequence.padding_mask]).all()
    assert np.isnan(sequence.monthly_indices[sequence.padding_mask]).all()
    assert np.isnan(sequence.absolute_month_encoding[sequence.padding_mask]).all()
    assert not sequence.padding_mask[1, 1]
    assert not sequence.radar_mask[1, 1]
    assert not sequence.raw_band_mask[1, 1, BANDS.index("VV")]
    assert sequence.radar_mask[1, 2]
    assert not sequence.optical_mask[1, 2]

    row = tabular.features.iloc[1]
    assert row["metadata__window_length"] == 5
    assert row["metadata__optical_valid_count"] == 2
    assert row["metadata__optical_gap_count"] == 2
    assert row["metadata__longest_optical_missing_run"] == 2
    assert row["metadata__radar_valid_count"] == 4
    assert row["metadata__radar_valid_proportion"] == pytest.approx(0.8)
    assert row["metadata__blue_valid_count"] == 2
    assert row["metadata__green_valid_count"] == 3
    assert row["relative__position_01__radar__vv"] == pytest.approx(-3.0)
    assert np.isnan(row["relative__position_02__radar__vv"])
    assert np.isnan(tabular.features.loc[0, "relative__position_05__radar__vv"])
    assert tabular.feature_names[:7] == (
        "relative__position_01__radar__vv",
        "relative__position_02__radar__vv",
        "relative__position_03__radar__vv",
        "relative__position_04__radar__vv",
        "relative__position_05__radar__vv",
        "relative__position_06__radar__vv",
        "relative__position_01__radar__vh",
    )
    assert tabular.original_ids.tolist() == ["original-a", "original-a", "original-b"]
    assert tabular.folds.tolist() == [0, 0, 1]
    assert tabular.labels is not None and tabular.labels.tolist() == [1, 1, 0]
    assert tabular.fingerprint == repeated_tabular.fingerprint
    assert sequence.fingerprint == repeated_sequence.fingerprint
    assert tabular.schema_fingerprint == repeated_tabular.schema_fingerprint
    assert sequence.schema_fingerprint == repeated_sequence.schema_fingerprint
    assert not any(pd.api.types.is_object_dtype(dtype) for dtype in tabular.features.dtypes)
    assert not np.isinf(tabular.features.to_numpy()).any()
    assert not any(
        {"id", "original_id", "window_id", "fold", "label", "target"}
        & {part.casefold() for part in name.split("__")}
        for name in tabular.feature_names
    )
    assert windows.manifest.groupby("original_id")["fold"].nunique().eq(1).all()
    ndwi_definition = next(
        definition
        for definition in tabular.registry.definitions
        if definition.name == "optical__ndwi__mean"
    )
    assert ndwi_definition.source_bands == ("green", "nir")
    assert ndwi_definition.formula == "(Green - NIR) / (Green + NIR)"
    assert ndwi_definition.temporal_aggregation == "mean"
    np.testing.assert_allclose(windows.values, values_before, equal_nan=True)
    np.testing.assert_array_equal(windows.optical_mask, optical_before)
    pd.testing.assert_frame_equal(windows.manifest, manifest_before)


def test_train_test_representations_align_without_labels_as_features() -> None:
    train_windows = _feature_windows()
    test_manifest = train_windows.manifest.drop(columns="label").copy(deep=True)
    test_manifest["ID"] = ["test-a", "test-b", "test-c"]
    test_manifest["original_id"] = test_manifest["ID"]
    test_manifest["window_id"] = ["test-window-a", "test-window-b", "test-window-c"]
    test_manifest["fold"] = np.int16(-1)
    test_windows = replace(train_windows, manifest=test_manifest)

    train_tabular, train_sequence = build_feature_representations(train_windows, FeatureConfig())
    test_tabular, test_sequence = build_feature_representations(test_windows, FeatureConfig())

    assert_feature_schema_alignment(
        train_tabular,
        test_tabular,
        train_sequence,
        test_sequence,
    )
    audit = build_feature_audit(
        train_tabular,
        test_tabular,
        train_sequence,
        test_sequence,
        deterministic_rebuild=True,
        input_windows_unchanged=True,
    )
    assert train_tabular.feature_names == test_tabular.feature_names
    assert train_tabular.schema_fingerprint == test_tabular.schema_fingerprint
    assert train_sequence.schema_fingerprint == test_sequence.schema_fingerprint
    assert test_tabular.labels is None
    assert test_sequence.labels is None
    assert audit.summary["tabular"]["feature_count"] == 688
    assert audit.summary["registry"]["definition_count"] == 739
    assert audit.summary["safety"]["deterministic_rebuild"] is True

    incompatible_tabular, incompatible_sequence = build_feature_representations(
        test_windows,
        FeatureConfig(version="phase3_incompatible"),
    )
    with pytest.raises(FeatureEngineeringError, match="schema fingerprints"):
        assert_feature_schema_alignment(
            train_tabular,
            incompatible_tabular,
            train_sequence,
            incompatible_sequence,
        )


def test_semantic_band_mapping_rejects_ambiguity_and_missing_roles() -> None:
    windows = _feature_windows()
    unavailable = FeatureConfig(bands=replace(BandSemanticMapping(), green="coastal"))

    with pytest.raises(ConfigError, match="unambiguously"):
        FeatureConfig(bands=replace(BandSemanticMapping(), green="red"))
    with pytest.raises(FeatureEngineeringError, match="unavailable bands"):
        build_monthly_features(windows, unavailable)


def _write_cli_fixture(root: Path) -> Path:
    train = pd.DataFrame(index=range(4), columns=FEATURE_COLUMNS, dtype=float)
    for row in range(train.shape[0]):
        for month in range(1, 13):
            for band_index, band in enumerate(BANDS):
                train.loc[row, f"{band}_{month:02d}"] = 10.0 + row + month + band_index / 10
    train.insert(0, "label", [0, 1, 0, 1])
    train.insert(0, "ID", [f"train-{row:03d}" for row in range(4)])

    test = pd.DataFrame(-9999.0, index=range(3), columns=FEATURE_COLUMNS)
    for row, (start, length) in enumerate(((1, 4), (4, 5), (7, 6))):
        for month in range(start, start + length):
            for band_index, band in enumerate(BANDS):
                test.loc[row, f"{band}_{month:02d}"] = 20.0 + row + month + band_index / 10
    test.loc[0, [f"{band}_02" for band in OPTICAL_BANDS]] = -9999.0
    test.loc[2, "blue_09"] = -9999.0
    test.insert(0, "ID", [f"test-{row:03d}" for row in range(3)])
    sample = pd.DataFrame({"ID": test["ID"], "TargetF1": [0, 0, 0], "TargetRAUC": [0.0, 0.0, 0.0]})
    train.to_csv(root / "Train.csv", index=False)
    test.to_csv(root / "Test.csv", index=False)
    sample.to_csv(root / "SampleSubmission.csv", index=False)
    config = """\
project:
  name: synthetic-features
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
  windows_per_sample: 2
features:
  version: phase3_test
  epsilon: 0.000001
  bands:
    vv: VV
    vh: VH
    blue: blue
    green: green
    red: red
    red_edge_1: re1
    red_edge_2: re2
    red_edge_3: re3
    nir: nir
    narrow_nir: nira
    swir1: swir1
    swir2: swir2
reporting:
  artifacts_dir: artifacts
"""
    path = root / "base.yaml"
    path.write_text(config, encoding="utf-8")
    return path


def test_feature_cli_writes_only_aggregate_audit_artifacts(tmp_path: Path) -> None:
    config = _write_cli_fixture(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_features.py"),
        "--config",
        str(config),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = tmp_path / "artifacts" / "features"
    expected = {
        "feature_counts_by_group.csv",
        "feature_registry.csv",
        "feature_summary.json",
        "report.md",
        "run_metadata.json",
        "sequence_mask_summary.csv",
        "tabular_missingness_by_group.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "feature_summary.json").read_text(encoding="utf-8"))
    assert summary["tabular"]["train_shape"] == [8, 688]
    assert summary["tabular"]["test_shape"] == [3, 688]
    assert summary["sequence"]["radar_channels"] == 8
    assert summary["sequence"]["optical_channels"] == 10
    assert summary["sequence"]["index_channels"] == 14
    assert summary["windows"]["test_length_distribution"] == {"4": 1, "5": 1, "6": 1}
    assert summary["safety"]["raw_rows_persisted"] is False
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.suffix != ".json"
    )
    assert "train-000" not in artifact_text
    assert "test-000" not in artifact_text


def test_config_loader_rejects_ambiguous_semantic_band_mapping(tmp_path: Path) -> None:
    config_path = _write_cli_fixture(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("green: green", "green: red"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unambiguously"):
        load_project_config(config_path)
