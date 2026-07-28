from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoai_aquaculture.constants import OPTICAL_BANDS, RADAR_BANDS
from geoai_aquaculture.data import (
    ConfigError,
    SchemaError,
    audit_competition_data,
    load_competition_data,
    load_project_config,
    parse_temporal_column,
    validate_competition_schema,
    write_audit_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANDS = RADAR_BANDS + OPTICAL_BANDS
FEATURE_COLUMNS = tuple(f"{band}_{month:02d}" for month in range(1, 13) for band in BANDS)


def _config_text() -> str:
    return """\
project:
  name: synthetic-geoai
  seed: 123
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
reporting:
  artifacts_dir: artifacts
"""


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        np.arange(4 * len(FEATURE_COLUMNS), dtype=float).reshape(4, -1) + 1.0,
        columns=FEATURE_COLUMNS,
    )
    train.insert(0, "label", [0, 1, 0, 1])
    train.insert(0, "ID", ["train-001", "train-002", "train-003", "train-004"])

    test = pd.DataFrame(-9999.0, index=range(3), columns=FEATURE_COLUMNS)
    windows = ((1, 4), (4, 5), (7, 6))
    for row, (start, length) in enumerate(windows):
        for month in range(start, start + length):
            for band in BANDS:
                test.loc[row, f"{band}_{month:02d}"] = 1000.0 + row * 100 + month
    for band in OPTICAL_BANDS:
        test.loc[0, f"{band}_02"] = -9999.0
    test.insert(0, "ID", ["test-001", "test-002", "test-003"])

    sample = pd.DataFrame(
        {
            "ID": ["test-001", "test-002", "test-003"],
            "TargetF1": [0, 0, 0],
            "TargetRAUC": [0.0, 0.0, 0.0],
        }
    )
    return train, test, sample


def _write_fixture(
    root: Path,
    mutate: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], None] | None = None,
) -> Path:
    train, test, sample = _synthetic_frames()
    if mutate is not None:
        mutate(train, test, sample)
    train.to_csv(root / "Train.csv", index=False)
    test.to_csv(root / "Test.csv", index=False)
    sample.to_csv(root / "SampleSubmission.csv", index=False)
    config_path = root / "base.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    return config_path


def test_load_converts_sentinel_and_preserves_source_order(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    source_before = (tmp_path / "Test.csv").read_bytes()

    config = load_project_config(config_path)
    data = load_competition_data(config)

    assert config.project_root == tmp_path
    assert data.feature_columns == FEATURE_COLUMNS
    assert [item.position for item in data.temporal_columns] == list(range(144))
    assert data.train.columns.tolist() == ["ID", "label", *FEATURE_COLUMNS]
    assert data.test.columns.tolist() == ["ID", *FEATURE_COLUMNS]
    assert str(data.test["ID"].dtype) == "string"
    assert data.raw_missing_counts == {"train": 0, "test": 262}
    assert pd.isna(data.test.loc[0, "VH_05"])
    assert pd.isna(data.test.loc[0, "blue_02"])
    assert data.test.loc[0, "VH_01"] == pytest.approx(1001.0)
    assert (tmp_path / "Test.csv").read_bytes() == source_before


def test_temporal_parser_rejects_unknown_band_and_month(tmp_path: Path) -> None:
    config = load_project_config(_write_fixture(tmp_path)).data

    parsed = parse_temporal_column("VH_09", config, position=17)
    assert (parsed.band, parsed.month, parsed.sensor, parsed.position) == (
        "VH",
        9,
        "radar",
        17,
    )
    with pytest.raises(SchemaError, match="unknown band"):
        parse_temporal_column("temperature_01", config)
    with pytest.raises(SchemaError, match="month 13"):
        parse_temporal_column("VH_13", config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda train, _test, _sample: train.__setitem__("ID", ["x", "x", "y", "z"]), "unique"),
        (lambda train, _test, _sample: train.__setitem__("label", 1), "both binary classes"),
        (
            lambda _train, test, _sample: test.rename(columns={"VH_01": "VH_13"}, inplace=True),
            "ordering do not match",
        ),
        (
            lambda train, _test, _sample: train.__setitem__("VH_01", np.nan),
            "NaN or infinite",
        ),
        (
            lambda _train, _test, sample: sample.__setitem__(
                "ID", ["test-002", "test-001", "test-003"]
            ),
            "IDs and order",
        ),
    ],
)
def test_schema_rejects_malformed_frames(
    tmp_path: Path,
    mutate: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], None],
    message: str,
) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_project_config(config_path).data
    train, test, sample = _synthetic_frames()
    mutate(train, test, sample)

    with pytest.raises(SchemaError, match=message):
        validate_competition_schema(train, test, sample, config)


def test_config_rejects_wrong_month_count(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(_config_text().replace("months: 12", "months: 11"), encoding="utf-8")

    with pytest.raises(ConfigError, match="must equal 12"):
        load_project_config(config_path)


def test_audit_detects_windows_optical_gaps_and_missingness(tmp_path: Path) -> None:
    data = load_competition_data(load_project_config(_write_fixture(tmp_path)))

    audit = audit_competition_data(data)
    summary = audit.summary
    window_rows = audit.tables["test_window_rows"]

    assert summary["datasets"]["train"]["target_counts"] == {"0": 2, "1": 2}
    assert summary["test_windows"]["length_distribution"] == {"4": 1, "5": 1, "6": 1}
    assert summary["test_windows"]["all_radar_windows_consecutive"] is True
    assert summary["test_windows"]["rows_with_optical_gaps"] == 1
    assert summary["test_windows"]["optical_gap_count_distribution"] == {"0": 2, "1": 1}
    assert summary["test_windows"]["radar_partial_months"] == 0
    assert summary["test_windows"]["optical_partial_months"] == 0
    assert window_rows["window_start"].tolist() == [1, 4, 7]
    assert window_rows["window_end"].tolist() == [4, 8, 12]
    assert window_rows["window_length"].tolist() == [4, 5, 6]
    assert window_rows["optical_valid_months_inside_window"].tolist() == [3, 5, 6]


def test_audit_reports_partial_sensor_month_without_hiding_it(tmp_path: Path) -> None:
    def make_partial(_train: pd.DataFrame, test: pd.DataFrame, _sample: pd.DataFrame) -> None:
        test.loc[1, "blue_04"] = -9999.0

    data = load_competition_data(load_project_config(_write_fixture(tmp_path, make_partial)))

    summary = audit_competition_data(data).summary

    assert summary["test_windows"]["optical_partial_months"] == 1
    assert summary["test_windows"]["rows_with_optical_gaps"] == 2


def test_audit_scientific_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    data = load_competition_data(load_project_config(_write_fixture(tmp_path)))
    audit_one = audit_competition_data(data)
    audit_two = audit_competition_data(data)
    output_one = tmp_path / "audit-one"
    output_two = tmp_path / "audit-two"

    paths_one = write_audit_artifacts(audit_one, output_one)
    paths_two = write_audit_artifacts(audit_two, output_two)

    assert [path.name for path in paths_one] == [path.name for path in paths_two]
    assert {path.name: path.read_bytes() for path in paths_one} == {
        path.name: path.read_bytes() for path in paths_two
    }


def test_audit_cli_fails_clearly_on_malformed_schema(tmp_path: Path) -> None:
    def remove_feature(train: pd.DataFrame, _test: pd.DataFrame, _sample: pd.DataFrame) -> None:
        train.drop(columns="VH_01", inplace=True)

    config_path = _write_fixture(tmp_path, remove_feature)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_data.py"),
            "--config",
            str(config_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "data audit failed" in result.stderr
    assert "feature columns or ordering do not match" in result.stderr
