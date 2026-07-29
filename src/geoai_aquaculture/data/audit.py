"""Deterministic missingness and temporal-window auditing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loading import CompetitionData


@dataclass(frozen=True, slots=True)
class DataAudit:
    """Machine-readable audit summary and normalized tabular details."""

    summary: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]
    report_markdown: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def git_provenance(project_root: Path) -> dict[str, Any]:
    """Return the current Git commit and tracked-file dirtiness when available."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_dirty": None}
    return {"commit": commit, "tracked_files_dirty": bool(status.strip())}


def _sensor_month_masks(
    frame: pd.DataFrame,
    bands: tuple[str, ...],
    months: int,
    column_lookup: Mapping[tuple[str, int], str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_valid: dict[int, pd.Series] = {}
    any_valid: dict[int, pd.Series] = {}
    for month in range(1, months + 1):
        columns = [column_lookup[(band, month)] for band in bands]
        validity = frame.loc[:, columns].notna()
        all_valid[month] = validity.all(axis=1)
        any_valid[month] = validity.any(axis=1)
    return pd.DataFrame(all_valid), pd.DataFrame(any_valid)


def sensor_month_availability(
    frame: pd.DataFrame, data: CompetitionData
) -> dict[str, pd.DataFrame]:
    """Return Phase 1 all-band and any-band sensor availability by calendar month."""

    lookup = {(item.band, item.month): item.name for item in data.temporal_columns}
    radar_all, radar_any = _sensor_month_masks(
        frame, data.config.data.radar_bands, data.config.data.months, lookup
    )
    optical_all, optical_any = _sensor_month_masks(
        frame, data.config.data.optical_bands, data.config.data.months, lookup
    )
    return {
        "radar_all": radar_all,
        "radar_any": radar_any,
        "optical_all": optical_all,
        "optical_any": optical_any,
    }


def _missingness_tables(data: CompetitionData) -> dict[str, pd.DataFrame]:
    config = data.config.data
    metadata = data.temporal_columns
    feature_columns = list(data.feature_columns)
    column_by_band = {
        band: [item.name for item in metadata if item.band == band] for band in config.bands
    }
    column_by_month = {
        month: [item.name for item in metadata if item.month == month]
        for month in range(1, config.months + 1)
    }
    column_by_sensor = {
        "radar": [item.name for item in metadata if item.sensor == "radar"],
        "optical": [item.name for item in metadata if item.sensor == "optical"],
    }

    band_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    row_frames: list[pd.DataFrame] = []

    for dataset, frame in (("train", data.train), ("test", data.test)):
        availability = sensor_month_availability(frame, data)
        for sensor, bands in (
            ("radar", config.radar_bands),
            ("optical", config.optical_bands),
        ):
            for band in bands:
                columns = column_by_band[band]
                missing = int(frame.loc[:, columns].isna().sum().sum())
                observations = frame.shape[0] * len(columns)
                band_rows.append(
                    {
                        "dataset": dataset,
                        "sensor": sensor,
                        "band": band,
                        "missing_count": missing,
                        "observation_count": observations,
                        "missing_rate": missing / observations,
                    }
                )
        for month, columns in column_by_month.items():
            missing = int(frame.loc[:, columns].isna().sum().sum())
            observations = frame.shape[0] * len(columns)
            month_rows.append(
                {
                    "dataset": dataset,
                    "month": month,
                    "missing_count": missing,
                    "observation_count": observations,
                    "missing_rate": missing / observations,
                }
            )
        for sensor, columns in column_by_sensor.items():
            missing = int(frame.loc[:, columns].isna().sum().sum())
            observations = frame.shape[0] * len(columns)
            sensor_rows.append(
                {
                    "dataset": dataset,
                    "sensor": sensor,
                    "missing_count": missing,
                    "observation_count": observations,
                    "missing_rate": missing / observations,
                }
            )

        missing_by_row = frame.loc[:, feature_columns].isna()
        radar_missing = frame.loc[:, column_by_sensor["radar"]].isna().sum(axis=1)
        optical_missing = frame.loc[:, column_by_sensor["optical"]].isna().sum(axis=1)
        row_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    config.id_column: frame[config.id_column].astype("string"),
                    "missing_count": missing_by_row.sum(axis=1).astype(int),
                    "feature_count": len(feature_columns),
                    "missing_rate": missing_by_row.mean(axis=1),
                    "radar_missing_count": radar_missing.astype(int),
                    "optical_missing_count": optical_missing.astype(int),
                    "radar_valid_months": availability["radar_all"].sum(axis=1).astype(int),
                    "optical_valid_months": availability["optical_all"].sum(axis=1).astype(int),
                    "radar_partial_months": (availability["radar_any"] ^ availability["radar_all"])
                    .sum(axis=1)
                    .astype(int),
                    "optical_partial_months": (
                        availability["optical_any"] ^ availability["optical_all"]
                    )
                    .sum(axis=1)
                    .astype(int),
                }
            )
        )

    return {
        "missingness_by_band": pd.DataFrame(band_rows),
        "missingness_by_month": pd.DataFrame(month_rows),
        "missingness_by_sensor": pd.DataFrame(sensor_rows),
        "missingness_by_row": pd.concat(row_frames, ignore_index=True),
    }


def _test_windows(data: CompetitionData) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    config = data.config.data
    availability = sensor_month_availability(data.test, data)
    radar = availability["radar_all"]
    optical = availability["optical_all"]
    radar_array = radar.to_numpy(dtype=bool)
    has_radar = radar_array.any(axis=1)
    starts = np.where(has_radar, radar_array.argmax(axis=1) + 1, 0)
    ends = np.where(
        has_radar,
        config.months - np.flip(radar_array, axis=1).argmax(axis=1),
        0,
    )
    lengths = radar.sum(axis=1).to_numpy(dtype=int)
    consecutive = has_radar & (lengths == ends - starts + 1)
    optical_inside = (radar & optical).sum(axis=1).to_numpy(dtype=int)
    cloud_gaps = (radar & ~optical).sum(axis=1).to_numpy(dtype=int)
    optical_outside = (~radar & availability["optical_any"]).sum(axis=1).to_numpy(dtype=int)

    rows = pd.DataFrame(
        {
            config.id_column: data.test[config.id_column].astype("string"),
            "window_start": pd.array(np.where(has_radar, starts, None), dtype="Int64"),
            "window_end": pd.array(np.where(has_radar, ends, None), dtype="Int64"),
            "window_length": lengths,
            "radar_valid_months": lengths,
            "optical_valid_months_inside_window": optical_inside,
            "cloud_gap_count": cloud_gaps,
            "optical_valid_months_outside_window": optical_outside,
            "radar_is_consecutive": consecutive,
            "radar_partial_months": (availability["radar_any"] ^ availability["radar_all"])
            .sum(axis=1)
            .astype(int),
            "optical_partial_months": (availability["optical_any"] ^ availability["optical_all"])
            .sum(axis=1)
            .astype(int),
            "radar_availability": radar.apply(
                lambda row: "".join(row.astype(int).astype(str)), axis=1
            ),
            "optical_availability": optical.apply(
                lambda row: "".join(row.astype(int).astype(str)), axis=1
            ),
        }
    )

    length_distribution = (
        rows.groupby("window_length", dropna=False, sort=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    length_distribution["row_share"] = length_distribution["row_count"] / len(rows)
    start_distribution = (
        rows.groupby(["window_start", "window_length"], dropna=False, sort=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    gap_distribution = (
        rows.groupby(["window_length", "cloud_gap_count"], dropna=False, sort=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    return rows, {
        "test_window_distribution": length_distribution,
        "test_window_start_distribution": start_distribution,
        "test_optical_gap_distribution": gap_distribution,
    }


def _file_provenance(data: CompetitionData) -> dict[str, dict[str, Any]]:
    config = data.config
    paths = {
        "train": config.data.train_path,
        "test": config.data.test_path,
        "sample_submission": config.data.sample_submission_path,
    }
    return {
        dataset: {
            "path": _display_path(path, config.project_root),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for dataset, path in paths.items()
    }


def _dataset_summary(data: CompetitionData, dataset: str, frame: pd.DataFrame) -> dict[str, Any]:
    missing = int(frame.loc[:, list(data.feature_columns)].isna().sum().sum())
    observations = frame.shape[0] * len(data.feature_columns)
    result: dict[str, Any] = {
        "rows": frame.shape[0],
        "columns": frame.shape[1],
        "feature_columns": len(data.feature_columns),
        "missing_cells": missing,
        "missing_rate": missing / observations,
        "rows_with_missing_features": int(
            frame.loc[:, list(data.feature_columns)].isna().any(axis=1).sum()
        ),
        "raw_sentinel_cells": data.raw_missing_counts[dataset],
    }
    if dataset == "train":
        counts = frame[data.config.data.target_column].value_counts().sort_index()
        result["target_counts"] = {str(int(label)): int(count) for label, count in counts.items()}
        result["positive_rate"] = float(frame[data.config.data.target_column].mean())
    return result


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _markdown_report(summary: Mapping[str, Any]) -> str:
    train = summary["datasets"]["train"]
    test = summary["datasets"]["test"]
    windows = summary["test_windows"]
    lines = [
        "# GeoAI competition data audit",
        "",
        f"- Train: {train['rows']} rows, {train['feature_columns']} temporal features, "
        f"target counts {train['target_counts']}.",
        f"- Test: {test['rows']} rows and {test['missing_cells']} sentinel-derived missing cells.",
        f"- Radar window lengths: {windows['length_distribution']}.",
        f"- All non-empty radar windows consecutive: {windows['all_radar_windows_consecutive']}.",
        f"- Rows with optical gaps inside radar windows: {windows['rows_with_optical_gaps']}.",
        f"- Radar partial sensor-months: {windows['radar_partial_months']}; "
        f"optical partial sensor-months: {windows['optical_partial_months']}.",
        f"- Optical observations outside radar windows: "
        f"{windows['optical_valid_months_outside_windows']} row-months.",
        "",
        "Missing values in loaded frames are `NaN`; the source CSV files remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def audit_competition_data(data: CompetitionData) -> DataAudit:
    """Compute schema, missingness, and test-window diagnostics without model fitting."""

    missingness_tables = _missingness_tables(data)
    test_window_rows, window_tables = _test_windows(data)
    temporal_columns = pd.DataFrame(
        [
            {
                "position": item.position,
                "column": item.name,
                "band": item.band,
                "month": item.month,
                "sensor": item.sensor,
            }
            for item in data.temporal_columns
        ]
    )
    length_distribution = window_tables["test_window_distribution"]
    windows = {
        str(int(row["window_length"])): int(row["row_count"])
        for row in _records(length_distribution)
    }
    summary: dict[str, Any] = {
        "audit_schema_version": 1,
        "project": data.config.project_name,
        "seed": data.config.seed,
        "schema": {
            "id_column": data.config.data.id_column,
            "target_column": data.config.data.target_column,
            "missing_sentinel": data.config.data.missing_sentinel,
            "months": data.config.data.months,
            "radar_bands": list(data.config.data.radar_bands),
            "optical_bands": list(data.config.data.optical_bands),
            "temporal_feature_count": len(data.temporal_columns),
            "sample_submission_columns": list(data.sample_submission.columns),
            "sample_ids_match_test_order": bool(
                data.sample_submission[data.config.data.id_column].equals(
                    data.test[data.config.data.id_column]
                )
            ),
        },
        "datasets": {
            "train": _dataset_summary(data, "train", data.train),
            "test": _dataset_summary(data, "test", data.test),
        },
        "test_windows": {
            "length_distribution": windows,
            "start_length_distribution": _records(window_tables["test_window_start_distribution"]),
            "all_radar_windows_consecutive": bool(test_window_rows["radar_is_consecutive"].all()),
            "nonconsecutive_radar_windows": int((~test_window_rows["radar_is_consecutive"]).sum()),
            "rows_with_optical_gaps": int((test_window_rows["cloud_gap_count"] > 0).sum()),
            "optical_gap_count_distribution": {
                str(int(gap)): int(count)
                for gap, count in test_window_rows["cloud_gap_count"]
                .value_counts()
                .sort_index()
                .items()
            },
            "radar_partial_months": int(test_window_rows["radar_partial_months"].sum()),
            "optical_partial_months": int(test_window_rows["optical_partial_months"].sum()),
            "optical_valid_months_outside_windows": int(
                test_window_rows["optical_valid_months_outside_window"].sum()
            ),
            "distinct_joint_availability_patterns": int(
                test_window_rows[["radar_availability", "optical_availability"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "source_files": _file_provenance(data),
        "provenance": {
            "config_path": _display_path(data.config.source_path, data.config.project_root),
            "config_sha256": _sha256(data.config.source_path),
            "git": git_provenance(data.config.project_root),
        },
    }
    tables = {
        "temporal_columns": temporal_columns,
        **missingness_tables,
        "test_window_rows": test_window_rows,
        **window_tables,
    }
    return DataAudit(summary=summary, tables=tables, report_markdown=_markdown_report(summary))


def write_audit_artifacts(audit: DataAudit, output_dir: Path) -> tuple[Path, ...]:
    """Write stable JSON, CSV, and Markdown audit artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(
        json.dumps(audit.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(summary_path)
    for name, table in audit.tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        written.append(path)
    report_path = output_dir / "report.md"
    report_path.write_text(audit.report_markdown, encoding="utf-8")
    written.append(report_path)
    return tuple(written)
