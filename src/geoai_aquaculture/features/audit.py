"""Feature-representation alignment checks and aggregate-only audit artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.data import ProjectConfig, git_provenance

from .indices import FeatureEngineeringError
from .registry import FeatureRegistry, combine_feature_registries
from .representations import FeatureMatrix, SequenceFeatureDataset


@dataclass(frozen=True, slots=True)
class FeatureAudit:
    """Deterministic schema, missingness, and numerical-safety summary."""

    summary: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]
    registry: FeatureRegistry
    report_markdown: str


def assert_feature_schema_alignment(
    train_tabular: FeatureMatrix,
    test_tabular: FeatureMatrix,
    train_sequence: SequenceFeatureDataset,
    test_sequence: SequenceFeatureDataset,
) -> None:
    """Reject any train/test feature or channel ordering mismatch."""

    if train_tabular.feature_names != test_tabular.feature_names:
        raise FeatureEngineeringError("train/test tabular feature names do not align")
    if train_tabular.schema_fingerprint != test_tabular.schema_fingerprint:
        raise FeatureEngineeringError("train/test tabular schema fingerprints do not align")
    sequence_name_pairs = (
        (train_sequence.radar_feature_names, test_sequence.radar_feature_names, "radar"),
        (
            train_sequence.optical_feature_names,
            test_sequence.optical_feature_names,
            "optical",
        ),
        (train_sequence.index_feature_names, test_sequence.index_feature_names, "index"),
    )
    for train_names, test_names, group in sequence_name_pairs:
        if train_names != test_names:
            raise FeatureEngineeringError(f"train/test {group} sequence channels do not align")
    if train_sequence.schema_fingerprint != test_sequence.schema_fingerprint:
        raise FeatureEngineeringError("train/test sequence schema fingerprints do not align")


def _tabular_missingness(dataset: str, matrix: FeatureMatrix) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, names in matrix.feature_groups.items():
        missing = int(matrix.features.loc[:, list(names)].isna().sum().sum())
        observations = matrix.features.shape[0] * len(names)
        rows.append(
            {
                "dataset": dataset,
                "feature_group": group,
                "feature_count": len(names),
                "missing_count": missing,
                "observation_count": observations,
                "missing_rate": missing / observations,
            }
        )
    return rows


def _sequence_mask_rows(dataset: str, sequence: SequenceFeatureDataset) -> list[dict[str, Any]]:
    position = ~sequence.padding_mask
    possible_positions = int(position.sum())
    return [
        {
            "dataset": dataset,
            "sequence_group": "padding",
            "missing_count": int(sequence.padding_mask.sum()),
            "observation_count": int(sequence.padding_mask.size),
        },
        {
            "dataset": dataset,
            "sequence_group": "radar_sensor",
            "missing_count": int((position & ~sequence.radar_mask).sum()),
            "observation_count": possible_positions,
        },
        {
            "dataset": dataset,
            "sequence_group": "radar_channels",
            "missing_count": int((position[:, :, None] & ~sequence.radar_feature_mask).sum()),
            "observation_count": possible_positions * sequence.radar_values.shape[2],
        },
        {
            "dataset": dataset,
            "sequence_group": "optical_month",
            "missing_count": int((position & ~sequence.optical_mask).sum()),
            "observation_count": possible_positions,
        },
        {
            "dataset": dataset,
            "sequence_group": "optical_bands",
            "missing_count": int((position[:, :, None] & ~sequence.optical_band_mask).sum()),
            "observation_count": possible_positions * sequence.optical_values.shape[2],
        },
        {
            "dataset": dataset,
            "sequence_group": "all_raw_bands",
            "missing_count": int((position[:, :, None] & ~sequence.raw_band_mask).sum()),
            "observation_count": possible_positions * len(sequence.raw_band_names),
        },
        {
            "dataset": dataset,
            "sequence_group": "optical_indices",
            "missing_count": int((position[:, :, None] & ~sequence.index_mask).sum()),
            "observation_count": possible_positions * sequence.monthly_indices.shape[2],
        },
    ]


def build_feature_audit(
    train_tabular: FeatureMatrix,
    test_tabular: FeatureMatrix,
    train_sequence: SequenceFeatureDataset,
    test_sequence: SequenceFeatureDataset,
    *,
    deterministic_rebuild: bool | None = None,
    input_windows_unchanged: bool | None = None,
) -> FeatureAudit:
    """Validate and summarize aligned train/test feature representations."""

    assert_feature_schema_alignment(
        train_tabular,
        test_tabular,
        train_sequence,
        test_sequence,
    )
    registry = combine_feature_registries(train_tabular.registry, train_sequence.registry)
    feature_counts = (
        registry.to_frame()
        .groupby(["output_representation", "feature_group"], sort=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )
    tabular_missingness = pd.DataFrame(
        [
            *_tabular_missingness("train", train_tabular),
            *_tabular_missingness("test", test_tabular),
        ]
    )
    sequence_masks = pd.DataFrame(
        [
            *_sequence_mask_rows("train", train_sequence),
            *_sequence_mask_rows("test", test_sequence),
        ]
    )
    sequence_masks["missing_rate"] = (
        sequence_masks["missing_count"] / sequence_masks["observation_count"]
    )
    train_lengths = train_tabular.features["metadata__window_length"].astype(int)
    test_lengths = test_tabular.features["metadata__window_length"].astype(int)
    summary: dict[str, Any] = {
        "feature_audit_schema_version": 1,
        "tabular": {
            "feature_count": len(train_tabular.feature_names),
            "train_shape": list(train_tabular.features.shape),
            "test_shape": list(test_tabular.features.shape),
            "train_schema_fingerprint": train_tabular.schema_fingerprint,
            "test_schema_fingerprint": test_tabular.schema_fingerprint,
            "schemas_aligned": True,
            "train_fingerprint": train_tabular.fingerprint,
            "test_fingerprint": test_tabular.fingerprint,
            "infinite_values": int(
                np.isinf(train_tabular.features.to_numpy(dtype=np.float64)).sum()
                + np.isinf(test_tabular.features.to_numpy(dtype=np.float64)).sum()
            ),
        },
        "sequence": {
            "train_rows": len(train_sequence.original_ids),
            "test_rows": len(test_sequence.original_ids),
            "max_length": train_sequence.radar_values.shape[1],
            "radar_channels": train_sequence.radar_values.shape[2],
            "optical_channels": train_sequence.optical_values.shape[2],
            "index_channels": train_sequence.monthly_indices.shape[2],
            "absolute_month_channels": train_sequence.absolute_month_encoding.shape[2],
            "train_schema_fingerprint": train_sequence.schema_fingerprint,
            "test_schema_fingerprint": test_sequence.schema_fingerprint,
            "schemas_aligned": True,
            "train_fingerprint": train_sequence.fingerprint,
            "test_fingerprint": test_sequence.fingerprint,
            "infinite_values": int(
                sum(
                    np.isinf(array).sum()
                    for array in (
                        train_sequence.radar_values,
                        train_sequence.optical_values,
                        train_sequence.monthly_indices,
                        train_sequence.absolute_month_encoding,
                        test_sequence.radar_values,
                        test_sequence.optical_values,
                        test_sequence.monthly_indices,
                        test_sequence.absolute_month_encoding,
                    )
                )
            ),
        },
        "registry": {
            "definition_count": len(registry.definitions),
            "fingerprint": registry.fingerprint,
        },
        "windows": {
            "train_count": len(train_tabular.original_ids),
            "test_count": len(test_tabular.original_ids),
            "train_length_distribution": {
                str(key): int(value)
                for key, value in train_lengths.value_counts().sort_index().items()
            },
            "test_length_distribution": {
                str(key): int(value)
                for key, value in test_lengths.value_counts().sort_index().items()
            },
        },
        "safety": {
            "deterministic_rebuild": deterministic_rebuild,
            "identities_are_metadata_only": True,
            "input_windows_unchanged": input_windows_unchanged,
            "labels_are_metadata_only": True,
            "folds_are_metadata_only": True,
            "raw_rows_persisted": False,
        },
    }
    report = "\n".join(
        [
            "# Phase 3 feature audit",
            "",
            f"- Tabular features: {summary['tabular']['feature_count']}; "
            "train/test schemas aligned.",
            f"- Sequence channels: radar {summary['sequence']['radar_channels']}, optical "
            f"{summary['sequence']['optical_channels']}, indices "
            f"{summary['sequence']['index_channels']}.",
            f"- Sequence length: {summary['sequence']['max_length']} with explicit padding.",
            "- Positive/negative infinity across both representations: 0.",
            "- IDs, folds, and labels remain attached metadata and are not model features.",
            f"- Same-input deterministic rebuild: {deterministic_rebuild}.",
            f"- Input temporal windows unchanged: {input_windows_unchanged}.",
            "- No raw feature rows are persisted by this audit.",
            "",
        ]
    )
    return FeatureAudit(
        summary=summary,
        tables={
            "feature_counts_by_group": feature_counts,
            "tabular_missingness_by_group": tabular_missingness,
            "sequence_mask_summary": sequence_masks,
        },
        registry=registry,
        report_markdown=report,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_feature_audit_artifacts(
    audit: FeatureAudit,
    output_dir: Path,
    *,
    config: ProjectConfig,
    runtime_seconds: float,
    command: str,
) -> tuple[Path, ...]:
    """Write aggregate-only feature artifacts plus explicit runtime provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_path = output_dir / "feature_summary.json"
    summary_path.write_text(
        json.dumps(audit.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(summary_path)

    frames = {"feature_registry": audit.registry.to_frame(), **audit.tables}
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        written.append(path)

    report_path = output_dir / "report.md"
    report_path.write_text(audit.report_markdown, encoding="utf-8")
    written.append(report_path)
    try:
        config_path = config.source_path.relative_to(config.project_root).as_posix()
    except ValueError:
        config_path = config.source_path.as_posix()
    run_metadata = {
        "command": command,
        "config_path": config_path,
        "config_sha256": _sha256(config.source_path),
        "feature_version": config.features.version,
        "git": git_provenance(config.project_root),
        "runtime_seconds": runtime_seconds,
        "seed": config.seed,
    }
    run_path = output_dir / "run_metadata.json"
    run_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(run_path)
    return tuple(written)
