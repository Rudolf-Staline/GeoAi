"""Audit and persist temporal-window manifests without model training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WindowGenerationConfig
from .folds import assert_no_fold_leakage
from .masks import MaskLibrary
from .windows import (
    TemporalWindowDataset,
    window_dataset_fingerprint,
    window_view_fingerprint,
)


@dataclass(frozen=True, slots=True)
class WindowAudit:
    """Deterministic temporal-window summary and normalized tables."""

    summary: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]
    report_markdown: str


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def audit_temporal_windows(
    dataset: TemporalWindowDataset,
    generation: WindowGenerationConfig,
    *,
    mask_library: MaskLibrary | None,
    same_seed_reproducible: bool,
    alternate_seed_changes_views: bool | None,
    alternate_seed_preserves_folds: bool | None,
) -> WindowAudit:
    """Summarize generated views and assert their leakage invariants."""

    assert_no_fold_leakage(dataset.manifest)
    manifest = dataset.manifest
    length_distribution = (
        manifest.groupby("window_length", sort=True).size().rename("window_count").reset_index()
    )
    start_distribution = (
        manifest.groupby(["window_length", "window_start"], sort=True)
        .size()
        .rename("window_count")
        .reset_index()
    )
    mask_usage = (
        manifest.groupby(
            ["test_mask_used", "mask_id", "window_length", "window_start"],
            sort=True,
        )
        .size()
        .rename("window_count")
        .reset_index()
    )
    fold_distribution = (
        manifest.groupby(["fold", "window_length"], sort=True)
        .size()
        .rename("window_count")
        .reset_index()
    )
    per_original = manifest.groupby("original_id", sort=False).size()
    used_mask_ids = set(manifest.loc[manifest["test_mask_used"], "mask_id"].tolist())
    available_mask_ids = (
        {template.mask_id for template in mask_library.templates}
        if mask_library is not None
        else set()
    )
    length_counts = {
        str(int(row["window_length"])): int(row["window_count"])
        for row in _records(length_distribution)
    }
    summary: dict[str, Any] = {
        "window_audit_schema_version": 1,
        "generation": {
            "mode": generation.mode,
            "seed": int(manifest["augmentation_seed"].iloc[0]),
            "use_test_missingness_masks": generation.use_test_missingness_masks,
            "windows_per_sample": generation.windows_per_sample,
            "temporal_dropout_enabled": generation.temporal_dropout_enabled,
            "temporal_dropout_probability": generation.temporal_dropout_probability,
            "optical_dropout_enabled": generation.optical_dropout_enabled,
            "optical_dropout_probability": generation.optical_dropout_probability,
        },
        "windows": {
            "original_rows": int(manifest["original_id"].nunique()),
            "generated_count": dataset.n_windows,
            "per_original_min": int(per_original.min()),
            "per_original_max": int(per_original.max()),
            "length_distribution": length_counts,
            "start_distribution": _records(start_distribution),
            "internal_optical_gap_distribution": {
                str(int(gaps)): int(count)
                for gaps, count in manifest["internal_optical_gap_count"]
                .value_counts()
                .sort_index()
                .items()
            },
            "padded_positions": int((~dataset.position_mask).sum()),
            "radar_missing_inside_windows": int(
                (dataset.position_mask & ~dataset.radar_mask).sum()
            ),
            "optical_band_missing_inside_windows": int(
                (dataset.position_mask[:, :, None] & ~dataset.optical_mask).sum()
            ),
        },
        "masks": {
            "available_pattern_count": len(available_mask_ids),
            "represented_test_rows": (
                mask_library.observation_count if mask_library is not None else 0
            ),
            "used_pattern_count": len(used_mask_ids),
            "all_used_patterns_are_from_library": used_mask_ids <= available_mask_ids,
        },
        "leakage": {
            "original_ids_crossing_folds": 0,
            "unique_window_ids": bool(manifest["window_id"].is_unique),
            "split_before_augmentation_verified": True,
        },
        "reproducibility": {
            "same_seed_identical": same_seed_reproducible,
            "alternate_seed_changes_views": alternate_seed_changes_views,
            "alternate_seed_preserves_folds": alternate_seed_preserves_folds,
            "dataset_fingerprint": window_dataset_fingerprint(dataset),
            "view_fingerprint": window_view_fingerprint(dataset),
        },
    }
    lines = [
        "# Leakage-safe temporal-window audit",
        "",
        f"- Mode: {generation.mode}; generated views: {dataset.n_windows} from "
        f"{summary['windows']['original_rows']} original rows.",
        f"- Counts by window length: {length_counts}.",
        f"- Availability-mask patterns: {len(available_mask_ids)} available, "
        f"{len(used_mask_ids)} used.",
        "- Original IDs crossing folds: 0.",
        f"- Same-seed generation identical: {same_seed_reproducible}.",
        f"- Alternate seed changes sampled views: {alternate_seed_changes_views}.",
        "",
        "Only fold assignments, metadata, and boolean availability patterns are persisted; raw "
        "window feature values remain in memory.",
        "",
    ]
    return WindowAudit(
        summary=summary,
        tables={
            "window_distribution": start_distribution,
            "mask_usage": mask_usage,
            "fold_distribution": fold_distribution,
        },
        report_markdown="\n".join(lines),
    )


def write_window_audit_artifacts(
    audit: WindowAudit,
    dataset: TemporalWindowDataset,
    fold_manifest: pd.DataFrame,
    mask_library: MaskLibrary | None,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write deterministic metadata artifacts while excluding raw competition values."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_path = output_dir / "window_summary.json"
    summary_path.write_text(
        json.dumps(audit.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(summary_path)

    frames = {
        "fold_manifest": fold_manifest,
        "window_manifest": dataset.manifest,
        "mask_templates": (
            mask_library.to_frame()
            if mask_library is not None
            else pd.DataFrame(
                columns=[
                    "mask_id",
                    "frequency",
                    "window_start",
                    "window_end",
                    "window_length",
                    "radar_availability",
                    "optical_month_availability",
                    "optical_band_availability",
                    "internal_optical_gap_count",
                ]
            )
        ),
        **audit.tables,
    }
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        written.append(path)

    report_path = output_dir / "report.md"
    report_path.write_text(audit.report_markdown, encoding="utf-8")
    written.append(report_path)
    return tuple(written)
