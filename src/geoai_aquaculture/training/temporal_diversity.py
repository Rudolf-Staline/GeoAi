"""Aligned Phase 6 temporal/tree OOF diversity and fixed-blend diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.data import ProjectConfig
from geoai_aquaculture.validation import (
    FoldManifest,
    OOFPredictions,
    ValidationReport,
    build_oof_predictions,
    build_validation_report,
    load_oof_predictions,
)


class TemporalDiversityError(ValueError):
    """Raised when temporal and tree OOF artifacts are not exactly comparable."""


@dataclass(frozen=True, slots=True)
class TemporalTreeDiversityReport:
    """Pairwise error evidence and predeclared fixed blends."""

    pairwise: dict[str, Any]
    blends: pd.DataFrame


def _load_artifact_oof(
    artifact_dir: str | Path,
    folds: FoldManifest,
    project: ProjectConfig,
) -> OOFPredictions:
    source = Path(artifact_dir)
    manifest_path = source / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "stage": "full",
        "fold_manifest_fingerprint": project.tabular.fold_manifest_fingerprint,
        "validation_window_fingerprint": project.tabular.validation_window_fingerprint,
    }
    mismatch = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise TemporalDiversityError(f"incompatible OOF artifact: {mismatch}")
    return load_oof_predictions(
        source / "oof_predictions.csv",
        source / "window_predictions.csv",
        folds,
        validation_window_fingerprint=project.tabular.validation_window_fingerprint,
        expected_fingerprint=str(manifest["oof_fingerprint"]),
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )


def pairwise_oof_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    """Compute exact aligned original-level diversity diagnostics."""

    keys = ["original_id", "repeat", "fold", "label"]
    aligned = left.loc[:, [*keys, "probability", "prediction"]].merge(
        right.loc[:, [*keys, "probability", "prediction"]],
        on=keys,
        how="inner",
        suffixes=("_temporal", "_tree"),
        validate="one_to_one",
    )
    if aligned.shape[0] != left.shape[0] or aligned.shape[0] != right.shape[0]:
        raise TemporalDiversityError("temporal and tree original-level OOF rows do not align")
    y = aligned["label"].to_numpy(dtype=np.int8)
    temporal_probability = aligned["probability_temporal"].to_numpy(dtype=np.float64)
    tree_probability = aligned["probability_tree"].to_numpy(dtype=np.float64)
    temporal_prediction = aligned["prediction_temporal"].to_numpy(dtype=np.int8)
    tree_prediction = aligned["prediction_tree"].to_numpy(dtype=np.int8)
    temporal_error = temporal_prediction != y
    tree_error = tree_prediction != y
    shared = temporal_error & tree_error
    union = temporal_error | tree_error
    return {
        "row_count": int(aligned.shape[0]),
        "pearson_probability_correlation": float(
            np.corrcoef(temporal_probability, tree_probability)[0, 1]
        ),
        "spearman_probability_correlation": float(
            pd.Series(temporal_probability).corr(pd.Series(tree_probability), method="spearman")
        ),
        "residual_correlation": float(
            np.corrcoef(y - temporal_probability, y - tree_probability)[0, 1]
        ),
        "binary_disagreement_rate": float(np.mean(temporal_prediction != tree_prediction)),
        "positive_class_disagreement_rate": float(
            np.mean(temporal_prediction[y == 1] != tree_prediction[y == 1])
        ),
        "temporal_only_correct": int((~temporal_error & tree_error).sum()),
        "tree_only_correct": int((temporal_error & ~tree_error).sum()),
        "shared_error_count": int(shared.sum()),
        "error_jaccard": float(shared.sum() / union.sum()) if union.any() else 1.0,
    }


def blend_oof_predictions(
    temporal: OOFPredictions,
    tree: OOFPredictions,
    folds: FoldManifest,
    project: ProjectConfig,
    *,
    tree_weight: float,
) -> OOFPredictions:
    """Blend exactly aligned window probabilities using a predeclared weight."""

    if not 0.0 <= tree_weight <= 1.0:
        raise TemporalDiversityError("tree weight must be within [0, 1]")
    keys = [
        "ID",
        "original_id",
        "repeat",
        "fold",
        "label",
        "window_id",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
        "fold_manifest_fingerprint",
        "validation_window_fingerprint",
    ]
    left = temporal.windows.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "probability_temporal"}
    )
    right = tree.windows.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "probability_tree"}
    )
    aligned = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if aligned.shape[0] != temporal.windows.shape[0] or aligned.shape[0] != tree.windows.shape[0]:
        raise TemporalDiversityError("temporal and tree window predictions do not align")
    probability = (1.0 - tree_weight) * aligned["probability_temporal"].to_numpy(
        dtype=np.float64
    ) + tree_weight * aligned["probability_tree"].to_numpy(dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int8)
    windows = aligned.loc[:, keys].copy()
    windows["y_true"] = windows["label"].to_numpy(dtype=np.int8)
    windows["probability"] = probability
    windows["prediction"] = prediction
    windows["predicted_class"] = prediction
    windows["experiment_id"] = f"BLEND-TREE-{tree_weight:.2f}"
    windows["model_id"] = "fixed_temporal_tree_blend"
    # Restore the authoritative persisted schema order through the temporal table.
    windows = windows.loc[:, temporal.windows.columns]
    return build_oof_predictions(
        windows,
        folds,
        validation_window_fingerprint=project.tabular.validation_window_fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )


def analyze_temporal_tree_diversity(
    temporal_artifact_dir: str | Path,
    tree_artifact_dir: str | Path,
    folds: FoldManifest,
    project: ProjectConfig,
    *,
    tree_weights: tuple[float, ...] = (0.5, 0.7),
) -> TemporalTreeDiversityReport:
    """Load full OOF artifacts and evaluate only fixed, non-optimized blends."""

    temporal = _load_artifact_oof(temporal_artifact_dir, folds, project)
    tree = _load_artifact_oof(tree_artifact_dir, folds, project)
    pairwise = pairwise_oof_summary(temporal.original, tree.original)
    records: list[dict[str, Any]] = []
    for tree_weight in tree_weights:
        blended = blend_oof_predictions(
            temporal,
            tree,
            folds,
            project,
            tree_weight=tree_weight,
        )
        report: ValidationReport = build_validation_report(blended, project.validation)
        official = report.summary["official_metric"]
        robust = report.summary["robust_selection"]
        records.append(
            {
                "tree_weight": tree_weight,
                "temporal_weight": 1.0 - tree_weight,
                "f1": official["mean_f1"],
                "roc_auc": official["mean_roc_auc"],
                "combined_score": official["mean_combined_score"],
                "robust_score": robust["score"],
                "worst_fold_score": official["worst_fold_score"],
                "worst_window_length_score": min(report.summary["window_length_scores"].values()),
                "worst_season_score": min(report.summary["season_scores"].values()),
                "optimization_performed": False,
            }
        )
    return TemporalTreeDiversityReport(pairwise=pairwise, blends=pd.DataFrame.from_records(records))


def write_temporal_tree_diversity(
    output_dir: str | Path,
    report: TemporalTreeDiversityReport,
) -> tuple[Path, Path]:
    """Persist pairwise and fixed-blend evidence under ignored artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pairwise_path = target / "temporal_tree_pairwise.json"
    pairwise_path.write_text(
        json.dumps(report.pairwise, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    blends_path = target / "temporal_tree_fixed_blends.csv"
    report.blends.to_csv(blends_path, index=False)
    return pairwise_path, blends_path
