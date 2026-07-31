"""OOF candidate compatibility, blending, and nested weight selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.constants import FIXED_THRESHOLD
from geoai_aquaculture.data import ProjectConfig
from geoai_aquaculture.metrics import metric_result
from geoai_aquaculture.validation import (
    OOFPredictions,
    ValidationReport,
    build_oof_predictions,
    build_validation_report,
    load_fold_manifest,
    load_oof_predictions,
)

from .config import FinalCandidateConfig, FinalDeliveryConfig


class FinalOOFError(ValueError):
    """Raised when final candidate artifacts are incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class FinalCandidate:
    """One accepted candidate with validated full OOF predictions."""

    declaration: FinalCandidateConfig
    artifact_manifest: dict[str, Any]
    metrics: dict[str, Any]
    oof: OOFPredictions

    @property
    def experiment_id(self) -> str:
        return self.declaration.experiment_id


@dataclass(frozen=True, slots=True)
class BlendEvaluation:
    """One fixed blend evaluated under the immutable Phase 4 report."""

    tree_weight: float
    temporal_weight: float
    oof: OOFPredictions
    report: ValidationReport
    label: str


@dataclass(frozen=True, slots=True)
class WeightSelectionResult:
    """Nested fold weight estimates and one production fixed-weight choice."""

    fold_weights: pd.DataFrame
    crossfit: BlendEvaluation
    production: BlendEvaluation
    alternatives: tuple[BlendEvaluation, ...]


def _load_manifest(path: Path) -> dict[str, Any]:
    source = path / "experiment_manifest.json"
    if not source.is_file():
        raise FileNotFoundError(f"candidate manifest not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalOOFError(f"invalid candidate manifest: {source}")
    return payload


def load_final_candidates(
    config: FinalDeliveryConfig,
    project: ProjectConfig,
) -> tuple[FinalCandidate, ...]:
    """Load accepted tree and temporal experts through the authoritative OOF loader."""

    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    candidates: list[FinalCandidate] = []
    for declaration in config.candidates:
        artifact = declaration.artifact_dir
        manifest = _load_manifest(artifact)
        expected = {
            "status": "complete",
            "stage": "full",
            "experiment_id": declaration.experiment_id,
            "fold_manifest_fingerprint": project.tabular.fold_manifest_fingerprint,
            "validation_window_fingerprint": project.tabular.validation_window_fingerprint,
            "original_oof_rows": project.tabular.expected_full_oof_rows,
        }
        mismatch = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatch:
            raise FinalOOFError(
                f"candidate {declaration.experiment_id} is incompatible: {mismatch}"
            )
        metrics_path = artifact / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        oof = load_oof_predictions(
            artifact / "oof_predictions.csv",
            artifact / "window_predictions.csv",
            folds,
            validation_window_fingerprint=project.tabular.validation_window_fingerprint,
            method=project.validation.aggregation_method,
            trimmed_fraction=project.validation.trimmed_mean_fraction,
            expected_fingerprint=str(manifest["oof_fingerprint"]),
        )
        candidates.append(
            FinalCandidate(
                declaration=declaration,
                artifact_manifest=manifest,
                metrics=metrics,
                oof=oof,
            )
        )
    if {candidate.declaration.kind for candidate in candidates} != {"tree", "temporal"}:
        raise FinalOOFError("accepted final candidates must be one tree and one temporal model")
    return tuple(candidates)


def _aligned_windows(tree: FinalCandidate, temporal: FinalCandidate) -> pd.DataFrame:
    keys = [
        "ID",
        "original_id",
        "repeat",
        "fold",
        "y_true",
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
    left = tree.oof.windows.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "tree_probability"}
    )
    right = temporal.oof.windows.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "temporal_probability"}
    )
    aligned = left.merge(right, on=keys, how="inner", validate="one_to_one")
    expected = tree.oof.windows.shape[0]
    if aligned.shape[0] != expected or temporal.oof.windows.shape[0] != expected:
        raise FinalOOFError("tree and temporal window OOF rows do not align exactly")
    return aligned.sort_values(
        ["repeat", "fold", "original_id", "window_id"], kind="stable", ignore_index=True
    )


def _blend_window_frame(
    aligned: pd.DataFrame,
    *,
    tree_weights: np.ndarray | float,
    experiment_id: str,
    model_id: str,
) -> pd.DataFrame:
    if np.isscalar(tree_weights):
        weights = np.full(aligned.shape[0], float(tree_weights), dtype=np.float64)
    else:
        weights = np.asarray(tree_weights, dtype=np.float64)
    if weights.shape != (aligned.shape[0],) or np.any((weights < 0.0) | (weights > 1.0)):
        raise FinalOOFError("blend weights must align with windows and lie in [0, 1]")
    probability = (
        weights * aligned["tree_probability"].to_numpy(dtype=np.float64)
        + (1.0 - weights) * aligned["temporal_probability"].to_numpy(dtype=np.float64)
    )
    prediction = (probability >= FIXED_THRESHOLD).astype(np.int8)
    frame = aligned.drop(columns=["tree_probability", "temporal_probability"]).copy()
    frame["probability"] = probability
    frame["prediction"] = prediction
    frame["predicted_class"] = prediction
    frame["experiment_id"] = experiment_id
    frame["model_id"] = model_id
    columns = [
        "ID",
        "original_id",
        "repeat",
        "fold",
        "y_true",
        "label",
        "probability",
        "prediction",
        "predicted_class",
        "window_id",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
        "experiment_id",
        "model_id",
        "fold_manifest_fingerprint",
        "validation_window_fingerprint",
    ]
    return frame.loc[:, columns]


def evaluate_fixed_blend(
    tree: FinalCandidate,
    temporal: FinalCandidate,
    project: ProjectConfig,
    *,
    tree_weight: float,
    label: str,
) -> BlendEvaluation:
    """Evaluate one predeclared fixed blend on complete aligned window OOF."""

    if not 0.0 <= tree_weight <= 1.0:
        raise FinalOOFError("tree blend weight must lie in [0, 1]")
    aligned = _aligned_windows(tree, temporal)
    windows = _blend_window_frame(
        aligned,
        tree_weights=tree_weight,
        experiment_id=f"PHASE8-{label}",
        model_id=f"blend:tree={tree_weight:.4f}:temporal={1.0-tree_weight:.4f}",
    )
    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    oof = build_oof_predictions(
        windows,
        folds,
        validation_window_fingerprint=tree.oof.validation_window_fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )
    return BlendEvaluation(
        tree_weight=float(tree_weight),
        temporal_weight=float(1.0 - tree_weight),
        oof=oof,
        report=build_validation_report(oof, project.validation),
        label=label,
    )


def _aligned_originals(tree: FinalCandidate, temporal: FinalCandidate) -> pd.DataFrame:
    keys = ["original_id", "repeat", "fold", "label"]
    left = tree.oof.original.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "tree_probability"}
    )
    right = temporal.oof.original.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "temporal_probability"}
    )
    aligned = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if (
        aligned.shape[0] != tree.oof.original.shape[0]
        or aligned.shape[0] != temporal.oof.original.shape[0]
    ):
        raise FinalOOFError("candidate original OOF rows do not align exactly")
    return aligned


def _best_weight(
    frame: pd.DataFrame,
    grid: np.ndarray,
) -> tuple[float, float, float]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    tree = frame["tree_probability"].to_numpy(dtype=np.float64)
    temporal = frame["temporal_probability"].to_numpy(dtype=np.float64)
    records: list[tuple[float, float, float, float]] = []
    for weight in grid:
        probability = weight * tree + (1.0 - weight) * temporal
        metrics = metric_result(labels, probability)
        assert metrics.combined_score is not None
        records.append(
            (
                float(weight),
                float(metrics.combined_score),
                metrics.brier_score,
                metrics.log_loss,
            )
        )
    # Competition score first, then Brier, then simplicity near an equal blend.
    records.sort(key=lambda row: (-row[1], row[2], abs(row[0] - 0.5), row[3]))
    winner = records[0]
    return winner[0], winner[1], winner[2]


def learn_nested_weight(
    tree: FinalCandidate,
    temporal: FinalCandidate,
    project: ProjectConfig,
    *,
    grid_step: float,
    fixed_tree_weights: tuple[float, ...],
) -> WeightSelectionResult:
    """Select fold-held-out weights and derive one simple production fixed weight."""

    grid = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step, dtype=np.float64)
    grid = np.clip(np.round(grid, 10), 0.0, 1.0)
    originals = _aligned_originals(tree, temporal)
    aligned_windows = _aligned_windows(tree, temporal)
    fold_records: list[dict[str, float | int]] = []
    row_weights = np.empty(aligned_windows.shape[0], dtype=np.float64)
    for repeat in sorted(originals["repeat"].unique().tolist()):
        for fold in sorted(originals.loc[originals["repeat"].eq(repeat), "fold"].unique().tolist()):
            training = originals.loc[
                originals["repeat"].eq(repeat) & ~originals["fold"].eq(fold)
            ]
            heldout = originals.loc[
                originals["repeat"].eq(repeat) & originals["fold"].eq(fold)
            ]
            weight, training_score, training_brier = _best_weight(training, grid)
            heldout_probability = (
                weight * heldout["tree_probability"].to_numpy(dtype=np.float64)
                + (1.0 - weight) * heldout["temporal_probability"].to_numpy(dtype=np.float64)
            )
            heldout_metrics = metric_result(heldout["label"], heldout_probability)
            mask = aligned_windows["repeat"].eq(repeat) & aligned_windows["fold"].eq(fold)
            row_weights[mask.to_numpy()] = weight
            fold_records.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "tree_weight": float(weight),
                    "temporal_weight": float(1.0 - weight),
                    "training_combined_score": training_score,
                    "training_brier_score": training_brier,
                    "heldout_combined_score": float(heldout_metrics.combined_score or np.nan),
                    "heldout_brier_score": heldout_metrics.brier_score,
                }
            )
    if not np.isfinite(row_weights).all():
        raise FinalOOFError("nested weight assignment left unfilled validation windows")
    crossfit_windows = _blend_window_frame(
        aligned_windows,
        tree_weights=row_weights,
        experiment_id="PHASE8-NESTED-WEIGHT",
        model_id="blend:nested-fold-weight",
    )
    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    crossfit_oof = build_oof_predictions(
        crossfit_windows,
        folds,
        validation_window_fingerprint=tree.oof.validation_window_fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )
    crossfit = BlendEvaluation(
        tree_weight=float(np.median(row_weights)),
        temporal_weight=float(1.0 - np.median(row_weights)),
        oof=crossfit_oof,
        report=build_validation_report(crossfit_oof, project.validation),
        label="nested_crossfit",
    )
    fold_weights = pd.DataFrame.from_records(fold_records).sort_values(
        ["repeat", "fold"], ignore_index=True
    )
    learned_weight = float(np.median(fold_weights["tree_weight"]))
    learned_weight = float(grid[np.argmin(np.abs(grid - learned_weight))])
    candidate_weights = tuple(dict.fromkeys((*fixed_tree_weights, learned_weight)))
    alternatives = tuple(
        evaluate_fixed_blend(
            tree,
            temporal,
            project,
            tree_weight=weight,
            label=f"fixed-tree-{weight:.2f}",
        )
        for weight in candidate_weights
    )
    best_robust = max(
        float(item.report.summary["robust_selection"]["score"]) for item in alternatives
    )
    # Prefer equal weights when practically equivalent; otherwise choose the most robust
    # fixed blend.
    near_best = [
        item
        for item in alternatives
        if float(item.report.summary["robust_selection"]["score"]) >= best_robust - 0.0002
    ]
    near_best.sort(
        key=lambda item: (
            abs(item.tree_weight - 0.5),
            -float(item.report.summary["official_metric"]["mean_combined_score"]),
        )
    )
    production = near_best[0]
    return WeightSelectionResult(
        fold_weights=fold_weights,
        crossfit=crossfit,
        production=production,
        alternatives=alternatives,
    )
