"""Typed Phase 5 fold, experiment, and artifact results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from geoai_aquaculture.features import FeatureRegistry
from geoai_aquaculture.validation import OOFPredictions, ValidationReport

from .config import ExperimentStage


@dataclass(frozen=True, slots=True)
class FoldTrainingResult:
    """One fitted repeat/fold with timing, weighting, and early-stopping provenance."""

    repeat: int
    fold: int
    seed: int
    train_original_count: int
    validation_original_count: int
    train_window_count: int
    validation_window_count: int
    train_positive_rate: float
    validation_positive_rate: float
    feature_count: int
    weighting_policy: str
    class_weight_zero: float
    class_weight_one: float
    original_weight_min: float
    original_weight_max: float
    best_iteration: int
    fitted_iterations: int
    validation_metric_name: str
    validation_metric_value: float
    training_seconds: float
    inference_seconds: float
    model_path: str
    model_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe fold metadata."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentArtifactManifest:
    """Compatibility contract used to reject unsafe resume or overwrite behavior."""

    schema_version: int
    status: str
    stage: ExperimentStage
    experiment_id: str
    experiment_config_fingerprint: str
    base_config_sha256: str
    git_commit: str | None
    tracked_files_dirty: bool | None
    fold_manifest_fingerprint: str
    validation_window_fingerprint: str
    full_feature_schema_fingerprint: str
    selected_feature_schema_fingerprint: str
    oof_fingerprint: str
    original_oof_rows: int
    window_prediction_rows: int

    def as_dict(self) -> dict[str, Any]:
        """Return the stable manifest schema persisted beside every experiment."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TabularTrainingResult:
    """Complete in-memory result for one smoke, screen, or authoritative run."""

    stage: ExperimentStage
    oof: OOFPredictions
    report: ValidationReport
    folds: tuple[FoldTrainingResult, ...]
    feature_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    feature_names: tuple[str, ...]
    selected_registry: FeatureRegistry
    full_feature_schema_fingerprint: str
    selected_feature_schema_fingerprint: str
    runtime_seconds: float
    peak_rss_megabytes: float
    artifact_dir: Path

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError("tabular training result must contain at least one fitted fold")
        if len(self.feature_names) < 1:
            raise ValueError("tabular training result must retain its exact feature list")
        if self.selected_registry.feature_names != self.feature_names:
            raise ValueError("training result registry must match its exact feature list")
        if self.runtime_seconds <= 0.0 or self.peak_rss_megabytes <= 0.0:
            raise ValueError("training runtime and peak memory must be positive")
