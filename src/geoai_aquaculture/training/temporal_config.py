"""Typed configuration for compact Phase 6 temporal viability experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import yaml

from geoai_aquaculture.constants import FIXED_THRESHOLD

from .config import ExperimentStage


class TemporalExperimentConfigError(ValueError):
    """Raised when a temporal experiment is broad, unsafe, or malformed."""


TemporalObjective = Literal["bce", "bce_consistency"]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TemporalTrainingConfig:
    """Bounded optimizer and early-stopping choices."""

    max_epochs: int = 50
    smoke_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    gradient_clip: float = 1.0
    input_clip: float = 8.0
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.max_epochs < 2 or not 1 <= self.smoke_epochs <= self.max_epochs:
            raise TemporalExperimentConfigError("temporal epoch limits are invalid")
        if self.batch_size < 2 or self.patience < 1 or self.cpu_threads < 1:
            raise TemporalExperimentConfigError("batch size, patience and CPU threads must be positive")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("gradient_clip", self.gradient_clip),
            ("input_clip", self.input_clip),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise TemporalExperimentConfigError(f"training.{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class TemporalViabilityGates:
    """Predeclared criteria preventing post-hoc neural acceptance."""

    best_tree_robust_score: float = 0.980404
    full_score_tolerance: float = 0.003
    blend_improvement: float = 0.0005
    screening_reference_robust: float = 0.980836
    screening_max_gap: float = 0.010

    def __post_init__(self) -> None:
        for name, value in (
            ("best_tree_robust_score", self.best_tree_robust_score),
            ("full_score_tolerance", self.full_score_tolerance),
            ("blend_improvement", self.blend_improvement),
            ("screening_reference_robust", self.screening_reference_robust),
            ("screening_max_gap", self.screening_max_gap),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise TemporalExperimentConfigError(f"viability.{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TemporalExperimentConfig:
    """One intentionally small GRU experiment."""

    source_path: Path
    experiment_id: str
    hypothesis: str
    objective: TemporalObjective
    architecture: MappingProxyType[str, Any]
    training: TemporalTrainingConfig
    viability: TemporalViabilityGates
    consistency_lambda: float
    seed: int
    allowed_stages: tuple[ExperimentStage, ...]
    threshold: float
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.experiment_id.strip() or not self.hypothesis.strip():
            raise TemporalExperimentConfigError("temporal experiment ID and hypothesis are required")
        if self.objective not in {"bce", "bce_consistency"}:
            raise TemporalExperimentConfigError("temporal objective is unsupported")
        if self.seed < 0:
            raise TemporalExperimentConfigError("temporal seed must be non-negative")
        if self.threshold != FIXED_THRESHOLD:
            raise TemporalExperimentConfigError(
                f"classification threshold must remain exactly {FIXED_THRESHOLD}"
            )
        if not self.allowed_stages or set(self.allowed_stages) - {"smoke", "screen", "full"}:
            raise TemporalExperimentConfigError("allowed stages are invalid")
        if not np.isfinite(self.consistency_lambda) or self.consistency_lambda < 0.0:
            raise TemporalExperimentConfigError("consistency lambda must be finite and non-negative")
        if self.objective == "bce" and self.consistency_lambda != 0.0:
            raise TemporalExperimentConfigError("BCE baseline cannot silently enable consistency")
        if self.objective == "bce_consistency" and self.consistency_lambda <= 0.0:
            raise TemporalExperimentConfigError("consistency objective requires a positive lambda")

    def require_stage(self, stage: ExperimentStage) -> None:
        if stage not in self.allowed_stages:
            raise TemporalExperimentConfigError(
                f"experiment {self.experiment_id} is not approved for stage '{stage}'"
            )

    def resolved_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "objective": self.objective,
            "architecture": dict(self.architecture),
            "training": {
                "max_epochs": self.training.max_epochs,
                "smoke_epochs": self.training.smoke_epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "patience": self.training.patience,
                "gradient_clip": self.training.gradient_clip,
                "input_clip": self.training.input_clip,
                "cpu_threads": self.training.cpu_threads,
            },
            "viability": {
                "best_tree_robust_score": self.viability.best_tree_robust_score,
                "full_score_tolerance": self.viability.full_score_tolerance,
                "blend_improvement": self.viability.blend_improvement,
                "screening_reference_robust": self.viability.screening_reference_robust,
                "screening_max_gap": self.viability.screening_max_gap,
            },
            "consistency_lambda": self.consistency_lambda,
            "seed": self.seed,
            "allowed_stages": list(self.allowed_stages),
            "threshold": self.threshold,
            "fingerprint": self.fingerprint,
        }


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TemporalExperimentConfigError(f"'{key}' must be a mapping")
    return value


def _positive_int(value: object, key: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TemporalExperimentConfigError(f"'{key}' must be an integer >= {minimum}")
    return value


def _positive_float(value: object, key: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TemporalExperimentConfigError(f"'{key}' must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise TemporalExperimentConfigError(f"'{key}' must be finite and {qualifier}")
    return result


def load_temporal_experiment_config(path: str | Path) -> TemporalExperimentConfig:
    """Load one explicit temporal experiment without any search or tuning behavior."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"temporal experiment configuration not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    architecture = _mapping(experiment.get("architecture", {}), "experiment.architecture")
    training_raw = _mapping(experiment.get("training", {}), "experiment.training")
    viability_raw = _mapping(experiment.get("viability", {}), "experiment.viability")
    allowed = experiment.get("allowed_stages", ["smoke", "screen", "full"])
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise TemporalExperimentConfigError("experiment.allowed_stages must be a list of strings")
    training = TemporalTrainingConfig(
        max_epochs=_positive_int(training_raw.get("max_epochs", 50), "training.max_epochs", minimum=2),
        smoke_epochs=_positive_int(training_raw.get("smoke_epochs", 2), "training.smoke_epochs"),
        batch_size=_positive_int(training_raw.get("batch_size", 256), "training.batch_size", minimum=2),
        learning_rate=_positive_float(training_raw.get("learning_rate", 1e-3), "training.learning_rate"),
        weight_decay=_positive_float(training_raw.get("weight_decay", 1e-4), "training.weight_decay"),
        patience=_positive_int(training_raw.get("patience", 8), "training.patience"),
        gradient_clip=_positive_float(training_raw.get("gradient_clip", 1.0), "training.gradient_clip"),
        input_clip=_positive_float(training_raw.get("input_clip", 8.0), "training.input_clip"),
        cpu_threads=_positive_int(training_raw.get("cpu_threads", 4), "training.cpu_threads"),
    )
    viability = TemporalViabilityGates(
        best_tree_robust_score=_positive_float(
            viability_raw.get("best_tree_robust_score", 0.980404),
            "viability.best_tree_robust_score",
            allow_zero=True,
        ),
        full_score_tolerance=_positive_float(
            viability_raw.get("full_score_tolerance", 0.003),
            "viability.full_score_tolerance",
            allow_zero=True,
        ),
        blend_improvement=_positive_float(
            viability_raw.get("blend_improvement", 0.0005),
            "viability.blend_improvement",
            allow_zero=True,
        ),
        screening_reference_robust=_positive_float(
            viability_raw.get("screening_reference_robust", 0.980836),
            "viability.screening_reference_robust",
            allow_zero=True,
        ),
        screening_max_gap=_positive_float(
            viability_raw.get("screening_max_gap", 0.010),
            "viability.screening_max_gap",
            allow_zero=True,
        ),
    )
    payload = {
        "schema_version": 1,
        "experiment_id": str(experiment.get("id", "")).strip(),
        "hypothesis": str(experiment.get("hypothesis", "")).strip(),
        "objective": str(experiment.get("objective", "bce")),
        "architecture": dict(architecture),
        "training": training.__dict__ if hasattr(training, "__dict__") else {
            field: getattr(training, field)
            for field in training.__dataclass_fields__
        },
        "viability": {
            field: getattr(viability, field)
            for field in viability.__dataclass_fields__
        },
        "consistency_lambda": float(experiment.get("consistency_lambda", 0.0)),
        "seed": _positive_int(experiment.get("seed", 6101), "experiment.seed", minimum=0),
        "allowed_stages": allowed,
        "threshold": float(experiment.get("threshold", FIXED_THRESHOLD)),
    }
    fingerprint = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return TemporalExperimentConfig(
        source_path=source,
        experiment_id=payload["experiment_id"],
        hypothesis=payload["hypothesis"],
        objective=payload["objective"],
        architecture=MappingProxyType(dict(architecture)),
        training=training,
        viability=viability,
        consistency_lambda=payload["consistency_lambda"],
        seed=payload["seed"],
        allowed_stages=tuple(allowed),
        threshold=payload["threshold"],
        fingerprint=fingerprint,
    )
