"""Validated hand-authored configuration for Phase 5 tree experiments."""

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
from geoai_aquaculture.features import FeatureSetName

ExperimentStage = Literal["smoke", "screen", "full"]
ModelFamily = Literal["catboost", "lightgbm"]
WeightingPolicy = Literal[
    "equal_original",
    "uniform",
    "class_weighted",
    "equal_original_class_weighted",
]

_STAGES = {"smoke", "screen", "full"}
_FEATURE_SETS = {"relative", "invariant", "full", "radar", "optical", "compact"}
_MODEL_FAMILIES = {"catboost", "lightgbm"}
_WEIGHTING_POLICIES = {
    "equal_original",
    "uniform",
    "class_weighted",
    "equal_original_class_weighted",
}
_FORBIDDEN_MODEL_PARAMETERS = {
    "class_weight",
    "class_weights",
    "scale_pos_weight",
    "random_seed",
    "random_state",
    "threshold",
}


class ExperimentConfigError(ValueError):
    """Raised when a Phase 5 experiment is not explicit and scientifically valid."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One manually declared CatBoost or LightGBM profile and its hypothesis."""

    family: ModelFamily
    name: str
    hypothesis: str
    parameters: MappingProxyType[str, Any]

    def __post_init__(self) -> None:
        if self.family not in _MODEL_FAMILIES:
            raise ExperimentConfigError(f"unsupported model family: {self.family}")
        if not self.name.strip() or not self.hypothesis.strip():
            raise ExperimentConfigError("model profile name and hypothesis must be non-empty")
        if not self.parameters:
            raise ExperimentConfigError("model profile parameters must be explicitly declared")
        forbidden = sorted(_FORBIDDEN_MODEL_PARAMETERS.intersection(self.parameters))
        if forbidden:
            raise ExperimentConfigError(
                "model profile cannot bypass fold-local weighting, deterministic seeds, or "
                f"threshold policy: {forbidden}"
            )
        iterations_key = "iterations" if self.family == "catboost" else "n_estimators"
        iterations = self.parameters.get(iterations_key)
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ExperimentConfigError(
                f"{self.family} profile requires positive integer '{iterations_key}'"
            )


@dataclass(frozen=True, slots=True)
class TabularExperimentConfig:
    """Complete reproducible scientific declaration for one staged experiment."""

    source_path: Path
    experiment_id: str
    hypothesis: str
    feature_set: FeatureSetName
    weighting: WeightingPolicy
    model: ModelProfile
    seed: int
    early_stopping_rounds: int
    smoke_iteration_limit: int
    permutation_feature_count: int
    allowed_stages: tuple[ExperimentStage, ...]
    threshold: float
    notes: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.experiment_id.strip() or not self.hypothesis.strip():
            raise ExperimentConfigError("experiment ID and hypothesis must be non-empty")
        if any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in self.experiment_id
        ):
            raise ExperimentConfigError(
                "experiment ID may contain uppercase letters, digits, '-' and '_' only"
            )
        if self.feature_set not in _FEATURE_SETS:
            raise ExperimentConfigError(f"unsupported feature set: {self.feature_set}")
        if self.weighting not in _WEIGHTING_POLICIES:
            raise ExperimentConfigError(f"unsupported weighting policy: {self.weighting}")
        if self.seed < 0:
            raise ExperimentConfigError("experiment seed must be non-negative")
        if self.early_stopping_rounds < 1 or self.smoke_iteration_limit < 2:
            raise ExperimentConfigError("early stopping and smoke iteration limits are invalid")
        if self.permutation_feature_count < 0:
            raise ExperimentConfigError("permutation feature count must be non-negative")
        if not self.allowed_stages or set(self.allowed_stages) - _STAGES:
            raise ExperimentConfigError(
                "allowed stages must be a non-empty subset of smoke/screen/full"
            )
        if len(self.allowed_stages) != len(set(self.allowed_stages)):
            raise ExperimentConfigError("allowed experiment stages must be unique")
        if self.threshold != FIXED_THRESHOLD:
            raise ExperimentConfigError(
                f"classification threshold must remain exactly {FIXED_THRESHOLD}"
            )

    def resolved_dict(self) -> dict[str, Any]:
        """Return the canonical versioned configuration persisted with every run."""

        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "feature_set": self.feature_set,
            "weighting": self.weighting,
            "model": {
                "family": self.model.family,
                "name": self.model.name,
                "hypothesis": self.model.hypothesis,
                "parameters": dict(self.model.parameters),
            },
            "seed": self.seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            "smoke_iteration_limit": self.smoke_iteration_limit,
            "permutation_feature_count": self.permutation_feature_count,
            "allowed_stages": list(self.allowed_stages),
            "threshold": self.threshold,
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    def require_stage(self, stage: ExperimentStage) -> None:
        """Reject accidental promotion of a screening-only profile."""

        if stage not in self.allowed_stages:
            raise ExperimentConfigError(
                f"experiment {self.experiment_id} is not approved for stage '{stage}'"
            )


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"'{key}' must be a mapping")
    return value


def _string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"'{key}' must be a non-empty string")
    return value


def _integer(value: object, key: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ExperimentConfigError(f"'{key}' must be an integer >= {minimum}")
    return value


def _parameters(value: object) -> MappingProxyType[str, Any]:
    parameters = _mapping(value, "experiment.model.parameters")
    for key, item in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise ExperimentConfigError("model parameter names must be non-empty strings")
        if isinstance(item, float) and not np.isfinite(item):
            raise ExperimentConfigError(f"model parameter '{key}' must be finite")
        if not isinstance(item, str | int | float | bool | type(None)):
            raise ExperimentConfigError(f"model parameter '{key}' must be a scalar YAML value")
    return MappingProxyType(dict(parameters))


def load_tabular_experiment_config(path: str | Path) -> TabularExperimentConfig:
    """Load one hand-authored experiment file without implicit search behavior."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"tabular experiment configuration not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid experiment YAML at {source}: {exc}") from exc
    root = _mapping(raw, "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    model = _mapping(experiment.get("model"), "experiment.model")
    family = _string(model.get("family"), "experiment.model.family")
    profile = ModelProfile(
        family=family,
        name=_string(model.get("name"), "experiment.model.name"),
        hypothesis=_string(model.get("hypothesis"), "experiment.model.hypothesis"),
        parameters=_parameters(model.get("parameters")),
    )
    stage_values = experiment.get("allowed_stages", ["smoke", "screen", "full"])
    if not isinstance(stage_values, list) or not all(
        isinstance(item, str) for item in stage_values
    ):
        raise ExperimentConfigError("experiment.allowed_stages must be a list of strings")
    payload = {
        "schema_version": 1,
        "experiment_id": _string(experiment.get("id"), "experiment.id"),
        "hypothesis": _string(experiment.get("hypothesis"), "experiment.hypothesis"),
        "feature_set": _string(experiment.get("feature_set"), "experiment.feature_set"),
        "weighting": _string(experiment.get("weighting", "equal_original"), "experiment.weighting"),
        "model": {
            "family": profile.family,
            "name": profile.name,
            "hypothesis": profile.hypothesis,
            "parameters": dict(profile.parameters),
        },
        "seed": _integer(experiment.get("seed", 2026), "experiment.seed"),
        "early_stopping_rounds": _integer(
            experiment.get("early_stopping_rounds", 50),
            "experiment.early_stopping_rounds",
            minimum=1,
        ),
        "smoke_iteration_limit": _integer(
            experiment.get("smoke_iteration_limit", 20),
            "experiment.smoke_iteration_limit",
            minimum=2,
        ),
        "permutation_feature_count": _integer(
            experiment.get("permutation_feature_count", 10),
            "experiment.permutation_feature_count",
        ),
        "allowed_stages": stage_values,
        "threshold": float(experiment.get("threshold", FIXED_THRESHOLD)),
        "notes": str(experiment.get("notes", "")),
    }
    fingerprint = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return TabularExperimentConfig(
        source_path=source,
        experiment_id=payload["experiment_id"],
        hypothesis=payload["hypothesis"],
        feature_set=payload["feature_set"],
        weighting=payload["weighting"],
        model=profile,
        seed=payload["seed"],
        early_stopping_rounds=payload["early_stopping_rounds"],
        smoke_iteration_limit=payload["smoke_iteration_limit"],
        permutation_feature_count=payload["permutation_feature_count"],
        allowed_stages=tuple(payload["allowed_stages"]),
        threshold=payload["threshold"],
        notes=payload["notes"],
        fingerprint=fingerprint,
    )
