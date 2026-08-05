"""Configuration contract for Phase 7 domain-shift diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from geoai_aquaculture.features import FeatureSetName

AdaptationMethod = Literal["feature_removal", "importance_weighting"]


class DomainShiftConfigError(ValueError):
    """Raised when a Phase 7 configuration violates the declared protocol."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    result = int(value)
    if result < minimum:
        raise DomainShiftConfigError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise DomainShiftConfigError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class DomainModelConfig:
    """One small, manually declared LightGBM domain-classifier profile."""

    parameters: dict[str, Any]
    early_stopping_rounds: int = 50

    def __post_init__(self) -> None:
        required = {"objective", "n_estimators", "learning_rate", "num_leaves"}
        missing = sorted(required - set(self.parameters))
        if missing:
            raise DomainShiftConfigError(f"domain model parameters are missing: {missing}")
        if self.parameters["objective"] != "binary":
            raise DomainShiftConfigError("domain model objective must be binary")
        if int(self.parameters["n_estimators"]) < 20:
            raise DomainShiftConfigError("domain model needs at least 20 estimators")
        if self.early_stopping_rounds < 1:
            raise DomainShiftConfigError("domain early stopping must be positive")


@dataclass(frozen=True, slots=True)
class ImportanceWeightConfig:
    """Conservative density-ratio clipping policy."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum <= 1.0:
            raise DomainShiftConfigError("importance-weight minimum must lie in (0, 1]")
        if self.maximum < 1.0 or self.maximum <= self.minimum:
            raise DomainShiftConfigError("importance-weight maximum must exceed one and minimum")


@dataclass(frozen=True, slots=True)
class Phase7Config:
    """Complete bounded Phase 7 experiment contract."""

    source_path: Path
    seed: int
    n_splits: int
    representations: tuple[FeatureSetName, ...]
    selection_representation: FeatureSetName
    top_feature_count: int
    removal_feature_count: int
    similarity_holdout_fraction: float
    similarity_holdout_minimum: int
    adaptation_seeds: tuple[int, ...]
    adaptation_methods: tuple[AdaptationMethod, ...]
    label_baseline_config: Path
    tree_oof_artifact: Path
    temporal_oof_artifact: Path
    output_dir: Path
    domain_model: DomainModelConfig
    importance_weights: ImportanceWeightConfig
    fingerprint: str

    def __post_init__(self) -> None:
        allowed = {"relative", "invariant", "full", "radar", "optical", "compact"}
        if not self.representations or set(self.representations) - allowed:
            raise DomainShiftConfigError("domain representations contain unsupported values")
        if self.selection_representation not in self.representations:
            raise DomainShiftConfigError(
                "selection representation must be included in diagnostic representations"
            )
        if self.n_splits < 3:
            raise DomainShiftConfigError("domain validation requires at least three folds")
        if not 0.0 < self.similarity_holdout_fraction < 1.0:
            raise DomainShiftConfigError("similarity holdout fraction must lie in (0, 1)")
        if len(set(self.adaptation_seeds)) != len(self.adaptation_seeds):
            raise DomainShiftConfigError("adaptation seeds must be unique")
        if len(self.adaptation_seeds) < 2:
            raise DomainShiftConfigError("adaptation decisions require at least two seeds")
        if not self.adaptation_methods:
            raise DomainShiftConfigError("at least one controlled adaptation method is required")
        if self.removal_feature_count > self.top_feature_count:
            raise DomainShiftConfigError("feature removal cannot exceed reported top features")


def load_phase7_config(path: str | Path) -> Phase7Config:
    """Load one stable Phase 7 YAML configuration."""

    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("phase7"), dict):
        raise DomainShiftConfigError("Phase 7 configuration requires a 'phase7' mapping")
    config = raw["phase7"]
    representations = tuple(str(value) for value in config.get("representations", []))
    methods = tuple(str(value) for value in config.get("adaptation_methods", []))
    allowed_methods = {"feature_removal", "importance_weighting"}
    if set(methods) - allowed_methods:
        raise DomainShiftConfigError("unsupported Phase 7 adaptation method")
    model_raw = config.get("domain_model")
    if not isinstance(model_raw, dict) or not isinstance(model_raw.get("parameters"), dict):
        raise DomainShiftConfigError("domain_model.parameters must be a mapping")
    clip_raw = config.get("importance_weight_clip")
    if not isinstance(clip_raw, list | tuple) or len(clip_raw) != 2:
        raise DomainShiftConfigError("importance_weight_clip must contain [minimum, maximum]")
    root = source.parents[2] if source.parent.name == "experiments" else source.parent
    payload = {
        "seed": _positive_int(config.get("seed", 7201), "phase7.seed", minimum=0),
        "n_splits": _positive_int(config.get("n_splits", 5), "phase7.n_splits", minimum=3),
        "representations": list(representations),
        "selection_representation": str(config.get("selection_representation", "full")),
        "top_feature_count": _positive_int(
            config.get("top_feature_count", 30), "phase7.top_feature_count"
        ),
        "removal_feature_count": _positive_int(
            config.get("removal_feature_count", 10), "phase7.removal_feature_count"
        ),
        "similarity_holdout_fraction": _finite_float(
            config.get("similarity_holdout_fraction", 0.20),
            "phase7.similarity_holdout_fraction",
        ),
        "similarity_holdout_minimum": _positive_int(
            config.get("similarity_holdout_minimum", 100),
            "phase7.similarity_holdout_minimum",
        ),
        "adaptation_seeds": [int(value) for value in config.get("adaptation_seeds", [])],
        "adaptation_methods": list(methods),
        "label_baseline_config": str(config["label_baseline_config"]),
        "tree_oof_artifact": str(config["tree_oof_artifact"]),
        "temporal_oof_artifact": str(config["temporal_oof_artifact"]),
        "output_dir": str(config.get("output_dir", "artifacts/domain_shift")),
        "domain_model": {
            "parameters": dict(model_raw["parameters"]),
            "early_stopping_rounds": _positive_int(
                model_raw.get("early_stopping_rounds", 50),
                "phase7.domain_model.early_stopping_rounds",
            ),
        },
        "importance_weight_clip": [
            _finite_float(clip_raw[0], "phase7.importance_weight_clip.minimum"),
            _finite_float(clip_raw[1], "phase7.importance_weight_clip.maximum"),
        ],
    }
    fingerprint = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return Phase7Config(
        source_path=source,
        seed=int(payload["seed"]),
        n_splits=int(payload["n_splits"]),
        representations=tuple(payload["representations"]),  # type: ignore[arg-type]
        selection_representation=payload["selection_representation"],  # type: ignore[arg-type]
        top_feature_count=int(payload["top_feature_count"]),
        removal_feature_count=int(payload["removal_feature_count"]),
        similarity_holdout_fraction=float(payload["similarity_holdout_fraction"]),
        similarity_holdout_minimum=int(payload["similarity_holdout_minimum"]),
        adaptation_seeds=tuple(int(value) for value in payload["adaptation_seeds"]),
        adaptation_methods=tuple(payload["adaptation_methods"]),  # type: ignore[arg-type]
        label_baseline_config=(root / payload["label_baseline_config"]).resolve(),
        tree_oof_artifact=(root / payload["tree_oof_artifact"]).resolve(),
        temporal_oof_artifact=(root / payload["temporal_oof_artifact"]).resolve(),
        output_dir=(root / payload["output_dir"]).resolve(),
        domain_model=DomainModelConfig(
            parameters=dict(payload["domain_model"]["parameters"]),
            early_stopping_rounds=int(payload["domain_model"]["early_stopping_rounds"]),
        ),
        importance_weights=ImportanceWeightConfig(
            minimum=float(payload["importance_weight_clip"][0]),
            maximum=float(payload["importance_weight_clip"][1]),
        ),
        fingerprint=fingerprint,
    )
