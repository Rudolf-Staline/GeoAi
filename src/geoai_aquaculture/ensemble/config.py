"""Explicit Phase 8 configuration for final candidate selection and delivery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from geoai_aquaculture.constants import FIXED_THRESHOLD

CandidateKind = Literal["tree", "temporal"]
CalibrationMethod = Literal["none", "sigmoid", "beta"]


class FinalConfigError(ValueError):
    """Raised when the final delivery configuration is ambiguous or unsafe."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class FinalCandidateConfig:
    """One accepted OOF-backed candidate eligible for the final ensemble."""

    experiment_id: str
    kind: CandidateKind
    artifact_dir: Path
    experiment_config: Path
    role: str

    def __post_init__(self) -> None:
        if self.kind not in {"tree", "temporal"}:
            raise FinalConfigError(f"unsupported final candidate kind: {self.kind}")
        if not self.experiment_id.strip() or not self.role.strip():
            raise FinalConfigError("candidate experiment ID and role are required")


@dataclass(frozen=True, slots=True)
class FinalDeliveryConfig:
    """Complete bounded Phase 8 delivery declaration."""

    source_path: Path
    project_config: Path
    output_dir: Path
    candidates: tuple[FinalCandidateConfig, ...]
    calibration_methods: tuple[CalibrationMethod, ...]
    weight_grid_step: float
    fixed_tree_weights: tuple[float, ...]
    rebuild_missing_oof: bool
    overwrite_oof: bool
    tta_enabled: bool
    prior_shift_correction_enabled: bool
    threshold: float
    source_commit: str | None
    fingerprint: str

    def __post_init__(self) -> None:
        if len(self.candidates) != 2:
            raise FinalConfigError(
                "Phase 8 currently requires exactly one tree and one temporal expert"
            )
        if {candidate.kind for candidate in self.candidates} != {"tree", "temporal"}:
            raise FinalConfigError("final candidates must contain one tree and one temporal expert")
        if len({candidate.experiment_id for candidate in self.candidates}) != 2:
            raise FinalConfigError("final candidate experiment IDs must be unique")
        if not self.calibration_methods or self.calibration_methods[0] != "none":
            raise FinalConfigError("calibration methods must begin with the uncalibrated baseline")
        if set(self.calibration_methods) - {"none", "sigmoid", "beta"}:
            raise FinalConfigError("unsupported final calibration method")
        if not 0.0 < self.weight_grid_step <= 0.25:
            raise FinalConfigError("weight grid step must be in (0, 0.25]")
        if not self.fixed_tree_weights:
            raise FinalConfigError("at least one fixed tree weight is required")
        if any(not 0.0 <= value <= 1.0 for value in self.fixed_tree_weights):
            raise FinalConfigError("fixed ensemble weights must lie in [0, 1]")
        if self.threshold != FIXED_THRESHOLD:
            raise FinalConfigError(
                f"classification threshold must remain exactly {FIXED_THRESHOLD}"
            )

    @property
    def tree_candidate(self) -> FinalCandidateConfig:
        return next(candidate for candidate in self.candidates if candidate.kind == "tree")

    @property
    def temporal_candidate(self) -> FinalCandidateConfig:
        return next(candidate for candidate in self.candidates if candidate.kind == "temporal")

    def resolved_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "project_config": self.project_config.as_posix(),
            "output_dir": self.output_dir.as_posix(),
            "candidates": [
                {
                    "experiment_id": candidate.experiment_id,
                    "kind": candidate.kind,
                    "artifact_dir": candidate.artifact_dir.as_posix(),
                    "experiment_config": candidate.experiment_config.as_posix(),
                    "role": candidate.role,
                }
                for candidate in self.candidates
            ],
            "calibration_methods": list(self.calibration_methods),
            "weight_grid_step": self.weight_grid_step,
            "fixed_tree_weights": list(self.fixed_tree_weights),
            "rebuild_missing_oof": self.rebuild_missing_oof,
            "overwrite_oof": self.overwrite_oof,
            "tta_enabled": self.tta_enabled,
            "prior_shift_correction_enabled": self.prior_shift_correction_enabled,
            "threshold": self.threshold,
            "source_commit": self.source_commit,
            "fingerprint": self.fingerprint,
        }


def _project_root(source: Path) -> Path:
    for parent in (source.parent, *source.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FinalConfigError(f"unable to locate project root from {source}")


def _resolve(root: Path, value: object, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FinalConfigError(f"'{key}' must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_final_delivery_config(path: str | Path) -> FinalDeliveryConfig:
    """Load one immutable final-delivery declaration."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"final delivery configuration not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("final"), dict):
        raise FinalConfigError("final configuration must contain a 'final' mapping")
    value = raw["final"]
    root = _project_root(source)
    candidates_raw = value.get("candidates")
    if not isinstance(candidates_raw, list):
        raise FinalConfigError("final.candidates must be a list")
    candidates: list[FinalCandidateConfig] = []
    for index, item in enumerate(candidates_raw):
        if not isinstance(item, dict):
            raise FinalConfigError("each final candidate must be a mapping")
        candidates.append(
            FinalCandidateConfig(
                experiment_id=str(item.get("experiment_id", "")),
                kind=str(item.get("kind", "")),  # type: ignore[arg-type]
                artifact_dir=_resolve(
                    root,
                    item.get("artifact_dir"),
                    f"candidates[{index}].artifact_dir",
                ),
                experiment_config=_resolve(
                    root,
                    item.get("experiment_config"),
                    f"candidates[{index}].experiment_config",
                ),
                role=str(item.get("role", "")),
            )
        )
    methods_raw = value.get("calibration_methods", ["none", "sigmoid", "beta"])
    if not isinstance(methods_raw, list) or not all(isinstance(item, str) for item in methods_raw):
        raise FinalConfigError("final.calibration_methods must be a list of strings")
    weights_raw = value.get("fixed_tree_weights", [0.5, 0.7])
    if not isinstance(weights_raw, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in weights_raw
    ):
        raise FinalConfigError("final.fixed_tree_weights must be numeric")
    canonical = {
        "schema_version": 1,
        "project_config": str(value.get("project_config", "configs/base.yaml")),
        "output_dir": str(value.get("output_dir", "artifacts/final")),
        "candidates": candidates_raw,
        "calibration_methods": methods_raw,
        "weight_grid_step": float(value.get("weight_grid_step", 0.01)),
        "fixed_tree_weights": [float(item) for item in weights_raw],
        "rebuild_missing_oof": bool(value.get("rebuild_missing_oof", True)),
        "overwrite_oof": bool(value.get("overwrite_oof", False)),
        "tta_enabled": bool(value.get("tta_enabled", False)),
        "prior_shift_correction_enabled": bool(
            value.get("prior_shift_correction_enabled", False)
        ),
        "threshold": float(value.get("threshold", FIXED_THRESHOLD)),
        "source_commit": (str(value.get("source_commit")) if value.get("source_commit") else None),
    }
    fingerprint = hashlib.sha256(_canonical(canonical).encode()).hexdigest()
    return FinalDeliveryConfig(
        source_path=source,
        project_config=_resolve(root, canonical["project_config"], "project_config"),
        output_dir=_resolve(root, canonical["output_dir"], "output_dir"),
        candidates=tuple(candidates),
        calibration_methods=tuple(methods_raw),  # type: ignore[arg-type]
        weight_grid_step=float(canonical["weight_grid_step"]),
        fixed_tree_weights=tuple(float(item) for item in weights_raw),
        rebuild_missing_oof=bool(canonical["rebuild_missing_oof"]),
        overwrite_oof=bool(canonical["overwrite_oof"]),
        tta_enabled=bool(canonical["tta_enabled"]),
        prior_shift_correction_enabled=bool(canonical["prior_shift_correction_enabled"]),
        threshold=float(canonical["threshold"]),
        source_commit=(str(canonical["source_commit"]) if canonical["source_commit"] else None),
        fingerprint=fingerprint,
    )
