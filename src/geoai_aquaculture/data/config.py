"""Typed configuration loading for competition data access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from geoai_aquaculture.constants import (
    ID_COLUMN,
    MISSING_SENTINEL,
    OPTICAL_BANDS,
    RADAR_BANDS,
    TARGET_COLUMN,
)


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the data contract."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Resolved paths and immutable schema choices for the competition files."""

    train_path: Path
    test_path: Path
    sample_submission_path: Path
    id_column: str = ID_COLUMN
    target_column: str = TARGET_COLUMN
    missing_sentinel: float = MISSING_SENTINEL
    months: int = 12
    radar_bands: tuple[str, ...] = RADAR_BANDS
    optical_bands: tuple[str, ...] = OPTICAL_BANDS

    @property
    def bands(self) -> tuple[str, ...]:
        """Return bands in the configured, competition-defined order."""

        return self.radar_bands + self.optical_bands


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Phase-independent project settings needed by the data audit."""

    source_path: Path
    project_root: Path
    project_name: str
    seed: int
    data: DataConfig
    artifacts_dir: Path


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"configuration key '{key}' must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required configuration key '{section}.{key}'")
    return mapping[key]


def _string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"configuration key '{key}' must be a non-empty string")
    return value


def _string_tuple(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"configuration key '{key}' must be a non-empty list")
    result = tuple(_string(item, key) for item in value)
    if len(result) != len(set(result)):
        raise ConfigError(f"configuration key '{key}' contains duplicate bands")
    return result


def _find_project_root(config_path: Path) -> Path:
    """Find the nearest repository root, falling back to the config directory."""

    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent


def _resolve_path(value: object, key: str, project_root: Path) -> Path:
    path = Path(_string(value, key)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate the subset of YAML used by ingestion and auditing."""

    source_path = Path(config_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {source_path}")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "root")
    project = _mapping(_required(root, "project", "root"), "project")
    data = _mapping(_required(root, "data", "root"), "data")
    reporting = _mapping(_required(root, "reporting", "root"), "reporting")
    project_root = _find_project_root(source_path)

    months = _required(data, "months", "data")
    if not isinstance(months, int) or isinstance(months, bool) or months != 12:
        raise ConfigError("configuration key 'data.months' must equal 12")

    sentinel = _required(data, "missing_sentinel", "data")
    if not isinstance(sentinel, int | float) or isinstance(sentinel, bool):
        raise ConfigError("configuration key 'data.missing_sentinel' must be numeric")
    if not np.isfinite(float(sentinel)):
        raise ConfigError("configuration key 'data.missing_sentinel' must be finite")

    radar_bands = _string_tuple(_required(data, "radar_bands", "data"), "data.radar_bands")
    optical_bands = _string_tuple(_required(data, "optical_bands", "data"), "data.optical_bands")
    if radar_bands != RADAR_BANDS:
        raise ConfigError(f"data.radar_bands must be exactly {list(RADAR_BANDS)}")
    if optical_bands != OPTICAL_BANDS:
        raise ConfigError(f"data.optical_bands must be exactly {list(OPTICAL_BANDS)}")

    id_column = _string(_required(data, "id_column", "data"), "data.id_column")
    target_column = _string(_required(data, "target_column", "data"), "data.target_column")
    if id_column != ID_COLUMN or target_column != TARGET_COLUMN:
        raise ConfigError(f"data columns must be id='{ID_COLUMN}' and target='{TARGET_COLUMN}'")

    seed = _required(project, "seed", "project")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("configuration key 'project.seed' must be a non-negative integer")

    data_config = DataConfig(
        train_path=_resolve_path(
            _required(data, "train_path", "data"), "data.train_path", project_root
        ),
        test_path=_resolve_path(
            _required(data, "test_path", "data"), "data.test_path", project_root
        ),
        sample_submission_path=_resolve_path(
            _required(data, "sample_submission_path", "data"),
            "data.sample_submission_path",
            project_root,
        ),
        id_column=id_column,
        target_column=target_column,
        missing_sentinel=float(sentinel),
        months=months,
        radar_bands=radar_bands,
        optical_bands=optical_bands,
    )
    artifacts_root = _resolve_path(
        _required(reporting, "artifacts_dir", "reporting"),
        "reporting.artifacts_dir",
        project_root,
    )

    return ProjectConfig(
        source_path=source_path,
        project_root=project_root,
        project_name=_string(_required(project, "name", "project"), "project.name"),
        seed=seed,
        data=data_config,
        artifacts_dir=artifacts_root,
    )
