"""Typed configuration loading for competition data access."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from geoai_aquaculture.constants import (
    FIXED_THRESHOLD,
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
class SeasonDefinition:
    """A named, non-overlapping group of valid window-start months."""

    name: str
    start_months: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("validation season names must be non-empty")
        if not self.start_months or any(month < 1 or month > 9 for month in self.start_months):
            raise ConfigError("validation season start months must be within 1-9")
        if len(self.start_months) != len(set(self.start_months)):
            raise ConfigError("validation season start months must be unique")


@dataclass(frozen=True, slots=True)
class RobustScoreWeights:
    """Fixed weights for the Phase 4 robust model-comparison diagnostic."""

    mean_combined: float = 0.50
    worst_fold: float = 0.20
    worst_window_length: float = 0.15
    worst_season: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.mean_combined,
            self.worst_fold,
            self.worst_window_length,
            self.worst_season,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ConfigError("validation.robust_score_weights must be finite and non-negative")
        if not np.isclose(sum(values), 1.0, atol=1e-12):
            raise ConfigError("validation.robust_score_weights must sum to one")


DEFAULT_SEASONS = (
    SeasonDefinition("early_year", (1, 2, 3)),
    SeasonDefinition("mid_year", (4, 5, 6)),
    SeasonDefinition("late_year", (7, 8, 9)),
)


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Authoritative Phase 4 splitting, views, aggregation, and scoring policy."""

    strategy: str = "stratified_group_kfold"
    seed: int = 2026
    n_splits: int = 5
    n_repeats: int = 3
    fixed_threshold: float = FIXED_THRESHOLD
    window_lengths: tuple[int, ...] = (4, 5, 6)
    split_before_augmentation: bool = True
    primary_window_mode: Literal["sampled"] = "sampled"
    sampled_windows_per_original: int = 8
    validation_window_seed: int = 2027
    aggregation_method: Literal["mean", "median", "logit_mean", "trimmed_mean"] = "mean"
    trimmed_mean_fraction: float = 0.10
    seasons: tuple[SeasonDefinition, ...] = DEFAULT_SEASONS
    optical_severe_limit: float = 0.50
    optical_high_completeness: float = 0.80
    robust_score_weights: RobustScoreWeights = field(default_factory=RobustScoreWeights)
    exhaustive_stress_enabled: bool = False
    reference_estimator_enabled: bool = True
    similarity_holdout_fraction: float = 0.20
    similarity_holdout_min_samples: int = 100
    cluster_n_clusters: int = 5
    cluster_min_size: int = 100

    def __post_init__(self) -> None:
        if self.strategy != "stratified_group_kfold":
            raise ConfigError("validation.strategy must be 'stratified_group_kfold'")
        if self.seed < 0 or self.validation_window_seed < 0:
            raise ConfigError("validation seeds must be non-negative")
        if self.n_splits < 2 or self.n_repeats < 1:
            raise ConfigError("validation requires at least two folds and one repeat")
        if self.fixed_threshold != FIXED_THRESHOLD:
            raise ConfigError(f"validation.threshold must remain exactly {FIXED_THRESHOLD}")
        if self.window_lengths != (4, 5, 6):
            raise ConfigError("validation.window_lengths must remain exactly (4, 5, 6)")
        if not self.split_before_augmentation:
            raise ConfigError("validation splitting must precede temporal augmentation")
        if self.primary_window_mode != "sampled":
            raise ConfigError("validation.primary_window_mode must remain 'sampled'")
        if self.sampled_windows_per_original < 1:
            raise ConfigError("validation sampled window count must be positive")
        if self.aggregation_method not in {"mean", "median", "logit_mean", "trimmed_mean"}:
            raise ConfigError("validation.aggregation_method is unsupported")
        if not 0.0 <= self.trimmed_mean_fraction < 0.5:
            raise ConfigError("validation.trimmed_mean_fraction must be in [0, 0.5)")
        occupied = [month for season in self.seasons for month in season.start_months]
        if sorted(occupied) != list(range(1, 10)):
            raise ConfigError("validation seasons must partition valid start months 1-9")
        if not 0.0 < self.optical_severe_limit < self.optical_high_completeness < 1.0:
            raise ConfigError("validation optical completeness cutoffs are invalid")
        if not 0.0 < self.similarity_holdout_fraction <= 1.0:
            raise ConfigError("validation similarity holdout fraction must be in (0, 1]")
        if (
            min(
                self.similarity_holdout_min_samples,
                self.cluster_min_size,
            )
            < 1
        ):
            raise ConfigError("validation diagnostic sizes must be positive")
        if self.cluster_n_clusters < 2:
            raise ConfigError("validation cluster count must be at least two")

    @property
    def threshold(self) -> float:
        """Expose the immutable threshold under the Phase 4 public name."""

        return self.fixed_threshold


@dataclass(frozen=True, slots=True)
class WindowGenerationConfig:
    """Configuration for exhaustive or sampled temporal views."""

    enabled: bool = False
    use_test_missingness_masks: bool = False
    exhaustive_windows: bool = False
    windows_per_sample: int = 8
    temporal_dropout_enabled: bool = False
    temporal_dropout_probability: float = 0.0
    optical_dropout_enabled: bool = False
    optical_dropout_probability: float = 0.0

    @property
    def mode(self) -> Literal["exhaustive", "sampled"]:
        """Return the configured generation mode."""

        return "exhaustive" if self.exhaustive_windows else "sampled"


@dataclass(frozen=True, slots=True)
class BandSemanticMapping:
    """Explicit, validated mapping from scientific roles to raw configured bands."""

    vv: str = "VV"
    vh: str = "VH"
    blue: str = "blue"
    green: str = "green"
    red: str = "red"
    red_edge_1: str = "re1"
    red_edge_2: str = "re2"
    red_edge_3: str = "re3"
    nir: str = "nir"
    narrow_nir: str = "nira"
    swir1: str = "swir1"
    swir2: str = "swir2"

    def __post_init__(self) -> None:
        values = tuple(self.roles.values())
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ConfigError("features.bands roles must be non-empty strings")
        if len(values) != len(set(values)):
            raise ConfigError("features.bands must map every semantic role unambiguously")

    @property
    def roles(self) -> dict[str, str]:
        """Return semantic roles in stable scientific order."""

        return {
            "vv": self.vv,
            "vh": self.vh,
            "blue": self.blue,
            "green": self.green,
            "red": self.red,
            "red_edge_1": self.red_edge_1,
            "red_edge_2": self.red_edge_2,
            "red_edge_3": self.red_edge_3,
            "nir": self.nir,
            "narrow_nir": self.narrow_nir,
            "swir1": self.swir1,
            "swir2": self.swir2,
        }


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Numerical and semantic choices for deterministic Phase 3 features."""

    version: str = "phase3_v1"
    epsilon: float = 1e-6
    bands: BandSemanticMapping = field(default_factory=BandSemanticMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ConfigError("features.version must be a non-empty string")
        if (
            not isinstance(self.epsilon, int | float)
            or isinstance(self.epsilon, bool)
            or not np.isfinite(float(self.epsilon))
            or self.epsilon <= 0.0
        ):
            raise ConfigError("features.epsilon must be finite and > 0")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Phase-independent project settings needed by the data audit."""

    source_path: Path
    project_root: Path
    project_name: str
    seed: int
    data: DataConfig
    validation: ValidationConfig
    augmentation: WindowGenerationConfig
    features: FeatureConfig
    artifacts_dir: Path


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"configuration key '{key}' must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required configuration key '{section}.{key}'")
    return mapping[key]


def _optional_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key, {})
    return _mapping(value, key)


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


def _boolean(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"configuration key '{key}' must be boolean")
    return value


def _positive_integer(value: object, key: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"configuration key '{key}' must be an integer >= {minimum}")
    return value


def _probability(value: object, key: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"configuration key '{key}' must be numeric")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ConfigError(f"configuration key '{key}' must be in [0, 1]")
    return result


def _positive_float(value: object, key: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"configuration key '{key}' must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigError(f"configuration key '{key}' must be finite and > 0")
    return result


def _non_negative_float(value: object, key: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"configuration key '{key}' must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ConfigError(f"configuration key '{key}' must be finite and >= 0")
    return result


def _seasons(value: object) -> tuple[SeasonDefinition, ...]:
    mapping = _mapping(value, "validation.seasons")
    seasons: list[SeasonDefinition] = []
    occupied: set[int] = set()
    for name, raw_months in mapping.items():
        if not isinstance(raw_months, list) or any(
            not isinstance(month, int) or isinstance(month, bool) for month in raw_months
        ):
            raise ConfigError("validation.seasons values must be lists of integer months")
        season = SeasonDefinition(_string(name, "validation.seasons name"), tuple(raw_months))
        overlap = occupied.intersection(season.start_months)
        if overlap:
            raise ConfigError(f"validation.seasons overlap on start months: {sorted(overlap)}")
        occupied.update(season.start_months)
        seasons.append(season)
    if occupied != set(range(1, 10)):
        raise ConfigError("validation.seasons must partition valid start months 1-9")
    return tuple(seasons)


def _window_lengths(value: object, key: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ConfigError(f"configuration key '{key}' must be a list of integers")
    result = tuple(value)
    if result != (4, 5, 6):
        raise ConfigError(f"configuration key '{key}' must be exactly [4, 5, 6]")
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
    validation = _optional_mapping(root, "validation")
    augmentation = _optional_mapping(root, "augmentation")
    features = _optional_mapping(root, "features")
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

    threshold_value = validation.get(
        "threshold",
        validation.get("fixed_threshold", FIXED_THRESHOLD),
    )
    if (
        "threshold" in validation
        and "fixed_threshold" in validation
        and float(validation["threshold"]) != float(validation["fixed_threshold"])
    ):
        raise ConfigError("validation.threshold and fixed_threshold must agree")
    robust_raw = _optional_mapping(validation, "robust_score_weights")
    seasons_value = validation.get(
        "seasons",
        {season.name: list(season.start_months) for season in DEFAULT_SEASONS},
    )
    validation_config = ValidationConfig(
        strategy=_string(
            validation.get("strategy", "stratified_group_kfold"),
            "validation.strategy",
        ),
        seed=_positive_integer(validation.get("seed", 2026), "validation.seed", minimum=0),
        n_splits=_positive_integer(validation.get("n_splits", 5), "validation.n_splits", minimum=2),
        n_repeats=_positive_integer(validation.get("n_repeats", 3), "validation.n_repeats"),
        fixed_threshold=_probability(
            threshold_value,
            "validation.threshold",
        ),
        window_lengths=_window_lengths(
            validation.get("window_lengths", [4, 5, 6]),
            "validation.window_lengths",
        ),
        split_before_augmentation=_boolean(
            validation.get("split_before_augmentation", True),
            "validation.split_before_augmentation",
        ),
        primary_window_mode=_string(
            validation.get("primary_window_mode", "sampled"),
            "validation.primary_window_mode",
        ),
        sampled_windows_per_original=_positive_integer(
            validation.get("sampled_windows_per_original", 8),
            "validation.sampled_windows_per_original",
        ),
        validation_window_seed=_positive_integer(
            validation.get("validation_window_seed", 2027),
            "validation.validation_window_seed",
            minimum=0,
        ),
        aggregation_method=_string(
            validation.get("aggregation_method", "mean"),
            "validation.aggregation_method",
        ),
        trimmed_mean_fraction=_probability(
            validation.get("trimmed_mean_fraction", 0.10),
            "validation.trimmed_mean_fraction",
        ),
        seasons=_seasons(seasons_value),
        optical_severe_limit=_probability(
            validation.get("optical_severe_limit", 0.50),
            "validation.optical_severe_limit",
        ),
        optical_high_completeness=_probability(
            validation.get("optical_high_completeness", 0.80),
            "validation.optical_high_completeness",
        ),
        robust_score_weights=RobustScoreWeights(
            mean_combined=_non_negative_float(
                robust_raw.get("mean_combined", 0.50),
                "validation.robust_score_weights.mean_combined",
            ),
            worst_fold=_non_negative_float(
                robust_raw.get("worst_fold", 0.20),
                "validation.robust_score_weights.worst_fold",
            ),
            worst_window_length=_non_negative_float(
                robust_raw.get("worst_window_length", 0.15),
                "validation.robust_score_weights.worst_window_length",
            ),
            worst_season=_non_negative_float(
                robust_raw.get("worst_season", 0.15),
                "validation.robust_score_weights.worst_season",
            ),
        ),
        exhaustive_stress_enabled=_boolean(
            validation.get("exhaustive_stress_enabled", False),
            "validation.exhaustive_stress_enabled",
        ),
        reference_estimator_enabled=_boolean(
            validation.get("reference_estimator_enabled", True),
            "validation.reference_estimator_enabled",
        ),
        similarity_holdout_fraction=_probability(
            validation.get("similarity_holdout_fraction", 0.20),
            "validation.similarity_holdout_fraction",
        ),
        similarity_holdout_min_samples=_positive_integer(
            validation.get("similarity_holdout_min_samples", 100),
            "validation.similarity_holdout_min_samples",
        ),
        cluster_n_clusters=_positive_integer(
            validation.get("cluster_n_clusters", 5),
            "validation.cluster_n_clusters",
            minimum=2,
        ),
        cluster_min_size=_positive_integer(
            validation.get("cluster_min_size", 100),
            "validation.cluster_min_size",
        ),
    )
    if validation_config.strategy != "stratified_group_kfold":
        raise ConfigError("validation.strategy must be 'stratified_group_kfold'")
    if validation_config.fixed_threshold != FIXED_THRESHOLD:
        raise ConfigError(f"validation.threshold must remain exactly {FIXED_THRESHOLD}")
    if not validation_config.split_before_augmentation:
        raise ConfigError("validation.split_before_augmentation must remain true")
    if validation_config.primary_window_mode != "sampled":
        raise ConfigError("validation.primary_window_mode must remain 'sampled'")
    if validation_config.aggregation_method not in {
        "mean",
        "median",
        "logit_mean",
        "trimmed_mean",
    }:
        raise ConfigError("validation.aggregation_method is unsupported")
    if not 0.0 <= validation_config.trimmed_mean_fraction < 0.5:
        raise ConfigError("validation.trimmed_mean_fraction must be in [0, 0.5)")
    if not (
        0.0
        < validation_config.optical_severe_limit
        < validation_config.optical_high_completeness
        < 1.0
    ):
        raise ConfigError(
            "validation optical completeness cutoffs must satisfy 0 < severe < high < 1"
        )

    window_config = WindowGenerationConfig(
        enabled=_boolean(augmentation.get("enabled", False), "augmentation.enabled"),
        use_test_missingness_masks=_boolean(
            augmentation.get("use_test_missingness_masks", False),
            "augmentation.use_test_missingness_masks",
        ),
        exhaustive_windows=_boolean(
            augmentation.get("exhaustive_windows", False),
            "augmentation.exhaustive_windows",
        ),
        windows_per_sample=_positive_integer(
            augmentation.get("windows_per_sample", 8),
            "augmentation.windows_per_sample",
        ),
        temporal_dropout_enabled=_boolean(
            augmentation.get("temporal_dropout_enabled", False),
            "augmentation.temporal_dropout_enabled",
        ),
        temporal_dropout_probability=_probability(
            augmentation.get("temporal_dropout_probability", 0.0),
            "augmentation.temporal_dropout_probability",
        ),
        optical_dropout_enabled=_boolean(
            augmentation.get("optical_dropout_enabled", False),
            "augmentation.optical_dropout_enabled",
        ),
        optical_dropout_probability=_probability(
            augmentation.get("optical_dropout_probability", 0.0),
            "augmentation.optical_dropout_probability",
        ),
    )
    if window_config.exhaustive_windows and window_config.use_test_missingness_masks:
        raise ConfigError(
            "test missingness masks are sampled views and cannot be combined with exhaustive mode"
        )

    semantic = _optional_mapping(features, "bands")
    default_semantic = BandSemanticMapping()
    semantic_mapping = BandSemanticMapping(
        **{
            role: _string(semantic.get(role, raw_band), f"features.bands.{role}")
            for role, raw_band in default_semantic.roles.items()
        }
    )
    semantic_values = tuple(semantic_mapping.roles.values())
    if len(semantic_values) != len(set(semantic_values)):
        raise ConfigError("features.bands must map every semantic role unambiguously")
    if {semantic_mapping.vv, semantic_mapping.vh} != set(data_config.radar_bands):
        raise ConfigError("features.bands vv/vh roles must exactly map configured radar bands")
    optical_semantics = set(semantic_values) - {semantic_mapping.vv, semantic_mapping.vh}
    if optical_semantics != set(data_config.optical_bands):
        raise ConfigError("features.bands optical roles must exactly map configured optical bands")
    feature_config = FeatureConfig(
        version=_string(features.get("version", "phase3_v1"), "features.version"),
        epsilon=_positive_float(features.get("epsilon", 1e-6), "features.epsilon"),
        bands=semantic_mapping,
    )

    return ProjectConfig(
        source_path=source_path,
        project_root=project_root,
        project_name=_string(_required(project, "name", "project"), "project.name"),
        seed=seed,
        data=data_config,
        validation=validation_config,
        augmentation=window_config,
        features=feature_config,
        artifacts_dir=artifacts_root,
    )
