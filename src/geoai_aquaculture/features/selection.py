"""Registry-backed Phase 5 tabular feature-set declarations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import pandas as pd

from geoai_aquaculture.data import BandSemanticMapping

from .indices import FeatureEngineeringError
from .registry import FeatureDefinition, FeatureRegistry
from .representations import FeatureMatrix

FeatureSetName = Literal["relative", "invariant", "full", "radar", "optical", "compact"]

FEATURE_SET_HYPOTHESES: dict[FeatureSetName, str] = {
    "relative": (
        "Relative-position values retain short-window dynamics while explicit registry metadata "
        "represents padding and sensor availability."
    ),
    "invariant": (
        "Gap-aware temporal aggregates should be more robust than raw positions to seasonal "
        "placement and incomplete optical observations."
    ),
    "full": (
        "Combining raw relative positions, invariant aggregates, and missingness metadata may let "
        "trees choose complementary temporal evidence."
    ),
    "radar": (
        "A radar-only expert should remain stable under optical gaps and may provide complementary "
        "errors to fused models."
    ),
    "optical": (
        "Optical bands and physical indices may isolate water signatures that radar alone misses, "
        "while retaining optical-validity metadata."
    ),
    "compact": (
        "A manually declared set of water, vegetation, radar, robust temporal, and missingness "
        "features may generalize better than the 688-column representation."
    ),
}

_RADAR_GROUPS = {
    "radar_raw",
    "radar_derived",
    "radar_temporal",
    "radar_stability",
    "relative_radar_raw",
    "relative_radar_derived",
    "relative_radar_temporal",
}
_OPTICAL_GROUPS = {
    "optical_raw",
    "optical_index",
    "relative_optical_raw",
    "relative_optical_index",
}
_COMPACT_CHANNELS = (
    "radar__vv",
    "radar__vh",
    "radar__vv_minus_vh",
    "radar__vv_plus_vh",
    "optical__ndwi",
    "optical__mndwi",
    "optical__ndmi",
    "optical__nbr",
    "optical__ndvi",
    "optical__ndre1",
    "optical__ndre2",
    "optical__chlorophyll_red_edge",
)
_COMPACT_AGGREGATIONS = (
    "valid_count",
    "median",
    "std",
    "amplitude",
    "iqr",
    "first_to_last",
    "slope",
)
_COMPACT_METADATA = (
    "metadata__window_length",
    "metadata__start_month",
    "metadata__end_month",
    "metadata__start_month_sin",
    "metadata__start_month_cos",
    "metadata__end_month_sin",
    "metadata__end_month_cos",
    "metadata__relative_position_count",
    "metadata__radar_valid_count",
    "metadata__optical_valid_count",
    "metadata__optical_gap_count",
    "metadata__longest_optical_valid_run",
    "metadata__longest_optical_missing_run",
    "metadata__radar_valid_proportion",
    "metadata__optical_valid_proportion",
)
_COMPACT_STABILITY = (
    "radar__vv__mean_abs_first_difference",
    "radar__vh__mean_abs_first_difference",
)
_FORBIDDEN_FEATURE_TOKENS = {"id", "original_id", "window_id", "fold", "label", "target"}


@dataclass(frozen=True, slots=True)
class SelectedFeatureMatrix:
    """One deterministic model matrix selected only through registry provenance."""

    selection: FeatureSetName
    features: pd.DataFrame
    feature_names: tuple[str, ...]
    registry: FeatureRegistry
    feature_groups: MappingProxyType[str, tuple[str, ...]]
    full_schema_fingerprint: str
    schema_fingerprint: str
    hypothesis: str

    def __post_init__(self) -> None:
        if tuple(self.features.columns) != self.feature_names:
            raise FeatureEngineeringError("selected features are not in declared registry order")
        if self.registry.feature_names != self.feature_names:
            raise FeatureEngineeringError("selected registry does not match selected columns")
        if self.features.empty or not self.feature_names:
            raise FeatureEngineeringError("selected feature set must not be empty")
        if self.features.isin([float("inf"), float("-inf")]).any().any():
            raise FeatureEngineeringError("selected features must not contain infinity")
        offending = [
            name
            for name in self.feature_names
            if _FORBIDDEN_FEATURE_TOKENS & {token.casefold() for token in name.split("__")}
        ]
        if offending:
            raise FeatureEngineeringError(f"selected model features contain metadata: {offending}")


def _metadata_for_sensor(
    definition: FeatureDefinition,
    *,
    sensor_bands: set[str],
) -> bool:
    if definition.feature_kind != "metadata":
        return False
    if definition.feature_group == "metadata_window":
        return True
    sources = set(definition.source_bands)
    if definition.feature_group == "metadata_position" and not sources:
        return True
    return bool(sources) and sources.issubset(sensor_bands)


def _compact_names(registry: FeatureRegistry) -> tuple[str, ...]:
    expected = {
        *(
            f"{channel}__{statistic}"
            for channel in _COMPACT_CHANNELS
            for statistic in _COMPACT_AGGREGATIONS
        ),
        *_COMPACT_METADATA,
        *_COMPACT_STABILITY,
    }
    available = set(registry.feature_names)
    missing = sorted(expected - available)
    if missing:
        raise FeatureEngineeringError(f"compact physical registry entries are missing: {missing}")
    return tuple(name for name in registry.feature_names if name in expected)


def _selected_names(
    matrix: FeatureMatrix,
    selection: FeatureSetName,
    semantics: BandSemanticMapping,
) -> tuple[str, ...]:
    definitions = matrix.registry.definitions
    if selection == "full":
        return matrix.feature_names
    if selection == "relative":
        return tuple(
            definition.name
            for definition in definitions
            if definition.feature_kind in {"monthly", "metadata"}
        )
    if selection == "invariant":
        return tuple(
            definition.name
            for definition in definitions
            if definition.feature_kind in {"aggregate", "metadata"}
        )
    if selection == "compact":
        return _compact_names(matrix.registry)

    radar_bands = {semantics.vv, semantics.vh}
    optical_bands = set(semantics.roles.values()) - radar_bands
    if selection == "radar":
        return tuple(
            definition.name
            for definition in definitions
            if definition.feature_group in _RADAR_GROUPS
            or _metadata_for_sensor(definition, sensor_bands=radar_bands)
        )
    if selection == "optical":
        return tuple(
            definition.name
            for definition in definitions
            if definition.feature_group in _OPTICAL_GROUPS
            or _metadata_for_sensor(definition, sensor_bands=optical_bands)
        )
    raise FeatureEngineeringError(f"unsupported registry-backed feature selection: {selection}")


def _selection_fingerprint(
    selection: FeatureSetName,
    definitions: tuple[FeatureDefinition, ...],
) -> str:
    registry = FeatureRegistry(definitions)
    digest = hashlib.sha256()
    digest.update(selection.encode())
    digest.update(registry.fingerprint.encode())
    digest.update("\x1f".join(registry.feature_names).encode())
    return digest.hexdigest()


def select_tabular_features(
    matrix: FeatureMatrix,
    selection: FeatureSetName,
    semantics: BandSemanticMapping,
) -> SelectedFeatureMatrix:
    """Select a declared Phase 5 representation without consulting labels or values."""

    names = _selected_names(matrix, selection, semantics)
    selected_set = set(names)
    definitions = tuple(
        definition for definition in matrix.registry.definitions if definition.name in selected_set
    )
    if tuple(definition.name for definition in definitions) != names:
        raise FeatureEngineeringError("feature selection changed deterministic registry ordering")
    registry = FeatureRegistry(definitions)
    groups: dict[str, list[str]] = {}
    for definition in definitions:
        groups.setdefault(definition.feature_group, []).append(definition.name)
    return SelectedFeatureMatrix(
        selection=selection,
        features=matrix.features.loc[:, list(names)].copy(),
        feature_names=names,
        registry=registry,
        feature_groups=MappingProxyType(
            {group: tuple(group_names) for group, group_names in groups.items()}
        ),
        full_schema_fingerprint=matrix.schema_fingerprint,
        schema_fingerprint=_selection_fingerprint(selection, definitions),
        hypothesis=FEATURE_SET_HYPOTHESES[selection],
    )
