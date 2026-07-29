"""Deterministic tabular and masked sequence feature representations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from geoai_aquaculture.data import FeatureConfig, TemporalWindowDataset

from .aggregation import AGGREGATION_NAMES, aggregate_temporal_series
from .indices import (
    FeatureEngineeringError,
    MonthlyFeatureCollection,
    build_monthly_features,
)
from .registry import FeatureDefinition, FeatureRegistry


class _HashDigest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _hash_array(digest: _HashDigest, array: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(array).tobytes())


def _hash_strings(digest: _HashDigest, values: np.ndarray) -> None:
    payload = "\x1f".join(str(value) for value in values.tolist()).encode()
    digest.update(payload)


def _schema_fingerprint(names: tuple[str, ...], registry: FeatureRegistry) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(names).encode())
    digest.update(registry.fingerprint.encode())
    return digest.hexdigest()


def _feature_groups(registry: FeatureRegistry) -> Mapping[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for definition in registry.definitions:
        groups.setdefault(definition.feature_group, []).append(definition.name)
    return MappingProxyType({group: tuple(names) for group, names in groups.items()})


def _attached_metadata(
    windows: TemporalWindowDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    required = {"original_id", "window_id", "fold"}
    missing = sorted(required - set(windows.manifest.columns))
    if missing:
        raise FeatureEngineeringError(f"window manifest is missing metadata: {missing}")
    original_ids = windows.manifest["original_id"].astype("string").to_numpy()
    window_ids = windows.manifest["window_id"].astype("string").to_numpy()
    folds = windows.manifest["fold"].to_numpy(dtype=np.int16)
    labels = (
        windows.manifest["label"].to_numpy(dtype=np.int8)
        if "label" in windows.manifest.columns
        else None
    )
    return original_ids, window_ids, folds, labels


def _validate_identity_folds(original_ids: np.ndarray, folds: np.ndarray) -> None:
    attached = pd.DataFrame({"original_id": original_ids, "fold": folds})
    if not attached.groupby("original_id", sort=False)["fold"].nunique().eq(1).all():
        raise FeatureEngineeringError("feature metadata contains cross-fold original IDs")


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Tabular model features with identity and fold data kept out-of-band."""

    features: pd.DataFrame
    feature_names: tuple[str, ...]
    original_ids: np.ndarray
    window_ids: np.ndarray
    folds: np.ndarray
    labels: np.ndarray | None
    registry: FeatureRegistry
    feature_groups: Mapping[str, tuple[str, ...]]
    fingerprint: str
    schema_fingerprint: str

    def __post_init__(self) -> None:
        rows = self.features.shape[0]
        if tuple(self.features.columns) != self.feature_names:
            raise FeatureEngineeringError("tabular columns must match ordered feature names")
        if self.registry.feature_names != self.feature_names:
            raise FeatureEngineeringError("tabular registry must exactly match feature columns")
        if any(len(values) != rows for values in (self.original_ids, self.window_ids, self.folds)):
            raise FeatureEngineeringError("tabular metadata must align with feature rows")
        if self.labels is not None and len(self.labels) != rows:
            raise FeatureEngineeringError("tabular labels must align with feature rows")
        if any(not is_numeric_dtype(dtype) for dtype in self.features.dtypes):
            raise FeatureEngineeringError("tabular model features must all be numeric")
        if np.isinf(self.features.to_numpy(dtype=np.float64)).any():
            raise FeatureEngineeringError("tabular model features must not contain infinity")
        forbidden_tokens = {"id", "original_id", "window_id", "fold", "label", "target"}
        offending = [
            name
            for name in self.feature_names
            if forbidden_tokens & {token.casefold() for token in name.split("__")}
        ]
        if offending:
            raise FeatureEngineeringError(
                f"identity, fold, and label columns cannot be features: {offending}"
            )
        if len(set(self.window_ids.tolist())) != rows:
            raise FeatureEngineeringError("window IDs attached to features must be unique")
        _validate_identity_folds(self.original_ids, self.folds)


@dataclass(frozen=True, slots=True)
class SequenceFeatureDataset:
    """Raw sensors and monthly indices with separate masks and explicit padding."""

    radar_values: np.ndarray
    optical_values: np.ndarray
    monthly_indices: np.ndarray
    relative_positions: np.ndarray
    calendar_months: np.ndarray
    absolute_month_encoding: np.ndarray
    radar_mask: np.ndarray
    radar_feature_mask: np.ndarray
    optical_mask: np.ndarray
    optical_band_mask: np.ndarray
    raw_band_mask: np.ndarray
    index_mask: np.ndarray
    padding_mask: np.ndarray
    radar_feature_names: tuple[str, ...]
    optical_feature_names: tuple[str, ...]
    index_feature_names: tuple[str, ...]
    raw_band_names: tuple[str, ...]
    original_ids: np.ndarray
    window_ids: np.ndarray
    folds: np.ndarray
    labels: np.ndarray | None
    registry: FeatureRegistry
    fingerprint: str
    schema_fingerprint: str

    def __post_init__(self) -> None:
        rows = len(self.original_ids)
        positions = (rows, 6)
        expected_radar = (*positions, len(self.radar_feature_names))
        expected_optical = (*positions, len(self.optical_feature_names))
        expected_indices = (*positions, len(self.index_feature_names))
        expected_raw_bands = (*positions, len(self.raw_band_names))
        if self.radar_values.shape != expected_radar:
            raise FeatureEngineeringError("radar sequence shape does not match channel names")
        if self.optical_values.shape != expected_optical:
            raise FeatureEngineeringError("optical sequence shape does not match channel names")
        if self.monthly_indices.shape != expected_indices:
            raise FeatureEngineeringError("index sequence shape does not match channel names")
        if self.radar_feature_mask.shape != expected_radar:
            raise FeatureEngineeringError("radar feature-mask shape is invalid")
        if self.optical_band_mask.shape != expected_optical:
            raise FeatureEngineeringError("optical band-mask shape is invalid")
        if self.index_mask.shape != expected_indices:
            raise FeatureEngineeringError("index mask shape is invalid")
        if self.raw_band_mask.shape != expected_raw_bands:
            raise FeatureEngineeringError("raw per-band mask shape is invalid")
        if len(self.raw_band_names) != len(set(self.raw_band_names)):
            raise FeatureEngineeringError("raw sequence band names must be unique")
        for name, array in (
            ("relative_positions", self.relative_positions),
            ("calendar_months", self.calendar_months),
            ("radar_mask", self.radar_mask),
            ("optical_mask", self.optical_mask),
            ("padding_mask", self.padding_mask),
        ):
            if array.shape != positions:
                raise FeatureEngineeringError(f"{name} sequence shape is invalid")
        if self.absolute_month_encoding.shape != (*positions, 2):
            raise FeatureEngineeringError("absolute month encoding must contain sine and cosine")
        for name, array in (
            ("radar_mask", self.radar_mask),
            ("radar_feature_mask", self.radar_feature_mask),
            ("optical_mask", self.optical_mask),
            ("optical_band_mask", self.optical_band_mask),
            ("raw_band_mask", self.raw_band_mask),
            ("index_mask", self.index_mask),
            ("padding_mask", self.padding_mask),
        ):
            if array.dtype != np.bool_:
                raise FeatureEngineeringError(f"{name} must use a boolean dtype")
        for values, mask, name in (
            (self.radar_values, self.radar_feature_mask, "radar"),
            (self.optical_values, self.optical_band_mask, "optical"),
            (self.monthly_indices, self.index_mask, "index"),
        ):
            if not np.isnan(values[~mask]).all():
                raise FeatureEngineeringError(f"invalid {name} sequence values must be NaN")
            if not np.isfinite(values[mask]).all():
                raise FeatureEngineeringError(f"valid {name} sequence values must be finite")
        if not np.isnan(self.absolute_month_encoding[self.padding_mask]).all():
            raise FeatureEngineeringError("padded cyclic month encodings must be NaN")
        for name, mask in (
            ("radar", self.radar_mask),
            ("optical", self.optical_mask),
            ("radar feature", self.radar_feature_mask.any(axis=2)),
            ("optical band", self.optical_band_mask.any(axis=2)),
            ("raw band", self.raw_band_mask.any(axis=2)),
            ("index", self.index_mask.any(axis=2)),
        ):
            if np.any(mask & self.padding_mask):
                raise FeatureEngineeringError(f"{name} validity cannot occupy padding")
        if np.isinf(self.absolute_month_encoding).any():
            raise FeatureEngineeringError("month encodings must not contain infinity")
        if len(self.window_ids) != rows or len(self.folds) != rows:
            raise FeatureEngineeringError("sequence metadata must align with rows")
        if self.labels is not None and len(self.labels) != rows:
            raise FeatureEngineeringError("sequence labels must align with rows")
        if len(set(self.window_ids.tolist())) != rows:
            raise FeatureEngineeringError("sequence window IDs must be unique")
        _validate_identity_folds(self.original_ids, self.folds)


def _aggregate_validity_rule(statistic: str, monthly_rule: str) -> str:
    if statistic in {"std", "amplitude", "iqr", "first_to_last", "slope"}:
        observations = "at least two valid observations"
    elif statistic == "valid_count":
        observations = "always defined as the number of valid observations"
    else:
        observations = "at least one valid observation"
    return f"{monthly_rule}; {observations}; padding excluded"


def _longest_run(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(mask.shape[0], dtype=np.float64)
    for row_index, row in enumerate(mask):
        longest = 0
        current = 0
        for value in row:
            current = current + 1 if value else 0
            longest = max(longest, current)
        result[row_index] = longest
    return result


def _metadata_features(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
) -> tuple[list[str], list[np.ndarray], list[FeatureDefinition]]:
    manifest = windows.manifest
    position = windows.position_mask
    radar = windows.radar_mask
    optical_bands = windows.optical_mask
    optical_month = optical_bands.all(axis=2) & position
    length = position.sum(axis=1).astype(np.float64)
    start = manifest["window_start"].to_numpy(dtype=np.float64)
    end = manifest["window_end"].to_numpy(dtype=np.float64)
    radar_count = radar.sum(axis=1).astype(np.float64)
    optical_count = optical_month.sum(axis=1).astype(np.float64)
    total_valid = radar_count * (
        len(windows.band_names) - len(windows.optical_bands)
    ) + optical_bands.sum(axis=(1, 2))
    gap = (radar & position & ~optical_month).sum(axis=1).astype(np.float64)
    angle_start = 2.0 * np.pi * (start - 1.0) / 12.0
    angle_end = 2.0 * np.pi * (end - 1.0) / 12.0

    names: list[str] = []
    values: list[np.ndarray] = []
    definitions: list[FeatureDefinition] = []

    def add(
        name: str,
        group: str,
        source_bands: tuple[str, ...],
        formula: str,
        validity_rule: str,
        feature_values: np.ndarray,
    ) -> None:
        normalized = np.asarray(feature_values, dtype=np.float64)
        if not np.isfinite(normalized).all():
            raise FeatureEngineeringError(f"metadata feature '{name}' must always be finite")
        names.append(name)
        values.append(normalized)
        definitions.append(
            FeatureDefinition(
                name=name,
                feature_group=group,
                source_bands=source_bands,
                formula=formula,
                temporal_aggregation=None,
                validity_rule=validity_rule,
                expected_dtype="float64",
                feature_kind="metadata",
                output_representation="tabular",
                version=config.version,
            )
        )

    all_bands = tuple(windows.band_names)
    add(
        "metadata__window_length",
        "metadata_window",
        (),
        "number of calendar slots in window",
        "always",
        length,
    )
    add("metadata__start_month", "metadata_window", (), "first calendar month", "always", start)
    add("metadata__end_month", "metadata_window", (), "last calendar month", "always", end)
    add(
        "metadata__start_month_sin",
        "metadata_window",
        (),
        "sin(2*pi*(start_month-1)/12)",
        "always",
        np.sin(angle_start),
    )
    add(
        "metadata__start_month_cos",
        "metadata_window",
        (),
        "cos(2*pi*(start_month-1)/12)",
        "always",
        np.cos(angle_start),
    )
    add(
        "metadata__end_month_sin",
        "metadata_window",
        (),
        "sin(2*pi*(end_month-1)/12)",
        "always",
        np.sin(angle_end),
    )
    add(
        "metadata__end_month_cos",
        "metadata_window",
        (),
        "cos(2*pi*(end_month-1)/12)",
        "always",
        np.cos(angle_end),
    )
    add(
        "metadata__relative_position_count",
        "metadata_window",
        (),
        "count of non-padding relative positions",
        "always",
        length,
    )
    add(
        "metadata__radar_valid_count",
        "metadata_missingness",
        tuple(band for band in windows.band_names if band not in windows.optical_bands),
        "count of radar-valid positions",
        "always",
        radar_count,
    )
    add(
        "metadata__optical_valid_count",
        "metadata_missingness",
        tuple(windows.optical_bands),
        "count of positions where all optical bands are valid",
        "always",
        optical_count,
    )
    add(
        "metadata__total_valid_band_count",
        "metadata_missingness",
        all_bands,
        "sum of valid raw band-position observations",
        "always",
        total_valid,
    )
    add(
        "metadata__optical_gap_count",
        "metadata_missingness",
        tuple(windows.optical_bands),
        "radar-valid positions lacking at least one optical band",
        "always",
        gap,
    )
    add(
        "metadata__longest_optical_valid_run",
        "metadata_missingness",
        tuple(windows.optical_bands),
        "longest consecutive run of fully optical-valid positions",
        "always",
        _longest_run(position & optical_month),
    )
    add(
        "metadata__longest_optical_missing_run",
        "metadata_missingness",
        tuple(windows.optical_bands),
        "longest consecutive optical-missing run inside base window",
        "always",
        _longest_run(position & radar & ~optical_month),
    )
    add(
        "metadata__radar_valid_proportion",
        "metadata_missingness",
        tuple(band for band in windows.band_names if band not in windows.optical_bands),
        "radar_valid_count / window_length",
        "window length positive",
        radar_count / length,
    )
    add(
        "metadata__optical_valid_proportion",
        "metadata_missingness",
        tuple(windows.optical_bands),
        "valid optical band-position observations / possible optical observations",
        "window length positive",
        optical_bands.sum(axis=(1, 2)) / (length * len(windows.optical_bands)),
    )

    role_by_raw = {raw: role for role, raw in config.bands.roles.items()}
    radar_bands = tuple(band for band in windows.band_names if band not in windows.optical_bands)
    for band in radar_bands:
        add(
            f"metadata__{role_by_raw[band]}_valid_count",
            "metadata_band_validity",
            (band,),
            f"count of valid {band} observations",
            "always",
            radar_count,
        )
    for optical_index, band in enumerate(windows.optical_bands):
        add(
            f"metadata__{role_by_raw[band]}_valid_count",
            "metadata_band_validity",
            (band,),
            f"count of valid {band} observations",
            "always",
            optical_bands[:, :, optical_index].sum(axis=1),
        )
    for relative_position in range(6):
        position_number = relative_position + 1
        add(
            f"metadata__position_{position_number:02d}__radar_valid",
            "metadata_position",
            radar_bands,
            f"radar validity at relative position {position_number}",
            "always",
            radar[:, relative_position],
        )
        add(
            f"metadata__position_{position_number:02d}__optical_valid",
            "metadata_position",
            tuple(windows.optical_bands),
            f"all-band optical validity at relative position {position_number}",
            "always",
            optical_month[:, relative_position],
        )
        add(
            f"metadata__position_{position_number:02d}__padding",
            "metadata_position",
            (),
            f"padding indicator at relative position {position_number}",
            "always",
            ~position[:, relative_position],
        )
    return names, values, definitions


def _build_tabular_from_monthly(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
    monthly: MonthlyFeatureCollection,
) -> FeatureMatrix:
    feature_names: list[str] = []
    columns: list[np.ndarray] = []
    definitions: list[FeatureDefinition] = []
    for channel_index, spec in enumerate(monthly.specs):
        for relative_position in range(6):
            position_number = relative_position + 1
            name = f"relative__position_{position_number:02d}__{spec.name}"
            feature_names.append(name)
            columns.append(monthly.values[:, relative_position, channel_index])
            definitions.append(
                FeatureDefinition(
                    name=name,
                    feature_group=f"relative_{spec.feature_group}",
                    source_bands=spec.source_bands,
                    formula=f"{spec.formula} at relative position {position_number}",
                    temporal_aggregation=None,
                    validity_rule=(
                        f"{spec.validity_rule}; relative position exists and is not padding"
                    ),
                    expected_dtype="float64",
                    feature_kind="monthly",
                    output_representation="tabular",
                    version=config.version,
                )
            )
    for channel_index, spec in enumerate(monthly.specs):
        aggregates = aggregate_temporal_series(
            monthly.values[:, :, channel_index],
            windows.relative_positions,
            monthly.masks[:, :, channel_index],
        )
        for statistic_index, statistic in enumerate(AGGREGATION_NAMES):
            name = f"{spec.name}__{statistic}"
            feature_names.append(name)
            columns.append(aggregates.values[:, statistic_index])
            definitions.append(
                FeatureDefinition(
                    name=name,
                    feature_group=spec.feature_group,
                    source_bands=spec.source_bands,
                    formula=spec.formula,
                    temporal_aggregation=statistic,
                    validity_rule=_aggregate_validity_rule(statistic, spec.validity_rule),
                    expected_dtype="float64",
                    feature_kind="aggregate",
                    output_representation="tabular",
                    version=config.version,
                )
            )

    for role in ("vv", "vh"):
        difference_name = f"radar__{role}_first_difference"
        channel_index = next(
            index for index, spec in enumerate(monthly.specs) if spec.name == difference_name
        )
        absolute_difference = np.abs(monthly.values[:, :, channel_index])
        aggregate = aggregate_temporal_series(
            absolute_difference,
            windows.relative_positions,
            monthly.masks[:, :, channel_index],
        )
        name = f"radar__{role}__mean_abs_first_difference"
        source_band = config.bands.roles[role]
        feature_names.append(name)
        columns.append(aggregate.values[:, AGGREGATION_NAMES.index("mean")])
        definitions.append(
            FeatureDefinition(
                name=name,
                feature_group="radar_stability",
                source_bands=(source_band,),
                formula=f"mean(abs({source_band}(t) - {source_band}(t-1)))",
                temporal_aggregation="mean_abs_adjacent_difference",
                validity_rule="at least one adjacent valid radar pair; padding and gaps excluded",
                expected_dtype="float64",
                feature_kind="aggregate",
                output_representation="tabular",
                version=config.version,
            )
        )

    metadata_names, metadata_values, metadata_definitions = _metadata_features(windows, config)
    feature_names.extend(metadata_names)
    columns.extend(metadata_values)
    definitions.extend(metadata_definitions)
    frame = pd.DataFrame(
        np.column_stack(columns),
        columns=feature_names,
        dtype=np.float64,
    )
    registry = FeatureRegistry(definitions=tuple(definitions))
    original_ids, window_ids, folds, labels = _attached_metadata(windows)
    schema_fingerprint = _schema_fingerprint(tuple(feature_names), registry)
    digest = hashlib.sha256()
    digest.update(schema_fingerprint.encode())
    _hash_array(digest, frame.to_numpy(dtype=np.float64))
    _hash_strings(digest, original_ids)
    _hash_strings(digest, window_ids)
    _hash_array(digest, folds)
    if labels is not None:
        _hash_array(digest, labels)
    return FeatureMatrix(
        features=frame,
        feature_names=tuple(feature_names),
        original_ids=original_ids,
        window_ids=window_ids,
        folds=folds,
        labels=labels,
        registry=registry,
        feature_groups=_feature_groups(registry),
        fingerprint=digest.hexdigest(),
        schema_fingerprint=schema_fingerprint,
    )


def _sequence_definition(
    name: str,
    group: str,
    source_bands: tuple[str, ...],
    formula: str,
    validity_rule: str,
    version: str,
    expected_dtype: str = "float64",
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_group=group,
        source_bands=source_bands,
        formula=formula,
        temporal_aggregation=None,
        validity_rule=validity_rule,
        expected_dtype=expected_dtype,
        feature_kind="monthly",
        output_representation="sequence",
        version=version,
    )


def _build_sequence_from_monthly(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
    monthly: MonthlyFeatureCollection,
) -> SequenceFeatureDataset:
    radar_specs = monthly.specs[monthly.radar_slice]
    optical_specs = monthly.specs[monthly.optical_raw_slice]
    index_specs = monthly.specs[monthly.optical_index_slice]
    radar_values = monthly.values[:, :, monthly.radar_slice].copy()
    optical_values = monthly.values[:, :, monthly.optical_raw_slice].copy()
    index_values = monthly.values[:, :, monthly.optical_index_slice].copy()
    radar_feature_mask = monthly.masks[:, :, monthly.radar_slice].copy()
    optical_band_mask = monthly.masks[:, :, monthly.optical_raw_slice].copy()
    index_mask = monthly.masks[:, :, monthly.optical_index_slice].copy()
    raw_band_mask = np.zeros(
        (*windows.position_mask.shape, len(windows.band_names)),
        dtype=bool,
    )
    optical_mask_index = {band: index for index, band in enumerate(windows.optical_bands)}
    for band_index, band in enumerate(windows.band_names):
        if band in optical_mask_index:
            raw_band_mask[:, :, band_index] = windows.optical_mask[:, :, optical_mask_index[band]]
        else:
            raw_band_mask[:, :, band_index] = windows.radar_mask
    padding_mask = ~windows.position_mask.copy()
    optical_mask = optical_band_mask.all(axis=2) & windows.position_mask
    angles = 2.0 * np.pi * (windows.calendar_months.astype(np.float64) - 1.0) / 12.0
    month_encoding = np.stack((np.sin(angles), np.cos(angles)), axis=2)
    month_encoding[padding_mask] = np.nan

    radar_names = tuple(spec.name for spec in radar_specs)
    optical_names = tuple(spec.name for spec in optical_specs)
    index_names = tuple(spec.name for spec in index_specs)
    definitions = [
        _sequence_definition(
            f"sequence__{spec.name}",
            spec.feature_group,
            spec.source_bands,
            spec.formula,
            spec.validity_rule,
            config.version,
        )
        for spec in (*radar_specs, *optical_specs, *index_specs)
    ]
    definitions.extend(
        [
            _sequence_definition(
                "sequence__relative_position",
                "sequence_position",
                (),
                "relative position 1-6; zero only for padding",
                "base window position available",
                config.version,
                "int8",
            ),
            _sequence_definition(
                "sequence__calendar_month",
                "sequence_position",
                (),
                "absolute calendar month 1-12; zero only for padding",
                "base window position available",
                config.version,
                "int8",
            ),
            _sequence_definition(
                "sequence__calendar_month_sin",
                "sequence_position",
                (),
                "sin(2*pi*(calendar_month-1)/12)",
                "base window position available; NaN for padding",
                config.version,
            ),
            _sequence_definition(
                "sequence__calendar_month_cos",
                "sequence_position",
                (),
                "cos(2*pi*(calendar_month-1)/12)",
                "base window position available; NaN for padding",
                config.version,
            ),
            _sequence_definition(
                "sequence__radar_valid",
                "sequence_mask",
                tuple(band for band in windows.band_names if band not in windows.optical_bands),
                "Phase 2 radar availability",
                "boolean and false for padding",
                config.version,
                "bool",
            ),
            _sequence_definition(
                "sequence__optical_valid",
                "sequence_mask",
                tuple(windows.optical_bands),
                "all-band optical availability",
                "boolean and false for padding or any missing optical band",
                config.version,
                "bool",
            ),
            _sequence_definition(
                "sequence__padding",
                "sequence_mask",
                (),
                "logical inverse of Phase 2 position mask",
                "boolean",
                config.version,
                "bool",
            ),
        ]
    )
    role_by_raw = {raw: role for role, raw in config.bands.roles.items()}
    for band in windows.band_names:
        definitions.append(
            _sequence_definition(
                f"sequence__raw_band__{role_by_raw[band]}__valid",
                "sequence_band_mask",
                (band,),
                f"per-band validity for {band}",
                "boolean and false for padding",
                config.version,
                "bool",
            )
        )
    registry = FeatureRegistry(definitions=tuple(definitions))
    sequence_schema_names = tuple(definition.name for definition in registry.definitions)
    schema_fingerprint = _schema_fingerprint(sequence_schema_names, registry)
    original_ids, window_ids, folds, labels = _attached_metadata(windows)
    digest = hashlib.sha256()
    digest.update(schema_fingerprint.encode())
    for array in (
        radar_values,
        optical_values,
        index_values,
        windows.relative_positions,
        windows.calendar_months,
        month_encoding,
        windows.radar_mask,
        radar_feature_mask,
        optical_mask,
        optical_band_mask,
        raw_band_mask,
        index_mask,
        padding_mask,
    ):
        _hash_array(digest, array)
    _hash_strings(digest, original_ids)
    _hash_strings(digest, window_ids)
    _hash_array(digest, folds)
    if labels is not None:
        _hash_array(digest, labels)
    return SequenceFeatureDataset(
        radar_values=radar_values,
        optical_values=optical_values,
        monthly_indices=index_values,
        relative_positions=windows.relative_positions.copy(),
        calendar_months=windows.calendar_months.copy(),
        absolute_month_encoding=month_encoding,
        radar_mask=windows.radar_mask.copy(),
        radar_feature_mask=radar_feature_mask,
        optical_mask=optical_mask,
        optical_band_mask=optical_band_mask,
        raw_band_mask=raw_band_mask,
        index_mask=index_mask,
        padding_mask=padding_mask,
        radar_feature_names=radar_names,
        optical_feature_names=optical_names,
        index_feature_names=index_names,
        raw_band_names=windows.band_names,
        original_ids=original_ids,
        window_ids=window_ids,
        folds=folds,
        labels=labels,
        registry=registry,
        fingerprint=digest.hexdigest(),
        schema_fingerprint=schema_fingerprint,
    )


def build_feature_representations(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
) -> tuple[FeatureMatrix, SequenceFeatureDataset]:
    """Build aligned tabular and sequence representations without fitting statistics."""

    monthly = build_monthly_features(windows, config)
    return (
        _build_tabular_from_monthly(windows, config, monthly),
        _build_sequence_from_monthly(windows, config, monthly),
    )


def build_tabular_features(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
) -> FeatureMatrix:
    """Build the deterministic tabular representation only."""

    monthly = build_monthly_features(windows, config)
    return _build_tabular_from_monthly(windows, config, monthly)


def build_sequence_features(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
) -> SequenceFeatureDataset:
    """Build the deterministic masked sequence representation only."""

    monthly = build_monthly_features(windows, config)
    return _build_sequence_from_monthly(windows, config, monthly)
