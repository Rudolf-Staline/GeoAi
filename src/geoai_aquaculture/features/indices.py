"""Vectorized monthly optical and radar features with explicit validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoai_aquaculture.data import (
    BandSemanticMapping,
    FeatureConfig,
    TemporalWindowDataset,
)


class FeatureEngineeringError(ValueError):
    """Raised when a feature cannot satisfy its schema or numerical contract."""


@dataclass(frozen=True, slots=True)
class SafeDivisionResult:
    """Finite division values and the exact elements where they are valid."""

    values: np.ndarray
    validity: np.ndarray


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    epsilon: float,
) -> SafeDivisionResult:
    """Divide finite arrays where ``abs(denominator) > epsilon``; otherwise emit NaN."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise FeatureEngineeringError("safe-division epsilon must be finite and positive")
    numerator_array, denominator_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    validity = (
        np.isfinite(numerator_array)
        & np.isfinite(denominator_array)
        & (np.abs(denominator_array) > epsilon)
    )
    values = np.full(numerator_array.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(numerator_array, denominator_array, out=values, where=validity)
    validity &= np.isfinite(values)
    values[~validity] = np.nan
    return SafeDivisionResult(values=values, validity=validity)


@dataclass(frozen=True, slots=True)
class MonthlyChannelSpec:
    """Formula and source provenance for one monthly series."""

    name: str
    feature_group: str
    source_bands: tuple[str, ...]
    formula: str
    validity_rule: str


@dataclass(frozen=True, slots=True)
class MonthlyFeatureCollection:
    """Ordered raw and engineered monthly series used by both representations."""

    specs: tuple[MonthlyChannelSpec, ...]
    values: np.ndarray
    masks: np.ndarray
    radar_channel_count: int
    optical_raw_channel_count: int
    optical_index_channel_count: int

    def __post_init__(self) -> None:
        if self.values.shape != self.masks.shape:
            raise FeatureEngineeringError("monthly values and masks must have identical shapes")
        if self.values.ndim != 3 or self.values.shape[2] != len(self.specs):
            raise FeatureEngineeringError("monthly feature shape must align with channel specs")
        expected = (
            self.radar_channel_count
            + self.optical_raw_channel_count
            + self.optical_index_channel_count
        )
        if expected != len(self.specs):
            raise FeatureEngineeringError("monthly feature group counts must cover all channels")
        if self.masks.dtype != np.bool_:
            raise FeatureEngineeringError("monthly feature masks must be boolean")
        if not np.isnan(self.values[~self.masks]).all():
            raise FeatureEngineeringError("invalid monthly values must remain NaN")
        if not np.isfinite(self.values[self.masks]).all():
            raise FeatureEngineeringError("valid monthly values must be finite")

    @property
    def radar_slice(self) -> slice:
        return slice(0, self.radar_channel_count)

    @property
    def optical_raw_slice(self) -> slice:
        start = self.radar_channel_count
        return slice(start, start + self.optical_raw_channel_count)

    @property
    def optical_index_slice(self) -> slice:
        start = self.radar_channel_count + self.optical_raw_channel_count
        return slice(start, len(self.specs))


def _validate_semantics(windows: TemporalWindowDataset, semantics: BandSemanticMapping) -> None:
    roles = semantics.roles
    raw_bands = tuple(roles.values())
    if len(raw_bands) != len(set(raw_bands)):
        raise FeatureEngineeringError("semantic band mapping is ambiguous")
    missing = sorted(set(raw_bands) - set(windows.band_names))
    if missing:
        raise FeatureEngineeringError(f"semantic mapping requires unavailable bands: {missing}")
    if {semantics.vv, semantics.vh} & set(windows.optical_bands):
        raise FeatureEngineeringError("radar semantic roles cannot map to optical bands")
    optical_roles = set(raw_bands) - {semantics.vv, semantics.vh}
    if optical_roles != set(windows.optical_bands):
        raise FeatureEngineeringError(
            "optical semantic roles must map every observed optical band exactly once"
        )


def _adjacent_difference(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    validity = np.zeros(values.shape, dtype=bool)
    validity[:, 1:] = mask[:, 1:] & mask[:, :-1]
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 1:] = values[:, 1:] - values[:, :-1]
    validity &= np.isfinite(result)
    result[~validity] = np.nan
    return result, validity


def _simple_binary(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    *,
    operation: str,
) -> tuple[np.ndarray, np.ndarray]:
    validity = left_mask & right_mask & np.isfinite(left) & np.isfinite(right)
    with np.errstate(over="ignore", invalid="ignore"):
        if operation == "subtract":
            values = left - right
        elif operation == "add":
            values = left + right
        else:
            raise FeatureEngineeringError(f"unsupported binary operation: {operation}")
    validity &= np.isfinite(values)
    values = np.where(validity, values, np.nan)
    return values, validity


def _normalized_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> SafeDivisionResult:
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = left - right
        denominator = left + right
    return safe_divide(numerator, denominator, epsilon=epsilon)


def build_monthly_features(
    windows: TemporalWindowDataset,
    config: FeatureConfig,
) -> MonthlyFeatureCollection:
    """Build the fixed small set of monthly raw, radar, and optical channels."""

    semantics = config.bands
    _validate_semantics(windows, semantics)
    raw_index = {band: index for index, band in enumerate(windows.band_names)}
    optical_mask_index = {band: index for index, band in enumerate(windows.optical_bands)}

    specs: list[MonthlyChannelSpec] = []
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    def add(
        name: str,
        group: str,
        source_bands: tuple[str, ...],
        formula: str,
        validity_rule: str,
        channel_values: np.ndarray,
        channel_mask: np.ndarray,
    ) -> None:
        finite_mask = channel_mask & np.isfinite(channel_values)
        normalized = np.where(finite_mask, channel_values, np.nan).astype(np.float64)
        specs.append(
            MonthlyChannelSpec(
                name=name,
                feature_group=group,
                source_bands=source_bands,
                formula=formula,
                validity_rule=validity_rule,
            )
        )
        values.append(normalized)
        masks.append(finite_mask)

    vv = windows.values[:, :, raw_index[semantics.vv]]
    vh = windows.values[:, :, raw_index[semantics.vh]]
    vv_mask = windows.radar_mask & np.isfinite(vv)
    vh_mask = windows.radar_mask & np.isfinite(vh)
    add(
        "radar__vv",
        "radar_raw",
        (semantics.vv,),
        f"raw band {semantics.vv}",
        "radar month available and raw value finite",
        vv,
        vv_mask,
    )
    add(
        "radar__vh",
        "radar_raw",
        (semantics.vh,),
        f"raw band {semantics.vh}",
        "radar month available and raw value finite",
        vh,
        vh_mask,
    )
    difference, difference_mask = _simple_binary(vv, vh, vv_mask, vh_mask, operation="subtract")
    add(
        "radar__vv_minus_vh",
        "radar_derived",
        (semantics.vv, semantics.vh),
        "VV - VH on the supplied numeric scale",
        "both radar bands available and result finite",
        difference,
        difference_mask,
    )
    radar_sum, sum_mask = _simple_binary(vv, vh, vv_mask, vh_mask, operation="add")
    add(
        "radar__vv_plus_vh",
        "radar_derived",
        (semantics.vv, semantics.vh),
        "VV + VH on the supplied numeric scale",
        "both radar bands available and result finite",
        radar_sum,
        sum_mask,
    )
    vv_ratio = safe_divide(vv, np.abs(vh), epsilon=config.epsilon)
    add(
        "radar__vv_over_abs_vh",
        "radar_derived",
        (semantics.vv, semantics.vh),
        "VV / abs(VH)",
        f"both radar bands finite and abs(VH) > {config.epsilon:g}",
        vv_ratio.values,
        vv_mask & vh_mask & vv_ratio.validity,
    )
    vh_ratio = safe_divide(vh, np.abs(vv), epsilon=config.epsilon)
    add(
        "radar__vh_over_abs_vv",
        "radar_derived",
        (semantics.vh, semantics.vv),
        "VH / abs(VV)",
        f"both radar bands finite and abs(VV) > {config.epsilon:g}",
        vh_ratio.values,
        vv_mask & vh_mask & vh_ratio.validity,
    )
    vv_difference, vv_difference_mask = _adjacent_difference(vv, vv_mask)
    add(
        "radar__vv_first_difference",
        "radar_temporal",
        (semantics.vv,),
        "VV(t) - VV(t-1) over adjacent relative positions",
        "current and immediately previous relative positions both valid",
        vv_difference,
        vv_difference_mask,
    )
    vh_difference, vh_difference_mask = _adjacent_difference(vh, vh_mask)
    add(
        "radar__vh_first_difference",
        "radar_temporal",
        (semantics.vh,),
        "VH(t) - VH(t-1) over adjacent relative positions",
        "current and immediately previous relative positions both valid",
        vh_difference,
        vh_difference_mask,
    )
    radar_channel_count = len(specs)

    role_by_raw = {raw_band: role for role, raw_band in semantics.roles.items()}
    for band in windows.optical_bands:
        role = role_by_raw[band]
        channel = windows.values[:, :, raw_index[band]]
        mask = windows.optical_mask[:, :, optical_mask_index[band]] & np.isfinite(channel)
        add(
            f"optical__{role}",
            "optical_raw",
            (band,),
            f"raw band {band}",
            "that optical band is available and finite",
            channel,
            mask,
        )
    optical_raw_channel_count = len(specs) - radar_channel_count

    raw = {
        role: windows.values[:, :, raw_index[band]]
        for role, band in semantics.roles.items()
        if role not in {"vv", "vh"}
    }

    def add_ratio_index(
        name: str,
        left_role: str,
        right_role: str,
        formula: str,
        *,
        normalized: bool,
        subtract_one: bool = False,
    ) -> None:
        if normalized:
            result = _normalized_difference(raw[left_role], raw[right_role], epsilon=config.epsilon)
        else:
            result = safe_divide(raw[left_role], raw[right_role], epsilon=config.epsilon)
        channel_values = result.values.copy()
        if subtract_one:
            channel_values[result.validity] -= 1.0
        source_bands = (
            semantics.roles[left_role],
            semantics.roles[right_role],
        )
        add(
            f"optical__{name}",
            "optical_index",
            source_bands,
            formula,
            f"all required bands finite and denominator magnitude exceeds {config.epsilon:g}",
            channel_values,
            result.validity,
        )

    add_ratio_index("ndvi", "nir", "red", "(NIR - Red) / (NIR + Red)", normalized=True)
    add_ratio_index("ndwi", "green", "nir", "(Green - NIR) / (Green + NIR)", normalized=True)
    add_ratio_index(
        "mndwi",
        "green",
        "swir1",
        "(Green - SWIR1) / (Green + SWIR1)",
        normalized=True,
    )
    add_ratio_index("ndmi", "nir", "swir1", "(NIR - SWIR1) / (NIR + SWIR1)", normalized=True)
    add_ratio_index("nbr", "nir", "swir2", "(NIR - SWIR2) / (NIR + SWIR2)", normalized=True)
    add_ratio_index(
        "ndre1",
        "narrow_nir",
        "red_edge_1",
        "(NarrowNIR - RE1) / (NarrowNIR + RE1)",
        normalized=True,
    )
    add_ratio_index(
        "ndre2",
        "narrow_nir",
        "red_edge_2",
        "(NarrowNIR - RE2) / (NarrowNIR + RE2)",
        normalized=True,
    )
    add_ratio_index(
        "chlorophyll_red_edge",
        "narrow_nir",
        "red_edge_1",
        "NarrowNIR / RE1 - 1",
        normalized=False,
        subtract_one=True,
    )
    add_ratio_index("nir_over_swir1", "nir", "swir1", "NIR / SWIR1", normalized=False)
    add_ratio_index("nir_over_swir2", "nir", "swir2", "NIR / SWIR2", normalized=False)
    add_ratio_index("green_over_swir1", "green", "swir1", "Green / SWIR1", normalized=False)
    add_ratio_index("green_over_swir2", "green", "swir2", "Green / SWIR2", normalized=False)
    add_ratio_index(
        "green_red_contrast",
        "green",
        "red",
        "(Green - Red) / (Green + Red)",
        normalized=True,
    )
    add_ratio_index(
        "blue_green_contrast",
        "blue",
        "green",
        "(Blue - Green) / (Blue + Green)",
        normalized=True,
    )
    optical_index_channel_count = len(specs) - radar_channel_count - optical_raw_channel_count

    return MonthlyFeatureCollection(
        specs=tuple(specs),
        values=np.stack(values, axis=2),
        masks=np.stack(masks, axis=2),
        radar_channel_count=radar_channel_count,
        optical_raw_channel_count=optical_raw_channel_count,
        optical_index_channel_count=optical_index_channel_count,
    )
