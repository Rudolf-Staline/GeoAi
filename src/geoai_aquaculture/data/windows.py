"""Leakage-safe temporal views over original training rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import WindowGenerationConfig
from .folds import (
    FoldAssignmentError,
    assert_no_fold_leakage,
    validate_original_fold_manifest,
)
from .loading import CompetitionData
from .masks import (
    MaskLibrary,
    MissingnessMaskTemplate,
    extract_test_mask_templates,
)

MAX_WINDOW_LENGTH = 6


class TemporalWindowError(ValueError):
    """Raised when temporal views cannot satisfy the scientific contract."""


@dataclass(frozen=True, slots=True, order=True)
class ConsecutiveWindow:
    """One unmasked consecutive calendar window."""

    window_length: int
    window_start: int
    window_end: int

    def __post_init__(self) -> None:
        if self.window_length not in {4, 5, 6}:
            raise TemporalWindowError("window length must be one of 4, 5, or 6")
        if self.window_start < 1 or self.window_end > 12:
            raise TemporalWindowError("window calendar months must remain within 1-12")
        if self.window_end != self.window_start + self.window_length - 1:
            raise TemporalWindowError("window start, end, and length must be consecutive")


def enumerate_consecutive_windows(
    window_lengths: tuple[int, ...] = (4, 5, 6),
    *,
    months: int = 12,
) -> tuple[ConsecutiveWindow, ...]:
    """Return the canonical 24 consecutive windows in length/start order."""

    if months != 12:
        raise TemporalWindowError("competition windows require exactly 12 calendar months")
    if window_lengths != (4, 5, 6):
        raise TemporalWindowError("window lengths must be exactly (4, 5, 6)")
    return tuple(
        ConsecutiveWindow(
            window_length=length,
            window_start=start,
            window_end=start + length - 1,
        )
        for length in window_lengths
        for start in range(1, months - length + 2)
    )


@dataclass(frozen=True, slots=True)
class TemporalWindowDataset:
    """Raw relative-window values, masks, and leakage-safe manifest."""

    manifest: pd.DataFrame
    values: np.ndarray
    calendar_months: np.ndarray
    relative_positions: np.ndarray
    position_mask: np.ndarray
    radar_mask: np.ndarray
    optical_mask: np.ndarray
    band_names: tuple[str, ...]
    optical_bands: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = self.manifest.shape[0]
        expected_values = (rows, MAX_WINDOW_LENGTH, len(self.band_names))
        expected_positions = (rows, MAX_WINDOW_LENGTH)
        expected_optical = (rows, MAX_WINDOW_LENGTH, len(self.optical_bands))
        if self.values.shape != expected_values:
            raise TemporalWindowError(
                f"window values shape must be {expected_values}; found {self.values.shape}"
            )
        for name, array in (
            ("calendar_months", self.calendar_months),
            ("relative_positions", self.relative_positions),
            ("position_mask", self.position_mask),
            ("radar_mask", self.radar_mask),
        ):
            if array.shape != expected_positions:
                raise TemporalWindowError(
                    f"{name} shape must be {expected_positions}; found {array.shape}"
                )
        if self.optical_mask.shape != expected_optical:
            raise TemporalWindowError(
                f"optical_mask shape must be {expected_optical}; found {self.optical_mask.shape}"
            )
        if len(self.band_names) != len(set(self.band_names)):
            raise TemporalWindowError("band names must be unique")
        if any(band not in self.band_names for band in self.optical_bands):
            raise TemporalWindowError("optical bands must be present in the raw band order")
        for name, array in (
            ("position_mask", self.position_mask),
            ("radar_mask", self.radar_mask),
            ("optical_mask", self.optical_mask),
        ):
            if array.dtype != np.bool_:
                raise TemporalWindowError(f"{name} must use a boolean dtype")
        if np.any(self.radar_mask & ~self.position_mask):
            raise TemporalWindowError("radar availability cannot occupy padded positions")
        if np.any(self.optical_mask & ~self.position_mask[:, :, None]):
            raise TemporalWindowError("optical availability cannot occupy padded positions")
        if np.any(self.calendar_months[~self.position_mask] != 0):
            raise TemporalWindowError("padded calendar months must use zero")
        if np.any(self.relative_positions[~self.position_mask] != 0):
            raise TemporalWindowError("padded relative positions must use zero")
        if not np.isnan(self.values[~self.position_mask]).all():
            raise TemporalWindowError("padded raw values must remain missing")
        optical_set = set(self.optical_bands)
        for band_index, band in enumerate(self.band_names):
            band_values = self.values[:, :, band_index]
            if band in optical_set:
                mask = self.optical_mask[:, :, self.optical_bands.index(band)]
            else:
                mask = self.radar_mask
            if not np.isnan(band_values[~mask]).all():
                raise TemporalWindowError(f"masked values for band '{band}' must remain missing")
            if not np.isfinite(band_values[mask]).all():
                raise TemporalWindowError(f"available values for band '{band}' must be finite")
        assert_no_fold_leakage(self.manifest)

    @property
    def n_windows(self) -> int:
        """Return the number of augmented temporal views."""

        return self.manifest.shape[0]


def _stable_window_id(
    *,
    original_id: str,
    fold: int,
    generation_mode: str,
    augmentation_seed: int,
    view_index: int,
    window_start: int,
    window_length: int,
    mask_id: str,
    radar_pattern: str,
    optical_pattern: str,
) -> str:
    payload = "|".join(
        (
            original_id,
            str(fold),
            generation_mode,
            str(augmentation_seed),
            str(view_index),
            str(window_start),
            str(window_length),
            mask_id,
            radar_pattern,
            optical_pattern,
        )
    ).encode()
    return f"window_{hashlib.sha256(payload).hexdigest()[:24]}"


def _bit_string(values: np.ndarray) -> str:
    return "".join(str(int(value)) for value in values)


def _optical_bit_string(values: np.ndarray) -> str:
    return ";".join(_bit_string(month) for month in values)


def _choose_views(
    generation: WindowGenerationConfig,
    candidates: tuple[ConsecutiveWindow, ...],
    mask_library: MaskLibrary | None,
    rng: np.random.Generator,
) -> tuple[tuple[ConsecutiveWindow, MissingnessMaskTemplate | None], ...]:
    if generation.exhaustive_windows:
        return tuple((candidate, None) for candidate in candidates)
    if generation.use_test_missingness_masks:
        if mask_library is None:
            raise TemporalWindowError(
                "sampled test-like generation requires an explicit mask library"
            )
        weights = np.asarray(
            [template.frequency for template in mask_library.templates], dtype=np.float64
        )
        weights /= weights.sum()
        indices = rng.choice(
            len(mask_library.templates),
            size=generation.windows_per_sample,
            replace=True,
            p=weights,
        )
        return tuple(
            (
                ConsecutiveWindow(
                    window_length=mask_library.templates[index].window_length,
                    window_start=mask_library.templates[index].window_start,
                    window_end=mask_library.templates[index].window_end,
                ),
                mask_library.templates[index],
            )
            for index in indices
        )
    indices = rng.choice(
        len(candidates),
        size=generation.windows_per_sample,
        replace=True,
    )
    return tuple((candidates[index], None) for index in indices)


def _validate_generation_inputs(
    data: CompetitionData,
    fold_manifest: pd.DataFrame,
    generation: WindowGenerationConfig,
    mask_library: MaskLibrary | None,
    seed: int,
) -> None:
    if not generation.enabled:
        raise TemporalWindowError("temporal augmentation must be explicitly enabled")
    if seed < 0:
        raise TemporalWindowError("augmentation seed must be non-negative")
    if generation.exhaustive_windows and generation.use_test_missingness_masks:
        raise TemporalWindowError(
            "test missingness masks are sampled views and cannot be combined with exhaustive mode"
        )
    if generation.use_test_missingness_masks and mask_library is None:
        raise TemporalWindowError("test-like generation requires a mask library")
    if not generation.use_test_missingness_masks and mask_library is not None:
        raise TemporalWindowError("mask library supplied while test-mask use is disabled")
    if mask_library is not None and mask_library.optical_bands != data.config.data.optical_bands:
        raise TemporalWindowError("mask-library optical bands do not match the data schema")
    if data.train.loc[:, list(data.feature_columns)].isna().any().any():
        raise TemporalWindowError("original training sequences must be complete before windowing")
    try:
        validate_original_fold_manifest(
            data.train,
            fold_manifest,
            id_column=data.config.data.id_column,
            target_column=data.config.data.target_column,
        )
    except FoldAssignmentError as exc:
        raise TemporalWindowError(
            "original-row fold assignment must be valid before augmentation"
        ) from exc


def _raw_value_cube(frame: pd.DataFrame, data: CompetitionData) -> np.ndarray:
    band_names = data.config.data.bands
    band_index = {band: index for index, band in enumerate(band_names)}
    lookup = {(item.band, item.month): item.name for item in data.temporal_columns}
    values = np.empty(
        (frame.shape[0], data.config.data.months, len(band_names)),
        dtype=np.float64,
    )
    for month in range(1, data.config.data.months + 1):
        for band in band_names:
            values[:, month - 1, band_index[band]] = frame[lookup[(band, month)]].to_numpy(
                dtype=np.float64
            )
    return values


def generate_temporal_windows(
    data: CompetitionData,
    fold_manifest: pd.DataFrame,
    generation: WindowGenerationConfig,
    *,
    seed: int,
    mask_library: MaskLibrary | None = None,
) -> TemporalWindowDataset:
    """Generate raw temporal views only after validating original-row folds."""

    _validate_generation_inputs(data, fold_manifest, generation, mask_library, seed)
    candidates = enumerate_consecutive_windows(
        data.config.validation.window_lengths,
        months=data.config.data.months,
    )
    rng = np.random.default_rng(seed)
    band_names = data.config.data.bands
    optical_bands = data.config.data.optical_bands
    band_index = {band: index for index, band in enumerate(band_names)}
    optical_index = {band: index for index, band in enumerate(optical_bands)}
    source_values = _raw_value_cube(data.train, data)

    manifests: list[dict[str, object]] = []
    value_views: list[np.ndarray] = []
    month_views: list[np.ndarray] = []
    relative_views: list[np.ndarray] = []
    position_views: list[np.ndarray] = []
    radar_views: list[np.ndarray] = []
    optical_views: list[np.ndarray] = []

    for row_index in range(data.train.shape[0]):
        fold_row = fold_manifest.iloc[row_index]
        original_id = str(fold_row["original_id"])
        fold = int(fold_row["fold"])
        views = _choose_views(generation, candidates, mask_library, rng)
        for view_index, (window, template) in enumerate(views):
            position_mask = np.zeros(MAX_WINDOW_LENGTH, dtype=bool)
            position_mask[: window.window_length] = True
            calendar_months = np.zeros(MAX_WINDOW_LENGTH, dtype=np.int8)
            calendar_months[: window.window_length] = np.arange(
                window.window_start,
                window.window_end + 1,
                dtype=np.int8,
            )
            relative_positions = np.zeros(MAX_WINDOW_LENGTH, dtype=np.int8)
            relative_positions[: window.window_length] = np.arange(
                1, window.window_length + 1, dtype=np.int8
            )
            radar_mask = position_mask.copy()
            optical_mask = np.zeros((MAX_WINDOW_LENGTH, len(optical_bands)), dtype=bool)
            optical_mask[: window.window_length, :] = True
            source_mask_frequency = 0
            source_mask_id = "full_window"
            test_mask_gap_count = 0
            if template is not None:
                source_mask_frequency = template.frequency
                source_mask_id = template.mask_id
                for position, month in enumerate(range(window.window_start, window.window_end + 1)):
                    radar_mask[position] = template.radar_availability[month - 1]
                    optical_mask[position, :] = template.optical_availability[month - 1]
                test_mask_gap_count = template.internal_optical_gap_count

            temporal_dropout = np.zeros(MAX_WINDOW_LENGTH, dtype=bool)
            if generation.temporal_dropout_enabled:
                temporal_dropout = position_mask & (
                    rng.random(MAX_WINDOW_LENGTH) < generation.temporal_dropout_probability
                )
                radar_mask[temporal_dropout] = False
                optical_mask[temporal_dropout, :] = False

            optical_dropout = np.zeros(MAX_WINDOW_LENGTH, dtype=bool)
            if generation.optical_dropout_enabled:
                optical_eligible = position_mask & ~temporal_dropout & optical_mask.any(axis=1)
                optical_dropout = optical_eligible & (
                    rng.random(MAX_WINDOW_LENGTH) < generation.optical_dropout_probability
                )
                optical_mask[optical_dropout, :] = False

            values = np.full((MAX_WINDOW_LENGTH, len(band_names)), np.nan, dtype=np.float64)
            values[: window.window_length, :] = source_values[
                row_index, window.window_start - 1 : window.window_end, :
            ]
            for band in data.config.data.radar_bands:
                values[~radar_mask, band_index[band]] = np.nan
            for band in optical_bands:
                values[~optical_mask[:, optical_index[band]], band_index[band]] = np.nan

            optical_month_mask = optical_mask.all(axis=1) & position_mask
            internal_optical_gaps = radar_mask & position_mask & ~optical_month_mask
            radar_pattern = _bit_string(radar_mask)
            optical_pattern = _optical_bit_string(optical_mask)
            window_id = _stable_window_id(
                original_id=original_id,
                fold=fold,
                generation_mode=generation.mode,
                augmentation_seed=seed,
                view_index=view_index,
                window_start=window.window_start,
                window_length=window.window_length,
                mask_id=source_mask_id,
                radar_pattern=radar_pattern,
                optical_pattern=optical_pattern,
            )
            manifests.append(
                {
                    "ID": original_id,
                    "window_id": window_id,
                    "original_id": original_id,
                    "fold": fold,
                    "label": int(fold_row[data.config.data.target_column]),
                    "generation_mode": generation.mode,
                    "view_index": view_index,
                    "augmentation_seed": seed,
                    "window_start": window.window_start,
                    "window_end": window.window_end,
                    "window_length": window.window_length,
                    "calendar_months": ",".join(str(value) for value in calendar_months),
                    "relative_positions": ",".join(str(value) for value in relative_positions),
                    "position_mask": _bit_string(position_mask),
                    "radar_availability": radar_pattern,
                    "optical_month_availability": _bit_string(optical_month_mask),
                    "optical_band_availability": optical_pattern,
                    "temporal_dropout_mask": _bit_string(temporal_dropout),
                    "optical_dropout_mask": _bit_string(optical_dropout),
                    "radar_months": int(radar_mask.sum()),
                    "optical_months": int(optical_month_mask.sum()),
                    "internal_optical_gap_count": int(internal_optical_gaps.sum()),
                    "test_mask_optical_gap_count": test_mask_gap_count,
                    "test_mask_used": template is not None,
                    "mask_id": source_mask_id,
                    "source_mask_frequency": source_mask_frequency,
                }
            )
            value_views.append(values)
            month_views.append(calendar_months)
            relative_views.append(relative_positions)
            position_views.append(position_mask)
            radar_views.append(radar_mask)
            optical_views.append(optical_mask)

    dataset = TemporalWindowDataset(
        manifest=pd.DataFrame(manifests),
        values=np.stack(value_views),
        calendar_months=np.stack(month_views),
        relative_positions=np.stack(relative_views),
        position_mask=np.stack(position_views),
        radar_mask=np.stack(radar_views),
        optical_mask=np.stack(optical_views),
        band_names=band_names,
        optical_bands=optical_bands,
    )
    return dataset


def materialize_test_windows(data: CompetitionData) -> TemporalWindowDataset:
    """Represent each observed test row as one masked Phase 2 temporal window."""

    templates = extract_test_mask_templates(data)
    if len(templates) != data.test.shape[0]:
        raise TemporalWindowError("row-aligned test masks must match the test row count")
    band_names = data.config.data.bands
    optical_bands = data.config.data.optical_bands
    band_index = {band: index for index, band in enumerate(band_names)}
    optical_index = {band: index for index, band in enumerate(optical_bands)}
    source_values = _raw_value_cube(data.test, data)

    manifests: list[dict[str, object]] = []
    value_views: list[np.ndarray] = []
    month_views: list[np.ndarray] = []
    relative_views: list[np.ndarray] = []
    position_views: list[np.ndarray] = []
    radar_views: list[np.ndarray] = []
    optical_views: list[np.ndarray] = []
    for row_index, template in enumerate(templates):
        position_mask = np.zeros(MAX_WINDOW_LENGTH, dtype=bool)
        position_mask[: template.window_length] = True
        calendar_months = np.zeros(MAX_WINDOW_LENGTH, dtype=np.int8)
        calendar_months[: template.window_length] = np.arange(
            template.window_start,
            template.window_end + 1,
            dtype=np.int8,
        )
        relative_positions = np.zeros(MAX_WINDOW_LENGTH, dtype=np.int8)
        relative_positions[: template.window_length] = np.arange(
            1, template.window_length + 1, dtype=np.int8
        )
        radar_mask = np.zeros(MAX_WINDOW_LENGTH, dtype=bool)
        optical_mask = np.zeros((MAX_WINDOW_LENGTH, len(optical_bands)), dtype=bool)
        for position, month in enumerate(range(template.window_start, template.window_end + 1)):
            radar_mask[position] = template.radar_availability[month - 1]
            optical_mask[position, :] = template.optical_availability[month - 1]

        values = np.full((MAX_WINDOW_LENGTH, len(band_names)), np.nan, dtype=np.float64)
        values[: template.window_length, :] = source_values[
            row_index, template.window_start - 1 : template.window_end, :
        ]
        for band in data.config.data.radar_bands:
            values[~radar_mask, band_index[band]] = np.nan
        for band in optical_bands:
            values[~optical_mask[:, optical_index[band]], band_index[band]] = np.nan

        optical_month_mask = optical_mask.all(axis=1) & position_mask
        original_id = str(data.test.iloc[row_index][data.config.data.id_column])
        radar_pattern = _bit_string(radar_mask)
        optical_pattern = _optical_bit_string(optical_mask)
        window_id = _stable_window_id(
            original_id=original_id,
            fold=-1,
            generation_mode="observed_test",
            augmentation_seed=-1,
            view_index=0,
            window_start=template.window_start,
            window_length=template.window_length,
            mask_id=template.mask_id,
            radar_pattern=radar_pattern,
            optical_pattern=optical_pattern,
        )
        manifests.append(
            {
                "ID": original_id,
                "window_id": window_id,
                "original_id": original_id,
                "fold": -1,
                "generation_mode": "observed_test",
                "view_index": 0,
                "augmentation_seed": -1,
                "window_start": template.window_start,
                "window_end": template.window_end,
                "window_length": template.window_length,
                "calendar_months": ",".join(str(value) for value in calendar_months),
                "relative_positions": ",".join(str(value) for value in relative_positions),
                "position_mask": _bit_string(position_mask),
                "radar_availability": radar_pattern,
                "optical_month_availability": _bit_string(optical_month_mask),
                "optical_band_availability": optical_pattern,
                "temporal_dropout_mask": "0" * MAX_WINDOW_LENGTH,
                "optical_dropout_mask": "0" * MAX_WINDOW_LENGTH,
                "radar_months": int(radar_mask.sum()),
                "optical_months": int(optical_month_mask.sum()),
                "internal_optical_gap_count": int(
                    (radar_mask & position_mask & ~optical_month_mask).sum()
                ),
                "test_mask_optical_gap_count": template.internal_optical_gap_count,
                "test_mask_used": True,
                "mask_id": template.mask_id,
                "source_mask_frequency": 1,
            }
        )
        value_views.append(values)
        month_views.append(calendar_months)
        relative_views.append(relative_positions)
        position_views.append(position_mask)
        radar_views.append(radar_mask)
        optical_views.append(optical_mask)

    return TemporalWindowDataset(
        manifest=pd.DataFrame(manifests),
        values=np.stack(value_views),
        calendar_months=np.stack(month_views),
        relative_positions=np.stack(relative_views),
        position_mask=np.stack(position_views),
        radar_mask=np.stack(radar_views),
        optical_mask=np.stack(optical_views),
        band_names=band_names,
        optical_bands=optical_bands,
    )


def window_dataset_fingerprint(dataset: TemporalWindowDataset) -> str:
    """Hash manifests, raw values, and masks for reproducibility checks."""

    return _window_fingerprint(dataset, dataset.manifest)


def window_view_fingerprint(dataset: TemporalWindowDataset) -> str:
    """Hash generated view semantics while excluding seed-derived identifiers."""

    semantic_manifest = dataset.manifest.drop(columns=["window_id", "augmentation_seed"])
    return _window_fingerprint(dataset, semantic_manifest)


def _window_fingerprint(dataset: TemporalWindowDataset, manifest: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(manifest.to_csv(index=False, lineterminator="\n").encode())
    for array in (
        dataset.values,
        dataset.calendar_months,
        dataset.relative_positions,
        dataset.position_mask,
        dataset.radar_mask,
        dataset.optical_mask,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update("|".join(dataset.band_names).encode())
    return digest.hexdigest()
