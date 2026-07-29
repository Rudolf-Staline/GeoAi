"""Unlabeled test-availability templates for temporal augmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import pandas as pd

from geoai_aquaculture.constants import OPTICAL_BANDS

from .audit import sensor_month_availability
from .loading import CompetitionData

_ALLOWED_WINDOW_LENGTHS = (4, 5, 6)
_MONTHS = 12


class MaskTemplateError(ValueError):
    """Raised when an availability pattern cannot be used safely."""


def _mask_identifier(
    radar_availability: tuple[bool, ...],
    optical_availability: tuple[tuple[bool, ...], ...],
    optical_bands: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "radar": [int(value) for value in radar_availability],
            "optical": [[int(value) for value in month] for month in optical_availability],
            "optical_bands": list(optical_bands),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"mask_{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class MissingnessMaskTemplate:
    """One distinct radar/optical availability pattern, never test feature values."""

    mask_id: str
    radar_availability: tuple[bool, ...]
    optical_availability: tuple[tuple[bool, ...], ...]
    optical_bands: tuple[str, ...]
    frequency: int = 1

    def __post_init__(self) -> None:
        if len(self.radar_availability) != _MONTHS:
            raise MaskTemplateError("radar availability must contain exactly 12 months")
        if len(self.optical_availability) != _MONTHS:
            raise MaskTemplateError("optical availability must contain exactly 12 months")
        if not self.optical_bands or len(self.optical_bands) != len(set(self.optical_bands)):
            raise MaskTemplateError("optical band names must be non-empty and unique")
        if any(len(month) != len(self.optical_bands) for month in self.optical_availability):
            raise MaskTemplateError("each optical month must contain one flag per optical band")
        if self.frequency < 1:
            raise MaskTemplateError("mask template frequency must be positive")

        radar_months = [
            month for month, available in enumerate(self.radar_availability, start=1) if available
        ]
        if not radar_months:
            raise MaskTemplateError("radar mask length must be one of 4, 5, or 6")
        expected = list(range(radar_months[0], radar_months[-1] + 1))
        if radar_months != expected:
            raise MaskTemplateError("radar availability must form one consecutive window")
        if len(radar_months) not in _ALLOWED_WINDOW_LENGTHS:
            raise MaskTemplateError("radar mask length must be one of 4, 5, or 6")
        for month, (radar, optical) in enumerate(
            zip(self.radar_availability, self.optical_availability, strict=True), start=1
        ):
            if not radar and any(optical):
                raise MaskTemplateError(
                    f"optical availability outside the radar window at month {month}"
                )
        expected_id = _mask_identifier(
            self.radar_availability,
            self.optical_availability,
            self.optical_bands,
        )
        if self.mask_id != expected_id:
            raise MaskTemplateError("mask_id does not match the availability pattern")

    @property
    def window_start(self) -> int:
        """Return the first radar-valid calendar month."""

        return self.radar_availability.index(True) + 1

    @property
    def window_end(self) -> int:
        """Return the last radar-valid calendar month."""

        return len(self.radar_availability) - self.radar_availability[::-1].index(True)

    @property
    def window_length(self) -> int:
        """Return the number of radar-valid months."""

        return sum(self.radar_availability)

    @property
    def optical_month_availability(self) -> tuple[bool, ...]:
        """Apply the Phase 1 all-optical-bands month definition."""

        return tuple(all(month) for month in self.optical_availability)

    @property
    def internal_optical_gap_count(self) -> int:
        """Count radar-valid months lacking at least one optical band."""

        return sum(
            radar and not optical
            for radar, optical in zip(
                self.radar_availability,
                self.optical_month_availability,
                strict=True,
            )
        )


def build_mask_template(
    radar_availability: tuple[bool, ...] | list[bool],
    optical_availability: tuple[tuple[bool, ...], ...] | list[list[bool]],
    *,
    optical_bands: tuple[str, ...] = OPTICAL_BANDS,
    frequency: int = 1,
) -> MissingnessMaskTemplate:
    """Normalize, identify, and validate an unlabeled availability template."""

    radar = tuple(bool(value) for value in radar_availability)
    optical = tuple(tuple(bool(value) for value in month) for month in optical_availability)
    return MissingnessMaskTemplate(
        mask_id=_mask_identifier(radar, optical, optical_bands),
        radar_availability=radar,
        optical_availability=optical,
        optical_bands=optical_bands,
        frequency=frequency,
    )


@dataclass(frozen=True, slots=True)
class MaskLibrary:
    """Deduplicated availability patterns with empirical test frequencies."""

    templates: tuple[MissingnessMaskTemplate, ...]

    def __post_init__(self) -> None:
        if not self.templates:
            raise MaskTemplateError("mask library must contain at least one template")
        identifiers = [template.mask_id for template in self.templates]
        if len(identifiers) != len(set(identifiers)):
            raise MaskTemplateError("mask library template IDs must be unique")
        if identifiers != sorted(identifiers):
            raise MaskTemplateError("mask library templates must use stable mask_id ordering")
        bands = {template.optical_bands for template in self.templates}
        if len(bands) != 1:
            raise MaskTemplateError("mask library templates must use identical optical bands")

    @property
    def observation_count(self) -> int:
        """Return the number of test rows represented by template frequencies."""

        return sum(template.frequency for template in self.templates)

    @property
    def optical_bands(self) -> tuple[str, ...]:
        """Return the shared optical-band order."""

        return self.templates[0].optical_bands

    def to_frame(self) -> pd.DataFrame:
        """Serialize patterns without test IDs or test feature values."""

        return pd.DataFrame(
            [
                {
                    "mask_id": template.mask_id,
                    "frequency": template.frequency,
                    "window_start": template.window_start,
                    "window_end": template.window_end,
                    "window_length": template.window_length,
                    "radar_availability": "".join(
                        str(int(value)) for value in template.radar_availability
                    ),
                    "optical_month_availability": "".join(
                        str(int(value)) for value in template.optical_month_availability
                    ),
                    "optical_band_availability": ";".join(
                        "".join(str(int(value)) for value in month)
                        for month in template.optical_availability
                    ),
                    "internal_optical_gap_count": template.internal_optical_gap_count,
                }
                for template in self.templates
            ]
        )


def extract_test_mask_library(data: CompetitionData) -> MaskLibrary:
    """Extract only boolean missingness patterns from the validated test frame."""

    availability = sensor_month_availability(data.test, data)
    radar_partial = availability["radar_any"] ^ availability["radar_all"]
    if radar_partial.any(axis=None):
        raise MaskTemplateError("test contains partial radar sensor-month availability")

    lookup = {(item.band, item.month): item.name for item in data.temporal_columns}
    observed: dict[str, MissingnessMaskTemplate] = {}
    frequencies: dict[str, int] = {}
    for row_index in range(data.test.shape[0]):
        radar = tuple(bool(value) for value in availability["radar_all"].iloc[row_index].tolist())
        optical = tuple(
            tuple(
                bool(pd.notna(data.test.iloc[row_index][lookup[(band, month)]]))
                for band in data.config.data.optical_bands
            )
            for month in range(1, data.config.data.months + 1)
        )
        template = build_mask_template(
            radar,
            optical,
            optical_bands=data.config.data.optical_bands,
        )
        observed.setdefault(template.mask_id, template)
        frequencies[template.mask_id] = frequencies.get(template.mask_id, 0) + 1

    templates = tuple(
        replace(observed[mask_id], frequency=frequencies[mask_id]) for mask_id in sorted(observed)
    )
    return MaskLibrary(templates=templates)
