"""Stable competition schema constants."""

from __future__ import annotations

ID_COLUMN = "ID"
TARGET_COLUMN = "label"
SUBMISSION_COLUMNS = ("ID", "TargetF1", "TargetRAUC")
MISSING_SENTINEL = -9999.0
FIXED_THRESHOLD = 0.5
MONTHS = tuple(range(1, 13))
RADAR_BANDS = ("VH", "VV")
OPTICAL_BANDS = (
    "blue",
    "green",
    "nir",
    "nira",
    "re1",
    "re2",
    "re3",
    "red",
    "swir1",
    "swir2",
)
BANDS = RADAR_BANDS + OPTICAL_BANDS
