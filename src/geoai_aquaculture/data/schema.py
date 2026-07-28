"""Competition schema and temporal-column validation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from geoai_aquaculture.constants import SUBMISSION_COLUMNS

from .config import DataConfig

Sensor = Literal["radar", "optical"]
_TEMPORAL_COLUMN = re.compile(r"^(?P<band>[A-Za-z0-9]+)_(?P<month>\d{2})$")


class SchemaError(ValueError):
    """Raised when supplied competition data do not match the expected schema."""


@dataclass(frozen=True, slots=True)
class TemporalColumn:
    """Parsed metadata for one monthly sensor-band column."""

    position: int
    name: str
    band: str
    month: int
    sensor: Sensor


def parse_temporal_column(name: str, config: DataConfig, position: int = 0) -> TemporalColumn:
    """Parse a feature name into validated band, month, and sensor metadata."""

    match = _TEMPORAL_COLUMN.fullmatch(name)
    if match is None:
        raise SchemaError(f"temporal column '{name}' must match '<band>_<two-digit month>'")
    band = match.group("band")
    if band not in config.bands:
        raise SchemaError(f"temporal column '{name}' uses unknown band '{band}'")
    month = int(match.group("month"))
    if not 1 <= month <= config.months:
        raise SchemaError(
            f"temporal column '{name}' has month {month}; expected 01-{config.months:02d}"
        )
    sensor: Sensor = "radar" if band in config.radar_bands else "optical"
    return TemporalColumn(position=position, name=name, band=band, month=month, sensor=sensor)


def parse_temporal_columns(
    feature_columns: Sequence[str], config: DataConfig
) -> tuple[TemporalColumn, ...]:
    """Parse features in source order and prove the full band-month grid is present."""

    if len(feature_columns) != len(set(feature_columns)):
        raise SchemaError("temporal feature columns must be unique")
    metadata = tuple(
        parse_temporal_column(name, config, position)
        for position, name in enumerate(feature_columns)
    )
    observed = {(item.band, item.month) for item in metadata}
    expected = {(band, month) for month in range(1, config.months + 1) for band in config.bands}
    if observed != expected or len(metadata) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SchemaError(
            "temporal features must contain every configured band-month exactly once; "
            f"missing={missing}, extra={extra}"
        )
    return metadata


def _validate_ids(frame: pd.DataFrame, dataset: str, id_column: str) -> None:
    if id_column not in frame.columns:
        raise SchemaError(f"{dataset} is missing ID column '{id_column}'")
    if frame[id_column].isna().any():
        raise SchemaError(f"{dataset} IDs must not contain missing values")
    if frame[id_column].duplicated().any():
        raise SchemaError(f"{dataset} IDs must be unique")


def _validate_numeric_features(
    frame: pd.DataFrame, dataset: str, feature_columns: Sequence[str]
) -> None:
    nonnumeric = [column for column in feature_columns if not is_numeric_dtype(frame[column])]
    if nonnumeric:
        raise SchemaError(f"{dataset} feature columns must be numeric: {nonnumeric}")
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise SchemaError(
            f"{dataset} contains NaN or infinite feature values; use only the configured sentinel"
        )


def validate_competition_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    config: DataConfig,
) -> tuple[TemporalColumn, ...]:
    """Validate all competition frames and return temporal metadata in source order."""

    _validate_ids(train, "train", config.id_column)
    _validate_ids(test, "test", config.id_column)
    _validate_ids(sample_submission, "sample submission", config.id_column)

    if config.target_column not in train.columns:
        raise SchemaError(f"train is missing target column '{config.target_column}'")
    if config.target_column in test.columns:
        raise SchemaError(f"test must not contain target column '{config.target_column}'")
    if train[config.target_column].isna().any():
        raise SchemaError("train target must not contain missing values")
    labels = set(train[config.target_column].unique().tolist())
    if labels != {0, 1}:
        raise SchemaError(f"train target must contain both binary classes 0 and 1; found {labels}")

    train_features = [
        column for column in train.columns if column not in {config.id_column, config.target_column}
    ]
    test_features = [column for column in test.columns if column != config.id_column]
    if train_features != test_features:
        raise SchemaError("train and test temporal feature columns or ordering do not match")
    metadata = parse_temporal_columns(train_features, config)
    _validate_numeric_features(train, "train", train_features)
    _validate_numeric_features(test, "test", test_features)

    if tuple(sample_submission.columns) != SUBMISSION_COLUMNS:
        raise SchemaError(f"sample submission columns must be exactly {list(SUBMISSION_COLUMNS)}")
    if sample_submission.shape[0] != test.shape[0]:
        raise SchemaError("sample submission row count must match test")
    if not sample_submission[config.id_column].equals(test[config.id_column]):
        raise SchemaError("sample submission IDs and order must match test")
    for column in SUBMISSION_COLUMNS[1:]:
        if not is_numeric_dtype(sample_submission[column]):
            raise SchemaError(f"sample submission column '{column}' must be numeric")
        values = sample_submission[column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise SchemaError(f"sample submission column '{column}' must be finite")

    return metadata
