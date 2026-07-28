"""Loading of competition CSV files without mutating source data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from .config import DataConfig, ProjectConfig
from .schema import TemporalColumn, validate_competition_schema


@dataclass(frozen=True, slots=True)
class CompetitionData:
    """Validated in-memory competition frames with sentinel values converted to NaN."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    temporal_columns: tuple[TemporalColumn, ...]
    raw_missing_counts: Mapping[str, int]
    config: ProjectConfig

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return temporal feature names in their original CSV order."""

        return tuple(item.name for item in self.temporal_columns)


def _read_csv(path: object, id_column: str, dataset: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={id_column: "string"})
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{dataset} CSV not found: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"could not parse {dataset} CSV at {path}: {exc}") from exc


def _replace_sentinel(
    frame: pd.DataFrame, feature_columns: tuple[str, ...], sentinel: float
) -> pd.DataFrame:
    columns = list(feature_columns)
    converted = frame.astype({column: "float64" for column in columns}, copy=True)
    converted.loc[:, columns] = converted.loc[:, columns].mask(
        converted.loc[:, columns].eq(sentinel)
    )
    return converted


def load_competition_data(config: ProjectConfig) -> CompetitionData:
    """Load, validate, and convert sentinel values in all supplied CSV files."""

    data_config: DataConfig = config.data
    train_raw = _read_csv(data_config.train_path, data_config.id_column, "train")
    test_raw = _read_csv(data_config.test_path, data_config.id_column, "test")
    sample = _read_csv(
        data_config.sample_submission_path,
        data_config.id_column,
        "sample submission",
    )
    metadata = validate_competition_schema(train_raw, test_raw, sample, data_config)
    feature_columns = tuple(item.name for item in metadata)
    raw_missing_counts = MappingProxyType(
        {
            "train": int(
                train_raw.loc[:, feature_columns].eq(data_config.missing_sentinel).sum().sum()
            ),
            "test": int(
                test_raw.loc[:, feature_columns].eq(data_config.missing_sentinel).sum().sum()
            ),
        }
    )

    return CompetitionData(
        train=_replace_sentinel(train_raw, feature_columns, data_config.missing_sentinel),
        test=_replace_sentinel(test_raw, feature_columns, data_config.missing_sentinel),
        sample_submission=sample.copy(deep=True),
        temporal_columns=metadata,
        raw_missing_counts=raw_missing_counts,
        config=config,
    )
