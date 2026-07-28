"""Data loading, schema validation, and competition auditing."""

from .audit import DataAudit, audit_competition_data, write_audit_artifacts
from .config import ConfigError, DataConfig, ProjectConfig, load_project_config
from .loading import CompetitionData, load_competition_data
from .schema import (
    SchemaError,
    TemporalColumn,
    parse_temporal_column,
    parse_temporal_columns,
    validate_competition_schema,
)

__all__ = [
    "CompetitionData",
    "ConfigError",
    "DataAudit",
    "DataConfig",
    "ProjectConfig",
    "SchemaError",
    "TemporalColumn",
    "audit_competition_data",
    "load_competition_data",
    "load_project_config",
    "parse_temporal_column",
    "parse_temporal_columns",
    "validate_competition_schema",
    "write_audit_artifacts",
]
