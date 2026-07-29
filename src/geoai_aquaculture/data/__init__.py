"""Data loading, schema validation, and competition auditing."""

from .audit import (
    DataAudit,
    audit_competition_data,
    sensor_month_availability,
    write_audit_artifacts,
)
from .config import (
    ConfigError,
    DataConfig,
    ProjectConfig,
    ValidationConfig,
    WindowGenerationConfig,
    load_project_config,
)
from .folds import (
    FoldAssignmentError,
    assert_no_fold_leakage,
    assign_original_folds,
    validate_original_fold_manifest,
)
from .loading import CompetitionData, load_competition_data
from .masks import (
    MaskLibrary,
    MaskTemplateError,
    MissingnessMaskTemplate,
    build_mask_template,
    extract_test_mask_library,
)
from .schema import (
    SchemaError,
    TemporalColumn,
    parse_temporal_column,
    parse_temporal_columns,
    validate_competition_schema,
)
from .window_audit import (
    WindowAudit,
    audit_temporal_windows,
    write_window_audit_artifacts,
)
from .windows import (
    MAX_WINDOW_LENGTH,
    ConsecutiveWindow,
    TemporalWindowDataset,
    TemporalWindowError,
    enumerate_consecutive_windows,
    generate_temporal_windows,
    window_dataset_fingerprint,
    window_view_fingerprint,
)

__all__ = [
    "MAX_WINDOW_LENGTH",
    "CompetitionData",
    "ConfigError",
    "ConsecutiveWindow",
    "DataAudit",
    "DataConfig",
    "FoldAssignmentError",
    "MaskLibrary",
    "MaskTemplateError",
    "MissingnessMaskTemplate",
    "ProjectConfig",
    "SchemaError",
    "TemporalColumn",
    "TemporalWindowDataset",
    "TemporalWindowError",
    "ValidationConfig",
    "WindowAudit",
    "WindowGenerationConfig",
    "assert_no_fold_leakage",
    "assign_original_folds",
    "audit_competition_data",
    "audit_temporal_windows",
    "build_mask_template",
    "enumerate_consecutive_windows",
    "extract_test_mask_library",
    "generate_temporal_windows",
    "load_competition_data",
    "load_project_config",
    "parse_temporal_column",
    "parse_temporal_columns",
    "sensor_month_availability",
    "validate_competition_schema",
    "validate_original_fold_manifest",
    "window_dataset_fingerprint",
    "window_view_fingerprint",
    "write_audit_artifacts",
    "write_window_audit_artifacts",
]
