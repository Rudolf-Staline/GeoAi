"""Physics-informed and temporal feature engineering."""

from .aggregation import (
    AGGREGATION_NAMES,
    TemporalAggregateResult,
    aggregate_temporal_series,
)
from .audit import (
    FeatureAudit,
    assert_feature_schema_alignment,
    build_feature_audit,
    write_feature_audit_artifacts,
)
from .indices import (
    FeatureEngineeringError,
    MonthlyChannelSpec,
    MonthlyFeatureCollection,
    SafeDivisionResult,
    build_monthly_features,
    safe_divide,
)
from .registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureRegistryError,
    combine_feature_registries,
)
from .representations import (
    FeatureMatrix,
    SequenceFeatureDataset,
    build_feature_representations,
    build_sequence_features,
    build_tabular_features,
)

__all__ = [
    "AGGREGATION_NAMES",
    "FeatureAudit",
    "FeatureDefinition",
    "FeatureEngineeringError",
    "FeatureMatrix",
    "FeatureRegistry",
    "FeatureRegistryError",
    "MonthlyChannelSpec",
    "MonthlyFeatureCollection",
    "SafeDivisionResult",
    "SequenceFeatureDataset",
    "TemporalAggregateResult",
    "aggregate_temporal_series",
    "assert_feature_schema_alignment",
    "build_feature_audit",
    "build_feature_representations",
    "build_monthly_features",
    "build_sequence_features",
    "build_tabular_features",
    "combine_feature_registries",
    "safe_divide",
    "write_feature_audit_artifacts",
]
