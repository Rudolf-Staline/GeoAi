"""Tree and temporal model implementations."""

from .tabular import (
    CatBoostAdapter,
    LightGBMAdapter,
    ModelAdapterError,
    ModelFitMetadata,
    TabularModelAdapter,
    create_tabular_model_adapter,
)
from .temporal import (
    SensorGatedGRU,
    TemporalArchitecture,
    TemporalForwardOutput,
    TemporalModelError,
    architecture_from_dict,
    count_trainable_parameters,
    masked_mean_pool,
)

__all__ = [
    "CatBoostAdapter",
    "LightGBMAdapter",
    "ModelAdapterError",
    "ModelFitMetadata",
    "SensorGatedGRU",
    "TabularModelAdapter",
    "TemporalArchitecture",
    "TemporalForwardOutput",
    "TemporalModelError",
    "architecture_from_dict",
    "count_trainable_parameters",
    "create_tabular_model_adapter",
    "masked_mean_pool",
]
