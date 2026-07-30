"""Tree and optional temporal model implementations."""

from .tabular import (
    CatBoostAdapter,
    LightGBMAdapter,
    ModelAdapterError,
    ModelFitMetadata,
    TabularModelAdapter,
    create_tabular_model_adapter,
)

__all__ = [
    "CatBoostAdapter",
    "LightGBMAdapter",
    "ModelAdapterError",
    "ModelFitMetadata",
    "TabularModelAdapter",
    "create_tabular_model_adapter",
]

try:
    from .temporal import (  # noqa: F401
        SensorGatedGRU,
        TemporalArchitecture,
        TemporalForwardOutput,
        TemporalModelError,
        architecture_from_dict,
        count_trainable_parameters,
        masked_mean_pool,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "SensorGatedGRU",
            "TemporalArchitecture",
            "TemporalForwardOutput",
            "TemporalModelError",
            "architecture_from_dict",
            "count_trainable_parameters",
            "masked_mean_pool",
        ]
    )
