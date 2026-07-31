"""OOF ensembling, calibration, final fitting, and submission delivery."""

from .calibration import (
    CalibrationError,
    CalibrationEvaluation,
    FittedCalibrator,
    crossfit_calibration,
    expected_calibration_error,
)
from .config import (
    FinalCandidateConfig,
    FinalConfigError,
    FinalDeliveryConfig,
    load_final_delivery_config,
)
from .delivery import (
    FinalDeliveryError,
    FinalDeliveryResult,
    build_final_delivery,
    ensure_final_oof_artifacts,
)
from .oof import (
    BlendEvaluation,
    FinalCandidate,
    FinalOOFError,
    WeightSelectionResult,
    evaluate_fixed_blend,
    learn_nested_weight,
    load_final_candidates,
)

__all__ = [
    "BlendEvaluation",
    "CalibrationError",
    "CalibrationEvaluation",
    "FinalCandidate",
    "FinalCandidateConfig",
    "FinalConfigError",
    "FinalDeliveryConfig",
    "FinalDeliveryError",
    "FinalDeliveryResult",
    "FinalOOFError",
    "FittedCalibrator",
    "WeightSelectionResult",
    "build_final_delivery",
    "crossfit_calibration",
    "ensure_final_oof_artifacts",
    "evaluate_fixed_blend",
    "expected_calibration_error",
    "learn_nested_weight",
    "load_final_candidates",
    "load_final_delivery_config",
]
