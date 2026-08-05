"""OOF ensembling, calibration, gating, final fitting, and submission delivery."""

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
from .gating import (
    META_FEATURES,
    FittedGate,
    GatingError,
    GatingResult,
    align_candidate_windows,
    build_gate_features,
    crossfit_gate,
    run_gate_pipeline,
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
    "META_FEATURES",
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
    "FittedGate",
    "GatingError",
    "GatingResult",
    "WeightSelectionResult",
    "align_candidate_windows",
    "build_final_delivery",
    "build_gate_features",
    "crossfit_calibration",
    "crossfit_gate",
    "ensure_final_oof_artifacts",
    "evaluate_fixed_blend",
    "expected_calibration_error",
    "learn_nested_weight",
    "load_final_candidates",
    "load_final_delivery_config",
    "run_gate_pipeline",
]
