"""Phase 7 domain-shift diagnostics and controlled adaptation utilities."""

from .adaptation import (
    AdaptationError,
    AdaptationFoldResult,
    AdaptationResult,
    run_label_adaptation,
    write_adaptation_result,
)
from .adversarial import (
    AdversarialValidationError,
    DomainMetrics,
    DomainValidationResult,
    run_adversarial_validation,
)
from .config import (
    AdaptationMethod,
    DomainModelConfig,
    DomainShiftConfigError,
    ImportanceWeightConfig,
    Phase7Config,
    load_phase7_config,
)
from .dataset import (
    DomainDataset,
    DomainDatasetError,
    DomainFeaturePanels,
    build_domain_dataset,
    build_domain_feature_panels,
)
from .diagnostics import (
    RepresentationSummary,
    feature_shift_table,
    grouped_feature_importance,
    representation_summary,
)
from .evaluation import (
    DomainEvaluationError,
    HoldoutEvaluation,
    build_importance_weights,
    evaluate_similarity_holdout,
    load_original_oof,
)

from .runner import Phase7RunResult, run_phase7

__all__ = [
    "AdaptationError",
    "AdaptationFoldResult",
    "AdaptationMethod",
    "AdaptationResult",
    "AdversarialValidationError",
    "DomainDataset",
    "DomainDatasetError",
    "DomainEvaluationError",
    "DomainFeaturePanels",
    "DomainMetrics",
    "DomainModelConfig",
    "DomainShiftConfigError",
    "DomainValidationResult",
    "HoldoutEvaluation",
    "ImportanceWeightConfig",
    "Phase7Config",
    "Phase7RunResult",
    "RepresentationSummary",
    "build_domain_dataset",
    "build_domain_feature_panels",
    "build_importance_weights",
    "evaluate_similarity_holdout",
    "feature_shift_table",
    "grouped_feature_importance",
    "load_original_oof",
    "load_phase7_config",
    "representation_summary",
    "run_adversarial_validation",
    "run_label_adaptation",
    "run_phase7",
    "write_adaptation_result",
]
