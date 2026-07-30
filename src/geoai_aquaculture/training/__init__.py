"""Training loops, reproducibility and artifact persistence."""

from .artifacts import (
    ExperimentArtifactError,
    assert_resume_compatible,
    experiment_artifact_dir,
    load_experiment_artifact_manifest,
    prepare_experiment_artifact_dir,
    summarize_feature_importance,
    write_tabular_experiment_artifacts,
)
from .config import (
    ExperimentConfigError,
    ExperimentStage,
    ModelFamily,
    ModelProfile,
    TabularExperimentConfig,
    WeightingPolicy,
    load_tabular_experiment_config,
)
from .diversity import (
    AcceptedCandidateRegistry,
    CandidateRecord,
    DiversityError,
    OOFDiversityReport,
    analyze_oof_diversity,
    load_accepted_candidate_registry,
    load_candidate_oof,
    write_oof_diversity_artifacts,
)
from .results import (
    ExperimentArtifactManifest,
    FoldTrainingResult,
    TabularTrainingResult,
)
from .tabular import (
    FoldRunOutput,
    PreparedTabularData,
    TabularTrainingError,
    execute_tabular_experiment,
    prepare_tabular_experiment_data,
    run_tabular_experiment,
    run_tabular_fold,
    stage_repeat_folds,
    validate_full_oof_contract,
    validate_phase3_feature_contract,
)
from .temporal_config import (
    TemporalExperimentConfig,
    TemporalExperimentConfigError,
    TemporalTrainingConfig,
    TemporalViabilityGates,
    load_temporal_experiment_config,
)
from .temporal_diversity import (
    TemporalDiversityError,
    TemporalTreeDiversityReport,
    analyze_temporal_tree_diversity,
    blend_oof_predictions,
    pairwise_oof_summary,
    write_temporal_tree_diversity,
)
from .weights import (
    SampleWeightResult,
    WeightingError,
    build_window_sample_weights,
)

__all__ = [
    "AcceptedCandidateRegistry",
    "CandidateRecord",
    "DiversityError",
    "ExperimentArtifactError",
    "ExperimentArtifactManifest",
    "ExperimentConfigError",
    "ExperimentStage",
    "FoldRunOutput",
    "FoldTrainingResult",
    "ModelFamily",
    "ModelProfile",
    "OOFDiversityReport",
    "PreparedTabularData",
    "SampleWeightResult",
    "TabularExperimentConfig",
    "TabularTrainingError",
    "TabularTrainingResult",
    "TemporalDiversityError",
    "TemporalExperimentConfig",
    "TemporalExperimentConfigError",
    "TemporalTrainingConfig",
    "TemporalTreeDiversityReport",
    "TemporalViabilityGates",
    "WeightingError",
    "WeightingPolicy",
    "analyze_oof_diversity",
    "analyze_temporal_tree_diversity",
    "assert_resume_compatible",
    "blend_oof_predictions",
    "build_window_sample_weights",
    "execute_tabular_experiment",
    "experiment_artifact_dir",
    "load_accepted_candidate_registry",
    "load_candidate_oof",
    "load_experiment_artifact_manifest",
    "load_tabular_experiment_config",
    "load_temporal_experiment_config",
    "pairwise_oof_summary",
    "prepare_experiment_artifact_dir",
    "prepare_tabular_experiment_data",
    "run_tabular_experiment",
    "run_tabular_fold",
    "stage_repeat_folds",
    "summarize_feature_importance",
    "validate_full_oof_contract",
    "validate_phase3_feature_contract",
    "write_oof_diversity_artifacts",
    "write_tabular_experiment_artifacts",
    "write_temporal_tree_diversity",
]

try:
    from .temporal import (  # noqa: F401
        ChannelStatistics,
        PreparedTemporalData,
        SameOriginalPairMap,
        SequenceNormalizer,
        TemporalFoldResult,
        TemporalTrainingError,
        TemporalTrainingResult,
        execute_temporal_experiment,
        prepare_temporal_experiment_data,
        run_temporal_experiment,
        write_temporal_experiment_artifacts,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "ChannelStatistics",
            "PreparedTemporalData",
            "SameOriginalPairMap",
            "SequenceNormalizer",
            "TemporalFoldResult",
            "TemporalTrainingError",
            "TemporalTrainingResult",
            "execute_temporal_experiment",
            "prepare_temporal_experiment_data",
            "run_temporal_experiment",
            "write_temporal_experiment_artifacts",
        ]
    )
