"""Run one declared Phase 5 CatBoost/LightGBM experiment or OOF diversity audit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from geoai_aquaculture.data import (
    ConfigError,
    FoldAssignmentError,
    MaskTemplateError,
    SchemaError,
    TemporalWindowError,
    load_project_config,
)
from geoai_aquaculture.features import FeatureEngineeringError, FeatureRegistryError
from geoai_aquaculture.models import ModelAdapterError
from geoai_aquaculture.training import (
    DiversityError,
    ExperimentArtifactError,
    ExperimentConfigError,
    TabularTrainingError,
    WeightingError,
    analyze_oof_diversity,
    execute_tabular_experiment,
    load_accepted_candidate_registry,
    load_tabular_experiment_config,
    write_oof_diversity_artifacts,
)
from geoai_aquaculture.validation import ValidationError, load_fold_manifest

LOGGER = logging.getLogger("geoai_aquaculture.tabular")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", type=Path, help="One hand-authored experiment YAML")
    source.add_argument(
        "--approved-list",
        type=Path,
        help="YAML containing an explicit ordered 'experiments' path list",
    )
    source.add_argument(
        "--diversity-registry",
        type=Path,
        help="Analyze complete Stage C candidates without fitting models",
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "screen", "full"),
        default="smoke",
        help="Stage A smoke, Stage B repeat-zero screen, or Stage C authoritative run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a complete artifact with identical config, commit, and fingerprints",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the exact ignored experiment artifact directory",
    )
    return parser


def _approved_paths(path: Path, project_root: Path) -> tuple[Path, ...]:
    source = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentConfigError(f"approved experiment list not found: {source}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid approved experiment list: {source}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("experiments"), list):
        raise ExperimentConfigError("approved list must contain an 'experiments' list")
    paths: list[Path] = []
    for item in raw["experiments"]:
        if not isinstance(item, str) or not item.strip():
            raise ExperimentConfigError("approved experiment paths must be non-empty strings")
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        paths.append(candidate.resolve())
    if not paths or len(paths) != len(set(paths)):
        raise ExperimentConfigError("approved experiment list must be non-empty and unique")
    return tuple(paths)


def main(argv: list[str] | None = None) -> int:
    """Run strictly declared experiments and surface scientific failures clearly."""

    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.resume and args.overwrite:
        LOGGER.error("--resume and --overwrite are mutually exclusive")
        return 2
    try:
        project = load_project_config(args.config)
        if args.diversity_registry is not None:
            folds = load_fold_manifest(
                project.tabular.validation_artifacts_dir / "fold_manifest.csv",
                project.validation,
                expected_fingerprint=project.tabular.fold_manifest_fingerprint,
            )
            registry = load_accepted_candidate_registry(args.diversity_registry, project)
            report = analyze_oof_diversity(registry, folds, project)
            output = project.tabular.experiments_artifacts_dir / "phase5_selection"
            paths = write_oof_diversity_artifacts(output, registry, report)
            LOGGER.info("OOF diversity artifacts complete: %d files", len(paths))
            return 0

        paths = (
            (args.experiment.resolve(),)
            if args.experiment is not None
            else _approved_paths(args.approved_list, project.project_root)
        )
        for path in paths:
            experiment = load_tabular_experiment_config(path)
            LOGGER.info(
                "starting experiment=%s stage=%s family=%s features=%s weighting=%s",
                experiment.experiment_id,
                args.stage,
                experiment.model.family,
                experiment.feature_set,
                experiment.weighting,
            )
            result = execute_tabular_experiment(
                project,
                experiment,
                stage=args.stage,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            if result is None:
                LOGGER.info("compatible completed artifact resumed: %s", experiment.experiment_id)
                continue
            official = result.report.summary["official_metric"]
            robust = result.report.summary["robust_selection"]
            LOGGER.info(
                "completed experiment=%s stage=%s rows=%d combined=%.6f robust=%.6f runtime=%.3f",
                experiment.experiment_id,
                args.stage,
                result.oof.original.shape[0],
                official["mean_combined_score"],
                robust["score"],
                result.runtime_seconds,
            )
        return 0
    except (
        ConfigError,
        DiversityError,
        ExperimentArtifactError,
        ExperimentConfigError,
        FeatureEngineeringError,
        FeatureRegistryError,
        FoldAssignmentError,
        MaskTemplateError,
        ModelAdapterError,
        SchemaError,
        TabularTrainingError,
        TemporalWindowError,
        ValidationError,
        WeightingError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("tabular experiment failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
