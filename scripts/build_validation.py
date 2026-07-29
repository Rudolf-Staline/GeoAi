"""Build fixed Phase 4 folds, validation views, and noncompetitive reference OOF."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from geoai_aquaculture.data import (
    ConfigError,
    FoldAssignmentError,
    MaskTemplateError,
    SchemaError,
    TemporalWindowError,
    extract_test_mask_library,
    load_competition_data,
    load_project_config,
)
from geoai_aquaculture.features import FeatureEngineeringError
from geoai_aquaculture.validation import (
    ValidationError,
    build_leave_season_out_manifest,
    build_repeated_fold_manifest,
    build_validation_windows,
    release_temporary_memory,
    run_reference_estimator,
    write_validation_artifacts,
)

LOGGER = logging.getLogger("geoai_aquaculture.validation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the ignored artifacts/validation directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override only the fold seed for reproducibility diagnostics",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Build structural manifests without the untuned integration estimator",
    )
    parser.add_argument(
        "--include-exhaustive",
        action="store_true",
        help="Also fingerprint all 24 windows per validation original for stress analysis",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute deterministic Phase 4 validation preparation and checks."""

    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = perf_counter()
    try:
        config = load_project_config(args.config)
        if args.seed is not None:
            if args.seed < 0:
                raise ValidationError("validation seed override must be non-negative")
            config = replace(config, validation=replace(config.validation, seed=args.seed))
        data = load_competition_data(config)
        run_reference = config.validation.reference_estimator_enabled and not args.skip_reference
        folds = build_repeated_fold_manifest(
            data.train,
            config.validation,
            id_column=config.data.id_column,
            target_column=config.data.target_column,
        )
        mask_library = extract_test_mask_library(data)
        windows = build_validation_windows(
            data,
            folds,
            config.validation,
            mask_library=mask_library,
            retain_datasets=run_reference,
        )
        leave_season_out = build_leave_season_out_manifest(
            windows.manifest,
            folds,
            config.validation,
        )
        exhaustive = None
        if args.include_exhaustive or config.validation.exhaustive_stress_enabled:
            exhaustive = build_validation_windows(
                data,
                folds,
                config.validation,
                mask_library=None,
                exhaustive=True,
                retain_datasets=False,
            )
        reference = None
        del data, mask_library
        release_temporary_memory()
        if run_reference:
            reference = run_reference_estimator(
                windows,
                folds,
                config.validation,
                config.features,
            )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else config.artifacts_dir / "validation"
        )
        command_arguments = argv if argv is not None else sys.argv[1:]
        command = "python scripts/build_validation.py"
        if command_arguments:
            command = f"{command} {' '.join(command_arguments)}"
        paths = write_validation_artifacts(
            output_dir,
            config=config,
            folds=folds,
            windows=windows,
            leave_season_out=leave_season_out,
            reference=reference,
            runtime_seconds=perf_counter() - started,
            command=command,
            exhaustive_windows=exhaustive,
        )
    except (
        ConfigError,
        FeatureEngineeringError,
        FoldAssignmentError,
        MaskTemplateError,
        SchemaError,
        TemporalWindowError,
        ValidationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("validation construction failed: %s", exc)
        return 2

    LOGGER.info("validation artifacts complete: %d files written", len(paths))
    LOGGER.info(
        "folds=%d repeats=%d originals=%d manifest_rows=%d",
        folds.n_splits,
        folds.n_repeats,
        folds.n_originals,
        folds.frame.shape[0],
    )
    LOGGER.info("validation_windows=%d", windows.manifest.frame.shape[0])
    LOGGER.info("fold_manifest_fingerprint=%s", folds.fingerprint)
    LOGGER.info("validation_window_fingerprint=%s", windows.manifest.fingerprint)
    if reference is not None:
        LOGGER.info(
            "noncompetitive_reference_combined=%.6f robust=%.6f",
            reference.report.summary["official_metric"]["mean_combined_score"],
            reference.report.summary["robust_selection"]["score"],
        )
    LOGGER.info("runtime_seconds=%.3f", perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
