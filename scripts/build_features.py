"""Build and audit deterministic Phase 3 features without training a model."""

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
    assign_original_folds,
    extract_test_mask_library,
    generate_temporal_windows,
    load_competition_data,
    load_project_config,
    materialize_test_windows,
    window_dataset_fingerprint,
)
from geoai_aquaculture.features import (
    FeatureEngineeringError,
    build_feature_audit,
    build_feature_representations,
    write_feature_audit_artifacts,
)

LOGGER = logging.getLogger("geoai_aquaculture.features")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--mode",
        choices=("sampled", "exhaustive"),
        default=None,
        help="Override the configured train-window generation mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured artifacts/features directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build aligned train/test representations and aggregate-only audit artifacts."""

    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = perf_counter()
    try:
        config = load_project_config(args.config)
        generation = config.augmentation
        if args.mode == "exhaustive":
            generation = replace(
                generation,
                exhaustive_windows=True,
                use_test_missingness_masks=False,
            )
        elif args.mode == "sampled":
            generation = replace(generation, exhaustive_windows=False)

        data = load_competition_data(config)
        folds = assign_original_folds(
            data.train,
            n_splits=config.validation.n_splits,
            seed=config.seed,
            id_column=config.data.id_column,
            target_column=config.data.target_column,
        )
        mask_library = (
            extract_test_mask_library(data) if generation.use_test_missingness_masks else None
        )
        train_windows = generate_temporal_windows(
            data,
            folds,
            generation,
            seed=config.seed,
            mask_library=mask_library,
        )
        test_windows = materialize_test_windows(data)
        train_window_fingerprint = window_dataset_fingerprint(train_windows)
        test_window_fingerprint = window_dataset_fingerprint(test_windows)

        train_tabular, train_sequence = build_feature_representations(
            train_windows,
            config.features,
        )
        test_tabular, test_sequence = build_feature_representations(
            test_windows,
            config.features,
        )
        repeated_test_tabular, repeated_test_sequence = build_feature_representations(
            test_windows,
            config.features,
        )
        deterministic_rebuild = (
            test_tabular.fingerprint == repeated_test_tabular.fingerprint
            and test_sequence.fingerprint == repeated_test_sequence.fingerprint
        )
        if not deterministic_rebuild:
            raise FeatureEngineeringError("same-input feature construction was not deterministic")
        input_windows_unchanged = train_window_fingerprint == window_dataset_fingerprint(
            train_windows
        ) and test_window_fingerprint == window_dataset_fingerprint(test_windows)
        if not input_windows_unchanged:
            raise FeatureEngineeringError("feature construction mutated temporal-window inputs")

        audit = build_feature_audit(
            train_tabular,
            test_tabular,
            train_sequence,
            test_sequence,
            deterministic_rebuild=deterministic_rebuild,
            input_windows_unchanged=input_windows_unchanged,
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else config.artifacts_dir / "features"
        )
        command_arguments = argv if argv is not None else sys.argv[1:]
        command = "python scripts/build_features.py"
        if command_arguments:
            command = f"{command} {' '.join(command_arguments)}"
        paths = write_feature_audit_artifacts(
            audit,
            output_dir,
            config=config,
            runtime_seconds=perf_counter() - started,
            command=command,
        )
    except (
        ConfigError,
        FeatureEngineeringError,
        FoldAssignmentError,
        MaskTemplateError,
        SchemaError,
        TemporalWindowError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("feature construction failed: %s", exc)
        return 2

    LOGGER.info("feature audit complete: %d aggregate-only artifacts written", len(paths))
    LOGGER.info("generation_mode=%s", generation.mode)
    LOGGER.info("train_windows=%d test_windows=%d", train_windows.n_windows, test_windows.n_windows)
    LOGGER.info(
        "tabular_shape=%s sequence_channels=(radar=%d,optical=%d,indices=%d)",
        train_tabular.features.shape,
        train_sequence.radar_values.shape[2],
        train_sequence.optical_values.shape[2],
        train_sequence.monthly_indices.shape[2],
    )
    LOGGER.info("runtime_seconds=%.3f", perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
