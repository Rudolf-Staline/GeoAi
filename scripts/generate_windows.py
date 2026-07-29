"""Generate and audit leakage-safe temporal windows without training a model."""

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
    audit_temporal_windows,
    extract_test_mask_library,
    generate_temporal_windows,
    load_competition_data,
    load_project_config,
    window_dataset_fingerprint,
    window_view_fingerprint,
    write_window_audit_artifacts,
)

LOGGER = logging.getLogger("geoai_aquaculture.temporal_windows")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--mode",
        choices=("sampled", "exhaustive"),
        default=None,
        help="Override the configured generation mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured artifacts/temporal_windows directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run window generation, reproducibility checks, and metadata auditing."""

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
        fold_manifest = assign_original_folds(
            data.train,
            n_splits=config.validation.n_splits,
            seed=config.seed,
            id_column=config.data.id_column,
            target_column=config.data.target_column,
        )
        mask_library = (
            extract_test_mask_library(data) if generation.use_test_missingness_masks else None
        )
        windows = generate_temporal_windows(
            data,
            fold_manifest,
            generation,
            seed=config.seed,
            mask_library=mask_library,
        )
        repeated = generate_temporal_windows(
            data,
            fold_manifest,
            generation,
            seed=config.seed,
            mask_library=mask_library,
        )
        same_seed_reproducible = window_dataset_fingerprint(windows) == window_dataset_fingerprint(
            repeated
        )
        del repeated
        if not same_seed_reproducible:
            raise TemporalWindowError("same-seed generation was not reproducible")

        alternate_seed_changes: bool | None = None
        alternate_seed_preserves_folds: bool | None = None
        if generation.mode == "sampled":
            alternate = generate_temporal_windows(
                data,
                fold_manifest,
                generation,
                seed=config.seed + 1,
                mask_library=mask_library,
            )
            alternate_seed_changes = window_view_fingerprint(windows) != window_view_fingerprint(
                alternate
            )
            alternate_seed_preserves_folds = (
                windows.manifest[["original_id", "fold"]]
                .drop_duplicates()
                .equals(alternate.manifest[["original_id", "fold"]].drop_duplicates())
            )
            if not alternate_seed_changes or not alternate_seed_preserves_folds:
                raise TemporalWindowError(
                    "alternate seed must change sampled views without changing folds"
                )

        audit = audit_temporal_windows(
            windows,
            generation,
            mask_library=mask_library,
            same_seed_reproducible=same_seed_reproducible,
            alternate_seed_changes_views=alternate_seed_changes,
            alternate_seed_preserves_folds=alternate_seed_preserves_folds,
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else config.artifacts_dir / "temporal_windows"
        )
        paths = write_window_audit_artifacts(
            audit,
            windows,
            fold_manifest,
            mask_library,
            output_dir,
        )
    except (
        ConfigError,
        FoldAssignmentError,
        MaskTemplateError,
        SchemaError,
        TemporalWindowError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("temporal-window generation failed: %s", exc)
        return 2

    LOGGER.info("temporal-window audit complete: %d artifacts written", len(paths))
    LOGGER.info("generated_windows=%d", windows.n_windows)
    LOGGER.info("runtime_seconds=%.3f", perf_counter() - started)
    LOGGER.info(
        "length_distribution=%s",
        audit.summary["windows"]["length_distribution"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
