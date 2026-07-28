"""Audit the supplied GeoAI competition CSV files without training a model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

from geoai_aquaculture.data import (
    ConfigError,
    SchemaError,
    audit_competition_data,
    load_competition_data,
    load_project_config,
    write_audit_artifacts,
)

LOGGER = logging.getLogger("geoai_aquaculture.data_audit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="YAML project configuration (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured artifacts/data_audit directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic data audit and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = perf_counter()
    try:
        config = load_project_config(args.config)
        data = load_competition_data(config)
        audit = audit_competition_data(data)
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else config.artifacts_dir / "data_audit"
        )
        paths = write_audit_artifacts(audit, output_dir)
    except (ConfigError, SchemaError, FileNotFoundError, OSError, ValueError) as exc:
        LOGGER.error("data audit failed: %s", exc)
        return 2

    elapsed = perf_counter() - started
    LOGGER.info("data audit complete: %d artifacts written to %s", len(paths), output_dir)
    LOGGER.info("runtime_seconds=%.3f", elapsed)
    LOGGER.info("test_window_lengths=%s", audit.summary["test_windows"]["length_distribution"])
    LOGGER.info(
        "rows_with_optical_gaps=%s",
        audit.summary["test_windows"]["rows_with_optical_gaps"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
