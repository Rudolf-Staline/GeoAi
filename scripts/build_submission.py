#!/usr/bin/env python3
"""Build the complete Phase 8 final delivery and competition submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoai_aquaculture.data import load_project_config
from geoai_aquaculture.ensemble import build_final_delivery, load_final_delivery_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final.yaml"))
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.reuse_existing and args.overwrite:
        raise SystemExit("--reuse-existing and --overwrite are mutually exclusive")
    final_config = load_final_delivery_config(args.config)
    project = load_project_config(final_config.project_config)
    result = build_final_delivery(
        final_config,
        project=project,
        reuse_existing=(args.reuse_existing or not args.overwrite),
        overwrite=args.overwrite,
    )
    print(f"Submission: {result.submission_path}")
    print(f"Notebook: {result.notebook_path}")
    print(f"Tree weight: {result.selected_tree_weight:.4f}")
    print(f"Calibration: {result.selected_calibration}")
    print(f"OOF combined: {result.oof_combined_score:.6f}")
    print(f"OOF robust: {result.oof_robust_score:.6f}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
