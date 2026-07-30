#!/usr/bin/env python3
"""Compare one complete temporal OOF artifact with one complete tree artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geoai_aquaculture.data import load_project_config
from geoai_aquaculture.training.temporal_diversity import (
    analyze_temporal_tree_diversity,
    write_temporal_tree_diversity,
)
from geoai_aquaculture.validation import load_fold_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--temporal-artifact", type=Path, required=True)
    parser.add_argument("--tree-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = load_project_config(args.config)
    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    report = analyze_temporal_tree_diversity(
        args.temporal_artifact,
        args.tree_artifact,
        folds,
        project,
    )
    paths = write_temporal_tree_diversity(args.output_dir, report)
    print(f"Pairwise: {report.pairwise}")
    print(report.blends.to_string(index=False))
    print(f"Artifacts: {[path.as_posix() for path in paths]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
