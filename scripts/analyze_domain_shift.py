#!/usr/bin/env python3
"""Run Phase 7 train/test diagnostics and controlled adaptations."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoai_aquaculture.data import load_project_config
from geoai_aquaculture.domain_shift import load_phase7_config
from geoai_aquaculture.domain_shift.runner import run_phase7


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--phase7-config",
        type=Path,
        default=Path("configs/experiments/phase7_domain_shift.yaml"),
    )
    parser.add_argument("--diagnostics-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project = load_project_config(args.config)
    config = load_phase7_config(args.phase7_config)
    result = run_phase7(project, config, run_adaptations=not args.diagnostics_only)
    selected = max(result.representation_results, key=lambda value: value.metrics.roc_auc)
    print(f"Selected diagnostic representation: {selected.representation}")
    print(f"Domain ROC-AUC: {selected.metrics.roc_auc:.6f}")
    for decision in result.decisions:
        print(
            f"{decision['method']}: {decision['decision']} "
            f"(robust={decision['mean_robust_score']:.6f})"
        )
    print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
