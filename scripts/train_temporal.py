#!/usr/bin/env python3
"""Run one bounded Phase 6 temporal viability experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoai_aquaculture.data import load_project_config
from geoai_aquaculture.training.temporal import execute_temporal_experiment
from geoai_aquaculture.training.temporal_config import load_temporal_experiment_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--stage", choices=("smoke", "screen", "full"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive")
    project = load_project_config(args.config)
    experiment = load_temporal_experiment_config(args.experiment)
    result = execute_temporal_experiment(
        project,
        experiment,
        stage=args.stage,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    if result is None:
        print(f"Compatible completed artifacts found for {experiment.experiment_id}.")
        return 0
    official = result.report.summary["official_metric"]
    robust = result.report.summary["robust_selection"]
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Stage: {result.stage}")
    print(f"Parameters: {result.folds[0].parameter_count}")
    print(f"F1: {official['mean_f1']:.6f}")
    print(f"ROC-AUC: {official['mean_roc_auc']:.6f}")
    print(f"Combined: {official['mean_combined_score']:.6f}")
    print(f"Robust: {robust['score']:.6f}")
    print(f"OOF rows: {result.oof.original.shape[0]}")
    print(f"Window predictions: {result.oof.windows.shape[0]}")
    print(f"Artifacts: {result.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
