#!/usr/bin/env python3
"""Cross-fit the CatBoost/invariant gate and build a test submission when accepted."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoai_aquaculture.data import load_project_config
from geoai_aquaculture.ensemble.gating import GatingError, run_gate_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--catboost-artifact",
        type=Path,
        default=Path("artifacts/experiments/EXP-TAB-003-CB-FULL-LOWLR"),
    )
    parser.add_argument(
        "--invariant-artifact",
        type=Path,
        default=Path("artifacts/experiments/EXP-TAB-002-LGB-INVARIANT"),
    )
    parser.add_argument(
        "--catboost-test",
        type=Path,
        default=Path("submissions/submission_catboost_low_lr.csv"),
    )
    parser.add_argument(
        "--invariant-test",
        type=Path,
        default=Path("submissions/submission_lightgbm_invariant.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase9_oof_gate"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/submission_oof_gated.csv"),
    )
    parser.add_argument(
        "--allow-unaccepted",
        action="store_true",
        help="Write a diagnostic submission even when the predeclared OOF gate is not met.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project = load_project_config(args.config)
        result = run_gate_pipeline(
            project,
            catboost_artifact=args.catboost_artifact,
            invariant_artifact=args.invariant_artifact,
            catboost_test_submission=args.catboost_test,
            invariant_test_submission=args.invariant_test,
            output_dir=args.output_dir,
            submission_path=args.submission,
            allow_unaccepted=args.allow_unaccepted,
        )
    except (FileNotFoundError, GatingError, ValueError) as exc:
        print(f"ERROR Phase 9 gate failed: {exc}")
        return 1

    boundary = result.boundary_report.summary
    print(f"Accepted: {result.accepted}")
    print(f"Decision: {result.acceptance_reason}")
    print(f"Production C: {result.production_c:.6g}")
    print(
        "Boundary OOF combined: "
        f"{boundary['official_metric']['mean_combined_score']:.6f}"
    )
    print(f"Boundary OOF robust: {boundary['robust_selection']['score']:.6f}")
    print(f"Report: {args.output_dir / 'gate_report.json'}")
    if result.accepted or args.allow_unaccepted:
        print(f"Submission: {args.submission}")
    else:
        print("Submission not written: the immutable OOF acceptance gate was not met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
