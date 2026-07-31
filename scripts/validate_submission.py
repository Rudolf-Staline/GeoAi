#!/usr/bin/env python3
"""Validate a generated competition submission against the official sample order."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geoai_aquaculture.submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/SampleSubmission.csv"),
    )
    args = parser.parse_args()
    submission = pd.read_csv(args.submission, dtype={"ID": "string"})
    sample = pd.read_csv(args.sample, dtype={"ID": "string"})
    validate_submission(submission, sample)
    print(
        f"Valid submission: rows={submission.shape[0]} "
        f"positive_rate={submission['TargetF1'].mean():.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
