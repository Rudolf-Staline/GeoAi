"""Submission construction and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .constants import FIXED_THRESHOLD, ID_COLUMN, SUBMISSION_COLUMNS


def build_submission(ids: ArrayLike, probabilities: ArrayLike) -> pd.DataFrame:
    """Build a competition submission without allowing threshold tuning."""

    id_values = np.asarray(ids)
    p = np.asarray(probabilities, dtype=float)

    if id_values.ndim != 1 or p.ndim != 1 or id_values.shape[0] != p.shape[0]:
        raise ValueError("ids and probabilities must be aligned one-dimensional arrays")
    if id_values.shape[0] == 0:
        raise ValueError("submission must contain at least one row")
    if pd.isna(id_values).any() or len(set(id_values.tolist())) != id_values.shape[0]:
        raise ValueError("submission IDs must be non-null and unique")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    return pd.DataFrame(
        {
            ID_COLUMN: id_values,
            "TargetF1": (p >= FIXED_THRESHOLD).astype(np.int8),
            "TargetRAUC": p,
        },
        columns=list(SUBMISSION_COLUMNS),
    )


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Validate schema and ID ordering against the official sample submission."""

    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"sample columns must be exactly {SUBMISSION_COLUMNS}")
    if submission.shape[0] != sample.shape[0]:
        raise ValueError("submission row count does not match sample submission")
    submission_ids = submission[ID_COLUMN].astype("string").reset_index(drop=True)
    sample_ids = sample[ID_COLUMN].astype("string").reset_index(drop=True)
    if not submission_ids.equals(sample_ids):
        raise ValueError("submission IDs or row order do not match sample submission")
    if submission.isna().any().any():
        raise ValueError("submission contains missing values")
    if not submission["TargetF1"].isin([0, 1]).all():
        raise ValueError("TargetF1 must be binary")
    if not submission["TargetRAUC"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("TargetRAUC must be in [0, 1]")

    expected = (submission["TargetRAUC"] >= FIXED_THRESHOLD).astype(np.int8)
    if not submission["TargetF1"].astype(np.int8).equals(expected):
        raise ValueError("TargetF1 must use the immutable probability threshold 0.5")
