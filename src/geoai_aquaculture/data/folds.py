"""Original-row fold assignment and leakage assertions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.model_selection import StratifiedGroupKFold


class FoldAssignmentError(ValueError):
    """Raised when original-row folds or augmented manifests violate the contract."""


def assign_original_folds(
    train: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    id_column: str = "ID",
    target_column: str = "label",
) -> pd.DataFrame:
    """Assign deterministic grouped folds to unique original rows before augmentation."""

    if n_splits < 2:
        raise FoldAssignmentError("n_splits must be at least 2")
    if seed < 0:
        raise FoldAssignmentError("seed must be non-negative")
    required = {id_column, target_column}
    missing = sorted(required - set(train.columns))
    if missing:
        raise FoldAssignmentError(f"train is missing fold columns: {missing}")
    if train.empty:
        raise FoldAssignmentError("train must contain original rows before fold assignment")
    if train[id_column].isna().any() or train[id_column].duplicated().any():
        raise FoldAssignmentError("original train IDs must be non-null and unique")
    labels = set(train[target_column].unique().tolist())
    if labels != {0, 1}:
        raise FoldAssignmentError("original train target must contain both binary classes")
    class_counts = train[target_column].value_counts()
    if int(class_counts.min()) < n_splits:
        raise FoldAssignmentError("each target class must contain at least n_splits original rows")

    folds = np.full(train.shape[0], -1, dtype=np.int16)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = train[id_column].astype("string").to_numpy()
    target = train[target_column].to_numpy(dtype=np.int8)
    placeholder = np.zeros((train.shape[0], 1), dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(splitter.split(placeholder, target, groups)):
        folds[validation_indices] = fold
    if (folds < 0).any():
        raise FoldAssignmentError("fold assignment did not cover every original row")

    ids = train[id_column].astype("string").reset_index(drop=True)
    manifest = pd.DataFrame(
        {
            id_column: ids,
            "original_id": ids.copy(),
            "fold": folds,
            target_column: train[target_column].reset_index(drop=True).copy(),
        }
    )
    validate_original_fold_manifest(
        train,
        manifest,
        id_column=id_column,
        target_column=target_column,
    )
    return manifest


def validate_original_fold_manifest(
    train: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    id_column: str = "ID",
    target_column: str = "label",
) -> None:
    """Prove a manifest is a one-to-one assignment of the unaugmented train rows."""

    required = {id_column, "original_id", "fold", target_column}
    missing = sorted(required - set(fold_manifest.columns))
    if missing:
        raise FoldAssignmentError(f"fold manifest is missing columns: {missing}")
    if fold_manifest.shape[0] != train.shape[0]:
        raise FoldAssignmentError(
            "fold manifest must contain exactly one row per original train ID"
        )
    if fold_manifest[[id_column, "original_id", "fold", target_column]].isna().any().any():
        raise FoldAssignmentError("fold manifest must not contain missing assignment values")
    if (
        fold_manifest[id_column].duplicated().any()
        or fold_manifest["original_id"].duplicated().any()
    ):
        raise FoldAssignmentError("fold manifest must assign each original ID exactly once")
    if not is_integer_dtype(fold_manifest["fold"]):
        raise FoldAssignmentError("fold assignments must use an integer dtype")
    if (fold_manifest["fold"] < 0).any():
        raise FoldAssignmentError("fold assignments must be non-negative")

    train_ids = train[id_column].astype("string").reset_index(drop=True)
    manifest_ids = fold_manifest[id_column].astype("string").reset_index(drop=True)
    original_ids = fold_manifest["original_id"].astype("string").reset_index(drop=True)
    if not train_ids.equals(manifest_ids) or not train_ids.equals(original_ids):
        raise FoldAssignmentError(
            "fold manifest IDs and order must exactly match the original train rows"
        )
    expected_target = train[target_column].reset_index(drop=True)
    actual_target = fold_manifest[target_column].reset_index(drop=True)
    if not expected_target.equals(actual_target):
        raise FoldAssignmentError("fold manifest targets must match the original train rows")


def assert_no_fold_leakage(window_manifest: pd.DataFrame) -> None:
    """Reject an augmented manifest when one original ID appears in multiple folds."""

    required = {"window_id", "original_id", "fold"}
    missing = sorted(required - set(window_manifest.columns))
    if missing:
        raise FoldAssignmentError(f"window manifest is missing leakage columns: {missing}")
    if window_manifest.empty:
        raise FoldAssignmentError("window manifest must not be empty")
    if window_manifest[list(required)].isna().any().any():
        raise FoldAssignmentError("window leakage columns must not contain missing values")
    if window_manifest["window_id"].duplicated().any():
        raise FoldAssignmentError("window_id values must be unique")
    fold_counts = window_manifest.groupby("original_id", sort=False, observed=True)[
        "fold"
    ].nunique()
    leaking_ids = fold_counts[fold_counts != 1]
    if not leaking_ids.empty:
        raise FoldAssignmentError(
            "augmented copies cross fold boundaries for original IDs: "
            f"{leaking_ids.index.astype(str).tolist()}"
        )
