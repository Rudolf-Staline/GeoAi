"""Repeated stratified original-row folds, independent of all model code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_integer_dtype

from geoai_aquaculture.data import ValidationConfig, assign_original_folds

from .common import ValidationError, dataframe_fingerprint

FOLD_COLUMNS = ("ID", "original_id", "label", "repeat", "fold", "repeat_seed")


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    """One original row's validation-fold assignment in one repeat."""

    original_id: str
    label: int
    repeat: int
    fold: int


@dataclass(frozen=True, slots=True)
class FoldManifest:
    """Validated repeated folds over original rows with a content fingerprint."""

    frame: pd.DataFrame
    n_originals: int
    n_splits: int
    n_repeats: int
    seed: int
    fingerprint: str

    def __post_init__(self) -> None:
        validate_fold_manifest(self)

    def for_repeat(self, repeat: int) -> pd.DataFrame:
        """Return one Phase 2-compatible original-row manifest in source ID order."""

        if repeat < 0 or repeat >= self.n_repeats:
            raise ValidationError(f"repeat must be within 0-{self.n_repeats - 1}")
        selected = self.frame.loc[self.frame["repeat"].eq(repeat)]
        if selected.shape[0] != self.n_originals:
            raise ValidationError("requested repeat does not cover every original")
        return selected.loc[:, ["ID", "original_id", "fold", "label"]].reset_index(drop=True)


def _repeat_seed(seed: int, repeat: int) -> int:
    return seed + 10_007 * repeat


def build_repeated_fold_manifest(
    train: pd.DataFrame,
    config: ValidationConfig,
    *,
    id_column: str = "ID",
    target_column: str = "label",
) -> FoldManifest:
    """Split unaugmented originals once per repeat before any window generation."""

    if train[id_column].duplicated().any():
        raise ValidationError("fold construction requires one row per original before augmentation")
    source = train.loc[:, [id_column, target_column]].copy(deep=True)
    pieces: list[pd.DataFrame] = []
    for repeat in range(config.n_repeats):
        repeat_seed = _repeat_seed(config.seed, repeat)
        assignment = assign_original_folds(
            source,
            n_splits=config.n_splits,
            seed=repeat_seed,
            id_column=id_column,
            target_column=target_column,
        ).rename(columns={id_column: "ID", target_column: "label"})
        assignment["repeat"] = repeat
        assignment["repeat_seed"] = repeat_seed
        pieces.append(assignment.loc[:, list(FOLD_COLUMNS)])
    frame = pd.concat(pieces, ignore_index=True)
    frame["ID"] = frame["ID"].astype("string")
    frame["original_id"] = frame["original_id"].astype("string")
    frame["repeat"] = frame["repeat"].astype("int16")
    frame["fold"] = frame["fold"].astype("int16")
    frame["repeat_seed"] = frame["repeat_seed"].astype("int64")
    fingerprint = dataframe_fingerprint(frame, columns=FOLD_COLUMNS)
    return FoldManifest(
        frame=frame,
        n_originals=source.shape[0],
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        seed=config.seed,
        fingerprint=fingerprint,
    )


def load_fold_manifest(
    path: str | Path,
    config: ValidationConfig,
    *,
    expected_fingerprint: str | None = None,
) -> FoldManifest:
    """Load the authoritative persisted manifest with its exact scientific dtypes."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"fold manifest not found: {source}")
    frame = pd.read_csv(
        source,
        dtype={
            "ID": "string",
            "original_id": "string",
            "label": "int64",
            "repeat": "int16",
            "fold": "int16",
            "repeat_seed": "int64",
        },
    )
    fingerprint = dataframe_fingerprint(frame, columns=FOLD_COLUMNS)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValidationError("loaded fold-manifest fingerprint mismatch")
    manifest = FoldManifest(
        frame=frame,
        n_originals=int(frame["original_id"].nunique()),
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        seed=config.seed,
        fingerprint=fingerprint,
    )
    if manifest.frame["repeat_seed"].iloc[0] != config.seed:
        raise ValidationError("loaded fold manifest does not use the configured seed")
    return manifest


def validate_fold_manifest(manifest: FoldManifest) -> None:
    """Prove exact repeated coverage, label consistency, and fold uniqueness."""

    frame = manifest.frame
    missing = sorted(set(FOLD_COLUMNS) - set(frame.columns))
    if missing:
        raise ValidationError(f"repeated fold manifest is missing columns: {missing}")
    if frame.shape[0] != manifest.n_originals * manifest.n_repeats:
        raise ValidationError("fold manifest must have one row per original per repeat")
    if frame.loc[:, list(FOLD_COLUMNS)].isna().any().any():
        raise ValidationError("fold manifest cannot contain missing assignments")
    if not all(is_integer_dtype(frame[column]) for column in ("label", "repeat", "fold")):
        raise ValidationError("labels, repeats, and folds must use integer dtypes")
    if frame.duplicated(["original_id", "repeat"]).any():
        raise ValidationError("an original may occur only once in each validation repeat")
    if not frame["ID"].astype("string").equals(frame["original_id"].astype("string")):
        raise ValidationError("ID and original_id must remain identical in fold metadata")
    expected_repeats = set(range(manifest.n_repeats))
    if set(frame["repeat"].unique()) != expected_repeats:
        raise ValidationError("fold manifest does not contain the configured repeats")
    expected_folds = set(range(manifest.n_splits))
    if any(
        set(group["fold"].unique()) != expected_folds
        for _, group in frame.groupby("repeat", observed=True)
    ):
        raise ValidationError("each repeat must contain every configured fold")
    canonical_ids: tuple[str, ...] | None = None
    for _, group in frame.groupby("repeat", sort=True, observed=True):
        ids = tuple(group["original_id"].astype(str))
        canonical_ids = ids if canonical_ids is None else canonical_ids
        if ids != canonical_ids:
            raise ValidationError(
                "every repeat must preserve identical original ID coverage and order"
            )
        if group["fold"].nunique() > manifest.n_splits:
            raise ValidationError("one original crosses validation folds within a repeat")
    if frame.groupby("original_id", sort=False, observed=True)["label"].nunique().ne(1).any():
        raise ValidationError("an original's label changed between repeats")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValidationError("fold manifest labels must contain both binary classes")
    actual = dataframe_fingerprint(frame, columns=FOLD_COLUMNS)
    if actual != manifest.fingerprint:
        raise ValidationError("fold-manifest content fingerprint mismatch")


def fold_balance_summary(manifest: FoldManifest) -> pd.DataFrame:
    """Report original-level train/validation class balance for every fold."""

    records: list[dict[str, float | int]] = []
    for (repeat, fold), valid in manifest.frame.groupby(
        ["repeat", "fold"], sort=True, observed=True
    ):
        repeat_frame = manifest.frame.loc[manifest.frame["repeat"].eq(repeat)]
        train = repeat_frame.loc[~repeat_frame["fold"].eq(fold)]
        records.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "train_count": int(train.shape[0]),
                "validation_count": int(valid.shape[0]),
                "train_positive_count": int(train["label"].sum()),
                "validation_positive_count": int(valid["label"].sum()),
                "train_positive_rate": float(train["label"].mean()),
                "validation_positive_rate": float(valid["label"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)
