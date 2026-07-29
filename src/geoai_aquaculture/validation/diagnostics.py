"""Deterministic stress-manifest preparation without Phase 5 model training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from geoai_aquaculture.data import ValidationConfig

from .common import ValidationError, dataframe_fingerprint
from .folds import FoldManifest
from .views import ValidationWindowManifest


@dataclass(frozen=True, slots=True)
class LeaveSeasonOutManifest:
    """Compact definitions for fold-isolated leave-season-out evaluations."""

    frame: pd.DataFrame
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ClusterHoldoutPlan:
    """Fold-local, train-only clustering contract deferred to model evaluation."""

    n_clusters: int
    minimum_cluster_size: int
    feature_policy: str = "label_free_invariant_aggregate_features_only"
    scaler_policy: str = "fit_robust_scaler_inside_outer_training_fold"
    test_features_allowed: bool = False
    execution_status: str = "deferred_to_phase5_fold_evaluation"


@dataclass(frozen=True, slots=True)
class ClusterHoldoutManifest:
    """One fold-local label-free clustering result with labels joined only for reporting."""

    frame: pd.DataFrame
    repeat: int
    outer_fold: int
    feature_names: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class SimilarityHoldoutManifest:
    """Fixed most-test-like original subset from externally supplied OOF domain scores."""

    frame: pd.DataFrame
    selected_count: int
    fingerprint: str


def _selection_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, ["repeat", "window_id"]].itertuples(index=False):
        digest.update(f"{row.repeat}|{row.window_id}\n".encode())
    return digest.hexdigest()


def build_leave_season_out_manifest(
    windows: ValidationWindowManifest,
    folds: FoldManifest,
    config: ValidationConfig,
) -> LeaveSeasonOutManifest:
    """Prepare fold-safe seasonal selections while keeping validation originals unseen."""

    if windows.fold_manifest_fingerprint != folds.fingerprint:
        raise ValidationError("leave-season-out windows do not match the fixed folds")
    records: list[dict[str, object]] = []
    for repeat in range(folds.n_repeats):
        repeated = windows.frame.loc[windows.frame["repeat"].eq(repeat)]
        for fold in range(folds.n_splits):
            for season in config.seasons:
                held_month = repeated["window_start"].isin(season.start_months)
                training = repeated.loc[~repeated["fold"].eq(fold) & ~held_month]
                validation = repeated.loc[repeated["fold"].eq(fold) & held_month]
                if training.empty or validation.empty:
                    raise ValidationError("leave-season-out definition contains an empty view")
                train_ids = set(training["original_id"].astype(str))
                valid_ids = set(validation["original_id"].astype(str))
                if train_ids & valid_ids:
                    raise ValidationError(
                        "leave-season-out validation originals cannot train under another season"
                    )
                records.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "held_out_season": season.name,
                        "held_out_start_months": ",".join(map(str, season.start_months)),
                        "training_original_count": len(train_ids),
                        "validation_original_count": len(valid_ids),
                        "training_window_count": int(training.shape[0]),
                        "validation_window_count": int(validation.shape[0]),
                        "training_selection_fingerprint": _selection_fingerprint(training),
                        "validation_selection_fingerprint": _selection_fingerprint(validation),
                        "fold_manifest_fingerprint": folds.fingerprint,
                        "validation_window_fingerprint": windows.fingerprint,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    fingerprint = dataframe_fingerprint(frame)
    return LeaveSeasonOutManifest(frame=frame, fingerprint=fingerprint)


def materialize_leave_season_split(
    windows: ValidationWindowManifest,
    config: ValidationConfig,
    *,
    repeat: int,
    fold: int,
    season_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize one compact seasonal definition for a future fold-local estimator."""

    matches = [season for season in config.seasons if season.name == season_name]
    if len(matches) != 1:
        raise ValidationError(f"unknown configured season: {season_name}")
    season = matches[0]
    repeated = windows.frame.loc[windows.frame["repeat"].eq(repeat)]
    held_month = repeated["window_start"].isin(season.start_months)
    training = repeated.loc[~repeated["fold"].eq(fold) & ~held_month].copy()
    validation = repeated.loc[repeated["fold"].eq(fold) & held_month].copy()
    if set(training["original_id"].astype(str)) & set(validation["original_id"].astype(str)):
        raise ValidationError("leave-season-out materialization leaked validation originals")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def build_cluster_holdout_manifest(
    invariant_features: pd.DataFrame,
    fold_rows: pd.DataFrame,
    *,
    repeat: int,
    outer_fold: int,
    n_clusters: int,
    minimum_cluster_size: int,
    seed: int,
) -> ClusterHoldoutManifest:
    """Fit robust scaling and label-free clusters only on one outer training scope."""

    required_fold = {"original_id", "repeat", "fold", "label"}
    if missing := sorted(required_fold - set(fold_rows.columns)):
        raise ValidationError(f"cluster fold rows are missing columns: {missing}")
    if "original_id" not in invariant_features.columns:
        raise ValidationError("cluster inputs require out-of-band original_id metadata")
    forbidden = {"label", "target", "fold", "repeat", "domain", "test_id", "ID"}
    feature_names = tuple(
        column for column in invariant_features.columns if column != "original_id"
    )
    if not feature_names or forbidden.intersection(feature_names):
        raise ValidationError("cluster inputs contain forbidden labels, folds, IDs, or domain data")
    if invariant_features["original_id"].duplicated().any():
        raise ValidationError("cluster inputs must have one invariant row per original")
    scope = fold_rows.loc[
        fold_rows["repeat"].eq(repeat) & ~fold_rows["fold"].eq(outer_fold),
        ["original_id", "label"],
    ]
    joined = scope.merge(invariant_features, on="original_id", how="left", validate="one_to_one")
    values = joined.loc[:, feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValidationError("cluster inputs must be finite after fold-local feature preparation")
    if joined.shape[0] < n_clusters * minimum_cluster_size:
        raise ValidationError("cluster holdout would force tiny unusable clusters")
    scaled = RobustScaler().fit_transform(values)
    assignments = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(scaled)
    frame = joined.loc[:, ["original_id", "label"]].copy()
    frame["repeat"] = repeat
    frame["outer_fold"] = outer_fold
    frame["cluster"] = assignments.astype(np.int16)
    cluster_sizes = frame["cluster"].value_counts()
    if int(cluster_sizes.min()) < minimum_cluster_size:
        raise ValidationError("fitted cluster holdout contains a tiny unusable cluster")
    frame = frame.sort_values(["cluster", "original_id"], kind="stable", ignore_index=True)
    fingerprint = dataframe_fingerprint(frame)
    return ClusterHoldoutManifest(
        frame=frame,
        repeat=repeat,
        outer_fold=outer_fold,
        feature_names=feature_names,
        fingerprint=fingerprint,
    )


def cluster_balance_summary(manifest: ClusterHoldoutManifest) -> pd.DataFrame:
    """Report cluster size and label balance without using labels to create clusters."""

    return (
        manifest.frame.groupby("cluster", sort=True)
        .agg(
            size=("original_id", "size"),
            positive_count=("label", "sum"),
            positive_rate=("label", "mean"),
        )
        .reset_index()
    )


def build_similarity_holdout_manifest(
    scores: pd.DataFrame,
    folds: FoldManifest,
    *,
    fraction: float,
    minimum_samples: int,
) -> SimilarityHoldoutManifest:
    """Select high OOF train-test similarity scores, stratified only for label balance."""

    required = {"original_id", "similarity_score", "is_oof"}
    if missing := sorted(required - set(scores.columns)):
        raise ValidationError(f"similarity scores are missing columns: {missing}")
    if scores["original_id"].duplicated().any():
        raise ValidationError("similarity scores must contain one row per training original")
    if not is_bool_dtype(scores["is_oof"]) or not scores["is_oof"].all():
        raise ValidationError("adversarial holdout requires out-of-fold domain probabilities")
    if not 0.0 < fraction <= 1.0 or minimum_samples < 1:
        raise ValidationError("similarity holdout size policy is invalid")
    values = scores["similarity_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValidationError("similarity scores must be finite probabilities within [0, 1]")
    labels = folds.frame.loc[folds.frame["repeat"].eq(0), ["original_id", "label"]]
    if set(labels["original_id"].astype(str)) != set(scores["original_id"].astype(str)):
        raise ValidationError("similarity scores must cover exactly the training originals")
    joined = labels.merge(scores, on="original_id", how="left", validate="one_to_one")
    if joined["similarity_score"].isna().any():
        raise ValidationError("similarity scores do not cover all original training rows")
    desired = max(minimum_samples, int(np.ceil(joined.shape[0] * fraction)))
    desired = min(desired, joined.shape[0])
    selected_indices: list[int] = []
    class_counts = joined["label"].value_counts().sort_index()
    allocated = 0
    for class_index, (label, count) in enumerate(class_counts.items()):
        if class_index == len(class_counts) - 1:
            take = desired - allocated
        else:
            take = round(desired * count / joined.shape[0])
            allocated += take
        candidates = joined.loc[joined["label"].eq(label)].sort_values(
            ["similarity_score", "original_id"], ascending=[False, True], kind="stable"
        )
        selected_indices.extend(candidates.head(take).index.tolist())
    joined["selected"] = False
    joined.loc[selected_indices, "selected"] = True
    joined = joined.sort_values(
        ["selected", "similarity_score", "original_id"],
        ascending=[False, False, True],
        kind="stable",
        ignore_index=True,
    )
    frame = joined.loc[:, ["original_id", "label", "similarity_score", "is_oof", "selected"]]
    fingerprint = dataframe_fingerprint(frame)
    return SimilarityHoldoutManifest(
        frame=frame,
        selected_count=int(frame["selected"].sum()),
        fingerprint=fingerprint,
    )
