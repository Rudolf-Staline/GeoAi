"""Feature and representation diagnostics for Phase 7 domain shift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from geoai_aquaculture.features import FeatureRegistry

from .adversarial import DomainValidationResult
from .dataset import DomainDataset


@dataclass(frozen=True, slots=True)
class RepresentationSummary:
    """One concise domain separability record."""

    representation: str
    roc_auc: float
    accuracy: float
    log_loss: float
    brier_score: float
    entity_count: int
    fingerprint: str


def representation_summary(result: DomainValidationResult) -> RepresentationSummary:
    metrics = result.metrics
    return RepresentationSummary(
        representation=result.representation,
        roc_auc=metrics.roc_auc,
        accuracy=metrics.accuracy,
        log_loss=metrics.log_loss,
        brier_score=metrics.brier_score,
        entity_count=metrics.entity_count,
        fingerprint=result.fingerprint,
    )


def grouped_feature_importance(
    result: DomainValidationResult,
    registry: FeatureRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate native domain importance by feature and registry group."""

    feature = (
        result.feature_importance.groupby("feature", observed=True, as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            folds=("fold", "nunique"),
        )
        .fillna({"std_importance": 0.0})
        .sort_values(["mean_importance", "feature"], ascending=[False, True], ignore_index=True)
    )
    group_lookup = {
        definition.name: definition.feature_group for definition in registry.definitions
    }
    feature["feature_group"] = feature["feature"].map(group_lookup)
    grouped = (
        feature.groupby("feature_group", observed=True, as_index=False)
        .agg(
            total_mean_importance=("mean_importance", "sum"),
            maximum_feature_importance=("mean_importance", "max"),
            feature_count=("feature", "size"),
        )
        .sort_values("total_mean_importance", ascending=False, ignore_index=True)
    )
    total = float(grouped["total_mean_importance"].sum())
    grouped["importance_share"] = (
        grouped["total_mean_importance"] / total if total > 0.0 else 0.0
    )
    return feature, grouped


def _psi(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    combined = np.concatenate((train, test))
    if combined.size == 0 or np.allclose(combined, combined[0]):
        return 0.0
    quantiles = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, bins + 1)))
    if quantiles.size < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    train_hist = np.histogram(train, bins=quantiles)[0].astype(np.float64)
    test_hist = np.histogram(test, bins=quantiles)[0].astype(np.float64)
    train_prop = np.clip(train_hist / max(1.0, train_hist.sum()), 1e-6, None)
    test_prop = np.clip(test_hist / max(1.0, test_hist.sum()), 1e-6, None)
    return float(np.sum((test_prop - train_prop) * np.log(test_prop / train_prop)))


def feature_shift_table(
    dataset: DomainDataset,
    ranked_importance: pd.DataFrame,
    *,
    feature_limit: int,
) -> pd.DataFrame:
    """Compute entity-level distribution distances for the most domain-important features."""

    top = tuple(ranked_importance.head(feature_limit)["feature"].astype(str))
    frame = dataset.features.loc[:, list(top)].copy()
    frame.insert(0, "source", dataset.metadata["source"].to_numpy())
    frame.insert(1, "entity_id", dataset.entity_ids)
    entity = frame.groupby(
        ["source", "entity_id"], observed=True, as_index=False
    ).mean(numeric_only=True)
    records: list[dict[str, float | str]] = []
    for name in top:
        train_all = entity.loc[entity["source"].eq("train"), name].to_numpy(dtype=np.float64)
        test_all = entity.loc[entity["source"].eq("test"), name].to_numpy(dtype=np.float64)
        train = train_all[np.isfinite(train_all)]
        test = test_all[np.isfinite(test_all)]
        if train.size == 0 or test.size == 0:
            ks = np.nan
            wasserstein = np.nan
            psi = np.nan
        else:
            ks = float(ks_2samp(train, test, alternative="two-sided", method="auto").statistic)
            wasserstein = float(wasserstein_distance(train, test))
            psi = _psi(train, test)
        records.append(
            {
                "feature": name,
                "train_missing_rate": float(np.isnan(train_all).mean()),
                "test_missing_rate": float(np.isnan(test_all).mean()),
                "train_mean": float(np.mean(train)) if train.size else np.nan,
                "test_mean": float(np.mean(test)) if test.size else np.nan,
                "train_median": float(np.median(train)) if train.size else np.nan,
                "test_median": float(np.median(test)) if test.size else np.nan,
                "ks_statistic": ks,
                "wasserstein_distance": wasserstein,
                "psi": psi,
            }
        )
    return pd.DataFrame.from_records(records).merge(
        ranked_importance.loc[:, ["feature", "mean_importance", "feature_group"]],
        on="feature",
        how="left",
        validate="one_to_one",
    )
