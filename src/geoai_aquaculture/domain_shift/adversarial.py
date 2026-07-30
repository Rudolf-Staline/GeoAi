"""Grouped OOF adversarial validation for train-vs-test separability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from geoai_aquaculture.models import LightGBMAdapter

from .config import DomainModelConfig
from .dataset import DomainDataset


class AdversarialValidationError(ValueError):
    """Raised when domain OOF predictions violate the grouped entity contract."""


@dataclass(frozen=True, slots=True)
class DomainMetrics:
    """Entity-level train-vs-test diagnostic metrics."""

    roc_auc: float
    accuracy: float
    log_loss: float
    brier_score: float
    predicted_test_rate: float
    entity_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "roc_auc": self.roc_auc,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "predicted_test_rate": self.predicted_test_rate,
            "entity_count": self.entity_count,
        }


@dataclass(frozen=True, slots=True)
class DomainValidationResult:
    """Complete grouped domain OOF result for one representation."""

    representation: str
    metrics: DomainMetrics
    window_oof: pd.DataFrame
    entity_oof: pd.DataFrame
    train_similarity_scores: pd.DataFrame
    feature_importance: pd.DataFrame
    fold_metrics: pd.DataFrame
    fingerprint: str


def _fold_weights(dataset: DomainDataset, indices: np.ndarray) -> np.ndarray:
    groups = dataset.groups[indices]
    labels = dataset.labels[indices]
    counts = pd.Series(groups).value_counts()
    weights = np.asarray([1.0 / float(counts[group]) for group in groups], dtype=np.float64)
    entity_count = int(pd.Series(groups).nunique())
    for domain in (0, 1):
        selector = labels == domain
        weights[selector] *= (entity_count / 2.0) / float(weights[selector].sum())
    return weights


def _entity_oof(window_oof: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        window_oof.groupby(
            ["source", "entity_id", "group_id", "domain_label"],
            sort=True,
            observed=True,
            as_index=False,
        )
        .agg(probability=("probability", "mean"), window_count=("probability", "size"))
        .sort_values(["source", "entity_id"], kind="stable", ignore_index=True)
    )
    if grouped["group_id"].duplicated().any():
        raise AdversarialValidationError("domain entity OOF contains duplicates")
    return grouped


def _metrics(frame: pd.DataFrame) -> DomainMetrics:
    labels = frame["domain_label"].to_numpy(dtype=np.int8)
    probability = frame["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise AdversarialValidationError("domain probabilities must be finite within [0, 1]")
    prediction = (probability >= 0.5).astype(np.int8)
    return DomainMetrics(
        roc_auc=float(roc_auc_score(labels, probability)),
        accuracy=float(accuracy_score(labels, prediction)),
        log_loss=float(log_loss(labels, probability, labels=[0, 1])),
        brier_score=float(brier_score_loss(labels, probability)),
        predicted_test_rate=float(prediction.mean()),
        entity_count=int(frame.shape[0]),
    )


def run_adversarial_validation(
    dataset: DomainDataset,
    model: DomainModelConfig,
    *,
    seed: int,
    n_splits: int,
    cpu_threads: int,
) -> DomainValidationResult:
    """Run grouped OOF domain classification and aggregate one probability per entity."""

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.full(dataset.features.shape[0], np.nan, dtype=np.float64)
    fold_assignments = np.full(dataset.features.shape[0], -1, dtype=np.int16)
    importances: list[pd.DataFrame] = []
    fold_records: list[dict[str, float | int]] = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(dataset.features, dataset.labels, dataset.groups)
    ):
        train_groups = set(dataset.groups[train_index].tolist())
        valid_groups = set(dataset.groups[valid_index].tolist())
        if train_groups & valid_groups:
            raise AdversarialValidationError("domain entity crossed train and validation folds")
        adapter = LightGBMAdapter(model.parameters, seed=seed + fold, cpu_threads=cpu_threads)
        train_weight = _fold_weights(dataset, train_index)
        valid_weight = _fold_weights(dataset, valid_index)
        metadata = adapter.fit(
            dataset.features.iloc[train_index],
            dataset.labels[train_index],
            sample_weight=train_weight,
            validation_features=dataset.features.iloc[valid_index],
            validation_labels=dataset.labels[valid_index],
            validation_weight=valid_weight,
            early_stopping_rounds=model.early_stopping_rounds,
        )
        probabilities[valid_index] = adapter.predict_proba(dataset.features.iloc[valid_index])
        fold_assignments[valid_index] = fold
        importance = adapter.get_feature_importance()
        importance.insert(0, "fold", fold)
        importances.append(importance)
        fold_window = pd.DataFrame(
            {
                "source": dataset.metadata.iloc[valid_index]["source"].to_numpy(),
                "entity_id": dataset.entity_ids[valid_index],
                "group_id": dataset.groups[valid_index],
                "domain_label": dataset.labels[valid_index],
                "probability": probabilities[valid_index],
            }
        )
        fold_entity = _entity_oof(fold_window)
        fold_metrics = _metrics(fold_entity)
        fold_records.append(
            {
                "fold": fold,
                "roc_auc": fold_metrics.roc_auc,
                "accuracy": fold_metrics.accuracy,
                "log_loss": fold_metrics.log_loss,
                "brier_score": fold_metrics.brier_score,
                "entity_count": fold_metrics.entity_count,
                "best_iteration": metadata.best_iteration,
            }
        )
    if np.isnan(probabilities).any() or (fold_assignments < 0).any():
        raise AdversarialValidationError("domain OOF coverage is incomplete")
    window_oof = dataset.metadata.loc[
        :, ["source", "entity_id", "group_id", "window_id", "domain_label"]
    ].copy()
    window_oof["fold"] = fold_assignments
    window_oof["probability"] = probabilities
    entity_oof = _entity_oof(window_oof)
    metrics = _metrics(entity_oof)
    train_scores = entity_oof.loc[
        entity_oof["source"].eq("train"), ["entity_id", "probability"]
    ].rename(columns={"entity_id": "original_id", "probability": "similarity_score"})
    train_scores["is_oof"] = True
    train_scores = train_scores.sort_values("original_id", kind="stable", ignore_index=True)
    digest = hashlib.sha256()
    digest.update(dataset.fingerprint.encode())
    digest.update(np.ascontiguousarray(probabilities).tobytes())
    return DomainValidationResult(
        representation=dataset.representation,
        metrics=metrics,
        window_oof=window_oof,
        entity_oof=entity_oof,
        train_similarity_scores=train_scores,
        feature_importance=pd.concat(importances, ignore_index=True),
        fold_metrics=pd.DataFrame.from_records(fold_records),
        fingerprint=digest.hexdigest(),
    )
