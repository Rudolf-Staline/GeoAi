"""Aligned train/test domain datasets built from approved Phase 3 features."""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from geoai_aquaculture.data import (
    ProjectConfig,
    extract_test_mask_library,
    load_competition_data,
    materialize_test_windows,
)
from geoai_aquaculture.features import (
    FeatureMatrix,
    FeatureRegistry,
    FeatureSetName,
    SelectedFeatureMatrix,
    build_tabular_features,
    select_tabular_features,
)
from geoai_aquaculture.training.tabular import validate_phase3_feature_contract
from geoai_aquaculture.validation import (
    FoldManifest,
    build_validation_windows,
    load_fold_manifest,
    load_validation_window_manifest,
)


class DomainDatasetError(ValueError):
    """Raised when train/test domain panels are not comparable or leakage-safe."""


@dataclass(frozen=True, slots=True)
class DomainFeaturePanels:
    """Full approved feature matrices for masked train and observed test windows."""

    train: FeatureMatrix
    test: FeatureMatrix
    train_manifest: pd.DataFrame
    folds: FoldManifest

    def __post_init__(self) -> None:
        if self.train.feature_names != self.test.feature_names:
            raise DomainDatasetError("train/test full feature names do not align")
        if self.train.schema_fingerprint != self.test.schema_fingerprint:
            raise DomainDatasetError("train/test full feature schemas do not align")
        if self.train.features.shape[0] != self.train_manifest.shape[0]:
            raise DomainDatasetError("train feature rows and fixed window metadata are misaligned")


@dataclass(frozen=True, slots=True)
class DomainDataset:
    """One registry-backed train-vs-test classification dataset."""

    representation: FeatureSetName
    features: pd.DataFrame
    feature_names: tuple[str, ...]
    registry: FeatureRegistry
    metadata: pd.DataFrame
    labels: np.ndarray
    groups: np.ndarray
    entity_ids: np.ndarray
    sample_weights: np.ndarray
    feature_groups: MappingProxyType[str, tuple[str, ...]]
    schema_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        rows = self.features.shape[0]
        aligned = (self.labels, self.groups, self.entity_ids, self.sample_weights)
        if any(len(values) != rows for values in aligned):
            raise DomainDatasetError("domain features and metadata are misaligned")
        if tuple(self.features.columns) != self.feature_names:
            raise DomainDatasetError("domain feature columns changed ordering")
        if set(np.unique(self.labels).tolist()) != {0, 1}:
            raise DomainDatasetError("domain dataset must contain train and test rows")
        if not np.isfinite(self.sample_weights).all() or (self.sample_weights <= 0.0).any():
            raise DomainDatasetError("domain sample weights must be finite and positive")
        if np.isinf(self.features.to_numpy(dtype=np.float64)).any():
            raise DomainDatasetError("domain features cannot contain infinity")
        if self.metadata["raw_id_exposed_as_feature"].any():
            raise DomainDatasetError("raw IDs cannot enter the domain model feature matrix")


def build_domain_feature_panels(project: ProjectConfig) -> DomainFeaturePanels:
    """Rebuild the approved masked train panel and one observed window per test row."""

    runtime = project.tabular
    validation_dir = runtime.validation_artifacts_dir
    folds = load_fold_manifest(
        validation_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=runtime.fold_manifest_fingerprint,
    )
    persisted_windows = load_validation_window_manifest(
        validation_dir / "validation_window_manifest.csv",
        folds,
        project.validation,
        expected_fingerprint=runtime.validation_window_fingerprint,
    )
    data = load_competition_data(project)
    masks = extract_test_mask_library(data)
    rebuilt = build_validation_windows(
        data,
        folds,
        project.validation,
        mask_library=masks,
        expected_fingerprint=runtime.validation_window_fingerprint,
        retain_datasets=True,
    )
    if rebuilt.manifest.fingerprint != persisted_windows.fingerprint:
        raise DomainDatasetError("regenerated Phase 4 validation windows changed")
    train_windows = rebuilt.for_repeat(0)
    train_matrix = build_tabular_features(train_windows, project.features)
    validate_phase3_feature_contract(
        feature_count=train_matrix.features.shape[1],
        schema_fingerprint=train_matrix.schema_fingerprint,
        project=project,
    )
    test_windows = materialize_test_windows(data)
    test_matrix = build_tabular_features(test_windows, project.features)
    if test_matrix.feature_names != train_matrix.feature_names:
        raise DomainDatasetError("observed test features do not align with approved train features")
    train_manifest = persisted_windows.frame.loc[
        persisted_windows.frame["repeat"].eq(0)
    ].reset_index(drop=True)
    del data, masks, rebuilt, train_windows, test_windows
    gc.collect()
    return DomainFeaturePanels(
        train=train_matrix,
        test=test_matrix,
        train_manifest=train_manifest,
        folds=folds,
    )


def _selected(
    panels: DomainFeaturePanels,
    representation: FeatureSetName,
    project: ProjectConfig,
) -> tuple[SelectedFeatureMatrix, SelectedFeatureMatrix]:
    train = select_tabular_features(panels.train, representation, project.features.bands)
    test = select_tabular_features(panels.test, representation, project.features.bands)
    if (
        train.feature_names != test.feature_names
        or train.schema_fingerprint != test.schema_fingerprint
    ):
        raise DomainDatasetError(f"{representation} train/test selected schemas differ")
    return train, test


def _balanced_entity_weights(groups: np.ndarray, labels: np.ndarray) -> np.ndarray:
    group_counts = pd.Series(groups).value_counts()
    weights = np.asarray([1.0 / float(group_counts[group]) for group in groups], dtype=np.float64)
    entity_count = int(pd.Series(groups).nunique())
    for domain in (0, 1):
        selector = labels == domain
        total = float(weights[selector].sum())
        if total <= 0.0:
            raise DomainDatasetError("domain class has zero entity weight")
        weights[selector] *= (entity_count / 2.0) / total
    return weights


def build_domain_dataset(
    panels: DomainFeaturePanels,
    project: ProjectConfig,
    representation: FeatureSetName,
) -> DomainDataset:
    """Combine train/test features without exposing IDs as model inputs."""

    train, test = _selected(panels, representation, project)
    train_ids = panels.train_manifest["original_id"].astype("string").to_numpy(dtype=str)
    test_ids = np.asarray(panels.test.original_ids, dtype=str)
    train_groups = np.char.add("train:", train_ids)
    test_groups = np.char.add("test:", test_ids)
    groups = np.concatenate((train_groups, test_groups))
    entity_ids = np.concatenate((train_ids, test_ids))
    labels = np.concatenate(
        (
            np.zeros(train.features.shape[0], dtype=np.int8),
            np.ones(test.features.shape[0], dtype=np.int8),
        )
    )
    features = pd.concat((train.features, test.features), ignore_index=True)
    source = np.where(labels == 0, "train", "test")
    window_ids = np.concatenate(
        (panels.train.window_ids.astype(str), panels.test.window_ids.astype(str))
    )
    metadata = pd.DataFrame(
        {
            "source": source,
            "entity_id": entity_ids,
            "group_id": groups,
            "window_id": window_ids,
            "domain_label": labels,
            "raw_id_exposed_as_feature": False,
        }
    )
    sample_weights = _balanced_entity_weights(groups, labels)
    digest = hashlib.sha256()
    digest.update(representation.encode())
    digest.update(train.schema_fingerprint.encode())
    digest.update("\x1f".join(groups.tolist()).encode())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return DomainDataset(
        representation=representation,
        features=features,
        feature_names=train.feature_names,
        registry=train.registry,
        metadata=metadata,
        labels=labels,
        groups=groups,
        entity_ids=entity_ids,
        sample_weights=sample_weights,
        feature_groups=train.feature_groups,
        schema_fingerprint=train.schema_fingerprint,
        fingerprint=digest.hexdigest(),
    )
