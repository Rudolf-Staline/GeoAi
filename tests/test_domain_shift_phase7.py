from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from geoai_aquaculture.domain_shift import (
    DomainDataset,
    DomainModelConfig,
    DomainShiftConfigError,
    build_importance_weights,
    feature_shift_table,
    grouped_feature_importance,
    load_phase7_config,
    run_adversarial_validation,
)
from geoai_aquaculture.features import FeatureDefinition, FeatureRegistry


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        (
            FeatureDefinition(
                name="optical__shifted__mean",
                feature_group="optical_index",
                source_bands=("nir", "swir1"),
                formula="synthetic shifted feature",
                temporal_aggregation="mean",
                validity_rule="finite synthetic value",
                expected_dtype="float64",
                feature_kind="aggregate",
                output_representation="tabular",
                version="test",
            ),
            FeatureDefinition(
                name="radar__noise__mean",
                feature_group="radar_raw",
                source_bands=("VV",),
                formula="synthetic noise feature",
                temporal_aggregation="mean",
                validity_rule="finite synthetic value",
                expected_dtype="float64",
                feature_kind="aggregate",
                output_representation="tabular",
                version="test",
            ),
        )
    )


def _domain_dataset() -> DomainDataset:
    rng = np.random.default_rng(42)
    train_entities = [f"tr_{index:02d}" for index in range(18)]
    test_entities = [f"te_{index:02d}" for index in range(18)]
    train_ids = np.repeat(train_entities, 2)
    test_ids = np.asarray(test_entities)
    entity_ids = np.concatenate((train_ids, test_ids))
    groups = np.concatenate(
        (
            np.asarray([f"train:{value}" for value in train_ids]),
            np.asarray([f"test:{value}" for value in test_ids]),
        )
    )
    labels = np.concatenate(
        (np.zeros(train_ids.size, dtype=np.int8), np.ones(test_ids.size, dtype=np.int8))
    )
    shifted = np.concatenate(
        (
            rng.normal(-2.0, 0.25, train_ids.size),
            rng.normal(2.0, 0.25, test_ids.size),
        )
    )
    noise = rng.normal(0.0, 1.0, labels.size)
    features = pd.DataFrame(
        {
            "optical__shifted__mean": shifted,
            "radar__noise__mean": noise,
        }
    )
    metadata = pd.DataFrame(
        {
            "source": np.where(labels == 0, "train", "test"),
            "entity_id": entity_ids,
            "group_id": groups,
            "window_id": [f"w_{index}" for index in range(labels.size)],
            "domain_label": labels,
            "raw_id_exposed_as_feature": False,
        }
    )
    counts = pd.Series(groups).value_counts()
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    return DomainDataset(
        representation="full",
        features=features,
        feature_names=tuple(features.columns),
        registry=_registry(),
        metadata=metadata,
        labels=labels,
        groups=groups,
        entity_ids=entity_ids,
        sample_weights=weights,
        feature_groups=MappingProxyType(
            {
                "optical_index": ("optical__shifted__mean",),
                "radar_raw": ("radar__noise__mean",),
            }
        ),
        schema_fingerprint="schema",
        fingerprint="dataset",
    )


def test_phase7_config_is_deterministic_and_rejects_one_seed(tmp_path: Path) -> None:
    source = Path("configs/experiments/phase7_domain_shift.yaml")
    first = load_phase7_config(source)
    second = load_phase7_config(source)
    assert first.fingerprint == second.fingerprint
    invalid = tmp_path / "phase7.yaml"
    invalid.write_text(
        source.read_text().replace("adaptation_seeds: [7201, 17208]", "adaptation_seeds: [7201]"),
        encoding="utf-8",
    )
    with pytest.raises(DomainShiftConfigError, match="at least two seeds"):
        load_phase7_config(invalid)


def test_grouped_adversarial_validation_is_entity_oof_and_detects_shift() -> None:
    dataset = _domain_dataset()
    result = run_adversarial_validation(
        dataset,
        DomainModelConfig(
            parameters={
                "objective": "binary",
                "n_estimators": 80,
                "learning_rate": 0.08,
                "num_leaves": 7,
                "min_child_samples": 3,
                "reg_lambda": 1.0,
            },
            early_stopping_rounds=15,
        ),
        seed=123,
        n_splits=3,
        cpu_threads=1,
    )
    assert result.metrics.roc_auc > 0.98
    assert result.entity_oof.shape[0] == 36
    assert result.window_oof["fold"].ge(0).all()
    assert result.train_similarity_scores.shape[0] == 18
    assert result.train_similarity_scores["is_oof"].all()
    assert result.entity_oof["group_id"].is_unique


def test_domain_importance_and_shift_metrics_identify_shifted_feature() -> None:
    dataset = _domain_dataset()
    result = run_adversarial_validation(
        dataset,
        DomainModelConfig(
            parameters={
                "objective": "binary",
                "n_estimators": 60,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 3,
            },
            early_stopping_rounds=10,
        ),
        seed=77,
        n_splits=3,
        cpu_threads=1,
    )
    importance, groups = grouped_feature_importance(result, dataset.registry)
    assert importance.iloc[0]["feature"] == "optical__shifted__mean"
    assert groups.iloc[0]["feature_group"] == "optical_index"
    shift = feature_shift_table(dataset, importance, feature_limit=2)
    shifted = shift.set_index("feature").loc["optical__shifted__mean"]
    assert shifted["ks_statistic"] > 0.9
    assert shifted["psi"] > 1.0


def test_importance_weights_are_clipped_normalized_and_oof_only() -> None:
    scores = pd.DataFrame(
        {
            "original_id": ["a", "b", "c", "d"],
            "similarity_score": [0.001, 0.2, 0.8, 0.999],
            "is_oof": [True, True, True, True],
        }
    )
    weights = build_importance_weights(scores, minimum=0.5, maximum=2.0)
    assert weights["importance_weight"].between(0.5, 2.0).all()
    assert weights["importance_weight"].gt(0).all()
    assert weights["was_clipped_low"].any()
    assert weights["was_clipped_high"].any()
    scores.loc[0, "is_oof"] = False
    with pytest.raises(Exception, match="one OOF score"):
        build_importance_weights(scores, minimum=0.5, maximum=2.0)


def test_domain_dataset_rejects_raw_id_feature_flag() -> None:
    dataset = _domain_dataset()
    metadata = dataset.metadata.copy()
    metadata.loc[0, "raw_id_exposed_as_feature"] = True
    with pytest.raises(Exception, match="Raw IDs|raw IDs"):
        DomainDataset(
            representation=dataset.representation,
            features=dataset.features,
            feature_names=dataset.feature_names,
            registry=dataset.registry,
            metadata=metadata,
            labels=dataset.labels,
            groups=dataset.groups,
            entity_ids=dataset.entity_ids,
            sample_weights=dataset.sample_weights,
            feature_groups=dataset.feature_groups,
            schema_fingerprint=dataset.schema_fingerprint,
            fingerprint=dataset.fingerprint,
        )
