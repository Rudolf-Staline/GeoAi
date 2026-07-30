from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geoai_aquaculture.features import (  # noqa: E402
    FeatureDefinition,
    FeatureRegistry,
    SequenceFeatureDataset,
)
from geoai_aquaculture.models.temporal import (  # noqa: E402
    SensorGatedGRU,
    TemporalArchitecture,
    TemporalModelError,
    count_trainable_parameters,
    masked_mean_pool,
)
from geoai_aquaculture.training.temporal import (  # noqa: E402
    SameOriginalPairMap,
    SequenceNormalizer,
)
from geoai_aquaculture.training.temporal_config import (  # noqa: E402
    TemporalExperimentConfigError,
    load_temporal_experiment_config,
)


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        definitions=(
            FeatureDefinition(
                name="sequence__dummy",
                feature_group="sequence_dummy",
                source_bands=(),
                formula="synthetic fixture",
                temporal_aggregation=None,
                validity_rule="synthetic",
                expected_dtype="float64",
                feature_kind="monthly",
                output_representation="sequence",
                version="test",
            ),
        )
    )


def _sequence() -> SequenceFeatureDataset:
    rows, time = 8, 6
    lengths = np.asarray([4, 4, 5, 5, 6, 6, 4, 4], dtype=np.int8)
    padding = np.arange(time)[None, :] >= lengths[:, None]
    position = ~padding
    radar_mask = position.copy()
    optical_mask = position.copy()
    optical_mask[1, 2] = False
    optical_mask[5, 4] = False
    rng = np.random.default_rng(17)

    def values(channels: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature_mask = np.repeat(mask[:, :, None], channels, axis=2)
        array = rng.normal(size=(rows, time, channels))
        array[~feature_mask] = np.nan
        return array, feature_mask

    radar, radar_features = values(8, radar_mask)
    optical, optical_features = values(10, optical_mask)
    indices, index_mask = values(14, optical_mask)
    raw_band_mask = np.concatenate((radar_features[:, :, :2], optical_features), axis=2)
    months = np.zeros((rows, time), dtype=np.int8)
    relative = np.zeros((rows, time), dtype=np.int8)
    month_encoding = np.full((rows, time, 2), np.nan, dtype=np.float64)
    for row, length in enumerate(lengths):
        months[row, :length] = np.arange(1, length + 1)
        relative[row, :length] = np.arange(1, length + 1)
        angle = 2.0 * np.pi * (months[row, :length] - 1) / 12.0
        month_encoding[row, :length, 0] = np.sin(angle)
        month_encoding[row, :length, 1] = np.cos(angle)
    original_ids = np.asarray([f"O{row // 2}" for row in range(rows)], dtype=object)
    return SequenceFeatureDataset(
        radar_values=radar,
        optical_values=optical,
        monthly_indices=indices,
        relative_positions=relative,
        calendar_months=months,
        absolute_month_encoding=month_encoding,
        radar_mask=radar_mask,
        radar_feature_mask=radar_features,
        optical_mask=optical_mask,
        optical_band_mask=optical_features,
        raw_band_mask=raw_band_mask,
        index_mask=index_mask,
        padding_mask=padding,
        radar_feature_names=tuple(f"r{i}" for i in range(8)),
        optical_feature_names=tuple(f"o{i}" for i in range(10)),
        index_feature_names=tuple(f"i{i}" for i in range(14)),
        raw_band_names=(
            "VH",
            "VV",
            "blue",
            "green",
            "nir",
            "nira",
            "re1",
            "re2",
            "re3",
            "red",
            "swir1",
            "swir2",
        ),
        original_ids=original_ids,
        window_ids=np.asarray([f"W{i}" for i in range(rows)], dtype=object),
        folds=np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int16),
        labels=np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8),
        registry=_registry(),
        fingerprint="fixture",
        schema_fingerprint="schema",
    )


def _manual_batch() -> dict[str, torch.Tensor]:
    batch, time = 2, 6
    padding = torch.tensor(
        [[False, False, False, False, True, True], [False] * 6], dtype=torch.bool
    )
    valid = ~padding
    generator = torch.Generator().manual_seed(3)
    values = {
        "radar_values": torch.randn(batch, time, 8, generator=generator),
        "optical_values": torch.randn(batch, time, 10, generator=generator),
        "index_values": torch.randn(batch, time, 14, generator=generator),
    }
    result: dict[str, torch.Tensor] = {
        **values,
        "radar_feature_mask": valid.unsqueeze(-1).expand(batch, time, 8).clone(),
        "optical_feature_mask": valid.unsqueeze(-1).expand(batch, time, 10).clone(),
        "index_mask": valid.unsqueeze(-1).expand(batch, time, 14).clone(),
        "radar_mask": valid.clone(),
        "optical_mask": valid.clone(),
        "padding_mask": padding,
        "relative_positions": torch.tensor(
            [[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 5, 6]], dtype=torch.float32
        )
        / 6.0,
        "month_encoding": torch.zeros(batch, time, 2),
    }
    for name in ("radar_values", "optical_values", "index_values"):
        result[name][padding.unsqueeze(-1).expand_as(result[name])] = 0.0
    return result


def test_masked_gru_is_compact_and_padding_invariant() -> None:
    architecture = TemporalArchitecture(8, 10, 14, hidden_dim=48)
    model = SensorGatedGRU(architecture).eval()
    assert count_trainable_parameters(model) < 300_000
    batch = _manual_batch()
    with torch.no_grad():
        baseline = model(batch).logits
        changed = {key: value.clone() for key, value in batch.items()}
        for name in ("radar_values", "optical_values", "index_values", "month_encoding"):
            mask = changed["padding_mask"].unsqueeze(-1).expand_as(changed[name])
            changed[name][mask] = 999.0
        result = model(changed).logits
    torch.testing.assert_close(baseline, result)


def test_sensor_gate_obeys_hard_availability() -> None:
    model = SensorGatedGRU(TemporalArchitecture(8, 10, 14)).eval()
    batch = _manual_batch()
    batch["optical_mask"][0, 1] = False
    batch["optical_feature_mask"][0, 1] = False
    batch["optical_values"][0, 1] = 0.0
    batch["radar_mask"][1, 2] = False
    batch["radar_feature_mask"][1, 2] = False
    batch["radar_values"][1, 2] = 0.0
    with torch.no_grad():
        gate = model(batch).optical_gate
    assert torch.equal(gate[0, 1], torch.zeros_like(gate[0, 1]))
    assert torch.equal(gate[1, 2], torch.ones_like(gate[1, 2]))
    assert torch.equal(gate[0, 4], torch.zeros_like(gate[0, 4]))


def test_masked_mean_pool_rejects_empty_sequences() -> None:
    values = torch.ones(2, 3, 4)
    padding = torch.tensor([[False, True, True], [True, True, True]])
    with pytest.raises(TemporalModelError, match="at least one"):
        masked_mean_pool(values, padding)


def test_fold_local_normalizer_and_pair_map_preserve_contracts() -> None:
    sequence = _sequence()
    selector = np.asarray([True, True, True, True, False, False, False, False])
    original_radar = sequence.radar_values.copy()
    normalizer = SequenceNormalizer.fit(sequence, selector, input_clip=6.0)
    tensors = normalizer.tensors(sequence, np.asarray([0, 1, 4]), device=torch.device("cpu"))
    assert torch.isfinite(tensors["radar_values"]).all()
    assert torch.equal(
        tensors["radar_values"][tensors["radar_feature_mask"] == 0],
        torch.zeros_like(tensors["radar_values"][tensors["radar_feature_mask"] == 0]),
    )
    np.testing.assert_equal(sequence.radar_values, original_radar)

    train_indices = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    pairs = SameOriginalPairMap.build(sequence.original_ids, train_indices)
    paired = pairs.paired_indices(train_indices, epoch=0)
    assert np.all(paired != train_indices)
    assert np.array_equal(sequence.original_ids[paired], sequence.original_ids[train_indices])


def test_temporal_config_is_bounded_and_threshold_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "experiment.yaml"
    source.write_text(
        """
experiment:
  id: TEST-SEQ
  hypothesis: compact test
  objective: bce
  consistency_lambda: 0.0
  seed: 7
  allowed_stages: [smoke]
  threshold: 0.5
  architecture:
    hidden_dim: 32
  training:
    max_epochs: 3
    smoke_epochs: 2
    batch_size: 4
    learning_rate: 0.001
    weight_decay: 0.0001
    patience: 1
    gradient_clip: 1.0
    input_clip: 6.0
    cpu_threads: 1
  viability: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_temporal_experiment_config(source)
    assert config.threshold == 0.5
    assert config.training.max_epochs == 3
    source.write_text(source.read_text().replace("threshold: 0.5", "threshold: 0.4"))
    with pytest.raises(TemporalExperimentConfigError, match=r"exactly 0\.5"):
        load_temporal_experiment_config(source)


def test_pairwise_temporal_tree_summary_tracks_complementary_errors() -> None:
    import pandas as pd

    from geoai_aquaculture.training.temporal_diversity import pairwise_oof_summary

    keys = {
        "original_id": ["a", "b", "c", "d"],
        "repeat": [0, 0, 0, 0],
        "fold": [0, 0, 1, 1],
        "label": [0, 1, 1, 0],
    }
    temporal = pd.DataFrame(
        {
            **keys,
            "probability": [0.1, 0.8, 0.4, 0.2],
            "prediction": [0, 1, 0, 0],
        }
    )
    tree = pd.DataFrame(
        {
            **keys,
            "probability": [0.2, 0.4, 0.9, 0.3],
            "prediction": [0, 0, 1, 0],
        }
    )
    result = pairwise_oof_summary(temporal, tree)
    assert result["row_count"] == 4
    assert result["binary_disagreement_rate"] == 0.5
    assert result["temporal_only_correct"] == 1
    assert result["tree_only_correct"] == 1
    assert result["shared_error_count"] == 0
