"""Leakage-safe compact temporal-model viability runner for Phase 6."""

from __future__ import annotations

import gc
import json
import random
import resource
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import log_loss
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from geoai_aquaculture.data import (
    ProjectConfig,
    extract_test_mask_library,
    git_provenance,
    load_competition_data,
)
from geoai_aquaculture.features import SequenceFeatureDataset, build_sequence_features
from geoai_aquaculture.models.temporal import (
    SensorGatedGRU,
    TemporalArchitecture,
    architecture_from_dict,
    count_trainable_parameters,
)
from geoai_aquaculture.validation import (
    ORIGINAL_OOF_COLUMNS,
    FoldManifest,
    OOFPredictions,
    ValidationReport,
    ValidationWindowManifest,
    aggregate_window_predictions,
    build_oof_predictions,
    build_validation_report,
    build_validation_windows,
    dataframe_fingerprint,
    load_fold_manifest,
    load_validation_window_manifest,
    make_window_prediction_frame,
)

from .artifacts import (
    ExperimentArtifactError,
    experiment_artifact_dir,
    prepare_experiment_artifact_dir,
    sha256_file,
)
from .config import ExperimentStage
from .tabular import stage_repeat_folds, validate_full_oof_contract
from .temporal_config import TemporalExperimentConfig


class TemporalTrainingError(ValueError):
    """Raised when temporal training violates immutable scientific contracts."""


@dataclass(frozen=True, slots=True)
class ChannelStatistics:
    """Fold-local mean and scale for one masked channel group."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise TemporalTrainingError("channel statistics must be aligned one-dimensional arrays")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise TemporalTrainingError("channel statistics must be finite")
        if np.any(self.scale <= 0.0):
            raise TemporalTrainingError("channel scales must be positive")


@dataclass(frozen=True, slots=True)
class SequenceNormalizer:
    """Fold-local masked normalizer; invalid and padded entries remain zero."""

    radar: ChannelStatistics
    optical: ChannelStatistics
    indices: ChannelStatistics
    input_clip: float

    @staticmethod
    def _fit_group(values: np.ndarray, mask: np.ndarray, selector: np.ndarray) -> ChannelStatistics:
        selected_values = values[selector]
        selected_mask = mask[selector]
        channels = values.shape[2]
        mean = np.zeros(channels, dtype=np.float64)
        scale = np.ones(channels, dtype=np.float64)
        for channel in range(channels):
            valid = selected_values[:, :, channel][selected_mask[:, :, channel]]
            if valid.size == 0:
                raise TemporalTrainingError(
                    f"training fold has no valid values for channel {channel}"
                )
            mean[channel] = float(np.mean(valid))
            std = float(np.std(valid, ddof=0))
            scale[channel] = std if np.isfinite(std) and std > 1e-8 else 1.0
        return ChannelStatistics(mean=mean, scale=scale)

    @classmethod
    def fit(
        cls,
        sequence: SequenceFeatureDataset,
        selector: np.ndarray,
        *,
        input_clip: float,
    ) -> SequenceNormalizer:
        selector = np.asarray(selector, dtype=bool)
        if selector.shape != (len(sequence.original_ids),) or not selector.any():
            raise TemporalTrainingError("normalizer selector is empty or misaligned")
        return cls(
            radar=cls._fit_group(sequence.radar_values, sequence.radar_feature_mask, selector),
            optical=cls._fit_group(
                sequence.optical_values, sequence.optical_band_mask, selector
            ),
            indices=cls._fit_group(sequence.monthly_indices, sequence.index_mask, selector),
            input_clip=float(input_clip),
        )

    def _transform_group(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        statistics: ChannelStatistics,
        indices: np.ndarray,
    ) -> np.ndarray:
        selected = values[indices]
        selected_mask = mask[indices]
        normalized = np.zeros(selected.shape, dtype=np.float32)
        centered = (selected - statistics.mean.reshape(1, 1, -1)) / statistics.scale.reshape(
            1, 1, -1
        )
        centered = np.clip(centered, -self.input_clip, self.input_clip)
        normalized[selected_mask] = centered[selected_mask].astype(np.float32)
        if not np.isfinite(normalized).all():
            raise TemporalTrainingError("fold-normalized temporal values must be finite")
        return normalized

    def tensors(
        self,
        sequence: SequenceFeatureDataset,
        indices: np.ndarray,
        *,
        device: torch.device,
    ) -> dict[str, Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        month = np.nan_to_num(sequence.absolute_month_encoding[indices], nan=0.0).astype(np.float32)
        relative = sequence.relative_positions[indices].astype(np.float32) / 6.0
        return {
            "radar_values": torch.from_numpy(
                self._transform_group(
                    sequence.radar_values,
                    sequence.radar_feature_mask,
                    self.radar,
                    indices,
                )
            ).to(device),
            "optical_values": torch.from_numpy(
                self._transform_group(
                    sequence.optical_values,
                    sequence.optical_band_mask,
                    self.optical,
                    indices,
                )
            ).to(device),
            "index_values": torch.from_numpy(
                self._transform_group(
                    sequence.monthly_indices,
                    sequence.index_mask,
                    self.indices,
                    indices,
                )
            ).to(device),
            "radar_feature_mask": torch.from_numpy(
                sequence.radar_feature_mask[indices]
            ).to(device),
            "optical_feature_mask": torch.from_numpy(
                sequence.optical_band_mask[indices]
            ).to(device),
            "index_mask": torch.from_numpy(sequence.index_mask[indices]).to(device),
            "radar_mask": torch.from_numpy(sequence.radar_mask[indices]).to(device),
            "optical_mask": torch.from_numpy(sequence.optical_mask[indices]).to(device),
            "padding_mask": torch.from_numpy(sequence.padding_mask[indices]).to(device),
            "relative_positions": torch.from_numpy(relative).to(device),
            "month_encoding": torch.from_numpy(month).to(device),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "radar_mean": self.radar.mean.tolist(),
            "radar_scale": self.radar.scale.tolist(),
            "optical_mean": self.optical.mean.tolist(),
            "optical_scale": self.optical.scale.tolist(),
            "index_mean": self.indices.mean.tolist(),
            "index_scale": self.indices.scale.tolist(),
            "input_clip": self.input_clip,
        }


class _IndexDataset(Dataset[int]):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> int:
        return int(self.indices[item])


@dataclass(frozen=True, slots=True)
class SameOriginalPairMap:
    """Deterministic distinct-view pair lookup restricted to training originals."""

    groups: dict[str, np.ndarray]
    positions: dict[int, tuple[str, int]]

    @classmethod
    def build(cls, original_ids: np.ndarray, train_indices: np.ndarray) -> SameOriginalPairMap:
        frame = pd.DataFrame(
            {
                "index": np.asarray(train_indices, dtype=np.int64),
                "original_id": np.asarray(original_ids)[train_indices].astype(str),
            }
        )
        groups = {
            str(original_id): group["index"].to_numpy(dtype=np.int64)
            for original_id, group in frame.groupby("original_id", sort=True)
        }
        if any(indices.size < 2 for indices in groups.values()):
            raise TemporalTrainingError("consistency training requires two views per original")
        positions: dict[int, tuple[str, int]] = {}
        for original_id, indices in groups.items():
            for position, index in enumerate(indices.tolist()):
                positions[int(index)] = (original_id, position)
        return cls(groups=groups, positions=positions)

    def paired_indices(self, indices: np.ndarray, *, epoch: int) -> np.ndarray:
        result = np.empty(len(indices), dtype=np.int64)
        for row, index in enumerate(np.asarray(indices, dtype=np.int64)):
            original_id, position = self.positions[int(index)]
            candidates = self.groups[original_id]
            offset = 1 + (epoch % (candidates.size - 1))
            result[row] = candidates[(position + offset) % candidates.size]
            if result[row] == index:
                raise TemporalTrainingError("same-original consistency pair must be distinct")
        return result


@dataclass(frozen=True, slots=True)
class PreparedTemporalData:
    """One sequence panel aligned to immutable repeated manifests."""

    sequence: SequenceFeatureDataset
    folds: FoldManifest
    windows: ValidationWindowManifest
    rows_per_repeat: int

    def __post_init__(self) -> None:
        if len(self.sequence.original_ids) != self.rows_per_repeat:
            raise TemporalTrainingError("sequence panel row count is misaligned")
        if self.windows.frame.shape[0] != self.rows_per_repeat * self.folds.n_repeats:
            raise TemporalTrainingError("validation-window manifest has incomplete repeats")
        expected_channels = (
            len(self.sequence.radar_feature_names),
            len(self.sequence.optical_feature_names),
            len(self.sequence.index_feature_names),
        )
        if expected_channels != (8, 10, 14):
            raise TemporalTrainingError(
                f"Phase 3 sequence channel contract changed: {expected_channels} != (8, 10, 14)"
            )


@dataclass(frozen=True, slots=True)
class TemporalFoldResult:
    repeat: int
    fold: int
    seed: int
    train_original_count: int
    validation_original_count: int
    train_window_count: int
    validation_window_count: int
    best_epoch: int
    epochs_run: int
    validation_log_loss: float
    training_seconds: float
    inference_seconds: float
    checkpoint_path: str
    checkpoint_sha256: str
    parameter_count: int
    consistency_lambda: float


@dataclass(frozen=True, slots=True)
class TemporalTrainingResult:
    stage: ExperimentStage
    oof: OOFPredictions
    report: ValidationReport
    folds: tuple[TemporalFoldResult, ...]
    architecture: TemporalArchitecture
    sequence_schema_fingerprint: str
    runtime_seconds: float
    peak_rss_megabytes: float
    sensor_ablation: pd.DataFrame
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class _FoldOutput:
    metadata: TemporalFoldResult
    predictions: pd.DataFrame
    ablations: pd.DataFrame


def _manifest_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "original_id",
        "generation_mode",
        "view_index",
        "augmentation_seed",
        "window_start",
        "window_end",
        "window_length",
        "radar_availability",
        "optical_month_availability",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
        "mask_id",
    ]
    result = frame.loc[:, columns].copy()
    for column in result.select_dtypes(include=["category", "string"]).columns:
        result[column] = result[column].astype("string")
    return result.reset_index(drop=True)


def _validate_shared_repeat_panel(windows: ValidationWindowManifest, n_repeats: int) -> int:
    counts = windows.frame.groupby("repeat", observed=True).size()
    if set(counts.index.tolist()) != set(range(n_repeats)) or counts.nunique() != 1:
        raise TemporalTrainingError("validation repeats do not share a complete sequence panel")
    rows_per_repeat = int(counts.iloc[0])
    reference = _manifest_semantics(
        windows.frame.loc[windows.frame["repeat"].eq(0)].reset_index(drop=True)
    )
    for repeat in range(1, n_repeats):
        candidate = _manifest_semantics(
            windows.frame.loc[windows.frame["repeat"].eq(repeat)].reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
        except AssertionError as exc:
            raise TemporalTrainingError(
                "fixed sequence values differ between validation repeats"
            ) from exc
    return rows_per_repeat


def prepare_temporal_experiment_data(project: ProjectConfig) -> PreparedTemporalData:
    """Rebuild the exact Phase 3 sequence panel linked to Phase 4 manifests."""

    runtime = project.tabular
    folds = load_fold_manifest(
        runtime.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=runtime.fold_manifest_fingerprint,
    )
    windows = load_validation_window_manifest(
        runtime.validation_artifacts_dir / "validation_window_manifest.csv",
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
    rows_per_repeat = _validate_shared_repeat_panel(windows, folds.n_repeats)
    sequence = build_sequence_features(rebuilt.for_repeat(0), project.features)
    if sequence.schema_fingerprint == runtime.feature_schema_fingerprint:
        raise TemporalTrainingError("tabular and sequence schemas must have distinct fingerprints")
    del data, masks, rebuilt
    gc.collect()
    return PreparedTemporalData(
        sequence=sequence,
        folds=folds,
        windows=windows,
        rows_per_repeat=rows_per_repeat,
    )


def _set_seed(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)


def _architecture(
    experiment: TemporalExperimentConfig,
    sequence: SequenceFeatureDataset,
) -> TemporalArchitecture:
    architecture = architecture_from_dict(
        dict(experiment.architecture),
        radar_channels=len(sequence.radar_feature_names),
        optical_channels=len(sequence.optical_feature_names),
        index_channels=len(sequence.index_feature_names),
    )
    model = SensorGatedGRU(architecture)
    parameters = count_trainable_parameters(model)
    if parameters > 300_000:
        raise TemporalTrainingError(
            f"compact temporal model exceeds 300,000 parameters: {parameters}"
        )
    return architecture


def _collate_indices(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _predict_probabilities(
    model: SensorGatedGRU,
    sequence: SequenceFeatureDataset,
    normalizer: SequenceNormalizer,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    ablation: Literal["none", "radar", "optical", "indices"] = "none",
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    loader = DataLoader(
        _IndexDataset(indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_indices,
    )
    with torch.no_grad():
        for batch_indices in loader:
            batch = normalizer.tensors(sequence, batch_indices, device=device)
            output = model(
                batch,
                ablate_radar=ablation == "radar",
                ablate_optical=ablation == "optical",
                ablate_indices=ablation == "indices",
            )
            probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise TemporalTrainingError("temporal probabilities must be finite within [0, 1]")
    return result


def _original_validation_log_loss(manifest: pd.DataFrame, probabilities: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "original_id": manifest["original_id"].astype(str).to_numpy(),
            "label": manifest["label"].to_numpy(dtype=np.int8),
            "probability": probabilities,
        }
    )
    grouped = frame.groupby("original_id", sort=False).agg(
        label=("label", "first"), probability=("probability", "mean")
    )
    return float(log_loss(grouped["label"], grouped["probability"], labels=[0, 1]))


def _training_weights(original_ids: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(original_ids)[train_indices].astype(str)
    counts = pd.Series(values).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[value] for value in values], dtype=np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise TemporalTrainingError("equal-original temporal weights are invalid")
    return weights


def _train_fold_model(
    prepared: PreparedTemporalData,
    experiment: TemporalExperimentConfig,
    *,
    repeat: int,
    fold: int,
    stage: ExperimentStage,
    output_dir: Path,
) -> tuple[SensorGatedGRU, SequenceNormalizer, TemporalFoldResult, pd.DataFrame, pd.DataFrame]:
    sequence = prepared.sequence
    manifest = prepared.windows.frame.loc[
        prepared.windows.frame["repeat"].eq(repeat)
    ].reset_index(drop=True)
    valid_selector = manifest["fold"].to_numpy(dtype=np.int16) == fold
    train_selector = ~valid_selector
    train_indices = np.flatnonzero(train_selector)
    valid_indices = np.flatnonzero(valid_selector)
    train_originals = set(manifest.loc[train_selector, "original_id"].astype(str))
    valid_originals = set(manifest.loc[valid_selector, "original_id"].astype(str))
    if train_originals & valid_originals:
        raise TemporalTrainingError("an original appears in both temporal train and validation")
    fold_seed = experiment.seed + repeat * 10_007 + fold * 101
    _set_seed(fold_seed, experiment.training.cpu_threads)
    architecture = _architecture(experiment, sequence)
    model = SensorGatedGRU(architecture)
    parameter_count = count_trainable_parameters(model)
    device = torch.device("cpu")
    model.to(device)
    normalizer = SequenceNormalizer.fit(
        sequence,
        train_selector,
        input_clip=experiment.training.input_clip,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, experiment.training.patience // 3)
    )
    max_epochs = (
        experiment.training.smoke_epochs if stage == "smoke" else experiment.training.max_epochs
    )
    train_weights = _training_weights(sequence.original_ids, train_indices)
    weight_lookup = {
        int(index): float(weight)
        for index, weight in zip(train_indices, train_weights, strict=True)
    }
    pair_map = (
        SameOriginalPairMap.build(sequence.original_ids, train_indices)
        if experiment.objective == "bce_consistency"
        else None
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    training_started = perf_counter()
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        generator = torch.Generator().manual_seed(fold_seed + epoch)
        loader = DataLoader(
            _IndexDataset(train_indices),
            batch_size=experiment.training.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=_collate_indices,
        )
        model.train()
        for batch_indices in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = normalizer.tensors(sequence, batch_indices, device=device)
            labels = torch.from_numpy(
                manifest.loc[batch_indices, "label"].to_numpy(dtype=np.float32)
            ).to(device)
            weights = torch.tensor(
                [weight_lookup[int(index)] for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            output = model(batch)
            per_row = F.binary_cross_entropy_with_logits(output.logits, labels, reduction="none")
            loss = torch.sum(per_row * weights) / torch.sum(weights)
            if pair_map is not None:
                paired_indices = pair_map.paired_indices(batch_indices, epoch=epoch)
                paired_batch = normalizer.tensors(sequence, paired_indices, device=device)
                paired_output = model(paired_batch)
                consistency = F.mse_loss(
                    torch.sigmoid(output.logits), torch.sigmoid(paired_output.logits)
                )
                loss = loss + experiment.consistency_lambda * consistency
            if not torch.isfinite(loss):
                raise TemporalTrainingError("temporal training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), experiment.training.gradient_clip
            )
            optimizer.step()
        valid_probabilities = _predict_probabilities(
            model,
            sequence,
            normalizer,
            valid_indices,
            batch_size=experiment.training.batch_size,
            device=device,
        )
        validation_loss = _original_validation_log_loss(
            manifest.loc[valid_selector].reset_index(drop=True), valid_probabilities
        )
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if stage != "smoke" and epochs_without_improvement >= experiment.training.patience:
            break
    training_seconds = perf_counter() - training_started
    if best_state is None:
        raise TemporalTrainingError("temporal early stopping never produced a valid checkpoint")
    model.load_state_dict(best_state)
    inference_started = perf_counter()
    probabilities = _predict_probabilities(
        model,
        sequence,
        normalizer,
        valid_indices,
        batch_size=experiment.training.batch_size,
        device=device,
    )
    inference_seconds = perf_counter() - inference_started
    checkpoint_dir = output_dir / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"repeat_{repeat:02d}_fold_{fold:02d}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "architecture": asdict(architecture),
            "normalizer": normalizer.as_dict(),
            "experiment_id": experiment.experiment_id,
            "repeat": repeat,
            "fold": fold,
            "seed": fold_seed,
        },
        checkpoint_path,
    )
    valid_manifest = manifest.loc[valid_selector].reset_index(drop=True)
    predictions = make_window_prediction_frame(
        valid_manifest,
        probabilities,
        experiment_id=experiment.experiment_id,
        model_id=f"temporal:gru:{experiment.objective}",
        fold_manifest_fingerprint=prepared.folds.fingerprint,
        validation_window_fingerprint=prepared.windows.fingerprint,
    )
    ablation_records: list[dict[str, Any]] = []
    # Sensor ablations are diagnostics only and never alter the selected checkpoint.
    for ablation in ("radar", "optical", "indices"):
        ablated = _predict_probabilities(
            model,
            sequence,
            normalizer,
            valid_indices,
            batch_size=experiment.training.batch_size,
            device=device,
            ablation=ablation,
        )
        ablation_records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "ablation": ablation,
                "original_log_loss": _original_validation_log_loss(valid_manifest, ablated),
                "mean_probability": float(np.mean(ablated)),
            }
        )
    metadata = TemporalFoldResult(
        repeat=repeat,
        fold=fold,
        seed=fold_seed,
        train_original_count=len(train_originals),
        validation_original_count=len(valid_originals),
        train_window_count=int(train_selector.sum()),
        validation_window_count=int(valid_selector.sum()),
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        validation_log_loss=best_loss,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        checkpoint_path=checkpoint_path.relative_to(output_dir).as_posix(),
        checkpoint_sha256=sha256_file(checkpoint_path),
        parameter_count=parameter_count,
        consistency_lambda=experiment.consistency_lambda,
    )
    return model, normalizer, metadata, predictions, pd.DataFrame.from_records(ablation_records)


def _partial_oof(
    windows: pd.DataFrame,
    project: ProjectConfig,
    prepared: PreparedTemporalData,
) -> OOFPredictions:
    original = aggregate_window_predictions(
        windows,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )
    fingerprint = dataframe_fingerprint(original, columns=ORIGINAL_OOF_COLUMNS)
    return OOFPredictions(
        original=original,
        windows=windows,
        fold_manifest_fingerprint=prepared.folds.fingerprint,
        validation_window_fingerprint=prepared.windows.fingerprint,
        aggregation_method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
        fingerprint=fingerprint,
    )


def run_temporal_experiment(
    prepared: PreparedTemporalData,
    project: ProjectConfig,
    experiment: TemporalExperimentConfig,
    *,
    stage: ExperimentStage,
    output_dir: Path,
) -> TemporalTrainingResult:
    """Fit the fixed staged folds and build authoritative original-level OOF."""

    experiment.require_stage(stage)
    started = perf_counter()
    outputs: list[_FoldOutput] = []
    architecture = _architecture(experiment, prepared.sequence)
    for repeat, fold in stage_repeat_folds(
        stage,
        n_repeats=prepared.folds.n_repeats,
        n_splits=prepared.folds.n_splits,
    ):
        model, normalizer, metadata, predictions, ablations = _train_fold_model(
            prepared,
            experiment,
            repeat=repeat,
            fold=fold,
            stage=stage,
            output_dir=output_dir,
        )
        outputs.append(_FoldOutput(metadata, predictions, ablations))
        print(
            f"[temporal] repeat={repeat} fold={fold} "
            f"best_epoch={metadata.best_epoch} epochs={metadata.epochs_run} "
            f"val_logloss={metadata.validation_log_loss:.6f} "
            f"train_seconds={metadata.training_seconds:.2f}",
            flush=True,
        )
        del model, normalizer
        gc.collect()
    windows = pd.concat([output.predictions for output in outputs], ignore_index=True).sort_values(
        ["repeat", "fold", "original_id", "window_id"], kind="stable", ignore_index=True
    )
    if stage == "full":
        oof = build_oof_predictions(
            windows,
            prepared.folds,
            validation_window_fingerprint=prepared.windows.fingerprint,
            method=project.validation.aggregation_method,
            trimmed_fraction=project.validation.trimmed_mean_fraction,
        )
        validate_full_oof_contract(oof, project)
    else:
        oof = _partial_oof(windows, project, prepared)
    report = build_validation_report(oof, project.validation)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return TemporalTrainingResult(
        stage=stage,
        oof=oof,
        report=report,
        folds=tuple(output.metadata for output in outputs),
        architecture=architecture,
        sequence_schema_fingerprint=prepared.sequence.schema_fingerprint,
        runtime_seconds=perf_counter() - started,
        peak_rss_megabytes=peak_rss,
        sensor_ablation=pd.concat(
            [output.ablations for output in outputs], ignore_index=True
        ),
        artifact_dir=output_dir,
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _screening_decision(
    result: TemporalTrainingResult, experiment: TemporalExperimentConfig
) -> dict[str, Any]:
    robust = float(result.report.summary["robust_selection"]["score"])
    if result.stage == "screen":
        gap = experiment.viability.screening_reference_robust - robust
        promoted = gap <= experiment.viability.screening_max_gap
        return {
            "stage": "screen",
            "promoted_to_full": promoted,
            "robust_score": robust,
            "screening_reference_robust": experiment.viability.screening_reference_robust,
            "gap": gap,
            "maximum_allowed_gap": experiment.viability.screening_max_gap,
        }
    if result.stage == "full":
        within_tolerance = robust >= (
            experiment.viability.best_tree_robust_score
            - experiment.viability.full_score_tolerance
        )
        return {
            "stage": "full",
            "standalone_gate_passed": within_tolerance,
            "robust_score": robust,
            "best_tree_robust_score": experiment.viability.best_tree_robust_score,
            "full_score_tolerance": experiment.viability.full_score_tolerance,
            "blend_gate_pending_phase5_oof": True,
        }
    return {"stage": "smoke", "selection_eligible": False}


def write_temporal_experiment_artifacts(
    output_dir: Path,
    *,
    project: ProjectConfig,
    experiment: TemporalExperimentConfig,
    result: TemporalTrainingResult,
) -> None:
    """Persist complete ignored Phase 6 outputs after successful execution."""

    provenance = git_provenance(project.project_root)
    resolved = {
        **experiment.resolved_dict(),
        "stage": result.stage,
        "authoritative_validation": {
            "fold_manifest_fingerprint": result.oof.fold_manifest_fingerprint,
            "validation_window_fingerprint": result.oof.validation_window_fingerprint,
            "sequence_schema_fingerprint": result.sequence_schema_fingerprint,
            "aggregation_method": project.validation.aggregation_method,
            "threshold": project.validation.threshold,
        },
        "architecture_resolved": asdict(result.architecture),
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    metrics = {
        **result.report.summary,
        "stage": result.stage,
        "selection_eligible": result.stage == "full",
        "viability_decision": _screening_decision(result, experiment),
        "phase5_reference": {
            "best_tree_combined_score": 0.986222,
            "best_tree_robust_score": experiment.viability.best_tree_robust_score,
            "best_tree_worst_fold": 0.972592,
        },
    }
    _write_json(output_dir / "metrics.json", metrics)
    for name, frame in (
        ("fold_metrics", result.report.fold_metrics),
        ("repeat_metrics", result.report.repeat_metrics),
        ("slice_metrics", result.report.slice_metrics),
        ("prediction_stability", result.report.prediction_stability),
        ("oof_predictions", result.oof.original),
        ("window_predictions", result.oof.windows),
        ("sensor_ablation", result.sensor_ablation),
    ):
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
    fold_models = {
        "schema_version": 1,
        "model_family": "masked_gru",
        "parameter_count": result.folds[0].parameter_count,
        "folds": [asdict(fold) for fold in result.folds],
        "median_best_epoch": int(np.median([fold.best_epoch for fold in result.folds])),
    }
    _write_json(output_dir / "fold_models_manifest.json", fold_models)
    runtime = {
        "runtime_seconds": result.runtime_seconds,
        "peak_rss_megabytes": result.peak_rss_megabytes,
        "fold_training_seconds": float(sum(fold.training_seconds for fold in result.folds)),
        "fold_inference_seconds": float(sum(fold.inference_seconds for fold in result.folds)),
        "torch_version": torch.__version__,
        "device": "cpu",
        "cpu_threads": experiment.training.cpu_threads,
    }
    _write_json(output_dir / "runtime.json", runtime)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "stage": result.stage,
        "experiment_id": experiment.experiment_id,
        "experiment_config_fingerprint": experiment.fingerprint,
        "base_config_sha256": sha256_file(project.source_path),
        "git_commit": provenance["commit"],
        "tracked_files_dirty": provenance["tracked_files_dirty"],
        "fold_manifest_fingerprint": result.oof.fold_manifest_fingerprint,
        "validation_window_fingerprint": result.oof.validation_window_fingerprint,
        "sequence_schema_fingerprint": result.sequence_schema_fingerprint,
        "oof_fingerprint": result.oof.fingerprint,
        "original_oof_rows": result.oof.original.shape[0],
        "window_prediction_rows": result.oof.windows.shape[0],
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    official = result.report.summary["official_metric"]
    robust = result.report.summary["robust_selection"]
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                f"# {experiment.experiment_id}",
                "",
                f"Stage: `{result.stage}`",
                f"Objective: `{experiment.objective}`",
                f"Parameters: `{result.folds[0].parameter_count}`",
                "",
                f"- F1 at 0.5: {official['mean_f1']:.6f}",
                f"- ROC-AUC: {official['mean_roc_auc']:.6f}",
                f"- Combined: {official['mean_combined_score']:.6f}",
                f"- Robust: {robust['score']:.6f}",
                "",
                "The temporal branch is retained only through the predeclared viability gates.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def execute_temporal_experiment(
    project: ProjectConfig,
    experiment: TemporalExperimentConfig,
    *,
    stage: ExperimentStage,
    overwrite: bool = False,
    resume: bool = False,
) -> TemporalTrainingResult | None:
    """Prepare, run, and persist one explicit compact temporal experiment."""

    experiment.require_stage(stage)
    output = experiment_artifact_dir(
        project.tabular.experiments_artifacts_dir, experiment.experiment_id, stage
    )
    if resume:
        manifest_path = output / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise ExperimentArtifactError("temporal resume artifact is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "status": "complete",
            "stage": stage,
            "experiment_id": experiment.experiment_id,
            "experiment_config_fingerprint": experiment.fingerprint,
            "fold_manifest_fingerprint": project.tabular.fold_manifest_fingerprint,
            "validation_window_fingerprint": project.tabular.validation_window_fingerprint,
        }
        mismatch = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatch:
            raise ExperimentArtifactError(f"temporal resume mismatch: {mismatch}")
        return None
    output = prepare_experiment_artifact_dir(
        project.tabular.experiments_artifacts_dir,
        experiment.experiment_id,
        stage,
        overwrite=overwrite,
    )
    prepared = prepare_temporal_experiment_data(project)
    result = run_temporal_experiment(
        prepared,
        project,
        experiment,
        stage=stage,
        output_dir=output,
    )
    write_temporal_experiment_artifacts(
        output,
        project=project,
        experiment=experiment,
        result=result,
    )
    return result
