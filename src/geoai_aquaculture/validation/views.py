"""Fixed Phase 2 temporal views inherited by every future model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from geoai_aquaculture.data import (
    CompetitionData,
    MaskLibrary,
    TemporalWindowDataset,
    ValidationConfig,
    WindowGenerationConfig,
    generate_temporal_windows,
    stable_window_id,
)

from .common import (
    ValidationError,
    dataframe_fingerprint,
    json_fingerprint,
    release_temporary_memory,
)
from .folds import FoldManifest

WINDOW_FINGERPRINT_COLUMNS = (
    "repeat",
    "window_id",
    "original_id",
    "fold",
    "label",
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
)


@dataclass(frozen=True, slots=True)
class ValidationWindowManifest:
    """Model-independent metadata for the fixed validation temporal views."""

    frame: pd.DataFrame
    mode: str
    seed: int
    fold_manifest_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        validate_validation_window_manifest(self)


@dataclass(frozen=True, slots=True)
class ValidationWindowSet:
    """Per-repeat temporal tensors plus their shared immutable manifest."""

    dataset_template: TemporalWindowDataset | None
    manifest: ValidationWindowManifest
    n_repeats: int

    def for_repeat(self, repeat: int) -> TemporalWindowDataset:
        """Return fixed views for one repeat."""

        if self.dataset_template is None:
            raise ValidationError("validation tensors were not retained by this metadata-only view")
        if repeat < 0 or repeat >= self.n_repeats:
            raise ValidationError("requested validation-window repeat is unavailable")
        repeated_manifest = self.manifest.frame.loc[
            self.manifest.frame["repeat"].eq(repeat)
        ].reset_index(drop=True)
        if repeated_manifest.shape[0] != self.dataset_template.n_windows:
            raise ValidationError("repeat metadata does not align with retained validation tensors")
        return replace(self.dataset_template, manifest=repeated_manifest)


def _generation_config(config: ValidationConfig, *, exhaustive: bool) -> WindowGenerationConfig:
    return WindowGenerationConfig(
        enabled=True,
        use_test_missingness_masks=not exhaustive,
        exhaustive_windows=exhaustive,
        windows_per_sample=config.sampled_windows_per_original,
        temporal_dropout_enabled=False,
        temporal_dropout_probability=0.0,
        optical_dropout_enabled=False,
        optical_dropout_probability=0.0,
    )


def _compact_manifest(
    dataset: TemporalWindowDataset,
    fold_manifest: pd.DataFrame,
    *,
    repeat: int,
) -> pd.DataFrame:
    columns = [column for column in WINDOW_FINGERPRINT_COLUMNS if column != "repeat"]
    frame = dataset.manifest.loc[:, columns].copy(deep=True)
    frame.insert(0, "repeat", np.int16(repeat))
    replacement = fold_manifest.set_index(fold_manifest["original_id"].astype("string"))
    row_ids = frame["original_id"].astype("string")
    frame["fold"] = row_ids.map(replacement["fold"]).to_numpy(dtype=np.int16)
    frame["label"] = row_ids.map(replacement["label"]).to_numpy()
    source = dataset.manifest
    identity_inputs = zip(
        frame["original_id"],
        frame["fold"],
        frame["generation_mode"],
        frame["augmentation_seed"],
        frame["view_index"],
        frame["window_start"],
        frame["window_length"],
        frame["mask_id"],
        frame["radar_availability"],
        source["optical_band_availability"],
        strict=True,
    )
    window_ids: list[str] = []
    for values in identity_inputs:
        (
            original_id,
            fold,
            generation_mode,
            augmentation_seed,
            view_index,
            window_start,
            window_length,
            mask_id,
            radar_pattern,
            optical_pattern,
        ) = values
        window_ids.append(
            stable_window_id(
                original_id=str(original_id),
                fold=int(fold),
                generation_mode=str(generation_mode),
                augmentation_seed=int(augmentation_seed),
                view_index=int(view_index),
                window_start=int(window_start),
                window_length=int(window_length),
                mask_id=str(mask_id),
                radar_pattern=str(radar_pattern),
                optical_pattern=str(optical_pattern),
            )
        )
    frame["window_id"] = window_ids
    frame["original_id"] = frame["original_id"].astype("category")
    for column in (
        "label",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
    ):
        frame[column] = frame[column].astype(np.int8)
    frame["view_index"] = frame["view_index"].astype(np.int16)
    frame["fold"] = frame["fold"].astype(np.int16)
    for column in (
        "generation_mode",
        "radar_availability",
        "optical_month_availability",
        "mask_id",
    ):
        frame[column] = frame[column].astype("category")
    return frame


def build_validation_windows(
    data: CompetitionData,
    folds: FoldManifest,
    config: ValidationConfig,
    *,
    mask_library: MaskLibrary | None,
    exhaustive: bool = False,
    expected_fingerprint: str | None = None,
    retain_datasets: bool = True,
) -> ValidationWindowSet:
    """Generate fixed views only after repeated original-row folds exist."""

    if data.train.shape[0] != folds.n_originals:
        raise ValidationError("fold manifest and competition train rows are misaligned")
    if not exhaustive and mask_library is None:
        raise ValidationError("primary masked validation requires the test mask-pattern library")
    generation = _generation_config(config, exhaustive=exhaustive)
    manifests: list[pd.DataFrame] = []
    base = generate_temporal_windows(
        data,
        folds.for_repeat(0),
        generation,
        seed=config.validation_window_seed,
        mask_library=None if exhaustive else mask_library,
    )
    for repeat in range(folds.n_repeats):
        manifests.append(
            _compact_manifest(
                base,
                folds.for_repeat(repeat),
                repeat=repeat,
            )
        )
    template_manifest = manifests[0].loc[:, ["window_id", "original_id", "fold", "label"]].copy()
    template = replace(base, manifest=template_manifest) if retain_datasets else None
    del base
    release_temporary_memory()
    combined = pd.concat(manifests, ignore_index=True)
    mode = "exhaustive" if exhaustive else config.primary_window_mode
    content_fingerprint = dataframe_fingerprint(combined, columns=WINDOW_FINGERPRINT_COLUMNS)
    fingerprint = json_fingerprint(
        {
            "content": content_fingerprint,
            "fold_manifest": folds.fingerprint,
            "mode": mode,
            "seed": config.validation_window_seed,
        }
    )
    manifest = ValidationWindowManifest(
        frame=combined,
        mode=mode,
        seed=config.validation_window_seed,
        fold_manifest_fingerprint=folds.fingerprint,
        fingerprint=fingerprint,
    )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValidationError("validation-window regeneration fingerprint mismatch")
    return ValidationWindowSet(template, manifest, folds.n_repeats)


def load_validation_window_manifest(
    path: str | Path,
    folds: FoldManifest,
    config: ValidationConfig,
    *,
    expected_fingerprint: str | None = None,
    mode: str = "sampled",
) -> ValidationWindowManifest:
    """Load fixed validation metadata and reject incompatible folds or regeneration."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"validation-window manifest not found: {source}")
    frame = pd.read_csv(
        source,
        dtype={
            "window_id": "string",
            "original_id": "string",
            "generation_mode": "string",
            "radar_availability": "string",
            "optical_month_availability": "string",
            "mask_id": "string",
        },
    )
    missing = sorted(set(WINDOW_FINGERPRINT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValidationError(f"persisted validation windows are missing columns: {missing}")
    frame = frame.loc[:, list(WINDOW_FINGERPRINT_COLUMNS)]
    frame["window_id"] = frame["window_id"].astype(object)
    frame["original_id"] = frame["original_id"].astype("category")
    for column in (
        "generation_mode",
        "radar_availability",
        "optical_month_availability",
        "mask_id",
    ):
        frame[column] = frame[column].astype("category")
    for column in (
        "label",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
    ):
        frame[column] = frame[column].astype(np.int8)
    frame["repeat"] = frame["repeat"].astype(np.int16)
    frame["fold"] = frame["fold"].astype(np.int16)
    frame["view_index"] = frame["view_index"].astype(np.int16)
    frame["augmentation_seed"] = frame["augmentation_seed"].astype(np.int64)
    content = dataframe_fingerprint(frame, columns=WINDOW_FINGERPRINT_COLUMNS)
    fingerprint = json_fingerprint(
        {
            "content": content,
            "fold_manifest": folds.fingerprint,
            "mode": mode,
            "seed": config.validation_window_seed,
        }
    )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValidationError("loaded validation-window fingerprint mismatch")
    expected_rows = folds.n_originals * folds.n_repeats
    if mode == "sampled":
        expected_rows *= config.sampled_windows_per_original
    elif mode == "exhaustive":
        expected_rows *= 24
    else:
        raise ValidationError(f"unsupported persisted validation-window mode: {mode}")
    if frame.shape[0] != expected_rows:
        raise ValidationError("loaded validation-window manifest has incomplete original coverage")
    return ValidationWindowManifest(
        frame=frame,
        mode=mode,
        seed=config.validation_window_seed,
        fold_manifest_fingerprint=folds.fingerprint,
        fingerprint=fingerprint,
    )


def validate_validation_window_manifest(manifest: ValidationWindowManifest) -> None:
    """Check complete metadata, uniqueness, fold inheritance, and content hash."""

    frame = manifest.frame
    missing = sorted(set(WINDOW_FINGERPRINT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValidationError(f"validation-window manifest is missing columns: {missing}")
    if frame.empty or frame.loc[:, list(WINDOW_FINGERPRINT_COLUMNS)].isna().any().any():
        raise ValidationError("validation-window manifest cannot be empty or incomplete")
    if frame.duplicated(["repeat", "window_id"]).any():
        raise ValidationError("window IDs must be unique within each repeat")
    crossing = frame.groupby(["repeat", "original_id"], sort=False, observed=True)["fold"].nunique()
    if crossing.ne(1).any():
        raise ValidationError("all augmented copies must inherit their original's repeat/fold")
    content = dataframe_fingerprint(frame, columns=WINDOW_FINGERPRINT_COLUMNS)
    actual = json_fingerprint(
        {
            "content": content,
            "fold_manifest": manifest.fold_manifest_fingerprint,
            "mode": manifest.mode,
            "seed": manifest.seed,
        }
    )
    if actual != manifest.fingerprint:
        raise ValidationError("validation-window content fingerprint mismatch")


def subset_temporal_windows(
    dataset: TemporalWindowDataset,
    selector: np.ndarray,
) -> TemporalWindowDataset:
    """Select rows without mutating the Phase 2 dataset or changing mask semantics."""

    selector = np.asarray(selector)
    if selector.dtype == np.bool_:
        if selector.shape != (dataset.n_windows,):
            raise ValidationError("boolean temporal-window selector has the wrong shape")
        indices = np.flatnonzero(selector)
    elif selector.ndim == 1 and np.issubdtype(selector.dtype, np.integer):
        indices = selector.astype(np.int64)
    else:
        raise ValidationError("temporal-window selector must be a boolean mask or integer indices")
    if indices.size == 0:
        raise ValidationError("temporal-window subset must not be empty")
    return TemporalWindowDataset(
        manifest=dataset.manifest.iloc[indices].reset_index(drop=True).copy(deep=True),
        values=dataset.values[indices].copy(),
        calendar_months=dataset.calendar_months[indices].copy(),
        relative_positions=dataset.relative_positions[indices].copy(),
        position_mask=dataset.position_mask[indices].copy(),
        radar_mask=dataset.radar_mask[indices].copy(),
        optical_mask=dataset.optical_mask[indices].copy(),
        band_names=dataset.band_names,
        optical_bands=dataset.optical_bands,
    )


def split_repeat_fold(
    dataset: TemporalWindowDataset,
    *,
    fold: int,
) -> tuple[TemporalWindowDataset, TemporalWindowDataset]:
    """Create disjoint train/validation views using inherited original folds."""

    validation_selector = dataset.manifest["fold"].to_numpy(dtype=np.int16) == fold
    train = subset_temporal_windows(dataset, ~validation_selector)
    validation = subset_temporal_windows(dataset, validation_selector)
    train_originals = set(train.manifest["original_id"].astype(str))
    valid_originals = set(validation.manifest["original_id"].astype(str))
    if train_originals & valid_originals:
        raise ValidationError("an original appears in both fold training and validation")
    train_windows = set(train.manifest["window_id"].astype(str))
    valid_windows = set(validation.manifest["window_id"].astype(str))
    if train_windows & valid_windows:
        raise ValidationError("a window appears in both fold training and validation")
    return train, validation
