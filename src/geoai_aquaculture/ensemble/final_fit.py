"""Full-data fitting and deterministic test inference for retained Phase 8 experts."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.data import load_competition_data, materialize_test_windows
from geoai_aquaculture.features import (
    build_sequence_features,
    build_tabular_features,
    select_tabular_features,
)
from geoai_aquaculture.training import (
    load_tabular_experiment_config,
    load_temporal_experiment_config,
    prepare_tabular_experiment_data,
)

from .config import FinalCandidateConfig


class FinalFitError(ValueError):
    """Raised when full-data fitting violates an accepted experiment contract."""


@dataclass(frozen=True, slots=True)
class FinalModelPrediction:
    """One fitted final model and its row-aligned test probabilities."""

    experiment_id: str
    model_family: str
    probabilities: np.ndarray
    model_path: Path
    model_sha256: str
    training_seconds: float
    inference_seconds: float
    training_parameter: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise FinalFitError("final model probabilities must be a non-empty vector")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise FinalFitError("final model probabilities must be finite in [0, 1]")
        if self.training_seconds <= 0.0 or self.inference_seconds < 0.0:
            raise FinalFitError("final fit runtime metadata is invalid")
        if self.training_parameter < 1:
            raise FinalFitError("final iteration or epoch count must be positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median_training_parameter(artifact_dir: Path, key: str, fallback: int) -> int:
    path = artifact_dir / "fold_models_manifest.json"
    if not path.is_file():
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(key, fallback)
    if not isinstance(value, int) or value < 1:
        raise FinalFitError(f"invalid {key} in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _assert_test_order(expected_ids: np.ndarray, actual_ids: np.ndarray, model_name: str) -> None:
    expected = np.asarray(expected_ids, dtype=str)
    actual = np.asarray(actual_ids, dtype=str)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise FinalFitError(f"{model_name} test predictions do not follow Test.csv ID order")


def fit_final_tree(
    project,
    candidate: FinalCandidateConfig,
    *,
    output_dir: Path,
) -> FinalModelPrediction:
    """Fit the accepted full-feature LightGBM on every labeled temporal view."""

    from lightgbm import LGBMClassifier

    experiment = load_tabular_experiment_config(candidate.experiment_config)
    if experiment.experiment_id != candidate.experiment_id or experiment.model.family != "lightgbm":
        raise FinalFitError("tree candidate declaration and experiment configuration differ")
    prepared = prepare_tabular_experiment_data(project, experiment)
    repeat_zero = prepared.windows.frame.loc[prepared.windows.frame["repeat"].eq(0)].reset_index(
        drop=True
    )
    if repeat_zero.shape[0] != prepared.features.shape[0]:
        raise FinalFitError("final tree labels and feature rows are misaligned")
    labels = repeat_zero["label"].to_numpy(dtype=np.int8)
    iterations = _median_training_parameter(
        candidate.artifact_dir,
        "median_best_iteration",
        int(experiment.model.parameters.get("n_estimators", 650)),
    )
    parameters = {
        **dict(experiment.model.parameters),
        "n_estimators": iterations,
        "random_state": experiment.seed,
        "n_jobs": project.tabular.cpu_threads,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    model = LGBMClassifier(**parameters)
    started = perf_counter()
    # Uniform weighting is scientifically equivalent to equal-original weighting because the
    # accepted fixed panel contains exactly eight views for every original.
    model.fit(prepared.features, labels)
    training_seconds = perf_counter() - started

    data = load_competition_data(project)
    test_windows = materialize_test_windows(data)
    full_test = build_tabular_features(test_windows, project.features)
    selected_test = select_tabular_features(
        full_test,
        experiment.feature_set,
        project.features.bands,
    )
    if selected_test.feature_names != prepared.feature_names:
        raise FinalFitError("final tree train/test feature schemas differ")
    _assert_test_order(
        data.test[project.data.id_column].to_numpy(),
        full_test.original_ids,
        "LightGBM",
    )
    inference_started = perf_counter()
    probabilities = np.asarray(model.predict_proba(selected_test.features)[:, 1], dtype=np.float64)
    inference_seconds = perf_counter() - inference_started

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lightgbm_final.txt"
    model.booster_.save_model(model_path.as_posix())
    from lightgbm import Booster

    restored_tree = Booster(model_file=model_path.as_posix())
    restored_probability = np.asarray(
        restored_tree.predict(selected_test.features), dtype=np.float64
    )
    if not np.allclose(probabilities, restored_probability, rtol=1e-12, atol=1e-12):
        raise FinalFitError("reloaded LightGBM probabilities differ from pre-save inference")
    importance = pd.DataFrame(
        {
            "feature": prepared.feature_names,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    importance.to_csv(output_dir / "lightgbm_feature_importance.csv", index=False)

    shap_status: dict[str, Any] = {"status": "not-run"}
    try:
        import matplotlib.pyplot as plt
        import shap

        sample_size = min(600, selected_test.features.shape[0])
        sample = selected_test.features.iloc[:sample_size]
        explainer = shap.TreeExplainer(model.booster_)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            array = np.asarray(values[-1], dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.ndim == 3:
            array = array[:, :, -1]
        mean_abs = np.mean(np.abs(array), axis=0)
        shap_table = pd.DataFrame(
            {"feature": selected_test.feature_names, "mean_absolute_shap": mean_abs}
        ).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
        shap_table.to_csv(output_dir / "lightgbm_shap_importance.csv", index=False)
        top = shap_table.head(20).sort_values("mean_absolute_shap")
        figure, axis = plt.subplots(figsize=(9, 7))
        axis.barh(top["feature"], top["mean_absolute_shap"])
        axis.set_xlabel("Mean absolute SHAP value")
        axis.set_title("Final LightGBM global feature importance")
        figure.tight_layout()
        figure.savefig(output_dir / "lightgbm_shap_top20.png", dpi=160)
        plt.close(figure)
        shap_status = {"status": "complete", "sample_size": sample_size}
    except Exception as exc:  # pragma: no cover - optional interpretation dependency
        shap_status = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    _write_json(output_dir / "lightgbm_interpretation.json", shap_status)

    return FinalModelPrediction(
        experiment_id=experiment.experiment_id,
        model_family="lightgbm",
        probabilities=probabilities,
        model_path=model_path,
        model_sha256=sha256_file(model_path),
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        training_parameter=iterations,
        metadata={
            "feature_count": len(prepared.feature_names),
            "training_windows": int(prepared.features.shape[0]),
            "training_originals": int(repeat_zero["original_id"].nunique()),
            "test_rows": int(probabilities.size),
            "experiment_config_fingerprint": experiment.fingerprint,
            "selected_feature_schema_fingerprint": prepared.selected_feature_schema_fingerprint,
            "interpretation": shap_status,
            "serialization_verified": True,
        },
    )


def _temporal_predict(model, sequence, normalizer, *, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    probabilities: list[np.ndarray] = []
    indices = np.arange(len(sequence.original_ids), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = normalizer.tensors(sequence, batch_indices, device=torch.device("cpu"))
            output = model(batch)
            probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise FinalFitError("final temporal inference produced invalid probabilities")
    return result


def fit_final_temporal(
    project,
    candidate: FinalCandidateConfig,
    *,
    output_dir: Path,
) -> FinalModelPrediction:
    """Fit the accepted compact GRU on all labeled windows for a fixed OOF-derived epoch count."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - explicit environment acceptance catches this
        raise FinalFitError("PyTorch is required for the retained temporal final model") from exc
    from geoai_aquaculture.models.temporal import (
        SensorGatedGRU,
        architecture_from_dict,
        count_trainable_parameters,
    )
    from geoai_aquaculture.training.temporal import (
        SequenceNormalizer,
        prepare_temporal_experiment_data,
    )

    experiment = load_temporal_experiment_config(candidate.experiment_config)
    if experiment.experiment_id != candidate.experiment_id:
        raise FinalFitError("temporal candidate declaration and experiment configuration differ")
    prepared = prepare_temporal_experiment_data(project)
    sequence = prepared.sequence
    if sequence.labels is None:
        raise FinalFitError("final temporal training sequence has no labels")
    epochs = _median_training_parameter(
        candidate.artifact_dir,
        "median_best_epoch",
        max(1, experiment.training.max_epochs // 2),
    )
    seed = experiment.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(experiment.training.cpu_threads)
    torch.use_deterministic_algorithms(True)
    architecture = architecture_from_dict(
        dict(experiment.architecture),
        radar_channels=len(sequence.radar_feature_names),
        optical_channels=len(sequence.optical_feature_names),
        index_channels=len(sequence.index_feature_names),
    )
    model = SensorGatedGRU(architecture).to(torch.device("cpu"))
    parameter_count = count_trainable_parameters(model)
    if parameter_count > 300_000:
        raise FinalFitError("final temporal model exceeds the accepted compact parameter limit")
    selector = np.ones(len(sequence.original_ids), dtype=bool)
    normalizer = SequenceNormalizer.fit(
        sequence,
        selector,
        input_clip=experiment.training.input_clip,
    )
    labels = sequence.labels.astype(np.float32)
    original_ids = sequence.original_ids.astype(str)
    counts = pd.Series(original_ids).value_counts().to_dict()
    row_weights = np.asarray([1.0 / counts[value] for value in original_ids], dtype=np.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
    )
    started = perf_counter()
    all_indices = np.arange(len(original_ids), dtype=np.int64)
    final_loss = np.nan
    for epoch in range(epochs):
        model.train()
        permutation = np.random.default_rng(seed + epoch).permutation(all_indices)
        for start in range(0, permutation.size, experiment.training.batch_size):
            batch_indices = permutation[start : start + experiment.training.batch_size]
            batch = normalizer.tensors(sequence, batch_indices, device=torch.device("cpu"))
            target = torch.from_numpy(labels[batch_indices]).to(torch.device("cpu"))
            weight = torch.from_numpy(row_weights[batch_indices]).to(torch.device("cpu"))
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            per_row = functional.binary_cross_entropy_with_logits(
                output.logits,
                target,
                reduction="none",
            )
            loss = torch.sum(per_row * weight) / torch.sum(weight)
            if not torch.isfinite(loss):
                raise FinalFitError("final temporal training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.training.gradient_clip)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    training_seconds = perf_counter() - started

    data = load_competition_data(project)
    test_windows = materialize_test_windows(data)
    test_sequence = build_sequence_features(test_windows, project.features)
    if test_sequence.schema_fingerprint != sequence.schema_fingerprint:
        raise FinalFitError("final temporal train/test sequence schemas differ")
    _assert_test_order(
        data.test[project.data.id_column].to_numpy(),
        test_sequence.original_ids,
        "masked GRU",
    )
    inference_started = perf_counter()
    probabilities = _temporal_predict(
        model,
        test_sequence,
        normalizer,
        batch_size=experiment.training.batch_size,
    )
    inference_seconds = perf_counter() - inference_started

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "gru_final.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": asdict(architecture),
            "normalizer": normalizer.as_dict(),
            "experiment_id": experiment.experiment_id,
            "seed": seed,
            "epochs": epochs,
            "parameter_count": parameter_count,
        },
        model_path,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    restored_temporal = SensorGatedGRU(architecture).to(torch.device("cpu"))
    restored_temporal.load_state_dict(checkpoint["state_dict"], strict=True)
    restored_probability = _temporal_predict(
        restored_temporal,
        test_sequence,
        normalizer,
        batch_size=experiment.training.batch_size,
    )
    if not np.allclose(probabilities, restored_probability, rtol=0.0, atol=1e-7):
        raise FinalFitError("reloaded GRU probabilities differ from pre-save inference")
    if checkpoint.get("normalizer") != normalizer.as_dict():
        raise FinalFitError("saved GRU normalizer metadata differs from the fitted normalizer")
    return FinalModelPrediction(
        experiment_id=experiment.experiment_id,
        model_family="masked_gru",
        probabilities=probabilities,
        model_path=model_path,
        model_sha256=sha256_file(model_path),
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        training_parameter=epochs,
        metadata={
            "parameter_count": parameter_count,
            "training_windows": len(original_ids),
            "training_originals": int(pd.Series(original_ids).nunique()),
            "test_rows": int(probabilities.size),
            "experiment_config_fingerprint": experiment.fingerprint,
            "sequence_schema_fingerprint": sequence.schema_fingerprint,
            "final_batch_loss": final_loss,
            "serialization_verified": True,
        },
    )
