from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoai_aquaculture.data import load_project_config
from geoai_aquaculture.ensemble import (
    FinalCandidate,
    FinalCandidateConfig,
    FinalConfigError,
    FittedCalibrator,
    crossfit_calibration,
    expected_calibration_error,
    learn_nested_weight,
    load_final_delivery_config,
)
from geoai_aquaculture.ensemble.delivery import _prior_shift_em, _trustworthiness_text
from geoai_aquaculture.ensemble.final_fit import (
    FinalFitError,
    FinalModelPrediction,
    _assert_test_order,
)
from geoai_aquaculture.validation import (
    FoldManifest,
    build_oof_predictions,
    dataframe_fingerprint,
    make_window_prediction_frame,
)
from geoai_aquaculture.validation.folds import FOLD_COLUMNS


def _synthetic_fold_manifest() -> FoldManifest:
    records: list[dict[str, object]] = []
    ids = [f"id_{index:02d}" for index in range(30)]
    labels = [index % 2 for index in range(30)]
    for repeat in range(3):
        for index, (original_id, label) in enumerate(zip(ids, labels, strict=True)):
            records.append(
                {
                    "ID": original_id,
                    "original_id": original_id,
                    "label": label,
                    "repeat": repeat,
                    "fold": index % 5,
                    "repeat_seed": 2026 + 10_007 * repeat,
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["ID"] = frame["ID"].astype("string")
    frame["original_id"] = frame["original_id"].astype("string")
    frame["label"] = frame["label"].astype("int64")
    frame["repeat"] = frame["repeat"].astype("int16")
    frame["fold"] = frame["fold"].astype("int16")
    frame["repeat_seed"] = frame["repeat_seed"].astype("int64")
    return FoldManifest(
        frame=frame,
        n_originals=30,
        n_splits=5,
        n_repeats=3,
        seed=2026,
        fingerprint=dataframe_fingerprint(frame, columns=FOLD_COLUMNS),
    )


def _candidate(kind: str, folds: FoldManifest, probabilities: np.ndarray) -> FinalCandidate:
    records: list[dict[str, object]] = []
    rows: list[float] = []
    for row in folds.frame.itertuples(index=False):
        for view, (start, length) in enumerate(((1, 4), (4, 5), (7, 6))):
            records.append(
                {
                    "original_id": str(row.original_id),
                    "repeat": int(row.repeat),
                    "fold": int(row.fold),
                    "label": int(row.label),
                    "window_id": f"{row.original_id}-r{row.repeat}-v{view}",
                    "window_start": start,
                    "window_end": start + length - 1,
                    "window_length": length,
                    "radar_months": length,
                    "optical_months": length - (view == 2),
                    "internal_optical_gap_count": int(view == 2),
                }
            )
            rows.append(float(probabilities[int(row.repeat) * 30 + int(str(row.original_id)[3:])]))
    manifest = pd.DataFrame.from_records(records)
    windows = make_window_prediction_frame(
        manifest,
        np.asarray(rows, dtype=np.float64),
        experiment_id=f"EXP-{kind.upper()}",
        model_id=kind,
        fold_manifest_fingerprint=folds.fingerprint,
        validation_window_fingerprint="window-fingerprint",
    )
    oof = build_oof_predictions(
        windows,
        folds,
        validation_window_fingerprint="window-fingerprint",
    )
    declaration = FinalCandidateConfig(
        experiment_id=f"EXP-{kind.upper()}",
        kind=kind,  # type: ignore[arg-type]
        artifact_dir=Path(f"artifacts/{kind}"),
        experiment_config=Path(f"configs/{kind}.yaml"),
        role=kind,
    )
    return FinalCandidate(declaration, {}, {}, oof)


def test_final_config_loads_and_threshold_is_immutable(tmp_path: Path) -> None:
    loaded = load_final_delivery_config("configs/final.yaml")
    assert loaded.threshold == 0.5
    assert loaded.tree_candidate.kind == "tree"
    invalid = tmp_path / "final.yaml"
    invalid.write_text(
        Path("configs/final.yaml").read_text().replace("threshold: 0.5", "threshold: 0.4"),
        encoding="utf-8",
    )
    # The temporary file is outside the project; copy pyproject to establish a root.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n", encoding="utf-8")
    with pytest.raises(FinalConfigError, match="threshold"):
        load_final_delivery_config(invalid)


def test_calibrators_are_bounded_and_ece_detects_miscalibration() -> None:
    p = np.asarray([0.05, 0.2, 0.8, 0.95])
    sigmoid = FittedCalibrator("sigmoid", (0.8,), -0.1)
    beta = FittedCalibrator("beta", (0.7, -0.4), 0.2)
    for calibrator in (sigmoid, beta):
        transformed = calibrator.transform(p)
        assert transformed.shape == p.shape
        assert np.isfinite(transformed).all()
        assert ((transformed >= 0.0) & (transformed <= 1.0)).all()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    assert expected_calibration_error(labels, p) < expected_calibration_error(labels, 1.0 - p)


def test_nested_weights_and_calibration_preserve_complete_oof(monkeypatch) -> None:
    folds = _synthetic_fold_manifest()
    labels = folds.frame["label"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(7)
    tree_probability = np.clip(
        0.12 + 0.76 * labels + rng.normal(0.0, 0.04, labels.size),
        0.01,
        0.99,
    )
    temporal_probability = np.clip(
        0.18 + 0.64 * labels + rng.normal(0.0, 0.07, labels.size), 0.01, 0.99
    )
    tree = _candidate("tree", folds, tree_probability)
    temporal = _candidate("temporal", folds, temporal_probability)
    project = load_project_config("configs/base.yaml")
    monkeypatch.setattr("geoai_aquaculture.ensemble.oof.load_fold_manifest", lambda *a, **k: folds)
    monkeypatch.setattr(
        "geoai_aquaculture.ensemble.calibration.load_fold_manifest", lambda *a, **k: folds
    )
    result = learn_nested_weight(
        tree,
        temporal,
        project,
        grid_step=0.1,
        fixed_tree_weights=(0.5, 0.7),
    )
    assert result.fold_weights.shape[0] == 15
    assert result.fold_weights["tree_weight"].between(0.0, 1.0).all()
    assert result.production.oof.original.shape[0] == 90
    calibrated = crossfit_calibration(result.production.oof, project, "sigmoid")
    assert calibrated.oof.original.shape[0] == 90
    assert calibrated.fold_parameters.shape[0] == 15
    assert calibrated.oof.original["probability"].between(0.0, 1.0).all()


def test_final_model_prediction_rejects_invalid_probability(tmp_path: Path) -> None:
    model = tmp_path / "model.txt"
    model.write_text("x", encoding="utf-8")
    with pytest.raises(FinalFitError, match="probabilities"):
        FinalModelPrediction(
            experiment_id="EXP",
            model_family="tree",
            probabilities=np.asarray([0.2, np.nan]),
            model_path=model,
            model_sha256="x",
            training_seconds=1.0,
            inference_seconds=0.1,
            training_parameter=10,
            metadata={},
        )


def test_prior_shift_is_diagnostic_and_trust_sections_fit_limits(tmp_path: Path) -> None:
    diagnostic = _prior_shift_em(np.asarray([0.1, 0.2, 0.8, 0.9]), 0.4)
    assert diagnostic["applied"] is False
    tree = FinalModelPrediction(
        experiment_id="TREE",
        model_family="lightgbm",
        probabilities=np.asarray([0.2, 0.8]),
        model_path=tmp_path / "tree",
        model_sha256="a",
        training_seconds=10.0,
        inference_seconds=1.0,
        training_parameter=20,
        metadata={},
    )
    temporal = FinalModelPrediction(
        experiment_id="GRU",
        model_family="masked_gru",
        probabilities=np.asarray([0.3, 0.7]),
        model_path=tmp_path / "gru",
        model_sha256="b",
        training_seconds=20.0,
        inference_seconds=1.0,
        training_parameter=15,
        metadata={},
    )
    text, counts = _trustworthiness_text(
        top_features=["optical__ndwi__max", "radar__vv_vh__median"],
        tree=tree,
        temporal=temporal,
    )
    assert "Trustworthiness" in text
    assert all(count <= 100 for count in counts.values())


def test_final_fit_requires_exact_test_id_order() -> None:
    _assert_test_order(np.asarray(["a", "b"]), np.asarray(["a", "b"]), "model")
    with pytest.raises(FinalFitError, match=r"Test\.csv ID order"):
        _assert_test_order(np.asarray(["a", "b"]), np.asarray(["b", "a"]), "model")
