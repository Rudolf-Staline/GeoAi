from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoai_aquaculture.constants import FIXED_THRESHOLD
from geoai_aquaculture.ensemble.gating import (
    META_FEATURES,
    GatingError,
    _fit_gate,
    _policy_probabilities,
    _select_c,
    build_gate_features,
)


def _frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold in range(5):
        for original in range(12):
            original_id = f"id_{fold}_{original}"
            label = (original + fold) % 2
            for repeat in range(2):
                difficult = original % 3 == 0
                if difficult:
                    cat_probability = 0.80 if label == 0 else 0.20
                    invariant_probability = 0.80 if label == 1 else 0.20
                    optical_months = 2
                    gaps = 2
                else:
                    cat_probability = 0.80 if label == 1 else 0.20
                    invariant_probability = 0.80 if label == 0 else 0.20
                    optical_months = 5
                    gaps = 0
                records.append(
                    {
                        "original_id": original_id,
                        "repeat": repeat,
                        "meta_fold": fold,
                        "label": label,
                        "cat_probability": cat_probability,
                        "invariant_probability": invariant_probability,
                        "window_start": 1 + (original % 7),
                        "window_length": 6,
                        "radar_months": 6,
                        "optical_months": optical_months,
                        "internal_optical_gap_count": gaps,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_gate_features_are_finite_and_stable() -> None:
    features = build_gate_features(_frame())
    assert tuple(features.columns) == META_FEATURES
    assert features.shape == (120, len(META_FEATURES))
    assert np.isfinite(features.to_numpy()).all()
    assert set(features["binary_disagreement"].unique()) == {1}


def test_gate_features_reject_invalid_probability() -> None:
    frame = _frame()
    frame.loc[0, "cat_probability"] = 1.2
    with pytest.raises(GatingError, match="must lie in"):
        build_gate_features(frame)


def test_gate_learns_missingness_regime() -> None:
    frame = _frame()
    gate = _fit_gate(frame, 1.0)
    prediction = gate.predict(build_gate_features(frame))
    target = (
        (frame["invariant_probability"] >= FIXED_THRESHOLD).to_numpy(dtype=np.int8)
        == frame["label"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    assert ((prediction >= 0.5).astype(np.int8) == target).mean() > 0.95
    assert gate.training_originals == frame["original_id"].nunique()


def test_boundary_policy_changes_only_selected_disagreements() -> None:
    frame = _frame().iloc[:10].copy().reset_index(drop=True)
    gate_probability = np.where(frame["optical_months"].eq(2), 0.9, 0.1)
    probability = _policy_probabilities(frame, gate_probability, policy="boundary")
    expected = np.where(
        gate_probability >= 0.5,
        frame["invariant_probability"].to_numpy() >= FIXED_THRESHOLD,
        frame["cat_probability"].to_numpy() >= FIXED_THRESHOLD,
    )
    assert np.array_equal(probability >= FIXED_THRESHOLD, expected)
    cat_label = frame["cat_probability"].to_numpy() >= FIXED_THRESHOLD
    unchanged = expected == cat_label
    assert np.allclose(
        probability[unchanged],
        frame.loc[unchanged, "cat_probability"].to_numpy(),
    )


def test_nested_c_selection_returns_declared_value() -> None:
    selected, table = _select_c(_frame().reset_index(drop=True), (0.05, 0.2, 1.0))
    assert selected in {0.05, 0.2, 1.0}
    assert set(table.columns) == {"c_value", "combined_score"}
    assert table["combined_score"].between(0.0, 1.0).all()


def test_unknown_policy_is_rejected() -> None:
    frame = _frame().iloc[:4].copy().reset_index(drop=True)
    with pytest.raises(GatingError, match="unsupported gate policy"):
        _policy_probabilities(frame, np.full(4, 0.5), policy="unknown")
