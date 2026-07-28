from __future__ import annotations

import numpy as np
import pytest

from geoai_aquaculture.metrics import competition_metrics


def test_competition_metrics_use_fixed_half_threshold() -> None:
    metrics = competition_metrics([0, 0, 1, 1], [0.10, 0.49, 0.50, 0.90])

    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.competition_score == pytest.approx(1.0)


def test_competition_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        competition_metrics([0, 1], [0.2, np.nan])


def test_competition_metrics_require_both_classes() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        competition_metrics([1, 1], [0.2, 0.8])
