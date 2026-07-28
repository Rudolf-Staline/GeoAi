from __future__ import annotations

import pandas as pd
import pytest

from geoai_aquaculture.submission import build_submission, validate_submission


def test_submission_has_exact_schema_and_threshold() -> None:
    ids = ["a", "b", "c"]
    submission = build_submission(ids, [0.1, 0.5, 0.9])

    assert submission.columns.tolist() == ["ID", "TargetF1", "TargetRAUC"]
    assert submission["TargetF1"].tolist() == [0, 1, 1]

    sample = pd.DataFrame({"ID": ids, "TargetF1": [0, 0, 0], "TargetRAUC": [0, 0, 0]})
    validate_submission(submission, sample)


def test_submission_rejects_wrong_id_order() -> None:
    submission = build_submission(["a", "b"], [0.1, 0.9])
    sample = pd.DataFrame(
        {"ID": ["b", "a"], "TargetF1": [0, 0], "TargetRAUC": [0.0, 0.0]}
    )

    with pytest.raises(ValueError, match="row order"):
        validate_submission(submission, sample)


def test_submission_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_submission(["a", "a"], [0.2, 0.8])
