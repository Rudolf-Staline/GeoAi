"""Ignored artifact writers for Phase 7 diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from geoai_aquaculture.data import git_provenance

from .adversarial import DomainValidationResult
from .config import Phase7Config
from .diagnostics import RepresentationSummary
from .evaluation import HoldoutEvaluation


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def write_domain_result(root: Path, result: DomainValidationResult) -> Path:
    output = root / "representations" / result.representation
    output.mkdir(parents=True, exist_ok=True)
    result.window_oof.to_csv(output / "window_oof.csv", index=False)
    result.entity_oof.to_csv(output / "entity_oof.csv", index=False)
    result.train_similarity_scores.to_csv(output / "train_similarity_scores.csv", index=False)
    result.feature_importance.to_csv(output / "feature_importance_raw.csv", index=False)
    result.fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    write_json(
        output / "metrics.json",
        {
            "representation": result.representation,
            "metrics": result.metrics.as_dict(),
            "fingerprint": result.fingerprint,
        },
    )
    return output


def write_phase7_summary(
    root: Path,
    config: Phase7Config,
    summaries: tuple[RepresentationSummary, ...],
    *,
    feature_importance: pd.DataFrame,
    group_importance: pd.DataFrame,
    feature_shift: pd.DataFrame,
    similarity_scores: pd.DataFrame,
    similarity_holdout: pd.DataFrame,
    importance_weights: pd.DataFrame,
    holdout_evaluations: tuple[HoldoutEvaluation, ...],
    decisions: list[dict[str, Any]],
    project_root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(summary) for summary in summaries]).to_csv(
        root / "representation_metrics.csv", index=False
    )
    feature_importance.to_csv(root / "domain_feature_importance.csv", index=False)
    group_importance.to_csv(root / "domain_group_importance.csv", index=False)
    feature_shift.to_csv(root / "feature_shift_metrics.csv", index=False)
    similarity_scores.to_csv(root / "train_oof_similarity_scores.csv", index=False)
    similarity_holdout.to_csv(root / "similarity_holdout_manifest.csv", index=False)
    importance_weights.to_csv(root / "importance_weights.csv", index=False)
    holdout_rows: list[dict[str, Any]] = []
    for evaluation in holdout_evaluations:
        for row in evaluation.repeat_metrics.to_dict("records"):
            holdout_rows.append(
                {
                    "model_name": evaluation.model_name,
                    "selected_count": evaluation.selected_count,
                    **row,
                }
            )
    pd.DataFrame.from_records(holdout_rows).to_csv(root / "holdout_model_metrics.csv", index=False)
    write_json(root / "adaptation_decisions.json", decisions)
    write_json(
        root / "run_metadata.json",
        {
            "config_path": config.source_path,
            "config_fingerprint": config.fingerprint,
            "git": git_provenance(project_root),
        },
    )


def write_decision_report(root: Path, payload: dict[str, Any]) -> None:
    write_json(root / "decision_report.json", payload)
    lines = [
        "# Phase 7 domain-shift decision report",
        "",
        f"- Selected domain representation: `{payload['selected_representation']}`",
        f"- Selected representation ROC-AUC: `{payload['selected_domain_auc']:.6f}`",
        f"- Similarity holdout size: `{payload['similarity_holdout_count']}`",
        "",
        "## Adaptation decisions",
        "",
    ]
    for decision in payload["adaptations"]:
        lines.extend(
            [
                f"### {decision['method']}",
                "",
                f"- Decision: **{decision['decision']}**",
                f"- Mean robust score: `{decision['mean_robust_score']:.6f}`",
                f"- Mean official score: `{decision['mean_combined_score']:.6f}`",
                f"- Baseline robust score: `{decision['baseline_robust_score']:.6f}`",
                f"- Rationale: {decision['rationale']}",
                "",
            ]
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
