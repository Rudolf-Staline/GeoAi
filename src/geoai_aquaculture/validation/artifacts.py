"""Aggregate-only Phase 4 validation artifact writer."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from geoai_aquaculture.data import ProjectConfig, git_provenance

from .common import ensure_directory
from .diagnostics import ClusterHoldoutPlan, LeaveSeasonOutManifest
from .folds import FoldManifest, fold_balance_summary
from .reference import ReferenceRunResult
from .views import ValidationWindowSet


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_markdown(
    folds: FoldManifest,
    windows: ValidationWindowSet,
    reference: ReferenceRunResult | None,
) -> str:
    lines = [
        "# Phase 4 validation report",
        "",
        "## Authoritative protocol",
        "",
        f"- Repeated grouped stratification: {folds.n_repeats} repeats x {folds.n_splits} folds.",
        "- Splitting unit: original labeled row; temporal windows inherit the original fold.",
        f"- Fixed primary validation windows: {windows.manifest.frame.shape[0]:,} sampled views.",
        "- Primary predictions: mean window probability per original; threshold fixed at 0.5.",
        "- The robust score is a model-selection diagnostic, not the official competition metric.",
        "",
        "## Structural fingerprints",
        "",
        f"- Fold manifest: `{folds.fingerprint}`",
        f"- Validation windows: `{windows.manifest.fingerprint}`",
        "",
    ]
    if reference is None:
        lines.extend(
            [
                "## Reference integration",
                "",
                (
                    "The optional untuned reference estimator was skipped. "
                    "No predictive claim is made."
                ),
                "",
            ]
        )
    else:
        official = reference.report.summary["official_metric"]
        robust = reference.report.summary["robust_selection"]
        lines.extend(
            [
                "## Noncompetitive reference integration",
                "",
                (
                    "This untuned logistic estimator verifies fold execution and artifact "
                    "contracts; it is not a Phase 5 baseline."
                ),
                f"- Mean F1: {official['mean_f1']:.6f}",
                f"- Mean ROC-AUC: {official['mean_roc_auc']:.6f}",
                f"- Mean official combined score: {official['mean_combined_score']:.6f}",
                f"- Robust diagnostic score: {robust['score']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Deferred diagnostics",
            "",
            (
                "Cluster holdouts are specified but fitted only inside a future model's outer "
                "training fold. Train-vs-test similarity holdouts require Phase 7 OOF domain "
                "scores."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_validation_artifacts(
    output_dir: Path,
    *,
    config: ProjectConfig,
    folds: FoldManifest,
    windows: ValidationWindowSet,
    leave_season_out: LeaveSeasonOutManifest,
    reference: ReferenceRunResult | None,
    runtime_seconds: float,
    command: str,
    exhaustive_windows: ValidationWindowSet | None = None,
) -> tuple[Path, ...]:
    """Write model-independent manifests and aggregate reference diagnostics."""

    output = ensure_directory(output_dir)
    written: list[Path] = []

    csv_tables: dict[str, pd.DataFrame] = {
        "fold_manifest": folds.frame,
        "validation_window_manifest": windows.manifest.frame,
        "leave_season_out_manifest": leave_season_out.frame,
    }
    if reference is not None:
        csv_tables.update(
            {
                "reference_oof": reference.oof.original,
                "reference_window_predictions": reference.oof.windows,
                "fold_metrics": reference.report.fold_metrics,
                "repeat_metrics": reference.report.repeat_metrics,
                "slice_metrics": reference.report.slice_metrics,
                "prediction_stability": reference.report.prediction_stability,
                "reference_fold_metadata": reference.fold_metadata,
            }
        )
    for name, frame in csv_tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        written.append(path)

    balance = fold_balance_summary(folds)
    fold_summary = {
        "n_originals": folds.n_originals,
        "n_repeats": folds.n_repeats,
        "n_splits": folds.n_splits,
        "manifest_rows": int(folds.frame.shape[0]),
        "fold_manifest_fingerprint": folds.fingerprint,
        "class_balance": balance.to_dict(orient="records"),
    }
    slice_definitions = {
        "window_lengths": list(config.validation.window_lengths),
        "start_months": list(range(1, 10)),
        "seasons": {season.name: list(season.start_months) for season in config.validation.seasons},
        "optical_gaps": {"none": "0", "one": "1", "two_or_more": ">=2"},
        "optical_valid_proportion": {
            "severely_limited": f"< {config.validation.optical_severe_limit}",
            "moderate": (
                f"[{config.validation.optical_severe_limit}, "
                f"{config.validation.optical_high_completeness})"
            ),
            "high_incomplete": (f"[{config.validation.optical_high_completeness}, 1)"),
            "complete": "1",
        },
        "availability": [
            "radar_complete",
            "optical_complete",
            "radar_only",
            "severely_optical_limited",
        ],
        "leave_season_out_interpretation": (
            "For an outer repeat/fold, validation uses only held-season windows of held-out "
            "originals; training uses non-held-season windows from other originals only."
        ),
    }
    cluster_plan = ClusterHoldoutPlan(
        n_clusters=config.validation.cluster_n_clusters,
        minimum_cluster_size=config.validation.cluster_min_size,
    )
    diagnostic_plans = {
        "cluster_holdout": asdict(cluster_plan),
        "adversarial_similarity_holdout": {
            "fraction": config.validation.similarity_holdout_fraction,
            "minimum_samples": config.validation.similarity_holdout_min_samples,
            "requires": "complete per-original OOF train-vs-test similarity probabilities",
            "status": "interface_ready_domain_classifier_deferred_to_phase7",
            "pond_model_feature": False,
        },
    }
    protocol = {
        "schema_version": 1,
        "seed": config.validation.seed,
        "repeat_seeds": folds.frame.groupby("repeat")["repeat_seed"].first().to_dict(),
        "n_splits": config.validation.n_splits,
        "n_repeats": config.validation.n_repeats,
        "primary_window_mode": config.validation.primary_window_mode,
        "sampled_windows_per_original": config.validation.sampled_windows_per_original,
        "validation_window_seed": config.validation.validation_window_seed,
        "validation_window_policy": (
            "one deterministic test-mask-derived panel reused for all models and repeats"
        ),
        "aggregation_method": config.validation.aggregation_method,
        "threshold": config.validation.threshold,
        "robust_score_weights": asdict(config.validation.robust_score_weights),
        "primary_metrics_are_original_level": True,
    }
    fingerprints = {
        "fold_manifest": folds.fingerprint,
        "validation_window_manifest": windows.manifest.fingerprint,
        "leave_season_out_manifest": leave_season_out.fingerprint,
        "reference_oof": reference.oof.fingerprint if reference is not None else None,
        "feature_schema": (
            str(reference.fold_metadata["feature_schema_fingerprint"].iloc[0])
            if reference is not None
            else None
        ),
        "exhaustive_window_manifest": (
            exhaustive_windows.manifest.fingerprint if exhaustive_windows is not None else None
        ),
    }
    json_values = {
        "fold_summary": fold_summary,
        "slice_definitions": slice_definitions,
        "diagnostic_plans": diagnostic_plans,
        "protocol": protocol,
        "fingerprints": fingerprints,
    }
    if reference is not None:
        json_values["reference_metrics"] = reference.report.summary
    for name, value in json_values.items():
        path = output / f"{name}.json"
        _write_json(path, value)
        written.append(path)

    report_path = output / "validation_report.md"
    report_path.write_text(_report_markdown(folds, windows, reference), encoding="utf-8")
    written.append(report_path)
    run_metadata = {
        "command": command,
        "config_path": config.source_path.as_posix(),
        "config_sha256": _config_sha256(config.source_path),
        "git": git_provenance(config.project_root),
        "runtime_seconds": runtime_seconds,
        "reference_runtime_seconds": reference.runtime_seconds if reference is not None else None,
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "raw_feature_rows_persisted": False,
    }
    run_path = output / "run_metadata.json"
    _write_json(run_path, run_metadata)
    written.append(run_path)
    return tuple(written)
