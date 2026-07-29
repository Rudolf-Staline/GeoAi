"""Phase 5 experiment artifacts, compatibility guards, and importance audit."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geoai_aquaculture.data import ProjectConfig, git_provenance
from geoai_aquaculture.features import FeatureRegistry

from .config import ExperimentStage, TabularExperimentConfig
from .results import ExperimentArtifactManifest, TabularTrainingResult


class ExperimentArtifactError(ValueError):
    """Raised when artifacts are incomplete, incompatible, or unsafe to overwrite."""


def sha256_file(path: str | Path) -> str:
    """Hash one persisted model or configuration file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def experiment_artifact_dir(
    root: Path,
    experiment_id: str,
    stage: ExperimentStage,
) -> Path:
    """Give full runs the canonical ID directory and isolate screening artifacts."""

    name = experiment_id if stage == "full" else f"{experiment_id}__{stage}"
    return root.resolve() / name


def _safe_experiment_target(root: Path, target: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target.parent != resolved_root or resolved_target == resolved_root:
        raise ExperimentArtifactError("experiment artifact target must be one direct child of root")


def prepare_experiment_artifact_dir(
    root: Path,
    experiment_id: str,
    stage: ExperimentStage,
    *,
    overwrite: bool,
) -> Path:
    """Create an empty ignored target; replacement requires an explicit flag."""

    output = experiment_artifact_dir(root, experiment_id, stage)
    _safe_experiment_target(root, output)
    root.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise ExperimentArtifactError(
                f"experiment artifacts already exist at {output}; use resume or explicit overwrite"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_experiment_artifact_manifest(path: str | Path) -> ExperimentArtifactManifest:
    """Load the stable manifest without trusting any other experiment files."""

    source = Path(path)
    if source.is_dir():
        source = source / "experiment_manifest.json"
    if not source.is_file():
        raise FileNotFoundError(f"experiment manifest not found: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    try:
        return ExperimentArtifactManifest(**value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentArtifactError(f"malformed experiment manifest: {source}") from exc


def assert_resume_compatible(
    output_dir: Path,
    *,
    stage: ExperimentStage,
    experiment: TabularExperimentConfig,
    base_config_sha256: str,
    provenance: dict[str, Any],
    fold_manifest_fingerprint: str,
    validation_window_fingerprint: str,
    full_feature_schema_fingerprint: str,
    selected_feature_schema_fingerprint: str,
) -> ExperimentArtifactManifest:
    """Accept resume only for an already complete byte-compatible scientific run."""

    manifest = load_experiment_artifact_manifest(output_dir)
    expected = {
        "status": "complete",
        "stage": stage,
        "experiment_id": experiment.experiment_id,
        "experiment_config_fingerprint": experiment.fingerprint,
        "base_config_sha256": base_config_sha256,
        "git_commit": provenance["commit"],
        "tracked_files_dirty": provenance["tracked_files_dirty"],
        "fold_manifest_fingerprint": fold_manifest_fingerprint,
        "validation_window_fingerprint": validation_window_fingerprint,
        "full_feature_schema_fingerprint": full_feature_schema_fingerprint,
        "selected_feature_schema_fingerprint": selected_feature_schema_fingerprint,
    }
    actual = manifest.as_dict()
    mismatch = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatch:
        raise ExperimentArtifactError(f"resume artifact compatibility mismatch: {mismatch}")
    required = {
        "resolved_config.yaml",
        "feature_list.txt",
        "metrics.json",
        "oof_predictions.csv",
        "window_predictions.csv",
        "fold_models_manifest.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise ExperimentArtifactError(f"resume artifact is incomplete: {missing}")
    return manifest


def summarize_feature_importance(
    importance: pd.DataFrame,
    registry: FeatureRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize fold importances, report stability, and flag—not remove—suspicious features."""

    required = {"repeat", "fold", "feature", "importance_type", "importance"}
    missing = sorted(required - set(importance.columns))
    if missing:
        raise ExperimentArtifactError(f"feature importance is missing columns: {missing}")
    group_by_feature = {
        definition.name: definition.feature_group for definition in registry.definitions
    }
    frame = importance.copy()
    if set(frame["feature"]) != set(registry.feature_names):
        raise ExperimentArtifactError("feature importance does not cover the selected registry")
    frame["feature_group"] = frame["feature"].map(group_by_feature)
    totals = frame.groupby(["repeat", "fold", "importance_type"], observed=True)[
        "importance"
    ].transform("sum")
    frame["normalized_importance"] = np.where(totals > 0.0, frame["importance"] / totals, 0.0)
    summary = (
        frame.groupby(["feature", "feature_group", "importance_type"], observed=True)
        .agg(
            mean_importance=("normalized_importance", "mean"),
            std_importance=("normalized_importance", "std"),
            minimum_importance=("normalized_importance", "min"),
            maximum_importance=("normalized_importance", "max"),
            fold_count=("normalized_importance", "size"),
        )
        .reset_index()
    )
    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    summary["importance_cv"] = np.where(
        summary["mean_importance"] > 0.0,
        summary["std_importance"] / summary["mean_importance"],
        0.0,
    )
    summary["mean_rank"] = summary.groupby("importance_type", observed=True)[
        "mean_importance"
    ].rank(method="first", ascending=False)
    per_fold_group = (
        frame.groupby(["repeat", "fold", "feature_group", "importance_type"], observed=True)[
            "normalized_importance"
        ]
        .sum()
        .reset_index(name="normalized_importance")
    )
    group_summary = (
        per_fold_group.groupby(["feature_group", "importance_type"], observed=True)
        .agg(
            mean_normalized_importance=("normalized_importance", "mean"),
            std_normalized_importance=("normalized_importance", "std"),
            minimum_normalized_importance=("normalized_importance", "min"),
            maximum_normalized_importance=("normalized_importance", "max"),
            fold_count=("normalized_importance", "size"),
        )
        .reset_index()
    )
    group_summary["std_normalized_importance"] = group_summary["std_normalized_importance"].fillna(
        0.0
    )
    group_summary = group_summary.sort_values(
        ["importance_type", "mean_normalized_importance"], ascending=[True, False]
    )
    dominant = summary.loc[summary["mean_importance"] >= 0.15, "feature"].unique().tolist()
    unstable = (
        summary.loc[
            (summary["mean_importance"] >= 0.005) & (summary["importance_cv"] >= 1.5),
            "feature",
        ]
        .unique()
        .tolist()
    )
    temporal_or_mask = (
        summary.loc[
            summary["feature_group"].isin(
                {"metadata_window", "metadata_missingness", "metadata_position"}
            )
            & summary["mean_rank"].le(10),
            "feature",
        ]
        .unique()
        .tolist()
    )
    audit = {
        "interpretation": "native importance is predictive association, not causal evidence",
        "automatic_removal_performed": False,
        "dominant_features_at_or_above_15_percent": sorted(dominant),
        "unstable_material_features": sorted(unstable),
        "top_ten_time_or_missingness_features": sorted(temporal_or_mask),
        "domain_specificity_status": "deferred_to_phase7_domain_diagnostics",
    }
    return summary, group_summary, audit


def _report_markdown(
    experiment: TabularExperimentConfig,
    result: TabularTrainingResult,
) -> str:
    official = result.report.summary["official_metric"]
    robust = result.report.summary["robust_selection"]
    lines = [
        f"# {experiment.experiment_id}",
        "",
        f"Stage: `{result.stage}`",
        f"Model: `{experiment.model.family}/{experiment.model.name}`",
        f"Feature set: `{experiment.feature_set}` ({len(result.feature_names)} columns)",
        f"Weighting: `{experiment.weighting}`",
        "",
        "## Hypothesis",
        "",
        experiment.hypothesis,
        "",
        "## Metrics",
        "",
        f"- Mean F1 at 0.5: {official['mean_f1']:.6f}",
        f"- Mean ROC-AUC: {official['mean_roc_auc']:.6f}",
        f"- Mean official combined score: {official['mean_combined_score']:.6f}",
        f"- Robust selection score: {robust['score']:.6f}",
        f"- Worst fold: {official['worst_fold_score']:.6f}",
        f"- Worst repeat: {official['worst_repeat_score']:.6f}",
        "",
        (
            "Only full Stage C runs are eligible for Phase 5 selection. Smoke and screening "
            "metrics are engineering or rejection evidence only."
        ),
        "",
    ]
    return "\n".join(lines)


def write_tabular_experiment_artifacts(
    output_dir: Path,
    *,
    project: ProjectConfig,
    experiment: TabularExperimentConfig,
    result: TabularTrainingResult,
) -> ExperimentArtifactManifest:
    """Write the complete ignored Phase 5 contract after successful fold execution."""

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance(project.project_root)
    base_sha = sha256_file(project.source_path)
    summary, group_summary, suspicious = summarize_feature_importance(
        result.feature_importance,
        result.selected_registry,
    )
    resolved = {
        **experiment.resolved_dict(),
        "stage": result.stage,
        "authoritative_validation": {
            "fold_manifest_fingerprint": result.oof.fold_manifest_fingerprint,
            "validation_window_fingerprint": result.oof.validation_window_fingerprint,
            "full_feature_schema_fingerprint": result.full_feature_schema_fingerprint,
            "selected_feature_schema_fingerprint": result.selected_feature_schema_fingerprint,
            "aggregation_method": project.validation.aggregation_method,
            "threshold": project.validation.threshold,
            "n_splits": project.validation.n_splits,
            "n_repeats": project.validation.n_repeats,
        },
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "feature_list.txt").write_text(
        "\n".join(result.feature_names) + "\n", encoding="utf-8"
    )
    feature_schema = {
        "feature_set": experiment.feature_set,
        "feature_count": len(result.feature_names),
        "full_schema_fingerprint": result.full_feature_schema_fingerprint,
        "selected_schema_fingerprint": result.selected_feature_schema_fingerprint,
        "ordered_features": list(result.feature_names),
    }
    _write_json(output_dir / "feature_schema.json", feature_schema)
    _write_json(
        output_dir / "fold_manifest_fingerprint.json",
        {"fingerprint": result.oof.fold_manifest_fingerprint},
    )
    _write_json(
        output_dir / "validation_window_fingerprint.json",
        {"fingerprint": result.oof.validation_window_fingerprint},
    )
    metrics = {
        **result.report.summary,
        "stage": result.stage,
        "selection_eligible": result.stage == "full",
        "phase4_reference": {
            "mean_combined_score": 0.9477106997982762,
            "robust_score": 0.9366465461773669,
            "worst_fold_score": 0.910595851544058,
        },
        "difference_from_phase4_reference": {
            "mean_combined_score": (
                result.report.summary["official_metric"]["mean_combined_score"] - 0.9477106997982762
            ),
            "robust_score": (
                result.report.summary["robust_selection"]["score"] - 0.9366465461773669
            ),
        },
    }
    _write_json(output_dir / "metrics.json", metrics)
    csv_tables = {
        "fold_metrics": result.report.fold_metrics,
        "repeat_metrics": result.report.repeat_metrics,
        "slice_metrics": result.report.slice_metrics,
        "prediction_stability": result.report.prediction_stability,
        "oof_predictions": result.oof.original,
        "window_predictions": result.oof.windows,
        "feature_importance": result.feature_importance,
        "feature_importance_summary": summary,
        "feature_group_importance": group_summary,
        "permutation_importance": result.permutation_importance,
    }
    for name, frame in csv_tables.items():
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
    _write_json(output_dir / "suspicious_feature_audit.json", suspicious)
    fold_models = {
        "schema_version": 1,
        "model_family": experiment.model.family,
        "profile": experiment.model.name,
        "folds": [fold.as_dict() for fold in result.folds],
        "median_best_iteration": int(np.median([fold.best_iteration for fold in result.folds])),
    }
    _write_json(output_dir / "fold_models_manifest.json", fold_models)
    runtime = {
        "runtime_seconds": result.runtime_seconds,
        "peak_rss_megabytes": result.peak_rss_megabytes,
        "peak_rss_measurement_scope": (
            "process_lifetime upper bound; cumulative when multiple experiments share one CLI"
        ),
        "cpu_threads_per_model": project.tabular.cpu_threads,
        "fold_training_seconds": float(sum(fold.training_seconds for fold in result.folds)),
        "fold_inference_seconds": float(sum(fold.inference_seconds for fold in result.folds)),
        "catboost_version": _package_version("catboost"),
        "lightgbm_version": _package_version("lightgbm"),
    }
    _write_json(output_dir / "runtime.json", runtime)
    (output_dir / "report.md").write_text(_report_markdown(experiment, result), encoding="utf-8")
    manifest = ExperimentArtifactManifest(
        schema_version=1,
        status="complete",
        stage=result.stage,
        experiment_id=experiment.experiment_id,
        experiment_config_fingerprint=experiment.fingerprint,
        base_config_sha256=base_sha,
        git_commit=provenance["commit"],
        tracked_files_dirty=provenance["tracked_files_dirty"],
        fold_manifest_fingerprint=result.oof.fold_manifest_fingerprint,
        validation_window_fingerprint=result.oof.validation_window_fingerprint,
        full_feature_schema_fingerprint=result.full_feature_schema_fingerprint,
        selected_feature_schema_fingerprint=result.selected_feature_schema_fingerprint,
        oof_fingerprint=result.oof.fingerprint,
        original_oof_rows=result.oof.original.shape[0],
        window_prediction_rows=result.oof.windows.shape[0],
    )
    _write_json(output_dir / "experiment_manifest.json", manifest.as_dict())
    return manifest


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))
