"""Phase 8 orchestration: OOF selection, final fitting, submission, and delivery artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.data import git_provenance, load_competition_data, load_project_config
from geoai_aquaculture.submission import build_submission, validate_submission

from .config import FinalDeliveryConfig, load_final_delivery_config
from .delivery_support import (
    FinalDeliveryError,
    _candidate_registry,
    _dependency_versions,
    _generate_notebook,
    _prior_shift_em,
    _report_dict,
    _select_calibration,
    _trustworthiness_text,
    _write_json,
    ensure_final_oof_artifacts,
)
from .final_fit import fit_final_temporal, fit_final_tree, sha256_file
from .oof import learn_nested_weight, load_final_candidates


@dataclass(frozen=True, slots=True)
class FinalDeliveryResult:
    """Paths and decisions from one completed final delivery."""

    output_dir: Path
    submission_path: Path
    notebook_path: Path
    selected_tree_weight: float
    selected_calibration: str
    oof_combined_score: float
    oof_robust_score: float
    manifest_path: Path


def _existing_result(config: FinalDeliveryConfig) -> FinalDeliveryResult | None:
    manifest = config.output_dir / "final_manifest.json"
    submission = config.output_dir / "submission.csv"
    notebook = config.output_dir.parent.parent / "notebooks" / "99_final_submission.ipynb"
    if not (manifest.is_file() and submission.is_file() and notebook.is_file()):
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("final_config_fingerprint") != config.fingerprint:
        return None
    project_root = config.source_path.parent.parent
    provenance = git_provenance(project_root)
    current_commit = provenance.get("commit")
    if current_commit and payload.get("source_commit") != current_commit:
        return None
    if provenance.get("tracked_files_dirty"):
        return None
    return FinalDeliveryResult(
        output_dir=config.output_dir,
        submission_path=submission,
        notebook_path=notebook,
        selected_tree_weight=float(payload["selection"]["tree_weight"]),
        selected_calibration=str(payload["selection"]["calibration"]),
        oof_combined_score=float(payload["selection"]["oof_combined_score"]),
        oof_robust_score=float(payload["selection"]["oof_robust_score"]),
        manifest_path=manifest,
    )


def build_final_delivery(
    config: FinalDeliveryConfig,
    *,
    project=None,
    reuse_existing: bool = False,
    overwrite: bool = False,
) -> FinalDeliveryResult:
    """Run the complete evidence-based final pipeline and write an immutable submission."""

    if reuse_existing:
        existing = _existing_result(config)
        if existing is not None:
            return existing
    project = project or load_project_config(config.project_config)
    output = config.output_dir
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FinalDeliveryError(
                f"final output already exists at {output}; use reuse or explicit overwrite"
            )
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "resolved_final_config.json", config.resolved_dict())
    started = perf_counter()
    ensure_final_oof_artifacts(project, config)
    candidates = load_final_candidates(config, project)
    tree_candidate = next(item for item in candidates if item.declaration.kind == "tree")
    temporal_candidate = next(item for item in candidates if item.declaration.kind == "temporal")
    weight_result = learn_nested_weight(
        tree_candidate,
        temporal_candidate,
        project,
        grid_step=config.weight_grid_step,
        fixed_tree_weights=config.fixed_tree_weights,
    )
    selected_calibration, calibration_evaluations = _select_calibration(
        weight_result,
        project,
        config.calibration_methods,
    )
    selected_calibration.oof.original.to_csv(
        output / "selected_oof_predictions.csv", index=False
    )
    selected_calibration.oof.windows.to_csv(
        output / "selected_window_oof_predictions.csv", index=False
    )
    _write_json(
        output / "ensemble_weights.json",
        {
            "selected_tree_weight": weight_result.production.tree_weight,
            "selected_temporal_weight": weight_result.production.temporal_weight,
            "selected_calibration": selected_calibration.method,
            "nested_median_tree_weight": weight_result.crossfit.tree_weight,
            "selection_rule": (
                "maximize immutable robust score; prefer equal weights when within 0.0002"
            ),
        },
    )

    tree_final = fit_final_tree(project, config.tree_candidate, output_dir=output)
    temporal_final = fit_final_temporal(project, config.temporal_candidate, output_dir=output)
    if tree_final.probabilities.shape != temporal_final.probabilities.shape:
        raise FinalDeliveryError("final tree and temporal test rows do not align")
    tree_weight = weight_result.production.tree_weight
    raw_blend = (
        tree_weight * tree_final.probabilities
        + (1.0 - tree_weight) * temporal_final.probabilities
    )
    final_probability = selected_calibration.production_calibrator.transform(raw_blend)
    data = load_competition_data(project)
    ids = data.sample_submission[project.data.id_column]
    submission = build_submission(ids, final_probability)
    submission[project.data.id_column] = submission[project.data.id_column].astype(
        data.sample_submission[project.data.id_column].dtype
    )
    validate_submission(submission, data.sample_submission)
    submission_path = output / "submission.csv"
    submission.to_csv(submission_path, index=False, float_format="%.17g", lineterminator="\n")

    disagreement = (
        (tree_final.probabilities >= 0.5) != (temporal_final.probabilities >= 0.5)
    )
    uncertainty = pd.DataFrame(
        {
            "ID": ids.astype(str).to_numpy(),
            "tree_probability": tree_final.probabilities,
            "temporal_probability": temporal_final.probabilities,
            "raw_blend_probability": raw_blend,
            "final_probability": final_probability,
            "probability_mean": np.mean(
                np.column_stack((tree_final.probabilities, temporal_final.probabilities)), axis=1
            ),
            "probability_standard_deviation": np.std(
                np.column_stack((tree_final.probabilities, temporal_final.probabilities)),
                axis=1,
                ddof=0,
            ),
            "probability_minimum": np.minimum(
                tree_final.probabilities, temporal_final.probabilities
            ),
            "probability_maximum": np.maximum(
                tree_final.probabilities, temporal_final.probabilities
            ),
            "binary_disagreement": disagreement,
            "tree_contribution": tree_weight * tree_final.probabilities,
            "temporal_contribution": (1.0 - tree_weight) * temporal_final.probabilities,
        }
    )
    uncertainty.to_csv(output / "test_prediction_uncertainty.csv", index=False)
    pd.DataFrame(
        {
            "ID": ids.astype(str).to_numpy(),
            "TargetRAUC": final_probability,
            "tree_probability": tree_final.probabilities,
            "temporal_probability": temporal_final.probabilities,
        }
    ).to_csv(output / "test_probabilities.csv", index=False)

    registry = _candidate_registry(candidates)
    _write_json(output / "candidate_registry.json", registry)
    weight_result.fold_weights.to_csv(output / "nested_fold_weights.csv", index=False)
    blend_metrics = {
        item.label: {
            "tree_weight": item.tree_weight,
            "temporal_weight": item.temporal_weight,
            "summary": _report_dict(item.report),
        }
        for item in (*weight_result.alternatives, weight_result.crossfit)
    }
    _write_json(output / "ensemble_metrics.json", blend_metrics)
    calibration_payload: dict[str, Any] = {}
    for item in calibration_evaluations:
        calibration_payload[item.method] = {
            "summary": _report_dict(item.report),
            "expected_calibration_error": item.expected_calibration_error,
            "production_calibrator": item.production_calibrator.as_dict(),
            "selected": item.method == selected_calibration.method,
        }
        if not item.fold_parameters.empty:
            item.fold_parameters.to_json(
                output / f"calibration_{item.method}_folds.json",
                orient="records",
                indent=2,
            )
    _write_json(output / "calibration_metrics.json", calibration_payload)
    _write_json(
        output / "calibration_model.json",
        selected_calibration.production_calibrator.as_dict(),
    )

    source_prior = float(data.train[project.data.target_column].mean())
    prior_shift = _prior_shift_em(final_probability, source_prior)
    prior_shift["correction_enabled_in_config"] = config.prior_shift_correction_enabled
    if config.prior_shift_correction_enabled:
        raise FinalDeliveryError(
            "prior-shift correction is intentionally disabled because Phase 7 invalidated "
            "pure prior shift"
        )
    _write_json(output / "prior_shift_diagnostic.json", prior_shift)

    model_manifest = {
        "tree": {
            "experiment_id": tree_final.experiment_id,
            "family": tree_final.model_family,
            "path": tree_final.model_path.relative_to(output).as_posix(),
            "sha256": tree_final.model_sha256,
            "iterations": tree_final.training_parameter,
            "training_seconds": tree_final.training_seconds,
            "inference_seconds": tree_final.inference_seconds,
            "metadata": tree_final.metadata,
        },
        "temporal": {
            "experiment_id": temporal_final.experiment_id,
            "family": temporal_final.model_family,
            "path": temporal_final.model_path.relative_to(output).as_posix(),
            "sha256": temporal_final.model_sha256,
            "epochs": temporal_final.training_parameter,
            "training_seconds": temporal_final.training_seconds,
            "inference_seconds": temporal_final.inference_seconds,
            "metadata": temporal_final.metadata,
        },
    }
    _write_json(output / "full_data_models.json", model_manifest)

    top_features: list[str] = []
    shap_path = output / "lightgbm_shap_importance.csv"
    if shap_path.is_file():
        top_features = pd.read_csv(shap_path)["feature"].head(3).astype(str).tolist()
    trust_text, trust_counts = _trustworthiness_text(
        top_features=top_features,
        tree=tree_final,
        temporal=temporal_final,
    )
    (output / "trustworthiness.md").write_text(trust_text, encoding="utf-8")
    _write_json(output / "trustworthiness_word_counts.json", trust_counts)

    selected_summary = selected_calibration.report.summary
    metrics = {
        "selected_oof": selected_summary,
        "selected_tree_weight": tree_weight,
        "selected_temporal_weight": 1.0 - tree_weight,
        "selected_calibration": selected_calibration.method,
        "test_positive_prediction_rate": float((final_probability >= 0.5).mean()),
        "test_probability_mean": float(final_probability.mean()),
        "test_model_disagreement_rate": float(disagreement.mean()),
        "tta": {
            "enabled": config.tta_enabled,
            "applied": False,
            "reason": "No Phase 7 OOF evidence justified test-time augmentation.",
        },
        "domain_shift": {
            "full_representation_auc": 0.991376,
            "adaptation_retained": False,
        },
    }
    _write_json(output / "metrics.json", metrics)

    notebook_path = _generate_notebook(project.project_root)
    provenance = git_provenance(project.project_root)
    runtime_commit = provenance.get("commit") or os.environ.get("GITHUB_SHA")
    source_commit = runtime_commit or config.source_commit
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "final_config_fingerprint": config.fingerprint,
        "source_commit": source_commit,
        "source_commit_basis": (
            "runtime_git_head" if runtime_commit else "approved_base_commit_fallback"
        ),
        "source_base_commit": config.source_commit,
        "tracked_files_dirty": provenance.get("tracked_files_dirty"),
        "fold_manifest_fingerprint": project.tabular.fold_manifest_fingerprint,
        "validation_window_fingerprint": project.tabular.validation_window_fingerprint,
        "selection": {
            "tree_weight": tree_weight,
            "temporal_weight": 1.0 - tree_weight,
            "calibration": selected_calibration.method,
            "oof_combined_score": float(
                selected_summary["official_metric"]["mean_combined_score"]
            ),
            "oof_robust_score": float(selected_summary["robust_selection"]["score"]),
        },
        "submission": {
            "path": submission_path.relative_to(output).as_posix(),
            "sha256": sha256_file(submission_path),
            "rows": int(submission.shape[0]),
            "columns": submission.columns.tolist(),
        },
        "models": model_manifest,
        "candidate_registry_sha256": sha256_file(output / "candidate_registry.json"),
        "resolved_final_config_sha256": sha256_file(output / "resolved_final_config.json"),
        "selected_oof_sha256": sha256_file(output / "selected_oof_predictions.csv"),
        "selected_window_oof_sha256": sha256_file(
            output / "selected_window_oof_predictions.csv"
        ),
        "runtime_seconds": perf_counter() - started,
        "versions": _dependency_versions(),
    }
    manifest_path = output / "final_manifest.json"
    _write_json(manifest_path, manifest)
    (output / "reproducibility.md").write_text(
        "# Reproduction\n\n"
        "```bash\n"
        "python -m pip install -e \".[dev,trees,deep,notebook]\"\n"
        "python scripts/build_submission.py --config configs/final.yaml\n"
        "python scripts/validate_submission.py --submission artifacts/final/submission.csv\n"
        "pytest\nruff check .\nruff format --check .\n"
        "```\n\n"
        "Place the supplied competition files in `data/raw/` before running. Core logic is in "
        "tested modules; the notebook is an orchestration layer.\n",
        encoding="utf-8",
    )
    return FinalDeliveryResult(
        output_dir=output,
        submission_path=submission_path,
        notebook_path=notebook_path,
        selected_tree_weight=tree_weight,
        selected_calibration=selected_calibration.method,
        oof_combined_score=float(selected_summary["official_metric"]["mean_combined_score"]),
        oof_robust_score=float(selected_summary["robust_selection"]["score"]),
        manifest_path=manifest_path,
    )
