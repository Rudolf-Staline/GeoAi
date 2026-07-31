"""Internal Phase 8 delivery helpers kept separate from the final orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from geoai_aquaculture.training import (
    execute_tabular_experiment,
    load_tabular_experiment_config,
    load_temporal_experiment_config,
)

from .calibration import CalibrationEvaluation, crossfit_calibration
from .config import FinalCandidateConfig, FinalDeliveryConfig
from .final_fit import FinalModelPrediction, sha256_file
from .oof import FinalCandidate, WeightSelectionResult


class FinalDeliveryError(ValueError):
    """Raised when the final pipeline is incomplete or scientifically incompatible."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _report_dict(report) -> dict[str, Any]:
    return report.summary


def _dependency_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "torch", "shap")
    values: dict[str, str | None] = {}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    values["python"] = sys.version.split()[0]
    return values


def _ensure_validation(project, final_config: FinalDeliveryConfig) -> None:
    required = (
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.tabular.validation_artifacts_dir / "validation_window_manifest.csv",
    )
    if all(path.is_file() for path in required):
        return
    command = [
        sys.executable,
        str(project.project_root / "scripts" / "build_validation.py"),
        "--config",
        str(final_config.project_config),
    ]
    subprocess.run(command, cwd=project.project_root, check=True)
    if not all(path.is_file() for path in required):
        raise FinalDeliveryError("validation prerequisite generation did not create required files")


def _ensure_candidate_artifact(
    project,
    final_config: FinalDeliveryConfig,
    candidate: FinalCandidateConfig,
) -> None:
    manifest = candidate.artifact_dir / "experiment_manifest.json"
    if manifest.is_file():
        return
    if not final_config.rebuild_missing_oof:
        raise FinalDeliveryError(f"candidate OOF artifact is missing: {candidate.artifact_dir}")
    if candidate.kind == "tree":
        experiment = load_tabular_experiment_config(candidate.experiment_config)
        execute_tabular_experiment(
            project,
            experiment,
            stage="full",
            overwrite=final_config.overwrite_oof,
        )
    else:
        try:
            from geoai_aquaculture.training.temporal import execute_temporal_experiment
        except ModuleNotFoundError as exc:
            raise FinalDeliveryError(
                "PyTorch is required to rebuild the temporal OOF artifact"
            ) from exc
        experiment = load_temporal_experiment_config(candidate.experiment_config)
        execute_temporal_experiment(
            project,
            experiment,
            stage="full",
            overwrite=final_config.overwrite_oof,
        )
    if not manifest.is_file():
        raise FinalDeliveryError(
            f"candidate OOF artifact remains incomplete: {candidate.artifact_dir}"
        )


def ensure_final_oof_artifacts(project, final_config: FinalDeliveryConfig) -> None:
    """Build only the accepted OOF experts when their ignored artifacts are absent."""

    _ensure_validation(project, final_config)
    for candidate in final_config.candidates:
        _ensure_candidate_artifact(project, final_config, candidate)


def _candidate_registry(candidates: tuple[FinalCandidate, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        artifact = candidate.declaration.artifact_dir
        records.append(
            {
                "experiment_id": candidate.experiment_id,
                "kind": candidate.declaration.kind,
                "role": candidate.declaration.role,
                "artifact_dir": artifact.as_posix(),
                "oof_path": (artifact / "oof_predictions.csv").as_posix(),
                "window_oof_path": (artifact / "window_predictions.csv").as_posix(),
                "oof_sha256": sha256_file(artifact / "oof_predictions.csv"),
                "window_oof_sha256": sha256_file(artifact / "window_predictions.csv"),
                "oof_fingerprint": candidate.oof.fingerprint,
                "fold_manifest_fingerprint": candidate.oof.fold_manifest_fingerprint,
                "validation_window_fingerprint": candidate.oof.validation_window_fingerprint,
                "official_score": float(
                    candidate.metrics["official_metric"]["mean_combined_score"]
                ),
                "robust_score": float(candidate.metrics["robust_selection"]["score"]),
            }
        )
    return records


def _select_calibration(
    weight_result: WeightSelectionResult,
    project,
    methods: tuple[str, ...],
) -> tuple[CalibrationEvaluation, tuple[CalibrationEvaluation, ...]]:
    evaluations = tuple(
        crossfit_calibration(
            weight_result.production.oof,
            project,
            method,  # type: ignore[arg-type]
        )
        for method in methods
    )
    baseline = next(item for item in evaluations if item.method == "none")
    baseline_robust = float(baseline.report.summary["robust_selection"]["score"])
    baseline_combined = float(baseline.report.summary["official_metric"]["mean_combined_score"])
    eligible = [baseline]
    for item in evaluations:
        if item.method == "none":
            continue
        robust = float(item.report.summary["robust_selection"]["score"])
        combined = float(item.report.summary["official_metric"]["mean_combined_score"])
        if robust >= baseline_robust + 0.0001 and combined >= baseline_combined - 0.00005:
            eligible.append(item)
    eligible.sort(
        key=lambda item: (
            item.method != "none",
            -float(item.report.summary["robust_selection"]["score"]),
        )
    )
    # Simplicity wins unless calibration provides a predeclared material robust improvement.
    selected = (
        baseline
        if len(eligible) == 1
        else max(
            eligible,
            key=lambda item: float(item.report.summary["robust_selection"]["score"]),
        )
    )
    return selected, evaluations


def _prior_shift_em(probabilities: np.ndarray, source_prior: float) -> dict[str, Any]:
    """Return an unselected EM prior-shift diagnostic without altering predictions."""

    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if not 0.0 < source_prior < 1.0:
        raise FinalDeliveryError("source prior must lie in (0, 1)")
    estimate = float(np.mean(p))
    iterations = 0
    for iteration in range(1, 101):
        iterations = iteration
        positive = p * (estimate / source_prior)
        negative = (1.0 - p) * ((1.0 - estimate) / (1.0 - source_prior))
        posterior = positive / np.maximum(positive + negative, 1e-12)
        updated = float(np.mean(posterior))
        if abs(updated - estimate) < 1e-8:
            estimate = updated
            break
        estimate = updated
    return {
        "method": "saerens_em_diagnostic",
        "source_prior": source_prior,
        "estimated_test_prior": estimate,
        "iterations": iterations,
        "applied": False,
        "reason": (
            "Not applied: Phase 7 found severe covariate shift, so a pure prior-shift assumption "
            "is not credible enough for production correction."
        ),
    }


def _word_count(text: str) -> int:
    return len(text.replace("-", " ").split())


def _trustworthiness_text(
    *,
    top_features: list[str],
    tree: FinalModelPrediction,
    temporal: FinalModelPrediction,
    domain_auc: float = 0.991376,
) -> tuple[str, dict[str, int]]:
    feature_text = ", ".join(top_features[:3]) if top_features else "water-index aggregates"
    final_seconds = tree.training_seconds + temporal.training_seconds
    energy_kwh = final_seconds / 3600.0 * 0.045
    carbon_kg = energy_kwh * 0.5
    sections = {
        "Data & Model Bias": (
            f"Adversarial validation separated masked training from test with ROC-AUC "
            f"{domain_auc:.3f}, showing strong temporal and environmental shift, mainly in "
            "optical indices and radar ratios. We evaluated test-like holdouts, month/window "
            "slices, optical-gap groups and sensor-only experts. Removing domain-specific "
            "variables and clipped importance weighting did not improve robust validation, so "
            "both were rejected. Regional coordinates were unavailable, preventing direct "
            "spatial-group checks. Performance may therefore weaken in unseen seasons, cloud "
            "regimes or aquaculture practices."
        ),
        "Model Transparency": (
            "We used LightGBM gain importance and SHAP on held-out test rows. Leading signals "
            f"included {feature_text}. Sensor ablations on the GRU showed that optical bands and "
            "monthly water indices carried most information, while radar still supplied useful "
            "conditional evidence. A notable finding was the dominance of maximum NDWI-like "
            "features, which were also domain-sensitive; we therefore reported this risk rather "
            "than interpreting importance causally. Per-row tree/GRU probabilities and "
            "disagreement are retained for audit."
        ),
        "Approach Reusability": (
            "The pipeline is configuration-driven and separates schema parsing, mask-aware "
            "windows, physical features, grouped validation, model adapters and submission "
            "assembly. It can be reused for other short incomplete satellite time-series "
            "classification tasks by changing band semantics and labels. Native missingness "
            "masks and CPU-compatible models improve portability. Reuse outside aquaculture "
            "still requires new labeled data, domain-specific indices and validation slices; "
            "the learned models themselves should not be transferred without retraining."
        ),
        "Sustainability and Efficiency": (
            "Final fitting used one LightGBM and a 26,329-parameter CPU GRU; rejected model "
            f"families and adaptations were not repeated. Final training consumed about "
            f"{final_seconds:.0f} CPU-seconds. Under an explicit 45 W CPU assumption, this is "
            f"roughly {energy_kwh:.4f} kWh or {carbon_kg:.4f} kg CO2e using a hypothetical "
            "0.5 kg CO2e/kWh factor. The estimate excludes earlier research runs. Deterministic "
            "folds, early stopping, small manual grids and compact models reduced unnecessary "
            "computation."
        ),
    }
    counts = {name: _word_count(text) for name, text in sections.items()}
    if any(count > 100 for count in counts.values()):
        raise FinalDeliveryError(f"trustworthiness section exceeds 100 words: {counts}")
    markdown = (
        "# Trustworthiness Evaluation\n\n"
        + "\n\n".join(
            f"## {name} ({counts[name]} words)\n\n{text}" for name, text in sections.items()
        )
        + "\n"
    )
    return markdown, counts


def _generate_notebook(project_root: Path) -> Path:
    import nbformat

    notebook_dir = project_root / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    path = notebook_dir / "99_final_submission.ipynb"
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# GeoAI Aquaculture Pond Identification — Final Submission\n\n"
            "Place `Train.csv`, `Test.csv`, and `SampleSubmission.csv` in `data/raw/`, "
            "then install the validated extras with "
            '`python -m pip install -e ".[dev,trees,deep,notebook]"`. This notebook '
            "only orchestrates tested project modules; it does not duplicate feature, "
            "validation, model, ensemble, calibration, or submission logic."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from geoai_aquaculture.constants import FIXED_THRESHOLD\n"
            "from geoai_aquaculture.data import load_project_config\n"
            "from geoai_aquaculture.ensemble import (\n"
            "    build_final_delivery,\n"
            "    load_final_delivery_config,\n"
            ")\n"
            "from geoai_aquaculture.submission import validate_submission\n"
            "\n"
            "final_config = load_final_delivery_config(Path('configs/final.yaml'))\n"
            "project = load_project_config(final_config.project_config)\n"
            "assert final_config.threshold == FIXED_THRESHOLD == 0.5\n"
            "print({\n"
            "    'validation_seed': project.validation.seed,\n"
            "    'folds': project.validation.n_splits,\n"
            "    'repeats': project.validation.n_repeats,\n"
            "    'threshold': final_config.threshold,\n"
            "    'config_fingerprint': final_config.fingerprint,\n"
            "})"
        ),
        nbformat.v4.new_code_cell(
            "result = build_final_delivery(final_config, project=project, reuse_existing=True)\n"
            "result"
        ),
        nbformat.v4.new_code_cell(
            "submission = pd.read_csv(result.submission_path)\n"
            "sample = pd.read_csv(project.data.sample_submission_path)\n"
            "validate_submission(submission, sample)\n"
            "metrics = json.loads((result.output_dir / 'metrics.json').read_text())\n"
            "models = json.loads((result.output_dir / 'full_data_models.json').read_text())\n"
            "display(submission.head())\n"
            "display(metrics['selected_oof'])\n"
            "display(models)"
        ),
        nbformat.v4.new_code_cell(
            "trustworthiness = (result.output_dir / 'trustworthiness.md').read_text()\n"
            "print(trustworthiness)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Audit trail\n\n"
            "The final directory contains candidate and OOF hashes, nested blend weights, "
            "calibration comparisons, model hashes, SHAP outputs, per-row tree/GRU "
            "disagreement, prior-shift diagnostics, runtime metadata, trustworthiness "
            "responses, and the exact submission hash."
        ),
    ]
    nbformat.write(notebook, path)
    return path
