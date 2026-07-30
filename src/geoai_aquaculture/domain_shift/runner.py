"""End-to-end bounded Phase 7 diagnostics and controlled adaptations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from geoai_aquaculture.data import ProjectConfig
from geoai_aquaculture.features import FeatureRegistry
from geoai_aquaculture.training import load_tabular_experiment_config
from geoai_aquaculture.validation import build_similarity_holdout_manifest

from .adaptation import AdaptationResult, run_label_adaptation, write_adaptation_result
from .adversarial import DomainValidationResult, run_adversarial_validation
from .artifacts import (
    write_decision_report,
    write_domain_result,
    write_phase7_summary,
)
from .config import Phase7Config
from .dataset import DomainDataset, build_domain_dataset, build_domain_feature_panels
from .diagnostics import (
    feature_shift_table,
    grouped_feature_importance,
    representation_summary,
)
from .evaluation import (
    HoldoutEvaluation,
    build_importance_weights,
    evaluate_similarity_holdout,
    load_original_oof,
)


@dataclass(frozen=True, slots=True)
class Phase7RunResult:
    """Complete diagnostic and adaptation decision payload."""

    representation_results: tuple[DomainValidationResult, ...]
    holdout_evaluations: tuple[HoldoutEvaluation, ...]
    adaptations: tuple[AdaptationResult, ...]
    decisions: tuple[dict[str, Any], ...]
    report_path: Path


def _modified_domain_dataset(
    dataset: DomainDataset,
    removed_features: tuple[str, ...],
) -> DomainDataset:
    retained = tuple(name for name in dataset.feature_names if name not in set(removed_features))
    definitions = tuple(
        definition
        for definition in dataset.registry.definitions
        if definition.name in set(retained)
    )
    registry = FeatureRegistry(definitions)
    groups: dict[str, list[str]] = {}
    for definition in definitions:
        groups.setdefault(definition.feature_group, []).append(definition.name)
    digest = hashlib.sha256()
    digest.update(dataset.fingerprint.encode())
    digest.update("\x1f".join(retained).encode())
    return DomainDataset(
        representation=dataset.representation,
        features=dataset.features.loc[:, list(retained)].copy(),
        feature_names=retained,
        registry=registry,
        metadata=dataset.metadata.copy(),
        labels=dataset.labels.copy(),
        groups=dataset.groups.copy(),
        entity_ids=dataset.entity_ids.copy(),
        sample_weights=dataset.sample_weights.copy(),
        feature_groups=MappingProxyType(
            {name: tuple(values) for name, values in groups.items()}
        ),
        schema_fingerprint=digest.hexdigest(),
        fingerprint=digest.hexdigest(),
    )


def _load_baseline_metrics(config: Phase7Config) -> dict[str, float]:
    path = config.tree_oof_artifact.parent / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "combined": float(payload["official_metric"]["mean_combined_score"]),
        "robust": float(payload["robust_selection"]["score"]),
        "worst_fold": float(payload["official_metric"]["worst_fold_score"]),
        "worst_season": float(payload["robust_selection"]["components"]["worst_season_score"]),
    }


def _adaptation_decision(
    method: str,
    results: list[AdaptationResult],
    holdouts: list[HoldoutEvaluation],
    baseline: dict[str, float],
    baseline_oof: pd.DataFrame,
    *,
    selected_domain_auc: float,
    reduced_domain_auc: float | None,
) -> dict[str, Any]:
    robust = np.asarray(
        [result.report.summary["robust_selection"]["score"] for result in results],
        dtype=np.float64,
    )
    combined = np.asarray(
        [result.report.summary["official_metric"]["mean_combined_score"] for result in results],
        dtype=np.float64,
    )
    worst_season = np.asarray(
        [
            result.report.summary["robust_selection"]["components"]["worst_season_score"]
            for result in results
        ],
        dtype=np.float64,
    )
    holdout_scores = np.asarray(
        [evaluation.mean_combined_score for evaluation in holdouts], dtype=np.float64
    )
    mean_robust = float(np.mean(robust))
    mean_combined = float(np.mean(combined))
    all_above = bool(np.all(robust >= baseline["robust"] + 0.0002))
    stable = bool(np.min(robust) >= baseline["robust"] - 0.0003)
    season_safe = bool(np.min(worst_season) >= baseline["worst_season"] - 0.0010)
    domain_mitigation_failed = (
        method == "feature_removal"
        and reduced_domain_auc is not None
        and reduced_domain_auc >= selected_domain_auc - 0.002
    )
    if all_above and stable and season_safe:
        decision = "accept"
        rationale = "Both predeclared seeds improve robust OOF without material seasonal harm."
    elif domain_mitigation_failed and mean_robust < baseline["robust"] + 0.0002:
        decision = "reject"
        rationale = (
            "Removing domain-important features neither reduces domain separability nor produces "
            "a practically meaningful robust gain."
        )
    elif mean_robust < baseline["robust"] - 0.0002 or not season_safe:
        decision = "reject"
        rationale = "The adaptation lowers mean robust OOF or damages the weakest seasonal slice."
    else:
        decision = "inconclusive"
        rationale = "Changes are within the predefined practical-equivalence band across seeds."
    correlations: list[float] = []
    disagreements: list[float] = []
    reference = baseline_oof.sort_values(["repeat", "original_id"], kind="stable")
    for result in results:
        candidate = result.oof.original.sort_values(["repeat", "original_id"], kind="stable")
        if not reference[["repeat", "original_id"]].reset_index(drop=True).equals(
            candidate[["repeat", "original_id"]].reset_index(drop=True)
        ):
            raise ValueError("adaptation and baseline OOF rows do not align")
        reference_probability = reference["probability"].to_numpy(dtype=np.float64)
        candidate_probability = candidate["probability"].to_numpy(dtype=np.float64)
        correlations.append(float(np.corrcoef(reference_probability, candidate_probability)[0, 1]))
        disagreements.append(
            float(np.mean((reference_probability >= 0.5) != (candidate_probability >= 0.5)))
        )
    return {
        "method": method,
        "decision": decision,
        "seeds": [result.seed for result in results],
        "robust_scores": robust.tolist(),
        "combined_scores": combined.tolist(),
        "holdout_combined_scores": holdout_scores.tolist(),
        "mean_robust_score": mean_robust,
        "mean_combined_score": mean_combined,
        "baseline_robust_score": baseline["robust"],
        "baseline_combined_score": baseline["combined"],
        "selected_domain_auc": selected_domain_auc,
        "reduced_domain_auc": reduced_domain_auc,
        "mean_probability_correlation_with_baseline": float(np.mean(correlations)),
        "mean_binary_disagreement_with_baseline": float(np.mean(disagreements)),
        "production_recommendation": "retain" if decision == "accept" else "do_not_retain",
        "rationale": rationale,
    }


def run_phase7(
    project: ProjectConfig,
    config: Phase7Config,
    *,
    run_adaptations: bool = True,
) -> Phase7RunResult:
    """Run all declared diagnostics and optionally the bounded adaptation matrix."""

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    panels = build_domain_feature_panels(project)
    datasets: dict[str, DomainDataset] = {}
    results: list[DomainValidationResult] = []
    for offset, representation in enumerate(config.representations):
        dataset = build_domain_dataset(panels, project, representation)
        result = run_adversarial_validation(
            dataset,
            config.domain_model,
            seed=config.seed + offset * 1009,
            n_splits=config.n_splits,
            cpu_threads=project.tabular.cpu_threads,
        )
        write_domain_result(output, result)
        datasets[representation] = dataset
        results.append(result)
    selected_result = next(
        result for result in results if result.representation == config.selection_representation
    )
    selected_dataset = datasets[config.selection_representation]
    feature_importance, group_importance = grouped_feature_importance(
        selected_result, selected_dataset.registry
    )
    shift = feature_shift_table(
        selected_dataset,
        feature_importance,
        feature_limit=config.top_feature_count,
    )
    holdout = build_similarity_holdout_manifest(
        selected_result.train_similarity_scores,
        panels.folds,
        fraction=config.similarity_holdout_fraction,
        minimum_samples=config.similarity_holdout_minimum,
    )
    importance_weights = build_importance_weights(
        selected_result.train_similarity_scores,
        minimum=config.importance_weights.minimum,
        maximum=config.importance_weights.maximum,
    )
    tree_oof = load_original_oof(config.tree_oof_artifact, panels.folds)
    temporal_oof = load_original_oof(config.temporal_oof_artifact, panels.folds)
    base_holdouts = (
        evaluate_similarity_holdout(tree_oof, holdout, model_name="tree_baseline"),
        evaluate_similarity_holdout(temporal_oof, holdout, model_name="temporal_baseline"),
    )
    adaptation_results: list[AdaptationResult] = []
    decisions: list[dict[str, Any]] = []
    if run_adaptations:
        baseline_config = load_tabular_experiment_config(config.label_baseline_config)
        removed = tuple(
            feature_importance.head(config.removal_feature_count)["feature"].astype(str)
        )
        reduced_dataset = _modified_domain_dataset(selected_dataset, removed)
        reduced_domain = run_adversarial_validation(
            reduced_dataset,
            config.domain_model,
            seed=config.seed + 90_001,
            n_splits=config.n_splits,
            cpu_threads=project.tabular.cpu_threads,
        )
        write_domain_result(
            output, replace_representation(reduced_domain, "full_removed_top_domain")
        )
        baseline = _load_baseline_metrics(config)
        for method in config.adaptation_methods:
            method_results: list[AdaptationResult] = []
            method_holdouts: list[HoldoutEvaluation] = []
            for seed in config.adaptation_seeds:
                result = run_label_adaptation(
                    project,
                    baseline_config,
                    method=method,
                    seed=seed,
                    removed_features=removed if method == "feature_removal" else (),
                    importance_weights=(
                        importance_weights if method == "importance_weighting" else None
                    ),
                )
                write_adaptation_result(output, result)
                method_results.append(result)
                adaptation_results.append(result)
                method_holdouts.append(
                    evaluate_similarity_holdout(
                        result.oof.original,
                        holdout,
                        model_name=f"{method}_seed_{seed}",
                    )
                )
            decisions.append(
                _adaptation_decision(
                    method,
                    method_results,
                    method_holdouts,
                    baseline,
                    tree_oof,
                    selected_domain_auc=selected_result.metrics.roc_auc,
                    reduced_domain_auc=(
                        reduced_domain.metrics.roc_auc if method == "feature_removal" else None
                    ),
                )
            )
    summaries = tuple(representation_summary(result) for result in results)
    all_holdouts = list(base_holdouts)
    for result in adaptation_results:
        all_holdouts.append(
            evaluate_similarity_holdout(
                result.oof.original,
                holdout,
                model_name=f"{result.method}_seed_{result.seed}",
            )
        )
    write_phase7_summary(
        output,
        config,
        summaries,
        feature_importance=feature_importance,
        group_importance=group_importance,
        feature_shift=shift,
        similarity_scores=selected_result.train_similarity_scores,
        similarity_holdout=holdout.frame,
        importance_weights=importance_weights,
        holdout_evaluations=tuple(all_holdouts),
        decisions=decisions,
        project_root=project.project_root,
    )
    report_payload = {
        "selected_representation": selected_result.representation,
        "selected_domain_auc": selected_result.metrics.roc_auc,
        "representation_metrics": [asdict(summary) for summary in summaries],
        "top_domain_features": feature_importance.head(config.top_feature_count).to_dict("records"),
        "similarity_holdout_count": holdout.selected_count,
        "baseline_holdout_metrics": {
            evaluation.model_name: {
                "mean_combined_score": evaluation.mean_combined_score,
                "worst_repeat_score": evaluation.worst_repeat_score,
            }
            for evaluation in base_holdouts
        },
        "adaptations": decisions,
    }
    write_decision_report(output, report_payload)
    return Phase7RunResult(
        representation_results=tuple(results),
        holdout_evaluations=tuple(all_holdouts),
        adaptations=tuple(adaptation_results),
        decisions=tuple(decisions),
        report_path=output / "report.md",
    )


def replace_representation(
    result: DomainValidationResult,
    representation: str,
) -> DomainValidationResult:
    """Rename one diagnostic artifact without changing its scientific content."""

    return replace(result, representation=representation)
