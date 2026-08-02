"""Leakage-safe OOF gating between retained tabular experts.

The gate is trained only on out-of-fold predictions.  Every outer meta-fold holds
out an original ID across all three validation repeats, preventing the same
original from appearing in both gate training and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geoai_aquaculture.constants import FIXED_THRESHOLD
from geoai_aquaculture.data import (
    ProjectConfig,
    load_competition_data,
    materialize_test_windows,
)
from geoai_aquaculture.metrics import metric_result
from geoai_aquaculture.submission import build_submission, validate_submission
from geoai_aquaculture.training.artifacts import load_experiment_artifact_manifest
from geoai_aquaculture.validation import (
    OOFPredictions,
    ValidationReport,
    build_oof_predictions,
    build_validation_report,
    load_fold_manifest,
    load_oof_predictions,
)


class GatingError(ValueError):
    """Raised when candidate OOF predictions cannot support safe gating."""


META_FEATURES = (
    "cat_probability",
    "invariant_probability",
    "mean_probability",
    "probability_difference",
    "absolute_probability_difference",
    "cat_logit",
    "invariant_logit",
    "cat_margin",
    "invariant_margin",
    "margin_difference",
    "cat_positive",
    "invariant_positive",
    "binary_disagreement",
    "window_length_scaled",
    "radar_fraction",
    "optical_fraction",
    "optical_gap_fraction",
    "window_start_sin",
    "window_start_cos",
)


@dataclass(frozen=True, slots=True)
class FittedGate:
    """One fitted selector or a constant fallback when one class is absent."""

    model: Pipeline | None
    constant_probability: float | None
    c_value: float
    training_rows: int
    training_originals: int
    positive_rate: float

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return the probability of preferring the invariant LightGBM."""

        if tuple(features.columns) != META_FEATURES:
            raise GatingError("gate feature schema differs from the fitted contract")
        if self.model is None:
            if self.constant_probability is None:
                raise GatingError("constant gate has no probability")
            return np.full(features.shape[0], self.constant_probability, dtype=np.float64)
        values = self.model.predict_proba(features)[:, 1]
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GatingResult:
    """Cross-fitted evidence and the production gate trained on all OOF rows."""

    boundary_oof: OOFPredictions
    boundary_report: ValidationReport
    hard_oof: OOFPredictions
    hard_report: ValidationReport
    soft_oof: OOFPredictions
    soft_report: ValidationReport
    catboost_report: ValidationReport
    invariant_report: ValidationReport
    fold_diagnostics: pd.DataFrame
    production_gate: FittedGate
    production_c: float
    accepted: bool
    acceptance_reason: str


def _load_candidate_oof(artifact_dir: Path, project: ProjectConfig) -> OOFPredictions:
    manifest = load_experiment_artifact_manifest(artifact_dir)
    if manifest.stage != "full" or manifest.status != "complete":
        raise GatingError(f"candidate is not a complete Stage C run: {artifact_dir}")
    if manifest.fold_manifest_fingerprint != project.tabular.fold_manifest_fingerprint:
        raise GatingError("candidate fold fingerprint differs from the project contract")
    if manifest.validation_window_fingerprint != project.tabular.validation_window_fingerprint:
        raise GatingError("candidate validation-window fingerprint differs")
    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    return load_oof_predictions(
        artifact_dir / "oof_predictions.csv",
        artifact_dir / "window_predictions.csv",
        folds,
        validation_window_fingerprint=project.tabular.validation_window_fingerprint,
        expected_fingerprint=manifest.oof_fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )


def align_candidate_windows(catboost: OOFPredictions, invariant: OOFPredictions) -> pd.DataFrame:
    """Align two complete candidate window OOF tables exactly."""

    keys = [
        "ID",
        "original_id",
        "repeat",
        "fold",
        "y_true",
        "label",
        "window_id",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
        "fold_manifest_fingerprint",
        "validation_window_fingerprint",
    ]
    left = catboost.windows.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "cat_probability"}
    )
    right = invariant.windows.loc[:, [*keys, "probability"]].rename(
        columns={"probability": "invariant_probability"}
    )
    aligned = left.merge(right, on=keys, how="inner", validate="one_to_one")
    expected = catboost.windows.shape[0]
    if aligned.shape[0] != expected or invariant.windows.shape[0] != expected:
        raise GatingError("candidate window OOF rows do not align exactly")
    if aligned.duplicated(["repeat", "window_id"]).any():
        raise GatingError("aligned OOF contains duplicate repeat/window rows")

    reference = (
        aligned.loc[aligned["repeat"].eq(0), ["original_id", "fold"]]
        .drop_duplicates()
        .rename(columns={"fold": "meta_fold"})
    )
    if reference["original_id"].duplicated().any():
        raise GatingError("repeat-zero fold assignment is not unique per original")
    aligned = aligned.merge(reference, on="original_id", how="left", validate="many_to_one")
    if aligned["meta_fold"].isna().any():
        raise GatingError("meta-fold assignment left unassigned originals")
    aligned["meta_fold"] = aligned["meta_fold"].astype(np.int16)
    return aligned.sort_values(
        ["meta_fold", "repeat", "original_id", "window_id"],
        kind="stable",
        ignore_index=True,
    )


def build_gate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the compact, test-available gate feature set."""

    required = {
        "cat_probability",
        "invariant_probability",
        "window_start",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise GatingError(f"gate input is missing columns: {missing}")
    cat = frame["cat_probability"].to_numpy(dtype=np.float64)
    invariant = frame["invariant_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(cat).all() or not np.isfinite(invariant).all():
        raise GatingError("gate probabilities must be finite")
    if np.any((cat < 0.0) | (cat > 1.0)) or np.any((invariant < 0.0) | (invariant > 1.0)):
        raise GatingError("gate probabilities must lie in [0, 1]")
    epsilon = 1e-6
    cat_clip = np.clip(cat, epsilon, 1.0 - epsilon)
    invariant_clip = np.clip(invariant, epsilon, 1.0 - epsilon)
    length = frame["window_length"].to_numpy(dtype=np.float64)
    if np.any(length <= 0.0):
        raise GatingError("window lengths must be positive")
    start = frame["window_start"].to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * (start - 1.0) / 12.0
    values = pd.DataFrame(
        {
            "cat_probability": cat,
            "invariant_probability": invariant,
            "mean_probability": (cat + invariant) / 2.0,
            "probability_difference": invariant - cat,
            "absolute_probability_difference": np.abs(invariant - cat),
            "cat_logit": np.log(cat_clip) - np.log1p(-cat_clip),
            "invariant_logit": np.log(invariant_clip) - np.log1p(-invariant_clip),
            "cat_margin": np.abs(cat - FIXED_THRESHOLD),
            "invariant_margin": np.abs(invariant - FIXED_THRESHOLD),
            "margin_difference": np.abs(invariant - FIXED_THRESHOLD)
            - np.abs(cat - FIXED_THRESHOLD),
            "cat_positive": (cat >= FIXED_THRESHOLD).astype(np.int8),
            "invariant_positive": (invariant >= FIXED_THRESHOLD).astype(np.int8),
            "binary_disagreement": (
                (cat >= FIXED_THRESHOLD) != (invariant >= FIXED_THRESHOLD)
            ).astype(np.int8),
            "window_length_scaled": length / 6.0,
            "radar_fraction": frame["radar_months"].to_numpy(dtype=np.float64) / length,
            "optical_fraction": frame["optical_months"].to_numpy(dtype=np.float64) / length,
            "optical_gap_fraction": frame["internal_optical_gap_count"].to_numpy(
                dtype=np.float64
            )
            / length,
            "window_start_sin": np.sin(angle),
            "window_start_cos": np.cos(angle),
        },
        columns=META_FEATURES,
    )
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise GatingError("gate features contain non-finite values")
    return values


def _fit_gate(frame: pd.DataFrame, c_value: float) -> FittedGate:
    disagreement = (
        (frame["cat_probability"].to_numpy(dtype=np.float64) >= FIXED_THRESHOLD)
        != (frame["invariant_probability"].to_numpy(dtype=np.float64) >= FIXED_THRESHOLD)
    )
    subset = frame.loc[disagreement].copy()
    if subset.empty:
        return FittedGate(None, 0.0, c_value, 0, 0, 0.0)
    cat_label = (
        subset["cat_probability"].to_numpy(dtype=np.float64) >= FIXED_THRESHOLD
    ).astype(np.int8)
    invariant_label = (
        subset["invariant_probability"].to_numpy(dtype=np.float64) >= FIXED_THRESHOLD
    ).astype(np.int8)
    truth = subset["label"].to_numpy(dtype=np.int8)
    if np.any(cat_label == invariant_label):
        raise GatingError("gate training subset contains agreeing candidate labels")
    target = (invariant_label == truth).astype(np.int8)
    counts = subset["original_id"].astype(str).value_counts()
    sample_weight = subset["original_id"].astype(str).map(1.0 / counts).to_numpy(dtype=np.float64)
    sample_weight *= sample_weight.size / sample_weight.sum()
    positive_rate = float(target.mean())
    if np.unique(target).size < 2:
        return FittedGate(
            None,
            positive_rate,
            c_value,
            subset.shape[0],
            subset["original_id"].nunique(),
            positive_rate,
        )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "gate",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=2026,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(
        build_gate_features(subset),
        target,
        gate__sample_weight=sample_weight,
    )
    return FittedGate(
        model,
        None,
        c_value,
        subset.shape[0],
        subset["original_id"].nunique(),
        positive_rate,
    )


def _policy_probabilities(
    frame: pd.DataFrame,
    gate_probability: np.ndarray,
    *,
    policy: str,
) -> np.ndarray:
    cat = frame["cat_probability"].to_numpy(dtype=np.float64)
    invariant = frame["invariant_probability"].to_numpy(dtype=np.float64)
    gate = np.asarray(gate_probability, dtype=np.float64)
    if gate.shape != cat.shape or not np.isfinite(gate).all():
        raise GatingError("gate predictions do not align with candidate probabilities")
    cat_label = cat >= FIXED_THRESHOLD
    invariant_label = invariant >= FIXED_THRESHOLD
    disagreement = cat_label != invariant_label
    prefer_invariant = disagreement & (gate >= 0.5)

    if policy == "soft":
        probability = cat.copy()
        probability[disagreement] = (
            gate[disagreement] * invariant[disagreement]
            + (1.0 - gate[disagreement]) * cat[disagreement]
        )
        return probability
    if policy == "hard":
        return np.where(prefer_invariant, invariant, cat)
    if policy != "boundary":
        raise GatingError(f"unsupported gate policy: {policy}")

    desired_label = cat_label.copy()
    desired_label[prefer_invariant] = invariant_label[prefer_invariant]
    probability = cat.copy()
    groups: Iterable[tuple[Any, pd.DataFrame]]
    if "repeat" in frame.columns:
        groups = frame.groupby("repeat", sort=False, observed=True)
    else:
        groups = [(0, frame)]
    for _, group in groups:
        indices = group.index.to_numpy(dtype=np.int64)
        move_up = indices[(~cat_label[indices]) & desired_label[indices]]
        move_down = indices[cat_label[indices] & ~desired_label[indices]]
        if move_up.size:
            order = move_up[np.argsort(cat[move_up], kind="stable")]
            probability[order] = np.linspace(0.5000001, 0.5000010, order.size)
        if move_down.size:
            order = move_down[np.argsort(cat[move_down], kind="stable")]
            probability[order] = np.linspace(0.4999990, 0.4999999, order.size)
    return probability


def _original_metric(frame: pd.DataFrame, probability: np.ndarray) -> float:
    compact = frame.loc[:, ["repeat", "original_id", "label"]].copy()
    compact["probability"] = np.asarray(probability, dtype=np.float64)
    aggregated = (
        compact.groupby(["repeat", "original_id"], observed=True, sort=False)
        .agg(label=("label", "first"), probability=("probability", "mean"))
        .reset_index()
    )
    result = metric_result(aggregated["label"], aggregated["probability"])
    if result.combined_score is None:
        raise GatingError("inner gate score has undefined ROC-AUC")
    return float(result.combined_score)


def _select_c(frame: pd.DataFrame, c_grid: tuple[float, ...]) -> tuple[float, pd.DataFrame]:
    records: list[dict[str, float | int]] = []
    folds = sorted(frame["meta_fold"].unique().tolist())
    for c_value in c_grid:
        prediction = np.full(frame.shape[0], np.nan, dtype=np.float64)
        for fold in folds:
            training = frame.loc[~frame["meta_fold"].eq(fold)]
            heldout = frame.loc[frame["meta_fold"].eq(fold)]
            fitted = _fit_gate(training, c_value)
            gate = fitted.predict(build_gate_features(heldout))
            prediction[heldout.index.to_numpy(dtype=np.int64)] = _policy_probabilities(
                heldout,
                gate,
                policy="boundary",
            )
        if not np.isfinite(prediction).all():
            raise GatingError("inner meta-CV left unfilled predictions")
        score = _original_metric(frame, prediction)
        records.append({"c_value": float(c_value), "combined_score": score})
    table = pd.DataFrame.from_records(records).sort_values(
        ["combined_score", "c_value"], ascending=[False, True], ignore_index=True
    )
    return float(table.iloc[0]["c_value"]), table


def _window_frame(
    aligned: pd.DataFrame,
    probability: np.ndarray,
    *,
    experiment_id: str,
    model_id: str,
) -> pd.DataFrame:
    frame = aligned.drop(
        columns=["cat_probability", "invariant_probability", "meta_fold"]
    ).copy()
    p = np.asarray(probability, dtype=np.float64)
    prediction = (p >= FIXED_THRESHOLD).astype(np.int8)
    frame["probability"] = p
    frame["prediction"] = prediction
    frame["predicted_class"] = prediction
    frame["experiment_id"] = experiment_id
    frame["model_id"] = model_id
    columns = [
        "ID",
        "original_id",
        "repeat",
        "fold",
        "y_true",
        "label",
        "probability",
        "prediction",
        "predicted_class",
        "window_id",
        "window_start",
        "window_end",
        "window_length",
        "radar_months",
        "optical_months",
        "internal_optical_gap_count",
        "experiment_id",
        "model_id",
        "fold_manifest_fingerprint",
        "validation_window_fingerprint",
    ]
    return frame.loc[:, columns]


def crossfit_gate(
    catboost: OOFPredictions,
    invariant: OOFPredictions,
    project: ProjectConfig,
    *,
    c_grid: tuple[float, ...] = (0.05, 0.2, 1.0, 5.0),
    minimum_robust_gain: float = 0.0005,
) -> GatingResult:
    """Cross-fit the gate while holding each original out across every repeat."""

    if not c_grid or any(value <= 0.0 for value in c_grid):
        raise GatingError("gate C grid must contain positive values")
    aligned = align_candidate_windows(catboost, invariant)
    gate_probability = np.full(aligned.shape[0], np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for outer_fold in sorted(aligned["meta_fold"].unique().tolist()):
        training = aligned.loc[~aligned["meta_fold"].eq(outer_fold)].copy()
        heldout = aligned.loc[aligned["meta_fold"].eq(outer_fold)].copy()
        training.index = np.arange(training.shape[0])
        selected_c, inner_scores = _select_c(training, c_grid)
        fitted = _fit_gate(training, selected_c)
        heldout_gate = fitted.predict(build_gate_features(heldout))
        gate_probability[heldout.index.to_numpy(dtype=np.int64)] = heldout_gate
        fold_records.append(
            {
                "outer_meta_fold": int(outer_fold),
                "selected_c": selected_c,
                "training_disagreement_rows": fitted.training_rows,
                "training_disagreement_originals": fitted.training_originals,
                "training_prefer_invariant_rate": fitted.positive_rate,
                "heldout_rows": heldout.shape[0],
                "heldout_originals": heldout["original_id"].nunique(),
                "heldout_mean_gate_probability": float(heldout_gate.mean()),
                "inner_best_combined_score": float(inner_scores.iloc[0]["combined_score"]),
            }
        )
    if not np.isfinite(gate_probability).all():
        raise GatingError("outer meta-CV left unfilled gate predictions")

    folds = load_fold_manifest(
        project.tabular.validation_artifacts_dir / "fold_manifest.csv",
        project.validation,
        expected_fingerprint=project.tabular.fold_manifest_fingerprint,
    )
    evaluations: dict[str, tuple[OOFPredictions, ValidationReport]] = {}
    for policy in ("boundary", "hard", "soft"):
        probability = _policy_probabilities(aligned, gate_probability, policy=policy)
        windows = _window_frame(
            aligned,
            probability,
            experiment_id=f"PHASE9-GATE-{policy.upper()}",
            model_id=f"catboost-vs-invariant:{policy}",
        )
        oof = build_oof_predictions(
            windows,
            folds,
            validation_window_fingerprint=project.tabular.validation_window_fingerprint,
            method=project.validation.aggregation_method,
            trimmed_fraction=project.validation.trimmed_mean_fraction,
        )
        evaluations[policy] = (oof, build_validation_report(oof, project.validation))

    catboost_report = build_validation_report(catboost, project.validation)
    invariant_report = build_validation_report(invariant, project.validation)
    base_reports = (catboost_report, invariant_report)
    base_robust = max(float(item.summary["robust_selection"]["score"]) for item in base_reports)
    base_combined = max(
        float(item.summary["official_metric"]["mean_combined_score"]) for item in base_reports
    )
    boundary_report = evaluations["boundary"][1]
    boundary_robust = float(boundary_report.summary["robust_selection"]["score"])
    boundary_combined = float(
        boundary_report.summary["official_metric"]["mean_combined_score"]
    )
    accepted = (
        boundary_robust >= base_robust + minimum_robust_gain
        and boundary_combined >= base_combined - 0.0001
    )
    if accepted:
        reason = (
            f"accepted: robust gain={boundary_robust - base_robust:+.6f}, "
            f"combined delta={boundary_combined - base_combined:+.6f}"
        )
    else:
        reason = (
            f"rejected: robust gain={boundary_robust - base_robust:+.6f} requires "
            f">={minimum_robust_gain:.6f}; combined delta={boundary_combined - base_combined:+.6f}"
        )

    production_c, _ = _select_c(aligned.reset_index(drop=True), c_grid)
    production_gate = _fit_gate(aligned, production_c)
    return GatingResult(
        boundary_oof=evaluations["boundary"][0],
        boundary_report=boundary_report,
        hard_oof=evaluations["hard"][0],
        hard_report=evaluations["hard"][1],
        soft_oof=evaluations["soft"][0],
        soft_report=evaluations["soft"][1],
        catboost_report=catboost_report,
        invariant_report=invariant_report,
        fold_diagnostics=pd.DataFrame.from_records(fold_records).sort_values(
            "outer_meta_fold", ignore_index=True
        ),
        production_gate=production_gate,
        production_c=production_c,
        accepted=accepted,
        acceptance_reason=reason,
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_gate_pipeline(
    project: ProjectConfig,
    *,
    catboost_artifact: Path,
    invariant_artifact: Path,
    catboost_test_submission: Path,
    invariant_test_submission: Path,
    output_dir: Path,
    submission_path: Path,
    allow_unaccepted: bool = False,
) -> GatingResult:
    """Evaluate the OOF gate and optionally emit its production test submission."""

    catboost = _load_candidate_oof(catboost_artifact, project)
    invariant = _load_candidate_oof(invariant_artifact, project)
    result = crossfit_gate(catboost, invariant, project)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.fold_diagnostics.to_csv(output_dir / "fold_diagnostics.csv", index=False)
    result.boundary_oof.original.to_csv(output_dir / "boundary_oof_predictions.csv", index=False)
    result.boundary_oof.windows.to_csv(output_dir / "boundary_window_predictions.csv", index=False)
    payload = {
        "accepted": result.accepted,
        "acceptance_reason": result.acceptance_reason,
        "production_c": result.production_c,
        "feature_names": list(META_FEATURES),
        "production_gate": {
            "training_rows": result.production_gate.training_rows,
            "training_originals": result.production_gate.training_originals,
            "prefer_invariant_rate": result.production_gate.positive_rate,
            "constant_probability": result.production_gate.constant_probability,
        },
        "reports": {
            "catboost": result.catboost_report.summary,
            "invariant": result.invariant_report.summary,
            "boundary": result.boundary_report.summary,
            "hard": result.hard_report.summary,
            "soft": result.soft_report.summary,
        },
    }
    (output_dir / "gate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    if not result.accepted and not allow_unaccepted:
        return result

    data = load_competition_data(project)
    cat_test = pd.read_csv(catboost_test_submission, dtype={project.data.id_column: "string"})
    invariant_test = pd.read_csv(
        invariant_test_submission,
        dtype={project.data.id_column: "string"},
    )
    required = {project.data.id_column, "TargetRAUC"}
    if required - set(cat_test.columns) or required - set(invariant_test.columns):
        raise GatingError("test candidate submissions lack ID or TargetRAUC")
    if not cat_test[project.data.id_column].equals(invariant_test[project.data.id_column]):
        raise GatingError("test candidate IDs do not align")
    if not cat_test[project.data.id_column].equals(
        data.sample_submission[project.data.id_column].astype("string")
    ):
        raise GatingError("test candidate IDs do not follow SampleSubmission order")
    test_windows = materialize_test_windows(data)
    manifest = test_windows.manifest.reset_index(drop=True)
    if not manifest["original_id"].astype("string").equals(cat_test[project.data.id_column]):
        raise GatingError("test window metadata do not align with candidate IDs")
    test_frame = pd.DataFrame(
        {
            "original_id": manifest["original_id"].astype("string"),
            "cat_probability": cat_test["TargetRAUC"].to_numpy(dtype=np.float64),
            "invariant_probability": invariant_test["TargetRAUC"].to_numpy(dtype=np.float64),
            "window_start": manifest["window_start"].to_numpy(dtype=np.int8),
            "window_length": manifest["window_length"].to_numpy(dtype=np.int8),
            "radar_months": manifest["radar_months"].to_numpy(dtype=np.int8),
            "optical_months": manifest["optical_months"].to_numpy(dtype=np.int8),
            "internal_optical_gap_count": manifest["internal_optical_gap_count"].to_numpy(
                dtype=np.int8
            ),
        }
    )
    gate_probability = result.production_gate.predict(build_gate_features(test_frame))
    probability = _policy_probabilities(test_frame, gate_probability, policy="boundary")
    submission = build_submission(
        data.sample_submission[project.data.id_column],
        probability,
    )
    validate_submission(submission, data.sample_submission)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            project.data.id_column: data.sample_submission[project.data.id_column],
            "cat_probability": test_frame["cat_probability"],
            "invariant_probability": test_frame["invariant_probability"],
            "gate_probability": gate_probability,
            "final_probability": probability,
            "candidate_disagreement": (
                (test_frame["cat_probability"] >= FIXED_THRESHOLD)
                != (test_frame["invariant_probability"] >= FIXED_THRESHOLD)
            ),
        }
    ).to_csv(output_dir / "test_gate_diagnostics.csv", index=False)
    return result
