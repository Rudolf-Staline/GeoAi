"""Model-agnostic OOF candidate registry and Phase 5 diversity diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geoai_aquaculture.data import ProjectConfig
from geoai_aquaculture.metrics import metric_result
from geoai_aquaculture.validation import (
    FoldManifest,
    OOFPredictions,
    load_oof_predictions,
)

from .artifacts import load_experiment_artifact_manifest


class DiversityError(ValueError):
    """Raised when candidate OOF predictions cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One accepted or complementary Stage C experiment with immutable provenance."""

    experiment_id: str
    model_family: str
    model_profile: str
    feature_set: str
    artifact_dir: Path
    candidate_role: str
    retention_reason: str
    official_score: float
    robust_score: float
    fold_manifest_fingerprint: str
    validation_window_fingerprint: str
    selected_feature_schema_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact_dir"] = self.artifact_dir.as_posix()
        return value


@dataclass(frozen=True, slots=True)
class AcceptedCandidateRegistry:
    """Only complete compatible Stage C candidates approved for later phases."""

    candidates: tuple[CandidateRecord, ...]
    fold_manifest_fingerprint: str
    validation_window_fingerprint: str

    def __post_init__(self) -> None:
        if not self.candidates:
            raise DiversityError("candidate registry must not be empty")
        ids = [candidate.experiment_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise DiversityError("candidate registry experiment IDs must be unique")
        for candidate in self.candidates:
            if candidate.fold_manifest_fingerprint != self.fold_manifest_fingerprint:
                raise DiversityError("candidate registry fold fingerprints differ")
            if candidate.validation_window_fingerprint != self.validation_window_fingerprint:
                raise DiversityError("candidate registry validation-window fingerprints differ")


@dataclass(frozen=True, slots=True)
class OOFDiversityReport:
    """Pairwise probabilities, errors, stress overlap, and equal-blend diagnostics."""

    pairwise: pd.DataFrame
    slice_error_overlap: pd.DataFrame
    diagnostic_blends: pd.DataFrame


def _load_candidate_declarations(path: Path) -> list[dict[str, str]]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DiversityError(f"invalid candidate registry YAML: {path}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
        raise DiversityError("candidate registry requires a 'candidates' list")
    declarations: list[dict[str, str]] = []
    for item in raw["candidates"]:
        if not isinstance(item, dict):
            raise DiversityError("candidate registry entries must be mappings")
        required = {"experiment_id", "artifact_dir", "candidate_role", "retention_reason"}
        missing = sorted(required - set(item))
        if missing or any(
            not isinstance(item[key], str) or not item[key].strip() for key in required
        ):
            raise DiversityError(f"candidate registry entry is incomplete: {missing}")
        declarations.append({key: item[key] for key in required})
    return declarations


def load_accepted_candidate_registry(
    path: str | Path,
    project: ProjectConfig,
) -> AcceptedCandidateRegistry:
    """Load only full, complete, fingerprint-compatible experiment artifacts."""

    source = Path(path).expanduser().resolve()
    declarations = _load_candidate_declarations(source)
    candidates: list[CandidateRecord] = []
    for declaration in declarations:
        artifact = Path(declaration["artifact_dir"])
        if not artifact.is_absolute():
            artifact = (project.project_root / artifact).resolve()
        manifest = load_experiment_artifact_manifest(artifact)
        if manifest.stage != "full" or manifest.status != "complete":
            raise DiversityError(
                f"candidate {manifest.experiment_id} is not a complete Stage C run"
            )
        if declaration["experiment_id"] != manifest.experiment_id:
            raise DiversityError("candidate declaration and artifact experiment IDs differ")
        if manifest.original_oof_rows != project.tabular.expected_full_oof_rows:
            raise DiversityError("candidate original-level OOF row count is incomplete")
        metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((artifact / "resolved_config.yaml").read_text(encoding="utf-8"))
        candidates.append(
            CandidateRecord(
                experiment_id=manifest.experiment_id,
                model_family=str(resolved["model"]["family"]),
                model_profile=str(resolved["model"]["name"]),
                feature_set=str(resolved["feature_set"]),
                artifact_dir=artifact,
                candidate_role=declaration["candidate_role"],
                retention_reason=declaration["retention_reason"],
                official_score=float(metrics["official_metric"]["mean_combined_score"]),
                robust_score=float(metrics["robust_selection"]["score"]),
                fold_manifest_fingerprint=manifest.fold_manifest_fingerprint,
                validation_window_fingerprint=manifest.validation_window_fingerprint,
                selected_feature_schema_fingerprint=(manifest.selected_feature_schema_fingerprint),
            )
        )
    return AcceptedCandidateRegistry(
        candidates=tuple(candidates),
        fold_manifest_fingerprint=project.tabular.fold_manifest_fingerprint,
        validation_window_fingerprint=project.tabular.validation_window_fingerprint,
    )


def load_candidate_oof(
    candidate: CandidateRecord,
    folds: FoldManifest,
    project: ProjectConfig,
) -> OOFPredictions:
    """Round-trip one candidate through the authoritative Phase 4 OOF loader."""

    manifest = load_experiment_artifact_manifest(candidate.artifact_dir)
    return load_oof_predictions(
        candidate.artifact_dir / "oof_predictions.csv",
        candidate.artifact_dir / "window_predictions.csv",
        folds,
        validation_window_fingerprint=project.tabular.validation_window_fingerprint,
        expected_fingerprint=manifest.oof_fingerprint,
        method=project.validation.aggregation_method,
        trimmed_fraction=project.validation.trimmed_mean_fraction,
    )


def _aligned_originals(left: OOFPredictions, right: OOFPredictions) -> pd.DataFrame:
    keys = ["original_id", "repeat", "fold", "label"]
    left_frame = left.original.loc[:, [*keys, "probability", "prediction"]].rename(
        columns={"probability": "probability_left", "prediction": "prediction_left"}
    )
    right_frame = right.original.loc[:, [*keys, "probability", "prediction"]].rename(
        columns={"probability": "probability_right", "prediction": "prediction_right"}
    )
    merged = left_frame.merge(right_frame, on=keys, how="inner", validate="one_to_one")
    if merged.shape[0] != left.original.shape[0] or merged.shape[0] != right.original.shape[0]:
        raise DiversityError("candidate original-level OOF rows do not align exactly")
    return merged


def _pairwise_record(
    left_id: str,
    right_id: str,
    aligned: pd.DataFrame,
) -> dict[str, Any]:
    y = aligned["label"].to_numpy(dtype=np.int8)
    p_left = aligned["probability_left"].to_numpy(dtype=np.float64)
    p_right = aligned["probability_right"].to_numpy(dtype=np.float64)
    c_left = aligned["prediction_left"].to_numpy(dtype=np.int8)
    c_right = aligned["prediction_right"].to_numpy(dtype=np.int8)
    error_left = c_left != y
    error_right = c_right != y
    overlap = error_left & error_right
    union = error_left | error_right
    return {
        "candidate_left": left_id,
        "candidate_right": right_id,
        "row_count": aligned.shape[0],
        "pearson_probability_correlation": float(np.corrcoef(p_left, p_right)[0, 1]),
        "spearman_probability_correlation": float(
            pd.Series(p_left).corr(pd.Series(p_right), method="spearman")
        ),
        "residual_correlation": float(np.corrcoef(y - p_left, y - p_right)[0, 1]),
        "binary_disagreement_rate": float(np.mean(c_left != c_right)),
        "positive_class_disagreement_rate": float(np.mean(c_left[y == 1] != c_right[y == 1])),
        "unique_true_positives_left": int(((c_left == 1) & (c_right == 0) & (y == 1)).sum()),
        "unique_true_positives_right": int(((c_right == 1) & (c_left == 0) & (y == 1)).sum()),
        "unique_false_positives_left": int(((c_left == 1) & (c_right == 0) & (y == 0)).sum()),
        "unique_false_positives_right": int(((c_right == 1) & (c_left == 0) & (y == 0)).sum()),
        "shared_error_count": int(overlap.sum()),
        "error_jaccard": float(overlap.sum() / union.sum()) if union.any() else 1.0,
    }


def _slice_specs(frame: pd.DataFrame, project: ProjectConfig) -> list[tuple[str, str, np.ndarray]]:
    length = frame["window_length"].to_numpy(dtype=np.int8)
    start = frame["window_start"].to_numpy(dtype=np.int8)
    gaps = frame["internal_optical_gap_count"].to_numpy(dtype=np.int8)
    specifications: list[tuple[str, str, np.ndarray]] = [
        ("window_length", str(value), length == value) for value in (4, 5, 6)
    ]
    for season in project.validation.seasons:
        specifications.append(("season", season.name, np.isin(start, season.start_months)))
    specifications.extend(
        (
            ("optical_gaps", "none", gaps == 0),
            ("optical_gaps", "one", gaps == 1),
            ("optical_gaps", "two_or_more", gaps >= 2),
        )
    )
    return specifications


def _slice_error_records(
    left_id: str,
    right_id: str,
    left_slices: dict[tuple[str, str], pd.DataFrame],
    right_slices: dict[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if left_slices.keys() != right_slices.keys():
        raise DiversityError("candidate stress-slice definitions do not align")
    for group, value in left_slices:
        aligned = _aligned_compact_originals(
            left_slices[(group, value)], right_slices[(group, value)]
        )
        y = aligned["label"].to_numpy(dtype=np.int8)
        error_left = aligned["prediction_left"].to_numpy(dtype=np.int8) != y
        error_right = aligned["prediction_right"].to_numpy(dtype=np.int8) != y
        overlap = error_left & error_right
        union = error_left | error_right
        records.append(
            {
                "candidate_left": left_id,
                "candidate_right": right_id,
                "slice_group": group,
                "slice_value": value,
                "original_count": aligned.shape[0],
                "left_error_count": int(error_left.sum()),
                "right_error_count": int(error_right.sum()),
                "shared_error_count": int(overlap.sum()),
                "left_only_error_count": int((error_left & ~error_right).sum()),
                "right_only_error_count": int((error_right & ~error_left).sum()),
                "error_jaccard": float(overlap.sum() / union.sum()) if union.any() else 1.0,
            }
        )
    return records


def _compact_mean_aggregate(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Vectorize the fixed mean aggregation needed repeatedly by diversity diagnostics."""

    grouped = window_predictions.groupby(["repeat", "original_id"], sort=False, observed=True)
    consistency = grouped[["fold", "label"]].nunique(dropna=False)
    if (consistency.to_numpy() != 1).any():
        raise DiversityError("stress-slice window metadata are inconsistent within an original")
    result = grouped.agg(
        fold=("fold", "first"),
        label=("label", "first"),
        probability=("probability", "mean"),
    ).reset_index()
    result["prediction"] = (result["probability"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    return result.sort_values(["repeat", "fold", "original_id"], kind="stable", ignore_index=True)


def _precompute_candidate_slices(
    oof: OOFPredictions,
    project: ProjectConfig,
) -> dict[tuple[str, str], pd.DataFrame]:
    if project.validation.aggregation_method != "mean":
        raise DiversityError("Phase 5 fast diversity diagnostics require fixed mean aggregation")
    return {
        (group, value): _compact_mean_aggregate(oof.windows.loc[selector])
        for group, value, selector in _slice_specs(oof.windows, project)
    }


def _aligned_compact_originals(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["original_id", "repeat", "fold", "label"]
    aligned = left.loc[:, [*keys, "probability", "prediction"]].merge(
        right.loc[:, [*keys, "probability", "prediction"]],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if aligned.shape[0] != left.shape[0] or aligned.shape[0] != right.shape[0]:
        raise DiversityError("candidate compact original predictions do not align")
    return aligned


def _combined_score(frame: pd.DataFrame) -> float:
    result = metric_result(frame["label"], frame["probability"])
    if result.combined_score is None:
        raise DiversityError("diagnostic blend metric has undefined ROC-AUC")
    return result.combined_score


def _blend_frame(aligned: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "repeat": aligned["repeat"].to_numpy(dtype=np.int16),
            "fold": aligned["fold"].to_numpy(dtype=np.int16),
            "label": aligned["label"].to_numpy(dtype=np.int8),
            "probability": (
                aligned["probability_left"].to_numpy(dtype=np.float64)
                + aligned["probability_right"].to_numpy(dtype=np.float64)
            )
            / 2.0,
        }
    )


def _diagnostic_blend(
    left: OOFPredictions,
    right: OOFPredictions,
    left_slices: dict[tuple[str, str], pd.DataFrame],
    right_slices: dict[tuple[str, str], pd.DataFrame],
    project: ProjectConfig,
) -> dict[str, float]:
    """Compute exact required blend components using mean aggregation linearity."""

    original = _blend_frame(_aligned_originals(left, right))
    repeat_scores = np.asarray(
        [_combined_score(group) for _, group in original.groupby("repeat", sort=True)],
        dtype=np.float64,
    )
    fold_scores = np.asarray(
        [_combined_score(group) for _, group in original.groupby(["repeat", "fold"], sort=True)],
        dtype=np.float64,
    )
    slice_scores: dict[tuple[str, str], float] = {}
    for key in left_slices:
        if key[0] not in {"window_length", "season"}:
            continue
        blended = _blend_frame(_aligned_compact_originals(left_slices[key], right_slices[key]))
        per_repeat = [_combined_score(group) for _, group in blended.groupby("repeat", sort=True)]
        slice_scores[key] = float(np.mean(per_repeat))
    length_scores = [
        value for (group, _), value in slice_scores.items() if group == "window_length"
    ]
    season_scores = [value for (group, _), value in slice_scores.items() if group == "season"]
    if len(length_scores) != 3 or not season_scores:
        raise DiversityError("diagnostic blend robust slices are incomplete")
    mean_combined = float(np.mean(repeat_scores))
    worst_fold = float(np.min(fold_scores))
    worst_length = float(np.min(length_scores))
    worst_season = float(np.min(season_scores))
    weights = project.validation.robust_score_weights
    robust = (
        weights.mean_combined * mean_combined
        + weights.worst_fold * worst_fold
        + weights.worst_window_length * worst_length
        + weights.worst_season * worst_season
    )
    return {
        "mean_combined_score": mean_combined,
        "robust_score": float(robust),
        "worst_fold_score": worst_fold,
        "worst_season_score": worst_season,
    }


def analyze_oof_diversity(
    registry: AcceptedCandidateRegistry,
    folds: FoldManifest,
    project: ProjectConfig,
) -> OOFDiversityReport:
    """Compare complete candidates without optimizing final ensemble weights."""

    loaded = {
        candidate.experiment_id: load_candidate_oof(candidate, folds, project)
        for candidate in registry.candidates
    }
    slice_cache = {
        experiment_id: _precompute_candidate_slices(oof, project)
        for experiment_id, oof in loaded.items()
    }
    pairwise_records: list[dict[str, Any]] = []
    slice_records: list[dict[str, Any]] = []
    blend_records: list[dict[str, Any]] = []
    for left_candidate, right_candidate in combinations(registry.candidates, 2):
        left = loaded[left_candidate.experiment_id]
        right = loaded[right_candidate.experiment_id]
        aligned = _aligned_originals(left, right)
        pairwise_records.append(
            _pairwise_record(
                left_candidate.experiment_id,
                right_candidate.experiment_id,
                aligned,
            )
        )
        slice_records.extend(
            _slice_error_records(
                left_candidate.experiment_id,
                right_candidate.experiment_id,
                slice_cache[left_candidate.experiment_id],
                slice_cache[right_candidate.experiment_id],
            )
        )
        blend = _diagnostic_blend(
            left,
            right,
            slice_cache[left_candidate.experiment_id],
            slice_cache[right_candidate.experiment_id],
            project,
        )
        blend_records.append(
            {
                "candidate_left": left_candidate.experiment_id,
                "candidate_right": right_candidate.experiment_id,
                **blend,
                "optimization_performed": False,
                "weights": "0.5,0.5",
            }
        )
    return OOFDiversityReport(
        pairwise=pd.DataFrame.from_records(pairwise_records),
        slice_error_overlap=pd.DataFrame.from_records(slice_records),
        diagnostic_blends=pd.DataFrame.from_records(blend_records),
    )


def write_oof_diversity_artifacts(
    output_dir: Path,
    registry: AcceptedCandidateRegistry,
    report: OOFDiversityReport,
) -> tuple[Path, ...]:
    """Persist candidate and equal-blend evidence separately from final ensembling."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    registry_path = output_dir / "candidate_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fold_manifest_fingerprint": registry.fold_manifest_fingerprint,
                "validation_window_fingerprint": registry.validation_window_fingerprint,
                "candidates": [candidate.as_dict() for candidate in registry.candidates],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(registry_path)
    for name, frame in (
        ("pairwise_oof_diversity", report.pairwise),
        ("slice_error_overlap", report.slice_error_overlap),
        ("equal_weight_diagnostic_blends", report.diagnostic_blends),
    ):
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        paths.append(path)
    return tuple(paths)
