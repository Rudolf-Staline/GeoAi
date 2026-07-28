# Task 08 — Calibrated ensemble, submission and final notebook

## Objective

Combine only accepted experts, create a valid submission and produce the reproducible final competition notebook.

## Required work

1. Load aligned OOF probabilities from accepted experts.
2. Learn non-negative blend weights summing to one using OOF data only.
3. Compare the learned blend with simple mean and hand-authored fixed blends.
4. Implement out-of-fold probability calibration with sigmoid and beta calibration; keep calibration only if robust validation improves.
5. Add optional test-time augmentation based on safe temporal or sensor dropout and report prediction variance.
6. Generate `TargetRAUC` and compute `TargetF1` strictly at threshold 0.5.
7. Validate row order, IDs, columns, probability bounds, dtypes and absence of missing values against the sample submission.
8. Build `notebooks/99_final_submission.ipynb` using reusable project modules.
9. Include SHAP or equivalent interpretation, bias discussion, reproducibility details and computational-efficiency notes required by the trustworthiness document.
10. Produce a final manifest with model hashes, configurations, seeds and source commit.

## Constraints

- No hidden manual edits to the generated CSV.
- No threshold tuning.
- No model enters the ensemble without stored OOF predictions.
- The notebook must run from a clean environment after placing competition files in `data/raw/`.

## Acceptance criteria

```bash
python scripts/build_submission.py --config configs/final.yaml
python scripts/validate_submission.py --submission artifacts/final/submission.csv
pytest
ruff check .
ruff format --check .
```
