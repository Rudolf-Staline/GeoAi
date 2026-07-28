# Task 04 — Robust validation and experiment records

## Objective

Build the validation system that every later model must use. This task defines the scientific truth of the project.

## Required work

1. Implement repeated stratified grouped folds where the group is the original train `ID`.
2. Split original rows before calling the window generator.
3. Implement the exact competition metrics with immutable threshold 0.5.
4. Generate and persist OOF predictions with the schema required by `AGENTS.md`.
5. Report metrics:
   - overall;
   - per fold;
   - window lengths 4, 5 and 6;
   - window start month;
   - optical-month count;
   - worst fold and worst subgroup.
6. Add optional leave-season-out evaluation.
7. Create an experiment manifest containing resolved config, seed, package versions and Git SHA.
8. Provide a dummy-estimator smoke script proving the complete loop without CatBoost or LightGBM.

## Leakage tests

Tests must intentionally construct duplicated augmented views and demonstrate that the validator rejects cross-fold contamination.

## Acceptance criteria

```bash
python scripts/smoke_validation.py --config configs/base.yaml
pytest
ruff check .
ruff format --check .
```

Two executions with the same seed must generate identical folds and OOF row order.
