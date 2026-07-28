# Task 05 — Strong tree baselines

## Objective

Implement the first competitive, reproducible models using the approved validation protocol. No automatic model search.

## Required work

1. Add explicit CatBoost and LightGBM configurations with a small hand-authored parameter set.
2. Implement three experts:
   - raw relative-window CatBoost;
   - aggregate-feature CatBoost;
   - aggregate-feature LightGBM.
3. Train fold-by-fold using only fold-training preprocessing.
4. Save OOF and test probabilities, fold models, feature manifests, metrics and runtime.
5. Support multiple declared seeds, but do not perform unbounded parameter search.
6. Implement feature importance and optional SHAP on a bounded sample.
7. Build a first fixed-weight blend and compare it against every component using OOF predictions.
8. Add a smoke mode that completes quickly on a synthetic dataset.

## Selection rule

Do not call a model better based only on mean CV. Compare worst fold, window-length subgroups and season holdout.

## Acceptance criteria

```bash
python scripts/train_trees.py --config configs/tree_baseline.yaml --smoke
pytest
ruff check .
ruff format --check .
```

The report must include an ablation table and enough metadata to reproduce every number.
