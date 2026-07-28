# AGENTS.md — GeoAI scientific contract

This repository is a competition research project. Correct validation is more important than local score. These rules are mandatory for every human or coding agent.

## Non-negotiable competition constraints

- Use only the files supplied by the competition. Do not add external imagery, coordinates, labels or pretrained geospatial datasets.
- Do not build or invoke an AutoML system.
- `TargetF1` must always be computed as `(TargetRAUC >= 0.5).astype(int)`. Never tune or replace the threshold.
- Do not select methods from public leaderboard feedback. Leaderboard submissions are final checks, not a validation set.
- Never commit competition data, model binaries, credentials or large generated artifacts.

## Leakage prevention

- Split original train rows before generating temporal windows, masks or augmented copies.
- All derived views of one original `ID` must remain in the same fold.
- Fit imputers, scalers, calibrators, feature selectors and domain models only on the training portion of each fold.
- Keep a persistent `original_id` and `fold` column in intermediate training manifests.
- Never use test labels or infer labels from submission feedback.
- Test missingness patterns may be analysed and reused as unlabeled masks only through an explicit, documented configuration flag.

## Validation contract

Every supervised experiment must report:

- F1 at the fixed 0.5 threshold;
- ROC-AUC on raw probabilities;
- competition score: `0.60 * F1 + 0.40 * ROC_AUC`;
- results by window length 4, 5 and 6;
- mean and standard deviation across folds;
- worst-fold score;
- the exact configuration, seed and Git commit.

OOF predictions must contain at least:

```text
ID, original_id, fold, y_true, probability, prediction, window_start, window_length, optical_months
```

## Engineering contract

- Python 3.11+ with typed, testable modules under `src/geoai_aquaculture/`.
- Configuration controls experiments; do not bury scientific choices in notebooks.
- Notebooks consume reusable modules. They must not contain the only implementation of core logic.
- Prefer deterministic CPU-compatible defaults. GPU support must remain optional.
- Use numerically safe divisions with an explicit epsilon.
- Treat `-9999` as a missing-value sentinel, never as a physical measurement.
- Each delegated task must remain within its prompt scope and arrive as a separate PR.
- Do not silently refactor unrelated modules.

## Required checks

Before reporting a task complete, run:

```bash
pytest
ruff check .
ruff format --check .
```

When model dependencies are installed, also run the task-specific smoke command stated in the prompt.

## Completion report

Every PR or agent report must state:

1. files changed;
2. scientific choices made;
3. tests and commands run;
4. known limitations;
5. artifacts produced;
6. the exact next task that is now unblocked.
