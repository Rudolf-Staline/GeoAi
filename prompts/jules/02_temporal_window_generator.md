# Task 02 — Leakage-safe temporal window generator

## Objective

Implement temporal views that reproduce test conditions while proving that no augmented copy crosses a validation boundary.

## Required work

1. Implement an original-row fold assignment API before any augmentation.
2. Parse test missingness into reusable unlabeled mask templates containing:
   - first and last available radar month;
   - window length;
   - radar availability per month;
   - optical availability per month and band.
3. Implement consecutive windows of length 4, 5 and 6 for train rows.
4. Support two modes:
   - sampled windows controlled by seed;
   - exhaustive 24-window generation.
5. Re-index available months to relative positions 1–6 while preserving absolute month metadata.
6. Retain `original_id`, fold, window start, length, sensor masks and augmentation seed.
7. Add temporal and optical dropout behind explicit configuration flags.
8. Add exhaustive leakage tests: every derived row from one `original_id` must have exactly one fold.

## Non-goals

- No spectral indices.
- No model training.
- No leaderboard submission.

## Acceptance criteria

- Deterministic output for the same seed.
- Different seeds produce different sampled views without changing fold membership.
- All windows are consecutive before optional cloud masking.
- Padding and missingness are distinguishable.
- Synthetic tests cover edge windows starting in months 1 and 9.

Run:

```bash
pytest tests -k "window or leakage or mask"
ruff check .
ruff format --check .
```
