# Task 06 — Compact dual-sensor temporal model

## Objective

Implement a CPU-compatible PyTorch temporal expert that separates radar and optical processing and respects variable-length masks.

## Required work

1. Implement separate radar and optical encoders.
2. Include relative position, absolute cyclic month and explicit sensor-availability masks.
3. Fuse sensor embeddings through a compact TCN or Transformer encoder. Keep architecture selectable by configuration.
4. Implement masked pooling for lengths 4–6.
5. Train with binary cross-entropy and optional cross-window consistency loss between two views of the same original row.
6. Add optional soft-F1 and pairwise ranking losses behind flags; BCE remains the default.
7. Save OOF predictions and use the exact validation API from Task 04.
8. Implement early stopping, deterministic seeds, gradient clipping and CPU smoke training.

## Constraints

- No large pretrained models.
- No test-label assumptions.
- Do not replace the accepted tree baselines.
- Every auxiliary loss must have a zero-weight ablation.

## Acceptance criteria

Tests cover tensor shapes, padding masks, all-optical-missing windows, consistency-pair grouping and deterministic inference.

```bash
python scripts/train_temporal.py --config configs/temporal_baseline.yaml --smoke
pytest
ruff check .
ruff format --check .
```
