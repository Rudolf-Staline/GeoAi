# Phase 7 — Domain-shift diagnostics

Phase 7 measures train-versus-test separability using the immutable Phase 4 folds and the approved Phase 3 feature registry. It does not use pond labels in the domain classifier, expose test IDs as features, tune the classification threshold, or use leaderboard feedback.

## Reproduction

```bash
python scripts/analyze_domain_shift.py \
  --config configs/base.yaml \
  --phase7-config configs/experiments/phase7_domain_shift.yaml
```

Use `--diagnostics-only` to build the five representation diagnostics and the test-like holdout without retraining label models.

## Findings

The entity-level train-versus-test ROC-AUC is severe:

| Representation | Domain ROC-AUC |
|---|---:|
| Relative | 0.987762 |
| Invariant | 0.990709 |
| Full | 0.991376 |
| Radar | 0.893633 |
| Optical | 0.988260 |

The strongest shift drivers are optical indices/raw optical values and radar ratios. Missingness-only groups contribute little. The most test-like 20% holdout contains 365 training originals and is materially harder than global OOF: the retained LightGBM scores about 0.9652 combined and the GRU about 0.9670.

## Controlled adaptations

Two methods were evaluated over the complete 5-fold × 3-repeat pond-label protocol with seeds `7201` and `17208`:

- Removing the ten most domain-important features: mean robust score `0.980475`. The gain over the retained LightGBM (`0.980404`) is below the practical threshold, domain ROC-AUC does not fall, and OOF predictions remain almost perfectly correlated. Rejected.
- Clipped OOF importance weighting in `[0.5, 2.0]`: mean robust score `0.979235`, with weaker worst folds and seasons. Rejected.

No CORAL, MMD, gradient reversal, reconstruction pretraining, pseudo-labeling, threshold adjustment, or leaderboard-guided adaptation was promoted because the simpler controlled interventions failed to establish robust value.

## Decision

Retain the original full LightGBM, the compact masked GRU, and the fixed 50/50 LightGBM–GRU blend as Phase 8 candidates. Promote no Phase 7 adaptation.

Generated diagnostics and OOF tables remain under ignored `artifacts/domain_shift/` paths.
