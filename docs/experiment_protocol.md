# Experiment protocol

## Experiment identity

Every run receives a unique name and stores:

- immutable resolved configuration;
- Git commit SHA;
- package versions;
- random seeds;
- train/validation IDs for each fold;
- OOF predictions;
- overall and subgroup metrics.

## Model selection

Models are selected using robust local validation, not the public leaderboard. The minimum selection table includes:

- standard masked grouped CV;
- scores for lengths 4, 5 and 6;
- leave-season-out score;
- adversarial or test-like holdout score when implemented;
- worst-fold score;
- runtime and peak memory.

## Metric

For probabilities `p`:

```python
prediction = (p >= 0.5).astype(int)
score = 0.60 * f1_score(y_true, prediction) + 0.40 * roc_auc_score(y_true, p)
```

The threshold is immutable.

## Ablation rule

A module is retained only when its gain is reproducible across seeds and does not materially degrade the worst subgroup. Report negative results rather than hiding them.
