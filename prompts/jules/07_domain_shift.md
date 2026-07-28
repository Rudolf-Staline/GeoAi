# Task 07 — Domain-shift diagnostics and controlled adaptation

## Objective

Measure train/test separability and implement adaptation experiments that can be accepted or rejected through ablation.

## Required work

1. Implement adversarial validation on approved representations.
2. Report domain ROC-AUC by feature family and identify the most domain-specific features.
3. Build a test-like holdout from train rows using out-of-fold domain probabilities; avoid fitting and evaluating the domain model on the same rows.
4. Implement optional, clipped importance weighting.
5. For the temporal model, implement optional CORAL, MMD or gradient-reversal adaptation. Each method must be isolated behind configuration.
6. Compare label performance, subgroup robustness and domain separability.
7. Reject an adaptation method when it lowers domain AUC but also harms robust label validation.

## Constraints

- Test data remain unlabeled.
- Do not pseudo-label in this task.
- Do not use leaderboard scores to decide whether adaptation works.
- Do not overwrite baseline artifacts.

## Acceptance criteria

Produce a decision report with `accept`, `reject` or `inconclusive` for each method, backed by OOF evidence and at least two seeds.
