# Phase 9 — Leakage-safe OOF gating

## Goal

Replace leaderboard-guided boundary edits with a selector learned exclusively from
complete CatBoost and invariant-LightGBM out-of-fold predictions.

The gate answers one narrow question: when the two retained tabular experts disagree
at the immutable threshold `0.5`, which expert should be trusted for that temporal
window?

## Leakage control

Each original appears in all three Phase 4 repeats. A normal fold split at the OOF-row
level would therefore leak the same original into gate training and evaluation.

Phase 9 assigns every original to its fold from repeat 0, then holds that original out
across **all repeats and all windows**. The outer cross-fit therefore has five disjoint
sets of original IDs. Regularization `C` is selected with another grouped cross-fit
inside each outer training partition.

## Inputs

The authoritative artifacts are:

- `EXP-TAB-003-CB-FULL-LOWLR` — CatBoost full representation;
- `EXP-TAB-002-LGB-INVARIANT` — invariant LightGBM representation.

Both artifacts must be complete Stage C runs with the current fold and validation-window
fingerprints. The script refuses stale or incomplete artifacts.

The gate uses only information available for the test rows:

- both candidate probabilities and logits;
- probability differences and confidence margins;
- binary disagreement;
- window length and starting month;
- radar and optical availability fractions;
- internal optical-gap fraction.

No target-derived feature, leaderboard score, test label, pseudo-label or external data
is used.

## Model and policies

The selector is a standardized logistic regression trained only on disagreement rows.
Every original contributes equal total sample weight. A constant fallback is used when
a training partition contains only one selector class.

Three policies are evaluated with fully cross-fitted gate predictions:

1. **boundary** — preserve CatBoost probabilities except when the gate selects the
   invariant label, moving only those rows immediately across `0.5`;
2. **hard** — use the complete probability from the selected expert;
3. **soft** — blend the two probabilities using the gate probability.

Only `boundary` is eligible for production because it is the lowest-variance extension
of the current public champion.

## Acceptance rule

The gate is accepted only when:

- its immutable Phase 4 robust score improves on the better base expert by at least
  `0.0005`; and
- its mean official combined score is no more than `0.0001` below the better base
  expert.

When the rule is not met, the report is still written but no submission is generated.
`--allow-unaccepted` exists only for diagnostics and must not be treated as promotion.

## Run

```bash
python scripts/build_oof_gate_submission.py \
  --config configs/base.yaml
```

Expected outputs:

- `artifacts/phase9_oof_gate/gate_report.json`;
- `artifacts/phase9_oof_gate/fold_diagnostics.csv`;
- cross-fitted boundary original/window predictions;
- `artifacts/phase9_oof_gate/test_gate_diagnostics.csv` when promoted;
- `submissions/submission_oof_gated.csv` when promoted.

## Interpretation

A public improvement is not evidence by itself. Promotion is based only on grouped OOF
results. The public leaderboard can be used afterward as an external check, not as a
training target or hyperparameter selector.
