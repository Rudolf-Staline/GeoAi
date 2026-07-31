# Phase 8 final delivery

## Accepted solution

The final solution uses only the two candidates accepted before Phase 8:

- `EXP-TAB-003-LGB-FULL-UNIFORM`, a 688-feature LightGBM;
- `EXP-SEQ-001-GRU-BCE`, the 26,329-parameter availability-aware masked GRU.

Both candidates reuse the immutable 5-fold × 3-repeat Phase 4 protocol and expose exactly 5,463
original-level OOF predictions plus 43,704 window-level predictions. Phase 7 adaptations are not
eligible because neither feature removal nor clipped importance weighting improved robust
validation.

## Ensemble selection

Weights were estimated inside held-out folds, never on test predictions. The nested estimates had
a median LightGBM weight of `0.45`. Production selection compared that fixed weight with the
predeclared `0.50` and `0.70` choices. The 45/55 blend had the highest mean combined score
(`0.987926`), but the equal blend had the best robust score (`0.981581`) and was preferred under
the declared robustness-first and simplicity rule.

| LightGBM / GRU | F1 | ROC-AUC | Combined | Robust | Worst fold |
|---|---:|---:|---:|---:|---:|
| 45% / 55% | 0.981718 | 0.997238 | 0.987926 | 0.981536 | 0.971649 |
| **50% / 50%** | **0.981493** | **0.997225** | **0.987786** | **0.981581** | **0.971561** |
| 70% / 30% | 0.980374 | 0.997140 | 0.987080 | 0.981087 | 0.971248 |

The selected equal blend scores `0.979913`, `0.984599`, and `0.985866` on 4-, 5-, and 6-month
windows. Its early/mid/late season scores are `0.984183`, `0.984794`, and `0.975923`; its score
for two-or-more optical gaps is `0.979774`.

## Calibration decision

Calibration was fitted cross-fold within each repeat, using different originals from the fold
being transformed. Sigmoid and beta calibration improved Brier score and expected calibration
error, but both reduced the competition and robust scores. The final probabilities therefore
remain uncalibrated.

| Calibration | Combined | Robust | Log loss | Brier | ECE | Decision |
|---|---:|---:|---:|---:|---:|---|
| **None** | **0.987786** | **0.981581** | 0.059304 | 0.014436 | 0.013685 | Retain |
| Sigmoid | 0.987297 | 0.981420 | 0.059272 | 0.014342 | 0.010873 | Reject |
| Beta | 0.987464 | 0.981346 | 0.061246 | 0.014302 | 0.010627 | Reject |

The classification rule remains exactly `TargetF1 = (TargetRAUC >= 0.5)`.

## Full-data fitting and test diagnostics

The final LightGBM uses the median accepted fold iteration count: `344`. The final GRU uses the
median accepted best epoch: `20`. Both are trained on all 1,821 labeled originals represented by
14,568 fixed sampled temporal views. Serialized model reloads must reproduce pre-save inference.

The generated test probabilities have:

- 1,030 rows in exact `SampleSubmission.csv` order;
- positive prediction rate `0.561165`;
- mean probability `0.539386`;
- LightGBM/GRU binary disagreement rate `0.131068`.

The Phase 7 covariate-shift warning remains active. An EM prior diagnostic estimates test
prevalence near `0.5856`, but no prior correction is applied because severe covariate shift makes
the pure-prior-shift assumption unreliable.

## Interpretation and trustworthiness

Global LightGBM SHAP is led by `optical__ndwi__max`, `optical__ndre1__min`, and
`radar__vv_plus_vh__min`. The dominance and domain sensitivity of maximum NDWI remain documented
risks, not causal claims. Four trustworthiness responses are generated under the required
100-word-per-section limit, together with model hashes, per-row uncertainty, runtime metadata,
and a clearly assumption-based energy estimate.

## Reproduction

Place `Train.csv`, `Test.csv`, and `SampleSubmission.csv` under `data/raw/`, then run:

```bash
python -m pip install -e ".[dev,trees,deep,notebook]"
python scripts/build_submission.py --config configs/final.yaml
python scripts/validate_submission.py --submission artifacts/final/submission.csv
pytest
ruff check .
ruff format --check .
```

The notebook `notebooks/99_final_submission.ipynb` calls the same tested modules and can either
reuse a compatible completed delivery or rebuild missing OOF and final artifacts from raw files.
Generated models, OOF predictions, diagnostics, and the submission remain ignored under
`artifacts/`.
