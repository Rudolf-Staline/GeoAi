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

## Authoritative Phase 4 folds and views

All supervised experiments must load the same persisted Phase 4 manifests. Their immutable
settings are:

- original-row unit of split;
- repeated stratified grouped CV with seed `2026`, 5 folds, and 3 repeats;
- deterministic repeat seeds `2026`, `12033`, and `22040`;
- 8 sampled windows per original from test availability patterns only;
- validation-window seed `2027`, fixed before comparing models;
- mean aggregation from windows to one original probability per repeat;
- fixed classification threshold `0.5`.

Training and validation windows inherit an already assigned original fold. A Phase 5+ runner must
reject incompatible fold fingerprints, validation-window fingerprints, feature schemas, missing
predictions, duplicate `(original_id, repeat)` rows, or out-of-range probabilities. Models may not
create their own folds or resample validation masks.

Primary OOF metrics are computed per repeat at original level, then summarized across repeats.
Window-level metrics are diagnostics only. A single-class slice reports undefined ROC-AUC and
combined score rather than substituting a value. Log loss and Brier score are secondary
calibration diagnostics and never replace the competition metric.

## Stress views

The fixed report aggregates within every slice before scoring originals. It includes window
lengths 4, 5, and 6; every valid start month; start-month groups `early_year` (1–3), `mid_year`
(4–6), and `late_year` (7–9); optical-gap groups 0, 1, and 2+; optical-valid-proportion bins;
radar-complete, optical-complete, radar-only, and severely optical-limited views when present; and
per-original probability dispersion/disagreement across windows.

Leave-season-out is an additional diagnostic, not primary CV. For a repeat/fold and held season,
validation uses held-season windows from held-out originals. Training uses non-held-season windows
from other originals only; a validation original cannot re-enter training under another season.

## Robust selection score

The fixed diagnostic used to compare Phase 5+ models is:

```text
robust_score =
    0.50 * mean original-level combined score across repeats
  + 0.20 * worst original-level repeat/fold combined score
  + 0.15 * worst mean original-level window-length score
  + 0.15 * worst mean original-level season score
```

All four components are combined scores on comparable original-level probabilities. Undefined
components fail the report. This criterion is configuration-backed and must not change between
model experiments. It is a conservative local selection diagnostic, not the official Zindi
metric.

## Phase 5 tree-run contract

CatBoost and LightGBM share one model-agnostic runner. Feature families are selected from the
Phase 3 registry and provenance; IDs, labels, folds, domains, and window IDs cannot enter the
model matrix. Both families receive native missing values. No fitted global imputer, scaler, or
feature selector exists in the Phase 5 path.

The default training weight for a window is `1 / number_of_windows_for_original`, computed from
the current fold's training rows. Optional balanced class weights are calculated from
de-duplicated training originals in that fold only. Validation weights can equalize originals but
never use validation class prevalence. Uniform weighting is an explicit ablation; on the fixed
eight-view panel it preserves equal relative contribution across originals while changing the
common loss scale.

Experiments progress through immutable stages:

- `smoke`: repeat 0/fold 0 with a small iteration cap; engineering evidence only;
- `screen`: all five folds of repeat 0; rejects broken or clearly weak declared profiles;
- `full`: all 5 folds x 3 repeats; the only selection-eligible result.

Early stopping uses only the current validation fold. Every full run must emit 43,704 window
predictions and exactly 5,463 original-level rows, one per `(original_id, repeat)`. Resume requires
matching base/experiment configuration, Git provenance, fold fingerprint, validation-window
fingerprint, full Phase 3 schema fingerprint, and selected-feature fingerprint. Completed output
is never silently overwritten.

Pairwise Phase 5 diversity uses aligned full OOF only. Pearson/Spearman probability correlation,
residual correlation, disagreement, unique errors, and temporal-slice overlap are diagnostic.
Equal 50/50 blends may demonstrate complementarity, but no ensemble weights are optimized until
Phase 8.

## Metric

For probabilities `p`:

```python
prediction = (p >= 0.5).astype(int)
score = 0.60 * f1_score(y_true, prediction) + 0.40 * roc_auc_score(y_true, p)
```

The threshold is immutable.

## Ablation rule

A module is retained only when its gain is reproducible across seeds and does not materially degrade the worst subgroup. Report negative results rather than hiding them.
