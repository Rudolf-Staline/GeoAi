# Target architecture

The project is organised as a sequence of independently validated experts rather than one opaque model.

```text
raw CSV
  │
  ├── schema and missingness audit
  ├── original-row fold assignment
  ├── temporal-window generation
  └── deterministic feature construction (no fitted preprocessing)
          │
          ├── relative-position + aggregate table ──► tree experts
          ├── radar temporal branch ────────────────► radar expert
          ├── optical temporal branch ──────────────► optical expert
          └── fused temporal encoder ───────────────► fusion expert
                                                       │
OOF predictions ───────────────────────────────────────┤
                                                       ▼
                                             calibration + ensemble
                                                       │
                                                       ▼
                                      TargetRAUC and fixed-threshold TargetF1
```

## Design principles

- Original rows are the unit of splitting.
- Temporal windows are views of one object, not independent samples.
- Relative temporal position and absolute month are represented separately.
- Radar and optical availability are explicit masks.
- Padding, sensor absence, and partial optical-band absence remain distinct.
- Feature provenance records the original raw band names, formula, validity rule, aggregation,
  dtype, representation, and version.
- Every sophisticated component must survive an ablation study.
- A simpler model is preferred when the robust validation difference is negligible.

## Phase 3 representation contract

`FeatureMatrix` keeps model features separate from original IDs, stable window IDs, folds, and
optional labels. Its 688 deterministic columns comprise 192 relative-position values, 448
valid-only temporal aggregates, two radar stability summaries, and 46 window/missingness
metadata features.

`SequenceFeatureDataset` uses length-six NumPy arrays. Four- and five-month windows are padded
explicitly; padding is never treated as a missing sensor measurement. Radar feature masks,
optical per-band masks, all-raw-band masks, optical-index masks, sensor masks, calendar months,
relative positions, and cyclic month encodings are retained independently. Both representations
share a versioned `FeatureRegistry`, and train/test schema fingerprints must match before use.

## Phase 4 validation boundary

`FoldManifest` is the sole fold authority for Phase 5 and later. It contains one assignment for
each `(original_id, repeat)`, is created from unaugmented labels only, and carries a stable content
fingerprint. The configured protocol uses 5 folds, 3 repeats, and seed `2026`; repeat seeds are
derived deterministically. A Phase 2-compatible one-repeat view is exposed only after this
original-level manifest exists.

`ValidationWindowManifest` fixes the sampled validation panel independently of model code. The
same eight test-availability-mask views per original are reused across repeats and models; only
their inherited fold provenance changes. A repeat-aware window ID and manifest fingerprint make
accidental mask regeneration detectable. Full tensors share one immutable view panel in memory,
while persisted artifacts contain metadata rather than raw competition feature values.

Models emit window probabilities into the `OOFPredictions` contract. Mean aggregation is applied
inside `(original_id, repeat)` before primary scoring, so duplicated windows cannot inflate the
official sample count. Window predictions remain separate for temporal slices and stability
diagnostics. Learned preprocessors are created and fitted inside the current repeat/fold training
scope; the Phase 4 reference runner rejects an already-fitted or reused estimator.

Leave-season-out selections preserve the outer original-row split. Cluster diagnostics require
label-free invariant aggregates and fold-local scaling/clustering. Test-like adversarial holdouts
accept only complete OOF domain scores. The latter two interfaces are prepared in Phase 4, but
their scientifically meaningful execution is deferred to model evaluation and domain diagnosis.

## Phase 5 tabular boundary

`select_tabular_features` consumes `FeatureMatrix` plus the Phase 3 registry and returns an exact,
ordered `SelectedFeatureMatrix`. Six declarations are supported: relative (238), invariant (496),
full (688), radar (186), optical (515), and compact physical (101). Sensor experts use source-band
and feature-group provenance; the compact set expands an exact declared channel/statistic list.
No selector reads labels, feature values, or loose substrings.

`TabularModelAdapter` defines fit, probability, native serialization, restore, and feature-
importance behavior. `CatBoostAdapter` and `LightGBMAdapter` add deterministic CPU settings and
native missing-value handling without owning folds or metrics. The model-independent fold runner
loads the authoritative manifests, derives current-fold sample weights, fits/early-stops within
that split, emits window probabilities, and delegates original aggregation and stress reporting
back to Phase 4.

Each completed experiment directory has a compatibility manifest plus resolved configuration,
schema and feature list, original/window OOF, fold/repeat/slice metrics, model checksums,
importance stability, bounded permutation diagnostics, and runtime metadata. Model binaries live
only below ignored artifacts. Candidate registries accept only complete full-stage runs with the
authoritative fold/window fingerprints and 5,463-row OOF contract.

The diversity layer aligns candidate OOF one-to-one and reports correlations, disagreement,
unique errors, slice overlap, and fixed 50/50 diagnostic blends. Mean aggregation linearity is
used to cache stress views without changing Phase 4 metric formulas. It never optimizes ensemble
weights; calibrated or optimized combination remains a Phase 8 responsibility.

## Phase 6 temporal boundary

`SensorGatedGRU` consumes the immutable Phase 3 sequence representation rather than rebuilding
features. Radar, raw optical, and monthly-index channels receive separate fold-local masked
normalization and projections. A sensor-availability-aware element-wise gate fuses radar and
optical embeddings; explicit index, cyclic month, relative-position, radar-availability, and
optical-availability channels enter a one-layer packed GRU. Right padding is excluded by packed
sequence lengths and masked mean pooling.

The accepted architecture has 26,329 trainable parameters and is CPU-compatible. Training uses
one fixed view panel, equal total weight per original, binary cross-entropy, deterministic seeds,
fold-local early stopping, gradient clipping, and exactly the Phase 4 OOF/metric contracts.
Sensor ablation is diagnostic only. The single tested cross-window probability-consistency
variant is isolated behind configuration and was rejected before authoritative confirmation.

`temporal_diversity` aligns complete temporal and tree OOF tables one-to-one, reports residual and
classification disagreement, and evaluates only predeclared fixed blends. It never tunes blend
weights. The final accepted candidate and any final weight selection remain Phase 8 decisions.
