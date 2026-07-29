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
