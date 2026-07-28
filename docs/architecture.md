# Target architecture

The project is organised as a sequence of independently validated experts rather than one opaque model.

```text
raw CSV
  │
  ├── schema and missingness audit
  ├── original-row fold assignment
  ├── temporal-window generation
  └── relative-time representation
          │
          ├── physics-informed aggregate features ──► tree experts
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
- Every sophisticated component must survive an ablation study.
- A simpler model is preferred when the robust validation difference is negligible.
