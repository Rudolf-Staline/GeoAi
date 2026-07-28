# Experiment log

This log records accepted and rejected scientific phases. Supervised metrics are intentionally
absent until the fixed validation protocol is implemented in Phase 4.

## phase-01-data-audit — accepted

| Field | Result |
|---|---|
| Hypothesis | The supplied schema and test missingness can be reproduced deterministically without using labels beyond validating the train target. |
| Representation | Raw monthly radar and optical columns; no derived model features. |
| Model and parameters | None. Missing sentinel `-9999`; 12 months; radar validity requires both radar bands and optical validity requires all ten optical bands. |
| Validation setup | Schema invariants plus synthetic ingestion, malformed-schema, missingness, window, optical-gap, CLI, and determinism tests. |
| F1 / ROC-AUC / combined score | Not applicable; Phase 1 trains no model. |
| Score by window length / worst season | Not applicable; subgroup scoring begins after supervised validation exists. |
| Runtime | 0.744 seconds inside the audit command; 2.64 seconds wall time; 193,148 KiB peak RSS on the local CPU environment. |
| Artifacts | `artifacts/data_audit/audit_summary.json`, `report.md`, temporal metadata, row/band/month/sensor missingness CSVs, and test-window CSVs. |
| Decision | Keep. The audit passed and establishes the factual input contract for temporal augmentation. |

### Findings

- Train has 1,821 rows, 144 complete temporal features, and label counts 1,086 negative / 735
  positive. Test has 1,030 rows and 89,756 sentinel cells (60.5151% of temporal cells).
- Test radar windows are consecutive. Lengths 4, 5, and 6 occur in 345, 343, and 342 rows,
  respectively; starts range from month 1 through month 9 subject to fitting within 12 months.
- Radar and optical bands have internally consistent month-level availability. There are no
  partial sensor-months and no optical observations outside radar-valid windows.
- 273 test rows contain optical gaps inside their radar windows: 231 have one gap, 37 have two,
  and 5 have three. The 1,030 rows contain 78 distinct joint radar/optical availability patterns.
- Source hashes, the resolved configuration hash, seed, and Git provenance are recorded in the
  ignored audit summary. No competition rows or generated artifacts are committed.

### Commands

```bash
python -m pip install -e ".[dev]"
python scripts/audit_data.py --config configs/base.yaml
python -m compileall src tests
pytest
ruff check .
ruff format --check .
```

Known limitation: Phase 1 validates structure, finiteness, missingness, and alignment, but does not
impose undocumented physical value ranges. No `configs/base.yaml` change was required. The exact
next task unblocked is Phase 2, leakage-safe temporal window generation from original-row fold
assignments and explicit test-derived availability masks.
