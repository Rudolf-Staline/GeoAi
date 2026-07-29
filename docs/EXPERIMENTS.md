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

## phase-02-temporal-windows — accepted

| Field | Result |
|---|---|
| Hypothesis | Assigning folds to original rows before augmentation and sampling only test availability booleans can reproduce test-like temporal missingness without feature-value or fold leakage. |
| Representation | Raw 4–6 month views with separate calendar months, relative positions, padding, radar masks, per-band optical masks, and stable window IDs. |
| Model and parameters | None. Five original-row stratified grouped folds; seed `20260728`; eight sampled views per original; test-mask sampling enabled; temporal and optical dropout disabled. |
| Validation setup | Synthetic exhaustive-count, edge-month, mask fidelity, test-value perturbation, per-band optical-gap, input immutability, reproducibility, malformed-mask, and intentional cross-fold leakage tests. |
| F1 / ROC-AUC / combined score | Not applicable; Phase 2 trains no model. |
| Runtime | Sampled audit: 29.191 seconds inside the command, 31.07 seconds wall time, 286,680 KiB peak RSS. Exhaustive audit: 15.691 seconds inside, 17.72 seconds wall time, 448,904 KiB peak RSS. |
| Artifacts | `artifacts/temporal_windows/window_summary.json`, fold/window manifests, deduplicated mask templates, distributions, and report. Raw window feature values are not persisted. |
| Decision | Accept. Fold leakage is structurally rejected, same-seed output is identical, alternate seeds change sampled view content without changing folds, and Phase 1 remains unchanged. |

### Findings

- Exhaustive generation produces 24 views per original row: 9 starts for length 4, 8 for length
  5, and 7 for length 6. Across 1,821 rows this is 43,704 views: 16,389 / 14,568 /
  12,747 by length. Every valid start contributes exactly 1,821 exhaustive views.
- The configured sampled run produces 14,568 views, exactly 8 per original. Counts by length are
  4,801 / 4,929 / 4,838 for lengths 4 / 5 / 6.
- Sampled start counts are length 4 months 1–9: `518, 513, 539, 540, 523, 545, 546, 535,
  542`; length 5 months 1–8: `604, 625, 633, 574, 653, 608, 641, 591`; and length 6
  months 1–7: `659, 706, 660, 687, 706, 689, 731`.
- The test set yields 78 deduplicated boolean availability patterns representing all 1,030 test
  rows. All 78 appear in the sampled train views, weighted by their test frequencies; no test ID
  or feature value is stored in a mask template.
- No original ID crosses folds, window IDs are unique, fixed-seed generation is identical, and
  seed `20260729` changes sampled view content while preserving the original fold manifest.

### Commands

```bash
python scripts/audit_data.py --config configs/base.yaml
python scripts/generate_windows.py --config configs/base.yaml
python scripts/generate_windows.py --config configs/base.yaml --mode exhaustive
pytest tests -k "window or leakage or mask"
python -m compileall src tests
pytest
ruff check .
ruff format --check .
```

Known limitations: sampling is with replacement, so one original row may receive semantically
duplicate views with distinct stable view indices; the fixed folds are a Phase 2 integrity
manifest rather than the repeated validation protocol planned for Phase 4; and optional dropout
paths are tested but disabled pending ablation evidence. The exact next task unblocked is Phase 3,
physics-informed feature engineering over these raw values and explicit masks.
