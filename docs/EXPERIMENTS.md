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

## phase-03-physics-temporal-features — accepted

| Field | Result |
|---|---|
| Hypothesis | A small, explicitly mapped set of water/vegetation/radar transformations plus gap-aware temporal summaries can represent 4–6 month windows without imputation, leakage, or loss of missingness semantics. |
| Representation | One 688-column tabular matrix and one length-six masked NumPy sequence representation. IDs, window IDs, folds, and optional labels remain attached metadata only. |
| Model and parameters | None. Feature version `phase3_v1`; safe-division epsilon `1e-6`; no fitted preprocessing. |
| Validation setup | Synthetic formula, denominator, validity, partial-band, gap-aware slope, insufficient-observation, padding, schema-alignment, fold-retention, immutability, determinism, CLI, and numerical-finiteness tests; real sampled and exhaustive audits. |
| F1 / ROC-AUC / combined score | Not applicable; Phase 3 trains no model. |
| Score by window length / worst season | Not applicable; subgroup scoring begins after supervised validation exists. |
| Runtime | Sampled: 45.460 seconds inside the command, 47.54 seconds wall time, 466,824 KiB peak RSS. Exhaustive: 34.422 seconds inside the command, 36.50 seconds wall time, 1,017,104 KiB peak RSS. |
| Artifacts | Ignored aggregate-only files under `artifacts/features/`: registry, feature-group counts, train/test group missingness, sequence-mask summary, shape/fingerprint summary, Markdown report, and run metadata. No raw rows, IDs, or labels are persisted. |
| Decision | Accept. Both representations are deterministic, numerically finite-or-missing, train/test aligned, input-preserving, and retain Phase 2 identities, folds, calendar time, relative time, sensor masks, per-band masks, and padding. |

### Feature groups and formulas

- The semantic mapping is explicit in `configs/base.yaml`: radar `VV`/`VH`; visible `blue`,
  `green`, `red`; red edge `re1`, `re2`, `re3`; NIR `nir`; narrow NIR `nira`; and SWIR
  `swir1`, `swir2`. Every configured/observed band is mapped exactly once; unavailable or
  ambiguous mappings fail.
- Eight radar monthly channels are retained: raw VV, raw VH, VV−VH, VV+VH,
  VV/abs(VH), VH/abs(VV), and adjacent-position first differences for VV and VH. Raw VV/VH
  aggregation supplies first-to-last change, amplitude, standard deviation, and stability;
  mean absolute adjacent change adds two explicit stability summaries.
- Ten raw optical channels and 14 indices are retained. The indices are NDVI, NDWI, MNDWI,
  NDMI, NBR, two narrow-NIR/red-edge normalized differences, narrow-NIR/RE1−1, NIR/SWIR1,
  NIR/SWIR2, Green/SWIR1, Green/SWIR2, normalized Green/Red contrast, and normalized
  Blue/Green contrast.
- Every ratio is valid only when all inputs are finite and the denominator magnitude exceeds
  `1e-6`. Invalid or overflowing divisions are explicitly missing; neither positive nor negative
  infinity is allowed.
- Each of the 32 monthly channels contributes six relative-position columns and 14 valid-only
  aggregates: count, mean, median, population standard deviation, minimum, maximum, amplitude,
  25th/75th percentiles, IQR, first/last valid values, first-to-last difference, and slope over
  the true relative positions. Variation/change/slope require at least two observations, so one
  observation is distinguishable from true zero variance.
- The 688 tabular columns are exactly 192 relative-position values, 448 temporal aggregates,
  two radar stability summaries, and 46 metadata/missingness features. The registry contains 739
  definitions across tabular and sequence outputs.
- Sequence arrays have shapes `(N, 6, 8)` radar, `(N, 6, 10)` raw optical, `(N, 6, 14)`
  optical indices, `(N, 6, 2)` cyclic month encoding, and `(N, 6, 12)` raw per-band masks,
  plus channel, sensor, index, and padding masks and integer calendar/relative positions.

### Real-data audit findings

- Sampled train features have shape `(14,568, 688)` and sequence row count 14,568; test
  features have shape `(1,030, 688)` and sequence row count 1,030. Length distributions remain
  4,801 / 4,929 / 4,838 in sampled train and 345 / 343 / 342 in test for lengths 4 / 5 / 6.
- Exhaustive train features have shape `(43,704, 688)`, preserving the Phase 2 counts 16,389 /
  14,568 / 12,747 by length.
- Inside non-padding positions, fully missing optical months occur at 6.3779% in masked train and
  6.2172% in test; all-raw-band missingness is 5.3149% versus 5.1810%. Including explicit
  padding, relative raw optical columns are missing at 21.9419% versus 21.8932%. These small
  differences are reported diagnostically, not corrected in this phase.
- Sampled masked train and test have no missing radar sensor-months. Derived radar sequence masks
  are false at the first position for first-difference channels by definition, not because the
  radar sensor is absent.
- Train/test schema fingerprints match exactly for both representations. Same-input rebuilds are
  identical, input temporal windows remain unchanged, and infinity counts are zero.
- The observed radar values are mostly negative and look compatible with a log-like supplied
  representation, but no competition documentation proves the physical scale. Required
  differences, sums, and safe ratios therefore operate on the supplied numbers; logarithmic
  ratio variants are rejected for now.

### Rejected ideas and limitations

- Log radar ratios are rejected because the radar scale is unproven and values are frequently
  negative. Hundreds of arbitrary pairwise band interactions are rejected as physically weak
  and likely to increase later overfitting risk.
- Zero filling, learned imputation, global scaling, label-based feature selection, and domain
  adaptation are outside Phase 3 and were not performed. Missing physical values remain `NaN`
  with explicit masks.
- The aggregate table happens to be fully defined on the supplied sampled/test windows because
  every real channel has enough valid observations; relative columns and sequence tensors still
  preserve all padding and optical absence. Synthetic tests cover zero- and one-observation cases.
- Exhaustive construction is CPU-compatible but requires about 1 GiB peak memory. No claim about
  predictive value is made until the fixed Phase 4 validation protocol exists.

### Commands

```bash
python scripts/audit_data.py --config configs/base.yaml
python scripts/generate_windows.py --config configs/base.yaml
python scripts/build_features.py --config configs/base.yaml
python scripts/build_features.py --config configs/base.yaml --mode exhaustive \
  --output-dir artifacts/features_exhaustive
python -m compileall src tests
pytest
ruff check .
ruff format --check .
```

The exact next task unblocked is Phase 4: implement the fixed repeated stratified grouped and
stress-validation framework, using persistent original-row folds and these immutable feature
schemas. No model training has occurred yet.
