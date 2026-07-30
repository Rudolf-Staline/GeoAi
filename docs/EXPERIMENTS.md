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

## phase-04-authoritative-validation — accepted

| Field | Result |
|---|---|
| Hypothesis | Fixed original-row folds plus a model-independent test-mask window panel can produce comparable OOF and temporal-stress evidence without duplicated-window inflation or fitted-preprocessor leakage. |
| Representation | Repeated original fold manifest; fixed sampled window manifest; original- and window-level OOF contracts; temporal slices and stability tables. |
| Model and parameters | One noncompetitive integration reference only: 16 predefined Phase 3 aggregates/metadata features, fold-local median imputation and standardization, untuned logistic regression (`C=1`, `liblinear`, `max_iter=500`). |
| Validation setup | 5 stratified grouped folds x 3 repeats; seed `2026`; eight fixed test-mask-derived windows per original/repeat using window seed `2027`; original-level mean probability; threshold `0.5`. |
| Reference F1 / ROC-AUC / combined score | `0.926808` / `0.979065` / `0.947711` mean across repeats. This verifies integration and is not a Phase 5 baseline or tuned result. |
| Robust / worst fold / worst repeat | `0.936647` / `0.910596` / `0.947181`; repeat combined-score standard deviation `0.000511`. |
| Reference score by window length | 4: `0.939364`; 5: `0.939846`; 6: `0.942795`. |
| Reference score by start-month season | early (starts 1–3): `0.940844`; middle (4–6): `0.937560`; late (7–9): `0.931783`. Two-plus optical-gap score: `0.945532`. |
| Runtime | 102.720 seconds inside the command (105.82 seconds wall); 77.185 seconds reference execution; 253.934 MiB peak RSS on the local CPU environment. |
| Artifacts | Ignored files under `artifacts/validation/`: fold/window/season manifests, fingerprints, reference original/window OOF, fold/repeat/slice metrics, stability, protocol, Markdown report, and run provenance. |
| Decision | Accept. Structural leakage checks, exact metrics, complete OOF linkage, deterministic fingerprints, fixed stress views, and fold-local reference execution all pass. |

### Fixed manifest facts

- The fold manifest has 5,463 rows: one for each of 1,821 originals in each of three repeats.
  Validation folds contain 364 or 365 originals and exactly 147 positives; positive rates range
  from 0.402740 to 0.403846. Its accepted fingerprint is
  `4dbc9029f242c5ff4f8d2e23b0fb0d83334d993c1a4ecd7ce95e8e18c37ceece`.
- The primary validation manifest has 43,704 rows, exactly 14,568 per repeat and eight views per
  original. Each repeat contains 4,938 / 4,859 / 4,771 windows of lengths 4 / 5 / 6. All 78
  deduplicated Phase 2 test availability patterns occur. Its accepted fingerprint is
  `89ef5e9a108a4cad09582db82ce1970dbf4873cbb3b01692c96a8fcc54b14492`.
- Across all repeats, internal optical-gap counts are 32,373 / 9,618 / 1,524 / 189 for 0 / 1 /
  2 / 3 gaps. The same window panel is reused across repeats so repeat variation reflects fold
  assignment rather than newly sampled masks.
- Original-level reference OOF contains exactly 5,463 rows; window-level OOF contains 43,704.
  The OOF fingerprint is
  `8934e8b6dc7580cba1cbf1b0176bd354dcf5f8dfe37e29f2bb1f3776faf3804e`.

### Scientific and engineering decisions

- Official F1, ROC-AUC, and combined score are calculated only after fixed mean aggregation to
  one prediction per original/repeat. Window scores remain diagnostic. Probability `0.5` is
  classified positive; configuration values other than `0.5` fail.
- The robust score keeps the official score separate and adds fixed penalties through worst fold,
  worst window length, and worst configured season. All components are original-level combined
  scores and weights are fixed in `configs/base.yaml`.
- Leave-season-out definitions are materialized now, while cluster fitting is deferred to the
  future outer-fold model runner because scaling and clustering must be fitted inside that scope.
  The adversarial holdout accepts only future per-original OOF train-vs-test similarity scores.
- A small Phase 2 implementation refactor preallocates window arrays and reuses the identical
  sampled panel across repeat assignments. Fingerprints and all Phase 2 scientific definitions
  are unchanged; this lowers Phase 4 memory without changing values, masks, or IDs.

### Limitations and rejected scope

- The season names describe start-month groups, not a proven local aquaculture phenology. No
  spatial or geographic holdout is possible because the supplied rows expose no documented
  grouping coordinates.
- Cluster robustness execution and adversarially selected holdout scores are deferred until a
  fold-local Phase 5 representation and Phase 7 OOF domain probabilities exist. No test feature
  values enter fold construction, and domain scores can never become pond-model features.
- The reference result must not be used as evidence that logistic regression is the selected
  competition model. CatBoost, LightGBM, tuning, calibration, ensembling, neural models, domain
  adaptation, test predictions, and submission generation remain outside Phase 4.

### Commands

```bash
python scripts/audit_data.py --config configs/base.yaml
python scripts/generate_windows.py --config configs/base.yaml
python scripts/build_features.py --config configs/base.yaml
python scripts/build_validation.py --config configs/base.yaml
python scripts/build_validation.py --config configs/base.yaml --skip-reference \
  --output-dir artifacts/validation_repro_a
python scripts/build_validation.py --config configs/base.yaml --skip-reference \
  --output-dir artifacts/validation_repro_b
python scripts/build_validation.py --config configs/base.yaml --skip-reference --seed 2027 \
  --output-dir artifacts/validation_seed_2027
python -m compileall src tests
pytest
ruff check .
ruff format --check .
```

The exact next unblocked task is Phase 5: run manually configured CatBoost and LightGBM tabular
baselines against these immutable fold/window fingerprints and original-level OOF/robust metrics.

## phase-05-strong-tabular-baselines — accepted

| Field | Result |
|---|---|
| Hypothesis | Registry-selected physical/temporal features plus deterministic boosted trees can improve robust original-level validation without changing Phase 4 folds, windows, aggregation, threshold, or metrics. |
| Models | CatBoost 1.2.10 and LightGBM 4.7.0, native missing values, four deterministic CPU threads, fold-local early stopping. No AutoML or unrestricted search. |
| Validation | Immutable 5 folds x 3 repeats; fold fingerprint `4dbc9029f242c5ff4f8d2e23b0fb0d83334d993c1a4ecd7ce95e8e18c37ceece`; window fingerprint `89ef5e9a108a4cad09582db82ce1970dbf4873cbb3b01692c96a8fcc54b14492`; mean original aggregation; threshold `0.5`. |
| Best single model | `EXP-TAB-003-LGB-FULL-UNIFORM`: F1 `0.979268`, ROC-AUC `0.996653`, combined `0.986222`, robust `0.980404`. |
| Strongest default-weight model | `EXP-TAB-002-LGB-INVARIANT`: F1 `0.979749`, ROC-AUC `0.996199`, combined `0.986329`, robust `0.979904`. |
| Strongest CatBoost | `EXP-TAB-003-CB-FULL-LOWLR`: F1 `0.979081`, ROC-AUC `0.996295`, combined `0.985967`, robust `0.979816`. |
| Diagnostic blend | Fixed 50/50 best-single LightGBM + best CatBoost: combined `0.986614`, robust `0.980647`. This is complementarity evidence, not optimized or final ensembling. |
| Runtime | Nine Stage C runners totaled 3,250.39 seconds; the approved-list command took 58:20 wall time and 1,969.19 MiB process peak RSS. Optimized all-candidate diversity took 84.10 seconds and 291.50 MiB. |
| Decision | Accept. Both tree families ran successfully, every Stage C OOF has exactly 5,463 rows, all immutable fingerprints match, leading models materially beat the Phase 4 reference, and CatBoost adds fixed-blend value. |

### Feature declarations

All selectors use the Phase 3 registry and ordered provenance, never labels or loose substring
matching. IDs, labels, folds, repeats, window IDs, and domain indicators remain metadata only.

| Feature set | Exact count | Declaration |
|---|---:|---|
| Relative | 238 | 192 monthly relative-position values plus all 46 window/missingness metadata features. |
| Invariant | 496 | 448 standard temporal aggregates, two radar-stability aggregates, and 46 metadata features; no relative raw values. |
| Full | 688 | Entire authoritative Phase 3 tabular schema, fingerprint `af93d8bfc1406583e1834519fb5012b97052446e44674fbb8a0cf917bc9032b9`. |
| Radar | 186 | Radar raw/derived/temporal/stability features, radar provenance-based validity, and neutral window metadata. |
| Optical | 515 | Optical raw bands/indices/aggregates, optical provenance-based validity, and neutral window metadata. |
| Compact physical | 101 | Exact expansion of NDWI, MNDWI, NDMI, NBR, NDVI, NDRE1/2, chlorophyll red edge, VV, VH, VV−VH and VV+VH over valid count, median, standard deviation, amplitude, IQR, first-to-last and slope; two radar-stability features; 15 declared window/gap metadata features. |

Every experiment artifact stores the complete ordered `feature_list.txt` and selected schema
fingerprint. The compact declaration is implemented as exact registry names; absent entries fail.

### Stage A engineering smoke

One CatBoost medium fold and one conservative LightGBM fold ran with 20-tree caps. Both produced
365 original predictions, 2,920 window predictions, bounded probabilities, early-stopping
metadata, native model files, checksums, and load-equivalent inference. The scores were marked
engineering-only and never used for selection. After configuring four CPU threads, CatBoost's
five-fold medium screen reproduced the single-thread scientific metrics exactly while reducing
wall time from 12:06 to 4:37; peak RSS increased from about 0.95 to 1.75 GiB.

### Stage B predeclared model-profile screen

All scores below use repeat 0 and the same five folds. They are screening evidence only.

| Profile | Weighting | Combined | Robust | Worst fold | Runner seconds | Decision |
|---|---|---:|---:|---:|---:|---|
| CatBoost shallow/regularized | equal original | 0.982232 | 0.978113 | 0.975935 | 96.6 | Reject: lower mean and robust score. |
| CatBoost medium depth | equal original | 0.985103 | 0.979888 | 0.973388 | 246.9 | Reject after stronger finalists. |
| CatBoost low learning rate | equal original | 0.985503 | 0.980138 | 0.975605 | 343.7 | Promote: highest CatBoost screen robust score. |
| CatBoost medium/class weighted | equal original + fold class | 0.984777 | 0.980052 | 0.977970 | 236.5 | Promote: strongest CatBoost worst fold. |
| CatBoost medium/uniform | uniform | 0.984853 | 0.979415 | 0.974009 | 228.1 | Reject: no scale-ablation benefit. |
| LightGBM conservative leaves | equal original | 0.985753 | 0.979724 | 0.973191 | 86.5 | Reject after stronger finalists. |
| LightGBM constrained depth | equal original | 0.986042 | 0.979952 | 0.977152 | 99.6 | Promote: strongest default-weight worst fold. |
| LightGBM strong L1/L2 | equal original | 0.985006 | 0.978994 | 0.974715 | 102.6 | Reject: regularization reduced robustness. |
| LightGBM conservative/class weighted | equal original + fold class | 0.984946 | 0.979121 | 0.973125 | 85.3 | Reject: no fixed-threshold benefit. |
| LightGBM conservative/uniform | uniform | 0.986230 | 0.980836 | 0.975242 | 72.0 | Promote: best LightGBM screen robust score. |

Uniform weighting is not the default. It was retained as an explicit common-loss-scale ablation;
because the immutable panel contains exactly eight views for every original, it does not change
relative original contributions. Class weights were calculated from de-duplicated current-fold
training originals only. Validation weighting never used validation prevalence.

### Stage B feature-representation screen

The six feature families used one shared LightGBM profile, seed `4100`, repeat, and folds so only
the selected columns changed.

| Experiment | Features | Combined | Robust | Four-month | Late season | 2+ gaps |
|---|---:|---:|---:|---:|---:|---:|
| EXP-TAB-001 relative | 238 | 0.981952 | 0.977789 | 0.975608 | 0.969158 | 0.965350 |
| EXP-TAB-002 invariant | 496 | 0.986031 | 0.979855 | 0.974644 | 0.971043 | 0.972851 |
| EXP-TAB-003 full | 688 | 0.985611 | 0.979632 | 0.974160 | 0.971533 | 0.974204 |
| EXP-TAB-004 radar | 186 | 0.931979 | 0.923536 | 0.916697 | 0.907556 | 0.899050 |
| EXP-TAB-005 optical | 515 | 0.984259 | 0.979377 | 0.977603 | 0.973490 | 0.966132 |
| EXP-TAB-006 compact | 101 | 0.980855 | 0.974900 | 0.969044 | 0.964969 | 0.961500 |

### Stage C authoritative confirmations

Only this table is selection-eligible. Each row contains 43,704 fixed window predictions and
exactly 5,463 original/repeat predictions.

| Experiment | F1 | ROC-AUC | Combined | Robust | Worst fold | L4 | L5 | L6 | Early | Mid | Late | 2+ gaps | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-TAB-001-LGB-RELATIVE | 0.973306 | 0.995609 | 0.982227 | 0.977045 | 0.969843 | 0.976301 | 0.980584 | 0.978149 | 0.977663 | 0.978916 | 0.970118 | 0.966972 | 178.0 |
| EXP-TAB-002-LGB-INVARIANT | 0.979749 | 0.996199 | 0.986329 | 0.979904 | 0.972655 | 0.974786 | 0.981622 | 0.984858 | 0.980661 | 0.981632 | 0.973272 | 0.974935 | 220.4 |
| EXP-TAB-003-LGB-FULL | 0.979043 | 0.996292 | 0.985943 | 0.979690 | 0.972805 | 0.974625 | 0.981798 | 0.984280 | 0.979587 | 0.981468 | 0.973093 | 0.974840 | 290.7 |
| EXP-TAB-003-LGB-FULL-UNIFORM | 0.979268 | 0.996653 | 0.986222 | 0.980404 | 0.972592 | 0.976812 | 0.982614 | 0.983585 | 0.981686 | 0.981657 | 0.975019 | 0.975405 | 237.4 |
| EXP-TAB-003-CB-FULL-LOWLR | 0.979081 | 0.996295 | 0.985967 | 0.979816 | 0.971865 | 0.975818 | 0.982148 | 0.984100 | 0.980130 | 0.981542 | 0.973913 | 0.975679 | 1,074.5 |
| EXP-TAB-003-CB-FULL-CLASSWT | 0.977556 | 0.996296 | 0.985052 | 0.979277 | 0.971551 | 0.975026 | 0.981505 | 0.982980 | 0.979814 | 0.980445 | 0.974578 | 0.975108 | 743.9 |
| EXP-TAB-004-LGB-RADAR | 0.900826 | 0.977010 | 0.931300 | 0.922398 | 0.914561 | 0.916548 | 0.920006 | 0.925765 | 0.920994 | 0.914242 | 0.909028 | 0.901199 | 127.0 |
| EXP-TAB-005-LGB-OPTICAL | 0.976819 | 0.996036 | 0.984506 | 0.978743 | 0.969128 | 0.977220 | 0.982668 | 0.982730 | 0.980494 | 0.979972 | 0.973875 | 0.973108 | 253.2 |
| EXP-TAB-006-LGB-COMPACT | 0.970358 | 0.994155 | 0.979877 | 0.972388 | 0.962939 | 0.969015 | 0.976602 | 0.976944 | 0.971715 | 0.974996 | 0.963393 | 0.963368 | 125.3 |

The robust winner is full/uniform LightGBM even though invariant/default-weight LightGBM has the
slightly higher official mean (`+0.000107`). The robust winner's repeat standard deviation is
`0.000551`, mean log loss `0.063445`, Brier score `0.015605`, positive prediction rate
`0.399780`, and mean across-window standard deviation/disagreement `0.029141` / `0.016131`.
Its start-month combined scores for starts 1–9 are `0.980148, 0.977042, 0.979868, 0.982511,
0.977981, 0.977034, 0.975842, 0.970817, 0.959499`. Optical-gap scores are `0.985631`
(none), `0.978826` (one), and `0.975405` (2+). Severe optical limitation remains the weakest
completeness bin at `0.946206`.

Against the Phase 4 engineering reference, the robust winner improves F1 by `0.052460`, ROC-AUC
by `0.017588`, combined score by `0.038511`, robust score by `0.043757`, and worst fold by
`0.061996`. Improvements for lengths 4/5/6 are `0.037448 / 0.042768 / 0.040790`; early/mid/late
season gains are `0.040842 / 0.044097 / 0.043236`; the 2+-gap gain is `0.029873`.

### OOF diversity and retention

The complete audit contains 36 pair rows, 324 length/season/gap overlap rows, and 36 fixed 50/50
blend rows. No weights were optimized.

- Best-single LightGBM versus low-rate CatBoost: Pearson `0.998640`, Spearman `0.960720`, residual
  correlation `0.981329`, binary disagreement `0.004942`. CatBoost contributes 9 unique true
  positives and 9 unique false positives in the pair orientation; the fixed blend improves
  combined/robust to `0.986614 / 0.980647`. Retain CatBoost for demonstrated complementarity.
- Invariant LightGBM plus low-rate CatBoost gives the highest diagnostic-blend combined score,
  `0.987057`, with robust `0.980606`. Retain invariant LightGBM as the simpler/default-weight
  candidate.
- Best-single LightGBM versus optical expert: Pearson `0.995632`, residual `0.940944`, disagreement
  `0.009336`; its blend scores `0.986186 / 0.980469`. Retain the optical expert for the best
  standalone four-month score and modest robustness complementarity.
- Radar-only is genuinely diverse (Pearson `0.903649`, residual `0.571923`, disagreement
  `0.071389`) but its blend collapses to `0.980007 / 0.974490`. Reject it from the retained
  candidate registry despite keeping its complete OOF as a diagnostic.
- Reject relative, equal-original full LightGBM, class-weighted CatBoost, and compact physical
  models: each is weaker and its equal blend does not exceed the leading evidence sufficiently.

The retained registry contains full/uniform LightGBM, invariant/default-weight LightGBM,
low-rate CatBoost, and optical LightGBM. This is a Phase 5 candidate registry, not a final
ensemble selection.

### Importance and suspicious-feature audit

For the robust winner, LightGBM gain is grouped as 73.70% optical indices, 10.66% raw optical,
9.92% raw radar, and 3.51% radar-derived features. Missingness contributes less than 0.001% gain;
no missingness or calendar feature appears in the top-ten gain list. `optical__ndwi__max` alone
has 38.82% mean gain and the largest bounded fold-0 permutation log-loss increase (`0.024156`),
so it is explicitly flagged as dominant rather than silently removed. Low-rate CatBoost spreads
its native importance more broadly: 60.22% optical indices, 20.23% raw optical, 7.39% raw radar,
and 6.98% radar-derived.

Importance is associative, not causal. Domain specificity is deferred to Phase 7. The optical
expert places `metadata__end_month` in its top-ten split list, but its metadata-window gain share
is only 0.39%; it remains flagged for later domain diagnostics rather than removed post hoc.

### Artifacts, reproducibility, and limitations

Every serious run writes resolved configuration, compatibility manifest, both scientific
fingerprints, feature schema/list, metrics, fold/repeat/slice tables, 5,463-row original OOF,
43,704-row window predictions, native/group/permutation importance, model checksums, runtime, and
report under ignored `artifacts/experiments/<experiment_id>/`. Generated model files and
competition data remain untracked.

The per-experiment RSS field is a process-lifetime upper bound. In approved-list mode later runs
inherit the earlier process maximum; the Stage C batch-level 1,969.19 MiB measurement is reliable,
but later per-model values are conservative rather than isolated peaks. The initial exhaustive
diversity implementation redundantly rebuilt stress aggregations and was stopped without writing
partial artifacts; a tested mean-linearity cache produces exactly equivalent official/robust
blend components and completed in 84.10 seconds. No test probabilities, final full-data models,
calibration, optimized ensemble, or submission were created.

### Commands

```bash
python scripts/audit_data.py --config configs/base.yaml
python scripts/generate_windows.py --config configs/base.yaml
python scripts/build_features.py --config configs/base.yaml
python scripts/build_validation.py --config configs/base.yaml
python scripts/train_tabular.py --config configs/base.yaml \
  --experiment configs/experiments/screen_cb_medium.yaml --stage smoke
python scripts/train_tabular.py --config configs/base.yaml \
  --experiment configs/experiments/screen_lgb_conservative.yaml --stage smoke
python scripts/train_tabular.py --config configs/base.yaml \
  --approved-list configs/experiments/phase5_screening.yaml --stage screen
python scripts/train_tabular.py --config configs/base.yaml \
  --approved-list configs/experiments/phase5_representation_screening.yaml --stage screen
python scripts/train_tabular.py --config configs/base.yaml \
  --approved-list configs/experiments/phase5_full_confirmations.yaml --stage full
python scripts/train_tabular.py --config configs/base.yaml \
  --diversity-registry configs/experiments/phase5_all_stage_c_candidates.yaml
python -m compileall src tests
pytest
ruff check .
ruff format --check .
```

The exact next unblocked task is Phase 6: compare a compact masked temporal model against the
retained Phase 5 candidates using these same fold/window manifests, original-level OOF contract,
and robust selection criterion. Phase 5 stops here.

## Phase 6 — Compact masked temporal viability

**Status:** accepted. The compact temporal branch passed the predeclared standalone and blend
viability gates. No Transformer, second encoder family, external pretraining, threshold tuning,
or domain adaptation was introduced.

### Architecture and protocol

`EXP-SEQ-001-GRU-BCE` uses separate radar (8 channels), optical (10 channels), and optical-index
(14 channels) projections, an availability-aware radar/optical gate, cyclic absolute month,
relative position, explicit sensor masks, one 64-unit GRU layer, masked mean pooling, and a small
classification head. It contains exactly 26,329 trainable parameters. Padding, radar absence,
optical absence, and per-band missingness remain distinct.

All normalization is fitted on the current fold's training windows only. Each original contributes
equal total loss weight across its eight views. The run reused fold fingerprint
`4dbc9029f242c5ff4f8d2e23b0fb0d83334d993c1a4ecd7ce95e8e18c37ceece` and validation-window
fingerprint `89ef5e9a108a4cad09582db82ce1970dbf4873cbb3b01692c96a8fcc54b14492`.
Window probabilities are averaged to exactly 5,463 original/repeat OOF rows before scoring.

### Staged results

| Experiment | Stage | F1 | ROC-AUC | Combined | Robust | Worst fold | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| EXP-SEQ-001-GRU-BCE | smoke, 1 fold / 2 epochs | 0.948805 | 0.983649 | 0.962743 | 0.958420 | — | Engineering chain passed. |
| EXP-SEQ-001-GRU-BCE | screen, repeat 0 | 0.978694 | 0.996566 | 0.985843 | 0.982327 | 0.980376 | Promote. |
| EXP-SEQ-002-GRU-CONSISTENCY (`lambda=0.1`) | screen, repeat 0 | 0.978022 | 0.996232 | 0.985306 | 0.980744 | 0.975605 | Reject: slower and weaker than BCE. |
| EXP-SEQ-001-GRU-BCE | full, 15 folds | 0.979209 | 0.996965 | 0.986311 | 0.981208 | 0.974172 | Retain: robust winner. |

The full BCE run improves the former best single LightGBM by `+0.000089` combined,
`+0.000804` robust, `+0.001580` worst-fold score, `-0.001515` log loss, and `-0.000345` Brier
score. Repeat combined-score standard deviation is `0.000333`. Total runtime was 308.69 seconds
on CPU with four threads; fold training totalled 257.61 seconds and peak RSS was 537.31 MiB.
Median best epoch was 23.

### Stress results

| Slice | Combined score |
|---|---:|
| 4 months | 0.978934 |
| 5 months | 0.985144 |
| 6 months | 0.984023 |
| Early season | 0.981636 |
| Mid season | 0.982662 |
| Late season | 0.975854 |
| No optical gaps | 0.986532 |
| One optical gap | 0.977580 |
| Two or more optical gaps | 0.975594 |
| Severely optical-limited | 0.944739 |
| Start month 9 | 0.958014 |

The known month-9 and severely optical-limited weaknesses remain. The temporal model improves the
robust criterion without eliminating those domain-risk slices, so Phase 7 remains necessary.

### Sensor ablation

Mean original-level validation log loss across the 15 folds is `0.061961` for the complete model.
Removing monthly indices raises it to `0.416356`; removing raw optical inputs raises it to
`0.337012`; removing radar raises it to `0.159206`. Radar is weak standalone but contributes
material conditional signal inside the fused temporal model.

### OOF diversity and fixed blends

The complete GRU and robust-winner LightGBM align on all 5,463 OOF rows. Their Pearson probability
correlation is `0.990933`, Spearman correlation `0.887802`, residual correlation `0.870960`, and
binary disagreement `0.010617`. Each model uniquely corrects 29 classifications made incorrectly
by the other; 62 errors are shared.

No blend weights were optimized. A fixed 50/50 blend reaches F1 `0.981493`, ROC-AUC `0.997225`,
combined `0.987786`, and robust `0.981581`; this improves the best tree robust score by
`0.001177` and the standalone temporal robust score by `0.000373`. A fixed 70% tree / 30%
temporal blend scores `0.987080 / 0.981087` combined/robust. The 50/50 blend is retained as Phase
8 evidence, not as a final selected submission model.

### Decision and limitations

Retain `EXP-SEQ-001-GRU-BCE` as the strongest standalone robust candidate and retain the
predeclared 50/50 tree/temporal blend as the best current diagnostic combination. Reject the
cross-window consistency objective after screening. Phase 6 makes no test predictions and no
leaderboard claim. Tree OOF was regenerated from the exact committed Phase 5 configuration to
measure temporal diversity because generated Phase 5 artifacts are intentionally ignored by Git.

Commands:

```bash
python scripts/train_temporal.py --config configs/base.yaml \
  --experiment configs/experiments/exp_seq_001_gru_bce.yaml --stage smoke --overwrite
python scripts/train_temporal.py --config configs/base.yaml \
  --experiment configs/experiments/exp_seq_001_gru_bce.yaml --stage screen --overwrite
python scripts/train_temporal.py --config configs/base.yaml \
  --experiment configs/experiments/exp_seq_001_gru_bce.yaml --stage full --overwrite
python scripts/train_temporal.py --config configs/base.yaml \
  --experiment configs/experiments/exp_seq_002_gru_consistency.yaml --stage screen --overwrite
python scripts/analyze_temporal_diversity.py --config configs/base.yaml \
  --temporal-artifact artifacts/experiments/EXP-SEQ-001-GRU-BCE \
  --tree-artifact artifacts/experiments/EXP-TAB-003-LGB-FULL-UNIFORM \
  --output-dir artifacts/experiments/phase6_selection
```

The exact next unblocked task is Phase 7: diagnose feature- and representation-level train/test
domain shift, with special attention to `optical__ndwi__max`, month 9, and severely
optical-limited windows. Adaptation remains optional and must improve the immutable robust label
validation before retention.
