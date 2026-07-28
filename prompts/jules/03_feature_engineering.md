# Task 03 — Physics-informed feature engineering

## Objective

Implement reusable raw, relative-time and aggregated representations for incomplete radar/optical windows.

## Required work

1. Implement numerically stable monthly indices:
   - NDVI, NDWI, MNDWI, NDMI and NBR;
   - selected red-edge contrasts;
   - `VV - VH`, ratios and radar magnitude interactions.
2. Implement per-band and per-index window statistics over available observations:
   - count, mean, median, standard deviation, min, max;
   - 25th and 75th percentiles, range;
   - first-to-last delta and linear slope when identifiable.
3. Add missingness and reliability features:
   - radar-month count;
   - optical-month count;
   - per-band valid count;
   - window length and absolute start month;
   - cyclic month encoding.
4. Produce:
   - a flat table for tree models;
   - a masked tensor representation for temporal models.
5. Maintain a feature manifest with names, formulas, required source bands and missing-value policy.

## Scientific constraints

- Fit no global statistics in this task.
- Never divide without an epsilon and explicit invalid-value handling.
- Missing values must remain missing or masked; do not replace them with physically meaningful zero.
- Do not use the target in feature construction.

## Acceptance criteria

Synthetic tests must verify formulas, all-missing windows, one-observation slopes, mask shapes and deterministic feature order.

```bash
pytest tests -k "feature or index or tensor"
ruff check .
ruff format --check .
```
