# Task 01 — Data ingestion and audit

## Objective

Implement the reliable data-ingestion layer and a reproducible audit for the supplied GeoAI competition CSV files. Do not train any model.

## Required work

1. Create typed modules under `src/geoai_aquaculture/data/` for:
   - configuration-backed path resolution;
   - loading train, test and sample submission;
   - schema validation;
   - temporal-column parsing into `(band, month)` metadata;
   - conversion of `-9999` to missing values without mutating source files.
2. Validate:
   - unique IDs;
   - binary train target;
   - exact train/test feature alignment;
   - expected sample-submission columns and ID order;
   - 12 months and the expected radar/optical bands.
3. Produce a CLI script `scripts/audit_data.py` that writes machine-readable JSON and concise CSV summaries to `artifacts/data_audit/`.
4. Report missingness by dataset, band, month, sensor and row.
5. Add focused tests using synthetic fixtures. Tests must not require the private competition CSV files.

## Scientific constraints

- Treat `-9999` only as missingness.
- Preserve original IDs and column order.
- Do not infer labels, build features or train models.
- Do not commit any real competition rows.

## Acceptance criteria

```bash
python scripts/audit_data.py --config configs/base.yaml
pytest
ruff check .
ruff format --check .
```

The audit command must fail clearly on a malformed schema and must produce deterministic outputs on valid input.

## Completion report

List files changed, audit outputs, tests run, discovered schema facts and anything that should change in `configs/base.yaml`.
