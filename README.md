# GeoAI Aquaculture Pond Identification

Repository for the Zindi **GeoAI Aquaculture Pond Identification Challenge**.

The task is binary classification at 10 m pixel level: predict whether each test row corresponds to an aquaculture pond using incomplete Sentinel-1 and Sentinel-2 temporal observations.

Competition page: https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge

## Scientific position

The main difficulty is not the choice of classifier. Train rows contain twelve complete monthly observations, while test rows expose only a consecutive window of four, five, or six radar months and may contain additional optical gaps. Every experiment must therefore be evaluated under a validation protocol that reproduces test-time missingness.

The project will progress through explicit ablations:

1. data audit and schema validation;
2. leakage-safe temporal windows;
3. relative-time and physics-informed features;
4. robust validation protocols;
5. tree-based baselines;
6. compact radar/optical temporal model;
7. domain-shift experiments;
8. calibrated ensemble and final notebook.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

The competition CSV files are deliberately excluded from Git. Place them in `data/raw/`:

```text
data/raw/Train.csv
data/raw/Test.csv
data/raw/SampleSubmission.csv
data/raw/Trustworthiness_Evaluation.pdf
```

## Phase 1 data audit

Run the schema and missingness audit before any augmentation or model training:

```bash
python scripts/audit_data.py --config configs/base.yaml
```

The command validates all three competition CSV files, converts `-9999` to in-memory missing
values, and writes deterministic JSON, CSV, and Markdown summaries under
`artifacts/data_audit/`. Raw competition files are never modified, and generated artifacts remain
excluded from Git.

## Phase 2 temporal windows

Generate the configured deterministic sampled views and metadata audit after assigning folds to
the original rows:

```bash
python scripts/generate_windows.py --config configs/base.yaml
```

Use `--mode exhaustive` to generate all 24 consecutive windows per original row. The default
sampled mode draws only boolean availability templates derived from test missingness; test feature
values are never copied. Manifests and summaries are written under
`artifacts/temporal_windows/`, while raw window values remain in memory and generated artifacts
remain excluded from Git.

## Phase 3 feature representations

Build aligned tabular and masked-sequence representations from the configured sampled windows:

```bash
python scripts/build_features.py --config configs/base.yaml
```

Use `--mode exhaustive` to build features for all 24 windows per original training row. The
tabular output contains relative-position values, valid-only temporal aggregates, and explicit
window/missingness metadata. The sequence output keeps radar, optical, spectral-index,
calendar-month, per-band, sensor, and padding masks separate. No imputer or other dataset-wide
preprocessor is fitted.

The CLI writes only aggregate summaries, a machine-readable feature registry, fingerprints, and
run provenance under `artifacts/features/`; competition rows, labels, and IDs are not persisted.
The default real-data schema has 688 tabular features and sequence arrays with 8 radar channels,
10 raw optical channels, 14 optical-index channels, 2 cyclic month channels, and a maximum length
of 6.

## Repository map

```text
configs/                 Versioned experiment configuration
data/                    Data placement contract; raw data ignored
artifacts/               Generated metrics, OOF predictions, models and reports
docs/                    Architecture and scientific protocol
notebooks/               Exploratory and final competition notebooks
prompts/jules/           Delegation prompts, executed in numerical order
src/geoai_aquaculture/   Reusable implementation
scripts/                 Reproducible CLI entrypoints
tests/                   Unit and leakage-safety tests
```

Read `AGENTS.md` before changing code. It is the authoritative scientific and engineering contract.
