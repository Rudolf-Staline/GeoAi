# Data contract

Competition data must stay local and must never be committed.

Expected layout:

```text
data/
├── raw/
│   ├── Train.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   └── Trustworthiness_Evaluation.pdf
├── interim/
└── processed/
```

Known schema:

- `Train.csv`: `ID`, `label`, and 144 temporal features;
- `Test.csv`: `ID` and the same 144 temporal features;
- `SampleSubmission.csv`: `ID`, `TargetF1`, `TargetRAUC`;
- 12 monthly positions;
- radar bands: `VH`, `VV`;
- optical bands: `blue`, `green`, `nir`, `nira`, `re1`, `re2`, `re3`, `red`, `swir1`, `swir2`;
- missing values are encoded with `-9999`.

No derived file should lose the original `ID` or its fold assignment.
