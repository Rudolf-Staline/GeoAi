# Jules task queue

Execute these prompts in numerical order. Each prompt is one bounded task and one separate pull request. Do not launch a dependent task until the previous PR has been reviewed and merged.

| Order | Prompt | Depends on | Gate |
|---:|---|---|---|
| 01 | Data audit and ingestion | scaffold | schema and audit tests pass |
| 02 | Temporal window generator | 01 | leakage tests pass |
| 03 | Feature engineering | 02 | numerical and mask tests pass |
| 04 | Validation protocol | 02–03 | deterministic OOF smoke run |
| 05 | Tree baselines | 04 | reproducible baseline report |
| 06 | Temporal model | 04 | tree baseline remains available |
| 07 | Domain shift | 05–06 | ablation proves benefit or records rejection |
| 08 | Ensemble and delivery | accepted experts | final notebook and submission validation |

Copy the full content of a prompt into Jules. Do not combine prompts. `AGENTS.md` remains authoritative if a task prompt is ambiguous.
