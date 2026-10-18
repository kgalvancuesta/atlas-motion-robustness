# CV Metrics Summary

- Model directory: `runs/uxnet`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5105 | 0.4726 | 0.0379 |
| 1 | 131 | 0.4956 | 0.4497 | 0.0459 |
| 2 | 131 | 0.5246 | 0.4778 | 0.0469 |
| 3 | 130 | 0.5173 | 0.4811 | 0.0362 |
| 4 | 130 | 0.5763 | 0.5302 | 0.0461 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5249 +/- 0.0274 | 0.4823 +/- 0.0263 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
