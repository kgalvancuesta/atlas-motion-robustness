# CV Metrics Summary

- Model directory: `runs/mednext`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5031 | 0.4637 | 0.0394 |
| 1 | 131 | 0.5445 | 0.5102 | 0.0343 |
| 2 | 131 | 0.5385 | 0.5078 | 0.0306 |
| 3 | 130 | 0.5242 | 0.5093 | 0.0149 |
| 4 | 130 | 0.5457 | 0.5089 | 0.0369 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5312 +/- 0.0160 | 0.5000 +/- 0.0182 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
