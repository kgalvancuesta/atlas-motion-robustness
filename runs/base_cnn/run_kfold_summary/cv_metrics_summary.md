# CV Metrics Summary

- Model directory: `runs/base_cnn`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5015 | 0.4486 | 0.0530 |
| 1 | 131 | 0.4937 | 0.4339 | 0.0598 |
| 2 | 131 | 0.5110 | 0.4599 | 0.0511 |
| 3 | 130 | 0.5364 | 0.4915 | 0.0449 |
| 4 | 130 | 0.5650 | 0.5094 | 0.0556 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5215 +/- 0.0261 | 0.4687 +/- 0.0278 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
