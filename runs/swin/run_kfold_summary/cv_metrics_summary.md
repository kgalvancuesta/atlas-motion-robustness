# CV Metrics Summary

- Model directory: `runs/swin`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5401 | 0.4861 | 0.0540 |
| 1 | 131 | 0.5046 | 0.4545 | 0.0501 |
| 2 | 131 | 0.5273 | 0.4654 | 0.0619 |
| 3 | 130 | 0.5593 | 0.4978 | 0.0615 |
| 4 | 130 | 0.5625 | 0.4997 | 0.0628 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5388 +/- 0.0214 | 0.4807 +/- 0.0179 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
