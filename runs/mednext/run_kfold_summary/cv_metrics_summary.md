# CV Metrics Summary

- Model directory: `runs/mednext`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5210 | 0.4832 | 0.0378 |
| 1 | 131 | 0.5038 | 0.4424 | 0.0615 |
| 2 | 131 | 0.5498 | 0.5068 | 0.0430 |
| 3 | 130 | 0.5562 | 0.5301 | 0.0261 |
| 4 | 130 | 0.5716 | 0.5348 | 0.0368 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5405 +/- 0.0246 | 0.4995 +/- 0.0340 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
