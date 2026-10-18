# CV Metrics Summary

- Model directory: `runs/swin`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5234 | 0.4918 | 0.0316 |
| 1 | 131 | 0.4808 | 0.4533 | 0.0275 |
| 2 | 131 | 0.5278 | 0.4970 | 0.0308 |
| 3 | 130 | 0.5336 | 0.5090 | 0.0246 |
| 4 | 130 | 0.5568 | 0.5189 | 0.0379 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5245 +/- 0.0247 | 0.4940 +/- 0.0224 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
