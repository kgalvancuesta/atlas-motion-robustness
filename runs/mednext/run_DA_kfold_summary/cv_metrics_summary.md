# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/mednext`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5409 | 0.5104 | 0.0305 |
| 1 | 131 | 0.5040 | 0.4726 | 0.0314 |
| 2 | 131 | 0.5390 | 0.5046 | 0.0344 |
| 3 | 130 | 0.5575 | 0.5367 | 0.0208 |
| 4 | 130 | 0.5745 | 0.5471 | 0.0274 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5432 +/- 0.0234 | 0.5143 +/- 0.0262 |
| jaccard | 0.4286 +/- 0.0189 | 0.4004 +/- 0.0208 |
| precision | 0.6810 +/- 0.0403 | 0.6395 +/- 0.0365 |
| recall | 0.5657 +/- 0.0519 | 0.5362 +/- 0.0508 |
| abs_volume_diff_ratio | 2.6750 +/- 3.5687 | 2.7447 +/- 3.5926 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
