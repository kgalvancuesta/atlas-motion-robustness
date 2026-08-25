# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/base_cnn`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5115 | 0.4525 | 0.0590 |
| 1 | 131 | 0.4976 | 0.4232 | 0.0744 |
| 2 | 131 | 0.4975 | 0.4612 | 0.0362 |
| 3 | 130 | 0.5182 | 0.4773 | 0.0408 |
| 4 | 130 | 0.5690 | 0.5336 | 0.0355 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5188 +/- 0.0264 | 0.4696 +/- 0.0365 |
| jaccard | 0.4023 +/- 0.0210 | 0.3584 +/- 0.0303 |
| precision | 0.6275 +/- 0.0212 | 0.5623 +/- 0.0484 |
| recall | 0.5465 +/- 0.0324 | 0.5003 +/- 0.0332 |
| abs_volume_diff_ratio | 2.7156 +/- 3.3732 | 3.4926 +/- 2.9598 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
