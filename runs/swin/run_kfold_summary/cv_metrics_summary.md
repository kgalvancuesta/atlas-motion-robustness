# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/swin`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.4959 | 0.4549 | 0.0410 |
| 1 | 131 | 0.5130 | 0.4741 | 0.0389 |
| 2 | 131 | 0.4999 | 0.4510 | 0.0488 |
| 3 | 130 | 0.5419 | 0.4932 | 0.0487 |
| 4 | 130 | 0.5706 | 0.5064 | 0.0642 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5242 +/- 0.0282 | 0.4759 +/- 0.0214 |
| jaccard | 0.4069 +/- 0.0213 | 0.3624 +/- 0.0158 |
| precision | 0.6723 +/- 0.0660 | 0.6190 +/- 0.0685 |
| recall | 0.5258 +/- 0.0461 | 0.4797 +/- 0.0379 |
| abs_volume_diff_ratio | 1.9118 +/- 2.2617 | 2.1179 +/- 2.1998 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
