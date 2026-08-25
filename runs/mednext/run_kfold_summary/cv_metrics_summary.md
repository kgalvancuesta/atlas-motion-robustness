# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/mednext`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5198 | 0.4678 | 0.0520 |
| 1 | 131 | 0.5092 | 0.4766 | 0.0327 |
| 2 | 131 | 0.5297 | 0.4825 | 0.0472 |
| 3 | 130 | 0.5349 | 0.4936 | 0.0413 |
| 4 | 130 | 0.5853 | 0.5410 | 0.0443 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5358 +/- 0.0262 | 0.4923 +/- 0.0257 |
| jaccard | 0.4214 +/- 0.0221 | 0.3802 +/- 0.0209 |
| precision | 0.6852 +/- 0.0506 | 0.6271 +/- 0.0450 |
| recall | 0.5489 +/- 0.0472 | 0.5033 +/- 0.0458 |
| abs_volume_diff_ratio | 2.5979 +/- 3.4413 | 2.0913 +/- 2.1287 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
