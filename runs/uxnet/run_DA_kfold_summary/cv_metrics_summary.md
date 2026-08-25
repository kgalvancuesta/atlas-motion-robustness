# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/uxnet`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5322 | 0.5010 | 0.0312 |
| 1 | 131 | 0.5082 | 0.4724 | 0.0358 |
| 2 | 131 | 0.4999 | 0.4728 | 0.0271 |
| 3 | 130 | 0.5492 | 0.5218 | 0.0274 |
| 4 | 130 | 0.5473 | 0.5083 | 0.0390 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5274 +/- 0.0201 | 0.4953 +/- 0.0197 |
| jaccard | 0.4127 +/- 0.0173 | 0.3826 +/- 0.0164 |
| precision | 0.6607 +/- 0.0454 | 0.6165 +/- 0.0325 |
| recall | 0.5464 +/- 0.0178 | 0.5185 +/- 0.0150 |
| abs_volume_diff_ratio | 2.7290 +/- 4.1000 | 2.8699 +/- 4.2009 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
