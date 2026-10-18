# CV Metrics Summary

- Model directory: `runs/uxnet`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5105 | 0.4656 | 0.0449 |
| 1 | 131 | 0.4717 | 0.4449 | 0.0268 |
| 2 | 131 | 0.5199 | 0.4939 | 0.0260 |
| 3 | 130 | 0.5286 | 0.5023 | 0.0263 |
| 4 | 130 | 0.5465 | 0.5158 | 0.0307 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5154 +/- 0.0249 | 0.4845 +/- 0.0257 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
