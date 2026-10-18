# CV Metrics Summary

- Model directory: `runs/base_cnn`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5179 | 0.4888 | 0.0291 |
| 1 | 131 | 0.5058 | 0.4745 | 0.0313 |
| 2 | 131 | 0.4951 | 0.4655 | 0.0296 |
| 3 | 130 | 0.5286 | 0.5124 | 0.0163 |
| 4 | 130 | 0.5495 | 0.5070 | 0.0426 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5194 +/- 0.0188 | 0.4896 +/- 0.0181 |
| jaccard | NA | NA |
| precision | NA | NA |
| recall | NA | NA |
| abs_volume_diff_ratio | NA | NA |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
