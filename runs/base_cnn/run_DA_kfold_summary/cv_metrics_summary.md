# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/base_cnn`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5101 | 0.4876 | 0.0226 |
| 1 | 131 | 0.4851 | 0.4715 | 0.0137 |
| 2 | 131 | 0.5311 | 0.5012 | 0.0299 |
| 3 | 130 | 0.5236 | 0.4644 | 0.0593 |
| 4 | 130 | 0.5427 | 0.5096 | 0.0331 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5185 +/- 0.0198 | 0.4868 +/- 0.0171 |
| jaccard | 0.4021 +/- 0.0158 | 0.3724 +/- 0.0135 |
| precision | 0.6284 +/- 0.0317 | 0.5973 +/- 0.0453 |
| recall | 0.5404 +/- 0.0482 | 0.5116 +/- 0.0496 |
| abs_volume_diff_ratio | 2.3780 +/- 3.0716 | 3.2079 +/- 3.3290 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
