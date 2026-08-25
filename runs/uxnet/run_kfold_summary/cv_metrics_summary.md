# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/uxnet`
- Run prefix: `run_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5473 | 0.4680 | 0.0793 |
| 1 | 131 | 0.4835 | 0.4556 | 0.0279 |
| 2 | 131 | 0.4929 | 0.4439 | 0.0490 |
| 3 | 130 | 0.5447 | 0.4969 | 0.0478 |
| 4 | 130 | 0.5709 | 0.5328 | 0.0381 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5279 +/- 0.0338 | 0.4794 +/- 0.0320 |
| jaccard | 0.4123 +/- 0.0246 | 0.3679 +/- 0.0238 |
| precision | 0.6704 +/- 0.0428 | 0.6049 +/- 0.0531 |
| recall | 0.5358 +/- 0.0361 | 0.4964 +/- 0.0350 |
| abs_volume_diff_ratio | 2.4118 +/- 3.3373 | 3.1967 +/- 2.9519 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
