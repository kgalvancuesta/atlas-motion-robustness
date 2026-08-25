# CV Metrics Summary

- Model directory: `<REPO_ROOT>/runs/swin`
- Run prefix: `run_DA_kfold`
- Expected pooled test subjects: `653`
- Warnings: `0`

## Per-fold Summary

| Fold | Test Subjects | Clean Dice | Augmented Dice | Robustness Delta |
| --- | --- | --- | --- | --- |
| 0 | 131 | 0.5296 | 0.4914 | 0.0382 |
| 1 | 131 | 0.5025 | 0.4771 | 0.0254 |
| 2 | 131 | 0.5378 | 0.5071 | 0.0308 |
| 3 | 130 | 0.5603 | 0.5307 | 0.0296 |
| 4 | 130 | 0.5637 | 0.5302 | 0.0336 |

## Fold-level Aggregates

| Metric | Clean | Augmented |
| --- | --- | --- |
| dice | 0.5388 +/- 0.0223 | 0.5073 +/- 0.0211 |
| jaccard | 0.4227 +/- 0.0198 | 0.3934 +/- 0.0190 |
| precision | 0.6449 +/- 0.0540 | 0.6171 +/- 0.0563 |
| recall | 0.5635 +/- 0.0459 | 0.5266 +/- 0.0411 |
| abs_volume_diff_ratio | 1.8564 +/- 2.1353 | 1.9290 +/- 2.1662 |

## Coverage Checks

| Set | Observed | Duplicates | Missing | Each Subject Exactly Once |
| --- | --- | --- | --- | --- |
| clean | 653 | 0 | 0 | yes |
| augmented | 653 | 0 | 0 | yes |
