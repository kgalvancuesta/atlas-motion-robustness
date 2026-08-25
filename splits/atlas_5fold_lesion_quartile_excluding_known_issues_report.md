# ATLAS 5-Fold Lesion-Quartile CV Splits

## Summary

- Historical subject-ID source: `splits/atlas_train_dev_test.json` (generation input intentionally not retained)
- Total labeled subjects before exclusion: `655`
- Total subjects after exclusion: `653`
- Stratification scheme: `quantile_k4`
- Quartile edges after exclusion: `[13, 1159, 5290, 36236, 496656]`
- Random state: `9001`

## Exclusions

- `sub-r039s002`: Excluded from CV generation: ATLAS issue subject with invalid/ambiguous lesion mask values (max ~= 0.01 instead of a valid binary lesion-mask convention).
- `sub-r009s003`: Excluded from CV generation: known ATLAS labeling issue and manual review confirmed unreliable ground truth.

## Per-Fold Counts

| Fold | Train | Val | Test |
| --- | ---: | ---: | ---: |
| 0 | 417 | 105 | 131 |
| 1 | 417 | 105 | 131 |
| 2 | 417 | 105 | 131 |
| 3 | 418 | 105 | 130 |
| 4 | 418 | 105 | 130 |

## Per-Fold Stratification Counts

| Fold | Train bins | Val bins | Test bins |
| --- | --- | --- | --- |
| 0 | {'[13, 1159)': 104, '[1159, 5290)': 104, '[5290, 36236)': 104, '[36236, 496656]': 105} | {'[13, 1159)': 26, '[1159, 5290)': 26, '[5290, 36236)': 27, '[36236, 496656]': 26} | {'[13, 1159)': 33, '[1159, 5290)': 33, '[5290, 36236)': 32, '[36236, 496656]': 33} |
| 1 | {'[13, 1159)': 104, '[1159, 5290)': 104, '[5290, 36236)': 104, '[36236, 496656]': 105} | {'[13, 1159)': 26, '[1159, 5290)': 27, '[5290, 36236)': 26, '[36236, 496656]': 26} | {'[13, 1159)': 33, '[1159, 5290)': 32, '[5290, 36236)': 33, '[36236, 496656]': 33} |
| 2 | {'[13, 1159)': 104, '[1159, 5290)': 104, '[5290, 36236)': 104, '[36236, 496656]': 105} | {'[13, 1159)': 26, '[1159, 5290)': 27, '[5290, 36236)': 26, '[36236, 496656]': 26} | {'[13, 1159)': 33, '[1159, 5290)': 32, '[5290, 36236)': 33, '[36236, 496656]': 33} |
| 3 | {'[13, 1159)': 105, '[1159, 5290)': 104, '[5290, 36236)': 104, '[36236, 496656]': 105} | {'[13, 1159)': 26, '[1159, 5290)': 26, '[5290, 36236)': 26, '[36236, 496656]': 27} | {'[13, 1159)': 32, '[1159, 5290)': 33, '[5290, 36236)': 33, '[36236, 496656]': 32} |
| 4 | {'[13, 1159)': 104, '[1159, 5290)': 104, '[5290, 36236)': 105, '[36236, 496656]': 105} | {'[13, 1159)': 27, '[1159, 5290)': 26, '[5290, 36236)': 26, '[36236, 496656]': 26} | {'[13, 1159)': 32, '[1159, 5290)': 33, '[5290, 36236)': 32, '[36236, 496656]': 33} |

## Verification Checks Passed

- No excluded subject appears in any generated train/val/test split.
- No duplicate subject IDs appear within any generated train/val/test split.
- Fold 0: train/val/test are disjoint and cover the full included subject set.
- Fold 1: train/val/test are disjoint and cover the full included subject set.
- Fold 2: train/val/test are disjoint and cover the full included subject set.
- Fold 3: train/val/test are disjoint and cover the full included subject set.
- Fold 4: train/val/test are disjoint and cover the full included subject set.
- All folds use the same included subject universe.
- Each included subject appears in outer test exactly once across folds.
- Outer test fold sizes differ by at most 1 subject: [131, 131, 131, 130, 130]
- Validation sizes differ by at most 1 subject: [105, 105, 105, 105, 105]
- Test stratification counts differ by at most 1 per bin across folds.
- Val stratification counts differ by at most 1 per bin across folds.

## Usage Note

Use this same split file for every model architecture to ensure paired comparisons across architectures.
