# ATLAS Lesion Voxel-Count Stratification Failure Analysis

## Method

- Primary strata are the pre-existing CV lesion voxel-count quartiles from `splits/cv/atlas_5fold_lesion_quartile_excluding_known_issues.json`.
- Bins are `[13, 1159)`, `[1159, 5290)`, `[5290, 36236)`, and `[36236, 496656]`; the final bin is inclusive.
- Catastrophic failure is defined as Dice < 0.1.
- Rows use held-out test-set predictions only.
- Physical volume conversion is not used for primary stratification.

## High-Level Result

- [13, 1159): mean clean-to-artifact Dice drop across model/training groups = 0.04098242058; mean failure-rate increase = 0.05291411043.
- [1159, 5290): mean clean-to-artifact Dice drop across model/training groups = 0.03869818974; mean failure-rate increase = 0.04294478528.
- [5290, 36236): mean clean-to-artifact Dice drop across model/training groups = 0.0456191526; mean failure-rate increase = 0.0291411043.
- [36236, 496656]: mean clean-to-artifact Dice drop across model/training groups = 0.03154503179; mean failure-rate increase = 0.009146341463.

Interpretation should use Dice drop, failure rate, and strong-Dice rate together. Small lesions can drive many Dice failures, but the bin-wise delta table is the direct test of whether artifact degradation persists across lesion sizes.

## Diagnostics

- Source metric files used: 80.
- Total rows written: 10448.
- Duplicate subject/model/training/evaluation rows: 0.
- Invalid Dice rows: 0.
- Unmatched clean/artifact pair rows: 0.
- Physical volume conversion used: False.

## Outputs

- `long_per_case_csv`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_per_case_metrics.csv`
- `pooled_bin_summary_csv`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_bin_summary_pooled.csv`
- `fold_aware_bin_summary_csv`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_bin_summary_fold_aware.csv`
- `robustness_delta_csv`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_bin_robustness_delta.csv`
- `paper_table_compact_csv`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_paper_table_compact.csv`
- `paper_table_delta_compact_csv`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_paper_table_delta_compact.csv`
- `diagnostics_json`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_stratification_diagnostics.json`
- `report_md`: `scripts/supporting_experiments/atlas_lesion_size_stratification/csv/20260704T034046Z/atlas_lesion_size_stratification_report.md`
