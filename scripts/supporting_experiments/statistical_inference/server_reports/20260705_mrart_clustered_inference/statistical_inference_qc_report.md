# Statistical Inference QC Report

## Inputs

- ATLAS metric files discovered: 80.
- ATLAS split JSON: `splits/cv/atlas_5fold_lesion_quartile_excluding_known_issues.json`.
- MR-ART fold CSV: `scripts/supporting_experiments/mrart_hallucination/server_reports/20260703_mrart_paper_stats/fold_lcc_stats/mrart_fold_lcc_stats_long.csv`.

## ATLAS QC

- Subject count: 653.
- Duplicate subject/model/training/evaluation rows: 0.
- Fold assignment mismatches: 0.
- Invalid Dice rows: 0.
- Failure flag mismatches: 0.
- Strong-Dice flag mismatches: 0.
- Source metric summary issues: 0.
- Per-file coverage issues: 0.

## MR-ART QC

- Row count: 8400 expected 8400.
- Status counts: {'success': 8400}.
- Non-success rows: 0.
- Architectures: ['mednext', 'swin_unetr'].
- Training conditions: ['augmented', 'standard'].
- Acquisition types: ['headmotion1', 'headmotion2', 'standard'].
- Folds: [0, 1, 2, 3, 4].
- Threshold values: [0.5].
- Invalid voxel-volume rows: 0.

## Statistical Settings

- Bootstrap iterations: 10000.
- Seed: 9001.
- Bootstrap CI: percentile 95% CI.
- ATLAS bootstrap: paired subject resampling.
- ATLAS between-architecture tests: paired by held-out subject and fold, conditional on fixed trained checkpoints.
- MR-ART bootstrap: clustered paired bootstrap over subject_id, retaining fold rows through cluster sums/counts.
- Wilcoxon: scipy signed-rank test with zero_method='pratt'; all-zero differences return p=1.
- Multiple-comparison correction: Holm adjustment within named endpoint/family groups.

## Warnings And Assumptions

- ATLAS source JSONs did not include volume_similarity; abs_volume_diff_ratio is preserved separately.
- Between-architecture inference is paired over held-out subjects for fixed checkpoints; it does not estimate repeated-seed or retraining variability.
