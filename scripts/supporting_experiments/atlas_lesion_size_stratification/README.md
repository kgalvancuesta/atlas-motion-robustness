# ATLAS lesion-size stratification

This supporting experiment contains the analysis used to select and validate
the lesion-volume quartiles behind the paper's deterministic five-fold split.

Live scripts:

- `scripts/analyze_lesion_volume_stratification.py`: measure the labeled-subject
  lesion-volume distribution and emit diagnostic evidence.
- `scripts/stress_test_lesion_volume_stratification.py`: compare candidate
  stratification schemes.
- `scripts/create_cv_splits.py`: deterministically regenerate the canonical
  five-fold split.
- `scripts/build_atlas_lesion_size_failure_tables.py`: build fold-aware
  lesion-size performance tables from completed runs.

The canonical fold assignments remain versioned under the repository-level
`splits/` directory. Generate into a temporary output directory when auditing
reproducibility; do not overwrite the canonical files casually.

New analysis and stress-test working outputs default to the Git-ignored
`outputs/supporting_experiments/atlas_lesion_size_stratification/` tree.

The timestamped `csv/` and `validation/` directories are retained evidence from
completed analyses.
