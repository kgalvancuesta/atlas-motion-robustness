# MR-ART supporting experiments

This directory consolidates MR-ART-specific execution, validation, and paper
analysis. Shared model builders and TorchIO augmentation definitions remain in
the repository-level `scripts/` directory.

Current fold-level metrics are under
`server_reports/det-fix_v1.1.0/fold_lcc_stats/`. They cover all four model
families, both training conditions, five folds, and 140 complete triplets
(16,800 rows). These metrics were exported from the full-volume pipeline in
`scripts/reproducibility/mrart.py`; the entrypoints below preserve the legacy
analysis workflow. Source and input-array SHA-256 values are retained in the CSV.

Live scripts:

- `scripts/run_mrart_negative_control.sh` and
  `scripts/infer_mrart_negative_control.py`: deterministic fold-level MR-ART
  negative-control inference.
- `scripts/validate_augmentations.py`: qualitative comparison of simulated
  motion against matched MR-ART acquisitions.
- `scripts/run_mrart_fold_lcc_stats_no_outputs.py`: fold-level inference and
  connected-component statistics without bulky prediction outputs.
- `scripts/generate_mrart_lcc_from_predictions.py`: secondary LCC analysis from
  existing predictions.
- `scripts/verify_mrart_voxel_volume_headers.py`: image-header and physical-volume
  verification.
- `scripts/build_mrart_hallucination_csvs.py` and
  `scripts/run_mrart_paper_stats.sh`: paper-table construction and orchestration.

Local MRI inputs and inference outputs belong under `data/` and `runs/`. The
augmentation validator writes local comparisons under the Git-ignored
`outputs/supporting_experiments/mrart/` tree. The
versioned `csv/` and `server_reports/` directories here contain lightweight
evidence. Source and checkpoint hashes identify the archived execution artifacts.
