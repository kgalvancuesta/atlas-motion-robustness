# MR-ART supporting experiments

This directory consolidates MR-ART-specific execution, validation, and paper
analysis. Shared model builders and TorchIO augmentation definitions remain in
the repository-level `scripts/` directory.

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
timestamped `csv/` and `server_reports/` directories here are retained
lightweight evidence; paths recorded inside them describe the original runs.
