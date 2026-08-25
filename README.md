# Lesion Segmentation Robustness Experiment on ATLAS

This repository provides the training, inference, and evaluation pipeline used
to compare four 3D lesion-segmentation architectures on ATLAS v2.0 (MNI space).

The paper experiment used Python 3.9, 45 training epochs with early stopping,
two-GPU distributed training, and the fixed cross-validation split committed in
`splits/`. The augmented condition applied the `motion_consistent` preset to a
fixed 50% of training subjects (`--augment_frac 0.5`). Exact dependency versions
are pinned in `requirements.txt`.

## Architecture and implementation attribution

The experiments use four published model families. Their architecture papers,
implementation sources, retained-code provenance, and license status are recorded
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). In summary:

- **3D U-Net:** MONAI `UNet`; Cicek et al. (MICCAI 2016) and MONAI.
- **MedNeXt-S:** adapted from the official DKFZ MedNeXt implementation; Roy
  et al. (MICCAI 2023). The retained source is Apache-2.0 licensed.
- **3D UX-Net:** adapted from [MASILab/3DUX-Net](https://github.com/MASILab/3DUX-Net)
  and uses MONAI decoder blocks; Lee et al. (ICLR 2023) and MONAI. The upstream
  README states that 3DUX-Net is released under the MIT License, but the linked
  upstream `LICENSE` file is currently absent. MONAI components are Apache-2.0.
- **Swin UNETR:** MONAI `SwinUNETR`; cite Hatamizadeh et al. (BrainLes 2021/2022)
  and MONAI.

## Dataset preparation

The `data/` directory is generated from the original ATLAS v2.0 files distributed
for the ATLAS challenge at ISLES 2022. It is not part of this repository and can
be deleted and recreated from those source files. Obtain the dataset through the
[official ISLES 2022 ATLAS materials](https://github.com/npnl/isles_2022) and
comply with its access and license terms.

The resulting layout expected by the training and inference code is:

```
data/
  train/derivatives/ATLAS/sub-*/ses-1/anat/
    sub-*_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
    sub-*_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
  test/derivatives/ATLAS/sub-*/ses-1/anat/
    sub-*_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
```

The repository does not modify or redistribute the source images. No extra
preprocessing is applied beyond per-volume intensity normalization in the
dataloader. The paper's five-fold assignments are not created during data
preparation; they are versioned separately under `splits/`.

## Paper cross-validation splits

All paper experiments use the committed split file
`splits/atlas_5fold_lesion_quartile_excluding_known_issues.json`. It defines
five deterministic outer folds with lesion-volume-quartile stratification and
seed `9001`; each outer-training set also has a deterministic validation subset.
Use the same file and fold indices for every architecture and augmentation
condition. Do not regenerate it when reproducing the reported results. See
`splits/README.md` for the fold-to-run mapping and execution commands.

## Models and training

- **Models:** MONAI 3D U-Net, MedNeXt-S, 3D UX-Net, and MONAI Swin UNETR
- **Protocol:** the committed five-fold split is shared across all architectures
  and augmented/non-augmented training conditions
- **Sampling:** patch-based with lesion-biased centers
- **Paper augmented condition:** TorchIO `motion_consistent`, assigned to a fixed
  50% of training subjects (`--augment_frac 0.5`)
- **Training length:** 45 epochs, with early stopping enabled
- **Output:** sigmoid → threshold 0.5
- **Checkpoint:** `runs/.../checkpoints/best.pt` (best validation Dice)

### Hardware context
The following was the hardware used for preliminary testing and experimental runs.
- **Local:** MacBook Air M4, 24GB RAM, MPS (use conservative patch size, batch 1, `num_workers=0`)
- **Lab:** Linux, 120GB RAM, 2× RTX 3080 (CUDA; enable AMP; optional DDP)

Set the run directory explicitly (required):
```
export ATLAS_DATA_ROOT=/path/to/ATLAS_BIDS
export RUN_DIR=runs/base_cnn/run_kfold_01
export SPLITS_JSON=splits/atlas_5fold_lesion_quartile_excluding_known_issues.json
export CV_FOLD=0
```

### Example: macOS MPS smoke run (1 epoch)
```
python scripts/train_base_cnn.py \
  --run_dir $RUN_DIR \
  --splits_json "$SPLITS_JSON" \
  --cv_fold "$CV_FOLD" \
  --max_epochs 1 \
  --batch_size 1 \
  --patch_size 96 96 96 \
  --patches_per_volume 4 \
  --num_workers 0
```

### Example: Linux CUDA paper configuration (one fold, AMP)
```
# Batch size is per DDP process: 1 x 2 processes x 1 accumulation step = 2 global.
torchrun --nproc_per_node=2 scripts/train_base_cnn.py \
  --data_root "$ATLAS_DATA_ROOT" \
  --run_dir $RUN_DIR \
  --splits_json "$SPLITS_JSON" \
  --cv_fold "$CV_FOLD" \
  --max_epochs 45 \
  --batch_size 1 \
  --patch_size 128 128 128 \
  --patches_per_volume 8 \
  --lesion_prob 0.7 \
  --lr 0.001 \
  --weight_decay 1e-5 \
  --accum_steps 1 \
  --num_workers 4 \
  --pin_memory \
  --val_interval 1 \
  --seed 9001 \
  --amp
```

The paper's global batch size is 2. Keep `--batch_size 1` when using two DDP
processes; setting it to 2 would produce a global batch size of 4.

For the artifact-augmented paper condition, append:
```
--augment motion_consistent --augment_frac 0.5
```

The all-model/fold paper runner is `scripts/run_exp_kfold.sh`.

## Inference (BIDS-derivatives output)
Predictions are written to:
```
runs/<model>/<run_id>/preds_bids/derivatives/atlas2_prediction/sub-*/ses-1/anat/*_mask.nii.gz
```
A `dataset_description.json` is written at both the BIDS root and derivative root.

### Predict on public test (unlabeled)
```
python scripts/infer.py \
  --run_dir $RUN_DIR \
  --split test
```

### Predict on dev split (labeled)
```
python scripts/infer.py \
  --run_dir $RUN_DIR \
  --splits_json "$SPLITS_JSON" \
  --cv_fold "$CV_FOLD" \
  --split dev
```

## Local evaluation

The paper pipeline uses the repository's local evaluator and does not require
the retired ISLES Docker evaluator:

```
python scripts/eval_local.py \
  --run_dir $RUN_DIR \
  --splits_json "$SPLITS_JSON" \
  --cv_fold "$CV_FOLD" \
  --split dev \
  --out_json $RUN_DIR/eval_local/dev_metrics.json
```

## Smoke test
Runs quick checks: data counts, forward pass, one training step, and one test prediction written.
```
python scripts/smoke_test.py
```

Infrastructure tests use the Python standard library test runner:
```
python -m unittest discover -s tests -v
```

## Repository components

- Training, inference, evaluation, and orchestration entry points are under
  `scripts/`.
- Architecture builders and the minimal retained MedNeXt/UX-Net sources are
  under `scripts/models/`.
- Chart-generation utilities are under `scripts/visualization/`.
- Paper-specific lesion-stratification, MR-ART, and statistical analyses are
  organized under `scripts/supporting_experiments/`; each experiment keeps its
  executable code in a local `scripts/` directory.
- Versioned paper fold definitions are under `splits/`.
- Experiment artifacts are written under `$RUN_DIR/`.

## Notes
- Public test labels are hidden; evaluate locally on dev/heldout from `data/train`.

## Lab Runbook (Linux CUDA)
Install the pinned environment, validate inputs, and inspect the complete paper
matrix before launching it:

```
export ATLAS_DATA_ROOT=/path/to/ATLAS_BIDS
export SPLITS_JSON=splits/atlas_5fold_lesion_quartile_excluding_known_issues.json

git clone <YOUR_REPO_URL> atlas && cd atlas
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python scripts/validate_reproduction_inputs.py \
  --data-root "$ATLAS_DATA_ROOT" \
  --splits-json "$SPLITS_JSON"

# Review all 40 paper commands without executing them.
DRY_RUN=1 NPROC_PER_NODE=2 bash scripts/run_exp_kfold.sh

# Launch only after reviewing paths, GPUs, and output directories.
NPROC_PER_NODE=2 bash scripts/run_exp_kfold.sh
```

Notes:
- The paper configuration already uses the minimum per-GPU batch size of 1. If
  it runs out of GPU memory, reduce `--patch_size` to `96 96 96`; this reduces
  memory use but deviates from the paper's `128 128 128` configuration.
  Increasing `--accum_steps` does not reduce memory further when the per-GPU
  batch size is already 1.
- The paper fold definitions are versioned under `splits/`; do not regenerate them.
- Checkpoints and NIfTI predictions are intentionally excluded from Git.
- Machine-specific paths in retained historical summaries use `<REPO_ROOT>`.
