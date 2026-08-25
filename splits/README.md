# ATLAS CV Usage

Use `splits/atlas_5fold_lesion_quartile_excluding_known_issues.json` for all 5-fold cross-validation experiments.

Excluded subjects:
- `sub-r039s002`: invalid/ambiguous lesion mask values in ATLAS
- `sub-r009s003`: manually confirmed incorrect lesion label

Mapping note:
- `run_kfold_01` / `run_DA_kfold_01` maps to `--cv_fold 0`
- `run_kfold_05` / `run_DA_kfold_05` maps to `--cv_fold 4`

Smoke test one fold:

```bash
DRY_RUN=1 MODELS="swin" AUGS="0" FOLDS="0" scripts/run_exp_kfold.sh
```

Run the full 40-model batch:

```bash
MODELS="base_cnn uxnet mednext swin" AUGS="0 1" FOLDS="0 1 2 3 4" scripts/run_exp_kfold.sh
```

Aggregate one model without DA:

```bash
python3 scripts/aggregate_cv_metrics.py \
  --model_dir runs/swin \
  --run_prefix run_kfold \
  --splits_json splits/atlas_5fold_lesion_quartile_excluding_known_issues.json \
  --out_dir runs/swin/run_kfold_summary
```

Aggregate one model with DA:

```bash
python3 scripts/aggregate_cv_metrics.py \
  --model_dir runs/swin \
  --run_prefix run_DA_kfold \
  --splits_json splits/atlas_5fold_lesion_quartile_excluding_known_issues.json \
  --out_dir runs/swin/run_DA_kfold_summary
```

Reminder:
- Use the same CV split JSON for every architecture so comparisons stay paired.
- Non-DA runs are `run_kfold_01` through `run_kfold_05`.
- DA runs are `run_DA_kfold_01` through `run_DA_kfold_05`.
