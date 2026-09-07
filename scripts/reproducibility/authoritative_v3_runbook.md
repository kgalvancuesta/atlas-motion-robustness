# Fresh authoritative IEEE BIBM run

Prepared run identity: **`atlas_corrected_authoritative_20260907_v3`**, exported by `scripts/reproducibility/pace_v3.env`. The immutable dataset-specific definition is created on PACE by the existing `prepare` stage, after preflight. No authoritative definition or full run was generated locally.

## Changes and preserved experiment

Both corrected regimes select `best.pt` from exactly the original clean fold-validation subjects (105 per fold). The same full-volume preprocessing, per-subject binary Dice (`pred > 0.5`, epsilon 1e-6), unweighted subject mean, strict `dev_dice > best_dice` improvement rule, and patience counter apply. No validation duplicates or validation artifact recipes are constructed. Historical non-corrected entrypoint behavior is preserved.

Training retains the four architectures, both regimes, five committed folds, 45-epoch maximum, seed 9001, original subject IDs, 50% augmented duplicates, historical corruption/no-op distribution, full-volume artifacts before normalization/cropping, deterministic patch centers/order, existing architecture-specific losses, optimizer/LR/weight decay/batch size/update counts, deterministic test challenge, inference threshold, and statistical definitions. Patience remains the existing 10; it was not changed to force every task to reach epoch 45.

The shared corrected numerical path:

1. Run normal AMP forward; collectively reduce logits-finite status before loss/backward.
2. If any rank reports nonfinite logits, every rank discards that graph and replays its exact already-loaded float32 tensor once with autocast disabled. There is no dataset access, resampling, or recipe generation in this branch.
3. Restore pre-forward Python, NumPy, CPU/CUDA RNG and model buffers before replay. Restore post-AMP RNG afterward: subsequent batches see one forward's RNG consumption; mutable buffers reflect one successful forward.
4. Use unchanged loss, gradient clipping (12.0), and AdamW update semantics on the FP32 replay. **GradScaler is bypassed completely for fallback updates**: scale and growth tracker stay unchanged. Successful ordinary AMP batches retain the original scale/backward/unscale/clip/step/update sequence. Existing gradient-overflow recovery remains capped at eight consecutive overflows; decisions are collective. The overflow-only check uses the pinned PyTorch 2.8 GradScaler's recorded `found_inf` state to prevent a rank-specific skip/update.
5. Nonfinite FP32 output (or a replay exception), loss, unrecovered gradients, or floating model parameters/buffers causes a collective failure with diagnostics. Floating state is checked at initialization/resume and after optimizer updates, before another update. Rank-0 validation exceptions are also propagated before peers continue. This handles numerical failure decisions; it is not a guarantee of recovery from a dead GPU/process or network failure. NCCL timeouts were not increased.
6. Every fallback appends a per-task `logs/amp_fp32_fallbacks.jsonl` event containing task/step identity, per-rank subject/role/patch slot, failing AMP ranks, input/AMP/FP32 statistics, scaler scale, FP32 success, and UTC timestamp. Fatal diagnostics are separate JSON files. `last.pt`, completed `logs/numerical_summary.json`, and `task.metadata.json` retain fallback counts. `logged_fallback_events` additionally counts attempts retained in the journal from interrupted/replayed epochs.

`checkpoint_selection` and `numerical_policy` participate in preparation identity, checkpoint/config compatibility, and task provenance. Production training rejects old v2 definitions; new-policy checkpoint loading rejects old metadata. The explicit read-only diagnostic entrypoint can load an old definition solely to replay/test the old checkpoint. RNG byte states loaded with CUDA `map_location` are moved back to CPU before generator restoration.

## Retained tests without scratch or GPUs

From the repository root, in the pinned environment:

```bash
python3 -m unittest discover -s tests -p 'test_training_numerics*.py' -v
python3 -m unittest discover -s tests -p 'test_clean_validation.py' -v
```

Tests include finite-path equivalence, same-tensor FP32 recovery and weight update, frozen scaler, Python/NumPy/Torch RNG, mutable buffers, collective FP32 failure/exception, rank-specific parameter/buffer failures including after update, diagnostic write failure, both DDP reducer modes, and the next DDP batch. Two-process CPU Gloo tests need permission to bind local sockets. On macOS, `GLOO_SOCKET_IFNAME=lo0` can select loopback; do not set `lo0` on Linux.

The replay integration test creates a tiny synthetic old checkpoint/cache, verifies a successful update and subsequent batch, proves that source bytes and modification times remain unchanged, and returns inconclusive when the AMP failure is absent. These tests remain useful after scratch deletion.

Full local regression command:

```bash
python3 -m unittest discover -s tests -v
```

## PACE synchronization after manual review/commits/push

First manually commit the two logical changes and push `determinism-reproducibility` from the local repository. The remote currently contains only the requested history correction until those dirty fixes are committed/pushed.

On PACE, start inside its existing repository clone with the pinned Python environment active. Stop the superseded training job before replaying its outputs. Fetch into a new local tracking branch because the old server branch may still contain the removed `fd00267`; this preserves that old local branch without a reset:

```bash
cd "$(git rev-parse --show-toplevel)"
git status --short
git fetch origin determinism-reproducibility
git switch --create codex/authoritative-v3 --track origin/determinism-reproducibility
git pull --ff-only
test -f scripts/reproducibility/pace_v3.env
python3 -m unittest discover -s tests -p 'test_training_numerics*.py' -v
```

A dirty PACE checkout must be preserved/reviewed before switching. If `codex/authoritative-v3` already exists, inspect and switch to it instead of recreating it. No history reset on PACE is necessary.

## Verify the actual old failure before deleting scratch

Run inside a two-GPU CUDA allocation using the old run's pinned software/device configuration, if available. This performs **902 local training batches per rank from the epoch-39 checkpoint**, then stops. It does not run validation, write checkpoints, or launch a full experiment. A different CUDA/GPU environment may not reproduce the old AMP failure; that is reported as inconclusive, not passed.

The default old-task path below follows the repository's configured project-output convention. If the old run used an output-root override, set `CORRECTED_EXPERIMENT_OUTPUT_ROOT` to that recorded root first. `--run-dir` must contain the original `config/train_config.json` and `checkpoints/last.pt`. Recorded data/definition/cache paths must still exist. Cache entries are opened read-only; missing/incompatible entries fail instead of rebuilding.

```bash
cd "$(git rev-parse --show-toplevel)"
OLD_TASK="${CORRECTED_EXPERIMENT_OUTPUT_ROOT:-${HOME}/r-cpradalier7-0/reproduction_outputs/corrected_experiments}/atlas_corrected_authoritative_20260903_v2/checkpoints/base_cnn/augmented/fold-0"
VERIFY_OUTPUT="${HOME}/r-cpradalier7-0/reproduction_outputs/numerics_verification/v2_epoch40_step901_$(date -u +%Y%m%dT%H%M%SZ).json"
test -f "${OLD_TASK}/config/train_config.json"
test -f "${OLD_TASK}/checkpoints/last.pt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/diagnose_training_numerics.py \
  --run-dir "${OLD_TASK}" --epoch 40 --step 901 \
  --verify-fix --output "${VERIFY_OUTPUT}"
python3 - "${VERIFY_OUTPUT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
assert report['status'] == 'passed', report['status']
assert report['source_checkpoint_epoch'] == 39
assert report['target_global_step'] == 98401
assert report['fp32_update_completed'] and report['scaler_state_unchanged']
assert report['subsequent_update_completed']
rank0 = next(r for r in report['fallback_event']['ranks'] if r['rank'] == 0)
assert rank0['batch']['subject_id'] == 'sub-r003s010', rank0['batch']
assert rank0['batch']['sample_role'] == 'clean_control', rank0['batch']
assert rank0['batch']['patch_slot'] == 2, rank0['batch']
print('PASS: observed failure recovered; source artifacts were opened read-only.')
PY
```

`--verify-fix` exits 0 only if the target AMP failure was reproduced and recovered with a successful FP32 update, frozen scaler, no earlier replay fallback changing the trajectory, and a subsequent optimizer update. Exit 2 is inconclusive; other nonzero exits are failures. Preserve the JSON and adjacent `.artifacts` directory outside scratch. The diagnostic-only AMP/FP32 activation probes remain available by omitting `--verify-fix`.

Only after this passes, with the old job and all readers stopped, manually remove the superseded v2 cache described in [storage_capacity.md](storage_capacity.md). There is no automatic deletion command. The estimated expanded full-run bound, including optional historical MR-ART exports and 25% headroom, is **5.86 TB and 386,000 inodes**, below 50% of both quotas; no independent cache optimization was needed.

## Preflight and eventual fresh launch

After old-cache cleanup, inside the intended two-A100 allocation and pinned environment:

```bash
cd "$(git rev-parse --show-toplevel)"
source scripts/reproducibility/pace_v3.env
export GPU_IDS=0,1
bash scripts/reproducibility/pace_preflight.sh
bash scripts/reproducibility/pace_full_run.sh
```

The full-run script deliberately repeats preflight, prepares the immutable v3 definition from server inputs, and runs all 40 tasks plus downstream analysis/MR-ART. It uses `${HOME}/scratch/atlas_corrected_cache/${EXPERIMENT_ID}` (resolved scratch symlink) and the existing project-output root. Never substitute the v2 ID or resume the superseded v2 task for authoritative results.

Local verification covers CPU behavior and real Gloo collectives, imports/compilation, shell syntax, lint, and whitespace. CUDA FP16/NCCL and the real epoch-40 replay require PACE and were not executed locally.
