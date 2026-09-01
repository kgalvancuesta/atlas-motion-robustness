#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${EXPERIMENT_ID:-}" ]]; then
  echo "ERROR: set a new, explicit EXPERIMENT_ID for this authoritative run." >&2
  exit 2
fi
if [[ "${EXPERIMENT_ID}" == test-* ]]; then
  echo "ERROR: authoritative EXPERIMENT_ID must not start with test-." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Apply the final device selection before preflight so it validates the same two
# A100 GPUs used by training. GPU memory capacity is diagnostic, not a gate.
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"

# The same quick preflight is mandatory immediately before preparation/training.
bash scripts/reproducibility/pace_preflight.sh

SCRATCH_RESOLVED="$(readlink -f "${HOME}/scratch")"
if [[ -z "${SCRATCH_RESOLVED}" || ! -d "${SCRATCH_RESOLVED}" || ! -w "${SCRATCH_RESOLVED}" ]]; then
  echo "ERROR: verified PACE scratch is unavailable." >&2
  exit 2
fi
pace-quota

DATA_ROOT="${ATLAS_DATA_ROOT:-${REPO_ROOT}/data}"
PROJECT_OUTPUT_ROOT="${CORRECTED_EXPERIMENT_OUTPUT_ROOT:-${HOME}/r-cpradalier7-0/reproduction_outputs/corrected_experiments}"
CACHE_ROOT="${SCRATCH_RESOLVED}/atlas_corrected_cache/${EXPERIMENT_ID}"
mkdir -p "${PROJECT_OUTPUT_ROOT}" "${CACHE_ROOT}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 scripts/run_reprod_experiment.py prepare \
  --experiment-id "${EXPERIMENT_ID}" \
  --dataset-authority authoritative \
  --output-root "${PROJECT_OUTPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --mrart-root "${DATA_ROOT}/mr-art" \
  --split-path splits/atlas_5fold_lesion_quartile_excluding_known_issues.json \
  --normalization-mode artifact_then_normalize \
  --epochs 45 \
  --replicate-count 1 \
  --global-seed 9001 \
  --model base_cnn --model uxnet --model mednext --model swin \
  --training-regime standard --training-regime augmented \
  --fold 0 --fold 1 --fold 2 --fold 3 --fold 4

python3 scripts/run_reprod_experiment.py all \
  --experiment-id "${EXPERIMENT_ID}" \
  --output-root "${PROJECT_OUTPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --split-path splits/atlas_5fold_lesion_quartile_excluding_known_issues.json \
  --normalization-mode artifact_then_normalize \
  --epochs 45 \
  --replicate-count 1 \
  --memory-mode high \
  --cache-root "${CACHE_ROOT}" \
  --nproc-per-node 2 \
  --model base_cnn --model uxnet --model mednext --model swin \
  --training-regime standard --training-regime augmented \
  --fold 0 --fold 1 --fold 2 --fold 3 --fold 4
