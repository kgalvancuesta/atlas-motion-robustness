#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_DIR}"

ORIGINAL_ARGS=("$@")

DATA_ROOT="data/mr-art"
OUTPUT_ROOT="runs/mrart_negative_control"
CHECKPOINT_ROOT="runs"
ARCH="mednext"
TRAINING="augmented"
FOLDS="0,1,2,3,4"
GPUS="0"
RUN_ID=""
SMOKE_TEST=0
MAX_SUBJECTS=""
SAVE_ENSEMBLE_PROB=1
SAVE_FOLD_OUTPUTS=0
PROB_FORMAT="nifti"
SW_BATCH_SIZE=1
CHECKPOINT_CHECKSUM=0
ALLOW_MISSING_CHECKPOINTS=0
ALLOW_SCAN_ERRORS=0
COMPUTE_LARGEST_COMPONENT=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/supporting_experiments/mrart/scripts/run_mrart_negative_control.sh [options]

Required workflow examples:
  bash scripts/supporting_experiments/mrart/scripts/run_mrart_negative_control.sh --data-root data/mr-art --output-root runs/mrart_negative_control --arch mednext --training augmented --folds 0 --gpus 0 --smoke-test --max-subjects 1 --save-ensemble-prob --save-fold-outputs
  bash scripts/supporting_experiments/mrart/scripts/run_mrart_negative_control.sh --data-root data/mr-art --output-root runs/mrart_negative_control --arch mednext --training both --folds 0,1,2,3,4 --gpus 0,1 --save-ensemble-prob

Options:
  --data-root PATH
  --output-root PATH
  --checkpoint-root PATH
  --arch mednext|swin_unetr|both
  --training standard|augmented|both
  --folds CSV
  --gpus CSV
  --run-id ID
  --smoke-test
  --max-subjects N
  --save-ensemble-prob | --no-save-ensemble-prob
  --save-fold-outputs
  --prob-format nifti|npz
  --sw-batch-size N
  --checkpoint-checksum
  --allow-missing-checkpoints
  --allow-scan-errors
  --compute-largest-component
  -h | --help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --training)
      TRAINING="$2"
      shift 2
      ;;
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --max-subjects)
      MAX_SUBJECTS="$2"
      shift 2
      ;;
    --save-ensemble-prob)
      SAVE_ENSEMBLE_PROB=1
      shift
      ;;
    --no-save-ensemble-prob)
      SAVE_ENSEMBLE_PROB=0
      shift
      ;;
    --save-fold-outputs)
      SAVE_FOLD_OUTPUTS=1
      shift
      ;;
    --prob-format)
      PROB_FORMAT="$2"
      shift 2
      ;;
    --sw-batch-size)
      SW_BATCH_SIZE="$2"
      shift 2
      ;;
    --checkpoint-checksum)
      CHECKPOINT_CHECKSUM=1
      shift
      ;;
    --allow-missing-checkpoints)
      ALLOW_MISSING_CHECKPOINTS=1
      shift
      ;;
    --allow-scan-errors)
      ALLOW_SCAN_ERRORS=1
      shift
      ;;
    --compute-largest-component)
      COMPUTE_LARGEST_COMPONENT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

sanitize_id_part() {
  printf '%s' "$1" | tr ',' '-' | tr '/' '-' | tr '[:upper:]' '[:lower:]'
}

timestamp="$(date '+%Y%m%d_%H%M%S')"
run_type="full_runs"
run_suffix="run"
if [ "${SMOKE_TEST}" -eq 1 ]; then
  run_type="smoke_tests"
  run_suffix="smoke"
fi

if [ -z "${RUN_ID}" ]; then
  arch_part="$(sanitize_id_part "${ARCH}")"
  training_part="$(sanitize_id_part "${TRAINING}")"
  if [ "${SMOKE_TEST}" -eq 1 ]; then
    RUN_ID="${timestamp}_${arch_part}_${training_part}_${run_suffix}"
  else
    RUN_ID="${timestamp}_${arch_part}_${training_part}"
  fi
fi

RUN_DIR="${OUTPUT_ROOT}/${run_type}/${RUN_ID}"
LAUNCHER_COMMAND="$(printf '%q ' "$0" "${ORIGINAL_ARGS[@]}")"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if [ "${#GPU_ARRAY[@]}" -lt 1 ] || [ -z "${GPU_ARRAY[0]}" ]; then
  echo "--gpus must contain at least one GPU id" >&2
  exit 2
fi

declare -a common_args=(
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --run-id "${RUN_ID}"
  --run-dir "${RUN_DIR}"
  --arch "${ARCH}"
  --training "${TRAINING}"
  --folds "${FOLDS}"
  --gpus "${GPUS}"
  --prob-format "${PROB_FORMAT}"
  --sw-batch-size "${SW_BATCH_SIZE}"
  --launcher-command "${LAUNCHER_COMMAND}"
)

if [ "${SMOKE_TEST}" -eq 1 ]; then
  common_args+=(--smoke-test)
fi
if [ -n "${MAX_SUBJECTS}" ]; then
  common_args+=(--max-subjects "${MAX_SUBJECTS}")
fi
if [ "${SAVE_ENSEMBLE_PROB}" -eq 1 ]; then
  common_args+=(--save-ensemble-prob)
else
  common_args+=(--no-save-ensemble-prob)
fi
if [ "${SAVE_FOLD_OUTPUTS}" -eq 1 ]; then
  common_args+=(--save-fold-outputs)
fi
if [ "${CHECKPOINT_CHECKSUM}" -eq 1 ]; then
  common_args+=(--checkpoint-checksum)
fi
if [ "${ALLOW_MISSING_CHECKPOINTS}" -eq 1 ]; then
  common_args+=(--allow-missing-checkpoints)
fi
if [ "${COMPUTE_LARGEST_COMPONENT}" -eq 1 ]; then
  common_args+=(--compute-largest-component)
fi

echo "Preparing MR-ART run: ${RUN_DIR}"
python3 scripts/supporting_experiments/mrart/scripts/infer_mrart_negative_control.py --mode prepare "${common_args[@]}"

mkdir -p "${RUN_DIR}/logs"

worker_fail=0
worker_count="${#GPU_ARRAY[@]}"
declare -a pids=()
for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  log_path="${RUN_DIR}/logs/worker_${idx}_gpu_${gpu}.log"
  echo "Starting worker ${idx}/${worker_count} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python3 scripts/supporting_experiments/mrart/scripts/infer_mrart_negative_control.py \
      --mode worker \
      --run-dir "${RUN_DIR}" \
      --worker-index "${idx}" \
      --num-workers "${worker_count}" \
      --gpu-id "${gpu}" \
      --sw-batch-size "${SW_BATCH_SIZE}" \
      --progress-position "${idx}"
  ) 2>&1 | tee "${log_path}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    worker_fail=1
  fi
done

echo "Finalizing MR-ART run: ${RUN_DIR}"
python3 scripts/supporting_experiments/mrart/scripts/infer_mrart_negative_control.py --mode finalize --run-dir "${RUN_DIR}"

scan_failures="$(python3 - "${RUN_DIR}/merged/summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
print(summary.get("total_failed", 0))
PY
)"

if [ "${worker_fail}" -ne 0 ]; then
  echo "One or more worker processes failed. Inspect ${RUN_DIR}/logs and ${RUN_DIR}/errors." >&2
  exit 1
fi

if [ "${ALLOW_SCAN_ERRORS}" -ne 1 ] && [ "${scan_failures}" -ne 0 ]; then
  echo "MR-ART inference finished with ${scan_failures} failed scan/model-condition jobs. Inspect ${RUN_DIR}/merged/errors.csv." >&2
  exit 1
fi

echo "MR-ART run complete: ${RUN_DIR}"
echo "Completion CSV: ${RUN_DIR}/merged/completion.csv"
echo "Error CSV: ${RUN_DIR}/merged/errors.csv"
echo "Summary JSON: ${RUN_DIR}/merged/summary.json"
