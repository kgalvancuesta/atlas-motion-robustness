#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

DATA="${ATLAS_DATA_ROOT:-${PROJECT_DIR}/data}"
SPLITS_JSON="${SPLITS_JSON:-splits/atlas_5fold_lesion_quartile_excluding_known_issues.json}"
MODELS="${MODELS:-base_cnn uxnet mednext swin}"
AUGS="${AUGS:-0 1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
DRY_RUN="${DRY_RUN:-0}"
LOG_FILTER_MODE="${LOG_FILTER_MODE:-tracebacks}"
STATUS_RESET="${STATUS_RESET:-0}"
FAILURE_TAIL_LINES="${FAILURE_TAIL_LINES:-80}"

MASTER_LOG_DIR="runs/batch_logs"
BATCH_TS="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="${MASTER_LOG_DIR}/run_exp_kfold_${BATCH_TS}.log"
LOGFILE="/dev/null"
CURRENT_MODEL_LOG="/dev/null"
ANY_FAILURE=0
INITIALIZED_MODELS=""

mkdir -p "${MASTER_LOG_DIR}"

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

join_cmd() {
  printf '%q ' "$@"
}

determine_gpu_visibility() {
  if [ -n "${GPU_IDS:-}" ]; then
    TRAIN_GPU_VISIBILITY_SET=1
    TRAIN_GPU_VISIBILITY_VALUE="${GPU_IDS}"
  elif [ "${CUDA_VISIBLE_DEVICES+x}" = "x" ] && [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    TRAIN_GPU_VISIBILITY_SET=1
    TRAIN_GPU_VISIBILITY_VALUE="${CUDA_VISIBLE_DEVICES}"
  else
    TRAIN_GPU_VISIBILITY_SET=0
    TRAIN_GPU_VISIBILITY_VALUE=""
  fi

  FIRST_INFER_GPU="${TRAIN_GPU_VISIBILITY_VALUE%%,*}"
  if [ -z "${FIRST_INFER_GPU}" ]; then
    FIRST_INFER_GPU="0"
  fi
}

apply_train_gpu_visibility() {
  if [ "${TRAIN_GPU_VISIBILITY_SET}" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_VISIBILITY_VALUE}"
  else
    unset CUDA_VISIBLE_DEVICES
  fi
}

apply_infer_gpu_visibility() {
  export CUDA_VISIBLE_DEVICES="${FIRST_INFER_GPU}"
}

log_message() {
  local line
  line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  printf '%s\n' "${line}" | tee -a "${LOGFILE}" "${MASTER_LOG}" "${CURRENT_MODEL_LOG}"
}

filter_command_output() {
  if [ "${LOG_FILTER_MODE}" = "full" ]; then
    cat
    return 0
  fi

  awk '
    function emit(line) {
      print line
      fflush()
    }

    /Traceback \(most recent call last\):/ {
      in_trace = 1
      blank_count = 0
      emit($0)
      next
    }

    in_trace {
      if ($0 ~ /^$/) {
        if (blank_count == 0) {
          emit($0)
        }
        blank_count++
        next
      }
      if ($0 ~ /^([[:space:]]+|\[[^]]+\]:|[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Warning):|RuntimeError:|Exception:|KeyboardInterrupt|SystemExit|={8,}|-{8,}|Failures:|Root Cause|Last error:|[[:space:]]*(time|host|rank|exitcode|error_file|traceback)[[:space:]]*:)/) {
        blank_count = 0
        emit($0)
        next
      }
      in_trace = 0
    }

    /(ERROR:|Error:|FAILED|Checkpoint not found|Missing training log|FileNotFoundError|RuntimeError|Exception|DistBackendError|NCCL|nccl|nvmlInit|Can'\''t initialize NVML|CUDA out of memory|out of memory|ChildFailedError|Root Cause|Failures:|Last error:)/ {
      emit($0)
      next
    }
  '
}

refresh_status_views() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

csv_path = Path(sys.argv[1])
model_json_path = Path(sys.argv[2])
run_json_path = Path(sys.argv[3])
current_run_dir = sys.argv[4]
rows = []
if csv_path.exists():
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

required_stages = [
    "train",
    "infer_clean",
    "eval_clean",
    "infer_augmented",
    "eval_augmented",
    "charts",
    "dice_comparison",
    "plot_training",
]

runs: "OrderedDict[str, dict]" = OrderedDict()
for row in rows:
    run_dir = row["run_dir"]
    entry = runs.setdefault(
        run_dir,
        {
            "model_dir": row["model_dir"],
            "display_name": row["display_name"],
            "augmentation": row["augmentation"],
            "fold_index": int(row["fold_index"]),
            "fold_name": row["fold_name"],
            "run_dir": run_dir,
            "stages": OrderedDict(),
        },
    )
    entry["stages"][row["stage"]] = {
        "exit_code": int(row["exit_code"]),
        "status": row["status"],
        "log_path": row["log_path"],
        "timestamp_start": row["timestamp_start"],
        "timestamp_end": row["timestamp_end"],
    }

for entry in runs.values():
    stage_statuses = {stage: entry["stages"].get(stage, {}).get("status", "missing") for stage in required_stages}
    entry["required_stage_statuses"] = stage_statuses
    entry["all_required_stages_success"] = all(status == "success" for status in stage_statuses.values())
    entry["any_failure"] = any(stage.get("status") == "failed" for stage in entry["stages"].values())

model_payload = {
    "generated_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "required_stages": required_stages,
    "rows": rows,
    "runs": list(runs.values()),
}
model_json_path.write_text(json.dumps(model_payload, indent=2))

run_payload = runs.get(
    current_run_dir,
    {
        "run_dir": current_run_dir,
        "required_stages": required_stages,
        "stages": {},
        "required_stage_statuses": {stage: "missing" for stage in required_stages},
        "all_required_stages_success": False,
        "any_failure": False,
    },
)
if "required_stages" not in run_payload:
    run_payload = dict(run_payload)
    run_payload["required_stages"] = required_stages
run_json_path.write_text(json.dumps(run_payload, indent=2))
PY
}

append_status_row() {
  local stage="$1"
  local exit_code="$2"
  local status="$3"
  local timestamp_start="$4"
  local timestamp_end="$5"
  local model_root="runs/${CURRENT_MODEL_DIR}"
  local csv_path="${model_root}/kfold_status.csv"
  local json_path="${model_root}/kfold_status.json"
  local run_status_path="${CURRENT_RUN_DIR}/run_status.json"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${CURRENT_MODEL_DIR}" \
    "${CURRENT_DISPLAY_NAME}" \
    "${CURRENT_AUGMENTATION_LABEL}" \
    "${CURRENT_FOLD_INDEX}" \
    "${CURRENT_FOLD_NAME}" \
    "${CURRENT_RUN_DIR}" \
    "${stage}" \
    "${exit_code}" \
    "${status}" \
    "${LOGFILE}" \
    "${timestamp_start}" \
    "${timestamp_end}" \
    >> "${csv_path}"

  refresh_status_views "${csv_path}" "${json_path}" "${run_status_path}" "${CURRENT_RUN_DIR}"
}

initialize_model_tracking() {
  local model_dir="$1"
  local model_root="runs/${model_dir}"
  local csv_path="${model_root}/kfold_status.csv"
  local json_path="${model_root}/kfold_status.json"

  mkdir -p "${model_root}" "${model_root}/logs"

  case " ${INITIALIZED_MODELS} " in
    *" ${model_dir} "*) return 0 ;;
  esac

  if [ "${STATUS_RESET}" = "1" ] || [ ! -s "${csv_path}" ]; then
    printf '%s\n' "model_dir,display_name,augmentation,fold_index,fold_name,run_dir,stage,exit_code,status,log_path,timestamp_start,timestamp_end" > "${csv_path}"
    printf '%s\n' '{"required_stages":[],"rows":[],"runs":[]}' > "${json_path}"
  fi
  INITIALIZED_MODELS="${INITIALIZED_MODELS} ${model_dir}"
}

run_and_log() {
  local stage="$1"
  shift

  local timestamp_start
  local timestamp_end
  local exit_code=0
  local status_label="success"
  local cmd_str
  local raw_output
  local filtered_pipe
  local filter_pid

  timestamp_start="$(timestamp_utc)"
  cmd_str="$(join_cmd "$@")"
  log_message "=== START ${stage} ==="
  log_message "Command: ${cmd_str}"

  if [ "${DRY_RUN}" = "1" ]; then
    status_label="dry_run"
    log_message "DRY_RUN=1; command not executed."
  else
    raw_output="$(mktemp "${TMPDIR:-/tmp}/atlas_kfold_${stage}.XXXXXX")"
    filtered_pipe="$(mktemp "${TMPDIR:-/tmp}/atlas_kfold_${stage}_filter.XXXXXX")"
    rm -f "${filtered_pipe}"
    mkfifo "${filtered_pipe}"
    filter_command_output < "${filtered_pipe}" | tee -a "${LOGFILE}" "${MASTER_LOG}" "${CURRENT_MODEL_LOG}" >/dev/null &
    filter_pid=$!

    "$@" 2>&1 | tee "${raw_output}" "${filtered_pipe}"
    exit_code=${PIPESTATUS[0]}
    wait "${filter_pid}" || true
    rm -f "${filtered_pipe}"

    if [ "${exit_code}" -ne 0 ]; then
      status_label="failed"
      ANY_FAILURE=1
      log_message "--- raw failure tail (${FAILURE_TAIL_LINES} lines) for ${stage} ---"
      tail -n "${FAILURE_TAIL_LINES}" "${raw_output}" | tee -a "${LOGFILE}" "${MASTER_LOG}" "${CURRENT_MODEL_LOG}"
      log_message "--- end raw failure tail for ${stage} ---"
    fi
    rm -f "${raw_output}"
  fi

  timestamp_end="$(timestamp_utc)"
  log_message "=== END ${stage} (exit_code=${exit_code}, status=${status_label}) ==="
  append_status_row "${stage}" "${exit_code}" "${status_label}" "${timestamp_start}" "${timestamp_end}"
  return "${exit_code}"
}

mark_stage_skipped() {
  local stage="$1"
  local reason="$2"
  local ts

  ts="$(timestamp_utc)"
  log_message "=== SKIP ${stage} (status=skipped): ${reason} ==="
  append_status_row "${stage}" "0" "skipped" "${ts}" "${ts}"
}

resolve_model_meta() {
  case "$1" in
    base_cnn)
      CURRENT_MODEL_DIR="base_cnn"
      CURRENT_TRAIN_SCRIPT="train_base_cnn.py"
      CURRENT_INFER_MODEL="baseline"
      CURRENT_DISPLAY_NAME="Base_CNN"
      CURRENT_MODEL_PORT_OFFSET=0
      ;;
    uxnet)
      CURRENT_MODEL_DIR="uxnet"
      CURRENT_TRAIN_SCRIPT="train_uxnet.py"
      CURRENT_INFER_MODEL="uxnet"
      CURRENT_DISPLAY_NAME="UXNet"
      CURRENT_MODEL_PORT_OFFSET=100
      ;;
    mednext)
      CURRENT_MODEL_DIR="mednext"
      CURRENT_TRAIN_SCRIPT="train_mednext.py"
      CURRENT_INFER_MODEL="mednext"
      CURRENT_DISPLAY_NAME="MedNeXt"
      CURRENT_MODEL_PORT_OFFSET=200
      ;;
    swin)
      CURRENT_MODEL_DIR="swin"
      CURRENT_TRAIN_SCRIPT="train_swin.py"
      CURRENT_INFER_MODEL="swin"
      CURRENT_DISPLAY_NAME="Swin"
      CURRENT_MODEL_PORT_OFFSET=300
      ;;
    *)
      log_message "Unknown model_dir: $1"
      ANY_FAILURE=1
      return 1
      ;;
  esac
  return 0
}

run_experiment() {
  local model_dir="$1"
  local use_aug="$2"
  local fold_index="$3"
  local fold_number
  local master_port
  local plot_label
  local run_name
  local run_ts
  local aug_pred_run_dir

  resolve_model_meta "${model_dir}" || return 1
  initialize_model_tracking "${CURRENT_MODEL_DIR}"

  if ! [[ "${fold_index}" =~ ^[0-9]+$ ]]; then
    log_message "Invalid fold index: ${fold_index}"
    ANY_FAILURE=1
    return 1
  fi

  fold_number=$((fold_index + 1))
  CURRENT_FOLD_INDEX="${fold_index}"
  CURRENT_FOLD_NAME="$(printf 'kfold_%02d' "${fold_number}")"

  case "${use_aug}" in
    0)
      run_name="$(printf 'run_kfold_%02d' "${fold_number}")"
      CURRENT_AUGMENTATION_LABEL="none"
      ;;
    1)
      run_name="$(printf 'run_DA_kfold_%02d' "${fold_number}")"
      CURRENT_AUGMENTATION_LABEL="motion_consistent_0.5"
      ;;
    *)
      log_message "Invalid augmentation flag: ${use_aug}"
      ANY_FAILURE=1
      return 1
      ;;
  esac

  CURRENT_RUN_DIR="runs/${CURRENT_MODEL_DIR}/${run_name}"
  aug_pred_run_dir="${CURRENT_RUN_DIR}/augmented_test"
  mkdir -p "${CURRENT_RUN_DIR}/logs" "${CURRENT_RUN_DIR}/eval_local" "${aug_pred_run_dir}"
  run_ts="$(date '+%Y%m%d_%H%M%S')"
  LOGFILE="${CURRENT_RUN_DIR}/logs/run_${run_ts}.log"
  CURRENT_MODEL_LOG="runs/${CURRENT_MODEL_DIR}/logs/run_exp_kfold_${BATCH_TS}.log"
  plot_label="${CURRENT_DISPLAY_NAME}_${run_name}"
  master_port=$((MASTER_PORT_BASE + CURRENT_MODEL_PORT_OFFSET + use_aug * 10 + fold_index))

  log_message "Experiment: ${CURRENT_DISPLAY_NAME}"
  log_message "Run directory: ${CURRENT_RUN_DIR}"
  log_message "Fold: ${CURRENT_FOLD_NAME} (cv_fold=${CURRENT_FOLD_INDEX})"
  log_message "Training augmentation: ${CURRENT_AUGMENTATION_LABEL}"
  log_message "SPLITS_JSON=${SPLITS_JSON}"
  log_message "DDP: torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${master_port}"

  apply_train_gpu_visibility
  log_message "Training GPU visibility: ${CUDA_VISIBLE_DEVICES:-<all_visible>}"

  local -a train_cmd=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${master_port}"
    "scripts/${CURRENT_TRAIN_SCRIPT}"
    --data_root "${DATA}"
    --splits_json "${SPLITS_JSON}"
    --cv_fold "${CURRENT_FOLD_INDEX}"
    --run_dir "${CURRENT_RUN_DIR}"
    --max_epochs 45
    --batch_size 1
    --patch_size 128 128 128
    --patches_per_volume 8
    --lesion_prob 0.7
    --lr 0.001
    --weight_decay 1e-5
    --accum_steps 1
    --num_workers 4
    --pin_memory
    --val_interval 1
    --seed 9001
    --amp
    --numerics_debug
  )
  if [ "${use_aug}" = "1" ]; then
    train_cmd+=(--augment motion_consistent --augment_frac 0.5)
  fi

  local train_exit=0
  run_and_log "train" "${train_cmd[@]}"
  train_exit=$?
  if [ "${train_exit}" -ne 0 ]; then
    log_message "Training failed; skipping dependent inference/evaluation/plotting stages for this run."
    mark_stage_skipped "infer_clean" "training failed"
    mark_stage_skipped "eval_clean" "training failed"
    mark_stage_skipped "infer_augmented" "training failed"
    mark_stage_skipped "eval_augmented" "training failed"
    mark_stage_skipped "charts" "training failed"
    mark_stage_skipped "dice_comparison" "training failed"
    mark_stage_skipped "plot_training" "training failed"
    apply_train_gpu_visibility
    return 0
  fi

  apply_infer_gpu_visibility
  log_message "Post-training GPU visibility: ${CUDA_VISIBLE_DEVICES}"

  run_and_log "infer_clean" \
    python3 scripts/infer.py \
      --data_root "${DATA}" \
      --run_dir "${CURRENT_RUN_DIR}" \
      --splits_json "${SPLITS_JSON}" \
      --cv_fold "${CURRENT_FOLD_INDEX}" \
      --split test \
    || true

  run_and_log "eval_clean" \
    python3 scripts/eval_local.py \
      --data_root "${DATA}" \
      --run_dir "${CURRENT_RUN_DIR}" \
      --splits_json "${SPLITS_JSON}" \
      --cv_fold "${CURRENT_FOLD_INDEX}" \
      --split test \
      --out_json "${CURRENT_RUN_DIR}/eval_local/test_clean_metrics.json" \
    || true

  run_and_log "infer_augmented" \
    python3 scripts/infer.py \
      --data_root "${DATA}" \
      --run_dir "${aug_pred_run_dir}" \
      --checkpoint "${CURRENT_RUN_DIR}/checkpoints/best.pt" \
      --model "${CURRENT_INFER_MODEL}" \
      --patch_size 128 128 128 \
      --splits_json "${SPLITS_JSON}" \
      --cv_fold "${CURRENT_FOLD_INDEX}" \
      --split test \
      --augment motion_consistent \
    || true

  run_and_log "eval_augmented" \
    python3 scripts/eval_local.py \
      --data_root "${DATA}" \
      --preds_bids "${aug_pred_run_dir}/preds_bids" \
      --splits_json "${SPLITS_JSON}" \
      --cv_fold "${CURRENT_FOLD_INDEX}" \
      --split test \
      --out_json "${CURRENT_RUN_DIR}/eval_local/test_augmented_metrics.json" \
    || true

  run_and_log "charts" \
    python3 scripts/visualization/generate_charts.py --run-dir "${CURRENT_RUN_DIR}" \
    || true

  run_and_log "dice_comparison" \
    python3 scripts/visualization/plot_dice_comparison.py --run-dir "${CURRENT_RUN_DIR}" \
    || true

  run_and_log "plot_training" \
    python3 scripts/visualization/plot_training.py \
      --run_dirs "${CURRENT_RUN_DIR}" \
      --labels "${plot_label}" \
      --out "${CURRENT_RUN_DIR}/training_curves.png" \
    || true

  apply_train_gpu_visibility
}

determine_gpu_visibility

log_message "Batch script: $(basename "${BASH_SOURCE[0]}")"
log_message "SPLITS_JSON=${SPLITS_JSON}"
log_message "MODELS=${MODELS}"
log_message "AUGS=${AUGS}"
log_message "FOLDS=${FOLDS}"
log_message "NPROC_PER_NODE=${NPROC_PER_NODE}"
log_message "MASTER_PORT_BASE=${MASTER_PORT_BASE}"
log_message "DRY_RUN=${DRY_RUN}"
log_message "LOG_FILTER_MODE=${LOG_FILTER_MODE}"
log_message "STATUS_RESET=${STATUS_RESET}"
log_message "FAILURE_TAIL_LINES=${FAILURE_TAIL_LINES}"
log_message "Training GPU visibility default: ${TRAIN_GPU_VISIBILITY_VALUE:-<all_visible>}"
log_message "Inference/Eval GPU: ${FIRST_INFER_GPU}"

for model_dir in ${MODELS}; do
  for use_aug in ${AUGS}; do
    for fold_index in ${FOLDS}; do
      run_experiment "${model_dir}" "${use_aug}" "${fold_index}"
    done
  done
done

if [ "${ANY_FAILURE}" -ne 0 ]; then
  log_message "Batch complete with failures. Inspect ${MASTER_LOG} and runs/<model_dir>/kfold_status.{csv,json}."
else
  log_message "Batch complete. All requested stages finished successfully."
fi

exit "${ANY_FAILURE}"
