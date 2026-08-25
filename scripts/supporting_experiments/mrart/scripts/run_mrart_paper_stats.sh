#!/usr/bin/env bash
set -uo pipefail

DATE_TAG="20260703"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 2

DATA_ROOT="${DATA_ROOT:-data/mr-art}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs}"
GPUS="${GPUS:-0,1}"
FOLDS="${FOLDS:-0,1,2,3,4}"
OUTPUT_BASE="${OUTPUT_BASE:-scripts/supporting_experiments/mrart/server_reports}"
STRICT_EXIT="${STRICT_EXIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SIMULATE_STAGE_FAILURE="${SIMULATE_STAGE_FAILURE:-}"

path_is_ignored() {
  git check-ignore -v "$1" >/tmp/mrart_check_ignore.$$ 2>/tmp/mrart_check_ignore_err.$$
}

choose_output_dir() {
  local bases=("${OUTPUT_BASE}" "scripts/supporting_experiments/mrart/server_reports_tracked")
  local base suffix candidate idx
  for base in "${bases[@]}"; do
    idx=1
    while [ "${idx}" -le 99 ]; do
      if [ "${idx}" -eq 1 ]; then
        suffix=""
      else
        suffix="_v${idx}"
      fi
      candidate="${base}/${DATE_TAG}_mrart_paper_stats${suffix}"
      if [ -e "${candidate}" ]; then
        idx=$((idx + 1))
        continue
      fi
      if path_is_ignored "${candidate}"; then
        idx=$((idx + 1))
        continue
      fi
      printf '%s\n' "${candidate}"
      rm -f /tmp/mrart_check_ignore.$$ /tmp/mrart_check_ignore_err.$$ || true
      return 0
    done
  done
  rm -f /tmp/mrart_check_ignore.$$ /tmp/mrart_check_ignore_err.$$ || true
  return 1
}

OUTPUT_DIR="$(choose_output_dir)"
if [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: could not choose a non-ignored, non-existing output directory" >&2
  exit 2
fi

LOG_DIR="${OUTPUT_DIR}/logs"
RUN_LOG="${LOG_DIR}/run.log"
STAGE_STATUS="${LOG_DIR}/stage_status.tsv"
COMMAND_LOG="${LOG_DIR}/commands_attempted.txt"
RUN_INFO="${OUTPUT_DIR}/mrart_paper_stats_run_info_${DATE_TAG}.json"
RUN_SUMMARY="${OUTPUT_DIR}/mrart_paper_stats_run_summary_${DATE_TAG}.md"

if ! mkdir -p "${LOG_DIR}"; then
  echo "ERROR: failed to create log directory: ${LOG_DIR}" >&2
  exit 2
fi
if ! : > "${RUN_LOG}" || ! : > "${COMMAND_LOG}" || ! printf 'stage\tstatus\ttimestamp\texit_code\tnote\n' > "${STAGE_STATUS}"; then
  echo "ERROR: failed to initialize run logs under ${LOG_DIR}" >&2
  exit 2
fi

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "${RUN_LOG}"
}

append_status() {
  local stage="$1"
  local status="$2"
  local rc="${3:-}"
  local note="${4:-}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${stage}" "${status}" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${rc}" "${note}" >> "${STAGE_STATUS}"
}

quote_command() {
  printf '%q ' "$@"
}

write_run_info() {
  python3 - "$RUN_INFO" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])

def cmd_text(cmd):
    try:
        return subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

payload = {
    "date": "2026-07-03",
    "command": " ".join(sys.argv),
    "project_root": os.getcwd(),
    "output_directory": os.environ.get("OUTPUT_DIR", ""),
    "data_root": os.environ.get("DATA_ROOT", ""),
    "checkpoint_root": os.environ.get("CHECKPOINT_ROOT", ""),
    "gpus": os.environ.get("GPUS", ""),
    "folds": os.environ.get("FOLDS", ""),
    "strict_exit": os.environ.get("STRICT_EXIT", "0"),
    "dry_run": os.environ.get("DRY_RUN", "0"),
    "fold_primary_ground_truth": "successful rows from mrart_fold_lcc_stats_long.csv",
    "completion_csv_role": "provenance_and_fallback_only_for_missing_or_failed_rerun_rows",
    "git_commit": cmd_text(["git", "rev-parse", "HEAD"]),
    "git_status_short": cmd_text(["git", "status", "--short", "--branch"]),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

export OUTPUT_DIR DATA_ROOT CHECKPOINT_ROOT GPUS FOLDS STRICT_EXIT DRY_RUN

if ! write_run_info; then
  log "ERROR: failed to write run info"
  append_status "stage_0_safety_checks" "fail" "2" "failed_to_write_run_info"
  exit 2
fi
append_status "stage_0_safety_checks" "pass" "0" "output_dir=${OUTPUT_DIR}"
log "MR-ART paper stats output_dir=${OUTPUT_DIR}"
log "DRY_RUN=${DRY_RUN} STRICT_EXIT=${STRICT_EXIT}"

run_stage() {
  local stage_name="$1"
  local error_log="$2"
  shift 2
  local stdout_tmp="${LOG_DIR}/${stage_name}_stdout.tmp"
  local command_text
  command_text="$(quote_command "$@")"

  log "START ${stage_name}"
  printf '%s\t%s\n' "${stage_name}" "${command_text}" >> "${COMMAND_LOG}"
  append_status "${stage_name}" "started" "" ""

  if [ "${DRY_RUN}" = "1" ]; then
    if [ "${SIMULATE_STAGE_FAILURE}" = "${stage_name}" ]; then
      echo "simulated dry-run failure" > "${error_log}"
      log "FAIL ${stage_name} rc=97; continuing"
      append_status "${stage_name}" "fail" "97" "simulated_dry_run_failure; see ${error_log}"
      return 0
    fi
    log "DRY_RUN ${stage_name}: ${command_text}"
    append_status "${stage_name}" "dry_run" "0" "command not executed"
    return 0
  fi

  "$@" > "${stdout_tmp}" 2> "${error_log}"
  local rc=$?

  if [ "${rc}" -eq 0 ]; then
    log "PASS ${stage_name}"
    append_status "${stage_name}" "pass" "0" ""
  else
    log "FAIL ${stage_name} rc=${rc}; continuing"
    append_status "${stage_name}" "fail" "${rc}" "see ${error_log}"
    if [ -s "${error_log}" ]; then
      {
        echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] ${stage_name} error tail:"
        tail -40 "${error_log}" || true
      } >> "${RUN_LOG}"
    fi
  fi

  rm -f "${stdout_tmp}" || true
  return 0
}

run_stage_stream() {
  local stage_name="$1"
  local error_log="$2"
  shift 2
  local command_text
  command_text="$(quote_command "$@")"

  log "START ${stage_name}"
  printf '%s\t%s\n' "${stage_name}" "${command_text}" >> "${COMMAND_LOG}"
  append_status "${stage_name}" "started" "" ""

  if [ "${DRY_RUN}" = "1" ]; then
    if [ "${SIMULATE_STAGE_FAILURE}" = "${stage_name}" ]; then
      echo "simulated dry-run failure" > "${error_log}"
      log "FAIL ${stage_name} rc=97; continuing"
      append_status "${stage_name}" "fail" "97" "simulated_dry_run_failure; see ${error_log}"
      return 0
    fi
    log "DRY_RUN ${stage_name}: ${command_text}"
    append_status "${stage_name}" "dry_run" "0" "command not executed"
    return 0
  fi

  "$@" 2> "${error_log}"
  local rc=$?

  if [ "${rc}" -eq 0 ]; then
    : > "${error_log}"
    log "PASS ${stage_name}"
    append_status "${stage_name}" "pass" "0" ""
  else
    log "FAIL ${stage_name} rc=${rc}; continuing"
    append_status "${stage_name}" "fail" "${rc}" "see ${error_log}"
    if [ -s "${error_log}" ]; then
      {
        echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] ${stage_name} error tail:"
        tail -40 "${error_log}" || true
      } >> "${RUN_LOG}"
    fi
  fi

  return 0
}

FOLD_DIR="${OUTPUT_DIR}/fold_lcc_stats"
ENSEMBLE_DIR="${OUTPUT_DIR}/ensemble_lcc"
VOXEL_DIR="${OUTPUT_DIR}/voxel_volume"
PAPER_DIR="${OUTPUT_DIR}/paper_csvs"

FOLD_LONG="${FOLD_DIR}/mrart_fold_lcc_stats_long.csv"
ENSEMBLE_ROWS="${ENSEMBLE_DIR}/mrart_ensemble_lcc_from_existing_masks.csv"
ENSEMBLE_SUMMARY="${ENSEMBLE_DIR}/mrart_ensemble_lcc_from_existing_masks_summary.csv"
VOXEL_ROWS="${VOXEL_DIR}/mrart_voxel_volume_verification_rows.csv"
VOXEL_SUMMARY="${VOXEL_DIR}/mrart_voxel_volume_verification_summary.csv"

run_stage_stream "stage_1_fold_lcc_stats" "${LOG_DIR}/stage_1_fold_lcc_stats_errors.log" \
  python3 scripts/supporting_experiments/mrart/scripts/run_mrart_fold_lcc_stats_no_outputs.py \
    --data-root "${DATA_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --arch both \
    --training both \
    --folds "${FOLDS}" \
    --gpus "${GPUS}" \
    --output-dir "${FOLD_DIR}"

validate_fold_rerun() {
  python3 - "${FOLD_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

fold_dir = Path(sys.argv[1])
info_path = fold_dir / "mrart_fold_lcc_run_info.json"
long_path = fold_dir / "mrart_fold_lcc_stats_long.csv"
if not info_path.exists():
    raise SystemExit(f"missing run info: {info_path}")
if not long_path.exists():
    raise SystemExit(f"missing fold long CSV: {long_path}")
info = json.loads(info_path.read_text(encoding="utf-8"))
expected = int(info.get("expected_rows", -1))
actual = int(info.get("actual_rows", -1))
success = int(info.get("n_success", -1))
errors = int(info.get("n_error", -1))
worker_errors = info.get("worker_errors") or []
with long_path.open("r", encoding="utf-8", newline="") as f:
    row_count = max(0, sum(1 for _ in f) - 1)
problems = []
if expected != 8400:
    problems.append(f"expected_rows should be 8400, found {expected}")
if actual != expected:
    problems.append(f"actual_rows {actual} != expected_rows {expected}")
if row_count != expected:
    problems.append(f"long CSV rows {row_count} != expected_rows {expected}")
if success != expected:
    problems.append(f"n_success {success} != expected_rows {expected}")
if errors != 0:
    problems.append(f"n_error should be 0, found {errors}")
if worker_errors:
    problems.append(f"worker_errors present: {len(worker_errors)}")
if problems:
    raise SystemExit("; ".join(problems))

with long_path.open("r", encoding="utf-8", newline="") as f:
    statuses = {}
    for row in csv.DictReader(f):
        status = row.get("status", "")
        statuses[status] = statuses.get(status, 0) + 1
if statuses != {"success": expected}:
    raise SystemExit(f"unexpected row status counts: {statuses}")
print(f"valid fold rerun: expected_rows={expected}, n_success={success}, n_error={errors}")
PY
}

if [ "${DRY_RUN}" != "1" ]; then
  log "START stage_1_fold_lcc_stats_validation"
  append_status "stage_1_fold_lcc_stats_validation" "started" "" ""
  if validate_fold_rerun > "${LOG_DIR}/stage_1_fold_lcc_stats_validation.log" 2> "${LOG_DIR}/stage_1_fold_lcc_stats_validation_errors.log"; then
    log "PASS stage_1_fold_lcc_stats_validation"
    append_status "stage_1_fold_lcc_stats_validation" "pass" "0" "all 8400 fold rows successful"
  else
    log "FAIL stage_1_fold_lcc_stats_validation; aborting before ensemble/build stages"
    append_status "stage_1_fold_lcc_stats_validation" "fail" "1" "see ${LOG_DIR}/stage_1_fold_lcc_stats_validation_errors.log"
    if [ -s "${LOG_DIR}/stage_1_fold_lcc_stats_validation_errors.log" ]; then
      tail -40 "${LOG_DIR}/stage_1_fold_lcc_stats_validation_errors.log" >> "${RUN_LOG}" || true
    fi
    exit 1
  fi
fi

run_stage_stream "stage_2_ensemble_lcc" "${LOG_DIR}/stage_2_ensemble_lcc_errors.log" \
  python3 scripts/supporting_experiments/mrart/scripts/generate_mrart_lcc_from_predictions.py \
    --scope ensemble \
    --output-csv "${ENSEMBLE_ROWS}" \
    --summary-csv "${ENSEMBLE_SUMMARY}"

validate_ensemble_lcc() {
  python3 - "${ENSEMBLE_ROWS}" "${ENSEMBLE_SUMMARY}" <<'PY'
import csv
import sys
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
if not rows_path.exists():
    raise SystemExit(f"missing ensemble rows CSV: {rows_path}")
if not summary_path.exists():
    raise SystemExit(f"missing ensemble summary CSV: {summary_path}")
with rows_path.open("r", encoding="utf-8", newline="") as f:
    row_count = max(0, sum(1 for _ in f) - 1)
with summary_path.open("r", encoding="utf-8", newline="") as f:
    summary_rows = list(csv.DictReader(f))
if len(summary_rows) != 1:
    raise SystemExit(f"expected one ensemble summary row, found {len(summary_rows)}")
summary = summary_rows[0]
n_rows = int(float(summary.get("n_rows", -1)))
n_success = int(float(summary.get("n_success", -1)))
n_missing = int(float(summary.get("n_missing_prediction", -1)))
n_error = int(float(summary.get("n_error", -1)))
problems = []
if row_count != 1680:
    problems.append(f"ensemble row CSV should have 1680 rows, found {row_count}")
if n_rows != 1680:
    problems.append(f"summary n_rows should be 1680, found {n_rows}")
if n_success != 1680:
    problems.append(f"summary n_success should be 1680, found {n_success}")
if n_missing != 0:
    problems.append(f"summary n_missing_prediction should be 0, found {n_missing}")
if n_error != 0:
    problems.append(f"summary n_error should be 0, found {n_error}")
if problems:
    raise SystemExit("; ".join(problems))
print(f"valid ensemble LCC: rows={n_rows}, n_success={n_success}, n_error={n_error}")
PY
}

if [ "${DRY_RUN}" != "1" ]; then
  log "START stage_2_ensemble_lcc_validation"
  append_status "stage_2_ensemble_lcc_validation" "started" "" ""
  if validate_ensemble_lcc > "${LOG_DIR}/stage_2_ensemble_lcc_validation.log" 2> "${LOG_DIR}/stage_2_ensemble_lcc_validation_errors.log"; then
    log "PASS stage_2_ensemble_lcc_validation"
    append_status "stage_2_ensemble_lcc_validation" "pass" "0" "all 1680 ensemble rows successful"
  else
    log "FAIL stage_2_ensemble_lcc_validation; aborting before voxel/build stages"
    append_status "stage_2_ensemble_lcc_validation" "fail" "1" "see ${LOG_DIR}/stage_2_ensemble_lcc_validation_errors.log"
    if [ -s "${LOG_DIR}/stage_2_ensemble_lcc_validation_errors.log" ]; then
      tail -40 "${LOG_DIR}/stage_2_ensemble_lcc_validation_errors.log" >> "${RUN_LOG}" || true
    fi
    exit 1
  fi
fi

run_stage "stage_3_voxel_volume_verification" "${LOG_DIR}/stage_3_voxel_volume_verification_errors.log" \
  python3 scripts/supporting_experiments/mrart/scripts/verify_mrart_voxel_volume_headers.py \
    --output-csv "${VOXEL_ROWS}" \
    --summary-csv "${VOXEL_SUMMARY}"

builder_args=(
  python3 scripts/supporting_experiments/mrart/scripts/build_mrart_hallucination_csvs.py
  --output-dir "${PAPER_DIR}"
  --fold-component-csv "${FOLD_LONG}"
)
log "stage_4_build_paper_csvs: requiring fold rerun CSV as fold-primary ground truth: ${FOLD_LONG}"
if [ -s "${ENSEMBLE_ROWS}" ]; then
  builder_args+=(--ensemble-lcc-csv "${ENSEMBLE_ROWS}")
else
  log "WARN stage_4_build_paper_csvs: ensemble LCC CSV missing; builder will run without it"
fi
if [ -s "${VOXEL_ROWS}" ]; then
  builder_args+=(--voxel-verification-csv "${VOXEL_ROWS}")
else
  log "WARN stage_4_build_paper_csvs: voxel verification CSV missing; builder will run without it"
fi
run_stage "stage_4_build_paper_csvs" "${LOG_DIR}/stage_4_build_paper_csvs_errors.log" "${builder_args[@]}"

write_final_summary() {
  python3 - "$OUTPUT_DIR" "$RUN_SUMMARY" "$STAGE_STATUS" "$COMMAND_LOG" <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
stage_status_path = Path(sys.argv[3])
command_log_path = Path(sys.argv[4])

expected = {
    "fold_lcc_long": (output_dir / "fold_lcc_stats" / "mrart_fold_lcc_stats_long.csv", 8400),
    "paper_long_per_fold": (output_dir / "paper_csvs" / "mrart_hallucination_long_per_fold.csv", 8400),
    "paper_per_scan_fold_summary": (output_dir / "paper_csvs" / "mrart_hallucination_per_scan_fold_summary.csv", 1680),
    "paper_grouped_fold_primary": (output_dir / "paper_csvs" / "mrart_hallucination_grouped_summary_fold_primary.csv", 12),
    "ensemble_secondary_summary": (output_dir / "paper_csvs" / "mrart_hallucination_ensemble_secondary_summary.csv", 12),
    "paper_fold_rerun_consistency_diagnostics": (output_dir / "paper_csvs" / "mrart_hallucination_fold_rerun_consistency_diagnostics.csv", None),
    "ensemble_lcc_rows": (output_dir / "ensemble_lcc" / "mrart_ensemble_lcc_from_existing_masks.csv", None),
    "ensemble_lcc_summary": (output_dir / "ensemble_lcc" / "mrart_ensemble_lcc_from_existing_masks_summary.csv", None),
    "voxel_verification_rows": (output_dir / "voxel_volume" / "mrart_voxel_volume_verification_rows.csv", None),
    "voxel_verification_summary": (output_dir / "voxel_volume" / "mrart_voxel_volume_verification_summary.csv", None),
    "fold_lcc_errors": (output_dir / "fold_lcc_stats" / "mrart_fold_lcc_errors.csv", None),
}

def row_count(path):
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)

def read_stage_rows():
    if not stage_status_path.exists():
        return []
    with stage_status_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

stage_rows = read_stage_rows()
missing = []
row_lines = []
row_mismatches = []
for name, (path, expected_rows) in expected.items():
    exists = path.exists()
    rows = row_count(path)
    if not exists:
        missing.append(name)
    if expected_rows is not None and rows is not None and rows != expected_rows:
        row_mismatches.append(f"{name}: expected {expected_rows}, found {rows}")
    row_lines.append((name, path, exists, rows, expected_rows))

bulky = [
    p for p in output_dir.rglob("*")
    if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz") or p.name.endswith(".npz"))
]

def first_summary_row(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}

voxel_summary = first_summary_row(output_dir / "voxel_volume" / "mrart_voxel_volume_verification_summary.csv")
ensemble_summary = first_summary_row(output_dir / "ensemble_lcc" / "mrart_ensemble_lcc_from_existing_masks_summary.csv")
paper_long_path = output_dir / "paper_csvs" / "mrart_hallucination_long_per_fold.csv"

def column_counts(path, column):
    if not path.exists():
        return {}
    counts = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            return {"__missing_column__": 1}
        for row in reader:
            value = row.get(column, "")
            counts[value] = counts.get(value, 0) + 1
    return counts

def count_summary(counts):
    if not counts:
        return "missing"
    return ", ".join(f"{key or '<blank>'}={value}" for key, value in sorted(counts.items()))

fold_source_counts = column_counts(paper_long_path, "fold_primary_source")
fold_source_status_counts = column_counts(paper_long_path, "fold_primary_source_status")
fold_consistency_counts = column_counts(paper_long_path, "fold_rerun_consistency_status")
long_rows = row_count(paper_long_path)
rerun_source_complete = (
    long_rows == 8400
    and fold_source_counts == {"rerun_fold_stats": 8400}
    and fold_source_status_counts == {"available": 8400}
)
consistency_needs_review = any(
    status in fold_consistency_counts
    for status in {"mismatch", "not_checked_missing_rerun", "not_checked_missing_completion_field"}
)

failed_or_warn = [
    row for row in stage_rows
    if row.get("status") in {"fail", "warn", "skipped"}
]
paper_complete = (
    not missing
    and not row_mismatches
    and not bulky
    and not failed_or_warn
    and rerun_source_complete
    and not consistency_needs_review
)

commands = command_log_path.read_text(encoding="utf-8") if command_log_path.exists() else ""

lines = [
    "# MR-ART Paper Stats Run Summary",
    "",
    f"- Output directory: `{output_dir}`",
    f"- Paper complete: `{str(paper_complete).lower()}`",
    "- MR-ART is a healthy-control hallucination / false-positive analysis, not lesion accuracy validation.",
    "",
    "## Stage Status",
    "",
    "| stage | status | timestamp | exit_code | note |",
    "| --- | --- | --- | --- | --- |",
]
for row in stage_rows:
    lines.append(
        f"| {row.get('stage','')} | {row.get('status','')} | {row.get('timestamp','')} | "
        f"{row.get('exit_code','')} | {row.get('note','')} |"
    )
lines.extend(["", "## Commands Attempted", "", "```text", commands.rstrip(), "```", "", "## Output Files", ""])
lines.append("| output | exists | rows | expected_rows | path |")
lines.append("| --- | ---: | ---: | ---: | --- |")
for name, path, exists, rows, expected_rows in row_lines:
    lines.append(f"| {name} | {int(exists)} | {'' if rows is None else rows} | {'' if expected_rows is None else expected_rows} | `{path}` |")
lines.extend(["", "## Missing Outputs", ""])
lines.extend([f"- {name}" for name in missing] or ["- none"])
lines.extend(["", "## Row Count Mismatches", ""])
lines.extend([f"- {item}" for item in row_mismatches] or ["- none"])
lines.extend(["", "## Fold-Primary Source Validation", ""])
lines.append("- Required source: `rerun_fold_stats` from `mrart_fold_lcc_stats_long.csv`.")
lines.append(f"- fold_primary_source counts: {count_summary(fold_source_counts)}")
lines.append(f"- fold_primary_source_status counts: {count_summary(fold_source_status_counts)}")
lines.append(f"- fold_rerun_consistency_status counts: {count_summary(fold_consistency_counts)}")
lines.append(f"- all 8,400 fold-primary rows rerun-sourced and available: `{str(rerun_source_complete).lower()}`")
lines.append(f"- completion-vs-rerun consistency needs review: `{str(consistency_needs_review).lower()}`")
lines.extend(["", "## Bulky Output Check", ""])
lines.append(f"- New NIfTI/NPZ files under output directory: {len(bulky)}")
for path in bulky[:20]:
    lines.append(f"- `{path}`")
lines.extend(["", "## Error And Warning Counts", ""])
if voxel_summary:
    lines.append(f"- voxel verification: n_warn={voxel_summary.get('n_warn','')}, n_fail={voxel_summary.get('n_fail','')}")
else:
    lines.append("- voxel verification summary missing")
if ensemble_summary:
    lines.append(
        "- ensemble LCC: "
        f"n_success={ensemble_summary.get('n_success','')}, "
        f"n_missing_prediction={ensemble_summary.get('n_missing_prediction','')}, "
        f"n_error={ensemble_summary.get('n_error','')}"
    )
else:
    lines.append("- ensemble LCC summary missing")
lines.extend(["", "## Required Follow-Up", ""])
if paper_complete:
    lines.append("- none; inspect scientific values before manuscript use.")
else:
    lines.append("- inspect `logs/stage_status.tsv`, stage error logs, and missing/mismatched outputs before treating this run as paper-final.")
lines.append("")

summary_path.write_text("\n".join(lines), encoding="utf-8")
PY
}

log "START stage_5_validate_outputs"
append_status "stage_5_validate_outputs" "started" "" ""
SUMMARY_WRITTEN=0
if write_final_summary 2> "${LOG_DIR}/stage_5_validate_outputs_errors.log"; then
  SUMMARY_WRITTEN=1
  log "PASS stage_5_validate_outputs"
  append_status "stage_5_validate_outputs" "pass" "0" "summary=${RUN_SUMMARY}"
  write_final_summary 2>> "${LOG_DIR}/stage_5_validate_outputs_errors.log" || true
else
  log "FAIL stage_5_validate_outputs; final summary was not written"
  append_status "stage_5_validate_outputs" "fail" "1" "see ${LOG_DIR}/stage_5_validate_outputs_errors.log"
fi

failure_count="$(awk -F '\t' 'NR>1 && $2=="fail" {count++} END {print count+0}' "${STAGE_STATUS}")"
if [ "${STRICT_EXIT}" = "1" ] && [ "${failure_count}" -gt 0 ]; then
  exit 1
fi
if [ "${SUMMARY_WRITTEN}" -eq 1 ]; then
  exit 0
fi
exit 1
