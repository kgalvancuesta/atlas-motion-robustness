#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "FAIL: PACE preflight requires Linux." >&2
  echo "UNSAFE TO START FULL RUN"
  exit 1
fi

SCRATCH_LINK="${HOME}/scratch"
if ! SCRATCH_RESOLVED="$(readlink -f "${SCRATCH_LINK}" 2>/dev/null)"; then
  echo "FAIL: could not resolve PACE scratch symlink: ${SCRATCH_LINK}" >&2
  echo "UNSAFE TO START FULL RUN"
  exit 1
fi
if [[ -z "${SCRATCH_RESOLVED}" || ! -d "${SCRATCH_RESOLVED}" || ! -w "${SCRATCH_RESOLVED}" ]]; then
  echo "FAIL: expected writable PACE scratch is unavailable: ${SCRATCH_LINK} -> ${SCRATCH_RESOLVED:-<unresolved>}" >&2
  echo "UNSAFE TO START FULL RUN"
  exit 1
fi
if [[ "${SCRATCH_RESOLVED}" == "/" || "${SCRATCH_RESOLVED}" == "${HOME}" ]]; then
  echo "FAIL: unsafe scratch resolution: ${SCRATCH_RESOLVED}" >&2
  echo "UNSAFE TO START FULL RUN"
  exit 1
fi
if ! command -v pace-quota >/dev/null 2>&1; then
  echo "FAIL: pace-quota is unavailable; scratch allocation cannot be verified." >&2
  echo "UNSAFE TO START FULL RUN"
  exit 1
fi
if ! pace-quota; then
  echo "FAIL: pace-quota could not verify the PACE allocation." >&2
  echo "UNSAFE TO START FULL RUN"
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
DATA_ROOT="${ATLAS_DATA_ROOT:-${REPO_ROOT}/data}"
CHECKPOINT_ROOT="${ATLAS_PHASE1_CHECKPOINT_ROOT:-${HOME}/r-cpradalier7-0/reproduction_outputs}"
PREFLIGHT_ROOT="${SCRATCH_RESOLVED}/atlas_corrected_preflight"
mkdir -p "${PREFLIGHT_ROOT}"

set +e
PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" python3 -m reproducibility.preflight \
  --data-root "${DATA_ROOT}" \
  --split-path splits/atlas_5fold_lesion_quartile_excluding_known_issues.json \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --scratch-root "${PREFLIGHT_ROOT}"
PREFLIGHT_EXIT=$?
set -e
if [[ "${PREFLIGHT_EXIT}" -ne 0 ]]; then
  echo "UNSAFE TO START FULL RUN"
fi
exit "${PREFLIGHT_EXIT}"
