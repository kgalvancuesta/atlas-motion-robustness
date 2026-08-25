#!/usr/bin/env python3
"""Build fold-primary MR-ART hallucination CSVs from lightweight summaries.

The paper's ATLAS model results are fold-level means across five CV folds.
This builder therefore treats individual fold predictions as the primary
MR-ART hallucination unit. Existing 5-fold mean-probability ensemble outputs
are kept only as explicitly labeled secondary consensus/sensitivity fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_COMPLETION_CSVS = [
    Path("runs/mrart_negative_control/full_runs/20260702_032906_mednext_both/merged/completion.csv"),
    Path("runs/mrart_negative_control/full_runs/20260702_233232_swin_unetr_both/merged/completion.csv"),
]

ARCH_LABELS = {
    "mednext": "MedNeXt",
    "swin_unetr": "Swin UNETR",
    "swin": "Swin UNETR",
}
ARCH_KEYS = {
    "mednext": "mednext",
    "mednext standard": "mednext",
    "swin": "swin_unetr",
    "swin_unetr": "swin_unetr",
    "swin unetr": "swin_unetr",
}
TRAINING_LABELS = {
    "standard": "Standard",
    "augmented": "Augmented",
}
TRAINING_KEYS = {
    "standard": "standard",
    "baseline": "standard",
    "augmented": "augmented",
    "aug": "augmented",
}

LONG_COLUMNS = [
    "subject_id",
    "scan_id",
    "acquisition_id",
    "acquisition_type",
    "model_architecture",
    "training_condition",
    "fold",
    "prediction_path",
    "voxel_volume_mm3",
    "predicted_lesion_voxel_count",
    "predicted_lesion_volume_mm3",
    "predicted_lesion_volume_ml",
    "scan_positive",
    "max_probability",
    "mean_probability",
    "connected_component_count_26conn",
    "largest_cc_voxel_count",
    "largest_cc_volume_mm3",
    "largest_cc_volume_ml",
    "largest_cc_fraction_of_prediction",
    "mean_component_voxel_count",
    "median_component_voxel_count",
    "std_component_voxel_count",
    "mean_component_volume_ml",
    "median_component_volume_ml",
    "std_component_volume_ml",
    "max_component_volume_ml",
    "component_count_ge_10_voxels",
    "component_count_ge_100_voxels",
    "component_count_ge_1000_voxels",
    "small_component_count_lt_10_voxels",
    "small_component_voxel_fraction_lt_10_voxels",
    "image_path",
    "metadata_path",
    "threshold",
    "lcc_status",
    "component_status",
    "fold_primary_source",
    "fold_primary_source_status",
    "source_fold_component_csv",
    "source_completion_csv",
    "source_lcc_csv",
    "completion_predicted_lesion_voxel_count",
    "rerun_predicted_lesion_voxel_count",
    "completion_rerun_voxel_count_abs_error",
    "completion_predicted_lesion_volume_ml",
    "rerun_predicted_lesion_volume_ml",
    "completion_rerun_volume_ml_abs_error",
    "completion_scan_positive",
    "rerun_scan_positive",
    "completion_rerun_scan_positive_match",
    "completion_max_probability",
    "rerun_max_probability",
    "completion_rerun_max_probability_abs_error",
    "completion_mean_probability",
    "rerun_mean_probability",
    "completion_rerun_mean_probability_abs_error",
    "fold_rerun_consistency_status",
]

FOLD_AGG_COLUMNS = [
    "subject_id",
    "scan_id",
    "acquisition_id",
    "acquisition_type",
    "model_architecture",
    "training_condition",
    "n_folds_available",
    "fold_positive_rate",
    "any_fold_scan_positive",
    "all_folds_scan_positive",
    "mean_fold_predicted_lesion_voxel_count",
    "median_fold_predicted_lesion_voxel_count",
    "std_fold_predicted_lesion_voxel_count",
    "min_fold_predicted_lesion_voxel_count",
    "max_fold_predicted_lesion_voxel_count",
    "mean_fold_predicted_lesion_volume_ml",
    "median_fold_predicted_lesion_volume_ml",
    "std_fold_predicted_lesion_volume_ml",
    "min_fold_predicted_lesion_volume_ml",
    "max_fold_predicted_lesion_volume_ml",
    "mean_fold_largest_cc_voxel_count",
    "median_fold_largest_cc_voxel_count",
    "std_fold_largest_cc_voxel_count",
    "min_fold_largest_cc_voxel_count",
    "max_fold_largest_cc_voxel_count",
    "mean_fold_largest_cc_volume_ml",
    "median_fold_largest_cc_volume_ml",
    "std_fold_largest_cc_volume_ml",
    "min_fold_largest_cc_volume_ml",
    "max_fold_largest_cc_volume_ml",
    "mean_connected_component_count_26conn",
    "median_connected_component_count_26conn",
    "std_connected_component_count_26conn",
    "max_connected_component_count_26conn",
    "mean_largest_cc_fraction_of_prediction",
    "median_largest_cc_fraction_of_prediction",
    "std_largest_cc_fraction_of_prediction",
    "max_largest_cc_fraction_of_prediction",
    "mean_component_count_ge_10_voxels",
    "mean_component_count_ge_100_voxels",
    "mean_component_count_ge_1000_voxels",
    "mean_small_component_voxel_fraction_lt_10_voxels",
    "max_small_component_voxel_fraction_lt_10_voxels",
    "lcc_status",
    "component_status",
    "source_completion_csv",
]

GROUP_COLUMNS = [
    "model_architecture",
    "training_condition",
    "acquisition_type",
    "n_scans",
    "n_fold_predictions",
    "scan_positive_rate_fold_level",
    "mean_fold_positive_rate_per_scan",
    "median_fold_positive_rate_per_scan",
    "mean_fold_predicted_lesion_volume_ml",
    "median_fold_predicted_lesion_volume_ml",
    "std_fold_predicted_lesion_volume_ml",
    "iqr_fold_predicted_lesion_volume_ml",
    "max_fold_predicted_lesion_volume_ml",
    "mean_fold_largest_cc_volume_ml",
    "median_fold_largest_cc_volume_ml",
    "std_fold_largest_cc_volume_ml",
    "iqr_fold_largest_cc_volume_ml",
    "max_fold_largest_cc_volume_ml",
    "n_scans_with_lcc",
    "lcc_status",
    "mean_connected_component_count_26conn",
    "median_connected_component_count_26conn",
    "std_connected_component_count_26conn",
    "iqr_connected_component_count_26conn",
    "max_connected_component_count_26conn",
    "mean_largest_cc_fraction_of_prediction",
    "median_largest_cc_fraction_of_prediction",
    "std_largest_cc_fraction_of_prediction",
    "iqr_largest_cc_fraction_of_prediction",
    "mean_component_count_ge_10_voxels",
    "mean_component_count_ge_100_voxels",
    "mean_component_count_ge_1000_voxels",
    "mean_small_component_voxel_fraction_lt_10_voxels",
    "n_scans_with_components",
    "component_status",
]

ENSEMBLE_COLUMNS = [
    "subject_id",
    "scan_id",
    "acquisition_id",
    "acquisition_type",
    "model_architecture",
    "training_condition",
    "n_folds_available",
    "ensemble_mean_probability_threshold_scan_positive_at_0_5",
    "ensemble_mean_probability_threshold_voxel_count_at_0_5",
    "ensemble_mean_probability_threshold_volume_ml_at_0_5",
    "ensemble_mean_probability_threshold_largest_cc_voxel_count_at_0_5",
    "ensemble_mean_probability_threshold_largest_cc_volume_mm3_at_0_5",
    "ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5",
    "ensemble_lcc_status",
    "voxel_volume_mm3",
    "threshold",
    "source_completion_csv",
    "source_ensemble_lcc_csv",
]

ENSEMBLE_GROUP_COLUMNS = [
    "model_architecture",
    "training_condition",
    "acquisition_type",
    "n_scans",
    "ensemble_mean_probability_threshold_scan_positive_rate_at_0_5",
    "mean_ensemble_mean_probability_threshold_volume_ml_at_0_5",
    "median_ensemble_mean_probability_threshold_volume_ml_at_0_5",
    "std_ensemble_mean_probability_threshold_volume_ml_at_0_5",
    "iqr_ensemble_mean_probability_threshold_volume_ml_at_0_5",
    "max_ensemble_mean_probability_threshold_volume_ml_at_0_5",
    "mean_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5",
    "median_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5",
    "std_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5",
    "iqr_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5",
    "max_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5",
    "n_scans_with_ensemble_lcc",
    "ensemble_lcc_status",
]

STATUS_COLUMNS = [
    "model_architecture",
    "training_condition",
    "folds_expected",
    "folds_found",
    "scans_found",
    "statistics_found",
    "summarizable_locally",
    "needs_server",
    "nifti_header_access_available_locally",
    "fold_primary_statistics_status",
    "notes",
]

MERGE_DIAGNOSTIC_COLUMNS = [
    "issue",
    "subject_id",
    "acquisition_id",
    "model_architecture",
    "training_condition",
    "fold",
    "source_csv",
    "existing_source_csv",
    "detail",
]

RERUN_PREFERRED_FIELDS = [
    "image_path",
    "voxel_volume_mm3",
    "predicted_lesion_voxel_count",
    "predicted_lesion_volume_mm3",
    "predicted_lesion_volume_ml",
    "scan_positive",
    "max_probability",
    "mean_probability",
    "threshold",
    "connected_component_count_26conn",
    "largest_cc_voxel_count",
    "largest_cc_volume_mm3",
    "largest_cc_volume_ml",
    "largest_cc_fraction_of_prediction",
    "mean_component_voxel_count",
    "median_component_voxel_count",
    "std_component_voxel_count",
    "mean_component_volume_ml",
    "median_component_volume_ml",
    "std_component_volume_ml",
    "max_component_volume_ml",
    "component_count_ge_10_voxels",
    "component_count_ge_100_voxels",
    "component_count_ge_1000_voxels",
    "small_component_count_lt_10_voxels",
    "small_component_voxel_fraction_lt_10_voxels",
]

RERUN_LCC_FIELDS = [
    "largest_cc_voxel_count",
    "largest_cc_volume_mm3",
    "largest_cc_volume_ml",
]

RERUN_COMPONENT_FIELDS = [
    "connected_component_count_26conn",
    "largest_cc_fraction_of_prediction",
    "mean_component_voxel_count",
    "median_component_voxel_count",
    "std_component_voxel_count",
    "mean_component_volume_ml",
    "median_component_volume_ml",
    "std_component_volume_ml",
    "max_component_volume_ml",
    "component_count_ge_10_voxels",
    "component_count_ge_100_voxels",
    "component_count_ge_1000_voxels",
    "small_component_count_lt_10_voxels",
    "small_component_voxel_fraction_lt_10_voxels",
]

COMPLETION_RERUN_COMPARE_FIELDS = [
    "predicted_lesion_voxel_count",
    "predicted_lesion_volume_ml",
    "scan_positive",
    "max_probability",
    "mean_probability",
]

VOLUME_ML_ATOL = 1e-6
PROBABILITY_ATOL = 1e-6
# Existing CSVs are formatted with limited significant digits. Differences
# above the strict tolerance but below this value are tracked as minor numeric
# differences instead of hard mismatches.
MINOR_NUMERIC_ATOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-csv", action="append", type=Path, default=None)
    parser.add_argument(
        "--fold-lcc-csv",
        action="append",
        type=Path,
        default=None,
        help="Optional fold-level LCC/component CSV from run_mrart_fold_lcc_stats_no_outputs.py. Kept as a backward-compatible alias.",
    )
    parser.add_argument(
        "--fold-component-csv",
        action="append",
        type=Path,
        default=None,
        help="Optional fold-level component CSV from run_mrart_fold_lcc_stats_no_outputs.py.",
    )
    parser.add_argument(
        "--ensemble-lcc-csv",
        action="append",
        type=Path,
        default=None,
        help="Optional ensemble LCC CSV from generate_mrart_lcc_from_predictions.py --scope ensemble.",
    )
    parser.add_argument(
        "--voxel-verification-csv",
        action="append",
        type=Path,
        default=None,
        help="Optional voxel-volume verification row CSV. Accepted for pipeline provenance; not required for table construction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory. Defaults to a timestamped tracked-safe directory.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_output_dir() -> Path:
    return Path("scripts/supporting_experiments/mrart/csv") / f"{utc_stamp()}_fold_primary"


def ensure_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def check_not_ignored(paths: Iterable[Path]) -> None:
    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "-v", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            raise RuntimeError(f"Refusing ignored output path: {path}\n{result.stdout.strip()}")
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"git check-ignore failed for {path}: {result.stderr.strip()}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_new_path(path)
    check_not_ignored([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def is_present(value: Any) -> bool:
    return value is not None and str(value) != ""


def is_success(row: dict[str, Any]) -> bool:
    return row.get("status", "") == "success"


def parse_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | int | float | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def normalize_arch(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    key = key.replace(" ", "_")
    if key == "swin_unetr":
        return "swin_unetr"
    if key == "swin":
        return "swin_unetr"
    if key == "mednext":
        return "mednext"
    return ARCH_KEYS.get(value.strip().lower(), key)


def normalize_training(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    return TRAINING_KEYS.get(key, key)


def model_label(architecture: str) -> str:
    return ARCH_LABELS.get(normalize_arch(architecture), architecture)


def training_label(condition: str) -> str:
    return TRAINING_LABELS.get(normalize_training(condition), condition)


def scan_id(subject_id: str, acquisition: str) -> str:
    return f"{subject_id}_acq-{acquisition}"


def folds_from_row(row: dict[str, str]) -> list[int]:
    folds = []
    for part in row.get("folds_used", "").split(","):
        part = part.strip()
        if part:
            folds.append(int(part))
    if folds:
        return folds
    discovered = []
    for key in row:
        if key.startswith("fold") and key.endswith("_predicted_voxels_at_0_5"):
            fold = key.removeprefix("fold").removesuffix("_predicted_voxels_at_0_5")
            if fold.isdigit():
                discovered.append(int(fold))
    return sorted(discovered)


def threshold_from_metadata(metadata_path: str) -> str:
    if not metadata_path:
        return "0.5"
    path = Path(metadata_path)
    if not path.exists():
        return "0.5"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "0.5"
    return format_number(float(payload.get("threshold", 0.5)))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None)


def std_population(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def iqr(values: list[float]) -> float | None:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = parse_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def int_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = parse_int(row.get(key))
        if value is not None:
            values.append(float(value))
    return values


def lcc_key(row: dict[str, str]) -> tuple[str, str, str, str, int] | None:
    subject_id = row.get("subject_id", "")
    acquisition = row.get("acquisition_id") or row.get("acquisition_type") or row.get("acquisition", "")
    arch = row.get("architecture") or row.get("model_architecture", "")
    training = row.get("training_condition", "")
    fold = parse_int(row.get("fold"))
    if not subject_id or not acquisition or not arch or not training or fold is None:
        return None
    return (subject_id, acquisition, normalize_arch(arch), normalize_training(training), fold)


def diagnostic_row(
    issue: str,
    key: tuple[str, str, str, str, int] | None,
    source_csv: str = "",
    existing_source_csv: str = "",
    detail: str = "",
) -> dict[str, str]:
    subject_id = acquisition = arch = training = fold = ""
    if key is not None:
        subject_id, acquisition, arch, training, fold_int = key
        fold = str(fold_int)
    return {
        "issue": issue,
        "subject_id": subject_id,
        "acquisition_id": acquisition,
        "model_architecture": model_label(arch) if arch else "",
        "training_condition": training_label(training) if training else "",
        "fold": fold,
        "source_csv": source_csv,
        "existing_source_csv": existing_source_csv,
        "detail": detail,
    }


def completion_fold_values(
    row: dict[str, str],
    fold: int,
    voxel_volume: float | None,
    threshold: str,
) -> dict[str, Any]:
    voxels = parse_int(row.get(f"fold{fold}_predicted_voxels_at_0_5"))
    volume_ml = parse_float(row.get(f"fold{fold}_predicted_volume_ml_at_0_5"))
    volume_mm3 = volume_ml * 1000.0 if volume_ml is not None else None
    present = parse_bool(row.get(f"fold{fold}_binary_present_at_0_5"))
    return {
        "image_path": row.get("image_path", ""),
        "voxel_volume_mm3": format_number(voxel_volume),
        "predicted_lesion_voxel_count": format_number(voxels),
        "predicted_lesion_volume_mm3": format_number(volume_mm3),
        "predicted_lesion_volume_ml": format_number(volume_ml),
        "scan_positive": int(present),
        "max_probability": row.get(f"fold{fold}_max_probability", ""),
        "mean_probability": row.get(f"fold{fold}_mean_probability", ""),
        "threshold": threshold,
        "connected_component_count_26conn": "",
        "largest_cc_voxel_count": "",
        "largest_cc_volume_mm3": "",
        "largest_cc_volume_ml": "",
        "largest_cc_fraction_of_prediction": "",
        "mean_component_voxel_count": "",
        "median_component_voxel_count": "",
        "std_component_voxel_count": "",
        "mean_component_volume_ml": "",
        "median_component_volume_ml": "",
        "std_component_volume_ml": "",
        "max_component_volume_ml": "",
        "component_count_ge_10_voxels": "",
        "component_count_ge_100_voxels": "",
        "component_count_ge_1000_voxels": "",
        "small_component_count_lt_10_voxels": "",
        "small_component_voxel_fraction_lt_10_voxels": "",
    }


def build_final_fold_values(
    completion_values: dict[str, Any],
    rerun_row: dict[str, str],
    rerun_success: bool,
    fold_key: tuple[str, str, str, str, int],
    diagnostics: list[dict[str, str]],
) -> tuple[dict[str, Any], str, str, str, str]:
    final_values = dict(completion_values)
    source_csv = rerun_row.get("source_lcc_csv", "")
    missing_fields: list[str] = []

    if not rerun_success:
        lcc_status = "missing_not_computed"
        component_status = "missing_not_computed"
        source_status = "rerun_row_error" if rerun_row else "missing_rerun_row"
        return final_values, "completion_csv", source_status, lcc_status, component_status

    for field in RERUN_PREFERRED_FIELDS:
        if is_present(rerun_row.get(field)):
            final_values[field] = rerun_row[field]
        else:
            missing_fields.append(field)
            diagnostics.append(
                diagnostic_row(
                    "missing_rerun_core_field",
                    fold_key,
                    source_csv=source_csv,
                    detail=f"Missing rerun field: {field}",
                )
            )

    lcc_available = all(is_present(final_values.get(field)) for field in RERUN_LCC_FIELDS)
    component_available = all(is_present(final_values.get(field)) for field in RERUN_COMPONENT_FIELDS)
    source_status = "partial_rerun_fields" if missing_fields else "available"
    lcc_status = "available" if lcc_available else "missing_not_computed"
    component_status = "available" if component_available else "missing_not_computed"
    return final_values, "rerun_fold_stats", source_status, lcc_status, component_status


def add_completion_rerun_consistency(
    out_row: dict[str, Any],
    completion_values: dict[str, Any],
    rerun_row: dict[str, str],
    rerun_success: bool,
    fold_key: tuple[str, str, str, str, int],
    diagnostics: list[dict[str, str]],
) -> None:
    source_csv = rerun_row.get("source_lcc_csv", "")
    out_row.update(
        {
            "completion_predicted_lesion_voxel_count": completion_values["predicted_lesion_voxel_count"],
            "rerun_predicted_lesion_voxel_count": rerun_row.get("predicted_lesion_voxel_count", ""),
            "completion_rerun_voxel_count_abs_error": "",
            "completion_predicted_lesion_volume_ml": completion_values["predicted_lesion_volume_ml"],
            "rerun_predicted_lesion_volume_ml": rerun_row.get("predicted_lesion_volume_ml", ""),
            "completion_rerun_volume_ml_abs_error": "",
            "completion_scan_positive": completion_values["scan_positive"],
            "rerun_scan_positive": rerun_row.get("scan_positive", ""),
            "completion_rerun_scan_positive_match": "",
            "completion_max_probability": completion_values["max_probability"],
            "rerun_max_probability": rerun_row.get("max_probability", ""),
            "completion_rerun_max_probability_abs_error": "",
            "completion_mean_probability": completion_values["mean_probability"],
            "rerun_mean_probability": rerun_row.get("mean_probability", ""),
            "completion_rerun_mean_probability_abs_error": "",
            "fold_rerun_consistency_status": "not_checked_missing_rerun",
        }
    )
    if not rerun_success:
        return

    missing_completion = [
        field for field in COMPLETION_RERUN_COMPARE_FIELDS if not is_present(completion_values.get(field))
    ]
    if missing_completion:
        out_row["fold_rerun_consistency_status"] = "not_checked_missing_completion_field"
        diagnostics.append(
            diagnostic_row(
                "missing_completion_compare_field",
                fold_key,
                source_csv=out_row.get("source_completion_csv", ""),
                existing_source_csv=source_csv,
                detail=";".join(missing_completion),
            )
        )
        return
    missing_rerun = [
        field for field in COMPLETION_RERUN_COMPARE_FIELDS if not is_present(rerun_row.get(field))
    ]
    if missing_rerun:
        out_row["fold_rerun_consistency_status"] = "not_checked_missing_rerun"
        diagnostics.append(
            diagnostic_row(
                "missing_rerun_core_field",
                fold_key,
                source_csv=source_csv,
                detail="Missing rerun comparison fields: " + ";".join(missing_rerun),
            )
        )
        return

    strict_mismatch = False
    minor_numeric = False

    completion_voxels = parse_int(completion_values.get("predicted_lesion_voxel_count"))
    rerun_voxels = parse_int(rerun_row.get("predicted_lesion_voxel_count"))
    if completion_voxels is None or rerun_voxels is None:
        out_row["fold_rerun_consistency_status"] = "not_checked_missing_completion_field"
        return
    voxel_abs_error = abs(completion_voxels - rerun_voxels)
    out_row["completion_rerun_voxel_count_abs_error"] = str(voxel_abs_error)
    if voxel_abs_error != 0:
        strict_mismatch = True
        diagnostics.append(
            diagnostic_row(
                "voxel_count_mismatch",
                fold_key,
                source_csv=source_csv,
                detail=f"completion={completion_voxels};rerun={rerun_voxels};abs_error={voxel_abs_error}",
            )
        )

    completion_scan = int(parse_bool(completion_values.get("scan_positive")))
    rerun_scan = int(parse_bool(rerun_row.get("scan_positive")))
    scan_match = completion_scan == rerun_scan
    out_row["completion_rerun_scan_positive_match"] = int(scan_match)
    if not scan_match:
        strict_mismatch = True
        diagnostics.append(
            diagnostic_row(
                "scan_positive_mismatch",
                fold_key,
                source_csv=source_csv,
                detail=f"completion={completion_scan};rerun={rerun_scan}",
            )
        )

    def compare_float(field: str, error_column: str, tolerance: float, issue: str) -> None:
        nonlocal strict_mismatch, minor_numeric
        completion_value = parse_float(completion_values.get(field))
        rerun_value = parse_float(rerun_row.get(field))
        if completion_value is None or rerun_value is None:
            strict_mismatch = True
            diagnostics.append(
                diagnostic_row(
                    issue,
                    fold_key,
                    source_csv=source_csv,
                    detail=f"missing numeric comparison field: {field}",
                )
            )
            return
        abs_error = abs(completion_value - rerun_value)
        out_row[error_column] = format_number(abs_error)
        if abs_error > tolerance:
            if abs_error <= MINOR_NUMERIC_ATOL:
                minor_numeric = True
            else:
                strict_mismatch = True
            diagnostics.append(
                diagnostic_row(
                    issue,
                    fold_key,
                    source_csv=source_csv,
                    detail=f"completion={completion_value};rerun={rerun_value};abs_error={abs_error}",
                )
            )

    compare_float("predicted_lesion_volume_ml", "completion_rerun_volume_ml_abs_error", VOLUME_ML_ATOL, "volume_mismatch")
    compare_float("max_probability", "completion_rerun_max_probability_abs_error", PROBABILITY_ATOL, "probability_mismatch")
    compare_float("mean_probability", "completion_rerun_mean_probability_abs_error", PROBABILITY_ATOL, "probability_mismatch")

    if strict_mismatch:
        out_row["fold_rerun_consistency_status"] = "mismatch"
    elif minor_numeric:
        out_row["fold_rerun_consistency_status"] = "minor_numeric_difference"
    else:
        out_row["fold_rerun_consistency_status"] = "match"


def ensemble_key(row: dict[str, str]) -> tuple[str, str, str, str] | None:
    subject_id = row.get("subject_id", "")
    acquisition = row.get("acquisition_id") or row.get("acquisition_type") or row.get("acquisition", "")
    arch = row.get("architecture") or row.get("model_architecture", "")
    training = row.get("training_condition", "")
    if not subject_id or not acquisition or not arch or not training:
        return None
    return (subject_id, acquisition, normalize_arch(arch), normalize_training(training))


def load_fold_lcc(
    paths: list[Path] | None,
) -> tuple[dict[tuple[str, str, str, str, int], dict[str, str]], list[dict[str, str]]]:
    records: dict[tuple[str, str, str, str, int], dict[str, str]] = {}
    diagnostics: list[dict[str, str]] = []
    for path in paths or []:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv(path):
            key = lcc_key(row)
            if key is None:
                diagnostics.append(
                    diagnostic_row(
                        "missing_fold_component_merge_key",
                        None,
                        source_csv=str(path),
                        detail=json.dumps({k: row.get(k, "") for k in ("subject_id", "acquisition_id", "acquisition_type", "architecture", "model_architecture", "training_condition", "fold")}, sort_keys=True),
                    )
                )
                continue
            if key in records:
                diagnostics.append(
                    diagnostic_row(
                        "duplicate_rerun_merge_key",
                        key,
                        source_csv=str(path),
                        existing_source_csv=records[key].get("source_lcc_csv", ""),
                        detail="Keeping first successful row when possible; otherwise keeping first row.",
                    )
                )
                if not is_success(records[key]) and row.get("status", "") == "success":
                    row = dict(row)
                    row["source_lcc_csv"] = str(path)
                    records[key] = row
                continue
            row = dict(row)
            row["source_lcc_csv"] = str(path)
            records[key] = row
    return records, diagnostics


def load_ensemble_lcc(paths: list[Path] | None) -> dict[tuple[str, str, str, str], dict[str, str]]:
    records: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for path in paths or []:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv(path):
            key = ensemble_key(row)
            if key is None:
                continue
            status = row.get("ensemble_lcc_status") or row.get("status", "")
            if status and status not in {"success", "pass"}:
                continue
            row = dict(row)
            row["source_ensemble_lcc_csv"] = str(path)
            records[key] = row
    return records


def lcc_status_for_scan(rows: list[dict[str, Any]]) -> str:
    statuses = {row.get("lcc_status", "") for row in rows}
    if statuses == {"available"}:
        return "available"
    if "available" in statuses:
        return "partial"
    return "missing_not_computed"


def component_status_for_scan(rows: list[dict[str, Any]]) -> str:
    statuses = {row.get("component_status", "") for row in rows}
    if statuses == {"available"}:
        return "available"
    if "available" in statuses:
        return "partial"
    return "missing_not_computed"


def build_rows(
    completion_paths: list[Path],
    fold_lcc: dict[tuple[str, str, str, str, int], dict[str, str]],
    ensemble_lcc: dict[tuple[str, str, str, str], dict[str, str]],
    fold_component_csv_supplied: bool,
    merge_diagnostics: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    status_counts: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "folds": set(),
            "scans": set(),
            "rows": 0,
            "fold_primary_sources": [],
            "fold_primary_source_statuses": [],
            "fold_rerun_consistency_statuses": [],
        }
    )
    seen_completion_fold_keys: set[tuple[str, str, str, str, int]] = set()

    for completion_path in completion_paths:
        for row in read_csv(completion_path):
            if row.get("status") and row["status"] != "success":
                continue
            subject_id = row["subject_id"]
            acquisition = row["acquisition"]
            arch_key = normalize_arch(row["architecture"])
            training_key = normalize_training(row["training_condition"])
            folds = folds_from_row(row)
            voxel_volume = parse_float(row.get("voxel_volume_mm3"))
            threshold = threshold_from_metadata(row.get("metadata_path", ""))
            sid = scan_id(subject_id, acquisition)

            status_key = (arch_key, training_key)
            status_counts[status_key]["folds"].update(folds)
            status_counts[status_key]["scans"].add(sid)
            status_counts[status_key]["rows"] += 1

            for fold in folds:
                prediction_path = row.get(f"fold{fold}_mask_path") or row.get(f"fold{fold}_prob_path") or ""
                fold_key = (subject_id, acquisition, arch_key, training_key, fold)
                if fold_key in seen_completion_fold_keys:
                    merge_diagnostics.append(
                        diagnostic_row(
                            "duplicate_completion_fold_key",
                            fold_key,
                            source_csv=str(completion_path),
                            detail="Duplicate completion fold key encountered; row retained for transparency.",
                        )
                    )
                seen_completion_fold_keys.add(fold_key)

                rerun = fold_lcc.get(fold_key, {})
                rerun_success = bool(rerun) and is_success(rerun)
                if fold_component_csv_supplied and not rerun:
                    merge_diagnostics.append(
                        diagnostic_row(
                            "missing_rerun_row",
                            fold_key,
                            source_csv=str(completion_path),
                            detail="Completion row expected a fold-level rerun row, but no matching key was found.",
                        )
                    )
                elif fold_component_csv_supplied and rerun and not rerun_success:
                    merge_diagnostics.append(
                        diagnostic_row(
                            "rerun_row_error",
                            fold_key,
                            source_csv=rerun.get("source_lcc_csv", ""),
                            detail=f"status={rerun.get('status', '')};error_type={rerun.get('error_type', '')};error_message={rerun.get('error_message', '')}",
                        )
                    )

                completion_values = completion_fold_values(row, fold, voxel_volume, threshold)
                final_values, fold_primary_source, source_status, lcc_status, component_status = build_final_fold_values(
                    completion_values,
                    rerun,
                    rerun_success,
                    fold_key,
                    merge_diagnostics,
                )
                out_row = {
                    "subject_id": subject_id,
                    "scan_id": sid,
                    "acquisition_id": acquisition,
                    "acquisition_type": acquisition,
                    "model_architecture": model_label(arch_key),
                    "training_condition": training_label(training_key),
                    "fold": fold,
                    "prediction_path": prediction_path,
                    "metadata_path": row.get("metadata_path", ""),
                    "lcc_status": lcc_status,
                    "component_status": component_status,
                    "fold_primary_source": fold_primary_source,
                    "fold_primary_source_status": source_status,
                    "source_fold_component_csv": rerun.get("source_lcc_csv", ""),
                    "source_completion_csv": str(completion_path),
                    "source_lcc_csv": rerun.get("source_lcc_csv", ""),
                }
                out_row.update(final_values)
                add_completion_rerun_consistency(
                    out_row,
                    completion_values,
                    rerun,
                    rerun_success,
                    fold_key,
                    merge_diagnostics,
                )
                status_counts[status_key]["fold_primary_sources"].append(fold_primary_source)
                status_counts[status_key]["fold_primary_source_statuses"].append(source_status)
                status_counts[status_key]["fold_rerun_consistency_statuses"].append(
                    out_row["fold_rerun_consistency_status"]
                )
                long_rows.append(out_row)

            ens = ensemble_lcc.get((subject_id, acquisition, arch_key, training_key), {})
            ens_status = "available" if ens else "missing_not_computed"
            ensemble_rows.append(
                {
                    "subject_id": subject_id,
                    "scan_id": sid,
                    "acquisition_id": acquisition,
                    "acquisition_type": acquisition,
                    "model_architecture": model_label(arch_key),
                    "training_condition": training_label(training_key),
                    "n_folds_available": len(folds),
                    "ensemble_mean_probability_threshold_scan_positive_at_0_5": int(
                        parse_bool(row.get("ensemble_binary_present_at_0_5"))
                    ),
                    "ensemble_mean_probability_threshold_voxel_count_at_0_5": row.get(
                        "ensemble_predicted_voxels_at_0_5", ""
                    ),
                    "ensemble_mean_probability_threshold_volume_ml_at_0_5": row.get(
                        "ensemble_predicted_volume_ml_at_0_5", ""
                    ),
                    "ensemble_mean_probability_threshold_largest_cc_voxel_count_at_0_5": ens.get(
                        "ensemble_largest_cc_voxel_count",
                        ens.get("largest_cc_voxel_count", ""),
                    ),
                    "ensemble_mean_probability_threshold_largest_cc_volume_mm3_at_0_5": ens.get(
                        "ensemble_largest_cc_volume_mm3",
                        ens.get("largest_cc_volume_mm3", ""),
                    ),
                    "ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5": ens.get(
                        "ensemble_largest_cc_volume_ml",
                        ens.get("largest_cc_volume_ml", ""),
                    ),
                    "ensemble_lcc_status": ens.get("ensemble_lcc_status", ens_status),
                    "voxel_volume_mm3": format_number(voxel_volume),
                    "threshold": threshold,
                    "source_completion_csv": str(completion_path),
                    "source_ensemble_lcc_csv": ens.get("source_ensemble_lcc_csv", ""),
                }
            )

    status_rows = []
    for arch_key in ["mednext", "swin_unetr"]:
        for training_key in ["standard", "augmented"]:
            data = status_counts.get((arch_key, training_key), {"folds": set(), "scans": set(), "rows": 0})
            scans_found = len(data["scans"])
            folds_found = ",".join(str(fold) for fold in sorted(data["folds"]))
            has_expected = scans_found == 420 and set(data["folds"]) == {0, 1, 2, 3, 4}
            lcc_available = any(
                key[2] == arch_key and key[3] == training_key
                and is_success(row)
                for key, row in fold_lcc.items()
            )
            component_available = any(
                key[2] == arch_key
                and key[3] == training_key
                and is_success(row)
                and row.get("connected_component_count_26conn", "") != ""
                for key, row in fold_lcc.items()
            )
            source_statuses = data.get("fold_primary_source_statuses", [])
            sources = data.get("fold_primary_sources", [])
            consistency_statuses = data.get("fold_rerun_consistency_statuses", [])
            if (
                sources
                and set(sources) == {"rerun_fold_stats"}
                and set(source_statuses) == {"available"}
                and set(consistency_statuses).issubset({"match", "minor_numeric_difference"})
            ):
                fold_primary_statistics_status = "rerun_fold_stats_primary"
            elif not sources or set(sources) == {"completion_csv"}:
                fold_primary_statistics_status = "completion_csv_primary_missing_rerun"
            else:
                fold_primary_statistics_status = "mixed_or_partial_needs_review"
            if fold_primary_statistics_status != "completion_csv_primary_missing_rerun" and any(
                status in {"mismatch", "not_checked_missing_rerun", "not_checked_missing_completion_field"}
                for status in consistency_statuses
            ):
                fold_primary_statistics_status = "mixed_or_partial_needs_review"
            if component_available:
                statistics_found = "fold_voxels_volumes_scan_positive_and_lcc_components"
                needs_server = "yes_for_header_reverification"
            elif lcc_available:
                statistics_found = "fold_voxels_volumes_scan_positive_and_lcc_without_components"
                needs_server = "yes_for_fold_components_and_header_reverification"
            else:
                statistics_found = "fold_voxels_volumes_scan_positive_without_lcc_components"
                needs_server = "yes_for_fold_lcc_components_and_header_reverification"
            status_rows.append(
                {
                    "model_architecture": model_label(arch_key),
                    "training_condition": training_label(training_key),
                    "folds_expected": "0,1,2,3,4",
                    "folds_found": folds_found,
                    "scans_found": scans_found,
                    "statistics_found": statistics_found,
                    "summarizable_locally": "yes_fold_primary_without_lcc" if has_expected and not lcc_available else str(has_expected).lower(),
                    "needs_server": needs_server,
                    "nifti_header_access_available_locally": "no_mrart_source_headers_absent",
                    "fold_primary_statistics_status": fold_primary_statistics_status,
                    "notes": "Final fold-primary fields use successful rerun rows when supplied; completion CSV fields are retained for provenance and used only when a matching successful rerun row is unavailable. Ensemble outputs remain secondary.",
                }
            )
    return long_rows, ensemble_rows, status_rows, merge_diagnostics


def build_fold_aggregates(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        key = (
            row["subject_id"],
            row["scan_id"],
            row["acquisition_id"],
            row["acquisition_type"],
            row["model_architecture"],
            row["training_condition"],
        )
        grouped[key].append(row)

    out_rows = []
    for key, rows in sorted(grouped.items()):
        subject_id, sid, acquisition_id, acquisition_type, architecture, training = key
        positives = [int(row["scan_positive"]) for row in rows]
        voxel_counts = int_values(rows, "predicted_lesion_voxel_count")
        volumes = numeric_values(rows, "predicted_lesion_volume_ml")
        lcc_voxels = int_values(rows, "largest_cc_voxel_count")
        lcc_volumes = numeric_values(rows, "largest_cc_volume_ml")
        component_counts = numeric_values(rows, "connected_component_count_26conn")
        lcc_fractions = numeric_values(rows, "largest_cc_fraction_of_prediction")
        components_ge_10 = numeric_values(rows, "component_count_ge_10_voxels")
        components_ge_100 = numeric_values(rows, "component_count_ge_100_voxels")
        components_ge_1000 = numeric_values(rows, "component_count_ge_1000_voxels")
        small_component_fractions = numeric_values(rows, "small_component_voxel_fraction_lt_10_voxels")
        out_rows.append(
            {
                "subject_id": subject_id,
                "scan_id": sid,
                "acquisition_id": acquisition_id,
                "acquisition_type": acquisition_type,
                "model_architecture": architecture,
                "training_condition": training,
                "n_folds_available": len(rows),
                "fold_positive_rate": format_number(mean([float(v) for v in positives])),
                "any_fold_scan_positive": int(any(positives)),
                "all_folds_scan_positive": int(all(positives)),
                "mean_fold_predicted_lesion_voxel_count": format_number(mean(voxel_counts)),
                "median_fold_predicted_lesion_voxel_count": format_number(median(voxel_counts)),
                "std_fold_predicted_lesion_voxel_count": format_number(std(voxel_counts)),
                "min_fold_predicted_lesion_voxel_count": format_number(min(voxel_counts) if voxel_counts else None),
                "max_fold_predicted_lesion_voxel_count": format_number(max(voxel_counts) if voxel_counts else None),
                "mean_fold_predicted_lesion_volume_ml": format_number(mean(volumes)),
                "median_fold_predicted_lesion_volume_ml": format_number(median(volumes)),
                "std_fold_predicted_lesion_volume_ml": format_number(std(volumes)),
                "min_fold_predicted_lesion_volume_ml": format_number(min(volumes) if volumes else None),
                "max_fold_predicted_lesion_volume_ml": format_number(max(volumes) if volumes else None),
                "mean_fold_largest_cc_voxel_count": format_number(mean(lcc_voxels)),
                "median_fold_largest_cc_voxel_count": format_number(median(lcc_voxels)),
                "std_fold_largest_cc_voxel_count": format_number(std(lcc_voxels)),
                "min_fold_largest_cc_voxel_count": format_number(min(lcc_voxels) if lcc_voxels else None),
                "max_fold_largest_cc_voxel_count": format_number(max(lcc_voxels) if lcc_voxels else None),
                "mean_fold_largest_cc_volume_ml": format_number(mean(lcc_volumes)),
                "median_fold_largest_cc_volume_ml": format_number(median(lcc_volumes)),
                "std_fold_largest_cc_volume_ml": format_number(std(lcc_volumes)),
                "min_fold_largest_cc_volume_ml": format_number(min(lcc_volumes) if lcc_volumes else None),
                "max_fold_largest_cc_volume_ml": format_number(max(lcc_volumes) if lcc_volumes else None),
                "mean_connected_component_count_26conn": format_number(mean(component_counts)),
                "median_connected_component_count_26conn": format_number(median(component_counts)),
                "std_connected_component_count_26conn": format_number(std_population(component_counts)),
                "max_connected_component_count_26conn": format_number(max(component_counts) if component_counts else None),
                "mean_largest_cc_fraction_of_prediction": format_number(mean(lcc_fractions)),
                "median_largest_cc_fraction_of_prediction": format_number(median(lcc_fractions)),
                "std_largest_cc_fraction_of_prediction": format_number(std_population(lcc_fractions)),
                "max_largest_cc_fraction_of_prediction": format_number(max(lcc_fractions) if lcc_fractions else None),
                "mean_component_count_ge_10_voxels": format_number(mean(components_ge_10)),
                "mean_component_count_ge_100_voxels": format_number(mean(components_ge_100)),
                "mean_component_count_ge_1000_voxels": format_number(mean(components_ge_1000)),
                "mean_small_component_voxel_fraction_lt_10_voxels": format_number(mean(small_component_fractions)),
                "max_small_component_voxel_fraction_lt_10_voxels": format_number(
                    max(small_component_fractions) if small_component_fractions else None
                ),
                "lcc_status": lcc_status_for_scan(rows),
                "component_status": component_status_for_scan(rows),
                "source_completion_csv": ";".join(sorted(set(row["source_completion_csv"] for row in rows))),
            }
        )
    return out_rows


def build_group_rows(long_rows: list[dict[str, Any]], fold_aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fold_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    scan_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        fold_grouped[(row["model_architecture"], row["training_condition"], row["acquisition_type"])].append(row)
    for row in fold_aggregate_rows:
        scan_grouped[(row["model_architecture"], row["training_condition"], row["acquisition_type"])].append(row)

    group_rows = []
    for key in sorted(fold_grouped):
        architecture, training, acquisition = key
        fold_rows = fold_grouped[key]
        scan_rows = scan_grouped[key]
        positives = [int(row["scan_positive"]) for row in fold_rows]
        fold_positive_rates = numeric_values(scan_rows, "fold_positive_rate")
        volumes = numeric_values(fold_rows, "predicted_lesion_volume_ml")
        lcc_volumes = numeric_values(fold_rows, "largest_cc_volume_ml")
        component_counts = numeric_values(fold_rows, "connected_component_count_26conn")
        lcc_fractions = numeric_values(fold_rows, "largest_cc_fraction_of_prediction")
        components_ge_10 = numeric_values(fold_rows, "component_count_ge_10_voxels")
        components_ge_100 = numeric_values(fold_rows, "component_count_ge_100_voxels")
        components_ge_1000 = numeric_values(fold_rows, "component_count_ge_1000_voxels")
        small_component_fractions = numeric_values(fold_rows, "small_component_voxel_fraction_lt_10_voxels")
        scans_with_lcc = sum(1 for row in scan_rows if row["lcc_status"] in {"available", "partial"})
        scans_with_components = sum(1 for row in scan_rows if row["component_status"] in {"available", "partial"})
        group_rows.append(
            {
                "model_architecture": architecture,
                "training_condition": training,
                "acquisition_type": acquisition,
                "n_scans": len(scan_rows),
                "n_fold_predictions": len(fold_rows),
                "scan_positive_rate_fold_level": format_number(mean([float(v) for v in positives])),
                "mean_fold_positive_rate_per_scan": format_number(mean(fold_positive_rates)),
                "median_fold_positive_rate_per_scan": format_number(median(fold_positive_rates)),
                "mean_fold_predicted_lesion_volume_ml": format_number(mean(volumes)),
                "median_fold_predicted_lesion_volume_ml": format_number(median(volumes)),
                "std_fold_predicted_lesion_volume_ml": format_number(std(volumes)),
                "iqr_fold_predicted_lesion_volume_ml": format_number(iqr(volumes)),
                "max_fold_predicted_lesion_volume_ml": format_number(max(volumes) if volumes else None),
                "mean_fold_largest_cc_volume_ml": format_number(mean(lcc_volumes)),
                "median_fold_largest_cc_volume_ml": format_number(median(lcc_volumes)),
                "std_fold_largest_cc_volume_ml": format_number(std(lcc_volumes)),
                "iqr_fold_largest_cc_volume_ml": format_number(iqr(lcc_volumes)),
                "max_fold_largest_cc_volume_ml": format_number(max(lcc_volumes) if lcc_volumes else None),
                "n_scans_with_lcc": scans_with_lcc,
                "lcc_status": "available" if scans_with_lcc == len(scan_rows) else ("partial" if scans_with_lcc else "missing_not_computed"),
                "mean_connected_component_count_26conn": format_number(mean(component_counts)),
                "median_connected_component_count_26conn": format_number(median(component_counts)),
                "std_connected_component_count_26conn": format_number(std_population(component_counts)),
                "iqr_connected_component_count_26conn": format_number(iqr(component_counts)),
                "max_connected_component_count_26conn": format_number(max(component_counts) if component_counts else None),
                "mean_largest_cc_fraction_of_prediction": format_number(mean(lcc_fractions)),
                "median_largest_cc_fraction_of_prediction": format_number(median(lcc_fractions)),
                "std_largest_cc_fraction_of_prediction": format_number(std_population(lcc_fractions)),
                "iqr_largest_cc_fraction_of_prediction": format_number(iqr(lcc_fractions)),
                "mean_component_count_ge_10_voxels": format_number(mean(components_ge_10)),
                "mean_component_count_ge_100_voxels": format_number(mean(components_ge_100)),
                "mean_component_count_ge_1000_voxels": format_number(mean(components_ge_1000)),
                "mean_small_component_voxel_fraction_lt_10_voxels": format_number(mean(small_component_fractions)),
                "n_scans_with_components": scans_with_components,
                "component_status": "available"
                if scans_with_components == len(scan_rows)
                else ("partial" if scans_with_components else "missing_not_computed"),
            }
        )
    return group_rows


def build_ensemble_group_rows(ensemble_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ensemble_rows:
        grouped[(row["model_architecture"], row["training_condition"], row["acquisition_type"])].append(row)

    out_rows = []
    for (architecture, training, acquisition), rows in sorted(grouped.items()):
        positives = [int(row["ensemble_mean_probability_threshold_scan_positive_at_0_5"]) for row in rows]
        volumes = numeric_values(rows, "ensemble_mean_probability_threshold_volume_ml_at_0_5")
        lcc_volumes = numeric_values(rows, "ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5")
        n_lcc = len(lcc_volumes)
        out_rows.append(
            {
                "model_architecture": architecture,
                "training_condition": training,
                "acquisition_type": acquisition,
                "n_scans": len(rows),
                "ensemble_mean_probability_threshold_scan_positive_rate_at_0_5": format_number(mean([float(v) for v in positives])),
                "mean_ensemble_mean_probability_threshold_volume_ml_at_0_5": format_number(mean(volumes)),
                "median_ensemble_mean_probability_threshold_volume_ml_at_0_5": format_number(median(volumes)),
                "std_ensemble_mean_probability_threshold_volume_ml_at_0_5": format_number(std(volumes)),
                "iqr_ensemble_mean_probability_threshold_volume_ml_at_0_5": format_number(iqr(volumes)),
                "max_ensemble_mean_probability_threshold_volume_ml_at_0_5": format_number(max(volumes) if volumes else None),
                "mean_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5": format_number(mean(lcc_volumes)),
                "median_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5": format_number(median(lcc_volumes)),
                "std_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5": format_number(std(lcc_volumes)),
                "iqr_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5": format_number(iqr(lcc_volumes)),
                "max_ensemble_mean_probability_threshold_largest_cc_volume_ml_at_0_5": format_number(max(lcc_volumes) if lcc_volumes else None),
                "n_scans_with_ensemble_lcc": n_lcc,
                "ensemble_lcc_status": "available" if n_lcc == len(rows) else ("partial" if n_lcc else "missing_not_computed"),
            }
        )
    return out_rows


def main() -> None:
    args = parse_args()
    completion_paths = args.completion_csv or DEFAULT_COMPLETION_CSVS
    for path in completion_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir or default_output_dir()
    long_path = output_dir / "mrart_hallucination_long_per_fold.csv"
    fold_agg_path = output_dir / "mrart_hallucination_per_scan_fold_summary.csv"
    group_path = output_dir / "mrart_hallucination_grouped_summary_fold_primary.csv"
    ensemble_path = output_dir / "mrart_hallucination_ensemble_secondary_per_scan.csv"
    ensemble_group_path = output_dir / "mrart_hallucination_ensemble_secondary_summary.csv"
    status_path = output_dir / "mrart_hallucination_status_matrix.csv"
    merge_diagnostics_path = output_dir / "mrart_hallucination_component_merge_diagnostics.csv"
    rerun_diagnostics_path = output_dir / "mrart_hallucination_fold_rerun_consistency_diagnostics.csv"
    output_paths = [
        long_path,
        fold_agg_path,
        group_path,
        ensemble_path,
        ensemble_group_path,
        status_path,
        merge_diagnostics_path,
        rerun_diagnostics_path,
    ]
    for path in output_paths:
        ensure_new_path(path)
    check_not_ignored(output_paths)

    fold_component_paths = (args.fold_lcc_csv or []) + (args.fold_component_csv or [])
    fold_lcc, merge_diagnostics = load_fold_lcc(fold_component_paths)
    ensemble_lcc = load_ensemble_lcc(args.ensemble_lcc_csv)
    long_rows, ensemble_rows, status_rows, merge_diagnostics = build_rows(
        completion_paths,
        fold_lcc,
        ensemble_lcc,
        bool(fold_component_paths),
        merge_diagnostics,
    )
    fold_agg_rows = build_fold_aggregates(long_rows)
    group_rows = build_group_rows(long_rows, fold_agg_rows)
    ensemble_group_rows = build_ensemble_group_rows(ensemble_rows)

    write_csv(long_path, LONG_COLUMNS, long_rows)
    write_csv(fold_agg_path, FOLD_AGG_COLUMNS, fold_agg_rows)
    write_csv(group_path, GROUP_COLUMNS, group_rows)
    write_csv(ensemble_path, ENSEMBLE_COLUMNS, ensemble_rows)
    write_csv(ensemble_group_path, ENSEMBLE_GROUP_COLUMNS, ensemble_group_rows)
    write_csv(status_path, STATUS_COLUMNS, status_rows)
    write_csv(merge_diagnostics_path, MERGE_DIAGNOSTIC_COLUMNS, merge_diagnostics)
    write_csv(rerun_diagnostics_path, MERGE_DIAGNOSTIC_COLUMNS, merge_diagnostics)

    print(f"Wrote {long_path} rows={len(long_rows)} cols={len(LONG_COLUMNS)}")
    print(f"Wrote {fold_agg_path} rows={len(fold_agg_rows)} cols={len(FOLD_AGG_COLUMNS)}")
    print(f"Wrote {group_path} rows={len(group_rows)} cols={len(GROUP_COLUMNS)}")
    print(f"Wrote {ensemble_path} rows={len(ensemble_rows)} cols={len(ENSEMBLE_COLUMNS)}")
    print(f"Wrote {ensemble_group_path} rows={len(ensemble_group_rows)} cols={len(ENSEMBLE_GROUP_COLUMNS)}")
    print(f"Wrote {status_path} rows={len(status_rows)} cols={len(STATUS_COLUMNS)}")
    print(f"Wrote {merge_diagnostics_path} rows={len(merge_diagnostics)} cols={len(MERGE_DIAGNOSTIC_COLUMNS)}")
    print(f"Wrote {rerun_diagnostics_path} rows={len(merge_diagnostics)} cols={len(MERGE_DIAGNOSTIC_COLUMNS)}")


if __name__ == "__main__":
    main()
