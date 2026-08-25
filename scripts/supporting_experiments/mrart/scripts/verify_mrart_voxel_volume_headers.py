#!/usr/bin/env python3
"""Verify MR-ART voxel-volume assumptions against source NIfTI headers.

Run this on the server if source MR-ART T1w NIfTI images or prediction NIfTI
headers are not present locally. The script reads lightweight stats/completion
CSVs, opens referenced NIfTI headers, and writes row-level plus summary reports.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nb
import numpy as np


DEFAULT_STATS_CSVS = [
    Path("runs/mrart_negative_control/full_runs/20260702_032906_mednext_both/merged/completion.csv"),
    Path("runs/mrart_negative_control/full_runs/20260702_233232_swin_unetr_both/merged/completion.csv"),
]

ROW_FIELDNAMES = [
    "source_csv",
    "row_index",
    "subject_id",
    "scan_id",
    "acquisition_id",
    "acquisition_type",
    "model_architecture",
    "training_condition",
    "fold",
    "reference_path",
    "reference_path_role",
    "path_exists",
    "header_spacing_x",
    "header_spacing_y",
    "header_spacing_z",
    "affine_spacing_x",
    "affine_spacing_y",
    "affine_spacing_z",
    "xyzt_units",
    "header_voxel_volume_mm3",
    "affine_voxel_volume_mm3_column_norm",
    "affine_voxel_volume_mm3_det",
    "stats_voxel_volume_mm3",
    "header_affine_det_relative_error",
    "header_affine_column_norm_relative_error",
    "column_norm_det_relative_error",
    "stats_header_relative_error",
    "status",
    "issues",
]

SUMMARY_FIELDNAMES = [
    "n_rows",
    "n_pass",
    "n_warn",
    "n_fail",
    "n_missing_reference_path",
    "n_missing_header_file",
    "n_unknown_units",
    "n_unexpected_units",
    "n_header_affine_det_mismatch",
    "n_header_affine_column_norm_mismatch",
    "n_column_norm_det_mismatch",
    "n_stats_header_mismatch",
    "max_stats_header_relative_error",
    "max_header_affine_det_relative_error",
    "max_header_affine_column_norm_relative_error",
    "row_csv",
]

WARNING_ISSUES = {"unknown_spatial_units"}
FAIL_ISSUES = {
    "missing_reference_path",
    "missing_header_file",
    "nonfinite_or_nonpositive_header_spacing",
    "nonfinite_or_nonpositive_affine_spacing",
    "unexpected_spatial_units",
    "affine_header_det_mismatch",
    "affine_header_column_norm_mismatch",
    "column_norm_det_mismatch",
    "missing_stats_voxel_volume",
    "stats_header_voxel_volume_mismatch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-csv", action="append", type=Path, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Row-level output CSV. Defaults to a timestamped server_reports path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Summary output CSV. Defaults to a timestamped server_reports path.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_output_paths() -> tuple[Path, Path]:
    stamp = utc_stamp()
    base = Path("scripts/supporting_experiments/mrart/server_reports")
    return (
        base / f"{stamp}_voxel_volume_verification_rows.csv",
        base / f"{stamp}_voxel_volume_verification_summary.csv",
    )


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


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def scan_id_from_row(row: dict[str, str]) -> str:
    if row.get("scan_id"):
        return row["scan_id"]
    subject_id = row.get("subject_id", "")
    acquisition = row.get("acquisition") or row.get("acquisition_id") or row.get("acquisition_type") or ""
    return f"{subject_id}_acq-{acquisition}" if subject_id and acquisition else ""


def acquisition_from_row(row: dict[str, str]) -> str:
    return row.get("acquisition") or row.get("acquisition_id") or row.get("acquisition_type") or ""


def architecture_from_row(row: dict[str, str]) -> str:
    return row.get("architecture") or row.get("model_architecture") or ""


def training_from_row(row: dict[str, str]) -> str:
    return row.get("training_condition", "")


def fold_from_row(row: dict[str, str]) -> str:
    return row.get("fold", "")


def voxel_volume_from_row(row: dict[str, str]) -> float | None:
    direct = parse_float(row.get("voxel_volume_mm3"))
    if direct is not None:
        return direct
    volume_ml = parse_float(row.get("predicted_lesion_volume_ml"))
    voxels = parse_float(row.get("predicted_lesion_voxel_count"))
    if volume_ml is not None and voxels not in {None, 0.0}:
        return volume_ml * 1000.0 / voxels
    return None


def candidate_path(row: dict[str, str]) -> tuple[str, str]:
    for key, role in [
        ("image_path", "source_mrart_t1w_image"),
        ("prediction_path", "prediction"),
        ("output_mask_path", "ensemble_mask"),
        ("output_prob_path", "ensemble_probability"),
    ]:
        value = row.get(key, "")
        if value:
            return value, role
    return "", ""


def relerr(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return abs(a - b) / abs(b)


def finite_positive(values: list[float]) -> bool:
    return all(math.isfinite(x) and x > 0 for x in values)


def row_status(issues: list[str]) -> str:
    issue_set = set(issues)
    if issue_set & FAIL_ISSUES:
        return "fail"
    if issue_set & WARNING_ISSUES:
        return "warn"
    return "pass"


def base_row(source: Path, index: int, row: dict[str, str], ref_path: str, role: str) -> dict[str, Any]:
    acquisition = acquisition_from_row(row)
    return {
        "source_csv": str(source),
        "row_index": index,
        "subject_id": row.get("subject_id", ""),
        "scan_id": scan_id_from_row(row),
        "acquisition_id": acquisition,
        "acquisition_type": acquisition,
        "model_architecture": architecture_from_row(row),
        "training_condition": training_from_row(row),
        "fold": fold_from_row(row),
        "reference_path": ref_path,
        "reference_path_role": role,
        "path_exists": int(bool(ref_path and Path(ref_path).exists())),
        "stats_voxel_volume_mm3": voxel_volume_from_row(row) or "",
    }


def verify_row(source: Path, index: int, row: dict[str, str], tolerance: float) -> dict[str, Any]:
    ref_path, role = candidate_path(row)
    base = base_row(source, index, row, ref_path, role)
    issues: list[str] = []
    if not ref_path:
        issues.append("missing_reference_path")
        return {**base, "status": row_status(issues), "issues": ";".join(issues)}
    if not Path(ref_path).exists():
        issues.append("missing_header_file")
        return {**base, "status": row_status(issues), "issues": ";".join(issues)}

    try:
        image = nb.load(ref_path)
        header_spacing = [float(x) for x in image.header.get_zooms()[:3]]
        affine = np.asarray(image.affine, dtype=float)
        spatial_affine = affine[:3, :3]
        affine_spacing = [float(np.linalg.norm(spatial_affine[:, i])) for i in range(3)]
        units = image.header.get_xyzt_units()
        xyzt_units = ",".join(str(x) for x in units)
        header_volume = float(np.prod(header_spacing))
        affine_volume_column_norm = float(np.prod(affine_spacing))
        affine_volume_det = float(abs(np.linalg.det(spatial_affine)))
        stats_volume = voxel_volume_from_row(row)
        header_affine_det_err = relerr(header_volume, affine_volume_det)
        header_affine_column_err = relerr(header_volume, affine_volume_column_norm)
        column_norm_det_err = relerr(affine_volume_column_norm, affine_volume_det)
        stats_header_err = relerr(stats_volume, header_volume)

        if not finite_positive(header_spacing):
            issues.append("nonfinite_or_nonpositive_header_spacing")
        if not finite_positive(affine_spacing):
            issues.append("nonfinite_or_nonpositive_affine_spacing")
        if units[0] == "unknown":
            issues.append("unknown_spatial_units")
        elif units[0] != "mm":
            issues.append("unexpected_spatial_units")
        if header_affine_det_err is not None and header_affine_det_err > tolerance:
            issues.append("affine_header_det_mismatch")
        if header_affine_column_err is not None and header_affine_column_err > tolerance:
            issues.append("affine_header_column_norm_mismatch")
        if column_norm_det_err is not None and column_norm_det_err > tolerance:
            issues.append("column_norm_det_mismatch")
        if stats_volume is None:
            issues.append("missing_stats_voxel_volume")
        elif stats_header_err is not None and stats_header_err > tolerance:
            issues.append("stats_header_voxel_volume_mismatch")

        return {
            **base,
            "header_spacing_x": header_spacing[0],
            "header_spacing_y": header_spacing[1],
            "header_spacing_z": header_spacing[2],
            "affine_spacing_x": affine_spacing[0],
            "affine_spacing_y": affine_spacing[1],
            "affine_spacing_z": affine_spacing[2],
            "xyzt_units": xyzt_units,
            "header_voxel_volume_mm3": header_volume,
            "affine_voxel_volume_mm3_column_norm": affine_volume_column_norm,
            "affine_voxel_volume_mm3_det": affine_volume_det,
            "stats_voxel_volume_mm3": stats_volume or "",
            "header_affine_det_relative_error": header_affine_det_err if header_affine_det_err is not None else "",
            "header_affine_column_norm_relative_error": header_affine_column_err if header_affine_column_err is not None else "",
            "column_norm_det_relative_error": column_norm_det_err if column_norm_det_err is not None else "",
            "stats_header_relative_error": stats_header_err if stats_header_err is not None else "",
            "status": row_status(issues),
            "issues": ";".join(issues),
        }
    except Exception as exc:
        issues.append(f"header_read_error:{type(exc).__name__}")
        return {**base, "status": "fail", "issues": ";".join(issues)}


def max_float(rows: list[dict[str, Any]], key: str) -> str:
    values = []
    for row in rows:
        value = parse_float(str(row.get(key, "")))
        if value is not None:
            values.append(value)
    return f"{max(values):.10g}" if values else ""


def build_summary(rows: list[dict[str, Any]], row_csv: Path) -> list[dict[str, Any]]:
    statuses = Counter(row["status"] for row in rows)
    issue_counts = Counter()
    for row in rows:
        for issue in str(row.get("issues", "")).split(";"):
            if issue:
                issue_counts[issue] += 1
    return [
        {
            "n_rows": len(rows),
            "n_pass": statuses["pass"],
            "n_warn": statuses["warn"],
            "n_fail": statuses["fail"],
            "n_missing_reference_path": issue_counts["missing_reference_path"],
            "n_missing_header_file": issue_counts["missing_header_file"],
            "n_unknown_units": issue_counts["unknown_spatial_units"],
            "n_unexpected_units": issue_counts["unexpected_spatial_units"],
            "n_header_affine_det_mismatch": issue_counts["affine_header_det_mismatch"],
            "n_header_affine_column_norm_mismatch": issue_counts["affine_header_column_norm_mismatch"],
            "n_column_norm_det_mismatch": issue_counts["column_norm_det_mismatch"],
            "n_stats_header_mismatch": issue_counts["stats_header_voxel_volume_mismatch"],
            "max_stats_header_relative_error": max_float(rows, "stats_header_relative_error"),
            "max_header_affine_det_relative_error": max_float(rows, "header_affine_det_relative_error"),
            "max_header_affine_column_norm_relative_error": max_float(rows, "header_affine_column_norm_relative_error"),
            "row_csv": str(row_csv),
        }
    ]


def main() -> None:
    args = parse_args()
    stats_paths = args.stats_csv or DEFAULT_STATS_CSVS
    default_row_csv, default_summary_csv = default_output_paths()
    row_csv = args.output_csv or default_row_csv
    summary_csv = args.summary_csv or default_summary_csv
    for path in [row_csv, summary_csv]:
        ensure_new_path(path)
    check_not_ignored([row_csv, summary_csv])

    out_rows: list[dict[str, Any]] = []
    for stats_path in stats_paths:
        if not stats_path.exists():
            raise FileNotFoundError(stats_path)
        for index, row in enumerate(read_csv(stats_path), start=1):
            out_rows.append(verify_row(stats_path, index, row, args.relative_tolerance))

    summary_rows = build_summary(out_rows, row_csv)
    write_csv(row_csv, ROW_FIELDNAMES, out_rows)
    write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)
    print(f"Wrote {row_csv} rows={len(out_rows)} cols={len(ROW_FIELDNAMES)}")
    print(f"Wrote {summary_csv} rows={len(summary_rows)} cols={len(SUMMARY_FIELDNAMES)}")


if __name__ == "__main__":
    main()
