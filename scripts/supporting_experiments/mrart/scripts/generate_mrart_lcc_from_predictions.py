#!/usr/bin/env python3
"""Compute MR-ART LCC stats from existing saved prediction masks/probabilities.

Recommended current use is ``--scope ensemble`` because the verified full MR-ART
runs saved ensemble masks/probability maps but did not save fold masks/probs.
Fold-level LCC should come from run_mrart_fold_lcc_stats_no_outputs.py, which
recomputes fold predictions in memory and writes CSV statistics only.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nb
import numpy as np
from scipy import ndimage
from tqdm import tqdm


DEFAULT_COMPLETION_CSVS = [
    Path("runs/mrart_negative_control/full_runs/20260702_032906_mednext_both/merged/completion.csv"),
    Path("runs/mrart_negative_control/full_runs/20260702_233232_swin_unetr_both/merged/completion.csv"),
]

ROW_FIELDNAMES = [
    "subject_id",
    "scan_id",
    "acquisition_id",
    "acquisition_type",
    "model_architecture",
    "training_condition",
    "prediction_scope",
    "fold",
    "prediction_path",
    "prediction_kind",
    "threshold",
    "voxel_volume_mm3_header",
    "voxel_volume_mm3_stats_csv",
    "voxel_volume_relative_error",
    "predicted_lesion_voxel_count",
    "predicted_lesion_volume_mm3",
    "predicted_lesion_volume_ml",
    "recomputed_predicted_lesion_voxel_count",
    "completion_predicted_lesion_voxel_count",
    "voxel_count_abs_error",
    "ensemble_voxel_count_abs_error",
    "voxel_count_relative_error",
    "recomputed_predicted_lesion_volume_mm3",
    "recomputed_predicted_lesion_volume_ml",
    "completion_predicted_lesion_volume_ml",
    "volume_abs_error_ml",
    "ensemble_volume_ml_abs_error",
    "volume_relative_error",
    "ensemble_volume_relative_error",
    "voxel_count_match",
    "volume_match",
    "scan_positive",
    "ensemble_largest_cc_voxel_count",
    "ensemble_largest_cc_volume_mm3",
    "ensemble_largest_cc_volume_ml",
    "ensemble_lcc_status",
    "status",
    "issue",
    "source_completion_csv",
]

SUMMARY_FIELDNAMES = [
    "scope",
    "n_rows",
    "n_success",
    "n_missing_prediction",
    "n_error",
    "n_voxel_count_mismatch",
    "n_volume_mismatch",
    "max_voxel_count_abs_error",
    "max_volume_relative_error",
    "row_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-csv", action="append", type=Path, default=None)
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scope", choices=["fold", "ensemble", "both"], default="ensemble")
    parser.add_argument("--volume-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--voxel-count-tolerance", type=int, default=0)
    parser.add_argument(
        "--infer-fold-paths",
        action="store_true",
        help="For diagnostic use only. Current full runs have blank fold paths and no saved fold NIfTIs.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_output_paths() -> tuple[Path, Path]:
    stamp = utc_stamp()
    base = Path("scripts/supporting_experiments/mrart/server_reports")
    return (
        base / f"{stamp}_ensemble_lcc_rows.csv",
        base / f"{stamp}_ensemble_lcc_summary.csv",
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


def scan_id(subject_id: str, acquisition: str) -> str:
    return f"{subject_id}_acq-{acquisition}"


def parse_float(value: str | None) -> float | None:
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


def folds_from_row(row: dict[str, str]) -> list[int]:
    values = []
    for part in row.get("folds_used", "").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def relative_error(observed: float | None, expected: float | None) -> float | None:
    if observed is None or expected is None or expected == 0:
        return None
    return abs(observed - expected) / abs(expected)


def load_mask(path: Path, threshold: float, kind: str) -> tuple[np.ndarray, float]:
    image = nb.load(str(path))
    data = np.asanyarray(image.dataobj)
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    if kind == "prob":
        mask = data >= threshold
    else:
        mask = data > 0
    return np.asarray(mask, dtype=bool), voxel_volume_mm3


def lcc_stats(mask: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    predicted_voxels = int(mask.sum())
    if predicted_voxels == 0:
        largest = 0
    else:
        labels, num = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        if num == 0:
            largest = 0
        else:
            counts = np.bincount(labels.ravel())
            largest = int(counts[1:].max()) if counts.size > 1 else 0
    return {
        "recomputed_predicted_lesion_voxel_count": predicted_voxels,
        "recomputed_predicted_lesion_volume_mm3": predicted_voxels * voxel_volume_mm3,
        "recomputed_predicted_lesion_volume_ml": predicted_voxels * voxel_volume_mm3 / 1000.0,
        "scan_positive": int(predicted_voxels > 0),
        "ensemble_largest_cc_voxel_count": largest,
        "ensemble_largest_cc_volume_mm3": largest * voxel_volume_mm3,
        "ensemble_largest_cc_volume_ml": largest * voxel_volume_mm3 / 1000.0,
    }


def infer_paths_from_ensemble(row: dict[str, str], fold: int) -> tuple[str, str]:
    ensemble_mask = row.get("output_mask_path", "")
    if not ensemble_mask:
        return "", ""
    base = Path(ensemble_mask)
    name = base.name.replace("_ensemble_mask.nii.gz", "")
    if name == base.name:
        return "", ""
    return str(base.parent / f"{name}_fold-{fold}_mask.nii.gz"), str(base.parent / f"{name}_fold-{fold}_prob.nii.gz")


def prediction_candidates(row: dict[str, str], scope: str, fold: int | str, infer_fold_paths: bool) -> list[tuple[str, str]]:
    if scope == "ensemble":
        return [
            (row.get("output_mask_path", ""), "mask"),
            (row.get("output_prob_path", ""), "prob"),
        ]
    mask_path = row.get(f"fold{fold}_mask_path", "")
    prob_path = row.get(f"fold{fold}_prob_path", "")
    if not mask_path and not prob_path:
        print(
            f"Warning: blank fold prediction paths for {row.get('subject_id')} {row.get('acquisition')} "
            f"{row.get('architecture')} {row.get('training_condition')} fold={fold}; expected for current full runs.",
            file=sys.stderr,
        )
    if infer_fold_paths and not mask_path and not prob_path:
        mask_path, prob_path = infer_paths_from_ensemble(row, int(fold))
    return [(mask_path, "mask"), (prob_path, "prob")]


def empty_result(row: dict[str, str], scope: str, fold: int | str, source: Path, status: str, issue: str) -> dict[str, Any]:
    acquisition = row.get("acquisition", "")
    subject_id = row.get("subject_id", "")
    return {
        "subject_id": subject_id,
        "scan_id": scan_id(subject_id, acquisition),
        "acquisition_id": acquisition,
        "acquisition_type": acquisition,
        "model_architecture": row.get("architecture", ""),
        "training_condition": row.get("training_condition", ""),
        "prediction_scope": scope,
        "fold": fold,
        "status": status,
        "issue": issue,
        "ensemble_lcc_status": status if scope == "ensemble" else "",
        "voxel_volume_mm3_stats_csv": row.get("voxel_volume_mm3", ""),
        "source_completion_csv": str(source),
    }


def completion_expected(row: dict[str, str], scope: str, fold: int | str) -> tuple[int | None, float | None]:
    if scope == "ensemble":
        return (
            parse_int(row.get("ensemble_predicted_voxels_at_0_5")),
            parse_float(row.get("ensemble_predicted_volume_ml_at_0_5")),
        )
    return (
        parse_int(row.get(f"fold{fold}_predicted_voxels_at_0_5")),
        parse_float(row.get(f"fold{fold}_predicted_volume_ml_at_0_5")),
    )


def compute_one(
    row: dict[str, str],
    scope: str,
    fold: int | str,
    source: Path,
    threshold: float,
    infer_fold_paths: bool,
    voxel_count_tolerance: int,
    volume_relative_tolerance: float,
) -> dict[str, Any]:
    candidates = prediction_candidates(row, scope, fold, infer_fold_paths)
    chosen_path = ""
    chosen_kind = ""
    for path_str, kind in candidates:
        if path_str and Path(path_str).exists():
            chosen_path = path_str
            chosen_kind = kind
            break
    if not chosen_path:
        issue = "fold_paths_blank_or_missing_prediction" if scope == "fold" else "missing_prediction"
        return empty_result(row, scope, fold, source, "missing_prediction", issue)

    stats_volume = parse_float(row.get("voxel_volume_mm3"))
    try:
        mask, header_volume = load_mask(Path(chosen_path), threshold, chosen_kind)
        stats = lcc_stats(mask, header_volume)
        expected_voxels, expected_volume_ml = completion_expected(row, scope, fold)
        recomputed_voxels = int(stats["recomputed_predicted_lesion_voxel_count"])
        recomputed_volume_ml = float(stats["recomputed_predicted_lesion_volume_ml"])
        voxel_abs_error = abs(recomputed_voxels - expected_voxels) if expected_voxels is not None else None
        volume_abs_error = abs(recomputed_volume_ml - expected_volume_ml) if expected_volume_ml is not None else None
        voxel_rel_error = relative_error(float(recomputed_voxels), float(expected_voxels) if expected_voxels is not None else None)
        volume_rel_error = relative_error(recomputed_volume_ml, expected_volume_ml)
        voxel_count_match = (
            int(voxel_abs_error <= voxel_count_tolerance)
            if voxel_abs_error is not None
            else ""
        )
        volume_match = (
            int(volume_rel_error <= volume_relative_tolerance)
            if volume_rel_error is not None
            else ""
        )
        issues = []
        if voxel_abs_error is not None and voxel_abs_error > voxel_count_tolerance:
            issues.append("voxel_count_mismatch")
        if volume_rel_error is not None and volume_rel_error > volume_relative_tolerance:
            issues.append("volume_mismatch")
        status = "success" if not issues else "success_with_mismatch"
        result = empty_result(row, scope, fold, source, status, ";".join(issues))
        result.update(
            {
                "prediction_path": chosen_path,
                "prediction_kind": chosen_kind,
                "threshold": threshold if chosen_kind == "prob" else "",
                "voxel_volume_mm3_header": header_volume,
                "voxel_volume_relative_error": relative_error(header_volume, stats_volume),
                "predicted_lesion_voxel_count": recomputed_voxels,
                "predicted_lesion_volume_mm3": stats["recomputed_predicted_lesion_volume_mm3"],
                "predicted_lesion_volume_ml": recomputed_volume_ml,
                "completion_predicted_lesion_voxel_count": expected_voxels if expected_voxels is not None else "",
                "voxel_count_abs_error": voxel_abs_error if voxel_abs_error is not None else "",
                "ensemble_voxel_count_abs_error": voxel_abs_error if voxel_abs_error is not None else "",
                "voxel_count_relative_error": voxel_rel_error if voxel_rel_error is not None else "",
                "completion_predicted_lesion_volume_ml": expected_volume_ml if expected_volume_ml is not None else "",
                "volume_abs_error_ml": volume_abs_error if volume_abs_error is not None else "",
                "ensemble_volume_ml_abs_error": volume_abs_error if volume_abs_error is not None else "",
                "volume_relative_error": volume_rel_error if volume_rel_error is not None else "",
                "ensemble_volume_relative_error": volume_rel_error if volume_rel_error is not None else "",
                "voxel_count_match": voxel_count_match,
                "volume_match": volume_match,
                "ensemble_lcc_status": "success" if scope == "ensemble" and not issues else ("success_with_mismatch" if scope == "ensemble" else ""),
                **stats,
            }
        )
        return result
    except Exception as exc:
        result = empty_result(row, scope, fold, source, "error", f"{type(exc).__name__}: {exc}")
        result["prediction_path"] = chosen_path
        result["prediction_kind"] = chosen_kind
        result["ensemble_lcc_status"] = "error" if scope == "ensemble" else ""
        return result


def build_summary(rows: list[dict[str, Any]], scope: str, row_csv: Path) -> list[dict[str, Any]]:
    status_counts = defaultdict(int)
    voxel_mismatch = 0
    volume_mismatch = 0
    max_voxel_error = 0
    max_volume_rel = 0.0
    for row in rows:
        status = row.get("status", "")
        status_counts[status] += 1
        issue = row.get("issue", "")
        if "voxel_count_mismatch" in issue:
            voxel_mismatch += 1
        if "volume_mismatch" in issue:
            volume_mismatch += 1
        voxel_error = parse_int(row.get("voxel_count_abs_error"))
        volume_rel = parse_float(row.get("volume_relative_error"))
        if voxel_error is not None:
            max_voxel_error = max(max_voxel_error, voxel_error)
        if volume_rel is not None:
            max_volume_rel = max(max_volume_rel, volume_rel)
    return [
        {
            "scope": scope,
            "n_rows": len(rows),
            "n_success": status_counts["success"] + status_counts["success_with_mismatch"],
            "n_missing_prediction": status_counts["missing_prediction"],
            "n_error": status_counts["error"],
            "n_voxel_count_mismatch": voxel_mismatch,
            "n_volume_mismatch": volume_mismatch,
            "max_voxel_count_abs_error": max_voxel_error,
            "max_volume_relative_error": max_volume_rel,
            "row_csv": str(row_csv),
        }
    ]


def main() -> None:
    args = parse_args()
    completion_paths = args.completion_csv or DEFAULT_COMPLETION_CSVS
    default_row_csv, default_summary_csv = default_output_paths()
    row_csv = args.output_csv or default_row_csv
    summary_csv = args.summary_csv or default_summary_csv
    if row_csv.exists() and not summary_csv.exists():
        check_not_ignored([summary_csv])
        out_rows = read_csv(row_csv)
        summary_rows = build_summary(out_rows, args.scope, row_csv)
        write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)
        print(f"Reused existing {row_csv} rows={len(out_rows)}")
        print(f"Wrote {summary_csv} rows={len(summary_rows)} cols={len(SUMMARY_FIELDNAMES)}")
        return
    for path in [row_csv, summary_csv]:
        ensure_new_path(path)
    check_not_ignored([row_csv, summary_csv])

    out_rows: list[dict[str, Any]] = []
    for completion_path in completion_paths:
        if not completion_path.exists():
            raise FileNotFoundError(completion_path)
        completion_rows = read_csv(completion_path)
        for row in tqdm(
            completion_rows,
            total=len(completion_rows),
            desc=f"ensemble LCC {completion_path.parent.parent.name}",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            if row.get("status") and row["status"] != "success":
                continue
            if args.scope in {"ensemble", "both"}:
                out_rows.append(
                    compute_one(
                        row,
                        "ensemble",
                        "ensemble",
                        completion_path,
                        args.threshold,
                        args.infer_fold_paths,
                        args.voxel_count_tolerance,
                        args.volume_relative_tolerance,
                    )
                )
            if args.scope in {"fold", "both"}:
                for fold in folds_from_row(row):
                    out_rows.append(
                        compute_one(
                            row,
                            "fold",
                            fold,
                            completion_path,
                            args.threshold,
                            args.infer_fold_paths,
                            args.voxel_count_tolerance,
                            args.volume_relative_tolerance,
                        )
                    )

    write_csv(row_csv, ROW_FIELDNAMES, out_rows)
    summary_rows = build_summary(out_rows, args.scope, row_csv)
    write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)
    print(f"Wrote {row_csv} rows={len(out_rows)} cols={len(ROW_FIELDNAMES)}")
    print(f"Wrote {summary_csv} rows={len(summary_rows)} cols={len(SUMMARY_FIELDNAMES)}")


if __name__ == "__main__":
    main()
