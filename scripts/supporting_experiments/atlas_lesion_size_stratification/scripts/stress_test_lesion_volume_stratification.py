#!/usr/bin/env python3
"""
Stress-test lesion-volume stratification schemes for later 5-fold CV.

Command:
    python scripts/supporting_experiments/atlas_lesion_size_stratification/scripts/stress_test_lesion_volume_stratification.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np

from analyze_lesion_volume_stratification import (
    HARD_MIN_PER_BIN,
    PREFERRED_MIN_PER_BIN,
    analyze_subjects,
    assign_bins,
    basic_summary,
    dedupe_preserve_order,
    format_edges,
    format_number,
    half_open_range_text,
    load_lesion_volume,
    load_subject_entries,
    log_edges,
    percentile_summary,
    quantile_edges,
    resolve_paths,
)

OUTER_SPLITS = 5
INNER_VAL_SPLITS = 5
DEFAULT_SEED = 9001
OUTER_PERCENTILES = [10, 25, 50, 75, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test lesion-volume stratification schemes for later 5-fold CV."
    )
    parser.add_argument(
        "--subjects_csv",
        default=(
            "outputs/supporting_experiments/atlas_lesion_size_stratification/"
            "stratification_analysis/lesion_volume_subjects.csv"
        ),
        help="Existing lesion-volume subject CSV from the analysis step.",
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "outputs/supporting_experiments/"
            "atlas_lesion_size_stratification/stratification_analysis"
        ),
        help="Directory for local Markdown/JSON working outputs.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Dataset root containing train/ and test/. Used only if the subject CSV is missing.",
    )
    parser.add_argument(
        "--gt_deriv",
        default=None,
        help="GT derivative root. Used only if the subject CSV is missing.",
    )
    parser.add_argument(
        "--splits_json",
        default="splits/atlas_5fold_lesion_quartile_excluding_known_issues.json",
        help="Split JSON. Defaults to the paper CV master and is used only if the subject CSV is missing.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Binary lesion threshold. Must match the analysis script convention.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic RNG seed for fold simulation.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_subject_csv(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def resolve_out_dir(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def parse_int_or_none(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(float(raw))


def parse_float_or_none(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def parse_bool(raw: str | None) -> bool:
    return str(raw).strip().lower() == "true"


def load_subject_rows_from_csv(subjects_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with subjects_csv.open() as f:
        reader = csv.DictReader(f)
        required = {
            "subject_id",
            "split_source",
            "split_key_used",
            "mask_path",
            "lesion_voxels",
            "lesion_mm3",
            "voxel_volume_mm3",
            "spacing_mm",
            "used_fallback_mask_path",
            "status",
            "error",
        }
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise SystemExit(
                f"ERROR: subject CSV is missing required columns: {missing}. "
                "Rerun the lesion-volume analysis script in this directory first."
            )
        for row in reader:
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "split_source": row["split_source"],
                    "split_key_used": row["split_key_used"],
                    "mask_path": row["mask_path"],
                    "lesion_voxels": parse_int_or_none(row["lesion_voxels"]),
                    "lesion_mm3": parse_float_or_none(row["lesion_mm3"]),
                    "voxel_volume_mm3": parse_float_or_none(row["voxel_volume_mm3"]),
                    "spacing_mm": row["spacing_mm"] or None,
                    "used_fallback_mask_path": parse_bool(row["used_fallback_mask_path"]),
                    "status": row["status"],
                    "error": row["error"] or None,
                }
            )
    return sorted(rows, key=lambda item: item["subject_id"])


def recompute_subject_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path_args = argparse.Namespace(
        data_root=args.data_root,
        gt_deriv=args.gt_deriv,
        splits_json=args.splits_json,
        out_dir=args.out_dir,
    )
    _repo_root, gt_deriv, splits_json, _out_dir = resolve_paths(path_args)
    entries, split_warnings, totals_by_split = load_subject_entries(splits_json)
    records, analysis_warnings = analyze_subjects(gt_deriv, entries, args.mask_threshold)
    rows = []
    for record in sorted(records, key=lambda item: item.subject_id):
        rows.append(
            {
                "subject_id": record.subject_id,
                "split_source": record.split_source,
                "split_key_used": record.split_key_used,
                "mask_path": record.mask_path,
                "lesion_voxels": record.lesion_voxels,
                "lesion_mm3": record.lesion_mm3,
                "voxel_volume_mm3": record.voxel_volume_mm3,
                "spacing_mm": record.spacing_mm,
                "used_fallback_mask_path": record.used_fallback_mask_path,
                "status": record.status,
                "error": record.error,
            }
        )
    return rows, {
        "split_warnings": split_warnings,
        "analysis_warnings": analysis_warnings,
        "subjects_per_split_from_split_file": totals_by_split,
        "gt_deriv": str(gt_deriv),
        "splits_json": str(splits_json),
    }


def load_or_recompute_subject_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subjects_csv = resolve_subject_csv(args.subjects_csv)
    if subjects_csv.exists():
        return load_subject_rows_from_csv(subjects_csv), {
            "source": str(subjects_csv),
            "source_kind": "csv",
            "split_warnings": [],
            "analysis_warnings": [],
        }
    rows, meta = recompute_subject_rows(args)
    meta["source"] = "recomputed_from_masks"
    meta["source_kind"] = "recomputed"
    return rows, meta


def percentile_subset(values: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def stats_block(values: np.ndarray) -> dict[str, Any]:
    return {
        **basic_summary(values.astype(np.float64)),
        "percentiles": percentile_subset(values.astype(np.float64), OUTER_PERCENTILES),
    }


def bin_counts(labels: np.ndarray, num_bins: int) -> list[int]:
    counts = [0] * num_bins
    uniques, unique_counts = np.unique(labels, return_counts=True)
    for label, count in zip(uniques.tolist(), unique_counts.tolist()):
        counts[int(label)] = int(count)
    return counts


def proportions_from_counts(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def stratified_fold_indices(labels: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(class_indices)
        chunks = np.array_split(shuffled, n_splits)
        for fold_idx, chunk in enumerate(chunks):
            folds[fold_idx].extend(int(i) for i in chunk.tolist())
    return [np.asarray(sorted(fold), dtype=np.int32) for fold in folds]


def complement_indices(total_size: int, held_out: np.ndarray) -> np.ndarray:
    mask = np.ones(total_size, dtype=bool)
    mask[held_out] = False
    return np.flatnonzero(mask)


def ks_statistic_against_full(fold_values: np.ndarray, full_values: np.ndarray) -> float | None:
    try:
        from scipy.stats import ks_2samp  # type: ignore
    except Exception:
        return None
    return float(ks_2samp(fold_values, full_values).statistic)


def max_abs_bin_prop_deviation(
    fold_bin_props: list[list[float]],
    full_bin_props: list[float],
) -> float:
    deviation = 0.0
    for fold_props in fold_bin_props:
        for fold_prop, full_prop in zip(fold_props, full_bin_props):
            deviation = max(deviation, abs(fold_prop - full_prop))
    return float(deviation)


def scheme_type_from_name(name: str) -> str:
    if name.startswith("quantile"):
        return "quantile"
    if name.startswith("log"):
        return "log"
    if name.startswith("interpret_ml"):
        return "interpretability"
    return "other"


def scheme_preference_bonus(name: str, scheme_type: str, requested_bins: int) -> int:
    bonus = 0
    if scheme_type == "quantile":
        bonus += 4
    elif scheme_type == "interpretability":
        bonus += 5

    if requested_bins == 4:
        bonus += 10
    elif requested_bins == 5:
        bonus += 8
    elif requested_bins == 6:
        bonus += 5
    elif requested_bins == 3:
        bonus += 2
    elif requested_bins > 6:
        bonus -= 4

    if name.startswith("log"):
        bonus -= 4
    return bonus


def build_candidate_schemes(values: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for k in [3, 4, 5, 6, 8]:
        edges, warnings = quantile_edges(values, k)
        candidates.append(
            {
                "name": f"quantile_k{k}",
                "type": "quantile",
                "requested_bins": k,
                "edges": edges,
                "warnings": warnings,
                "description": f"Lesion-volume quantile stratification with {k} bins.",
            }
        )

    for k in [3, 4, 5]:
        edges, warnings = log_edges(values, k)
        candidates.append(
            {
                "name": f"log_k{k}",
                "type": "log",
                "requested_bins": k,
                "edges": edges,
                "warnings": warnings,
                "description": f"Log-volume binning with {k} bins.",
            }
        )

    interpret_edges = [float(np.min(values)), 1000.0, 10000.0, 50000.0, float(np.max(values))]
    interpret_edges = [float(x) for x in np.unique(np.asarray(interpret_edges, dtype=np.float64))]
    interpret_warnings: list[str] = []
    if len(interpret_edges) < 5:
        interpret_warnings.append(
            "Interpretable mL thresholds collapsed because one or more boundaries were outside the observed range."
        )
    candidates.append(
        {
            "name": "interpret_ml_4bin_lt1_1to10_10to50_ge50",
            "type": "interpretability",
            "requested_bins": 4,
            "edges": interpret_edges,
            "warnings": interpret_warnings,
            "description": "Exploratory interpretable bins: <1 mL, 1-10 mL, 10-50 mL, >=50 mL.",
        }
    )

    return candidates


def evaluate_candidate_scheme(
    scheme: dict[str, Any],
    values: np.ndarray,
    subject_ids: list[str],
    seed: int,
) -> dict[str, Any]:
    warnings = list(scheme["warnings"])
    edges = [float(edge) for edge in scheme["edges"]]
    if len(edges) < 2:
        return {
            **scheme,
            "effective_bins": 0,
            "bin_counts": [],
            "bin_ranges": [],
            "overall_hard_min_ok": False,
            "overall_preferred_min_ok": False,
            "outer_supported": False,
            "inner_supported": False,
            "outer_folds": [],
            "outer_summary": None,
            "inner_summary": None,
            "label": "bad",
            "recommendation_score": -999.0,
            "warnings": warnings + ["Scheme has fewer than two edges and is unusable."],
        }

    labels = assign_bins(values, edges)
    num_bins = len(edges) - 1
    counts = bin_counts(labels, num_bins)
    bin_ranges = [half_open_range_text(i, edges) for i in range(num_bins)]
    overall_hard_min_ok = all(count >= HARD_MIN_PER_BIN for count in counts)
    overall_preferred_min_ok = all(count >= PREFERRED_MIN_PER_BIN for count in counts)
    if not overall_hard_min_ok:
        warnings.append("At least one bin has fewer than 5 subjects, so some outer folds will miss that stratum.")
    if not overall_preferred_min_ok:
        warnings.append("At least one bin has fewer than 25 subjects, so inner val strata may become thin.")

    full_stats = stats_block(values)
    full_bin_props = proportions_from_counts(counts)

    outer_fold_indices = stratified_fold_indices(labels, OUTER_SPLITS, seed)
    outer_folds: list[dict[str, Any]] = []
    fold_means: list[float] = []
    fold_medians: list[float] = []
    fold_bin_props: list[list[float]] = []
    fold_ks_stats: list[float] = []
    outer_min_test_bin_count = math.inf
    outer_zero_bin_events = 0

    inner_val_sizes: list[int] = []
    inner_min_val_bin_count = math.inf
    inner_val_bin_prop_deviations: list[float] = []
    inner_val_mean_values: list[float] = []
    inner_val_median_values: list[float] = []
    inner_any_bin_below_5 = 0
    inner_any_bin_below_25 = 0
    inner_details: list[dict[str, Any]] = []

    for outer_idx, test_idx in enumerate(outer_fold_indices):
        fold_values = values[test_idx]
        fold_labels = labels[test_idx]
        fold_counts = bin_counts(fold_labels, num_bins)
        fold_props = proportions_from_counts(fold_counts)
        fold_stats = stats_block(fold_values)
        fold_means.append(float(fold_stats["mean"]))
        fold_medians.append(float(fold_stats["median"]))
        fold_bin_props.append(fold_props)
        outer_min_test_bin_count = min(outer_min_test_bin_count, min(fold_counts))
        outer_zero_bin_events += sum(count == 0 for count in fold_counts)

        ks_stat = ks_statistic_against_full(fold_values.astype(np.float64), values.astype(np.float64))
        if ks_stat is not None:
            fold_ks_stats.append(ks_stat)

        outer_train_idx = complement_indices(values.shape[0], test_idx)
        outer_train_labels = labels[outer_train_idx]
        inner_fold_indices = stratified_fold_indices(outer_train_labels, INNER_VAL_SPLITS, seed + 1000 + outer_idx)
        val_local_idx = inner_fold_indices[0]
        val_idx = outer_train_idx[val_local_idx]
        val_values = values[val_idx]
        val_labels = labels[val_idx]
        val_counts = bin_counts(val_labels, num_bins)
        val_props = proportions_from_counts(val_counts)
        val_stats = stats_block(val_values)
        inner_val_sizes.append(int(val_idx.size))
        inner_min_val_bin_count = min(inner_min_val_bin_count, min(val_counts))
        inner_val_bin_prop_deviations.append(
            max(abs(a - b) for a, b in zip(val_props, full_bin_props))
        )
        inner_val_mean_values.append(float(val_stats["mean"]))
        inner_val_median_values.append(float(val_stats["median"]))
        if any(count < HARD_MIN_PER_BIN for count in val_counts):
            inner_any_bin_below_5 += 1
        if any(count < PREFERRED_MIN_PER_BIN for count in val_counts):
            inner_any_bin_below_25 += 1

        inner_details.append(
            {
                "outer_fold": outer_idx + 1,
                "train_size": int(outer_train_idx.size),
                "val_size": int(val_idx.size),
                "val_bin_counts": val_counts,
                "val_bin_proportions": val_props,
                "val_stats": val_stats,
                "val_max_abs_bin_prop_deviation_from_full": inner_val_bin_prop_deviations[-1],
                "any_bin_below_5": any(count < HARD_MIN_PER_BIN for count in val_counts),
                "any_bin_below_25": any(count < PREFERRED_MIN_PER_BIN for count in val_counts),
            }
        )

        outer_folds.append(
            {
                "fold": outer_idx + 1,
                "test_subject_count": int(test_idx.size),
                "test_subject_ids": [subject_ids[i] for i in test_idx.tolist()],
                "test_bin_counts": fold_counts,
                "test_bin_proportions": fold_props,
                "lesion_voxels_stats": fold_stats,
                "ks_distance_vs_full": ks_stat,
                "inner_train_val": inner_details[-1],
            }
        )

    outer_supported = overall_hard_min_ok
    inner_supported = inner_min_val_bin_count >= HARD_MIN_PER_BIN

    outer_summary = {
        "full_dataset_stats": full_stats,
        "full_bin_counts": counts,
        "full_bin_proportions": full_bin_props,
        "std_fold_mean_lesion_voxels": float(np.std(np.asarray(fold_means), ddof=0)),
        "std_fold_median_lesion_voxels": float(np.std(np.asarray(fold_medians), ddof=0)),
        "max_min_ratio_fold_mean_lesion_voxels": float(max(fold_means) / min(fold_means)) if min(fold_means) > 0 else None,
        "max_abs_bin_prop_deviation_from_full": max_abs_bin_prop_deviation(fold_bin_props, full_bin_props),
        "outer_min_test_bin_count": int(outer_min_test_bin_count),
        "outer_zero_bin_events": int(outer_zero_bin_events),
        "ks_distance_mean": float(np.mean(np.asarray(fold_ks_stats))) if fold_ks_stats else None,
        "ks_distance_max": float(np.max(np.asarray(fold_ks_stats))) if fold_ks_stats else None,
    }

    inner_summary = {
        "val_size_mean": float(np.mean(np.asarray(inner_val_sizes))),
        "val_size_min": int(min(inner_val_sizes)),
        "val_size_max": int(max(inner_val_sizes)),
        "inner_min_val_bin_count": int(inner_min_val_bin_count),
        "inner_any_bin_below_5_count": int(inner_any_bin_below_5),
        "inner_any_bin_below_25_count": int(inner_any_bin_below_25),
        "std_val_mean_lesion_voxels": float(np.std(np.asarray(inner_val_mean_values), ddof=0)),
        "std_val_median_lesion_voxels": float(np.std(np.asarray(inner_val_median_values), ddof=0)),
        "max_abs_val_bin_prop_deviation_from_full": float(max(inner_val_bin_prop_deviations)),
        "details": inner_details,
    }

    score = 0.0
    if overall_hard_min_ok:
        score += 50.0
    else:
        score -= 150.0
    if overall_preferred_min_ok:
        score += 15.0
    score += scheme_preference_bonus(scheme["name"], scheme["type"], int(scheme["requested_bins"]))
    if inner_supported:
        score += 20.0
    else:
        score -= 60.0
    if inner_summary["inner_any_bin_below_25_count"] == 0:
        score += 18.0
    else:
        score -= float(inner_summary["inner_any_bin_below_25_count"]) * 4.0
    score -= outer_summary["max_abs_bin_prop_deviation_from_full"] * 250.0
    score -= outer_summary["std_fold_mean_lesion_voxels"] / max(float(full_stats["mean"]), 1.0) * 80.0
    score -= outer_summary["std_fold_median_lesion_voxels"] / max(float(full_stats["median"]), 1.0) * 40.0
    score -= inner_summary["max_abs_val_bin_prop_deviation_from_full"] * 180.0
    score -= inner_summary["std_val_mean_lesion_voxels"] / max(float(full_stats["mean"]), 1.0) * 45.0
    score -= inner_summary["std_val_median_lesion_voxels"] / max(float(full_stats["median"]), 1.0) * 20.0
    score -= len(warnings) * 3.0

    if not overall_hard_min_ok or not inner_supported:
        label = "bad"
    elif overall_preferred_min_ok and inner_min_val_bin_count >= 15 and outer_summary["max_abs_bin_prop_deviation_from_full"] <= 0.03:
        label = "good"
    elif overall_preferred_min_ok and inner_min_val_bin_count >= 8:
        label = "acceptable"
    else:
        label = "risky"

    return {
        **scheme,
        "effective_bins": num_bins,
        "bin_counts": counts,
        "bin_ranges": bin_ranges,
        "overall_hard_min_ok": overall_hard_min_ok,
        "overall_preferred_min_ok": overall_preferred_min_ok,
        "outer_supported": outer_supported,
        "inner_supported": inner_supported,
        "outer_folds": outer_folds,
        "outer_summary": outer_summary,
        "inner_summary": inner_summary,
        "label": label,
        "recommendation_score": float(score),
        "warnings": warnings,
    }


def choose_recommendation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["label"] == "good",
            item["label"] == "acceptable",
            item["recommendation_score"],
            -len(item["warnings"]),
        ),
        reverse=True,
    )
    return ranked[0]


def zero_lesion_subject_check(
    rows: list[dict[str, Any]],
    mask_threshold: float,
) -> dict[str, Any]:
    zero_rows = [row for row in rows if row["status"] == "ok" and row["lesion_voxels"] == 0]
    if not zero_rows:
        return {
            "found": False,
            "subject_id": None,
            "recommendation": "No zero-lesion subject was found under the current threshold.",
        }

    target = next((row for row in zero_rows if row["subject_id"] == "sub-r039s002"), zero_rows[0])
    mask_path = Path(target["mask_path"])
    img = nb.load(str(mask_path))
    data = img.get_fdata(dtype=np.float32)
    positive_gt0 = int(np.count_nonzero(data > 0.0))
    positive_gt05 = int(np.count_nonzero(data > mask_threshold))
    unique_values = np.unique(data)
    unique_preview = [float(x) for x in unique_values[:10].tolist()]

    if positive_gt05 == 0 and float(np.max(data)) == 0.0:
        recommendation = (
            "Manual inspect before CV generation. Under the current mask-threshold convention (>0.5), this mask is "
            "truly empty rather than borderline. Do not remove it automatically; keep it only if the empty mask is "
            "confirmed to be valid in the source dataset, otherwise exclude it after audit."
        )
    elif positive_gt05 == 0:
        recommendation = (
            "Manual inspect before CV generation. This case is empty at >0.5 but not at >0, which suggests a "
            "thresholding or mask-value convention issue."
        )
    else:
        recommendation = (
            "Keep for now. The subject is not actually empty under the current threshold, so the prior zero-lesion "
            "flag likely came from stale or mismatched analysis output."
        )

    return {
        "found": True,
        "subject_id": target["subject_id"],
        "mask_path": str(mask_path),
        "split_source": target["split_source"],
        "shape": [int(x) for x in data.shape],
        "spacing_mm": [float(x) for x in img.header.get_zooms()[:3]],
        "stored_dtype": str(img.header.get_data_dtype()),
        "data_min": float(np.min(data)),
        "data_max": float(np.max(data)),
        "data_mean": float(np.mean(data)),
        "positive_voxels_gt0": positive_gt0,
        "positive_voxels_gt05": positive_gt05,
        "unique_value_count": int(unique_values.size),
        "unique_value_preview": unique_preview,
        "recommendation": recommendation,
    }


def maybe_make_plots(
    out_dir: Path,
    values: np.ndarray,
    recommendation: dict[str, Any],
    subject_value_map: dict[str, int],
) -> dict[str, str]:
    mpl_cache = out_dir / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    outputs: dict[str, str] = {}

    hist_path = out_dir / "lesion_volume_stress_test_log_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log1p(values), bins=40, color="#1f6aa5", edgecolor="white")
    ax.set_title("Lesion Volume Distribution (log1p scale)")
    ax.set_xlabel("log(1 + lesion voxels)")
    ax.set_ylabel("Subjects")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    outputs["log_histogram"] = str(hist_path)

    boxplot_path = out_dir / "recommended_scheme_outer_fold_boxplot.png"
    raw_fold_values = [
        [subject_value_map[subj] for subj in fold["test_subject_ids"]]
        for fold in recommendation["outer_folds"]
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.boxplot(raw_fold_values, tick_labels=[f"Fold {i}" for i in range(1, len(raw_fold_values) + 1)], showfliers=False)
    ax.set_title(f"{recommendation['name']} outer-fold lesion volume")
    ax.set_xlabel("Outer test fold")
    ax.set_ylabel("Lesion voxels")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=160)
    plt.close(fig)
    outputs["recommended_scheme_boxplot"] = str(boxplot_path)

    return outputs


def executive_summary_lines(
    recommendation: dict[str, Any],
    zero_check: dict[str, Any],
) -> list[str]:
    survived = recommendation["name"] == "quantile_k5"
    lines = [
        f"Recommended scheme: `{recommendation['name']}` ({recommendation['type']})",
        f"`quantile_k5` survived stress test: `{survived}`",
        f"Recommendation label: `{recommendation['label']}`",
        f"Outer fold max abs bin-proportion deviation from full dataset: "
        f"`{recommendation['outer_summary']['max_abs_bin_prop_deviation_from_full']:.4f}`",
        f"Inner val minimum per-bin count across outer folds: `{recommendation['inner_summary']['inner_min_val_bin_count']}`",
        f"Zero-lesion subject finding: `{zero_check['subject_id']}` -> `{zero_check['recommendation']}`" if zero_check["found"]
        else f"Zero-lesion subject finding: `{zero_check['recommendation']}`",
    ]
    return lines


def markdown_candidate_table(candidates: list[dict[str, Any]]) -> list[str]:
    lines = []
    lines.append("| Scheme | Type | Bin counts | >=5/bin | >=25/bin | Outer max prop dev | Outer std mean | Inner min val bin | Score | Label |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for item in candidates:
        outer = item["outer_summary"]
        inner = item["inner_summary"]
        lines.append(
            f"| {item['name']} | {item['type']} | {item['bin_counts']} | {item['overall_hard_min_ok']} | "
            f"{item['overall_preferred_min_ok']} | {outer['max_abs_bin_prop_deviation_from_full']:.4f} | "
            f"{outer['std_fold_mean_lesion_voxels']:.1f} | {inner['inner_min_val_bin_count']} | "
            f"{item['recommendation_score']:.2f} | {item['label']} |"
        )
    return lines


def markdown_outer_fold_details(candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in candidates:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append(f"- Description: {item['description']}")
        lines.append(f"- Edges: `{format_edges(item['edges'])}`")
        lines.append(f"- Bin counts: `{item['bin_counts']}`")
        lines.append(f"- Warnings: `{'; '.join(item['warnings'])}`" if item["warnings"] else "- Warnings: none")
        lines.append("")
        lines.append("| Fold | Test n | Bin counts | Min | Max | Mean | Std | Median | IQR | p10 | p25 | p50 | p75 | p90 | KS vs full |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for fold in item["outer_folds"]:
            stats = fold["lesion_voxels_stats"]
            pct = stats["percentiles"]
            ks_text = "n/a" if fold["ks_distance_vs_full"] is None else f"{fold['ks_distance_vs_full']:.4f}"
            lines.append(
                f"| {fold['fold']} | {fold['test_subject_count']} | {fold['test_bin_counts']} | "
                f"{format_number(stats['min'])} | {format_number(stats['max'])} | {format_number(stats['mean'])} | "
                f"{format_number(stats['std'])} | {format_number(stats['median'])} | {format_number(stats['iqr'])} | "
                f"{format_number(pct['p10'])} | {format_number(pct['p25'])} | {format_number(pct['p50'])} | "
                f"{format_number(pct['p75'])} | {format_number(pct['p90'])} | {ks_text} |"
            )
        lines.append("")
    return lines


def markdown_inner_summary_table(candidates: list[dict[str, Any]]) -> list[str]:
    lines = []
    lines.append("| Scheme | Mean val size | Min val size | Max val size | Min val bin | Outer folds with val bin <5 | Outer folds with val bin <25 | Max val prop dev | Std val mean | Std val median |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in candidates:
        inner = item["inner_summary"]
        lines.append(
            f"| {item['name']} | {inner['val_size_mean']:.1f} | {inner['val_size_min']} | {inner['val_size_max']} | "
            f"{inner['inner_min_val_bin_count']} | {inner['inner_any_bin_below_5_count']} | "
            f"{inner['inner_any_bin_below_25_count']} | {inner['max_abs_val_bin_prop_deviation_from_full']:.4f} | "
            f"{inner['std_val_mean_lesion_voxels']:.1f} | {inner['std_val_median_lesion_voxels']:.1f} |"
        )
    return lines


def markdown_inner_recommended_details(recommendation: dict[str, Any]) -> list[str]:
    lines = []
    lines.append(f"### {recommendation['name']}")
    lines.append("")
    lines.append("| Outer fold | Train n | Val n | Val bin counts | Any val bin <5 | Any val bin <25 | Val mean | Val median | Val max prop dev |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |")
    for row in recommendation["inner_summary"]["details"]:
        stats = row["val_stats"]
        lines.append(
            f"| {row['outer_fold']} | {row['train_size']} | {row['val_size']} | {row['val_bin_counts']} | "
            f"{row['any_bin_below_5']} | {row['any_bin_below_25']} | {format_number(stats['mean'])} | "
            f"{format_number(stats['median'])} | {row['val_max_abs_bin_prop_deviation_from_full']:.4f} |"
        )
    lines.append("")
    return lines


def final_recommendation_text(
    recommendation: dict[str, Any],
    quantile_k5_survived: bool,
    zero_check: dict[str, Any],
) -> list[str]:
    lines = []
    if recommendation["name"] == "quantile_k5":
        lines.append(
            "`quantile_k5` remains the most defensible carry-forward choice. The reason is not the perfectly equal "
            "overall bin counts by themselves; it is that lesion-volume quintile stratification preserves the "
            "right-skewed lesion-burden spectrum while keeping both outer test folds and inner validation splits "
            "stable enough for 5-fold CV."
        )
    else:
        lines.append(
            f"`quantile_k5` did not win the stress test. `{recommendation['name']}` provided a better balance of "
            "outer-fold stability, inner validation stability, and interpretability."
        )

    lines.append(
        f"Recommended edges on `lesion_voxels`: `{format_edges(recommendation['edges'])}` with bin counts "
        f"`{recommendation['bin_counts']}`."
    )
    lines.append(
        f"Outer-fold balance summary: max bin-proportion deviation `{recommendation['outer_summary']['max_abs_bin_prop_deviation_from_full']:.4f}`, "
        f"std of fold means `{recommendation['outer_summary']['std_fold_mean_lesion_voxels']:.1f}`, "
        f"std of fold medians `{recommendation['outer_summary']['std_fold_median_lesion_voxels']:.1f}`."
    )
    lines.append(
        f"Inner train/val stability summary: minimum val-bin count `{recommendation['inner_summary']['inner_min_val_bin_count']}`, "
        f"outer folds with any val bin <5: `{recommendation['inner_summary']['inner_any_bin_below_5_count']}`, "
        f"outer folds with any val bin <25: `{recommendation['inner_summary']['inner_any_bin_below_25_count']}`."
    )
    if zero_check["found"]:
        lines.append(f"Zero-lesion subject recommendation: `{zero_check['recommendation']}`")
    lines.append(
        "Exact next step for CV split generation: compute lesion_voxels for the final subject pool, assign the "
        f"recommended bin labels using `{recommendation['name']}` and edges `{format_edges(recommendation['edges'])}`, "
        "then generate deterministic 5-fold outer splits plus per-outer-fold deterministic 80/20 stratified "
        "train/val splits using those labels. Do not write those fold files until the zero-lesion subject has been audited."
    )
    lines.append(f"`quantile_k5` survived stress test: `{quantile_k5_survived}`")
    return lines


def write_markdown_report(
    out_path: Path,
    dataset_summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    recommendation: dict[str, Any],
    zero_check: dict[str, Any],
    plot_paths: dict[str, str],
    warnings: list[str],
) -> None:
    quantile_k5_survived = recommendation["name"] == "quantile_k5"
    lines: list[str] = []
    lines.append("# Lesion Volume Stratification Stress Test")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    for line in executive_summary_lines(recommendation, zero_check):
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## Candidate Scheme Comparison Table")
    lines.append("")
    lines.extend(markdown_candidate_table(candidates))
    lines.append("")
    lines.append("## Fold-Balance Diagnostics Table")
    lines.append("")
    lines.extend(markdown_outer_fold_details(candidates))
    lines.append("")
    lines.append("## Inner Train/Val Stability Diagnostics")
    lines.append("")
    lines.extend(markdown_inner_summary_table(candidates))
    lines.append("")
    lines.extend(markdown_inner_recommended_details(recommendation))
    lines.append("## Zero-Lesion Subject Check")
    lines.append("")
    if zero_check["found"]:
        lines.append(f"- Subject ID: `{zero_check['subject_id']}`")
        lines.append(f"- Mask path: `{zero_check['mask_path']}`")
        lines.append(f"- Split source: `{zero_check['split_source']}`")
        lines.append(f"- Shape: `{zero_check['shape']}`")
        lines.append(f"- Spacing (mm): `{zero_check['spacing_mm']}`")
        lines.append(f"- Stored dtype: `{zero_check['stored_dtype']}`")
        lines.append(f"- Data min/max/mean: `{zero_check['data_min']}`, `{zero_check['data_max']}`, `{zero_check['data_mean']}`")
        lines.append(f"- Positive voxels > 0: `{zero_check['positive_voxels_gt0']}`")
        lines.append(f"- Positive voxels > 0.5: `{zero_check['positive_voxels_gt05']}`")
        lines.append(f"- Unique value count: `{zero_check['unique_value_count']}`")
        lines.append(f"- Unique value preview: `{zero_check['unique_value_preview']}`")
        lines.append(f"- Recommendation: {zero_check['recommendation']}")
    else:
        lines.append(f"- {zero_check['recommendation']}")
    lines.append("")
    lines.append("## Final Recommendation")
    lines.append("")
    for line in final_recommendation_text(recommendation, quantile_k5_survived, zero_check):
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## Dataset Context")
    lines.append("")
    lines.append(f"- Subject source: `{dataset_summary['source']}`")
    lines.append(f"- Readable subjects analyzed: `{dataset_summary['n_subjects']}`")
    lines.append(f"- Full-dataset stats: `{json.dumps(dataset_summary['lesion_voxels_stats'], sort_keys=True)}`")
    if plot_paths:
        for label, path in plot_paths.items():
            lines.append(f"- Plot `{label}`: `{path}`")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, source_meta = load_or_recompute_subject_rows(args)
    ok_rows = [row for row in rows if row["status"] == "ok" and row["lesion_voxels"] is not None]
    if not ok_rows:
        raise SystemExit("ERROR: no readable subjects available for stress testing.")

    subject_ids = [row["subject_id"] for row in ok_rows]
    values = np.asarray([int(row["lesion_voxels"]) for row in ok_rows], dtype=np.int64)
    candidates = [
        evaluate_candidate_scheme(scheme, values, subject_ids, args.seed)
        for scheme in build_candidate_schemes(values)
    ]
    candidates = sorted(
        candidates,
        key=lambda item: (
            item["label"] == "good",
            item["label"] == "acceptable",
            item["recommendation_score"],
        ),
        reverse=True,
    )
    recommendation = choose_recommendation(candidates)
    zero_check = zero_lesion_subject_check(ok_rows, args.mask_threshold)
    subject_value_map = {row["subject_id"]: int(row["lesion_voxels"]) for row in ok_rows if row["lesion_voxels"] is not None}
    plot_paths = maybe_make_plots(out_dir, values.astype(np.float64), recommendation, subject_value_map)

    warnings = dedupe_preserve_order(
        source_meta.get("split_warnings", [])
        + source_meta.get("analysis_warnings", [])
        + [warning for candidate in candidates for warning in candidate["warnings"]]
    )

    dataset_summary = {
        "source": source_meta["source"],
        "source_kind": source_meta["source_kind"],
        "n_subjects": len(ok_rows),
        "lesion_voxels_stats": stats_block(values.astype(np.float64)),
    }
    json_payload = {
        "dataset_summary": dataset_summary,
        "candidates": candidates,
        "recommendation": recommendation,
        "zero_lesion_subject_check": zero_check,
        "plots": plot_paths,
        "warnings": warnings,
    }
    json_path = out_dir / "lesion_volume_stratification_stress_test.json"
    md_path = out_dir / "lesion_volume_stratification_stress_test.md"
    json_path.write_text(json.dumps(json_payload, indent=2))
    write_markdown_report(md_path, dataset_summary, candidates, recommendation, zero_check, plot_paths, warnings)

    print(f"stress_test_report={md_path}")
    print(f"stress_test_json={json_path}")
    print(f"recommended_scheme={recommendation['name']}")
    print(f"quantile_k5_survived={recommendation['name'] == 'quantile_k5'}")
    print(f"fold_balance_max_abs_bin_prop_dev={recommendation['outer_summary']['max_abs_bin_prop_deviation_from_full']:.6f}")
    print(f"inner_val_min_bin_count={recommendation['inner_summary']['inner_min_val_bin_count']}")
    if zero_check["found"]:
        print(f"zero_lesion_subject={zero_check['subject_id']}")
        print(f"zero_lesion_recommendation={zero_check['recommendation']}")


if __name__ == "__main__":
    main()
