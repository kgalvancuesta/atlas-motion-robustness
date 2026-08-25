#!/usr/bin/env python3
"""Build paper-supporting statistical inference tables without model inference.

This script consumes existing ATLAS per-subject metric JSON files, the ATLAS CV
split metadata, and the MR-ART fold-level lightweight statistics CSV. It writes
only CSV, JSON, Markdown, and LaTeX table candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_EDGES = [13, 1159, 5290, 36236, 496656]
PRIMARY_LABELS = [
    "[13, 1159)",
    "[1159, 5290)",
    "[5290, 36236)",
    "[36236, 496656]",
]
FAILURE_DICE_THRESHOLD = 0.10
STRONG_DICE_THRESHOLD = 0.70

ARCH_CONFIGS = {
    "base_cnn": {"root": "base_cnn", "label": "Base CNN"},
    "mednext": {"root": "mednext", "label": "MedNeXt"},
    "swin": {"root": "swin", "label": "Swin UNETR"},
    "uxnet": {"root": "uxnet", "label": "UXNet"},
}
TRAINING_CONFIGS = {
    "standard": {"prefix": "run_kfold", "label": "Standard"},
    "augmented": {"prefix": "run_DA_kfold", "label": "Augmented"},
}
EVAL_FILES = {
    "clean": "test_clean_metrics.json",
    "artifact": "test_augmented_metrics.json",
}
MRART_ENDPOINTS = [
    "predicted_lesion_volume_ml",
    "largest_cc_volume_ml",
    "connected_component_count_26conn",
    "largest_cc_fraction_of_prediction",
    "scan_positive",
]
MRART_PRIMARY_ENDPOINTS = [
    "predicted_lesion_volume_ml",
    "largest_cc_volume_ml",
    "connected_component_count_26conn",
    "largest_cc_fraction_of_prediction",
]


@dataclass(frozen=True)
class InferenceStats:
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    effect_size_paired_cohens_dz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ATLAS and MR-ART statistical inference tables from existing "
            "lightweight metrics only. Does not run model inference."
        )
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("splits/atlas_5fold_lesion_quartile_excluding_known_issues.json"),
        help="ATLAS CV split JSON containing lesion_voxels quartile metadata.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root containing ATLAS run directories.",
    )
    parser.add_argument(
        "--mrart-fold-csv",
        type=Path,
        default=Path(
            "scripts/supporting_experiments/mrart/server_reports/"
            "20260703_mrart_paper_stats/fold_lcc_stats/mrart_fold_lcc_stats_long.csv"
        ),
        help="MR-ART fold-level no-output rerun statistics CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("scripts/supporting_experiments/statistical_inference/server_reports"),
        help="Root where a new versioned output directory will be created.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit new output directory. Must not already exist.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=9001)
    return parser.parse_args()


def run_command(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def git_commit() -> str:
    code, out, _err = run_command(["git", "rev-parse", "HEAD"])
    return out.strip() if code == 0 else ""


REPO_ROOT = Path(__file__).resolve().parents[4]


def _validate_external_output_parent(path: Path) -> None:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise RuntimeError(f"External output path has no valid directory ancestor: {path}")
    if not os.access(ancestor, os.W_OK):
        raise RuntimeError(f"External output path has no writable directory ancestor: {path} (nearest: {ancestor})")


def check_output_paths_safe(paths: Iterable[Path], *, repo_root: Path = REPO_ROOT) -> None:
    resolved_repo = repo_root.resolve()
    for path in paths:
        resolved_path = path.resolve(strict=False)
        try:
            relative_path = resolved_path.relative_to(resolved_repo)
        except ValueError:
            _validate_external_output_parent(resolved_path)
            continue

        code, _out, err = run_command(
            ["git", "-C", str(resolved_repo), "check-ignore", "-q", "--", str(relative_path)]
        )
        if code == 1:
            raise RuntimeError(
                "Refusing output path inside the repository unless it is ignored by Git: "
                f"{path}"
            )
        if code != 0:
            raise RuntimeError(f"git check-ignore failed for {path}: {err.strip()}")


def choose_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out_dir = args.output_dir
        if out_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")
        return out_dir

    stem = f"{date.today().strftime('%Y%m%d')}_paper_stats"
    candidate = args.output_root / stem
    if not candidate.exists():
        return candidate
    for idx in range(2, 1000):
        versioned = args.output_root / f"{stem}_v{idx}"
        if not versioned.exists():
            return versioned
    raise RuntimeError(f"Could not find unused output directory under {args.output_root}")


def output_paths(output_dir: Path) -> dict[str, Path]:
    table_dir = output_dir / "paper_table_candidates"
    return {
        "atlas_long": output_dir / "atlas_subject_level_metrics_long.csv",
        "atlas_paired": output_dir / "atlas_clean_artifact_paired_inference.csv",
        "atlas_gap": output_dir / "atlas_augmentation_gap_reduction_inference.csv",
        "atlas_training_effects": output_dir / "atlas_clean_and_artifact_training_effects.csv",
        "atlas_arch_pairwise_dice": output_dir / "atlas_architecture_pairwise_dice_inference.csv",
        "atlas_arch_pairwise_gap": output_dir / "atlas_architecture_pairwise_robustness_gap_inference.csv",
        "lesion_full": output_dir / "atlas_lesion_size_stratified_full.csv",
        "lesion_drops": output_dir / "atlas_lesion_size_paired_drops.csv",
        "lesion_fold": output_dir / "atlas_lesion_size_fold_aware_summary.csv",
        "mrart_motion": output_dir / "mrart_acquisition_motion_contrasts_inference.csv",
        "mrart_training": output_dir / "mrart_training_condition_contrasts_inference.csv",
        "run_info": output_dir / "statistical_inference_run_info.json",
        "qc_report": output_dir / "statistical_inference_qc_report.md",
        "paper_atlas_gap_csv": table_dir / "atlas_paired_robustness_gap_reduction.csv",
        "paper_atlas_gap_tex": table_dir / "atlas_paired_robustness_gap_reduction.tex",
        "paper_lesion_aug_csv": table_dir / "atlas_lesion_size_augmented_focus.csv",
        "paper_lesion_aug_tex": table_dir / "atlas_lesion_size_augmented_focus.tex",
        "paper_mrart_burden_csv": table_dir / "mrart_false_positive_burden.csv",
        "paper_mrart_burden_tex": table_dir / "mrart_false_positive_burden.tex",
        "paper_mrart_deltas_csv": table_dir / "mrart_paired_deltas.csv",
        "paper_mrart_deltas_tex": table_dir / "mrart_paired_deltas.tex",
        "paper_mednext_arch_csv": table_dir / "atlas_mednext_pairwise_architecture_tests.csv",
        "paper_mednext_arch_tex": table_dir / "atlas_mednext_pairwise_architecture_tests.tex",
    }


def ensure_no_existing_files(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing files:\n" + "\n".join(existing))


def assign_lesion_bin(lesion_voxels: int) -> tuple[int, str]:
    for idx, (lower, upper) in enumerate(zip(PRIMARY_EDGES, PRIMARY_EDGES[1:]), start=1):
        if idx == len(PRIMARY_LABELS):
            in_bin = lower <= lesion_voxels <= upper
        else:
            in_bin = lower <= lesion_voxels < upper
        if in_bin:
            return idx, PRIMARY_LABELS[idx - 1]
    raise ValueError(f"lesion_voxels={lesion_voxels} outside bins {PRIMARY_EDGES}")


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def discover_metric_files(runs_root: Path) -> list[Path]:
    paths: list[Path] = []
    for arch_cfg in ARCH_CONFIGS.values():
        model_root = runs_root / arch_cfg["root"]
        for training_cfg in TRAINING_CONFIGS.values():
            for fold in range(5):
                run_dir = model_root / f"{training_cfg['prefix']}_{fold + 1:02d}" / "eval_local"
                for filename in EVAL_FILES.values():
                    path = run_dir / filename
                    if path.exists():
                        paths.append(path)
    return sorted(paths)


def infer_metric_context(path: Path) -> dict[str, Any]:
    parts = set(path.parts)
    arch_key = next((key for key in ARCH_CONFIGS if key in parts), "")
    if not arch_key:
        raise ValueError(f"Cannot infer architecture from {path}")

    run_name = path.parent.parent.name
    training_key = ""
    fold = None
    for key, cfg in TRAINING_CONFIGS.items():
        prefix = str(cfg["prefix"])
        if run_name.startswith(prefix + "_"):
            training_key = key
            fold = int(run_name.removeprefix(prefix + "_")) - 1
            break
    if training_key == "" or fold is None:
        raise ValueError(f"Cannot infer training/fold from {path}")

    if path.name == EVAL_FILES["clean"]:
        evaluation = "clean"
    elif path.name == EVAL_FILES["artifact"]:
        evaluation = "artifact"
    else:
        raise ValueError(f"Cannot infer evaluation condition from {path}")

    return {
        "architecture_key": arch_key,
        "architecture": ARCH_CONFIGS[arch_key]["label"],
        "training_key": training_key,
        "training_condition": TRAINING_CONFIGS[training_key]["label"],
        "evaluation_condition": evaluation,
        "fold": int(fold),
    }


def load_split(split_json: Path) -> tuple[dict[str, int], dict[int, set[str]], dict[str, Any]]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    edges = [int(float(x)) for x in metadata.get("quartile_edges", [])]
    if edges and edges != PRIMARY_EDGES:
        raise RuntimeError(f"Split quartile edges {edges} do not match required bins {PRIMARY_EDGES}")
    subject_to_fold: dict[str, int] = {}
    fold_to_test: dict[int, set[str]] = {}
    for fold_payload in payload.get("folds", []):
        fold = int(fold_payload["fold_index"])
        test_ids = {str(subject) for subject in fold_payload.get("test_ids", [])}
        fold_to_test[fold] = test_ids
        for subject in test_ids:
            if subject in subject_to_fold:
                raise RuntimeError(f"Subject appears in multiple test folds: {subject}")
            subject_to_fold[subject] = fold
    return subject_to_fold, fold_to_test, metadata


def load_atlas_metrics(
    metric_files: list[Path],
    subject_to_fold: dict[str, int],
    fold_to_test: dict[int, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qc: dict[str, Any] = {
        "source_metric_files": [str(path) for path in metric_files],
        "source_metric_file_count": len(metric_files),
        "source_metric_summary_issues": [],
        "per_file_coverage_issues": [],
        "skipped_fold_mismatch_rows": 0,
        "skipped_invalid_metric_rows": 0,
        "duplicate_subjects_within_metric_files": 0,
    }

    for path in metric_files:
        context = infer_metric_context(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        per_subject = list(payload.get("per_subject", []))
        fold = int(context["fold"])
        expected_subjects = fold_to_test.get(fold, set())
        seen = Counter()
        observed: set[str] = set()

        issue_counts = {
            name: int(summary.get(name, 0) or 0)
            for name in [
                "n_missing_pred",
                "n_missing_gt",
                "n_shape_mismatch",
                "n_affine_mismatch",
                "n_load_errors",
                "n_fallback_gt",
                "n_fallback_pred",
            ]
        }
        if (
            summary.get("split_key_used") != "test_ids"
            or int(summary.get("cv_fold", fold)) != fold
            or any(value != 0 for value in issue_counts.values())
        ):
            qc["source_metric_summary_issues"].append(
                {
                    "source_metric_file": str(path),
                    "split_key_used": summary.get("split_key_used"),
                    "cv_fold": summary.get("cv_fold"),
                    **issue_counts,
                }
            )

        for raw in per_subject:
            subject = str(raw.get("subject") or raw.get("subject_id") or "")
            if not subject:
                qc["skipped_invalid_metric_rows"] += 1
                continue
            seen[subject] += 1
            observed.add(subject)
            if subject_to_fold.get(subject) != fold:
                qc["skipped_fold_mismatch_rows"] += 1
                continue

            dice = safe_float(raw.get("dice"))
            gt_voxels = safe_int(raw.get("gt_voxels") or raw.get("ground_truth_voxels"))
            if not math.isfinite(dice) or dice < 0.0 or dice > 1.0 or gt_voxels <= 0:
                qc["skipped_invalid_metric_rows"] += 1
                continue
            lesion_bin, lesion_label = assign_lesion_bin(gt_voxels)
            rows.append(
                {
                    "subject_id": subject,
                    "fold": fold,
                    "architecture_key": context["architecture_key"],
                    "architecture": context["architecture"],
                    "training_key": context["training_key"],
                    "training_condition": context["training_condition"],
                    "evaluation_condition": context["evaluation_condition"],
                    "lesion_size_bin": lesion_bin,
                    "lesion_size_bin_label": lesion_label,
                    "lesion_voxel_range": lesion_label,
                    "dice": dice,
                    "jaccard": safe_float(raw.get("jaccard")),
                    "precision": safe_float(raw.get("precision")),
                    "recall": safe_float(raw.get("recall")),
                    "volume_similarity": safe_float(raw.get("volume_similarity")),
                    "abs_volume_diff_ratio": safe_float(raw.get("abs_volume_diff_ratio")),
                    "predicted_voxel_count": safe_int(raw.get("pred_voxels") or raw.get("predicted_voxels")),
                    "ground_truth_voxel_count": gt_voxels,
                    "failure_dice_lt_0_1": int(dice < FAILURE_DICE_THRESHOLD),
                    "strong_dice_ge_0_7": int(dice >= STRONG_DICE_THRESHOLD),
                    "source_metric_file": str(path),
                    "source_prediction_path": str(raw.get("pred_path", "")),
                    "source_ground_truth_path": str(raw.get("gt_path", "")),
                    "source_metric_mode": summary.get("metric_mode", ""),
                    "source_pred_threshold": summary.get("metric_definition", {}).get("pred_threshold", ""),
                    "source_gt_threshold": summary.get("metric_definition", {}).get("gt_threshold", ""),
                }
            )

        qc["duplicate_subjects_within_metric_files"] += sum(1 for count in seen.values() if count > 1)
        missing = sorted(expected_subjects - observed)
        unexpected = sorted(observed - expected_subjects)
        if missing or unexpected:
            qc["per_file_coverage_issues"].append(
                {
                    "source_metric_file": str(path),
                    "fold": fold,
                    "n_missing": len(missing),
                    "missing_first20": missing[:20],
                    "n_unexpected": len(unexpected),
                    "unexpected_first20": unexpected[:20],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No ATLAS metric rows loaded.")
    df = df.sort_values(
        ["architecture", "training_condition", "evaluation_condition", "fold", "subject_id"]
    ).reset_index(drop=True)
    qc.update(validate_atlas_long(df, set(subject_to_fold), subject_to_fold))
    return df, qc


def validate_atlas_long(df: pd.DataFrame, expected_subjects: set[str], subject_to_fold: dict[str, int]) -> dict[str, Any]:
    qc: dict[str, Any] = {}
    key_cols = ["subject_id", "architecture", "training_condition", "evaluation_condition"]
    duplicates = df.duplicated(key_cols, keep=False)
    qc["duplicate_subject_arch_training_eval_rows"] = int(duplicates.sum())
    qc["invalid_dice_rows"] = int((~np.isfinite(df["dice"]) | (df["dice"] < 0.0) | (df["dice"] > 1.0)).sum())
    qc["invalid_precision_rows"] = int((df["precision"].notna() & ((df["precision"] < 0.0) | (df["precision"] > 1.0))).sum())
    qc["invalid_recall_rows"] = int((df["recall"].notna() & ((df["recall"] < 0.0) | (df["recall"] > 1.0))).sum())
    qc["failure_flag_mismatches"] = int((df["failure_dice_lt_0_1"] != (df["dice"] < FAILURE_DICE_THRESHOLD).astype(int)).sum())
    qc["strong_flag_mismatches"] = int((df["strong_dice_ge_0_7"] != (df["dice"] >= STRONG_DICE_THRESHOLD).astype(int)).sum())
    qc["fold_assignment_mismatches"] = int(
        sum(int(row.fold) != subject_to_fold.get(str(row.subject_id), -1) for row in df.itertuples())
    )
    coverage: dict[str, Any] = {}
    for key, group in df.groupby(["architecture", "training_condition", "evaluation_condition"], sort=True):
        subjects = set(group["subject_id"])
        missing = sorted(expected_subjects - subjects)
        unexpected = sorted(subjects - expected_subjects)
        coverage["|".join(key)] = {
            "rows": int(len(group)),
            "unique_subjects": int(len(subjects)),
            "missing_subject_count": len(missing),
            "unexpected_subject_count": len(unexpected),
            "appears_exactly_once": len(group) == len(expected_subjects)
            and len(subjects) == len(expected_subjects)
            and not missing
            and not unexpected,
        }
    qc["coverage_by_condition"] = coverage
    qc["unique_subject_count"] = int(df["subject_id"].nunique())
    qc["lesion_bin_counts_per_condition_unique_patterns"] = {
        str(pattern): count
        for pattern, count in Counter(
            tuple(group["lesion_size_bin_label"].value_counts().sort_index().items())
            for _key, group in df.groupby(["architecture", "training_condition", "evaluation_condition"])
        ).items()
    }
    return qc


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray([np.nan if value is None else float(value) for value in p_values], dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return adjusted.tolist()
    order = valid_idx[np.argsort(p[valid_idx])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def paired_bootstrap_ci(delta: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(delta), size=(n_bootstrap, len(delta)))
    boot_means = delta[indices].mean(axis=1)
    return tuple(np.percentile(boot_means, [2.5, 97.5]).astype(float))


def clustered_bootstrap_ci(
    subject_ids: pd.Series,
    delta: pd.Series,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    # Use positional arrays. The MR-ART delta series has a MultiIndex, while
    # subject_ids is constructed from index levels; a plain DataFrame
    # constructor would align by index and turn valid deltas into NaN.
    subjects = np.asarray(list(subject_ids), dtype=str)
    deltas = pd.to_numeric(pd.Series(np.asarray(delta)), errors="coerce").to_numpy(float)
    if len(subjects) != len(deltas):
        raise ValueError(f"Cluster bootstrap length mismatch: subjects={len(subjects)} deltas={len(deltas)}")
    tmp = pd.DataFrame({"subject_id": subjects, "delta": deltas})
    tmp = tmp[np.isfinite(tmp["delta"])]
    grouped = tmp.groupby("subject_id")["delta"].agg(["sum", "count"]).reset_index()
    if grouped.empty:
        return float("nan"), float("nan")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    indices = rng.integers(0, len(grouped), size=(n_bootstrap, len(grouped)))
    boot_means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return tuple(np.percentile(boot_means, [2.5, 97.5]).astype(float))


def wilcoxon_p(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return float("nan")
    if np.allclose(delta, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def paired_effect_size(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2:
        return float("nan")
    sd = float(np.std(delta, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return float("nan")
    return float(np.mean(delta) / sd)


def inference_stats(a: pd.Series, b: pd.Series, n_bootstrap: int, rng: np.random.Generator) -> InferenceStats:
    a_values = pd.to_numeric(a, errors="coerce").to_numpy(float)
    b_values = pd.to_numeric(b, errors="coerce").to_numpy(float)
    mask = np.isfinite(a_values) & np.isfinite(b_values)
    a_values = a_values[mask]
    b_values = b_values[mask]
    delta = b_values - a_values
    ci_low, ci_high = paired_bootstrap_ci(delta, n_bootstrap, rng)
    return InferenceStats(
        n=int(len(delta)),
        mean_a=float(np.mean(a_values)) if len(delta) else float("nan"),
        mean_b=float(np.mean(b_values)) if len(delta) else float("nan"),
        mean_delta=float(np.mean(delta)) if len(delta) else float("nan"),
        median_delta=float(np.median(delta)) if len(delta) else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
        wilcoxon_p=wilcoxon_p(delta),
        effect_size_paired_cohens_dz=paired_effect_size(delta),
    )


def add_holm_by_family(df: pd.DataFrame, family_col: str = "holm_family") -> pd.DataFrame:
    out = df.copy()
    out["wilcoxon_p_holm"] = np.nan
    for family, idx in out.groupby(family_col).groups.items():
        del family
        out.loc[idx, "wilcoxon_p_holm"] = holm_adjust(out.loc[idx, "wilcoxon_p"].tolist())
    return out


def source_files_for(group: pd.DataFrame) -> str:
    return ";".join(sorted(set(group["source_metric_file"].astype(str))))


def build_atlas_clean_artifact(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arch, training), group in df.groupby(["architecture", "training_condition"], sort=True):
        wide = group.pivot_table(
            index=["subject_id", "fold"],
            columns="evaluation_condition",
            values="dice",
            aggfunc="first",
        ).dropna(subset=["clean", "artifact"])
        stats_row = inference_stats(wide["artifact"], wide["clean"], n_bootstrap, rng)
        # inference_stats reports b - a, so a=artifact, b=clean gives clean-artifact drop.
        rows.append(
            {
                "architecture": arch,
                "training_condition": training,
                "n": stats_row.n,
                "clean_mean_dice": stats_row.mean_b,
                "artifact_mean_dice": stats_row.mean_a,
                "mean_dice_drop_clean_minus_artifact": stats_row.mean_delta,
                "median_dice_drop_clean_minus_artifact": stats_row.median_delta,
                "bootstrap_ci95_low": stats_row.ci_low,
                "bootstrap_ci95_high": stats_row.ci_high,
                "wilcoxon_p": stats_row.wilcoxon_p,
                "effect_size_paired_cohens_dz": stats_row.effect_size_paired_cohens_dz,
                "holm_family": "atlas_clean_artifact_all_arch_training",
                "source_metric_files": source_files_for(group),
            }
        )
    return add_holm_by_family(pd.DataFrame(rows))


def build_atlas_aug_effects(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for arch, group in df.groupby("architecture", sort=True):
        wide = group.pivot_table(
            index=["subject_id", "fold"],
            columns=["training_condition", "evaluation_condition"],
            values="dice",
            aggfunc="first",
        )
        required = [
            ("Standard", "clean"),
            ("Standard", "artifact"),
            ("Augmented", "clean"),
            ("Augmented", "artifact"),
        ]
        wide = wide.dropna(subset=required)
        standard_gap = wide[("Standard", "clean")] - wide[("Standard", "artifact")]
        augmented_gap = wide[("Augmented", "clean")] - wide[("Augmented", "artifact")]
        gap_stats = inference_stats(augmented_gap, standard_gap, n_bootstrap, rng)
        # b - a = standard_gap - augmented_gap = positive gap reduction.
        gap_rows.append(
            {
                "architecture": arch,
                "n": gap_stats.n,
                "mean_standard_gap_clean_minus_artifact": gap_stats.mean_b,
                "mean_augmented_gap_clean_minus_artifact": gap_stats.mean_a,
                "mean_gap_reduction_standard_minus_augmented": gap_stats.mean_delta,
                "median_gap_reduction_standard_minus_augmented": gap_stats.median_delta,
                "bootstrap_ci95_low": gap_stats.ci_low,
                "bootstrap_ci95_high": gap_stats.ci_high,
                "wilcoxon_p": gap_stats.wilcoxon_p,
                "effect_size_paired_cohens_dz": gap_stats.effect_size_paired_cohens_dz,
                "holm_family": "atlas_gap_reduction_by_architecture",
                "source_metric_files": source_files_for(group),
            }
        )
        for evaluation in ["clean", "artifact"]:
            effect_stats = inference_stats(
                wide[("Standard", evaluation)],
                wide[("Augmented", evaluation)],
                n_bootstrap,
                rng,
            )
            effect_rows.append(
                {
                    "architecture": arch,
                    "evaluation_condition": evaluation,
                    "endpoint": f"{evaluation}_dice_augmented_minus_standard",
                    "n": effect_stats.n,
                    "mean_standard_dice": effect_stats.mean_a,
                    "mean_augmented_dice": effect_stats.mean_b,
                    "mean_augmented_minus_standard": effect_stats.mean_delta,
                    "median_augmented_minus_standard": effect_stats.median_delta,
                    "bootstrap_ci95_low": effect_stats.ci_low,
                    "bootstrap_ci95_high": effect_stats.ci_high,
                    "wilcoxon_p": effect_stats.wilcoxon_p,
                    "effect_size_paired_cohens_dz": effect_stats.effect_size_paired_cohens_dz,
                    "holm_family": f"atlas_training_effect_{evaluation}",
                    "source_metric_files": source_files_for(group),
                }
            )
    return add_holm_by_family(pd.DataFrame(gap_rows)), add_holm_by_family(pd.DataFrame(effect_rows))


def build_atlas_architecture_pairwise(
    df: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare architectures with paired subject/fold contrasts.

    Dice comparisons are stratified by training and evaluation condition.
    Robustness-gap comparisons are stratified by training condition and compare
    clean-artifact gaps. For the gap table, negative architecture_a_minus_b means
    architecture A had a smaller clean-to-artifact gap than architecture B.
    """
    architectures = sorted(df["architecture"].unique().tolist())
    dice_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for (training, evaluation), group in df.groupby(["training_condition", "evaluation_condition"], sort=True):
        wide = group.pivot_table(
            index=["subject_id", "fold"],
            columns="architecture",
            values="dice",
            aggfunc="first",
        )
        for arch_a, arch_b in combinations(architectures, 2):
            pair = wide.dropna(subset=[arch_a, arch_b])
            stats_row = inference_stats(pair[arch_b], pair[arch_a], n_bootstrap, rng)
            dice_rows.append(
                {
                    "architecture_a": arch_a,
                    "architecture_b": arch_b,
                    "training_condition": training,
                    "evaluation_condition": evaluation,
                    "endpoint": "dice",
                    "contrast": "architecture_a_minus_architecture_b",
                    "n": stats_row.n,
                    "mean_architecture_a": stats_row.mean_b,
                    "mean_architecture_b": stats_row.mean_a,
                    "mean_difference_architecture_a_minus_b": stats_row.mean_delta,
                    "median_difference_architecture_a_minus_b": stats_row.median_delta,
                    "bootstrap_ci95_low": stats_row.ci_low,
                    "bootstrap_ci95_high": stats_row.ci_high,
                    "wilcoxon_p": stats_row.wilcoxon_p,
                    "effect_size_paired_cohens_dz": stats_row.effect_size_paired_cohens_dz,
                    "holm_family": f"atlas_architecture_pairwise_dice_{training}_{evaluation}",
                    "source_metric_files": source_files_for(group[group["architecture"].isin([arch_a, arch_b])]),
                    "interpretation": "positive means architecture_a has higher Dice than architecture_b",
                }
            )

    for training, group in df.groupby("training_condition", sort=True):
        wide = group.pivot_table(
            index=["subject_id", "fold"],
            columns=["architecture", "evaluation_condition"],
            values="dice",
            aggfunc="first",
        )
        for arch_a, arch_b in combinations(architectures, 2):
            required = [(arch_a, "clean"), (arch_a, "artifact"), (arch_b, "clean"), (arch_b, "artifact")]
            pair = wide.dropna(subset=required)
            gap_a = pair[(arch_a, "clean")] - pair[(arch_a, "artifact")]
            gap_b = pair[(arch_b, "clean")] - pair[(arch_b, "artifact")]
            stats_row = inference_stats(gap_b, gap_a, n_bootstrap, rng)
            gap_rows.append(
                {
                    "architecture_a": arch_a,
                    "architecture_b": arch_b,
                    "training_condition": training,
                    "endpoint": "clean_minus_artifact_dice_gap",
                    "contrast": "architecture_a_gap_minus_architecture_b_gap",
                    "n": stats_row.n,
                    "mean_gap_architecture_a": stats_row.mean_b,
                    "mean_gap_architecture_b": stats_row.mean_a,
                    "mean_gap_difference_architecture_a_minus_b": stats_row.mean_delta,
                    "median_gap_difference_architecture_a_minus_b": stats_row.median_delta,
                    "bootstrap_ci95_low": stats_row.ci_low,
                    "bootstrap_ci95_high": stats_row.ci_high,
                    "wilcoxon_p": stats_row.wilcoxon_p,
                    "effect_size_paired_cohens_dz": stats_row.effect_size_paired_cohens_dz,
                    "holm_family": f"atlas_architecture_pairwise_gap_{training}",
                    "source_metric_files": source_files_for(group[group["architecture"].isin([arch_a, arch_b])]),
                    "interpretation": "negative means architecture_a has smaller clean-to-artifact Dice gap than architecture_b",
                }
            )

    return add_holm_by_family(pd.DataFrame(dice_rows)), add_holm_by_family(pd.DataFrame(gap_rows))


def iqr(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(0.75) - values.quantile(0.25))


def semicolon_unique(series: pd.Series) -> str:
    return ";".join(sorted(set(series.astype(str))))


def build_lesion_size_tables(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = (
        df.groupby(
            [
                "architecture",
                "training_condition",
                "evaluation_condition",
                "lesion_size_bin",
                "lesion_size_bin_label",
                "lesion_voxel_range",
            ],
            sort=True,
        )
        .agg(
            n=("subject_id", "nunique"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            iqr_dice=("dice", iqr),
            catastrophic_failure_rate_dice_lt_0_1=("failure_dice_lt_0_1", "mean"),
            strong_performance_rate_dice_ge_0_7=("strong_dice_ge_0_7", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            median_predicted_voxel_count=("predicted_voxel_count", "median"),
            median_ground_truth_voxel_count=("ground_truth_voxel_count", "median"),
            source_metric_files=("source_metric_file", semicolon_unique),
        )
        .reset_index()
    )

    paired_rows: list[dict[str, Any]] = []
    for (arch, training, bin_id, bin_label, voxel_range), group in df.groupby(
        ["architecture", "training_condition", "lesion_size_bin", "lesion_size_bin_label", "lesion_voxel_range"],
        sort=True,
    ):
        wide = group.pivot_table(
            index=["subject_id", "fold"],
            columns="evaluation_condition",
            values=["dice", "failure_dice_lt_0_1", "strong_dice_ge_0_7"],
            aggfunc="first",
        ).dropna(subset=[("dice", "clean"), ("dice", "artifact")])
        dice_stats = inference_stats(wide[("dice", "artifact")], wide[("dice", "clean")], n_bootstrap, rng)
        clean_failure = wide[("failure_dice_lt_0_1", "clean")].astype(float)
        artifact_failure = wide[("failure_dice_lt_0_1", "artifact")].astype(float)
        clean_strong = wide[("strong_dice_ge_0_7", "clean")].astype(float)
        artifact_strong = wide[("strong_dice_ge_0_7", "artifact")].astype(float)
        paired_rows.append(
            {
                "architecture": arch,
                "training_condition": training,
                "lesion_size_bin": bin_id,
                "lesion_size_bin_label": bin_label,
                "lesion_voxel_range": voxel_range,
                "n": dice_stats.n,
                "clean_mean_dice": dice_stats.mean_b,
                "artifact_mean_dice": dice_stats.mean_a,
                "mean_dice_drop_clean_minus_artifact": dice_stats.mean_delta,
                "median_dice_drop_clean_minus_artifact": dice_stats.median_delta,
                "bootstrap_ci95_low": dice_stats.ci_low,
                "bootstrap_ci95_high": dice_stats.ci_high,
                "wilcoxon_p": dice_stats.wilcoxon_p,
                "effect_size_paired_cohens_dz": dice_stats.effect_size_paired_cohens_dz,
                "clean_failure_rate_dice_lt_0_1": float(clean_failure.mean()),
                "artifact_failure_rate_dice_lt_0_1": float(artifact_failure.mean()),
                "failure_rate_increase_artifact_minus_clean": float(artifact_failure.mean() - clean_failure.mean()),
                "clean_strong_rate_dice_ge_0_7": float(clean_strong.mean()),
                "artifact_strong_rate_dice_ge_0_7": float(artifact_strong.mean()),
                "strong_rate_change_artifact_minus_clean": float(artifact_strong.mean() - clean_strong.mean()),
                "holm_family": "atlas_lesion_size_clean_artifact_drops",
                "source_metric_files": source_files_for(group),
            }
        )
    paired = add_holm_by_family(pd.DataFrame(paired_rows))

    fold_level = (
        df.groupby(
            [
                "architecture",
                "training_condition",
                "evaluation_condition",
                "lesion_size_bin",
                "lesion_size_bin_label",
                "lesion_voxel_range",
                "fold",
            ],
            sort=True,
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            fold_mean_dice=("dice", "mean"),
            fold_failure_rate_dice_lt_0_1=("failure_dice_lt_0_1", "mean"),
            fold_strong_rate_dice_ge_0_7=("strong_dice_ge_0_7", "mean"),
        )
        .reset_index()
    )
    fold_summary = (
        fold_level.groupby(
            [
                "architecture",
                "training_condition",
                "evaluation_condition",
                "lesion_size_bin",
                "lesion_size_bin_label",
                "lesion_voxel_range",
            ],
            sort=True,
        )
        .agg(
            n_folds=("fold", "nunique"),
            total_subjects=("n_subjects", "sum"),
            mean_fold_mean_dice=("fold_mean_dice", "mean"),
            std_fold_mean_dice=("fold_mean_dice", "std"),
            mean_fold_failure_rate_dice_lt_0_1=("fold_failure_rate_dice_lt_0_1", "mean"),
            std_fold_failure_rate_dice_lt_0_1=("fold_failure_rate_dice_lt_0_1", "std"),
            mean_fold_strong_rate_dice_ge_0_7=("fold_strong_rate_dice_ge_0_7", "mean"),
            std_fold_strong_rate_dice_ge_0_7=("fold_strong_rate_dice_ge_0_7", "std"),
        )
        .reset_index()
    )
    return full, paired, fold_summary


def normalize_mrart(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["architecture"] = out["architecture"].astype(str).str.lower()
    out["training_condition"] = out["training_condition"].astype(str).str.lower()
    out["acquisition_type"] = out["acquisition_type"].astype(str).str.lower()
    for col in [
        "fold",
        "threshold",
        "voxel_volume_mm3",
        "predicted_lesion_volume_ml",
        "largest_cc_volume_ml",
        "connected_component_count_26conn",
        "largest_cc_fraction_of_prediction",
        "scan_positive",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_mrart(df: pd.DataFrame) -> dict[str, Any]:
    expected_arch = {"mednext", "swin_unetr"}
    expected_training = {"standard", "augmented"}
    expected_acq = {"standard", "headmotion1", "headmotion2"}
    status_counts = df["status"].astype(str).value_counts().to_dict()
    return {
        "row_count": int(len(df)),
        "expected_row_count": 8400,
        "status_counts": status_counts,
        "non_success_rows": int((df["status"].astype(str) != "success").sum()),
        "architectures": sorted(set(df["architecture"].astype(str))),
        "unexpected_architecture_rows": int((~df["architecture"].isin(expected_arch)).sum()),
        "training_conditions": sorted(set(df["training_condition"].astype(str))),
        "unexpected_training_rows": int((~df["training_condition"].isin(expected_training)).sum()),
        "acquisition_types": sorted(set(df["acquisition_type"].astype(str))),
        "unexpected_acquisition_rows": int((~df["acquisition_type"].isin(expected_acq)).sum()),
        "folds": sorted(int(x) for x in df["fold"].dropna().unique()),
        "threshold_values": sorted(float(x) for x in df["threshold"].dropna().unique()),
        "invalid_voxel_volume_rows": int((~np.isfinite(df["voxel_volume_mm3"]) | (df["voxel_volume_mm3"] <= 0)).sum()),
    }


def build_mrart_motion(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator, source_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arch, training), group in df.groupby(["architecture", "training_condition"], sort=True):
        for motion in ["headmotion1", "headmotion2"]:
            for endpoint in MRART_ENDPOINTS:
                wide = group.pivot_table(
                    index=["subject_id", "fold"],
                    columns="acquisition_type",
                    values=endpoint,
                    aggfunc="first",
                ).dropna(subset=["standard", motion])
                delta = wide[motion] - wide["standard"]
                ci_low, ci_high = clustered_bootstrap_ci(
                    pd.Series(wide.index.get_level_values("subject_id")),
                    delta,
                    n_bootstrap,
                    rng,
                )
                subject_delta = delta.groupby(level="subject_id").mean()
                rows.append(
                    {
                        "architecture": arch,
                        "training_condition": training,
                        "contrast": f"{motion}_minus_standard",
                        "motion_acquisition": motion,
                        "endpoint": endpoint,
                        "endpoint_family": "primary" if endpoint in MRART_PRIMARY_ENDPOINTS else "secondary",
                        "n_subjects": int(wide.index.get_level_values("subject_id").nunique()),
                        "n_fold_predictions": int(len(wide)),
                        "mean_standard": float(wide["standard"].mean()),
                        "mean_motion": float(wide[motion].mean()),
                        "mean_delta_motion_minus_standard": float(delta.mean()),
                        "median_delta_motion_minus_standard": float(delta.median()),
                        "bootstrap_ci95_low": float(ci_low),
                        "bootstrap_ci95_high": float(ci_high),
                        "wilcoxon_subject_average_p": wilcoxon_p(subject_delta.to_numpy(float)),
                        "effect_size_subject_average_paired_cohens_dz": paired_effect_size(subject_delta.to_numpy(float)),
                        "holm_family": f"mrart_motion_{endpoint}",
                        "source_mrart_csv": str(source_csv),
                    }
                )
    out = pd.DataFrame(rows)
    out = out.rename(columns={"wilcoxon_subject_average_p": "wilcoxon_p"})
    return add_holm_by_family(out)


def build_mrart_training(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator, source_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arch, acq), group in df.groupby(["architecture", "acquisition_type"], sort=True):
        for endpoint in MRART_ENDPOINTS:
            wide = group.pivot_table(
                index=["subject_id", "fold"],
                columns="training_condition",
                values=endpoint,
                aggfunc="first",
            ).dropna(subset=["standard", "augmented"])
            delta = wide["augmented"] - wide["standard"]
            ci_low, ci_high = clustered_bootstrap_ci(
                pd.Series(wide.index.get_level_values("subject_id")),
                delta,
                n_bootstrap,
                rng,
            )
            subject_delta = delta.groupby(level="subject_id").mean()
            rows.append(
                {
                    "architecture": arch,
                    "acquisition_type": acq,
                    "contrast": "augmented_minus_standard",
                    "endpoint": endpoint,
                    "endpoint_family": "primary" if endpoint in MRART_PRIMARY_ENDPOINTS else "secondary",
                    "n_subjects": int(wide.index.get_level_values("subject_id").nunique()),
                    "n_fold_predictions": int(len(wide)),
                    "mean_standard": float(wide["standard"].mean()),
                    "mean_augmented": float(wide["augmented"].mean()),
                    "mean_delta_augmented_minus_standard": float(delta.mean()),
                    "median_delta_augmented_minus_standard": float(delta.median()),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "wilcoxon_p": wilcoxon_p(subject_delta.to_numpy(float)),
                    "effect_size_subject_average_paired_cohens_dz": paired_effect_size(subject_delta.to_numpy(float)),
                    "holm_family": f"mrart_training_{endpoint}",
                    "source_mrart_csv": str(source_csv),
                }
            )
    return add_holm_by_family(pd.DataFrame(rows))


def format_ci(row: pd.Series, low: str = "bootstrap_ci95_low", high: str = "bootstrap_ci95_high") -> str:
    return f"[{row[low]:.3f}, {row[high]:.3f}]"


def write_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.write_text(
        df.to_latex(index=False, escape=True, caption=caption, label=label, float_format="%.3f"),
        encoding="utf-8",
    )


def build_paper_candidates(
    atlas_gap: pd.DataFrame,
    arch_pairwise_dice: pd.DataFrame,
    arch_pairwise_gap: pd.DataFrame,
    lesion_full: pd.DataFrame,
    mrart_df: pd.DataFrame,
    mrart_motion: pd.DataFrame,
    mrart_training: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    gap = atlas_gap.copy()
    gap["CI95"] = gap.apply(format_ci, axis=1)
    gap_table = gap[
        [
            "architecture",
            "n",
            "mean_standard_gap_clean_minus_artifact",
            "mean_augmented_gap_clean_minus_artifact",
            "mean_gap_reduction_standard_minus_augmented",
            "CI95",
            "wilcoxon_p_holm",
        ]
    ].rename(
        columns={
            "architecture": "Architecture",
            "n": "N",
            "mean_standard_gap_clean_minus_artifact": "Std gap",
            "mean_augmented_gap_clean_minus_artifact": "Aug gap",
            "mean_gap_reduction_standard_minus_augmented": "Gap reduction",
            "wilcoxon_p_holm": "Holm p",
        }
    )
    gap_table.to_csv(paths["paper_atlas_gap_csv"], index=False)
    write_latex(gap_table, paths["paper_atlas_gap_tex"], "ATLAS paired robustness gap reduction.", "tab:atlas_gap_reduction")
    counts["paper_atlas_gap_rows"] = len(gap_table)

    lesion_focus = lesion_full[
        (lesion_full["architecture"].isin(["MedNeXt", "Swin UNETR"]))
        & (lesion_full["training_condition"] == "Augmented")
        & (lesion_full["evaluation_condition"] == "artifact")
    ].copy()
    lesion_table = lesion_focus[
        [
            "architecture",
            "lesion_size_bin_label",
            "n",
            "mean_dice",
            "median_dice",
            "catastrophic_failure_rate_dice_lt_0_1",
            "strong_performance_rate_dice_ge_0_7",
        ]
    ].rename(
        columns={
            "architecture": "Architecture",
            "lesion_size_bin_label": "Lesion voxel stratum",
            "n": "N",
            "mean_dice": "Mean artifact Dice",
            "median_dice": "Median artifact Dice",
            "catastrophic_failure_rate_dice_lt_0_1": "Failure rate",
            "strong_performance_rate_dice_ge_0_7": "Strong rate",
        }
    )
    lesion_table.to_csv(paths["paper_lesion_aug_csv"], index=False)
    write_latex(lesion_table, paths["paper_lesion_aug_tex"], "Lesion-size stratified augmented-model artifact performance.", "tab:lesion_augmented")
    counts["paper_lesion_aug_rows"] = len(lesion_table)

    burden = (
        mrart_df.groupby(["architecture", "training_condition", "acquisition_type"], sort=True)
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_fold_predictions=("subject_id", "size"),
            mean_predicted_volume_ml=("predicted_lesion_volume_ml", "mean"),
            median_predicted_volume_ml=("predicted_lesion_volume_ml", "median"),
            mean_largest_cc_volume_ml=("largest_cc_volume_ml", "mean"),
            median_largest_cc_volume_ml=("largest_cc_volume_ml", "median"),
            mean_connected_component_count=("connected_component_count_26conn", "mean"),
            scan_positive_rate=("scan_positive", "mean"),
        )
        .reset_index()
    )
    burden_table = burden.rename(
        columns={
            "architecture": "Architecture",
            "training_condition": "Training",
            "acquisition_type": "Acquisition",
            "n_subjects": "Subjects",
            "n_fold_predictions": "Fold predictions",
            "mean_predicted_volume_ml": "Mean predicted mL",
            "median_predicted_volume_ml": "Median predicted mL",
            "mean_largest_cc_volume_ml": "Mean LCC mL",
            "median_largest_cc_volume_ml": "Median LCC mL",
            "mean_connected_component_count": "Mean component count",
            "scan_positive_rate": "Scan-positive rate",
        }
    )
    burden_table.to_csv(paths["paper_mrart_burden_csv"], index=False)
    write_latex(burden_table, paths["paper_mrart_burden_tex"], "MR-ART fold-level false-positive burden.", "tab:mrart_burden")
    counts["paper_mrart_burden_rows"] = len(burden_table)

    motion_focus = mrart_motion[mrart_motion["endpoint"].isin(["predicted_lesion_volume_ml", "largest_cc_volume_ml"])].copy()
    motion_focus["comparison_type"] = "motion"
    motion_focus["delta"] = motion_focus["mean_delta_motion_minus_standard"]
    training_focus = mrart_training[mrart_training["endpoint"].isin(["predicted_lesion_volume_ml", "largest_cc_volume_ml"])].copy()
    training_focus["comparison_type"] = "training"
    training_focus["training_condition"] = ""
    training_focus["motion_acquisition"] = training_focus["acquisition_type"]
    training_focus["delta"] = training_focus["mean_delta_augmented_minus_standard"]
    deltas = pd.concat(
        [
            motion_focus[
                [
                    "comparison_type",
                    "architecture",
                    "training_condition",
                    "motion_acquisition",
                    "contrast",
                    "endpoint",
                    "n_subjects",
                    "delta",
                    "bootstrap_ci95_low",
                    "bootstrap_ci95_high",
                    "wilcoxon_p_holm",
                ]
            ],
            training_focus[
                [
                    "comparison_type",
                    "architecture",
                    "training_condition",
                    "motion_acquisition",
                    "contrast",
                    "endpoint",
                    "n_subjects",
                    "delta",
                    "bootstrap_ci95_low",
                    "bootstrap_ci95_high",
                    "wilcoxon_p_holm",
                ]
            ],
        ],
        ignore_index=True,
    )
    deltas["CI95"] = deltas.apply(format_ci, axis=1)
    deltas_table = deltas[
        [
            "comparison_type",
            "architecture",
            "training_condition",
            "motion_acquisition",
            "contrast",
            "endpoint",
            "n_subjects",
            "delta",
            "CI95",
            "wilcoxon_p_holm",
        ]
    ].rename(
        columns={
            "comparison_type": "Type",
            "architecture": "Architecture",
            "training_condition": "Training",
            "motion_acquisition": "Acquisition",
            "contrast": "Contrast",
            "endpoint": "Endpoint",
            "n_subjects": "Subjects",
            "delta": "Mean delta",
            "wilcoxon_p_holm": "Holm p",
        }
    )
    deltas_table.to_csv(paths["paper_mrart_deltas_csv"], index=False)
    write_latex(deltas_table, paths["paper_mrart_deltas_tex"], "MR-ART paired motion and training-condition deltas.", "tab:mrart_deltas")
    counts["paper_mrart_deltas_rows"] = len(deltas_table)

    mednext_rows: list[dict[str, Any]] = []
    for row in arch_pairwise_dice.itertuples(index=False):
        if "MedNeXt" not in (row.architecture_a, row.architecture_b):
            continue
        comparator = row.architecture_b if row.architecture_a == "MedNeXt" else row.architecture_a
        sign = 1.0 if row.architecture_a == "MedNeXt" else -1.0
        ci_low = row.bootstrap_ci95_low if sign > 0 else -row.bootstrap_ci95_high
        ci_high = row.bootstrap_ci95_high if sign > 0 else -row.bootstrap_ci95_low
        mednext_rows.append(
            {
                "Comparison": "Dice",
                "Training": row.training_condition,
                "Evaluation": row.evaluation_condition,
                "Comparator": comparator,
                "N": row.n,
                "MedNeXt minus comparator": sign * row.mean_difference_architecture_a_minus_b,
                "CI95": f"[{ci_low:.3f}, {ci_high:.3f}]",
                "Holm p": row.wilcoxon_p_holm,
                "Interpretation": "positive favors MedNeXt",
            }
        )
    for row in arch_pairwise_gap.itertuples(index=False):
        if "MedNeXt" not in (row.architecture_a, row.architecture_b):
            continue
        comparator = row.architecture_b if row.architecture_a == "MedNeXt" else row.architecture_a
        sign = 1.0 if row.architecture_a == "MedNeXt" else -1.0
        ci_low = row.bootstrap_ci95_low if sign > 0 else -row.bootstrap_ci95_high
        ci_high = row.bootstrap_ci95_high if sign > 0 else -row.bootstrap_ci95_low
        mednext_rows.append(
            {
                "Comparison": "Clean-artifact Dice gap",
                "Training": row.training_condition,
                "Evaluation": "",
                "Comparator": comparator,
                "N": row.n,
                "MedNeXt minus comparator": sign * row.mean_gap_difference_architecture_a_minus_b,
                "CI95": f"[{ci_low:.3f}, {ci_high:.3f}]",
                "Holm p": row.wilcoxon_p_holm,
                "Interpretation": "negative favors MedNeXt smaller robustness gap",
            }
        )
    mednext_table = pd.DataFrame(mednext_rows)
    mednext_table.to_csv(paths["paper_mednext_arch_csv"], index=False)
    write_latex(
        mednext_table,
        paths["paper_mednext_arch_tex"],
        "MedNeXt paired architecture comparisons.",
        "tab:mednext_architecture_comparisons",
    )
    counts["paper_mednext_arch_rows"] = len(mednext_table)
    return counts


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_qc_report(path: Path, payload: dict[str, Any]) -> None:
    atlas_qc = payload["atlas_qc"]
    mrart_qc = payload["mrart_qc"]
    lines = [
        "# Statistical Inference QC Report",
        "",
        "## Inputs",
        "",
        f"- ATLAS metric files discovered: {atlas_qc['source_metric_file_count']}.",
        f"- ATLAS split JSON: `{payload['inputs']['split_json']}`.",
        f"- MR-ART fold CSV: `{payload['inputs']['mrart_fold_csv']}`.",
        "",
        "## ATLAS QC",
        "",
        f"- Subject count: {atlas_qc['unique_subject_count']}.",
        f"- Duplicate subject/model/training/evaluation rows: {atlas_qc['duplicate_subject_arch_training_eval_rows']}.",
        f"- Fold assignment mismatches: {atlas_qc['fold_assignment_mismatches']}.",
        f"- Invalid Dice rows: {atlas_qc['invalid_dice_rows']}.",
        f"- Failure flag mismatches: {atlas_qc['failure_flag_mismatches']}.",
        f"- Strong-Dice flag mismatches: {atlas_qc['strong_flag_mismatches']}.",
        f"- Source metric summary issues: {len(atlas_qc['source_metric_summary_issues'])}.",
        f"- Per-file coverage issues: {len(atlas_qc['per_file_coverage_issues'])}.",
        "",
        "## MR-ART QC",
        "",
        f"- Row count: {mrart_qc['row_count']} expected {mrart_qc['expected_row_count']}.",
        f"- Status counts: {mrart_qc['status_counts']}.",
        f"- Non-success rows: {mrart_qc['non_success_rows']}.",
        f"- Architectures: {mrart_qc['architectures']}.",
        f"- Training conditions: {mrart_qc['training_conditions']}.",
        f"- Acquisition types: {mrart_qc['acquisition_types']}.",
        f"- Folds: {mrart_qc['folds']}.",
        f"- Threshold values: {mrart_qc['threshold_values']}.",
        f"- Invalid voxel-volume rows: {mrart_qc['invalid_voxel_volume_rows']}.",
        "",
        "## Statistical Settings",
        "",
        f"- Bootstrap iterations: {payload['settings']['n_bootstrap']}.",
        f"- Seed: {payload['settings']['seed']}.",
        "- Bootstrap CI: percentile 95% CI.",
        "- ATLAS bootstrap: paired subject resampling.",
        "- ATLAS between-architecture tests: paired by held-out subject and fold, conditional on fixed trained checkpoints.",
        "- MR-ART bootstrap: clustered paired bootstrap over subject_id, retaining fold rows through cluster sums/counts.",
        "- Wilcoxon: scipy signed-rank test with zero_method='pratt'; all-zero differences return p=1.",
        "- Multiple-comparison correction: Holm adjustment within named endpoint/family groups.",
        "",
        "## Warnings And Assumptions",
        "",
    ]
    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No blocking warnings.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = choose_output_dir(args)
    paths = output_paths(output_dir)
    check_output_paths_safe([output_dir, *paths.values()])
    ensure_no_existing_files(paths.values())
    output_dir.mkdir(parents=True, exist_ok=False)
    paths["paper_atlas_gap_csv"].parent.mkdir(parents=True, exist_ok=False)

    print(f"Output directory: {output_dir}")
    print("Loading ATLAS metrics...")
    subject_to_fold, fold_to_test, split_metadata = load_split(args.split_json)
    metric_files = discover_metric_files(args.runs_root)
    atlas_df, atlas_qc = load_atlas_metrics(metric_files, subject_to_fold, fold_to_test)

    print("Computing ATLAS inference tables...")
    rng = np.random.default_rng(args.seed)
    atlas_paired = build_atlas_clean_artifact(atlas_df, args.n_bootstrap, rng)
    atlas_gap, atlas_training_effects = build_atlas_aug_effects(atlas_df, args.n_bootstrap, rng)
    atlas_arch_pairwise_dice, atlas_arch_pairwise_gap = build_atlas_architecture_pairwise(
        atlas_df, args.n_bootstrap, rng
    )
    lesion_full, lesion_drops, lesion_fold = build_lesion_size_tables(atlas_df, args.n_bootstrap, rng)

    print("Loading MR-ART fold statistics...")
    mrart_df = normalize_mrart(pd.read_csv(args.mrart_fold_csv))
    mrart_qc = validate_mrart(mrart_df)
    print("Computing MR-ART inference tables...")
    mrart_motion = build_mrart_motion(mrart_df, args.n_bootstrap, rng, args.mrart_fold_csv)
    mrart_training = build_mrart_training(mrart_df, args.n_bootstrap, rng, args.mrart_fold_csv)

    print("Writing outputs...")
    atlas_df.to_csv(paths["atlas_long"], index=False)
    atlas_paired.to_csv(paths["atlas_paired"], index=False)
    atlas_gap.to_csv(paths["atlas_gap"], index=False)
    atlas_training_effects.to_csv(paths["atlas_training_effects"], index=False)
    atlas_arch_pairwise_dice.to_csv(paths["atlas_arch_pairwise_dice"], index=False)
    atlas_arch_pairwise_gap.to_csv(paths["atlas_arch_pairwise_gap"], index=False)
    lesion_full.to_csv(paths["lesion_full"], index=False)
    lesion_drops.to_csv(paths["lesion_drops"], index=False)
    lesion_fold.to_csv(paths["lesion_fold"], index=False)
    mrart_motion.to_csv(paths["mrart_motion"], index=False)
    mrart_training.to_csv(paths["mrart_training"], index=False)
    paper_counts = build_paper_candidates(
        atlas_gap,
        atlas_arch_pairwise_dice,
        atlas_arch_pairwise_gap,
        lesion_full,
        mrart_df,
        mrart_motion,
        mrart_training,
        paths,
    )

    output_counts = {}
    for key, path in paths.items():
        if path.suffix == ".csv":
            output_counts[key] = int(pd.read_csv(path).shape[0])

    warnings = []
    if atlas_df["volume_similarity"].notna().sum() == 0:
        warnings.append(
            "ATLAS source JSONs did not include volume_similarity; abs_volume_diff_ratio is preserved separately."
        )
    warnings.append(
        "Between-architecture inference is paired over held-out subjects for fixed checkpoints; it does not estimate repeated-seed or retraining variability."
    )
    if int(mrart_qc["row_count"]) != int(mrart_qc["expected_row_count"]):
        warnings.append("MR-ART fold CSV row count differs from expected 8400.")
    if int(mrart_qc["non_success_rows"]) != 0:
        warnings.append("MR-ART fold CSV contains non-success rows.")

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "inputs": {
            "split_json": str(args.split_json),
            "runs_root": str(args.runs_root),
            "mrart_fold_csv": str(args.mrart_fold_csv),
            "atlas_metric_files": [str(path) for path in metric_files],
        },
        "settings": {"n_bootstrap": args.n_bootstrap, "seed": args.seed},
        "lesion_bins": {
            "target": split_metadata.get("stratification_target", "lesion_voxels"),
            "scheme": split_metadata.get("stratification_scheme", "quantile_k4"),
            "edges": PRIMARY_EDGES,
            "labels": PRIMARY_LABELS,
            "split_json_bin_counts": split_metadata.get("quartile_bin_counts", {}),
        },
        "atlas_qc": atlas_qc,
        "mrart_qc": mrart_qc,
        "output_counts": output_counts,
        "paper_table_candidate_counts": paper_counts,
        "outputs": {key: str(path) for key, path in paths.items()},
        "multiple_comparison_families": {
            "atlas_clean_artifact": "all architecture x training clean-artifact tests",
            "atlas_gap_reduction": "architecture-level gap-reduction tests",
            "atlas_training_effects": "separate clean and artifact augmented-minus-standard families",
            "atlas_architecture_pairwise_dice": "architecture pairwise tests within each training x evaluation condition",
            "atlas_architecture_pairwise_gap": "architecture pairwise tests within each training condition for clean-artifact gaps",
            "atlas_lesion_size": "all lesion-bin clean-artifact drop tests",
            "mrart_motion": "separate family per endpoint",
            "mrart_training": "separate family per endpoint",
        },
        "warnings": warnings,
    }
    write_json(paths["run_info"], run_info)
    write_qc_report(paths["qc_report"], run_info)

    for key, path in paths.items():
        if path.exists():
            print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
