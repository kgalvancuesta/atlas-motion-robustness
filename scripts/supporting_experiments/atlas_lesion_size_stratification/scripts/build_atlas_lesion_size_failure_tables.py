#!/usr/bin/env python3
"""Build ATLAS lesion voxel-count quartile failure tables from per-case metrics.

This is a paper-support script for the ATLAS robustness analysis. It uses the
pre-existing CV lesion_voxels quartile bins and held-out test metrics only. It
does not run inference or read prediction/label NIfTI files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PRIMARY_EDGES = [13, 1159, 5290, 36236, 496656]
PRIMARY_LABELS = [
    "[13, 1159)",
    "[1159, 5290)",
    "[5290, 36236)",
    "[36236, 496656]",
]
EXPECTED_SUBJECTS = 653
FAILURE_DICE_THRESHOLD = 0.1
STRONG_DICE_THRESHOLD = 0.7

ARCH_CONFIGS = {
    "base_cnn": {
        "root": Path("runs/base_cnn"),
        "label": "Base CNN",
        "compact_prefix": "Base CNN",
    },
    "mednext": {
        "root": Path("runs/mednext"),
        "label": "MedNeXt",
        "compact_prefix": "MedNeXt",
    },
    "swin": {
        "root": Path("runs/swin"),
        "label": "Swin UNETR",
        "compact_prefix": "Swin UNETR",
    },
    "uxnet": {
        "root": Path("runs/uxnet"),
        "label": "UXNet",
        "compact_prefix": "UXNet",
    },
}
TRAINING_CONFIGS = {
    "standard": {
        "prefix": "run_kfold",
        "label": "Standard",
        "compact_suffix": "Std",
    },
    "augmented": {
        "prefix": "run_DA_kfold",
        "label": "Augmented",
        "compact_suffix": "Aug",
    },
}
EVAL_FILES = {
    "clean": "test_clean_metrics.json",
    "artifact": "test_augmented_metrics.json",
}
COMPACT_MODEL_ORDER = ["Base CNN", "MedNeXt", "Swin UNETR", "UXNet"]
COMPACT_TRAINING_ORDER = ["Standard", "Augmented"]


def compact_prefix(architecture: str, training: str) -> str:
    training_suffix = "Std" if training == "Standard" else "Aug"
    return f"{architecture} {training_suffix}"

LONG_COLUMNS = [
    "subject_id",
    "fold",
    "lesion_size_bin",
    "lesion_size_bin_label",
    "lesion_voxels",
    "model_architecture",
    "training_condition",
    "evaluation_condition",
    "dice",
    "precision",
    "recall",
    "volume_similarity",
    "predicted_voxels",
    "ground_truth_voxels",
    "failure_dice_lt_0_1",
    "strong_dice_ge_0_7",
    "source_metric_file",
]

POOLED_COLUMNS = [
    "lesion_size_bin",
    "lesion_size_bin_label",
    "lesion_voxel_range",
    "model_architecture",
    "training_condition",
    "evaluation_condition",
    "n_subjects",
    "mean_dice",
    "median_dice",
    "std_dice",
    "iqr_dice",
    "failure_rate_dice_lt_0_1",
    "strong_rate_dice_ge_0_7",
    "mean_precision",
    "mean_recall",
    "mean_volume_similarity",
]

FOLD_AWARE_COLUMNS = [
    "lesion_size_bin",
    "lesion_size_bin_label",
    "lesion_voxel_range",
    "model_architecture",
    "training_condition",
    "evaluation_condition",
    "n_folds",
    "total_subjects",
    "mean_fold_mean_dice",
    "std_fold_mean_dice",
    "mean_fold_failure_rate_dice_lt_0_1",
    "std_fold_failure_rate_dice_lt_0_1",
    "mean_fold_strong_rate_dice_ge_0_7",
    "std_fold_strong_rate_dice_ge_0_7",
]

DELTA_COLUMNS = [
    "lesion_size_bin",
    "lesion_size_bin_label",
    "lesion_voxel_range",
    "model_architecture",
    "training_condition",
    "n_subjects",
    "mean_clean_dice",
    "mean_artifact_dice",
    "mean_paired_dice_drop_clean_minus_artifact",
    "median_paired_dice_drop_clean_minus_artifact",
    "std_paired_dice_drop_clean_minus_artifact",
    "clean_failure_rate_dice_lt_0_1",
    "artifact_failure_rate_dice_lt_0_1",
    "failure_rate_increase_artifact_minus_clean",
    "clean_strong_rate_dice_ge_0_7",
    "artifact_strong_rate_dice_ge_0_7",
    "strong_rate_decrease_clean_minus_artifact",
]

COMPACT_PREFIX_ORDER = [
    compact_prefix(architecture, training)
    for architecture in COMPACT_MODEL_ORDER
    for training in COMPACT_TRAINING_ORDER
]
COMPACT_COLUMNS = (
    ["lesion_size_stratum", "lesion_voxel_range", "N"]
    + [f"{prefix} artifact Dice" for prefix in COMPACT_PREFIX_ORDER]
    + [f"{prefix} failure rate Dice < 0.1" for prefix in COMPACT_PREFIX_ORDER]
    + [f"{prefix} Delta Dice" for prefix in COMPACT_PREFIX_ORDER]
)

DELTA_COMPACT_COLUMNS = (
    ["lesion_size_stratum", "lesion_voxel_range", "N"]
    + [f"{prefix} Delta Dice" for prefix in COMPACT_PREFIX_ORDER]
    + [f"{prefix} failure rate increase" for prefix in COMPACT_PREFIX_ORDER]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("splits/atlas_5fold_lesion_quartile_excluding_known_issues.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--metrics-file",
        "--metrics-csv",
        dest="metric_files",
        action="append",
        type=Path,
        default=None,
        help="Explicit per-case metric JSON/CSV. If omitted, expected k-fold metric JSONs are auto-discovered.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory. Defaults to scripts/supporting_experiments/atlas_lesion_size_stratification/csv/<timestamp>.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_output_dir() -> Path:
    return Path("scripts/supporting_experiments/atlas_lesion_size_stratification/csv") / utc_stamp()


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

        result = subprocess.run(
            ["git", "-C", str(resolved_repo), "check-ignore", "-q", "--", str(relative_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 1:
            raise RuntimeError(
                "Refusing output path inside the repository unless it is ignored by Git: "
                f"{path}"
            )
        if result.returncode != 0:
            raise RuntimeError(f"git check-ignore failed for {path}: {result.stderr.strip()}")


def ensure_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_new_path(path)
    check_output_paths_safe([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_new_path(path)
    check_output_paths_safe([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_new_path(path)
    check_output_paths_safe([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def std_population(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def iqr(values: list[float]) -> float | None:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = parse_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def load_split(path: Path) -> tuple[dict[str, int], dict[int, set[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    split_edges = [int(x) for x in metadata.get("quartile_edges", [])]
    if split_edges and split_edges != PRIMARY_EDGES:
        raise RuntimeError(
            "Split JSON quartile edges do not match the required primary bins: "
            f"metadata={split_edges} required={PRIMARY_EDGES}"
        )

    subject_to_fold: dict[str, int] = {}
    fold_to_test: dict[int, set[str]] = {}
    duplicates: list[str] = []
    for fold in payload.get("folds", []):
        fold_index = int(fold["fold_index"])
        test_ids = {str(subject) for subject in fold.get("test_ids", [])}
        fold_to_test[fold_index] = test_ids
        for subject in sorted(test_ids):
            if subject in subject_to_fold:
                duplicates.append(subject)
            subject_to_fold[subject] = fold_index
    if duplicates:
        raise RuntimeError(f"Subjects appear in outer test more than once: {duplicates[:20]}")
    return subject_to_fold, fold_to_test, metadata


def assign_bin(lesion_voxels: int) -> tuple[int, str]:
    for idx in range(len(PRIMARY_EDGES) - 1):
        lower = PRIMARY_EDGES[idx]
        upper = PRIMARY_EDGES[idx + 1]
        if idx == len(PRIMARY_EDGES) - 2:
            in_bin = lower <= lesion_voxels <= upper
        else:
            in_bin = lower <= lesion_voxels < upper
        if in_bin:
            return idx + 1, PRIMARY_LABELS[idx]
    raise ValueError(f"lesion_voxels={lesion_voxels} outside primary bin edges {PRIMARY_EDGES}")


def discover_metric_files(runs_root: Path) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    warnings: list[str] = []
    for arch_key, arch_cfg in ARCH_CONFIGS.items():
        model_root = runs_root / arch_cfg["root"].name
        for training_cfg in TRAINING_CONFIGS.values():
            for fold in range(5):
                run_dir = model_root / f"{training_cfg['prefix']}_{fold + 1:02d}"
                for filename in EVAL_FILES.values():
                    path = run_dir / "eval_local" / filename
                    if path.exists():
                        paths.append(path)
                    else:
                        warnings.append(f"Missing expected metric file for {arch_key}: {path}")
    return paths, warnings


def infer_file_context(path: Path) -> dict[str, Any]:
    parts = path.parts
    arch_key = ""
    for candidate in ARCH_CONFIGS:
        if candidate in parts:
            arch_key = candidate
            break
    if not arch_key:
        raise ValueError(f"Cannot infer architecture from metric path: {path}")

    run_name = path.parent.parent.name
    training_key = ""
    fold = None
    for key, cfg in TRAINING_CONFIGS.items():
        prefix = cfg["prefix"]
        if run_name.startswith(prefix + "_"):
            training_key = key
            suffix = run_name.removeprefix(prefix + "_")
            fold = int(suffix) - 1
            break
    if training_key == "" or fold is None:
        raise ValueError(f"Cannot infer training/fold from metric path: {path}")

    name = path.name.lower()
    if "clean" in name:
        evaluation = "clean"
    elif "augmented" in name or "artifact" in name or "corrupt" in name:
        evaluation = "artifact"
    else:
        raise ValueError(f"Cannot infer evaluation condition from metric path: {path}")

    return {
        "architecture_key": arch_key,
        "model_architecture": ARCH_CONFIGS[arch_key]["label"],
        "training_key": training_key,
        "training_condition": TRAINING_CONFIGS[training_key]["label"],
        "evaluation_condition": evaluation,
        "fold": fold,
    }


def read_metric_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.suffix.lower() == ".json":
        report = json.loads(path.read_text(encoding="utf-8"))
        return list(report.get("per_subject", [])), report.get("summary", {})
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f)), {}
    raise ValueError(f"Unsupported metric file extension: {path}")


def source_subject(row: dict[str, Any]) -> str:
    return str(row.get("subject") or row.get("subject_id") or row.get("id") or "")


def first_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def build_long_rows(
    metric_files: list[Path],
    subject_to_fold: dict[str, int],
    fold_to_test: dict[int, set[str]],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lesion_voxels_by_subject: dict[str, int] = {}
    duplicate_file_subjects = 0
    invalid_metric_rows = 0
    fold_mismatch_rows = 0
    missing_subject_rows = 0
    source_files_used: list[str] = []

    for path in sorted(metric_files):
        context = infer_file_context(path)
        source_files_used.append(str(path))
        raw_rows, summary = read_metric_rows(path)
        fold = int(context["fold"])
        expected_subjects = fold_to_test.get(fold, set())
        seen_in_file: Counter[str] = Counter()
        observed_subjects: set[str] = set()

        for raw in raw_rows:
            subject = source_subject(raw)
            if not subject:
                missing_subject_rows += 1
                continue
            seen_in_file[subject] += 1
            observed_subjects.add(subject)

            expected_fold = subject_to_fold.get(subject)
            if expected_fold != fold:
                fold_mismatch_rows += 1
                continue

            dice = parse_float(raw.get("dice"))
            precision = parse_float(raw.get("precision"))
            recall = parse_float(raw.get("recall"))
            volume_similarity = parse_float(first_value(raw, ["volume_similarity", "volumetric_similarity"]))
            pred_voxels = parse_int(first_value(raw, ["predicted_voxels", "pred_voxels"]))
            gt_voxels = parse_int(first_value(raw, ["ground_truth_voxels", "gt_voxels", "lesion_voxels"]))
            if dice is None or dice < 0.0 or dice > 1.0 or gt_voxels is None:
                invalid_metric_rows += 1
                continue

            previous_voxels = lesion_voxels_by_subject.get(subject)
            if previous_voxels is not None and previous_voxels != gt_voxels:
                diagnostics.setdefault("lesion_voxel_mismatches", []).append(
                    {
                        "subject_id": subject,
                        "previous_lesion_voxels": previous_voxels,
                        "current_lesion_voxels": gt_voxels,
                        "source_metric_file": str(path),
                    }
                )
            lesion_voxels_by_subject[subject] = gt_voxels

            lesion_bin, lesion_label = assign_bin(gt_voxels)
            rows.append(
                {
                    "subject_id": subject,
                    "fold": fold,
                    "lesion_size_bin": lesion_bin,
                    "lesion_size_bin_label": lesion_label,
                    "lesion_voxels": gt_voxels,
                    "model_architecture": context["model_architecture"],
                    "training_condition": context["training_condition"],
                    "evaluation_condition": context["evaluation_condition"],
                    "dice": format_number(dice),
                    "precision": format_number(precision),
                    "recall": format_number(recall),
                    "volume_similarity": format_number(volume_similarity),
                    "predicted_voxels": format_number(pred_voxels),
                    "ground_truth_voxels": format_number(gt_voxels),
                    "failure_dice_lt_0_1": int(dice < FAILURE_DICE_THRESHOLD),
                    "strong_dice_ge_0_7": int(dice >= STRONG_DICE_THRESHOLD),
                    "source_metric_file": str(path),
                }
            )

        duplicate_file_subjects += sum(1 for count in seen_in_file.values() if count > 1)
        missing = sorted(expected_subjects - observed_subjects)
        unexpected = sorted(observed_subjects - expected_subjects)
        if missing or unexpected:
            diagnostics.setdefault("per_file_subject_coverage_issues", []).append(
                {
                    "source_metric_file": str(path),
                    "fold": fold,
                    "missing_subjects": missing[:50],
                    "n_missing_subjects": len(missing),
                    "unexpected_subjects": unexpected[:50],
                    "n_unexpected_subjects": len(unexpected),
                    "summary_split_key_used": summary.get("split_key_used"),
                    "summary_cv_fold": summary.get("cv_fold"),
                }
            )

    diagnostics["source_metric_files_used"] = source_files_used
    diagnostics["total_rows_read"] = sum(len(read_metric_rows(path)[0]) for path in metric_files)
    diagnostics["missing_subject_id_rows"] = missing_subject_rows
    diagnostics["duplicate_subjects_within_metric_files"] = duplicate_file_subjects
    diagnostics["fold_mismatch_rows_skipped"] = fold_mismatch_rows
    diagnostics["invalid_metric_rows_skipped"] = invalid_metric_rows
    diagnostics["unique_subject_lesion_voxel_counts"] = len(lesion_voxels_by_subject)
    diagnostics["lesion_voxel_subject_bin_counts"] = dict(
        Counter(assign_bin(voxels)[1] for voxels in lesion_voxels_by_subject.values())
    )
    return sorted(
        rows,
        key=lambda row: (
            row["model_architecture"],
            row["training_condition"],
            row["evaluation_condition"],
            int(row["fold"]),
            row["subject_id"],
        ),
    )


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def build_pooled_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["lesion_size_bin", "lesion_size_bin_label", "model_architecture", "training_condition", "evaluation_condition"]
    grouped = group_rows(rows, keys)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        lesion_bin, label, arch, training, evaluation = key
        dice_values = numeric_values(group, "dice")
        precision_values = numeric_values(group, "precision")
        recall_values = numeric_values(group, "recall")
        volume_similarity_values = numeric_values(group, "volume_similarity")
        failures = numeric_values(group, "failure_dice_lt_0_1")
        strong = numeric_values(group, "strong_dice_ge_0_7")
        out.append(
            {
                "lesion_size_bin": lesion_bin,
                "lesion_size_bin_label": label,
                "lesion_voxel_range": label,
                "model_architecture": arch,
                "training_condition": training,
                "evaluation_condition": evaluation,
                "n_subjects": len({row["subject_id"] for row in group}),
                "mean_dice": format_number(mean(dice_values)),
                "median_dice": format_number(median(dice_values)),
                "std_dice": format_number(std_population(dice_values)),
                "iqr_dice": format_number(iqr(dice_values)),
                "failure_rate_dice_lt_0_1": format_number(mean(failures)),
                "strong_rate_dice_ge_0_7": format_number(mean(strong)),
                "mean_precision": format_number(mean(precision_values)),
                "mean_recall": format_number(mean(recall_values)),
                "mean_volume_similarity": format_number(mean(volume_similarity_values)),
            }
        )
    return out


def build_fold_aware_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fold_keys = [
        "lesion_size_bin",
        "lesion_size_bin_label",
        "model_architecture",
        "training_condition",
        "evaluation_condition",
        "fold",
    ]
    fold_groups = group_rows(rows, fold_keys)
    fold_stats: list[dict[str, Any]] = []
    for key, group in fold_groups.items():
        lesion_bin, label, arch, training, evaluation, fold = key
        dice_values = numeric_values(group, "dice")
        failures = numeric_values(group, "failure_dice_lt_0_1")
        strong = numeric_values(group, "strong_dice_ge_0_7")
        fold_stats.append(
            {
                "lesion_size_bin": lesion_bin,
                "lesion_size_bin_label": label,
                "model_architecture": arch,
                "training_condition": training,
                "evaluation_condition": evaluation,
                "fold": fold,
                "n_subjects": len({row["subject_id"] for row in group}),
                "fold_mean_dice": mean(dice_values),
                "fold_failure_rate": mean(failures),
                "fold_strong_rate": mean(strong),
            }
        )

    group_keys = ["lesion_size_bin", "lesion_size_bin_label", "model_architecture", "training_condition", "evaluation_condition"]
    grouped = group_rows(fold_stats, group_keys)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        lesion_bin, label, arch, training, evaluation = key
        fold_mean_dice = [float(row["fold_mean_dice"]) for row in group if row["fold_mean_dice"] is not None]
        fold_failure = [float(row["fold_failure_rate"]) for row in group if row["fold_failure_rate"] is not None]
        fold_strong = [float(row["fold_strong_rate"]) for row in group if row["fold_strong_rate"] is not None]
        out.append(
            {
                "lesion_size_bin": lesion_bin,
                "lesion_size_bin_label": label,
                "lesion_voxel_range": label,
                "model_architecture": arch,
                "training_condition": training,
                "evaluation_condition": evaluation,
                "n_folds": len({row["fold"] for row in group}),
                "total_subjects": sum(int(row["n_subjects"]) for row in group),
                "mean_fold_mean_dice": format_number(mean(fold_mean_dice)),
                "std_fold_mean_dice": format_number(std_population(fold_mean_dice)),
                "mean_fold_failure_rate_dice_lt_0_1": format_number(mean(fold_failure)),
                "std_fold_failure_rate_dice_lt_0_1": format_number(std_population(fold_failure)),
                "mean_fold_strong_rate_dice_ge_0_7": format_number(mean(fold_strong)),
                "std_fold_strong_rate_dice_ge_0_7": format_number(std_population(fold_strong)),
            }
        )
    return out


def build_delta_summary(rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["subject_id"], row["model_architecture"], row["training_condition"], row["lesion_size_bin_label"])
        by_pair[key][row["evaluation_condition"]] = row

    paired_rows: list[dict[str, Any]] = []
    unpaired = 0
    for (_subject, _arch, _training, _label), pair in by_pair.items():
        if "clean" not in pair or "artifact" not in pair:
            unpaired += 1
            continue
        clean = pair["clean"]
        artifact = pair["artifact"]
        clean_dice = parse_float(clean["dice"])
        artifact_dice = parse_float(artifact["dice"])
        if clean_dice is None or artifact_dice is None:
            unpaired += 1
            continue
        paired_rows.append(
            {
                "subject_id": clean["subject_id"],
                "lesion_size_bin": clean["lesion_size_bin"],
                "lesion_size_bin_label": clean["lesion_size_bin_label"],
                "model_architecture": clean["model_architecture"],
                "training_condition": clean["training_condition"],
                "clean_dice": clean_dice,
                "artifact_dice": artifact_dice,
                "dice_drop": clean_dice - artifact_dice,
                "clean_failure": int(clean["failure_dice_lt_0_1"]),
                "artifact_failure": int(artifact["failure_dice_lt_0_1"]),
                "clean_strong": int(clean["strong_dice_ge_0_7"]),
                "artifact_strong": int(artifact["strong_dice_ge_0_7"]),
            }
        )
    diagnostics["unmatched_clean_artifact_pair_rows"] = unpaired
    diagnostics["matched_clean_artifact_pairs"] = len(paired_rows)

    keys = ["lesion_size_bin", "lesion_size_bin_label", "model_architecture", "training_condition"]
    grouped = group_rows(paired_rows, keys)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        lesion_bin, label, arch, training = key
        clean_dice = numeric_values(group, "clean_dice")
        artifact_dice = numeric_values(group, "artifact_dice")
        drops = numeric_values(group, "dice_drop")
        clean_failures = numeric_values(group, "clean_failure")
        artifact_failures = numeric_values(group, "artifact_failure")
        clean_strong = numeric_values(group, "clean_strong")
        artifact_strong = numeric_values(group, "artifact_strong")
        clean_failure_rate = mean(clean_failures)
        artifact_failure_rate = mean(artifact_failures)
        clean_strong_rate = mean(clean_strong)
        artifact_strong_rate = mean(artifact_strong)
        out.append(
            {
                "lesion_size_bin": lesion_bin,
                "lesion_size_bin_label": label,
                "lesion_voxel_range": label,
                "model_architecture": arch,
                "training_condition": training,
                "n_subjects": len({row["subject_id"] for row in group}),
                "mean_clean_dice": format_number(mean(clean_dice)),
                "mean_artifact_dice": format_number(mean(artifact_dice)),
                "mean_paired_dice_drop_clean_minus_artifact": format_number(mean(drops)),
                "median_paired_dice_drop_clean_minus_artifact": format_number(median(drops)),
                "std_paired_dice_drop_clean_minus_artifact": format_number(std_population(drops)),
                "clean_failure_rate_dice_lt_0_1": format_number(clean_failure_rate),
                "artifact_failure_rate_dice_lt_0_1": format_number(artifact_failure_rate),
                "failure_rate_increase_artifact_minus_clean": format_number(
                    artifact_failure_rate - clean_failure_rate
                    if artifact_failure_rate is not None and clean_failure_rate is not None
                    else None
                ),
                "clean_strong_rate_dice_ge_0_7": format_number(clean_strong_rate),
                "artifact_strong_rate_dice_ge_0_7": format_number(artifact_strong_rate),
                "strong_rate_decrease_clean_minus_artifact": format_number(
                    clean_strong_rate - artifact_strong_rate
                    if clean_strong_rate is not None and artifact_strong_rate is not None
                    else None
                ),
            }
        )
    return out


def build_compact_tables(
    pooled_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    expected_bin_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pooled_lookup = {
        (
            row["lesion_size_bin_label"],
            row["model_architecture"],
            row["training_condition"],
            row["evaluation_condition"],
        ): row
        for row in pooled_rows
    }
    delta_lookup = {
        (row["lesion_size_bin_label"], row["model_architecture"], row["training_condition"]): row
        for row in delta_rows
    }

    compact_rows: list[dict[str, Any]] = []
    delta_compact_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(PRIMARY_LABELS, start=1):
        stratum = f"Bin {idx}"
        compact = {"lesion_size_stratum": stratum, "lesion_voxel_range": label, "N": expected_bin_counts.get(label, "")}
        delta_compact = {"lesion_size_stratum": stratum, "lesion_voxel_range": label, "N": expected_bin_counts.get(label, "")}
        for arch in COMPACT_MODEL_ORDER:
            for training in COMPACT_TRAINING_ORDER:
                prefix = compact_prefix(arch, training)
                artifact = pooled_lookup.get((label, arch, training, "artifact"), {})
                delta = delta_lookup.get((label, arch, training), {})
                compact[f"{prefix} artifact Dice"] = artifact.get("mean_dice", "")
                compact[f"{prefix} failure rate Dice < 0.1"] = artifact.get("failure_rate_dice_lt_0_1", "")
                compact[f"{prefix} Delta Dice"] = delta.get("mean_paired_dice_drop_clean_minus_artifact", "")
                delta_compact[f"{prefix} Delta Dice"] = delta.get("mean_paired_dice_drop_clean_minus_artifact", "")
                delta_compact[f"{prefix} failure rate increase"] = delta.get("failure_rate_increase_artifact_minus_clean", "")
        compact_rows.append(compact)
        delta_compact_rows.append(delta_compact)
    return compact_rows, delta_compact_rows


def validate_long_rows(rows: list[dict[str, Any]], expected_subjects: set[str], diagnostics: dict[str, Any]) -> None:
    key_counts: Counter[tuple[str, str, str, str]] = Counter()
    invalid_dice = 0
    nonfinite_metrics = 0
    for row in rows:
        key_counts[
            (
                row["subject_id"],
                row["model_architecture"],
                row["training_condition"],
                row["evaluation_condition"],
            )
        ] += 1
        dice = parse_float(row["dice"])
        if dice is None or dice < 0.0 or dice > 1.0:
            invalid_dice += 1
        for metric in ["precision", "recall", "volume_similarity"]:
            value = row.get(metric)
            if value != "" and parse_float(value) is None:
                nonfinite_metrics += 1

    duplicate_keys = [key for key, count in key_counts.items() if count > 1]
    diagnostics["duplicate_subject_model_training_evaluation_rows"] = len(duplicate_keys)
    diagnostics["duplicate_examples_first20"] = [list(key) for key in duplicate_keys[:20]]
    diagnostics["invalid_dice_rows"] = invalid_dice
    diagnostics["nonfinite_optional_metric_values"] = nonfinite_metrics

    coverage_by_condition: dict[str, dict[str, Any]] = {}
    grouped = group_rows(rows, ["model_architecture", "training_condition", "evaluation_condition"])
    for key, group in grouped.items():
        arch, training, evaluation = key
        subjects = {row["subject_id"] for row in group}
        missing = sorted(expected_subjects - subjects)
        unexpected = sorted(subjects - expected_subjects)
        coverage_by_condition[f"{arch}|{training}|{evaluation}"] = {
            "observed_subjects": len(subjects),
            "expected_subjects": len(expected_subjects),
            "missing_subject_count": len(missing),
            "unexpected_subject_count": len(unexpected),
            "missing_subjects_first20": missing[:20],
            "unexpected_subjects_first20": unexpected[:20],
            "appears_exactly_once": len(subjects) == len(expected_subjects)
            and not missing
            and not unexpected
            and all(
                key_counts[(subject, arch, training, evaluation)] == 1
                for subject in subjects
            ),
        }
    diagnostics["coverage_by_model_training_evaluation"] = coverage_by_condition

    bin_counts = Counter(row["lesion_size_bin_label"] for row in rows if row["evaluation_condition"] == "clean")
    diagnostics["long_clean_row_bin_counts"] = dict(bin_counts)


def report_findings(delta_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for label in PRIMARY_LABELS:
        bin_rows = [row for row in delta_rows if row["lesion_size_bin_label"] == label]
        if not bin_rows:
            continue
        drops = numeric_values(bin_rows, "mean_paired_dice_drop_clean_minus_artifact")
        failure_increases = numeric_values(bin_rows, "failure_rate_increase_artifact_minus_clean")
        lines.append(
            f"- {label}: mean clean-to-artifact Dice drop across model/training groups = "
            f"{format_number(mean(drops))}; mean failure-rate increase = {format_number(mean(failure_increases))}."
        )
    return lines


def build_markdown_report(
    output_paths: dict[str, Path],
    diagnostics: dict[str, Any],
    delta_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# ATLAS Lesion Voxel-Count Stratification Failure Analysis",
        "",
        "## Method",
        "",
        "- Primary strata are the pre-existing CV lesion voxel-count quartiles from `splits/atlas_5fold_lesion_quartile_excluding_known_issues.json`.",
        "- Bins are `[13, 1159)`, `[1159, 5290)`, `[5290, 36236)`, and `[36236, 496656]`; the final bin is inclusive.",
        "- Catastrophic failure is defined as Dice < 0.1.",
        "- Rows use held-out test-set predictions only.",
        "- Physical volume conversion is not used for primary stratification.",
        "",
        "## High-Level Result",
        "",
    ]
    lines.extend(report_findings(delta_rows) or ["- No paired clean/artifact delta rows were available."])
    lines.extend(
        [
            "",
            "Interpretation should use Dice drop, failure rate, and strong-Dice rate together. Small lesions can drive many Dice failures, but the bin-wise delta table is the direct test of whether artifact degradation persists across lesion sizes.",
            "",
            "## Diagnostics",
            "",
            f"- Source metric files used: {len(diagnostics.get('source_metric_files_used', []))}.",
            f"- Total rows written: {diagnostics.get('total_rows_written')}.",
            f"- Duplicate subject/model/training/evaluation rows: {diagnostics.get('duplicate_subject_model_training_evaluation_rows')}.",
            f"- Invalid Dice rows: {diagnostics.get('invalid_dice_rows')}.",
            f"- Unmatched clean/artifact pair rows: {diagnostics.get('unmatched_clean_artifact_pair_rows')}.",
            f"- Physical volume conversion used: {diagnostics.get('physical_volume_conversion_used')}.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in output_paths.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def planned_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "long_per_case_csv": output_dir / "atlas_lesion_size_per_case_metrics.csv",
        "pooled_bin_summary_csv": output_dir / "atlas_lesion_size_bin_summary_pooled.csv",
        "fold_aware_bin_summary_csv": output_dir / "atlas_lesion_size_bin_summary_fold_aware.csv",
        "robustness_delta_csv": output_dir / "atlas_lesion_size_bin_robustness_delta.csv",
        "paper_table_compact_csv": output_dir / "atlas_lesion_size_paper_table_compact.csv",
        "paper_table_delta_compact_csv": output_dir / "atlas_lesion_size_paper_table_delta_compact.csv",
        "diagnostics_json": output_dir / "atlas_lesion_size_stratification_diagnostics.json",
        "report_md": output_dir / "atlas_lesion_size_stratification_report.md",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_paths = planned_output_paths(output_dir)
    for path in output_paths.values():
        ensure_new_path(path)
    check_output_paths_safe(output_paths.values())

    subject_to_fold, fold_to_test, split_metadata = load_split(args.split_json)
    expected_subjects = set(subject_to_fold)
    expected_bin_counts = {label: int(count) for label, count in split_metadata.get("quartile_bin_counts", {}).items()}
    if not expected_bin_counts:
        expected_bin_counts = {label: 0 for label in PRIMARY_LABELS}

    diagnostics: dict[str, Any] = {
        "split_json_used": str(args.split_json),
        "stratification_target": split_metadata.get("stratification_target", "lesion_voxels"),
        "stratification_scheme": split_metadata.get("stratification_scheme", "quantile_k4"),
        "bin_thresholds": PRIMARY_EDGES,
        "bin_labels": PRIMARY_LABELS,
        "split_json_bin_counts": expected_bin_counts,
        "expected_subjects_per_condition": len(expected_subjects),
        "expected_subjects_per_condition_nominal": EXPECTED_SUBJECTS,
        "physical_volume_conversion_used": False,
        "caveats": [
            "Primary bins are lesion voxel-count quartiles, not physical-volume quartiles.",
            "volume_similarity is left blank unless present in the source metric file; local test metrics provide abs_volume_diff_ratio instead.",
            "Pooled summaries are per-subject summaries; fold-aware summaries are provided as sensitivity analyses.",
        ],
    }

    if args.metric_files:
        metric_files = args.metric_files
        discovery_warnings: list[str] = []
    else:
        metric_files, discovery_warnings = discover_metric_files(args.runs_root)
    diagnostics["metric_discovery_warnings"] = discovery_warnings
    diagnostics["metric_files_found"] = len(metric_files)

    long_rows = build_long_rows(metric_files, subject_to_fold, fold_to_test, diagnostics)
    validate_long_rows(long_rows, expected_subjects, diagnostics)
    pooled_rows = build_pooled_summary(long_rows)
    fold_rows = build_fold_aware_summary(long_rows)
    delta_rows = build_delta_summary(long_rows, diagnostics)
    compact_rows, delta_compact_rows = build_compact_tables(pooled_rows, delta_rows, expected_bin_counts)

    diagnostics["total_rows_written"] = len(long_rows)
    diagnostics["output_rows"] = {
        "long_per_case_csv": len(long_rows),
        "pooled_bin_summary_csv": len(pooled_rows),
        "fold_aware_bin_summary_csv": len(fold_rows),
        "robustness_delta_csv": len(delta_rows),
        "paper_table_compact_csv": len(compact_rows),
        "paper_table_delta_compact_csv": len(delta_compact_rows),
    }
    diagnostics["output_columns"] = {
        "long_per_case_csv": len(LONG_COLUMNS),
        "pooled_bin_summary_csv": len(POOLED_COLUMNS),
        "fold_aware_bin_summary_csv": len(FOLD_AWARE_COLUMNS),
        "robustness_delta_csv": len(DELTA_COLUMNS),
        "paper_table_compact_csv": len(COMPACT_COLUMNS),
        "paper_table_delta_compact_csv": len(DELTA_COMPACT_COLUMNS),
    }

    report_md = build_markdown_report(output_paths, diagnostics, delta_rows)

    write_csv(output_paths["long_per_case_csv"], LONG_COLUMNS, long_rows)
    write_csv(output_paths["pooled_bin_summary_csv"], POOLED_COLUMNS, pooled_rows)
    write_csv(output_paths["fold_aware_bin_summary_csv"], FOLD_AWARE_COLUMNS, fold_rows)
    write_csv(output_paths["robustness_delta_csv"], DELTA_COLUMNS, delta_rows)
    write_csv(output_paths["paper_table_compact_csv"], COMPACT_COLUMNS, compact_rows)
    write_csv(output_paths["paper_table_delta_compact_csv"], DELTA_COMPACT_COLUMNS, delta_compact_rows)
    write_json(output_paths["diagnostics_json"], diagnostics)
    write_text(output_paths["report_md"], report_md)

    for key, path in output_paths.items():
        rows = diagnostics["output_rows"].get(key)
        cols = diagnostics["output_columns"].get(key)
        if rows is not None and cols is not None:
            print(f"Wrote {path} rows={rows} cols={cols}")
        else:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
