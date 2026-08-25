#!/usr/bin/env python3
"""
Create deterministic lesion-volume-stratified 5-fold CV split JSONs for ATLAS.

Command:
    python3 scripts/supporting_experiments/atlas_lesion_size_stratification/scripts/create_cv_splits.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_lesion_volume_stratification import (
    analyze_subjects,
    assign_bins,
    dedupe_preserve_order,
    format_edges,
    half_open_range_text,
    load_subject_entries,
    quantile_edges,
)

OUTER_SPLITS = 5
INNER_VAL_FRACTION = 0.20
RANDOM_STATE = 9001
SCRIPT_NAME = (
    "scripts/supporting_experiments/atlas_lesion_size_stratification/"
    "scripts/create_cv_splits.py"
)
MASTER_JSON_NAME = "atlas_5fold_lesion_quartile_excluding_known_issues.json"
REPORT_NAME = "atlas_5fold_lesion_quartile_excluding_known_issues_report.md"

EXCLUDED_SUBJECTS: dict[str, str] = {
    "sub-r039s002": (
        "Excluded from CV generation: ATLAS issue subject with invalid/ambiguous lesion mask values "
        "(max ~= 0.01 instead of a valid binary lesion-mask convention)."
    ),
    "sub-r009s003": (
        "Excluded from CV generation: known ATLAS labeling issue and manual review confirmed unreliable "
        "ground truth."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic 5-fold lesion-volume-stratified CV split JSONs for ATLAS."
    )
    parser.add_argument(
        "--splits_json",
        default=None,
        help=(
            "Optional legacy split JSON used as the source of labeled subject IDs. "
            "By default, subjects are discovered from the labeled ATLAS derivative."
        ),
    )
    parser.add_argument(
        "--subjects_csv",
        default=(
            "outputs/supporting_experiments/atlas_lesion_size_stratification/"
            "stratification_analysis/lesion_volume_subjects.csv"
        ),
        help="Per-subject lesion-volume CSV from the stratification analysis step.",
    )
    parser.add_argument(
        "--out_dir",
        default="splits",
        help="Directory for the generated master JSON, per-fold JSONs, and report.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Dataset root containing train/ and test/. Used only if lesion volumes must be recomputed.",
    )
    parser.add_argument(
        "--gt_deriv",
        default=None,
        help="GT derivative root. Used only if lesion volumes must be recomputed.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Binary lesion threshold used when recomputing lesion volumes.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def resolve_subjects_csv(path_value: str) -> Path:
    return resolve_path(path_value)


def resolve_out_dir(path_value: str) -> Path:
    return resolve_path(path_value)


def resolve_gt_deriv(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    gt_deriv = Path(args.gt_deriv) if args.gt_deriv else data_root / "train" / "derivatives" / "ATLAS"
    if not gt_deriv.is_absolute():
        gt_deriv = (repo_root() / gt_deriv).resolve()
    return gt_deriv


def discover_subject_entries(gt_deriv: Path) -> list[tuple[str, str, str]]:
    entries = [
        (subject_dir.name, "labeled", "data_discovery")
        for subject_dir in sorted(gt_deriv.glob("sub-*"))
        if subject_dir.is_dir()
    ]
    if not entries:
        raise SystemExit(f"ERROR: no labeled subject directories found under {gt_deriv}")
    return entries


def parse_int_or_none(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(float(raw))


def load_subject_rows_from_csv(subjects_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "subject_id",
        "split_source",
        "split_key_used",
        "mask_path",
        "lesion_voxels",
        "status",
        "error",
    }
    with subjects_csv.open() as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise SystemExit(
                f"ERROR: lesion-volume CSV is missing required columns: {missing}. "
                "Rerun the lesion-volume analysis script in this directory first."
            )
        for row in reader:
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "split_source": row["split_source"],
                    "split_key_used": row["split_key_used"],
                    "mask_path": row["mask_path"],
                    "lesion_voxels": parse_int_or_none(row.get("lesion_voxels")),
                    "status": row["status"],
                    "error": row["error"] or None,
                }
            )
    return sorted(rows, key=lambda item: item["subject_id"])


def recompute_subject_rows(
    args: argparse.Namespace,
    entries: list[tuple[str, str, str]],
    split_warnings: list[str],
    totals_by_split: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_deriv = resolve_gt_deriv(args)
    records, analysis_warnings = analyze_subjects(gt_deriv, entries, args.mask_threshold)
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.subject_id):
        rows.append(
            {
                "subject_id": record.subject_id,
                "split_source": record.split_source,
                "split_key_used": record.split_key_used,
                "mask_path": record.mask_path,
                "lesion_voxels": record.lesion_voxels,
                "status": record.status,
                "error": record.error,
            }
        )
    return rows, {
        "source_kind": "recomputed",
        "source_path": str(gt_deriv),
        "split_warnings": split_warnings,
        "analysis_warnings": analysis_warnings,
        "subjects_per_split_from_split_file": totals_by_split,
    }


def load_or_recompute_subject_rows(
    args: argparse.Namespace,
    required_subjects: set[str],
    entries: list[tuple[str, str, str]],
    split_warnings: list[str],
    totals_by_split: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subjects_csv = resolve_subjects_csv(args.subjects_csv)
    if subjects_csv.exists():
        rows = load_subject_rows_from_csv(subjects_csv)
        ok_subjects = {row["subject_id"] for row in rows if row["status"] == "ok" and row["lesion_voxels"] is not None}
        if required_subjects.issubset(ok_subjects):
            return rows, {
                "source_kind": "csv",
                "source_path": str(subjects_csv),
                "split_warnings": [],
                "analysis_warnings": [],
            }
    return recompute_subject_rows(args, entries, split_warnings, totals_by_split)


def counter_list(labels: np.ndarray, num_bins: int) -> list[int]:
    counts = [0] * num_bins
    unique_labels, unique_counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels.tolist(), unique_counts.tolist()):
        counts[int(label)] = int(count)
    return counts


def count_dict_for_indices(labels: np.ndarray, indices: np.ndarray, bin_ranges: list[str]) -> dict[str, int]:
    subset_counts = counter_list(labels[indices], len(bin_ranges))
    return {bin_ranges[i]: subset_counts[i] for i in range(len(bin_ranges))}


def stratified_fold_indices(labels: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(class_indices)
        base_size, remainder = divmod(int(shuffled.shape[0]), n_splits)
        chunk_sizes = [base_size + 1] * remainder + [base_size] * (n_splits - remainder)
        target_folds = sorted(range(n_splits), key=lambda idx: (fold_sizes[idx], idx))
        start = 0
        for chunk_size, fold_idx in zip(chunk_sizes, target_folds):
            chunk = shuffled[start : start + chunk_size]
            start += chunk_size
            folds[fold_idx].extend(int(i) for i in chunk.tolist())
            fold_sizes[fold_idx] += int(chunk_size)
    return [np.asarray(sorted(fold), dtype=np.int32) for fold in folds]


def complement_indices(total_size: int, held_out: np.ndarray) -> np.ndarray:
    mask = np.ones(total_size, dtype=bool)
    mask[held_out] = False
    return np.flatnonzero(mask)


def stratified_shuffle_split_indices(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    n_total = int(labels.shape[0])
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    n_test = int(math.ceil(n_total * test_size))
    rng = np.random.default_rng(seed)
    unique_labels = sorted(np.unique(labels).tolist())

    shuffled_by_label: dict[int, np.ndarray] = {}
    base_allocations: list[int] = []
    fractional_parts: list[float] = []
    label_sizes: list[int] = []

    for label in unique_labels:
        class_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(class_indices)
        shuffled_by_label[label] = shuffled
        ideal = class_indices.shape[0] * n_test / n_total
        base = int(math.floor(ideal))
        base_allocations.append(base)
        fractional_parts.append(ideal - base)
        label_sizes.append(int(class_indices.shape[0]))

    remainder = n_test - sum(base_allocations)
    order = sorted(
        range(len(unique_labels)),
        key=lambda idx: (fractional_parts[idx], label_sizes[idx], -unique_labels[idx]),
        reverse=True,
    )
    allocations = list(base_allocations)
    for idx in order[:remainder]:
        allocations[idx] += 1

    test_indices: list[int] = []
    train_indices: list[int] = []
    for allocation, label in zip(allocations, unique_labels):
        shuffled = shuffled_by_label[label]
        test_part = shuffled[:allocation]
        train_part = shuffled[allocation:]
        test_indices.extend(int(i) for i in test_part.tolist())
        train_indices.extend(int(i) for i in train_part.tolist())

    return (
        np.asarray(sorted(train_indices), dtype=np.int32),
        np.asarray(sorted(test_indices), dtype=np.int32),
    )


def assert_no_duplicates(values: list[str], split_name: str, fold_index: int) -> None:
    if len(values) != len(set(values)):
        raise AssertionError(f"Duplicate subject IDs found in {split_name} for fold {fold_index}.")


def verify_reasonable_balance(
    folds: list[dict[str, Any]],
    bin_ranges: list[str],
) -> list[str]:
    checks: list[str] = []
    test_sizes = [fold["counts"]["test"] for fold in folds]
    val_sizes = [fold["counts"]["val"] for fold in folds]
    if max(test_sizes) - min(test_sizes) > 1:
        raise AssertionError(f"Outer test fold sizes are not balanced: {test_sizes}")
    if max(val_sizes) - min(val_sizes) > 1:
        raise AssertionError(f"Validation fold sizes are not balanced: {val_sizes}")
    checks.append(f"Outer test fold sizes differ by at most 1 subject: {test_sizes}")
    checks.append(f"Validation sizes differ by at most 1 subject: {val_sizes}")

    for split_name in ["test", "val"]:
        for bin_range in bin_ranges:
            per_fold = [fold["stratification_counts"][split_name][bin_range] for fold in folds]
            if max(per_fold) - min(per_fold) > 1:
                raise AssertionError(
                    f"Stratification counts for {split_name} and bin {bin_range} are not balanced: {per_fold}"
                )
        checks.append(f"{split_name.title()} stratification counts differ by at most 1 per bin across folds.")
    return checks


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root().resolve()))
    except Exception:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def build_report_lines(
    metadata: dict[str, Any],
    folds: list[dict[str, Any]],
    verification_checks: list[str],
) -> list[str]:
    lines: list[str] = []
    lines.append("# ATLAS 5-Fold Lesion-Quartile CV Splits")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    subject_id_source = metadata["subject_id_source"]
    lines.append(
        f"- Subject ID source: `{subject_id_source['path']}` ({subject_id_source['kind']})"
    )
    lines.append(f"- Total labeled subjects before exclusion: `{metadata['total_subjects_before_exclusion']}`")
    lines.append(f"- Total subjects after exclusion: `{metadata['total_subjects_after_exclusion']}`")
    lines.append(f"- Stratification scheme: `{metadata['stratification_scheme']}`")
    lines.append(f"- Quartile edges after exclusion: `{format_edges(metadata['quartile_edges'])}`")
    lines.append(f"- Random state: `{metadata['random_state']}`")
    lines.append("")
    lines.append("## Exclusions")
    lines.append("")
    for item in metadata["excluded_subjects"]:
        lines.append(f"- `{item['subject_id']}`: {item['reason']}")
    lines.append("")
    lines.append("## Per-Fold Counts")
    lines.append("")
    lines.append("| Fold | Train | Val | Test |")
    lines.append("| --- | ---: | ---: | ---: |")
    for fold in folds:
        lines.append(
            f"| {fold['fold_index']} | {fold['counts']['train']} | {fold['counts']['val']} | {fold['counts']['test']} |"
        )
    lines.append("")
    lines.append("## Per-Fold Stratification Counts")
    lines.append("")
    lines.append("| Fold | Train bins | Val bins | Test bins |")
    lines.append("| --- | --- | --- | --- |")
    for fold in folds:
        lines.append(
            f"| {fold['fold_index']} | {fold['stratification_counts']['train']} | "
            f"{fold['stratification_counts']['val']} | {fold['stratification_counts']['test']} |"
        )
    lines.append("")
    lines.append("## Verification Checks Passed")
    lines.append("")
    for check in verification_checks:
        lines.append(f"- {check}")
    lines.append("")
    lines.append("## Usage Note")
    lines.append("")
    lines.append(
        "Use this same split file for every model architecture to ensure paired comparisons across architectures."
    )
    return lines


def main() -> None:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.splits_json:
        splits_json = resolve_path(args.splits_json)
        entries, split_warnings, totals_by_split = load_subject_entries(splits_json)
        subject_id_source = {
            "kind": "split_json",
            "path": relative_to_repo(splits_json),
        }
    else:
        gt_deriv = resolve_gt_deriv(args)
        entries = discover_subject_entries(gt_deriv)
        split_warnings = []
        totals_by_split = {"labeled": len(entries)}
        subject_id_source = {
            "kind": "bids_derivative",
            "path": relative_to_repo(gt_deriv),
        }
    all_subjects = sorted(subject_id for subject_id, _split_source, _split_key_used in entries)
    all_subject_set = set(all_subjects)

    missing_excluded = [subject_id for subject_id in EXCLUDED_SUBJECTS if subject_id not in all_subject_set]
    if missing_excluded:
        raise SystemExit(f"ERROR: excluded subjects not found in source split file: {missing_excluded}")

    included_subjects = sorted(subject_id for subject_id in all_subjects if subject_id not in EXCLUDED_SUBJECTS)
    included_subject_set = set(included_subjects)

    subject_rows, subject_source_meta = load_or_recompute_subject_rows(
        args,
        included_subject_set,
        entries,
        split_warnings,
        totals_by_split,
    )
    row_by_subject = {
        row["subject_id"]: row
        for row in subject_rows
        if row["status"] == "ok" and row["lesion_voxels"] is not None
    }
    missing_rows = sorted(subject_id for subject_id in included_subjects if subject_id not in row_by_subject)
    if missing_rows:
        raise SystemExit(
            "ERROR: missing readable lesion-volume rows for included subjects. "
            f"First missing IDs: {missing_rows[:10]}"
        )

    subject_ids = sorted(included_subjects)
    lesion_voxels = np.asarray([int(row_by_subject[subject_id]["lesion_voxels"]) for subject_id in subject_ids], dtype=np.int64)
    quartile_edges, quartile_warnings = quantile_edges(lesion_voxels, 4)
    if len(quartile_edges) != 5:
        raise SystemExit(
            "ERROR: quartile edge computation did not produce 4 usable bins after exclusion. "
            f"Edges={quartile_edges} warnings={quartile_warnings}"
        )

    strat_labels = assign_bins(lesion_voxels, quartile_edges)
    bin_ranges = [half_open_range_text(i, quartile_edges) for i in range(len(quartile_edges) - 1)]
    full_bin_counts = counter_list(strat_labels, len(bin_ranges))

    outer_test_folds = stratified_fold_indices(strat_labels, OUTER_SPLITS, RANDOM_STATE)
    folds: list[dict[str, Any]] = []
    per_fold_json_paths: list[str] = []

    for fold_index, test_idx in enumerate(outer_test_folds):
        train_val_idx = complement_indices(len(subject_ids), test_idx)
        inner_train_local, inner_val_local = stratified_shuffle_split_indices(
            strat_labels[train_val_idx],
            test_size=INNER_VAL_FRACTION,
            seed=RANDOM_STATE + fold_index,
        )
        train_idx = train_val_idx[inner_train_local]
        val_idx = train_val_idx[inner_val_local]

        train_ids = [subject_ids[i] for i in train_idx.tolist()]
        val_ids = [subject_ids[i] for i in val_idx.tolist()]
        test_ids = [subject_ids[i] for i in test_idx.tolist()]

        assert_no_duplicates(train_ids, "train_ids", fold_index)
        assert_no_duplicates(val_ids, "val_ids", fold_index)
        assert_no_duplicates(test_ids, "test_ids", fold_index)

        fold_payload = {
            "fold_index": fold_index,
            "train_ids": sorted(train_ids),
            "val_ids": sorted(val_ids),
            "test_ids": sorted(test_ids),
            "counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "stratification_counts": {
                "bin_ranges": bin_ranges,
                "train": count_dict_for_indices(strat_labels, train_idx, bin_ranges),
                "val": count_dict_for_indices(strat_labels, val_idx, bin_ranges),
                "test": count_dict_for_indices(strat_labels, test_idx, bin_ranges),
            },
            "inner_random_state": RANDOM_STATE + fold_index,
        }
        folds.append(fold_payload)

        per_fold_path = out_dir / f"fold_{fold_index}.json"
        write_json(per_fold_path, fold_payload)
        per_fold_json_paths.append(relative_to_repo(per_fold_path))

    verification_checks: list[str] = []

    excluded_ids = set(EXCLUDED_SUBJECTS)
    verification_checks.append("No excluded subject appears in any generated train/val/test split.")
    verification_checks.append("No duplicate subject IDs appear within any generated train/val/test split.")
    for fold in folds:
        fold_index = int(fold["fold_index"])
        train_set = set(fold["train_ids"])
        val_set = set(fold["val_ids"])
        test_set = set(fold["test_ids"])

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise AssertionError(f"Train/val/test overlap detected in fold {fold_index}.")
        if (train_set | val_set | test_set) != included_subject_set:
            raise AssertionError(f"Fold {fold_index} does not cover the full included subject universe.")
        if train_set & excluded_ids or val_set & excluded_ids or test_set & excluded_ids:
            raise AssertionError(f"Excluded subject detected in fold {fold_index}.")
        if fold["counts"]["train"] + fold["counts"]["val"] + fold["counts"]["test"] != len(included_subjects):
            raise AssertionError(f"Fold {fold_index} counts do not sum to the full included subject set.")
        verification_checks.append(f"Fold {fold_index}: train/val/test are disjoint and cover the full included subject set.")

    outer_test_counter: Counter[str] = Counter()
    for fold in folds:
        outer_test_counter.update(fold["test_ids"])
    if sorted(outer_test_counter.keys()) != subject_ids:
        raise AssertionError("Outer test folds do not cover the full included subject universe.")
    bad_test_frequency = {subject_id: count for subject_id, count in outer_test_counter.items() if count != 1}
    if bad_test_frequency:
        raise AssertionError(f"Each included subject must appear in outer test exactly once: {bad_test_frequency}")
    verification_checks.append("All folds use the same included subject universe.")
    verification_checks.append("Each included subject appears in outer test exactly once across folds.")

    verification_checks.extend(verify_reasonable_balance(folds, bin_ranges))

    metadata = {
        "subject_id_source": subject_id_source,
        "total_subjects_before_exclusion": len(all_subjects),
        "excluded_subjects": [
            {"subject_id": subject_id, "reason": reason}
            for subject_id, reason in EXCLUDED_SUBJECTS.items()
        ],
        "total_subjects_after_exclusion": len(subject_ids),
        "n_splits": OUTER_SPLITS,
        "outer_cv_method": (
            "Deterministic stratified outer split equivalent to "
            "StratifiedKFold(n_splits=5, shuffle=True, random_state=9001)."
        ),
        "inner_val_method": (
            "Deterministic stratified inner split equivalent to "
            "StratifiedShuffleSplit(test_size=0.20, random_state=9001 + fold_index)."
        ),
        "stratification_target": "lesion_voxels",
        "stratification_scheme": "quantile_k4",
        "quartile_edges": [float(edge) for edge in quartile_edges],
        "quartile_bin_ranges": bin_ranges,
        "quartile_bin_counts": {bin_ranges[i]: full_bin_counts[i] for i in range(len(bin_ranges))},
        "random_state": RANDOM_STATE,
        "created_by_script": SCRIPT_NAME,
        "lesion_volume_source": {
            "kind": subject_source_meta["source_kind"],
            "path": relative_to_repo(Path(subject_source_meta["source_path"])),
        },
        "per_fold_json_paths": per_fold_json_paths,
    }

    warnings = dedupe_preserve_order(
        split_warnings
        + subject_source_meta.get("split_warnings", [])
        + subject_source_meta.get("analysis_warnings", [])
        + quartile_warnings
    )

    master_payload = {
        "metadata": metadata,
        "folds": folds,
        "verification_checks_passed": verification_checks,
        "warnings": warnings,
    }

    master_json_path = out_dir / MASTER_JSON_NAME
    report_path = out_dir / REPORT_NAME
    write_json(master_json_path, master_payload)
    report_lines = build_report_lines(metadata, folds, verification_checks)
    if warnings:
        report_lines.append("")
        report_lines.append("## Warnings")
        report_lines.append("")
        for warning in warnings:
            report_lines.append(f"- {warning}")
    report_path.write_text("\n".join(report_lines))

    print(f"master_json={master_json_path}")
    print(f"report_md={report_path}")
    print(f"included_subject_count={len(subject_ids)}")
    print(f"excluded_subjects={list(EXCLUDED_SUBJECTS.keys())}")
    print(f"quartile_edges={format_edges(metadata['quartile_edges'])}")
    for fold in folds:
        print(
            f"fold_{fold['fold_index']}_counts="
            f"train:{fold['counts']['train']},val:{fold['counts']['val']},test:{fold['counts']['test']}"
        )
    print("validation_checks_passed=True")


if __name__ == "__main__":
    main()
