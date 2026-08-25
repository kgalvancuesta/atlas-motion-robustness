#!/usr/bin/env python3
"""Aggregate fold-wise ATLAS CV metrics for one model/augmentation condition.

Examples:
    python3 scripts/aggregate_cv_metrics.py \
      --model_dir runs/swin \
      --run_prefix run_kfold \
      --splits_json splits/atlas_5fold_lesion_quartile_excluding_known_issues.json \
      --out_dir runs/swin/run_kfold_summary
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from split_utils import is_cv_split_payload, load_split_payload


METRIC_KEYS = ["dice", "jaccard", "precision", "recall", "abs_volume_diff_ratio"]
REPORT_FILES = {
    "clean": "test_clean_metrics.json",
    "augmented": "test_augmented_metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fold-wise ATLAS CV evaluation metrics.")
    parser.add_argument("--model_dir", required=True, help="Model root directory, e.g. runs/swin")
    parser.add_argument("--run_prefix", required=True, help="Run prefix, e.g. run_kfold or run_DA_kfold")
    parser.add_argument("--splits_json", required=True, help="Master CV split JSON")
    parser.add_argument("--out_dir", required=True, help="Output summary directory")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean_std(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def get_metric_mean(report: dict[str, Any], metric: str) -> float | None:
    summary = report.get("summary", {})
    metric_summary = summary.get(metric)
    if isinstance(metric_summary, dict) and metric_summary.get("mean") is not None:
        return float(metric_summary["mean"])
    values = [
        float(item[metric])
        for item in report.get("per_subject", [])
        if isinstance(item, dict) and item.get(metric) is not None
    ]
    if not values:
        return None
    return float(statistics.fmean(values))


def per_subject_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for item in report.get("per_subject", []):
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        if not subject:
            continue
        rows[str(subject)] = item
    return rows


def aggregate_per_subject(rows: list[dict[str, Any]], metric: str) -> dict[str, float] | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return mean_std(values)


def coverage_report(subjects: list[str], expected_subjects: set[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for subject in subjects:
        counts[subject] = counts.get(subject, 0) + 1
    duplicates = sorted(subject for subject, count in counts.items() if count > 1)
    observed = set(subjects)
    missing = sorted(expected_subjects - observed)
    unexpected = sorted(observed - expected_subjects)
    return {
        "expected_subjects": len(expected_subjects),
        "observed_subjects": len(observed),
        "duplicate_subjects": duplicates,
        "missing_subjects": missing,
        "unexpected_subjects": unexpected,
        "appears_exactly_once": not duplicates and not missing and not unexpected,
    }


def fold_run_dir(model_dir: Path, run_prefix: str, fold_index: int) -> Path:
    return model_dir / f"{run_prefix}_{fold_index + 1:02d}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_payload = load_split_payload(Path(args.splits_json).resolve())
    if not is_cv_split_payload(split_payload):
        raise SystemExit("Expected a CV master split JSON with top-level 'folds'.")

    folds = split_payload.get("folds", [])
    expected_test_subjects = {
        str(subject)
        for fold in folds
        for subject in fold.get("test_ids", [])
    }

    warnings: list[str] = []
    per_fold_rows: list[dict[str, Any]] = []
    pooled_subject_rows = {"clean": [], "augmented": []}
    pooled_subject_ids = {"clean": [], "augmented": []}
    robustness_subject_deltas: list[float] = []

    for fold in folds:
        fold_index = int(fold["fold_index"])
        run_dir = fold_run_dir(model_dir, args.run_prefix, fold_index)
        fold_entry: dict[str, Any] = {
            "fold_index": fold_index,
            "run_dir": str(run_dir),
            "expected_test_subjects": len(fold.get("test_ids", [])),
        }

        clean_report = None
        augmented_report = None
        for kind, filename in REPORT_FILES.items():
            report_path = run_dir / "eval_local" / filename
            fold_entry[f"{kind}_report_path"] = str(report_path)
            if not report_path.exists():
                warnings.append(f"Missing {kind} metrics for fold {fold_index}: {report_path}")
                continue
            report = load_json(report_path)
            fold_entry[f"{kind}_summary"] = report.get("summary", {})
            if kind == "clean":
                clean_report = report
            else:
                augmented_report = report

        clean_subject_rows = per_subject_map(clean_report or {})
        augmented_subject_rows = per_subject_map(augmented_report or {})
        expected_fold_subjects = {str(subject) for subject in fold.get("test_ids", [])}
        fold_entry["subject_checks"] = {
            "clean": coverage_report(sorted(clean_subject_rows), expected_fold_subjects),
            "augmented": coverage_report(sorted(augmented_subject_rows), expected_fold_subjects),
        }

        for kind, subject_rows in (("clean", clean_subject_rows), ("augmented", augmented_subject_rows)):
            pooled_subject_ids[kind].extend(sorted(subject_rows))
            pooled_subject_rows[kind].extend(subject_rows.values())

        common_subjects = sorted(set(clean_subject_rows) & set(augmented_subject_rows))
        fold_entry["robustness_delta_dice_mean"] = None
        if common_subjects:
            deltas = [
                float(clean_subject_rows[subject]["dice"]) - float(augmented_subject_rows[subject]["dice"])
                for subject in common_subjects
                if clean_subject_rows[subject].get("dice") is not None and augmented_subject_rows[subject].get("dice") is not None
            ]
            if deltas:
                fold_entry["robustness_delta_dice_mean"] = float(statistics.fmean(deltas))
                robustness_subject_deltas.extend(deltas)

        for metric in METRIC_KEYS:
            clean_value = get_metric_mean(clean_report or {}, metric)
            aug_value = get_metric_mean(augmented_report or {}, metric)
            fold_entry[f"clean_{metric}_mean"] = clean_value
            fold_entry[f"augmented_{metric}_mean"] = aug_value

        if fold_entry["clean_dice_mean"] is not None and fold_entry["augmented_dice_mean"] is not None:
            fold_entry["fold_robustness_delta_mean"] = (
                fold_entry["clean_dice_mean"] - fold_entry["augmented_dice_mean"]
            )
        else:
            fold_entry["fold_robustness_delta_mean"] = None

        per_fold_rows.append(fold_entry)

    fold_level_summary: dict[str, Any] = {}
    for kind in REPORT_FILES:
        fold_level_summary[kind] = {}
        for metric in METRIC_KEYS:
            values = [
                row[f"{kind}_{metric}_mean"]
                for row in per_fold_rows
                if row.get(f"{kind}_{metric}_mean") is not None
            ]
            fold_level_summary[kind][metric] = mean_std(values)

    robustness_fold_values = [
        row["fold_robustness_delta_mean"]
        for row in per_fold_rows
        if row.get("fold_robustness_delta_mean") is not None
    ]
    fold_level_summary["robustness_delta_dice"] = {
        "fold_mean_summary": mean_std(robustness_fold_values),
        "pooled_subject_summary": mean_std(robustness_subject_deltas),
    }

    pooled_summary: dict[str, Any] = {}
    for kind, rows in pooled_subject_rows.items():
        pooled_summary[kind] = {metric: aggregate_per_subject(rows, metric) for metric in METRIC_KEYS}

    coverage_checks = {
        "clean": coverage_report(pooled_subject_ids["clean"], expected_test_subjects),
        "augmented": coverage_report(pooled_subject_ids["augmented"], expected_test_subjects),
    }

    payload = {
        "config": {
            "model_dir": str(model_dir),
            "run_prefix": args.run_prefix,
            "splits_json": str(Path(args.splits_json).resolve()),
            "out_dir": str(out_dir),
        },
        "expected_test_subjects": len(expected_test_subjects),
        "per_fold": per_fold_rows,
        "fold_level_summary": fold_level_summary,
        "pooled_per_subject_summary": pooled_summary,
        "coverage_checks": coverage_checks,
        "warnings": warnings,
    }

    json_path = out_dir / "cv_metrics_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = out_dir / "cv_metrics_summary.csv"
    csv_headers = [
        "fold_index",
        "run_dir",
        "expected_test_subjects",
        "clean_subjects",
        "augmented_subjects",
        "clean_dice_mean",
        "augmented_dice_mean",
        "fold_robustness_delta_mean",
        "clean_jaccard_mean",
        "augmented_jaccard_mean",
        "clean_precision_mean",
        "augmented_precision_mean",
        "clean_recall_mean",
        "augmented_recall_mean",
        "clean_abs_volume_diff_ratio_mean",
        "augmented_abs_volume_diff_ratio_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in per_fold_rows:
            writer.writerow(
                {
                    "fold_index": row["fold_index"],
                    "run_dir": row["run_dir"],
                    "expected_test_subjects": row["expected_test_subjects"],
                    "clean_subjects": row["subject_checks"]["clean"]["observed_subjects"],
                    "augmented_subjects": row["subject_checks"]["augmented"]["observed_subjects"],
                    "clean_dice_mean": row["clean_dice_mean"],
                    "augmented_dice_mean": row["augmented_dice_mean"],
                    "fold_robustness_delta_mean": row["fold_robustness_delta_mean"],
                    "clean_jaccard_mean": row["clean_jaccard_mean"],
                    "augmented_jaccard_mean": row["augmented_jaccard_mean"],
                    "clean_precision_mean": row["clean_precision_mean"],
                    "augmented_precision_mean": row["augmented_precision_mean"],
                    "clean_recall_mean": row["clean_recall_mean"],
                    "augmented_recall_mean": row["augmented_recall_mean"],
                    "clean_abs_volume_diff_ratio_mean": row["clean_abs_volume_diff_ratio_mean"],
                    "augmented_abs_volume_diff_ratio_mean": row["augmented_abs_volume_diff_ratio_mean"],
                }
            )

    markdown_lines = [
        "# CV Metrics Summary",
        "",
        f"- Model directory: `{model_dir}`",
        f"- Run prefix: `{args.run_prefix}`",
        f"- Expected pooled test subjects: `{len(expected_test_subjects)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Per-fold Summary",
        "",
    ]
    fold_table_rows = []
    for row in per_fold_rows:
        fold_table_rows.append(
            [
                str(row["fold_index"]),
                str(row["subject_checks"]["clean"]["observed_subjects"]),
                f"{row['clean_dice_mean']:.4f}" if row["clean_dice_mean"] is not None else "NA",
                f"{row['augmented_dice_mean']:.4f}" if row["augmented_dice_mean"] is not None else "NA",
                f"{row['fold_robustness_delta_mean']:.4f}" if row["fold_robustness_delta_mean"] is not None else "NA",
            ]
        )
    markdown_lines.append(
        render_table(
            ["Fold", "Test Subjects", "Clean Dice", "Augmented Dice", "Robustness Delta"],
            fold_table_rows,
        )
    )
    markdown_lines.extend(
        [
            "",
            "## Fold-level Aggregates",
            "",
        ]
    )
    aggregate_rows = []
    for metric in ("dice", "jaccard", "precision", "recall", "abs_volume_diff_ratio"):
        clean_stats = fold_level_summary["clean"].get(metric)
        aug_stats = fold_level_summary["augmented"].get(metric)
        aggregate_rows.append(
            [
                metric,
                "NA" if clean_stats is None else f"{clean_stats['mean']:.4f} +/- {clean_stats['std']:.4f}",
                "NA" if aug_stats is None else f"{aug_stats['mean']:.4f} +/- {aug_stats['std']:.4f}",
            ]
        )
    markdown_lines.append(render_table(["Metric", "Clean", "Augmented"], aggregate_rows))
    markdown_lines.extend(["", "## Coverage Checks", ""])
    coverage_rows = []
    for kind in ("clean", "augmented"):
        check = coverage_checks[kind]
        coverage_rows.append(
            [
                kind,
                str(check["observed_subjects"]),
                str(len(check["duplicate_subjects"])),
                str(len(check["missing_subjects"])),
                "yes" if check["appears_exactly_once"] else "no",
            ]
        )
    markdown_lines.append(
        render_table(["Set", "Observed", "Duplicates", "Missing", "Each Subject Exactly Once"], coverage_rows)
    )
    if warnings:
        markdown_lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            markdown_lines.append(f"- {warning}")

    md_path = out_dir / "cv_metrics_summary.md"
    md_path.write_text("\n".join(markdown_lines) + "\n")

    has_errors = bool(warnings)
    if has_errors:
        print(f"Wrote summary with warnings: {md_path}")
        return 1

    print(f"Wrote summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
