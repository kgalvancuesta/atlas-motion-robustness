#!/usr/bin/env python3
"""Derive lesion-size FNR/PPV from the published binary Dice and voxel counts.

TP = round(Dice * (predicted + reference) / 2), FP = predicted - TP,
FN = reference - TP. This requires unsmoothed binary Dice, not rounded table
values. Means weight subjects equally; PPV excludes empty predictions, while
FNR includes them. Zero overlap is a case-level reference miss and can coexist
with false positives elsewhere; it is not a connected-lesion detection metric.
Uses only the Python standard library; no images or model inference are needed.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / (
    "scripts/supporting_experiments/statistical_inference/server_reports/"
    "det-fix_v1.1.0/atlas_subject_level_metrics_long.csv"
)


def analyze(rows):
    groups = defaultdict(list)
    derived = []
    seen = set()
    for row in rows:
        key = tuple(row[field] for field in (
            "architecture", "training_condition", "evaluation_condition", "subject_id"
        ))
        if key in seen:
            raise ValueError(f"Duplicate case: {key}")
        seen.add(key)
        if row["source_metric_mode"] != "deterministic_experiment_binary_dice":
            raise ValueError(f"Expected unsmoothed deterministic binary Dice: {key}")
        pred = int(row["predicted_voxel_count"])
        truth = int(row["ground_truth_voxel_count"])
        dice = float(row["dice"])
        if pred < 0 or truth <= 0 or not math.isfinite(dice) or not 0 <= dice <= 1:
            raise ValueError(f"Invalid counts or Dice: {key}")
        raw_tp = dice * (pred + truth) / 2
        tp = round(raw_tp)
        if abs(raw_tp - tp) >= 1e-5 or not 0 <= tp <= min(pred, truth):
            raise ValueError(f"Dice/counts cannot recover integer overlap: {key}")
        result = {
            "subject_id": row["subject_id"], "fold": row["fold"],
            "architecture": row["architecture"],
            "training_condition": row["training_condition"],
            "evaluation_condition": row["evaluation_condition"],
            "lesion_size_bin": row["lesion_size_bin"],
            "lesion_voxel_range": row["lesion_voxel_range"],
            "dice": dice, "predicted_voxels": pred, "target_voxels": truth,
            "tp": tp, "fp": pred - tp, "fn": truth - tp,
            "voxel_fnr": (truth - tp) / truth,
            "voxel_ppv": tp / pred if pred else None,
            "empty_prediction": int(pred == 0), "zero_overlap": int(tp == 0),
            "predicted_gt_target": int(pred > truth),
            "catastrophic_failure": int(dice < 0.1),
        }
        derived.append(result)
        groups[key[:3] + (row["lesion_size_bin"],)].append(result)
    if not derived:
        raise ValueError("Input contains no cases")

    summary = []
    for key, cases in sorted(groups.items()):
        if len({r["lesion_voxel_range"] for r in cases}) != 1:
            raise ValueError(f"Inconsistent lesion-size range: {key}")
        for subset, selected in (
            ("all_cases", cases),
            ("dice_lt_0_1", [r for r in cases if r["catastrophic_failure"]]),
        ):
            ppv = [r["voxel_ppv"] for r in selected if r["voxel_ppv"] is not None]
            n = len(selected)
            zero_overlap = sum(r["zero_overlap"] for r in selected)
            summary.append({
                "architecture": key[0], "training_condition": key[1],
                "evaluation_condition": key[2], "lesion_size_bin": key[3],
                "lesion_voxel_range": cases[0]["lesion_voxel_range"],
                "subset": subset, "n_stratum": len(cases), "n": n,
                "macro_voxel_fnr": statistics.mean(r["voxel_fnr"] for r in selected) if n else None,
                "macro_voxel_ppv_nonempty": statistics.mean(ppv) if ppv else None,
                "ppv_defined_n": len(ppv),
                "empty_predictions": sum(r["empty_prediction"] for r in selected),
                "zero_overlap_cases": zero_overlap,
                "predicted_gt_target_cases": sum(r["predicted_gt_target"] for r in selected),
                "zero_overlap_fraction": zero_overlap / n if n else None,
            })
    return derived, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE,
                        help="Published atlas_subject_level_metrics_long.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/fnr_ppv",
                        help="New directory for generated CSVs (default: ignored outputs/fnr_ppv)")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        derived, summary = analyze(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("lesion_failure_case_metrics.csv", derived),
        ("lesion_size_fnr_ppv.csv", summary),
    ):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    print(f"Input SHA-256: {hashlib.sha256(args.input.read_bytes()).hexdigest()}")
    print(f"Cases: {len(derived)}; stratum/subset summaries: {len(summary)}")
    focus = [r for r in summary if r["training_condition"] == "Augmented"
             and r["evaluation_condition"] == "artifact" and r["lesion_size_bin"] == "1"
             and r["subset"] == "dice_lt_0_1"]
    if focus:
        print("Smallest-quartile failures, augmented training, artifact evaluation:")
        for field in ("zero_overlap_fraction", "macro_voxel_fnr", "macro_voxel_ppv_nonempty"):
            values = [100 * r[field] for r in focus if r[field] is not None]
            if values:
                print(f"  {field}: {min(values):.1f}--{max(values):.1f}% across architectures")


if __name__ == "__main__":
    main()
