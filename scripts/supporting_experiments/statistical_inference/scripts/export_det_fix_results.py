#!/usr/bin/env python3
"""Export verified deterministic metrics into the paper repository's run layout.

This copies lightweight evidence only. It never loads checkpoints, runs inference,
changes the source experiment, or invents unavailable segmentation metrics.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def export(source, output, split_path, permanent_root):
    require(not output.exists(), f"Output must be new: {output}")
    definition = read(source / "definition/experiment.json")
    require(
        definition.get("schema_version") == "atlas.det-fix_v1.1.0.experiment"
        and not definition.get("publication_view"),
        "Source must be a v1.1.0 execution definition; use the pinned exporter for other schemas",
    )
    require(
        sha(split_path) == definition["dataset"]["split_sha256"],
        "Split checksum mismatch",
    )
    models = definition["configuration"]["models"]
    regimes = definition["configuration"]["training_regimes"]
    folds = definition["configuration"]["folds"]
    require(
        set(models) == {"base_cnn", "mednext", "uxnet", "swin"}, "Unexpected models"
    )
    require(
        set(regimes) == {"standard", "augmented"} and folds == list(range(5)),
        "Incomplete matrix",
    )
    require(
        definition["challenge"]["replicate_ids"] == [0],
        "This export expects one corruption replicate",
    )
    challenge = definition["challenge"]["challenge_id"]
    clean_id = hashlib.sha256(
        json.dumps(
            {
                "kind": "clean_control",
                "experiment_id": definition["experiment_id"],
                "source_inventory_id": definition["dataset"]["source_inventory_id"],
                "preprocessing_version": definition["configuration"][
                    "preprocessing_version"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    require(
        sha(source / definition["challenge"]["manifest_relative_path"])
        == definition["challenge"]["manifest_file_sha256"],
        "Challenge checksum mismatch",
    )
    split = read(split_path)
    expected_subjects = {
        int(f["fold_index"]): set(f["test_ids"]) for f in split["folds"]
    }
    task_metadata = {}
    for model, regime, fold in itertools.product(models, regimes, folds):
        p = source / f"checkpoints/{model}/{regime}/fold-{fold}/task.metadata.json"
        meta = read(p)
        require(
            (
                meta["experiment_id"],
                meta["model"],
                meta["training_regime"],
                meta["fold"],
            )
            == (definition["experiment_id"], model, regime, fold),
            f"Task identity mismatch: {p}",
        )
        task_metadata[model, regime, fold] = meta
    analysis_meta = read(source / "metrics/analysis.metadata.json")
    mrart_meta = read(source / "mrart/summary/summary.metadata.json")
    for meta, keys in [
        (
            analysis_meta,
            [
                ("case_metrics_path", "case_metrics_sha256"),
                ("statistics_path", "statistics_sha256"),
            ],
        ),
        (
            mrart_meta,
            [
                ("combined_metrics_path", "combined_metrics_sha256"),
                ("statistics_path", "statistics_sha256"),
            ],
        ),
    ]:
        for path_key, hash_key in keys:
            require(
                sha(source / meta[path_key]) == meta[hash_key],
                f"Summary checksum mismatch: {meta[path_key]}",
            )
    rows = csv_rows(source / "metrics/case_metrics.csv")
    require(len(rows) == 10448, "Expected 10448 ATLAS rows")
    groups = collections.defaultdict(list)
    seen = set()
    provenance = []
    source_inventory = {
        s["subject_id"]: s for s in definition["dataset"]["source_inventory"]
    }
    recipes = {
        (r["fold"], r["subject_id"], r["replicate_id"]): r["recipe_id"]
        for r in read(source / definition["challenge"]["manifest_relative_path"])[
            "recipes"
        ]
    }
    for row in rows:
        model, regime, fold = row["model"], row["training_regime"], int(row["fold"])
        key = (
            model,
            regime,
            fold,
            row["subject_id"],
            row["challenge_id"],
            int(row["corruption_replicate"]),
        )
        require(key not in seen, f"Duplicate ATLAS key: {key}")
        seen.add(key)
        meta = read(source / row["metadata_path"])
        expected_identity = {
            "experiment_id": definition["experiment_id"],
            "fold": fold,
            "corruption_replicate": 0,
            "fold_definition_id": definition["dataset"]["fold_definition_id"],
            "normalization_mode": definition["configuration"]["normalization_mode"],
            "preprocessing_version": definition["configuration"][
                "preprocessing_version"
            ],
            "model_configuration_id": definition["model_configurations"][model][
                "model_configuration_id"
            ],
        }
        require(
            all(meta[k] == v for k, v in expected_identity.items()),
            f"Case differs from immutable definition: {key}",
        )
        require(
            row["challenge_id"] in {clean_id, challenge}, f"Unknown challenge: {key}"
        )
        for field in [
            "experiment_id",
            "model",
            "training_regime",
            "subject_id",
            "challenge_id",
            "normalization_mode",
            "preprocessing_version",
        ]:
            require(
                meta[field] == row[field], f"Case identity mismatch: {field}: {key}"
            )
        require(
            meta["checkpoint_sha256"]
            == task_metadata[model, regime, fold]["checkpoint_sha256"],
            f"Checkpoint mismatch: {key}",
        )
        require(
            meta["source_sha256"] == source_inventory[row["subject_id"]]["sha256"],
            f"Source mismatch: {key}",
        )
        for field, value in meta["metrics"].items():
            if field in row:
                require(
                    float(value) == float(row[field]),
                    f"Case metric mismatch: {field}: {key}",
                )
        dice = float(row["dice"])
        require(math.isfinite(dice) and 0 <= dice <= 1, f"Invalid Dice: {key}")
        condition = "augmented" if row["challenge_id"] == challenge else "clean"
        if condition == "clean":
            require(meta.get("recipe_id") is None, f"Unexpected clean recipe: {key}")
        else:
            require(
                meta["recipe_id"] == recipes[fold, row["subject_id"], 0],
                f"Recipe identity mismatch: {key}",
            )
        groups[model, regime, fold, condition].append(row)
        provenance.append(
            {
                k: row[k]
                for k in [
                    "model",
                    "training_regime",
                    "fold",
                    "subject_id",
                    "challenge_id",
                    "metadata_path",
                ]
            }
            | {
                k: meta.get(k)
                for k in [
                    "source_sha256",
                    "checkpoint_sha256",
                    "recipe_id",
                    "input_array_sha256",
                    "prediction_path",
                    "prediction_sha256",
                ]
            }
            | {"metadata_sha256": sha(source / row["metadata_path"])}
        )
    clean_ids = {r["challenge_id"] for r in rows if r["challenge_id"] != challenge}
    require(clean_ids == {clean_id}, "Unexpected clean challenges")
    for model, regime, fold, condition in itertools.product(
        models, regimes, folds, ["clean", "augmented"]
    ):
        items = groups[model, regime, fold, condition]
        require(
            {r["subject_id"] for r in items} == expected_subjects[fold],
            f"ATLAS subject coverage: {model}/{regime}/{fold}/{condition}",
        )
    mrart = csv_rows(source / "mrart/summary/false_positive_metrics.csv")
    subjects = {
        s["subject_id"] for s in definition["dataset"]["mrart"]["complete_subjects"]
    }
    acquisitions = ["standard", "headmotion1", "headmotion2"]
    expected = set(itertools.product(models, regimes, folds, subjects, acquisitions))
    actual = [
        (
            r["model"],
            r["training_regime"],
            int(r["fold"]),
            r["subject_id"],
            r["acquisition"],
        )
        for r in mrart
    ]
    require(
        len(actual) == len(set(actual))
        and set(actual) == expected
        and len(mrart) == 16800,
        "MR-ART complete matrix mismatch",
    )
    for model, regime, fold in itertools.product(models, regimes, folds):
        meta = read(source / f"mrart/{model}/{regime}/fold-{fold}/task.metadata.json")
        require(
            meta["checkpoint_sha256"]
            == task_metadata[model, regime, fold]["checkpoint_sha256"],
            "MR-ART checkpoint mismatch",
        )
        require(
            sha(source / meta["metrics_path"]) == meta["metrics_sha256"],
            "MR-ART task checksum mismatch",
        )
        task_rows = csv_rows(source / meta["metrics_path"])
        combined = [
            r
            for r in mrart
            if (r["model"], r["training_regime"], int(r["fold"]))
            == (model, regime, fold)
        ]
        require(task_rows == combined, "MR-ART combined rows differ from task rows")

    output.mkdir(parents=True)
    lightweight = output / "runs/det-fix_v1.1.0"
    for name in ["definition", "metrics", "statistics"]:
        shutil.copytree(source / name, lightweight / name)
    shutil.copytree(source / "mrart/summary", lightweight / "mrart_summary")
    write_csv(lightweight / "case_provenance.csv", provenance)
    write_csv(
        lightweight / "checkpoint_provenance.csv",
        [
            {
                "model": model,
                "training_regime": regime,
                "fold": fold,
                "checkpoint_path": task_metadata[model, regime, fold][
                    "checkpoint_path"
                ],
                "checkpoint_sha256": task_metadata[model, regime, fold][
                    "checkpoint_sha256"
                ],
                "training_run_seed": task_metadata[model, regime, fold][
                    "training_run_seed"
                ],
                "training_metadata_sha256": sha(
                    source
                    / f"checkpoints/{model}/{regime}/fold-{fold}/task.metadata.json"
                ),
                "mrart_metadata_path": f"mrart/{model}/{regime}/fold-{fold}/task.metadata.json",
                "mrart_metadata_sha256": sha(
                    source / f"mrart/{model}/{regime}/fold-{fold}/task.metadata.json"
                ),
            }
            for model, regime, fold in itertools.product(models, regimes, folds)
        ],
    )
    for model, regime, fold in itertools.product(models, regimes, folds):
        prefix = "run_kfold" if regime == "standard" else "run_DA_kfold"
        run = output / f"runs/{model}/{prefix}_{fold + 1:02d}"
        original = source / f"checkpoints/{model}/{regime}/fold-{fold}"
        (run / "config").mkdir(parents=True)
        shutil.copy2(
            original / "config/train_config.json", run / "config/train_config.json"
        )
        (run / "logs").mkdir()
        for name in ["train_log.csv", "numerical_summary.json"]:
            if (original / "logs" / name).is_file():
                shutil.copy2(original / "logs" / name, run / "logs" / name)
        write(
            run / "run_status.json",
            {
                "experiment_id": definition["experiment_id"],
                "status": "completed",
                "source_task_metadata_path": f"checkpoints/{model}/{regime}/fold-{fold}/task.metadata.json",
                "source_task_metadata_sha256": sha(original / "task.metadata.json"),
                **task_metadata[model, regime, fold],
            },
        )
        for condition in ["clean", "augmented"]:
            items = sorted(
                groups[model, regime, fold, condition], key=lambda r: r["subject_id"]
            )
            scores = [float(r["dice"]) for r in items]
            per_subject = [
                {
                    "subject": r["subject_id"],
                    "dice": float(r["dice"]),
                    "gt_voxels": int(r["target_voxels"]),
                    "pred_voxels": int(r["predicted_voxels"]),
                    "source_metadata_path": r["metadata_path"],
                }
                for r in items
            ]
            summary = {
                "split_key_used": "test_ids",
                "cv_fold": fold,
                "n_evaluated": len(items),
                "metric_mode": "deterministic_experiment_binary_dice",
                "metric_definition": {
                    "pred_threshold": 0.5,
                    "pred_threshold_comparison": ">=",
                    "gt_threshold": 0.5,
                    "eps": 1e-6,
                },
                "dice": {
                    "mean": statistics.fmean(scores),
                    "std": statistics.pstdev(scores),
                    "min": min(scores),
                    "median": statistics.median(scores),
                    "max": max(scores),
                },
            }
            write(
                run / f"eval_local/test_{condition}_metrics.json",
                {
                    "experiment_id": definition["experiment_id"],
                    "source": "runs/det-fix_v1.1.0/metrics/case_metrics.csv",
                    "summary": summary,
                    "per_subject": per_subject,
                },
            )
            bins = collections.Counter(min(9, int(s * 10)) for s in scores)
            write_csv(
                run / f"eval_local/dice_bins_10pct_{condition}.csv",
                [
                    {
                        "bin_label": f"{i * 10}-{i * 10 + 9 if i < 9 else 100}",
                        "lower_percent": i * 10,
                        "upper_percent": i * 10 + 9 if i < 9 else 100,
                        "count": bins[i],
                    }
                    for i in range(10)
                ],
            )
    for model in models:
        status = [
            task_metadata[model, regime, fold]
            for regime, fold in itertools.product(regimes, folds)
        ]
        write(output / f"runs/{model}/kfold_status.json", status)
        write_csv(
            output / f"runs/{model}/kfold_status.csv",
            [
                {
                    "model": model,
                    "training_regime": r["training_regime"],
                    "fold": r["fold"],
                    "status": "completed",
                    "checkpoint_sha256": r["checkpoint_sha256"],
                    "fallback_count": r.get("fallback_count", 0),
                }
                for r in status
            ],
        )
    compatible = []
    mrart_sources = {
        (s["subject_id"], s["acquisition"]): s
        for s in definition["dataset"]["mrart"]["sources"]
    }
    for row in mrart:
        n, lcc = (
            int(row["predicted_voxels"]),
            int(row["largest_connected_component_voxels_26"]),
        )
        source_scan = mrart_sources[row["subject_id"], row["acquisition"]]
        require(
            source_scan["sha256"] == row["source_sha256"],
            "MR-ART source checksum mismatch",
        )
        voxel_volume = math.prod(source_scan["zooms"])
        require(
            math.isclose(
                float(row["predicted_volume_ml"]), n * voxel_volume / 1000, abs_tol=1e-9
            ),
            "MR-ART volume units mismatch",
        )
        compatible.append(
            {
                **row,
                "architecture": "swin_unetr"
                if row["model"] == "swin"
                else row["model"],
                "training_condition": row["training_regime"],
                "acquisition_type": row["acquisition"],
                "threshold": 0.5,
                "voxel_volume_mm3": voxel_volume,
                "predicted_lesion_volume_ml": float(row["predicted_volume_ml"]),
                "largest_cc_volume_ml": float(row["largest_connected_component_ml_26"]),
                "connected_component_count_26conn": int(
                    row["connected_component_count_26"]
                ),
                "largest_cc_fraction_of_prediction": lcc / n if n else 0.0,
                "scan_positive": int(n > 0),
                "status": "success",
            }
        )
    mrart_path = (
        output
        / "scripts/supporting_experiments/mrart/server_reports/det-fix_v1.1.0/fold_lcc_stats/mrart_fold_lcc_stats_long.csv"
    )
    write_csv(mrart_path, compatible)
    write(
        lightweight / "export_provenance.json",
        {
            "experiment_id": definition["experiment_id"],
            "permanent_source_root": permanent_root,
            "exporter_source_sha256": sha(Path(__file__)),
            "execution_revision": definition["environment_at_prepare"]["repository"],
            "split_sha256": sha(split_path),
            "definition_sha256": sha(source / "definition/experiment.json"),
            "atlas_source_sha256": sha(source / "metrics/case_metrics.csv"),
            "mrart_source_sha256": sha(
                source / "mrart/summary/false_positive_metrics.csv"
            ),
            "atlas_rows": len(rows),
            "mrart_rows": len(mrart),
            "case_metadata_checked": len(provenance),
            "training_tasks_checked": len(task_metadata),
            "missing_metrics_policy": "Unavailable overlap metrics are omitted; no values imputed.",
            "statistics_policy": "Original pipeline statistics retained unchanged (2000 bootstrap draws). Paper tables recomputed separately with the existing paper-analysis code (10000 draws).",
            "checkpoint_files_exported": False,
            "medical_images_exported": False,
        },
    )
    print(
        f"Verified and exported {len(rows)} ATLAS and {len(mrart)} MR-ART rows to {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--permanent-root", required=True)
    args = parser.parse_args()
    export(args.source_root, args.output_root, args.split_path, args.permanent_root)
