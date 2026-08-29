"""Deterministic real MR-ART healthy-control false-positive analysis."""
from __future__ import annotations

import csv
import fcntl
import os
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np

from .core import (
    ARTIFACT_SCHEMA_VERSION,
    CompatibilityError,
    ConflictError,
    PREPROCESSING_VERSION,
    CorrectedExperimentError,
    array_sha256,
    configure_strict_determinism,
    read_json,
    sha256_file,
    software_provenance,
    stable_hash,
    subject_ids_hash,
    validate_metadata,
    write_json_atomic_replace,
    write_json_new,
)
from .inference import build_model, checkpoint_path, load_checkpoint
from .preprocessing import preprocess_volume
from .rng import derive_seed


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _source_path(data_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    return candidate if candidate.is_absolute() else data_root / candidate


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _preprocessed_mrart(
    *,
    source: dict[str, Any],
    source_path: Path,
    normalization_mode: str,
    memory_mode: str,
    cache_root: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if sha256_file(source_path) != source["sha256"]:
        raise CompatibilityError(f"MR-ART source checksum differs from immutable definition: {source_path}")
    raw = nb.load(str(source_path)).get_fdata(dtype=np.float32)
    if memory_mode == "low":
        value, diagnostics = preprocess_volume(raw, mode=normalization_mode, recipe=None)
        return value, {"memory_mode": "low", "cache_status": "not_used", "preprocessing": diagnostics}
    if memory_mode != "high" or cache_root is None:
        raise CorrectedExperimentError("MR-ART memory-high requires an explicit verified cache root")
    key = stable_hash(
        {
            "dataset": "MR-ART",
            "source_sha256": source["sha256"],
            "normalization_mode": normalization_mode,
            "preprocessing_version": PREPROCESSING_VERSION,
        }
    )
    base = cache_root / "mrart_preprocessed" / PREPROCESSING_VERSION / normalization_mode / key[:2]
    array_path = base / f"{key}.npy"
    metadata_path = base / f"{key}.metadata.json"
    expected = {
        "dataset": "MR-ART",
        "source_sha256": source["sha256"],
        "normalization_mode": normalization_mode,
        "preprocessing_version": PREPROCESSING_VERSION,
        "cache_key": key,
    }
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / f"{key}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if array_path.exists() or metadata_path.exists():
            if not array_path.exists() or not metadata_path.exists():
                raise ConflictError(f"Incomplete MR-ART memory-high cache: {array_path}")
            metadata = read_json(metadata_path)
            validate_metadata(metadata, expected, context=str(metadata_path))
            value = np.load(array_path, allow_pickle=False)
            if array_sha256(value) != metadata.get("array_sha256"):
                raise ConflictError(f"MR-ART cache checksum validation failed: {array_path}")
            return np.asarray(value, dtype=np.float32), {
                "memory_mode": "high",
                "cache_status": "reused",
                "cache_key": key,
            }
        value, _diagnostics = preprocess_volume(raw, mode=normalization_mode, recipe=None)
        _atomic_save_npy(array_path, value)
        write_json_atomic_replace(
            metadata_path,
            {**expected, "array_sha256": array_sha256(value), "shape": list(value.shape)},
        )
        return value, {"memory_mode": "high", "cache_status": "created", "cache_key": key}


def false_positive_metrics(prediction: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    from scipy import ndimage

    mask = np.asarray(prediction, dtype=bool)
    voxels = int(mask.sum())
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    counts = np.bincount(labels.ravel()) if count else np.asarray([0])
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    return {
        "predicted_voxels": voxels,
        "predicted_volume_ml": float(voxels * voxel_volume_mm3 / 1000.0),
        "binary_false_positive_present": bool(voxels > 0),
        "connected_component_count_26": int(count),
        "largest_connected_component_voxels_26": largest,
        "largest_connected_component_ml_26": float(largest * voxel_volume_mm3 / 1000.0),
    }


def run_mrart_task(
    *,
    repo_root: Path,
    root: Path,
    definition: dict[str, Any],
    data_root: Path,
    cache_root: Path | None,
    memory_mode: str,
    model_name: str,
    regime: str,
    fold: int,
    sw_batch_size: int = 1,
    repeat_check: bool = False,
    max_scans: int | None = None,
) -> dict[str, Any]:
    import torch
    from monai.inferers import sliding_window_inference

    sources = list(definition["dataset"]["mrart"]["sources"])
    if max_scans is not None:
        sources = sources[:max_scans]
    source_ids = [f"{entry['subject_id']}:{entry['acquisition']}" for entry in sources]
    global_seed = int(definition["seeds"]["global_seed"])
    runtime_seed = derive_seed(
        global_seed,
        "mrart_inference_runtime",
        model=model_name,
        training_regime=regime,
        fold=int(fold),
    )
    runtime = configure_strict_determinism(runtime_seed)
    device = _device()
    checkpoint = checkpoint_path(root, model_name, regime, fold)
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.exists() else "missing"
    state = load_checkpoint(
        checkpoint,
        definition=definition,
        model=model_name,
        regime=regime,
        fold=fold,
        device=device,
    )
    model = build_model(model_name).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    patch_size = tuple(int(value) for value in definition["model_configurations"][model_name]["patch_size"])
    task_dir = root / "mrart" / model_name / regime / f"fold-{fold}"
    task_path = task_dir / "task.metadata.json"
    expected = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "mrart_false_positive_task",
        "experiment_id": definition["experiment_id"],
        "analysis_dataset": "MR-ART healthy controls",
        "mrart_inventory_id": definition["dataset"]["mrart"]["inventory_id"],
        "model": model_name,
        "model_configuration_id": definition["model_configurations"][model_name]["model_configuration_id"],
        "fold": int(fold),
        "training_regime": regime,
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
        "source_ids_hash": subject_ids_hash(source_ids),
        "checkpoint_sha256": checkpoint_hash,
    }
    if task_path.exists():
        existing = read_json(task_path)
        validate_metadata(existing, expected, context=str(task_path))
        metrics_path = task_dir / "false_positive_metrics.csv"
        if not metrics_path.is_file() or sha256_file(metrics_path) != existing.get("metrics_sha256"):
            raise ConflictError(f"MR-ART task metrics are missing or changed: {metrics_path}")
        return {"status": "skipped-compatible", **existing}

    task_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in sources:
        source_path = _source_path(data_root, source["relative_path"])
        if not source_path.exists():
            raise CorrectedExperimentError(f"MR-ART source is missing: {source_path}")
        data, cache = _preprocessed_mrart(
            source=source,
            source_path=source_path,
            normalization_mode=definition["configuration"]["normalization_mode"],
            memory_mode=memory_mode,
            cache_root=cache_root,
        )
        inp = torch.from_numpy(data[None, None, ...]).float().to(device)
        with torch.inference_mode():
            logits = sliding_window_inference(inp, roi_size=patch_size, sw_batch_size=sw_batch_size, predictor=model)
            if not bool(torch.isfinite(logits).all().item()):
                raise CorrectedExperimentError(f"Non-finite MR-ART logits for {source_path}")
            prediction = torch.sigmoid(logits).ge(0.5).to(torch.uint8).cpu().numpy()[0, 0]
        repeat_metrics = None
        if repeat_check:
            configure_strict_determinism(runtime_seed)
            with torch.inference_mode():
                repeat_logits = sliding_window_inference(
                    inp, roi_size=patch_size, sw_batch_size=sw_batch_size, predictor=model
                )
                repeat_prediction = torch.sigmoid(repeat_logits).ge(0.5).to(torch.uint8).cpu().numpy()[0, 0]
            first_metrics = false_positive_metrics(prediction, float(np.prod(source["zooms"])))
            second_metrics = false_positive_metrics(repeat_prediction, float(np.prod(source["zooms"])))
            repeat_metrics = {
                "binary_equal": bool(np.array_equal(prediction, repeat_prediction)),
                "voxel_count_equal": first_metrics["predicted_voxels"] == second_metrics["predicted_voxels"],
                "predicted_volume_equal": first_metrics["predicted_volume_ml"] == second_metrics["predicted_volume_ml"],
                "connected_components_equal": first_metrics["connected_component_count_26"]
                == second_metrics["connected_component_count_26"],
                "largest_component_equal": first_metrics["largest_connected_component_voxels_26"]
                == second_metrics["largest_connected_component_voxels_26"],
            }
            if not all(repeat_metrics.values()):
                raise CorrectedExperimentError(f"MR-ART repeated inference differs for {source_path}: {repeat_metrics}")
        metrics = false_positive_metrics(prediction, float(np.prod(source["zooms"])))
        rows.append(
            {
                "subject_id": source["subject_id"],
                "acquisition": source["acquisition"],
                "source_relative_path": source["relative_path"],
                "source_sha256": source["sha256"],
                "model": model_name,
                "training_regime": regime,
                "fold": int(fold),
                **metrics,
                "input_array_sha256": array_sha256(data),
                "cache_status": cache["cache_status"],
                "repeat_consistent": all(repeat_metrics.values()) if repeat_metrics else "not_run",
            }
        )

    metrics_path = task_dir / "false_positive_metrics.csv"
    if metrics_path.exists():
        raise ConflictError(f"Refusing to overwrite MR-ART metrics: {metrics_path}")
    with metrics_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        **expected,
        "source_ids": source_ids,
        "global_seed": global_seed,
        "relevant_derived_seeds": {"mrart_inference_runtime": runtime_seed},
        "metrics_path": str(metrics_path.relative_to(root)),
        "metrics_sha256": sha256_file(metrics_path),
        "scan_count": len(rows),
        "strict_runtime": runtime,
        "repeat_inference_checked": repeat_check,
        "software_environment": software_provenance(repo_root),
        "note": "Real MR-ART healthy-control false-positive analysis; no synthetic challenge identity is assigned.",
    }
    write_json_new(task_path, metadata)
    return {"status": "completed", **metadata}


def analyze_mrart(
    *,
    root: Path,
    definition: dict[str, Any],
    models: list[str],
    regimes: list[str],
    folds: list[int],
) -> dict[str, Any]:
    from collections import defaultdict

    from .analysis import _holm_adjust, _paired_summary

    task_paths = [
        root / "mrart" / model / regime / f"fold-{fold}" / "task.metadata.json"
        for model in models
        for regime in regimes
        for fold in folds
    ]
    missing = [path for path in task_paths if not path.exists()]
    if missing:
        raise CorrectedExperimentError(f"MR-ART statistics require completed tasks; first missing: {missing[0]}")
    input_manifest_id = stable_hash([(str(path.relative_to(root)), sha256_file(path)) for path in task_paths])
    rows: list[dict[str, Any]] = []
    for task_path in task_paths:
        task = read_json(task_path)
        metrics_path = root / task["metrics_path"]
        if sha256_file(metrics_path) != task["metrics_sha256"]:
            raise ConflictError(f"MR-ART task metrics checksum changed: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))

    output_dir = root / "mrart" / "summary"
    combined_path = output_dir / "false_positive_metrics.csv"
    statistics_path = output_dir / "paired_statistics.json"
    metadata_path = output_dir / "summary.metadata.json"
    expected = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "mrart_summary",
        "experiment_id": definition["experiment_id"],
        "analysis_dataset": "MR-ART healthy controls",
        "input_manifest_id": input_manifest_id,
        "mrart_inventory_id": definition["dataset"]["mrart"]["inventory_id"],
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    if metadata_path.exists():
        existing = read_json(metadata_path)
        validate_metadata(existing, expected, context=str(metadata_path))
        if (
            not combined_path.is_file()
            or sha256_file(combined_path) != existing.get("combined_metrics_sha256")
            or not statistics_path.is_file()
            or sha256_file(statistics_path) != existing.get("statistics_sha256")
        ):
            raise ConflictError("MR-ART summary outputs are missing or changed")
        return {"status": "skipped-compatible", **existing}

    output_dir.mkdir(parents=True, exist_ok=True)
    with combined_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    indexed: dict[tuple[str, str, int, str, str], float] = {}
    for row in rows:
        key = (
            row["model"],
            row["training_regime"],
            int(row["fold"]),
            row["subject_id"],
            row["acquisition"],
        )
        indexed[key] = float(row["predicted_volume_ml"])
    comparisons: list[dict[str, Any]] = []
    global_seed = int(definition["seeds"]["global_seed"])
    subjects = [entry["subject_id"] for entry in definition["dataset"]["mrart"]["complete_subjects"]]
    for model in models:
        for regime in regimes:
            for motion in ("headmotion1", "headmotion2"):
                standard_values: dict[str, list[float]] = defaultdict(list)
                motion_values: dict[str, list[float]] = defaultdict(list)
                for subject in subjects:
                    for fold in folds:
                        standard_values[subject].append(indexed[(model, regime, fold, subject, "standard")])
                        motion_values[subject].append(indexed[(model, regime, fold, subject, motion)])
                comparisons.append(
                    _paired_summary(
                        standard_values,
                        motion_values,
                        global_seed=global_seed,
                        comparison_id=f"mrart:{model}:{regime}:{motion}-minus-standard-volume",
                        b_minus_a_label=f"{motion} minus standard predicted false-positive volume (mL)",
                    )
                )
    if {"standard", "augmented"}.issubset(regimes):
        for model in models:
            for acquisition in ("standard", "headmotion1", "headmotion2"):
                standard_training: dict[str, list[float]] = defaultdict(list)
                augmented_training: dict[str, list[float]] = defaultdict(list)
                for subject in subjects:
                    for fold in folds:
                        standard_training[subject].append(indexed[(model, "standard", fold, subject, acquisition)])
                        augmented_training[subject].append(indexed[(model, "augmented", fold, subject, acquisition)])
                comparisons.append(
                    _paired_summary(
                        standard_training,
                        augmented_training,
                        global_seed=global_seed,
                        comparison_id=f"mrart:{model}:{acquisition}:augmented-minus-standard-volume",
                        b_minus_a_label="augmented-training minus standard-training false-positive volume (mL)",
                    )
                )
    _holm_adjust(comparisons)
    statistics = {
        **expected,
        "artifact_type": "mrart_paired_statistics",
        "fold_handling": "fold predictions nested within subject; mean within subject before paired inference",
        "subject_count": len(subjects),
        "comparisons": comparisons,
    }
    write_json_new(statistics_path, statistics)
    metadata = {
        **expected,
        "combined_metrics_path": str(combined_path.relative_to(root)),
        "combined_metrics_sha256": sha256_file(combined_path),
        "statistics_path": str(statistics_path.relative_to(root)),
        "statistics_sha256": sha256_file(statistics_path),
        "row_count": len(rows),
        "models": models,
        "training_regimes": regimes,
        "folds": folds,
    }
    write_json_new(metadata_path, metadata)
    return {"status": "completed", **metadata}
