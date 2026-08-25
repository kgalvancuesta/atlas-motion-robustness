#!/usr/bin/env python3
"""MR-ART negative-control inference for ATLAS-trained segmentation models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_SCRIPTS = REPO_ROOT / "scripts"
if str(PROJECT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS))

import nibabel as nb
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from train_common import assert_logits_shape, load_nifti, normalize_volume


ACQUISITIONS = ("standard", "headmotion1", "headmotion2")
ARCH_ALIASES = {
    "mednext": "mednext",
    "swin": "swin_unetr",
    "swin_unetr": "swin_unetr",
    "swinunetr": "swin_unetr",
}
ARCH_DIRS = {"mednext": "mednext", "swin_unetr": "swin"}
TRAINING_ALIASES = {
    "standard": "standard",
    "baseline": "standard",
    "no_aug": "standard",
    "nonaug": "standard",
    "none": "standard",
    "aug": "augmented",
    "augmented": "augmented",
    "motion": "augmented",
    "motion_consistent": "augmented",
}
TRAINING_RUN_PREFIX = {"standard": "run_kfold", "augmented": "run_DA_kfold"}
THRESHOLD = 0.5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def normalize_arches(value: str) -> list[str]:
    raw = parse_csv_list(value)
    if not raw:
        raise ValueError("--arch must not be empty")
    if len(raw) == 1 and raw[0].lower() == "both":
        return ["mednext", "swin_unetr"]
    arches: list[str] = []
    for item in raw:
        key = item.lower().replace("-", "_")
        if key not in ARCH_ALIASES:
            raise ValueError(f"Unsupported architecture '{item}'")
        arch = ARCH_ALIASES[key]
        if arch not in arches:
            arches.append(arch)
    return arches


def normalize_training(value: str) -> list[str]:
    raw = parse_csv_list(value)
    if not raw:
        raise ValueError("--training must not be empty")
    if len(raw) == 1 and raw[0].lower() == "both":
        return ["standard", "augmented"]
    conditions: list[str] = []
    for item in raw:
        key = item.lower().replace("-", "_")
        if key not in TRAINING_ALIASES:
            raise ValueError(f"Unsupported training condition '{item}'")
        condition = TRAINING_ALIASES[key]
        if condition not in conditions:
            conditions.append(condition)
    return conditions


def parse_folds(value: str) -> list[int]:
    folds: list[int] = []
    for item in parse_csv_list(value):
        fold = int(item)
        if fold < 0:
            raise ValueError(f"Fold must be non-negative, got {fold}")
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise ValueError("--folds must include at least one fold")
    return folds


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_nvidia_smi() -> list[dict[str, str]]:
    query = "index,name,memory.total,memory.used,driver_version"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    rows = []
    keys = ["index", "name", "memory_total_mib", "memory_used_mib", "driver_version"]
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == len(keys):
            rows.append(dict(zip(keys, parts)))
    return rows


def torch_cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return info


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    ensure_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {message}\n")


def load_run_config(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config" / "train_config.json"
    if not config_path.exists():
        return None
    return read_json(config_path)


def build_model(arch: str) -> torch.nn.Module:
    if arch == "mednext":
        from models.mednext_model import build_mednext_model

        return build_mednext_model()
    if arch == "swin_unetr":
        from models.swin_model import build_swin_model

        return build_swin_model()
    raise ValueError(f"Unsupported architecture: {arch}")


def validate_patch_size(arch: str, patch_size: tuple[int, int, int]) -> None:
    if arch == "mednext":
        from models.mednext_model import validate_mednext_patch_size

        validate_mednext_patch_size(patch_size)
    elif arch == "swin_unetr":
        from models.swin_model import validate_swin_patch_size

        validate_swin_patch_size(patch_size)


def checkpoint_path_for(checkpoint_root: Path, arch: str, training: str, fold: int) -> Path:
    arch_dir = ARCH_DIRS[arch]
    fold_number = fold + 1
    run_name = f"{TRAINING_RUN_PREFIX[training]}_{fold_number:02d}"
    return checkpoint_root / arch_dir / run_name / "checkpoints" / "best.pt"


def checkpoint_run_dir(checkpoint_root: Path, arch: str, training: str, fold: int) -> Path:
    arch_dir = ARCH_DIRS[arch]
    fold_number = fold + 1
    run_name = f"{TRAINING_RUN_PREFIX[training]}_{fold_number:02d}"
    return checkpoint_root / arch_dir / run_name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoints(
    *,
    checkpoint_root: Path,
    arches: list[str],
    training_conditions: list[str],
    folds: list[int],
    checksum: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arch in arches:
        for training in training_conditions:
            for fold in folds:
                run_dir = checkpoint_run_dir(checkpoint_root, arch, training, fold)
                config = load_run_config(run_dir)
                patch_size = None
                model_name = None
                if config is not None:
                    model_name = config.get("model_name")
                    patch_size = config.get("args", {}).get("patch_size")
                ckpt_path = checkpoint_path_for(checkpoint_root, arch, training, fold)
                exists = ckpt_path.exists()
                stat = ckpt_path.stat() if exists else None
                rows.append(
                    {
                        "architecture": arch,
                        "training_condition": training,
                        "fold": fold,
                        "run_dir": str(run_dir),
                        "config_path": str(run_dir / "config" / "train_config.json"),
                        "config_exists": bool((run_dir / "config" / "train_config.json").exists()),
                        "config_model_name": model_name or "",
                        "patch_size": json_dumps(patch_size) if patch_size else "",
                        "checkpoint_path": str(ckpt_path),
                        "exists": bool(exists),
                        "file_size_bytes": stat.st_size if stat else "",
                        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
                        if stat
                        else "",
                        "checksum_sha256": sha256_file(ckpt_path) if exists and checksum else "",
                        "checksum_note": "" if checksum else "omitted by default; pass --checkpoint-checksum to compute",
                    }
                )
    return rows


def discover_mrart(data_root: Path, max_subjects: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not data_root.exists():
        raise FileNotFoundError(f"MR-ART data root not found: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"MR-ART data root is not a directory: {data_root}")

    subject_dirs = sorted([p for p in data_root.glob("sub-*") if p.is_dir()])
    incomplete_rows: list[dict[str, Any]] = []
    complete_rows_all: list[dict[str, Any]] = []

    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        anat_dir = sub_dir / "anat"
        paths: dict[str, Path] = {}
        missing: list[str] = []
        for acquisition in ACQUISITIONS:
            expected = anat_dir / f"{subject_id}_acq-{acquisition}_T1w.nii.gz"
            if expected.exists():
                paths[acquisition] = expected
            else:
                missing.append(acquisition)
        if missing:
            incomplete_rows.append(
                {
                    "subject_id": subject_id,
                    "missing_acquisitions": ",".join(missing),
                    "anat_dir": str(anat_dir),
                }
            )
            continue
        complete_rows_all.append(
            {
                "subject_id": subject_id,
                "standard_path": str(paths["standard"]),
                "headmotion1_path": str(paths["headmotion1"]),
                "headmotion2_path": str(paths["headmotion2"]),
            }
        )

    complete_rows = complete_rows_all[:max_subjects] if max_subjects is not None else complete_rows_all
    scan_rows: list[dict[str, Any]] = []
    for row in complete_rows:
        for acquisition in ACQUISITIONS:
            scan_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "acquisition": acquisition,
                    "image_path": row[f"{acquisition}_path"],
                }
            )

    summary = {
        "subjects_discovered": len(subject_dirs),
        "complete_triplet_subjects_total": len(complete_rows_all),
        "complete_triplet_subjects_used": len(complete_rows),
        "skipped_incomplete_subjects": len(incomplete_rows),
        "max_subjects": max_subjects,
        "scan_rows_used": len(scan_rows),
    }
    return complete_rows, incomplete_rows, {"scan_rows": scan_rows, **summary}


def completion_fieldnames(folds: list[int]) -> list[str]:
    base = [
        "subject_id",
        "acquisition",
        "image_path",
        "architecture",
        "training_condition",
        "folds_used",
        "output_mask_path",
        "output_prob_path",
        "metadata_path",
        "status",
        "error_type",
        "error_message",
        "runtime_seconds",
        "image_shape",
        "voxel_spacing",
        "voxel_volume_mm3",
        "output_shape",
        "ensemble_max_probability",
        "ensemble_mean_probability",
        "ensemble_predicted_voxels_at_0_5",
        "ensemble_predicted_volume_ml_at_0_5",
        "ensemble_binary_present_at_0_5",
        "ensemble_largest_component_ml_at_0_5",
        "min_fold_predicted_voxels_at_0_5",
        "max_fold_predicted_voxels_at_0_5",
        "mean_fold_predicted_voxels_at_0_5",
        "min_fold_predicted_volume_ml_at_0_5",
        "max_fold_predicted_volume_ml_at_0_5",
        "mean_fold_predicted_volume_ml_at_0_5",
        "num_folds_binary_present_at_0_5",
        "any_fold_binary_present_at_0_5",
        "all_folds_binary_present_at_0_5",
    ]
    for fold in range(5):
        base.extend(
            [
                f"fold{fold}_predicted_voxels_at_0_5",
                f"fold{fold}_predicted_volume_ml_at_0_5",
                f"fold{fold}_binary_present_at_0_5",
                f"fold{fold}_max_probability",
                f"fold{fold}_mean_probability",
                f"fold{fold}_prob_path",
                f"fold{fold}_mask_path",
            ]
        )
    extra_folds = [fold for fold in folds if fold > 4]
    for fold in extra_folds:
        base.extend(
            [
                f"fold{fold}_predicted_voxels_at_0_5",
                f"fold{fold}_predicted_volume_ml_at_0_5",
                f"fold{fold}_binary_present_at_0_5",
                f"fold{fold}_max_probability",
                f"fold{fold}_mean_probability",
                f"fold{fold}_prob_path",
                f"fold{fold}_mask_path",
            ]
        )
    return base


def error_fieldnames() -> list[str]:
    return [
        "subject_id",
        "acquisition",
        "image_path",
        "architecture",
        "training_condition",
        "folds_attempted",
        "checkpoint_path",
        "error_type",
        "error_message",
        "traceback_path",
        "timestamp",
    ]


def save_nifti_new(data: np.ndarray, affine: np.ndarray, header: nb.Nifti1Header, out_path: Path, dtype: Any) -> None:
    ensure_new_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_header = header.copy()
    out_header.set_data_dtype(dtype)
    img = nb.Nifti1Image(data.astype(dtype, copy=False), affine, out_header)
    nb.save(img, str(out_path))
    if not out_path.exists() or out_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to write non-empty NIfTI output: {out_path}")
    nb.load(str(out_path))


def save_probability(
    prob: np.ndarray,
    affine: np.ndarray,
    header: nb.Nifti1Header,
    out_path_base: Path,
    *,
    prob_format: str,
) -> Path:
    if prob_format == "nifti":
        out_path = out_path_base.parent / f"{out_path_base.name}.nii.gz"
        save_nifti_new(prob, affine, header, out_path, np.float32)
        return out_path
    if prob_format == "npz":
        out_path = out_path_base.with_name(out_path_base.name + ".npz")
        ensure_new_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, probability=prob.astype(np.float16), affine=affine)
        if not out_path.exists() or out_path.stat().st_size <= 0:
            raise RuntimeError(f"Failed to write non-empty probability output: {out_path}")
        return out_path
    raise ValueError(f"Unsupported probability format: {prob_format}")


def largest_component_ml(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    if not bool(mask.any()):
        return 0.0
    from scipy import ndimage

    labels, num = ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if num == 0:
        return 0.0
    counts = np.bincount(labels.ravel())
    if counts.size <= 1:
        return 0.0
    largest_voxels = int(counts[1:].max())
    return float(largest_voxels * voxel_volume_mm3 / 1000.0)


def format_output_stem(subject_id: str, acquisition: str, arch: str, training: str) -> str:
    return f"{subject_id}_acq-{acquisition}_arch-{arch}_train-{training}"


def load_checkpoint_map(checkpoint_manifest_path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = read_csv(checkpoint_manifest_path)
    mapping: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["architecture"], row["training_condition"], int(row["fold"]))
        mapping[key] = row
    return mapping


def output_dirs(run_dir: Path, arch: str, training: str) -> dict[str, Path]:
    base = run_dir / arch / training
    dirs = {
        "outputs": base / "outputs",
        "logs": base / "logs",
        "errors": base / "errors",
        "metadata": base / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def infer_job(
    *,
    job: dict[str, str],
    folds: list[int],
    checkpoint_map: dict[tuple[str, str, int], dict[str, str]],
    run_dir: Path,
    device: torch.device,
    sw_batch_size: int,
    save_ensemble_prob: bool,
    save_fold_outputs: bool,
    prob_format: str,
    compute_lcc: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    start = time.time()
    subject_id = job["subject_id"]
    acquisition = job["acquisition"]
    image_path = Path(job["image_path"])
    arch = job["architecture"]
    training = job["training_condition"]
    checkpoint_path_for_error = ""
    base_row: dict[str, Any] = {
        "subject_id": subject_id,
        "acquisition": acquisition,
        "image_path": str(image_path),
        "architecture": arch,
        "training_condition": training,
        "folds_used": ",".join(str(fold) for fold in folds),
        "output_mask_path": "",
        "output_prob_path": "",
        "metadata_path": "",
        "status": "error",
        "error_type": "",
        "error_message": "",
        "runtime_seconds": "",
        "image_shape": "",
        "voxel_spacing": "",
        "voxel_volume_mm3": "",
        "output_shape": "",
        "ensemble_largest_component_ml_at_0_5": "",
    }

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"MR-ART image not found: {image_path}")
        data, affine, header = load_nifti(image_path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D MR-ART T1w volume, got shape {data.shape} for {image_path}")
        image_shape = tuple(int(x) for x in data.shape)
        spacing = tuple(float(x) for x in header.get_zooms()[:3])
        voxel_volume_mm3 = float(np.prod(spacing))
        data = normalize_volume(data)
        inp = torch.from_numpy(data[None, None, ...]).float().to(device)

        dirs = output_dirs(run_dir, arch, training)
        stem = format_output_stem(subject_id, acquisition, arch, training)
        ensemble_sum: np.ndarray | None = None
        fold_metrics: list[dict[str, Any]] = []

        for fold in folds:
            ckpt_row = checkpoint_map[(arch, training, fold)]
            checkpoint_path = Path(ckpt_row["checkpoint_path"])
            checkpoint_path_for_error = str(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            patch_size_value = json.loads(ckpt_row["patch_size"]) if ckpt_row.get("patch_size") else [128, 128, 128]
            patch_size = tuple(int(x) for x in patch_size_value)
            validate_patch_size(arch, patch_size)

            model = build_model(arch).to(device)
            state = torch.load(checkpoint_path, map_location=device)
            state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
            model.load_state_dict(state_dict)
            model.eval()

            with torch.inference_mode():
                logits = sliding_window_inference(
                    inp,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )
                assert_logits_shape(logits, inp)
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError(f"Non-finite logits for {image_path} fold={fold}")
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0].astype(np.float32, copy=False)

            mask = prob >= THRESHOLD
            predicted_voxels = int(mask.sum())
            fold_prob_path = ""
            fold_mask_path = ""
            if save_fold_outputs:
                prob_base = dirs["outputs"] / f"{stem}_fold-{fold}_prob"
                fold_prob_path = str(save_probability(prob, affine, header, prob_base, prob_format=prob_format))
                fold_mask_path_obj = dirs["outputs"] / f"{stem}_fold-{fold}_mask.nii.gz"
                save_nifti_new(mask.astype(np.uint8), affine, header, fold_mask_path_obj, np.uint8)
                fold_mask_path = str(fold_mask_path_obj)

            fold_metrics.append(
                {
                    "fold": fold,
                    "predicted_voxels_at_0_5": predicted_voxels,
                    "predicted_volume_ml_at_0_5": float(predicted_voxels * voxel_volume_mm3 / 1000.0),
                    "binary_present_at_0_5": bool(predicted_voxels > 0),
                    "max_probability": float(prob.max()),
                    "mean_probability": float(prob.mean()),
                    "prob_path": fold_prob_path,
                    "mask_path": fold_mask_path,
                }
            )
            if ensemble_sum is None:
                ensemble_sum = prob.copy()
            else:
                ensemble_sum += prob

            del model, state, state_dict, logits, prob
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if ensemble_sum is None:
            raise RuntimeError("No fold predictions were produced")
        ensemble_prob = ensemble_sum / float(len(folds))
        ensemble_mask = ensemble_prob >= THRESHOLD
        ensemble_voxels = int(ensemble_mask.sum())
        ensemble_volume_ml = float(ensemble_voxels * voxel_volume_mm3 / 1000.0)
        ensemble_prob_path = ""
        if save_ensemble_prob:
            prob_base = dirs["outputs"] / f"{stem}_ensemble_prob"
            ensemble_prob_path = str(save_probability(ensemble_prob, affine, header, prob_base, prob_format=prob_format))
        ensemble_mask_path = dirs["outputs"] / f"{stem}_ensemble_mask.nii.gz"
        save_nifti_new(ensemble_mask.astype(np.uint8), affine, header, ensemble_mask_path, np.uint8)

        lcc_ml: float | str = ""
        if compute_lcc:
            lcc_ml = largest_component_ml(ensemble_mask, voxel_volume_mm3)

        metadata_path = dirs["metadata"] / f"{stem}_metadata.json"
        write_json_new(
            metadata_path,
            {
                "subject_id": subject_id,
                "acquisition": acquisition,
                "image_path": str(image_path),
                "architecture": arch,
                "training_condition": training,
                "folds": folds,
                "threshold": THRESHOLD,
                "input_shape": list(image_shape),
                "output_shape": list(ensemble_prob.shape),
                "voxel_spacing": list(spacing),
                "voxel_volume_mm3": voxel_volume_mm3,
                "affine": affine.tolist(),
                "source_header_zooms": list(header.get_zooms()),
                "output_space": "input voxel grid; no resampling or cropping outside MONAI sliding-window tiling",
                "preprocessing": "train_common.load_nifti + train_common.normalize_volume; no MR-ART-specific intensity tuning",
                "ensemble_probability_path": ensemble_prob_path,
                "ensemble_mask_path": str(ensemble_mask_path),
                "fold_metrics": fold_metrics,
                "largest_component_connectivity": "26-neighborhood" if compute_lcc else "",
            },
        )

        fold_voxels = [int(m["predicted_voxels_at_0_5"]) for m in fold_metrics]
        fold_volumes = [float(m["predicted_volume_ml_at_0_5"]) for m in fold_metrics]
        fold_presence = [bool(m["binary_present_at_0_5"]) for m in fold_metrics]
        row = dict(base_row)
        row.update(
            {
                "output_mask_path": str(ensemble_mask_path),
                "output_prob_path": ensemble_prob_path,
                "metadata_path": str(metadata_path),
                "status": "success",
                "runtime_seconds": f"{time.time() - start:.3f}",
                "image_shape": json_dumps(list(image_shape)),
                "voxel_spacing": json_dumps(list(spacing)),
                "voxel_volume_mm3": f"{voxel_volume_mm3:.10g}",
                "output_shape": json_dumps(list(ensemble_prob.shape)),
                "ensemble_max_probability": float(ensemble_prob.max()),
                "ensemble_mean_probability": float(ensemble_prob.mean()),
                "ensemble_predicted_voxels_at_0_5": ensemble_voxels,
                "ensemble_predicted_volume_ml_at_0_5": ensemble_volume_ml,
                "ensemble_binary_present_at_0_5": bool(ensemble_voxels > 0),
                "ensemble_largest_component_ml_at_0_5": lcc_ml,
                "min_fold_predicted_voxels_at_0_5": min(fold_voxels),
                "max_fold_predicted_voxels_at_0_5": max(fold_voxels),
                "mean_fold_predicted_voxels_at_0_5": float(np.mean(fold_voxels)),
                "min_fold_predicted_volume_ml_at_0_5": min(fold_volumes),
                "max_fold_predicted_volume_ml_at_0_5": max(fold_volumes),
                "mean_fold_predicted_volume_ml_at_0_5": float(np.mean(fold_volumes)),
                "num_folds_binary_present_at_0_5": int(sum(fold_presence)),
                "any_fold_binary_present_at_0_5": any(fold_presence),
                "all_folds_binary_present_at_0_5": all(fold_presence),
            }
        )
        for metric in fold_metrics:
            fold = int(metric["fold"])
            row[f"fold{fold}_predicted_voxels_at_0_5"] = metric["predicted_voxels_at_0_5"]
            row[f"fold{fold}_predicted_volume_ml_at_0_5"] = metric["predicted_volume_ml_at_0_5"]
            row[f"fold{fold}_binary_present_at_0_5"] = metric["binary_present_at_0_5"]
            row[f"fold{fold}_max_probability"] = metric["max_probability"]
            row[f"fold{fold}_mean_probability"] = metric["mean_probability"]
            row[f"fold{fold}_prob_path"] = metric["prob_path"]
            row[f"fold{fold}_mask_path"] = metric["mask_path"]
        return row, None
    except Exception as exc:
        runtime = time.time() - start
        row = dict(base_row)
        row.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "runtime_seconds": f"{runtime:.3f}",
            }
        )
        error = {
            "subject_id": subject_id,
            "acquisition": acquisition,
            "image_path": str(image_path),
            "architecture": arch,
            "training_condition": training,
            "folds_attempted": ",".join(str(fold) for fold in folds),
            "checkpoint_path": checkpoint_path_for_error,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": utc_now(),
        }
        return row, error


def make_run_id(args: argparse.Namespace, arches: list[str], training: list[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_label = arches[0] if len(arches) == 1 else "multiarch"
    training_label = training[0] if len(training) == 1 else "both"
    suffix = "smoke" if args.smoke_test else "run"
    return f"{timestamp}_{arch_label}_{training_label}_{suffix}"


def resolve_run_dir(args: argparse.Namespace, arches: list[str], training: list[str]) -> tuple[str, Path, str]:
    run_type = "smoke_tests" if args.smoke_test else "full_runs"
    run_id = args.run_id or make_run_id(args, arches, training)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.output_root) / run_type / run_id
    return run_id, run_dir, run_type


def prepare(args: argparse.Namespace) -> int:
    arches = normalize_arches(args.arch)
    training = normalize_training(args.training)
    folds = parse_folds(args.folds)
    run_id, run_dir, run_type = resolve_run_dir(args, arches, training)
    if run_dir.exists():
        if args.resume:
            raise SystemExit(f"--resume is not implemented for MR-ART runs. Choose a new --run-id. Existing: {run_dir}")
        raise SystemExit(f"Run directory already exists; refusing to overwrite: {run_dir}")

    complete_rows, incomplete_rows, manifest_summary = discover_mrart(Path(args.data_root), args.max_subjects)
    scan_rows = manifest_summary.pop("scan_rows")
    if not complete_rows:
        raise SystemExit(f"No complete MR-ART triplets found under {args.data_root}")

    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ["manifests", "outputs", "logs", "errors", "metadata", "merged"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "errors" / "tracebacks").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata" / "workers").mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "logs" / "run.log"
    append_log(log_path, "Preparing MR-ART negative-control inference run")

    checkpoint_rows = audit_checkpoints(
        checkpoint_root=Path(args.checkpoint_root),
        arches=arches,
        training_conditions=training,
        folds=folds,
        checksum=args.checkpoint_checksum,
    )
    missing_checkpoints = [row for row in checkpoint_rows if not row["exists"]]

    write_csv(
        run_dir / "manifests" / "mrart_complete_triplets.csv",
        ["subject_id", "standard_path", "headmotion1_path", "headmotion2_path"],
        complete_rows,
    )
    write_csv(
        run_dir / "manifests" / "mrart_incomplete_subjects.csv",
        ["subject_id", "missing_acquisitions", "anat_dir"],
        incomplete_rows,
    )
    write_csv(
        run_dir / "manifests" / "mrart_scan_manifest.csv",
        ["subject_id", "acquisition", "image_path"],
        scan_rows,
    )
    write_csv(
        run_dir / "manifests" / "checkpoint_manifest.csv",
        [
            "architecture",
            "training_condition",
            "fold",
            "run_dir",
            "config_path",
            "config_exists",
            "config_model_name",
            "patch_size",
            "checkpoint_path",
            "exists",
            "file_size_bytes",
            "mtime",
            "checksum_sha256",
            "checksum_note",
        ],
        checkpoint_rows,
    )

    jobs: list[dict[str, Any]] = []
    job_index = 0
    for scan in scan_rows:
        for arch in arches:
            for condition in training:
                jobs.append(
                    {
                        "job_index": job_index,
                        "subject_id": scan["subject_id"],
                        "acquisition": scan["acquisition"],
                        "image_path": scan["image_path"],
                        "architecture": arch,
                        "training_condition": condition,
                    }
                )
                job_index += 1
    write_csv(
        run_dir / "manifests" / "planned_jobs.csv",
        ["job_index", "subject_id", "acquisition", "image_path", "architecture", "training_condition"],
        jobs,
    )

    repo_root = REPO_ROOT
    run_info = {
        "run_id": run_id,
        "run_type": run_type,
        "run_dir": str(run_dir),
        "timestamp_utc": utc_now(),
        "start_time_epoch": time.time(),
        "launcher_command": args.launcher_command,
        "python_command": " ".join([sys.executable] + sys.argv),
        "git_commit": get_git_commit(repo_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_cuda_info": torch_cuda_info(),
        "nvidia_smi": run_nvidia_smi(),
        "data_root": str(Path(args.data_root)),
        "output_root": str(Path(args.output_root)),
        "checkpoint_root": str(Path(args.checkpoint_root)),
        "selected_architectures": arches,
        "selected_training_conditions": training,
        "selected_folds": folds,
        "selected_gpus": args.gpus,
        "smoke_test": args.smoke_test,
        "max_subjects": args.max_subjects,
        "save_fold_outputs": args.save_fold_outputs,
        "save_ensemble_prob": args.save_ensemble_prob,
        "prob_format": args.prob_format,
        "sw_batch_size": args.sw_batch_size,
        "compute_largest_component": args.compute_largest_component,
        "checkpoint_checksum": args.checkpoint_checksum,
        "checkpoint_checksum_note": "omitted unless --checkpoint-checksum is passed",
        "manifest_summary": manifest_summary,
        "planned_jobs": len(jobs),
        "paths": {
            "complete_triplets_manifest": str(run_dir / "manifests" / "mrart_complete_triplets.csv"),
            "scan_manifest": str(run_dir / "manifests" / "mrart_scan_manifest.csv"),
            "incomplete_subjects_manifest": str(run_dir / "manifests" / "mrart_incomplete_subjects.csv"),
            "checkpoint_manifest": str(run_dir / "manifests" / "checkpoint_manifest.csv"),
            "planned_jobs": str(run_dir / "manifests" / "planned_jobs.csv"),
        },
        "preprocessing_assumption": "Matches existing scripts/infer.py: load_nifti, normalize nonzero voxels, sliding_window_inference, sigmoid.",
        "geometry_assumption": "Outputs are written in the source MR-ART voxel grid with source affine/header; no resampling.",
    }
    write_json_new(run_dir / "metadata" / "run_info.json", run_info)
    write_json_new(run_dir / "logs" / "run_info.json", run_info)
    append_log(log_path, f"MR-ART subjects discovered: {manifest_summary['subjects_discovered']}")
    append_log(log_path, f"Complete triplets used: {manifest_summary['complete_triplet_subjects_used']}")
    append_log(log_path, f"Planned scan/model-condition jobs: {len(jobs)}")
    append_log(log_path, f"Checkpoint manifest: {run_dir / 'manifests' / 'checkpoint_manifest.csv'}")

    if missing_checkpoints and not args.allow_missing_checkpoints:
        append_log(log_path, f"Fatal: missing {len(missing_checkpoints)} required checkpoints")
        missing = "\n".join(f"  - {row['checkpoint_path']}" for row in missing_checkpoints)
        raise SystemExit(f"Missing required checkpoint(s):\n{missing}\nAudit: {run_dir / 'manifests' / 'checkpoint_manifest.csv'}")

    if missing_checkpoints:
        append_log(log_path, f"Continuing despite {len(missing_checkpoints)} missing checkpoints because --allow-missing-checkpoints was passed")

    print(f"Prepared run directory: {run_dir}")
    print(f"Planned jobs: {len(jobs)}")
    print(f"Checkpoint audit: {run_dir / 'manifests' / 'checkpoint_manifest.csv'}")
    return 0


def worker(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_info = read_json(run_dir / "metadata" / "run_info.json")
    folds = [int(fold) for fold in run_info["selected_folds"]]
    checkpoint_map = load_checkpoint_map(run_dir / "manifests" / "checkpoint_manifest.csv")
    jobs = read_csv(run_dir / "manifests" / "planned_jobs.csv")
    worker_jobs = [job for job in jobs if int(job["job_index"]) % args.num_workers == args.worker_index]
    fieldnames = completion_fieldnames(folds)
    completion_path = run_dir / "metadata" / "workers" / f"worker_{args.worker_index}_completion.csv"
    error_path = run_dir / "errors" / f"worker_{args.worker_index}_errors.csv"
    ensure_new_path(completion_path)
    ensure_new_path(error_path)

    device = choose_device()
    print(f"Worker {args.worker_index}/{args.num_workers}: device={device}, gpu_id={args.gpu_id}, jobs={len(worker_jobs)}")

    completion_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / "errors" / "tracebacks"
    error_count = 0
    with completion_path.open("w", encoding="utf-8", newline="") as comp_f, error_path.open(
        "w", encoding="utf-8", newline=""
    ) as err_f:
        comp_writer = csv.DictWriter(comp_f, fieldnames=fieldnames, extrasaction="ignore")
        err_writer = csv.DictWriter(err_f, fieldnames=error_fieldnames(), extrasaction="ignore")
        comp_writer.writeheader()
        err_writer.writeheader()

        pbar = tqdm(
            worker_jobs,
            total=len(worker_jobs),
            desc=f"worker {args.worker_index}",
            dynamic_ncols=True,
            position=args.progress_position if args.progress_position is not None else args.worker_index,
        )
        for job in pbar:
            pbar.set_postfix(
                {
                    "arch": job["architecture"],
                    "train": job["training_condition"],
                    "scan": f"{job['subject_id']}:{job['acquisition']}",
                    "errors": error_count,
                }
            )
            row, error = infer_job(
                job=job,
                folds=folds,
                checkpoint_map=checkpoint_map,
                run_dir=run_dir,
                device=device,
                sw_batch_size=args.sw_batch_size,
                save_ensemble_prob=bool(run_info["save_ensemble_prob"]),
                save_fold_outputs=bool(run_info["save_fold_outputs"]),
                prob_format=str(run_info["prob_format"]),
                compute_lcc=bool(run_info["compute_largest_component"]),
            )
            comp_writer.writerow(row)
            comp_f.flush()
            if error is not None:
                error_count += 1
                trace_path = trace_dir / (
                    f"worker-{args.worker_index}_job-{job['job_index']}_"
                    f"{job['subject_id']}_{job['acquisition']}_{job['architecture']}_{job['training_condition']}.txt"
                )
                ensure_new_path(trace_path)
                trace_path.write_text(error.pop("traceback"), encoding="utf-8")
                error["traceback_path"] = str(trace_path)
                err_writer.writerow(error)
                err_f.flush()
        pbar.close()

    print(f"Worker {args.worker_index} complete: errors={error_count}, completion_csv={completion_path}, error_csv={error_path}")
    return 0


def merge_csvs(paths: list[Path], out_path: Path, fieldnames: list[str]) -> int:
    ensure_new_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for path in paths:
            if not path.exists():
                continue
            for row in read_csv(path):
                writer.writerow(row)
                total += 1
    return total


def finalize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_info = read_json(run_dir / "metadata" / "run_info.json")
    folds = [int(fold) for fold in run_info["selected_folds"]]
    completion_paths = sorted((run_dir / "metadata" / "workers").glob("worker_*_completion.csv"))
    error_paths = sorted((run_dir / "errors").glob("worker_*_errors.csv"))
    completion_out = run_dir / "merged" / "completion.csv"
    errors_out = run_dir / "merged" / "errors.csv"
    completion_rows = merge_csvs(completion_paths, completion_out, completion_fieldnames(folds))
    error_rows = merge_csvs(error_paths, errors_out, error_fieldnames())

    statuses = {"success": 0, "error": 0, "skipped": 0}
    for row in read_csv(completion_out):
        statuses[row.get("status", "")] = statuses.get(row.get("status", ""), 0) + 1

    total_planned = int(run_info["planned_jobs"])
    completed = int(statuses.get("success", 0))
    failed = int(statuses.get("error", 0))
    skipped = int(statuses.get("skipped", 0))
    percent_completed = float(completed * 100.0 / total_planned) if total_planned else 0.0
    wall_clock_runtime = float(time.time() - float(run_info["start_time_epoch"]))
    summary = {
        "run_id": run_info["run_id"],
        "total_planned_scan_model_condition_jobs": total_planned,
        "total_completion_rows": completion_rows,
        "total_completed": completed,
        "total_failed": failed,
        "total_skipped": skipped,
        "total_error_rows": error_rows,
        "percent_completed": percent_completed,
        "wall_clock_runtime_seconds": wall_clock_runtime,
        "output_root": run_info["output_root"],
        "run_dir": str(run_dir),
        "manifest_path": run_info["paths"]["scan_manifest"],
        "checkpoint_manifest_path": run_info["paths"]["checkpoint_manifest"],
        "completion_csv_path": str(completion_out),
        "error_csv_path": str(errors_out),
    }
    write_json_new(run_dir / "merged" / "summary.json", summary)
    append_log(run_dir / "logs" / "run.log", f"Finalized run summary: {json_dumps(summary)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MR-ART negative-control inference with ATLAS-trained checkpoints.")
    parser.add_argument("--mode", choices=["prepare", "worker", "finalize"], default="prepare")
    parser.add_argument("--data-root", default="data/mr-art")
    parser.add_argument("--output-root", default="runs/mrart_negative_control")
    parser.add_argument("--checkpoint-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Reserved; resume is not implemented and will fail safely.")
    parser.add_argument("--arch", default="mednext")
    parser.add_argument("--training", default="augmented")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--save-ensemble-prob", dest="save_ensemble_prob", action="store_true", default=True)
    parser.add_argument("--no-save-ensemble-prob", dest="save_ensemble_prob", action="store_false")
    parser.add_argument("--save-fold-outputs", action="store_true")
    parser.add_argument("--prob-format", choices=["nifti", "npz"], default="nifti")
    parser.add_argument("--compute-largest-component", dest="compute_largest_component", action="store_true", default=False)
    parser.add_argument("--no-compute-largest-component", dest="compute_largest_component", action="store_false")
    parser.add_argument("--checkpoint-checksum", action="store_true")
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    parser.add_argument("--launcher-command", default="")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--progress-position", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "prepare":
        return prepare(args)
    if args.mode == "worker":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode worker")
        if args.worker_index < 0 or args.worker_index >= args.num_workers:
            raise SystemExit("--worker-index must be in [0, --num-workers)")
        return worker(args)
    if args.mode == "finalize":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode finalize")
        return finalize(args)
    raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
