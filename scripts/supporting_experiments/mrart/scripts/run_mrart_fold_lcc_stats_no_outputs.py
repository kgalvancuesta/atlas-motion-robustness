#!/usr/bin/env python3
"""Recompute MR-ART fold predictions and write lightweight component statistics only.

This script is intended for the server. It reruns fold-level inference, computes
thresholded mask statistics in memory, and writes CSV/JSON reports. It never
saves fold masks, fold probability maps, ensemble masks, or ensemble probability
maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import queue as queue_module
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable


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
    "aug": "augmented",
    "augmented": "augmented",
    "motion": "augmented",
}
TRAINING_RUN_PREFIX = {"standard": "run_kfold", "augmented": "run_DA_kfold"}
THRESHOLD = 0.5

LONG_FIELDNAMES = [
    "subject_id",
    "scan_id",
    "acquisition_id",
    "acquisition_type",
    "image_path",
    "architecture",
    "training_condition",
    "fold",
    "checkpoint_path",
    "checkpoint_exists",
    "patch_size",
    "threshold",
    "image_shape",
    "voxel_spacing",
    "voxel_volume_mm3",
    "predicted_lesion_voxel_count",
    "predicted_lesion_volume_mm3",
    "predicted_lesion_volume_ml",
    "scan_positive",
    "connected_component_count_26conn",
    "largest_cc_voxel_count",
    "largest_cc_volume_mm3",
    "largest_cc_volume_ml",
    "largest_cc_fraction_of_prediction",
    "mean_component_voxel_count",
    "median_component_voxel_count",
    "std_component_voxel_count",
    "mean_component_volume_ml",
    "median_component_volume_ml",
    "std_component_volume_ml",
    "max_component_volume_ml",
    "component_count_ge_10_voxels",
    "component_count_ge_100_voxels",
    "component_count_ge_1000_voxels",
    "small_component_count_lt_10_voxels",
    "small_component_voxel_fraction_lt_10_voxels",
    "max_probability",
    "mean_probability",
    "runtime_seconds",
    "status",
    "error_type",
    "error_message",
]

SUMMARY_FIELDNAMES = [
    "architecture",
    "training_condition",
    "acquisition_type",
    "n_fold_predictions",
    "n_success",
    "n_error",
    "n_scans",
    "n_unique_subjects",
    "scan_positive_rate_fold_level",
    "mean_fold_predicted_lesion_volume_ml",
    "median_fold_predicted_lesion_volume_ml",
    "std_fold_predicted_lesion_volume_ml",
    "iqr_fold_predicted_lesion_volume_ml",
    "max_fold_predicted_lesion_volume_ml",
    "mean_fold_largest_cc_volume_ml",
    "median_fold_largest_cc_volume_ml",
    "std_fold_largest_cc_volume_ml",
    "iqr_fold_largest_cc_volume_ml",
    "max_fold_largest_cc_volume_ml",
    "mean_connected_component_count_26conn",
    "median_connected_component_count_26conn",
    "std_connected_component_count_26conn",
    "iqr_connected_component_count_26conn",
    "max_connected_component_count_26conn",
    "mean_largest_cc_fraction_of_prediction",
    "median_largest_cc_fraction_of_prediction",
    "std_largest_cc_fraction_of_prediction",
    "iqr_largest_cc_fraction_of_prediction",
    "mean_component_count_ge_10_voxels",
    "mean_component_count_ge_100_voxels",
    "mean_component_count_ge_1000_voxels",
    "mean_small_component_voxel_fraction_lt_10_voxels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/mr-art"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path, default=Path("scripts/supporting_experiments/mrart/server_reports"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact output directory for this run.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--arch", default="both", help="mednext|swin_unetr|both|comma-separated list")
    parser.add_argument("--training", default="both", help="standard|augmented|both|comma-separated list")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Write MR-ART triplet/missing-acquisition counts and exit before inference.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def normalize_arches(value: str) -> list[str]:
    raw = parse_csv_list(value)
    if len(raw) == 1 and raw[0].lower() == "both":
        return ["mednext", "swin_unetr"]
    arches: list[str] = []
    for item in raw:
        key = item.lower().replace("-", "_")
        if key not in ARCH_ALIASES:
            raise ValueError(f"Unsupported architecture: {item}")
        arch = ARCH_ALIASES[key]
        if arch not in arches:
            arches.append(arch)
    if not arches:
        raise ValueError("--arch must include at least one architecture")
    return arches


def normalize_training(value: str) -> list[str]:
    raw = parse_csv_list(value)
    if len(raw) == 1 and raw[0].lower() == "both":
        return ["standard", "augmented"]
    conditions: list[str] = []
    for item in raw:
        key = item.lower().replace("-", "_")
        if key not in TRAINING_ALIASES:
            raise ValueError(f"Unsupported training condition: {item}")
        condition = TRAINING_ALIASES[key]
        if condition not in conditions:
            conditions.append(condition)
    if not conditions:
        raise ValueError("--training must include at least one condition")
    return conditions


def parse_folds(value: str) -> list[int]:
    folds = []
    for item in parse_csv_list(value):
        fold = int(item)
        if fold < 0:
            raise ValueError(f"Fold must be non-negative: {fold}")
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise ValueError("--folds must include at least one fold")
    return folds


def parse_gpus(value: str) -> list[int]:
    gpus = []
    for item in parse_csv_list(value):
        gpu = int(item)
        if gpu not in gpus:
            gpus.append(gpu)
    return gpus


def json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def check_not_ignored(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.resolve().relative_to(repo_root().resolve())
        except ValueError:
            # Server run directories are deliberately outside the repository.
            # Git ignore rules do not apply to them.
            continue
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


def ensure_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_new_path(path)
    check_not_ignored([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_new_path(path)
    check_not_ignored([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def add_project_scripts_to_path() -> None:
    scripts_dir = repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
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
    keys = ["index", "name", "memory_total_mib", "memory_used_mib", "driver_version"]
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == len(keys):
            rows.append(dict(zip(keys, parts)))
    return rows


def torch_cuda_info() -> dict[str, Any]:
    try:
        import torch

        info: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info["devices"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                }
                for idx in range(torch.cuda.device_count())
            ]
        return info
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config" / "train_config.json"
    if not config_path.exists():
        return None
    return read_json(config_path)


def checkpoint_path_for(checkpoint_root: Path, arch: str, training: str, fold: int) -> Path:
    run_name = f"{TRAINING_RUN_PREFIX[training]}_{fold + 1:02d}"
    return checkpoint_root / ARCH_DIRS[arch] / run_name / "checkpoints" / "best.pt"


def checkpoint_run_dir(checkpoint_root: Path, arch: str, training: str, fold: int) -> Path:
    run_name = f"{TRAINING_RUN_PREFIX[training]}_{fold + 1:02d}"
    return checkpoint_root / ARCH_DIRS[arch] / run_name


def patch_size_for(checkpoint_root: Path, arch: str, training: str, fold: int) -> list[int]:
    config = load_run_config(checkpoint_run_dir(checkpoint_root, arch, training, fold))
    if config is None:
        return [128, 128, 128]
    patch_size = config.get("args", {}).get("patch_size") or [128, 128, 128]
    return [int(x) for x in patch_size]


def discover_mrart(data_root: Path, max_subjects: int | None) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    if not data_root.exists():
        raise FileNotFoundError(f"MR-ART data root not found: {data_root}")
    subject_dirs = sorted([p for p in data_root.glob("sub-*") if p.is_dir()])
    complete_rows: list[dict[str, str]] = []
    incomplete_rows: list[dict[str, str]] = []
    missing_acquisition_counts = {acquisition: 0 for acquisition in ACQUISITIONS}
    present_expected_acquisition_files = 0
    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        anat_dir = sub_dir / "anat"
        paths: dict[str, Path] = {}
        missing = []
        for acquisition in ACQUISITIONS:
            expected = anat_dir / f"{subject_id}_acq-{acquisition}_T1w.nii.gz"
            if expected.exists():
                paths[acquisition] = expected
                present_expected_acquisition_files += 1
            else:
                missing.append(acquisition)
                missing_acquisition_counts[acquisition] += 1
        if missing:
            incomplete_rows.append(
                {
                    "subject_id": subject_id,
                    "missing_acquisitions": ",".join(missing),
                    "anat_dir": str(anat_dir),
                }
            )
            continue
        complete_rows.append(
            {
                "subject_id": subject_id,
                "standard_path": str(paths["standard"]),
                "headmotion1_path": str(paths["headmotion1"]),
                "headmotion2_path": str(paths["headmotion2"]),
            }
        )
    used_rows = complete_rows[:max_subjects] if max_subjects is not None else complete_rows
    scan_rows: list[dict[str, str]] = []
    for row in used_rows:
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
        "complete_triplet_subjects_total": len(complete_rows),
        "complete_triplet_subjects_used": len(used_rows),
        "subjects_missing_one_or_more_acquisitions": len(incomplete_rows),
        "skipped_incomplete_subjects": len(incomplete_rows),
        "expected_acquisition_files_for_discovered_subjects": len(subject_dirs) * len(ACQUISITIONS),
        "present_expected_acquisition_files": present_expected_acquisition_files,
        "missing_expected_acquisition_files_total": sum(missing_acquisition_counts.values()),
        "missing_acquisition_counts": missing_acquisition_counts,
        "scan_rows_used": len(scan_rows),
        "max_subjects": max_subjects,
    }
    return scan_rows, incomplete_rows, summary


def build_jobs(
    data_root: Path,
    checkpoint_root: Path,
    arches: list[str],
    training_conditions: list[str],
    folds: list[int],
    max_subjects: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scan_rows, incomplete_rows, manifest_summary = discover_mrart(data_root, max_subjects)
    jobs = []
    for scan in scan_rows:
        for arch in arches:
            for training in training_conditions:
                for fold in folds:
                    checkpoint_path = checkpoint_path_for(checkpoint_root, arch, training, fold)
                    patch_size = patch_size_for(checkpoint_root, arch, training, fold)
                    jobs.append(
                        {
                            "subject_id": scan["subject_id"],
                            "scan_id": f"{scan['subject_id']}_acq-{scan['acquisition']}",
                            "acquisition_id": scan["acquisition"],
                            "acquisition_type": scan["acquisition"],
                            "image_path": scan["image_path"],
                            "architecture": arch,
                            "training_condition": training,
                            "fold": fold,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_exists": bool(checkpoint_path.exists()),
                            "patch_size": patch_size,
                        }
                    )
    return jobs, {"manifest_summary": manifest_summary, "incomplete_subjects": incomplete_rows}


def validate_patch_size(arch: str, patch_size: tuple[int, int, int]) -> None:
    add_project_scripts_to_path()
    if arch == "mednext":
        from models.mednext_model import validate_mednext_patch_size

        validate_mednext_patch_size(patch_size)
    elif arch == "swin_unetr":
        from models.swin_model import validate_swin_patch_size

        validate_swin_patch_size(patch_size)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def build_model(arch: str):
    add_project_scripts_to_path()
    if arch == "mednext":
        from models.mednext_model import build_mednext_model

        return build_mednext_model()
    if arch == "swin_unetr":
        from models.swin_model import build_swin_model

        return build_swin_model()
    raise ValueError(f"Unsupported architecture: {arch}")


def component_stats(mask: Any, voxel_volume_mm3: float) -> dict[str, Any]:
    import numpy as np
    from scipy import ndimage

    predicted_voxels = int(mask.sum())
    empty_stats = {
        "connected_component_count_26conn": 0,
        "largest_cc_voxel_count": 0,
        "largest_cc_volume_mm3": 0.0,
        "largest_cc_volume_ml": 0.0,
        "largest_cc_fraction_of_prediction": 0.0,
        "mean_component_voxel_count": 0.0,
        "median_component_voxel_count": 0.0,
        "std_component_voxel_count": 0.0,
        "mean_component_volume_ml": 0.0,
        "median_component_volume_ml": 0.0,
        "std_component_volume_ml": 0.0,
        "max_component_volume_ml": 0.0,
        "component_count_ge_10_voxels": 0,
        "component_count_ge_100_voxels": 0,
        "component_count_ge_1000_voxels": 0,
        "small_component_count_lt_10_voxels": 0,
        "small_component_voxel_fraction_lt_10_voxels": 0.0,
    }
    if not bool(mask.any()):
        return empty_stats
    labels, num = ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if num == 0:
        return empty_stats
    counts = np.bincount(labels.ravel())
    component_counts = counts[1:].astype(np.float64, copy=False)
    if component_counts.size == 0:
        return empty_stats
    largest_voxels = int(component_counts.max())
    component_volumes_ml = component_counts * float(voxel_volume_mm3) / 1000.0
    small_mask = component_counts < 10
    small_voxels = float(component_counts[small_mask].sum()) if small_mask.any() else 0.0
    return {
        "connected_component_count_26conn": int(num),
        "largest_cc_voxel_count": largest_voxels,
        "largest_cc_volume_mm3": float(largest_voxels * voxel_volume_mm3),
        "largest_cc_volume_ml": float(largest_voxels * voxel_volume_mm3 / 1000.0),
        "largest_cc_fraction_of_prediction": float(largest_voxels / predicted_voxels) if predicted_voxels else 0.0,
        "mean_component_voxel_count": float(component_counts.mean()),
        "median_component_voxel_count": float(np.median(component_counts)),
        "std_component_voxel_count": float(component_counts.std(ddof=0)),
        "mean_component_volume_ml": float(component_volumes_ml.mean()),
        "median_component_volume_ml": float(np.median(component_volumes_ml)),
        "std_component_volume_ml": float(component_volumes_ml.std(ddof=0)),
        "max_component_volume_ml": float(component_volumes_ml.max()),
        "component_count_ge_10_voxels": int((component_counts >= 10).sum()),
        "component_count_ge_100_voxels": int((component_counts >= 100).sum()),
        "component_count_ge_1000_voxels": int((component_counts >= 1000).sum()),
        "small_component_count_lt_10_voxels": int(small_mask.sum()),
        "small_component_voxel_fraction_lt_10_voxels": float(small_voxels / predicted_voxels) if predicted_voxels else 0.0,
    }


def error_row(job: dict[str, Any], start: float, exc: BaseException) -> dict[str, Any]:
    return {
        "subject_id": job["subject_id"],
        "scan_id": job["scan_id"],
        "acquisition_id": job["acquisition_id"],
        "acquisition_type": job["acquisition_type"],
        "image_path": job["image_path"],
        "architecture": job["architecture"],
        "training_condition": job["training_condition"],
        "fold": job["fold"],
        "checkpoint_path": job["checkpoint_path"],
        "checkpoint_exists": int(bool(job.get("checkpoint_exists"))),
        "patch_size": json.dumps(job.get("patch_size", []), separators=(",", ":")),
        "threshold": THRESHOLD,
        "runtime_seconds": f"{time.time() - start:.3f}",
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def infer_group(
    jobs: list[dict[str, Any]],
    device: Any,
    sw_batch_size: int,
    worker_index: int,
) -> list[dict[str, Any]]:
    add_project_scripts_to_path()

    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from train_common import assert_logits_shape, load_nifti, normalize_volume
    from tqdm import tqdm

    first = jobs[0]
    arch = first["architecture"]
    training = first["training_condition"]
    fold = first["fold"]
    patch_size = tuple(int(x) for x in first["patch_size"])
    validate_patch_size(arch, patch_size)
    checkpoint_path = Path(first["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(arch).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    out_rows = []
    try:
        iterator = tqdm(
            jobs,
            total=len(jobs),
            desc=f"worker {worker_index} {arch} {training} fold {fold}",
            dynamic_ncols=True,
            position=worker_index,
            leave=True,
            file=sys.stdout,
        )
        for job in iterator:
            iterator.set_postfix(
                {
                    "scan": f"{job['subject_id']}:{job['acquisition_id']}",
                    "device": str(device),
                }
            )
            start = time.time()
            try:
                data, _affine, header = load_nifti(Path(job["image_path"]))
                if data.ndim != 3:
                    raise ValueError(f"Expected 3D source MR-ART T1w image, got shape {data.shape}")
                image_shape = [int(x) for x in data.shape]
                spacing = [float(x) for x in header.get_zooms()[:3]]
                voxel_volume_mm3 = float(np.prod(spacing))
                data = normalize_volume(data)
                inp = torch.from_numpy(data[None, None, ...]).float().to(device)
                with torch.inference_mode():
                    logits = sliding_window_inference(
                        inp,
                        roi_size=patch_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                    )
                    assert_logits_shape(logits, inp)
                    if not bool(torch.isfinite(logits).all().item()):
                        raise RuntimeError("Non-finite logits")
                    prob = torch.sigmoid(logits).cpu().numpy()[0, 0].astype(np.float32, copy=False)

                mask = prob >= THRESHOLD
                predicted_voxels = int(mask.sum())
                predicted_volume_mm3 = float(predicted_voxels * voxel_volume_mm3)
                components = component_stats(mask, voxel_volume_mm3)
                out_rows.append(
                    {
                        "subject_id": job["subject_id"],
                        "scan_id": job["scan_id"],
                        "acquisition_id": job["acquisition_id"],
                        "acquisition_type": job["acquisition_type"],
                        "image_path": job["image_path"],
                        "architecture": job["architecture"],
                        "training_condition": job["training_condition"],
                        "fold": job["fold"],
                        "checkpoint_path": job["checkpoint_path"],
                        "checkpoint_exists": int(bool(job.get("checkpoint_exists"))),
                        "patch_size": json.dumps(list(patch_size), separators=(",", ":")),
                        "threshold": THRESHOLD,
                        "image_shape": json.dumps(image_shape, separators=(",", ":")),
                        "voxel_spacing": json.dumps(spacing, separators=(",", ":")),
                        "voxel_volume_mm3": f"{voxel_volume_mm3:.10g}",
                        "predicted_lesion_voxel_count": predicted_voxels,
                        "predicted_lesion_volume_mm3": f"{predicted_volume_mm3:.10g}",
                        "predicted_lesion_volume_ml": f"{predicted_volume_mm3 / 1000.0:.10g}",
                        "scan_positive": int(predicted_voxels > 0),
                        "connected_component_count_26conn": components["connected_component_count_26conn"],
                        "largest_cc_voxel_count": components["largest_cc_voxel_count"],
                        "largest_cc_volume_mm3": f"{components['largest_cc_volume_mm3']:.10g}",
                        "largest_cc_volume_ml": f"{components['largest_cc_volume_ml']:.10g}",
                        "largest_cc_fraction_of_prediction": f"{components['largest_cc_fraction_of_prediction']:.10g}",
                        "mean_component_voxel_count": f"{components['mean_component_voxel_count']:.10g}",
                        "median_component_voxel_count": f"{components['median_component_voxel_count']:.10g}",
                        "std_component_voxel_count": f"{components['std_component_voxel_count']:.10g}",
                        "mean_component_volume_ml": f"{components['mean_component_volume_ml']:.10g}",
                        "median_component_volume_ml": f"{components['median_component_volume_ml']:.10g}",
                        "std_component_volume_ml": f"{components['std_component_volume_ml']:.10g}",
                        "max_component_volume_ml": f"{components['max_component_volume_ml']:.10g}",
                        "component_count_ge_10_voxels": components["component_count_ge_10_voxels"],
                        "component_count_ge_100_voxels": components["component_count_ge_100_voxels"],
                        "component_count_ge_1000_voxels": components["component_count_ge_1000_voxels"],
                        "small_component_count_lt_10_voxels": components["small_component_count_lt_10_voxels"],
                        "small_component_voxel_fraction_lt_10_voxels": f"{components['small_component_voxel_fraction_lt_10_voxels']:.10g}",
                        "max_probability": f"{float(prob.max()):.10g}",
                        "mean_probability": f"{float(prob.mean()):.10g}",
                        "runtime_seconds": f"{time.time() - start:.3f}",
                        "status": "success",
                        "error_type": "",
                        "error_message": "",
                    }
                )
                del inp, logits, prob, mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                out_rows.append(error_row(job, start, exc))
        iterator.close()
    finally:
        del model, state, state_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out_rows


def worker_main(worker_index: int, gpu_id: int | None, jobs: list[dict[str, Any]], sw_batch_size: int, queue: Any) -> None:
    try:
        import torch

        if torch.cuda.is_available() and gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        rows: list[dict[str, Any]] = []
        groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for job in jobs:
            key = (job["architecture"], job["training_condition"], int(job["fold"]))
            groups.setdefault(key, []).append(job)
        for group_jobs in groups.values():
            try:
                rows.extend(infer_group(group_jobs, device, sw_batch_size, worker_index))
            except Exception as exc:
                for job in group_jobs:
                    rows.append(error_row(job, time.time(), exc))
        queue.put({"worker_index": worker_index, "rows": rows, "error": ""})
    except Exception:
        queue.put({"worker_index": worker_index, "rows": [], "error": traceback.format_exc()})


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    import statistics

    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    import statistics

    return statistics.median(values) if values else None


def std(values: list[float]) -> float | None:
    import statistics

    return statistics.stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None)


def std_population(values: list[float]) -> float | None:
    import statistics

    return statistics.pstdev(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def iqr(values: list[float]) -> float | None:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = parse_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("architecture", "")), str(row.get("training_condition", "")), str(row.get("acquisition_type", "")))
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (arch, training, acquisition), group_rows in sorted(grouped.items()):
        success_rows = [row for row in group_rows if row.get("status") == "success"]
        volumes = values(success_rows, "predicted_lesion_volume_ml")
        lcc_volumes = values(success_rows, "largest_cc_volume_ml")
        component_counts = values(success_rows, "connected_component_count_26conn")
        lcc_fractions = values(success_rows, "largest_cc_fraction_of_prediction")
        components_ge_10 = values(success_rows, "component_count_ge_10_voxels")
        components_ge_100 = values(success_rows, "component_count_ge_100_voxels")
        components_ge_1000 = values(success_rows, "component_count_ge_1000_voxels")
        small_component_fractions = values(success_rows, "small_component_voxel_fraction_lt_10_voxels")
        positives = values(success_rows, "scan_positive")
        scans = {row.get("scan_id", "") for row in success_rows}
        subjects = {row.get("subject_id", "") for row in success_rows}
        summary_rows.append(
            {
                "architecture": arch,
                "training_condition": training,
                "acquisition_type": acquisition,
                "n_fold_predictions": len(group_rows),
                "n_success": len(success_rows),
                "n_error": len(group_rows) - len(success_rows),
                "n_scans": len(scans),
                "n_unique_subjects": len(subjects),
                "scan_positive_rate_fold_level": format_number(mean(positives)),
                "mean_fold_predicted_lesion_volume_ml": format_number(mean(volumes)),
                "median_fold_predicted_lesion_volume_ml": format_number(median(volumes)),
                "std_fold_predicted_lesion_volume_ml": format_number(std(volumes)),
                "iqr_fold_predicted_lesion_volume_ml": format_number(iqr(volumes)),
                "max_fold_predicted_lesion_volume_ml": format_number(max(volumes) if volumes else None),
                "mean_fold_largest_cc_volume_ml": format_number(mean(lcc_volumes)),
                "median_fold_largest_cc_volume_ml": format_number(median(lcc_volumes)),
                "std_fold_largest_cc_volume_ml": format_number(std(lcc_volumes)),
                "iqr_fold_largest_cc_volume_ml": format_number(iqr(lcc_volumes)),
                "max_fold_largest_cc_volume_ml": format_number(max(lcc_volumes) if lcc_volumes else None),
                "mean_connected_component_count_26conn": format_number(mean(component_counts)),
                "median_connected_component_count_26conn": format_number(median(component_counts)),
                "std_connected_component_count_26conn": format_number(std_population(component_counts)),
                "iqr_connected_component_count_26conn": format_number(iqr(component_counts)),
                "max_connected_component_count_26conn": format_number(max(component_counts) if component_counts else None),
                "mean_largest_cc_fraction_of_prediction": format_number(mean(lcc_fractions)),
                "median_largest_cc_fraction_of_prediction": format_number(median(lcc_fractions)),
                "std_largest_cc_fraction_of_prediction": format_number(std_population(lcc_fractions)),
                "iqr_largest_cc_fraction_of_prediction": format_number(iqr(lcc_fractions)),
                "mean_component_count_ge_10_voxels": format_number(mean(components_ge_10)),
                "mean_component_count_ge_100_voxels": format_number(mean(components_ge_100)),
                "mean_component_count_ge_1000_voxels": format_number(mean(components_ge_1000)),
                "mean_small_component_voxel_fraction_lt_10_voxels": format_number(mean(small_component_fractions)),
            }
        )
    return summary_rows


def planned_output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "long_csv": run_dir / "mrart_fold_lcc_stats_long.csv",
        "summary_csv": run_dir / "mrart_fold_lcc_stats_summary.csv",
        "run_info_json": run_dir / "mrart_fold_lcc_run_info.json",
        "errors_csv": run_dir / "mrart_fold_lcc_errors.csv",
    }


def main() -> None:
    args = parse_args()
    arches = normalize_arches(args.arch)
    training_conditions = normalize_training(args.training)
    folds = parse_folds(args.folds)
    gpus = parse_gpus(args.gpus)
    run_id = args.run_id or f"{utc_stamp()}_fold_lcc_stats"
    run_dir = args.run_dir or args.output_dir or (args.output_root / run_id)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_dir}")

    jobs, discovery_info = build_jobs(args.data_root, args.checkpoint_root, arches, training_conditions, folds, args.max_subjects)
    manifest_summary = discovery_info["manifest_summary"]
    print(
        "MR-ART dataset counts: "
        f"subjects={manifest_summary['subjects_discovered']} "
        f"complete_triplets={manifest_summary['complete_triplet_subjects_total']} "
        f"subjects_missing_one_or_more={manifest_summary['subjects_missing_one_or_more_acquisitions']} "
        f"missing_files={manifest_summary['missing_expected_acquisition_files_total']} "
        f"missing_by_acquisition={json.dumps(manifest_summary['missing_acquisition_counts'], sort_keys=True)}",
        flush=True,
    )
    if args.inventory_only:
        inventory_path = run_dir / "mrart_dataset_inventory.json"
        write_json(
            inventory_path,
            {
                "timestamp_utc": utc_now(),
                "data_root": str(args.data_root),
                "acquisitions": list(ACQUISITIONS),
                **discovery_info,
            },
        )
        print(f"Wrote {inventory_path}")
        return

    output_paths = planned_output_paths(run_dir)
    check_not_ignored(output_paths.values())
    expected_rows = len(jobs)
    if not jobs:
        raise RuntimeError("No fold prediction jobs were planned")

    workers = max(1, len(gpus))
    partitions = [jobs[index::workers] for index in range(workers)]
    start = time.time()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for index, partition in enumerate(partitions):
        gpu_id = gpus[index] if index < len(gpus) else None
        process = ctx.Process(target=worker_main, args=(index, gpu_id, partition, args.sw_batch_size, queue))
        process.start()
        processes.append(process)

    worker_payloads = []
    received_workers: set[int] = set()
    while len(worker_payloads) < len(processes):
        try:
            payload = queue.get(timeout=30)
            worker_payloads.append(payload)
            received_workers.add(int(payload.get("worker_index", -1)))
            continue
        except queue_module.Empty:
            pass

        dead_without_payload = [
            (index, process.exitcode)
            for index, process in enumerate(processes)
            if index not in received_workers and not process.is_alive() and process.exitcode != 0
        ]
        if dead_without_payload:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10)
            detail = ", ".join(f"worker={index},exitcode={exitcode}" for index, exitcode in dead_without_payload)
            raise RuntimeError(f"MR-ART fold worker died before reporting progress/results: {detail}")

    for process in processes:
        process.join()
        if process.exitcode not in {0, None}:
            raise RuntimeError(f"MR-ART fold worker exited nonzero after reporting: worker={processes.index(process)}, exitcode={process.exitcode}")

    rows: list[dict[str, Any]] = []
    worker_errors = []
    for payload in sorted(worker_payloads, key=lambda item: item["worker_index"]):
        rows.extend(payload["rows"])
        if payload["error"]:
            worker_errors.append(payload)

    rows.sort(
        key=lambda row: (
            str(row.get("architecture", "")),
            str(row.get("training_condition", "")),
            str(row.get("subject_id", "")),
            str(row.get("acquisition_id", "")),
            int(row.get("fold", -1)) if str(row.get("fold", "")).isdigit() else -1,
        )
    )
    summary_rows = build_summary(rows)
    error_rows = [row for row in rows if row.get("status") != "success"]

    run_info = {
        "command": " ".join(sys.argv),
        "timestamp_utc": utc_now(),
        "run_dir": str(run_dir),
        "selected_architectures": arches,
        "selected_training_conditions": training_conditions,
        "selected_folds": folds,
        "selected_gpus": gpus,
        "data_root": str(args.data_root),
        "checkpoint_root": str(args.checkpoint_root),
        "threshold": THRESHOLD,
        "component_connectivity": "26-neighborhood",
        "component_std_ddof": 0,
        "empty_component_stat_policy": "component counts, fractions, means, medians, and stds are 0 when predicted_lesion_voxel_count is 0",
        "sw_batch_size": args.sw_batch_size,
        "git_commit": get_git_commit(),
        "torch_cuda_info": torch_cuda_info(),
        "nvidia_smi": run_nvidia_smi(),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "n_success": sum(1 for row in rows if row.get("status") == "success"),
        "n_error": len(error_rows),
        "worker_errors": worker_errors,
        "runtime_seconds": time.time() - start,
        "outputs": {key: str(path) for key, path in output_paths.items()},
        **discovery_info,
    }

    write_csv(output_paths["long_csv"], LONG_FIELDNAMES, rows)
    write_csv(output_paths["summary_csv"], SUMMARY_FIELDNAMES, summary_rows)
    write_csv(output_paths["errors_csv"], LONG_FIELDNAMES, error_rows)
    write_json(output_paths["run_info_json"], run_info)

    print(f"Wrote {output_paths['long_csv']} rows={len(rows)} cols={len(LONG_FIELDNAMES)}")
    print(f"Wrote {output_paths['summary_csv']} rows={len(summary_rows)} cols={len(SUMMARY_FIELDNAMES)}")
    print(f"Wrote {output_paths['errors_csv']} rows={len(error_rows)} cols={len(LONG_FIELDNAMES)}")
    print(f"Wrote {output_paths['run_info_json']}")

    n_success = run_info["n_success"]
    if worker_errors or len(rows) != expected_rows or n_success != expected_rows:
        raise SystemExit(
            "Fold-level MR-ART rerun incomplete: "
            f"expected_rows={expected_rows}, actual_rows={len(rows)}, "
            f"n_success={n_success}, n_error={len(error_rows)}, "
            f"worker_errors={len(worker_errors)}. "
            f"Inspect {output_paths['errors_csv']} and {output_paths['run_info_json']}."
        )


if __name__ == "__main__":
    main()
