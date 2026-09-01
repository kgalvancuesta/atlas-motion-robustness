"""Self-interpreting Linux/CUDA determinism preflight for PACE A100."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import nibabel as nb
import numpy as np

from .challenge import apply_recipe, load_or_create_validated_cache, sample_legacy_recipe
from .core import array_sha256, configure_strict_determinism, read_json, sha256_file
from .inference import build_model
from .mrart import false_positive_metrics
from .preprocessing import preprocess_volume


class Preflight:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.results: list[dict[str, Any]] = []

    def check(self, name: str, function: Callable[[], Any]) -> Any | None:
        try:
            diagnostics = function()
            self.results.append({"check": name, "status": "PASS", "diagnostics": diagnostics})
            print(f"PASS: {name}: {json.dumps(diagnostics, sort_keys=True, default=str)}", flush=True)
            return diagnostics
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failures.append(f"{name}: {message}")
            self.results.append({"check": name, "status": "FAIL", "diagnostics": message})
            print(f"FAIL: {name}: {message}", flush=True)
            return None


def _find_checkpoint(checkpoint_root: Path, repo_root: Path, model: str) -> Path:
    candidates = [
        checkpoint_root / "runs" / model / "run_kfold_01" / "checkpoints" / "best.pt",
        checkpoint_root / model / "run_kfold_01" / "checkpoints" / "best.pt",
        repo_root / "runs" / model / "run_kfold_01" / "checkpoints" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Historical fold-0 checkpoint not found for {model}; checked: {candidates}")


def _checkpoint_patch_size(checkpoint: Path) -> tuple[int, int, int]:
    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config" / "train_config.json"
    if config_path.exists():
        config = read_json(config_path)
        value = config.get("args", {}).get("patch_size")
        if value:
            return tuple(int(item) for item in value)
    return (128, 128, 128)


def _state_dict(checkpoint: Path, device):
    import torch

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    return state["model"] if isinstance(state, dict) and "model" in state else state


def _prediction_metrics(prediction: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    metrics = false_positive_metrics(prediction, voxel_volume_mm3)
    return {
        "binary_sha256": array_sha256(prediction.astype(np.uint8)),
        "predicted_voxels": metrics["predicted_voxels"],
        "predicted_volume_ml": metrics["predicted_volume_ml"],
        "connected_component_count_26": metrics["connected_component_count_26"],
        "largest_connected_component_voxels_26": metrics["largest_connected_component_voxels_26"],
    }


def _repeat_model_inference(model_name: str, checkpoint: Path, data: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    import torch
    from monai.inferers import sliding_window_inference

    device = torch.device("cuda")
    configure_strict_determinism(9001)
    model = build_model(model_name).to(device)
    model.load_state_dict(_state_dict(checkpoint, device))
    model.eval()
    patch_size = _checkpoint_patch_size(checkpoint)
    inp = torch.from_numpy(data[None, None, ...]).float().to(device)

    def predict() -> np.ndarray:
        configure_strict_determinism(9001)
        with torch.inference_mode():
            logits = sliding_window_inference(inp, roi_size=patch_size, sw_batch_size=1, predictor=model)
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("non-finite logits")
            return torch.sigmoid(logits).ge(0.5).to(torch.uint8).cpu().numpy()[0, 0]

    first = predict()
    second = predict()
    first_metrics = _prediction_metrics(first, voxel_volume_mm3)
    second_metrics = _prediction_metrics(second, voxel_volume_mm3)
    if not np.array_equal(first, second) or first_metrics != second_metrics:
        raise RuntimeError(f"repeat mismatch: first={first_metrics}, second={second_metrics}")
    return {"model": model_name, "checkpoint": str(checkpoint), **first_metrics}


def _first_atlas_case(data_root: Path, split_path: Path) -> tuple[str, Path]:
    split = read_json(split_path)
    subject = str(split["folds"][0].get("test_ids", split["folds"][0].get("heldout_ids"))[0])
    path = (
        data_root
        / "train"
        / "derivatives"
        / "ATLAS"
        / subject
        / "ses-1"
        / "anat"
        / f"{subject}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return subject, path


def _first_mrart_case(data_root: Path) -> tuple[str, Path]:
    for subject_dir in sorted((data_root / "mr-art").glob("sub-*")):
        subject = subject_dir.name
        path = subject_dir / "anat" / f"{subject}_acq-standard_T1w.nii.gz"
        if path.exists():
            return subject, path
    raise FileNotFoundError(f"No MR-ART standard acquisition under {data_root / 'mr-art'}")


def _run_ddp_smoke(repo_root: Path, work_dir: Path) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for run_index in (1, 2):
        output = work_dir / f"ddp-smoke-{run_index}.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "reproducibility.ddp_smoke",
            "--output",
            str(output),
            "--seed",
            "9001",
        ]
        result = subprocess.run(command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (work_dir / f"ddp-smoke-{run_index}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"DDP smoke run {run_index} failed; see {work_dir / f'ddp-smoke-{run_index}.log'}")
        outputs.append(read_json(output))
    if outputs[0]["model_state_sha256"] != outputs[1]["model_state_sha256"]:
        raise RuntimeError(f"DDP training state differs across repeats: {outputs}")
    if outputs[0]["world_size"] != 2 or outputs[0]["batch_size_per_gpu"] != 1:
        raise RuntimeError(f"DDP smoke did not use required semantics: {outputs[0]}")
    return {
        "world_size": 2,
        "batch_size_per_gpu": 1,
        "amp": True,
        "repeated_model_state_sha256": outputs[0]["model_state_sha256"],
        "logs": [str(work_dir / "ddp-smoke-1.log"), str(work_dir / "ddp-smoke-2.log")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick, self-interpreting two-A100 determinism preflight.")
    parser.add_argument("--data-root", default=os.environ.get("ATLAS_DATA_ROOT", "data"))
    parser.add_argument("--split-path", default="splits/atlas_5fold_lesion_quartile_excluding_known_issues.json")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root)
    split_path = Path(args.split_path)
    checkpoint_root = Path(args.checkpoint_root)
    scratch_root = Path(args.scratch_root).resolve()
    run_id = datetime.now(timezone.utc).strftime("test-pace-preflight-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    work_dir = scratch_root / "preflight_runs" / run_id
    work_dir.mkdir(parents=True, exist_ok=False)
    print(f"PACE determinism preflight: {run_id}")
    print(f"Diagnostics directory: {work_dir}")
    preflight = Preflight()

    def environment_check():
        import monai
        import numpy
        import torch
        import torchio

        versions = {
            "torch": torch.__version__,
            "torchio": torchio.__version__,
            "monai": monai.__version__,
            "numpy": numpy.__version__,
        }
        expected = {"torchio": "0.20.23", "monai": "1.5.2", "numpy": "2.0.2"}
        mismatches = {key: (versions[key], value) for key, value in expected.items() if versions[key] != value}
        if not versions["torch"].split("+")[0] == "2.8.0":
            mismatches["torch"] = (versions["torch"], "2.8.0")
        if mismatches:
            raise RuntimeError(f"pinned package mismatch: {mismatches}")
        if not sys.platform.startswith("linux"):
            raise RuntimeError(f"authoritative preflight requires Linux, got {sys.platform}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError(f"expected exactly 2 visible CUDA GPUs, got {torch.cuda.device_count()}")
        devices = []
        for index in range(2):
            properties = torch.cuda.get_device_properties(index)
            info = {"index": index, "name": properties.name, "memory_gib": properties.total_memory / 2**30}
            if "A100" not in properties.name:
                raise RuntimeError(f"GPU {index} is not an A100: {info}")
            devices.append(info)
        runtime = configure_strict_determinism(9001)
        return {"versions": versions, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "devices": devices, "runtime": runtime}

    preflight.check("Linux/CUDA/cuDNN/two-A100 environment", environment_check)

    atlas_context: dict[str, Any] = {}

    def reconstruction_check():
        subject, source_path = _first_atlas_case(data_root, split_path)
        image = nb.load(str(source_path))
        raw = image.get_fdata(dtype=np.float32)
        recipe = None
        selected_replicate = None
        for replicate in range(100):
            candidate = sample_legacy_recipe(
                tuple(int(value) for value in raw.shape),
                global_seed=9001,
                subject_id=subject,
                fold=0,
                replicate_id=replicate,
                namespace="preflight_evaluation_challenge",
            )
            if not candidate["noop"]:
                recipe = candidate
                selected_replicate = replicate
                break
        if recipe is None:
            raise RuntimeError("could not sample a non-noop representative recipe in 100 deterministic attempts")
        first = apply_recipe(raw, recipe)
        second = apply_recipe(raw, recipe)
        if not np.array_equal(first, second):
            raise RuntimeError("concrete TorchIO recipe reconstruction is not bit-identical")
        for mode in ("artifact_then_normalize", "normalize_then_artifact"):
            value, diagnostics = preprocess_volume(raw, mode=mode, recipe=recipe)
            if not np.isfinite(value).all() or diagnostics["normalization_count"] != 1:
                raise RuntimeError(f"corrected normalization check failed for {mode}: {diagnostics}")
        atlas_context.update(
            {
                "subject": subject,
                "source_path": source_path,
                "raw": raw,
                "recipe": recipe,
                "source_sha256": sha256_file(source_path),
                "voxel_volume_mm3": float(np.prod(image.header.get_zooms()[:3])),
            }
        )
        return {
            "subject": subject,
            "representative_replicate": selected_replicate,
            "recipe_id": recipe["recipe_id"],
            "applied_transforms": [entry["name"] for entry in recipe["applied_transforms"]],
            "reconstruction_sha256": array_sha256(first),
        }

    preflight.check("deterministic concrete corruption reconstruction", reconstruction_check)

    def memory_equivalence_check():
        if not atlas_context:
            raise RuntimeError("reconstruction prerequisite failed")
        raw = atlas_context["raw"]
        recipe = atlas_context["recipe"]
        low, _diagnostics = preprocess_volume(raw, mode="artifact_then_normalize", recipe=recipe)
        high_first, first_status = load_or_create_validated_cache(
            cache_root=work_dir / "cache",
            challenge_id="test-preflight-" + recipe["recipe_id"],
            normalization_mode="artifact_then_normalize",
            fold=0,
            replicate_id=int(recipe["replicate_id"]),
            subject_id=atlas_context["subject"],
            source_sha256=atlas_context["source_sha256"],
            recipe_id=recipe["recipe_id"],
            builder=lambda: preprocess_volume(raw, mode="artifact_then_normalize", recipe=recipe)[0],
        )
        high_second, second_status = load_or_create_validated_cache(
            cache_root=work_dir / "cache",
            challenge_id="test-preflight-" + recipe["recipe_id"],
            normalization_mode="artifact_then_normalize",
            fold=0,
            replicate_id=int(recipe["replicate_id"]),
            subject_id=atlas_context["subject"],
            source_sha256=atlas_context["source_sha256"],
            recipe_id=recipe["recipe_id"],
            builder=lambda: (_ for _ in ()).throw(RuntimeError("cache should have been reused")),
        )
        if not np.array_equal(low, high_first) or not np.array_equal(high_first, high_second):
            raise RuntimeError("memory-low and memory-high inputs are not bit-identical")
        return {"array_sha256": array_sha256(low), "first_cache_status": first_status, "second_cache_status": second_status}

    preflight.check("memory-low versus memory-high scientific equivalence", memory_equivalence_check)

    inference_results: dict[str, Any] = {}
    for model_name in ("base_cnn", "uxnet", "mednext", "swin"):
        def check_model(model_name=model_name):
            if not atlas_context:
                raise RuntimeError("reconstruction prerequisite failed")
            checkpoint = _find_checkpoint(checkpoint_root, repo_root, model_name)
            legacy_input, _diagnostics = preprocess_volume(
                atlas_context["raw"], mode="legacy", recipe=atlas_context["recipe"]
            )
            corrected_input, _corrected_diagnostics = preprocess_volume(
                atlas_context["raw"], mode="artifact_then_normalize", recipe=atlas_context["recipe"]
            )
            legacy_result = _repeat_model_inference(
                model_name,
                checkpoint,
                legacy_input,
                atlas_context["voxel_volume_mm3"],
            )
            corrected_result = _repeat_model_inference(
                model_name,
                checkpoint,
                corrected_input,
                atlas_context["voxel_volume_mm3"],
            )
            result = {
                "model": model_name,
                "legacy_fixed_distribution": legacy_result,
                "artifact_then_normalize_numerics": corrected_result,
            }
            inference_results[model_name] = result
            return result

        preflight.check(f"repeated CUDA inference ({model_name})", check_model)

    def mrart_check():
        subject, source_path = _first_mrart_case(data_root)
        image = nb.load(str(source_path))
        raw = image.get_fdata(dtype=np.float32)
        value, _diagnostics = preprocess_volume(raw, mode="artifact_then_normalize", recipe=None)
        checkpoint = _find_checkpoint(checkpoint_root, repo_root, "mednext")
        result = _repeat_model_inference(
            "mednext", checkpoint, value, float(np.prod(image.header.get_zooms()[:3]))
        )
        return {"analysis_dataset": "MR-ART healthy controls", "subject": subject, **result}

    preflight.check("MR-ART rerun consistency pathway", mrart_check)
    preflight.check("two-GPU deterministic AMP training/RNG smoke", lambda: _run_ddp_smoke(repo_root, work_dir))

    summary = {
        "run_id": run_id,
        "results": preflight.results,
        "failures": preflight.failures,
        "safe_to_start": not preflight.failures,
        "diagnostics_directory": str(work_dir),
    }
    (work_dir / "preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if preflight.failures:
        print("Required checks still unverified or failed:")
        for failure in preflight.failures:
            print(f"  - {failure}")
        print("Smallest next test: fix the listed failure and rerun this same quick preflight.")
        print("UNSAFE TO START FULL RUN")
        return 1
    print("SAFE TO START FULL RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
