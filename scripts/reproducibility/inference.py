"""Strict corrected ATLAS inference against immutable challenge recipes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np

from .challenge import load_or_create_validated_cache, recipe_for
from .core import (
    ARTIFACT_SCHEMA_VERSION,
    CompatibilityError,
    ConflictError,
    PREPROCESSING_VERSION,
    CorrectedExperimentError,
    array_sha256,
    completed_artifact_status,
    configure_strict_determinism,
    read_json,
    sha256_file,
    software_provenance,
    stable_hash,
    subject_ids_hash,
    validate_metadata,
    write_json_new,
)
from .experiment import fold_definition
from .preprocessing import preprocess_volume
from .rng import derive_seed


def build_model(model_name: str):
    if model_name == "base_cnn":
        from models.base_cnn_model import build_base_model

        return build_base_model()
    if model_name == "uxnet":
        from models.uxnet_model import build_uxnet_model

        return build_uxnet_model()
    if model_name == "mednext":
        from models.mednext_model import build_mednext_model

        return build_mednext_model()
    if model_name == "swin":
        from models.swin_model import build_swin_model

        return build_swin_model()
    raise CorrectedExperimentError(f"Unsupported model: {model_name}")


def checkpoint_path(root: Path, model: str, regime: str, fold: int) -> Path:
    return root / "checkpoints" / model / regime / f"fold-{fold}" / "checkpoints" / "best.pt"


def checkpoint_expected_metadata(definition: dict[str, Any], *, model: str, regime: str, fold: int) -> dict[str, Any]:
    config = definition["model_configurations"][model]
    return {
        "experiment_id": definition["experiment_id"],
        "generation": "corrected_experiment_v1",
        "model": model,
        "model_configuration_id": config["model_configuration_id"],
        "fold": int(fold),
        "training_regime": regime,
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
        "global_seed": int(definition["seeds"]["global_seed"]),
        **{key: definition["configuration"][key] for key in ("checkpoint_selection", "numerical_policy")
           if key in definition["configuration"]},
    }


def load_checkpoint(
    path: Path,
    *,
    definition: dict[str, Any],
    model: str,
    regime: str,
    fold: int,
    device,
):
    import torch

    if not path.exists():
        raise CorrectedExperimentError(f"Corrected checkpoint is missing: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or "model" not in state:
        raise CompatibilityError(f"Checkpoint does not contain a corrected model payload: {path}")
    metadata = state.get("corrected_experiment_metadata")
    if not isinstance(metadata, dict):
        raise CompatibilityError(
            f"Checkpoint {path} lacks corrected-experiment metadata. Historical checkpoints may only be used by the explicit "
            "fixed-legacy inference compatibility path, not corrected treatment comparisons."
        )
    validate_metadata(
        metadata,
        checkpoint_expected_metadata(definition, model=model, regime=regime, fold=fold),
        context=str(path),
    )
    return state


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_challenge_id(definition: dict[str, Any]) -> str:
    return stable_hash(
        {
            "kind": "clean_control",
            "experiment_id": definition["experiment_id"],
            "source_inventory_id": definition["dataset"]["source_inventory_id"],
            "preprocessing_version": PREPROCESSING_VERSION,
        }
    )


def resolve_challenges(definition: dict[str, Any], requested: str | None) -> list[dict[str, Any]]:
    clean = {"name": "clean", "challenge_id": clean_challenge_id(definition), "corrupted": False}
    corrupted = {
        "name": definition["challenge"]["name"],
        "challenge_id": definition["challenge"]["challenge_id"],
        "corrupted": True,
    }
    if requested is None or requested == "all":
        return [clean, corrupted]
    matches = [entry for entry in (clean, corrupted) if requested in (entry["name"], entry["challenge_id"])]
    if len(matches) != 1:
        raise CorrectedExperimentError(f"Unknown challenge selector {requested!r}; inference will not create a missing challenge")
    return matches


def _source_path(data_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    return candidate if candidate.is_absolute() else data_root / candidate


def _mask_path(source_path: Path) -> Path:
    return source_path.with_name(
        source_path.name.replace("_T1w.nii.gz", "_label-L_desc-T1lesion_mask.nii.gz")
    )


def _save_nifti_new(array: np.ndarray, reference: nb.Nifti1Image, path: Path) -> None:
    if path.exists():
        raise ConflictError(f"Refusing to overwrite prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".nii.gz", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nb.save(nb.Nifti1Image(array.astype(np.uint8, copy=False), reference.affine, header), str(temporary_path))
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ConflictError(f"Refusing to overwrite prediction: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    from scipy import ndimage

    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(pred, truth).sum())
    pred_voxels = int(pred.sum())
    target_voxels = int(truth.sum())
    denominator = pred_voxels + target_voxels
    dice = float((2 * intersection) / denominator) if denominator else 1.0
    labels, component_count = ndimage.label(pred, structure=np.ones((3, 3, 3), dtype=bool))
    counts = np.bincount(labels.ravel()) if component_count else np.asarray([0])
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    return {
        "dice": dice,
        "predicted_voxels": pred_voxels,
        "target_voxels": target_voxels,
        "predicted_volume_ml": float(pred_voxels * voxel_volume_mm3 / 1000.0),
        "connected_component_count_26": int(component_count),
        "largest_connected_component_voxels_26": largest,
        "largest_connected_component_ml_26": float(largest * voxel_volume_mm3 / 1000.0),
    }


def _preprocessed_input(
    *,
    source_path: Path,
    source_sha256: str,
    recipe: dict[str, Any] | None,
    challenge_id: str,
    normalization_mode: str,
    memory_mode: str,
    cache_root: Path | None,
    fold: int,
    replicate_id: int,
    subject_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if sha256_file(source_path) != source_sha256:
        raise CompatibilityError(f"Source checksum differs from immutable definition: {source_path}")
    raw = nb.load(str(source_path)).get_fdata(dtype=np.float32)

    def builder():
        return preprocess_volume(raw, mode=normalization_mode, recipe=recipe)[0]

    if memory_mode == "low":
        value, diagnostics = preprocess_volume(raw, mode=normalization_mode, recipe=recipe)
        return value, {"memory_mode": "low", "cache_status": "not_used", "preprocessing": diagnostics}
    if memory_mode != "high" or cache_root is None:
        raise CorrectedExperimentError("memory-high requires an explicit verified cache root")
    value, status = load_or_create_validated_cache(
        cache_root=cache_root,
        challenge_id=challenge_id,
        normalization_mode=normalization_mode,
        fold=fold,
        replicate_id=replicate_id,
        subject_id=subject_id,
        source_sha256=source_sha256,
        recipe_id=recipe["recipe_id"] if recipe else "clean",
        builder=builder,
    )
    return value, {"memory_mode": "high", "cache_status": status, "array_sha256": array_sha256(value)}


def _case_expected_metadata(
    definition: dict[str, Any],
    *,
    model: str,
    regime: str,
    fold: int,
    subject_id: str,
    challenge_id: str,
    replicate_id: int,
    checkpoint_sha256: str,
    checkpoint_generation: str,
) -> dict[str, Any]:
    model_config = definition["model_configurations"][model]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": definition["experiment_id"],
        "generation": "corrected_experiment_v1",
        "model": model,
        "model_configuration_id": model_config["model_configuration_id"],
        "fold": int(fold),
        "training_regime": regime,
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
        "subject_id": subject_id,
        "challenge_id": challenge_id,
        "corruption_replicate": int(replicate_id),
        "global_seed": int(definition["seeds"]["global_seed"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_generation": checkpoint_generation,
    }


def run_inference_task(
    *,
    repo_root: Path,
    root: Path,
    definition: dict[str, Any],
    challenge_manifest: dict[str, Any],
    data_root: Path,
    cache_root: Path | None,
    memory_mode: str,
    model_name: str,
    regime: str,
    fold: int,
    challenge: dict[str, Any],
    replicate_id: int,
    sw_batch_size: int = 1,
    repeat_check: bool = False,
    checkpoint_override: Path | None = None,
    checkpoint_generation: str = "corrected_experiment_v1",
) -> dict[str, Any]:
    import torch
    from monai.inferers import sliding_window_inference

    fold_entry = fold_definition(definition, fold)
    subject_ids = list(fold_entry["test_ids"])
    global_seed = int(definition["seeds"]["global_seed"])
    inference_seed = derive_seed(
        global_seed,
        "inference_runtime",
        model=model_name,
        training_regime=regime,
        fold=fold,
        challenge_id=challenge["challenge_id"],
        replicate_id=replicate_id,
    )
    runtime = configure_strict_determinism(inference_seed)
    device = _device()
    checkpoint = checkpoint_override or checkpoint_path(root, model_name, regime, fold)
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.exists() else "missing"
    if checkpoint_generation == "corrected_experiment_v1":
        state = load_checkpoint(
            checkpoint,
            definition=definition,
            model=model_name,
            regime=regime,
            fold=fold,
            device=device,
        )
        state_dict = state["model"]
    elif checkpoint_generation == "historical_phase1":
        if definition["configuration"]["normalization_mode"] != "legacy":
            raise CompatibilityError(
                "Historical checkpoints are permitted only in an explicit legacy-normalization experiment definition"
            )
        if not checkpoint.exists():
            raise CorrectedExperimentError(f"Historical checkpoint is missing: {checkpoint}")
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    else:
        raise CorrectedExperimentError(f"Unsupported checkpoint generation: {checkpoint_generation}")
    model = build_model(model_name).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    patch_size = tuple(int(value) for value in definition["model_configurations"][model_name]["patch_size"])

    inventory = {entry["subject_id"]: entry for entry in definition["dataset"]["source_inventory"]}
    task_dir = (
        root
        / "inference"
        / challenge["challenge_id"]
        / model_name
        / regime
        / f"fold-{fold}"
        / f"replicate-{replicate_id}"
    )
    task_metadata_path = task_dir / "task.metadata.json"
    task_expected = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "inference_task",
        "experiment_id": definition["experiment_id"],
        "model": model_name,
        "model_configuration_id": definition["model_configurations"][model_name]["model_configuration_id"],
        "fold": int(fold),
        "training_regime": regime,
        "challenge_id": challenge["challenge_id"],
        "corruption_replicate": int(replicate_id),
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
        "subject_ids_hash": subject_ids_hash(subject_ids),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_generation": checkpoint_generation,
    }
    if task_metadata_path.exists():
        existing = read_json(task_metadata_path)
        validate_metadata(existing, task_expected, context=str(task_metadata_path))
        sidecars = list(existing.get("case_metadata_paths", []))
        if len(sidecars) != len(subject_ids):
            raise ConflictError(
                f"Completed inference task has {len(sidecars)} case records for {len(subject_ids)} subjects"
            )
        completed_subjects: list[str] = []
        for sidecar in sidecars:
            sidecar_path = root / sidecar
            if not sidecar_path.exists():
                raise ConflictError(f"Completed inference task references missing case metadata: {sidecar}")
            case_metadata = read_json(sidecar_path)
            subject_id = str(case_metadata.get("subject_id"))
            if subject_id not in subject_ids or subject_id in completed_subjects:
                raise ConflictError(f"Invalid subject coverage in completed inference task: {subject_id}")
            validate_metadata(
                case_metadata,
                _case_expected_metadata(
                    definition,
                    model=model_name,
                    regime=regime,
                    fold=fold,
                    subject_id=subject_id,
                    challenge_id=challenge["challenge_id"],
                    replicate_id=replicate_id,
                    checkpoint_sha256=checkpoint_hash,
                    checkpoint_generation=checkpoint_generation,
                ),
                context=str(sidecar_path),
            )
            prediction_path = root / str(case_metadata.get("prediction_path", ""))
            if not prediction_path.is_file() or sha256_file(prediction_path) != case_metadata.get("prediction_sha256"):
                raise ConflictError(f"Completed inference prediction is missing or changed: {prediction_path}")
            completed_subjects.append(subject_id)
        if sorted(completed_subjects) != sorted(subject_ids):
            raise ConflictError("Completed inference task subject identities conflict with the immutable fold")
        return {"status": "skipped-compatible", **task_expected}

    task_dir.mkdir(parents=True, exist_ok=True)
    case_metadata_paths: list[str] = []
    skipped = 0
    created = 0
    for subject_id in subject_ids:
        source = inventory[subject_id]
        source_path = _source_path(data_root, source["relative_path"])
        mask_path = (
            _source_path(data_root, source["mask_relative_path"])
            if source.get("mask_relative_path")
            else _mask_path(source_path)
        )
        if not source_path.exists() or not mask_path.exists():
            raise CorrectedExperimentError(f"ATLAS source/mask missing for {subject_id}: {source_path}, {mask_path}")
        if source.get("mask_sha256") and sha256_file(mask_path) != source["mask_sha256"]:
            raise CompatibilityError(f"ATLAS mask checksum differs from immutable definition: {mask_path}")
        recipe = (
            recipe_for(challenge_manifest, subject_id=subject_id, fold=fold, replicate_id=replicate_id)
            if challenge["corrupted"]
            else None
        )
        expected = _case_expected_metadata(
            definition,
            model=model_name,
            regime=regime,
            fold=fold,
            subject_id=subject_id,
            challenge_id=challenge["challenge_id"],
            replicate_id=replicate_id,
            checkpoint_sha256=checkpoint_hash,
            checkpoint_generation=checkpoint_generation,
        )
        prediction_path = task_dir / "predictions" / f"{subject_id}.nii.gz"
        metadata_path = task_dir / "metadata" / f"{subject_id}.json"
        status = completed_artifact_status(metadata_path, expected, [prediction_path])
        if status == "complete":
            metadata = read_json(metadata_path)
            if sha256_file(prediction_path) != metadata.get("prediction_sha256"):
                raise ConflictError(f"Prediction checksum changed after completion: {prediction_path}")
            skipped += 1
            case_metadata_paths.append(str(metadata_path.relative_to(root)))
            continue

        data, cache_diagnostics = _preprocessed_input(
            source_path=source_path,
            source_sha256=source["sha256"],
            recipe=recipe,
            challenge_id=challenge["challenge_id"],
            normalization_mode=definition["configuration"]["normalization_mode"],
            memory_mode=memory_mode,
            cache_root=cache_root,
            fold=fold,
            replicate_id=replicate_id,
            subject_id=subject_id,
        )
        reference = nb.load(str(source_path))
        target = nb.load(str(mask_path)).get_fdata(dtype=np.float32) > 0.5
        voxel_volume = float(np.prod(reference.header.get_zooms()[:3]))
        inp = torch.from_numpy(data[None, None, ...]).float().to(device)
        with torch.inference_mode():
            logits = sliding_window_inference(
                inp,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
            )
            if not bool(torch.isfinite(logits).all().item()):
                raise CorrectedExperimentError(f"Non-finite logits for {subject_id}")
            prediction_tensor = torch.sigmoid(logits).ge(0.5).to(torch.uint8).cpu()
        prediction = prediction_tensor.numpy()[0, 0]
        repeat = None
        if repeat_check:
            configure_strict_determinism(inference_seed)
            with torch.inference_mode():
                repeat_logits = sliding_window_inference(
                    inp,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )
                if not bool(torch.isfinite(repeat_logits).all().item()):
                    raise CorrectedExperimentError(f"Non-finite repeated logits for {subject_id}")
                repeat_prediction = torch.sigmoid(repeat_logits).ge(0.5).to(torch.uint8).cpu().numpy()[0, 0]
            first_repeat_metrics = segmentation_metrics(prediction, target, voxel_volume)
            second_repeat_metrics = segmentation_metrics(repeat_prediction, target, voxel_volume)
            repeat = {
                "binary_equal": bool(np.array_equal(prediction, repeat_prediction)),
                "voxel_count_equal": first_repeat_metrics["predicted_voxels"]
                == second_repeat_metrics["predicted_voxels"],
                "predicted_volume_equal": first_repeat_metrics["predicted_volume_ml"]
                == second_repeat_metrics["predicted_volume_ml"],
                "connected_components_equal": first_repeat_metrics["connected_component_count_26"]
                == second_repeat_metrics["connected_component_count_26"],
                "largest_component_equal": first_repeat_metrics["largest_connected_component_voxels_26"]
                == second_repeat_metrics["largest_connected_component_voxels_26"],
            }
            if not all(repeat.values()):
                raise CorrectedExperimentError(f"Repeated inference differed for {subject_id}: {repeat}")

        metrics = segmentation_metrics(prediction, target, voxel_volume)
        _save_nifti_new(prediction, reference, prediction_path)
        metadata = {
            **expected,
            "artifact_type": "atlas_binary_segmentation",
            "source_relative_path": source["relative_path"],
            "source_sha256": source["sha256"],
            "recipe_id": recipe["recipe_id"] if recipe else None,
            "relevant_derived_seeds": {"inference_runtime": inference_seed},
            "corruption_replicate_ids": [int(replicate_id)],
            "subject_ids": [subject_id],
            "subject_ids_hash": subject_ids_hash([subject_id]),
            "prediction_path": str(prediction_path.relative_to(root)),
            "prediction_sha256": sha256_file(prediction_path),
            "input_array_sha256": array_sha256(data),
            "metrics": metrics,
            "cache": cache_diagnostics,
            "strict_runtime": runtime,
            "repeat_inference": repeat,
            "software_environment": software_provenance(repo_root),
        }
        write_json_new(metadata_path, metadata)
        created += 1
        case_metadata_paths.append(str(metadata_path.relative_to(root)))

    task_metadata = {
        **task_expected,
        "artifact_type": "inference_task",
        "subject_ids": subject_ids,
        "corruption_replicate_ids": [int(replicate_id)],
        "case_metadata_paths": case_metadata_paths,
        "created_cases": created,
        "skipped_compatible_cases": skipped,
        "strict_runtime": runtime,
        "software_environment": software_provenance(repo_root),
    }
    write_json_new(task_metadata_path, task_metadata)
    return {"status": "completed", **task_metadata}
