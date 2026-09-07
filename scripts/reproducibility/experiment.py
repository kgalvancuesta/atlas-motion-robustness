"""Persistent corrected experiment definitions."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import nibabel as nb

from .challenge import build_challenge_manifest
from .core import (
    ConflictError,
    CorrectedExperimentError,
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    read_json,
    sha256_file,
    software_provenance,
    stable_hash,
    utc_now,
    validate_normalization_mode,
    write_immutable_json,
)
from .rng import SEED_DERIVATION_VERSION, derive_seed
from .numerics import NUMERICAL_POLICY

DEFAULT_MODELS = ("base_cnn", "uxnet", "mednext", "swin")
DEFAULT_REGIMES = ("standard", "augmented")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
ORIGINAL_EARLY_STOPPING_PATIENCE = 10
ORIGINAL_VALIDATION_INTERVAL = 1
CHECKPOINT_SELECTION = {
    "subjects": "original_fold_clean_validation_only",
    "identical_across_training_regimes": True,
    "augmentation": False,
    "metric": "mean_subject_binary_dice",
    "threshold": 0.5,
    "threshold_comparison": ">",
    "dice_epsilon": 1e-6,
    "best_rule": "strictly_greater_than_previous_best",
    "early_stopping_metric": "same_clean_validation_dice",
}
MODEL_CONFIGURATIONS = {
    "base_cnn": {
        "entrypoint": "scripts/train_base_cnn.py",
        "inference_name": "baseline",
        "patch_size": [128, 128, 128],
        "loss": "baseline",
    },
    "uxnet": {
        "entrypoint": "scripts/train_uxnet.py",
        "inference_name": "uxnet",
        "patch_size": [128, 128, 128],
        "loss": "dicece",
    },
    "mednext": {
        "entrypoint": "scripts/train_mednext.py",
        "inference_name": "mednext",
        "patch_size": [128, 128, 128],
        "loss": "mednext",
    },
    "swin": {
        "entrypoint": "scripts/train_swin.py",
        "inference_name": "swin",
        "patch_size": [128, 128, 128],
        "loss": "dicece",
    },
}


def validate_training_protocol(definition: dict[str, Any]) -> None:
    """Old corrected definitions require a fresh ID, never silent protocol adoption."""
    for key, expected in (("checkpoint_selection", CHECKPOINT_SELECTION), ("numerical_policy", NUMERICAL_POLICY)):
        if definition.get("configuration", {}).get(key) != expected:
            raise ConflictError(f"Incompatible {key}; prepare a fresh authoritative experiment ID")


def experiment_root(output_root: Path, experiment_id: str) -> Path:
    return output_root / experiment_id


def definition_path(output_root: Path, experiment_id: str) -> Path:
    return experiment_root(output_root, experiment_id) / "definition" / "experiment.json"


def challenge_path(root: Path, challenge_id: str) -> Path:
    return root / "definition" / "challenges" / f"{challenge_id}.json"


def load_experiment(output_root: Path, experiment_id: str) -> tuple[Path, dict[str, Any]]:
    root = experiment_root(output_root, experiment_id)
    path = definition_path(output_root, experiment_id)
    if not path.exists():
        raise CorrectedExperimentError(f"Experiment definition does not exist: {path}. Run the prepare stage first.")
    definition = read_json(path)
    if definition.get("schema_version") != SCHEMA_VERSION or definition.get("experiment_id") != experiment_id:
        raise ConflictError(f"Invalid experiment identity in {path}")
    return root, definition


def _normalize_list(values: list[str] | tuple[str, ...], allowed: tuple[str, ...], label: str) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in allowed:
            raise CorrectedExperimentError(f"Unsupported {label} {value!r}; allowed={allowed}")
        if value not in result:
            result.append(value)
    if not result:
        raise CorrectedExperimentError(f"At least one {label} is required")
    return result


def _fold_entries(split_payload: dict[str, Any], folds: list[int]) -> list[dict[str, Any]]:
    raw_folds = split_payload.get("folds")
    if not isinstance(raw_folds, list):
        raise CorrectedExperimentError("Corrected experiments require the committed CV master split format")
    selected: list[dict[str, Any]] = []
    for fold_index in folds:
        matches = [entry for entry in raw_folds if int(entry.get("fold_index", -1)) == fold_index]
        if len(matches) != 1:
            raise CorrectedExperimentError(f"Expected one split entry for fold {fold_index}; found {len(matches)}")
        entry = matches[0]
        normalized = {
            "fold_index": fold_index,
            "train_ids": [str(value) for value in entry.get("train_ids", [])],
            "val_ids": [str(value) for value in entry.get("val_ids", entry.get("dev_ids", []))],
            "test_ids": [str(value) for value in entry.get("test_ids", entry.get("heldout_ids", []))],
        }
        groups = [set(normalized[key]) for key in ("train_ids", "val_ids", "test_ids")]
        if not all(groups) or groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise CorrectedExperimentError(f"Fold {fold_index} has an empty or overlapping train/validation/test split")
        selected.append(normalized)
    return selected


def _atlas_path(atlas_derivative: Path, subject_id: str) -> Path:
    return (
        atlas_derivative
        / subject_id
        / "ses-1"
        / "anat"
        / f"{subject_id}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
    )


def _source_inventory(
    atlas_derivative: Path,
    folds: list[dict[str, Any]],
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_to_fold: dict[str, int] = {}
    universe: set[str] = set()
    for entry in folds:
        universe.update(entry["train_ids"])
        universe.update(entry["val_ids"])
        universe.update(entry["test_ids"])
        for subject_id in entry["test_ids"]:
            if subject_id in subject_to_fold:
                raise CorrectedExperimentError(f"Subject appears in more than one held-out fold: {subject_id}")
            subject_to_fold[subject_id] = int(entry["fold_index"])
    inventory: list[dict[str, Any]] = []
    challenge_cases: list[dict[str, Any]] = []
    for subject_id in sorted(universe):
        path = _atlas_path(atlas_derivative, subject_id)
        mask_path = path.with_name(
            path.name.replace("_T1w.nii.gz", "_label-L_desc-T1lesion_mask.nii.gz")
        )
        if not path.exists() or not mask_path.exists():
            raise CorrectedExperimentError(f"ATLAS source/mask required by immutable split is missing: {path}, {mask_path}")
        try:
            image = nb.load(str(path))
            shape = [int(value) for value in image.shape]
        except Exception as exc:
            raise CorrectedExperimentError(f"Cannot read ATLAS source header {path}: {exc}") from exc
        if len(shape) != 3:
            raise CorrectedExperimentError(f"Expected 3D ATLAS source {path}, got shape={shape}")
        try:
            relative_path = str(path.relative_to(data_root))
            mask_relative_path = str(mask_path.relative_to(data_root))
        except ValueError:
            relative_path = str(path.resolve())
            mask_relative_path = str(mask_path.resolve())
        source = {
            "subject_id": subject_id,
            "relative_path": relative_path,
            "sha256": sha256_file(path),
            "mask_relative_path": mask_relative_path,
            "mask_sha256": sha256_file(mask_path),
            "shape": shape,
        }
        inventory.append(source)
        if subject_id in subject_to_fold:
            challenge_cases.append(
                {
                    "subject_id": subject_id,
                    "fold": subject_to_fold[subject_id],
                    "relative_path": relative_path,
                    "source_sha256": source["sha256"],
                    "shape": shape,
                }
            )
    return inventory, challenge_cases


def _mrart_inventory(mrart_root: Path, data_root: Path) -> dict[str, Any]:
    acquisitions = ("standard", "headmotion1", "headmotion2")
    complete_subjects: list[dict[str, Any]] = []
    incomplete_subjects: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    if not mrart_root.exists():
        raise CorrectedExperimentError(f"MR-ART root required by the experiment definition is missing: {mrart_root}")
    for subject_dir in sorted(path for path in mrart_root.glob("sub-*") if path.is_dir()):
        subject_id = subject_dir.name
        acquisition_paths: dict[str, Path] = {}
        missing: list[str] = []
        for acquisition in acquisitions:
            path = subject_dir / "anat" / f"{subject_id}_acq-{acquisition}_T1w.nii.gz"
            if path.exists():
                acquisition_paths[acquisition] = path
            else:
                missing.append(acquisition)
        if missing:
            incomplete_subjects.append({"subject_id": subject_id, "missing_acquisitions": missing})
            continue
        complete_subjects.append({"subject_id": subject_id, "acquisitions": list(acquisitions)})
        for acquisition, path in acquisition_paths.items():
            image = nb.load(str(path))
            try:
                relative_path = str(path.relative_to(data_root))
            except ValueError:
                relative_path = str(path.resolve())
            sources.append(
                {
                    "subject_id": subject_id,
                    "acquisition": acquisition,
                    "relative_path": relative_path,
                    "sha256": sha256_file(path),
                    "shape": [int(value) for value in image.shape],
                    "zooms": [float(value) for value in image.header.get_zooms()[:3]],
                }
            )
    if not complete_subjects:
        raise CorrectedExperimentError(f"No complete MR-ART acquisition triplets found under {mrart_root}")
    return {
        "root_at_prepare": str(mrart_root.resolve()),
        "acquisitions": list(acquisitions),
        "complete_subjects": complete_subjects,
        "incomplete_subjects": incomplete_subjects,
        "sources": sources,
        "inventory_id": stable_hash(sources),
    }


def _duplicate_ids(ids: list[str], *, global_seed: int, fold: int, role: str, fraction: float) -> tuple[list[str], int]:
    seed = derive_seed(global_seed, "duplicated_subject_selection", fold=fold, sample_role=role)
    count = int(len(ids) * fraction)
    values = sorted(ids)
    selected = random.Random(seed).sample(values, count)
    return sorted(selected), seed


def _request_signature(
    *,
    experiment_id: str,
    split_sha256: str,
    normalization_mode: str,
    global_seed: int,
    models: list[str],
    regimes: list[str],
    folds: list[int],
    epochs: int,
    replicate_count: int,
    dataset_authority: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "split_sha256": split_sha256,
        "normalization_mode": normalization_mode,
        "global_seed": int(global_seed),
        "models": models,
        "training_regimes": regimes,
        "folds": folds,
        "max_epochs": int(epochs),
        "early_stopping_patience": ORIGINAL_EARLY_STOPPING_PATIENCE,
        "validation_interval": ORIGINAL_VALIDATION_INTERVAL,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "numerical_policy": NUMERICAL_POLICY,
        "corruption_replicate_count": int(replicate_count),
        "dataset_authority": dataset_authority,
    }


def validate_existing_request(existing: dict[str, Any], request: dict[str, Any]) -> None:
    actual = existing.get("prepare_request")
    if actual != request:
        raise ConflictError(
            "Experiment ID already exists with a conflicting immutable request. "
            f"existing={json.dumps(actual, sort_keys=True)} requested={json.dumps(request, sort_keys=True)}. "
            "Use a new experiment ID."
        )


def prepare_experiment(
    *,
    repo_root: Path,
    output_root: Path,
    experiment_id: str,
    data_root: Path,
    mrart_root: Path | None,
    split_path: Path,
    normalization_mode: str,
    global_seed: int = 9001,
    models: list[str] | tuple[str, ...] = DEFAULT_MODELS,
    regimes: list[str] | tuple[str, ...] = DEFAULT_REGIMES,
    folds: list[int] | tuple[int, ...] = DEFAULT_FOLDS,
    epochs: int = 45,
    replicate_count: int = 1,
    dataset_authority: str = "test",
    augment_fraction: float = 0.5,
) -> tuple[Path, dict[str, Any], str]:
    normalization_mode = validate_normalization_mode(normalization_mode)
    models_list = _normalize_list(list(models), DEFAULT_MODELS, "model")
    regimes_list = _normalize_list(list(regimes), DEFAULT_REGIMES, "training regime")
    folds_list = sorted(set(int(value) for value in folds))
    if any(value not in DEFAULT_FOLDS for value in folds_list) or not folds_list:
        raise CorrectedExperimentError(f"Folds must be a non-empty subset of {DEFAULT_FOLDS}")
    if epochs < 1 or replicate_count < 1:
        raise CorrectedExperimentError("Epochs and replicate count must be positive")
    if dataset_authority not in ("test", "authoritative"):
        raise CorrectedExperimentError("dataset_authority must be 'test' or 'authoritative'")
    if dataset_authority == "test" and not experiment_id.startswith("test-"):
        raise CorrectedExperimentError("Local/test experiment IDs must start with 'test-'")
    if dataset_authority == "authoritative" and experiment_id.startswith("test-"):
        raise CorrectedExperimentError("Authoritative experiment IDs must not start with 'test-'")
    if not 0 <= augment_fraction <= 1:
        raise CorrectedExperimentError("augment_fraction must be in [0, 1]")

    split_sha256 = sha256_file(split_path)
    request = _request_signature(
        experiment_id=experiment_id,
        split_sha256=split_sha256,
        normalization_mode=normalization_mode,
        global_seed=global_seed,
        models=models_list,
        regimes=regimes_list,
        folds=folds_list,
        epochs=epochs,
        replicate_count=replicate_count,
        dataset_authority=dataset_authority,
    )
    path = definition_path(output_root, experiment_id)
    if path.exists():
        existing = read_json(path)
        validate_existing_request(existing, request)
        referenced = existing.get("challenge", {})
        manifest_path = challenge_path(experiment_root(output_root, experiment_id), str(referenced.get("challenge_id")))
        if not manifest_path.exists() or sha256_file(manifest_path) != referenced.get("manifest_file_sha256"):
            raise ConflictError(f"Existing experiment has a missing or modified challenge manifest: {manifest_path}")
        return experiment_root(output_root, experiment_id), existing, "skipped-compatible"

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    fold_entries = _fold_entries(split_payload, folds_list)
    atlas_derivative = data_root / "train" / "derivatives" / "ATLAS"
    inventory, challenge_cases = _source_inventory(atlas_derivative, fold_entries, data_root)
    mrart_inventory = _mrart_inventory(mrart_root or (data_root / "mr-art"), data_root)
    fold_definition_id = stable_hash(fold_entries)

    definition_folds: list[dict[str, Any]] = []
    for entry in fold_entries:
        fold = int(entry["fold_index"])
        train_duplicates, train_seed = _duplicate_ids(
            entry["train_ids"], global_seed=global_seed, fold=fold, role="train", fraction=augment_fraction
        )
        definition_folds.append(
            {
                **entry,
                "duplicated_subjects": {"train_ids": train_duplicates, "val_ids": []},
                "derived_seeds": {
                    "train_duplicate_selection": train_seed,
                },
            }
        )

    try:
        import torchio as tio

        torchio_version = str(tio.__version__)
    except ImportError as exc:
        raise CorrectedExperimentError("Preparing a corrected challenge requires the pinned TorchIO dependency") from exc
    challenge = build_challenge_manifest(
        experiment_id=experiment_id,
        global_seed=global_seed,
        cases=challenge_cases,
        replicate_count=replicate_count,
        torchio_version=torchio_version,
    )
    root = experiment_root(output_root, experiment_id)
    manifest_path = challenge_path(root, challenge["challenge_id"])
    write_immutable_json(manifest_path, challenge)
    manifest_file_sha256 = sha256_file(manifest_path)

    model_configs: dict[str, Any] = {}
    for model in models_list:
        config = {
            **MODEL_CONFIGURATIONS[model],
            "batch_size_per_gpu": 1,
            "patches_per_volume": 8,
            "lesion_probability": 0.7,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "accumulation_steps": 1,
            "amp": True,
        }
        model_configs[model] = {**config, "model_configuration_id": stable_hash(config)}

    definition: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generation": "corrected_experiment_v1",
        "created_at": utc_now(),
        "dataset_authority": dataset_authority,
        "prepare_request": request,
        "configuration": {
            "models": models_list,
            "training_regimes": regimes_list,
            "folds": folds_list,
            "max_epochs": int(epochs),
            "early_stopping_patience": ORIGINAL_EARLY_STOPPING_PATIENCE,
            "validation_interval": ORIGINAL_VALIDATION_INTERVAL,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "numerical_policy": NUMERICAL_POLICY,
            "batch_size_per_gpu": 1,
            "nproc_per_node": 2,
            "augment_fraction": float(augment_fraction),
            "normalization_mode": normalization_mode,
            "preprocessing_version": PREPROCESSING_VERSION,
            "corruption_replicate_count": int(replicate_count),
        },
        "dataset": {
            "data_root_at_prepare": str(data_root.resolve()),
            "atlas_derivative_relative": "train/derivatives/ATLAS",
            "split_path_at_prepare": str(split_path.resolve()),
            "split_sha256": split_sha256,
            "fold_definition_id": fold_definition_id,
            "split_metadata": split_payload.get("metadata", {}),
            "source_inventory_id": stable_hash(inventory),
            "source_inventory": inventory,
            "mrart": mrart_inventory,
        },
        "folds": definition_folds,
        "seeds": {
            "global_seed": int(global_seed),
            "derivation_version": SEED_DERIVATION_VERSION,
            "independent_streams": [
                "fold_generation",
                "train_validation_split",
                "duplicated_subject_selection",
                "dataloader_order",
                "patch_center_sampling",
                "augmentation_application",
                "augmentation_parameters",
                "evaluation_challenge",
                "model_initialization",
            ],
            "patch_seed_identifiers": ["global_seed", "fold", "subject_id", "epoch", "patch_slot", "operation"],
        },
        "preprocessing": {
            "mode": normalization_mode,
            "version": PREPROCESSING_VERSION,
            "corrected_modes_normalization_count": 1,
            "support_rule": "observed_abs_gt_1e-6",
            "uses_paired_clean_support": False,
        },
        "challenge": {
            "name": challenge["distribution"]["name"],
            "challenge_id": challenge["challenge_id"],
            "replicate_ids": challenge["replicate_ids"],
            "manifest_relative_path": str(manifest_path.relative_to(root)),
            "manifest_file_sha256": manifest_file_sha256,
        },
        "model_configurations": model_configs,
        "environment_at_prepare": software_provenance(repo_root),
    }
    definition["definition_hash"] = stable_hash(definition)
    write_immutable_json(path, definition)
    return root, definition, "created"


def load_challenge(root: Path, definition: dict[str, Any]) -> dict[str, Any]:
    reference = definition["challenge"]
    path = root / reference["manifest_relative_path"]
    if not path.exists():
        raise CorrectedExperimentError(f"Immutable challenge manifest is missing: {path}")
    if sha256_file(path) != reference["manifest_file_sha256"]:
        raise ConflictError(f"Immutable challenge manifest checksum changed: {path}")
    manifest = read_json(path)
    if manifest.get("challenge_id") != reference["challenge_id"]:
        raise ConflictError(f"Challenge ID mismatch in {path}")
    return manifest


def fold_definition(definition: dict[str, Any], fold: int) -> dict[str, Any]:
    matches = [entry for entry in definition["folds"] if int(entry["fold_index"]) == int(fold)]
    if len(matches) != 1:
        raise CorrectedExperimentError(f"Fold {fold} is not defined by experiment {definition['experiment_id']}")
    return matches[0]
