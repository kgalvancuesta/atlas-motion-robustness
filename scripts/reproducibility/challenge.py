"""Immutable fixed corruption recipes and validated reusable caches."""
from __future__ import annotations

import os
import tempfile
import fcntl
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import (
    CHALLENGE_SCHEMA_VERSION,
    ConflictError,
    CorrectedExperimentError,
    PREPROCESSING_VERSION,
    array_sha256,
    read_json,
    stable_hash,
    write_json_atomic_replace,
)
from .rng import derive_seed

FIXED_LEGACY_DISTRIBUTION = {
    "name": "fixed_legacy_distribution",
    "version": 1,
    "description": "Fixed paired realization of the historical motion_consistent distribution.",
    "allows_noop": True,
    "transform_order": ["motion", "ghosting", "blur"],
    "transforms": {
        "motion": {
            "probability": 0.3,
            "torchio_random_class": "RandomMotion",
            "parameters": {"degrees": 10, "translation": 10, "num_transforms": 2, "image_interpolation": "linear"},
        },
        "ghosting": {
            "probability": 0.3,
            "torchio_random_class": "RandomGhosting",
            "parameters": {"num_ghosts": [4, 10], "axes": [0, 1, 2], "intensity": [0.5, 1.0], "restore": None},
        },
        "blur": {
            "probability": 0.3,
            "torchio_random_class": "RandomBlur",
            "parameters": {"std": [0.5, 2.0]},
        },
    },
}


def _require_torchio():
    try:
        import torch
        import torchio as tio
    except ImportError as exc:
        raise CorrectedExperimentError("TorchIO is required for challenge preparation/reconstruction") from exc
    return torch, tio


def _torch_seed_scope(seed: int):
    torch, _tio = _require_torchio()
    return torch.random.fork_rng(devices=[]), torch, int(seed)


def _bernoulli(seed: int, probability: float) -> bool:
    scope, torch, value = _torch_seed_scope(seed)
    with scope:
        torch.manual_seed(value)
        return bool(torch.rand(1).item() <= probability)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def sample_legacy_recipe(
    shape: tuple[int, int, int],
    *,
    global_seed: int,
    subject_id: str,
    fold: int,
    replicate_id: int,
    namespace: str = "evaluation_challenge",
    extra_identifiers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample concrete TorchIO parameters with independent application/parameter streams."""
    torch, tio = _require_torchio()
    extra = dict(extra_identifiers or {})
    applied: list[dict[str, Any]] = []
    seed_records: dict[str, dict[str, int]] = {}
    for name in FIXED_LEGACY_DISTRIBUTION["transform_order"]:
        definition = FIXED_LEGACY_DISTRIBUTION["transforms"][name]
        common = {"subject_id": subject_id, "fold": int(fold), "replicate_id": int(replicate_id), **extra}
        application_seed = derive_seed(global_seed, f"{namespace}.application.{name}", **common)
        parameter_seed = derive_seed(global_seed, f"{namespace}.parameters.{name}", **common)
        seed_records[name] = {"application_seed": application_seed, "parameter_seed": parameter_seed}
        if not _bernoulli(application_seed, float(definition["probability"])):
            continue

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(parameter_seed)
            if name == "motion":
                random_transform = tio.RandomMotion(**definition["parameters"])
                times, degrees, translation = random_transform.get_params(
                    random_transform.degrees_range,
                    random_transform.translation_range,
                    random_transform.num_transforms,
                    is_2d=any(int(size) == 1 for size in shape),
                )
                params = {
                    "times": _jsonable(times),
                    "degrees": _jsonable(degrees),
                    "translation": _jsonable(translation),
                    "image_interpolation": random_transform.image_interpolation,
                }
            elif name == "ghosting":
                random_transform = tio.RandomGhosting(**definition["parameters"])
                axes = [axis for axis in random_transform.axes if axis != 2] if any(s == 1 for s in shape) else random_transform.axes
                num_ghosts, axis, intensity, restore = random_transform.get_params(
                    tuple(int(x) for x in random_transform.num_ghosts_range),
                    axes,
                    random_transform.intensity_range,
                    random_transform.restore,
                )
                params = {
                    "num_ghosts": int(num_ghosts),
                    "axis": int(axis),
                    "intensity": float(intensity),
                    "restore": None if restore is None else float(restore),
                }
            elif name == "blur":
                random_transform = tio.RandomBlur(**definition["parameters"])
                params = {"std": _jsonable(random_transform.get_params(random_transform.std_ranges))}
            else:
                raise AssertionError(f"Unhandled transform: {name}")
        applied.append({"name": name, "concrete_class": name.title(), "parameters": params})

    recipe = {
        "recipe_version": 1,
        "subject_id": subject_id,
        "fold": int(fold),
        "replicate_id": int(replicate_id),
        "shape": [int(x) for x in shape],
        "noop": not applied,
        "applied_transforms": applied,
        "stable_seeds": seed_records,
    }
    recipe["recipe_id"] = stable_hash(recipe)
    return recipe


def apply_recipe(volume: np.ndarray, recipe: Mapping[str, Any]) -> np.ndarray:
    """Apply stored concrete parameters; no stochastic TorchIO class is used."""
    _torch, tio = _require_torchio()
    data = np.asarray(volume, dtype=np.float32)
    if list(data.shape) != list(recipe["shape"]):
        raise CorrectedExperimentError(f"Recipe shape {recipe['shape']} does not match input {list(data.shape)}")
    subject = tio.Subject(image=tio.ScalarImage(tensor=_torch.from_numpy(data[None, ...]).float()))
    for item in recipe["applied_transforms"]:
        params = item["parameters"]
        if item["name"] == "motion":
            transform = tio.Motion(
                degrees={"image": np.asarray(params["degrees"], dtype=np.float32)},
                translation={"image": np.asarray(params["translation"], dtype=np.float32)},
                times={"image": np.asarray(params["times"], dtype=np.float32)},
                image_interpolation={"image": str(params["image_interpolation"])},
            )
        elif item["name"] == "ghosting":
            transform = tio.Ghosting(
                num_ghosts={"image": int(params["num_ghosts"])},
                axis={"image": int(params["axis"])},
                intensity={"image": float(params["intensity"])},
                restore={"image": params["restore"]},
            )
        elif item["name"] == "blur":
            transform = tio.Blur(std={"image": tuple(float(x) for x in params["std"])})
        else:
            raise CorrectedExperimentError(f"Unsupported stored transform: {item['name']}")
        subject = transform(subject)
    result = subject.image.data.numpy()[0].astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise CorrectedExperimentError(f"Non-finite reconstructed corruption for recipe {recipe.get('recipe_id')}")
    return result


def build_challenge_manifest(
    *,
    experiment_id: str,
    global_seed: int,
    cases: list[dict[str, Any]],
    replicate_count: int,
    torchio_version: str,
) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: (int(item["fold"]), str(item["subject_id"]))):
        for replicate_id in range(replicate_count):
            recipe = sample_legacy_recipe(
                tuple(int(x) for x in case["shape"]),
                global_seed=global_seed,
                subject_id=str(case["subject_id"]),
                fold=int(case["fold"]),
                replicate_id=replicate_id,
            )
            recipe["source"] = {
                "relative_path": case["relative_path"],
                "sha256": case["source_sha256"],
                "shape": case["shape"],
            }
            recipes.append(recipe)
    definition = {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "distribution": FIXED_LEGACY_DISTRIBUTION,
        "global_seed": int(global_seed),
        "replicate_ids": list(range(replicate_count)),
        "torchio_definition_version": torchio_version,
        "recipes": recipes,
    }
    challenge_id = stable_hash(definition)
    return {**definition, "challenge_id": challenge_id}


def recipe_for(manifest: Mapping[str, Any], *, subject_id: str, fold: int, replicate_id: int) -> dict[str, Any]:
    matches = [
        recipe
        for recipe in manifest.get("recipes", [])
        if recipe.get("subject_id") == subject_id
        and int(recipe.get("fold", -1)) == int(fold)
        and int(recipe.get("replicate_id", -1)) == int(replicate_id)
    ]
    if len(matches) != 1:
        raise CorrectedExperimentError(
            f"Expected exactly one immutable challenge recipe for subject={subject_id}, fold={fold}, "
            f"replicate={replicate_id}; found {len(matches)}. Inference will not generate one automatically."
        )
    return dict(matches[0])


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


def cache_paths(
    cache_root: Path,
    *,
    challenge_id: str,
    normalization_mode: str,
    fold: int,
    replicate_id: int,
    subject_id: str,
) -> tuple[Path, Path]:
    base = (
        cache_root
        / "reconstructed_inputs"
        / challenge_id
        / PREPROCESSING_VERSION
        / normalization_mode
        / f"fold-{fold}"
        / f"replicate-{replicate_id}"
    )
    return base / f"{subject_id}.npy", base / f"{subject_id}.metadata.json"


def load_or_create_validated_cache(
    *,
    cache_root: Path,
    challenge_id: str,
    normalization_mode: str,
    fold: int,
    replicate_id: int,
    subject_id: str,
    source_sha256: str,
    recipe_id: str,
    builder,
) -> tuple[np.ndarray, str]:
    array_path, metadata_path = cache_paths(
        cache_root,
        challenge_id=challenge_id,
        normalization_mode=normalization_mode,
        fold=fold,
        replicate_id=replicate_id,
        subject_id=subject_id,
    )
    expected = {
        "challenge_id": challenge_id,
        "normalization_mode": normalization_mode,
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold": int(fold),
        "replicate_id": int(replicate_id),
        "subject_id": subject_id,
        "source_sha256": source_sha256,
        "recipe_id": recipe_id,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = metadata_path.with_suffix(metadata_path.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if array_path.exists() or metadata_path.exists():
            if not array_path.exists() or not metadata_path.exists():
                raise ConflictError(f"Incomplete memory-high cache entry: {array_path}")
            metadata = read_json(metadata_path)
            mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
            if mismatches:
                raise ConflictError(f"Incompatible memory-high cache metadata at {metadata_path}: {mismatches}")
            array = np.load(array_path, allow_pickle=False)
            actual_checksum = array_sha256(array)
            if actual_checksum != metadata.get("array_sha256"):
                raise ConflictError(f"Checksum validation failed for memory-high cache: {array_path}")
            return np.asarray(array, dtype=np.float32), "reused"

        array = np.asarray(builder(), dtype=np.float32)
        checksum = array_sha256(array)
        _atomic_save_npy(array_path, array)
        write_json_atomic_replace(
            metadata_path,
            {**expected, "array_sha256": checksum, "shape": list(array.shape), "dtype": str(array.dtype)},
        )
        return array, "created"
