"""Explicit legacy and corrected preprocessing modes."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .challenge import apply_recipe
from .core import CorrectedExperimentError, PREPROCESSING_VERSION, validate_normalization_mode

SUPPORT_RULE = "observed_abs_gt_1e-6"
SUPPORT_EPSILON = 1e-6
DEFAULT_DYNAMIC_RANGE_LIMIT = 1e6


def _validate_input(volume: np.ndarray, *, context: str) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise CorrectedExperimentError(f"Expected a 3D volume for {context}, got {array.shape}")
    if not np.isfinite(array).all():
        raise CorrectedExperimentError(f"Non-finite input values for {context}")
    return array


def normalize_legacy(volume: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = _validate_input(volume, context="legacy normalization")
    support = array > 0
    if not support.any():
        support = np.ones(array.shape, dtype=bool)
    mean = float(array[support].mean(dtype=np.float64))
    std = float(array[support].std(dtype=np.float64))
    if std == 0:
        std = 1.0
    result = ((array - mean) / std).astype(np.float32, copy=False)
    return result, {
        "support_rule": "raw_value_gt_zero",
        "support_voxels": int(support.sum()),
        "mean": mean,
        "std": std,
        "zero_background_value_after_normalization": float(-mean / std),
    }


def normalize_observed(volume: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = _validate_input(volume, context="observed-image normalization")
    support = np.abs(array) > SUPPORT_EPSILON
    if not support.any():
        raise CorrectedExperimentError("Observed-image support is empty; normalization is undefined")
    mean = float(array[support].mean(dtype=np.float64))
    std = float(array[support].std(dtype=np.float64))
    if not np.isfinite(std) or std <= 0:
        raise CorrectedExperimentError(f"Observed-image normalization has pathological std={std}")
    result = ((array - mean) / std).astype(np.float32, copy=False)
    return result, {
        "support_rule": SUPPORT_RULE,
        "support_epsilon": SUPPORT_EPSILON,
        "support_voxels": int(support.sum()),
        "mean": mean,
        "std": std,
        "zero_background_value_after_normalization": float(-mean / std),
        "uses_paired_clean_image": False,
    }


def _legacy_second_normalize(volume: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = _validate_input(volume, context="legacy post-artifact normalization")
    support = np.abs(array) > 1e-6
    if not support.any():
        return array, {"support_rule": "abs_gt_1e-6", "support_voxels": 0, "mean": None, "std": None}
    mean = float(array[support].mean(dtype=np.float64))
    std = float(array[support].std(dtype=np.float64))
    if std <= 0:
        return array, {"support_rule": "abs_gt_1e-6", "support_voxels": int(support.sum()), "mean": mean, "std": std}
    result = ((array - mean) / std).astype(np.float32, copy=False)
    return result, {"support_rule": "abs_gt_1e-6", "support_voxels": int(support.sum()), "mean": mean, "std": std}


def preprocess_volume(
    raw_volume: np.ndarray,
    *,
    mode: str,
    recipe: Mapping[str, Any] | None = None,
    dynamic_range_limit: float = DEFAULT_DYNAMIC_RANGE_LIMIT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess exactly once in corrected modes; safeguards only validate."""
    validate_normalization_mode(mode)
    raw = _validate_input(raw_volume, context=f"preprocessing mode {mode}")
    diagnostics: dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "normalization_mode": mode,
        "recipe_id": recipe.get("recipe_id") if recipe else None,
        "normalization_count": 0,
        "artifact_applied": recipe is not None,
    }

    if mode == "legacy":
        result, first = normalize_legacy(raw)
        diagnostics["normalization_count"] = 1
        diagnostics["first_normalization"] = first
        if recipe is not None:
            result = apply_recipe(result, recipe)
            result, second = _legacy_second_normalize(result)
            diagnostics["normalization_count"] = 2
            diagnostics["second_normalization"] = second
    elif mode == "artifact_then_normalize":
        observed = apply_recipe(raw, recipe) if recipe is not None else raw
        result, normalization = normalize_observed(observed)
        diagnostics["normalization_count"] = 1
        diagnostics["normalization"] = normalization
        diagnostics["operation_order"] = ["artifact", "normalize"] if recipe is not None else ["normalize"]
    else:
        result, normalization = normalize_observed(raw)
        diagnostics["normalization_count"] = 1
        diagnostics["normalization"] = normalization
        if recipe is not None:
            result = apply_recipe(result, recipe)
        diagnostics["operation_order"] = ["normalize", "artifact"] if recipe is not None else ["normalize"]

    if not np.isfinite(result).all():
        raise CorrectedExperimentError(f"Non-finite preprocessed values in mode {mode}; no silent fallback is permitted")
    abs_max = float(np.abs(result).max(initial=0.0))
    diagnostics["output_abs_max"] = abs_max
    diagnostics["output_min"] = float(result.min(initial=0.0))
    diagnostics["output_max"] = float(result.max(initial=0.0))
    diagnostics["finite"] = True
    diagnostics["dynamic_range_limit"] = float(dynamic_range_limit)
    if abs_max > dynamic_range_limit:
        raise CorrectedExperimentError(
            f"Pathological preprocessed dynamic range abs_max={abs_max} exceeds {dynamic_range_limit}; "
            "the corrected path will not silently renormalize it"
        )
    return np.asarray(result, dtype=np.float32), diagnostics
