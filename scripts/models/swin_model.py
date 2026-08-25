#!/usr/bin/env python3
"""Local SwinUNETR wrapper for the ATLAS training pipeline."""
from __future__ import annotations

from typing import Tuple

from monai.networks.nets import SwinUNETR


SWIN_UNETR_PATCH_SIZE = 2
SWIN_UNETR_INPUT_DIVISOR = SWIN_UNETR_PATCH_SIZE ** 5


def _require_swin_dependencies() -> None:
    try:
        import einops  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MONAI SwinUNETR requires the optional 'einops' package. "
            "Install it from requirements.txt before running train_swin.py."
        ) from exc


def validate_swin_patch_size(patch_size: Tuple[int, int, int]) -> None:
    if any(dim <= 0 for dim in patch_size):
        raise AssertionError(f"Patch size must be positive, got {patch_size}")
    if any(dim % SWIN_UNETR_INPUT_DIVISOR != 0 for dim in patch_size):
        raise AssertionError(
            "MONAI SwinUNETR checks that each spatial dimension is divisible by "
            f"{SWIN_UNETR_PATCH_SIZE}**5={SWIN_UNETR_INPUT_DIVISOR}, got {patch_size}"
        )


def build_swin_model():
    _require_swin_dependencies()
    # Keep the built-in MONAI SwinUNETR configuration close to its standard
    # reference setup while preserving the repo's one-channel binary interface.
    return SwinUNETR(
        in_channels=1,
        out_channels=1,
        patch_size=SWIN_UNETR_PATCH_SIZE,
        feature_size=24,
        norm_name="instance",
        spatial_dims=3,
    )
