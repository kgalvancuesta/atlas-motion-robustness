#!/usr/bin/env python3
"""Local MedNeXt wrapper for the ATLAS training pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


def _ensure_mednext_on_path() -> None:
    mednext_root = Path(__file__).resolve().parent / "mednext"
    if not mednext_root.exists():
        raise FileNotFoundError(f"MedNeXt repo not found at {mednext_root}")
    mednext_path = str(mednext_root)
    if mednext_path not in sys.path:
        sys.path.insert(0, mednext_path)


def validate_mednext_patch_size(patch_size: Tuple[int, int, int]) -> None:
    if any(dim <= 0 for dim in patch_size):
        raise AssertionError(f"Patch size must be positive, got {patch_size}")
    if any(dim % 32 != 0 for dim in patch_size):
        raise AssertionError(
            f"MedNeXt expects patch dimensions divisible by 32 for this integration, got {patch_size}"
        )


def build_mednext_model():
    _ensure_mednext_on_path()
    from nnunet_mednext import create_mednext_v1

    # Small kernel-3 MedNeXt is the lightest upstream preset and keeps the
    # baseline batch/patch CLI more likely to fit on both CUDA and MPS devices.
    return create_mednext_v1(
        num_input_channels=1,
        num_classes=1,
        model_id="S",
        kernel_size=3,
        deep_supervision=False,
    )
