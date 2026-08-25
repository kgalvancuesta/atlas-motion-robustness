#!/usr/bin/env python3
"""Local 3D UX-Net wrapper for the ATLAS training pipeline."""
from __future__ import annotations

from typing import Tuple


def validate_uxnet_patch_size(patch_size: Tuple[int, int, int]) -> None:
    if any(dim <= 0 for dim in patch_size):
        raise AssertionError(f"Patch size must be positive, got {patch_size}")
    if any(dim % 16 != 0 for dim in patch_size):
        raise AssertionError(
            f"3D UX-Net expects patch dimensions divisible by 16, got {patch_size}"
        )


def build_uxnet_model():
    from .uxnet_3d.network_backbone import UXNET

    return UXNET(
        in_chans=1,
        out_chans=1,  # binary segmentation, sigmoid activation pipeline
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        spatial_dims=3,
    )
