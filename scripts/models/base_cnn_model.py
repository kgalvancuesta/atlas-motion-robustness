#!/usr/bin/env python3
"""MONAI 3D U-Net baseline configuration for the ATLAS pipeline."""
from __future__ import annotations

import torch
from monai.networks.nets import UNet


def build_base_model() -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    )
