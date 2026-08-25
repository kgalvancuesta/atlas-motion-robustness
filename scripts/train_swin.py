#!/usr/bin/env python3
"""Train MONAI SwinUNETR on ATLAS using the existing local training pipeline."""
from __future__ import annotations

from models.swin_model import build_swin_model, validate_swin_patch_size
from train_common import build_argument_parser, run_training


def build_model():
    return build_swin_model()


def main() -> int:
    parser = build_argument_parser(
        description="Train a MONAI SwinUNETR model on ATLAS v2.0 using the local training pipeline.",
        default_loss="dicece",
    )
    args = parser.parse_args()
    return run_training(
        args,
        model_builder=build_model,
        model_name="swin",
        patch_size_validator=validate_swin_patch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
