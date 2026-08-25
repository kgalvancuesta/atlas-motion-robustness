#!/usr/bin/env python3
"""Minimal 3D UNet training for ATLAS v2.0 (MNI space)."""
from __future__ import annotations

from models.base_cnn_model import build_base_model
from train_common import (
    build_argument_parser,
    extract_patch,
    list_labeled_samples,
    load_nifti,
    normalize_volume,
    run_training,
)


def build_model():
    return build_base_model()


def main() -> int:
    parser = build_argument_parser(
        description="Train a baseline 3D UNet on ATLAS v2.0.",
        default_loss="baseline",
    )
    args = parser.parse_args()
    return run_training(args, model_builder=build_model, model_name="base_cnn")


if __name__ == "__main__":
    raise SystemExit(main())
