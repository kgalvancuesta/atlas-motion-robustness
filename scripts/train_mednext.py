#!/usr/bin/env python3
"""Train MedNeXt on ATLAS using the existing local training pipeline."""
from __future__ import annotations
from models.mednext_model import build_mednext_model, validate_mednext_patch_size
from train_common import build_argument_parser, run_training


def build_model():
    return build_mednext_model()


def main() -> int:
    parser = build_argument_parser(
        description="Train a MedNeXt model on ATLAS v2.0 using the local training pipeline.",
        default_loss="mednext",
    )
    args = parser.parse_args()
    return run_training(
        args,
        model_builder=build_model,
        model_name="mednext",
        patch_size_validator=validate_mednext_patch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
