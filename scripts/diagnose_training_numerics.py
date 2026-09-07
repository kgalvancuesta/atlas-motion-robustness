#!/usr/bin/env python3
"""Replay one corrected Base CNN training step from its operational checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.base_cnn_model import build_base_model
from train_common import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay all optimizer updates after a corrected epoch-boundary checkpoint, then compare "
            "AMP and FP32 forwards at one target step without updating that step or overwriting the checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir", required=True, help="Existing corrected Base CNN task directory"
    )
    parser.add_argument(
        "--epoch",
        required=True,
        type=int,
        help="Target epoch; must immediately follow last.pt",
    )
    parser.add_argument(
        "--step", required=True, type=int, help="One-based per-rank target step"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New JSON path; existing files are never overwritten",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of largest state/activation values to retain",
    )
    parser.add_argument(
        "--verify-fix",
        action="store_true",
        help="Require natural AMP failure at the target, a successful production FP32 update, and one subsequent update. "
        "Exit 0 only on a reproduced and recovered failure; 2 means inconclusive. Sources are read-only.",
    )
    return parser


def main() -> int:
    cli_args = build_parser().parse_args()
    run_dir = Path(cli_args.run_dir)
    config_path = run_dir / "config" / "train_config.json"
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if not config_path.is_file():
        raise SystemExit(
            f"ERROR: corrected training config does not exist: {config_path}"
        )
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"ERROR: operational checkpoint does not exist: {checkpoint_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_name") != "base_cnn":
        raise SystemExit(f"ERROR: expected model_name=base_cnn in {config_path}")
    if "corrected_experiment_metadata" not in config:
        raise SystemExit(f"ERROR: {config_path} is not a corrected-experiment config")
    recorded_args = config.get("args")
    if not isinstance(recorded_args, dict):
        raise SystemExit(f"ERROR: missing recorded training arguments in {config_path}")

    training_args = argparse.Namespace(**recorded_args)
    training_args.run_dir = str(run_dir)
    training_args.corrected_resume = True
    training_args.diagnostic_replay_epoch = int(cli_args.epoch)
    training_args.diagnostic_replay_step = int(cli_args.step)
    training_args.diagnostic_output = str(Path(cli_args.output))
    training_args.numerics_debug_topk = max(1, int(cli_args.top_k))
    training_args.diagnostic_verify_fix = cli_args.verify_fix
    return run_training(
        training_args, model_builder=build_base_model, model_name="base_cnn"
    )


if __name__ == "__main__":
    raise SystemExit(main())
