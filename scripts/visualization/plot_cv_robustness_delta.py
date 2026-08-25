#!/usr/bin/env python3
"""Plot fold-averaged k-fold Dice drop by training condition."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "atlas_chart_cache"
_MPL_DIR = _CACHE_ROOT / "mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
_XDG_CACHE_DIR = _CACHE_ROOT / "xdg_cache"
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    ("base_cnn", "3D U-Net"),
    ("mednext", "MedNeXt"),
    ("uxnet", "UXNet"),
    ("swin", "Swin UNETR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot k-fold clean-to-artifact Dice drop.")
    parser.add_argument(
        "--summary_json",
        default="runs/cv_aggregate_summary/all_model_cv_metrics_summary.json",
        help="All-model CV metrics summary JSON.",
    )
    parser.add_argument(
        "--out",
        default="runs/cv_aggregate_summary/all_model_robustness_delta.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_delta(rows: list[dict], model_dir: str, condition: str) -> float:
    for row in rows:
        if row.get("model_dir") == model_dir and row.get("condition") == condition:
            return float(row["robustness_delta_fold_mean"])
    raise KeyError(f"Missing robustness delta for {model_dir} / {condition}")


def load_delta_std(rows: list[dict], model_dir: str, condition: str) -> float:
    for row in rows:
        if row.get("model_dir") == model_dir and row.get("condition") == condition:
            return float(row["robustness_delta_fold_std"])
    raise KeyError(f"Missing robustness delta std for {model_dir} / {condition}")


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_json)
    out_path = Path(args.out)
    payload = json.loads(summary_path.read_text())
    rows = payload["rows"]

    labels = [label for _, label in MODEL_ORDER]
    std_values = [load_delta(rows, model_dir, "No DA") for model_dir, _ in MODEL_ORDER]
    aug_values = [load_delta(rows, model_dir, "DA") for model_dir, _ in MODEL_ORDER]
    std_errors = [load_delta_std(rows, model_dir, "No DA") for model_dir, _ in MODEL_ORDER]
    aug_errors = [load_delta_std(rows, model_dir, "DA") for model_dir, _ in MODEL_ORDER]

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    std_bars = ax.bar(
        x - width / 2,
        std_values,
        width,
        yerr=std_errors,
        capsize=4,
        error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2},
        label="Std. Training",
        color="#1f77b4",
    )
    aug_bars = ax.bar(
        x + width / 2,
        aug_values,
        width,
        yerr=aug_errors,
        capsize=4,
        error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2},
        label="Aug. Training",
        color="#ff7f0e",
    )

    ax.set_title("5-Fold CV Dice Drop by Training Condition")
    ax.set_ylabel("Dice Drop (Clean - Artifact)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.07)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    for bars in (std_bars, aug_bars):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.0012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
