#!/usr/bin/env python3
"""Generate large-font pooled k-fold Dice distribution plots for the paper figures."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

_LOCAL_CACHE_ROOT = Path(tempfile.gettempdir()) / "atlas_chart_cache"
_LOCAL_MPL_DIR = _LOCAL_CACHE_ROOT / "mplconfig"
_LOCAL_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPL_DIR))
_LOCAL_CACHE_DIR = _LOCAL_CACHE_ROOT / "xdg_cache"
_LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DICE_COLORS = ["#1f77b4", "#aec7e8", "#d62728", "#f5a4a4"]

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"

MODELS = [
    ("base_cnn", "3D U-Net"),
    ("mednext",  "MedNeXt"),
    ("uxnet",    "UXNet"),
    ("swin",     "Swin UNETR"),
]


def load_pooled_counts(csv_path: Path) -> tuple[list[str], list[int], list[int]]:
    labels: list[str] = []
    clean_counts: list[int] = []
    augmented_counts: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["bin_label"])
            clean_counts.append(int(row["clean_count"]))
            augmented_counts.append(int(row["augmented_count"]))

    if len(labels) != 10:
        raise ValueError(f"Expected 10 Dice bins in {csv_path}, found {len(labels)}")
    if sum(clean_counts) != 653 or sum(augmented_counts) != 653:
        raise ValueError(
            f"Expected pooled k-fold counts to sum to 653 in {csv_path}; "
            f"clean={sum(clean_counts)} augmented={sum(augmented_counts)}"
        )
    return labels, clean_counts, augmented_counts


def plot_model(model_dir: str, display_name: str) -> None:
    std_csv = RUNS_ROOT / model_dir / "run_kfold_summary" / "pooled_dice_bins_clean_vs_augmented.csv"
    aug_csv = RUNS_ROOT / model_dir / "run_DA_kfold_summary" / "pooled_dice_bins_clean_vs_augmented.csv"
    if not std_csv.exists():
        raise FileNotFoundError(f"Missing pooled standard-training Dice-bin CSV: {std_csv}")
    if not aug_csv.exists():
        raise FileNotFoundError(f"Missing pooled augmented-training Dice-bin CSV: {aug_csv}")

    bin_labels, std_clean_counts, std_artifact_counts = load_pooled_counts(std_csv)
    aug_bin_labels, aug_clean_counts, aug_artifact_counts = load_pooled_counts(aug_csv)
    if aug_bin_labels != bin_labels:
        raise ValueError(f"Dice-bin labels differ between {std_csv} and {aug_csv}")

    datasets = [
        ("Std. / Clean", std_clean_counts, DICE_COLORS[0]),
        ("Std. / Artifact", std_artifact_counts, DICE_COLORS[1]),
        ("Aug. / Clean", aug_clean_counts, DICE_COLORS[2]),
        ("Aug. / Artifact", aug_artifact_counts, DICE_COLORS[3]),
    ]

    n_groups = 10
    n_bars = len(datasets)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for j, (label, counts, color) in enumerate(datasets):
        offsets = x + j * bar_width
        bars = ax.bar(offsets, counts, bar_width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.2,
                    str(count),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    rotation=55,
                )

    ax.set_xticks(x + (n_bars - 1) * bar_width / 2)
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Dice Score Bin (%)", fontsize=15)
    ax.set_ylabel("Number of Subjects", fontsize=15)
    max_count = max(max(counts) for _, counts, _ in datasets)
    ax.set_ylim(0, max_count * 1.18)
    ax.set_title(f"{display_name}: Per-subject Dice Distribution", fontsize=18, fontweight="bold")
    ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=13,
        framealpha=0.85,
        edgecolor="gray",
        bbox_to_anchor=(0.44, 0.97),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path = RUNS_ROOT / f"combined_dice_{model_dir}.png"
    fig.savefig(out_path, dpi=200)
    model_out_path = RUNS_ROOT / model_dir / f"combined_dice_{model_dir}.png"
    shutil.copyfile(out_path, model_out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {model_out_path}")


def main():
    for model_dir, display_name in MODELS:
        plot_model(model_dir, display_name)


if __name__ == "__main__":
    main()
