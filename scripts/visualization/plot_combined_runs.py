#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
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

TRAIN_COLORS = {
    "base_loss": "#1f77b4",
    "base_dice": "#2ca02c",
    "da_loss":   "#ff7f0e",
    "da_dice":   "#9467bd",
}


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def load_dice_scores(json_path: Path) -> list[float]:
    payload = json.loads(json_path.read_text())
    scores = []
    for item in payload.get("per_subject", []):
        v = _safe_float(item.get("dice"))
        if v is not None:
            scores.append(max(0.0, min(1.0, v)))
    if not scores:
        raise ValueError(f"No per-subject dice scores in {json_path}")
    return scores


def build_bins(scores: list[float]) -> tuple[list[str], list[int]]:
    labels = [f"{i*10}–{i*10+9}" for i in range(9)] + ["90–100"]
    counts = [0] * 10
    for s in scores:
        pct = s * 100.0
        idx = 9 if pct >= 100.0 else max(0, min(9, int(pct // 10)))
        counts[idx] += 1
    return labels, counts


def plot_combined_dice(
    base_dir: Path,
    da_dir: Path,
    out_path: Path,
    model_name: str,
) -> None:
    datasets: list[tuple[str, list[str], list[int]]] = []
    colors: list[str] = []

    pairs = [
        (base_dir, "clean",     DICE_COLORS[0]),
        (base_dir, "augmented", DICE_COLORS[1]),
        (da_dir,   "clean",     DICE_COLORS[2]),
        (da_dir,   "augmented", DICE_COLORS[3]),
    ]

    run_base_name = base_dir.name
    run_da_name   = da_dir.name

    label_map = {
        (run_base_name, "clean"):     f"{run_base_name} – clean",
        (run_base_name, "augmented"): f"{run_base_name} – augmented",
        (run_da_name,   "clean"):     f"{run_da_name} – clean",
        (run_da_name,   "augmented"): f"{run_da_name} – augmented",
    }

    for run_dir, tag, color in pairs:
        json_path = run_dir / "eval_local" / f"heldout_{tag}_metrics.json"
        if not json_path.exists():
            print(f"  [warn] missing {json_path}, skipping")
            continue
        scores = load_dice_scores(json_path)
        labels, counts = build_bins(scores)
        run_key = (run_dir.name, tag)
        datasets.append((label_map[run_key], labels, counts))
        colors.append(color)

    if not datasets:
        raise RuntimeError("No eval JSON files found")

    n_groups = 10
    n_bars = len(datasets)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(13, 6))
    for j, (tag, _labels, counts) in enumerate(datasets):
        offsets = x + j * bar_width
        bars = ax.bar(offsets, counts, bar_width, label=tag, color=colors[j],
                      edgecolor="white", linewidth=0.4)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.15,
                    str(count),
                    ha="center", va="bottom", fontsize=7,
                )

    bin_labels = datasets[0][1]
    ax.set_xticks(x + (n_bars - 1) * bar_width / 2)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Dice Score Bin (%)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title(f"{model_name}: Dice Distribution — {run_base_name} vs {run_da_name}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")



def load_train_log(csv_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    epochs, losses, dices, val_losses = [], [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e = _safe_float(row.get("epoch"))
            l = _safe_float(row.get("train_loss"))
            d = _safe_float(row.get("dev_dice"))
            if None in (e, l, d):
                continue
            epochs.append(int(e))
            losses.append(l)
            dices.append(d)
            vl = _safe_float(row.get("dev_loss"))
            val_losses.append(vl if vl is not None else 1.0 - d)
    if not epochs:
        raise ValueError(f"No valid rows in {csv_path}")
    return epochs, losses, dices, val_losses


def plot_combined_training(
    base_dir: Path,
    da_dir: Path,
    out_path: Path,
    model_name: str,
) -> None:
    run_base_name = base_dir.name
    run_da_name   = da_dir.name

    base_epochs, base_loss, base_dice, base_val_loss = load_train_log(base_dir / "logs" / "train_log.csv")
    da_epochs,   da_loss,   da_dice,  da_val_loss   = load_train_log(da_dir   / "logs" / "train_log.csv")

    fig, ax = plt.subplots(figsize=(11, 6))

    # Solid lines = train loss; dashed = val dice loss (1 − Dice).
    ax.plot(base_epochs, base_loss,
            color=TRAIN_COLORS["base_loss"], marker="o", linewidth=1.8, markersize=3,
            linestyle="-", label=f"{run_base_name} – train loss")
    ax.plot(base_epochs, base_val_loss,
            color=TRAIN_COLORS["base_loss"], marker="s", linewidth=1.8, markersize=3,
            linestyle="--", alpha=0.7, label=f"{run_base_name} – val loss")

    ax.plot(da_epochs, da_loss,
            color=TRAIN_COLORS["da_loss"], marker="o", linewidth=1.8, markersize=3,
            linestyle="-", label=f"{run_da_name} – train loss")
    ax.plot(da_epochs, da_val_loss,
            color=TRAIN_COLORS["da_loss"], marker="s", linewidth=1.8, markersize=3,
            linestyle="--", alpha=0.7, label=f"{run_da_name} – val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    ax.set_title(f"Train vs Val Loss for {model_name}, {run_base_name} vs {run_da_name}")
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined comparison plots for two runs.")
    p.add_argument("--run-base",   type=Path, required=True, help="Baseline run directory")
    p.add_argument("--run-da",     type=Path, required=True, help="DA run directory")
    p.add_argument("--out-dir",    type=Path, required=True, help="Directory to save combined plots")
    p.add_argument("--model-name", type=str, default="Model",  help="Display name for chart titles")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_combined_dice(
        args.run_base, args.run_da,
        args.out_dir / "combined_dice_comparison.png",
        args.model_name,
    )
    plot_combined_training(
        args.run_base, args.run_da,
        args.out_dir / "combined_training_curves.png",
        args.model_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
