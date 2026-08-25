#!/usr/bin/env python3
"""Plot training curves from train_log.csv files across multiple runs.

Usage:
    python scripts/visualization/plot_training.py \
        --run_dirs runs/base_cnn/run_kfold_01 runs/mednext/run_kfold_01 runs/uxnet/run_kfold_01 \
        --labels "Base CNN" "MedNeXt" "3D UX-Net" \
        --out training_curves.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(csv_path: Path) -> dict:
    epochs, losses, dices, val_losses = [], [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            d = float(row["dev_dice"])
            dices.append(d)
            vl = row.get("dev_loss")
            val_losses.append(float(vl) if vl else 1.0 - d)
    return {"epochs": epochs, "train_loss": losses, "dev_dice": dices, "dev_loss": val_losses}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.csv files.")
    parser.add_argument(
        "--run_dirs", nargs="+", required=True
    )
    parser.add_argument(
        "--labels", nargs="+", default=None
    )
    parser.add_argument("--out", default="training_curves.png", help="Output image path")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.run_dirs):
        parser.error("--labels must have the same count as --run_dirs")

    labels = args.labels or [
        Path(d).parent.name + "/" + Path(d).name for d in args.run_dirs
    ]

    # Colour cycle so each run gets a consistent colour across all three subplots.
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [c["color"] for c in prop_cycle]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    for i, (run_dir, label) in enumerate(zip(args.run_dirs, labels)):
        log_path = Path(run_dir) / "logs" / "train_log.csv"
        if not log_path.exists():
            print(f"Warning: {log_path} not found, skipping")
            continue
        data = load_log(log_path)
        colour = colours[i % len(colours)]
        ax1.plot(data["epochs"], data["train_loss"], label=label, marker=".", color=colour)
        ax2.plot(data["epochs"], data["dev_loss"], label=label, marker=".", color=colour)
        ax3.plot(data["epochs"], data["train_loss"], label=f"{label} train", marker=".", color=colour, linestyle="-")
        ax3.plot(data["epochs"], data["dev_loss"], label=f"{label} val", marker=".", color=colour, linestyle="--")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss per Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss per Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Train vs Val Loss (Overfitting View)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
