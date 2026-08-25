#!/usr/bin/env python3
"""Generate training and local-eval charts for a single model run.

Outputs per run directory:
- logs/train_loss_curve.png
- logs/dev_dice_curve.png
- logs/training_loss_and_val_dice.png
- eval_local/dice_bins_10pct_<tag>.csv   (one per test_*_metrics.json / heldout_*_metrics.json)
- eval_local/dice_bins_10pct_<tag>.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

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


def _safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_train_log(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_loss: list[float] = []
    dev_dice: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_raw = row.get("epoch")
            loss_raw = row.get("train_loss")
            dice_raw = row.get("dev_dice")

            epoch_val = _safe_float(epoch_raw)
            loss_val = _safe_float(loss_raw)
            dice_val = _safe_float(dice_raw)
            if epoch_val is None or loss_val is None or dice_val is None:
                continue

            epochs.append(int(epoch_val))
            train_loss.append(float(loss_val))
            dev_dice.append(float(dice_val))

    if not epochs:
        raise ValueError(f"No valid rows found in {csv_path}")

    return epochs, train_loss, dev_dice


def _save_line_plot(
    x: Iterable[int],
    y: Iterable[float],
    *,
    title: str,
    y_label: str,
    output_path: Path,
    color: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(list(x), list(y), marker="o", linewidth=1.8, markersize=4, color=color)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_training_plots(
    logs_dir: Path,
    run_label: str,
    epochs: list[int],
    train_loss: list[float],
    dev_dice: list[float],
) -> list[Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)

    loss_path = logs_dir / "train_loss_curve.png"
    _save_line_plot(
        epochs,
        train_loss,
        title=f"{run_label}: Train Loss vs Epoch",
        y_label="Train Loss",
        output_path=loss_path,
        color="#1f77b4",
    )

    dice_path = logs_dir / "dev_dice_curve.png"
    _save_line_plot(
        epochs,
        dev_dice,
        title=f"{run_label}: Validation Dice vs Epoch",
        y_label="Validation Dice",
        output_path=dice_path,
        color="#2ca02c",
    )

    combined_path = logs_dir / "training_loss_and_val_dice.png"
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(epochs, train_loss, color="#1f77b4", marker="o", linewidth=1.8, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(epochs, dev_dice, color="#2ca02c", marker="s", linewidth=1.8, markersize=4)
    ax2.set_ylabel("Validation Dice", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    plt.title(f"{run_label}: Training Loss and Validation Dice")
    fig.tight_layout()
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    return [loss_path, dice_path, combined_path]


def load_eval_dice_scores(eval_json_path: Path) -> list[float]:
    with eval_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    per_subject = payload.get("per_subject", [])
    scores: list[float] = []
    for item in per_subject:
        dice_val = _safe_float(item.get("dice"))
        if dice_val is None:
            continue
        scores.append(max(0.0, min(1.0, float(dice_val))))

    if not scores:
        raise ValueError(f"No per-subject Dice scores found in {eval_json_path}")

    return scores


def build_dice_bins(scores: Iterable[float]) -> tuple[list[str], list[int]]:
    labels = [f"{i * 10}-{i * 10 + 9}" for i in range(9)] + ["90-100"]
    counts = [0] * 10

    for score in scores:
        pct = score * 100.0
        if pct >= 100.0:
            idx = 9
        else:
            idx = int(pct // 10)
            idx = max(0, min(9, idx))
        counts[idx] += 1

    return labels, counts


def save_dice_bins_outputs(
    eval_dir: Path,
    run_label: str,
    labels: list[str],
    counts: list[int],
    suffix: str = "",
) -> list[Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"dice_bins_10pct{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_label", "lower_percent", "upper_percent", "count"])
        for i, (label, count) in enumerate(zip(labels, counts)):
            lower = i * 10
            upper = 100 if i == 9 else (i * 10 + 9)
            writer.writerow([label, lower, upper, count])

    png_path = eval_dir / f"dice_bins_10pct{suffix}.png"
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, counts, color="#ff7f0e")
    plt.title(f"{run_label}: Dice Distribution by 10% Bin")
    plt.xlabel("Dice Score Bin (%)")
    plt.ylabel("Number of Patients")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.25,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    return [csv_path, png_path]


def process_run(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "eval_local"

    train_log_path = logs_dir / "train_log.csv"
    if not train_log_path.exists():
        raise FileNotFoundError(f"Missing training log: {train_log_path}")

    eval_jsons = sorted(eval_dir.glob("test_*_metrics.json"))
    if not eval_jsons:
        eval_jsons = sorted(eval_dir.glob("heldout_*_metrics.json"))
    if not eval_jsons:
        for legacy_name in ("test_quick_metrics.json", "heldout_quick_metrics.json"):
            legacy = eval_dir / legacy_name
            if legacy.exists():
                eval_jsons = [legacy]
                break
    if not eval_jsons:
        raise FileNotFoundError(
            f"No test_*_metrics.json or heldout_*_metrics.json files found in {eval_dir}"
        )

    run_label = f"{run_dir.parent.name}/{run_dir.name}"

    epochs, train_loss, dev_dice = load_train_log(train_log_path)
    training_outputs = save_training_plots(logs_dir, run_label, epochs, train_loss, dev_dice)

    all_eval_outputs: list[Path] = []
    for eval_json_path in eval_jsons:
        tag = eval_json_path.stem.replace("test_", "").replace("heldout_", "").replace("_metrics", "")
        dice_scores = load_eval_dice_scores(eval_json_path)
        labels, counts = build_dice_bins(dice_scores)
        eval_outputs = save_dice_bins_outputs(
            eval_dir, f"{run_label} ({tag})", labels, counts, suffix=f"_{tag}",
        )
        all_eval_outputs.extend(eval_outputs)

    return training_outputs + all_eval_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training and local-eval charts for a model run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing logs/ and eval_local/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = process_run(args.run_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
