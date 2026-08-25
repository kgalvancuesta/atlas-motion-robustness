#!/usr/bin/env python3
"""Grouped bar chart overlaying Dice distributions from multiple evaluations.

Usage:
    python scripts/visualization/plot_dice_comparison.py --run-dir runs/uxnet/run_DA_kfold_01
    python scripts/visualization/plot_dice_comparison.py --eval-files path/a.json path/b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_eval_dice_scores(eval_json_path: Path) -> list[float]:
    with eval_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    scores = []
    for item in payload.get("per_subject", []):
        dice_val = _safe_float(item.get("dice"))
        if dice_val is None:
            continue
        scores.append(max(0.0, min(1.0, float(dice_val))))
    if not scores:
        raise ValueError(f"No per-subject Dice scores found in {eval_json_path}")
    return scores


def build_dice_bins(scores) -> tuple[list[str], list[int]]:
    labels = [f"{i * 10}-{i * 10 + 9}" for i in range(9)] + ["90-100"]
    counts = [0] * 10
    for score in scores:
        pct = score * 100.0
        idx = 9 if pct >= 100.0 else max(0, min(9, int(pct // 10)))
        counts[idx] += 1
    return labels, counts


def extract_tag(p: Path, use_parent: bool = False) -> str:
    tag = p.stem.replace("test_", "").replace("heldout_", "").replace("_metrics", "")
    return f"{p.parent.parent.name}/{tag}" if use_parent else tag


def plot_grouped_bars(
    datasets: list[tuple[str, list[str], list[int]]],
    output_path: Path,
) -> None:
    n_groups = 10
    n_bars = len(datasets)
    bar_width = 0.8 / n_bars
    x = list(range(n_groups))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for j, (tag, _labels, counts) in enumerate(datasets):
        offsets = [xi + j * bar_width for xi in x]
        color = COLORS[j % len(COLORS)]
        bars = ax.bar(offsets, counts, bar_width, label=tag, color=color)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.25,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    bin_labels = datasets[0][1]
    ax.set_xticks([xi + (n_bars - 1) * bar_width / 2 for xi in x])
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Dice Score Bin (%)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Dice Score Distribution Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--eval-files", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.run_dir:
        eval_files = sorted((args.run_dir / "eval_local").glob("test_*_metrics.json"))
        if not eval_files:
            eval_files = sorted((args.run_dir / "eval_local").glob("heldout_*_metrics.json"))
        use_parent = False
        default_out = args.run_dir / "eval_local" / "dice_comparison.png"
    elif args.eval_files:
        eval_files = args.eval_files
        use_parent = True
        default_out = eval_files[0].parent / "dice_comparison.png"
    else:
        print("Error: provide --run-dir or --eval-files", file=sys.stderr)
        return 1

    if not eval_files:
        print("Error: no test_*_metrics.json or heldout_*_metrics.json files found", file=sys.stderr)
        return 1

    datasets = []
    for ef in eval_files:
        scores = load_eval_dice_scores(ef)
        labels, counts = build_dice_bins(scores)
        datasets.append((extract_tag(ef, use_parent=use_parent), labels, counts))

    output_path = args.output or default_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_grouped_bars(datasets, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
