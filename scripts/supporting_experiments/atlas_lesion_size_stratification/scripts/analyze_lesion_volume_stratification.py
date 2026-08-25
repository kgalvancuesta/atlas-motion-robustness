#!/usr/bin/env python3
"""
Analyze ATLAS lesion-volume distribution and recommend a carry-forward
stratification scheme for later 5-fold cross-validation.

Command:
    python scripts/supporting_experiments/atlas_lesion_size_stratification/scripts/analyze_lesion_volume_stratification.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np

VERY_SMALL_VOXEL_THRESHOLD = 1000
PERCENTILES = [1, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 99]
QUANTILE_KS = [3, 4, 5, 6, 8, 10]
LOG_KS = [4, 5, 6]
JENKS_KS = [4, 5, 6]
PREFERRED_MIN_PER_BIN = 25
HARD_MIN_PER_BIN = 5


@dataclass
class SubjectRecord:
    subject_id: str
    split_source: str
    split_key_used: str
    mask_path: str
    lesion_voxels: int | None
    lesion_mm3: float | None
    voxel_volume_mm3: float | None
    spacing_mm: str | None
    used_fallback_mask_path: bool
    status: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze lesion-volume distribution for ATLAS lesion-volume stratification."
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Dataset root containing train/ and test/. Defaults to $ATLAS_DATA_ROOT or data/.",
    )
    parser.add_argument(
        "--gt_deriv",
        default=None,
        help="GT derivative root. Defaults to <data_root>/train/derivatives/ATLAS.",
    )
    parser.add_argument(
        "--splits_json",
        default="splits/atlas_5fold_lesion_quartile_excluding_known_issues.json",
        help="Split JSON containing the labeled subject IDs; defaults to the paper CV master.",
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "outputs/supporting_experiments/"
            "atlas_lesion_size_stratification/stratification_analysis"
        ),
        help="Directory for local CSV/JSON/Markdown working outputs.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Binary lesion threshold. Matches train_common.py default.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    gt_deriv = Path(args.gt_deriv) if args.gt_deriv else (data_root / "train" / "derivatives" / "ATLAS")
    splits_json = Path(args.splits_json)
    out_dir = Path(args.out_dir)
    if not splits_json.is_absolute():
        splits_json = (repo_root / splits_json).resolve()
    if not gt_deriv.is_absolute():
        gt_deriv = (repo_root / gt_deriv).resolve()
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    return repo_root, gt_deriv, splits_json, out_dir


def normalize_subject_id(subject_id: str) -> str:
    return subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"


def _present_list(data: dict[str, Any], key: str) -> list[str] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit(f"ERROR: split key '{key}' is present but is not a list in the split JSON.")
    return [normalize_subject_id(str(item)) for item in value]


def load_subject_entries(splits_json: Path) -> tuple[list[tuple[str, str, str]], list[str], dict[str, int]]:
    payload = json.loads(splits_json.read_text())
    if "folds" in payload:
        folds = payload["folds"]
        if not isinstance(folds, list) or not folds:
            raise SystemExit("ERROR: CV split JSON has no usable folds.")

        warnings: list[str] = []
        subject_ids: set[str] = set()
        fold_universes: list[set[str]] = []
        for fold_position, fold in enumerate(folds):
            if not isinstance(fold, dict):
                raise SystemExit(f"ERROR: CV fold at position {fold_position} is not an object.")
            fold_index = fold.get("fold_index", fold_position)
            fold_subjects: list[str] = []
            for key in ("train_ids", "val_ids", "dev_ids", "test_ids", "heldout_ids"):
                ids = _present_list(fold, key)
                if ids is not None:
                    fold_subjects.extend(ids)
            if not fold_subjects:
                raise SystemExit(f"ERROR: CV fold {fold_index} has no subject IDs.")
            if len(fold_subjects) != len(set(fold_subjects)):
                warnings.append(f"Duplicate subject membership within CV fold {fold_index}.")
            fold_universe = set(fold_subjects)
            fold_universes.append(fold_universe)
            subject_ids.update(fold_universe)

        reference_universe = fold_universes[0]
        if any(universe != reference_universe for universe in fold_universes[1:]):
            warnings.append("CV folds do not all cover the same subject universe; using their union.")

        entries = [(subject_id, "cv", "folds") for subject_id in sorted(subject_ids)]
        return entries, warnings, {"cv": len(entries)}

    alias_groups = [
        ("train", ["train_ids"]),
        ("val", ["dev_ids", "val_ids"]),
        ("test", ["heldout_ids", "test_ids"]),
    ]
    entries: list[tuple[str, str, str]] = []
    warnings: list[str] = []
    totals_by_split: dict[str, int] = {}
    seen_within_group: dict[str, set[str]] = {}
    seen_global: dict[str, tuple[str, str]] = {}

    for canonical_split, keys in alias_groups:
        group_ids: list[tuple[str, str]] = []
        for key in keys:
            ids = _present_list(payload, key)
            if ids is None:
                continue
            group_ids.extend((subject_id, key) for subject_id in ids)
        if not group_ids:
            continue
        group_seen = seen_within_group.setdefault(canonical_split, set())
        totals_by_split[canonical_split] = 0
        for subject_id, key in group_ids:
            if subject_id in group_seen:
                warnings.append(
                    f"Duplicate subject within {canonical_split} aliases: {subject_id} appears multiple times "
                    f"across {keys}."
                )
                continue
            group_seen.add(subject_id)
            totals_by_split[canonical_split] += 1
            if subject_id in seen_global:
                prev_split, prev_key = seen_global[subject_id]
                warnings.append(
                    f"Overlapping split membership: {subject_id} appears in both {prev_key} ({prev_split}) and "
                    f"{key} ({canonical_split}). Keeping first occurrence."
                )
                continue
            seen_global[subject_id] = (canonical_split, key)
            entries.append((subject_id, canonical_split, key))

    if not entries:
        raise SystemExit(
            "ERROR: no subject IDs found in the split JSON. Expected keys like "
            "'train_ids', 'dev_ids'/'val_ids', and 'heldout_ids'/'test_ids'."
        )

    return entries, warnings, totals_by_split


def expected_mask_path(gt_deriv: Path, subject_id: str) -> Path:
    return (
        gt_deriv
        / subject_id
        / "ses-1"
        / "anat"
        / f"{subject_id}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
    )


def resolve_mask_path(gt_deriv: Path, subject_id: str) -> tuple[Path | None, bool, str | None]:
    direct = expected_mask_path(gt_deriv, subject_id)
    if direct.exists():
        return direct, False, None

    subject_root = gt_deriv / subject_id
    if not subject_root.exists():
        return None, False, f"Subject directory not found under GT derivative root: {subject_root}"

    candidates = sorted(subject_root.rglob("*label-L_desc-T1lesion_mask.nii*"))
    if len(candidates) == 1:
        return candidates[0], True, None
    if len(candidates) > 1:
        return None, True, f"Multiple candidate lesion masks found for {subject_id}: {candidates}"

    fallback_candidates = sorted(
        p for p in subject_root.rglob("*mask*.nii*") if p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz")
    )
    if len(fallback_candidates) == 1:
        return fallback_candidates[0], True, None
    if len(fallback_candidates) > 1:
        return None, True, f"Multiple generic mask candidates found for {subject_id}: {fallback_candidates}"
    return None, False, f"No lesion mask NIfTI found for {subject_id} under {subject_root}"


def load_lesion_volume(mask_path: Path, threshold: float) -> tuple[int, float | None, float | None, str | None]:
    img = nb.load(str(mask_path))
    data = img.get_fdata(dtype=np.float32)
    lesion_voxels = int(np.count_nonzero(data > threshold))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if len(zooms) == 3 and all(math.isfinite(z) and z > 0 for z in zooms):
        voxel_volume_mm3 = float(np.prod(zooms))
        lesion_mm3 = float(lesion_voxels * voxel_volume_mm3)
        spacing_mm = " x ".join(f"{z:.6g}" for z in zooms)
        return lesion_voxels, lesion_mm3, voxel_volume_mm3, spacing_mm
    return lesion_voxels, None, None, None


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


def basic_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def try_skewness(values: np.ndarray) -> tuple[float | None, str | None]:
    try:
        from scipy.stats import skew  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, f"scipy unavailable: {exc}"
    return float(skew(values, bias=False)), None


def unique_sorted(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    raw = np.asarray(values, dtype=np.float64)
    unique = np.unique(raw)
    return [float(x) for x in unique]


def quantile_edges(values: np.ndarray, k: int) -> tuple[list[float], list[str]]:
    warnings: list[str] = []
    raw = np.quantile(values, np.linspace(0.0, 1.0, num=k + 1))
    raw[0] = float(np.min(values))
    raw[-1] = float(np.max(values))
    edges = unique_sorted(raw)
    if len(edges) - 1 < k:
        warnings.append(
            f"Requested {k} quantile bins, but repeated lesion-volume values collapsed edges to {len(edges) - 1} bins."
        )
    return edges, warnings


def log_edges(values: np.ndarray, k: int) -> tuple[list[float], list[str]]:
    warnings: list[str] = []
    transformed = np.log1p(values.astype(np.float64))
    raw = np.expm1(np.linspace(np.min(transformed), np.max(transformed), num=k + 1))
    raw[0] = float(np.min(values))
    raw[-1] = float(np.max(values))
    edges = unique_sorted(raw)
    if len(edges) - 1 < k:
        warnings.append(
            f"Requested {k} log-volume bins, but repeated/degenerate edges collapsed to {len(edges) - 1} bins."
        )
    return edges, warnings


def jenks_edges(values: np.ndarray, k: int) -> tuple[list[float] | None, list[str]]:
    try:
        import jenkspy  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, [f"Natural breaks skipped: jenkspy unavailable ({exc})."]

    unique_count = len(np.unique(values))
    if unique_count < k:
        return None, [f"Natural breaks skipped for k={k}: only {unique_count} unique lesion-volume values."]

    raw = jenkspy.jenks_breaks(values.tolist(), n_classes=k)
    raw[0] = float(np.min(values))
    raw[-1] = float(np.max(values))
    edges = unique_sorted(np.asarray(raw, dtype=np.float64))
    warnings: list[str] = []
    if len(edges) - 1 < k:
        warnings.append(
            f"Requested {k} natural-break bins, but repeated lesion-volume values collapsed edges to {len(edges) - 1} bins."
        )
    return edges, warnings


def assign_bins(values: np.ndarray, edges: list[float]) -> np.ndarray:
    if len(edges) < 2:
        raise ValueError("Need at least two bin edges.")
    bins = np.full(values.shape[0], fill_value=-1, dtype=np.int32)
    for idx in range(len(edges) - 1):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == len(edges) - 2:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        bins[mask] = idx
    if np.any(bins < 0):
        missing = values[bins < 0]
        raise RuntimeError(f"Internal bin assignment failure for lesion volumes: {missing[:10].tolist()}")
    return bins


def scheme_type_preference(name: str) -> int:
    if name.startswith("quantile"):
        return 3
    if name.startswith("jenks"):
        return 2
    if name.startswith("log"):
        return 1
    return 0


def k_preference(k: int) -> int:
    if k == 5:
        return 3
    if k in {4, 6}:
        return 2
    if k == 3:
        return 1
    return 0


def evaluate_scheme(
    scheme_name: str,
    scheme_type: str,
    k: int,
    values: np.ndarray,
    edges: list[float],
    warnings: list[str],
) -> dict[str, Any]:
    scheme_warnings = list(warnings)
    if len(edges) < 2:
        return {
            "name": scheme_name,
            "type": scheme_type,
            "k_requested": k,
            "k_effective": 0,
            "edges": edges,
            "bin_stats": [],
            "warnings": scheme_warnings + ["Fewer than two bin edges; scheme unusable."],
            "all_bins_hard_min_ok": False,
            "all_bins_preferred_min_ok": False,
            "label": "bad",
            "score": -999,
            "balance_ratio": 0.0,
        }

    assignments = assign_bins(values, edges)
    bin_stats: list[dict[str, Any]] = []
    counts: list[int] = []
    for idx in range(len(edges) - 1):
        bin_values = values[assignments == idx]
        count = int(bin_values.size)
        counts.append(count)
        if count == 0:
            scheme_warnings.append(f"Bin {idx + 1} is empty.")
            min_value = None
            median_value = None
            max_value = None
        else:
            min_value = float(np.min(bin_values))
            median_value = float(np.median(bin_values))
            max_value = float(np.max(bin_values))
        bin_stats.append(
            {
                "bin_index": idx + 1,
                "lower_inclusive": float(edges[idx]),
                "upper_inclusive_last_bin_only": float(edges[idx + 1]),
                "count": count,
                "min": min_value,
                "median": median_value,
                "max": max_value,
                "hard_min_ok": count >= HARD_MIN_PER_BIN,
                "preferred_min_ok": count >= PREFERRED_MIN_PER_BIN,
            }
        )

    k_effective = len(bin_stats)
    min_count = min(counts)
    max_count = max(counts)
    balance_ratio = float(min_count / max_count) if max_count else 0.0
    hard_ok = all(count >= HARD_MIN_PER_BIN for count in counts)
    preferred_ok = all(count >= PREFERRED_MIN_PER_BIN for count in counts)
    preferred_count = sum(count >= PREFERRED_MIN_PER_BIN for count in counts)
    collapsed = k_effective < k

    score = 0
    if hard_ok:
        score += 20
    else:
        score -= 100
    score += preferred_count * 3
    score += scheme_type_preference(scheme_name) * 2
    score += k_preference(k) * 3
    score += int(balance_ratio * 10)
    if preferred_ok and k in {4, 5, 6}:
        score += 10
    if collapsed:
        score -= 10 * (k - k_effective)
    if k > 6:
        score -= 5

    if not hard_ok or k_effective < 2:
        label = "bad"
    elif preferred_ok and not collapsed and k in {4, 5, 6}:
        label = "good"
    elif preferred_count >= max(k_effective - 1, 1):
        label = "acceptable"
    else:
        label = "risky"

    return {
        "name": scheme_name,
        "type": scheme_type,
        "k_requested": k,
        "k_effective": k_effective,
        "edges": [float(edge) for edge in edges],
        "bin_stats": bin_stats,
        "warnings": scheme_warnings,
        "all_bins_hard_min_ok": hard_ok,
        "all_bins_preferred_min_ok": preferred_ok,
        "label": label,
        "score": score,
        "balance_ratio": balance_ratio,
    }


def choose_recommendation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No candidate binning schemes were generated.")
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["score"],
            scheme_type_preference(item["name"]),
            k_preference(int(item["k_requested"])),
            -len(item["warnings"]),
        ),
        reverse=True,
    )
    return ranked[0]


def summarize_spacing(records: list[SubjectRecord]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.status == "ok" and record.spacing_mm:
            counter[record.spacing_mm] += 1
    return [{"spacing_mm": spacing, "count": count} for spacing, count in counter.most_common()]


def spacing_consistent(spacing_rows: list[dict[str, Any]]) -> bool:
    return len(spacing_rows) <= 1


def analyze_subjects(
    gt_deriv: Path,
    entries: list[tuple[str, str, str]],
    mask_threshold: float,
) -> tuple[list[SubjectRecord], list[str]]:
    records: list[SubjectRecord] = []
    warnings: list[str] = []

    for subject_id, split_source, split_key_used in entries:
        mask_path, used_fallback, error = resolve_mask_path(gt_deriv, subject_id)
        if mask_path is None:
            records.append(
                SubjectRecord(
                    subject_id=subject_id,
                    split_source=split_source,
                    split_key_used=split_key_used,
                    mask_path="",
                    lesion_voxels=None,
                    lesion_mm3=None,
                    voxel_volume_mm3=None,
                    spacing_mm=None,
                    used_fallback_mask_path=used_fallback,
                    status="missing_mask",
                    error=error,
                )
            )
            continue
        try:
            lesion_voxels, lesion_mm3, voxel_volume_mm3, spacing_mm = load_lesion_volume(mask_path, mask_threshold)
        except Exception as exc:
            records.append(
                SubjectRecord(
                    subject_id=subject_id,
                    split_source=split_source,
                    split_key_used=split_key_used,
                    mask_path=str(mask_path),
                    lesion_voxels=None,
                    lesion_mm3=None,
                    voxel_volume_mm3=None,
                    spacing_mm=None,
                    used_fallback_mask_path=used_fallback,
                    status="unreadable_mask",
                    error=str(exc),
                )
            )
            continue
        if used_fallback:
            warnings.append(f"{subject_id}: mask resolved via fallback path discovery -> {mask_path}")
        records.append(
            SubjectRecord(
                subject_id=subject_id,
                split_source=split_source,
                split_key_used=split_key_used,
                mask_path=str(mask_path),
                lesion_voxels=lesion_voxels,
                lesion_mm3=lesion_mm3,
                voxel_volume_mm3=voxel_volume_mm3,
                spacing_mm=spacing_mm,
                used_fallback_mask_path=used_fallback,
                status="ok",
                error=None,
            )
        )

    return records, warnings


def maybe_make_histogram(values: np.ndarray, out_path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(values, bins=40, color="#2b6cb0", edgecolor="white")
    axes[0].set_title("Lesion Volume Histogram")
    axes[0].set_xlabel("Lesion voxels")
    axes[0].set_ylabel("Subjects")

    axes[1].hist(np.log1p(values), bins=40, color="#dd6b20", edgecolor="white")
    axes[1].set_title("Log1p Lesion Volume Histogram")
    axes[1].set_xlabel("log(1 + lesion voxels)")
    axes[1].set_ylabel("Subjects")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def format_edges(edges: list[float]) -> str:
    return "[" + ", ".join(format_number(edge, digits=3) for edge in edges) + "]"


def half_open_range_text(index: int, edges: list[float]) -> str:
    lower = format_number(edges[index], digits=3)
    upper = format_number(edges[index + 1], digits=3)
    if index == len(edges) - 2:
        return f"[{lower}, {upper}]"
    return f"[{lower}, {upper})"


def recommendation_rationale(
    recommendation: dict[str, Any],
    stats_summary: dict[str, Any],
    spacing_rows: list[dict[str, Any]],
) -> list[str]:
    rationale = [
        f"{recommendation['name']} keeps {recommendation['k_effective']} bins while staying focused on later 5-fold CV.",
        f"All bins hard-min check (>= {HARD_MIN_PER_BIN} cases): {recommendation['all_bins_hard_min_ok']}.",
        f"All bins preferred-min check (>= {PREFERRED_MIN_PER_BIN} cases): {recommendation['all_bins_preferred_min_ok']}.",
        f"Balance ratio (smallest / largest bin): {recommendation['balance_ratio']:.3f}.",
    ]
    if recommendation["type"] == "quantile":
        rationale.append("Quantile bins are the most stable option for stratified fold construction because they avoid sparse tails.")
    elif recommendation["type"] == "log":
        rationale.append("Log-volume bins better separate the long upper tail, but can create sparse bins if over-fragmented.")
    elif recommendation["type"] == "jenks":
        rationale.append("Natural breaks can capture visible gaps in the distribution, but their cutpoints are less standard than quantiles.")

    if spacing_rows and spacing_consistent(spacing_rows):
        rationale.append("Voxel spacing appears uniform across readable masks, so voxel-count bins also map cleanly to physical volume.")
    if stats_summary["distribution_diagnostics"]["skewness"] is not None:
        rationale.append(
            f"Observed skewness is {stats_summary['distribution_diagnostics']['skewness']:.3f}, "
            "so tail-aware stratification was evaluated explicitly."
        )
    return rationale


def recommendation_caveats(
    recommendation: dict[str, Any],
    zero_lesion_subjects: list[str],
    missing_records: list[SubjectRecord],
) -> list[str]:
    caveats = [
        "These bins are defined on lesion_voxels as the primary metric; lesion_mm3 is secondary output for interpretability.",
        "If the subject pool changes, rerun this script before generating CV folds so the edges remain tied to the exact analysis cohort.",
        "Use half-open intervals [lower, upper) for every bin except the final bin, which is [lower, upper].",
    ]
    if zero_lesion_subjects:
        caveats.append(
            f"Zero-lesion masks were detected ({len(zero_lesion_subjects)} subjects), so CV generation should verify those cases are handled intentionally."
        )
    if missing_records:
        caveats.append(
            f"{len(missing_records)} subjects were missing or unreadable and therefore excluded from the recommendation."
        )
    if recommendation["warnings"]:
        caveats.append("Recommendation still carries scheme-specific warnings; see the candidate scheme section for details.")
    return caveats


def next_step_instruction(recommendation: dict[str, Any]) -> str:
    return (
        "For the later CV generator, compute lesion_voxels for the subject pool to be folded, assign each subject to "
        f"`{recommendation['name']}` using edges {format_edges(recommendation['edges'])}, then stratify a "
        "`StratifiedKFold(n_splits=5, shuffle=True, random_state=9001)`-style split on those bin labels before "
        "writing any fold files."
    )


def dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def write_csv(records: list[SubjectRecord], out_path: Path) -> None:
    fieldnames = [
        "subject_id",
        "split_source",
        "split_key_used",
        "mask_path",
        "lesion_voxels",
        "lesion_mm3",
        "voxel_volume_mm3",
        "spacing_mm",
        "used_fallback_mask_path",
        "status",
        "error",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_markdown_report(
    out_path: Path,
    splits_json: Path,
    gt_deriv: Path,
    records: list[SubjectRecord],
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    recommendation: dict[str, Any],
    histogram_path: str | None,
    split_warnings: list[str],
    analysis_warnings: list[str],
) -> None:
    ok_records = [record for record in records if record.status == "ok"]
    missing_records = [record for record in records if record.status != "ok"]
    zero_lesion_subjects = [record.subject_id for record in ok_records if record.lesion_voxels == 0]
    lines: list[str] = []
    lines.append("# Lesion Volume Stratification Report")
    lines.append("")
    lines.append("Command:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts/supporting_experiments/atlas_lesion_size_stratification/"
        "scripts/analyze_lesion_volume_stratification.py"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Split JSON: `{splits_json}`")
    lines.append(f"- GT derivative root: `{gt_deriv}`")
    lines.append(f"- Subjects listed in split file: `{summary['subject_counts']['subjects_listed_in_split_file']}`")
    lines.append(f"- Readable masks analyzed: `{summary['subject_counts']['readable_masks']}`")
    lines.append(f"- Missing or unreadable masks: `{summary['subject_counts']['missing_or_unreadable_masks']}`")
    lines.append("")
    lines.append("## Lesion-Volume Summary")
    lines.append("")
    lines.append(f"- Primary stratification metric: `lesion_voxels`")
    lines.append(f"- Summary stats: `{json.dumps(summary['lesion_voxel_summary'], sort_keys=True)}`")
    lines.append(f"- Percentiles: `{json.dumps(summary['lesion_voxel_percentiles'], sort_keys=True)}`")
    if summary["lesion_mm3_summary"] is not None:
        lines.append(f"- Physical-volume summary (mm^3): `{json.dumps(summary['lesion_mm3_summary'], sort_keys=True)}`")
    lines.append("")
    lines.append("## Distribution Diagnostics")
    lines.append("")
    diagnostics = summary["distribution_diagnostics"]
    lines.append(f"- Unique lesion-volume values: `{diagnostics['unique_lesion_voxel_values']}`")
    lines.append(
        f"- Very small lesions (`{diagnostics['very_small_definition']}`): "
        f"`{diagnostics['very_small_count']}` / `{summary['subject_counts']['readable_masks']}` "
        f"({diagnostics['very_small_proportion']:.3%})"
    )
    lines.append(f"- Zero-lesion masks: `{diagnostics['zero_lesion_mask_count']}`")
    if zero_lesion_subjects:
        lines.append(f"- Zero-lesion subject IDs: `{', '.join(zero_lesion_subjects)}`")
    if diagnostics["skewness"] is None:
        lines.append(f"- Skewness: skipped (`{diagnostics['skewness_note']}`)")
    else:
        lines.append(f"- Skewness: `{diagnostics['skewness']:.6f}`")
    lines.append(f"- Heavy-skew heuristic triggered: `{diagnostics['heavy_skew_detected']}`")
    lines.append(f"- Observed voxel spacing variants: `{json.dumps(summary['spacing_summary'], sort_keys=False)}`")
    if histogram_path is not None:
        lines.append(f"- Histogram: `{histogram_path}`")
    lines.append("")
    lines.append("## Candidate Binning Schemes")
    lines.append("")
    lines.append(
        f"Each scheme is checked against a hard minimum of `{HARD_MIN_PER_BIN}` subjects per bin and a preferred "
        f"minimum of `{PREFERRED_MIN_PER_BIN}` subjects per bin for later 5-fold CV."
    )
    lines.append("")
    for candidate in candidates:
        lines.append(f"### {candidate['name']}")
        lines.append("")
        lines.append(f"- Type: `{candidate['type']}`")
        lines.append(f"- Requested bins: `{candidate['k_requested']}`")
        lines.append(f"- Effective bins: `{candidate['k_effective']}`")
        lines.append(f"- Bin edges (lesion_voxels): `{format_edges(candidate['edges'])}`")
        lines.append(f"- Recommendation label: `{candidate['label']}`")
        lines.append(f"- Score: `{candidate['score']}`")
        lines.append(f"- All bins >= {HARD_MIN_PER_BIN}: `{candidate['all_bins_hard_min_ok']}`")
        lines.append(f"- All bins >= {PREFERRED_MIN_PER_BIN}: `{candidate['all_bins_preferred_min_ok']}`")
        lines.append(f"- Balance ratio: `{candidate['balance_ratio']:.3f}`")
        if candidate["warnings"]:
            lines.append(f"- Warnings: `{'; '.join(candidate['warnings'])}`")
        else:
            lines.append("- Warnings: none")
        lines.append("")
        lines.append("| Bin | Range (lesion_voxels) | Count | Min | Median | Max | >=5 | >=25 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for idx, bin_row in enumerate(candidate["bin_stats"]):
            lines.append(
                f"| {bin_row['bin_index']} | {half_open_range_text(idx, candidate['edges'])} | "
                f"{bin_row['count']} | {format_number(bin_row['min'])} | {format_number(bin_row['median'])} | "
                f"{format_number(bin_row['max'])} | {bin_row['hard_min_ok']} | {bin_row['preferred_min_ok']} |"
            )
        lines.append("")

    lines.append("## Recommended Carry-Forward Stratification")
    lines.append("")
    lines.append(f"- Chosen binning type: `{recommendation['type']}`")
    lines.append(f"- Chosen scheme: `{recommendation['name']}`")
    lines.append(f"- Chosen bin edges (lesion_voxels): `{format_edges(recommendation['edges'])}`")
    lines.append("- Rationale:")
    for item in recommendation_rationale(recommendation, summary, summary["spacing_summary"]):
        lines.append(f"  {item}")
    lines.append("- Caveats:")
    for item in recommendation_caveats(recommendation, zero_lesion_subjects, missing_records):
        lines.append(f"  {item}")
    lines.append("- Exact next-step instruction for CV fold generation:")
    lines.append("")
    lines.append("```text")
    lines.append(next_step_instruction(recommendation))
    lines.append("```")
    lines.append("")

    if split_warnings or analysis_warnings or missing_records:
        lines.append("## Warnings")
        lines.append("")
        for warning in split_warnings:
            lines.append(f"- {warning}")
        for warning in analysis_warnings:
            lines.append(f"- {warning}")
        for record in missing_records:
            lines.append(f"- {record.subject_id}: {record.status} -> {record.error}")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo_root, gt_deriv, splits_json, out_dir = resolve_paths(args)
    del repo_root
    if not gt_deriv.exists():
        raise SystemExit(f"ERROR: GT derivative root not found: {gt_deriv}")
    if not splits_json.exists():
        raise SystemExit(f"ERROR: split JSON not found: {splits_json}")

    out_dir.mkdir(parents=True, exist_ok=True)
    entries, split_warnings, totals_by_split = load_subject_entries(splits_json)
    records, analysis_warnings = analyze_subjects(gt_deriv, entries, args.mask_threshold)

    records = sorted(records, key=lambda item: item.subject_id)
    write_csv(records, out_dir / "lesion_volume_subjects.csv")

    ok_records = [record for record in records if record.status == "ok" and record.lesion_voxels is not None]
    bad_records = [record for record in records if record.status != "ok"]
    if not ok_records:
        raise SystemExit("ERROR: no readable lesion masks were found for the subjects listed in the split file.")

    lesion_voxels = np.asarray([int(record.lesion_voxels) for record in ok_records], dtype=np.int64)
    lesion_mm3_values = [
        float(record.lesion_mm3) for record in ok_records if record.lesion_mm3 is not None
    ]
    lesion_mm3 = np.asarray(lesion_mm3_values, dtype=np.float64) if lesion_mm3_values else None
    skewness_value, skewness_note = try_skewness(lesion_voxels.astype(np.float64))
    p95 = float(np.percentile(lesion_voxels, 95))
    p50 = float(np.percentile(lesion_voxels, 50))
    heavy_skew = (skewness_value is not None and skewness_value >= 1.0) or (p50 > 0 and (p95 / p50) >= 10.0)

    very_small_count = int(np.count_nonzero(lesion_voxels <= VERY_SMALL_VOXEL_THRESHOLD))
    zero_lesion_count = int(np.count_nonzero(lesion_voxels == 0))
    spacing_rows = summarize_spacing(ok_records)

    candidates: list[dict[str, Any]] = []
    for k in QUANTILE_KS:
        edges, warnings = quantile_edges(lesion_voxels, k)
        candidates.append(evaluate_scheme(f"quantile_k{k}", "quantile", k, lesion_voxels, edges, warnings))
    if heavy_skew:
        for k in LOG_KS:
            edges, warnings = log_edges(lesion_voxels, k)
            candidates.append(evaluate_scheme(f"log_k{k}", "log", k, lesion_voxels, edges, warnings))
    for k in JENKS_KS:
        edges, warnings = jenks_edges(lesion_voxels, k)
        if edges is not None:
            candidates.append(evaluate_scheme(f"jenks_k{k}", "jenks", k, lesion_voxels, edges, warnings))
        else:
            analysis_warnings.extend(warnings)

    split_warnings = dedupe_preserve_order(split_warnings)
    analysis_warnings = dedupe_preserve_order(analysis_warnings)

    recommendation = choose_recommendation(candidates)

    histogram_path = maybe_make_histogram(lesion_voxels.astype(np.float64), out_dir / "lesion_volume_histogram.png")
    summary = {
        "inputs": {
            "splits_json": str(splits_json),
            "gt_deriv": str(gt_deriv),
            "mask_threshold": args.mask_threshold,
        },
        "subject_counts": {
            "subjects_listed_in_split_file": len(entries),
            "subjects_per_split_from_split_file": totals_by_split,
            "readable_masks": len(ok_records),
            "missing_or_unreadable_masks": len(bad_records),
        },
        "lesion_voxel_summary": basic_summary(lesion_voxels.astype(np.float64)),
        "lesion_voxel_percentiles": percentile_summary(lesion_voxels.astype(np.float64)),
        "lesion_mm3_summary": basic_summary(lesion_mm3) if lesion_mm3 is not None and lesion_mm3.size else None,
        "distribution_diagnostics": {
            "unique_lesion_voxel_values": int(np.unique(lesion_voxels).size),
            "very_small_definition": f"lesion_voxels <= {VERY_SMALL_VOXEL_THRESHOLD}",
            "very_small_count": very_small_count,
            "very_small_proportion": float(very_small_count / len(ok_records)),
            "zero_lesion_mask_count": zero_lesion_count,
            "zero_lesion_subject_ids": [record.subject_id for record in ok_records if record.lesion_voxels == 0],
            "skewness": skewness_value,
            "skewness_note": skewness_note,
            "heavy_skew_detected": bool(heavy_skew),
        },
        "spacing_summary": spacing_rows,
        "candidate_schemes": candidates,
        "recommendation": recommendation,
        "split_warnings": split_warnings,
        "analysis_warnings": analysis_warnings,
    }

    (out_dir / "lesion_volume_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_report(
        out_dir / "lesion_volume_stratification_report.md",
        splits_json=splits_json,
        gt_deriv=gt_deriv,
        records=records,
        summary=summary,
        candidates=candidates,
        recommendation=recommendation,
        histogram_path=histogram_path,
        split_warnings=split_warnings,
        analysis_warnings=analysis_warnings,
    )

    print(f"report_dir={out_dir}")
    print(f"subjects_analyzed={len(ok_records)}")
    print(f"missing_or_unreadable_masks={len(bad_records)}")
    print(f"recommended_scheme={recommendation['name']}")
    print(f"recommended_edges_lesion_voxels={format_edges(recommendation['edges'])}")
    if recommendation["warnings"]:
        print(f"recommended_scheme_warnings={'; '.join(recommendation['warnings'])}")
    if bad_records:
        for record in bad_records[:10]:
            print(f"mask_issue={record.subject_id}: {record.status}: {record.error}")


if __name__ == "__main__":
    main()
