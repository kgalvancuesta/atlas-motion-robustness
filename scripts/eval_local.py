#!/usr/bin/env python3
"""Quick non-Docker local evaluation for ATLAS predictions.

This script is designed for local evaluation of performance on heldout/dev/train
metrics. It mirrors the training-time Dice definition and adds strict safety
checks for split usage, subject normalization, and GT/pred path pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

from split_utils import load_split_payload, resolve_split_ids


MASK_GLOB = "*_label-L_desc-T1lesion_mask.nii.gz"
SUBJECT_REGEX = re.compile(r"^sub-r\d+s\d+$")
DEFAULT_PREDICTION_DERIVATIVE = "atlas2_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick local Dice-based evaluation without Docker.")
    parser.add_argument("--run_dir", default=None, help="Run directory containing preds_bids/")
    parser.add_argument("--preds_bids", default=None, help="Path to preds_bids root (overrides --run_dir)")
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/)")
    parser.add_argument(
        "--gt_deriv",
        default=None,
        help="GT derivative root (default: <data_root>/train/derivatives/ATLAS)",
    )
    parser.add_argument(
        "--splits_json",
        default="splits/atlas_5fold_lesion_quartile_excluding_known_issues.json",
        help="Split JSON path (defaults to the paper CV master; pass --cv_fold).",
    )
    parser.add_argument("--split", default="heldout", choices=["train", "dev", "val", "heldout", "test"])
    parser.add_argument("--cv_fold", type=int, default=None, help="CV fold index for master CV split JSONs")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on split subjects")
    parser.add_argument("--out_json", default=None, help="Output JSON report path")
    parser.add_argument("--strict", action="store_true", help="Fail on fallback resolution and suspicious path states")
    parser.add_argument(
        "--sample_check",
        action="store_true",
        help="Run deeper GT-vs-pred equality/hash/affine checks on worst/best/random subjects",
    )
    parser.add_argument("--sample_size", type=int, default=10, help="Requested number of sample subjects")
    parser.add_argument("--sample_seed", type=int, default=9001, help="Seed used for random sample subset")
    parser.add_argument(
        "--metric_mode",
        default="training_compatible",
        choices=["training_compatible", "soft"],
        help="training_compatible mirrors train_base_cnn thresholded Dice; soft uses probabilistic Dice",
    )
    parser.add_argument("--pred_threshold", type=float, default=0.5, help="Prediction threshold for binary Dice")
    parser.add_argument("--gt_threshold", type=float, default=0.5, help="GT threshold for binary Dice")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for Dice/Jaccard/precision/recall")
    parser.add_argument("--affine_atol", type=float, default=1e-5, help="Absolute tolerance for affine check")
    parser.add_argument("--affine_rtol", type=float, default=1e-5, help="Relative tolerance for affine check")
    return parser.parse_args()


def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def list_symlinks(root: Path, max_items: int = 10) -> list[Path]:
    links: list[Path] = []
    if not root.exists():
        return links
    for p in root.rglob("*"):
        if p.is_symlink():
            links.append(p)
            if len(links) >= max_items:
                break
    return links


def detect_subject_convention(gt_deriv: Path) -> tuple[str, list[str]]:
    names = sorted([p.name for p in gt_deriv.glob("sub-*") if p.is_dir()])
    sample = names[:20]
    if names and all(SUBJECT_REGEX.match(x) for x in names):
        return "sub-r<digits>s<digits>", sample
    return "sub-<token>", sample


def normalize_subject_id(raw: str) -> str:
    """Normalize IDs into dataset convention (e.g., sub-r009s077)."""
    value = raw.strip().lower()
    if value.startswith("sub-sub-"):
        value = value[4:]

    if re.fullmatch(r"sub-r\d+s\d+", value):
        return value
    if re.fullmatch(r"subr\d+s\d+", value):
        match = re.fullmatch(r"subr(\d+)s(\d+)", value)
        if match:
            return f"sub-r{match.group(1)}s{match.group(2)}"
    if re.fullmatch(r"r\d+s\d+", value):
        return f"sub-{value}"
    if value.startswith("sub-"):
        return value
    if value.startswith("sub") and len(value) > 3:
        return f"sub-{value[3:]}"
    return f"sub-{value}"


def expected_mask_path(root: Path, subject: str) -> Path:
    return root / subject / "ses-1" / "anat" / f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"


def resolve_subject_mask(root: Path, subject: str, role: str) -> tuple[Path | None, bool]:
    expected = expected_mask_path(root, subject)
    if expected.exists():
        return expected, False

    subject_root = root / subject
    if not subject_root.exists():
        return None, False

    matches = sorted(subject_root.rglob(MASK_GLOB))
    if len(matches) == 1:
        return matches[0], True
    if len(matches) > 1:
        joined = "\n".join(f"    - {p}" for p in matches)
        raise RuntimeError(
            f"Ambiguous {role} mask discovery for {subject} under {root}.\n"
            f"Expected path missing: {expected}\n"
            f"Found {len(matches)} matches:\n{joined}"
        )
    return None, False


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    img = nb.load(str(path))
    data = img.get_fdata()
    return data, img.affine, data.shape


def binarize(data: np.ndarray, threshold: float) -> np.ndarray:
    return data > threshold


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> tuple[float, int, int, int]:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    inter = int(np.logical_and(pred, target).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(target.sum())
    denom = pred_sum + gt_sum
    dice = float((2.0 * inter + eps) / (denom + eps))
    return dice, inter, pred_sum, gt_sum


def soft_dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float64, copy=False).reshape(-1)
    target = target.astype(np.float64, copy=False).reshape(-1)
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return float((2.0 * inter + eps) / (denom + eps))


def summarize_metric(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def resolve_pred_deriv_root(preds_bids: Path) -> Path:
    derivatives_root = preds_bids / "derivatives"
    if not derivatives_root.exists():
        raise SystemExit(f"ERROR: derivatives directory not found under predictions root: {derivatives_root}")

    candidate = derivatives_root / DEFAULT_PREDICTION_DERIVATIVE
    if candidate.exists():
        return candidate

    subdirs = sorted([p for p in derivatives_root.iterdir() if p.is_dir()])
    if len(subdirs) == 1:
        return subdirs[0]
    raise SystemExit(
        "ERROR: could not resolve prediction derivative directory.\n"
        f"  looked in: {derivatives_root}\n"
        f"  expected '{DEFAULT_PREDICTION_DERIVATIVE}' or exactly one derivative directory"
    )


def choose_sample_subjects(rows: list[dict], sample_size: int, seed: int) -> list[str]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: r["dice"])
    worst = [r["subject"] for r in rows_sorted[:3]]
    best = [r["subject"] for r in rows_sorted[-3:]]

    target_total = max(10, sample_size)
    random_needed = max(4, target_total - 6)

    excluded = set(worst + best)
    candidates = [r["subject"] for r in rows if r["subject"] not in excluded]
    rng = random.Random(seed)
    random_ids = rng.sample(candidates, min(random_needed, len(candidates)))

    selected: list[str] = []
    for sid in worst + best + random_ids:
        if sid not in selected:
            selected.append(sid)
    return selected


def mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8), axis=None)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def run_sample_check(rows: list[dict], args: argparse.Namespace) -> dict:
    subjects = choose_sample_subjects(rows, args.sample_size, args.sample_seed)
    if not subjects:
        return {"subjects": [], "identical_subjects": [], "near_identical_subjects": []}

    by_subject = {r["subject"]: r for r in rows}
    details: list[dict] = []
    identical_subjects: list[str] = []
    near_identical_subjects: list[str] = []

    print("\n=== Sample Check (GT vs Pred Integrity) ===")
    print(f"subjects={subjects}")
    print(
        "columns: subject,dice,pred_voxels,gt_voxels,equality_fraction,"
        "hash_pred[:16],hash_gt[:16],identical,shape_pred,shape_gt,affine_close"
    )

    for sid in subjects:
        row = by_subject[sid]
        pred_data, pred_affine, pred_shape = load_mask(Path(row["pred_path"]))
        gt_data, gt_affine, gt_shape = load_mask(Path(row["gt_path"]))
        pred_mask = binarize(pred_data, args.pred_threshold)
        gt_mask = binarize(gt_data, args.gt_threshold)

        eq_fraction = float((pred_mask == gt_mask).mean())
        h_pred = mask_hash(pred_mask)
        h_gt = mask_hash(gt_mask)
        identical = bool(np.array_equal(pred_mask, gt_mask))
        affine_close = bool(np.allclose(pred_affine, gt_affine, atol=args.affine_atol, rtol=args.affine_rtol))

        if identical:
            identical_subjects.append(sid)
        if eq_fraction > 0.99999 and abs(int(pred_mask.sum()) - int(gt_mask.sum())) <= 1:
            near_identical_subjects.append(sid)

        print(
            f"{sid},{row['dice']:.4f},{int(pred_mask.sum())},{int(gt_mask.sum())},"
            f"{eq_fraction:.6f},{h_pred[:16]},{h_gt[:16]},{identical},"
            f"{pred_shape},{gt_shape},{affine_close}"
        )
        print(f"  affine_pred={np.array2string(pred_affine, precision=3, suppress_small=True)}")
        print(f"  affine_gt={np.array2string(gt_affine, precision=3, suppress_small=True)}")

        details.append(
            {
                "subject": sid,
                "dice": row["dice"],
                "pred_voxels": int(pred_mask.sum()),
                "gt_voxels": int(gt_mask.sum()),
                "equality_fraction": eq_fraction,
                "pred_hash_sha256": h_pred,
                "gt_hash_sha256": h_gt,
                "identical": identical,
                "shape_pred": list(pred_shape),
                "shape_gt": list(gt_shape),
                "affine_close": affine_close,
                "affine_pred": pred_affine.tolist(),
                "affine_gt": gt_affine.tolist(),
            }
        )

    # Full evaluated-subject identity sweep
    full_identical: list[str] = []
    for row in rows:
        pred_data, _pred_affine, _pred_shape = load_mask(Path(row["pred_path"]))
        gt_data, _gt_affine, _gt_shape = load_mask(Path(row["gt_path"]))
        if np.array_equal(
            binarize(pred_data, args.pred_threshold),
            binarize(gt_data, args.gt_threshold),
        ):
            full_identical.append(row["subject"])

    print(f"full_identical_masks={len(full_identical)} / {len(rows)}")
    if full_identical:
        print(f"PRED==GT SUSPICIOUS: identical subjects (first20): {full_identical[:20]}")

    return {
        "subjects": details,
        "identical_subjects": identical_subjects,
        "near_identical_subjects": near_identical_subjects,
        "full_identical_subjects": full_identical,
    }


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data")).resolve()
    gt_deriv = (Path(args.gt_deriv).resolve() if args.gt_deriv else (data_root / "train" / "derivatives" / "ATLAS").resolve())

    if args.preds_bids:
        preds_bids = Path(args.preds_bids).resolve()
    elif run_dir is not None:
        preds_bids = (run_dir / "preds_bids").resolve()
    else:
        raise SystemExit("ERROR: provide either --run_dir or --preds_bids")

    if not gt_deriv.exists():
        raise SystemExit(f"ERROR: GT derivative not found: {gt_deriv}")
    if not preds_bids.exists():
        raise SystemExit(f"ERROR: predictions root not found: {preds_bids}")

    pred_deriv_root = resolve_pred_deriv_root(preds_bids).resolve()
    splits_path = Path(args.splits_json).resolve()
    if not splits_path.exists():
        raise SystemExit(f"ERROR: splits JSON not found: {splits_path}")

    print("=== Path Resolution ===")
    print(f"data_root={data_root}")
    print(f"gt_deriv={gt_deriv}")
    print(f"preds_bids={preds_bids}")
    print(f"pred_deriv_root={pred_deriv_root}")
    print(f"splits_json={splits_path}")

    pred_derivatives_root = (preds_bids / "derivatives").resolve()
    if not path_is_within(pred_deriv_root, pred_derivatives_root):
        raise SystemExit(
            "ERROR: pred_deriv_root is not inside preds_bids/derivatives.\n"
            f"  pred_deriv_root={pred_deriv_root}\n"
            f"  expected_parent={pred_derivatives_root}"
        )

    if path_is_within(pred_deriv_root, gt_deriv) or path_is_within(gt_deriv, pred_deriv_root) or pred_deriv_root == gt_deriv:
        raise SystemExit(
            "ERROR: GT and prediction paths overlap or alias.\n"
            f"  gt_deriv={gt_deriv}\n"
            f"  pred_deriv_root={pred_deriv_root}"
        )

    if run_dir is not None:
        expected_run_pred_root = (run_dir / "preds_bids" / "derivatives").resolve()
        if not path_is_within(pred_deriv_root, expected_run_pred_root):
            raise SystemExit(
                "ERROR: pred_deriv_root is not under run_dir/preds_bids/derivatives.\n"
                f"  run_dir={run_dir}\n"
                f"  pred_deriv_root={pred_deriv_root}"
            )

    gt_links = list_symlinks(gt_deriv)
    pred_links = list_symlinks(pred_deriv_root)
    print(f"gt_symlink_count_first_scan={len(gt_links)}")
    print(f"pred_symlink_count_first_scan={len(pred_links)}")
    if gt_links:
        print(f"gt_symlink_examples={[str(p) for p in gt_links]}")
    if pred_links:
        print(f"pred_symlink_examples={[str(p) for p in pred_links]}")
    if args.strict and (gt_links or pred_links):
        raise SystemExit("ERROR: --strict enabled and symlinks detected under GT or prediction roots.")

    splits = load_split_payload(splits_path)
    raw_ids, split_meta = resolve_split_ids(splits, args.split, cv_fold=args.cv_fold)
    split_key = split_meta["split_key_used"]
    split_reason = split_meta["split_key_reason"]
    if args.limit is not None:
        raw_ids = raw_ids[: args.limit]

    convention, convention_sample = detect_subject_convention(gt_deriv)
    print("\n=== Subject ID Convention ===")
    print(f"detected_convention={convention}")
    print(f"dataset_subject_samples={convention_sample[:10]}")

    ids = [normalize_subject_id(sid) for sid in raw_ids]
    print("split_raw_to_normalized_first10:")
    for raw, norm in list(zip(raw_ids, ids))[:10]:
        print(f"  raw={raw} -> normalized={norm}")

    malformed = [sid for sid in ids if not SUBJECT_REGEX.match(sid)]
    if malformed:
        print(f"WARNING: malformed_normalized_ids_count={len(malformed)}")
        print(f"malformed_examples_first20={malformed[:20]}")
        if args.strict:
            raise SystemExit("ERROR: --strict enabled and malformed normalized subject IDs were found.")

    if not ids:
        raise SystemExit(f"ERROR: no IDs found for key '{split_key}' in {splits_path}")

    rows: list[dict] = []
    missing_pred: list[str] = []
    missing_gt: list[str] = []
    shape_mismatch: list[str] = []
    affine_mismatch: list[str] = []
    load_errors: list[dict] = []
    fallback_gt: list[str] = []
    fallback_pred: list[str] = []

    global_inter = 0
    global_pred_sum = 0
    global_gt_sum = 0

    for subject in ids:
        try:
            gt_path, gt_used_fallback = resolve_subject_mask(gt_deriv, subject, role="GT")
            pred_path, pred_used_fallback = resolve_subject_mask(pred_deriv_root, subject, role="Prediction")
        except RuntimeError as exc:
            raise SystemExit(f"ERROR: {exc}")

        if gt_used_fallback:
            fallback_gt.append(subject)
            if args.strict:
                raise SystemExit(
                    f"ERROR: --strict enabled and GT mask for {subject} relied on fallback discovery."
                )
        if pred_used_fallback:
            fallback_pred.append(subject)
            if args.strict:
                raise SystemExit(
                    f"ERROR: --strict enabled and prediction mask for {subject} relied on fallback discovery."
                )

        if gt_path is None:
            missing_gt.append(subject)
            continue
        if pred_path is None:
            missing_pred.append(subject)
            continue

        try:
            gt_data, gt_affine, gt_shape = load_mask(gt_path)
            pred_data, pred_affine, pred_shape = load_mask(pred_path)
        except Exception as exc:
            load_errors.append({"subject": subject, "error": str(exc)})
            continue

        if gt_shape != pred_shape:
            shape_mismatch.append(subject)
            continue

        affine_close = bool(np.allclose(gt_affine, pred_affine, atol=args.affine_atol, rtol=args.affine_rtol))
        if not affine_close:
            affine_mismatch.append(subject)
            if args.strict:
                raise SystemExit(
                    f"ERROR: --strict enabled and affine mismatch found for {subject}. "
                    f"gt={gt_path} pred={pred_path}"
                )

        if args.metric_mode == "training_compatible":
            gt_eval = binarize(gt_data, args.gt_threshold)
            pred_eval = binarize(pred_data, args.pred_threshold)
            dice, inter, pred_sum, gt_sum = dice_score(pred_eval, gt_eval, eps=args.eps)
        else:
            gt_eval = gt_data
            pred_eval = pred_data
            dice = soft_dice_score(pred_eval, gt_eval, eps=args.eps)
            pred_sum = float(pred_eval.sum())
            gt_sum = float(gt_eval.sum())
            inter = float((pred_eval * gt_eval).sum())

        union = int(np.logical_or(pred_eval > 0.5, gt_eval > 0.5).sum()) if args.metric_mode == "soft" else int(np.logical_or(pred_eval, gt_eval).sum())
        jaccard = float((inter + args.eps) / (union + args.eps))
        precision = float((inter + args.eps) / (pred_sum + args.eps))
        recall = float((inter + args.eps) / (gt_sum + args.eps))
        abs_vol_diff_ratio = float(abs(pred_sum - gt_sum) / (gt_sum + args.eps))

        if args.metric_mode == "training_compatible":
            global_inter += inter
            global_pred_sum += pred_sum
            global_gt_sum += gt_sum

        rows.append(
            {
                "subject": subject,
                "dice": float(dice),
                "jaccard": jaccard,
                "precision": precision,
                "recall": recall,
                "pred_voxels": int(pred_sum) if args.metric_mode == "training_compatible" else float(pred_sum),
                "gt_voxels": int(gt_sum) if args.metric_mode == "training_compatible" else float(gt_sum),
                "abs_volume_diff_ratio": abs_vol_diff_ratio,
                "shape_gt": list(gt_shape),
                "shape_pred": list(pred_shape),
                "affine_close": affine_close,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
            }
        )

    dice_values = [r["dice"] for r in rows]
    jaccard_values = [r["jaccard"] for r in rows]
    precision_values = [r["precision"] for r in rows]
    recall_values = [r["recall"] for r in rows]
    vol_diff_values = [r["abs_volume_diff_ratio"] for r in rows]

    micro_dice = None
    if args.metric_mode == "training_compatible":
        micro_dice = float((2.0 * global_inter + args.eps) / (global_pred_sum + global_gt_sum + args.eps))

    rows_sorted = sorted(rows, key=lambda r: r["dice"])
    worst10 = [{"subject": r["subject"], "dice": r["dice"]} for r in rows_sorted[:10]]
    best10 = [{"subject": r["subject"], "dice": r["dice"]} for r in rows_sorted[-10:]][::-1]

    summary = {
        "split_requested": args.split,
        "split_key_used": split_key,
        "split_key_reason": split_reason,
        "cv_fold": args.cv_fold,
        "metric_mode": args.metric_mode,
        "metric_definition": {
            "pred_threshold": args.pred_threshold if args.metric_mode == "training_compatible" else None,
            "gt_threshold": args.gt_threshold if args.metric_mode == "training_compatible" else None,
            "eps": args.eps,
            "training_compatible_note": "Matches train_base_cnn validation Dice when using binary masks.",
        },
        "n_subjects_in_split": len(ids),
        "n_evaluated": len(rows),
        "n_missing_pred": len(missing_pred),
        "n_missing_gt": len(missing_gt),
        "n_shape_mismatch": len(shape_mismatch),
        "n_affine_mismatch": len(affine_mismatch),
        "n_load_errors": len(load_errors),
        "n_fallback_gt": len(fallback_gt),
        "n_fallback_pred": len(fallback_pred),
        "micro_dice": micro_dice,
        "dice": summarize_metric(dice_values),
        "jaccard": summarize_metric(jaccard_values),
        "precision": summarize_metric(precision_values),
        "recall": summarize_metric(recall_values),
        "abs_volume_diff_ratio": summarize_metric(vol_diff_values),
        "missing_pred_subjects_first20": missing_pred[:20],
        "missing_gt_subjects_first20": missing_gt[:20],
        "shape_mismatch_subjects_first20": shape_mismatch[:20],
        "affine_mismatch_subjects_first20": affine_mismatch[:20],
        "fallback_gt_subjects_first20": fallback_gt[:20],
        "fallback_pred_subjects_first20": fallback_pred[:20],
        "worst10_dice": worst10,
        "best10_dice": best10,
    }

    sample_check_report = None
    if args.sample_check:
        sample_check_report = run_sample_check(rows, args)
        if sample_check_report["full_identical_subjects"]:
            print("PRED==GT SUSPICIOUS")
            if args.strict:
                raise SystemExit("ERROR: --strict enabled and identical GT/pred masks were found.")

    report = {
        "config": {
            "run_dir": str(run_dir) if run_dir else None,
            "data_root": str(data_root),
            "gt_deriv": str(gt_deriv),
            "preds_bids": str(preds_bids),
            "pred_deriv": str(pred_deriv_root),
            "splits_json": str(splits_path),
            "cv_fold": args.cv_fold,
            "split_resolution": split_meta,
            "limit": args.limit,
            "strict": args.strict,
            "sample_check": args.sample_check,
        },
        "summary": summary,
        "per_subject": rows,
        "load_errors": load_errors,
        "sample_check": sample_check_report,
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
    elif run_dir is not None:
        out_path = (run_dir / "eval_local" / f"{args.split}_quick_metrics.json").resolve()
    else:
        out_path = (Path("eval") / "local" / f"{args.split}_quick_metrics.json").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== Quick Local Evaluation (No Docker) ===")
    print(f"Split requested: {args.split}")
    print(f"Split key used: {split_key} ({split_reason})")
    print(f"Metric mode: {args.metric_mode}")
    print(f"GT root: {gt_deriv}")
    print(f"Pred root: {pred_deriv_root}")
    print(f"Subjects in split: {len(ids)}")
    print(f"Evaluated: {len(rows)}")
    print(f"Missing predictions: {len(missing_pred)}")
    print(f"Missing GT: {len(missing_gt)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")
    print(f"Affine mismatches: {len(affine_mismatch)}")
    print(f"Load errors: {len(load_errors)}")
    print(f"Fallback GT resolutions: {len(fallback_gt)}")
    print(f"Fallback prediction resolutions: {len(fallback_pred)}")

    if dice_values:
        print(
            "Dice: "
            f"mean={summary['dice']['mean']:.4f} "
            f"median={summary['dice']['median']:.4f} "
            f"std={summary['dice']['std']:.4f} "
            f"min={summary['dice']['min']:.4f} "
            f"max={summary['dice']['max']:.4f}"
        )
        if micro_dice is not None:
            print(f"Micro Dice: {micro_dice:.4f}")
    else:
        print("No subjects were successfully scored.")

    print(f"Saved report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
