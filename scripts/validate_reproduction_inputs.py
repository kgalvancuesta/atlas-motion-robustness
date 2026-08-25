#!/usr/bin/env python3
"""Validate the fixed paper folds and their ATLAS inputs.

The full check is intended for the authoritative server dataset. ``--limit`` is
only a local implementation smoke hook; it never reports full dataset success.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

EXPECTED_INCLUDED = 653
EXPECTED_EXCLUDED = {"sub-r039s002", "sub-r009s003"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path("splits/atlas_5fold_lesion_quartile_excluding_known_issues.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Local-only cap; makes dataset status NOT RUN")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def normalized(values: list[Any]) -> list[str]:
    return [str(value) if str(value).startswith("sub-") else f"sub-{value}" for value in values]


def validate_folds(payload: dict[str, Any]) -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        return [Check("fold_count", "FAIL", f"expected 5 folds, found {0 if not isinstance(folds, list) else len(folds)}")], []

    expected_indices = list(range(5))
    actual_indices = sorted(int(fold.get("fold_index", -1)) for fold in folds)
    checks.append(Check("fold_indices", "PASS" if actual_indices == expected_indices else "FAIL", str(actual_indices)))

    universes: list[set[str]] = []
    test_counts: dict[str, int] = {}
    all_ids: set[str] = set()
    for fold in folds:
        index = int(fold["fold_index"])
        train = normalized(fold.get("train_ids", []))
        val = normalized(fold.get("val_ids", []))
        test = normalized(fold.get("test_ids", []))
        sets = [set(train), set(val), set(test)]
        duplicate_within = any(len(raw) != len(unique) for raw, unique in zip((train, val, test), sets))
        overlap = bool((sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]))
        universe = set().union(*sets)
        universes.append(universe)
        all_ids.update(universe)
        for subject in test:
            test_counts[subject] = test_counts.get(subject, 0) + 1
        ok = not duplicate_within and not overlap and len(universe) == EXPECTED_INCLUDED
        checks.append(
            Check(
                f"fold_{index}_partition",
                "PASS" if ok else "FAIL",
                f"train={len(train)} val={len(val)} test={len(test)} universe={len(universe)} duplicate={duplicate_within} overlap={overlap}",
            )
        )

    same_universe = all(universe == universes[0] for universe in universes[1:])
    checks.append(Check("fold_subject_universe", "PASS" if same_universe else "FAIL", f"subjects={len(all_ids)}"))
    invalid_test_counts = sorted(subject for subject, count in test_counts.items() if count != 1)
    checks.append(
        Check(
            "outer_test_once",
            "PASS" if not invalid_test_counts and len(test_counts) == EXPECTED_INCLUDED else "FAIL",
            f"subjects={len(test_counts)} invalid_count_subjects={len(invalid_test_counts)}",
        )
    )
    exclusions = set(normalized([item.get("subject_id", "") for item in payload.get("metadata", {}).get("excluded_subjects", [])]))
    exclusions.discard("sub-")
    exclusion_ok = exclusions == EXPECTED_EXCLUDED and not (EXPECTED_EXCLUDED & all_ids)
    checks.append(Check("configured_exclusions", "PASS" if exclusion_ok else "FAIL", f"configured={sorted(exclusions)}"))
    return checks, sorted(all_ids)


def subject_paths(derivative_root: Path, subject: str) -> tuple[Path, Path]:
    anat = derivative_root / subject / "ses-1" / "anat"
    stem = f"{subject}_ses-1_space-MNI152NLin2009aSym"
    return anat / f"{stem}_T1w.nii.gz", anat / f"{stem}_label-L_desc-T1lesion_mask.nii.gz"


def validate_subject(derivative_root: Path, subject: str) -> list[str]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        return [f"validation dependency missing: {exc}"]
    errors: list[str] = []
    image_path, mask_path = subject_paths(derivative_root, subject)
    if not image_path.is_file():
        errors.append("missing MRI")
    if not mask_path.is_file():
        errors.append("missing lesion mask")
    if errors:
        return errors
    try:
        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))
        image_data = np.asanyarray(image.dataobj)
        mask_data = np.asanyarray(mask.dataobj)
    except Exception as exc:
        return [f"unreadable: {type(exc).__name__}: {exc}"]
    if image.shape != mask.shape:
        errors.append(f"shape mismatch MRI={image.shape} mask={mask.shape}")
    if len(image.shape) != 3:
        errors.append(f"expected 3D MRI, got {image.shape}")
    if not bool(np.isfinite(image_data).all()):
        errors.append("MRI contains non-finite values")
    if not bool(np.isfinite(mask_data).all()):
        errors.append("mask contains non-finite values")
    unique = np.unique(mask_data)
    if unique.size == 0 or float(unique.min()) < 0 or float(unique.max()) > 1:
        errors.append(f"mask outside [0, 1]: min={float(unique.min()) if unique.size else None} max={float(unique.max()) if unique.size else None}")
    if not bool(np.any(mask_data > 0.5)):
        errors.append("mask has no lesion voxels above 0.5")
    return errors


def main() -> int:
    args = parse_args()
    payload = json.loads(args.splits_json.read_text())
    checks, subject_ids = validate_folds(payload)

    derivative_root = args.data_root / "train" / "derivatives" / "ATLAS"
    if not derivative_root.is_dir():
        checks.append(Check("atlas_derivative_root", "FAIL", f"not found: {derivative_root}"))
    else:
        selected = subject_ids[: args.limit] if args.limit is not None else subject_ids
        failures: dict[str, list[str]] = {}
        for subject in selected:
            errors = validate_subject(derivative_root, subject)
            if errors:
                failures[subject] = errors
        if args.limit is None:
            actual_dirs = {path.name for path in derivative_root.glob("sub-*") if path.is_dir()}
            missing = sorted(set(subject_ids) - actual_dirs)
            unexpected = sorted(actual_dirs - set(subject_ids) - EXPECTED_EXCLUDED)
            status = "PASS" if not failures and not missing and not unexpected and len(subject_ids) == EXPECTED_INCLUDED else "FAIL"
            detail = f"validated={len(selected)} failures={len(failures)} missing={len(missing)} unexpected={len(unexpected)}"
        else:
            status = "NOT RUN" if not failures else "FAIL"
            detail = f"local subset validated={len(selected)}/{len(subject_ids)} failures={len(failures)}; full validation intentionally not run"
        checks.append(Check("atlas_subject_files", status, detail))
        if failures:
            checks.append(Check("atlas_subject_failure_sample", "FAIL", json.dumps(dict(list(failures.items())[:10]), sort_keys=True)))

    payload_out = {
        "mode": "local_subset" if args.limit is not None else "authoritative_full",
        "splits_json": str(args.splits_json),
        "data_root": str(args.data_root),
        "expected_included_subjects": EXPECTED_INCLUDED,
        "expected_exclusions": sorted(EXPECTED_EXCLUDED),
        "checks": [asdict(check) for check in checks],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        if args.output_json.exists():
            raise SystemExit(f"Refusing to overwrite {args.output_json}")
        args.output_json.write_text(json.dumps(payload_out, indent=2) + "\n")
    for check in checks:
        print(f"{check.name}: {check.status} | {check.detail}")
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
