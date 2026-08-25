#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_SCRIPTS = REPO_ROOT / "scripts"
if str(PROJECT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS))

import nibabel as nib
import numpy as np
import torch


@dataclass
class MRArtSubject:
    subject_id: str
    standard_path: Path
    headmotion1_path: Path
    headmotion2_path: Path


def discover_subjects(root: Path) -> list[MRArtSubject]:
    subjects = []
    for sub_dir in sorted(root.glob("sub-*/anat")):
        sid = sub_dir.parent.name
        standard = sub_dir / f"{sid}_acq-standard_T1w.nii.gz"
        hm1 = sub_dir / f"{sid}_acq-headmotion1_T1w.nii.gz"
        hm2 = sub_dir / f"{sid}_acq-headmotion2_T1w.nii.gz"
        if standard.exists() and hm1.exists() and hm2.exists():
            subjects.append(MRArtSubject(sid, standard, hm1, hm2))
    return subjects


def load_scores(root: Path) -> dict[str, int]:
    scores_path = root / "derivatives" / "scores.tsv"
    scores = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                scores[row["bids_name"]] = int(row["score"])
    return scores


def apply_and_save(clean_path: Path, output_path: Path, transform_name: str, available_artifacts: dict) -> None:
    img = nib.load(str(clean_path))
    data = img.get_fdata().astype(np.float32)

    cls, kwargs = available_artifacts[transform_name]
    transform = cls(p=1.0, **kwargs)

    tensor = torch.from_numpy(data[None, ...]).float()
    aug_data = transform(tensor).squeeze(0).numpy()

    out_img = nib.Nifti1Image(aug_data, affine=img.affine, header=img.header)
    nib.save(out_img, str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate TorchIO augmentations against real MR-ART motion artifacts."
    )
    parser.add_argument("--mr_art_root", default="comparison-MRI-Scans")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--n_subjects", type=int, default=3)
    parser.add_argument("--artifacts", nargs="+", default=["motion"])
    parser.add_argument(
        "--output_dir",
        default="outputs/supporting_experiments/mrart/augmentation_validation",
    )
    args = parser.parse_args()

    from torchio_augmentations import AVAILABLE_ARTIFACTS

    root = Path(args.mr_art_root)
    output_dir = Path(args.output_dir)

    for name in args.artifacts:
        if name not in AVAILABLE_ARTIFACTS:
            raise SystemExit(f"Unknown artifact '{name}'. Available: {sorted(AVAILABLE_ARTIFACTS)}")

    all_subjects = discover_subjects(root)
    if not all_subjects:
        raise SystemExit(f"No complete MR-ART subjects found under {root}")

    if args.subjects:
        id_set = {s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects}
        subjects = [s for s in all_subjects if s.subject_id in id_set]
        missing = id_set - {s.subject_id for s in subjects}
        if missing:
            print(f"Warning: subjects not found: {missing}")
        if not subjects:
            raise SystemExit("None of the requested subjects were found.")
    else:
        subjects = all_subjects[:args.n_subjects]

    scores = load_scores(root)

    print(f"MR-ART subjects found: {len(all_subjects)}")
    print(f"Processing: {len(subjects)} subjects")
    print(f"Artifacts: {args.artifacts}")
    print(f"Output: {output_dir}\n")

    for subj in subjects:
        sid = subj.subject_id
        sub_out = output_dir / sid
        sub_out.mkdir(parents=True, exist_ok=True)

        print(f"=== {sid} ===")

        clean_out = sub_out / "clean.nii.gz"
        if not clean_out.exists():
            shutil.copy2(subj.standard_path, clean_out)
        print(f"  Clean scan: {clean_out}")

        for label, src in [
            ("real_headmotion1.nii.gz", subj.headmotion1_path),
            ("real_headmotion2.nii.gz", subj.headmotion2_path),
        ]:
            link = sub_out / label
            if not link.exists():
                link.symlink_to(src.resolve())
            print(f"  {label}: {link}")

        for artifact in args.artifacts:
            out_path = sub_out / f"augmented_{artifact}.nii.gz"
            print(f"  Applying {artifact}...", end=" ", flush=True)
            apply_and_save(subj.standard_path, out_path, artifact, AVAILABLE_ARTIFACTS)
            print(f"saved: {out_path}")

        summary_path = sub_out / "summary.txt"
        std_key = f"{sid}_acq-standard_T1w"
        hm1_key = f"{sid}_acq-headmotion1_T1w"
        hm2_key = f"{sid}_acq-headmotion2_T1w"
        with open(summary_path, "w") as f:
            f.write(f"Subject: {sid}\n\n")
            f.write("Quality scores (1=clean, 2=mild artifacts, 3=severe artifacts):\n")
            f.write(f"  standard:    {scores.get(std_key, '?')}\n")
            f.write(f"  headmotion1: {scores.get(hm1_key, '?')}\n")
            f.write(f"  headmotion2: {scores.get(hm2_key, '?')}\n\n")
            f.write("Files in this directory:\n")
            f.write("  clean.nii.gz               - Original clean scan\n")
            for artifact in args.artifacts:
                f.write(f"  augmented_{artifact}.nii.gz  - TorchIO {artifact} applied to clean\n")
            f.write("  real_headmotion1.nii.gz    - Real mild motion artifacts\n")
            f.write("  real_headmotion2.nii.gz    - Real severe motion artifacts\n")
        print()

    print("=" * 60)
    print("VIEWING IN 3D SLICER")
    print("=" * 60)
    print()
    print("1. Open 3D Slicer")
    print("2. Drag & drop all .nii.gz files from one subject folder")
    print(f"   e.g.: {output_dir / subjects[0].subject_id}/*.nii.gz")
    print("3. Use 'Four-Up' or 'Three Over Three' layout")
    print("4. Pin each viewer panel and select a different volume")
    print("5. Link viewers with the chain-link icon to scroll together")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
