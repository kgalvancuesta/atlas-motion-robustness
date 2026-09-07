from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_common import Sample, VolumeDataset, corrected_training_samples
from reproducibility.experiment import CHECKPOINT_SELECTION, _duplicate_ids
from reproducibility.rng import CorrectedSampler


class CleanValidationTests(unittest.TestCase):
    def test_all_committed_folds_have_identical_validation_ids_and_lengths(self):
        split = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "splits/atlas_5fold_lesion_quartile_excluding_known_issues.json"
            ).read_text()
        )
        for fold in split["folds"]:
            originals = [
                Sample(sid, Path(sid), Path(sid))
                for sid in sorted(set(fold["train_ids"] + fold["val_ids"]))
            ]
            duplicates, _ = _duplicate_ids(
                fold["train_ids"],
                global_seed=9001,
                fold=fold["fold_index"],
                role="train",
                fraction=0.5,
            )
            current = {
                **fold,
                "duplicated_subjects": {"train_ids": duplicates, "val_ids": []},
            }
            standard, a = corrected_training_samples(originals, current, "standard")
            augmented, b = corrected_training_samples(originals, current, "augmented")
            self.assertEqual([s.subject for s in a], sorted(fold["val_ids"]))
            self.assertEqual(a, b)
            self.assertEqual(len(VolumeDataset(a)), len(VolumeDataset(b)))
            self.assertEqual(len(a), 105)
            self.assertFalse(any(s.augment for s in b))
            self.assertEqual(
                len(augmented), len(standard) + len(fold["train_ids"]) // 2
            )

    def test_regimes_share_clean_validation_and_preserve_training_duplicates(self):
        samples = [
            Sample(f"sub-{i}", Path(f"image-{i}"), Path(f"mask-{i}")) for i in range(8)
        ]
        train_ids = [s.subject for s in samples[:4]]
        duplicates, _ = _duplicate_ids(
            train_ids, global_seed=9001, fold=0, role="train", fraction=0.5
        )
        # Even an old fold carrying validation duplicates cannot introduce them.
        fold = {
            "train_ids": train_ids,
            "val_ids": ["sub-4", "sub-5"],
            "duplicated_subjects": {"train_ids": duplicates, "val_ids": ["sub-4"]},
        }
        standard, standard_val = corrected_training_samples(samples, fold, "standard")
        augmented, augmented_val = corrected_training_samples(
            samples, fold, "augmented"
        )
        self.assertEqual(standard_val, augmented_val)
        self.assertEqual(len(standard_val), 2)
        self.assertFalse(any(s.augment for s in augmented_val))
        self.assertEqual(augmented[:4], standard)
        self.assertEqual([s.subject for s in augmented[4:]], duplicates)
        self.assertTrue(all(s.augment for s in augmented[4:]))
        self.assertEqual(len(augmented), 6)
        for rank in (0, 1):
            clean = list(
                CorrectedSampler(
                    range(32), [], global_seed=9001, fold=0, num_replicas=2, rank=rank
                )
            )
            aug = list(
                CorrectedSampler(
                    range(32),
                    range(32, 48),
                    global_seed=9001,
                    fold=0,
                    num_replicas=2,
                    rank=rank,
                )
            )
            self.assertEqual(aug[: len(clean)], clean)
            self.assertEqual(len(aug), 24)
        with tempfile.TemporaryDirectory() as temporary:
            context = {
                "experiment_id": "test-validation",
                "global_seed": 9001,
                "fold": 0,
                "normalization_mode": "artifact_then_normalize",
                "memory_mode": "high",
                "cache_root": temporary,
                "source_sha256": {"sub-4": "source4", "sub-5": "source5"},
            }
            ds = [
                VolumeDataset(values, experiment_context=context)
                for values in (standard_val, augmented_val)
            ]
            volume = np.arange(120, dtype=np.float32).reshape(4, 5, 6)

            def load(path):
                return (
                    (
                        (volume > 70).astype(np.float32)
                        if "mask" in str(path)
                        else volume
                    ),
                    np.eye(4),
                    None,
                )

            with (
                mock.patch("train_common.load_nifti", side_effect=load),
                mock.patch(
                    "reproducibility.challenge.sample_legacy_recipe",
                    side_effect=AssertionError("validation augmentation"),
                ),
            ):
                ds[0].set_epoch(1)
                ds[1].set_epoch(45)
                for index in range(2):
                    a, b = ds[0][index], ds[1][index]
                    self.assertTrue(torch.equal(a[0], b[0]))
                    self.assertTrue(torch.equal(a[1], b[1]))
                    self.assertEqual(a[2], b[2])
            paths = list(Path(temporary).rglob("*.npy"))
            self.assertEqual(len(paths), 2)
            self.assertTrue(
                all(
                    "validation-clean" in str(p) and "replicate-0" in str(p)
                    for p in paths
                )
            )
        self.assertEqual(CHECKPOINT_SELECTION["metric"], "mean_subject_binary_dice")
        self.assertEqual(
            CHECKPOINT_SELECTION["best_rule"], "strictly_greater_than_previous_best"
        )


if __name__ == "__main__":
    unittest.main()
