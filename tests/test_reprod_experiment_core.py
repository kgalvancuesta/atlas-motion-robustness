from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from reproducibility.core import (  # noqa: E402
    CompatibilityError,
    ConflictError,
    completed_artifact_status,
    validate_paired_artifacts,
    write_immutable_json,
    write_json_new,
)
from reproducibility.rng import CorrectedSampler, derive_seed  # noqa: E402
from train_common import sample_center  # noqa: E402


class ImmutableDefinitionTests(unittest.TestCase):
    def test_identical_definition_skips_and_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "definition.json"
            self.assertEqual(write_immutable_json(path, {"a": 1, "b": [2]}), "created")
            self.assertEqual(write_immutable_json(path, {"b": [2], "a": 1}), "skipped-compatible")
            with self.assertRaises(ConflictError):
                write_immutable_json(path, {"a": 2, "b": [2]})

    def test_resume_completion_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "artifact.json"
            output = root / "artifact.bin"
            self.assertEqual(completed_artifact_status(metadata, {"experiment_id": "x"}, [output]), "missing")
            output.write_bytes(b"done")
            write_json_new(metadata, {"experiment_id": "x", "model": "m"})
            self.assertEqual(completed_artifact_status(metadata, {"experiment_id": "x"}, [output]), "complete")
            with self.assertRaises(CompatibilityError):
                completed_artifact_status(metadata, {"experiment_id": "other"}, [output])

    def test_partial_resume_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifact.bin"
            output.write_bytes(b"partial")
            with self.assertRaises(ConflictError):
                completed_artifact_status(root / "missing.json", {"experiment_id": "x"}, [output])


class IndependentRngTests(unittest.TestCase):
    def test_seed_is_stable_and_operation_specific(self) -> None:
        first = derive_seed(9001, "patch_center_sampling", fold=0, subject_id="sub-1", epoch=1, patch_slot=0)
        second = derive_seed(9001, "patch_center_sampling", patch_slot=0, epoch=1, subject_id="sub-1", fold=0)
        different = derive_seed(9001, "augmentation_parameters", fold=0, subject_id="sub-1", epoch=1, patch_slot=0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_augmentation_does_not_change_clean_order_or_patch_center(self) -> None:
        clean = list(range(12))
        no_augmentation = CorrectedSampler(clean, [], global_seed=9001, fold=2)
        with_augmentation = CorrectedSampler(clean, list(range(12, 18)), global_seed=9001, fold=2)
        no_augmentation.set_epoch(4)
        with_augmentation.set_epoch(4)
        clean_order = list(no_augmentation)
        combined_order = list(with_augmentation)
        self.assertEqual(combined_order[: len(clean_order)], clean_order)

        mask = np.zeros((20, 21, 22), dtype=np.uint8)
        mask[5:8, 7:10, 9:12] = 1
        seed = derive_seed(
            9001,
            "patch_center_sampling",
            fold=2,
            subject_id="sub-1",
            epoch=4,
            patch_slot=3,
        )
        clean_center = sample_center(mask, (8, 8, 8), 0.7, rng=random.Random(seed))
        augmented_center = sample_center(mask, (8, 8, 8), 0.7, rng=random.Random(seed))
        self.assertEqual(clean_center, augmented_center)


class CompatibilityTests(unittest.TestCase):
    def _artifact(self, **changes):
        value = {
            "experiment_id": "exp",
            "fold_definition_id": "folds",
            "subject_ids_hash": "subjects",
            "subject_ids": ["sub-1"],
            "normalization_mode": "artifact_then_normalize",
            "preprocessing_version": "v1",
            "model": "base_cnn",
            "model_configuration_id": "config",
            "checkpoint_generation": "corrected_experiment_v1",
            "challenge_id": "challenge",
            "corruption_replicate_ids": [0],
        }
        value.update(changes)
        return value

    def test_compatible_pair_passes(self) -> None:
        validate_paired_artifacts([self._artifact(), self._artifact()])

    def test_subject_or_challenge_mismatch_fails(self) -> None:
        with self.assertRaises(CompatibilityError):
            validate_paired_artifacts([self._artifact(), self._artifact(subject_ids=["sub-2"])])
        with self.assertRaises(CompatibilityError):
            validate_paired_artifacts([self._artifact(), self._artifact(challenge_id="other")])
        with self.assertRaises(CompatibilityError):
            validate_paired_artifacts([self._artifact(), self._artifact(model="mednext")])


if __name__ == "__main__":
    unittest.main()
