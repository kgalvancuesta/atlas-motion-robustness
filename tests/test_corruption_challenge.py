from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    import torchio  # noqa: F401

    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False

from reproducibility.challenge import (  # noqa: E402
    apply_recipe,
    load_or_create_validated_cache,
    sample_legacy_recipe,
)
from reproducibility.core import ConflictError, array_sha256  # noqa: E402
from reproducibility.preprocessing import preprocess_volume  # noqa: E402


@unittest.skipUnless(TORCHIO_AVAILABLE, "TorchIO is required")
class ChallengeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.volume = np.zeros((24, 25, 26), dtype=np.float32)
        self.volume[3:21, 4:22, 5:23] = np.linspace(1, 100, 18 * 18 * 18, dtype=np.float32).reshape(18, 18, 18)

    def recipe(self):
        for replicate in range(100):
            recipe = sample_legacy_recipe(
                self.volume.shape,
                global_seed=9001,
                subject_id="sub-test",
                fold=0,
                replicate_id=replicate,
            )
            if not recipe["noop"]:
                return recipe
        self.fail("Expected at least one deterministic non-noop recipe")

    def test_stable_recipe_id_and_exact_reconstruction(self) -> None:
        first_recipe = self.recipe()
        second_recipe = sample_legacy_recipe(
            self.volume.shape,
            global_seed=9001,
            subject_id="sub-test",
            fold=0,
            replicate_id=first_recipe["replicate_id"],
        )
        self.assertEqual(first_recipe, second_recipe)
        first = apply_recipe(self.volume, first_recipe)
        second = apply_recipe(self.volume, second_recipe)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(array_sha256(first), array_sha256(second))

    def test_historical_noop_is_possible(self) -> None:
        recipes = [
            sample_legacy_recipe(
                self.volume.shape,
                global_seed=9001,
                subject_id=f"sub-{index}",
                fold=0,
                replicate_id=0,
            )
            for index in range(50)
        ]
        self.assertTrue(any(recipe["noop"] for recipe in recipes))
        self.assertTrue(any(not recipe["noop"] for recipe in recipes))

    def test_preprocessing_modes_and_no_double_normalization(self) -> None:
        recipe = self.recipe()
        legacy, legacy_diagnostics = preprocess_volume(self.volume, mode="legacy", recipe=recipe)
        artifact_first, artifact_diagnostics = preprocess_volume(
            self.volume, mode="artifact_then_normalize", recipe=recipe
        )
        normalize_first, normalize_diagnostics = preprocess_volume(
            self.volume, mode="normalize_then_artifact", recipe=recipe
        )
        self.assertEqual(legacy_diagnostics["normalization_count"], 2)
        self.assertEqual(artifact_diagnostics["normalization_count"], 1)
        self.assertEqual(normalize_diagnostics["normalization_count"], 1)
        self.assertTrue(np.isfinite(legacy).all())
        self.assertTrue(np.isfinite(artifact_first).all())
        self.assertTrue(np.isfinite(normalize_first).all())
        clean, clean_diagnostics = preprocess_volume(self.volume, mode="artifact_then_normalize", recipe=None)
        self.assertNotEqual(float(clean[0, 0, 0]), 0.0)
        self.assertFalse(clean_diagnostics["normalization"]["uses_paired_clean_image"])

    def test_memory_low_high_equivalence_and_cache_validation(self) -> None:
        recipe = self.recipe()
        low, _ = preprocess_volume(self.volume, mode="artifact_then_normalize", recipe=recipe)
        with tempfile.TemporaryDirectory() as temporary:
            kwargs = {
                "cache_root": Path(temporary),
                "challenge_id": "challenge",
                "normalization_mode": "artifact_then_normalize",
                "fold": 0,
                "replicate_id": int(recipe["replicate_id"]),
                "subject_id": "sub-test",
                "source_sha256": "source",
                "recipe_id": recipe["recipe_id"],
            }
            first, first_status = load_or_create_validated_cache(
                **kwargs,
                builder=lambda: preprocess_volume(self.volume, mode="artifact_then_normalize", recipe=recipe)[0],
            )
            second, second_status = load_or_create_validated_cache(
                **kwargs,
                builder=lambda: self.fail("cache should be reused"),
            )
            self.assertEqual(first_status, "created")
            self.assertEqual(second_status, "reused")
            self.assertTrue(np.array_equal(low, first))
            self.assertTrue(np.array_equal(first, second))
            with self.assertRaises(ConflictError):
                load_or_create_validated_cache(**{**kwargs, "source_sha256": "changed"}, builder=lambda: low)


if __name__ == "__main__":
    unittest.main()
