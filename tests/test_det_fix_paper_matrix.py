"""Reject plausible-looking but incomplete MR-ART paper inputs."""

import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/supporting_experiments/statistical_inference/scripts/build_paper_statistical_inference.py"
)
SPEC = importlib.util.spec_from_file_location("det_fix_paper_matrix", SCRIPT)
paper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper
SPEC.loader.exec_module(paper)


class DetFixPaperMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keys = itertools.product(
            ["base_cnn", "mednext", "uxnet", "swin_unetr"],
            ["standard", "augmented"],
            range(5),
            [f"sub-{i:06d}" for i in range(140)],
            ["standard", "headmotion1", "headmotion2"],
        )
        cls.valid = pd.DataFrame(
            keys,
            columns=[
                "architecture",
                "training_condition",
                "fold",
                "subject_id",
                "acquisition_type",
            ],
        )
        cls.valid["status"] = "success"
        cls.valid["threshold"] = 0.5
        cls.valid["voxel_volume_mm3"] = 1.0

    def test_complete_matrix_passes(self):
        self.assertEqual(paper.validate_mrart(self.valid)["row_count"], 16800)

    def test_duplicate_replacing_missing_prediction_fails(self):
        frame = self.valid.copy()
        frame.iloc[1] = frame.iloc[0]
        with self.assertRaises(ValueError):
            paper.validate_mrart(frame)

    def test_wrong_fold_preserving_row_and_group_counts_fails(self):
        frame = self.valid.copy()
        frame.loc[frame["fold"] == 0, "fold"] = 5
        with self.assertRaises(ValueError):
            paper.validate_mrart(frame)

    def test_subject_identity_drift_preserving_group_counts_fails(self):
        frame = self.valid.copy()
        frame.loc[0, "subject_id"] = "sub-unexpected"
        with self.assertRaises(ValueError):
            paper.validate_mrart(frame)

    def test_changed_threshold_fails(self):
        frame = self.valid.copy()
        frame.loc[0, "threshold"] = 0.4
        with self.assertRaises(ValueError):
            paper.validate_mrart(frame)

    def test_nonfinite_voxel_volume_fails(self):
        frame = self.valid.copy()
        frame.loc[0, "voxel_volume_mm3"] = float("nan")
        with self.assertRaises(ValueError):
            paper.validate_mrart(frame)


if __name__ == "__main__":
    unittest.main()
