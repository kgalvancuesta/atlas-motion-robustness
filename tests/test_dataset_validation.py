from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DatasetValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_module(
            "validate_reproduction_inputs",
            ROOT / "scripts/validate_reproduction_inputs.py",
        )
        cls.mrart = load_module(
            "run_mrart_fold_lcc_stats_no_outputs",
            ROOT
            / "scripts/supporting_experiments/mrart/scripts/"
            "run_mrart_fold_lcc_stats_no_outputs.py",
        )

    def test_fixed_folds_pass_structural_validation(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "splits/atlas_5fold_lesion_quartile_excluding_known_issues.json"
            ).read_text()
        )
        checks, subjects = self.validator.validate_folds(payload)
        self.assertEqual(653, len(subjects))
        self.assertFalse([check for check in checks if check.status == "FAIL"])

    def test_mrart_inventory_counts_complete_and_missing_triplets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            complete = root / "sub-complete" / "anat"
            incomplete = root / "sub-incomplete" / "anat"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            for acquisition in self.mrart.ACQUISITIONS:
                (complete / f"sub-complete_acq-{acquisition}_T1w.nii.gz").touch()
            (incomplete / "sub-incomplete_acq-standard_T1w.nii.gz").touch()

            _rows, incomplete_rows, summary = self.mrart.discover_mrart(root, None)

        self.assertEqual(2, summary["subjects_discovered"])
        self.assertEqual(1, summary["complete_triplet_subjects_total"])
        self.assertEqual(1, summary["subjects_missing_one_or_more_acquisitions"])
        self.assertEqual(2, summary["missing_expected_acquisition_files_total"])
        self.assertEqual(
            {"standard": 0, "headmotion1": 1, "headmotion2": 1},
            summary["missing_acquisition_counts"],
        )
        self.assertEqual(
            ["sub-incomplete"],
            [row["subject_id"] for row in incomplete_rows],
        )


if __name__ == "__main__":
    unittest.main()
