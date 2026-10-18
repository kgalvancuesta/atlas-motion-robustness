"""Check voxel-count recovery and failure/empty-prediction denominators."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / (
    "scripts/supporting_experiments/statistical_inference/scripts/analyze_lesion_failures.py"
)
SPEC = importlib.util.spec_from_file_location("lesion_failures", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def case(subject, predicted, reference, overlap):
    return {
        "subject_id": subject, "fold": "0", "architecture": "Synthetic",
        "training_condition": "Augmented", "evaluation_condition": "artifact",
        "lesion_size_bin": "1", "lesion_voxel_range": "[13, 1159)",
        "dice": str(2 * overlap / (predicted + reference)),
        "predicted_voxel_count": str(predicted), "ground_truth_voxel_count": str(reference),
        "source_metric_mode": "deterministic_experiment_binary_dice",
    }


class LesionFailureTests(unittest.TestCase):
    def test_partial_overlap_empty_and_nonempty_misses(self):
        derived, summary = analysis.analyze([
            case("partial", 30, 100, 20), case("empty", 0, 100, 0),
            case("miss", 40, 100, 0), case("perfect", 100, 100, 100),
        ])
        self.assertEqual((derived[0]["tp"], derived[0]["fp"], derived[0]["fn"]), (20, 10, 80))
        self.assertEqual(derived[0]["voxel_fnr"], 0.8)
        self.assertEqual(derived[0]["voxel_ppv"], 2 / 3)
        self.assertIsNone(derived[1]["voxel_ppv"])
        self.assertEqual(derived[2]["voxel_ppv"], 0)
        self.assertAlmostEqual(summary[0]["macro_voxel_fnr"], 0.7)
        self.assertAlmostEqual(summary[0]["macro_voxel_ppv_nonempty"], 5 / 9)
        failed = summary[1]
        self.assertEqual((failed["n"], failed["zero_overlap_cases"], failed["empty_predictions"]), (2, 2, 1))
        self.assertEqual((failed["macro_voxel_fnr"], failed["macro_voxel_ppv_nonempty"], failed["ppv_defined_n"]), (1, 0, 1))

    def test_no_failures_has_undefined_failure_means(self):
        _, summary = analysis.analyze([case("perfect", 100, 100, 100)])
        self.assertEqual(summary[1]["n"], 0)
        self.assertIsNone(summary[1]["macro_voxel_fnr"])
        self.assertIsNone(summary[1]["macro_voxel_ppv_nonempty"])

    def test_empty_predictions_have_fnr_one_and_undefined_ppv(self):
        _, summary = analysis.analyze([case("empty", 0, 100, 0)])
        for row in summary:
            self.assertEqual(row["macro_voxel_fnr"], 1)
            self.assertIsNone(row["macro_voxel_ppv_nonempty"])
            self.assertEqual(row["ppv_defined_n"], 0)

    def test_failure_threshold_is_strictly_below_point_one(self):
        derived, summary = analysis.analyze([case("boundary", 100, 100, 10)])
        self.assertEqual(derived[0]["dice"], 0.1)
        self.assertEqual(derived[0]["catastrophic_failure"], 0)
        self.assertEqual(summary[1]["n"], 0)

    def test_invalid_metrics_and_duplicate_cases_are_rejected(self):
        for changes in (
            {"dice": "0.123"}, {"dice": "nan"}, {"dice": "inf"},
            {"dice": "1.1"}, {"ground_truth_voxel_count": "0"},
            {"predicted_voxel_count": "-1"}, {"source_metric_mode": "legacy"},
            {"predicted_voxel_count": "0", "dice": "0.2"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analysis.analyze([dict(case("invalid", 30, 100, 20), **changes)])
        row = case("duplicate", 30, 100, 20)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analysis.analyze([row, row])


if __name__ == "__main__":
    unittest.main()
