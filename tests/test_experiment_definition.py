from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nb
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

from reproducibility.core import ConflictError  # noqa: E402
from reproducibility.experiment import load_challenge, prepare_experiment  # noqa: E402


@unittest.skipUnless(TORCHIO_AVAILABLE, "TorchIO is required")
class ExperimentDefinitionTests(unittest.TestCase):
    def _write_image(self, path: Path, offset: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.zeros((8, 9, 10), dtype=np.float32)
        data[1:7, 1:8, 1:9] = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8) + 1 + offset
        nb.save(nb.Nifti1Image(data, np.eye(4)), str(path))

    def _fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "data"
        subjects = [f"sub-test{i}" for i in range(5)]
        for index, subject in enumerate(subjects):
            anat = data_root / "train" / "derivatives" / "ATLAS" / subject / "ses-1" / "anat"
            self._write_image(anat / f"{subject}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz", index)
            mask = np.zeros((8, 9, 10), dtype=np.uint8)
            mask[3:5, 4:6, 5:7] = 1
            nb.save(
                nb.Nifti1Image(mask, np.eye(4)),
                str(anat / f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"),
            )
        folds = []
        for fold, subject in enumerate(subjects):
            validation = subjects[(fold + 1) % 5]
            training = [value for value in subjects if value not in {subject, validation}]
            folds.append(
                {"fold_index": fold, "train_ids": training, "val_ids": [validation], "test_ids": [subject]}
            )
        split_path = root / "splits.json"
        split_path.write_text(json.dumps({"metadata": {"test_fixture": True}, "folds": folds}), encoding="utf-8")
        mrart_subject = "sub-mrart"
        for index, acquisition in enumerate(("standard", "headmotion1", "headmotion2")):
            self._write_image(
                data_root / "mr-art" / mrart_subject / "anat" / f"{mrart_subject}_acq-{acquisition}_T1w.nii.gz",
                100 + index,
            )
        return data_root, split_path

    def test_prepare_resume_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            data_root, split_path = self._fixture(temporary_root)
            output_root = temporary_root / "outputs"
            kwargs = {
                "repo_root": REPO_ROOT,
                "output_root": output_root,
                "experiment_id": "test-definition",
                "data_root": data_root,
                "mrart_root": data_root / "mr-art",
                "split_path": split_path,
                "normalization_mode": "artifact_then_normalize",
                "models": ["base_cnn"],
                "regimes": ["standard", "augmented"],
                "folds": [0, 1, 2, 3, 4],
                "epochs": 1,
                "replicate_count": 1,
                "dataset_authority": "test",
            }
            root, definition, status = prepare_experiment(**kwargs)
            self.assertEqual(status, "created")
            self.assertEqual(definition["configuration"]["batch_size_per_gpu"], 1)
            self.assertEqual(len(definition["dataset"]["source_inventory"]), 5)
            self.assertEqual(len(definition["dataset"]["mrart"]["sources"]), 3)
            manifest = load_challenge(root, definition)
            self.assertEqual(len(manifest["recipes"]), 5)
            _root, repeated, repeated_status = prepare_experiment(**kwargs)
            self.assertEqual(repeated_status, "skipped-compatible")
            self.assertEqual(repeated["definition_hash"], definition["definition_hash"])
            with self.assertRaises(ConflictError):
                prepare_experiment(**{**kwargs, "normalization_mode": "normalize_then_artifact"})

    def test_prepare_supports_a_filtered_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            data_root, split_path = self._fixture(temporary_root)
            root, definition, status = prepare_experiment(
                repo_root=REPO_ROOT,
                output_root=temporary_root / "outputs",
                experiment_id="test-single-fold",
                data_root=data_root,
                mrart_root=data_root / "mr-art",
                split_path=split_path,
                normalization_mode="artifact_then_normalize",
                models=["base_cnn"],
                regimes=["standard"],
                folds=[0],
                epochs=1,
                replicate_count=1,
                dataset_authority="test",
            )
            self.assertEqual(status, "created")
            self.assertEqual(definition["configuration"]["folds"], [0])
            self.assertEqual(len(definition["dataset"]["source_inventory"]), 5)
            self.assertEqual(len(load_challenge(root, definition)["recipes"]), 1)


if __name__ == "__main__":
    unittest.main()
