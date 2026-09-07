from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from reproducibility.numerics import (  # noqa: E402
    classify_failure,
    model_state_statistics,
    run_forward_precision_probes,
    tensor_statistics,
)
from train_common import PatchDataset, Sample  # noqa: E402


class _AutocastNaN(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled(x.device.type):
            return torch.full_like(x, float("nan"))
        return x + 1.0


class NumericsProbeTests(unittest.TestCase):
    def test_corrected_gradient_probe_restores_rng_buffers_and_clears_gradients(self):
        from train_common import run_fp32_probe

        model = torch.nn.Sequential(
            torch.nn.BatchNorm3d(1), torch.nn.Dropout3d(0.3), torch.nn.Conv3d(1, 1, 1)
        )
        x = torch.arange(16.0).reshape(2, 1, 2, 2, 2)
        before_rng = torch.get_rng_state()
        before_state = {key: value.clone() for key, value in model.state_dict().items()}
        probe = run_fp32_probe(
            model,
            torch.nn.MSELoss(),
            x,
            torch.zeros_like(x),
            accum_steps=1,
            top_k=3,
            collective_safe=True,
        )
        self.assertTrue(probe["logits_are_finite"])
        self.assertTrue(probe["loss_is_finite"])
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(
            all(
                torch.equal(before_state[key], value)
                for key, value in model.state_dict().items()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_tensor_statistics_are_json_safe_and_expose_nonfinite_counts(self) -> None:
        stats = tensor_statistics(
            torch.tensor([1.0, -2.0, float("nan"), float("inf"), -float("inf")])
        )
        self.assertFalse(stats["finite"])
        self.assertEqual(stats["finite_count"], 2)
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["positive_inf_count"], 1)
        self.assertEqual(stats["negative_inf_count"], 1)
        self.assertEqual(stats["min"], -2.0)
        self.assertEqual(stats["max"], 1.0)
        json.dumps(stats, allow_nan=False)

    def test_amp_failure_is_compared_with_fp32_and_traced(self) -> None:
        model = torch.nn.Sequential(_AutocastNaN())
        x = torch.ones(1, 1, 2, 2, 2)
        probes = run_forward_precision_probes(model, x, amp_enabled=True)
        state = model_state_statistics(model)
        input_stats = tensor_statistics(x)

        self.assertFalse(probes["amp"]["logits_are_finite"])
        self.assertTrue(probes["fp32"]["logits_are_finite"])
        first = probes["amp_activation_replay"]["activations"][
            "first_nonfinite_activation"
        ]
        self.assertEqual(first["module"], "0")
        self.assertEqual(
            classify_failure(input_stats, state, probes),
            "autocast_only_forward_failure",
        )
        json.dumps(probes, allow_nan=False)

    def test_nonfinite_model_state_takes_priority(self) -> None:
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        state = model_state_statistics(model)
        probes = {
            "amp": {"logits_are_finite": False},
            "fp32": {"logits_are_finite": False},
        }
        self.assertFalse(state["parameters_are_finite"])
        self.assertEqual(
            classify_failure(tensor_statistics(torch.ones(1, 2)), state, probes),
            "nonfinite_model_state",
        )


class PatchDescriptionTests(unittest.TestCase):
    def test_clean_patch_description_matches_dataset_index(self) -> None:
        volume = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8) + 1.0
        mask = np.zeros_like(volume)
        mask[2:4, 3:5, 4:6] = 1.0
        sample = Sample("sub-test", Path("image.nii.gz"), Path("mask.nii.gz"))
        context = {
            "experiment_id": "test",
            "global_seed": 9001,
            "fold": 0,
            "normalization_mode": "artifact_then_normalize",
            "memory_mode": "low",
            "source_sha256": {"sub-test": "unused"},
        }
        dataset = PatchDataset(
            [sample],
            patch_size=(4, 4, 4),
            patches_per_volume=4,
            lesion_prob=0.7,
            experiment_context=context,
        )
        dataset.set_epoch(40)

        def fake_load(path: Path):
            value = mask if path == sample.mask_path else volume
            return value.copy(), np.eye(4), None

        with mock.patch("train_common.load_nifti", side_effect=fake_load):
            description = dataset.describe_index(2)
            x, y = dataset[2]

        self.assertEqual(description["dataset_index"], 2)
        self.assertEqual(description["subject_id"], "sub-test")
        self.assertEqual(description["sample_role"], "clean_control")
        self.assertEqual(description["patch_slot"], 2)
        self.assertIsNone(description["augmentation_recipe"])
        self.assertEqual(tuple(x.shape), (1, 4, 4, 4))
        self.assertEqual(tuple(y.shape), (1, 4, 4, 4))


class ReplayVerificationTests(unittest.TestCase):
    def test_source_run_and_cache_stay_read_only_and_nonreproduction_is_not_a_pass(
        self,
    ):
        import hashlib
        import tempfile
        import test_experiment_definition as fixtures
        import train_common
        import diagnose_training_numerics
        from reproducibility.experiment import prepare_experiment
        from reproducibility.numerics import TrainingNumerics

        class TinySegmentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv3d(1, 1, 1)
                self.fail_amp = False

            def forward(self, x):
                output = self.layer(x)
                if self.fail_amp and torch.is_autocast_enabled(x.device.type):
                    return output * float("nan")
                return output

        class EpochBoundaryStop(Exception):
            pass

        def fingerprint(root):
            return {
                str(p.relative_to(root)): (
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                    p.stat().st_mtime_ns,
                )
                for p in root.rglob("*")
                if p.is_file()
            }

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, split = fixtures.ExperimentDefinitionTests()._fixture(root)
            definition_root, definition, _ = prepare_experiment(
                repo_root=REPO_ROOT,
                output_root=root / "definitions",
                experiment_id="test-replay",
                data_root=data,
                mrart_root=data / "mr-art",
                split_path=split,
                normalization_mode="artifact_then_normalize",
                models=["base_cnn"],
                regimes=["standard"],
                folds=[0],
                epochs=2,
                replicate_count=1,
                dataset_authority="test",
            )
            # Tiny CPU-only integration fixture; no authoritative definition is modified.
            definition["model_configurations"]["base_cnn"]["patch_size"] = [4, 4, 4]
            definition["configuration"]["nproc_per_node"] = 1
            definition_file = definition_root / "definition/experiment.json"
            definition_file.write_text(json.dumps(definition))
            run_dir, cache = root / "old-run", root / "old-cache"
            args = train_common.build_argument_parser(
                description="test", default_loss="baseline"
            ).parse_args(
                [
                    "--data_root",
                    str(data),
                    "--splits_json",
                    str(split),
                    "--cv_fold",
                    "0",
                    "--run_dir",
                    str(run_dir),
                    "--max_epochs",
                    "2",
                    "--batch_size",
                    "1",
                    "--patch_size",
                    "4",
                    "4",
                    "4",
                    "--num_workers",
                    "0",
                    "--amp",
                    "--experiment_definition",
                    str(definition_file),
                    "--experiment_fold",
                    "0",
                    "--experiment_training_regime",
                    "standard",
                    "--normalization_mode",
                    "artifact_then_normalize",
                    "--experiment_memory_mode",
                    "high",
                    "--experiment_cache_root",
                    str(cache),
                ]
            )
            original_save = train_common._atomic_torch_save

            def stop_after_epoch(payload, path):
                original_save(payload, path)
                raise EpochBoundaryStop()

            with (
                mock.patch("train_common.get_device", return_value=torch.device("cpu")),
                mock.patch(
                    "train_common.sliding_window_inference",
                    side_effect=lambda x, **kw: kw["predictor"](x),
                ),
                mock.patch(
                    "train_common._atomic_torch_save", side_effect=stop_after_epoch
                ),
            ):
                with self.assertRaises(EpochBoundaryStop):
                    train_common.run_training(
                        args, model_builder=TinySegmentation, model_name="base_cnn"
                    )
            # Make a legacy-policy checkpoint/config as in the superseded v2 run.
            for key in ("checkpoint_selection", "numerical_policy"):
                definition["configuration"].pop(key)
            definition_file.write_text(json.dumps(definition))
            config_file = run_dir / "config/train_config.json"
            config = json.loads(config_file.read_text())
            checkpoint_file = run_dir / "checkpoints/last.pt"
            checkpoint = torch.load(checkpoint_file, weights_only=False)
            for key in ("checkpoint_selection", "numerical_policy"):
                config["immutable_training_request"].pop(key)
                config["corrected_experiment_metadata"].pop(key)
                checkpoint["corrected_experiment_metadata"].pop(key)
            config_file.write_text(json.dumps(config))
            torch.save(checkpoint, checkpoint_file)
            before_run, before_cache = fingerprint(run_dir), fingerprint(cache)
            original_forward = TrainingNumerics.forward
            for reproduce in (False, True):
                output = root / f"replay-{reproduce}.json"

                def forced_cpu_autocast(controller, x, **kwargs):
                    controller.amp_enabled = True
                    controller.model.fail_amp = (
                        reproduce and kwargs["context"]["step"] == 2
                    )
                    return original_forward(controller, x, **kwargs)

                with (
                    mock.patch(
                        "sys.argv",
                        [
                            "diagnose_training_numerics.py",
                            "--run-dir",
                            str(run_dir),
                            "--epoch",
                            "2",
                            "--step",
                            "2",
                            "--output",
                            str(output),
                            "--verify-fix",
                        ],
                    ),
                    mock.patch(
                        "diagnose_training_numerics.build_base_model",
                        side_effect=TinySegmentation,
                    ),
                    mock.patch(
                        "train_common.get_device", return_value=torch.device("cpu")
                    ),
                    mock.patch.object(TrainingNumerics, "forward", forced_cpu_autocast),
                ):
                    result = diagnose_training_numerics.main()
                self.assertEqual(result, 0 if reproduce else 2)
                report = json.loads(output.read_text())
                self.assertEqual(
                    report["status"],
                    "passed"
                    if reproduce
                    else "inconclusive_failure_not_reproduced_exactly",
                )
                self.assertTrue(report["subsequent_update_completed"])
                self.assertEqual(fingerprint(run_dir), before_run)
                self.assertEqual(fingerprint(cache), before_cache)
            # The old policy must still be rejected by normal training.
            with self.assertRaisesRegex(
                RuntimeError, "fresh authoritative experiment ID"
            ):
                train_common.run_training(
                    args, model_builder=TinySegmentation, model_name="base_cnn"
                )

    def test_read_only_cache_never_builds_missing_entries(self):
        import tempfile
        from reproducibility.challenge import load_or_create_validated_cache

        with tempfile.TemporaryDirectory() as root:
            builder = mock.Mock(side_effect=AssertionError("must not build"))
            with self.assertRaisesRegex(RuntimeError, "Missing/incomplete"):
                load_or_create_validated_cache(
                    cache_root=Path(root),
                    challenge_id="old",
                    normalization_mode="artifact_then_normalize",
                    fold=0,
                    replicate_id=0,
                    subject_id="sub-test",
                    source_sha256="source",
                    recipe_id="clean",
                    builder=builder,
                    read_only=True,
                )
            builder.assert_not_called()
            self.assertEqual(list(Path(root).rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
