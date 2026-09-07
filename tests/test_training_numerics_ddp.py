from __future__ import annotations

import copy
from datetime import timedelta
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reproducibility.numerics import TrainingNumerics, NUMERICAL_POLICY
from reproducibility.experiment import CHECKPOINT_SELECTION, validate_training_protocol
from reproducibility.core import ConflictError, CompatibilityError, validate_metadata


class TinyModel(torch.nn.Module):
    def __init__(self, mode="finite", rank=0, unused=False):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        if unused:
            self.unused = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("forward_count", torch.tensor(0))
        self.register_buffer("floating_buffer", torch.tensor(0.0))
        self.mode, self.rank = mode, rank
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.data_ptr())
        self.forward_count.add_(1)
        # Exercise all three RNGs. The replay must leave one forward's state.
        random.random()
        np.random.random()
        y = self.linear(x) * torch.rand_like(x)
        amp = torch.is_autocast_enabled(x.device.type)
        if self.mode == "fp32_exception" and not amp and self.rank == 0:
            raise RuntimeError("simulated FP32 forward exception")
        if (self.mode == "both" and self.rank == 0) or (
            self.mode in ("amp", "fp32_exception") and amp and self.rank == 0
        ):
            return y * float("nan")
        return y


def controller(model, root):
    return TrainingNumerics(
        model,
        device=torch.device("cpu"),
        run_dir=Path(root),
        amp_enabled=True,
        task={"model": "tiny", "regime": "augmented", "fold": 0},
    )


def one_step(numerics, x, optimizer, scaler, step=1):
    logits, fallback = numerics.forward(
        x,
        context={"epoch": 40, "step": step, "global_step": 98400 + step},
        batch={
            "subject_id": "sub-test",
            "sample_role": "clean_control",
            "patch_slot": 2,
        },
        scaler_scale=scaler.get_scale(),
    )
    norm = numerics.backward_step(
        logits.float().square().mean(),
        fallback=fallback,
        optimizer=optimizer,
        scaler=scaler,
        max_consecutive_overflows=16,
    )
    return fallback, norm


def ddp_worker(rank, rendezvous, root, mode, find_unused):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        model = TinyModel(
            mode
            if mode in ("amp", "both", "fp32_exception")
            else ("amp" if mode == "write_error" else "finite"),
            rank,
            unused=find_unused,
        )
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=find_unused
        )
        numerics = controller(ddp, root)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=0.001, eps=1e-4)
        scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1)
        if mode == "post_buffer":
            original_step = optimizer.step

            def corrupt_step(*args, **kwargs):
                original_step(*args, **kwargs)
                if rank == 1:
                    model.floating_buffer.fill_(float("nan"))

            optimizer.step = corrupt_step
        if mode == "write_error":
            if rank == 0:
                Path(root, "logs").write_text("block diagnostic directory")
            dist.barrier()
        if mode in ("parameter", "buffer") and rank == 1:
            with torch.no_grad():
                if mode == "parameter":
                    model.linear.weight.fill_(float("nan"))
                else:
                    model.floating_buffer.fill_(float("inf"))
        result = {"rank": rank}
        try:
            numerics.check_model_state()
            x = torch.ones(2, 4) * (rank + 1)
            result["fallback"], result["norm"] = one_step(
                numerics, x, optimizer, scaler
            )
            result["scale"] = scaler.get_scale()
            result["calls_after_first_batch"] = int(model.forward_count)
            model.mode = "finite"
            result["next_fallback"], _ = one_step(
                numerics, x, optimizer, scaler, step=2
            )
            weights = model.linear.weight.detach().clone()
            other = weights.clone()
            dist.broadcast(other, src=0)
            result["weights_equal"] = bool(torch.equal(weights, other))
            result["fallback_count"] = numerics.fallback_count
            numerics.save_summary(completed=True)
        except RuntimeError as exc:
            result["error"] = str(exc)
        Path(root, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class ProductionNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_finite_path_matches_original_update_and_scaler(self):
        with tempfile.TemporaryDirectory() as root:
            model = TinyModel()
            reference = copy.deepcopy(model)
            optimizers = [
                torch.optim.AdamW(m.parameters(), lr=0.001, eps=1e-4)
                for m in (model, reference)
            ]
            scalers = [
                torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1)
                for _ in range(2)
            ]
            x = torch.ones(2, 4)
            state = torch.get_rng_state()
            with torch.autocast("cpu"):
                logits = reference(x)
            scalers[1].scale(logits.float().square().mean()).backward()
            scalers[1].unscale_(optimizers[1])
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 12.0)
            scalers[1].step(optimizers[1])
            scalers[1].update()
            torch.set_rng_state(state)
            numerics = controller(model, root)
            fallback, norm = one_step(numerics, x, optimizers[0], scalers[0])
            self.assertFalse(fallback)
            self.assertTrue(np.isfinite(norm))
            self.assertEqual(numerics.fallback_count, 0)
            self.assertEqual(scalers[0].state_dict(), scalers[1].state_dict())
            for a, b in zip(model.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(list(Path(root).rglob("*.json")), [])

    def test_fallback_reuses_batch_rng_buffers_update_and_frozen_scaler(self):
        with tempfile.TemporaryDirectory() as root:
            model = TinyModel("amp")
            numerics = controller(model, root)
            x = torch.ones(2, 4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-4)
            scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=1)
            before_scaler = copy.deepcopy(scaler.state_dict())
            before_weights = model.linear.weight.detach().clone()
            initial = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            reference = copy.deepcopy(model)
            with torch.autocast("cpu"):
                reference(x)
            expected_rng = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
            )
            random.setstate(initial[0])
            np.random.set_state(initial[1])
            torch.set_rng_state(initial[2])
            fallback, _ = one_step(numerics, x, optimizer, scaler)
            self.assertTrue(fallback)
            self.assertEqual(model.inputs, [x.data_ptr(), x.data_ptr()])
            self.assertEqual(int(model.forward_count), 1)
            self.assertFalse(torch.equal(before_weights, model.linear.weight))
            self.assertEqual(scaler.state_dict(), before_scaler)
            self.assertEqual(random.getstate(), expected_rng[0])
            self.assertTrue(
                np.array_equal(np.random.get_state()[1], expected_rng[1][1])
            )
            self.assertEqual(np.random.get_state()[2:], expected_rng[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng[2]))
            numerics.save_summary(completed=True)
            self.assertEqual(
                json.loads(Path(root, "logs/numerical_summary.json").read_text())[
                    "fallback_count"
                ],
                1,
            )
            event = numerics.last_event
            self.assertEqual(event["nonfinite_amp_ranks"], [0])
            self.assertTrue(event["ranks"][0]["fp32_succeeded"])
            self.assertEqual(event["ranks"][0]["batch"]["patch_slot"], 2)

    def test_optimizer_corruption_is_caught_after_update(self):
        with tempfile.TemporaryDirectory() as root:
            model = TinyModel()
            numerics = controller(model, root)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            original_step = optimizer.step

            def corrupt(*args, **kwargs):
                original_step(*args, **kwargs)
                model.floating_buffer.fill_(float("nan"))

            scaler = torch.amp.GradScaler("cpu")
            with mock.patch.object(optimizer, "step", side_effect=corrupt):
                with self.assertRaisesRegex(RuntimeError, "nonfinite_model_state"):
                    one_step(numerics, torch.ones(2, 4), optimizer, scaler)
            self.assertTrue(
                list(Path(root, "logs").glob("numerics_nonfinite_model_state*"))
            )

    def test_gradient_overflow_keeps_existing_scaler_recovery_and_debug_probe(self):
        with tempfile.TemporaryDirectory() as root:
            model = TinyModel()
            model.linear.weight.register_hook(
                lambda gradient: torch.full_like(gradient, float("inf"))
            )
            numerics = controller(model, root)
            numerics.diagnostics = lambda: {"gradient_max": float("inf")}
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-4)
            scaler = torch.amp.GradScaler("cpu", init_scale=16)
            probe = mock.Mock(return_value={"logits_are_finite": True})
            before = model.linear.weight.detach().clone()
            logits, fallback = numerics.forward(
                torch.ones(2, 4), context={"step": 1}, batch={}, scaler_scale=16
            )
            norm = numerics.backward_step(
                logits.float().square().mean(),
                fallback=fallback,
                optimizer=optimizer,
                scaler=scaler,
                max_consecutive_overflows=8,
                overflow_probe=probe,
            )
            self.assertIsNone(norm)
            self.assertEqual(scaler.get_scale(), 8)
            self.assertTrue(torch.equal(before, model.linear.weight))
            probe.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "test_debug"):
                numerics.fail("test_debug")
            diagnostic = json.loads(
                next(Path(root, "logs").glob("*test_debug*")).read_text()
            )
            self.assertEqual(
                diagnostic["ranks"][0]["debug"]["after_unscale"]["gradient_max"], "inf"
            )

    def test_old_policy_rejected_and_checkpoint_metadata_requires_current_policy(self):
        definition = {
            "configuration": {
                "checkpoint_selection": CHECKPOINT_SELECTION,
                "numerical_policy": NUMERICAL_POLICY,
            }
        }
        validate_training_protocol(definition)
        old = copy.deepcopy(definition)
        del old["configuration"]["numerical_policy"]
        with self.assertRaisesRegex(ConflictError, "numerical_policy"):
            validate_training_protocol(old)
        with self.assertRaises(CompatibilityError):
            validate_metadata(
                {}, {"numerical_policy": NUMERICAL_POLICY}, context="old checkpoint"
            )

    def test_ddp_fallback_fatal_and_next_forward_both_reducer_modes(self):
        ctx = mp.get_context("spawn")
        for mode, unused in [
            ("amp", False),
            ("amp", True),
            ("both", False),
            ("parameter", False),
            ("buffer", True),
            ("post_buffer", False),
            ("write_error", False),
            ("fp32_exception", True),
        ]:
            with (
                self.subTest(mode=mode, find_unused=unused),
                tempfile.TemporaryDirectory() as root,
            ):
                workers = [
                    ctx.Process(
                        target=ddp_worker,
                        args=(rank, str(Path(root, "rendezvous")), root, mode, unused),
                    )
                    for rank in range(2)
                ]
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 45
                try:
                    for worker in workers:
                        worker.join(max(0, deadline - time.monotonic()))
                    self.assertTrue(
                        all(not worker.is_alive() for worker in workers), "DDP hang"
                    )
                    self.assertEqual([w.exitcode for w in workers], [0, 0])
                    results = [
                        json.loads(Path(root, f"rank-{r}.json").read_text())
                        for r in range(2)
                    ]
                    if mode == "amp":
                        for result in results:
                            self.assertNotIn("error", result)
                            self.assertTrue(result["fallback"])
                            self.assertFalse(result["next_fallback"])
                            self.assertTrue(result["weights_equal"])
                            self.assertEqual(result["scale"], 16)
                            self.assertEqual(result["fallback_count"], 1)
                            self.assertEqual(result["calls_after_first_batch"], 1)
                        event = json.loads(
                            Path(root, "logs/amp_fp32_fallbacks.jsonl").read_text()
                        )
                        self.assertEqual(event["nonfinite_amp_ranks"], [0])
                        self.assertEqual(len(event["ranks"]), 2)
                    elif mode == "write_error":
                        self.assertTrue(
                            all(
                                "diagnostic write failed" in result.get("error", "")
                                for result in results
                            )
                        )
                    else:
                        reason = (
                            "nonfinite_fp32_forward"
                            if mode in ("both", "fp32_exception")
                            else "nonfinite_model_state"
                        )
                        self.assertTrue(
                            all(
                                reason in result.get("error", "") for result in results
                            ),
                            results,
                        )
                        self.assertTrue(list(Path(root, "logs").glob(f"*{reason}*")))
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                        worker.join()


if __name__ == "__main__":
    unittest.main()
