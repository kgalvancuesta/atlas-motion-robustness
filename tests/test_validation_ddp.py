"""Exercise the production validation block with short CPU process-group deadlines."""
from __future__ import annotations

import ast
from datetime import timedelta
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_common as training
from reproducibility.numerics import TrainingNumerics


def production_blocks():
    # Execute the actual inline production code, without dataset setup or training.
    tree = ast.parse(inspect.getsource(training.run_training))
    setup = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "corrected_enabled and distributed")
    epoch = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "epoch")
    start = next(i for i, n in enumerate(epoch.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "dev_dice")
    end = next(i for i, n in enumerate(epoch.body) if isinstance(n, ast.If)
               and any(isinstance(a, ast.Name) and a.id == "local_rng_state"
                       for a in ast.walk(n)))
    return [compile(ast.Module(body=nodes, type_ignores=[]), training.__file__, "exec")
            for nodes in ([setup], epoch.body[start:end])]


class ValidationLoader:
    def __init__(self, rank, mode):
        self.rank, self.mode, self.calls = rank, mode, 0

    def __len__(self):
        return 1

    def __iter__(self):
        self.calls += 1
        assert self.rank == 0, "Only rank zero may validate"
        time.sleep(4)  # Exceeds the default training group's two-second deadline.
        if self.mode == "error":
            raise RuntimeError("simulated validation failure")
        yield torch.ones(1, 1, 2, 2, 2), torch.ones(1, 1, 2, 2, 2), ["sub-clean"]


def validation_worker(rank, rendezvous, root, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=2))
    group = None
    try:
        model = torch.nn.Conv3d(1, 1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        model = torch.nn.parallel.DistributedDataParallel(model)
        loader = ValidationLoader(rank, mode)
        numerics = TrainingNumerics(model, device=torch.device("cpu"), run_dir=Path(root),
                                    task={"model": "tiny"}, amp_enabled=False)
        env = dict(vars(training), distributed=True, corrected_enabled=True,
                   validation_control_group=None, args=SimpleNamespace(val_interval=2, patience=1),
                   epoch=1 if mode == "not_due" else 2, run_dir=Path(root), model=model,
                   dev_loader=loader, device=torch.device("cpu"), patch_size=(2, 2, 2),
                   loss_fn=torch.nn.MSELoss(), best_dice=1.0 if mode == "tie" else 0.0,
                   epochs_without_improvement=0, main_process=rank == 0, numerics=numerics,
                   corrected_checkpoint_metadata={"experiment_id": "test-validation"},
                   log_path=Path(root, "logs/train_log.csv"), running=0.0, train_loader=[None])
        setup, validate = production_blocks()
        exec(setup, env)
        group = env["validation_control_group"]
        assert group is not dist.group.WORLD and dist.get_backend(group) == "gloo"
        # Model a completed training update before entering validation.
        model(torch.ones(1, 1, 2, 2, 2)).sum().backward()
        original_broadcast = dist.broadcast_object_list
        control_calls = []

        def control_broadcast(*args, **kwargs):
            if kwargs.get("group") is group:
                control_calls.append(True)
            return original_broadcast(*args, **kwargs)

        result = {}
        try:
            with mock.patch.object(dist, "broadcast_object_list", side_effect=control_broadcast):
                exec(validate, env)
            # The next DDP forward/backward must still work after the long wait.
            model(torch.ones(1, 1, 2, 2, 2)).sum().backward()
            result["continued"] = True
        except RuntimeError as exc:
            result["error"] = str(exc)
        result.update(calls=loader.calls, control_calls=len(control_calls),
                      validation_error=env["validation_error"], dice=env["dev_dice"],
                      loss=env["dev_loss"], best=env["best_dice"], stop=env["should_stop"],
                      patience=env["epochs_without_improvement"])
        Path(root, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


class ValidationDDPTests(unittest.TestCase):
    def test_long_validation_success_exception_tie_and_not_due(self):
        ctx = mp.get_context("spawn")
        for mode in ("success", "error", "tie", "not_due"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                Path(root, "checkpoints").mkdir()
                Path(root, "logs").mkdir()
                workers = [ctx.Process(target=validation_worker,
                           args=(r, str(Path(root, "rendezvous")), root, mode)) for r in range(2)]
                for worker in workers:
                    worker.start()
                try:
                    deadline = time.monotonic() + 45
                    for worker in workers:
                        worker.join(max(0, deadline - time.monotonic()))
                    self.assertEqual([w.exitcode for w in workers], [0, 0], "DDP failure or hang")
                    results = [json.loads(Path(root, f"rank-{r}.json").read_text()) for r in range(2)]
                    self.assertEqual([r["calls"] for r in results], [int(mode != "not_due"), 0])
                    self.assertEqual([r["control_calls"] for r in results], [int(mode != "not_due")] * 2)
                    if mode == "error":
                        for result in results:
                            self.assertIn("validation_failed", result["error"])
                            self.assertEqual(result["validation_error"],
                                             "RuntimeError: simulated validation failure")
                        event = json.loads(next(Path(root, "logs").glob("numerics_validation_failed*.json")).read_text())
                        self.assertEqual([r["error"] for r in event["ranks"]],
                                         ["RuntimeError: simulated validation failure"] * 2)
                    else:
                        self.assertTrue(all(r["continued"] for r in results))
                        self.assertEqual([r["stop"] for r in results], [mode == "tie"] * 2)
                        if mode != "not_due":
                            self.assertEqual(results[0]["dice"], 1.0)
                            self.assertEqual(results[0]["loss"], 0.0)
                            self.assertEqual(results[0]["best"], 1.0)
                            self.assertEqual(results[0]["patience"], int(mode == "tie"))
                    best = Path(root, "checkpoints/best.pt")
                    self.assertEqual(best.exists(), mode == "success")
                    if best.exists():
                        payload = torch.load(best, weights_only=True)
                        self.assertEqual(payload["epoch"], 2)
                        self.assertEqual(payload["best_dice"], 1.0)
                        self.assertEqual(payload["corrected_experiment_metadata"],
                                         {"experiment_id": "test-validation"})
                        self.assertTrue(torch.equal(payload["model"]["weight"], torch.ones(1, 1, 1, 1, 1)))
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                        worker.join()


if __name__ == "__main__":
    unittest.main()
