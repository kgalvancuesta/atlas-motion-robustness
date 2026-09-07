"""Collective training fail-safes and read-only numerical probes."""

from __future__ import annotations

import random
import json
import os
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist

NUMERICAL_POLICY = {
    "amp": True,
    "nonfinite_amp_forward": "retry_same_batch_fp32",
    "nonfinite_fp32_forward": "fail",
    "nonfinite_model_state": "fail",
    "fallback_scaler": "unscaled_fp32_update_freeze_scale_and_growth_tracker",
    "fallback_rng": "replay_pre_forward_restore_post_amp",
    "fallback_buffers": "restore_pre_forward_before_replay",
    "collective_decisions": True,
}


def collective_all(local_ok: bool | torch.Tensor, device: torch.device) -> bool:
    """One shared decision before any rank can branch or raise."""
    status = (
        torch.as_tensor(local_ok, dtype=torch.int32, device=device).detach().clone()
    )
    if dist.is_initialized():
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
    return bool(status.item())


def floating_state_is_finite(
    model: torch.nn.Module, device: torch.device
) -> torch.Tensor:
    # Device-side per-tensor reductions; one host synchronization at the collective.
    flags = [
        torch.isfinite(value).all()
        for value in (*model.parameters(), *model.buffers())
        if value.is_floating_point() or value.is_complex()
    ]
    return (
        torch.stack(flags).all()
        if flags
        else torch.ones((), dtype=torch.bool, device=device)
    )


def _capture_forward_rng(device: torch.device) -> tuple:
    return random.getstate(), np.random.get_state(), _capture_local_torch_rng(device)


def _restore_forward_rng(state: tuple, device: torch.device) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    _restore_local_torch_rng(state[2], device)


def _safe_debug(value: Any) -> Any:
    """Keep legacy nonfinite diagnostic magnitudes explicit and JSON-safe."""
    if isinstance(value, dict):
        return {key: _safe_debug(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_debug(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class TrainingNumerics:
    """Production corrected path. Corrected experiments fix accumulation_steps=1."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        run_dir: Path,
        task: dict[str, Any],
        amp_enabled: bool,
        diagnostics: Callable[[], dict] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.run_dir = run_dir
        self.task = task
        self.amp_enabled = amp_enabled
        self.fallback_count = 0
        self.consecutive_overflows = 0
        self.context: dict[str, Any] = {}
        self.batch: dict[str, Any] = {}
        self.x: torch.Tensor | None = None
        self.last_event: dict[str, Any] | None = None
        self.diagnostics = diagnostics
        self.step_debug: dict[str, Any] = {}

    def _write(
        self,
        filename: str,
        payload: dict,
        *,
        replace: bool = False,
        append: bool = False,
    ) -> None:
        from .core import write_json_atomic_replace, write_json_new

        error = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                path = self.run_dir / "logs" / filename
                if append:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload, allow_nan=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                else:
                    writer = write_json_atomic_replace if replace else write_json_new
                    writer(path, payload)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            status = [error]
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error:
            raise RuntimeError(f"Collective numerical diagnostic write failed: {error}")

    def _event(self, reason: str, local: dict[str, Any]) -> dict:
        from .core import utc_now

        record = {
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "batch": self.batch,
            "debug": _safe_debug(self.step_debug),
            **local,
        }
        ranks: list[Any] = [record]
        if dist.is_initialized():
            ranks = [None] * dist.get_world_size()
            dist.all_gather_object(ranks, record)
        event = {
            **self.task,
            **self.context,
            "reason": reason,
            "timestamp": utc_now(),
            "fallback_count": self.fallback_count,
            "ranks": ranks,
        }
        if reason == "amp_to_fp32_fallback":
            event["nonfinite_amp_ranks"] = [
                r["rank"] for r in ranks if not r["amp_logits"]["finite"]
            ]
        # Only rank 0 writes; its timestamp names the event. Every rank participates
        # in the write-status broadcast even if the filesystem write fails.
        stamp = event["timestamp"].replace(":", "").replace(".", "")
        if reason == "amp_to_fp32_fallback":
            # One journal per task bounds inode usage even for frequent fallbacks.
            self._write("amp_fp32_fallbacks.jsonl", event, append=True)
        else:
            self._write(
                f"numerics_{reason}_{self.context.get('global_step', 0)}_{stamp}.json",
                event,
            )
        self.last_event = event
        return event

    def fail(self, reason: str, local: dict[str, Any] | None = None) -> None:
        # Call only after a collective decision (or unconditionally on every rank).
        self._event(
            reason,
            {
                "input": tensor_statistics(self.x) if self.x is not None else None,
                "model_state": model_state_statistics(self.model),
                **(local or {}),
            },
        )
        raise RuntimeError(
            f"Collective numerical failure: {reason}; diagnostics in {self.run_dir / 'logs'}"
        )

    def check_model_state(self) -> None:
        if not collective_all(
            floating_state_is_finite(self.model, self.device), self.device
        ):
            self.fail("nonfinite_model_state")

    def forward(
        self, x: torch.Tensor, *, context: dict, batch: dict, scaler_scale: float | None
    ) -> tuple[torch.Tensor, bool]:
        self.x, self.context, self.batch = (
            x,
            {**context, "amp_scaler_scale": scaler_scale},
            batch,
        )
        self.step_debug = {}
        initial_rng = _capture_forward_rng(self.device) if self.amp_enabled else None
        # Stateful normalization must observe this batch once, even after replay.
        buffers = (
            [(value, value.detach().clone()) for value in self.model.buffers()]
            if self.amp_enabled
            else []
        )
        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            logits = self.model(x)
        if collective_all(torch.isfinite(logits).all(), self.device):
            return logits, False
        if not self.amp_enabled:
            self.fail(
                "nonfinite_fp32_forward", {"fp32_logits": tensor_statistics(logits)}
            )

        if self.diagnostics is not None:
            self.step_debug["discarded_amp_forward"] = self.diagnostics()
        amp_stats = tensor_statistics(logits)
        del logits  # No backward or scaler operation may use this graph.
        post_amp_rng = _capture_forward_rng(self.device)
        _restore_forward_rng(initial_rng, self.device)
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)
        fp32_error = None
        logits = None
        try:
            with torch.autocast(device_type=self.device.type, enabled=False):
                logits = self.model(x)
        except Exception as exc:
            fp32_error = f"{type(exc).__name__}: {exc}"
        finally:
            # Dropout uses the same pre-forward stream; later batches see exactly
            # the RNG state left by a single AMP forward, including Python/NumPy.
            _restore_forward_rng(post_amp_rng, self.device)
        succeeded = collective_all(
            torch.isfinite(logits).all() if logits is not None else False, self.device
        )
        fp32_stats = (
            tensor_statistics(logits)
            if logits is not None
            else {"finite": False, "error": fp32_error}
        )
        self.fallback_count += 1
        self._event(
            "amp_to_fp32_fallback",
            {
                "input": tensor_statistics(x),
                "amp_logits": amp_stats,
                "fp32_logits": fp32_stats,
                "fp32_succeeded": succeeded,
            },
        )
        if not succeeded:
            self.fail(
                "nonfinite_fp32_forward",
                {"amp_logits": amp_stats, "fp32_logits": fp32_stats},
            )
        return logits, True

    def backward_step(
        self,
        loss: torch.Tensor,
        *,
        fallback: bool,
        optimizer,
        scaler,
        max_consecutive_overflows: int,
        overflow_probe: Callable[[], dict] | None = None,
    ) -> float | None:
        if not collective_all(torch.isfinite(loss).all(), self.device):
            self.fail("nonfinite_loss", {"loss": tensor_statistics(loss)})
        scaled = self.amp_enabled and not fallback
        if scaled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if self.diagnostics is not None:
            self.step_debug["after_unscale"] = self.diagnostics()
        # Same clipping and optimizer ordering as the original finite AMP path.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12.0)
        if not collective_all(torch.isfinite(grad_norm), self.device):
            if not scaled:
                self.fail(
                    "nonfinite_grad_norm", {"grad_norm": tensor_statistics(grad_norm)}
                )
            # unscale_ recorded the overflow before clipping. Guard against a
            # rank-specific skip/update or norm overflow with finite gradients.
            found_inf = sum(
                value for value in scaler._found_inf_per_device(optimizer).values()
            )
            if not collective_all(found_inf > 0, self.device):
                self.fail("nonfinite_grad_norm_without_collective_scaler_overflow")
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not collective_all(scaler.get_scale() < scale_before, self.device):
                self.fail("amp_overflow_not_recovered")
            self.consecutive_overflows += 1
            if overflow_probe is not None:
                self.step_debug["fp32_probe"] = overflow_probe()
            if self.diagnostics is not None and (
                not dist.is_initialized() or dist.get_rank() == 0
            ):
                print(
                    f"AMP overflow recovered at epoch={self.context.get('epoch')}, "
                    f"step={self.context.get('step')}, global_step={self.context.get('global_step')}, "
                    f"scale {scale_before} -> {scaler.get_scale()}",
                    flush=True,
                )
            if self.consecutive_overflows > max_consecutive_overflows:
                self.fail("persistent_amp_overflow")
            self.check_model_state()
            return None
        self.consecutive_overflows = 0
        if scaled:
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 fallback bypasses GradScaler entirely: both scale and growth
            # tracker remain unchanged. The discarded AMP forward never touched it.
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.check_model_state()
        return float(grad_norm.item())

    def save_summary(self, *, completed: bool) -> None:
        # The journal also retains attempts from an interrupted epoch. Keep the
        # completed trajectory's checkpoint count separate from all logged events.
        logged_events = 0
        error = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                path = self.run_dir / "logs" / "amp_fp32_fallbacks.jsonl"
                if path.exists():
                    with path.open(encoding="utf-8") as handle:
                        logged_events = sum(1 for line in handle if line.strip())
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            status = [logged_events, error]
            dist.broadcast_object_list(status, src=0)
            logged_events, error = status
        if error:
            raise RuntimeError(f"Collective fallback journal read failed: {error}")
        self._write(
            "numerical_summary.json",
            {
                **self.task,
                "numerical_policy": NUMERICAL_POLICY,
                "fallback_count": self.fallback_count,
                "logged_fallback_events": logged_events,
                "completed": completed,
            },
            replace=True,
        )


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe statistics without hiding non-finite values."""
    value = tensor.detach()
    numel = int(value.numel())
    if value.is_floating_point() or value.is_complex():
        finite = torch.isfinite(value)
        nan_count = int(torch.isnan(value).sum().item())
        positive_inf_count = int(torch.isposinf(value).sum().item())
        negative_inf_count = int(torch.isneginf(value).sum().item())
    else:
        finite = torch.ones_like(value, dtype=torch.bool)
        nan_count = 0
        positive_inf_count = 0
        negative_inf_count = 0

    finite_count = int(finite.sum().item())
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": numel,
        "finite": finite_count == numel,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "positive_inf_count": positive_inf_count,
        "negative_inf_count": negative_inf_count,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "abs_max": None,
    }
    if finite_count:
        finite_values = value[finite].float()
        std, mean = torch.std_mean(finite_values, correction=0)
        result.update(
            {
                "min": float(finite_values.amin().item()),
                "max": float(finite_values.amax().item()),
                "mean": float(mean.item()),
                "std": float(std.item()),
                "abs_max": float(finite_values.abs().amax().item()),
            }
        )
    return result


def model_state_statistics(
    model: torch.nn.Module, *, top_k: int = 12
) -> dict[str, Any]:
    """Summarize parameters and buffers immediately before a forward probe."""
    entries: list[dict[str, Any]] = []
    for kind, named_values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in named_values:
            entries.append({"kind": kind, "name": name, **tensor_statistics(value)})

    parameters = [entry for entry in entries if entry["kind"] == "parameter"]
    buffers = [entry for entry in entries if entry["kind"] == "buffer"]
    nonfinite = [entry for entry in entries if not entry["finite"]]
    ranked = sorted(
        entries,
        key=lambda entry: (
            float(entry["abs_max"]) if entry["abs_max"] is not None else float("inf")
        ),
        reverse=True,
    )
    return {
        "parameters_are_finite": all(entry["finite"] for entry in parameters),
        "buffers_are_finite": all(entry["finite"] for entry in buffers),
        "num_parameters": len(parameters),
        "num_buffers": len(buffers),
        "nonfinite_state": nonfinite,
        "top_abs_state": ranked[: max(1, int(top_k))],
    }


class ForwardActivationTrace:
    """Record module inputs/outputs in completion order for one diagnostic forward."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        for module_name, module in model.named_modules():
            if not module_name:
                continue

            def hook(_module, _inputs, output, *, name=module_name):
                if torch.is_tensor(output):
                    input_tensor = next(
                        (item for item in _inputs if torch.is_tensor(item)), None
                    )
                    self.records.append(
                        {
                            "call_index": len(self.records),
                            "module": name,
                            "class_name": _module.__class__.__name__,
                            "input": tensor_statistics(input_tensor)
                            if input_tensor is not None
                            else None,
                            **tensor_statistics(output),
                        }
                    )

            self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self, *, top_k: int = 12) -> dict[str, Any]:
        first_index = next(
            (i for i, record in enumerate(self.records) if not record["finite"]), None
        )
        first_nonfinite = self.records[first_index] if first_index is not None else None
        last_finite = (
            self.records[first_index - 1] if first_index not in (None, 0) else None
        )
        ranked = sorted(
            self.records,
            key=lambda record: (
                float(record["abs_max"])
                if record["abs_max"] is not None
                else float("inf")
            ),
            reverse=True,
        )
        return {
            "num_module_calls": len(self.records),
            "first_nonfinite_activation": first_nonfinite,
            "last_finite_activation_before_failure": last_finite,
            "top_abs_activations": ranked[: max(1, int(top_k))],
            "activation_trace": self.records,
        }


def _autocast(device_type: str, enabled: bool):
    return torch.autocast(device_type=device_type, enabled=enabled)


def _forward_once(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    amp_enabled: bool,
    record_activations: bool,
    top_k: int,
) -> dict[str, Any]:
    trace = ForwardActivationTrace(model) if record_activations else None
    result: dict[str, Any] = {"autocast_enabled": bool(amp_enabled)}
    try:
        # Keep autograd enabled to match the original training forward. No
        # backward or optimizer step is performed for the intercepted batch.
        with _autocast(x.device.type, amp_enabled):
            logits = model(x.float())
        if isinstance(logits, (list, tuple)):
            raise AssertionError(
                "Unexpected deep-supervision outputs in numerics replay"
            )
        result["logits"] = tensor_statistics(logits)
        result["logits_are_finite"] = bool(result["logits"]["finite"])
    except Exception as exc:
        result["logits_are_finite"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if trace is not None:
            result["activations"] = trace.summary(top_k=top_k)
            trace.close()
    return result


def _capture_local_torch_rng(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_local_torch_rng(
    state: tuple[torch.Tensor, torch.Tensor | None],
    device: torch.device,
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def run_forward_precision_probes(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    amp_enabled: bool,
    top_k: int = 12,
) -> dict[str, Any]:
    """Compare an uninstrumented AMP forward with FP32 from identical RNG state."""
    initial_rng = _capture_local_torch_rng(x.device)
    amp = _forward_once(
        model,
        x,
        amp_enabled=amp_enabled,
        record_activations=False,
        top_k=top_k,
    )

    amp_trace = None
    if not amp["logits_are_finite"]:
        _restore_local_torch_rng(initial_rng, x.device)
        amp_trace = _forward_once(
            model,
            x,
            amp_enabled=amp_enabled,
            record_activations=True,
            top_k=top_k,
        )

    _restore_local_torch_rng(initial_rng, x.device)
    fp32 = _forward_once(
        model,
        x,
        amp_enabled=False,
        record_activations=False,
        top_k=top_k,
    )
    fp32_trace = None
    if not fp32["logits_are_finite"]:
        _restore_local_torch_rng(initial_rng, x.device)
        fp32_trace = _forward_once(
            model,
            x,
            amp_enabled=False,
            record_activations=True,
            top_k=top_k,
        )
    _restore_local_torch_rng(initial_rng, x.device)

    return {
        "amp": amp,
        "amp_activation_replay": amp_trace,
        "fp32": fp32,
        "fp32_activation_replay": fp32_trace,
    }


def classify_failure(
    input_stats: dict[str, Any],
    state_stats: dict[str, Any],
    probes: dict[str, Any],
) -> str:
    if not input_stats["finite"]:
        return "nonfinite_input"
    if (
        not state_stats["parameters_are_finite"]
        or not state_stats["buffers_are_finite"]
    ):
        return "nonfinite_model_state"
    amp_finite = bool(probes["amp"]["logits_are_finite"])
    fp32_finite = bool(probes["fp32"]["logits_are_finite"])
    if not amp_finite and fp32_finite:
        return "autocast_only_forward_failure"
    if not amp_finite and not fp32_finite:
        return "finite_input_and_state_but_both_precisions_fail"
    if amp_finite and not fp32_finite:
        return "fp32_only_forward_failure"
    return "failure_not_reproduced_at_target_forward"
