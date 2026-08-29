"""Two-process deterministic AMP/DDP training smoke used by the PACE preflight."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

from models.base_cnn_model import build_base_model
from reproducibility.core import configure_strict_determinism, write_json_new
from reproducibility.rng import derive_seed


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9001)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    configure_strict_determinism(args.seed)
    device = torch.device("cuda", local_rank)
    model = build_base_model().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(derive_seed(args.seed, "preflight.ddp_input", rank=rank))
    x = torch.randn((1, 1, 32, 32, 32), generator=input_generator, device=device)
    target = (torch.rand((1, 1, 32, 32, 32), generator=input_generator, device=device) > 0.98).float()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Non-finite DDP smoke loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
    if not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError("Non-finite DDP smoke gradient norm")
    scaler.step(optimizer)
    scaler.update()
    dist.barrier()
    if rank == 0:
        write_json_new(
            Path(args.output),
            {
                "world_size": dist.get_world_size(),
                "batch_size_per_gpu": 1,
                "amp": True,
                "loss": float(loss.detach().item()),
                "model_state_sha256": state_hash(model.module),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
