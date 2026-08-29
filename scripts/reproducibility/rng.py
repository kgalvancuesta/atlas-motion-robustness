"""Stable independent random streams for corrected experiments."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator, Sequence
from typing import Any

from torch.utils.data import Sampler

SEED_DERIVATION_VERSION = "sha256-json-v1"


def derive_seed(global_seed: int, operation: str, **identifiers: Any) -> int:
    """Derive a stable 63-bit seed without Python's process-randomized hash()."""
    payload = {
        "version": SEED_DERIVATION_VERSION,
        "global_seed": int(global_seed),
        "operation": str(operation),
        "identifiers": identifiers,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


class CorrectedSampler(Sampler[int]):
    """DDP-aware sampler with independent clean and augmented permutations.

    Clean indices retain the same relative order and rank assignment when an
    augmented bucket is enabled. This prevents augmentation from defining the
    clean-control execution schedule.
    """

    def __init__(
        self,
        clean_indices: Sequence[int],
        augmented_indices: Sequence[int],
        *,
        global_seed: int,
        fold: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid CorrectedSampler replica configuration")
        self.clean_indices = list(clean_indices)
        self.augmented_indices = list(augmented_indices)
        self.global_seed = int(global_seed)
        self.fold = int(fold)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_bucket(self, values: list[int], role: str) -> list[int]:
        order = list(values)
        rng = random.Random(
            derive_seed(self.global_seed, "dataloader_order", fold=self.fold, epoch=self.epoch, sample_role=role)
        )
        rng.shuffle(order)
        if not order:
            return []
        target = math.ceil(len(order) / self.num_replicas) * self.num_replicas
        order.extend(order[: target - len(order)])
        return order[self.rank:target:self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        clean = self._rank_bucket(self.clean_indices, "clean_control")
        augmented = self._rank_bucket(self.augmented_indices, "augmented_copy")
        return iter(clean + augmented)

    def __len__(self) -> int:
        clean = math.ceil(len(self.clean_indices) / self.num_replicas)
        augmented = math.ceil(len(self.augmented_indices) / self.num_replicas)
        return clean + augmented
