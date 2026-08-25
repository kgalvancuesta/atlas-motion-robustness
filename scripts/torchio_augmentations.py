#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchio as tio

AVAILABLE_ARTIFACTS = {
    "motion": (tio.RandomMotion, dict(degrees=10, translation=10, num_transforms=2)),
    "ghosting": (tio.RandomGhosting, dict(num_ghosts=(4, 10), intensity=(0.5, 1))),
    "spike": (tio.RandomSpike, dict(num_spikes=1, intensity=(1, 3))),
    "bias_field": (tio.RandomBiasField, dict(coefficients=0.5)),
    "blur": (tio.RandomBlur, dict(std=(0.5, 2.0))),
    "noise": (tio.RandomNoise, dict(mean=0, std=(0, 0.05))),
    "anisotropy": (tio.RandomAnisotropy, dict(downsampling=(1.5, 5))),
}

PRESETS = {
    "all": {
        "motion": 0.3,
        "ghosting": 0.3,
        "spike": 0.2,
        "bias_field": 0.2,
        "blur": 0.3,
        "noise": 0.3,
        "anisotropy": 0.2,
    },
    "blur_only": {
        "blur": 0.3,
        "anisotropy": 0.2,
    },
    "motion_heavy": {
        "motion": 0.5,
        "ghosting": 0.4,
        "blur": 0.3,
    },
    "motion_consistent": {
        "motion": 0.3,
        "ghosting": 0.3,
        "blur": 0.3,
    },
    "light": {
        "motion": 0.1,
        "ghosting": 0.1,
        "spike": 0.1,
        "bias_field": 0.1,
        "blur": 0.15,
        "noise": 0.15,
        "anisotropy": 0.1,
    },
    "heavy": {
        "motion": 0.5,
        "ghosting": 0.5,
        "spike": 0.4,
        "bias_field": 0.4,
        "blur": 0.5,
        "noise": 0.5,
        "anisotropy": 0.4,
    },
}


class TorchIOAugmentation:
    def __init__(self, transform: tio.Compose, *, renormalize: bool = True) -> None:
        self.transform = transform
        self.renormalize = renormalize

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(image[None, ...]).float()
        augmented = self.transform(tensor)
        result = augmented.squeeze(0).numpy()
        if self.renormalize:
            # Re-normalize to prevent extreme values from augmentation
            # causing float16 overflow in AMP training
            brain_mask = np.abs(result) > 1e-6
            if np.any(brain_mask):
                mean = result[brain_mask].mean()
                std = result[brain_mask].std()
                if std > 0:
                    result = (result - mean) / std
        return result, mask


def _build_transforms(artifact_probs: dict) -> List[tio.Transform]:
    transforms = []
    for name, prob in artifact_probs.items():
        if name not in AVAILABLE_ARTIFACTS:
            raise ValueError(f"Unknown artifact '{name}'. Available: {sorted(AVAILABLE_ARTIFACTS)}")
        cls, kwargs = AVAILABLE_ARTIFACTS[name]
        transforms.append(cls(p=prob, **kwargs))
    return transforms


def get_torchio_augmentation(
    artifacts: Optional[List[str]] = None,
    preset: Optional[str] = None,
) -> TorchIOAugmentation:
    if artifacts is not None:
        all_probs = PRESETS["all"]
        artifact_probs = {name: all_probs.get(name, 0.3) for name in artifacts}
    else:
        preset = preset or "all"
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}")
        artifact_probs = PRESETS[preset]

    transforms = _build_transforms(artifact_probs)
    return TorchIOAugmentation(tio.Compose(transforms))
