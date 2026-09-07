#!/usr/bin/env python3
"""Shared ATLAS training utilities for all model entry points."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import nibabel as nb
import numpy as np
import torch
import torch.distributed as dist
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from split_utils import load_split_payload, resolve_training_split_ids


@dataclass
class Sample:
    subject: str
    t1w_path: Path
    mask_path: Path
    augment: bool = False


def corrected_training_samples(samples: List[Sample], fold: dict, regime: str) -> tuple[List[Sample], List[Sample]]:
    """Keep checkpoint selection clean; append only the stored training duplicates."""
    train = select_samples(samples, fold["train_ids"])
    validation = select_samples(samples, fold["val_ids"])
    if any(sample.augment for sample in train + validation):
        raise ValueError("Corrected split originals must be clean")
    if regime == "augmented":
        by_id = {sample.subject: sample for sample in train}
        duplicate_ids = fold["duplicated_subjects"]["train_ids"]
        if not set(duplicate_ids).issubset(by_id):
            raise ValueError("Stored training duplicates conflict with the resolved split")
        train += [Sample(sid, by_id[sid].t1w_path, by_id[sid].mask_path, augment=True) for sid in duplicate_ids]
    elif regime != "standard":
        raise ValueError(f"Unknown corrected training regime: {regime}")
    return train, validation


def list_labeled_samples(deriv_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for sub_dir in sorted(deriv_root.glob("sub-*/ses-1/anat")):
        subject = sub_dir.parent.parent.name
        t1w = sub_dir / f"{subject}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
        mask = sub_dir / f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
        if t1w.exists() and mask.exists():
            samples.append(Sample(subject=subject, t1w_path=t1w, mask_path=mask))
    return samples


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, nb.Nifti1Header]:
    img = nb.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if np.any(mask):
        mean = vol[mask].mean()
        std = vol[mask].std()
    else:
        mean = vol.mean()
        std = vol.std()
    if std == 0:
        std = 1.0
    return (vol - mean) / std


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask > threshold).astype(np.float32, copy=False)


def extract_patch(vol: np.ndarray, center: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> np.ndarray:
    slices = []
    pads = []
    for c, p, dim in zip(center, patch_size, vol.shape):
        start = c - p // 2
        end = start + p
        if start < 0:
            end -= start
            start = 0
        if end > dim:
            start -= end - dim
            end = dim
        start = max(start, 0)
        end = min(end, dim)
        slices.append(slice(start, end))
        size = end - start
        pads.append((0, max(p - size, 0)))
    patch = vol[tuple(slices)]
    if any(after > 0 for _before, after in pads):
        patch = np.pad(patch, pads, mode="constant", constant_values=0)
    return patch


def sample_center(
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    lesion_prob: float,
    *,
    rng: random.Random | None = None,
) -> Tuple[int, int, int]:
    del patch_size
    source = rng or random
    if mask is not None and mask.max() > 0 and source.random() < lesion_prob:
        coords = np.argwhere(mask > 0)
        idx = source.randrange(coords.shape[0])
        center = coords[idx]
        return int(center[0]), int(center[1]), int(center[2])
    return (
        source.randrange(mask.shape[0]),
        source.randrange(mask.shape[1]),
        source.randrange(mask.shape[2]),
    )


class PatchDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        patch_size: Tuple[int, int, int],
        patches_per_volume: int,
        lesion_prob: float,
        augmentation=None,
        experiment_context: dict[str, Any] | None = None,
    ) -> None:
        self.samples = samples
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.lesion_prob = lesion_prob
        self.augmentation = augmentation
        self.experiment_context = experiment_context
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_volume

    def batch_identity(self, idx: int) -> dict[str, Any]:
        """Identify an already-loaded batch without I/O, recipes, or RNG calls."""
        sample = self.samples[idx // self.patches_per_volume]
        return {"dataset_index": idx, "subject_id": sample.subject,
                "sample_role": "augmented_copy" if sample.augment else "clean_control",
                "patch_slot": idx % self.patches_per_volume}

    def describe_index(self, idx: int) -> dict[str, Any]:
        """Reconstruct the identity and deterministic recipe for one dataset index."""
        if self.experiment_context is None:
            raise RuntimeError("Exact patch descriptions require a corrected experiment context")
        if not 0 <= idx < len(self):
            raise IndexError(f"Patch dataset index {idx} is outside [0, {len(self)})")

        from reproducibility.challenge import sample_legacy_recipe
        from reproducibility.rng import derive_seed

        sample_index = idx // self.patches_per_volume
        patch_slot = idx % self.patches_per_volume
        sample = self.samples[sample_index]
        vol, _, _ = load_nifti(sample.t1w_path)
        mask, _, _ = load_nifti(sample.mask_path)
        mask = binarize_mask(mask)
        global_seed = int(self.experiment_context["global_seed"])
        fold = int(self.experiment_context["fold"])
        patch_seed = derive_seed(
            global_seed,
            "patch_center_sampling",
            fold=fold,
            subject_id=sample.subject,
            epoch=self.epoch,
            patch_slot=patch_slot,
        )
        center = sample_center(mask, self.patch_size, self.lesion_prob, rng=random.Random(patch_seed))
        recipe = None
        if sample.augment:
            recipe = sample_legacy_recipe(
                tuple(int(value) for value in vol.shape),
                global_seed=global_seed,
                subject_id=sample.subject,
                fold=fold,
                replicate_id=0,
                namespace="training_augmentation",
                extra_identifiers={"epoch": self.epoch, "sample_role": "augmented_copy"},
            )
        return {
            "dataset_index": int(idx),
            "sample_index": int(sample_index),
            "subject_id": sample.subject,
            "sample_role": "augmented_copy" if sample.augment else "clean_control",
            "is_augmented_copy": bool(sample.augment),
            "patch_slot": int(patch_slot),
            "patch_center": [int(value) for value in center],
            "patch_seed": int(patch_seed),
            "t1w_path": str(sample.t1w_path),
            "mask_path": str(sample.mask_path),
            "augmentation_recipe": recipe,
        }

    def __getitem__(self, idx: int):
        sample = self.samples[idx // self.patches_per_volume]
        patch_slot = idx % self.patches_per_volume
        vol, _, _ = load_nifti(sample.t1w_path)
        mask, _, _ = load_nifti(sample.mask_path)
        mask = binarize_mask(mask)
        if self.experiment_context is None:
            vol = normalize_volume(vol)
            center = sample_center(mask, self.patch_size, self.lesion_prob)
        else:
            from reproducibility.challenge import load_or_create_validated_cache, sample_legacy_recipe
            from reproducibility.preprocessing import preprocess_volume
            from reproducibility.rng import derive_seed

            global_seed = int(self.experiment_context["global_seed"])
            fold = int(self.experiment_context["fold"])
            patch_seed = derive_seed(
                global_seed,
                "patch_center_sampling",
                fold=fold,
                subject_id=sample.subject,
                epoch=self.epoch,
                patch_slot=patch_slot,
            )
            center = sample_center(
                mask,
                self.patch_size,
                self.lesion_prob,
                rng=random.Random(patch_seed),
            )
            recipe = None
            if sample.augment:
                recipe = sample_legacy_recipe(
                    tuple(int(value) for value in vol.shape),
                    global_seed=global_seed,
                    subject_id=sample.subject,
                    fold=fold,
                    replicate_id=0,
                    namespace="training_augmentation",
                    extra_identifiers={"epoch": self.epoch, "sample_role": "augmented_copy"},
                )
            mode = str(self.experiment_context["normalization_mode"])
            if self.experiment_context.get("memory_mode") == "high":
                cache_root = Path(str(self.experiment_context["cache_root"]))
                challenge_label = "augmentation" if sample.augment else "clean"
                vol, _cache_status = load_or_create_validated_cache(
                    cache_root=cache_root,
                    challenge_id=f"training-{challenge_label}-{self.experiment_context['experiment_id']}",
                    normalization_mode=mode,
                    fold=fold,
                    replicate_id=self.epoch if sample.augment else 0,
                    subject_id=sample.subject,
                    source_sha256=self.experiment_context["source_sha256"][sample.subject],
                    recipe_id=recipe["recipe_id"] if recipe else "clean",
                    builder=lambda: preprocess_volume(vol, mode=mode, recipe=recipe)[0],
                    read_only=bool(self.experiment_context.get("cache_read_only", False)),
                )
            else:
                vol, _diagnostics = preprocess_volume(vol, mode=mode, recipe=recipe)
        vol_patch = extract_patch(vol, center, self.patch_size)
        mask_patch = extract_patch(mask, center, self.patch_size)
        if self.experiment_context is None and self.augmentation is not None and sample.augment:
            vol_patch, mask_patch = self.augmentation(vol_patch, mask_patch)
        vol_patch = torch.from_numpy(vol_patch[None, ...]).float()
        mask_patch = torch.from_numpy(mask_patch[None, ...]).float()
        return vol_patch, mask_patch


class VolumeDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        augmentation=None,
        experiment_context: dict[str, Any] | None = None,
    ) -> None:
        self.samples = samples
        self.augmentation = augmentation
        self.experiment_context = experiment_context
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        vol, _, _ = load_nifti(sample.t1w_path)
        mask, _, _ = load_nifti(sample.mask_path)
        mask = binarize_mask(mask)
        if self.experiment_context is None:
            vol = normalize_volume(vol)
        else:
            from reproducibility.challenge import load_or_create_validated_cache, sample_legacy_recipe
            from reproducibility.preprocessing import preprocess_volume

            recipe = None
            if sample.augment:
                recipe = sample_legacy_recipe(
                    tuple(int(value) for value in vol.shape),
                    global_seed=int(self.experiment_context["global_seed"]),
                    subject_id=sample.subject,
                    fold=int(self.experiment_context["fold"]),
                    replicate_id=0,
                    namespace="validation_augmentation",
                    extra_identifiers={"epoch": self.epoch, "sample_role": "augmented_copy"},
                )
            mode = str(self.experiment_context["normalization_mode"])
            if self.experiment_context.get("memory_mode") == "high":
                vol, _cache_status = load_or_create_validated_cache(
                    cache_root=Path(str(self.experiment_context["cache_root"])),
                    challenge_id=(
                        f"validation-augmentation-{self.experiment_context['experiment_id']}"
                        if sample.augment
                        else f"validation-clean-{self.experiment_context['experiment_id']}"
                    ),
                    normalization_mode=mode,
                    fold=int(self.experiment_context["fold"]),
                    replicate_id=self.epoch if sample.augment else 0,
                    subject_id=sample.subject,
                    source_sha256=self.experiment_context["source_sha256"][sample.subject],
                    recipe_id=recipe["recipe_id"] if recipe else "clean",
                    builder=lambda: preprocess_volume(vol, mode=mode, recipe=recipe)[0],
                    read_only=bool(self.experiment_context.get("cache_read_only", False)),
                )
            else:
                vol, _diagnostics = preprocess_volume(vol, mode=mode, recipe=recipe)
        if self.experiment_context is None and self.augmentation is not None and sample.augment:
            vol, mask = self.augmentation(vol, mask)
        vol = torch.from_numpy(vol[None, ...]).float()
        mask = torch.from_numpy(mask[None, ...]).float()
        return vol, mask, sample.subject


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2.0 * inter + eps) / (denom + eps)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> Tuple[bool, int]:
    if not is_distributed():
        return False, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_splits(subject_ids: List[str], seed: int) -> dict:
    ids = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_heldout = int(n_total * 0.15)
    n_dev = int((n_total - n_heldout) * 0.20)
    heldout = ids[:n_heldout]
    dev = ids[n_heldout : n_heldout + n_dev]
    train = ids[n_heldout + n_dev :]
    if set(train) & set(dev) or set(train) & set(heldout) or set(dev) & set(heldout):
        raise RuntimeError("Split overlap detected")
    return {
        "train_ids": train,
        "dev_ids": dev,
        "heldout_ids": heldout,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Deterministic split: 15% heldout, 20% of remainder dev.",
    }


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text())


def select_samples(samples: List[Sample], ids: List[str]) -> List[Sample]:
    id_set = {sid if sid.startswith("sub-") else f"sub-{sid}" for sid in ids}
    return [s for s in samples if s.subject in id_set]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


class MedNeXtBinaryLoss(torch.nn.Module):
    """Binary equivalent of MedNeXt's DC_and_BCE_loss for one-channel logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        axes = tuple(range(2, logits.ndim))
        intersection = (probs * target).sum(dim=axes)
        fp = (probs * (1.0 - target)).sum(dim=axes)
        fn = ((1.0 - probs) * target).sum(dim=axes)
        dice = (2.0 * intersection + self.smooth) / (2.0 * intersection + fp + fn + self.smooth + 1e-8)
        return self.bce(logits, target) - dice.mean()


def build_loss(loss_name: str) -> torch.nn.Module:
    if loss_name == "baseline":
        return DiceLoss(sigmoid=True)
    if loss_name == "mednext":
        return MedNeXtBinaryLoss()
    if loss_name == "dicece":
        return DiceCELoss(sigmoid=True)
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_argument_parser(description: str, default_loss: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data_root",
        default=None,
        help="Dataset root (contains train/ and test/). Overrides --gt_deriv if provided.",
    )
    parser.add_argument(
        "--gt_deriv",
        default=None,
        help="GT derivative root (defaults to <data_root>/train/derivatives/ATLAS)",
    )
    parser.add_argument("--run_dir", required=True, help="Run directory (e.g., runs/base_cnn/run_kfold_01)")
    parser.add_argument(
        "--splits_json",
        default="splits/atlas_5fold_lesion_quartile_excluding_known_issues.json",
        help="Split JSON path (defaults to the paper CV master; pass --cv_fold).",
    )
    parser.add_argument("--cv_fold", type=int, default=None, help="CV fold index for master CV split JSONs")
    parser.add_argument("--make_splits", action="store_true", help="Create splits and exit")
    parser.add_argument("--max_epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--patches_per_volume", type=int, default=8)
    parser.add_argument("--lesion_prob", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", help="Force pin_memory=True")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience: stop if dev Dice doesn't improve for this many validation epochs. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--numerics_debug", action="store_true", help="Enable detailed numerics diagnostics")
    parser.add_argument(
        "--numerics_debug_hooks",
        action="store_true",
        help="With --numerics_debug, record MedNeXt block activation stats via forward hooks",
    )
    parser.add_argument(
        "--numerics_debug_fp32_probe",
        action="store_true",
        help="With --numerics_debug, re-run failing AMP batch in fp32 to check autocast sensitivity",
    )
    parser.add_argument(
        "--numerics_debug_topk",
        type=int,
        default=8,
        help="How many largest grad/activation magnitudes to include in numerics diagnostics",
    )
    parser.add_argument("--loss", default=default_loss, choices=["baseline", "mednext", "dicece"])
    parser.add_argument(
        "--augment", type=str, default=None, metavar="PRESET",
        help="Enable on-the-fly TorchIO augmentation with given preset (e.g. motion_heavy)",
    )
    parser.add_argument(
        "--augment_frac", type=float, default=0.5,
        help="Fraction of training subjects to duplicate with augmentation (default: 0.5)",
    )
    parser.add_argument(
        "--experiment_definition",
        default=None,
        help="Immutable corrected experiment definition. Omit for historical/original behavior.",
    )
    parser.add_argument("--experiment_fold", type=int, default=None)
    parser.add_argument("--experiment_training_regime", choices=["standard", "augmented"], default=None)
    parser.add_argument(
        "--normalization_mode",
        choices=["legacy", "artifact_then_normalize", "normalize_then_artifact"],
        default=None,
    )
    parser.add_argument(
        "--corrected_resume",
        action="store_true",
        help="Resume only from a compatible corrected operational last checkpoint.",
    )
    parser.add_argument("--experiment_memory_mode", choices=["low", "high"], default="low")
    parser.add_argument("--experiment_cache_root", default=None)
    return parser


def resolve_data_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    if args.gt_deriv is None:
        args.gt_deriv = str(data_root / "train" / "derivatives" / "ATLAS")
    return data_root, Path(args.gt_deriv)


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(*, strict: bool = False) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if strict and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if strict and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=not strict)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


MEDNEXT_DEBUG_BLOCK_CLASS_NAMES = {"MedNeXtBlock", "MedNeXtDownBlock", "MedNeXtUpBlock", "OutBlock"}
MAX_CONSECUTIVE_AMP_OVERFLOWS = 8


def _module_from_param_name(param_name: str) -> str:
    if "." not in param_name:
        return "<root>"
    return param_name.rsplit(".", 1)[0]


def _summarize_tensor_output(output: Any) -> dict[str, Any] | None:
    if torch.is_tensor(output):
        t = output.detach()
        return {
            "is_finite": bool(torch.isfinite(t).all().item()),
            "abs_max": float(t.abs().amax().item()) if t.numel() > 0 else 0.0,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }

    if isinstance(output, (list, tuple)):
        tensors = [item.detach() for item in output if torch.is_tensor(item)]
        if not tensors:
            return None
        return {
            "is_finite": all(bool(torch.isfinite(t).all().item()) for t in tensors),
            "abs_max": max(float(t.abs().amax().item()) if t.numel() > 0 else 0.0 for t in tensors),
            "shape": [list(t.shape) for t in tensors[:4]],
            "dtype": sorted({str(t.dtype) for t in tensors}),
            "num_tensors": len(tensors),
        }
    return None


def collect_gradient_diagnostics(model: torch.nn.Module, *, top_k: int) -> dict[str, Any]:
    top_k = max(1, int(top_k))
    first_nonfinite: dict[str, Any] | None = None
    nonfinite_params: List[dict[str, Any]] = []
    top_abs: List[dict[str, Any]] = []
    grad_params = 0

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_params += 1
        g = grad.detach()
        abs_max = float(g.abs().amax().item()) if g.numel() > 0 else 0.0
        is_finite = bool(torch.isfinite(g).all().item())
        entry = {
            "param": name,
            "module": _module_from_param_name(name),
            "abs_max": abs_max,
            "is_finite": is_finite,
            "dtype": str(g.dtype),
        }

        if not is_finite:
            if first_nonfinite is None:
                first_nonfinite = entry
            if len(nonfinite_params) < top_k:
                nonfinite_params.append(entry)

        top_abs.append(entry)

    top_abs.sort(key=lambda item: item["abs_max"], reverse=True)
    return {
        "num_params_with_grad": grad_params,
        "num_nonfinite_grad_params": int(sum(1 for item in top_abs if not item["is_finite"])),
        "first_nonfinite_grad": first_nonfinite,
        "nonfinite_grad_params": nonfinite_params,
        "top_abs_grad_params": top_abs[:top_k],
    }


class ActivationMonitor:
    def __init__(self, model: torch.nn.Module, *, enabled: bool) -> None:
        self.enabled = enabled
        self.records: dict[str, dict[str, Any]] = {}
        self.handles: List[Any] = []
        if not self.enabled:
            return
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            class_name = module.__class__.__name__
            if class_name not in MEDNEXT_DEBUG_BLOCK_CLASS_NAMES:
                continue

            def _hook(_module, _inputs, output, *, _module_name=module_name, _class_name=class_name):
                summary = _summarize_tensor_output(output)
                if summary is None:
                    return
                self.records[_module_name] = {"class_name": _class_name, **summary}

            self.handles.append(module.register_forward_hook(_hook))

    def clear_step(self) -> None:
        self.records.clear()

    def snapshot(self, *, top_k: int) -> dict[str, Any] | None:
        if not self.records:
            return None
        top_k = max(1, int(top_k))
        items = sorted(self.records.items(), key=lambda kv: kv[1]["abs_max"], reverse=True)
        first_nonfinite = None
        for module_name, stats in self.records.items():
            if not stats.get("is_finite", True):
                first_nonfinite = {"module": module_name, **stats}
                break
        return {
            "num_tracked_modules": len(self.records),
            "first_nonfinite_activation": first_nonfinite,
            "top_abs_activations": [{"module": name, **stats} for name, stats in items[:top_k]],
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_fp32_probe(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    accum_steps: int,
    top_k: int,
    collective_safe: bool = False,
) -> dict[str, Any]:
    if not collective_safe:
        model.zero_grad(set_to_none=True)
        probe: dict[str, Any] = {}
        try:
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                raise AssertionError("Unexpected deep supervision outputs in fp32 probe")
            assert_logits_shape(logits, x)
            probe["logits_are_finite"] = bool(torch.isfinite(logits).all().item())
            probe["max_abs_logit"] = float(logits.detach().abs().amax().item()) if logits.numel() > 0 else 0.0
            loss = loss_fn(logits.float(), y.float()) / accum_steps
            probe["loss"] = float(loss.detach().item())
            probe["loss_is_finite"] = bool(torch.isfinite(loss).item())
            if probe["loss_is_finite"]:
                loss.backward()
                probe["gradients"] = collect_gradient_diagnostics(model, top_k=top_k)
            else:
                probe["gradients"] = None
        except Exception as exc:
            probe["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            model.zero_grad(set_to_none=True)
        return probe

    from reproducibility.numerics import _capture_forward_rng, _restore_forward_rng, collective_all

    initial_rng = _capture_forward_rng(x.device)
    buffers = [(value, value.detach().clone()) for value in model.buffers()]
    model.zero_grad(set_to_none=True)
    probe: dict[str, Any] = {}
    loss = None
    try:
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = model(x)
            if isinstance(logits, (list, tuple)):
                raise AssertionError("Unexpected deep supervision outputs in fp32 probe")
            assert_logits_shape(logits, x)
            probe["logits_are_finite"] = bool(torch.isfinite(logits).all().item())
            probe["max_abs_logit"] = float(logits.detach().abs().amax().item()) if logits.numel() > 0 else 0.0
            loss = loss_fn(logits.float(), y.float()) / accum_steps
            probe["loss"] = float(loss.detach().item())
            probe["loss_is_finite"] = bool(torch.isfinite(loss).item())
        except Exception as exc:
            probe["error"] = f"{type(exc).__name__}: {exc}"
        ready = "error" not in probe and probe.get("logits_are_finite", False) and probe.get("loss_is_finite", False)
        if collective_all(ready, x.device):
            loss.backward()
            probe["gradients"] = collect_gradient_diagnostics(model, top_k=top_k)
        else:
            probe["gradients"] = None
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)
        _restore_forward_rng(initial_rng, x.device)
    return probe


def assert_patch_batch(x: torch.Tensor, y: torch.Tensor, patch_size: Tuple[int, int, int]) -> None:
    if x.ndim != 5 or y.ndim != 5:
        raise AssertionError(f"Expected training tensors [B, C, D, H, W], got x={tuple(x.shape)} y={tuple(y.shape)}")
    if tuple(x.shape[2:]) != patch_size or tuple(y.shape[2:]) != patch_size:
        raise AssertionError(
            f"Patch shape mismatch: expected {patch_size}, got x={tuple(x.shape[2:])} y={tuple(y.shape[2:])}"
        )
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise AssertionError(f"Expected one input/output channel, got x={x.shape[1]} y={y.shape[1]}")


def assert_logits_shape(logits: torch.Tensor, reference: torch.Tensor) -> None:
    if logits.ndim != 5:
        raise AssertionError(f"Expected logits [B, C, D, H, W], got {tuple(logits.shape)}")
    if logits.shape[0] != reference.shape[0]:
        raise AssertionError(f"Batch mismatch between logits {tuple(logits.shape)} and ref {tuple(reference.shape)}")
    if logits.shape[1] != 1:
        raise AssertionError(f"Expected one logit channel, got {logits.shape[1]}")
    if tuple(logits.shape[2:]) != tuple(reference.shape[2:]):
        raise AssertionError(
            f"Logit spatial shape mismatch: logits={tuple(logits.shape[2:])}, ref={tuple(reference.shape[2:])}"
        )


def get_path_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def write_numerics_abort(
    run_dir: Path,
    *,
    epoch: int,
    step: int,
    global_step: int,
    reason: str,
    loss: float | None,
    max_abs_logit: float | None,
    grad_norm: float | None,
    scaler_scale: float | None,
    lr: float,
    debug: dict[str, Any] | None = None,
) -> Path:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"numerics_abort_epoch_{epoch:03d}_step_{step:05d}_{stamp}.json"
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "reason": reason,
        "loss": loss,
        "max_abs_logit": max_abs_logit,
        "grad_norm": grad_norm,
        "scaler_scale": scaler_scale,
        "lr": lr,
    }
    if debug:
        payload["debug"] = debug
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_validation_artifacts(
    run_dir: Path,
    epoch: int,
    case_rows: List[dict],
    *,
    model_training_flag_at_val_entry: bool,
    model_training_flag_after_eval: bool,
    num_val_batches: int,
    subject_ids: List[str],
    validation_checkpoint: dict | None,
    best_checkpoint_before_val: dict | None,
) -> tuple[Path, Path]:
    logs_dir = run_dir / "logs"
    csv_path = logs_dir / f"val_epoch_{epoch:03d}_cases.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "gt_voxels", "pred_voxels", "dice"])
        writer.writeheader()
        writer.writerows(case_rows)

    subject_ids_hash = hashlib.sha256("\n".join(subject_ids).encode("utf-8")).hexdigest()
    meta_path = logs_dir / f"val_epoch_{epoch:03d}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "model_training_flag_at_val_entry": model_training_flag_at_val_entry,
                "model_training_flag_after_eval": model_training_flag_after_eval,
                "num_val_batches": num_val_batches,
                "num_val_cases": len(case_rows),
                "subject_ids": subject_ids,
                "subject_ids_hash": subject_ids_hash,
                "validation_weights_source": "in_memory_current_model",
                "validation_checkpoint": validation_checkpoint,
                "best_checkpoint_before_val": best_checkpoint_before_val,
            },
            indent=2,
        )
    )
    return csv_path, meta_path


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=device) also relocates RNG byte tensors. Generator
    # state APIs require CPU byte tensors even when restoring a CUDA generator.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(
    args: argparse.Namespace,
    *,
    model_builder: Callable[[], torch.nn.Module],
    model_name: str,
    patch_size_validator: Callable[[Tuple[int, int, int]], None] | None = None,
) -> int:
    set_seed(args.seed)
    distributed, local_rank = setup_distributed()
    activation_monitor: ActivationMonitor | None = None

    try:
        diagnostic_epoch = getattr(args, "diagnostic_replay_epoch", None)
        diagnostic_step = getattr(args, "diagnostic_replay_step", None)
        diagnostic_output_value = getattr(args, "diagnostic_output", None)
        diagnostic_requested = any(
            value is not None for value in (diagnostic_epoch, diagnostic_step, diagnostic_output_value)
        )
        if diagnostic_requested and any(
            value is None for value in (diagnostic_epoch, diagnostic_step, diagnostic_output_value)
        ):
            raise RuntimeError("Diagnostic replay requires epoch, step, and output together")

        verify_fix = bool(getattr(args, "diagnostic_verify_fix", False))
        if verify_fix and not diagnostic_requested:
            raise RuntimeError("Fix verification requires the explicit diagnostic replay entrypoint")
        data_root, gt_root = resolve_data_paths(args)
        experiment_definition: dict[str, Any] | None = None
        experiment_context: dict[str, Any] | None = None
        corrected_checkpoint_metadata: dict[str, Any] | None = None
        corrected_enabled = args.experiment_definition is not None
        if corrected_enabled:
            from reproducibility.core import PREPROCESSING_VERSION, read_json, sha256_file, validate_metadata
            from reproducibility.experiment import fold_definition, validate_training_protocol

            experiment_definition = read_json(Path(args.experiment_definition))
            if not diagnostic_requested:
                validate_training_protocol(experiment_definition)
            if experiment_definition.get("generation") != "corrected_experiment_v1":
                raise SystemExit("ERROR: --experiment_definition is not a corrected experiment definition.")
            if args.experiment_fold is None or args.experiment_training_regime is None or args.normalization_mode is None:
                raise SystemExit(
                    "ERROR: corrected training requires --experiment_fold, --experiment_training_regime, and --normalization_mode."
                )
            if args.cv_fold != args.experiment_fold:
                raise SystemExit("ERROR: --cv_fold and --experiment_fold must match.")
            if args.normalization_mode != experiment_definition["configuration"]["normalization_mode"]:
                raise SystemExit("ERROR: normalization mode conflicts with the immutable experiment definition.")
            if args.max_epochs != int(experiment_definition["configuration"]["max_epochs"]):
                raise SystemExit("ERROR: max epochs conflicts with the immutable experiment definition.")
            if args.batch_size != 1:
                raise SystemExit("ERROR: corrected experiments preserve batch size 1 per GPU.")
            if args.accum_steps != 1:
                raise SystemExit("ERROR: corrected experiments preserve accumulation_steps=1.")
            if sha256_file(Path(args.splits_json)) != experiment_definition["dataset"]["split_sha256"]:
                raise SystemExit("ERROR: split file checksum conflicts with the immutable experiment definition.")
            if model_name not in experiment_definition["configuration"]["models"]:
                raise SystemExit(f"ERROR: model {model_name} is not defined by this experiment.")
            if args.experiment_training_regime not in experiment_definition["configuration"]["training_regimes"]:
                raise SystemExit("ERROR: training regime is not defined by this experiment.")
            if args.experiment_training_regime == "standard" and args.augment is not None:
                raise SystemExit("ERROR: standard corrected training cannot enable augmentation.")
            if args.experiment_training_regime == "augmented" and args.augment != "motion_consistent":
                raise SystemExit("ERROR: augmented corrected training requires --augment motion_consistent.")
            fold_definition(experiment_definition, args.experiment_fold)
            experiment_context = {
                "experiment_id": experiment_definition["experiment_id"],
                "global_seed": int(experiment_definition["seeds"]["global_seed"]),
                "fold": int(args.experiment_fold),
                "training_regime": args.experiment_training_regime,
                "normalization_mode": args.normalization_mode,
                "memory_mode": args.experiment_memory_mode,
                "cache_root": args.experiment_cache_root,
                "cache_read_only": diagnostic_requested,
                "source_sha256": {
                    entry["subject_id"]: entry["sha256"]
                    for entry in experiment_definition["dataset"]["source_inventory"]
                },
            }
            if args.experiment_memory_mode == "high" and not args.experiment_cache_root:
                raise SystemExit("ERROR: --experiment_memory_mode high requires --experiment_cache_root.")
            model_configuration = experiment_definition["model_configurations"][model_name]
            scientific_actual = {
                "patch_size": list(args.patch_size),
                "patches_per_volume": int(args.patches_per_volume),
                "lesion_probability": float(args.lesion_prob),
                "learning_rate": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "accumulation_steps": int(args.accum_steps),
                "loss": args.loss,
                "amp": bool(args.amp),
            }
            scientific_expected = {
                key: model_configuration[key] for key in scientific_actual
            }
            if scientific_actual != scientific_expected:
                raise SystemExit(
                    "ERROR: corrected training configuration conflicts with the immutable model definition: "
                    f"actual={scientific_actual}, expected={scientific_expected}"
                )
            expected_patience = int(experiment_definition["configuration"]["early_stopping_patience"])
            expected_val_interval = int(experiment_definition["configuration"]["validation_interval"])
            if args.patience != expected_patience or args.val_interval != expected_val_interval:
                raise SystemExit(
                    "ERROR: early-stopping/validation settings conflict with the immutable experiment definition: "
                    f"patience={args.patience} expected={expected_patience}, "
                    f"val_interval={args.val_interval} expected={expected_val_interval}."
                )
            corrected_checkpoint_metadata = {
                "experiment_id": experiment_definition["experiment_id"],
                "generation": "corrected_experiment_v1",
                "model": model_name,
                "model_configuration_id": model_configuration["model_configuration_id"],
                "fold": int(args.experiment_fold),
                "training_regime": args.experiment_training_regime,
                "normalization_mode": args.normalization_mode,
                "preprocessing_version": PREPROCESSING_VERSION,
                "fold_definition_id": experiment_definition["dataset"]["fold_definition_id"],
                "global_seed": int(experiment_definition["seeds"]["global_seed"]),
                "training_run_seed": int(args.seed),
                **{key: experiment_definition["configuration"][key]
                   for key in ("checkpoint_selection", "numerical_policy")
                   if key in experiment_definition["configuration"]},
            }
        patch_size = tuple(args.patch_size)
        if patch_size_validator is not None:
            patch_size_validator(patch_size)

        samples = list_labeled_samples(gt_root)
        if not samples:
            raise SystemExit(f"No labeled samples found under {gt_root}")

        splits_path = Path(args.splits_json)
        if corrected_enabled and (args.make_splits or not splits_path.exists()):
            raise SystemExit("ERROR: corrected experiments never create or regenerate a missing split.")
        if args.make_splits or not splits_path.exists():
            if args.cv_fold is not None:
                raise SystemExit("ERROR: --cv_fold cannot be used with --make_splits or a missing split file.")
            splits = make_splits([s.subject for s in samples], args.seed)
            save_splits(splits, splits_path)
            if args.make_splits:
                print(f"Wrote splits to {splits_path}")
                return 0
        else:
            splits = load_split_payload(splits_path)

        train_ids, dev_ids, split_resolution = resolve_training_split_ids(splits, cv_fold=args.cv_fold)
        train_samples = select_samples(samples, train_ids)
        dev_samples = select_samples(samples, dev_ids)

        augmentation = None
        n_aug = 0
        n_dev_aug = 0
        if args.augment:
            if corrected_enabled:
                assert experiment_definition is not None
                from reproducibility.experiment import fold_definition

                fold_payload = fold_definition(experiment_definition, int(args.experiment_fold))
                if sorted(fold_payload["train_ids"]) != sorted(train_ids) or sorted(fold_payload["val_ids"]) != sorted(dev_ids):
                    raise RuntimeError("Immutable fold subjects conflict with the resolved split")
                train_samples, dev_samples = corrected_training_samples(
                    samples, fold_payload, args.experiment_training_regime
                )
                n_aug = sum(sample.augment for sample in train_samples)
            else:
                try:
                    from torchio_augmentations import get_torchio_augmentation
                except ImportError as exc:
                    raise ImportError(
                        "TorchIO augmentation was requested, but the required 'torchio' dependency is not installed."
                    ) from exc
                augmentation = get_torchio_augmentation(preset=args.augment)
                # Historical behavior: preserve the original dedicated-but-shared stream.
                aug_rng = random.Random(args.seed + 7)
                n_aug = int(len(train_samples) * args.augment_frac)
                aug_subjects = aug_rng.sample(train_samples, n_aug)
                aug_copies = [Sample(s.subject, s.t1w_path, s.mask_path, augment=True) for s in aug_subjects]
                train_samples = train_samples + aug_copies
                n_dev_aug = int(len(dev_samples) * args.augment_frac)
                dev_aug_subjects = aug_rng.sample(dev_samples, n_dev_aug)
                dev_aug_copies = [Sample(s.subject, s.t1w_path, s.mask_path, augment=True) for s in dev_aug_subjects]
                dev_samples = dev_samples + dev_aug_copies

        source_run_dir = Path(args.run_dir)
        run_dir = source_run_dir
        if diagnostic_requested:
            diagnostic_path = Path(diagnostic_output_value).resolve()
            if diagnostic_path.is_relative_to(source_run_dir.resolve()):
                raise RuntimeError("Diagnostic output must be outside the source task directory")
            if args.experiment_cache_root and diagnostic_path.is_relative_to(Path(args.experiment_cache_root).resolve()):
                raise RuntimeError("Diagnostic output must be outside the source cache directory")
            run_dir = diagnostic_path.with_suffix(".artifacts")
            if run_dir.exists() or diagnostic_path.exists():
                raise RuntimeError(f"Refusing to overwrite diagnostic output: {diagnostic_path}, {run_dir}")
            if distributed:
                dist.barrier()
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "config").mkdir(parents=True, exist_ok=True)
        (run_dir / "preds_bids").mkdir(parents=True, exist_ok=True)
        (run_dir / "raw_preds").mkdir(parents=True, exist_ok=True)

        device = get_device(local_rank)
        configure_runtime(strict=corrected_enabled)

        train_ds = PatchDataset(
            train_samples,
            patch_size,
            args.patches_per_volume,
            args.lesion_prob,
            augmentation=augmentation,
            experiment_context=experiment_context,
        )
        if corrected_enabled:
            if sorted(sample.subject for sample in dev_samples) != sorted(dev_ids) or any(s.augment for s in dev_samples):
                raise RuntimeError("Checkpoint selection must use exactly the original clean validation IDs")
        dev_ds = VolumeDataset(
            dev_samples, augmentation=None if corrected_enabled else augmentation, experiment_context=experiment_context
        )

        if not distributed or dist.get_rank() == 0:
            print(f"Device: {device}")
            print(f"Model: {model_name}")
            print(f"Train subjects: {len(train_samples)} | Dev subjects: {len(dev_samples)}")
            print(f"Augmentation: {args.augment or 'none'} ({n_aug} train + {n_dev_aug} dev augmented subjects added)")
            print(f"Split resolution: {json.dumps(split_resolution, sort_keys=True)}")

        if corrected_enabled:
            from reproducibility.rng import CorrectedSampler, derive_seed

            clean_indices: list[int] = []
            augmented_indices: list[int] = []
            for sample_index, sample in enumerate(train_samples):
                target = augmented_indices if sample.augment else clean_indices
                target.extend(
                    range(
                        sample_index * args.patches_per_volume,
                        (sample_index + 1) * args.patches_per_volume,
                    )
                )
            train_sampler = CorrectedSampler(
                clean_indices,
                augmented_indices,
                global_seed=int(experiment_context["global_seed"]),
                fold=int(args.experiment_fold),
                num_replicas=dist.get_world_size() if distributed else 1,
                rank=dist.get_rank() if distributed else 0,
            )
        else:
            train_sampler = DistributedSampler(train_ds) if distributed else None

        if args.num_workers is None:
            args.num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = args.pin_memory or torch.cuda.is_available()

        repo_root = Path(__file__).resolve().parents[1]
        git_commit = get_git_commit(repo_root)
        ddp_find_unused_parameters = model_name == "mednext"
        config_path = (source_run_dir if diagnostic_requested else run_dir) / "config" / "train_config.json"
        recorded_args = dict(vars(args))
        if not corrected_enabled:
            for key in (
                "experiment_definition",
                "experiment_fold",
                "experiment_training_regime",
                "normalization_mode",
                "corrected_resume",
                "experiment_memory_mode",
                "experiment_cache_root",
            ):
                recorded_args.pop(key, None)
        config_payload = {
            "args": recorded_args,
            "ddp_find_unused_parameters": ddp_find_unused_parameters,
            "git_commit": git_commit,
            "model_name": model_name,
            "split_resolution": split_resolution,
        }
        if corrected_enabled:
            config_payload["corrected_experiment_metadata"] = corrected_checkpoint_metadata
            scientific_keys = {
                "max_epochs",
                "batch_size",
                "patch_size",
                "patches_per_volume",
                "lesion_prob",
                "lr",
                "weight_decay",
                "accum_steps",
                "val_interval",
                "patience",
                "seed",
                "amp",
                "loss",
                "augment",
                "augment_frac",
                "experiment_fold",
                "experiment_training_regime",
                "normalization_mode",
            }
            immutable_training_request = {
                key: value for key, value in vars(args).items() if key in scientific_keys
            }
            immutable_training_request["experiment_id"] = experiment_definition["experiment_id"]
            immutable_training_request["split_sha256"] = experiment_definition["dataset"]["split_sha256"]
            if not diagnostic_requested:
                immutable_training_request.update({key: experiment_definition["configuration"][key]
                                                   for key in ("checkpoint_selection", "numerical_policy")})
            config_payload["immutable_training_request"] = immutable_training_request
            if not distributed or dist.get_rank() == 0:
                if config_path.exists():
                    existing_config = json.loads(config_path.read_text())
                    if existing_config.get("immutable_training_request") != immutable_training_request:
                        raise RuntimeError(
                            f"Corrected training configuration conflicts with existing {config_path}; use a new experiment ID."
                        )
                elif diagnostic_requested:
                    raise RuntimeError(f"Diagnostic replay requires existing configuration: {config_path}")
                else:
                    config_path.write_text(json.dumps(config_payload, indent=2))
            if distributed:
                dist.barrier()
        else:
            config_path.write_text(json.dumps(config_payload, indent=2))

        loader_generator = None
        worker_init_fn = None
        if corrected_enabled:
            loader_generator = torch.Generator()
            loader_generator.manual_seed(
                derive_seed(
                    int(experiment_context["global_seed"]),
                    "dataloader_worker_base",
                    fold=int(args.experiment_fold),
                    rank=dist.get_rank() if distributed else 0,
                )
            )

            def seed_worker(_worker_id: int) -> None:
                worker_seed = torch.initial_seed() % (2**32)
                random.seed(worker_seed)
                np.random.seed(worker_seed)

            worker_init_fn = seed_worker

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            generator=loader_generator,
            worker_init_fn=worker_init_fn,
        )
        dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False, num_workers=0)

        model = model_builder().to(device)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=ddp_find_unused_parameters,
            )

        loss_fn = build_loss(args.loss)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
        use_amp = args.amp and torch.cuda.is_available()
        scaler = make_grad_scaler(enabled=use_amp)
        numerics_topk = max(1, int(args.numerics_debug_topk))
        main_process = not distributed or dist.get_rank() == 0
        if args.numerics_debug and args.numerics_debug_hooks and model_name == "mednext":
            activation_monitor = ActivationMonitor(unwrap_model(model), enabled=True)

        from reproducibility.numerics import TrainingNumerics

        numerics = TrainingNumerics(
            model, device=device, run_dir=run_dir, amp_enabled=use_amp,
            diagnostics=(lambda: {
                "activations": activation_monitor.snapshot(top_k=numerics_topk) if activation_monitor else None,
                "gradients": collect_gradient_diagnostics(model, top_k=numerics_topk),
            }) if args.numerics_debug else None,
            task={"model": model_name, "regime": args.experiment_training_regime,
                  "fold": args.experiment_fold, "experiment_id": experiment_definition["experiment_id"]},
        ) if corrected_enabled else None
        if numerics is not None:
            numerics.check_model_state()

        best_dice = -1.0
        epochs_without_improvement = 0
        global_step = 0
        consecutive_amp_overflows = 0
        start_epoch = 1
        log_path = run_dir / "logs" / "train_log.csv"
        if not log_path.exists() and (not distributed or dist.get_rank() == 0):
            log_path.write_text("epoch,train_loss,dev_dice,dev_loss\n")

        last_checkpoint_path = source_run_dir / "checkpoints" / "last.pt"
        resume_checkpoint_epoch = None
        resume_checkpoint_global_step = None
        if corrected_enabled and args.corrected_resume:
            if not last_checkpoint_path.exists():
                raise RuntimeError(f"Corrected resume requested but operational checkpoint is missing: {last_checkpoint_path}")
            try:
                resume_state = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                resume_state = torch.load(last_checkpoint_path, map_location=device)
            if corrected_checkpoint_metadata is None:
                raise AssertionError("Missing corrected-experiment checkpoint metadata")
            from reproducibility.core import validate_metadata

            validate_metadata(
                resume_state.get("corrected_experiment_metadata", {}),
                corrected_checkpoint_metadata,
                context=str(last_checkpoint_path),
            )
            unwrap_model(model).load_state_dict(resume_state["model"])
            optimizer.load_state_dict(resume_state["optimizer"])
            scaler.load_state_dict(resume_state["scaler"])
            numerics.fallback_count = int(resume_state.get("fallback_count", 0))
            numerics.check_model_state()
            best_dice = float(resume_state["best_dice"])
            epochs_without_improvement = int(resume_state["epochs_without_improvement"])
            global_step = int(resume_state["global_step"])
            start_epoch = int(resume_state["epoch"]) + 1
            resume_checkpoint_epoch = int(resume_state["epoch"])
            resume_checkpoint_global_step = int(resume_state["global_step"])
            rank = dist.get_rank() if distributed else 0
            rng_states = resume_state.get("rng_states_by_rank", [])
            if rank >= len(rng_states):
                raise RuntimeError(f"Resume checkpoint lacks RNG state for rank {rank}")
            _restore_rng_state(rng_states[rank])
            if main_process:
                print(f"Resuming corrected training at epoch {start_epoch} from {last_checkpoint_path}")

        diagnostic_output = Path(diagnostic_output_value) if diagnostic_output_value is not None else None
        if diagnostic_requested:
            if not corrected_enabled or not args.corrected_resume:
                raise RuntimeError("Diagnostic replay requires a corrected operational checkpoint resume")
            if model_name != "base_cnn":
                raise RuntimeError("The targeted diagnostic replay currently supports only base_cnn")
            if int(args.accum_steps) != 1:
                raise RuntimeError("Exact diagnostic replay requires accumulation_steps=1")
            if int(args.batch_size) != 1:
                raise RuntimeError("Exact diagnostic index mapping requires corrected batch_size=1")
            expected_world_size = int(experiment_definition["configuration"]["nproc_per_node"])
            actual_world_size = dist.get_world_size() if distributed else 1
            if actual_world_size != expected_world_size:
                raise RuntimeError(
                    f"Diagnostic replay requires world_size={expected_world_size}, got {actual_world_size}"
                )
            if int(diagnostic_epoch) != start_epoch:
                raise RuntimeError(
                    f"Diagnostic epoch must be the first epoch after last.pt: expected {start_epoch}, "
                    f"got {diagnostic_epoch}"
                )
            if int(diagnostic_epoch) > args.max_epochs:
                raise RuntimeError("Diagnostic epoch exceeds the recorded training schedule")
            if not 1 <= int(diagnostic_step) <= len(train_loader):
                raise RuntimeError(
                    f"Diagnostic step must be within [1, {len(train_loader)}], got {diagnostic_step}"
                )
            if verify_fix and int(diagnostic_step) >= len(train_loader):
                raise RuntimeError("Fix verification needs one subsequent batch to verify DDP can continue")
            assert diagnostic_output is not None
            if diagnostic_output.exists():
                raise RuntimeError(f"Refusing to overwrite diagnostic output: {diagnostic_output}")
            if main_process:
                diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            if distributed:
                dist.barrier()

        target_verification = None
        replay_initial_fallback_count = numerics.fallback_count if numerics is not None else 0
        for epoch in range(start_epoch, args.max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_indices = list(train_sampler) if corrected_enabled else []
            train_ds.set_epoch(epoch)
            dev_ds.set_epoch(epoch)

            model.train()
            running = 0.0
            last_grad_norm = None
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(distributed and dist.get_rank() != 0))
            for step, (x, y) in enumerate(pbar, start=1):
                global_step += 1
                assert_patch_batch(x, y, patch_size)
                x = x.to(device)
                y = y.to(device)

                if diagnostic_requested and not verify_fix and epoch == int(diagnostic_epoch) and step == int(diagnostic_step):
                    from reproducibility.core import software_provenance, write_json_new
                    from reproducibility.numerics import (
                        classify_failure,
                        model_state_statistics,
                        run_forward_precision_probes,
                        tensor_statistics,
                    )

                    rank = dist.get_rank() if distributed else 0
                    sampler_indices = list(train_sampler)
                    dataset_index = int(sampler_indices[step - 1])
                    input_stats = tensor_statistics(x)
                    target_stats = tensor_statistics(y)
                    state_stats = model_state_statistics(unwrap_model(model), top_k=numerics_topk)
                    probes = run_forward_precision_probes(
                        unwrap_model(model),
                        x,
                        amp_enabled=use_amp,
                        top_k=numerics_topk,
                    )
                    local_diagnostic = {
                        "rank": rank,
                        "batch": train_ds.describe_index(dataset_index),
                        "input": input_stats,
                        "target": target_stats,
                        "model_state_before_forward": state_stats,
                        "precision_probes": probes,
                        "classification": classify_failure(input_stats, state_stats, probes),
                    }
                    if distributed:
                        rank_diagnostics: list[dict[str, Any] | None] = [None] * dist.get_world_size()
                        dist.all_gather_object(rank_diagnostics, local_diagnostic)
                    else:
                        rank_diagnostics = [local_diagnostic]
                    write_error = None
                    if main_process:
                        assert diagnostic_output is not None
                        try:
                            write_json_new(
                                diagnostic_output,
                                {
                                    "diagnostic_type": "corrected_training_exact_step_replay_v1",
                                    "model": model_name,
                                    "run_dir": str(run_dir),
                                    "resume_checkpoint": str(last_checkpoint_path),
                                    "resume_checkpoint_epoch": resume_checkpoint_epoch,
                                    "resume_checkpoint_global_step": resume_checkpoint_global_step,
                                    "target_epoch": epoch,
                                    "target_step": step,
                                    "target_global_step": global_step,
                                    "optimizer_updates_replayed": step - 1,
                                    "amp_enabled": use_amp,
                                    "world_size": len(rank_diagnostics),
                                    "ranks": rank_diagnostics,
                                    "software_environment": software_provenance(repo_root),
                                },
                            )
                            print(f"Wrote exact-step numerics diagnostic to {diagnostic_output}", flush=True)
                        except Exception as exc:
                            write_error = f"{type(exc).__name__}: {exc}"
                    if distributed:
                        write_status = [write_error]
                        dist.broadcast_object_list(write_status, src=0)
                        write_error = write_status[0]
                    if write_error is not None:
                        raise RuntimeError(f"Could not write exact-step numerics diagnostic: {write_error}")
                    return 0

                activation_snapshot = None
                if activation_monitor is not None:
                    activation_monitor.clear_step()

                if numerics is not None:
                    scale_state_before = scaler.state_dict() if verify_fix and step == int(diagnostic_step) else None
                    logits, used_fallback = numerics.forward(
                        x, context={"epoch": epoch, "step": step, "global_step": global_step},
                        batch=train_ds.batch_identity(epoch_indices[step - 1]),
                        scaler_scale=float(scaler.get_scale()) if use_amp else None,
                    )
                    assert_logits_shape(logits, x)
                    max_abs_logit = float(logits.detach().abs().amax().item())
                    loss = loss_fn(logits.float(), y.float())
                    last_grad_norm = numerics.backward_step(
                        loss, fallback=used_fallback, optimizer=optimizer, scaler=scaler,
                        max_consecutive_overflows=MAX_CONSECUTIVE_AMP_OVERFLOWS,
                        overflow_probe=(lambda: run_fp32_probe(
                            model, loss_fn, x, y, accum_steps=1, top_k=numerics_topk, collective_safe=True
                        )) if args.numerics_debug and args.numerics_debug_fp32_probe else None,
                    )
                    consecutive_amp_overflows = numerics.consecutive_overflows
                else:
                    with autocast_context(enabled=use_amp):
                        logits = model(x)
                        if isinstance(logits, (list, tuple)):
                            raise AssertionError("Unexpected deep supervision outputs in shared training loop")
                        assert_logits_shape(logits, x)
                    max_abs_logit = float(logits.detach().abs().amax().item())
                    if activation_monitor is not None:
                        activation_snapshot = activation_monitor.snapshot(top_k=numerics_topk)
                    if not bool(torch.isfinite(logits).all().item()):
                        diag_path = write_numerics_abort(
                            run_dir,
                            epoch=epoch,
                            step=step,
                            global_step=global_step,
                            reason="non_finite_logits",
                            loss=None,
                            max_abs_logit=max_abs_logit,
                            grad_norm=last_grad_norm,
                            scaler_scale=float(scaler.get_scale()) if use_amp else None,
                            lr=float(optimizer.param_groups[0]["lr"]),
                            debug={"activations": activation_snapshot} if activation_snapshot is not None else None,
                        )
                        raise RuntimeError(
                            f"Non-finite logits detected at epoch={epoch}, step={step}, "
                            f"global_step={global_step}. Diagnostic saved to {diag_path}"
                        )

                    loss = loss_fn(logits.float(), y.float())
                    loss = loss / args.accum_steps
                    if not bool(torch.isfinite(loss).item()):
                        diag_path = write_numerics_abort(
                            run_dir,
                            epoch=epoch,
                            step=step,
                            global_step=global_step,
                            reason="non_finite_loss",
                            loss=float(loss.detach().item()),
                            max_abs_logit=max_abs_logit,
                            grad_norm=last_grad_norm,
                            scaler_scale=float(scaler.get_scale()) if use_amp else None,
                            lr=float(optimizer.param_groups[0]["lr"]),
                            debug={"activations": activation_snapshot} if activation_snapshot is not None else None,
                        )
                        raise RuntimeError(
                            f"Non-finite loss detected at epoch={epoch}, step={step}, "
                            f"global_step={global_step}. Diagnostic saved to {diag_path}"
                        )

                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if step % args.accum_steps == 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        grad_diagnostics = (
                            collect_gradient_diagnostics(model, top_k=numerics_topk) if args.numerics_debug else None
                        )
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                        grad_norm_value = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                        skip_optimizer_step = False
                        if not np.isfinite(grad_norm_value):
                            fp32_probe = None
                            amp_overflow_debug = None
                            if use_amp:
                                scale_before = float(scaler.get_scale())
                                scaler.step(optimizer)
                                scaler.update()
                                scale_after = float(scaler.get_scale())
                                optimizer.zero_grad(set_to_none=True)
                                amp_overflow_recovered = scale_after < scale_before
                                amp_overflow_debug = {
                                    "scale_before": scale_before,
                                    "scale_after": scale_after,
                                    "overflow_recovered_by_scaler": amp_overflow_recovered,
                                    "consecutive_overflows_before_step": consecutive_amp_overflows,
                                }
                                if args.numerics_debug and args.numerics_debug_fp32_probe:
                                    fp32_probe = run_fp32_probe(
                                        model,
                                        loss_fn,
                                        x,
                                        y,
                                        accum_steps=args.accum_steps,
                                        top_k=numerics_topk,
                                    )
                                if amp_overflow_recovered:
                                    consecutive_amp_overflows += 1
                                    amp_overflow_debug["consecutive_overflows_after_step"] = consecutive_amp_overflows
                                    skip_optimizer_step = True
                                    last_grad_norm = None
                                    if main_process and args.numerics_debug:
                                        print(
                                            f"AMP overflow recovered at epoch={epoch}, step={step}, "
                                            f"global_step={global_step}, scale {scale_before} -> {scale_after}"
                                        )
                                    if consecutive_amp_overflows > MAX_CONSECUTIVE_AMP_OVERFLOWS:
                                        debug_payload = {
                                            "activations": activation_snapshot,
                                            "gradients_after_unscale": grad_diagnostics,
                                            "amp_overflow": amp_overflow_debug,
                                        }
                                        if fp32_probe is not None:
                                            debug_payload["fp32_probe"] = fp32_probe
                                        diag_path = write_numerics_abort(
                                            run_dir,
                                            epoch=epoch,
                                            step=step,
                                            global_step=global_step,
                                            reason="persistent_amp_overflow",
                                            loss=float(loss.detach().item()),
                                            max_abs_logit=max_abs_logit,
                                            grad_norm=grad_norm_value,
                                            scaler_scale=scale_after,
                                            lr=float(optimizer.param_groups[0]["lr"]),
                                            debug=debug_payload,
                                        )
                                        raise RuntimeError(
                                            f"Persistent AMP overflow detected at epoch={epoch}, step={step}, "
                                            f"global_step={global_step}. Diagnostic saved to {diag_path}"
                                        )
                                else:
                                    debug_payload = {
                                        "activations": activation_snapshot,
                                        "gradients_after_unscale": grad_diagnostics,
                                        "amp_overflow": amp_overflow_debug,
                                    }
                                    if fp32_probe is not None:
                                        debug_payload["fp32_probe"] = fp32_probe
                                    diag_path = write_numerics_abort(
                                        run_dir,
                                        epoch=epoch,
                                        step=step,
                                        global_step=global_step,
                                        reason="non_finite_grad_norm",
                                        loss=float(loss.detach().item()),
                                        max_abs_logit=max_abs_logit,
                                        grad_norm=grad_norm_value,
                                        scaler_scale=scale_after,
                                        lr=float(optimizer.param_groups[0]["lr"]),
                                        debug=debug_payload,
                                    )
                                    raise RuntimeError(
                                        f"Non-finite gradient norm detected at epoch={epoch}, step={step}, "
                                        f"global_step={global_step}. Diagnostic saved to {diag_path}"
                                    )
                            else:
                                debug_payload = {
                                    "activations": activation_snapshot,
                                    "gradients_after_unscale": grad_diagnostics,
                                }
                                diag_path = write_numerics_abort(
                                    run_dir,
                                    epoch=epoch,
                                    step=step,
                                    global_step=global_step,
                                    reason="non_finite_grad_norm",
                                    loss=float(loss.detach().item()),
                                    max_abs_logit=max_abs_logit,
                                    grad_norm=grad_norm_value,
                                    scaler_scale=None,
                                    lr=float(optimizer.param_groups[0]["lr"]),
                                    debug=debug_payload,
                                )
                                raise RuntimeError(
                                    f"Non-finite gradient norm detected at epoch={epoch}, step={step}, "
                                    f"global_step={global_step}. Diagnostic saved to {diag_path}"
                                )
                        if not skip_optimizer_step:
                            last_grad_norm = grad_norm_value
                            consecutive_amp_overflows = 0
                            if use_amp:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                if verify_fix and step == int(diagnostic_step):
                    target_verification = {
                        "fallback_triggered": used_fallback,
                        "fp32_update_completed": used_fallback,
                        "scaler_state_unchanged": scaler.state_dict() == scale_state_before,
                        "target_global_step": global_step,
                        "fallback_event": numerics.last_event if used_fallback else None,
                        "earlier_replay_fallbacks": numerics.fallback_count - replay_initial_fallback_count - int(used_fallback),
                    }
                if verify_fix and step == int(diagnostic_step) + 1:
                    from reproducibility.core import software_provenance

                    passed = (target_verification["fallback_triggered"]
                              and target_verification["scaler_state_unchanged"]
                              and target_verification["earlier_replay_fallbacks"] == 0
                              and last_grad_norm is not None)
                    report = {
                        "diagnostic_type": "production_amp_fp32_fix_verification_v1",
                        "status": "passed" if passed else "inconclusive_failure_not_reproduced_exactly",
                        "source_run_dir": str(source_run_dir),
                        "source_checkpoint": str(last_checkpoint_path),
                        "source_checkpoint_epoch": resume_checkpoint_epoch,
                        "source_checkpoint_global_step": resume_checkpoint_global_step,
                        "epoch": epoch, "step": int(diagnostic_step),
                        "subsequent_update_completed": last_grad_norm is not None,
                        "fallback_count": numerics.fallback_count,
                        "software_environment": software_provenance(repo_root),
                        **target_verification,
                    }
                    # Reuse the controller's collective-safe writer, outside the
                    # source task. Source config/checkpoints/cache stay read-only.
                    numerics._write("fix_verification.json", report)
                    numerics.save_summary(completed=True)
                    from reproducibility.numerics import collective_all
                    write_ok = True
                    if main_process:
                        try:
                            from reproducibility.core import write_json_new
                            write_json_new(diagnostic_output, report)
                            print(f"Fix verification: {report['status']}; {diagnostic_output}", flush=True)
                        except Exception:
                            write_ok = False
                    if not collective_all(write_ok, device):
                        raise RuntimeError("Could not write fix verification report")
                    return 0 if passed else 2

                running += loss.item() * args.accum_steps
                postfix = {
                    "loss": running / step,
                    "logit_abs_max": max_abs_logit,
                }
                if last_grad_norm is not None:
                    postfix["grad_norm"] = last_grad_norm
                if use_amp:
                    postfix["amp_scale"] = float(scaler.get_scale())
                    if consecutive_amp_overflows > 0:
                        postfix["amp_overflow"] = consecutive_amp_overflows
                pbar.set_postfix(postfix)

            dev_dice = 0.0
            dev_loss = 0.0
            should_stop = False
            validation_error = None
            if epoch % args.val_interval == 0 and (not distributed or dist.get_rank() == 0):
                try:
                    ckpt_path = run_dir / "checkpoints" / "best.pt"
                    model_training_flag_at_val_entry = bool(model.training)
                    model.eval()
                    model_training_flag_after_eval = bool(model.training)
                    case_rows = []
                    subject_ids = []
                    with torch.no_grad():
                        scores = []
                        val_losses = []
                        predictor = unwrap_model(model)
                        best_checkpoint_before_val = get_path_metadata(ckpt_path)
                        for x, y, _sid in tqdm(dev_loader, desc="Valid", leave=False):
                            if x.ndim != 5 or y.ndim != 5:
                                raise AssertionError(
                                    f"Expected validation tensors [B, C, D, H, W], got x={tuple(x.shape)} y={tuple(y.shape)}"
                                )
                            sid = _sid[0] if isinstance(_sid, (list, tuple)) else str(_sid)
                            x = x.to(device)
                            y = y.to(device)
                            logits = sliding_window_inference(
                                x,
                                roi_size=patch_size,
                                sw_batch_size=1,
                                predictor=predictor,
                            )
                            assert_logits_shape(logits, x)
                            if not bool(torch.isfinite(logits).all().item()):
                                raise RuntimeError(
                                    f"Non-finite validation logits at epoch={epoch} for subject={sid}"
                                )
                            val_losses.append(float(loss_fn(logits.float(), y.float()).item()))
                            probs = torch.sigmoid(logits)
                            preds = (probs > 0.5).float()
                            case_dice = dice_score(preds, y)
                            scores.append(case_dice)
                            subject_ids.append(sid)
                            case_rows.append(
                                {
                                    "subject_id": sid,
                                    "gt_voxels": int(y.sum().item()),
                                    "pred_voxels": int(preds.sum().item()),
                                    "dice": case_dice,
                                }
                            )
                        dev_dice = float(np.mean(scores)) if scores else 0.0
                        dev_loss = float(np.mean(val_losses)) if val_losses else 0.0

                    csv_path, meta_path = write_validation_artifacts(
                        run_dir,
                        epoch,
                        case_rows,
                        model_training_flag_at_val_entry=model_training_flag_at_val_entry,
                        model_training_flag_after_eval=model_training_flag_after_eval,
                        num_val_batches=len(dev_loader),
                        subject_ids=subject_ids,
                        validation_checkpoint=None,
                        best_checkpoint_before_val=best_checkpoint_before_val,
                    )
                    print(
                        f"Validation epoch {epoch}: model.training before eval={model_training_flag_at_val_entry}, "
                        f"after eval={model_training_flag_after_eval}, val_batches={len(dev_loader)}, "
                        f"subject_hash={hashlib.sha256(chr(10).join(subject_ids).encode('utf-8')).hexdigest()}, "
                        f"cases_csv={csv_path}, meta_json={meta_path}"
                    )
                    if dev_dice > best_dice:
                        best_dice = dev_dice
                        epochs_without_improvement = 0
                        best_payload = {
                            "model": unwrap_model(model).state_dict(),
                            "epoch": epoch,
                            "best_dice": best_dice,
                        }
                        if corrected_enabled:
                            best_payload["corrected_experiment_metadata"] = corrected_checkpoint_metadata
                        torch.save(best_payload, ckpt_path)
                    else:
                        epochs_without_improvement += 1

                    if args.patience > 0 and epochs_without_improvement >= args.patience:
                        if main_process:
                            print(
                                f"Early stopping at epoch {epoch}: no improvement in dev Dice "
                                f"for {args.patience} validation epochs (best={best_dice:.4f})"
                            )
                        should_stop = True

                except Exception as exc:
                    if not corrected_enabled:
                        raise
                    validation_error = f"{type(exc).__name__}: {exc}"
            if corrected_enabled:
                from reproducibility.numerics import collective_all
                if not collective_all(validation_error is None, device):
                    numerics.fail("validation_failed", {"error": validation_error})

            if distributed:
                stop_tensor = torch.tensor([int(should_stop)], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if not distributed or dist.get_rank() == 0:
                with log_path.open("a") as f:
                    f.write(f"{epoch},{running / max(1, len(train_loader))},{dev_dice},{dev_loss}\n")

            if corrected_enabled:
                local_rng_state = _capture_rng_state()
                if distributed:
                    rng_states_by_rank: list[dict[str, Any] | None] = [None] * dist.get_world_size()
                    dist.all_gather_object(rng_states_by_rank, local_rng_state)
                else:
                    rng_states_by_rank = [local_rng_state]
                if main_process:
                    _atomic_torch_save(
                        {
                            "operational_checkpoint": True,
                            "model": unwrap_model(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "fallback_count": numerics.fallback_count,
                            "epoch": epoch,
                            "best_dice": best_dice,
                            "epochs_without_improvement": epochs_without_improvement,
                            "global_step": global_step,
                            "rng_states_by_rank": rng_states_by_rank,
                            "corrected_experiment_metadata": corrected_checkpoint_metadata,
                        },
                        last_checkpoint_path,
                    )
                if distributed:
                    dist.barrier()

            if should_stop:
                break

        if numerics is not None:
            numerics.save_summary(completed=True)
        return 0
    finally:
        if activation_monitor is not None:
            activation_monitor.close()
        cleanup_distributed()
