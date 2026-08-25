#!/usr/bin/env python3
"""Run inference and write BIDS-derivative predictions."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nb
import torch
from monai.inferers import sliding_window_inference

from models.base_cnn_model import build_base_model
from split_utils import load_split_payload, resolve_split_ids
from train_common import normalize_volume, load_nifti, list_labeled_samples


MODEL_NAME_ALIASES = {
    "baseline": "baseline",
    "base_cnn": "baseline",
    "mednext": "mednext",
    "swin": "swin",
    "uxnet": "uxnet",
}
DEFAULT_PREDICTION_DERIVATIVE = "atlas2_prediction"
DEFAULT_CV_SPLITS = "splits/atlas_5fold_lesion_quartile_excluding_known_issues.json"


@dataclass
class PreparedVolume:
    t1w_path: Path
    affine: np.ndarray
    header: nb.Nifti1Header
    cpu_tensor: torch.Tensor


def resolve_prediction_derivative(cli_value: str | None) -> str:
    return cli_value or DEFAULT_PREDICTION_DERIVATIVE


def write_dataset_description(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Name": name,
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [{"Name": name}],
    }
    path.write_text(json.dumps(payload, indent=2))


def mask_name_from_t1w(t1w_name: str) -> str:
    if "_T1w" in t1w_name:
        return t1w_name.replace("_T1w", "_label-L_desc-T1lesion_mask")
    base = t1w_name.replace(".nii.gz", "").replace(".nii", "")
    return f"{base}_label-L_desc-T1lesion_mask.nii.gz"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_volume(t1w_path: Path, *, pin_memory: bool, augmentation=None) -> PreparedVolume:
    data, affine, header = load_nifti(t1w_path)
    data = normalize_volume(data)
    if augmentation is not None:
        # Create a dummy mask (not used for inference output, only to match augmentation API)
        dummy_mask = np.zeros_like(data)
        data, _ = augmentation(data, dummy_mask)
    cpu_tensor = torch.from_numpy(data[None, None, ...]).float()
    if pin_memory:
        cpu_tensor = cpu_tensor.pin_memory()
    return PreparedVolume(t1w_path=t1w_path, affine=affine, header=header, cpu_tensor=cpu_tensor)


def output_path_for_t1w(t1w_path: Path, deriv_root: Path) -> Path:
    parts = t1w_path.parts
    subject = next(p for p in parts if p.startswith("sub-"))
    session = next((p for p in parts if p.startswith("ses-")), "ses-1")
    out_dir = deriv_root / subject / session / "anat"
    out_name = mask_name_from_t1w(t1w_path.name)
    return out_dir / out_name


def save_prediction(pred: np.ndarray, affine: np.ndarray, header: nb.Nifti1Header, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_header = header.copy()
    out_header.set_data_dtype(np.uint8)
    pred_img = nb.Nifti1Image(pred.astype(np.uint8, copy=False), affine, out_header)
    nb.save(pred_img, str(out_path))


def list_test_t1w(test_deriv: Path) -> List[Path]:
    return sorted(test_deriv.rglob("*_T1w.nii.gz"))


def load_run_config(run_dir: Path) -> dict | None:
    config_path = run_dir / "config" / "train_config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


def canonicalize_model_name(model_name: str) -> str:
    try:
        return MODEL_NAME_ALIASES[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_NAME_ALIASES))
        raise ValueError(f"Unsupported model_name '{model_name}'. Supported values: {supported}") from exc


def resolve_model_name(cli_model: str | None, run_config: dict | None) -> str:
    if cli_model is not None:
        return canonicalize_model_name(cli_model)
    if run_config is not None:
        config_model = run_config.get("model_name")
        if config_model:
            return canonicalize_model_name(str(config_model))
    return "baseline"


def resolve_patch_size(cli_patch_size: list[int] | None, run_config: dict | None) -> tuple[int, int, int]:
    if cli_patch_size is not None:
        return tuple(cli_patch_size)
    if run_config is not None:
        config_patch = run_config.get("args", {}).get("patch_size")
        if config_patch is not None:
            return tuple(config_patch)
    return (96, 96, 96)


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "baseline":
        return build_base_model()
    if model_name == "mednext":
        from models.mednext_model import build_mednext_model

        return build_mednext_model()
    if model_name == "swin":
        from models.swin_model import build_swin_model

        return build_swin_model()
    if model_name == "uxnet":
        from models.uxnet_model import build_uxnet_model

        return build_uxnet_model()
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference and write BIDS-derivative predictions.")
    parser.add_argument("--run_dir", required=True, help="Run directory (e.g., runs/base_cnn/run_kfold_01)")
    parser.add_argument(
        "--model",
        default=None,
        choices=["baseline", "mednext", "swin", "uxnet"],
        help="Override the model type. By default this is read from run_dir/config/train_config.json.",
    )
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: run_dir/checkpoints/best.pt)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "val", "heldout", "test"])
    parser.add_argument(
        "--splits_json",
        default=None,
        help=(
            f"Split JSON path for labeled inference (default: {DEFAULT_CV_SPLITS}). "
            "Omit it with '--split test' and no '--cv_fold' to use the public unlabeled test set."
        ),
    )
    parser.add_argument("--cv_fold", type=int, default=None, help="CV fold index for master CV split JSONs")
    parser.add_argument("--train_deriv", default=None)
    parser.add_argument("--test_deriv", default=None)
    parser.add_argument(
        "--prediction_derivative",
        default=None,
        help=f"Prediction derivative folder name (default: {DEFAULT_PREDICTION_DERIVATIVE}).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=None,
        help="Sliding-window ROI size. Defaults to the training patch size recorded in run config, else 96 96 96.",
    )
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--augment", type=str, default=None, metavar="PRESET",
        help="Apply TorchIO augmentation to inference volumes (e.g. motion_consistent). "
             "Only applied when explicitly specified.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    if args.train_deriv is None:
        args.train_deriv = str(data_root / "train" / "derivatives" / "ATLAS")
    if args.test_deriv is None:
        args.test_deriv = str(data_root / "test" / "derivatives" / "ATLAS")

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    run_config = load_run_config(run_dir)
    model_name = resolve_model_name(args.model, run_config)
    patch_size = resolve_patch_size(args.patch_size, run_config)
    if model_name == "mednext":
        from models.mednext_model import validate_mednext_patch_size

        validate_mednext_patch_size(patch_size)
    if model_name == "swin":
        from models.swin_model import validate_swin_patch_size

        validate_swin_patch_size(patch_size)
    if model_name == "uxnet":
        from models.uxnet_model import validate_uxnet_patch_size

        validate_uxnet_patch_size(patch_size)

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Inference patch size: {patch_size}")

    model = build_model(model_name).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    derivative = resolve_prediction_derivative(args.prediction_derivative)
    out_root = run_dir / "preds_bids"
    deriv_root = out_root / "derivatives" / derivative
    deriv_root.mkdir(parents=True, exist_ok=True)
    write_dataset_description(out_root / "dataset_description.json", name=derivative)
    write_dataset_description(deriv_root / "dataset_description.json", name=derivative)

    splits_path = Path(args.splits_json) if args.splits_json else None
    if splits_path is None and (args.split != "test" or args.cv_fold is not None):
        splits_path = Path(DEFAULT_CV_SPLITS)
    split_resolution = None
    split_payload = None
    if splits_path is not None and splits_path.exists():
        split_payload = load_split_payload(splits_path)

    use_public_test = args.split == "test" and args.cv_fold is None and splits_path is None

    if use_public_test:
        t1w_paths = list_test_t1w(Path(args.test_deriv))
        split_resolution = {
            "format": "public_test",
            "fold_index": None,
            "split_key_used": None,
            "split_key_reason": "public test derivative for legacy --split test usage",
        }
    else:
        if split_payload is None:
            raise SystemExit(f"Split JSON not found: {splits_path}")
        ids, split_resolution = resolve_split_ids(split_payload, args.split, cv_fold=args.cv_fold)
        samples = list_labeled_samples(Path(args.train_deriv))
        id_set = {sid if sid.startswith("sub-") else f"sub-{sid}" for sid in ids}
        t1w_paths = [s.t1w_path for s in samples if s.subject in id_set]

    if not t1w_paths:
        raise SystemExit("No T1w images found for inference")

    # Augmentation only applied at inference when explicitly requested via --augment
    augmentation = None
    if args.augment:
        try:
            from torchio_augmentations import get_torchio_augmentation
        except ImportError as exc:
            raise ImportError(
                "TorchIO augmentation was requested, but the required 'torchio' dependency is not installed."
            ) from exc
        augmentation = get_torchio_augmentation(preset=args.augment)
        print(f"Augmentation: {args.augment}")
    else:
        print("Augmentation: none (pass --augment PRESET to enable)")

    print(f"Running inference on {len(t1w_paths)} volumes ({args.split})")
    print(f"Split resolution: {json.dumps(split_resolution, sort_keys=True)}")

    pin_memory = device.type == "cuda"
    use_non_blocking = device.type == "cuda"
    pending_save: tuple[Future[None], Path] | None = None

    with ThreadPoolExecutor(max_workers=1) as load_pool, ThreadPoolExecutor(max_workers=1) as save_pool:
        path_iter = iter(t1w_paths)
        next_path = next(path_iter, None)
        next_volume: Future[PreparedVolume] | None = None
        if next_path is not None:
            next_volume = load_pool.submit(prepare_volume, next_path, pin_memory=pin_memory, augmentation=augmentation)

        while next_volume is not None:
            prepared = next_volume.result()

            next_path = next(path_iter, None)
            if next_path is not None:
                next_volume = load_pool.submit(prepare_volume, next_path, pin_memory=pin_memory, augmentation=augmentation)
            else:
                next_volume = None

            inp = prepared.cpu_tensor.to(device, non_blocking=use_non_blocking)

            with torch.inference_mode():
                logits = sliding_window_inference(
                    inp,
                    roi_size=patch_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                )
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError(f"Non-finite inference logits for {prepared.t1w_path}")
                pred = torch.sigmoid(logits).gt(args.threshold).to(torch.uint8).cpu().numpy()[0, 0]

            out_path = output_path_for_t1w(prepared.t1w_path, deriv_root)
            current_save = save_pool.submit(save_prediction, pred, prepared.affine, prepared.header, out_path)
            if pending_save is not None:
                save_future, save_path = pending_save
                save_future.result()
                print(f"Wrote {save_path}")
            pending_save = (current_save, out_path)

        if pending_save is not None:
            save_future, save_path = pending_save
            save_future.result()
            print(f"Wrote {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
