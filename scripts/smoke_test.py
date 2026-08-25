#!/usr/bin/env python3
"""Quick SwinUNETR smoke test for data + training + inference plumbing."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nb
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from models.swin_model import build_swin_model, validate_swin_patch_size
from train_common import (
    assert_logits_shape,
    assert_patch_batch,
    autocast_context,
    binarize_mask,
    build_loss,
    extract_patch,
    get_device,
    list_labeled_samples,
    load_nifti,
    make_grad_scaler,
    normalize_volume,
)

PATCH_SIZE = (128, 128, 128)
DEFAULT_LOSS = "dicece"


def mask_name_from_t1w(t1w_name: str) -> str:
    if "_T1w" in t1w_name:
        return t1w_name.replace("_T1w", "_label-L_desc-T1lesion_mask")
    base = t1w_name.replace(".nii.gz", "").replace(".nii", "")
    return f"{base}_label-L_desc-T1lesion_mask.nii.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick SwinUNETR smoke test for data + training + inference plumbing.")
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/)")
    parser.add_argument("--train_steps", type=int, default=1)
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    train_deriv = data_root / "train" / "derivatives" / "ATLAS"
    test_deriv = data_root / "test" / "derivatives" / "ATLAS"

    validate_swin_patch_size(PATCH_SIZE)

    train_samples = list_labeled_samples(train_deriv)
    t1w_test = sorted(test_deriv.rglob("*_T1w.nii.gz"))

    print(f"Train labeled subjects: {len(train_samples)}")
    print(f"Test T1w count: {len(t1w_test)}")

    if not train_samples or not t1w_test:
        print("Missing expected data files; aborting smoke test")
        return 1

    sample = train_samples[0]
    t1w, affine, _hdr = load_nifti(sample.t1w_path)
    mask, _aff2, _hdr2 = load_nifti(sample.mask_path)
    print(f"Sample T1w: {sample.t1w_path}")
    print(f"Shape: {t1w.shape}, dtype: {t1w.dtype}")
    print(f"Affine[0:2]: {affine[:2].tolist()}")

    device = get_device(0)
    use_amp = torch.cuda.is_available()
    print(f"Device: {device}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"AMP enabled: {use_amp}")

    model = build_swin_model().to(device)
    model.eval()

    t1w_norm = normalize_volume(t1w)
    mask = binarize_mask(mask)
    center = tuple(s // 2 for s in t1w.shape)
    patch = extract_patch(t1w_norm, center, PATCH_SIZE)
    inp = torch.from_numpy(patch[None, None, ...]).float().to(device)
    with torch.no_grad():
        with autocast_context(enabled=use_amp):
            out = model(inp)
        assert_logits_shape(out, inp)
    print(f"Forward pass output shape: {tuple(out.shape)}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-4)
    scaler = make_grad_scaler(enabled=use_amp)
    loss_fn = build_loss(DEFAULT_LOSS)
    y_patch = extract_patch(mask, center, PATCH_SIZE)
    x = torch.from_numpy(patch[None, None, ...]).float().to(device)
    y = torch.from_numpy(y_patch[None, None, ...]).float().to(device)
    assert_patch_batch(x, y, PATCH_SIZE)
    last_loss = None

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=use_amp):
            logits = model(x)
            assert_logits_shape(logits, x)
            loss = loss_fn(logits.float(), y.float())
        if not bool(torch.isfinite(loss).item()):
            print("Loss is not finite during training; smoke test failed")
            return 1

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer_state = "amp"
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer_state = "fp32"
            optimizer.step()
        last_loss = float(loss.item())
        grad_norm_value = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        print(f"Train step {step}: loss={last_loss}, grad_norm={grad_norm_value}, mode={optimizer_state}")

    if last_loss is None or not np.isfinite(last_loss):
        print("Loss is not finite; smoke test failed")
        return 1

    test_path = t1w_test[0]
    data, affine, header = load_nifti(test_path)
    data = normalize_volume(data)
    inp = torch.from_numpy(data[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = sliding_window_inference(inp, roi_size=PATCH_SIZE, sw_batch_size=1, predictor=model)
        assert_logits_shape(logits, inp)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]

    header = header.copy()
    header.set_data_dtype(np.uint8)
    pred_img = nb.Nifti1Image(pred.astype(np.uint8), affine, header)

    out_dir = Path("eval") / "smoke_pred" / "swin"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = mask_name_from_t1w(test_path.name)
    out_path = out_dir / out_name
    nb.save(pred_img, str(out_path))
    print(f"Wrote smoke prediction: {out_path}")
    if not out_path.exists():
        print("Smoke prediction not written")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
