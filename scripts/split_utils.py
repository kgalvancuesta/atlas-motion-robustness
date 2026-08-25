#!/usr/bin/env python3
"""Helpers for resolving legacy and CV split JSON formats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_subject_id(raw: str) -> str:
    value = str(raw).strip()
    return value if value.startswith("sub-") else f"sub-{value}"


def normalize_subject_ids(values: list[str]) -> list[str]:
    return [normalize_subject_id(value) for value in values]


def load_split_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def is_cv_split_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("folds"), list)


def _require_list(payload: dict[str, Any], key: str, *, context: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        raise SystemExit(f"ERROR: split key '{key}' is missing in {context}.")
    if not isinstance(value, list):
        raise SystemExit(f"ERROR: split key '{key}' is not a list in {context}.")
    return normalize_subject_ids([str(item) for item in value])


def resolve_cv_fold_entry(payload: dict[str, Any], cv_fold: int | None) -> dict[str, Any]:
    if not is_cv_split_payload(payload):
        raise SystemExit("ERROR: --cv_fold was provided, but the split JSON is not in CV master format.")
    if cv_fold is None:
        raise SystemExit("ERROR: this split JSON is a CV master file, so --cv_fold is required.")
    folds = payload.get("folds", [])
    matches = [fold for fold in folds if int(fold.get("fold_index", -1)) == int(cv_fold)]
    if len(matches) != 1:
        available = sorted(
            int(fold.get("fold_index", -1))
            for fold in folds
            if isinstance(fold, dict) and "fold_index" in fold
        )
        raise SystemExit(
            "ERROR: could not resolve CV fold from split JSON.\n"
            f"  requested cv_fold: {cv_fold}\n"
            f"  available fold_index values: {available}"
        )
    return matches[0]


def resolve_split_ids(
    payload: dict[str, Any],
    requested_split: str,
    *,
    cv_fold: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if is_cv_split_payload(payload):
        fold = resolve_cv_fold_entry(payload, cv_fold)
        aliases = {
            "train": ["train_ids"],
            "dev": ["val_ids", "dev_ids"],
            "val": ["val_ids", "dev_ids"],
            "test": ["test_ids", "heldout_ids"],
            "heldout": ["test_ids", "heldout_ids"],
        }
        candidate_keys = aliases.get(requested_split, [f"{requested_split}_ids"])
        for key in candidate_keys:
            if key in fold:
                ids = _require_list(fold, key, context=f"CV fold {cv_fold}")
                return ids, {
                    "format": "cv",
                    "fold_index": int(cv_fold),
                    "split_key_used": key,
                    "split_key_reason": f"CV fold {cv_fold}: key '{key}'",
                }
        raise SystemExit(
            "ERROR: could not resolve requested split from CV fold.\n"
            f"  requested split: {requested_split}\n"
            f"  cv_fold: {cv_fold}\n"
            f"  available keys in fold: {sorted(fold.keys())}"
        )

    if cv_fold is not None:
        raise SystemExit("ERROR: --cv_fold was provided, but the split JSON is not a CV master file.")

    direct = f"{requested_split}_ids"
    if direct in payload:
        return _require_list(payload, direct, context="legacy split JSON"), {
            "format": "legacy",
            "fold_index": None,
            "split_key_used": direct,
            "split_key_reason": "direct '<split>_ids' key",
        }
    if requested_split == "test" and "heldout_ids" in payload:
        return _require_list(payload, "heldout_ids", context="legacy split JSON"), {
            "format": "legacy",
            "fold_index": None,
            "split_key_used": "heldout_ids",
            "split_key_reason": "fallback: test -> heldout_ids (test_ids missing)",
        }
    if requested_split == "val" and "dev_ids" in payload:
        return _require_list(payload, "dev_ids", context="legacy split JSON"), {
            "format": "legacy",
            "fold_index": None,
            "split_key_used": "dev_ids",
            "split_key_reason": "fallback: val -> dev_ids (val_ids missing)",
        }
    raise SystemExit(
        "ERROR: could not resolve requested split.\n"
        f"  requested split: {requested_split}\n"
        f"  available keys: {sorted(payload.keys())}"
    )


def resolve_training_split_ids(
    payload: dict[str, Any],
    *,
    cv_fold: int | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    train_ids, train_meta = resolve_split_ids(payload, "train", cv_fold=cv_fold)
    val_ids, val_meta = resolve_split_ids(payload, "val", cv_fold=cv_fold)
    meta = {
        "format": train_meta["format"],
        "fold_index": train_meta["fold_index"],
        "train": train_meta,
        "val": val_meta,
    }
    return train_ids, val_ids, meta
