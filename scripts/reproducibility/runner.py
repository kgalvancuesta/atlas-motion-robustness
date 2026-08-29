"""Staged production runner for immutable corrected experiments."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .analysis import analyze, make_figures
from .core import (
    ARTIFACT_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
    CompatibilityError,
    ConflictError,
    CorrectedExperimentError,
    read_json,
    sha256_file,
    software_provenance,
    validate_metadata,
    write_json_new,
)
from .experiment import (
    DEFAULT_FOLDS,
    DEFAULT_MODELS,
    DEFAULT_REGIMES,
    definition_path,
    load_challenge,
    load_experiment,
    fold_definition,
    prepare_experiment,
)
from .inference import load_checkpoint, resolve_challenges, run_inference_task
from .mrart import analyze_mrart, run_mrart_task
from .rng import derive_seed

STAGES = ("prepare", "train", "infer", "analyze", "figures", "mrart", "all")


def _parse_strings(values: list[str] | None, defaults: Iterable[str], allowed: Iterable[str], label: str) -> list[str]:
    raw = list(defaults) if not values else [item for value in values for item in value.split(",")]
    result: list[str] = []
    allowed_set = set(allowed)
    for item in raw:
        item = item.strip()
        if item not in allowed_set:
            raise CorrectedExperimentError(f"Unknown {label} {item!r}; allowed={sorted(allowed_set)}")
        if item not in result:
            result.append(item)
    return result


def _parse_ints(values: list[int] | None, defaults: Iterable[int], allowed: Iterable[int], label: str) -> list[int]:
    result = sorted(set(int(value) for value in (list(defaults) if not values else values)))
    invalid = sorted(set(result) - set(allowed))
    if invalid:
        raise CorrectedExperimentError(f"Unknown {label} values {invalid}; allowed={list(allowed)}")
    return result


def _validate_runtime_overrides(args: argparse.Namespace, definition: dict[str, Any]) -> None:
    if args.normalization_mode and args.normalization_mode != definition["configuration"]["normalization_mode"]:
        raise ConflictError("Requested normalization mode conflicts with the immutable experiment; use a new experiment ID")
    if args.epochs and args.epochs != int(definition["configuration"]["max_epochs"]):
        raise ConflictError("Requested epoch override conflicts with the immutable experiment; use a new experiment ID")
    if args.replicate_count and args.replicate_count != int(definition["configuration"]["corruption_replicate_count"]):
        raise ConflictError("Requested replicate count conflicts with the immutable experiment; use a new experiment ID")


def _selected(args: argparse.Namespace, definition: dict[str, Any]) -> tuple[list[str], list[str], list[int], list[int]]:
    models = _parse_strings(
        args.model,
        definition["configuration"]["models"],
        definition["configuration"]["models"],
        "model",
    )
    regimes = _parse_strings(
        args.training_regime,
        definition["configuration"]["training_regimes"],
        definition["configuration"]["training_regimes"],
        "training regime",
    )
    folds = _parse_ints(args.fold, definition["configuration"]["folds"], definition["configuration"]["folds"], "fold")
    replicates = _parse_ints(
        args.replicate,
        definition["challenge"]["replicate_ids"],
        definition["challenge"]["replicate_ids"],
        "replicate",
    )
    return models, regimes, folds, replicates


def _next_log_path(root: Path, model: str, regime: str, fold: int) -> Path:
    directory = root / "logs" / "training" / model / regime / f"fold-{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (candidate := directory / f"attempt-{attempt}.log").exists():
        attempt += 1
    return candidate


def _historical_checkpoint(root: Path, model: str, regime: str, fold: int) -> Path:
    prefix = "run_kfold" if regime == "standard" else "run_DA_kfold"
    return root / model / f"{prefix}_{fold + 1:02d}" / "checkpoints" / "best.pt"


def _training_expected(definition: dict[str, Any], *, model: str, regime: str, fold: int) -> dict[str, Any]:
    run_seed = derive_seed(
        int(definition["seeds"]["global_seed"]),
        "model_initialization",
        model=model,
        fold=int(fold),
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "training_task",
        "experiment_id": definition["experiment_id"],
        "model": model,
        "model_configuration_id": definition["model_configurations"][model]["model_configuration_id"],
        "fold": int(fold),
        "training_regime": regime,
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
        "global_seed": int(definition["seeds"]["global_seed"]),
        "training_run_seed": run_seed,
        "max_epochs": int(definition["configuration"]["max_epochs"]),
        "batch_size_per_gpu": 1,
        "nproc_per_node": int(definition["configuration"]["nproc_per_node"]),
    }


def run_training_task(
    *,
    repo_root: Path,
    root: Path,
    definition_file: Path,
    definition: dict[str, Any],
    data_root: Path,
    split_path: Path,
    model: str,
    regime: str,
    fold: int,
    nproc_per_node: int,
    memory_mode: str,
    cache_root: Path | None,
) -> dict[str, Any]:
    expected = _training_expected(definition, model=model, regime=regime, fold=fold)
    expected["nproc_per_node"] = int(nproc_per_node)
    if nproc_per_node != int(definition["configuration"]["nproc_per_node"]):
        raise ConflictError(
            "The corrected definition fixes the two-GPU execution path. A different process count requires a test experiment ID."
        )
    config = definition["model_configurations"][model]
    run_dir = root / "checkpoints" / model / regime / f"fold-{fold}"
    completion_path = run_dir / "task.metadata.json"
    best_path = run_dir / "checkpoints" / "best.pt"
    if completion_path.exists():
        metadata = read_json(completion_path)
        validate_metadata(metadata, expected, context=str(completion_path))
        if not best_path.exists() or sha256_file(best_path) != metadata.get("checkpoint_sha256"):
            raise ConflictError(f"Completed training checkpoint is missing or changed: {best_path}")
        return {"status": "skipped-compatible", **metadata}

    inventory = {entry["subject_id"]: entry for entry in definition["dataset"]["source_inventory"]}
    split_entry = fold_definition(definition, fold)
    for subject_id in [*split_entry["train_ids"], *split_entry["val_ids"]]:
        source = inventory[subject_id]
        for relative_key, checksum_key in (("relative_path", "sha256"), ("mask_relative_path", "mask_sha256")):
            value = Path(source[relative_key])
            path = value if value.is_absolute() else data_root / value
            if not path.exists() or sha256_file(path) != source[checksum_key]:
                raise CompatibilityError(f"Training source differs from immutable definition: {path}")

    last_path = run_dir / "checkpoints" / "last.pt"
    resume = last_path.exists()
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise ConflictError(
            f"Incomplete training output exists without a resumable corrected checkpoint: {run_dir}. "
            "Refusing to overwrite it."
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        config["entrypoint"],
        "--data_root",
        str(data_root),
        "--splits_json",
        str(split_path),
        "--cv_fold",
        str(fold),
        "--run_dir",
        str(run_dir),
        "--max_epochs",
        str(definition["configuration"]["max_epochs"]),
        "--batch_size",
        "1",
        "--patch_size",
        *[str(value) for value in config["patch_size"]],
        "--patches_per_volume",
        str(config["patches_per_volume"]),
        "--lesion_prob",
        str(config["lesion_probability"]),
        "--lr",
        str(config["learning_rate"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--accum_steps",
        str(config["accumulation_steps"]),
        "--num_workers",
        "4",
        "--pin_memory",
        "--val_interval",
        "1",
        "--patience",
        "0",
        "--seed",
        str(expected["training_run_seed"]),
        "--amp",
        "--numerics_debug",
        "--experiment_definition",
        str(definition_file),
        "--experiment_fold",
        str(fold),
        "--experiment_training_regime",
        regime,
        "--normalization_mode",
        definition["configuration"]["normalization_mode"],
        "--experiment_memory_mode",
        memory_mode,
    ]
    if memory_mode == "high":
        if cache_root is None:
            raise CorrectedExperimentError("Corrected memory-high training requires a cache root")
        command.extend(["--experiment_cache_root", str(cache_root)])
    if regime == "augmented":
        command.extend(["--augment", "motion_consistent", "--augment_frac", str(definition["configuration"]["augment_fraction"])])
    if resume:
        command.append("--corrected_resume")
    log_path = _next_log_path(root, model, regime, fold)
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with log_path.open("x", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=repo_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise CorrectedExperimentError(f"Training failed with exit code {result.returncode}; inspect {log_path}")
    if not best_path.exists():
        raise CorrectedExperimentError(f"Training reported success but no best checkpoint exists: {best_path}")
    import torch

    device = torch.device("cpu")
    load_checkpoint(best_path, definition=definition, model=model, regime=regime, fold=fold, device=device)
    metadata = {
        **expected,
        "checkpoint_path": str(best_path.relative_to(root)),
        "checkpoint_sha256": sha256_file(best_path),
        "operational_resume_checkpoint": str(last_path.relative_to(root)),
        "resumed": resume,
        "command": command,
        "log_path": str(log_path.relative_to(root)),
        "software_environment": software_provenance(repo_root),
    }
    write_json_new(completion_path, metadata)
    return {"status": "completed", **metadata}


def run_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    split_path = Path(args.split_path)
    if args.stage == "prepare":
        mode = args.normalization_mode or "artifact_then_normalize"
        models = _parse_strings(args.model, DEFAULT_MODELS, DEFAULT_MODELS, "model")
        regimes = _parse_strings(args.training_regime, DEFAULT_REGIMES, DEFAULT_REGIMES, "training regime")
        folds = _parse_ints(args.fold, DEFAULT_FOLDS, DEFAULT_FOLDS, "fold")
        root, definition, status = prepare_experiment(
            repo_root=repo_root,
            output_root=output_root,
            experiment_id=args.experiment_id,
            data_root=data_root,
            mrart_root=Path(args.mrart_root) if args.mrart_root else data_root / "mr-art",
            split_path=split_path,
            normalization_mode=mode,
            global_seed=args.global_seed,
            models=models,
            regimes=regimes,
            folds=folds,
            epochs=args.epochs or 45,
            replicate_count=args.replicate_count or 1,
            dataset_authority=args.dataset_authority,
        )
        return [{"stage": "prepare", "status": status, "root": str(root), "definition_hash": definition["definition_hash"]}]

    root, definition = load_experiment(output_root, args.experiment_id)
    _validate_runtime_overrides(args, definition)
    models, regimes, folds, replicates = _selected(args, definition)
    definition_file = definition_path(output_root, args.experiment_id)
    challenge_manifest = load_challenge(root, definition)
    cache_root = Path(args.cache_root) if args.cache_root else None
    historical_checkpoint_root = Path(args.historical_checkpoint_root) if args.historical_checkpoint_root else None
    if historical_checkpoint_root is not None and args.stage != "infer":
        raise ConflictError(
            "--historical-checkpoint-root is an explicit inference-only path. Run analyze/figures separately without it."
        )
    if args.memory_mode == "high":
        if cache_root is None:
            raise CorrectedExperimentError("--memory-mode high requires --cache-root")
        cache_root.mkdir(parents=True, exist_ok=True)
        if not os.access(cache_root, os.W_OK | os.X_OK):
            raise CorrectedExperimentError(f"Cache root is not writable: {cache_root}")

    stages = ["train", "infer", "mrart", "analyze", "figures"] if args.stage == "all" else [args.stage]
    results: list[dict[str, Any]] = []
    for stage in stages:
        if stage == "train":
            for model in models:
                for regime in regimes:
                    for fold in folds:
                        result = run_training_task(
                            repo_root=repo_root,
                            root=root,
                            definition_file=definition_file,
                            definition=definition,
                            data_root=data_root,
                            split_path=split_path,
                            model=model,
                            regime=regime,
                            fold=fold,
                            nproc_per_node=args.nproc_per_node,
                            memory_mode=args.memory_mode,
                            cache_root=cache_root,
                        )
                        results.append({"stage": stage, **result})
        elif stage == "infer":
            challenges = resolve_challenges(definition, args.challenge)
            for model in models:
                for regime in regimes:
                    for fold in folds:
                        for challenge in challenges:
                            for replicate in replicates:
                                result = run_inference_task(
                                    repo_root=repo_root,
                                    root=root,
                                    definition=definition,
                                    challenge_manifest=challenge_manifest,
                                    data_root=data_root,
                                    cache_root=cache_root,
                                    memory_mode=args.memory_mode,
                                    model_name=model,
                                    regime=regime,
                                    fold=fold,
                                    challenge=challenge,
                                    replicate_id=replicate,
                                    sw_batch_size=1,
                                    repeat_check=args.repeat_check,
                                    checkpoint_override=(
                                        _historical_checkpoint(historical_checkpoint_root, model, regime, fold)
                                        if historical_checkpoint_root is not None
                                        else None
                                    ),
                                    checkpoint_generation=(
                                        "historical_phase1" if historical_checkpoint_root is not None else "corrected_experiment_v1"
                                    ),
                                )
                                results.append({"stage": stage, **result})
        elif stage == "analyze":
            results.append(
                {
                    "stage": stage,
                    **analyze(
                        root=root,
                        definition=definition,
                        models=models,
                        regimes=regimes,
                        folds=folds,
                        replicate_ids=replicates,
                    ),
                }
            )
        elif stage == "figures":
            results.append({"stage": stage, **make_figures(root=root, definition=definition)})
        elif stage == "mrart":
            for model in models:
                for regime in regimes:
                    for fold in folds:
                        result = run_mrart_task(
                            repo_root=repo_root,
                            root=root,
                            definition=definition,
                            data_root=data_root,
                            cache_root=cache_root,
                            memory_mode=args.memory_mode,
                            model_name=model,
                            regime=regime,
                            fold=fold,
                            sw_batch_size=1,
                            repeat_check=args.repeat_check,
                            max_scans=args.max_mrart_scans,
                        )
                        results.append({"stage": stage, **result})
            results.append(
                {
                    "stage": "mrart_analyze",
                    **analyze_mrart(
                        root=root,
                        definition=definition,
                        models=models,
                        regimes=regimes,
                        folds=folds,
                    ),
                }
            )
        else:
            raise AssertionError(f"Unhandled stage: {stage}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run immutable corrected ATLAS experiments in stages.")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-root", default="corrected_experiments")
    parser.add_argument("--data-root", default=os.environ.get("ATLAS_DATA_ROOT", "data"))
    parser.add_argument("--mrart-root", default=None)
    parser.add_argument("--split-path", default="splits/atlas_5fold_lesion_quartile_excluding_known_issues.json")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--historical-checkpoint-root",
        default=None,
        help="Explicit inference-only Phase 1 checkpoint root; requires a legacy-normalization experiment definition.",
    )
    parser.add_argument("--memory-mode", choices=["low", "high"], default="low")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument("--training-regime", action="append", default=None)
    parser.add_argument("--normalization-mode", choices=["legacy", "artifact_then_normalize", "normalize_then_artifact"], default=None)
    parser.add_argument("--challenge", default=None, help="Challenge name or immutable challenge ID; defaults to clean and fixed.")
    parser.add_argument("--replicate", action="append", type=int, default=None)
    parser.add_argument("--replicate-count", type=int, default=None, help="Prepare-time immutable count; later must match.")
    parser.add_argument("--epochs", type=int, default=None, help="Prepare-time epoch count; later must match.")
    parser.add_argument("--global-seed", type=int, default=9001)
    parser.add_argument("--dataset-authority", choices=["test", "authoritative"], default="test")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--max-mrart-scans", type=int, default=None, help="Test/preflight filter only; omit for production.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        results = run_stage(args)
    except (CorrectedExperimentError, CompatibilityError, ConflictError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    for result in results:
        print(json.dumps(result, sort_keys=True))
    return 0
