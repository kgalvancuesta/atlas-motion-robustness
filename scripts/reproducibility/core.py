"""Immutable definitions, provenance, hashing, and compatibility checks."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "atlas.corrected.experiment.v1"
CHALLENGE_SCHEMA_VERSION = "atlas.corrected.challenge.v1"
ARTIFACT_SCHEMA_VERSION = "atlas.corrected.artifact.v1"
PREPROCESSING_VERSION = "atlas.corrected.preprocessing.v1"
NORMALIZATION_MODES = ("legacy", "artifact_then_normalize", "normalize_then_artifact")
REQUIRED_COMPATIBILITY_FIELDS = (
    "experiment_id",
    "fold_definition_id",
    "subject_ids_hash",
    "normalization_mode",
    "preprocessing_version",
    "model",
    "model_configuration_id",
    "checkpoint_generation",
)


class CorrectedExperimentError(RuntimeError):
    """Base error for explicit corrected-experiment validation failures."""


class ConflictError(CorrectedExperimentError):
    """Raised when an immutable artifact conflicts with a request."""


class CompatibilityError(CorrectedExperimentError):
    """Raised before comparing scientifically incompatible artifacts."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    header = canonical_json_bytes({"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)})
    digest = hashlib.sha256(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def subject_ids_hash(subject_ids: Iterable[str]) -> str:
    return stable_hash(sorted(str(value) for value in subject_ids))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedExperimentError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorrectedExperimentError(f"Expected a JSON object in {path}")
    return value


def _atomic_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ConflictError(f"Refusing to overwrite immutable JSON: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Create JSON once; an identical existing artifact is a valid resume skip."""
    requested_hash = stable_hash(payload)
    if path.exists():
        existing = read_json(path)
        existing_hash = stable_hash(existing)
        if existing_hash != requested_hash:
            raise ConflictError(
                f"Immutable artifact conflict at {path}: existing={existing_hash}, requested={requested_hash}. "
                "Use a new experiment ID."
            )
        return "skipped-compatible"
    _atomic_json_create(path, payload)
    return "created"


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ConflictError(f"Refusing to overwrite existing artifact: {path}")
    _atomic_json_create(path, payload)


def write_json_atomic_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace mutable status/cache metadata atomically; never use for definitions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "revision": revision,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def software_provenance(repo_root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "nibabel", "torch", "torchio", "monai", "scipy"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = None
    cuda: dict[str, Any] = {"available": False, "device_count": 0}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
        }
    except Exception:
        pass
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda": cuda,
        "repository": git_provenance(repo_root),
    }


def validate_normalization_mode(mode: str) -> str:
    if mode not in NORMALIZATION_MODES:
        raise CorrectedExperimentError(f"Unsupported normalization mode {mode!r}; choose from {NORMALIZATION_MODES}")
    return mode


def configure_strict_determinism(seed: int) -> dict[str, Any]:
    """Configure strict runtime determinism without a warn-only escape hatch."""
    import random

    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "seed": int(seed),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if torch.cuda.is_available() else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def validate_metadata(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    conflicts = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if conflicts:
        raise CompatibilityError(f"Metadata conflict for {context}: {json.dumps(conflicts, sort_keys=True)}")


def validate_paired_artifacts(
    artifacts: list[Mapping[str, Any]],
    *,
    require_challenge: bool = True,
    allow_differences: Iterable[str] = (),
) -> None:
    if len(artifacts) < 2:
        raise CompatibilityError("Paired comparison requires at least two artifacts")
    fields = list(REQUIRED_COMPATIBILITY_FIELDS)
    for field in allow_differences:
        if field in fields:
            fields.remove(field)
    fields.extend(["subject_ids"])
    if require_challenge:
        fields.extend(["challenge_id", "corruption_replicate_ids"])
    reference = artifacts[0]
    for index, artifact in enumerate(artifacts[1:], start=1):
        mismatches = {
            field: {"reference": reference.get(field), "artifact": artifact.get(field)}
            for field in fields
            if artifact.get(field) != reference.get(field)
        }
        if mismatches:
            raise CompatibilityError(
                f"Paired artifact {index} is scientifically incompatible: {json.dumps(mismatches, sort_keys=True)}"
            )


def completed_artifact_status(metadata_path: Path, expected: Mapping[str, Any], outputs: Iterable[Path]) -> str:
    """Return missing/complete; fail on partial or conflicting existing outputs."""
    output_paths = list(outputs)
    existing_outputs = [path for path in output_paths if path.exists()]
    if not metadata_path.exists() and not existing_outputs:
        return "missing"
    if not metadata_path.exists() or len(existing_outputs) != len(output_paths):
        raise ConflictError(
            f"Partial output exists for {metadata_path}; refusing to overwrite. "
            "Inspect or move the incomplete artifact before resuming."
        )
    metadata = read_json(metadata_path)
    validate_metadata(metadata, expected, context=str(metadata_path))
    for path in output_paths:
        if path.stat().st_size <= 0:
            raise ConflictError(f"Completed metadata references an empty output: {path}")
    return "complete"


def resolve_verified_scratch(path: Path | None = None) -> Path:
    """Resolve PACE scratch without hard-coding its symlink destination."""
    candidate = path or (Path.home() / "scratch")
    if not candidate.exists():
        raise CorrectedExperimentError(f"PACE scratch path does not exist: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir() or not os.access(resolved, os.W_OK | os.X_OK):
        raise CorrectedExperimentError(f"PACE scratch is not a writable directory: {candidate} -> {resolved}")
    if resolved == Path.home().resolve() or str(resolved) == "/":
        raise CorrectedExperimentError(f"Refusing unsafe scratch resolution: {candidate} -> {resolved}")
    return resolved
