"""Compatibility-gated metrics, paired statistics, and figures."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import (
    ARTIFACT_SCHEMA_VERSION,
    CompatibilityError,
    PREPROCESSING_VERSION,
    CorrectedExperimentError,
    read_json,
    sha256_file,
    stable_hash,
    validate_paired_artifacts,
    write_json_new,
)
from .inference import clean_challenge_id
from .rng import derive_seed


CASE_FIELDS = [
    "experiment_id",
    "model",
    "training_regime",
    "fold",
    "subject_id",
    "challenge_id",
    "corruption_replicate",
    "normalization_mode",
    "preprocessing_version",
    "checkpoint_generation",
    "dice",
    "predicted_voxels",
    "target_voxels",
    "predicted_volume_ml",
    "connected_component_count_26",
    "largest_connected_component_voxels_26",
    "largest_connected_component_ml_26",
    "metadata_path",
]


def _load_requested_cases(
    root: Path,
    definition: dict[str, Any],
    *,
    models: Iterable[str],
    regimes: Iterable[str],
    folds: Iterable[int],
    replicate_ids: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    challenge_ids = [clean_challenge_id(definition), definition["challenge"]["challenge_id"]]
    rows: list[dict[str, Any]] = []
    metadata_values: list[dict[str, Any]] = []
    missing: list[str] = []
    for challenge_id in challenge_ids:
        for model in models:
            for regime in regimes:
                for fold in folds:
                    for replicate in replicate_ids:
                        task = (
                            root
                            / "inference"
                            / challenge_id
                            / model
                            / regime
                            / f"fold-{fold}"
                            / f"replicate-{replicate}"
                            / "task.metadata.json"
                        )
                        if not task.exists():
                            missing.append(str(task))
                            continue
                        task_metadata = read_json(task)
                        for relative in task_metadata.get("case_metadata_paths", []):
                            path = root / relative
                            if not path.exists():
                                raise CompatibilityError(f"Inference task references missing case metadata: {path}")
                            metadata = read_json(path)
                            expected = {
                                "schema_version": ARTIFACT_SCHEMA_VERSION,
                                "experiment_id": definition["experiment_id"],
                                "model": model,
                                "training_regime": regime,
                                "fold": int(fold),
                                "challenge_id": challenge_id,
                                "corruption_replicate": int(replicate),
                                "normalization_mode": definition["configuration"]["normalization_mode"],
                                "preprocessing_version": PREPROCESSING_VERSION,
                                "fold_definition_id": definition["dataset"]["fold_definition_id"],
                            }
                            conflicts = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
                            if conflicts:
                                raise CompatibilityError(f"Incompatible inference artifact {path}: {conflicts}")
                            metrics = metadata["metrics"]
                            rows.append(
                                {
                                    **{key: metadata[key] for key in CASE_FIELDS if key in metadata},
                                    **metrics,
                                    "metadata_path": relative,
                                }
                            )
                            metadata_values.append(metadata)
    if missing:
        raise CorrectedExperimentError(
            f"Analysis requires completed inference tasks; {len(missing)} are missing. First missing: {missing[0]}"
        )
    return rows, metadata_values


def _atomic_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise CorrectedExperimentError(f"Refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_summary(
    values_a: dict[str, list[float]],
    values_b: dict[str, list[float]],
    *,
    global_seed: int,
    comparison_id: str,
    b_minus_a_label: str,
) -> dict[str, Any]:
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise CorrectedExperimentError("SciPy is required for corrected-experiment paired statistics") from exc
    subjects = sorted(set(values_a) & set(values_b))
    if not subjects or set(values_a) != set(values_b):
        raise CompatibilityError(
            f"Paired comparison {comparison_id} has different subject identities: "
            f"a_only={sorted(set(values_a) - set(values_b))[:5]}, b_only={sorted(set(values_b) - set(values_a))[:5]}"
        )
    # Replicates remain nested: average within subject before inferential statistics.
    a = np.asarray([np.mean(values_a[subject]) for subject in subjects], dtype=np.float64)
    b = np.asarray([np.mean(values_b[subject]) for subject in subjects], dtype=np.float64)
    delta = b - a
    if np.all(delta == 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    rng = np.random.default_rng(derive_seed(global_seed, "statistics.bootstrap", comparison_id=comparison_id))
    indices = rng.integers(0, len(subjects), size=(2000, len(subjects)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "comparison_id": comparison_id,
        "delta_definition": b_minus_a_label,
        "subject_count": len(subjects),
        "replicate_handling": "mean within subject before paired inference",
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta_b_minus_a": float(delta.mean()),
        "median_delta_b_minus_a": float(np.median(delta)),
        "bootstrap_95_ci_mean_delta": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value_unadjusted": p_value,
    }


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["wilcoxon_p_value_unadjusted"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index]["wilcoxon_p_value_unadjusted"])
        running = max(running, adjusted)
        rows[index]["wilcoxon_p_value_holm"] = running


def analyze(
    *,
    root: Path,
    definition: dict[str, Any],
    models: list[str],
    regimes: list[str],
    folds: list[int],
    replicate_ids: list[int],
) -> dict[str, Any]:
    rows, metadata_values = _load_requested_cases(
        root,
        definition,
        models=models,
        regimes=regimes,
        folds=folds,
        replicate_ids=replicate_ids,
    )
    input_manifest = sorted(
        {row["metadata_path"]: sha256_file(root / row["metadata_path"]) for row in rows}.items()
    )
    input_manifest_id = stable_hash(input_manifest)
    analysis_dir = root / "metrics"
    stats_path = root / "statistics" / "paired_statistics.json"
    metadata_path = analysis_dir / "analysis.metadata.json"
    expected = {
        "experiment_id": definition["experiment_id"],
        "artifact_type": "analysis",
        "input_manifest_id": input_manifest_id,
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
    }
    if metadata_path.exists():
        existing = read_json(metadata_path)
        conflicts = {key: (existing.get(key), value) for key, value in expected.items() if existing.get(key) != value}
        if conflicts:
            raise CompatibilityError(f"Existing analysis conflicts with current inputs: {conflicts}")
        case_path = analysis_dir / "case_metrics.csv"
        if (
            not stats_path.is_file()
            or sha256_file(stats_path) != existing.get("statistics_sha256")
            or not case_path.is_file()
            or sha256_file(case_path) != existing.get("case_metrics_sha256")
        ):
            raise CompatibilityError("Analysis outputs are missing or changed")
        return {"status": "skipped-compatible", **existing}

    by_key: dict[tuple[str, str, str, int, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, metadata in zip(rows, metadata_values):
        key = (row["model"], row["training_regime"], row["challenge_id"], int(row["fold"]), row["subject_id"])
        by_key[key].append((metadata, float(row["dice"])))

    clean_id = clean_challenge_id(definition)
    corrupted_id = definition["challenge"]["challenge_id"]
    comparisons: list[dict[str, Any]] = []
    global_seed = int(definition["seeds"]["global_seed"])
    for model in models:
        for regime in regimes:
            a_values: dict[str, list[float]] = defaultdict(list)
            b_values: dict[str, list[float]] = defaultdict(list)
            for fold in folds:
                fold_subjects = [entry["subject_id"] for entry in rows if entry["fold"] == fold]
                for subject in sorted(set(fold_subjects)):
                    clean_items = by_key.get((model, regime, clean_id, fold, subject), [])
                    corrupt_items = by_key.get((model, regime, corrupted_id, fold, subject), [])
                    if clean_items and corrupt_items:
                        validate_paired_artifacts([clean_items[0][0], corrupt_items[0][0]], require_challenge=False)
                        a_values[subject].extend(value for _metadata, value in corrupt_items)
                        b_values[subject].extend(value for _metadata, value in clean_items)
            comparisons.append(
                _paired_summary(
                    a_values,
                    b_values,
                    global_seed=global_seed,
                    comparison_id=f"{model}:{regime}:clean-minus-corrupted-dice",
                    b_minus_a_label="clean Dice minus fixed-corruption Dice",
                )
            )

    if set(("standard", "augmented")).issubset(regimes):
        for model in models:
            for challenge_id, challenge_name in ((clean_id, "clean"), (corrupted_id, "fixed_legacy_distribution")):
                standard_values: dict[str, list[float]] = defaultdict(list)
                augmented_values: dict[str, list[float]] = defaultdict(list)
                for fold in folds:
                    subjects = sorted({row["subject_id"] for row in rows if int(row["fold"]) == fold})
                    for subject in subjects:
                        standard_items = by_key.get((model, "standard", challenge_id, fold, subject), [])
                        augmented_items = by_key.get((model, "augmented", challenge_id, fold, subject), [])
                        if standard_items and augmented_items:
                            validate_paired_artifacts([standard_items[0][0], augmented_items[0][0]], require_challenge=True)
                            standard_values[subject].extend(value for _metadata, value in standard_items)
                            augmented_values[subject].extend(value for _metadata, value in augmented_items)
                comparisons.append(
                    _paired_summary(
                        standard_values,
                        augmented_values,
                        global_seed=global_seed,
                        comparison_id=f"{model}:{challenge_name}:augmented-minus-standard-dice",
                        b_minus_a_label="augmented-training Dice minus standard-training Dice",
                    )
                )
    _holm_adjust(comparisons)

    case_path = analysis_dir / "case_metrics.csv"
    _atomic_csv_new(case_path, rows)
    stats_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": definition["experiment_id"],
        "artifact_type": "paired_statistics",
        "input_manifest_id": input_manifest_id,
        "normalization_mode": definition["configuration"]["normalization_mode"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_definition_id": definition["dataset"]["fold_definition_id"],
        "replicate_ids": replicate_ids,
        "replicate_independence_policy": "replicates are nested within subject and never counted as independent subjects",
        "multiple_comparison_adjustment": "Holm across emitted paired comparisons",
        "comparisons": comparisons,
    }
    write_json_new(stats_path, stats_payload)
    metadata = {
        **expected,
        "case_metrics_path": str(case_path.relative_to(root)),
        "case_metrics_sha256": sha256_file(case_path),
        "statistics_path": str(stats_path.relative_to(root)),
        "statistics_sha256": sha256_file(stats_path),
        "case_count": len(rows),
        "models": models,
        "training_regimes": regimes,
        "folds": folds,
        "replicate_ids": replicate_ids,
    }
    write_json_new(metadata_path, metadata)
    return {"status": "completed", **metadata}


def make_figures(*, root: Path, definition: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats_path = root / "statistics" / "paired_statistics.json"
    if not stats_path.exists():
        raise CorrectedExperimentError("Figures require the analyze stage first")
    stats = read_json(stats_path)
    stats_hash = sha256_file(stats_path)
    figure_dir = root / "figures"
    figure_path = figure_dir / "paired_effects.png"
    metadata_path = figure_dir / "figures.metadata.json"
    expected = {
        "experiment_id": definition["experiment_id"],
        "artifact_type": "figures",
        "statistics_sha256": stats_hash,
    }
    if metadata_path.exists():
        existing = read_json(metadata_path)
        conflicts = {key: (existing.get(key), value) for key, value in expected.items() if existing.get(key) != value}
        if (
            conflicts
            or not figure_path.is_file()
            or sha256_file(figure_path) != existing.get("figure_sha256")
        ):
            raise CompatibilityError(f"Existing figure output is incompatible or incomplete: {conflicts}")
        return {"status": "skipped-compatible", **existing}
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparisons = stats["comparisons"]
    labels = [entry["comparison_id"] for entry in comparisons]
    values = [entry["mean_delta_b_minus_a"] for entry in comparisons]
    lower = [value - entry["bootstrap_95_ci_mean_delta"][0] for value, entry in zip(values, comparisons)]
    upper = [entry["bootstrap_95_ci_mean_delta"][1] - value for value, entry in zip(values, comparisons)]
    height = max(4.0, 0.42 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    positions = np.arange(len(labels))
    ax.barh(positions, values, xerr=np.asarray([lower, upper]), color="#315f8c", alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Paired mean Dice difference (95% subject bootstrap CI)")
    ax.set_title(f"Corrected reproducibility paired effects: {definition['experiment_id']}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    metadata = {
        **expected,
        "figure_path": str(figure_path.relative_to(root)),
        "figure_sha256": sha256_file(figure_path),
        "comparison_count": len(comparisons),
    }
    write_json_new(metadata_path, metadata)
    return {"status": "completed", **metadata}
