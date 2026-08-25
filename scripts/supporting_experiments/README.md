# Supporting experiments

Paper-specific analyses that are not part of the shared ATLAS experiment training and
inference pipeline live here.

- `atlas_lesion_size_stratification/`: lesion-volume analysis, stratification
  stress testing, deterministic CV generation, and lesion-size failure tables.
- `mrart/`: MR-ART negative-control inference, augmentation validation,
  hallucination/component analyses, and their retained evidence.
- `statistical_inference/`: paper-level statistical inference across ATLAS and
  MR-ART results.

Each experiment keeps executable code under its own `scripts/` directory.
Timestamped CSV, validation, and server-report directories are retained
scientific evidence; their recorded historical paths should not be rewritten.
