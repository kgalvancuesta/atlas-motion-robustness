# Authoritative v3 persistent-storage audit

This is a capacity estimate, not a PACE filesystem measurement. It uses the supplied v2 observations, the committed split, local NIfTI headers, and the actual cache consumers/writers. All estimates assume all 45 epochs finish; early stopping can only reduce them. GiB/TiB are binary units. For caution, compare against the smaller **15,000 GB decimal** capacity, despite the reported 15,360 GB quota.

## Cache identity and reuse

- `train_common.PatchDataset.__getitem__` passes the same cache root from the runner to `challenge.load_or_create_validated_cache`. The path is `reconstructed_inputs/<training-role>-<experiment-id>/<preprocessing-version>/<normalization-order>/fold-N/replicate-N/<subject>.npy`.
- Validation uses `VolumeDataset` with the analogous `validation-clean` namespace. Corrected checkpoint-selection samples now contain no augmented copies, so normal training never requests `validation-augmentation` or samples validation artifact recipes.
- Path dimensions: cache root/experiment, training/validation/inference role, clean/artifact challenge, preprocessing version, normalization order, fold, replicate, subject. Sidecars additionally validate source SHA-256, recipe ID, and the reconstructed array SHA-256. Recipe identity includes global seed, namespace, subject, fold, epoch/role identifiers, historical distribution, concrete transform parameters, and shape. Architecture, training-run seed, rank, worker, and patch slot do not enter reconstructed-cache identity.
- In augmented training, directory `replicate-N` means **epoch N** (1–45). Recipe generation itself uses `replicate_id=0` plus `epoch=N`, `sample_role=augmented_copy`, and `training_augmentation` namespace. Different epochs/folds intentionally select different deterministic realizations; allowed no-op recipes remain unchanged.
- Base CNN, MedNeXt, UXNet, and Swin request identical cache paths and validated metadata for a given fold/subject/epoch. The cache reader returns the existing checked array without calling its builder. Sharing is confirmed in both consumer arguments and writer/read logic, not just directory names.
- Standard and augmented regimes share all their original training-clean and validation-clean entries. Augmented caches are used only by augmented training and shared across its four architectures. No 40-fold multiplier applies to a shared reconstruction.
- **Clean volumes already use `replicate-0` at every epoch.** They have no epoch-dependent preprocessing. Clean reconstruction accepts only the raw volume and normalization mode; no fold-fitted state exists. Cross-fold and training/validation clean arrays could be shared numerically, but their current namespaces deliberately retain separate files. We leave this unchanged because capacity is comfortable.
- The full float32 reconstructed/normalized volume is persisted before `extract_patch`. Patch centers still depend on global seed, fold, subject, epoch, patch slot, and operation. No sampling or preprocessing change was made.
- Each entry has `.npy`, `.metadata.json`, and a retained zero-byte lock file. `flock` releases when the file descriptor closes. Atomic `.tmp` files are removed in `finally`; disk/file reservations below include temporary files. No cache deletion or deduplication was added. The new read-only cache option is solely for old-run verification; it never creates an entry or touches a lock.

## Inputs and reconstruction estimates

Committed split: 653 eligible subjects; fold training counts 417, 417, 417, 418, 418; validation 105 each; held-out counts 131, 131, 131, 130, 130. The retained 50% training duplicates are 208, 208, 208, 209, 209: **1,042** in total. Original clean training/validation entries across folds total **2,612**.

Local ATLAS headers: 197×233×189, float32 reconstruction = 34,701,156 bytes (0.032318 GiB). The observed 6.8 GiB for 208 augmented subjects per epoch agrees with this. However, the supplied clean sizes (68+17 GiB for one observed fold) are about five times the local shape-derived values. This discrepancy cannot be resolved locally. The estimates below conservatively charge the larger observed clean footprint **for every fold**, without multiplying clean data by epochs.

| Projection | Derivation | Persistent reconstruction | Files excluding directories |
|---|---|---:|---:|
| Old behavior, 45 epochs × 5 folds | corrected estimate + `(66/39)×45×5` GiB augmented validation | 2,339 GiB / 2.284 TiB | 183,606 |
| Corrected clean validation, all 40 training tasks | `45×1042×(6.8/208) + 2087×(68/417) + 525×(17/105)` GiB | **1,958 GiB / 1.912 TiB** | **148,506** |
| Shape-only corrected cross-check | `(45×1042 + 2612)×34,701,156` bytes | 1,600 GiB | same |

File formulas include all three files per entry: corrected `3×(45×1042+2612)`; old adds `3×45×5×52`. The observed 38,596 files are larger than the 31,986 modeled files for fold 0 through epoch 39. To cover that discrepancy, inflate the main reconstructed-cache count by **25% before the final overall 25% headroom**: 185,633 files.

## Complete pipeline budget

The runner normally keeps checkpoints/final outputs under the project allocation and caches in scratch. The table charges **everything to scratch** as a conservative bound.

| Component | Code-derived scope and conservative allowance | GiB reserved |
|---|---|---:|
| Shared training + clean-validation reconstruction | Larger supplied observed sizes, as above | 1,958.3 |
| Shared ATLAS clean + corrupted inference inputs | 2×653 full volumes, one held-out fold each; actual local-shape estimate 42.2 GiB; allowance exceeds five times that | 256 |
| ATLAS final masks | 4 architectures×2 regimes×653 subjects×2 challenges = 10,448 uint8 NIfTI masks, plus JSON/CSV; 84.4 GiB uncompressed at local shapes; allowance exceeds five times that | 448 |
| All 40 task checkpoints, including temporary checkpoint replacement | Parameter counts: Base CNN 4,805,534; MedNeXt 5,550,882; UXNet 53,005,873; Swin 15,702,979. Best=4 bytes/parameter; last≈12 bytes/parameter for weights+Adam moments. Ten tasks per architecture total≈11.8 GiB | 32 |
| MR-ART reconstructed inputs | Local inventory has 140 complete triplets out of 148 subjects (436 available scans); reserve for 148 complete triplets, 444 scans at 192×256×256. Actual full-triplet float32 cache≤20.82 GiB. Source-hash cache shared across all architectures/regimes/folds; ample size margin | 100 |
| Training logs, fallback journals, ATLAS analysis/figures, MR-ART tables, metadata | Validation produces two small files/task/epoch. Fallbacks append to one JSONL/task, preventing a file per update even if frequent. Completed-task counts also persist in metadata and last.pt | 32 |
| **Implemented authoritative `all` pipeline total** | MR-ART computes masks, probabilities, connected components and LCC statistics in memory; persists cache+CSV/JSON, **no prediction/intermediate volumes** | **2,826.3** |
| Optional historical MR-ART export scripts, extra upper bound | Not called by authoritative `all`. Even if separately requested: 444 scans×(40 fold outputs+8 ensembles)×(float32 probabilities+uint8 masks+uint8 LCC masks) = 1,498.5 GiB uncompressed. NPZ float16/compression would reduce this | **1,536** |
| **Expanded full-run worst-case reservation** | Includes optional historical exports as well | **4,362.3** |

Final **25% headroom** gives:

- Authoritative pipeline: **3,532.9 GiB = 3.45 TiB = 3.79 TB decimal**.
- Expanded bound including optional historical MR-ART exports: **5,452.9 GiB = 5.33 TiB = 5.86 TB decimal**, under **39.1% of 15 TB**.

Conservative files/inodes before final headroom: 185,633 inflated main-cache files + 3,918 ATLAS inference-cache files + 20,896 ATLAS mask/sidecar files + 1,332 MR-ART cache files + 6,000 task/log files + 1,000 analysis outputs + 70,000 optional MR-ART export files + 20,000 directories/temporary files = **308,779**. With 25% headroom: **385,974 inodes**, under **38.6% of 1,000,000**.

Both expanded bounds stay below the requested 50% thresholds. **No independent storage optimization is necessary.** Retain useful augmented reconstructions and existing clean-cache namespaces. Recheck actual quota with mandatory PACE preflight before launching; these bounds assume the old run's scratch cache is removed and no unrelated large scratch workload is added.

## Old-cache removal

After stopping the old job and any readers, and after the retained old-run verification command exits 0 with `status=passed`, it is safe to remove the superseded derived cache:

`/storage/scratch1/1/kcuesta3/atlas_corrected_cache/atlas_corrected_authoritative_20260903_v2/`

The new run uses a distinct v3 cache root and starts fresh. Retain the verification report outside scratch, plus any old checkpoint/provenance you want for audit. Do not delete the parent `atlas_corrected_cache` directory indiscriminately. No deletion is performed by this correction.
