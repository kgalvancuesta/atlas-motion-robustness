# Model implementations

Architecture construction is centralized here; training and inference entry
points import the same builders.

- `base_cnn_model.py`: MONAI 3D U-Net configuration.
- `swin_model.py`: MONAI Swin UNETR configuration.
- `mednext_model.py`: adapter for the retained source under `mednext/`.
- `uxnet_model.py`: adapter for the retained source under `uxnet_3d/`.

See the repository-level `THIRD_PARTY_NOTICES.md` and the retained-source
READMEs for architecture citations, upstream revisions, modifications, and
licenses.