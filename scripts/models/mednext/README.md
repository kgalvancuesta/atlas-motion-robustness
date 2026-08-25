# Minimal MedNeXt v1 runtime

This directory contains only the MedNeXt v1 architecture required by the
experimental pipeline. The unused nnU-Net training, preprocessing, evaluation,
conversion, documentation, and test trees were removed.

Retained upstream implementation files:

- `nnunet_mednext/network_architecture/mednextv1/MedNextV1.py`
- `nnunet_mednext/network_architecture/mednextv1/blocks.py`
- `nnunet_mednext/network_architecture/mednextv1/create_mednext_v1.py`

The retained code comes from upstream commit
`0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba` at
<https://github.com/MIC-DKFZ/MedNeXt>. Package initializers were reduced and
the standalone CUDA/FLOP demo was removed; the model classes and factory
definitions used by this pipeline are unchanged.

Copyright © German Cancer Research Center (DKFZ), Division of Medical Image
Computing. MedNeXt is distributed under the Apache License 2.0; see
`LICENSE` in this directory.

Please cite:

> Roy, S. et al. MedNeXt: Transformer-driven Scaling of ConvNets for Medical
> Image Segmentation. MICCAI 2023.
