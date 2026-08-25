# Minimal 3D UX-Net runtime

This package retains only the two 3D UX-Net implementation files used by the
experimental pipeline. They were adapted from MASILab/3DUX-Net at upstream commit
`14ea46b7b4c4980b46aba066aaaa24b1d9c1bb0d`:

- `network_backbone.py`: removed the unused projection head and its unrelated
  framework imports; changed the encoder import to a package-relative import.
- `uxnet_encoder.py`: removed dead comments and the unnecessary `timm`
  stochastic-depth dependency; the paper configuration fixes `drop_path=0`.

Upstream: <https://github.com/MASILab/3DUX-Net>

The upstream README states that 3DUX-Net is released under the MIT License, but
the referenced upstream `LICENSE` file is currently absent. The MONAI decoder
components used by this project are Apache-2.0.

Please cite:

> Lee, H. H., Bao, S., Huo, Y., and Landman, B. A. 3D UX-Net: A Large Kernel
> Volumetric ConvNet Modernizing Hierarchical Transformer for Medical Image
> Segmentation. ICLR 2023.
