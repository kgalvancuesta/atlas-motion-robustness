# Model architecture and implementation attribution

This project evaluates published architectures; it does not claim authorship of
the four model families below.

| Model used here | Architecture credit | Implementation used here | Source and license status |
| --- | --- | --- | --- |
| 3D U-Net baseline | O. Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation," MICCAI 2016. The original U-Net lineage is O. Ronneberger et al., MICCAI 2015. | `monai.networks.nets.UNet`, configured in `scripts/models/base_cnn_model.py`. MONAI's implementation includes residual units and is not represented as the authors' original code. | Runtime dependency from [Project MONAI](https://github.com/Project-MONAI/MONAI), Apache License 2.0. No MONAI source is vendored here. |
| MedNeXt-S | S. Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation," MICCAI 2023. | A minimal adapted subset of the official MedNeXt v1 source, pinned and described in `scripts/models/mednext/README.md`. | [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), Apache License 2.0. The upstream license is retained at `scripts/models/mednext/LICENSE`. |
| 3D UX-Net | H. H. Lee, S. Bao, Y. Huo, and B. A. Landman, "3D UX-Net: A Large Kernel Volumetric ConvNet Modernizing Hierarchical Transformer for Medical Image Segmentation," ICLR 2023. | A minimal adapted subset of the official UX-Net encoder/backbone plus MONAI `UnetrBasicBlock`, `UnetrUpBlock`, and `UnetOutBlock`; modifications are documented in `scripts/models/uxnet_3d/README.md`. | The [MASILab/3DUX-Net](https://github.com/MASILab/3DUX-Net) README states that the project is released under the MIT License, but the upstream repository currently does not retain the referenced `LICENSE` file. MONAI components are Apache-2.0. |
| Swin UNETR | A. Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images," BrainLes 2021, published in LNCS 2022. | `monai.networks.nets.SwinUNETR`, configured in `scripts/models/swin_model.py`; no upstream Swin UNETR source is vendored. | Runtime dependency from [Project MONAI](https://github.com/Project-MONAI/MONAI), Apache License 2.0. |

## Framework citation

MONAI supplies the complete 3D U-Net and Swin UNETR implementations, the decoder
blocks in the retained UX-Net implementation, sliding-window inference, and loss
components. Cite:

> M. J. Cardoso et al., "MONAI: An open-source framework for deep learning in
> healthcare," arXiv:2211.02701, 2022.
