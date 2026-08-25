"""Minimal MedNeXt v1 architecture subset used by the ATLAS pipeline.

This initializer was reduced from upstream so importing the architecture does
not require the unused nnU-Net training stack.
"""

from .network_architecture.mednextv1.MedNextV1 import MedNeXt
from .network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock
from .network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1

__all__ = [
    "MedNeXt",
    "MedNeXtBlock",
    "MedNeXtDownBlock",
    "MedNeXtUpBlock",
    "create_mednext_v1",
]
