"""MS-TCN++ temporal action segmentation model.

Lightweight multi-stage refinement architecture for rally detection
with explicit boundary awareness.
"""

from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig

__all__ = [
    "MSTCN",
    "MSTCNConfig",
]
