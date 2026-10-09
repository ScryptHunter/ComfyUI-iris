"""ComfyUI-Iris3B custom nodes.

Importing this package only registers lightweight node classes. Checkpoints,
Transformers, CUDA and optional attention kernels are initialized on demand.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
