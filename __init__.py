"""ComfyUI custom-node entry point.

ComfyUI loads each custom-node folder directly by file path and does not add the
folder itself to ``sys.path``. Temporarily expose this folder while importing
the namespaced package; after import, its normal package search path is enough
for all later lazy imports. This keeps the extension independent of the
ComfyUI install directory and avoids a persistent sys.path mutation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_extension_root = str(Path(__file__).resolve().parent)
_inserted_extension_root = _extension_root not in sys.path
if _inserted_extension_root:
    sys.path.insert(0, _extension_root)
try:
    from comfyui_iris3b import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
finally:
    if _inserted_extension_root:
        sys.path.remove(_extension_root)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
