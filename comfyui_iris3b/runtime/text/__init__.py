"""Text-encoding interfaces used by the internal Iris inference helpers.

Public ComfyUI prompt encoding is provided by the host's native CLIP nodes;
this package intentionally does not ship a second Transformers-based encoder.
"""

from comfyui_iris3b.runtime.text.base import TextEncoder, TextEncoding

__all__ = ["TextEncoder", "TextEncoding"]

