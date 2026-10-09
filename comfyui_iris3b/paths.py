"""Local-only checkpoint location helpers.

Model weights are never fetched during node execution. Download them manually
from the official links in the extension README, then select the local file in
the ComfyUI model dropdown.
"""

from __future__ import annotations

import os
from pathlib import Path


_MANUAL_DOWNLOAD_HINT = (
    "Automatic model downloads are disabled. Download the checkpoint manually "
    "from the official links in the ComfyUI-Iris3B README, then place it under "
    "ComfyUI/models/diffusion_models (or the legacy ComfyUI/models/unet folder)."
)


def resolve_base_checkpoint(source: str | os.PathLike[str]) -> Path:
    """Resolve an existing local Iris checkpoint or exported model directory."""
    value = str(source).strip()
    if not value:
        raise ValueError("Select a local Iris checkpoint from the model dropdown")
    local = Path(value).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"Iris checkpoint not found locally: {value}. {_MANUAL_DOWNLOAD_HINT}")
    if local.is_dir():
        if not (local / "model.safetensors").is_file():
            raise FileNotFoundError(f"Iris export directory has no model.safetensors: {local}")
    elif local.suffix.lower() not in {".safetensors", ".pth", ".pt"}:
        raise ValueError("Iris weights must be a local .safetensors, .pth, or .pt checkpoint")
    return local.resolve()


def resolve_downstream_export(source: str | os.PathLike[str], task: str) -> Path:
    """Resolve a local depth/upscaler export directory or safetensors file."""
    if task not in {"depth", "upscaler"}:
        raise ValueError(f"Unsupported Iris downstream task: {task}")
    value = str(source).strip()
    if not value:
        raise ValueError(f"Select a local Iris {task} checkpoint from the model dropdown")
    local = Path(value).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"Iris {task} checkpoint not found locally: {value}. {_MANUAL_DOWNLOAD_HINT}")
    if local.is_file() and local.suffix.lower() != ".safetensors":
        raise ValueError(f"Iris {task} checkpoints must be local .safetensors files")
    return local.resolve()
