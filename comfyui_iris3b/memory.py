"""Small scoped execution lock and device helpers for task-specific models."""

from __future__ import annotations

import logging
import threading

import torch

LOG = logging.getLogger("ComfyUI-Iris3B")
IRIS_EXECUTION_LOCK = threading.RLock()


def check_interrupted():
    """Use Comfy's cancellation signal without owning or changing its queue."""
    try:
        import comfy.model_management
    except ImportError:
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def comfy_device() -> torch.device:
    try:
        import comfy.model_management

        return torch.device(comfy.model_management.get_torch_device())
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_status() -> dict[str, int | str | None]:
    """Return allocator counters only when CUDA is available; never empties cache."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated": None, "reserved": None, "peak": None}
    device = comfy_device()
    return {
        "device": str(device),
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak": int(torch.cuda.max_memory_allocated(device)),
    }
