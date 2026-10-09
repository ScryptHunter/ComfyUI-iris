"""Optional SageAttention probes; SDPA is the mandatory conservative fallback."""

from __future__ import annotations

import logging
import threading

import torch

_LOG = logging.getLogger("ComfyUI-Iris3B")
_WARNED: set[str] = set()
_LOCK = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    with _LOCK:
        if key not in _WARNED:
            _WARNED.add(key)
            _LOG.warning(message)


def warn_fallback(backend: str, reason: BaseException | str) -> None:
    _warn_once(f"{backend}:runtime:{reason}", f"Iris attention fallback to PyTorch SDPA ({reason}).")


def _sage_function(backend: str):
    try:
        import sageattention
    except (ImportError, OSError) as exc:
        raise RuntimeError("SageAttention could not be imported") from exc
    if backend == "sage2":
        fn = getattr(sageattention, "sageattn", None)
    elif backend == "sage2pp":
        # SageAttention 2++'s INT8-QK / FP8-PV kernel is the intended fast path.
        fn = getattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda", None)
        if fn is None:
            fn = getattr(sageattention, "sageattn_qk_int8_pv_fp16_cuda", None)
    else:
        fn = None
    if fn is None:
        raise RuntimeError(f"installed SageAttention does not expose a {backend} function")
    return fn


def sage_attention(q, k, v, *, backend: str, attn_mask=None, enable_gqa=False):
    """Call an installed Sage kernel only for validated, mask-free CUDA inputs.

    Sage's public entry points and GQA support vary between package builds. Until
    real Iris GQA correctness is validated on the installed build, 20/5 GQA stays
    on PyTorch SDPA rather than expanding K/V heads or guessing a layout.
    """
    if attn_mask is not None:
        raise RuntimeError("SageAttention mask fallback: arbitrary masks require SDPA")
    if q.device.type != "cuda" or q.device != k.device or q.device != v.device:
        raise RuntimeError("SageAttention requires Q/K/V on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("SageAttention requires matching fp16/bf16 tensors")
    fn = _sage_function(backend)
    if enable_gqa and q.shape[-3] != k.shape[-3]:
        # Do not expand K/V or assume that a particular wheel implements GQA.
        raise RuntimeError("installed SageAttention GQA support is not validated; using SDPA")
    return fn(q, k, v, tensor_layout="HND", is_causal=False)


def select_backend(requested: str, q, k, v, *, attn_mask=None, enable_gqa=False) -> str:
    """Resolve an attention preference for one operation without global patches."""
    if requested not in {"auto", "sdpa", "torch_flash", "torch_cudnn", "sage2", "sage2pp", "fa3", "fa4"}:
        raise ValueError(f"unknown attention backend {requested!r}")
    if requested == "auto":
        if attn_mask is None and q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
            requested = "sage2"
        else:
            return "sdpa"
    if requested.startswith("sage"):
        try:
            _sage_function(requested)
            if attn_mask is not None:
                raise RuntimeError("SageAttention cannot consume this mask; preserving it with SDPA")
            if q.device.type != "cuda" or q.device != k.device or q.device != v.device:
                raise RuntimeError("SageAttention requires Q/K/V on one CUDA device")
            if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(f"SageAttention requires matching fp16/bf16 Q/K/V; got {q.dtype}/{k.dtype}/{v.dtype}")
            if enable_gqa and q.shape[-3] != k.shape[-3]:
                raise RuntimeError("installed SageAttention GQA support is not validated")
            return requested
        except RuntimeError as exc:
            _warn_once(f"{requested}:{exc}", f"Iris attention fallback to PyTorch SDPA ({exc}).")
            return "sdpa"
    return requested
