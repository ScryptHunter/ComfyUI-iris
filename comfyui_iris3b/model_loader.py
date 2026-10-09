"""Lazy checkpoint and IrisDiT loading for the ComfyUI MODEL adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .paths import resolve_base_checkpoint

LOG = logging.getLogger("ComfyUI-Iris3B")


def _runtime():
    from comfyui_iris3b.runtime.config import inference_config
    from comfyui_iris3b.runtime.models.dit import IrisDiT
    from comfyui_iris3b.runtime.sampling import load_for_inference

    return inference_config, IrisDiT, load_for_inference


def read_inference_config(source: str | Path):
    """Read only a local exported config; never download model weights here."""
    from omegaconf import OmegaConf

    from comfyui_iris3b.runtime.config import inference_config

    path = Path(source).expanduser()
    if path.is_dir():
        config_path = path / "config.yaml"
    elif path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        config_path = path
    else:
        raise ValueError("Text config source must be a local exported directory or config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.yaml at {config_path}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return inference_config(raw, [])


def load_iris_model(
    checkpoint: str,
    attention: str = "sdpa",
    precision: str = "bf16_autocast",
    config_path: str | Path | None = None,
):
    """Create a Comfy MODEL patcher from official local/HF Iris weights.

    The module is built on ``meta`` and receives checkpoint tensors with
    ``assign=True`` to avoid allocating a second 3B-parameter initialized model.
    Checkpoint tensors stay FP32; BF16 is a scoped inference autocast option.
    """
    if precision not in {"bf16_autocast", "fp32"}:
        raise ValueError("precision must be 'bf16_autocast' or 'fp32'")
    if attention not in {"auto", "sdpa", "torch_flash", "torch_cudnn", "sage2", "sage2pp", "fa3", "fa4"}:
        raise ValueError(f"unsupported attention choice: {attention}")

    path = resolve_base_checkpoint(checkpoint)
    if config_path is None and (path.is_dir() or path.suffix.lower() == ".safetensors") and not (
        (path / "config.yaml").is_file() if path.is_dir() else (path.parent / "config.yaml").is_file()
    ):
        LOG.warning("No sidecar config.yaml for %s; using the bundled Iris-3B inference defaults", path)
    inference_config, IrisDiT, load_for_inference = _runtime()
    raw, weights = load_for_inference(path, config_path=config_path)
    cfg = inference_config(raw, [])
    if cfg.flow.prediction not in {"v", "x"}:
        raise ValueError(f"Unsupported flow.prediction={cfg.flow.prediction!r}; only v/x are supported")
    cfg.model.attn_backend = attention

    with torch.device("meta"):
        dit = IrisDiT(cfg.model)
    result = dit.load_state_dict(weights, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    dit.eval().requires_grad_(False)
    # Released weights are FP32. Explicitly preserve this; no hidden downcast.
    dit.to(device="cpu", dtype=torch.float32)
    del weights

    from .comfy_adapter import build_comfy_model_patcher

    patcher = build_comfy_model_patcher(dit, cfg, precision=precision)
    param_count = sum(p.numel() for p in dit.parameters())
    LOG.info(
        "Loaded Iris DiT: parameters=%s, checkpoint=%s, prediction=%s, shift=%s, attention=%s, precision=%s",
        f"{param_count:,}", path, cfg.flow.prediction, cfg.flow.shift, attention, precision,
    )
    return patcher, cfg
