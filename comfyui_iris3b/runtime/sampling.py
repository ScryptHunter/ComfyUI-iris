"""End-to-end text-to-image generation (used by inference and train-time validation)."""

from pathlib import Path
import pickle

import torch

from comfyui_iris3b.runtime.flow.solver import FlowDPMSolver
from comfyui_iris3b.runtime.models.dit import IrisDiT
from comfyui_iris3b.runtime.text.base import TextEncoder


def load_for_inference(path: str | Path, config_path: str | Path | None = None) -> tuple[dict, dict[str, torch.Tensor]]:
    """``(config dict, weights)`` from a training checkpoint or an exported directory.

    A training ``.pth`` contributes its embedded config and its EMA weights,
    or its raw weights when it was trained without EMA. The checkpoint is
    memory-mapped, so its optimizer state is never read. An exported directory
    holds ``config.yaml`` and ``model.safetensors``. For the standard Iris-3B
    architecture, a missing YAML uses this package's bundled inference
    defaults; a sidecar or explicit ``config_path`` overrides them for custom
    checkpoints. A standalone safetensors file can be selected from ComfyUI's
    ``models/diffusion_models`` (legacy ``models/unet``).
    """
    path = Path(path)
    if path.is_dir():
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        config_file = _resolve_config_file(config_path, path / "config.yaml")
        raw = OmegaConf.to_container(OmegaConf.load(config_file)) if config_file else {}
        return raw, load_file(path / "model.safetensors")
    if path.suffix.lower() == ".safetensors":
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        config_file = _resolve_config_file(config_path, path.parent / "config.yaml")
        raw = OmegaConf.to_container(OmegaConf.load(config_file)) if config_file else {}
        return raw, load_file(path)
    if path.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("Iris weights must be safetensors or a restricted-load .pth/.pt training export")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Legacy Iris checkpoint requires unsafe pickle objects and was rejected. "
            "Only tensor/plain-dict training exports are accepted. Convert a checkpoint "
            "you explicitly trust to safetensors in an isolated environment; do not disable "
            "weights_only or allowlist arbitrary classes in a running ComfyUI."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Iris training checkpoint must contain a plain dictionary")
    if "config" not in payload:
        raise ValueError(f"{path} has no embedded config; it is not a training checkpoint")
    key = "state_dict_ema" if "state_dict_ema" in payload else "state_dict"
    if not isinstance(payload["config"], dict) or key not in payload:
        raise ValueError("Iris training checkpoint needs plain-dict config and state_dict[_ema]")
    if not isinstance(payload[key], dict) or not all(isinstance(v, torch.Tensor) for v in payload[key].values()):
        raise ValueError("Iris state_dict must map weight names to tensors")
    return payload["config"], payload[key]


def _resolve_config_file(config_path: str | Path | None, default_path: Path) -> Path | None:
    """Use an explicit or sibling config when available, else bundled defaults."""
    candidate = Path(config_path).expanduser() if config_path else default_path
    if candidate.is_dir():
        candidate = candidate / "config.yaml"
    if candidate.is_file():
        return candidate
    if config_path:
        raise FileNotFoundError(f"Iris config file not found: {candidate}")
    return None


@torch.no_grad()
def generate(
    model: IrisDiT,
    text_encoder: TextEncoder,
    prompts: list[str],
    height: int,
    width: int,
    steps: int = 100,
    order: int = 2,
    cfg_scale: float = 3.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    shift: float = 4.0,
    negative_prompt: str = "",
    generator: torch.Generator | None = None,
    device: torch.device | str = "cuda",
    noise: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
    prediction: str = "v",
) -> torch.Tensor:
    """Sample [B, C, H, W] images, clamped to [-1, 1].

    Integrator state stays in fp32; the network is evaluated in its own
    parameter dtype. The CFG unconditional is ``text_encoder.null`` of the
    negative prompt, which the encoder formats exactly like a positive prompt
    (an empty negative prompt is the dropout null used in training).
    """
    model_dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=model_dtype)
    cond_mask = encoding.mask.to(device=device)
    uncond, cfg_mask = None, None
    if cfg_scale != 1.0:
        null = text_encoder.null(negative_prompt)
        uncond = null.embeddings.to(device=device, dtype=model_dtype).expand(
            len(prompts), *([-1] * (null.embeddings.ndim - 1))
        )
        null_mask = null.mask.to(device=device).expand(len(prompts), -1)
        # the solver's CFG batch is cat([uncond, cond]); the mask follows suit
        cfg_mask = torch.cat([null_mask, cond_mask], dim=0)

    if noise is None:
        noise = torch.randn(
            len(prompts), model.cfg.in_channels, height, width, generator=generator, device=device
        )
    z = noise.to(device=device, dtype=torch.float32)

    def model_fn(x: torch.Tensor, t_model: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # the solver either passes `cond` itself or the doubled CFG batch
        out = model(x.to(model_dtype), t_model, y, y_mask=cond_mask if y is cond else cfg_mask)
        return out.x.float()

    solver = FlowDPMSolver(
        model_fn,
        num_timesteps=num_train_timesteps,
        cfg_scale=cfg_scale,
        cfg_interval=tuple(cfg_interval),
        prediction=prediction,
    )
    sample = solver.sample(z, cond, uncond, steps=steps, order=order, shift=shift)
    return sample.clamp(-1, 1)

