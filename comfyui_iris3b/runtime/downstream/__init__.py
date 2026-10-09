"""Iris-3B fine-tuned for downstream tasks: monocular depth and image restoration.

Both tasks run the full Iris-3B transformer in a single forward pass, conditioned
on the embedding of the empty prompt, so no text encoder is loaded. A task
export is a directory holding the files below; the release ships them in the
``depth/`` and ``upscaler/`` folders of the official Iris-3B repository.

    config.yaml               optional model/flow sections plus a ``task`` section
    model.safetensors         FP32 weights of the fine-tuned model
    empty_prompt.safetensors  ``embeddings`` [1, T, L, D] FP32 and ``mask`` [1, T] bool
"""

import logging
from pathlib import Path

import torch

from comfyui_iris3b.runtime.config import inference_config

LOG = logging.getLogger("ComfyUI-Iris3B")


def load_export(
    source: str | Path,
    task: str,
    config_path: str | Path | None = None,
    empty_prompt_path: str | Path | None = None,
) -> tuple[object, dict, dict[str, torch.Tensor], dict]:
    """Load ``(config, task settings, weights, empty prompt)`` from a local export.

    Standard release configs have bundled defaults. A sibling YAML or explicit
    path overrides those defaults for custom exports. The empty-prompt cache is
    loaded from the task folder unless overridden. Model downloads are never
    started by this loader.
    """
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    if task not in {"depth", "restoration"}:
        raise ValueError(f"Unsupported Iris downstream task: {task}")
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Iris {task} local export not found: {path}. Automatic downloads are disabled; "
            "download the official checkpoint manually and select it in Iris3B Model Loader."
        )

    if path.is_dir():
        root = path
        weights_path = root / "model.safetensors"
    else:
        root = path.parent
        weights_path = path

    config_file = Path(config_path).expanduser() if config_path else root / "config.yaml"
    if config_file.is_dir():
        config_file = config_file / "config.yaml"
    if config_file.is_file():
        raw = OmegaConf.to_container(OmegaConf.load(config_file))
    elif config_path:
        raise FileNotFoundError(f"Iris {task} config file not found: {config_file}")
    else:
        # Model/flow defaults are bundled in the typed Iris config. Strict
        # checkpoint loading below still detects architectural incompatibility.
        LOG.warning("No config.yaml for Iris %s export %s; using bundled task defaults", task, path)
        raw = {}

    default_settings = {
        "depth": {"name": "depth"},
        "restoration": {"name": "restoration", "sigma": 0.5, "tile": 1024},
    }
    settings = raw.pop("task", default_settings[task])
    if settings["name"] != task:
        raise ValueError(f"{path} holds a {settings['name']!r} export, not {task!r}")
    prompt_path = Path(empty_prompt_path).expanduser() if empty_prompt_path else root / "empty_prompt.safetensors"
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Iris {task} requires an empty-prompt cache; looked for {prompt_path}. "
            "Select the task's empty_prompt.safetensors or keep it beside the model weights."
        )
    if not weights_path.is_file():
        raise FileNotFoundError(f"Iris {task} model weights not found: {weights_path}")
    prompt = load_file(prompt_path)
    embeddings, mask = prompt.get("embeddings"), prompt.get("mask")
    if embeddings is None or mask is None:
        raise ValueError("Iris empty-prompt cache needs embeddings and mask tensors")
    if embeddings.ndim not in {3, 4} or embeddings.shape[0] != 1:
        raise ValueError("Iris empty-prompt embeddings must be [1,T,D] or [1,T,L,D]")
    if mask.shape != embeddings.shape[:2]:
        raise ValueError("Iris empty-prompt mask must align with embeddings [1,T]")
    if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
        raise ValueError("Iris empty-prompt embeddings must be finite floating-point values")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Iris empty-prompt mask must contain only zero/one values")
    return inference_config(raw, []), settings, load_file(weights_path), prompt

