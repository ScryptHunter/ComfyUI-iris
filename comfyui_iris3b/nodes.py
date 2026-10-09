"""ComfyUI node catalog. Heavy models and optional libraries load on node execution."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .downstream import build_task_model_patcher, load_depth_model, load_restoration_model, task_handle_from_model
from .model_loader import load_iris_model
from .depth_palettes import PALETTES, colorize_normalized
from .noise import iris_reference_noise_node
from .pixels import center_crop_image_to_multiple, image_to_pixel_latent, pixel_latent_to_image
from .sigma_scheduler import img2img_sigma_subset, iris_sigma_grid
from .memory import IRIS_EXECUTION_LOCK, check_interrupted

LOG = logging.getLogger("ComfyUI-Iris3B")

_IRIS_TEXT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
_IRIS_TEXT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_IRIS_HIDDEN_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
_IRIS_TEXT_MAX_LENGTH = 300
_IRIS_QWEN_PAD_ID = 151643
_NO_LOCAL_CHECKPOINT = "No local Iris checkpoint found"
_LOCAL_CHECKPOINT_HINT = (
    "Download the weights manually from the official links in the extension README and place them under "
    "ComfyUI/models/diffusion_models (or the legacy ComfyUI/models/unet folder)."
)


def _model_file_choices():
    """Expose only checkpoint files already present in ComfyUI's model folders."""
    choices = []
    try:
        import folder_paths

        choices.extend(
            name for name in folder_paths.get_filename_list("diffusion_models")
            if str(name).lower().endswith((".safetensors", ".pth", ".pt"))
            and Path(str(name)).name.lower() != "empty_prompt.safetensors"
        )
    except (ImportError, KeyError, AttributeError):  # permits static imports/tests outside ComfyUI
        pass
    choices = list(dict.fromkeys(choices))
    return choices or [_NO_LOCAL_CHECKPOINT]


def _empty_prompt_file_choices():
    """Optional override picker; task exports still auto-load the sibling cache."""
    choices = ["None"]
    try:
        import folder_paths

        choices.extend(
            name for name in folder_paths.get_filename_list("diffusion_models")
            if Path(str(name)).name.lower() == "empty_prompt.safetensors"
        )
    except (ImportError, KeyError, AttributeError):
        pass
    return list(dict.fromkeys(choices))


def _resolve_task_weights(weights: str, custom_path: str = "") -> str:
    source = custom_path.strip() or weights
    if not source or source == _NO_LOCAL_CHECKPOINT:
        raise FileNotFoundError(f"No local Iris checkpoint is installed. {_LOCAL_CHECKPOINT_HINT}")
    # Dropdown filenames resolve through ComfyUI's diffusion_models registry,
    # which also includes the legacy models/unet location.
    if not custom_path.strip() and source.lower().endswith((".safetensors", ".pth", ".pt")) and not Path(source).expanduser().exists():
        try:
            import folder_paths
        except ImportError:
            return source
        try:
            return folder_paths.get_full_path_or_raise("diffusion_models", source)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Local Iris checkpoint not found in ComfyUI models/diffusion_models or models/unet: {source}. "
                f"{_LOCAL_CHECKPOINT_HINT}"
            ) from exc
    return source


def _resolve_optional_model_file(value: str | None) -> str | None:
    if not value or value == "None":
        return None
    try:
        import folder_paths

        return folder_paths.get_full_path_or_raise("diffusion_models", value)
    except (ImportError, KeyError, AttributeError, FileNotFoundError):
        return str(Path(value).expanduser())


def _clip_tokens_for_text(clip, text: str):
    """Get one flat Qwen token sequence from the public Comfy CLIP interface."""
    encoded = clip.tokenize(text, skip_template=True)
    if not isinstance(encoded, dict) or len(encoded) != 1:
        raise ValueError("Iris Comfy-CLIP path requires a single Qwen3-VL text encoder")
    key, chunks = next(iter(encoded.items()))
    if len(chunks) != 1:
        raise ValueError("Comfy split the Qwen prompt into chunks; this backend does not meet the Iris Qwen3-VL token-window contract")
    return key, list(chunks[0])


def _comfy_clip_model_type(clip) -> str | None:
    """Find Comfy's text-model marker across its SD1ClipModel wrappers.

    In supported Comfy versions, ``CLIP.cond_stage_model`` is commonly an
    ``SD1ClipModel`` whose ``clip`` attribute is the name of a child module;
    that child owns ``transformer.model_type``. Other Comfy wrappers expose
    the transformer more directly, so walk only these known wrapper links.
    """
    queue = [getattr(clip, "cond_stage_model", None)]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        model_type = getattr(current, "model_type", None) or getattr(type(current), "model_type", None)
        if model_type:
            return str(model_type)

        for attr in ("transformer", "model"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append(child)

        child = getattr(current, "clip", None)
        if isinstance(child, str):
            child = getattr(current, child, None)
        if child is not None:
            queue.append(child)
    return None


def _encode_with_comfy_clip(clip, prompt: str):
    """Encode Iris's layerwise Qwen conditioning through Comfy's native CLIP object.

    This deliberately uses Comfy's Qwen model and tokenizer but retains Iris's
    own template, layer selection, output axes, fixed sequence length, and mask.
    Stock CLIP Text Encode emits only the active layer and is not a drop-in here.
    """
    if not all(callable(getattr(clip, name, None)) for name in ("tokenize", "encode_from_tokens", "clip_layer")):
        raise TypeError("Connect a ComfyUI Load CLIP output (Qwen3-VL 4B) to this node")

    model_type = _comfy_clip_model_type(clip)
    if model_type != "qwen3vl_4b":
        raise ValueError(
            "Iris conditioning requires a Comfy Load CLIP model identified as Qwen3-VL 4B "
            f"(model_type='qwen3vl_4b'); detected {model_type!r}. Select the Iris Qwen3-VL 4B weights; "
            "8B/other encoders are incompatible."
        )

    prefix_key, prefix_tokens = _clip_tokens_for_text(clip, _IRIS_TEXT_PREFIX)
    prompt_key, prompt_tokens = _clip_tokens_for_text(clip, prompt) if prompt else (prefix_key, [])
    suffix_key, suffix_tokens = _clip_tokens_for_text(clip, _IRIS_TEXT_SUFFIX)
    if prefix_key != prompt_key or prefix_key != suffix_key:
        raise ValueError("Comfy CLIP returned inconsistent tokenizer keys for the prompt template")
    if not suffix_tokens or len(suffix_tokens) >= _IRIS_TEXT_MAX_LENGTH:
        raise ValueError("Iris prompt template leaves no room within its 300-token conditioning window")

    caption_budget = _IRIS_TEXT_MAX_LENGTH - len(suffix_tokens)
    if len(prompt_tokens) > caption_budget:
        LOG.warning(
            "Iris prompt exceeds the Comfy-CLIP caption budget (%d tokens); truncating caption while preserving the assistant suffix",
            caption_budget,
        )
        prompt_tokens = prompt_tokens[:caption_budget]

    valid_tokens = prefix_tokens + prompt_tokens + suffix_tokens
    prefix_length = len(prefix_tokens)
    total_length = prefix_length + _IRIS_TEXT_MAX_LENGTH
    if len(valid_tokens) > total_length:
        raise ValueError("Iris Qwen prompt template exceeded the expected fixed sequence length")
    # Qwen3-VL Comfy tokenizer uses 151643 as its pad token and returns token-weight pairs.
    pad = (_IRIS_QWEN_PAD_ID, 1.0)
    padded_tokens = valid_tokens + [pad] * (total_length - len(valid_tokens))
    tokens = {prefix_key: [padded_tokens]}
    mask = torch.zeros((1, total_length), dtype=torch.bool)
    mask[:, :len(valid_tokens)] = True

    # Comfy records the state at the start of block i, which is the post-block-i
    # hidden state for i >= 1. This matches HF's 1-based hidden_states index.
    comfy_layers = list(_IRIS_HIDDEN_LAYERS)
    with IRIS_EXECUTION_LOCK:
        previous_layer = getattr(clip, "layer_idx", None)
        try:
            clip.clip_layer(comfy_layers)
            result = clip.encode_from_tokens(tokens, return_dict=True)
        finally:
            clip.clip_layer(previous_layer)

    if not isinstance(result, dict) or "cond" not in result:
        raise RuntimeError("Comfy CLIP did not return a conditioning dictionary")
    states = result["cond"]
    if states.ndim != 4 or states.shape[1] != len(_IRIS_HIDDEN_LAYERS):
        raise ValueError(
            "Comfy Qwen3-VL did not return all Iris hidden layers; expected [B,L,T,D] with "
            f"L={len(_IRIS_HIDDEN_LAYERS)}, got {tuple(states.shape)}. Use a compatible Qwen3-VL 4B checkpoint/backend in standard Load CLIP."
        )
    if states.shape[-1] != 2560:
        raise ValueError(f"Iris Qwen3-VL 4B expects hidden size 2560, got {states.shape[-1]}")
    if states.shape[2] < total_length:
        raise ValueError(f"Comfy Qwen sequence is shorter than the Iris window: {states.shape[2]} < {total_length}")

    # Comfy emits [B,L,T,D]; Iris's native path and model contract use [B,T,L,D].
    embeddings = states[:, :, prefix_length:prefix_length + _IRIS_TEXT_MAX_LENGTH]
    embeddings = embeddings.permute(0, 2, 1, 3).contiguous()
    if embeddings.shape[0] != 1:
        raise ValueError("A single Comfy-CLIP prompt must return batch size 1; use Comfy conditioning batching for image batches")
    iris_mask = mask[:, prefix_length:prefix_length + _IRIS_TEXT_MAX_LENGTH].to(embeddings.device)
    # Match upstream before the learned text adapter. A mask alone does not
    # zero states consumed by biased projections and later joint attention.
    embeddings = embeddings.masked_fill(~iris_mask[:, :, None, None], 0)
    return embeddings, iris_mask


def _image_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1).mul(255).round().byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _depth_preview(depth: np.ndarray, invert: bool, percentile_low: float = 2.0, percentile_high: float = 98.0, palette: str = "grayscale") -> torch.Tensor:
    if not np.isfinite(depth).all():
        raise ValueError("Depth predictor returned non-finite values")
    low, high = np.percentile(depth, [percentile_low, percentile_high])
    # Normalize the ORIGINAL prediction once. Negating depth before
    # normalizing and then reversing orientation cancels inversion.
    near_bright = np.clip((high - depth) / max(float(high - low), 1e-6), 0, 1)
    if invert:
        near_bright = 1.0 - near_bright
    return torch.from_numpy(colorize_normalized(near_bright, palette)).contiguous()


class Iris3BModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = _model_file_choices()
        return {"required": {
            "checkpoint": (checkpoints, {
                "default": checkpoints[0],
                "tooltip": "Local checkpoint only; Iris3B Model Loader never downloads weights. See the README for official model links.",
            }),
            "empty_prompt": (_empty_prompt_file_choices(), {"default": "None", "tooltip": "Optional empty_prompt.safetensors for depth/upscaler tasks. None uses the task export's sibling cache."}),
            "attention": (["auto", "sdpa", "torch_flash", "torch_cudnn", "sage2", "sage2pp", "fa3", "fa4"], {"default": "sdpa", "tooltip": "T2I only; per-operation preference with SDPA fallback for unsupported inputs/masks/GQA. Depth/upscaler use their task defaults."}),
            "precision": (["bf16_autocast", "fp32"], {"default": "bf16_autocast", "tooltip": "T2I only. Depth/upscaler use their existing task runtime defaults."}),
            "task": (["t2i", "depth", "upscaler"], {"default": "t2i", "tooltip": "Choose the checkpoint's task. Connect the model output directly to the matching sampler, Depth, or Restore node."}),
            "device_policy": (["cpu_after_run", "keep_on_gpu", "cpu"], {"default": "cpu_after_run", "tooltip": "Depth/upscaler policy. T2I placement is managed by Comfy ModelPatcher."}),
        }}

    RETURN_TYPES = ("MODEL", "IRIS_EMPTY_PROMPT")
    RETURN_NAMES = ("model", "empty_prompt")
    FUNCTION = "load_model"
    CATEGORY = "Iris3B/Generation"

    def load_model(
        self, checkpoint, custom_path="", config_path="", attention="sdpa",
        precision="bf16_autocast", empty_prompt="None", task="t2i", device_policy="cpu_after_run",
    ):
        if task not in {"t2i", "depth", "upscaler"}:
            raise ValueError(f"Unsupported Iris task: {task}")
        prompt_path = _resolve_optional_model_file(empty_prompt)
        source = _resolve_task_weights(checkpoint, custom_path)
        if task != "t2i":
            if task == "depth":
                handle = load_depth_model(source, device_policy, config_path or None, prompt_path)
            else:
                handle = load_restoration_model(source, device_policy, config_path or None, prompt_path)
            return build_task_model_patcher(handle), prompt_path
        patcher, _ = load_iris_model(source, attention, precision, config_path=config_path or None)
        return patcher, prompt_path


class Iris3BCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {
                "default": "", "multiline": True, "dynamicPrompts": True,
                "tooltip": "Text only. Iris has no official reference-image conditioning; do not use images as positive or negative prompt input.",
            }),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Iris3B/Generation"

    def encode(self, clip, prompt):
        embeddings, mask = _encode_with_comfy_clip(clip, prompt)
        return ([[embeddings, {"iris_mask": mask, "iris_format": "BTLD"}]],)


class Iris3BEmptyPixelLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "Iris3B/Generation"

    def generate(self, width, height, batch_size):
        if width % 16 or height % 16:
            raise ValueError("Iris pixel dimensions must be divisible by 16; no implicit resizing is done")
        if width * height < 786432:
            LOG.warning(
                "Iris T2I %sx%s is well below the released checkpoint's native ~1MP range; "
                "visible patch/grid artifacts have been reported at 512px even with reference sampling. "
                "Prefer 1024x1024 or a comparable-area aspect ratio, then resize the final image if needed. "
                "This is a quality advisory, not a shape error; resolution is not changed automatically.",
                width, height,
            )
        return ({"samples": torch.zeros((batch_size, 3, height, width), dtype=torch.float32)},)


class Iris3BPixelDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "Iris3B/Generation"

    def decode(self, latent):
        return (pixel_latent_to_image(latent),)


class Iris3BPixelEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {
            "tooltip": "Low-level full-resolution RGB-to-pixel-LATENT conversion only. For reference-sampler Img2Img with strength control, use Iris3B Img2Img Init.",
        })}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("pixel_latent_low_level",)
    FUNCTION = "encode"
    CATEGORY = "Iris3B/Generation"

    def encode(self, image):
        return (image_to_pixel_latent(image),)


class Iris3BImg2ImgInit:
    """Center-crop to patch alignment, then prepare pixel latent and active sigmas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Pixel-space source; no VAE is used. Automatically center-crops to the nearest dimensions divisible by 16 without interpolation or distortion (at most 15 pixels total per dimension). Images smaller than 16x16 are rejected; aligned images are preserved."}),
            "sigmas": ("SIGMAS", {"tooltip": "Connect the complete schedule from Iris3B Sigma Scheduler."}),
            "strength": ("FLOAT", {
                "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "0 is an exact no-op. Above 0, approximately round(strength × schedule steps) intervals are used; earlier Iris sigmas are skipped. Lower values usually preserve more of the source, but are not a percentage guarantee.",
            }),
        }}

    RETURN_TYPES = ("LATENT", "SIGMAS", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "sigmas", "start_sigma", "active_steps")
    FUNCTION = "prepare"
    CATEGORY = "Iris3B/Generation"

    def prepare(self, image, sigmas, strength):
        aligned_image = center_crop_image_to_multiple(image, 16)
        latent = image_to_pixel_latent(aligned_image)
        active_sigmas = img2img_sigma_subset(sigmas, strength)
        return (
            latent,
            active_sigmas,
            float(active_sigmas[0].detach().cpu()),
            int(active_sigmas.numel() - 1),
        )


class Iris3BSigmaScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 100, "min": 1, "max": 1000}),
            "shift_mode": (["checkpoint", "manual"], {"default": "checkpoint"}),
            "manual_shift": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "schedule"
    CATEGORY = "Iris3B/Generation"

    def schedule(self, model, steps, shift_mode, manual_shift):
        if shift_mode == "manual":
            shift = float(manual_shift)
        else:
            wrapped = getattr(model, "model", None)
            cfg = getattr(wrapped, "iris_config", None)
            if cfg is None:
                raise ValueError("Checkpoint shift mode requires a MODEL created by Iris3B Model Loader")
            shift = float(cfg.flow.shift)
        # Preserve the upstream FP64 grid through SamplerCustomAdvanced so the
        # model adapter can avoid FP32 double-rounding of Iris timesteps.
        return (iris_sigma_grid(int(steps), shift),)


class Iris3BReferenceNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS", {"tooltip": "Connect the exact schedule supplied to the sampler. For Img2Img this must be the strength-trimmed output of Iris3B Img2Img Init."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Fixed seed plus identical source image and settings gives repeatable initial noise."}),
        }}

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "make_noise"
    CATEGORY = "Iris3B/Generation"

    def make_noise(self, sigmas, seed):
        return (iris_reference_noise_node(seed, sigmas),)


class Iris3BDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "depth_model": ("MODEL",),
            "max_side": ("INT", {"default": 1024, "min": 0, "max": 8192, "tooltip": "Inference resolution cap; 0 keeps native size. Output returns to original resolution. Higher values need more VRAM."}),
            "invert": ("BOOLEAN", {"default": False, "tooltip": "Reverse both raw relative depth and preview orientation."}),
            "percentile_low": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 99.9, "step": 0.1, "tooltip": "Preview only: lower normalization percentile. Raw depth is unchanged."}),
            "percentile_high": ("FLOAT", {"default": 98.0, "min": 0.1, "max": 100.0, "step": 0.1, "tooltip": "Preview only: upper normalization percentile; must exceed percentile_low."}),
            "palette": (list(PALETTES), {"default": "grayscale", "tooltip": "Preview only. inferno matches the upstream demo palette; raw depth is unchanged. No extra dependencies."}),
        }}

    RETURN_TYPES = ("IMAGE", "IRIS_DEPTH")
    RETURN_NAMES = ("depth_preview", "relative_log_depth")
    FUNCTION = "predict"
    CATEGORY = "Iris3B/Depth"

    def predict(self, image, depth_model, max_side, invert, percentile_low=2.0, percentile_high=98.0, palette="grayscale"):
        if palette not in PALETTES:
            raise ValueError(f"Unknown depth palette: {palette}")
        if not 0 <= percentile_low < percentile_high <= 100:
            raise ValueError("Depth preview requires 0 <= percentile_low < percentile_high <= 100")
        if max_side < 0:
            raise ValueError("Depth max_side must be nonnegative; 0 keeps native resolution")
        patcher = depth_model if hasattr(depth_model, "load_device") else None
        depth_model = task_handle_from_model(depth_model, "depth")
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Iris depth expects IMAGE [B,H,W,3]")
        pil_images = [_image_to_pil(row) for row in image]
        def execute(predictor):
            results = []
            for pil in pil_images:
                check_interrupted()
                results.append(predictor(pil, max_side=int(max_side)))
            return results

        results = depth_model.run(execute, patcher=patcher)
        raw, preview = [], []
        for depth in results:
            depth = np.asarray(depth, dtype=np.float32)
            raw_depth = -depth if invert else depth
            raw.append(torch.from_numpy(raw_depth.copy()).unsqueeze(0))
            preview.append(_depth_preview(depth, invert=invert, percentile_low=percentile_low, percentile_high=percentile_high, palette=palette))
        return (torch.stack(preview), torch.stack(raw))


class Iris3BRestore4x:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "restorer": ("MODEL",),
            "budget": (["fit_budget", "no_budget"], {"default": "fit_budget", "tooltip": "fit_budget limits inference input to short side 512 / long side 1024 for lower compute/memory; output still uses original input size times output_scale. no_budget keeps native input and may require much more VRAM."}),
            "color_fix": ("BOOLEAN", {"default": True, "tooltip": "Wavelet correction: preserve input color/illumination."}),
            "output_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Final size relative to the ORIGINAL input (e.g. 1.2, 3.1, 4). Model restoration remains trained 4x; final size uses Lanczos resampling. Values above the internal result's scale add interpolated pixels, not learned detail. Lower values do not reduce inference compute."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored",)
    FUNCTION = "restore"
    CATEGORY = "Iris3B/Restoration"

    def restore(self, image, restorer, budget, color_fix, output_scale=4.0, *, budget_short_side=512, budget_long_side=1024):
        output_scale = float(output_scale)
        if not math.isfinite(output_scale) or not 0.1 <= output_scale <= 8.0:
            raise ValueError("Restore output_scale must be finite and between 0.1 and 8.0")
        if budget not in {"fit_budget", "no_budget"}:
            raise ValueError("Unknown restoration budget mode")
        if not 1 <= budget_short_side <= budget_long_side:
            raise ValueError("Restoration budget requires 1 <= short side <= long side")
        patcher = restorer if hasattr(restorer, "load_device") else None
        restorer = task_handle_from_model(restorer, "restoration")
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Iris restoration expects IMAGE [B,H,W,3]; alpha is discarded by Comfy RGB IMAGE")
        pil_images = [_image_to_pil(row) for row in image]
        from comfyui_iris3b.runtime.downstream.restoration import fit_budget

        def execute(model):
            output = []
            for pil in pil_images:
                check_interrupted()
                source = fit_budget(pil, short_side=int(budget_short_side), long_side=int(budget_long_side)) if budget == "fit_budget" else pil
                result = model(source, scale=4.0, color_fix=bool(color_fix))
                restored_size = result.size
                target_size = tuple(max(1, round(side * output_scale)) for side in pil.size)
                if result.size != target_size:
                    result = result.resize(target_size, Image.Resampling.LANCZOS)
                LOG.info("Iris restoration %s: original=%s inference_input=%s restored_4x=%s output=%s output_scale=%s", budget, pil.size, source.size, restored_size, result.size, output_scale)
                pixels = torch.from_numpy(np.asarray(result.convert("RGB"), dtype=np.uint8).copy()).float().div_(255)
                output.append(pixels)
            return output

        results = restorer.run(execute, patcher=patcher)
        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "Iris3BModelLoader": Iris3BModelLoader,
    "Iris3BCLIPTextEncode": Iris3BCLIPTextEncode,
    "Iris3BEmptyPixelLatent": Iris3BEmptyPixelLatent,
    "Iris3BPixelDecode": Iris3BPixelDecode,
    "Iris3BPixelEncode": Iris3BPixelEncode,
    "Iris3BImg2ImgInit": Iris3BImg2ImgInit,
    "Iris3BSigmaScheduler": Iris3BSigmaScheduler,
    "Iris3BReferenceNoise": Iris3BReferenceNoise,
    "Iris3BDepth": Iris3BDepth,
    "Iris3BRestore4x": Iris3BRestore4x,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Iris3BModelLoader": "Iris3B Model Loader",
    "Iris3BCLIPTextEncode": "Iris3B Text Encode (Comfy CLIP)",
    "Iris3BEmptyPixelLatent": "Iris3B Empty Pixel Latent",
    "Iris3BPixelDecode": "Iris3B Pixel Decode",
    "Iris3BPixelEncode": "Iris3B Pixel Encode (low-level)",
    "Iris3BImg2ImgInit": "Iris3B Img2Img Init",
    "Iris3BSigmaScheduler": "Iris3B Sigma Scheduler",
    "Iris3BReferenceNoise": "Iris3B Seeded Sampler Noise",
    "Iris3BDepth": "Iris3B Depth",
    "Iris3BRestore4x": "Iris3B Restore 4x",
}
