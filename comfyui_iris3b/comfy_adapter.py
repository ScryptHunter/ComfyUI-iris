"""ComfyUI-native MODEL/ModelPatcher adapter for IrisDiT."""

from __future__ import annotations

from types import SimpleNamespace
from .memory import check_interrupted

import torch


def scheduled_timestep(sigma, grid, multiplier, fallback):
    """Recover a scheduled timestep from exact unique sigma values only.

    If Comfy passed a sigma in FP32, comparison is exact after quantizing the
    FP64 schedule to FP32. If both values are FP64, retain exact equality at
    source precision. Repeated/quantization-colliding entries and off-grid
    evaluations keep Comfy's fallback mapping; this never performs nearest
    neighbor snapping.
    """
    if grid.ndim != 1 or grid.numel() == 0:
        return fallback
    grid = grid.to(device=sigma.device)
    if sigma.dtype == torch.float64 and grid.dtype == torch.float64:
        sigma_compare = sigma.reshape(-1, 1)
        grid_compare = grid.reshape(1, -1)
    else:
        sigma_compare = sigma.float().reshape(-1, 1)
        grid_compare = grid.float().reshape(1, -1)
    matches = sigma_compare == grid_compare
    unique = matches.sum(dim=1) == 1
    index = matches.to(torch.int64).argmax(dim=1)
    exact = (grid[index].to(torch.float64) * multiplier).to(dtype=fallback.dtype).reshape_as(fallback)
    return torch.where(unique.reshape_as(sigma), exact, fallback)


class IrisPixelLatentFormat:
    """Identity format for RGB pixel-space samples, at full image resolution."""

    scale_factor = 1.0
    latent_channels = 3
    latent_dimensions = 2
    spacial_downscale_ratio = 1
    temporal_downscale_ratio = 1
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    latent_rgb_factors_reshape = None
    taesd_decoder_name = None
    compile_preview = False

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent

    def fix_empty_latent(self, latent):
        return latent


def build_comfy_model_patcher(dit, cfg, *, precision: str = "bf16_autocast"):
    """Wrap an official IrisDiT in Comfy's FLOW + CONST sampling contract.

    We subclass BaseModel so Comfy's ModelPatcher and guider use the supported
    host APIs. Iris receives the already-shifted sigma as ``t * 1000`` exactly
    once; the model output is converted from v to x0 for Comfy's denoiser path.
    """
    try:
        import comfy.conds
        import comfy.model_base
        import comfy.model_management
        import comfy.model_patcher
    except ImportError as exc:  # pragma: no cover - host only
        raise RuntimeError("Iris model loading must run inside a ComfyUI Python environment") from exc

    latent_format = IrisPixelLatentFormat()
    proxy_config = SimpleNamespace(
        unet_config={"disable_unet_model_creation": True, "adm_in_channels": 0},
        latent_format=latent_format,
        manual_cast_dtype=None,
        sampling_settings={
            "shift": float(cfg.flow.shift),
            "multiplier": int(cfg.flow.num_train_timesteps),
            "noise_scale": 1.0,
        },
        optimizations={},
        custom_operations=None,
        memory_usage_factor=1.0,
    )

    class IrisComfyModel(comfy.model_base.BaseModel):
        def __init__(self):
            super().__init__(proxy_config, model_type=comfy.model_base.ModelType.FLOW)
            self.diffusion_model = dit
            self.latent_format = latent_format
            self.iris_config = cfg
            self.iris_prediction = cfg.flow.prediction
            self.iris_precision = precision
            self.iris_task = "t2i"
            self.memory_usage_factor = 1.0
            # BaseModel's generic estimate assumes low-resolution VAE latents.
            # Iris instead receives full-resolution pixels and layerwise text.
            self.memory_usage_factor_conds = ()
            self.eval().requires_grad_(False)

        def get_dtype(self):
            try:
                return next(self.diffusion_model.parameters()).dtype
            except StopIteration:
                return torch.float32

        def get_dtype_inference(self):
            # Keep the FP32 checkpoint contract; autocast is explicitly scoped
            # around the forward below (rather than silently storing BF16).
            return self.get_dtype()

        def memory_required(self, input_shape, cond_shapes=None):
            """Uncalibrated planning allowance, not a measured VRAM bound.

            Count patch tokens rather than treating RGB pixels as VAE latent
            tokens. Account for layerwise text as stored tensors, not images.
            Fused SDPA avoids materializing quadratic attention. Math fallback
            is permitted but may exceed this uncalibrated planning allowance.
            """
            batch, _, height, width = input_shape
            patch = int(self.iris_config.model.patch_size)
            hidden = int(getattr(self.iris_config.model, "hidden_size", 2560))
            tokens = (height // patch) * (width // patch)
            text_bytes = 0
            text_tokens = 0
            for shape in (cond_shapes or {}).get("c_crossattn", ()):
                if len(shape) == 4:
                    text_tokens = max(text_tokens, int(shape[1]))
                    text_bytes += int(shape[0]) * int(shape[1]) * int(shape[2]) * int(shape[3]) * 4
            workspace = int(batch) * (tokens + text_tokens) * hidden * 4 * 64
            return 1024 ** 3 + workspace + 2 * text_bytes

        def extra_conds(self, **kwargs):
            if self.iris_task != "t2i":
                raise ValueError("Depth/upscaler MODEL must connect to Iris3B Depth/Restore 4x, not a T2I sampler")
            # CONDCrossAttn assumes [B,T,D] and pads by repeating tokens. Iris
            # needs [B,T,L,D] + its real-token mask, so use shape-preserving
            # regular conditions. Comfy splits incompatible T lengths rather
            # than dropping or flattening either tensor.
            cross_attn = kwargs.get("cross_attn")
            if cross_attn is None:
                raise ValueError("Iris conditioning is missing its cross_attn tensor")
            mask = kwargs.get("iris_mask")
            if mask is None and getattr(self.diffusion_model, "_adapter_needs_mask", False):
                raise ValueError("Iris masked text adapter requires iris_mask [B,T]")
            out = {"c_crossattn": comfy.conds.CONDRegular(cross_attn)}
            if mask is not None:
                out["iris_mask"] = comfy.conds.CONDRegular(mask.to(torch.bool))
            return out

        def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
            check_interrupted()
            if self.iris_task != "t2i":
                raise ValueError("Depth/upscaler MODEL must connect to its task node, not a T2I sampler")
            if x.ndim != 4 or x.shape[1] != 3:
                raise ValueError(f"Iris expects pixel latent [B,3,H,W], received {tuple(x.shape)}")
            patch = int(self.iris_config.model.patch_size)
            if x.shape[-2] % patch or x.shape[-1] % patch:
                raise ValueError(f"Iris image dimensions must be divisible by patch_size={patch}")
            if c_crossattn is None:
                raise ValueError("Iris text conditioning is required")

            dtype = self.get_dtype_inference()
            x_in = x.to(dtype=dtype)
            y = c_crossattn.to(device=x.device, dtype=dtype)
            if y.ndim not in (3, 4):
                raise ValueError("Iris text conditioning must be [B,T,D] or [B,T,L,D]")
            mask = kwargs.get("iris_mask")
            if mask is not None:
                mask = mask.to(device=x.device, dtype=torch.bool)
                if mask.ndim != 2 or mask.shape[:2] != y.shape[:2]:
                    raise ValueError(f"Iris mask shape {tuple(mask.shape)} must match [B,T]={tuple(y.shape[:2])}")
                y = y.masked_fill(~mask.reshape(*mask.shape, *([1] * (y.ndim - 2))), 0)
            elif getattr(self.diffusion_model, "_adapter_needs_mask", False):
                raise ValueError("Iris masked text adapter requires iris_mask [B,T]")

            sigma_source = t.to(device=x.device)
            sigma = sigma_source.to(dtype=torch.float32)
            t_model = self.model_sampling.timestep(sigma).float()
            # Keep the FP64 schedule through Comfy's sampler. If its model call
            # receives the corresponding FP32-rounded sigma, recover only an
            # exact unique schedule value; off-grid evaluations keep host time.
            grid = (transformer_options or {}).get("sample_sigmas")
            if isinstance(grid, torch.Tensor) and grid.dtype == torch.float64:
                t_model = scheduled_timestep(sigma_source, grid, int(self.iris_config.flow.num_train_timesteps), t_model)
            use_bf16 = precision == "bf16_autocast" and x.device.type == "cuda"
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                prediction = self.diffusion_model(x_in, t_model, y, y_mask=mask).x
            prediction = prediction.float()
            if self.iris_prediction == "v":
                return self.model_sampling.calculate_denoised(sigma, prediction, x.float())
            if self.iris_prediction == "x":
                return prediction
            raise ValueError(f"Unsupported Iris flow.prediction={self.iris_prediction!r}; expected v or x")

    model = IrisComfyModel()
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    class IrisModelPatcher(comfy.model_patcher.ModelPatcher):
        """Whole-model host placement for upstream plain torch.nn layers.

        These layers have no Comfy cast-on-forward hooks. Partial placement
        would leave ordinary Linear weights on CPU with CUDA activations.
        Keep normal patch/clone/detach handling, but require full loads and
        decline partial unloads so host management falls back to full detach.
        """

        def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
            return super().load(
                device_to=device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                full_load=True,
            )

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            return 0

    patcher = IrisModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        size=size_bytes,
    )
    return patcher
