"""Conversions between Comfy IMAGE and Iris full-resolution RGB pixel LATENT."""

from __future__ import annotations

import torch


def center_crop_image_to_multiple(image: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Center-crop a Comfy IMAGE [B,H,W,3] to patch-compatible dimensions.

    The crop is an integer tensor slice: it preserves all retained pixel values
    without interpolation or aspect-ratio distortion. For Iris' default patch
    size, at most 15 pixels are removed per dimension in total.
    """
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"Comfy IMAGE must be [B,H,W,3], got {tuple(image.shape)}")
    if multiple <= 0:
        raise ValueError("Crop multiple must be positive")
    height, width = int(image.shape[1]), int(image.shape[2])
    if height < multiple or width < multiple:
        raise ValueError(
            f"Iris Img2Img source image must be at least {multiple}x{multiple}; "
            f"received {width}x{height}"
        )
    target_height = (height // multiple) * multiple
    target_width = (width // multiple) * multiple
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return image[:, top:top + target_height, left:left + target_width, :]


def validate_pixel_latent(samples: torch.Tensor, patch_size: int = 16) -> None:
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"Iris pixel LATENT must be [B,3,H,W], got {tuple(samples.shape)}")
    if patch_size <= 0:
        raise ValueError("Iris patch_size must be positive")
    if samples.shape[-2] % patch_size or samples.shape[-1] % patch_size:
        raise ValueError(f"Iris pixel LATENT dimensions must be divisible by {patch_size}")


def image_to_pixel_latent(image: torch.Tensor, patch_size: int = 16) -> dict:
    """Convert Comfy IMAGE [B,H,W,3] in [0,1] into full-resolution Iris BCHW.

    The marker distinguishes an image-originated, potentially all-zero latent
    (for example a solid 50% gray image) from the zero-filled T2I latent. The
    sampler noise adapter needs that distinction to preserve the correct
    ``(1-sigma)*x0 + sigma*epsilon`` initialization.
    """
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"Comfy IMAGE must be [B,H,W,3], got {tuple(image.shape)}")
    if any(int(size) <= 0 for size in image.shape[:3]):
        raise ValueError("Comfy IMAGE batch, height, and width must be positive")
    if not torch.isfinite(image).all():
        raise ValueError("Comfy IMAGE contains NaN or infinity")
    image = image.to(torch.float32)
    if image.numel() and (float(image.min()) < -1e-6 or float(image.max()) > 1.0 + 1e-6):
        raise ValueError("Comfy IMAGE values must be in [0,1]; normalize the image explicitly")
    samples = image.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous().mul(2).sub(1)
    validate_pixel_latent(samples, patch_size)
    return {"samples": samples, "iris3b_img2img": True}


def pixel_latent_to_image(latent: dict) -> torch.Tensor:
    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or samples.shape[1] != 3:
        shape = tuple(samples.shape) if isinstance(samples, torch.Tensor) else type(samples).__name__
        raise ValueError(f"Iris Pixel Decode requires samples [B,3,H,W], received {shape}")
    return samples.to(torch.float32).add(1).mul(0.5).clamp_(0, 1).permute(0, 2, 3, 1).contiguous()
