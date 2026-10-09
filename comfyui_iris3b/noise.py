"""Seeded Comfy noise object for Iris T2I and pixel-space Img2Img."""

from __future__ import annotations

import torch


class IrisReferenceNoise:
    """Adapt seeded noise to Iris's Comfy CONST initialization.

    Comfy's CONST sampler forms ``sigma0 * noise + (1-sigma0) * latent``. Iris
    T2I starts from raw standard-normal noise, so an unmarked empty latent uses
    noise/sigma0 to cancel Comfy's scaling. Image-originated latents carry a
    marker and always use the raw epsilon, including the important solid-gray
    case where x0 itself is numerically all zeros.
    """

    def __init__(self, seed: int, sigma0: float):
        if not torch.isfinite(torch.tensor(float(sigma0))) or sigma0 < 0.0:
            raise ValueError("Iris sampler noise requires a finite, nonnegative initial sigma")
        self.seed = int(seed)
        self.sigma0 = float(sigma0)

    def generate_noise(self, input_latent: dict) -> torch.Tensor:
        try:
            import comfy.sample
        except ImportError as exc:  # pragma: no cover - host only
            raise RuntimeError("IrisReferenceNoise must run inside ComfyUI") from exc
        latent = input_latent["samples"]
        batch_indices = input_latent.get("batch_index")
        noise = comfy.sample.prepare_noise(latent, self.seed, batch_indices)
        is_img2img = bool(input_latent.get("iris3b_img2img", False))
        if not is_img2img and torch.count_nonzero(latent) == 0:
            if self.sigma0 <= 0.0:
                raise ValueError("T2I requires a positive initial sigma; zero-sigma schedules are Img2Img no-ops only")
            return noise / self.sigma0
        return noise


def iris_reference_noise_node(seed: int, sigmas: torch.Tensor) -> IrisReferenceNoise:
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 1:
        raise ValueError("Expected a nonempty one-dimensional Iris SIGMAS schedule")
    sigma0 = float(sigmas[0].detach().cpu())
    return IrisReferenceNoise(seed, sigma0)
