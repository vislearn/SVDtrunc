"""Text-to-image sampling with FLUX, and image files."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from svdtrunc.flux.autoencoder import AutoEncoder
from svdtrunc.flux.model import Flux
from svdtrunc.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack


@torch.no_grad()
def encode_prompts(t5: nn.Module, clip: nn.Module, prompts: list[str], batch_size: int = 8) -> tuple[Tensor, Tensor]:
    """T5 sequence embeddings (`[N, 512, 4096]`) and pooled CLIP embeddings (`[N, 768]`) of `prompts`, on the CPU.

    Encoding everything up front lets the text encoders be freed before the transformer is loaded.
    """
    txt, vec = [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        txt.append(t5(batch).cpu())
        vec.append(clip(batch).cpu())
    return torch.cat(txt), torch.cat(vec)


@torch.no_grad()
def render(
    model: Flux,
    ae: AutoEncoder,
    txt: Tensor,
    vec: Tensor,
    seeds: list[int],
    height: int = 1024,
    width: int = 1024,
    steps: int = 50,
    guidance: float = 3.5,
    shift: bool = True,
) -> Tensor:
    """Samples one image per prompt embedding as the BFL reference does; returns `[N, 3, height, width]` in [0, 1].

    `txt` and `vec` come from `encode_prompts`. The defaults are FLUX.1-dev's; FLUX.1-schnell samples 4 steps
    without `shift` (see `svdtrunc.loading.SCHNELL`) and ignores `guidance`.
    """
    if not len(txt) == len(vec) == len(seeds):
        raise ValueError(f"Need one seed per prompt, got {len(txt)} prompts and {len(seeds)} seeds.")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    noise = torch.cat([get_noise(1, height, width, device=device, dtype=dtype, seed=seed) for seed in seeds])
    # `prepare` packs the latents and builds position ids; the embeddings are already computed.
    inputs = prepare(lambda _: txt.to(device), lambda _: vec.to(device), noise, [""] * len(seeds))
    # The text encoders return bfloat16; a float32 model needs float32 inputs.
    inputs["txt"], inputs["vec"] = inputs["txt"].to(dtype), inputs["vec"].to(dtype)
    timesteps = get_schedule(steps, inputs["img"].shape[1], shift=shift)
    latents = unpack(denoise(model, **inputs, timesteps=timesteps, guidance=guidance).float(), height, width)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        images = ae.decode(latents.to(next(ae.parameters()).dtype))
    return (images.float().clamp(-1.0, 1.0) + 1.0) / 2.0


def save_images(images: Tensor, paths: list[Path]) -> None:
    """Writes images in [0, 1] (`[N, 3, H, W]`) as PNG files."""
    for image, path in zip(images, paths, strict=True):
        pixels = (image.permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
        Image.fromarray(pixels).save(path)


def load_images(paths: list[Path]) -> Tensor:
    """Reads image files as `[N, 3, H, W]` in [0, 1]."""
    arrays = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
