"""CMMD (Jayasumana et al., 2024): MMD between CLIP image embeddings, with a Gaussian kernel.

Follows the reference implementation (google-research/cmmd): CLIP ViT-L/14 at 336 px on images resized
bicubically to a square, unit-norm embeddings, kernel bandwidth 10, and a scale of 1000.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from svdtrunc.render import load_images

CLIP_IMAGE_MODEL = "openai/clip-vit-large-patch14-336"
SIGMA = 10.0
SCALE = 1000.0


class ClipImageEmbedder:
    """CLIP image embeddings as CMMD uses them."""

    def __init__(self, model: str = CLIP_IMAGE_MODEL, device: torch.device | str = "cpu"):
        processor = CLIPImageProcessor.from_pretrained(model)
        self.size = processor.crop_size["height"]
        self.mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model).eval().to(device)
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, images: Tensor) -> Tensor:
        """Unit-norm embeddings (`[N, D]`, float64) of images in [0, 1] (`[N, 3, H, W]`)."""
        images = F.interpolate(images.float(), size=(self.size, self.size), mode="bicubic", antialias=True)
        pixels = (images.clamp(0.0, 1.0) - self.mean) / self.std
        embeddings = self.model(pixel_values=pixels.to(self.device)).image_embeds
        return F.normalize(embeddings.double(), dim=-1).cpu()

    def embed_files(self, paths: list[Path], batch_size: int = 32) -> Tensor:
        """Embeddings of image files, read `batch_size` at a time."""
        return torch.cat([self(load_images(paths[i : i + batch_size])) for i in range(0, len(paths), batch_size)])


def mmd(x: Tensor, y: Tensor, sigma: float = SIGMA, scale: float = SCALE) -> float:
    """Biased MMD² estimate between embedding sets `x` and `y` (`[N, D]`) with a Gaussian kernel."""
    x, y = x.double(), y.double()
    gamma = 1.0 / (2.0 * sigma**2)

    def mean_kernel(a: Tensor, b: Tensor) -> Tensor:
        squared_distances = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
        return torch.exp(-gamma * squared_distances).mean()

    return scale * (mean_kernel(x, x) + mean_kernel(y, y) - 2.0 * mean_kernel(x, y)).item()
