"""A tiny random FLUX, autoencoder, and text encoders for tests."""

import dataclasses

import torch
from torch import Tensor, nn

from svdtrunc.flux.autoencoder import AutoEncoder, AutoEncoderParams
from svdtrunc.flux.layers import timestep_embedding
from svdtrunc.flux.model import Flux, FluxParams

TINY = FluxParams(
    in_channels=16,
    out_channels=16,
    vec_in_dim=12,
    context_in_dim=20,
    hidden_size=64,
    mlp_ratio=4.0,
    num_heads=2,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[8, 12, 12],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)
# FLUX's 16 latent channels packed in 2 x 2 patches, as sampling and the autoencoder need.
TINY_PIPELINE = dataclasses.replace(TINY, in_channels=64, out_channels=64)
TINY_AE = AutoEncoderParams(
    resolution=32,
    in_channels=3,
    ch=32,
    out_ch=3,
    ch_mult=[1, 1, 1, 1],
    num_res_blocks=1,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)


def tiny_flux(seed: int = 0, params: FluxParams = TINY) -> Flux:
    """Random tiny FLUX in float32. QK-norm scales are random too, so a lost norm changes outputs."""
    torch.manual_seed(seed)
    model = Flux(params)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("_norm.scale"):
                parameter.uniform_(0.2, 2.0)
    return model


def tiny_ae(seed: int = 0) -> AutoEncoder:
    torch.manual_seed(seed)
    return AutoEncoder(TINY_AE).eval()


class StubTextEncoder(nn.Module):
    """Stands in for T5 or CLIP: an embedding that depends on the prompt, in bfloat16 like `HFEmbedder`."""

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, prompts: list[str]) -> Tensor:
        return torch.stack([torch.full(self.shape, len(prompt) / 10.0) for prompt in prompts]).bfloat16()


def stub_text_encoders(params: FluxParams = TINY_PIPELINE) -> tuple[StubTextEncoder, StubTextEncoder]:
    return StubTextEncoder((7, params.context_in_dim)), StubTextEncoder((params.vec_in_dim,))


def tiny_inputs(
    params: FluxParams = TINY, batch: int = 2, height: int = 3, width: int = 4, txt_tokens: int = 5, seed: int = 1
) -> dict[str, Tensor]:
    """Model inputs with an image of `height x width` packed tokens."""
    generator = torch.Generator().manual_seed(seed)
    img_ids = torch.zeros(height, width, 3)
    img_ids[..., 1] = torch.arange(height)[:, None]
    img_ids[..., 2] = torch.arange(width)[None, :]
    return {
        "img": torch.randn(batch, height * width, params.in_channels, generator=generator),
        "img_ids": img_ids.reshape(1, height * width, 3).expand(batch, -1, -1),
        "txt": torch.randn(batch, txt_tokens, params.context_in_dim, generator=generator),
        "txt_ids": torch.zeros(batch, txt_tokens, 3),
        "timesteps": torch.rand(batch, generator=generator),
        "y": torch.randn(batch, params.vec_in_dim, generator=generator),
        "guidance": torch.full((batch,), 3.5),
    }


def block_inputs(model: Flux, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Inputs of the first double block (`img`, `txt`, `vec`, `pe`) for `inputs`."""
    vec = model.time_in(timestep_embedding(inputs["timesteps"], 256))
    vec = vec + model.guidance_in(timestep_embedding(inputs["guidance"], 256)) + model.vector_in(inputs["y"])
    pe = model.pe_embedder(torch.cat((inputs["txt_ids"], inputs["img_ids"]), dim=1))
    return {"img": model.img_in(inputs["img"]), "txt": model.txt_in(inputs["txt"]), "vec": vec, "pe": pe}
