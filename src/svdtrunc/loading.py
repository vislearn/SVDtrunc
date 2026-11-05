"""Loading and saving FLUX.1 checkpoints (dev and schnell), the autoencoder, and the text encoders.

Checkpoints are local files or files on the Hugging Face Hub, written `<repo id>/<file>`, e.g.
`black-forest-labs/FLUX.1-dev/flux1-dev.safetensors`. FLUX.1-dev is gated: accept its license on the Hub and
log in (`hf auth login`) before the first download.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from svdtrunc.compression import build, describe
from svdtrunc.flux.autoencoder import AutoEncoder, AutoEncoderParams
from svdtrunc.flux.conditioner import HFEmbedder
from svdtrunc.flux.model import Flux, FluxParams
from svdtrunc.spec import CompressionSpec

# FLUX.1-dev settings, from the BFL reference `flux.util.configs["flux-dev"]`.
FLUX_DEV = FluxParams(
    in_channels=64,
    out_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)
# FLUX.1-schnell is FLUX.1-dev without the guidance embedding.
FLUX_SCHNELL = dataclasses.replace(FLUX_DEV, guidance_embed=False)
FLUX_AE = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)
SPEC_METADATA_KEY = "svdtrunc.compression_spec"


@dataclass(frozen=True)
class Base:
    """A FLUX.1 base model: its architecture, its Hub repository, and its sampling settings in the BFL reference."""

    name: str
    params: FluxParams
    repo: str
    checkpoint_file: str
    t5_max_length: int
    steps: int
    guidance: float
    shift: bool  # resolution-dependent timestep shift

    @property
    def checkpoint(self) -> str:
        return f"{self.repo}/{self.checkpoint_file}"

    @property
    def ae(self) -> str:
        return f"{self.repo}/ae.safetensors"


DEV = Base("FLUX.1-dev", FLUX_DEV, "black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", 512, 50, 3.5, True)
# Guidance is unused: FLUX.1-schnell has no guidance embedding.
SCHNELL = Base(
    "FLUX.1-schnell", FLUX_SCHNELL, "black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors", 256, 4, 0.0, False
)


def resolve(checkpoint: str | Path) -> Path:
    """Local path of a checkpoint: the file itself if it exists, else `<repo id>/<file>` downloaded from the Hub."""
    path = Path(checkpoint)
    if path.exists():
        return path
    parts = str(checkpoint).split("/")
    if len(parts) < 3 or path.is_absolute() or str(checkpoint).startswith("."):
        raise FileNotFoundError(f"{checkpoint} is neither a file nor a Hub file of the form <owner>/<repo>/<file>.")
    return Path(hf_hub_download("/".join(parts[:2]), "/".join(parts[2:])))


def detect_base(checkpoint: str | Path) -> Base:
    """FLUX.1-dev or FLUX.1-schnell, told apart by the guidance embedding only FLUX.1-dev has."""
    with safe_open(str(resolve(checkpoint)), framework="pt") as file:
        keys = file.keys()
    return DEV if any(key.startswith("guidance_in.") for key in keys) else SCHNELL


def save_flux(model: Flux, path: Path, dtype: torch.dtype | None = None) -> None:
    """Saves the weights (cast to `dtype`, if given) with the compression spec in the file metadata."""
    state = {
        name: tensor.detach().to(dtype or tensor.dtype).contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path), metadata={SPEC_METADATA_KEY: describe(model).to_json()})


def read_spec(checkpoint: str | Path) -> CompressionSpec:
    """Compression spec stored in a checkpoint. Files without one (e.g. `flux1-dev.safetensors`) are dense."""
    with safe_open(str(resolve(checkpoint)), framework="pt") as file:
        metadata = file.metadata() or {}
    if SPEC_METADATA_KEY not in metadata:
        return CompressionSpec()
    return CompressionSpec.from_json(metadata[SPEC_METADATA_KEY])


def load_flux(
    checkpoint: str | Path,
    params: FluxParams | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Flux:
    """Loads a dense FLUX checkpoint or a compressed one written by `save_flux`.

    `params` defaults to those of the checkpoint's base model (`detect_base`).
    """
    path = resolve(checkpoint)
    model = load_structure(path, params)
    state = load_file(str(path), device=str(device))
    model.load_state_dict({name: tensor.to(dtype) for name, tensor in state.items()}, strict=True, assign=True)
    return model


def load_structure(checkpoint: str | Path, params: FluxParams | None = None) -> Flux:
    """The model a checkpoint describes, on the meta device: shapes only, no weights."""
    with torch.device("meta"):
        model = Flux(params or detect_base(checkpoint).params)
    build(model, read_spec(checkpoint))
    return model


def load_ae(
    checkpoint: str | Path = DEV.ae,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    sample_latents: bool = False,
) -> AutoEncoder:
    """Loads the FLUX.1 autoencoder (`ae.safetensors`). `sample_latents` samples the posterior when encoding."""
    with torch.device("meta"):
        ae = AutoEncoder(FLUX_AE, sample_z=sample_latents)
    state = load_file(str(resolve(checkpoint)), device=str(device))
    ae.load_state_dict({name: tensor.to(dtype) for name, tensor in state.items()}, strict=True, assign=True)
    return ae


def load_text_encoders(
    base: Base = DEV, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> tuple[HFEmbedder, HFEmbedder]:
    """Loads T5-XXL and CLIP-L as `base` uses them, from its repository on the Hub."""
    t5 = HFEmbedder(
        base.repo,
        base.t5_max_length,
        is_clip=False,
        subfolder="text_encoder_2",
        tokenizer_subfolder="tokenizer_2",
        dtype=dtype,
    ).to(device)
    clip = HFEmbedder(
        base.repo, 77, is_clip=True, subfolder="text_encoder", tokenizer_subfolder="tokenizer", dtype=dtype
    ).to(device)
    return t5, clip
