"""Compression of FLUX: low-rank linear layers, removed blocks, and the layout of FLUX blocks."""

import math
import re
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

from svdtrunc.flux.model import Flux
from svdtrunc.lowrank import LowRankLinear, SplitLinear, rank_for_ratio, retruncate, truncated_svd
from svdtrunc.spec import CompressionSpec, LinearSpec


class RemovedDoubleBlock(nn.Module):
    """Stand-in for a removed double block: both streams pass through unchanged."""

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        return img, txt


class RemovedSingleBlock(nn.Module):
    """Stand-in for a removed single block: the joint stream passes through unchanged."""

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        return x


def block_paths(model: Flux) -> list[str]:
    """Module paths of all transformer blocks, double blocks first."""
    return [f"double_blocks.{i}" for i in range(len(model.double_blocks))] + [
        f"single_blocks.{i}" for i in range(len(model.single_blocks))
    ]


def block_layout(model: Flux, block_path: str) -> dict[str, tuple[tuple[int, str], ...]]:
    """Linear layers of a block (paths relative to it), with the output size and group of each part.

    Parts follow FLUX's fused layers: `qkv` splits into Q, K and V, and a single block's `linear1` into
    Q, K, V and the MLP input. Groups name the parts that can be compressed together (e.g. `img_attn`, `mlp2`).
    """
    hidden = model.hidden_size
    mlp_hidden = int(hidden * model.params.mlp_ratio)
    if block_path.startswith("double_blocks."):
        layout = {}
        for stream in ("img", "txt"):
            layout[f"{stream}_mod.lin"] = ((6 * hidden, f"{stream}_mod"),)
            layout[f"{stream}_attn.qkv"] = ((hidden, f"{stream}_attn"),) * 3
            layout[f"{stream}_attn.proj"] = ((hidden, f"{stream}_proj"),)
            layout[f"{stream}_mlp.0"] = ((mlp_hidden, f"{stream}_mlp"),)
            layout[f"{stream}_mlp.2"] = ((hidden, f"{stream}_mlp"),)
        return layout
    if block_path.startswith("single_blocks."):
        return {
            "modulation.lin": ((3 * hidden, "mod"),),
            "linear1": ((hidden, "attn"),) * 3 + ((mlp_hidden, "mlp"),),
            "linear2": ((hidden, "mlp2"),),
        }
    raise ValueError(f"Not a FLUX block path: {block_path!r}.")


def describe(model: nn.Module) -> CompressionSpec:
    """Compression spec of `model`, read from its modules."""
    linears: dict[str, LinearSpec] = {}
    removed: list[str] = []
    split_prefixes: list[str] = []
    for path, module in model.named_modules():
        if any(path.startswith(prefix) for prefix in split_prefixes):
            continue
        if isinstance(module, SplitLinear):
            split_prefixes.append(f"{path}.")
            linears[path] = LinearSpec(
                tuple(part.out_features for part in module.parts), tuple(_rank(part) for part in module.parts)
            )
        elif isinstance(module, LowRankLinear):
            linears[path] = LinearSpec((module.out_features,), (module.rank,))
        elif isinstance(module, (RemovedDoubleBlock, RemovedSingleBlock)):
            removed.append(path)
    return CompressionSpec(linears, tuple(removed))


def build(model: Flux, spec: CompressionSpec) -> None:
    """Gives a dense `model` the structure in `spec`, with uninitialised weights, to load a checkpoint into."""
    _check_spec(spec)
    for path in spec.removed_blocks:
        model.set_submodule(path, _removed_block(path), strict=True)
    for path, linear_spec in spec.linears.items():
        dense = model.get_submodule(path)
        if not isinstance(dense, nn.Linear):
            raise TypeError(f"{path} is a {type(dense).__name__}; expected the nn.Linear of a dense model.")
        model.set_submodule(path, _empty_linear(dense, linear_spec), strict=True)


def compress(model: Flux, spec: CompressionSpec) -> None:
    """Brings `model` to the structure in `spec`, removing blocks and truncating layers via SVD.

    `spec` describes the whole target model. Low-rank layers are re-truncated from their factors. Ranks
    can only go down: removed blocks and low-rank layers cannot become dense again.
    """
    _check_spec(spec)
    current = describe(model)
    targets = {path: linear_spec for path, linear_spec in spec.linears.items() if not linear_spec.is_dense}
    restored = set(current.removed_blocks) - set(spec.removed_blocks)
    if restored:
        raise ValueError(f"Removed blocks cannot be restored: {sorted(restored)}.")
    densified = {
        path for path in current.linears if path not in targets and not _removed_parent(path, spec.removed_blocks)
    }
    if densified:
        raise ValueError(f"Low-rank layers cannot become dense again: {sorted(densified)}.")
    for path in spec.removed_blocks:
        model.set_submodule(path, _removed_block(path), strict=True)
    for path, target in targets.items():
        model.set_submodule(path, _compress_linear(model.get_submodule(path), target), strict=True)


def ratio_block_spec(
    model: Flux, block_path: str, ratio: float, groups: Iterable[str] | None = None
) -> dict[str, LinearSpec]:
    """Specs storing the chosen groups of a block with `ratio` of their dense weights removed.

    Ratios are relative to the dense layers and apply to every part of the selected groups (default: all).
    Parts outside `groups` keep their current storage.
    """
    return _block_spec(model, block_path, groups, lambda in_features, size, _: rank_for_ratio(in_features, size, ratio))


def step_block_spec(
    model: Flux, block_path: str, step: float, groups: Iterable[str] | None = None
) -> dict[str, LinearSpec]:
    """Specs compressing the chosen groups of a block (default: all) by a further `step` of their weights."""
    if not 0.0 <= step < 1.0:
        raise ValueError(f"Step must be in [0, 1), got {step}.")
    return _block_spec(
        model, block_path, groups, lambda in_features, size, rank: _rank_after_step(in_features, size, rank, step)
    )


def linear_parameters(in_features: int, spec: LinearSpec, bias: bool = True) -> int:
    """Parameters of a linear layer stored as `spec`."""
    total = 0
    for size, rank in zip(spec.part_sizes, spec.ranks, strict=True):
        total += in_features * size if rank is None else rank * (in_features + size)
        total += size if bias else 0
    return total


def parameters_removed(model: Flux, specs: dict[str, LinearSpec]) -> int:
    """Parameters that re-storing the given linear layers of `model` as `specs` would remove."""
    removed = 0
    for path, linear_spec in specs.items():
        module = model.get_submodule(path)
        current = sum(parameter.numel() for parameter in module.parameters())
        removed += current - linear_parameters(module.in_features, linear_spec)
    return removed


def compressible_parameters(model: Flux, block_path: str) -> int:
    """Current weights of a block's linear layers, without biases: what compression acts on."""
    block = model.get_submodule(block_path)
    return sum(
        parameter.numel()
        for name in block_layout(model, block_path)
        for parameter_name, parameter in block.get_submodule(name).named_parameters()
        if not parameter_name.endswith("bias")
    )


# Private implementation


def _check_spec(spec: CompressionSpec) -> None:
    for block in spec.removed_blocks:
        if not re.fullmatch(r"(double|single)_blocks\.\d+", block):
            raise ValueError(f"Only FLUX blocks can be removed, got {block!r}.")
    for path in spec.linears:
        if _removed_parent(path, spec.removed_blocks):
            raise ValueError(f"{path} lies inside a removed block.")


def _removed_parent(path: str, removed_blocks: tuple[str, ...]) -> bool:
    return any(path.startswith(f"{block}.") for block in removed_blocks)


def _removed_block(path: str) -> nn.Module:
    return RemovedDoubleBlock() if path.startswith("double_blocks.") else RemovedSingleBlock()


def _rank(part: nn.Module) -> int | None:
    return part.rank if isinstance(part, LowRankLinear) else None


def _empty_linear(dense: nn.Linear, spec: LinearSpec) -> nn.Module:
    if sum(spec.part_sizes) != dense.out_features:
        raise ValueError(f"Parts {spec.part_sizes} do not add up to {dense.out_features} output features.")
    factory = {"bias": dense.bias is not None, "device": dense.weight.device, "dtype": dense.weight.dtype}
    parts = [
        nn.Linear(dense.in_features, size, **factory)
        if rank is None
        else LowRankLinear(dense.in_features, size, rank, **factory)
        for size, rank in zip(spec.part_sizes, spec.ranks, strict=True)
    ]
    return parts[0] if len(parts) == 1 else SplitLinear(parts)


def _compress_linear(module: nn.Module, target: LinearSpec) -> nn.Module:
    parts = _parts(module, target.part_sizes)
    compressed = [_compress_part(part, rank) for part, rank in zip(parts, target.ranks, strict=True)]
    return compressed[0] if len(compressed) == 1 else SplitLinear(compressed)


def _parts(module: nn.Module, sizes: tuple[int, ...]) -> list[nn.Linear | LowRankLinear]:
    if isinstance(module, SplitLinear):
        current = tuple(part.out_features for part in module.parts)
        if current != sizes:
            raise ValueError(f"Cannot re-split parts {current} into {sizes}.")
        return list(module.parts)
    if sum(sizes) != module.out_features:
        raise ValueError(f"Parts {sizes} do not add up to {module.out_features} output features.")
    if len(sizes) == 1:
        return [module]
    if isinstance(module, LowRankLinear):
        raise ValueError("A jointly factorised layer cannot be split into parts.")
    weights = module.weight.detach().split(list(sizes))
    biases = module.bias.detach().split(list(sizes)) if module.bias is not None else [None] * len(sizes)
    return [_dense_linear(weight, bias) for weight, bias in zip(weights, biases, strict=True)]


def _part_ranks(module: nn.Module, sizes: tuple[int, ...]) -> tuple[int | None, ...]:
    if isinstance(module, SplitLinear):
        current = tuple(part.out_features for part in module.parts)
        if current != sizes:
            raise ValueError(f"Layer has parts {current}, expected {sizes}.")
        return tuple(_rank(part) for part in module.parts)
    if isinstance(module, LowRankLinear) and len(sizes) > 1:
        raise ValueError("A jointly factorised layer cannot be split into parts.")
    return (_rank(module),) * len(sizes)


def _compress_part(part: nn.Linear | LowRankLinear, rank: int | None) -> nn.Linear | LowRankLinear:
    if rank is None:
        if isinstance(part, LowRankLinear):
            raise ValueError("A low-rank part cannot become dense again.")
        return part
    if isinstance(part, LowRankLinear):
        if rank > part.rank:
            raise ValueError(f"Cannot raise a part's rank from {part.rank} to {rank}.")
        if rank == part.rank:
            return part
        down, up = retruncate(part.down.detach(), part.up.detach(), rank)
        dtype = part.down.dtype
    else:
        down, up = truncated_svd(part.weight.detach(), rank)
        dtype = part.weight.dtype
    bias = None if part.bias is None else part.bias.detach()
    layer = LowRankLinear(down.shape[1], up.shape[0], rank, bias=bias is not None, device=down.device, dtype=dtype)
    with torch.no_grad():
        layer.down.copy_(down)
        layer.up.copy_(up)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


def _dense_linear(weight: Tensor, bias: Tensor | None) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


def _block_spec(
    model: Flux,
    block_path: str,
    groups: Iterable[str] | None,
    new_rank: Callable[[int, int, int | None], int],
) -> dict[str, LinearSpec]:
    block = model.get_submodule(block_path)
    if isinstance(block, (RemovedDoubleBlock, RemovedSingleBlock)):
        raise ValueError(f"{block_path} is removed.")
    layout = block_layout(model, block_path)
    known = {group for parts in layout.values() for _, group in parts}
    selected = known if groups is None else set(groups)
    if not selected <= known:
        raise ValueError(f"Unknown groups {sorted(selected - known)} for {block_path}; choose from {sorted(known)}.")
    specs = {}
    for name, parts in layout.items():
        module = block.get_submodule(name)
        sizes = tuple(size for size, _ in parts)
        ranks = tuple(
            new_rank(module.in_features, size, rank) if group in selected else rank
            for (size, group), rank in zip(parts, _part_ranks(module, sizes), strict=True)
        )
        specs[f"{block_path}.{name}"] = LinearSpec(sizes, ranks)
    return specs


def _rank_after_step(in_features: int, out_features: int, rank: int | None, step: float) -> int:
    if rank is None:
        return rank_for_ratio(in_features, out_features, step)
    # The small offset keeps float error (e.g. 1100 * 0.4 = 440.00000000000006) from adding a rank.
    return max(1, math.ceil(rank * (1.0 - step) - 1e-9))
