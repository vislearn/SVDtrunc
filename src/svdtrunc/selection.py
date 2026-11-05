"""Importance-aware block selection (Sec. 3.1-3.2 of the paper).

1. Probe: render prompts with one block compressed a further `PROBE_STEP` of its current weights.
2. Score: CMMD of the probe images against the dense model's images, per million parameters removed.
3. Schedule: compress the best-ranked blocks by steps that fall linearly with rank and together remove a
   set number of parameters.
"""

from svdtrunc.compression import compressible_parameters, describe, step_block_spec
from svdtrunc.flux.model import Flux
from svdtrunc.spec import CompressionSpec

PROBE_STEP = 0.6


def probe_spec(model: Flux, block_path: str, step: float = PROBE_STEP) -> CompressionSpec:
    """Spec of `model` with one block probed."""
    current = describe(model)
    return CompressionSpec(current.linears | step_block_spec(model, block_path, step), current.removed_blocks)


def score(cmmd: float, parameters_removed: int) -> float:
    """CMMD per million parameters removed. Lower is better: the block is cheap to compress."""
    return cmmd / (parameters_removed / 1e6)


def schedule_steps(block_parameters: list[int], start: float, target: float) -> list[float]:
    """Steps `start - i * x` for blocks in rank order, with `x` such that they remove `target` parameters.

    `block_parameters[i]` is the current size of the block ranked `i`. Steps below 0 become 0, which removes
    slightly more than `target`.
    """
    numerator = start * sum(block_parameters) - target
    denominator = sum(i * parameters for i, parameters in enumerate(block_parameters))
    x = numerator / denominator if denominator else 0.0
    if x < 0:
        raise ValueError(f"Removing {target:.4g} parameters needs steps above {start}; raise `start`.")
    return [max(0.0, start - i * x) for i in range(len(block_parameters))]


def schedule_spec(model: Flux, ranking: list[str], num_blocks: int, start: float, target: float) -> CompressionSpec:
    """Spec of `model` with its `num_blocks` best-ranked blocks compressed by their scheduled steps."""
    blocks = ranking[:num_blocks]
    steps = schedule_steps([compressible_parameters(model, block) for block in blocks], start, target)
    current = describe(model)
    linears = dict(current.linears)
    for block, step in zip(blocks, steps, strict=True):
        if step > 0:
            linears |= step_block_spec(model, block, step)
    return CompressionSpec(linears, current.removed_blocks)
