"""Tests for svdtrunc.compression and checkpoint round trips in svdtrunc.loading."""

import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tiny import TINY, block_inputs, tiny_flux, tiny_inputs

from svdtrunc.compression import (
    compress,
    describe,
    parameters_removed,
    ratio_block_spec,
    step_block_spec,
)
from svdtrunc.loading import load_flux, save_flux
from svdtrunc.lowrank import LowRankLinear, SplitLinear, rank_for_ratio, truncated_svd
from svdtrunc.spec import CompressionSpec, LinearSpec

HIDDEN = TINY.hidden_size
MLP = int(TINY.hidden_size * TINY.mlp_ratio)


def full_rank_spec(model) -> CompressionSpec:
    """Every part of every block factorised at full rank, which must not change the function."""
    linears = {}
    for path in ("double_blocks.0", "single_blocks.1"):
        for linear_path, linear_spec in ratio_block_spec(model, path, 0.0).items():
            in_features = model.get_submodule(linear_path).in_features
            ranks = tuple(min(in_features, size) for size in linear_spec.part_sizes)
            linears[linear_path] = LinearSpec(linear_spec.part_sizes, ranks)
    return CompressionSpec(linears)


class TestCompress:
    def test_full_rank_factorisation_keeps_outputs(self):
        model = tiny_flux()
        inputs = tiny_inputs()
        expected = model(**inputs)
        compress(model, full_rank_spec(model))
        assert isinstance(model.double_blocks[0].img_attn.qkv, SplitLinear)
        torch.testing.assert_close(model(**inputs), expected, atol=1e-4, rtol=1e-4)

    def test_part_is_best_approximation_of_its_weight_slice(self):
        model = tiny_flux()
        k_weight = model.single_blocks[0].linear1.weight[HIDDEN : 2 * HIDDEN].detach().clone()
        compress(
            model,
            CompressionSpec({"single_blocks.0.linear1": LinearSpec((HIDDEN,) * 3 + (MLP,), (None, 7, None, None))}),
        )
        k_part = model.single_blocks[0].linear1.parts[1]
        down, up = truncated_svd(k_weight, 7)
        torch.testing.assert_close(k_part.up @ k_part.down, up @ down, atol=1e-5, rtol=1e-5)

    def test_second_compression_lowers_existing_ranks(self):
        model = tiny_flux()
        compress(model, CompressionSpec(ratio_block_spec(model, "single_blocks.0", 0.2)))
        compress(model, CompressionSpec(step_block_spec(model, "single_blocks.0", 0.5)))
        ranks = describe(model).linears["single_blocks.0.linear2"].ranks
        assert ranks == (math.ceil(rank_for_ratio(HIDDEN + MLP, HIDDEN, 0.2) * 0.5),)

    def test_rejects_raising_a_rank(self):
        model = tiny_flux()
        compress(model, CompressionSpec({"double_blocks.0.img_mlp.0": LinearSpec((MLP,), (10,))}))
        with pytest.raises(ValueError, match="raise a part's rank"):
            compress(model, CompressionSpec({"double_blocks.0.img_mlp.0": LinearSpec((MLP,), (12,))}))

    def test_rejects_making_a_low_rank_layer_dense(self):
        model = tiny_flux()
        compress(model, CompressionSpec({"double_blocks.0.img_mlp.0": LinearSpec((MLP,), (10,))}))
        with pytest.raises(ValueError, match="cannot become dense"):
            compress(model, CompressionSpec())

    def test_removed_blocks_pass_their_inputs_through(self):
        model = tiny_flux()
        compress(model, CompressionSpec(removed_blocks=("double_blocks.1", "single_blocks.0")))
        _, outputs = model(**tiny_inputs(), return_block_outputs=True)
        torch.testing.assert_close(outputs[1], outputs[0])
        torch.testing.assert_close(outputs[2], outputs[1])

    def test_compressed_block_can_be_removed_later(self):
        model = tiny_flux()
        compress(model, CompressionSpec(ratio_block_spec(model, "double_blocks.1", 0.3)))
        compress(model, CompressionSpec(removed_blocks=("double_blocks.1",)))
        assert describe(model) == CompressionSpec(removed_blocks=("double_blocks.1",))

    @pytest.mark.parametrize("path", ["double_blocks.7", "double_blocks.0.img_attn", "final_layer"])
    def test_rejects_removing_what_is_not_a_block(self, path: str):
        with pytest.raises((AttributeError, ValueError)):
            compress(tiny_flux(), CompressionSpec(removed_blocks=(path,)))


class TestBlockSpecs:
    def test_ratio_spec_only_touches_selected_groups(self):
        model = tiny_flux()
        specs = ratio_block_spec(model, "single_blocks.0", 0.3, groups=["attn", "mlp2"])
        attn_rank = rank_for_ratio(HIDDEN, HIDDEN, 0.3)
        assert specs["single_blocks.0.linear1"].ranks == (attn_rank, attn_rank, attn_rank, None)
        assert specs["single_blocks.0.linear2"].ranks == (rank_for_ratio(HIDDEN + MLP, HIDDEN, 0.3),)
        assert specs["single_blocks.0.modulation.lin"].is_dense

    def test_rejects_unknown_group(self):
        with pytest.raises(ValueError, match="Unknown groups"):
            ratio_block_spec(tiny_flux(), "double_blocks.0", 0.3, groups=["mlp2"])

    def test_parameters_removed_matches_actual_count(self):
        model = tiny_flux()
        specs = ratio_block_spec(model, "double_blocks.1", 0.4)
        before = sum(parameter.numel() for parameter in model.parameters())
        predicted = parameters_removed(model, specs)
        compress(model, CompressionSpec(specs))
        assert before - sum(parameter.numel() for parameter in model.parameters()) == predicted


class TestCheckpoint:
    def test_round_trip_keeps_structure_and_outputs(self, tmp_path: Path):
        model = tiny_flux()
        linears = ratio_block_spec(model, "double_blocks.0", 0.5) | ratio_block_spec(
            model, "single_blocks.1", 0.3, groups=["attn"]
        )
        compress(model, CompressionSpec(linears, removed_blocks=("single_blocks.0",)))
        save_flux(model, tmp_path / "model.safetensors")
        loaded = load_flux(tmp_path / "model.safetensors", params=TINY, dtype=torch.float32)
        assert describe(loaded) == describe(model)
        assert isinstance(loaded.double_blocks[0].img_mod.lin, LowRankLinear)
        inputs = tiny_inputs()
        torch.testing.assert_close(loaded(**inputs), model(**inputs))

    def test_dense_checkpoint_loads_without_spec(self, tmp_path: Path):
        model = tiny_flux()
        save_file(
            {name: tensor.contiguous() for name, tensor in model.state_dict().items()}, tmp_path / "dense.safetensors"
        )
        loaded = load_flux(tmp_path / "dense.safetensors", params=TINY, dtype=torch.float32)
        assert describe(loaded) == CompressionSpec()
        block = block_inputs(model, tiny_inputs())
        torch.testing.assert_close(loaded.double_blocks[0](**block), model.double_blocks[0](**block))
