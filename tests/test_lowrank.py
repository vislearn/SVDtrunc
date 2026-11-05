"""Tests for svdtrunc.lowrank."""

import math

import pytest
import torch

from svdtrunc.lowrank import LowRankLinear, SplitLinear, rank_for_ratio, retruncate, truncated_svd


class TestRankForRatio:
    @pytest.mark.parametrize(
        ("in_features", "out_features", "ratio"),
        [(3072, 18432, 0.3798), (3072, 3072, 0.25), (15360, 3072, 0.0089), (3072, 12288, 0.6)],
    )
    def test_rank_is_rounded_up_and_capped(self, in_features: int, out_features: int, ratio: float):
        expected = max(1, math.ceil(in_features * out_features * (1.0 - ratio) / (in_features + out_features)))
        assert rank_for_ratio(in_features, out_features, ratio) == min(expected, in_features, out_features)

    def test_rank_removes_requested_fraction(self):
        rank = rank_for_ratio(3072, 12288, 0.3)
        kept = rank * (3072 + 12288) / (3072 * 12288)
        assert kept == pytest.approx(0.7, abs=1e-3)

    def test_rejects_full_removal(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\)"):
            rank_for_ratio(64, 64, 1.0)


class TestTruncatedSvd:
    def test_full_rank_reconstructs_weight(self):
        weight = torch.randn(7, 5, generator=torch.Generator().manual_seed(0))
        down, up = truncated_svd(weight, rank=5)
        torch.testing.assert_close(up @ down, weight, atol=1e-5, rtol=1e-5)

    def test_error_equals_discarded_singular_values(self):
        weight = torch.randn(9, 6, generator=torch.Generator().manual_seed(0))
        down, up = truncated_svd(weight, rank=2)
        singular_values = torch.linalg.svdvals(weight)
        assert torch.linalg.norm(weight - up @ down) == pytest.approx(singular_values[2:].norm().item(), rel=1e-4)


class TestRetruncate:
    def test_equals_truncating_the_dense_product(self):
        generator = torch.Generator().manual_seed(0)
        down, up = torch.randn(5, 9, generator=generator), torch.randn(7, 5, generator=generator)
        new_down, new_up = retruncate(down, up, rank=3)
        expected_down, expected_up = truncated_svd(up @ down, rank=3)
        torch.testing.assert_close(new_up @ new_down, expected_up @ expected_down, atol=1e-4, rtol=1e-4)


class TestLayers:
    def test_low_rank_linear_applies_up_down_product(self):
        torch.manual_seed(0)
        layer = LowRankLinear(6, 4, rank=2)
        for parameter in layer.parameters():
            torch.nn.init.normal_(parameter)
        x = torch.randn(3, 6)
        torch.testing.assert_close(layer(x), x @ (layer.up @ layer.down).T + layer.bias)

    def test_split_linear_concatenates_parts(self):
        torch.manual_seed(0)
        dense, low_rank = torch.nn.Linear(6, 3), LowRankLinear(6, 5, rank=2)
        torch.nn.init.normal_(low_rank.down), torch.nn.init.normal_(low_rank.up), torch.nn.init.zeros_(low_rank.bias)
        x = torch.randn(3, 6)
        torch.testing.assert_close(SplitLinear([dense, low_rank])(x), torch.cat([dense(x), low_rank(x)], dim=-1))
