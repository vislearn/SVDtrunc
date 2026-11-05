"""Tests for svdtrunc.cmmd."""

import math

import pytest
import torch

from svdtrunc.cmmd import mmd


def pairwise_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0, scale: float = 1000.0) -> float:
    def kernel(a: torch.Tensor, b: torch.Tensor) -> float:
        return math.exp(-float(((a - b) ** 2).sum()) / (2 * sigma**2))

    def mean_kernel(a: torch.Tensor, b: torch.Tensor) -> float:
        return sum(kernel(u, v) for u in a for v in b) / (len(a) * len(b))

    return scale * (mean_kernel(x, x) + mean_kernel(y, y) - 2 * mean_kernel(x, y))


class TestMmd:
    def test_matches_pairwise_definition(self):
        generator = torch.Generator().manual_seed(0)
        x = torch.nn.functional.normalize(torch.randn(6, 4, generator=generator, dtype=torch.float64), dim=-1)
        y = torch.nn.functional.normalize(torch.randn(5, 4, generator=generator, dtype=torch.float64) + 1.0, dim=-1)
        assert mmd(x, y) == pytest.approx(pairwise_mmd(x, y), rel=1e-9)

    def test_is_zero_for_identical_sets_and_grows_with_shift(self):
        x = torch.nn.functional.normalize(torch.randn(8, 4, generator=torch.Generator().manual_seed(0)), dim=-1)
        near = torch.nn.functional.normalize(x + 0.1, dim=-1)
        far = torch.nn.functional.normalize(x + 1.0, dim=-1)
        assert mmd(x, x) == pytest.approx(0.0, abs=1e-9)
        assert 0 < mmd(x, near) < mmd(x, far)
