"""Low-rank linear layers and truncated-SVD factorisation."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LowRankLinear(nn.Module):
    """Linear layer stored as two factors: `y = x @ down.T @ up.T + bias`.

    Attributes:
        down: `[rank, in_features]`.
        up: `[out_features, rank]`.
        bias: `[out_features]`, or `None`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not 0 <= rank <= min(in_features, out_features):
            raise ValueError(
                f"Rank {rank} is outside [0, {min(in_features, out_features)}] for a "
                f"{in_features} -> {out_features} layer."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.down = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.up = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def rank(self) -> int:
        return self.down.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.down), self.up, self.bias)


class SplitLinear(nn.Module):
    """Linear layer whose output is the concatenation of independent parts, each dense or low-rank.

    FLUX fuses Q, K and V (and, in single blocks, the MLP input) into one linear layer. Splitting lets
    each of them keep its own rank.
    """

    def __init__(self, parts: list[nn.Linear | LowRankLinear]):
        super().__init__()
        in_features = {part.in_features for part in parts}
        if len(in_features) != 1:
            raise ValueError(f"All parts must have the same in_features, got {sorted(in_features)}.")
        self.parts = nn.ModuleList(parts)
        self.in_features = in_features.pop()
        self.out_features = sum(part.out_features for part in parts)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([part(x) for part in self.parts], dim=-1)


def rank_for_ratio(in_features: int, out_features: int, ratio: float) -> int:
    """Rank at which the two factors hold `1 - ratio` of the dense layer's weights, rounded up.

    `ratio` is the fraction of the dense weights removed. Even ratio 0 truncates: a square layer keeps half
    its rank, since two factors of that rank already hold as many weights as the dense layer.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"Compression ratio must be in [0, 1), got {ratio}.")
    return math.ceil(in_features * out_features * (1.0 - ratio) / (in_features + out_features))


def truncated_svd(weight: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Best rank-`rank` factors `(down, up)` of `weight` (`[out, in]`), in float32.

    The singular values are split evenly between the two factors.
    """
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    root = s[:rank].sqrt()
    return root[:, None] * vh[:rank], u[:, :rank] * root


def retruncate(down: Tensor, up: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Best rank-`rank` factors of `up @ down`, computed without forming the dense product."""
    q_up, r_up = torch.linalg.qr(up.float())
    q_down, r_down = torch.linalg.qr(down.float().T)
    u, s, vh = torch.linalg.svd(r_up @ r_down.T)
    root = s[:rank].sqrt()
    return root[:, None] * (vh[:rank] @ q_down.T), (q_up @ u[:, :rank]) * root
