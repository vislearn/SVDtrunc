"""Description of a compressed model: how each linear layer is stored and which blocks are removed."""

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinearSpec:
    """Storage of one linear layer, split along its output features into parts.

    Attributes:
        part_sizes: Output features of each part, in order.
        ranks: Rank of each part, or `None` for a dense part.
    """

    part_sizes: tuple[int, ...]
    ranks: tuple[int | None, ...]

    def __post_init__(self):
        if not self.part_sizes or len(self.part_sizes) != len(self.ranks):
            raise ValueError(f"Need one rank per part, got sizes {self.part_sizes} and ranks {self.ranks}.")

    @property
    def is_dense(self) -> bool:
        return all(rank is None for rank in self.ranks)


@dataclass(frozen=True)
class CompressionSpec:
    """Compressed-model description. Linear layers not listed in `linears` are dense.

    Attributes:
        linears: Module path of a linear layer -> how it is stored.
        removed_blocks: Module paths of blocks replaced by an identity.
    """

    linears: dict[str, LinearSpec] = field(default_factory=dict)
    removed_blocks: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "linears": {
                    path: {"part_sizes": list(spec.part_sizes), "ranks": list(spec.ranks)}
                    for path, spec in sorted(self.linears.items())
                },
                "removed_blocks": sorted(self.removed_blocks),
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> "CompressionSpec":
        data = json.loads(text)
        linears = {
            path: LinearSpec(tuple(entry["part_sizes"]), tuple(entry["ranks"]))
            for path, entry in data["linears"].items()
        }
        return cls(linears=linears, removed_blocks=tuple(data["removed_blocks"]))
