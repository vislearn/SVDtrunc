"""Tests for svdtrunc.cli at FLUX.1-dev size, on checkpoints that hold only a compression spec."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from svdtrunc.cli import main
from svdtrunc.compression import ratio_block_spec
from svdtrunc.flux.model import Flux
from svdtrunc.loading import FLUX_DEV, FLUX_SCHNELL, SPEC_METADATA_KEY, load_structure
from svdtrunc.spec import CompressionSpec


def shapes_only(path: Path, schnell: bool = False) -> Path:
    """A full-size checkpoint without weights. `describe`, `rank` and `schedule` read only its spec and keys.

    The base model is detected from the keys: only FLUX.1-dev has `guidance_in`.
    """
    with torch.device("meta"):
        model = Flux(FLUX_DEV)
    spec = CompressionSpec(ratio_block_spec(model, "double_blocks.0", 0.5), removed_blocks=("single_blocks.37",))
    key = "img_in.weight" if schnell else "guidance_in.in_layer.weight"
    save_file({key: torch.zeros(1)}, str(path), metadata={SPEC_METADATA_KEY: spec.to_json()})
    return path


@pytest.fixture
def shapes_only_checkpoint(tmp_path: Path) -> Path:
    return shapes_only(tmp_path / "model.safetensors")


class TestCli:
    def test_describe_reports_each_block_and_the_total(self, shapes_only_checkpoint: Path, capsys):
        main(["describe", str(shapes_only_checkpoint)])
        rows = {line.split()[0]: line.split()[-1] for line in capsys.readouterr().out.splitlines()[1:]}
        assert len(rows) == 19 + 38 + 1
        assert rows["single_blocks.37"] == "removed"
        assert rows["double_blocks.1"] == "100.0%"
        assert float(rows["double_blocks.0"].rstrip("%")) == pytest.approx(50.0, abs=0.5)

    def test_describe_compares_schnell_checkpoints_with_schnell(self, tmp_path: Path, capsys):
        path = shapes_only(tmp_path / "schnell.safetensors", schnell=True)
        main(["describe", str(path)])
        share = float(capsys.readouterr().out.splitlines()[-1].split()[-1].rstrip("%"))
        model = load_structure(path)
        with torch.device("meta"):
            dense = Flux(FLUX_SCHNELL)
        assert model.params == FLUX_SCHNELL
        kept = sum(parameter.numel() for parameter in model.parameters())
        total = sum(parameter.numel() for parameter in dense.parameters())
        assert share == pytest.approx(100 * kept / total, abs=0.05)

    def test_schedule_compresses_top_ranked_blocks(self, shapes_only_checkpoint: Path, tmp_path: Path, capsys):
        ranking = tmp_path / "ranking.json"
        ranking.write_text(json.dumps([{"block": f"single_blocks.{index}"} for index in range(10)]))
        output = tmp_path / "next.json"
        arguments = ["--checkpoint", str(shapes_only_checkpoint), "--ranking", str(ranking), "--output", str(output)]
        main(["schedule", *arguments, "--blocks", "4", "--start", "0.3", "--target", "1.5e8"])

        spec = CompressionSpec.from_json(output.read_text())
        compressed = {".".join(path.split(".")[:2]) for path in spec.linears}
        assert compressed == {"double_blocks.0"} | {f"single_blocks.{index}" for index in range(4)}
        assert spec.removed_blocks == ("single_blocks.37",)
        assert "target 0.150B" in capsys.readouterr().out
