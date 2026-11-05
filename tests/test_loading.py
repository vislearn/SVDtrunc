"""Tests for svdtrunc.loading."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from svdtrunc.loading import DEV, SCHNELL, detect_base, resolve


class TestDetectBase:
    def test_guidance_embedding_means_dev(self, tmp_path: Path):
        path = tmp_path / "dev.safetensors"
        save_file({"guidance_in.in_layer.weight": torch.zeros(1), "img_in.weight": torch.zeros(1)}, str(path))
        assert detect_base(path) is DEV

    def test_no_guidance_embedding_means_schnell(self, tmp_path: Path):
        path = tmp_path / "schnell.safetensors"
        save_file({"img_in.weight": torch.zeros(1)}, str(path))
        assert detect_base(path) is SCHNELL

    def test_bases_follow_the_bfl_reference(self):
        assert (DEV.steps, DEV.guidance, DEV.shift, DEV.t5_max_length) == (50, 3.5, True, 512)
        assert (SCHNELL.steps, SCHNELL.shift, SCHNELL.t5_max_length) == (4, False, 256)
        assert DEV.params.guidance_embed and not SCHNELL.params.guidance_embed


class TestResolve:
    def test_existing_file_is_returned_as_is(self, tmp_path: Path):
        path = tmp_path / "model.safetensors"
        path.touch()
        assert resolve(path) == path

    @pytest.mark.parametrize("name", ["missing.safetensors", "./a/b/c.safetensors", "/abs/a/b.safetensors"])
    def test_missing_local_paths_are_not_sent_to_the_hub(self, name: str):
        with pytest.raises(FileNotFoundError, match="neither a file nor a Hub file"):
            resolve(name)
