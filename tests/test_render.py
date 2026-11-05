"""Tests for svdtrunc.render."""

from pathlib import Path

import torch
from tiny import TINY_PIPELINE, stub_text_encoders, tiny_ae, tiny_flux

from svdtrunc.render import encode_prompts, load_images, render, save_images


class TestRender:
    def test_renders_one_image_per_prompt_in_unit_range(self):
        txt, vec = encode_prompts(*stub_text_encoders(), ["a cat", "a dog"])
        images = render(tiny_flux(params=TINY_PIPELINE), tiny_ae(), txt, vec, [3, 4], 32, 48, steps=2)
        assert images.shape == (2, 3, 32, 48)
        assert 0.0 <= images.min() and images.max() <= 1.0

    def test_same_seed_and_prompt_give_same_image(self):
        model, ae = tiny_flux(params=TINY_PIPELINE), tiny_ae()
        txt, vec = encode_prompts(*stub_text_encoders(), ["a cat", "a cat"])
        first = render(model, ae, txt, vec, [3, 3], 32, 32, steps=2)
        torch.testing.assert_close(first[0], first[1])

    def test_batched_encoding_matches_one_batch(self):
        prompts = ["a", "bb", "ccc", "dddd", "eeeee"]
        one = encode_prompts(*stub_text_encoders(), prompts)
        batched = encode_prompts(*stub_text_encoders(), prompts, batch_size=2)
        for expected, actual in zip(one, batched, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_saved_images_round_trip(self, tmp_path: Path):
        images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        paths = [tmp_path / "0.png", tmp_path / "1.png"]
        save_images(images, paths)
        torch.testing.assert_close(load_images(paths), images, atol=0.5 / 255 + 1e-6, rtol=0)
