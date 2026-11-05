"""Tests for svdtrunc.selection."""

import pytest
from tiny import tiny_flux

from svdtrunc.compression import (
    compress,
    compressible_parameters,
    describe,
    parameters_removed,
    ratio_block_spec,
    step_block_spec,
)
from svdtrunc.selection import probe_spec, schedule_spec, schedule_steps
from svdtrunc.spec import CompressionSpec


class TestScheduleSteps:
    def test_steps_remove_target_and_fall_with_rank(self):
        sizes = [300, 100, 200, 100]
        steps = schedule_steps(sizes, start=0.3, target=150)
        assert sum(size * step for size, step in zip(sizes, steps, strict=True)) == pytest.approx(150)
        assert steps == sorted(steps, reverse=True)

    def test_rejects_target_beyond_start(self):
        with pytest.raises(ValueError, match="raise `start`"):
            schedule_steps([100, 100], start=0.1, target=50)


class TestProbe:
    def test_probe_removes_step_of_one_block_only(self):
        model = tiny_flux()
        linears = step_block_spec(model, "single_blocks.1", step=0.6)
        assert all(path.startswith("single_blocks.1.") for path in linears)
        removed = parameters_removed(model, linears)
        assert removed == pytest.approx(0.6 * compressible_parameters(model, "single_blocks.1"), rel=0.03)

    def test_probe_keeps_earlier_compression_of_other_blocks(self):
        model = tiny_flux()
        compress(model, CompressionSpec(ratio_block_spec(model, "double_blocks.0", 0.3)))
        spec = probe_spec(model, "single_blocks.0")
        assert {path: spec.linears[path] for path in describe(model).linears} == describe(model).linears


class TestScheduleSpec:
    def test_compresses_top_blocks_by_about_the_target(self):
        model = tiny_flux()
        ranking = ["single_blocks.1", "double_blocks.0", "single_blocks.0", "double_blocks.1"]
        target = 60_000
        spec = schedule_spec(model, ranking, num_blocks=3, start=0.4, target=target)
        assert {".".join(path.split(".")[:2]) for path in spec.linears} == set(ranking[:3])
        assert parameters_removed(model, spec.linears) == pytest.approx(target, rel=0.05)
