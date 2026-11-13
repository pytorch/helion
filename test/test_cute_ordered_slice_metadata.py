"""Prove sequence ownership and bound purity before replaying host slices."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.state import cute_env_state
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_ordered_slice_metadata import sliced_metadata_votes

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode", ["prefix", "bounds", "reverse", "local"])
def test_pure_scalar_sequence_slice_keeps_normal_ordered_hosts(mode, tmp_path):
    x = torch.ones(3, 32, 32)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(sliced_metadata_votes, (x, mode))
        assert cute_env_state(bound.env).dependent_phase_plan is not None
        source = bound.to_code(helion.Config())
        assert len(stage_sources(source)) == 2
        path = tmp_path / "sliced_metadata.py"
        path.write_text(source)
        module = load_generated_module(path)
        calls = []

        def launch(kernel, grid, *args, **kwargs):
            calls.append((kernel, args))

        output = module.sliced_metadata_votes(x, mode, _launcher=launch)
    assert len(calls) == 2 and output.shape == (3,)
    assert calls[0][0].__name__.endswith("__region_0")
    assert calls[1][0].__name__.endswith("__region_1")
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize(
    "mode", ["mutable", "opaque_bound", "opaque_container", "tensor"]
)
def test_slice_does_not_admit_opaque_state_or_tensor_views(mode):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(sliced_metadata_votes, (torch.ones(3, 32, 32), mode))
