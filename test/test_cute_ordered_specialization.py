"""Keep guarded integer specialization when projecting a source prelude."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute.state import cute_env_state
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_dependent_full_votes import specialized_dependent_votes

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("increment", [2, 7])
def test_self_specialization_keeps_guarded_constants_and_shared_host_state(
    increment, tmp_path
):
    x = torch.ones(3, 32, 32)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(specialized_dependent_votes, (x, increment, 13, "identity"))
        assert bound.env.specialized_vars
        assert cute_env_state(bound.env).dependent_phase_plan is not None
        source = bound.to_code(helion.Config())
    sources = [source, *stage_sources(source)]
    assert len(sources) == 3
    values = [
        node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "increment"
        and isinstance(node.value, ast.Constant)
    ]
    assert values == [increment]
    # Each stage uses the guarded constant in its device expression, so the
    # ordinary host dead-code pass can erase its now-unused local assignment.
    stage_zero = ast.parse(sources[1])
    device = next(
        node
        for node in stage_zero.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_helion_specialized_dependent_votes__region_0"
    )
    assert any(
        isinstance(node, ast.Constant) and node.value == increment
        for node in ast.walk(device)
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "increment"
        for node in ast.walk(device)
    )
    path = tmp_path / "specialized_phases.py"
    path.write_text(source)
    calls = []

    def launch(kernel, grid, *args, **kwargs):
        calls.append((kernel, args))

    with cpu_render_target(fake_device_sm_count=148):
        module = load_generated_module(path)
        output = module.specialized_dependent_votes(
            x, increment, 13, "identity", _launcher=launch
        )
    assert len(calls) == 2 and output.shape == (3,)
    assert calls[0][0].__name__.endswith("__region_0")
    assert calls[1][0].__name__.endswith("__region_1")
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize("mode", ["arithmetic", "other", "opaque", "tensor"])
def test_nonidentity_argument_rebinding_still_fails_during_binding(mode):
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="cannot be flattened"),
    ):
        cpu_bind(specialized_dependent_votes, (torch.ones(3, 32, 32), 2, 13, mode))
