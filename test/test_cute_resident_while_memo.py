"""``resident_while_plan``'s memo: one plan per owner of the graphs, refusals
raised fresh, nothing kept past the environment."""

from __future__ import annotations

import gc
from unittest import mock
import weakref

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute import resident_while
from helion._compiler.cute.resident_while import resident_while_plan
from helion._compiler.cute.state import cute_env_state
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
import helion.language as hl
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _while_recipe(x, limits, swap: hl.constexpr, steps: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))
        left = x[row, lane]
        right = left + 1
        iteration = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            if swap:
                left, right = right, left
            else:
                left, right = left + right, left - right
            for _packet in range(steps):
                left = torch.gather(left + 1, 0, ((lane * 3 + 1) % x.size(1)).long())
            iteration = iteration + 1
        out[row, lane] = left + right
    return out


def _bind(*, swap: bool = False):
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        return cpu_bind(
            _while_recipe,
            (
                torch.arange(48).reshape(3, 16).float(),
                torch.tensor([0, 1, 3], dtype=torch.int32),
                swap,
                2,
            ),
        )


def _while_call(graphs):
    return next(
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is _tracing_ops._while_loop
    )


def _config(bound, **overrides: object):
    config = bound.config_spec.default_config()
    config.config["cute_fragment_bounded_gather"] = True
    config.config.update(overrides)
    return config


@skipIfRefEager("plans the traced graphs")
def test_plan_is_memoized_per_environment_for_the_traced_graphs() -> None:
    bound = _bind()
    graphs = bound.host_function.device_ir.graphs
    call = _while_call(graphs)
    with bound.env:
        plan = resident_while_plan(call, graphs)
        assert resident_while_plan(call, graphs) is plan
        # Another config's graph copies are planned on their own: the copies
        # carry fresh ids and the plan names them, never the traced graphs.
        copies = bound.host_function.device_ir.build_codegen_graphs(_config(bound))
        other = resident_while_plan(_while_call(copies), copies)
        assert other is not plan
        assert any(other.body.graph is info for info in copies)
        assert not any(plan.body.graph is info for info in copies)
    # Outside a compile environment nothing is cached (the structural tests
    # mutate a graph in place and re-plan the same call).
    assert resident_while_plan(call, graphs) is not plan


@skipIfRefEager("generates code")
def test_two_configs_in_one_environment_plan_their_own_graph_copies() -> None:
    """Each code generation pass plans its own graph copies (nothing is served
    from the previous pass) and the pass's memo dies with it: the shared
    environment renders exactly what a fresh one does."""
    bound = _bind()
    first_config = _config(bound)
    second_config = _config(bound, cute_pdl=True)
    with (
        mock.patch.object(
            resident_while,
            "_resident_while_plan",
            wraps=resident_while._resident_while_plan,
        ) as planner,
        mock_cuda_unavailable(),
        _target(),
        forbid_native_compile(),
    ):
        first = bound.to_code(first_config)
        # Keep both passes' owners alive: reset_mock() otherwise frees the
        # first graphs, allowing Python to reuse their ids during the next pass.
        first_graphs = {
            id(info): info for call in planner.call_args_list for info in call.args[1]
        }
        planner.reset_mock()
        second = bound.to_code(second_config)
        second_graphs = {
            id(info): info for call in planner.call_args_list for info in call.args[1]
        }
    assert first_graphs and second_graphs
    assert first_graphs.keys().isdisjoint(second_graphs)
    memo = cute_env_state(bound.env).resident_while_plans
    assert memo
    assert not any(
        id(info) in first_graphs | second_graphs
        for _call, graphs, _plan in memo.values()
        for info in graphs
    )
    assert first != second
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        assert _bind().to_code(second_config) == second


@skipIfRefEager("plans the traced graphs")
def test_refusal_is_raised_fresh_and_the_environment_is_collected() -> None:
    bound = _bind(swap=True)
    graphs = bound.host_function.device_ir.graphs
    call = _while_call(graphs)
    with bound.env:
        with pytest.raises(exc.InvalidConfig, match="explicit initialized phi") as a:
            resident_while_plan(call, graphs)
        with pytest.raises(exc.InvalidConfig, match="explicit initialized phi") as b:
            resident_while_plan(call, graphs)
    assert a.value is not b.value
    assert str(a.value) == str(b.value)
    environment = weakref.ref(bound.env)
    del bound, graphs, call, a, b
    gc.collect()
    assert environment() is None
