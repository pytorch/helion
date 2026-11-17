from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def selected_container(x, values: hl.constexpr, index: int):
    index = hl.specialize(index)
    selected = values[index]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if selected is values[0]:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_fresh_specialized_index_preserves_selected_container():
    left, right = tuple([4, 8]), tuple([4, 8])
    assert left == right and left is not right
    with cpu_render_target():
        kernel = helion.kernel(selected_container.fn, autotune_effort="none")
        bound = kernel.bind((torch.empty(17), (left, right), 1))
        offsets = [
            n.args[1]
            for graph in bound.host_function.device_ir.graphs
            for n in graph.graph.nodes
            if n.target is torch.ops.aten.add.Tensor
        ]
        assert offsets == [5.0]


@helion.kernel(autotune_effort="none")
def selected_scalar(x, values: hl.constexpr, index: int):
    index = hl.specialize(index)
    offset = values[index]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + offset
    return out


@pytest.mark.parametrize("index,expected", [(0, 2.0), (1, 5.0), (-1, 5.0)])
@onlyBackends(["triton", "cute"])
def test_specialized_index_preserves_selected_scalar(index, expected):
    with cpu_render_target():
        kernel = helion.kernel(selected_scalar.fn, autotune_effort="none")
        bound = kernel.bind((torch.empty(17), (2.0, 5.0), index))
        offsets = [
            n.args[1]
            for graph in bound.host_function.device_ir.graphs
            for n in graph.graph.nodes
            if n.target is torch.ops.aten.add.Tensor
        ]
        assert offsets == [expected]


def _offsets(bound):
    return [
        n.args[1]
        for graph in bound.host_function.device_ir.graphs
        for n in graph.graph.nodes
        if n.target is torch.ops.aten.add.Tensor
    ]


@onlyBackends(["triton", "cute"])
def test_specialized_index_keeps_distinct_cached_selections():
    tensor = torch.empty(17)
    with cpu_render_target():
        kernel = helion.kernel(selected_scalar.fn, autotune_effort="none")
        first = kernel.bind((tensor, (2.0, 5.0), 0))
        second = kernel.bind((tensor, (2.0, 5.0), 1))
        assert first is not second
        assert _offsets(first) == [2.0]
        assert _offsets(second) == [5.0]
        assert kernel.bind((tensor, (2.0, 5.0), 0)) is first
        assert kernel.bind((tensor, (2.0, 5.0), 1)) is second


@onlyBackends(["triton", "cute"])
def test_specialized_index_accepts_heterogeneous_element_types():
    with cpu_render_target():
        kernel = helion.kernel(selected_scalar.fn, autotune_effort="none")
        bound = kernel.bind((torch.empty(17), (2, 5.5), 1))
        bound.to_code()
    assert _offsets(bound) == [5.5]


@helion.kernel(autotune_effort="none")
def selected_derived_index(x, values: hl.constexpr, index: int):
    chosen = values[hl.specialize(index) + 1]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + chosen
    return out


@onlyBackends(["triton", "cute"])
def test_expression_of_specialized_index_selects_actual_element():
    with cpu_render_target():
        bound = selected_derived_index.bind((torch.empty(17), (2.0, 5.0), 0))
        bound.to_code()
    assert _offsets(bound) == [5.0]


@onlyBackends(["triton", "cute"])
def test_specialized_index_rejects_out_of_bounds():
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.TypeInferenceError, match="IndexError"),
    ):
        kernel = helion.kernel(selected_scalar.fn, autotune_effort="none")
        kernel.bind((torch.empty(17), (2.0, 5.0), 2))
