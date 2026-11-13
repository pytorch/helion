from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from helion._compiler.device_ir import _get_custom_decomp_table
from helion._compiler.inductor_lowering import ReductionLowering
from helion._compiler.merge_reduction_dims import _META_KEY

pytestmark = pytest.mark.cpu_only


def _operation(operation, dims, keepdim):
    def reduce(x):
        return (
            operation(x)
            if dims is None and not keepdim
            else operation(x, dim=dims, keepdim=keepdim)
        )

    return reduce


@pytest.mark.parametrize("operation", (torch.any, torch.all))
def test_logical_chains_preserve_eager_axes_dtypes_and_truth_values(operation):
    # Each form exercises a distinct API rule; both operators need their own
    # identity check, especially when a reduced dimension has zero elements.
    cases = [
        (torch.tensor([[0.0, float("nan")], [float("inf"), -0.0]]), None, False),
        (torch.arange(48, dtype=torch.uint8).reshape(2, 3, 2, 4) % 3, (3, 1), True),
        (torch.zeros(2, 3, 4, dtype=torch.bool), (-1, -2), False),
        (torch.ones(2, 1, 4), (1, 2), True),
        (torch.zeros(2, 3), None, True),
        (torch.empty(2, 0, 3, dtype=torch.uint8), (1, 2), False),
        (torch.tensor([[0, 2]], dtype=torch.uint8), (), True),
        (torch.tensor(0.0), None, False),
        (torch.tensor([[0j, 1j], [complex(float("nan"), 0), 0j]]), (0, 1), False),
    ]
    # Each exceptional value must determine the result, independently of the
    # other truthy values or zeros that would otherwise mask a bad conversion.
    for value in (float("nan"), float("inf"), -float("inf"), complex(0, float("nan"))):
        cases.extend(
            (
                (torch.tensor([[0, value], [0, 0]]), None, False),
                (torch.tensor([[1, value], [1, 1]]), None, False),
            )
        )
    for value, dims, keepdim in cases:
        reduce = _operation(operation, dims, keepdim)
        expected = reduce(value)
        graph = make_fx(reduce, decomposition_table=_get_custom_decomp_table())(value)
        actual = graph(value)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", (torch.any, torch.all))
def test_logical_duplicate_axes_still_raise(operation):
    def reduce(x):
        return operation(x, dim=(0, -2))

    x = torch.zeros(2, 3)
    with pytest.raises(RuntimeError, match="dim.*multiple"):
        reduce(x)
    with pytest.raises(RuntimeError, match="dim.*multiple"):
        make_fx(reduce, decomposition_table=_get_custom_decomp_table())(x)


def test_reduction_axis_uses_decomposed_overload_schema():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.amax.default, (x, [1]))
    node.meta["original_aten"] = torch.ops.aten.any.default
    assert ReductionLowering._get_reduction_dims(node, torch.empty(32, 128)) == [1]


@pytest.mark.parametrize("keepdim", (False, True))
def test_reduction_axis_ignores_source_overload_positional_offset(keepdim):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.sum.dim_IntList, (x, [2], keepdim))
    # The source has ord before dim; the decomposed sum has dim immediately
    # after self, so the retained source schema would read keepdim as an axis.
    node.meta["original_aten"] = torch.ops.aten.linalg_vector_norm.default
    assert ReductionLowering._get_reduction_dims(node, torch.empty(3, 7, 9)) == [2]


@pytest.mark.parametrize("operation", (torch.any, torch.all))
def test_logical_chains_retain_layout_merge_metadata(operation):
    reduce = _operation(operation, (1, 2), False)
    graph = make_fx(reduce, decomposition_table=_get_custom_decomp_table())(
        torch.zeros(2, 3, 4)
    )
    tags = [
        node.meta[_META_KEY] for node in graph.graph.nodes if _META_KEY in node.meta
    ]
    assert len(tags) == 1
    assert tags[0].dims == [1, 2]
    assert tags[0].op == (
        torch.ops.aten.amax.default.name()
        if operation is torch.any
        else torch.ops.aten.amin.default.name()
    )
