from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _args
from .test_cute_chained_load_only_preparation import _config
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._compiler.cute import chained_matmul as chain
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from torch.fx.node import Argument


def _load(shape: tuple[int, ...], selectors: list[object]) -> Node:
    graph = Graph()
    source = graph.call_function(_tracing_ops._host_tensor, ("input",))
    source.meta["val"] = torch.empty(shape)
    return graph.call_function(
        memory_ops.load, (source, cast("Argument", selectors), None, None)
    )


@pytest.mark.parametrize(
    "selectors,expected",
    [
        ([-1, -1], [2, 4]),
        ([-3, -5], [0, 0]),
        ([slice(None), -2], [slice(None), 3]),
        ([None, -1, slice(None)], [None, 2, slice(None)]),
    ],
)
def test_negative_literal_uses_original_host_dimension(
    selectors: list[object], expected: list[object]
) -> None:
    node = _load((3, 5), selectors)
    expression = chain._Expression.__new__(chain._Expression)
    assert expression.selectors(node) == expected
    assert node.args[1] == selectors


@pytest.mark.parametrize(
    "shape,selectors", [((3, 5), [-4, 0]), ((3, 5), [0, -6]), ((0, 5), [-1, 0])]
)
def test_negative_literal_rejects_outside_original_extent(
    shape: tuple[int, ...], selectors: list[object]
) -> None:
    expression = chain._Expression.__new__(chain._Expression)
    with pytest.raises(chain._UnsupportedChain, match="exceeds host extent"):
        expression.selectors(_load(shape, selectors))


def test_positive_dynamic_and_internal_selectors_remain_unchanged() -> None:
    expression = chain._Expression.__new__(chain._Expression)
    node = _load((3, 5), [1, 2])
    assert expression.selectors(node) is node.args[1]
    dynamic = node.graph.placeholder("dynamic")
    dynamic.meta["val"] = torch.tensor(-1)
    node.args = (node.args[0], [dynamic, 2], None, None)
    assert expression.selectors(node) is node.args[1]
    cast("Node", node.args[0]).target = torch.ops.aten.clone.default
    node.args = (node.args[0], [-1, 2], None, None)
    assert expression.selectors(node) is node.args[1]


def test_unresolved_host_extent_remains_rejected() -> None:
    expression = chain._Expression.__new__(chain._Expression)
    with (
        patch.object(
            chain,
            "_host_shape",
            side_effect=chain._UnsupportedChain("dynamic host extent"),
        ),
        pytest.raises(chain._UnsupportedChain, match="dynamic host extent"),
    ):
        expression.selectors(_load((3, 5), [-1, 2]))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gated_dot(
    left: torch.Tensor, right: torch.Tensor, gate: torch.Tensor, last: hl.constexpr
) -> torch.Tensor:
    output = torch.empty_like(left, dtype=torch.float32)
    for batch in hl.tile(left.size(0), block_size=1):
        product = hl.dot(left[batch.id, :, :], right[batch.id, :, :])
        output[batch.id, :, :] = product * gate[batch.id, last]
    return output


def _device_ast(source: str) -> str:
    tree = ast.parse(source)
    return ast.dump(
        next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    )


@pytest.mark.parametrize("last,positive", [(-1, 4), (-5, 0)])
def test_negative_and_positive_literal_generate_identical_device_code(
    last: int, positive: int
) -> None:
    tensors = (
        torch.empty((2, 64, 64), dtype=torch.bfloat16),
        torch.empty((2, 64, 64), dtype=torch.bfloat16),
        torch.empty((2, 5)),
    )
    config = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
    with _cpu_codegen():
        negative = _gated_dot._bind_isolated((*tensors, last)).to_code(config)
        normalized = _gated_dot._bind_isolated((*tensors, positive)).to_code(config)
    assert _device_ast(negative) == _device_ast(normalized)


def test_load_only_scalar_decay_keeps_real_last_gate() -> None:
    arguments = _args(steps=1, dtype=torch.float16)
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(arguments)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_load_only_recurrence, arguments)
        ):
            source = bound.to_code(_config())
    assert "chain_resident_carry_" in source
    assert (
        "cutlass.Int32(31) * cutlass.Int32(input_tensor_1.layout.stride[2])" in source
    )
    assert "0 <= -1 < 32" not in source
