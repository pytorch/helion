from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _dot
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_coordinates import CoordinateContext
from helion._compiler.cute.chained_coordinates import CoordinateDomain
from helion._compiler.cute.chained_coordinates import CoordinateText
from helion.language import _tracing_ops
from helion.language import scan_ops

if TYPE_CHECKING:
    from torch.fx import Node

    from helion._compiler.generate_ast import GenerateAST


@pytest.fixture(autouse=True)
def static_tensor_shapes():
    # These FX fixtures have only literal dimensions and no active tile context.
    with (
        patch.object(chain, "_shape", side_effect=lambda n: tuple(n.meta["val"].shape)),
        patch.object(
            chain.CompileEnvironment,
            "current",
            return_value=SimpleNamespace(get_block_id=lambda size: None),
        ),
    ):
        yield


def domain(node: Node, coordinates: tuple[str, ...]) -> chain._DomainExpression:
    plan = chain.ChainedMatmulPlan(
        root_graph_id=0,
        dots=(),
        store=node,
        axes=(),
        shapes=(),
        dtype=torch.float32,
        threads=128,
    )
    return chain._DomainExpression(
        cast("GenerateAST", object()), plan, node, coordinates
    )


def padded_iota(graph: Graph) -> Node:
    return _call(graph, torch.ops.prims.iota.default, (8,), (16,), torch.int32)


@pytest.mark.parametrize("depth", [1, 8, 16])
def test_shared_contraction_dag_keeps_padding_with_linear_visits(depth: int) -> None:
    graph = Graph()
    iota = padded_iota(graph)
    column = _call(
        graph, torch.ops.aten.unsqueeze.default, (iota, 1), (16, 1), torch.float32
    )
    node = _call(
        graph,
        torch.ops.aten.expand.default,
        (column, (16, 16)),
        (16, 16),
        torch.float32,
    )
    for _ in range(depth):
        node = _dot(graph, node, node)
    expression = domain(node, ("row", "column"))
    with patch.object(chain, "_shape_domain", wraps=chain._shape_domain) as visit:
        expression.value(node, ("row", "column"))
        # Each shared contraction has at most four distinct coordinate images.
        assert visit.call_count <= 4 * depth + 12
    assert chain._domain_bounds(expression) == ["0 <= (row) < 8"]
    assert len(expression.unavailable_values) == depth


@pytest.mark.parametrize("operation", ["scan", "sum", "mask"])
def test_collective_paths_keep_domains(operation: str) -> None:
    graph = Graph()
    iota = padded_iota(graph)
    column = _call(
        graph, torch.ops.aten.unsqueeze.default, (iota, 1), (16, 1), torch.float32
    )
    matrix = _call(
        graph,
        torch.ops.aten.expand.default,
        (column, (16, 16)),
        (16, 16),
        torch.float32,
    )
    if operation == "scan":
        node = _call(
            graph,
            scan_ops._associative_scan,
            (None, matrix, 1, False, False),
            (16, 16),
            torch.float32,
        )
        coordinates = ("row", "column")
    elif operation == "sum":
        node = _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (matrix, [1], False),
            (16,),
            torch.float32,
        )
        coordinates = ("row",)
    else:
        node = _call(graph, _tracing_ops._mask_to, (matrix, 0), (16, 16), torch.float32)
        coordinates = ("row", "column")
    expression = domain(node, coordinates)
    first = expression.value(node, coordinates)
    assert expression.value(node, coordinates) == first
    assert chain._domain_bounds(expression) == ["0 <= (row) < 8"]


def test_equal_rendering_retains_original_padding_identity() -> None:
    graph = Graph()
    node = _call(graph, torch.ops.prims.iota.default, (16,), (16,), torch.int32)
    expression = domain(node, ("original",))
    expression.coordinate_normalization = CoordinateContext(
        CoordinateDomain("row", "column", (0, 31), (0, 31))
    )
    expression.value(node, (CoordinateText("row", "original"),))
    assert not chain._domain_bounds(expression)
    expression.value(node, (CoordinateText("row", "derived"),))
    assert chain._domain_bounds(expression) == ["0 <= (row) < 16"]


def test_distinct_coordinates_and_instances_keep_their_own_domains() -> None:
    graph = Graph()
    node = padded_iota(graph)
    expression = domain(node, ("row",))
    expression.value(node, ("row",))
    expression.value(node, ("column",))
    assert chain._domain_bounds(expression) == ["0 <= (row) < 8", "0 <= (column) < 8"]
    other = domain(node, ("column",))
    other.value(node, ("column",))
    assert chain._domain_bounds(other) == ["0 <= (column) < 8"]


def test_failed_visits_are_never_cached() -> None:
    graph = Graph()
    node = _call(graph, torch.ops.prims.iota.default, ("dynamic",), (16,), torch.int32)
    expression = domain(node, ("row",))
    for _ in range(2):
        with pytest.raises(chain._UnsupportedChain, match="symbolic iota domain"):
            expression.value(node, ("row",))
    assert not expression.domain_memo
