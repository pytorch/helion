from __future__ import annotations

import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import sympy
import torch

from helion._compiler.cute.flat_index_partition import is_flat_column_partition
from helion.language import tile_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment


def _environment(width: int) -> CompileEnvironment:
    return cast(
        "CompileEnvironment",
        SimpleNamespace(
            block_sizes=[SimpleNamespace(size=width)],
            specialize_expr=sympy.sympify,
            resolve_block_id=lambda value: 0 if value == width else None,
        ),
    )


def _index_graph(
    width: int, dtype: torch.dtype
) -> tuple[torch.fx.Graph, torch.fx.Node, torch.fx.Node, torch.fx.Node]:
    graph = torch.fx.Graph()
    rows = graph.placeholder("rows")
    rows.meta["val"] = torch.empty((3, width), dtype=dtype)
    columns = graph.call_function(tile_ops.tile_index, (width,))
    columns.meta["val"] = torch.empty(width, dtype=torch.int32)
    expanded = graph.call_function(view_ops.subscript, (columns, [None, slice(None)]))
    expanded.meta["val"] = torch.empty((1, width), dtype=torch.int32)
    base = graph.call_function(torch.ops.aten.mul.Tensor, (rows, width))
    base.meta["val"] = rows.meta["val"]
    index = graph.call_function(torch.ops.aten.add.Tensor, (base, expanded))
    index.meta["val"] = rows.meta["val"]
    graph.output(index)
    return graph, rows, columns, index


@pytest.mark.parametrize("width", [8, 32, 128, 256])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_wrap_preserves_disjoint_column_residues(
    width: int, dtype: torch.dtype
) -> None:
    graph, _rows, _columns, index = _index_graph(width, dtype)
    assert is_flat_column_partition(index, 0, _environment(width))
    # The row expression may vary with the column and overflow. Those facts
    # must not invalidate disjoint ownership of each column residue.
    limits = torch.iinfo(dtype)
    rows = torch.tensor([limits.min, limits.max, -3, 0, 3], dtype=dtype)[:, None]
    columns = torch.arange(width, dtype=dtype)[None, :]
    flattened = (rows + columns) * width + columns
    assert torch.equal(flattened.remainder(width), columns.expand_as(flattened))
    graph.lint()


@pytest.mark.parametrize("width", [0, 1, 3, 31, 65, 257])
def test_rejects_unproved_modulus(width: int) -> None:
    _graph, _rows, _columns, index = _index_graph(width, torch.int64)
    assert not is_flat_column_partition(index, 0, _environment(width))


@pytest.mark.parametrize(
    "change",
    [
        "float_rows",
        "shifted",
        "wrong_factor",
        "reversed_columns",
        "narrow_cast",
        "opaque",
    ],
)
def test_rejects_indices_that_do_not_preserve_column_residue(change: str) -> None:
    graph, rows, columns, index = _index_graph(512, torch.int64)
    if change == "float_rows":
        rows.meta["val"] = torch.empty((3, 512), dtype=torch.float32)
    elif change == "wrong_factor":
        base = index.args[0]
        assert isinstance(base, torch.fx.Node)
        base.args = (rows, 511)
    elif change == "reversed_columns":
        expanded = index.args[1]
        assert isinstance(expanded, torch.fx.Node)
        expanded.args = (columns, [None, slice(None, None, -1)])
    else:
        with graph.inserting_before(next(iter(graph.find_nodes(op="output")))):
            if change == "shifted":
                changed = graph.call_function(operator.add, (index, 1))
                changed.meta["val"] = index.meta["val"]
            elif change == "narrow_cast":
                changed = graph.call_function(
                    torch.ops.prims.convert_element_type.default, (index, torch.int8)
                )
                changed.meta["val"] = torch.empty((3, 512), dtype=torch.int8)
            else:
                changed = graph.call_function(torch.ops.aten.sin.default, (index,))
                changed.meta["val"] = index.meta["val"]
        index = changed
    assert not is_flat_column_partition(index, 0, _environment(512))


def test_repeated_integer_dag_has_linear_proof_cost() -> None:
    graph, _rows, _columns, index = _index_graph(32, torch.int64)
    output = next(iter(graph.find_nodes(op="output")))
    with graph.inserting_before(output):
        for _ in range(60):
            doubled = graph.call_function(operator.add, (index, index))
            doubled.meta["val"] = index.meta["val"]
            same = graph.call_function(operator.sub, (doubled, index))
            same.meta["val"] = index.meta["val"]
            index = same
    assert is_flat_column_partition(index, 0, _environment(32))
