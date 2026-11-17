"""Saved input strides must survive replacement by contiguous masked values."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion._testing import cpu_render_target
from helion._testing.autodiff_shapes import alias_shape_backward
from helion._testing.autodiff_shapes import alias_shape_case
from helion._testing.cute_codegen import cpu_bind

if TYPE_CHECKING:
    import helion

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("case", ["sliced_unsqueeze_mask", "sliced_unsqueeze_count"])
def test_masked_fragment_reads_each_logical_column(case: str) -> None:
    forward, inputs, _blocks = alias_shape_case(case, "cpu")
    outputs = forward(*inputs)
    _graph, _converter, module, _source = alias_shape_backward(forward, inputs, outputs)
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel),
            (*(torch.ones_like(output) for output in outputs), *inputs),
        )
        tree = ast.parse(bound.to_code(bound.config_spec.default_config()))

    # Locate the shared Boolean input by its allocation, without depending on
    # generated variable numbering. The mask is expanded with stride zero,
    # then _mask_to supplies a contiguous fake value before fragment emission.
    allocations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "allocate_tensor"
        and node.value.args
        and ast.unparse(node.value.args[0]) == "cutlass.Boolean"
    ]
    assert len(allocations) == 1
    buffer = ast.unparse(allocations[0].targets[0])
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and ast.unparse(node.value) == buffer
    ]
    assert len(reads) == 1
    read = reads[0]
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    current: ast.AST = read
    loops = []
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.For):
            loops.append(current)
    assert len(loops) == 2
    inner, outer = loops
    assignment = next(
        node
        for node in inner.body
        if isinstance(node, ast.Assign) and read in ast.walk(node)
    )
    assert isinstance(assignment.value, ast.IfExp)

    # Evaluate only the emitted integer index and its guard. Every logical
    # column must read its own mask element, including columns above one.
    # Using the replacement fake's strides read element zero and limited the
    # valid extent to two, despite the sixteen-element input.
    observed = []
    for column in range(inputs[1].numel()):
        values = {ast.unparse(outer.target): column, ast.unparse(inner.target): 0}
        for statement in inner.body:
            if statement is assignment:
                break
            assert isinstance(statement, ast.Assign) and len(statement.targets) == 1
            values[ast.unparse(statement.targets[0])] = eval(
                compile(
                    ast.Expression(statement.value), "<fragment-coordinate>", "eval"
                ),
                {"__builtins__": {}},
                values,
            )
        observed.append(
            (
                eval(
                    compile(
                        ast.Expression(assignment.value.test),
                        "<fragment-guard>",
                        "eval",
                    ),
                    {"__builtins__": {}},
                    values,
                ),
                eval(
                    compile(ast.Expression(read.slice), "<fragment-index>", "eval"),
                    {"__builtins__": {}},
                    values,
                ),
            )
        )
    assert observed == [(True, column) for column in range(inputs[1].numel())]
