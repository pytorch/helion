from __future__ import annotations

import ast
import copy

import pytest
import torch

from .test_cute_chained_row_collective_emission import _args
from .test_cute_chained_row_collective_emission import _capture
from helion._compiler.cute import chained_matmul as chain


def _assert_original_expressions(
    cg, plan, frame, candidate, boundaries, operands, lines
):
    assert lines is not None
    tag = f"chain_row_collective_{candidate.first_event}"
    row, column = f"{tag}_row", f"{tag}_column"
    tree = ast.parse("\n".join(lines))
    definitions = {}
    retained = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and not target.id.startswith(tag):
                definitions[target.id] = node.value
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id.startswith(tag + "_retained_")
            ):
                retained[target.value.id] = node.value
    sums = {
        f"{tag}_sum_{index}": ast.parse(
            f"{buffer.name}[{row}{', 0' if len(buffer.shape) == 2 else ''}]",
            mode="eval",
        ).body
        for index, buffer in enumerate(candidate.buffers)
    }

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            value = definitions.get(node.id, sums.get(node.id))
            return self.visit(copy.deepcopy(value)) if value is not None else node

        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in retained:
                return self.visit(copy.deepcopy(retained[node.value.id]))
            return self.generic_visit(node)

    def expanded(value):
        return ast.dump(Expand().visit(copy.deepcopy(value)), include_attributes=False)

    def original_value(node, coords, original_boundaries):
        expression = chain._Expression(cg, plan, original_boundaries)
        expression.coordinate_names.update((row, column))
        value = expression.value(node, coords)
        dtype = {
            torch.float32: "cutlass.Float32",
            torch.bfloat16: "cutlass.BFloat16",
            torch.float16: "cutlass.Float16",
        }[node.meta["val"].dtype]
        value = chain._masked_operand(
            value, dtype, chain._operand_domain(cg, node, coords, plan)
        )
        for statement in ast.parse("\n".join(expression.lines)).body:
            if isinstance(statement, ast.Assign) and isinstance(
                statement.targets[0], ast.Name
            ):
                definitions[statement.targets[0].id] = statement.value
        return expanded(ast.parse(value, mode="eval").body)

    for index, operation in enumerate(candidate.collectives):
        actual = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign)
            and ast.unparse(node.target) == f"{tag}_acc_{index}"
            and "shuffle_sync_down" not in ast.unparse(node.value)
        )
        assert expanded(actual) == original_value(
            operation.source, (row, column), boundaries
        )
    original_boundaries = {
        **boundaries,
        **{
            operation.node: buffer.name
            for operation, buffer in zip(
                candidate.collectives, candidate.buffers, strict=True
            )
        },
    }
    for operand in operands:
        coords = operand.geometry.operand(operand.role, row, column)[1]
        expected = original_value(operand.node, coords, original_boundaries)
        stores = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and ast.unparse(node.targets[0])
            == f"{operand.target}[{row} + {operand.offset}, {column}]"
        ]
        assert len(stores) == 2
        assert expanded(stores[0].value) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [16, 48, 64])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_retained_expansion_keeps_original_sum_and_output_masks_and_casts(
    dtype, width, keepdim, nonlinear
):
    _capture(
        args=_args(dtype, width, keepdim=keepdim, nonlinear=nonlinear),
        check=_assert_original_expressions,
    )
