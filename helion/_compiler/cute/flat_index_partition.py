"""Prove disjoint column partitions of flattened integer-indexed storage."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import sympy
import torch

from ...language import _tracing_ops
from ...language import tile_ops
from ...language import view_ops

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def is_flat_column_partition(
    index: object, block_id: int, env: CompileEnvironment
) -> bool:
    """Prove ``index % extent == tile_index % extent`` for every integer input.

    The extent must divide the index arithmetic's power-of-two modulus, so
    signed/unsigned overflow cannot change the column residue. This accepts
    arbitrary row expressions multiplied by the row width without assuming
    valid, unique, or bounded gather indices. Unknown arithmetic fails closed.
    """
    size = env.block_sizes[block_id].size
    if not isinstance(size, (int, torch.SymInt)):
        return False
    extent = env.specialize_expr(sympy.sympify(size))
    if not isinstance(extent, sympy.Integer):
        return False
    width = int(extent)
    if width <= 1 or width & (width - 1):
        return False
    integers = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
    memo: dict[torch.fx.Node, tuple[int, int] | None] = {}

    def constant(value: object) -> int | None:
        if type(value) is int:
            return value
        if (
            isinstance(value, torch.fx.Node)
            and value.target is _tracing_ops._get_symnode
        ):
            scalar = value.meta.get("val")
            if isinstance(scalar, (int, torch.SymInt)):
                resolved = env.specialize_expr(sympy.sympify(scalar))
                if isinstance(resolved, sympy.Integer):
                    return int(resolved)
        return None

    def residue(value: object, depth: int = 0) -> tuple[int, int] | None:
        if (number := constant(value)) is not None:
            return 0, number % width
        if not isinstance(value, torch.fx.Node):
            return None
        if value in memo:
            return memo[value]
        if depth > 128 or len(memo) >= 4096:
            return None
        memo[value] = None
        tensor = value.meta.get("val")
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype not in integers
            or (1 << (8 * tensor.element_size())) % width
        ):
            return None
        result = None
        if value.target is tile_ops.tile_index and tensor.ndim == 1:
            if env.resolve_block_id(tensor.shape[0]) == block_id:
                result = (1, 0)
        elif value.target is view_ops.subscript:
            subscripts = value.args[1]
            if isinstance(subscripts, (list, tuple)) and all(
                item is None or isinstance(item, slice) and item == slice(None)
                for item in subscripts
            ):
                result = residue(value.args[0], depth + 1)
        elif value.target in (
            _tracing_ops._new_var,
            torch.ops.aten.unsqueeze.default,
            torch.ops.prims.convert_element_type.default,
        ):
            result = residue(value.args[0], depth + 1)
        elif (
            value.target
            in (
                operator.add,
                operator.sub,
                operator.mul,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.mul.Tensor,
            )
            and len(value.args) == 2
            and not value.kwargs
        ):
            left, right = value.args
            if value.target in (operator.mul, torch.ops.aten.mul.Tensor):
                for scalar, other in ((left, right), (right, left)):
                    factor = constant(scalar)
                    if factor is None:
                        continue
                    # The other input must itself be integer-valued, even
                    # when multiplication by the width erases its residue.
                    other_value = (
                        other.meta.get("val")
                        if isinstance(other, torch.fx.Node)
                        else None
                    )
                    if not (
                        type(other) is int
                        or isinstance(other_value, torch.Tensor)
                        and other_value.dtype in integers
                    ):
                        continue
                    if factor % width == 0:
                        result = (0, 0)
                    elif (parts := residue(other, depth + 1)) is not None:
                        result = (parts[0] * factor % width, parts[1] * factor % width)
                    break
            else:
                a, b = residue(left, depth + 1), residue(right, depth + 1)
                if a is not None and b is not None:
                    sign = (
                        -1
                        if value.target in (operator.sub, torch.ops.aten.sub.Tensor)
                        else 1
                    )
                    result = (
                        (a[0] + sign * b[0]) % width,
                        (a[1] + sign * b[1]) % width,
                    )
        memo[value] = result
        return result

    return residue(index) == (1, 0)
