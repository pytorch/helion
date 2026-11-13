"""Prove separable logical coordinates of an explicitly indexed storage view.

The map preserves the physical tensor and never invents affine tensor strides
for a folded logical axis. Consumers must explicitly implement this address
contract; recognizing a map alone does not admit an ordinary MMA transfer.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from ....language import tile_ops
from ....language import view_ops
from ...host_function import HostFunction
from ...variable_origin import TileBeginOrigin
from ...variable_origin import TileIdOrigin
from ..cute_mma_dag import AXIS_RDIM
from ..cute_mma_dag import AXIS_ROOT
from ..cute_mma_dag import _NotMine
from ..fx_matcher import _is_call

if TYPE_CHECKING:
    from torch.fx import Node

    from ..cute_mma_dag import Axis
    from ..cute_mma_dag import _Matcher


@dataclasses.dataclass(frozen=True)
class FoldedIndexAxis:
    """One logical coordinate split into major and minor physical steps."""

    axis: Axis
    symbol: sympy.Symbol
    origin: Node
    divisor: int
    major_stride: int
    minor_stride: int

    @property
    def expression(self) -> sympy.Expr:
        return sympy.Add(
            sympy.Mul(FloorDiv(self.symbol, self.divisor), self.major_stride),
            sympy.Mul(PythonMod(self.symbol, self.divisor), self.minor_stride),
        )


@dataclasses.dataclass(frozen=True)
class SeparableIndexMap:
    """Element offset from a uniform base and independently folded axes.

    Logical coordinates are zero-based with unit stride.
    All integer operations in the original recipe fit their recorded dtype
    over valid logical coordinates. The physical offset is in bounds for the
    storage view. Padding coordinates are not covered by this proof and must
    remain masked by the eventual consumer.
    """

    base: sympy.Expr
    axes: tuple[FoldedIndexAxis, ...]
    expression: sympy.Expr
    dtype: torch.dtype


class _UnsupportedIndexMap(Exception):
    pass


class _IndexMapProof:
    def __init__(self, matcher: _Matcher) -> None:
        self.matcher = matcher
        self.noncanonical = (
            HostFunction.current().device_ir.noncanonical_task_origin_block_ids
        )
        self.ranges: dict[sympy.Symbol, ValueRanges] = {}
        self.origins: dict[Axis, Node] = {}
        self.symbols: dict[Axis, sympy.Symbol] = {}
        self.memo: dict[Node, sympy.Expr] = {}

    def bounds(self, expression: sympy.Expr) -> ValueRanges:
        for symbol in expression.free_symbols - self.ranges.keys():
            info = HostFunction.current().expr_to_origin.get(symbol)
            if (
                not isinstance(symbol, sympy.Symbol)
                or info is None
                or not isinstance(info.origin, (TileBeginOrigin, TileIdOrigin))
                or not self.matcher.lead_coordinate(info.origin)
            ):
                raise _UnsupportedIndexMap
            axis = self.matcher.axis_of_block(info.origin.block_id)
            if axis.block_id in self.noncanonical:
                raise _UnsupportedIndexMap
            self.ranges[symbol] = ValueRanges(0, axis.extent - 1)
        return bound_sympy(expression, self.ranges)

    def check_dtype(self, node: Node, expression: sympy.Expr) -> None:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return
        if value.dtype not in (torch.int32, torch.int64):
            raise _UnsupportedIndexMap
        bounds = self.bounds(expression)
        limits = torch.iinfo(value.dtype)
        if bounds.lower < limits.min or bounds.upper > limits.max:
            raise _UnsupportedIndexMap

    def axis_expression(self, node: Node, origin: Node) -> sympy.Expr:
        axes = self.matcher.shape_axes(node)
        if len(axes) != 1 or axes[0] is None:
            raise _UnsupportedIndexMap
        axis = axes[0]
        if axis.kind not in (AXIS_ROOT, AXIS_RDIM) or axis.extent <= 0:
            raise _UnsupportedIndexMap
        if axis.block_id in self.noncanonical:
            raise _UnsupportedIndexMap
        if axis in self.origins and self.origins[axis] is not origin:
            raise _UnsupportedIndexMap
        self.origins[axis] = origin
        symbol = self.symbols.setdefault(
            axis, sympy.Symbol(f"_mma_map_axis_{axis.index}", integer=True)
        )
        self.ranges[symbol] = ValueRanges(0, axis.extent - 1)
        return symbol

    def expression(self, value: object) -> sympy.Expr:
        if type(value) is int:
            return sympy.Integer(value)
        if not isinstance(value, torch.fx.Node):
            raise _UnsupportedIndexMap
        node = self.matcher.resolve(value)
        if node in self.memo:
            return self.memo[node]
        fake = node.meta.get("val")
        if isinstance(fake, (int, torch.SymInt)) and not isinstance(fake, bool):
            result, loops = self.matcher.uniform_expr(node)
            if loops:
                raise _UnsupportedIndexMap
            self.bounds(result)
        elif _is_call(node, tile_ops.tile_index):
            if len(node.args) != 1 or not isinstance(node.args[0], torch.fx.Node):
                raise _UnsupportedIndexMap
            result = self.axis_expression(node, node.args[0])
        elif _is_call(node, torch.ops.prims.iota.default):
            if (
                len(node.args) != 1
                or type(node.args[0]) is not int
                or node.kwargs.get("start", 0) != 0
                or node.kwargs.get("step", 1) != 1
            ):
                raise _UnsupportedIndexMap
            result = self.axis_expression(node, node)
        elif _is_call(node, view_ops.subscript):
            if (
                len(node.args) != 2
                or not isinstance(node.args[1], (tuple, list))
                or any(
                    part is not None and part != slice(None) for part in node.args[1]
                )
            ):
                raise _UnsupportedIndexMap
            self.matcher.shape_axes(node)
            result = self.expression(node.args[0])
        elif _is_call(node, torch.ops.aten.expand.default):
            self.matcher.shape_axes(node)
            result = self.expression(node.args[0])
        elif _is_call(node, torch.ops.prims.convert_element_type.default):
            if len(node.args) != 2 or node.args[1] not in (torch.int32, torch.int64):
                raise _UnsupportedIndexMap
            result = self.expression(node.args[0])
        elif node.target in (
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.mod,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor_mode,
            torch.ops.aten.floor_divide.default,
            torch.ops.aten.remainder.Tensor,
            torch.ops.aten.remainder.Scalar,
        ):
            if len(node.args) != 2 or node.kwargs.get("alpha", 1) != 1:
                raise _UnsupportedIndexMap
            left, right = (self.expression(arg) for arg in node.args)
            if node.target in (operator.add, torch.ops.aten.add.Tensor):
                result = sympy.Add(left, right)
            elif node.target in (operator.sub, torch.ops.aten.sub.Tensor):
                result = sympy.Add(left, sympy.Mul(-1, right))
            elif node.target in (operator.mul, torch.ops.aten.mul.Tensor):
                result = sympy.Mul(left, right)
            else:
                if not isinstance(right, sympy.Integer) or right <= 0:
                    raise _UnsupportedIndexMap
                if node.target in (
                    operator.mod,
                    torch.ops.aten.remainder.Tensor,
                    torch.ops.aten.remainder.Scalar,
                ):
                    result = cast("sympy.Expr", PythonMod(left, right))
                else:
                    if (
                        node.target == torch.ops.aten.div.Tensor_mode
                        and node.kwargs.get("rounding_mode") != "floor"
                    ):
                        raise _UnsupportedIndexMap
                    result = cast("sympy.Expr", FloorDiv(left, right))
        else:
            raise _UnsupportedIndexMap
        self.check_dtype(node, result)
        self.memo[node] = result
        return result

    def folded_axis(self, axis: Axis, expression: sympy.Expr) -> FoldedIndexAxis:
        symbol = self.symbols[axis]
        if expression.free_symbols - {symbol}:
            raise _UnsupportedIndexMap
        expression = sympy.expand(expression)
        linear = expression.coeff(symbol)
        if expression == linear * symbol and isinstance(linear, sympy.Integer):
            if linear <= 0:
                raise _UnsupportedIndexMap
            return FoldedIndexAxis(axis, symbol, self.origins[axis], 1, int(linear), 0)
        divisors = {
            int(part.args[1])
            for part in sympy.preorder_traversal(expression)
            if isinstance(part, (FloorDiv, PythonMod))
            and part.args[0] == symbol
            and isinstance(part.args[1], sympy.Integer)
        }
        if len(divisors) != 1:
            raise _UnsupportedIndexMap
        divisor = divisors.pop()
        if divisor <= 1:
            raise _UnsupportedIndexMap
        major = expression.coeff(FloorDiv(symbol, divisor))
        minor = expression.coeff(PythonMod(symbol, divisor))
        if not all(
            isinstance(step, sympy.Integer) and step > 0 for step in (major, minor)
        ):
            raise _UnsupportedIndexMap
        result = FoldedIndexAxis(
            axis, symbol, self.origins[axis], divisor, int(major), int(minor)
        )
        if sympy.expand(result.expression - expression) != 0:
            raise _UnsupportedIndexMap
        return result


def prove_separable_index_map(
    matcher: _Matcher, node: Node, storage_extent: int
) -> SeparableIndexMap | None:
    """Prove a rank-one or rank-two integer recipe over root/iota axes."""
    fake = node.meta.get("val")
    if (
        not isinstance(fake, torch.Tensor)
        or fake.ndim not in (1, 2)
        or fake.dtype not in (torch.int32, torch.int64)
        or storage_extent <= 0
    ):
        return None
    proof = _IndexMapProof(matcher)
    try:
        expression = sympy.expand(proof.expression(node))
        inferred_axes = matcher.shape_axes(node)
        if any(axis is None or axis not in proof.symbols for axis in inferred_axes):
            return None
        axes = tuple(axis for axis in inferred_axes if axis is not None)
        symbols = {proof.symbols[axis]: 0 for axis in axes}
        base = expression.xreplace(symbols)
        mapped = tuple(
            proof.folded_axis(
                axis,
                expression.xreplace(
                    {symbol: 0 for symbol in symbols if symbol != proof.symbols[axis]}
                )
                - base,
            )
            for axis in axes
        )
        if (
            sympy.expand(
                sympy.Add(
                    base,
                    *(part.expression for part in mapped),
                    sympy.Mul(-1, expression),
                )
            )
            != 0
        ):
            return None
        bounds = proof.bounds(expression)
        if bounds.lower < 0 or bounds.upper >= storage_extent:
            return None
        return SeparableIndexMap(base, mapped, expression, fake.dtype)
    except (_UnsupportedIndexMap, _NotMine):
        return None
