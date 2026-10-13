"""Translated, injective store coordinates independent of native lane ownership.

This proof does not grant completion, storage omission or alias permission. The
completed-store caller retains its original runtime storage guards and typed
store renderer. Data-derived scatter indices deliberately have no such proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ...language import tile_ops
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileIdOrigin
from . import chained_matmul as chain
from .chained_grid_scalar import grid_scalar_bounds
from .chained_grid_scalar import grid_scalar_expression
from .chained_store_expression import _node_facts

if TYPE_CHECKING:
    import sympy

    from .chained_matmul import ChainedMatmulPlan


@dataclass(frozen=True)
class _Index:
    axis: int | None
    scale: int
    bounds: tuple[int, int]


def _checked(value: _Index, dtype: torch.dtype) -> _Index:
    if dtype not in (torch.int32, torch.int64):
        raise chain._UnsupportedChain("noninteger completed-store selector")
    limits = torch.iinfo(dtype)
    if not limits.min <= value.bounds[0] <= value.bounds[1] <= limits.max:
        raise chain._UnsupportedChain("overflowing completed-store selector")
    return value


class _Selectors:
    def __init__(self, plan: ChainedMatmulPlan) -> None:
        self.plan = plan
        self.env = CompileEnvironment.current()
        self.origins = {
            axis: (0, (extent - 1) // block * block)
            for axis, extent, block in plan.axes
        }
        if plan.loop is not None:
            begin = self.scalar(plan.loop.begin).bounds
            end = self.scalar(plan.loop.end).bounds
            if begin[0] >= end[1]:
                raise chain._UnsupportedChain("empty completed-store loop")
            self.origins[plan.loop.block_id] = (begin[0], end[1] - 1)

    def block(self, node: Node) -> int:
        axis = self.env.resolve_block_id(node.meta.get("val"))
        if axis is None:
            raise chain._UnsupportedChain("unbound completed-store selector")
        return self.env.canonical_block_id(axis)

    def scalar(self, value: object) -> _Index:
        dtype = self.env.index_dtype
        if type(value) is int:
            return _checked(_Index(None, 0, (value, value)), dtype)
        if not isinstance(value, Node):
            raise chain._UnsupportedChain("unknown completed-store scalar")
        if value.target in (tile_ops.tile_begin, tile_ops.tile_id):
            axis = self.block(cast("Node", value.args[0]))
            bounds = self.origins.get(axis)
            if bounds is None:
                raise chain._UnsupportedChain("inactive completed-store origin")
            if value.target is tile_ops.tile_id:
                block = next(size for key, _, size in self.plan.axes if key == axis)
                bounds = (bounds[0] // block, bounds[1] // block)
            return _checked(_Index(None, 0, bounds), dtype)
        origin = chain._Expression._grid_origin(value)
        expr = grid_scalar_expression(value)
        if origin is not None or expr is not None:
            symbolic = value.meta.get("val")
            assert isinstance(symbolic, torch.SymInt)
            expr = symbolic._sympy_()
            symbols = {}
            for symbol in expr.free_symbols:
                info = HostFunction.current().expr_to_origin.get(symbol)
                if info is None or type(info.origin) not in (
                    GridOrigin,
                    TileBeginOrigin,
                    TileIdOrigin,
                ):
                    raise chain._UnsupportedChain("unknown completed-store origin")
                axis = self.env.canonical_block_id(
                    cast("GridOrigin", info.origin).block_id
                )
                bounds = self.origins.get(axis)
                if bounds is None:
                    raise chain._UnsupportedChain("inactive completed-store origin")
                if isinstance(info.origin, TileIdOrigin):
                    block = next(size for key, _, size in self.plan.axes if key == axis)
                    bounds = (bounds[0] // block, bounds[1] // block)
                if not -(2**31) <= bounds[0] <= bounds[1] < 2**31:
                    raise chain._UnsupportedChain("overflowing physical origin")
                symbols[cast("sympy.Symbol", symbol)] = bounds
            limits = torch.iinfo(dtype)
            bounds = grid_scalar_bounds(expr, symbols, (limits.min, limits.max))
            if bounds is None:
                raise chain._UnsupportedChain("unproved completed-store scalar")
            return _Index(None, 0, bounds)
        if value.target in (
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.mod,
        ):
            return self.binary(
                value.target,
                self.scalar(value.args[0]),
                self.scalar(value.args[1]),
                dtype,
            )
        if value.target is _tracing_ops._get_symnode:
            symbolic = value.meta.get("val")
            if type(symbolic) is int:
                return self.scalar(symbolic)
        raise chain._UnsupportedChain("data-derived completed-store scalar")

    @staticmethod
    def binary(
        target: object, left: _Index, right: _Index, dtype: torch.dtype
    ) -> _Index:
        if left.axis is not None and right.axis is not None:
            raise chain._UnsupportedChain("combined completed-store coordinate axes")
        axis = left.axis if left.axis is not None else right.axis
        if target in (
            operator.add,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
        ):
            scale = left.scale + right.scale
            bounds = (
                left.bounds[0] + right.bounds[0],
                left.bounds[1] + right.bounds[1],
            )
        elif target in (
            operator.sub,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub.Scalar,
        ):
            scale = left.scale - right.scale
            bounds = (
                left.bounds[0] - right.bounds[1],
                left.bounds[1] - right.bounds[0],
            )
        elif target in (
            operator.mul,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
        ):
            uniform = right if left.axis is not None else left
            if axis is not None and uniform.bounds[0] != uniform.bounds[1]:
                raise chain._UnsupportedChain(
                    "varying completed-store coordinate scale"
                )
            scale = (
                left.scale * right.bounds[0]
                if left.axis is not None
                else right.scale * left.bounds[0]
            )
            products = [a * b for a in left.bounds for b in right.bounds]
            bounds = (min(products), max(products))
        elif target in (operator.floordiv, operator.mod):
            # Only uniform, constant-denominator coordinates are admitted.
            # The original scalar renderer keeps Python's signed semantics;
            # this proof bounds that same result, including min-int / -1.
            if axis is not None or right.bounds[0] != right.bounds[1]:
                raise chain._UnsupportedChain("varying completed-store divisor")
            divisor = right.bounds[0]
            if divisor == 0:
                raise chain._UnsupportedChain("zero completed-store divisor")
            if divisor == -1 and left.bounds[0] == torch.iinfo(dtype).min:
                raise chain._UnsupportedChain("overflowing completed-store division")
            scale = 0
            if target is operator.floordiv:
                quotients = tuple(value // divisor for value in left.bounds)
                bounds = (min(quotients), max(quotients))
            else:
                bounds = (0, divisor - 1) if divisor > 0 else (divisor + 1, 0)
        else:
            raise chain._UnsupportedChain(
                "unsupported completed-store index arithmetic"
            )
        return _checked(_Index(axis, scale, bounds), dtype)

    def vector(self, node: Node, coordinate: _Index) -> _Index:
        shape = chain._shape(node)
        dtype = node.meta["val"].dtype
        if len(shape) != 1 or shape[0] != coordinate.bounds[1] + 1:
            raise chain._UnsupportedChain("nonseparable completed-store selector")
        if node.target is tile_index:
            axis = self.block(cast("Node", node.args[0]))
            bounds = self.origins.get(axis)
            if bounds is None:
                raise chain._UnsupportedChain("inactive completed-store origin")
            return self.binary(operator.add, coordinate, _Index(None, 0, bounds), dtype)
        if node.target is torch.ops.prims.iota.default:
            start, step = node.kwargs.get("start", 0), node.kwargs.get("step", 1)
            if type(start) is not int or type(step) is not int:
                raise chain._UnsupportedChain("dynamic completed-store iota")
            return self.binary(
                operator.add,
                self.binary(operator.mul, coordinate, self.scalar(step), dtype),
                self.scalar(start),
                dtype,
            )
        if node.target in (
            _tracing_ops._new_var,
            torch.ops.prims.convert_element_type.default,
        ):
            source = cast("Node", node.args[0])
            if chain._shape(source) != shape or node.kwargs:
                raise chain._UnsupportedChain(
                    "changed completed-store conversion domain"
                )
            if node.target is torch.ops.prims.convert_element_type.default and (
                len(node.args) != 2 or node.args[1] != dtype
            ):
                raise chain._UnsupportedChain("changed completed-store conversion")
            return _checked(self.vector(source, coordinate), dtype)
        if node.target in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub.Scalar,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
        ):
            if len(node.args) != 2 or any(
                key != "alpha" or value != 1 for key, value in node.kwargs.items()
            ):
                raise chain._UnsupportedChain("changed completed-store arithmetic")
            args = [
                self.vector(arg, coordinate)
                if isinstance(arg, Node)
                and isinstance(arg.meta.get("val"), torch.Tensor)
                else self.scalar(arg)
                for arg in node.args
            ]
            return self.binary(node.target, args[0], args[1], dtype)
        raise chain._UnsupportedChain("data-derived completed-store vector")


@dataclass(frozen=True)
class StoreOwnership:
    store: Node
    shape: tuple[int, int]
    axes: tuple[int | None, ...]
    facts: tuple[object, ...]


def plan_store_ownership(plan: ChainedMatmulPlan, store: Node) -> StoreOwnership | None:
    """Prove two translated unit-coordinate selectors into distinct target axes."""
    try:
        if store.target is not memory_ops.store or store.kwargs or len(store.args) != 4:
            return None
        target, selectors, value = store.args[:3]
        if (
            not isinstance(target, Node)
            or target.target is not _tracing_ops._host_tensor
            or not isinstance(selectors, (list, tuple))
            or not isinstance(value, Node)
        ):
            return None
        shape = chain._shape(value)
        if (
            len(shape) != 2
            or min(shape) <= 0
            or len(selectors) != target.meta["val"].ndim
        ):
            return None
        proof = _Selectors(plan)
        axes: list[int | None] = []
        dim = 0
        for selector in selectors:
            scalar_grid = isinstance(selector, Node) and (
                chain._Expression._grid_origin(selector) is not None
                or grid_scalar_expression(selector) is not None
            )
            vector = not scalar_grid and (
                selector == slice(None)
                or isinstance(selector, Node)
                and (
                    selector.target
                    in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int)
                    or isinstance(selector.meta.get("val"), torch.Tensor)
                    and bool(chain._shape(selector))
                )
            )
            if vector:
                if dim >= 2:
                    return None
                coordinate = _Index(dim, 1, (0, shape[dim] - 1))
                if selector == slice(None):
                    index = coordinate
                elif isinstance(selector, Node) and isinstance(
                    selector.meta.get("val"), torch.Tensor
                ):
                    index = proof.vector(selector, coordinate)
                else:
                    assert isinstance(selector, Node)
                    bounds = proof.origins.get(proof.block(selector))
                    if bounds is None:
                        return None
                    index = proof.binary(
                        operator.add,
                        coordinate,
                        _Index(None, 0, bounds),
                        proof.env.index_dtype,
                    )
                if index.axis != dim or abs(index.scale) != 1:
                    return None
                axes.append(dim)
                dim += 1
            else:
                proof.scalar(selector)
                axes.append(None)
        if dim != 2:
            return None
        pending = [store]
        nodes: set[Node] = set()
        while pending:
            node = pending.pop()
            if node in nodes:
                continue
            nodes.add(node)
            pending.extend(node.all_input_nodes)
        ordered = tuple(node for node in store.graph.nodes if node in nodes)
        return StoreOwnership(
            store, cast("tuple[int, int]", shape), tuple(axes), _node_facts(ordered)
        )
    except (chain._UnsupportedChain, StopIteration):
        return None
