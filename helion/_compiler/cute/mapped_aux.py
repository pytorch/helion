"""Bounded quotient/remainder auxiliary loads for ordinary TCGEN05 epilogues.

The source remains an original rank-three argument. A complete digit map proves
its logical two-dimensional access; a tile-local view selects one source plane
without changing its dtype, strides, index association or pointer ownership.
No gather, mask, overlapping storage, partial tile or integer wrap is admitted.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import TYPE_CHECKING

import sympy
import torch

from .split_store import _normalize
from .split_store import integer_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class MappedAuxLoadMap:
    """One logical axis is a complete pair of source mixed-radix digits."""

    extents: tuple[int, int]
    shape: tuple[int, int, int]
    strides: tuple[int, int, int]
    split_axis: int
    order: tuple[int, int, int]  # outer digit, inner digit, unsplit axis
    owned_index_nodes: tuple[torch.fx.Node, ...] = ()

    @property
    def period(self) -> int:
        return self.shape[self.order[1]]

    def offset(self, row: int, col: int) -> int:
        coordinates = (row, col)
        split = coordinates[self.split_axis]
        other = coordinates[1 - self.split_axis]
        outer, inner, unsplit = self.order
        return (
            split // self.period * self.strides[outer]
            + split % self.period * self.strides[inner]
            + other * self.strides[unsplit]
        )

    def tile_reason(self, rows: int, cols: int) -> str | None:
        if any(type(value) is not int or value <= 0 for value in (rows, cols)):
            return "mapped auxiliary tiles require positive static integer dimensions"
        if any(
            extent % tile
            for extent, tile in zip(self.extents, (rows, cols), strict=True)
        ):
            return "mapped auxiliary loads require complete M/N tiles"
        if self.period % (rows, cols)[self.split_axis]:
            return "mapped auxiliary tile must divide its source split period"
        return None

    def view_source(
        self,
        result_name: str,
        tensor_name: str,
        tile_m: str,
        tile_n: str,
        rows: int,
        cols: int,
    ) -> tuple[str, tuple[str, str]]:
        """Select a source plane and return its local tile coordinates."""
        reason = self.tile_reason(rows, cols)
        if reason is not None:
            raise ValueError(reason)
        outer, inner, unsplit = self.order
        tiles_per_plane = self.period // (rows, cols)[self.split_axis]
        split_coord = (tile_m, tile_n)[self.split_axis]
        base = ["cutlass.Int32(0)"] * 3
        base[outer] = f"cutlass.Int32({split_coord}) // {tiles_per_plane}"
        if self.split_axis == 0:
            axes = (inner, unsplit)
            local = (f"cutlass.Int32({tile_m}) % {tiles_per_plane}", tile_n)
        else:
            axes = (unsplit, inner)
            local = (tile_m, f"cutlass.Int32({tile_n}) % {tiles_per_plane}")
        source = (
            f"{result_name} = cute.make_tensor("
            f"{tensor_name}.iterator + cute.crd2idx(({', '.join(base)}), "
            f"{tensor_name}.layout), "
            f"cute.make_layout(({tensor_name}.shape[{axes[0]}], "
            f"{tensor_name}.shape[{axes[1]}]), "
            f"stride=({tensor_name}.layout.stride[{axes[0]}], "
            f"{tensor_name}.layout.stride[{axes[1]}])))"
        )
        return source, local


def prove_mapped_aux_load_map(
    indices: Sequence[sympy.Expr],
    symbols: tuple[sympy.Symbol, sympy.Symbol],
    extents: tuple[int, int],
    shape: Sequence[int],
    strides: Sequence[int],
) -> MappedAuxLoadMap | None:
    """Prove all coordinates and dense storage, never sampled pointer equality."""
    if len(indices) != 3 or len(shape) != 3 or len(strides) != 3:
        return None
    if symbols[0] == symbols[1] or any(
        sympy.ask(sympy.Q.integer(symbol)) is not True for symbol in symbols
    ):
        return None
    if any(type(size) is not int or size <= 1 for size in (*extents, *shape)):
        return None
    if any(type(stride) is not int or stride <= 0 for stride in strides):
        return None
    if math.prod(shape) != math.prod(extents) or math.prod(shape) >= 2**31:
        return None
    span = 1
    for axis in sorted(range(3), key=lambda axis: strides[axis]):
        if strides[axis] != span:
            return None
        span *= shape[axis]
    bounds = dict(zip(symbols, ((0, extents[0] - 1), (0, extents[1] - 1)), strict=True))
    normalized = tuple(_normalize(index, bounds) for index in indices)
    if any(index is None for index in normalized):
        return None
    for split_axis in (0, 1):
        for order in itertools.permutations(range(3)):
            outer, inner, unsplit = order
            if shape[outer] * shape[inner] != extents[split_axis]:
                continue
            if shape[unsplit] != extents[1 - split_axis]:
                continue
            split = symbols[split_axis]
            quotient = sympy.floor(sympy.Mul(split, sympy.Rational(1, shape[inner])))
            remainder = sympy.Mod(split, shape[inner])
            if not isinstance(remainder, sympy.Expr):
                return None
            expected: list[sympy.Expr | None] = [None] * 3
            expected[outer] = _normalize(quotient, bounds)
            expected[inner] = _normalize(remainder, bounds)
            expected[unsplit] = symbols[1 - split_axis]
            if normalized == tuple(expected):
                return MappedAuxLoadMap(
                    extents,
                    (shape[0], shape[1], shape[2]),
                    (strides[0], strides[1], strides[2]),
                    split_axis,
                    (outer, inner, unsplit),
                )
    return None


def mapped_aux_load_map(
    node: torch.fx.Node,
    *,
    carrier_tile_shape: tuple[object, ...] | None,
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None,
    carrier_global_shape: tuple[object, ...] | None,
) -> MappedAuxLoadMap | None:
    """Recognize exact typed index provenance for a rank-three source load."""
    # Language modules import the CuTe epilogue analyzer during initialization.
    # Resolve the tracing targets here rather than creating that import cycle.
    from ...language import memory_ops
    from ...language import tile_ops

    if (
        node.op != "call_function"
        or node.target is not memory_ops.load
        or node.kwargs
        or not 2 <= len(node.args) <= 4
        or any(value is not None for value in node.args[2:])
        or carrier_tile_shape is None
        or len(carrier_tile_shape) != 2
        or carrier_tile_index_nodes is None
        or len(carrier_tile_index_nodes) != 2
        or carrier_tile_index_nodes[0] is carrier_tile_index_nodes[1]
        or carrier_global_shape is None
        or len(carrier_global_shape) != 2
        or any(
            type(value) is not int or not 1 < value < 2**31
            for value in carrier_global_shape
        )
    ):
        return None
    tensor, indices = node.args[:2]
    if (
        not isinstance(tensor, torch.fx.Node)
        or not isinstance(indices, (tuple, list))
        or len(indices) != 3
    ):
        return None
    host = tensor.meta.get("val")
    result = node.meta.get("val")
    if (
        not isinstance(host, torch.Tensor)
        or host.ndim != 3
        or host.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not isinstance(result, torch.Tensor)
        or result.dtype != host.dtype
        or tuple(result.shape) != carrier_tile_shape
    ):
        return None
    shape = tuple(host.shape)
    strides = tuple(host.stride())
    if any(type(value) is not int for value in (*shape, *strides)):
        return None
    m_extent, n_extent = carrier_global_shape
    if not isinstance(m_extent, int) or not isinstance(n_extent, int):
        return None
    extents = (m_extent, n_extent)
    row = sympy.Symbol("mapped_aux_row", integer=True, nonnegative=True)
    col = sympy.Symbol("mapped_aux_col", integer=True, nonnegative=True)
    symbols = (row, col)
    bounds = {row: (0, extents[0] - 1), col: (0, extents[1] - 1)}
    memo: dict[torch.fx.Node, sympy.Expr] = {}
    active: set[torch.fx.Node] = set()
    visited: set[torch.fx.Node] = set()

    def expression(index: object) -> sympy.Expr | None:
        if type(index) is int and 0 <= index < 2**31:
            return sympy.Integer(index)
        if not isinstance(index, torch.fx.Node) or index.graph is not node.graph:
            return None
        if index in memo:
            return memo[index]
        if index in active or len(visited) >= 64:
            return None
        visited.add(index)
        active.add(index)
        val = index.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype != torch.int32:
            active.remove(index)
            return None
        parsed: sympy.Expr | None = None
        if (
            index.op == "call_function"
            and index.target is tile_ops.tile_index
            and not index.kwargs
            and len(index.args) == 1
        ):
            for axis in (0, 1):
                if index.args[0] is carrier_tile_index_nodes[axis] and tuple(
                    val.shape
                ) == (carrier_tile_shape[axis],):
                    parsed = symbols[axis]
                    break
        elif (
            index.op == "call_function"
            and index.target
            in (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
            and not index.kwargs
            and len(index.args) == 2
        ):
            source, target_shape = index.args
            child = expression(source)
            if child in symbols and isinstance(target_shape, (tuple, list)):
                axis = symbols.index(child)
                expected_shape = (
                    (carrier_tile_index_nodes[0], 1)
                    if axis == 0
                    else (1, carrier_tile_index_nodes[1])
                )
                expected_metadata = (
                    (carrier_tile_shape[0], 1)
                    if axis == 0
                    else (1, carrier_tile_shape[1])
                )
                if (
                    tuple(target_shape) == expected_shape
                    and tuple(val.shape) == expected_metadata
                ):
                    parsed = child
        elif index.op == "call_function" and len(index.args) == 2:
            left, right = index.args
            left_value = (
                left.meta.get("val") if isinstance(left, torch.fx.Node) else None
            )
            lhs = expression(left)
            rhs = expression(right)
            if (
                isinstance(left_value, torch.Tensor)
                and tuple(val.shape) == tuple(left_value.shape)
                and type(right) is int
                and lhs is not None
                and rhs is not None
            ):
                if index.target is torch.ops.aten.add.Tensor and not index.kwargs:
                    parsed = sympy.Add(lhs, rhs)
                elif index.target is torch.ops.aten.mul.Tensor and not index.kwargs:
                    parsed = sympy.Mul(lhs, rhs)
                elif 0 < right < 2**31:
                    if (
                        index.target is torch.ops.aten.div.Tensor_mode
                        and index.kwargs == {"rounding_mode": "floor"}
                    ):
                        parsed = sympy.floor(sympy.Mul(lhs, sympy.Rational(1, right)))
                    elif (
                        index.target is torch.ops.aten.remainder.Scalar
                        and not index.kwargs
                    ):
                        remainder = sympy.Mod(lhs, right)
                        if isinstance(remainder, sympy.Expr):
                            parsed = remainder
        active.remove(index)
        if parsed is None:
            return None
        interval = integer_bounds(parsed, bounds)
        if interval is None or interval[0] < 0 or interval[1] >= 2**31:
            return None
        memo[index] = parsed
        return parsed

    parsed_indices: list[sympy.Expr] = []
    for index in indices:
        parsed = expression(index)
        if parsed is None or not isinstance(index, torch.fx.Node):
            return None
        val = index.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return None
        # A bare vector would align to the wrong broadcast axis for square
        # tiles. Require each final digit's exact row/column broadcast shape.
        axes = parsed.free_symbols
        if axes == {row}:
            expected = (carrier_tile_shape[0], 1)
        elif axes == {col}:
            expected = (1, carrier_tile_shape[1])
        else:
            return None
        if tuple(val.shape) != expected:
            return None
        parsed_indices.append(parsed)
    mapping = prove_mapped_aux_load_map(
        parsed_indices,
        symbols,
        extents,
        shape,
        strides,
    )
    if mapping is None or mapping.tile_reason(64, 32) is not None:
        # The ordinary TCGEN05 owner has a 64-row one-CTA instruction. Its
        # 32-column geometry is also safe for FP32-as-TF32. Require at least
        # this complete grid before native-only search admission; unsupported
        # periods keep the original collective/universal configuration domain.
        return None
    # These are only the typed, pure index operations just proved above.
    # GenerateAST's existing liveness DCE keeps any assignment still read by
    # another consumer; no source statement is removed unconditionally.
    return dataclasses.replace(mapping, owned_index_nodes=tuple(memo))
