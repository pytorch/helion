"""Tile-contained rank-two modular auxiliary accesses for native epilogues.

One logical coordinate may repeat an original physical axis through a literal
positive modulus. This is not a bijection or a claim that source M*N covers the
logical grid: it preserves the original rank-two pointer/strides and selects the
same complete physical tile. Readonly and alias proofs belong to the complete
single-anchor epilogue DAG, not to this index-only classifier.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import sympy
import torch

from .split_store import integer_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class ModularAuxLoadMap:
    extents: tuple[int, int]
    shape: tuple[int, int]
    strides: tuple[int, int]
    split_axis: int
    # Source axes that carry the modular logical coordinate and the other one.
    order: tuple[int, int]
    owned_index_nodes: tuple[torch.fx.Node, ...] = ()

    @property
    def period(self) -> int:
        return self.shape[self.order[0]]

    def offset(self, row: int, col: int) -> int:
        coordinates = (row, col)
        return (
            coordinates[self.split_axis] % self.period * self.strides[self.order[0]]
            + coordinates[1 - self.split_axis] * self.strides[self.order[1]]
        )

    def tile_reason(self, rows: int, cols: int) -> str | None:
        if any(type(value) is not int or value <= 0 for value in (rows, cols)):
            return "modular auxiliary tiles require positive static integer dimensions"
        if any(
            extent % tile
            for extent, tile in zip(self.extents, (rows, cols), strict=True)
        ):
            return "modular auxiliary loads require complete M/N tiles"
        if self.period % (rows, cols)[self.split_axis]:
            return "modular auxiliary tile must divide its source period"
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
        reason = self.tile_reason(rows, cols)
        if reason is not None:
            raise ValueError(reason)
        modular, other = self.order
        axes = (modular, other) if self.split_axis == 0 else (other, modular)
        # Reorder only the device view's axes if required; neither allocate nor
        # recast, reshape a host tensor, add an offset, or fabricate metadata.
        source = (
            f"{result_name} = cute.make_tensor({tensor_name}.iterator, "
            f"cute.make_layout(({tensor_name}.shape[{axes[0]}], "
            f"{tensor_name}.shape[{axes[1]}]), "
            f"stride=({tensor_name}.layout.stride[{axes[0]}], "
            f"{tensor_name}.layout.stride[{axes[1]}])))"
        )
        coordinates = [tile_m, tile_n]
        axis = self.split_axis
        coordinates[axis] = (
            f"cutlass.Int32({coordinates[axis]}) % {self.period // (rows, cols)[axis]}"
        )
        return source, (coordinates[0], coordinates[1])


def prove_modular_aux_load_map(
    indices: Sequence[sympy.Expr],
    symbols: tuple[sympy.Symbol, sympy.Symbol],
    extents: tuple[int, int],
    shape: Sequence[int],
    strides: Sequence[int],
) -> ModularAuxLoadMap | None:
    """Exact literal-modulus map with positive dense, nonoverlapping storage."""
    if len(indices) != 2 or len(shape) != 2 or len(strides) != 2:
        return None
    if symbols[0] == symbols[1] or any(
        sympy.ask(sympy.Q.integer(s)) is not True for s in symbols
    ):
        return None
    if any(
        type(value) is not int or not 1 < value < 2**31 for value in (*extents, *shape)
    ):
        return None
    if any(type(value) is not int or not 0 < value < 2**31 for value in strides):
        return None
    if math.prod(shape) >= 2**31 or math.prod(extents) >= 2**31:
        return None
    # Both row-major and column-major source storage are admitted, including
    # logical axis permutations, but gaps, overlap and negative strides are not.
    span = 1
    for axis in sorted((0, 1), key=lambda axis: strides[axis]):
        if strides[axis] != span:
            return None
        span *= shape[axis]
    for split_axis in (0, 1):
        for modular, other in ((0, 1), (1, 0)):
            period = shape[modular]
            if shape[other] != extents[1 - split_axis]:
                continue
            expected = sympy.Mod(symbols[split_axis], period)
            if (
                indices[modular] == expected
                and indices[other] == symbols[1 - split_axis]
            ):
                return ModularAuxLoadMap(
                    extents,
                    (shape[0], shape[1]),
                    (strides[0], strides[1]),
                    split_axis,
                    (modular, other),
                )
    return None


def modular_aux_load_map(
    node: torch.fx.Node,
    *,
    carrier_tile_shape: tuple[object, ...] | None,
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None,
    carrier_global_shape: tuple[object, ...] | None,
) -> ModularAuxLoadMap | None:
    """Parse bounded typed index provenance, never shape-only axis matching.

    The grammar is intentionally narrower than the mixed-radix rank-three
    parser: original tile_index, its exact row/column singleton index view and
    one positive literal remainder. Integer overflow, arithmetic before modulo,
    inferred reshapes, casts and unknown calls have no transport proof here.
    """
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
        or any(index.graph is not node.graph for index in carrier_tile_index_nodes)
        or carrier_global_shape is None
        or len(carrier_global_shape) != 2
        or any(
            type(value) is not int or not 1 < value < 2**31
            for value in carrier_global_shape
        )
    ):
        return None
    source, indices = node.args[:2]
    if (
        not isinstance(source, torch.fx.Node)
        or source.graph is not node.graph
        or not isinstance(indices, (tuple, list))
        or len(indices) != 2
    ):
        return None
    host, result = source.meta.get("val"), node.meta.get("val")
    if (
        not isinstance(host, torch.Tensor)
        or host.ndim != 2
        or host.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not isinstance(result, torch.Tensor)
        or result.dtype != host.dtype
        or result.device != host.device
        or tuple(result.shape) != carrier_tile_shape
    ):
        return None
    shape, strides = tuple(host.shape), tuple(host.stride())
    if any(type(value) is not int for value in (*shape, *strides)):
        return None
    rows, cols = carrier_global_shape
    if not isinstance(rows, int) or not isinstance(cols, int):
        return None
    row = sympy.Symbol("modular_aux_row", integer=True, nonnegative=True)
    col = sympy.Symbol("modular_aux_col", integer=True, nonnegative=True)
    symbols = (row, col)
    bounds = {row: (0, rows - 1), col: (0, cols - 1)}
    memo: dict[torch.fx.Node, sympy.Expr] = {}
    active: set[torch.fx.Node] = set()
    modulo_nodes: list[tuple[int, int]] = []

    def expression(index: object, depth: int = 0) -> sympy.Expr | None:
        if (
            not isinstance(index, torch.fx.Node)
            or index.graph is not node.graph
            or depth > 32
            or len(memo) >= 64
            or index in active
        ):
            return None
        if index in memo:
            return memo[index]
        value = index.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype is not torch.int32
            or value.device != result.device
        ):
            return None
        if index.op != "call_function" or index.kwargs:
            return None
        active.add(index)
        parsed: sympy.Expr | None = None
        if index.target is tile_ops.tile_index and len(index.args) == 1:
            for axis in (0, 1):
                if index.args[0] is carrier_tile_index_nodes[axis] and tuple(
                    value.shape
                ) == (carrier_tile_shape[axis],):
                    parsed = symbols[axis]
                    break
        elif (
            index.target
            in (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
            and len(index.args) == 2
        ):
            source_index, target = index.args
            child = expression(source_index, depth + 1)
            if (
                child is not None
                and isinstance(source_index, torch.fx.Node)
                and isinstance(target, (tuple, list))
            ):
                source_value = source_index.meta.get("val")
                axes = [
                    axis for axis in (0, 1) if child.free_symbols == {symbols[axis]}
                ]
                if len(axes) == 1:
                    axis = axes[0]
                    wanted = (
                        (carrier_tile_index_nodes[0], 1)
                        if axis == 0
                        else (1, carrier_tile_index_nodes[1])
                    )
                    metadata = (
                        (carrier_tile_shape[0], 1)
                        if axis == 0
                        else (1, carrier_tile_shape[1])
                    )
                    if (
                        isinstance(source_value, torch.Tensor)
                        and tuple(source_value.shape) == (carrier_tile_shape[axis],)
                        and tuple(target) == wanted
                        and tuple(value.shape) == metadata
                    ):
                        parsed = child
        elif index.target is torch.ops.aten.remainder.Scalar and len(index.args) == 2:
            source_index, modulus = index.args
            child = expression(source_index, depth + 1)
            source_value = (
                source_index.meta.get("val")
                if isinstance(source_index, torch.fx.Node)
                else None
            )
            if (
                type(modulus) is int
                and 1 < modulus < 2**31
                and child in symbols
                and isinstance(source_value, torch.Tensor)
                and tuple(source_value.shape) == tuple(value.shape)
            ):
                axis = symbols.index(child)
                remainder = sympy.Mod(child, modulus)
                if isinstance(remainder, sympy.Expr):
                    parsed = remainder
                    modulo_nodes.append((axis, modulus))
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
        value = index.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        if parsed.free_symbols == {row}:
            expected = (carrier_tile_shape[0], 1)
        elif parsed.free_symbols == {col}:
            expected = (1, carrier_tile_shape[1])
        else:
            return None
        if tuple(value.shape) != expected:
            return None
        parsed_indices.append(parsed)
    if len(modulo_nodes) != 1:
        return None
    mapping = prove_modular_aux_load_map(
        parsed_indices, symbols, (rows, cols), shape, strides
    )
    if (
        mapping is None
        or modulo_nodes != [(mapping.split_axis, mapping.period)]
        or mapping.tile_reason(64, 32) is not None
    ):
        return None
    # Existing liveness DCE may drop only these proven index computations once
    # replaced by source coordinates. Any other reader keeps them alive.
    return dataclasses.replace(mapping, owned_index_nodes=tuple(memo))
