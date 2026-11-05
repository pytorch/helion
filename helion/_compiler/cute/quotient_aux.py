"""Bounded many-to-one readonly quotient auxiliaries and scalar tile transport.

Unlike a store digit map, this read map needs neither a remainder partner nor
injectivity. Only complete, regular quotient plateaus/nested repeats are lowered.
Readonly and writer-alias obligations belong to the complete anchored DAG.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from .split_store import _normalize
from .split_store import integer_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class QuotientAuxLoadMap:
    extents: tuple[int, int]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    source_axes: tuple[int, int]
    quotient_axis: int
    divisor: int
    storage_offset: int
    storage_elements: int
    element_size: int
    owned_index_nodes: tuple[torch.fx.Node, ...] = ()

    def runtime_view_matches(self, tensor: torch.Tensor) -> bool:
        """Do not generate from live tensors contradicting the bound view."""
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        offset, nbytes = tensor.storage_offset(), tensor.untyped_storage().nbytes()
        if any(type(v) is not int for v in (*shape, *strides, offset, nbytes)):
            return False
        return bool(
            shape == self.shape
            and strides == self.strides
            and offset == self.storage_offset
            and tensor.element_size() == self.element_size
            and nbytes // self.element_size == self.storage_elements
        )

    def offset(self, row: int, col: int) -> int:
        coordinates = [row, col]
        coordinates[self.quotient_axis] //= self.divisor
        return sum(
            coordinate * self.strides[axis]
            for coordinate, axis in zip(coordinates, self.source_axes, strict=True)
        )

    def tile_reason(self, rows: int, cols: int) -> str | None:
        if any(type(v) is not int or v <= 0 for v in (rows, cols)):
            return "quotient auxiliary tiles require positive static dimensions"
        if any(e % t for e, t in zip(self.extents, (rows, cols), strict=True)):
            return "quotient auxiliary loads require complete M/N tiles"
        tile = (rows, cols)[self.quotient_axis]
        if tile % self.divisor and self.divisor % tile:
            return "valid quotient read lacks a regular plateau/nested scalar transport"
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
        """A tile-local no-copy logical view of the ORIGINAL compact storage.

        A quotient plateau has stride zero on that logical axis. A nested
        repeat has j = r + c*q and offset r*0 + q*source_stride. Other phases
        are not approximated by an affine layout.
        """
        if reason := self.tile_reason(rows, cols):
            raise ValueError(reason)
        origins = [
            f"cutlass.Int32({tile_m}) * {rows}",
            f"cutlass.Int32({tile_n}) * {cols}",
        ]
        origins[self.quotient_axis] = (
            f"({origins[self.quotient_axis]}) // {self.divisor}"
        )
        weights = [f"{tensor_name}.layout.stride[{axis}]" for axis in self.source_axes]
        offset = " + ".join(
            f"({origin}) * {stride}"
            for origin, stride in zip(origins, weights, strict=True)
        )
        shapes = [str(rows), str(cols)]
        tile = (rows, cols)[self.quotient_axis]
        if tile <= self.divisor:
            weights[self.quotient_axis] = "0"
        else:
            shapes[self.quotient_axis] = f"({self.divisor}, {tile // self.divisor})"
            weights[self.quotient_axis] = f"(0, {weights[self.quotient_axis]})"
        source = (
            f"{result_name} = cute.make_tensor(cute.make_ptr("
            f"{tensor_name}.element_type, ({tensor_name}.iterator + ({offset})).toint(), "
            f"cute.AddressSpace.gmem, assumed_align={self.element_size}), "
            f"cute.make_layout(({', '.join(shapes)}), stride=({', '.join(weights)})))"
        )
        # The view already owns this tile; do not select the global tile twice.
        return source, ("0", "0")


def prove_quotient_aux_load_map(
    indices: Sequence[sympy.Expr],
    symbols: tuple[sympy.Symbol, sympy.Symbol],
    extents: tuple[int, int],
    shape: Sequence[int],
    strides: Sequence[int],
    *,
    storage_offset: int,
    storage_elements: int,
    element_size: int = 4,
) -> QuotientAuxLoadMap | None:
    """Prove exact coordinate bounds and original strided storage, not samples."""
    if type(element_size) is not int or element_size not in (2, 4):
        return None
    if (
        len(shape) not in (2, 3)
        or len(indices) != len(shape)
        or len(strides) != len(shape)
    ):
        return None
    if symbols[0] == symbols[1] or any(
        sympy.ask(sympy.Q.integer(s)) is not True for s in symbols
    ):
        return None
    if any(
        type(v) is not int or not 0 < v < 2**31
        for v in (*extents, *shape, storage_elements)
    ):
        return None
    if type(storage_offset) is not int or not 0 <= storage_offset < 2**31:
        return None
    if any(type(v) is not int or not 0 <= v < 2**31 for v in strides):
        return None
    span = storage_offset + sum(
        (e - 1) * s for e, s in zip(shape, strides, strict=True)
    )
    if span >= storage_elements or span >= 2**31:
        return None
    bounds = dict(zip(symbols, ((0, extents[0] - 1), (0, extents[1] - 1)), strict=True))
    normalized = tuple(_normalize(index, bounds) for index in indices)
    for quotient_axis in (0, 1):
        for q_axis, expression in enumerate(normalized):
            if expression == symbols[quotient_axis]:
                divisor = 1
            elif isinstance(expression, sympy.floor):
                numerator, denominator = sympy.fraction(expression.args[0])
                if numerator != symbols[quotient_axis] or not isinstance(
                    denominator, sympy.Integer
                ):
                    continue
                divisor = int(denominator)
            else:
                continue
            if not 0 < divisor < 2**31:
                continue
            for other_axis, other in enumerate(normalized):
                if other_axis == q_axis or other != symbols[1 - quotient_axis]:
                    continue
                axes = (
                    (q_axis, other_axis) if quotient_axis == 0 else (other_axis, q_axis)
                )
                if any(
                    normalized[axis] != 0
                    for axis in range(len(shape))
                    if axis not in axes
                ):
                    continue
                if (
                    shape[q_axis] < (extents[quotient_axis] - 1) // divisor + 1
                    or shape[other_axis] < extents[1 - quotient_axis]
                ):
                    continue
                return QuotientAuxLoadMap(
                    extents,
                    tuple(shape),
                    tuple(strides),
                    axes,
                    quotient_axis,
                    divisor,
                    storage_offset,
                    storage_elements,
                    element_size,
                )
    return None


def quotient_aux_load_map(
    node: torch.fx.Node,
    *,
    carrier_tile_shape: tuple[object, ...] | None,
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None,
    carrier_global_shape: tuple[object, ...] | None,
) -> QuotientAuxLoadMap | None:
    """Parse only typed original carrier indices, exact views, one literal floor."""
    from ...language import memory_ops
    from ...language import tile_ops
    from ..compile_environment import CompileEnvironment

    if (
        node.op != "call_function"
        or node.target is not memory_ops.load
        or node.kwargs
        or not 2 <= len(node.args) <= 4
        or any(v is not None for v in node.args[2:])
    ):
        return None
    if (
        carrier_tile_shape is None
        or len(carrier_tile_shape) != 2
        or carrier_tile_index_nodes is None
        or len(carrier_tile_index_nodes) != 2
        or carrier_tile_index_nodes[0] is carrier_tile_index_nodes[1]
    ):
        return None
    if (
        carrier_global_shape is None
        or len(carrier_global_shape) != 2
        or any(type(v) is not int or not 0 < v < 2**31 for v in carrier_global_shape)
    ):
        return None
    source, indices = node.args[:2]
    if (
        not isinstance(source, torch.fx.Node)
        or source.graph is not node.graph
        or not isinstance(indices, (tuple, list))
    ):
        return None
    host, result = source.meta.get("val"), node.meta.get("val")
    if (
        not isinstance(host, torch.Tensor)
        or host.ndim not in (2, 3)
        or host.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not isinstance(result, torch.Tensor)
        or result.dtype != host.dtype
        or result.device != host.device
        or tuple(result.shape) != carrier_tile_shape
        or len(indices) != host.ndim
    ):
        return None
    shape, strides = tuple(host.shape), tuple(host.stride())
    env = CompileEnvironment.current()
    runtime = env.runtime_value_for_tensor(host)
    if isinstance(runtime, torch.Tensor) and not isinstance(
        runtime, torch._subclasses.fake_tensor.FakeTensor
    ):
        if (
            runtime.dtype != host.dtype
            or tuple(runtime.shape) != shape
            or tuple(runtime.stride()) != strides
        ):
            return None
        # FakeTensor storage may describe only the normalized view. Preserve
        # the ORIGINAL input's concrete offset/allocation span when available.
        offset, nbytes = runtime.storage_offset(), runtime.untyped_storage().nbytes()
    else:
        offset, nbytes = host.storage_offset(), host.untyped_storage().nbytes()
    if (
        any(type(v) is not int for v in (*shape, *strides, offset, nbytes))
        or nbytes % host.element_size()
    ):
        return None
    row = sympy.Symbol("quotient_aux_row", integer=True, nonnegative=True)
    col = sympy.Symbol("quotient_aux_col", integer=True, nonnegative=True)
    symbols = (row, col)
    m_extent, n_extent = carrier_global_shape
    assert isinstance(m_extent, int) and isinstance(n_extent, int)
    extents = (m_extent, n_extent)
    bounds = {row: (0, extents[0] - 1), col: (0, extents[1] - 1)}
    memo: dict[torch.fx.Node, sympy.Expr] = {}
    active: set[torch.fx.Node] = set()
    quotient_nodes: dict[torch.fx.Node, tuple[int, int]] = {}

    def expression(index: object, depth: int = 0) -> sympy.Expr | None:
        if type(index) is int and index == 0:
            return sympy.Integer(0)
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
        if not isinstance(value, torch.Tensor) or value.dtype is not torch.int32:
            return None
        active.add(index)
        parsed: sympy.Expr | None = None
        if (
            index.target is tile_ops.tile_index
            and not index.kwargs
            and len(index.args) == 1
        ):
            for axis in (0, 1):
                if index.args[0] is carrier_tile_index_nodes[axis] and tuple(
                    value.shape
                ) == (carrier_tile_shape[axis],):
                    parsed = symbols[axis]
        elif (
            index.target
            in (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
            and not index.kwargs
            and len(index.args) == 2
        ):
            child = expression(index.args[0], depth + 1)
            if child in symbols:
                axis = symbols.index(child)
                target = (
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
                    isinstance(index.args[1], (list, tuple))
                    and tuple(index.args[1]) == target
                    and tuple(value.shape) == metadata
                ):
                    parsed = child
        elif (
            index.target is torch.ops.aten.div.Tensor_mode
            and index.kwargs == {"rounding_mode": "floor"}
            and len(index.args) == 2
        ):
            left, divisor = index.args
            child = expression(left, depth + 1)
            if (
                child in symbols
                and type(divisor) is int
                and 0 < divisor < 2**31
                and isinstance(left, torch.fx.Node)
                and tuple(value.shape) == tuple(left.meta["val"].shape)
            ):
                parsed = sympy.floor(sympy.Mul(child, sympy.Rational(1, divisor)))
                quotient_nodes[index] = (symbols.index(child), divisor)
        active.remove(index)
        if parsed is None:
            return None
        interval = integer_bounds(parsed, bounds)
        if interval is None or interval[0] < 0 or interval[1] >= 2**31:
            return None
        memo[index] = parsed
        return parsed

    parsed = tuple(expression(index) for index in indices)
    if any(v is None for v in parsed) or len(quotient_nodes) != 1:
        return None
    for index, value in zip(indices, parsed, strict=True):
        if value == 0:
            continue
        assert isinstance(value, sympy.Expr)
        expected = (
            (carrier_tile_shape[0], 1)
            if value.free_symbols == {row}
            else (1, carrier_tile_shape[1])
            if value.free_symbols == {col}
            else None
        )
        if (
            not isinstance(index, torch.fx.Node)
            or tuple(index.meta["val"].shape) != expected
        ):
            return None
    assert all(isinstance(v, sympy.Expr) for v in parsed)
    assert isinstance(offset, int)
    mapping = prove_quotient_aux_load_map(
        tuple(cast("sympy.Expr", v) for v in parsed),
        symbols,
        extents,
        shape,
        strides,
        storage_offset=offset,
        storage_elements=nbytes // host.element_size(),
        element_size=host.element_size(),
    )
    if mapping is None:
        return None
    ((quotient_axis, divisor),) = quotient_nodes.values()
    if mapping.divisor != divisor:
        return None
    if divisor == 1:
        # Algebraic identity does not erase the original index-axis ownership.
        mapping = dataclasses.replace(mapping, quotient_axis=quotient_axis)
    elif mapping.quotient_axis != quotient_axis:
        return None
    if mapping.tile_reason(64, 32) is not None:
        return None
    return dataclasses.replace(mapping, owned_index_nodes=tuple(memo))
