"""Batch-digit output stores for batched (leading-axis) native GEMMs.

A batched matmul's output is sometimes stored through a mixed-radix split of
its batch axis, for example Inductor's head-split ``out[b // H, m, b % H, n]``.
This module proves that such a store addresses exactly the logical
``(batch, M, N)`` output:

* every subscript is typed integer index IR over the three output tiles
  (tile indices, broadcast views, additions, constant multiples, and floor
  division or remainder by positive constants); nothing else is parsed;
* after exact bound normalization, the subscripts are one complete pair of
  batch digits (``floor(b / c)``, ``b mod c``) plus the whole M and N indices;
* the destination is dense, so the digit map is a bijection onto the output,
  and its (M, N) plane has unit stride on M or N.

The batch tile is a single coordinate, so each output tile lies in one
``(M, N)`` plane. That plane's base is computed from the proved digits, and the
plane keeps the destination's own strides. Only static shapes are admitted.
Anything else keeps the existing refusal.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import TYPE_CHECKING

import sympy
import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language.atomic_ops import ATOMIC_OPS
from ..runtime_tensor_guards import register_tensor_alias_specializations
from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
from .split_store import _normalize
from .split_store import integer_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import GraphInfo

_INT32_LIMIT = 2**31
# Bounds the parsed index expression DAG of one store (four subscripts of a
# few nodes each in every admitted form); a larger DAG is refused, not walked.
_MAX_INDEX_NODES = 64


@dataclasses.dataclass(frozen=True, eq=False)
class BatchDigitOutputMap:
    """One store's batch axis is a complete pair of destination digits."""

    store_node: torch.fx.Node
    shape: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]
    extents: tuple[int, int, int]  # (batch, M, N)
    outer_dim: int  # destination dim indexed by floor(batch / period)
    inner_dim: int  # destination dim indexed by batch mod period
    m_dim: int
    n_dim: int
    owned_index_nodes: tuple[torch.fx.Node, ...] = ()

    def layout_key(self) -> tuple[object, ...]:
        """Everything the lowering reads, without the proof's graph nodes."""
        return (
            self.shape,
            self.strides,
            self.extents,
            self.outer_dim,
            self.inner_dim,
            self.m_dim,
            self.n_dim,
        )

    @property
    def period(self) -> int:
        return self.shape[self.inner_dim]

    @property
    def plane_column_major(self) -> bool:
        """Whether the (M, N) plane is M-contiguous."""
        return self.strides[self.m_dim] == 1

    def offset(self, batch: int, row: int, col: int) -> int:
        """The destination element offset of logical ``(batch, row, col)``."""
        return (
            batch // self.period * self.strides[self.outer_dim]
            + batch % self.period * self.strides[self.inner_dim]
            + row * self.strides[self.m_dim]
            + col * self.strides[self.n_dim]
        )

    def plane_view_source(
        self, result_name: str, tensor_name: str, batch_index: str
    ) -> str:
        """Select the batch's (M, N) plane over the original output pointer.

        The plane base is ``crd2idx`` of the proved digit coordinates in the
        destination's own layout; the view keeps the destination's M and N
        extents and strides. No allocation or copy is made.
        """
        coords = ["cutlass.Int32(0)"] * 4
        coords[self.outer_dim] = (
            f"cutlass.Int32({batch_index}) // cutlass.Int32({self.period})"
        )
        coords[self.inner_dim] = (
            f"cutlass.Int32({batch_index}) % cutlass.Int32({self.period})"
        )
        m, n = self.m_dim, self.n_dim
        return (
            f"{result_name} = cute.make_tensor("
            f"{tensor_name}.iterator + cute.crd2idx(({', '.join(coords)}), "
            f"{tensor_name}.layout), "
            f"cute.make_layout(({tensor_name}.shape[{m}], {tensor_name}.shape[{n}]), "
            f"stride=({tensor_name}.layout.stride[{m}], "
            f"{tensor_name}.layout.stride[{n}])))"
        )


def prove_batch_digit_output_dims(
    indices: Sequence[sympy.Expr],
    symbols: tuple[sympy.Symbol, sympy.Symbol, sympy.Symbol],
    extents: tuple[int, int, int],
    shape: Sequence[int],
    strides: Sequence[int],
) -> tuple[int, int, int, int] | None:
    """Return ``(outer, inner, m, n)`` destination dims, or no proof.

    Proves from the normalized expressions and the static destination layout
    only: complete digits, whole M and N, dense storage (hence a bijection
    with disjoint element intervals), and a unit-stride plane axis.
    """
    if len(indices) != 4 or len(shape) != 4 or len(strides) != 4:
        return None
    if len(set(symbols)) != 3:
        return None
    if any(type(value) is not int or value <= 0 for value in (*extents, *shape)):
        return None
    if any(type(stride) is not int or stride <= 0 for stride in strides):
        return None
    if math.prod(shape) != math.prod(extents) or math.prod(shape) >= _INT32_LIMIT:
        return None
    span = 1
    for axis in sorted(range(4), key=lambda axis: (strides[axis], axis)):
        if strides[axis] != span:
            return None
        span *= shape[axis]
    bounds = {
        symbol: (0, extent - 1) for symbol, extent in zip(symbols, extents, strict=True)
    }
    normalized = tuple(_normalize(index, bounds) for index in indices)
    if any(index is None for index in normalized):
        return None
    batch, row, col = symbols
    for outer, inner, m, n in itertools.permutations(range(4)):
        period = shape[inner]
        if (
            shape[outer] < 2
            or period < 2
            or shape[outer] * period != extents[0]
            or shape[m] != extents[1]
            or shape[n] != extents[2]
            or 1 not in (strides[m], strides[n])
        ):
            continue
        quotient = sympy.floor(sympy.Mul(batch, sympy.Rational(1, period)))
        remainder = sympy.Mod(batch, period)
        if not isinstance(remainder, sympy.Expr):
            continue
        expected: list[sympy.Expr | None] = [None] * 4
        expected[outer] = _normalize(quotient, bounds)
        expected[inner] = _normalize(remainder, bounds)
        expected[m] = row
        expected[n] = col
        if normalized == tuple(expected):
            return outer, inner, m, n
    return None


def batch_digit_output_map(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    store: torch.fx.Node,
    *,
    output_block_ids: tuple[int, ...],
    input_tensors: Sequence[torch.Tensor],
    allow_unbound: bool = False,
) -> BatchDigitOutputMap | None:
    """Recognize a batch-digit store from its exact typed index provenance.

    ``output_block_ids`` are the matmul's (batch, M, N) tile block ids, and
    ``input_tensors`` are the matmul's source operands. The destination must
    be a static, dense, offset-free kernel argument that no other load or
    store in the kernel touches, and it must have a registered POSITIVE
    runtime storage-disjointness fact against every operand
    (``tensor_storage_disjoint_matrix_v1``): different argument names or
    tensor objects are not a disjointness proof. ``allow_unbound`` reads the
    registered classifier's live facts during search admission, before the
    bound facts are recorded; codegen keeps the default and requires the
    bound, cache-specialized fact.
    """
    if (
        len(output_block_ids) != 3
        or store.op != "call_function"
        or store.target is not memory_ops.store
        or store.kwargs
        or not 3 <= len(store.args) <= 4
        or (len(store.args) == 4 and store.args[3] is not None)
    ):
        return None
    tensor_node, indices = store.args[:2]
    if not isinstance(tensor_node, torch.fx.Node) or not isinstance(
        indices, (list, tuple)
    ):
        return None
    tensor = tensor_node.meta.get("val")
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 4
        or len(indices) != 4
        or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or env.tensor_input_source(tensor) is None
    ):
        return None
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    storage_offset = tensor.storage_offset()
    if (
        any(type(value) is not int for value in (*shape, *strides))
        or type(storage_offset) is not int
        or storage_offset != 0
    ):
        return None
    if not _program_effects_owned(env, graphs, store, tensor, input_tensors):
        return None
    if not _destination_storage_proven_disjoint(
        env, tensor, input_tensors, allow_unbound=allow_unbound
    ):
        return None
    extents_list: list[int] = []
    for block_id in output_block_ids:
        size = env.block_sizes[block_id].size
        if type(size) is not int or not 0 < size < _INT32_LIMIT:
            return None
        extents_list.append(size)
    extents = (extents_list[0], extents_list[1], extents_list[2])
    names = ("batch_digit_b", "batch_digit_m", "batch_digit_n")
    symbols = tuple(
        sympy.Symbol(name, integer=True, nonnegative=True) for name in names
    )
    assert len(symbols) == 3
    bounds = {
        symbol: (0, extent - 1) for symbol, extent in zip(symbols, extents, strict=True)
    }
    memo: dict[torch.fx.Node, sympy.Expr] = {}
    active: set[torch.fx.Node] = set()

    def tile_axis(node: torch.fx.Node) -> int | None:
        if (
            node.op != "call_function"
            or node.target is not tile_ops.tile_index
            or node.kwargs
            or len(node.args) != 1
            or not isinstance(node.args[0], torch.fx.Node)
        ):
            return None
        size = node.args[0].meta.get("val")
        if not isinstance(size, (int, torch.SymInt)):
            return None
        block_id = env.get_block_id(size)
        if block_id not in output_block_ids:
            return None
        return output_block_ids.index(block_id)

    def int32_tensor(node: torch.fx.Node) -> torch.Tensor | None:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and value.dtype == torch.int32:
            return value
        return None

    def placed(value: torch.Tensor, parsed: sympy.Expr) -> bool:
        """Each axis of the index broadcast is 1 or that axis's own tile.

        Torch right-aligns index broadcasting, so a bare rank-1 tile index
        runs along the last (N) axis whatever tile it indexes; the subscript
        may depend on a symbol only along that symbol's own axis.
        """
        if value.ndim > 3:
            return False
        sizes = (1,) * (3 - value.ndim) + tuple(value.shape)
        for axis, size in enumerate(sizes):
            unit = type(size) is int and size == 1
            own_tile = isinstance(size, (int, torch.SymInt)) and (
                env.get_block_id(size) == output_block_ids[axis]
            )
            if not (unit or own_tile):
                return False
            if unit and symbols[axis] in parsed.free_symbols:
                return False
        return True

    def expression(index: object) -> sympy.Expr | None:
        if type(index) is int:
            return sympy.Integer(index) if 0 <= index < _INT32_LIMIT else None
        if not isinstance(index, torch.fx.Node) or index.graph is not store.graph:
            return None
        if index in memo:
            return memo[index]
        if index in active or len(memo) >= _MAX_INDEX_NODES:
            return None
        value = int32_tensor(index)
        if value is None:
            return None
        active.add(index)
        parsed: sympy.Expr | None = None
        target = index.target
        if index.op != "call_function":
            parsed = None
        elif target is tile_ops.tile_index:
            axis = tile_axis(index)
            if axis is not None and value.ndim == 1:
                parsed = symbols[axis]
        elif (
            target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
            and not index.kwargs
            and len(index.args) == 2
            and isinstance(index.args[0], torch.fx.Node)
        ):
            source = index.args[0]
            axis = tile_axis(source)
            if axis is not None and _broadcast_shape_matches(
                env, index.args[1], value, axis, output_block_ids[axis]
            ):
                child = expression(source)
                if child is symbols[axis]:
                    parsed = child
        elif not index.kwargs and len(index.args) == 2:
            left, right = index.args
            lhs, rhs = expression(left), expression(right)
            if lhs is not None and rhs is not None:
                if target is torch.ops.aten.add.Tensor:
                    parsed = sympy.Add(lhs, rhs)
                elif target is torch.ops.aten.mul.Tensor and (
                    type(left) is int or type(right) is int
                ):
                    parsed = sympy.Mul(lhs, rhs)
                elif (
                    target is torch.ops.aten.remainder.Scalar
                    and type(right) is int
                    and right > 0
                ):
                    remainder = sympy.Mod(lhs, right)
                    if isinstance(remainder, sympy.Expr):
                        parsed = remainder
        elif (
            target is torch.ops.aten.div.Tensor_mode
            and index.kwargs == {"rounding_mode": "floor"}
            and len(index.args) == 2
            and type(index.args[1]) is int
            and index.args[1] > 0
        ):
            lhs = expression(index.args[0])
            if lhs is not None:
                parsed = sympy.floor(sympy.Mul(lhs, sympy.Rational(1, index.args[1])))
        active.discard(index)
        if parsed is None:
            return None
        if target is not tile_ops.tile_index and not placed(value, parsed):
            return None
        interval = integer_bounds(parsed, bounds)
        if interval is None or interval[0] < 0 or interval[1] >= _INT32_LIMIT:
            return None
        memo[index] = parsed
        return parsed

    parsed_indices: list[sympy.Expr] = []
    for index in indices:
        parsed = expression(index)
        if parsed is None:
            return None
        if isinstance(index, torch.fx.Node):
            value = int32_tensor(index)
            if value is None or not placed(value, parsed):
                return None
        parsed_indices.append(parsed)
    dims = prove_batch_digit_output_dims(
        parsed_indices, symbols, extents, shape, strides
    )
    if dims is None:
        return None
    outer, inner, m, n = dims
    return BatchDigitOutputMap(
        store,
        (shape[0], shape[1], shape[2], shape[3]),
        (strides[0], strides[1], strides[2], strides[3]),
        extents,
        outer,
        inner,
        m,
        n,
        owned_index_nodes=tuple(memo),
    )


def _broadcast_shape_matches(
    env: CompileEnvironment,
    target_shape: object,
    value: torch.Tensor,
    axis: int,
    block_id: int,
) -> bool:
    """A tile index viewed into its own axis of a rank-3 broadcast, exactly."""
    if not isinstance(target_shape, (list, tuple)) or len(target_shape) != 3:
        return False
    if value.ndim != 3:
        return False
    for dim, (entry, size) in enumerate(zip(target_shape, value.shape, strict=True)):
        if dim == axis:
            if not isinstance(entry, torch.fx.Node):
                return False
            size = entry.meta.get("val")
            if not isinstance(size, (int, torch.SymInt)):
                return False
            if env.get_block_id(size) != block_id:
                return False
        elif entry != 1 or size != 1:
            return False
    return True


def _destination_storage_proven_disjoint(
    env: CompileEnvironment,
    tensor: torch.Tensor,
    input_tensors: Sequence[torch.Tensor],
    *,
    allow_unbound: bool,
) -> bool:
    """A registered positive storage-span disjointness fact per operand.

    The existing CuTe alias matrix compares whole storage spans of the
    runtime arguments and is part of the bound kernel's cache key, so a later
    launch with overlapping (or equal, or same-storage) views selects another
    binding. A False or missing fact refuses.
    """
    register_tensor_alias_specializations(env)
    return all(
        source is not tensor
        and runtime_tensors_are_proven_disjoint(
            env, tensor, source, allow_unbound=allow_unbound
        )
        for source in input_tensors
    )


def _program_effects_owned(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    store: torch.fx.Node,
    tensor: torch.Tensor,
    input_tensors: Sequence[torch.Tensor],
) -> bool:
    """The whole program's memory effects are this store and operand loads.

    The first version owns exactly one output: the program's only store (or
    atomic) is ``store``, and every other access to a host tensor is a pure
    ``load`` of one of the matmul operands. Any other read or write (a second
    output, a read of another input, an in-place update) refuses, so no
    pairing of the destination with another output, input or extra argument
    needs an alias proof beyond the operand facts. Argument names and tensor
    objects are never used as a non-aliasing proof here.
    """
    host_tensors = tuple(env.input_sources)
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is memory_ops.store or node.target in ATOMIC_OPS:
                if node is not store:
                    return False
                continue
            value = node.meta.get("val")
            if node.target is not _tracing_ops._host_tensor and not (
                isinstance(value, torch.Tensor)
                and any(value is host for host in host_tensors)
            ):
                continue
            if not isinstance(value, torch.Tensor):
                return False
            is_destination = value is tensor
            is_operand = any(value is source for source in input_tensors)
            for user in node.users:
                if is_destination and user is store and store.args[0] is node:
                    continue
                if (
                    is_operand
                    and user.op == "call_function"
                    and user.target is memory_ops.load
                    and user.args[0] is node
                    and not any(arg is node for arg in user.args[1:])
                ):
                    continue
                return False
    return True
