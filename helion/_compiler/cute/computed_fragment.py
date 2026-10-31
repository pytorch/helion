"""CTA-local SIMT lowering for multidimensional static fragments.

The scalar lowering gives each block ID one coordinate. Static dimensions,
including repeated dimensions of a matrix, need independent coordinates. This
path owns a complete root: its pure expressions retain logical index maps and
its loads, contractions, and loop carries are materialized in shared memory.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from contextlib import nullcontext
import dataclasses
from dataclasses import dataclass
import math
import operator
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.fx.node import map_arg

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import matmul_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import HelperFunctionGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..inductor_lowering import GenerateASTFromInductor
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..inductor_lowering import SympyExprLowering
from ..inductor_lowering import install_inductor_kernel_handlers
from ..matmul_utils import _compute_out_dtype
from ..matmul_utils import _needs_f32_accumulator
from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
from ..variable_origin import GridOrigin
from ..variable_origin import HostOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileEndOrigin
from ..variable_origin import TileIdOrigin
from .atomic_consumer_fusion import AtomicConsumerRegion
from .atomic_consumer_fusion import atomic_consumer_regions
from .captured_reduction import captured_reduction_coordinates
from .captured_reduction import physical_capture_axes
from .completed_scan import CompletedScan
from .completed_scan import completed_scan_plans
from .dead_zero_atomics import dead_zero_atomic_results
from .direct_affine_plan import DIRECT_AFFINE_ORDINARY_SCHEDULE
from .fragment_expression import FragmentExpression
from .fragment_indexing import memory_index_coordinates
from .fragment_storage import aligned_shared_bytes
from .fragment_storage import configured_fragment_expr
from .fragment_storage import metadata_guarded
from .free_iota_reduction import free_iota_reductions
from .free_iota_reduction import owned_iota_reduction_axes
from .independent_reduction import independent_reduction_coordinates
from .integer_atomic_epochs import can_defer_integer_epoch
from .local_atomic import local_atomic_allocations
from .local_atomic import local_indexed_load
from .local_atomic import prove_local_atomics
from .local_atomic import terminal_finalizer_inputs
from .local_atomic import terminal_loop_symbols
from .local_atomic import uniform_local_branch_inputs
from .local_atomic_registers import MAX_SLOTS
from .local_atomic_registers import local_atomic_register_chains
from .ordered_selection import is_ordered_selection
from .ordered_selection import ordered_selection_args
from .ordered_selection import selection_values_only
from .private_scalar_loops import private_scalar_loop_nodes
from .private_scalar_loops import privatize_scalar_loop
from .pure_producer_regions import pure_producer_plan
from .register_loads import host_load_is_readonly
from .register_loads import lane_private_load
from .register_snapshots import SnapshotBinding
from .register_snapshots import SnapshotOwner
from .register_snapshots import frame_snapshot_recipes
from .register_snapshots import snapshot_capture_slots
from .register_snapshots import snapshot_chains
from .register_snapshots import snapshot_logical_shape
from .scan_helper import additive_scan_helper
from .tcgen05_config import CuteTcgen05Config
from .warp_results import warp_result_chain

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Iterable
    from collections.abc import Mapping

    from ..autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .fragment_slab_loads import ScalarLoad
    from .fragment_slab_loads import SlabGroup
    from .warp_producer_regions import WarpProducerRecorder


# The most slots per thread an element loop walks as a constexpr unroll.
UNROLLED_SLOTS = 16

# ``mma.sync`` helpers of the tf32 fragment contraction (``cute_fp32_dot``).
_FRAGMENT_MMA_RUNTIME = "_helion_fragment_mma"
_FRAGMENT_MMA_PREAMBLE = (
    f"import helion._compiler.cute._mma_dag_runtime as {_FRAGMENT_MMA_RUNTIME}\n"
)


def _mma_operand_conflict_free(layout: tuple[int, int], role: str) -> bool:
    """Whether one ``mma.sync.m16n8k8`` operand word read from a staged tile
    of *layout* ``(row bytes, element bytes)`` is bank-conflict-free (lanes
    that share a bank share the word).  Lane ``(g, c) = (lane // 4, lane %
    4)`` reads the A word at ``(g, c)`` (the other words shift rows or k by
    a multiple of 4) and the B word at ``(c, g)``."""
    row_bytes, itemsize = layout
    if row_bytes % 4:
        return False
    banks: dict[int, set[int]] = {}
    for lane in range(32):
        g, c = lane // 4, lane % 4
        row, col = (g, c) if role == "a" else (c, g)
        word = (row * row_bytes + col * itemsize) // 4
        banks.setdefault(word % 32, set()).add(word)
    return all(len(words) == 1 for words in banks.values())


def _mma_row_pad(extent: int, unit: int) -> int:
    """Elements appended to a staged mma operand row so the row stride is an
    odd multiple of *unit* words modulo the 32 shared-memory banks.

    A lane fragment of ``mma.sync.m16n8k8`` reads 8 rows x 4 consecutive
    columns (A, ``unit=4``) or 4 rows x 8 consecutive columns (B, ``unit=8``):
    with such a stride the 32 addresses of one instruction fall in 32 distinct
    banks, where a stride that is a multiple of 32 serializes them 8-fold.
    """
    return min((residue - extent) % 32 for residue in range(unit, 32, 2 * unit))


def _resolve(value: object, values: dict[Node, object]) -> object:
    if isinstance(value, Node):
        return values[value]
    if isinstance(value, list):
        return [_resolve(item, values) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve(item, values) for item in value)
    return value


def _loop_carry_slots(node: Node) -> dict[int, int]:
    captured = cast("list[Node]", node.args[3])
    slots: dict[int, int] = {}
    for getitem in node.users:
        assert getitem.target is operator.getitem
        output_index = cast("int", getitem.args[1])
        for user in getitem.users:
            if user.target is _tracing_ops._phi and user.args[1] is getitem:
                slots[output_index] = captured.index(cast("Node", user.args[0]))
    return slots


@dataclass(frozen=True)
class Fragment:
    shape: tuple[int, ...]
    dtype: torch.dtype
    element: Callable[[tuple[str, ...]], str]
    resident: bool = False
    dependencies: tuple[Fragment, ...] = ()
    storage: str | None = None
    logical_domain: Callable[[tuple[str, ...]], tuple[str, ...]] | None = None

    snapshot: SnapshotBinding | None = None
    # A leaf whose recipe reads no memory at all (``hl.arange``, a tile index,
    # a constant): any lane may evaluate it at any coordinate for free.
    pure: bool = False
    # ``(start, step)`` of a one-dimensional ``hl.arange`` window with static
    # integer bounds: the index form the row-packet proof understands.
    affine: tuple[int, int] | None = None
    # A dtype cast or an order-preserving view of one staged tile: reading
    # it costs the same shared-memory access as reading the tile itself.
    cheap: bool = False
    # ``(row bytes, element bytes)`` of the staged tile this recipe reads,
    # when it is one tile read in its own element order.
    physical_row: tuple[int, int] | None = None

    def read(self, indices: tuple[str, ...]) -> str:
        assert len(indices) == len(self.shape)
        return self.element(indices)

    def broadcast(self, indices: tuple[str, ...]) -> str:
        indices = indices[len(indices) - len(self.shape) :] if self.shape else ()
        return self.read(
            tuple(
                "0" if size == 1 else index
                for size, index in zip(self.shape, indices, strict=True)
            )
        )

    def domain(self, indices: tuple[str, ...]) -> tuple[str, ...]:
        """Coordinate ownership, independent of a value or memory-load mask."""
        return tuple(
            dict.fromkeys(
                (
                    *(
                        f"0 <= ({index}) and ({index}) < ({size})"
                        for index, size in zip(indices, self.shape, strict=True)
                    ),
                    *(self.logical_domain(indices) if self.logical_domain else ()),
                )
            )
        )

    def broadcast_domain(self, indices: tuple[str, ...]) -> tuple[str, ...]:
        indices = indices[len(indices) - len(self.shape) :] if self.shape else ()
        return self.domain(
            tuple(
                "0" if size == 1 else index
                for size, index in zip(self.shape, indices, strict=True)
            )
        )


@dataclass(frozen=True)
class HostTensor:
    name: str
    value: torch.Tensor


@dataclass
class DirectStoreSink:
    """The host store that consumes a tf32 mma result straight from the lane
    registers (``emit_tf32_mma``): ``memory`` fills in the window's address
    recipe without emitting the staged walk, the lane loop writes through it,
    and the store node itself then lowers to nothing."""

    node: Node
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    address: Callable[[tuple[str, ...]], tuple[str, str]] | None = None
    # ``cute_fragment_vector_io``'s packet width when the window is proved to
    # walk whole aligned 16-byte packets: the two adjacent columns of one
    # lane then leave as one aligned vector store.
    lanes: int | None = None


def _consumed_by_direct_store(_coords: tuple[str, ...]) -> str:
    raise AssertionError("a dot result stored by its lanes has no other reader")


class FragmentOps(GenerateASTFromInductor):
    """Use Inductor's index expressions instead of dropping them as SIMT does."""

    def __init__(
        self, compiler: FragmentCompiler, inputs: dict[str, tuple[Node, Fragment]]
    ) -> None:
        super().__init__(compiler.cg, {})
        self.compiler = compiler
        self.inputs = inputs

    def load(self, name: str, index: sympy.Expr) -> str:
        node, value = self.inputs[name]
        fake = node.meta["val"]
        if not isinstance(fake, torch.Tensor):
            return self._lift(expr_from_string(value.read(())))
        assert isinstance(fake, torch.Tensor)
        # Inductor indexes the fake input's physical layout, including its
        # transposes and broadcasts. Recover coordinates in that same layout.
        strides = self.compiler.shape(fake.stride())
        indices = self.direct_coordinates(index, strides, value.shape)
        if indices is None:
            indices = ["0"] * len(value.shape)
            remaining = self.compiler.sym(index)
            for dim in sorted(
                range(len(strides)), key=lambda dim: strides[dim], reverse=True
            ):
                if value.shape[dim] == 1 or strides[dim] == 0:
                    continue
                indices[dim] = f"(({remaining}) // {strides[dim]})"
                remaining = f"(({remaining}) % {strides[dim]})"
        coordinates = self.compiler.coordinate_locals(tuple(indices))
        return self._lift(expr_from_string(value.read(coordinates)))

    def direct_coordinates(
        self, index: sympy.Expr, strides: tuple[int, ...], shape: tuple[int, ...]
    ) -> list[str] | None:
        """The input coordinates when *index* is exactly the layout's affine
        form ``sum(stride[d] * i_d)`` over the current coordinate symbols:
        each symbol names one dimension's coordinate outright, so no
        ``// stride`` / ``% stride`` round trip (which the compiler cannot
        fold without the symbols' ranges) is emitted.  None when a term is
        not a plain coordinate symbol, a constant offset remains, or two
        live dimensions share a stride."""
        symbols = self.compiler.sym_indices
        if not symbols:
            return None
        indices = ["0"] * len(shape)
        used: set[int] = set()
        # A singleton dimension's coordinate is the literal zero: its stride
        # term (which Inductor keeps when the stride is not zero) vanishes.
        index = index.xreplace(
            {
                symbol: sympy.Integer(0)
                for symbol, coord in symbols.items()
                if coord == "0"
            }
        )
        # Stride factors may still be size symbols (``Max(1, u2)``): resolve
        # them the way the staged shape resolved the strides themselves.
        resolved: dict[sympy.Basic, sympy.Integer] = {}
        for symbol in index.free_symbols:
            if symbol in symbols:
                continue
            extent = self.compiler.static_extent(symbol)
            if extent is None:
                return None
            resolved[symbol] = sympy.Integer(extent)
        index = sympy.expand(index.xreplace(resolved))
        for term, coefficient in index.as_coefficients_dict().items():
            if term == 1:
                if coefficient != 0:
                    return None
                continue
            if not isinstance(term, sympy.Symbol) or term not in symbols:
                return None
            if not coefficient.is_Integer:
                return None
            candidates = [
                dim
                for dim in range(len(shape))
                if dim not in used
                and shape[dim] != 1
                and strides[dim] != 0
                and strides[dim] == int(coefficient)
            ]
            if len(candidates) != 1:
                return None
            indices[candidates[0]] = symbols[term]
            used.add(candidates[0])
        return indices

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        # Fragment loads and shared buffers contain logical typed scalars.
        # The parent handler's FP8 byte decoder belongs to ordinary raw-byte
        # global loads; applying it to a typed Float8 register corrupts its bits.
        return self._lift(self._create_cast_expr(x, dtype))

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        return self._lift(
            self._cast_scalar_ast(expr_from_string(self.compiler.sym(expr)), dtype)
        )


def _fragment_logical_shape(
    env: CompileEnvironment,
    node: Node,
    *,
    scalar_indexed_loads: bool = False,
    tensor_indexed_loads: bool = False,
    pure_inline_asm: bool = False,
    all_axis_reductions: bool = False,
    proven_domains: dict[Node, tuple[sympy.Expr, ...]] | None = None,
) -> tuple[sympy.Expr, ...] | None:
    """Prove fragment coordinates from producers, never padded fake capacity.

    Full slices and factories retain their declared extents. Pointwise operations
    broadcast those extents; layout changes must prove the same logical mapping.
    Unknown captures and padding-changing reshapes decline. Indexed-load
    domains are proved only for the opt-in bounded gather path.
    """
    memo: dict[Node, tuple[sympy.Expr, ...] | None] = {}

    def dimension(value: object) -> sympy.Expr:
        if isinstance(value, Node):
            value = value.meta["val"]
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        return env.specialize_expr(sympy.sympify(value))

    def fake_shape(value: torch.Tensor) -> tuple[sympy.Expr, ...]:
        return tuple(dimension(size) for size in value.shape)

    def equal(a: sympy.Expr, b: sympy.Expr) -> bool:
        return sympy.expand(a) == sympy.expand(b)

    def infer(source: Node) -> tuple[sympy.Expr, ...] | None:
        if proven_domains is not None and source in proven_domains:
            return proven_domains[source]
        if source in memo:
            return memo[source]
        memo[source] = None
        result = calculate(source)
        memo[source] = result
        return result

    def calculate(source: Node) -> tuple[sympy.Expr, ...] | None:
        value = source.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        target = source.target
        if pure_inline_asm and target is inline_asm_ops.inline_asm_elementwise:
            from .bounded_gather import inline_asm_shape

            return inline_asm_shape(source, infer, dimension)
        if all_axis_reductions:
            from .bounded_gather import all_axis_reduction_shape

            reduced = all_axis_reduction_shape(source, infer, dimension)
            if reduced is not None:
                return reduced
        if target is memory_ops.load:
            tensor_node = cast("Node", source.args[0])
            tensor = tensor_node.meta.get("val")
            if not isinstance(tensor, torch.Tensor):
                return None
            tensor_shape = (
                tuple(dimension(size) for size in tensor.shape)
                if tensor_node.target is _tracing_ops._host_tensor
                else infer(tensor_node)
            )
            if tensor_shape is None:
                return None
            index_shapes = {}
            if tensor_indexed_loads:
                from .bounded_gather import indexed_load_shapes

                index_shapes = indexed_load_shapes(
                    env, cast("list[object]", source.args[1]), infer, dimension
                )
                if index_shapes is None:
                    return None
            sizes: list[sympy.Expr] = []
            axis = 0
            for ordinal, index in enumerate(cast("list[object]", source.args[1])):
                if index is None:
                    sizes.append(sympy.Integer(1))
                    continue
                if index == slice(None):
                    sizes.append(tensor_shape[axis])
                elif isinstance(index, Node):
                    proxy = index.meta.get("val")
                    if isinstance(proxy, torch.Tensor):
                        if ordinal not in index_shapes:
                            return None
                        sizes.extend(index_shapes[ordinal])
                    elif isinstance(proxy, torch.SymInt):
                        bid = env.resolve_block_id(proxy)
                        if (
                            bid is None
                            or proxy._sympy_() != env.block_sizes[bid].var._sympy_()
                        ):
                            if not scalar_indexed_loads:
                                return None
                            # A scalar grid/load index removes this source axis;
                            # it does not determine any gathered-axis capacity.
                        else:
                            sizes.append(dimension(proxy))
                elif not isinstance(index, int):
                    return None
                axis += 1
            sizes.extend(tensor_shape[axis:])
            return tuple(sizes)
        if target is torch.ops.prims.iota.default:
            return (dimension(source.args[0]),)
        if target in (creation_ops.full, torch.ops.aten.full.default):
            return tuple(
                dimension(size) for size in cast("list[object]", source.args[0])
            )
        if target is torch.ops.aten.scalar_tensor.default:
            return ()
        if target is operator.getitem:
            parent = source.args[0]
            if not isinstance(parent, Node) or not is_ordered_selection(parent):
                return None
            arguments = ordered_selection_args(parent)
            if arguments is None:
                return None
            source_node, k, *_ = arguments
            shape = infer(source_node)
            if not shape:
                return None
            return (*shape[:-1], dimension(k)) if k is not None else shape
        if target is torch.ops.aten.gather.default:
            return infer(cast("Node", source.args[2]))
        tensors = [
            n
            for n in source.all_input_nodes
            if isinstance(n.meta.get("val"), torch.Tensor)
        ]
        if not tensors:
            return None
        shapes = [infer(n) for n in tensors]
        if any(shape is None for shape in shapes):
            return None
        inputs = cast("list[tuple[sympy.Expr, ...]]", shapes)
        first = inputs[0]
        if target in (
            _tracing_ops._mask_to,
            _tracing_ops._new_var,
            scan_ops._associative_scan,
            torch.ops.aten.alias.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.view.dtype,
        ):
            return first
        if target is view_ops.subscript:
            subscript_shape: list[sympy.Expr] = []
            axis = 0
            for index in cast("list[object]", source.args[1]):
                if index is None:
                    subscript_shape.append(sympy.Integer(1))
                    continue
                if index == slice(None):
                    subscript_shape.append(first[axis])
                elif not isinstance(index, int):
                    return None
                axis += 1
            return (*subscript_shape, *first[axis:])
        if target is torch.ops.aten.permute.default:
            return tuple(first[axis] for axis in cast("list[int]", source.args[1]))
        if target is torch.ops.aten.unsqueeze.default:
            axis = cast("int", source.args[1]) % (len(first) + 1)
            return (*first[:axis], sympy.Integer(1), *first[axis:])
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            # The emitter flattens physical coordinates. A reshape of a padded
            # axis cannot be justified just by matching logical element counts.
            physical = fake_shape(cast("torch.Tensor", tensors[0].meta["val"]))
            if any(not equal(a, b) for a, b in zip(first, physical, strict=True)):
                return None
            result = fake_shape(value)
            return result if equal(sympy.prod(first), sympy.prod(result)) else None
        if target is torch.ops.aten.expand.default:
            result = tuple(
                dimension(size) for size in cast("list[object]", source.args[1])
            )
            padded = (sympy.Integer(1),) * (len(result) - len(first)) + first
            if any(
                a != 1 and not equal(a, b) for a, b in zip(padded, result, strict=True)
            ):
                return None
            return result
        if isinstance(source.meta.get("lowering"), ReductionLowering):
            # Dimension-list reductions remove axes without reinterpreting the
            # remaining coordinates. Other reduction signatures are unproved.
            if (
                len(source.args) < 2
                or not isinstance(source.args[1], (list, tuple))
                or not source.args[1]
            ):
                return None
            axes = {axis % len(first) for axis in cast("list[int]", source.args[1])}
            physical = fake_shape(cast("torch.Tensor", tensors[0].meta["val"]))
            if any(not equal(first[axis], physical[axis]) for axis in axes):
                return None
            keepdim = (
                source.args[2]
                if len(source.args) > 2
                else source.kwargs.get("keepdim", False)
            )
            return tuple(
                sympy.Integer(1) if axis in axes else size
                for axis, size in enumerate(first)
                if keepdim or axis not in axes
            )
        if (
            isinstance(source.meta.get("lowering"), PointwiseLowering)
            or target is torch.ops.aten.where.self
        ):
            rank = max(map(len, inputs))
            broadcast_shape: list[sympy.Expr] = [sympy.Integer(1)] * rank
            for shape in inputs:
                for axis, size in enumerate(
                    (sympy.Integer(1),) * (rank - len(shape)) + shape
                ):
                    if size == 1:
                        continue
                    if broadcast_shape[axis] != 1 and not equal(
                        broadcast_shape[axis], size
                    ):
                        return None
                    broadcast_shape[axis] = size
            return tuple(broadcast_shape)
        return None

    return infer(node)


def _static_selection_extent(env: CompileEnvironment, source: Node) -> int | None:
    shape = _fragment_logical_shape(env, source)
    if not shape:
        return None
    width = shape[-1]
    # Dynamic or tiled selected extents remain unsupported by this owner.
    return int(width) if width.is_Integer and int(width) > 0 else None


class FragmentCompiler:
    # Tile staging knobs (``cute_fragment_padded_rows``,
    # ``cute_fragment_unrolled_loops``, ``cute_fragment_vector_io``,
    # ``cute_fragment_batched_loads``, ``cute_fragment_split_k``,
    # ``cute_fragment_slab_loads``); the config sets them in ``__init__``.
    padded_rows = False
    unrolled_loops = False
    vector_io = False
    batched_loads = False
    split_k = False
    mma_rows = False
    pending_barrier = False
    slab_plan: Mapping[Node, SlabGroup] = MappingProxyType({})

    def __init__(self, cg: GenerateAST, graphs: list[GraphInfo] | None = None) -> None:
        self.cg = cg
        self.df = cg.device_function
        self.env = CompileEnvironment.current()
        self.offsets: dict[int, str] = {}
        self.bounds: dict[int, str] = {}
        self.allocator = ""
        self.smem_bytes = 0
        self.buffers: list[tuple[str, torch.dtype, int]] = []
        self.scopes: list[dict[Node, object]] = []
        self.held: list[object] = []
        self.threads = cast("int", self.df.config.get("cute_fragment_threads", 128))
        self.thread = ""
        self.padded_rows = bool(self.df.config.get("cute_fragment_padded_rows", False))
        self.unrolled_loops = bool(
            self.df.config.get("cute_fragment_unrolled_loops", False)
        )
        # Tile transport knobs: 16-byte row packets for proved host windows
        # (``cute_fragment_vector_io``) and one barrier per run of
        # independent host loads (``cute_fragment_batched_loads``).
        self.vector_io = bool(self.df.config.get("cute_fragment_vector_io", False))
        self.batched_loads = bool(
            self.df.config.get("cute_fragment_batched_loads", False)
        )
        # ``cute_fragment_split_k``: idle warps of a tf32 mma phase take a
        # share of K (partials summed at the read).
        self.split_k = bool(self.df.config.get("cute_fragment_split_k", False))
        # ``cute_fragment_mma_rows``: the tf32 mma lanes walk whole tile rows
        # and evaluate each row's A words once for all of its column tiles.
        self.mma_rows = bool(self.df.config.get("cute_fragment_mma_rows", False))
        # A load phase emitted without its trailing barrier: the next node
        # that is not another independent host load issues it.
        self.pending_barrier = False
        # A tf32 ``mma.sync`` phase was emitted: the generated module needs
        # the helpers' import, added once the root is committed (a capacity
        # decline of the scratch body must leave the module untouched).
        self.wants_mma_preamble = False
        # Host stores already written by the tf32 mma lanes of the dot they
        # consume (``DirectStoreSink``): the store node lowers to nothing.
        self.direct_stores: set[Node] = set()
        # The graph walk's active atomic-consumer region and producer-cache
        # nodes: a dot inside either keeps its staged result.
        self.active_region: AtomicConsumerRegion | None = None
        self.cache_nodes: frozenset[Node] = frozenset()
        self.sym_indices: dict[sympy.Symbol, str] = {}
        self.expression: FragmentExpression | None = None
        self.lazy_opaque_producers: frozenset[Node] = frozenset()
        self.local_allocations: frozenset[Node] = frozenset()
        self.local_logical_sizes: dict[int, int] = {}
        self.local_storage: list[Fragment] = []
        self.pending_local_atomics: set[str] = set()
        self.scalar_ordered_tickets: dict[Node, Fragment] = {}
        self.graphs = cg.codegen_graphs if graphs is None else graphs
        # ``cute_fragment_slab_loads``: repeated static windows of one tensor
        # read views of a slab staged once at the start of the root.
        self.slab_plan = {}
        # The uniform scalar loads (a varlen slab's row offset and limit)
        # read once per CTA ahead of the slabs: load -> the DSL variable.
        self.slab_scalars: dict[ScalarLoad, str] = {}
        if self.df.config.get("cute_fragment_slab_loads", False):
            from .fragment_slab_loads import plan_slab_loads

            self.slab_plan = plan_slab_loads(self.env, self.graphs)
        self.dead_zero_results = (
            dead_zero_atomic_results(self.graphs, self.env)
            if self.df.config.get("cute_fragment_skip_zero_atomics", False)
            else frozenset()
        )
        self.private_scalar_loops = (
            private_scalar_loop_nodes(self.graphs)
            if self.df.config.get("cute_fragment_private_scalar_loops", False)
            else frozenset()
        )
        self.warp_result_nodes: dict[Node, int] = {}
        if self.df.config.get("cute_fragment_warp_results", False):
            for graph in self.graphs:
                for node in graph.graph.nodes:
                    self.warp_result_nodes.update(
                        warp_result_chain(node, self.env, self.threads)
                    )

        self.warp_producer_recorder: WarpProducerRecorder | None = None
        self.scan_plans: dict[Node, CompletedScan] = {}
        self.scan_reductions: dict[Node, CompletedScan] = {}
        self.scan_pending: list[tuple[Fragment, Callable, torch.dtype, str]] = []
        self.snapshot_owner: SnapshotOwner | None = None
        self.snapshot_loads: set[Node] = set()
        self.register_snapshot_buffers: dict[str, int] = {}
        self.snapshot_reductions: set[Node] = set()
        self.snapshot_shapes: set[tuple[int, ...]] = set()
        if self.df.config.get("cute_fragment_register_snapshots", False):
            for node, chain in snapshot_chains(self.graphs, self.env).items():
                shape = self.shape(node.meta["val"].shape)
                if len(shape) == 1 and 0 < shape[0] <= self.threads * MAX_SLOTS:
                    if self.warp_result_nodes.keys() & chain:
                        continue
                    self.snapshot_loads.add(node)
                    self.snapshot_shapes.add(shape)
                    self.snapshot_shapes.add(self.shape(snapshot_logical_shape(node)))
                    self.snapshot_reductions.update(
                        item
                        for item in chain
                        if isinstance(item.meta.get("lowering"), ReductionLowering)
                    )

            if not self.snapshot_loads and any(
                graph.graph_id
                in self.env.config_spec.cute_fragment_register_snapshots_root_ids
                for graph in self.graphs
            ):
                raise exc.InvalidConfig(
                    "register snapshot root has no eligible immutable owner slots"
                )

        self.local_register_nodes: set[Node] = set()
        self.local_register_slot: str | None = None
        self.iteration_local_returns: frozenset[Node] = frozenset()
        self.iteration_owner: SnapshotOwner | None = None
        self.iteration_defined: set[Node] = set()
        if self.df.config.get("cute_fragment_local_atomic_registers", False):
            for node, chain in local_atomic_register_chains(
                self.graphs, self.env
            ).items():
                if self.warp_result_nodes.keys() & chain:
                    continue
                fake = cast("torch.Tensor", node.meta["val"])
                count = math.prod(self.shape(fake.shape))
                if 0 < count <= self.threads * MAX_SLOTS:
                    self.local_register_nodes.update(chain)

    @contextmanager
    def snapshot_scope(
        self, index: str, slot: str, size: int
    ) -> Generator[None, None, None]:
        previous = self.snapshot_owner
        if (size,) in self.snapshot_shapes:
            self.snapshot_owner = SnapshotOwner(index, slot, size)
        try:
            yield
        finally:
            self.snapshot_owner = previous

    def begin(self) -> None:
        self.allocator = self.df.new_var("fragment_smem")
        self.thread = self.df.new_var("fragment_thread")

    @contextmanager
    def iteration_scope(self, index: str, size: int) -> Generator[None, None, None]:
        """A returned scalar is valid only in this active owner's iteration."""
        assert self.iteration_owner is None and not self.iteration_defined
        self.iteration_owner = SnapshotOwner(index, "", size)
        try:
            yield
        finally:
            self.iteration_owner = None
            self.iteration_defined.clear()

    def allocations(self) -> list[ast.AST]:
        return [
            statement_from_string(f"{self.allocator} = cutlass.utils.SmemAllocator()"),
            *[
                statement_from_string(
                    f"{name} = {self.allocator}.allocate_tensor({self.dtype(dtype)}, cute.make_layout(({count},)), byte_alignment=16)"
                )
                for name, dtype, count in self.buffers
            ],
        ]

    def metadata_guarded(self, expr: sympy.Expr) -> bool:
        return metadata_guarded(self.env, expr)

    def configured_expr(self, value: sympy.Basic) -> sympy.Basic:
        return configured_fragment_expr(self.env, value, self.df.resolved_block_size)

    def extent(self, value: object) -> int:
        resolved = self.static_extent(value)
        if resolved is None:
            raise exc.InvalidConfig(
                f"computed fragments require static local extents: {value}"
            )
        return resolved

    def static_extent(self, value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        assert isinstance(value, sympy.Expr)
        value = self.configured_expr(value)
        value = self.env.specialize_expr(sympy.sympify(value))
        return int(value) if value.is_number else None

    def shape(self, shape: object) -> tuple[int, ...]:
        return tuple(self.extent(x) for x in cast("tuple[object, ...]", shape))

    def sym(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value._sympy_()
        if isinstance(value, sympy.Basic):
            value = sympy.sympify(self.configured_expr(value))

            def symbol_expression(symbol: sympy.Symbol) -> str:
                if symbol in self.sym_indices:
                    return self.sym_indices[symbol]
                origin_info = HostFunction.current().expr_to_origin.get(symbol)
                origin = origin_info.origin if origin_info is not None else None
                if isinstance(origin, GridOrigin) and origin.block_id in self.offsets:
                    offset = self.offsets[origin.block_id]
                    size = self.df.resolved_block_size(origin.block_id)
                    if isinstance(origin, TileIdOrigin):
                        return f"({offset}) // {size}"
                    if isinstance(origin, TileEndOrigin):
                        return (
                            f"min(({offset}) + {size}, {self.bounds[origin.block_id]})"
                        )
                    if type(origin) in (GridOrigin, TileBeginOrigin):
                        return offset
                return self.df.literal_expr(symbol)

            substitutions = {
                symbol: sympy.Symbol(f"({symbol_expression(symbol)})")
                for symbol in value.free_symbols
            }
            return self.env.backend.sympy_printer_expr(value.xreplace(substitutions))
        return self.df.literal_expr(value)

    def scalar(self, value: object) -> str:
        if isinstance(value, Fragment):
            assert math.prod(value.shape) == 1
            return value.read(tuple("0" for _ in value.shape))
        return self.sym(value)

    def host_stride(self, tensor: HostTensor, dim: int) -> str:
        # Fresh wrapper allocations have configured symbolic strides, which
        # need the fragment block-size resolver. Other host layouts must use
        # the ordinary metadata argument/specialization proof: fake stride
        # factors alone need not have a recoverable host symbol origin.
        if self.env.tensor_layout_is_symbolically_exact(tensor.value) and all(
            tensor.value.untyped_storage() != source.untyped_storage()
            for source in self.env.input_sources
        ):
            return self.sym(tensor.value.stride(dim))
        return self.df.tensor_stride(tensor.value, dim).name

    def emit(self, source: str) -> None:
        self.cg.add_statement(statement_from_string(source))

    def synchronize(self) -> None:
        self.emit("cute.arch.sync_threads()")
        self.pending_local_atomics.clear()
        self.pending_barrier = False

    def flush_pending_barrier(self) -> None:
        """Close a run of batched host loads before anything else reads or
        recycles shared memory."""
        if self.pending_barrier:
            self.synchronize()

    # Targets whose fragment lowering returns a lazy recipe or a scalar and
    # emits no statement: a run of batched loads may continue across them.
    _TRANSPARENT_TARGETS = frozenset(
        {
            _tracing_ops._host_tensor,
            _tracing_ops._get_symnode,
            _tracing_ops._new_var,
            _tracing_ops._phi,
            operator.getitem,
            tile_ops.tile_begin,
            tile_ops.tile_end,
            tile_ops.tile_id,
            tile_ops.tile_index,
            view_ops.subscript,
            torch.ops.aten.sym_size.int,
            torch.ops.prims.iota.default,
            torch.ops.aten.alias.default,
            torch.ops.aten.view.dtype,
            torch.ops.aten.permute.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.aten.where.self,
            _tracing_ops._mask_to,
        }
    )

    # Targets whose fragment lowering touches no host memory beyond the
    # transparent ones: another dot, a scan, a local allocation.  Pointwise
    # and reduction lowerings join them in ``direct_store_in_order``; a host
    # load, store or atomic needs a storage proof, and everything else (a
    # nested loop or branch, inline assembly, a gather or selection) is
    # opaque to the direct lane store.
    _HOST_MEMORY_FREE_TARGETS = _TRANSPARENT_TARGETS | frozenset(
        {
            matmul_ops.dot,
            scan_ops._associative_scan,
            creation_ops.full,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
            _tracing_ops._constant_tensor,
        }
    )

    def transparent_node(self, node: Node) -> bool:
        """Whether lowering *node* emits nothing (a lazy pointwise recipe, a
        view, a coordinate function or a scalar), so a pending batched-load
        barrier may stay pending across it."""
        if node in self.warp_result_nodes or node in self.local_register_nodes:
            return False
        if node.target in self._TRANSPARENT_TARGETS:
            return True
        if node.target in (
            creation_ops.full,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
            _tracing_ops._constant_tensor,
        ):
            return node not in self.local_allocations
        lowering = node.meta.get("lowering")
        return (
            isinstance(lowering, PointwiseLowering)
            and not isinstance(lowering, ReductionLowering)
            and node.target is not matmul_ops.dot
            and node.target not in (memory_ops.load, memory_ops.store)
            and node not in self.lazy_opaque_producers
        )

    def batchable_host_load(self, node: Node, values: dict[Node, object]) -> bool:
        """Whether *node* is a host load the batched phase may absorb: its
        address and mask recipes read no shared memory (so no barrier is
        needed before it) and it publishes through an ordinary staged copy."""
        if (
            not self.batched_loads
            or node.target is not memory_ops.load
            or self.warp_producer_recorder is not None
            or node in self.snapshot_loads
            or not isinstance(values.get(cast("Node", node.args[0])), HostTensor)
        ):
            return False
        inputs = [
            values.get(index)
            for index in (*cast("list[object]", node.args[1]), node.args[2])
            if isinstance(index, Node)
        ]
        if any(
            isinstance(value, Fragment) and value.snapshot is not None
            for value in inputs
        ):
            return False
        return not self.referenced_buffers(inputs)

    def synchronize_local_atomics(self) -> None:
        if self.pending_local_atomics:
            self.synchronize()

    def predicate(self, masks: list[str] | tuple[str, ...]) -> str:
        """Bound SDK short-circuit AST expansion without changing mask order.

        The CuTe preprocessor repeats the accumulated left operand in each
        short-circuit check. A flat N-way conjunction therefore expands
        exponentially even when its DAG is small. Name successive prefixes in
        this exact lexical body; each binary conjunction keeps Python/DSL
        short-circuit behavior, and no value is cached across a later mutation.
        """
        source = " and ".join(masks) or "True"
        expression = expr_from_string(source)
        if not isinstance(expression, ast.BoolOp):
            return source
        operands: list[ast.expr] = []

        def flatten(value: ast.expr) -> None:
            if isinstance(value, ast.BoolOp) and type(value.op) is type(expression.op):
                for child in value.values:
                    flatten(child)
            else:
                operands.append(value)

        flatten(expression)
        if len(operands) <= 3:
            return source
        operation = "and" if isinstance(expression.op, ast.And) else "or"
        current = self.df.new_var("fragment_predicate")
        self.emit(f"{current} = {ast.unparse(operands[0])}")
        for operand in operands[1:]:
            previous = current
            current = self.df.new_var("fragment_predicate")
            self.emit(f"{current} = {previous} {operation} ({ast.unparse(operand)})")
        return current

    def dtype(self, dtype: torch.dtype) -> str:
        return self.env.backend.dtype_str(dtype)

    def cast(self, expression: str, dtype: torch.dtype) -> str:
        return f"{self.dtype(dtype)}({expression})"

    def coordinate_locals(self, coordinates: tuple[str, ...]) -> tuple[str, ...]:
        """Evaluate composed scalar coordinates in their current lexical body.

        Recursively expanding a reshape or physical-stride inverse duplicates
        its incoming coordinates in every dimension. Name those pure integer
        expressions before visiting the next fragment. Do not cast them or
        reuse locals across reads, loops, branches, or shared-memory writes.
        """
        if (
            self.expression is not None
            and self.expression.body is self.cg.statements_stack[-1]
        ):
            result = tuple(self.expression.coordinate(value) for value in coordinates)
            for owner in (self.snapshot_owner, self.iteration_owner):
                if owner is not None:
                    for name, value in zip(result, coordinates, strict=True):
                        if name != value:
                            owner.aliases.setdefault(name, value)
            return result
        result = []
        for coordinate in coordinates:
            expression = expr_from_string(coordinate)
            if isinstance(expression, (ast.Name, ast.Constant)):
                result.append(coordinate)
                continue
            name = self.cg.lift(expression, prefix="fragment_coordinate").id
            for owner in (self.snapshot_owner, self.iteration_owner):
                if owner is not None:
                    owner.aliases.setdefault(name, coordinate)
            result.append(name)
        return tuple(result)

    def pointwise_read(
        self,
        producer: object,
        coordinates: tuple[str, ...],
        evaluate: Callable[[], str],
    ) -> str:
        previous = self.expression
        if previous is None or previous.body is not self.cg.statements_stack[-1]:
            self.expression = FragmentExpression(self.cg)
        try:
            assert self.expression is not None
            return self.expression.read(producer, coordinates, evaluate)
        finally:
            self.expression = previous

    @staticmethod
    def referenced_buffers(values: Iterable[object]) -> set[str]:
        live: set[str] = set()
        seen: set[int] = set()

        def visit(value: object) -> None:
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Fragment):
                if value.storage is not None:
                    live.add(value.storage)
                for child in value.dependencies:
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        for value in values:
            visit(value)
        return live

    def live_buffers(self) -> set[str]:
        return self.referenced_buffers(
            [*[value for scope in self.scopes for value in scope.values()], *self.held]
        )

    def physical_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[int, ...]:
        """The staged layout of a *shape* tile: its rows, padded by four
        32-bit words when the dense row stride is a multiple of the 32 shared
        memory banks, so that a column walk (a transposed operand read, the
        lane fragments of an ``mma``) touches 32 distinct banks."""
        if not self.padded_rows or len(shape) < 2 or math.prod(shape[:-1]) <= 1:
            return shape
        row_bytes = shape[-1] * dtype.itemsize
        if row_bytes % 128 or row_bytes < 128:
            return shape
        return (*shape[:-1], shape[-1] + max(1, 16 // dtype.itemsize))

    def allocate(self, value: Fragment) -> Fragment:
        physical = self.physical_shape(value.shape, value.dtype)
        count = math.prod(physical)
        live = self.live_buffers()
        available = sorted(
            (capacity, name, index)
            for index, (name, dtype, capacity) in enumerate(self.buffers)
            if name not in live and dtype == value.dtype
        )
        selected = next((item for item in available if item[0] >= count), None)
        if selected is None and available:
            selected = available[-1]
        if selected is None:
            name = self.df.new_var("fragment_buffer")
            self.buffers.append((name, value.dtype, count))
            self.smem_bytes += aligned_shared_bytes(count, value.dtype)
        else:
            capacity, name, index = selected
            # A dead allocation can be recycled even when its last user was
            # an unused atomic. Finish that lifetime before ordinary writes.
            if name in self.pending_local_atomics:
                self.synchronize_local_atomics()
            if capacity >= count:
                count = capacity
            else:
                # All allocations are declared at the root. A dead buffer can
                # grow for its next lifetime without retaining a separate size
                # class in shared memory; earlier flat indices stay valid.
                self.buffers[index] = (name, value.dtype, count)
                self.smem_bytes += aligned_shared_bytes(
                    count, value.dtype
                ) - aligned_shared_bytes(capacity, value.dtype)
        return Fragment(
            value.shape,
            value.dtype,
            lambda indices: f"{name}[{self.flatten(indices, physical)}]",
            True,
            (),
            name,
            value.logical_domain,
            physical_row=(
                (physical[-1] * value.dtype.itemsize, value.dtype.itemsize)
                if len(physical) >= 2
                else None
            ),
        )

    @staticmethod
    def flatten(indices: tuple[str, ...], shape: tuple[int, ...]) -> str:
        return (
            " + ".join(
                f"({index}) * {math.prod(shape[i + 1 :])}"
                for i, index in enumerate(indices)
            )
            or "0"
        )

    @staticmethod
    def coordinates(index: str, shape: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(
            "0" if size == 1 else f"(({index}) // {math.prod(shape[i + 1 :])} % {size})"
            for i, size in enumerate(shape)
        )

    def elements(
        self,
        shape: tuple[int, ...],
        body: Callable[[tuple[str, ...]], None],
        *,
        threads_per_element: int = 1,
        warp_owned: bool = False,
        synchronize: bool = True,
        register_owned: bool = False,
        iteration_local: bool = False,
        batched: bool = False,
    ) -> None:
        assert not iteration_local or (len(shape) == 1 and not warp_owned)
        # Any phase but a batched host load may read or recycle shared
        # memory: publish the pending batched loads first.
        if not batched:
            self.flush_pending_barrier()
        register_owned = register_owned or (
            shape in self.snapshot_shapes
            and not warp_owned
            and threads_per_element == 1
        )
        if register_owned:
            assert not warp_owned and threads_per_element == 1
            slots = (math.prod(shape) + self.threads - 1) // self.threads
            assert 0 < slots <= MAX_SLOTS
            slot = self.df.new_var("fragment_register_slot")
            index = self.df.new_var("fragment_register_index")
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                ),
            )
            loop.body.clear()
            prior = self.local_register_slot
            self.local_register_slot = slot
            try:
                with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                    self.emit(f"{index} = {self.thread} + {slot} * {self.threads}")
                    branch = cast(
                        "ast.If",
                        statement_from_string(
                            f"if {index} < {math.prod(shape)}:\n    pass"
                        ),
                    )
                    branch.body.clear()
                    with (
                        self.cg.set_statements(cast("list[ast.AST]", branch.body)),
                        self.snapshot_scope(index, slot, math.prod(shape)),
                        self.iteration_scope(index, math.prod(shape))
                        if iteration_local
                        else nullcontext(),
                    ):
                        body(self.coordinates(index, shape))
                    self.cg.add_statement(branch)
            finally:
                self.local_register_slot = prior
            self.cg.add_statement(loop)
            if synchronize:
                self.synchronize()
            return
        if warp_owned:
            assert threads_per_element == 1
            threads_per_element = 32
        assert threads_per_element in (1, 32)
        assert self.threads % threads_per_element == 0
        count = math.prod(shape)
        slots = -(-count // self.threads)
        if (
            self.unrolled_loops
            and threads_per_element == 1
            and not iteration_local
            and self.warp_producer_recorder is None
            and 0 < slots <= UNROLLED_SLOTS
        ):
            # A constexpr walk over the thread's slots: every global load of
            # the phase is issued before the first shared store consumes one,
            # where the rolled loop serialized one round trip per slot.
            slot = self.df.new_var("fragment_slot")
            index = self.df.new_var("fragment_index")
            loop = statement_from_string(
                f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{index} = {self.thread} + {slot} * {self.threads}")
                if count % self.threads:
                    branch = statement_from_string(f"if {index} < {count}:\n    pass")
                    assert isinstance(branch, ast.If)
                    branch.body.clear()
                    with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                        body(self.coordinates(index, shape))
                    self.cg.add_statement(branch)
                else:
                    body(self.coordinates(index, shape))
            self.cg.add_statement(loop)
            if synchronize:
                self.synchronize()
            return
        index = self.df.new_var("fragment_index")
        start = (
            self.thread
            if threads_per_element == 1
            else f"{self.thread} // {threads_per_element}"
        )
        loop = statement_from_string(
            f"for {index} in range({start}, {math.prod(shape)}, {self.threads // threads_per_element}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            if warp_owned:
                branch = cast(
                    "ast.If",
                    statement_from_string(f"if {self.thread} % 32 == 0:\n    pass"),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    body(self.coordinates(index, shape))
                self.cg.add_statement(branch)
            else:
                with (
                    self.iteration_scope(index, math.prod(shape))
                    if iteration_local
                    else nullcontext()
                ):
                    body(self.coordinates(index, shape))
        self.cg.add_statement(loop)
        if synchronize:
            self.synchronize()

    def copy(
        self,
        source: Fragment,
        target: Fragment,
        *,
        threads_per_element: int = 1,
        batched: bool = False,
    ) -> None:
        assert source.shape == target.shape

        def write(indices: tuple[str, ...]) -> None:
            # Cooperative producers execute in every participating lane; only
            # their leader writes the resulting shared element.
            value = self.cast(source.read(indices), target.dtype)
            store = f"{target.read(indices)} = {value}"
            if threads_per_element != 1:
                store = f"if {self.thread} % {threads_per_element} == 0:\n    {store}"
            self.emit(store)

        statements = self.cg.statements_stack[-1]
        start = len(statements)
        self.elements(
            source.shape,
            write,
            threads_per_element=threads_per_element,
            synchronize=not batched,
            batched=batched,
        )
        if self.warp_producer_recorder is not None:
            self.warp_producer_recorder.copy(
                statements[start:], source, target, threads_per_element
            )

    def materialize(
        self,
        value: Fragment,
        *,
        copy: bool = False,
        threads_per_element: int = 1,
        batched: bool = False,
    ) -> Fragment:
        if value.resident and not copy:
            return value
        self.held.append(value)
        result = self.allocate(value)
        self.copy(
            value, result, threads_per_element=threads_per_element, batched=batched
        )
        self.held.pop()
        return result

    def materialize_padded(self, value: Fragment, pad: int) -> Fragment:
        """Copy *value* into a fresh shared buffer whose rows are *pad*
        elements longer than the tile's.  The returned fragment keeps the
        tile's shape and reads the padded rows; it lives only as a contraction
        operand (``emit_tf32_mma``), never as a result."""
        shape = value.shape
        padded_shape = (*shape[:-1], shape[-1] + pad)
        self.held.append(value)
        padded = self.allocate(Fragment(padded_shape, value.dtype, lambda _: "0"))
        view = Fragment(
            shape, value.dtype, padded.read, True, (padded,), padded.storage
        )
        self.copy(value, view)
        self.held.pop()
        return view

    def row_packet_lanes(
        self,
        tensor: HostTensor,
        indices: list[object],
        values: dict[Node, object],
        shape: tuple[int, ...],
        dtype: torch.dtype,
        index_coordinates: list[tuple[int | None, ...]],
    ) -> int | None:
        """``cute_fragment_vector_io``: the 16-byte packet width (in elements)
        when this host window's innermost coordinate is proved to walk whole,
        aligned, in-bounds packets of a unit-stride dimension; None otherwise.
        A staged destination must also keep its packets contiguous."""
        if not self.vector_io:
            return None
        from .fragment_vector_io import row_packet_lanes

        lanes = row_packet_lanes(
            self, tensor, indices, values, shape, dtype, index_coordinates
        )
        if lanes is not None and self.physical_shape(shape, dtype)[-1] % lanes:
            return None
        return lanes

    def vector_load(
        self,
        value: Fragment,
        shape: tuple[int, ...],
        lanes: int,
        address: Callable[[tuple[str, ...]], tuple[str, str]],
        *,
        batched: bool = False,
    ) -> Fragment:
        """Stage *value* (a host window) through 16-byte packets: one address
        and one predicate per packet, a vector global load into registers
        (zeros when the packet is out of bounds, like the scalar walk) and a
        vector store into the dense staged rows."""
        from .fragment_vector_io import copy_atom

        self.held.append(value)
        result = self.allocate(value)
        self.held.pop()
        physical = self.physical_shape(shape, value.dtype)
        scalar = self.dtype(value.dtype)
        atom = copy_atom(scalar)
        assert result.storage is not None

        def body(coords: tuple[str, ...]) -> None:
            inner = self.df.new_var("fragment_packet_inner")
            self.emit(f"{inner} = ({coords[-1]}) * {lanes}")
            pointer, mask = address((*coords[:-1], inner))
            packet = self.df.new_var("fragment_packet")
            self.emit(
                f"{packet} = cute.make_rmem_tensor(cute.make_layout(({lanes},)), {scalar})"
            )
            self.emit(f"{packet}.fill({scalar}(0))")
            self.emit(
                f"if {mask}:\n"
                f"    cute.copy({atom}, cute.make_tensor({pointer}.align({lanes * value.dtype.itemsize}), cute.make_layout(({lanes},))), {packet})"
            )
            flat = self.flatten((*coords[:-1], inner), physical)
            self.emit(
                f"cute.copy({atom}, {packet}, cute.make_tensor({result.storage}.iterator + cute.assume({flat}, divby={lanes}), cute.make_layout(({lanes},))))"
            )

        self.elements(
            (*shape[:-1], shape[-1] // lanes),
            body,
            synchronize=not batched,
            batched=batched,
        )
        return result

    def vector_store(
        self,
        tensor: HostTensor,
        value: object,
        shape: tuple[int, ...],
        lanes: int,
        address: Callable[[tuple[str, ...]], tuple[str, str]],
    ) -> None:
        """Store a window through 16-byte packets: the value recipe is
        evaluated per lane into registers, then one predicated vector store."""
        from .fragment_vector_io import copy_atom

        scalar = self.dtype(tensor.value.dtype)
        atom = copy_atom(scalar)

        def body(coords: tuple[str, ...]) -> None:
            inner = self.df.new_var("fragment_packet_inner")
            self.emit(f"{inner} = ({coords[-1]}) * {lanes}")
            pointer, mask = address((*coords[:-1], inner))
            packet = self.df.new_var("fragment_packet")
            self.emit(
                f"{packet} = cute.make_rmem_tensor(cute.make_layout(({lanes},)), {scalar})"
            )
            for lane in range(lanes):
                lane_coords = (*coords[:-1], f"{inner} + {lane}")
                stored = (
                    value.broadcast(lane_coords)
                    if isinstance(value, Fragment)
                    else self.scalar(value)
                )
                self.emit(f"{packet}[{lane}] = {self.cast(stored, tensor.value.dtype)}")
            self.emit(
                f"if {mask}:\n"
                f"    cute.copy({atom}, {packet}, cute.make_tensor({pointer}.align({lanes * tensor.value.dtype.itemsize}), cute.make_layout(({lanes},))))"
            )

        self.elements((*shape[:-1], shape[-1] // lanes), body)

    def stage_slabs(self) -> None:
        """``cute_fragment_slab_loads``: stage every planned slab at the start
        of the root, all loads in flight together, so each member window
        reads a view of its slab wherever (root body or a conditional tier)
        it appears."""
        groups: list[SlabGroup] = []
        for group in self.slab_plan.values():
            if all(group is not other for other in groups):
                groups.append(group)
        for group in groups:
            fake = cast("torch.Tensor", group.tensor.meta["val"])
            # Only the root's own grid blocks have CTA-uniform offsets here;
            # a window under a device loop's block keeps its own walk.
            staged = all(
                (kind != "block" or value in self.offsets)
                and (kind != "slice" or self.static_extent(fake.shape[dim]) is not None)
                and (kind != "uniform" or self.uniform_at_root(value))
                for dim, (kind, value) in enumerate(group.descriptors)
            ) and all(
                self.uniform_at_root(expr)
                for load in (group.offset, group.limit)
                if load is not None
                for expr in load.indices
            )
            if not staged:
                for member in group.members:
                    cast("dict[Node, SlabGroup]", self.slab_plan).pop(member, None)
                continue
            for load in (group.offset, group.limit):
                if load is not None:
                    self.slab_scalar(load)
            group.fragment = self.load_slab(group)
            # Alive for the whole root: no later phase may recycle it.
            self.held.append(group.fragment)
        self.flush_pending_barrier()

    def uniform_at_root(self, expr: object) -> bool:
        """Whether the uniform integer expression ``expr`` is defined at the
        start of the root: its grid symbols are the root's own blocks (a
        device loop's tile symbols are not yet bound there)."""
        if not isinstance(expr, sympy.Expr):
            return False
        origins = HostFunction.current().expr_to_origin
        for symbol in expr.free_symbols:
            info = origins.get(symbol)
            origin = info.origin if info is not None else None
            if isinstance(origin, GridOrigin) and origin.block_id not in self.offsets:
                return False
        return True

    def slab_scalar(self, load: ScalarLoad) -> str:
        """The uniform 0-d integer ``load`` (a slab's row offset or limit) as
        an Int32 read once per CTA ahead of the slabs: zero when its
        subscript is outside the tensor, like the body's own load of it.
        (An int64 table value is narrowed: a row index or count of a static
        extent below 2**31.)"""
        if load in self.slab_scalars:
            return self.slab_scalars[load]
        fake = cast("torch.Tensor", load.node.meta["val"])
        tensor = HostTensor(
            self.df.tensor_arg(fake, prefer_name=load.tensor).name, fake
        )
        indices = [self.sym(expr) for expr in load.indices]
        masks = [
            f"(({index}) >= 0 and ({index}) < ({self.sym(fake.shape[dim])}))"
            for dim, index in enumerate(indices)
        ]
        offset = (
            " + ".join(
                f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                for dim, index in enumerate(indices)
            )
            or "0"
        )
        name = self.df.new_var("fragment_slab_scalar")
        self.emit(f"{name} = cutlass.Int32(0)")
        self.emit(
            f"if {self.predicate(masks)}:\n"
            f"    {name} = cutlass.Int32(({tensor.name}.iterator + ({offset})).load())"
        )
        self.slab_scalars[load] = name
        return name

    def slab_member(
        self, node: Node, shape: tuple[int, ...], dtype: torch.dtype
    ) -> Fragment:
        """The window *node* as a view of its staged slab."""
        group = self.slab_plan[node]
        slab = group.fragment
        assert slab is not None, "slabs are staged before the root body"
        start, extent = group.members[node]
        offset = start - group.start

        def view(coords: tuple[str, ...]) -> str:
            mapped: list[str] = []
            position = 0
            for dim, (kind, _value) in enumerate(group.descriptors):
                if dim == group.window_dim:
                    if extent == 0:
                        # An integer index drops the dimension.
                        mapped.append(str(offset))
                    else:
                        mapped.append(f"({coords[position]}) + {offset}")
                        position += 1
                elif kind in ("const", "uniform"):
                    mapped.append("0")
                else:
                    mapped.append(coords[position])
                    position += 1
            assert position == len(coords)
            return slab.read(tuple(mapped))

        # The view keeps the slab's row order: a cast of it may be read in
        # place by the mma lanes like the slab itself.
        return Fragment(
            shape,
            dtype,
            view,
            True,
            (slab,),
            slab.storage,
            cheap=True,
            physical_row=slab.physical_row,
        )

    def load_slab(self, group: SlabGroup) -> Fragment:
        """Stage one slab: the member windows' union along the varying
        dimension, with the shared block / slice / constant indices."""
        from .fragment_vector_io import host_base_aligned
        from .fragment_vector_io import packet_lanes
        from .fragment_vector_io import static_strides

        fake = cast("torch.Tensor", group.tensor.meta["val"])
        tensor = HostTensor(
            self.df.tensor_arg(
                fake, prefer_name=cast("str", group.tensor.args[0])
            ).name,
            fake,
        )
        shape: list[int] = []
        for dim, (kind, value) in enumerate(group.descriptors):
            if dim == group.window_dim:
                shape.append(group.extent)
            elif kind == "block":
                shape.append(
                    self.extent(self.df.resolved_block_size(cast("int", value)))
                )
            elif kind == "slice":
                shape.append(self.extent(fake.shape[dim]))
            elif kind in ("const", "uniform"):
                shape.append(1)
            else:
                shape.append(cast("tuple[int, int]", value)[1])
        slab_shape = tuple(shape)
        offset_var = None if group.offset is None else self.slab_scalars[group.offset]
        limit_var = None if group.limit is None else self.slab_scalars[group.limit]

        def address(coords: tuple[str, ...]) -> tuple[str, str]:
            indices: list[str] = []
            masks: list[str] = []
            for dim, (kind, value) in enumerate(group.descriptors):
                if dim == group.window_dim:
                    row = f"({group.start}) + ({coords[dim]})"
                    if offset_var is not None:
                        # A window at a runtime row offset: live below the
                        # limit (in the window's own coordinates) and
                        # inside the tensor; zero past either.
                        if limit_var is not None:
                            masks.append(f"(({row}) < ({limit_var}))")
                        row = f"({offset_var}) + {row}"
                        masks.append(
                            f"(({row}) >= 0 and ({row}) < ({self.sym(fake.shape[dim])}))"
                        )
                    indices.append(row)
                elif kind == "block":
                    block = cast("int", value)
                    expression = f"({self.offsets[block]}) + ({coords[dim]})"
                    masks.append(f"(({expression}) < ({self.bounds[block]}))")
                    indices.append(expression)
                elif kind == "slice":
                    indices.append(coords[dim])
                elif kind == "const":
                    indices.append(str(value))
                elif kind == "uniform":
                    expression = self.sym(value)
                    masks.append(
                        f"(({expression}) >= 0 and ({expression}) < ({self.sym(fake.shape[dim])}))"
                    )
                    indices.append(expression)
                else:
                    indices.append(
                        f"({cast('tuple[int, int]', value)[0]}) + ({coords[dim]})"
                    )
            offset = (
                " + ".join(
                    f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                    for dim, index in enumerate(indices)
                )
                or "0"
            )
            return f"({tensor.name}.iterator + ({offset}))", self.predicate(masks)

        def load(coords: tuple[str, ...]) -> str:
            pointer, mask = address(coords)
            name = self.df.new_var("fragment_load")
            self.emit(f"{name} = {self.cast('0', fake.dtype)}")
            loaded = self.cast(f"{pointer}.load()", fake.dtype)
            self.emit(f"if {mask}:\n    {name} = {loaded}")
            return name

        value = Fragment(slab_shape, fake.dtype, load)
        batched = self.batched_loads and self.warp_producer_recorder is None
        lanes = packet_lanes(fake.dtype)
        strides = static_strides(fake)
        # The slab's rows are packets when its last dim is the whole
        # contiguous dim, or a static window of it at a packet-aligned start
        # and extent inside the dim (``hl.arange(D)`` over a D-wide dim).
        # The varying dimension (``("window", None)``: the members' windows
        # along it) is walked element by element, never packed.
        inner_size = self.static_extent(fake.shape[-1])
        inner_kind, inner_value = group.descriptors[-1]
        if group.window_dim == len(slab_shape) - 1:
            inner_packets = False
        elif inner_kind == "slice":
            inner_packets = True
        elif inner_kind == "window":
            inner_start, inner_extent = cast("tuple[int, int]", inner_value)
            inner_packets = (
                lanes is not None
                and inner_size is not None
                and inner_start % lanes == 0
                and inner_extent % lanes == 0
                and inner_start + inner_extent <= inner_size
            )
        else:
            inner_packets = False
        if (
            self.vector_io
            and lanes is not None
            and strides is not None
            and inner_packets
            and strides[-1] == 1
            and all(stride % lanes == 0 for stride in strides[:-1])
            and slab_shape[-1] % lanes == 0
            and self.physical_shape(slab_shape, fake.dtype)[-1] % lanes == 0
            and host_base_aligned(self.env, fake)
        ):
            result = self.vector_load(
                value, slab_shape, lanes, address, batched=batched
            )
        else:
            result = self.materialize(value, batched=batched)
        self.pending_barrier = batched
        return result

    def aggregate_local_updates(
        self,
        shape: tuple[int, ...],
        storage: str,
        variables: tuple[str, ...],
        prepare: Callable[[tuple[str, ...]], None],
        *,
        register_owned: bool = False,
    ) -> None:
        """Aggregate one private relaxed Int32 update epoch, without new barriers.

        All CTA threads enter every round. Only lanes in the ballot execute
        match/redux, using the same member mask for each equal wrapped address.
        No observer can read the private target until its existing epoch barrier.
        """
        assert self.threads % 32 == 0
        register_owned = register_owned or shape in self.snapshot_shapes
        size = math.prod(shape)
        iteration = self.df.new_var("fragment_atomic_round")
        index = self.df.new_var("fragment_atomic_element")
        address, value, active = variables
        slots = (size + self.threads - 1) // self.threads
        if register_owned:
            assert 0 < slots <= MAX_SLOTS
        loop_range = "cutlass.range_constexpr" if register_owned else "range"
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {iteration} in {loop_range}({slots}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            self.emit(f"{index} = {iteration} * {self.threads} + {self.thread}")
            # Dominating definitions are required by CuTe's staged SSA lowering.
            self.emit(f"{address} = cutlass.Int64(0)")
            self.emit(f"{value} = cutlass.Int32(0)")
            self.emit(f"{active} = False")
            valid = cast(
                "ast.If", statement_from_string(f"if {index} < {size}:\n    pass")
            )
            valid.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                prior = self.local_register_slot
                if register_owned:
                    self.local_register_slot = iteration
                try:
                    with self.snapshot_scope(index, iteration, size):
                        prepare(self.coordinates(index, shape))
                finally:
                    self.local_register_slot = prior
            self.cg.add_statement(valid)
            members = self.df.new_var("fragment_atomic_members")
            peers = self.df.new_var("fragment_atomic_peers")
            total = self.df.new_var("fragment_atomic_total")
            self.emit(f"{members} = cute.arch.vote_ballot_sync({active})")
            self.emit(
                f"if {active}:\n"
                f"    {peers} = cute.arch.match_sync({members}, {address})\n"
                f"    {total} = cute.arch.warp_redux_sync({value}, 'add', {peers})\n"
                f"    if ({peers} & cute.arch.lanemask_lt()) == 0:\n"
                f"        cute.arch.atomic_add(({storage}.iterator + {address}).llvm_ptr, "
                f"{total}, sem='relaxed', scope='cta')"
            )
        self.cg.add_statement(loop)

    def atomic_add(
        self,
        node: Node,
        values: dict[Node, object],
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
    ) -> Fragment | None:
        """Each contribution is owned once; barriers are outside lane loops."""
        target, indices, value, sem = cast(
            "tuple[object, object, object, object]", _resolve(node.args, values)
        )
        assert isinstance(target, (Fragment, HostTensor))
        assert isinstance(indices, (tuple, list))
        fake = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(fake.shape)
        target_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
        if len(indices) != target_fake.ndim or any(
            isinstance(i, slice) or i is None for i in indices
        ):
            raise exc.InvalidConfig(
                "fragment atomics require scalar/tensor indices for every axis"
            )
        local = isinstance(target, Fragment)
        if sem != "relaxed" and (local or target_fake.dtype != torch.int32):
            raise exc.InvalidConfig(
                "ordered fragment atomics require a global int32 target"
            )
        if node.users and target_fake.dtype != torch.int32:
            raise exc.InvalidConfig(
                "fragment atomic results require a global int32 target "
                "or a relaxed local int32 target"
            )
        if isinstance(target, Fragment) and (
            target.storage is None or len(target.shape) != 1
        ):
            raise exc.InvalidConfig("CTA-local atomics require direct resident storage")
        if target_fake.dtype not in (torch.float32, torch.int32):
            raise exc.InvalidConfig("fragment atomic add supports float32/int32")

        if sem in ("release", "acq_rel"):
            # Each worker publishes its preceding global writes before any
            # elected scalar/vector owner performs the release operation.
            # A fence in only the owner would not publish other workers' writes.
            # Both operations are collective, outside contribution masks/loops.
            self.emit("cute.arch.fence_acq_rel_gpu()")
            self.synchronize()

        def logical_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            masks = [
                f"({coord}) < ({self.logical_axis_extent(fake, dim, shape[dim])})"
                for dim, coord in enumerate(coords)
            ]
            # Indexing can pad the result shape after an index tensor has
            # retained its concrete logical extent. Destination bounds do not
            # exclude those extra contributions when an index wraps or clamps.
            # Preserve each index operand's broadcast domain independently.
            index_nodes = cast("list[Node | int]", node.args[1])
            for source, index in zip(index_nodes, indices, strict=True):
                if not isinstance(index, Fragment):
                    continue
                index_fake = cast("torch.Tensor", cast("Node", source).meta["val"])
                offset = len(coords) - len(index.shape)
                for dim, capacity in enumerate(index.shape):
                    if capacity == 1:
                        continue  # A singleton broadcasts over the output axis.
                    extent = self.logical_axis_extent(index_fake, dim, capacity)
                    mask = f"({coords[offset + dim]}) < ({extent})"
                    if mask not in masks:
                        masks.append(mask)
                if index.logical_domain is not None:
                    for mask in index.broadcast_domain(coords):
                        if mask not in masks:
                            masks.append(mask)
            return tuple(masks)

        # The side effect is emitted once below, never embedded in a lazy
        # element recipe. Repeated consumers read this immutable snapshot.
        warp_groups = self.warp_result_nodes.get(node)
        warp_owned = (
            not local and warp_groups is not None and math.prod(shape) == warp_groups
        )
        register = (
            self.df.new_var("fragment_warp_atomic")
            if node.users and warp_owned
            else None
        )
        if register is not None:
            self.emit(f"{register} = {self.cast('0', target_fake.dtype)}")
        iteration_ticket = (
            self.df.new_var("fragment_iteration_ticket")
            if node in self.iteration_local_returns
            else None
        )
        if iteration_ticket is not None:
            assert deferred_updates is not None and local and node.users
            assert sem == "relaxed" and target_fake.dtype == torch.int32
            assert len(shape) == 1 and not warp_owned
        local_registers = (
            self.df.new_var("fragment_local_tickets")
            if local
            and node.users
            and node in self.local_register_nodes
            and iteration_ticket is None
            else None
        )
        if local_registers is not None:
            slots = (math.prod(shape) + self.threads - 1) // self.threads
            self.emit(
                f"{local_registers} = cute.make_rmem_tensor(({slots},), cutlass.Int32)"
            )
            self.emit(f"{local_registers}.fill(0)")

        def local_register_read(_coords: tuple[str, ...]) -> str:
            assert self.local_register_slot is not None
            return f"{local_registers}[{self.local_register_slot}]"

        def iteration_read(coords: tuple[str, ...]) -> str:
            assert self.iteration_owner is not None
            assert node in self.iteration_local_returns
            assert node in self.iteration_defined
            self.iteration_owner.prove(coords[0])
            assert iteration_ticket is not None
            return iteration_ticket

        result = (
            Fragment(
                shape,
                target_fake.dtype,
                iteration_read,
                True,
                logical_domain=logical_domain,
            )
            if iteration_ticket is not None
            else Fragment(
                shape,
                target_fake.dtype,
                local_register_read,
                True,
                logical_domain=logical_domain,
            )
            if local_registers is not None
            else Fragment(
                shape,
                target_fake.dtype,
                lambda _coords: cast("str", register),
                True,
                logical_domain=logical_domain,
            )
            if register is not None
            else self.allocate(
                Fragment(
                    shape,
                    target_fake.dtype,
                    lambda coords: "0",
                    logical_domain=logical_domain,
                )
            )
            if node.users
            else None
        )

        aggregate = (
            local
            and target_fake.dtype == torch.int32
            and sem == "relaxed"
            and not node.users
            and not warp_owned
            and self.df.config.get("cute_fragment_atomic_aggregation", False) is True
        )
        if aggregate:
            root = self.cg.current_root_graph_info
            assert root is not None
            aggregate = (
                root.graph_id
                in self.env.config_spec.cute_fragment_atomic_aggregation_root_ids
            )
        # Zero has no counter effect; returned zeros additionally require every
        # observable consumer to be disabled. Keep masks, value conversion,
        # result snapshot writes and epoch publication when updates are skipped.
        skip_zero = (
            local
            and target_fake.dtype == torch.int32
            and sem == "relaxed"
            and (not node.users or node in self.dead_zero_results)
            and self.df.config.get("cute_fragment_skip_zero_atomics", False) is True
        )
        if skip_zero:
            root = self.cg.current_root_graph_info
            assert root is not None
            skip_zero = (
                root.graph_id
                in self.env.config_spec.cute_fragment_skip_zero_atomics_root_ids
            )
        aggregate_vars = (
            tuple(
                self.df.new_var(name)
                for name in (
                    "fragment_atomic_address",
                    "fragment_atomic_value",
                    "fragment_atomic_active",
                )
            )
            if aggregate
            else None
        )

        def update_in_domain(coords: tuple[str, ...], previous: str | None) -> None:
            masks = []
            positions = []
            for dim, index in enumerate(indices):
                position = (
                    index.broadcast(coords)
                    if isinstance(index, Fragment)
                    else self.scalar(index)
                )
                position = self.cg.lift(
                    expr_from_string(position), prefix="fragment_atomic_index"
                ).id
                # The logical allocation can be smaller than its padded capacity.
                size = self.sym(target_fake.shape[dim])
                if local:
                    assert isinstance(target, Fragment) and target.storage is not None
                    size = str(self.local_logical_sizes[id(target)])
                logical_extent = (
                    self.local_logical_sizes[id(target)]
                    if isinstance(target, Fragment)
                    else target_fake.shape[dim]
                )
                if (
                    isinstance(index, int)
                    and isinstance(logical_extent, int)
                    and not -logical_extent <= index < logical_extent
                ):
                    raise exc.InvalidConfig(
                        "fragment atomic static index is outside the logical target"
                    )
                # Python/PyTorch indexing wraps one valid negative extent.
                # Widen before adding a possibly 64-bit host extent; this is
                # address arithmetic, not a conversion of the update value.
                wrapped = self.df.new_var("fragment_atomic_wrapped")
                self.emit(f"{wrapped} = cutlass.Int64({position})")
                self.emit(
                    f"if {wrapped} < 0:\n    {wrapped} = {wrapped} + cutlass.Int64({size})"
                )
                masks.append(f"0 <= ({wrapped}) and ({wrapped}) < ({size})")
                positions.append(wrapped)
            if isinstance(target, Fragment):
                pointer = f"({target.storage}.iterator + ({positions[0]})).llvm_ptr"
                scope = "cta"
            else:
                offset = " + ".join(
                    f"cutlass.Int64({position}) * cutlass.Int64({self.host_stride(target, dim)})"
                    for dim, position in enumerate(positions)
                )
                pointer = f"({target.name}.iterator + ({offset})).llvm_ptr"
                scope = "gpu"
            contribution = (
                value.broadcast(coords)
                if isinstance(value, Fragment)
                else self.scalar(value)
            )
            if aggregate_vars is not None:
                address, update_value, active = aggregate_vars
                self.emit(f"{address} = cutlass.Int64({positions[0]})")
                self.emit(
                    f"if {self.predicate(masks)}:\n"
                    f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                    f"    {active} = {update_value} != 0"
                )
                return
            atomic = (
                f"cute.arch.atomic_add({pointer}, "
                f"{self.cast(contribution, target_fake.dtype)}, "
                f"sem={sem!r}, scope={scope!r})"
            )
            if skip_zero and result is None:
                update_value = self.df.new_var("fragment_atomic_nonzero_value")
                self.emit(
                    f"if {self.predicate(masks)}:\n"
                    f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                    f"    if {update_value} != cutlass.Int32(0):\n"
                    f"        cute.arch.atomic_add({pointer}, {update_value}, "
                    f"sem={sem!r}, scope={scope!r})"
                )
            elif result is None:
                self.emit(f"if {self.predicate(masks)}:\n    {atomic}")
            else:
                assert previous is not None
                if skip_zero:
                    update_value = self.df.new_var("fragment_atomic_nonzero_value")
                    self.emit(
                        f"if {self.predicate(masks)}:\n"
                        f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                        f"    if {update_value} != cutlass.Int32(0):\n"
                        f"        {previous} = cute.arch.atomic_add({pointer}, "
                        f"{update_value}, sem={sem!r}, scope={scope!r})"
                    )
                else:
                    self.emit(f"if {self.predicate(masks)}:\n    {previous} = {atomic}")

        def update(coords: tuple[str, ...]) -> None:
            previous = None
            if result is not None:
                previous = iteration_ticket or self.df.new_var(
                    "fragment_atomic_previous"
                )
                self.emit(f"{previous} = {self.cast('0', target_fake.dtype)}")

            # Evaluating a lazy index can read a captured tensor whose declared
            # extent is smaller than the padded atomic iteration domain. Check
            # the contribution domain before evaluating that index, rather than
            # masking only the final atomic after the index has already read it.
            masks = logical_domain(coords)
            if masks:
                branch = cast(
                    "ast.If",
                    statement_from_string(f"if {self.predicate(masks)}:\n    pass"),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    update_in_domain(coords, previous)
                self.cg.add_statement(branch)
            else:
                update_in_domain(coords, previous)

            if result is not None:
                if iteration_ticket is not None:
                    self.iteration_defined.add(node)
                else:
                    # Invalid contributions still produce the existing zero
                    # result, including padded lanes of shared snapshots.
                    self.emit(f"{result.read(coords)} = {previous}")

        if deferred_updates is not None:
            assert local and not warp_owned
            assert aggregate_vars is None and target_fake.dtype == torch.int32
            deferred_updates.append((update, result is not None))
        elif aggregate_vars is None:
            # Returned local values are immutable per-element snapshots. Finish
            # every worker's update/result write before a cross-lane consumer;
            # the target and result have distinct live allocations. Local
            # unused-result epochs retain their existing coalescing behavior.
            self.elements(
                shape,
                update,
                warp_owned=warp_owned,
                synchronize=not local or result is not None,
                register_owned=node in self.local_register_nodes,
            )
        else:
            assert isinstance(target, Fragment) and target.storage is not None
            self.aggregate_local_updates(
                shape,
                target.storage,
                aggregate_vars,
                update,
                register_owned=node in self.local_register_nodes,
            )
        if local and result is None:
            assert isinstance(target, Fragment) and target.storage is not None
            self.pending_local_atomics.add(target.storage)
        if register is not None:
            # Preserve the existing atomic ordering barrier, including unrelated
            # alias loads. Only the return-value exchange becomes warp-local.
            self.emit(
                f"if {self.thread} // 32 < {warp_groups}:\n"
                f"    {register} = cute.arch.shuffle_sync({register}, 0, mask=0xffffffff, mask_and_clamp=31)"
            )
        if result is not None and not shape and sem in ("acquire", "acq_rel"):
            self.scalar_ordered_tickets[node] = result
        return result

    def completed_local_load(
        self, node: Node, values: dict[Node, object], source: Fragment
    ) -> Fragment:
        """Snapshot a masked indexed read after the target's final atomic epoch."""
        assert id(source) in self.local_logical_sizes and source.storage is not None
        assert local_indexed_load(node)
        index_node = cast("list[object]", node.args[1])[0]
        index = values[index_node] if isinstance(index_node, Node) else index_node
        mask_node = node.args[2]
        mask = values[mask_node] if isinstance(mask_node, Node) else None
        fake = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(fake.shape)
        if len(shape) > 1 or (
            isinstance(mask, Fragment)
            and (
                len(mask.shape) > len(shape)
                or any(
                    left not in (1, right)
                    for left, right in zip(
                        mask.shape, shape[len(shape) - len(mask.shape) :], strict=True
                    )
                )
            )
        ):
            raise exc.InvalidConfig("local indexed loads require scalar/vector masks")
        extent = self.local_logical_sizes[id(source)]
        tensor_index = isinstance(index_node, Node) and isinstance(
            index_node.meta.get("val"), torch.Tensor
        )
        # Match hl.load's reference contract: tensor indices with an explicit
        # mask use nonnegative bounds, while unmasked/scalar indexing follows
        # PyTorch's one-extent negative normalization. Do not assign an output
        # value to unmasked out-of-bounds accesses: require a proved range.
        wrap_negative = mask_node is None or not tensor_index
        if wrap_negative:
            from .bounded_gather import integer_bounds

            interval = integer_bounds(index_node, captured_bounds={})
            if interval is None or not -extent <= interval[0] <= interval[1] < extent:
                raise exc.InvalidConfig(
                    "unmasked local indexed loads require a proved logical index range"
                )
        dependencies = (
            source,
            *(value for value in (index, mask) if isinstance(value, Fragment)),
        )

        def output_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            # Shape ownership survives the copy; runtime index/mask values do
            # not. Never retain a lazy shared read in the result's domain.
            return tuple(
                f"({coord}) < ({self.logical_axis_extent(fake, dim, size)})"
                for dim, (coord, size) in enumerate(zip(coords, shape, strict=True))
            )

        def read(coords: tuple[str, ...]) -> str:
            result = self.df.new_var("fragment_local_load")
            self.emit(f"{result} = {self.cast('0', source.dtype)}")
            domains = list(output_domain(coords))
            if isinstance(index, Fragment):
                domains.extend(index.broadcast_domain(coords))
            if isinstance(mask, Fragment):
                domains.extend(mask.broadcast_domain(coords))
            branch = statement_from_string(f"if {self.predicate(domains)}:\n    pass")
            assert isinstance(branch, ast.If)
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                location = self.df.new_var("fragment_local_index")
                expression = (
                    index.broadcast(coords)
                    if isinstance(index, Fragment)
                    else self.scalar(index)
                )
                if wrap_negative:
                    self.emit(f"{location} = cutlass.Int64({expression})")
                    self.emit(
                        f"if {location} < 0:\n"
                        f"    {location} = {location} + cutlass.Int64({extent})"
                    )
                else:
                    self.emit(f"{location} = {expression}")
                valid = [f"0 <= ({location}) and ({location}) < ({extent})"]
                if isinstance(mask, Fragment):
                    valid.append(mask.broadcast(coords))
                predicate = self.predicate(valid)
                self.emit(f"if {predicate}:\n    {result} = {source.read((location,))}")
            self.cg.add_statement(branch)
            return result

        # The normal node boundary published pending writes before entering
        # memory(). Holding every dependency prevents result allocation from
        # recycling the source or a still-lazy index/mask operand.
        return self.materialize(
            Fragment(
                shape,
                source.dtype,
                read,
                dependencies=dependencies,
                logical_domain=output_domain,
            )
        )

    def memory(
        self,
        node: Node,
        values: dict[Node, object],
        store: bool,
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
        direct: DirectStoreSink | None = None,
    ) -> object:
        if store and node in self.direct_stores:
            # Written by the tf32 mma lanes of the dot it stores.
            return None
        if deferred_updates is not None:
            values = dict(values)
        tensor = values[cast("Node", node.args[0])]
        indices = cast("list[object]", node.args[1])
        output = (
            cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
            if store
            else cast("torch.Tensor", node.meta["val"])
        )
        # A scalar or broadcast value does not describe the indexed write
        # domain. Use the same destination shape rule as fake tensor loads.
        shape = self.shape(
            SubscriptIndexing.compute_shape(
                output,
                [
                    index.meta["val"] if isinstance(index, Node) else index
                    for index in indices
                ],
            )
            if store
            else output.shape
        )
        extra_mask = node.args[3] if store else node.args[2]
        mask_value = values[extra_mask] if isinstance(extra_mask, Node) else extra_mask
        if isinstance(tensor, Fragment):
            if (
                not store
                and id(tensor) in self.local_logical_sizes
                and local_indexed_load(node)
            ):
                return self.completed_local_load(node, values, tensor)
            if (
                store
                or extra_mask is not None
                or node.args[3] is not None
                or any(index is not None and index != slice(None) for index in indices)
            ):
                raise exc.BackendUnsupported("cute", "computed fragment indexed memory")

            def view_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                selected = tuple(
                    coord
                    for coord, index in zip(coords, indices, strict=False)
                    if index is not None
                )
                return (*selected, *coords[len(indices) :])

            def view(coords: tuple[str, ...]) -> str:
                return tensor.read(self.coordinate_locals(view_coordinates(coords)))

            return Fragment(
                shape,
                output.dtype,
                view,
                tensor.resident,
                dependencies=(tensor,),
                logical_domain=lambda coords: tensor.domain(view_coordinates(coords)),
            )
        assert isinstance(tensor, HostTensor)
        if not store and node in self.slab_plan:
            return self.slab_member(node, shape, output.dtype)
        index_coordinates = memory_index_coordinates(
            self.env,
            [
                index.meta["val"] if isinstance(index, Node) else index
                for index in indices
            ],
            {
                i: value.shape
                for i, index in enumerate(indices)
                if isinstance(index, Node)
                and isinstance(value := values[index], Fragment)
            },
            shape,
        )

        def selected_coordinates(
            ordinal: int, coords: tuple[str, ...]
        ) -> tuple[str, ...]:
            return tuple(
                "0" if axis is None else coords[axis]
                for axis in index_coordinates[ordinal]
            )

        def address(coords: tuple[str, ...]) -> tuple[str, str]:
            tensor_indices: list[str] = []
            masks: list[str] = []
            dim = 0
            for ordinal, index in enumerate(indices):
                if index is None:
                    continue
                proxy = index.meta["val"] if isinstance(index, Node) else index
                block = (
                    self.env.resolve_block_id(proxy)
                    if isinstance(proxy, torch.SymInt)
                    else None
                )
                if (
                    block is not None
                    and cast("torch.SymInt", proxy)._sympy_()
                    != self.env.block_sizes[block].var._sympy_()
                ):
                    block = None
                if isinstance(index, slice):
                    assert index == slice(None)
                    expression = selected_coordinates(ordinal, coords)[0]
                elif block is not None:
                    expression = f"({self.offsets[block]}) + ({selected_coordinates(ordinal, coords)[0]})"
                    masks.append(f"(({expression}) < ({self.bounds[block]}))")
                elif isinstance(index, Node):
                    value = values[index]
                    if isinstance(value, Fragment) and value.shape:
                        selected = selected_coordinates(ordinal, coords)
                        domain = value.domain(selected)
                        masks.extend(domain)
                        # Index fragments may have exact static capacity while
                        # the indexed output is padded. Read only coordinates
                        # owned by the index, before dereferencing host memory.
                        expression = self.df.new_var("fragment_tensor_index")
                        self.emit(f"{expression} = {self.cast('0', value.dtype)}")
                        branch = statement_from_string(
                            f"if {self.predicate(domain)}:\n    pass"
                        )
                        assert isinstance(branch, ast.If)
                        branch.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                            self.emit(f"{expression} = {value.read(selected)}")
                        self.cg.add_statement(branch)
                    else:
                        expression = self.scalar(value)
                else:
                    expression = self.scalar(index)
                tensor_indices.append(expression)
                size = self.sym(tensor.value.shape[dim])
                masks.append(f"(({expression}) >= 0 and ({expression}) < ({size}))")
                dim += 1
            if isinstance(mask_value, Fragment):
                masks.append(mask_value.broadcast(coords))
            elif mask_value is not None:
                masks.append(self.scalar(mask_value))
            offset = (
                " + ".join(
                    f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                    for dim, index in enumerate(tensor_indices)
                )
                or "0"
            )
            return f"({tensor.name}.iterator + ({offset}))", self.predicate(masks)

        if store:
            if direct is not None:
                # Hand the window's address recipe to the producing dot's lane
                # loop instead of walking a staged result.  The walks this
                # method would otherwise choose (deferred region updates,
                # warp- or register-owned results, an extra mask) keep the
                # staged path: the sink stays unfilled.
                if (
                    deferred_updates is not None
                    or mask_value is not None
                    or self.warp_result_nodes.get(node) == math.prod(shape)
                    or node in self.local_register_nodes
                ):
                    return None
                direct.shape = shape
                direct.dtype = tensor.value.dtype
                direct.address = address
                direct.lanes = (
                    self.row_packet_lanes(
                        tensor, indices, values, shape, output.dtype, index_coordinates
                    )
                    if self.warp_producer_recorder is None
                    else None
                )
                return None
            value = _resolve(node.args[2], values)

            def write(coords: tuple[str, ...]) -> None:
                pointer, mask = address(coords)
                stored = (
                    value.broadcast(coords)
                    if isinstance(value, Fragment)
                    else self.scalar(value)
                )
                # Round/truncate to the logical tensor dtype before CuTe
                # converts to the pointer's storage type. In particular, an
                # unsigned byte must retain 128..255 when its i8 pointer is signed.
                self.emit(
                    f"if {mask}:\n    {pointer}.store({self.cast(stored, tensor.value.dtype)})"
                )

            if deferred_updates is not None:
                deferred_updates.append((write, True))
                return None
            warp_owned = self.warp_result_nodes.get(node) == math.prod(shape)
            register_owned = node in self.local_register_nodes
            lanes = (
                self.row_packet_lanes(
                    tensor, indices, values, shape, output.dtype, index_coordinates
                )
                if not warp_owned
                and not register_owned
                and mask_value is None
                and self.warp_producer_recorder is None
                else None
            )
            if lanes is not None:
                self.vector_store(tensor, value, shape, lanes, address)
                return None
            self.elements(
                shape,
                write,
                warp_owned=warp_owned,
                register_owned=register_owned,
            )
            return None

        def load(coords: tuple[str, ...]) -> str:
            pointer, mask = address(coords)
            name = self.df.new_var("fragment_load")
            self.emit(f"{name} = {self.cast('0', output.dtype)}")
            # Pointer arithmetic can erase unsigned signedness (Uint8 -> Int8),
            # and bool pointers use byte storage. Restore the logical type before
            # a masked branch joins its typed zero or the value enters arithmetic.
            loaded = self.cast(f"{pointer}.load()", output.dtype)
            self.emit(f"if {mask}:\n    {name} = {loaded}")
            return name

        index_domains: list[tuple[int, Fragment]] = []
        domain_bounds: list[tuple[int, str]] = []
        dim = 0
        for ordinal, index in enumerate(indices):
            if index is None:
                continue
            if isinstance(index, slice):
                domain_bounds.append((ordinal, self.sym(tensor.value.shape[dim])))
            elif isinstance(index, Node) and isinstance(values[index], Fragment):
                index_domains.append((ordinal, cast("Fragment", values[index])))
            elif isinstance(index, Node):
                proxy = index.meta["val"]
                if isinstance(proxy, torch.SymInt):
                    block = self.env.resolve_block_id(proxy)
                    if (
                        block in self.offsets
                        and proxy._sympy_() == self.env.block_sizes[block].var._sympy_()
                    ):
                        domain_bounds.append(
                            (
                                ordinal,
                                f"({self.bounds[block]}) - ({self.offsets[block]})",
                            )
                        )
            dim += 1

        def logical_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            return (
                *(
                    f"({selected_coordinates(ordinal, coords)[0]}) < ({bound})"
                    for ordinal, bound in domain_bounds
                ),
                *(
                    mask
                    for ordinal, index_value in index_domains
                    for mask in index_value.domain(
                        selected_coordinates(ordinal, coords)
                    )
                ),
            )

        loaded_fragment = Fragment(
            shape,
            output.dtype,
            load,
            logical_domain=logical_domain if index_domains or domain_bounds else None,
        )
        if node in self.snapshot_loads:
            slots = (math.prod(shape) + self.threads - 1) // self.threads
            name = self.df.new_var("fragment_snapshot")
            self.register_snapshot_buffers[name] = shape[0]
            self.emit(
                f"{name} = cute.make_rmem_tensor(({slots},), {self.dtype(output.dtype)})"
            )
            self.emit(f"{name}.fill({self.cast('0', output.dtype)})")

            def read_snapshot(coords: tuple[str, ...]) -> str:
                owner = self.snapshot_owner
                if (
                    owner is None
                    or len(coords) != 1
                    or owner.size
                    not in (shape[0], self.shape(snapshot_logical_shape(node))[0])
                ):
                    raise exc.InvalidConfig(
                        "register snapshot lacks its active coordinate owner"
                    )
                owner.prove(coords[0])
                return f"{name}[{owner.slot}]"

            def fill_snapshot(coords: tuple[str, ...]) -> None:
                self.emit(
                    f"{read_snapshot(coords)} = {self.cast(load(coords), output.dtype)}"
                )

            # Preserve the old materialization barrier and every guarded load.
            self.elements(shape, fill_snapshot)
            return Fragment(
                shape,
                output.dtype,
                read_snapshot,
                True,
                logical_domain=loaded_fragment.logical_domain,
                snapshot=SnapshotBinding(
                    node,
                    name,
                    self.threads,
                    shape,
                    output.dtype,
                    frozenset(
                        self.referenced_buffers(
                            [
                                values[index]
                                for index in indices
                                if isinstance(index, Node)
                            ]
                        )
                    ),
                    self.shape(snapshot_logical_shape(node)),
                ),
            )
        batched = self.batchable_host_load(node, values)
        lanes = (
            self.row_packet_lanes(
                tensor, indices, values, shape, output.dtype, index_coordinates
            )
            if mask_value is None and self.warp_producer_recorder is None
            else None
        )
        if lanes is not None:
            result = self.vector_load(
                loaded_fragment, shape, lanes, address, batched=batched
            )
            self.pending_barrier = batched
            return result
        if self.df.config.get(
            "cute_fragment_packet_loads", False
        ) and host_load_is_readonly(node, self.env, self.graphs):
            from .packet_loads import materialize_packet_load

            index_values = (
                values[index] for index in indices if isinstance(index, Node)
            )
            statements = self.cg.statements_stack[-1]
            start = len(statements)
            packet = materialize_packet_load(
                self,
                tensor,
                loaded_fragment,
                load,
                tuple(value for value in index_values if isinstance(value, Fragment))
                + ((mask_value,) if isinstance(mask_value, Fragment) else ()),
            )
            if packet is not None:
                if self.warp_producer_recorder is not None:
                    self.warp_producer_recorder.publication(statements[start:], packet)
                return packet
        if (
            self.df.config.get("cute_fragment_register_loads", False)
            and math.prod(shape) <= self.threads
            and lane_private_load(node, self.env)
            and host_load_is_readonly(node, self.env, self.graphs)
        ):
            # One typed scalar per active lane, in the current lexical body.
            # The graph proof forbids remapping, carries and branch escapes.
            # Keep the ordinary atomic/store element loops and their barriers.
            name = self.df.new_var("fragment_register_load")
            self.emit(f"{name} = {self.cast('0', output.dtype)}")
            branch = statement_from_string(
                f"if {self.thread} < {math.prod(shape)}:\n    pass"
            )
            assert isinstance(branch, ast.If)
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                loaded = load(self.coordinates(self.thread, shape))
                self.emit(f"{name} = {self.cast(loaded, output.dtype)}")
            self.cg.add_statement(branch)
            return Fragment(
                shape,
                output.dtype,
                lambda _coords: name,
                True,
                logical_domain=loaded_fragment.logical_domain,
            )
        result = self.materialize(loaded_fragment, batched=batched)
        self.pending_barrier = batched
        return result

    def logical_axis_extent(self, fake: torch.Tensor, dim: int, capacity: int) -> str:
        """Use logical bounds for arithmetic over a padded local allocation."""
        bid = self.env.resolve_block_id(fake.shape[dim])
        extent = str(capacity)
        if bid is not None:
            if bid in self.offsets:
                extent = f"min({extent}, ({self.bounds[bid]}) - ({self.offsets[bid]}))"
            elif self.env.block_sizes[bid].reduction:
                extent = f"min({extent}, {self.sym(self.env.block_sizes[bid].numel)})"
        return extent

    def scan_axis_extent(self, node: Node, source: Fragment, dim: int) -> str:
        """Intersect physical/tile bounds with the scan producer's proved domain."""
        from .gather_domains import loop_domain_facts

        fake = cast("torch.Tensor", node.meta["val"])
        capacity = source.shape[dim]
        extent = self.logical_axis_extent(fake, dim, capacity)
        logical = _fragment_logical_shape(
            self.env,
            cast("Node", node.args[1]),
            scalar_indexed_loads=True,
            tensor_indexed_loads=True,
            proven_domains=loop_domain_facts(self.env, self.graphs).shapes,
        )
        if logical is not None and len(logical) == len(source.shape):
            bound = logical[dim]
            if not self.env.known_nonnegative(sympy.Add(bound, -capacity)):
                extent = f"min({extent}, {self.sym(bound)})"
        return extent

    def topk(self, node: Node, values: dict[Node, object]) -> tuple[Fragment, Fragment]:
        """Complete last-axis sorting/selection in a uniform fragment owner.

        Each worker owns one logical leading coordinate and all its selected
        slots. Integer ranks preserve subnormals, zeros, NaNs and stable index
        ties without moving comparisons across shared-memory lifetimes.
        """
        arguments = ordered_selection_args(node)
        assert arguments is not None
        source_node, k, dim, largest, _ = arguments
        source = values[source_node]
        assert isinstance(source, Fragment)
        fake = cast("torch.Tensor", source_node.meta["val"])
        n = _static_selection_extent(self.env, source_node)
        if k is None:
            k = n
        bid = self.env.resolve_block_id(fake.shape[-1])
        # The configured capacity can include padding, but a rolling selection
        # would choose independent subsets instead of one complete top-k.
        if (
            n is None
            or k is None
            or dim not in (-1, fake.ndim - 1)
            or bid in self.offsets
            or source.shape[-1] < n
            or not 0 < k <= n
        ):
            raise exc.InvalidConfig(
                "fragment topk requires a complete static last axis"
            )
        from ..autotuner_heuristics.cute_fragment_topk_network import (
            network_topk_supported,
        )

        values_only64 = source.dtype == torch.int64 and selection_values_only(node)
        use_network = self.df.config.get(
            "cute_fragment_topk_network", False
        ) and network_topk_supported(source.dtype, n, k, values_only=values_only64)
        source = self.materialize(source)
        self.held.append(source)
        out_shape = (*source.shape[:-1], k)
        selected = self.allocate(Fragment(out_shape, source.dtype, lambda _: "0"))
        self.held.append(selected)
        # Keep tuple metadata for the proved-dead index result without reserving
        # shared storage. Every observed-index/default path keeps its old owner.
        if use_network and values_only64:
            indices = Fragment(out_shape, torch.int64, lambda _: "0", True)
        else:
            indices = self.allocate(Fragment(out_shape, torch.int64, lambda _: "0"))
        self.held.append(indices)
        float_rank = source.dtype in (torch.float16, torch.bfloat16, torch.float32)
        rank_type = "cutlass.Int32" if float_rank else self.dtype(source.dtype)
        better = ">" if largest else "<"
        worse = "<" if largest else ">"

        def select(outer: tuple[str, ...]) -> None:
            slot = self.df.new_var("fragment_topk_slot")
            col = self.df.new_var("fragment_topk_column")
            last_rank = self.df.new_var("fragment_topk_previous_rank")
            last_col = self.df.new_var("fragment_topk_previous_column")
            best_rank = self.df.new_var("fragment_topk_rank")
            best_col = self.df.new_var("fragment_topk_column_selected")
            rank = self.df.new_var("fragment_topk_candidate")
            found = self.df.new_var("fragment_topk_found")
            self.emit(f"{last_rank} = {rank_type}(0)")
            self.emit(f"{last_col} = cutlass.Int64(-1)")
            loop = cast(
                "ast.For", statement_from_string(f"for {slot} in range({k}):\n    pass")
            )
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{best_rank} = {rank_type}(0)")
                self.emit(f"{best_col} = cutlass.Int64(0)")
                self.emit(f"{found} = cutlass.Boolean(False)")
                scan = cast(
                    "ast.For",
                    statement_from_string(f"for {col} in range({n}):\n    pass"),
                )
                scan.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", scan.body)):
                    value = source.read((*outer, col))
                    if float_rank:
                        bits = self.df.new_var("fragment_topk_bits")
                        magnitude = self.df.new_var("fragment_topk_magnitude")
                        sign = self.df.new_var("fragment_topk_sign")
                        narrow = source.dtype != torch.float32
                        bit_type = "cutlass.Int16" if narrow else "cutlass.Int32"
                        mask = 32767 if narrow else 2147483647
                        infinity = (
                            0x7C00
                            if source.dtype == torch.float16
                            else (0x7F80 if narrow else 0x7F800000)
                        )
                        self.emit(
                            f"{bits} = cutlass.Int32({value}.bitcast({bit_type}))"
                        )
                        self.emit(f"{magnitude} = {bits} & cutlass.Int32({mask})")
                        self.emit(f"{sign} = {bits} >> cutlass.Int32(31)")
                        self.emit(f"{rank} = ({magnitude} ^ {sign}) - {sign}")
                        self.emit(
                            f"if {magnitude} > cutlass.Int32({infinity}):\n    {rank} = cutlass.Int32({mask})"
                        )
                    else:
                        self.emit(f"{rank} = {rank_type}({value})")
                    eligible = f"({slot} == 0 or {rank} {worse} {last_rank} or ({rank} == {last_rank} and {col} > {last_col}))"
                    preferred = f"(not {found} or {rank} {better} {best_rank})"
                    predicate = self.predicate((eligible, preferred))
                    self.emit(
                        f"if {predicate}:\n    {found} = cutlass.Boolean(True)\n    {best_rank} = {rank}\n    {best_col} = cutlass.Int64({col})"
                    )
                self.cg.add_statement(scan)
                self.emit(
                    f"{selected.read((*outer, slot))} = {source.read((*outer, best_col))}"
                )
                self.emit(f"{indices.read((*outer, slot))} = {best_col}")
                self.emit(f"{last_rank} = {best_rank}")
                self.emit(f"{last_col} = {best_col}")
            self.cg.add_statement(loop)

        if use_network:
            self.network_topk(
                source, selected, None if values_only64 else indices, n, k, largest
            )
        else:
            self.elements(source.shape[:-1], select)
        self.held.pop()
        self.held.pop()
        self.held.pop()

        # Last-axis capacity is exact. Logical leading tails retain their own
        # shape/offset ownership; no data-derived column owns an output row.
        def domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            return source.domain((*coords[:-1], "0"))

        return (
            dataclasses.replace(
                selected, logical_domain=domain, dependencies=(source,)
            ),
            dataclasses.replace(indices, logical_domain=domain, dependencies=(source,)),
        )

    def network_topk(
        self,
        source: Fragment,
        selected: Fragment,
        indices: Fragment | None,
        n: int,
        k: int,
        largest: bool,
    ) -> None:
        """A physical warp selects one complete leading coordinate.

        Shared source/output lifetimes and CTA barriers are the serial owner's.
        Only key loading and exact selection change. Every lane participates in
        each network, including logical/physical padding; cyclic output owners
        store each public slot once and gather the original typed payload.
        An Int64 value-only boundary instead carries the value itself, without
        another original-column field or a payload gather. Complementing signed
        keys reverses their complete order, including both Int64 extrema.
        """
        from .selection_lowering import emit_selection_network

        assert self.threads % 32 == 0
        assert indices is not None or source.dtype == torch.int64
        if (
            indices is None
            and self.df.config.get("cute_fragment_cta_topk", False)
            and source.dtype == torch.int64
        ):
            self.cta_int64_topk(source, selected, n, k, largest)
            return
        padded_n = 1 << (n - 1).bit_length()
        padded_k = 1 << (k - 1).bit_length()
        local_n = max(1, padded_n // 32)
        local_k = max(1, padded_k // 32)
        bits = max(1, (n - 1).bit_length())
        index_mask = (1 << bits) - 1

        def line(row: tuple[str, ...]) -> None:
            lane = self.df.new_var("fragment_topk_lane")
            keys = self.df.new_var("fragment_topk_keys")
            local = self.df.new_var("fragment_topk_local")
            col = self.df.new_var("fragment_topk_col")
            value = self.df.new_var("fragment_topk_value")
            rank = self.df.new_var("fragment_topk_rank")
            self.emit(f"{lane} = {self.thread} % 32")
            self.emit(f"{keys} = cute.make_rmem_tensor({local_n}, cutlass.Int64)")
            self.emit(f"{keys}.fill(cutlass.Int64(-9223372036854775808))")
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {local} in cutlass.range_constexpr({local_n}):\n    pass"
                ),
            )
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{col} = {local} * 32 + {lane}")
                branch = cast(
                    "ast.If", statement_from_string(f"if {col} < {n}:\n    pass")
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    self.emit(
                        f"{value} = {self.cast(source.read((*row, col)), source.dtype)}"
                    )
                    if indices is None:
                        self.emit(
                            f"{keys}[{local}] = {value if largest else f'~{value}'}"
                        )
                    elif source.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        narrow = source.dtype != torch.float32
                        mask = 0x7FFF if narrow else 0x7FFFFFFF
                        infinity = {
                            torch.float16: 0x7C00,
                            torch.bfloat16: 0x7F80,
                            torch.float32: 0x7F800000,
                        }[source.dtype]
                        word = self.df.new_var("fragment_topk_bits")
                        magnitude = self.df.new_var("fragment_topk_magnitude")
                        sign = self.df.new_var("fragment_topk_sign")
                        self.emit(
                            f"{word} = cutlass.Int32({value}.bitcast(cutlass.{'Int16' if narrow else 'Int32'}))"
                        )
                        self.emit(f"{magnitude} = {word} & cutlass.Int32({mask})")
                        self.emit(f"{sign} = {word} >> cutlass.Int32(31)")
                        self.emit(
                            f"{rank} = cutlass.Int64(({magnitude} ^ {sign}) - {sign})"
                        )
                        self.emit(
                            f"if {magnitude} > cutlass.Int32({infinity}):\n    {rank} = cutlass.Int64({mask})"
                        )
                    else:
                        self.emit(f"{rank} = cutlass.Int64({value})")
                    if indices is not None and not largest:
                        # Widen first: smallest Int32 is a valid finite key.
                        self.emit(f"{rank} = -{rank}")
                    if indices is not None:
                        self.emit(
                            f"{keys}[{local}] = ({rank} << cutlass.Int32({bits})) | cutlass.Int64({index_mask} - {col})"
                        )
                self.cg.add_statement(branch)
            self.cg.add_statement(loop)
            result = emit_selection_network(
                self.cg,
                keys=keys,
                dtype=torch.int64,
                groups=32,
                registers=local_n,
                k=padded_k,
                lane=lane,
            ).name
            output = self.df.new_var("fragment_topk_output")
            slot = self.df.new_var("fragment_topk_slot")
            column = self.df.new_var("fragment_topk_selected_column")
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {output} in cutlass.range_constexpr({local_k}):\n    pass"
                ),
            )
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{slot} = {output} * 32 + {lane}")
                branch = cast(
                    "ast.If", statement_from_string(f"if {slot} < {k}:\n    pass")
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    if indices is None:
                        value = f"{result}[{output}]"
                        self.emit(
                            f"{selected.read((*row, slot))} = {self.cast(value if largest else f'~{value}', torch.int64)}"
                        )
                    else:
                        self.emit(
                            f"{column} = cutlass.Int32({index_mask}) - cutlass.Int32({result}[{output}] & cutlass.Int64({index_mask}))"
                        )
                        # Resident views can still emit coordinate assignments.
                        # Resolve their read only after the selected column exists,
                        # inside the public-slot mask, never before this loop.
                        self.emit(
                            f"{selected.read((*row, slot))} = {self.cast(source.read((*row, column)), source.dtype)}"
                        )
                        self.emit(
                            f"{indices.read((*row, slot))} = cutlass.Int64({column})"
                        )
                self.cg.add_statement(branch)
            self.cg.add_statement(loop)

        self.elements(source.shape[:-1], line, threads_per_element=32)

    def cta_int64_topk(
        self,
        source: Fragment,
        selected: Fragment,
        n: int,
        k: int,
        largest: bool,
    ) -> None:
        """Sort complete Int64 values with bounded private slots and CTA peers.

        Every physical thread enters each shared publication and retirement
        barrier, including padded owners. Comparisons within a lane use a
        simultaneous private snapshot; exchanges within a warp use shuffles;
        exchanges between warps use one held shared scratch vector. Original
        indices are not observed on this path.
        """
        if not (0 < k <= n <= 1024 and self.threads & (self.threads - 1) == 0):
            raise exc.InvalidConfig("CTA selection requires a bounded Int64 fragment")
        padded = max(32, 1 << (n - 1).bit_length())
        slots = (padded + self.threads - 1) // self.threads
        keys = self.df.new_var("fragment_cta_sort_keys")
        temporary = self.df.new_var("fragment_cta_sort_next")
        self.emit(f"{keys} = cute.make_rmem_tensor(({slots},), cutlass.Int64)")
        self.emit(f"{temporary} = cute.make_rmem_tensor(({slots},), cutlass.Int64)")
        scratch = (
            self.allocate(Fragment((padded,), torch.int64, lambda _: "0"))
            if self.threads > 32 and padded > 32
            else None
        )
        self.held.append(scratch)
        leading = self.df.new_var("fragment_cta_sort_row")
        slot = self.df.new_var("fragment_cta_sort_slot")
        column = self.df.new_var("fragment_cta_sort_column")
        mine = self.df.new_var("fragment_cta_sort_mine")
        peer = self.df.new_var("fragment_cta_sort_peer")
        maximum = self.df.new_var("fragment_cta_sort_maximum")
        choose = self.df.new_var("fragment_cta_sort_choose")
        outer = cast(
            "ast.For",
            statement_from_string(
                f"for {leading} in range({math.prod(source.shape[:-1])}):\n    pass"
            ),
        )
        outer.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", outer.body)):
            coords = self.coordinates(leading, source.shape[:-1])
            fill = cast(
                "ast.For",
                statement_from_string(
                    f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                ),
            )
            fill.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", fill.body)):
                self.emit(f"{column} = {self.thread} + {slot} * {self.threads}")
                self.emit(f"{mine} = cutlass.Int64(-9223372036854775808)")
                valid = cast(
                    "ast.If", statement_from_string(f"if {column} < {n}:\n    pass")
                )
                valid.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                    value = self.cast(source.read((*coords, column)), torch.int64)
                    self.emit(f"{mine} = {value if largest else f'~({value})'}")
                self.cg.add_statement(valid)
                self.emit(f"{keys}[{slot}] = {mine}")
            self.cg.add_statement(fill)
            for level in range(padded.bit_length() - 1):
                span = 1 << (level + 1)
                for bit in range(level, -1, -1):
                    distance = 1 << bit
                    cross_warp = 32 <= distance < self.threads
                    if cross_warp:
                        assert scratch is not None
                        self.emit(
                            f"for {slot} in cutlass.range_constexpr({slots}):\n"
                            f"    {column} = {self.thread} + {slot} * {self.threads}\n"
                            f"    if {column} < {padded}:\n"
                            f"        {scratch.read((column,))} = {keys}[{slot}]"
                        )
                        self.synchronize()
                    update = cast(
                        "ast.For",
                        statement_from_string(
                            f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                        ),
                    )
                    update.body.clear()
                    with self.cg.set_statements(cast("list[ast.AST]", update.body)):
                        self.emit(f"{column} = {self.thread} + {slot} * {self.threads}")
                        self.emit(f"{mine} = {keys}[{slot}]")
                        valid = cast(
                            "ast.If",
                            statement_from_string(f"if {column} < {padded}:\n    pass"),
                        )
                        valid.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                            if distance >= self.threads:
                                expression = (
                                    f"{keys}[{slot} ^ {distance // self.threads}]"
                                )
                            elif distance < 32:
                                expression = (
                                    f"cute.arch.shuffle_sync_bfly({mine}, offset={distance}, "
                                    "mask=0xffffffff, mask_and_clamp=31)"
                                )
                            else:
                                assert scratch is not None
                                expression = scratch.read((f"{column} ^ {distance}",))
                            self.emit(f"{peer} = {expression}")
                            self.emit(
                                f"{maximum} = (({column} & {span}) == 0) == (({column} & {distance}) == 0)"
                            )
                            self.emit(
                                f"{choose} = ({mine} < {peer}) if {maximum} else ({mine} > {peer})"
                            )
                            self.emit(
                                f"{temporary}[{slot}] = {peer} if {choose} else {mine}"
                            )
                        valid.orelse = [
                            statement_from_string(f"{temporary}[{slot}] = {mine}")
                        ]
                        self.cg.add_statement(valid)
                    self.cg.add_statement(update)
                    if cross_warp:
                        # Every peer read completes before the next shared pass.
                        self.synchronize()
                    self.emit(
                        f"for {slot} in cutlass.range_constexpr({slots}):\n    {keys}[{slot}] = {temporary}[{slot}]"
                    )
            self.emit(
                f"for {slot} in cutlass.range_constexpr({slots}):\n"
                f"    {column} = {self.thread} + {slot} * {self.threads}\n"
                f"    if {column} < {k}:\n"
                f"        {selected.read((*coords, column))} = "
                f"{keys + '[' + slot + ']' if largest else '~' + keys + '[' + slot + ']'}"
            )
            self.synchronize()
        self.cg.add_statement(outer)
        self.held.pop()

    def topk_gather(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[0])]
        indices = values[cast("Node", node.args[2])]
        assert isinstance(source, Fragment) and isinstance(indices, Fragment)
        source = self.materialize(source)
        self.held.append(source)
        result = self.materialize(
            Fragment(
                indices.shape,
                source.dtype,
                lambda coords: source.read((*coords[:-1], indices.read(coords))),
                dependencies=(source, indices),
                logical_domain=indices.logical_domain,
            )
        )
        self.held.pop()
        return result

    def bounded_gather(self, node: Node, values: dict[Node, object]) -> Fragment:
        from .bounded_gather import prove_gather

        proof = prove_gather(self.env, node, graphs=self.graphs)
        assert proof is not None
        source_shape, index_shape = proof.source_shape, proof.index_shape
        source = values[cast("Node", node.args[0])]
        indices = values[cast("Node", node.args[2])]
        assert isinstance(source, Fragment) and isinstance(indices, Fragment)
        # Both original recipes remain live while either is materialized. In
        # particular an index recipe may read storage that source allocation
        # would otherwise recycle. Snapshot indices before gathering the source.
        self.held.extend((source, indices))
        indices = self.materialize(indices)
        self.held.append(indices)
        source = self.materialize(source)
        self.held.append(source)

        def source_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
            return (*coords[:-1], indices.read(coords))

        def output_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(
                f"0 <= ({coord}) and ({coord}) < ({self.sym(size)})"
                for coord, size in zip(coords, index_shape, strict=True)
            )

        def read(coords: tuple[str, ...]) -> str:
            location = source_coordinates(coords)
            valid = self.predicate(
                [
                    *output_domain(coords),
                    *source.domain(location),
                    *(
                        f"0 <= ({coord}) and ({coord}) < ({self.sym(size)})"
                        for coord, size in zip(location, source_shape, strict=True)
                    ),
                ]
            )
            # Padded output coordinates are not covered by the logical range
            # proof. Do not evaluate their source reads, even if a padded iota
            # makes an otherwise proved index exceed the source allocation.
            return f"({source.read(location)} if {valid} else {self.cast('0', source.dtype)})"

        result = self.materialize(
            Fragment(
                indices.shape,
                source.dtype,
                read,
                dependencies=(source, indices),
                # Once copied, the result domain must not retain a value read
                # from the index/source allocations that can now be recycled.
                logical_domain=output_domain,
            )
        )
        self.held.pop()
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return result

    def dot(self, node: Node, values: dict[Node, object]) -> Fragment:
        lhs, rhs = (values[cast("Node", x)] for x in node.args[:2])
        assert isinstance(lhs, Fragment) and isinstance(rhs, Fragment)
        acc = values[node.args[2]] if isinstance(node.args[2], Node) else None
        assert acc is None or isinstance(acc, Fragment)
        out = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(out.shape)
        out_dtype = cast("torch.dtype | None", node.args[3]) or _compute_out_dtype(
            lhs.dtype, rhs.dtype, acc.dtype if acc is not None else None
        )
        # A narrow float accumulator still sums in fp32 and rounds once, as
        # the other hl.dot lowerings do.
        compute_dtype = (
            torch.float32
            if _needs_f32_accumulator(lhs.dtype, rhs.dtype)
            or (out_dtype.is_floating_point and out_dtype.itemsize < 4)
            else out_dtype
        )
        left_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
        mma_contraction = self.tf32_mma_contraction(
            node, lhs, rhs, acc, shape, left_fake, compute_dtype
        )
        # Operand recipes must finish before the contraction begins: the same
        # input can have different coordinates in its two operand roles.
        if mma_contraction is None:
            lhs = self.materialize(lhs)
            self.held.append(lhs)
            rhs = self.materialize(rhs)
            self.held.append(rhs)
        else:
            # An operand every lane fragment reads exactly once (whole tiles,
            # one tile along the other output axis) whose recipe rests on
            # staged data is read in place by the mma lanes: no copy, no
            # barrier, no second buffer; its leaves' padded rows keep the
            # reads conflict-free.  Otherwise rows are padded to an odd
            # multiple of 4 (A) / 8 (B) banks so the lane fragments of one
            # mma read 32 distinct banks.
            whole = (
                shape[-2] % 16 == 0
                and shape[-1] % 8 == 0
                and mma_contraction % 8 == 0
                and mma_contraction == lhs.shape[-1]
            )

            # A recipe that reads no memory at all (an ``hl.arange`` mask, a
            # constant) is evaluated in place however often it is re-read: a
            # compare or a literal per operand word costs less than a staging
            # phase, its barrier and its buffer.  So is a bare cast / view of
            # one staged tile whose rows already spread the lane reads over
            # the banks: the padded copy would only repeat the tile.
            def in_place(operand: Fragment, role: str, read_once: bool) -> bool:
                if self.pure_recipe(operand):
                    return True
                if not self.staged_recipe(operand):
                    return False
                if whole and read_once:
                    return True
                return (
                    operand.cheap
                    and operand.physical_row is not None
                    and _mma_operand_conflict_free(operand.physical_row, role)
                )

            if not in_place(lhs, "a", shape[-1] <= 8):
                lhs = self.materialize_padded(lhs, _mma_row_pad(lhs.shape[-1], 4))
            self.held.append(lhs)
            if not in_place(rhs, "b", shape[-2] <= 16):
                rhs = self.materialize_padded(rhs, _mma_row_pad(rhs.shape[-1], 8))
            self.held.append(rhs)
        if mma_contraction is not None:
            result = self.emit_tf32_mma(
                lhs,
                rhs,
                acc,
                shape,
                mma_contraction,
                out_dtype,
                sink=self.direct_store_sink(node, values, shape),
            )
            self.held.pop()
            self.held.pop()
            return result
        result = self.allocate(Fragment(shape, out_dtype, lambda _: "0"))

        contraction = self.logical_axis_extent(left_fake, -1, lhs.shape[-1])

        def contract(coords: tuple[str, ...]) -> None:
            total = self.df.new_var("fragment_dot")
            k = self.df.new_var("fragment_k")
            initial = acc.broadcast(coords) if acc is not None else "0"
            self.emit(f"{total} = {self.cast(initial, compute_dtype)}")
            # All operands are in shared memory, so this serial per-output
            # contraction has ordinary IEEE FP32 multiplication and addition.
            left = lhs.broadcast((*coords[:-2], coords[-2], k))
            right = rhs.broadcast((*coords[:-2], k, coords[-1]))
            self.emit(
                f"for {k} in range({contraction}):\n    {total} = {total} + {self.cast(left, compute_dtype)} * {self.cast(right, compute_dtype)}"
            )
            self.emit(f"{result.read(coords)} = {self.cast(total, out_dtype)}")

        self.elements(shape, contract)
        self.held.pop()
        self.held.pop()
        return result

    @staticmethod
    def pure_recipe(value: Fragment) -> bool:
        """Whether *value*'s recipe reads no memory: every leaf is a pure
        coordinate function (``hl.arange``, a tile index, a constant) or a
        scalar, so a lane evaluates it in a few ALU instructions and it needs
        no staging phase, no barrier and no buffer."""
        pending = [value]
        seen: set[int] = set()
        while pending:
            fragment = pending.pop()
            if id(fragment) in seen:
                continue
            seen.add(id(fragment))
            if fragment.resident or fragment.snapshot is not None:
                return False
            if not fragment.dependencies:
                if fragment.shape and not fragment.pure:
                    return False
                continue
            pending.extend(fragment.dependencies)
        return True

    @staticmethod
    def staged_recipe(value: Fragment) -> bool:
        """Whether every leaf of *value*'s recipe is resident (a staged
        buffer or a scalar) or pure (an ``hl.arange`` row mask, a tile
        index), so any lane may evaluate it at any coordinate."""
        pending = [value]
        seen: set[int] = set()
        while pending:
            fragment = pending.pop()
            if id(fragment) in seen:
                continue
            seen.add(id(fragment))
            if fragment.resident:
                if fragment.snapshot is not None:
                    return False
                continue
            if not fragment.dependencies and fragment.shape and not fragment.pure:
                return False
            pending.extend(fragment.dependencies)
        return True

    def static_contraction(self, fake: torch.Tensor, capacity: int) -> int | None:
        """The contraction length when it is a compile-time constant.

        ``logical_axis_extent`` intersects the staged capacity with the live
        bound of a tiled axis: a per-CTA boundary (``offsets``) is dynamic and
        declines, a static reduction extent clips the capacity.
        """
        bid = self.env.resolve_block_id(fake.shape[-1])
        if bid is None:
            return capacity
        if bid in self.offsets:
            return None
        if self.env.block_sizes[bid].reduction:
            extent = self.static_extent(self.env.block_sizes[bid].numel)
            return None if extent is None else min(capacity, extent)
        return capacity

    def tf32_mma_contraction(
        self,
        node: Node,
        lhs: Fragment,
        rhs: Fragment,
        acc: Fragment | None,
        shape: tuple[int, ...],
        left_fake: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> int | None:
        """The static K of a dot ``cute_fp32_dot="tf32_mma"`` lowers through
        ``mma.sync``; None keeps the serial FMA contraction (knob off, half or
        wide operands, a dynamic contraction bound, a one-warp-short CTA)."""
        from ..autotuner_heuristics.cute_fp32_dot import fragment_tf32_dot_supported

        if self.df.config.get("cute_fp32_dot", "simt") != "tf32_mma":
            return None
        if (
            compute_dtype != torch.float32
            or self.threads % 32 != 0
            or not fragment_tf32_dot_supported(node, self.env)
            or (acc is not None and acc.dtype != torch.float32)
            or len(lhs.shape) < 2
            or len(rhs.shape) < 2
            or len(shape) < 2
        ):
            return None
        contraction = self.static_contraction(left_fake, lhs.shape[-1])
        if contraction is None or contraction <= 0:
            return None
        return contraction

    def host_storages_disjoint(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        """Whether two host tensors are proved to view disjoint storage: a
        different fake storage and either one freshly allocated by a wrapper
        factory on this call (an in-kernel ``torch.empty``), or the bound
        dispatch key's alias matrix.  Two inputs with different fake storages
        alone prove nothing: a caller may pass overlapping views."""
        if left.untyped_storage() == right.untyped_storage():
            return False
        if self.env.tensor_storage_is_compiler_allocated(
            left
        ) or self.env.tensor_storage_is_compiler_allocated(right):
            return True
        return runtime_tensors_are_proven_disjoint(self.env, left, right)

    def direct_store_in_order(self, node: Node, user: Node) -> bool:
        """Whether the lanes may write ``user``'s window at ``node``'s position.

        The direct store moves the global write from the store node up to the
        dot, so every node between them in graph order must be free of host
        memory effects on the window: lazy recipes, views, coordinate
        functions, scalars, pointwise and reduction evaluation over
        fragments, another dot, a scan or a local allocation pass; a host
        load, store or atomic passes only when its tensor is proved disjoint
        from the window; a nested loop or branch, inline assembly, a gather
        or selection and any other opaque lowering keep the staged tile."""
        target_node = user.args[0]
        assert isinstance(target_node, Node)
        window = target_node.meta["val"]
        assert isinstance(window, torch.Tensor)
        current = node.next
        while current is not user:
            if current.op != "call_function":
                return False
            target = current.target
            if (
                target in (memory_ops.load, memory_ops.store)
                or target in atomic_ops.ATOMIC_OPS
            ):
                other = current.args[0]
                if (
                    not isinstance(other, Node)
                    or other.target is not _tracing_ops._host_tensor
                ):
                    return False
                fake = other.meta["val"]
                if not isinstance(
                    fake, torch.Tensor
                ) or not self.host_storages_disjoint(window, fake):
                    return False
            elif target not in self._HOST_MEMORY_FREE_TARGETS and not isinstance(
                current.meta.get("lowering"), (PointwiseLowering, ReductionLowering)
            ):
                return False
            current = current.next
        return True

    def direct_store_sink(
        self, node: Node, values: dict[Node, object], shape: tuple[int, ...]
    ) -> DirectStoreSink | None:
        """The host store a tf32 mma may write from its lanes: the dot's only
        user, storing the dot itself (no cast or view between them) into a
        host tensor window of the dot's shape, in the same graph, with every
        other input of the store already lowered and no host memory effect
        on the window between the dot and the store in graph order
        (``direct_store_in_order``).  Anything else -- another reader, an
        extra mask, a region or producer-cache walk, a warp-result or
        register-owned chain, a recorder -- keeps the staged result."""
        if (
            len(node.users) != 1
            or self.active_region is not None
            or node in self.cache_nodes
            or self.warp_producer_recorder is not None
        ):
            return None
        (user,) = node.users
        if (
            user.op != "call_function"
            or user.target is not memory_ops.store
            or user.graph is not node.graph
            or user.args[2] is not node
            or user.args[3] is not None
            or user in self.direct_stores
            or user in self.warp_result_nodes
            or user in self.local_register_nodes
            or not self.direct_store_in_order(node, user)
        ):
            return None
        for source in user.all_input_nodes:
            if source is node or source in values:
                continue
            # The store's host tensor (and a size symbol) may follow the dot
            # in graph order; both lower to a value without emitting anything.
            if source.target is _tracing_ops._host_tensor:
                fake = source.meta["val"]
                assert isinstance(fake, torch.Tensor)
                values[source] = HostTensor(
                    self.df.tensor_arg(
                        fake, prefer_name=cast("str", source.args[0])
                    ).name,
                    fake,
                )
            elif source.target in (
                _tracing_ops._get_symnode,
                torch.ops.aten.sym_size.int,
            ):
                values[source] = source.meta.get("val")
            else:
                return None
        if not isinstance(values.get(cast("Node", user.args[0])), HostTensor):
            return None
        sink = DirectStoreSink(user)
        self.memory(user, values, True, direct=sink)
        if sink.address is None or sink.shape != shape:
            return None
        return sink

    def emit_direct_lane_stores(
        self,
        sink: DirectStoreSink,
        batch: tuple[str, ...],
        slots: tuple[tuple[str, str], ...],
        accs: list[str],
        rows: int,
        cols: int,
        out_dtype: torch.dtype,
    ) -> None:
        """Write one lane's four ``mma.sync`` results through the sink's
        window: the two adjacent columns of a row leave as one aligned vector
        store when the window is a proved packet walk, else as two scalar
        stores.  Each value takes the dot's result dtype and then the
        tensor's, exactly as the staged tile and its store did."""
        assert sink.address is not None and sink.dtype is not None
        itemsize = sink.dtype.itemsize
        scalar = self.dtype(sink.dtype)
        paired = sink.lanes is not None and sink.lanes % 2 == 0 and cols % 2 == 0

        def value(name: str) -> str:
            return self.cast(self.cast(name, out_dtype), sink.dtype)

        for first in (0, 2):
            (r, c0), (_, c1) = slots[first], slots[first + 1]
            if not paired:
                for name, c in zip(accs[first : first + 2], (c0, c1), strict=True):
                    pointer, mask = sink.address((*batch, r, c))
                    guard = self.predicate(
                        [mask, *self.in_bounds(r, rows, 16), *self.in_bounds(c, cols)]
                    )
                    self.emit(f"if {guard}:\n    {pointer}.store({value(name)})")
                continue
            # ``c0`` is even and the packet proof makes every even column of
            # the window ``2 * itemsize``-aligned; ``cols`` is even, so the
            # pair is in bounds together.
            pointer, mask = sink.address((*batch, r, c0))
            guard = self.predicate(
                [mask, *self.in_bounds(r, rows, 16), *self.in_bounds(c0, cols)]
            )
            packet = self.df.new_var("fragment_mma_packet")
            self.emit(
                f"{packet} = cute.make_rmem_tensor(cute.make_layout((2,)), {scalar})"
            )
            self.emit(f"{packet}[0] = {value(accs[first])}")
            self.emit(f"{packet}[1] = {value(accs[first + 1])}")
            atom = (
                "cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                f"{scalar}, num_bits_per_copy={16 * itemsize})"
            )
            self.emit(
                f"if {guard}:\n"
                f"    cute.copy({atom}, {packet}, cute.make_tensor({pointer}.align({2 * itemsize}), cute.make_layout((2,))))"
            )

    def emit_tf32_mma(
        self,
        lhs: Fragment,
        rhs: Fragment,
        acc: Fragment | None,
        shape: tuple[int, ...],
        contraction: int,
        out_dtype: torch.dtype,
        sink: DirectStoreSink | None = None,
    ) -> Fragment:
        """Contract through ``mma.sync.aligned.m16n8k8`` tf32 (``cute_fp32_dot``).

        Each warp owns whole 16 x 8 output tiles of the (batched) product and
        walks K in steps of 8: lane ``(g, c) = (lane // 4, lane % 4)`` reads
        A rows ``g``/``g + 8`` at k ``c``/``c + 4`` and B column ``g`` at the
        same k straight from the staged fp32 operand buffers (six smem reads
        per 1024 MACs instead of two per MAC; the padded rows keep the 32
        reads of one instruction in 32 banks), feeds their bit patterns to the
        tensor core, which truncates them to tf32, and writes its four fp32
        results ``(g, 2c)``, ``(g, 2c + 1)``, ``(g + 8, 2c)``, ``(g + 8, 2c + 1)``.
        Partial tiles (M % 16, N % 8, K % 8) read zeros past the operand and
        skip the writes past the result, so every qualifying dot of the root
        takes the mode.

        ``cute_fragment_split_k``: when the dot has fewer tiles than the CTA
        has warps, the idle warps take a share of K: ``splits`` warps walk
        disjoint step ranges of one tile (the accumulator joins split 0) and
        write fp32 partials; the result reads their sum.  Returns the result
        fragment (a staged tile, or the partial-summing view).

        With a *sink* (the dot's only reader is a host store of this shape)
        and no K split, the lanes write their results straight into the
        host window: no staged tile, no barrier before the store walk and no
        barrier after it.  The store node then lowers to nothing and the
        returned fragment must never be read.
        """
        rows, cols = shape[-2], shape[-1]
        batch_shape = shape[:-2]
        tiles_m = -(-rows // 16)
        tiles_n = -(-cols // 8)
        tiles = math.prod(batch_shape) * tiles_m * tiles_n
        warps = self.threads // 32
        steps = -(-contraction // 8)
        splits = 1
        if self.split_k and tiles < warps and steps > 1:
            splits = max(
                d for d in range(1, min(warps // tiles, steps) + 1) if steps % d == 0
            )
        direct = sink is not None and splits == 1
        # The lanes read staged operands: batched loads must be published.
        self.flush_pending_barrier()
        self.emit_fragment_mma_preamble()
        rt = _FRAGMENT_MMA_RUNTIME
        if splits > 1:
            partials = self.allocate(
                Fragment((splits, *shape), torch.float32, lambda _: "0")
            )
            result = partials
        elif direct:
            result = Fragment(shape, out_dtype, _consumed_by_direct_store)
        else:
            result = self.allocate(Fragment(shape, out_dtype, lambda _: "0"))
        # ``cute_fragment_mma_rows``: a warp owns whole 16-row tile rows of
        # the product and evaluates each row's A words once per K step for
        # all of the row's column tiles (the words do not depend on the
        # column).  The same mma sequence per output tile: bitwise identical
        # to the tile walk.  A K split keeps the (tile, split) unit walk.
        row_walk = self.mma_rows and splits == 1
        tile = self.df.new_var("fragment_mma_tile")
        # Without a split the warp loop walks tiles directly (the established
        # render); with one it walks (tile, split) units.
        unit = self.df.new_var("fragment_mma_unit") if splits > 1 else tile
        split = self.df.new_var("fragment_mma_split") if splits > 1 else ""
        lane = self.df.new_var("fragment_mma_lane")
        group = self.df.new_var("fragment_mma_group")
        quad = self.df.new_var("fragment_mma_quad")
        row = self.df.new_var("fragment_mma_row")
        col = self.df.new_var("fragment_mma_col")
        self.emit(f"{lane} = {self.thread} % 32")
        self.emit(f"{group} = {lane} // 4")
        self.emit(f"{quad} = {lane} % 4")
        f32 = self.dtype(torch.float32)
        row8 = f"{row} + 8"
        # Result slots: (row, 2 quad), (row, 2 quad + 1) and the same of
        # row + 8; a slot's column is relative to the tile's first column.
        slots = (
            (row, f"2 * {quad}"),
            (row, f"2 * {quad} + 1"),
            (row8, f"2 * {quad}"),
            (row8, f"2 * {quad} + 1"),
        )

        def init_accumulators(
            accs: list[str], batch: tuple[str, ...], col_base: str
        ) -> None:
            for name, (r, c) in zip(accs, slots, strict=True):
                c_abs = f"{col_base} + {c}"
                if acc is None:
                    self.emit(f"{name} = {f32}(0)")
                    continue
                self.guarded_assign(
                    name,
                    self.cast(acc.broadcast((*batch, r, c_abs)), torch.float32),
                    f"{f32}(0)",
                    self.in_bounds(r, rows, 16)
                    + self.in_bounds(c_abs, cols)
                    + ([f"{split} == 0"] if splits > 1 else []),
                )

        def operand(
            fragment: Fragment,
            coords: tuple[str, str],
            bounds: list[str],
            batch: tuple[str, ...],
        ) -> str:
            name = self.df.new_var("fragment_mma_operand")
            self.guarded_assign(
                name,
                self.cast(fragment.broadcast((*batch, *coords)), torch.float32),
                f"{f32}(0)",
                bounds,
            )
            return f"{rt}.f32_bits({name})"

        def a_words(k0: str, k1: str, batch: tuple[str, ...]) -> list[str]:
            return [
                operand(
                    lhs,
                    (r, k),
                    self.in_bounds(r, rows, 16) + self.in_bounds(k, contraction),
                    batch,
                )
                for r, k in ((row, k0), (row8, k0), (row, k1), (row8, k1))
            ]

        def b_words(k0: str, k1: str, batch: tuple[str, ...]) -> list[str]:
            return [
                operand(
                    rhs,
                    (k, col),
                    self.in_bounds(k, contraction) + self.in_bounds(col, cols),
                    batch,
                )
                for k in (k0, k1)
            ]

        def emit_stores(accs: list[str], batch: tuple[str, ...], col_base: str) -> None:
            if direct:
                assert sink is not None
                self.emit_direct_lane_stores(
                    sink,
                    batch,
                    tuple((r, f"{col_base} + {c}") for r, c in slots),
                    accs,
                    rows,
                    cols,
                    out_dtype,
                )
                return
            for name, (r, c) in zip(accs, slots, strict=True):
                c_abs = f"{col_base} + {c}"
                if splits > 1:
                    store = f"{result.read((split, *batch, r, c_abs))} = {name}"
                else:
                    store = f"{result.read((*batch, r, c_abs))} = {self.cast(name, out_dtype)}"
                bounds = self.in_bounds(r, rows, 16) + self.in_bounds(c_abs, cols)
                if bounds:
                    store = f"if {' and '.join(bounds)}:\n    {store}"
                self.emit(store)

        if row_walk:
            tile_rows = math.prod(batch_shape) * tiles_m
            tile_row = self.df.new_var("fragment_mma_tile_row")
            col_tile = self.df.new_var("fragment_mma_col_tile")
            loop = statement_from_string(
                f"for {tile_row} in range({self.thread} // 32, {tile_rows}, {warps}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                batch = self.coordinate_locals(
                    self.coordinates(f"({tile_row}) // {tiles_m}", batch_shape)
                )
                self.emit(f"{row} = ({tile_row}) % {tiles_m} * 16 + {group}")
                ks = [
                    (f"{index * 8} + {quad}", f"{index * 8 + 4} + {quad}")
                    for index in range(steps)
                ]
                hoisted = [a_words(k0, k1, batch) for k0, k1 in ks]
                n_loop = statement_from_string(
                    f"for {col_tile} in range({tiles_n}):\n    pass"
                )
                assert isinstance(n_loop, ast.For)
                n_loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", n_loop.body)):
                    col_base = f"({col_tile}) * 8"
                    self.emit(f"{col} = {col_base} + {group}")
                    accs = [self.df.new_var("fragment_mma_c") for _ in slots]
                    init_accumulators(accs, batch, col_base)
                    for (k0, k1), words in zip(ks, hoisted, strict=True):
                        words = [*words, *b_words(k0, k1, batch)]
                        self.emit(
                            f"{', '.join(accs)} = {rt}.mma_tf32({', '.join([*words, *accs])})"
                        )
                    emit_stores(accs, batch, col_base)
                self.cg.add_statement(n_loop)
            self.cg.add_statement(loop)
        else:
            loop = statement_from_string(
                f"for {unit} in range({self.thread} // 32, {tiles * splits}, {warps}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                if splits > 1:
                    self.emit(f"{tile} = {unit} // {splits}")
                    self.emit(f"{split} = {unit} % {splits}")
                batch = self.coordinate_locals(
                    self.coordinates(f"({tile}) // {tiles_m * tiles_n}", batch_shape)
                )
                self.emit(f"{row} = ({tile}) // {tiles_n} % {tiles_m} * 16 + {group}")
                col_base = f"({tile}) % {tiles_n} * 8"
                self.emit(f"{col} = {col_base} + {group}")
                accs = [self.df.new_var("fragment_mma_c") for _ in slots]
                init_accumulators(accs, batch, col_base)
                step = self.df.new_var("fragment_mma_step")
                k_loop = statement_from_string(
                    f"for {step} in cutlass.range_constexpr({steps // splits}):\n    pass"
                )
                assert isinstance(k_loop, ast.For)
                k_loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", k_loop.body)):
                    base_step = (
                        f"({split} * {steps // splits} + {step})"
                        if splits > 1
                        else step
                    )
                    k0 = f"{base_step} * 8 + {quad}"
                    k1 = f"{base_step} * 8 + {quad} + 4"
                    words = [*a_words(k0, k1, batch), *b_words(k0, k1, batch)]
                    self.emit(
                        f"{', '.join(accs)} = {rt}.mma_tf32({', '.join([*words, *accs])})"
                    )
                self.cg.add_statement(k_loop)
                emit_stores(accs, batch, col_base)
            self.cg.add_statement(loop)
        # The lanes read staged operands a later phase may recycle.
        self.synchronize()
        if direct:
            assert sink is not None
            self.direct_stores.add(sink.node)
        if splits == 1:
            return result
        partials = result

        def summed(coords: tuple[str, ...]) -> str:
            terms = " + ".join(
                partials.read((str(index), *coords)) for index in range(splits)
            )
            return self.cast(f"({terms})", out_dtype)

        return Fragment(
            shape,
            out_dtype,
            summed,
            True,
            (partials,),
            partials.storage,
        )

    @staticmethod
    def in_bounds(index: str, extent: int, tile: int = 8) -> list[str]:
        """The predicate keeping *index* inside an axis the mma tiles may
        overhang (16 rows, 8 columns, 8 k); empty when *tile* divides *extent*."""
        return [f"({index}) < {extent}"] if extent % tile else []

    def guarded_assign(
        self, name: str, value: str, fallback: str, bounds: list[str]
    ) -> None:
        if not bounds:
            self.emit(f"{name} = {value}")
            return
        self.emit(f"{name} = {fallback}")
        self.emit(f"if {' and '.join(bounds)}:\n    {name} = {value}")

    def emit_fragment_mma_preamble(self) -> None:
        """Record that the ``mma.sync`` helpers' import is needed; the root's
        commit emits it once per generated module (``emit_mma_preamble``)."""
        self.wants_mma_preamble = True

    def emit_mma_preamble(self) -> None:
        """Import the ``mma.sync`` helpers once per generated module."""
        if not self.wants_mma_preamble or getattr(
            self.cg, "_helion_fragment_mma_module_emitted", False
        ):
            return
        self.cg._helion_fragment_mma_module_emitted = True  # type: ignore[attr-defined]
        for stmt in ast.parse(_FRAGMENT_MMA_PREAMBLE).body:
            self.cg.module_statements.append(stmt)

    def pointwise(self, node: Node, values: dict[Node, object]) -> Fragment:
        lowering = cast("PointwiseLowering | ReductionLowering", node.meta["lowering"])
        inputs: list[Node] = []
        map_arg(
            (node.args, {**node.kwargs, "_extra_deps": None}),
            lambda x: inputs.append(x),
        )
        lookup: dict[str, tuple[Node, Fragment]] = {}
        for name, source in zip(lowering.input_names, inputs, strict=True):
            value = values[source]
            if not isinstance(value, Fragment):
                scalar = self.scalar(value)
                value = Fragment((), torch.float32, lambda _, scalar=scalar: scalar)
            lookup[name] = (source, value)
        fake = cast("torch.Tensor", node.meta["val"])
        ranges = self.shape(lowering.buffer.data.ranges)
        result_shape = self.shape(fake.shape)
        # Inductor's iteration ranges usually differ from the fake shape only
        # by singleton dimensions: pass the live coordinates through instead
        # of a flatten/unflatten round trip the compiler cannot fold.
        range_live = [i for i, size in enumerate(ranges) if size != 1]
        result_live = [i for i, size in enumerate(result_shape) if size != 1]
        structural_ranges = [ranges[i] for i in range_live] == [
            result_shape[i] for i in result_live
        ]

        def element(
            coords: tuple[str, ...], reduction_coords: tuple[str, ...] = ()
        ) -> str:
            if ranges != result_shape:
                if structural_ranges:
                    mapped = ["0"] * len(ranges)
                    for range_dim, result_dim in zip(
                        range_live, result_live, strict=True
                    ):
                        mapped[range_dim] = coords[result_dim]
                    coords = tuple(mapped)
                else:
                    coords = self.coordinate_locals(
                        self.coordinates(self.flatten(coords, result_shape), ranges)
                    )
            prior = self.sym_indices
            self.sym_indices = {
                sympy.Symbol(f"i{i}"): coord
                for i, coord in enumerate((*coords, *reduction_coords))
            }
            try:
                with (
                    node.meta["location"],
                    V.set_current_node(node),
                    install_inductor_kernel_handlers(self.cg, {}),
                    V.set_ops_handler(FragmentOps(self, lookup)),
                ):
                    indices = [
                        sympy.Symbol(f"i{i}")
                        for i in range(len(lowering.buffer.data.ranges))
                    ]
                    if isinstance(lowering, ReductionLowering):
                        reduction_indices = [
                            sympy.Symbol(f"i{i + len(indices)}")
                            for i in range(len(reduction_coords))
                        ]
                        return str(
                            lowering.buffer.data.inner_fn(indices, reduction_indices)
                        )
                    return str(lowering.buffer.data.inner_fn(indices))
            finally:
                self.sym_indices = prior

        dependencies = tuple(value for _, value in lookup.values())
        if isinstance(lowering, ReductionLowering):
            from ..autotuner_heuristics.cute_fragment_reduction import (
                fragment_warp_reduction_supported,
            )

            assert isinstance(lowering.buffer.data, Reduction)
            reduction_shape = self.shape(lowering.buffer.data.reduction_ranges)
            indexed = lowering.reduction_type in ("argmin", "argmax")
            value_dtype = lowering.buffer.data.src_dtype if indexed else fake.dtype
            compute_dtype = (
                torch.float32
                if value_dtype in (torch.float16, torch.bfloat16)
                else value_dtype
            )
            warp = self.df.config.get(
                "cute_fragment_reduction", "serial"
            ) == "warp" and fragment_warp_reduction_supported(node)

            def combine(left: str, right: str) -> str:
                expression = self.env.backend.reduction_combine_expr(
                    lowering.reduction_type, left, right, compute_dtype
                )
                if (
                    lowering.reduction_type in ("min", "max")
                    and compute_dtype.is_floating_point
                ):
                    # A parallel fold must preserve NaNs from either operand,
                    # including a NaN encountered in a preceding local tile.
                    expression = f"{left} if {left} != {left} else ({expression})"
                return expression

            if node in self.scan_reductions:
                plan = self.scan_reductions[node]
                result = self.allocate(
                    Fragment(result_shape, fake.dtype, lambda _: "0")
                )
                if not self.scan_pending:
                    self.held.append([])
                cast("list[object]", self.held[-1]).extend([result, *dependencies])
                self.scan_pending.append(
                    (result, element, compute_dtype, lowering.reduction_type)
                )
                if node is plan.reductions[-1]:
                    self.completed_scan_reductions(reduction_shape)
                    self.scan_pending.clear()
                    self.held.pop()
                return result

            if node in self.snapshot_reductions:
                if (
                    self.shape(fake.shape) != ()
                    or len(reduction_shape) != 1
                    or reduction_shape not in self.snapshot_shapes
                    or compute_dtype not in (torch.int32, torch.int64)
                    or lowering.reduction_type not in ("sum", "min", "max")
                ):
                    raise exc.InvalidConfig(
                        "register snapshot requires an exact scalar integer reduction"
                    )
                total = self.df.new_var("fragment_snapshot_reduce")
                identity = self.cast(
                    self.scalar(
                        Reduction.default_accumulator(
                            lowering.reduction_type, compute_dtype
                        )
                    ),
                    compute_dtype,
                )
                self.emit(f"{total} = {identity}")

                def partial(coords: tuple[str, ...]) -> None:
                    value = element((), coords)
                    self.emit(
                        f"{total} = {self.cast(combine(total, value), compute_dtype)}"
                    )

                self.elements(reduction_shape, partial, synchronize=False)
                peer = self.df.new_var("fragment_snapshot_peer")
                for distance in (16, 8, 4, 2, 1):
                    self.emit(
                        f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                    )
                    self.emit(
                        f"{total} = {self.cast(combine(total, peer), compute_dtype)}"
                    )
                scratch = self.allocate(
                    Fragment((self.threads // 32,), compute_dtype, lambda _: "0")
                )
                self.held.append(scratch)
                self.emit(
                    f"if {self.thread} % 32 == 0:\n    {scratch.read((f'{self.thread} // 32',))} = {total}"
                )
                self.synchronize()
                result = self.allocate(Fragment((), fake.dtype, lambda _: "0"))
                branch = cast(
                    "ast.If", statement_from_string(f"if {self.thread} < 32:\n    pass")
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    self.emit(f"{total} = {identity}")
                    self.emit(
                        f"if {self.thread} < {self.threads // 32}:\n    {total} = {scratch.read((self.thread,))}"
                    )
                    for distance in (16, 8, 4, 2, 1):
                        self.emit(
                            f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                        )
                        self.emit(
                            f"{total} = {self.cast(combine(total, peer), compute_dtype)}"
                        )
                    self.emit(
                        f"if {self.thread} == 0:\n    {result.read(())} = {self.cast(total, fake.dtype)}"
                    )
                self.cg.add_statement(branch)
                self.synchronize()
                self.held.pop()
                return result

            if self.df.config.get("cute_fragment_reduction", "serial") in (
                "cta",
                "cta_unroll",
            ) and fragment_warp_reduction_supported(node):
                return self.cta_reduction(
                    result_shape,
                    fake.dtype,
                    reduction_shape,
                    compute_dtype,
                    lowering.reduction_type,
                    element,
                    combine,
                    dependencies,
                )

            def reduce(coords: tuple[str, ...]) -> str:
                reduction_type = lowering.reduction_type
                total = self.df.new_var(f"fragment_{reduction_type}")
                index = self.df.new_var("fragment_reduce_index")
                identity = (
                    0
                    if indexed
                    else Reduction.default_accumulator(reduction_type, compute_dtype)
                )
                self.emit(
                    f"{total} = {self.cast(self.scalar(identity), compute_dtype)}"
                )
                selected = self.df.new_var("fragment_selected_index") if indexed else ""
                if indexed:
                    self.emit(f"{selected} = {self.cast('0', fake.dtype)}")
                reduction_range = str(math.prod(reduction_shape))
                if warp:
                    reduction_range = f"{self.thread} % 32, {reduction_range}, 32"
                loop = statement_from_string(
                    f"for {index} in range({reduction_range}):\n    pass"
                )
                # The fragment owner retains full producer coordinates and
                # carries this accumulator across configured reduction tiles.
                # Generic graph rolling would instead capture a full fragment
                # inside a tile-local graph, losing both its coordinate window
                # and its implicit reduction carry.
                tile = self.df.config.reduction_loops
                block = lowering.block_index
                tile = self.env.config_spec.reduction_loops.config_get(
                    tile, block, None
                )
                tiled_loop: ast.For | None = None
                if tile is not None:
                    tile = self.extent(tile)
                    tile_begin = self.df.new_var("fragment_reduce_tile")
                    tiled_loop = cast(
                        "ast.For",
                        statement_from_string(
                            f"for {tile_begin} in range(0, {math.prod(reduction_shape)}, {tile}):\n    pass"
                        ),
                    )
                    assert isinstance(tiled_loop, ast.For)
                    start = (
                        f"{tile_begin} + ({self.thread} % 32)" if warp else tile_begin
                    )
                    stop = f"min({tile_begin} + {tile}, {math.prod(reduction_shape)})"
                    reduction_range = f"{start}, {stop}" + (", 32" if warp else "")
                    loop = statement_from_string(
                        f"for {index} in range({reduction_range}):\n    pass"
                    )
                assert isinstance(loop, ast.For)
                loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                    value = element(coords, self.coordinates(index, reduction_shape))
                    if indexed:
                        value = self.cast(value, compute_dtype)
                        compare = ">" if reduction_type == "argmax" else "<"
                        better = f"({value}) {compare} {total}"
                        if compute_dtype.is_floating_point:
                            better = f"({better}) or (({value}) != ({value}) and {total} == {total})"
                        # The increasing local reduction coordinate and strict
                        # comparison preserve the first tie, including NaNs.
                        self.emit(
                            f"if {index} == 0 or ({better}):\n"
                            f"    {total} = {value}\n"
                            f"    {selected} = {self.cast(index, fake.dtype)}"
                        )
                    else:
                        self.emit(f"{total} = {combine(total, value)}")
                if tiled_loop is not None:
                    tiled_loop.body = [loop]
                    self.cg.add_statement(tiled_loop)
                else:
                    self.cg.add_statement(loop)
                if warp:
                    # Output coordinates are warp-uniform, including a short
                    # final output tile. Lanes with no input use the identity;
                    # the complete warp reconverges before each shuffle.
                    peer = self.df.new_var("fragment_reduce_peer")
                    for distance in (16, 8, 4, 2, 1):
                        self.emit(
                            f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                        )
                        self.emit(f"{total} = {combine(total, peer)}")
                return selected if indexed else self.cast(total, fake.dtype)

            if node in self.warp_result_nodes:
                groups = self.warp_result_nodes[node]
                name = self.df.new_var("fragment_warp_reduction")
                self.emit(f"{name} = {self.cast('0', fake.dtype)}")
                branch = cast(
                    "ast.If",
                    statement_from_string(
                        f"if {self.thread} // 32 < {groups}:\n    pass"
                    ),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    coords = self.coordinates(
                        f"{self.thread} // 32", self.shape(fake.shape)
                    )
                    if warp:
                        self.emit(f"{name} = {reduce(coords)}")
                    else:
                        leader = cast(
                            "ast.If",
                            statement_from_string(
                                f"if {self.thread} % 32 == 0:\n    pass"
                            ),
                        )
                        leader.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", leader.body)):
                            self.emit(f"{name} = {reduce(coords)}")
                        self.cg.add_statement(leader)
                        self.emit(
                            f"{name} = cute.arch.shuffle_sync({name}, 0, mask=0xffffffff, mask_and_clamp=31)"
                        )
                self.cg.add_statement(branch)
                # Keep the existing lifetime/mutation ordering boundary.
                self.synchronize()
                return Fragment(
                    self.shape(fake.shape), fake.dtype, lambda _coords: name, True
                )
            return self.materialize(
                Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    reduce,
                    dependencies=dependencies,
                ),
                threads_per_element=32 if warp else 1,
            )
        producer = object()
        logical_dependencies = tuple(
            value for value in dependencies if value.logical_domain is not None
        )
        # These elementwise operations preserve broadcast coordinates. Other
        # lowerings (for example gather) define a different output domain.
        pointwise_domain = isinstance(node.target, torch._ops.OpOverload) and (
            torch.Tag.pointwise in node.target.tags
            or node.target is torch.ops.aten._to_copy.default
        )
        # A bare dtype cast of one staged tile (or of a cheap view of one)
        # reads exactly that tile: the mma lanes may read it in place.
        cheap = (
            node.target
            in (
                torch.ops.aten._to_copy.default,
                torch.ops.prims.convert_element_type.default,
            )
            and len(dependencies) == 1
            and ranges == result_shape
            and (dependencies[0].cheap or dependencies[0].physical_row is not None)
            and dependencies[0].shape == result_shape
        )
        return Fragment(
            self.shape(fake.shape),
            fake.dtype,
            lambda coords: self.pointwise_read(
                producer, coords, lambda: self.cast(element(coords), fake.dtype)
            ),
            dependencies=dependencies,
            logical_domain=(
                lambda coords: tuple(
                    mask
                    for value in logical_dependencies
                    for mask in value.broadcast_domain(coords)
                )
            )
            if pointwise_domain and logical_dependencies
            else None,
            cheap=cheap,
            physical_row=dependencies[0].physical_row if cheap else None,
        )

    def loop(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        graph_id = node.args[0]
        assert isinstance(graph_id, int)
        graph = self.graphs[graph_id]
        assert isinstance(graph, ForLoopGraphInfo)
        if self.local_allocations and any(
            self.df.resolved_block_size(bid) != 1 for bid in graph.block_ids
        ):
            raise exc.InvalidConfig("CTA-local atomics require uniform scalar loops")
        loop_index_type = "Int32"
        if self.local_allocations:
            parent, _slot = control_flow_parent_entries(self.graphs)[graph_id]
            owner = next(info for info in self.graphs if info.graph is parent.graph)
            if isinstance(owner, IfGraphInfo):
                self.uniform_finalizer_symbols(
                    terminal_loop_symbols(node, self.graphs), set(graph.block_ids)
                )
                # A proved scalar finalizer loop accepts runtime host integers.
                # Keep their signed 64-bit range through its induction variable;
                # narrowing to tile-index Int32 changes bounds and body indices.
                loop_index_type = "Int64"
        args = [values[x] for x in cast("list[Node]", node.args[3])]
        placeholders = list(graph.graph.find_nodes(op="placeholder"))
        output_node = graph.graph.find_nodes(op="output")[0]
        outputs = cast("list[Node]", output_node.args[0])
        carry_slots = _loop_carry_slots(node)
        # Match each output to its captured value through the actual phi edge.
        # Read-only captures can appear before, between, or after mutable ones.
        carries: list[Fragment] = []
        self.held.append(carries)
        inner_args = list(args)
        for index, output in enumerate(outputs):
            if index in carry_slots:
                slot = carry_slots[index]
                incoming = cast("Fragment", args[slot])
                carry = (
                    incoming
                    if id(incoming) in self.local_logical_sizes
                    else self.materialize(incoming, copy=True)
                )
                inner_args[slot] = carry
            else:
                fake = cast("torch.Tensor", output.meta["val"])
                carry = self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            carries.append(carry)
        body: list[ast.AST] = []
        deferred_barrier: ast.AST | None = None
        outer_offsets, outer_bounds = dict(self.offsets), dict(self.bounds)
        with self.cg.set_statements(body):
            for bid, _begin, end in zip(
                graph.block_ids,
                cast("list[object]", node.args[1]),
                cast("list[object]", node.args[2]),
                strict=True,
            ):
                self.offsets[bid] = self.df.new_var("fragment_tile")
                self.bounds[bid] = self.scalar(
                    values[end] if isinstance(end, Node) else end
                )
            results = self.graph(
                graph.graph, dict(zip(placeholders, inner_args, strict=True))
            )
            assert isinstance(results, list)
            # Snapshot every value that reads a carry before writing any carry.
            # Disjoint recipes can write directly: none of their inputs changes
            # during the parallel assignment. Keep both sets live throughout.
            snapshots = []
            self.held.extend([results, snapshots])
            destinations = self.referenced_buffers(carries)
            for result in results:
                value = cast("Fragment", result)
                snapshots.append(
                    self.materialize(value, copy=True)
                    if self.referenced_buffers([value]) & destinations
                    and id(value) not in self.local_logical_sizes
                    else value
                )
            for target, source in zip(carries, snapshots, strict=True):
                if id(target) in self.local_logical_sizes:
                    if source is not target:
                        raise exc.InvalidConfig(
                            "CTA-local atomic loop changed its allocation"
                        )
                else:
                    self.copy(source, target)
            self.held.pop()
            self.held.pop()
            # The backedge can revisit the same atomic target. Finish this
            # iteration's epoch even when no loop carry was materialized.
            pending = set(self.pending_local_atomics)
            self.synchronize_local_atomics()
            if (
                pending
                and self.df.config.get("cute_fragment_integer_atomic_epochs", False)
                and node not in self.private_scalar_loops
                and can_defer_integer_epoch(
                    body[:-1],
                    {name: dtype for name, dtype, _count in self.buffers},
                    pending,
                )
            ):
                assert ast.unparse(body[-1]) == "cute.arch.sync_threads()"
                deferred_barrier = body.pop()
        for index in reversed(range(len(graph.block_ids))):
            bid = graph.block_ids[index]
            begin = cast("list[object]", node.args[1])[index]
            start = self.scalar(values[begin] if isinstance(begin, Node) else begin)
            step = self.df.resolved_block_size(bid)
            if node.target is _tracing_ops._for_loop_step:
                explicit = cast("list[object]", node.args[4])[index]
                if explicit is not None:
                    step = self.scalar(
                        values[explicit] if isinstance(explicit, Node) else explicit
                    )
            loop = statement_from_string(
                f"for {self.offsets[bid]} in range(cutlass.{loop_index_type}({start}), cutlass.{loop_index_type}({self.bounds[bid]}), {step}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body = cast("list[ast.stmt]", body)
            body = [loop]
        self.offsets, self.bounds = outer_offsets, outer_bounds
        if node in self.private_scalar_loops:
            if len(body) != 1 or not isinstance(body[0], ast.For):
                raise exc.InvalidConfig("private scalar loops require one scalar axis")
            outgoing = {carry.storage for carry in carries}
            if None in outgoing or any(carry.shape for carry in carries):
                raise exc.InvalidConfig(
                    "private scalar loops require shared scalar carries"
                )
            body = list(
                privatize_scalar_loop(self, body[0], cast("set[str]", outgoing))
            )
        for stmt in body:
            self.cg.add_statement(stmt)
        if deferred_barrier is not None:
            # Unconditional publication also covers zero-trip loops. Nothing
            # can read or reuse target storage before this boundary.
            self.cg.add_statement(deferred_barrier)
        self.held.pop()
        return carries

    def derived_snapshot(
        self, value: Fragment, sources: frozenset[Node]
    ) -> Fragment | None:
        """Initialize a distinct typed slot; retain the recipe's domain lifetime."""
        bindings: dict[Node, SnapshotBinding] = {}
        seen: set[int] = set()

        def collect(fragment: Fragment) -> None:
            if id(fragment) in seen:
                return
            seen.add(id(fragment))
            if fragment.snapshot is not None:
                bindings[fragment.snapshot.source] = fragment.snapshot
            for dependency in fragment.dependencies:
                collect(dependency)

        collect(value)
        if (
            not sources
            or set(bindings) != sources
            or any(
                binding.source not in self.snapshot_loads
                or binding.threads != self.threads
                or binding.shape != value.shape
                or binding.domain_storage
                or binding.logical_shape
                != self.shape(snapshot_logical_shape(binding.source))
                for binding in bindings.values()
            )
        ):
            return None
        shape = value.shape
        if len(shape) != 1:
            return None
        size = shape[0]
        owner_sizes = {
            size,
            *(binding.logical_shape[0] for binding in bindings.values()),
        }
        slots = (size + self.threads - 1) // self.threads
        name = self.df.new_var("fragment_derived_snapshot")
        self.register_snapshot_buffers[name] = size
        self.emit(
            f"{name} = cute.make_rmem_tensor(({slots},), {self.dtype(value.dtype)})"
        )
        self.emit(f"{name}.fill({self.cast('0', value.dtype)})")

        def read(coords: tuple[str, ...]) -> str:
            owner = self.snapshot_owner
            if owner is None or len(coords) != 1 or owner.size not in owner_sizes:
                raise exc.InvalidConfig(
                    "derived snapshot lacks its active coordinate owner"
                )
            owner.prove(coords[0])
            return f"{name}[{owner.slot}]"

        def fill(coords: tuple[str, ...]) -> None:
            self.emit(f"{read(coords)} = {self.cast(value.read(coords), value.dtype)}")

        self.held.append(value)
        self.elements(shape, fill)
        self.held.pop()
        # This is not a SnapshotBinding for the original load. Preserve the
        # result dtype and all domain dependencies; no lifetime is discarded.
        return Fragment(
            shape,
            value.dtype,
            read,
            True,
            dependencies=(value,),
            logical_domain=value.logical_domain,
        )

    def resident_while(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        from .gather_domains import loop_domain_facts
        from .resident_while import resident_while_plan

        plan = resident_while_plan(node, self.graphs)
        if node not in loop_domain_facts(self.env, self.graphs).resident_whiles:
            raise exc.InvalidConfig(
                "resident while lacks complete logical/readonly proof"
            )
        if (
            not plan.local_allocations
            and not plan.readonly_control
            and self.df.config.get("cute_fragment_bounded_gather", False) is not True
        ):
            raise exc.InvalidConfig("resident while requires explicit bounded gather")
        if any(
            self.df.config.get(key, False)
            for key in (
                "cute_fragment_register_loads",
                "cute_fragment_register_producers",
                "cute_fragment_producer_cache",
                "cute_fragment_published_scalars",
                "cute_fragment_private_scalar_loops",
                "cute_fragment_warp_results",
                "cute_fragment_packet_loads",
                "cute_integer_loop_reduction",
            )
        ):
            raise exc.InvalidConfig(
                "resident while requires shared uncached carry storage"
            )
        grid = self.cg.current_grid_state
        assert grid is not None
        if any(self.df.resolved_block_size(bid) != 1 for bid in grid.block_id_to_info):
            raise exc.InvalidConfig("resident while requires one scalar row per CTA")
        if any(
            self.df.resolved_block_size(bid) != 1
            for child in plan.nested_fors
            for bid in cast("ForLoopGraphInfo", child.graph).block_ids
        ):
            raise exc.InvalidConfig(
                "resident while requires scalar static packet loops"
            )

        # Preserve only already initialized immutable snapshots. All mutable
        # carries and other recipes retain the original held shared copies.
        captures: list[Fragment] = []
        self.held.append(captures)
        recipes = (
            frame_snapshot_recipes(node, self.graphs, self.env).get(
                plan.body.graph.graph_id, {}
            )
            if self.df.config.get("cute_fragment_register_snapshots", False)
            else {}
        )
        for slot, entry in enumerate(plan.body.captures):
            value = cast("Fragment", values[entry])
            binding = value.snapshot
            keep_snapshot = (
                binding is not None
                and self.df.config.get("cute_fragment_register_snapshots", False)
                and binding.source in self.snapshot_loads
                and binding.threads == self.threads
                and binding.shape == value.shape
                and binding.dtype == value.dtype
                and binding.logical_shape
                == self.shape(snapshot_logical_shape(binding.source))
                and not binding.domain_storage
                and value.resident
                and value.storage is None
                and not value.dependencies
                and slot in snapshot_capture_slots(node, self.graphs, binding.source)
            )
            derived = (
                self.derived_snapshot(value, recipes[slot])
                if not keep_snapshot and slot in recipes
                else None
            )
            captured = (
                value
                if keep_snapshot
                else derived
                if derived is not None
                else self.materialize(value, copy=True)
            )
            if plan.composed and id(value) in self.local_logical_sizes:
                # The frame proof permits only completed readonly local captures.
                # Preserve declared extent on the held shared copy, never its
                # padded capacity; vector carries remain forbidden by the plan.
                self.local_logical_sizes[id(captured)] = self.local_logical_sizes[
                    id(value)
                ]
                self.local_storage.append(captured)
            captures.append(captured)
        carries = [captures[slot] for _index, slot in plan.body.carry_map]
        predicate = self.allocate(Fragment((), torch.bool, lambda _: "False"))
        self.held.append(predicate)
        take = self.df.new_var("fragment_while_take")

        def condition() -> None:
            result = self.graph(
                plan.condition.graph.graph,
                dict(zip(plan.condition.placeholders, captures, strict=True)),
            )
            assert isinstance(result, list) and len(result) == 1
            # One owned scalar writer, followed by unconditional publication.
            self.copy(cast("Fragment", result[0]), predicate)
            self.emit(f"{take} = {predicate.read(())}")
            # Every physical thread must finish reading before any can enter
            # the next iteration and overwrite this same predicate slot.
            self.synchronize()

        condition()
        statements: list[ast.AST] = []
        with self.cg.set_statements(statements):
            results = self.graph(
                plan.body.graph.graph,
                dict(zip(plan.body.placeholders, captures, strict=True)),
            )
            assert isinstance(results, list)
            # An unused final update still belongs to this iteration. Close it
            # before carry copies or any next-iteration reset can reuse storage.
            self.synchronize_local_atomics()
            snapshots: list[Fragment] = []
            self.held.extend([results, snapshots])
            destinations = self.referenced_buffers(carries)
            for result in results:
                value = cast("Fragment", result)
                snapshots.append(
                    self.materialize(value, copy=True)
                    if self.referenced_buffers([value]) & destinations
                    else value
                )
            for target, source in zip(carries, snapshots, strict=True):
                self.copy(source, target)
            self.held.pop()
            self.held.pop()
            condition()
        loop = statement_from_string(f"while {take}:\n    pass")
        assert isinstance(loop, ast.While)
        loop.body = cast("list[ast.stmt]", statements)
        self.cg.add_statement(loop)
        self.held.pop()
        self.held.pop()
        return carries

    def serial_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        source = self.materialize(source)
        self.held.append(source)
        result = self.allocate(source)
        dim = cast("int", node.args[2]) % len(source.shape)
        reverse = bool(node.args[3])
        extent = self.scan_axis_extent(node, source, dim)

        def line(coords: tuple[str, ...]) -> None:
            total = self.df.new_var("fragment_scan")
            index = self.df.new_var("fragment_scan_index")
            initialized = self.df.new_var("fragment_scan_initialized")
            self.emit(f"{total} = {self.cast('0', source.dtype)}")
            self.emit(f"{initialized} = False")
            position = f"{source.shape[dim] - 1} - {index}" if reverse else index
            location = (*coords[:dim], position, *coords[dim:])
            value = source.read(location)
            self.emit(
                f"for {index} in range({source.shape[dim]}):\n"
                f"    if ({position}) < ({extent}):\n"
                f"        if {initialized}:\n"
                f"            {total} = {self.cast(f'{total} + {value}', source.dtype)}\n"
                f"        else:\n"
                f"            {total} = {value}\n"
                f"        {initialized} = True\n"
                f"        {result.read(location)} = {total}\n"
                f"    else:\n"
                f"        {result.read(location)} = {self.cast('0', source.dtype)}"
            )

        self.elements((*source.shape[:dim], *source.shape[dim + 1 :]), line)
        self.held.pop()
        return result

    def scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        from ..autotuner_heuristics.cute_fragment_warp_scan import (
            active_warp_scan_roots,
        )
        from ..autotuner_heuristics.cute_fragment_warp_scan import warp_scan_supported

        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        dim = cast("int", node.args[2]) % len(source.shape)
        root = self.cg.current_root_graph_info
        assert root is not None
        if (
            self.df.config.get("cute_fragment_warp_scan", False)
            and root.graph_id in active_warp_scan_roots(self.env, self.df.config)
            and warp_scan_supported(source.dtype, source.shape[dim])
        ):
            return self.warp_scan(node, values)
        mode = self.df.config.get("cute_fragment_scan", "serial")
        if mode == "serial":
            return self.serial_scan(node, values)
        if mode == "cooperative":
            return self.cooperative_scan(node, values)
        raise exc.InvalidConfig(f"unsupported computed fragment scan: {mode!r}")

    def warp_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        """Two CTA phases, each with full-warp, typed prefix exchanges.

        A warp owns one row chunk. The second phase scans at most 32 chunk
        totals, then adds the exclusive chunk carry to its own partials.
        No input is overwritten and no collective is guarded by a lane test.
        """
        from ..autotuner_heuristics.cute_fragment_warp_scan import warp_scan_supported

        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        dim = cast("int", node.args[2]) % len(source.shape)
        capacity = source.shape[dim]
        if not warp_scan_supported(source.dtype, capacity):
            raise exc.InvalidConfig(
                "warp-prefix scan requires a positive Int32/64 axis or Float32/64 axis capacity <= 1024"
            )
        assert self.threads % 32 == 0
        reverse = bool(node.args[3])
        extent = self.scan_axis_extent(node, source, dim)
        return self.warp_scan_fragment(
            source, dim, extent, reverse, export=node in self.scan_plans
        )

    def warp_scan_fragment(
        self,
        source: Fragment,
        dim: int,
        extent: str,
        reverse: bool,
        *,
        export: bool = False,
    ) -> Fragment:
        """Recursively scan chunk totals using associative integer addition.

        Every chunk writes a total, including zero for logical padding. These
        totals have their own complete logical domain. Their recursive scan is
        forward even when the original positions are visited in reverse order.
        Small scans retain the original two-phase floating-point tree verbatim.
        """
        statements = self.cg.statements_stack[-1]
        start = len(statements)
        capacity = source.shape[dim]
        assert capacity <= 1024 or source.dtype in (torch.int32, torch.int64)
        row_shape = (*source.shape[:dim], *source.shape[dim + 1 :])
        chunks = (capacity + 31) // 32
        work_shape = (*row_shape, chunks)
        zero = self.cast("0", source.dtype)
        register = self.df.new_var("fragment_scan_export") if export else ""
        slot = self.df.new_var("fragment_scan_slot") if export else ""
        if export:
            slots = (capacity + self.threads - 1) // self.threads
            self.emit(
                f"{register} = cute.make_rmem_tensor(({slots},), {self.dtype(source.dtype)})"
            )
            self.emit(f"{register}.fill({zero})")

        def scan_elements(body: Callable[[tuple[str, ...]], None]) -> None:
            if not export:
                self.elements(work_shape, body, threads_per_element=32)
                return
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                ),
            )
            loop.body.clear()
            chunk = self.df.new_var("fragment_scan_chunk")
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(
                    f"{chunk} = {self.thread} // 32 + {slot} * {self.threads // 32}"
                )
                guard = cast(
                    "ast.If", statement_from_string(f"if {chunk} < {chunks}:\n    pass")
                )
                guard.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
                    body((*("0" for _ in row_shape), chunk))
                self.cg.add_statement(guard)
            self.cg.add_statement(loop)
            self.synchronize()

        self.held.append(source)
        result = self.allocate(source)
        self.held.append(result)
        totals = self.allocate(Fragment(work_shape, source.dtype, lambda _: zero))
        self.held.append(totals)

        def prefix(value: str, lane: str) -> None:
            for distance in (1, 2, 4, 8, 16):
                peer = self.df.new_var("fragment_scan_peer")
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_up({value}, offset={distance}, "
                    "mask=0xffffffff, mask_and_clamp=0)"
                )
                self.emit(
                    f"if {lane} >= {distance}:\n"
                    f"    {value} = {self.cast(f'{peer} + {value}', source.dtype)}"
                )

        def positions(coords: tuple[str, ...]) -> tuple[str, str, tuple[str, ...]]:
            lane = self.df.new_var("fragment_scan_lane")
            rank = self.df.new_var("fragment_scan_rank")
            self.emit(f"{lane} = {self.thread} % 32")
            self.emit(f"{rank} = ({coords[-1]}) * 32 + {lane}")
            position = f"({extent}) - 1 - {rank}" if reverse else rank
            row = coords[:-1]
            return lane, rank, (*row[:dim], position, *row[dim:])

        def partials(coords: tuple[str, ...]) -> None:
            lane, rank, location = positions(coords)
            value = self.df.new_var("fragment_scan_value")
            self.emit(f"{value} = {zero}")
            # A lazy source read can emit statements. Keep all such statements
            # under the logical-bound guard, including reverse-tail addresses.
            branch = cast(
                "ast.If", statement_from_string(f"if {rank} < ({extent}):\n    pass")
            )
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                self.emit(f"{value} = {self.cast(source.read(location), source.dtype)}")
            branch.orelse = [statement_from_string(f"{value} = {zero}")]
            self.cg.add_statement(branch)
            prefix(value, lane)
            if export:
                self.emit(f"{register}[{slot}] = {value}")
            row = coords[:-1]
            padding = (*row[:dim], rank, *row[dim:])
            self.emit(
                f"if {rank} < ({extent}):\n"
                f"    {result.read(location)} = {value}\n"
                f"elif {rank} < {capacity}:\n"
                f"    {result.read(padding)} = {zero}"
            )
            last = self.df.new_var("fragment_scan_last")
            total = self.df.new_var("fragment_scan_total")
            self.emit(f"{last} = min(31, max(0, ({extent}) - ({coords[-1]}) * 32 - 1))")
            self.emit(
                f"{total} = cute.arch.shuffle_sync({value}, {last}, "
                "mask=0xffffffff, mask_and_clamp=31)"
            )
            self.emit(
                f"if {lane} == 0:\n"
                f"    {totals.read(coords)} = {total} if ({coords[-1]}) * 32 < ({extent}) else {zero}"
            )

        scan_elements(partials)

        if chunks > 32:
            prefixes = self.warp_scan_fragment(
                totals, len(work_shape) - 1, str(chunks), False
            )
            self.held.append(prefixes)

            def wide_carries(coords: tuple[str, ...]) -> None:
                position = coords[dim]
                rank = f"({extent}) - 1 - ({position})" if reverse else position
                chunk = f"(({rank}) // 32)"
                row = (*coords[:dim], *coords[dim + 1 :])
                branch = cast(
                    "ast.If",
                    statement_from_string(
                        f"if ({position}) < ({extent}) and {chunk} > 0:\n    pass"
                    ),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    carry = prefixes.read((*row, f"{chunk} - 1"))
                    partial = result.read(coords)
                    self.emit(
                        f"{partial} = {self.cast(f'{carry} + {partial}', source.dtype)}"
                    )
                self.cg.add_statement(branch)

            self.elements(source.shape, wide_carries)
            self.held.pop()
            self.held.pop()
            self.held.pop()
            self.held.pop()
            return result

        def carries(coords: tuple[str, ...]) -> None:
            lane, rank, location = positions(coords)
            value = self.df.new_var("fragment_scan_chunks")
            self.emit(
                f"{value} = {totals.read((*coords[:-1], lane))} if {lane} < {chunks} else {zero}"
            )
            prefix(value, lane)
            carry = self.df.new_var("fragment_scan_carry")
            self.emit(
                f"{carry} = cute.arch.shuffle_sync({value}, max(0, ({coords[-1]}) - 1), "
                "mask=0xffffffff, mask_and_clamp=31)"
            )
            # Chunk zero retains its partial bit-for-bit: adding an artificial
            # zero would change negative zero and is unnecessary.
            partial = f"{register}[{slot}]" if export else result.read(location)
            self.emit(
                f"if {rank} < ({extent}) and ({coords[-1]}) > 0:\n"
                f"    {result.read(location)} = {self.cast(f'{carry} + {partial}', source.dtype)}"
            )
            if export:
                self.emit(
                    f"if {rank} < ({extent}) and ({coords[-1]}) > 0:\n"
                    f"    {register}[{slot}] = {self.cast(f'{carry} + {partial}', source.dtype)}"
                )

        scan_elements(carries)
        self.held.pop()
        self.held.pop()
        self.held.pop()
        if self.warp_producer_recorder is not None:
            self.warp_producer_recorder.scan(statements[start:], source, result, totals)
        if export:

            def read(coords: tuple[str, ...]) -> str:
                if self.snapshot_owner is None:
                    return result.read(coords)
                self.snapshot_owner.prove(self.flatten(coords, source.shape))
                return f"{register}[{self.snapshot_owner.slot}]"

            return dataclasses.replace(result, element=read, dependencies=(result,))
        return result

    def cta_reduction(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        reduction_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
        kind: str,
        element: Callable,
        combine: Callable,
        dependencies: tuple[Fragment, ...],
    ) -> Fragment:
        """Cooperate across the CTA for each complete output coordinate.

        Warp leaders publish partials, then a complete first warp combines
        them. Every thread participates in the publication and retirement
        barriers, including threads with an empty input stripe.
        """
        assert self.threads % 32 == 0
        self.held.append(dependencies)
        result = self.allocate(Fragment(shape, dtype, lambda _: "0"))
        self.held.append(result)
        warps = self.threads // 32
        scratch = self.allocate(Fragment((warps,), compute_dtype, lambda _: "0"))
        self.held.append(scratch)
        output = self.df.new_var("fragment_cta_output")
        index = self.df.new_var("fragment_cta_index")
        total = self.df.new_var("fragment_cta_reduce")
        peer = self.df.new_var("fragment_cta_peer")
        identity = self.cast(
            self.scalar(Reduction.default_accumulator(kind, compute_dtype)),
            compute_dtype,
        )
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {output} in range({math.prod(shape)}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            coords = self.coordinates(output, shape)
            self.emit(f"{total} = {identity}")
            unroll = self.df.config.get("cute_fragment_reduction") == "cta_unroll"
            slot = self.df.new_var("fragment_cta_slot") if unroll else index
            count = math.prod(reduction_shape)
            iteration = (
                f"cutlass.range_constexpr({(count + self.threads - 1) // self.threads})"
                if unroll
                else f"range({self.thread}, {count}, {self.threads})"
            )
            inputs = cast(
                "ast.For",
                statement_from_string(f"for {slot} in {iteration}:\n    pass"),
            )
            inputs.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", inputs.body)):
                if unroll:
                    self.emit(f"{index} = {self.thread} + {slot} * {self.threads}")
                    guard = cast(
                        "ast.If",
                        statement_from_string(f"if {index} < {count}:\n    pass"),
                    )
                    guard.body.clear()
                    with (
                        self.cg.set_statements(cast("list[ast.AST]", guard.body)),
                        self.snapshot_scope(index, slot, count),
                    ):
                        value = element(
                            coords, self.coordinates(index, reduction_shape)
                        )
                        self.emit(
                            f"{total} = {self.cast(combine(total, value), compute_dtype)}"
                        )
                    self.cg.add_statement(guard)
                else:
                    value = element(coords, self.coordinates(index, reduction_shape))
                    self.emit(
                        f"{total} = {self.cast(combine(total, value), compute_dtype)}"
                    )
            self.cg.add_statement(inputs)
            for distance in (16, 8, 4, 2, 1):
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                )
                self.emit(f"{total} = {self.cast(combine(total, peer), compute_dtype)}")
            self.emit(
                f"if {self.thread} % 32 == 0:\n    {scratch.read((f'{self.thread} // 32',))} = {total}"
            )
            self.synchronize()
            guard = cast(
                "ast.If", statement_from_string(f"if {self.thread} < 32:\n    pass")
            )
            guard.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
                self.emit(f"{total} = {identity}")
                self.emit(
                    f"if {self.thread} < {warps}:\n    {total} = {scratch.read((self.thread,))}"
                )
                for distance in (16, 8, 4, 2, 1):
                    self.emit(
                        f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                    )
                    self.emit(
                        f"{total} = {self.cast(combine(total, peer), compute_dtype)}"
                    )
                self.emit(
                    f"if {self.thread} == 0:\n    {result.read(coords)} = {self.cast(total, dtype)}"
                )
            self.cg.add_statement(guard)
            self.synchronize()
        self.cg.add_statement(loop)
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return result

    def completed_scan_reductions(self, shape: tuple[int, ...]) -> None:
        size = math.prod(shape)
        pending = self.scan_pending
        dtype = pending[0][2]
        assert all(item[2] == dtype for item in pending)
        totals = [self.df.new_var("fragment_scan_reduce") for _ in pending]
        identities = [
            self.cast(self.scalar(Reduction.default_accumulator(item[3], dtype)), dtype)
            for item in pending
        ]

        def combine(kind: str, left: str, right: str) -> str:
            return self.cast(
                self.env.backend.reduction_combine_expr(kind, left, right, dtype), dtype
            )

        for total, identity in zip(totals, identities, strict=True):
            self.emit(f"{total} = {identity}")
        slot = self.df.new_var("fragment_scan_consumer_slot")
        index = self.df.new_var("fragment_scan_consumer_index")
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {slot} in cutlass.range_constexpr({(size + self.threads - 1) // self.threads}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            self.emit(f"{index} = {self.thread} + {slot} * {self.threads}")
            guard = cast(
                "ast.If", statement_from_string(f"if {index} < {size}:\n    pass")
            )
            guard.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
                assert self.snapshot_owner is None
                self.snapshot_owner = SnapshotOwner(index, slot, size)
                try:
                    for (result, element, _, kind), total in zip(
                        pending, totals, strict=True
                    ):
                        value = element(
                            tuple("0" for _ in result.shape),
                            self.coordinates(index, shape),
                        )
                        self.emit(f"{total} = {combine(kind, total, value)}")
                finally:
                    self.snapshot_owner = None
            self.cg.add_statement(guard)
        self.cg.add_statement(loop)
        peer = self.df.new_var("fragment_scan_reduce_peer")
        for distance in (16, 8, 4, 2, 1):
            for (_, _, _, kind), total in zip(pending, totals, strict=True):
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                )
                self.emit(f"{total} = {combine(kind, total, peer)}")
        warps = self.threads // 32
        scratch = self.allocate(Fragment((warps * len(pending),), dtype, lambda _: "0"))
        self.held.append(scratch)
        for i, total in enumerate(totals):
            self.emit(
                f"if {self.thread} % 32 == 0:\n    {scratch.read((f'{i * warps} + {self.thread} // 32',))} = {total}"
            )
        self.synchronize()
        guard = cast(
            "ast.If", statement_from_string(f"if {self.thread} < 32:\n    pass")
        )
        guard.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
            for i, ((result, _, _, kind), total, identity) in enumerate(
                zip(pending, totals, identities, strict=True)
            ):
                self.emit(f"{total} = {identity}")
                self.emit(
                    f"if {self.thread} < {warps}:\n    {total} = {scratch.read((f'{i * warps} + {self.thread}',))}"
                )
                for distance in (16, 8, 4, 2, 1):
                    self.emit(
                        f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                    )
                    self.emit(f"{total} = {combine(kind, total, peer)}")
                self.emit(
                    f"if {self.thread} == 0:\n    {result.read(tuple('0' for _ in result.shape))} = {self.cast(total, result.dtype)}"
                )
        self.cg.add_statement(guard)
        self.synchronize()
        self.held.pop()

    def cooperative_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        self.held.append(source)
        current = self.allocate(source)
        self.held.append(current)
        other = self.allocate(source)
        self.held.append(other)
        dim = cast("int", node.args[2]) % len(source.shape)
        reverse = bool(node.args[3])
        extent = self.scan_axis_extent(node, source, dim)

        def initialize(coords: tuple[str, ...]) -> None:
            self.emit(
                f"if ({coords[dim]}) < ({extent}):\n"
                f"    {current.read(coords)} = {self.cast(source.read(coords), source.dtype)}\n"
                f"else:\n"
                f"    {current.read(coords)} = {self.cast('0', source.dtype)}"
            )

        statements = self.cg.statements_stack[-1]
        start = len(statements)
        self.elements(source.shape, initialize)
        if self.warp_producer_recorder is not None:
            self.warp_producer_recorder.publication(statements[start:], current)
        # Every stage reads only the preceding stage. A CTA barrier follows
        # each complete tile write, including when a thread owns several
        # elements. Never overwrite the input: it may have other consumers.
        for stage in range((source.shape[dim] - 1).bit_length()):
            distance = 1 << stage

            def combine(
                coords: tuple[str, ...],
                distance: int = distance,
                current: Fragment = current,
                other: Fragment = other,
            ) -> None:
                position = coords[dim]
                neighbor = (
                    f"({position}) + {distance}"
                    if reverse
                    else f"({position}) - {distance}"
                )
                in_range = (
                    f"({neighbor}) < ({extent})" if reverse else f"({neighbor}) >= 0"
                )
                location = (*coords[:dim], neighbor, *coords[dim + 1 :])
                left = current.read(location)
                right = current.read(coords)
                self.emit(
                    f"if ({position}) < ({extent}) and ({in_range}):\n"
                    f"    {other.read(coords)} = {self.cast(f'{left} + {right}', source.dtype)}\n"
                    f"else:\n"
                    f"    {other.read(coords)} = {right}"
                )

            start = len(statements)
            self.elements(source.shape, combine)
            if self.warp_producer_recorder is not None:
                self.warp_producer_recorder.publication(statements[start:], other)
            current, other = other, current
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return current

    def uniform_finalizer_symbols(
        self, nodes: frozenset[Node], loop_blocks: set[int] | None = None
    ) -> None:
        for node in nodes:
            value = node.meta["val"]
            expression = value._sympy_() if isinstance(value, torch.SymInt) else value
            for symbol in sympy.sympify(expression).free_symbols:
                entry = HostFunction.current().expr_to_origin.get(symbol)
                origin = entry.origin if entry is not None else None
                if not isinstance(origin, HostOrigin) and not (
                    type(origin) is GridOrigin
                    and (
                        origin.block_id in self.offsets
                        or loop_blocks is not None
                        and origin.block_id in loop_blocks
                    )
                    and self.df.resolved_block_size(origin.block_id) == 1
                ):
                    raise exc.InvalidConfig(
                        "local finalizer requires proved uniform scalar origins"
                    )

    def conditional(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        composed_tree = None
        try:
            local_inputs = (
                uniform_local_branch_inputs(node, self.graphs)
                if self.local_allocations
                else None
            )
        except exc.InvalidConfig:
            local_inputs = None
        if local_inputs is None and self.local_allocations:
            from .uniform_region_tree import uniform_local_regions

            try:
                composed_tree = uniform_local_regions(self.graphs)
            except exc.InvalidConfig:
                pass
            else:
                members = [n for f in composed_tree.frames for n in f.graph.graph.nodes]
                local_inputs = (
                    frozenset(
                        n
                        for n in members
                        if isinstance(n.meta.get("lowering"), ReductionLowering)
                    ),
                    frozenset(
                        n for n in members if n.target is _tracing_ops._get_symnode
                    ),
                    frozenset(
                        n
                        for n in members
                        if n.target is memory_ops.load
                        and n not in {read.node for read in composed_tree.indexed_reads}
                    ),
                )
        if local_inputs is not None:
            reductions, symbols, loads = local_inputs
            if any(
                not host_load_is_readonly(load, self.env, self.graphs) for load in loads
            ):
                raise exc.InvalidConfig(
                    "local-buffer predicate requires readonly bound host storage"
                )
            self.uniform_finalizer_symbols(symbols)
            if reductions & self.warp_result_nodes.keys():
                raise exc.InvalidConfig(
                    "local-buffer predicate requires CTA-shared reduction ownership"
                )
            predicate = cast("Node", node.args[0])
            value = values[predicate]
            if not isinstance(value, Fragment) or value.shape != ():
                raise exc.InvalidConfig("local-buffer predicate must be scalar")
            # Even a one-element reduction may remain lazy. Evaluate exactly once
            # and exchange before all CTA threads read the same scalar address.
            values[predicate] = self.materialize(value, copy=True)
            self.held.append(
                [
                    values[predicate],
                    *(
                        values[source]
                        for side in node.args[3:5]
                        for source in cast("list[Node]", side)
                    ),
                ]
            )
        elif self.local_allocations:
            tickets, symbols = terminal_finalizer_inputs(node, self.graphs)
            predicate_value = values[cast("Node", node.args[0])]
            predicate_buffers = self.referenced_buffers([predicate_value])
            for ticket in tickets:
                result = self.scalar_ordered_tickets.get(ticket)
                if (
                    result is None
                    or result.shape != ()
                    or not result.resident
                    or result.storage is None
                    or result.storage not in predicate_buffers
                    or ticket in self.warp_result_nodes
                ):
                    raise exc.InvalidConfig(
                        "local finalizer requires one shared scalar ticket and CTA broadcast"
                    )
            self.uniform_finalizer_symbols(symbols)
        info = self.graphs[cast("int", node.args[1])]
        assert isinstance(info, IfGraphInfo)
        assert info.branches_outputs is not None
        # An unchanged branch output can be captured only by the other branch.
        # Both lists bind values in this outer graph, not branch-local results.
        outer_captures: dict[str, Node] = {}
        for side, names in enumerate((info.if_arg_names, info.else_arg_names)):
            assert names is not None
            for name, source in zip(
                names, cast("list[Node]", node.args[3 + side]), strict=True
            ):
                if name in outer_captures and outer_captures[name] is not source:
                    raise exc.InvalidConfig(
                        "computed fragment conditional has conflicting outer captures"
                    )
                outer_captures[name] = source
        for slots in info.branches_outputs:
            if any(
                isinstance(slot, str) and slot not in outer_captures for slot in slots
            ):
                raise exc.InvalidConfig(
                    "computed fragment conditional requires a captured unchanged output"
                )
        merged: list[Fragment] = []
        self.held.append(merged)
        fake_outputs = cast("list[torch.Tensor]", node.meta["val"])
        for fake in fake_outputs[: len(info.branches_outputs)]:
            merged.append(
                self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            )
        branch_local = local_inputs is not None and any(
            allocation.graph is self.graphs[cast("int", graph_id)].graph
            for allocation in self.local_allocations
            for graph_id in node.args[1:3]
        )
        if composed_tree is not None:
            # Descendant fresh epochs require the same per-arm retirement even
            # when an immediate arm owns no allocation itself.
            branch_local = True
        incoming_pending = self.pending_local_atomics.copy()
        branches: list[list[ast.AST]] = []
        for side in range(2):
            if composed_tree is not None:
                self.pending_local_atomics = incoming_pending.copy()
            branch = self.graphs[cast("int", node.args[1 + side])]
            captures = [
                values[source] for source in cast("list[Node]", node.args[3 + side])
            ]
            body: list[ast.AST] = []
            with self.cg.set_statements(body):
                outputs = self.graph(
                    branch.graph,
                    dict(
                        zip(
                            branch.graph.find_nodes(op="placeholder"),
                            captures,
                            strict=True,
                        )
                    ),
                )
                if branch_local:
                    # Each arm starts with no pending incoming atomic epoch.
                    # An unused final update must complete inside this arm, not
                    # borrow a barrier emitted while generating its sibling.
                    self.synchronize_local_atomics()
                assert isinstance(outputs, list)
                self.held.append(outputs)
                for target, slots in zip(merged, info.branches_outputs, strict=True):
                    slot = slots[side]
                    source = (
                        outputs[slot]
                        if isinstance(slot, int)
                        else values[outer_captures[slot]]
                    )
                    assert isinstance(source, Fragment)
                    self.copy(source, target)
                self.held.pop()
            branches.append(body or [ast.Pass()])
        predicate = node.args[0]
        proxy = predicate.meta["val"] if isinstance(predicate, Node) else predicate
        constant = self.df.evaluate_constexpr_condition(proxy)
        test = (
            repr(constant)
            if constant is not None
            else self.scalar(
                values[predicate] if isinstance(predicate, Node) else predicate
            )
        )
        statement = statement_from_string(f"if {test}:\n    pass")
        assert isinstance(statement, ast.If)
        statement.body = cast("list[ast.stmt]", branches[0])
        statement.orelse = cast("list[ast.stmt]", branches[1])
        self.cg.add_statement(statement)
        if branch_local:
            # Keep predicate/captures alive until all CTA threads have left the
            # selected arm, before any branch-local storage lifetime can end.
            self.synchronize()
        self.held.pop()
        if local_inputs is not None:
            self.held.pop()
        return [*merged, *merged]

    def can_fuse_atomics(
        self, region: AtomicConsumerRegion, values: dict[Node, object]
    ) -> bool:
        if self.df.config.get("cute_fragment_atomic_aggregation", False):
            return False
        shapes = set()
        storages = set()
        for node in region.atomics:
            fake = cast("torch.Tensor", node.meta["val"])
            target = values.get(cast("Node", node.args[0]))
            if (
                node in self.warp_result_nodes
                or not isinstance(target, Fragment)
                or target.storage is None
                or len(target.shape) != 1
                or id(target) not in self.local_logical_sizes
                or target.storage in storages
            ):
                return False
            storages.add(target.storage)
            shapes.add(self.shape(fake.shape))
        if len(shapes) != 1:
            return False
        external = {
            arg
            for node in region.nodes
            for arg in node.all_input_nodes
            if arg not in region.nodes and arg not in region.targets
        }
        if (
            self.referenced_buffers([values[arg] for arg in external if arg in values])
            & storages
        ):
            return False
        shape = next(iter(shapes))
        register_owned = any(node in self.local_register_nodes for node in region.nodes)
        if register_owned and math.prod(shape) > self.threads * MAX_SLOTS:
            return False
        return not any(node in self.warp_result_nodes for node in region.nodes)

    def fused_atomic_updates(
        self,
        region: AtomicConsumerRegion,
        updates: list[tuple[Callable[[tuple[str, ...]], None], bool]],
    ) -> None:
        assert len(updates) == len(region.atomics) + len(region.stores)
        shape = self.shape(cast("torch.Tensor", region.atomics[0].meta["val"]).shape)

        def body(coords: tuple[str, ...]) -> None:
            previous = self.expression
            # Only this proved immutable dependency closure survives the
            # individual updates. Producer objects are current SSA recipes;
            # their values include every original logical-dtype cast.
            self.expression = FragmentExpression(self.cg)
            try:
                for update, _synchronize in updates:
                    update(coords)
            finally:
                self.expression = previous

        self.elements(
            shape,
            body,
            synchronize=False,
            register_owned=any(
                node in self.local_register_nodes for node in region.nodes
            ),
            iteration_local=bool(region.local_returns),
        )
        pending = self.pending_local_atomics.copy()
        for _update, synchronize in updates:
            if synchronize:
                self.synchronize()
        # Keep the original conservative post-region publication obligations.
        # Removing these now-redundant boundaries is a separate optimization.
        self.pending_local_atomics.update(pending)

    def graph(self, graph: torch.fx.Graph, values: dict[Node, object]) -> object:
        from .producer_cache import producer_cache_candidates

        cache_nodes = (
            producer_cache_candidates(graph, self.shape)
            if self.df.config.get("cute_fragment_producer_cache", False)
            and not self.scopes
            else frozenset()
        )
        previous_opaque = self.lazy_opaque_producers
        pure_plan = (
            pure_producer_plan(
                graph,
                self.env,
                cache_nodes=cache_nodes,
                shape=self.shape,
                graphs=self.graphs,
            )
            if self.df.config.get("cute_fragment_pure_producer_regions", False)
            and not self.scopes
            else None
        )
        self.lazy_opaque_producers = pure_plan.lazy if pure_plan else frozenset()
        if pure_plan is not None:
            cache_nodes = (cache_nodes - pure_plan.replicated) | pure_plan.publications
        previous_scans, previous_reductions = self.scan_plans, self.scan_reductions
        plans = (
            completed_scan_plans(graph, self.shape)
            if self.df.config.get("cute_fragment_scan_exports", False)
            and not self.scopes
            else ()
        )
        self.scan_plans = {plan.scan: plan for plan in plans}
        self.scan_reductions = {
            node: plan for plan in plans for node in plan.reductions
        }
        uses = {node: len(node.users) for node in graph.nodes}
        regions = (
            atomic_consumer_regions(self.graphs, self.env)
            if self.df.config.get("cute_fragment_atomic_consumer_fusion", False)
            and not self.scopes
            else {}
        )
        region: AtomicConsumerRegion | None = None
        updates: list[tuple[Callable[[tuple[str, ...]], None], bool]] = []
        held: list[object] = []
        previous_cache_nodes, previous_region = self.cache_nodes, self.active_region
        self.cache_nodes = frozenset(cache_nodes)
        self.active_region = None
        self.scopes.append(values)
        try:
            for node in graph.nodes:
                if self.warp_producer_recorder is not None:
                    self.warp_producer_recorder.enter(node)
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    assert region is None
                    self.flush_pending_barrier()
                    return _resolve(node.args[0], values)
                if (
                    node in regions
                    and not cache_nodes.intersection(regions[node].nodes)
                    and self.can_fuse_atomics(regions[node], values)
                ):
                    assert region is None
                    region = regions[node]
                    self.active_region = region
                    assert not self.iteration_local_returns
                    self.iteration_local_returns = frozenset(region.local_returns)
                    updates, held = [], []
                    self.held.append(held)
                if region is not None:
                    held.extend(
                        (
                            _resolve(node.args, values),
                            [_resolve(value, values) for value in node.kwargs.values()],
                        )
                    )
                with node.meta["location"], V.set_current_node(node):
                    values[node] = (
                        self.node(node, values, updates)
                        if region is not None
                        and node in (*region.atomics, *region.stores)
                        else self.node(node, values)
                    )
                    if region is not None:
                        held.append(values[node])
                    if node in cache_nodes:
                        value = values[node]
                        assert isinstance(value, Fragment)
                        values[node] = self.materialize(value)
                for source in node.all_input_nodes:
                    uses[source] -= 1
                    if uses[source] == 0:
                        values.pop(source, None)
                if uses[node] == 0:
                    values.pop(node, None)
                if region is not None and node is region.nodes[-1]:
                    self.fused_atomic_updates(region, updates)
                    self.held.pop()
                    region = None
                    self.active_region = None
                    self.iteration_local_returns = frozenset()
        finally:
            self.iteration_local_returns = frozenset()
            self.lazy_opaque_producers = previous_opaque
            self.scan_plans, self.scan_reductions = previous_scans, previous_reductions
            self.cache_nodes, self.active_region = previous_cache_nodes, previous_region
            self.scopes.pop()
        raise AssertionError("graph has no output")

    def node(
        self,
        node: Node,
        values: dict[Node, object],
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
    ) -> object:
        target = node.target
        # A run of batched host loads ends at the first node that is neither
        # one of them nor a lazy recipe (which emits nothing until consumed).
        if (
            self.pending_barrier
            and not self.batchable_host_load(node, values)
            and not self.transparent_node(node)
        ):
            self.synchronize()
        args = cast("tuple[object, ...]", _resolve(node.args, values))
        fake = node.meta.get("val")
        if target is _tracing_ops._host_tensor:
            assert isinstance(fake, torch.Tensor)
            return HostTensor(
                self.df.tensor_arg(fake, prefer_name=cast("str", node.args[0])).name,
                fake,
            )
        if target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            return fake
        if target is _tracing_ops._new_var:
            return args[0]
        if target is _tracing_ops._phi:
            return args[1]
        if target is operator.getitem:
            sequence, index = args
            assert isinstance(sequence, (list, tuple)) and isinstance(index, int)
            return sequence[index]
        # CTA-local targets have a proved initialize/update/final-read lifetime.
        # Coalesce only updates to disjoint targets. Repeated updates to one
        # target retain their collective epoch boundary, including floating
        # association across threads. Finish pending writes before a read,
        # storage reuse, or conservative effect boundary.
        # Alias-only nodes above do not access memory. Check physical storage,
        # including lazy dependencies, so consumers cannot hide an alias.
        local_update = (
            target is atomic_ops.atomic_add
            and isinstance(args[0], Fragment)
            and id(args[0]) in self.local_logical_sizes
            and not node.users
        )
        read_args = args[1:] if local_update else args
        if (
            target in (_tracing_ops._if, _tracing_ops._while_loop)
            or _tracing_ops.is_for_loop_target(target)
            or (target is atomic_ops.atomic_add and not local_update)
            or (
                local_update
                and cast("Fragment", args[0]).storage in self.pending_local_atomics
            )
            or self.pending_local_atomics
            & self.referenced_buffers(
                [
                    *read_args,
                    *(_resolve(value, values) for value in node.kwargs.values()),
                ]
            )
        ):
            # A loop can execute zero times. Never let a barrier generated only
            # in its body discharge a write from before that loop.
            self.synchronize_local_atomics()
        if target is _tracing_ops._if:
            return self.conditional(node, values)
        if target is _tracing_ops._while_loop:
            return self.resident_while(node, values)
        if _tracing_ops.is_for_loop_target(target):
            return self.loop(node, values)
        if target is atomic_ops.atomic_add:
            if deferred_updates is not None:
                return self.atomic_add(node, values, deferred_updates)
            return self.atomic_add(node, values)
        if target in (memory_ops.load, memory_ops.store):
            return self.memory(
                node, values, target is memory_ops.store, deferred_updates
            )
        if is_ordered_selection(node):
            return self.topk(node, values)
        if target is torch.ops.aten.gather.default:
            from .bounded_gather import prove_gather

            if (
                self.df.config.get("cute_fragment_bounded_gather", False) is True
                and prove_gather(self.env, node, graphs=self.graphs) is not None
            ):
                return self.bounded_gather(node, values)
            return self.topk_gather(node, values)
        if target is matmul_ops.dot:
            return self.dot(node, values)
        if target is scan_ops._associative_scan:
            return self.scan(node, values)
        if target is inline_asm_ops.inline_asm_elementwise:
            assert isinstance(fake, torch.Tensor)
            asm, constraints, operands, dtype, _is_pure, _pack = args
            assert isinstance(operands, (tuple, list))
            fragments = tuple(cast("Fragment", operand) for operand in operands)

            def inline_asm(coords: tuple[str, ...]) -> str:
                inputs = ", ".join(
                    self.cast(operand.broadcast(coords), operand.dtype)
                    for operand in fragments
                )
                arguments = f"({inputs},)" if inputs else "()"
                return (
                    f"_cute_inline_asm_elementwise({arguments}, asm={asm!r}, "
                    f"constraints={constraints!r}, dtype={self.dtype(fake.dtype)}, "
                    "is_pure=True)"
                )

            if node in self.lazy_opaque_producers:
                return Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    lambda coords: self.pointwise_read(
                        node, coords, lambda: self.cast(inline_asm(coords), fake.dtype)
                    ),
                    dependencies=fragments,
                )

            # Opaque scalar programs may be expensive and can feed several
            # consumers. Evaluate each logical element once before sharing it.
            return self.materialize(
                Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    inline_asm,
                    dependencies=fragments,
                )
            )
        if target in (tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id):
            return fake
        if target is tile_ops.tile_index:
            assert isinstance(fake, torch.Tensor)
            block_id = self.env.resolve_block_id(args[0])
            assert block_id in self.offsets
            offset = self.offsets[block_id]
            return Fragment(
                self.shape(fake.shape),
                fake.dtype,
                lambda coords: self.cast(f"({offset}) + ({coords[0]})", fake.dtype),
                pure=True,
            )
        if isinstance(node.meta["lowering"], (PointwiseLowering, ReductionLowering)):
            return self.pointwise(node, values)
        if not isinstance(fake, torch.Tensor):
            return fake
        shape = self.shape(fake.shape)
        if target in (
            creation_ops.full,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
            _tracing_ops._constant_tensor,
        ):
            value = (
                args[1]
                if target in (creation_ops.full, torch.ops.aten.full.default)
                else args[0]
            )
            declared_shape = (
                self.shape(args[0])
                if target in (creation_ops.full, torch.ops.aten.full.default)
                else shape
            )
            result = Fragment(
                shape,
                fake.dtype,
                lambda _: self.cast(self.scalar(value), fake.dtype),
                dependencies=(value,) if isinstance(value, Fragment) else (),
                logical_domain=(
                    lambda coords: tuple(
                        f"({coord}) < ({size})"
                        for coord, size in zip(coords, declared_shape, strict=True)
                    )
                )
                if declared_shape != shape
                else None,
                pure=not isinstance(value, Fragment),
            )
            if node in self.local_allocations:
                result = self.materialize(result)
                assert result.storage is not None
                self.local_storage.append(result)
                self.local_logical_sizes[id(result)] = cast("list[int]", node.args[0])[
                    0
                ]
            return result
        if target is torch.ops.prims.iota.default:
            start = _resolve(node.kwargs.get("start", 0), values)
            step = _resolve(node.kwargs.get("step", 1), values)
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: self.cast(
                    f"{self.scalar(start)} + ({coords[0]}) * {self.scalar(step)}",
                    fake.dtype,
                ),
                pure=True,
                affine=(start, step)
                if len(shape) == 1
                and isinstance(start, int)
                and isinstance(step, int)
                and not isinstance(start, bool)
                and not isinstance(step, bool)
                else None,
            )
        source = args[0]
        assert isinstance(source, Fragment)
        if target is torch.ops.aten.alias.default:
            return source
        if target is torch.ops.aten.view.dtype:
            # Equal-width reinterpretation retains element ownership. Round
            # any widened arithmetic back to its declared dtype first.
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: (
                    f"{self.cast(source.read(coords), source.dtype)}"
                    f".bitcast({self.dtype(fake.dtype)})"
                ),
                dependencies=(source,),
                logical_domain=source.logical_domain,
            )
        if target is _tracing_ops._mask_to:
            source_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])

            def masked(coords: tuple[str, ...]) -> str:
                # A local allocation can have padded physical capacity while
                # retaining its declared logical domain. Its initialized pad
                # values are not necessarily neutral for this reduction.
                masks = list(source.domain(coords)) if source.logical_domain else []
                for coord, size in zip(coords, source_fake.shape, strict=True):
                    bid = self.env.resolve_block_id(size)
                    if bid is None:
                        continue
                    if bid in self.offsets:
                        masks.append(
                            f"(({self.offsets[bid]}) + ({coord}) < ({self.bounds[bid]}))"
                        )
                    elif self.env.block_sizes[bid].reduction:
                        masks.append(
                            f"(({coord}) < ({self.sym(self.env.block_sizes[bid].numel)}))"
                        )
                mask = self.predicate(masks)
                return f"({self.cast(source.read(coords), fake.dtype)} if {mask} else {self.cast(self.scalar(args[1]), fake.dtype)})"

            return Fragment(
                shape,
                fake.dtype,
                masked,
                dependencies=(source,),
                logical_domain=source.logical_domain,
            )
        if target is view_ops.subscript:
            slices = cast("list[object]", args[1])

            def subscript_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                result: list[str] = []
                dim = 0
                source_dim = 0
                for item in slices:
                    if item is None:
                        dim += 1
                    elif isinstance(item, slice):
                        begin = item.start or 0
                        step = item.step or 1
                        result.append(
                            f"({self.scalar(begin)}) + ({coords[dim]}) * ({self.scalar(step)})"
                        )
                        dim += 1
                        source_dim += 1
                    else:
                        position = self.scalar(item)
                        result.append(f"({position}) % {source.shape[source_dim]}")
                        source_dim += 1
                result.extend(coords[dim:])
                return tuple(result)

            def subscript(coords: tuple[str, ...]) -> str:
                return source.read(
                    self.coordinate_locals(subscript_coordinates(coords))
                )

            # ``x[None, :, :]``: new singleton axes and whole slices only keep
            # the staged tile's row order.
            cheap_view = (
                (source.cheap or source.physical_row is not None)
                and all(item is None or item == slice(None) for item in slices)
                and len(shape) >= 2
                and len(source.shape) >= 2
                and slices[-1] == slice(None)
                and (len(slices) < 2 or slices[-2] == slice(None))
            )
            return Fragment(
                shape,
                fake.dtype,
                subscript,
                source.resident,
                (source,),
                logical_domain=lambda coords: source.domain(
                    subscript_coordinates(coords)
                ),
                cheap=cheap_view,
                physical_row=source.physical_row if cheap_view else None,
            )
        if target is torch.ops.aten.permute.default:
            order = cast("list[int]", args[1])
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    tuple(coords[order.index(i)] for i in range(len(order)))
                ),
                source.resident,
                (source,),
                logical_domain=lambda coords: source.domain(
                    tuple(coords[order.index(i)] for i in range(len(order)))
                ),
            )
        if target in (torch.ops.aten.expand.default, torch.ops.aten.clone.default):
            # A broadcast of a staged tile (or of a cheap view of one) keeps
            # the tile's own row order wherever it is not broadcast.
            cheap_view = (
                (source.cheap or source.physical_row is not None)
                and len(shape) >= 2
                and source.shape[-2:] == shape[-2:]
            )
            return Fragment(
                shape,
                fake.dtype,
                source.broadcast,
                source.resident,
                (source,),
                logical_domain=source.broadcast_domain,
                cheap=cheap_view,
                physical_row=source.physical_row if cheap_view else None,
            )
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
        ):
            # Singleton views preserve logical element order. Read through the
            # source recipe so permuted layouts and resident aliases retain
            # their own physical addressing and shared-buffer lifetimes.
            source_live = [i for i, size in enumerate(source.shape) if size != 1]
            target_live = [i for i, size in enumerate(shape) if size != 1]
            structural = [source.shape[i] for i in source_live] == [
                shape[i] for i in target_live
            ]

            def view_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                if structural:
                    # The view only inserts or drops singleton dimensions:
                    # each live coordinate passes through unchanged, with no
                    # flatten/unflatten round trip for the compiler to fold.
                    mapped = ["0"] * len(source.shape)
                    for source_dim, target_dim in zip(
                        source_live, target_live, strict=True
                    ):
                        mapped[source_dim] = coords[target_dim]
                    return tuple(mapped)
                return self.coordinates(self.flatten(coords, shape), source.shape)

            def view_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
                if structural:
                    return source.domain(view_coordinates(coords))
                # Indexing may pad a singleton view without preserving numel.
                # Bound the flat coordinate before unflattening can wrap it.
                return (
                    f"({self.flatten(coords, shape)}) < ({math.prod(source.shape)})",
                    *source.domain(view_coordinates(coords)),
                )

            # A singleton-only view of a staged tile (or of a cheap view of
            # one) whose last two live dimensions are the tile's keeps the
            # tile's row order.
            cheap_view = (
                structural
                and (source.cheap or source.physical_row is not None)
                and len(target_live) >= 2
                and len(source_live) >= 2
                and target_live[-1] == len(shape) - 1
                and source_live[-1] == len(source.shape) - 1
            )
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    self.coordinate_locals(view_coordinates(coords))
                ),
                source.resident,
                (source,),
                logical_domain=view_domain,
                cheap=cheap_view,
                physical_row=source.physical_row if cheap_view else None,
            )
        if target is torch.ops.aten.where.self:
            condition, lhs, rhs = cast("tuple[Fragment, Fragment, Fragment]", args)
            logical_dependencies = tuple(
                value
                for value in (condition, lhs, rhs)
                if value.logical_domain is not None
            )
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: (
                    f"({self.cast(lhs.broadcast(coords), fake.dtype)} if {condition.broadcast(coords)} else {self.cast(rhs.broadcast(coords), fake.dtype)})"
                ),
                dependencies=(condition, lhs, rhs),
                # All operands share the broadcast output domain, regardless of
                # which value the predicate selects. Preserve logical tails for
                # the reduction's neutral mask; load/value masks are separate.
                logical_domain=(
                    lambda coords: tuple(
                        mask
                        for value in logical_dependencies
                        for mask in value.broadcast_domain(coords)
                    )
                )
                if logical_dependencies
                else None,
            )
        raise exc.BackendUnsupported("cute", f"computed fragment operation {target}")


def computed_fragment_supported(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    physical_axes: frozenset[int] = frozenset(),
    owned_iota_axes: frozenset[int] = frozenset(),
    snapshot_owned: bool = False,
    bounded_gather_owned: bool = False,
    allow_unbound: bool = False,
) -> bool:
    """Structural root ownership shared by search discovery and code generation.

    Exact local sizes, strides, scalar predicates and shared capacity remain
    configuration-dependent checks in the emitter.
    """
    # Jagged loops carry per-parent bounds in their first captured tensor.
    # This owner currently tracks only scalar loop ends; preserve the ordinary
    # jagged lowering instead of treating the maximum end as every row's end.
    if any(
        isinstance(info, ForLoopGraphInfo)
        and any(env.is_jagged_tile(bid) for bid in info.block_ids)
        for info in graphs
    ):
        return False
    while_calls = {
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is _tracing_ops._while_loop
    }
    if while_calls:
        from .gather_domains import loop_domain_facts
        from .resident_while import resident_while_plan

        try:
            plans = [resident_while_plan(call, graphs) for call in while_calls]
            mutable_while = any(plan.local_allocations for plan in plans)
            needs_bounded_gather = any(
                not plan.local_allocations and not plan.readonly_control
                for plan in plans
            )
            for plan in plans:
                if plan.readonly_control:
                    from .uniform_region_tree import uniform_output_stores_are_writeonly
                    from .uniform_region_tree import uniform_readonly_regions

                    if not uniform_output_stores_are_writeonly(
                        env,
                        graphs,
                        uniform_readonly_regions(graphs),
                        allow_unbound=allow_unbound,
                    ):
                        return False
            if mutable_while:
                prove_local_atomics(graphs)
        except exc.InvalidConfig:
            return False

        if (
            (not bounded_gather_owned and needs_bounded_gather)
            or loop_domain_facts(
                env, graphs, allow_unbound=allow_unbound
            ).resident_whiles
            != while_calls
            or (
                not mutable_while
                and any(
                    node.target in atomic_ops.ATOMIC_OPS
                    for info in graphs
                    for node in info.graph.nodes
                )
            )
        ):
            return False
    # If-only frames also require the complete bound output proof: they need
    # not have a host load or a while-domain check to trigger readonly validation.
    if any(
        node.target is memory_ops.store
        for info in graphs
        if not isinstance(info, RootGraphInfo)
        for node in info.graph.nodes
    ):
        from .uniform_region_tree import uniform_local_regions
        from .uniform_region_tree import uniform_output_stores_are_writeonly

        if any(
            len(node.args) != 4 or node.kwargs
            for info in graphs
            for node in info.graph.nodes
            if node.target is memory_ops.store
        ):
            return False
        try:
            tree = uniform_local_regions(graphs)
        except exc.InvalidConfig:
            pass  # Preserve the independently proved legacy terminal path.
        else:
            if not uniform_output_stores_are_writeonly(
                env, graphs, tree, allow_unbound=allow_unbound
            ):
                return False
    graph_by_id = {info.graph_id: info for info in graphs}
    independent_reductions = independent_reduction_coordinates(env, graphs)
    free_reductions = {
        node
        for node in free_iota_reductions(env, graphs)
        if node.meta["lowering"].block_index not in owned_iota_axes
    }
    captured_reductions = captured_reduction_coordinates(
        env, graphs, physical_axes=physical_axes
    )
    local_allocations = local_atomic_allocations(graphs)
    if not local_allocations and any(
        node.target is atomic_ops.atomic_add
        for info in graphs
        for node in info.graph.nodes
    ):
        return False

    def configured_axis(size: int | torch.SymInt) -> bool:
        if isinstance(size, int):
            return size > 0
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if expr.is_number:
            return int(expr) > 0
        return all(env.get_block_id(symbol) is not None for symbol in expr.free_symbols)

    grid_block_ids = {*env.config_spec.grid_block_ids}

    def resident_state(value: object) -> bool:
        # Per-row state such as an online-softmax max over ``hl.tile([B, L])``
        # has one element per grid coordinate, like a 1-D carry: the ordinary
        # lowering already owns each element. Only carries with a non-grid axis
        # are resident matrices that need independent coordinates.
        return (
            isinstance(value, torch.Tensor)
            and value.ndim > 1
            and not all(
                env.resolve_block_id(size) in grid_block_ids for size in value.shape
            )
        )

    carried: set[Node] = set()
    captures: dict[Node, Node] = {}
    parents = control_flow_parent_entries(graphs)
    for info in graphs:
        if info.graph_id not in parents:
            continue
        node, argument_slot = parents[info.graph_id]
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        captures.update(
            zip(
                placeholders,
                cast("list[Node]", node.args[argument_slot]),
                strict=True,
            )
        )
        if _tracing_ops.is_for_loop_target(node.target):
            for slot in _loop_carry_slots(node).values():
                if resident_state(placeholders[slot].meta.get("val")):
                    carried.add(placeholders[slot])

    def reads_carry(node: Node) -> bool:
        pending = [node]
        seen: set[Node] = set()
        while pending:
            source = pending.pop()
            if source in seen:
                continue
            seen.add(source)
            if source in carried:
                return True
            if source.target in (memory_ops.load, _tracing_ops._host_tensor):
                continue
            if source in captures:
                pending.append(captures[source])
            else:
                pending.extend(source.all_input_nodes)
        return False

    # Only roots needing independent static coordinates or resident computed
    # contractions/scans use this fallback. Explicitly tiled contractions and
    # native collective configurations keep their existing code generation.
    def computed(node: Node) -> bool:
        if node.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.prims.convert_element_type.default,
            view_ops.subscript,
        ):
            return isinstance(node.args[0], Node) and computed(node.args[0])
        return node.target not in (memory_ops.load, _tracing_ops._host_tensor)

    def repeated_axes(value: torch.Tensor) -> bool:
        sizes = [
            size._sympy_() if isinstance(size, torch.SymInt) else size
            for size in value.shape
        ]
        sizes = [size for size in sizes if size != 1]
        return len(sizes) != len(set(sizes))

    selections = [
        node
        for info in graphs
        for node in info.graph.nodes
        if is_ordered_selection(node)
    ]

    def needs_coordinates(node: Node) -> bool:
        if is_ordered_selection(node):
            source_node = cast("Node", node.args[0])
            source = source_node.meta.get("val")
            if len(selections) > 1 or (
                isinstance(source, torch.Tensor) and source.ndim > 2
            ):
                return True
            if (
                not isinstance(source, torch.Tensor)
                or source.ndim != 2
                or not computed(source_node)
            ):
                return False
            from .topk import match_topk_root

            # A computed row may lack the scalar sortable-load recipe while
            # still having complete static coordinates. Preserve the existing
            # optimized owner's structural match, including its later alias
            # and layout checks; only an unmatched root needs this fallback.
            return (
                match_topk_root(
                    graphs,
                    noncanonical_block_ids=HostFunction.current().device_ir.noncanonical_task_origin_block_ids,
                )
                is None
            )
        if (
            node in local_allocations
            or node in independent_reductions
            or node in captured_reductions
            or node in free_reductions
        ):
            return True
        if node.target is _tracing_ops._host_tensor:
            return False
        if node.target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            source = node.args[0]
            assert isinstance(source, Node)
            before = cast("torch.Tensor", source.meta["val"])
            after = cast("torch.Tensor", node.meta["val"])
            # Factoring a full static axis requires independent coordinates for
            # every factor. A scalar recipe for its producer cannot represent
            # both the original flat consumer and the factored consumer.
            static_before = [
                size for size in before.shape if isinstance(size, int) and size != 1
            ]
            static_after = [
                size for size in after.shape if isinstance(size, int) and size != 1
            ]
            symbolic_before = [
                size._sympy_()
                for size in before.shape
                if isinstance(size, torch.SymInt)
            ]
            symbolic_after = [
                size._sympy_() for size in after.shape if isinstance(size, torch.SymInt)
            ]
            if (
                static_before != static_after
                and math.prod(static_before) == math.prod(static_after)
                and symbolic_before == symbolic_after
            ):
                return True
        if node.target is scan_ops._associative_scan:
            source = node.args[1]
            return isinstance(source, Node) and computed(source)
        if isinstance(node.meta.get("lowering"), ReductionLowering):
            if any(reads_carry(source) for source in node.all_input_nodes):
                return True
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and node.target in (
            torch.ops.aten.gt.Tensor,
            torch.ops.aten.ge.Tensor,
            torch.ops.aten.lt.Tensor,
            torch.ops.aten.le.Tensor,
            torch.ops.aten.eq.Tensor,
            torch.ops.aten.ne.Tensor,
        ):
            if repeated_axes(value):
                return True
        if node.target is not matmul_ops.dot:
            return False
        lhs, rhs = node.args[:2]
        assert isinstance(lhs, Node) and isinstance(rhs, Node)
        left, right = lhs.meta["val"], rhs.meta["val"]
        bid = env.resolve_block_id(left.shape[-1])
        static_k = isinstance(left.shape[-1], int) and left.shape[-1] > 1
        if bid is not None:
            static_k = env.block_sizes[bid].reduction
        return (
            left.dtype == right.dtype
            and left.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and static_k
            and (
                # A single scalar block coordinate cannot represent two axes
                # within the same operand. This also applies to captured full
                # matrix loads, not just matrices created by comparisons.
                repeated_axes(left)
                or repeated_axes(right)
                # Resident state cannot be reconstructed from global loads for
                # each contraction coordinate, including after a dtype cast.
                or reads_carry(lhs)
                or reads_carry(rhs)
                # Existing scalar/native half-precision recipes retain their
                # paths when their operands have independent coordinates.
                or (
                    left.dtype == torch.float32
                    and (
                        computed(lhs)
                        or computed(rhs)
                        or left.ndim > 2
                        or right.ndim > 2
                    )
                )
            )
        )

    if not (snapshot_owned or bounded_gather_owned) and not any(
        needs_coordinates(node) for info in graphs for node in info.graph.nodes
    ):
        return False
    supported = {
        _tracing_ops._if,
        _tracing_ops._while_loop,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._constant_tensor,
        _tracing_ops._mask_to,
        _tracing_ops._for_loop,
        _tracing_ops._for_loop_step,
        memory_ops.load,
        memory_ops.store,
        atomic_ops.atomic_add,
        matmul_ops.dot,
        scan_ops._associative_scan,
        inline_asm_ops.inline_asm_elementwise,
        creation_ops.full,
        torch.ops.aten.full.default,
        view_ops.subscript,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_id,
        tile_ops.tile_index,
        operator.getitem,
        torch.ops.aten.sym_size.int,
        torch.ops.prims.iota.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.where.self,
        torch.ops.aten.view.dtype,
        torch.ops.aten.alias.default,
        torch.ops.aten.topk.default,
        torch.ops.aten.sort.default,
        torch.ops.aten.sort.stable,
        torch.ops.aten.gather.default,
    }
    if any(
        node.op == "call_function"
        and node.target not in supported
        and not isinstance(
            node.meta.get("lowering"),
            (PointwiseLowering, SympyExprLowering, ReductionLowering),
        )
        for info in graphs
        for node in info.graph.nodes
    ):
        return False
    for info in graphs:
        for node in info.graph.nodes:
            if is_ordered_selection(node):
                arguments = ordered_selection_args(node)
                if arguments is None:
                    return False
                source_node, k, dim, *_ = arguments
                source = source_node.meta.get("val")
                width = _static_selection_extent(env, source_node)
                if k is None:
                    k = width
                if (
                    not isinstance(source, torch.Tensor)
                    or source.ndim < 1
                    or source.dtype
                    not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.int32,
                        torch.int64,
                    )
                    or dim not in (-1, source.ndim - 1)
                    or width is None
                    or not isinstance(k, int)
                    or not 0 < k <= width
                    or not all(configured_axis(size) for size in source.shape)
                ):
                    return False
            if node.target is torch.ops.aten.gather.default:
                from .bounded_gather import prove_gather

                if (
                    bounded_gather_owned
                    and prove_gather(
                        env, node, graphs=graphs, allow_unbound=allow_unbound
                    )
                    is not None
                ):
                    continue
                source, dim, index = node.args[:3]
                if not isinstance(source, Node) or not isinstance(index, Node):
                    return False
                sf, ix = source.meta.get("val"), index.meta.get("val")
                if (
                    not isinstance(sf, torch.Tensor)
                    or not isinstance(ix, torch.Tensor)
                    or sf.ndim != ix.ndim
                    or dim not in (-1, sf.ndim - 1)
                    or index.target is not operator.getitem
                    or index.args[1] != 1
                    or not isinstance(index.args[0], Node)
                    or not is_ordered_selection(index.args[0])
                ):
                    return False
                original_node = cast("Node", index.args[0].args[0])
                original = original_node.meta["val"]
                original_shape = _fragment_logical_shape(env, original_node)
                gathered_shape = _fragment_logical_shape(env, source)
                if (
                    original_shape is None
                    or gathered_shape is None
                    or tuple(map(sympy.expand, original_shape))
                    != tuple(map(sympy.expand, gathered_shape))
                ):
                    return False
                if (
                    original.ndim != sf.ndim
                    or any(
                        not env.known_equal(a, b)
                        for a, b in zip(original.shape, sf.shape, strict=True)
                    )
                    or any(
                        not env.known_equal(a, b)
                        for a, b in zip(ix.shape[:-1], sf.shape[:-1], strict=True)
                    )
                ):
                    return False
            if node.target is torch.ops.aten.view.dtype:
                source = cast("Node", node.args[0]).meta["val"]
                target = node.meta["val"]
                if (
                    not isinstance(source, torch.Tensor)
                    or not isinstance(target, torch.Tensor)
                    or source.dtype == torch.bool
                    or target.dtype == torch.bool
                    or source.dtype.itemsize != target.dtype.itemsize
                ):
                    return False
            if node.target is inline_asm_ops.inline_asm_elementwise:
                if (
                    node.args[4] is not True
                    or node.args[5] != 1
                    or not isinstance(node.args[3], torch.dtype)
                    or not isinstance(node.meta.get("val"), torch.Tensor)
                ):
                    return False
            if node.target is scan_ops._associative_scan:
                if node.args[4]:
                    return False
                value = node.meta["val"]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype
                    not in (
                        torch.int8,
                        torch.uint8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                    )
                    or value.ndim == 0
                ):
                    return False
                if not -value.ndim <= cast("int", node.args[2]) < value.ndim:
                    return False
                # Every local axis must be constant or have a configured block
                # owner. Config-dependent values and shared capacity are checked
                # again by the emitter, never replaced by a shape-specific mode.
                if not all(configured_axis(size) for size in value.shape):
                    return False
                helper = graph_by_id[cast("int", node.args[0])]
                assert isinstance(helper, HelperFunctionGraphInfo)
                if not additive_scan_helper(helper):
                    return False
            lowering = node.meta.get("lowering")
            if isinstance(
                lowering, ReductionLowering
            ) and lowering.reduction_type not in (
                "sum",
                "max",
                "min",
                "prod",
                "argmin",
                "argmax",
            ):
                return False
            if _tracing_ops.is_for_loop_target(node.target) and any(
                not isinstance(value, torch.Tensor) for value in node.meta["val"]
            ):
                return False
    return True


def computed_fragment_specialization_facts(
    env: CompileEnvironment, ir: DeviceIR
) -> frozenset[CompilerHeuristicSpecializationFact]:
    """Guard shape hints used as capacities before bound-cache publication."""
    if env.settings.static_shapes or ir.host_function is None:
        return frozenset()
    with ir.host_function:
        if computed_fragment_supported(
            env, ir.graphs
        ) or independent_reduction_coordinates(
            env, ir.graphs, include_rolled_outputs=True
        ):
            return frozenset({"input_tensor_metadata"})
    return frozenset()


def codegen_computed_fragment_root(cg: GenerateAST) -> bool:
    """Own static axes and complete contractions that scalar indexing cannot express."""
    from ..autotuner_heuristics.cute_fragment_reduction import (
        fragment_warp_reduction_supported,
    )
    from ..autotuner_heuristics.cute_fragment_warp_scan import active_warp_scan_roots

    env = CompileEnvironment.current()
    if env.backend_name != "cute":
        return False
    root = cg.current_root_graph_info
    assert root is not None
    threads_required = (
        cg.device_function.config.get("cute_fragment_threads", 128) != 128
        and root.graph_id in env.config_spec.cute_fragment_thread_root_ids
    )
    published_scalars_required = (
        cg.device_function.config.get("cute_fragment_published_scalars", False)
        and root.graph_id in env.config_spec.cute_fragment_published_scalar_root_ids
    )
    pure_regions_required = (
        cg.device_function.config.get("cute_fragment_pure_producer_regions", False)
        and root.graph_id
        in env.config_spec.cute_fragment_pure_producer_regions_root_ids
    )
    producer_cache_required = (
        cg.device_function.config.get("cute_fragment_producer_cache", False)
        and root.graph_id in env.config_spec.cute_fragment_producer_cache_root_ids
    )
    warp_scan_required = cg.device_function.config.get(
        "cute_fragment_warp_scan", False
    ) and root.graph_id in active_warp_scan_roots(env, cg.device_function.config)
    snapshots_required = (
        cg.device_function.config.get("cute_fragment_register_snapshots", False) is True
        and root.graph_id in env.config_spec.cute_fragment_register_snapshots_root_ids
    )
    topk_network_required = (
        cg.device_function.config.get("cute_fragment_topk_network", False)
        and root.graph_id in env.config_spec.cute_fragment_topk_network_root_ids
    )
    bounded_gather_required = (
        cg.device_function.config.get("cute_fragment_bounded_gather", False) is True
        and root.graph_id in env.config_spec.cute_fragment_bounded_gather_root_ids
    )
    register_loads_required = (
        cg.device_function.config.get("cute_fragment_register_loads", False)
        and root.graph_id in env.config_spec.cute_fragment_register_load_root_ids
    )
    warp_results_required = (
        cg.device_function.config.get("cute_fragment_warp_results", False)
        and root.graph_id in env.config_spec.cute_fragment_warp_result_root_ids
    )
    # These options require the fragment owner's coordinate layout instead of
    # borrowing the ordinary scalar/iota layout. Warp-result publication also
    # owns captured reductions, but does not change free-iota ownership.
    coordinate_schedule_required = (
        threads_required
        or register_loads_required
        or producer_cache_required
        or pure_regions_required
        or published_scalars_required
        or topk_network_required
        or bounded_gather_required
        or snapshots_required
        or warp_scan_required
    )
    capture_schedule_required = coordinate_schedule_required or warp_results_required
    local_required = bool(local_atomic_allocations(cg.host_function.device_ir.graphs))
    owned_iota_axes = (
        frozenset()
        if coordinate_schedule_required
        or cg.device_function.config.get("cute_fragment_reduction", "serial")
        != "serial"
        else owned_iota_reduction_axes(cg)
    )
    free_required = any(
        node.graph is cg.host_function.device_ir.graphs[root.graph_id].graph
        and node.meta["lowering"].block_index not in owned_iota_axes
        for node in free_iota_reductions(env, cg.host_function.device_ir.graphs)
    )
    physical_axes = (
        frozenset()
        if capture_schedule_required
        or cg.device_function.config.get("cute_fragment_scan", "serial") != "serial"
        or cg.device_function.config.get("cute_fragment_reduction", "serial")
        != "serial"
        else physical_capture_axes(cg)
    )
    captured_required = bool(
        captured_reduction_coordinates(
            env,
            cg.host_function.device_ir.graphs,
            root_graph_id=root.graph_id,
            physical_axes=physical_axes,
        )
    )
    fragment_schedule_required = (
        (
            cg.device_function.config.get("cute_fragment_private_scalar_loops", False)
            and root.graph_id
            in env.config_spec.cute_fragment_private_scalar_loop_root_ids
        )
        or capture_schedule_required
        or (
            cg.device_function.config.get("cute_fragment_scan", "serial")
            == "cooperative"
            and root.graph_id in env.config_spec.cute_fragment_scan_root_ids
        )
        or (
            cg.device_function.config.get("cute_fragment_reduction", "serial")
            in ("warp", "cta", "cta_unroll")
            and root.graph_id in env.config_spec.cute_fragment_reduction_root_ids
        )
    )
    state = cg.device_function.cute_state
    if any(
        plan is not None
        for plan in (
            state.block_scaled_plan,
            state.topk_plan,
            state.radix_selection_plan,
            state.chunk_prepare_plan,
            state.chunk_recurrence_plan,
            state.single_token_rank1_plan,
            state.split_single_token_rank1_plan,
            state.fixed_token_rank1_plan,
            state.mma_dag_plan,
            state.attention_flash_block_ids,
            state.attention_flash_bwd_block_ids,
        )
    ):
        # Pre-codegen selected a complete root owner. It must execute its
        # late layout/alias checks; an implicit coordinate requirement must
        # not bypass those checks by replacing its producer with fragments.
        if local_required or fragment_schedule_required:
            raise exc.InvalidConfig(
                "a planned root cannot share a computed fragment schedule"
            )
        return False
    if free_required:
        # A new complete owner changes the placement of global stores relative
        # to vector reductions. Separate destinations need a disjointness
        # proof; declining into the unowned scalar iota path would lose the
        # complete reduction. Repeated stores through one tensor retain their
        # existing ordered semantics.
        destinations = {
            cg.device_function.tensor_arg(target.meta["val"]).name
            for info in cg.host_function.device_ir.graphs
            for node in info.graph.nodes
            if node.target is memory_ops.store
            and isinstance(target := node.args[0], Node)
            and isinstance(target.meta.get("val"), torch.Tensor)
            and target.meta["val"] in HostFunction.current().tensor_to_origin
        }
        disjoint = cg.device_function.proven_disjoint_tensor_pairs()
        if any(
            frozenset((left, right)) not in disjoint
            for left in destinations
            for right in destinations
            if left < right
        ):
            if local_required or fragment_schedule_required:
                raise exc.InvalidConfig(
                    "fragment store destinations require disjoint storage"
                )
            raise exc.BackendUnsupported(
                "cute", "fragment store destinations require disjoint storage"
            )
    if (
        cg.device_function.config.get(
            "cute_affine_scan_schedule", DIRECT_AFFINE_ORDINARY_SCHEDULE
        )
        != DIRECT_AFFINE_ORDINARY_SCHEDULE
    ):
        # The direct affine schedule must first emit its ordinary producer,
        # then prove and replace that entire region. Implicit free-iota and
        # capture ownership must not erase the producer before that proof.
        if (
            local_required
            or fragment_schedule_required
            or cg.device_function.config.get("cute_reduction_schedule", "scalar")
            != "scalar"
        ):
            raise exc.InvalidConfig(
                "direct affine schedules cannot share a computed fragment root"
            )
        return False
    if cg.device_function.config.get("cute_reduction_schedule", "scalar") != "scalar":
        # An explicit resident schedule owns its rectangular carries and
        # performs its own complete proof during materialization. Replacing
        # its scalar producer here erases the structure that proof consumes.
        if local_required or free_required or fragment_schedule_required:
            raise exc.InvalidConfig(
                "resident reduction schedules cannot share a computed fragment root"
            )
        return False
    if (
        len(cg.host_function.device_ir.root_ids) != 1
        or cg.device_function.config.get("cute_collective_mma", False)
        or cg.device_function.config.get("cute_register_chain", False)
    ):
        if local_required:
            raise exc.InvalidConfig(
                "CTA-local atomics require a complete fragment root"
            )
        if free_required:
            raise exc.InvalidConfig(
                "free iota reductions require a complete fragment root"
            )
        if captured_required or capture_schedule_required:
            raise exc.InvalidConfig(
                "captured full reductions require a computed fragment root"
                if captured_required
                else "CTA threads/register loads require a computed fragment root"
            )
        return False
    graphs = (
        cg.host_function.device_ir.build_codegen_graphs(
            cg.device_function.config, roll_reductions=False
        )
        if captured_required
        or local_required
        or free_required
        or snapshots_required
        or bounded_gather_required
        else cg.codegen_graphs
    )

    scan_required = warp_scan_required or (
        cg.device_function.config.get("cute_fragment_scan", "serial") == "cooperative"
        and root.graph_id
        in CompileEnvironment.current().config_spec.cute_fragment_scan_root_ids
    )
    reduction_required = (
        cg.device_function.config.get("cute_fragment_reduction", "serial")
        in ("warp", "cta", "cta_unroll")
        and root.graph_id
        in CompileEnvironment.current().config_spec.cute_fragment_reduction_root_ids
    )

    def decline() -> bool:
        if free_required:
            raise exc.InvalidConfig(
                "free iota reductions require a supported complete fragment root"
            )
        if local_required:
            raise exc.InvalidConfig(
                "CTA-local atomics require a supported complete fragment root"
            )
        if (
            captured_required
            or scan_required
            or reduction_required
            or capture_schedule_required
        ):
            raise exc.InvalidConfig(
                "captured full reductions require a computed fragment root"
                if captured_required
                else "cooperative scan requires a computed fragment root"
                if scan_required
                else "warp reduction requires a computed fragment root"
                if reduction_required
                else "CTA threads/register loads require a computed fragment root"
            )
        return False

    if scan_required and not any(
        node.target is scan_ops._associative_scan
        for info in graphs
        for node in info.graph.nodes
    ):
        return decline()
    if reduction_required and not any(
        fragment_warp_reduction_supported(node)
        for info in graphs
        for node in info.graph.nodes
    ):
        return decline()
    if not computed_fragment_supported(
        CompileEnvironment.current(),
        graphs,
        physical_axes=physical_axes,
        owned_iota_axes=owned_iota_axes,
        snapshot_owned=snapshots_required,
        bounded_gather_owned=bounded_gather_required,
    ):
        return decline()
    # This owner implements configured reduction tiling directly. Keep the
    # logical producer graph, including every other codegen transformation,
    # rather than applying scalar graph rolling before fragment ownership.
    if (
        not captured_required
        and not local_required
        and not free_required
        and not snapshots_required
        and not bounded_gather_required
    ):
        graphs = cg.host_function.device_ir.build_codegen_graphs(
            cg.device_function.config, roll_reductions=False
        )
    if not computed_fragment_supported(
        CompileEnvironment.current(),
        graphs,
        physical_axes=physical_axes,
        owned_iota_axes=owned_iota_axes,
        snapshot_owned=snapshots_required,
        bounded_gather_owned=bounded_gather_required,
    ):
        return decline()
    root = graphs[root.graph_id]
    compiler = FragmentCompiler(cg, graphs)
    if compiler.df.config.get("cute_fragment_warp_producer_regions", False):
        from .warp_producer_regions import WarpProducerRecorder
        from .warp_producer_regions import producer_prefix

        plan = producer_prefix(root.graph, env, graphs, compiler.shape)
        if plan is not None:
            compiler.warp_producer_recorder = WarpProducerRecorder(compiler, plan)
    if (
        pure_regions_required
        and not pure_producer_plan(root.graph, env, shape=compiler.shape).lazy
    ):
        raise exc.InvalidConfig(
            "pure producer regions require a current same-owner publication"
        )
    compiler.local_allocations = prove_local_atomics(graphs)
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is _tracing_ops._if:
                predicate = node.args[0]
                value = (
                    predicate.meta["val"] if isinstance(predicate, Node) else predicate
                )
                if (
                    isinstance(value, torch.Tensor)
                    and math.prod(compiler.shape(value.shape)) != 1
                ):
                    return decline()
    for info in graphs:
        for node in info.graph.nodes:
            lowering = node.meta.get("lowering")
            if not isinstance(lowering, (PointwiseLowering, ReductionLowering)):
                continue
            ranges = lowering.buffer.data.ranges
            if isinstance(lowering, ReductionLowering):
                assert isinstance(lowering.buffer.data, Reduction)
                ranges = [*ranges, *lowering.buffer.data.reduction_ranges]
            if any(compiler.static_extent(dim) is None for dim in ranges):
                return decline()
            for source in node.all_input_nodes:
                value = source.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    if isinstance(
                        value,
                        (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool),
                    ):
                        continue
                    return decline()
                shape = tuple(compiler.static_extent(dim) for dim in value.shape)
                strides = tuple(compiler.static_extent(dim) for dim in value.stride())
                if any(dim is None for dim in (*shape, *strides)):
                    # Ordinary lowering can retain host constexpr dimensions.
                    # This owner requires proved capacities and coordinate strides.
                    return decline()
                span = 1
                for stride, size in sorted(
                    zip(
                        cast("tuple[int, ...]", strides),
                        cast("tuple[int, ...]", shape),
                        strict=True,
                    )
                ):
                    if size <= 1 or stride == 0:
                        continue
                    if stride < span:
                        return decline()
                    span += (size - 1) * stride
    grid = cg.current_grid_state
    assert grid is not None
    if local_required:
        if any(
            cg.device_function.resolved_block_size(bid) != 1
            for bid in grid.block_id_to_info
        ):
            raise exc.InvalidConfig("CTA-local atomics require scalar grid owners")
    compiler.begin()
    for bid, info in grid.block_id_to_info.items():
        compiler.offsets[bid] = grid.strategy.grid_origin_var(bid)
        compiler.bounds[bid] = cast("str", info.end_var_name)
    body: list[ast.AST] = []
    with cg.set_statements(body):
        compiler.emit(f"{compiler.thread} = cutlass.Int32(cute.arch.thread_idx()[0])")
        compiler.stage_slabs()
        compiler.graph(root.graph, {})
        compiler.flush_pending_barrier()
        # Root lifetime can be nested in an outer generated grid loop.
        compiler.synchronize_local_atomics()
    if compiler.warp_producer_recorder is not None:
        compiler.warp_producer_recorder.lower(body)
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(compiler.env.device)
    if not capacity and any(
        node.target is _tracing_ops._while_loop
        for info in graphs
        for node in info.graph.nodes
    ):
        raise exc.InvalidConfig(
            "resident while requires a known shared memory capacity"
        )
    if local_required and not capacity:
        raise exc.InvalidConfig(
            "CTA-local atomics require a known shared memory capacity"
        )
    if producer_cache_required and not capacity:
        raise exc.InvalidConfig(
            "shared producer caches require a known shared memory capacity"
        )
    if capacity and compiler.smem_bytes > capacity:
        if (
            local_required
            or free_required
            or fragment_schedule_required
            or any(
                node.target is _tracing_ops._while_loop
                for info in graphs
                for node in info.graph.nodes
            )
        ):
            # These roots have no other owner: the config is invalid.
            raise exc.InvalidConfig(
                f"computed fragments need {compiler.smem_bytes} shared bytes, exceeding {capacity}"
            )
        # The fragment program materializes every row of this root in shared
        # memory, so a wide static axis can exceed the CTA capacity for every
        # tile size (layer_norm backward at N=8192 needs 245 KB with one row).
        # The implicit preference for this owner (captured reductions) is a
        # performance choice; the ordinary scalar lowering owns such roots
        # with lane/thread accumulation, so decline instead of failing the
        # default and every other scalar-schedule config.  Nothing emitted
        # above escaped the scratch ``body``: the compiler only allocated
        # variable names and kernel tensor arguments.  A root only this owner
        # can lower (repeated block ids, static matrices) makes the generic
        # lowering refuse; GenerateAST reports that as this config failure.
        cg.device_function.cute_state.fragment_capacity_decline = (
            compiler.smem_bytes,
            capacity,
        )
        return False
    if published_scalars_required:
        from .published_scalars import PublishedScalarRequest

        compiler.df.cute_state.published_scalar_requests.append(
            PublishedScalarRequest(
                frozenset(name for name, _, _ in compiler.buffers),
                compiler.thread,
                compiler.threads,
            )
        )
    if compiler.df.config.get("cute_fragment_register_producers", False):
        from .register_producers import RegisterProducerRequest

        compiler.df.cute_state.register_producer_requests.append(
            RegisterProducerRequest(
                frozenset(name for name, _, _ in compiler.buffers),
                tuple(compiler.register_snapshot_buffers.items()),
                compiler.thread,
                compiler.threads,
            )
        )
    cg.device_function.cute_state.owned_root_block_dims = (compiler.threads, 1, 1)
    compiler.emit_mma_preamble()
    for statement in (*compiler.allocations(), *body):
        cg.add_statement(statement)
    return True
