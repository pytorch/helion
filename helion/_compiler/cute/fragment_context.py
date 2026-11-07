"""Shared state and primitives of the computed-fragment emitters.

``FragmentContext`` holds the per-root compile state (thread count, shared-memory
allocator, scopes, the active snapshot / local-atomic owners) and the primitive
operations every emitter composes: symbol rendering, element loops over a
staged tile, allocation and materialization of fragments, predicates and
barriers.  The ``Fragment`` recipe type and the logical-shape proof live here
too so the emitter mixins and their analyses import them without importing
the root owner (``computed_fragment``)."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
import math
import operator
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
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
from ..host_function import HostFunction
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileEndOrigin
from ..variable_origin import TileIdOrigin
from .dead_zero_atomics import dead_zero_atomic_results
from .fragment_storage import aligned_shared_bytes
from .fragment_storage import configured_fragment_expr
from .fragment_storage import metadata_guarded
from .local_atomic_registers import MAX_SLOTS
from .local_atomic_registers import local_atomic_register_chains
from .private_scalar_loops import private_scalar_loop_nodes
from .register_snapshots import SnapshotBinding
from .register_snapshots import SnapshotOwner
from .register_snapshots import snapshot_chains
from .register_snapshots import snapshot_logical_shape
from .selection.ordered_selection import is_ordered_selection
from .selection.ordered_selection import ordered_selection_args
from .warp_results import warp_result_chain

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Iterable
    from collections.abc import Mapping

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .atomic_consumer_fusion import AtomicConsumerRegion
    from .fragment_expression import FragmentExpression
    from .fragment_slab_loads import ScalarLoad
    from .fragment_slab_loads import SlabGroup
    from .scan_analysis import CompletedScan
    from .warp_producer_regions import WarpProducerRecorder


# The most slots per thread an element loop walks as a constexpr unroll.
UNROLLED_SLOTS = 16


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
    """The host store that consumes an mma result straight from the lane
    registers (``emit_mma``): ``memory`` fills in the window's address
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


class FragmentContext:
    """Per-root compile state plus the primitives the fragment emitters share."""

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

    def graph(self, graph: torch.fx.Graph, values: dict[Node, object]) -> object:
        """Walk a (nested) graph, binding every node's lowering into ``values``.

        The control-flow emitters call back into the walk for their bodies;
        ``FragmentCompiler`` provides it.
        """
        raise NotImplementedError

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
        # ``cute_fragment_split_k``: idle warps of an mma phase take a
        # share of K (partials summed at the read).
        self.split_k = bool(self.df.config.get("cute_fragment_split_k", False))
        # ``cute_fragment_mma_rows``: the mma lanes walk whole tile rows
        # and evaluate each row's A words once for all of its column tiles.
        self.mma_rows = bool(self.df.config.get("cute_fragment_mma_rows", False))
        # A load phase emitted without its trailing barrier: the next node
        # that is not another independent host load issues it.
        self.pending_barrier = False
        # An ``mma.sync`` phase was emitted: the generated module needs
        # the helpers' import, added once the root is committed (a capacity
        # decline of the scratch body must leave the module untouched).
        self.wants_mma_preamble = False
        # Host stores already written by the mma lanes of the dot they
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
