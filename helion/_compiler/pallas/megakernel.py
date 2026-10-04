"""Lower a multi-root Pallas kernel to one sequential program (the megakernel).

All top-level loops run on one TensorCore in a single program: the launch grid
is ``(1,)`` and each root walks its tiles in a ``pl.loop``.  Program order
satisfies every cross-root RAW, WAR and WAW hazard, so the tile dependency
graph is only used to classify storage, never to order work.

Read-only tiles loaded inside the inner loops (the weights) are streamed from
HBM through global DMA rings, one per tile shape, each shared by every loop of
every root that loads that shape: the stream runs ahead across loop and root
boundaries, so the next root's first weights are in flight while the current
root finishes.
"""

from __future__ import annotations

import ast
import bisect
import collections
import dataclasses
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import exc
from ..._utils import is_scalar_index
from ...language import memory_ops
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _for_loop_step
from ...language._tracing_ops import _host_tensor
from ...language._tracing_ops import _if
from ...language._tracing_ops import _new_var
from ...language._tracing_ops import _phi
from ...language._tracing_ops import _while_loop
from ...language.atomic_ops import ATOMIC_OPS
from ...language.distributed_ops import start_async_remote_copy_descriptor
from ...runtime.pallas.launcher import _get_vmem_limit_bytes
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..compile_environment import LoopSpecBlockSizeSource
from ..device_function import DeviceFunction
from ..device_function import PallasMemorySpace
from ..host_function import HostFunction
from ..indexing_strategy import subscript_index_scale
from ..indexing_strategy import subscript_tile_info
from ..tile_dependency import StorageRole
from ..tile_dependency import owner_roots_by_graph_id
from ..tile_strategy import DeviceGridState
from ..tile_strategy import ForiLoopState
from ..tile_strategy import PersistentReductionState
from .dma import DmaResources
from .internal_scratch import accessed_storages
from .internal_scratch import returned_name_dependencies
from .internal_scratch import top_level_empty_names
from .lane_dense import lane_dense_perm
from .lane_dense import logical_load
from .lane_dense import lowered_transpose
from .lane_dense import minor_swap
from .lane_dense import physical_order
from .lane_dense import row_packing
from .lane_dense import row_word
from .lane_dense import unpack_row
from .plan_tiling import ArbitraryIndexPattern
from .vmem_scalar_load import VmemScalarLoad
from .vmem_scalar_load import emit_vmem_scalar_load

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ..device_ir import DeviceIR
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import GraphInfo
    from ..device_ir import HostLoopInfo
    from ..inductor_lowering import CodegenState
    from ..tile_dependency import TaskAxis
    from ..tile_dependency import TileAccess
    from ..tile_dispatch import TileStrategyDispatch


@dataclasses.dataclass
class MegakernelPlan:
    """Per-config decisions for one sequential-roots program."""

    # (storage id, dim) -> logical size, for every dim whose ref is larger
    # than the tensor (launcher padding or padded scratch).
    padded_dims: dict[tuple[int, int], int]
    # Zero the padded slab of each intermediate scratch before root 0.
    scratch_zeroing: list[str]
    # Host names of the intermediates placed in scratch.
    host_scratch_names: list[str]
    # One weight ring per streamed tile shape.
    stream: list[Ring]
    # Storages of the tensors kept in HBM (``plan_hbm_resident``).
    hbm_resident: frozenset[int]
    # Ring window -> the integer-indexed dims of the load it serves, which
    # the window does not have.
    window_scalar_dims: dict[str, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    # Storage -> the row read of a read-only tensor kept in HBM that starts
    # at kernel start (``_plan_early_row_reads``).
    early_row_reads: dict[int, RowRead] = dataclasses.field(default_factory=dict)
    # (root graph id, storage, ordinal) -> the copy of a whole-slice load of
    # a tensor kept in HBM in the root's own body, which may load several
    # slices of it; ``ordinal`` counts the root's loads of the tensor.
    slice_loads: dict[tuple[int, int, int], HbmCopy] = dataclasses.field(
        default_factory=dict
    )
    # (inner loop graph id, storage) -> the first copy of the loop's stream
    # of a tensor kept in HBM, when ``_plan_early_copies`` starts it early.
    loop_primes: dict[tuple[int, int], LoopPrime] = dataclasses.field(
        default_factory=dict
    )
    # The copies ``_plan_early_copies`` starts ahead of a ring's copy of a
    # stream index: (ring buffer, index) -> (copy, folded loop iteration)
    # for the prologue's copies, and (ring buffer, site graph id, load) ->
    # runs of them for a site's refills.
    early_prologue: dict[tuple[str, int], list[tuple[HbmCopy, int]]] = (
        dataclasses.field(default_factory=dict)
    )
    early_refills: dict[tuple[str, int, int], list[_EarlyRun]] = dataclasses.field(
        default_factory=dict
    )
    # The statements that complete the DMA writes the last root left in
    # flight (``finish_root``).
    carried_writes: list[ast.stmt] = dataclasses.field(default_factory=list)
    # Loop-carried scratch -> the graph id of the root whose body holds it
    # (``share_root_scratch``).
    root_scratch: dict[str, int] = dataclasses.field(default_factory=dict)

    # Index vector -> its SMEM mirror, which the rings' copies read
    # (``_plan_index_mirrors``).
    index_mirrors: dict[int, IndexMirror] = dataclasses.field(default_factory=dict)
    # The last remote copy start of each exchange root -> the root
    # (``_exchange_starts``), and the roots whose gates' copies started there.
    remote_starts: dict[torch.fx.Node, int] = dataclasses.field(default_factory=dict)
    caught_up: set[int] = dataclasses.field(default_factory=set)
    # The roots share the buffers of the whole-slice loads of the tensors
    # kept in HBM (``_plan_hbm_copies``, ``slices_pooled``).
    pool_slices: bool = False

    def share_root_scratch(self, device_fn: DeviceFunction) -> None:
        """Let the roots share their loop-carried scratch, after codegen.

        A root's carried scratch holds state only while the root runs: the
        root writes it before its loop and reads the result back into a value
        right after.  So each root takes the buffers of earlier roots of the
        same shape and dtype, each at most once, before buffers of its own,
        and the kernel binds the name of each buffer it takes at start.
        """
        buffers: dict[tuple[object, ...], list[str]] = {}
        taken: dict[int, collections.Counter[tuple[object, ...]]] = {}
        aliases: dict[str, str] = {}
        for scratch in device_fn._scratch_args:
            root = self.root_scratch.get(scratch.name)
            if root is None:
                continue
            key = (tuple(scratch.shape), scratch.dtype)
            names = buffers.setdefault(key, [])
            count = taken.setdefault(root, collections.Counter())
            if count[key] < len(names):
                aliases[scratch.name] = names[count[key]]
            else:
                names.append(scratch.name)
            count[key] += 1
        device_fn._scratch_args = [
            scratch
            for scratch in device_fn._scratch_args
            if scratch.name not in aliases
        ]
        device_fn.preamble[:0] = [
            statement_from_string(f"{alias} = {name}")
            for alias, name in aliases.items()
        ]

    def prologue(self, device_fn: DeviceFunction) -> list[str]:
        """Statements emitted at kernel start, before root 0.

        The early row reads go first: each is one sublane tile, which the
        rings' copies would otherwise queue ahead of.  The rings' first
        ``depth`` copies are issued next so they overlap everything up to
        their first use, except those of the rings that start after another
        ring's last refill (see ``_ring_predecessor``), each after the copies
        ``_plan_early_copies`` starts ahead of it.  The SMEM mirrors of the
        read-only index vectors the rings read are filled before them.
        """
        statements = [
            statement
            for read in self.early_row_reads.values()
            for statement in read.starts
        ]
        statements.extend(
            statement
            for mirror in self.index_mirrors.values()
            if None in mirror.producers
            for statement in mirror.fill(device_fn)
        )
        for ring in self.stream:
            if _ring_predecessor(self.stream, ring) is not None:
                continue
            # A gate defers some of the first copies; those ahead of them
            # still start here.
            starts = dict(
                zip(
                    [i for i in range(ring.depth) if ring.gate_at(i) is None],
                    ring.prime(device_fn),
                    strict=True,
                )
            )
            for index in range(ring.depth):
                statements.extend(
                    copy.copy(device_fn, layer) + ".start()"
                    for copy, layer in self.early_prologue.get((ring.buffer, index), ())
                )
                if index in starts:
                    statements.append(starts[index])
        return [*statements, *self.scratch_zeroing]

    def stream_sites(self, graph_id: int) -> list[tuple[Ring, RingSite]]:
        """The ring sites of inner loop ``graph_id``, one per ring it reads."""
        return [
            (ring, site)
            for ring in self.stream
            for site in ring.sites
            if site.graph_id == graph_id and site.block_id is not None
        ]

    def root_stream_load(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: Sequence[object],
    ) -> ast.AST | None:
        """The ring read of a streamed load in a root's own body, or None
        when the load is not one: wait on its slot and read the window."""
        root_graph = state.codegen.current_root_graph_info
        assert state.fx_node is not None
        if root_graph is None or state.fx_node.graph is not root_graph.graph:
            return None
        for ring in self.stream:
            for site in ring.sites:
                if site.block_id is not None or site.graph_id != root_graph.graph_id:
                    continue
                for load_index, load in enumerate(site.loads):
                    if load.fake is tensor:
                        return _emit_root_stream_load(
                            state, ring, site, load_index, subscript
                        )
        return None

    def root_slice_load(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: Sequence[object],
    ) -> ast.AST | None:
        """The read of a whole-slice load of a tensor kept in HBM in a root's
        own body, or None when the load is not one: copy the slice into its
        buffer, unless ``_plan_early_copies`` started the copy ahead, wait,
        and read the buffer."""
        root_graph = state.codegen.current_root_graph_info
        assert state.fx_node is not None
        if root_graph is None or state.fx_node.graph is not root_graph.graph:
            return None
        storage = id(tensor.untyped_storage())
        # The codegen graph is a copy of the planned one: match the load by
        # its place among the root's loads of the tensor.
        copy = self.slice_loads.get(
            (root_graph.graph_id, storage, _load_ordinal(state.fx_node, storage))
        )
        if copy is None:
            return None
        device_fn = state.device_function
        # The copy reads what the stores before it in program order wrote to
        # its slice.
        for statement in (
            *slice_read_waits(device_fn, storage, copy.source_parts(device_fn, None)),
            *copy.started(
                device_fn,
                [statement_from_string(copy.copy(device_fn, None) + ".start()")],
            ),
            statement_from_string(copy.copy(device_fn, None) + ".wait()"),
        ):
            state.codegen.add_statement(statement)
        # The buffer holds the slice without its integer-indexed dims: keep
        # its dims and expand None ones, as ``index_parts`` does.
        parts: list[str] = []
        none_dims: list[int] = []
        for idx in subscript:
            if idx is None:
                none_dims.append(len(parts) + len(none_dims))
            elif not is_scalar_index(idx):
                parts.append(":")
        result = expr_from_string(f"{copy.buffer}[{', '.join(parts)}]")
        if copy.perm is not None:
            # ``_lane_dense_access`` left no None dims.
            kept = [not is_scalar_index(idx) for idx in subscript]
            result = logical_load(state, result, copy.perm, kept)
        for dim in none_dims:
            result = expr_from_string(
                f"jnp.expand_dims({{result}}, axis={dim})", result=result
            )
        return result

    def finish_root(
        self, device_fn: DeviceFunction, root: int, single: bool
    ) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """The statements that complete DMA writes at the end of root
        ``root``: those to emit at the end of each of its tiles, and after
        its last.

        A tile completes its writes, but a root of a ``single`` tile leaves
        those to the tensors the next root does not use in flight to the end
        of that root, so they overlap its work.  The next root must run in
        the same folded loop body, or with this one outside every folded
        loop.  ``_plan_early_copies`` sees their stores before the copies
        the next root starts.
        """
        carried, self.carried_writes = self.carried_writes, []
        device_ir = HostFunction.current().device_ir
        following = root + 1
        loops = device_ir.host_loops
        if (
            single
            and following < len(device_ir.root_ids)
            and [loop for loop in loops if root in loop.roots]
            == [loop for loop in loops if following in loop.roots]
        ):
            graphs = {info.graph_id: info for info in device_ir.graphs}
            used = _loop_storages(graphs.__getitem__, device_ir.root_ids[following])
            for storage in [*device_fn.pallas_pending_writes]:
                if storage not in used:
                    self.carried_writes.extend(
                        device_fn.pallas_pending_writes.pop(storage)
                    )
                    device_fn.pallas_ring_writes.pop(storage, None)
        return device_fn.flush_pending_writes(), carried

    def root_refills(self, device_fn: DeviceFunction, root: int) -> list[ast.stmt]:
        """Statements that end a tile of root ``root``: refill the SMEM
        mirrors of the index vectors it writes, then start the stream index
        ``D`` past each load of its own body, now that its reads are done,
        each after the copies ``_plan_early_copies`` starts ahead of it."""
        statements: list[ast.stmt] = [
            statement
            for mirror in self.index_mirrors.values()
            for statement in mirror.refresh(device_fn, root)
        ]
        for ring in self.stream:
            for site in ring.sites:
                if site.block_id is not None or site.root.root != root:
                    continue
                branches = [
                    _refill_branches(ring, site, load_index)
                    for load_index in range(len(site.loads))
                ]
                early = [
                    self.early_refills.get((ring.buffer, site.graph_id, load_index), [])
                    for load_index in range(len(site.loads))
                ]
                if (
                    not any(branches)
                    and not any(early)
                    and _handoff(self.stream, ring, site) is None
                ):
                    continue
                index = device_fn.new_var("_g")
                statements.append(
                    statement_from_string(
                        f"{index} = {_root_site_index(device_fn, site)}"
                    )
                )
                for load_index, load_branches in enumerate(branches):
                    statements.extend(
                        _early_statements(
                            device_fn,
                            early[load_index],
                            f"{index} + {load_index + ring.depth}",
                        )
                    )
                    if load_branches:
                        slot = device_fn.new_var("_slot")
                        position = (
                            index if load_index == 0 else f"({index} + {load_index})"
                        )
                        statements.append(
                            statement_from_string(f"{slot} = {position} % {ring.depth}")
                        )
                        statements.extend(
                            _refill_statements(
                                device_fn,
                                ring,
                                load_branches,
                                f"{index} + {load_index + ring.depth}",
                                slot,
                            )
                        )
                    statements.extend(
                        _handoff_statements(
                            device_fn, self.stream, ring, site, load_index, index
                        )
                    )
        for ring in self.stream:
            for gate in ring.gates:
                if gate.producer == root and not (
                    gate.readers and root in self.caught_up
                ):
                    statements.extend(_catch_up_statements(device_fn, ring, gate))
        return statements

    def remote_copy_started(self, state: CodegenState) -> None:
        """After the last remote copy start of an exchange root
        (``_exchange_starts``): start the copies its gates deferred, behind
        the remote copies, in the last iteration of the loops around it, in
        the order they are read.  A loop of another kind leaves them to the
        end of the tile (``root_refills``)."""
        root = self.remote_starts.get(state.fx_node)  # pyrefly: ignore[bad-argument-type]
        if root is None:
            return
        conditions: dict[int, str] = {}
        for loops in state.codegen.active_device_loops.values():
            for loop in loops:
                if isinstance(loop, ForiLoopState) and loop.iteration_count:
                    conditions[id(loop)] = (
                        f"{loop.loop_var_name} == {loop.iteration_count} - 1"
                    )
                elif not isinstance(loop, (DeviceGridState, PersistentReductionState)):
                    return
        device_fn = state.device_function
        held = [
            (ring, gate)
            for ring in self.stream
            for gate in ring.gates
            if gate.readers and gate.producer == root
        ]
        held.sort(
            key=lambda held: (
                held[1].lag,
                *held[0].read_order(held[0].catch_up(held[1])[0]),
            )
        )
        statements = [
            statement
            for ring, gate in held
            for statement in _catch_up_statements(device_fn, ring, gate)
        ]
        if not statements:
            return
        self.caught_up.add(root)
        for statement in _when(
            device_fn, _all([*conditions.values()]), statements, "_catch_up"
        ):
            state.codegen.add_statement(statement)

    def check_root_decode(
        self,
        device_fn: DeviceFunction,
        root: int,
        pid_var: str,
        body: Sequence[ast.AST],
        num_pids: int | None,
    ) -> None:
        """Prove that root ``root`` walks the tiles the rings model: as many,
        at the offsets ``RingRoot.tile_offsets`` decodes (the refills address
        other tiles with it).  Interprets the root's emitted pid decode."""
        ring_root = next(
            (r for ring in self.stream for r in ring.roots if r.root == root), None
        )
        if ring_root is None:
            return
        assert num_pids == ring_root.num_tiles, (num_pids, ring_root.num_tiles)
        constants = {
            name: size
            for key, name in device_fn.block_size_var_cache.items()
            if len(key) == 1
            and isinstance(size := device_fn.resolved_block_size(key[0]), int)
        }
        decode = [
            (statement.targets[0].id, statement.value)
            for statement in body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ]
        for tile in _boundary_range(
            ring_root.num_tiles, 2, ring_root.num_tiles <= _STREAM_CHECK_LIMIT
        ):
            values = {**constants, pid_var: tile}
            for name, value in decode:
                result = _eval_int(value, values)
                if result is None:
                    values.pop(name, None)
                else:
                    values[name] = result
            expected = ring_root.tile_offsets(tile)
            for (block_id, _, _), offset_var in zip(
                ring_root.axes, ring_root.offset_vars, strict=True
            ):
                assert values.get(offset_var) == expected[block_id], (
                    f"root {root} tile {tile}: {offset_var} decodes to "
                    f"{values.get(offset_var)}, the ring expects {expected[block_id]}"
                )


def enter_sequential_roots_mode(
    device_ir: DeviceIR,
    config_spec: ConfigSpec,
    host: HostFunction,
    accesses: Sequence[TileAccess],
) -> None:
    """Select the megakernel lowering for a multi-root Pallas kernel."""
    # Data-dependent inner loops are fine: they lower to a fori loop with a
    # traced trip count, and the weight ring only streams static-trip loops.
    if config_spec.has_symbolic_bounds or not all(
        isinstance(axis.extent, sympy.Expr) and axis.extent.is_Integer
        for family in device_ir.task_families
        for axis in family.axes
    ):
        raise exc.BackendUnsupported(
            backend="pallas",
            detail="multiple top-level loops with dynamic shapes "
            "(use static_shapes=True)",
        )
    device_ir.storage_roles = classify_storage_roles(device_ir, host, accesses)
    device_ir.sequential_roots = True
    config_spec.pallas_sequential_roots = True
    for family in device_ir.task_families:
        for block_id in family.logical_axis_order:
            # Every root tile is addressed with an explicit pl.ds offset, which
            # a flattened grid does not provide.
            config_spec.flatten_loops.disable_block_id(block_id)
    _require_whole_sublane_tiles(device_ir)
    # After the block size minimums above, which decide what can stream.
    layouts = config_spec.pallas_lane_dense_layouts = _lane_dense_layouts(device_ir)
    stream, hbm_default, loop_tiles = _sequential_roots_models(device_ir, layouts)
    config_spec.pallas_stream_model = stream
    config_spec.pallas_hbm_resident_default = hbm_default
    streams = [stream]
    if layouts:
        try:
            off_stream, off_hbm_default, _ = _sequential_roots_models(device_ir, {})
        except exc.BackendUnsupported:
            # Passed as given, the inputs do not fit in VMEM (e.g. a narrow
            # state padded to 128 lanes): every config passes them lane dense.
            pass
        else:
            config_spec.pallas_lane_dense_off_models = (off_stream, off_hbm_default)
            streams.append(off_stream)
    if any(model is not None for model in streams):
        # Large streamed tiles often need a whole-axis size that is not a
        # power of two (e.g. 17 * 128).
        for spec in config_spec.block_sizes:
            spec.search_whole_axis_sizes()
            whole = {
                model.whole_block_sizes[spec.block_id]
                for model in streams
                if model is not None and spec.block_id in model.whole_block_sizes
            }
            if whole:
                spec.search_only(tuple(sorted(whole)))
    config_spec.pallas_loop_tile_model = loop_tiles[
        config_spec.pallas_keeps_hbm_resident({})
    ]


def _sequential_roots_models(
    device_ir: DeviceIR, lane_dense: Mapping[int, tuple[int, ...]]
) -> tuple[StreamModel | None, bool | None, dict[bool, LoopTileModel | None]]:
    """The weight stream model, whether the default config keeps the
    tensors that can stay in HBM there, and the inner-loop tile model of a
    config that does or not, of a program that passes the inputs
    ``lane_dense`` lane dense."""
    stream = _build_stream_model(device_ir, lane_dense)
    hbm_default = _default_hbm_resident(device_ir, stream, lane_dense)
    loop_tiles = {
        keep_hbm: _build_loop_tile_model(device_ir, stream, keep_hbm, lane_dense)
        for keep_hbm in (False, True)
    }
    if stream is not None:
        # The rings share the VMEM with the inner loops' DMA buffers and the
        # loop-carried scratch.
        stream = dataclasses.replace(
            stream, loop_tiles=loop_tiles, carried=_build_carried_scratch(device_ir)
        )
    return stream, hbm_default, loop_tiles


def _memory_accesses(
    graphs: Iterable[torch.fx.Graph],
) -> Iterator[tuple[torch.fx.Node, torch.Tensor, list[object]]]:
    """Yield ``(node, accessed tensor, subscript per tensor dim)`` for each load/store."""
    for graph in graphs:
        for node in graph.nodes:
            if node.op != "call_function" or node.target not in (
                memory_ops.load,
                memory_ops.store,
            ):
                continue
            tensor = node.args[0]
            index = node.args[1]
            if not isinstance(tensor, torch.fx.Node) or not isinstance(
                index, (list, tuple)
            ):
                continue
            fake = tensor.meta.get("val")
            if isinstance(fake, torch.Tensor):
                yield node, fake, tensor_dim_subscripts(index, fake.ndim)


def tensor_dim_subscripts(index: Sequence[object], ndim: int) -> list[object]:
    """Subscript for each dim of an ``ndim`` tensor.

    ``None`` inserts an output dim without consuming a tensor dim, and missing
    trailing subscripts are full slices.
    """
    subscripts = [subscript for subscript in index if subscript is not None]
    if len(subscripts) > ndim or any(subscript is Ellipsis for subscript in subscripts):
        raise exc.BackendUnsupported(
            backend="pallas",
            detail=f"subscript {list(index)!r} in a kernel with multiple "
            "top-level loops",
        )
    return subscripts + [slice(None)] * (ndim - len(subscripts))


def _require_whole_sublane_tiles(device_ir: DeviceIR) -> None:
    """Make every tile over a second-minor dim a whole native sublane tile.

    Without BlockSpecs every tile is a ``pl.ds`` slice of a VMEM ref, and Mosaic
    rejects slices smaller than the native tile (e.g. one row of bf16), even
    when the dim itself is smaller.  Unlike the BlockSpec minimum this is not
    capped by the dim size; the launcher pads the arguments instead.
    """
    env = CompileEnvironment.current()
    graphs = (graph_info.graph for graph_info in device_ir.graphs)
    for _, fake, subscripts in _memory_accesses(graphs):
        if fake.ndim < 2:
            continue
        block_id, _ = subscript_index_scale(env, subscripts[-2])
        if block_id is not None:
            env.block_sizes[block_id].update_min_block(
                env.backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
            )


@dataclasses.dataclass
class _HostTensorUse:
    """Every device-side use of one host storage."""

    fake: torch.Tensor
    names: set[str] = dataclasses.field(default_factory=set)
    read: bool = False
    written: bool = False
    # (graph id, node) of each load.
    loads: list[tuple[int, torch.fx.Node]] = dataclasses.field(default_factory=list)
    # (graph id, node) of each store.
    stores: list[tuple[int, torch.fx.Node]] = dataclasses.field(default_factory=list)
    # A use that is neither a load nor a store.
    other: bool = False
    # Graphs that reference the tensor.
    graph_ids: set[int] = dataclasses.field(default_factory=set)


def _host_tensor_uses(device_ir: DeviceIR) -> dict[int, _HostTensorUse]:
    """The device-side uses of each host tensor, keyed by storage.

    A load reads; an atomic reads and writes; any other use (a store, a
    remote copy, ...) is conservatively a write.
    """
    uses: dict[int, _HostTensorUse] = {}
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.find_nodes(
            op="call_function", target=_host_tensor
        ):
            fake = node.meta["val"]
            use = uses.setdefault(id(fake.untyped_storage()), _HostTensorUse(fake))
            use.names.add(str(node.args[0]))
            use.graph_ids.add(graph_info.graph_id)
            for user in node.users:
                if user.target is memory_ops.load and user.args[0] is node:
                    use.read = True
                    use.loads.append((graph_info.graph_id, user))
                else:
                    use.written = True
                    use.read |= user.target in ATOMIC_OPS
                    if user.target is memory_ops.store and user.args[0] is node:
                        use.stores.append((graph_info.graph_id, user))
                    else:
                        use.other = True
    return uses


def classify_storage_roles(
    device_ir: DeviceIR, host: HostFunction, accesses: Sequence[TileAccess]
) -> dict[int, StorageRole]:
    """Classify each host tensor the device code uses, by storage.

    Keyed by storage rather than fx node or name so the roles survive
    per-config graph copies and tell apart the tensors of one list argument.
    One storage reached through several host names (a view or alias) is
    rejected.  The same tensor passed as two arguments is not detected: the
    arguments are distinct at trace time.
    """
    if any(access.allocation_id < 0 for access in accesses):
        raise exc.BackendUnsupported(
            backend="pallas",
            detail="memory accesses with an unknown allocation in a kernel "
            "with multiple top-level loops",
        )
    body_local = top_level_empty_names(host) - returned_name_dependencies(host)
    roles: dict[int, StorageRole] = {}
    for storage, use in _host_tensor_uses(device_ir).items():
        if len(use.names) > 1:
            raise exc.BackendUnsupported(
                backend="pallas",
                detail=f"aliased tensors {sorted(use.names)} in a kernel with "
                "multiple top-level loops",
            )
        (name,) = use.names
        if not use.written:
            roles[storage] = StorageRole.READ_ONLY
        elif use.read and name in body_local:
            roles[storage] = StorageRole.INTERMEDIATE
        else:
            roles[storage] = StorageRole.OUTPUT
    return roles


def _lane_dense_layouts(device_ir: DeviceIR) -> dict[int, tuple[int, ...]]:
    """The inputs to pass lane dense (``pallas_lane_dense``, see
    ``lane_dense``), by storage: the permutation of each one's dims into its
    physical order.

    An input qualifies when ``lane_dense_perm`` gives it a dim order (XLA's
    default layout of it permutes its dims and its minor dim is narrow, so
    the permuted array the launcher passes is a bitcast of it) and every
    access is a load or store in a root's own body that
    ``_lane_dense_access`` can index in that order: then its refs are whole
    VMEM blocks, or windows of a ring that streams it at root level.  Only
    the default layout is taken, so the launcher never introduces a copy.
    """
    env = CompileEnvironment.current()
    input_storages = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    root_graph_ids = set(device_ir.root_ids)
    layouts: dict[int, tuple[int, ...]] = {}
    for storage, use in _host_tensor_uses(device_ir).items():
        fake = use.fake
        if (
            storage not in input_storages
            or use.other
            or fake.ndim < 2
            or not all(isinstance(size, int) for size in fake.shape)
        ):
            continue
        perm = lane_dense_perm([int(size) for size in fake.shape], fake.dtype)
        if perm is None or not all(
            graph_id in root_graph_ids and _lane_dense_access(fake, node, perm)
            for graph_id, node in [*use.loads, *use.stores]
        ):
            continue
        layouts[storage] = perm
    return layouts


def _lane_dense_access(
    fake: torch.Tensor, node: torch.fx.Node, perm: tuple[int, ...]
) -> bool:
    """Whether load or store ``node`` can index ``fake`` in dim order
    ``perm``: a subscript per dim, none a tensor, and no extra mask; on each
    of the two minor dims of that order a whole or static slice, or, for a
    load, an integer (on the second-minor dim only of a dtype that
    ``row_packing`` reads rows of); a value over the sliced dims that
    ``lowered_transpose`` takes between the orders; and a stored value that
    is a scalar or has a dim per sliced dim, which a transpose permutes like
    the subscripts."""
    index = node.args[1]
    if (
        not isinstance(index, (list, tuple))
        or len(index) != fake.ndim
        or any(subscript is None for subscript in index)
    ):
        return False
    mask_arg = 3 if node.target is memory_ops.store else 2
    if len(node.args) > mask_arg and node.args[mask_arg] is not None:
        return False
    values = [
        subscript.meta.get("val") if isinstance(subscript, torch.fx.Node) else subscript
        for subscript in index
    ]
    if any(isinstance(value, torch.Tensor) for value in values):
        return False
    kept = [not is_scalar_index(value) for value in values]
    *_, rows, cols = perm
    for dim in (rows, cols):
        subscript = index[dim]
        if not kept[dim]:
            if node.target is memory_ops.store or (
                dim == rows and row_packing(fake.dtype, int(fake.shape[dim])) is None
            ):
                return False
        elif not (
            isinstance(subscript, slice)
            and all(
                bound is None or type(bound) is int
                for bound in (subscript.start, subscript.stop)
            )
            and subscript.step in (None, 1)
        ):
            return False
    if not lowered_transpose(perm, kept):
        return False
    if node.target is memory_ops.store:
        stored = node.args[2]
        if isinstance(stored, torch.fx.Node):
            stored = stored.meta.get("val")
        if isinstance(stored, torch.Tensor) and stored.ndim not in (0, sum(kept)):
            return False
    return True


def _hbm_resident_tensors(
    device_ir: DeviceIR, lane_dense: Mapping[int, tuple[int, ...]]
) -> dict[int, torch.Tensor]:
    """Written tensors kept in HBM instead of whole in VMEM, by storage.

    A tensor qualifies when every access is a tile load in an inner loop,
    which streams it through a 2-slot DMA buffer of its own, a load of a
    whole slice in a root's own body, which a DMA copies into a VMEM buffer
    of its own (``MegakernelPlan.root_slice_load``), or a store of whole
    rows or slices at scalar indices in a root's own body, which the store
    lowering writes back with a DMA: a KV cache that a decode step reads
    once and writes one row of, or a recurrent state it reads and rewrites
    whole.  Scalar indices on leading dims may be expressions in the folded
    host loops around the access, so each layer's state of a folded stack
    qualifies.  A tensor whose minor dim is ragged (not whole 128 lanes)
    qualifies when every access is a whole slice at scalar leading indices,
    which a DMA copies through a view of the tensor as rows of that dim.
    An input passed lane dense (``lane_dense``) qualifies when it is only
    stored, each time a whole slice at scalar leading indices, and its dim
    order swaps the last two (``_lane_dense_slice_stores``): a snapshot of
    a narrow state that a decode step writes and never reads.
    """
    env = CompileEnvironment.current()
    root_graph_ids = set(device_ir.root_ids)
    loop_block_ids = {
        graph_id: graph_info.block_ids
        for graph_id, (graph_info, _) in _root_loops(device_ir).items()
    }
    host_loops = _graph_host_loops(device_ir)
    resident: dict[int, torch.Tensor] = {}
    for storage, use in _host_tensor_uses(device_ir).items():
        loop_load_ids = [
            graph_id for graph_id, _ in use.loads if graph_id in loop_block_ids
        ]
        ragged = _ragged_rows(use.fake)
        if (
            device_ir.storage_roles.get(storage) is not StorageRole.OUTPUT
            or use.other
            or not use.stores
        ):
            continue
        if storage in lane_dense:
            if _lane_dense_slices(use, lane_dense[storage], root_graph_ids, host_loops):
                resident[storage] = use.fake
            continue
        if (
            not (_whole_vmem_tiles(use.fake) or ragged)
            # The loop DMAs one tile of a tensor per iteration; a root body
            # may load several slices, each copied into a buffer of its own.
            or len(loop_load_ids) != len(set(loop_load_ids))
        ):
            continue
        if all(
            graph_id in root_graph_ids
            and _is_row_store(use.fake, node)
            and (not ragged or _slice_scalar_dims(use.fake, node) is not None)
            for graph_id, node in use.stores
        ) and all(
            (
                graph_id in loop_block_ids
                and not ragged
                and _is_tile_load(
                    env,
                    use.fake,
                    node,
                    loop_block_ids[graph_id],
                    host_loops[graph_id],
                )
            )
            or (
                graph_id in root_graph_ids
                and _slice_load_indices(use.fake, node, host_loops[graph_id])
                is not None
            )
            for graph_id, node in use.loads
        ):
            resident[storage] = use.fake
    return resident


def _lane_dense_slices(
    use: _HostTensorUse,
    perm: tuple[int, ...],
    root_graph_ids: set[int],
    host_loops: Mapping[int, Sequence[HostLoopInfo]],
) -> bool:
    """Whether every access to ``use``'s tensor, passed lane dense in dim
    order ``perm``, is a store or load of a whole slice (the last two dims
    whole, every other whole or an integer) in a root's own body, and
    ``perm`` swaps the last two dims into a minor dim of whole 128 lanes:
    then each slice is a whole slice in physical order too, which a DMA
    writes from a stage of its transpose or copies into a buffer the load
    transposes back (``MegakernelPlan.root_slice_load``): a narrow state, or
    a snapshot of one that a decode step writes and never reads."""
    fake = use.fake
    return (
        perm == minor_swap(fake.ndim)
        and all(isinstance(size, int) for size in fake.shape)
        and fake.shape[-2] % 128 == 0
        and all(
            graph_id in root_graph_ids
            and _is_row_store(fake, node)
            and all(
                isinstance(subscript, slice) and subscript == slice(None)
                for subscript in tensor_dim_subscripts(
                    cast("list[object]", node.args[1]), fake.ndim
                )[-2:]
            )
            for graph_id, node in use.stores
        )
        and all(
            graph_id in root_graph_ids
            and _slice_load_indices(fake, node, host_loops[graph_id], perm) is not None
            for graph_id, node in use.loads
        )
    )


def _graph_host_loops(device_ir: DeviceIR) -> dict[int, list[HostLoopInfo]]:
    """The folded host loops around each root, by the graph id of the root
    and of every inner loop nested in it through loops alone."""
    graphs = {graph_info.graph_id: graph_info for graph_info in device_ir.graphs}
    host_loops: dict[int, list[HostLoopInfo]] = {}
    for root, root_graph_id in enumerate(device_ir.root_ids):
        loops = [loop for loop in device_ir.host_loops if root in loop.roots]
        pending = [root_graph_id]
        while pending:
            graph_id = pending.pop()
            host_loops[graph_id] = loops
            pending.extend(
                cast("int", node.args[0])
                for node in graphs[graph_id].graph.find_nodes(
                    op="call_function", target=_for_loop
                )
            )
    return host_loops


def _ragged_rows(fake: torch.Tensor) -> bool:
    """Whether ``fake``'s minor dim is ragged (not whole 128 lanes) and its
    rows are whole sublane tiles."""
    if fake.ndim < 2 or not all(isinstance(size, int) for size in fake.shape):
        return False
    sublane = CompileEnvironment.current().backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
    return fake.shape[-1] % 128 != 0 and fake.shape[-2] % sublane == 0


def _slice_scalar_dims(fake: torch.Tensor, node: torch.fx.Node) -> set[int] | None:
    """The scalar dims of whole-slice access ``node`` to ``fake``: every
    dim before the last two scalar, those two whole; None otherwise."""
    subscripts = tensor_dim_subscripts(cast("list[object]", node.args[1]), fake.ndim)
    scalar_dims: set[int] = set()
    for dim, subscript in enumerate(subscripts):
        whole = isinstance(subscript, slice) and subscript == slice(None)
        if dim >= fake.ndim - 2:
            if not whole:
                return None
        elif whole:
            return None
        else:
            scalar_dims.add(dim)
    return scalar_dims


def _slice_load_indices(
    fake: torch.Tensor,
    node: torch.fx.Node,
    loops: Sequence[HostLoopInfo],
    perm: tuple[int, ...] | None = None,
) -> dict[int, sympy.Expr] | None:
    """The integer index of each scalar dim of load ``node`` when it reads a
    whole slice of ``fake``: the last two dims whole, every other whole or
    an in-bounds integer in the symbols of the folded host loops ``loops``
    (all of them integers for a ragged minor dim, unless ``fake`` is passed
    lane dense in dim order ``perm``); None otherwise."""
    index = node.args[1]
    if node.args[2] is not None or not isinstance(index, (list, tuple)):
        return None
    scalars: dict[int, sympy.Expr] = {}
    for dim, subscript in enumerate(tensor_dim_subscripts(index, fake.ndim)):
        if isinstance(subscript, slice) and subscript == slice(None):
            continue
        scalar = _scalar_subscript(subscript, loops)
        if (
            scalar is None
            or dim >= fake.ndim - 2
            or not _scalar_in_bounds(scalar, loops, fake.shape[dim])
        ):
            return None
        scalars[dim] = scalar
    if perm is None and _ragged_rows(fake) and len(scalars) < fake.ndim - 2:
        return None
    return scalars


def _root_loops(
    device_ir: DeviceIR,
) -> dict[int, tuple[ForLoopGraphInfo, torch.fx.Node]]:
    """The graph and ``_for_loop`` node of each inner loop nested in a root
    through loops alone, by graph id: the loops whose first DMA is issued in
    the root's own body."""
    graphs = {graph_info.graph_id: graph_info for graph_info in device_ir.graphs}
    loops: dict[int, tuple[ForLoopGraphInfo, torch.fx.Node]] = {}
    pending = [*device_ir.root_ids]
    while pending:
        graph = graphs[pending.pop()].graph
        for node in graph.find_nodes(op="call_function", target=_for_loop):
            graph_id = cast("int", node.args[0])
            loops[graph_id] = (cast("ForLoopGraphInfo", graphs[graph_id]), node)
            pending.append(graph_id)
    return loops


def _whole_vmem_tiles(fake: torch.Tensor) -> bool:
    """Whether ``fake`` is a whole number of (sublane, 128) VMEM tiles."""
    if fake.ndim < 2 or not all(isinstance(size, int) for size in fake.shape):
        return False
    sublane = CompileEnvironment.current().backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
    return fake.shape[-1] % 128 == 0 and fake.shape[-2] % sublane == 0


def _is_tile_load(
    env: CompileEnvironment,
    fake: torch.Tensor,
    node: torch.fx.Node,
    block_ids: list[int],
    loops: Sequence[HostLoopInfo],
) -> bool:
    """Whether ``node`` loads a tile of ``fake`` that a loop over
    ``block_ids`` can DMA: every dim whole, a plain tile, or (a dim before
    the last two) an in-bounds integer in the symbols of the folded host
    loops ``loops``; the last one whole, and at least one tiled by the loop.
    A tile over the second-minor dim is a whole sublane tile
    (``_require_whole_sublane_tiles``)."""
    index = node.args[1]
    if (
        node.args[2] is not None
        or not isinstance(index, (list, tuple))
        or len(index) != fake.ndim
    ):
        return False
    tiled = False
    for dim, subscript in enumerate(index):
        if isinstance(subscript, slice) and subscript == slice(None):
            continue
        scalar = _scalar_subscript(subscript, loops)
        if scalar is not None:
            if dim >= fake.ndim - 2 or not _scalar_in_bounds(
                scalar, loops, fake.shape[dim]
            ):
                return False
            continue
        info = subscript_tile_info(env, subscript)
        if (
            dim == fake.ndim - 1
            or info is None
            or info.offset != 0
            or info.block_size is not None
        ):
            return False
        tiled |= info.block_id in block_ids
    return tiled


def _is_row_store(fake: torch.Tensor, node: torch.fx.Node) -> bool:
    """Whether ``node`` stores whole rows of ``fake``: scalar indices on any
    dims before the last, every other dim whole, and a value of exactly the
    rows' shape."""
    index, value, extra_mask = node.args[1:4]
    if extra_mask is not None or not isinstance(index, (list, tuple)):
        return False
    if len(index) != fake.ndim:
        return False
    rows: list[int] = []
    for dim, subscript in enumerate(index):
        if isinstance(subscript, slice) and subscript == slice(None):
            rows.append(fake.shape[dim])
            continue
        scalar = (
            subscript.meta.get("val")
            if isinstance(subscript, torch.fx.Node)
            else subscript
        )
        if dim == fake.ndim - 1 or not is_scalar_index(scalar):
            return False
    rows_value = value.meta.get("val") if isinstance(value, torch.fx.Node) else None
    return isinstance(rows_value, torch.Tensor) and list(rows_value.shape) == rows


# Below this size a read-only table read a row at a time stays whole in VMEM:
# copying it in costs less than a DMA round trip per row read.
_ROW_GATHER_MIN_BYTES = 1024 * 1024


def _row_gather_tables(device_ir: DeviceIR) -> dict[int, torch.Tensor]:
    """Read-only tensors always kept in HBM, by storage: those of more than
    ``_ROW_GATHER_MIN_BYTES`` whose every access is a row load at a runtime
    index in a root's own body (``_is_row_gather``), which the load lowering
    reads with a DMA of the sublane tile holding the row (``row_read``): an
    embedding table a decode step reads the token's row of.

    Unlike the written tensors of ``_hbm_resident_tensors`` there is no
    trade-off for ``pallas_hbm_resident`` to tune: kept whole in VMEM, the
    step would copy in the whole table to read one row.
    """
    root_graph_ids = set(device_ir.root_ids)
    return {
        storage: use.fake
        for storage, use in _host_tensor_uses(device_ir).items()
        if device_ir.storage_roles.get(storage) is StorageRole.READ_ONLY
        and not use.other
        and _whole_vmem_tiles(use.fake)
        and _padded_bytes(use.fake) > _ROW_GATHER_MIN_BYTES
        and all(
            graph_id in root_graph_ids and _is_row_gather(use.fake, node)
            for graph_id, node in use.loads
        )
    }


def _is_row_gather(fake: torch.Tensor, node: torch.fx.Node) -> bool:
    """Whether ``node`` loads whole rows of ``fake`` at a runtime index:
    scalar indices on some dims before the last, at least one a value the
    kernel computes (not a loop index, at which a loop would read every
    row), and every other dim whole."""
    index = node.args[1]
    if (
        node.args[2] is not None
        or not isinstance(index, (list, tuple))
        or len(index) != fake.ndim
    ):
        return False
    runtime = False
    for dim, subscript in enumerate(index):
        if isinstance(subscript, slice) and subscript == slice(None):
            continue
        scalar = (
            subscript.meta.get("val")
            if isinstance(subscript, torch.fx.Node)
            else subscript
        )
        if dim == fake.ndim - 1 or not is_scalar_index(scalar):
            return False
        runtime |= isinstance(scalar, torch.Tensor)
    return runtime


@dataclasses.dataclass
class RowRead:
    """The DMA of a row load from a read-only tensor kept in HBM.

    Mosaic DMAs the tiled second-minor dim only at tile-aligned offsets, so
    a row at runtime position ``row`` is read with the sublane tile that
    holds it, rows ``base .. base + tile - 1``, into VMEM ``stage``; the
    load selects row ``offset`` (``row - base``) on ``axis`` of the stage.
    Scalar indices on the dims before the rows index the copy directly, and
    leave ``axis`` None when the rows themselves are whole.
    """

    starts: list[str]
    wait: str
    stage: str
    axis: int | None
    offset: str | None


def row_read(
    device_fn: DeviceFunction, fake: torch.Tensor, parts: Sequence[str]
) -> RowRead:
    """Plan the DMA of the row load from ``fake`` with index ``parts`` (the
    source of each scalar index, ``":"`` for the whole dims)."""
    name = device_fn.tensor_arg(fake).name
    tile = CompileEnvironment.current().backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
    row_dim = fake.ndim - 2
    source = [*parts]
    stage_shape = [
        int(size) for size, part in zip(fake.shape, parts, strict=True) if part == ":"
    ]
    starts: list[str] = []
    axis = offset = None
    if parts[row_dim] != ":":
        row = device_fn.new_var("row")
        base = device_fn.new_var("row_base")
        starts += [
            f"{row} = {parts[row_dim]}",
            f"{base} = pl.multiple_of(({row} // {tile}) * {tile}, {tile})",
        ]
        source[row_dim] = f"pl.ds({base}, {tile})"
        axis = sum(1 for part in parts[:row_dim] if part == ":")
        stage_shape.insert(axis, tile)
        offset = f"{row} - {base}"
    stage = device_fn.register_scratch(
        tuple(stage_shape), fake.dtype, name_hint=f"{name}_rows"
    )
    semaphore = device_fn.register_dma_semaphore(name_hint=f"{name}_rows_sem")
    copy = device_fn.new_var(f"{name}_row_copy")
    source_ref = f"{name}.at[{', '.join(source)}]"
    starts += [
        f"{copy} = pltpu.make_async_copy({source_ref}, {stage}, {semaphore})",
        f"{copy}.start()",
    ]
    return RowRead(starts, f"{copy}.wait()", stage, axis, offset)


def _plan_early_row_reads(
    device_ir: DeviceIR, device_fn: DeviceFunction, hbm_resident: frozenset[int]
) -> dict[int, RowRead]:
    """Start the row reads of read-only tensors kept in HBM at kernel start,
    by storage, so the copy overlaps everything up to the load.

    That is legal for a tensor's only load when it runs once, in a root of
    one tile outside the folded host loops, and its indices are constants
    or constant elements of read-only SMEM inputs (a token id), which the
    prologue reads as well.  Other row loads start their copy at the load.
    """
    reads: dict[int, RowRead] = {}
    roles = device_ir.storage_roles
    for storage, use in _host_tensor_uses(device_ir).items():
        if (
            storage not in hbm_resident
            or roles.get(storage) is not StorageRole.READ_ONLY
            or len(use.loads) != 1
        ):
            continue
        ((graph_id, node),) = use.loads
        root = device_ir.root_ids.index(graph_id)
        tiles = math.prod(
            -(
                -_static_extent(axis)
                // cast("int", device_fn.resolved_block_size(axis.block_id))
            )
            for axis in device_ir.task_families[root].axes
        )
        if tiles != 1 or any(root in loop.roots for loop in device_ir.host_loops):
            continue
        parts = [
            _prologue_scalar(device_ir, device_fn, subscript)
            for subscript in cast("list[object]", node.args[1])
        ]
        if all(part is not None for part in parts):
            reads[storage] = row_read(device_fn, use.fake, cast("list[str]", parts))
    return reads


def _prologue_scalar(
    device_ir: DeviceIR, device_fn: DeviceFunction, subscript: object
) -> str | None:
    """Source for ``subscript`` of a row load that the kernel prologue can
    evaluate: ``":"`` for a whole dim, a constant, or a read of a constant
    element of a read-only SMEM input; None otherwise."""
    if isinstance(subscript, slice):
        return ":"
    if isinstance(subscript, int):
        return str(subscript)
    if (
        not isinstance(subscript, torch.fx.Node)
        or subscript.target is not memory_ops.load
    ):
        return None
    host, index, extra_mask = subscript.args[:3]
    assert isinstance(host, torch.fx.Node)
    fake = host.meta["val"]
    if (
        extra_mask is not None
        or not isinstance(index, (list, tuple))
        or not all(isinstance(i, int) for i in index)
        or device_ir.storage_roles.get(id(fake.untyped_storage()))
        is not StorageRole.READ_ONLY
        or device_fn.pallas_memory_space.get(id(fake)) is not PallasMemorySpace.SMEM
    ):
        return None
    return f"{device_fn.tensor_arg(fake).name}[{', '.join(map(str, index))}]"


def _hbm_resident_whole_bytes(
    device_ir: DeviceIR, lane_dense: Mapping[int, tuple[int, ...]]
) -> dict[int, int]:
    """The VMEM each tensor ``_hbm_resident_tensors`` finds would take kept
    whole in VMEM instead, by storage: twice its bytes, as the launcher
    counts VMEM arguments."""
    return {
        storage: 2 * _padded_bytes(fake, lane_dense.get(storage))
        for storage, fake in _hbm_resident_tensors(device_ir, lane_dense).items()
    }


def _hbm_slot_bytes(
    device_ir: DeviceIR,
    lane_dense: Mapping[int, tuple[int, ...]],
    pooled: bool = False,
) -> int:
    """The VMEM the tensors ``_hbm_resident_tensors`` finds take kept in
    HBM, besides the inner loops' tile buffers (``LoopTileModel``): the
    buffers of the whole-slice loads in the roots' own bodies, one per load
    or, ``pooled``, as many of each shape and dtype as one root loads
    (``_plan_hbm_copies``), and the
    stages the stores of each shape to a tensor share (``hbm_store_rings``),
    each of which holds the sublane tile of a row, or the window of two a
    run of row stores writes (``_emit_hbm_resident_store``)."""
    root_graph_ids = set(device_ir.root_ids)
    uses = _host_tensor_uses(device_ir)
    runs = _root_row_store_runs(device_ir, lane_dense)
    resident = _hbm_resident_tensors(device_ir, lane_dense)
    rings: dict[torch.fx.Node, int] = {}
    for graph_id in root_graph_ids:
        rings.update(
            hbm_store_rings(device_ir.graphs[graph_id].graph, resident, lane_dense)
        )
    total = 0
    loads: collections.Counter[tuple[int, tuple[int, ...], torch.dtype]] = (
        collections.Counter()
    )
    for storage, fake in resident.items():
        use = uses[storage]
        store_stages: dict[tuple[int, ...], int] = {}
        for graph_id, node in [*use.loads, *use.stores]:
            if graph_id not in root_graph_ids:
                continue
            shape = _hbm_stage_shape(fake, node, node in runs, lane_dense.get(storage))
            if node.target is memory_ops.store:
                slots = store_stages.get(shape, 1)
                store_stages[shape] = max(slots, rings.get(node, 1))
            else:
                loads[graph_id, shape, fake.dtype] += 1
        total += sum(
            _padded_shape_bytes(shape, fake.dtype) * slots
            for shape, slots in store_stages.items()
        )
    buffers: dict[tuple[tuple[int, ...], torch.dtype], int] = {}
    for (_, shape, dtype), count in loads.items():
        other = buffers.get((shape, dtype), 0)
        buffers[shape, dtype] = max(other, count) if pooled else other + count
    return total + sum(
        _padded_shape_bytes(shape, dtype) * count
        for (shape, dtype), count in buffers.items()
    )


def _root_row_store_runs(
    device_ir: DeviceIR, lane_dense: Mapping[int, tuple[int, ...]]
) -> dict[torch.fx.Node, RowStoreRun]:
    """The runs of row stores (``row_store_runs``) in the roots' own bodies."""
    runs: dict[torch.fx.Node, RowStoreRun] = {}
    for graph_id in set(device_ir.root_ids):
        runs.update(row_store_runs(device_ir.graphs[graph_id].graph, lane_dense))
    return runs


def _hbm_stage_shape(
    fake: torch.Tensor,
    node: torch.fx.Node,
    in_run: bool,
    perm: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """The shape of the VMEM buffer a load or store ``node`` of tensor
    ``fake`` kept in HBM moves its slice through: the dims it slices whole,
    and the sublane tile, or the window of two tiles of a run of row stores
    (``in_run``), that holds an indexed row; in physical order when the
    tensor is laid out in dim order ``perm``."""
    tile = CompileEnvironment.current().backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
    sizes: list[int | None] = []
    for dim, subscript in enumerate(
        tensor_dim_subscripts(cast("list[object]", node.args[1]), fake.ndim)
    ):
        if isinstance(subscript, slice) and subscript == slice(None):
            sizes.append(int(fake.shape[dim]))
        else:
            row = tile * (2 if in_run else 1)
            sizes.append(row if dim == fake.ndim - 2 else None)
    if perm is not None:
        # Staged in physical order (``_lane_dense_slice_stores``).
        sizes = physical_order(sizes, perm)
    return tuple(size for size in sizes if size is not None)


def hbm_store_rings(
    graph: torch.fx.Graph,
    resident: Collection[int],
    lane_dense: Mapping[int, tuple[int, ...]],
) -> dict[torch.fx.Node, int]:
    """The slots of the ring of stages each store of whole slices in
    ``graph`` to a tensor kept in HBM (storage in ``resident``) takes turns
    in (``_emit_hbm_resident_store``).

    A store that writes whole slices at indices of its leading dims reads
    nothing back, so its write may still be pending while the next one to
    another slice stages its rows: the stores of one shape to a tensor in a
    root's body (e.g. one state snapshot per row of a block) take turns in
    a ring of stages, as many as they are, within ``_STORE_RING_BYTES``,
    and each waits only for the write of the store before it in its slot.
    """
    sites: dict[tuple[int, tuple[int, ...]], list[torch.fx.Node]] = (
        collections.defaultdict(list)
    )
    for node in graph.nodes:
        if node.target is not memory_ops.store:
            continue
        fake = cast("torch.fx.Node", node.args[0]).meta["val"]
        storage = id(fake.untyped_storage())
        if storage not in resident or fake.ndim < 2:
            continue
        perm = lane_dense.get(storage)
        whole = [
            isinstance(subscript, slice) and subscript == slice(None)
            for subscript in tensor_dim_subscripts(
                cast("list[object]", node.args[1]), fake.ndim
            )
        ]
        if perm is not None:
            whole = physical_order(whole, perm)
        if all(whole[-2:]):
            sites[(storage, _hbm_stage_shape(fake, node, False, perm))].append(node)
    rings: dict[torch.fx.Node, int] = {}
    for (_, shape), nodes in sites.items():
        fake = cast("torch.fx.Node", nodes[0].args[0]).meta["val"]
        stage_bytes = _padded_shape_bytes(shape, fake.dtype)
        slots = max(1, min(len(nodes), _STORE_RING_BYTES // stage_bytes))
        rings.update(dict.fromkeys(nodes, slots))
    return rings


@dataclasses.dataclass
class RingWrite:
    """The pending write of a store through slot ``slot`` of the ring of
    stages ``stage_key`` (``hbm_store_rings``) to the slice at ``parts``,
    which ``waits`` complete."""

    stage_key: tuple[int, tuple[int, ...]]
    slot: int
    parts: tuple[str, ...]
    waits: list[ast.stmt]


def ring_waits(
    device_fn: DeviceFunction, storage: int, ring: RingWrite
) -> list[ast.stmt]:
    """The waits a ring store ``ring`` to ``storage`` emits before it
    stages its rows: for the write before it in its slot, when the writes
    pending to the storage are all ring writes to other slices; else for
    every pending write (``flush_pending_writes``)."""
    writes = device_fn.pallas_ring_writes.get(storage, [])
    if not all(_disjoint_slices(write.parts, ring.parts) for write in writes):
        return device_fn.flush_pending_writes([storage])
    return _take_ring_waits(
        device_fn,
        storage,
        lambda write: (write.stage_key, write.slot) == (ring.stage_key, ring.slot),
    )


def slice_read_waits(
    device_fn: DeviceFunction, storage: int, parts: Sequence[str]
) -> list[ast.stmt]:
    """The waits a copy of the slice at ``parts`` (per dim, physical order)
    of ``storage`` emits before it starts: for the ring writes to slices it
    may overlap, when the writes pending to the storage are all ring writes;
    else for every pending write.  The writes to the other slices stay in
    flight, e.g. the state of one sequence while the next one's is read."""
    return _take_ring_waits(
        device_fn, storage, lambda write: not _disjoint_slices(write.parts, parts)
    )


def _take_ring_waits(
    device_fn: DeviceFunction,
    storage: int,
    selected: Callable[[RingWrite], bool],
) -> list[ast.stmt]:
    """Pop the waits of the ring writes to ``storage`` that ``selected``
    picks, when the writes pending to it are all ring writes; else every
    pending write's (``flush_pending_writes``)."""
    writes = device_fn.pallas_ring_writes.get(storage, [])
    pending = device_fn.pallas_pending_writes.get(storage, [])
    ring_waits = {id(wait) for write in writes for wait in write.waits}
    if not all(id(wait) in ring_waits for wait in pending):
        return device_fn.flush_pending_writes([storage])
    statements: list[ast.stmt] = []
    for write in [*writes]:
        if selected(write):
            statements.extend(write.waits)
            writes.remove(write)
    done = {id(wait) for wait in statements}
    device_fn.pallas_pending_writes[storage] = [
        wait for wait in pending if id(wait) not in done
    ]
    return statements


def _disjoint_slices(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether the slices of a tensor at subscripts ``a`` and ``b`` (per
    dim) are disjoint: they take different integers at some dim."""
    return any(
        x != y and x.lstrip("-").isdigit() and y.lstrip("-").isdigit()
        for x, y in zip(a, b, strict=True)
    )


def store_ring_slots(state: CodegenState) -> int:
    """The slots of the ring of stages store ``state.fx_node`` takes turns
    in (``hbm_store_rings``); 1 for a store that is in none."""
    device_fn = state.device_function
    node = state.fx_node
    assert node is not None and device_fn.pallas_megakernel is not None
    if node.graph not in device_fn.pallas_store_rings:
        device_fn.pallas_store_rings[node.graph] = hbm_store_rings(
            node.graph,
            device_fn.pallas_megakernel.hbm_resident,
            device_fn.pallas_lane_dense,
        )
    return device_fn.pallas_store_rings[node.graph].get(node, 1)


def _default_hbm_resident(
    device_ir: DeviceIR,
    stream: StreamModel | None,
    lane_dense: Mapping[int, tuple[int, ...]],
) -> bool | None:
    """Whether the default config keeps the tensors ``_hbm_resident_tensors``
    finds in HBM; None when there are none.

    Kept whole in VMEM, a tensor is copied in and out once, at kernel start
    and end.  Kept in HBM, each slice, and the first tile of each inner
    loop's stream of it, is copied in at its place in the rings' copy order
    (``_plan_early_copies``), which costs about what the whole copies do, but
    the buffers and stages (``_hbm_slot_bytes``) take VMEM of their own.  So
    with rings, the tensors stay in HBM when that leaves the rings more VMEM
    (many layers' state, long contexts: larger tiles, deeper rings); without,
    they stay whole in VMEM when they fit next to the other tensors kept
    whole and a reserve.
    """
    hbm = sum(_hbm_resident_whole_bytes(device_ir, lane_dense).values())
    if not hbm:
        return None
    if stream is not None:
        return _hbm_slot_bytes(device_ir, lane_dense) < hbm
    resident = sum(
        _whole_vmem_bytes(
            device_ir, _host_tensor_uses(device_ir), set(), lane_dense
        ).values()
    )
    return resident + hbm > _vmem_capacity() - _WHOLE_RESERVE_BYTES


def plan_hbm_resident(device_ir: DeviceIR, device_fn: DeviceFunction) -> frozenset[int]:
    """Keep the tensors ``_hbm_resident_tensors`` finds in HBM, where the
    launcher aliases them in place, if the config does, and the tables
    ``_row_gather_tables`` finds always; returns their storages."""
    config_spec = CompileEnvironment.current().config_spec
    resident = _row_gather_tables(device_ir)
    if config_spec.pallas_keeps_hbm_resident(device_fn.config):
        resident.update(_hbm_resident_tensors(device_ir, device_fn.pallas_lane_dense))
    for fake in resident.values():
        device_fn.pallas_memory_space[id(fake)] = PallasMemorySpace.HBM
    return frozenset(resident)


def is_hbm_resident(device_fn: DeviceFunction, fake: torch.Tensor) -> bool:
    """Whether ``fake`` is a megakernel tensor kept in HBM."""
    plan = device_fn.pallas_megakernel
    return plan is not None and id(fake.untyped_storage()) in plan.hbm_resident


def is_indexed_weight_load(
    device_fn: DeviceFunction, fake: torch.Tensor, subscript: Sequence[object]
) -> bool:
    """Whether an inner-loop tile load of a megakernel reads a read-only
    tensor at a runtime scalar index (a value the kernel reads, such as an
    expert id) that the weight ring cannot follow.

    The index is defined before the loop, so every tile is read once and
    the next one can be prefetched.
    """
    return (
        device_fn.pallas_megakernel is not None
        and any(
            isinstance(index, torch.Tensor) and index.ndim == 0 for index in subscript
        )
        and HostFunction.current().device_ir.storage_roles.get(
            id(fake.untyped_storage())
        )
        is StorageRole.READ_ONLY
    )


def flush_loop_reads(state: CodegenState, graph_id: int) -> None:
    """Wait for the pending DMA writes to the tensors that inner loop
    ``graph_id`` or a loop nested in it uses, ahead of the loop's first DMA.

    The loads of a tensor kept in HBM are all in loops nested in a root
    through loops alone, so the outermost one is in the root's own body,
    like the stores whose writes it waits for.  The writes of the stores
    deferred to the loop stay pending (``row_forward_loop``).
    """
    device_fn = state.device_function
    if not device_fn.pallas_pending_writes:
        return
    forwarded = {
        storage
        for storage, forward in device_fn.pallas_row_forwards.items()
        if forward.graph_id == graph_id
    }
    storages = _loop_storages(state.get_graph, graph_id) - forwarded
    for statement in device_fn.flush_pending_writes(storages):
        state.add_statement(statement)


def _loop_storages(get_graph: Callable[[int], GraphInfo], graph_id: int) -> set[int]:
    """The storages that inner loop ``graph_id`` or a loop nested in it uses."""
    storages: set[int] = set()
    pending = [graph_id]
    while pending:
        graph = get_graph(pending.pop()).graph
        storages.update(
            id(node.meta["val"].untyped_storage())
            for node in graph.find_nodes(op="call_function", target=_host_tensor)
        )
        pending.extend(
            cast("int", node.args[0])
            for node in graph.find_nodes(op="call_function", target=_for_loop)
        )
    return storages


@dataclasses.dataclass(frozen=True)
class RowStoreRun:
    """A store in a run of row stores to one tensor kept in HBM at rows
    ``base + offset`` (``row_store_runs``).

    ``low`` is the smallest offset in the run; ``first`` and ``last`` say
    where the store is in it.
    """

    offset: int
    low: int
    first: bool
    last: bool


def row_run_tensor(
    fake: torch.Tensor, lane_dense: Mapping[int, tuple[int, ...]]
) -> bool:
    """Whether row stores to ``fake`` can be combined into runs: rows of a
    tensor in logical order with whole 128-lane rows and at least a window
    of two sublane tiles of rows, which the tiles divide."""
    tile = CompileEnvironment.current().backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
    if fake.ndim < 2 or id(fake.untyped_storage()) in lane_dense:
        return False
    rows, lanes = fake.shape[-2:]
    return (
        isinstance(rows, int)
        and isinstance(lanes, int)
        and rows % tile == 0
        and rows >= 2 * tile
        and (fake.ndim == 2 or lanes % 128 == 0)
    )


def _row_base_offset(index: object) -> tuple[torch.fx.Node, int] | None:
    """``index`` as a runtime base plus a static offset, or None."""
    if not isinstance(index, torch.fx.Node):
        return None
    if index.target in (torch.ops.aten.add.Tensor, operator.add):
        lhs, rhs = index.args[:2]
        if isinstance(lhs, torch.fx.Node) and isinstance(rhs, int):
            return lhs, rhs
        if isinstance(rhs, torch.fx.Node) and isinstance(lhs, int):
            return rhs, lhs
    return index, 0


def row_store_runs(
    graph: torch.fx.Graph, lane_dense: Mapping[int, tuple[int, ...]]
) -> dict[torch.fx.Node, RowStoreRun]:
    """The row stores of ``graph`` that write one sublane-tile-sized span of
    rows of a tensor together, as a run.

    A run is two or more stores to a tensor (``row_run_tensor``) with equal
    subscripts but for the row, at runtime rows ``base + offset`` of one
    base, offsets less than a sublane tile apart, and no other use of the
    tensor, nor a nested loop or branch, between them.  So the rows lie in
    one window of two sublane tiles, which the run reads once, patches in
    VMEM, and writes once (``_emit_hbm_resident_store``).
    """
    runs: list[list[tuple[torch.fx.Node, int]]] = []
    open_runs: dict[int, tuple[tuple[object, ...], torch.fx.Node, int]] = {}

    def close(storage: int) -> None:
        open_runs.pop(storage, None)

    for node in graph.nodes:
        if node.target in (_for_loop, _while_loop, _if):
            open_runs.clear()
            continue
        member = None
        if node.target is memory_ops.store and node.args[3] is None:
            fake = cast("torch.fx.Node", node.args[0]).meta["val"]
            index = node.args[1]
            if (
                isinstance(index, (list, tuple))
                and len(index) == fake.ndim
                and row_run_tensor(fake, lane_dense)
            ):
                row = cast("torch.fx.Node", index[fake.ndim - 2])
                split = _row_base_offset(row)
                if split is not None and is_scalar_index(row.meta.get("val")):
                    key = (*index[: fake.ndim - 2], split[0], *index[fake.ndim - 1 :])
                    member = (id(fake.untyped_storage()), key, split[1])
        for host in node.all_input_nodes:
            if host.target is _host_tensor:
                storage = id(host.meta["val"].untyped_storage())
                if member is None or member[0] != storage:
                    close(storage)
        if member is None:
            continue
        storage, key, offset = member
        tile = CompileEnvironment.current().backend.sublane_tiling(  # pyrefly: ignore[missing-attribute]
            cast("torch.fx.Node", node.args[0]).meta["val"].dtype
        )
        current = open_runs.get(storage)
        if current is not None and current[0] == key:
            run = runs[current[2]]
            offsets = [offset, *(member_offset for _, member_offset in run)]
            if max(offsets) - min(offsets) <= tile:
                run.append((node, offset))
                continue
        open_runs[storage] = (key, node, len(runs))
        runs.append([(node, offset)])
    result: dict[torch.fx.Node, RowStoreRun] = {}
    for run in runs:
        if len(run) < 2:
            continue
        low = min(offset for _, offset in run)
        for position, (node, offset) in enumerate(run):
            result[node] = RowStoreRun(
                offset, low, position == 0, position == len(run) - 1
            )
    return result


def row_store_run(state: CodegenState) -> RowStoreRun | None:
    """Where row store ``state.fx_node`` lies in a run (``row_store_runs``)."""
    device_fn = state.device_function
    node = state.fx_node
    assert node is not None
    if node.graph not in device_fn.pallas_row_store_runs:
        device_fn.pallas_row_store_runs[node.graph] = row_store_runs(
            node.graph, device_fn.pallas_lane_dense
        )
    return device_fn.pallas_row_store_runs[node.graph].get(node)


@dataclasses.dataclass
class RowForward:
    """A row store to a tensor kept in HBM, deferred to inner loop
    ``graph_id`` (``row_forward_loop``).

    ``base`` is the first row of the ``tile``-row sublane tile that holds
    the stored row.  ``starts`` write it from the store, like a store that
    is not deferred; ``forward(block)`` patches the row into ``block``, a
    ref to the loop's copy of the sublane tile, and writes it from there.
    """

    graph_id: int
    base: str
    tile: int
    starts: list[ast.stmt]
    forward: Callable[[str], list[ast.stmt]]


def row_forward_loop(state: CodegenState, fake: torch.Tensor) -> int | None:
    """The inner loop that row store ``state.fx_node`` to ``fake``, a tensor
    kept in HBM, can defer its write to, or None (``_row_forward_graph``)."""
    assert state.fx_node is not None
    return _row_forward_graph(
        state.get_graph, state.device_function, state.fx_node, fake
    )


def _row_forward_graph(
    get_graph: Callable[[int], GraphInfo],
    device_fn: DeviceFunction,
    node: torch.fx.Node,
    fake: torch.Tensor,
) -> int | None:
    """The inner loop that row store ``node`` to ``fake``, a tensor kept in
    HBM, can defer its write to, or None.

    That is the loop that uses the tensor next in the root, when it loads
    tiles of it itself (not in a nested loop): tiles of its only block over
    the rows, every other dim whole or at the integer index the store has
    there, and each sublane tile of rows inside one of them, because the
    loop's begin and block size are whole sublane tiles and the block size
    divides the rows.
    """
    storage = id(fake.untyped_storage())
    nodes = list(node.graph.nodes)
    for loop in nodes[nodes.index(node) + 1 :]:
        if loop.target is memory_ops.store:
            target = cast("torch.fx.Node", loop.args[0])
            if id(target.meta["val"].untyped_storage()) == storage:
                return None
        if loop.target is _for_loop and storage in _loop_storages(
            get_graph, cast("int", loop.args[0])
        ):
            break
    else:
        return None
    graph_id = cast("int", loop.args[0])
    graph_info = cast("ForLoopGraphInfo", get_graph(graph_id))
    graph = graph_info.graph
    nested = [
        cast("int", node.args[0])
        for node in graph.find_nodes(op="call_function", target=_for_loop)
    ]
    if len(graph_info.block_ids) != 1 or any(
        storage in _loop_storages(get_graph, nested_id) for nested_id in nested
    ):
        return None
    (block_id,) = graph_info.block_ids
    (load,) = [
        user
        for host in graph.find_nodes(op="call_function", target=_host_tensor)
        if id(host.meta["val"].untyped_storage()) == storage
        for user in host.users
        if user.target is memory_ops.load
    ]
    env = CompileEnvironment.current()
    row_dim = fake.ndim - 2
    index = cast("list[object]", load.args[1])
    stored = cast("list[object]", node.args[1])
    info = subscript_tile_info(env, index[row_dim])
    if info is None or info.block_id != block_id:
        return None
    loops = HostFunction.current().device_ir.host_loops
    for dim, (subscript, store_subscript) in enumerate(zip(index, stored, strict=True)):
        if dim == row_dim:
            continue
        whole = isinstance(subscript, slice) and subscript == slice(None)
        if whole != (
            isinstance(store_subscript, slice) and store_subscript == slice(None)
        ):
            return None
        scalar = _scalar_subscript(subscript, loops)
        if not whole and (
            scalar is None or scalar != _scalar_subscript(store_subscript, loops)
        ):
            return None
    tile = env.backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
    block = device_fn.resolved_block_size(block_id)
    (begin,) = cast("list[object]", loop.args[1])
    if (
        not isinstance(block, int)
        or not isinstance(begin, int)
        or block % tile != 0
        or begin % tile != 0
        or fake.shape[row_dim] % block != 0
    ):
        return None
    return graph_id


# Copies of tensors kept in HBM
# -----------------------------
#
# A whole-slice load of a tensor kept in HBM in a root's own body copies the
# slice into a VMEM buffer of its own, and an inner loop that loads tiles of
# one streams them through two buffers of its own.  Where the program reaches
# them, both copies queue behind whatever the weight rings have in flight;
# ``_plan_early_copies`` starts them at their place in the rings' copy order
# instead when it can.


@dataclasses.dataclass
class HbmCopy:
    """A DMA from a tensor kept in HBM into a VMEM buffer of its own that a
    root starts: the copy of a whole-slice load in its own body, or of the
    first tile an inner loop in its body streams."""

    fake: torch.Tensor
    root: int
    # The root's graph, and the index in it of the node that starts the
    # copy: the load, or the inner loop.
    graph_id: int
    node: int
    # The folded host loops around the root.
    loops: tuple[HostLoopInfo, ...]
    # Per dim of the tensor: an integer index (the copy drops the dim), the
    # ``(start, size)`` of a slice, or None (whole).
    parts: tuple[sympy.Expr | tuple[sympy.Expr, int] | None, ...]
    # The destination ref and its semaphore.
    buffer: str
    semaphore: str
    # The iterations of the folded loop, counted from its start (0 without
    # one), whose copy ``_plan_early_copies`` starts ahead.
    early: set[int] = dataclasses.field(default_factory=set)
    # The dim order of a tensor passed lane dense, which the copy and its
    # buffer take.
    perm: tuple[int, ...] | None = None

    @property
    def loop(self) -> HostLoopInfo | None:
        return self.loops[0] if len(self.loops) == 1 else None

    def scalar(self, dim: int, layer: int) -> int | None:
        """The integer index of dim ``dim`` in iteration ``layer``, if any."""
        part = self.parts[dim]
        if isinstance(part, tuple):
            part, size = part
            if size != 1:
                return None
        if part is None:
            return None
        loop = self.loop
        if loop is not None:
            part = part.xreplace({loop.symbol: sympy.Integer(loop.start + layer)})
        return int(part) if part.is_Integer else None

    def copy(self, device_fn: DeviceFunction, layer: str | int | None) -> str:
        """``make_async_copy`` of the copy in iteration ``layer`` of the
        folded loop, or inside its body when ``None``."""
        name = device_fn.tensor_arg(self.fake).name
        shape = [int(size) for size in self.fake.shape]
        if self.perm is not None:
            shape = physical_order(shape, self.perm)
        parts = self.source_parts(device_fn, layer)
        if len(shape) > 2 and shape[-1] % 128 != 0:
            # ``_slice_load_indices`` made every leading dim an integer.
            source = ragged_rows_ref(name, shape, parts[:-2])
        elif all(part == ":" for part in parts):
            source = name
        else:
            source = f"{name}.at[{', '.join(parts)}]"
        return f"pltpu.make_async_copy({source}, {self.buffer}, {self.semaphore})"

    def source_parts(
        self, device_fn: DeviceFunction, layer: str | int | None
    ) -> list[str]:
        """The subscript per dim, in physical order, of the slice the copy
        reads in iteration ``layer`` of the folded loop (as ``copy``)."""
        copy_parts = list(self.parts)
        if self.perm is not None:
            copy_parts = physical_order(copy_parts, self.perm)

        def index(expr: sympy.Expr) -> str:
            return _layer_index(device_fn, self.loop, expr, layer)

        return [
            ":"
            if part is None
            else f"pl.ds({index(part[0])}, {part[1]})"
            if isinstance(part, tuple)
            else index(part)
            for part in copy_parts
        ]

    def started(
        self, device_fn: DeviceFunction, statements: list[ast.stmt]
    ) -> list[ast.stmt]:
        """``statements``, which start the copy where the program reaches
        it, for the iterations of the folded loop that do not start it
        ahead."""
        if not self.early:
            return statements
        loop = self.loop
        layers = 1 if loop is None else loop.trips
        late = [layer for layer in range(layers) if layer not in self.early]
        if not late:
            return []
        assert loop is not None
        var = device_fn.expr_to_var_info[loop.symbol].name
        conditions = [
            f"{var} == {loop.start + lo}"
            if lo == hi
            else f"({var} >= {loop.start + lo}) & ({var} <= {loop.start + hi})"
            for lo, hi in _merge_intervals([(layer, layer) for layer in late])
        ]
        condition = (
            " | ".join(f"({c})" for c in conditions)
            if len(conditions) > 1
            else conditions[0]
        )
        return _when(device_fn, condition, statements, name="_copy")


@dataclasses.dataclass(frozen=True)
class _EarlyRun:
    """Copies that the refill starting stream index ``index + k * stride``
    of a ring starts ahead, for iteration ``first + k`` of the folded loop
    around their root, ``0 <= k < count``."""

    copy: HbmCopy
    first: int
    count: int
    index: int
    stride: int


def _load_ordinal(load: torch.fx.Node, storage: int) -> int:
    """How many loads of the tensor of ``storage`` precede ``load`` in its
    graph."""
    ordinal = 0
    for node in load.graph.nodes:
        if node is load:
            break
        if (
            node.target is memory_ops.load
            and id(cast("torch.fx.Node", node.args[0]).meta["val"].untyped_storage())
            == storage
        ):
            ordinal += 1
    return ordinal


def ragged_rows_ref(name: str, shape: Sequence[int], indices: Sequence[str]) -> str:
    """The ``[rows, cols]`` slice at integer indices ``indices`` (of every
    leading dim) of ref ``name``, a ``[*leading, rows, cols]`` tensor of a
    ragged minor dim, as a slice of a view of it as rows of that dim: Mosaic
    slices a ref of more than two dims only in whole VMEM tiles of its minor
    dims (see ``Ring.copy``)."""
    terms = [
        _scaled(
            int(index) if index.isdigit() else f"({index})",
            math.prod(shape[dim + 1 : -1]),
        )
        for dim, index in enumerate(indices)
    ]
    start = " + ".join(str(term) for term in terms if term != 0) or "0"
    view = f"{name}.reshape({math.prod(shape[:-1])}, {shape[-1]})"
    return f"{view}.at[pl.ds({start}, {shape[-2]}), :]"


def _plan_hbm_copies(
    plan: MegakernelPlan, device_ir: DeviceIR, device_fn: DeviceFunction
) -> None:
    """Register the buffers of the whole-slice loads of the tensors kept in
    HBM, and start their copies and the inner loops' first copies early
    where ``_plan_early_copies`` can.  The row gather tables have row reads
    of their own (``row_read``).

    A root reads a slice when its copy lands, so when VMEM bounds the rings
    (``plan.pool_slices``) the roots share the buffers: the ``n``-th load
    of a shape and dtype in each root's body copies into the ``n``-th
    buffer of that shape and dtype, and an early copy waits until the root
    before it that reads the buffer is done.  Per-layer state of a model
    whose host loop holds several layers (one slice per sequence and layer)
    then takes one layer's buffers, not one per layer of the loop body, and
    leaves the rings the rest.  Otherwise each load has a buffer of its own,
    which its copy can fill as early as the previous iteration's read."""
    resident = plan.hbm_resident - _row_gather_tables(device_ir).keys()
    graphs = {graph_info.graph_id: graph_info for graph_info in device_ir.graphs}
    host_loops = _graph_host_loops(device_ir)
    host = HostFunction.current()
    early: list[HbmCopy] = []
    loop_primes: dict[tuple[int, int], tuple[HbmCopy, tuple[int, ...], bool]] = {}
    buffers: dict[tuple[tuple[int, ...], torch.dtype], list[str]] = {}
    for root, graph_id in enumerate(device_ir.root_ids):
        loops = tuple(host_loops[graph_id])
        single = _root_tiles(device_ir, device_fn, root) == 1
        root_loads: collections.Counter[tuple[tuple[int, ...], torch.dtype]] = (
            collections.Counter()
        )
        for position, node in enumerate(graphs[graph_id].graph.nodes):
            if node.op != "call_function":
                continue
            if node.target is _for_loop:
                if single:
                    for storage, prime in _loop_prime_copies(
                        device_fn,
                        graphs,
                        node,
                        root,
                        graph_id,
                        position,
                        loops,
                        resident,
                    ):
                        loop_primes[(cast("int", node.args[0]), storage)] = prime
                        early.append(prime[0])
                continue
            if node.target is not memory_ops.load:
                continue
            fake = cast("torch.fx.Node", node.args[0]).meta["val"]
            storage = id(fake.untyped_storage())
            if storage not in resident:
                continue
            perm = device_fn.pallas_lane_dense.get(storage)
            scalars = _slice_load_indices(fake, node, loops, perm)
            assert scalars is not None
            # A tensor passed lane dense is copied in its physical dim order,
            # which keeps the integer-indexed leading dims in place.
            shape = [int(size) for size in fake.shape]
            if perm is not None:
                shape = physical_order(shape, perm)
            shape = [size for dim, size in enumerate(shape) if dim not in scalars]
            hint = host.tensor_to_origin[fake].suggest_var_name()
            key = (tuple(shape), fake.dtype)
            pool = buffers.setdefault(key, [])
            if not plan.pool_slices or root_loads[key] == len(pool):
                pool.append(
                    device_fn.register_scratch(
                        tuple(shape), fake.dtype, name_hint=f"{hint}_slice"
                    )
                )
            copy = HbmCopy(
                fake,
                root,
                graph_id,
                position,
                loops,
                tuple(scalars.get(dim) for dim in range(fake.ndim)),
                pool[root_loads[key]] if plan.pool_slices else pool[-1],
                device_fn.register_dma_semaphore(name_hint=f"{hint}_slice_sem"),
                perm=perm,
            )
            root_loads[key] += 1
            plan.slice_loads[(graph_id, storage, _load_ordinal(node, storage))] = copy
            if single:
                early.append(copy)
    _plan_early_copies(plan, device_ir, device_fn, early)
    for (loop_graph_id, storage), (copy, vmem_shape, dynamic) in loop_primes.items():
        if not copy.early:
            continue
        hint = host.tensor_to_origin[copy.fake].suggest_var_name()
        scratch = device_fn.register_scratch(
            (2, *vmem_shape), copy.fake.dtype, name_hint=f"{hint}_buf"
        )
        semaphore = device_fn.register_dma_semaphore(
            name_hint=f"{hint}_sem", shape=(2,)
        )
        copy.buffer = f"{scratch}.at[0]"
        copy.semaphore = f"{semaphore}.at[0]"
        plan.loop_primes[(loop_graph_id, storage)] = LoopPrime(
            copy, vmem_shape, DmaResources(scratch, semaphore, 2), dynamic
        )


# A program position: the run of a root (``_root_run``), its tile, the index
# of a node in its graph, and an inner loop iteration and load.  It sorts
# after every node of the root.
_END = 1 << 30


def _plan_early_copies(
    plan: MegakernelPlan,
    device_ir: DeviceIR,
    device_fn: DeviceFunction,
    copies: Sequence[HbmCopy],
) -> None:
    """Start each of ``copies``, in each iteration of the folded loop around
    its root, just before the first copy the rings start of the stream
    indices read after it, so the DMA engine runs it in the order the
    program reads the data; the iterations go in ``copy.early``.

    That needs the destination free there (the root that reads it before,
    in the previous iteration or, for a buffer roots share
    (``_plan_hbm_copies``), another root, is done with it; ring copies
    started before that are passed over) and no
    write to the source in between: every store to the tensor from the root
    before the one that starts the ring copy (whose writes may be pending
    until the end of the next root, ``MegakernelPlan.finish_root``) up to
    the copy's own place must be proven disjoint by an integer index that
    differs, or be a row store that inner loop forwards into its copy
    (``_row_forward_graph``).  The other iterations start the copy where the
    program reaches it.
    """
    loops = device_ir.host_loops
    if not plan.stream or not copies:
        return
    if any(
        sum(root in loop.roots for loop in loops) > 1
        for root in range(len(device_ir.root_ids))
    ):
        return
    loop_of = {root: loop for loop in loops for root in loop.roots}

    def run(root: int, layer: int) -> tuple[int, int, int]:
        """When root ``root`` runs, in iteration ``layer`` of its loop."""
        loop = loop_of.get(root)
        return (root, 0, root) if loop is None else (loop.first_root, layer, root)

    graphs = {graph_info.graph_id: graph_info for graph_info in device_ir.graphs}
    # The node index in its root's graph of each loop's outermost loop, by
    # graph id, and of each root-level load.
    loop_nodes: dict[int, int] = {}
    load_nodes: dict[tuple[int, int], int] = {}
    for graph_id in device_ir.root_ids:
        for position, node in enumerate(graphs[graph_id].graph.nodes):
            if node.target is memory_ops.load:
                tensor = cast("torch.fx.Node", node.args[0])
                load_nodes.setdefault((graph_id, id(tensor.meta["val"])), position)
            elif node.target is _for_loop:
                pending = [cast("int", node.args[0])]
                while pending:
                    loop_graph_id = pending.pop()
                    loop_nodes[loop_graph_id] = position
                    pending.extend(
                        cast("int", inner.args[0])
                        for inner in graphs[loop_graph_id].graph.find_nodes(
                            op="call_function", target=_for_loop
                        )
                    )

    # Per stream index of every ring: where the program reads it, and where
    # the ring starts its copy (None at a handoff, ``_ring_predecessor``).
    reads: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []
    refills: list[dict[int, tuple[tuple[int, ...], tuple[str, int, int]]]] = []
    for order, ring in enumerate(plan.stream):
        consumed: dict[int, tuple[int, ...]] = {}
        refilled: dict[int, tuple[tuple[int, ...], tuple[str, int, int]]] = {}
        for site in ring.sites:
            root = site.root
            loads = len(site.loads)
            graph_id = device_ir.root_ids[root.root]
            for layer, tile, iteration, load in itertools.product(
                range(root.layers),
                range(root.num_tiles),
                range(site.trips),
                range(loads),
            ):
                index = (
                    root.base
                    + layer * root.layer_stride
                    + tile * root.stride
                    + site.offset
                    + iteration * loads
                    + load
                )
                key = run(root.root, layer)
                if site.block_id is None:
                    node = load_nodes[(graph_id, id(site.loads[load].fake))]
                    consumed[index] = (*key, tile, node, 0, load)
                    # ``root_refills``, at the end of the tile.
                    issue = (*key, tile, _END, 0, load)
                else:
                    consumed[index] = (
                        *key,
                        tile,
                        loop_nodes[site.graph_id],
                        iteration,
                        load,
                    )
                    issue = consumed[index]
                refilled[index + ring.depth] = (
                    issue,
                    (ring.buffer, site.graph_id, load),
                )
        prologue = _ring_predecessor(plan.stream, ring) is None
        for index, position in consumed.items():
            if index < ring.depth:
                if prologue:
                    reads.append(
                        (position, (-1, order, index, 0, 0, 0, 0), order, index)
                    )
            else:
                reads.append((position, refilled[index][0], order, index))
        refills.append(refilled)
    reads.sort()
    positions = [position for position, _, _, _ in reads]

    stores: dict[int, list[tuple[int, int, torch.fx.Node]]] = {}
    for root, graph_id in enumerate(device_ir.root_ids):
        for position, node in enumerate(graphs[graph_id].graph.nodes):
            if node.target is memory_ops.store:
                fake = cast("torch.fx.Node", node.args[0]).meta["val"]
                stores.setdefault(id(fake.untyped_storage()), []).append(
                    (root, position, node)
                )

    # The runs of the roots that read each slice buffer, in program order.
    readers: dict[str, list[tuple[int, int, int]]] = {}
    for copy in plan.slice_loads.values():
        loop = loop_of.get(copy.root)
        readers.setdefault(copy.buffer, []).extend(
            run(copy.root, layer) for layer in range(1 if loop is None else loop.trips)
        )
    for reader_runs in readers.values():
        reader_runs.sort()

    # (copy number, iteration, stream index, copy); the first two are unique.
    runs: dict[tuple[str, int, int], list[tuple[int, int, int, HbmCopy]]] = {}
    for number, copy in enumerate(copies):
        loop = loop_of.get(copy.root)
        storage = id(copy.fake.untyped_storage())
        # The inner loop that primes the copy, if it does.
        node = list(graphs[copy.graph_id].graph.nodes)[copy.node]
        forwarded = cast("int", node.args[0]) if node.target is _for_loop else None
        for layer in range(1 if loop is None else loop.trips):
            key = run(copy.root, layer)
            position = (*key, 0, copy.node, -1, -1)
            # The first copy the rings start, once the root that reads the
            # destination before is done with it, of the stream indices read
            # after this copy's place; those started before precede it anyway.
            before = readers.get(copy.buffer) or [
                run(copy.root, earlier) for earlier in range(layer)
            ]
            previous = bisect.bisect_left(before, key)
            free = (*before[previous - 1], 0, _END) if previous else None
            first = min(
                (
                    (issue, order, index)
                    for _, issue, order, index in itertools.islice(
                        reads, bisect.bisect_left(positions, position), None
                    )
                    if free is None or issue > free
                ),
                default=None,
            )
            if first is None or first[0] >= position:
                continue
            issue, order, index = first
            low = issue[:3] if issue[0] >= 0 else (-1, 0, 0)
            if issue[0] >= 0 and issue[2] > 0:
                previous = issue[2] - 1
                if loop_of.get(previous) is loop_of.get(issue[2]):
                    low = run(previous, issue[1])
            if not _early_stores_safe(
                device_fn,
                graphs,
                copy,
                layer,
                [
                    (store, store_layer, store_key == key)
                    for store_root, store_position, store in stores.get(storage, ())
                    for store_layer in range(
                        1 if store_root not in loop_of else loop_of[store_root].trips
                    )
                    if low <= (store_key := run(store_root, store_layer)) <= key
                    and (store_key != key or store_position < copy.node)
                ],
                forwarded,
            ):
                continue
            copy.early.add(layer)
            if issue[0] < 0:
                plan.early_prologue.setdefault(
                    (plan.stream[order].buffer, index), []
                ).append((copy, layer))
            else:
                site_key = refills[order][index][1]
                runs.setdefault(site_key, []).append((number, layer, index, copy))
    for site_key, items in runs.items():
        grouped: list[_EarlyRun] = []
        for _, layer, index, copy in sorted(items):
            last = grouped[-1] if grouped else None
            if (
                last is not None
                and last.copy is copy
                and layer == last.first + last.count
            ):
                stride = index - last.index if last.count == 1 else last.stride
                if stride > 0 and index == last.index + last.count * stride:
                    grouped[-1] = dataclasses.replace(
                        last, count=last.count + 1, stride=stride
                    )
                    continue
            grouped.append(_EarlyRun(copy, layer, 1, index, 1))
        plan.early_refills[site_key] = grouped


def _early_stores_safe(
    device_fn: DeviceFunction,
    graphs: Mapping[int, GraphInfo],
    copy: HbmCopy,
    layer: int,
    stores: Sequence[tuple[torch.fx.Node, int, bool]],
    forwarded: int | None,
) -> bool:
    """Whether the stores ``stores`` between where ``_plan_early_copies``
    starts ``copy`` for iteration ``layer`` and where the program reaches it
    leave the copy's source alone.

    Each is ``(node, store_layer, same_run)``: a store in iteration
    ``store_layer`` of the folded loop around its root, in the copy's root
    run or not.  It must store at an integer index that differs from the
    copy's, or be a row store in the copy's root run that the copy's inner
    loop ``forwarded`` patches into its tiles (``_row_forward_graph``).
    """
    fake = copy.fake
    loops = HostFunction.current().device_ir.host_loops
    for node, store_layer, same_run in stores:
        subscripts = tensor_dim_subscripts(
            cast("list[object]", node.args[1]), fake.ndim
        )
        disjoint = False
        for dim, subscript in enumerate(subscripts):
            ours = copy.scalar(dim, layer)
            theirs = _scalar_subscript(subscript, loops)
            if ours is None or theirs is None:
                continue
            for loop in loops:
                theirs = theirs.xreplace(
                    {loop.symbol: sympy.Integer(loop.start + store_layer)}
                )
            if theirs.is_Integer and int(theirs) != ours:
                disjoint = True
                break
        if disjoint:
            continue
        if (
            forwarded is None
            or not same_run
            or _row_forward_graph(graphs.__getitem__, device_fn, node, fake)
            != forwarded
        ):
            return False
    return True


def _root_tiles(device_ir: DeviceIR, device_fn: DeviceFunction, root: int) -> int:
    """How many tiles root ``root`` walks."""
    tiles = 1
    for axis in device_ir.task_families[root].axes:
        block_size = device_fn.resolved_block_size(axis.block_id)
        if not isinstance(block_size, int):
            return 0
        tiles *= -(-_static_extent(axis) // block_size)
    return tiles


def _loop_prime_copies(
    device_fn: DeviceFunction,
    graphs: Mapping[int, GraphInfo],
    node: torch.fx.Node,
    root: int,
    root_graph_id: int,
    position: int,
    loops: tuple[HostLoopInfo, ...],
    hbm_resident: frozenset[int],
) -> list[tuple[int, tuple[HbmCopy, tuple[int, ...], bool]]]:
    """The first copy of each tile stream of a tensor kept in HBM that inner
    loop ``node`` in the body of root ``root`` primes, with the shape of one
    of its two buffers and whether the loop's end is dynamic, by storage:
    when the loop starts at a static index and its block size is known, so
    the copy's address does not depend on the code before the loop.  A loop
    of a dynamic end (``hl.tile(t + 1)``) may run no iteration; its first
    copy starts anyway and is waited after the loop.  The buffers are
    registered if the copy starts early."""
    graph_info = cast("ForLoopGraphInfo", graphs[cast("int", node.args[0])])
    (begin, *more_begins) = cast("list[object]", node.args[1])
    (end, *more_ends) = cast("list[object]", node.args[2])
    if (
        more_begins
        or more_ends
        or len(graph_info.block_ids) != 1
        or not isinstance(begin, int)
        or (isinstance(end, int) and end <= begin)
    ):
        return []
    (block_id,) = graph_info.block_ids
    block_size = device_fn.resolved_block_size(block_id)
    if not isinstance(block_size, int):
        return []
    env = CompileEnvironment.current()
    primes: list[tuple[int, tuple[HbmCopy, tuple[int, ...], bool]]] = []
    for host in graph_info.graph.find_nodes(op="call_function", target=_host_tensor):
        fake = host.meta["val"]
        storage = id(fake.untyped_storage())
        loads = [user for user in host.users if user.target is memory_ops.load]
        if storage not in hbm_resident or len(loads) != 1:
            continue
        # ``_is_tile_load``: every dim whole, an integer, or a plain tile.
        (load,) = loads
        parts: list[sympy.Expr | tuple[sympy.Expr, int] | None] = []
        shape: list[int] = []
        for dim, subscript in enumerate(cast("list[object]", load.args[1])):
            scalar = _scalar_subscript(subscript, loops)
            if isinstance(subscript, slice) and subscript == slice(None):
                parts.append(None)
                shape.append(int(fake.shape[dim]))
            elif scalar is not None:
                parts.append((scalar, 1))
                shape.append(1)
            else:
                info = subscript_tile_info(env, subscript)
                if info is None or info.block_id != block_id:
                    break
                parts.append((sympy.Integer(begin), block_size))
                shape.append(block_size)
        else:
            copy = HbmCopy(
                fake, root, root_graph_id, position, loops, tuple(parts), "", ""
            )
            primes.append((storage, (copy, tuple(shape), not isinstance(end, int))))
    return primes


@dataclasses.dataclass(frozen=True)
class LoopPrime:
    """The first copy of an inner loop's stream of a tensor kept in HBM,
    which ``_plan_early_copies`` starts early, with the shape of each of the
    loop's two buffers of the stream and the buffers.  The copy of a loop of
    a ``dynamic`` end starts even when the loop runs no iteration."""

    copy: HbmCopy
    vmem_shape: tuple[int, ...]
    resources: DmaResources
    dynamic: bool


# Inner loop tiles
# ----------------
#
# An inner loop over a block id that the weight ring does not stream loads
# tensors kept whole in VMEM, which cost no copies, or streams tiles of the
# others through DMA buffers of its own.  ``LoopTileModel`` picks the default
# size of its block id.

# Below this many bytes, a fixed per-copy cost keeps an inner loop's DMA from
# streaming at full bandwidth.
_LOOP_STREAM_MIN_SLOT_BYTES = 256 * 1024


@dataclasses.dataclass(frozen=True)
class LoopTiles:
    """What the default block size of one inner loop depends on.

    ``values`` and ``streams`` hold ``(numel, itemsize)``, the numel in the
    block size symbols, of each value the loop body computes and of each
    tile it streams.  ``streams`` is empty when every tensor the loop loads
    is kept whole in VMEM.
    """

    block_id: int
    # The elements the loop can reach: its extent, or the size of the dims
    # it tiles when it stops at a runtime value.
    extent: int
    values: tuple[tuple[sympy.Expr, int], ...]
    streams: tuple[tuple[sympy.Expr, int], ...]
    # Whether the loop spans a static extent while the weight rings stream
    # the sites of later roots (or of the next iteration of the folded loop
    # around its root): then its copies queue behind the rings' in flight.
    behind_rings: bool = False


@dataclasses.dataclass(frozen=True)
class LoopTileModel:
    """The config-independent part of a megakernel's inner loop blocks."""

    loops: tuple[LoopTiles, ...]
    # The block size symbol of each block id.
    symbols: Mapping[int, sympy.Expr]
    # VMEM left for one loop's DMA buffers and values.
    budget: int

    def seed_block_sizes(
        self, legal: Mapping[int, Sequence[int]], base: Mapping[int, int]
    ) -> dict[int, int]:
        """Default sizes of the loops' block ids, given the sizes ``legal``
        the config spec allows each and the size ``base`` of every block id.

        A loop that loads only tensors kept whole in VMEM spans its whole
        extent, in the fewest iterations.  A loop that streams takes the
        smallest block whose tile of each streamed tensor fills
        ``_LOOP_STREAM_MIN_SLOT_BYTES``: a larger tile streams no faster but
        lengthens the first copy, which nothing overlaps, and the rows read
        past a runtime end.  Except behind the rings (``behind_rings``): its
        first copies start among the rings' (``_loop_prime_copies``), but
        every later one waits for the ring copies in flight, so it spans its
        whole extent too.  Either way the block shrinks until two buffers
        per streamed tile and the values of the loop body fit in ``budget``.
        """
        sizes: dict[int, int] = {}
        for loop in self.loops:
            choices = sorted(legal[loop.block_id])
            if not choices:
                continue
            whole = next((size for size in choices if size >= loop.extent), choices[-1])
            choices = [size for size in choices if size <= whole]
            target = whole
            if loop.streams and not loop.behind_rings:
                target = next(
                    (
                        size
                        for size in choices
                        if min(
                            self._nbytes(loop.streams, {**base, loop.block_id: size})
                        )
                        >= _LOOP_STREAM_MIN_SLOT_BYTES
                    ),
                    whole,
                )
            fitting = [
                size
                for size in choices
                if size <= target
                and self._vmem_bytes(loop, {**base, loop.block_id: size}) <= self.budget
            ]
            size = fitting[-1] if fitting else choices[0]
            sizes[loop.block_id] = min(sizes.get(loop.block_id, size), size)
        return sizes

    def buffer_bytes(self, block_sizes: Mapping[int, int]) -> int:
        """VMEM the loops' DMA buffers take at ``block_sizes``: two per
        streamed tile, every loop its own."""
        return sum(
            2 * sum(self._nbytes(loop.streams, block_sizes)) for loop in self.loops
        )

    def _vmem_bytes(self, loop: LoopTiles, block_sizes: Mapping[int, int]) -> int:
        """VMEM ``loop`` takes at ``block_sizes``: two DMA buffers per
        streamed tile, and every value of its body."""
        return 2 * sum(self._nbytes(loop.streams, block_sizes)) + sum(
            self._nbytes(loop.values, block_sizes)
        )

    def _nbytes(
        self,
        exprs: Sequence[tuple[sympy.Expr, int]],
        block_sizes: Mapping[int, int],
    ) -> list[int]:
        """Bytes of each ``(numel, itemsize)`` at ``block_sizes``; 0 for a
        numel that depends on a runtime value."""
        replacements = {
            self.symbols[block_id]: sympy.Integer(size)
            for block_id, size in block_sizes.items()
            if block_id in self.symbols
        }
        nbytes: list[int] = []
        for numel, itemsize in exprs:
            value = numel.xreplace(replacements)
            nbytes.append(int(value) * itemsize if value.is_Integer else 0)
        return nbytes


def _numel_bytes(fake: torch.Tensor) -> tuple[sympy.Expr, int]:
    """``(numel, itemsize)`` of ``fake``, the numel in the block size symbols."""
    numel = fake.numel()
    expr = numel._sympy_() if isinstance(numel, torch.SymInt) else sympy.Integer(numel)
    return expr, fake.dtype.itemsize


def _build_loop_tile_model(
    device_ir: DeviceIR,
    stream: StreamModel | None,
    keep_hbm: bool,
    lane_dense: Mapping[int, tuple[int, ...]],
) -> LoopTileModel | None:
    """Find the inner loops whose default block size ``LoopTileModel`` picks:
    those over one tuned block id that the weight ``stream`` does not shape,
    which load tiles over it, in the default config, which keeps the tensors
    that can stay in HBM there (``keep_hbm``) or not, and passes the inputs
    ``lane_dense`` lane dense."""
    env = CompileEnvironment.current()
    uses = _host_tensor_uses(device_ir)
    _, minimum = _stream_block_size_bounds()
    ring_block_ids: set[int] = set()
    streamed: set[int] = set()
    if stream is not None:
        ring_block_ids = stream.block_ids()
        streamed = {
            id(load.fake.untyped_storage())
            for site in stream.sites
            for load in site.loads
        }
    whole = _whole_vmem_bytes(device_ir, uses, streamed, lane_dense)
    if not keep_hbm:
        whole.update(_hbm_resident_whole_bytes(device_ir, lane_dense))
    # The root around each inner loop.
    graphs = {info.graph_id: info for info in device_ir.graphs}
    loop_roots: dict[int, int] = {}
    for root, root_graph_id in enumerate(device_ir.root_ids):
        pending = [root_graph_id]
        while pending:
            for for_node in graphs[pending.pop()].graph.find_nodes(
                op="call_function", target=_for_loop
            ):
                loop_roots[cast("int", for_node.args[0])] = root
                pending.append(cast("int", for_node.args[0]))
    loops: list[LoopTiles] = []
    for graph_id, (graph_info, node) in _root_loops(device_ir).items():
        if len(graph_info.block_ids) != 1:
            continue
        (block_id,) = graph_info.block_ids
        if block_id not in minimum or block_id in ring_block_ids:
            continue
        graph = graph_info.graph
        tiled_sizes: list[int] = []
        streams: list[tuple[sympy.Expr, int]] = []
        for host in graph.find_nodes(op="call_function", target=_host_tensor):
            fake = host.meta["val"]
            for user in host.users:
                if user.target is not memory_ops.load or user.args[0] is not host:
                    continue
                index = cast("list[object]", user.args[1])
                sizes = [
                    int(size)
                    for size, subscript in zip(
                        fake.shape,
                        tensor_dim_subscripts(index, fake.ndim),
                        strict=True,
                    )
                    if subscript_index_scale(env, subscript)[0] == block_id
                ]
                tiled_sizes.extend(sizes)
                if sizes and id(fake.untyped_storage()) not in whole:
                    streams.append(_numel_bytes(user.meta["val"]))
        if not tiled_sizes:
            continue
        (begin,) = cast("list[object]", node.args[1])
        (end,) = cast("list[object]", node.args[2])
        behind_rings = False
        if isinstance(begin, int) and isinstance(end, int):
            extent = end - begin
            if stream is not None:
                root = loop_roots[graph_id]
                folded = stream.loop_of(root)
                behind_rings = any(
                    site.root > root
                    or (folded is not None and site.root in folded.roots)
                    for site in stream.sites
                )
        else:
            extent = max(tiled_sizes) - (begin if isinstance(begin, int) else 0)
        values = tuple(
            _numel_bytes(value.meta["val"])
            for value in graph.nodes
            if value.op in ("placeholder", "call_function")
            and value.target is not _host_tensor
            and isinstance(value.meta.get("val"), torch.Tensor)
        )
        loops.append(LoopTiles(block_id, extent, values, tuple(streams), behind_rings))
    if not loops:
        return None
    # What the rings and the tensors kept whole in VMEM leave of the VMEM the
    # plan sizes; the rings' default depth puts about
    # ``_STREAM_TARGET_BYTES`` in them, or what is left if less.
    budget = _vmem_capacity() - _STREAM_RESERVE_BYTES - sum(whole.values())
    if keep_hbm:
        budget -= _hbm_slot_bytes(device_ir, lane_dense)
    if stream is not None:
        budget -= min(_STREAM_TARGET_BYTES, max(budget, 0))
    symbols = {info.block_id: info.var._sympy_() for info in env.block_sizes}
    return LoopTileModel(tuple(loops), symbols, budget)


# Loop-carried scratch
# --------------------
#
# Each value an inner loop carries lives in a VMEM scratch buffer of its own
# (``_setup_loop_carried_state``), which later roots take over
# (``MegakernelPlan.share_root_scratch``).  Its size follows the config's
# block sizes, so the rings count it per config, next to the inner loops' DMA
# buffers.


@dataclasses.dataclass(frozen=True)
class CarriedScratch:
    """The config-independent part of the roots' loop-carried scratch.

    ``roots`` holds, per root, the shape (in the block size symbols), dtype
    and native sublane tile of each value its inner loops carry.
    """

    roots: tuple[tuple[tuple[tuple[sympy.Expr, ...], torch.dtype, int], ...], ...]
    # The block size symbol of each block id.
    symbols: Mapping[int, sympy.Expr]

    def nbytes(self, block_sizes: Mapping[int, int]) -> int:
        """VMEM the scratch takes at ``block_sizes``, as the roots share it:
        of each shape and dtype, as many buffers as the root that carries
        the most of them, each padded to whole (sublane, 128) VMEM tiles;
        nothing for a shape that depends on a runtime value."""
        replacements = {
            self.symbols[block_id]: sympy.Integer(size)
            for block_id, size in block_sizes.items()
            if block_id in self.symbols
        }
        buffers: collections.Counter[tuple[tuple[int, ...], torch.dtype, int]] = (
            collections.Counter()
        )
        for values in self.roots:
            counts: collections.Counter[tuple[tuple[int, ...], torch.dtype, int]] = (
                collections.Counter()
            )
            for shape, dtype, sublane in values:
                dims = [dim.xreplace(replacements) for dim in shape]
                if all(dim.is_Integer for dim in dims):
                    counts[(tuple(int(dim) for dim in dims), dtype, sublane)] += 1
            buffers |= counts
        return sum(
            count * _tiled_bytes(shape, dtype.itemsize, sublane)
            for (shape, dtype, sublane), count in buffers.items()
        )


def _build_carried_scratch(device_ir: DeviceIR) -> CarriedScratch | None:
    """Find the values the inner loops of each root carry: the initial values
    that a ``_phi`` after the loop merges with its result.  None when no
    loop carries any."""
    env = CompileEnvironment.current()
    fixed, _ = _stream_block_size_bounds()
    symbols = {info.block_id: info.var._sympy_() for info in env.block_sizes}
    pinned = {
        symbols[block_id]: sympy.Integer(size) for block_id, size in fixed.items()
    }
    graphs = {info.graph_id: info for info in device_ir.graphs}
    roots: list[tuple[tuple[tuple[sympy.Expr, ...], torch.dtype, int], ...]] = []
    for root_graph_id in device_ir.root_ids:
        values: list[tuple[tuple[sympy.Expr, ...], torch.dtype, int]] = []
        pending = [root_graph_id]
        while pending:
            graph = graphs[pending.pop()].graph
            for node in graph.nodes:
                if node.op != "call_function" or node.target not in (
                    _for_loop,
                    _for_loop_step,
                ):
                    continue
                pending.append(cast("int", node.args[0]))
                carried = {
                    phi.args[0]
                    for item in node.users
                    for phi in item.users
                    if phi.op == "call_function" and phi.target is _phi
                }
                for arg in cast("list[object]", node.args[3]):
                    if arg not in carried:
                        continue
                    assert isinstance(arg, torch.fx.Node)
                    fake = arg.meta.get("val")
                    if isinstance(fake, torch.Tensor):
                        values.append(
                            (
                                tuple(
                                    sympy.sympify(size).xreplace(pinned)
                                    for size in fake.shape
                                ),
                                fake.dtype,
                                env.backend.sublane_tiling(fake.dtype),  # pyrefly: ignore[missing-attribute]
                            )
                        )
        roots.append(tuple(values))
    if not any(roots):
        return None
    return CarriedScratch(tuple(roots), symbols)


# A tensor kept whole in VMEM takes the extent its tiles reach
# (``_ref_extents``), not just its own shape: a [1, 5120] f32 activation read
# in 16-row tiles takes 16 rows, of which ``StreamModel.resident_bytes`` counts
# the 8 of one sublane tile.  The excess follows the config's block sizes, so
# the rings count it per config, next to the loop-carried scratch.


@dataclasses.dataclass(frozen=True)
class TileOverhang:
    """The config-independent part of the tile overhang of the tensors kept
    whole in VMEM.

    ``tensors`` holds, per tensor, its shape, itemsize, native sublane tile,
    the copies of it the launcher keeps (``_whole_vmem_bytes``) and, per dim,
    the ``(block id, block size if fixed, scale, offset, noncanonical)`` of
    each tile that indexes it.
    """

    tensors: tuple[
        tuple[
            tuple[int, ...],
            int,
            int,
            int,
            tuple[tuple[tuple[int, int | None, int, int, bool], ...], ...],
        ],
        ...,
    ]
    # Block ids of a fixed size (hl.grid), with that size.
    fixed_block_sizes: Mapping[int, int]

    def nbytes(self, block_sizes: Mapping[int, int]) -> int:
        """VMEM the tensors take past their own shape at ``block_sizes``,
        each dim extended as ``_ref_extents`` extends it; nothing for a tile
        of a block id without a size."""
        sizes = {**self.fixed_block_sizes, **block_sizes}
        total = 0
        for shape, itemsize, sublane, copies, dims in self.tensors:
            extent = list(shape)
            for dim, tiles in enumerate(dims):
                for block_id, fixed, scale, offset, noncanonical in tiles:
                    block_size = sizes.get(block_id) if fixed is None else fixed
                    if block_size is None:
                        continue
                    reach = -(-shape[dim] // block_size) * block_size
                    if noncanonical:
                        reach += block_size
                    extent[dim] = max(extent[dim], scale * reach + max(offset, 0))
            total += copies * (
                _tiled_bytes(extent, itemsize, sublane)
                - _tiled_bytes(shape, itemsize, sublane)
            )
        return total


def _build_tile_overhang(
    device_ir: DeviceIR, copies: Mapping[int, int]
) -> TileOverhang | None:
    """Find the tiles that index each tensor kept whole in VMEM, given the
    copies of each the launcher keeps, by storage (``copies``).  None when
    no tile indexes one."""
    env = CompileEnvironment.current()
    fixed, _ = _stream_block_size_bounds()
    noncanonical = device_ir.noncanonical_task_origin_block_ids
    found: dict[
        int, tuple[torch.Tensor, list[set[tuple[int, int | None, int, int, bool]]]]
    ] = {}
    graphs = (graph_info.graph for graph_info in device_ir.graphs)
    for _, fake, subscripts in _memory_accesses(graphs):
        storage = id(fake.untyped_storage())
        if storage not in copies:
            continue
        _, dims = found.setdefault(storage, (fake, [set() for _ in fake.shape]))
        for dim, subscript in enumerate(subscripts):
            info = subscript_tile_info(env, subscript)
            block_id, scale = subscript_index_scale(env, subscript)
            if block_id is None:
                continue
            offset = info.offset if info is not None else 0
            block_size = info.block_size if info is not None else None
            if isinstance(offset, int) and (
                block_size is None or isinstance(block_size, int)
            ):
                dims[dim].add(
                    (block_id, block_size, scale, offset, block_id in noncanonical)
                )
    tensors = tuple(
        (
            tuple(int(size) for size in fake.shape),
            fake.dtype.itemsize,
            env.backend.sublane_tiling(fake.dtype),  # pyrefly: ignore[missing-attribute]
            copies[storage],
            tuple(tuple(sorted(tiles)) for tiles in dims),
        )
        for storage, (fake, dims) in found.items()
        if any(dims)
    )
    if not tensors:
        return None
    return TileOverhang(tensors, fixed)


def plan_megakernel(tile_strategy: TileStrategyDispatch) -> None:
    """Per-config planning of a sequential-roots program, before device codegen."""
    host = HostFunction.current()
    device_fn = DeviceFunction.current()
    config_spec = CompileEnvironment.current().config_spec
    device_fn.pallas_lane_dense = dict(config_spec.pallas_lane_dense(device_fn.config))
    model = config_spec.pallas_stream_model_for(device_fn.config)
    # A grid=(1,) program has no BlockSpec tiling: every tile is an explicit
    # pl.ds slice at its offset.
    for dim_tilings in device_fn.pallas_tensor_dim_tilings.values():
        for dim_tiling in dim_tilings:
            dim_tiling.can_tile = False

    extents = _ref_extents(host.device_ir, device_fn)
    host_scratch_names, scratch_zeroing = _plan_intermediate_scratch(
        host, device_fn, extents
    )
    _scratch_lives_in_vmem(host, device_fn)
    padded_dims = {
        (storage, dim): int(size)
        for storage, (fake, extent) in extents.items()
        for dim, size in enumerate(fake.shape)
        if isinstance(size, int) and extent[dim] > size
    }
    exchanges = _exchange_starts(
        device_fn.codegen.codegen_graphs, host.device_ir.root_ids
    )
    stream, pool_slices = (
        ([], False)
        if model is None
        else _plan_stream(model, host.device_ir, device_fn, tile_strategy, exchanges)
    )
    hbm_resident = plan_hbm_resident(host.device_ir, device_fn)
    plan = MegakernelPlan(
        padded_dims,
        scratch_zeroing,
        host_scratch_names,
        stream,
        hbm_resident,
        early_row_reads=_plan_early_row_reads(host.device_ir, device_fn, hbm_resident),
        index_mirrors=_plan_index_mirrors(device_fn, stream),
        remote_starts={node: root for root, node in exchanges.items()},
        pool_slices=pool_slices,
    )
    if plan.hbm_resident:
        _plan_hbm_copies(plan, host.device_ir, device_fn)
    device_fn.pallas_megakernel = plan


def _ref_extents(
    device_ir: DeviceIR, device_fn: DeviceFunction
) -> dict[int, tuple[torch.Tensor, list[int]]]:
    """Per storage: its tensor and the extent each dim's tiles can reach.

    An argument is padded by the launcher to cover its tiles; scratch is
    allocated with this extent.  A dim of dynamic size keeps extent 0.
    """
    env = CompileEnvironment.current()
    noncanonical = device_ir.noncanonical_task_origin_block_ids
    extents: dict[int, tuple[torch.Tensor, list[int]]] = {}
    graphs = (graph_info.graph for graph_info in device_fn.codegen.codegen_graphs)
    for _, fake, subscripts in _memory_accesses(graphs):
        _, extent = extents.setdefault(
            id(fake.untyped_storage()),
            (fake, [size if isinstance(size, int) else 0 for size in fake.shape]),
        )
        for dim, (size, subscript) in enumerate(
            zip(fake.shape, subscripts, strict=True)
        ):
            if not isinstance(size, int):
                continue
            info = subscript_tile_info(env, subscript)
            block_id, scale = subscript_index_scale(env, subscript)
            if block_id is None:
                continue
            offset = info.offset if info is not None else 0
            block_size = (
                info.block_size
                if info is not None and info.block_size is not None
                else device_fn.resolved_block_size(block_id)
            )
            if not isinstance(block_size, int) or not isinstance(offset, int):
                continue
            # A tile loop that starts at an arbitrary offset can overhang
            # its end by up to one more block.
            reach = -(-size // block_size) * block_size
            if block_id in noncanonical:
                reach += block_size
            extent[dim] = max(extent[dim], scale * reach + max(offset, 0))
    return extents


def _plan_intermediate_scratch(
    host: HostFunction,
    device_fn: DeviceFunction,
    extents: dict[int, tuple[torch.Tensor, list[int]]],
) -> tuple[list[str], list[str]]:
    """Place every intermediate in VMEM scratch sized to the tiles that touch it.

    Unlike launcher arguments, scratch is not padded by the launcher, so a
    ``pl.ds`` tile that overhangs the dim (e.g. 16 rows over M=1) must still
    land inside the allocation.  The padding is zeroed at kernel start, since
    lowering relies on padded positions reading as zero.  Returns the host
    names placed in scratch and the zeroing statements.
    """
    env = CompileEnvironment.current()
    roles = host.device_ir.storage_roles
    uses = _host_tensor_uses(host.device_ir)
    read, written, _ = accessed_storages(device_fn)
    input_storages = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    named: dict[int, str] = {}
    for tensor in host.tensor_to_origin:
        storage = id(tensor.untyped_storage())
        if (
            roles.get(storage) is StorageRole.INTERMEDIATE
            and storage in read & written
            and storage not in input_storages
            and storage in extents
        ):
            # classify_storage_roles admits one name per storage.
            named.setdefault(storage, min(uses[storage].names))

    placed: list[str] = []
    zeroing: list[str] = []
    for storage, name in named.items():
        fake, extent = extents[storage]
        if not all(isinstance(size, int) for size in fake.shape):
            continue
        placed.append(name)
        shape = list(extent)
        aligns = [1] * fake.ndim
        if fake.ndim >= 2:
            aligns[-2] = env.backend.sublane_tiling(fake.dtype)  # pyrefly: ignore[missing-attribute]
            aligns[-1] = 128
            shape[-2] = -(-shape[-2] // aligns[-2]) * aligns[-2]
        scratch = device_fn.register_scratch(tuple(shape), fake.dtype, name_hint=name)
        device_fn.pallas_internal_scratch_storage_names[storage] = scratch
        extent[:] = shape
        dtype = env.backend.dtype_str(fake.dtype)
        for dim, (size, padded) in enumerate(zip(fake.shape, shape, strict=True)):
            if padded == size:
                continue
            # Start the slab on a native tile boundary; nothing has been
            # written yet, so zeroing a few real positions is harmless.
            start = size // aligns[dim] * aligns[dim]
            index = [":"] * fake.ndim
            index[dim] = f"{start}:{padded}"
            slab = [*shape]
            slab[dim] = padded - start
            zeroing.append(
                f"{scratch}[{', '.join(index)}] = jnp.zeros({tuple(slab)!r}, {dtype})"
            )
    return placed, zeroing


def _scratch_lives_in_vmem(host: HostFunction, device_fn: DeviceFunction) -> None:
    """Address every tensor placed in scratch as the VMEM ref it is.

    Tiling keeps a remote-copy destination that one graph only receives into
    in HBM, so it persists until a later graph reads it.  All roots of a
    megakernel share one body, so scratch already persists across them;
    leaving the HBM placement would stage each local read through a DMA.
    """
    scratch = device_fn.pallas_internal_scratch_storage_names
    for tensor in host.tensor_to_origin:
        if id(tensor.untyped_storage()) in scratch:
            device_fn.pallas_memory_space[id(tensor)] = PallasMemorySpace.VMEM


def logical_region_parts(
    device_fn: DeviceFunction, tensor: torch.Tensor, first_dim: int
) -> list[str]:
    """Index parts that trim dims ``first_dim..`` of a ref to ``tensor``.

    A DMA of a whole padded ref (e.g. a decode row in a scratch padded to a
    16-row tile) would move the padding too; the padding is zero on every
    device, so a remote copy only needs the logical region.  Mosaic slices
    tiled dims in whole (8, 128) tiles, so the trim rounds up to them, except
    that a 32-bit ref (one row per sublane) also slices to a single row: a
    one-row f32 exchange moves one row, a one-row bf16 exchange 8.
    """
    plan = device_fn.pallas_megakernel
    if plan is None:
        return []
    storage = id(tensor.untyped_storage())
    tiles = {tensor.ndim - 1: 128, tensor.ndim - 2: 8}
    parts = []
    for dim in range(first_dim, tensor.ndim):
        size = plan.padded_dims.get((storage, dim))
        if size is None:
            parts.append(":")
            continue
        if not (size == 1 and dim == tensor.ndim - 2 and tensor.dtype.itemsize == 4):
            size = -(-size // tiles.get(dim, 1)) * tiles.get(dim, 1)
        parts.append(f"pl.ds(0, {size})")
    while parts and parts[-1] == ":":
        parts.pop()
    return parts


# Weight streaming
# ----------------
#
# Every streamed tile load executed by the program gets a global stream index
# ``g`` in program order.  Root ``r`` starts at ``T_r``; each of its ``N_r``
# tiles consumes ``S_r`` indices, split between its loops in program order
# (loop ``s`` starts ``O_s`` into the tile).  Iteration ``j`` of loop ``s``
# consumes ``c_s`` indices, one per streamed load ``l``:
#
#     g = T_r + p * S_r + O_s + j * c_s + l
#
# where ``p`` is the root's tile index (its ``pid_shared``).  Index ``g``
# lives in ring slot ``g % D``.  The prologue starts ``0 .. D-1``; a consumer
# waits on its slot, computes, then starts ``g + D`` into the slot it just
# freed, decoding which loop, tile and load ``g + D`` belongs to.
#
# The roots of a host loop folded around them (``for i in range(start,
# stop)``) consume ``T_body`` indices per iteration, so ``T_r`` is the root's
# first index in iteration ``start`` and iteration ``i`` adds
# ``(i - start) * T_body``.  A refill into the loop decodes the iteration
# first, which also selects the layer of a stacked weight (``w[i, ...]``).
#
# Each tile shape streams through a ring of its own, with its own indices,
# prologue and refills, so no slot is padded to a larger tile; shapes that
# fill ``_RING_SLOT_FILL`` of a common slot share one.  The loops that read
# different rings mostly run one after another, so a ring has copies in
# flight only during its own loops, and streams faster the more bytes it
# keeps in flight; the VMEM for the rings goes, slot by slot, to the ring
# whose roots it speeds up the most (``_ring_depths``).
# ``pallas_stream_depth`` pins the depth of the ring that streams the most
# bytes.  A ring read only after another ring's reads end starts its prologue after
# that ring's last refill instead of at kernel start, so the DMA queue does
# not drain between them.

# VMEM kept free for everything the plan does not size: Mosaic's internal
# scratch and spills, and, outside the rings' fit (``_resolve_stream``), which
# counts it, the loop-carried scratch.
_STREAM_RESERVE_BYTES = 3 * 1024 * 1024
# The VMEM the ring of stages of the stores of one shape to a tensor kept
# in HBM may take (``hbm_store_rings``).
_STORE_RING_BYTES = 1024 * 1024
# The same, for the choice to keep tensors whole in VMEM in a kernel without
# rings (``_default_hbm_resident``): it is made before any config, so it also
# leaves room for the loop-carried scratch and inner-loop buffers.
_WHOLE_RESERVE_BYTES = 8 * 1024 * 1024
# Bytes in flight the default depths aim for, per ring.
_STREAM_TARGET_BYTES = 16 * 1024 * 1024
# The bytes of ring copies an exchange root starts behind its remote copies
# (``_hold_for_exchanges``): about what streams during their round trip
# (some 4 us on TPU v7); more delays the exchange.
_EXCHANGE_HOLD_BYTES = 12 * 1024 * 1024
# One ring's copies stream at about ``1 - exp(-x / _STREAM_INFLIGHT_BYTES)``
# of the full HBM bandwidth with ``x`` bytes in flight (measured on TPU v7,
# where a copy takes about a microsecond to land).
_STREAM_INFLIGHT_BYTES = 2_400_000
# A tile shares the ring of other tile shapes when every one of them fills at
# least this fraction of the common slot.
_RING_SLOT_FILL = 0.9
# The default tiles: below this many bytes, a fixed per-copy cost keeps a DMA
# from streaming at full bandwidth.
_STREAM_MIN_SLOT_BYTES = 512 * 1024
# The default tiles: narrower tiles leave part of a 256-wide MXU idle.
_STREAM_MIN_TILE_WIDTH = 256
# The default tiles of an arena (``pallas_stream_arena``): on TPU v7, an
# arena of 12 slots of about this many bytes hides all but some 1.5 us of an
# exchange behind its copies, while tiles of 640 KB leave 8 us exposed at any
# depth (an 8-device synthetic decode layer, 80 MB per device).
_ARENA_SLOT_BYTES = 5 << 19
# Whether the default config streams through an arena.  On TPU v7, with the
# tiles above, the Qwen3.8 decode step at 64 layers takes 1928 us instead of
# 2045 on one core, and 2210 instead of 2617 on eight (kernel body).
_ARENA_BY_DEFAULT = True
# The default tiles: roots with more block size combinations than this keep
# their configured sizes.
_STREAM_SEED_LIMIT = 1 << 14
# Above this many stream indices the static check walks only the tiles and
# iterations near each boundary; the schedule is periodic in between.  A full
# walk costs about 4us per index, so the limit bounds it near 0.25s a config.
_STREAM_CHECK_LIMIT = 1 << 16
# Below this size a read-only tensor loaded at root level stays a whole VMEM
# block: one up-front copy beats a DMA per root tile and the ring slots it
# would take.
_ROOT_STREAM_MIN_BYTES = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class IndexRead:
    """A scalar index that the kernel reads from an integer tensor (an index
    vector, such as the ids a router picked): ``fake[indices]``, at indices
    that are constants or expressions in the symbols of the folded host loops
    around the read (see ``_index_read``).

    A copy that runs ahead reads the element again, clamped to ``[0, size)``
    of the dim it indexes: a bad id then reads a valid tile, as the regular
    path's DMA would not.  ``producer`` is the root that writes the tensor
    last before the read, or None for a read-only tensor; a copy must not
    read it before that root ends (see ``RingGate``).
    """

    fake: torch.Tensor
    indices: tuple[sympy.Expr, ...]
    size: int = 0
    producer: int | None = None


# An index vector that the rings' copies read from VMEM is mirrored into SMEM
# when it has at most this many elements, each copied by a statement of its
# own: a copy then reads its element with one scalar load instead of a vector
# load, a rotate and a vector-to-scalar move.
_INDEX_MIRROR_MAX_ELEMENTS = 256


@dataclasses.dataclass(frozen=True)
class IndexMirror:
    """An SMEM copy of an index vector kept in VMEM, which the rings' copies
    read instead of the vector (``_index_read_source``).

    A copy that reads the vector starts after the root that writes it last
    before the copy's load ends (``RingGate``), so the mirror is filled at
    the end of the last tile of each such producer, or at kernel start when
    the vector is read-only (producer ``None``).
    """

    fake: torch.Tensor
    name: str
    # Producer -> its tiles.
    producers: dict[int | None, int]

    def fill(self, device_fn: DeviceFunction) -> list[str]:
        """Statements that copy every element of the vector to the mirror."""
        return [
            f"{self.name}[{', '.join(map(str, index))}] = "
            + _index_element(device_fn, self.fake, [str(part) for part in index])
            for index in itertools.product(*(range(int(n)) for n in self.fake.shape))
        ]

    def refresh(self, device_fn: DeviceFunction, root: int) -> list[ast.stmt]:
        """``fill``, at the end of the last tile of root ``root`` if it is a
        producer of the vector."""
        if root not in self.producers:
            return []
        statements = [statement_from_string(line) for line in self.fill(device_fn)]
        tiles = self.producers[root]
        if tiles == 1:
            return statements
        pid = device_fn.pid.shared_pid_var  # pyrefly: ignore[missing-attribute]
        return _when(device_fn, f"{pid} == {tiles - 1}", statements, "_mirror")


@dataclasses.dataclass(frozen=True)
class StreamLoad:
    """A streamed tile load: its tensor and, per dim, the tile block id.

    ``None`` is a whole dim.  ``sublane`` is the native sublane tile of the
    slots' ``dtype``.  ``scalars`` holds the integer-indexed dims as ``(dim,
    index)``: a constant, an expression in the symbol of the folded host loop
    around the load, or an element of an index vector.  They are not part of
    the tile, so tiles that differ only in them share ring slots.  ``perm``
    is the dim order of a tensor passed lane dense (``pallas_lane_dense``);
    the dims of ``dim_block_ids``, ``scalars`` and the tile follow it.
    """

    fake: torch.Tensor
    dim_block_ids: tuple[int | None, ...]
    sublane: int
    scalars: tuple[tuple[int, sympy.Expr | IndexRead], ...] = ()
    perm: tuple[int, ...] | None = None

    @property
    def producer(self) -> int | None:
        """The root whose writes the load's index vector reads, if any."""
        return next(
            (
                index.producer
                for _, index in self.scalars
                if isinstance(index, IndexRead) and index.producer is not None
            ),
            None,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor's shape in its physical dim order."""
        shape = [int(size) for size in self.fake.shape]
        return tuple(shape if self.perm is None else physical_order(shape, self.perm))

    @property
    def packing(self) -> int:
        """Rows of the tensor per slot element: ``row_packing`` for a row of
        a packed dtype (an integer index on the second-minor dim), which the
        ring copies as the 32-bit words that hold it, else 1."""
        if self.fake.dtype.itemsize == 4 or len(self.shape) - 2 not in dict(
            self.scalars
        ):
            return 1
        return 4 // self.fake.dtype.itemsize

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the ring slots that hold the tile."""
        return torch.uint32 if self.packing > 1 else self.fake.dtype

    def tile_block_ids(self) -> tuple[int | None, ...]:
        """The block id of each dim of the tile (``tile_shape``)."""
        scalar_dims = {dim for dim, _ in self.scalars}
        return tuple(
            block_id
            for dim, block_id in enumerate(self.dim_block_ids)
            if dim not in scalar_dims
        )

    def tile_shape(self, block_sizes: Mapping[int, int]) -> tuple[int, ...]:
        scalar_dims = {dim for dim, _ in self.scalars}
        return tuple(
            size if block_id is None else block_sizes[block_id]
            for dim, (size, block_id) in enumerate(
                zip(self.shape, self.dim_block_ids, strict=True)
            )
            if dim not in scalar_dims
        )


@dataclasses.dataclass(frozen=True)
class StreamSite:
    """A loop whose read-only tile loads are streamed: an inner loop of a
    root, or the root's own tile loop (``block_id`` None, one trip)."""

    graph_id: int
    root: int
    block_id: int | None
    extent: int
    loads: tuple[StreamLoad, ...]

    def trips(self, block_sizes: Mapping[int, int]) -> int:
        if self.block_id is None:
            return 1
        return self.extent // block_sizes[self.block_id]


@dataclasses.dataclass(frozen=True)
class StreamModel:
    """The config-independent part of a megakernel's weight stream."""

    # In program order.
    sites: tuple[StreamSite, ...]
    # Per root: ``(block id, extent)`` of each tile axis.
    root_axes: tuple[tuple[tuple[int, int], ...], ...]
    # VMEM held by the tensors kept whole in VMEM, as the launcher counts
    # it: VMEM arguments twice, intermediate scratch once.
    resident_bytes: int
    # Block ids of a fixed size (hl.grid), with that size.
    fixed_block_sizes: Mapping[int, int]
    # Tuned block ids (``config_spec.block_sizes`` ids), with the smallest
    # size that tiles every streamed dim in whole VMEM tiles.
    min_block_sizes: Mapping[int, int]
    # VMEM the rings are sized against.
    vmem_capacity: int
    # Every inner-loop load of an argument is streamed, so no load is left
    # for ``pallas_load_buffer_count`` to buffer.
    streams_every_inner_load: bool
    # Host loops folded around roots.
    host_loops: tuple[HostLoopInfo, ...] = ()
    # VMEM the tensors that can stay in HBM (``_hbm_resident_tensors``),
    # left out of ``resident_bytes``, take in a config that keeps them whole.
    hbm_resident_bytes: int = 0
    # And their buffers and stages in a config that keeps them in HBM
    # (``_hbm_slot_bytes``), with a buffer per slice load or with the roots
    # sharing them.
    hbm_slot_bytes: int = 0
    hbm_pooled_slot_bytes: int = 0
    # The inner loops outside the rings, which stream tiles through DMA
    # buffers of their own, in a config that keeps the tensors that can stay
    # in HBM there (key True) or not.
    loop_tiles: Mapping[bool, LoopTileModel | None] = dataclasses.field(
        default_factory=dict
    )
    # The values the roots' inner loops carry in VMEM scratch.
    carried: CarriedScratch | None = None
    # The tiles that overhang the tensors kept whole in VMEM.
    overhang: TileOverhang | None = None
    # Tuned block ids whose streamed tiles no smaller size copies in whole
    # VMEM tiles, with the full size of their dims, the one size they take.
    whole_block_sizes: Mapping[int, int] = dataclasses.field(default_factory=dict)

    def loop_of(self, root: int) -> HostLoopInfo | None:
        """The folded host loop whose body holds ``root``."""
        return next((loop for loop in self.host_loops if root in loop.roots), None)

    def trips(self, root: int) -> int:
        """How many times ``root`` runs."""
        loop = self.loop_of(root)
        return 1 if loop is None else loop.trips

    def run_order(self) -> list[int]:
        """Every run of every root, in execution order."""
        order: list[int] = []
        root = 0
        while root < len(self.root_axes):
            loop = self.loop_of(root)
            if loop is None:
                order.append(root)
                root += 1
            else:
                order += [*loop.roots] * loop.trips
                root = loop.end_root
        return order

    def block_ids(self) -> set[int]:
        """Every block id that shapes the stream."""
        return {site.block_id for site in self.sites if site.block_id is not None} | {
            block_id for site in self.sites for block_id, _ in self.root_axes[site.root]
        }

    def block_sizes(self, tuned: Mapping[int, int]) -> dict[int, int]:
        """The size of every block id that shapes the stream, given the size
        of each tuned block id (codegen and the autotuner's viability check
        both resolve sizes here)."""
        return {
            **self.fixed_block_sizes,
            **{block_id: tuned[block_id] for block_id in self.min_block_sizes},
        }

    def resident(self, keep_hbm: bool, pooled: bool = False) -> int:
        """VMEM held by the tensors kept whole in VMEM, in a config that
        keeps the tensors that can stay in HBM there (``keep_hbm``), with
        the roots sharing the slice buffers (``pooled``) or not, or keeps
        them whole."""
        if not keep_hbm:
            return self.resident_bytes + self.hbm_resident_bytes
        return self.resident_bytes + (
            self.hbm_pooled_slot_bytes if pooled else self.hbm_slot_bytes
        )

    def stream_budget(self, keep_hbm: bool) -> int:
        """VMEM left for the rings next to the resident tensors and a reserve."""
        return self.vmem_capacity - _STREAM_RESERVE_BYTES - self.resident(keep_hbm)

    def loop_buffer_bytes(self, tuned: Mapping[int, int], keep_hbm: bool) -> int:
        """VMEM the DMA buffers of the inner loops outside the rings take,
        given the size of each tuned block id, in a config that keeps the
        tensors that can stay in HBM there (``keep_hbm``) or not."""
        loops = self.loop_tiles.get(keep_hbm)
        return 0 if loops is None else loops.buffer_bytes(tuned)

    def carried_bytes(self, tuned: Mapping[int, int]) -> int:
        """VMEM the roots' loop-carried scratch takes, given the size of each
        tuned block id."""
        return 0 if self.carried is None else self.carried.nbytes(tuned)

    def overhang_bytes(self, tuned: Mapping[int, int]) -> int:
        """VMEM the tensors kept whole in VMEM take past their own shape, to
        the extent their tiles reach, given the size of each tuned block id."""
        return 0 if self.overhang is None else self.overhang.nbytes(tuned)

    def config_error(
        self,
        tuned: Mapping[int, int],
        depth: int | None,
        keep_hbm: bool,
        arena: bool = False,
    ) -> str | None:
        """Why a config's rings cannot be laid out or fit, if they cannot."""
        resolved = _resolve_stream(
            self, tuned, depth, self.vmem_capacity, keep_hbm, arena
        )
        return resolved if isinstance(resolved, str) else None

    def rings_shallow(
        self, tuned: Mapping[int, int], keep_hbm: bool, arena: bool = False
    ) -> bool:
        """Whether a config's rings, which fit (``config_error``), share the
        VMEM so that at the default depths the ring that streams the most
        holds fewer than the four iterations' loads ``seed_block_sizes``
        sizes tiles for."""
        layout, depth = max(
            resolve_ring_depths(self, tuned, None, self.vmem_capacity, keep_hbm, arena),
            key=lambda ring: ring[0].traffic,
        )
        return depth < min(4 * layout.loads, layout.total)

    def depth_choices(self) -> tuple[int | None, ...]:
        """Autotuned ``pallas_stream_depth`` values (the depth of the ring
        that streams the most bytes): ``None`` (the default depths), then
        from twice the loads of one iteration up to the largest stream, since
        deeper rings clamp to the same depth."""
        loads = max(len(site.loads) for site in self.sites)
        high = max(16, 2 * loads)
        layouts = _ring_layouts(self, self.block_sizes(self.min_block_sizes))
        if not isinstance(layouts, str):
            # The smallest blocks give the longest stream.
            high = min(high, max(layout.total for layout in layouts))
        return (None, *range(min(2 * loads, high), high + 1))

    def seed_limit(self, keep_hbm: bool) -> int:
        """The largest ``limit`` for ``seed_block_sizes``:
        ``_STREAM_TARGET_BYTES``, or the VMEM left for the rings in a config
        that keeps the tensors that can stay in HBM there (``keep_hbm``) or
        not, if less."""
        return min(_STREAM_TARGET_BYTES, self.stream_budget(keep_hbm))

    def seed_block_sizes(
        self, legal: Mapping[int, Sequence[int]], limit: int, arena: bool = False
    ) -> dict[int, int]:
        """Default sizes of the tuned block ids that shape streamed tiles,
        given the sizes ``legal`` the config spec allows each.

        Per root, among the sizes that divide their dims, prefer tiles that
        (in order) let four iterations' loads of a site fit in ``limit``,
        span the whole minor dim or ``_STREAM_MIN_TILE_WIDTH``, fill
        ``_STREAM_MIN_SLOT_BYTES`` (or come closest), and span the whole
        minor dim, which makes one contiguous copy and lets sites of the same
        width share a ring; then the smallest tiles; then the longest inner
        loop blocks, which keep the accumulators small.

        Small tiles stream faster: during a root, only its own ring has
        copies in flight (the other rings sit full), and a copy takes about
        a microsecond to land, so the bandwidth grows with the bytes in
        flight, ``(depth - loads) * slot``.  The same ring bytes in smaller
        slots leave more of them in flight.

        An arena (``arena``) keeps copies in flight across roots, so its
        tiles come as close to ``_ARENA_SLOT_BYTES`` as they can without
        going over, whatever ``limit``.
        """
        sizes: dict[int, int] = {}
        for root in dict.fromkeys(site.root for site in self.sites):
            sites = [site for site in self.sites if site.root == root]
            extents = dict(self.root_axes[root]) | {
                site.block_id: site.extent
                for site in sites
                if site.block_id is not None
            }
            block_ids = sorted(
                {
                    block_id
                    for site in sites
                    for load in site.loads
                    for block_id in load.dim_block_ids
                    if block_id in self.min_block_sizes
                }
            )
            choices = [
                [
                    size
                    for size in legal[block_id]
                    if extents[block_id] % size == 0
                    and size % self.min_block_sizes[block_id] == 0
                ]
                for block_id in block_ids
            ]
            if not all(choices) or math.prod(map(len, choices)) > _STREAM_SEED_LIMIT:
                continue
            _, best = min(
                (
                    _seed_cost(
                        self.fixed_block_sizes,
                        sites,
                        block_ids,
                        candidate,
                        limit,
                        arena,
                    ),
                    candidate,
                )
                for candidate in itertools.product(*choices)
            )
            sizes.update(zip(block_ids, best, strict=True))
        return sizes


def _seed_cost(
    fixed_block_sizes: Mapping[int, int],
    sites: Sequence[StreamSite],
    block_ids: Sequence[int],
    candidate: Sequence[int],
    stream_bytes: int,
    arena: bool,
) -> tuple[int, ...]:
    """``StreamModel.seed_block_sizes``'s ranking of the sizes ``candidate``
    for ``block_ids``, with ``stream_bytes`` to hold four iterations' loads of
    a site, or tiles for an ``arena``; lower is better."""
    block_sizes = {**fixed_block_sizes, **dict(zip(block_ids, candidate, strict=True))}
    over = narrow = shortfall = partial = total = 0
    for site in sites:
        limit = stream_bytes // (4 * len(site.loads))
        target = _STREAM_MIN_SLOT_BYTES
        if arena:
            limit = target = _ARENA_SLOT_BYTES
        for load in site.loads:
            tile = load.tile_shape(block_sizes)
            minor = load.shape[-1]
            slot = math.prod(tile) * load.dtype.itemsize
            over += max(slot - limit, 0)
            narrow += tile[-1] < min(_STREAM_MIN_TILE_WIDTH, minor)
            shortfall += max(target - slot, 0)
            partial += tile[-1] < minor
            total += slot
    inner = sum(
        block_sizes[site.block_id] for site in sites if site.block_id is not None
    )
    return over, narrow, shortfall, partial, total, -inner


@dataclasses.dataclass(frozen=True)
class RingLoop:
    """A folded host loop's share of one ring's stream.

    Each iteration of the body consumes ``stride`` indices; iteration ``k``
    (counted from the loop's start) of a body root starts ``k * stride`` after
    the root's ``base``.
    """

    loop: HostLoopInfo
    # B: first stream index of the first iteration.
    base: int
    # T_body: stream indices per iteration.
    stride: int

    @property
    def layers(self) -> int:
        return self.loop.trips

    @property
    def end(self) -> int:
        return self.base + self.layers * self.stride


@dataclasses.dataclass(frozen=True)
class RingRoot:
    """A root's share of one ring's stream."""

    root: int
    # T: first stream index of the root.
    base: int
    # N: tiles of the root.
    num_tiles: int
    # S: stream indices per tile.
    stride: int
    # (block id, number of tiles, block size) per tile axis, in the order the
    # root decodes its pid (fastest first).
    axes: tuple[tuple[int, int, int], ...]
    # The variable holding each axis's tile offset in the root's body.
    offset_vars: tuple[str, ...]
    # The folded host loop around the root; ``base`` and ``end`` are then
    # those of its first iteration.
    loop: RingLoop | None = None

    @property
    def end(self) -> int:
        return self.base + self.num_tiles * self.stride

    @property
    def layers(self) -> int:
        return 1 if self.loop is None else self.loop.layers

    @property
    def layer_stride(self) -> int:
        return 0 if self.loop is None else self.loop.stride

    def tile_offsets(self, tile: str | int) -> dict[int, str | int]:
        """Per tile axis, the offset of tile ``tile`` (a Python int or expr)."""
        offsets: dict[int, str | int] = {}
        divisor = 1
        for i, (block_id, num_blocks, block_size) in enumerate(self.axes):
            if num_blocks == 1:
                offsets[block_id] = 0
            elif isinstance(tile, int):
                offsets[block_id] = tile // divisor % num_blocks * block_size
            else:
                index = tile if divisor == 1 else f"({tile} // {divisor})"
                if i < len(self.axes) - 1:
                    index = f"{index} % {num_blocks}"
                offsets[block_id] = f"({index}) * {block_size}"
            divisor *= num_blocks
        return offsets


@dataclasses.dataclass(frozen=True)
class RingSite:
    """One inner loop's share of one ring's stream."""

    graph_id: int
    root: RingRoot
    # None: the root's own tile loop.
    block_id: int | None
    block_size: int
    # NK: loop iterations.
    trips: int
    # O: first stream index of the loop within a tile of its root.
    offset: int
    loads: tuple[StreamLoad, ...]

    @property
    def span(self) -> int:
        return self.trips * len(self.loads)


@dataclasses.dataclass(frozen=True)
class RingGate:
    """The copies of one ring that read an index vector a root writes.

    The loads whose index vector root ``producer`` writes
    (``StreamLoad.producer``) run after it, in the same iteration of its folded
    host loop ``loop`` if it has one.  The stream indices they read start at
    ``first``, plus ``k * loop.stride`` in iteration ``k``.  The copy of such
    an index ``h`` starts at kernel start or after ``h - D`` is read, which is
    before the producer ends (or before its last iteration's run, which the
    copy must see) when ``h < threshold`` (``first + D``, shifted alike).  Those
    copies are deferred to the end of the producer's last tile instead.

    The gate of an exchange root (``_hold_for_exchanges``) holds the loads of
    the roots ``readers`` instead, and defers the copies of ``first ..
    threshold - 1`` to its last remote copy start.  With ``lag`` 1 they run in
    the iteration after the producer's, so the producer's run in iteration
    ``k`` defers those of iteration ``k + 1``.
    """

    producer: int
    first: int
    threshold: int
    loop: RingLoop | None
    # Tiles of the producer.
    tiles: int
    readers: frozenset[int] = frozenset()
    lag: int = 0

    def shift(self, layer: int) -> int:
        """The offset of iteration ``layer``'s indices from the first's."""
        return 0 if self.loop is None else layer * self.loop.stride

    def intervals(self, target: RingRoot) -> list[tuple[int, int]]:
        """The stream indices the gate may defer, in the indices of the
        first iteration of ``target``'s folded loop if it has one."""
        lo, hi = self.first, self.threshold - 1
        if self.loop is not None or target.loop is None:
            return [(lo, hi)]
        loop = target.loop
        lo, hi = max(lo, loop.base), min(hi, loop.end - 1)
        return _first_iteration(loop, lo, hi) if lo <= hi else []

    def binds(self, consumer: RingRoot) -> bool:
        """Whether a refill after a load of root ``consumer`` may be deferred.
        A load after a producer outside the folded loops reads an index at or
        past ``first``, so its refills are at or past ``threshold``; one in the
        producer's loop may refill the next iteration's early indices."""
        return self.loop is not None or consumer.root <= self.producer

    def _target_stride(self, target: RingRoot) -> int:
        # A producer outside the folded loops defers absolute indices.
        return 0 if self.loop is not None else target.layer_stride

    def admits(self, target: RingRoot, index: int, layer: int) -> bool:
        """Whether a refill may start index ``index`` of ``target`` (in the
        indices of its folded loop's first iteration, in iteration
        ``layer``), as ``condition`` emits it."""
        return (
            layer < self.lag
            or index + layer * self._target_stride(target) >= self.threshold
        )

    def condition(self, target: RingRoot, index: str, layer: str | int | None) -> str:
        """Source code of ``admits``."""
        stride = self._target_stride(target)
        if stride == 0 or layer is None or layer == 0:
            condition = f"{index} >= {self.threshold}"
        elif isinstance(layer, int):
            condition = f"{index} >= {self.threshold - layer * stride}"
        else:
            condition = f"{index} + {layer} * {stride} >= {self.threshold}"
        if not self.lag:
            return condition
        assert isinstance(layer, str)
        return f"({condition}) | ({layer} < {self.lag})"


@dataclasses.dataclass(frozen=True)
class Ring:
    """A ``(depth, *slot_shape)`` VMEM ring and its per-slot DMA semaphores;
    ``(depth * rows, cols)`` if the slot's minor dim is ragged, and
    ``(*lead, rows, depth * cols)`` if its rows are (not whole sublane tiles:
    the narrow dim of a tensor passed lane dense).  An arena
    (``raw``) has ``(tiles, sublanes, 128)`` slots of native VMEM tiles, stacked
    as ``(depth * tiles, sublanes, 128)``, and views each as the tile it holds
    (``ring_view.ring_tile``)."""

    buffer: str
    semaphores: str
    slot_shape: tuple[int, ...]
    dtype: torch.dtype
    depth: int
    # G: stream indices.
    total: int
    sites: tuple[RingSite, ...]
    block_sizes: Mapping[int, int]
    gates: tuple[RingGate, ...] = ()
    raw: bool = False

    @property
    def roots(self) -> list[RingRoot]:
        return list(dict.fromkeys(site.root for site in self.sites))

    def gate_of(self, root: int, load: StreamLoad) -> RingGate | None:
        """The gate on the copies of ``load`` of root ``root``, if a root
        writes its index vector or an exchange root holds it."""
        return next(
            (
                gate
                for gate in self.gates
                if (
                    root in gate.readers
                    if gate.readers
                    else gate.producer == load.producer
                )
            ),
            None,
        )

    def gate_at(self, index: int) -> RingGate | None:
        """The gate that defers the copy of stream index ``index``, if any."""
        site, layer, _, _, load = self.instance_at(index)
        gate = self.gate_of(site.root.root, site.loads[load])
        if (
            gate is None
            or layer < gate.lag
            or index >= gate.threshold + gate.shift(layer)
        ):
            return None
        return gate

    def catch_up(self, gate: RingGate) -> list[int]:
        """The stream indices whose copies ``gate`` defers, in the indices of
        its loop's first iteration if it has one."""
        end = self.total if gate.loop is None else gate.loop.base + gate.loop.stride
        shift = gate.shift(gate.lag)
        return [
            index
            for index in range(gate.first, min(gate.threshold, end))
            if self.gate_at(index + shift) is gate
        ]

    @property
    def traffic(self) -> int:
        """Bytes the ring streams, counted in whole slots."""
        return self.total * math.prod(self.slot_shape) * self.dtype.itemsize

    def window(self, slot: str | int, load: StreamLoad) -> str:
        """The part of ring slot ``slot`` that holds a tile of ``load``."""
        tile = load.tile_shape(self.block_sizes)
        if self.raw:
            start = (
                slot if isinstance(slot, int) or slot.isidentifier() else f"({slot})"
            )
            return (
                f"_helion_ring_tile({self.buffer}, "
                f"{_scaled(start, self.slot_shape[0])}, {tuple(tile)})"
            )
        rows, cols = self.slot_shape[-2:]
        start = slot if isinstance(slot, int) or slot.isidentifier() else f"({slot})"
        if cols % 128 != 0:
            # The slots of a ragged minor dim are stacked as rows.
            return f"{self.buffer}.at[pl.ds({_scaled(start, rows)}, {rows}), :]"
        if rows % load.sublane != 0:
            # The slots of ragged rows are stacked along the lanes.
            offset = _scaled(start, cols)
            if not isinstance(offset, int):
                offset = f"pl.multiple_of({offset}, {cols})"
            lead = ":, " * (len(tile) - 1)
            return f"{self.buffer}.at[{lead}pl.ds({offset}, {cols})]"
        if tile == self.slot_shape:
            return f"{self.buffer}.at[{slot}]"
        parts = [
            ":" if size == slot_size else f"pl.ds(0, {size})"
            for size, slot_size in zip(tile, self.slot_shape, strict=True)
        ]
        return f"{self.buffer}.at[{slot}, {', '.join(parts)}]"

    def copy(
        self,
        device_fn: DeviceFunction,
        site: RingSite,
        load: StreamLoad,
        iteration: str | int,
        tile_offsets: Mapping[int, str | int],
        slot: str | int,
        window: str | None = None,
        layer: str | int | None = None,
        wait: bool = False,
    ) -> str:
        """``make_async_copy`` of one tile of ``load`` into ring slot ``slot``.

        ``layer`` is the iteration of the root's folded host loop, counted
        from its start, or ``None`` inside the loop body.  A ``wait`` only
        needs the copy's shape and semaphore, so it does not read the index
        vectors again.
        """
        parts = []
        scalars = dict(load.scalars)
        shape = list(load.shape)
        # (start, size) of the tile's rows (its second-minor dim).
        rows: tuple[str | int, int] = (0, shape[-2])
        for dim, block_id in enumerate(load.dim_block_ids):
            if dim in scalars:
                index = scalars[dim]
                parts.append(
                    "0"
                    if wait and isinstance(index, IndexRead)
                    else _scalar_index(device_fn, site.root, index, layer)
                )
                continue
            # A tile spanning a dim that is not whole VMEM tiles copies like
            # ``:`` (``_stream_load``).
            if block_id is None or (
                self.block_sizes[block_id] == shape[dim]
                and shape[dim] % _vmem_align(load, dim) != 0
            ):
                parts.append(":")
                continue
            block_size = self.block_sizes[block_id]
            offset = (
                _scaled(iteration, block_size)
                if block_id == site.block_id
                else tile_offsets[block_id]
            )
            if not isinstance(offset, int):
                offset = f"pl.multiple_of({offset}, {block_size})"
            parts.append(f"pl.ds({offset}, {block_size})")
            if dim == len(shape) - 2:
                rows = (offset, block_size)
        name = device_fn.tensor_arg(load.fake).name
        if load.packing > 1:
            # A row of a packed dtype copies as the 32-bit words that hold it.
            parts[-2] = row_word(parts[-2], load.packing)
            name = f"{name}.bitcast(jnp.uint32)"
        if len(shape) > 2 and shape[-1] % 128 != 0:
            # Mosaic slices a ref of more than two dims only in whole VMEM
            # tiles of its minor dims, so slice a view of it as rows of its
            # ragged minor dim; ``_stream_load`` made its leading dims scalar.
            terms = [
                _scaled(
                    int(part) if part.isdigit() else f"({part})",
                    math.prod(shape[dim + 1 : -1]),
                )
                for dim, part in enumerate(parts[:-2])
            ]
            start = (
                " + ".join(str(term) for term in (*terms, rows[0]) if term != 0) or "0"
            )
            view = f"{name}.reshape({math.prod(shape[:-1])}, {shape[-1]})"
            source = f"{view}.at[pl.ds({start}, {rows[1]}), :]"
        else:
            source = f"{name}.at[{', '.join(parts)}]"
        return (
            f"pltpu.make_async_copy({source}, {window or self.window(slot, load)}, "
            f"{self.semaphores}.at[{slot}])"
        )

    def instance_at(self, index: int) -> tuple[RingSite, int, int, int, int]:
        """Reference decode of stream index ``index``:
        ``(site, layer, tile, j, load)``."""
        for root in self.roots:
            position, layer = index, 0
            if root.loop is not None and root.loop.base <= index < root.loop.end:
                layer, rest = divmod(index - root.loop.base, root.loop.stride)
                position = root.loop.base + rest
            if root.base <= position < root.end:
                tile, within = divmod(position - root.base, root.stride)
                for site in self.sites:
                    if site.root is root and (
                        site.offset <= within < site.offset + site.span
                    ):
                        iteration, load = divmod(within - site.offset, len(site.loads))
                        return site, layer, tile, iteration, load
        raise AssertionError(f"stream index {index} is outside the ring's stream")

    def read_order(self, index: int) -> tuple[int, int, int, float]:
        """When stream index ``index`` is read, in execution order: its
        root's run, as ``(first root of its folded loop or the root, loop
        iteration, root)``, and the fraction of the run before it."""
        site, layer, tile, iteration, load = self.instance_at(index)
        root = site.root
        within = tile * root.stride + site.offset + iteration * len(site.loads) + load
        start = root.root if root.loop is None else root.loop.loop.first_root
        return start, layer, root.root, within / (root.num_tiles * root.stride)

    def prime(self, device_fn: DeviceFunction) -> list[str]:
        """Start the copies of stream indices ``0 .. depth-1``, except those
        a gate defers."""
        statements = []
        for index in range(self.depth):
            if self.gate_at(index) is not None:
                continue
            site, layer, tile, iteration, load = self.instance_at(index)
            statements.append(
                self.copy(
                    device_fn,
                    site,
                    site.loads[load],
                    iteration,
                    site.root.tile_offsets(tile),
                    index,
                    layer=layer,
                )
                + ".start()"
            )
        return statements


def _scaled(index: str | int, scale: int) -> str | int:
    if isinstance(index, int):
        return index * scale
    return f"{index} * {scale}"


def _scalar_index(
    device_fn: DeviceFunction,
    root: RingRoot,
    index: sympy.Expr | IndexRead,
    layer: str | int | None,
) -> str:
    """Source code of a scalar index of a streamed load.

    In iteration ``layer`` of the root's folded loop, or inside its body (the
    loop symbol is then bound to the induction variable) when ``None``.
    """
    if isinstance(index, IndexRead):
        return _index_read_source(
            device_fn,
            index,
            [_scalar_index(device_fn, root, part, layer) for part in index.indices],
        )
    return _layer_index(
        device_fn, None if root.loop is None else root.loop.loop, index, layer
    )


def _layer_index(
    device_fn: DeviceFunction,
    loop: HostLoopInfo | None,
    index: sympy.Expr,
    layer: str | int | None,
) -> str:
    """Source code of integer ``index``, an expression in the symbol of the
    folded host loop ``loop``: in iteration ``layer`` of the loop, or
    inside its body when ``None``."""
    printer = CompileEnvironment.current().backend.sympy_printer_expr
    if not index.free_symbols:
        return printer(index)
    if layer is None:
        return device_fn.sympy_expr(index)
    assert loop is not None
    value = (
        loop.start + layer
        if isinstance(layer, int)
        else sympy.Add(sympy.Symbol(layer, integer=True), loop.start)
    )
    return printer(index.xreplace({loop.symbol: value}))


def _index_read_source(
    device_fn: DeviceFunction, read: IndexRead, parts: list[str]
) -> str:
    """Source code that reads index vector element ``parts`` of ``read``,
    from its SMEM mirror if it has one (``IndexMirror``), clamped to the
    indexed dim."""
    plan = device_fn.pallas_megakernel
    mirror = None if plan is None else plan.index_mirrors.get(id(read.fake))
    value = (
        _index_element(device_fn, read.fake, parts)
        if mirror is None
        else f"{mirror.name}[{', '.join(parts)}]"
    )
    return f"jnp.clip({value}, 0, {read.size - 1})"


def _index_element(
    device_fn: DeviceFunction, fake: torch.Tensor, parts: list[str]
) -> str:
    """Source code that reads element ``parts`` of index vector ``fake`` as
    the kernel's own scalar load of it does, as an int32."""
    name = device_fn.pallas_tensor_ref_name(fake)
    space = device_fn.pallas_memory_space.get(id(fake))
    value = f"{name}[{', '.join(parts)}]"
    if space is PallasMemorySpace.VMEM and fake.dtype.itemsize <= 4:
        # Mosaic projects no runtime index out of the two minor dims of a
        # VMEM ref; ``emit_vmem_scalar_load`` reads it as the load lowering
        # does (the whole ref is resident, see ``plan_megakernel``).
        scalar_dims = list(range(max(0, fake.ndim - 2), fake.ndim))
        static = {
            dim: int(parts[dim]) % int(fake.shape[dim])
            if parts[dim].isdigit()
            else None
            for dim in scalar_dims
        }
        if fake.dtype.itemsize < 4 or None in static.values():
            value = ast.unparse(
                emit_vmem_scalar_load(
                    fake,
                    name,
                    parts,
                    VmemScalarLoad(
                        scalar_dims=scalar_dims,
                        extents={dim: int(fake.shape[dim]) for dim in scalar_dims},
                        static_indices=static,
                        patterns=tuple(ArbitraryIndexPattern(None) for _ in parts),
                        lanes=int(fake.shape[-1]),
                    ),
                )
            )
    elif space is not PallasMemorySpace.SMEM and space is not PallasMemorySpace.VMEM:
        raise exc.BackendUnsupported(
            backend="pallas",
            detail=f"index vector {name} in {space} memory, read by a streamed "
            "load in a kernel with multiple top-level loops",
        )
    if fake.dtype != torch.int32:
        value = f"lax.convert_element_type({value}, jnp.int32)"
    return value


def _plan_index_mirrors(
    device_fn: DeviceFunction, stream: Sequence[Ring]
) -> dict[int, IndexMirror]:
    """An SMEM mirror (``IndexMirror``) of each index vector in VMEM, of at
    most ``_INDEX_MIRROR_MAX_ELEMENTS`` elements, that the rings' copies
    read."""
    tiles = {gate.producer: gate.tiles for ring in stream for gate in ring.gates}
    reads: dict[int, list[IndexRead]] = {}
    for ring in stream:
        for site in ring.sites:
            for load in site.loads:
                for _, index in load.scalars:
                    if isinstance(index, IndexRead):
                        reads.setdefault(id(index.fake), []).append(index)
    mirrors = {}
    for key, group in reads.items():
        fake = group[0].fake
        if (
            device_fn.pallas_memory_space.get(key) is not PallasMemorySpace.VMEM
            or fake.ndim == 0
            or fake.numel() > _INDEX_MIRROR_MAX_ELEMENTS
        ):
            continue
        name = device_fn.register_scratch(
            tuple(fake.shape), torch.int32, name_hint="index_smem", scratch_type="smem"
        )
        mirrors[key] = IndexMirror(
            fake,
            name,
            {
                read.producer: 1 if read.producer is None else tiles[read.producer]
                for read in group
            },
        )
    return mirrors


def _build_stream_model(
    device_ir: DeviceIR, lane_dense: Mapping[int, tuple[int, ...]]
) -> StreamModel | None:
    """Find the streamed loads of a multi-root Pallas kernel that passes
    the inputs ``lane_dense`` lane dense.

    A read-only tensor is streamed when every load of it is a tile load the
    ring can serve, at most one per loop: in an inner loop that tiles it, or
    in a root's own tile loop when the tensor is too large to keep whole in
    VMEM.  Every other tensor takes the regular path; raises if the tensors
    kept whole in VMEM do not fit.
    """
    env = CompileEnvironment.current()
    graphs = {graph_info.graph_id: graph_info for graph_info in device_ir.graphs}
    uses = _host_tensor_uses(device_ir)
    roles = device_ir.storage_roles
    fixed, minimum = _stream_block_size_bounds()
    rolled = {
        info.original_graph_id for info in device_ir.rolled_reductions if info.used_rdim
    }
    loop_graph_ids = _loop_graph_ids(device_ir)
    owners = owner_roots_by_graph_id(device_ir)

    # Graph id -> (root, loop block id, loop extent, extent per block id) of
    # each loop that can be a stream site, per root in program order: the
    # root's tile loop, then its inner loops.
    site_loops: dict[int, tuple[int, int | None, int, dict[int, int]]] = {}
    for root, root_graph_id in enumerate(device_ir.root_ids):
        axes = device_ir.task_families[root].axes
        extents = {axis.block_id: _static_extent(axis) for axis in axes}
        if (
            root_graph_id in rolled
            or not all(axis.canonical_origin for axis in axes)
            or not all(block_id in fixed or block_id in minimum for block_id in extents)
        ):
            continue
        site_loops[root_graph_id] = (root, None, 1, extents)
        for node in graphs[root_graph_id].graph.nodes:
            if node.op != "call_function" or node.target is not _for_loop:
                continue
            graph_id, begin, end, _ = node.args
            assert isinstance(begin, (list, tuple)) and isinstance(end, (list, tuple))
            block_ids = cast("ForLoopGraphInfo", graphs[graph_id]).block_ids  # pyrefly: ignore[bad-index]
            if (
                len(block_ids) == 1
                and list(begin) == [0]
                and isinstance(end[0], int)
                and block_ids[0] in minimum
            ):
                site_loops[graph_id] = (  # pyrefly: ignore[unsupported-operation]
                    root,
                    block_ids[0],
                    end[0],
                    {**extents, block_ids[0]: end[0]},
                )

    # The extent of each block id a stream site's loads can tile, and those
    # whose block can be the full size of it.
    block_extents = {
        block_id: extent
        for *_, extents in site_loops.values()
        for block_id, extent in extents.items()
    }
    specs = {spec.block_id: spec for spec in env.config_spec.block_sizes}
    whole_blocks = {
        block_id
        for block_id, extent in block_extents.items()
        if fixed.get(block_id) == extent
        or (block_id in specs and specs[block_id].admits(extent))
    }

    streamed: dict[torch.fx.Node, StreamLoad] = {}
    for storage, use in uses.items():
        if roles.get(storage) is not StorageRole.READ_ONLY or not use.loads:
            continue
        loads: dict[torch.fx.Node, StreamLoad] = {}
        for graph_id, node in use.loads:
            site_loop = site_loops.get(graph_id)
            index = node.args[1]
            if (
                site_loop is None
                or not isinstance(index, (list, tuple))
                or any(other.graph is node.graph for other in loads)
            ):
                break
            root, block_id, _, extents = site_loop
            subscripts = tensor_dim_subscripts(index, use.fake.ndim)
            loops = [loop for loop in device_ir.host_loops if root in loop.roots]
            load = _stream_load(
                use.fake,
                subscripts,
                extents,
                fixed,
                minimum,
                whole_blocks,
                loops,
                [
                    _index_read(device_ir, uses, owners, subscript, root, loops)
                    for subscript in subscripts
                ],
                lane_dense.get(storage),
            )
            # A root-level load streams a window that moves with the root's
            # tile or with a folded host loop around the root.
            if load is None or (
                load.dim_block_ids.count(None) == len(load.dim_block_ids)
                and not load.scalars
                if block_id is None
                else block_id not in load.dim_block_ids
            ):
                break
            if (
                block_id is None
                and _padded_bytes(use.fake, lane_dense.get(storage))
                <= _ROOT_STREAM_MIN_BYTES
            ):
                break
            loads[node] = load
        else:
            streamed.update(loads)

    sites = [
        StreamSite(graph_id, root, block_id, extent, loads)
        for graph_id, (root, block_id, extent, _) in site_loops.items()
        if (
            loads := tuple(
                streamed[node]
                for node in graphs[graph_id].graph.nodes
                if node in streamed
            )
        )
    ]
    streamed_storages = {id(load.fake.untyped_storage()) for load in streamed.values()}
    input_storages = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    streams_every_inner_load = all(
        node in streamed
        for storage, use in uses.items()
        if storage in input_storages
        for graph_id, node in use.loads
        if graph_id in loop_graph_ids
    )

    resident = 0
    pinned: list[tuple[int, str, str]] = []
    whole = _whole_vmem_bytes(device_ir, uses, streamed_storages, lane_dense)
    for storage, size in whole.items():
        resident += size
        reason = (
            "read by a load the weight ring cannot stream"
            if roles.get(storage) is StorageRole.READ_ONLY
            else "written by the kernel"
        )
        pinned.append((size, min(uses[storage].names), reason))
    capacity = _vmem_capacity()
    budget = capacity - _STREAM_RESERVE_BYTES
    if resident > budget:
        size, name, reason = max(pinned)
        raise exc.BackendUnsupported(
            backend="pallas",
            detail=f"{resident} bytes of tensors kept whole in VMEM, more than the "
            f"{budget} of {capacity} bytes left after a {_STREAM_RESERVE_BYTES} "
            f"byte reserve; the largest, {name} ({size} bytes), is {reason}, in "
            "a kernel with multiple top-level loops",
        )
    if not sites:
        return None
    root_axes = tuple(
        tuple((axis.block_id, _static_extent(axis)) for axis in family.axes)
        for family in device_ir.task_families
    )
    tuned = {
        block_id
        for site in sites
        for block_id in (
            *(axis for axis, _ in root_axes[site.root]),
            *(() if site.block_id is None else (site.block_id,)),
        )
        if block_id in minimum
    }
    min_block_sizes = {block_id: minimum[block_id] for block_id in tuned}
    for load in streamed.values():
        for dim, block_id in enumerate(load.dim_block_ids):
            if block_id in min_block_sizes:
                min_block_sizes[block_id] = max(
                    min_block_sizes[block_id], _vmem_align(load, dim)
                )
    # A streamed tile that no smaller block copies in whole VMEM tiles spans
    # its whole dim (``_stream_load``).
    whole_block_sizes = {
        block_id: block_extents[block_id]
        for load in streamed.values()
        for block_id in load.dim_block_ids
        if block_id in min_block_sizes
        and block_extents[block_id] % min_block_sizes[block_id] != 0
    }
    min_block_sizes.update(whole_block_sizes)
    return StreamModel(
        tuple(sites),
        root_axes,
        resident,
        {
            block_id: size
            for block_id, size in fixed.items()
            if any(block_id in dict(root_axes[site.root]) for site in sites)
        },
        min_block_sizes,
        capacity,
        streams_every_inner_load,
        tuple(device_ir.host_loops),
        sum(_hbm_resident_whole_bytes(device_ir, lane_dense).values()),
        _hbm_slot_bytes(device_ir, lane_dense),
        _hbm_slot_bytes(device_ir, lane_dense, pooled=True),
        overhang=_build_tile_overhang(
            device_ir,
            {
                storage: 1 if roles.get(storage) is StorageRole.INTERMEDIATE else 2
                for storage in whole
            },
        ),
        whole_block_sizes=whole_block_sizes,
    )


def _loop_graph_ids(device_ir: DeviceIR) -> set[int]:
    """The graph ids of every inner loop."""
    return {
        cast("int", node.args[0])
        for graph_info in device_ir.graphs
        for node in graph_info.graph.nodes
        if node.op == "call_function" and node.target in (_for_loop, _for_loop_step)
    }


def _whole_vmem_bytes(
    device_ir: DeviceIR,
    uses: Mapping[int, _HostTensorUse],
    streamed_storages: set[int],
    lane_dense: Mapping[int, tuple[int, ...]],
) -> dict[int, int]:
    """The VMEM each tensor kept whole in VMEM holds, by storage, as the
    launcher counts it: VMEM arguments twice, intermediate scratch once.

    Tensors used outside the inner loops are whole VMEM blocks, unless the
    weight ring streams them (``streamed_storages``) or they are kept in
    HBM, and intermediates are whole VMEM scratch.  The inputs passed lane
    dense (``lane_dense``) take their physical shape.
    """
    hbm_resident = {
        **_hbm_resident_tensors(device_ir, lane_dense),
        **_row_gather_tables(device_ir),
    }
    loop_graph_ids = _loop_graph_ids(device_ir)
    sizes: dict[int, int] = {}
    for storage, use in uses.items():
        role = device_ir.storage_roles.get(storage)
        if (
            storage in streamed_storages
            or storage in hbm_resident
            or not (role is StorageRole.INTERMEDIATE or use.graph_ids - loop_graph_ids)
        ):
            continue
        sizes[storage] = (1 if role is StorageRole.INTERMEDIATE else 2) * _padded_bytes(
            use.fake, lane_dense.get(storage)
        )
    return sizes


def _vmem_capacity() -> int:
    """VMEM of one TensorCore, as the launcher sizes it."""
    # jax is an optional dependency, imported only by Pallas code paths.
    import jax.experimental.pallas.tpu as pltpu

    return _get_vmem_limit_bytes(
        pltpu, CompileEnvironment.current().settings.pallas_interpret
    )


def _stream_block_size_bounds() -> tuple[dict[int, int], dict[int, int]]:
    """The block ids a streamed tile can use: those of a fixed size
    (hl.grid) with their size, and the tuned ones with their minimum."""
    env = CompileEnvironment.current()
    spec_minimum = {
        spec.block_id: spec.min_size for spec in env.config_spec.block_sizes
    }
    fixed: dict[int, int] = {}
    minimum: dict[int, int] = {}
    for info in env.block_sizes:
        source = info.block_size_source
        if isinstance(source, FixedBlockSizeSource) and isinstance(source.value, int):
            fixed[info.block_id] = source.value
        elif (
            isinstance(source, LoopSpecBlockSizeSource)
            and info.block_id in spec_minimum
        ):
            minimum[info.block_id] = spec_minimum[info.block_id]
    return fixed, minimum


def _vmem_align(load: StreamLoad, dim: int) -> int:
    """Granule of dim ``dim`` in a whole number of (sublane, 128) VMEM tiles."""
    ndim = len(load.dim_block_ids)
    if dim == ndim - 1:
        return 128
    return load.sublane if dim == ndim - 2 else 1


def _stream_load(
    fake: torch.Tensor,
    subscripts: list[object],
    extents: Mapping[int, int],
    fixed: Mapping[int, int],
    minimum: Mapping[int, int],
    whole: Collection[int],
    loops: Sequence[HostLoopInfo],
    index_reads: Sequence[IndexRead | None],
    perm: tuple[int, ...] | None = None,
) -> StreamLoad | None:
    """The ring load of one tile load, or None when the ring cannot serve it.

    Every dim must be whole, a plain tile of a block in ``extents``, or (a
    leading dim) an integer in the symbols of the folded host loops ``loops``
    or the index vector element ``index_reads`` has for it, and every block
    size the config can pick must copy whole VMEM tiles.  That
    leaves out a tile over the rows of a tensor smaller than one sublane tile
    (a decode activation), which the regular path pads instead.  The whole
    minor dim may be ragged (not a whole number of 128 lanes): the DMA copies
    it into a slot padded to whole tiles, through a view of the tensor as rows
    of that dim when it has more than two dims (see ``Ring.copy``), so the
    leading dims must then be scalar and the rows whole sublane tiles.  The
    dims of a tensor passed lane dense are taken in its dim order ``perm``,
    and its whole rows (its narrow dim) may be ragged too.  Its second-minor
    dim may also be an integer, when ``row_packing`` reads rows of its dtype
    and its minor dim is whole VMEM tiles: a tile of a packed dtype then
    copies as 32-bit words (``StreamLoad.packing``).

    A tile over a dim that no block smaller than the dim copies in whole VMEM
    tiles (576 lanes, or 192 rows of a block that must also be a multiple of
    128) streams when its block can be the full size of the dim (``whole``):
    the config must then pick that size (``StreamModel.whole_block_sizes``),
    so the tile is the whole dim, like a ``:`` one.
    """
    if fake.ndim < 2 or not all(isinstance(size, int) for size in fake.shape):
        return None
    env = CompileEnvironment.current()
    load = StreamLoad(
        fake,
        (None,) * fake.ndim,
        env.backend.sublane_tiling(fake.dtype),  # pyrefly: ignore[missing-attribute]
        perm=perm,
    )
    shape = load.shape
    if perm is not None:
        subscripts = physical_order(subscripts, perm)
        index_reads = physical_order(index_reads, perm)
    # The dims that may be integers.
    scalar_dims = range(fake.ndim - 2)
    if (
        perm is not None
        and row_packing(fake.dtype, shape[-2]) is not None
        and shape[-1] % 128 == 0
    ):
        scalar_dims = range(fake.ndim - 1)
    dim_block_ids: list[int | None] = []
    scalars: list[tuple[int, sympy.Expr | IndexRead]] = []
    for dim, (size, subscript, index_read) in enumerate(
        zip(shape, subscripts, index_reads, strict=True)
    ):
        align = _vmem_align(load, dim)
        if isinstance(subscript, slice) and subscript == slice(None):
            # The rows of a tensor passed lane dense are its narrow dim, which
            # the slot pads to whole tiles like a ragged minor dim.
            ragged = dim == fake.ndim - 1 or (perm is not None and dim == fake.ndim - 2)
            if size % align != 0 and not ragged:
                return None
            dim_block_ids.append(None)
            continue
        if index_read is not None:
            if dim not in scalar_dims:
                return None
            scalars.append((dim, dataclasses.replace(index_read, size=size)))
            dim_block_ids.append(None)
            continue
        scalar = _scalar_subscript(subscript, loops)
        if scalar is not None:
            # Provably wrong in some iteration: an error, not a fallback.
            if not _scalar_in_bounds(scalar, loops, size):
                raise exc.BackendUnsupported(
                    backend="pallas",
                    detail=f"a load indexes dim {dim} of a [{', '.join(map(str, shape))}] "
                    f"tensor with {scalar}, out of bounds in some folded loop iteration",
                )
            if dim not in scalar_dims:
                return None
            scalars.append((dim, scalar))
            dim_block_ids.append(None)
            continue
        info = subscript_tile_info(env, subscript)
        if (
            info is None
            or not isinstance(info.offset, int)
            or info.offset != 0
            or info.block_size is not None
            or info.block_id not in extents
        ):
            return None
        extent = extents[info.block_id]
        smallest = fixed.get(info.block_id)
        if smallest is None:
            smallest = max(minimum[info.block_id], align)
        if extent > size:
            return None
        if (smallest % align != 0 or extent % smallest != 0) and (
            extent != size
            or info.block_id not in whole
            or (size % align != 0 and dim != fake.ndim - 1)
        ):
            return None
        dim_block_ids.append(info.block_id)
    minor = dim_block_ids[-1]
    if len(scalars) > fake.ndim - 2:
        # The tile is at least 2-D.
        return None
    if (
        fake.ndim > 2
        and shape[-1] % 128 != 0
        and (
            len(scalars) < fake.ndim - 2
            or shape[-2] % load.sublane != 0
            or (minor is not None and extents[minor] != shape[-1])
        )
    ):
        return None
    # One root's writes gate the copies (``RingGate``).
    producers = {
        index.producer
        for _, index in scalars
        if isinstance(index, IndexRead) and index.producer is not None
    }
    if len(producers) > 1:
        return None
    load = dataclasses.replace(
        load, dim_block_ids=tuple(dim_block_ids), scalars=tuple(scalars)
    )
    return dataclasses.replace(
        load,
        sublane=env.backend.sublane_tiling(load.dtype),  # pyrefly: ignore[missing-attribute]
    )


def _padded_bytes(fake: torch.Tensor, perm: Sequence[int] | None = None) -> int:
    """Bytes of ``fake``, laid out in dim order ``perm`` if given, padded to
    whole (sublane, 128) VMEM tiles."""
    shape = [int(size) for size in fake.shape]
    if perm is not None:
        shape = physical_order(shape, perm)
    return _padded_shape_bytes(shape, fake.dtype)


def _padded_shape_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    """Bytes of a ``shape`` buffer of ``dtype`` padded to whole (sublane,
    128) VMEM tiles."""
    sublane = CompileEnvironment.current().backend.sublane_tiling(dtype)  # pyrefly: ignore[missing-attribute]
    return _tiled_bytes(shape, dtype.itemsize, sublane)


def _tiled_bytes(shape: Sequence[int], itemsize: int, sublane: int) -> int:
    """Bytes of a ``shape`` buffer of ``itemsize`` elements padded to whole
    (``sublane``, 128) VMEM tiles."""
    shape = list(shape)
    if shape:
        shape[-1] = -(-shape[-1] // 128) * 128
    if len(shape) >= 2:
        shape[-2] = -(-shape[-2] // sublane) * sublane
    return math.prod(shape) * itemsize


def _scalar_subscript(
    subscript: object, loops: Sequence[HostLoopInfo]
) -> sympy.Expr | None:
    """The integer a subscript selects: a constant, or an expression in the
    symbols of the folded host loops ``loops``; ``None`` for anything else."""
    value = (
        subscript.meta.get("val") if isinstance(subscript, torch.fx.Node) else subscript
    )
    if isinstance(value, int) and not isinstance(value, bool):
        return sympy.Integer(value)
    if isinstance(value, torch.SymInt):
        expr = value._sympy_()
        if expr.free_symbols <= {loop.symbol for loop in loops}:
            return expr
    return None


def _scalar_in_bounds(
    index: sympy.Expr, loops: Sequence[HostLoopInfo], size: int
) -> bool:
    """Whether ``index`` stays in ``[0, size)`` over every loop iteration."""
    values = [index]
    for loop in loops:
        values = [
            value.xreplace({loop.symbol: sympy.Integer(i)})
            for value in values
            for i in range(loop.start, loop.stop)
        ]
    return all(0 <= int(value) < size for value in values)


def _index_read(
    device_ir: DeviceIR,
    uses: Mapping[int, _HostTensorUse],
    owners: Sequence[tuple[int, ...]],
    subscript: object,
    root: int,
    loops: Sequence[HostLoopInfo],
) -> IndexRead | None:
    """The index vector element a scalar subscript of a load in root ``root``
    reads, or None when it is anything else.

    The subscript must be the value of a scalar load from an integer tensor,
    followed through renames and into nested loops (a value passed into a
    loop, not one the loop carries), at in-bounds indices that
    ``_scalar_subscript`` accepts.  The tensor is read-only, or written only
    by stores in earlier roots, the last of which (the producer) is outside
    the folded host loops or in the one around root ``root``: a copy can then
    read the element again at any time after the producer ends.
    """
    # The graph classes import this module.
    from ..device_ir import HelperFunctionGraphInfo
    from ..device_ir import NodeArgsGraphInfo

    infos = {id(info.graph): info for info in device_ir.graphs}
    node = subscript
    while isinstance(node, torch.fx.Node):
        if node.op == "placeholder":
            info = infos.get(id(node.graph))
            if not isinstance(info, NodeArgsGraphInfo) or isinstance(
                info, HelperFunctionGraphInfo
            ):
                return None
            node = info.placeholder_to_outer_arg(node)
            if any(user.target is _phi for user in node.users):
                return None
        elif node.op == "call_function" and node.target is _new_var:
            node = node.args[0]
        else:
            break
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not memory_ops.load
    ):
        return None
    host, index, extra_mask = node.args[:3]
    assert isinstance(host, torch.fx.Node)
    fake = host.meta["val"]
    if (
        host.target is not _host_tensor
        or extra_mask is not None
        or fake.dtype.is_floating_point
        or fake.dtype.is_complex
        or fake.dtype == torch.bool
        or not isinstance(index, (list, tuple))
        or len(index) != fake.ndim
        or not all(isinstance(size, int) for size in fake.shape)
    ):
        return None
    indices = [_scalar_subscript(part, loops) for part in index]
    if not all(
        part is not None and _scalar_in_bounds(part, loops, size)
        for part, size in zip(indices, fake.shape, strict=True)
    ):
        return None
    storage = id(fake.untyped_storage())
    producer = None
    if device_ir.storage_roles.get(storage) is not StorageRole.READ_ONLY:
        use = uses[storage]
        writers = [owners[graph_id] for graph_id, _ in use.stores]
        if (
            use.other
            or not writers
            or not all(len(roots) == 1 and roots[0] < root for roots in writers)
        ):
            return None
        producer = max(roots[0] for roots in writers)
        loop = next(
            (loop for loop in device_ir.host_loops if producer in loop.roots), None
        )
        if loop is not None and root not in loop.roots:
            return None
    return IndexRead(
        fake,
        tuple(cast("sympy.Expr", part) for part in indices),
        producer=producer,
    )


@dataclasses.dataclass(frozen=True)
class _RingLayout:
    """The config-dependent shape of one ring, before codegen names exist."""

    dtype: torch.dtype
    slot_shape: tuple[int, ...]
    total: int
    sites: tuple[tuple[StreamSite, tuple[StreamLoad, ...]], ...]
    # An arena of native VMEM tiles (``Ring.raw``).
    raw: bool = False

    @property
    def sublane(self) -> int:
        """Rows of a VMEM tile of the ring's dtype."""
        return self.sites[0][1][0].sublane

    @property
    def slot_bytes(self) -> int:
        """VMEM of one slot, padded to whole (sublane, 128) VMEM tiles."""
        *lead, rows, cols = self.slot_shape
        sublane = self.sublane
        rows, cols = -(-rows // sublane) * sublane, -(-cols // 128) * 128
        return math.prod((*lead, rows, cols)) * self.dtype.itemsize

    @property
    def traffic(self) -> int:
        """Bytes the ring streams, counted in whole slots."""
        return self.total * self.slot_bytes

    @property
    def loads(self) -> int:
        """The most streamed tiles one loop iteration reads from the ring."""
        return max(len(loads) for _, loads in self.sites)


def _static_extent(axis: TaskAxis) -> int:
    """Extent of a root axis; ``enter_sequential_roots_mode`` admits only ints."""
    assert isinstance(axis.extent, sympy.Expr)
    return int(axis.extent)


def _num_tiles(model: StreamModel, root: int, block_sizes: Mapping[int, int]) -> int:
    return math.prod(
        -(-extent // block_sizes[block_id])
        for block_id, extent in model.root_axes[root]
    )


def _ring_layouts(
    model: StreamModel, block_sizes: Mapping[int, int], arena: bool = False
) -> list[_RingLayout] | str:
    """Group the streamed loads into one ring per tile shape, dtype and rank,
    with the root-level loads in rings of their own; a str says why a config
    cannot stream.

    With ``arena`` (``pallas_stream_arena``), the 2-D tiles of whole VMEM
    tiles share one arena per dtype instead (and the root-level ones
    another), in slots of the largest tile's native tiles.  The loads whose
    index a root of the kernel writes (``StreamLoad.producer``) take an arena
    of their own: an arena streams in program order, so a slot waiting for
    its producer would hold back every later copy behind it, and the copies
    after the producer, the next layer's weights included, could not stream
    while it runs.
    """
    groups: dict[
        tuple[torch.dtype, int, bool, tuple[int, ...] | None, bool, bool],
        dict[tuple[int, ...], list[StreamLoad]],
    ] = {}
    for site in model.sites:
        if site.block_id is not None:
            block_size = block_sizes[site.block_id]
            if site.extent % block_size != 0:
                return (
                    f"streamed inner loop of extent {site.extent} needs a block "
                    f"size that divides it, got {block_size}"
                )
        extents = dict(model.root_axes[site.root])
        for load in site.loads:
            for block_id in load.dim_block_ids:
                if (
                    block_id is not None
                    and block_id != site.block_id
                    and extents[block_id] % block_sizes[block_id] != 0
                ):
                    return (
                        f"streamed tiles need block sizes that divide their dims; "
                        f"block size {block_sizes[block_id]} does not divide "
                        f"{extents[block_id]}"
                    )
            tile = load.tile_shape(block_sizes)
            # Only a tile over the whole minor dim of a tensor, or the whole
            # rows of one passed lane dense (its narrow dim), may be ragged.
            ragged_cols = tile[-1] % 128 != 0 and tile[-1] == load.shape[-1]
            ragged_rows = (
                load.perm is not None
                and load.tile_block_ids()[-2] is None
                and tile[-2] % load.sublane != 0
            )
            ragged = ragged_cols or ragged_rows
            if (tile[-1] % 128 != 0 and not ragged_cols) or (
                tile[-2] % load.sublane != 0 and not ragged_rows
            ):
                return (
                    f"streamed tile {tile} of {load.dtype} is not a whole number "
                    f"of ({load.sublane}, 128) VMEM tiles"
                )
            # A ragged tile shares a slot only with its shape.
            raw = arena and len(tile) == 2 and not ragged
            key = (
                load.dtype,
                len(tile),
                site.block_id is None,
                tile if ragged else None,
                raw,
                raw and load.producer is not None,
            )
            groups.setdefault(key, {}).setdefault(tile, []).append(load)

    layouts = []
    for (dtype, _, _, _, raw, _), tiles in groups.items():
        slots = _ring_slots(list(tiles))
        if raw:
            sublane = next(iter(tiles.values()))[0].sublane
            slots = [
                (
                    (
                        max(rows // sublane * cols // 128 for rows, cols in tiles),
                        sublane,
                        128,
                    ),
                    list(tiles),
                )
            ]
        for slot_shape, shapes in slots:
            members = {id(load) for shape in shapes for load in tiles[shape]}
            sites = tuple(
                (site, ring_loads)
                for site in model.sites
                if (ring_loads := tuple(ld for ld in site.loads if id(ld) in members))
            )
            total = sum(
                model.trips(site.root)
                * _num_tiles(model, site.root, block_sizes)
                * site.trips(block_sizes)
                * len(ring_loads)
                for site, ring_loads in sites
            )
            layouts.append(_RingLayout(dtype, slot_shape, total, sites, raw))
    return layouts


def _ring_slots(
    shapes: list[tuple[int, ...]],
) -> list[tuple[tuple[int, ...], list[tuple[int, ...]]]]:
    """Cluster tile shapes, in first-use order, into ring slots.

    Largest first, a shape joins the first slot that, grown to hold it, every
    one of its shapes still fills to ``_RING_SLOT_FILL``.  Returns ``(slot
    shape, tile shapes)`` per slot, in first-use order.
    """
    slots: list[tuple[tuple[int, ...], list[tuple[int, ...]]]] = []
    for shape in sorted(shapes, key=math.prod, reverse=True):
        for index, (slot, members) in enumerate(slots):
            grown = tuple(map(max, slot, shape))
            if all(
                math.prod(member) >= _RING_SLOT_FILL * math.prod(grown)
                for member in (*members, shape)
            ):
                slots[index] = (grown, [*members, shape])
                break
        else:
            slots.append((shape, [shape]))
    return sorted(slots, key=lambda slot: min(map(shapes.index, slot[1])))


def _ring_runs(
    model: StreamModel, layout: _RingLayout, block_sizes: Mapping[int, int]
) -> collections.Counter[int]:
    """How many runs of consecutive roots that read a ring read each number
    of its tiles.  Roots that stream no ring do not end a run."""
    reads: dict[int, int] = {}
    for site, loads in layout.sites:
        reads[site.root] = reads.get(site.root, 0) + (
            _num_tiles(model, site.root, block_sizes)
            * site.trips(block_sizes)
            * len(loads)
        )
    streaming = {site.root for site in model.sites}
    sizes: collections.Counter[int] = collections.Counter()
    run = 0
    for root in model.run_order():
        if root in reads:
            run += reads[root]
        elif root in streaming and run:
            sizes[run] += 1
            run = 0
    if run:
        sizes[run] += 1
    return sizes


def _ring_floor(layout: _RingLayout, runs: collections.Counter[int]) -> int:
    """The depth of a ring before ``_ring_depths`` deepens it: the loads of
    two iterations, so a root can read one while the next lands.

    A ring of tiles narrower than the 128 lanes holds mostly padding, and
    its few real bytes land in the time other roots take.  So it gets no
    more than its longest run (``_ring_runs``), which then lands whole
    between runs: a narrow tile read once per root keeps one slot, not two.
    Wide rings keep both iterations, since their tiles in flight are what
    keeps the DMA engine busy when every run is short."""
    floor = min(layout.total, 2 * layout.loads)
    if layout.slot_shape[-1] < 128:
        return min(floor, max(runs))
    return floor


def _ring_depths(
    layouts: Sequence[_RingLayout],
    runs: Sequence[collections.Counter[int]],
    depths: Sequence[int],
    pinned: int | None,
    budget: int,
) -> list[int]:
    """Deepen the rings from ``depths`` a slot at a time while they fit in
    ``budget`` bytes, each slot to the ring (other than ``pinned``) where it
    saves the most stream time per byte, up to about
    ``_STREAM_TARGET_BYTES`` or four iterations' loads per ring.

    A ring has copies in flight only while roots read it: while roots
    stream other rings, it sits full.  So each run of consecutive roots that
    read a ring (``runs``, per ring) finds its first ``depth`` slots landed,
    and the rest of the run's tiles stream at the bandwidth ``depth - loads``
    slots in flight reach (``_STREAM_INFLIGHT_BYTES``).  An arena, read by
    every root, may take all the VMEM.
    """

    def stall(index: int, depth: int) -> float:
        """Stream time of ring ``index`` at ``depth`` beyond the time at full
        bandwidth, in bytes at full bandwidth."""
        layout = layouts[index]
        late = layout.slot_bytes * sum(
            count * max(size - depth, 0) for size, count in runs[index].items()
        )
        in_flight = (depth - layout.loads) * layout.slot_bytes
        return late / math.expm1(in_flight / _STREAM_INFLIGHT_BYTES) if late else 0.0

    caps = [
        layout.total
        if layout.raw
        else min(
            layout.total,
            max(4 * layout.loads, -(-_STREAM_TARGET_BYTES // layout.slot_bytes)),
        )
        for layout in layouts
    ]
    depths = list(depths)
    used = sum(d * layout.slot_bytes for d, layout in zip(depths, layouts, strict=True))
    while True:
        best = None
        for index, layout in enumerate(layouts):
            if (
                index == pinned
                or depths[index] >= caps[index]
                or used + layout.slot_bytes > budget
            ):
                continue
            gain = stall(index, depths[index]) - stall(index, depths[index] + 1)
            if gain > 0 and (best is None or gain / layout.slot_bytes > best[0]):
                best = (gain / layout.slot_bytes, index)
        if best is None:
            return depths
        depths[best[1]] += 1
        used += layouts[best[1]].slot_bytes


def resolve_ring_depths(
    model: StreamModel,
    tuned: Mapping[int, int],
    requested: int | None,
    vmem_capacity: int,
    keep_hbm: bool,
    arena: bool = False,
) -> list[tuple[_RingLayout, int]]:
    """The rings of a config, given the size of each tuned block id, each
    with its depth.

    Every ring holds the loads of two loop iterations, or a narrow ring
    those of its longest run of reads if fewer (``_ring_floor``).
    ``_ring_depths`` deepens them in the VMEM left next to the resident
    tensors (``StreamModel.resident``), the inner loops' own DMA buffers
    (``StreamModel.loop_buffer_bytes``), the loop-carried scratch
    (``StreamModel.carried_bytes``) and a reserve.  ``requested``
    (``pallas_stream_depth``) pins the depth of the ring that streams the
    most bytes, which must cover the loads of one iteration.  ``arena`` is
    ``pallas_stream_arena`` (``_ring_layouts``).  Raises ``InvalidConfig`` if
    the rings do not fit.
    """
    resolved = _resolve_stream(model, tuned, requested, vmem_capacity, keep_hbm, arena)
    if isinstance(resolved, str):
        raise exc.InvalidConfig(resolved)
    return resolved


def slices_pooled(
    model: StreamModel,
    tuned: Mapping[int, int],
    requested: int | None,
    vmem_capacity: int,
    keep_hbm: bool,
    arena: bool = False,
) -> bool:
    """Whether the roots of a config share the buffers of the slice loads
    of the tensors kept in HBM (``_plan_hbm_copies``): when that lets the
    rings fit, or deepens them.  A shared buffer's copy cannot start before
    the previous root that reads it is done, which can leave it no ring copy
    to start ahead of, so with VMEM to spare each load keeps its own."""
    if not keep_hbm or model.hbm_pooled_slot_bytes >= model.hbm_slot_bytes:
        return False
    own = _resolve_rings(model, tuned, requested, vmem_capacity, keep_hbm, arena)
    if isinstance(own, str):
        return True
    shared = _resolve_rings(
        model, tuned, requested, vmem_capacity, keep_hbm, arena, pooled=True
    )
    assert not isinstance(shared, str)
    return [depth for _, depth in shared] != [depth for _, depth in own]


def _resolve_stream(
    model: StreamModel,
    tuned: Mapping[int, int],
    requested: int | None,
    vmem_capacity: int,
    keep_hbm: bool,
    arena: bool = False,
) -> list[tuple[_RingLayout, int]] | str:
    """``resolve_ring_depths``, with a str saying why a config is invalid."""
    pooled = slices_pooled(model, tuned, requested, vmem_capacity, keep_hbm, arena)
    return _resolve_rings(
        model, tuned, requested, vmem_capacity, keep_hbm, arena, pooled
    )


def _resolve_rings(
    model: StreamModel,
    tuned: Mapping[int, int],
    requested: int | None,
    vmem_capacity: int,
    keep_hbm: bool,
    arena: bool = False,
    pooled: bool = False,
) -> list[tuple[_RingLayout, int]] | str:
    """``_resolve_stream``, with the roots sharing the slice buffers
    (``pooled``) or not."""
    block_sizes = model.block_sizes(tuned)
    resolved_layouts = _ring_layouts(model, block_sizes, arena)
    if isinstance(resolved_layouts, str):
        return resolved_layouts
    layouts = resolved_layouts
    resident = (
        model.resident(keep_hbm, pooled)
        + model.overhang_bytes(tuned)
        + model.loop_buffer_bytes(tuned, keep_hbm)
        + model.carried_bytes(tuned)
    )
    budget = vmem_capacity - _STREAM_RESERVE_BYTES - resident
    runs = [_ring_runs(model, layout, block_sizes) for layout in layouts]
    depths = list(itertools.starmap(_ring_floor, zip(layouts, runs, strict=True)))
    pinned = None
    if requested is not None:
        pinned = max(range(len(layouts)), key=lambda index: layouts[index].traffic)
        primary = layouts[pinned]
        if requested < primary.loads:
            return (
                f"pallas_stream_depth={requested} is smaller than the "
                f"{primary.loads} streamed tiles one inner-loop iteration reads"
            )
        depths[pinned] = min(requested, primary.total)
    need = sum(d * layout.slot_bytes for d, layout in zip(depths, layouts, strict=True))
    if need > budget:
        return (
            f"weight rings of depths {depths} need {need} bytes of VMEM, but "
            f"only {budget} of {vmem_capacity} are left after {resident} "
            f"resident (with tile overhang), inner-loop buffer and "
            f"loop-carried bytes and a "
            f"{_STREAM_RESERVE_BYTES} byte reserve"
        )
    depths = _ring_depths(layouts, runs, depths, pinned, budget)
    return list(zip(layouts, depths, strict=True))


def _plan_stream(
    model: StreamModel,
    device_ir: DeviceIR,
    device_fn: DeviceFunction,
    tile_strategy: TileStrategyDispatch,
    exchanges: Mapping[int, torch.fx.Node],
) -> tuple[list[Ring], bool]:
    """Size and register the weight rings of one config; ``exchanges``
    holds the exchange roots (``_exchange_starts``).  Also whether the roots
    share the slice buffers (``slices_pooled``)."""
    env = CompileEnvironment.current()
    tuned: dict[int, int] = {}
    for spec in env.config_spec.block_sizes:
        block_size = device_fn.resolved_block_size(spec.block_id)
        assert isinstance(block_size, int) or spec.block_id not in model.min_block_sizes
        if isinstance(block_size, int):
            tuned[spec.block_id] = block_size
    block_sizes = model.block_sizes(tuned)
    requested = device_fn.config.get("pallas_stream_depth")
    assert requested is None or isinstance(requested, int)
    keep_hbm = env.config_spec.pallas_keeps_hbm_resident(device_fn.config)
    arena = bool(device_fn.config.get("pallas_stream_arena"))
    rings = []
    pooled = slices_pooled(
        model, tuned, requested, model.vmem_capacity, keep_hbm, arena
    )
    for layout, depth in resolve_ring_depths(
        model, tuned, requested, model.vmem_capacity, keep_hbm, arena
    ):
        by_root: dict[int, list[tuple[StreamSite, tuple[StreamLoad, ...]]]] = {}
        for site, loads in layout.sites:
            by_root.setdefault(site.root, []).append((site, loads))
        sites: list[RingSite] = []
        base = 0
        for root, root_sites in by_root.items():
            strategy = tile_strategy.block_id_to_strategy[
                tuple(device_ir.grid_block_ids[root])
            ]
            order = [strategy.block_ids[i] for i in strategy.loop_order]  # pyrefly: ignore[missing-attribute]
            extents = dict(model.root_axes[root])
            assert set(order) == set(extents)
            axes = tuple(
                (
                    block_id,
                    -(-extents[block_id] // block_sizes[block_id]),
                    block_sizes[block_id],
                )
                for block_id in order
            )
            stride = sum(
                site.trips(block_sizes) * len(loads) for site, loads in root_sites
            )
            ring_root = RingRoot(
                root,
                base,
                math.prod(axis[1] for axis in axes),
                stride,
                axes,
                tuple(strategy.offset_var(block_id) for block_id in order),
            )
            offset = 0
            for site, loads in root_sites:
                sites.append(
                    RingSite(
                        site.graph_id,
                        ring_root,
                        site.block_id,
                        1 if site.block_id is None else block_sizes[site.block_id],
                        site.trips(block_sizes),
                        offset,
                        loads,
                    )
                )
                offset += sites[-1].span
            base = ring_root.end
        sites = _fold_ring_sites(sites, model)
        gates = tuple(
            _ring_gate(model, device_fn, sites, producer, depth)
            for producer in sorted(
                {
                    load.producer
                    for site in sites
                    for load in site.loads
                    if load.producer is not None
                }
            )
        )
        rows, cols = layout.slot_shape[-2:]
        # Mosaic slices a VMEM ref of more than two dims only in whole VMEM
        # tiles of its minor dims, so a ring of a ragged minor dim (whose
        # slots hold one 2-D tile shape) stacks its slots as rows, and one of
        # ragged rows (one tile shape too) stacks them along the lanes.
        shape = (depth, *layout.slot_shape)
        if cols % 128 != 0:
            shape = (depth * rows, cols)
        elif rows % layout.sublane != 0:
            shape = (*layout.slot_shape[:-1], depth * cols)
        if layout.raw:
            shape = (depth * layout.slot_shape[0], *layout.slot_shape[1:])
        ring = Ring(
            buffer=device_fn.register_scratch(shape, layout.dtype, name_hint="ring"),
            semaphores=device_fn.register_dma_semaphore("ring_sem", shape=(depth,)),
            slot_shape=layout.slot_shape,
            dtype=layout.dtype,
            depth=depth,
            total=layout.total,
            sites=tuple(sites),
            block_sizes=block_sizes,
            gates=gates,
            raw=layout.raw,
        )
        rings.append(ring)
    rings = _hold_for_exchanges(model, device_fn, rings, exchanges)
    for ring in rings:
        _check_ring(ring)
    for site in model.sites:
        for load in site.loads:
            device_fn.pallas_memory_space[id(load.fake)] = PallasMemorySpace.HBM
    return rings, pooled


def _ring_gate(
    model: StreamModel,
    device_fn: DeviceFunction,
    sites: Sequence[RingSite],
    producer: int,
    depth: int,
) -> RingGate:
    """The gate of root ``producer`` on a ring of depth ``depth`` with the
    (folded) sites ``sites``: its loads that read what the producer writes
    run in the roots after it, in the same folded loop if it has one
    (``_index_read``), so their indices start at the first such root's."""
    loop = model.loop_of(producer)
    later = [
        site.root
        for site in sites
        if site.root.root > producer
        and (
            loop is None or (site.root.loop is not None and site.root.loop.loop is loop)
        )
    ]
    first = min(root.base for root in later)
    return RingGate(
        producer,
        first,
        first + depth,
        None if loop is None else later[0].loop,
        _producer_tiles(model, device_fn, producer),
    )


def _producer_tiles(model: StreamModel, device_fn: DeviceFunction, root: int) -> int:
    """The number of tiles of root ``root`` in this config."""
    tiles = 1
    for block_id, extent in model.root_axes[root]:
        block_size = device_fn.resolved_block_size(block_id)
        assert isinstance(block_size, int)
        tiles *= -(-extent // block_size)
    return tiles


def _exchange_starts(
    graphs: Sequence[GraphInfo], root_ids: Sequence[int]
) -> dict[int, torch.fx.Node]:
    """Per exchange root, a root that pushes data to other devices: its last
    remote copy start in program order that every tile runs, in for loops
    of static, nonempty ranges but under no branch."""

    def last_start(graph_id: int) -> torch.fx.Node | None:
        last = None
        for node in graphs[graph_id].graph.nodes:
            if node.target is start_async_remote_copy_descriptor:
                last = node
            elif node.target is _for_loop:
                graph, begin, end, _ = node.args
                assert isinstance(graph, int)
                assert isinstance(begin, (list, tuple))
                assert isinstance(end, (list, tuple))
                if all(
                    isinstance(lo, int) and isinstance(hi, int) and hi > lo
                    for lo, hi in zip(begin, end, strict=True)
                ):
                    last = last_start(graph) or last
        return last

    return {
        root: node
        for root, graph_id in enumerate(root_ids)
        if (node := last_start(graph_id)) is not None
    }


def _exchange_windows(
    model: StreamModel,
    device_fn: DeviceFunction,
    ring: Ring,
    exchanges: Mapping[int, torch.fx.Node],
) -> list[RingGate]:
    """Per exchange root, the gate that would hold the first ``D`` copies of
    ``ring`` read after it (``_hold_for_exchanges``): those of the roots
    that read the ring after it, before the next exchange root, in the same
    folded loop iteration; or, when none in the body follows a loop's last
    exchange root, those before the body's first one, an iteration later.
    An exchange root that reads the ring itself holds nothing."""
    roots = ring.roots
    gates = []
    for producer in sorted(exchanges):
        if any(root.root == producer for root in roots):
            continue
        loop = model.loop_of(producer)
        readers = [
            root
            for root in roots
            if root.root > producer
            and not any(producer < other < root.root for other in exchanges)
            and all(model.loop_of(r) is loop for r in range(producer, root.root + 1))
        ]
        lag = 0
        if (
            loop is not None
            and loop.trips > 1
            and not any(
                root.root > producer and root.root in loop.roots for root in roots
            )
            and not any(other > producer and other in loop.roots for other in exchanges)
        ):
            first = min(other for other in exchanges if other in loop.roots)
            readers = [
                root for root in roots if root.root in loop.roots and root.root < first
            ]
            lag = 1
        if not readers:
            continue
        gates.append(
            RingGate(
                producer,
                readers[0].base,
                readers[0].base + ring.depth,
                None if loop is None else readers[0].loop,
                _producer_tiles(model, device_fn, producer),
                frozenset(root.root for root in readers),
                lag,
            )
        )
    return gates


def _exchange_before(
    model: StreamModel,
    exchanges: Mapping[int, torch.fx.Node],
    producer: int,
    layer: int,
) -> tuple[int, int, int] | None:
    """The run (``Ring.read_order``) of the exchange root before the run of
    ``producer`` in iteration ``layer`` of its folded loop, if any."""
    loop = model.loop_of(producer)
    earlier = [root for root in exchanges if root < producer]
    if loop is not None:
        if any(root in loop.roots for root in earlier):
            return loop.first_root, layer, max(earlier)
        if layer > 0:
            return loop.first_root, layer - 1, max(set(loop.roots) & set(exchanges))
        earlier = [root for root in earlier if root < loop.first_root]
    if not earlier:
        return None
    root = max(earlier)
    outer = model.loop_of(root)
    if outer is None:
        return root, 0, root
    return outer.first_root, outer.trips - 1, root


def _hold_for_exchanges(
    model: StreamModel,
    device_fn: DeviceFunction,
    rings: list[Ring],
    exchanges: Mapping[int, torch.fx.Node],
) -> list[Ring]:
    """Gate the ring copies each exchange root holds (F23).

    An exchange root pushes data to the other devices with remote copies,
    then waits for theirs: the DMA engines idle for the round trip unless
    copies are queued, while copies queued ahead of the remote copies, or too
    many behind them, delay the exchange.  A ring starts the copy of an index
    once the index ``D`` earlier is read, so the first ``D`` copies read
    after an exchange root start before it (``_exchange_windows``), many
    roots early if the ring is not read in between.  Those whose slots free
    up before the exchange root ahead of it (``_exchange_before``) would
    stream across that one and the roots in between, taking bandwidth from
    their own copies, and land long before they are read: the exchange root
    holds them, earliest first, as a prefix of each window, up to
    ``_EXCHANGE_HOLD_BYTES``, and starts them right behind its last remote
    copy (``MegakernelPlan.remote_copy_started``), where they fill the round
    trip.  The rest are what the rings stream into the exchange anyway, and
    holding them only queues more behind its remote copies, so the only
    exchange root of a folded loop holds nothing.  Rings whose copies read
    index vectors keep their schedules.  With an arena, no ring holds: the
    arena streams in program order, so what it has in flight at an exchange
    root is what the roots right after read, and it keeps the DMA engines
    busy through the round trip.  A copy held there would queue behind the
    arena's copies in flight and land after the roots right after the
    exchange root, which read it.
    """
    if any(ring.raw for ring in rings):
        return rings
    trials = {
        number: dataclasses.replace(
            ring, gates=tuple(_exchange_windows(model, device_fn, ring, exchanges))
        )
        for number, ring in enumerate(rings)
        if not ring.gates
        and not any(
            isinstance(index, IndexRead)
            for site in ring.sites
            for load in site.loads
            for _, index in load.scalars
        )
    }
    held: dict[int, list[RingGate]] = {number: [] for number in trials}
    for producer in exchanges:
        # (start order key, ring, gate, catch-up index) per held candidate.
        candidates: list[tuple[tuple[int, int, int, float], int, RingGate, int]] = []
        for number, trial in trials.items():
            for gate in trial.gates:
                if gate.producer != producer:
                    continue
                # A steady-state run of the producer: its loop's second
                # iteration if the gate's readers run in a later one.
                run = int(gate.loop is not None and gate.loop.layers > gate.lag + 1)
                before = _exchange_before(model, exchanges, producer, run)
                if before is None:
                    continue
                for index in trial.catch_up(gate):
                    predecessor = index + gate.shift(run + gate.lag) - trial.depth
                    key = (
                        trial.read_order(predecessor)
                        if predecessor >= 0
                        else (-1, -1, -1, 0.0)
                    )
                    if key < before:
                        candidates.append((key, number, gate, index))
        candidates.sort(key=operator.itemgetter(0))
        budget = _EXCHANGE_HOLD_BYTES
        last: dict[tuple[int, int], int] = {}
        for _, number, gate, index in candidates:
            ring = trials[number]
            budget -= math.prod(ring.slot_shape) * ring.dtype.itemsize
            if budget < 0:
                break
            last[number, id(gate)] = index
        for number, trial in trials.items():
            held[number].extend(
                dataclasses.replace(gate, threshold=last[number, id(gate)] + 1)
                for gate in trial.gates
                if (number, id(gate)) in last
            )
    return [
        dataclasses.replace(ring, gates=tuple(held[number]))
        if held.get(number)
        else ring
        for number, ring in enumerate(rings)
    ]


def _fold_ring_sites(sites: list[RingSite], model: StreamModel) -> list[RingSite]:
    """Lay out the roots of folded host loops in a ring laid out as if every
    root ran once: each iteration of a loop's body takes its roots' indices
    again, and the roots after the loop start after its last iteration."""
    if not model.host_loops:
        return sites
    roots = list(dict.fromkeys(site.root for site in sites))
    folded: dict[int, RingRoot] = {}
    shift = 0
    index = 0
    while index < len(roots):
        loop = model.loop_of(roots[index].root)
        if loop is None:
            folded[roots[index].root] = dataclasses.replace(
                roots[index], base=roots[index].base + shift
            )
            index += 1
            continue
        end = index
        while end < len(roots) and model.loop_of(roots[end].root) is loop:
            end += 1
        body = roots[index:end]
        ring_loop = RingLoop(loop, body[0].base + shift, body[-1].end - body[0].base)
        for root in body:
            folded[root.root] = dataclasses.replace(
                root, base=root.base + shift, loop=ring_loop
            )
        shift += (ring_loop.layers - 1) * ring_loop.stride
        index += len(body)
    return [dataclasses.replace(site, root=folded[site.root.root]) for site in sites]


@dataclasses.dataclass(frozen=True)
class _SiteRefill:
    """A refill branch into one loop of a target root."""

    site: RingSite
    # The root has several loops: test that the index falls in this one.
    check_site: bool
    # The load the index belongs to when it is static, else ``None``: branch
    # on the index modulo the loop's loads.
    load: int | None
    # Per load of the loop, the gate whose deferred indices the branch may
    # reach, if any: test that the index is not one.
    gates: tuple[RingGate | None, ...] = ()


@dataclasses.dataclass(frozen=True)
class _RootRefill:
    """A refill branch into one target root, with only the bound tests the
    consumer's range of indices does not already imply."""

    root: RingRoot
    check_lower: bool
    check_upper: bool
    sites: tuple[_SiteRefill, ...]


@dataclasses.dataclass(frozen=True)
class _LoopRefill:
    """A refill branch into the roots of a folded host loop: decode the
    iteration, then branch into the body roots as in the first iteration."""

    loop: RingLoop
    check_lower: bool
    check_upper: bool
    roots: tuple[_RootRefill, ...]


def _refill_branches(
    ring: Ring, site: RingSite, load: int
) -> list[_RootRefill | _LoopRefill]:
    """The branches that start index ``g + D`` after load ``load`` of ``site``."""
    root = site.root
    loads = len(site.loads)
    first = root.base + site.offset + load + ring.depth
    last = first + (root.num_tiles - 1) * root.stride + (site.trips - 1) * loads
    # Each iteration of the root's folded loop refills [first, last] shifted.
    reach = _merge_intervals(
        [
            (first + layer * root.layer_stride, last + layer * root.layer_stride)
            for layer in range(root.layers)
        ]
    )
    branches: list[_RootRefill | _LoopRefill] = []
    loops: list[RingLoop] = []
    for target in ring.roots:
        branch: _RootRefill | _LoopRefill | None
        if target.loop is None:
            branch = _root_refill(ring, site, first, target, reach)
        elif target.loop in loops:
            continue
        else:
            loops.append(target.loop)
            branch = _loop_refill(ring, site, first, target.loop, reach)
        if branch is not None:
            branches.append(branch)
    return branches


def _root_refill(
    ring: Ring,
    site: RingSite,
    first: int,
    target: RingRoot,
    reach: list[tuple[int, int]],
) -> _RootRefill | None:
    """The branch into ``target`` that starts the refill indices ``reach``
    (sorted, disjoint) after a load of ``site`` whose first refill is
    ``first``.  For a root in a folded loop, ``reach`` is in its first
    iteration."""
    if not any(lo < target.end and hi >= target.base for lo, hi in reach):
        return None
    root = site.root
    loads = len(site.loads)
    targets = [s for s in ring.sites if s.root is target]
    site_refills = []
    for target_site in targets:
        target_loads = len(target_site.loads)
        # The load is static when every term of g + D - T' - O' that
        # varies is a multiple of the target loop's loads.
        static = (
            (root.num_tiles == 1 or root.stride % target_loads == 0)
            and (site.trips == 1 or loads % target_loads == 0)
            and (root.layers == 1 or root.layer_stride % target_loads == 0)
            and (target.num_tiles == 1 or target.stride % target_loads == 0)
            and (target.layers == 1 or target.layer_stride % target_loads == 0)
        )
        site_refills.append(
            _SiteRefill(
                target_site,
                check_site=len(targets) > 1,
                load=(first - target.base - target_site.offset) % target_loads
                if static
                else None,
                gates=tuple(
                    gate
                    if (gate := ring.gate_of(target.root, load)) is not None
                    and gate.binds(root)
                    and any(
                        lo <= gate_hi and gate_lo <= hi
                        for lo, hi in reach
                        for gate_lo, gate_hi in gate.intervals(target)
                    )
                    else None
                    for load in target_site.loads
                ),
            )
        )
    return _RootRefill(
        target,
        check_lower=reach[0][0] < target.base,
        check_upper=reach[-1][1] >= target.end,
        sites=tuple(site_refills),
    )


def _loop_refill(
    ring: Ring,
    site: RingSite,
    first: int,
    loop: RingLoop,
    reach: list[tuple[int, int]],
) -> _LoopRefill | None:
    """The branch into the roots of folded loop ``loop``; see ``_root_refill``."""
    inside = [
        (max(lo, loop.base), min(hi, loop.end - 1))
        for lo, hi in reach
        if lo < loop.end and hi >= loop.base
    ]
    if not inside:
        return None
    body_reach = _merge_intervals(
        [piece for lo, hi in inside for piece in _first_iteration(loop, lo, hi)]
    )
    return _LoopRefill(
        loop,
        check_lower=reach[0][0] < loop.base,
        check_upper=reach[-1][1] >= loop.end,
        roots=tuple(
            branch
            for target in ring.roots
            if target.loop is loop
            and (branch := _root_refill(ring, site, first, target, body_reach))
            is not None
        ),
    )


def _first_iteration(loop: RingLoop, lo: int, hi: int) -> list[tuple[int, int]]:
    """The stream indices ``[lo, hi]`` of ``loop`` moved to its first iteration."""
    if hi - lo + 1 >= loop.stride:
        return [(loop.base, loop.base + loop.stride - 1)]
    lo = loop.base + (lo - loop.base) % loop.stride
    hi = loop.base + (hi - loop.base) % loop.stride
    if lo <= hi:
        return [(lo, hi)]
    return [(lo, loop.base + loop.stride - 1), (loop.base, hi)]


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, disjoint union of the closed integer intervals ``intervals``."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _simulate_refill(
    branches: Sequence[_RootRefill | _LoopRefill], index: int, layer: int = 0
) -> list[tuple[RingSite, int, int, int, int]]:
    """The copies the emitted refill branches start for stream index ``index``
    (in iteration ``layer`` of a folded loop around the branches' roots)."""
    started = []
    for branch in branches:
        if isinstance(branch, _LoopRefill):
            loop = branch.loop
            if (branch.check_lower and index < loop.base) or (
                branch.check_upper and index >= loop.end
            ):
                continue
            if loop.layers > 1:
                loop_layer, rest = divmod(index - loop.base, loop.stride)
                started.extend(
                    _simulate_refill(branch.roots, loop.base + rest, loop_layer)
                )
            else:
                started.extend(_simulate_refill(branch.roots, index))
            continue
        root = branch.root
        if (branch.check_lower and index < root.base) or (
            branch.check_upper and index >= root.end
        ):
            continue
        rel = index - root.base
        tile, within = divmod(rel, root.stride) if root.num_tiles > 1 else (0, rel)
        for site_refill in branch.sites:
            site = site_refill.site
            if site_refill.check_site and not (
                site.offset <= within < site.offset + site.span
            ):
                continue
            pos = within - site.offset
            loads = len(site.loads)
            iteration = pos // loads
            for load in range(loads):
                if site_refill.load is None:
                    if pos % loads != load:
                        continue
                elif site_refill.load != load:
                    continue
                gate = site_refill.gates[load] if site_refill.gates else None
                if gate is not None and not gate.admits(root, index, layer):
                    continue
                started.append((site, layer, tile, iteration, load))
    return started


def _boundary_range(count: int, margin: int, walk_all: bool) -> list[int]:
    if walk_all or count <= 2 * margin:
        return list(range(count))
    return [*range(margin), *range(count - margin, count)]


def _check_ring(ring: Ring) -> None:
    """Statically prove the ring schedule: every stream index is started once,
    by the prologue or by the consumer ``D`` indices earlier, and is waited on
    by exactly one consumer."""
    base = 0
    for root in ring.roots:
        assert root.base == base, (root, base)
        offset = 0
        for site in ring.sites:
            if site.root is root:
                assert site.offset == offset, (site, offset)
                offset += site.span
        assert offset == root.stride, (root, offset)
        assert root.num_tiles == math.prod(axis[1] for axis in root.axes)
        base = root.end
        if root.loop is not None and base == root.loop.base + root.loop.stride:
            # The last root of a folded loop's body: skip its other iterations.
            base = root.loop.end
    assert base == ring.total, (base, ring.total)
    assert max(len(site.loads) for site in ring.sites) <= ring.depth <= ring.total

    walk_all = ring.total <= _STREAM_CHECK_LIMIT
    consumed: list[int] = []
    for site in ring.sites:
        root = site.root
        loads = len(site.loads)
        branches = [_refill_branches(ring, site, load) for load in range(loads)]
        tile_margin = -(-(ring.depth + root.stride) // root.stride) + 1
        trip_margin = -(-ring.depth // loads) + 1
        layer_margin = -(-ring.depth // max(root.layer_stride, 1)) + 1
        for tile, iteration, layer in itertools.product(
            _boundary_range(root.num_tiles, tile_margin, walk_all),
            _boundary_range(site.trips, trip_margin, walk_all),
            _boundary_range(root.layers, layer_margin, walk_all),
        ):
            for load in range(loads):
                index = (
                    root.base
                    + layer * root.layer_stride
                    + tile * root.stride
                    + site.offset
                    + iteration * loads
                    + load
                )
                if ring.instance_at(index) != (site, layer, tile, iteration, load):
                    raise AssertionError(
                        f"stream index {index} does not decode to its consumer"
                    )
                consumed.append(index)
                gate = ring.gate_of(root.root, site.loads[load])
                if (
                    gate is not None
                    and not gate.readers
                    and index < gate.first + gate.shift(layer)
                ):
                    raise AssertionError(
                        f"stream index {index} is read before its producer ends"
                    )
                refill = index + ring.depth
                expected = (
                    [ring.instance_at(refill)]
                    if refill < ring.total and ring.gate_at(refill) is None
                    else []
                )
                started = _simulate_refill(branches[load], refill)
                if started != expected:
                    raise AssertionError(
                        f"the refill after stream index {index} starts "
                        f"{started}, expected {expected}"
                    )
    if walk_all and sorted(consumed) != list(range(ring.total)):
        raise AssertionError("the ring's consumers do not wait on each index once")
    # The producers' catch-up batches start each deferred index once, after
    # the producer ends and once its slot's previous index is read.
    caught: list[int] = []
    deferred: set[int] = set()
    for gate in ring.gates:
        batch = ring.catch_up(gate)
        for layer in range(gate.lag, 1 if gate.loop is None else gate.loop.layers):
            first = gate.first + gate.shift(layer)
            deferred.update(
                index
                for index in range(first, min(first + ring.depth, ring.total))
                if ring.gate_at(index) is gate
            )
            for index in (index + gate.shift(layer) for index in batch):
                if not index - ring.depth < first <= index or (
                    gate.loop is not None and ring.instance_at(index)[1] != layer
                ):
                    raise AssertionError(
                        f"the catch-up after root {gate.producer} starts stream "
                        f"index {index} out of order"
                    )
                caught.append(index)
    if sorted(caught) != sorted(deferred):
        raise AssertionError(
            f"the catch-up batches start {sorted(caught)}, expected {sorted(deferred)}"
        )


def _ring_predecessor(rings: Sequence[Ring], ring: Ring) -> Ring | None:
    """The ring after whose last refill ``ring``'s prologue starts, or
    ``None`` for kernel start.

    A ring read only after other rings' reads end would sit full while they
    drain.  Its prologue starts instead after the last refill of the one of
    them read last (then streaming the most bytes), which keeps the DMA queue
    full as one shared ring would.  Rings without refills drain right after
    kernel start, so they do not count.  A gated ring primes at kernel start,
    so its catch-up batches never precede its prologue.
    """
    if ring.gates:
        return None
    first = min(root.root for root in ring.roots)
    loops = _repeated_loops(ring)
    earlier = [
        other
        for other in rings
        if other.total > other.depth
        and max(root.root for root in other.roots) < first
        and not any(loop in loops for loop in _repeated_loops(other))
    ]
    return max(
        earlier,
        key=lambda other: (max(root.root for root in other.roots), other.traffic),
        default=None,
    )


def _repeated_loops(ring: Ring) -> list[int]:
    """Ids of the folded loops of more than one iteration around ``ring``'s
    roots, whose reads interleave with those of other rings in the loop."""
    return [
        id(root.loop.loop)
        for root in ring.roots
        if root.loop is not None and root.layers > 1
    ]


def _catch_up_statements(
    device_fn: DeviceFunction, ring: Ring, gate: RingGate
) -> list[ast.stmt]:
    """At the end of the last tile of ``gate``'s producer: start the copies
    it deferred (``Ring.catch_up``), of the current iteration of its folded
    loop if it has one (of the next one, with a ``lag``, unless it is the
    last)."""
    statements: list[ast.stmt] = []
    next_layer = None if not gate.lag else device_fn.new_var("_layer")
    shift = gate.shift(gate.lag)
    for index in ring.catch_up(gate):
        site, iteration_of_loop, tile, iteration, load = ring.instance_at(index)
        # Inside the producer's folded loop, the copy reads the current
        # iteration's index vector and slots.
        layer = None if gate.loop is not None else iteration_of_loop
        slot: str | int = index % ring.depth
        if gate.loop is not None and gate.loop.layers > 1:
            layer = next_layer
            slot = (
                f"({index + shift} + {_layer_term(device_fn, gate.loop)}) "
                f"% {ring.depth}"
            )
        statements.append(
            statement_from_string(
                ring.copy(
                    device_fn,
                    site,
                    site.loads[load],
                    iteration,
                    site.root.tile_offsets(tile),
                    slot,
                    layer=layer,
                )
                + ".start()"
            )
        )
    if not statements:
        return statements
    if next_layer is not None:
        assert gate.loop is not None
        statements = [
            statement_from_string(
                f"{next_layer} = {_loop_iteration(device_fn, gate.loop)} + {gate.lag}"
            ),
            *_when(device_fn, f"{next_layer} < {gate.loop.layers}", statements),
        ]
    if gate.tiles == 1:
        return statements
    pid = device_fn.pid.shared_pid_var  # pyrefly: ignore[missing-attribute]
    return _when(device_fn, f"{pid} == {gate.tiles - 1}", statements, "_catch_up")


def _handoff(
    rings: Sequence[Ring], ring: Ring, site: RingSite
) -> tuple[int, int, list[Ring]] | None:
    """``(load, index, successors)`` when load ``load`` of ``site`` reads
    ``ring``'s first stream index ``index`` without a refill, which starts the
    prologues of the rings ``successors``."""
    successors = [other for other in rings if _ring_predecessor(rings, other) is ring]
    index = ring.total - ring.depth
    consumer, _, _, _, load = ring.instance_at(index)
    if not successors or consumer is not site:
        return None
    return load, index, successors


def _handoff_statements(
    device_fn: DeviceFunction,
    rings: Sequence[Ring],
    ring: Ring,
    site: RingSite,
    load: int,
    index: str,
) -> list[ast.stmt]:
    """After load ``load`` of ``site``, whose first load reads the stream
    index in variable ``index``: start the prologues ``_handoff`` places
    there, if any."""
    handoff = _handoff(rings, ring, site)
    if handoff is None or handoff[0] != load:
        return []
    _, target, successors = handoff
    return _when(
        device_fn,
        f"{index} == {target - load}",
        [
            statement_from_string(statement)
            for successor in successors
            for statement in successor.prime(device_fn)
        ],
        name="_prime",
    )


# A ring-consumer ``fori_loop`` step has a fixed cost (~0.2 us on v7: Mosaic
# does not overlap one step's dot with the next), so a step should stream at
# least ``_STREAM_STEP_BYTES``.  Narrow tiles (e.g. 128-row weight tiles) leave
# a single-iteration step compute bound below the DMA rate.  A step that
# already streams this much gains little from replication, and Mosaic can
# schedule a replicated body worse (a 2.5 MiB LM-head argmax step drops from
# 2.26 to 1.83 TB/s at any unroll >= 2 on v7).
_STREAM_STEP_BYTES = 2 << 20
_MAX_STREAM_UNROLL = 8


def stream_unroll(device_fn: DeviceFunction, graph_id: int) -> int:
    """Iterations of ring-consumer loop ``graph_id`` traced per ``fori_loop``
    step: ``pallas_stream_unroll``, or by default the smallest power of two
    whose iterations stream ``_STREAM_STEP_BYTES``; at most the trip count."""
    plan = device_fn.pallas_megakernel
    assert plan is not None
    sites = plan.stream_sites(graph_id)
    trips = sites[0][1].trips
    unroll = device_fn.config.get("pallas_stream_unroll")
    if unroll is None:
        step = sum(
            len(site.loads) * math.prod(ring.slot_shape) * ring.dtype.itemsize
            for ring, site in sites
        )
        unroll = 1
        while unroll * step < _STREAM_STEP_BYTES and unroll < _MAX_STREAM_UNROLL:
            unroll *= 2
    assert isinstance(unroll, int)
    return max(1, min(unroll, trips))


def stream_wait_group(device_fn: DeviceFunction, graph_id: int, unroll: int) -> int:
    """Iterations of ring-consumer loop ``graph_id`` whose ring waits run
    before their bodies: ``pallas_stream_wait_group``, at most ``unroll``.

    A DMA wait orders every later vector op after it, so a wait per dot
    exposes the MXU latency on every dot (VMEM-fed M=16 GEMV on v7: 6.1 TB/s
    without waits, 2.7 with one per tile, 4.7 / 5.0 with one group of 4 / 8).
    But a group's refills start only after its last tile lands, so G - 1
    fewer copies are in flight, which slows a loop at the HBM rate (TPU step
    L=64: 2044.5 µs with a wait per iteration, 2146.1 with the whole unroll
    grouped).  Groups are small enough that every ring holds a group's
    tiles: the copy of index ``g`` starts after index ``g - D`` is read.
    """
    plan = device_fn.pallas_megakernel
    assert plan is not None
    requested = device_fn.config.get("pallas_stream_wait_group") or 1
    assert isinstance(requested, int)
    group = min(unroll, requested)
    for ring, site in plan.stream_sites(graph_id):
        group = min(group, ring.depth // len(site.loads))
    return max(1, group)


def emit_stream_site(
    state: CodegenState, graph_id: int, loop_var: str
) -> tuple[dict[str, str], list[ast.stmt], list[ast.stmt]]:
    """Ring code for the fori body of inner loop ``graph_id``.

    Returns the ring window each streamed tensor's loads read, the waits that
    precede the body, and the refills that follow it.
    """
    device_fn = state.device_function
    plan = device_fn.pallas_megakernel
    assert plan is not None
    routes: dict[str, str] = {}
    waits: list[ast.stmt] = []
    refills: list[ast.stmt] = []
    for ring, site in plan.stream_sites(graph_id):
        root = site.root
        loads = len(site.loads)
        index = device_fn.new_var("_g")
        terms = [str(root.base + site.offset)] if root.base + site.offset else []
        if root.loop is not None and root.layers > 1:
            terms.append(_layer_term(device_fn, root.loop))
        if root.num_tiles > 1:
            terms.append(f"{device_fn.pid.shared_pid_var} * {root.stride}")  # pyrefly: ignore[missing-attribute]
        if site.trips > 1:
            terms.append(str(_scaled(loop_var, loads)) if loads > 1 else loop_var)
        waits.append(statement_from_string(f"{index} = {' + '.join(terms) or 0}"))
        tile_offsets: dict[int, str | int] = {
            block_id: state.codegen.offset_var(block_id) for block_id, _, _ in root.axes
        }
        for load_index, load in enumerate(site.loads):
            slot = device_fn.new_var("_slot")
            position = index if load_index == 0 else f"({index} + {load_index})"
            waits.append(statement_from_string(f"{slot} = {position} % {ring.depth}"))
            name = device_fn.tensor_arg(load.fake).name
            window = device_fn.new_var(f"{name}_tile")
            waits.append(statement_from_string(f"{window} = {ring.window(slot, load)}"))
            copy = ring.copy(
                device_fn, site, load, loop_var, tile_offsets, slot, window, wait=True
            )
            waits.append(statement_from_string(f"{copy}.wait()"))
            routes[name] = window
            if load.scalars:
                plan.window_scalar_dims[window] = frozenset(
                    dim for dim, _ in load.scalars
                )
            refills.extend(
                _early_statements(
                    device_fn,
                    plan.early_refills.get(
                        (ring.buffer, site.graph_id, load_index), []
                    ),
                    f"{index} + {load_index + ring.depth}",
                )
            )
            refills.extend(
                _refill_statements(
                    device_fn,
                    ring,
                    _refill_branches(ring, site, load_index),
                    f"{index} + {load_index + ring.depth}",
                    slot,
                )
            )
            refills.extend(
                _handoff_statements(
                    device_fn, plan.stream, ring, site, load_index, index
                )
            )
    return routes, waits, refills


def ring_window_parts(
    device_fn: DeviceFunction, active_name: str, parts: list[str]
) -> list[str]:
    """The index ``parts`` of a load through ``active_name`` without the
    integer-indexed dims a ring window does not have."""
    plan = device_fn.pallas_megakernel
    if plan is None or active_name not in plan.window_scalar_dims:
        return parts
    dims = plan.window_scalar_dims[active_name]
    return [part for dim, part in enumerate(parts) if dim not in dims]


def _loop_iteration(device_fn: DeviceFunction, loop: RingLoop) -> str:
    """The current iteration of folded loop ``loop``, counted from its
    start, in the loop body."""
    var = device_fn.expr_to_var_info[loop.loop.symbol].name
    return f"({var} - {loop.loop.start})" if loop.loop.start else var


def _layer_term(device_fn: DeviceFunction, loop: RingLoop) -> str:
    """The stream index offset of the current iteration of folded loop
    ``loop``, in the loop body."""
    return str(_scaled(_loop_iteration(device_fn, loop), loop.stride))


def _refill_statements(
    device_fn: DeviceFunction,
    ring: Ring,
    branches: Sequence[_RootRefill | _LoopRefill],
    next_index: str,
    slot: str,
    layer: str | int | None = None,
) -> list[ast.stmt]:
    """Emit ``branches``: start stream index ``next_index`` into ``slot``.

    ``layer`` is the iteration of the folded loop around the branches' roots.
    """
    if not branches:
        return []
    index = device_fn.new_var("_gn")
    statements = [statement_from_string(f"{index} = {next_index}")]
    for branch in branches:
        if isinstance(branch, _LoopRefill):
            statements.extend(
                _loop_refill_statements(device_fn, ring, branch, index, slot)
            )
            continue
        root = branch.root
        body: list[ast.stmt] = []
        rel = index
        if root.base:
            rel = device_fn.new_var("_e")
            body.append(statement_from_string(f"{rel} = {index} - {root.base}"))
        tile_offsets: dict[int, str | int]
        if root.num_tiles > 1:
            tile = device_fn.new_var("_p")
            within = device_fn.new_var("_q")
            body.extend(
                [
                    statement_from_string(f"{tile} = {rel} // {root.stride}"),
                    statement_from_string(f"{within} = {rel} % {root.stride}"),
                ]
            )
            tile_offsets = root.tile_offsets(tile)
        else:
            within = rel
            tile_offsets = root.tile_offsets(0)
        for site_refill in branch.sites:
            site = site_refill.site
            loads = len(site.loads)
            site_body: list[ast.stmt] = []
            pos = within
            if site.offset:
                pos = device_fn.new_var("_u")
                site_body.append(
                    statement_from_string(f"{pos} = {within} - {site.offset}")
                )
            iteration = pos
            if loads > 1 and site.block_id is not None:
                iteration = device_fn.new_var("_jn")
                site_body.append(
                    statement_from_string(f"{iteration} = {pos} // {loads}")
                )
            for load_index, load in enumerate(site.loads):
                if site_refill.load is not None and site_refill.load != load_index:
                    continue
                start = [
                    statement_from_string(
                        ring.copy(
                            device_fn,
                            site,
                            load,
                            iteration,
                            tile_offsets,
                            slot,
                            layer=layer,
                        )
                        + ".start()"
                    )
                ]
                gate = site_refill.gates[load_index] if site_refill.gates else None
                if gate is not None:
                    start = _when(device_fn, gate.condition(root, index, layer), start)
                site_body.extend(
                    start
                    if site_refill.load is not None
                    else _when(device_fn, f"{pos} % {loads} == {load_index}", start)
                )
            site_conditions = []
            if site_refill.check_site and site.offset > 0:
                site_conditions.append(f"{within} >= {site.offset}")
            if site_refill.check_site and site.offset + site.span < root.stride:
                site_conditions.append(f"{within} < {site.offset + site.span}")
            body.extend(_when(device_fn, _all(site_conditions), site_body))
        conditions = [
            *([f"{index} >= {root.base}"] if branch.check_lower else []),
            *([f"{index} < {root.end}"] if branch.check_upper else []),
        ]
        statements.extend(_when(device_fn, _all(conditions), body))
    return statements


def _early_statements(
    device_fn: DeviceFunction, runs: Sequence[_EarlyRun], next_index: str
) -> list[ast.stmt]:
    """Start the copies of ``runs`` that go ahead of the refill that starts
    stream index ``next_index``."""
    if not runs:
        return []
    index = device_fn.new_var("_ge")
    statements = [statement_from_string(f"{index} = {next_index}")]
    for run in runs:
        if run.count == 1:
            condition = f"{index} == {run.index}"
            layer: str | int = run.first
            body: list[ast.stmt] = []
        else:
            last = run.index + (run.count - 1) * run.stride
            condition = _all(
                [
                    f"{index} >= {run.index}",
                    f"{index} <= {last}",
                    *(
                        [f"({index} - {run.index}) % {run.stride} == 0"]
                        if run.stride > 1
                        else []
                    ),
                ]
            )
            layer = device_fn.new_var("_layer")
            step = f"({index} - {run.index})"
            if run.stride > 1:
                step = f"{step} // {run.stride}"
            body = [
                statement_from_string(
                    f"{layer} = {step} + {run.first}"
                    if run.first
                    else f"{layer} = {step}"
                )
            ]
        body.append(statement_from_string(run.copy.copy(device_fn, layer) + ".start()"))
        statements.extend(_when(device_fn, condition, body, name="_early"))
    return statements


def _root_site_index(device_fn: DeviceFunction, site: RingSite) -> str:
    """The stream index of the first load of ``site``, a root's own body, in
    the current tile of the root."""
    root = site.root
    terms = [str(root.base + site.offset)] if root.base + site.offset else []
    # A folded host loop around the root repeats its sites once per layer.
    if root.loop is not None and root.layers > 1:
        terms.append(_layer_term(device_fn, root.loop))
    if root.num_tiles > 1:
        pid = device_fn.pid.shared_pid_var  # pyrefly: ignore[missing-attribute]
        terms.append(f"{pid} * {root.stride}" if root.stride > 1 else pid)
    return " + ".join(terms) or "0"


def _emit_root_stream_load(
    state: CodegenState,
    ring: Ring,
    site: RingSite,
    load_index: int,
    subscript: Sequence[object],
) -> ast.AST:
    """Wait for load ``load_index`` of root-body ``site`` and read its window."""
    device_fn = state.device_function
    load = site.loads[load_index]
    index = _root_site_index(device_fn, site)
    slot = device_fn.new_var("_slot")
    position = index if load_index == 0 else f"{index} + {load_index}"
    window = device_fn.new_var(f"{device_fn.tensor_arg(load.fake).name}_tile")
    tile_offsets: dict[int, str | int] = {
        block_id: state.codegen.offset_var(block_id)
        for block_id, _, _ in site.root.axes
    }
    for statement in (
        f"{slot} = ({position}) % {ring.depth}",
        f"{window} = {ring.window(slot, load)}",
        ring.copy(device_fn, site, load, 0, tile_offsets, slot, window, wait=True)
        + ".wait()",
    ):
        state.codegen.add_statement(statement_from_string(statement))
    # The window holds the whole tile, without the scalar-indexed dims: keep
    # its dims and expand None ones, as ``index_parts`` does.
    parts: list[str] = []
    none_dims: list[int] = []
    for idx in subscript:
        if idx is None:
            none_dims.append(len(parts) + len(none_dims))
        elif not is_scalar_index(idx):
            parts.append(":")
    result = expr_from_string(f"{window}[{', '.join(parts)}]")
    if load.packing > 1:
        row = dict(load.scalars)[len(load.shape) - 2]
        result = unpack_row(
            result, _scalar_index(device_fn, site.root, row, None), load.fake.dtype
        )
    if load.perm is not None:
        # ``_lane_dense_access`` left no None dims.
        result = logical_load(
            state, result, load.perm, [not is_scalar_index(idx) for idx in subscript]
        )
    for dim in none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result


_INT_OPS = (
    (ast.Add, operator.add),
    (ast.Sub, operator.sub),
    (ast.Mult, operator.mul),
    (ast.FloorDiv, operator.floordiv),
    (ast.Mod, operator.mod),
)


def _eval_int(node: ast.AST, values: Mapping[str, int]) -> int | None:
    """Value of an integer expression over ``values``, or None if unknown."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_int(node.operand, values)
        return None if operand is None else -operand
    if isinstance(node, ast.BinOp):
        left = _eval_int(node.left, values)
        right = _eval_int(node.right, values)
        if left is None or right is None:
            return None
        for op_type, op in _INT_OPS:
            # Codegen's nodes are subclasses of the ast classes.
            if isinstance(node.op, op_type):
                return op(left, right)
    return None


def _loop_refill_statements(
    device_fn: DeviceFunction,
    ring: Ring,
    branch: _LoopRefill,
    index: str,
    slot: str,
) -> list[ast.stmt]:
    """Emit a refill branch into a folded loop: decode the iteration of stream
    index ``index``, then branch into the body roots."""
    loop = branch.loop
    body: list[ast.stmt] = []
    layer: str | int = 0
    within = index
    if loop.layers > 1:
        rel = index
        if loop.base:
            rel = device_fn.new_var("_e")
            body.append(statement_from_string(f"{rel} = {index} - {loop.base}"))
        layer = device_fn.new_var("_layer")
        body.append(statement_from_string(f"{layer} = {rel} // {loop.stride}"))
        within = f"{rel} % {loop.stride}"
        if loop.base:
            within = f"{loop.base} + {within}"
    body.extend(_refill_statements(device_fn, ring, branch.roots, within, slot, layer))
    conditions = [
        *([f"{index} >= {loop.base}"] if branch.check_lower else []),
        *([f"{index} < {loop.end}"] if branch.check_upper else []),
    ]
    return _when(device_fn, _all(conditions), body)


def _all(conditions: list[str]) -> str | None:
    if len(conditions) > 1:
        return " & ".join(f"({condition})" for condition in conditions)
    return conditions[0] if conditions else None


def _when(
    device_fn: DeviceFunction,
    condition: str | None,
    body: list[ast.stmt],
    name: str = "_refill",
) -> list[ast.stmt]:
    """``body`` under ``@pl.when(condition)``, or inline without a condition."""
    if condition is None:
        return body
    fn_def = statement_from_string(
        f"@pl.when({condition})\ndef {device_fn.new_var(name)}():\n    pass"
    )
    assert isinstance(fn_def, ast.FunctionDef)
    fn_def.body = body
    return [fn_def]
