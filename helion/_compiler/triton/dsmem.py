"""Codegen of a dsmem transport's stores and load (``cross_loop_dsmem``): a
store also sends its tile into the slab of each transport it feeds, its
cluster's rank 0's (a per-rank consumer's or a replicated slab's: every
rank's; a local slab's: its own CTA's), the load reads the slab from its
CTA's shared memory (``gluon_ops.cluster_send`` / ``cluster_recv``)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..cross_loop_dsmem import CLUSTER_READY
from ..cross_loop_dsmem import DsmemFallback
from ..cross_loop_dsmem import config_int
from ..cross_loop_dsmem import dsmem_transports
from ..cross_loop_dsmem import send_layout
from ..device_function import SyncRole
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..indexing_strategy import _block_size_origin
from ..indexing_strategy import _resolve_codegen_block_id
from ..tile_dependency import TILE_ACCESS_META

if TYPE_CHECKING:
    import torch

    from ..cross_loop_dsmem import DsmemTransport
    from ..inductor_lowering import CodegenState
    from ..tile_dependency import TileAccess
    from .distributed_ops import LoadCodegen
    from .distributed_ops import StoreCodegen


def _access(state: CodegenState) -> TileAccess | None:
    graph = HostFunction.current().device_ir.tile_dependency_graph
    if graph is None or state.fx_node is None:
        return None
    ids = state.fx_node.meta.get(TILE_ACCESS_META, ())
    return graph.accesses[ids[0]] if ids else None


def _transports(state: CodegenState) -> list[tuple[DsmemTransport, int]]:
    """The transports this access feeds (a store: with its index among the
    transport's stores) or reads (its load: -1)."""
    access = _access(state)
    if access is None:
        return []
    transports = dsmem_transports(state.device_function)
    if access.kind == "load":
        transport = transports.get(access.access_id)
        return [] if transport is None else [(transport, -1)]
    return [
        (transport, index)
        for transport in transports.values()
        for index, store in enumerate(transport.stores)
        if store.access_id == access.access_id
    ]


def _slab_offsets(
    state: CodegenState,
    transport: DsmemTransport,
    fake_tensor: torch.Tensor,
    subscript: list[object],
    *,
    store: int,
) -> tuple[str, str, int, int, str | None]:
    """The tile's byte offsets into ``CLUSTER_SMEM`` (a lifted name), its
    shape, the 32-bit elements a thread moves at once, the threads holding
    each element (one of which sends it) and the load's tile mask (a lifted
    name, None: unmasked), for the transport's ``store``-th store or (-1) its
    load: a ranged index (the slab's ``store_ranges`` or ``load_ranges``)
    counts from its tile's first element.  A load's padded full slice (``6``
    columns in a tile of 8) reads the slab only where its mask holds: its
    padding lanes' offsets point at the slab's first element, and the receive
    gives them a masked load's zero."""
    slab = transport.slab
    for_store = store >= 0
    strides, ranges = (
        (slab.store_strides[store], slab.store_ranges[store])
        if for_store
        else (slab.load_strides, slab.load_ranges)
    )
    per_dim = SubscriptIndexing.compute_per_dim_indexing(
        state, fake_tensor, subscript, for_store=for_store
    )
    if per_dim.has_mask() and for_store:
        raise transport.fallback("C1: a masked tile")
    if len(subscript) != len(strides):
        raise transport.fallback("C1: a subscript of new or elided dimensions")
    env = CompileEnvironment.current()
    config = state.device_function.config
    # This config's sizes, never a block's size hint: the tile Gluon lays out.
    shape = [config_int(size, config) for size in per_dim.output_size]
    if None in shape:
        raise transport.fallback("C1: a tile of a size the config leaves dynamic")
    if not shape and not for_store:
        raise transport.fallback("C1: a load of one element")
    # One element moves as a tile of one.
    shape_str = state.tile_strategy.shape_str(per_dim.output_size) if shape else "[1]"
    # The send lays its tile over ``num_warps`` warps (a ring falls back when
    # it sits in a partition of another count, ``gluon_ring``; a receive takes
    # its partition's), with the fewest copies of an element and then the most
    # elements a thread the slab's alignment allows (``send_layout``).
    vec, copies = send_layout(
        cast("list[int]", shape) or [1], transport.vec, config.num_warps
    )
    terms = []
    for index, stride, ranged, k in zip(
        per_dim.dim_index_exprs, strides, ranges, subscript, strict=True
    ):
        if not stride:
            continue
        if ranged and not isinstance(k, slice):
            origin = _block_size_origin(k)
            if origin is None:
                raise transport.fallback("C1: a range that is no tile")
            begin = state.codegen.offset_var(
                _resolve_codegen_block_id(state, origin.block_id)
            )
            index = f"({index} - ({begin}))"
        terms.append(f"({index}) * {4 * stride}")
    offsets = " + ".join(
        [str(transport.buffer), *terms, env.backend.zeros_expr(shape_str, "tl.int32")]
    )
    mask = None
    if per_dim.has_mask():
        mask = state.codegen.lift(per_dim.mask_expr, dce=True, prefix="cluster_mask").id
        offsets = f"tl.where({mask}, {offsets}, {transport.buffer})"
    name = state.codegen.lift(
        expr_from_string(f"tl.cast({offsets}, tl.int32)"),
        dce=True,
        prefix="cluster_offsets",
    ).id
    return name, shape_str, vec, copies, mask


def dsmem_store_codegen(
    state: CodegenState, codegen_store: StoreCodegen
) -> StoreCodegen:
    """Make a dsmem transport's store also send its tile to the slab of each
    transport it feeds (rank 0's; every rank's for a per-rank consumer or a
    replicated slab; its own CTA's for a local slab)."""
    transports = _transports(state)
    if not transports:
        return codegen_store

    def send(
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
        cache_modifier: ast.AST | None,
    ) -> ast.AST:
        if extra_mask is not None:
            # Off for every load it feeds.
            raise DsmemFallback(
                (transport.index for transport, _ in transports),
                f"{transports[0][0].name}: C1: a masked store",
            )
        device_function = state.device_function
        if not isinstance(value, ast.Constant):
            value = state.codegen.lift(value, dce=True, prefix="cluster_value")
        stored = codegen_store(
            state, fake_tensor, subscript, value, extra_mask, cache_modifier
        )
        dtype = CompileEnvironment.current().backend.dtype_str(fake_tensor.dtype)
        slabs = [
            _slab_offsets(state, transport, fake_tensor, subscript, store=index)
            for transport, index in transports
        ]
        # The store's tile, the same for every transport.
        _, shape, _, _, _ = slabs[0]
        words = state.codegen.lift(
            expr_from_string(
                f"tl.broadcast_to(tl.cast({{value}}, {dtype}), {shape})",
                value=value,
            ),
            dce=True,
            prefix="cluster_words",
        ).id
        for (transport, _), (offsets, _, vec, copies, _) in zip(
            transports, slabs, strict=True
        ):
            # Every rank's slab for a per-rank consumer or a replicated slab,
            # this CTA's for a local slab (-1), else rank 0's.
            ranks = (
                range(transport.size)
                if transport.per_rank or transport.replicated
                else [-1]
                if transport.local
                else [0]
            )
            for rank in ranks:
                sent = device_function.new_var("cluster_sent", dce=False)
                device_function.sync_roles[sent] = SyncRole("cluster")
                state.add_statement(
                    statement_from_string(
                        f"{sent} = _helion_gluon.cluster_send({words}, {offsets}, "
                        f"{CLUSTER_READY}, {transport.barrier}, {vec}, {rank}, "
                        f"{copies})"
                    )
                )
        return stored

    return send


def dsmem_load_codegen(state: CodegenState, codegen_load: LoadCodegen) -> LoadCodegen:
    """Make a dsmem transport's load read its cluster's slab."""
    transports = _transports(state)
    if not transports:
        return codegen_load
    ((transport, _),) = transports

    def receive(
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
        cache_modifier: ast.AST | None,
    ) -> ast.AST:
        if extra_mask is not None:
            raise transport.fallback("C1: a masked load")
        offsets, _, vec, _, mask = _slab_offsets(
            state, transport, fake_tensor, subscript, store=-1
        )
        dtype = CompileEnvironment.current().backend.dtype_str(fake_tensor.dtype)
        received = (
            f"_helion_gluon.cluster_recv({offsets}, {dtype}, {transport.barrier}, "
            f"{vec})"
        )
        if mask is not None:
            received = f"tl.where({mask}, {received}, 0)"
        return expr_from_string(received)

    return receive
