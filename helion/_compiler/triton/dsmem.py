"""Codegen of a dsmem transport's store and load (``cross_loop_dsmem``): the
store also sends its tile into its cluster's rank 0 slab, the load reads the
slab from shared memory (``gluon_ops.cluster_send`` / ``cluster_recv``)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..cross_loop_dsmem import CLUSTER_READY
from ..cross_loop_dsmem import cluster_layout
from ..cross_loop_dsmem import config_int
from ..cross_loop_dsmem import dsmem_transports
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
    from .distributed_ops import LoadCodegen
    from .distributed_ops import StoreCodegen


def _transport(state: CodegenState) -> DsmemTransport | None:
    graph = HostFunction.current().device_ir.tile_dependency_graph
    if graph is None or state.fx_node is None:
        return None
    ids = state.fx_node.meta.get(TILE_ACCESS_META, ())
    if not ids:
        return None
    return dsmem_transports(state.device_function).get(
        graph.accesses[ids[0]].allocation_id
    )


def _slab_offsets(
    state: CodegenState,
    transport: DsmemTransport,
    fake_tensor: torch.Tensor,
    subscript: list[object],
    *,
    for_store: bool,
) -> tuple[str, str, int]:
    """The tile's byte offsets into ``CLUSTER_SMEM`` (a lifted name), its
    shape and the 32-bit elements a thread moves at once: a range's index
    counts from its first element (a tile's offset)."""
    slab = transport.slab
    strides, ranges = (
        (slab.store_strides, slab.store_ranges)
        if for_store
        else (slab.load_strides, slab.load_ranges)
    )
    per_dim = SubscriptIndexing.compute_per_dim_indexing(
        state, fake_tensor, subscript, for_store=for_store
    )
    if not (
        isinstance(per_dim.mask_expr, ast.Constant) and per_dim.mask_expr.value is None
    ):
        raise transport.fallback("C1: a masked tile")
    if len(subscript) != len(strides):
        raise transport.fallback("C1: a subscript of new or elided dimensions")
    env = CompileEnvironment.current()
    config = state.device_function.config
    # This config's sizes, never a block's size hint: the tile Gluon lays out.
    shape = [config_int(size, config) for size in per_dim.output_size]
    if None in shape:
        raise transport.fallback("C1: a tile of a size the config leaves dynamic")
    # The send and the receive lay their tile over ``num_warps`` warps (a ring
    # falls back when one sits in a partition of another count,
    # ``gluon_ring``); each moves the most elements a thread the slab's
    # alignment allows at which no element lands on two threads.
    vec = next(
        (
            vec
            for vec in (4, 2, 1)
            if vec <= transport.vec
            and cluster_layout(cast("list[int]", shape), vec, config.num_warps)
        ),
        None,
    )
    if vec is None:
        raise transport.fallback(f"C3: a {shape} tile has duplicated elements")
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
    shape_str = state.tile_strategy.shape_str(per_dim.output_size)
    offsets = " + ".join(
        [str(transport.buffer), *terms, env.backend.zeros_expr(shape_str, "tl.int32")]
    )
    name = state.codegen.lift(
        expr_from_string(f"tl.cast({offsets}, tl.int32)"),
        dce=True,
        prefix="cluster_offsets",
    ).id
    return name, shape_str, vec


def dsmem_store_codegen(
    state: CodegenState, codegen_store: StoreCodegen
) -> StoreCodegen:
    """Make a dsmem transport's store also send its tile to its cluster's slab."""
    transport = _transport(state)
    if transport is None:
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
            raise transport.fallback("C1: a masked store")
        device_function = state.device_function
        if not isinstance(value, ast.Constant):
            value = state.codegen.lift(value, dce=True, prefix="cluster_value")
        stored = codegen_store(
            state, fake_tensor, subscript, value, extra_mask, cache_modifier
        )
        offsets, shape, vec = _slab_offsets(
            state, transport, fake_tensor, subscript, for_store=True
        )
        dtype = CompileEnvironment.current().backend.dtype_str(fake_tensor.dtype)
        sent = device_function.new_var("cluster_sent", dce=False)
        device_function.sync_roles[sent] = SyncRole("cluster")
        state.add_statement(
            statement_from_string(
                f"{sent} = _helion_gluon.cluster_send(tl.broadcast_to(tl.cast("
                f"{{value}}, {dtype}), {shape}), {offsets}, {CLUSTER_READY}, "
                f"{transport.barrier}, {vec})",
                value=value,
            )
        )
        return stored

    return send


def dsmem_load_codegen(state: CodegenState, codegen_load: LoadCodegen) -> LoadCodegen:
    """Make a dsmem transport's load read its cluster's slab."""
    transport = _transport(state)
    if transport is None:
        return codegen_load

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
        offsets, _, vec = _slab_offsets(
            state, transport, fake_tensor, subscript, for_store=False
        )
        dtype = CompileEnvironment.current().backend.dtype_str(fake_tensor.dtype)
        return expr_from_string(
            f"_helion_gluon.cluster_recv({offsets}, {dtype}, {transport.barrier}, "
            f"{vec})"
        )

    return receive
