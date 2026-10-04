"""Pallas-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ..._utils import is_scalar_index
from ...language import _decorators
from ...language.memory_ops import _maybe_materialize_tile_index_load
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..tile_dependency import StorageRole
from . import codegen as pallas_codegen
from .dma import async_copy_statements
from .lane_dense import physical_order
from .lane_dense import physical_store_value
from .megakernel import RingWrite
from .megakernel import RowForward
from .megakernel import is_hbm_resident
from .megakernel import ragged_rows_ref
from .megakernel import ring_waits
from .megakernel import row_forward_loop
from .megakernel import row_read
from .megakernel import row_store_run
from .megakernel import store_ring_slots

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    from ... import exc
    from .dma import emit_immediate_indirect_transfer
    from .tensorcore_plan import TENSORCORE_PLAN_META
    from .tensorcore_plan import DmaScatterPlan
    from .tensorcore_plan import OneHotScatterPlan

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    arg_name = state.device_function.tensor_arg(tensor).name
    name = state.device_function.pallas_tensor_ref_name(tensor)
    name = pallas_codegen.vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    value = pallas_codegen.sliced_value_for_store(
        state, tensor, subscript, parts, value
    )
    perm = device_fn.pallas_lane_dense.get(id(tensor.untyped_storage()))
    if perm is not None:
        # An input passed lane dense is indexed in its physical dim order.
        parts = physical_order(parts, perm)
        stored = state.proxy_arg(2)
        if isinstance(stored, torch.Tensor) and stored.ndim > 0:
            value = physical_store_value(
                state, value, perm, [not is_scalar_index(idx) for idx in subscript]
            )
    idx_str = ", ".join(parts)
    plan = state.fx_node.meta.get(TENSORCORE_PLAN_META) if state.fx_node else None
    if isinstance(plan, DmaScatterPlan):
        dma_ref = pallas_codegen.memory_op_dma_scratch(state)
        if dma_ref is None:
            raise exc.InvalidConfig(
                "indirect DMA store was not admitted by the active scheduler"
            )
        state.codegen.add_statement(
            statement_from_string(f"{dma_ref}[...] = {{value}}", value=value)
        )
        # The fori scheduler emits the writeback after the body. Root grids
        # have no enclosing scheduler, so this call emits it immediately.
        emit_immediate_indirect_transfer(state, plan, arg_name)
        return
    from .gather import emit_scatter_store

    is_scatter = isinstance(plan, OneHotScatterPlan)
    if is_scatter:
        value = emit_scatter_store(state, plan.plan, name, idx_str, value)
    from .ordered_carry import emit_carry_store

    if not is_scatter and state.device_function.carry_tiles:
        if emit_carry_store(state, tensor, subscript, name, idx_str, value):
            return
    if _emit_hbm_resident_store(state, tensor, subscript, parts, value):
        return
    value = pallas_codegen.megakernel_scratch_store_value(
        state, tensor, subscript, name, idx_str, value
    )
    if not is_scatter and _emit_packed_dynamic_row_store(
        state, tensor, subscript, parts, name, value
    ):
        return
    state.codegen.add_statement(
        statement_from_string(f"{name}[{idx_str}] = {{value}}", value=value)
    )


def _is_dynamic_scalar_index(index: object, part: str) -> bool:
    if isinstance(index, torch.Tensor):
        dynamic = index.ndim == 0
    elif isinstance(index, torch.SymInt):
        dynamic = not isinstance(index.node.expr, sympy.Integer)
    else:
        dynamic = False
    return dynamic and ":" not in part and "pl.ds" not in part


def _emit_packed_dynamic_row_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    parts: list[str],
    name: str,
    value: ast.AST,
) -> bool:
    """Store one dynamically indexed row of a packed-dtype VMEM ref.

    Mosaic stores to packed (sub-32-bit) refs must start on a native sublane
    tile (16 rows for bf16), which it cannot prove for ``ref[..., row, :]``
    when ``row`` is a runtime scalar such as a decode position.  Rewrite the
    store as a read-modify-write of the aligned tile that contains the row
    (or of the whole dim when it is shorter than one tile).
    """
    ndim = tensor.ndim
    if ndim < 2 or len(parts) != ndim or tensor.dtype.itemsize >= 4:
        return False
    indices = [index for index in subscript if index is not None]
    if len(indices) != ndim:
        return False
    sublane_dim = ndim - 2
    if not _is_dynamic_scalar_index(indices[sublane_dim], parts[sublane_dim]):
        return False
    from .backend import PallasBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, PallasBackend)
    tile = backend.sublane_tiling(tensor.dtype)
    size = tensor.size(sublane_dim)
    if not isinstance(size, int) or (size % tile != 0 and size > tile):
        return False
    row = state.codegen.tmpvar(prefix="row")
    block = state.codegen.tmpvar(prefix="row_block")
    block_parts = [*parts]
    stmts = [f"{row} = {parts[sublane_dim]}"]
    if size % tile == 0:
        base = state.codegen.tmpvar(prefix="row_base")
        block_parts[sublane_dim] = f"pl.ds({base}, {tile})"
        stmts.append(f"{base} = pl.multiple_of(({row} // {tile}) * {tile}, {tile})")
        offset = f"{row} - {base}"
    else:
        # The whole dim fits in one sublane tile: rewrite all of it.
        block_parts[sublane_dim] = ":"
        offset = row
    block_idx = ", ".join(block_parts)
    stmts.append(f"{block} = {name}[{block_idx}]")
    # Position of the sublane dim among the dims that survive indexing.
    axis = sum(1 for part in parts[:sublane_dim] if ":" in part or "pl.ds" in part)
    for stmt in stmts:
        state.codegen.add_statement(statement_from_string(stmt))
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{block_idx}] = {_patched_row(block, axis, offset)}", value=value
        )
    )
    return True


def _patched_row(block: str, axis: int, offset: str) -> str:
    """``block`` with its row ``offset`` along ``axis`` replaced by ``{value}``."""
    return (
        "jnp.where("
        f"lax.broadcasted_iota(jnp.int32, {block}.shape, {axis}) == {offset}, "
        f"jnp.expand_dims({{value}}, {axis}).astype({block}.dtype), {block})"
    )


def _emit_hbm_resident_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    parts: list[str],
    value: ast.AST,
) -> bool:
    """Store rows of a megakernel tensor kept in HBM (``plan_hbm_resident``).

    The rows are staged in VMEM and written with a DMA that is left pending:
    its wait is emitted before the next read of the tensor, at the latest
    at the end of the root tile (``DeviceFunction.flush_pending_writes``).
    Mosaic DMAs the tiled second-minor dim only at tile-aligned offsets, so
    a row at runtime position ``row`` moves the sublane tile that holds it:
    read the tile, patch the row, write the tile back.  When an inner loop
    reads the tensor next, the write is deferred to the loop, which patches
    the row into the copy of the tile it reads anyway (``row_forward_loop``).
    A run of row stores near one row (``row_store_runs``) moves the window
    of two sublane tiles that holds them once instead.
    A slice of a tensor whose minor dim is ragged (not whole 128 lanes) is
    written through a view of the tensor as rows of that dim, which Mosaic
    can slice (``ragged_rows_ref``).
    """
    device_fn = state.device_function
    if not is_hbm_resident(device_fn, tensor):
        return False
    from .backend import PallasBackend

    env = CompileEnvironment.current()
    backend = env.backend
    assert isinstance(backend, PallasBackend)
    tile = backend.sublane_tiling(tensor.dtype)
    shape = [int(size) for size in tensor.shape]
    perm = device_fn.pallas_lane_dense.get(id(tensor.untyped_storage()))
    if perm is not None:
        # ``parts`` and ``value`` are in physical order already.
        shape = physical_order(shape, perm)
        subscript = physical_order(list(subscript), perm)
    scalar_dims = {dim for dim, index in enumerate(subscript) if is_scalar_index(index)}
    target_parts = [
        parts[dim] if dim in scalar_dims else ":" for dim in range(tensor.ndim)
    ]
    stage_shape = [size for dim, size in enumerate(shape) if dim not in scalar_dims]
    row_dim = tensor.ndim - 2
    row = base = run = None
    axis = sum(1 for dim in range(row_dim) if dim not in scalar_dims)
    storage = id(tensor.untyped_storage())
    if row_dim in scalar_dims:
        run = row_store_run(state)
        row = state.codegen.tmpvar(prefix="row")
        if run is None or run.first:
            base = state.codegen.tmpvar(prefix="row_base")
        else:
            base = device_fn.pallas_row_windows[storage]
        window = tile if run is None else 2 * tile
        target_parts[row_dim] = f"pl.ds({base}, {window})"
        stage_shape.insert(axis, window)
    name = device_fn.tensor_arg(tensor).name
    # The stores of one shape to the storage share a ring of stages
    # (``hbm_store_rings``; one stage for most).  A store waits for the
    # writes before it to the storage first, or, in a ring, only for the
    # one before it in its slot, so no write the stage holds is still
    # pending when the next one stages its rows.
    stage_key = (storage, tuple(stage_shape))
    slots = store_ring_slots(state) if row is None else 1
    stages = device_fn.pallas_hbm_store_stages.setdefault(stage_key, [])
    while len(stages) < slots:
        stages.append(
            (
                device_fn.register_scratch(
                    tuple(stage_shape), tensor.dtype, name_hint=f"{name}_rows"
                ),
                device_fn.register_dma_semaphore(name_hint=f"{name}_rows_sem"),
            )
        )
    slot = 0
    if slots > 1:
        slot = device_fn.pallas_store_ring_next.get(stage_key, 0) % slots
        device_fn.pallas_store_ring_next[stage_key] = slot + 1
    stage, semaphore = stages[slot]
    if (
        tensor.ndim > 2
        and shape[-1] % 128 != 0
        # Interpret mode cannot write through the view, but writes the slice.
        and not env.settings.pallas_interpret
    ):
        # ``_hbm_resident_tensors``: whole slices at integer leading indices.
        target = ragged_rows_ref(name, shape, [parts[dim] for dim in range(row_dim)])
    else:
        target = f"{name}.at[{', '.join(target_parts)}]"

    def write_rows(source: str, value: ast.AST) -> list[ast.stmt]:
        # Patch the row into the sublane tile at ref ``source``, stage the
        # tile and start writing it.
        block = state.codegen.tmpvar(prefix="row_block")
        return [
            statement_from_string(f"{block} = {source}[...]"),
            statement_from_string(
                f"{stage}[...] = {_patched_row(block, axis, f'{row} - {base}')}",
                value=value,
            ),
            *async_copy_statements(
                state, stage, target, semaphore, ("start",), "_rows_out"
            ),
        ]

    # Writes to one storage stay in program order, and a read of the
    # sublane tile sees the rows already written to it.  A ring store's
    # write may overtake the pending ones to other slices.
    ring = RingWrite(stage_key, slot, tuple(target_parts), [])
    statements = (
        ring_waits(device_fn, storage, ring)
        if slots > 1
        else device_fn.flush_pending_writes([storage])
    )
    pending: list[ast.stmt] = []
    if row is None:
        statements.extend(
            [
                statement_from_string(
                    f"{stage}[...] = {{value}}.astype({stage}.dtype)", value=value
                ),
                *async_copy_statements(
                    state, stage, target, semaphore, ("start",), "_rows_out"
                ),
            ]
        )
    elif run is not None:
        # A run of row stores (``row_store_runs``) reads the window of two
        # sublane tiles that holds its rows once, patches each row into it,
        # and writes it once.
        statements.append(statement_from_string(f"{row} = {parts[row_dim]}"))
        if run.first:
            low = f"{row} + {run.low - run.offset}"
            statements.extend(
                [
                    statement_from_string(
                        f"{base} = pl.multiple_of(jnp.minimum(({low}) // {tile} "
                        f"* {tile}, {shape[row_dim] - 2 * tile}), {tile})"
                    ),
                    *async_copy_statements(
                        state, target, stage, semaphore, ("start", "wait"), "_rows_in"
                    ),
                ]
            )
            assert base is not None
            device_fn.pallas_row_windows[storage] = base
        block = state.codegen.tmpvar(prefix="row_block")
        statements.extend(
            [
                statement_from_string(f"{block} = {stage}[...]"),
                statement_from_string(
                    f"{stage}[...] = {_patched_row(block, axis, f'{row} - {base}')}",
                    value=value,
                ),
            ]
        )
        if not run.last:
            for statement in statements:
                state.codegen.add_statement(statement)
            return True
        statements.extend(
            async_copy_statements(
                state, stage, target, semaphore, ("start",), "_rows_out"
            )
        )
    else:
        statements.extend(
            [
                statement_from_string(f"{row} = {parts[row_dim]}"),
                statement_from_string(
                    f"{base} = pl.multiple_of(({row} // {tile}) * {tile}, {tile})"
                ),
            ]
        )
        loop = row_forward_loop(state, tensor)
        if loop is not None:
            rows = state.codegen.tmpvar(prefix=f"{name}_row")
            statements.append(statement_from_string(f"{rows} = {{value}}", value=value))
            value = expr_from_string(rows)
        write = [
            *async_copy_statements(
                state, target, stage, semaphore, ("start", "wait"), "_rows_in"
            ),
            *write_rows(stage, value),
        ]
        if loop is None:
            statements.extend(write)
        else:
            pending = write

            def forward(block: str) -> list[ast.stmt]:
                return [
                    *write_rows(block, expr_from_string(rows)),
                    statement_from_string(f"{block}[...] = {stage}[...]"),
                ]

            assert base is not None
            device_fn.pallas_row_forwards[storage] = RowForward(
                loop, base, tile, write, forward
            )
    for statement in statements:
        state.codegen.add_statement(statement)
    waits = async_copy_statements(
        state, stage, target, semaphore, ("wait",), "_rows_out"
    )
    if slots > 1:
        ring.waits.extend(waits)
        device_fn.pallas_ring_writes.setdefault(storage, []).append(ring)
        device_fn.pallas_pending_writes.setdefault(storage, []).extend(waits)
    else:
        device_fn.pallas_pending_writes[storage] = [*pending, *waits]
    return True


@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .view_ops import _resident_plan

    assert state.fx_node is not None
    if _resident_plan(state.fx_node) is not None:
        return _codegen_resident_load(state)

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
    if tile_index_result is not None:
        return tile_index_result

    row = _emit_hbm_resident_row_load(state, tensor, subscript)
    if row is not None:
        return row
    return pallas_codegen.load_expr(state, list(subscript), tensor)


def _emit_hbm_resident_row_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> ast.AST | None:
    """Load rows of a read-only megakernel tensor kept in HBM (an embedding
    row at a token id), or None for any other load.

    Waits for the copy of the sublane tile holding the rows, started at
    kernel start when legal (``_plan_early_row_reads``) or here, and selects
    the row from it: with the runtime sublane index in the ref subscript for
    32-bit dtypes, like ``vmem_scalar_load``, else by rotating the widened
    tile.
    """
    device_fn = state.device_function
    plan = device_fn.pallas_megakernel
    if (
        not is_hbm_resident(device_fn, tensor)
        or HostFunction.current().device_ir.storage_roles.get(
            id(tensor.untyped_storage())
        )
        is not StorageRole.READ_ONLY
    ):
        return None
    assert plan is not None
    read = plan.early_row_reads.get(id(tensor.untyped_storage()))
    if read is None:
        parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
        read = row_read(
            device_fn,
            tensor,
            [
                parts[dim] if is_scalar_index(index) else ":"
                for dim, index in enumerate(subscript)
            ],
        )
        for statement in read.starts:
            state.codegen.add_statement(statement_from_string(statement))
    state.codegen.add_statement(statement_from_string(read.wait))
    if read.axis is None:
        return expr_from_string(f"{read.stage}[...]")
    selectors = [":"] * (tensor.ndim - sum(map(is_scalar_index, subscript)) + 1)
    selectors[read.axis] = cast("str", read.offset)
    if tensor.dtype.itemsize == 4:
        return expr_from_string(f"{read.stage}[{', '.join(selectors)}]")
    backend = CompileEnvironment.current().backend
    wide = torch.float32 if tensor.dtype.is_floating_point else torch.int32
    tile = backend.sublane_tiling(tensor.dtype)  # pyrefly: ignore[missing-attribute]
    selectors[read.axis] = "0"
    rolled = (
        f"pltpu.roll(lax.convert_element_type({read.stage}[...], "
        f"{backend.dtype_str(wide)}), -({read.offset}) % {tile}, axis={read.axis})"
    )
    return expr_from_string(
        f"lax.convert_element_type({rolled}[{', '.join(selectors)}], "
        f"{backend.dtype_str(tensor.dtype)})"
    )


def _codegen_resident_load(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    return pallas_codegen.resident_ref_load_expr(state, list(subscript), tensor)
