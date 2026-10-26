"""Triton-backend codegen for the atomic ops defined in ``helion.language.atomic_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``atomic_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

from ... import exc
from ...language import _decorators
from ...language.atomic_ops import _to_ast_values
from ...language.atomic_ops import atomic_add
from ...language.atomic_ops import atomic_and
from ...language.atomic_ops import atomic_cas
from ...language.atomic_ops import atomic_max
from ...language.atomic_ops import atomic_min
from ...language.atomic_ops import atomic_or
from ...language.atomic_ops import atomic_xchg
from ...language.atomic_ops import atomic_xor
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


def _sem(state: CodegenState) -> object:
    return state.proxy_arg(len(state.ast_args) - 1)


def _program_sync_supported() -> bool:
    # TileIR does not legalize tl.debug_barrier (ttg.barrier); see generate_ast.
    return CompileEnvironment.current().backend_name != "tileir"


def _sync_before_release(state: CodegenState) -> None:
    # Triton issues a scalar atomic from one thread; bar.sync puts every thread's
    # writes (TMA drained) before that release. Branches are uniform per program.
    if _sem(state) in ("release", "acq_rel") and _program_sync_supported():
        for drain in state.device_function.async_store_drain():
            state.add_statement(drain)
        state.add_statement(statement_from_string("tl.debug_barrier()"))


def _sync_after_acquire(state: CodegenState, atomic: ast.AST) -> ast.AST:
    if _sem(state) not in ("acquire", "acq_rel") or not _program_sync_supported():
        return atomic
    fx_node = state.fx_node
    # A used scalar result is broadcast through smem behind a bar.sync (Triton's
    # AtomicRMWOpConversion/AtomicCASOpConversion); else only the issuing threads
    # have acquired.
    broadcast = (
        fx_node is not None and len(fx_node.users) > 0 and fx_node.meta["val"].ndim == 0
    )
    fences = state.device_function.async_load_fence()
    if broadcast and not fences:
        return atomic
    old = state.codegen.lift(atomic, prefix="atomic_old")
    if not broadcast:
        state.add_statement(statement_from_string("tl.debug_barrier()"))
    for fence in fences:
        state.add_statement(fence)
    return old


def _codegen_common(
    op: str, state: CodegenState, value_exprs: list[ast.AST]
) -> ast.AST:
    """Route any single-value atomic op through the atomic_indexing strategy."""
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(op)

    device_fn = state.device_function
    _sync_before_release(state)
    fx_node = state.fx_node
    epilogue_subtile_group_id = (
        None if fx_node is None else fx_node.meta.get("epilogue_subtile_group_id")
    )
    if epilogue_subtile_group_id is None:
        indexing_idx = device_fn.atomic_op_index
        device_fn.atomic_op_index += 1
    elif fx_node is not None and fx_node.meta.get(
        "epilogue_subtile_primary_output", False
    ):
        indexing_idx = device_fn.atomic_op_index
        device_fn.atomic_op_index += 1
        device_fn.epilogue_subtile_atomic_indices[epilogue_subtile_group_id] = (
            indexing_idx
        )
    else:
        indexing_idx = device_fn.epilogue_subtile_atomic_indices[
            epilogue_subtile_group_id
        ]
    strategy = device_fn.get_atomic_indexing_strategy(indexing_idx)
    return _sync_after_acquire(
        state, strategy.codegen_atomic(op, state, target, index, value_exprs[0], sem)
    )


@_decorators.codegen(atomic_add, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_add", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xchg, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xchg", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_and, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_and", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_or, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_or", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xor, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xor", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_max, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_max", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_min, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_min", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_cas, "triton")
def _(state: CodegenState) -> ast.AST:
    exp_expr = state.ast_args[2]
    val_expr = state.ast_args[3]
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    # CAS always uses pointer (not a TMA reduction op, two values),
    # but increment the counter to keep per-op atomic_indexing aligned.
    device_fn = state.device_function
    device_fn.atomic_op_index += 1
    _sync_before_release(state)

    indices = SubscriptIndexing.create(state, target, index, for_store=True)
    name = state.device_function.tensor_arg(target).name
    output_size = SubscriptIndexing.compute_shape(target, index, state)
    if indices.has_mask():
        # Triton's CAS takes no mask: a masked-off lane instead swaps an
        # element for its own expected value, a no-op.  Each index past its
        # dim wraps around (clamped in case it is past twice), which keeps
        # such lanes in bounds without piling them onto a few addresses.
        fn = state.device_function
        terms = []
        for i, index in enumerate(indices.dim_index_exprs):
            if CompileEnvironment.current().known_equal(target.size(i), 1):
                continue
            size = fn.tensor_size(target, i).name
            stride = fn.tensor_stride(target, i).name
            terms.append(
                f"tl.minimum(tl.where({index} < {size}, {index}, {index} - {size}),"
                f" {size} - 1) * {stride}"
            )
        if terms:
            indices = indices._replace(index_expr=expr_from_string(" + ".join(terms)))
    offset = indices.broadcast_offset(state, output_size)

    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    # Triton's CAS neither broadcasts nor converts cmp and val to the
    # pointer's block and element type (a value computed in float32 for a
    # float16 target is not converted either), so make them match.
    dtype = triton_type(target.dtype)
    shape = state.tile_strategy.shape_str(output_size)

    def as_operand(proxy: object, value: ast.AST) -> ast.AST:
        if not output_size:
            # A literal Triton already types like the target needs no cast
            if (
                isinstance(proxy, int)
                and not isinstance(proxy, bool)
                and target.dtype == torch.int32
                and -(2**31) <= proxy < 2**31
            ) or (isinstance(proxy, float) and target.dtype == torch.float32):
                return value
            return expr_from_string(f"tl.cast({{value}}, {dtype})", value=value)
        rank = proxy.ndim if isinstance(proxy, torch.Tensor) else 0
        if 0 < rank < len(output_size):
            # Broadcasting aligns trailing dims
            dims = ["None"] * (len(output_size) - rank) + [":"] * rank
            value = expr_from_string(f"{{value}}[{', '.join(dims)}]", value=value)
        return expr_from_string(
            f"tl.broadcast_to(tl.cast({{value}}, {dtype}), {shape})", value=value
        )

    exp_ast = as_operand(state.proxy_arg(2), exp_ast)
    val_ast = as_operand(state.proxy_arg(3), val_ast)
    if indices.has_mask():
        mask = state.codegen.lift(indices.mask_expr, dce=True, prefix="cas_mask").id
        exp = state.codegen.lift(exp_ast, dce=True, prefix="cas_exp").id
        exp_ast = expr_from_string(exp)
        val_ast = expr_from_string(f"tl.where({mask}, {{val}}, {exp})", val=val_ast)
    atomic = expr_from_string(
        f"tl.atomic_cas({name} + {{offset}}, {{exp}}, {{val}}, sem={{sem}})",
        offset=offset,
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )
    return _sync_after_acquire(state, atomic)
