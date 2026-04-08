from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING
from typing import Callable

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import _symint_expr
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.variable_origin import GridOrigin
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = [
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
]


_VALID_SEMS: set[str] = {"relaxed", "acquire", "release", "acq_rel"}


def _validate_sem(sem: str) -> None:
    if sem not in _VALID_SEMS:
        raise exc.InternalError(
            ValueError(
                f"Invalid memory semantic '{sem}'. Valid options are: relaxed, acquire, release, acq_rel"
            )
        )


def _prepare_mem_args(
    target: torch.Tensor,
    index: list[object],
    *values: object,
    sem: str = "relaxed",
) -> tuple:
    from .tile_proxy import Tile

    _validate_sem(sem)
    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes_for_index(index)
    return (target, index, *values, sem)


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
    indexing_idx = device_fn.atomic_op_index
    device_fn.atomic_op_index += 1
    strategy = device_fn.get_atomic_indexing_strategy(indexing_idx)
    return strategy.codegen_atomic(op, state, target, index, value_exprs[0], sem)


def _cute_pointer_expr(
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    ast_index: list[object] | tuple[object, ...] | None = None,
) -> str:
    from .memory_ops import _cute_index_exprs

    index_exprs = _cute_index_exprs(state, index, ast_index)
    name = state.device_function.tensor_arg(target).name
    coord = (
        f"({index_exprs[0]},)"
        if len(index_exprs) == 1
        else f"({', '.join(index_exprs)})"
    )
    return f"({name}.iterator + cute.crd2idx({coord}, {name}.layout)).llvm_ptr"


def _codegen_common_cute(
    cute_func: str,
    state: CodegenState,
    *,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
) -> ast.AST:
    from .._compiler.compile_environment import CompileEnvironment

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(cute_func)

    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(target.dtype)
    cast_value_exprs = [
        expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value_expr,
        )
        for value_expr in value_exprs
    ]
    iota_tensor_index_stmt = _codegen_iota_tensor_index_common_cute(
        cute_func,
        state,
        target,
        index,
        sem,
        cast_value_exprs,
        keyword_names,
    )
    if iota_tensor_index_stmt is not None:
        return iota_tensor_index_stmt
    ast_index = state.ast_args[1]
    assert isinstance(ast_index, (list, tuple))
    pointer = _cute_pointer_expr(state, target, index, ast_index)
    values_section = ", ".join(f"{k}={{{k}}}" for k in keyword_names)
    placeholders = dict(zip(keyword_names, cast_value_exprs, strict=True))
    atomic_expr = expr_from_string(
        f"cute.arch.{cute_func}({{ptr}}, {values_section}, sem={{sem}})",
        ptr=expr_from_string(pointer),
        sem=sem,
        **placeholders,
    )
    return _guard_cute_atomic_expr(state, index, target_dtype, atomic_expr)


def _guard_cute_atomic_expr(
    state: CodegenState,
    index: list[object],
    target_dtype: str,
    atomic_expr: ast.AST,
) -> ast.AST:
    predicates = [
        predicate
        for predicate in (
            _cute_active_mask_predicate(state),
            _cute_leader_thread_predicate(state, index),
        )
        if predicate is not None
    ]
    if not predicates:
        return atomic_expr
    predicate_expr = expr_from_string(" and ".join(predicates))
    assert isinstance(predicate_expr, ast.expr)
    assert isinstance(atomic_expr, ast.expr)
    if state.fx_node is not None and len(state.fx_node.users) == 0:
        state.codegen.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=predicate_expr,
                    body=[ast.Expr(value=atomic_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)

    result_var = state.device_function.new_var("_atomic_prev", dce=True)
    zero_value = expr_from_string(f"{target_dtype}(0)")
    assert isinstance(zero_value, ast.expr)
    state.codegen.add_statement(
        ast.fix_missing_locations(
            ast.Assign(
                targets=[ast.Name(id=result_var, ctx=ast.Store())],
                value=zero_value,
            )
        )
    )
    state.codegen.add_statement(
        ast.fix_missing_locations(
            ast.If(
                test=predicate_expr,
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=result_var, ctx=ast.Store())],
                        value=atomic_expr,
                    )
                ],
                orelse=[],
            )
        )
    )
    return expr_from_string(result_var)


def _cute_leader_thread_predicate(
    state: CodegenState,
    index: list[object],
) -> str | None:
    scalar_origin_block_ids: set[int] = set()
    for idx in index:
        if not isinstance(idx, torch.SymInt):
            continue
        expr = _symint_expr(idx)
        if expr is None:
            continue
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is None or not isinstance(origin_info.origin, GridOrigin):
            continue
        if type(origin_info.origin) is GridOrigin:
            continue
        scalar_origin_block_ids.add(origin_info.origin.block_id)
    if not scalar_origin_block_ids:
        return None

    axes: set[int] = set()
    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for block_id in scalar_origin_block_ids:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                axes.add(thread_axis)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            for block_id in scalar_origin_block_ids:
                thread_axis = loop_state.block_thread_axes.get(block_id)
                if thread_axis is not None:
                    axes.add(thread_axis)
    if not axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in sorted(axes)
    )


def _cute_active_mask_predicate(state: CodegenState) -> str | None:
    masks: list[str] = []
    seen_blocks: set[int] = set()

    for block_id, loops in state.codegen.active_device_loops.items():
        if block_id in seen_blocks or not loops:
            continue
        seen_blocks.add(block_id)
        mask_var = loops[-1].strategy.mask_var(block_id)
        if mask_var is not None:
            masks.append(f"({mask_var})")

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for block_id in grid_state.block_ids:
            if block_id in seen_blocks:
                continue
            seen_blocks.add(block_id)
            mask_var = grid_state.strategy.mask_var(block_id)
            if mask_var is not None:
                masks.append(f"({mask_var})")

    if not masks:
        return None
    return " and ".join(masks)


def _codegen_iota_tensor_index_common_cute(
    cute_func: str,
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    sem: ast.AST,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
) -> ast.AST | None:
    from .._compiler.compile_environment import CompileEnvironment
    from .memory_ops import _cute_active_index_var

    fx_node = state.fx_node
    if fx_node is None or len(index) != 1 or len(fx_node.args) < 2:
        return None
    tensor_index = index[0]
    if not isinstance(tensor_index, torch.Tensor) or tensor_index.ndim != 1:
        return None
    fx_index = fx_node.args[1]
    if not isinstance(fx_index, (list, tuple)) or len(fx_index) != 1:
        return None
    index_node = fx_index[0]
    if (
        not isinstance(index_node, torch.fx.Node)
        or index_node.target is not torch.ops.prims.iota.default
        or index_node.kwargs.get("start", 0) != 0
        or index_node.kwargs.get("step", 1) != 1
    ):
        return None

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor_index.shape[0])
    if block_id is None:
        return None
    block_id = env.resolve_codegen_block_id(block_id, state.codegen, fx_node.graph)
    index_var = _cute_active_index_var(state, block_id)
    if index_var is None:
        return None

    tensor_name = state.device_function.tensor_arg(target).name
    values_section = ", ".join(f"{k}={{{k}}}" for k in keyword_names)
    placeholders = dict(zip(keyword_names, value_exprs, strict=True))
    atomic_expr = expr_from_string(
        "cute.arch."
        + cute_func
        + "("
        + f"({tensor_name}.iterator + "
        + f"cute.crd2idx(({index_var},), {tensor_name}.layout)).llvm_ptr, "
        + values_section
        + ", sem={sem})",
        sem=sem,
        **placeholders,
    )
    target_dtype = env.backend.dtype_str(target.dtype)
    return _guard_cute_atomic_expr(state, index, target_dtype, atomic_expr)


def _pallas_atomic_load_prev(
    state: CodegenState,
) -> tuple[str, str, str]:
    """Load previous value for a Pallas atomic op.

    On TPU, each kernel instance has exclusive access to its tile, so
    atomics are implemented as regular load-compute-store sequences.

    Returns (tensor_name, index_str, prev_var_name).
    """
    from .._compiler.ast_extension import statement_from_string
    from .memory_ops import _pallas_index_str

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("pallas atomic")

    name = state.device_function.tensor_arg(target).name
    index_str, _ = _pallas_index_str(state, index, target)

    prev_var = state.device_function.new_var("_prev", dce=True)
    state.codegen.add_statement(
        statement_from_string(f"{prev_var} = {name}[{index_str}]")
    )
    return name, index_str, prev_var  # pyrefly: ignore[bad-return]


def _to_ast_values(values: list[object]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for v in values:
        if isinstance(v, (int, float, bool)):
            out.append(expr_from_string(constant_repr(v)))
        else:
            assert isinstance(v, ast.AST)
            out.append(v)
    return out


def _ref_atomic_binop(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Shared ref implementation for simple atomic binary ops (xchg/and/or/xor/max/min).

    Processes indices, clones the previous value, applies the op, and returns prev.
    For xchg, pass op=lambda old, val: val.
    """
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = op(target[idx_tuple], val)
    return prev


def _ref_apply(
    target: torch.Tensor,
    index: list[object],
    apply_fn: Callable[[torch.Tensor, tuple, object], None],
    value: object,
) -> None:
    from .ref_tile import RefTile

    # Convert indices to proper format
    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)

    # Find tensor indices that need element-wise processing
    tensor_indices = [
        (i, idx)
        for i, idx in enumerate(processed_index)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1
    ]

    if tensor_indices:
        # Element-wise processing for tensor indices (handle first tensor index)
        i, tensor_idx = tensor_indices[0]

        if tensor_idx.ndim == 0:
            coords_iter = [()]
        else:
            ranges = [range(dim) for dim in tensor_idx.shape]
            coords_iter = itertools.product(*ranges)

        for coords in coords_iter:
            elem = tensor_idx[coords].item()
            new_index = processed_index.copy()
            new_index[i] = int(elem)
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                next_value = value[coords]
            else:
                next_value = value
            _ref_apply(target, new_index, apply_fn, next_value)
    else:
        apply_fn(target, tuple(processed_index), value)


# -- atomic_add --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_add(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically add a value to a target tensor.

    Performs an atomic read-modify-write that adds ``value`` to
    ``target[index]``. This is safe for concurrent access from multiple
    threads/blocks.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to add (tensor or scalar).
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.

    Example:
        @helion.kernel
        def global_sum(x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_add(result, [0], x[tile].sum())
            return result

    Notes:
        - Use for race-free accumulation across parallel execution.
        - Higher memory semantics may reduce performance.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, torch.Tensor | float | int, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_add)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    # Convert indices for shape computation and fast path detection
    processed_index: list[object] = []
    has_tensor_index = False
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor):
            if idx.numel() == 1:
                processed_index.append(int(idx.item()))
            else:
                processed_index.append(idx)
                has_tensor_index = True
        else:
            processed_index.append(idx)

    def _convert_value_to_target_dtype(val: object) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            vt = val.to(device=target.device)
            if vt.dtype != target.dtype:
                vt = vt.to(dtype=target.dtype)
            return vt
        return torch.as_tensor(val, dtype=target.dtype, device=target.device)

    if has_tensor_index:
        ret_shape = SubscriptIndexing.compute_shape(target, processed_index)
        prev_chunks: list[torch.Tensor] = []

        def apply(t: torch.Tensor, idx_tuple: tuple, v: object) -> None:
            prev_val = t[idx_tuple].clone()
            val_tensor = _convert_value_to_target_dtype(v)
            t[idx_tuple] = t[idx_tuple] + val_tensor
            prev_chunks.append(prev_val.reshape(-1))

        _ref_apply(target, index, apply, value)
        if prev_chunks:
            flat_prev = torch.cat(prev_chunks)
        else:
            flat_prev = target.new_empty(0, dtype=target.dtype, device=target.device)
        return flat_prev.reshape(ret_shape)

    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val_tensor = _convert_value_to_target_dtype(value)
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = target[idx_tuple] + val_tensor
    return prev


@_decorators.codegen(atomic_add, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_add", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_add, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_add",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_add, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.compile_environment import CompileEnvironment

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    target = state.proxy_arg(0)
    assert isinstance(target, torch.Tensor)
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(target.dtype)
    # Cast the sum to the target dtype so the store doesn't fail when
    # the value dtype differs (e.g. float32 accumulator into bfloat16 ref).
    cast = backend.cast_expr(f"{prev_var} + {{value}}", target_dtype)
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {cast}", value=value_ast)
    )
    return expr_from_string(prev_var)


# -- atomic_xchg --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_xchg(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically exchange (set) a value at ``target[index]``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: New value(s) to set.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_xchg)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_xchg)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_xchg)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, lambda old, val: val)


@_decorators.codegen(atomic_xchg, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xchg", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xchg, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_exch",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_xchg, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {{value}}", value=value_ast)
    )
    return expr_from_string(prev_var)


# -- atomic_and/or/xor --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_and(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise AND with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to AND with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_and)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_and)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_and)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, torch.bitwise_and)


@_decorators.codegen(atomic_and, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_and", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_and, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_and",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_and, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} & {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_or(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise OR with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to OR with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_or)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_or)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_or)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, torch.bitwise_or)


@_decorators.codegen(atomic_or, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_or", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_or, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_or",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_or, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} | {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_xor(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise XOR with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to XOR with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_xor)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_xor)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_xor)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, torch.bitwise_xor)


@_decorators.codegen(atomic_xor, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xor", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xor, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_xor",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_xor, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} ^ {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


# -- atomic_max/min --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_max(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically update ``target[index]`` with the maximum of current value
    and ``value``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to compare with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_max)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_max)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_max)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, torch.maximum)


@_decorators.codegen(atomic_max, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_max", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_max, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_max",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_max, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.maximum({prev_var}, {{value}})",
            value=value_ast,
        )
    )
    return expr_from_string(prev_var)


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_min(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically update ``target[index]`` with the minimum of current value
    and ``value``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to compare with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
        ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_min)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_min)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_min)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, torch.minimum)


@_decorators.codegen(atomic_min, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_min", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_min, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_min",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_min, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.minimum({prev_var}, {{value}})",
            value=value_ast,
        )
    )
    return expr_from_string(prev_var)


# -- atomic_cas --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_cas(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor | float | bool,
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically compare-and-swap a value at ``target[index]``.

    If the current value equals ``expected``, writes ``value``. Otherwise
    leaves memory unchanged.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        expected: Expected current value(s) used for comparison.
        value: New value(s) to write if comparison succeeds.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the compare-and-swap.

    Note:
        Triton CAS doesn’t support a masked form; our generated code uses
        an unmasked CAS and relies on index masking to avoid OOB.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: object,
    value: object,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, object, object, str]:
    return _prepare_mem_args(target, index, expected, value, sem=sem)


@_decorators.register_fake(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor,
    value: torch.Tensor,
    sem: str = "relaxed",
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor | float | bool,
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    exp_t = (
        expected
        if isinstance(expected, torch.Tensor)
        else torch.as_tensor(expected, dtype=target.dtype, device=target.device)
    )
    val_t = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index]
    mask = target[idx_tuple] == exp_t
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = torch.where(mask, val_t, target[idx_tuple])
    return prev


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

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    return expr_from_string(
        f"tl.atomic_cas({name} + {{offset}}, {{exp}}, {{val}}, sem={{sem}})",
        offset=indices.index_expr,
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )


@_decorators.codegen(atomic_cas, "cute")
def _(state: CodegenState) -> ast.AST:
    exp_expr = state.ast_args[2]
    val_expr = state.ast_args[3]
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("atomic_cas")

    pointer = _cute_pointer_expr(state, target, index)
    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    return expr_from_string(
        "cute.arch.atomic_cas({ptr}, cmp={exp}, val={val}, sem={sem})",
        ptr=expr_from_string(pointer),
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )


@_decorators.codegen(atomic_cas, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string
    from .memory_ops import _pallas_index_str

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("pallas atomic_cas")

    name = state.device_function.tensor_arg(target).name
    index_str, _ = _pallas_index_str(state, index, target)

    prev_var = state.device_function.new_var("_prev", dce=True)
    state.codegen.add_statement(
        statement_from_string(f"{prev_var} = {name}[{index_str}]")
    )

    exp_ast, val_ast = _to_ast_values([state.ast_args[2], state.ast_args[3]])
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.where({prev_var} == {{exp}}, {{val}}, {prev_var})",
            exp=exp_ast,
            val=val_ast,
        )
    )
    return expr_from_string(prev_var)
