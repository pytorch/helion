"""CuTe-backend codegen for the atomic ops defined in ``helion.language.atomic_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``atomic_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language import _tracing_ops
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
from ..compile_environment import _symint_expr
from ..host_function import HostFunction
from ..variable_origin import GridOrigin

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _cute_pointer_expr(
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    ast_index: list[object] | tuple[object, ...] | None = None,
) -> str:
    from ...language.memory_ops import _cute_index_exprs

    index_exprs = _cute_index_exprs(state, index, ast_index)
    name = state.device_function.tensor_arg(target).name
    coord = (
        f"({index_exprs[0]},)"
        if len(index_exprs) == 1
        else f"({', '.join(index_exprs)})"
    )
    return f"({name}.iterator + cute.crd2idx({coord}, {name}.layout)).llvm_ptr"


def _resolve_cute_atomic_kwargs(cute_func: str, requested: list[str]) -> list[str]:
    """Map our intended ``cute.arch.<cute_func>`` kwarg names onto whatever
    the live signature actually exposes.

    Helion's emitted code refers to ``cute.arch.atomic_*`` parameters by
    name (``val``, ``cmp``). Some nvidia-cutlass-dsl wheels have shipped
    with these renamed (e.g. ``val`` -> ``value``); the old emission
    style then trips a ``TypeError`` deep inside CUTLASS at run time.
    Probe the signature at codegen time and rewrite the kwarg names to
    match what the live wrapper accepts. Falls back to the requested
    name when none of the rename candidates appears, so healthy installs
    are unaffected.
    """
    import inspect

    try:
        import cutlass.cute as cute  # type: ignore[import-not-found]
    except ImportError:
        return list(requested)
    func = getattr(getattr(cute, "arch", None), cute_func, None)
    if func is None:
        return list(requested)
    try:
        params = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return list(requested)
    rename_candidates: dict[str, tuple[str, ...]] = {
        "val": ("val", "value", "rhs", "src", "a"),
        "cmp": ("cmp", "compare", "expected", "exp"),
    }
    resolved: list[str] = []
    for name in requested:
        candidates = rename_candidates.get(name, (name,))
        chosen = next((c for c in candidates if c in params), name)
        resolved.append(chosen)
    return resolved


_CUTE_FLOAT_ATOMIC_HELPERS: dict[str, str] = {
    "atomic_max": "_cute_atomic_max_float32",
    "atomic_min": "_cute_atomic_min_float32",
}


def _cute_atomic_callee(cute_func: str, target_dtype_torch: torch.dtype) -> str:
    """Pick the callee for a CuTe atomic op.

    NVVM/PTX has no native ``atom.max``/``atom.min`` for floating point, so
    float ``atomic_max``/``atomic_min`` are routed through runtime helpers that
    emulate them with integer atomics (registered in
    ``CuteBackend.library_imports``). All other ops, and integer max/min, use
    the native ``cute.arch.<func>`` directly.
    """
    helper = _CUTE_FLOAT_ATOMIC_HELPERS.get(cute_func)
    if helper is None or not target_dtype_torch.is_floating_point:
        return f"cute.arch.{cute_func}"
    if target_dtype_torch is not torch.float32:
        raise exc.BackendUnsupported(
            "cute",
            f"{cute_func} on floating-point dtype {target_dtype_torch} "
            "(only float32 is supported)",
        )
    return helper


def _codegen_common_cute(
    cute_func: str,
    state: CodegenState,
    *,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
) -> ast.AST:
    from ..compile_environment import CompileEnvironment

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
    callee = _cute_atomic_callee(cute_func, target.dtype)
    cast_value_exprs = [
        expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value_expr,
        )
        for value_expr in value_exprs
    ]
    tensor_index_stmt = _codegen_tensor_index_common_cute(
        cute_func,
        state,
        target,
        index,
        sem,
        cast_value_exprs,
        keyword_names,
        callee,
    )
    if tensor_index_stmt is not None:
        return tensor_index_stmt
    ast_index = state.ast_args[1]
    assert isinstance(ast_index, (list, tuple))
    pointer = _cute_pointer_expr(state, target, index, ast_index)
    resolved_kwargs = _resolve_cute_atomic_kwargs(cute_func, keyword_names)
    values_section = ", ".join(
        f"{actual}={{{intent}}}"
        for intent, actual in zip(keyword_names, resolved_kwargs, strict=True)
    )
    placeholders = dict(zip(keyword_names, cast_value_exprs, strict=True))
    atomic_expr = expr_from_string(
        f"{callee}({{ptr}}, {values_section}, sem={{sem}})",
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
    *,
    extra_predicates: list[str] | None = None,
) -> ast.AST:
    predicates = [
        predicate
        for predicate in (
            _cute_active_mask_predicate(state),
            _cute_leader_thread_predicate(state, index),
            _cute_unindexed_axis_leader_predicate(state, index),
            *(extra_predicates or []),
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


def _cute_tensor_index_leader_predicate(
    state: CodegenState,
    tensor_index: torch.Tensor,
) -> str | None:
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor_index.shape[0])
    if block_id is None:
        return None
    assert state.fx_node is not None
    block_id = env.resolve_codegen_block_id(
        block_id, state.codegen, state.fx_node.graph
    )

    index_axes: set[int] = set()
    other_axes: set[int] = set()

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for candidate_block_id, thread_axis in grid_state.block_thread_axes.items():
            if candidate_block_id == block_id:
                index_axes.add(thread_axis)
            else:
                other_axes.add(thread_axis)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            for candidate_block_id, thread_axis in loop_state.block_thread_axes.items():
                if candidate_block_id == block_id:
                    index_axes.add(thread_axis)
                else:
                    other_axes.add(thread_axis)

    leader_axes = sorted(axis for axis in other_axes if axis not in index_axes)
    if not leader_axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in leader_axes
    )


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


def _cute_unindexed_axis_leader_predicate(
    state: CodegenState,
    index: list[object],
) -> str | None:
    """Return predicate restricting atomics to one thread per unindexed parallel axis.

    Two complementary mechanisms:

    1. **Active-axis leaders (block-size-index form):** When an atomic op
       is invoked inside ``hl.tile([m, n])`` but the index only covers a
       subset of the tile dimensions (e.g. ``hl.atomic_add(dy, [tile_i],
       reduced)`` after a reduction across ``tile_j``), every thread on
       the unindexed axis would otherwise re-issue the atomic with the
       same value. This mechanism uses ``BlockSizeOrigin`` symbols in the
       index to decide which currently-active thread axes are indexed,
       and restricts the rest to ``thread_idx[axis] == 0``.

    2. **Ghost-axis leaders (any index form):** A CTA-resident thread
       axis whose owning device loop has exited cannot be referenced by
       any current expression — including gather (``idxs[tile]``) or
       offset-constant (``[tile.begin]``) index forms whose index does
       not flow through a ``BlockSizeOrigin``. Threads on a ghost axis
       still exist (the CUDA ``blockDim`` is fixed for the kernel) and
       would re-issue the atomic with whatever value the surviving
       expression resolved to, multiplying the result by the ghost
       axis's size. Predicate the ghost axis to leader unconditionally.
       The canonical case is ``examples.matmul_split_k`` under
       autotune-picked configs that pack the inner-K loop onto a CTA
       thread axis.
    """
    from ..compile_environment import CompileEnvironment
    from ..variable_origin import BlockSizeOrigin

    env = CompileEnvironment.current()
    indexed_block_ids: set[int] = set()
    has_block_size_index = False
    for idx in index:
        if isinstance(idx, torch.Tensor):
            # A gather/scatter index tensor (e.g. ``output[idxs, tile_f]``)
            # covers the tile dimension it was loaded along: the per-thread
            # ``idxs`` value differs across that axis, so it is *indexed* and
            # must not be collapsed to a single leader thread.
            tensor_block_id = (
                env.resolve_block_id(idx.shape[0]) if idx.ndim >= 1 else None
            )
            if tensor_block_id is not None and state.fx_node is not None:
                has_block_size_index = True
                indexed_block_ids.add(
                    env.resolve_codegen_block_id(
                        tensor_block_id,
                        state.codegen,
                        state.fx_node.graph,
                    )
                )
            continue
        if not isinstance(idx, torch.SymInt):
            continue
        expr = _symint_expr(idx)
        if expr is None:
            continue
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            continue
        has_block_size_index = True
        assert state.fx_node is not None
        indexed_block_ids.add(
            env.resolve_codegen_block_id(
                origin_info.origin.block_id,
                state.codegen,
                state.fx_node.graph,
            )
        )

    fx_graph = state.fx_node.graph if state.fx_node is not None else None

    leader_axes: set[int] = set()
    active_thread_axes: set[int] = set()

    def collect(thread_axes: dict[int, int]) -> None:
        for candidate_block_id, thread_axis in thread_axes.items():
            active_thread_axes.add(thread_axis)
            if not has_block_size_index or fx_graph is None:
                # Without a block-size index there is no reliable mapping
                # from "block_id indexed by this atomic" to a thread axis,
                # so skip the active-axis mechanism. Ghost-axis predicates
                # below still fire.
                continue
            resolved = env.resolve_codegen_block_id(
                candidate_block_id, state.codegen, fx_graph
            )
            if resolved in indexed_block_ids:
                continue
            leader_axes.add(thread_axis)

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        collect(grid_state.block_thread_axes)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            collect(loop_state.block_thread_axes)

    # Ghost-axis leaders: any CTA-resident thread axis (size > 1) that
    # no active loop currently owns. ``max_thread_block_dims`` tracks the
    # per-axis CTA size accumulated as device loops are entered and is
    # never decremented on exit, so it correctly captures axes whose
    # owning loop has finished but whose threads remain live.
    for axis, size in enumerate(state.codegen.max_thread_block_dims):
        if size > 1 and axis not in active_thread_axes:
            leader_axes.add(axis)

    if not leader_axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in sorted(leader_axes)
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


def _resolve_tensor_index_iota_node(
    state: CodegenState, index_node: torch.fx.Node
) -> torch.fx.Node | None:
    from ..device_ir import NodeArgsGraphInfo

    current = index_node
    visited: set[torch.fx.Node] = set()
    while True:
        if current in visited:
            return None
        visited.add(current)
        if current.target is torch.ops.prims.iota.default:
            return current
        if current.op == "call_function" and current.target in {
            _tracing_ops._new_var,
            _tracing_ops._phi,
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
            torch.ops.prims.convert_element_type.default,
        }:
            arg = current.args[0] if current.args else None
            if not isinstance(arg, torch.fx.Node):
                return None
            current = arg
            continue
        if current.op != "placeholder":
            return None
        graph_infos = [
            graph_info
            for graph_info in state.codegen.codegen_graphs
            if graph_info.graph is current.graph
        ]
        if len(graph_infos) != 1:
            return None
        graph_info = graph_infos[0]
        if not isinstance(graph_info, NodeArgsGraphInfo):
            return None
        outer_node = graph_info.placeholder_to_outer_arg(current)
        if not isinstance(outer_node, torch.fx.Node):
            return None
        current = outer_node


def _codegen_tensor_index_common_cute(
    cute_func: str,
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    sem: ast.AST,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
    callee: str,
) -> ast.AST | None:
    from ...language.memory_ops import _cute_active_index_var
    from ..compile_environment import CompileEnvironment

    fx_node = state.fx_node
    if fx_node is None or len(index) != 1 or len(fx_node.args) < 2:
        return None
    tensor_index = index[0] if isinstance(index[0], torch.Tensor) else None
    fx_index = fx_node.args[1]
    if not isinstance(fx_index, (list, tuple)) or len(fx_index) != 1:
        return None
    index_node = fx_index[0]
    if not isinstance(index_node, torch.fx.Node):
        return None
    iota_node = _resolve_tensor_index_iota_node(state, index_node)
    if iota_node is None:
        return None
    iota_val = iota_node.meta.get("val")
    if isinstance(iota_val, torch.Tensor) and iota_val.ndim == 1:
        tensor_index = iota_val
    if tensor_index is None or tensor_index.ndim != 1:
        return None
    iota_start = iota_node.kwargs.get("start", 0)
    iota_step = iota_node.kwargs.get("step", 1)
    if iota_step != 1 or not isinstance(iota_start, int):
        return _codegen_tensor_index_loop_common_cute(
            cute_func,
            state,
            target,
            tensor_index,
            index_node,
            sem,
            value_exprs,
            keyword_names,
            callee,
        )

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor_index.shape[0])
    if block_id is None:
        return _codegen_tensor_index_loop_common_cute(
            cute_func,
            state,
            target,
            tensor_index,
            index_node,
            sem,
            value_exprs,
            keyword_names,
            callee,
        )
    block_id = env.resolve_codegen_block_id(block_id, state.codegen, fx_node.graph)
    if (index_var := _cute_active_index_var(state, block_id)) is None:
        return _codegen_tensor_index_loop_common_cute(
            cute_func,
            state,
            target,
            tensor_index,
            index_node,
            sem,
            value_exprs,
            keyword_names,
            callee,
        )

    tensor_name = state.device_function.tensor_arg(target).name
    resolved_kwargs = _resolve_cute_atomic_kwargs(cute_func, keyword_names)
    values_section = ", ".join(
        f"{actual}={{{intent}}}"
        for intent, actual in zip(keyword_names, resolved_kwargs, strict=True)
    )
    placeholders = dict(zip(keyword_names, value_exprs, strict=True))
    atomic_expr = expr_from_string(
        callee
        + "("
        + f"({tensor_name}.iterator + "
        + f"cute.crd2idx((cutlass.Int32({iota_start}) + {index_var},), {tensor_name}.layout)).llvm_ptr, "
        + values_section
        + ", sem={sem})",
        sem=sem,
        **placeholders,
    )
    target_dtype = env.backend.dtype_str(target.dtype)
    extra_predicates = [
        predicate
        for predicate in (_cute_tensor_index_leader_predicate(state, tensor_index),)
        if predicate is not None
    ]
    return _guard_cute_atomic_expr(
        state,
        index,
        target_dtype,
        atomic_expr,
        extra_predicates=extra_predicates,
    )


def _codegen_tensor_index_loop_common_cute(
    cute_func: str,
    state: CodegenState,
    target: torch.Tensor,
    tensor_index: torch.Tensor,
    index_node: torch.fx.Node,
    sem: ast.AST,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
    callee: str,
) -> ast.AST | None:
    from ..ast_extension import statement_from_string

    fx_node = state.fx_node
    if fx_node is None or len(fx_node.users) > 0:
        return None
    if tensor_index.ndim != 1:
        return None
    extent = tensor_index.shape[0]
    if not isinstance(extent, int):
        return None

    ast_index = state.ast_args[1]
    if not isinstance(ast_index, (list, tuple)) or len(ast_index) != 1:
        return None
    ast_index_expr = ast_index[0]
    if not isinstance(ast_index_expr, ast.AST):
        return None

    iota_node = _resolve_tensor_index_iota_node(state, index_node)
    indexed_values: list[ast.AST] = []
    value_arg_offset = 2
    for value_expr, _keyword_name in zip(value_exprs, keyword_names, strict=True):
        value_proxy = state.proxy_arg(value_arg_offset)
        value_arg_offset += 1
        if isinstance(value_proxy, torch.Tensor) and value_proxy.ndim == 1:
            tensor_arg = state.device_function.tensor_arg(value_proxy)
            indexed_values.append(
                expr_from_string(
                    "{value}[{idx}]",
                    value=expr_from_string(tensor_arg.name),
                    idx=expr_from_string("_tensor_index_i"),
                )
            )
            continue
        if extent != 1:
            return None
        indexed_values.append(value_expr)

    if iota_node is not None:
        start = iota_node.kwargs.get("start", 0)
        step = iota_node.kwargs.get("step", 1)
        if not isinstance(start, int) or not isinstance(step, int):
            return None
        index_expr = expr_from_string(
            f"cutlass.Int32({start}) + cutlass.Int32({step}) * cutlass.Int32(_tensor_index_i)"
        )
    else:
        index_expr = expr_from_string(
            "cutlass.Int32({index}[{idx}])",
            index=ast_index_expr,
            idx=expr_from_string("_tensor_index_i"),
        )

    tensor_name = state.device_function.tensor_arg(target).name
    resolved_kwargs = _resolve_cute_atomic_kwargs(cute_func, keyword_names)
    values_section = ", ".join(
        f"{actual}={{{intent}}}"
        for intent, actual in zip(keyword_names, resolved_kwargs, strict=True)
    )
    placeholders = dict(zip(keyword_names, indexed_values, strict=True))
    atomic_expr = expr_from_string(
        callee
        + "("
        + f"({tensor_name}.iterator + "
        + f"cute.crd2idx(({{index}},), {tensor_name}.layout)).llvm_ptr, "
        + values_section
        + ", sem={sem})",
        index=index_expr,
        sem=sem,
        **placeholders,
    )
    assert isinstance(atomic_expr, ast.expr)
    predicate_terms = [
        predicate
        for predicate in (
            _cute_active_mask_predicate(state),
            _cute_tensor_index_leader_predicate(state, tensor_index),
        )
        if predicate is not None
    ]
    predicate_expr = (
        ast.parse(" and ".join(predicate_terms), mode="eval").body
        if predicate_terms
        else None
    )
    inner = (
        ast.fix_missing_locations(
            ast.If(
                test=predicate_expr,
                body=[ast.Expr(value=atomic_expr)],
                orelse=[],
            )
        )
        if predicate_expr is not None
        else ast.Expr(value=atomic_expr)
    )
    loop = statement_from_string(f"for _tensor_index_i in range({extent}):\n    pass")
    assert isinstance(loop, ast.For)
    loop.body = [inner]
    state.codegen.add_statement(loop)
    return ast.Constant(value=None)


@_decorators.codegen(atomic_add, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_add",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_xchg, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_exch",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_and, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_and",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_or, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_or",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_xor, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_xor",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_max, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_max",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_min, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_min",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
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
    cmp_kw, val_kw = _resolve_cute_atomic_kwargs("atomic_cas", ["cmp", "val"])
    return expr_from_string(
        f"cute.arch.atomic_cas({{ptr}}, {cmp_kw}={{exp}}, {val_kw}={{val}}, sem={{sem}})",
        ptr=expr_from_string(pointer),
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )
