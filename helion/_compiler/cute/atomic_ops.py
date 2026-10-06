"""CuTe-backend codegen for the atomic ops defined in ``helion.language.atomic_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``atomic_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree

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
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_ATOMIC_UNIFORM_LANES_ATTR
from ..ast_read_writes import ReadWrites
from ..compile_environment import _symint_expr
from ..host_function import HostFunction
from ..variable_origin import GridOrigin
from ..variable_origin import NameOrigin

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..inductor_lowering import CodegenState

# Lane counts of the ``red.global.add.v{N}.f32`` forms (8 and 16 bytes).
_CUTE_VECTOR_ATOMIC_WIDTHS = (2, 4)
# ``red.global.add.v{2,4}.f32`` is an sm_90+ PTX form.
_CUTE_VECTOR_ATOMIC_MIN_CAPABILITY = (9, 0)


def _cute_pointer_expr(
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    ast_index: list[object] | tuple[object, ...] | None = None,
) -> str:
    from ...language.memory_ops import _cute_index_exprs

    index_exprs = _cute_index_exprs(state, index, ast_index, tensor=target)
    name = state.device_function.tensor_arg(target).name
    return _cute_coord_pointer_expr(name, index_exprs)


def _cute_coord_pointer_expr(name: str, index_exprs: list[str]) -> str:
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
    extra_predicates: list[str] = []
    relaxed = state.proxy_arg(len(state.ast_args) - 1) == "relaxed"
    # The single-tensor-index forms below index their values per element and
    # keep their own guards; the elision applies to the relaxed pointer form
    # only (a release / acquire RMW is a fence-carrying write in the cell's
    # modification order and is never skipped).
    elision = (
        _cute_where_zero_elision(state, target)
        if cute_func == "atomic_add" and relaxed and len(index) > 1
        else None
    )
    if elision is not None:
        # ``atomic_add(t, i, where(c, x, 0))``: add ``x`` under ``c`` and skip
        # the exact zeros (see ``_cute_where_zero_elision`` for the proof).
        predicate, kept = elision
        extra_predicates.append(predicate)
        cast_value_exprs = [
            expr_from_string(
                backend.ast_to_dtype_expr("{value}", target_dtype), value=kept
            )
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
        extra_predicates=extra_predicates,
    )
    if tensor_index_stmt is not None:
        return tensor_index_stmt
    from ...language.memory_ops import _cute_index_exprs

    ast_index = state.ast_args[1]
    assert isinstance(ast_index, (list, tuple))
    index_exprs = _cute_index_exprs(state, index, ast_index, tensor=target)
    tensor_name = state.device_function.tensor_arg(target).name
    pointer = _cute_coord_pointer_expr(tensor_name, index_exprs)
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
    if (
        cute_func == "atomic_add"
        and relaxed
        and _cute_vector_atomic_site(
            state,
            target,
            index,
            index_exprs,
            cast_value_exprs[0],
            atomic_expr,
            _cute_atomic_predicates(state, index, extra_predicates, atomic_expr)[0],
        )
    ):
        return ast.Constant(value=None)
    return _guard_cute_atomic_expr(
        state, index, target_dtype, atomic_expr, extra_predicates=extra_predicates
    )


def _cute_atomic_predicates(
    state: CodegenState,
    index: list[object],
    extra_predicates: list[str] | None,
    atomic_expr: ast.AST | None = None,
) -> tuple[list[str], set[int]]:
    """The guards of an atomic at this site (bounds masks of the covered
    axes, leader threads of the uncovered ones, and any caller-supplied
    predicate) and the leader axes; records the lanes the atomic is uniform
    along on ``atomic_expr`` for the lane-loop placement."""
    indexed_block_ids = _cute_atomic_indexed_blocks(state, index)
    leader_axes = _cute_unindexed_leader_axes(state, indexed_block_ids)
    if indexed_block_ids is None:
        # Without a known coverage, a tile attribute in the index
        # (``tile.begin``) still names the thread axis of its block as one
        # the atomic does not vary along.
        leader_axes |= _cute_leader_thread_axes(state, index)
    if atomic_expr is not None:
        setattr(
            atomic_expr,
            HELION_ATOMIC_UNIFORM_LANES_ATTR,
            _cute_uniform_lane_vars(state, indexed_block_ids),
        )
    return [
        predicate
        for predicate in (
            _cute_active_mask_predicate(state, indexed_block_ids),
            _cute_leader_predicate(leader_axes),
            *(extra_predicates or []),
        )
        if predicate is not None
    ], leader_axes


def _cute_literal_positive_zero(node: object) -> bool:
    """Whether ``node`` is a literal ``+0`` operand of a ``where``: a Python
    ``0`` / ``0.0`` (not ``-0.0``, not a bool) or a ``scalar_tensor`` of one.
    A computed zero never qualifies."""
    from torch.fx.node import Node

    value: object = node
    if isinstance(node, Node):
        if node.target is not torch.ops.aten.scalar_tensor.default or not node.args:
            return False
        value = node.args[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == 0 and math.copysign(1.0, float(value)) > 0


def _cute_where_zero_elision(
    state: CodegenState, target: torch.Tensor
) -> tuple[str, ast.AST] | None:
    """``(predicate, x)`` when the relaxed ``atomic_add(t, i, where(c, x, 0))``
    may skip the zero lanes and add ``x`` under ``c`` alone; ``None`` fails
    closed.  The caller admits relaxed atomics only.

    (a) The ``where``'s other operand is the literal ``+0`` (a Python zero or
        a ``scalar_tensor`` of one); ``-0.0`` and computed zeros never
        qualify.
    (b) The atomic's old value is unused.
    (c) ``x`` already has the ``where``'s dtype and both ``where`` operands
        are generated names (``state.env``), so the predicate and the addend
        can be spelled at the site.
    (d) Integer target: nothing else.  Adding ``0`` to an integer is exact,
        so the elision is always on.
    (e) Floating target: only under the ``fast_math`` setting, and only into
        a private zero-filled sum
        (``_cute_atomic_target_is_private_zero_filled_sum``).  There is no
        bit-exact proof for floats: ``atom/red.global.add.f32`` flushes
        subnormal inputs and results to a signed zero and canonicalises NaNs
        (measured on sm_100; the Triton backend emits the same op), so adding
        ``+0.0`` rewrites a ``-0.0`` cell to ``+0.0``, a subnormal cell to a
        signed zero and a NaN payload to the canonical NaN, and a zeros-filled
        cell written only by atomic adds CAN hold ``-0.0`` (a negative result
        that underflows is flushed to ``-0.0``).  The private-sum proof buys
        admissibility instead: the target starts at ``+0.0`` and every write
        is an atomic add.  Across threads the interleaving of a kernel's
        atomics is unspecified, so scheduling every skipped ``+0.0`` add
        first (``+0.0 + +0.0 = +0.0``, exact) shows the elided result is one
        the unelided kernel may produce.  Within one thread the skipped add
        stays program-ordered after that thread's earlier adds to the cell
        (PTX orders one thread's accesses to one location), so when those
        adds have underflowed to ``-0.0`` the unelided kernel flushes the
        cell to ``+0.0`` and the elided one leaves ``-0.0``.  That sign of a
        zero reached by underflow is the residual, and it is what
        ``fast_math`` accepts.

    The non-zero adds and their interleaving are unchanged.  When the atomic
    is the ``where``'s only user the dead select is removed from the body.
    """
    from torch.fx.node import Node

    from ..compile_environment import CompileEnvironment

    fx_node = state.fx_node
    if fx_node is None or len(fx_node.args) < 3 or len(fx_node.users) != 0:
        return None
    value_node = fx_node.args[2]
    if (
        not isinstance(value_node, Node)
        or value_node.target is not torch.ops.aten.where.self
        or len(value_node.args) != 3
    ):
        return None
    condition, on_true, on_false = value_node.args
    if not isinstance(condition, Node):
        return None
    kept, negate = on_true, False
    if not _cute_literal_positive_zero(on_false):
        kept, negate = on_false, True
        if not _cute_literal_positive_zero(on_true):
            return None
    if not isinstance(kept, Node):
        return None
    if target.dtype is torch.bool or target.dtype.is_complex:
        return None
    where_value = value_node.meta.get("val")
    kept_value = kept.meta.get("val")
    if (
        not isinstance(where_value, torch.Tensor)
        or not isinstance(kept_value, torch.Tensor)
        or where_value.dtype != kept_value.dtype
    ):
        return None
    if target.dtype.is_floating_point and not (
        CompileEnvironment.current().settings.fast_math
        and _cute_atomic_target_is_private_zero_filled_sum(state, target)
    ):
        return None
    condition_ast = state.env.get(condition)
    kept_ast = state.env.get(kept)
    if not isinstance(condition_ast, ast.AST) or not isinstance(kept_ast, ast.AST):
        return None
    if len(value_node.users) == 1:
        state.codegen.remove_statements_owned_by_nodes((value_node,))
    predicate = ast.unparse(condition_ast)
    if negate:
        predicate = f"not ({predicate})"
    return predicate, kept_ast


def _cute_atomic_target_is_private_zero_filled_sum(
    state: CodegenState, target: torch.Tensor
) -> bool:
    """Whether ``target`` is a private sum that starts at ``+0.0`` and is only
    ever written by atomic adds while the kernel runs.

    The storage is a wrapper allocation filled with ``+0``
    (``CompileEnvironment.tensor_storage_is_zero_filled_allocation``), its
    host binding is private until the launch (``private_fresh_host_bindings``:
    bound once, never aliased, only metadata reads, the atomic destination and
    the return), and every device use of that storage is a load or the
    destination of an ``atomic_add``.  This is the premise of the
    admissible-interleaving argument in ``_cute_where_zero_elision``; it does
    not (and under flush-to-zero atomics cannot) exclude ``-0.0`` cells.
    """
    from ...language.memory_ops import load as language_load
    from ..compile_environment import CompileEnvironment
    from .atomic_output_promotions import private_fresh_host_bindings

    env = CompileEnvironment.current()
    if not env.tensor_storage_is_zero_filled_allocation(target):
        return False
    host_fn = HostFunction.current()
    origin = host_fn.tensor_to_origin.get(target)
    if type(origin) is not NameOrigin:
        return False
    if origin.name not in private_fresh_host_bindings(host_fn.body, {origin.name}):
        return False
    storages = {target.untyped_storage()}
    for graph_info in host_fn.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op != "call_function"
                or node.target is not _tracing_ops._host_tensor
            ):
                continue
            value = node.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or value.untyped_storage() not in storages
            ):
                continue
            for user in node.users:
                if (
                    user.op != "call_function"
                    or not user.args
                    or user.args[0] is not node
                ):
                    return False
                if user.target is language_load:
                    continue
                if user.target is atomic_add and not any(
                    leaf is node for leaf in pytree.tree_leaves(user.args[1:])
                ):
                    continue
                return False
    return True


def _cute_lane_varying_names(
    statements: Iterable[ast.AST], seeds: Iterable[str]
) -> set[str]:
    """Names whose values may differ between the lanes of a constexpr vector
    loop: ``seeds`` (its lane variable, the per-element index) and everything
    defined from them, transitively through the given statements."""
    varying = set(seeds)
    analysed = [ReadWrites.from_ast(stmt) for stmt in statements]
    changed = True
    while changed:
        changed = False
        for rw in analysed:
            if set(rw.reads) & varying and not set(rw.writes) <= varying:
                varying |= set(rw.writes)
                changed = True
    return varying


def _cute_vector_atomic_site(
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    index_exprs: list[str],
    value_expr: ast.AST,
    atomic_expr: ast.AST,
    predicates: list[str],
) -> bool:
    """Defer a per-lane fp32 ``atomic_add`` along a vectorized tile axis to one
    ``red.global.add.v{V}.f32`` after the constexpr V-loop.

    The site follows the vector store protocol
    (``_cute_register_tile_unroll_vec_store``): the scalar atomic is emitted
    now and, when ``DeviceGridState.wrap_body`` confirms the V-loop is the
    innermost live lane loop with no barrier in the body, replaced by an
    append onto a trace-time list that one flush after the V-loop reduces.
    Admitted only on sm_90+ targets (the PTX form's minimum; unknown or lower
    capabilities keep the scalar atomic), when the atomic's old value is
    unused, the lane axis is the target's stride-1 dim indexed by the plain
    per-element index of a grid block with a vector partition, the packet is
    naturally aligned (``cute_reduction_vector_layout_aligned`` plus a
    zero-origin tile whose extent is a multiple of ``V``), and every other
    coordinate and every guard is uniform across the V lanes.  Returns whether
    the site was deferred (the caller then emits nothing else).
    """
    from ...language.memory_ops import _cute_active_index_var
    from ...language.memory_ops import _cute_lane_vloop_insert_pos
    from ..compile_environment import CompileEnvironment
    from ..tile_strategy import DeviceGridState
    from .memory_ops import _cute_defer_grid_vector_op
    from .memory_ops import _cute_lane_strategy
    from .memory_ops import _cute_tile_axis_block_id
    from .memory_ops import cute_reduction_vector_layout_aligned

    fx_node = state.fx_node
    if (
        fx_node is None
        or len(fx_node.users) != 0
        or target.dtype is not torch.float32
        or len(index) != target.ndim
        or len(index_exprs) != target.ndim
        or "None" in index_exprs
    ):
        return False
    # Fail closed to the scalar atomic when the target capability is unknown
    # or below sm_90.
    capability = CompileEnvironment.current().config_spec.target_device_capability
    if capability is None or capability < _CUTE_VECTOR_ATOMIC_MIN_CAPABILITY:
        return False
    # An atomic that is uniform along a live lane loop must stay a scalar
    # atomic: the lane-loop placement pass pins that form to the loop's
    # first lane (or rejects it), whereas the flush protocol below would
    # re-issue the reduction once per iteration of that loop.
    if _cute_uniform_lane_vars(state, _cute_atomic_indexed_blocks(state, index)):
        return False
    lane_axes = [
        (pos, block_id)
        for pos, idx in enumerate(index)
        if isinstance(idx, torch.SymInt)
        and (block_id := _cute_tile_axis_block_id(idx)) is not None
        and target.stride(pos) == 1
    ]
    if len(lane_axes) != 1:
        return False
    pos, block_id = lane_axes[0]
    strategy = _cute_lane_strategy(state, block_id)
    vec_by_block = getattr(strategy, "_cute_lane_vec_width_by_block", None)
    if not isinstance(vec_by_block, dict):
        return False
    vec_width = vec_by_block.get(block_id, 1)
    if vec_width not in _CUTE_VECTOR_ATOMIC_WIDTHS:
        return False
    base_index_var = getattr(strategy, "_cute_lane_base_index_var_by_block", {}).get(
        block_id
    )
    lane_body = getattr(strategy, "_cute_lane_body_by_block", {}).get(block_id)
    vec_lane_var = getattr(strategy, "_cute_vec_lane_var_by_block", {}).get(block_id)
    vloop = getattr(strategy, "_cute_lane_vloop_by_block", {}).get(block_id)
    if (
        not isinstance(base_index_var, str)
        or not isinstance(lane_body, list)
        or not isinstance(vec_lane_var, str)
        or not isinstance(vloop, ast.For)
    ):
        return False
    if index_exprs[pos] != _cute_active_index_var(state, block_id):
        return False
    loops = state.codegen.active_device_loops.get(block_id)
    owner = loops[-1] if loops else None
    stack = state.codegen.statements_stack
    if (
        not isinstance(owner, DeviceGridState)
        or len(stack) < 2
        or stack[-2] is not owner.hoist_parent_statements
        or not any(
            wrapper.vloop is vloop for wrapper in owner.vec_lane_wrappers.values()
        )
    ):
        return False
    fact = state.device_function.cute_state.vloop_sink_wrappers.get(vec_lane_var)
    if fact is None or fact.block_id != block_id or not fact.uniform_vector_mask:
        return False
    env = CompileEnvironment.current()
    if not cute_reduction_vector_layout_aligned(env, target, pos, vec_width):
        return False
    varying = _cute_lane_varying_names(
        [*owner.lane_setup_statements, *stack[-1]], {vec_lane_var, fact.index_var}
    )
    # The block's own bounds mask is uniform across the chunk (proven above).
    varying.discard(fact.mask_var or "")
    uniform_texts = [
        expr for axis, expr in enumerate(index_exprs) if axis != pos
    ] + predicates
    for text in uniform_texts:
        if set(ReadWrites.from_ast(ast.parse(text, mode="eval")).reads) & varying:
            return False
    tensor_name = state.device_function.tensor_arg(target).name
    base_exprs = list(index_exprs)
    base_exprs[pos] = base_index_var
    base_pointer = _cute_coord_pointer_expr(tensor_name, base_exprs)
    sites_by_block = getattr(strategy, "_cute_lane_vec_stores_by_block", None)
    if sites_by_block is None:
        sites_by_block = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_stores_by_block = sites_by_block
    guard = " and ".join(predicates)
    body = stack[-1]

    def emit() -> ast.AST | None:
        from .lane_loop_distribution import _tensor_mentions

        # The flush runs after the whole V-loop: a later statement of the
        # body that touches the target would see the atomic out of order.
        site = next((i for i, stmt in enumerate(body) if stmt is scalar), len(body))
        if any(tensor_name in _tensor_mentions(stmt) for stmt in body[site + 1 :]):
            return None
        # Shares the store flush numbering so flushes keep source order.
        sites = sites_by_block.setdefault(block_id, [])
        site_index = len(sites)
        list_var = state.device_function.new_var(
            f"_tile_atomic_vals_{block_id}_{site_index}", dce=False
        )
        sites.append(list_var)
        lane_body.insert(
            _cute_lane_vloop_insert_pos(strategy, block_id, lane_body),
            statement_from_string(f"{list_var} = []"),
        )
        flush = f"_cute_red_add_f32_vec({base_pointer}, {list_var})"
        if guard:
            flush = f"if {guard}:\n    {flush}"
        lane_body.insert(
            _cute_lane_vloop_insert_pos(strategy, block_id, lane_body) + 1 + site_index,
            statement_from_string(flush),
        )
        return statement_from_string(f"{list_var}.append({{value}})", value=value_expr)

    assert isinstance(atomic_expr, ast.expr)
    scalar: ast.stmt = ast.Expr(value=atomic_expr)
    if guard:
        guard_expr = expr_from_string(guard)
        assert isinstance(guard_expr, ast.expr)
        scalar = ast.If(test=guard_expr, body=[scalar], orelse=[])
    scalar = ast.fix_missing_locations(scalar)
    if not _cute_defer_grid_vector_op(state, strategy, block_id, scalar, emit):
        return False
    state.codegen.add_statement(scalar)
    return True


def _guard_cute_atomic_expr(
    state: CodegenState,
    index: list[object],
    target_dtype: str,
    atomic_expr: ast.AST,
    *,
    extra_predicates: list[str] | None = None,
) -> ast.AST:
    predicates, leader_axes = _cute_atomic_predicates(
        state, index, extra_predicates, atomic_expr
    )
    if not predicates:
        return atomic_expr
    if (
        (leader_axes or extra_predicates)
        and state.fx_node is not None
        and len(state.fx_node.users) > 0
    ):
        # Only the leader thread of a collapsed axis performs the atomic, so
        # only it holds the previous value; the other threads' elements would
        # consume the zero placeholder below.
        raise exc.BackendUnsupported(
            "cute",
            "the result of an atomic issued by one leader thread is not shared "
            "with the other threads of its tile axis",
        )
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


def _cute_leader_thread_axes(state: CodegenState, index: list[object]) -> set[int]:
    """Thread axes of the tile attributes (``tile.begin``, ``tile.id``) in ``index``."""
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
    axes: set[int] = set()
    if not scalar_origin_block_ids:
        return axes

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
    return axes


def _cute_uniform_index_value_blocks(
    state: CodegenState, index: list[object]
) -> set[int] | None:
    """Tile axes a tile-uniform atomic varies along, or None when not uniform.

    A constant index, or one made of tile attributes (``tile.begin``,
    ``tile.id``) and 0-d tensors (a scalar the body loaded or reduced,
    ``idx[tile.begin]``), addresses one element for the whole tile.  Such an atomic
    varies only along the tile axes of a tensor update value
    (``hl.atomic_add(total, [0], x[tile])``): along those axes every element
    is applied to that address; along the others the leader thread issues it
    once, outside the axis's lane loop or pinned to that loop's first lane
    (:func:`_cute_uniform_lane_vars`).  ``None`` when the index has any other
    component or a value's shape is not made of tile axes and ones.
    """
    from ..compile_environment import CompileEnvironment

    host_function = HostFunction.current()
    for idx in index:
        if idx is None or isinstance(idx, int):
            continue
        if isinstance(idx, torch.Tensor) and idx.ndim == 0:
            continue
        if isinstance(idx, torch.SymInt):
            expr = _symint_expr(idx)
            origin_info = (
                host_function.expr_to_origin.get(expr) if expr is not None else None
            )
            origin = origin_info.origin if origin_info is not None else None
            if isinstance(origin, GridOrigin) and type(origin) is not GridOrigin:
                continue
        return None
    fx_node = state.fx_node
    if fx_node is None:
        return None
    env = CompileEnvironment.current()
    block_ids: set[int] = set()
    for position in _cute_atomic_value_positions(state):
        value = state.proxy_arg(position)
        if isinstance(value, (bool, int, float)):
            continue
        if not isinstance(value, torch.Tensor):
            return None
        for size in value.shape:
            if isinstance(size, int):
                if size == 1:
                    continue
                return None
            expr = _symint_expr(size)
            if expr is None or not expr.free_symbols:
                return None
            for symbol in expr.free_symbols:
                block_id = env.get_block_id(symbol)
                if block_id is None:
                    return None
                block_ids.add(
                    env.resolve_codegen_block_id(block_id, state.codegen, fx_node.graph)
                )
    return block_ids


def _cute_atomic_value_positions(state: CodegenState) -> tuple[int, ...]:
    assert state.fx_node is not None
    return (2, 3) if state.fx_node.target is atomic_cas else (2,)


def _cute_atomic_indexed_blocks(
    state: CodegenState,
    index: list[object],
) -> set[int] | None:
    """Resolved ids of the tile axes the atomic varies along; None when unknown.

    A ``BlockSizeOrigin`` index covers its tile axis, a gather index the axis
    it was loaded along and a slice the axis the pointer lowering assigns it.
    A tile-uniform index (a constant, a tile attribute, a 0-d tensor) covers
    nothing by itself; the atomic then varies along the tile axes of its
    update value (:func:`_cute_uniform_index_value_blocks`).  ``None`` when
    no component has a known coverage: the atomic then runs on every thread,
    as it always has.
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

    if any(isinstance(idx, slice) for idx in index):
        from ...language.memory_ops import _cute_resolve_active_slice_block_id
        from ..utils import compute_slice_size

        target = state.proxy_arg(0)
        assert isinstance(target, torch.Tensor)
        assert state.fx_node is not None
        assert fx_graph is not None
        # Match pointer lowering's choice of slice axis, including equal-sized
        # tile dimensions. A slice indexes that axis; it is not a broadcast
        # atomic that should run only on the axis's leader thread.
        used_block_ids = {
            block_id
            for idx in index
            if isinstance(idx, torch.SymInt)
            if (block_id := env.get_block_id(idx)) is not None
        }
        tensor_dim = 0
        for idx in index:
            if idx is None:
                continue
            if isinstance(idx, slice) and idx.step in (None, 1):
                size = compute_slice_size(idx, target.shape[tensor_dim])
                block_id = _cute_resolve_active_slice_block_id(
                    state, size, used_block_ids
                )
                if block_id is not None:
                    used_block_ids.add(block_id)
                    indexed_block_ids.add(
                        env.resolve_codegen_block_id(block_id, state.codegen, fx_graph)
                    )
                    has_block_size_index = True
            tensor_dim += 1

        for position in _cute_atomic_value_positions(state):
            value = state.proxy_arg(position)
            if not isinstance(value, torch.Tensor):
                continue
            # A full slice can introduce a new persistent axis even when an
            # equally sized explicit tile already supplies the update value.
            # Until those coordinate systems can be remapped, do not collapse
            # the value's distinct axis to its leader and replicate one lane.
            for size in value.shape:
                if not isinstance(size, torch.SymInt):
                    continue
                expr = _symint_expr(size)
                if expr is None:
                    continue
                for symbol in expr.free_symbols:
                    value_block = env.get_block_id(symbol)
                    if (
                        value_block is not None
                        and env.resolve_codegen_block_id(
                            value_block, state.codegen, fx_graph
                        )
                        not in indexed_block_ids
                    ):
                        raise exc.BackendUnsupported(
                            "cute",
                            "atomic slice and update value use distinct tile axes",
                        )

    uniform_value_blocks = _cute_uniform_index_value_blocks(state, index)
    if uniform_value_blocks is not None:
        has_block_size_index = True
        indexed_block_ids |= uniform_value_blocks
    if not has_block_size_index:
        return None
    return indexed_block_ids


def _cute_unindexed_leader_axes(
    state: CodegenState,
    indexed_block_ids: set[int] | None,
) -> set[int]:
    """Return the thread axes whose leader alone issues the atomic.

    Two complementary mechanisms:

    1. **Active-axis leaders (known coverage):** When an atomic op is
       invoked inside ``hl.tile([m, n])`` but varies along only a subset of
       the tile dimensions (e.g. ``hl.atomic_add(dy, [tile_i], reduced)``
       after a reduction across ``tile_j``, or ``hl.atomic_add(count, [0],
       1)``), every thread on an uncovered axis would otherwise re-issue the
       atomic with the same value.  This mechanism uses the covered tile axes
       (:func:`_cute_atomic_indexed_blocks`) to decide which currently-active
       thread axes the atomic varies along, and restricts the rest to
       ``thread_idx[axis] == 0``.

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

    env = CompileEnvironment.current()
    fx_graph = state.fx_node.graph if state.fx_node is not None else None
    leader_axes: set[int] = set()
    active_thread_axes: set[int] = set()

    def collect(thread_axes: dict[int, int]) -> None:
        for candidate_block_id, thread_axis in thread_axes.items():
            active_thread_axes.add(thread_axis)
            if indexed_block_ids is None or fx_graph is None:
                # Without a known coverage there is no reliable mapping from
                # "tile axis the atomic varies along" to a thread axis, so
                # skip the active-axis mechanism. Ghost-axis predicates
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
    return leader_axes


def _cute_leader_predicate(axes: set[int]) -> str | None:
    if not axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in sorted(axes)
    )


def _cute_uniform_lane_vars(
    state: CodegenState, indexed_block_ids: set[int] | None
) -> frozenset[str]:
    """The lane loops (by lane variable) along whose tile axes the atomic is uniform.

    The leader thread of an uncovered axis issues the atomic, but a lane loop
    walks that thread through several elements of the axis, and the generated
    statement may still depend on the loop: a partial tile's mask guards a
    tensor value's load, or the loop structure nests the atomic's own loop
    inside it.  Recorded on the atomic call for the lane-loop distribution of
    the grid body and the full-nest check of an enclosing device loop's body
    (``cute/lane_loop_distribution.py``), which pin the atomic to the loop's
    first lane, where the thread holds the tile's first element along the
    axis under every lane layout, instead of repeating it once per iteration.
    Empty when the coverage is unknown: the atomic then runs on every thread
    under every mask, as it always has.
    """
    from ..compile_environment import CompileEnvironment
    from ..tile_strategy import DeviceLoopState

    if indexed_block_ids is None:
        return frozenset()
    lane_loop_block_ids: dict[str, frozenset[int]] = {}
    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        lane_loop_block_ids.update(grid_state.lane_loop_block_ids)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            if isinstance(loop_state, DeviceLoopState):
                lane_loop_block_ids.update(loop_state.lane_loop_block_ids)
    env = CompileEnvironment.current()
    fx_graph = state.fx_node.graph if state.fx_node is not None else None
    return frozenset(
        lane_var
        for lane_var, block_ids in lane_loop_block_ids.items()
        if not any(
            block_id >= 0
            and env.resolve_codegen_block_id(block_id, state.codegen, fx_graph)
            in indexed_block_ids
            for block_id in block_ids
        )
    )


def _cute_active_mask_predicate(
    state: CodegenState, indexed_block_ids: set[int] | None = None
) -> str | None:
    """The tile masks the atomic is subject to.

    With a known coverage only the covered axes' masks apply: the atomic
    runs on the leader thread of every other axis, whose element is the
    tile's first along that axis and always in range, and a mask of an
    uncovered lane-looped axis would tie the atomic to that loop.
    """
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    fx_graph = state.fx_node.graph if state.fx_node is not None else None

    def covered(block_id: int) -> bool:
        return (
            indexed_block_ids is None
            or env.resolve_codegen_block_id(block_id, state.codegen, fx_graph)
            in indexed_block_ids
        )

    masks: list[str] = []
    seen_blocks: set[int] = set()

    for block_id, loops in state.codegen.active_device_loops.items():
        if block_id in seen_blocks or not loops:
            continue
        seen_blocks.add(block_id)
        if not covered(block_id):
            continue
        mask_var = loops[-1].strategy.mask_var(block_id)
        if mask_var is not None:
            masks.append(f"({mask_var})")

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for block_id in grid_state.block_ids:
            if block_id in seen_blocks:
                continue
            seen_blocks.add(block_id)
            if not covered(block_id):
                continue
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
    *,
    extra_predicates: list[str] | None = None,
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
    site_predicates = [
        predicate
        for predicate in (
            _cute_tensor_index_leader_predicate(state, tensor_index),
            *(extra_predicates or []),
        )
        if predicate is not None
    ]
    return _guard_cute_atomic_expr(
        state,
        index,
        target_dtype,
        atomic_expr,
        extra_predicates=site_predicates,
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
    return _codegen_common_cute(
        "atomic_cas",
        state,
        value_exprs=_to_ast_values([state.ast_args[2], state.ast_args[3]]),
        keyword_names=["cmp", "val"],
    )
