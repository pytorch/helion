from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..dtype_utils import cast_ast
from ..matmul_utils import _needs_f32_accumulator
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..helper_function import CodegenInterface


# fx ops that preserve the underlying byte storage of a tensor (no dtype
# change, no arithmetic).  Used to trace a matmul operand back to its raw
# ``memory_ops.load`` so we can tell a raw fp8 *byte* load apart from a
# *computed* fp8 register value (e.g. ``p = exp2(...).to(fp8)``).
_FP8_LOAD_PASSTHROUGH_TARGETS = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.expand.default,
    }
)


def _cute_operand_is_computed_fp8(node: object) -> bool:
    """Return True when an fp8 operand is a *computed* register value.

    A computed fp8 value (e.g. ``p = exp2(...).to(fp8)``) is produced by a
    dtype conversion from a non-fp8 source and lands in registers as a typed
    ``cutlass.Float8E4M3FN``.  Widening it uses an ordinary numeric cast -
    routing it through the raw-byte PTX decode emits an invalid
    ``(i8) -> f8E4M3FN`` conversion that fails to lower.

    A *raw* fp8 load (possibly hoisted out of the K loop, so its fx node is a
    loop-carried ``_new_var`` placeholder) keeps the value as raw
    ``cutlass.Uint8`` bytes and MUST go through the PTX decode.  Anything that
    is not provably a from-non-fp8 conversion is treated as a raw load.
    """
    import torch.fx

    cur = node
    for _ in range(64):
        if not isinstance(cur, torch.fx.Node):
            return False
        if (
            cur.op == "call_function"
            and cur.target is torch.ops.prims.convert_element_type.default
        ):
            source = cur.args[0] if cur.args else None
            src_dtype = None
            if isinstance(source, torch.fx.Node):
                src_val = source.meta.get("val")
                if isinstance(src_val, torch.Tensor):
                    src_dtype = src_val.dtype
            # A convert *from* a non-fp8 dtype produces a computed fp8 value.
            return src_dtype is not torch.float8_e4m3fn
        if cur.op != "call_function" or cur.target not in _FP8_LOAD_PASSTHROUGH_TARGETS:
            return False
        inputs = [a for a in cur.args if isinstance(a, torch.fx.Node)]
        if len(inputs) != 1:
            return False
        cur = inputs[0]
    return False


# fx ops that re-expose a value at the same numeric magnitude (no rescale).
# Tracing the accumulator through these reaches the loop-carried phi without
# crossing an arithmetic rescale.
_ACC_PHI_PASSTHROUGH_TARGETS = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.expand.default,
        torch.ops.prims.convert_element_type.default,
    }
)


def _cute_acc_is_rescaled_loop_carried(acc_node: object) -> bool:
    """Return True for a *rescaled* loop-carried accumulator (online softmax).

    The cross-lane ``dot_acc`` accumulator (sum the K products across lane
    iterations in fp32, then add the accumulator once) is correct - and more
    precise - for:

    * a standalone ``addmm`` whose bias is a plain load (loop-invariant), and
    * a plain matmul K-loop ``acc = hl.dot(x, y, acc=acc)`` where ``acc`` is the
      loop-carried phi added to verbatim (no rescale inside the K loop).

    A flash-attention online-softmax recurrence instead *rescales* the
    accumulator every K iteration (``acc = acc * alpha + p @ v``).  The ``acc``
    operand passed to the dot is therefore a value *derived from* the loop phi
    through an arithmetic rescale rather than the bare phi.  There ``dot_acc``
    double-counts every prior product (the running sum is re-added each
    iteration while the rescaled base is recomputed), so the matmul must emit a
    per-iteration ``acc = (rescaled acc) + product`` update and let the loop phi
    carry the running sum.

    Detection: strip pure dtype-casts / views off ``acc``.  If what remains is
    a ``_new_var`` loop-carried phi (or anything not derived from a phi), the
    accumulator is NOT rescaled -> keep ``dot_acc``.  If the phi is only
    reachable *through* an arithmetic op (mul/add/sub/div/...), the accumulator
    is rescaled -> use the per-iteration form.
    """
    import torch.fx

    from ...language._tracing_ops import _new_var

    if not isinstance(acc_node, torch.fx.Node):
        return False

    # Strip pure casts / views: a bare (possibly re-typed) loop phi is a plain
    # accumulation, not a rescale.
    cur = acc_node
    for _ in range(64):
        if not isinstance(cur, torch.fx.Node):
            return False
        if cur.op == "call_function" and cur.target is _new_var:
            return False  # bare loop phi -> plain accumulation
        if cur.op == "call_function" and cur.target in _ACC_PHI_PASSTHROUGH_TARGETS:
            inputs = [a for a in cur.args if isinstance(a, torch.fx.Node)]
            if len(inputs) == 1:
                cur = inputs[0]
                continue
        break

    # ``cur`` is an arithmetic node (or otherwise).  It is a rescaled
    # accumulator only if it is fed by a loop-carried phi.
    seen: set[torch.fx.Node] = set()
    stack = [cur]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if len(seen) > 256:
            break
        if node.op == "call_function" and node.target is _new_var:
            src = node.args[0] if node.args else None
            if isinstance(src, torch.fx.Node) and src.op == "placeholder":
                return True
        if node.op != "call_function":
            continue
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                stack.append(arg)
            elif isinstance(arg, (list, tuple)):
                stack.extend(a for a in arg if isinstance(a, torch.fx.Node))
    return False


def _node_args_iter(node: object) -> list[object]:
    """Flatten a node's args (including list/tuple-nested args) to a list."""
    out: list[object] = []
    if not hasattr(node, "args"):
        return out
    for arg in node.args:  # type: ignore[attr-defined]
        if isinstance(arg, (list, tuple)):
            out.extend(arg)
        else:
            out.append(arg)
    return out


def _cute_k_block_varying_nodes(graph: object, k_block_id: int) -> set[object]:
    """Return the fx nodes whose value varies along the K-contraction tile.

    The within-K-tile per-element index for block ``k_block_id`` is the
    ``_get_symnode('block_size_{k_block_id}')`` node (used directly as a load
    index) and any ``tile_index`` / ``tile_id`` taken over it.  Every fx value
    transitively computed from one of those seeds varies as the lane / K
    contraction index advances.  ``acc`` rescales that touch this set are
    lane-VARYING (flash-attention's ``alpha``); rescales that avoid it are
    lane-INVARIANT (GDN's per-chunk decay, which indexes ``g`` by the chunk
    boundary, not the within-chunk position).
    """
    import torch.fx

    from ...language import _tracing_ops

    if not isinstance(graph, torch.fx.Graph):
        return set()

    block_size_name = f"block_size_{k_block_id}"
    seeds: set[torch.fx.Node] = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if (
            node.target is _tracing_ops._get_symnode
            and node.args
            and node.args[0] == block_size_name
        ):
            seeds.add(node)
    # Forward-propagate: any node consuming a varying value also varies.  This
    # also captures the ``tile_index`` / ``tile_id`` taken over the K block-size
    # symnode (still the within-tile index) and everything downstream.
    varying: set[torch.fx.Node] = set(seeds)
    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node in varying or node.op != "call_function":
                continue
            for arg in _node_args_iter(node):
                if isinstance(arg, torch.fx.Node) and arg in varying:
                    varying.add(node)
                    changed = True
                    break
    return cast("set[object]", varying)


def _cute_rescale_is_lane_invariant(acc_node: object, k_block_id: int | None) -> bool:
    """Return True when ``acc``'s rescale does NOT depend on the K-contraction
    tile index (lane-invariant rescale).

    Only meaningful when ``acc`` is a rescaled loop-carried accumulator (see
    ``_cute_acc_is_rescaled_loop_carried``).  GDN's chunk recurrence multiplies
    the accumulator by a per-chunk decay (``b_h *= exp(g[..., chunk_last]))``
    that is constant across the within-chunk / lane index, so the cross-lane
    ``dot_acc`` running sum stays correct (the rescale factors out of the sum).
    Flash-attention's ``acc = acc * alpha`` instead rescales by ``alpha``, which
    is derived from the per-K-tile scores, so it is lane-varying and the
    per-iteration update must be kept.
    """
    import torch.fx

    from ...language._tracing_ops import _new_var

    if k_block_id is None or not isinstance(acc_node, torch.fx.Node):
        return False

    graph = acc_node.graph
    varying = _cute_k_block_varying_nodes(graph, k_block_id)
    if not varying:
        # No identifiable K-tile index node: be conservative and treat the
        # rescale as lane-varying (keep the existing per-iteration behavior).
        return False

    # Walk the rescale subgraph feeding ``acc_node``.  Stop at the loop-carried
    # phi (``_new_var`` of a placeholder) — the phi carries the running value
    # and is not part of the rescale dependency we are classifying.
    seen: set[torch.fx.Node] = set()
    stack: list[torch.fx.Node] = [acc_node]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if len(seen) > 1024:
            return False  # too large to analyze safely -> conservative
        if node in varying:
            return False  # rescale touches a K-varying value -> lane-varying
        if node.op == "call_function" and node.target is _new_var:
            continue  # loop-carried phi: do not recurse past it
        if node.op != "call_function":
            continue
        for arg in _node_args_iter(node):
            if isinstance(arg, torch.fx.Node):
                stack.append(arg)
    return True


def _cast_operand_to_f32(
    term: ast.AST, dtype: torch.dtype | None, *, is_computed_fp8: bool
) -> ast.AST:
    """Promote a sub-f32 matmul operand to float32 for the scalar fallback.

    Raw ``float8_e4m3fn`` tensors are loaded as raw ``cutlass.Uint8`` bytes
    (the CuTe DSL has no scalar fp8 dereference), so a plain numeric cast would
    interpret the storage byte as an integer (e.g. ``0x40`` -> ``64.0``)
    instead of decoding the fp8 value.  Route those through the bit-exact PTX
    decode helper.

    A *computed* fp8 value (``exp2(...).to(fp8)``) is already a typed
    ``cutlass.Float8E4M3FN`` register value, so it widens with the ordinary
    numeric cast - applying the byte decode there emits an invalid
    ``(i8) -> f8E4M3FN`` conversion.  bf16/fp16 are genuine numeric widenings
    and always use the ordinary cast.
    """
    if dtype is torch.float8_e4m3fn and not is_computed_fp8:
        return expr_from_string("_cute_fp8e4m3fn_to_float32({x})", x=term)
    return cast_ast(term, torch.float32)


def _cute_active_thread_layout(
    cg: CodegenInterface,
) -> tuple[dict[int, int], dict[int, int]]:
    axis_sizes: dict[int, int] = {}
    block_axes: dict[int, int] = {}
    seen: set[int] = set()
    active_device_loops = getattr(cg, "active_device_loops", None)
    if isinstance(active_device_loops, dict):
        for loops in active_device_loops.values():
            for state in loops:
                key = id(state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
                block_axes.update(state.block_thread_axes)
    current_grid_state = getattr(cg, "current_grid_state", None)
    if current_grid_state is not None:
        for axis, size in current_grid_state.thread_axis_sizes.items():
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        block_axes.update(current_grid_state.block_thread_axes)
    return axis_sizes, block_axes


def _emit_cute_grouped_sum_reduction_shared_two_stage(
    cg: CodegenInterface,
    input_name: str,
    *,
    value_dtype: torch.dtype,
    identity_expr: str,
    lane_var: str,
    lane_in_group_var: str,
    lane_mod_pre_var: str,
    pre: int,
    group_span: int,
    group_count: int,
) -> str:
    result_var = cg.device_function.new_var("dot_reduce_result")
    cg.add_statement(
        f"{result_var} = _cute_grouped_reduce_shared_two_stage("
        f"{input_name}, 'sum', {identity_expr}, "
        f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
        f"pre={pre}, group_span={group_span}, group_count={group_count})"
    )
    return result_var


def _emit_cute_grouped_sum_reduction_shared_tree(
    cg: CodegenInterface,
    input_name: str,
    *,
    value_dtype: torch.dtype,
    identity_expr: str,
    lane_var: str,
    lane_in_group_var: str,
    lane_mod_pre_var: str,
    pre: int,
    group_span: int,
    num_threads: int,
    group_count: int,
) -> str:
    result_var = cg.device_function.new_var("dot_reduce_result")
    cg.add_statement(
        f"{result_var} = _cute_grouped_reduce_shared_tree("
        f"{input_name}, 'sum', {identity_expr}, "
        f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
        f"pre={pre}, group_span={group_span}, "
        f"num_threads={num_threads}, group_count={group_count})"
    )
    return result_var


def _emit_cute_grouped_sum_reduction(
    cg: CodegenInterface,
    input_name: str,
    *,
    value_dtype: torch.dtype,
    loop_state: object,
    k_block_id: int,
) -> str:
    backend = CompileEnvironment.current().backend
    if backend.name != "cute":
        return backend.reduction_expr(input_name, "sum", 0, threads_in_group=1)

    axis_sizes, block_axes = _cute_active_thread_layout(cg)
    loop_block_axes = getattr(loop_state, "block_thread_axes", {})
    thread_axis = block_axes.get(k_block_id)
    if thread_axis is None and isinstance(loop_block_axes, dict):
        thread_axis = loop_block_axes.get(k_block_id)
    if thread_axis is None:
        return input_name

    reduce_extent = axis_sizes.get(thread_axis, 1)
    if reduce_extent <= 1:
        return input_name

    pre = 1
    for axis in range(thread_axis):
        pre *= axis_sizes.get(axis, 1)
    group_span = pre * reduce_extent
    # ``cute.arch.warp_reduction_sum`` shuffles only within a single 32-thread
    # warp (``shuffle_sync_bfly`` clamps to lane 31), so a direct warp reduce
    # is only correct when the whole reduction group fits inside one warp.  A
    # contraction dim with >32 threads (e.g. head_dim=64 on the QK matmul) must
    # use the shared-memory two-stage reduction instead - otherwise it silently
    # sums only the first 32 of the contraction elements.
    direct_warp_ok = pre <= 1 and reduce_extent <= 32
    if direct_warp_ok:
        return backend.reduction_expr(
            input_name, "sum", 0, threads_in_group=reduce_extent
        )

    lane_expr = backend.thread_linear_index_expr(axis_sizes)
    if lane_expr is None:
        if reduce_extent <= 32:
            return backend.reduction_expr(
                input_name, "sum", 0, threads_in_group=reduce_extent
            )
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback cannot reduce a >32-thread contraction "
            "without a linear thread index",
        )

    num_threads = 1
    for size in axis_sizes.values():
        num_threads *= size
    actual_threads = 1
    for size in getattr(cg, "max_thread_block_dims", ()):
        actual_threads *= max(size, 1)
    if actual_threads > 0 and num_threads > actual_threads:
        if reduce_extent <= 32:
            return backend.reduction_expr(
                input_name, "sum", 0, threads_in_group=reduce_extent
            )
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback cannot reduce a >32-thread contraction "
            "when the planned thread count exceeds the launch block",
        )

    identity_expr = f"{backend.dtype_str(value_dtype)}(0)"
    if group_span <= 32:
        return (
            "_cute_grouped_reduce_warp("
            f"{input_name}, 'sum', {identity_expr}, {lane_expr}, "
            f"pre={pre}, group_span={group_span})"
        )

    assert num_threads % group_span == 0, (
        f"num_threads ({num_threads}) must be divisible by group_span ({group_span})"
    )
    lane_var = cg.device_function.new_var("dot_lane")
    lane_in_group_var = cg.device_function.new_var("dot_lane_in_group")
    lane_mod_pre_var = cg.device_function.new_var("dot_lane_mod_pre")
    cg.add_statement(f"{lane_var} = {lane_expr}")
    cg.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
    cg.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}")
    if group_span % 32 == 0:
        return _emit_cute_grouped_sum_reduction_shared_two_stage(
            cg,
            input_name,
            value_dtype=value_dtype,
            identity_expr=identity_expr,
            lane_var=lane_var,
            lane_in_group_var=lane_in_group_var,
            lane_mod_pre_var=lane_mod_pre_var,
            pre=pre,
            group_span=group_span,
            group_count=num_threads // group_span,
        )
    return _emit_cute_grouped_sum_reduction_shared_tree(
        cg,
        input_name,
        value_dtype=value_dtype,
        identity_expr=identity_expr,
        lane_var=lane_var,
        lane_in_group_var=lane_in_group_var,
        lane_mod_pre_var=lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        num_threads=num_threads,
        group_count=num_threads // group_span,
    )


def _emit_cute_matmul_n_collapse(
    cg: CodegenInterface,
    lhs: ast.AST,
    *,
    rhs_at_n: Callable[[str], ast.AST],
    n_extent: int,
    k_block_id: int | None,
    static_k_extent: int | None,
    acc: ast.AST,
    acc_dtype: torch.dtype | None,
    lhs_dtype: torch.dtype | None,
    rhs_dtype: torch.dtype | None,
    lhs_node: object,
    rhs_node: object,
) -> ast.AST:
    """Lower a static-M==N-collapse baddbmm reduced over N (CuTe layout A).

    The matmul's M (lhs free) and N (rhs free) axes share a block id, so the
    standard fallback indexes the rhs N at the M thread index and computes only
    the diagonal.  Here M stays the thread axis and N becomes a serial loop:

        out_acc = 0
        for n in range(N):
            out_acc += K-reduce( lhs[m, k] * rhs[n, k] )
        result = acc + out_acc

    where the K reduction is the existing per-lane / cross-thread reduction.
    Because the only consumer of this matmul's result is a sum over N, folding
    the N reduction here makes ``result`` already the N-summed value and the
    downstream ``.sum(-1)`` a no-op.
    """
    from ..generate_ast import GenerateAST

    assert isinstance(cg, GenerateAST)
    if hasattr(cg, "cute_uses_matmul"):
        cg.cute_uses_matmul = True  # type: ignore[attr-defined]

    reduction_dtype: torch.dtype | None = acc_dtype
    if (
        lhs_dtype is not None
        and rhs_dtype is not None
        and _needs_f32_accumulator(lhs_dtype, rhs_dtype)
    ):
        reduction_dtype = torch.float32
    value_dtype = reduction_dtype or lhs_dtype or rhs_dtype or torch.float32

    loop_state = None
    if k_block_id is not None:
        from ..tile_strategy import DeviceLoopOrGridState

        active_device_loops = getattr(cg, "active_device_loops", None)
        if isinstance(active_device_loops, dict):
            loops = active_device_loops.get(k_block_id)
            if loops and isinstance(loops[-1], DeviceLoopOrGridState):
                loop_state = loops[-1]
        # This path sums each K element across hardware threads (cross-thread
        # reduction).  A K axis lowered as a *serial* per-thread lane loop would
        # need the products accumulated across lane iterations, which this fold
        # does not emit - reject it rather than silently summing one lane.
        if loop_state is not None:
            lane_vars = getattr(loop_state.strategy, "_lane_var_by_block", None)
            if isinstance(lane_vars, dict) and k_block_id in lane_vars:
                raise exc.BackendUnsupported(
                    "cute",
                    "CuTe static-MN-collapse baddbmm requires the contraction "
                    "axis to be a cross-thread reduction, not a serial lane loop",
                )

    backend = CompileEnvironment.current().backend
    zero_expr = f"{backend.dtype_str(value_dtype)}(0)"
    out_acc = cg.device_function.new_var("dot_n_acc")
    cg.add_statement(f"{out_acc} = {zero_expr}")

    n_var = cg.device_function.new_var("dot_n")
    loop_body: list[ast.AST] = []
    with cg.set_statements(loop_body):
        rhs = rhs_at_n(n_var)
        lhs_term: ast.AST = lhs
        rhs_term: ast.AST = rhs
        if reduction_dtype is not None:
            lhs_computed_fp8 = _cute_operand_is_computed_fp8(lhs_node)
            rhs_computed_fp8 = _cute_operand_is_computed_fp8(rhs_node)
            if (
                lhs_dtype is not None
                and rhs_dtype is not None
                and _needs_f32_accumulator(lhs_dtype, rhs_dtype)
            ):
                lhs_term = _cast_operand_to_f32(
                    lhs_term, lhs_dtype, is_computed_fp8=lhs_computed_fp8
                )
                rhs_term = _cast_operand_to_f32(
                    rhs_term, rhs_dtype, is_computed_fp8=rhs_computed_fp8
                )
        product = expr_from_string("{lhs} * {rhs}", lhs=lhs_term, rhs=rhs_term)
        if reduction_dtype is not None:
            product = cast_ast(product, reduction_dtype)
        if k_block_id is not None:
            reduction_input = cg.lift(product, dce=True, prefix="dot_product").id
            reduced_expr: ast.AST = expr_from_string(
                _emit_cute_grouped_sum_reduction(
                    cg,
                    reduction_input,
                    value_dtype=value_dtype,
                    loop_state=loop_state,
                    k_block_id=k_block_id,
                )
            )
        else:
            reduced_expr = product
            if static_k_extent is not None and static_k_extent > 1:
                scale = expr_from_string(
                    f"{backend.dtype_str(value_dtype)}({static_k_extent})"
                )
                reduced_expr = expr_from_string(
                    "({product}) * ({scale})", product=reduced_expr, scale=scale
                )
        cg.add_statement(
            statement_from_string(
                f"{out_acc} = {out_acc} + ({{reduced}})", reduced=reduced_expr
            )
        )

    for_node = statement_from_string(f"for {n_var} in range({n_extent}):\n    pass")
    assert isinstance(for_node, ast.For)
    for_node.body = cast("list[ast.stmt]", loop_body)
    cg.add_statement(for_node)

    result: ast.AST = expr_from_string(out_acc)
    base_acc = acc
    if acc_dtype is not None and acc_dtype != reduction_dtype and reduction_dtype:
        base_acc = cast_ast(base_acc, reduction_dtype)
    result = expr_from_string("{acc} + {product}", acc=base_acc, product=result)
    if acc_dtype is not None and acc_dtype != reduction_dtype:
        result = cast_ast(result, acc_dtype)
    return result


def _emit_cute_matmul(
    cg: CodegenInterface,
    lhs: ast.AST | CutePackedAffineLoad,
    rhs: ast.AST | CutePackedTerms,
    *,
    accumulate_in_lane_loop: bool = True,
    k_block_id: int | None,
    static_k_extent: int | None = None,
    acc: ast.AST | None = None,
    out_dtype: torch.dtype | None = None,
    acc_dtype: torch.dtype | None = None,
    lhs_dtype: torch.dtype | None = None,
    rhs_dtype: torch.dtype | None = None,
    lhs_node: object = None,
    rhs_node: object = None,
    acc_node: object = None,
) -> ast.AST:
    """Build a CuTe matmul fallback using a cross-thread reduction over K."""
    if hasattr(cg, "cute_uses_matmul"):
        cg.cute_uses_matmul = True  # type: ignore[attr-defined]
    reduction_dtype: torch.dtype | None = acc_dtype or out_dtype
    lhs_terms: tuple[ast.AST, ...]
    if isinstance(lhs, CutePackedAffineLoad):
        lhs_terms = tuple(lhs.terms)
    else:
        lhs_terms = (lhs,)
    rhs_terms: tuple[ast.AST, ...]
    if isinstance(rhs, CutePackedTerms):
        rhs_terms = tuple(rhs.terms)
    else:
        rhs_terms = (rhs,)
    if (
        lhs_dtype is not None
        and rhs_dtype is not None
        and _needs_f32_accumulator(lhs_dtype, rhs_dtype)
    ):
        reduction_dtype = torch.float32
        lhs_computed_fp8 = _cute_operand_is_computed_fp8(lhs_node)
        rhs_computed_fp8 = _cute_operand_is_computed_fp8(rhs_node)
        rhs_terms = tuple(
            _cast_operand_to_f32(term, rhs_dtype, is_computed_fp8=rhs_computed_fp8)
            for term in rhs_terms
        )
        lhs_terms = tuple(
            _cast_operand_to_f32(term, lhs_dtype, is_computed_fp8=lhs_computed_fp8)
            for term in lhs_terms
        )
    if len(lhs_terms) == len(rhs_terms):
        term_pairs = zip(lhs_terms, rhs_terms, strict=True)
    elif len(lhs_terms) == 1:
        term_pairs = ((lhs_terms[0], rhs_term) for rhs_term in rhs_terms)
    elif len(rhs_terms) == 1:
        term_pairs = ((lhs_term, rhs_terms[0]) for lhs_term in lhs_terms)
    else:
        raise RuntimeError(
            f"unsupported packed CuTe matmul arity: lhs={len(lhs_terms)} rhs={len(rhs_terms)}"
        )
    product_terms = [
        expr_from_string("{lhs} * {rhs}", lhs=lhs_term, rhs=rhs_term)
        for lhs_term, rhs_term in term_pairs
    ]
    if reduction_dtype is not None:
        product_terms = [cast_ast(term, reduction_dtype) for term in product_terms]
    product = product_terms[0]
    for term in product_terms[1:]:
        product = expr_from_string("{lhs} + {rhs}", lhs=product, rhs=term)
    loop_state = None
    if k_block_id is not None:
        from ..tile_strategy import DeviceLoopOrGridState

        active_device_loops = getattr(cg, "active_device_loops", None)
        if isinstance(active_device_loops, dict):
            loops = active_device_loops.get(k_block_id)
            if loops and isinstance(loops[-1], DeviceLoopOrGridState):
                loop_state = loops[-1]
    reduction_base_acc = acc
    if loop_state is not None and k_block_id is not None:
        lane_vars = getattr(loop_state.strategy, "_lane_var_by_block", None)
        lane_var = lane_vars.get(k_block_id) if isinstance(lane_vars, dict) else None
        if not accumulate_in_lane_loop:
            lane_var = None
        # BUG#1 fix: a flash-attention online-softmax accumulator is rescaled
        # (``acc = acc * alpha + p @ v``) every iteration of the K lane loop.
        # The cross-lane ``dot_acc`` running sum would then be re-added each
        # iteration while ``acc`` is independently rescaled, double-counting
        # every prior product.  When ``acc`` is such a loop-carried value, emit
        # the per-iteration ``acc = acc + product`` update instead (the loop
        # phi carries the running sum) by skipping the ``dot_acc`` path.  A
        # loop-invariant accumulator (e.g. a standalone ``addmm`` bias) keeps
        # ``dot_acc`` so the per-lane products are summed before the bias add.
        #
        # EXCEPTION: a *lane-invariant* rescale (GDN's chunk recurrence
        # ``b_h *= decay`` where ``decay`` depends only on the chunk, not the
        # within-chunk / lane index) factors out of the cross-lane sum:
        # ``sum_c (b_h*decay + p_k[c]*b_v[c]) over c`` is wrong, but the
        # mathematically intended update ``b_h = b_h*decay + sum_c p_k[c]*b_v[c]``
        # is exactly what the ``dot_acc`` path produces once the AST post-pass
        # hoists the rescale + final add out of the lane loop.  Keep ``dot_acc``
        # in that case; only flash-attention's lane-varying rescale falls back.
        if (
            lane_var is not None
            and acc is not None
            and _cute_acc_is_rescaled_loop_carried(acc_node)
            and not _cute_rescale_is_lane_invariant(acc_node, k_block_id)
        ):
            lane_var = None
        if lane_var is not None:
            product_name = cg.lift(product, dce=True, prefix="dot_product").id
            dot_acc = cg.device_function.new_var("dot_acc")
            dot_acc_base = None
            base_acc_source: ast.AST | None = None
            capture_base_outside_lane = False
            if acc is not None:
                dot_acc_base = cg.device_function.new_var("dot_acc_base")
                base_acc_source = acc
                if isinstance(acc, ast.Name) and "_copy" in acc.id:
                    capture_base_outside_lane = True
                    base_acc_source = expr_from_string(acc.id.split("_copy", 1)[0])
            if reduction_dtype is not None:
                zero_init = f"{CompileEnvironment.current().backend.dtype_str(reduction_dtype)}(0)"
            else:
                zero_init = "0"
            statements_stack = getattr(cg, "statements_stack", None)
            if isinstance(statements_stack, list) and len(statements_stack) >= 2:
                if (
                    dot_acc_base is not None
                    and base_acc_source is not None
                    and capture_base_outside_lane
                ):
                    statements_stack[-2].append(
                        statement_from_string(
                            f"{dot_acc_base} = {{acc}}", acc=base_acc_source
                        )
                    )
                statements_stack[-2].append(
                    statement_from_string(f"{dot_acc} = {zero_init}")
                )
            else:
                if (
                    dot_acc_base is not None
                    and base_acc_source is not None
                    and capture_base_outside_lane
                ):
                    cg.add_statement(
                        statement_from_string(
                            f"{dot_acc_base} = {{acc}}", acc=base_acc_source
                        )
                    )
                cg.add_statement(f"{dot_acc} = {zero_init}")
            if (
                dot_acc_base is not None
                and base_acc_source is not None
                and not capture_base_outside_lane
            ):
                cg.add_statement(
                    statement_from_string(
                        f"{dot_acc_base} = {{acc}}", acc=base_acc_source
                    )
                )
            if dot_acc_base is not None:
                reduction_base_acc = expr_from_string(dot_acc_base)
            cg.add_statement(f"{dot_acc} = {dot_acc} + {product_name}")
            reduction_input = dot_acc
        else:
            reduction_input = cg.lift(product, dce=True, prefix="dot_product").id
        reduction_value_dtype = (
            reduction_dtype or lhs_dtype or rhs_dtype or out_dtype or torch.float32
        )
        product = expr_from_string(
            _emit_cute_grouped_sum_reduction(
                cg,
                reduction_input,
                value_dtype=reduction_value_dtype,
                loop_state=loop_state,
                k_block_id=k_block_id,
            )
        )
    elif static_k_extent is not None and static_k_extent > 1:
        scale_dtype = reduction_dtype or lhs_dtype or rhs_dtype or out_dtype
        scale_expr = str(static_k_extent)
        if scale_dtype is not None:
            scale_expr = (
                f"{CompileEnvironment.current().backend.dtype_str(scale_dtype)}"
                f"({static_k_extent})"
            )
        product = expr_from_string(
            "({product}) * ({scale})",
            product=product,
            scale=expr_from_string(scale_expr),
        )
    if reduction_base_acc is not None and reduction_dtype is not None:
        if acc_dtype != reduction_dtype:
            reduction_base_acc = cast_ast(reduction_base_acc, reduction_dtype)
        product = expr_from_string(
            "{acc} + {product}", acc=reduction_base_acc, product=product
        )
    if acc is None and out_dtype is not None and out_dtype != reduction_dtype:
        product = cast_ast(product, out_dtype)
    elif (
        reduction_base_acc is not None
        and acc_dtype is not None
        and acc_dtype != reduction_dtype
    ):
        product = cast_ast(product, acc_dtype)
    return product
