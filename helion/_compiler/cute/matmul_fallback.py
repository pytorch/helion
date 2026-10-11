from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..dtype_utils import cast_ast
from ..generate_ast import GenerateAST
from ..matmul_utils import _needs_f32_accumulator
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms
from .matmul_utils import SHAPE_QUERY_TARGETS
from .matmul_utils import CuteAtomicLaneRoute
from .matmul_utils import cute_atomic_consumer_lane_route
from .matmul_utils import cute_per_lane_atomic_consumer
from .matmul_utils import cute_scalar_reduction_dtype
from .matmul_utils import loop_carries
from .matmul_utils import reads_loop_carry
from .matmul_utils import reads_value_of
from .matmul_utils import strip_new_vars

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
    """Return True when arithmetic transforms a loop-carried accumulator.

    The cross-lane ``dot_acc`` accumulator (sum the K products across lane
    iterations in fp32, then add the accumulator once) is correct - and more
    precise - for:

    * a standalone ``addmm`` whose bias is a plain load (loop-invariant), and
    * a plain matmul K-loop ``acc = hl.dot(x, y, acc=acc)`` where ``acc`` is the
      loop-carried phi added to verbatim (no rescale inside the K loop).

    A rescaled accumulator is derived from the loop phi through arithmetic.
    If that arithmetic depends on another contraction-lane reduction, eagerly
    updating the carry would reuse partial products or an incomplete rescale.
    The product then needs its own complete owned sum; the reduction scheduler
    must prove that the rescale and carry update execute once afterward.

    Detection: strip pure dtype-casts / views off ``acc``.  If what remains is
    a ``_new_var`` loop-carried phi (or anything not derived from a phi), the
    accumulator is NOT rescaled -> keep ``dot_acc``.  If the phi is only
    reachable *through* an arithmetic op (mul/add/sub/div/...), the accumulator
    is rescaled -> require the separate lane-invariance/scheduling proof.
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
    A rescale derived from per-K-tile scores instead needs those reductions
    finalized before its once-per-tile update. The owned product-sum path
    preserves that dependency without re-adding a running partial sum.
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
    from ..generate_ast import GenerateAST

    assert isinstance(cg, GenerateAST)
    # A reduction in a later kernel phase still reserves physical thread axes
    # in this launch. Include them when computing the stride between K lanes.
    axis_sizes = cg.device_function.tile_strategy.thread_axis_sizes()
    cg._record_thread_axis_sizes(axis_sizes)
    block_axes: dict[int, int] = {}
    seen: set[int] = set()
    for loops in cg.active_device_loops.values():
        for state in loops:
            key = id(state)
            if key in seen:
                continue
            seen.add(key)
            for axis, size in state.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
            block_axes.update(state.block_thread_axes)
    current_grid_state = cg.current_grid_state
    if current_grid_state is not None:
        for axis, size in current_grid_state.thread_axis_sizes.items():
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        block_axes.update(current_grid_state.block_thread_axes)
    # Free ``hl.arange`` dims live on synthetic thread axes outside the
    # strategy loop states, but they are launched thread rows all the same.
    for axis, size in cg.cute_synthetic_arange_axis_sizes.items():
        axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
    return axis_sizes, block_axes


def _cute_launch_layout_matches(
    cg: CodegenInterface,
    axis_sizes: dict[int, int],
    *,
    thread_axis: int,
    group_span: int,
) -> bool:
    """Whether the active thread layout accounts for every launched thread.

    The grouped reductions key lanes by their linear thread index.  A warp
    group (``group_span <= 32``) only needs the axes up to the contraction axis
    to match the launch block, but the shared-memory stages index partials by
    ``lane // group_span`` over the whole CTA, so any launch axis the layout
    omits (a free ``hl.arange`` row, a wider axis recorded by a sibling loop)
    makes distinct rows share one partial slot and silently sum together.
    """
    launch_dims = tuple(getattr(cg, "max_thread_block_dims", ()))
    if not launch_dims:
        return True
    launch = {axis: size for axis, size in enumerate(launch_dims) if size > 1}
    active = {axis: size for axis, size in axis_sizes.items() if size > 1}
    if group_span > 32:
        return active == launch
    return all(
        active.get(axis, 1) == launch.get(axis, 1) for axis in range(thread_axis + 1)
    )


def _emit_cute_grouped_sum_reduction_shared_two_stage(
    cg: CodegenInterface,
    input_name: str,
    *,
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


def _widen_lane_layout_for_barrier_phases(
    cg: CodegenInterface, axis_sizes: dict[int, int], *, subject: str
) -> None:
    """Widen a cross-lane reduce's thread layout to the ``hl.barrier()`` launch.

    The phases of a barrier kernel share one launch block (the elementwise
    max of their thread extents), so another phase may run more lanes on the
    axes around this reduce.  Surplus lanes on the reduce axis load the
    identity through the tile/K bounds masks, so folding them in keeps every
    lane's result complete; surplus rows on the other axes form extra
    (redundant) groups.  ``axis_sizes`` is widened in place on all three axes
    so the linear lane index and group count cover the whole launch, and the
    assumed layout is recorded under ``subject`` so the launcher
    (``backend._multi_phase_block_dims``) rejects a final block shape that
    differs from it on any axis.  Single-phase kernels are left alone.
    """
    # Unit tests drive these emitters with a bare namespace; only a real
    # ``GenerateAST`` carries the host function whose phases matter here.
    host_function = getattr(cg, "host_function", None)
    if host_function is None or len(host_function.device_ir.phases) <= 1:
        return
    device_function = DeviceFunction.current()
    launch_dims = device_function.tile_strategy.thread_block_dims()
    for axis in range(3):
        if launch_dims[axis] > axis_sizes.get(axis, 1):
            axis_sizes[axis] = launch_dims[axis]
    device_function.cute_state.multi_phase_lane_reduce_layouts.append(
        (subject, {axis: axis_sizes.get(axis, 1) for axis in range(3)})
    )


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
    _widen_lane_layout_for_barrier_phases(
        cg, axis_sizes, subject="staged matmul product sum"
    )

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
    if not _cute_launch_layout_matches(
        cg, axis_sizes, thread_axis=thread_axis, group_span=group_span
    ):
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback cannot reduce a multi-warp contraction "
            "when the launch block has thread rows outside the contraction layout",
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
        identity_expr=identity_expr,
        lane_var=lane_var,
        lane_in_group_var=lane_in_group_var,
        lane_mod_pre_var=lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        num_threads=num_threads,
        group_count=num_threads // group_span,
    )


def _emit_cute_owned_product_sum(
    cg: CodegenInterface,
    input_name: str,
    *,
    value_dtype: torch.dtype,
    loop_state: object,
    k_block_id: int,
    owner_lane: str,
) -> str:
    """Describe a complete serial-K and physical-thread FP32 product sum."""
    from ..tile_strategy import _lane_reduce_marker_expr

    backend = CompileEnvironment.current().backend
    if value_dtype != torch.float32:
        raise exc.BackendUnsupported("cute", "staged matmul sum requires FP32")
    axis_sizes, block_axes = _cute_active_thread_layout(cg)
    loop_block_axes = getattr(loop_state, "block_thread_axes", {})
    thread_axis = block_axes.get(k_block_id)
    if thread_axis is None and isinstance(loop_block_axes, dict):
        thread_axis = loop_block_axes.get(k_block_id)
    if thread_axis is not None:
        _widen_lane_layout_for_barrier_phases(
            cg, axis_sizes, subject="staged matmul product sum"
        )
    reduce_extent, pre, group_span, group_count, lane_expr = (
        _cute_lane_reduce_thread_group(
            cg,
            axis_sizes,
            thread_axis,
            subject="staged matmul sum",
        )
    )
    # The grouped (shared-memory / multi-warp) reduce must see exactly the
    # launch layout, or rows on the other thread axes collide in the
    # two-stage buffer.
    if group_span and not _cute_launch_layout_matches(
        cg, axis_sizes, thread_axis=thread_axis or 0, group_span=group_span
    ):
        raise exc.BackendUnsupported(
            "cute", "staged matmul sum has no proved physical thread group"
        )
    return _lane_reduce_marker_expr(
        input_name,
        "sum",
        f"{backend.dtype_str(value_dtype)}(0)",
        reduce_extent,
        group_pre=pre,
        group_span=group_span,
        group_lane_expr=lane_expr,
        group_count=group_count,
        owner_lane=owner_lane,
        matmul_contribution=True,
    )


def _cute_lane_reduce_thread_group(
    cg: CodegenInterface,
    axis_sizes: dict[int, int],
    thread_axis: int | None,
    *,
    subject: str,
) -> tuple[int, int, int, int, str]:
    """Physical thread group of a ``_helion_lane_reduce`` marker whose live
    thread axis is ``thread_axis``.

    Returns ``(reduce_extent, pre, group_span, group_count, lane_expr)``.
    ``axis_sizes`` must be the FULL launch layout (every live thread axis,
    including the ones a persistent reduction owns): ``pre`` is the product of
    the sibling extents below ``thread_axis``, so the finalize folds exactly
    the threads that share this lane's sibling coordinates instead of
    consecutive warp lanes. ``group_span`` is 0 (and ``lane_expr`` empty)
    when a plain consecutive-lane warp reduce is already correct.  The
    reduction-op markers over the same lane loop derive their group from the
    same launch layout (``BlockReductionStrategy._lane_loop_group_params``),
    so the matmul contribution's finalize folds exactly the threads the
    reduction finalizes.
    """
    backend = CompileEnvironment.current().backend
    reduce_extent = axis_sizes.get(thread_axis, 1) if thread_axis is not None else 1
    pre = 1
    for axis in range(thread_axis or 0):
        pre *= axis_sizes.get(axis, 1)
    group_span = pre * reduce_extent
    if reduce_extent <= 1 or (pre <= 1 and reduce_extent <= 32):
        return reduce_extent, pre, 0, 1, ""
    lane_expr = backend.thread_linear_index_expr(axis_sizes)
    num_threads = 1
    for size in axis_sizes.values():
        num_threads *= size
    actual_threads = 1
    for size in getattr(cg, "max_thread_block_dims", ()):
        actual_threads *= max(size, 1)
    if (
        lane_expr is None
        or num_threads > actual_threads
        or num_threads % group_span
        or (group_span > 32 and group_span % 32)
    ):
        raise exc.BackendUnsupported(
            "cute", f"{subject} has no proved physical thread group"
        )
    return reduce_extent, pre, group_span, num_threads // group_span, lane_expr


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

    reduction_dtype = cute_scalar_reduction_dtype(lhs_dtype, rhs_dtype, acc_dtype)
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


def _cute_k_lane_loop_is_innermost(cg: GenerateAST, loop_state: object) -> bool:
    """Whether the K block's loop body is the innermost open statement scope.

    The per-thread ``dot_acc`` running sum is zeroed one statement list up,
    outside the K loop body and the lane loop that wraps it, and then
    accumulates on every iteration below that point.  That equals the K
    contraction only when nothing but the K loop and its lane loop sit between
    the zero-init and the accumulate.  A free-axis device loop or lane loop
    nested inside the K loop (``for lane_K: ... for tile_M: for lane_M:
    dot_acc += ...``) would be summed as well, so such matmuls take the owned
    product-sum marker route whose lane scheduler proves or rejects a lowering.

    A device loop's body is its ``inner_statements``; the grid body is the
    list pushed directly below the grid's ``hoist_parent_statements``.
    """
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState

    statements_stack = cg.statements_stack
    if isinstance(loop_state, DeviceLoopState):
        return statements_stack[-1] is loop_state.inner_statements
    if isinstance(loop_state, DeviceGridState):
        # Every grid lane loop sits between the hoist parent and the body, so
        # a second (free-axis) lane loop would also be folded into the sum.
        return (
            len(statements_stack) >= 2
            and statements_stack[-2] is loop_state.hoist_parent_statements
            and len(loop_state.lane_loops) == 1
        )
    return False


def _loop_outputs_reading(node: torch.fx.Node) -> list[torch.fx.Node]:
    """The outputs of ``node``'s graph whose values depend on ``node``."""
    (output,) = node.graph.find_nodes(op="output")
    results = output.args[0]
    if not isinstance(results, (list, tuple)):
        results = [results]
    reached = {node}
    for graph_node in node.graph.nodes:
        if graph_node.op != "output" and any(
            arg in reached for arg in graph_node.all_input_nodes
        ):
            reached.add(graph_node)
    dependents: list[torch.fx.Node] = []
    for value in results:
        if isinstance(value, torch.fx.Node) and value in reached:
            if value not in dependents:
                dependents.append(value)
    return dependents


def _strip_passthroughs(node: torch.fx.Node) -> torch.fx.Node:
    """``node`` without the pure dtype / layout changes applied to it."""
    while (
        node.op == "call_function"
        and node.target in _ACC_PHI_PASSTHROUGH_TARGETS
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    return node


def _cute_running_sum_route(
    cg: GenerateAST,
    fx_node: object,
    acc_node: object,
    *,
    has_acc: bool,
    loop_state: object,
    k_block_id: int,
    lane_var: str,
) -> str:
    """How a matmul in a K-loop body may use the per-K-lane ``dot_acc`` running sum.

    The running sum is zeroed before the K loop and adds the products of
    every K lane of every K tile; ``dot_acc_base`` adds the accumulator from
    before the loop.  That reduces the whole K extent, which is what a bare
    matmul or one seeded by a loop-invariant accumulator lowers to when it
    is stored or overwrites its carry, and the program for
    ``carry = dot(..., acc=carry)`` when nothing else in the body reads the
    carry or the result ("running").  ``hoist_lane_invariant_chunk_recurrence``
    turns two more recurrences into once-per-tile updates ("recurrence"): an
    accumulator rescaled by lane-invariant values (``acc=carry * decay``)
    and ``carry = carry + dot(...)`` without one; it does not look inside a
    vectorized lane loop.  Any other matmul (the carry changed after the
    dot, another carry reading the result or the running carry, an
    accumulator computed from another carry) takes the owned product-sum
    route, which completes each tile's sum before its consumers ("owned").
    A matmul outside the K loop's body graph keeps the running sum.
    """
    from ..device_ir import ForLoopGraphInfo

    assert isinstance(cg, GenerateAST)
    graphs = cg.codegen_graphs
    if not isinstance(fx_node, torch.fx.Node) or (
        has_acc and not isinstance(acc_node, torch.fx.Node)
    ):
        return "running"
    carries = loop_carries(fx_node.graph, graphs)
    if carries is None or not any(
        isinstance(graph_info, ForLoopGraphInfo)
        and graph_info.graph is fx_node.graph
        and k_block_id in graph_info.block_ids
        for graph_info in graphs
    ):
        return "running"
    vectorized = lane_var in getattr(loop_state, "vec_lane_wrappers", {})
    dependents = _loop_outputs_reading(fx_node)
    if has_acc:
        assert isinstance(acc_node, torch.fx.Node)
        if not reads_loop_carry(acc_node, carries):
            return "running" if dependents in ([], [fx_node]) else "owned"
        carry = carries.get(fx_node)
        if carry is None or dependents != [fx_node]:
            return "owned"
        if strip_new_vars(acc_node) is carry:
            # Every lane rewrites the carry, so another read of it would see
            # a partial sum instead of the tile's entry value.
            copies = [acc_node]
            while copies[-1] is not carry:
                (source,) = copies[-1].all_input_nodes
                copies.append(source)
            if any(
                user not in copies
                and user is not fx_node
                and user.target not in SHAPE_QUERY_TARGETS
                for copy in copies
                for user in copy.users
            ):
                return "owned"
            return "running"
        if (
            not vectorized
            and _cute_acc_is_rescaled_loop_carried(acc_node)
            and reads_value_of(acc_node, {carry})
        ):
            return "recurrence"
        return "owned"
    if not dependents:
        return "running"
    if len(dependents) == 1:
        (update,) = dependents
        if _strip_passthroughs(update) is fx_node:
            return "running"
        carry = carries.get(update)
        if (
            not vectorized
            and carry is not None
            and update.target is torch.ops.aten.add.Tensor
            and len(update.args) == 2
            and not update.kwargs
            and {
                strip_new_vars(_strip_passthroughs(arg))
                if isinstance(arg, torch.fx.Node)
                else arg
                for arg in update.args
            }
            == {fx_node, carry}
        ):
            return "recurrence"
    return "owned"


def _cute_product_uses_owned_lane_reduction(
    cg: CodegenInterface, product: ast.AST, owner_lane: str
) -> bool:
    """Prove a product's dependence on a prior reduction of its K lane.

    Only the current straight-line scalar prefix participates. An owned
    compiler marker, followed by unique ordered pure definitions, supplies
    the dependence; names alone and enclosing scopes do not. This selects
    the existing product-sum marker, whose complete schedule must still pass
    every ownership, carry, effect and alias check in the lane scheduler.
    """
    from ..ast_read_writes import ReadWrites
    from ..generate_ast import GenerateAST
    from ..tile_strategy import _is_lane_reduce_marker_assign
    from ..tile_strategy import _is_proven_relocatable_assignment
    from ..tile_strategy import _plain_assignment_name

    if not isinstance(cg, GenerateAST):
        return False
    body = cg.statements_stack[-1]
    names = [_plain_assignment_name(statement) for statement in body]
    if None in names or len(set(names)) != len(names):
        return False
    all_names = set(names)
    defined: set[str] = set()
    dependent: set[str] = set()
    for statement, name in zip(body, names, strict=True):
        assert name is not None
        reads = set(ReadWrites.from_ast(statement).reads)
        if reads & (all_names - defined):
            return False
        marker = _is_lane_reduce_marker_assign(statement)
        if marker is not None:
            if marker.owner_lane != owner_lane:
                return False
            dependent.add(name)
        elif not _is_proven_relocatable_assignment(
            statement, allow_load=True, allow_reduction=True
        ):
            return False
        elif reads & dependent:
            dependent.add(name)
        defined.add(name)
    return bool(set(ReadWrites.from_ast(product).reads) & dependent)


def _emit_cute_matmul(
    cg: CodegenInterface,
    lhs: ast.AST | CutePackedAffineLoad | CutePackedTerms,
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
    fx_node: torch.fx.Node | None = None,
) -> ast.AST:
    """Build a CuTe matmul fallback using a cross-thread reduction over K.

    ``fx_node`` is the matmul's own device IR node.  When its K axis turns out
    to be split across a serial lane loop, the node's consumers decide whether
    an ``hl.atomic_*`` user may see per-lane partial sums instead of the
    per-thread running sum (``cute_atomic_consumer_lane_route``).
    """
    if hasattr(cg, "cute_uses_matmul"):
        cg.cute_uses_matmul = True  # type: ignore[attr-defined]
    reduction_dtype = cute_scalar_reduction_dtype(
        lhs_dtype, rhs_dtype, acc_dtype or out_dtype
    )
    lhs_terms: tuple[ast.AST, ...]
    if isinstance(lhs, (CutePackedAffineLoad, CutePackedTerms)):
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
    if loop_state is not None:
        from ..tile_strategy import PerThreadFlattenedTileStrategy

        if (
            isinstance(loop_state.strategy, PerThreadFlattenedTileStrategy)
            and loop_state.strategy._lane_var is not None
        ):
            # The fallback reduces K across the tile's threads only; the
            # flattened tile's lane loop (fewer threads than rows) would keep
            # one lane's products per thread: ``x[tile, :].T @ g[tile, :]``.
            raise exc.BackendUnsupported(
                "cute",
                "matmul contraction over a flattened tile split across a lane "
                "loop (fewer threads than tile elements)",
            )
    reduction_base_acc = acc
    product_lane: str | None = None
    if loop_state is not None and k_block_id is not None:
        lane_vars = getattr(loop_state.strategy, "_lane_var_by_block", None)
        lane_var = lane_vars.get(k_block_id) if isinstance(lane_vars, dict) else None
        if not accumulate_in_lane_loop:
            lane_var = None
        assert isinstance(cg, GenerateAST)
        if lane_var is not None:
            # K really is split across this lane loop, so an atomic consumer
            # of the running sum would add every prefix of the contraction.
            atomic_route = cute_atomic_consumer_lane_route(
                fx_node, is_acc_none=acc is None, get_graph=cg.get_graph
            )
            if atomic_route is CuteAtomicLaneRoute.PER_LANE:
                # Every K lane adds its own partial, so the atomic varies
                # along this lane loop although its index does not cover the
                # K block; the atomic lowering must not record it as uniform
                # along the loop (and have the lane placement pin it to the
                # first lane).
                assert fx_node is not None
                cg.device_function.cute_state.per_lane_atomic_lane_vars.setdefault(
                    cute_per_lane_atomic_consumer(fx_node), set()
                ).add(lane_var)
                lane_var = None
            elif atomic_route is CuteAtomicLaneRoute.OWNED:
                product_lane = lane_var
                lane_var = None
        # Keep the running sum when its inputs are independent of the owned
        # lane reductions. A reduction-fed rescale or product needs an explicit
        # product marker and a complete staged reduction schedule.  So does a K
        # lane loop that is not the innermost open scope: the running sum would
        # also fold the free-axis iterations nested inside it.
        if lane_var is not None and (
            not _cute_k_lane_loop_is_innermost(cg, loop_state)
            or (
                acc is not None
                and (
                    (
                        _cute_acc_is_rescaled_loop_carried(acc_node)
                        and not _cute_rescale_is_lane_invariant(acc_node, k_block_id)
                    )
                    or _cute_product_uses_owned_lane_reduction(cg, product, lane_var)
                )
            )
        ):
            # Complete dependent reductions before the product sum and consume
            # that sum in the once-per-tile carry update, with or without rescale.
            product_lane = lane_var
            lane_var = None
        running_sum_route = "running"
        if lane_var is not None:
            assert k_block_id is not None
            running_sum_route = _cute_running_sum_route(
                cg,
                fx_node,
                acc_node,
                has_acc=acc is not None,
                loop_state=loop_state,
                k_block_id=k_block_id,
                lane_var=lane_var,
            )
            if running_sum_route == "owned":
                product_lane = lane_var
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
                emit = statements_stack[-2].append
            else:
                emit = cg.add_statement
            if (
                dot_acc_base is not None
                and base_acc_source is not None
                and capture_base_outside_lane
            ):
                emit(
                    statement_from_string(
                        f"{dot_acc_base} = {{acc}}", acc=base_acc_source
                    )
                )
            emit(statement_from_string(f"{dot_acc} = {zero_init}"))
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
            # Tell ``hoist_warp_reduce`` this is a per-thread running sum: it
            # already combines across V-lanes, so its FINAL value is reduced
            # once after the loop rather than re-folded per lane.
            running_sums = getattr(cg.device_function, "cute_matmul_running_sums", None)
            if isinstance(running_sums, set):
                running_sums.add(dot_acc)
            if running_sum_route == "recurrence":
                # Exact only once restructured to update once per tile.
                cg.device_function.cute_state.chunk_recurrence_running_sums.add(dot_acc)
        else:
            reduction_input = cg.lift(product, dce=True, prefix="dot_product").id
        reduction_value_dtype = (
            reduction_dtype or lhs_dtype or rhs_dtype or out_dtype or torch.float32
        )
        if product_lane is not None:
            product = cg.lift(
                expr_from_string(
                    _emit_cute_owned_product_sum(
                        cg,
                        reduction_input,
                        value_dtype=reduction_value_dtype,
                        loop_state=loop_state,
                        k_block_id=k_block_id,
                        owner_lane=product_lane,
                    )
                ),
                dce=True,
                prefix="dot_sum",
            )
        else:
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
