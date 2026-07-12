"""CuTe-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Backend-specific codegen bodies live here (not in the backend-neutral
``aten_lowering`` module).  Importing this module runs the
``@<op>_lowering.register_codegen("cute")`` registrations; ``aten_lowering``
imports it at the bottom so registration keeps the same eager timing as before.
"""

from __future__ import annotations

import ast
import contextlib
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx.node import Node
from torch.fx.node import map_arg

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..aten_lowering import LoweringContext
from ..aten_lowering import _argreduce_schema
from ..aten_lowering import _env_arg
from ..aten_lowering import addmm_lowering
from ..aten_lowering import arange_default_lowering
from ..aten_lowering import argmax_lowering
from ..aten_lowering import argmin_lowering
from ..aten_lowering import baddbmm_lowering
from ..aten_lowering import bmm_lowering
from ..aten_lowering import expand_lowering
from ..aten_lowering import gather_lowering
from ..aten_lowering import iota_lowering
from ..aten_lowering import mm_lowering
from ..aten_lowering import permute_lowering
from ..aten_lowering import reshape_lowering
from ..aten_lowering import sort_lowering
from ..aten_lowering import squeeze_lowering
from ..aten_lowering import stack_lowering
from ..aten_lowering import topk_lowering
from ..aten_lowering import unsqueeze_lowering
from ..aten_lowering import view_dtype_lowering
from ..aten_lowering import view_lowering
from ..aten_lowering import where_lowering
from ..compile_environment import CompileEnvironment
from .argreduce import codegen_cute_tile_argreduce
from .cute_mma import codegen_cute_mma
from .cute_mma import codegen_cute_mma_direct_mm
from .indexing import CutePackedAffineLoad
from .indexing import CuteShapeChainView
from .indexing import CuteSortableLoad
from .indexing import is_cute_shape_chain_target
from .indexing import match_cute_affine_range_iota
from .iota_utils import cute_free_arange_indexed_dim_key
from .iota_utils import cute_iota_has_atomic_tensor_index_only_users
from .iota_utils import cute_iota_is_free_memory_index
from .matmul_fallback import _emit_cute_matmul
from .matmul_fallback import _emit_cute_matmul_n_collapse
from .matmul_utils import cute_lower_rhs_for_matmul
from .matmul_utils import cute_outer_accumulates_result
from .matmul_utils import cute_outer_accumulator_dtype
from .matmul_utils import cute_outer_accumulator_out_dtype
from .matmul_utils import cute_rematerialize_rhs_at_contraction_block
from .matmul_utils import cute_rematerialize_rhs_at_index_override
from .matmul_utils import cute_resolve_active_block_id
from .matmul_utils import cute_resolve_active_matmul_k_block_id
from .matmul_utils import cute_static_k_invariant_extent
from .matmul_utils import cute_static_mn_collapse_n_block_id
from .matmul_utils import cute_static_serial_matmul_k_extent
from .matmul_utils import cute_synthetic_lane_k_extent
from .matmul_utils import emit_cute_serial_scalar_mm_from_loads
from .matmul_utils import emit_cute_synthetic_lane_fold_mm
from .strategies import is_pure_matmul_role_lifecycle_config
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


def _requested_pure_matmul_role_lifecycle(ctx: LoweringContext) -> bool:
    return is_pure_matmul_role_lifecycle_config(ctx.cg.device_function.config)


def _requested_tcgen05_flat_role_coordinates(ctx: LoweringContext) -> bool:
    return bool(
        ctx.cg.device_function.config.get(
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False
        )
    )


def _reject_tcgen05_flat_role_coordinates_fallback() -> None:
    raise exc.BackendUnsupported(
        "cute",
        f"{TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY}=True requires "
        "active-K-loop tcgen05 MMA lowering",
    )


@where_lowering.register_codegen("cute")
def codegen_where_cute(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    cond, x, y = map_arg(node.args, lambda arg: _env_arg(ctx, arg))

    def ensure_ast(value: object) -> ast.AST:
        if isinstance(value, ast.AST):
            return value
        if isinstance(value, (int, float, bool)):
            return expr_from_string(constant_repr(value))
        raise AssertionError(f"unsupported where operand: {type(value)!r}")

    output = node.meta.get("val")
    x_ast = ensure_ast(x)
    y_ast = ensure_ast(y)
    if isinstance(output, torch.Tensor):
        x_ast = env.backend.cast_ast(x_ast, output.dtype)
        y_ast = env.backend.cast_ast(y_ast, output.dtype)
    return expr_from_string(
        env.backend.where_expr("{cond}", "{x}", "{y}"),
        cond=ensure_ast(cond),
        x=x_ast,
        y=y_ast,
    )


@unsqueeze_lowering.register_codegen("cute")
def codegen_unsqueeze_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute_reshape import resolve_cute_shape_chain_value

    # One scalar per thread — adding a unit dimension cannot change the value.
    assert not node.kwargs, "unsqueeze kwargs not supported"
    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, tensor.node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


def _shape_chain_only_users(node: Node) -> bool:
    return bool(node.users) and all(
        user.op == "call_function" and is_cute_shape_chain_target(user.target)
        for user in node.users
    )


def _cute_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    _, dim, keepdim = _argreduce_schema(node)
    return codegen_cute_tile_argreduce(
        ctx,
        node,
        reduction_type,
        dim=dim,
        keepdim=keepdim,
    )


@argmax_lowering.register_codegen("cute")
def codegen_argmax_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    return _cute_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("cute")
def codegen_argmin_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    return _cute_argreduce(ctx, node, "argmin")


@squeeze_lowering.register_codegen("cute")
def codegen_squeeze_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute_reshape import resolve_cute_shape_chain_value

    # Squeeze removes a dimension of size 1 — no data movement needed
    # since each thread still holds the same element.
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, tensor.node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


@view_lowering.register_codegen("cute")
@reshape_lowering.register_codegen("cute")
def codegen_view_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute_reshape import codegen_cute_reshape

    return codegen_cute_reshape(ctx, node)


@view_dtype_lowering.register_codegen("cute")
def codegen_view_dtype_cute(ctx: LoweringContext, node: Node) -> object:
    """Per-element bitcast through shared memory ``cute.recast_tensor``.

    CuTe DSL operates on per-thread scalars, so a dtype reinterpret has to
    round-trip a value through shared memory: write as the source dtype, then
    read the same memory through a recast view typed as the target dtype.
    """
    from .cute_reshape import _flat_index_from_coords
    from .cute_reshape import _get_dim_local_coord
    from .cute_reshape import _get_tile_shape

    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    target_dtype = node.args[1]
    assert isinstance(target_dtype, torch.dtype)

    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    if input_val.dtype.itemsize != target_dtype.itemsize:
        raise exc.BackendUnsupported(
            "cute",
            f"view.dtype with mismatched widths: "
            f"{input_val.dtype} ({input_val.dtype.itemsize} bytes) -> "
            f"{target_dtype} ({target_dtype.itemsize} bytes)",
        )

    from ..generate_ast import GenerateAST

    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    df = cg.device_function
    env = CompileEnvironment.current()
    config = df.config

    shape = _get_tile_shape(input_val, env, config)
    if not shape:
        shape = [1]
    numel = 1
    for s in shape:
        numel *= s

    src_dtype_str = env.backend.dtype_str(input_val.dtype)
    tgt_dtype_str = env.backend.dtype_str(target_dtype)

    smem_ptr = df.new_var("view_dtype_smem_ptr")
    smem = df.new_var("view_dtype_smem")
    smem_recast = df.new_var("view_dtype_smem_recast")

    coords = [_get_dim_local_coord(cg, input_val, i) for i in range(len(shape))]
    flat = _flat_index_from_coords(coords, shape) if coords else "cutlass.Int32(0)"

    cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({src_dtype_str}, {numel})"
        )
    )
    cg.add_statement(
        statement_from_string(f"{smem} = cute.make_tensor({smem_ptr}, ({numel},))")
    )
    cg.add_statement(
        statement_from_string(
            f"{smem}[{flat}] = {src_dtype_str}({{_inp}})", _inp=tensor
        )
    )
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    cg.add_statement(
        statement_from_string(
            f"{smem_recast} = cute.recast_tensor({smem}, {tgt_dtype_str})"
        )
    )

    result = df.new_var("view_dtype_value")
    cg.add_statement(statement_from_string(f"{result} = {smem_recast}[{flat}]"))
    return expr_from_string(result)


@permute_lowering.register_codegen("cute")
def codegen_permute_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute_reshape import codegen_cute_permute

    return codegen_cute_permute(ctx, node)


@stack_lowering.register_codegen("cute")
def codegen_stack_cute(ctx: LoweringContext, node: Node) -> object:
    tensors = node.args[0]
    assert isinstance(tensors, (list, tuple))
    if not tensors:
        raise ValueError("Cannot stack empty tensor list")
    if not all(isinstance(tensor, Node) for tensor in tensors):
        raise exc.BackendUnsupported("cute", "stack inputs")
    if _shape_chain_only_users(node):
        return CuteShapeChainView(node)
    # A stack materialized to a per-thread scalar is only correct when every
    # consumer reads it element-wise (each output element is one stacked
    # element), e.g. a direct store. A reduction or matmul that contracts over
    # the virtual stacked dimension cannot gather the stacked operands from a
    # single per-thread scalar and would silently produce wrong values, so only
    # materialize when the direct consumers are stores (or further shape-chain
    # ops); otherwise keep rejecting the pattern.
    from ...language import memory_ops

    if not all(
        user.op == "call_function"
        and (user.target is memory_ops.store or is_cute_shape_chain_target(user.target))
        for user in node.users
    ):
        raise exc.BackendUnsupported(
            "cute", "virtual shape-chain direct consumers are not yet supported"
        )
    from .cute_reshape import resolve_cute_shape_chain_value

    materialized = resolve_cute_shape_chain_value(ctx, node)
    if materialized is None:
        raise exc.BackendUnsupported(
            "cute", "virtual shape-chain direct consumers are not yet supported"
        )
    return materialized


@expand_lowering.register_codegen("cute")
def codegen_expand_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute_reshape import resolve_cute_shape_chain_value

    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


@bmm_lowering.register_codegen("cute")
@mm_lowering.register_codegen("cute")
def codegen_mm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    lhs_node, rhs_node = node.args[:2]
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    if k_block_id is None and packed_rhs is None:
        remat = cute_rematerialize_rhs_at_contraction_block(ctx, lhs_node, rhs_node)
        if remat is not None:
            rhs, k_block_id = remat
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    serial_k_extent = (
        None
        if k_block_id is not None or static_k_extent is not None
        else cute_static_serial_matmul_k_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if (
        static_k_extent is None
        and serial_k_extent is None
        and k_block_id is None
        and not k_is_one
    ):
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    out_dtype = node.meta["val"].dtype if "val" in node.meta else None
    outer_acc_dtype = cute_outer_accumulator_dtype(node, is_acc_none=True)
    effective_out_dtype = (
        cute_outer_accumulator_out_dtype(out_dtype, outer_acc_dtype)
        if out_dtype is not None
        else None
    )
    if node.target is torch.ops.aten.bmm.default:
        mma_result = codegen_cute_mma(ctx, node, with_acc=False)
        if mma_result is not None:
            return mma_result
    direct_mma_result = codegen_cute_mma_direct_mm(
        ctx,
        node,
        serial_k_extent=serial_k_extent,
    )
    if direct_mma_result is not None:
        if _requested_tcgen05_flat_role_coordinates(ctx):
            _reject_tcgen05_flat_role_coordinates_fallback()
        if _requested_pure_matmul_role_lifecycle(ctx):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_strategy='pure_matmul_role_lifecycle' requires the "
                "active-K-loop tcgen05 matmul lowering, not direct-mm fallback",
            )
        return direct_mma_result
    serial_result = emit_cute_serial_scalar_mm_from_loads(
        ctx,
        lhs_node,
        rhs_node,
        k_extent=serial_k_extent,
        out_dtype=effective_out_dtype,
    )
    if serial_result is not None:
        if _requested_tcgen05_flat_role_coordinates(ctx):
            _reject_tcgen05_flat_role_coordinates_fallback()
        if _requested_pure_matmul_role_lifecycle(ctx):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_strategy='pure_matmul_role_lifecycle' requires the "
                "active-K-loop tcgen05 matmul lowering, not serial scalar fallback",
            )
        return serial_result
    if serial_k_extent is not None:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe direct mm without an active K tile only supports contiguous direct-load operands",
        )
    if _requested_pure_matmul_role_lifecycle(ctx):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires aten.mm "
            "to lower through the tcgen05 K-loop path",
        )
    if _requested_tcgen05_flat_role_coordinates(ctx):
        _reject_tcgen05_flat_role_coordinates_fallback()
    lane_fold_k = cute_synthetic_lane_k_extent(ctx.cg, k_block_id)
    if lane_fold_k is not None:
        fold_result = emit_cute_synthetic_lane_fold_mm(
            ctx,
            lhs_node,
            rhs_node,
            k_extent=lane_fold_k,
            acc=None,
            out_dtype=effective_out_dtype,
            acc_dtype=None,
            lhs_dtype=lhs_node.meta["val"].dtype,
            rhs_dtype=rhs_node.meta["val"].dtype,
        )
        if fold_result is not None:
            return fold_result
        raise exc.BackendUnsupported(
            "cute",
            "CuTe synthetic-lane K matmul fold only supports direct-load operands",
        )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        accumulate_in_lane_loop=not cute_outer_accumulates_result(
            node,
            is_acc_none=True,
        ),
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        out_dtype=effective_out_dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
    )


@addmm_lowering.register_codegen("cute")
def codegen_addmm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    from .cute_mma import codegen_cute_mma

    result = codegen_cute_mma(ctx, node, with_acc=True)
    if result is not None:
        return result
    if _requested_tcgen05_flat_role_coordinates(ctx):
        _reject_tcgen05_flat_role_coordinates_fallback()
    if _requested_pure_matmul_role_lifecycle(ctx):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires the "
            "active-K-loop tcgen05 addmm lowering",
        )
    acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    acc_node = node.args[0]
    lhs_node = node.args[1]
    rhs_node = node.args[2]
    assert isinstance(acc_node, Node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    lane_fold_k = cute_synthetic_lane_k_extent(ctx.cg, k_block_id)
    if lane_fold_k is not None:
        fold_result = emit_cute_synthetic_lane_fold_mm(
            ctx,
            lhs_node,
            rhs_node,
            k_extent=lane_fold_k,
            acc=acc,
            out_dtype=node.meta["val"].dtype if "val" in node.meta else None,
            acc_dtype=acc_node.meta["val"].dtype,
            lhs_dtype=lhs_node.meta["val"].dtype,
            rhs_dtype=rhs_node.meta["val"].dtype,
        )
        if fold_result is not None:
            return fold_result
        raise exc.BackendUnsupported(
            "cute",
            "CuTe synthetic-lane K matmul fold only supports direct-load operands",
        )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
        acc_node=acc_node,
    )


def _cute_baddbmm_result_reduced_over_block(
    node: Node,
    n_block_id: int,
) -> bool:
    """Whether *node*'s collapsed result is summed away over *n_block_id*.

    The matmul's M (lhs free) and N (rhs free) axes collapse to ``n_block_id``,
    so the standard fallback would compute only the diagonal.  Folding the N
    reduction into the matmul (layout A) is correct *only* when N is genuinely
    reduced out downstream.  This guard requires:

    * ``n_block_id`` is a reduction block (allocated for a ``sum`` /
      reduction), and a reduction node over it exists somewhere in the device
      IR (the ``.sum(-1)``), and
    * the baddbmm result does not escape to any non-passthrough consumer in its
      own graph - either it is consumed only by a same-graph reduction over the
      block, or it is purely loop-carried (its only users are the graph output
      and/or pure casts), so the carried value reaches the downstream reduction.

    Returns ``False`` otherwise, leaving the (unchanged) standard path.
    """
    from ...language._tracing_ops import _new_var
    from ..host_function import HostFunction
    from ..inductor_lowering import ReductionLowering

    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    target_canonical = canonical_block_id(n_block_id)

    if not env.block_sizes[n_block_id].reduction:
        return False

    # A reduction over the (canonical) N block must exist in the device IR.
    device_ir = HostFunction.current().device_ir
    has_block_reduction = False
    for graph_info in getattr(device_ir, "graphs", ()):
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for other in graph.nodes:
            lowering = other.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and canonical_block_id(lowering.block_index) == target_canonical
            ):
                has_block_reduction = True
                break
        if has_block_reduction:
            break
    if not has_block_reduction:
        return False

    passthrough = {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.to.dtype,
        torch.ops.prims.convert_element_type.default,
        _new_var,
    }

    # Walk forward inside the baddbmm's own graph; the result must not reach any
    # non-passthrough consumer other than a reduction over the block or the
    # graph output (loop carry).
    seen: set[Node] = set()
    stack: list[Node] = [node]
    while stack:
        cur = stack.pop()
        for user in cur.users:
            if not isinstance(user, Node) or user in seen:
                continue
            seen.add(user)
            if len(seen) > 64:
                return False
            if user.op == "output":
                continue
            lowering = user.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and canonical_block_id(lowering.block_index) == target_canonical
            ):
                continue
            if user.op == "call_function" and user.target in passthrough:
                stack.append(user)
                continue
            return False
    return True


def _maybe_codegen_cute_baddbmm_n_collapse(
    ctx: LoweringContext,
    node: Node,
    *,
    lhs: ast.AST | CutePackedAffineLoad,
    acc: ast.AST,
    lhs_node: Node,
    rhs_node: Node,
    acc_node: Node,
    k_block_id: int | None,
    static_k_extent: int | None,
) -> ast.AST | None:
    """Layout (A) for a static-M==N-collapse baddbmm reduced over N.

    See ``cute_static_mn_collapse_n_block_id`` /
    ``_emit_cute_matmul_n_collapse``.  Returns ``None`` (caller keeps the
    standard path) unless the tightly-gated pattern holds: the lhs M axis and
    rhs N axis share a block id and the result is summed away over that block.
    """
    if not isinstance(lhs, ast.AST):
        return None
    n_block_id = cute_static_mn_collapse_n_block_id(ctx.cg, lhs_node, rhs_node)
    if n_block_id is None:
        return None
    if not _cute_baddbmm_result_reduced_over_block(node, n_block_id):
        return None
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)
    n_size = rhs_node.meta["val"].shape[-1]
    n_extent = size_hint(n_size) if callable(size_hint) else int(n_size)
    if not isinstance(n_extent, int) or n_extent <= 0:
        return None

    def rhs_at_n(n_var: str) -> ast.AST:
        rematerialized = cute_rematerialize_rhs_at_index_override(
            ctx, rhs_node, n_block_id, n_var
        )
        if rematerialized is None:
            raise exc.BackendUnsupported(
                "cute",
                "CuTe static-MN-collapse baddbmm could not re-materialize the rhs "
                "at the serial N index",
            )
        return rematerialized

    return _emit_cute_matmul_n_collapse(
        ctx.cg,
        lhs,
        rhs_at_n=rhs_at_n,
        n_extent=n_extent,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
    )


@baddbmm_lowering.register_codegen("cute")
def codegen_baddbmm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    from .cute_mma import codegen_cute_mma

    result = codegen_cute_mma(ctx, node, with_acc=True)
    if result is not None:
        return result
    if _requested_tcgen05_flat_role_coordinates(ctx):
        _reject_tcgen05_flat_role_coordinates_fallback()
    if _requested_pure_matmul_role_lifecycle(ctx):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires "
            "aten.baddbmm to lower through the tcgen05 K-loop path",
        )
    acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    acc_node = node.args[0]
    lhs_node = node.args[1]
    rhs_node = node.args[2]
    assert isinstance(acc_node, Node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    n_collapse_result = _maybe_codegen_cute_baddbmm_n_collapse(
        ctx,
        node,
        lhs=lhs,
        acc=acc,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
        acc_node=acc_node,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
    )
    if n_collapse_result is not None:
        return n_collapse_result
    lane_fold_k = cute_synthetic_lane_k_extent(ctx.cg, k_block_id)
    if lane_fold_k is not None:
        fold_result = emit_cute_synthetic_lane_fold_mm(
            ctx,
            lhs_node,
            rhs_node,
            k_extent=lane_fold_k,
            acc=acc,
            out_dtype=node.meta["val"].dtype if "val" in node.meta else None,
            acc_dtype=acc_node.meta["val"].dtype,
            lhs_dtype=lhs_node.meta["val"].dtype,
            rhs_dtype=rhs_node.meta["val"].dtype,
        )
        if fold_result is not None:
            return fold_result
        raise exc.BackendUnsupported(
            "cute",
            "CuTe synthetic-lane K matmul fold only supports direct-load operands",
        )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
        acc_node=acc_node,
    )


def _cute_compacted_tile_begin_lane_expr(
    ctx: LoweringContext,
    source_node: Node,
    start: object,
    step: object,
    dtype: torch.dtype,
) -> ast.AST | None:
    """Resolve ``hl.arange(block // F)`` to the tile-local lane ``lane // F``.

    Handles the compacted sub-block store ``out[tile.begin + hl.arange(block //
    F)] = split_result`` where the value is a spread/compacted tile whose element
    on lane ``t`` is the ``t // F`` piece. The arange must contribute only the
    tile-local coordinate ``lane // F`` because ``tile.begin`` is added
    explicitly; resolving it to the global ``index_var // F`` would fold the
    tile's offset in twice. Returns ``None`` unless the pattern matches.
    """
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_block_local_coord
    from .iota_utils import cute_free_arange_compacted_tile_begin_factor

    assert isinstance(ctx.cg, GenerateAST)
    cg = ctx.cg
    match = cute_free_arange_compacted_tile_begin_factor(source_node, cg)
    if match is None:
        return None
    block_id, factor = match
    local_coord = _get_block_local_coord(cg, block_id)
    if local_coord is None:
        return None
    expr = f"({local_coord}) // cutlass.Int32({factor})"
    return _wrap_iota_coord_expr(ctx, expr, start, step, dtype)


def _cute_free_arange_axis_expr(
    cg: GenerateAST,
    source_node: Node,
    length_hint: int | None,
    start: object,
    step: object,
) -> str | None:
    """Map a free/unbound ``hl.arange`` index dim onto a synthetic thread axis.

    Returns the per-thread coordinate expression (``thread_idx()[axis]``) when
    ``source_node`` is an iota that flows into a load/store index but is bound
    to no tile/reduction/grid axis. Returns ``None`` otherwise so the caller
    keeps its existing (raising) behavior — this keeps the synthetic-axis path
    a strict no-op for every already-supported arange.
    """
    if not isinstance(length_hint, int) or length_hint <= 0:
        return None
    if not cute_iota_is_free_memory_index(source_node, cg):
        return None
    # The synthetic axis is keyed by the size of the tensor dimension this
    # arange indexes. Two arange dims that address the same logical dimension (the
    # load and store ``hl.arange(k)`` over a K-sized axis) share one axis so a
    # value loaded on a lane is stored back on that lane, while a cartesian
    # ``row``/``col`` pair addressing differently-sized dims gets distinct axes.
    # ``length``/``start``/``step`` round out the key so two distinct arange dims
    # that happen to index equal-sized dims still separate.
    dim_key = cute_free_arange_indexed_dim_key(source_node, cg)
    if dim_key is None:
        return None
    key = (
        dim_key,
        length_hint,
        _arange_endpoint_key(start),
        _arange_endpoint_key(step),
    )
    return cg.allocate_cute_synthetic_arange_coord(key, length_hint)


def _arange_endpoint_key(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        return str(value._sympy_())
    return repr(value)


def _wrap_iota_coord_expr(
    ctx: LoweringContext,
    coord_expr: str,
    start: object,
    step: object,
    dtype: torch.dtype,
) -> ast.AST:
    """Apply ``start``/``step``/``dtype`` to a per-thread iota coordinate.

    Mirrors the trailing wrapping ``_cute_iota_expr`` applies to a resolved
    block-id coordinate so the synthetic-axis path produces identical
    ``start + step * coord`` arithmetic.
    """
    expr = coord_expr
    if step != 1:
        expr = f"{{step}} * ({expr})"
    if start != 0:
        expr = f"{{start}} + ({expr})"
    if dtype != torch.int32:
        expr = f"{CompileEnvironment.current().backend.dtype_str(dtype)}({expr})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
    )


def _cute_iota_expr(
    ctx: LoweringContext,
    *,
    source_node: Node,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype_arg: object = None,
) -> object:
    from ..device_ir import ForLoopGraphInfo
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_dim_local_coord
    from .cute_reshape import _grid_local_coord_expr

    assert isinstance(ctx.cg, GenerateAST)
    cg = ctx.cg
    dtype = (
        dtype_arg
        if isinstance(dtype_arg, torch.dtype)
        else CompileEnvironment.current().index_dtype
    )

    env = CompileEnvironment.current()
    length_hint: int | None = None
    if isinstance(length_arg, int):
        length_hint = length_arg
    elif isinstance(length_arg, torch.SymInt):
        length_hint = env.size_hint(length_arg)

    def active_iota_expr() -> ast.AST | None:
        active_block_ids: list[int] = []
        graph_block_ids = [
            graph_info.block_ids
            for graph_info in cg.codegen_graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            and graph_info.graph is source_node.graph
        ]
        if len(graph_block_ids) == 1:
            active_block_ids = [
                candidate
                for candidate in graph_block_ids[0]
                if cg.active_device_loops.get(candidate)
            ]
        if not active_block_ids and cg.current_grid_state is not None:
            active_block_ids = list(cg.current_grid_state.block_ids)
        if not active_block_ids:
            active_block_ids = [
                candidate
                for candidate, loops in cg.active_device_loops.items()
                if loops
            ]
        if not active_block_ids:
            return None

        def local_expr_and_extent(
            candidate: int,
        ) -> tuple[str | None, int | None]:
            loops = cg.active_device_loops.get(candidate)
            if loops:
                loop_state = loops[-1]
                thread_axis = loop_state.block_thread_axes.get(candidate)
                if thread_axis is None:
                    return None, None
                local_expr = _grid_local_coord_expr(cg, candidate, thread_axis)
                elements_per_thread_fn = getattr(
                    loop_state.strategy, "_elements_per_thread_for_block", None
                )
                elements_per_thread = (
                    elements_per_thread_fn(candidate)
                    if callable(elements_per_thread_fn)
                    else 1
                )
                if not isinstance(elements_per_thread, int):
                    return local_expr, None
                return (
                    local_expr,
                    loop_state.thread_axis_sizes.get(thread_axis, 1)
                    * elements_per_thread,
                )
            if cg.current_grid_state is not None:
                thread_axis = cg.current_grid_state.block_thread_axes.get(candidate)
                if thread_axis is None:
                    return None, None
                local_expr = _grid_local_coord_expr(cg, candidate, thread_axis)
                elements_per_thread_fn = getattr(
                    cg.current_grid_state.strategy,
                    "_elements_per_thread_for_block",
                    None,
                )
                elements_per_thread = (
                    elements_per_thread_fn(candidate)
                    if callable(elements_per_thread_fn)
                    else 1
                )
                if not isinstance(elements_per_thread, int):
                    return local_expr, None
                return (
                    local_expr,
                    cg.current_grid_state.thread_axis_sizes.get(thread_axis, 1)
                    * elements_per_thread,
                )
            return None, None

        matched: list[tuple[int, str]] = []
        for candidate in active_block_ids:
            loops = cg.active_device_loops.get(candidate)
            if loops:
                expr = loops[-1].strategy.index_var(candidate)
            elif (
                cg.current_grid_state is not None
                and candidate in cg.current_grid_state.block_ids
            ):
                expr = cg.current_grid_state.strategy.index_var(candidate)
            else:
                continue

            candidate_size = cg.device_function.resolved_block_size(candidate)
            if (
                not isinstance(candidate_size, int)
                or candidate_size <= 0
                or not isinstance(length_hint, int)
                or length_hint <= 0
            ):
                continue
            if candidate_size == length_hint:
                matched.append((candidate, expr))
            elif candidate_size % length_hint == 0:
                matched.append(
                    (candidate, f"({expr}) // {candidate_size // length_hint}")
                )
            else:
                local_expr, local_extent = local_expr_and_extent(candidate)
                if (
                    local_expr is not None
                    and isinstance(local_extent, int)
                    and local_extent > 0
                ):
                    if local_extent == length_hint:
                        matched.append((candidate, local_expr))
                    elif local_extent % length_hint == 0:
                        matched.append(
                            (
                                candidate,
                                f"({local_expr}) // {local_extent // length_hint}",
                            )
                        )
        if len(matched) != 1:
            return None
        _, expr = matched[0]
        if step != 1:
            expr = f"{{step}} * ({expr})"
        if start != 0:
            expr = f"{{start}} + ({expr})"
        if dtype != torch.int32:
            expr = f"{env.backend.dtype_str(dtype)}({expr})"
        return expr_from_string(
            expr,
            start=ctx.to_ast(start),
            step=ctx.to_ast(step),
        )

    block_id = env.resolve_block_id(length_arg)
    original_block_id = block_id
    if block_id is None:
        if (affine_range := match_cute_affine_range_iota(source_node)) is not None:
            return affine_range
    if (
        compacted := _cute_compacted_tile_begin_lane_expr(
            ctx, source_node, start, step, dtype
        )
    ) is not None:
        return compacted
    if "val" in source_node.meta:
        fake_val = source_node.meta["val"]
        if isinstance(fake_val, torch.Tensor) and fake_val.ndim == 1:
            with contextlib.suppress(Exception):
                length_hint = int(fake_val.shape[0])
            local_coord = _get_dim_local_coord(cg, fake_val, 0)
            if local_coord != "cutlass.Int32(0)":
                expr = local_coord
                if step != 1:
                    expr = f"{{step}} * ({expr})"
                if start != 0:
                    expr = f"{{start}} + ({expr})"
                if dtype != torch.int32:
                    expr = f"{env.backend.dtype_str(dtype)}({expr})"
                return expr_from_string(
                    expr,
                    start=ctx.to_ast(start),
                    step=ctx.to_ast(step),
                )
            if block_id is None:
                block_id = env.resolve_block_id(fake_val.shape[0])
            if block_id is None and cg.current_grid_state is not None:
                grid_candidates = [
                    candidate
                    for candidate in cg.current_grid_state.block_ids
                    if isinstance(length_hint, int)
                    and isinstance(
                        cg.device_function.resolved_block_size(candidate),
                        int,
                    )
                    and cg.device_function.resolved_block_size(candidate) == length_hint
                ]
                if len(grid_candidates) == 1:
                    block_id = grid_candidates[0]
    if block_id is None:
        if (active_expr := active_iota_expr()) is not None:
            return active_expr
        if (
            cute_iota_has_atomic_tensor_index_only_users(source_node, cg)
            and isinstance(start, int)
            and isinstance(step, int)
        ):
            return expr_from_string(
                "cute.make_identity_tensor({length})",
                length=ctx.to_ast(length_arg),
            )
        if (
            synthetic := _cute_free_arange_axis_expr(
                cg, source_node, length_hint, start, step
            )
        ) is not None:
            return _wrap_iota_coord_expr(ctx, synthetic, start, step, dtype)
        raise exc.BackendUnsupported(
            "cute",
            "hl.arange() requires an active tile/reduction axis in cute kernels",
        )
    resolved_block_id = env.resolve_codegen_block_id(block_id, cg, source_node.graph)
    candidate_block_ids = [resolved_block_id]
    if (
        original_block_id is not None
        and original_block_id != resolved_block_id
        and original_block_id not in candidate_block_ids
    ):
        candidate_block_ids.append(original_block_id)

    expr: str | None = None
    active_block_id: int | None = None
    for candidate_block_id in candidate_block_ids:
        loops = cg.active_device_loops.get(candidate_block_id)
        if loops:
            expr = loops[-1].strategy.index_var(candidate_block_id)
            active_block_id = candidate_block_id
            break
        if (
            cg.current_grid_state is not None
            and candidate_block_id in cg.current_grid_state.block_ids
        ):
            expr = cg.current_grid_state.strategy.index_var(candidate_block_id)
            active_block_id = candidate_block_id
            break
    block_id = resolved_block_id if active_block_id is None else active_block_id

    if expr is None:
        thread_axis: int | None = None
        if cg.current_grid_state is not None:
            thread_axis = cg.current_grid_state.block_thread_axes.get(block_id)
        if thread_axis is None:
            for loops_for_block in cg.active_device_loops.values():
                for loop_state in loops_for_block:
                    block_axes = getattr(loop_state, "block_thread_axes", {})
                    if isinstance(block_axes, dict) and block_id in block_axes:
                        thread_axis = block_axes[block_id]
                        break
                if thread_axis is not None:
                    break
        if thread_axis is not None:
            expr = _grid_local_coord_expr(cg, block_id, thread_axis)
        elif (active_expr := active_iota_expr()) is not None:
            return active_expr
        elif (
            cute_iota_has_atomic_tensor_index_only_users(source_node, cg)
            and isinstance(start, int)
            and isinstance(step, int)
        ):
            return expr_from_string(
                "cute.make_identity_tensor({length})",
                length=ctx.to_ast(length_arg),
            )
        elif (
            synthetic := _cute_free_arange_axis_expr(
                cg, source_node, length_hint, start, step
            )
        ) is not None:
            return _wrap_iota_coord_expr(ctx, synthetic, start, step, dtype)
        else:
            raise exc.BackendUnsupported(
                "cute",
                f"hl.arange() axis block_id={block_id} is not active in this scope",
            )
    if step != 1:
        expr = f"{{step}} * ({expr})"
    if start != 0:
        expr = f"{{start}} + ({expr})"
    if dtype != torch.int32:
        expr = f"{env.backend.dtype_str(dtype)}({expr})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
    )


@iota_lowering.register_codegen("cute")
def codegen_iota_cute(ctx: LoweringContext, node: Node) -> object:
    return _cute_iota_expr(
        ctx,
        source_node=node,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype_arg=node.kwargs.get("dtype"),
    )


@arange_default_lowering.register_codegen("cute")
def codegen_arange_default_cute(ctx: LoweringContext, node: Node) -> object:
    return _cute_iota_expr(
        ctx,
        source_node=node,
        length_arg=node.args[0],
        dtype_arg=node.kwargs.get("dtype"),
    )


def _sort_args(node: Node) -> tuple[int, bool]:
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )
    assert isinstance(dim, int), f"sort dim must be int, got {type(dim)}"
    assert isinstance(descending, bool), (
        f"sort descending must be bool, got {type(descending)}"
    )
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim
    if dim < 0:
        dim = ndim + dim
    assert dim == ndim - 1, (
        f"sort only supports sorting on last dimension, got dim={dim}"
    )
    return dim, descending


def _emit_cute_rank_sort(
    ctx: LoweringContext,
    load: CuteSortableLoad,
    input_tensor: torch.Tensor,
    *,
    descending: bool,
    k: int | None = None,
) -> tuple[ast.AST, ast.AST]:
    env = CompileEnvironment.current()
    fn = ctx.cg.device_function
    n = input_tensor.shape[-1]
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    if not isinstance(n_hint, int):
        raise exc.BackendUnsupported("cute", "dynamic sort extent")
    dtype_str = env.backend.dtype_str(load.dtype)
    index_dtype = env.backend.dtype_str(env.index_dtype)
    out_pos = fn.new_var("sort_out_pos")
    sorted_vals = fn.new_var("sorted_vals")
    sorted_indices = fn.new_var("sorted_indices")
    candidate = fn.new_var("sort_k")
    probe = fn.new_var("sort_j")
    candidate_rank = fn.new_var("sort_rank")
    candidate_value = fn.new_var("sort_candidate")
    probe_value = fn.new_var("sort_probe")
    before = fn.new_var("sort_before")
    selected = fn.new_var("sort_selected")

    ctx.cg.add_statement(
        statement_from_string(
            f"{out_pos} = {index_dtype}({load.index_exprs[load.sort_index_pos]})"
        )
    )
    ctx.cg.add_statement(statement_from_string(f"{sorted_vals} = {dtype_str}(0)"))
    ctx.cg.add_statement(statement_from_string(f"{sorted_indices} = {index_dtype}(0)"))

    cmp_op = ">" if descending else "<"

    def indexed_load(index: str) -> str:
        index_exprs = list(load.index_exprs)
        index_exprs[load.sort_index_pos] = index
        expr = f"{load.tensor_name}[{', '.join(index_exprs)}]"
        if load.mask_expr is not None:
            return f"({expr} if {load.mask_expr} else {dtype_str}(0))"
        return expr

    mask_suffix = f" and {out_pos} < {k}" if k is not None else ""
    ctx.cg.add_statement(
        statement_from_string(
            "\n".join(
                [
                    f"for {candidate} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    f"    {candidate_value} = {indexed_load(candidate)}",
                    f"    {candidate_rank} = {index_dtype}(0)",
                    f"    for {probe} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    f"        {probe_value} = {indexed_load(probe)}",
                    f"        {before} = ({probe_value} {cmp_op} {candidate_value}) or (({probe_value} == {candidate_value}) and ({probe} < {candidate}))",
                    f"        {candidate_rank} = {candidate_rank} + ({index_dtype}(1) if {before} else {index_dtype}(0))",
                    f"    {selected} = ({candidate_rank} == {out_pos}{mask_suffix})",
                    f"    {sorted_vals} = {candidate_value} if {selected} else {sorted_vals}",
                    f"    {sorted_indices} = {index_dtype}({candidate}) if {selected} else {sorted_indices}",
                ]
            )
        )
    )
    return expr_from_string(sorted_vals), expr_from_string(sorted_indices)


@sort_lowering.register_codegen("cute")
def codegen_sort_cute(ctx: LoweringContext, node: Node) -> object:
    _, descending = _sort_args(node)
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_tensor = input_node.meta["val"]
    load = _env_arg(ctx, input_node)
    if not isinstance(load, CuteSortableLoad):
        load = input_node.meta.get("cute_sortable_load")
    if not isinstance(load, CuteSortableLoad):
        raise exc.BackendUnsupported("cute", "torch.sort input")
    node.meta["cute_sort_load"] = load
    node.meta["cute_sort_descending"] = descending
    return _emit_cute_rank_sort(ctx, load, input_tensor, descending=descending)


@gather_lowering.register_codegen("cute")
def codegen_gather_cute(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "gather does not support keyword arguments"
    assert len(node.args) == 3, f"gather expects 3 arguments, got {len(node.args)}"

    input_node = node.args[0]
    dim = node.args[1]
    index_node = node.args[2]
    assert isinstance(input_node, Node)
    assert isinstance(dim, int)
    assert isinstance(index_node, Node)

    from ...language.memory_ops import _cute_combined_mask
    from ...language.memory_ops import _cute_index_exprs
    from ...language.memory_ops import load
    from ..inductor_lowering import CodegenState

    if input_node.target is not load:
        raise exc.BackendUnsupported("cute", "torch.gather input")
    tensor_node = input_node.args[0]
    if not isinstance(tensor_node, Node):
        raise exc.BackendUnsupported("cute", "torch.gather tensor input")
    tensor = tensor_node.meta["val"]
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", "torch.gather tensor input")
    input_subscript = input_node.args[1]
    if not isinstance(input_subscript, (list, tuple)):
        raise exc.BackendUnsupported("cute", "torch.gather input subscript")

    ndim = len(input_subscript)
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise exc.InvalidReductionDim(dim)

    proxy_subscript = cast(
        "list[object]",
        list(map_arg(tuple(input_subscript), lambda arg: arg.meta["val"])),
    )
    ast_subscript = cast(
        "list[object]",
        list(map_arg(tuple(input_subscript), lambda arg: _env_arg(ctx, arg))),
    )
    index_ast = _env_arg(ctx, index_node)
    assert isinstance(index_ast, ast.AST)
    proxy_subscript[dim] = index_node.meta["val"]
    ast_subscript[dim] = index_ast

    from ..generate_ast import GenerateAST

    if not isinstance(ctx.cg, GenerateAST):
        raise exc.NotAllowedInHelperFunction

    state = CodegenState(ctx.cg, fx_node=node, env=ctx.env)
    index_exprs = _cute_index_exprs(
        state,
        proxy_subscript,
        ast_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    tensor_name = ctx.cg.device_function.tensor_arg(tensor).name
    load_expr = f"{tensor_name}[{', '.join(index_exprs)}]"
    mask_expr = _cute_combined_mask(state, proxy_subscript, None, tensor=tensor)
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else cutlass.Boolean(0))")
    if mask_expr is None:
        return expr_from_string(load_expr)
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")


def _topk_args(node: Node) -> tuple[int, int, bool]:
    k = node.args[1]
    assert isinstance(k, int), f"topk k must be int, got {type(k)}"
    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    largest = node.args[3] if len(node.args) > 3 else node.kwargs.get("largest", True)
    assert isinstance(dim, int), f"topk dim must be int, got {type(dim)}"
    assert isinstance(largest, bool), f"topk largest must be bool, got {type(largest)}"
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim
    if dim < 0:
        dim = ndim + dim
    assert dim == ndim - 1, f"topk only supports the last dimension, got dim={dim}"
    return k, dim, largest


@topk_lowering.register_codegen("cute")
def codegen_topk_cute(ctx: LoweringContext, node: Node) -> object:
    k, _, largest = _topk_args(node)
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_tensor = input_node.meta["val"]
    load = _env_arg(ctx, input_node)
    if not isinstance(load, CuteSortableLoad):
        load = input_node.meta.get("cute_sortable_load")
    if not isinstance(load, CuteSortableLoad):
        raise exc.BackendUnsupported("cute", "torch.topk input")
    node.meta["cute_topk_lane_expr"] = load.index_exprs[load.sort_index_pos]
    node.meta["cute_topk_k"] = k
    return _emit_cute_rank_sort(ctx, load, input_tensor, descending=largest, k=k)
