from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..dtype_utils import cast_ast
from ..matmul_utils import _needs_f32_accumulator
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms

if TYPE_CHECKING:
    from ..helper_function import CodegenInterface


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
    backend = CompileEnvironment.current().backend
    dtype_str = backend.dtype_str(value_dtype)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_size = partials_size + results_size
    smem_ptr_var = cg.device_function.new_var("dot_reduce_smem_ptr")
    smem_var = cg.device_function.new_var("dot_reduce_smem")
    group_id_var = cg.device_function.new_var("dot_group_id")
    lane_in_warp_var = cg.device_function.new_var("dot_lane_in_warp")
    warp_in_group_var = cg.device_function.new_var("dot_warp_in_group")
    partials_base_var = cg.device_function.new_var("dot_partials_base")
    results_base_var = cg.device_function.new_var("dot_results_base")
    cg.add_statement(f"{smem_ptr_var} = cute.arch.alloc_smem({dtype_str}, {smem_size})")
    cg.add_statement(f"{smem_var} = cute.make_tensor({smem_ptr_var}, ({smem_size},))")
    cg.add_statement(f"{group_id_var} = ({lane_var}) // {group_span}")
    cg.add_statement(f"{lane_in_warp_var} = ({lane_var}) % 32")
    cg.add_statement(f"{warp_in_group_var} = ({lane_in_group_var}) // 32")
    cg.add_statement(
        f"{partials_base_var} = ({group_id_var}) * {pre * warps_per_group}"
    )
    cg.add_statement(f"{results_base_var} = {partials_size} + ({group_id_var}) * {pre}")

    for p in range(pre):
        masked_input_var = cg.device_function.new_var(f"dot_masked_input_{p}")
        warp_partial_var = cg.device_function.new_var(f"dot_warp_partial_{p}")
        partial_idx_var = cg.device_function.new_var(f"dot_partial_idx_{p}")
        stage2_input_var = cg.device_function.new_var(f"dot_stage2_input_{p}")
        group_result_var = cg.device_function.new_var(f"dot_group_result_{p}")
        cg.add_statement(
            f"{masked_input_var} = ({input_name}) if ({lane_mod_pre_var}) == {p} else ({identity_expr})"
        )
        cg.add_statement(
            f"{warp_partial_var} = {backend.reduction_expr(masked_input_var, 'sum', 0, threads_in_group=32)}"
        )
        cg.add_statement(
            f"{partial_idx_var} = ({partials_base_var}) + {p * warps_per_group} + ({warp_in_group_var})"
        )
        cg.add_statement(
            statement_from_string(
                f"""if ({lane_in_warp_var}) == 0:
    {smem_var}[{partial_idx_var}] = {warp_partial_var}"""
            )
        )
        cg.add_statement("cute.arch.sync_threads()")
        cg.add_statement(
            statement_from_string(
                f"""if ({warp_in_group_var}) == 0:
    {stage2_input_var} = {smem_var}[({partials_base_var}) + {p * warps_per_group} + ({lane_in_warp_var})] if ({lane_in_warp_var}) < {warps_per_group} else ({identity_expr})
    {group_result_var} = {backend.reduction_expr(stage2_input_var, "sum", 0, threads_in_group=32)}
    if ({lane_in_warp_var}) == 0:
        {smem_var}[({results_base_var}) + {p}] = {group_result_var}"""
            )
        )
        cg.add_statement("cute.arch.sync_threads()")

    result_var = cg.device_function.new_var("dot_reduce_result")
    cg.add_statement(
        f"{result_var} = {smem_var}[({results_base_var}) + ({lane_mod_pre_var})]"
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
    backend = CompileEnvironment.current().backend
    dtype_str = backend.dtype_str(value_dtype)
    smem_size = num_threads + group_count * pre
    smem_ptr_var = cg.device_function.new_var("dot_reduce_smem_ptr")
    smem_var = cg.device_function.new_var("dot_reduce_smem")
    group_base_var = cg.device_function.new_var("dot_group_base")
    group_id_var = cg.device_function.new_var("dot_group_id")
    result_base_var = cg.device_function.new_var("dot_result_base")
    cg.add_statement(f"{smem_ptr_var} = cute.arch.alloc_smem({dtype_str}, {smem_size})")
    cg.add_statement(f"{smem_var} = cute.make_tensor({smem_ptr_var}, ({smem_size},))")
    cg.add_statement(f"{group_base_var} = ({lane_var}) - ({lane_in_group_var})")
    cg.add_statement(f"{group_id_var} = ({lane_var}) // {group_span}")
    cg.add_statement(f"{result_base_var} = {num_threads} + ({group_id_var}) * {pre}")

    for p in range(pre):
        masked_input_var = cg.device_function.new_var(f"dot_masked_input_{p}")
        cg.add_statement(
            f"{masked_input_var} = ({input_name}) if ({lane_mod_pre_var}) == {p} else ({identity_expr})"
        )
        cg.add_statement(f"{smem_var}[{lane_var}] = {masked_input_var}")
        cg.add_statement("cute.arch.sync_threads()")
        stride = 1
        while stride < group_span:
            cond = (
                f"(({lane_in_group_var}) % {stride * 2}) == 0"
                f" and ({lane_in_group_var}) + {stride} < {group_span}"
            )
            lhs = f"{smem_var}[{lane_var}]"
            rhs = f"{smem_var}[({group_base_var}) + ({lane_in_group_var}) + {stride}]"
            combined = backend.reduction_combine_expr("sum", lhs, rhs, value_dtype)
            cg.add_statement(
                statement_from_string(
                    f"""if {cond}:
    {smem_var}[{lane_var}] = {combined}"""
                )
            )
            cg.add_statement("cute.arch.sync_threads()")
            stride *= 2

        cg.add_statement(
            statement_from_string(
                f"""if ({lane_in_group_var}) == 0:
    {smem_var}[({result_base_var}) + {p}] = {smem_var}[{lane_var}]"""
            )
        )
        cg.add_statement("cute.arch.sync_threads()")

    result_var = cg.device_function.new_var("dot_reduce_result")
    cg.add_statement(
        f"{result_var} = {smem_var}[({result_base_var}) + ({lane_mod_pre_var})]"
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
        return backend.reduction_expr(input_name, "sum", 0, threads_in_group=1)

    reduce_extent = axis_sizes.get(thread_axis, 1)
    if reduce_extent <= 1:
        return input_name

    pre = 1
    for axis in range(thread_axis):
        pre *= axis_sizes.get(axis, 1)
    if pre <= 1:
        return backend.reduction_expr(
            input_name, "sum", 0, threads_in_group=reduce_extent
        )

    lane_expr = backend.thread_linear_index_expr(axis_sizes)
    if lane_expr is None:
        return backend.reduction_expr(
            input_name, "sum", 0, threads_in_group=reduce_extent
        )

    num_threads = 1
    for size in axis_sizes.values():
        num_threads *= size
    actual_threads = 1
    for size in getattr(cg, "max_thread_block_dims", ()):
        actual_threads *= max(size, 1)
    if actual_threads > 0 and num_threads > actual_threads:
        return backend.reduction_expr(
            input_name, "sum", 0, threads_in_group=reduce_extent
        )

    identity_expr = f"{backend.dtype_str(value_dtype)}(0)"
    group_span = pre * reduce_extent
    if group_span <= 32:
        lane_mod_pre = f"((({lane_expr}) % {group_span}) % {pre})"
        reduced_terms: list[str] = []
        for p in range(pre):
            masked_input_var = cg.device_function.new_var(f"dot_masked_input_{p}")
            reduced_var = cg.device_function.new_var(f"dot_reduced_{p}")
            cg.add_statement(
                f"{masked_input_var} = ({input_name}) if ({lane_mod_pre}) == {p} else ({identity_expr})"
            )
            cg.add_statement(
                f"{reduced_var} = {backend.reduction_expr(masked_input_var, 'sum', 0, threads_in_group=group_span)}"
            )
            reduced_terms.append(reduced_var)
        selected_result = reduced_terms[0]
        for p, reduced in enumerate(reduced_terms[1:], start=1):
            selected_result = (
                f"({reduced}) if ({lane_mod_pre}) == {p} else ({selected_result})"
            )
        return selected_result

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
) -> ast.AST:
    """Build a CuTe matmul fallback using a cross-thread reduction over K."""
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
        rhs_terms = tuple(cast_ast(term, reduction_dtype) for term in rhs_terms)
        lhs_terms = tuple(cast_ast(term, reduction_dtype) for term in lhs_terms)
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
