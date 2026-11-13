"""The host-side metadata of a rendered CuTe kernel.

``post_kernel_metadata_statements`` resolves the wrapper plans the lowering
registered (``CuteCodegenState.wrapper_plans``) against the launcher's argument
order, records them on the compile (``CuteEnvState.resolved_wrapper_plans``, the
runtime hooks' copy) and renders the module statements that follow the kernel
(``<kernel>._helion_cute_wrapper_plans``, ``<kernel>._helion_cute_cluster_shape``);
``can_bake_tensor_shapes`` decides the ``_helion_cute_disable_bake_tensor_shapes``
exemption at the head of the host wrapper.  Both run from
``CuteBackend.finalize_host_statements``, before its host passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import ConstExprArg
from .state import cute_codegen_state
from .state import cute_env_state

if TYPE_CHECKING:
    import ast

    from ..generate_ast import GenerateAST
    from ..host_function import HostFunction

# The plain (non-grouped) tcgen05 GEMM wrapper plans: A/B TMA loads and the D
# TMA store over the problem tensors themselves.  Baking the tensor layouts
# into the wrapper schema makes the TMA coordinate tensors' strides immediates
# (the launcher's full-tile guard still decides per plan); with runtime
# layouts every tcgen05 kernel fetched them from two constant-bank lines (LDCU
# misses) and multiplied them into its TMA coordinates on the TMA-load warp's
# post-release critical path.  The grouped variants (device tensormaps, shared
# RHS, full-allocation operands, device split sizes, bounded output shifts)
# re-point descriptors in-kernel and keep the runtime layouts.
_PLAIN_TCGEN05_AB_PLAN_GROUPED_KEYS = frozenset(
    {
        "shared_rhs",
        "fixed_ab_tensormaps",
        "dynamic_ab_tensormaps",
        "full_allocation_a",
        "full_allocation_b",
        "device_split_sizes",
        "fixed_grouped_b_rank3",
        "source_m_tile",
    }
)
_PLAIN_TCGEN05_D_PLAN_GROUPED_KEYS = frozenset(
    {"fixed_tensormap", "bounded_output_shift", "dynamic_d_tensormap"}
)


def _plain_tcgen05_plan_bakes(plan: dict[str, object]) -> bool:
    kind = plan.get("kind")
    if kind == "tcgen05_ab_tma":
        return not any(plan.get(key) for key in _PLAIN_TCGEN05_AB_PLAN_GROUPED_KEYS)
    if kind == "tcgen05_d_tma":
        return not any(plan.get(key) for key in _PLAIN_TCGEN05_D_PLAN_GROUPED_KEYS)
    return False


# Plans exempting a declared matmul from the compiler's shape-bake veto.
# Runtime launch_descriptor.wrapper_plan_bakes_tensor_shapes separately checks
# the launch plan (including partial TMA tiles); membership here is not enough.
_BAKING_PLAN_KINDS = frozenset(
    {
        "helion_small_biased_attention",
        "helion_flash_row_mma",
        "helion_warp_mma_gemm",
        "helion_register_subblocks",
        "helion_flash",
        "helion_flash_gated",
        "helion_flash_gated_row_mma",
        "chunk_prepare_tma",
        "chunk_recurrence_sm100",
        "chunk_recurrence_warp_dv4",
        "gdn_recurrence_sm100",
        "helion_flash_bwd",
        "helion_flash_bwd_row_mma",
        "gathered_mma_tma",
        "block_scaled_mma",
        "helion_mma_dag_tma",
        "helion_mma_dag_alias",
    }
)

# The wrapper-plan entries naming a tensor argument; each is rewritten to the
# ``_idx`` position of that argument in the launcher's argument order.
_PLAN_TENSOR_NAME_KEYS: tuple[str, ...] = (
    "lhs_name",
    "rhs_name",
    "lhs_scale_name",
    "rhs_scale_name",
    "workspace_name",
    "c_name",
    "d_name",
    "q_name",
    "k_name",
    "g_name",
    "w_name",
    "u_name",
    "h_name",
    "beta_name",
    "a_log_name",
    "dt_name",
    "cu_seqlens_name",
    "cu_chunks_name",
    "chunk_to_seq_name",
    "kd_name",
    "qd_name",
    "ak_name",
    "aq_name",
    "gt_name",
    "v_name",
    "o_name",
    "out_name",
    "state_name",
    "do_name",
    "delta_name",
    "dq_name",
    "dk_name",
    "dv_name",
    "lse_name",
    "row_max_name",
    "bias_name",
    "alibi_name",
    "document_name",
    "document_range_name",
    "row_kv_begin_name",
    "row_kv_begin_work_name",
    "scale_name",
    "layout_name",
    "n_sizes_name",
    "k_sizes_name",
    "direct_pointers_name",
    "direct_strides_name",
    "m_extent_name",
    "epi_aux0_name",
    "epi_aux1_name",
    "epi_aux2_name",
    "epi_aux3_name",
    # cute_mma_dag tcgen05 tensor maps (one per operand).
    *(f"tma{index}_name" for index in range(16)),
)


def can_bake_tensor_shapes(cg: GenerateAST) -> bool:
    """Whether the host wrapper may bake the tensor layouts into the kernel
    schema: no matmul lowering, wrapper plans that all bake, a completely
    lowered collective register chain, or complete static collectives."""
    state = cute_codegen_state(cg)
    if not state.uses_matmul:
        return True
    if state.wrapper_plans:
        return all(
            _plain_tcgen05_plan_bakes(plan) or plan.get("kind") in _BAKING_PLAN_KINDS
            for plan in state.wrapper_plans
        )
    lanes = cg.device_function.cute_state.lanes
    return (
        lanes.collective_register_chain_lowered
        and state.matmul_declaration_count >= 2
        and state.matmul_declaration_count == len(lanes.collective_mma_sites)
    ) or has_complete_static_collectives(cg)


def has_complete_static_collectives(cg: GenerateAST) -> bool:
    """Whether every matmul declared under ``cute_collective_static_layouts``
    lowered as a collective with static layouts."""
    state = cute_codegen_state(cg)
    lanes = cg.device_function.cute_state.lanes
    return (
        cast(
            "bool",
            cg.device_function.config.get("cute_collective_static_layouts", False),
        )
        and not state.wrapper_plans
        and lanes.collective_mma_static_layouts
        and state.matmul_declaration_count > 0
        and state.matmul_declaration_count == len(lanes.collective_mma_sites)
    )


def post_kernel_metadata_statements(
    cg: GenerateAST, host: HostFunction
) -> list[ast.AST]:
    """Resolve the wrapper plans against the launcher's argument order, record
    them on the compile and return the module statements after the kernel."""
    fn = cg.device_function
    plans = cute_codegen_state(cg).wrapper_plans
    resolved_plans = _resolve_plan_arg_positions(cg, host, plans) if plans else []
    cute_env_state(CompileEnvironment.current()).resolved_wrapper_plans = resolved_plans
    statements: list[ast.AST] = []
    if plans:
        statements.append(
            statement_from_string(
                f"{fn.name}._helion_cute_wrapper_plans = {resolved_plans!r}"
            )
        )
    if fn.cute_state.tcgen05.cluster_shape is not None:
        statements.append(
            statement_from_string(
                f"{fn.name}._helion_cute_cluster_shape = {fn.cute_state.tcgen05.cluster_shape!r}"
            )
        )
    return statements


def _resolve_plan_arg_positions(
    cg: GenerateAST, host: HostFunction, plans: list[dict[str, object]]
) -> list[dict[str, object]]:
    launcher_arg_positions = {
        arg.name: idx
        for idx, arg in enumerate(
            [
                arg
                for arg in cg.device_function.sorted_args()
                if not (isinstance(arg, ConstExprArg) and arg.host_str() != arg.name)
            ]
        )
    }
    host_arg_positions = {arg.arg: idx for idx, arg in enumerate(host.args.args)}
    resolved_plans: list[dict[str, object]] = []
    for plan in plans:
        resolved = dict(plan)
        # The ``key[:-5] + "_idx"`` substring rewrite turns each tensor name
        # into a positional index resolved against the device function's
        # sorted-arg ordering so the runtime launcher can identify the tensor
        # args.
        for key in _PLAN_TENSOR_NAME_KEYS:
            if key in resolved:
                arg_name = str(resolved.pop(key))
                resolved[key[:-5] + "_idx"] = launcher_arg_positions[arg_name]
                if key in {"layout_name", "n_sizes_name"} and (
                    arg_name in host_arg_positions
                ):
                    resolved[key[:-5] + "_bind_idx"] = host_arg_positions[arg_name]
        # The cute_mma_dag alias guard names every tensor argument of its
        # program (the names stay for the launcher's error message).
        tensor_names = resolved.get("tensor_names")
        if isinstance(tensor_names, list):
            resolved["tensor_idx"] = [
                launcher_arg_positions[str(name)] for name in tensor_names
            ]
        resolved_plans.append(resolved)
    return resolved_plans
