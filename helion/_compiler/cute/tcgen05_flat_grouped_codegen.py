"""Complete native candidate composed at the ordinary host launch.

This API takes an explicit schedule choice, proves the original IR and binding, and
otherwise returns the very same binding's ordinary generated module. The typed
plan checks complete shared storage and physical roles; the resulting callable
uses the ordinary CuTe compiler, resource checks and launch-cache machinery.
"""

from __future__ import annotations

import ast
import hashlib
from typing import TYPE_CHECKING
from typing import cast

from ._ast_pass_utils import _bound_names
from .tcgen05_flat_grouped_config import CONFIG_KEYS
from .tcgen05_flat_grouped_config import PREFIX_SCAN_KEY
from .tcgen05_flat_grouped_config import RESIDENT_CTAS_KEY
from .tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from .tcgen05_flat_grouped_kernel import render_flat_grouped_rna_kernel
from .tcgen05_flat_grouped_plan import flat_grouped_rna_schedule
from .tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from .tcgen05_grouped_descriptors import DYNAMIC
from .tcgen05_tma_rn import CONVERSION_KEY
from .tcgen05_tma_rn import TMA_RN

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..host_function import HostFunction
    from .tcgen05_flat_grouped_ir import FlatGroupedIRProof
    from .tcgen05_flat_grouped_plan import FlatGroupedRnaSchedule


def flat_grouped_wrapper_plans(
    schedule: FlatGroupedRnaSchedule,
) -> list[dict[str, object]]:
    plans: list[dict[str, object]] = [
        {
            "kind": "tcgen05_grouped_tma_rn"
            if schedule.tma_rn
            else "tcgen05_grouped_rna",
            "bm": 128,
            "bn": 128,
            "bk": schedule.block_k,
            "cluster_m": 1,
            "cluster_n": 1,
            "ab_stage_count": schedule.ab_stages,
            "input_dtype": "cutlass.TFloat32",
            "acc_dtype": "cutlass.Float32",
            "operand_transform": "tf32_tma_rn" if schedule.tma_rn else "tf32_rna",
            "converter_warps": schedule.roles.converter_warps,
            "scheduler_self_consumer": False,
            "provider": "typed_flat_element_bases_v1",
            "m_size": schedule.rows,
            "n_size": schedule.columns,
            "k_total_size": schedule.reduction,
            "kernel_args": ["tma_atom_a", "tma_tensor_a", "tma_atom_b", "tma_tensor_b"],
            "orientation": "nm",
            "a_k_major": False,
            "b_k_major": True,
            "lhs_rank3_grouped_nt": True,
            "dynamic_ab_tensormaps": True,
            "dynamic_ab_tensormap_rank": 2,
            "lhs_idx": 2,
            "rhs_idx": 1,
            "flat_provider": schedule.provider.cache_identity(),
            "shared_storage": schedule.storage.cache_identity,
        },
        {
            "kind": "tcgen05_d_tma",
            "bm": 128,
            "bn": 128,
            "c_stage_count": 2,
            "output_dtype": "cutlass.Float32",
            "kernel_args": ["tcgen05_tma_store_atom", "tcgen05_tma_store_tensor"],
            "orientation": "nm",
            "rank3_mnl_tensor": True,
            "epi_tile_m": 128,
            "epi_tile_n": 32,
            "d_store_box_n": 32,
            "d_idx": 4,
        },
    ]

    if schedule.block_prefix is not None:
        plans[0]["grouped_prefix_scan"] = schedule.block_prefix.cache_identity()
    if schedule.descriptors is not None:
        plans[0]["dynamic_ab_tensormaps"] = False
        plans[0]["fixed_ab_tensormaps"] = True
        plans[0]["fixed_grouped_b_rank3"] = True
        for plan in plans:
            plan["wrapped_grouped_descriptors"] = schedule.descriptors.cache_identity()
    return plans


def flat_grouped_kernel_module(schedule: FlatGroupedRnaSchedule) -> str:
    imports = """from __future__ import annotations
import cutlass
import cutlass.cute as cute
from helion._compiler.cute.tcgen05_operand_pipeline import make_converted_operand_pipeline
from helion._compiler.cute.tcgen05_operand_pipeline import convert_and_publish_operand_stage
from helion._compiler.cute.tcgen05_flattened_prefix import build_flat_tile_prefix_warp as build_flat_tile_prefix
from helion._compiler.cute.tcgen05_flattened_prefix import resolve_flat_nm_work
"""
    if schedule.tma_rn:
        imports = imports.replace(
            "from helion._compiler.cute.tcgen05_operand_pipeline import make_converted_operand_pipeline\n"
            "from helion._compiler.cute.tcgen05_operand_pipeline import convert_and_publish_operand_stage\n",
            "",
        )
    if schedule.block_prefix is not None:
        imports = imports.replace(
            "build_flat_tile_prefix_warp", "build_flat_tile_prefix_block"
        )
    return (
        imports
        + render_flat_grouped_rna_kernel(schedule)
        + "_helion_native_grouped_rna._helion_cute_wrapper_plans = "
        + repr(flat_grouped_wrapper_plans(schedule))
        + "\n_helion_native_grouped_rna._helion_cute_disable_bake_tensor_shapes = True\n"
    )


def generate_flat_grouped_native_candidate(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    *,
    converter_warps: int,
    block_k: int,
    ab_stages: int,
    shared_capacity: int,
) -> ast.Module:
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import generate_ast

    env = CompileEnvironment.current()
    with func:
        proof = prove_flat_grouped_rna(env, func.device_ir)
        schedule = (
            flat_grouped_rna_schedule(
                env,
                proof,
                converter_warps=converter_warps,
                block_k=block_k,
                ab_stages=ab_stages,
                shared_capacity=shared_capacity,
                descriptor_policy=cast("str", config.get(DESCRIPTOR_KEY, DYNAMIC)),
                tma_rn=config.get(CONVERSION_KEY) == TMA_RN,
                resident_ctas=cast("int", config.get(RESIDENT_CTAS_KEY, 1)),
                prefix_scan=cast("str", config.get(PREFIX_SCAN_KEY, "warp")),
            )
            if proof is not None
            else None
        )
    # Unknown structure, precision, resource and binding domains retain all
    # existing numerical, descriptor, SIMD and alias handling for this binding.
    ordinary_config = type(config).from_dict(
        {key: value for key, value in config.config.items() if key not in CONFIG_KEYS}
    )
    module = generate_ast(func, ordinary_config, emit_repro_caller)
    if proof is None or schedule is None:
        return module
    return _compose_flat_grouped_host(
        func,
        module,
        proof,
        native_source=flat_grouped_kernel_module(schedule),
        plans=flat_grouped_wrapper_plans(schedule),
        identity=schedule.cache_identity(),
        rows=schedule.rows,
        columns=schedule.columns,
        reduction=schedule.reduction,
        workspace_upper_bound=schedule.provider.tile_upper_bound,
        thread_block=schedule.thread_block,
        resident_ctas=schedule.resident_ctas,
    )


def generate_grouped_warp_tf32_candidate(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    *,
    shared_capacity: int,
) -> ast.Module:
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import generate_ast
    from .grouped_warp_tf32_kernel import grouped_warp_tf32_module
    from .grouped_warp_tf32_plan import grouped_warp_tf32_plan

    env = CompileEnvironment.current()
    with func:
        proof = prove_flat_grouped_rna(env, func.device_ir)
        plan = (
            grouped_warp_tf32_plan(env, proof, shared_capacity=shared_capacity)
            if proof is not None
            else None
        )
    ordinary_config = type(config).from_dict(
        {key: value for key, value in config.config.items() if key not in CONFIG_KEYS}
    )
    module = generate_ast(func, ordinary_config, emit_repro_caller)
    if proof is None or plan is None:
        return module
    return _compose_flat_grouped_host(
        func,
        module,
        proof,
        native_source=grouped_warp_tf32_module(plan),
        plans=[],
        identity=plan.cache_identity(),
        rows=plan.rows,
        columns=plan.columns,
        reduction=plan.reduction,
        workspace_upper_bound=plan.workspace_upper_bound,
        thread_block=plan.thread_block,
        grid=plan.grid,
    )


def _compose_flat_grouped_host(
    func: HostFunction,
    module: ast.Module,
    proof: FlatGroupedIRProof,
    *,
    native_source: str,
    plans: list[dict[str, object]],
    identity: tuple[object, ...],
    rows: int,
    columns: int,
    reduction: int,
    workspace_upper_bound: int,
    thread_block: tuple[int, int, int],
    resident_ctas: int = 1,
    grid: tuple[int, int, int] | None = None,
) -> ast.Module:
    """Preserve original host statements and the ordinary six-tensor launcher."""
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    host = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == func.name
    )
    launches = [
        stmt
        for stmt in host.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "_launcher"
    ]
    if len(launches) != 1:
        return module
    used = _bound_names(module)

    def fresh(prefix: str) -> str:
        while prefix in used:
            prefix += "_"
        used.add(prefix)
        return prefix

    code_cache = fresh("_helion_native_code_cache")
    tensor_library = fresh("_helion_native_tensor_library")
    get_num_sm = fresh("_helion_native_get_num_sm")
    native_module = fresh("_helion_native_grouped_module")
    workspace = fresh("_helion_native_tensormap_workspace")
    grid_z = fresh("_helion_native_grid_z")
    native = f"{native_module}._helion_native_grouped_rna"
    a = proof.a_flat_expression
    output = proof.output_flat_expression
    m, n, k = rows, columns, reduction
    # The source's original view/empty/return statements stay in place. These
    # two additional views preserve their actual pointers and create no device
    # allocation. The original Int64 offsets remain the actual launch argument.
    grid_multiplier = f" * {resident_ctas}" if resident_ctas != 1 else ""
    launch_grid = f"(1, 1, {grid_z})" if grid is None else repr(grid)
    replacement = ast.parse(
        f"{grid_z} = {get_num_sm}({a}.device){grid_multiplier}\n"
        f"if {grid_z} > {workspace_upper_bound}:\n"
        f"    {grid_z} = {workspace_upper_bound}\n"
        f"{workspace} = {tensor_library}.empty(({grid_z}, 3, 16), dtype={tensor_library}.int64, device={a}.device)\n"
        f"_launcher({native}, {launch_grid}, {proof.offsets_argument}, {a}.view(({m}, {k})), "
        f"{proof.b_argument}.transpose(1, 2), {proof.bias_argument}, {output}.view(({m}, {n})), "
        f"{workspace}, block={thread_block!r})\n"
    ).body
    position = host.body.index(launches[0])
    host.body[position : position + 1] = replacement
    loads = ast.parse(
        f"import torch as {tensor_library}\n"
        f"from helion.runtime import get_num_sm as {get_num_sm}\n"
        f"from torch._inductor.codecache import PyCodeCache as {code_cache}\n"
        f"{native_module} = {code_cache}.load({native_source!r})\n"
        f"{native}._helion_cute_source_hash = {hashlib.sha256(native_source.encode()).hexdigest()!r}\n"
    ).body
    position = module.body.index(host)
    module.body[position:position] = loads
    module.body[module.body.index(host) + 1 : module.body.index(host) + 1] = ast.parse(
        f"{func.name}._helion_cute_kernels = ({native},)\n"
        f"{func.name}._helion_cute_native_grouped_plan = {identity!r}\n"
    ).body
    env.cute_resolved_wrapper_plans = plans
    return module
