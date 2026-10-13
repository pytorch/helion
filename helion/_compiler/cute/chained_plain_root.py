"""Straight-line full-K roots executed by the common contraction stage.

The direct K-major single-stage capability is assembled here. Accepted pairs
reuse the original root scheduler with shared physical stage actions. Other
root schedules retain their original stages; no synthetic loop or arithmetic
template is introduced.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from . import chained_tcgen_stage as stages
from .chained_collectives import uses_general_collectives
from .chained_stage_operands import plan_direct_stage_operands
from .chained_tcgen05 import _epilogue
from .chained_tcgen05 import _layout
from .chained_tcgen05 import supported_plan

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def supports_plain_root(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """Check semantic/selected-schedule facts before any expression emission."""
    if (
        plan.strategy != "tcgen05_tmem"
        or plan.loop is not None
        or len(plan.dots) != 1
        or plan.shapes[0][0] != 128
        or plan.dots[0].args[2] is not None
        or plan.scans
        or plan.scan_exports
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.direct_output
        or plan.k_schedule is not None
        or plan.pointwise_cache is not None
        or plan.warp_mma_stages
        or plan.preparation_pipeline is not None
        or uses_general_collectives(plan)
        or not all(
            chain._direct_operand(cast("Node", node)) for node in plan.dots[0].args[:2]
        )
        or not supported_plan(plan)
    ):
        return False
    return _ordinary_root_options(cg)


def _ordinary_root_options(cg: GenerateAST) -> bool:
    config = cg.device_function.config.config
    # Dormant auxiliary caching is safe: its candidates exclude direct operands,
    # and this one-dot region has no late bridge. Keep the actual saved option.
    return all(
        config.get(key, default) == default
        for key, default in (
            ("cute_chained_pointwise_vectorize", False),
            ("cute_chained_vector_group", False),
            ("cute_chained_pointwise_unroll", 1),
            ("cute_chained_pointwise_read_cache", False),
            ("cute_chained_pointwise_inplace_async", False),
            ("cute_chained_startup_transfer", "legacy"),
            ("cute_chained_leaf_pipeline", "legacy"),
            ("cute_chained_tmem_free", "legacy"),
            ("cute_chained_seed_tile_columns", 0),
            ("cute_chained_pointwise_cache_layout", "auto"),
            ("cute_chained_scratch_layout", "row_major"),
        )
    )


def codegen_plain_root(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    try:
        return _codegen_plain_root(cg, plan)
    except chain._UnsupportedChain as error:
        raise exc.BackendUnsupported(
            "cute", f"unsupported TCgen05 chain expression: {error}"
        ) from error


def _codegen_plain_root(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """Return false before installation when a complete capability is absent."""
    snapshot_columns = cast(
        "int",
        cg.device_function.config.config.get("cute_chained_snapshot_tile_columns", 0),
    )
    if snapshot_columns:
        from .chained_root_stage import supports_root_pair
        from .chained_tcgen05 import codegen_shared_root_sequence

        if not supports_root_pair(cg, plan):
            raise chain._UnsupportedChain(
                "streamed root snapshot requires a supported exclusive root pair"
            )
        return codegen_shared_root_sequence(
            cg, plan, snapshot_tile_columns=snapshot_columns
        )
    if cg.device_function.config.config.get("cute_chained_drain_tile_columns", 0):
        from .chained_body_program import supports_materialized_root
        from .chained_tcgen05 import codegen_shared_root_sequence

        if not supports_materialized_root(cg, plan):
            raise chain._UnsupportedChain(
                "FP32 drain tiling requires an existing materialized root result"
            )
        return codegen_shared_root_sequence(cg, plan)
    if not supports_plain_root(cg, plan):
        from .chained_root_stage import supports_independent_root
        from .chained_root_stage import supports_initialized_root
        from .chained_root_stage import supports_local_root_pair
        from .chained_root_stage import supports_root_pair
        from .chained_tcgen05 import codegen_shared_root_sequence

        if (
            supports_root_pair(cg, plan)
            or supports_independent_root(cg, plan)
            or supports_initialized_root(cg, plan)
            or supports_local_root_pair(cg, plan)
        ):
            return codegen_shared_root_sequence(cg, plan)
        return False
    geometry = stages.StageGeometry(plan.shapes[0], transpose=False)
    operands = plan_direct_stage_operands(cg, plan, 0, geometry)
    if operands is None:
        return False
    if cg.device_function.config.config.get("cute_chained_fragment_epilogues") is True:
        raise chain._UnsupportedChain(
            "fragment epilogues lack an admitted original result image"
        )
    env = CompileEnvironment.current()
    dtype = env.backend.dtype_str(plan.operand_dtype(0))
    output_dtype = env.backend.dtype_str(
        cast("Node", plan.store.args[0]).meta["val"].dtype
    )
    index_dtype = env.backend.dtype_str(env.index_dtype)
    m, n, k = geometry.physical
    early_release = cast(
        "bool",
        cg.device_function.config.config.get("cute_chained_tmem_early_release", False),
    )
    lines = [
        "from cutlass.cute.nvgpu import tcgen05",
        "from cutlass.utils import blackwell_helpers as chain_sm100",
        "from cutlass.utils import TmemAllocator",
        "import cutlass.pipeline as chain_pipeline",
        "chain_thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "chain_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
        "chain_pid = cutlass.Int32(cute.arch.block_idx()[0])",
    ]
    suffix = 1
    for axis, extent, block in reversed(plan.axes):
        count = extent // block
        lines.append(
            f"chain_origin_{axis} = {index_dtype}(chain_pid // {suffix} % {count}) * {block}"
        )
        suffix *= count
    lines.extend(
        [
            f"chain_a_workspace = cute.arch.alloc_smem({dtype}, {m * k}, alignment=128)",
            f"chain_b_workspace = cute.arch.alloc_smem({dtype}, {n * k}, alignment=128)",
            f"chain_output_ptr = cute.arch.alloc_smem({output_dtype}, {math.prod((m, n))}, alignment=128)",
            *_layout("chain_output", (m, n), 1, output_dtype),
            *stages.tmem_resource_setup(n, 1, 128, early_release=early_release),
        ]
    )
    boundaries: dict[Node, str] = {}
    from .chained_body_program import emit_body_program
    from .chained_body_program import plan_root_body
    from .chained_scratch_layout import ScratchLayouts

    scratch = ScratchLayouts.for_plan("row_major", plan)
    body = plan_root_body(
        cg,
        plan,
        scratch,
        inner_axes={(0, "a"): 1, (0, "b"): 1},
        bridges={},
        early_release=early_release,
        direct=operands,
    )
    if body is None:
        raise chain._UnsupportedChain("accepted direct root body changed")
    lines.extend(
        emit_body_program(
            cg,
            plan,
            None,
            body.execution,
            None,
            None,
            scratch,
            root=body,
        )
    )
    boundaries.update(body.boundaries)
    staged: list[chain._StagedInput] = []
    for role, shape, operand in zip(
        ("a", "b"), ((m, k), (n, k)), operands.nodes, strict=True
    ):
        cached = chain._stage_input(cg, plan, operand, f"chain_0_{role}", role, shape)
        if cached is not None:
            staged.append(cached)
    lines.extend(_epilogue(cg, plan, boundaries, [], staged))
    lines.extend(stages.free_stages())
    return chain._install_chained_body(cg, plan, lines)
