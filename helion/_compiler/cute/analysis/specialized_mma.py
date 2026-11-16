"""Planner: the specialized-MMA plan (tcgen05 / warp MMA) of a kernel's matmul roots.

Role: planner.  ``_kernel_specialized_mma_plan`` decides, from the device IR
and one config, which specialized MMA implementation the kernel's root tile
loops take (``_SpecializedMmaPlan`` with the M/N block ids) and
``_kernel_specialized_mma_impl`` exposes the implementation name the tile
strategies and persistent program ids probe.  The rank-3 helpers prove the
grouped-N ``x[b, m, k] @ w[n, k]`` form may take the specialized path.  Moved
verbatim from ``helion/_compiler/backend.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Sequence

import torch

from ..tcgen05.knobs import cluster_m_knob
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from ..tcgen05_constants import resolve_tcgen05_grouped_worklist_mma_profile

if TYPE_CHECKING:
    from collections.abc import Collection

    from ....runtime.config import Config
    from ...compile_environment import CompileEnvironment
    from ...device_function import DeviceFunction
    from ...generate_ast import GenerateAST
    from ..cute_mma import _CuteMmaNode


def _cute_rank3_rhs_static_full_tiles(
    env: CompileEnvironment,
    *,
    root_grid_ids: Sequence[int],
    k_block_id: int,
    bm: int,
    bn: int,
    bk: int,
) -> bool:
    for block_id, block_size in (
        (root_grid_ids[0], bm),
        (root_grid_ids[1], bn),
        (k_block_id, bk),
    ):
        size = env.block_sizes[block_id].size
        if not isinstance(size, (int, torch.SymInt)):
            return False
        extent = env.size_hint(size)
        if not env.known_equal(size, extent) or extent % block_size != 0:
            return False
    return True


def _rank3_rhs_grouped_nt_can_use_specialized_mma(
    node: torch.fx.Node,
    *,
    env: CompileEnvironment,
    cg: GenerateAST,
    config: Config,
    mn_root_grid_ids: Sequence[int],
    segment_root_grid_id: int | None,
    k_block_id: int,
    bm: int,
    bn: int,
    bk: int,
) -> bool:
    from ..cute_mma import _GroupedMmaAxes
    from ..cute_mma import _prove_rank3_rhs_grouped_mma

    proof = _prove_rank3_rhs_grouped_mma(
        cg,
        node,
        config=config,
        axes=_GroupedMmaAxes(
            m_block_id=mn_root_grid_ids[0],
            n_block_id=mn_root_grid_ids[1],
            k_block_id=k_block_id,
            segment_block_id=segment_root_grid_id,
        ),
    )
    if proof is None:
        return False
    return proof.is_worklist or (
        cluster_m_knob(config) == 1
        and _cute_rank3_rhs_static_full_tiles(
            env,
            root_grid_ids=mn_root_grid_ids,
            k_block_id=k_block_id,
            bm=bm,
            bn=bn,
            bk=bk,
        )
    )


def _specialized_mma_root_mn_block_ids(
    candidate: _CuteMmaNode,
    config: Config,
    *,
    inactive_root_block_ids: Collection[int] = (),
) -> tuple[int, int] | None:
    """Return exact root-grid matrix axes for an analyzed MMA candidate.

    ``inactive_root_block_ids`` names root axes whose elements the body never
    observes (a split-K outer K tile that only supplies the inner loop's
    begin/end/id).  The exact fp32 SIMT GEMM owns the remaining axes; every
    other lowering keeps requiring the root to be exactly the output tile.
    """
    from ...compile_environment import CompileEnvironment
    from ...host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return None
    operands = candidate.operands
    root_block_ids = tuple(device_ir.grid_block_ids[0])
    if root_block_ids != operands.output_block_ids:
        from ..mma_support import cute_fp32_dot_uses_tf32
        from ..simt_gemm import simt_mma_requested

        if (
            not simt_mma_requested(config)
            or operands.lhs.source_fake.dtype != torch.float32
            or cute_fp32_dot_uses_tf32()
        ) or (
            tuple(
                block_id
                for block_id in root_block_ids
                if block_id not in inactive_root_block_ids
            )
            != operands.output_block_ids
        ):
            return None
    if (leading_id := operands.leading_passthrough_block_id) is not None:
        block_size = (
            CompileEnvironment.current().block_sizes[leading_id].from_config(config)
        )
        if not isinstance(block_size, int) or block_size != 1:
            return None
    return operands.m_block_id, operands.n_block_id


class _SpecializedMmaPlan(NamedTuple):
    impl: str
    m_block_id: int
    n_block_id: int


def _grouped_rank3_specialized_mma_plan(
    node: torch.fx.Node,
    *,
    fn: DeviceFunction,
    k_block_id: int,
    bk: int,
    config: Config,
    env: CompileEnvironment,
) -> _SpecializedMmaPlan | None:
    from ...host_function import HostFunction
    from ..cute_mma import _choose_mma_impl
    from ..cute_mma import _rank3_grouped_root_axes
    from ..grouped_row_union import physical_schedule

    if node.target is not torch.ops.aten.addmm.default:
        return None
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return None
    root_grid_ids = device_ir.grid_block_ids[0]
    axes = None
    semantic_block_ids = env.config_spec.tcgen05.matmul_block_ids
    if semantic_block_ids is not None:
        m_block_id, n_block_id, semantic_k_block_id = semantic_block_ids
        if env.canonical_block_id(semantic_k_block_id) == env.canonical_block_id(
            k_block_id
        ):
            axes = _rank3_grouped_root_axes(
                env,
                device_ir,
                m_block_id=m_block_id,
                n_block_id=n_block_id,
                k_block_id=k_block_id,
            )
    if semantic_block_ids is not None and axes is None:
        return None
    if axes is None:
        if len(root_grid_ids) == 2:
            segment_root_grid_id = None
            mn_root_grid_ids = root_grid_ids
        elif len(root_grid_ids) == 3:
            segment_root_grid_id = root_grid_ids[0]
            mn_root_grid_ids = root_grid_ids[1:]
        else:
            return None
    else:
        segment_root_grid_id = axes.segment_block_id
        mn_root_grid_ids = [axes.m_block_id, axes.n_block_id]
    if segment_root_grid_id is not None:
        segment_block = env.block_sizes[segment_root_grid_id].from_config(config)
        if segment_block != 1:
            return None
    bm = env.block_sizes[mn_root_grid_ids[0]].from_config(config)
    bn = env.block_sizes[mn_root_grid_ids[1]].from_config(config)
    if not isinstance(bm, int) or not isinstance(bn, int):
        return None
    if not _rank3_rhs_grouped_nt_can_use_specialized_mma(
        node,
        env=env,
        cg=fn.codegen,
        config=config,
        mn_root_grid_ids=mn_root_grid_ids,
        segment_root_grid_id=segment_root_grid_id,
        k_block_id=k_block_id,
        bm=bm,
        bn=bn,
        bk=bk,
    ):
        return None
    lhs_node = node.args[1] if len(node.args) > 1 else None
    if not isinstance(lhs_node, torch.fx.Node):
        return None
    lhs_val = lhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor):
        return None
    mma_bm = bm
    mma_bn = bn
    worklist_profile = resolve_tcgen05_grouped_worklist_mma_profile(
        config,
        block_k=bk,
    )
    if (
        config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) == TCGEN05_GROUPED_MODE_WORKLIST_NM
        and worklist_profile is None
    ):
        return None
    row_profile = physical_schedule(config)
    collective_profile = row_profile or worklist_profile
    if collective_profile is not None:
        mma_bm, mma_bn = collective_profile.mma_m, collective_profile.mma_n
    mma_impl = _choose_mma_impl(
        lhs_val.dtype,
        bm=mma_bm,
        bn=mma_bn,
        bk=bk,
        config=config,
        input_device=lhs_val.device,
        defer_grouped_worklist_smem_check=collective_profile is not None,
    )
    if mma_impl != "tcgen05":
        return None
    return _SpecializedMmaPlan(mma_impl, *mn_root_grid_ids)


def _analyzed_specialized_mma_plan(
    node: torch.fx.Node,
    *,
    fn: DeviceFunction,
    k_block_id: int,
    bk: int,
    config: Config,
    env: CompileEnvironment,
) -> _SpecializedMmaPlan | None:
    from ..cute_mma import _choose_mma_impl
    from ..cute_mma import _mma_tiles_are_static_full
    from ..cute_mma import _tcgen05_cp_async_a_mma_bm
    from ..cute_mma import analyze_cute_mma_node
    from ..cute_mma import ensure_tcgen05_fragment_epilogue_plan

    candidate = analyze_cute_mma_node(node, graphs=fn.codegen.codegen_graphs)
    if (
        candidate is None
        or candidate.requires_accumulator_seed
        or candidate.operands.k_block_id != k_block_id
    ):
        return None
    root_mn_block_ids = _specialized_mma_root_mn_block_ids(candidate, config)
    if root_mn_block_ids is None:
        return None
    bm = env.block_sizes[root_mn_block_ids[0]].from_config(config)
    bn = env.block_sizes[root_mn_block_ids[1]].from_config(config)
    if not isinstance(bm, int) or not isinstance(bn, int):
        return None
    if candidate.operands.has_leading_passthrough and not _mma_tiles_are_static_full(
        candidate.operands, bm=bm, bn=bn, bk=bk
    ):
        return None
    lhs_val = candidate.operands.lhs.source_fake
    mma_impl = _choose_mma_impl(
        lhs_val.dtype,
        # the cp.async-A kernel's 256-row tile runs 128-row subtiles
        bm=_tcgen05_cp_async_a_mma_bm(
            env, config, candidate.operands.lhs, candidate.operands.rhs, bm=bm
        ),
        bn=bn,
        bk=bk,
        config=config,
        input_device=lhs_val.device,
    )
    if mma_impl == "universal":
        return None
    if mma_impl == "tcgen05" and not ensure_tcgen05_fragment_epilogue_plan(
        fn,
        node,
        candidate,
        bm=bm,
        bn=bn,
        bk=bk,
        config=config,
    ):
        return None
    return _SpecializedMmaPlan(mma_impl, *root_mn_block_ids)


def _kernel_specialized_mma_plan(
    fn: DeviceFunction,
    *,
    config: Config,
) -> _SpecializedMmaPlan | None:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import ForLoopGraphInfo
    from ...host_function import HostFunction

    env = CompileEnvironment.current()
    grid_ids = {
        bid for ids in HostFunction.current().device_ir.grid_block_ids for bid in ids
    }
    seen_block_ids: set[tuple[int, ...]] = set()
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = tuple(graph_info.block_ids)
        if block_ids in seen_block_ids or not any(
            bid not in grid_ids for bid in block_ids
        ):
            continue
        seen_block_ids.add(block_ids)
        block_sizes = [env.block_sizes[bid].from_config(config) for bid in block_ids]
        if len(block_sizes) != 1 or not isinstance(block_sizes[0], int):
            continue
        bk = block_sizes[0]
        for node in graph_info.graph.nodes:
            plan = _analyzed_specialized_mma_plan(
                node,
                fn=fn,
                k_block_id=block_ids[0],
                bk=bk,
                config=config,
                env=env,
            )
            if plan is not None:
                return plan
            plan = _grouped_rank3_specialized_mma_plan(
                node,
                fn=fn,
                k_block_id=block_ids[0],
                bk=bk,
                config=config,
                env=env,
            )
            if plan is not None:
                return plan
    return None


def _kernel_specialized_mma_impl(
    fn: DeviceFunction,
    *,
    config: Config,
) -> str | None:
    plan = _kernel_specialized_mma_plan(fn, config=config)
    return None if plan is None else plan.impl
