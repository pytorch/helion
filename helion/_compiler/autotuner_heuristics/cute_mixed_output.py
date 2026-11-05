"""Generic ONE-CTA carriers for complete mixed-width native output stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..cute.tcgen05_config import CuteTcgen05Config
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteMixedOutputStoreHeuristic(AutotunerHeuristic):
    """Add legal pipeline carriers without replacing any existing seed/default."""

    name = "cute_mixed_output_store"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        return (
            spec.cute_tcgen05_search_enabled
            and spec._cute_tcgen05_config.mixed_output_store_plan is not None
        )

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        spec = env.config_spec
        # A seed consumes only the source-admission certificate. It does not
        # borrow constructor values or publish a partial general snapshot.
        plan = spec._cute_tcgen05_config.mixed_output_store_plan
        extents = spec._tcgen05_matmul_compile_time_static_extents()
        if plan is None or extents is None or extents[2] is None:
            return []
        k = extents[2]
        if k <= 0:
            return []
        budget = CuteTcgen05Config.per_cta_smem_capacity_bytes(plan.device)
        if budget <= 0:
            return []
        seeds = []
        for bm in (64, 128):
            for bn in (32, 64, 128):
                for bk in (32, 64, 128):
                    # Input tail support is not part of this output contract.
                    if k % bk:
                        continue
                    blocks = spec._tcgen05_matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
                    if blocks is None:
                        continue
                    for ab_stages in (2, 4):
                        required = plan.seed_smem_upper_bound(
                            bm=bm,
                            bn=bn,
                            bk=bk,
                            input_dtype_bytes=plan.input_dtype.itemsize,
                            ab_stages=ab_stages,
                            c_stages=2,
                        )
                        if required is None or required > budget:
                            continue
                        seeds.append(
                            Config(
                                block_sizes=blocks,
                                pid_type="flat",
                                tcgen05_cluster_m=1,
                                tcgen05_cluster_n=1,
                                tcgen05_ab_stages=ab_stages,
                                tcgen05_acc_stages=2,
                                tcgen05_c_stages=2,
                                tcgen05_num_epi_warps=4,
                                tcgen05_aux_load_placement="post_acc_wait",
                                tcgen05_aux_load_mode="simt",
                            )
                        )
        return seeds
