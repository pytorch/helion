from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_fragment import BlockSizeFragment
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteTcgen05ClusterM2Heuristic(AutotunerHeuristic):
    name = "cute_tcgen05_cluster_m2"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        constraints = spec._tcgen05_cluster_m2_search_constraints
        if (
            constraints is None
            or TCGEN05_TWO_CTA_SEED_PID_TYPE not in spec.allowed_pid_types
        ):
            return False
        if len(spec.block_sizes) != 3:
            return False

        bm_fragment = cast("BlockSizeFragment", spec.block_sizes[0]._fragment(spec))
        bn_fragment = cast("BlockSizeFragment", spec.block_sizes[1]._fragment(spec))
        return (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
            and cls._select_bk(env) is not None
        )

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        spec = env.config_spec
        bk = cls._select_bk(env)
        if bk is None:
            raise AssertionError(f"{cls.name} get_seed_config called while ineligible")

        block_sizes = [
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_BLOCK_N,
            bk,
        ]
        if spec.indexing.length == 3:
            # Pure matmul has exactly the A/B/C indexing slots. Fused epilogues
            # add more memory ops, so leave those seeds to the spec default
            # rather than constructing a partial list.
            return Config(
                block_sizes=block_sizes,
                l2_groupings=[TCGEN05_TWO_CTA_SEED_L2_GROUPING],
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
                tcgen05_cluster_m=2,
                tcgen05_num_epi_warps=4,
                indexing=[
                    "tensor_descriptor",
                    "tensor_descriptor",
                    "tensor_descriptor",
                ],
            )

        return Config(
            block_sizes=[
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            l2_groupings=[TCGEN05_TWO_CTA_SEED_L2_GROUPING],
            pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
            tcgen05_cluster_m=2,
            # Matches the validated tcgen05 search restriction.
            tcgen05_num_epi_warps=4,
        )

    @staticmethod
    def _select_bk(env: CompileEnvironment) -> int | None:
        spec = env.config_spec
        constraints = spec._tcgen05_cluster_m2_search_constraints
        if constraints is None or len(spec.block_sizes) != 3:
            return None
        bk_fragment = cast("BlockSizeFragment", spec.block_sizes[2]._fragment(spec))
        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if spec._tcgen05_cluster_m2_bk_is_valid(bk, constraints):
                return bk
            bk //= 2
        return None
