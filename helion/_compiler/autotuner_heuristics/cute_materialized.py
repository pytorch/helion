from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..cute.strategies import Tcgen05PersistenceModel
from .common import dedupe_configs
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteMaterializedMmaHeuristic(AutotunerHeuristic):
    """Seed complete native MMA schedules for independent device launches."""

    name = "cute_materialized_mma"
    backend = "cute"
    promote_seed_to_default = True

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls.get_seed_configs(env, device_ir))

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        spec = env.config_spec
        axes = spec._cute_tcgen05_config.materialized_matmul_block_ids
        if not axes:
            return []
        fragments = [item._fragment(spec) for item in spec.block_sizes]
        configurations = []
        # Coherent small/large tiles seed the initial population. Every stage's
        # M/N/K slots remain independent dimensions in the ordinary search.
        for bm, bn, bk in (
            (128, 128, 64),
            (64, 64, 64),
            (128, 64, 64),
            (64, 128, 64),
            (128, 128, 128),
            (64, 64, 32),
        ):
            block_sizes = [fragment.default() for fragment in fragments]
            for triple in axes:
                for block_id, value in zip(triple, (bm, bn, bk), strict=True):
                    index = spec.block_sizes.block_id_to_index(block_id)
                    fragment = fragments[index]
                    block_sizes[index] = min(max(value, fragment.low), fragment.high)
            for pid in ("flat", "persistent_interleaved"):
                if pid not in spec.allowed_pid_types:
                    continue
                configurations.append(
                    Config(
                        block_sizes=block_sizes.copy(),
                        pid_type=pid,
                        tcgen05_cluster_m=1,
                        tcgen05_cluster_n=1,
                        tcgen05_ab_stages=2,
                        tcgen05_acc_stages=2,
                        tcgen05_c_stages=2,
                        tcgen05_num_epi_warps=4,
                        tcgen05_persistence_model=(
                            Tcgen05PersistenceModel.NON_PERSISTENT.value
                            if pid == "flat"
                            else Tcgen05PersistenceModel.STATIC_PERSISTENT.value
                        ),
                    )
                )
        for config in configurations:
            # Apply the same tail/pipeline projections used by search before
            # promoting a seed, so its measured schedule stays reachable.
            spec.normalize(config, _fix_invalid=True)
        return dedupe_configs(configurations)
