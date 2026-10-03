from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.config import Config
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteBlockScaledMmaHeuristic(AutotunerHeuristic):
    """Seed packed pipeline geometry independently of the scalar source tiles."""

    name = "cute_block_scaled_mma"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata", "config_num_sm"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        return (
            cls.CACHE_SPECIALIZATION_FACTS
            if env.config_spec.cute_scaled_mma_available
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        return (
            spec.cute_scaled_mma_available
            and spec.target_device_capability is not None
            and spec.target_device_capability[0] == 10
            and len(device_ir.root_ids) == 1
        )

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
        if not cls.is_eligible(env, device_ir):
            return []
        base = env.config_spec._base_default_config().config
        return [
            Config.from_dict(
                base
                | {
                    "cute_scaled_mma": True,
                    "cute_collective_mma": False,
                    "cute_scaled_cluster_m": cluster,
                    "cute_scaled_tile_n": n,
                    "cute_scaled_tile_k": k,
                    "cute_scaled_stages": stages,
                    "cute_scaled_persistent": persistent,
                }
            )
            for cluster, n, k, stages, persistent in (
                (2, 256, 256, 5, True),
                (1, 128, 64, 3, True),
                (2, 256, 256, 4, True),
                (2, 256, 128, 4, True),
                (1, 256, 128, 3, True),
                (2, 128, 128, 4, True),
                (1, 128, 128, 4, False),
                (2, 256, 256, 4, False),
            )
        ]
