from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.config import Config
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


def _seed_geometry_int(
    values: Mapping[str, object], key: str, default: int | None = None
) -> int:
    """Read the integer geometry constructed by this compiler seed family."""
    value = values[key] if default is None else values.get(key, default)
    assert type(value) is int
    return value


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
        if (partition_key := env.config_spec.cute_mxfp8_partition_key) is not None:
            base[partition_key] = 1
        variants = (
            (
                (1, 128, 128, 2, True),
                (1, 256, 128, 3, True),
                (2, 256, 128, 3, True),
                (2, 256, 256, 2, True),
                (1, 128, 256, 2, True),
                (1, 128, 128, 3, False),
                (1, 64, 128, 2, True),
                (1, 64, 256, 2, True),
                (2, 64, 256, 2, True),
            )
            if env.config_spec.cute_mxfp8_mma_available
            else (
                (2, 256, 256, 5, True),
                (1, 128, 64, 3, True),
                (2, 256, 256, 4, True),
                (2, 256, 128, 4, True),
                (1, 256, 128, 3, True),
                (2, 128, 128, 4, True),
                (1, 128, 128, 4, False),
                (2, 256, 256, 4, False),
            )
        )
        seeds = [
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
            for cluster, n, k, stages, persistent in variants
        ]
        if env.config_spec.cute_mxfp8_fused_scales_available:
            fused_variants = (
                (2, 32, 256, 4, 64, True),
                (1, 32, 256, 4, 64, True),
                (2, 16, 256, 4, 64, True),
                (1, 16, 256, 4, 64, True),
                (1, 32, 256, 4, 128, True),
                (2, 32, 256, 5, 64, True),
                (2, 224, 128, 6, 128, False),
                (2, 128, 256, 4, 128, False),
                (1, 128, 128, 8, 64, False),
            )
            seeds = [
                Config.from_dict(seed.config | {"cute_scaled_fused_scales": True})
                for seed in seeds
            ] + seeds
            seeds = [
                Config.from_dict(
                    base
                    | {
                        "cute_scaled_mma": True,
                        "cute_collective_mma": False,
                        "cute_scaled_fused_scales": True,
                        "cute_scaled_cluster_m": cluster,
                        "cute_scaled_tile_n": n,
                        "cute_scaled_tile_k": k,
                        "cute_scaled_stages": stages,
                        "cute_scaled_tile_m": m,
                        "cute_scaled_swap_ab": swap,
                        "cute_scaled_persistent": True,
                    }
                )
                for cluster, n, k, stages, m, swap in fused_variants
            ] + seeds
            compact_seeds = [
                Config.from_dict(seed.config | {"cute_scaled_compact_m": True})
                for seed in seeds
                if seed.config.get("cute_scaled_fused_scales") is True
                and _seed_geometry_int(seed.config, "cute_scaled_tile_m", 128) < 128
            ]
            compact_variants = (
                (1, 16, 256, 2, 32),
                (2, 16, 256, 4, 32),
                (1, 16, 128, 8, 32),
                (2, 16, 128, 12, 64),
                (1, 32, 128, 8, 64),
                (2, 32, 128, 12, 64),
            )
            compact_seeds += [
                Config.from_dict(
                    base
                    | {
                        "cute_scaled_mma": True,
                        "cute_collective_mma": False,
                        "cute_scaled_fused_scales": True,
                        "cute_scaled_compact_m": True,
                        "cute_scaled_cluster_m": cluster,
                        "cute_scaled_tile_n": n,
                        "cute_scaled_tile_k": k,
                        "cute_scaled_stages": stages,
                        "cute_scaled_tile_m": m,
                        "cute_scaled_swap_ab": True,
                        "cute_scaled_persistent": True,
                    }
                )
                for cluster, n, k, stages, m in compact_variants
            ]
            seeds = compact_seeds + seeds
        return seeds
