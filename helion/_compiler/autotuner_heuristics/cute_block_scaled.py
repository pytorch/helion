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
                # 2-CTA pair, bn=128, bk=256, 5 stages: the cold-autotune
                # winners of nvfp4 W4A4 at 512^3 (persistent) and 1024^3.
                (2, 128, 256, 5, True),
                (2, 128, 256, 5, False),
                # bn=64 single CTAs: twice the CTAs of the 2-CTA bn=128 pair
                # for the same K steps; wins at 1024x2048x1024 and 512^3
                # where the pair leaves most SMs idle.
                (1, 64, 256, 5, False),
                (1, 64, 256, 5, True),
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
        if not env.config_spec.cute_mxfp8_mma_available:
            # The bn=64 seeds again with 16-row preparation tiles (twice the
            # transpose blocks; the 1024x2048x1024 winner), and again with the
            # preparation fused into the consumer (no workspace kernel).
            narrow = [seed for seed in seeds if seed.config["cute_scaled_tile_n"] == 64]
            seeds.extend(
                Config.from_dict(seed.config | {"cute_scaled_prepare_rows": 16})
                for seed in narrow
            )
            seeds.extend(
                Config.from_dict(seed.config | {"cute_scaled_prepare": "fused"})
                for seed in narrow
            )
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
            # Add joint geometries without changing the existing seed prefix.
            # N=8 is a one-CTA instruction choice, admitted only for this
            # proved packed MXFP8 contraction. Both K-window widths remain
            # independent choices; no workload dimensions select a geometry.
            joint_variants = (
                (1, 8, 128, 2, 64, True),
                (1, 8, 128, 8, 64, True),
                (1, 8, 128, 12, 64, True),
                (1, 8, 128, 2, 128, False),
                (1, 8, 128, 8, 128, False),
                (1, 8, 128, 12, 128, False),
                (1, 8, 256, 4, 64, True),
                (1, 8, 256, 8, 64, True),
                (1, 16, 128, 12, 128, False),
                (2, 16, 128, 12, 128, False),
            )
            for cluster, n, k, stages, m, compact in joint_variants:
                seed = Config.from_dict(
                    base
                    | {
                        "cute_scaled_mma": True,
                        "cute_collective_mma": False,
                        "cute_scaled_fused_scales": True,
                        "cute_scaled_compact_m": compact,
                        "cute_scaled_cluster_m": cluster,
                        "cute_scaled_tile_n": n,
                        "cute_scaled_tile_k": k,
                        "cute_scaled_stages": stages,
                        "cute_scaled_tile_m": m,
                        "cute_scaled_swap_ab": True,
                        "cute_scaled_persistent": True,
                    }
                )
                if seed not in seeds:
                    seeds.append(seed)
        if env.config_spec.cute_mxfp8_fused_scales_available:
            # Keep the restored high-stage profiles and admit the short-K
            # two-stage N8 pipeline independently of workload dimensions.
            seeds += [
                Config.from_dict(
                    base
                    | {
                        "cute_scaled_mma": True,
                        "cute_collective_mma": False,
                        "cute_scaled_fused_scales": True,
                        "cute_scaled_compact_m": True,
                        "cute_scaled_cluster_m": 1,
                        "cute_scaled_tile_n": 8,
                        "cute_scaled_tile_k": 256,
                        "cute_scaled_stages": 2,
                        "cute_scaled_tile_m": 64,
                        "cute_scaled_swap_ab": True,
                        "cute_scaled_persistent": persistent,
                    }
                )
                for persistent in (False, True)
            ]
        if env.config_spec.cute_mxfp8_sf_tma_available:
            from ..cute.block_scaled_config import mxfp8_shared_memory_bytes

            shape = env.config_spec.cute_mxfp8_problem_shape
            assert shape is not None
            m, n, _k = shape
            sf_seeds = []
            candidates = [
                seed.config
                for seed in seeds
                if seed.config.get("cute_scaled_fused_scales") is True
            ]
            candidates += [
                base
                | {
                    "cute_scaled_mma": True,
                    "cute_collective_mma": False,
                    "cute_scaled_fused_scales": True,
                    "cute_scaled_compact_m": True,
                    "cute_scaled_cluster_m": 1,
                    "cute_scaled_tile_n": 32,
                    "cute_scaled_tile_k": 256,
                    "cute_scaled_stages": 2,
                    "cute_scaled_tile_m": 64,
                    "cute_scaled_swap_ab": True,
                    "cute_scaled_persistent": persistent,
                }
                for persistent in (False, True)
            ]
            for candidate in candidates:
                runtime_m = n if candidate.get("cute_scaled_swap_ab") else m
                smem = mxfp8_shared_memory_bytes(
                    runtime_m,
                    _seed_geometry_int(candidate, "cute_scaled_tile_m", 128),
                    _seed_geometry_int(candidate, "cute_scaled_tile_n"),
                    _seed_geometry_int(candidate, "cute_scaled_tile_k"),
                    _seed_geometry_int(candidate, "cute_scaled_stages"),
                    _seed_geometry_int(candidate, "cute_scaled_cluster_m"),
                    env.config_spec.cute_mxfp8_output_element_size,
                    compact_m=bool(candidate.get("cute_scaled_compact_m", False)),
                    sf_tma=True,
                )
                if smem <= 232448:
                    sf_seeds.append(
                        Config.from_dict(candidate | {"cute_scaled_sf_tma": True})
                    )
            seeds += sf_seeds
            inplace_candidates = candidates + [
                base
                | {
                    "cute_scaled_mma": True,
                    "cute_collective_mma": False,
                    "cute_scaled_fused_scales": True,
                    "cute_scaled_compact_m": False,
                    "cute_scaled_cluster_m": 2,
                    "cute_scaled_tile_n": 16,
                    "cute_scaled_tile_k": 128,
                    "cute_scaled_stages": 12,
                    "cute_scaled_tile_m": 128,
                    "cute_scaled_swap_ab": True,
                    "cute_scaled_persistent": persistent,
                }
                for persistent in (False, True)
            ]
            for candidate in inplace_candidates:
                runtime_m = n if candidate.get("cute_scaled_swap_ab") else m
                smem = mxfp8_shared_memory_bytes(
                    runtime_m,
                    _seed_geometry_int(candidate, "cute_scaled_tile_m", 128),
                    _seed_geometry_int(candidate, "cute_scaled_tile_n"),
                    _seed_geometry_int(candidate, "cute_scaled_tile_k"),
                    _seed_geometry_int(candidate, "cute_scaled_stages"),
                    _seed_geometry_int(candidate, "cute_scaled_cluster_m"),
                    env.config_spec.cute_mxfp8_output_element_size,
                    compact_m=bool(candidate.get("cute_scaled_compact_m", False)),
                    sf_tma=True,
                    sf_inplace=True,
                )
                if smem <= 232448:
                    seed = Config.from_dict(
                        candidate
                        | {"cute_scaled_sf_tma": True, "cute_scaled_sf_inplace": True}
                    )
                    if seed not in seeds:
                        seeds.append(seed)
            # Preserve the legal seed prefix and expose the six-stage compact
            # SF pipeline for both scheduler choices. The resource filter
            # below decides legality independently of workload dimensions.
            for persistent in (False, True):
                seed = Config.from_dict(
                    base
                    | {
                        "cute_scaled_mma": True,
                        "cute_collective_mma": False,
                        "cute_scaled_fused_scales": True,
                        "cute_scaled_sf_tma": True,
                        "cute_scaled_sf_inplace": True,
                        "cute_scaled_tile_m": 64,
                        "cute_scaled_tile_n": 32,
                        "cute_scaled_tile_k": 256,
                        "cute_scaled_stages": 6,
                        "cute_scaled_cluster_m": 1,
                        "cute_scaled_swap_ab": True,
                        "cute_scaled_compact_m": True,
                        "cute_scaled_persistent": persistent,
                    }
                )
                if seed not in seeds:
                    seeds.append(seed)
        if env.config_spec.cute_mxfp8_mma_available:
            from ..cute.block_scaled_config import mxfp8_shared_memory_bytes

            shape = env.config_spec.cute_mxfp8_problem_shape
            assert shape is not None
            m, n, _k = shape
            # Drop only impossible hints, preserving the order and values of
            # every legal old seed and all independently selectable configs.
            seeds = [
                seed
                for seed in seeds
                if mxfp8_shared_memory_bytes(
                    n if seed.config.get("cute_scaled_swap_ab") else m,
                    _seed_geometry_int(seed.config, "cute_scaled_tile_m", 128),
                    _seed_geometry_int(seed.config, "cute_scaled_tile_n"),
                    _seed_geometry_int(seed.config, "cute_scaled_tile_k"),
                    _seed_geometry_int(seed.config, "cute_scaled_stages"),
                    _seed_geometry_int(seed.config, "cute_scaled_cluster_m"),
                    env.config_spec.cute_mxfp8_output_element_size,
                    compact_m=bool(seed.config.get("cute_scaled_compact_m", False)),
                    sf_tma=bool(seed.config.get("cute_scaled_sf_tma", False)),
                    sf_inplace=bool(seed.config.get("cute_scaled_sf_inplace", False)),
                )
                <= 232448
            ]
        if env.config_spec.cute_mxfp8_tma_output_available:
            from ..cute.block_scaled_config import MXFP8_EPILOGUE_MODES
            from ..cute.block_scaled_config import mxfp8_epilogue_supported
            from ..cute.block_scaled_config import mxfp8_shared_memory_bytes

            shape = env.config_spec.cute_mxfp8_problem_shape
            assert shape is not None
            m, n, _k = shape
            candidates = [seed.config for seed in seeds]
            sf_modes = [(False, False)]
            if env.config_spec.cute_mxfp8_sf_tma_available:
                sf_modes += [(True, False), (True, True)]
            candidates += [
                base
                | {
                    "cute_scaled_mma": True,
                    "cute_collective_mma": False,
                    "cute_scaled_fused_scales": True,
                    "cute_scaled_compact_m": False,
                    "cute_scaled_cluster_m": 2,
                    "cute_scaled_tile_n": 224,
                    "cute_scaled_tile_k": 128,
                    "cute_scaled_stages": 6,
                    "cute_scaled_tile_m": 128,
                    "cute_scaled_swap_ab": False,
                    "cute_scaled_persistent": persistent,
                    "cute_scaled_sf_tma": sf_tma,
                    "cute_scaled_sf_inplace": sf_inplace,
                }
                for persistent in (True, False)
                for sf_tma, sf_inplace in sf_modes
            ]
            # Add joint pipeline/store choices after every existing legal seed.
            # Admission follows output metadata, packet and resource proofs;
            # problem dimensions select no geometry or algorithm.
            for candidate in candidates:
                for mode in tuple(MXFP8_EPILOGUE_MODES)[1:]:
                    if not mxfp8_epilogue_supported(
                        mode,
                        _seed_geometry_int(candidate, "cute_scaled_tile_n"),
                        _seed_geometry_int(candidate, "cute_scaled_tile_k"),
                        _seed_geometry_int(candidate, "cute_scaled_tile_m", 128),
                        swap_ab=bool(candidate.get("cute_scaled_swap_ab", False)),
                        fused_scales=bool(
                            candidate.get("cute_scaled_fused_scales", False)
                        ),
                        tma_output_available=True,
                    ):
                        continue
                    smem = mxfp8_shared_memory_bytes(
                        m,
                        _seed_geometry_int(candidate, "cute_scaled_tile_m", 128),
                        _seed_geometry_int(candidate, "cute_scaled_tile_n"),
                        _seed_geometry_int(candidate, "cute_scaled_tile_k"),
                        _seed_geometry_int(candidate, "cute_scaled_stages"),
                        _seed_geometry_int(candidate, "cute_scaled_cluster_m"),
                        env.config_spec.cute_mxfp8_output_element_size,
                        compact_m=False,
                        sf_tma=bool(candidate.get("cute_scaled_sf_tma", False)),
                        sf_inplace=bool(candidate.get("cute_scaled_sf_inplace", False)),
                        epilogue_mode=mode,
                    )
                    if smem <= 232448:
                        seed = Config.from_dict(
                            candidate | {"cute_scaled_epilogue_mode": mode}
                        )
                        if seed not in seeds:
                            seeds.append(seed)
        return seeds
