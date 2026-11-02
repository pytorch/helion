"""Qualify legal MXFP8 output-transfer modes after the nominal seed budget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..cute.block_scaled_config import MXFP8_EPILOGUE_MODES
from ..cute.block_scaled_config import mxfp8_shared_memory_bytes

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


KEY = "cute_scaled_epilogue_mode"


def register_mxfp8_epilogue_coverage(env: CompileEnvironment) -> None:
    """Retain the ordinary seed order and qualify the coupled SF carriers.

    The existing shape-independent seed family supplies the geometry. This
    registration changes neither the field domain nor any seed/default config;
    the generic population builder appends and pins at most four witnesses.
    """
    spec = env.config_spec
    if (
        env.backend_name != "cute"
        or spec.target_device_capability is None
        or spec.target_device_capability[0] != 10
        or not spec.cute_mxfp8_mma_available
        or not spec.cute_mxfp8_fused_scales_available
        or not spec.cute_mxfp8_sf_tma_available
        or not spec.cute_mxfp8_tma_output_available
        or spec.cute_mxfp8_problem_shape is None
    ):
        return
    # These are compiler hint coordinates, independent of problem dimensions.
    # The complete joint family is needed: changing only the output mode on a
    # compact/swapped carrier is deliberately rejected by strict normalization.
    geometry = {
        "cute_scaled_mma": True,
        "cute_scaled_fused_scales": True,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_tile_m": 128,
        "cute_scaled_tile_n": 224,
        "cute_scaled_tile_k": 128,
        "cute_scaled_stages": 6,
        "cute_scaled_cluster_m": 2,
        "cute_scaled_swap_ab": False,
        "cute_scaled_compact_m": False,
    }
    generation = spec.create_config_generation()
    carriers: dict[bool, Config] = {}
    for seed in spec.compiler_seed_configs:
        if any(seed.config.get(key) != value for key, value in geometry.items()):
            continue
        persistent = seed.config.get("cute_scaled_persistent")
        if type(persistent) is not bool or persistent in carriers:
            continue
        try:
            _, carrier = generation.strict_config_pair(
                Config.from_dict(seed.config | {KEY: "default"})
            )
            for mode in tuple(MXFP8_EPILOGUE_MODES)[1:]:
                _, requested = generation.strict_config_pair(
                    Config.from_dict(carrier.config | {KEY: mode})
                )
                if requested.config.get(KEY) != mode:
                    raise InvalidConfig("MXFP8 epilogue witness was normalized away")
                m, _n, _k = spec.cute_mxfp8_problem_shape
                if (
                    mxfp8_shared_memory_bytes(
                        m,
                        128,
                        224,
                        128,
                        6,
                        2,
                        spec.cute_mxfp8_output_element_size,
                        compact_m=False,
                        sf_tma=True,
                        sf_inplace=True,
                        epilogue_mode=mode,
                    )
                    > 232448
                ):
                    raise InvalidConfig(
                        "MXFP8 epilogue carrier exceeds shared capacity"
                    )
        except (InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
            continue
        carriers[persistent] = carrier
    if not carriers:
        return
    primary = True in carriers
    witnesses = tuple(
        CoverageWitness(carriers[primary], mode)
        for mode in tuple(MXFP8_EPILOGUE_MODES)[1:]
    )
    supplemental = (
        (CoverageWitness(carriers[not primary], "tma_prefetch_2"),)
        if (not primary) in carriers
        else ()
    )
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.mxfp8_output_transfer",
            version=1,
            key=KEY,
            domain=tuple(MXFP8_EPILOGUE_MODES),
            legacy="default",
            witnesses=witnesses,
            supplemental_witnesses=supplemental,
        )
    )
