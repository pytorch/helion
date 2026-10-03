"""Offer proved native grouped RNA schedules beside all existing schedules."""

from __future__ import annotations

from copy import deepcopy
from itertools import zip_longest
from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from ..cute.tcgen05_config import CuteTcgen05Config
from ..cute.tcgen05_flat_grouped_config import K_KEY
from ..cute.tcgen05_flat_grouped_config import K_STAGES
from ..cute.tcgen05_flat_grouped_config import PREFIX_SCAN_KEY
from ..cute.tcgen05_flat_grouped_config import RESIDENT_CTAS_KEY
from ..cute.tcgen05_flat_grouped_config import STAGES_KEY
from ..cute.tcgen05_flat_grouped_config import WARPS_KEY
from ..cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from ..cute.tcgen05_grouped_descriptors import WRAPPED
from ..cute.tcgen05_grouped_descriptors import wrapped_grouped_descriptor_supported
from ..cute.tcgen05_tma_rn import CONVERSION_KEY
from ..cute.tcgen05_tma_rn import TMA_RN
from ..cute.tcgen05_tma_rn import WARP_RAW
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


def interleave_grouped_rna_seeds(configs: list[Config]) -> list[Config]:
    """Keep the existing first seed/order and expose the new schedule family.

    No old seed is removed or changed. This changes only the merged order when
    both complete native choices and ordinary choices are present, so a large
    family of ordinary tile/recipe variants cannot consume the entire prefix.
    """

    def is_native(config: Config) -> bool:
        return bool(config.get(WARPS_KEY, 0)) or config.get(CONVERSION_KEY) in (
            TMA_RN,
            WARP_RAW,
        )

    ordinary = [config for config in configs if not is_native(config)]
    native = [config for config in configs if is_native(config)]
    if not ordinary or not native:
        return configs
    return [
        config
        for pair in zip_longest(ordinary, native)
        for config in pair
        if config is not None
    ]


class CuteGroupedRnaHeuristic(AutotunerHeuristic):
    name = "cute_grouped_rna"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.grouped_warp_tf32_plan import grouped_warp_tf32_plan
        from ..cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
        from ..cute.tcgen05_flat_grouped_plan import flat_grouped_rna_schedule

        available: list[int] = []
        warp_raw_available = False
        wrapped_available = False
        two_stage_ctas: list[int] = []
        block_prefix_recipes: list[tuple[int, int, int]] = []
        block_prefix_seed_enabled = False
        host = device_ir.host_function
        if host is not None and env.settings.static_shapes:
            with host:
                proof = prove_flat_grouped_rna(env, device_ir)
                if proof is not None:
                    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
                    wrapped_available = wrapped_grouped_descriptor_supported(
                        proof.b.shape[0],
                        proof.a.shape[0],
                        proof.a.shape[1],
                        proof.b.shape[2],
                        proof.offset_bits,
                    )
                    # Seed registration precedes the runtime's immutable storage
                    # snapshot. Evaluate that same registered, metadata-only
                    # classifier here; restore the prior snapshot so unrelated
                    # seed families keep their existing registration behavior.
                    saved = env.bound_runtime_input_specialization_results
                    try:
                        if env.runtime_arg_values_by_name:
                            env.snapshot_runtime_input_specialization_results(
                                env.runtime_arg_values_by_name
                            )
                        warp_raw_available = (
                            grouped_warp_tf32_plan(env, proof, shared_capacity=capacity)
                            is not None
                        )
                        for block_k, stages in K_STAGES.items():
                            if (
                                flat_grouped_rna_schedule(
                                    env,
                                    proof,
                                    converter_warps=4,
                                    block_k=block_k,
                                    ab_stages=stages,
                                    shared_capacity=capacity,
                                )
                                is not None
                            ):
                                available.append(block_k)
                        if 32 in available:
                            for ctas in (1, 2):
                                if (
                                    flat_grouped_rna_schedule(
                                        env,
                                        proof,
                                        converter_warps=0,
                                        block_k=32,
                                        ab_stages=2,
                                        shared_capacity=capacity,
                                        tma_rn=True,
                                        resident_ctas=ctas,
                                    )
                                    is not None
                                ):
                                    two_stage_ctas.append(ctas)
                        for block_k, stages, ctas in (
                            *((k, K_STAGES[k], 1) for k in available),
                            *((32, 2, ctas) for ctas in two_stage_ctas),
                        ):
                            if (
                                flat_grouped_rna_schedule(
                                    env,
                                    proof,
                                    converter_warps=0,
                                    block_k=block_k,
                                    ab_stages=stages,
                                    shared_capacity=capacity,
                                    tma_rn=True,
                                    resident_ctas=ctas,
                                    prefix_scan="block",
                                )
                                is not None
                            ):
                                block_prefix_recipes.append((block_k, stages, ctas))
                        block_prefix_seed_enabled = (
                            bool(block_prefix_recipes) and proof.b.shape[0] > 32
                        )
                    finally:
                        env.bound_runtime_input_specialization_results = saved
        env.config_spec.cute_grouped_rna_k_choices = tuple(available)
        env.config_spec.cute_grouped_warp_tf32_available = warp_raw_available
        env.config_spec.cute_grouped_rn_two_stage_ctas = tuple(two_stage_ctas)
        env.config_spec.cute_grouped_block_prefix_recipes = tuple(block_prefix_recipes)
        env.config_spec.cute_grouped_block_prefix_seed_enabled = (
            block_prefix_seed_enabled
        )
        env.config_spec.cute_grouped_wrapped_descriptors_available = (
            bool(available) and wrapped_available
        )
        # Only static bindings are admitted. Their complete tensor metadata,
        # device and alignment facts already belong to the ordinary bind key.
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return (
            bool(env.config_spec.cute_grouped_rna_k_choices)
            or env.config_spec.cute_grouped_warp_tf32_available
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
        spec = env.config_spec
        base = spec._base_default_config()
        # Every seed is the complete fixed physical schedule. The ordinary
        # base supplies only the retained host construction/fallback path.
        seeds = [
            Config.from_dict(deepcopy(base.config) | {WARPS_KEY: warps, K_KEY: block_k})
            for block_k in spec.cute_grouped_rna_k_choices
            for warps in (4, 8, 2, 1)
        ]

        if spec.cute_grouped_wrapped_descriptors_available:
            seeds.extend(
                Config.from_dict(deepcopy(seed.config) | {DESCRIPTOR_KEY: WRAPPED})
                for seed in tuple(seeds)
            )
        # Append a distinct approximation policy after the exact old RNA seed
        # prefix. All admitted K/stage choices remain searchable independently.
        rn_seeds = [
            Config.from_dict(
                deepcopy(base.config) | {CONVERSION_KEY: TMA_RN, K_KEY: block_k}
            )
            for block_k in spec.cute_grouped_rna_k_choices
        ]
        if spec.cute_grouped_wrapped_descriptors_available:
            rn_seeds.extend(
                Config.from_dict(deepcopy(seed.config) | {DESCRIPTOR_KEY: WRAPPED})
                for seed in tuple(rn_seeds)
            )
        seeds.extend(rn_seeds)
        low_stage_seeds = [
            Config.from_dict(
                deepcopy(base.config)
                | {
                    CONVERSION_KEY: TMA_RN,
                    K_KEY: 32,
                    STAGES_KEY: 2,
                    **({RESIDENT_CTAS_KEY: ctas} if ctas != 1 else {}),
                }
            )
            for ctas in spec.cute_grouped_rn_two_stage_ctas
        ]
        if spec.cute_grouped_wrapped_descriptors_available:
            low_stage_seeds.extend(
                Config.from_dict(deepcopy(seed.config) | {DESCRIPTOR_KEY: WRAPPED})
                for seed in tuple(low_stage_seeds)
            )
        seeds.extend(low_stage_seeds)
        if spec.cute_grouped_warp_tf32_available:
            seeds.append(
                Config.from_dict(deepcopy(base.config) | {CONVERSION_KEY: WARP_RAW})
            )
        # Preserve the entire previous seed prefix, including raw-warp TF32.
        # Only complete resource-admitted RN recipes gain a block-scan sibling.
        if spec.cute_grouped_block_prefix_seed_enabled:
            seeds.extend(
                Config.from_dict(deepcopy(seed.config) | {PREFIX_SCAN_KEY: "block"})
                for seed in (*rn_seeds, *low_stage_seeds)
                if (
                    seed[K_KEY],
                    seed.get(STAGES_KEY, K_STAGES[cast("int", seed[K_KEY])]),
                    seed.get(RESIDENT_CTAS_KEY, 1),
                )
                in spec.cute_grouped_block_prefix_recipes
            )
        return seeds
