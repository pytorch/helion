from __future__ import annotations

from typing import TYPE_CHECKING

from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.seed_coverage import CompilerSeedWitness
from ...exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment


def chained_seed_witnesses(
    env: CompileEnvironment, seeds: Sequence[Config]
) -> tuple[CompilerSeedWitness, ...]:
    """Bounded common-graph compositions, with ordinary lowering admission.

    These are supplemental search requests, not certificates of codegen
    legality. No default, legacy seed, field or sampler is changed.
    """
    spec = env.config_spec
    fields = spec._flat_fields()
    result: list[CompilerSeedWitness] = []
    seen = set(seeds)

    def add(
        mechanism: str, parent: Config, additions: dict[str, object]
    ) -> Config | None:
        if not additions.keys() <= fields.keys():
            return None
        proposed = parent.config | additions
        if any(
            proposed.get(key) != value
            for key, value in env.settings.autotune_config_overrides.items()
        ):
            return None
        try:
            config = spec.normalized_config(proposed)
        except InvalidConfig:
            return None
        for key, value in additions.items():
            field = fields[key]
            if not isinstance(field, ConfigSpecFragment):
                return None
            if config.config.get(key, field.default()) != value:
                return None
        if config not in seen:
            seen.add(config)
            result.append(CompilerSeedWitness(mechanism, config))
        return config

    for seed in seeds:
        values = seed.config
        if (
            values.get("cute_chained_mma_schedule") == "cp_async"
            and values.get("cute_chained_pointwise_schedule") == "warp_rows_parallel"
            and values.get("cute_chained_operand_coordinates") is True
            and values.get("cute_chained_completed_image") is True
            and values.get("cute_native_matmul_metadata") is True
            and values.get("cute_chained_scan_input_materialize") is True
        ):
            raw = add(
                "joined_root_raw_export",
                seed,
                {
                    "cute_chained_pointwise_cache_bytes": 4096,
                    "cute_chained_row_panel_cache": True,
                    "cute_chained_row_subwarp_packing": True,
                    "cute_chained_row_frontier": True,
                    "cute_chained_mma_frontier": True,
                    "cute_chained_raw_operand_exports": True,
                },
            )
            if raw is not None:
                add(
                    "joined_root_endpoint_export_schedule",
                    raw,
                    {
                        "cute_chained_c_endpoint_reuse": True,
                        "cute_chained_export_unroll": 4,
                    },
                )
                break

    for seed in seeds:
        values = seed.config
        if (
            values.get("cute_chained_mma_schedule") == "tcgen05_tmem"
            and values.get("cute_chained_rectangular_async") is True
            and values.get("cute_chained_prepared_group_multi_atom") is True
        ):
            if (
                add(
                    "completed_member_store_rematerialization",
                    seed,
                    {
                        "cute_chained_scratch_layout": "xor",
                        "cute_chained_pointwise_unroll": 8,
                        "cute_native_matmul_metadata": True,
                        "cute_chained_grouped_fragment_bridge": False,
                        "cute_chained_completed_member_store": True,
                        "cute_chained_fragment_reuse": True,
                        "cute_chained_store_rematerialize": True,
                    },
                )
                is not None
            ):
                break
    return tuple(result)
