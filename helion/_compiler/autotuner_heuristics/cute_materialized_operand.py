from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from .common import dedupe_configs
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteMaterializedOperandHeuristic(AutotunerHeuristic):
    """Pair native GEMM seeds with a vectorized pointwise producer layout."""

    name = "cute_materialized_operand"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        plan = env.cute_fission_plan
        return (
            plan is not None
            and len(plan.pointwise_region_indices) == 1
            and env.config_spec.cute_tcgen05_search_enabled
        )

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        if not cls.is_eligible(env, device_ir):
            return []
        state = env.config_spec._cute_tcgen05_config
        configs = cls._with_producer_layout(
            env, device_ir, state.autotune_seed_configs()
        )
        if state.materialized_operand_pdl_roots is not None:
            # Append after the complete old prefix, including its flat producer
            # alternatives. Price ACC1 with the same complete allocation facts.
            configs.extend(
                cls._with_producer_layout(
                    env, device_ir, state._paired_pipeline_seed_configs(acc_stages=1)
                )
            )
        return dedupe_configs(configs)

    @classmethod
    def _with_producer_layout(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
        native_seeds: Sequence[Config],
    ) -> list[Config]:
        plan = env.cute_fission_plan
        assert plan is not None
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        [region_index] = plan.pointwise_region_indices
        axes = device_ir.grid_block_ids[region_index]
        if len(axes) != 2:
            return []
        row_id, column_id = axes
        # The materialization proof gives the output a contiguous final axis.
        # A 16-byte chunk is representable for the widest stored dtype; input
        # loads still apply their own stride/alignment proof. Scalar and narrow
        # alternatives remain in the ordinary per-axis search.
        region_id = device_ir.root_ids[region_index]
        element_bytes = max(
            (
                fact.dtype.itemsize
                for fact in spec.memory_op_facts
                if fact.graph_id == region_id and fact.dtype is not None
            ),
            default=1,
        )
        vector = min(8, max(1, 16 // element_bytes))
        fragments = [item._fragment(spec) for item in spec.block_sizes]
        row_slot = spec.block_sizes.block_id_to_index(row_id)
        column_slot = spec.block_sizes.block_id_to_index(column_id)
        row_fragment = fragments[row_slot]
        column_fragment = fragments[column_slot]
        column_spec = spec.block_sizes[column_slot]
        column_low = min(
            max(column_spec.min_size, column_spec.autotuner_min), column_spec.max_size
        )
        columns = min(max(128 * vector, column_low), column_fragment.high)
        vector = min(vector, columns)
        # One 16-byte packet per thread and a 128-thread CTA. The producer is a
        # single cold pass over the operand: a thread that owns several rows
        # serializes its DRAM round trips, and a CTA narrower than a warp
        # quartet leaves the SM's load queue idle. When the column tile holds
        # fewer than 128 packets, stack rows across threads instead of looping.
        column_threads = min(128, columns // vector)
        row_threads = max(
            row_fragment.low, min(128 // column_threads, row_fragment.high)
        )
        rows = row_threads
        configs = []
        for native in native_seeds:
            block_sizes = native.block_sizes.copy()
            block_sizes[row_slot] = rows
            block_sizes[column_slot] = columns
            values = native.config | {
                "block_sizes": block_sizes,
                "num_threads": cast(
                    "list[int]",
                    _seq_config_list(
                        spec.num_threads,
                        {row_id: row_threads, column_id: column_threads},
                    ),
                ),
                "cute_vector_widths": cast(
                    "list[int]",
                    _seq_config_list(spec.cute_vector_widths, {column_id: vector}),
                ),
                "cute_lane_layouts": _seq_config_list(spec.cute_lane_layouts, {}),
            }
            # The consumer's TMA role waits on the producer grid instead of the
            # launch boundary, so its prologue (barrier init, TMEM allocation,
            # descriptor prefetch) overlaps the producer's tail. The proof and
            # schedule gates decide admission; the seed only asks for it.
            if state.materialized_operand_pdl_supported(values):
                values["tcgen05_materialized_pdl"] = True
            configs.append(Config.from_dict(values))
        if tuple(axes) in spec.cute_pointwise_region_grid_groups:
            # A flat producer grid launches one CTA per tile; the persistent
            # grid above serializes tiles once there are more of them than SMs.
            configs.extend(
                [
                    Config.from_dict(
                        config.config | {"cute_pointwise_pid_type": "flat"}
                    )
                    for config in configs
                ]
            )
        return dedupe_configs(configs)
