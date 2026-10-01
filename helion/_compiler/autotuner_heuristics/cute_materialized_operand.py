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
        # Retain the original persistent seeds even though independent
        # pointwise grids can now search below the occupancy heuristic floor.
        row_spec = spec.block_sizes[row_slot]
        column_spec = spec.block_sizes[column_slot]
        rows = min(max(row_spec.min_size, row_spec.autotuner_min), row_spec.max_size)
        column_low = min(
            max(column_spec.min_size, column_spec.autotuner_min), column_spec.max_size
        )
        columns = min(max(128 * vector, column_low), column_fragment.high)
        vector = min(vector, columns)
        threads = min(128, columns // vector)
        configs = []
        for native in native_seeds:
            block_sizes = native.block_sizes.copy()
            block_sizes[row_slot] = rows
            block_sizes[column_slot] = columns
            configs.append(
                Config.from_dict(
                    native.config
                    | {
                        "block_sizes": block_sizes,
                        "num_threads": cast(
                            "list[int]",
                            _seq_config_list(
                                spec.num_threads, {row_id: 1, column_id: threads}
                            ),
                        ),
                        "cute_vector_widths": cast(
                            "list[int]",
                            _seq_config_list(
                                spec.cute_vector_widths, {column_id: vector}
                            ),
                        ),
                        "cute_lane_layouts": _seq_config_list(
                            spec.cute_lane_layouts, {}
                        ),
                    }
                )
            )
        if tuple(axes) in spec.cute_pointwise_region_grid_groups:
            flat_configs = []
            for config in configs:
                block_sizes = config.block_sizes.copy()
                block_sizes[row_slot] = row_fragment.low
                flat_configs.append(
                    Config.from_dict(
                        config.config
                        | {
                            "block_sizes": block_sizes,
                            "cute_pointwise_pid_type": "flat",
                        }
                    )
                )
            configs.extend(flat_configs)
        return dedupe_configs(configs)
