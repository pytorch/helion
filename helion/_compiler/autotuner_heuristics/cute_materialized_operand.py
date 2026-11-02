from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from .common import dedupe_configs
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...autotuner.config_spec import MemoryOpFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _contiguous_axis(facts: Sequence[MemoryOpFact], block_id: int) -> bool:
    """Whether every op addressing ``block_id`` does so at unit stride and scale."""
    addressed = False
    for fact in facts:
        for position, subscript_block_id in enumerate(fact.subscript_block_ids):
            if subscript_block_id != block_id:
                continue
            addressed = True
            if (
                position >= len(fact.subscript_strides)
                or fact.subscript_strides[position] != 1
                or (
                    position < len(fact.subscript_index_scales)
                    and fact.subscript_index_scales[position] != 1
                )
            ):
                return False
    return addressed


class CuteMaterializedOperandHeuristic(AutotunerHeuristic):
    """Pair native GEMM seeds with a vectorized pointwise producer layout.

    The pointwise launch is either a fission plan's proved pointwise region or
    the matmul-free phase of an ``hl.barrier()`` chain launched beside a
    native GEMM (``cute_pointwise_region_grid_groups``).  Either way it is a
    separate device function whose SIMT layout knobs the tcgen05 surface
    keeps, so each native seed is paired with one 16-byte packet per thread
    along the launch's contiguous axis.
    """

    name = "cute_materialized_operand"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return (
            env.config_spec.cute_tcgen05_search_enabled
            and cls._pointwise_region(env, device_ir) is not None
        )

    @classmethod
    def _pointwise_region(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> tuple[int, tuple[int, ...]] | None:
        """``(region index, grid axes)`` of the single pointwise launch."""
        plan = env.cute_fission_plan
        if plan is not None:
            if len(plan.pointwise_region_indices) != 1:
                return None
            [index] = plan.pointwise_region_indices
            return index, tuple(device_ir.grid_block_ids[index])
        groups = env.config_spec.cute_pointwise_region_grid_groups
        if len(groups) != 1:
            return None
        [axes] = groups
        indices = [
            index
            for index, grid in enumerate(device_ir.grid_block_ids)
            if tuple(grid) == tuple(axes)
        ]
        if len(indices) != 1:
            return None
        return indices[0], tuple(axes)

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        if not cls.is_eligible(env, device_ir):
            return []
        spec = env.config_spec
        state = spec._cute_tcgen05_config
        native_seeds = state.autotune_seed_configs()
        if env.cute_fission_plan is None:
            if state.materialized_matmul_block_ids:
                from .cute_materialized import CuteMaterializedMmaHeuristic

                native_seeds = CuteMaterializedMmaHeuristic.get_seed_configs(
                    env, device_ir
                )
            if not native_seeds:
                # A chain whose GEMM phase has no complete pipeline proof
                # has no native seeds; the layout still deserves its head
                # start on the default schedule.
                native_seeds = [spec.default_config()]
        configs = cls._with_producer_layout(env, device_ir, native_seeds)
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
        from ..cute.materialized_fission_codegen import _region_graph_ids

        spec = env.config_spec
        state = spec._cute_tcgen05_config
        region = cls._pointwise_region(env, device_ir)
        assert region is not None
        region_index, axes = region
        if len(axes) != 2:
            return []
        region_graph_ids = _region_graph_ids(
            device_ir.graphs, device_ir.root_ids[region_index]
        )
        facts = [
            fact for fact in spec.memory_op_facts if fact.graph_id in region_graph_ids
        ]
        if env.cute_fission_plan is not None:
            # The materialization proof gives the output a contiguous final
            # axis.
            row_id, column_id = axes
        else:
            # A barrier phase carries no such proof: the packet axis is the
            # one every memory op of the phase addresses at stride 1.
            contiguous = [
                block_id for block_id in axes if _contiguous_axis(facts, block_id)
            ]
            if len(contiguous) != 1:
                return []
            [column_id] = contiguous
            [row_id] = [block_id for block_id in axes if block_id != column_id]
        # A 16-byte chunk is representable for the widest stored dtype; input
        # loads still apply their own stride/alignment proof. Scalar and narrow
        # alternatives remain in the ordinary per-axis search.
        element_bytes = max(
            (fact.dtype.itemsize for fact in facts if fact.dtype is not None),
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
        if tuple(axes) in spec.cute_pointwise_region_grid_groups and any(
            pid_type != "flat" for pid_type in spec.allowed_pid_types
        ):
            # A flat producer grid launches one CTA per tile; the persistent
            # grid above serializes tiles once there are more of them than SMs
            # (a 128-thread CTA per SM cannot cover DRAM latency, and the
            # dependent GEMM's wait stretches with it), so the flat twins go
            # first: the search population keeps the first seeds in list order,
            # and a persistent twin that fits is one knob away from its flat
            # twin either way. (When every launch is flat already the knob is
            # inert and off the surface.)
            flat = [
                Config.from_dict(config.config | {"cute_pointwise_pid_type": "flat"})
                for config in configs
            ]
            configs = [*flat, *configs]
        return dedupe_configs(configs)
