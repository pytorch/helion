"""Seeds for ordered reductions over a logical row tile."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ...runtime.config import Config
from ..cute.loop_nesting import sibling_row_loop_blocks
from .common import dedupe_configs
from .cute import _cute_reread_cache_policies
from .cute import _cute_tile_inner_block_dtype
from .cute import _cute_tile_seed_vec_width_for_dtype
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteResidentSequenceHeuristic(AutotunerHeuristic):
    """Expose a structural superset; codegen proves the exact axis and effects."""

    name = "cute_resident_sequence"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @staticmethod
    def _plan(
        env: CompileEnvironment, device_ir: DeviceIR
    ) -> tuple[int, tuple[int, ...]] | None:
        from ..device_ir import ForLoopGraphInfo
        from ..device_ir import RootGraphInfo

        if env.config_spec.matmul_facts or any(
            not isinstance(info, (RootGraphInfo, ForLoopGraphInfo))
            for info in device_ir.graphs
        ):
            return None
        plan = sibling_row_loop_blocks(env, device_ir, device_ir.graphs)
        if plan is None:
            return None
        reductions = {
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.mean.dim,
            torch.ops.aten.amax.default,
            torch.ops.aten.amin.default,
            torch.ops.aten.prod.dim_int,
        }
        if not any(
            isinstance(info, ForLoopGraphInfo)
            and sum(node.target in reductions for node in info.graph.nodes) >= 1
            for info in device_ir.graphs
        ):
            return None
        return plan

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        plan = cls._plan(env, device_ir)
        if plan is None:
            return frozenset()
        env.config_spec.cute_sequence_reduction_blocks.update(plan[1])
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._plan(env, device_ir) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        plan = cls._plan(env, device_ir)
        if plan is None:
            return None
        row, columns = plan
        spec = env.config_spec
        extent = env.block_sizes[columns[0]].size_hint()
        if extent <= 0:
            return None
        full_width = 1 << (extent - 1).bit_length()
        dtype = _cute_tile_inner_block_dtype(env, device_ir, columns[0])
        vector = _cute_tile_seed_vec_width_for_dtype(dtype)
        axes = {row, *columns}
        if any(
            len(item.block_ids) != 1 or item.block_id not in axes
            for item in spec.block_sizes
        ):
            return None
        cache_policies = _cute_reread_cache_policies(spec, columns)
        seeds = []
        for rows, threads in (
            (4, 32),
            (1, 128),
            (1, 256),
            (4, 64),
            (4, 128),
            (1, 512),
            (1, 1024),
        ):
            # A vector-only scalarization can omit DeviceLoopState's lane
            # wrapper. Keep at least two lanes in these optional seeds; the
            # proof deliberately declines regions without that owned wrapper.
            vector_limit = full_width // (2 * threads)
            if full_width // threads > 128 or vector_limit < 1:
                continue
            seed_vector = min(vector, vector_limit)
            sizes = {row: rows, **dict.fromkeys(columns, full_width)}
            if any(
                not item.min_size <= sizes[item.block_id] <= item.max_size
                for item in spec.block_sizes
            ):
                continue
            for schedule in ("resident", "reload"):
                seeds.append(
                    Config(
                        block_sizes=cast(
                            "list[int]", _seq_config_list(spec.block_sizes, sizes)
                        ),
                        num_threads=cast(
                            "list[int]",
                            _seq_config_list(
                                spec.num_threads,
                                {row: rows, **dict.fromkeys(columns, threads)},
                            ),
                        ),
                        cute_vector_widths=cast(
                            "list[int]",
                            _seq_config_list(
                                spec.cute_vector_widths,
                                {row: 1, **dict.fromkeys(columns, seed_vector)},
                            ),
                        ),
                        cute_lane_layouts=cast(
                            "list[str]",
                            _seq_config_list(
                                spec.cute_lane_layouts,
                                {row: "blocked", **dict.fromkeys(columns, "strided")},
                            ),
                        ),
                        cute_reduction_sequence=schedule,
                        cute_cluster_n=1,
                        pid_type="flat",
                    )
                )
                if schedule == "reload" and any(cache_policies):
                    seeds.append(
                        Config.from_dict(
                            {
                                **seeds[-1].config,
                                "load_eviction_policies": cache_policies,
                            }
                        )
                    )
        return dedupe_configs(seeds) or None
