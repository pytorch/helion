"""Search resident feature carries inside serial row loops."""

from __future__ import annotations

from copy import deepcopy
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...runtime.config import Config
from ..cute.loop_nesting import tile_loop_paths
from ..cute.thread_budget import MAX_THREADS_PER_BLOCK
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


@dataclasses.dataclass(frozen=True)
class _ResidentPlan:
    coarse_block: int
    row_block: int
    reduction_block: int
    features: int


class CuteResidentReductionHeuristic(AutotunerHeuristic):
    """A structural superset; the AST matcher proves effects and carries."""

    name = "cute_resident_reduction"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @staticmethod
    def _plan(env: CompileEnvironment, device_ir: DeviceIR) -> _ResidentPlan | None:
        from ..device_ir import ForLoopGraphInfo
        from ..device_ir import RootGraphInfo

        spec = env.config_spec
        paths = tile_loop_paths(device_ir, device_ir.graphs)
        if (
            spec.matmul_facts
            or len(paths) != 1
            or len(paths[0]) != 2
            or any(len(level) != 1 for level in paths[0])
            or len(spec.block_sizes) != 2
            or any(
                not isinstance(info, (RootGraphInfo, ForLoopGraphInfo))
                for info in device_ir.graphs
            )
        ):
            return None
        coarse, row = (level[0] for level in paths[0])
        reductions = [block for block in env.block_sizes if block.reduction]
        if len(reductions) != 1:
            return None
        (reduction,) = reductions
        if (
            reduction.block_id in spec.cute_indexed_reduction_block_ids
            or not isinstance(reduction.numel, (int, sympy.Integer))
        ):
            return None
        features = int(reduction.numel)
        if features <= 0 or features & (features - 1):
            return None
        for info in device_ir.graphs:
            for node in info.graph.nodes:
                if node.target not in (
                    torch.ops.aten.sum.dim_IntList,
                    torch.ops.aten.mean.dim,
                ):
                    continue
                argument = node.args[0]
                output = node.meta.get("val")
                if (
                    argument is None
                    and node.target == torch.ops.aten.mean.dim
                    and isinstance(output, torch.Tensor)
                    and output.dtype == torch.float32
                    and node.args[1] == [-1]
                ):
                    # Lowering splits mean into a sum and division, erasing
                    # this argument. The static full-width reduction block
                    # establishes a searchable shape; the AST pass still
                    # requires explicit FP32 sum inputs and carries.
                    return _ResidentPlan(coarse, row, reduction.block_id, features)
                if not isinstance(argument, torch.fx.Node):
                    continue
                value = argument.meta.get("val")
                if not (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 1
                    and value.dtype == torch.float32
                    and isinstance(value.shape[-1], int)
                    and value.shape[-1] == features
                    and isinstance(output, torch.Tensor)
                    and output.dtype == torch.float32
                ):
                    continue
                dimensions = node.args[1]
                if isinstance(dimensions, (list, tuple)) and tuple(dimensions) in (
                    (-1,),
                    (value.ndim - 1,),
                ):
                    return _ResidentPlan(coarse, row, reduction.block_id, features)
        return None

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        plan = cls._plan(env, device_ir)
        if plan is None:
            return frozenset()
        env.config_spec.cute_resident_reduction_blocks.add(plan.reduction_block)
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
        spec = env.config_spec
        if any(
            len(item.block_ids) != 1
            or item.block_id not in (plan.coarse_block, plan.row_block)
            for item in spec.block_sizes
        ):
            return None
        seeds = []
        for coarse, threads, vector, rows, schedule in (
            (32, 256, 8, 2, "pipelined"),
            (32, 256, 8, 1, "resident"),
            (32, 128, 4, 1, "pipelined"),
            (16, 512, 8, 2, "pipelined"),
            (32, 512, 8, 4, "pipelined"),
        ):
            block_sizes = {plan.coarse_block: coarse, plan.row_block: 1}
            if plan.features % (threads * vector) or any(
                not item.min_size <= block_sizes[item.block_id] <= item.max_size
                for item in spec.block_sizes
            ):
                continue
            seeds.append(
                Config(
                    block_sizes=cast(
                        "list[int]", _seq_config_list(spec.block_sizes, block_sizes)
                    ),
                    num_threads=[
                        threads if item.block_id == plan.reduction_block else 1
                        for item in spec.num_threads
                    ],
                    cute_vector_widths=[
                        vector if item.block_id == plan.reduction_block else 1
                        for item in spec.cute_vector_widths
                    ],
                    cute_lane_layouts=["strided" for item in spec.cute_lane_layouts],
                    cute_reduction_reloads=[
                        "register" for item in spec.cute_reduction_reloads
                    ],
                    cute_reduction_schedule=schedule,
                    cute_reduction_group_rows=rows,
                    cute_cluster_n=1,
                    pid_type="flat",
                )
            )
        return seeds or None

    @classmethod
    def pipeline_depth_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        """Try deeper prefetch without increasing the resident row group.

        Reuse the typed feature/thread/vector lattice of the existing one-row
        pipeline seeds. A sixteen-row coarse tile amortizes the three-group
        prefill while keeping more producer CTAs available. The AST matcher
        still proves every access and charges all four shared-memory slots.

        These are appended after every old compiler seed, including host-tail
        overlays and serial-row seeds; they never replace the default.
        """
        plan = cls._plan(env, device_ir)
        if plan is None:
            return []
        spec = env.config_spec
        blocks = {plan.coarse_block: 16, plan.row_block: 1}
        if any(
            len(item.block_ids) != 1
            or item.block_id not in blocks
            or not item._fragment(spec).low
            <= blocks[item.block_id]
            <= item._fragment(spec).high
            for item in spec.block_sizes
        ):
            return []
        result = []
        for seed in cls.get_seed_configs(env, device_ir) or ():
            if (
                seed.config["cute_reduction_schedule"] != "pipelined"
                or seed.config["cute_reduction_group_rows"] != 1
            ):
                continue
            values = deepcopy(seed.config) | {
                "block_sizes": _seq_config_list(spec.block_sizes, blocks),
                "cute_reduction_pipeline_depth": 4,
                "cute_reduction_sequence": "scalar",
                "cute_min_blocks_per_mp": 0,
            }
            # SIMT has no PID search coordinate here. Omission already means
            # flat; keeping an explicit key would not survive strict transfer.
            values.pop("pid_type")
            if spec.cute_host_paired_sum_available:
                result.extend(
                    Config.from_dict(deepcopy(values) | {"cute_host_paired_sum": mode})
                    for mode in ("off", "mapped", "narrow")
                )
            else:
                result.append(Config.from_dict(values))
        return result

    @classmethod
    def serial_row_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        """Keep a short serial row group on the full feature-thread lattice.

        Vectorized resident seeds couple a large coarse tile with row tile one.
        A small coarse tile and matching serial row tile trade fewer partial
        rows for more work in each producer CTA. This family uses one thread per
        feature and keeps all other axes serial; no vector or cluster schedule
        needs to be inferred from historical positional configurations.

        The assembler appends these after the old host-tail overlay, preserving
        its primary producer and every existing compiler seed in order.
        """
        plan = cls._plan(env, device_ir)
        if plan is None or not 32 <= plan.features <= MAX_THREADS_PER_BLOCK:
            return []
        spec = env.config_spec
        axes = {plan.coarse_block, plan.row_block, plan.reduction_block}
        if len(axes) != 3 or any(len(item.block_ids) != 1 for item in spec.block_sizes):
            return []
        blocks = {item.block_id: item for item in spec.block_sizes}
        threads = {
            plan.coarse_block: 1,
            plan.row_block: 1,
            plan.reduction_block: plan.features,
        }
        if (
            set(blocks) != {plan.coarse_block, plan.row_block}
            or set(spec.num_threads.valid_block_ids()) != axes
            or not axes.issubset(spec.cute_vector_widths.valid_block_ids())
            or not axes.issubset(spec.cute_lane_layouts.valid_block_ids())
            or any(
                not item._fragment(spec).low <= 4 <= item._fragment(spec).high
                for item in spec.block_sizes
            )
            or any(
                threads[item.block_id]
                not in (item._fragment(spec).search_values() or ())
                for item in spec.num_threads
            )
        ):
            return []
        values = {
            "block_sizes": _seq_config_list(spec.block_sizes, dict.fromkeys(blocks, 4)),
            "num_threads": _seq_config_list(spec.num_threads, threads),
            "cute_vector_widths": _seq_config_list(
                spec.cute_vector_widths, dict.fromkeys(axes, 1)
            ),
            "cute_lane_layouts": _seq_config_list(
                spec.cute_lane_layouts, dict.fromkeys(axes, "strided")
            ),
            "cute_reduction_reloads": ["auto" for item in spec.cute_reduction_reloads],
            "cute_reduction_schedule": "resident",
            "cute_reduction_sequence": "scalar",
            "cute_reduction_group_rows": 1,
            "load_eviction_policies": spec.load_eviction_policies.default(),
            "cute_cluster_n": 1,
            "cute_min_blocks_per_mp": 0,
        }
        if spec.cute_host_paired_sum_available:
            return [
                Config.from_dict(deepcopy(values) | {"cute_host_paired_sum": mode})
                for mode in ("off", "mapped", "narrow")
            ]
        return [Config.from_dict(values)]

    @classmethod
    def serial_output_carrier(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        """A short staged row group with one live row and a vector output.

        Every dimension is mapped by its structural block ID. The lowering
        independently proves uniform row masks, disjoint outputs, FP32 carries
        and the terminal product. This carrier never replaces an old seed.
        """
        plan = cls._plan(env, device_ir)
        if plan is None:
            return None
        spec = env.config_spec
        if (
            spec.target_device_capability is None
            or spec.target_device_capability[0] != 10
        ):
            return None
        blocks = {plan.coarse_block: 64, plan.row_block: 1}
        axes = {plan.coarse_block, plan.row_block, plan.reduction_block}
        if (
            plan.features % (256 * 2)
            or any(
                len(item.block_ids) != 1
                or item.block_id not in blocks
                or not item._fragment(spec).low
                <= blocks[item.block_id]
                <= item._fragment(spec).high
                for item in spec.block_sizes
            )
            or set(spec.num_threads.valid_block_ids()) != axes
        ):
            return None
        values = {
            "block_sizes": _seq_config_list(spec.block_sizes, blocks),
            "num_threads": _seq_config_list(
                spec.num_threads,
                {axis: 256 if axis == plan.reduction_block else 1 for axis in axes},
            ),
            "cute_vector_widths": _seq_config_list(
                spec.cute_vector_widths,
                {axis: 1 if axis == plan.coarse_block else 2 for axis in axes},
            ),
            "cute_lane_layouts": _seq_config_list(
                spec.cute_lane_layouts,
                {
                    axis: "blocked" if axis == plan.reduction_block else "strided"
                    for axis in axes
                },
            ),
            "cute_reduction_reloads": ["register" for _ in spec.cute_reduction_reloads],
            "cute_reduction_sequence": "scalar",
            "cute_reduction_schedule": "pipelined",
            "cute_reduction_pipeline_depth": 2,
            "cute_reduction_group_rows": 3,
            "cute_cluster_n": 1,
            "cute_min_blocks_per_mp": 1,
            "load_eviction_policies": spec.load_eviction_policies.default(),
        }
        if spec.cute_host_paired_sum_available:
            values["cute_host_paired_sum"] = "narrow"
        return Config.from_dict(values)
