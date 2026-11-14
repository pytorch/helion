"""Search resident feature carries inside serial row loops."""

from __future__ import annotations

from copy import deepcopy
import dataclasses
import logging
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ....._compat import num_compute_units
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...loop_nesting import tile_loop_paths
from ...thread_budget import MAX_THREADS_PER_BLOCK
from .cute import _seq_config_list

if TYPE_CHECKING:
    from .....autotuner.config_spec import ConfigSpec
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _ResidentPlan:
    coarse_block: int
    row_block: int
    reduction_block: int
    features: int


# (coarse rows per CTA, threads over the features, features per thread,
# resident row group, schedule): the typed lattice the seeds are drawn from.
_SEED_LATTICE: tuple[tuple[int, int, int, int, str], ...] = (
    (32, 256, 8, 2, "pipelined"),
    (32, 256, 8, 1, "resident"),
    (32, 128, 4, 1, "pipelined"),
    (16, 512, 8, 2, "pipelined"),
    (32, 512, 8, 4, "pipelined"),
)
# Feature vectors a seed may carry per thread: 16 bytes of fp32 or more.
_SEED_VECTORS = (4, 8)


def _seed_vector(features: int, threads: int, vector: int) -> int | None:
    """The feature vector per thread for ``threads`` lanes: the lattice value
    when it tiles the features exactly, else the exact quotient when it is a
    seed width (1024 features over 256 threads: 4, not the lattice's 8)."""
    if features % (threads * vector) == 0:
        return vector
    if features % threads == 0 and features // threads in _SEED_VECTORS:
        return features // threads
    return None


def _vector_row_entries(features: int) -> list[tuple[int, int, int, int, str]]:
    """One 8-wide feature vector per thread over two- and four-row groups,
    for features no 8-wide lattice entry tiles.

    The lattice gives 2048 and 4096 features a 16-byte bf16 load per thread
    (256 or 512 threads of 8); below that it offers at most 128 threads of 4
    (512 and 1024 features; nothing for 256). These entries give 256, 512
    and 1024 features 32, 64 and 128 threads of 8, whatever the row loop's
    form.
    512 bf16 features (better-benchmark's mix-order RMSNorm backward) ran
    15-50% faster as 64 threads of 8 with four-row groups than as 128
    threads of 4, matching Triton's kernel.
    """
    if any(
        vector == 8 and features % (threads * vector) == 0
        for _coarse, threads, vector, _rows, _schedule in _SEED_LATTICE
    ):
        return []
    threads = features // 8
    if threads % 32 or not 32 <= threads <= MAX_THREADS_PER_BLOCK:
        return []
    return [(32, threads, 8, rows, "pipelined") for rows in (4, 2)]


def _occupancy_coarse(rows: int, coarse: int) -> int | None:
    """The power-of-two coarse tile that gives about one CTA per SM, when the
    lattice's ``coarse`` would leave SMs idle.

    The lattice tiles 16 or 32 rows per CTA: on 256 rows that is 8 CTAs for
    148 SMs, and the search has to halve the coarse tile several times from
    any seed to reach the occupancy-filling geometry (4 rows per CTA won the
    layer_norm / rms_norm backward kernels at that size). ``None`` when the
    lattice already fills the machine or the row count is unknown.
    """
    if rows <= 0:
        return None
    units = max(1, num_compute_units())
    if -(-rows // coarse) >= units:
        return None
    per_cta = max(1, -(-rows // units))
    occupancy = 1 << (per_cta - 1).bit_length()
    return occupancy if occupancy < coarse else None


def _plan_blocks(spec: ConfigSpec, plan: _ResidentPlan) -> bool:
    """Whether the searched block sizes are exactly the plan's coarse and row
    blocks, one block ID each."""
    return all(
        len(item.block_ids) == 1
        and item.block_id in (plan.coarse_block, plan.row_block)
        for item in spec.block_sizes
    )


def _rows_subdivide_coarse(spec: ConfigSpec, plan: _ResidentPlan) -> bool:
    """Whether the row loop tiles the coarse tile's own rows
    (``hl.tile(coarse.begin, coarse.end)``, as in the normalization backward
    examples) rather than a dim of its own (``x[tile_s, tile_r, :]`` beside
    ``for tile_r in hl.tile(rows)``)."""
    row = spec.block_sizes.block_id_lookup(plan.row_block)
    return row.bounded_by_block_id == plan.coarse_block


def _coarse_tile(spec: ConfigSpec, plan: _ResidentPlan, coarse: int) -> int:
    """The coarse tile a seed can schedule: ``coarse`` when the row loop
    subdivides the coarse tile (``_rows_subdivide_coarse``), else 1.

    A row loop over a dim of its own runs every row of each coarse index, so
    a coarse tile wider than one index wraps the row loop in a lane loop over
    the coarse dim, which the resident lowering does not schedule (it
    refuses the config). Each CTA then takes one coarse index and all of its
    rows.
    """
    return coarse if _rows_subdivide_coarse(spec, plan) else 1


def _cta_rows(env: CompileEnvironment, plan: _ResidentPlan, coarse: int) -> int | None:
    """The rows one CTA's row loop runs at a coarse tile of ``coarse``: the
    tile's own when the row loop subdivides it, else every row of the CTA's
    one coarse index (``_coarse_tile``); ``None`` when that count is unknown."""
    if _rows_subdivide_coarse(env.config_spec, plan):
        return coarse
    rows = env.block_sizes[plan.row_block].size
    return env.size_hint(rows) if isinstance(rows, (int, torch.SymInt)) else None


class CuteResidentReductionHeuristic(AutotunerHeuristic):
    """A structural superset; the AST matcher proves effects and carries."""

    name = "cute_resident_reduction"
    backend = "cute"
    priority = 32
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @staticmethod
    def _plan(env: CompileEnvironment, device_ir: DeviceIR) -> _ResidentPlan | None:
        from ....device_ir import ForLoopGraphInfo
        from ....device_ir import RootGraphInfo

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
        spec = env.config_spec
        spec.cute_resident_reduction_blocks.add(plan.reduction_block)
        if _plan_blocks(spec, plan) and not _rows_subdivide_coarse(spec, plan):
            # One coarse index per CTA is the only coarse tile the resident
            # lowering schedules in this form (``_coarse_tile``), below the
            # occupancy floor a large grid gets
            # (``ConfigSpec.raise_grid_block_minimums``, which never exceeds
            # the default block size, so the default stays put).  Search it
            # when a resident family seeds that tile; other surfaces' floors
            # stay.  The seeds follow the plan's structural match: the
            # lowering proves each config, and a kernel it refuses searches
            # the tiles below the floor too.  A split of at most one row keeps
            # the floor: the lowering proves none of its seeds' carries.
            coarse = spec.block_sizes.block_id_lookup(plan.coarse_block)
            rows = _cta_rows(env, plan, 1)
            if (
                coarse.occupancy_floor is not None
                and coarse.autotuner_min == coarse.occupancy_floor[1]
                and (rows is None or rows > 1)
                and cls._seeds_resident(env, device_ir)
            ):
                coarse.drop_occupancy_floor()
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def _seeds_resident(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        """Whether a resident family seeds a row loop over a dim of its own:
        the base seeds, which the depth seeds follow (the serial output
        carrier needs a feature width the base seeds tile too, and the
        serial-row family offers none in this form).  No seed is planted
        with autotuner heuristics disabled, nor by a seed generator that
        fails (``compiler_seed_configs`` logs and skips it)."""
        if env.settings.disable_autotuner_heuristics:
            return False
        try:
            return bool(cls.get_seed_configs(env, device_ir))
        except Exception:
            log.debug(
                "%s failed while probing its seeds; the occupancy floor stays",
                cls.name,
                exc_info=True,
            )
            return False

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
        return cls._seed_configs(env, device_ir, with_vector_rows=True)

    @classmethod
    def _seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR, *, with_vector_rows: bool
    ) -> list[Config] | None:
        """The lattice's seeds and their occupancy twins, with the 8-wide
        entries' (``_vector_row_entries``) when ``with_vector_rows``."""
        plan = cls._plan(env, device_ir)
        if plan is None:
            return None
        spec = env.config_spec
        if not _plan_blocks(spec, plan):
            return None
        seeds = []
        vector_rows = _vector_row_entries(plan.features) if with_vector_rows else []
        # Each entry ends with whether its row group shrinks to the rows a
        # CTA's row loop runs (``_cta_rows``): the 8-wide entries' groups do,
        # the lattice's keep their size.
        lattice = [
            (coarse, threads, vector, rows, schedule, False)
            for coarse, threads, vector, rows, schedule in _SEED_LATTICE
            if plan.features % (threads * vector) == 0
        ]
        lattice.extend(
            (coarse, threads, vector, rows, schedule, True)
            for coarse, threads, vector, rows, schedule in vector_rows
        )
        # The same thread/row/schedule choices at an occupancy-filling coarse
        # tile, after the lattice so the existing primaries keep their rank.
        # Through ``env``: callers hold the environment without entering it.
        coarse_size = env.block_sizes[plan.coarse_block].size
        coarse_count = (
            env.size_hint(coarse_size)
            if isinstance(coarse_size, (int, torch.SymInt))
            else 0
        )
        for entries, fit in ((_SEED_LATTICE, False), (vector_rows, True)):
            for coarse, threads, vector, rows, schedule in entries:
                occupancy = _occupancy_coarse(coarse_count, coarse)
                seed_vector = _seed_vector(plan.features, threads, vector)
                if occupancy is None or seed_vector is None:
                    continue
                entry = (occupancy, threads, seed_vector, rows, schedule, fit)
                if entry not in lattice:
                    lattice.append(entry)
        scheduled = []
        for coarse, threads, vector, rows, schedule, fit in lattice:
            cta_rows = _cta_rows(env, plan, coarse) if fit else None
            # The rows past a CTA's would only take masked staging; a split
            # of no rows still takes a group of one.
            group = rows if cta_rows is None else max(1, min(rows, cta_rows))
            entry = (_coarse_tile(spec, plan, coarse), threads, vector, group, schedule)
            if entry not in scheduled:
                scheduled.append(entry)
        for coarse, threads, vector, rows, schedule in scheduled:
            block_sizes = {plan.coarse_block: coarse, plan.row_block: 1}
            if any(
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
        prefill while keeping more producer CTAs available (one coarse index
        per CTA where the rows lie in a dim of their own, ``_coarse_tile``,
        whose row loop runs all of them). The AST matcher
        still proves every access and charges all four shared-memory slots.

        These are appended after every old compiler seed, including host-tail
        overlays and serial-row seeds; they never replace the default.
        """
        plan = cls._plan(env, device_ir)
        if plan is None:
            return []
        spec = env.config_spec
        if not _plan_blocks(spec, plan):
            return []
        blocks = {plan.coarse_block: _coarse_tile(spec, plan, 16), plan.row_block: 1}
        if any(
            not item._fragment(spec).low
            <= blocks[item.block_id]
            <= item._fragment(spec).high
            for item in spec.block_sizes
        ):
            return []
        coarse_index = next(
            index
            for index, item in enumerate(spec.block_sizes)
            if item.block_id == plan.coarse_block
        )
        result = []
        # The lattice's seeds alone: the 8-wide entries' one-row twins (an
        # occupancy tile of one row) are not its one-row pipeline seeds, and
        # keep their prefetch.
        for seed in cls._seed_configs(env, device_ir, with_vector_rows=False) or ():
            if (
                seed.config["cute_reduction_schedule"] != "pipelined"
                or seed.config["cute_reduction_group_rows"] != 1
            ):
                continue
            seed_blocks = cast("list[int]", seed.config["block_sizes"])
            # An occupancy seed tiles fewer rows per CTA than the lattice;
            # deepening its prefetch must not undo that, but a four-deep ring
            # wants at least one row per slot.
            depth_blocks = blocks | {
                plan.coarse_block: min(
                    blocks[plan.coarse_block], max(seed_blocks[coarse_index], 4)
                )
            }
            values = deepcopy(seed.config) | {
                "block_sizes": _seq_config_list(spec.block_sizes, depth_blocks),
                "cute_reduction_pipeline_depth": 4,
                "cute_reduction_sequence": "scalar",
                "cute_min_blocks_per_mp": 0,
            }
            # SIMT has no PID search coordinate here. Omission already means
            # flat; keeping an explicit key would not survive strict transfer.
            values.pop("pid_type")
            variants = [values]
            if depth_blocks[plan.coarse_block] == 4:
                # A four-row occupancy tile fills the four-deep ring as one
                # group: also hold the whole tile resident at once.
                variants.append(values | {"cute_reduction_group_rows": 4})
            for variant in variants:
                if spec.cute_host_paired_sum_available:
                    result.extend(
                        Config.from_dict(
                            deepcopy(variant) | {"cute_host_paired_sum": mode}
                        )
                        for mode in ("off", "mapped", "narrow")
                    )
                else:
                    result.append(Config.from_dict(variant))
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
        if not _plan_blocks(spec, plan) or not _rows_subdivide_coarse(spec, plan):
            # A row loop over a dim of its own does not lower this family
            # resident: its seed renders the scalar code, at four coarse
            # indices per CTA as at one (``_coarse_tile``).
            return []
        axes = {plan.coarse_block, plan.row_block, plan.reduction_block}
        blocks = dict.fromkeys((plan.coarse_block, plan.row_block), 4)
        threads = {
            plan.coarse_block: 1,
            plan.row_block: 1,
            plan.reduction_block: plan.features,
        }
        if (
            set(spec.num_threads.valid_block_ids()) != axes
            or not axes.issubset(spec.cute_vector_widths.valid_block_ids())
            or not axes.issubset(spec.cute_lane_layouts.valid_block_ids())
            or any(
                not item._fragment(spec).low
                <= blocks[item.block_id]
                <= item._fragment(spec).high
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
            "block_sizes": _seq_config_list(spec.block_sizes, blocks),
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
        if not _plan_blocks(spec, plan):
            return None
        blocks = {plan.coarse_block: _coarse_tile(spec, plan, 64), plan.row_block: 1}
        axes = {plan.coarse_block, plan.row_block, plan.reduction_block}
        if (
            plan.features % (256 * 2)
            or any(
                not item._fragment(spec).low
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
