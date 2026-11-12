"""The CuTe backend's hooks into one :class:`ConfigGeneration`.

``CuteSearchSurface.config_generation_extension`` hands one
:class:`CuteConfigGenerationExtension` to every CuTe ``ConfigGeneration``
(``generation.extension``). It owns these CuTe-specific parts of config
generation:

* the conditional neighbor prerequisites of paired PDL policies;
* the SIMT launch-thread repairs (``repair_flat_config``): ``num_threads``
  slots follow their block sizes, the collective warp MMA's ownership rule,
  flattened loop groups and simultaneous launch axes stay under 1024 threads,
  the packet-prefetch and reduction-ring coordinates stay consistent with
  their gates;
* the flash structural coverage design (``design_flash_coverage``) and every
  ``flash_*`` accessor the flash searches read from it, plus the structurally
  designed initial population (``initial_population_flats``).

The launch-thread repairs and flash coverage design were extracted from
``ConfigGeneration``. The generic population hooks
(``population_floor``, ``deterministic_population_configs``, ...) are the
``flash_*`` methods under the names of the backend-neutral
:class:`ConfigGenerationExtension` contract.  The extension also bounds the
elements per thread of random and mutated configs (``shrink_for_threads``),
which ``ConfigGeneration`` reads off ``num_warps`` for the other backends.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

from ...._utils import next_power_of_2
from ....autotuner.config_fragment import EnumFragment
from ....autotuner.config_spec import fit_thread_count
from ....autotuner.search_surface import ConfigGenerationExtension
from ....exc import AutotuneError
from ....exc import InvalidConfig
from ....runtime.config_cache import ConfigSet
from ...per_thread_planning import cap_auto_thread_axes
from ...per_thread_planning import demote_auto_thread_axes
from ...per_thread_planning import reductions_keep_threads
from ...per_thread_planning import shrink_auto_thread_counts
from ..cute_flash import FLASH_DERIVED_CONFIG_KEYS
from ..cute_flash import FLASH_EXP2_PACKET_KEY
from ..cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
from ..cute_flash import FLASH_PERSISTENT_KEY
from ..cute_flash import FLASH_PIPELINE_FAMILY_KEY
from ..cute_flash import FLASH_ROW_SUM_SCHEDULE_KEY
from ..cute_flash import FLASH_SOFTMAX_DISC_KEY
from ..cute_flash import FLASH_SOFTMAX_LOWERING_KEY
from ..cute_flash import _flash_resident_softmax_overrides
from ..cute_flash import flash_exp2_packet_is_compound
from ..cute_mma_dag import MMA_DAG_KEY
from ..cute_mma_dag import MMA_DAG_OFF
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from ..cute_mma_dag import MMA_DAG_TCGEN05
from ..flash_knobs import FLASH_COVERAGE_CHILD_PARENTS
from ..pdl_release import CUTE_PDL_CAPTURE_RELEASE_CHOICES
from ..pdl_release import CUTE_PDL_RELEASE_DEFAULT
from ..pdl_release import CUTE_PDL_SPREAD_KEY
from ..thread_budget import MAX_THREADS_PER_BLOCK
from ..thread_budget import tile_level_axes
from ..tile_packing import packed_grid_block

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import MutableSet
    from collections.abc import Sequence

    from ....autotuner.config_fragment import ConfigSpecFragment
    from ....autotuner.config_generation import ConfigGeneration
    from ....autotuner.config_generation import FlatConfig
    from ....runtime.config import Config
    from ...backend import Backend
    from ..cute_flash import FlashStructuralLeaf
    from ..loop_nesting import TileLoopPath
    from ..thread_budget import TileAxis
    from .flash_search import FlashCoverageDesign


def cute_generation_extension(
    generation: ConfigGeneration,
) -> CuteConfigGenerationExtension:
    """``generation.extension`` under the CuTe surface (typed for the flash
    searches, which run only there)."""
    return cast("CuteConfigGenerationExtension", generation.extension)


@dataclasses.dataclass(frozen=True)
class _ThreadSlots:
    """The flat slots the SIMT launch-thread repairs read and write."""

    num_threads_indices: list[int]
    num_thread_block_pairs: list[tuple[int, int]]
    block_index_by_id: dict[int, int]
    num_thread_index_by_id: dict[int, int]
    loop_order_indices: list[int]
    flatten_loop_groups: list[tuple[int, list[int]]]
    collective_mma_index: int | None
    register_chain_index: int | None
    collective_compute_index: int | None
    # (m block, m threads, n threads, k threads, n block) slots per contraction
    collective_thread_layouts: list[tuple[int, int, int, int, int]]
    collective_vector_width_indices: list[int]


_NO_THREAD_SLOTS = _ThreadSlots([], [], {}, {}, [], [], None, None, None, [], [])


class _Reduction(NamedTuple):
    """One reduction on the path of the tile the shrink bounds."""

    extent: int
    # Its ``num_threads`` entry; ``0`` is auto.
    configured: int
    # The threads its strategy asks for before it is fitted beside the tiles:
    # a thread per element of its extent (of its loop chunk when rolled) in a
    # power of two, or the configured count when that is smaller.
    asks: int


def _reduction_axes(reductions: Sequence[_Reduction], tile_threads: int) -> int:
    """The launch axes ``reductions`` take beside ``tile_threads`` threads of
    tiles: one for each that keeps several threads when
    ``CuteBackend.adjust_reduction_thread_count`` fits it, in block order, into
    what the tiles and the reductions before it leave.  A full slice no
    reduction reduces may index another reduction's axis instead
    (``plan_axis_partner``), and reductions on the two sides of a branch on
    the program id share one; each is counted all the same, which leaves the
    tile fewer axes than it gets (whether a slice shares an axis depends on
    the values that carry it, which only code generation sees)."""
    axes = 0
    others = tile_threads
    for reduction in reductions:
        threads = reduction.asks
        while threads > 1 and others * threads > MAX_THREADS_PER_BLOCK:
            threads //= 2
        if threads > 1:
            axes += 1
            others *= threads
    return axes


class _PathLevel(NamedTuple):
    """One level of a loop path as ``_lay_out_path`` reads it."""

    # Its axes in loop order (``CuteConfigGenerationExtension._level_axes``)
    # and its ``hl.tile``'s block ids.
    axes: Sequence[TileAxis]
    level: Sequence[int]
    # Whether the planner flattens it (``flatten_loops`` set, or a single
    # axis, in a loop it may flatten: ``tile_loop_keeps_nd_layout``): all of
    # its axes run on one launch axis.
    flattened: bool


class _Layout(NamedTuple):
    """What ``_plan_path`` and ``_lay_out_path`` read of a loop path besides
    its levels (``CuteConfigGenerationExtension._path_layout``)."""

    packed: tuple[int, int] | None
    reductions: Sequence[_Reduction]
    budget: int
    axis_limit: int
    extent_limits: Sequence[int]


class _LevelLaunch(NamedTuple):
    """One level of a loop path on the launch (``_lay_out_path``)."""

    # Its axes in loop order, the threads each walks the level's elements on
    # and the most threads the launch axis each takes allows (0 for an axis
    # that takes none).  The axes of a flattened tile share one launch axis
    # and its limit.
    axes: Sequence[TileAxis]
    threads: list[int]
    limits: list[int]
    # The launch axis each takes, or None.
    positions: list[int | None]


def _plan_tile(
    axes: Sequence[TileAxis],
    level: Sequence[int],
    *,
    device_loop: bool,
    grid_axes: int,
    cap_start: int,
    packed: tuple[int, int] | None,
    reductions: Sequence[_Reduction],
    budget: int,
    axis_limit: int,
    extent_limits: Sequence[int],
) -> tuple[list[int], list[bool]]:
    """The threads the per-thread planner gives each of an N-D tile's
    ``axes`` (in loop order; ``level`` is its ``hl.tile``'s block ids), and
    whether each takes a launch axis, by the planner's rules
    (``plan_per_thread_loop_strategy`` calls the same functions):

    * a ``device_loop`` tile runs its auto axes on one thread when the
      path's first three axes ask for more threads than a block (``budget``,
      the planner's ``tile_loop_thread_count``);
    * the tile gets the launch's thread axes but, below the root grid tile,
      those the grid's tiles take (``grid_axes``; not those of enclosing
      device loops) and, when the path has a reduction (``reductions``), the
      one reserved for it whatever its thread count; while more axes ask for
      threads, the auto ones latest in the ``hl.tile`` run on one thread
      (``demote_auto_thread_axes``);
    * the auto counts shrink until the tile fits in what the packed tiles
      leave of a block (``shrink_auto_thread_counts``; the repair fits the
      tunable axes, but not a fixed-size one);
    * an auto count is capped at what its launch axis takes
      (``cap_auto_thread_axes``, 64 threads on CUDA's z), the tile's axes
      counted from ``cap_start``: the planner's count of the axes below it.

    A grid axis ``packed`` several tiles wide (``(block_id,
    tiles_per_block)``, ``cute_tiles_per_block``) takes a launch axis
    whatever its count, one tile per thread sub-group: its threads widen the
    block but walk no more of a tile, so it plans one thread.
    """
    pack_id, pack = packed or (None, 1)
    # The planner reads a tile in block order and lays it out in loop order.
    by_block = sorted(range(len(axes)), key=lambda i: level.index(axes[i].block_id))
    block_ids = [axes[i].block_id for i in by_block]
    sizes = [axes[i].size for i in by_block]
    threads = [axes[i].threads for i in by_block]
    loop_order = sorted(range(len(axes)), key=by_block.__getitem__)
    if device_loop and budget > MAX_THREADS_PER_BLOCK:
        threads = [count or 1 for count in threads]

    def takes_axis(i: int, count: int) -> bool:
        return block_ids[i] == pack_id or (count or sizes[i]) > 1

    demote_auto_thread_axes(
        threads,
        max(0, axis_limit - int(bool(reductions)) - grid_axes),
        takes_axis,
        lambda i: True,
    )
    auto = {i for i, count in enumerate(threads) if count == 0}
    shrink_auto_thread_counts(sizes, threads, MAX_THREADS_PER_BLOCK // pack)
    cap_auto_thread_axes(
        threads,
        sizes,
        loop_order,
        cap_start,
        extent_limits,
        takes_axis,
        lambda i: i in auto and block_ids[i] != pack_id,
    )
    planned = [0] * len(axes)
    takes = [False] * len(axes)
    for i, position in enumerate(by_block):
        planned[position] = 1 if block_ids[i] == pack_id else threads[i] or sizes[i]
        takes[position] = takes_axis(i, threads[i])
    return planned, takes


def _plan_path(
    levels: Sequence[_PathLevel],
    *,
    packed: tuple[int, int] | None,
    reductions: Sequence[_Reduction],
    budget: int,
    axis_limit: int,
    extent_limits: Sequence[int],
) -> list[tuple[list[int], list[bool]]]:
    """The threads each axis of the levels of a loop path (outermost first)
    plans, and whether it takes a launch axis, as the per-thread planner and
    ``PerThreadNDTileStrategy`` plan them.

    Each N-D level plans its threads by the planner's rules
    (``_plan_tile``), its auto axes capped from the axes the planner counts
    below it: those its root grid tile and the device loops enclosing it ask
    for (each loop's at most the axes the root leaves it), and the
    reduction's when a reduction surely keeps several threads beside the
    ``budget`` (``reductions_keep_threads``).  A flattened level, only ever a
    root's, runs its axes on one launch axis.
    """
    pack_id, pack = packed or (None, 1)

    def drawn_axes(axes: Sequence[TileAxis]) -> int:
        # The planner's ``grid_thread_axes``: from the drawn counts.
        return sum(axis.extent > 1 or axis.block_id == pack_id for axis in axes)

    surely = reductions_keep_threads(
        [(reduction.extent, reduction.configured) for reduction in reductions],
        budget * pack,
    )
    grid_axes = drawn_axes(levels[0].axes) if levels else 0
    available = max(0, axis_limit - int(bool(reductions)) - grid_axes)
    plans: list[tuple[list[int], list[bool]]] = []
    for depth, path_level in enumerate(levels):
        axes = path_level.axes
        device_loop = depth > 0
        if path_level.flattened:
            # Only a root's level: the planner keeps every loop of a kernel
            # with a device loop N-D (``tile_loop_keeps_nd_layout``).
            threads = [
                1 if axis.block_id == pack_id else axis.threads or axis.size
                for axis in axes
            ]
            whole = math.prod(threads) > 1 or any(
                axis.block_id == pack_id for axis in axes
            )
            plans.append((threads, [whole] * len(axes)))
            continue
        cap_start = int(surely) + sum(
            drawn_axes(enclosing.axes)
            if depth_above == 0
            else min(available, drawn_axes(enclosing.axes))
            for depth_above, enclosing in enumerate(levels[:depth])
        )
        plans.append(
            _plan_tile(
                axes,
                path_level.level,
                device_loop=device_loop,
                grid_axes=grid_axes if device_loop else 0,
                cap_start=cap_start,
                packed=packed,
                reductions=reductions,
                budget=budget,
                axis_limit=axis_limit,
                extent_limits=extent_limits,
            )
        )
    return plans


def _launch_axis_threads(
    levels: Sequence[_PathLevel], plans: Sequence[tuple[list[int], list[bool]]]
) -> list[int]:
    """The threads of each launch axis the tiles of a path take, in order
    (``_plan_path``'s ``plans``): an N-D level's axes that take one, a
    flattened level's axes together."""
    threads_by_axis: list[int] = []
    for path_level, (threads, takes) in zip(levels, plans, strict=True):
        if path_level.flattened:
            if takes and takes[0]:
                threads_by_axis.append(math.prod(threads))
        else:
            threads_by_axis.extend(
                count for count, taken in zip(threads, takes, strict=True) if taken
            )
    return threads_by_axis


def _lay_out_path(
    levels: Sequence[_PathLevel],
    *,
    packed: tuple[int, int] | None,
    reductions: Sequence[_Reduction],
    budget: int,
    axis_limit: int,
    extent_limits: Sequence[int],
    tile_threads: int | None = None,
) -> list[_LevelLaunch]:
    """The levels of a loop path (outermost first) as the per-thread planner
    and ``PerThreadNDTileStrategy`` lay them out on the launch, each axis
    planned by ``_plan_path``.

    The launch gives its leading axes to the reductions that keep several
    threads beside the tiles (``_reduction_axes``; one left a single thread
    claims none), and then the levels' axes that take one, in loop order,
    level after level, as their strategies take them (``thread_axes_used``);
    the axes past its last (``axis_limit``) run on one thread.  A packed
    grid axis (``packed``) takes a launch axis on one thread per tile.  The
    reductions fit beside ``tile_threads``, the tiles of every path the
    launch runs (``_pack_fits``), or beside this path's own.
    """
    pack = 1 if packed is None else packed[1]
    plans = _plan_path(
        levels,
        packed=packed,
        reductions=reductions,
        budget=budget,
        axis_limit=axis_limit,
        extent_limits=extent_limits,
    )
    if tile_threads is None:
        tile_threads = math.prod(_launch_axis_threads(levels, plans))
    position = _reduction_axes(reductions, tile_threads * pack)
    launches: list[_LevelLaunch] = []
    for path_level, (threads, takes) in zip(levels, plans, strict=True):
        count = len(path_level.axes)
        launched = [1] * count
        limits = [0] * count
        positions: list[int | None] = [None] * count
        if path_level.flattened:
            if takes and takes[0]:
                if position < axis_limit:
                    launched = list(threads)
                    positions = [position] * count
                    if position < len(extent_limits):
                        limits = [extent_limits[position]] * count
                position += 1
        else:
            for i in range(count):
                if takes[i]:
                    if position < axis_limit:
                        launched[i] = threads[i]
                        positions[i] = position
                        if position < len(extent_limits):
                            limits[i] = extent_limits[position]
                    position += 1
        launches.append(_LevelLaunch(path_level.axes, launched, limits, positions))
    return launches


def _axis_limit(backend: Backend) -> int:
    """The launch's thread axes (``Backend.thread_axis_limit``: CuTe's 3; only
    a backend that never demotes a tile's axes has None)."""
    limit = backend.thread_axis_limit()
    assert limit is not None
    return limit


def _thread_budget(
    paths_axes: Sequence[Sequence[Sequence[TileAxis]]], axis_limit: int
) -> int:
    """The planner's thread budget (``tile_loop_thread_count``) of loop
    paths whose levels' axes are ``paths_axes``: each path's first
    ``axis_limit`` axes that ask for threads, sibling paths sharing the
    launch's axes at the widest of each."""
    extents: list[int] = []
    for path_axes in paths_axes:
        asking = [axis.extent for axes in path_axes for axis in axes if axis.extent > 1]
        for position, extent in enumerate(asking[:axis_limit]):
            if position == len(extents):
                extents.append(extent)
            else:
                extents[position] = max(extents[position], extent)
    return math.prod(extents)


class _ShrinkModel(NamedTuple):
    """What ``shrink_for_threads`` reads of the kernel once: its single loop
    path, the depth on it of the level holding the tile, the sizes of the
    path's fixed-size axes, the ``cute_mma_dag`` slot and the launch's number
    of thread axes.  The layout (``_lay_out``) reads the path's reductions
    and the axes' limits itself."""

    path: TileLoopPath
    depth: int
    fixed_sizes: Mapping[int, int]
    mma_dag: tuple[list[int], bool] | None
    axis_limit: int


class _TileCount(NamedTuple):
    """One count of the tile the shrink bounds (``_count_tile``)."""

    # Its axes, in the draw's loop order, the threads each walks it on, the
    # launch axis each takes (or None) and the most threads that axis allows
    # (0: no launch axis); a flattened tile's axes share one.
    axes: list[TileAxis]
    threads: list[int]
    positions: list[int | None]
    limits: list[int]
    elements: int
    # The threads its block launches for one tile.
    launched: int
    # Whether it is a flattened tile with every count auto and more threads
    # than a block, which it runs as a lane loop on a block's threads.
    lane_loop: bool


class CuteConfigGenerationExtension(ConfigGenerationExtension):
    def __init__(self, generation: ConfigGeneration) -> None:
        super().__init__(generation)
        self._flash_coverage_design_cache: FlashCoverageDesign | None = None

    def conditional_neighbor_updates(
        self, base: FlatConfig, requested: FlatConfig
    ) -> Mapping[int, object]:
        """Activate paired PDL policies together with their prerequisites.

        Enabling ordinary PDL, capture overlap, or loader-tail release also
        disables generic spread padding, which the imported attention body
        cannot implement. Capture and loader-tail additionally activate their
        parent PDL policies. Keep explicit configs unchanged.
        """
        generation = self.generation
        spec = generation.config_spec
        if not spec.cute_pdl_capture_overlap_available or spec.has_explicit_pdl:
            return {}
        layout = generation._key_to_flat_indices
        keys = (
            MMA_DAG_KEY,
            MMA_DAG_PASS_WARPS_KEY,
            "cute_pdl",
            "cute_pdl_capture_overlap",
            "cute_pdl_release",
            CUTE_PDL_SPREAD_KEY,
        )
        if any(key not in layout for key in keys):
            return {}
        slots = {key: layout[key][0][0] for key in keys}
        for values in (base, requested):
            if (
                values[slots[MMA_DAG_KEY]] != MMA_DAG_TCGEN05
                or values[slots[MMA_DAG_PASS_WARPS_KEY]]
                != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
            ):
                return {}
        pdl = slots["cute_pdl"]
        capture = slots["cute_pdl_capture_overlap"]
        release = slots["cute_pdl_release"]
        enable_capture = requested[capture] is True and base[capture] is not True
        enable_tail = (
            requested[release] == "loader_tail" and base[release] != "loader_tail"
        )
        enable_pdl = requested[pdl] is True and base[pdl] is not True
        if not (enable_capture or enable_tail or enable_pdl):
            return {}
        updates: dict[int, object] = {pdl: True, slots[CUTE_PDL_SPREAD_KEY]: False}
        if enable_tail:
            updates[capture] = True
        elif (
            enable_capture
            and requested[release] not in CUTE_PDL_CAPTURE_RELEASE_CHOICES
        ):
            updates[release] = CUTE_PDL_RELEASE_DEFAULT
        return updates

    # -- SIMT launch threads ---------------------------------------------------

    @functools.cached_property
    def _threads(self) -> _ThreadSlots:
        """Pair each num_threads flat slot with its block_size slot, and record
        the thread slots the collective MMA's ownership rule fixes.

        The collective warp MMA engages only when the contraction's N tile
        owns exactly ``bn`` threads, its M tile ``threads // bn`` and its K
        tile one, each with a vector width of one; any other layout silently
        lowers the kernel to the scalar SIMT lane loops (tens of times
        slower). Random and mutated configs that enable
        ``cute_collective_mma`` therefore get their thread layout derived from
        the block sizes instead of wasting evaluations there.
        """
        generation = self.generation
        config_spec = generation.config_spec
        layout = generation._key_to_flat_indices
        try:
            block_indices, _ = layout["block_sizes"]
            num_thread_indices, _ = layout["num_threads"]
        except KeyError:
            return _NO_THREAD_SLOTS
        block_index_by_id = {
            spec.block_id: block_indices[i]
            for i, spec in enumerate(config_spec.block_sizes)
            if i < len(block_indices)
        }
        num_thread_index_by_id = {
            spec.block_id: num_thread_indices[i]
            for i, spec in enumerate(config_spec.num_threads)
            if i < len(num_thread_indices)
        }
        num_thread_block_pairs = [
            (num_thread_indices[i], block_index_by_id[spec.block_id])
            for i, spec in enumerate(config_spec.num_threads)
            if i < len(num_thread_indices) and spec.block_id in block_index_by_id
        ]
        loop_order_indices, _ = layout.get("loop_orders", ([], True))
        flatten_loop_groups: list[tuple[int, list[int]]] = []
        if "flatten_loops" in layout:
            flatten_indices, _ = layout["flatten_loops"]
            flatten_loop_groups = [
                (
                    flatten_indices[i],
                    [
                        block_id
                        for block_id in spec.block_ids
                        if block_id in block_index_by_id
                        and block_id in num_thread_index_by_id
                    ],
                )
                for i, spec in enumerate(config_spec.flatten_loops)
                if i < len(flatten_indices)
            ]

        def scalar_index(key: str) -> int | None:
            indices, _is_sequence = layout.get(key, ([], False))
            return indices[0] if len(indices) == 1 else None

        collective_mma_index = scalar_index("cute_collective_mma")
        register_chain_index: int | None = None
        collective_compute_index: int | None = None
        collective_thread_layouts: list[tuple[int, int, int, int, int]] = []
        collective_vector_width_indices: list[int] = []
        if collective_mma_index is not None:
            register_chain_index = scalar_index("cute_register_chain")
            collective_compute_index = scalar_index("cute_collective_compute")
            width_indices, _is_sequence = layout.get("cute_vector_widths", ([], True))
            width_index_by_id = {
                spec.block_id: width_indices[i]
                for i, spec in enumerate(config_spec.cute_vector_widths)
                if i < len(width_indices)
            }
            for fact in config_spec.matmul_facts:
                block_ids = (fact.m_block_id, fact.n_block_id, fact.k_block_id)
                if any(
                    block_id is None
                    or block_id not in block_index_by_id
                    or block_id not in num_thread_index_by_id
                    for block_id in block_ids
                ):
                    continue
                m_block_id, n_block_id, k_block_id = cast(
                    "tuple[int, int, int]", block_ids
                )
                collective_thread_layouts.append(
                    (
                        block_index_by_id[m_block_id],
                        num_thread_index_by_id[m_block_id],
                        num_thread_index_by_id[n_block_id],
                        num_thread_index_by_id[k_block_id],
                        block_index_by_id[n_block_id],
                    )
                )
                collective_vector_width_indices.extend(
                    width_index_by_id[block_id]
                    for block_id in block_ids
                    if block_id in width_index_by_id
                )
        return _ThreadSlots(
            num_threads_indices=num_thread_indices,
            num_thread_block_pairs=num_thread_block_pairs,
            block_index_by_id=block_index_by_id,
            num_thread_index_by_id=num_thread_index_by_id,
            loop_order_indices=loop_order_indices,
            flatten_loop_groups=flatten_loop_groups,
            collective_mma_index=collective_mma_index,
            register_chain_index=register_chain_index,
            collective_compute_index=collective_compute_index,
            collective_thread_layouts=collective_thread_layouts,
            collective_vector_width_indices=collective_vector_width_indices,
        )

    @property
    def num_threads_indices(self) -> list[int]:
        """The flat slots of ``num_threads`` (empty without block sizes)."""
        return self._threads.num_threads_indices

    def _level_axes(
        self,
        flat_config: FlatConfig,
        level: Sequence[int],
        seen: set[int],
        fixed_sizes: Mapping[int, int],
        packed_id: int | None = None,
    ) -> list[TileAxis]:
        """The axes of the path level ``level`` under ``flat_config``
        (``tile_level_axes``), in its tuned loop order: each tunable block with
        the values of its ``block_sizes`` and ``num_threads`` slots, and each
        block in ``fixed_sizes`` with that size and auto threads (it has no
        ``num_threads`` slot).  Reductions and boundary-only grid blocks take
        no part, but a boundary-only grid block ``cute_tiles_per_block`` packs
        (``packed_id``) takes a launch axis all the same, one tile per thread
        sub-group (the planner's ``uses_thread_axis_for``): it stays, as an
        axis of one element on one thread."""
        spec = self.generation.config_spec
        slots = self._threads
        inactive = spec.cute_inactive_tile_block_ids | spec.reduction_block_ids
        # The slot of the order codegen gives the tile: [row, col] in one root
        # and [k, row] in another both take [k, row]'s, a 1-D tile none.
        order_index = spec.loop_orders.config_get_for_tile(
            slots.loop_order_indices, level
        )
        order = (
            cast("list[int]", flat_config[order_index])
            if order_index is not None
            else range(len(level))
        )

        def size_and_threads(block_id: int) -> tuple[int, int] | None:
            if block_id in inactive:
                return (1, 1) if block_id == packed_id else None
            block_index = slots.block_index_by_id.get(block_id)
            thread_index = slots.num_thread_index_by_id.get(block_id)
            if block_index is None or thread_index is None:
                size = fixed_sizes.get(block_id)
                return None if size is None else (size, 0)
            size = flat_config[block_index]
            threads = flat_config[thread_index]
            if type(size) is not int or type(threads) is not int:
                return None
            return size, threads

        return tile_level_axes(level, order, size_and_threads, seen)

    def _path_axes(
        self,
        flat_config: FlatConfig,
        path: Sequence[Sequence[int]],
        fixed_sizes: Mapping[int, int],
        packed_id: int | None = None,
    ) -> list[list[TileAxis]]:
        """The axes of each level of ``path`` (``_level_axes``); a block
        repeated on the path counts at its outermost level."""
        seen: set[int] = set()
        return [
            self._level_axes(flat_config, level, seen, fixed_sizes, packed_id)
            for level in path
        ]

    def shrink_for_threads(
        self, flat_config: FlatConfig, max_elements_per_thread: int
    ) -> None:
        """Halve a tile's block sizes until each thread walks at most
        ``max_elements_per_thread`` of its elements: the tile's elements over
        the threads it runs on with its loop path laid out on the launch
        (``_count_tile``, ``_lay_out_path``), counted anew after each halving
        on the config ``unflatten`` makes of the draw.  The repair and
        normalization in between can take threads from the launch axes: the
        repair fits the threads of all the tile's axes into one block, a
        lane-loop axis's too (the per-thread planner counts them), and
        normalization halves a tile that leaves a persistent reduction fewer
        than two threads.  An axis asks for its ``num_threads`` (``0``, auto,
        a thread per element); a fixed-size axis has no slot and is auto.
        Once halving stops, a launch axis whose threads pass what it takes
        (64 on CUDA's z), which code generation refuses, gets its widest
        count cut to fit (``_cut_to_axis_limits``), and the tile is counted
        again.  The path's reductions take no threads from the tile: the
        backend fits them into what the tile leaves of the block, and each
        that keeps several threads takes a launch axis (``_reduction_axes``:
        full slices that share one count one each, so such a tile can shrink
        more than it needs to).

        Only a single tile is bounded (``_shrink_model``): one level of the
        kernel's single loop path holds every tunable block size, as in a
        pointwise or row-reduction kernel (whose reduction chunks keep their
        own limits), and no fixed size on the path is computed from the
        config.  That level's fixed-size axes count with their threads and
        lane loops but never shrink; of the path's other levels (an enclosing
        ``hl.grid``, a fixed-size loop inside) only the launch axes and the
        thread budget they take count.  Block sizes spread over several loops
        (sibling roots, barrier phases, nested tile loops) keep only the
        numel constraints, like their default configs: those of sibling
        roots, of phases or of a matmul's K loop multiply to no tile a thread
        walks.  An MMA DAG config (a tile whose matmul
        ``cute_mma_dag`` lowers) keeps its block sizes too: it launches
        ``cute_mma_dag_warps`` warps and normalizes ``num_threads`` to auto,
        so ``num_threads`` does not count its threads.  The SIMT fallback of a
        matmul is counted by its ``num_threads`` alone: what the planner does
        to a matmul loop's auto counts (its matmul thread limit, the
        contraction's reserve, a tensor-core root plan) is not counted, so
        such a tile can still run more elements per thread than drawn.
        """
        model = self._shrink_model
        if model is None:
            return
        generation = self.generation
        slots = self._threads
        minimums = {
            index: generation.block_size_minimum(index)
            for index in slots.block_index_by_id.values()
        }
        # Canonicalization writes an overridden value back, so halving or
        # cutting one would change nothing the count sees.
        overridden = generation.overridden_flat_indices
        while True:
            count = self._count_tile(model, flat_config)
            if count is None:
                return
            if count.elements <= max_elements_per_thread * count.launched:
                # Bounded: fit each launch axis's threads, and count again on
                # the threads a cut leaves.  Fitting them only once halving
                # stops keeps the halvings of a config that fits: a smaller
                # tile leaves the reductions more threads, which can push
                # the tile onto an axis that takes fewer.
                if self._cut_to_axis_limits(count, flat_config, overridden):
                    continue
                return
            # Past a block, the threads the draw asks for get cut on whichever
            # axis, so every axis halves; within one, halving an axis with a
            # thread per element would only take its threads away.
            requested = math.prod(
                min(axis.extent, axis.size)
                for axis in self._path_axes(flat_config, model.path, model.fixed_sizes)[
                    model.depth
                ]
            )
            shrink = [
                index
                for axis, axis_threads in zip(count.axes, count.threads, strict=True)
                if (index := slots.block_index_by_id.get(axis.block_id)) is not None
                and index not in overridden
                and (requested > MAX_THREADS_PER_BLOCK or axis.size > axis_threads)
                and cast("int", flat_config[index]) // 2 >= minimums[index]
            ]
            if not shrink:
                if self._cut_to_axis_limits(count, flat_config, overridden):
                    continue
                return
            for index in shrink:
                flat_config[index] = cast("int", flat_config[index]) // 2

    def _cut_to_axis_limits(
        self, count: _TileCount, flat_config: FlatConfig, overridden: set[int]
    ) -> bool:
        """Cut the widest count of each launch axis whose threads in
        ``count`` pass what it takes (64 on CUDA's z) so that they fit,
        making an auto count explicit; whether any was cut.  An overridden
        count stays, and a cut only ever lowers an explicit count, so the
        shrink's loop ends even where the repair hands a count back.  A
        flattened tile whose counts are all auto is left alone past a block:
        there its threads run as a lane loop on a block's threads, and an
        explicit count would make the planner multiply its fixed-size axes'
        threads in.  Within a block, a count made explicit runs as many.

        Code generation refuses such an axis.  The planner caps only auto
        counts, at the limit of the axis it expects each on, and counts the
        axes below a tile low (one for any reductions), so an explicit count
        (the draw's, or one the repair makes explicit), the auto count of a
        tile that two reductions' axes push onto z and a flattened tile's
        threads there can all pass it.  A fixed-size axis has no count to
        cut."""
        if count.lane_loop:
            return False
        slots = self._threads
        cut = False
        for position in dict.fromkeys(count.positions):
            if position is None:
                continue
            on_axis = [i for i, p in enumerate(count.positions) if p == position]
            limit = count.limits[on_axis[0]]
            threads = math.prod(count.threads[i] for i in on_axis)
            if not 0 < limit < threads:
                continue
            cuttable = [
                i
                for i in on_axis
                if count.threads[i] > 1
                and count.axes[i].block_id in slots.num_thread_index_by_id
                and slots.num_thread_index_by_id[count.axes[i].block_id]
                not in overridden
            ]
            if cuttable:
                widest = max(cuttable, key=count.threads.__getitem__)
                index = slots.num_thread_index_by_id[count.axes[widest].block_id]
                value = max(1, count.threads[widest] * limit // threads)
                current = cast("int", flat_config[index])
                if current == 0 or value < current:
                    flat_config[index] = value
                    cut = True
        return cut

    @functools.cached_property
    def _shrink_model(self) -> _ShrinkModel | None:
        """What the shrink reads of the kernel once, or None when it bounds
        no tile (see ``shrink_for_threads``)."""
        generation = self.generation
        config_spec = generation.config_spec
        slots = self._threads
        tile = set(slots.block_index_by_id)
        paths = config_spec.cute_tile_loop_paths
        levels = (
            [level for level in paths[0] if tile & set(level)]
            if len(paths) == 1
            else []
        )
        if (
            len(levels) != 1
            or not tile <= set(levels[0])
            or len(slots.num_thread_block_pairs) != len(tile)
            # A fixed size the config computes (``block_size=
            # next_power_of_2(rows)``), on the tile's level or an enclosing
            # loop's, is no static count of elements or threads.
            or config_spec.cute_tile_symbolic_fixed_blocks
        ):
            return None
        (path,) = paths
        (level,) = levels
        return _ShrinkModel(
            path=path,
            depth=path.index(level),
            fixed_sizes=config_spec.cute_tile_fixed_block_sizes,
            mma_dag=generation._key_to_flat_indices.get(MMA_DAG_KEY),
            axis_limit=_axis_limit(config_spec.backend),
        )

    @functools.cached_property
    def _path_reduction_slots(
        self,
    ) -> dict[TileLoopPath, list[tuple[int, int, int | None, int]]]:
        """The block id, extent, ``num_threads`` slot and thread limit of each
        reduction of each loop path (``cute_tile_loop_path_reductions``); one
        without a slot (no static extent) is auto."""
        config_spec = self.generation.config_spec
        slots = self._threads
        table: dict[TileLoopPath, list[tuple[int, int, int | None, int]]] = {}
        for path in config_spec.cute_tile_loop_paths:
            reductions: list[tuple[int, int, int | None, int]] = []
            for block_id in sorted(
                config_spec.cute_tile_loop_path_reductions.get(path, ())
            ):
                index = slots.num_thread_index_by_id.get(block_id)
                extent = (
                    2
                    if index is None
                    else config_spec.num_threads.block_id_lookup(block_id).size_hint
                )
                # argmin/argmax reduce within a warp (``plan_max_threads``).
                limit = (
                    32
                    if block_id in config_spec.cute_indexed_reduction_block_ids
                    else MAX_THREADS_PER_BLOCK
                )
                reductions.append((block_id, extent, index, limit))
            table[path] = reductions
        return table

    def _path_reductions(
        self,
        reduction_slots: Sequence[tuple[int, int, int | None, int]],
        counted: FlatConfig,
        config: Config | None,
    ) -> list[_Reduction]:
        """The reductions of ``reduction_slots`` (``_path_reduction_slots``)
        under ``counted``, each asking for a thread per element of its extent
        or, under a ``config`` that rolls it, of its loop chunk."""
        config_spec = self.generation.config_spec
        reductions = []
        for block_id, extent, index, limit in reduction_slots:
            configured = 0 if index is None else cast("int", counted[index])
            chunk = (
                None
                if config is None
                else config_spec.reduction_loops.config_get(
                    config.reduction_loops, block_id, None
                )
            )
            asks = next_power_of_2(min(extent if chunk is None else chunk, limit))
            reductions.append(
                _Reduction(
                    extent,
                    configured,
                    configured if 0 < configured < asks else asks,
                )
            )
        return reductions

    def _flattened(self, counted: FlatConfig, level: Sequence[int]) -> bool:
        """Whether the planner flattens the tile of ``level`` under
        ``counted``: one it may flatten (``tile_loop_keeps_nd_layout``) with a
        single axis or with its ``flatten_loops`` entry set."""
        if tuple(level) in self.generation.config_spec.cute_nd_tile_loops:
            return False
        if len(level) == 1:
            return True
        slots = self._threads
        tunable = {
            block_id for block_id in level if block_id in slots.block_index_by_id
        }
        return any(
            counted[flatten_index] is True and set(block_ids) == tunable
            for flatten_index, block_ids in slots.flatten_loop_groups
        )

    def _path_layout(
        self,
        counted: FlatConfig,
        config: Config | None,
        path: TileLoopPath,
        path_axes: Sequence[Sequence[TileAxis]],
        *,
        packed: tuple[int, int] | None,
        budget: int,
    ) -> tuple[list[_PathLevel], _Layout]:
        """The levels of ``path`` (their axes ``path_axes``) under ``counted``
        and what ``_plan_path`` and ``_lay_out_path`` read besides, with what
        ``config``, the normalized config when there is one, says of its
        reductions' loops and of a reduction laying the tiles out itself."""
        config_spec = self.generation.config_spec
        backend = config_spec.backend
        levels = [
            _PathLevel(axes, level, self._flattened(counted, level))
            for level, axes in zip(path, path_axes, strict=True)
        ]
        return levels, _Layout(
            packed=packed,
            reductions=self._path_reductions(
                self._path_reduction_slots.get(path, []), counted, config
            ),
            budget=budget,
            axis_limit=_axis_limit(backend),
            # A reduction laying the tiles out itself leaves them uncapped, as
            # the planner does.
            extent_limits=()
            if config is not None
            and backend.per_thread_reduction_lays_out_tiles(config_spec, config, budget)
            else backend.thread_axis_extent_limits() or (),
        )

    def _lay_out(
        self,
        counted: FlatConfig,
        config: Config | None,
        path: TileLoopPath,
        path_axes: Sequence[Sequence[TileAxis]],
        *,
        packed: tuple[int, int] | None,
        budget: int,
        tile_threads: int | None = None,
    ) -> list[_LevelLaunch]:
        """The levels of ``path`` (their axes ``path_axes``) on the launch
        under ``counted`` (``_lay_out_path``; ``_path_layout``)."""
        levels, layout = self._path_layout(
            counted, config, path, path_axes, packed=packed, budget=budget
        )
        return _lay_out_path(levels, **layout._asdict(), tile_threads=tile_threads)

    def _launch_axis_threads(
        self,
        counted: FlatConfig,
        path: TileLoopPath,
        path_axes: Sequence[Sequence[TileAxis]],
        *,
        packed: tuple[int, int] | None,
        budget: int,
    ) -> list[int]:
        """The threads of each launch axis the tiles of ``path`` take
        (``_launch_axis_threads``), under ``counted`` and no normalized
        config."""
        levels, layout = self._path_layout(
            counted, None, path, path_axes, packed=packed, budget=budget
        )
        return _launch_axis_threads(levels, _plan_path(levels, **layout._asdict()))

    def _count_tile(
        self, model: _ShrinkModel, flat_config: FlatConfig
    ) -> _TileCount | None:
        """The tile's axes, the threads each walks it on as its path is laid
        out on the launch (``_lay_out_path``), its elements and the threads
        its block launches for it, on the config ``unflatten`` makes of
        ``flat_config``; None for an MMA DAG config, whose threads
        ``num_threads`` does not set."""
        config_spec = self.generation.config_spec
        slots = self._threads
        config, counted = self._normalized(flat_config)
        if model.mma_dag is not None and counted[model.mma_dag[0][0]] != MMA_DAG_OFF:
            return None
        if config is not None and "num_threads" not in config.config:
            # Normalization drops an all-auto ``num_threads``, which flatten
            # then refills with its fragments' defaults; the launch runs every
            # axis auto.
            for index in slots.num_thread_index_by_id.values():
                counted[index] = 0
        packed = None if config is None else packed_grid_block(config_spec, config)
        path_axes = self._path_axes(
            counted,
            model.path,
            model.fixed_sizes,
            None if packed is None else packed[0],
        )
        budget = _thread_budget([path_axes], model.axis_limit)
        launch = self._lay_out(
            counted, config, model.path, path_axes, packed=packed, budget=budget
        )[model.depth]
        axes, threads = list(launch.axes), launch.threads
        level = model.path[model.depth]
        return _TileCount(
            axes=axes,
            threads=threads,
            positions=launch.positions,
            limits=launch.limits,
            elements=math.prod(axis.size for axis in axes),
            # A flattened auto tile past a block (beside a fixed-size axis)
            # walks a lane loop on a block's threads.
            launched=min(math.prod(threads), MAX_THREADS_PER_BLOCK),
            lane_loop=math.prod(threads) > MAX_THREADS_PER_BLOCK
            and self._flattened(counted, level)
            and all(
                counted[slots.num_thread_index_by_id[block_id]] == 0
                for block_id in level
                if block_id in slots.num_thread_index_by_id
            ),
        )

    def _normalized(self, flat_config: FlatConfig) -> tuple[Config | None, FlatConfig]:
        """The config ``unflatten`` makes of ``flat_config`` (repaired and
        normalized) and its flat form, or only the repaired copy when
        normalization rejects it."""
        try:
            counted, config = self.generation.canonicalize_flat(flat_config)
        except InvalidConfig:
            repaired = copy.deepcopy(flat_config)
            self.repair_flat_config(repaired)
            return None, repaired
        return config, counted

    def _repair_packet_prefetch(self, flat_config: FlatConfig) -> None:
        """Keep the packet choice consistent with its complete-tile gate."""
        layout = self.generation._key_to_flat_indices
        prefetch = layout.get("cute_packet_prefetch")
        bounds = layout.get("cute_proven_bounds")
        if prefetch is None or bounds is None:
            return
        if flat_config[bounds[0][0]] is not True:
            flat_config[prefetch[0][0]] = 0

    def _repair_reduction_pipeline_depth(self, flat_config: FlatConfig) -> None:
        """Keep inactive ring coordinates at the existing two-slot default."""
        layout = self.generation._key_to_flat_indices
        depth = layout.get("cute_reduction_pipeline_depth")
        schedule = layout.get("cute_reduction_schedule")
        if depth is None or schedule is None:
            return
        if flat_config[schedule[0][0]] != "pipelined":
            flat_config[depth[0][0]] = 2

    def _repair_collective_thread_layouts(self, flat_config: FlatConfig) -> None:
        """Give an enabled collective MMA the thread layout its lowering admits."""
        slots = self._threads
        mma_index = slots.collective_mma_index
        if (
            mma_index is None
            or not slots.collective_thread_layouts
            or flat_config[mma_index] is not True
        ):
            return
        compute_index = slots.collective_compute_index
        if compute_index is not None and flat_config[compute_index] not in (
            "warp",
            "tcgen05",
        ):
            return
        chain_index = slots.register_chain_index
        register_chain = chain_index is not None and flat_config[chain_index] is True
        threads = 64 if register_chain else 128
        # The tiles the collective lowering admits (``collective_matmul``'s
        # ``m_choices`` / ``n_choices``); outside them the collective declines
        # whatever the thread layout says, so the lane loops keep the layout
        # ``repair_flat_config`` gave them.
        m_choices = (16, 32, 64, 128) if register_chain else (32, 64, 128)
        n_choices = (16, 32, 64) if register_chain else (32, 64)
        for (
            m_block_index,
            m_index,
            n_index,
            k_index,
            n_block_index,
        ) in slots.collective_thread_layouts:
            bm = flat_config[m_block_index]
            bn = flat_config[n_block_index]
            if type(bm) is not int or type(bn) is not int:
                continue
            if bm not in m_choices or bn not in n_choices or threads % bn:
                continue
            m_threads = threads // bn
            if m_threads > bm or bm % m_threads:
                continue
            flat_config[n_index] = bn
            flat_config[m_index] = m_threads
            flat_config[k_index] = 1
        for width_index in slots.collective_vector_width_indices:
            flat_config[width_index] = 1

    def repair_flat_config(self, flat_config: FlatConfig) -> None:
        """Keep CuTe launch-thread choices compatible with tuned block sizes."""
        self._repair_packet_prefetch(flat_config)
        self._repair_reduction_pipeline_depth(flat_config)
        slots = self._threads
        if not slots.num_thread_block_pairs:
            return

        for num_threads_idx, block_size_idx in slots.num_thread_block_pairs:
            num_threads = flat_config[num_threads_idx]
            block_size = flat_config[block_size_idx]
            if (
                type(num_threads) is not int
                or num_threads == 0
                or type(block_size) is not int
                or block_size <= 0
            ):
                continue
            flat_config[num_threads_idx] = fit_thread_count(num_threads, block_size)
        self._repair_collective_thread_layouts(flat_config)

        for flatten_idx, block_ids in slots.flatten_loop_groups:
            if flat_config[flatten_idx] is not True:
                continue
            group: list[tuple[int, int, bool, int]] = []
            for block_id in block_ids:
                block_size = flat_config[slots.block_index_by_id[block_id]]
                num_threads_idx = slots.num_thread_index_by_id[block_id]
                num_threads = flat_config[num_threads_idx]
                if (
                    type(block_size) is not int
                    or block_size <= 0
                    or type(num_threads) is not int
                ):
                    group = []
                    break
                resolved_threads = num_threads if num_threads > 0 else block_size
                group.append(
                    (num_threads_idx, block_size, num_threads == 0, resolved_threads)
                )
            if not group:
                continue
            thread_product = functools.reduce(
                operator.mul, (item[3] for item in group), 1
            )
            auto_positions = [i for i, item in enumerate(group) if item[2]]
            while thread_product > 1024 and auto_positions:
                largest_pos = max(auto_positions, key=lambda i: group[i][3])
                num_threads_idx, block_size, is_auto, resolved_threads = group[
                    largest_pos
                ]
                if resolved_threads <= 1:
                    auto_positions.remove(largest_pos)
                    continue
                next_threads = resolved_threads // 2
                while next_threads > 1 and block_size % next_threads != 0:
                    next_threads //= 2
                if next_threads == resolved_threads:
                    auto_positions.remove(largest_pos)
                    continue
                flat_config[num_threads_idx] = next_threads
                group[largest_pos] = (
                    num_threads_idx,
                    block_size,
                    is_auto,
                    next_threads,
                )
                thread_product = (thread_product // resolved_threads) * next_threads

        config_spec = self.generation.config_spec
        if config_spec.cute_tile_loop_paths and not config_spec.matmul_facts:
            self._repair_tile_loop_threads(flat_config)
            return

        explicit_indices = [
            idx
            for idx, _ in slots.num_thread_block_pairs
            if type(flat_config[idx]) is int and cast("int", flat_config[idx]) > 0
        ]
        for indices in self._coexisting_thread_groups(explicit_indices):
            thread_product = math.prod(cast("int", flat_config[idx]) for idx in indices)
            while thread_product > 1024 and indices:
                largest_idx = max(
                    indices,
                    key=lambda idx: cast("int", flat_config[idx]),
                )
                largest = cast("int", flat_config[largest_idx])
                if largest <= 1:
                    break
                flat_config[largest_idx] = largest // 2
                thread_product //= 2

    def _coexisting_thread_groups(self, indices: list[int]) -> list[list[int]]:
        """Repair simultaneous axes, not the product of independent root grids.

        Root ownership does not predict physical axis placement. The backend's
        final-AST launch check still rejects independent grids whose per-axis
        maximum requires more than 1024 threads. Unowned/nested axes remain in
        every group conservatively; missing topology retains the old bound.
        """
        config_spec = self.generation.config_spec
        topology = config_spec.kernel_grid_fact
        if topology is None or len(topology.roots) < 2:
            return [indices]
        num_thread_index_by_id = self._threads.num_thread_index_by_id
        roots = [set(root.block_ids) for root in topology.roots]
        root_ids = set().union(*roots)
        owners = {
            num_thread_index_by_id[item.block_id]: set(item.block_ids)
            for item in config_spec.num_threads
            if item.block_id in num_thread_index_by_id
        }
        return [
            [
                index
                for index in indices
                if not owners[index].intersection(root_ids)
                or owners[index].intersection(root)
            ]
            for root in roots
        ]

    def _repair_tile_loop_threads(self, flat_config: FlatConfig) -> None:
        """Fit simultaneous SIMT axes without multiplying sequential passes."""
        spec = self.generation.config_spec
        slots = self._threads
        thread_index_by_id = slots.num_thread_index_by_id

        def fixed_sizes(level: Sequence[int]) -> Mapping[int, int]:
            # The fixed-size axes of a flattened tile multiply its threads
            # once one of its counts is explicit (an auto one gets a lane
            # loop past a block); the planner fits an N-D tile's itself.
            if self._flattened(flat_config, level) and any(
                flat_config[thread_index_by_id[block_id]] != 0
                for block_id in level
                if block_id in thread_index_by_id
            ):
                return spec.cute_tile_fixed_block_sizes
            return {}

        def launch_axes() -> list[dict[int, int]]:
            axes: list[dict[int, int]] = []
            for path in spec.cute_tile_loop_paths:
                seen: set[int] = set()
                threaded = [
                    axis
                    for level in path
                    for axis in self._level_axes(
                        flat_config, level, seen, fixed_sizes(level)
                    )
                    if axis.extent > 1
                ]
                for position, axis in enumerate(threaded):
                    if position == len(axes):
                        axes.append({})
                    # A fixed-size axis has no thread count to repair (a
                    # negative key, no flat slot).
                    index = thread_index_by_id.get(axis.block_id, -1 - axis.block_id)
                    axes[position][index] = axis.extent
            return axes

        while True:
            axes = launch_axes()
            extents = [max(axis.values()) for axis in axes]
            if functools.reduce(operator.mul, extents, 1) <= 1024:
                self._fit_tiles_per_block(flat_config)
                return
            # The physical launch uses the maximum of each axis across paths,
            # not the largest path product. Only shrink a current maximum;
            # equally wide siblings may need to shrink together.
            candidates = [
                (index, extent)
                for axis, maximum in zip(axes, extents, strict=True)
                for index, extent in axis.items()
                if extent == maximum and index >= 0
            ]
            if not candidates:
                return
            index, threads = max(candidates, key=operator.itemgetter(1))
            flat_config[index] = self.generation._largest_power_of_two_at_most(
                threads // 2
            )

    def _fit_tiles_per_block(self, flat_config: FlatConfig) -> None:
        """Pack only as many grid tiles into a CTA (``cute_tiles_per_block``)
        as fit in one block beside the threads each tile launches.

        Every packed tile runs on threads of its own, so the pack halves
        until the tiles' threads times the pack fit in one block
        (``_pack_fits``); the tiles keep the layout they were drawn with,
        since packing serves tiles too small to fill a CTA.  A pack that
        packs no grid axis (``tile_packing.packed_grid_block``) is left to
        normalization.  Kernels with matmul facts get no
        ``cute_tiles_per_block`` (``search_surface``), so their thread repair
        has no pack to fit."""
        layout = self.generation._key_to_flat_indices
        if "cute_tiles_per_block" not in layout:
            return
        values: dict[str, object] = {}
        for key in (
            "cute_tiles_per_block",
            "pid_type",
            "cute_cluster_n",
            "l2_groupings",
        ):
            if key in layout:
                indices, is_sequence = layout[key]
                values[key] = (
                    [flat_config[index] for index in indices]
                    if is_sequence
                    else flat_config[indices[0]]
                )
        config_spec = self.generation.config_spec
        packed = packed_grid_block(config_spec, values)
        if packed is None:
            return
        drawn = packed[1]
        while packed is not None and not self._pack_fits(flat_config, packed):
            values["cute_tiles_per_block"] = packed[1] // 2
            packed = packed_grid_block(config_spec, values)
        fitted = 1 if packed is None else packed[1]
        if fitted != drawn:
            ((index,), _) = layout["cute_tiles_per_block"]
            flat_config[index] = fitted

    def _pack_fits(self, flat_config: FlatConfig, packed: tuple[int, int]) -> bool:
        """Whether the grid tiles ``packed`` (``(block_id, tiles_per_block)``)
        packs into a CTA fit in one block: with every loop path laid out on
        the launch as the planner lays it out (``_lay_out_path``), sibling
        paths sharing its axes at the widest of each, the threads of each
        packed tile times the pack fit in a block.  The reductions count by
        their extents: the repair has no normalized config to read their
        loops from.  An explicit count past what its launch axis takes (64 on
        CUDA's z) is the shrink's to cut (``shrink_for_threads``)."""
        config_spec = self.generation.config_spec
        paths_axes = [
            self._path_axes(
                flat_config, path, config_spec.cute_tile_fixed_block_sizes, packed[0]
            )
            for path in config_spec.cute_tile_loop_paths
        ]
        budget = _thread_budget(paths_axes, _axis_limit(config_spec.backend))
        # Codegen fits the reductions beside the launch the sibling paths
        # share, the widest of each of its axes, not beside one path's tiles.
        tile_threads = math.prod(
            max(column)
            for column in itertools.zip_longest(
                *(
                    self._launch_axis_threads(
                        flat_config, path, path_axes, packed=packed, budget=budget
                    )
                    for path, path_axes in zip(
                        config_spec.cute_tile_loop_paths, paths_axes, strict=True
                    )
                ),
                fillvalue=1,
            )
        )
        extents: dict[int, int] = {}
        for path, path_axes in zip(
            config_spec.cute_tile_loop_paths, paths_axes, strict=True
        ):
            for launch in self._lay_out(
                flat_config,
                None,
                path,
                path_axes,
                packed=packed,
                budget=budget,
                tile_threads=tile_threads,
            ):
                level_extents: dict[int, int] = {}
                for threads, position in zip(
                    launch.threads, launch.positions, strict=True
                ):
                    if position is not None:
                        level_extents[position] = (
                            level_extents.get(position, 1) * threads
                        )
                for position, threads in level_extents.items():
                    extents[position] = max(extents.get(position, 1), threads)
        return packed[1] * math.prod(extents.values()) <= MAX_THREADS_PER_BLOCK

    # -- flash structural coverage ---------------------------------------------

    def narrow_search_choices(self) -> None:
        """Keep the sampled softmax-lowering and row-sum choices reachable
        under the fixed overrides (the resident lowerings pin parent knobs)."""
        if not self.generation.config_spec.cute_flash_search_enabled:
            return

        # Explicit contradictory requests are rejected by override normalization.
        # Only sampled modes need a narrower domain shared by random draws and
        # deterministic coverage, so neither can request an impossible lowering.
        fields = self.generation._flat_fields()
        fragment = fields.get(FLASH_SOFTMAX_LOWERING_KEY)
        if not isinstance(fragment, EnumFragment):
            return
        fixed_lowering = self.generation._override_values.get(
            FLASH_SOFTMAX_LOWERING_KEY
        )
        choices = (
            (fixed_lowering,)
            if fixed_lowering is not None
            else tuple(
                choice
                for choice in fragment._active_choices()
                if all(
                    key not in self.generation._override_values
                    or self.generation._override_values[key] == required
                    for key, required in _flash_resident_softmax_overrides(
                        choice
                    ).items()
                )
            )
        )
        if fixed_lowering is None and choices != fragment._active_choices():
            self.generation._field_view = {
                **fields,
                FLASH_SOFTMAX_LOWERING_KEY: dataclasses.replace(
                    fragment,
                    search_choices=choices,
                    coverage_choices=(
                        None
                        if fragment.coverage_choices is None
                        else tuple(c for c in fragment.coverage_choices if c in choices)
                    ),
                ),
            }
        # This schedule exists only inside the explicit stateful lowering.
        # Fixed parent controls can remove that lowering even on a workload
        # with detector support. Keep random draws and coverage in the same
        # resulting legal domain, rather than advertising an unreachable value.
        row_sum = fields.get(FLASH_ROW_SUM_SCHEDULE_KEY)
        if "resident_stateful" not in choices and isinstance(row_sum, EnumFragment):
            self.generation._field_view = {
                **self.generation._flat_fields(),
                FLASH_ROW_SUM_SCHEDULE_KEY: dataclasses.replace(
                    row_sum,
                    search_choices=tuple(
                        c for c in row_sum._active_choices() if c == "post_acquire"
                    ),
                    coverage_choices=(
                        None
                        if row_sum.coverage_choices is None
                        else tuple(
                            c for c in row_sum.coverage_choices if c == "post_acquire"
                        )
                    ),
                ),
            }

    def _flash_coverage_design(self) -> FlashCoverageDesign:
        """This generation's flash structural coverage design, computed once.

        The design pass (``design_flash_coverage``) normalizes thousands of
        candidates; every ``flash_*`` accessor below reads one of its fields.
        """
        if self._flash_coverage_design_cache is None:
            from .flash_search import design_flash_coverage

            self._flash_coverage_design_cache = design_flash_coverage(self.generation)
        return self._flash_coverage_design_cache

    def _flash_deterministic_coverage_flats(self) -> list[FlatConfig]:
        """The design's rows: a normalized covering of the active flash choices."""
        return copy.deepcopy(self._flash_coverage_design().coverage_flats)

    def _flash_coverage_parent_overridden(self, goal: tuple[str, object]) -> bool:
        """Whether ``goal`` is a conditional child whose parent value the
        overrides pin away (``FLASH_COVERAGE_CHILD_PARENTS``): unreachable by
        the design's construction, so uncovered telemetry rather than a
        defect (like a family the overrides make unreachable)."""
        return any(
            key in self.generation._override_values
            and self.generation._override_values[key] != value
            for key, value in FLASH_COVERAGE_CHILD_PARENTS.get(goal, ())
        )

    def flash_structural_coverage_uncovered_values(
        self,
    ) -> list[tuple[str, object]]:
        """Return active flash values for which the design found no witness."""
        return copy.deepcopy(self._flash_coverage_design().uncovered_values)

    def flash_structural_coverage_active_values(self) -> list[tuple[str, object]]:
        """Return the live normalized-fragment values the design must cover."""
        return copy.deepcopy(self._flash_coverage_design().active_values)

    def flash_structural_coverage_underqualified_values(
        self,
    ) -> list[tuple[str, object, int]]:
        """Return family values with fewer than two witnesses."""
        return copy.deepcopy(self._flash_coverage_design().underqualified_values)

    def flash_structural_leaf_catalog(self) -> list[FlashStructuralLeaf]:
        """Return exact ordinary-family and compound-packet schedule leaves."""
        return copy.deepcopy(self._flash_coverage_design().leaf_catalog)

    def flash_pipeline_lane_catalog(
        self,
    ) -> dict[FlashStructuralLeaf, tuple[tuple[str, object], ...]]:
        """Return every reachable nonsingleton KV/S value for each exact leaf."""
        return copy.deepcopy(self._flash_coverage_design().pipeline_lane_catalog)

    def flash_pipeline_lane_witnesses(
        self,
    ) -> dict[tuple[FlashStructuralLeaf, str, object], Config]:
        """Return one deterministic normalized config for every catalog lane."""
        return copy.deepcopy(self._flash_coverage_design().pipeline_lane_witnesses)

    def flash_clc_lane_catalog(
        self,
    ) -> dict[FlashStructuralLeaf, dict[str, tuple[int, ...]]]:
        """Return hierarchical legal, anchor, and refinement CLC divisors."""
        return copy.deepcopy(self._flash_coverage_design().clc_lane_catalog)

    def flash_clc_lane_witnesses(
        self,
    ) -> dict[tuple[FlashStructuralLeaf, int], Config]:
        """Return a normalized witness for each attempted ordinary CLC divisor."""
        return copy.deepcopy(self._flash_coverage_design().clc_lane_witnesses)

    def flash_structural_coverage_active_leaves(self) -> list[FlashStructuralLeaf]:
        """Compatibility spelling for the exact structural leaf catalog."""
        return self.flash_structural_leaf_catalog()

    def flash_structural_coverage_underqualified_leaves(
        self,
    ) -> list[tuple[FlashStructuralLeaf, int]]:
        """Return nonsingleton ordinary leaves with fewer than two prefix rows."""
        return copy.deepcopy(self._flash_coverage_design().underqualified_leaves)

    def flash_structural_singleton_leaves(self) -> list[FlashStructuralLeaf]:
        """Return ordinary leaves with a single reachable normalized config.

        Qualification reserves no second row for them: there is no other
        config of the leaf to rank against, so one witness measures it
        completely.
        """
        return copy.deepcopy(self._flash_coverage_design().singleton_leaves)

    def flash_structural_coverage_uncovered_interactions(
        self,
    ) -> list[tuple[tuple[str, ...], tuple[object, ...]]]:
        """Return reachable declared field interactions lacking a design row."""
        return copy.deepcopy(self._flash_coverage_design().uncovered_interactions)

    def flash_structural_coverage_active_interactions(
        self,
    ) -> list[tuple[tuple[str, ...], tuple[object, ...]]]:
        """Return reachable declared field interactions the design must cover."""
        return copy.deepcopy(self._flash_coverage_design().active_interactions)

    def structural_qualification_prefix_count(self) -> int:
        """Return rows reserved for two-witness ordinary-leaf qualification."""
        return self._flash_coverage_design().qualification_prefix_count

    def flash_structural_parent_coverage_prefix_count(self) -> int:
        """Return leading rows that cover every live family and packet once."""
        return self._flash_coverage_design().parent_coverage_prefix_count

    def flash_structural_starting_path_limit(
        self,
        *,
        minimum: int,
        retained_families: int | None,
        retained_candidates_per_leaf: int,
    ) -> int:
        """Size full-search continuation capacity from the live leaf catalog."""
        if not self.generation.config_spec.cute_flash_search_enabled:
            return minimum
        leaves = self.flash_structural_leaf_catalog()
        ordinary_by_family: dict[str, int] = {}
        compound_count = 0
        for leaf in leaves:
            if leaf.compound_exp2_packet is None:
                ordinary_by_family[leaf.pipeline_family] = (
                    ordinary_by_family.get(leaf.pipeline_family, 0) + 1
                )
            else:
                compound_count += 1
        promoted_count = self._resolve_flash_structural_family_limit(
            retained_families, len(ordinary_by_family)
        )
        promoted_protocol_count = sum(
            sorted(ordinary_by_family.values(), reverse=True)[:promoted_count]
        )
        secondary_count = promoted_count if retained_candidates_per_leaf > 1 else 0
        # One unrestricted path, every ordinary protocol in the widest possible
        # promoted-family set, one secondary per promoted family, every compound
        # leaf. The global lane alternate occupies its family's secondary slot.
        required = 1 + promoted_protocol_count + secondary_count + compound_count
        return max(minimum, required)

    def flash_structural_family_probe_path_limit(
        self,
        retained_families: int | None,
        family_probe_generations: int,
    ) -> int:
        """Return the measured pre-promotion probe capacity for the live catalog."""
        if (
            not self.generation.config_spec.cute_flash_search_enabled
            or family_probe_generations <= 0
        ):
            return 0
        leaves = self.flash_structural_leaf_catalog()
        ordinary_families = {
            leaf.pipeline_family for leaf in leaves if leaf.compound_exp2_packet is None
        }
        if retained_families is None or len(ordinary_families) <= retained_families:
            return 0
        compound_count = sum(leaf.compound_exp2_packet is not None for leaf in leaves)
        # One constrained path per ordinary family and compound leaf, plus one
        # unrestricted path that can discover a different structural basin.
        return len(ordinary_families) + compound_count + 1

    def flash_structural_effective_family_limit(
        self, retained_families: int | None
    ) -> int:
        """Resolve a configured family cap against the live ordinary catalog.

        ``None`` is the full-search policy: every ordinary family gets a
        conditional continuation. Explicit integer caps preserve the bounded
        policy used by custom effort profiles.
        """
        if not self.generation.config_spec.cute_flash_search_enabled:
            return 0
        live_families = {
            leaf.pipeline_family
            for leaf in self.flash_structural_leaf_catalog()
            if leaf.compound_exp2_packet is None
        }
        return self._resolve_flash_structural_family_limit(
            retained_families, len(live_families)
        )

    @staticmethod
    def _resolve_flash_structural_family_limit(
        retained_families: int | None, live_family_count: int
    ) -> int:
        return (
            live_family_count
            if retained_families is None
            else min(retained_families, live_family_count)
        )

    def validate_flash_structural_coverage(self) -> None:
        """Reject an incomplete structural design before flash autotuning starts."""
        if not self.generation.config_spec.cute_flash_search_enabled:
            return

        structural_override_keys = {
            FLASH_PIPELINE_FAMILY_KEY,
            FLASH_PERSISTENT_KEY,
            *FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS,
            *FLASH_DERIVED_CONFIG_KEYS,
        }
        packet_override = self.generation._override_values.get(FLASH_EXP2_PACKET_KEY)
        if self.generation._override_values.keys() & structural_override_keys or (
            packet_override is not None
            and flash_exp2_packet_is_compound(packet_override)
        ):
            # Structural overrides can pin conditional parents and make
            # advertised children unreachable. Arithmetic overrides must not
            # suppress validation of the remaining flash search design.
            return
        design = self._flash_coverage_design()
        # A normalized family can legitimately have only one distinct effective
        # config. Keep two-witness shortfalls as strict-harness telemetry rather
        # than rejecting an otherwise complete ordinary search. A family the
        # overrides make unreachable is telemetry as well, not a design defect.
        unreachable_families = design.override_unreachable_families
        uncovered = [
            goal
            for goal in design.uncovered_values
            if (
                goal[0] != FLASH_PIPELINE_FAMILY_KEY
                or goal[1] not in unreachable_families
            )
            and not self._flash_coverage_parent_overridden(goal)
        ]
        problems: list[str] = []
        if uncovered:
            problems.append(f"uncovered values={uncovered!r}")
        if design.uncovered_interactions:
            problems.append(f"uncovered interactions={design.uncovered_interactions!r}")
        if problems:
            raise AutotuneError(
                "incomplete CuTe flash structural coverage design: "
                + "; ".join(problems)
            )

    def population_floor(self, n: int) -> int:
        """Raise a flash population to one row per structural leaf.

        The parent-coverage prefix measures every ordinary family schedule
        and compound packet once; a smaller population would leave a
        pipeline family unmeasured. Searches apply this to the effort
        profile's population when they are constructed; an explicit
        population request is honored as given. Non-flash surfaces and
        non-positive requests are returned unchanged.
        """
        if n <= 0 or not self.generation.config_spec.cute_flash_search_enabled:
            return n
        return max(n, self.flash_structural_parent_coverage_prefix_count())

    def structural_population_budget(self, population_size: int) -> int:
        """Return the deterministic-row budget for a flash population size."""
        half_population = population_size // 2
        coverage_count = len(self._flash_deterministic_coverage_flats())
        if coverage_count <= population_size:
            return max(half_population, coverage_count)
        qualification_prefix_count = self.structural_qualification_prefix_count()
        if qualification_prefix_count <= half_population:
            return half_population
        return min(
            population_size,
            max(
                half_population,
                self.flash_structural_parent_coverage_prefix_count(),
            ),
        )

    def deterministic_population_configs(self) -> list[Config]:
        """Return the full normalized structural-coverage design.

        Population construction always consumes its family/packet prefix, but
        may reserve later child-value rows for random exploration.
        """
        self.validate_flash_structural_coverage()
        result: list[Config] = []
        seen: MutableSet[Config] = ConfigSet()
        for flat in self._flash_deterministic_coverage_flats():
            try:
                config = self.generation.unflatten(flat)
            except InvalidConfig:
                continue
            if config in seen:
                continue
            seen.add(config)
            result.append(config)
        return result

    def flash_low_confound_schedule_anchor_configs(self) -> list[Config]:
        """Enumerate neutral anchors for every live flash schedule protocol.

        Marginal covering rows deliberately combine many axes, which makes them
        useful for reachability checks but poor optimization starting points.  A
        flash pipeline family, ordinary exp2 packet, and softmax protocol form
        the high-level schedule decision.  Enumerate that live product from the
        fragment default while leaving every other field neutral.  The anchors
        are still benchmarked and ranked; they contain no measured winner or
        sequence-length lookup.
        """
        if not self.generation.config_spec.cute_flash_search_enabled:
            return []

        keys = (
            FLASH_PIPELINE_FAMILY_KEY,
            FLASH_EXP2_PACKET_KEY,
            FLASH_SOFTMAX_DISC_KEY,
        )
        axes: dict[str, tuple[int, tuple[object, ...]]] = {}
        for key in keys:
            layout = self.generation._key_to_flat_indices.get(key)
            if layout is None or layout[1] or len(layout[0]) != 1:
                return []
            index = layout[0][0]
            fragment = self.generation.flat_spec[index]
            if not isinstance(fragment, EnumFragment):
                return []
            choices = (
                (self.generation._override_values[key],)
                if key in self.generation._override_values
                else fragment.choices
                if fragment.search_choices is None
                else fragment.search_choices
            )
            axes[key] = (index, tuple(choices))

        result: list[Config] = []
        seen: MutableSet[Config] = ConfigSet()
        base = self.generation._fragment_default_flat()
        for family, packet, softmax_disc in itertools.product(
            axes[FLASH_PIPELINE_FAMILY_KEY][1],
            axes[FLASH_EXP2_PACKET_KEY][1],
            axes[FLASH_SOFTMAX_DISC_KEY][1],
        ):
            if flash_exp2_packet_is_compound(packet):
                continue
            flat = copy.deepcopy(base)
            flat[axes[FLASH_PIPELINE_FAMILY_KEY][0]] = family
            flat[axes[FLASH_EXP2_PACKET_KEY][0]] = packet
            flat[axes[FLASH_SOFTMAX_DISC_KEY][0]] = softmax_disc
            try:
                config = self.generation.unflatten(flat)
            except InvalidConfig:
                continue
            values = config.config
            if (
                values.get(FLASH_PIPELINE_FAMILY_KEY) != family
                or values.get(FLASH_EXP2_PACKET_KEY) != packet
                or values.get(FLASH_SOFTMAX_DISC_KEY) != softmax_disc
                or config in seen
            ):
                continue
            seen.add(config)
            result.append(config)
        return result

    def exact_effective_search_space_configs(
        self, max_raw_configs: int
    ) -> list[Config] | None:
        """Enumerate a small CuTe-flash space after config normalization.

        The global flat surface can contain aliases that normalize to the same
        conditional schedule.  The space is enumerated family by family: each
        live pipeline family contributes the product of the knobs live for it
        (every other knob pinned as that family's fragments pin it), so the
        knobs of one family do not multiply another family's product.  When
        every family's raw product is bounded, this returns every distinct
        normalized config in stable product order.  A larger or
        non-enumerable space returns ``None`` without partial work.
        """
        if max_raw_configs < 1:
            raise ValueError("max_raw_configs must be positive")
        if not self.generation.config_spec.cute_flash_search_enabled:
            return None

        overridden = self.generation.overridden_flat_indices
        frozen_flat: FlatConfig | None = None
        if overridden:
            # Extract only the frozen slots. Normalizing the complete default
            # point can fail even when the override has legal combinations in
            # the remaining dimensions (for example a high register allocation
            # paired with a smaller non-overridden allocation).
            from ....runtime.config import Config

            frozen_flat = self.generation.flatten(
                Config.from_dict(self.generation._override_values)
            )
        family_index: int | None = None
        families: list[object | None] = [None]
        family_layout = self.generation._key_to_flat_indices.get(
            FLASH_PIPELINE_FAMILY_KEY
        )
        if (
            family_layout is not None
            and not family_layout[1]
            and len(family_layout[0]) == 1
            and family_layout[0][0] not in overridden
        ):
            family_fragment = self.generation.flat_spec[family_layout[0][0]]
            if isinstance(family_fragment, EnumFragment):
                family_values = family_fragment.search_values(max_raw_configs)
                if family_values is not None and len(family_values) > 1:
                    family_index = family_layout[0][0]
                    families = list(family_values)
        index_to_key = {
            indices[0]: key
            for key, (
                indices,
                is_sequence,
            ) in self.generation._key_to_flat_indices.items()
            if not is_sequence and len(indices) == 1
        }

        def family_value_sets(family: object | None) -> list[list[object]] | None:
            fragments_for_family: Mapping[str, ConfigSpecFragment] = (
                self.generation.config_spec.search_surface.family_fragments(
                    self.generation.config_spec, cast("str", family)
                )
                if family is not None
                else {}
            )
            value_sets: list[list[object]] = []
            raw_size = 1
            for index, fragment in enumerate(self.generation.flat_spec):
                if index in overridden:
                    assert frozen_flat is not None
                    value_sets.append([frozen_flat[index]])
                    continue
                if index == family_index:
                    value_sets.append([family])
                    continue
                key = index_to_key.get(index)
                live_fragment = (
                    fragments_for_family.get(key, fragment)
                    if key is not None
                    else fragment
                )
                cardinality = live_fragment.cardinality()
                if (
                    cardinality is None
                    or cardinality < 1
                    or raw_size > max_raw_configs // cardinality
                ):
                    return None
                values = live_fragment.search_values(max_raw_configs)
                if values is None or len(values) != cardinality:
                    return None
                raw_size *= cardinality
                value_sets.append(values)
            return value_sets

        products: list[list[list[object]]] = []
        for family in families:
            value_sets = family_value_sets(family)
            if value_sets is None:
                return None
            products.append(value_sets)

        result: list[Config] = []
        seen: MutableSet[Config] = ConfigSet()
        for value_sets in products:
            for values in itertools.product(*value_sets):
                try:
                    config = self.generation.unflatten(list(values))
                except InvalidConfig:
                    continue
                if config in seen:
                    continue
                seen.add(config)
                result.append(config)
        return result

    def flash_terminal_coordinate_surface_catalog(
        self, *, radius: int = 1
    ) -> dict[str, object]:
        """Return the ordered coordinate surface for every reachable flash leaf."""
        if not self.generation.config_spec.cute_flash_search_enabled:
            raise AutotuneError("terminal coordinate surface requires CuTe flash")

        leaves: list[dict[str, object]] = []
        for leaf in self.flash_structural_leaf_catalog():
            overrides = dict(self.generation._override_values)
            overrides[FLASH_PIPELINE_FAMILY_KEY] = leaf.pipeline_family
            overrides[FLASH_SOFTMAX_DISC_KEY] = leaf.softmax_disc
            if leaf.compound_exp2_packet is not None:
                overrides[FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
            leaf_generation = self.generation.config_spec.create_config_generation(
                overrides=overrides,
                advanced_controls_files=self.generation._advanced_controls_files,
                process_group_name=self.generation.process_group_name,
            )
            surface = leaf_generation.coordinate_surface_catalog(radius=radius)
            leaves.append(
                {
                    "leaf": {
                        "family": leaf.pipeline_family,
                        "compound_packet": leaf.compound_exp2_packet,
                        "softmax_disc": leaf.softmax_disc,
                    },
                    "coordinates": surface["coordinates"],
                }
            )
        return {
            "schema_version": 1,
            "radius": radius,
            "leaves": leaves,
        }

    def flash_owned_coordinate_indices(self, config: Config) -> list[int]:
        """Coordinates fixed by the parent's explicit softmax implementation.

        This applies to one-coordinate proposals only. A multi-coordinate
        request can switch the lowering and change a formerly owned field.
        """
        if not self.generation.config_spec.cute_flash_search_enabled:
            return []

        owned = _flash_resident_softmax_overrides(
            config.config.get(FLASH_SOFTMAX_LOWERING_KEY)
        )
        return [
            index
            for index, (key, _sequence_index) in enumerate(
                self.generation._flat_coordinate_identities()
            )
            if key in owned and index not in self.generation.overridden_flat_indices
        ]

    def initial_population_flats(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config],
        log_func: Callable[[str], None] | None,
    ) -> list[FlatConfig] | None:
        """The structurally designed flash population, or ``None`` off flash."""
        if not self.generation.config_spec.cute_flash_search_enabled:
            return None
        default_flat = self.generation.default_flat()
        self.validate_flash_structural_coverage()
        result: list[FlatConfig] = []
        seen: MutableSet[Config] = ConfigSet()

        def append_valid(config: Config, *, required: bool = False) -> bool:
            if config in seen or (not required and len(result) >= n):
                return False
            seen.add(config)
            result.append(self.generation.flatten(config))
            return True

        def append_if_valid(flat: FlatConfig, *, required: bool = False) -> bool:
            try:
                config = self.generation.unflatten(flat)
            except InvalidConfig:
                return False
            return append_valid(config, required=required)

        # Flash schedules are compound categorical choices. The design first
        # reaches every exact ordinary-family or compound-packet leaf, then adds
        # a second ordinary witness before child coverage. Use half of the
        # population when qualification fits. If the complete compact design fits,
        # include all of it; a smaller population may exceed half only to preserve
        # one-witness parent coverage.
        coverage_flats = self._flash_deterministic_coverage_flats()
        qualification_prefix_count = self.structural_qualification_prefix_count()
        coverage_budget = min(n, self.structural_population_budget(n))
        coverage_added = 0
        reserved_prefix_count = min(qualification_prefix_count, coverage_budget)
        for flat in coverage_flats[:reserved_prefix_count]:
            try:
                config = self.generation.unflatten(flat)
            except InvalidConfig:
                continue
            if append_valid(config, required=True):
                coverage_added += 1

        # Explicit user seeds are retained even when they make the population
        # larger than the nominal target. Compiler seeds are heuristic hints,
        # so admit only those that fit; exact structural coverage is inserted
        # first and cannot be crowded out by a large same-family seed set.
        for _flat, config in self.generation.user_seed_flat_config_pairs(
            user_seed_configs, log_func
        ):
            append_valid(config, required=True)

        for _flat, config in self.generation.seed_flat_config_pairs(log_func):
            append_valid(config)

        if coverage_added < coverage_budget:
            for flat in coverage_flats[reserved_prefix_count:]:
                if coverage_added >= coverage_budget:
                    break
                try:
                    config = self.generation.unflatten(flat)
                except InvalidConfig:
                    continue
                if append_valid(config, required=True):
                    coverage_added += 1

        exact_space = self.exact_effective_search_space_configs(n)
        if exact_space is not None:
            for config in exact_space:
                # Exhaustive rows cannot be crowded out by mandatory seeds.
                append_valid(config, required=True)

        append_if_valid(default_flat)

        # Fill the remainder with random configs. When the backend supplies
        # value priors, half the random fill is drawn from those priors (biased
        # toward the region good configs occupy) and half stays uniform so the
        # search keeps full coverage; with no priors every fill is uniform,
        # leaving the historical behavior unchanged.
        priors_present = bool(self.generation._config_value_priors)
        invalid = 0
        duplicate = 0
        attempts = 0
        max_attempts = max(64, (n - len(result)) * 64)
        while exact_space is None and len(result) < n and attempts < max_attempts:
            j = attempts
            attempts += 1
            if priors_present and j % 2 == 0:
                flat = self.generation.biased_random_flat()
            else:
                flat = self.generation.random_flat()
            try:
                config = self.generation.unflatten(flat)
            except InvalidConfig:
                invalid += 1
                continue
            if config in seen:
                duplicate += 1
                continue
            seen.add(config)
            result.append(self.generation.flatten(config))
        if len(result) < n and log_func is not None:
            if exact_space is not None:
                log_func(
                    "Exhausted the exact CuTe-flash effective search space at "
                    f"{len(exact_space)} normalized configs; padding the "
                    f"{n}-row initial population with duplicates."
                )
            else:
                log_func(
                    "Generated only "
                    f"{len(result)}/{n} valid initial population configs "
                    f"after {attempts} random attempts "
                    f"({invalid} invalid, {duplicate} duplicate); "
                    "padding with duplicate valid configs."
                )
        if len(result) < n:
            if not result:
                raise InvalidConfig(
                    "failed to generate any valid initial population configs"
                )
            pad_from = [copy.deepcopy(flat) for flat in result]
            pad_idx = 0
            while len(result) < n:
                result.append(copy.deepcopy(pad_from[pad_idx % len(pad_from)]))
                pad_idx += 1
        return result
