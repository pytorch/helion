"""Row seeds for the short rows ``CuteReductionTileHeuristic`` leaves to the
search's generic population.

``CuteReductionTileHeuristic`` seeds rows that fill whole warps with one
16-byte packet per thread, on kernels whose grid admits one row per CTA.
Three families fall outside it:

* a row that is no whole number of warps of packets (128 or 192 bf16
  elements: 16 and 24 packets), which gets the narrow chunk seed alone
  (``CuteSubwarpRowHeuristic``);
* a row kernel whose grid is too large for one row per CTA
  (``ConfigSpec.raise_grid_block_minimums`` starts the search at two rows or
  more), which gets no reduction seed at all although the warp rows hold
  several rows per CTA (``CuteFlooredWarpRowHeuristic``; the rows of the
  first family need no such gate);
* the full-slice blocks a kernel sweeps after the reduction (the half rows of
  a rotation after an RMSNorm), whose vector width every row seed leaves at 1,
  so the sweep loads and stores scalars (``slice_packet_entries``, which the
  rows of the first family carry and ``CuteSliceSweepPacketHeuristic`` adds
  to the warp rows).

All three read one row plan (``_row_plan``), which is stricter than that
heuristic's gate: beside its rows it needs one root grid, packets along the
row and a CTA of the seeds' rows and threads alone.

All are seeds only: no default changes, and ``postprocess_seed_configs``
appends them after every other seed (``row_packet_seed_configs``), so the
existing seed prefixes stay as they were.  The three classes are seed
families of that postprocessor, not entries of the ranked registry.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ....utils import compute_slice_size
from ...reduction_lanes import WARP_THREADS
from ...thread_budget import MAX_THREADS_PER_BLOCK
from .cute import CuteReductionTileHeuristic
from .cute import _cute_tile_inner_block_dtype
from .cute import _cute_tile_seed_vec_width_for_dtype
from .cute import _has_one_sized_reduction_loop
from .cute import _memory_accesses
from .cute import _reduction_blocks_of_extent
from .cute import _row_tile_axes

if TYPE_CHECKING:
    from .....autotuner.config_spec import ReductionLoopSpec
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


class _RowPlan(NamedTuple):
    """A reduced row and the row axis: the innermost tile axis of the rows,
    as an index into the block sizes."""

    block_id: int
    extent: int
    row_index: int
    #: Elements of a 16-byte packet of the dtypes loaded along the row.
    vec: int


def _row_is_no_run_of_packets(
    env: CompileEnvironment, device_ir: DeviceIR, block_id: int
) -> bool:
    """Whether some load or store rules out packets along the full-slice
    block ``block_id``, the row.  Two kinds do:

    * one that slices the row at a stride other than 1: along a dim of its
      tensor that is not the stride-1 one, or with a step (``x[tile, ::2]``).
      The packets of a row are contiguous elements.  A column reduction
      (``x[:, tile].mean(0)``) reduces a full slice too, along a strided dim,
      and on a square tensor its extent matches the stride-1 dim's, which is
      all ``_cute_tile_inner_block_dtype`` compares; the rows of a strided
      view (``base[:, ::2]``) or of a broadcast have no stride-1 dim at all;
    * one that slices its tensor from a bound that is a node of the graph:
      a value the kernel computes (``x[tile, n:]`` for
      ``n = row.size(1) // 2``) or, under dynamic shapes, a size it
      specializes (``x[tile, : hl.specialize(x.size(1)) // 2]``).  The
      subscripts are read as the graph spells them, where every other slice
      bound is a size; a node's value is not resolved, so the plan does not
      know which block such a slice walks.

    A stepped slice of another block's extent does not walk the row: the
    interleaved halves ``x[tile, ::2]`` and ``x[tile, 1::2]`` swept beside a
    contiguous row leave the row its packets.

    Every load and store is read (``_memory_accesses``), and a node bound
    declines before the slice's extent is taken, so the answer depends on no
    other gate.
    """
    for access in _memory_accesses(device_ir):
        if access is None:
            # A stack tensor, or indices that are no list: no row of a host
            # tensor.
            continue
        fake, subscripts = access
        for dim, subscript in enumerate(subscripts):
            if not isinstance(subscript, slice):
                continue
            if any(
                isinstance(bound, torch.fx.Node)
                for bound in (subscript.start, subscript.stop, subscript.step)
            ):
                return True
            extent = compute_slice_size(subscript, fake.shape[dim])
            if block_id not in _reduction_blocks_of_extent(env, extent):
                continue
            stride = fake.stride(dim)
            if subscript.step not in (None, 1) or not (
                isinstance(stride, int) and stride == 1
            ):
                return True
    return False


def _full_slices_meet(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Whether the kernel carries two of its full-slice blocks in one value,
    or in the region of one load, store or atomic.

    Two full-slice blocks that meet (``x[tile, :, :]`` holds a channel slice
    beside the row; ``out[tile, :, :] = v[:, :, None]`` and
    ``hl.atomic_add(out, [tile, slice(None), slice(None)], v[:, :, None])``
    span the row and a slice that no value carries together) need the cross
    product of their coordinates, so the lowering gives each a thread axis
    of its own.  The answer is the lowering's
    (``_cute_blocks_share_a_value``, which decides whether a swept slice may
    index the row's thread axis), not a copy of it: it reads the values, the
    subscripts of loads, stores and atomics with their operands, and matches
    a literal dim (``row[:, None, :].expand(-1, 64, -1)``) to a block by its
    extent.
    """
    # The lowering imports the language, which imports the heuristics.
    from ...reduction_emitter import _cute_blocks_share_a_value

    host_function = device_ir.host_function
    assert host_function is not None
    full_slices = [info.block_id for info in env.block_sizes if info.reduction]
    # The lowering's check reads the host function's symbol origins, which
    # codegen has current and the seed hooks do not.
    with host_function:
        return any(
            _cute_blocks_share_a_value(env, first, second)
            for first, second in combinations(full_slices, 2)
        )


def _row_plan(env: CompileEnvironment, device_ir: DeviceIR) -> _RowPlan | None:
    """The kernel's row reduction, when its row loads as 16-byte packets.

    The structure ``CuteReductionTileHeuristic`` seeds (tile axes over the
    rows, one reduction loop that carries a sized reduction, no matmul)
    without its requirement that the innermost tile axis admits one row per
    CTA: the seeds built from the plan put several rows of that axis in a
    CTA, so a grid whose occupancy floor starts it above one row
    (``ConfigSpec.raise_grid_block_minimums``) takes them too.  The outer
    tile axes stay at one block per CTA, which their floors must admit.

    As for that heuristic's seeds, the row must be the one block the kernel
    reduces, and every block size a tile axis of the rows with one extent
    (``_row_tile_axes``): the lowering refuses these layouts beside a second
    reduced block, and the seeds put their rows per CTA on the innermost of
    those axes.  Here the kernel also needs one root grid: the slices swept
    beside the row share its thread axis only there
    (``slice_packet_entries``).

    The row needs a static extent (which gives it thread, vector-width and
    lane-layout slots: ``cute_static_reduction_extents``) and loads and
    stores that walk it at stride 1 (``_row_is_no_run_of_packets``, which
    also declines a kernel that slices a tensor from a value it computes).
    An argmin or argmax row is declined: its loop is at most a warp of
    elements (``reduction_loop_limits``), so the one rolled trip of these
    seeds would be rewritten into another layout.  So is a loop pinned
    persistent (``ReductionLoopSpec.pinned``: its sweeps would replay an
    effectful producer, an impure ``hl.inline_asm_elementwise`` over the
    row): normalization takes the trip out of the seed, and the lowering
    refuses the persistent row that is left.

    The CTA of these seeds is their rows per CTA times the row's threads,
    which two more kinds of kernel do not launch:

    * one with a tile loop whose block size is fixed at something other than
      one block (``hl.tile([b, h], block_size=[128, None])``, a nested
      ``hl.tile(n, block_size=4)``; ``cute_tile_fixed_block_sizes``, or an
      expression in ``cute_tile_symbolic_fixed_blocks``).  A fixed block is
      no block size of the config, so the floors above do not see it, and it
      takes a thread axis of its own: 128 batches of 8 rows of 8 threads
      leave each row one thread, which then walks all 24 packets of a
      192-wide row.  An axis fixed at one block (and an ``hl.grid`` loop,
      which steps by one) adds no threads;
    * one that carries two full-slice blocks together, the row and a second
      slice or two slices swept beside the row (``_full_slices_meet``): the
      two take thread axes of their own, so a slice is not swept on the
      row's axis (``slice_packet_entries``), and where a store or an atomic
      spans the row and a slice the lowering refuses the rolled row.
    """
    spec = env.config_spec
    if not _has_one_sized_reduction_loop(spec) or len(device_ir.grid_block_ids) != 1:
        return None
    axes = _row_tile_axes(env, device_ir)
    if not axes:
        return None
    loop = cast("ReductionLoopSpec", spec.reduction_loops[0])
    block_id = loop.block_id
    row_index = spec.block_sizes.block_id_to_index(axes[-1])
    extent = spec.cute_static_reduction_extents.get(block_id)
    if (
        extent is None
        or loop.pinned
        or block_id in spec.cute_indexed_reduction_block_ids
        or any(
            spec.block_sizes[outer]._fragment(spec).low > 1
            for outer in range(len(spec.block_sizes))
            if outer != row_index
        )
        or spec.cute_tile_symbolic_fixed_blocks
        or any(size != 1 for size in spec.cute_tile_fixed_block_sizes.values())
        or _row_is_no_run_of_packets(env, device_ir, block_id)
        or _full_slices_meet(env, device_ir)
    ):
        return None
    vec = _cute_tile_seed_vec_width_for_dtype(
        _cute_tile_inner_block_dtype(env, device_ir, block_id)
    )
    if vec <= 1:
        return None
    return _RowPlan(block_id, extent, row_index, vec)


def slice_packet_entries(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    block_id: int,
    threads: int,
    packet_bytes: int = 16,
) -> tuple[dict[int, int], dict[int, int]] | None:
    """``num_threads`` and ``cute_vector_widths`` entries that give each slice
    swept beside ``block_id``, the reduced row of a row plan, one packet per
    row thread, or ``None`` when some slice cannot take one.

    The row is the one block the kernel reduces and no two full-slice
    blocks meet in a value or an access (``_row_plan``), so every other
    full-slice block with a static extent is swept, not reduced, beside it
    (the half rows of rotate_half beside an RMSNorm row).  Such a block
    walks its elements on the reduced row's threads instead of an axis of
    its own when the row has at least the threads the block would take
    alone, or the config spells the row's thread count for it
    (``CuteReductionEmitter._reduction_axis_partner``, which needs the
    plan's one root grid).

    The slice's lanes vectorize only as one vector per thread (a slice of two
    packets per thread stays scalar), so its width is its share of the
    ``threads`` row threads, when that is a whole vector of at most
    ``packet_bytes`` (the row's own packet) of every dtype loaded along the
    slice: an fp32 weight beside a bf16 row limits a 16-byte packet to four
    elements.  The explicit thread count makes the slice index the row's
    thread axis where the row has fewer threads than the slice would take
    alone.  A slice whose share is no such vector, or whose dtype loads no
    packets at all (fp64, an integer: ``multiwarp_vector_row_seed_config``
    declines such a row too), would keep scalar lanes on an axis of its own,
    a layout these seeds do not spell.
    """
    spec = env.config_spec
    thread_entries: dict[int, int] = {}
    width_entries: dict[int, int] = {}
    for slice_block, extent in sorted(spec.cute_static_reduction_extents.items()):
        if slice_block == block_id:
            continue
        width, rest = divmod(extent, threads)
        # Elements of a 16-byte packet of the dtypes loaded along the slice;
        # 1 for a dtype that loads no packets (fp64, the integers), at any
        # packet width.
        vec = _cute_tile_seed_vec_width_for_dtype(
            _cute_tile_inner_block_dtype(env, device_ir, slice_block)
        )
        choices = (
            spec.cute_vector_widths.block_id_lookup(slice_block)._fragment(spec).choices
        )
        if (
            vec <= 1
            or rest
            or not 2 <= width <= vec * packet_bytes // 16
            or width not in choices
        ):
            return None
        thread_entries[slice_block] = threads
        width_entries[slice_block] = width
    return thread_entries, width_entries


class _RowPlanFamily(AutotunerHeuristic):
    """A seed family of the row plan.  ``get_seed_configs`` makes the plan
    for a caller that wants one family; ``row_packet_seed_configs`` makes it
    once for the three."""

    backend = "cute"

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        plan = _row_plan(env, device_ir)
        return [] if plan is None else cls.plan_seed_configs(env, device_ir, plan)

    @classmethod
    def plan_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR, plan: _RowPlan
    ) -> list[Config]:
        """The family's seeds for a kernel with the row plan ``plan``."""
        raise NotImplementedError


class CuteSubwarpRowHeuristic(_RowPlanFamily):
    """Several rows per warp for rows that are no whole number of warps of
    16-byte packets.

    The warp rows give every thread of a warp its own packets, so they need a
    row of a multiple of ``32 * V`` elements
    (``CuteReductionTileHeuristic.warp_rows_seed_configs``).  Any other row
    of whole packets takes here the largest power-of-two thread count whose
    packets tile it exactly, ``T = 2**k`` with ``extent = T * V *
    packets_per_thread``: 16 threads of one packet for 128 bf16 elements, 8
    threads of three for 192, 16 threads of three for 384.  A warp then
    holds ``32 / T`` rows.  The row is one rolled trip, exact for a length
    that is no power of two (a persistent row of such a length runs scalar),
    in the warp rows' form (``CuteReductionTileHeuristic.rolled_rows_seed``:
    strided lanes and register-prefetch reloads).

    Each row is seeded at both packet widths the target loads, 16 bytes and,
    on sm_100, 32 bytes (half the threads), as
    ``CuteReductionTileHeuristic.get_seed_configs`` does for the one-vector
    rows: which is faster depends on the kernel.  On B200, cold, against
    Inductor's Triton kernel in the same process (three processes each),
    Gemma's q/k RMSNorm with rotation (rows of 128) read 1.02x with 16
    threads of one 16-byte packet and 0.97x with 8 threads of one 32-byte
    packet, where DeiT's residual LayerNorm (rows of 192) read 0.87x with 8
    threads of three 16-byte packets and reached 1.00x only with 4 threads
    of three 32-byte packets (there with its other knobs tuned).

    A CTA holds ``ROWS_PER_CTA`` rows of the innermost tile axis (the RMSNorm
    rows read 0.95x at 4 rows and 1.02x at 8), raised to the grid's
    occupancy floor and lowered to the axis' largest block.  A packet width
    at which those rows hold less than one warp of threads (an axis of one
    or two rows) or more threads than a CTA launches (a floor of 128 rows of
    16 threads) gets no seed.
    Slices swept beside the row take one packet per row thread
    (``slice_packet_entries``); a packet width at which some slice cannot
    gets no seed.

    A thread walks at most ``MAX_THREAD_PACKETS`` 16-byte packets of its row
    (256 bytes; eight 32-byte packets), the most a thread of the warp rows
    walks of the longest fp16 or bf16 row they take
    (``WARP_ROWS_MAX_EXTENT / 32`` elements).  The layouts timed are one and
    three packets per thread; beyond the cap a row has few threads walking
    most of it (a 4000-element bf16 row would run 4 threads of 125 packets),
    which is no row of packets shared by a warp, and past 32 packets the
    lowering drops the register prefetch.  With at most 16 threads on a row,
    the cap bounds these rows at 1920 bf16 or fp16 elements and 960 fp32.

    No seed for:

    * rows of whole warps of packets (the warp rows);
    * rows whose thread count would fall below ``MIN_ROW_THREADS`` (a packet
      count that is odd or twice an odd number: 200, 208 or 240 bf16
      elements), and rows whose threads would walk more than
      ``MAX_THREAD_PACKETS`` packets (1600 bf16 elements: 8 threads of 25);
    * kernels whose outer tile axes do not admit one block per CTA, kernels
      with a block size that is no axis of the root grid or has no single
      extent, kernels with a tile loop fixed at more than one block, kernels
      that reduce a second block or carry two full slices in one value,
      store or atomic, argmin and argmax rows, and reductions along a
      strided dim (``_row_plan``).
    """

    name = "cute_subwarp_rows"
    PACKET_BYTES = (16, 32)
    ROWS_PER_CTA = 8
    MIN_ROW_THREADS = 4
    #: The most 16-byte packets of a row one thread walks.
    MAX_THREAD_PACKETS = 16

    @classmethod
    def row_threads(cls, extent: int, vec: int, packet_bytes: int = 16) -> int | None:
        """Threads of a row of ``extent`` elements, ``vec`` to a 16-byte
        packet, loaded as ``packet_bytes`` packets, or ``None`` for a row
        these seeds do not take."""
        if extent % (WARP_THREADS * vec) == 0:
            return None
        packets, rest = divmod(extent, vec * packet_bytes // 16)
        # The largest power of two dividing the packet count.
        threads = packets & -packets
        if (
            rest
            or threads < cls.MIN_ROW_THREADS
            or extent > threads * vec * cls.MAX_THREAD_PACKETS
        ):
            return None
        return threads

    @classmethod
    def plan_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR, plan: _RowPlan
    ) -> list[Config]:
        spec = env.config_spec
        # A grid too large for these CTAs starts at its occupancy floor
        # (``ConfigSpec.raise_grid_block_minimums``); a short axis holds
        # fewer rows than a CTA would take.
        rows_fragment = spec.block_sizes[plan.row_index]._fragment(spec)
        rows = min(max(cls.ROWS_PER_CTA, rows_fragment.low), rows_fragment.high)
        seeds: list[Config] = []
        for packet_bytes in cls.PACKET_BYTES:
            if packet_bytes == 32 and not (
                (spec.target_device_capability or (0, 0)) >= (10, 0)
            ):
                continue
            threads = cls.row_threads(plan.extent, plan.vec, packet_bytes)
            if threads is None:
                continue
            # A CTA of less than a warp (one or two rows of a short axis) is
            # no row packed into warps, and one past the launch limit would
            # not run on the threads the seed spells.
            if not WARP_THREADS <= rows * threads <= MAX_THREADS_PER_BLOCK:
                continue
            slices = slice_packet_entries(
                env, device_ir, plan.block_id, threads, packet_bytes
            )
            if slices is None:
                continue
            slice_threads, slice_widths = slices
            seeds.append(
                Config.from_dict(
                    CuteReductionTileHeuristic.rolled_rows_seed(
                        spec,
                        row_index=plan.row_index,
                        block_id=plan.block_id,
                        extent=plan.extent,
                        rows=rows,
                        threads=threads,
                        vec=plan.vec * packet_bytes // 16,
                        slice_threads=slice_threads,
                        slice_widths=slice_widths,
                    )
                )
            )
        return seeds


class CuteFlooredWarpRowHeuristic(_RowPlanFamily):
    """The warp rows on grids too large for one row per CTA.

    ``CuteReductionTileHeuristic`` declines a row kernel whose innermost tile
    axis cannot be searched at one row per CTA: 25216 rows exceed the grid
    budget, so ``ConfigSpec.raise_grid_block_minimums`` starts the axis at
    two rows.  Its narrow-chunk and one-vector seeds are one row per CTA,
    but its warp rows (``warp_rows_seed_configs``) hold two and four, so
    such a kernel takes the ones at or above its floor.

    The gate is the row plan and that floor, read from the row axis: a row
    kernel ``CuteReductionTileHeuristic`` declines for another reason gets
    no warp rows here, and neither does one the plan declines where that
    heuristic would not (a second root loop, a row that loads no packets).
    """

    name = "cute_floored_warp_rows"

    @classmethod
    def plan_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR, plan: _RowPlan
    ) -> list[Config]:
        spec = env.config_spec
        if spec.block_sizes[plan.row_index]._fragment(spec).low <= 1:
            return []
        return CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir)


class CuteSliceSweepPacketHeuristic(_RowPlanFamily):
    """The warp rows with one packet per thread on the slices swept beside
    the row.

    The warp-rows seeds (``CuteReductionTileHeuristic.warp_rows_seed_configs``)
    set the reduced row's vector width only.  A kernel that sweeps other
    full-slice blocks after the reduction (Qwen3's and Gemma's q/k RMSNorm
    followed by the rotary embedding over the two half rows) then loads and
    stores those slices as scalars: with 32 row threads each thread walks
    ``slice / 32`` scalar lanes.  This twin of each warp-rows seed keeps its
    geometry and gives every such slice its per-thread share as one vector
    (``slice_packet_entries``): 8-byte packets for the 128-element half rows
    of a 256-wide bf16 row.  Only the widths change: beside 32 row threads
    and two or more rows a slice fits at most 16 threads of its own, so it
    indexes the row's axis without a thread count of its own.
    """

    name = "cute_slice_sweep_packets"

    @classmethod
    def plan_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR, plan: _RowPlan
    ) -> list[Config]:
        spec = env.config_spec
        slices = slice_packet_entries(env, device_ir, plan.block_id, WARP_THREADS)
        if slices is None or not slices[1]:
            return []
        seeds: list[Config] = []
        for warp_rows in CuteReductionTileHeuristic.warp_rows_seed_configs(
            env, device_ir
        ):
            values = deepcopy(warp_rows.config)
            widths = cast("list[int]", values["cute_vector_widths"])
            for slice_block, width in slices[1].items():
                widths[spec.cute_vector_widths.block_id_to_index(slice_block)] = width
            seeds.append(Config.from_dict(values))
        return seeds


def row_packet_seed_configs(
    env: CompileEnvironment, device_ir: DeviceIR
) -> list[tuple[str, list[Config]]]:
    """The seeds of the three families by family name, in the order
    ``postprocess_seed_configs`` appends them: the warp rows of grids above
    one row per CTA, the rows that are no whole number of warps of packets,
    then the warp rows with packets on the slices swept beside the row.  One
    row plan serves the three."""
    plan = _row_plan(env, device_ir)
    if plan is None:
        return []
    return [
        (family.name, family.plan_seed_configs(env, device_ir, plan))
        for family in (
            CuteFlooredWarpRowHeuristic,
            CuteSubwarpRowHeuristic,
            CuteSliceSweepPacketHeuristic,
        )
    ]
