"""The ``cute_mma_dag="tcgen05"`` compute mode: MMA dataflow programs on
Blackwell's fifth-generation tensor cores.

The matcher, the typed IR, the operand routes and the autotuner plumbing are
the warp mode's (``cute_mma_dag.py``); this module owns the tcgen05 admission
facts, the per-config legality, the plan and the emitter.  Program of one CTA
(``32 * warps`` threads, warps 0-3 the TMEM *epilogue* warps -- one TMEM lane
quadrant each -- and, with eight warps, warps 4-7 the *store* warps; ``S``
ring slots)::

    mbarriers: full[S] (ring TMA transactions), acc_ready (loop commits),
        efull (the epilogue operand tiles' TMA), eacc (epilogue commits)
    TMEM: every fragment slot (the carries, the epilogue dot results) at its
        own 32-column-aligned offset (warp 0 allocates / deallocates)
    carries <- zeros | the seed tiles (tcgen05.st from per-row register loads)
    warp 0: TMA the loop operand tiles of the first S - 1 iterations (one
        arrive.expect_tx per slot) and the epilogue operand tiles (efull)
    for i in loop:
        wait acc_ready(i - 1): the carries hold the state at loop entry
        operands TMA cannot fetch: cooperative passes into the slot's UMMA
            layout (sync), or in-place rewrites of the landed tile (inplace /
            convert) by every thread, fp32 math on the unpacked elements with
            one rounding in the pack, then fence.proxy.async
        epilogue warps: tcgen05.ld the carry rows (64 columns at a time) ->
            snapshot stores (bf16 / f16 / fp32 pairs into the double-buffered
            staging tile, or straight to global), rescales (Scalar / row Vec /
            col Vec through Inductor's pointwise lowering) -> tcgen05.st
        tcgen05.fence::before_thread_sync; CTA barrier
        warp 0: wait full[i % S]; for every loop dot in program order the K
            / 16 tcgen05.mma over the UMMA smem descriptors (SW32 / SW64 /
            SW128, slabs of one swizzle atom width); one tcgen05.commit ->
            acc_ready; TMA iteration i + S - 1 into the slot MMA(i - 1) read
        store warps (or the epilogue warps): staging tile -> 16-byte global
            packets, overlapping the MMA and the next iteration's loads
    wait acc_ready(last)
    epilogue: wait efull, passes of the epilogue operand tiles; fragment
        operands: tcgen05.ld the finished accumulator -> the pointwise chain
        (mask / scale / cast) per element -> packed pairs into a UMMA
        exchange tile (fence.proxy.async), read K-major or MN-major (a
        transposed use) by the epilogue dots; accumulator rescales / copies
        through tcgen05.ld -> tcgen05.st; CTA barrier; warp 0 issues the
        epilogue dots in program order (a dot that consumes an earlier
        epilogue dot's result waits for its commit first), one commit ->
        eacc; the epilogue stores: tcgen05.ld -> the store's pointwise
        chain -> staging tile -> 16-byte global packets
    CTA barrier; warp 0: tcgen05.dealloc

Phase bits: ``full[s]`` completes once per use of slot ``s``, so iteration
``i`` waits parity ``((i - begin) // S) & 1``; ``acc_ready`` completes once
per iteration, so the carries of iteration ``i`` wait parity ``(i - begin -
1) & 1``; ``eacc`` completes once per epilogue commit, waited in order.  A
ring slot is refilled only after the MMA that read it committed (that commit
is the acc_ready the refilling iteration waited for), and the loop staging
buffer of parity ``p`` is rewritten only after the CTA barrier that follows
its drain.

Admission (fail closed: the knob value is ``InvalidConfig`` / the mode is
absent from the search surface): sm100 / sm103, bf16 / f16 operands, fp32
fragments, loop dots over load operands, epilogue dots over load or fragment
operands, stores of fragment values; per fragment slot M in {64, 128} and N
a multiple of 8 up to 256, at most 512 TMEM columns over all slots, K a
multiple of 16, operand and exchange tiles with rows at most 256 and a
contiguous extent of 32 / 64 bytes or a multiple of 128 bytes (one swizzle
atom row, or whole 128-byte atoms side by side); TMA needs the direct-route
alignment proof and at most five non-unit tensor dims, otherwise the operand
takes the synchronous pass.

Four warps (the epilogue warps drain the staging tile after the MMA issue)
match or beat eight on the s2 state kernels; the pass sits in the chain
because rewriting the next slot from the store warps measured slower.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_RDIM
from .cute_mma_dag import AXIS_ROOT
from .cute_mma_dag import MMA_DAG_K_STEP
from .cute_mma_dag import MMA_DAG_STAGE_CHOICES
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import MMA_DAG_WARPS_KEY
from .cute_mma_dag import ROUTE_CONVERT
from .cute_mma_dag import ROUTE_DIRECT
from .cute_mma_dag import ROUTE_EXCHANGE
from .cute_mma_dag import ROUTE_IDENTITY
from .cute_mma_dag import ROUTE_INPLACE
from .cute_mma_dag import ROUTE_SYNC
from .cute_mma_dag import Access
from .cute_mma_dag import Axis
from .cute_mma_dag import CuteMmaDagFacts
from .cute_mma_dag import Dot
from .cute_mma_dag import MmaDagPlan
from .cute_mma_dag import MmaDagProgram
from .cute_mma_dag import Operand
from .cute_mma_dag import Store
from .cute_mma_dag import _decline
from .cute_mma_dag import staged_store_reason
from .cute_mma_dag_codegen import _RUNTIME
from .cute_mma_dag_codegen import Coords
from .cute_mma_dag_codegen import _broadcast
from .cute_mma_dag_codegen import _Emitter
from .cute_mma_dag_codegen import _Evaluator
from .cute_mma_dag_codegen import _shape
from .cute_mma_dag_codegen import emit_mma_dag_module_statements
from .cute_warp_mma_gemm import _swizzled_chunk
from .matmul_utils import strip_new_vars

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..generate_ast import GenerateAST

TCGEN05_WARP_CHOICES: tuple[int, ...] = (4, 8)
TCGEN05_EPILOGUE_WARPS = 4
TCGEN05_M_CHOICES: tuple[int, ...] = (64, 128)
TCGEN05_MAX_N = 256
TCGEN05_MAX_ROWS = 256
TCGEN05_MAX_TMEM_COLS = 512
# TMEM allocation granularity: slot offsets keep it.
TCGEN05_SLOT_ALIGN = 32
# Four warps (the epilogue warps drain the staging tile themselves) match or
# beat eight at s2: fwd 104.5 vs 104.6 us, bwd 189.6 vs 222 us device time.
TCGEN05_DEFAULT_WARPS = 4
TCGEN05_SEED_WARPS: tuple[int, ...] = (4, 8)
TCGEN05_DEFAULT_STAGES = 8
TCGEN05_SEED_STAGES: tuple[int, ...] = (8, 4)
TCGEN05_TILE_LADDER: tuple[int, ...] = (64, 128)
TCGEN05_N_LADDER: tuple[int, ...] = (64, 128, 32, 16)
# Columns per tcgen05.ld / st (64 fp32 registers per thread).
TCGEN05_LD_COLS = 64
# Side loads (decay scalars / rows) carried one iteration ahead in registers.
TCGEN05_MAX_PREFETCH = 16
# One swizzle atom row: the widest swizzle (SW128) spans 128 bytes.
ATOM_MAX_BYTES = 128
SLAB_ALIGNMENT = 1024
TMA_MAX_RANK = 5
TMA_KIND = "helion_mma_dag_tma"
_SWIZZLE_BY_ROW_BYTES = {32: "s32b", 64: "s64b", 128: "s128b"}
_DESC_SWIZZLE = {"s32b": "SWIZZLE_32B", "s64b": "SWIZZLE_64B", "s128b": "SWIZZLE_128B"}
_RT = "_helion_mma_dag_tc"
_PRIMS = f"{_RT}.prims"
_PREAMBLE = f"import helion._compiler.cute._mma_dag_tcgen05_runtime as {_RT}\n"
REGION_LOOP = "loop"
REGION_EPILOGUE = "epilogue"
REGION_EXCHANGE = "exchange"


# ---------------------------------------------------------------------------
# Facts (config-free) and per-config legality
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tcgen05TileFacts:
    """One unique operand tile: the axis indices of its shared tile (rows =
    the non-contiguous dim), the element sizes and the region it serves.
    Whether TMA fetches the source is the planner's route (the bound pointer
    alignment is not known here)."""

    rows: int
    cols: int
    elem_bytes: int
    source_bytes: int
    region: str


@dataclasses.dataclass(frozen=True)
class Tcgen05ExchangeFacts:
    """A fragment read as an epilogue operand: its slot's (rows, cols)
    axis indices and the operand element size."""

    slot: int
    rows: int
    cols: int
    elem_bytes: int


@dataclasses.dataclass(frozen=True)
class Tcgen05Facts:
    tiles: tuple[Tcgen05TileFacts, ...]
    exchanges: tuple[Tcgen05ExchangeFacts, ...]
    # (slot, element size) of the loop snapshots (two staging buffers each)
    # and of the epilogue stores (one staging tile at a time).
    loop_stores: tuple[tuple[int, int], ...]
    epilogue_stores: tuple[tuple[int, int], ...]


def _tma_dims(access: Access) -> tuple[int, ...]:
    """The tensor dims of ``access`` in TMA order (innermost first), unit
    dims dropped."""
    sizes = [int(s) for s in access.fake.shape]
    dims = [d for d in range(len(sizes)) if sizes[d] != 1]
    dims.sort(key=lambda d: access.strides[d])
    return tuple(dims)


def operand_k_major(operand: Operand) -> bool:
    """Whether the operand's K dim runs along its shared tile's columns."""
    return not operand.trans


def tcgen05_facts(program: MmaDagProgram) -> Tcgen05Facts | None:
    """The tcgen05 admission facts of ``program`` (None: not admitted on this
    target or program shape)."""
    env = CompileEnvironment.current()
    target = env.config_spec.target_device_capability
    if target is None or target[0] != 10:
        return None
    if program.dtype not in (torch.bfloat16, torch.float16):
        return None
    # Derived fragments and mid-iteration stores of values other than a
    # loop dot's result (the warp mode's program-order body) are not
    # emitted here: they keep the warp mode.
    loop_nodes = {dot.node for dot in program.dots_in("loop")}
    if program.derived or any(
        store.source == "body" and store.frag not in loop_nodes
        for store in program.stores
    ):
        return None
    # The loop issues every dot of an iteration under one barrier and reads
    # the non-snapshot stores after them: a chained accumulator (``x =
    # dot(acc=h); h = dot(acc=x * 0.5)``) or a store before a later dot
    # needs the warp mode's program-order body.
    if any(dot.acc_frag in loop_nodes for dot in program.dots_in("loop")):
        return None
    order = [
        step
        for step in program.steps
        if not (isinstance(step, Store) and step.source == "carry_in")
    ]
    stores_from = next(
        (i for i, step in enumerate(order) if not isinstance(step, Dot)), len(order)
    )
    if any(isinstance(step, Dot) for step in order[stores_from:]):
        return None
    tiles: list[Tcgen05TileFacts] = []
    exchanges: list[Tcgen05ExchangeFacts] = []
    seen: set[tuple[object, ...]] = set()
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.chain_key in seen:
                continue
            seen.add(operand.chain_key)
            if operand.kind == "load":
                load = operand.load
                assert load is not None
                tiles.append(
                    Tcgen05TileFacts(
                        rows=operand.rows.index,
                        cols=operand.cols.index,
                        elem_bytes=operand.dtype.itemsize,
                        source_bytes=load.dtype.itemsize,
                        region=dot.region,
                    )
                )
            else:
                if dot.region != "epilogue" or operand.frag is None:
                    return None
                slot = program.frag_slots[operand.frag]
                exchanges.append(
                    Tcgen05ExchangeFacts(
                        slot=slot,
                        rows=operand.rows.index,
                        cols=operand.cols.index,
                        elem_bytes=operand.dtype.itemsize,
                    )
                )
    loop_stores = tuple(
        (program.frag_slots[store.frag], store.access.dtype.itemsize)
        for store in program.stores
        if store.region == "loop"
    )
    epilogue_stores = tuple(
        (program.frag_slots[store.frag], store.access.dtype.itemsize)
        for store in program.stores
        if store.region == "epilogue"
    )
    return Tcgen05Facts(
        tiles=tuple(tiles),
        exchanges=tuple(exchanges),
        loop_stores=loop_stores,
        epilogue_stores=epilogue_stores,
    )


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def tcgen05_atom_bytes(row_bytes: int) -> int | None:
    """The swizzle atom row of a UMMA tile whose rows are ``row_bytes`` wide
    (None: no canonical swizzled layout covers the row)."""
    if row_bytes in _SWIZZLE_BY_ROW_BYTES:
        return row_bytes
    if row_bytes % ATOM_MAX_BYTES == 0:
        return ATOM_MAX_BYTES
    return None


def tcgen05_tmem_cols(total: int) -> int:
    cols = 32
    while cols < total:
        cols *= 2
    return cols


def tcgen05_slot_columns(facts: CuteMmaDagFacts, resolved: Sequence[int]) -> int:
    """TMEM columns of all fragment slots at 32-column-aligned offsets."""
    return sum(
        _align(resolved[cols], TCGEN05_SLOT_ALIGN) for _rows, cols in facts.frags
    )


def tcgen05_stage_bytes(facts: Tcgen05Facts, resolved: Sequence[int]) -> int:
    """Shared memory of one ring slot: every loop operand tile (and the raw
    source tile of a converted operand) rounded up to the swizzle alignment."""
    return sum(
        _tile_bytes(tile, resolved)
        for tile in facts.tiles
        if tile.region == REGION_LOOP
    )


def _tile_bytes(tile: Tcgen05TileFacts, resolved: Sequence[int]) -> int:
    rows, cols = resolved[tile.rows], resolved[tile.cols]
    total = _align(rows * cols * tile.elem_bytes, SLAB_ALIGNMENT)
    if tile.source_bytes != tile.elem_bytes:
        total += _align(rows * cols * tile.source_bytes, SLAB_ALIGNMENT)
    return total


def tcgen05_fixed_bytes(
    facts: CuteMmaDagFacts, resolved: Sequence[int], staged: bool
) -> int:
    """Shared memory beside the ring: the epilogue operand tiles, the
    exchange tiles, the loop staging buffers and the epilogue staging tile."""
    tc = facts.tcgen05
    assert tc is not None
    total = sum(
        _tile_bytes(tile, resolved)
        for tile in tc.tiles
        if tile.region == REGION_EPILOGUE
    )
    for exchange in tc.exchanges:
        total += _align(
            resolved[exchange.rows] * resolved[exchange.cols] * exchange.elem_bytes,
            SLAB_ALIGNMENT,
        )
    if staged:
        for slot, es in tc.loop_stores:
            rows, cols = facts.frags[slot]
            total += 2 * resolved[rows] * resolved[cols] * es
    epilogue_staging = 0
    for slot, es in tc.epilogue_stores:
        rows, cols = facts.frags[slot]
        epilogue_staging = max(epilogue_staging, resolved[rows] * resolved[cols] * es)
    return total + epilogue_staging


def tcgen05_smem_bytes(
    facts: CuteMmaDagFacts, resolved: Sequence[int], stages: int, staged: bool
) -> int:
    tc = facts.tcgen05
    assert tc is not None
    total = stages * tcgen05_stage_bytes(tc, resolved)
    total += tcgen05_fixed_bytes(facts, resolved, staged)
    # Barriers and the TMEM slot.
    return total + 8 * (stages + 3) + 16


def tcgen05_warps_reason(warps: object) -> str | None:
    if type(warps) is not int or warps not in TCGEN05_WARP_CHOICES:
        return (
            f"{MMA_DAG_WARPS_KEY} {warps!r} must be one of {TCGEN05_WARP_CHOICES} "
            f"for {MMA_DAG_TCGEN05}"
        )
    return None


def tcgen05_config_reason(
    facts: CuteMmaDagFacts, resolved: Sequence[int], warps: int, stages: int
) -> str | None:
    """Why the resolved block sizes cannot run the tcgen05 mode (None: they
    can).  The caller checked the warp / stage choices, the power-of-two and
    divisibility rules and the dot K step."""
    tc = facts.tcgen05
    if tc is None:
        return f"the kernel or the target does not admit {MMA_DAG_TCGEN05}"
    for rows, cols in facts.frags:
        bm, bn = resolved[rows], resolved[cols]
        if bm not in TCGEN05_M_CHOICES:
            return f"a tcgen05 tile needs {TCGEN05_M_CHOICES} rows, not {bm}"
        if bn % 8 or bn > TCGEN05_MAX_N:
            return (
                "a tcgen05 tile needs columns in multiples of 8 up to "
                f"{TCGEN05_MAX_N}, not {bn}"
            )
    columns = tcgen05_slot_columns(facts, resolved)
    if columns > TCGEN05_MAX_TMEM_COLS:
        return (
            f"the fragments need {columns} TMEM columns (limit {TCGEN05_MAX_TMEM_COLS})"
        )
    for tile in tc.tiles:
        rows, cols = resolved[tile.rows], resolved[tile.cols]
        if rows > TCGEN05_MAX_ROWS:
            return f"an operand tile of {rows} rows exceeds the TMA box"
        if tcgen05_atom_bytes(cols * tile.elem_bytes) is None:
            return (
                f"an operand tile of {cols} contiguous {tile.elem_bytes}-byte "
                "elements has no swizzled UMMA layout (32 / 64 bytes or "
                "multiples of 128)"
            )
    for exchange in tc.exchanges:
        cols = resolved[exchange.cols]
        if tcgen05_atom_bytes(cols * exchange.elem_bytes) is None:
            return (
                f"a fragment operand of {cols} columns has no swizzled UMMA "
                "layout (32 / 64 bytes or multiples of 128)"
            )
    smem = tcgen05_smem_bytes(facts, resolved, stages, True)
    if facts.smem_capacity and smem > facts.smem_capacity:
        return (
            f"{stages} tcgen05 stages need {smem} B of shared memory "
            f"(capacity {facts.smem_capacity})"
        )
    return None


def _ladder_for(facts: CuteMmaDagFacts, index: int) -> tuple[int, ...]:
    """The seed ladder of tunable axis ``index``: fragment rows take the
    {64, 128} M ladder, everything else the N ladder."""
    if any(rows == index for rows, _cols in facts.frags):
        return TCGEN05_TILE_LADDER
    return TCGEN05_N_LADDER


def tcgen05_repair(
    facts: CuteMmaDagFacts,
    sizes: Mapping[int, object],
    warps: object,
    stages: object,
) -> tuple[dict[int, int], int, int] | None:
    """A legal tcgen05 ``(sizes, warps, stages)`` near the request: fewer
    stages first, then the other warp count, then the tunable tiles from
    the ladders (None: nothing legal)."""
    from .cute_mma_dag import mma_dag_config_reason

    tc = facts.tcgen05
    if tc is None:
        return None
    tunable = [
        (index, axis)
        for index, axis in enumerate(facts.axes)
        if axis.kind in (AXIS_ROOT, AXIS_LOOP)
    ]
    current: dict[int, int] = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
        if axis.kind not in (AXIS_ROOT, AXIS_LOOP)
    }
    for _index, axis in tunable:
        value = sizes.get(axis.block_id)
        current[axis.block_id] = value if type(value) is int else 64
    warps_now = warps if warps in TCGEN05_WARP_CHOICES else TCGEN05_DEFAULT_WARPS
    stages_now = stages if stages in MMA_DAG_STAGE_CHOICES else TCGEN05_DEFAULT_STAGES
    assert type(warps_now) is int and type(stages_now) is int
    stage_ladder = [
        s for s in sorted(MMA_DAG_STAGE_CHOICES, reverse=True) if s <= stages_now
    ] or [1]
    ladders = [
        (current[axis.block_id], *_ladder_for(facts, index)) for index, axis in tunable
    ]
    for choice in itertools.product(*ladders):
        candidate = dict(current)
        for (_index, axis), tile in zip(tunable, choice, strict=True):
            if axis.extent % tile:
                break
            candidate[axis.block_id] = tile
        else:
            for w in (warps_now, *(w for w in TCGEN05_WARP_CHOICES if w != warps_now)):
                for s in stage_ladder:
                    if (
                        mma_dag_config_reason(facts, candidate, w, s, MMA_DAG_TCGEN05)
                        is None
                    ):
                        return candidate, w, s
    return None


def tcgen05_seed_configs(
    facts: CuteMmaDagFacts,
) -> list[tuple[dict[int, int], int, int]]:
    """Ranked tcgen05 ``(sizes, warps, stages)`` seeds: fragment rows from
    the {64, 128} ladder, other tunable axes from {64, 128, 32, 16}, four
    warps then eight, stages 8 then 4 (a tile loop: also 2 and 1, never
    deeper than its trip count).

    Ranked by CTAs >= 128, then fitting the grid in one wave: a grid beyond
    the SM count wants two CTAs per SM of shared memory *and* tensor memory
    (the K-loop kernels overlap one CTA's epilogue with the other's loop:
    fwd_o at s2 runs 136 us with one 128-deep stage at two CTAs per SM
    and 214 us with four 64-deep stages at one; the state recurrence at
    s1 wants four stages, 25 vs 31 us, where at s2 its 128 CTAs fit one
    wave and eight stages win, 104 vs 120), then the largest and squarest
    root tile, the loop tile, the stage depth and the warp count."""
    from .cute_mma_dag import mma_dag_config_reason

    tc = facts.tcgen05
    if tc is None:
        return []
    tunable = [
        (index, axis)
        for index, axis in enumerate(facts.axes)
        if axis.kind in (AXIS_ROOT, AXIS_LOOP)
    ]
    fixed = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
        if axis.kind not in (AXIS_ROOT, AXIS_LOOP)
    }
    ladders = [
        [tile for tile in _ladder_for(facts, index) if axis.extent % tile == 0]
        for index, axis in tunable
    ]
    ranked: list[tuple[tuple[int, ...], tuple[dict[int, int], int, int]]] = []
    for choice in itertools.product(*ladders):
        sizes: dict[int, int] = dict(fixed)
        trip = 0
        for (_index, axis), tile in zip(tunable, choice, strict=True):
            sizes[axis.block_id] = tile
            if axis.kind == AXIS_LOOP:
                trip = axis.extent // tile
        stage_choices: tuple[int, ...] = TCGEN05_SEED_STAGES
        if trip:
            stage_choices = tuple(
                s for s in sorted(MMA_DAG_STAGE_CHOICES, reverse=True) if s <= trip
            )
        for stages, warps in itertools.product(stage_choices, TCGEN05_SEED_WARPS):
            if (
                mma_dag_config_reason(facts, sizes, warps, stages, MMA_DAG_TCGEN05)
                is not None
            ):
                continue
            ctas = facts.grid_blocks
            root_area = 1
            root_perimeter = 0
            loop_area = 1
            for (_index, axis), tile in zip(tunable, choice, strict=True):
                if axis.kind == AXIS_ROOT:
                    ctas *= axis.extent // tile
                    root_area *= tile
                    root_perimeter += tile
                else:
                    loop_area *= tile
            resolved = [
                cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))
            ]
            smem = tcgen05_smem_bytes(facts, resolved, stages, True)
            tmem = tcgen05_tmem_cols(tcgen05_slot_columns(facts, resolved))
            two_per_sm = (
                not facts.smem_capacity or 2 * smem <= facts.smem_capacity
            ) and 2 * tmem <= TCGEN05_MAX_TMEM_COLS
            fits_wave = not facts.sm_count or ctas <= facts.sm_count or two_per_sm
            rank = (
                int(ctas >= 128),
                int(fits_wave),
                root_area,
                -root_perimeter,
                loop_area,
                stages,
                -warps,
            )
            ranked.append((rank, (sizes, warps, stages)))
    ranked.sort(key=operator.itemgetter(0), reverse=True)
    return [seed for _rank, seed in ranked]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TmaSpec:
    """One host-built tensor map: the kernel parameter, the tensor dims in
    TMA order, the box (same order) and the shared slab it fills."""

    index: int
    param: str
    access: Access
    dims: tuple[int, ...]
    box: tuple[int, ...]
    swizzle: str
    # Position (in ``dims``) of the contiguous dim the slabs step along, and
    # the step in elements (the atom width).
    slab_dim: int
    slab_elems: int
    slabs: int
    slab_bytes: int
    # Byte offset of the first slab: within a ring slot (loop tiles) or
    # within the fixed region (epilogue tiles).
    offset: int

    @property
    def bytes(self) -> int:
        return self.slabs * self.slab_bytes


@dataclasses.dataclass(frozen=True)
class Tcgen05Tile:
    """A UMMA shared tile: an operand tile (loop: one per ring slot, epilogue:
    once) or an exchange tile written from a fragment.  ``offset`` is
    relative to the ring slot for loop tiles and to the fixed region
    otherwise."""

    key: tuple[object, ...]
    # Short unique tag (the passes' fixed per-packet names).
    tag: str
    region: str
    operand: Operand
    route: str
    rows: int
    cols: int
    elem_bytes: int
    atom_bytes: int
    swizzle: str
    slabs: int
    offset: int
    tma: TmaSpec | None = None
    # Converted operands: the raw source tile beside the operand tile.
    raw_offset: int = 0
    raw_row_bytes: int = 0
    raw_tma: TmaSpec | None = None
    # Exchange tiles: the fragment slot they are written from.
    source_slot: int | None = None

    @property
    def row_bytes(self) -> int:
        return self.cols * self.elem_bytes

    @property
    def slab_bytes(self) -> int:
        return self.rows * self.atom_bytes

    @property
    def bytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def chunks_per_atom(self) -> int:
        return self.atom_bytes // 16

    @property
    def chunks_per_row(self) -> int:
        return self.row_bytes // 16

    @property
    def k_steps_per_slab(self) -> int:
        return self.atom_bytes // (MMA_DAG_K_STEP * self.elem_bytes)

    def k_step_offset(self, step: int, k_major: bool) -> int:
        """Byte advance of the descriptor start for K step ``step``."""
        if k_major:
            slab, local = divmod(step, self.k_steps_per_slab)
            return slab * self.slab_bytes + local * MMA_DAG_K_STEP * self.elem_bytes
        return step * MMA_DAG_K_STEP * self.atom_bytes

    def leading_byte_offset(self, k_major: bool) -> int:
        return 16 if k_major else self.slab_bytes

    @property
    def stride_byte_offset(self) -> int:
        return 8 * self.atom_bytes


@dataclasses.dataclass(frozen=True)
class SlotGeometry:
    rows: int
    cols: int
    # First TMEM column of the slot.
    column: int

    @property
    def ld_chunks(self) -> int:
        return -(-self.cols // TCGEN05_LD_COLS)

    def chunk_cols(self, chunk: int) -> int:
        return min(TCGEN05_LD_COLS, self.cols - chunk * TCGEN05_LD_COLS)


@dataclasses.dataclass(frozen=True)
class Tcgen05Plan:
    warps: int
    stages: int
    slots: tuple[SlotGeometry, ...]
    # Shared tiles keyed by (dot node, role).
    tiles: dict[tuple[Node, str], Tcgen05Tile]
    loop_tiles: tuple[Tcgen05Tile, ...]
    epilogue_tiles: tuple[Tcgen05Tile, ...]
    exchange_tiles: tuple[Tcgen05Tile, ...]
    tma: tuple[TmaSpec, ...]
    stage_bytes: int
    tx_bytes: int
    # Fixed region (after the ring): epilogue tiles, exchange tiles, staging.
    fixed_offset: int
    epilogue_tx_bytes: int
    staged_stores: tuple[Store, ...]
    staging_offsets: dict[Node, int]
    epilogue_staging_offset: int
    tmem_cols: int
    smem_bytes: int

    @property
    def store_warps(self) -> int:
        return self.warps - TCGEN05_EPILOGUE_WARPS

    def tile(self, dot: Dot, operand: Operand) -> Tcgen05Tile:
        return self.tiles[(dot.node, operand.role)]


def plan_tcgen05(
    program: MmaDagProgram,
    facts: CuteMmaDagFacts,
    sizes: tuple[int, ...],
    routes: Mapping[tuple[Node, str], str],
    warps: int,
    stages: int,
    store_mode: str,
) -> Tcgen05Plan:
    """The tcgen05 plan for the resolved ``sizes`` (the config passed
    :func:`tcgen05_config_reason`)."""
    env = CompileEnvironment.current()
    tc = facts.tcgen05
    assert tc is not None
    tma: list[TmaSpec] = []

    def tma_spec(
        access: Access,
        rows: Axis,
        cols: Axis,
        row_count: int,
        col_count: int,
        elem_bytes: int,
        swizzle: str,
        atom_bytes: int,
        at: int,
    ) -> TmaSpec:
        dims = _tma_dims(access)
        tensor_dims = {axis_dim: axis for axis, axis_dim in _axis_dims(access)}
        box = []
        slab_dim = None
        slab_elems = col_count
        for position, dim in enumerate(dims):
            axis = tensor_dims.get(dim)
            if axis is cols:
                slab_elems = min(col_count, atom_bytes // elem_bytes)
                box.append(slab_elems)
                slab_dim = position
            elif axis is rows:
                box.append(row_count)
            else:
                box.append(1)
        assert slab_dim is not None
        spec = TmaSpec(
            index=len(tma),
            param=f"md_tma{len(tma)}",
            access=access,
            dims=dims,
            box=tuple(box),
            swizzle=swizzle,
            slab_dim=slab_dim,
            slab_elems=slab_elems,
            slabs=col_count // slab_elems,
            slab_bytes=row_count * slab_elems * elem_bytes,
            offset=at,
        )
        tma.append(spec)
        return spec

    tiles: dict[tuple[Node, str], Tcgen05Tile] = {}
    by_key: dict[tuple[object, ...], Tcgen05Tile] = {}
    loop_tiles: list[Tcgen05Tile] = []
    epilogue_tiles: list[Tcgen05Tile] = []
    exchange_tiles: list[Tcgen05Tile] = []
    stage_offset = 0
    fixed = 0

    def load_tile(
        dot: Dot, operand: Operand, region: str, at: int
    ) -> tuple[Tcgen05Tile, int]:
        route = routes[(dot.node, operand.role)]
        load = operand.load
        assert load is not None
        if route != ROUTE_SYNC and len(_tma_dims(load)) > TMA_MAX_RANK:
            route = ROUTE_SYNC
        rows, cols = sizes[operand.rows.index], sizes[operand.cols.index]
        es = operand.dtype.itemsize
        atom = tcgen05_atom_bytes(cols * es)
        assert atom is not None
        swizzle = _SWIZZLE_BY_ROW_BYTES[atom]
        tile_offset = at
        at += _align(rows * cols * es, SLAB_ALIGNMENT)
        spec = raw_spec = None
        raw_offset = raw_row_bytes = 0
        if route in (ROUTE_DIRECT, ROUTE_INPLACE):
            spec = tma_spec(
                load,
                operand.rows,
                operand.cols,
                rows,
                cols,
                es,
                swizzle,
                atom,
                tile_offset,
            )
        elif route == ROUTE_CONVERT:
            raw_offset = at
            raw_row_bytes = cols * load.dtype.itemsize
            at += _align(rows * raw_row_bytes, SLAB_ALIGNMENT)
            raw_spec = tma_spec(
                load,
                operand.rows,
                operand.cols,
                rows,
                cols,
                load.dtype.itemsize,
                "none",
                raw_row_bytes,
                raw_offset,
            )
        elif route != ROUTE_SYNC:
            raise _decline(f"operand route {route!r} is not a tcgen05 route")
        return (
            Tcgen05Tile(
                key=operand.chain_key,
                tag=f"t{len(by_key)}",
                region=region,
                operand=operand,
                route=route,
                rows=rows,
                cols=cols,
                elem_bytes=es,
                atom_bytes=atom,
                swizzle=swizzle,
                slabs=(cols * es) // atom,
                offset=tile_offset,
                tma=spec,
                raw_offset=raw_offset,
                raw_row_bytes=raw_row_bytes,
                raw_tma=raw_spec,
            ),
            at,
        )

    # Loop operand tiles share the ring; epilogue operand tiles and the
    # exchange tiles live once in the fixed region after the ring.
    for dot in program.dots_in("loop"):
        for operand in (dot.a, dot.b):
            if operand.kind != "load":
                raise _decline("a loop dot operand must be a load")
            tile = by_key.get(operand.chain_key)
            if tile is None:
                tile, stage_offset = load_tile(dot, operand, REGION_LOOP, stage_offset)
                by_key[operand.chain_key] = tile
                loop_tiles.append(tile)
            tiles[(dot.node, operand.role)] = tile
    stage_bytes = stage_offset
    tx_bytes = sum(spec.bytes for spec in tma)
    fixed_offset = stages * stage_bytes
    epilogue_specs_from = len(tma)
    for dot in program.dots_in("epilogue"):
        for operand in (dot.a, dot.b):
            tile = by_key.get(operand.chain_key)
            if tile is not None:
                tiles[(dot.node, operand.role)] = tile
                continue
            if operand.kind == "load":
                tile, fixed = load_tile(dot, operand, REGION_EPILOGUE, fixed)
                epilogue_tiles.append(tile)
            else:
                assert operand.frag is not None
                route = routes[(dot.node, operand.role)]
                if route not in (ROUTE_EXCHANGE, ROUTE_IDENTITY):
                    raise _decline(f"operand route {route!r} is not a tcgen05 route")
                slot = program.frag_slots[operand.frag]
                rows, cols = sizes[operand.rows.index], sizes[operand.cols.index]
                es = operand.dtype.itemsize
                atom = tcgen05_atom_bytes(cols * es)
                assert atom is not None
                tile = Tcgen05Tile(
                    key=operand.chain_key,
                    tag=f"t{len(by_key)}",
                    region=REGION_EXCHANGE,
                    operand=operand,
                    route=ROUTE_EXCHANGE,
                    rows=rows,
                    cols=cols,
                    elem_bytes=es,
                    atom_bytes=atom,
                    swizzle=_SWIZZLE_BY_ROW_BYTES[atom],
                    slabs=(cols * es) // atom,
                    offset=fixed,
                    source_slot=slot,
                )
                fixed += _align(rows * cols * es, SLAB_ALIGNMENT)
                exchange_tiles.append(tile)
            by_key[operand.chain_key] = tile
            tiles[(dot.node, operand.role)] = tile
    epilogue_tx_bytes = sum(spec.bytes for spec in tma[epilogue_specs_from:])
    # Fragment slots: TMEM columns at 32-column-aligned offsets.
    slots: list[SlotGeometry] = []
    column = 0
    for rows_axis, cols_axis in program.slots:
        rows, cols = sizes[rows_axis.index], sizes[cols_axis.index]
        slots.append(SlotGeometry(rows=rows, cols=cols, column=column))
        column += _align(cols, TCGEN05_SLOT_ALIGN)
    # Loop snapshots through double-buffered staging tiles; the epilogue
    # stores through one staging tile (reusing the ring when it fits).
    staged: list[Store] = []
    staging_offsets: dict[Node, int] = {}
    for store in program.stores:
        slot = slots[program.frag_slots[store.frag]]
        if store_mode != MMA_DAG_STORE_STAGED:
            continue
        if store.region == "loop":
            reason = staged_store_reason(env, store, slot.cols)
        else:
            reason = epilogue_staged_reason(env, store, slot.cols)
        if reason is not None:
            continue
        staged.append(store)
        if store.region == "loop":
            staging_offsets[store.access.node] = fixed_offset + fixed
            fixed += 2 * slot.rows * slot.cols * store.access.dtype.itemsize
    epilogue_staging = max(
        (
            slots[program.frag_slots[store.frag]].rows
            * slots[program.frag_slots[store.frag]].cols
            * store.access.dtype.itemsize
            for store in staged
            if store.region == "epilogue"
        ),
        default=0,
    )
    if epilogue_staging <= fixed_offset:
        epilogue_staging_offset = 0
    else:
        epilogue_staging_offset = fixed_offset + fixed
        fixed += epilogue_staging
    smem = fixed_offset + fixed + 8 * (stages + 3) + 16
    if facts.smem_capacity and smem > facts.smem_capacity:
        raise _decline(
            f"{smem} B of shared memory exceed the capacity {facts.smem_capacity}"
        )
    if column > TCGEN05_MAX_TMEM_COLS:
        raise _decline(f"{column} TMEM columns exceed {TCGEN05_MAX_TMEM_COLS}")
    return Tcgen05Plan(
        warps=warps,
        stages=stages,
        slots=tuple(slots),
        tiles=tiles,
        loop_tiles=tuple(loop_tiles),
        epilogue_tiles=tuple(epilogue_tiles),
        exchange_tiles=tuple(exchange_tiles),
        tma=tuple(tma),
        stage_bytes=stage_bytes,
        tx_bytes=tx_bytes,
        fixed_offset=fixed_offset,
        epilogue_tx_bytes=epilogue_tx_bytes,
        staged_stores=tuple(staged),
        staging_offsets=staging_offsets,
        epilogue_staging_offset=epilogue_staging_offset,
        tmem_cols=tcgen05_tmem_cols(column),
        smem_bytes=fixed_offset + fixed,
    )


def epilogue_staged_reason(
    env: CompileEnvironment, store: Store, bn: int
) -> str | None:
    """Why an epilogue store cannot be written as 16-byte packets from the
    shared staging tile (None: it can): the loop snapshot rules without the
    snapshot-only restriction."""
    from .memory_ops import cute_tensor_base_is_aligned

    if store.swap:
        return "a transposed store is written element-wise"
    access = store.access
    es = access.dtype.itemsize
    if access.tile_dim_strides[1] != 1 or (bn * es) % 16:
        return "the stored rows must be contiguous whole packets"
    if not cute_tensor_base_is_aligned(env, access.fake, 16):
        return "the tensor base is not provably 16-byte aligned"
    for stride in access.strides:
        if stride != 1 and (stride * es) % 16:
            return "a non-contiguous stride is not a multiple of 16 bytes"
    return None


def _axis_dims(access: Access) -> list[tuple[Axis, int]]:
    """``(axis, tensor dim)`` of the access' axis subscripts."""
    result = []
    for dim, subscript in enumerate(access.subscripts):
        if subscript.kind == "axis" and subscript.axis is not None:
            result.append((subscript.axis, dim))
    return result


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def codegen_mma_dag_tcgen05(cg: GenerateAST, plan: MmaDagPlan) -> None:
    """Render the tcgen05 program of the planned root into the current
    statements and register the tensor-map wrapper plan."""
    import ast
    import textwrap

    tc = plan.tcgen05
    assert tc is not None
    df = cg.device_function
    emit_mma_dag_module_statements(cg)
    emitted = getattr(cg, "_helion_mma_dag_tc_module_emitted", False)
    if not emitted:
        cg._helion_mma_dag_tc_module_emitted = True  # type: ignore[attr-defined]
        for stmt in ast.parse(_PREAMBLE).body:
            cg.module_statements.append(stmt)
    emitter = _Tcgen05Emitter(cg, plan)
    source = emitter.render()
    for statement in ast.parse(textwrap.dedent(source)).body:
        cg.add_statement(statement)
    if tc.tma:
        wrapper: dict[str, object] = {"kind": TMA_KIND, "tma_count": len(tc.tma)}
        for spec in tc.tma:
            name = emitter.tensor_names[spec.access.node]
            wrapper[f"tma{spec.index}_name"] = name
            wrapper[f"tma{spec.index}_param"] = spec.param
            wrapper[f"tma{spec.index}_dims"] = list(spec.dims)
            wrapper[f"tma{spec.index}_box"] = list(spec.box)
            wrapper[f"tma{spec.index}_swizzle"] = spec.swizzle
            wrapper[f"tma{spec.index}_elem"] = spec.access.dtype.itemsize
            df.wrapper_only_params.append(spec.param)
            df.wrapper_only_param_annotations[spec.param] = (
                f"cutlass.GridConstant[{_RT}.cuda.TensorMap]"
            )
        cg.cute_wrapper_plans.append(wrapper)
    df.cute_state.owned_root_block_dims = (plan.threads, 1, 1)


class _PassEvaluator(_Evaluator):
    """The warp evaluator with the operand passes' rounding folded into the
    pack: the pointwise lowering casts a 16-bit node's result itself
    (``cutlass.BFloat16(v)``); for the pass value's root that cast is dropped
    and the fp32 value goes to ``pack_bf16x2`` / ``pack_f16x2``, whose
    round-to-nearest-even is the same rounding (one ``cvt`` per pair instead
    of two scalar ones per element)."""

    def __init__(self, emitter: _Emitter) -> None:
        super().__init__(emitter)
        self.round_root: Node | None = None

    def _load(self, node: Node, coords: Coords) -> str:
        em = cast("_Tcgen05Emitter", self.emitter)
        access = em.accesses[node]
        if len(access.tile_dims) == 2:
            return super()._load(node, coords)
        key = (node, coords)
        register = em.prefetched.get(key)
        if register is not None:
            return em.bind("md_v", em.typed(register, access.dtype))
        if em.recording is not None:
            em.recording.append(key)
        return super()._load(node, coords)

    def _lowered(self, node: Node, coords: Coords) -> str:
        import ast

        from torch._inductor.virtualized import V
        from torch.fx.node import map_arg

        from ..ast_extension import expr_from_string
        from ..aten_lowering import AtenLowering
        from ..aten_lowering import LoweringContext
        from ..inductor_lowering import PointwiseLowering

        lowering = node.meta.get("lowering")
        em = self.emitter
        shape = _shape(node)
        env: dict[Node, object] = {}

        def visit(value: Node) -> Node:
            if value not in env:
                env[value] = expr_from_string(
                    self.evaluate(value, _broadcast(coords, shape, _shape(value)))
                )
            return value

        map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = em.cg
        ctx.env = env  # pyrefly: ignore[bad-assignment]
        statements: list[ast.AST] = []
        with em.cg.set_statements(statements), V.set_current_node(node):
            if isinstance(lowering, PointwiseLowering):
                result = lowering.codegen_from_input_asts(
                    ctx, node, lowering.input_asts(ctx, node)
                )
            elif isinstance(lowering, AtenLowering):
                result = lowering.codegen(ctx, node)
            else:
                raise _decline(f"{node.name} is not a pointwise value")
        if not isinstance(result, ast.AST):
            raise _decline(f"invalid pointwise result for {node.name}")
        if node is self.round_root:
            val = node.meta.get("val")
            dtype = val.dtype if isinstance(val, torch.Tensor) else None
            if dtype is not None and dtype in (torch.bfloat16, torch.float16):
                cast_name = em.env.backend.dtype_str(dtype)
                # The cast is the result expression, or the last statement
                # the result names.
                if (
                    isinstance(result, ast.Name)
                    and statements
                    and isinstance(statements[-1], ast.Assign)
                    and len(statements[-1].targets) == 1
                    and isinstance(statements[-1].targets[0], ast.Name)
                    and statements[-1].targets[0].id == result.id
                ):
                    last = statements[-1].value
                    if (
                        isinstance(last, ast.Call)
                        and len(last.args) == 1
                        and not last.keywords
                        and ast.unparse(last.func) == cast_name
                    ):
                        statements.pop()
                        result = last.args[0]
                if (
                    isinstance(result, ast.Call)
                    and len(result.args) == 1
                    and not result.keywords
                    and ast.unparse(result.func) == cast_name
                ):
                    result = result.args[0]
        for statement in statements:
            for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines():
                em.emit(line)
        return em.bind("md_v", ast.unparse(result))


class _Tcgen05Emitter(_Emitter):
    """The tcgen05 renderer: the warp emitter's evaluator, addressing and
    pointwise machinery over TMEM-resident fragments (one row per epilogue
    thread, 64 columns per load) and UMMA-swizzled shared tiles."""

    def __init__(self, cg: GenerateAST, plan: MmaDagPlan) -> None:
        # The warp emitter's constructor builds mma.sync warp grids and the
        # ldmatrix tile map; this mode has neither, so set up the shared
        # state by hand.
        tc = plan.tcgen05
        assert tc is not None
        self.cg = cg
        self.df = cg.device_function
        self.env = CompileEnvironment.current()
        self.plan = plan
        self.tc = tc
        self.program = plan.program
        self.lines = []
        self.indent = ""
        self.accesses = {access.node: access for access in self.program.accesses}
        self.base_vars = {}
        self.iter_vars = {}
        self.deferred = []
        self._defined_tiles = set()
        self.evaluator: _Evaluator = _PassEvaluator(self)
        self.es = self.program.elem_bytes
        self.kind = "bf16" if self.program.dtype == torch.bfloat16 else "f16"
        self.sync = "cute.arch.barrier()"
        self.loop_var = "md_i"
        self.tensor_names: dict[Node, str] = {}
        # Name prefix of the fragment row vectors (``md_v{slot}_{chunk}``).
        self.vec = "md_v"
        # Side loads (decay scalars / rows) loaded one iteration ahead into
        # loop-carried registers: (load node, coords) -> register, and the
        # per-thread row names of the passes with their expressions / guards.
        self.prefetched: dict[tuple[Node, Coords], str] = {}
        self.recording: list[tuple[Node, Coords]] | None = None
        self.row_exprs: dict[str, tuple[str, str | None]] = {}
        # Epilogue commits issued so far (eacc phases).
        self.epilogue_commits = 0
        self.smem_bytes = tc.smem_bytes
        self.dot_index = {dot.node: i for i, dot in enumerate(self.program.dots)}

    # -- fragment element model -------------------------------------------------

    def slot(self, index: int) -> SlotGeometry:
        return self.tc.slots[index]

    def row_var(self, slot: int) -> str:
        return f"md_row{self.slot(slot).rows}"

    def half_rows(self, slot: int) -> bool:
        return self.slot(slot).rows == 64

    @contextlib.contextmanager
    def block(self, header: str | None) -> Iterator[None]:
        """A dynamic block (``header``, or none) with its own evaluator
        scope: values rendered inside a staged ``if`` are not visible after
        it, so nothing evaluated there may be reused outside."""
        self.evaluator.push()
        if header is None:
            yield
        else:
            with self.nest(header):
                yield
        self.evaluator.pop()

    def row_guard(self, slot: int) -> contextlib.AbstractContextManager[None]:
        """The epilogue-warp lanes that own a row of ``slot`` (lanes 16-31
        duplicate rows of a 64-row fragment and never store)."""
        return self.block("if md_rowok:" if self.half_rows(slot) else None)

    def reg(self, slot: int, *key: object) -> str:  # type: ignore[override]
        chunk, j = key
        return f"{self.vec}{slot}_{chunk}[{j}]"

    def element_coords(
        self, node: Node, slot: int, col: int, swap: bool = False
    ) -> Coords:
        """Positional coordinates of ``node`` for the element of fragment
        ``slot`` at this thread's row and column ``col``."""
        shape = _shape(node)
        positions = [
            i
            for i, size in enumerate(shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(node, i)
        ]
        if len(positions) != 2:
            raise _decline(f"{node.name} is not a two-dim fragment value")
        coords = ["0"] * len(shape)
        row = self.row_var(slot)
        first, second = (str(col), row) if swap else (row, str(col))
        coords[positions[0]] = first
        coords[positions[1]] = second
        return tuple(coords)

    def rescales(self, dot: Dot) -> bool:
        """The dot accumulates onto a pointwise DAG over a fragment (not the
        fragment itself)."""
        acc = dot.acc
        return acc is not None and strip_new_vars(acc) is not dot.acc_frag

    def prepares(self, dot: Dot) -> bool:
        """The dot's accumulator slot must be written before the issue: a
        rescale, or a fragment accumulated into another slot."""
        return dot.acc_frag is not None and (
            self.rescales(dot) or dot.slot != dot.acc_slot
        )

    # -- shared tile addressing -----------------------------------------------

    def ring_base(self, stage: str) -> str:
        return f"md_s0 + ({stage}) * {self.tc.stage_bytes}"

    def tile_base(self, tile: Tcgen05Tile, stage: str) -> str:
        """Byte base of the tile: ring slot ``stage`` for loop tiles, the
        fixed region otherwise."""
        if tile.region == REGION_LOOP:
            return f"{self.ring_base(stage)} + {tile.offset}"
        return f"md_s0 + {self.tc.fixed_offset + tile.offset}"

    def tile_address(self, tile: Tcgen05Tile, stage: str, prow: str, pch: str) -> str:
        """Byte address of 16-byte chunk ``pch`` of row ``prow`` of the tile
        (UMMA layout: whole swizzle atoms side by side as slabs, the atom's
        XOR swizzle inside)."""
        cpa = tile.chunks_per_atom
        if tile.slabs == 1:
            slab = "0"
            local = pch
        else:
            slab = f"({pch} // {cpa})"
            local = f"({pch} % {cpa})"
        swz = _swizzled_chunk(prow, local, cpa)
        terms = [
            self.tile_base(tile, stage),
            f"{prow} * {tile.atom_bytes}",
            f"{swz} * 16",
        ]
        if slab != "0":
            terms.append(f"{slab} * {tile.slab_bytes}")
        return " + ".join(terms)

    def raw_address(self, tile: Tcgen05Tile, stage: str, prow: str, pch: str) -> str:
        if tile.region == REGION_LOOP:
            base = f"{self.ring_base(stage)} + {tile.raw_offset}"
        else:
            base = f"md_s0 + {self.tc.fixed_offset + tile.raw_offset}"
        return f"{base} + {prow} * {tile.raw_row_bytes} + {pch} * 16"

    def desc(self, name: str, tile: Tcgen05Tile, stage: str, k_major: bool) -> None:
        self.emit(
            f"{name} = {_PRIMS}.Tcgen05SmemDesc.build({self.tile_base(tile, stage)}, "
            f"leading_byte_offset={tile.leading_byte_offset(k_major)}, "
            f"stride_byte_offset={tile.stride_byte_offset}, "
            f"layout={_PRIMS}.Tcgen05SmemSwizzle.{_DESC_SWIZZLE[tile.swizzle]})"
        )

    def packets(
        self,
        count: int,
        threads: int,
        idx_base: str,
        cpr: int,
        body: object,
        tag: str | None = None,
    ) -> None:
        """``body(prow, pch)`` for every 16-byte packet this thread owns of a
        ``count``-packet tile, round-robin over ``threads`` threads.  With a
        ``tag`` the row / chunk names are fixed per repetition (the passes:
        their side loads are then addressable one iteration ahead)."""
        for rep in range(-(-count // threads)):
            idx = f"({rep * threads} + {idx_base})" if rep else idx_base
            guard = f"{idx} < {count}" if count < (rep + 1) * threads else None
            nest = self.nest(f"if {guard}:") if guard else None
            if nest is not None:
                nest.__enter__()
            self.evaluator.push()
            if tag is None:
                prow = self.bind("md_prow", f"{idx} // {cpr}")
                pch = self.bind("md_pch", f"{idx} % {cpr}")
            else:
                prow, pch = f"md_pr_{tag}_{rep}", f"md_pc_{tag}_{rep}"
                self.emit(f"{prow} = {idx} // {cpr}")
                self.emit(f"{pch} = {idx} % {cpr}")
                self.row_exprs[prow] = (f"({idx} // {cpr})", guard)
            body(prow, pch)  # type: ignore[operator]
            self.evaluator.pop()
            if nest is not None:
                nest.__exit__(None, None, None)

    # -- TMA ------------------------------------------------------------------------

    def tma_coords(self, spec: TmaSpec, loop_value: str, slab: int) -> list[str]:
        """TMA coordinates (innermost first) of slab ``slab`` of the tile
        of iteration ``loop_value``."""
        access = spec.access
        loop_axis = self.program.loop_axis
        coords = []
        for position, dim in enumerate(spec.dims):
            subscript = access.subscripts[dim]
            if subscript.kind == "scalar":
                assert subscript.expr is not None
                value = self.uniform(subscript.expr, loop_value)
            else:
                axis = subscript.axis
                assert axis is not None
                if axis.kind == AXIS_RDIM:
                    value = "0"
                elif axis is loop_axis:
                    value = f"({loop_value}) * {self.plan.size(axis)}"
                else:
                    value = self.axis_base(axis)
            if position == spec.slab_dim and slab:
                value = f"{value} + {slab * spec.slab_elems}"
            coords.append(f"cutlass.Int32({value})")
        return coords

    def emit_tma_issue(self, loop_value: str, stage: str) -> None:
        """One elected thread: expect the ring slot's bytes and issue every
        box of iteration ``loop_value`` into ring slot ``stage``."""
        tc = self.tc
        specs = [t.tma or t.raw_tma for t in tc.loop_tiles]
        self.emit(
            f"{_PRIMS}.mbarrier_arrive_expect_tx(md_full + ({stage}), {tc.tx_bytes})"
        )
        for spec in specs:
            if spec is None:
                continue
            for slab in range(spec.slabs):
                dst = (
                    f"md_smem + (({stage}) * {tc.stage_bytes} + "
                    f"{spec.offset + slab * spec.slab_bytes})"
                )
                coords = ", ".join(self.tma_coords(spec, loop_value, slab))
                self.emit(
                    f"{_PRIMS}.cp_async_bulk_tensor_shared_cta_global({dst}, "
                    f"{spec.param}.get_ptr(), [{coords}], md_full + ({stage}))"
                )

    def emit_epilogue_tma_issue(self) -> None:
        """One elected thread: the epilogue operand tiles, on ``efull``."""
        tc = self.tc
        self.emit(
            f"{_PRIMS}.mbarrier_arrive_expect_tx(md_efull, {tc.epilogue_tx_bytes})"
        )
        for tile in tc.epilogue_tiles:
            spec = tile.tma or tile.raw_tma
            if spec is None:
                continue
            for slab in range(spec.slabs):
                dst = f"md_smem + {tc.fixed_offset + spec.offset + slab * spec.slab_bytes}"
                coords = ", ".join(self.tma_coords(spec, self.loop_var, slab))
                self.emit(
                    f"{_PRIMS}.cp_async_bulk_tensor_shared_cta_global({dst}, "
                    f"{spec.param}.get_ptr(), [{coords}], md_efull)"
                )

    # -- operand passes -----------------------------------------------------------

    def fold_root(self, value: Node) -> None:
        """Fold the rounding of ``value``'s pointwise root (below its views)
        into the pack that consumes the evaluated elements.

        The passes and the exchange / store packing keep 16-bit elements as
        the fp32 values they exactly are and let the final ``pack`` round
        once: the chain's trailing cast to the 16-bit dtype is the same
        round-to-nearest-even, so it is dropped, and every op in between
        computes in fp32 on fp32 inputs (the DSL promotes a 16-bit operand of
        a mixed op the same way).  This spares the scalar ``cvt`` round
        trips the re-typed leaves cost (the XU pipe serialized the chain).
        """
        from ...language import view_ops
        from .cute_mma_dag import _is_swap_permute

        root = strip_new_vars(value)
        while _is_swap_permute(root) or (
            root.op == "call_function" and root.target is view_ops.subscript
        ):
            root = strip_new_vars(cast("Node", root.args[0]))
        assert isinstance(self.evaluator, _PassEvaluator)
        self.evaluator.round_root = root

    def pass_value(self, operand: Operand) -> Node:
        self.fold_root(operand.value)
        return operand.value

    def emit_sync_pass(  # type: ignore[override]
        self, tile: Tcgen05Tile, stage: str, idx_base: str, threads: int
    ) -> None:
        """Cooperative pass for a source TMA cannot fetch: element pairs
        straight from global memory through the pointwise DAG into the
        swizzled operand tile."""
        operand = tile.operand
        load = operand.load
        assert load is not None
        es = tile.elem_bytes
        pairs = tile.rows * tile.cols // 2
        cpp = tile.cols // 2
        paired = self.pairs_aligned(load, operand.cols_pos)
        load_es = load.dtype.itemsize
        row_stride = load.tile_dim_strides[operand.rows_pos] * load_es
        value = self.pass_value(operand)
        self.hoist_scalars(value)
        tag = tile.tag
        for rep in range(-(-pairs // threads)):
            idx = f"({rep * threads} + {idx_base})" if rep else idx_base
            guard = f"{idx} < {pairs}" if pairs < (rep + 1) * threads else None
            nest = self.nest(f"if {guard}:") if guard else None
            if nest is not None:
                nest.__enter__()
            self.evaluator.push()
            prow, pcol = f"md_spr_{tag}_{rep}", f"md_spc_{tag}_{rep}"
            self.emit(f"{prow} = {idx} // {cpp}")
            self.emit(f"{pcol} = {idx} % {cpp}")
            self.row_exprs[prow] = (f"({idx} // {cpp})", guard)
            address = (
                f"{self.iter_vars[load.node]} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pcol}) * {2 * load_es}"
            )
            lo, hi = self.pair_load(address, load.dtype, paired)
            self.evaluator.tile_leaves[(load.node, 0)] = lo
            self.evaluator.tile_leaves[(load.node, 1)] = hi
            values = []
            for half in range(2):
                self.evaluator.element = half
                col = f"(2 * {pcol} + {half})" if half else f"(2 * {pcol})"
                values.append(
                    self.evaluator.evaluate(value, self.tile_coords(operand, prow, col))
                )
            self.evaluator.element = None
            per_chunk = 16 // es
            chunk = f"({pcol} // {per_chunk // 2})"
            within = f"({pcol} % {per_chunk // 2}) * {2 * es}"
            self.emit(
                f"{_RUNTIME}.sts_b32({self.tile_address(tile, stage, prow, chunk)} + {within}, "
                f"{self.pack(values[0], values[1])})"
            )
            self.evaluator.pop()
            if nest is not None:
                nest.__exit__(None, None, None)

    def emit_inplace_pass(  # type: ignore[override]
        self, tile: Tcgen05Tile, stage: str, idx_base: str, threads: int
    ) -> None:
        """Rewrite the landed tile's packets through the pointwise DAG in
        place (every thread its round-robin share)."""
        operand = tile.operand
        load = operand.load
        assert load is not None
        unpack = "unpack_f16x2" if self.kind == "f16" else "unpack_bf16x2"
        value = self.pass_value(operand)
        self.hoist_scalars(value)

        def body(prow: str, pch: str) -> None:
            address = self.bind("md_pa", self.tile_address(tile, stage, prow, pch))
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({address})")
            for q, word in enumerate(words):
                lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                self.evaluator.tile_leaves[(load.node, 2 * q)] = lo
                self.evaluator.tile_leaves[(load.node, 2 * q + 1)] = hi
            packed = []
            for q in range(4):
                values = []
                for e in (2 * q, 2 * q + 1):
                    self.evaluator.element = e
                    coords = self.tile_coords(operand, prow, f"({pch} * 8 + {e})")
                    values.append(self.evaluator.evaluate(value, coords))
                self.evaluator.element = None
                packed.append(self.pack(values[0], values[1]))
            self.emit(f"{_RUNTIME}.sts_v4_b32({address}, {', '.join(packed)})")

        self.packets(
            tile.rows * tile.chunks_per_row,
            threads,
            idx_base,
            tile.chunks_per_row,
            body,
            tag=tile.tag,
        )

    def emit_convert_pass(  # type: ignore[override]
        self, tile: Tcgen05Tile, stage: str, idx_base: str, threads: int
    ) -> None:
        """Rewrite the raw fp32 tile (linear rows) through the pointwise DAG
        into the 16-bit swizzled operand tile."""
        operand = tile.operand
        load = operand.load
        assert load is not None
        value = self.pass_value(operand)
        self.hoist_scalars(value)
        raw_cpr = tile.raw_row_bytes // 16
        es = tile.elem_bytes

        def body(prow: str, pch: str) -> None:
            raws = [self.df.new_var("md_f") for _ in range(4)]
            self.emit(
                f"{', '.join(raws)} = {_RT}.lds_v4_f32({self.raw_address(tile, stage, prow, pch)})"
            )
            for e, raw in enumerate(raws):
                self.evaluator.tile_leaves[(load.node, e)] = raw
            packed = []
            for q in range(2):
                pair = []
                for e in (2 * q, 2 * q + 1):
                    self.evaluator.element = e
                    coords = self.tile_coords(operand, prow, f"({pch} * 4 + {e})")
                    pair.append(self.evaluator.evaluate(value, coords))
                self.evaluator.element = None
                packed.append(self.pack(pair[0], pair[1]))
            # Raw packet p holds columns 4p..4p+3: 16-bit chunk p // 2, byte
            # offset 8 * (p % 2) within it.
            chunk = f"({pch} // 2)"
            self.emit(
                f"{_RUNTIME}.sts_v2_b32({self.tile_address(tile, stage, prow, chunk)} "
                f"+ ({pch} % 2) * {4 * es}, {', '.join(packed)})"
            )

        self.packets(
            tile.rows * raw_cpr, threads, idx_base, raw_cpr, body, tag=tile.tag
        )

    def emit_passes(
        self,
        tiles: Sequence[Tcgen05Tile],
        stage: str,
        wait: str | None,
        idx_base: str,
        threads: int,
    ) -> None:
        """Every pass of ``tiles`` by the ``threads`` threads with local
        index ``idx_base``: ``wait`` (the TMA of in-place rewrites), the
        passes, then publish the generic writes to the async proxy."""
        if wait is not None and any(
            tile.route in (ROUTE_INPLACE, ROUTE_CONVERT) for tile in tiles
        ):
            self.emit(wait)
        for tile in tiles:
            if tile.route == ROUTE_SYNC:
                self.emit_sync_pass(tile, stage, idx_base, threads)
            elif tile.route == ROUTE_INPLACE:
                self.emit_inplace_pass(tile, stage, idx_base, threads)
            elif tile.route == ROUTE_CONVERT:
                self.emit_convert_pass(tile, stage, idx_base, threads)
        self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')

    def has_passes(self, tiles: Sequence[Tcgen05Tile]) -> bool:
        return any(
            tile.route in (ROUTE_SYNC, ROUTE_INPLACE, ROUTE_CONVERT) for tile in tiles
        )

    # -- side loads one iteration ahead ------------------------------------------

    def loop_side_work(self) -> None:
        """The per-iteration work whose side loads are prefetched: the loop
        operand passes and the carry rescales."""
        tc = self.tc
        if self.has_passes(tc.loop_tiles):
            self.emit_passes(
                tc.loop_tiles,
                "md_stage",
                f"{_RT}.mbar_wait(md_full + (md_stage), md_round)",
                "md_tidx",
                self.plan.threads,
            )
        loaded: set[int] = set()
        for dot in self.program.dots_in("loop"):
            if self.prepares(dot):
                self.emit_rescale(dot, loaded)

    def discover_prefetch_sites(self) -> list[tuple[Node, Coords]]:
        """Render the loop side work once into scratch, recording every side
        load whose coordinates are fixed per thread (the thread's fragment
        row, a pass packet row, or none): those are loaded one iteration
        ahead."""
        saved = (self.lines, self.indent, self.iter_vars, self.evaluator, self.deferred)
        self.lines, self.indent, self.deferred = [], "", []
        self.define_iteration_bases(in_loop=True)
        self.evaluator = _PassEvaluator(self)
        self.recording = []
        self.loop_side_work()
        rows = {f"md_row{m}" for m in TCGEN05_M_CHOICES}
        sites: list[tuple[Node, Coords]] = []
        for key in self.recording:
            node, coords = key
            if key in sites or not self.loop_dependent(self.accesses[node]):
                continue
            if all(c == "0" or c in rows or c in self.row_exprs for c in coords):
                sites.append(key)
        self.recording = None
        self.lines, self.indent, self.iter_vars, self.evaluator, self.deferred = saved
        return sites[:TCGEN05_MAX_PREFETCH]

    def emit_line_prefetch(self, loop_value: str) -> None:
        """``prefetch.global.L2`` of every line the prefetched side loads of
        iteration ``loop_value`` touch (thread ``t`` takes line ``t``): the
        rows are first touched from DRAM, whose latency exceeds the one
        iteration the register prefetch runs ahead."""
        lines: list[str] = []
        seen: set[Node] = set()
        for node, _coords in self.prefetched:
            if node in seen:
                continue
            seen.add(node)
            access = self.accesses[node]
            base = self.iteration_base(access, loop_value)
            span = access.dtype.itemsize
            for position in access.tile_positions:
                span = max(
                    span,
                    self.plan.size(access.dims[position])
                    * access.stride_at(position)
                    * access.dtype.itemsize,
                )
            for offset in range(0, span, 128):
                lines.append(f"{base} + {offset}" if offset else base)
        for index, line in enumerate(lines):
            with self.nest(f"if md_tidx == {index}:"):
                self.emit(
                    f"{_PRIMS}.prefetch_l2(cutlass.inttoptr({line}, 1, cutlass.Uint8))"
                )

    def emit_prefetch_loads(self, loop_value: str) -> None:
        """Load every prefetch site of iteration ``loop_value`` into its
        register (guarded like the packet it belongs to)."""
        saved = self.iter_vars
        self.iter_vars = {}
        for node, _coords in self.prefetched:
            if node not in self.iter_vars:
                self.iter_vars[node] = self.iteration_base(
                    self.accesses[node], loop_value
                )
        for (node, coords), register in self.prefetched.items():
            access = self.accesses[node]
            guards = []
            resolved = []
            for c in coords:
                if c in self.row_exprs:
                    expr, guard = self.row_exprs[c]
                    resolved.append(expr)
                    if guard is not None and guard not in guards:
                        guards.append(guard)
                else:
                    resolved.append(c)
            load = self.scalar_load(self.address(access, tuple(resolved)), access.dtype)
            with self.nest(f"if {' and '.join(guards)}:") if guards else _NoNest():
                self.emit(f"{register} = {load}")
        self.iter_vars = saved

    # -- fragments in TMEM ---------------------------------------------------------

    def tmem_ptr(self, slot: int, col: int) -> str:
        column = self.slot(slot).column + col
        return (
            f"cutlass.inttoptr(((md_warp % 4) * 32 << 16) | (md_tmem + {column}), "
            f"{_RT}.TMEM_SPACE, cutlass.Float32)"
        )

    def emit_ld(self, slot: int) -> None:
        """Load this thread's row of fragment ``slot`` into
        ``md_v{slot}_{c}`` (64 columns each)."""
        geometry = self.slot(slot)
        for chunk in range(geometry.ld_chunks):
            self.emit(
                f'{self.vec}{slot}_{chunk} = {_PRIMS}.tcgen05_ld("32x32b", '
                f"{self.tmem_ptr(slot, chunk * TCGEN05_LD_COLS)}, "
                f"num={geometry.chunk_cols(chunk)})"
            )
        self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.LOAD)")

    def emit_st_array(self, name: str, slot: int, chunk: int) -> None:
        self.emit(
            f'{_PRIMS}.tcgen05_st("32x32b", {self.tmem_ptr(slot, chunk * TCGEN05_LD_COLS)}, '
            f"{name}[0:{self.slot(slot).chunk_cols(chunk)}])"
        )

    def emit_carry_init(self) -> None:  # type: ignore[override]
        """Seed the TMEM carries: zeros, or this thread's row of the init tile."""
        with self.block("if md_epi:"):
            for carry in self.program.carries:
                slot = carry.slot
                geometry = self.slot(slot)
                init = carry.init
                positions: list[int] = []
                paired = False
                if init is not None:
                    self.iter_vars[init.node] = self.base_vars[init.node]
                    positions = list(init.tile_positions)
                    paired = self.pairs_aligned(init, 1)
                for chunk in range(geometry.ld_chunks):
                    cols = geometry.chunk_cols(chunk)
                    name = self.bind(
                        "md_seed",
                        f"cutlass.Array(cutlass.Float32, {cols}, alignment=16)",
                    )
                    if init is None:
                        for j in range(cols):
                            self.emit(f"{name}[{j}] = cutlass.Float32(0.0)")
                    else:
                        with self.row_guard(slot):
                            for j in range(0, cols, 2):
                                col = chunk * TCGEN05_LD_COLS + j
                                coords = ["0"] * len(init.dims)
                                coords[positions[0]] = self.row_var(slot)
                                coords[positions[1]] = str(col)
                                address = self.address(init, tuple(coords))
                                lo, hi = self.pair_load(address, init.dtype, paired)
                                self.emit(f"{name}[{j}] = cutlass.Float32({lo})")
                                self.emit(f"{name}[{j + 1}] = cutlass.Float32({hi})")
                    self.emit_st_array(name, slot, chunk)
            self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")

    def emit_rescale(self, dot: Dot, loaded: set[int]) -> None:
        """``dot.acc`` is a pointwise DAG over fragment ``acc_slot`` (or that
        fragment headed for another slot): evaluate it per element of the
        loaded row (``loaded`` names the slots already in registers) and
        store the result into the dot's slot."""
        acc = dot.acc
        assert acc is not None and dot.acc_slot is not None
        source, target = dot.acc_slot, dot.slot
        geometry = self.slot(target)
        self.hoist_scalars(acc)
        if source not in loaded:
            self.emit_ld(source)
            loaded.add(source)
        for chunk in range(geometry.ld_chunks):
            cols = geometry.chunk_cols(chunk)
            name = self.bind(
                "md_rs", f"cutlass.Array(cutlass.Float32, {cols}, alignment=16)"
            )
            for j in range(cols):
                self.evaluator.reg = (chunk, j)  # type: ignore[assignment]
                coords = self.element_coords(acc, source, chunk * TCGEN05_LD_COLS + j)
                value = self.evaluator.evaluate(acc, coords)
                self.emit(f"{name}[{j}] = cutlass.Float32({value})")
            self.evaluator.reg = None
            self.emit_st_array(name, target, chunk)
        self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")

    # -- stores ---------------------------------------------------------------------

    def staging_base(self, store: Store, parity: str) -> str:
        slot = self.slot(self.program.frag_slots[store.frag])
        if store.region == "loop":
            tile = slot.rows * slot.cols * store.access.dtype.itemsize
            return (
                f"md_s0 + {self.tc.staging_offsets[store.access.node]} "
                f"+ ({parity}) * {tile}"
            )
        return f"md_s0 + {self.tc.epilogue_staging_offset}"

    def emit_stage_in(  # type: ignore[override]
        self, store: Store, parity: str, evaluate: bool
    ) -> None:
        """This thread's fragment row as 16-byte packets of the store dtype
        into the staging buffer (chunk ``c`` of row ``r`` at ``c ^ (r % 8)``).
        ``evaluate`` renders the store's pointwise chain per element (its
        trailing cast folded into the pack); otherwise the value is the
        fragment itself or its cast and the registers are packed directly."""
        slot = self.program.frag_slots[store.frag]
        geometry = self.slot(slot)
        dtype = store.access.dtype
        es = dtype.itemsize
        row_bytes = geometry.cols * es
        cpr = row_bytes // 16
        per_packet = 16 // es
        base = self.staging_base(store, parity)
        row = self.row_var(slot)
        row_base = self.bind("md_srow", f"{base} + {row} * {row_bytes}")
        pack = "pack_f16x2" if dtype == torch.float16 else "pack_bf16x2"
        if evaluate:
            self.fold_root(store.value)
            self.hoist_scalars(store.value)
        for col in range(0, geometry.cols, per_packet):
            chunk = col // per_packet
            swz = _swizzled_chunk(row, str(chunk), cpr)
            address = f"{row_base} + {swz} * 16"
            values = []
            for c in range(col, col + per_packet):
                chunk_index, j = c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS
                if evaluate:
                    self.evaluator.reg = (chunk_index, j)  # type: ignore[assignment]
                    coords = self.element_coords(store.value, slot, c)
                    values.append(self.evaluator.evaluate(store.value, coords))
                else:
                    values.append(self.reg(slot, chunk_index, j))
            self.evaluator.reg = None
            if dtype == torch.float32:
                self.emit(f"{_RT}.sts_v4_f32({address}, {', '.join(values)})")
            else:
                words = [
                    f"{_RUNTIME}.{pack}({values[2 * q]}, {values[2 * q + 1]})"
                    for q in range(4)
                ]
                self.emit(f"{_RUNTIME}.sts_v4_b32({address}, {', '.join(words)})")

    def emit_stage_out(  # type: ignore[override]
        self, store: Store, parity: str, loop_value: str, idx: str, threads: int
    ) -> None:
        """Staging buffer -> global memory as 16-byte packets, round-robin
        over the ``threads`` threads whose local index is ``idx``."""
        slot = self.program.frag_slots[store.frag]
        geometry = self.slot(slot)
        access = store.access
        es = access.dtype.itemsize
        row_bytes = geometry.cols * es
        cpr = row_bytes // 16
        row_stride = access.tile_dim_strides[0] * es
        gbase = self.iteration_base(access, loop_value)
        base = self.staging_base(store, parity)

        def body(prow: str, pch: str) -> None:
            swz = _swizzled_chunk(prow, pch, cpr)
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({base} + {prow} * {row_bytes} + {swz} * 16)"
            )
            self.emit(
                f"{_RUNTIME}.stg_v4_b32({gbase} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pch} * 16), {', '.join(words)})"
            )

        self.packets(geometry.rows * cpr, threads, idx, cpr, body)

    def emit_direct_store(self, store: Store) -> None:
        """Pair stores of the store value straight from the fragment row."""
        slot = self.program.frag_slots[store.frag]
        geometry = self.slot(slot)
        access = store.access
        positions = access.tile_positions
        paired = self.pairs_aligned(access, 1) and not store.swap
        self.hoist_scalars(store.value)
        row = self.row_var(slot)
        for col in range(0, geometry.cols, 2):
            values = []
            for bit in range(2):
                c = col + bit
                self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                coords = self.element_coords(store.value, slot, c, store.swap)
                values.append(self.evaluator.evaluate(store.value, coords))
            self.evaluator.reg = None
            coords = ["0"] * len(access.dims)
            first, second = (str(col), row) if store.swap else (row, str(col))
            coords[positions[0]] = first
            coords[positions[1]] = second
            if paired:
                self.pair_store(
                    self.address(access, tuple(coords)),
                    values[0],
                    values[1],
                    access.dtype,
                    True,
                )
            else:
                for bit in range(2):
                    element = list(coords)
                    element[positions[1 if not store.swap else 0]] = str(col + bit)
                    scalar = {
                        torch.float32: "stg_f32",
                        torch.float16: "stg_f16",
                        torch.bfloat16: "stg_bf16",
                    }[access.dtype]
                    self.emit(
                        f"{_RUNTIME}.{scalar}({self.address(access, tuple(element))}, {values[bit]})"
                    )

    # -- MMA issue ------------------------------------------------------------------

    def emit_dot_mmas(self, dot: Dot, stage: str, accumulate: bool) -> None:
        """Warp 0: the K / 16 ``tcgen05.mma`` of ``dot`` over its tiles."""
        plan = self.plan
        a_tile, b_tile = self.tc.tile(dot, dot.a), self.tc.tile(dot, dot.b)
        a_major, b_major = operand_k_major(dot.a), operand_k_major(dot.b)
        index = self.dot_index[dot.node]
        self.desc(f"md_da{index}", a_tile, stage, a_major)
        self.desc(f"md_db{index}", b_tile, stage, b_major)
        k = plan.size(dot.k)
        for step in range(k // MMA_DAG_K_STEP):
            with self.nest(f"if {_PRIMS}.elect_sync():"):
                enable = "True" if accumulate or step else "False"
                self.emit(
                    f"{_PRIMS}.tcgen05_mma({_PRIMS}.Tcgen05MMAKind.F16, {_PRIMS}.CTAGroup.CTA_1, "
                    f"md_acc{dot.slot}, "
                    f"md_da{index}.advance_start_address({a_tile.k_step_offset(step, a_major)}), "
                    f"md_db{index}.advance_start_address({b_tile.k_step_offset(step, b_major)}), "
                    f"md_idesc{index}, {enable})"
                )

    # -- the body -----------------------------------------------------------------

    def render(self) -> str:  # type: ignore[override]
        tc, plan, program = self.tc, self.plan, self.program
        self.dot_index = {dot.node: i for i, dot in enumerate(program.dots)}
        self.emit("md_tidx, _, _ = cute.arch.thread_idx()")
        self.emit("md_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
        self.emit("md_lane = md_tidx % 32")
        self.emit(f"md_epi = md_warp < {TCGEN05_EPILOGUE_WARPS}")
        heights = {slot.rows for slot in tc.slots}
        if 64 in heights:
            # A 64-row fragment fills TMEM lanes 0-15 of each quadrant: lanes
            # 16-31 compute on a duplicate row (in-bounds addresses) and
            # never store.
            self.emit("md_row64 = (md_warp % 4) * 16 + md_lane % 16")
            self.emit("md_rowok = md_lane < 16")
        if 128 in heights:
            self.emit("md_row128 = (md_warp % 4) * 32 + md_lane")
        self.define_axis_bases()
        self.emit(
            f"md_smem = cute.arch.alloc_smem(cutlass.Uint8, {self.smem_bytes}, alignment={SLAB_ALIGNMENT})"
        )
        self.emit("md_s0 = cutlass.Int32(md_smem.toint())")
        self.emit(
            f"md_full = cute.arch.alloc_smem(cutlass.Int64, {tc.stages}, alignment=8)"
        )
        self.emit("md_accb = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)")
        self.emit("md_efull = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)")
        self.emit("md_eacc = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)")
        self.emit("md_tslot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)")
        with self.nest("if md_tidx == 0:"):
            for stage in range(tc.stages):
                self.emit(f"{_PRIMS}.mbarrier_init(md_full + {stage}, 1)")
            for name in ("md_accb", "md_efull", "md_eacc"):
                self.emit(f"{_PRIMS}.mbarrier_init({name}, 1)")
            self.emit(f"{_PRIMS}.fence_mbarrier_init()")
            self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
        with self.nest("if md_warp == 0:"):
            self.emit(
                f"{_PRIMS}.tcgen05_alloc(md_tslot, {tc.tmem_cols}, group={_PRIMS}.CTAGroup.CTA_1)"
            )
            self.emit(
                f"{_PRIMS}.tcgen05_relinquish_alloc_permit(group={_PRIMS}.CTAGroup.CTA_1)"
            )
        self.emit(self.sync)
        self.emit("md_tmem = cute.make_tensor(md_tslot, cute.make_layout((1,)))[0]")
        for index, slot in enumerate(tc.slots):
            self.emit(
                f"md_acc{index} = cutlass.inttoptr(md_tmem + {slot.column}, "
                f"{_RT}.TMEM_SPACE, cutlass.Float32)"
            )
        self.define_bases()
        for access in program.accesses:
            self.tensor_names[access.node] = self.df.tensor_arg(
                access.fake, prefer_name=access.name
            ).name
        dtype = "cutlass.BFloat16" if self.kind == "bf16" else "cutlass.Float16"
        for dot in program.dots:
            index = self.dot_index[dot.node]
            self.emit(
                f"md_idesc{index} = {_PRIMS}.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, "
                f"a_dtype={dtype}, b_dtype={dtype}, n_dim={plan.size(dot.cols)}, "
                f"m_dim={plan.size(dot.rows)}, a_major={int(not operand_k_major(dot.a))}, "
                f"b_major={int(not operand_k_major(dot.b))})"
            )
        self.define_iteration_bases(in_loop=False)
        self.emit_carry_init()
        begin, end, trip_static = self.loop_bounds()
        # Side loads of the first iteration; the loop refills them one
        # iteration ahead (dependent global loads would otherwise sit on
        # the serial chain).
        for index, site in enumerate(self.discover_prefetch_sites()):
            self.prefetched[site] = f"md_ax{index}"
        if self.prefetched:
            for register in self.prefetched.values():
                self.emit(f"{register} = cutlass.Float32(0.0)")
            with (
                self.nest(f"if {end} > {begin}:") if trip_static is None else _NoNest()
            ):
                if trip_static != 0:
                    self.emit_prefetch_loads(begin)
            for ahead in range(1, max(tc.stages - 1, 2)):
                value = f"({begin} + {ahead})"
                if trip_static is not None and ahead < trip_static:
                    self.emit_line_prefetch(value)
                elif trip_static is None:
                    with self.nest(f"if {value} < {end}:"):
                        self.emit_line_prefetch(value)
        if tc.tma:
            # The first ``S - 1`` iterations' tiles (the first iteration's
            # with a single slot) and the epilogue tiles land during the
            # carry seed.
            with self.nest("if md_warp == 0:"), self.nest(f"if {_PRIMS}.elect_sync():"):
                for spec in tc.tma:
                    self.emit(f"{_PRIMS}.prefetch_tensormap({spec.param}.get_ptr())")
                if tc.tx_bytes:
                    for ahead in range(max(tc.stages - 1, 1)):
                        value = f"({begin} + {ahead})" if ahead else begin
                        if trip_static is not None and ahead < trip_static:
                            self.emit_tma_issue(value, str(ahead))
                        elif trip_static is None:
                            with self.nest(f"if {value} < {end}:"):
                                self.emit_tma_issue(value, str(ahead))
                if tc.epilogue_tx_bytes:
                    self.emit_epilogue_tma_issue()
        self.emit(self.sync)
        with self.nest(f"for {self.loop_var} in cutlass.range({begin}, {end}):"):
            self.emit_iteration(begin, end)
        # Drain: the last MMA must complete before the epilogue reads TMEM.
        if trip_static != 0:
            guard = (
                self.nest(f"if {end} > {begin}:") if trip_static is None else _NoNest()
            )
            with guard:
                self.emit(f"{_RT}.mbar_wait(md_accb, ({end} - {begin} - 1) & 1)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)")
        self.emit_epilogue()
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")
        self.emit(self.sync)
        with self.nest("if md_warp == 0:"):
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
            )
            self.emit(
                f"{_PRIMS}.tcgen05_dealloc(md_acc0, {tc.tmem_cols}, group={_PRIMS}.CTAGroup.CTA_1)"
            )
        assert plan.threads == 32 * tc.warps
        return "\n".join(self.lines) + "\n"

    def emit_iteration_header(self, begin: str) -> str:
        """The loop axis base, the ring slot and its phase of iteration
        ``md_i``; returns the iteration offset expression."""
        tc, program = self.tc, self.program
        loop = self.loop_var
        offset = loop if begin == "0" else f"({loop} - {begin})"
        if program.loop_axis is not None:
            step = self.plan.size(program.loop_axis)
            self.emit(f"{self.axis_base(program.loop_axis)} = {loop} * {step}")
        self.emit(f"md_stage = {offset} % {tc.stages}")
        self.emit(f"md_round = ({offset} // {tc.stages}) & 1")
        return offset

    def emit_iteration(self, begin: str, end: str) -> None:  # type: ignore[override]
        tc, program = self.tc, self.program
        loop = self.loop_var
        offset = self.emit_iteration_header(begin)
        self.emit(f"md_parity = {offset} % 2")
        self.define_iteration_bases(in_loop=True)
        self.evaluator = _PassEvaluator(self)
        with self.nest(f"if {loop} > {begin}:"):
            self.emit(f"{_RT}.mbar_wait(md_accb, md_parity ^ 1)")
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
            )
        # Every thread runs this iteration's operand passes into slot i % S
        # (its previous reader, MMA(i - S), completed with the acc_ready
        # just waited) before the barrier.
        waited_full = any(
            tile.route in (ROUTE_INPLACE, ROUTE_CONVERT) for tile in tc.loop_tiles
        )
        if self.has_passes(tc.loop_tiles):
            self.emit_passes(
                tc.loop_tiles,
                "md_stage",
                f"{_RT}.mbar_wait(md_full + (md_stage), md_round)",
                "md_tidx",
                self.plan.threads,
            )
        loop_dots = program.dots_in("loop")
        carry_in = [s for s in program.stores if s.source == "carry_in"]
        carry_out = [s for s in program.stores if s.source == "body"]
        prepared = [dot for dot in loop_dots if self.prepares(dot)]
        if carry_in or prepared:
            with self.block("if md_epi:"):
                loaded: set[int] = set()
                for store in carry_in:
                    slot = program.frag_slots[store.frag]
                    if slot not in loaded:
                        self.emit_ld(slot)
                        loaded.add(slot)
                    with self.row_guard(slot):
                        if store in tc.staged_stores:
                            self.emit_stage_in(store, "md_parity", evaluate=False)
                        else:
                            self.emit_direct_store(store)
                for dot in prepared:
                    self.emit_rescale(dot, loaded)
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
            )
        self.emit(self.sync)
        with self.nest("if md_warp == 0:"):
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
            )
            if tc.tx_bytes and not waited_full:
                self.emit(f"{_RT}.mbar_wait(md_full + md_stage, md_round)")
            for dot in loop_dots:
                self.emit_dot_mmas(dot, "md_stage", accumulate=dot.acc is not None)
            with self.nest(f"if {_PRIMS}.elect_sync():"):
                self.emit(
                    f"{_PRIMS}.tcgen05_commit(md_accb, group={_PRIMS}.CTAGroup.CTA_1)"
                )
                if tc.tx_bytes and tc.stages > 1:
                    ahead = tc.stages - 1
                    with self.nest(f"if {loop} + {ahead} < {end}:"):
                        self.emit_tma_issue(
                            f"({loop} + {ahead})", f"(md_stage + {ahead}) % {tc.stages}"
                        )
        if tc.tx_bytes and tc.stages == 1:
            # Refill after the MMA completed: every thread waits the commit
            # (the next iteration's wait is then immediate).
            self.emit(f"{_RT}.mbar_wait(md_accb, md_parity)")
            with (
                self.nest("if md_warp == 0:"),
                self.nest(f"if {_PRIMS}.elect_sync():"),
                self.nest(f"if {loop} + 1 < {end}:"),
            ):
                self.emit_tma_issue(f"({loop} + 1)", "0")
        # The next iteration's side loads land in their registers while the
        # MMA runs; the lines of the iteration a ring depth ahead go to L2.
        if self.prefetched:
            with self.nest(f"if {loop} + 1 < {end}:"):
                self.emit_prefetch_loads(f"({loop} + 1)")
            ahead = max(tc.stages - 1, 2)
            with self.nest(f"if {loop} + {ahead} < {end}:"):
                self.emit_line_prefetch(f"({loop} + {ahead})")
        # The snapshots stream out while the MMA runs: the store warps, or
        # the epilogue warps when there are none (packing the rows after the
        # issue instead, with the previous iteration's drain, measured 122
        # vs 104 us on the s2 forward).
        loop_staged = [s for s in tc.staged_stores if s.region == "loop"]
        if loop_staged:
            if tc.store_warps:
                nest = self.nest(f"if md_warp >= {TCGEN05_EPILOGUE_WARPS}:")
                idx, threads = (
                    f"(md_tidx - {32 * TCGEN05_EPILOGUE_WARPS})",
                    32 * tc.store_warps,
                )
            else:
                nest = _NoNest()
                idx, threads = "md_tidx", self.plan.threads
            with nest:
                for store in loop_staged:
                    self.emit_stage_out(store, "md_parity", loop, idx, threads)
        if carry_out:
            # A snapshot after the dot: wait for this iteration's MMA.
            self.emit(f"{_RT}.mbar_wait(md_accb, md_parity)")
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
            )
            self.vec = "md_vo"
            with self.block("if md_epi:"):
                loaded = set()
                for store in carry_out:
                    slot = program.frag_slots[store.frag]
                    if slot not in loaded:
                        self.emit_ld(slot)
                        loaded.add(slot)
                    with self.row_guard(slot):
                        self.emit_direct_store(store)
            self.vec = "md_v"
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
            )

    # -- epilogue -----------------------------------------------------------------

    def emit_epilogue_commit_wait(self) -> None:
        """Warp 0 commits the epilogue MMAs issued so far; every thread
        waits (phase = the number of epilogue commits)."""
        with self.nest("if md_warp == 0:"), self.nest(f"if {_PRIMS}.elect_sync():"):
            self.emit(
                f"{_PRIMS}.tcgen05_commit(md_eacc, group={_PRIMS}.CTAGroup.CTA_1)"
            )
        self.emit(f"{_RT}.mbar_wait(md_eacc, {self.epilogue_commits & 1})")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)")
        self.epilogue_commits += 1

    def emit_exchange(self, tile: Tcgen05Tile) -> None:  # type: ignore[override]
        """Epilogue warps: the fragment operand's value (its pointwise chain
        over the finished accumulator) per element of this thread's row,
        packed into the UMMA exchange tile."""
        operand = tile.operand
        slot = tile.source_slot
        assert slot is not None
        geometry = self.slot(slot)
        es = tile.elem_bytes
        per_packet = 16 // es
        value = self.pass_value(operand)
        self.hoist_scalars(value)
        row = self.row_var(slot)
        with self.row_guard(slot):
            for col in range(0, geometry.cols, per_packet):
                values = []
                for c in range(col, col + per_packet):
                    self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                    coords = self.element_coords(value, slot, c, operand.swap)
                    values.append(self.evaluator.evaluate(value, coords))
                self.evaluator.reg = None
                words = [self.pack(values[2 * q], values[2 * q + 1]) for q in range(4)]
                address = self.tile_address(tile, "0", row, str(col // per_packet))
                self.emit(f"{_RUNTIME}.sts_v4_b32({address}, {', '.join(words)})")

    def emit_epilogue(self) -> None:  # type: ignore[override]
        tc, program = self.tc, self.program
        dots = program.dots_in("epilogue")
        stores = program.stores_in("epilogue")
        if not dots and not stores:
            return
        self.evaluator = _PassEvaluator(self)
        self.define_iteration_bases(in_loop=False)
        # Slots written by epilogue dots whose commit has not been waited.
        pending: set[int] = set()
        # Exchange tiles already written.
        built: set[tuple[object, ...]] = set()

        def settle(slots: Sequence[int]) -> None:
            if any(slot in pending for slot in slots):
                self.emit_epilogue_commit_wait()
                pending.clear()

        if tc.epilogue_tx_bytes:
            self.emit(f"{_RT}.mbar_wait(md_efull, 0)")
        if self.has_passes(tc.epilogue_tiles):
            self.emit_passes(tc.epilogue_tiles, "0", None, "md_tidx", self.plan.threads)
        issued: list[Dot] = []

        def issue(dot: Dot) -> None:
            # Exchange tiles and accumulator preparation of this dot, after
            # the commits they depend on.
            exchanges = [
                self.tc.tile(dot, operand)
                for operand in (dot.a, dot.b)
                if operand.kind == "frag"
                and self.tc.tile(dot, operand).key not in built
            ]
            sources = [cast("int", tile.source_slot) for tile in exchanges]
            if dot.acc_frag is not None:
                assert dot.acc_slot is not None
                sources.append(dot.acc_slot)
            settle(sources)
            prepares = self.prepares(dot)
            if exchanges or prepares:
                with self.block("if md_epi:"):
                    loaded: set[int] = set()
                    for tile in exchanges:
                        source = cast("int", tile.source_slot)
                        if source not in loaded:
                            self.emit_ld(source)
                            loaded.add(source)
                        self.emit_exchange(tile)
                        built.add(tile.key)
                    if prepares:
                        self.emit_rescale(dot, loaded)
                if exchanges:
                    self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
                )
                self.emit(self.sync)
            elif not issued:
                # The first dot after the epilogue passes still needs them
                # published.
                self.emit(self.sync)
            with self.nest("if md_warp == 0:"):
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
                )
                self.emit_dot_mmas(dot, "0", accumulate=dot.acc is not None)
            pending.add(dot.slot)
            issued.append(dot)

        staged_count = 0

        def emit_store(store: Store, index: int) -> None:
            nonlocal staged_count
            slot = program.frag_slots[store.frag]
            settle([slot])
            self.vec = f"md_ve{index}_"
            if store in tc.staged_stores:
                if staged_count:
                    # The previous drain must have read the staging tile.
                    self.emit(self.sync)
                staged_count += 1
                with self.block("if md_epi:"):
                    self.emit_ld(slot)
                    with self.row_guard(slot):
                        self.emit_stage_in(store, "0", evaluate=True)
                self.vec = "md_v"
                self.emit(self.sync)
                self.emit_stage_out(
                    store, "0", self.loop_var, "md_tidx", self.plan.threads
                )
            else:
                with self.block("if md_epi:"):
                    self.emit_ld(slot)
                    with self.row_guard(slot):
                        self.emit_direct_store(store)
                self.vec = "md_v"

        # Program order: a store reads its slot after the commits pending
        # on it; the results stream out through the staging tile (or pair
        # stores).
        by_node: dict[Node, Dot | Store] = {dot.node: dot for dot in dots}
        by_node.update({store.access.node: store for store in stores})
        order = program.epilogue_order or (
            *(dot.node for dot in dots),
            *(store.access.node for store in stores),
        )
        for index, node in enumerate(order):
            item = by_node[node]
            if isinstance(item, Dot):
                issue(item)
            else:
                emit_store(item, index)
        if pending:
            self.emit_epilogue_commit_wait()
            pending.clear()


class _NoNest:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        return None


__all__ = [
    "MMA_DAG_TCGEN05",
    "TCGEN05_DEFAULT_STAGES",
    "TCGEN05_DEFAULT_WARPS",
    "TCGEN05_EPILOGUE_WARPS",
    "TCGEN05_WARP_CHOICES",
    "TMA_KIND",
    "SlotGeometry",
    "Tcgen05Facts",
    "Tcgen05Plan",
    "Tcgen05Tile",
    "TmaSpec",
    "codegen_mma_dag_tcgen05",
    "operand_k_major",
    "plan_tcgen05",
    "tcgen05_atom_bytes",
    "tcgen05_config_reason",
    "tcgen05_facts",
    "tcgen05_repair",
    "tcgen05_seed_configs",
    "tcgen05_slot_columns",
    "tcgen05_smem_bytes",
    "tcgen05_stage_bytes",
    "tcgen05_tmem_cols",
    "tcgen05_warps_reason",
]
