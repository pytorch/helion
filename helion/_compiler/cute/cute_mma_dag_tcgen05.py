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
import re
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_RDIM
from .cute_mma_dag import AXIS_ROOT
from .cute_mma_dag import MMA_DAG_K_STEP
from .cute_mma_dag import MMA_DAG_LOADS_TMA
from .cute_mma_dag import MMA_DAG_STAGE_CHOICES
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import MMA_DAG_WARPS_KEY
from .cute_mma_dag import ROUTE_CONVERT
from .cute_mma_dag import ROUTE_DIRECT
from .cute_mma_dag import ROUTE_EXCHANGE
from .cute_mma_dag import ROUTE_INPLACE
from .cute_mma_dag import ROUTE_SYNC
from .cute_mma_dag import Access
from .cute_mma_dag import Axis
from .cute_mma_dag import AxisFacts
from .cute_mma_dag import CuteMmaDagFacts
from .cute_mma_dag import Derived
from .cute_mma_dag import Dot
from .cute_mma_dag import MmaDagPlan
from .cute_mma_dag import MmaDagProgram
from .cute_mma_dag import Operand
from .cute_mma_dag import Store
from .cute_mma_dag import _decline
from .cute_mma_dag import _is_call
from .cute_mma_dag import _storage
from .cute_mma_dag import mma_dag_ctas_per_sm
from .cute_mma_dag import staged_store_reason
from .cute_mma_dag_codegen import _RUNTIME
from .cute_mma_dag_codegen import Coords
from .cute_mma_dag_codegen import _broadcast
from .cute_mma_dag_codegen import _Emitter
from .cute_mma_dag_codegen import _Evaluator
from .cute_mma_dag_codegen import _Hoist
from .cute_mma_dag_codegen import _shape
from .cute_mma_dag_codegen import emit_mma_dag_module_statements
from .cute_mma_dag_codegen import mma_dag_alias_plan
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
# The epilogue warps' own barrier while the pass warps run their loop.
EPILOGUE_WARPS_BARRIER = (
    f"cute.arch.barrier(barrier_id=1, number_of_threads={32 * TCGEN05_EPILOGUE_WARPS})"
)
TCGEN05_M_CHOICES: tuple[int, ...] = (64, 128)
TCGEN05_MAX_N = 256
TCGEN05_MAX_ROWS = 256
TCGEN05_MAX_TMEM_COLS = 512
# fp32 columns per thread a derived value may hold in registers (its input
# rows and its own) while it is evaluated.
TCGEN05_MAX_DERIVED_COLS = 192
# TMEM allocation granularity: slot offsets keep it.
TCGEN05_SLOT_ALIGN = 32
# Four warps (the epilogue warps drain the staging tile themselves) match or
# beat eight at s2: fwd 104.5 vs 104.6 us, bwd 189.6 vs 222 us device time.
TCGEN05_DEFAULT_WARPS = 4
TCGEN05_SEED_WARPS: tuple[int, ...] = (4, 8)
TCGEN05_DEFAULT_STAGES = 8
TCGEN05_SEED_STAGES: tuple[int, ...] = (8, 4)
# Grid loops whose tiles fit neither eight nor four stages seed two.
TCGEN05_FALLBACK_STAGES: tuple[int, ...] = (2,)
TCGEN05_TILE_LADDER: tuple[int, ...] = (64, 128)
TCGEN05_N_LADDER: tuple[int, ...] = (64, 128, 32, 16)
# Columns per tcgen05.ld / st (64 fp32 registers per thread).
TCGEN05_LD_COLS = 64
# Bytes of one operand row per MMA K step (K = 16 for 16-bit, 8 for tf32).
TCGEN05_K_STEP_BYTES = 32
# Side loads (decay scalars / rows) carried one iteration ahead in registers.
TCGEN05_MAX_PREFETCH = 16
# Full packet rounds of an operand pass loaded ahead of their rewrites
# (four 32-bit words each held live).
TCGEN05_PASS_ROUNDS = 8
# Registers (entries plus the vector elements they read) an iteration-ahead
# table set may take per thread; beyond it the values stay inline.
TCGEN05_MAX_TABLE_REGS = 96
# The epilogue's per-column side loads (one register per column of the
# thread's half-row, carried through the loop) go out ahead of the drain only
# when the half-row has at most this many columns.
TCGEN05_EPILOGUE_PREFETCH_COLS = 32
# One swizzle atom row: the widest swizzle (SW128) spans 128 bytes.
ATOM_MAX_BYTES = 128
# Region programs rank their seeds by innermost-loop iterations per resident
# CTA: every iteration is a commit round trip the CTA waits through, and up
# to three resident CTAs overlap their chains (beyond three the SM is busy
# already).  B200, device us at s2 / s1: dk [64] s2 (2 iterations at two
# per SM) 150 / 49 against [32] s2 (4 at three) 152 / 51; wy_dL [64, 32, 128]
# s3 (7 at three) 261 / 78 against [32, 32, 128] s3 (9 at three) 289 / 80 and
# [64, 64, 128] s2 (5 at two) 333 / 92; gated dqkw [128, 32, 32, 64, 128] s3
# (19 at two) 690 / 186 against [128, 32, 16, 64, 128] s2 (35 at three) 717 /
# 193.
TCGEN05_REGION_SEED_CTAS = 3
# ... and up to four for a program with a chain loop (a loop without a tile
# axis), whose commit round trips only the co-resident CTAs overlap.
TCGEN05_CHAIN_SEED_CTAS = 4
SLAB_ALIGNMENT = 1024
TMA_MAX_RANK = 5
# Tensor maps the launcher passes to one kernel.
TMA_MAX_MAPS = 16
# The thread issuing the TMA stores (bulk groups are tracked per thread):
# warp 0's first lane, between the batch's MMAs and its commit -- ahead of
# the MMAs the issue delayed them by ~15 us at s2 on the delta chains, and
# the wait for the previous iteration's groups must precede the commit the
# other threads wait for before reusing a staging buffer.
TMA_STORE_THREAD = 0
TMA_KIND = "helion_mma_dag_tma"
_SWIZZLE_BY_ROW_BYTES = {32: "s32b", 64: "s64b", 128: "s128b"}
# The 32-bit MN-major UMMA layout: 128-byte rows whose 32-byte units XOR with
# the row mod 4 (``Swizzle<2,5,2>``); the descriptor's stride byte offset is
# the four-row swizzle period.
ATOM_32B_SWIZZLE = "s128b_atom_32b"
_DESC_SWIZZLE = {
    "s32b": "SWIZZLE_32B",
    "s64b": "SWIZZLE_64B",
    "s128b": "SWIZZLE_128B",
    ATOM_32B_SWIZZLE: "SWIZZLE_128B_ATOM_32B",
}
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
    # The tensor core reads the tile MN-major (its K along the rows).
    mn_major: bool = False
    # A per-element pass rewrites the tile (a pointwise chain or a dtype
    # change between the source and the operand).
    passed: bool = False


@dataclasses.dataclass(frozen=True)
class Tcgen05ExchangeFacts:
    """A fragment read as a dot operand through a shared exchange tile: its
    slot's (rows, cols) axis indices, the operand element size and the
    region (a loop operand's tile is rewritten every iteration)."""

    slot: int
    rows: int
    cols: int
    elem_bytes: int
    region: str = "epilogue"
    mn_major: bool = False


@dataclasses.dataclass(frozen=True)
class RegisterCarryFacts:
    """A loop carry that may live in the epilogue warps' registers instead
    of tensor memory (``cute_mma_dag_tcgen05_regions.register_carries``):
    its slot and, per derived value of its loop that does not read it, the
    column axes live while that value is evaluated (its inputs' rows, its
    own and this carry's).  The carry is resident at a config when every
    such live set fits the derived-value register budget
    (``resident_slots``); otherwise it keeps its tensor-memory columns."""

    slot: int
    live: tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class Tcgen05Facts:
    tiles: tuple[Tcgen05TileFacts, ...]
    exchanges: tuple[Tcgen05ExchangeFacts, ...]
    # (slot, element size) of the loop snapshots (two staging buffers each)
    # and of the epilogue stores (one staging tile at a time).
    loop_stores: tuple[tuple[int, int], ...]
    epilogue_stores: tuple[tuple[int, int], ...]
    # Column axis indices of the fragments each derived value reads plus its
    # own: one row of each sits in registers while it is evaluated.
    derived_cols: tuple[tuple[int, ...], ...] = ()
    # The loop body is emitted in program order (with its step barrier).
    general_loop: bool = False
    # Region programs (``cute_mma_dag_tcgen05_regions``): the shared-memory
    # items, the TMEM-resident slots, the ring loops and the regions with
    # staged tiles.
    regions: bool = False
    items: tuple[Tcgen05SmemItem, ...] = ()
    tmem_slots: tuple[int, ...] = ()
    # Region programs: the carries that live in registers at the configs
    # whose register budget admits them (``resident_slots``).
    register_carries: tuple[RegisterCarryFacts, ...] = ()
    # Slot pairs that may not share TMEM columns beyond the live groups (a
    # dot and the accumulator it copies, the batch reorder's pairs).
    tmem_conflicts: tuple[tuple[int, int], ...] = ()
    # (output slot, carry slot) of the loop outputs computed in another
    # slot: the output takes the carry's columns when the two never
    # conflict, and the end-of-iteration copy vanishes.
    tmem_aliases: tuple[tuple[int, int], ...] = ()
    # Region programs: operand tiles rewritten by a per-packet pass (a
    # dtype conversion or a pointwise chain) -- the work that spreads over
    # every warp; the rest of the render runs on the four epilogue warps.
    pass_tiles: int = 0
    # Some dot runs on fp32 (tf32) operands.
    tf32: bool = False
    ring_loops: int = 0
    staged_regions: int = 0
    # Derived values' tile inputs that may ride a loop ring (the
    # ``cute_mma_dag_loads`` knob has a subject).
    aux_items: int = 0
    # Region programs: the live groups with the region emitter's exchange
    # points (a root fragment's exchange tile is written once ahead of the
    # loop reading it), for the TMEM layout; None: the planner's.
    live_groups: tuple[tuple[int, ...], ...] | None = None


@dataclasses.dataclass(frozen=True)
class Tcgen05SmemItem:
    """A shared-memory item of a region program: an operand tile (``copies``
    0: one per ring stage), the raw source tile beside a converted operand,
    an exchange tile, a staging buffer or the vector scratch; ``rows`` /
    ``cols`` are axis indices."""

    kind: str
    frame: int
    rows: int
    cols: int
    elem_bytes: int
    copies: int
    # Root items live within one root segment (the k-th run of root steps
    # between the loops; 0 is the prologue for a raw tile settled there)
    # overlay the loop frames, which are dead then (-1: live across loops).
    segment: int = -1


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
    if not program.simple:
        from .cute_mma_dag_tcgen05_regions import region_facts

        return region_facts(program)
    # A loop output computed in another slot (a no-acc dot over the carry,
    # a rescaled accumulator whose leaf an operand reads) is copied back into
    # the carry's registers at the iteration end by the warp emitter; this
    # emitter's TMEM carry has no such copy, so the next iteration's exchange
    # and the epilogue would read the carry's columns unchanged.  (The region
    # renderer copies the output into the carry's slot: wy's Neumann chain.)
    for carry in program.carries:
        output = program.resolve(carry.output)
        if output is not carry.placeholder and program.frag_slots[output] != carry.slot:
            return None
    # This emitter's epilogue issues dots and stores only: a derived value
    # there (a pointwise value over a fragment and a 2-D tile input) would
    # be dropped, its store staging the input fragment instead.
    if any(step.loop < 0 for step in program.derived):
        return None
    # The single-loop emitter packs 16-bit operands only (tf32 dots take the
    # region renderer).
    if any(
        operand.dtype not in (torch.bfloat16, torch.float16)
        for dot in program.dots
        for operand in (dot.a, dot.b)
    ):
        return None
    # What this emitter renders: the loop's dots over staged (passed /
    # exchanged) operands accumulating into TMEM carries, the carries'
    # snapshots and body stores, derived fragment values of the loop, the
    # epilogue's dots and fragment stores.  Vector values -- reductions of
    # a fragment, vector ops, vector / scalar stores -- are the region
    # renderer's; here they would be dropped (``s_out[idx, tile_d] =
    # s_in[...] + h.sum(-1) * scale`` inside the recurrence rendered
    # without the reduction, the load or the store).
    if program.reduces or program.vec_ops or program.vec_stores or program.vec_leaves:
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
                        mn_major=not operand_k_major(operand),
                        passed=operand.pointwise or load.dtype != operand.dtype,
                    )
                )
            else:
                assert operand.frag is not None
                slot = program.frag_slots[operand.frag]
                exchanges.append(
                    Tcgen05ExchangeFacts(
                        slot=slot,
                        rows=operand.rows.index,
                        cols=operand.cols.index,
                        elem_bytes=operand.dtype.itemsize,
                        region=dot.region,
                        mn_major=not operand_k_major(operand),
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
    derived_cols = tuple(
        (
            *(program.slots[program.frag_slots[frag]][1].index for frag in step.frags),
            step.cols.index,
        )
        for step in program.derived
    )
    return Tcgen05Facts(
        tiles=tuple(tiles),
        exchanges=tuple(exchanges),
        loop_stores=loop_stores,
        epilogue_stores=epilogue_stores,
        derived_cols=derived_cols,
        general_loop=program_order_loop(program),
    )


def program_order_loop(program: MmaDagProgram) -> bool:
    """The loop body has steps beyond "rescale the carries, every dot, then
    the snapshots": fragments as loop operands, derived values, mid-iteration
    stores (the delta-rule state shapes) or a dot prepared (rescaled or
    copied) from a result of the same iteration (``h = dot(acc=dot(acc=h) *
    0.5)``); it is emitted as blocks in program order."""
    loop_dots = program.dots_in("loop")
    results = {dot.node for dot in loop_dots}
    return (
        bool(program.derived)
        or any(
            operand.kind == "frag" for dot in loop_dots for operand in (dot.a, dot.b)
        )
        or any(store.source == "body" for store in program.stores)
        or any(
            dot.acc_frag in results
            and (
                dot.slot != dot.acc_slot
                or (dot.acc is not None and strip_new_vars(dot.acc) is not dot.acc_frag)
            )
            for dot in loop_dots
        )
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


def resident_slots(tc: Tcgen05Facts, resolved: Sequence[int]) -> tuple[int, ...]:
    """The register carries (``Tcgen05Facts.register_carries``) resident at
    the resolved block sizes: every derived value of the carry's loop keeps
    its live rows plus the carry's within the register budget."""
    return tuple(
        carry.slot
        for carry in tc.register_carries
        if all(
            sum(resolved[axis] for axis in live) <= TCGEN05_MAX_DERIVED_COLS
            for live in carry.live
        )
    )


def tcgen05_slot_columns(facts: CuteMmaDagFacts, resolved: Sequence[int]) -> int:
    tc = facts.tcgen05
    if tc is not None and tc.regions:
        from .cute_mma_dag_tcgen05_regions import region_tmem_columns

        return region_tmem_columns(facts, resolved)
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
    facts: CuteMmaDagFacts, resolved: Sequence[int], staged: bool, ring_bytes: int
) -> int:
    """Shared memory beside the ring: the epilogue operand tiles, the
    exchange tiles, the loop staging buffers and the epilogue staging tile
    (which ``plan_tcgen05`` overlays on the drained ring when it fits in
    ``ring_bytes``)."""
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
    if epilogue_staging <= ring_bytes:
        return total
    return total + epilogue_staging


def tcgen05_smem_bytes(
    facts: CuteMmaDagFacts, resolved: Sequence[int], stages: int, staged: bool
) -> int:
    tc = facts.tcgen05
    assert tc is not None
    if tc.regions:
        from .cute_mma_dag_tcgen05_regions import region_smem_bytes

        return region_smem_bytes(facts, resolved, stages)
    total = stages * tcgen05_stage_bytes(tc, resolved)
    total += tcgen05_fixed_bytes(facts, resolved, staged, total)
    # Barriers (the program-order loop adds its step barrier) and the TMEM
    # slot.
    return total + 8 * (stages + 3 + int(tc.general_loop)) + 16


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
    for tile in tc.tiles:
        if tile.passed and tile.elem_bytes == 4 and tile.source_bytes == 4:
            return "no in-place pass over fp32 (the pass rewrites 16-bit pairs)"
    for cols in tc.derived_cols:
        live = sum(resolved[c] for c in cols)
        if live > TCGEN05_MAX_DERIVED_COLS:
            return (
                f"a derived value holds {live} fp32 columns per thread in "
                f"registers (limit {TCGEN05_MAX_DERIVED_COLS})"
            )
    if tc.regions:
        from .cute_mma_dag_tcgen05_regions import region_config_reason

        return region_config_reason(facts, resolved, stages)
    smem = tcgen05_smem_bytes(facts, resolved, stages, True)
    if facts.smem_capacity and smem > facts.smem_capacity:
        return (
            f"{stages} tcgen05 stages need {smem} B of shared memory "
            f"(capacity {facts.smem_capacity})"
        )
    return None


def region_warps_matter(facts: CuteMmaDagFacts, sizes: Mapping[int, int]) -> bool:
    """Whether a region program's render changes with the warp count at the
    tile ``sizes``: only the per-packet work spreads over every warp -- the
    operand passes (``Tcgen05Facts.pass_tiles``) and the packet drains of
    staging tiles whose rows are not one swizzle atom (the others store by
    TMA from one thread); exchanges, derived values, reductions and stores'
    stage-ins run on the four epilogue warps."""
    tc = facts.tcgen05
    assert tc is not None
    if tc.pass_tiles:
        return True
    for item in tc.items:
        if item.kind != "staging":
            continue
        cols = facts.block_size(item.cols, sizes)
        if cols is None or cols * item.elem_bytes not in (16, 32, 64, 128):
            return True
    return False


def region_iterations(
    facts: CuteMmaDagFacts,
    tunable: Sequence[tuple[int, AxisFacts]],
    choice: Sequence[int],
) -> int:
    """Innermost-loop iterations per CTA of a region program at the tile
    ``choice`` of the ``tunable`` axes: for every innermost loop the product
    of the trip counts of the loops enclosing it (a loop without a tile
    axis counts one)."""
    trips: dict[int, int] = {}
    for (_index, axis), tile in zip(tunable, choice, strict=True):
        if axis.kind == AXIS_LOOP and axis.loop >= 0:
            trips[axis.loop] = trips.get(axis.loop, 1) * (axis.extent // tile)
    total = 0
    for index, loop in enumerate(facts.loops):
        if not loop.innermost:
            continue
        count = 1
        while index >= 0:
            count *= trips.get(index, 1)
            index = facts.loops[index].parent
        total += count
    return total


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


def _seed_choices(
    ladders: Sequence[Sequence[int]],
) -> Iterator[tuple[int, ...] | None]:
    """Every tile choice, a ``None`` marker, then every tile choice again
    (the second pass is the fallback stage depth, taken only when the
    first produced no seed)."""
    yield from itertools.product(*ladders)
    yield None
    yield from itertools.product(*ladders)


def tcgen05_persistent_ctas(
    facts: CuteMmaDagFacts, sizes: Mapping[int, int], stages: int
) -> int:
    """The CTAs per SM a persistent launch of the tcgen05 config should
    run: the shared- and tensor-memory occupancy of one CTA (capped at the
    knob's largest choice), 0 when the grid would not fill two waves of
    them (nothing to amortize)."""
    from .cute_mma_dag import MMA_DAG_PERSISTENT_CHOICES

    resolved = [cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))]
    smem = tcgen05_smem_bytes(facts, resolved, stages, True)
    tmem = tcgen05_tmem_cols(tcgen05_slot_columns(facts, resolved))
    ctas = mma_dag_ctas_per_sm(
        smem, facts.smem_capacity, tmem, max(MMA_DAG_PERSISTENT_CHOICES)
    )
    grid = facts.grid_blocks
    for axis, size in zip(facts.axes, resolved, strict=True):
        if axis.kind == AXIS_ROOT:
            grid *= axis.extent // size
    if not facts.sm_count or grid < 2 * ctas * facts.sm_count:
        return 0
    return ctas


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
    # Serial chains with derived values (the delta-rule state kernels) run
    # waves of one CTA per SM whose iteration cost grows with the tile:
    # every stage depth is a candidate and the rank is fewest waves, then
    # the narrowest tile (dstate at s2: [32] 441 us vs [64] 557 vs [16],
    # two waves, 767; at s1 [64] 76 vs [32] 118).
    chain = bool(tc.derived_cols)
    # A region program's loop without a tile axis (``hl.grid`` / a fixed
    # step: no ring; its dots run on fragments and staged tiles).
    loop_axes = {axis.loop for axis in facts.axes if axis.kind == AXIS_LOOP}
    chain_loop = tc.regions and any(
        index not in loop_axes for index in range(len(facts.loops))
    )
    fallback = False
    for choice in _seed_choices(ladders):
        if choice is None:
            # No tile fits eight or four stages: seed two stages instead.
            if (
                ranked
                or chain
                or any(axis.kind == AXIS_LOOP for _index, axis in tunable)
            ):
                break
            fallback = True
            continue
        sizes: dict[int, int] = dict(fixed)
        # The deepest ring any loop fills (a region program's loops each
        # own a ring of ``stages`` slots; a one-trip loop beside a four-trip
        # one does not cap the depth).
        trip = 0
        for (_index, axis), tile in zip(tunable, choice, strict=True):
            sizes[axis.block_id] = tile
            if axis.kind == AXIS_LOOP:
                trip = max(trip, axis.extent // tile)
        stage_choices: tuple[int, ...] = TCGEN05_SEED_STAGES
        if trip:
            stage_choices = tuple(
                s for s in sorted(MMA_DAG_STAGE_CHOICES, reverse=True) if s <= trip
            )
        elif chain:
            stage_choices = TCGEN05_SEED_STAGES + TCGEN05_FALLBACK_STAGES
        if fallback:
            stage_choices = TCGEN05_FALLBACK_STAGES
        for stages, warps in itertools.product(stage_choices, TCGEN05_SEED_WARPS):
            if (
                mma_dag_config_reason(facts, sizes, warps, stages, MMA_DAG_TCGEN05)
                is not None
            ):
                continue
            if (
                tc.regions
                and warps > TCGEN05_DEFAULT_WARPS
                and not region_warps_matter(facts, sizes)
            ):
                # The extra warps would idle: the render is the four-warp one
                # (state_du_kda [16] rendered identically at four and eight).
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
            if tc.regions:
                # Region programs (many small CTAs, a few iterations each):
                # two CTAs per SM of shared memory and TMEM first (one CTA
                # leaves the SM idle through every commit round trip: wy_dL
                # at s2 [64, 32, 16] s2 677 us against [32, 32, 32] s2 403;
                # dk [64] s3 at one per SM 241 against s2 at two 150), a ring
                # that pipelines (two stages: [32, 32, 128] s1 443 against s2
                # 392), then the fewest innermost-loop iterations per
                # resident CTA (``TCGEN05_REGION_SEED_CTAS``) -- every
                # iteration is at least one commit round trip, overlapped by
                # the CTAs sharing the SM (dqkw at s2: [32, 32, 32] s2 662 us
                # with 20 against [64, 16, 16] 1311 with 66; wy_dL [32, 32,
                # 128] s2 392 with 9 against [16, 16, 16] 488 with 24) -- the
                # stage depth, the loop area and the warp count.  A program
                # with a chain loop -- a loop without a tile axis, whose
                # iterations are MMA commit round trips with no ring to
                # overlap them (wy's Neumann doubling) -- ranks by the CTAs
                # per SM up to four ahead of the ring depth: only the
                # co-resident CTAs overlap the chain (wy at s2: [128, 32, 32]
                # s2 at four per SM 332 us against [128, 64, 64] s2 at three
                # 407 and [128, 64, 64] s1 at four 373; wy_dL measured four
                # no better than three).
                per_sm = mma_dag_ctas_per_sm(smem, facts.smem_capacity, tmem)
                cap = (
                    TCGEN05_CHAIN_SEED_CTAS if chain_loop else TCGEN05_REGION_SEED_CTAS
                )
                resident = max(min(per_sm, cap), 1)
                # Iterations per resident CTA, exact in twelfths (the lcm of
                # one to four residents).
                rank = (
                    int(ctas >= 128),
                    min(per_sm, cap) if chain_loop else int(per_sm >= 2),
                    int(stages >= 2),
                    -(region_iterations(facts, tunable, choice) * 12) // resident,
                    stages,
                    loop_area,
                    -warps,
                )
            elif chain:
                waves = -(-ctas // facts.sm_count) if facts.sm_count else 1
                rank = (-waves, -root_area, stages, -warps)
            else:
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
    # Region programs: the frame holding the tile and whether it is a ring
    # tile of that frame's loop (``offset`` is then within the ring slot).
    frame: int = -1
    ring: bool = False

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
    def k_elems(self) -> int:
        """Elements per MMA K step: 32 bytes (16 of a 16-bit type, 8 tf32)."""
        return TCGEN05_K_STEP_BYTES // self.elem_bytes

    @property
    def k_steps_per_slab(self) -> int:
        return self.atom_bytes // TCGEN05_K_STEP_BYTES

    def k_step_offset(self, step: int, k_major: bool) -> int:
        """Byte advance of the descriptor start for K step ``step``."""
        if k_major:
            slab, local = divmod(step, self.k_steps_per_slab)
            return slab * self.slab_bytes + local * TCGEN05_K_STEP_BYTES
        return step * self.k_elems * self.atom_bytes

    def leading_byte_offset(self, k_major: bool) -> int:
        return 16 if k_major else self.slab_bytes

    @property
    def stride_byte_offset(self) -> int:
        """Stride between the row groups of one swizzle period: eight
        128-byte rows, four under the 32-byte-unit swizzle."""
        if self.swizzle == ATOM_32B_SWIZZLE:
            return 4 * self.atom_bytes
        return 8 * self.atom_bytes


@dataclasses.dataclass(frozen=True)
class AuxTile:
    """A 2-D tile a derived value reads at the fragment coordinates (region
    programs), fetched by TMA into the derived value's loop ring beside the
    operand tiles -- the swizzled UMMA layout, so each epilogue thread reads
    its row's 16-byte chunks conflict-free -- instead of per-element global
    loads in fragment layout (lane = row: one sector per lane per load)."""

    key: tuple[object, ...]
    tag: str
    access: Access
    frame: int
    rows: int
    cols: int
    elem_bytes: int
    atom_bytes: int
    swizzle: str
    slabs: int
    # Byte offset within the ring slot.
    offset: int
    # None: the tile is a ring operand's (the same load as loaded), brought
    # in by that operand's TMA and, when ``rewritten``, rewritten in place
    # by its pass after the aux reads.
    tma: TmaSpec | None
    raw_tma: TmaSpec | None = None
    ring: bool = True
    rewritten: bool = False
    # The owning operand tile's key when the aux reads share one (a staged
    # or hoisted owner is waited by its region's barrier before the reads).
    owner_key: tuple[object, ...] | None = None

    @property
    def chunks_per_atom(self) -> int:
        return self.atom_bytes // 16

    @property
    def slab_bytes(self) -> int:
        return self.rows * self.atom_bytes


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
    # The epilogue staging tile's size (it starts at epilogue_staging_offset).
    epilogue_staging_bytes: int = 0
    # Region programs (``cute_mma_dag_tcgen05_regions``).
    regions: bool = False
    ring_offsets: dict[int, int] = dataclasses.field(default_factory=dict)
    ring_bytes: dict[int, int] = dataclasses.field(default_factory=dict)
    ring_tiles: dict[int, tuple[Tcgen05Tile, ...]] = dataclasses.field(
        default_factory=dict
    )
    ring_tx: dict[int, int] = dataclasses.field(default_factory=dict)
    staged_tiles: dict[int, tuple[Tcgen05Tile, ...]] = dataclasses.field(
        default_factory=dict
    )
    staged_tx: dict[int, int] = dataclasses.field(default_factory=dict)
    # Region programs: the derived values' tile inputs riding each loop's
    # ring (``ring_tx`` counts their bytes).
    aux_tiles: dict[int, tuple[AuxTile, ...]] = dataclasses.field(default_factory=dict)
    scratch_offset: int = -1
    # The fragment slots holding tensor-memory columns (the others stay in
    # registers).
    tmem_slots: tuple[int, ...] = ()
    # The loop carries living in the epilogue registers at this config
    # (``resident_slots``): never loaded from or stored to tensor memory.
    register_carries: tuple[int, ...] = ()
    # Loop stores whose staging tile goes to global memory by TMA (store
    # access node -> its tensor map), one bulk group per issue point.
    store_tma: dict[Node, TmaSpec] = dataclasses.field(default_factory=dict)
    # Warps 4-7 rewrite the loop's operand tiles (the passes) one tile ahead
    # of the epilogue warps' chain, handing each over through a per-stage
    # mbarrier (``cute_mma_dag_pass_warps``); the stores then take the
    # four-warp path.
    pass_warps: bool = False

    @property
    def store_warps(self) -> int:
        if self.pass_warps:
            return 0
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
    pass_warps: bool = False,
    loads_mode: str = MMA_DAG_LOADS_TMA,
    persistent: bool = False,
) -> Tcgen05Plan:
    """The tcgen05 plan for the resolved ``sizes`` (the config passed
    :func:`tcgen05_config_reason`)."""
    env = CompileEnvironment.current()
    tc = facts.tcgen05
    assert tc is not None
    if tc.regions:
        from .cute_mma_dag_tcgen05_regions import plan_tcgen05_regions

        return plan_tcgen05_regions(
            program, facts, sizes, routes, warps, stages, store_mode, loads_mode
        )
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
        # The tile's columns are the contiguous tensor dim (innermost in TMA
        # order), its rows the other tile dim -- by position, since both may
        # index one axis (``dL[idx, :, :]`` over one rdim).
        tile_dims = [access.subscript_index(p) for p in access.tile_positions]
        rows_dim = next(dim for dim in tile_dims if dim != dims[0])
        box = []
        slab_dim = None
        slab_elems = col_count
        for position, dim in enumerate(dims):
            if position == 0:
                slab_elems = min(col_count, atom_bytes // elem_bytes)
                box.append(slab_elems)
                slab_dim = position
            elif dim == rows_dim:
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
        if route == ROUTE_INPLACE and es == 4:
            # The in-place pass rewrites 16-bit pairs.
            raise _decline(f"operand {operand.value.name}: no in-place pass over fp32")
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

    def exchange_tile(dot: Dot, operand: Operand, at: int) -> tuple[Tcgen05Tile, int]:
        # Every fragment operand goes through the shared tile here, whatever
        # register route the warp mode planned for it.
        assert operand.frag is not None
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
            offset=at,
            source_slot=slot,
        )
        return tile, at + _align(rows * cols * es, SLAB_ALIGNMENT)

    # Loop operand tiles share the ring; epilogue operand tiles and the
    # exchange tiles (loop and epilogue fragment operands) live once in the
    # fixed region after the ring.
    for dot in program.dots_in("loop"):
        for operand in (dot.a, dot.b):
            tile = by_key.get(operand.chain_key)
            if tile is None:
                if operand.kind == "load":
                    tile, stage_offset = load_tile(
                        dot, operand, REGION_LOOP, stage_offset
                    )
                    loop_tiles.append(tile)
                else:
                    tile, fixed = exchange_tile(dot, operand, fixed)
                    exchange_tiles.append(tile)
                by_key[operand.chain_key] = tile
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
                tile, fixed = exchange_tile(dot, operand, fixed)
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
    # Loop stores through double-buffered staging tiles; the epilogue
    # stores through one staging tile (reusing the ring when it fits).
    staged: list[Store] = []
    staging_offsets: dict[Node, int] = {}
    store_tma: dict[Node, TmaSpec] = {}
    for store in program.stores:
        slot = slots[program.frag_slots[store.frag]]
        if store_mode != MMA_DAG_STORE_STAGED:
            continue
        if store.region == "loop" and store.source == "carry_in":
            reason = staged_store_reason(env, store, slot.cols)
        else:
            reason = epilogue_staged_reason(env, store, slot.cols)
        if reason is not None:
            continue
        staged.append(store)
        if store.region == "loop":
            # Double-buffered by iteration parity: a buffer drains behind
            # the barrier that follows its stage-in (a body store of the
            # first block, after the batch issue), and the next write to it
            # comes two iterations later, past at least one more barrier.
            staging_offsets[store.access.node] = fixed_offset + fixed
            fixed += 2 * slot.rows * slot.cols * store.access.dtype.itemsize
            # Without store warps the epilogue warps drain the packets on
            # the chain (59 of 259 us on fwd_h_delta at s2); the staging
            # tile's swizzle is TMA's for rows of one atom, so one thread
            # stores the whole tile by TMA instead.  Store warps drain off
            # the chain already, and the TMA issue would only add to it.
            row_bytes = slot.cols * store.access.dtype.itemsize
            # A bulk store completes at the kernel's end only (the in-loop
            # waits order the staging reads, not the global writes): any
            # other access of the storage would race it.  The matcher
            # declines programs that load a storage they store, so this is
            # about a second store to it -- the epilogue storing the same
            # tile.
            storage = _storage(store.access)
            aliased = sum(1 for a in program.accesses if _storage(a) == storage) > 1
            if (
                (warps == TCGEN05_EPILOGUE_WARPS or pass_warps)
                and not aliased
                and (row_bytes in (16, 32, 64) or row_bytes % ATOM_MAX_BYTES == 0)
                and len(_tma_dims(store.access)) <= TMA_MAX_RANK
                and len(tma) < TMA_MAX_MAPS
                # Rows wider than one atom are staged as 128-byte slabs
                # (``staging_address``): one box per slab; the swizzle
                # pattern needs a 1024-byte aligned base.
                and staging_offsets[store.access.node] % SLAB_ALIGNMENT == 0
            ):
                atom = min(row_bytes, ATOM_MAX_BYTES)
                rows_axis, cols_axis = program.slots[program.frag_slots[store.frag]]
                store_tma[store.access.node] = tma_spec(
                    store.access,
                    rows_axis,
                    cols_axis,
                    slot.rows,
                    slot.cols,
                    store.access.dtype.itemsize,
                    _SWIZZLE_BY_ROW_BYTES.get(atom, "none"),
                    atom,
                    staging_offsets[store.access.node],
                )
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
        # A persistent launch prefetches the next tile's first ring slot under
        # this epilogue: the staging tile overlays the ring's last bytes, so
        # the slot's leading TMA boxes can land while it is read.
        epilogue_staging_offset = (
            (fixed_offset - epilogue_staging) // SLAB_ALIGNMENT * SLAB_ALIGNMENT
            if persistent
            else 0
        )
    else:
        epilogue_staging_offset = fixed_offset + fixed
        fixed += epilogue_staging
    general = program_order_loop(program)
    smem = (
        fixed_offset
        + fixed
        + 8 * (stages + 3 + int(general) + (2 * stages if pass_warps else 0))
        + 16
    )
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
        epilogue_staging_bytes=epilogue_staging,
        tmem_cols=tcgen05_tmem_cols(column),
        smem_bytes=fixed_offset + fixed,
        store_tma=store_tma,
        pass_warps=pass_warps,
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


def produced_nodes(dots: Sequence[Dot]) -> set[Node]:
    """The result nodes of the dots of a batch."""
    return {dot.node for dot in dots}


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
    if tc.regions:
        from .cute_mma_dag_tcgen05_regions import _Tcgen05RegionEmitter

        emitter: _Tcgen05Emitter = _Tcgen05RegionEmitter(cg, plan)
    else:
        emitter = _Tcgen05Emitter(cg, plan)
    source = emitter.render()
    if plan.persistent:
        # The generic persistent program ids wrap the body in the tile loop:
        # the barrier / tensor-memory setup goes ahead of it (the preamble)
        # and the dealloc after it (a post-loop statement).
        lines = source.split("\n")
        setup = "\n".join(lines[: emitter.setup_end])
        body = "\n".join(lines[emitter.setup_end : emitter.teardown_start])
        teardown = "\n".join(lines[emitter.teardown_start :])
        df.preamble.extend(ast.parse(textwrap.dedent(setup)).body)
        for statement in ast.parse(textwrap.dedent(body)).body:
            cg.add_statement(statement)
        post_loop = ast.parse(textwrap.dedent(teardown)).body
        for statement in post_loop:
            cg.add_statement(statement)
        df.cute_state.register_tcgen05_post_loop_stmts(post_loop)
    else:
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
    cg.cute_wrapper_plans.append(mma_dag_alias_plan(df, plan.program))
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

    def evaluate(self, node: Node, coords: Coords) -> str:
        em = cast("_Tcgen05Emitter", self.emitter)
        if em.table_recording is not None:
            if node in em.table_nodes:
                em.table_recording.append((node, coords))
                return "0.0"
        elif em.table_lookup and not em.computing_tables:
            register = em.table_regs.get((node, coords))
            if register is not None:
                return register
        lane_value = em.row_lanes.get(node)
        if lane_value is not None:
            # A row-only value of the pass at a packet round's row: the lane
            # holding that row computed it once (``emit_row_lanes``).
            for coord in coords:
                found = em.round_of.get(coord)
                if found is None:
                    continue
                key = (node, coords)
                cached = self._lookup(key)
                if cached is not None:
                    return cached
                first, cpr = found
                result = em.bind(
                    "md_v",
                    f"cute.arch.shuffle_sync({lane_value}, {first} + md_lane // {cpr})",
                )
                self.scopes[-1][key] = result
                return result
        return super().evaluate(node, coords)

    def _load(self, node: Node, coords: Coords) -> str:
        em = cast("_Tcgen05Emitter", self.emitter)
        access = em.accesses[node]
        if len(access.tile_dims) == 2:
            # A tile leaf: nothing to record (the scratch render has none).
            if em.table_recording is not None:
                return "0.0"
            return super()._load(node, coords)
        key = (node, coords)
        if em.computing_tables:
            # The table's vectors: registers loaded an iteration earlier.
            if em.table_load_recording is not None:
                em.table_load_recording.append(key)
            else:
                register = em.table_loads.get(key)
                if register is not None:
                    return em.bind("md_v", em.typed(register, access.dtype))
            return super()._load(node, coords)
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
        # The warp emitter's loop state its ``bind`` / raw-ring helpers
        # consult: this mode's single loop is ``define_iteration_bases``'s
        # ``in_loop`` form and hoists its side loads through ``prefetched``.
        self.values = {}
        self.raw_stages = {}
        # Line indices splitting a persistent launch's render into the
        # once-per-CTA setup, the tile body and the teardown.
        self.setup_end = 0
        self.teardown_start = 0
        # The in-place pass being rendered: its row-only values, one row per
        # lane (node -> register), and the packet rounds' row names -> the
        # first lane holding their rows and the chunks per row.
        self.row_lanes: dict[Node, str] = {}
        self.round_of: dict[str, tuple[int, int]] = {}
        # A persistent launch that issues the next tile's first ring slots
        # under this tile's epilogue (``plan_cross_tile``): the ring phase
        # base (iterations of the CTA's earlier tiles), the next tile's
        # virtual program id, and each tile coordinate name -> its value at
        # the next tile.
        self.phase_base: str | None = None
        self.next_pid: str | None = None
        self.next_coords: dict[str, str] = {}
        self.hoists = []
        self.name_depth = {}
        self.stored_storages = {
            _storage(store.access)
            for store in (*self.program.stores, *self.program.vec_stores)
        }
        self.overlaid_keys = set()
        self.overlaid_tiles = set()
        # This mode renders its passes element by element.
        self.pairs = False
        self.packable_memo = {}
        self.redirect = None
        self.scope_path = []
        self.scope_count = 0
        self.lane_bases = {}
        # This mode spells every address out (its own addressing is TMA).
        self.fold_addresses = False
        self.early_prologues = {}
        self.early_issued = set()
        self.evaluator: _Evaluator = _PassEvaluator(self)
        self.es = self.program.elem_bytes
        self.kind = "bf16" if self.program.dtype == torch.bfloat16 else "f16"
        self.sync = "cute.arch.barrier()"
        # The loop body's thread count: the epilogue warps alone when the
        # pass warps run their own loop.
        self.loop_threads = (
            32 * TCGEN05_EPILOGUE_WARPS if tc.pass_warps else plan.threads
        )
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
        # Iteration-ahead tables: pointwise values over the iteration's
        # vectors alone (a decay factor), computed one iteration ahead into
        # registers at the coordinates this thread evaluates them at --
        # (table node, site value, slot, swap) and (node, coords) -> register.
        self.table_nodes: dict[Node, int] = {}
        self.table_regs: dict[tuple[Node, Coords], str] = {}
        self.table_recording: list[tuple[Node, Coords]] | None = None
        self.computing_tables = False
        # The vector elements the tables read, (load node, coords) ->
        # register: loaded two iterations ahead (the compute runs under the
        # MMAs of the iteration before, a global load's latency exceeds it).
        self.table_loads: dict[tuple[Node, Coords], str] = {}
        self.table_load_recording: list[tuple[Node, Coords]] | None = None
        # Entry -> the window (a batch issue, or the iteration end = the
        # batch count) its next-iteration value is computed in: after the
        # last consumer of its table in the iteration; the loads reload in
        # the last window used.  Lookups serve the loop body only.
        self.table_entry_window: dict[tuple[Node, Coords], int] = {}
        self.table_reload_window = 0
        self.table_lookup = False
        # The epilogue's side loads with per-thread coordinates, loaded into
        # ``md_ex{i}`` under the last iteration's MMAs (or after the loop).
        self.epilogue_sites: list[tuple[Node, Coords]] = []
        self.epilogue_sites_in_loop = False
        # Epilogue commits issued so far (eacc phases).
        self.epilogue_commits = 0
        self.smem_bytes = tc.smem_bytes
        self.dot_index = {dot.node: i for i, dot in enumerate(self.program.dots)}
        # Register arrays holding a slot's row in the current block: the
        # tcgen05.ld vectors, or the arrays a derived value / rescale built.
        self.slot_regs: dict[int, str] = {}
        # Prefetched pair words of the derived values' tile inputs in the
        # current iteration: (load node, column) -> (word,) or (lo, hi).
        self.derived_words: dict[tuple[Node, int], tuple[str, ...]] = {}

    # -- fragment element model -------------------------------------------------

    def slot(self, index: int) -> SlotGeometry:
        return self.tc.slots[index]

    def row_var(self, slot: int) -> str:
        return f"md_row{self.slot(slot).rows}"

    def half_rows(self, slot: int) -> bool:
        return self.slot(slot).rows == 64

    def half_cols(self, slot: int) -> bool:
        """A 64-row fragment is read with ``tcgen05.ld.16x32bx2``: lanes 0-15
        of a warp hold the first half of their row's columns, lanes 16-31
        the second (``md_cb{slot}`` is the lane's first column), so every
        lane carries data (the ``32x32b`` shape leaves lanes 16-31 idle).
        The half must be a power of two of at least one 16-byte packet."""
        geometry = self.slot(slot)
        half = geometry.cols // 2
        return (
            geometry.rows == 64
            and geometry.cols >= 16
            and half & (half - 1) == 0
            and half <= 2 * TCGEN05_LD_COLS
        )

    def local_cols(self, slot: int) -> int:
        """Columns of the fragment this thread holds."""
        geometry = self.slot(slot)
        return geometry.cols // 2 if self.half_cols(slot) else geometry.cols

    def col_expr(self, slot: int, j: int) -> str:
        """The fragment column of this thread's local column ``j``."""
        return f"(md_cb{slot} + {j})" if self.half_cols(slot) else str(j)

    def chunk_expr(self, slot: int, chunk: int, per_packet: int) -> str:
        """The 16-byte chunk of the row holding this thread's local packet
        ``chunk`` (``per_packet`` elements per chunk)."""
        if self.half_cols(slot):
            return f"(md_cb{slot} // {per_packet} + {chunk})"
        return str(chunk)

    def ld_chunks(self, slot: int) -> int:
        return -(-self.local_cols(slot) // TCGEN05_LD_COLS)

    def chunk_cols(self, slot: int, chunk: int) -> int:
        return min(TCGEN05_LD_COLS, self.local_cols(slot) - chunk * TCGEN05_LD_COLS)

    def ld_shape(self, slot: int) -> str:
        return '"16x32bx2"' if self.half_cols(slot) else '"32x32b"'

    def ld_offset(self, slot: int) -> str:
        """The half split-off keyword of the 16x32bx2 shape (empty otherwise)."""
        if self.half_cols(slot):
            return f", offset={self.slot(slot).cols // 2}"
        return ""

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
        guard = self.half_rows(slot) and not self.half_cols(slot)
        return self.block("if md_rowok:" if guard else None)

    def reg(self, slot: int, *key: object) -> str:  # type: ignore[override]
        chunk, j = key
        prefix = self.slot_regs.get(slot, f"{self.vec}{slot}_")
        return f"{prefix}{chunk}[{j}]"

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
        column = self.col_expr(slot, col)
        first, second = (column, row) if swap else (row, column)
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
        if tile.swizzle == ATOM_32B_SWIZZLE:
            # 32-byte units XOR the row mod 4: the 16-byte chunk index
            # XORs with twice it.
            swz = f"(({local}) ^ ((({prow}) % 4) * 2))"
        else:
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

    def packet_rounds(
        self, count: int, threads: int, idx_base: str, cpr: int, tag: str
    ) -> list[tuple[str, str, str | None]]:
        """``(prow, pch, guard)`` of every packet round of a ``count``-packet
        tile over ``threads`` threads, the row / chunk names fixed per round
        (``packets`` with a tag) and defined up front; only the last round
        can be partial (its guard)."""
        rounds: list[tuple[str, str, str | None]] = []
        for rep in range(-(-count // threads)):
            idx = f"({rep * threads} + {idx_base})" if rep else idx_base
            guard = f"{idx} < {count}" if count < (rep + 1) * threads else None
            prow, pch = f"md_pr_{tag}_{rep}", f"md_pc_{tag}_{rep}"
            self.emit(f"{prow} = {idx} // {cpr}")
            self.emit(f"{pch} = {idx} % {cpr}")
            self.row_exprs[prow] = (f"({idx} // {cpr})", guard)
            rounds.append((prow, pch, guard))
        return rounds

    def emit_pipelined_pass(
        self,
        rounds: Sequence[tuple[str, str, str | None]],
        load: object,
        rewrite: object,
    ) -> None:
        """The packet rounds of an in-place / convert pass as every full
        round's ``load(prow, pch) -> state`` first, then every round's
        ``rewrite(state, prow, pch)``: the compiler cannot tell the packets
        of one shared tile apart, so a store ahead of the next round's load
        serializes the rounds on the shared-memory latency (the gated
        fwd_h_delta pass cost 156 of 434 us at s2 that way), in groups of
        ``TCGEN05_PASS_ROUNDS`` rounds (the live words).  A partial last
        round runs under its guard, load and rewrite together."""
        full = [r for r in rounds if r[2] is None]
        for start in range(0, len(full), TCGEN05_PASS_ROUNDS):
            group = full[start : start + TCGEN05_PASS_ROUNDS]
            states = [load(prow, pch) for prow, pch, _guard in group]  # type: ignore[operator]
            for state, (prow, pch, _guard) in zip(states, group, strict=True):
                self.evaluator.push()
                rewrite(state, prow, pch)  # type: ignore[operator]
                self.evaluator.pop()
        for prow, pch, guard in rounds[len(full) :]:
            with self.nest(f"if {guard}:"):
                self.evaluator.push()
                rewrite(load(prow, pch), prow, pch)  # type: ignore[operator]
                self.evaluator.pop()

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
                if subscript.offset is not None:
                    value = f"{value} + {self.uniform(subscript.offset, {})}"
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
                f"{self.pack(values[0], values[1], operand.dtype)})"
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
        # The tile's own dtype: a region program mixes dots (wy: a bf16 kk
        # dot ahead of the tf32 chain), so the program-wide kind is not it.
        unpack = "unpack_f16x2" if load.dtype == torch.float16 else "unpack_bf16x2"
        value = self.pass_value(operand)
        self.hoist_scalars(value)

        def packet_load(prow: str, pch: str) -> tuple[str, list[str]]:
            address = self.bind("md_pa", self.tile_address(tile, stage, prow, pch))
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({address})")
            return address, words

        def rewrite(state: tuple[str, list[str]], prow: str, pch: str) -> None:
            address, words = state
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
                packed.append(self.pack(values[0], values[1], operand.dtype))
            self.emit(f"{_RUNTIME}.sts_v4_b32({address}, {', '.join(packed)})")

        cpr = tile.chunks_per_row
        rounds = self.packet_rounds(tile.rows * cpr, threads, idx_base, cpr, tile.tag)
        self.emit_row_lanes(tile, value, rounds, idx_base, threads)
        self.emit_pipelined_pass(rounds, packet_load, rewrite)
        self.row_lanes, self.round_of = {}, {}

    def row_only_roots(
        self, tile: Tcgen05Tile, value: Node
    ) -> list[tuple[Node, int, int]]:
        """The maximal sub-values of the pass value ``value`` that vary only
        with the shared tile's row (a decay of the K position: loads of side
        vectors along that axis and uniform scalars, never the tile itself):
        ``(node, row dim, rank)``."""
        from ...language import memory_ops

        operand = tile.operand
        source = operand.load.node if operand.load is not None else None
        probe = self.tile_coords(operand, "md_probe_row", "md_probe_col")
        if source is None or "md_probe_row" not in probe:
            return []
        row_block = self.env.resolve_block_id(
            _shape(operand.value)[probe.index("md_probe_row")]
        )
        if row_block is None:
            return []
        row_block = self.env.canonical_block_id(row_block)
        free: dict[Node, tuple[bool, bool]] = {}

        def facts(node: Node) -> tuple[bool, bool]:
            """(free of the tile, reads a side load)."""
            if node not in free:
                if node is source:
                    free[node] = (False, True)
                else:
                    inputs = [facts(n) for n in node.all_input_nodes]
                    free[node] = (
                        all(f for f, _ in inputs),
                        _is_call(node, memory_ops.load) or any(r for _, r in inputs),
                    )
            return free[node]

        roots: list[tuple[Node, int, int]] = []
        seen: set[Node] = set()

        def visit(node: Node) -> None:
            if node in seen or node.op != "call_function":
                return
            seen.add(node)
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                shape = _shape(node)
                dims = [
                    i
                    for i, size in enumerate(shape)
                    if not (isinstance(size, int) and size == 1)
                ]
                if len(dims) == 1 and all(facts(node)):
                    block = self.env.resolve_block_id(shape[dims[0]])
                    if (
                        block is not None
                        and self.env.canonical_block_id(block) == row_block
                    ):
                        roots.append((node, dims[0], len(shape)))
                        return
            for inp in node.all_input_nodes:
                visit(inp)

        visit(value)
        return roots

    def emit_row_lanes(
        self,
        tile: Tcgen05Tile,
        value: Node,
        rounds: Sequence[tuple[str, str, str | None]],
        idx_base: str,
        threads: int,
    ) -> None:
        """Evaluate the pass value's row-only sub-values once per row: a
        warp's packet round ``j`` covers ``32 / cpr`` consecutive rows, so
        with at most 32 rows over all its rounds lane ``l`` computes the row
        of round ``l // (32 / cpr)``, slot ``l % (32 / cpr)`` (its side
        loads prefetched like the packets'), and every packet reads it with
        one shuffle.  Same values (the same expression at the same row),
        ``cpr`` times fewer side loads and transcendentals."""
        cpr = tile.chunks_per_row
        count = len(rounds)
        if (
            threads % 32
            or 32 % cpr
            or any(guard is not None for _prow, _pch, guard in rounds)
        ):
            return
        per = 32 // cpr
        if count * per > 32:
            return
        roots = self.row_only_roots(tile, value)
        if not roots:
            return
        name = f"md_lr_{tile.tag}"
        expr = (
            f"(md_lane // {per}) % {count} * {threads // cpr} "
            f"+ ({idx_base}) // 32 * {32 // cpr} + md_lane % {per}"
        )
        self.emit(f"{name} = {expr}")
        self.row_exprs[name] = (f"({expr})", None)
        for node, dim, rank in roots:
            coords = tuple(name if d == dim else "0" for d in range(rank))
            self.row_lanes[node] = self.evaluator.evaluate(node, coords)
        for rep, (prow, _pch, _guard) in enumerate(rounds):
            self.round_of[prow] = (rep * per, cpr)

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

        def packet_load(prow: str, pch: str) -> list[str]:
            raws = [self.df.new_var("md_f") for _ in range(4)]
            self.emit(
                f"{', '.join(raws)} = {_RT}.lds_v4_f32({self.raw_address(tile, stage, prow, pch)})"
            )
            return raws

        def rewrite(raws: list[str], prow: str, pch: str) -> None:
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
                packed.append(self.pack(pair[0], pair[1], operand.dtype))
            # Raw packet p holds columns 4p..4p+3: 16-bit chunk p // 2, byte
            # offset 8 * (p % 2) within it.
            chunk = f"({pch} // 2)"
            self.emit(
                f"{_RUNTIME}.sts_v2_b32({self.tile_address(tile, stage, prow, chunk)} "
                f"+ ({pch} % 2) * {4 * es}, {', '.join(packed)})"
            )

        self.emit_pipelined_pass(
            self.packet_rounds(
                tile.rows * raw_cpr, threads, idx_base, raw_cpr, tile.tag
            ),
            packet_load,
            rewrite,
        )

    def emit_passes(
        self,
        tiles: Sequence[Tcgen05Tile],
        stage: str,
        wait: str | None,
        idx_base: str,
        threads: int,
        fence: bool = True,
    ) -> None:
        """Every pass of ``tiles`` by the ``threads`` threads with local
        index ``idx_base``: ``wait`` (the TMA of in-place rewrites), the
        passes, then publish the generic writes to the async proxy (unless
        a later fence of the same threads precedes the reading batch)."""
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
        if fence:
            self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')

    def has_passes(self, tiles: Sequence[Tcgen05Tile]) -> bool:
        return any(
            tile.route in (ROUTE_SYNC, ROUTE_INPLACE, ROUTE_CONVERT) for tile in tiles
        )

    def late_pass_windows(
        self, blocks: Sequence[tuple[list[Derived | Store], list[Dot]]]
    ) -> dict[tuple[object, ...], int]:
        """Loop pass tiles no dot of the first batch reads: tile key -> the
        window (batch index) after whose issue the pass runs, the one
        before the tile's first reading batch."""
        first: dict[tuple[object, ...], int] = {}
        batch = 0
        for _steps, dots in blocks:
            if not dots:
                continue
            for dot in dots:
                for operand in (dot.a, dot.b):
                    if operand.kind != "load":
                        continue
                    tile = self.tc.tile(dot, operand)
                    if tile.route in (ROUTE_SYNC, ROUTE_INPLACE, ROUTE_CONVERT):
                        first.setdefault(tile.key, batch)
            batch += 1
        return {key: b - 1 for key, b in first.items() if b > 0}

    # -- iteration-ahead tables ---------------------------------------------------

    def iteration_only(self, node: Node) -> bool:
        """Whether ``node``'s value depends on the iteration's vectors and
        scalars alone (no fragment, no 2-D tile), so it is the same for
        every thread evaluating it at a coordinate and can be computed
        ahead."""
        from ...language import _tracing_ops
        from ...language import memory_ops

        program = self.program
        memo: dict[Node, bool] = {}
        stored = {
            _storage(store.access) for store in (*program.stores, *program.vec_stores)
        }

        def visit(current: Node) -> bool:
            if current in memo:
                return memo[current]
            if current in program.aliases:
                result = visit(program.aliases[current])
            elif (
                current in program.frag_slots
                or current in program.vec_leaves
                or current.op != "call_function"
            ):
                result = False
            elif current.target is memory_ops.load:
                # A vector of a storage the program never writes (it is
                # read two iterations ahead; the matcher declines programs
                # loading a stored storage -- defense in depth).
                access = self.accesses.get(current)
                result = (
                    access is not None
                    and len(access.tile_dims) <= 1
                    and _storage(access) not in stored
                )
            elif current.target in (
                _tracing_ops._get_symnode,
                torch.ops.prims.iota.default,
                torch.ops.aten.scalar_tensor.default,
            ):
                result = True
            else:
                result = all(visit(arg) for arg in current.all_input_nodes)
            memo[current] = result
            return result

        return visit(node)

    def table_roots(self, value: Node) -> list[Node]:
        """The maximal iteration-only sub-values of ``value`` holding an
        exponential over a loop-dependent vector."""
        from ...language import memory_ops

        program = self.program
        roots: list[Node] = []
        seen: set[Node] = set()

        def worth(current: Node) -> tuple[bool, bool]:
            """(has an exp, reads a loop-dependent vector)."""
            exp = current.op == "call_function" and current.target in (
                torch.ops.aten.exp2.default,
                torch.ops.aten.exp.default,
            )
            loop = (
                current.op == "call_function"
                and current.target is memory_ops.load
                and current in self.accesses
                and self.loop_dependent(self.accesses[current])
            )
            for arg in current.all_input_nodes:
                e, lo = worth(arg)
                exp, loop = exp or e, loop or lo
            return exp, loop

        def walk(current: Node, root: bool = False) -> None:
            if current in seen:
                return
            seen.add(current)
            if current in program.aliases:
                walk(program.aliases[current], root)
                return
            if self.iteration_only(current):
                if all(worth(current)):
                    roots.append(current)
                return
            if not root and (
                current in program.frag_slots or current in program.vec_leaves
            ):
                return  # a register: nothing under it is evaluated
            for arg in current.all_input_nodes:
                walk(arg)

        # The value itself may be a derived fragment: its definition.
        walk(value, root=True)
        return roots

    def table_sites(self) -> list[tuple[Node, int, bool, bool]]:
        """The loop's per-element value sites: (value, slot, swap, derived)
        of the exchanged fragment operands and the derived steps."""
        program = self.program
        sites: list[tuple[Node, int, bool, bool]] = []
        for dot in program.dots_in("loop"):
            for operand in (dot.a, dot.b):
                if operand.kind == "frag" and operand.pointwise:
                    assert operand.frag is not None
                    sites.append(
                        (
                            self.pass_value(operand),
                            program.frag_slots[operand.frag],
                            operand.swap,
                            False,
                        )
                    )
        for step in program.derived:
            if step.loop >= 0:
                sites.append((step.node, step.slot, False, True))
        return sites

    def discover_tables(self) -> None:
        """Render the loop's value sites once into scratch, recording the
        coordinates each table is evaluated at by this thread: one register
        per (table, coordinates)."""
        sites = self.table_sites()
        for value, _slot, _swap, _derived in sites:
            for table in self.table_roots(value):
                self.table_nodes.setdefault(table, len(self.table_nodes))
        if not self.table_nodes:
            return
        saved = (
            self.lines,
            self.indent,
            self.iter_vars,
            self.values,
            self.evaluator,
            self.deferred,
        )
        self.lines, self.indent, self.deferred = [], "", []
        self.define_iteration_bases(in_loop=True)
        self.evaluator = _PassEvaluator(self)
        self.table_recording = []
        for value, slot, swap, derived in sites:
            self.hoist_scalars(value)
            for c in range(self.local_cols(slot)):
                self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                self.evaluator.element = c % 2
                coords = self.element_coords(value, slot, c, swap)
                if derived:
                    self.evaluator.define(value, coords)
                else:
                    self.evaluator.evaluate(value, coords)
        for table, coords in self.table_recording:
            key = (table, coords)
            if key not in self.table_regs:
                index = self.table_nodes[table]
                self.table_regs[key] = f"md_tb{index}_{len(self.table_regs)}"
        self.table_recording = None
        # The compute's vector loads, from a scratch render of it.
        self.table_load_recording = []
        self.emit_table_compute("md_i")
        for key in self.table_load_recording:
            if key not in self.table_loads:
                self.table_loads[key] = f"md_tl{len(self.table_loads)}"
        self.table_load_recording = None
        (
            self.lines,
            self.indent,
            self.iter_vars,
            self.values,
            self.evaluator,
            self.deferred,
        ) = saved
        if len(self.table_regs) + len(self.table_loads) > TCGEN05_MAX_TABLE_REGS:
            self.table_nodes, self.table_regs, self.table_loads = {}, {}, {}
            return
        self.assign_table_windows()

    def consumer_windows(self) -> tuple[list[tuple[Node, int]], int]:
        """The loop body's evaluated values with the first window after
        them, and the batch count (window index of the iteration end)."""
        program = self.program
        consumers: list[tuple[Node, int]] = []

        def dot_values(dot: Dot) -> list[Node]:
            values = [
                self.pass_value(operand)
                for operand in (dot.a, dot.b)
                if operand.kind == "frag"
            ]
            if dot.acc is not None and self.prepares(dot):
                values.append(dot.acc)
            return values

        if self.general_loop:
            batches = 0
            blocks = self.loop_blocks()
            for index, (steps, dots) in enumerate(blocks):
                window = batches if dots else sum(1 for _s, d in blocks if d)
                if index == 0:
                    for store in program.stores:
                        if store.source == "carry_in":
                            consumers.append((store.value, window))
                for step in steps:
                    value = step.node if isinstance(step, Derived) else step.value
                    consumers.append((value, window))
                for dot in dots:
                    consumers.extend((value, window) for value in dot_values(dot))
                if dots:
                    batches += 1
            return consumers, batches
        for store in program.stores:
            consumers.append((store.value, 1 if store.source == "body" else 0))
        for dot in program.dots_in("loop"):
            consumers.extend((value, 0) for value in dot_values(dot))
        for step in program.derived:
            if step.loop >= 0:
                consumers.append((step.node, 0))
        return consumers, 1

    def assign_table_windows(self) -> None:
        """Spread each table's entries over the windows after its last
        consumer (the iteration end when none follows)."""
        program = self.program
        consumers, batches = self.consumer_windows()
        memo: dict[Node, set[Node]] = {}

        def tables_under(node: Node) -> set[Node]:
            """The tables an evaluation of ``node`` reaches (a fragment leaf
            below it is a register: nothing under it is evaluated)."""
            if node in memo:
                return memo[node]
            memo[node] = set()
            found: set[Node] = set()
            if node in self.table_nodes:
                found.add(node)
            if node in program.aliases:
                found |= tables_under(program.aliases[node])
            elif node not in program.frag_slots and node not in program.vec_leaves:
                for arg in node.all_input_nodes:
                    found |= tables_under(arg)
            memo[node] = found
            return found

        ready: dict[Node, int] = dict.fromkeys(self.table_nodes, 0)
        for value, window in consumers:
            under = set(tables_under(program.aliases.get(value, value)))
            if value in program.frag_slots:
                # A derived step: its own definition is evaluated.
                for arg in value.all_input_nodes:
                    under |= tables_under(arg)
            for table in under:
                ready[table] = max(ready[table], window)
        counts: dict[Node, int] = dict.fromkeys(self.table_nodes, 0)
        for table, coords in self.table_regs:
            first = ready[table]
            avail = list(range(first, batches)) or [batches]
            self.table_entry_window[(table, coords)] = avail[counts[table] % len(avail)]
            counts[table] += 1
        self.table_reload_window = max(self.table_entry_window.values())

    def emit_table_loads(self, loop_value: str) -> None:
        """Load the tables' vector elements of iteration ``loop_value`` into
        their registers."""
        saved = self.iter_vars
        self.iter_vars = {}
        for (node, coords), register in self.table_loads.items():
            access = self.accesses[node]
            if node not in self.iter_vars:
                self.iter_vars[node] = self.iteration_base(access, loop_value)
            load = self.scalar_load(self.address(access, coords), access.dtype)
            self.emit(f"{register} = {load}")
        self.iter_vars = saved

    def emit_table_compute(self, loop_value: str, window: int | None = None) -> None:
        """This thread's table entries (those of ``window``, or all) for
        iteration ``loop_value``, from the vectors read at that iteration's
        bases."""
        saved = (self.iter_vars, self.evaluator)
        self.iter_vars = dict(saved[0])
        for access in self.program.accesses:
            if self.loop_dependent(access):
                self.iter_vars[access.node] = self.iteration_base(access, loop_value)
        self.evaluator = _PassEvaluator(self)
        self.computing_tables = True
        entries = [
            (key, register)
            for key, register in self.table_regs.items()
            if window is None or self.table_entry_window[key] == window
        ]
        for table in dict.fromkeys(table for (table, _coords), _r in entries):
            self.hoist_scalars(table)
        for key, register in entries:
            result = self.evaluator.evaluate(*key)
            self.emit(f"{register} = cutlass.Float32({result})")
        self.computing_tables = False
        self.iter_vars, self.evaluator = saved

    def emit_tables_ahead(self, loop: str, end: str, window: int) -> None:
        """The next iteration's table entries of ``window`` (after a batch
        issue, before its wait: under its MMAs); the last window used then
        reloads the vectors for the iteration after."""
        if window not in self.table_entry_window.values():
            return
        with self.nest(f"if {loop} + 1 < {end}:"):
            self.emit_table_compute(f"({loop} + 1)", window)
        if window == self.table_reload_window and self.table_loads:
            with self.nest(f"if {loop} + 2 < {end}:"):
                self.emit_table_loads(f"({loop} + 2)")

    # -- side loads one iteration ahead ------------------------------------------

    def loop_side_work(self) -> None:
        """The per-iteration work whose side loads are prefetched: the loop
        operand passes and the carry rescales."""
        tc = self.tc
        if self.has_passes(tc.loop_tiles) and not tc.pass_warps:
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
            if all(
                c.isdigit()
                or c in rows
                or c in self.row_exprs
                or c.startswith("(md_cb")
                for c in coords
            ):
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

    def discover_epilogue_prefetch_sites(self) -> list[tuple[Node, Coords]]:
        """Render the epilogue once into scratch, recording its loop-invariant
        side loads whose coordinates are fixed per thread (the thread's
        fragment row, a static column of its half-row).  Programs without
        a loop-invariant vector access skip the dry run (their renders are
        untouched).  Loads of a storage the program stores stay in the
        epilogue (hoisted, they would run ahead of the store they follow;
        the matcher declines such programs -- the check is defense in
        depth);
        the row and static sites come first, the half-row's column sites
        all together or not at all (see ``TCGEN05_EPILOGUE_PREFETCH_COLS``)."""
        program = self.program
        if not (program.dots_in("epilogue") or program.stores_in("epilogue")):
            return []
        if not any(
            len(access.tile_dims) == 1 and not self.loop_dependent(access)
            for access in program.accesses
        ):
            return []
        written = {
            _storage(store.access) for store in (*program.stores, *program.vec_stores)
        }
        saved = (
            self.lines,
            self.indent,
            self.iter_vars,
            self.evaluator,
            self.deferred,
            self.vec,
            dict(self.slot_regs),
            dict(self.row_exprs),
            self.epilogue_commits,
        )
        self.lines, self.indent, self.deferred = [], "", []
        self.recording = []
        self.emit_epilogue()
        rows = {f"md_row{m}" for m in TCGEN05_M_CHOICES}
        sites: list[tuple[Node, Coords]] = []
        columns: list[tuple[Node, Coords]] = []
        for key in self.recording:
            node, coords = key
            access = self.accesses[node]
            if (
                key in sites
                or key in columns
                or self.loop_dependent(access)
                or _storage(access) in written
            ):
                continue
            if not all(
                c.isdigit() or c in rows or c.startswith("(md_cb") for c in coords
            ):
                continue
            # A column site reads a tile dim at a static column (``md_cb``
            # plus a local column, or the column itself without half-rows).
            if all(coords[p] in rows for p in access.tile_positions):
                sites.append(key)
            else:
                columns.append(key)
        self.recording = None
        if len(columns) <= TCGEN05_EPILOGUE_PREFETCH_COLS:
            sites.extend(columns)
        (
            self.lines,
            self.indent,
            self.iter_vars,
            self.evaluator,
            self.deferred,
            self.vec,
            self.slot_regs,
            self.row_exprs,
            self.epilogue_commits,
        ) = saved
        return sites

    def emit_epilogue_prefetch_loads(self) -> None:
        """Load the epilogue's side loads (``epilogue_sites``) into their
        ``md_ex{i}`` registers: every thread, the rows and half-row columns
        being valid addresses for every lane; the epilogue's evaluator then
        reads them through ``prefetched``."""
        for index, (node, coords) in enumerate(self.epilogue_sites):
            access = self.accesses[node]
            self.emit(
                f"md_ex{index} = "
                f"{self.scalar_load(self.address(access, coords), access.dtype)}"
            )

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
        shape = self.ld_shape(slot)
        for chunk in range(self.ld_chunks(slot)):
            self.emit(
                f"{self.vec}{slot}_{chunk} = {_PRIMS}.tcgen05_ld({shape}, "
                f"{self.tmem_ptr(slot, chunk * TCGEN05_LD_COLS)}, "
                f"num={self.chunk_cols(slot, chunk)}{self.ld_offset(slot)})"
            )
        self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.LOAD)")
        self.slot_regs[slot] = f"{self.vec}{slot}_"

    def emit_st_array(self, name: str, slot: int, chunk: int) -> None:
        self.emit(
            f"{_PRIMS}.tcgen05_st({self.ld_shape(slot)}, "
            f"{self.tmem_ptr(slot, chunk * TCGEN05_LD_COLS)}, "
            f"{name}[0:{self.chunk_cols(slot, chunk)}]{self.ld_offset(slot)})"
        )

    def emit_carry_init(self) -> None:  # type: ignore[override]
        """Seed the TMEM carries: zeros, or this thread's row of the init tile."""
        with self.block("if md_epi:"):
            for carry in self.program.carries:
                slot = carry.slot
                self.slot(slot)
                init = carry.init
                positions: list[int] = []
                paired = False
                if init is not None:
                    self.iter_vars[init.node] = self.base_vars[init.node]
                    positions = list(init.tile_positions)
                    paired = self.pairs_aligned(init, 1)
                for chunk in range(self.ld_chunks(slot)):
                    cols = self.chunk_cols(slot, chunk)
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
                                coords[positions[1]] = self.col_expr(slot, col)
                                address = self.address(init, tuple(coords))
                                lo, hi = self.pair_load(address, init.dtype, paired)
                                self.emit(f"{name}[{j}] = cutlass.Float32({lo})")
                                self.emit(f"{name}[{j + 1}] = cutlass.Float32({hi})")
                    self.emit_st_array(name, slot, chunk)
            self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")

    def emit_rescale(
        self, dot: Dot, loaded: set[int], prefix: str | None = None
    ) -> None:
        """``dot.acc`` is a pointwise DAG over fragment ``acc_slot`` (or that
        fragment headed for another slot): evaluate it per element of the
        loaded row (``loaded`` names the slots already in registers) and
        store the result into the dot's slot.  With ``prefix`` the arrays
        are named ``{prefix}{chunk}`` and become the slot's register source
        for the rest of the block."""
        acc = dot.acc
        assert acc is not None and dot.acc_slot is not None
        source, target = dot.acc_slot, dot.slot
        self.hoist_scalars(acc)
        if source not in loaded:
            self.emit_ld(source)
            loaded.add(source)
        for chunk in range(self.ld_chunks(target)):
            cols = self.chunk_cols(target, chunk)
            if prefix is None:
                name = self.bind(
                    "md_rs", f"cutlass.Array(cutlass.Float32, {cols}, alignment=16)"
                )
            else:
                name = f"{prefix}{chunk}"
                self.emit(
                    f"{name} = cutlass.Array(cutlass.Float32, {cols}, alignment=16)"
                )
            for j in range(cols):
                self.evaluator.reg = (chunk, j)  # type: ignore[assignment]
                coords = self.element_coords(acc, source, chunk * TCGEN05_LD_COLS + j)
                value = self.evaluator.evaluate(acc, coords)
                self.emit(f"{name}[{j}] = cutlass.Float32({value})")
            self.evaluator.reg = None
            self.emit_st_array(name, target, chunk)
        self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")
        if prefix is not None:
            self.slot_regs[target] = prefix
            loaded.add(target)

    def derived_pair_address(self, step: Derived, access: Access, col: int) -> str:
        """Address of the pair ``(col, col + 1)`` of the derived value's
        2-D tile input in this thread's row."""
        coords = ["0"] * len(access.dims)
        positions = access.tile_positions
        coords[positions[0]] = self.row_var(step.slot)
        coords[positions[1]] = self.col_expr(step.slot, col)
        return self.address(access, tuple(coords))

    def emit_derived_prefetch(self, step: Derived) -> None:
        """Issue this iteration's pair loads of the derived value's 2-D tile
        inputs into registers at the iteration top (they land while the
        carry is read and the first batch runs)."""
        for access in step.loads:
            self.emit_derived_access_prefetch(step, access)

    def emit_derived_access_prefetch(self, step: Derived, access: Access) -> None:
        """The pair loads of one 2-D tile input of ``step`` (this thread's
        row at the fragment coordinates, straight from global memory)."""
        paired = self.pairs_aligned(access, 1)
        for col in range(0, self.local_cols(step.slot), 2):
            address = self.derived_pair_address(step, access, col)
            if paired and access.dtype != torch.float32:
                word = self.bind("md_u", f"{_RUNTIME}.ldg_b32({address})")
                self.derived_words[(access.node, col)] = (word,)
            else:
                lo, hi = self.pair_load(address, access.dtype, paired)
                self.derived_words[(access.node, col)] = (lo, hi)

    def emit_derived_line_prefetch(self, loop_value: str) -> None:
        """``prefetch.global.L2`` of the rows the derived values' tile inputs
        read in iteration ``loop_value`` (thread ``t`` takes row ``t``): the
        register prefetch at the iteration top runs one iteration ahead,
        less than the DRAM latency of a first touch."""
        for step in self.program.derived:
            if step.loop < 0:
                continue
            geometry = self.slot(step.slot)
            for access in step.loads:
                base = self.iteration_base(access, loop_value)
                es = access.dtype.itemsize
                row_stride = access.stride_at(access.tile_positions[0]) * es
                with self.nest(f"if md_tidx < {geometry.rows}:"):
                    for offset in range(0, geometry.cols * es, 128):
                        line = f"{base} + cutlass.Int64(md_tidx) * {row_stride}"
                        if offset:
                            line = f"{line} + {offset}"
                        self.emit(
                            f"{_PRIMS}.prefetch_l2(cutlass.inttoptr({line}, 1, cutlass.Uint8))"
                        )

    def derived_needs_tmem(self, step: Derived) -> bool:
        """Whether the derived value must be written to its slot: the loop
        output (the next iteration reads it), an accumulator a later dot
        continues in tensor memory, or a reader in a later block (the
        registers do not cross the block)."""
        program = self.program
        if any(carry.output is step.node for carry in program.carries):
            return True
        blocks = self.loop_blocks()
        home = next(i for i, (steps, _dots) in enumerate(blocks) if step in steps)
        for index, (steps, dots) in enumerate(blocks):
            for dot in dots:
                if dot.acc_frag is step.node and (
                    dot.slot == dot.acc_slot or index != home
                ):
                    return True
                if index != home and any(
                    operand.frag is step.node for operand in (dot.a, dot.b)
                ):
                    return True
            if index != home:
                for other in steps:
                    if isinstance(other, Derived) and step.node in other.frags:
                        return True
                    if isinstance(other, Store) and other.frag is step.node:
                        return True
        return False

    def emit_derived(  # type: ignore[override]
        self, step: Derived, loaded: set[int]
    ) -> None:
        """Materialize a derived fragment: its pointwise DAG per element of
        this thread's row over the input fragments' rows (loaded on demand)
        and the 2-D tile inputs read as pairs at the element's coordinates
        (prefetched at the iteration top when possible); the result stays
        the slot's register source for the rest of the block and goes to the
        slot (tcgen05.st) when a later block or iteration reads it."""
        slot = step.slot
        for frag in step.frags:
            source = self.program.frag_slots[frag]
            if source not in loaded:
                self.emit_ld(source)
                loaded.add(source)
        self.hoist_scalars(step.node)
        prefix = self.df.new_var("md_dr") + "_"
        to_tmem = self.derived_needs_tmem(step)
        for chunk in range(self.ld_chunks(slot)):
            cols = self.chunk_cols(slot, chunk)
            name = f"{prefix}{chunk}"
            self.emit(f"{name} = cutlass.Array(cutlass.Float32, {cols}, alignment=16)")
            for col in range(0, cols, 2):
                c = chunk * TCGEN05_LD_COLS + col
                for access in step.loads:
                    words = self.derived_words.get((access.node, c))
                    if words is None:
                        lo, hi = self.pair_load(
                            self.derived_pair_address(step, access, c),
                            access.dtype,
                            self.pairs_aligned(access, 1),
                        )
                    elif len(words) == 1:
                        unpack = (
                            "unpack_f16x2"
                            if access.dtype == torch.float16
                            else "unpack_bf16x2"
                        )
                        lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                        self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({words[0]})")
                    else:
                        lo, hi = words
                    self.set_tile_leaf(access, 0, lo)
                    self.set_tile_leaf(access, 1, hi)
                for bit in range(2):
                    self.evaluator.reg = (chunk, col + bit)  # type: ignore[assignment]
                    self.evaluator.element = bit
                    coords = self.element_coords(step.node, slot, c + bit)
                    value = self.evaluator.define(step.node, coords)
                    self.emit(f"{name}[{col + bit}] = cutlass.Float32({value})")
                self.evaluator.reg = None
                self.evaluator.element = None
            if to_tmem:
                self.emit_st_array(name, slot, chunk)
        if to_tmem:
            self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")
        self.slot_regs[slot] = prefix
        loaded.add(slot)

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

    def staging_address(
        self, base: str, rows: int, row_bytes: int, row: str, chunk: str
    ) -> str:
        """Byte address of 16-byte chunk ``chunk`` of row ``row`` of a
        staging tile: rows of one swizzle atom (<= 128 B) with the atom's
        XOR, wider rows as 128-byte slabs side by side (the TMA box layout,
        so the tile can go out by TMA)."""
        cpr = row_bytes // 16
        if cpr <= 8:
            return f"{base} + {row} * {row_bytes} + {_swizzled_chunk(row, chunk, cpr)} * 16"
        slab = f"({chunk}) // 8"
        local = f"({chunk}) % 8"
        return (
            f"{base} + {slab} * {rows * ATOM_MAX_BYTES} + {row} * {ATOM_MAX_BYTES} "
            f"+ {_swizzled_chunk(row, local, 8)} * 16"
        )

    def emit_stage_in(  # type: ignore[override]
        self, store: Store, parity: str, evaluate: bool
    ) -> None:
        """This thread's fragment row as 16-byte packets of the store dtype
        into the staging buffer (``staging_address``).  ``evaluate`` renders
        the store's pointwise chain per element (its trailing cast folded
        into the pack); otherwise the value is the fragment itself or its
        cast and the registers are packed directly."""
        slot = self.program.frag_slots[store.frag]
        geometry = self.slot(slot)
        dtype = store.access.dtype
        es = dtype.itemsize
        row_bytes = geometry.cols * es
        per_packet = 16 // es
        base = self.staging_base(store, parity)
        row = self.row_var(slot)
        if row_bytes <= ATOM_MAX_BYTES:
            row_base = self.bind("md_srow", f"{base} + {row} * {row_bytes}")
        else:
            row_base = self.bind("md_srow", f"{base} + {row} * {ATOM_MAX_BYTES}")
        pack = "pack_f16x2" if dtype == torch.float16 else "pack_bf16x2"
        if evaluate:
            self.fold_root(store.value)
            self.hoist_scalars(store.value)
        for col in range(0, self.local_cols(slot), per_packet):
            chunk = self.chunk_expr(slot, col // per_packet, per_packet)
            if row_bytes <= ATOM_MAX_BYTES:
                swz = _swizzled_chunk(row, chunk, row_bytes // 16)
                address = f"{row_base} + {swz} * 16"
            else:
                address = (
                    f"{row_base} + ({chunk}) // 8 * {geometry.rows * ATOM_MAX_BYTES} "
                    f"+ {_swizzled_chunk(row, f'({chunk}) % 8', 8)} * 16"
                )
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
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32("
                f"{self.staging_address(base, geometry.rows, row_bytes, prow, pch)})"
            )
            self.emit(
                f"{_RUNTIME}.stg_v4_b32({gbase} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pch} * 16), {', '.join(words)})"
            )

        self.packets(geometry.rows * cpr, threads, idx, cpr, body)

    def tma_stores(self, stores: Sequence[Store]) -> list[Store]:
        """The staged stores of ``stores`` that go out by TMA."""
        return [
            store
            for store in stores
            if store in self.tc.staged_stores and store.access.node in self.tc.store_tma
        ]

    def emit_tma_stores(
        self, stores: Sequence[Store], parity: str, loop_value: str
    ) -> None:
        """The store thread: the staging tiles of ``stores`` to global memory
        as one bulk group.  The caller then waits until at most one
        iteration's groups are still reading their staging buffers, ahead
        of a barrier every thread passes before the stage-ins reusing those
        buffers (two iterations after the write): the batch's commit in
        the program-order loop, the next iteration's barrier in the simple
        loop (whose drain follows its own barrier, after the MMA issue)."""
        for store in stores:
            spec = self.tc.store_tma[store.access.node]
            slot = self.slot(self.program.frag_slots[store.frag])
            tile = slot.rows * slot.cols * store.access.dtype.itemsize
            for slab in range(spec.slabs):
                coords = ", ".join(self.tma_coords(spec, loop_value, slab))
                src = f"{spec.offset + slab * spec.slab_bytes} + ({parity}) * {tile}"
                self.emit(
                    f"{_PRIMS}.cp_async_bulk_tensor_global_shared_cta({spec.param}.get_ptr(), "
                    f"md_smem + ({src}), [{coords}])"
                )
        self.emit(f"{_PRIMS}.cp_async_bulk_commit_group()")

    def emit_direct_store(self, store: Store) -> None:
        """Pair stores of the store value straight from the fragment row."""
        slot = self.program.frag_slots[store.frag]
        self.slot(slot)
        access = store.access
        positions = access.tile_positions
        paired = self.pairs_aligned(access, 1) and not store.swap
        self.hoist_scalars(store.value)
        row = self.row_var(slot)
        for col in range(0, self.local_cols(slot), 2):
            values = []
            for bit in range(2):
                c = col + bit
                self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                coords = self.element_coords(store.value, slot, c, store.swap)
                values.append(self.evaluator.evaluate(store.value, coords))
            self.evaluator.reg = None
            coords = ["0"] * len(access.dims)
            column = self.col_expr(slot, col)
            first, second = (column, row) if store.swap else (row, column)
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
                    element[positions[1 if not store.swap else 0]] = self.col_expr(
                        slot, col + bit
                    )
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
            # A 64-row fragment fills TMEM lanes 0-15 of each quadrant: with
            # the 16x32bx2 shape lanes 16-31 hold the second half of the
            # columns of the same rows; a fragment too narrow for that
            # leaves them computing on a duplicate row, never storing.
            self.emit("md_row64 = (md_warp % 4) * 16 + md_lane % 16")
            if any(
                self.half_rows(index) and not self.half_cols(index)
                for index in range(len(tc.slots))
            ):
                self.emit("md_rowok = md_lane < 16")
            for index, slot in enumerate(tc.slots):
                if self.half_cols(index):
                    self.emit(f"md_cb{index} = (md_lane // 16) * {slot.cols // 2}")
        if 128 in heights:
            self.emit("md_row128 = (md_warp % 4) * 32 + md_lane")
        persistent = plan.persistent > 0
        if not persistent:
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
        general = self.general_loop
        if general:
            # Commits of the loop body's intermediate dot batches.
            self.emit("md_stepb = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)")
        self.emit("md_tslot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)")
        if tc.pass_warps:
            # Per ring slot: the pass warps' threads arrive on ``md_pass``
            # once the slot's tile is rewritten (and fenced to the async
            # proxy); the MMAs reading the slot commit to ``md_free`` when
            # they complete, which paces the pass warps' rewrite of the
            # slot's next tile whatever route fills it (a tile they copy
            # from global memory themselves never arms the full barrier).
            self.emit(
                f"md_pass = cute.arch.alloc_smem(cutlass.Int64, {tc.stages}, alignment=8)"
            )
            self.emit(
                f"md_free = cute.arch.alloc_smem(cutlass.Int64, {tc.stages}, alignment=8)"
            )
        self.emit_barrier_init()
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
        if persistent:
            # Everything above runs once per CTA, ahead of the tile loop;
            # the tile's coordinates and bases start its body.
            self.emit_instr_descs()
            if self.plan_cross_tile():
                self.emit(f"{self.phase_base} = cutlass.Int32(0)")
            self.setup_end = len(self.lines)
            self.define_axis_bases()
        self.define_bases()
        for access in program.accesses:
            self.tensor_names[access.node] = self.df.tensor_arg(
                access.fake, prefer_name=access.name
            ).name
        if not persistent:
            self.emit_instr_descs()
        self.define_iteration_bases(in_loop=False)
        self.emit_carry_init()
        begin, end, trip_static = self.loop_bounds()
        # Side loads of the first iteration; the loop refills them one
        # iteration ahead (dependent global loads would otherwise sit on
        # the serial chain).
        if not general:
            for index, site in enumerate(self.discover_prefetch_sites()):
                self.prefetched[site] = f"md_ax{index}"
        # The epilogue's per-thread side loads (a decay row / column of its
        # pointwise chains): one register each, defined ahead of the loop
        # and loaded under the last iteration's MMAs (the program-order loop
        # and a dynamic trip count load them after the loop, ahead of the
        # drain).  Issued after the drain, the chains' first instructions
        # stall on them (gated fwd_o at s1: 11 % of the kernel's stall
        # samples on the ``gc_i - gc_j`` subtraction).
        self.epilogue_sites = self.discover_epilogue_prefetch_sites()
        self.epilogue_sites_in_loop = (
            bool(self.epilogue_sites) and not general and bool(trip_static)
        )
        for index, (node, _coords) in enumerate(self.epilogue_sites):
            self.emit(f"md_ex{index} = {self.zero(self.accesses[node].dtype)}")
        if self.prefetched:
            for (node, _coords), register in self.prefetched.items():
                self.emit(f"{register} = {self.zero(self.accesses[node].dtype)}")
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
        self.discover_tables()
        if self.table_regs:
            for register in self.table_regs.values():
                self.emit(f"{register} = cutlass.Float32(0.0)")
            for (node, _coords), register in self.table_loads.items():
                self.emit(f"{register} = {self.zero(self.accesses[node].dtype)}")
            with (
                self.nest(f"if {end} > {begin}:") if trip_static is None else _NoNest()
            ):
                if trip_static != 0:
                    self.emit_table_loads(begin)
                    self.emit_table_compute(begin)
            if self.table_loads:
                value = f"({begin} + 1)"
                if trip_static is None:
                    with self.nest(f"if {value} < {end}:"):
                        self.emit_table_loads(value)
                elif trip_static > 1:
                    self.emit_table_loads(value)
        if general and any(step.loads for step in program.derived if step.loop >= 0):
            for ahead in range(1, max(tc.stages - 1, 2)):
                value = f"({begin} + {ahead})"
                if trip_static is not None and ahead < trip_static:
                    self.emit_derived_line_prefetch(value)
                elif trip_static is None:
                    with self.nest(f"if {value} < {end}:"):
                        self.emit_derived_line_prefetch(value)
        if tc.tma:
            # The first ``S - 1`` iterations' tiles (the first iteration's
            # with a single slot) and the epilogue tiles land during the
            # carry seed.  Under a cross-tile prefetch only the CTA's first
            # tile issues them here: its predecessor issued every later
            # tile's.
            with (
                self.nest("if md_warp == 0:"),
                self.nest(f"if {_PRIMS}.elect_sync():"),
                self.nest(f"if {self.phase_base} == 0:")
                if self.phase_base is not None
                else _NoNest(),
            ):
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
        self.table_lookup = True
        if tc.pass_warps:
            # Warps 4-7 rewrite the operand tiles as their TMA lands, up to
            # S - 1 tiles ahead; the epilogue warps run the chain on their
            # own barrier and warp 0 waits each tile's pass before the batch
            # that reads it.
            with self.nest(f"if md_warp >= {TCGEN05_EPILOGUE_WARPS}:"):
                self.emit_pass_warp_loop(begin, end)
            self.sync = EPILOGUE_WARPS_BARRIER
            with (
                self.nest("else:"),
                self.nest(f"for {self.loop_var} in cutlass.range({begin}, {end}):"),
            ):
                if general:
                    self.emit_general_iteration(begin, end)
                else:
                    self.emit_iteration(begin, end)
            self.sync = "cute.arch.barrier()"
        else:
            # Loop-invariant per-thread side loads (a scalar read by a pass)
            # go out once ahead of the loop instead of behind every
            # iteration's TMA wait.
            loop_axis = program.loop_axis
            self.hoists.append(
                _Hoist(
                    len(self.lines),
                    self.indent,
                    len(self.values),
                    loop_axis.loop if loop_axis is not None else 0,
                )
            )
            with self.nest(f"for {self.loop_var} in cutlass.range({begin}, {end}):"):
                if general:
                    self.emit_general_iteration(begin, end)
                else:
                    self.emit_iteration(begin, end)
            self.hoists.pop()
        self.table_lookup = False
        if self.epilogue_sites and not self.epilogue_sites_in_loop:
            self.define_iteration_bases(in_loop=False)
            self.emit_epilogue_prefetch_loads()
        for index, site in enumerate(self.epilogue_sites):
            self.prefetched[site] = f"md_ex{index}"
        # Drain: the last MMA must complete before the epilogue reads TMEM.
        if trip_static != 0:
            guard = (
                self.nest(f"if {end} > {begin}:") if trip_static is None else _NoNest()
            )
            with guard:
                self.emit(f"{_RT}.mbar_wait(md_accb, ({end} - {begin} - 1) & 1)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)")
        if self.phase_base is not None:
            # Every MMA of this tile completed, so every ring slot is free:
            # the next tile's leading slots go out now, under the epilogue
            # (the boxes overlapping its staging tile after its reads).
            self.emit(f"md_npid = {self.next_pid}")
            self.emit_next_tile_issue(begin, end, early=True)
        # The epilogue's derived values load their tile inputs themselves.
        self.derived_words = {}
        self.emit_epilogue()
        if persistent and tc.store_tma:
            # The next tile reuses the staging tiles the TMA stores read.
            with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                self.emit(f"{_PRIMS}.cp_async_bulk_wait_group(0)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")
        self.emit(self.sync)
        if persistent:
            # Every barrier is drained here (each TMA landed and was waited,
            # each commit waited): re-arm them at phase zero so the next tile
            # runs the same phases as a fresh CTA, then publish the inits.
            # A cross-tile prefetch leaves the ring's full barriers armed
            # for the next tile (their phases run on from the iteration
            # count) and issues its boxes the staging tile held.
            self.emit_barrier_init(ring=self.phase_base is None)
            if self.phase_base is not None:
                self.emit_next_tile_issue(begin, end, early=False)
                self.emit(f"{self.phase_base} = {self.phase_base} + ({end} - {begin})")
            self.emit(self.sync)
            self.teardown_start = len(self.lines)
        with self.nest("if md_warp == 0:"):
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
            )
            self.emit(
                f"{_PRIMS}.tcgen05_dealloc(md_acc0, {tc.tmem_cols}, group={_PRIMS}.CTAGroup.CTA_1)"
            )
        if tc.store_tma and not persistent:
            # The TMA stores complete before the kernel ends.
            with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                self.emit(f"{_PRIMS}.cp_async_bulk_wait_group(0)")
        assert plan.threads == 32 * tc.warps
        return "\n".join(self.lines) + "\n"

    def plan_cross_tile(self) -> bool:
        """Whether this persistent launch issues the next tile's leading ring
        slots under the epilogue (and set up the phase base and the next
        tile's coordinates): a single static-trip TMA ring without pass
        warps, epilogue tiles or TMA stores, over the generic
        persistent_interleaved program ids (their mixed-radix decomposition
        of ``virtual_pid + grid`` gives the next tile)."""
        from ..program_id import PersistentInterleavedProgramIDs

        tc = self.tc
        pid = self.df.pid
        _begin, _end, trip = self.loop_bounds()
        if (
            not tc.tma
            or not tc.tx_bytes
            or tc.pass_warps
            or self.general_loop
            or tc.epilogue_tx_bytes
            or tc.store_tma
            or not trip
            or not isinstance(pid, PersistentInterleavedProgramIDs)
            or self.df.config.get("xcd_remap", False)
        ):
            return False
        infos = pid.pid_info
        counts = [f"({info.num_pids_expr(is_device=True)})" for info in infos[:-1]]
        nxt = f"({pid.virtual_pid_var} + {pid.grid_size_expr})"
        coords: dict[str, str] = {}
        for i, info in enumerate(infos):
            value = nxt
            if i:
                value = f"({value}) // ({' * '.join(counts[:i])})"
            if i + 1 < len(infos):
                value = f"({value}) % ({counts[i]})"
            if info.block_size_var != "1":
                value = f"({value}) * {info.block_size_var}"
            coords[self.cg.tile_begin_var(info.block_id)] = value
        for axis in self.program.axes:
            if axis.kind in (AXIS_ROOT, AXIS_LEAD):
                begin = self.cg.tile_begin_var(axis.block_id)
                if begin not in coords:
                    return False
                coords[self.axis_base(axis)] = coords[begin]
        self.phase_base = "md_ib"
        self.next_pid = nxt
        self.next_coords = coords
        return True

    def next_tile(self, text: str) -> str:
        """``text`` (a TMA coordinate) at the next tile's coordinates."""
        names = sorted(self.next_coords, key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(name) for name in names) + r")\b"
        return re.sub(pattern, lambda m: f"({self.next_coords[m.group(0)]})", text)

    def emit_next_tile_issue(self, begin: str, end: str, *, early: bool) -> None:
        """One elected thread: the next tile's first ``max(S - 1, 1)`` ring
        slots -- ``early`` (right after this tile's drain) arms each slot's
        full barrier and issues the boxes clear of the epilogue staging tile
        in every slot; otherwise (behind the epilogue's staging reads) the
        rest."""
        from ..program_id import PersistentProgramIDs

        tc = self.tc
        pid = self.df.pid
        assert isinstance(pid, PersistentProgramIDs)
        _begin, _end, trip = self.loop_bounds()
        assert trip
        staging = (
            tc.epilogue_staging_offset,
            tc.epilogue_staging_offset + tc.epilogue_staging_bytes,
        )
        ring = tc.stages * tc.stage_bytes

        def clear(offset: int, size: int) -> bool:
            if not tc.epilogue_staging_bytes or staging[0] >= ring:
                return True
            return all(
                slot * tc.stage_bytes + offset + size <= staging[0]
                or slot * tc.stage_bytes + offset >= staging[1]
                for slot in range(tc.stages)
            )

        specs = [t.tma or t.raw_tma for t in tc.loop_tiles]
        with (
            self.nest("if md_warp == 0:"),
            self.nest(f"if {_PRIMS}.elect_sync():"),
            self.nest(f"if md_npid < {pid.total_pids_var}:"),
        ):
            for ahead in range(min(max(tc.stages - 1, 1), trip)):
                stage = (
                    f"({self.phase_base} + ({end} - {begin}) + {ahead}) % {tc.stages}"
                )
                if early:
                    self.emit(
                        f"{_PRIMS}.mbarrier_arrive_expect_tx(md_full + ({stage}), {tc.tx_bytes})"
                    )
                value = f"({begin} + {ahead})" if ahead else begin
                for spec in specs:
                    if spec is None:
                        continue
                    for slab in range(spec.slabs):
                        offset = spec.offset + slab * spec.slab_bytes
                        if clear(offset, spec.slab_bytes) != early:
                            continue
                        dst = f"md_smem + (({stage}) * {tc.stage_bytes} + {offset})"
                        coords = ", ".join(
                            self.next_tile(c)
                            for c in self.tma_coords(spec, value, slab)
                        )
                        self.emit(
                            f"{_PRIMS}.cp_async_bulk_tensor_shared_cta_global({dst}, "
                            f"{spec.param}.get_ptr(), [{coords}], md_full + ({stage}))"
                        )

    def emit_barrier_init(self, ring: bool = True) -> None:
        """Thread 0 initializes every mbarrier of the program (and frees the
        pass warps' ring slots; ``ring=False`` leaves the ring's full
        barriers alone), fenced for the async proxy."""
        tc = self.tc
        with self.nest("if md_tidx == 0:"):
            for stage in range(tc.stages):
                if ring:
                    self.emit(f"{_PRIMS}.mbarrier_init(md_full + {stage}, 1)")
                if tc.pass_warps:
                    self.emit(
                        f"{_PRIMS}.mbarrier_init(md_pass + {stage}, {32 * TCGEN05_EPILOGUE_WARPS})"
                    )
                    self.emit(f"{_PRIMS}.mbarrier_init(md_free + {stage}, 1)")
            for name in ("md_accb", "md_efull", "md_eacc"):
                self.emit(f"{_PRIMS}.mbarrier_init({name}, 1)")
            if self.general_loop:
                self.emit(f"{_PRIMS}.mbarrier_init(md_stepb, 1)")
            self.emit(f"{_PRIMS}.fence_mbarrier_init()")
            if tc.pass_warps:
                # Every slot starts free: its first tile has no earlier reader.
                for stage in range(tc.stages):
                    self.emit(f"{_PRIMS}.mbarrier_arrive(md_free + {stage})")
            self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')

    def emit_instr_descs(self) -> None:
        """The tcgen05 instruction descriptor of every dot."""
        plan, program = self.plan, self.program
        dtype = "cutlass.BFloat16" if self.kind == "bf16" else "cutlass.Float16"
        for dot in program.dots:
            index = self.dot_index[dot.node]
            self.emit(
                f"md_idesc{index} = {_PRIMS}.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, "
                f"a_dtype={dtype}, b_dtype={dtype}, n_dim={plan.size(dot.cols)}, "
                f"m_dim={plan.size(dot.rows)}, a_major={int(not operand_k_major(dot.a))}, "
                f"b_major={int(not operand_k_major(dot.b))})"
            )

    def pass_tiles(self) -> list[Tcgen05Tile]:
        """The loop operand tiles rewritten by a pass."""
        return [
            tile
            for tile in self.tc.loop_tiles
            if tile.route in (ROUTE_SYNC, ROUTE_INPLACE, ROUTE_CONVERT)
        ]

    def pass_batch(
        self, blocks: Sequence[tuple[list[Derived | Store], list[Dot]]]
    ) -> int:
        """The first batch reading a passed tile (warp 0 waits the pass
        warps' arrival ahead of it)."""
        passed = {tile.key for tile in self.pass_tiles()}
        batch = 0
        for _steps, dots in blocks:
            if not dots:
                continue
            for dot in dots:
                for operand in (dot.a, dot.b):
                    if (
                        operand.kind == "load"
                        and self.tc.tile(dot, operand).key in passed
                    ):
                        return batch
            batch += 1
        return 0

    def emit_pass_warp_loop(self, begin: str, end: str) -> None:
        """The pass warps' loop: tile ``md_j`` is rewritten in its ring slot
        as soon as its TMA landed (the slot's previous reader, MMA(j - S),
        completed before that TMA was issued), published to the async
        proxy and signalled on the slot's pass barrier; the side loads run
        at that tile's bases (no prefetch registers: the warps have the
        whole iteration of slack)."""
        saved = (
            self.loop_var,
            self.iter_vars,
            self.evaluator,
            self.prefetched,
            self.recording,
        )
        self.loop_var = "md_j"
        self.prefetched = {}
        self.recording = None
        with self.nest(f"for md_j in cutlass.range({begin}, {end}):"):
            self.emit_iteration_header(begin)
            self.iter_vars = dict(saved[1])
            for access in self.program.accesses:
                if self.loop_dependent(access):
                    self.iter_vars[access.node] = self.iteration_base(access, "md_j")
            self.evaluator = _PassEvaluator(self)
            # The slot's previous tile's MMAs completed (the prologue arrived
            # once per slot for the first tiles); then its TMA landed, when
            # a tile comes by TMA.
            self.emit(f"{_RT}.mbar_wait_sleep(md_free + md_stage, md_round)")
            self.emit_passes(
                self.pass_tiles(),
                "md_stage",
                f"{_RT}.mbar_wait_sleep(md_full + (md_stage), md_round)",
                f"(md_tidx - {32 * TCGEN05_EPILOGUE_WARPS})",
                32 * TCGEN05_EPILOGUE_WARPS,
            )
            self.emit(f"{_PRIMS}.mbarrier_arrive(md_pass + md_stage)")
        (
            self.loop_var,
            self.iter_vars,
            self.evaluator,
            self.prefetched,
            self.recording,
        ) = saved

    def emit_iteration_header(self, begin: str) -> str:
        """The loop axis base, the ring slot and its phase of iteration
        ``md_i``; returns the iteration offset expression."""
        tc, program = self.tc, self.program
        loop = self.loop_var
        offset = loop if begin == "0" else f"({loop} - {begin})"
        if program.loop_axis is not None:
            step = self.plan.size(program.loop_axis)
            self.emit(f"{self.axis_base(program.loop_axis)} = {loop} * {step}")
        # The ring's full barriers count the CTA's iterations across its
        # tiles under a cross-tile prefetch (``phase_base``).
        ring = offset if self.phase_base is None else f"({self.phase_base} + {offset})"
        self.emit(f"md_stage = {ring} % {tc.stages}")
        self.emit(f"md_round = ({ring} // {tc.stages}) & 1")
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
        waited_full = not tc.pass_warps and any(
            tile.route in (ROUTE_INPLACE, ROUTE_CONVERT) for tile in tc.loop_tiles
        )
        if self.has_passes(tc.loop_tiles) and not tc.pass_warps:
            self.emit_passes(
                tc.loop_tiles,
                "md_stage",
                f"{_RT}.mbar_wait(md_full + (md_stage), md_round)",
                "md_tidx",
                self.loop_threads,
            )
        loop_dots = program.dots_in("loop")
        carry_in = [s for s in program.stores if s.source == "carry_in"]
        carry_out = [s for s in program.stores if s.source == "body"]
        prepared = [dot for dot in loop_dots if self.prepares(dot)]
        tma_stores = self.tma_stores(carry_in)
        loop_staged = [
            s
            for s in tc.staged_stores
            if s.region == "loop"
            and s.source == "carry_in"
            and s.access.node not in tc.store_tma
        ]
        # The snapshot stores of the carry go out after the MMA issue: the
        # tensor-memory read alone gates the MMA (the WAR hazard), while
        # the pack, the staging writes and their proxy fence run under the
        # MMA; a second barrier then hands the staging tile to its drain
        # (the store warps, the TMA store thread or every thread).  On the
        # s2 vanilla forward the pack and staging writes ahead of the
        # barrier were 6 of 122 us.  With store warps the epilogue warps'
        # iteration is one branch of ``md_epi`` (the registers of a staged
        # ``if`` do not outlive it) and the store warps' the other; both
        # branches run the same barriers.  Under pass warps the snapshot
        # keeps its place ahead of the barrier: the operand passes are
        # already off the chain there and the second (named) barrier cost
        # the pinned s2 bwd_dh 175 -> 179 us.
        late = bool(carry_in) and not tc.pass_warps
        split = late and tc.store_warps > 0
        # The ring refill (iteration i + S - 1 into slot (i - 1) % S, whose
        # reader MMA(i - 1) completed with the acc wait at the top): with
        # store warps the first of them issues it right after the barrier
        # (they idle until the hand-off); otherwise the MMA warp issues it
        # after its commit, under the MMA.  On an epilogue warp ahead of its
        # pack it delayed the hand-off (gated s2 forward 177 -> 182 us).
        refill = bool(tc.tx_bytes) and tc.stages > 1
        refill_warp = TCGEN05_EPILOGUE_WARPS if split else None
        ahead = tc.stages - 1

        def emit_loads() -> None:
            loaded: set[int] = set()
            for store in carry_in:
                slot = program.frag_slots[store.frag]
                if slot not in loaded:
                    self.emit_ld(slot)
                    loaded.add(slot)
            for dot in prepared:
                self.emit_rescale(dot, loaded)

        def emit_snapshots() -> None:
            for store in carry_in:
                slot = program.frag_slots[store.frag]
                with self.row_guard(slot):
                    if store in tc.staged_stores:
                        self.emit_stage_in(store, "md_parity", evaluate=False)
                    else:
                        self.emit_direct_store(store)
            if tma_stores:
                # The staging writes, to the async proxy (the writers
                # only: the fence costs every thread that runs it).
                self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')

        def emit_mmas() -> None:
            with self.nest("if md_warp == 0:"):
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
                )
                if tc.tx_bytes and not waited_full:
                    self.emit(f"{_RT}.mbar_wait(md_full + md_stage, md_round)")
                if tc.pass_warps:
                    self.emit(f"{_RT}.mbar_wait(md_pass + md_stage, md_round)")
                for dot in loop_dots:
                    self.emit_dot_mmas(dot, "md_stage", accumulate=dot.acc is not None)
                with self.nest(f"if {_PRIMS}.elect_sync():"):
                    self.emit(
                        f"{_PRIMS}.tcgen05_commit(md_accb, group={_PRIMS}.CTAGroup.CTA_1)"
                    )
                    if tc.pass_warps:
                        self.emit(
                            f"{_PRIMS}.tcgen05_commit(md_free + md_stage, group={_PRIMS}.CTAGroup.CTA_1)"
                        )
                    if refill and refill_warp is None:
                        with self.nest(f"if {loop} + {ahead} < {end}:"):
                            self.emit_tma_issue(
                                f"({loop} + {ahead})",
                                f"(md_stage + {ahead}) % {tc.stages}",
                            )

        def emit_refill() -> None:
            if not refill or refill_warp is None:
                return
            with (
                self.nest(f"if md_warp == {refill_warp}:"),
                self.nest(f"if {_PRIMS}.elect_sync():"),
                self.nest(f"if {loop} + {ahead} < {end}:"),
            ):
                self.emit_tma_issue(
                    f"({loop} + {ahead})", f"(md_stage + {ahead}) % {tc.stages}"
                )

        def emit_side_loads() -> None:
            # The next iteration's side loads land in their registers while
            # the MMA runs (issued after the drain they stalled the gated
            # forward's pass: 177 -> 186 us at s2); the lines of the
            # iteration a ring depth ahead go to L2.
            if not self.prefetched:
                return
            with self.nest(f"if {loop} + 1 < {end}:"):
                self.emit_prefetch_loads(f"({loop} + 1)")
            ahead_lines = max(tc.stages - 1, 2)
            with self.nest(f"if {loop} + {ahead_lines} < {end}:"):
                self.emit_line_prefetch(f"({loop} + {ahead_lines})")

        def emit_drain(idx: str, threads: int) -> None:
            if tma_stores:
                with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                    self.emit_tma_stores(tma_stores, "md_parity", loop)
                    self.emit(f"{_PRIMS}.cp_async_bulk_wait_group(1, read=True)")
            for store in loop_staged:
                self.emit_stage_out(store, "md_parity", loop, idx, threads)

        if not late:
            # Fail closed: the snapshot ahead of the barrier, the refill by
            # the MMA warp after its commit, the drain after the MMA block.
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
                    if tma_stores:
                        # The staging writes, to the async proxy (the writers
                        # only: the fence costs every thread that runs it).
                        self.emit(
                            'cute.arch.fence_proxy(kind="async.shared", space="cta")'
                        )
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
                if tc.pass_warps:
                    self.emit(f"{_RT}.mbar_wait(md_pass + md_stage, md_round)")
                for dot in loop_dots:
                    self.emit_dot_mmas(dot, "md_stage", accumulate=dot.acc is not None)
                if tma_stores:
                    with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                        self.emit_tma_stores(tma_stores, "md_parity", loop)
                        self.emit(f"{_PRIMS}.cp_async_bulk_wait_group(1, read=True)")
                with self.nest(f"if {_PRIMS}.elect_sync():"):
                    self.emit(
                        f"{_PRIMS}.tcgen05_commit(md_accb, group={_PRIMS}.CTAGroup.CTA_1)"
                    )
                    if tc.pass_warps:
                        self.emit(
                            f"{_PRIMS}.tcgen05_commit(md_free + md_stage, group={_PRIMS}.CTAGroup.CTA_1)"
                        )
                    if tc.tx_bytes and tc.stages > 1:
                        ahead = tc.stages - 1
                        with self.nest(f"if {loop} + {ahead} < {end}:"):
                            self.emit_tma_issue(
                                f"({loop} + {ahead})",
                                f"(md_stage + {ahead}) % {tc.stages}",
                            )
            # The epilogue's side loads go out under the last iteration's MMAs.
            if self.epilogue_sites_in_loop:
                with self.nest(f"if {loop} + 1 == {end}:"):
                    self.emit_epilogue_prefetch_loads()
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
            self.emit_tables_ahead(loop, end, 0)
            # The snapshots stream out while the MMA runs: the store warps, or
            # the epilogue warps when there are none (packing the rows after the
            # issue instead, with the previous iteration's drain, measured 122
            # vs 104 us on the s2 forward).
            if loop_staged:
                if tc.store_warps:
                    nest = self.nest(f"if md_warp >= {TCGEN05_EPILOGUE_WARPS}:")
                    idx, threads = (
                        f"(md_tidx - {32 * TCGEN05_EPILOGUE_WARPS})",
                        32 * tc.store_warps,
                    )
                else:
                    nest = _NoNest()
                    idx, threads = "md_tidx", self.loop_threads
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
            self.emit_tables_ahead(loop, end, 1)
            return
        if not split:
            # Every warp of the loop is an epilogue warp (``md_epi`` holds).
            emit_loads()
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
            )
            self.emit(self.sync)
            emit_refill()
            emit_mmas()
            emit_side_loads()
            emit_snapshots()
            if tma_stores or loop_staged:
                self.emit(self.sync)
                emit_drain("md_tidx", self.loop_threads)
        else:
            # The store warps alone drain the staging tile: the epilogue
            # warps taking half of the packets after their pack lengthened
            # their path (130 vs 122 us on the s2 forward).
            with self.block("if md_epi:"):
                emit_loads()
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
                )
                self.emit(self.sync)
                emit_mmas()
                emit_side_loads()
                emit_snapshots()
                if loop_staged:
                    self.emit(self.sync)
            with self.nest("else:"):
                self.emit(self.sync)
                emit_refill()
                emit_side_loads()
                if loop_staged:
                    self.emit(self.sync)
                    emit_drain(
                        f"(md_tidx - {32 * TCGEN05_EPILOGUE_WARPS})",
                        32 * tc.store_warps,
                    )
        # The epilogue's side loads go out under the last iteration's MMAs.
        if self.epilogue_sites_in_loop:
            with self.nest(f"if {loop} + 1 == {end}:"):
                self.emit_epilogue_prefetch_loads()
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
        self.emit_tables_ahead(loop, end, 0)
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
        self.emit_tables_ahead(loop, end, 1)

    # -- the program-order loop body ---------------------------------------------

    @property
    def general_loop(self) -> bool:
        return program_order_loop(self.program)

    def loop_blocks(self) -> list[tuple[list[Derived | Store], list[Dot]]]:
        """The loop body (``program.steps`` without the hoisted snapshots) as
        blocks of epilogue-warp work followed by a batch of dots issued
        together: a step reading a slot a dot of the pending batch writes --
        or writing a slot the batch reads -- opens a new block (the batch
        is committed and waited first); a dot accumulating in place onto a
        batch dot joins the batch (the tensor core orders them)."""
        program = self.program
        blocks: list[tuple[list[Derived | Store], list[Dot]]] = []
        steps: list[Derived | Store] = []
        dots: list[Dot] = []
        produced: set[int] = set()
        batch_reads: set[int] = set()

        def flush() -> None:
            nonlocal steps, dots, produced, batch_reads
            if steps or dots:
                blocks.append((steps, dots))
            steps, dots, produced, batch_reads = [], [], set(), set()

        for step in program.steps:
            if isinstance(step, Store) and step.source == "carry_in":
                continue
            if isinstance(step, Dot):
                reads = {
                    program.frag_slots[op.frag]
                    for op in (step.a, step.b)
                    if op.kind == "frag" and op.frag is not None
                }
                if step.acc_slot is not None and (
                    self.prepares(step) or step.acc_frag not in produced_nodes(dots)
                ):
                    reads.add(step.acc_slot)
                if reads & produced:
                    flush()
                dots.append(step)
                produced.add(step.slot)
                batch_reads |= reads
                continue
            if isinstance(step, Derived):
                reads = {program.frag_slots[frag] for frag in step.frags}
                writes = step.slot
            else:
                reads = {program.frag_slots[step.frag]}
                writes = -1
            if (reads & produced) or writes in batch_reads or writes in produced:
                flush()
            steps.append(step)
        flush()
        return blocks

    def emit_general_iteration(self, begin: str, end: str) -> None:
        """One iteration of the program-order loop body (see ``loop_blocks``):
        the acc wait, the operand passes, then per block the epilogue-warp
        work (snapshots in the first, derived values, body stores, exchange
        tiles and accumulator preparation of the block's dots), a barrier
        and warp 0's batch issue with its commit; intermediate batches commit
        to the step barrier, the last one to acc_ready."""
        tc, program = self.tc, self.program
        loop = self.loop_var
        offset = self.emit_iteration_header(begin)
        self.emit(f"md_parity = {offset} % 2")
        self.define_iteration_bases(in_loop=True)
        self.evaluator = _PassEvaluator(self)
        blocks = self.loop_blocks()
        batches = [index for index, (_steps, dots) in enumerate(blocks) if dots]
        assert batches, "a loop body without a dot"
        step_commits = len(batches) - 1
        carry_in = [s for s in program.stores if s.source == "carry_in"]
        # The loop's derived values' tile inputs (an epilogue derived value
        # loads its own in the epilogue: hoisted here they would read a
        # stored tensor ahead of the last iteration's store).
        self.derived_words = {}
        loop_derived = [step for step in program.derived if step.loop >= 0]
        for step in loop_derived:
            self.emit_derived_prefetch(step)
        if any(step.loads for step in loop_derived):
            ahead = max(tc.stages - 1, 2)
            with self.nest(f"if {loop} + {ahead} < {end}:"):
                self.emit_derived_line_prefetch(f"({loop} + {ahead})")
        # Bulk groups per iteration issued with a batch (a trailing block's
        # group has no commit behind it and is drained by the next
        # iteration's first batch).
        groups = 0
        pending_stores: list[Store] = self.tma_stores(carry_in)
        for steps, dots in blocks:
            pending_stores.extend(
                self.tma_stores([step for step in steps if isinstance(step, Store)])
            )
            if dots and pending_stores:
                groups += 1
                pending_stores = []
        with self.nest(f"if {loop} > {begin}:"):
            self.emit(f"{_RT}.mbar_wait(md_accb, md_parity ^ 1)")
            self.emit(
                f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
            )
        # The passes no dot of the first batch reads run together after the
        # issue of the batch before their earliest reader (under those MMAs)
        # instead of ahead of the first barrier, published by the next
        # block's proxy fence; the window is shorter than a pass, so only its
        # slack is won (s2: gated fwd_h_delta 372 -> 352 us, dstate 557 ->
        # 550; a second window and its own fence cost dstate 14 us).
        late_passes = {} if tc.pass_warps else self.late_pass_windows(blocks)
        late_window = min(late_passes.values(), default=-1)
        late_tiles = [t for t in tc.loop_tiles if t.key in late_passes]
        pass_pending = False
        passed = {t.key for t in self.pass_tiles()} if tc.pass_warps else set()
        top_tiles = [
            t for t in tc.loop_tiles if t.key not in late_passes and t.key not in passed
        ]
        pass_batch = self.pass_batch(blocks) if tc.pass_warps else -1
        waited_full = any(
            tile.route in (ROUTE_INPLACE, ROUTE_CONVERT) for tile in top_tiles
        )
        if self.has_passes(top_tiles):
            self.emit_passes(
                top_tiles,
                "md_stage",
                f"{_RT}.mbar_wait(md_full + (md_stage), md_round)",
                "md_tidx",
                self.loop_threads,
            )
        if tc.tx_bytes and tc.stages > 1:
            # The slot refilled here was read by the previous iteration's
            # batches, all complete with the acc wait above.
            ahead = tc.stages - 1
            with (
                self.nest("if md_warp == 0:"),
                self.nest(f"if {_PRIMS}.elect_sync():"),
                self.nest(f"if {loop} + {ahead} < {end}:"),
            ):
                self.emit_tma_issue(
                    f"({loop} + {ahead})", f"(md_stage + {ahead}) % {tc.stages}"
                )
        written: set[tuple[object, ...]] = set()
        # Exchange tiles of the carry at loop entry are written in the first
        # block when nothing before their dot rewrites the carry.
        hoisted: list[Tcgen05Tile] = []
        touched: set[int] = set()
        for step in program.steps:
            if isinstance(step, Dot):
                for operand in (step.a, step.b):
                    frag = operand.frag
                    if frag is not None and frag in program.carry_placeholders:
                        slot = program.frag_slots[frag]
                        tile = tc.tile(step, operand)
                        if slot not in touched and tile.key not in written:
                            hoisted.append(tile)
                            written.add(tile.key)
                touched.add(step.slot)
            elif isinstance(step, Derived):
                touched.add(step.slot)
        batch_index = 0
        # Staged body stores waiting for the barrier of the next batch.
        drains: list[Store] = []
        tma_snapshots = self.tma_stores(carry_in)
        for index, (steps, dots) in enumerate(blocks):
            if index > 0:
                previous = batch_index - 1
                if previous == step_commits:
                    self.emit(f"{_RT}.mbar_wait(md_accb, md_parity)")
                else:
                    self.emit(
                        f"{_RT}.mbar_wait(md_stepb, ({offset} * {step_commits} + {previous}) & 1)"
                    )
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
                )
            exchanges = list(hoisted) if index == 0 else []
            for dot in dots:
                for operand in (dot.a, dot.b):
                    if operand.kind != "frag":
                        continue
                    tile = tc.tile(dot, operand)
                    if tile.key not in written:
                        written.add(tile.key)
                        exchanges.append(tile)
            preps = [dot for dot in dots if self.prepares(dot)]
            snapshots = carry_in if index == 0 else []
            if snapshots or steps or exchanges or preps:
                with self.block("if md_epi:"):
                    self.slot_regs = {}
                    loaded: set[int] = set()
                    for store in snapshots:
                        slot = program.frag_slots[store.frag]
                        if slot not in loaded:
                            self.emit_ld(slot)
                            loaded.add(slot)
                        with self.row_guard(slot):
                            if store in tc.staged_stores:
                                self.emit_stage_in(store, "md_parity", evaluate=False)
                            else:
                                self.emit_direct_store(store)
                    for step in steps:
                        if isinstance(step, Derived):
                            self.emit_derived(step, loaded)
                        else:
                            slot = program.frag_slots[step.frag]
                            if slot not in loaded:
                                self.emit_ld(slot)
                                loaded.add(slot)
                            with self.row_guard(slot):
                                if step in tc.staged_stores:
                                    self.emit_stage_in(step, "md_parity", evaluate=True)
                                    drains.append(step)
                                else:
                                    self.emit_direct_store(step)
                    for tile in exchanges:
                        source = cast("int", tile.source_slot)
                        if source not in loaded:
                            self.emit_ld(source)
                            loaded.add(source)
                        self.emit_exchange(tile)
                    for dot in preps:
                        self.emit_rescale(
                            dot, loaded, prefix=self.df.new_var("md_rs") + "_"
                        )
                    self.slot_regs = {}
                    staged_tma = self.tma_stores(
                        [step for step in steps if isinstance(step, Store)]
                    )
                    if staged_tma or (index == 0 and tma_snapshots):
                        self.emit(
                            'cute.arch.fence_proxy(kind="async.shared", space="cta")'
                        )
                if exchanges or pass_pending:
                    self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
                    pass_pending = False
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
                )
            elif pass_pending and dots:
                self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
                pass_pending = False
            if not dots:
                continue
            self.emit(self.sync)
            tma_now = [
                *(tma_snapshots if batch_index == 0 else []),
                *self.tma_stores(drains),
            ]
            with self.nest("if md_warp == 0:"):
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
                )
                if batch_index == 0 and tc.tx_bytes and not waited_full:
                    self.emit(f"{_RT}.mbar_wait(md_full + md_stage, md_round)")
                if batch_index == pass_batch:
                    self.emit(f"{_RT}.mbar_wait(md_pass + md_stage, md_round)")
                for dot in dots:
                    self.emit_dot_mmas(dot, "md_stage", accumulate=dot.acc is not None)
                if tma_now:
                    with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                        self.emit_tma_stores(tma_now, "md_parity", loop)
                        self.emit(
                            f"{_PRIMS}.cp_async_bulk_wait_group({groups}, read=True)"
                        )
                barrier = "md_accb" if batch_index == step_commits else "md_stepb"
                with self.nest(f"if {_PRIMS}.elect_sync():"):
                    self.emit(
                        f"{_PRIMS}.tcgen05_commit({barrier}, group={_PRIMS}.CTAGroup.CTA_1)"
                    )
                    if tc.pass_warps and batch_index == step_commits:
                        self.emit(
                            f"{_PRIMS}.tcgen05_commit(md_free + md_stage, group={_PRIMS}.CTAGroup.CTA_1)"
                        )
            for store in drains:
                if store.access.node not in tc.store_tma:
                    self.emit_stage_out(
                        store, "md_parity", loop, "md_tidx", self.loop_threads
                    )
            drains = []
            if batch_index == 0:
                # The snapshots stream out while the first batch runs.
                loop_staged = [
                    s
                    for s in tc.staged_stores
                    if s.region == "loop"
                    and s.source == "carry_in"
                    and s.access.node not in tc.store_tma
                ]
                if loop_staged:
                    if tc.store_warps:
                        nest = self.nest(f"if md_warp >= {TCGEN05_EPILOGUE_WARPS}:")
                        idx, threads = (
                            f"(md_tidx - {32 * TCGEN05_EPILOGUE_WARPS})",
                            32 * tc.store_warps,
                        )
                    else:
                        nest = _NoNest()
                        idx, threads = "md_tidx", self.loop_threads
                    with nest:
                        for store in loop_staged:
                            self.emit_stage_out(store, "md_parity", loop, idx, threads)
            if batch_index == late_window:
                self.emit_passes(
                    late_tiles,
                    "md_stage",
                    f"{_RT}.mbar_wait(md_full + (md_stage), md_round)",
                    "md_tidx",
                    self.loop_threads,
                    fence=False,
                )
                pass_pending = True
            self.emit_tables_ahead(loop, end, batch_index)
            batch_index += 1
        if drains:
            # Body stores of the trailing block drain behind their own barrier.
            self.emit(self.sync)
            trailing = self.tma_stores(drains)
            if trailing:
                with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                    self.emit_tma_stores(trailing, "md_parity", loop)
                    if not groups:
                        # No batch waits in this loop: the previous
                        # iteration's group is read out here, before the
                        # commit of the next iteration's batch that the
                        # stage-in reusing its buffer waits on.
                        self.emit(f"{_PRIMS}.cp_async_bulk_wait_group(1, read=True)")
            for store in drains:
                if store.access.node not in tc.store_tma:
                    self.emit_stage_out(
                        store, "md_parity", loop, "md_tidx", self.loop_threads
                    )
        if tc.tx_bytes and tc.stages == 1:
            # Refill after every batch read the slot (the next iteration's
            # acc wait is then immediate).
            self.emit(f"{_RT}.mbar_wait(md_accb, md_parity)")
            with (
                self.nest("if md_warp == 0:"),
                self.nest(f"if {_PRIMS}.elect_sync():"),
                self.nest(f"if {loop} + 1 < {end}:"),
            ):
                self.emit_tma_issue(f"({loop} + 1)", "0")
        self.emit_tables_ahead(loop, end, step_commits + 1)

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
        self.slot(slot)
        es = tile.elem_bytes
        per_packet = 16 // es
        value = self.pass_value(operand)
        self.hoist_scalars(value)
        row = self.row_var(slot)
        with self.row_guard(slot):
            for col in range(0, self.local_cols(slot), per_packet):
                values = []
                for c in range(col, col + per_packet):
                    self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                    coords = self.element_coords(value, slot, c, operand.swap)
                    values.append(self.evaluator.evaluate(value, coords))
                self.evaluator.reg = None
                words = [
                    self.pack(values[2 * q], values[2 * q + 1], operand.dtype)
                    for q in range(4)
                ]
                chunk = self.chunk_expr(slot, col // per_packet, per_packet)
                address = self.tile_address(tile, "0", row, chunk)
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
    "Tcgen05SmemItem",
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
