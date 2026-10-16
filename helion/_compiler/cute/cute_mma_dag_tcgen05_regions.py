"""The tcgen05 compute mode of ``cute_mma_dag`` over the region tree.

The simple shape (one innermost loop carrying fragments, then an epilogue)
is rendered by :mod:`cute_mma_dag_tcgen05`; this module lowers the rest of
the matched programs (the region programs: nested and sequential loops,
hoisted operand tiles, fresh accumulators seeded from loads, vector carries
with row / column reductions and vector / scalar stores) with the same
primitives:

- Fragments live in tensor memory (one row per epilogue thread); vectors
  are one fp32 register per epilogue thread, the value of the thread's row
  (a column reduction relays its sums into that layout through the
  scratch tile); scalars are replicated.
- Innermost loops stage their per-iteration operand tiles through a TMA
  ring of ``stages`` slots (``md_full{L}`` barriers); a loop's other tiles
  land once per iteration in single slots (``md_sf{L}``), loop-invariant
  tiles once per CTA.  Sibling loops' frames overlap in shared memory.
- Every MMA batch commits to one barrier (``md_cb``); ``md_cn`` counts the
  commits and ``mbar_wait(md_cb, (md_cn + 1) & 1)`` waits for the latest.
  At most one commit is outstanding: a batch is issued only after the
  previous one was waited, so the parity never aliases.
- Per-loop running counters (``md_rk{L}``: iterations so far, across
  re-entries) give the ring slot, its phase and the staging parity.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_RDIM
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import ROUTE_CONVERT
from .cute_mma_dag import ROUTE_DIRECT
from .cute_mma_dag import ROUTE_EXCHANGE
from .cute_mma_dag import ROUTE_INPLACE
from .cute_mma_dag import ROUTE_SYNC
from .cute_mma_dag import Derived
from .cute_mma_dag import Dot
from .cute_mma_dag import Loop
from .cute_mma_dag import Reduce
from .cute_mma_dag import Step
from .cute_mma_dag import Store
from .cute_mma_dag import VecOp
from .cute_mma_dag import VecStore
from .cute_mma_dag import _decline
from .cute_mma_dag import staged_store_reason
from .cute_mma_dag_codegen import _RUNTIME
from .cute_mma_dag_codegen import _shape
from .cute_mma_dag_tcgen05 import _PRIMS
from .cute_mma_dag_tcgen05 import _RT
from .cute_mma_dag_tcgen05 import _SWIZZLE_BY_ROW_BYTES
from .cute_mma_dag_tcgen05 import REGION_EPILOGUE
from .cute_mma_dag_tcgen05 import REGION_EXCHANGE
from .cute_mma_dag_tcgen05 import REGION_LOOP
from .cute_mma_dag_tcgen05 import SLAB_ALIGNMENT
from .cute_mma_dag_tcgen05 import TCGEN05_EPILOGUE_WARPS
from .cute_mma_dag_tcgen05 import TCGEN05_LD_COLS
from .cute_mma_dag_tcgen05 import TCGEN05_MAX_TMEM_COLS
from .cute_mma_dag_tcgen05 import TCGEN05_SLOT_ALIGN
from .cute_mma_dag_tcgen05 import TMA_MAX_RANK
from .cute_mma_dag_tcgen05 import SlotGeometry
from .cute_mma_dag_tcgen05 import Tcgen05ExchangeFacts
from .cute_mma_dag_tcgen05 import Tcgen05Facts
from .cute_mma_dag_tcgen05 import Tcgen05Plan
from .cute_mma_dag_tcgen05 import Tcgen05SmemItem
from .cute_mma_dag_tcgen05 import Tcgen05Tile
from .cute_mma_dag_tcgen05 import Tcgen05TileFacts
from .cute_mma_dag_tcgen05 import TmaSpec
from .cute_mma_dag_tcgen05 import _align
from .cute_mma_dag_tcgen05 import _PassEvaluator
from .cute_mma_dag_tcgen05 import _Tcgen05Emitter
from .cute_mma_dag_tcgen05 import _tma_dims
from .cute_mma_dag_tcgen05 import epilogue_staged_reason
from .cute_mma_dag_tcgen05 import produced_nodes
from .cute_mma_dag_tcgen05 import tcgen05_atom_bytes
from .cute_mma_dag_tcgen05 import tcgen05_tmem_cols

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..generate_ast import GenerateAST
    from .cute_mma_dag import Access
    from .cute_mma_dag import Axis
    from .cute_mma_dag import CuteMmaDagFacts
    from .cute_mma_dag import LoopFacts
    from .cute_mma_dag import MmaDagPlan
    from .cute_mma_dag import MmaDagProgram
    from .cute_mma_dag import Operand

# The epilogue warps' own barrier (cross-warp vector reductions inside the
# ``md_epi`` blocks; the CTA barrier would hang the store warps).
EPILOGUE_BARRIER_ID = 1
# Entries of one warp's vector scratch (the TMEM lane count: a vector never
# exceeds the fragment rows it is relaid into).
SCRATCH_ENTRIES = 128
# 16-bit operands take the F16 MMA kind, fp32 operands the TF32 kind (the
# tensor core truncates to tf32 as Triton's dot does on this target).
OPERAND_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
ITEM_TILE = "tile"
ITEM_RAW = "raw"
ITEM_EXCHANGE = "exchange"
ITEM_STAGING = "staging"
ITEM_SCRATCH = "scratch"
_OPERAND_DTYPE_NAMES = {
    torch.bfloat16: "cutlass.BFloat16",
    torch.float16: "cutlass.Float16",
    torch.float32: "cutlass.Float32",
}


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def region_program_reason(program: MmaDagProgram) -> str | None:
    """Why the region renderer cannot lower ``program`` (None: it can)."""
    for loop in program.loops:
        if not loop.innermost and any(
            store.loop == loop.index and store.source == "carry_in"
            for store in program.stores
        ):
            return "a carry snapshot in an outer loop"
    for dot in program.dots:
        if dot.a.dtype != dot.b.dtype or dot.a.dtype not in OPERAND_DTYPES:
            return f"a dot over {dot.a.dtype} x {dot.b.dtype} operands"
        for operand in (dot.a, dot.b):
            # UMMA reads tf32 tiles K-major here (the MN-major layout wants
            # the ATOM_32B swizzle, which neither TMA nor the exchange
            # writes produce).  Only a B fragment or a widened 16-bit B load
            # takes the planner's transposed [N, K] tile (the latter once the
            # source proves aligned, at plan time), so an A operand or an
            # fp32 source read with K along its rows stays MN-major.
            if operand.dtype == torch.float32 and operand.trans:
                load = operand.load
                if operand.role == "a" or (
                    load is not None and load.dtype == operand.dtype
                ):
                    return f"{operand.value.name}: an MN-major tf32 operand"
    for step in program.vec_stores:
        slot = program.vec_leaves[step.vec]
        if program.vec_slots[slot].orient == "cols" and (
            program.vec_slots[slot].axis is not None
        ):
            return "a store of a column vector"
    return None


def operand_geometry(
    program: MmaDagProgram, operand: Operand
) -> tuple[Axis, Axis, bool]:
    """``(rows, cols, k_major)`` of the operand's shared tile as the tensor
    core reads it.  The planner's transposed fp32 B tiles (``[N, K]``) are
    kept and read K-major: an MN-major tf32 operand under the 16-byte
    SW128 layout (TMA's, and the exchange writes') returned garbage on
    sm100 -- 32-bit MN-major wants the BASE32B swizzle variant -- so the
    exchange / widen passes scatter into the transposed tile instead."""
    return operand.rows, operand.cols, not operand.trans


def source_geometry(operand: Operand) -> tuple[Axis, Axis]:
    """``(rows, cols)`` of a load operand's source tile as TMA fetches it
    (the contiguous dim last): the natural orientation."""
    if operand.transposed_tile:
        return operand.cols, operand.rows
    return operand.rows, operand.cols


def tile_key(program: MmaDagProgram, operand: Operand) -> tuple[object, ...]:
    """The sharing key of an operand's shared tile: the pointwise chain key,
    except that a load read through views and at most a dtype cast shares
    one tile with every other such read of the same source (``dL`` straight
    and transposed hold the same bytes)."""
    load = operand.load
    if load is None:
        # A fragment read straight and transposed needs two tiles (both
        # read K-major; the transposed one is written transposed).
        return operand.chain_key
    if operand.aux_loads:
        return operand.chain_key
    from ...language import view_ops
    from .cute_mma_dag import _is_swap_permute

    value = program.resolve(operand.value)
    while value is not load.node and (
        _is_swap_permute(value)
        or (
            value.op == "call_function"
            and value.target
            in (view_ops.subscript, torch.ops.prims.convert_element_type.default)
        )
    ):
        value = program.resolve(cast("Node", value.args[0]))
    if value is not load.node:
        return operand.chain_key
    # Two loads of one tensor at the same subscripts are one tile.
    return (
        "plain",
        load.fake,
        load.subscripts,
        load.strides,
        operand.rows_pos,
        operand.dtype,
    )


def tmem_slots(program: MmaDagProgram) -> tuple[int, ...]:
    """The fragment slots that live in tensor memory: dot results and
    accumulators, carries, derived values over fragments, and a derived
    value over tiles alone that leaves its block (``derived_crosses``); the
    other tile-only derived values stay in registers, recomputed in every
    block that reads them."""
    slots: set[int] = set()
    for dot in program.dots:
        slots.add(dot.slot)
        if dot.acc_slot is not None:
            slots.add(dot.acc_slot)
    for loop in program.loops:
        slots.update(carry.slot for carry in loop.carries)
    for derived in program.derived:
        if derived.frags or derived_crosses(program, derived):
            slots.add(derived.slot)
    return tuple(sorted(slots))


def derived_crosses(program: MmaDagProgram, derived: Derived) -> bool:
    """Whether a derived value is read beyond the block computing it: a dot
    takes it (accumulator or operand), a loop carries it, another derived
    value reads it, or a store / reduction in another region does."""
    node = derived.node
    for loop in program.loops:
        if any(program.resolve(carry.output) is node for carry in loop.carries):
            return True
    for dot in program.dots:
        if dot.acc_frag is node or any(
            operand.frag is node for operand in (dot.a, dot.b)
        ):
            return True
    if any(node in other.frags for other in program.derived):
        return True
    readers: list[Step] = [
        *(store for store in program.stores if store.frag is node),
        *(reduce for reduce in program.reduces if reduce.frag is node),
    ]
    if any(reader.loop != derived.loop for reader in readers):  # type: ignore[attr-defined]
        return True
    # A reader of the same region behind a nested loop sits in another
    # segment (the blocks are built per segment): crossing as well.
    steps = program.body if derived.loop < 0 else program.loops[derived.loop].steps
    start = next(i for i, step in enumerate(steps) if step is derived)
    for reader in readers:
        end = next(i for i, step in enumerate(steps) if step is reader)
        if any(isinstance(step, Loop) for step in steps[start:end]):
            return True
    return False


def region_items(
    program: MmaDagProgram,
) -> list[tuple[Tcgen05SmemItem, object]]:
    """The shared-memory items of ``program`` in a fixed order with their
    program payloads (the facts keep the items, the plan the payloads)."""
    items: list[tuple[Tcgen05SmemItem, object]] = []
    seen: set[tuple[object, ...]] = set()
    # An exchange tile lives in the frame of the loop whose dots read it
    # (sibling frames overlap), in the root frame when several loops do.
    consumers: dict[tuple[object, ...], set[int]] = {}
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.kind == "frag":
                consumers.setdefault(tile_key(program, operand), set()).add(dot.loop)
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            key = tile_key(program, operand)
            if key in seen:
                continue
            seen.add(key)
            rows_axis, cols_axis, _k_major = operand_geometry(program, operand)
            rows, cols = rows_axis.index, cols_axis.index
            es = operand.dtype.itemsize
            if operand.kind == "load":
                load = operand.load
                assert load is not None
                ring = program.is_ring(dot, operand)
                frame = program.operand_frame(dot, operand)
                items.append(
                    (
                        Tcgen05SmemItem(
                            ITEM_TILE, frame, rows, cols, es, 0 if ring else 1
                        ),
                        (dot, operand),
                    )
                )
                if load.dtype != operand.dtype:
                    src_rows, src_cols = source_geometry(operand)
                    items.append(
                        (
                            Tcgen05SmemItem(
                                ITEM_RAW,
                                frame,
                                src_rows.index,
                                src_cols.index,
                                load.dtype.itemsize,
                                0 if ring else 1,
                            ),
                            (dot, operand),
                        )
                    )
            else:
                items.append(
                    (
                        Tcgen05SmemItem(
                            ITEM_EXCHANGE,
                            next(iter(consumers[key]))
                            if len(consumers[key]) == 1
                            else -1,
                            rows,
                            cols,
                            es,
                            1,
                        ),
                        (dot, operand),
                    )
                )
    for store in program.stores:
        rows, cols = program.slots[program.frag_slots[store.frag]]
        items.append(
            (
                Tcgen05SmemItem(
                    ITEM_STAGING,
                    store.loop,
                    rows.index,
                    cols.index,
                    store.access.dtype.itemsize,
                    2 if store.loop >= 0 else 1,
                ),
                store,
            )
        )
    if program.vec_slots:
        # One vector (at most the TMEM lane count of entries) per epilogue
        # warp.
        items.append(
            (
                Tcgen05SmemItem(ITEM_SCRATCH, -1, -1, -1, 4, TCGEN05_EPILOGUE_WARPS),
                None,
            )
        )
    return items


def item_bytes(item: Tcgen05SmemItem, resolved: Sequence[int]) -> int:
    """Bytes of one copy of ``item`` (aligned to the swizzle atom row)."""
    if item.kind == ITEM_SCRATCH:
        return SCRATCH_ENTRIES * item.elem_bytes
    return _align(
        resolved[item.rows] * resolved[item.cols] * item.elem_bytes, SLAB_ALIGNMENT
    )


@dataclasses.dataclass(frozen=True)
class RegionLayout:
    # Item -> byte offset: absolute for single / multi-copy items, within
    # the ring slot for ring items.
    offsets: tuple[int, ...]
    ring_base: dict[int, int]
    ring_bytes: dict[int, int]
    total: int


def region_layout(
    items: Sequence[Tcgen05SmemItem],
    loops: Sequence[LoopFacts],
    resolved: Sequence[int],
    stages: int,
) -> RegionLayout:
    """Frame layout: a loop's items follow the frames of the loops nested
    in it (sibling frames share one extent); a frame's ring (every ring
    item of the loop, one slot per stage) comes before its other items."""
    children: dict[int, list[int]] = {}
    for index, loop in enumerate(loops):
        children.setdefault(loop.parent, []).append(index)
    offsets = [0] * len(items)
    ring_base: dict[int, int] = {}
    ring_bytes: dict[int, int] = {}

    def layout(frame: int, base: int) -> int:
        offset = max(
            (layout(child, base) for child in children.get(frame, ())), default=base
        )
        slot = 0
        for index, item in enumerate(items):
            if item.frame == frame and item.copies == 0:
                offsets[index] = slot
                slot += item_bytes(item, resolved)
        if slot:
            ring_base[frame] = offset
            ring_bytes[frame] = slot
            offset += slot * stages
        for index, item in enumerate(items):
            if item.frame == frame and item.copies != 0:
                offsets[index] = offset
                offset += item_bytes(item, resolved) * item.copies
        return offset

    total = layout(-1, 0)
    return RegionLayout(tuple(offsets), ring_base, ring_bytes, total)


def region_barrier_bytes(tc: Tcgen05Facts, stages: int) -> int:
    """The mbarriers (one per ring slot, one per staged region, the commit
    barrier) and the TMEM slot."""
    return 8 * (stages * tc.ring_loops + tc.staged_regions + 1) + 16


def region_smem_bytes(
    facts: CuteMmaDagFacts, resolved: Sequence[int], stages: int
) -> int:
    tc = facts.tcgen05
    assert tc is not None
    layout = region_layout(tc.items, facts.loops, resolved, stages)
    return layout.total + region_barrier_bytes(tc, stages)


def region_tmem_layout(
    facts: CuteMmaDagFacts, resolved: Sequence[int]
) -> tuple[dict[int, int], int]:
    """TMEM column of every TMEM-resident slot and the columns used: slots
    that are never live together (the planner's live groups, plus the pairs
    the facts name) share columns -- wy's six 64-column slots fit in 192
    instead of 384, two CTAs per SM instead of one."""
    tc = facts.tcgen05
    assert tc is not None
    conflicts: set[tuple[int, int]] = set(tc.tmem_conflicts)
    for group in facts.live_groups:
        for a in group:
            for b in group:
                if a != b:
                    conflicts.add((a, b))
    placed: dict[int, tuple[int, int]] = {}
    total = 0
    for slot in tc.tmem_slots:
        _rows, cols = facts.frags[slot]
        width = _align(resolved[cols], TCGEN05_SLOT_ALIGN)
        busy = sorted(
            span
            for other, span in placed.items()
            if (slot, other) in conflicts or (other, slot) in conflicts
        )
        column = 0
        for start, end in busy:
            if column + width <= start:
                break
            column = max(column, end)
        placed[slot] = (column, column + width)
        total = max(total, column + width)
    return {slot: span[0] for slot, span in placed.items()}, total


def region_tmem_columns(facts: CuteMmaDagFacts, resolved: Sequence[int]) -> int:
    return region_tmem_layout(facts, resolved)[1]


def tmem_conflicts(program: MmaDagProgram) -> tuple[tuple[int, int], ...]:
    """Slot pairs beyond the live groups that must not share columns: a
    carry and the slot seeding it, a dot's accumulator and its result, and
    every pair of slots a region's segment touches -- the batches reorder
    the segment (independent dots issue together ahead of the work
    reading earlier results), so two slots whose program-order lifetimes
    do not overlap may be live at once (bwd_o_kda's ``dA`` and ``dv_new``
    dots of one batch)."""
    pairs: set[tuple[int, int]] = set()

    def conflict(a: int, b: int) -> None:
        if a != b:
            pairs.add((min(a, b), max(a, b)))

    for loop in program.loops:
        for carry in loop.carries:
            if carry.init_frag is not None:
                source = program.frag_slots[program.resolve(carry.init_frag)]
                conflict(source, carry.slot)
    touched: dict[int, set[int]] = {}
    for dot in program.dots:
        if dot.acc_slot is not None:
            conflict(dot.acc_slot, dot.slot)
            touched.setdefault(dot.loop, set()).add(dot.acc_slot)
        touched.setdefault(dot.loop, set()).add(dot.slot)
        for operand in (dot.a, dot.b):
            if operand.frag is not None:
                touched[dot.loop].add(program.frag_slots[operand.frag])
    for derived in program.derived:
        touched.setdefault(derived.loop, set()).add(derived.slot)
        touched[derived.loop].update(program.frag_slots[frag] for frag in derived.frags)
    for store in program.stores:
        touched.setdefault(store.loop, set()).add(program.frag_slots[store.frag])
    for reduce in program.reduces:
        if reduce.frag is not None:
            touched.setdefault(reduce.loop, set()).add(program.frag_slots[reduce.frag])
    for loop in program.loops:
        touched.setdefault(loop.index, set()).update(
            carry.slot for carry in loop.carries
        )
    for slots in touched.values():
        for a in slots:
            for b in slots:
                conflict(a, b)
    # A slot defined in one region and read in a loop that does not contain
    # the definition is read again by every iteration of the outermost such
    # loop: live throughout it, so it conflicts with every slot touched
    # anywhere in that loop (the planner's liveness is linear and ends it at
    # the read; pk_h1: a root carry read by an inner loop's dot shared its
    # columns with a fresh dot of the enclosing loop).
    ancestors: dict[int, list[int]] = {-1: []}
    for loop in program.loops:
        chain = [loop.index]
        parent = loop.parent
        while parent >= 0:
            chain.append(parent)
            parent = program.loops[parent].parent
        ancestors[loop.index] = chain
    children: dict[int, list[int]] = {}
    for loop in program.loops:
        children.setdefault(loop.parent, []).append(loop.index)

    def closure(region: int) -> set[int]:
        slots = set(touched.get(region, ()))
        for child in children.get(region, ()):
            slots |= closure(child)
        return slots

    defined: dict[int, int] = {}
    uses: dict[int, set[int]] = {}
    for dot in program.dots:
        defined[dot.slot] = dot.loop
        for operand in (dot.a, dot.b):
            if operand.frag is not None:
                uses.setdefault(program.frag_slots[operand.frag], set()).add(dot.loop)
        if dot.acc_slot is not None:
            uses.setdefault(dot.acc_slot, set()).add(dot.loop)
    for derived in program.derived:
        defined[derived.slot] = derived.loop
        for frag in derived.frags:
            uses.setdefault(program.frag_slots[frag], set()).add(derived.loop)
    for store in program.stores:
        uses.setdefault(program.frag_slots[store.frag], set()).add(store.loop)
    for reduce in program.reduces:
        if reduce.frag is not None:
            uses.setdefault(program.frag_slots[reduce.frag], set()).add(reduce.loop)
    for loop in program.loops:
        for carry in loop.carries:
            defined[carry.slot] = loop.parent
            uses.setdefault(carry.slot, set()).add(loop.index)
    for slot, regions in uses.items():
        definition = defined.get(slot, -1)
        holders = {definition, *ancestors[definition]}
        for region in regions:
            outermost = None
            for loop in ancestors[region]:
                if loop in holders:
                    break
                outermost = loop
            if outermost is not None:
                for other in closure(outermost):
                    conflict(slot, other)
    return tuple(sorted(pairs))


def region_facts(program: MmaDagProgram) -> Tcgen05Facts | None:
    """The tcgen05 facts of a region program (None: declined)."""
    if region_program_reason(program) is not None:
        return None
    items = tuple(item for item, _payload in region_items(program))
    tiles: list[Tcgen05TileFacts] = []
    exchanges: list[Tcgen05ExchangeFacts] = []
    for item, payload in region_items(program):
        if item.kind == ITEM_TILE:
            dot, operand = cast("tuple[Dot, Operand]", payload)
            load = operand.load
            assert load is not None
            tiles.append(
                Tcgen05TileFacts(
                    rows=item.rows,
                    cols=item.cols,
                    elem_bytes=item.elem_bytes,
                    source_bytes=load.dtype.itemsize,
                    region=REGION_LOOP if dot.loop >= 0 else REGION_EPILOGUE,
                )
            )
        elif item.kind == ITEM_EXCHANGE:
            dot, operand = cast("tuple[Dot, Operand]", payload)
            exchanges.append(
                Tcgen05ExchangeFacts(
                    slot=program.frag_slots[cast("Node", operand.frag)],
                    rows=item.rows,
                    cols=item.cols,
                    elem_bytes=item.elem_bytes,
                    region=REGION_LOOP if dot.loop >= 0 else REGION_EPILOGUE,
                )
            )
    derived_cols = tuple(
        (
            *(program.slots[program.frag_slots[frag]][1].index for frag in step.frags),
            step.cols.index,
        )
        for step in program.derived
    )
    ring_loops = sum(1 for loop in program.loops if program.loop_has_ring(loop.index))
    staged_regions = len(
        {
            program.level(operand.load)
            for dot in program.dots
            for operand in (dot.a, dot.b)
            if operand.load is not None and not program.is_ring(dot, operand)
        }
    )
    return Tcgen05Facts(
        tiles=tuple(tiles),
        exchanges=tuple(exchanges),
        loop_stores=(),
        epilogue_stores=(),
        derived_cols=derived_cols,
        general_loop=True,
        regions=True,
        items=items,
        tmem_slots=tmem_slots(program),
        tmem_conflicts=tmem_conflicts(program),
        tf32=any(dot.a.dtype == torch.float32 for dot in program.dots),
        ring_loops=ring_loops,
        staged_regions=staged_regions,
    )


def region_config_reason(
    facts: CuteMmaDagFacts, resolved: Sequence[int], stages: int
) -> str | None:
    """The region-specific legality of the resolved sizes (the shared tile
    / fragment rules ran already)."""
    columns = region_tmem_columns(facts, resolved)
    if columns > TCGEN05_MAX_TMEM_COLS:
        return (
            f"the fragments need {columns} TMEM columns (limit {TCGEN05_MAX_TMEM_COLS})"
        )
    smem = region_smem_bytes(facts, resolved, stages)
    if facts.smem_capacity and smem > facts.smem_capacity:
        return (
            f"{stages} tcgen05 stages need {smem} B of shared memory "
            f"(capacity {facts.smem_capacity})"
        )
    return None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def plan_tcgen05_regions(
    program: MmaDagProgram,
    facts: CuteMmaDagFacts,
    sizes: tuple[int, ...],
    routes: Mapping[tuple[Node, str], str],
    warps: int,
    stages: int,
    store_mode: str,
) -> Tcgen05Plan:
    env = CompileEnvironment.current()
    tc = facts.tcgen05
    assert tc is not None and tc.regions
    items = region_items(program)
    layout = region_layout([item for item, _p in items], facts.loops, sizes, stages)
    tma: list[TmaSpec] = []
    # One tensor map per (tensor, dims, box, swizzle): the loads of one
    # tensor in several loops share it (the coordinates come at issue time).
    shared_maps: dict[tuple[object, ...], TmaSpec] = {}

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
        key = (access.fake, dims, tuple(box), swizzle, elem_bytes)
        shared = shared_maps.get(key)
        if shared is not None:
            return dataclasses.replace(shared, access=access, offset=at)
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
        shared_maps[key] = spec
        return spec

    tiles: dict[tuple[Node, str], Tcgen05Tile] = {}
    by_key: dict[tuple[object, ...], Tcgen05Tile] = {}
    ring_tiles: dict[int, list[Tcgen05Tile]] = {}
    staged_tiles: dict[int, list[Tcgen05Tile]] = {}
    exchange_tiles: list[Tcgen05Tile] = []
    raw_offsets: dict[tuple[object, ...], int] = {}
    for (item, payload), offset in zip(items, layout.offsets, strict=True):
        if item.kind == ITEM_RAW:
            _dot, operand = cast("tuple[Dot, Operand]", payload)
            raw_offsets[tile_key(program, operand)] = offset
    for (item, payload), offset in zip(items, layout.offsets, strict=True):
        if item.kind not in (ITEM_TILE, ITEM_EXCHANGE):
            continue
        dot, operand = cast("tuple[Dot, Operand]", payload)
        rows_axis, cols_axis, _k_major = operand_geometry(program, operand)
        rows, cols = sizes[rows_axis.index], sizes[cols_axis.index]
        es = operand.dtype.itemsize
        atom = tcgen05_atom_bytes(cols * es)
        if atom is None:
            raise _decline(f"operand {operand.value.name} has no swizzled UMMA layout")
        swizzle = _SWIZZLE_BY_ROW_BYTES[atom]
        key = tile_key(program, operand)
        if item.kind == ITEM_EXCHANGE:
            tile = Tcgen05Tile(
                key=key,
                tag=f"t{len(by_key)}",
                region=REGION_EXCHANGE,
                operand=operand,
                route=ROUTE_EXCHANGE,
                rows=rows,
                cols=cols,
                elem_bytes=es,
                atom_bytes=atom,
                swizzle=swizzle,
                slabs=(cols * es) // atom,
                offset=offset,
                source_slot=program.frag_slots[cast("Node", operand.frag)],
                frame=item.frame,
            )
            exchange_tiles.append(tile)
        else:
            route = routes[(dot.node, operand.role)]
            load = operand.load
            assert load is not None
            if route != ROUTE_SYNC and len(_tma_dims(load)) > TMA_MAX_RANK:
                route = ROUTE_SYNC
            spec = raw_spec = None
            raw_offset = raw_row_bytes = 0
            src_rows_axis, src_cols_axis = source_geometry(operand)
            src_rows, src_cols = sizes[src_rows_axis.index], sizes[src_cols_axis.index]
            if operand.transposed_tile and route != ROUTE_CONVERT:
                raise _decline(
                    f"operand {operand.value.name}: a transposed fp32 tile needs the "
                    "widen pass"
                )
            if es == 4 and operand.trans and not operand.transposed_tile:
                raise _decline(
                    f"operand {operand.value.name}: an MN-major tf32 tile needs the "
                    "ATOM_32B TMA swizzle"
                )
            if route in (ROUTE_DIRECT, ROUTE_INPLACE):
                spec = tma_spec(
                    load,
                    rows_axis,
                    cols_axis,
                    rows,
                    cols,
                    es,
                    swizzle,
                    atom,
                    offset,
                )
            elif route == ROUTE_CONVERT:
                raw_offset = raw_offsets[key]
                raw_row_bytes = src_cols * load.dtype.itemsize
                raw_spec = tma_spec(
                    load,
                    src_rows_axis,
                    src_cols_axis,
                    src_rows,
                    src_cols,
                    load.dtype.itemsize,
                    "none",
                    raw_row_bytes,
                    raw_offset,
                )
            elif route == ROUTE_SYNC and es == 4:
                raise _decline(
                    f"operand {operand.value.name}: fp32 operands take TMA routes only"
                )
            elif route != ROUTE_SYNC:
                raise _decline(f"operand route {route!r} is not a tcgen05 route")
            ring = item.copies == 0
            tile = Tcgen05Tile(
                key=key,
                tag=f"t{len(by_key)}",
                region=REGION_LOOP if ring else REGION_EPILOGUE,
                operand=operand,
                route=route,
                rows=rows,
                cols=cols,
                elem_bytes=es,
                atom_bytes=atom,
                swizzle=swizzle,
                slabs=(cols * es) // atom,
                offset=offset,
                tma=spec,
                raw_offset=raw_offset,
                raw_row_bytes=raw_row_bytes,
                raw_tma=raw_spec,
                frame=item.frame,
                ring=ring,
            )
            if ring:
                ring_tiles.setdefault(item.frame, []).append(tile)
            else:
                staged_tiles.setdefault(item.frame, []).append(tile)
        by_key[key] = tile
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            tiles[(dot.node, operand.role)] = by_key[tile_key(program, operand)]
    # Fragment slots: TMEM columns at 32-column-aligned offsets, shared by
    # slots never live together (register-only slots take none).
    if tuple(tc.tmem_slots) != tmem_slots(program):
        raise _decline("the tensor-memory slots changed since admission")
    tmem_columns, column = region_tmem_layout(facts, sizes)
    slots: list[SlotGeometry] = []
    for index, (rows_axis, cols_axis) in enumerate(program.slots):
        rows, cols = sizes[rows_axis.index], sizes[cols_axis.index]
        slots.append(
            SlotGeometry(rows=rows, cols=cols, column=tmem_columns.get(index, 0))
        )
    staged: list[Store] = []
    staging_offsets: dict[Node, int] = {}
    scratch_offset = -1
    for (item, payload), offset in zip(items, layout.offsets, strict=True):
        if item.kind == ITEM_SCRATCH:
            scratch_offset = offset
        if item.kind != ITEM_STAGING or store_mode != MMA_DAG_STORE_STAGED:
            continue
        store = cast("Store", payload)
        slot = slots[program.frag_slots[store.frag]]
        if store.region == "loop" and store.source == "carry_in":
            reason = staged_store_reason(env, store, slot.cols)
        else:
            reason = epilogue_staged_reason(env, store, slot.cols)
        if reason is not None:
            continue
        staged.append(store)
        staging_offsets[store.access.node] = offset
    smem = layout.total + region_barrier_bytes(tc, stages)
    if facts.smem_capacity and smem > facts.smem_capacity:
        raise _decline(
            f"{smem} B of shared memory exceed the capacity {facts.smem_capacity}"
        )
    if column > TCGEN05_MAX_TMEM_COLS:
        raise _decline(f"{column} TMEM columns exceed {TCGEN05_MAX_TMEM_COLS}")
    ring_tx = {
        loop: sum(
            (tile.tma or tile.raw_tma).bytes  # type: ignore[union-attr]
            for tile in group
            if tile.tma or tile.raw_tma
        )
        for loop, group in ring_tiles.items()
    }
    staged_tx = {
        region: sum(
            (tile.tma or tile.raw_tma).bytes  # type: ignore[union-attr]
            for tile in group
            if tile.tma or tile.raw_tma
        )
        for region, group in staged_tiles.items()
    }
    return Tcgen05Plan(
        warps=warps,
        stages=stages,
        slots=tuple(slots),
        tiles=tiles,
        loop_tiles=(),
        epilogue_tiles=(),
        exchange_tiles=tuple(exchange_tiles),
        tma=tuple(tma),
        stage_bytes=0,
        tx_bytes=0,
        fixed_offset=0,
        epilogue_tx_bytes=0,
        staged_stores=tuple(staged),
        staging_offsets=staging_offsets,
        epilogue_staging_offset=0,
        tmem_cols=tcgen05_tmem_cols(column),
        smem_bytes=layout.total,
        regions=True,
        ring_offsets=dict(layout.ring_base),
        ring_bytes=dict(layout.ring_bytes),
        ring_tiles={loop: tuple(group) for loop, group in ring_tiles.items()},
        ring_tx=ring_tx,
        staged_tiles={region: tuple(group) for region, group in staged_tiles.items()},
        staged_tx=staged_tx,
        scratch_offset=scratch_offset,
        tmem_slots=tuple(tc.tmem_slots),
    )


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

Block = tuple[list[Step], list[Dot]]


class _Tcgen05RegionEmitter(_Tcgen05Emitter):
    """The tcgen05 renderer over the region tree (see the module doc)."""

    def __init__(self, cg: GenerateAST, plan: MmaDagPlan) -> None:
        super().__init__(cg, plan)
        self.values = {}
        # Exchange tiles valid at this point (sharing keys), the fragments
        # defined so far, the staged tiles whose passes ran in the current
        # execution of their frame, and whether the latest commit was waited.
        self.written: set[tuple[object, ...]] = set()
        self.defined: set[Node] = set()
        self.settled: set[tuple[object, ...]] = set()
        self.waited = True
        # Staged body stores waiting for the next CTA barrier: (store,
        # parity expression, loop values).
        self.drains: list[tuple[Store, str, dict[int, str]]] = []
        self.current_blocks: list[Block] = []
        self.ring_waited: set[int] = set()
        # Exchange tiles written at the root (valid for the whole kernel).
        self.root_written: set[tuple[object, ...]] = set()
        program = self.program
        self.ring_loops = [
            loop.index for loop in program.loops if self.tc.ring_tiles.get(loop.index)
        ]
        self.counted = {
            loop.index
            for loop in program.loops
            if loop.index in self.tc.ring_tiles
            or loop.index in self.tc.staged_tiles
            or any(store.loop == loop.index for store in self.tc.staged_stores)
        }

    # -- naming ---------------------------------------------------------------

    def counter(self, loop: int) -> str:
        return f"md_rk{loop}"

    def sf_name(self, region: int) -> str:
        return "md_sfr" if region < 0 else f"md_sf{region}"

    def stage_expr(self, loop: int) -> str:
        return f"md_st{loop}" if loop in self.ring_loops else "0"

    def parity_expr(self, loop: int) -> str:
        return f"md_pt{loop}" if loop >= 0 else "0"

    def vreg(self, slot: int, atom: int = 0, half: int = 0) -> str:  # type: ignore[override]
        return f"md_vv{slot}"

    def k_major(self, operand: Operand) -> bool:
        return operand_geometry(self.program, operand)[2]

    def emit_dot_mmas(self, dot: Dot, stage: str, accumulate: bool) -> None:  # type: ignore[override]
        """Warp 0: the K / k_elems MMAs of ``dot`` over its tiles (F16 kind
        for 16-bit operands, TF32 for fp32)."""
        a_tile, b_tile = self.tc.tile(dot, dot.a), self.tc.tile(dot, dot.b)
        a_major, b_major = self.k_major(dot.a), self.k_major(dot.b)
        index = self.dot_index[dot.node]
        self.desc(f"md_da{index}", a_tile, stage, a_major)
        self.desc(f"md_db{index}", b_tile, stage, b_major)
        kind = "TF32" if dot.a.dtype == torch.float32 else "F16"
        k = self.plan.size(dot.k)
        for step in range(k // a_tile.k_elems):
            with self.nest(f"if {_PRIMS}.elect_sync():"):
                enable = "True" if accumulate or step else "False"
                self.emit(
                    f"{_PRIMS}.tcgen05_mma({_PRIMS}.Tcgen05MMAKind.{kind}, {_PRIMS}.CTAGroup.CTA_1, "
                    f"md_acc{dot.slot}, "
                    f"md_da{index}.advance_start_address({a_tile.k_step_offset(step, a_major)}), "
                    f"md_db{index}.advance_start_address({b_tile.k_step_offset(step, b_major)}), "
                    f"md_idesc{index}, {enable})"
                )

    def emit_exchange(self, tile: Tcgen05Tile) -> None:  # type: ignore[override]
        """Epilogue warps: the fragment operand's value per element of this
        thread's row into the UMMA exchange tile -- 16-bit values packed in
        pairs, fp32 values as four-wide packets."""
        if tile.elem_bytes != 4:
            super().emit_exchange(tile)
            return
        operand = tile.operand
        slot = tile.source_slot
        assert slot is not None
        value = self.pass_value(operand)
        self.hoist_scalars(value)
        row = self.row_var(slot)
        with self.row_guard(slot):
            if operand.transposed_tile:
                # Element (row, c) of the fragment is row ``c`` of the [N, K]
                # tile at column ``row``: one 4-byte store per element.
                chunk = f"({row} // 4)"
                within = f"({row} % 4) * 4"
                for c in range(self.local_cols(slot)):
                    self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                    coords = self.element_coords(value, slot, c, operand.swap)
                    scalar = self.evaluator.evaluate(value, coords)
                    address = self.tile_address(
                        tile, "0", self.col_expr(slot, c), chunk
                    )
                    self.emit(f"{_RUNTIME}.sts_f32({address} + {within}, {scalar})")
                self.evaluator.reg = None
                return
            for col in range(0, self.local_cols(slot), 4):
                values = []
                for c in range(col, col + 4):
                    self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                    coords = self.element_coords(value, slot, c, operand.swap)
                    values.append(self.evaluator.evaluate(value, coords))
                self.evaluator.reg = None
                chunk = self.chunk_expr(slot, col // 4, 4)
                address = self.tile_address(tile, "0", row, chunk)
                self.emit(f"{_RT}.sts_v4_f32({address}, {', '.join(values)})")

    def emit_convert_pass(  # type: ignore[override]
        self, tile: Tcgen05Tile, stage: str, idx_base: str, threads: int
    ) -> None:
        """A 16-bit raw tile widened through the pointwise DAG into the fp32
        (tf32) operand tile: raw packet ``p`` of a row holds columns ``8p ..
        8p + 7``, fp32 chunks ``2p`` and ``2p + 1``; fp32 raw tiles narrow
        as in the simple emitter."""
        operand = tile.operand
        load = operand.load
        assert load is not None
        if load.dtype == torch.float32 or tile.elem_bytes != 4:
            super().emit_convert_pass(tile, stage, idx_base, threads)
            return
        unpack = "unpack_f16x2" if load.dtype == torch.float16 else "unpack_bf16x2"
        value = self.pass_value(operand)
        self.hoist_scalars(value)
        raw_cpr = tile.raw_row_bytes // 16

        def body(prow: str, pch: str) -> None:
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({self.raw_address(tile, stage, prow, pch)})"
            )
            for q, word in enumerate(words):
                lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                self.evaluator.tile_leaves[(load.node, 2 * q)] = lo
                self.evaluator.tile_leaves[(load.node, 2 * q + 1)] = hi
            if operand.transposed_tile:
                # Source element (prow, 8 pch + e) is row ``8 pch + e`` of the
                # [N, K] tile at column ``prow``.
                chunk = f"({prow} // 4)"
                within = f"({prow} % 4) * 4"
                for e in range(8):
                    self.evaluator.element = e
                    coords = self.tile_coords(operand, prow, f"({pch} * 8 + {e})")
                    scalar = self.evaluator.evaluate(value, coords)
                    address = self.tile_address(
                        tile, stage, f"({pch} * 8 + {e})", chunk
                    )
                    self.emit(f"{_RUNTIME}.sts_f32({address} + {within}, {scalar})")
                self.evaluator.element = None
                return
            for half in range(2):
                values = []
                for e in range(4 * half, 4 * half + 4):
                    self.evaluator.element = e
                    coords = self.tile_coords(operand, prow, f"({pch} * 8 + {e})")
                    values.append(self.evaluator.evaluate(value, coords))
                self.evaluator.element = None
                chunk = f"({pch} * 2 + {half})" if half else f"({pch} * 2)"
                self.emit(
                    f"{_RT}.sts_v4_f32({self.tile_address(tile, stage, prow, chunk)}, "
                    f"{', '.join(values)})"
                )

        src_rows = (
            self.plan.size(operand.cols) if operand.transposed_tile else tile.rows
        )
        self.packets(src_rows * raw_cpr, threads, idx_base, raw_cpr, body, tag=tile.tag)

    def tile_base(self, tile: Tcgen05Tile, stage: str) -> str:  # type: ignore[override]
        if tile.ring:
            return (
                f"md_s0 + {self.tc.ring_offsets[tile.frame]} + ({stage}) * "
                f"{self.tc.ring_bytes[tile.frame]} + {tile.offset}"
            )
        return f"md_s0 + {tile.offset}"

    def raw_address(self, tile: Tcgen05Tile, stage: str, prow: str, pch: str) -> str:  # type: ignore[override]
        if tile.ring:
            base = (
                f"md_s0 + {self.tc.ring_offsets[tile.frame]} + ({stage}) * "
                f"{self.tc.ring_bytes[tile.frame]} + {tile.raw_offset}"
            )
        else:
            base = f"md_s0 + {tile.raw_offset}"
        return f"{base} + {prow} * {tile.raw_row_bytes} + {pch} * 16"

    def staging_base(self, store: Store, parity: str) -> str:  # type: ignore[override]
        slot = self.slot(self.program.frag_slots[store.frag])
        offset = self.tc.staging_offsets[store.access.node]
        if store.loop >= 0:
            tile = _align(
                slot.rows * slot.cols * store.access.dtype.itemsize, SLAB_ALIGNMENT
            )
            return f"md_s0 + {offset} + ({parity}) * {tile}"
        return f"md_s0 + {offset}"

    def scratch_base(self) -> str:  # type: ignore[override]
        assert self.tc.scratch_offset >= 0
        return f"md_s0 + {self.tc.scratch_offset}"

    @property
    def epilogue_sync(self) -> str:
        return (
            f"cute.arch.barrier(barrier_id={EPILOGUE_BARRIER_ID}, "
            f"number_of_threads={32 * TCGEN05_EPILOGUE_WARPS})"
        )

    # -- commits --------------------------------------------------------------

    def wait_latest(self) -> None:
        """Every thread waits for the latest MMA commit (immediate when it
        was waited already or none was issued)."""
        if self.waited:
            return
        self.emit(f"{_RT}.mbar_wait(md_cb, (md_cn + 1) & 1)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)")
        self.waited = True

    def emit_commit(self) -> None:
        with self.nest(f"if {_PRIMS}.elect_sync():"):
            self.emit(f"{_PRIMS}.tcgen05_commit(md_cb, group={_PRIMS}.CTAGroup.CTA_1)")

    # -- TMA ------------------------------------------------------------------

    def region_tma_coords(
        self, spec: TmaSpec, values: Mapping[int, str], slab: int
    ) -> list[str]:
        access = spec.access
        coords = []
        for position, dim in enumerate(spec.dims):
            subscript = access.subscripts[dim]
            if subscript.kind == "scalar":
                assert subscript.expr is not None
                value = self.uniform(subscript.expr, dict(values))
            else:
                axis = subscript.axis
                assert axis is not None
                if axis.kind == AXIS_RDIM:
                    value = "0"
                elif axis.kind == AXIS_LOOP:
                    offset = self.axis_offset(axis, values[axis.loop])
                    value = f"({offset}) * {self.plan.size(axis)}"
                else:
                    value = self.axis_base(axis)
            if position == spec.slab_dim and slab:
                value = f"{value} + {slab * spec.slab_elems}"
            coords.append(f"cutlass.Int32({value})")
        return coords

    def emit_tma_group(
        self,
        tiles: Sequence[Tcgen05Tile],
        values: Mapping[int, str],
        barrier: str,
        tx_bytes: int,
        base: str,
    ) -> None:
        """One elected thread: expect ``tx_bytes`` on ``barrier`` and issue
        every box of ``tiles`` at ``base`` + the tile offsets."""
        self.emit(f"{_PRIMS}.mbarrier_arrive_expect_tx({barrier}, {tx_bytes})")
        for tile in tiles:
            spec = tile.tma or tile.raw_tma
            if spec is None:
                continue
            for slab in range(spec.slabs):
                dst = f"md_smem + ({base} + {spec.offset + slab * spec.slab_bytes})"
                coords = ", ".join(self.region_tma_coords(spec, values, slab))
                self.emit(
                    f"{_PRIMS}.cp_async_bulk_tensor_shared_cta_global({dst}, "
                    f"{spec.param}.get_ptr(), [{coords}], {barrier})"
                )

    def emit_ring_issue(self, loop: int, values: Mapping[int, str], count: str) -> None:
        """The ring tiles of iteration ``count`` (the running count) of loop
        ``loop`` into slot ``count % stages``."""
        tc = self.tc
        stage = f"({count}) % {tc.stages}"
        self.emit_tma_group(
            tc.ring_tiles[loop],
            values,
            f"md_full{loop} + ({stage})",
            tc.ring_tx[loop],
            f"{tc.ring_offsets[loop]} + ({stage}) * {tc.ring_bytes[loop]}",
        )

    def emit_staged_issue(self, region: int) -> None:
        tiles = self.tc.staged_tiles.get(region, ())
        if not any(tile.tma or tile.raw_tma for tile in tiles):
            return
        with self.nest("if md_warp == 0:"), self.nest(f"if {_PRIMS}.elect_sync():"):
            self.emit_tma_group(
                tiles, self.values, self.sf_name(region), self.tc.staged_tx[region], "0"
            )

    def staged_phase(self, region: int) -> str:
        return "0" if region < 0 else f"({self.counter(region)}) & 1"

    # -- settling staged tiles --------------------------------------------------

    def settle_tiles(self, consumers: Sequence[Dot], region: int) -> None:
        """Before ``consumers`` run in region ``region``: the staged tiles
        they read (of ``region`` and the regions enclosing it, issued
        already) that are not yet settled in this execution of their frame
        -- wait for their TMA, run their passes, publish them (fence + CTA
        barrier)."""
        tc = self.tc
        keys = {
            tile_key(self.program, operand)
            for dot in consumers
            for operand in (dot.a, dot.b)
            if operand.load is not None
        }
        active = {-1, *self.enclosing(region)}
        by_region: dict[int, list[Tcgen05Tile]] = {}
        for frame, tiles in tc.staged_tiles.items():
            if frame not in active:
                continue
            for tile in tiles:
                if tile.key in keys and tile.key not in self.settled:
                    by_region.setdefault(frame, []).append(tile)
        if not by_region:
            return
        passes = False
        for frame, tiles in by_region.items():
            wait = (
                f"{_RT}.mbar_wait({self.sf_name(frame)}, {self.staged_phase(frame)})"
                if tc.staged_tx.get(frame)
                else None
            )
            if self.has_passes(tiles):
                self.emit_passes(tiles, "0", wait, "md_tidx", self.plan.threads)
                passes = True
            elif wait is not None:
                self.emit(wait)
            for tile in tiles:
                self.settled.add(tile.key)
        if passes:
            self.emit(self.sync)

    # -- blocks ---------------------------------------------------------------

    def region_blocks(self, steps: Sequence[Step]) -> list[Block]:
        """``steps`` (no loops) as blocks of epilogue-warp work followed by
        a batch of dots issued together (see the simple emitter's
        ``loop_blocks``; vector steps read no tensor memory)."""
        program = self.program
        blocks: list[Block] = []
        work: list[Step] = []
        dots: list[Dot] = []
        produced: set[int] = set()
        batch_reads: set[int] = set()
        # Work that reads the pending batch's results waits for the next
        # block; later independent dots still join the batch (one commit
        # round trip instead of two).
        deferred: list[Step] = []
        deferred_reads: set[int] = set()
        deferred_writes: set[int] = set()
        # A tile-only derived value without a tensor-memory slot is
        # recomputed in every block that reads it (the registers do not
        # cross the block).
        register_only: dict[int, Derived] = {}

        def flush() -> None:
            nonlocal \
                work, \
                dots, \
                produced, \
                batch_reads, \
                deferred, \
                deferred_reads, \
                deferred_writes
            if work or dots:
                blocks.append((work, dots))
            work, dots, produced, batch_reads = deferred, [], set(), set()
            deferred, deferred_reads, deferred_writes = [], set(), set()

        for step in steps:
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
                if (
                    reads & produced
                    or reads & deferred_writes
                    or step.slot in deferred_reads
                    or step.slot in deferred_writes
                ):
                    flush()
                dots.append(step)
                produced.add(step.slot)
                batch_reads |= reads
                continue
            if isinstance(step, Derived):
                reads = {program.frag_slots[frag] for frag in step.frags}
                writes = step.slot
                if writes not in self.tc.tmem_slots:
                    register_only[writes] = step
            elif isinstance(step, Store):
                reads = {program.frag_slots[step.frag]}
                writes = -1
            elif isinstance(step, Reduce):
                reads = (
                    {program.frag_slots[step.frag]} if step.frag is not None else set()
                )
                writes = -1
            elif isinstance(step, (VecOp, VecStore)):
                reads = set()
                writes = -1
            else:
                raise AssertionError(step)
            if (
                (reads & produced)
                or writes in batch_reads
                or writes in produced
                or deferred
            ):
                # Behind the batch (or behind deferred work it may depend on:
                # program order among the deferred steps is kept).
                target = deferred
                deferred_reads |= reads
                if writes >= 0:
                    deferred_writes.add(writes)
            else:
                target = work
            for slot in sorted(reads):
                source = register_only.get(slot)
                if source is not None and source not in target:
                    target.append(source)
            target.append(step)
        flush()
        if work:
            blocks.append((work, []))
        return blocks

    def derived_needs_tmem(self, step: Derived) -> bool:  # type: ignore[override]
        """A derived value goes to its slot when a dot reads it (as an
        accumulator or an operand), a loop carries it, or a step outside its
        block reads it (the registers do not cross the block)."""
        program = self.program
        node = step.node
        if step.slot not in self.tc.tmem_slots:
            return False
        for loop in program.loops:
            if any(program.resolve(carry.output) is node for carry in loop.carries):
                return True
        for dot in program.dots:
            if dot.acc_frag is node or any(
                operand.frag is node for operand in (dot.a, dot.b)
            ):
                return True
        home: list[Step] | None = None
        for work, _dots in self.current_blocks:
            if step in work:
                home = work
        readers: list[Step] = [
            *(d for d in program.derived if node in d.frags),
            *(s for s in program.stores if s.frag is node),
            *(r for r in program.reduces if r.frag is node),
        ]
        return any(home is None or reader not in home for reader in readers)

    def emit_ld(self, slot: int) -> None:
        assert slot in self.tc.tmem_slots, slot
        super().emit_ld(slot)

    def emit_st_array(self, name: str, slot: int, chunk: int) -> None:
        assert slot in self.tc.tmem_slots, slot
        super().emit_st_array(name, slot, chunk)

    def reads_tmem(self, steps: Sequence[Step]) -> bool:
        for step in steps:
            if isinstance(step, (Derived, Store)) or (
                isinstance(step, Reduce) and step.frag is not None
            ):
                return True
        return False

    def emit_blocks(self, blocks: list[Block], loop: Loop | None) -> None:
        """The blocks of a segment: per block the epilogue-warp work (the
        loop's carry snapshots in the first block, derived values, stores,
        reductions, exchange tiles and accumulator preparation of the
        batch's dots), a barrier and warp 0's batch issue with its commit;
        staged stores drain behind the next barrier."""
        tc, program = self.tc, self.program
        region = loop.index if loop is not None else -1
        parity = self.parity_expr(region)
        self.current_blocks = blocks
        snapshots = (
            [
                store
                for store in program.stores
                if store.loop == region and store.source == "carry_in"
            ]
            if loop is not None
            else []
        )
        for index, (work, dots) in enumerate(blocks):
            exchanges: list[Tcgen05Tile] = []
            for dot in dots:
                for operand in (dot.a, dot.b):
                    if operand.kind != "frag":
                        continue
                    tile = tc.tile(dot, operand)
                    if tile.key not in self.written and tile not in exchanges:
                        exchanges.append(tile)
            preps = [dot for dot in dots if self.prepares(dot)]
            first_snapshots = snapshots if index == 0 else []
            if first_snapshots or self.reads_tmem(work) or exchanges or preps:
                self.wait_latest()
            if first_snapshots or work or exchanges or preps:
                with self.block("if md_epi:"):
                    self.slot_regs = {}
                    loaded: set[int] = set()
                    for store in first_snapshots:
                        self.emit_store_step(store, loaded, parity)
                    for step in work:
                        if isinstance(step, Derived):
                            self.emit_derived(step, loaded)
                            self.defined.add(step.node)
                        elif isinstance(step, Store):
                            self.emit_store_step(step, loaded, parity)
                        elif isinstance(step, Reduce):
                            self.emit_reduce(step, loaded)
                        elif isinstance(step, VecOp):
                            self.emit_vec_op(step)
                        elif isinstance(step, VecStore):
                            self.emit_vec_store(step)
                        else:
                            raise AssertionError(step)
                    for tile in exchanges:
                        source = cast("int", tile.source_slot)
                        if source not in loaded:
                            self.emit_ld(source)
                            loaded.add(source)
                        self.emit_exchange(tile)
                        self.written.add(tile.key)
                        if loop is None:
                            self.root_written.add(tile.key)
                    for dot in preps:
                        self.emit_rescale(
                            dot, loaded, prefix=self.df.new_var("md_rs") + "_"
                        )
                    self.slot_regs = {}
                if exchanges:
                    self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)"
                )
            if not dots:
                continue
            # One commit in flight: a batch without epilogue work ahead of
            # it still waits for the previous one (the barrier's parity
            # tells one phase apart).
            self.wait_latest()
            self.emit(self.sync)
            with self.nest("if md_warp == 0:"):
                self.emit(
                    f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.AFTER_THREAD_SYNC)"
                )
                ring = {
                    dot.loop
                    for dot in dots
                    for operand in (dot.a, dot.b)
                    if tc.tile(dot, operand).ring
                }
                for ring_loop in sorted(ring):
                    if ring_loop not in self.ring_waited and tc.ring_tx.get(ring_loop):
                        self.emit(
                            f"{_RT}.mbar_wait(md_full{ring_loop} + md_st{ring_loop}, md_rd{ring_loop})"
                        )
                        self.ring_waited.add(ring_loop)
                for dot in dots:
                    self.emit_dot_mmas(
                        dot, self.stage_expr(dot.loop), accumulate=dot.acc is not None
                    )
                    self.defined.add(dot.node)
                self.emit_commit()
            self.emit("md_cn = md_cn + 1")
            self.waited = False
            self.flush_drains()
        self.current_blocks = []

    def emit_store_step(self, store: Store, loaded: set[int], parity: str) -> None:
        slot = self.program.frag_slots[store.frag]
        if slot not in loaded:
            self.emit_ld(slot)
            loaded.add(slot)
        with self.row_guard(slot):
            if store in self.tc.staged_stores:
                self.emit_stage_in(store, parity, evaluate=True)
                self.drains.append((store, parity, dict(self.values)))
            else:
                self.emit_direct_store(store)

    def flush_drains(self) -> None:
        """Drain the staged stores (behind a CTA barrier the caller emitted)."""
        for store, parity, values in self.drains:
            self.emit_stage_out(
                store,
                parity,
                cast("str", values),
                "md_tidx",
                self.plan.threads,
            )
        self.drains = []

    def emit_segment(self, steps: Sequence[Step], loop: Loop | None) -> None:
        if not steps:
            return
        region = loop.index if loop is not None else -1
        self.settle_tiles([step for step in steps if isinstance(step, Dot)], region)
        self.emit_blocks(self.region_blocks(steps), loop)

    # -- vectors --------------------------------------------------------------

    def vec_family_slot(self, slot: int) -> int | None:
        family = self.program.vec_frag_slots.get(slot, ())
        return family[0] if family else None

    def vec_row(self, slot: int) -> str:
        family = self.vec_family_slot(slot)
        assert family is not None
        return self.row_var(family)

    def vector_coords(self, node: Node, slot: int) -> tuple[str, ...]:
        shape = _shape(node)
        coords = ["0"] * len(shape)
        if self.vec_family_slot(slot) is None:
            return tuple(coords)
        positions = [
            i
            for i, size in enumerate(shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(node, i)
        ]
        if len(positions) != 1:
            raise _decline(f"{node.name} is not a vector value")
        coords[positions[0]] = self.vec_row(slot)
        return tuple(coords)

    def vec_owner(self, slot: int) -> str | None:  # type: ignore[override]
        """The lane predicate holding each vector element once (None: every
        epilogue thread owns a distinct row)."""
        family = self.vec_family_slot(slot)
        if family is None:
            return "md_tidx == 0"
        if self.half_rows(family):
            return "md_lane < 16"
        return None

    def emit_vec_init(self, loop: Loop) -> None:  # type: ignore[override]
        for carry in loop.vec_carries:
            if carry.identity:
                continue
            if carry.init_vec is not None:
                source = self.program.vec_leaves[carry.init_vec]
                if source != carry.slot:
                    self.emit(f"{self.vreg(carry.slot)} = {self.vreg(source)}")
                continue
            self.emit(f"{self.vreg(carry.slot)} = cutlass.Float32(0.0)")

    def emit_reduce(self, step: Reduce, loaded: set[int]) -> None:  # type: ignore[override]
        """Sums of a pointwise value over a fragment along its columns (the
        thread's row: its local columns, then the other half) or its rows
        (the warp's rows by shuffles, the epilogue warps through the
        scratch, relaid into the row layout), or of a vector into a scalar."""
        program = self.program
        if step.vec is not None:
            self.emit_scalar_reduce(step)
            return
        assert step.frag is not None
        frag_slot = program.frag_slots[step.frag]
        if frag_slot not in loaded:
            self.emit_ld(frag_slot)
            loaded.add(frag_slot)
        self.hoist_scalars(step.value)
        values = []
        for c in range(self.local_cols(frag_slot)):
            self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
            coords = self.element_coords(step.value, frag_slot, c)
            values.append(self.evaluator.evaluate(step.value, coords))
        self.evaluator.reg = None
        if step.orient == "rows":
            total = values[0]
            for value in values[1:]:
                total = self.bind("md_s", f"{total} + {value}")
            if self.half_cols(frag_slot):
                total = self.bind(
                    "md_s", f"{total} + cute.arch.shuffle_sync_bfly({total}, 16)"
                )
            self.emit(f"{self.vreg(step.slot)} = {total}")
            return
        # Column sums: the lanes holding the warp's rows meet by shuffles
        # (16 rows per warp of a 64-row fragment, 32 of a 128-row one), the
        # epilogue warps' partials meet in the scratch, read back at the
        # thread's row index.
        geometry = self.slot(frag_slot)
        width = 16 if self.half_rows(frag_slot) else 32
        base = self.scratch_base()
        extent = geometry.cols
        if extent > geometry.rows:
            raise _decline(
                f"{step.node.name} sums {extent} columns into {geometry.rows} rows"
            )
        partials = [
            self.bind("md_s", f"{_RUNTIME}.lane_group_sum({value}, {width})")
            for value in values
        ]
        owner = (
            "md_lane % 16 == 0"
            if self.half_rows(frag_slot) and self.half_cols(frag_slot)
            else "md_lane == 0"
        )
        with self.nest(f"if {owner}:"):
            for c, partial in enumerate(partials):
                column = self.col_expr(frag_slot, c)
                self.emit(
                    f"{_RUNTIME}.sts_f32({base} + ((md_warp % 4) * {extent} + {column}) * 4, {partial})"
                )
        self.emit(self.epilogue_sync)
        row = self.vec_row(step.slot)
        total = None
        with (
            self.block(f"if {row} < {extent}:")
            if geometry.rows > extent
            else self.block(None)
        ):
            for warp in range(TCGEN05_EPILOGUE_WARPS):
                value = self.bind(
                    "md_s",
                    f"{_RUNTIME}.lds_f32({base} + ({warp * extent} + {row}) * 4)",
                )
                total = (
                    value if total is None else self.bind("md_s", f"{total} + {value}")
                )
            assert total is not None
            self.emit(f"{self.vreg(step.slot)} = {total}")
        self.emit(self.epilogue_sync)

    def emit_scalar_reduce(self, step: Reduce) -> None:  # type: ignore[override]
        """A vector summed into a scalar: the distinct rows of each warp by
        shuffles (the duplicate half of a 64-row family zeroed), the
        epilogue warps through the scratch."""
        program = self.program
        assert step.vec is not None
        source = program.vec_leaves[step.vec]
        family = self.vec_family_slot(source)
        assert family is not None
        width = 16 if self.half_rows(family) else 32
        total = self.bind(
            "md_s", f"{_RUNTIME}.lane_group_sum({self.vreg(source)}, {width})"
        )
        base = self.scratch_base()
        with self.nest("if md_lane == 0:"):
            self.emit(f"{_RUNTIME}.sts_f32({base} + (md_warp % 4) * 4, {total})")
        self.emit(self.epilogue_sync)
        result = None
        for warp in range(TCGEN05_EPILOGUE_WARPS):
            value = self.bind("md_s", f"{_RUNTIME}.lds_f32({base} + {warp * 4})")
            result = (
                value if result is None else self.bind("md_s", f"{result} + {value}")
            )
        self.emit(self.epilogue_sync)
        self.emit(f"{self.vreg(step.slot)} = {result}")

    def emit_vec_op(self, step: VecOp) -> None:  # type: ignore[override]
        self.hoist_scalars(step.node)
        self.evaluator.vec_reg = (0, 0)
        coords = self.vector_coords(step.node, step.slot)
        value = self.evaluator.define(step.node, coords)
        self.evaluator.vec_reg = None
        self.emit(f"{self.vreg(step.slot)} = {value}")

    def emit_vec_store(self, step: VecStore) -> None:  # type: ignore[override]
        slot = self.program.vec_leaves[step.vec]
        access = step.access
        positions = access.tile_positions
        self.hoist_scalars(step.value)
        owner = self.vec_owner(slot)
        with self.block(f"if {owner}:" if owner else None):
            self.evaluator.vec_reg = (0, 0)
            coords = self.vector_coords(step.value, slot)
            value = self.evaluator.evaluate(step.value, coords)
            self.evaluator.vec_reg = None
            element = ["0"] * len(access.dims)
            if positions:
                element[positions[0]] = self.vec_row(slot)
            self.emit(
                f"{_RUNTIME}.stg_f32({self.address(access, tuple(element))}, {value})"
            )

    # -- loops ----------------------------------------------------------------

    def emit_loop_carry_init(self, loop: Loop) -> None:
        """Seed the loop's TMEM carries: zeros, this thread's row of the init
        tile, an earlier loop's accumulator (in place, or copied), or a
        pointwise value over one (``A = where(strict_lower, -beta * kk, 0)``,
        evaluated per element of the loaded row)."""
        program = self.program
        carries = [
            carry
            for carry in loop.carries
            if not (
                carry.init_frag is not None
                and carry.init_value is None
                and program.frag_slots[program.resolve(carry.init_frag)] == carry.slot
            )
        ]
        if not carries:
            return
        self.wait_latest()
        with self.block("if md_epi:"):
            self.slot_regs = {}
            loaded: set[int] = set()
            for carry in carries:
                slot = carry.slot
                init = carry.init
                if carry.init_frag is not None:
                    source = program.frag_slots[program.resolve(carry.init_frag)]
                    if source not in loaded:
                        self.emit_ld(source)
                        loaded.add(source)
                    if carry.init_value is None:
                        for chunk in range(self.ld_chunks(slot)):
                            self.emit_st_array(
                                f"{self.vec}{source}_{chunk}", slot, chunk
                            )
                        continue
                    self.hoist_scalars(carry.init_value)
                    for chunk in range(self.ld_chunks(slot)):
                        cols = self.chunk_cols(slot, chunk)
                        name = self.bind(
                            "md_seed",
                            f"cutlass.Array(cutlass.Float32, {cols}, alignment=16)",
                        )
                        for j in range(cols):
                            c = chunk * TCGEN05_LD_COLS + j
                            self.evaluator.reg = (chunk, j)  # type: ignore[assignment]
                            coords = self.element_coords(carry.init_value, source, c)
                            value = self.evaluator.evaluate(carry.init_value, coords)
                            self.emit(f"{name}[{j}] = cutlass.Float32({value})")
                        self.evaluator.reg = None
                        self.emit_st_array(name, slot, chunk)
                    continue
                positions: list[int] = []
                paired = False
                if init is not None:
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
            self.slot_regs = {}
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")

    def emit_carry_copies(self, loop: Loop) -> None:  # type: ignore[override]
        """Carry outputs computed in another slot (``Apow = dot(Apow, Apow)``
        lands in a fresh slot) move into the carry's slot for the next
        iteration."""
        program = self.program
        copies = []
        for carry in loop.carries:
            out = program.resolve(carry.output)
            if out is carry.placeholder:
                continue
            source = program.frag_slots[out]
            if source != carry.slot:
                copies.append((source, carry.slot))
        if not copies:
            return
        self.wait_latest()
        with self.block("if md_epi:"):
            self.slot_regs = {}
            loaded: set[int] = set()
            for source, target in copies:
                if source not in loaded:
                    self.emit_ld(source)
                    loaded.add(source)
                for chunk in range(self.ld_chunks(target)):
                    self.emit_st_array(f"{self.vec}{source}_{chunk}", target, chunk)
            self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")

    def loop_consumers(self, loop: Loop) -> list[Dot]:
        return [
            dot
            for dot in self.program.dots
            if dot.loop >= 0 and loop.index in self.enclosing(dot.loop)
        ]

    def emit_region_loop(self, loop: Loop, region: int) -> None:
        tc = self.tc
        index = loop.index
        self.emit_loop_carry_init(loop)
        self.emit_vec_init(loop)
        for carry in loop.carries:
            self.defined.add(carry.placeholder)
        begin, end, trip = self.loop_bounds(loop)
        var = self.loop_index_var(index)
        ring = index in self.ring_loops
        if ring and tc.ring_tx.get(index):
            # The ring prologue: the first ``stages - 1`` iterations' tiles
            # (the first iteration's with a single slot), in flight while
            # the enclosing regions' tiles settle.
            self.wait_latest()
            with self.nest("if md_warp == 0:"), self.nest(f"if {_PRIMS}.elect_sync():"):
                for ahead in range(max(tc.stages - 1, 1)):
                    value = f"({begin} + {ahead})" if ahead else begin
                    values = {**self.values, index: value}
                    count = (
                        f"{self.counter(index)} + {ahead}"
                        if ahead
                        else self.counter(index)
                    )
                    if trip is not None and ahead < trip:
                        self.emit_ring_issue(index, values, count)
                    elif trip is None:
                        with self.nest(f"if {value} < {end}:"):
                            self.emit_ring_issue(index, values, count)
        self.settle_tiles(self.loop_consumers(loop), region)
        with self.nest(f"for {var} in cutlass.range({begin}, {end}):"):
            self.values[index] = var
            if loop.innermost:
                self.emit_ring_iteration(loop, begin, end, trip)
            else:
                self.emit_outer_iteration(loop)
            if index in self.counted:
                self.emit(f"{self.counter(index)} = {self.counter(index)} + 1")
            del self.values[index]
        self.forget_iteration_bases(index)
        self.evaluator = _PassEvaluator(self)
        self.waited = False
        for carry in loop.carries:
            self.defined.add(carry.phi)
        # Exchange tiles written inside the loop of fragments it defines are
        # stale; the barrier keeps a sibling loop's prologue behind this
        # loop's reads of the overlapping frame.
        self.written = {key for key in self.written if key in self.root_written}
        self.emit(self.sync)

    def enter_iteration(self, loop: Loop) -> None:  # type: ignore[override]
        var = self.loop_index_var(loop.index)
        for axis in loop.axes:
            offset = self.axis_offset(axis, var)
            self.emit(f"{self.axis_base(axis)} = {offset} * {self.plan.size(axis)}")
        self.forget_iteration_bases(loop.index)
        if loop.index in self.counted:
            count = self.counter(loop.index)
            if loop.index in self.ring_loops:
                self.emit(f"md_st{loop.index} = {count} % {self.tc.stages}")
                self.emit(f"md_rd{loop.index} = ({count} // {self.tc.stages}) & 1")
            self.emit(f"md_pt{loop.index} = {count} % 2")
        self.define_iteration_bases(loop.index)
        self.evaluator = _PassEvaluator(self)
        self.written = {key for key in self.written if key in self.root_written}
        for tile in self.tc.staged_tiles.get(loop.index, ()):
            self.settled.discard(tile.key)

    def emit_ring_iteration(
        self, loop: Loop, begin: str, end: str, trip: int | None
    ) -> None:
        tc, program = self.tc, self.program
        index = loop.index
        var = self.loop_index_var(index)
        self.enter_iteration(loop)
        self.ring_waited = set()
        self.derived_words = {}
        # The back edge may carry the previous iteration's commit.
        self.waited = False
        derived = [
            step for step in program.derived if step.loop == index and step.loads
        ]
        for step in derived:
            self.emit_derived_prefetch(step)
        if derived:
            # The next iteration's rows into L2 (the register prefetch above
            # would otherwise pay the DRAM latency on the chain).
            offset = var if begin == "0" else f"({var} - {begin})"
            with self.nest(f"if {offset} + 1 < {end} - {begin}:"):
                self.emit_region_line_prefetch(
                    derived, {**self.values, index: f"({var} + 1)"}
                )
        ring = index in self.ring_loops
        tiles = tc.ring_tiles.get(index, ())
        if ring:
            # The previous iteration's batches read the slot refilled below
            # (and the ring tiles every pass rewrites).
            self.wait_latest()
            if self.has_passes(tiles):
                wait = (
                    f"{_RT}.mbar_wait(md_full{index} + md_st{index}, md_rd{index})"
                    if tc.ring_tx.get(index)
                    else None
                )
                self.emit_passes(
                    tiles, f"md_st{index}", wait, "md_tidx", self.plan.threads
                )
                if any(tile.route in (ROUTE_INPLACE, ROUTE_CONVERT) for tile in tiles):
                    self.ring_waited.add(index)
            if tc.stages > 1 and tc.ring_tx.get(index):
                ahead = tc.stages - 1
                offset = var if begin == "0" else f"({var} - {begin})"
                with (
                    self.nest("if md_warp == 0:"),
                    self.nest(f"if {_PRIMS}.elect_sync():"),
                    self.nest(f"if {offset} + {ahead} < {end} - {begin}:"),
                ):
                    self.emit_ring_issue(
                        index,
                        {**self.values, index: f"({var} + {ahead})"},
                        f"{self.counter(index)} + {ahead}",
                    )
        self.emit_segment(loop.steps, loop)
        if self.drains:
            self.emit(self.sync)
            self.flush_drains()
        self.emit_carry_copies(loop)
        if ring and tc.stages == 1 and tc.ring_tx.get(index):
            # Refill after every batch read the slot.
            self.wait_latest()
            with (
                self.nest("if md_warp == 0:"),
                self.nest(f"if {_PRIMS}.elect_sync():"),
                self.nest(f"if {var} + 1 < {end}:"),
            ):
                self.emit_ring_issue(
                    index,
                    {**self.values, index: f"({var} + 1)"},
                    f"{self.counter(index)} + 1",
                )

    def emit_region_line_prefetch(
        self, derived: Sequence[Derived], values: Mapping[int, str]
    ) -> None:
        """``prefetch.global.L2`` of the rows the derived values' tile inputs
        read in the iteration at ``values`` (thread ``t`` takes row ``t``)."""
        for step in derived:
            geometry = self.slot(step.slot)
            for access in step.loads:
                base = self.iteration_base(access, dict(values))
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

    def emit_outer_iteration(self, loop: Loop) -> None:  # type: ignore[override]
        index = loop.index
        self.enter_iteration(loop)
        self.waited = False
        if self.tc.staged_tiles.get(index):
            # The frame overlaps a sibling loop's, whose last batch may still
            # be reading it on entry.
            self.wait_latest()
        self.emit_staged_issue(index)
        self.emit_region(loop.steps, index)
        if self.drains:
            self.emit(self.sync)
            self.flush_drains()
        self.emit_carry_copies(loop)
        # The next iteration's tiles and exchange tiles are rewritten behind
        # every thread's reads (and the tensor core's) of this one.
        self.wait_latest()
        self.emit(self.sync)

    def emit_region(self, steps: Sequence[Step], region: int) -> None:  # type: ignore[override]
        loop = self.program.loops[region] if region >= 0 else None
        segment: list[Step] = []
        for step in steps:
            if isinstance(step, Loop):
                self.emit_segment(segment, loop)
                segment = []
                if self.drains:
                    self.emit(self.sync)
                    self.flush_drains()
                self.emit_region_loop(step, region)
                continue
            segment.append(step)
        self.emit_segment(segment, loop)

    # -- the kernel -------------------------------------------------------------

    def render(self) -> str:  # type: ignore[override]
        tc, plan, program = self.tc, self.plan, self.program
        self.emit("md_tidx, _, _ = cute.arch.thread_idx()")
        self.emit("md_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
        self.emit("md_lane = md_tidx % 32")
        self.emit(f"md_epi = md_warp < {TCGEN05_EPILOGUE_WARPS}")
        heights = {slot.rows for slot in tc.slots}
        if 64 in heights:
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
        self.define_axis_bases()
        self.emit(
            f"md_smem = cute.arch.alloc_smem(cutlass.Uint8, {max(self.smem_bytes, 16)}, alignment={SLAB_ALIGNMENT})"
        )
        self.emit("md_s0 = cutlass.Int32(md_smem.toint())")
        self.emit("md_cb = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)")
        barriers = ["md_cb"]
        for loop in self.ring_loops:
            self.emit(
                f"md_full{loop} = cute.arch.alloc_smem(cutlass.Int64, {tc.stages}, alignment=8)"
            )
        staged_regions = sorted(region for region, tx in tc.staged_tx.items() if tx)
        for region in staged_regions:
            self.emit(
                f"{self.sf_name(region)} = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)"
            )
            barriers.append(self.sf_name(region))
        self.emit("md_tslot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)")
        with self.nest("if md_tidx == 0:"):
            for name in barriers:
                self.emit(f"{_PRIMS}.mbarrier_init({name}, 1)")
            for loop in self.ring_loops:
                for stage in range(tc.stages):
                    self.emit(f"{_PRIMS}.mbarrier_init(md_full{loop} + {stage}, 1)")
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
        for dot in program.dots:
            index = self.dot_index[dot.node]
            dtype = _OPERAND_DTYPE_NAMES[dot.a.dtype]
            self.emit(
                f"md_idesc{index} = {_PRIMS}.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, "
                f"a_dtype={dtype}, b_dtype={dtype}, n_dim={plan.size(dot.cols)}, "
                f"m_dim={plan.size(dot.rows)}, a_major={int(not self.k_major(dot.a))}, "
                f"b_major={int(not self.k_major(dot.b))})"
            )
        self.emit("md_cn = cutlass.Int32(0)")
        for loop in sorted(self.counted):
            self.emit(f"{self.counter(loop)} = cutlass.Int32(0)")
        for slot in range(len(program.vec_slots)):
            self.emit(f"{self.vreg(slot)} = cutlass.Float32(0.0)")
        self.define_iteration_bases(-1)
        if tc.tma:
            with self.nest("if md_warp == 0:"), self.nest(f"if {_PRIMS}.elect_sync():"):
                for spec in tc.tma:
                    self.emit(f"{_PRIMS}.prefetch_tensormap({spec.param}.get_ptr())")
                if tc.staged_tx.get(-1):
                    self.emit_tma_group(
                        tc.staged_tiles[-1], {}, self.sf_name(-1), tc.staged_tx[-1], "0"
                    )
        self.emit(self.sync)
        self.emit_region(program.body, -1)
        if self.drains:
            self.emit(self.sync)
            self.flush_drains()
        self.wait_latest()
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


__all__ = [
    "EPILOGUE_BARRIER_ID",
    "plan_tcgen05_regions",
    "region_config_reason",
    "region_facts",
    "region_items",
    "region_layout",
    "region_program_reason",
    "region_smem_bytes",
    "region_tmem_columns",
]
