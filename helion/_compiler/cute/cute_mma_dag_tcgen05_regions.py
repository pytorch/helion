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
from .cute_mma_dag import MMA_DAG_LOADS_TMA
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import ROUTE_CONVERT
from .cute_mma_dag import ROUTE_DIRECT
from .cute_mma_dag import ROUTE_EXCHANGE
from .cute_mma_dag import ROUTE_INPLACE
from .cute_mma_dag import ROUTE_SYNC
from .cute_mma_dag import Carry
from .cute_mma_dag import Derived
from .cute_mma_dag import Dot
from .cute_mma_dag import Loop
from .cute_mma_dag import Reduce
from .cute_mma_dag import Step
from .cute_mma_dag import Store
from .cute_mma_dag import VecOp
from .cute_mma_dag import VecStore
from .cute_mma_dag import _decline
from .cute_mma_dag import async_tile_reason
from .cute_mma_dag import staged_store_reason
from .cute_mma_dag_codegen import _RUNTIME
from .cute_mma_dag_codegen import _shape
from .cute_mma_dag_tcgen05 import _PRIMS
from .cute_mma_dag_tcgen05 import _RT
from .cute_mma_dag_tcgen05 import _SWIZZLE_BY_ROW_BYTES
from .cute_mma_dag_tcgen05 import ATOM_32B_SWIZZLE
from .cute_mma_dag_tcgen05 import ATOM_MAX_BYTES
from .cute_mma_dag_tcgen05 import REGION_EPILOGUE
from .cute_mma_dag_tcgen05 import REGION_EXCHANGE
from .cute_mma_dag_tcgen05 import REGION_LOOP
from .cute_mma_dag_tcgen05 import SLAB_ALIGNMENT
from .cute_mma_dag_tcgen05 import TCGEN05_EPILOGUE_WARPS
from .cute_mma_dag_tcgen05 import TCGEN05_LD_COLS
from .cute_mma_dag_tcgen05 import TCGEN05_MAX_ROWS
from .cute_mma_dag_tcgen05 import TCGEN05_MAX_TMEM_COLS
from .cute_mma_dag_tcgen05 import TCGEN05_SLOT_ALIGN
from .cute_mma_dag_tcgen05 import TMA_MAX_MAPS
from .cute_mma_dag_tcgen05 import TMA_MAX_RANK
from .cute_mma_dag_tcgen05 import TMA_STORE_THREAD
from .cute_mma_dag_tcgen05 import AuxTile
from .cute_mma_dag_tcgen05 import RegisterCarryFacts
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
from .cute_mma_dag_tcgen05 import resident_slots
from .cute_mma_dag_tcgen05 import tcgen05_atom_bytes
from .cute_mma_dag_tcgen05 import tcgen05_tmem_cols
from .cute_warp_mma_gemm import _swizzled_chunk

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
# A derived value's 2-D tile input riding its loop's ring (``AuxTile``).
ITEM_AUX = "aux"
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
            # MN-major tf32 tiles are read under the 32-byte-unit swizzle
            # (verified for B operands); an A operand with K along its rows is
            # not.
            if operand.dtype == torch.float32 and operand.trans and operand.role == "a":
                return f"{operand.value.name}: an MN-major tf32 A operand"
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
    core reads it: the natural orientation.  An fp32 B operand the matcher
    stores transposed for the warp mode (``[N, K]``: ``ldmatrix`` cannot
    read fp32 MN-major) is natural here, read MN-major under the 32-byte
    unit swizzle (``ATOM_32B_SWIZZLE``)."""
    if operand.transposed_tile:
        return operand.cols, operand.rows, False
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


def register_carries(program: MmaDagProgram) -> tuple[int, ...]:
    """The slots of the loop carries that may live in the epilogue warps'
    registers instead of tensor memory: a carry of one innermost loop whose
    body output is a derived value in the carry's own slot, and whose slot
    no dot writes (result or accumulator) or reads (operand, accumulator)
    and no carry copy reads: every reader is epilogue work (derived values,
    stores, reductions, the seed of a later carry), so the value never
    needs the tensor-memory round trip -- gated dqkw's dW = dW - dot(...)
    cost a tcgen05.ld, a tcgen05.st and two waits in every tile_dv
    iteration.  Whether a carry is resident at a config is the register
    budget's call (``register_carry_facts``, ``resident_slots``): the
    carry's row is live throughout its loop."""
    carried: dict[int, list[Loop]] = {}
    for loop in program.loops:
        for carry in loop.carries:
            carried.setdefault(carry.slot, []).append(loop)
    derived_by_node = {step.node: step for step in program.derived}
    # The slots a dot or a carry copy touches stay in tensor memory.
    tmem: set[int] = {source for _loop, _carry, source in copied_carries(program)}
    for dot in program.dots:
        tmem.add(dot.slot)
        if dot.acc_slot is not None:
            tmem.add(dot.acc_slot)
        if dot.acc_frag is not None:
            tmem.add(program.frag_slots[dot.acc_frag])
        for operand in (dot.a, dot.b):
            if operand.frag is not None:
                tmem.add(program.frag_slots[operand.frag])
    result: list[int] = []
    for slot, loops in carried.items():
        if len(loops) != 1 or not loops[0].innermost or slot in tmem:
            continue
        (carry,) = [carry for carry in loops[0].carries if carry.slot == slot]
        output = derived_by_node.get(program.resolve(carry.output))
        if output is None or output.slot != slot or output.loop != loops[0].index:
            continue
        result.append(slot)
    return tuple(sorted(result))


def register_carry_facts(program: MmaDagProgram) -> tuple[RegisterCarryFacts, ...]:
    """``register_carries`` with the register-budget live sets of the
    derived values in each carry's loop (and the loops nested in it) that
    neither read nor write the carry: their inputs' rows, their own and
    the carry's (``Tcgen05Facts.derived_cols`` counts the others)."""
    loops = {
        carry.slot: loop.index
        for loop in program.loops
        for carry in loop.carries
        if carry.slot in register_carries(program)
    }
    result = []
    for slot, loop in sorted(loops.items()):
        live: list[tuple[int, ...]] = []
        for step in program.derived:
            if loop not in program.enclosing_loops(step.loop) or step.slot == slot:
                continue
            slots = [program.frag_slots[frag] for frag in step.frags]
            if slot in slots:
                continue
            live.append(
                (
                    *(program.slots[s][1].index for s in slots),
                    step.cols.index,
                    program.slots[slot][1].index,
                )
            )
        result.append(RegisterCarryFacts(slot=slot, live=tuple(live)))
    return tuple(result)


def tmem_slots(program: MmaDagProgram) -> tuple[int, ...]:
    """The fragment slots that live in tensor memory: dot results and
    accumulators, carries, and the derived values that leave their block
    (``derived_crosses``); the other derived values stay in the epilogue
    registers of the block computing them (a tile-only one is recomputed in
    every block that reads it).  Gated wy_dL's ``dLL = dA * beta * kk * L``
    is read by its two reductions only: in registers it leaves the chain's
    peak live set at dA + kk, 128 columns instead of 192 (256 allocated),
    two CTAs per SM instead of one."""
    slots: set[int] = set()
    for dot in program.dots:
        slots.add(dot.slot)
        if dot.acc_slot is not None:
            slots.add(dot.acc_slot)
    for loop in program.loops:
        slots.update(carry.slot for carry in loop.carries)
    for derived in program.derived:
        if derived_crosses(program, derived):
            slots.add(derived.slot)
    # (A register carry -- ``register_carries`` -- leaves the set at the
    # configs whose budget admits it: ``resident_slots`` at plan time.)
    return tuple(sorted(slots))


def dot_prepares(dot: Dot) -> bool:
    """The dot's accumulator slot is written by the epilogue warps before
    the issue: a rescale (the dot accumulates onto a pointwise DAG over a
    fragment) or a fragment accumulated into another slot."""
    from .matmul_utils import strip_new_vars

    return dot.acc_frag is not None and (
        (dot.acc is not None and strip_new_vars(dot.acc) is not dot.acc_frag)
        or dot.slot != dot.acc_slot
    )


Block = tuple[list[Step], list[Dot]]


def region_block_steps(program: MmaDagProgram, steps: Sequence[Step]) -> list[Block]:
    """``steps`` (one segment: no loops) as blocks of epilogue-warp work
    followed by a batch of dots issued together (see the simple emitter's
    ``loop_blocks``; vector steps read no tensor memory).  Work that reads
    the pending batch's results waits for the next block -- and so does,
    in program order, every step behind it -- while later independent
    dots still join the batch (one commit round trip instead of two).
    The placement depends on the program alone: the emitter renders it,
    and ``derived_crosses`` reads it to decide which derived values leave
    their block."""
    blocks: list[Block] = []
    work: list[Step] = []
    dots: list[Dot] = []
    produced: set[int] = set()
    batch_reads: set[int] = set()
    deferred: list[Step] = []
    deferred_reads: set[int] = set()
    deferred_writes: set[int] = set()

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
                dot_prepares(step) or step.acc_frag not in produced_nodes(dots)
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
        elif isinstance(step, Store):
            reads = {program.frag_slots[step.frag]}
            writes = -1
        elif isinstance(step, Reduce):
            reads = {program.frag_slots[step.frag]} if step.frag is not None else set()
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
            deferred.append(step)
            deferred_reads |= reads
            if writes >= 0:
                deferred_writes.add(writes)
        else:
            work.append(step)
    flush()
    if work:
        blocks.append((work, []))
    return blocks


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
    # segment (the blocks are built per segment): crossing as well.  Within
    # the segment the blocks are the emitter's (``region_block_steps``): a
    # value over fragments whose reader sits in another block crosses --
    # its registers do not reach that block, and a recomputation there
    # would read input slots that the blocks between may have rewritten
    # (a derived value whose store a later dot batch deferred was recomputed
    # after another value had taken its input's slot); a tile-only value is
    # recomputed where read (its inputs are loads).
    steps = program.body if derived.loop < 0 else program.loops[derived.loop].steps
    start = next(i for i, step in enumerate(steps) if step is derived)
    first = start
    while first > 0 and not isinstance(steps[first - 1], Loop):
        first -= 1
    last = start
    while last + 1 < len(steps) and not isinstance(steps[last + 1], Loop):
        last += 1
    block_of: dict[int, int] = {}
    for index, (work, dots) in enumerate(
        region_block_steps(program, steps[first : last + 1])
    ):
        for step in [*work, *dots]:
            block_of[id(step)] = index
    for reader in readers:
        end = next(i for i, step in enumerate(steps) if step is reader)
        if any(isinstance(step, Loop) for step in steps[start:end]):
            return True
        if derived.frags and block_of[id(reader)] != block_of[id(derived)]:
            return True
    return False


def aux_tile_loads(program: MmaDagProgram) -> list[tuple[Derived, Access]]:
    """The derived values' 2-D tile inputs that ride their loop's ring (the
    warp planner's ``ring_derived_load`` rule: the derived value sits in an
    innermost loop the tile varies with) and whose columns are contiguous,
    once per (tile, loop); the planner fetches them by TMA when the source
    proves aligned, else the lanes load their pairs from global memory.
    A tile a ring operand of the same loop already brings in (the same load,
    in the operand's dtype, in the source orientation: wy_dL's ``v`` and
    ``k``, rewritten in place into ``beta * v`` / ``beta * k`` after the
    reads) is read from that operand's tile instead (``aux_shared_operand``)
    and takes no item."""
    result: list[tuple[Derived, Access]] = []
    seen: set[tuple[Node, int]] = set()
    for derived in program.derived:
        for access in derived.loads:
            if (
                len(access.tile_dims) == 2
                and access.tile_dim_strides[1] == 1
                and program.ring_derived_load(derived, access)
                and (access.node, derived.loop) not in seen
                and aux_shared_operand(program, derived, access) is None
            ):
                seen.add((access.node, derived.loop))
                result.append((derived, access))
    return result


def aux_shared_operand(
    program: MmaDagProgram, derived: Derived, access: Access
) -> tuple[Dot, Operand] | None:
    """The operand whose tile holds ``access``'s load as loaded (same load
    node, the load's dtype, rows and columns the access's axes) and is valid
    while ``derived`` runs: a ring operand of the derived's own loop, or an
    unrewritten (no pointwise chain) staged tile of that loop, of a loop
    enclosing it, or hoisted at the root (gated dqkw's row sums read q and
    k, the staged B operands of the tile_d loop's dq / dk_as dots)."""
    positions = access.tile_positions
    rows, cols = access.dims[positions[0]], access.dims[positions[1]]
    frames = {-1, *program.enclosing_loops(derived.loop)}
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            load = operand.load
            if (
                load is None
                or load.node is not access.node
                or load.dtype != operand.dtype
                or operand.transposed_tile
                or operand.rows is not rows
                or operand.cols is not cols
            ):
                continue
            if dot.loop == derived.loop and program.is_ring(dot, operand):
                return dot, operand
            if (
                not operand.pointwise
                and not program.is_ring(dot, operand)
                and program.operand_frame(dot, operand) in frames
            ):
                return dot, operand
    return None


def hoisted_exchange_loop(
    program: MmaDagProgram, dot: Dot, operand: Operand
) -> int | None:
    """The top-level loop ahead of which the exchange tile of ``operand``
    (a fragment operand of ``dot``, a loop's dot) is written once, or None
    when the tile is written where the dot runs.  A fragment defined at the
    root and read by a loop's dot has one value for every iteration, and
    so has the operand: the matcher admits a pointwise chain over it only
    when the chain's aux loads do not vary below the fragment's frame
    (dqkw's ``where(causal, dattn, 0)``, gated ``* L``, feeds dq and dk_as
    in every tile_d iteration; wy's ``T`` feeds the dots of two loops)."""
    if operand.frag is None or dot.loop < 0:
        return None
    if program.operand_frame(dot, operand) != -1:
        return None
    return program.enclosing_loops(dot.loop)[-1]


def copied_carries(program: MmaDagProgram) -> list[tuple[Loop, Carry, int]]:
    """``(loop, carry, output slot)`` of the carries whose body output is
    computed in another slot (``Apow = dot(Apow, Apow)`` lands in a fresh
    slot): the region emitter copies it into the carry's slot at the
    iteration's end, unless the two share their columns."""
    result = []
    for loop in program.loops:
        for carry in loop.carries:
            out = program.resolve(carry.output)
            if out is carry.placeholder:
                continue
            source = program.frag_slots[out]
            if source != carry.slot:
                result.append((loop, carry, source))
    return result


def region_flat(
    program: MmaDagProgram,
) -> tuple[list[tuple[int, Step | tuple[str, Loop]]], dict[int, tuple[int, int]]]:
    """The region emitter's flat program order: the steps with every loop
    body inlined between an ``("entry", loop)`` position (the carry inits,
    in the enclosing region) and an ``("exit", loop)`` position (the end of
    an iteration: the copies of outputs computed in another slot, in the
    loop), each with its region -- a loop index, or ``-1 - k`` for the
    k-th root segment between the loops -- and each loop's (entry, exit)
    positions."""
    ordered: list[tuple[int, Step | tuple[str, Loop]]] = []
    spans: dict[int, tuple[int, int]] = {}
    segment = 0

    def walk(steps: Sequence[Step], region: int) -> None:
        nonlocal segment
        for step in steps:
            here = region if region >= 0 else -1 - segment
            if isinstance(step, Loop):
                entry = len(ordered)
                ordered.append((here, ("entry", step)))
                walk(step.steps, step.index)
                ordered.append((step.index, ("exit", step)))
                spans[step.index] = (entry, len(ordered) - 1)
                segment += 1
            else:
                ordered.append((here, step))

    walk(program.body, -1)
    return ordered, spans


def region_live_slots(program: MmaDagProgram) -> tuple[tuple[int, ...], ...]:
    """The slot sets live at each position of ``region_flat`` -- the
    region emitter's own liveness, finer than the planner's
    ``mma_dag_live_slots`` (the warp emitter's register budget):

    - a step reads its inputs before it writes its result (an exchange's
      tcgen05.ld precedes the MMA, a derived value's loads its store), so
      a read at a position is live through the position before it, a
      write from its position on;
    - a hoisted exchange (``hoisted_exchange_loop``) and a carry's seed
      (``init_frag``) are read ahead of the loop's entry, where the inits
      write the carries;
    - a carry whose output is computed in another slot
      (``copied_carries``) is live from the entry to its last read in the
      body and again from the exit, where the copy rewrites it: its
      columns are free in between, so the output may take them and the
      copy vanishes (wy's Apow: 128 TMEM columns instead of 192; the
      aliased carry is placed by ``region_tmem_layout``);
    - every other carry is live from the entry to the exit and on to its
      phi's readers.

    A slot may have several live intervals."""
    ordered, spans = region_flat(program)
    intervals: dict[int, list[list[int]]] = {}

    def define(slot: int, position: int) -> None:
        own = intervals.setdefault(slot, [])
        if own and own[-1][1] >= position - 1:
            own[-1][1] = max(own[-1][1], position)
        else:
            own.append([position, position])

    def use(slot: int, position: int) -> None:
        own = intervals.get(slot)
        if own:
            own[-1][1] = max(own[-1][1], position - 1)

    copied = {
        (loop.index, carry.slot): source
        for loop, carry, source in copied_carries(program)
    }
    for index, (_region, step) in enumerate(ordered):
        if isinstance(step, tuple):
            kind, loop = step
            if kind == "entry":
                for carry in loop.carries:
                    if carry.init_frag is not None:
                        use(program.frag_slots[program.resolve(carry.init_frag)], index)
                for carry in loop.carries:
                    define(carry.slot, index)
            else:
                for carry in loop.carries:
                    source = copied.get((loop.index, carry.slot))
                    if source is None:
                        use(carry.slot, index)
                    else:
                        use(source, index)
                    define(carry.slot, index)
            continue
        if isinstance(step, Dot):
            for operand in (step.a, step.b):
                if operand.frag is not None:
                    loop_index = hoisted_exchange_loop(program, step, operand)
                    at = index if loop_index is None else spans[loop_index][0]
                    use(program.frag_slots[program.resolve(operand.frag)], at)
            if step.acc_frag is not None:
                use(program.frag_slots[step.acc_frag], index)
            define(step.slot, index)
        elif isinstance(step, Derived):
            for leaf in step.frags:
                use(program.frag_slots[leaf], index)
            define(step.slot, index)
        elif isinstance(step, (Store, Reduce)) and step.frag is not None:
            use(program.frag_slots[step.frag], index)
    return tuple(
        tuple(
            sorted(
                slot
                for slot, own in intervals.items()
                if any(first <= index <= last for first, last in own)
            )
        )
        for index in range(len(ordered))
    )


def step_depends(program: MmaDagProgram, later: Dot | Derived, earlier: Dot) -> bool:
    """Whether ``later`` reads ``earlier``'s result (through fragments,
    accumulators or derived chains): the block builder then flushes the
    pending batch ahead of it, so the two never share a batch."""
    if isinstance(later, Derived):
        pending = list(later.frags)
    else:
        pending = [
            operand.frag for operand in (later.a, later.b) if operand.frag is not None
        ]
        if later.acc_frag is not None:
            pending.append(later.acc_frag)
    derived_by_node = {step.node: step for step in program.derived}
    seen: set[Node] = set()
    while pending:
        node = program.resolve(pending.pop())
        if node is earlier.node:
            return True
        if node in seen:
            continue
        seen.add(node)
        derived = derived_by_node.get(node)
        if derived is not None:
            pending.extend(derived.frags)
    return False


def exchange_phases(program: MmaDagProgram) -> dict[tuple[object, ...], int]:
    """The batch phase of the exchange tiles written inside a loop body
    (tile key -> phase); tiles of different phases of one loop share their
    bytes.  A tile's exchange is written in the block of its first consumer
    dot behind the previous batch's commit wait, and read by that batch's
    MMAs only, so two tiles may share bytes when every consumer of the
    later one depends on every consumer of the earlier one: the later
    exchange then runs behind the wait for the batch that read the earlier
    tile, and the earlier tile is never read again in the iteration.  The
    tiles are grouped in order of their first consumer: a tile that
    depends on every tile of every group opens a new phase; one that
    depends on every tile of every group but the last joins the last;
    any other keeps its own bytes.  wy's chain: Apow's two tiles (the
    squaring MMA) in phase 0, the squared Apow's and T's tiles (the
    accumulate MMA, which reads the squaring's result) in phase 1 -- 32 KB
    instead of 64.  Tiles read by several loops, by nested loops' dots or
    exchanged once ahead of the loop (``hoisted_exchange_loop``) take no
    phase."""
    consumers: dict[tuple[object, ...], list[Dot]] = {}
    loops: dict[tuple[object, ...], set[int]] = {}
    hoisted: set[tuple[object, ...]] = set()
    order: dict[Node, int] = {}
    for loop in program.loops:
        for position, step in enumerate(loop.steps):
            if isinstance(step, Dot):
                order[step.node] = position
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.kind != "frag":
                continue
            key = tile_key(program, operand)
            consumers.setdefault(key, []).append(dot)
            loops.setdefault(key, set()).add(dot.loop)
            if hoisted_exchange_loop(program, dot, operand) is not None:
                hoisted.add(key)
    phases: dict[tuple[object, ...], int] = {}
    for loop in program.loops:
        keys = [
            key
            for key, where in loops.items()
            if where == {loop.index} and key not in hoisted
        ]
        keys.sort(key=lambda key: min(order[dot.node] for dot in consumers[key]))
        groups: list[list[tuple[object, ...]]] = []

        def after(later: tuple[object, ...], earlier: tuple[object, ...]) -> bool:
            return all(
                step_depends(program, y, x)
                for x in consumers[earlier]
                for y in consumers[later]
            )

        for key in keys:
            before = [all(after(key, other) for other in group) for group in groups]
            if all(before):
                groups.append([key])
            elif groups and all(before[:-1]):
                groups[-1].append(key)
            else:
                continue
            phases[key] = len(groups) - 1
    return phases


def raw_overlays_tile(operand: Operand) -> bool:
    """Whether a converted operand's raw source tile lands on its own
    tile's bytes instead of a tile of its own: a 16-bit source widened into
    an fp32 (tf32) tile is half the tile's size, and the region emitter's
    widening pass reads every raw packet into registers behind a CTA
    barrier before any thread writes (``emit_convert_pass``), so the raw
    tile needs no bytes of its own -- 8 KB per ring slot of wy's w / u
    loops at C = 64."""
    load = operand.load
    return (
        load is not None and load.dtype.itemsize == 2 and operand.dtype == torch.float32
    )


def staging_copies(program: MmaDagProgram, store: Store) -> int:
    """Copies of a staged store's buffer.  A store in an innermost loop is
    double-buffered: the next iteration stages in while the TMA store of
    this one still reads.  A store in a loop with nested loops stages once
    per outer iteration, with a whole inner loop between the drain and the
    next stage-in -- the store thread waits for its groups' reads ahead of
    that stage-in (``emit_blocks``) and one buffer serves (gated dqkw's dW
    / dq / dk_as at [128, 32, 32, 64, 128]: 40 -> 20 KB of staging).  A
    root store drains within its segment: one buffer."""
    if store.loop >= 0 and program.loops[store.loop].innermost:
        return 2
    return 1


def region_items(
    program: MmaDagProgram,
) -> list[tuple[Tcgen05SmemItem, object]]:
    """The shared-memory items of ``program`` in a fixed order with their
    program payloads (the facts keep the items, the plan the payloads)."""
    items: list[tuple[Tcgen05SmemItem, object]] = []
    seen: set[tuple[object, ...]] = set()
    segments = root_segments(program)
    phases = exchange_phases(program)
    # An exchange tile lives in the frame of the loop whose dots read it
    # (sibling frames overlap), in the root frame when several loops do;
    # one read only by the root dots of one segment overlays the loop
    # frames.  In a loop frame the item's segment is the tile's batch
    # phase (``exchange_phases``): the phases of one loop overlay each
    # other.
    consumers: dict[tuple[object, ...], set[int]] = {}
    exchange_segments: dict[tuple[object, ...], set[int]] = {}
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.kind == "frag":
                key = tile_key(program, operand)
                consumers.setdefault(key, set()).add(dot.loop)
                exchange_segments.setdefault(key, set()).add(segments.get(dot.node, -1))
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
                if load.dtype != operand.dtype and not raw_overlays_tile(operand):
                    # A root raw tile is converted in the prologue: dead
                    # once the loops run.  (A raw tile widened into an fp32
                    # tile lands on the tile's own bytes: no item.)
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
                                0 if frame < 0 else -1,
                            ),
                            (dot, operand),
                        )
                    )
            else:
                regions = consumers[key]
                frame = next(iter(regions)) if len(regions) == 1 else -1
                segment = phases.get(key, -1)
                if regions == {-1} and len(exchange_segments[key]) == 1:
                    segment = next(iter(exchange_segments[key]))
                items.append(
                    (
                        Tcgen05SmemItem(
                            ITEM_EXCHANGE, frame, rows, cols, es, 1, segment
                        ),
                        (dot, operand),
                    )
                )
    for derived, access in aux_tile_loads(program):
        positions = access.tile_positions
        items.append(
            (
                Tcgen05SmemItem(
                    ITEM_AUX,
                    derived.loop,
                    access.dims[positions[0]].index,
                    access.dims[positions[1]].index,
                    access.dtype.itemsize,
                    0,
                ),
                (derived, access),
            )
        )
    for store in program.stores:
        # A root store's staging buffer drains within its segment.
        rows, cols = program.slots[program.frag_slots[store.frag]]
        items.append(
            (
                Tcgen05SmemItem(
                    ITEM_STAGING,
                    store.loop,
                    rows.index,
                    cols.index,
                    store.access.dtype.itemsize,
                    staging_copies(program, store),
                    segments.get(store.access.node, -1) if store.loop < 0 else -1,
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


def root_segments(program: MmaDagProgram) -> dict[Node, int]:
    """The root segment (the k-th run of root steps between the loops) of
    every root dot and root store (by its access node)."""
    result: dict[Node, int] = {}
    segment = 0
    for step in program.body:
        if isinstance(step, Loop):
            segment += 1
        elif isinstance(step, Dot):
            result[step.node] = segment
        elif isinstance(step, Store):
            result[step.access.node] = segment
    return result


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
    skip: frozenset[int] = frozenset(),
) -> RegionLayout:
    """Frame layout: a loop's items follow the frames of the loops nested
    in it (sibling frames share one extent); a frame's ring (every ring
    item of the loop, one slot per stage) comes before its other items.
    The root items of one segment (``Tcgen05SmemItem.segment``) overlay the
    loop frames from the top of their extent (each segment packs its own
    from there, the frames are dead while they are live); the extent grows
    only when a segment's pack is larger than the frames.  A loop frame's
    items with a segment are its exchange tiles' batch phases
    (``exchange_phases``): every phase packs from the same offset behind
    the ring, the frame grows by the largest.  The items at the ``skip``
    indices take no space (offset -1)."""
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
        if frame < 0:
            packs: dict[int, int] = {}
            for index, item in enumerate(items):
                if index in skip or item.frame != frame or item.segment < 0:
                    continue
                packs[item.segment] = (
                    packs.get(item.segment, 0)
                    + item_bytes(item, resolved) * item.copies
                )
            offset = max(offset, *packs.values()) if packs else offset
            tops = dict.fromkeys(packs, offset)
            for index, item in enumerate(items):
                if index in skip or item.frame != frame or item.segment < 0:
                    continue
                tops[item.segment] -= item_bytes(item, resolved) * item.copies
                offsets[index] = tops[item.segment]
        slot = 0
        for index, item in enumerate(items):
            if index in skip:
                offsets[index] = -1
            elif item.frame == frame and item.copies == 0:
                offsets[index] = slot
                slot += item_bytes(item, resolved)
        if slot:
            ring_base[frame] = offset
            ring_bytes[frame] = slot
            offset += slot * stages
        if frame >= 0:
            phase_tops: dict[int, int] = {}
            for index, item in enumerate(items):
                if index in skip or item.frame != frame or item.segment < 0:
                    continue
                offsets[index] = phase_tops.get(item.segment, offset)
                phase_tops[item.segment] = (
                    offsets[index] + item_bytes(item, resolved) * item.copies
                )
            offset = max(offset, *phase_tops.values()) if phase_tops else offset
        for index, item in enumerate(items):
            if index in skip or item.segment >= 0:
                continue
            if item.frame == frame and item.copies != 0:
                offsets[index] = offset
                offset += item_bytes(item, resolved) * item.copies
        return offset

    total = layout(-1, 0)
    return RegionLayout(tuple(offsets), ring_base, ring_bytes, total)


def region_aux_skip(
    items: Sequence[Tcgen05SmemItem],
    loops: Sequence[LoopFacts],
    resolved: Sequence[int],
    stages: int,
    capacity: int,
    loads: str = MMA_DAG_LOADS_TMA,
) -> frozenset[int]:
    """The aux items left out of the layout: all of them under the ``lane``
    loads mode; under ``tma`` the largest ring footprints first until the
    layout (with its barriers) fits ``capacity`` (0: unlimited)."""
    aux = [index for index, item in enumerate(items) if item.kind == ITEM_AUX]
    if loads != MMA_DAG_LOADS_TMA:
        return frozenset(aux)
    skip: set[int] = set()
    barriers = 8 * (stages * len(loops) + len(loops) + 2) + 16
    while capacity and aux:
        layout = region_layout(items, loops, resolved, stages, frozenset(skip))
        if layout.total + barriers <= capacity:
            break
        largest = max(
            aux, key=lambda index: (item_bytes(items[index], resolved), index)
        )
        aux.remove(largest)
        skip.add(largest)
    return frozenset(skip)


def region_barrier_bytes(tc: Tcgen05Facts, stages: int) -> int:
    """The mbarriers (one per ring slot, one per staged region, the commit
    barrier) and the TMEM slot."""
    return 8 * (stages * tc.ring_loops + tc.staged_regions + 1) + 16


def region_smem_bytes(
    facts: CuteMmaDagFacts,
    resolved: Sequence[int],
    stages: int,
    loads: str = MMA_DAG_LOADS_TMA,
) -> int:
    """The shared memory of the resolved sizes under ``loads`` (``tma``
    keeps the aux tiles that fit, so the ``lane`` footprint is never
    larger and a config is legal under one mode iff under the other)."""
    tc = facts.tcgen05
    assert tc is not None
    skip = region_aux_skip(
        tc.items, facts.loops, resolved, stages, facts.smem_capacity or 0, loads
    )
    layout = region_layout(tc.items, facts.loops, resolved, stages, skip)
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
    live_groups = tc.live_groups if tc.live_groups is not None else facts.live_groups
    for group in live_groups:
        for a in group:
            for b in group:
                if a != b:
                    conflicts.add((a, b))
    placed: dict[int, tuple[int, int]] = {}
    total = 0
    resident = resident_slots(tc, resolved)
    for slot in tc.tmem_slots:
        if slot in resident:
            continue
        _rows, cols = facts.frags[slot]
        width = _align(resolved[cols], TCGEN05_SLOT_ALIGN)
        busy = sorted(
            span
            for other, span in placed.items()
            if (slot, other) in conflicts or (other, slot) in conflicts
        )
        column = 0
        # A loop output computed in another slot takes its carry's columns
        # when the two never conflict (the carry is dead from its last read
        # to the copy, which then vanishes) and the carry is placed already.
        for output, carry in tc.tmem_aliases:
            if output != slot or carry not in placed:
                continue
            if (slot, carry) in conflicts or (carry, slot) in conflicts:
                continue
            start, end = placed[carry]
            if end - start == width and not any(
                start < other_end and other_start < end
                for other_start, other_end in busy
            ):
                column = start
                break
        else:
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
    dot's accumulator and its result, and
    -- because the batches reorder a region (a later dot independent of the
    pending batch joins it, a derived value independent of it is hoisted
    ahead of the batch's exchanges, work reading the batch's results waits
    for the next block) -- the result slot of a dot or derived value that
    may move up to an earlier dot's position against every slot live
    anywhere between the two in program order (bwd_o_kda's ``dA`` and
    ``dv_new`` dots issue in one batch while the work between them, reading
    ``dA``, is deferred; a derived value after a batch dot is stored to
    tensor memory before the dot's operands are read from it).  A step
    moves only when it does not read the earlier dot's result (through
    fragments, accumulators or derived values), so a dependent pair keeps
    the program-order liveness.  A slot defined
    outside a loop and read inside it is live throughout the outermost such
    loop (every iteration reads it again), beyond the planner's linear
    liveness: it conflicts with every slot touched anywhere in that loop.
    A hoisted exchange (``hoisted_exchange_loop``) reads its fragment's slot
    at the root, ahead of the loop, not where the dot runs."""
    pairs: set[tuple[int, int]] = set()

    def conflict(a: int, b: int) -> None:
        if a != b:
            pairs.add((min(a, b), max(a, b)))

    def exchange_region(dot: Dot, operand: Operand) -> int:
        # A hoisted exchange reads the fragment's slot at the root, ahead of
        # the loop (``hoisted_exchange_loop``), not where the dot runs.
        if hoisted_exchange_loop(program, dot, operand) is not None:
            return -1
        return dot.loop

    copied = {
        (loop.index, carry.slot): source
        for loop, carry, source in copied_carries(program)
    }
    touched: dict[int, set[int]] = {}
    for dot in program.dots:
        if dot.acc_slot is not None:
            conflict(dot.acc_slot, dot.slot)
            touched.setdefault(dot.loop, set()).add(dot.acc_slot)
        touched.setdefault(dot.loop, set()).add(dot.slot)
        for operand in (dot.a, dot.b):
            if operand.frag is not None:
                touched.setdefault(exchange_region(dot, operand), set()).add(
                    program.frag_slots[operand.frag]
                )
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
        for carry in loop.carries:
            if carry.init_frag is not None:
                # The seed is read at the loop's entry, in the parent region
                # (the init loads every seed before it writes a carry, so a
                # seed and the carries it seeds may share columns).
                touched.setdefault(loop.parent, set()).add(
                    program.frag_slots[program.resolve(carry.init_frag)]
                )
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
                uses.setdefault(program.frag_slots[operand.frag], set()).add(
                    exchange_region(dot, operand)
                )
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
            source = copied.get((loop.index, carry.slot))
            # A copied carry is redefined inside its loop (the copy at the
            # iteration's end), so within that loop its liveness is the
            # linear one: from the entry to its last read; the copy reads
            # the output slot in the loop.
            defined[carry.slot] = loop.parent if source is None else loop.index
            uses.setdefault(carry.slot, set()).add(loop.index)
            if source is not None:
                uses.setdefault(source, set()).add(loop.index)
            if carry.init_frag is not None:
                uses.setdefault(
                    program.frag_slots[program.resolve(carry.init_frag)], set()
                ).add(loop.parent)
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
    # The region emitter's flat program order (``region_flat``), the live
    # sets at its positions, each step's region: a loop body, or the k-th
    # root segment between the loops.
    live = region_live_slots(program)
    ordered, _spans = region_flat(program)

    def depends(later: Dot | Derived, earlier: Dot) -> bool:
        return step_depends(program, later, earlier)

    # The moving elements of a batch (``region_blocks``): a later dot
    # independent of the pending batch joins it, and a derived value
    # independent of it is hoisted into the block's epilogue work, written
    # to tensor memory ahead of the batch's exchanges and rescales, which
    # read their sources after it.  Either lands at the earlier dot's
    # position, so its slot conflicts with every slot live between the two.
    # A derived value independent of every dot from the head on is hoisted
    # into the head's block: its store also precedes the head dot's own
    # operand and accumulator reads (live through the position before the
    # dot only; a hoisted operand is exchanged at the root, not there).
    dots = [
        (index, region, step)
        for index, (region, step) in enumerate(ordered)
        if isinstance(step, Dot)
    ]
    movers = [
        (index, region, step)
        for index, (region, step) in enumerate(ordered)
        if isinstance(step, (Dot, Derived))
    ]
    for ia, region_a, earlier in dots:
        for ib, region_b, later in movers:
            if ib <= ia or region_b != region_a or depends(later, earlier):
                continue
            for index in range(ia, ib + 1):
                for slot in live[index]:
                    conflict(later.slot, slot)
            if isinstance(later, Derived) and not any(
                depends(later, between)
                for ik, region_k, between in dots
                if ia < ik < ib and region_k == region_a
            ):
                for operand in (earlier.a, earlier.b):
                    if (
                        operand.frag is not None
                        and hoisted_exchange_loop(program, earlier, operand) is None
                    ):
                        conflict(later.slot, program.frag_slots[operand.frag])
                if earlier.acc_frag is not None:
                    conflict(later.slot, program.frag_slots[earlier.acc_frag])
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
                    mn_major=not operand_geometry(program, operand)[2],
                    passed=operand.pointwise or load.dtype != operand.dtype,
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
                    mn_major=not operand_geometry(program, operand)[2],
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
        live_groups=region_live_slots(program),
        items=items,
        tmem_slots=tmem_slots(program),
        register_carries=register_carry_facts(program),
        tmem_conflicts=tmem_conflicts(program),
        tmem_aliases=tuple(
            (source, carry.slot) for _loop, carry, source in copied_carries(program)
        ),
        pass_tiles=sum(
            1
            for item, payload in region_items(program)
            if item.kind == ITEM_TILE
            and (
                cast("tuple[Dot, Operand]", payload)[1].pointwise
                or cast("tuple[Dot, Operand]", payload)[1].load.dtype  # type: ignore[union-attr]
                != cast("tuple[Dot, Operand]", payload)[1].dtype
            )
        ),
        tf32=any(dot.a.dtype == torch.float32 for dot in program.dots),
        ring_loops=ring_loops,
        staged_regions=staged_regions,
        aux_items=sum(1 for item in items if item.kind == ITEM_AUX),
    )


def region_config_reason(
    facts: CuteMmaDagFacts, resolved: Sequence[int], stages: int
) -> str | None:
    """The region-specific legality of the resolved sizes (the shared tile
    / fragment rules ran already)."""
    tc = facts.tcgen05
    assert tc is not None
    # 32-bit MN-major tiles live under the 32-byte-unit swizzle, whose rows
    # are 128 bytes: 32 fp32 columns at least (the planner would decline the
    # config at codegen otherwise, after normalization admitted it).
    for cols, elem_bytes, mn_major in (
        *((t.cols, t.elem_bytes, t.mn_major) for t in tc.tiles),
        *((x.cols, x.elem_bytes, x.mn_major) for x in tc.exchanges),
    ):
        if (
            mn_major
            and elem_bytes == 4
            and resolved[cols] * elem_bytes < ATOM_MAX_BYTES
        ):
            return (
                f"an MN-major tf32 tile of {resolved[cols]} columns needs "
                f"{ATOM_MAX_BYTES}-byte rows ({ATOM_MAX_BYTES // elem_bytes} columns)"
            )
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
    loads_mode: str = MMA_DAG_LOADS_TMA,
) -> Tcgen05Plan:
    env = CompileEnvironment.current()
    tc = facts.tcgen05
    assert tc is not None and tc.regions
    items = region_items(program)
    skip = region_aux_skip(
        [item for item, _p in items],
        facts.loops,
        sizes,
        stages,
        facts.smem_capacity or 0,
        loads_mode,
    )
    layout = region_layout(
        [item for item, _p in items], facts.loops, sizes, stages, skip
    )
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
        slabs = col_count // slab_elems
        slab_bytes = row_count * slab_elems * elem_bytes
        key = (access.fake, dims, tuple(box), swizzle, elem_bytes)
        shared = shared_maps.get(key)
        if shared is not None:
            # The box is one swizzle atom wide, so a 64- and a 128-column
            # tile of one tensor share the map; the slab count is the
            # tile's own (the wider tile loads two atoms, not the first
            # tile's one: wy_dL at [.., 64, 128] loaded half of dvni).
            return dataclasses.replace(
                shared, access=access, offset=at, slabs=slabs, slab_bytes=slab_bytes
            )
        spec = TmaSpec(
            index=len(tma),
            param=f"md_tma{len(tma)}",
            access=access,
            dims=dims,
            box=tuple(box),
            swizzle=swizzle,
            slab_dim=slab_dim,
            slab_elems=slab_elems,
            slabs=slabs,
            slab_bytes=slab_bytes,
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
        rows_axis, cols_axis, k_major = operand_geometry(program, operand)
        rows, cols = sizes[rows_axis.index], sizes[cols_axis.index]
        es = operand.dtype.itemsize
        atom = tcgen05_atom_bytes(cols * es)
        if atom is None:
            raise _decline(f"operand {operand.value.name} has no swizzled UMMA layout")
        swizzle = _SWIZZLE_BY_ROW_BYTES[atom]
        if es == 4 and not k_major:
            # 32-bit MN-major: 128-byte rows under the 32-byte-unit swizzle
            # (the exchange / widen writes and TMA's s128b_atom_32b boxes).
            if atom != ATOM_MAX_BYTES:
                raise _decline(
                    f"operand {operand.value.name}: an MN-major tf32 tile needs "
                    "128-byte rows"
                )
            swizzle = ATOM_32B_SWIZZLE
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
            if route == ROUTE_INPLACE and es == 4:
                # The in-place pass rewrites 16-bit pairs.
                raise _decline(
                    f"operand {operand.value.name}: no in-place pass over fp32"
                )
            spec = raw_spec = None
            raw_offset = raw_row_bytes = 0
            src_rows_axis, src_cols_axis = source_geometry(operand)
            src_rows, src_cols = sizes[src_rows_axis.index], sizes[src_cols_axis.index]
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
                # A widened 16-bit raw tile lands on its tile's own bytes
                # (``raw_overlays_tile``: no raw item).
                raw_offset = raw_offsets.get(key, offset)
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
    # The derived values' tile inputs: TMA into the loop ring when the source
    # is aligned and the row has a swizzled layout (the lanes read their
    # chunks conflict-free); otherwise the item stays unused and the lanes
    # load their pairs from global memory as before.
    aux_tiles: dict[int, list[AuxTile]] = {}
    for (item, payload), offset in zip(items, layout.offsets, strict=True):
        if item.kind != ITEM_AUX or offset < 0:
            continue
        derived, access = cast("tuple[Derived, Access]", payload)
        rows, cols = sizes[item.rows], sizes[item.cols]
        es = item.elem_bytes
        atom = tcgen05_atom_bytes(cols * es)
        if (
            atom is None
            or rows > TCGEN05_MAX_ROWS
            or len(_tma_dims(access)) > TMA_MAX_RANK
            or async_tile_reason(env, access, 1) is not None
            or (rows, cols) != (sizes[derived.rows.index], sizes[derived.cols.index])
        ):
            continue
        swizzle = _SWIZZLE_BY_ROW_BYTES[atom]
        positions = access.tile_positions
        spec = tma_spec(
            access,
            access.dims[positions[0]],
            access.dims[positions[1]],
            rows,
            cols,
            es,
            swizzle,
            atom,
            offset,
        )
        aux_tiles.setdefault(item.frame, []).append(
            AuxTile(
                key=("aux", access.node, item.frame),
                tag=f"x{sum(len(group) for group in aux_tiles.values())}",
                access=access,
                frame=item.frame,
                rows=rows,
                cols=cols,
                elem_bytes=es,
                atom_bytes=atom,
                swizzle=swizzle,
                slabs=(cols * es) // atom,
                offset=offset,
                tma=spec,
            )
        )
    # Aux reads sharing a ring operand's tile (TMA'd straight or rewritten
    # in place behind the reads): no TMA of their own, no bytes in the
    # ring's transaction count; a synchronous-pass operand writes its tile
    # at the iteration top, so those keep the per-lane loads.
    shared_seen: set[tuple[Node, int]] = set()
    for derived in program.derived:
        for access in derived.loads:
            if loads_mode != MMA_DAG_LOADS_TMA:
                break
            if (
                len(access.tile_dims) != 2
                or access.tile_dim_strides[1] != 1
                or (access.node, derived.loop) in shared_seen
            ):
                continue
            found = aux_shared_operand(program, derived, access)
            if found is None:
                continue
            shared_seen.add((access.node, derived.loop))
            owner = by_key[tile_key(program, found[1])]
            if owner.route not in (ROUTE_DIRECT, ROUTE_INPLACE) or (
                not owner.ring and owner.route != ROUTE_DIRECT
            ):
                continue
            aux_tiles.setdefault(derived.loop, []).append(
                AuxTile(
                    key=("aux", access.node, derived.loop),
                    tag=f"x{sum(len(group) for group in aux_tiles.values())}",
                    access=access,
                    frame=owner.frame,
                    rows=owner.rows,
                    cols=owner.cols,
                    elem_bytes=owner.elem_bytes,
                    atom_bytes=owner.atom_bytes,
                    swizzle=owner.swizzle,
                    slabs=owner.slabs,
                    offset=owner.offset,
                    tma=None,
                    ring=owner.ring,
                    rewritten=owner.route == ROUTE_INPLACE,
                    owner_key=owner.key,
                )
            )
    # Fragment slots: TMEM columns at 32-column-aligned offsets, shared by
    # slots never live together (register-only slots take none).
    if tuple(tc.tmem_slots) != tmem_slots(program):
        raise _decline("the tensor-memory slots changed since admission")
    resident = resident_slots(tc, sizes)
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
    # Staging tiles whose rows are one swizzle atom hold TMA's layout: one
    # thread stores the whole tile by TMA instead of every thread draining
    # 16-byte packets (wy_dL at s2: 116 global store instructions per CTA
    # against Triton's 18).
    store_tma: dict[Node, TmaSpec] = {}
    for store in staged:
        slot = slots[program.frag_slots[store.frag]]
        row_bytes = slot.cols * store.access.dtype.itemsize
        if (
            row_bytes in (16, 32, 64, 128)
            and len(_tma_dims(store.access)) <= TMA_MAX_RANK
            and len(tma) < TMA_MAX_MAPS
        ):
            rows_axis, cols_axis = program.slots[program.frag_slots[store.frag]]
            store_tma[store.access.node] = tma_spec(
                store.access,
                rows_axis,
                cols_axis,
                slot.rows,
                slot.cols,
                store.access.dtype.itemsize,
                _SWIZZLE_BY_ROW_BYTES.get(row_bytes, "none"),
                row_bytes,
                staging_offsets[store.access.node],
            )
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
    for loop, aux_group in aux_tiles.items():
        ring_tx[loop] = ring_tx.get(loop, 0) + sum(
            tile.tma.bytes for tile in aux_group if tile.tma is not None
        )
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
        store_tma=store_tma,
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
        aux_tiles={loop: tuple(group) for loop, group in aux_tiles.items()},
        scratch_offset=scratch_offset,
        tmem_slots=tuple(slot for slot in tc.tmem_slots if slot not in resident),
        register_carries=resident,
    )


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


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
        # Loop carries living in the epilogue registers at this config
        # (``register_carries``, ``resident_slots``): ``md_rc{slot}_{chunk}_{j}``
        # scalars, defined at the kernel's start, never loaded or stored.
        self.resident: set[int] = set(self.tc.register_carries)
        self.ring_loops = [
            loop.index
            for loop in program.loops
            if self.tc.ring_tiles.get(loop.index) or self.tc.aux_tiles.get(loop.index)
        ]
        self.counted = {
            loop.index
            for loop in program.loops
            if loop.index in self.tc.ring_tiles
            or loop.index in self.tc.aux_tiles
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

    def reg(self, slot: int, *key: object) -> str:  # type: ignore[override]
        if slot in self.resident and slot not in self.slot_regs:
            chunk, j = key
            return f"md_rc{slot}_{chunk}_{j}"
        return super().reg(slot, *key)

    def resident_names(self, slot: int) -> list[tuple[int, int, str]]:
        return [
            (chunk, j, f"md_rc{slot}_{chunk}_{j}")
            for chunk in range(self.ld_chunks(slot))
            for j in range(self.chunk_cols(slot, chunk))
        ]

    def row_array(self, source: int, chunk: int) -> str:
        """The array holding this thread's row of ``source`` (chunk
        ``chunk``) in a seed or copy block: the tcgen05.ld array, or for a
        register carry a fresh array of its scalars."""
        if source not in self.resident:
            return f"{self.vec}{source}_{chunk}"
        cols = self.chunk_cols(source, chunk)
        name = self.bind(
            "md_seed", f"cutlass.Array(cutlass.Float32, {cols}, alignment=16)"
        )
        for j in range(cols):
            self.emit(f"{name}[{j}] = {self.reg(source, chunk, j)}")
        return name

    def emit_carry_registers(self, name: str, slot: int, chunk: int) -> None:
        """A register carry's row from the array ``name`` (a seed or a
        derived value's result); a tensor-memory carry stores it."""
        if slot in self.resident:
            for _chunk, j, scalar in self.resident_names(slot):
                if _chunk == chunk:
                    self.emit(f"{scalar} = {name}[{j}]")
            return
        self.emit_st_array(name, slot, chunk)

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
        cols = self.local_cols(slot)
        packets = cols // 4
        # Under the 32-byte-unit swizzle a packet column lands on four of
        # the eight bank groups across a warp's rows (the XOR runs over the
        # row mod 4), so every store instruction took twice the wavefronts:
        # the lanes of rows 4-7 of each eight write the next packet instead
        # (packets of adjacent indices differ in parity, hence in bank
        # group half), which spreads each instruction over all eight.
        rotate = tile.swizzle == ATOM_32B_SWIZZLE and packets > 1
        with self.row_guard(slot):
            values = []
            for c in range(cols):
                self.evaluator.reg = (c // TCGEN05_LD_COLS, c % TCGEN05_LD_COLS)  # type: ignore[assignment]
                coords = self.element_coords(value, slot, c, operand.swap)
                values.append(self.evaluator.evaluate(value, coords))
            self.evaluator.reg = None
            turn = odd = ""
            if rotate:
                turn = self.bind("md_rot", "(md_lane // 4) % 2")
                odd = self.bind("md_rot", f"{turn} == 1")
            for packet in range(packets):
                if rotate:
                    other = (packet + 1) % packets
                    words = [
                        self.bind(
                            "md_v",
                            f"cutlass.Float32({values[4 * other + e]}) if {odd} "
                            f"else cutlass.Float32({values[4 * packet + e]})",
                        )
                        for e in range(4)
                    ]
                    index = f"({packet} + {turn}) % {packets}"
                    chunk = (
                        f"(md_cb{slot} // 4 + {index})"
                        if self.half_cols(slot)
                        else index
                    )
                else:
                    words = values[4 * packet : 4 * packet + 4]
                    chunk = self.chunk_expr(slot, packet, 4)
                address = self.tile_address(tile, "0", row, chunk)
                self.emit(f"{_RT}.sts_v4_f32({address}, {', '.join(words)})")

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
            rewrite(words, prow, pch)

        def rewrite(words: list[str], prow: str, pch: str) -> None:
            for q, word in enumerate(words):
                lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                self.evaluator.tile_leaves[(load.node, 2 * q)] = lo
                self.evaluator.tile_leaves[(load.node, 2 * q + 1)] = hi
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

        src_rows = tile.rows
        if tile.raw_offset != tile.offset:
            self.packets(
                src_rows * raw_cpr, threads, idx_base, raw_cpr, body, tag=tile.tag
            )
            return
        # The raw tile lands on the fp32 tile's own bytes
        # (``raw_overlays_tile``): every thread reads all of its raw packets
        # into registers, a CTA barrier, then every thread writes -- a
        # thread's fp32 row covers two threads' raw rows.
        rounds = self.packet_rounds(
            src_rows * raw_cpr, threads, idx_base, raw_cpr, tile.tag
        )
        states: list[list[str]] = []
        for prow, pch, guard in rounds:
            words = [self.df.new_var("md_w") for _ in range(4)]
            load_line = (
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32("
                f"{self.raw_address(tile, stage, prow, pch)})"
            )
            if guard is None:
                self.emit(load_line)
            else:
                for word in words:
                    self.emit(f"{word} = cutlass.Int32(0)")
                with self.nest(f"if {guard}:"):
                    self.emit(load_line)
            states.append(words)
        self.emit(self.sync)
        for words, (prow, pch, guard) in zip(states, rounds, strict=True):
            with self.block(f"if {guard}:" if guard else None):
                self.evaluator.push()
                rewrite(words, prow, pch)
                self.evaluator.pop()

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
        if staging_copies(self.program, store) == 2:
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

    # -- derived values' tile inputs ------------------------------------------

    def aux_tile(self, step: Derived, access: Access) -> AuxTile | None:
        for tile in self.tc.aux_tiles.get(step.loop, ()):
            if tile.access.node is access.node:
                return tile
        return None

    def aux_address(self, tile: AuxTile, stage: str, row: str, chunk: str) -> str:
        """Byte address of 16-byte chunk ``chunk`` of row ``row`` of the aux
        tile in ring slot ``stage`` (the operand tiles' UMMA layout: slabs of
        one swizzle atom width, the atom's XOR swizzle inside)."""
        cpa = tile.chunks_per_atom
        if tile.slabs == 1:
            slab, local = "0", chunk
        else:
            slab, local = f"({chunk}) // {cpa}", f"({chunk}) % {cpa}"
        base = f"md_s0 + {tile.offset}"
        if tile.ring:
            base = (
                f"md_s0 + {self.tc.ring_offsets[tile.frame]} + ({stage}) * "
                f"{self.tc.ring_bytes[tile.frame]} + {tile.offset}"
            )
        terms = [
            base,
            f"{row} * {tile.atom_bytes}",
            f"{_swizzled_chunk(row, local, cpa)} * 16",
        ]
        if slab != "0":
            terms.append(f"{slab} * {tile.slab_bytes}")
        return " + ".join(terms)

    def emit_derived_access_prefetch(self, step: Derived, access: Access) -> None:  # type: ignore[override]
        """This thread's row of a tile input from its ring slot: one 16-byte
        shared load per chunk (four fp32 or eight 16-bit columns) in place
        of the per-pair global loads."""
        tile = self.aux_tile(step, access)
        if tile is None:
            super().emit_derived_access_prefetch(step, access)
            return
        if (
            not tile.ring
            and tile.owner_key is not None
            and tile.owner_key not in self.settled
            and self.tc.staged_tx.get(tile.frame)
        ):
            # A staged / hoisted owner its consumers' settle has not waited
            # yet: its TMA landed on the frame's barrier (a direct tile, no
            # pass to run).  The reads may sit under the epilogue warps'
            # guard, so the owner stays unsettled for the consumers' wait
            # (a wait on a completed phase returns at once).
            self.emit(
                f"{_RT}.mbar_wait({self.sf_name(tile.frame)}, {self.staged_phase(tile.frame)})"
            )
        slot = step.slot
        per_chunk = 16 // tile.elem_bytes
        stage = self.stage_expr(tile.frame) if tile.ring else "0"
        row = self.row_var(slot)
        for col in range(0, self.local_cols(slot), per_chunk):
            chunk = self.chunk_expr(slot, col // per_chunk, per_chunk)
            address = self.aux_address(tile, stage, row, chunk)
            if tile.elem_bytes == 4:
                names = [self.df.new_var("md_f") for _ in range(4)]
                self.emit(f"{', '.join(names)} = {_RT}.lds_v4_f32({address})")
                for q in range(2):
                    self.derived_words[(access.node, col + 2 * q)] = (
                        names[2 * q],
                        names[2 * q + 1],
                    )
            else:
                names = [self.df.new_var("md_u") for _ in range(4)]
                self.emit(f"{', '.join(names)} = {_RUNTIME}.lds_v4_b32({address})")
                for q in range(4):
                    self.derived_words[(access.node, col + 2 * q)] = (names[q],)

    def emit_derived(  # type: ignore[override]
        self, step: Derived, loaded: set[int]
    ) -> None:
        """Outside a ring iteration's prefetch (an outer loop, the root), a
        derived value's shared tile inputs are read here, just ahead of the
        value, and forgotten after it (the registers live in this block)."""
        fresh = []
        for access in step.loads:
            if (access.node, 0) in self.derived_words:
                continue
            if self.aux_tile(step, access) is not None:
                self.emit_derived_access_prefetch(step, access)
                fresh.append(access.node)
        super().emit_derived(step, loaded)
        if step.slot in self.resident:
            # The carry's next value: into its registers (the base emitter
            # left the result array as the block's register source).
            prefix = self.slot_regs.pop(step.slot)
            for chunk, j, scalar in self.resident_names(step.slot):
                self.emit(f"{scalar} = {prefix}{chunk}[{j}]")
        for node in fresh:
            for key in [key for key in self.derived_words if key[0] is node]:
                del self.derived_words[key]

    def forget_overlaid(self, tile: Tcgen05Tile) -> None:
        """Writing ``tile`` stales every exchange tile of its frame sharing
        its bytes (another batch phase of the loop, ``exchange_phases``; a
        root tile of another segment): such a tile has no later consumer
        by construction, and would be exchanged again if one appeared."""
        start, end = tile.offset, tile.offset + tile.bytes
        for other in self.tc.exchange_tiles:
            if other.key == tile.key or other.ring or other.frame != tile.frame:
                continue
            if other.offset < end and start < other.offset + other.bytes:
                self.written.discard(other.key)
                self.root_written.discard(other.key)

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
                if subscript.offset is not None:
                    value = f"{value} + {self.uniform(subscript.offset, {})}"
            if position == spec.slab_dim and slab:
                value = f"{value} + {slab * spec.slab_elems}"
            coords.append(f"cutlass.Int32({value})")
        return coords

    def emit_tma_group(
        self,
        tiles: Sequence[Tcgen05Tile | AuxTile],
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
            [*tc.ring_tiles.get(loop, ()), *tc.aux_tiles.get(loop, ())],
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
        self.settle_keys(
            {
                tile_key(self.program, operand)
                for dot in consumers
                for operand in (dot.a, dot.b)
                if operand.load is not None
            },
            region,
        )

    def settle_keys(self, keys: set[tuple[object, ...]], region: int) -> None:
        """``settle_tiles`` by the tiles' sharing keys."""
        tc = self.tc
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
        """The segment's blocks (``region_block_steps``) with every
        tile-only derived value without a tensor-memory slot recomputed in
        each block that reads it (the registers do not cross the block); a
        value over fragments read in another block holds a slot
        (``derived_crosses``), so it is never recomputed."""
        program = self.program
        blocks: list[Block] = []
        register_only: dict[int, Derived] = {}
        for work, dots in region_block_steps(program, steps):
            placed: list[Step] = []
            for step in work:
                if isinstance(step, Derived):
                    reads = {program.frag_slots[frag] for frag in step.frags}
                elif isinstance(step, (Store, Reduce)) and step.frag is not None:
                    reads = {program.frag_slots[step.frag]}
                else:
                    reads = set()
                for slot in sorted(reads):
                    source = register_only.get(slot)
                    if source is not None and source not in placed:
                        assert not source.frags, source.node.name
                        placed.append(source)
                placed.append(step)
                if isinstance(step, Derived) and step.slot not in self.tc.tmem_slots:
                    register_only[step.slot] = step
            blocks.append((placed, dots))
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
            if any(
                isinstance(step, Store)
                and step.loop >= 0
                and staging_copies(program, step) == 1
                for step in work
            ) and self.tma_stores([step for step in work if isinstance(step, Store)]):
                # A single-buffered loop store: the TMA store of the previous
                # iteration finishes reading the buffer before any thread
                # stages into it again.
                self.emit_tma_store_wait(read=True)
                self.emit(self.sync)
            if first_snapshots or work or exchanges or preps:
                with self.block("if md_epi:"):
                    self.slot_regs = {}
                    loaded: set[int] = set(self.resident)
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
                        self.forget_overlaid(tile)
                        self.written.add(tile.key)
                        if loop is None:
                            self.root_written.add(tile.key)
                    for dot in preps:
                        self.emit_rescale(
                            dot, loaded, prefix=self.df.new_var("md_rs") + "_"
                        )
                    self.slot_regs = {}
                if exchanges or self.tma_stores([d[0] for d in self.drains]):
                    # Exchange tiles and TMA-drained staging to the async
                    # proxy (the writers only).
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
                self.emit_tma_store_wait(read=True)
                self.emit_commit()
            self.emit("md_cn = md_cn + 1")
            self.waited = False
            self.flush_drains()
        self.current_blocks = []

    def emit_tma_store_wait(self, read: bool, pending: int = 0) -> None:
        """The store thread waits until at most ``pending`` of its TMA store
        groups are outstanding (``read``: their reads of the staging
        buffers).  Ahead of a commit: the stage-in reusing a buffer two
        iterations after its write waits for a later commit, which this
        wait precedes; ahead of a loop: its prologue TMA may land on a
        staging buffer overlaying the frames; at the top of a dot-less
        loop's iteration (no commit to wait behind): all but the previous
        iteration's group, whose buffer this iteration does not touch."""
        if not self.tc.store_tma:
            return
        with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
            self.emit(
                f"{_PRIMS}.cp_async_bulk_wait_group({pending}{', read=True' if read else ''})"
            )

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
        """Drain the staged stores (behind a CTA barrier the caller emitted):
        the store thread issues the TMA stores as one bulk group, every
        thread drains the other tiles' packets."""
        tma = self.tma_stores([store for store, _parity, _values in self.drains])
        for store, parity, values in self.drains:
            if store in tma:
                continue
            self.emit_stage_out(
                store,
                parity,
                cast("str", values),
                "md_tidx",
                self.plan.threads,
            )
        if tma:
            with self.nest(f"if md_tidx == {TMA_STORE_THREAD}:"):
                for store, parity, values in self.drains:
                    if store not in tma:
                        continue
                    spec = self.tc.store_tma[store.access.node]
                    base = self.staging_base(store, parity).replace("md_s0 + ", "", 1)
                    coords = ", ".join(
                        self.region_tma_coords(spec, cast("dict[int, str]", values), 0)
                    )
                    self.emit(
                        f"{_PRIMS}.cp_async_bulk_tensor_global_shared_cta({spec.param}.get_ptr(), "
                        f"md_smem + ({base}), [{coords}])"
                    )
                self.emit(f"{_PRIMS}.cp_async_bulk_commit_group()")
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
        if any(carry.slot not in self.resident for carry in carries) or any(
            carry.init_frag is not None for carry in carries
        ):
            self.wait_latest()
        with self.block("if md_epi:"):
            self.slot_regs = {}
            loaded: set[int] = set(self.resident)
            # Every seed is read before any carry is written: a seed may
            # share its columns with the carries it seeds (``tmem_conflicts``
            # counts the read at the loop's entry, nothing more).
            for carry in carries:
                if carry.init_frag is not None:
                    source = program.frag_slots[program.resolve(carry.init_frag)]
                    if source not in loaded:
                        self.emit_ld(source)
                        loaded.add(source)
            for carry in carries:
                slot = carry.slot
                init = carry.init
                if carry.init_frag is not None:
                    source = program.frag_slots[program.resolve(carry.init_frag)]
                    if carry.init_value is None:
                        for chunk in range(self.ld_chunks(slot)):
                            self.emit_carry_registers(
                                self.row_array(source, chunk), slot, chunk
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
                        self.emit_carry_registers(name, slot, chunk)
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
                    self.emit_carry_registers(name, slot, chunk)
            if any(carry.slot not in self.resident for carry in carries):
                self.emit(f"{_PRIMS}.tcgen05_wait(kind={_PRIMS}.Tcgen05Wait.STORE)")
            self.slot_regs = {}
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")

    def emit_carry_alias(self, source: int, target: int) -> None:
        """The point where an elided carry copy would have run: from here
        the output slot's columns hold the carry (nothing is emitted; the
        hardware-order checker of the tests marks it)."""

    def emit_carry_copies(self, loop: Loop) -> None:  # type: ignore[override]
        """Carry outputs computed in another slot (``Apow = dot(Apow, Apow)``
        lands in a fresh slot) move into the carry's slot for the next
        iteration -- unless the layout gave the output the carry's columns
        (``region_tmem_layout``), where the value already sits."""
        copies = []
        for other, carry, source in copied_carries(self.program):
            if other.index != loop.index:
                continue
            target = carry.slot
            if self.slot(source).column == self.slot(target).column:
                assert (self.slot(source).rows, self.slot(source).cols) == (
                    self.slot(target).rows,
                    self.slot(target).cols,
                )
                self.emit_carry_alias(source, target)
                continue
            copies.append((source, target))
        if not copies:
            return
        self.wait_latest()
        with self.block("if md_epi:"):
            self.slot_regs = {}
            loaded: set[int] = set(self.resident)
            for source, target in copies:
                assert source not in self.resident and target not in self.resident
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

    def pending_root_raw(self) -> list[Tcgen05Tile]:
        """The root raw tiles whose conversion has not run yet.  The layout
        overlays them on the loop frames (``region_items``: segment 0, dead
        once the loops run), so every one of them converts before the first
        top-level loop writes its frame, whoever reads the converted tile
        (a root dot behind the loop included).  A raw tile landing on its
        own tile's bytes (``raw_overlays_tile``) overlays nothing and
        converts at its first consumer."""
        return [
            tile
            for tile in self.tc.staged_tiles.get(-1, ())
            if tile.raw_tma is not None
            and tile.raw_offset != tile.offset
            and tile.key not in self.settled
        ]

    def ring_under_raw(self, loop: Loop, tiles: Sequence[Tcgen05Tile]) -> bool:
        """Whether a raw tile of ``tiles`` overlaps the ring slots of
        ``loop``: the prologue TMA would land on it."""
        tc = self.tc
        index = loop.index
        base = tc.ring_offsets.get(index, 0)
        extent = tc.ring_bytes.get(index, 0) * tc.stages
        for tile in tiles:
            assert tile.raw_tma is not None
            raw_bytes = tile.raw_tma.bytes
            if tile.raw_offset < base + extent and base < tile.raw_offset + raw_bytes:
                return True
        return False

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
        pending = self.pending_root_raw()
        # The root raw tiles all convert ahead of the first top-level loop
        # (below); a nested loop never finds one pending (a conversion in a
        # loop body would re-run over the frame's bytes).
        assert region < 0 or not pending
        if ring and tc.ring_tx.get(index) and self.ring_under_raw(loop, pending):
            # A root raw tile overlaying this ring (its conversion is the
            # last read of it) settles ahead of the prologue issue.
            self.settle_keys({tile.key for tile in pending}, region)
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
        # The other root raw tiles (read by no dot of this loop: a root dot
        # behind it) overlay the frame this loop's body writes (its refills,
        # staged tiles, staging and exchange tiles): they convert here, with
        # the prologue in flight.
        self.settle_keys({tile.key for tile in self.pending_root_raw()}, region)
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
        # loop's reads of the overlapping frame.  The loop's last TMA store
        # groups finish reading their staging buffers first: the following
        # root segment's exchange tiles and staging overlay the frames.
        self.written = {key for key in self.written if key in self.root_written}
        self.emit_tma_store_wait(read=True)
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
        aux = tc.aux_tiles.get(index, ())
        if any(tile.ring for tile in aux):
            # The derived values' tile inputs landed in this iteration's
            # ring slot: every thread waits for it (warp 0 need not again).
            # A tile staged by an enclosing frame waits on that frame's
            # barrier in its prefetch.
            self.emit(f"{_RT}.mbar_wait(md_full{index} + md_st{index}, md_rd{index})")
            self.ring_waited.add(index)
        for step in derived:
            self.emit_derived_prefetch(step)
        if any(tile.rewritten for tile in aux):
            # Reads of an operand tile the in-place pass rewrites below:
            # every thread's reads complete before any thread's rewrite.
            self.emit(self.sync)
        if any(
            self.aux_tile(step, access) is None
            for step in derived
            for access in step.loads
        ):
            # The next iteration's rows into L2 (the register prefetch above
            # would otherwise pay the DRAM latency on the chain).
            offset = var if begin == "0" else f"({var} - {begin})"
            with self.nest(f"if {offset} + 1 < {end} - {begin}:"):
                self.emit_region_line_prefetch(
                    derived, {**self.values, index: f"({var} + 1)"}
                )
        # The refill of the slot the previous iteration's aux reads took
        # sits behind that iteration's batch barrier; a loop without a dot
        # batch (a load-only reduction loop) needs its own.
        dotless = not any(isinstance(step, Dot) for step in loop.steps)
        aux_sync = bool(aux) and dotless
        if dotless and self.tma_stores(
            [store for store in program.stores if store.loop == index]
        ):
            # No commit in this loop: the stage-in below rewrites the
            # buffer the TMA store of two iterations ago reads, so the store
            # thread lets all but the previous iteration's group finish
            # before any thread writes.
            self.emit_tma_store_wait(read=True, pending=1)
            self.emit(self.sync)
        ring = index in self.ring_loops
        tiles = tc.ring_tiles.get(index, ())
        if ring:
            # The previous iteration's batches read the slot refilled below
            # (and the ring tiles every pass rewrites).
            self.wait_latest()
            if tc.stages > 1 and tc.ring_tx.get(index):
                # The refill goes out ahead of this iteration's pass.  The
                # slot it lands in was read by the previous iteration's
                # batches (waited above) and its aux reads (behind that
                # iteration's batch barrier, or the sync below), never by
                # this iteration's pass (its own slot), so the TMA has the
                # pass, the batch barrier and the batch to land in: the
                # pass's wait on its own slot was gated dqkw's hottest stall
                # (14 % of the warp samples) with the refill behind the pass.
                ahead = tc.stages - 1
                offset = var if begin == "0" else f"({var} - {begin})"
                if aux_sync:
                    self.emit(self.sync)
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
        self.emit_segment(loop.steps, loop)
        if self.drains:
            self.emit(self.sync)
            self.flush_drains()
        self.emit_carry_copies(loop)
        if ring and tc.stages == 1 and tc.ring_tx.get(index):
            # Refill after every batch read the slot.
            self.wait_latest()
            if aux_sync:
                self.emit(self.sync)
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
                if self.aux_tile(step, access) is not None:
                    continue
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

    def emit_hoisted_exchanges(self, loop: Loop) -> None:
        """Ahead of top-level loop ``loop``: the exchange tiles of the
        root-defined fragments its dots (at any depth) read, written once
        (``hoisted_exchange_loop``); the loop's blocks find them written.
        The segment's TMA stores finish reading their staging buffers (a
        root buffer overlays the frames) and its dots have committed (the
        source slots are final, the segment's own exchange tiles read)."""
        tc, program = self.tc, self.program
        tiles: list[Tcgen05Tile] = []
        for dot in program.dots:
            if loop.index not in program.enclosing_loops(dot.loop):
                continue
            for operand in (dot.a, dot.b):
                if hoisted_exchange_loop(program, dot, operand) != loop.index:
                    continue
                tile = tc.tile(dot, operand)
                if tile.key not in self.written and tile not in tiles:
                    tiles.append(tile)
        if not tiles:
            return
        if tc.store_tma:
            self.emit_tma_store_wait(read=True)
            self.emit(self.sync)
        self.wait_latest()
        with self.block("if md_epi:"):
            self.slot_regs = {}
            loaded: set[int] = set(self.resident)
            for tile in tiles:
                source = cast("int", tile.source_slot)
                if source not in loaded:
                    self.emit_ld(source)
                    loaded.add(source)
                self.emit_exchange(tile)
                self.written.add(tile.key)
                self.root_written.add(tile.key)
            self.slot_regs = {}
        self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
        self.emit(f"{_PRIMS}.tcgen05_fence({_PRIMS}.Tcgen05Fence.BEFORE_THREAD_SYNC)")
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
                if self.tc.store_tma:
                    # The staging buffers may overlay the loop's frames (a
                    # root buffer; a sibling loop's): every TMA store group
                    # issued so far -- behind a batch's commit, or just now
                    # -- finishes reading before the prologue TMA lands.
                    self.emit_tma_store_wait(read=True)
                    self.emit(self.sync)
                if loop is None:
                    self.emit_hoisted_exchanges(step)
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
        for slot in sorted(self.resident):
            # A register carry's row (every thread defines the scalars so the
            # epilogue warps' assignments keep their type across the loops).
            for _chunk, _j, scalar in self.resident_names(slot):
                self.emit(f"{scalar} = cutlass.Float32(0.0)")
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
        # The TMA stores complete before the kernel ends.
        self.emit_tma_store_wait(read=False)
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
    "aux_shared_operand",
    "aux_tile_loads",
    "hoisted_exchange_loop",
    "plan_tcgen05_regions",
    "region_aux_skip",
    "region_config_reason",
    "region_facts",
    "region_items",
    "region_layout",
    "region_live_slots",
    "region_program_reason",
    "region_smem_bytes",
    "region_tmem_columns",
    "register_carries",
    "register_carry_facts",
    "root_segments",
]
