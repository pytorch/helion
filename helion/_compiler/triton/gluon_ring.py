# ruff: noqa: ANN001, ANN202
"""The gluon backend's shared-memory ring (``num_stages`` >= 2): the kernel
warp-specializes slices of Helion's body: a TMA producer per gating group, an MMA
partition for the dots ``range_warp_specializes`` asks for, and the body.  The
body stays the single source of what is loaded, when, and into which slot.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import functools
import inspect
import logging
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.hopper import tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.runtime.jit import JITFunction

from ..._compat import torch_dtype_to_tl
from ...autotuner.config_spec import ring_depth
from ...language.matmul_ops import dot_scaled
from ..ast_extension import clone_ast
from ..ast_extension import create
from ..ast_extension import create_arg
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import ast_rename
from ..ast_read_writes import dead_assignment_elimination
from ..autotuner_heuristics.triton import TritonB200FormulaMatmulHeuristic as _B200
from ..compile_environment import CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
from ..compile_environment import CompileEnvironment
from ..cross_loop_codegen import TaggedPeerFallback
from ..cross_loop_codegen import _access_allocations
from ..cross_loop_codegen import _all_names
from ..cross_loop_codegen import _allocation
from ..cross_loop_codegen import _argument_allocations
from ..cross_loop_codegen import _effect_kind
from ..cross_loop_codegen import _is_load
from ..cross_loop_codegen import _is_tl_call
from ..cross_loop_codegen import _may_write
from ..cross_loop_codegen import _named_allocations
from ..cross_loop_codegen import _stored_names
from ..cross_loop_codegen import _written_allocations
from ..cross_loop_codegen import peer_state
from ..device_function import RingFallback
from ..emitter_protocols import RetryPolicy
from ..host_function import HostFunction
from ..indexing_strategy import BlockedSubscriptIndexing
from ..indexing_strategy import TensorDescriptorIndexingStrategy
from ..indexing_strategy import in_warp_specialized_loop
from ..tile_dependency import TILE_ACCESS_META
from . import gluon_ops
from .distributed_ops import tagged_access
from .distributed_ops import tagged_tile
from .gluon_ops import SharedValue
from .gluon_ops import reinterpret

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator

    from ...runtime.config import Config
    from ..device_function import DeviceBodyContext
    from ..device_function import DeviceFunction
    from ..device_function import SyncRole
    from ..device_function import TensorDescriptorArg
    from ..generate_ast import GenerateAST
    from ..indexing_strategy import IndexingStrategy
    from ..inductor_lowering import CodegenState
    from ..inductor_lowering import GraphInterpreter

log = logging.getLogger(__name__)

# The generated code's name for this module (``GluonBackend.library_imports``).
ALIAS = "_helion_ring"
_MODULES = (ALIAS, gluon_ops.ALIAS)


# ---- the ring protocol -------------------------------------------------------
# A channel (the steps one partition issues) has slots, ``full`` / ``empty``
# barriers and a step counter.  A slot holds one step's operands of every gating
# group; each group's producer arrives on every step (so parity waits cannot
# alias), and the step's MMA (or register read) frees the slot.


def ring_descriptor(
    base: torch.Tensor,
    shape: list[int],
    strides: list[int],
    block_shape: list[int],
    swizzle: bool = True,
) -> TensorDescriptor:
    """Host descriptor of a ring operand in Triton's default shared layout; an
    unswizzled slot (scales for ``tcgen05.cp``) keeps its whole contiguous rows."""
    layout = _ring_layout(tuple(block_shape), base.dtype, swizzle)
    return TensorDescriptor(base, shape, strides, block_shape, layout)


@functools.cache
def _ring_layout(
    block_shape: tuple[int, ...], dtype: torch.dtype, swizzle: bool
) -> gl.NVMMASharedLayout:  # per launch otherwise (host overhead)
    if swizzle:
        return gl.NVMMASharedLayout.get_default_for(  # pyrefly: ignore [bad-return]
            list(block_shape), torch_dtype_to_tl(dtype)
        )
    return gl.NVMMASharedLayout(0, dtype.itemsize * 8, len(block_shape))


@gluon.jit
def ring_barriers(S: gl.constexpr, COUNT: gl.constexpr = 1):  # pyrefly: ignore [bad-function-definition]
    bars = gl.allocate_shared_memory(gl.int64, [S, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(S):
        mbarrier.init(bars.index(i), count=COUNT)
    return bars


@gluon.jit
def ring_buffer(desc, S: gl.constexpr):
    SHAPE: gl.constexpr = [S] + desc.block_type.shape  # noqa: RUF005
    return gl.allocate_shared_memory(desc.dtype, SHAPE, desc.layout)


@gluon.jit
def ring_view(lane, desc, BOXES: gl.constexpr):
    """A narrower operand's view of a lane: ``BOXES`` of its boxes."""
    SHAPE: gl.constexpr = [BOXES] + desc.block_type.shape  # noqa: RUF005
    return reinterpret(lane, desc.dtype, SHAPE, desc.layout)  # pyrefly: ignore [bad-argument-count]


@gluon.jit
def ring_index(ring, s, S: gl.constexpr, BOX: gl.constexpr):
    """Slot ``s`` of a lane or of a view of it: the view's box ``BOX`` there."""
    return ring.index(s * (ring.shape[0] // S) + BOX)


@gluon.jit
def ring_acquire(
    ring,
    full,
    empty,
    step,
    S: gl.constexpr,
    desc,
    coords,
    FIRST: gl.constexpr,
    NBYTES: gl.constexpr,
    BOX: gl.constexpr,
    TAGGED=None,
    WAIT: gl.constexpr = True,  # pyrefly: ignore [bad-function-definition]
):
    """Consumer side of a ring load (the producer's ``ring_put`` arguments): the
    step's filled slot; only a partition's first acquire of a step waits."""
    s = step % S
    if WAIT:
        mbarrier.wait(full.index(s), (step // S) & 1)
    return ring_index(ring, s, S, BOX)  # pyrefly: ignore [bad-argument-count]


@gluon.jit
def ring_put(
    ring,
    full,
    empty,
    step,
    S: gl.constexpr,
    desc,
    coords,
    FIRST: gl.constexpr,
    NBYTES: gl.constexpr,
    BOX: gl.constexpr,
):
    """Producer side: ``FIRST`` (the group's first operand of the step) waits for
    the slot and arrives on the full barrier expecting the group's ``NBYTES``."""
    s = step % S
    if FIRST:
        mbarrier.wait(empty.index(s), ((step // S) & 1) ^ 1)
        mbarrier.expect(full.index(s), NBYTES)
    slot = ring_index(ring, s, S, BOX)  # pyrefly: ignore [bad-argument-count]
    tma.async_load(desc, coords, full.index(s), slot)  # pyrefly: ignore [missing-attribute]


@gluon.constexpr_function
def _word_layout(rows: int, words: int, num_warps: int) -> gl.BlockedLayout:
    """Registers of a [rows, words] tile of tagged words: pairs along a row."""
    # T6 (``tagged_tile``): a power-of-two count of at least two words.
    assert words >= 2 and words & (words - 1) == 0
    cols = min(32, words // 2)
    return gl.BlockedLayout([1, 2], [32 // cols, cols], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _word_slots(layout: gl.NVMMASharedLayout) -> gl.NVMMASharedLayout:
    """A slot's layout viewed as rows of 4-byte words."""
    assert not layout.transposed and not layout.fp4_padded
    return gl.NVMMASharedLayout(
        swizzle_byte_width=layout.swizzle_byte_width, element_bitwidth=32, rank=2
    )


# Tagged words' payloads once their tags are the epoch: a thread's pair of words
# re-loads (both in flight) until both are (operands: payloads, words, the
# mirror's words, word offsets, epochs).
_TAGGED_POLL = gl.constexpr(
    "{ .reg .pred p; .reg .b64 v0, v1, a; .reg .b32 l0, h0, l1, h1; "
    "mov.b64 v0, $2; mov.b64 v1, $3; "
    "TAGGED_POLL: mov.b64 {l0, h0}, v0; mov.b64 {l1, h1}, v1; "
    "setp.ne.u32 p, h0, $8; setp.ne.or.u32 p, h1, $8, p; @!p bra TAGGED_DONE; "
    "mad.wide.u32 a, $6, 8, $4; ld.relaxed.sys.global.b64 v0, [a]; "
    "mad.wide.u32 a, $7, 8, $5; ld.relaxed.sys.global.b64 v1, [a]; bra TAGGED_POLL; "
    "TAGGED_DONE: mov.b32 $0, l0; mov.b32 $1, l1; }"
)


@gluon.jit
def _tagged_payloads(loaded, words, offsets, epoch):
    """The payloads of tagged words, re-loading stale ones until each tag is ``epoch``."""
    zero = gl.zeros_like(offsets)
    return gl.inline_asm_elementwise(
        _TAGGED_POLL,  # pyrefly: ignore [bad-argument-type]
        "=r,=r,l,l,l,l,r,r,r,r",
        [
            loaded,
            words + zero,
            offsets.to(gl.uint32),
            epoch.to(gl.uint32) + zero.to(gl.uint32),  # pyrefly: ignore [missing-attribute]
        ],
        gl.uint32,
        False,
        2,
    )


@gluon.jit
def _tagged_tile(
    word0,
    ROWS: gl.constexpr,
    ROW_WORDS: gl.constexpr,
    WORDS: gl.constexpr,
    PRESENCE: gl.constexpr,
    CHUNK: gl.constexpr,
    CHUNKS: gl.constexpr,
):
    """A tagged tile's word offsets, row indices and row presence word offsets."""
    layout: gl.constexpr = _word_layout(ROWS, WORDS, gl.num_warps())  # pyrefly: ignore [bad-assignment, bad-argument-type]
    rows = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))  # pyrefly: ignore [bad-argument-type]
    cols = gl.arange(0, WORDS, layout=gl.SliceLayout(0, layout))  # pyrefly: ignore [bad-argument-type]
    offsets = word0 + rows[:, None] * ROW_WORDS + cols[None, :]
    col = word0 + cols[None, :]
    index = (
        PRESENCE
        + col // (CHUNK * ROWS * CHUNKS) * CHUNKS
        + col % (CHUNK * CHUNKS) // CHUNK
    )
    return offsets, rows[:, None], index


@gluon.jit
def tagged_load(
    words,
    word0,
    epoch,
    ROWS: gl.constexpr,
    ROW_WORDS: gl.constexpr,
    WORDS: gl.constexpr,
    PRESENCE: gl.constexpr,
    CHUNK: gl.constexpr,
    CHUNKS: gl.constexpr,
):
    """Issue the loads of a tagged operand's step (``ring_put_tagged`` completes it):
    its tile's words and, with ``PRESENCE``, its columns' row presence words."""
    offsets, rows, index = _tagged_tile(
        word0, ROWS, ROW_WORDS, WORDS, PRESENCE, CHUNK, CHUNKS  # pyrefly: ignore [bad-argument-count]
    )  # fmt: skip
    loaded = gl.load(words + offsets, volatile=True)
    written = loaded
    if PRESENCE >= 0:
        written = gl.load(words + index, volatile=True)
    return loaded, written, words, word0


@gluon.jit
def tagged_none(
    words,
    ROWS: gl.constexpr,
    ROW_WORDS: gl.constexpr,
    WORDS: gl.constexpr,
    PRESENCE: gl.constexpr,
    CHUNK: gl.constexpr,
    CHUNKS: gl.constexpr,
):
    """``tagged_load``'s result for a step another poller fills."""
    word0 = gl.full([], 0, gl.int64)
    offsets, _, index = _tagged_tile(
        word0, ROWS, ROW_WORDS, WORDS, PRESENCE, CHUNK, CHUNKS  # pyrefly: ignore [bad-argument-count]
    )  # fmt: skip
    loaded = gl.zeros_like(offsets).to(gl.uint64)  # pyrefly: ignore [bad-argument-type, missing-attribute]
    written = loaded
    if PRESENCE >= 0:
        written = gl.zeros_like(index).to(gl.uint64)  # pyrefly: ignore [bad-argument-type, missing-attribute]
    return loaded, written, words, word0


@gluon.jit
def ring_put_tagged(
    ring,
    full,
    empty,
    step,
    S: gl.constexpr,
    issued,
    epoch,
    FIRST: gl.constexpr,
    LAST: gl.constexpr,
    BOX: gl.constexpr,
    ROWS: gl.constexpr,
    ROW_WORDS: gl.constexpr,
    WORDS: gl.constexpr,
    PRESENCE: gl.constexpr,
    CHUNK: gl.constexpr,
    CHUNKS: gl.constexpr,
):
    """Producer side of a tagged operand in place of a TMA: the words
    ``tagged_load`` issued, re-polled until their tag is ``epoch``, stored to the
    slot as payloads (with ``PRESENCE``: rows nobody wrote store zeros).  ``LAST``
    fences the stores and arrives on the step's full barrier."""
    loaded, written, words, word0 = issued
    offsets, rows, index = _tagged_tile(
        word0, ROWS, ROW_WORDS, WORDS, PRESENCE, CHUNK, CHUNKS  # pyrefly: ignore [bad-argument-count]
    )  # fmt: skip
    fresh = epoch.to(gl.uint32).to(gl.uint64) << 32
    if PRESENCE >= 0:
        bits = _tagged_payloads(written, words, index, epoch)  # pyrefly: ignore [bad-argument-count]
        loaded = gl.where(((bits >> rows.to(gl.uint32)) & 1) != 0, loaded, fresh)  # pyrefly: ignore [missing-attribute]
    payloads = _tagged_payloads(loaded, words, offsets, epoch)  # pyrefly: ignore [bad-argument-count]
    s = step % S
    if FIRST:
        mbarrier.wait(empty.index(s), ((step // S) & 1) ^ 1)
    slot = ring_index(ring, s, S, BOX)  # pyrefly: ignore [bad-argument-count]
    WORD_SLOTS: gl.constexpr = _word_slots(ring.layout)  # pyrefly: ignore [bad-assignment]
    reinterpret(slot, gl.uint32, [ROWS, WORDS], WORD_SLOTS).store(payloads)  # pyrefly: ignore [bad-argument-count, missing-attribute]
    if LAST:
        fence_async_shared()
        mbarrier.arrive(full.index(s))


@gluon.jit
def _pace_forward(
    full, pace, step, S: gl.constexpr, paced, run, K: gl.constexpr, ALL: gl.constexpr
):
    """A paced run's arrivals on its steps' full barriers once their bytes landed:
    the step K back, or (``ALL``: the run ends) every step still owed."""
    if ALL:
        for j in gl.static_range(K):
            if j < run:
                n = paced - 1 - j
                mbarrier.wait(pace.index(n % K), (n // K) & 1)
                mbarrier.arrive(full.index((step - 1 - j) % S))
    else:
        if run >= K:
            n = paced - K
            mbarrier.wait(pace.index(n % K), (n // K) & 1)
            mbarrier.arrive(full.index((step - K) % S))


@gluon.jit
def ring_put_paced(
    ring,
    full,
    pace,
    empty,
    step,
    S: gl.constexpr,
    desc,
    coords,
    FIRST: gl.constexpr,
    NBYTES: gl.constexpr,
    BOX: gl.constexpr,
    paced,
    run,
    fill,
    K: gl.constexpr,
):
    """``ring_put`` of the weight producer's paced first fill beside gated producers
    (``fill``): a step of the run lands on a pace barrier and issues once the step K
    back landed, forwarding that arrival to its full barrier.  Returns
    ``(paced, run)``."""
    s = step % S
    lead = (step < S - 1) & fill
    if FIRST:
        if lead:
            _pace_forward(full, pace, step, S, paced, run, K, False)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
        else:
            _pace_forward(full, pace, step, S, paced, run, K, True)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
        mbarrier.wait(empty.index(s), ((step // S) & 1) ^ 1)
    slot = ring_index(ring, s, S, BOX)  # pyrefly: ignore [bad-argument-count]
    if lead:
        n = paced if FIRST else paced - 1
        if FIRST:
            mbarrier.expect(pace.index(n % K), NBYTES)
        tma.async_load(desc, coords, pace.index(n % K), slot)  # pyrefly: ignore [missing-attribute]
    else:
        if FIRST:
            mbarrier.expect(full.index(s), NBYTES)
        tma.async_load(desc, coords, full.index(s), slot)  # pyrefly: ignore [missing-attribute]
    if FIRST:
        if lead:
            paced = paced + 1
            run = run + 1
        else:
            run = run * 0
    return paced, run


@gluon.jit
def ring_pace_flush(full, pace, step, S: gl.constexpr, paced, run, K: gl.constexpr):
    """A paced group's owed arrivals (before a skip or kept wait, or at the end)."""
    _pace_forward(full, pace, step, S, paced, run, K, True)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
    return run * 0


@gluon.jit
def ring_skip(full, empty, step, S: gl.constexpr, NBYTES: gl.constexpr):
    """A producer's step it fills nothing of: it waits for the slot and arrives on
    the full barrier (a producer a phase behind would alias a parity wait)."""
    s = step % S
    mbarrier.wait(empty.index(s), ((step // S) & 1) ^ 1)
    mbarrier.arrive(full.index(s))


@gluon.jit
def ring_read(slot, empty, step, S: gl.constexpr, FREE: gl.constexpr = True):  # pyrefly: ignore [bad-function-definition]
    """A register read of the step's slot, which then frees it (``FREE``; else
    the step's dot does)."""
    value = slot.load(gluon_ops.vector_layout(slot.shape, slot.dtype, gl.num_warps()))
    fence_async_shared()  # the read ahead of the producer's next TMA write
    if FREE:
        mbarrier.arrive(empty.index(step % S))  # (an arrive syncs its warps first)
    return value


# The task channel: one partition claims the tickets, the others take each.


@gluon.jit
def task_buffer(S: gl.constexpr):
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])  # pyrefly: ignore [bad-assignment]
    return gl.allocate_shared_memory(gl.uint64, [S, 1], layout)


@gluon.jit
def task_take(tasks, full, empty, step, S: gl.constexpr):
    """A consumer's ticket of the step, which frees the slot once read."""
    mbarrier.wait(full.index(step % S), (step // S) & 1)
    return gl.max(ring_read(tasks.index(step % S), empty, step, S), axis=0)  # pyrefly: ignore [bad-argument-count]


@gluon.jit
def task_put(tasks, full, empty, step, S: gl.constexpr, ticket):
    """The claiming partition's side: publish the step's ticket, then wait for the
    next step's slot (every consumer took the ticket ``S`` steps before it)."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])  # pyrefly: ignore [bad-assignment]
    tasks.index(step % S).store(gl.full([1], ticket, ticket.dtype, layout))
    mbarrier.arrive(full.index(step % S))
    after = step + 1
    mbarrier.wait(empty.index(after % S), ((after // S) & 1) ^ 1)
    return ticket


# ---- the lowering ------------------------------------------------------------

_DEPTH = "_helion_ring_S"
# Gating groups, one producer each: weights stream ungated; activations earlier
# roots write stream after the body's waits.
_UNGATED, _GATED = 0, 1
_ACC_DEPTH = 2
# Gated producers polling tagged words (round-trip bound): each fills every
# _TAGGED_POLLERS-th tagged step.
_TAGGED_POLLERS = 2
_TMEM_COLUMNS = cast("int", _B200.TMEM_COLUMN_BUDGET)


def _tmem_columns(tiles: list[tuple[int, int]]) -> int:
    """Tensor-memory columns of allocations live together (``(rows, columns)``
    each), as Triton's allocator packs them: two 64-row ones of a width share
    its columns (the lanes' halves)."""
    halves = collections.Counter(c for r, c in tiles if r <= 64)
    return sum(c * -(-r // 128) for r, c in tiles if r > 64) + sum(
        -(-n // 2) * c for c, n in halves.items()
    )


# Calls of a dot's MMA side: on its channels, and on its accumulator ring.
_ACC_CALLS = frozenset(["acc_claim", "acc_commit"])
_MMA_CALLS = frozenset(["ring_acquire", "tmem_dot", "tmem_dot_scaled"]) | _ACC_CALLS


# Worker register budgets: default, floors (producers, MMA side); one warp's: the minimum.
# setmaxnreg's per-thread register range.
MIN_REGISTERS, MAX_REGISTERS = 24, 256
_WORKER_REGS = 88
_WORKER_FLOOR = (40, 48)


def _cta_warps(body_warps: int, workers: list[tuple[int, bool]]) -> int:
    """A CTA's warps: the body's and the workers' (Triton's AllocateWarpGroups
    pads them to warpgroups)."""
    return body_warps + 4 * -(-sum(w for w, _ in workers) // 4)


def _maxnreg(body_warps: int, workers: list[tuple[int, bool]], ctas: int = 1) -> int:
    """Triton's AllocateWarpGroups registers a thread: a CTA's share of the
    register file with ``ctas`` resident a SM (a persistent grid's)."""
    return 65536 // ctas // _cta_warps(body_warps, workers) // 32 // 8 * 8


def _body_registers(
    body_warps: int,
    workers: list[tuple[int, bool]],
    registers: list[int],
    ctas: int = 1,
) -> int:
    """The registers Triton's AllocateWarpGroups leaves the body beside ``workers``
    ((warps, MMA side storing tensors)) of ``registers``: warpgroups at their
    widest budget, the body the rest."""
    warps = sum(w for w, _ in workers)
    groups = -(-warps // 4)
    padded = [(w, r) for (w, _), r in zip(workers, registers, strict=True)]
    pad = 4 * groups - warps
    while pad:
        width = 1 << (pad.bit_length() - 1)
        padded.append((width, 16))
        pad -= width
    maxnreg = _maxnreg(body_warps, workers, ctas)
    start, budgets = body_warps, []
    for w, r in sorted(padded, key=lambda p: -p[0]):
        if start % 4 == 0:
            budgets.append([0, 0])
        budgets[-1] = [
            max(budgets[-1][0], -(-max(r, MIN_REGISTERS) // 8) * 8),
            budgets[-1][1] + w,
        ]
        start += w
    left = maxnreg * body_warps + sum((maxnreg - r) * w for r, w in budgets)
    return min(left // body_warps // 8 * 8, MAX_REGISTERS)


def _worker_registers(
    body_warps: int, workers: list[tuple[int, bool]], ctas: int = 1
) -> list[int]:
    """The rule's worker budgets: one-warp workers the scalar one, the others the
    (producer, MMA side) defaults, or the floors when those leave the body more."""

    def budgets(regs: tuple[int, int]) -> list[int]:
        return [MIN_REGISTERS if w == 1 else regs[mma] for w, mma in workers]

    # ``max`` keeps the first on a tie: the defaults.
    return max(
        [budgets((_WORKER_REGS, _WORKER_REGS)), budgets(_WORKER_FLOOR)],
        key=lambda regs: _body_registers(body_warps, workers, regs, ctas),
    )


def _poller_registers(
    body_warps: int,
    workers: list[tuple[int, bool]],
    registers: list[int],
    polling: list[int],
    added: list[bool],
    ctas: int = 1,
) -> list[int]:
    """Worker budgets once the tagged pollers (``added``: the extra ones) take what
    the body leaves unused; raises ``TaggedPeerFallback`` if they cannot."""
    body = _body_registers(body_warps, workers, registers, ctas)
    while all(registers[i] + 8 <= MAX_REGISTERS for i in polling):
        more = [r + 8 * (i in polling) for i, r in enumerate(registers)]
        if _body_registers(body_warps, workers, more, ctas) < body:
            break
        registers = more
    plain = [worker for worker, extra in zip(workers, added, strict=True) if not extra]
    plain_body = _body_registers(
        body_warps, plain, _worker_registers(body_warps, plain, ctas), ctas
    )
    held = min(registers[i] for i in polling)
    if body < plain_body or held < _WORKER_REGS:
        raise TaggedPeerFallback(
            f"tagged pollers would take body registers ({body} < {plain_body}) "
            f"or hold their words in {held}"
        )
    return registers


def _channel(name: str, channel: int) -> str:
    """A channel's ``full`` / ``empty`` / ``step`` (or ``pace`` / ``paced`` / ``run``)."""
    return f"_helion_ring_{name}{channel}"


def _beside(full: ast.expr, name: str) -> ast.Name:
    """The ``_channel`` ``name`` (``pace``, ``paced``, ``run``) beside ``full``."""
    assert isinstance(full, ast.Name) and full.id.startswith(_channel("full", 0)[:-1])
    return create(ast.Name, id=full.id.replace("_full", f"_{name}"), ctx=ast.Load())


def _link(dot: int, group: int, name: str) -> str:
    """The body's name for a link of ``dot``'s ``group`` operands (resolved by ``emit``)."""
    return f"_helion_ring_{name}_{dot}_{group}"


def _links(dot: int, group: int, *names: str) -> str:
    return ", ".join(_link(dot, group, name) for name in names)


# The task channel: the weight producer claims each task once it issued the
# previous one's loads, else a claimer partition claims one task ahead.
_TASKS, _TASK_FULL, _TASK_EMPTY, _TASK_STEP = (
    f"_helion_task{k}" for k in ("s", "_full", "_empty", "_step")
)
_TASK_DEPTH = "_helion_task_S"


def _acc(index: int) -> tuple[str, ...]:
    """Buffers, full and empty barriers and tile counter of accumulator ring."""
    return tuple(f"_helion_acc{k}{index}" for k in ("", "_full", "_empty", "_tile"))


@dataclasses.dataclass
class _Site:
    dot: int
    group: int
    descriptor: TensorDescriptorArg
    # The body's name for the site's ring buffer (resolved by ``emit``).
    name: str
    nbytes: int
    slot_type: tuple[torch.dtype, tuple[int, ...], bool]
    # A tagged operand's ``ring_put_tagged`` geometry (ROWS, ROW_WORDS, WORDS,
    # PRESENCE, CHUNK, CHUNKS), else None: a TMA copy.
    tagged: tuple[int, ...] | None = None

    @property
    def tile(self) -> tuple[object, ...]:
        """The slot's dtype and its box's last two dims."""
        return (self.slot_type[0], *self.slot_type[1][-2:])


class RingLowering:
    """Per-kernel ring state: the sites found while lowering the body, and the ring
    kernel built around the body."""

    def __init__(self, device_function: DeviceFunction) -> None:
        self.device_function = device_function
        self.depth = ring_depth(device_function.config.config, "gluon")
        # By load node, or (load, dot) for a deferred operand's slot of a dot.
        self.sites: dict[object, _Site] = {}
        self.dots: dict[torch.fx.Node, int] = {}
        # Dots with a slot acquired that have not issued yet (their step is
        # open), in acquire order.
        self.open: dict[torch.fx.Node, None] = {}
        # Per ring dot: whether it issues from the MMA partition.
        self.mma: dict[torch.fx.Node, bool] = {}
        # (shape, ordinal in its task) -> accumulator ring; each dot's root.
        self.acc_rings: dict[tuple[tuple[int, ...], int], int] = {}
        self.dot_roots: dict[torch.fx.Node, int | None] = {}
        # The MMA partition stores scales from registers (4-warp layouts).
        self.mma_tensors = False
        # Tensor memory of the dots outside the ring, (rows, columns) each.
        self.tmem: list[tuple[int, int]] = []
        # An operand of several dots (each acquires its own slot), by name.
        self.deferred: dict[
            str, tuple[torch.fx.Node, torch.Tensor, BlockedSubscriptIndexing, int, bool]
        ] = {}

    # -- the body: ring sites and their consumers

    def codegen_load(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        strategy: IndexingStrategy,
    ) -> SharedValue | ast.AST | None:
        """A ring acquire in place of an eligible load, else None."""
        node = state.fx_node
        if (
            node is None
            or extra_mask is not None
            or not isinstance(strategy, TensorDescriptorIndexingStrategy)
        ):
            return None
        if (uses := gluon_ops.dot_operands(node)) is None:
            return None
        if uses and (
            # A tf32 MMA reads K-major operands only: a slot keeps the box's order.
            tensor.element_size() > 2
            or not all(
                gluon_ops.tmem_dot_supported(self.device_function, d)
                and not gluon_ops.padded_k(d)
                for d, _ in uses
            )
        ):
            log.info("the ring: %s feeds a register dot: a register read", node.name)
            uses = []
        if (group := self._group(node, tensor)) is None or not strategy.is_supported(
            state, tensor, subscript
        ):
            return None
        indexing = BlockedSubscriptIndexing.create(state, tensor, subscript)
        if indexing.broadcasts_tile and not indexing.pad_overshoot():
            # A size-1 dimension broadcast along a wider loop: no slot holds it.
            return None
        if not uses:
            # Inside an open step, a register operand of its dot joins the step
            # (the dot frees it); another load stays plain.
            reached = gluon_ops.reaches(node)
            joined = next((d for d in self.open if d in reached), None)
            if self.open and joined is None:
                log.info("the ring: %s is read inside an open step", node.name)
                return None
            return self._register_read(
                state,
                node,
                tensor,
                indexing,
                group,
                self._tagged(state, tensor, indexing, True),
                joined,
            )
        block_shape = gluon_ops.concrete(indexing.block_shape)
        # A scale tcgen05.cp copies takes an unswizzled slot (as Triton's): whole
        # rows of a 16-byte multiple, at most one TMA box row (256 elements).
        swizzles = {not gluon_ops.copied_scale(d, p) for d, p in uses}
        if len(swizzles) > 1 or (
            swizzles == {False}
            and (
                block_shape[-1] % (16 // tensor.element_size())
                or block_shape[-1] > CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
            )
        ):
            return None
        (swizzle,) = swizzles
        tagged = self._tagged(state, tensor, indexing, swizzle)
        # An operand of several dots, or loaded ahead of another dot than its
        # own, takes its slot at its dot: a site's step is its dot's.
        if len(uses) > 1 or _dot_between(node, uses[0][0]):
            if tagged is not None:
                return None
            slot = self.device_function.new_var("ring_deferred")
            self.deferred[slot] = (node, tensor, indexing, group, swizzle)
            descriptor = self._descriptor(tensor, indexing, swizzle)
        else:
            dot_index = self.dots.setdefault(uses[0][0], len(self.dots))
            self.open[uses[0][0]] = None
            slot, descriptor = self._acquire(
                state, node, tensor, indexing, dot_index, group, swizzle, tagged
            )
        value: ast.AST = expr_from_string(slot)
        if descriptor.permutation is not None:
            value = gluon_ops.permute(value, descriptor.inverse_permutation)
        if indexing.need_reshape(value):
            dims = state.tile_strategy.shape_dims(indexing.reshaped_size)
            value = gluon_ops.reshape(value, dims)
        return SharedValue(value)

    def _register_read(
        self,
        state: CodegenState,
        node: torch.fx.Node,
        tensor: torch.Tensor,
        indexing: BlockedSubscriptIndexing,
        group: int,
        tagged: tuple[str, tuple[int, ...]] | None,
        dot: torch.fx.Node | None = None,
    ) -> ast.AST:
        """A ring step read into registers in place of an eligible load no dot
        reads in place: its own step, or a site of open ``dot``'s."""
        step = (
            self.dots[dot]
            if dot is not None
            else self.dots.setdefault(node, len(self.dots))
        )
        if dot is None:
            self.mma[node] = False
        slot, descriptor = self._acquire(
            state, node, tensor, indexing, step, group, True, tagged
        )
        value = self.device_function.new_var("ring_value")
        links = _links(step, group, "empty", "step")
        state.add_statement(
            statement_from_string(
                f"{value} = {ALIAS}.ring_read({slot}, {links}, {_DEPTH}, {dot is None})"
            )
        )
        if dot is None:
            counter = _link(step, group, "step")
            state.add_statement(statement_from_string(f"{counter} = {counter} + 1"))
        result = value
        if descriptor.permutation is not None:
            result = f"tl.permute({result}, {list(descriptor.inverse_permutation)})"
        if indexing.need_reshape(expr_from_string(result)):
            dims = state.tile_strategy.shape_dims(indexing.reshaped_size)
            result = f"tl.reshape({result}, [{', '.join(dims)}])"
        return expr_from_string(result)

    def _tagged(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        indexing: BlockedSubscriptIndexing,
        swizzle: bool,
    ) -> tuple[str, tuple[int, ...]] | None:
        """A tagged load's producer arguments (words, first word, epoch) and
        geometry, else None; raises TaggedPeerFallback for a tile it cannot poll."""
        access = tagged_access(state)
        if access is None:
            return None
        peer = peer_state(self.device_function)
        assert peer is not None
        mirror = peer.mirrors[access.allocation_id]
        lanes = mirror.lanes
        strides = tagged_tile(state, tensor, access, lanes)
        descriptor = self._descriptor(tensor, indexing, swizzle)
        if descriptor.permutation is not None or not swizzle:
            raise TaggedPeerFallback(
                f"T6: {access.tensor_name}: a ring operand not in an unpermuted, "
                "swizzled slot"
            )
        box = gluon_ops.concrete(indexing.block_shape)
        last = tensor.ndim - 1
        rows = [dim for dim in range(last) if box[dim] != 1]
        presence, chunk, chunks = -1, 1, 1
        if mirror.presence is not None:
            # The tile spans the row block its presence words name.
            index = cast("torch.fx.Node", state.fx_node).args[1]
            assert isinstance(index, (list, tuple))
            if rows != [last - 1] or index[-2] != slice(None):
                raise TaggedPeerFallback(
                    f"{access.tensor_name}: a consumer tile of part of a row block"
                )
            presence, chunk, chunks = mirror.presence, mirror.chunk, mirror.chunks
        offsets = [f"tl.cast({offset}, tl.int64)" for offset in indexing.offsets]
        word0 = " + ".join(
            [
                *(f"{offsets[d]} * {strides[d] // lanes}" for d in range(last)),
                f"{offsets[last]} // {lanes}",
            ]
        )
        words = f"{peer.state} + ({peer.mirror(access.allocation_id)})"
        geometry = (
            box[rows[0]] if rows else 1,
            strides[rows[0]] // lanes if rows else 0,
            box[last] // lanes,
            presence,
            chunk,
            chunks,
        )
        return f"({words}, {word0}, {peer.epoch})", geometry

    def _descriptor(
        self, tensor: torch.Tensor, indexing: BlockedSubscriptIndexing, swizzle: bool
    ) -> TensorDescriptorArg:
        return self.device_function.tensor_descriptor_arg(
            tensor, indexing.block_shape, ring_swizzle=swizzle
        )

    def _acquire(
        self,
        state: CodegenState | GenerateAST,
        node: object,
        tensor: torch.Tensor,
        indexing: BlockedSubscriptIndexing,
        step: int,
        group: int,
        swizzle: bool,
        tagged: tuple[str, tuple[int, ...]] | None = None,
    ) -> tuple[str, TensorDescriptorArg]:
        """Emit ring site ``node``'s acquire; returns the slot and the descriptor."""
        block_shape = gluon_ops.concrete(indexing.block_shape)
        descriptor = self._descriptor(tensor, indexing, swizzle)
        site = self.sites.get(node)
        if site is None:
            nbytes = functools.reduce(operator.mul, block_shape, tensor.element_size())
            # The slot holds the descriptor's box, in its dimension order.
            order = descriptor.permutation or range(len(block_shape))
            slot_type = (tensor.dtype, tuple(block_shape[i] for i in order), swizzle)
            name = f"_helion_ring_site{len(self.sites)}"
            site = self.sites[node] = _Site(
                step,
                group,
                descriptor,
                name,
                nbytes,
                slot_type,
                None if tagged is None else tagged[1],
            )
        first = next(
            s for s in self.sites.values() if (s.dot, s.group) == (step, group)
        )
        offsets = indexing.offsets
        if descriptor.permutation is not None:
            offsets = [offsets[i] for i in descriptor.permutation]
        coords = "".join(f"tl.cast({offset}, tl.int32), " for offset in offsets)
        slot = self.device_function.new_var("ring_slot")
        state.add_statement(
            statement_from_string(
                f"{slot} = {ALIAS}.ring_acquire({site.name}, "
                f"{_links(step, group, 'full', 'empty', 'step')}, {_DEPTH}, "
                f"{descriptor.name}, ({coords}), {site is first}, "
                f"{_link(step, group, 'bytes')}, {site.name}_box"
                f"{'' if tagged is None else f', {tagged[0]}'})"
            )
        )
        return slot, descriptor

    def codegen_node(self, interp: GraphInterpreter, node: torch.fx.Node) -> object:
        """Views of a ring slot and its dot; NotImplemented for other nodes."""
        env = cast("dict[torch.fx.Node, object]", interp.env)
        if not any(
            isinstance(env.get(arg), SharedValue) for arg in node.all_input_nodes
        ):
            return NotImplemented
        if gluon_ops.is_dot(node):
            taken = self._take_deferred(interp, node)
            if taken:
                try:
                    return self.codegen_node(interp, node)
                finally:
                    env.update(taken)
        if node not in self.dots:
            return gluon_ops.codegen_view(interp, node)
        dot_index = self.dots[node]
        self.open.pop(node, None)
        groups = sorted({s.group for s in self.sites.values() if s.dot == dot_index})
        # A dot under a warp-specialized loop reading only ring slots into a
        # fresh accumulator issues from the MMA partition (else from the body).
        operands = [
            cast("torch.fx.Node", node.args[i]) for i in gluon_ops.operands(node.target)
        ]
        self.mma[node] = (
            in_warp_specialized_loop(
                self.device_function, cast("GenerateAST", interp.cg), requested=True
            )
            and all(isinstance(env.get(a), SharedValue) for a in operands)
            and gluon_ops.resident(self.device_function, node)
            and gluon_ops.fresh_accumulator(self.device_function, node)
        )
        ring = None
        if self.mma[node]:
            # One accumulator ring per dot of a task (root), so the MMA stays
            # _ACC_DEPTH tasks ahead of the body.
            shape = gluon_ops.concrete(node.meta["val"].shape)[-2:]
            root = cast("GenerateAST", interp.cg).current_root_graph_info
            self.dot_roots[node] = None if root is None else root.graph_id
            peers = [
                n
                for n, m in self.mma.items()
                if m and self.dot_roots.get(n) == self.dot_roots[node]
            ]
            shapes = [gluon_ops.concrete(n.meta["val"].shape)[-2:] for n in peers]
            key = (tuple(shape), shapes.count(shape))
            if key not in self.acc_rings and self._tmem_left(key[0]) < 0:
                # No tensor memory for another ring: share the shape's first.
                key = (tuple(shape), 1)
                if key not in self.acc_rings:
                    raise RingFallback(
                        "the ring: no tensor memory for an accumulator ring"
                    )
            ring = _acc(self.acc_rings.setdefault(key, len(self.acc_rings)))
            self.mma_tensors |= node.target is dot_scaled and not all(
                gluon_ops.copied_scale(node, i) for i in gluon_ops._SCALES[dot_scaled]
            )
        # The step's groups share its channel's slot: one commit frees it.
        empty, step = (_link(dot_index, groups[0], name) for name in ("empty", "step"))
        return gluon_ops.codegen_tmem_dot(interp, node, (empty, step, _DEPTH), ring)

    def _tmem_left(self, *rings: tuple[int, ...]) -> int:
        """Tensor-memory columns the accumulator rings (and ``rings`` more) and the
        other dots' tensor memory leave."""
        shapes = [*(shape for shape, _ in self.acc_rings), *rings]
        acc = [(m, n * _ACC_DEPTH) for m, n in shapes]
        return _TMEM_COLUMNS - _tmem_columns([*acc, *self.tmem])

    def _take_deferred(
        self, interp: GraphInterpreter, node: torch.fx.Node
    ) -> dict[torch.fx.Node, object]:
        """Acquire the dot's own slot of each deferred operand it reads; returns the
        bindings it replaced."""
        replaced: dict[torch.fx.Node, object] = {}
        env = cast("dict[torch.fx.Node, object]", interp.env)
        cg = cast("GenerateAST", interp.cg)
        taken: dict[str, str] = {}  # one slot per tile, even read as both operands
        for position in gluon_ops.operands(node.target):
            operand = cast("torch.fx.Node", node.args[position])
            value = env.get(operand)
            if not isinstance(value, SharedValue):
                continue
            slots, read = {}, _names(value.expr)
            # In the loads' order (a set's varies from run to run).
            for name in (n for n in self.deferred if n in read):
                if name not in taken:
                    load, tensor, indexing, group, swizzle = self.deferred[name]
                    step = self.dots.setdefault(node, len(self.dots))
                    taken[name], _ = self._acquire(
                        cg, (load, node), tensor, indexing, step, group, swizzle
                    )
                slots[name] = taken[name]
            if slots:
                replaced[operand] = value
                expr = cast("ast.AST", clone_ast(value.expr))
                env[operand] = SharedValue(ast_rename(expr, slots))
        return replaced

    @staticmethod
    def _group(node: torch.fx.Node, tensor: torch.Tensor) -> int | None:
        """Ungated if nothing writes the load's allocation, gated if only other
        roots do, else None."""
        allocation = _allocation(tensor)
        if (roots := _roots()) is None:
            return None if allocation in _written_allocations() else _UNGATED
        writers = roots.get(allocation, (set(), set()))[0]
        graph = HostFunction.current().device_ir.tile_dependency_graph
        own = {graph.accesses[i].root for i in node.meta[TILE_ACCESS_META]}  # pyrefly: ignore [missing-attribute]
        return _UNGATED if not writers else None if writers & own else _GATED

    # -- the ring kernel

    def emit(self, body: ast.FunctionDef) -> list[ast.stmt]:
        """The partitions and the ring kernel launching them."""
        name, arguments = body.name, [a.arg for a in body.args.args]
        # A channel carries the steps of one issuing partition; a group without
        # operands in a step arrives from its skip.
        df = self.device_function
        issuer = {index: self.mma[node] for node, index in self.dots.items()}
        channels: dict[bool, int] = {}
        group_puts: dict[int, set[str]] = {}
        mma: set[str] = set()
        rename: dict[str, str] = {}
        expects: dict[tuple[int, int], int] = {}
        steps: dict[tuple[int, int], list[_Site]] = {}
        dot_groups: dict[int, set[int]] = {}
        channel_groups: dict[bool, set[int]] = {}
        for site in self.sites.values():
            channel = channels.setdefault(issuer[site.dot], len(channels))
            for link in ("full", "empty", "step"):
                rename[_link(site.dot, site.group, link)] = _channel(link, channel)
            group_puts.setdefault(site.group, set()).add(
                _link(site.dot, site.group, "bytes")
            )
            if issuer[site.dot]:
                mma |= {_channel("full", channel), _channel("empty", channel)}
            steps.setdefault((channel, site.dot), []).append(site)
            # A tagged operand's bytes are stores, which its last put arrives for.
            key = (site.dot, site.group)
            expects[key] = expects.get(key, 0) + (0 if site.tagged else site.nbytes)
            dot_groups.setdefault(site.dot, set()).add(site.group)
            channel_groups.setdefault(issuer[site.dot], set()).add(site.group)
        # Tagged puts' geometry by box; by a group step's last box, the step's
        # tagged operands (box, site, first), which complete there.
        tagged: dict[str, tuple[int, ...]] = {}
        group_boxes: dict[str, list[tuple[str, str, bool]]] = {}
        for dot, group in list(expects):
            mine = [s for s in self.sites.values() if (s.dot, s.group) == (dot, group)]
            if not any(s.tagged for s in mine):
                continue
            if not all(s.tagged for s in mine):
                raise TaggedPeerFallback(
                    "T6: a ring step's operands of one group mix tagged and TMA ones"
                )
            tagged |= {f"{s.name}_box": cast("tuple[int, ...]", s.tagged) for s in mine}
            group_boxes[f"{mine[-1].name}_box"] = [
                (f"{s.name}_box", s.name, s is mine[0]) for s in mine
            ]
        # Each producer replays every step of its channel; without operands a skip.
        skips: dict[str, list[ast.stmt]] = {}
        for dot, groups in dot_groups.items():
            lead = min(groups)
            for group in channel_groups[issuer[dot]] - groups:
                for link in ("full", "empty", "step"):
                    rename[_link(dot, group, link)] = rename[_link(dot, lead, link)]
                group_puts[group].add(_link(dot, group, "bytes"))
                expects[(dot, group)] = 0
                skips.setdefault(_link(dot, lead, "step"), []).append(
                    statement_from_string(
                        f"{ALIAS}.ring_skip({_links(dot, group, 'full', 'empty', 'step')}, "
                        f"{_DEPTH}, {_link(dot, group, 'bytes')})"
                    )
                )
        body.body = _with_skips(body.body, skips)
        buffers, views, box = self._lanes(channels.values(), steps, rename)
        ast_rename(body, rename)
        # The completing put reads each tagged operand's slot (after renames).
        for boxes in group_boxes.values():
            boxes[:] = [(b, rename[site], first) for b, site, first in boxes]
        claims = _take_claims(body.body, df.sync_names("claim"))
        ids = list(channels.values())
        accs = [_acc(index) for index in self.acc_rings.values()]
        launch_state = _launch_state(body, df)
        body.body[:0] = [
            statement_from_string(f"{counter} = tl.full([], 0, tl.int32)")
            for counter in [
                *(_channel("step", c) for c in ids),
                *(acc[3] for acc in accs),
                *([_TASK_STEP] if claims else []),
            ]
        ]
        mma |= {n for acc in accs for n in acc}
        producers = {
            group: _Slice(df, puts, group == _GATED, tagged=(tagged, group_boxes))
            for group, puts in sorted(group_puts.items())
        }
        slices = [(f"producer{group}", s) for group, s in producers.items()]
        pollers = 1
        if tagged and _GATED in producers:
            # More pollers of the gated group's tagged steps; each arrives on
            # every step.
            pollers = min(_TAGGED_POLLERS, self.depth)
            producers[_GATED].owner = (0, pollers)
            slices += [
                (
                    f"producer{_GATED}_{i}",
                    _Slice(
                        df,
                        group_puts[_GATED],
                        True,
                        tagged=(tagged, group_boxes),
                        owner=(i, pollers),
                    ),
                )
                for i in range(1, pollers)
            ]
        if any(issuer.values()):
            slices.append(("mma", _Slice(df, None, False, mma)))
        task_depth = _ACC_DEPTH + 1
        if claims and _claims_lead(body.body, claims, group_puts.get(_UNGATED)):
            producers[_UNGATED].claims = claims
            task_depth = self.depth + _ACC_DEPTH + 1
        elif claims:
            slices.append(("claimer", _Slice(df, set(), False, claims=claims)))
        # The weight producer paces its first fill (one load in flight) beside
        # gated producers on CTAs running a loading root ahead of the ring: there
        # the fill is a prefetch whose bulk loads would queue that root's; on the
        # others it is the critical path.
        ahead = _loading_ahead(body.body, df)
        pace = int(_UNGATED in group_puts and _GATED in group_puts)
        partitions: list[ast.FunctionDef] = []
        workers: list[tuple[int, bool]] = []  # (warps, MMA side storing tensors)
        for suffix, body_slice in slices:
            partition_body, read = _partition(body.body, body_slice)
            if body_slice.gated or read:
                # A partition reading what other roots write keeps the waits
                # ordering its reads.
                names = _all_names(partition_body) if body_slice.gated else read
                storage = _argument_allocations(df)
                body_slice.reads = frozenset(storage[n] for n in names if n in storage)
                body_slice.gated = True
                partition_body, read = _partition(body.body, body_slice)
            if read:
                raise RingFallback(
                    f"the ring: a ring partition reads {sorted(read)}, "
                    "which this kernel writes"
                )
            if pace and body_slice.weights:
                stepped = _all_names(partition_body)
                partition_body = _paced_fill(
                    partition_body,
                    [c for c in ids if _channel("step", c) in stepped],
                    pace,
                    df.sync_names("wait"),
                    _any(ahead),
                )
            # Warps: producers 2, an MMA side storing scales 4, scalar-only 1.
            stores = body_slice.puts is None and self.mma_tensors
            warps = 4 if stores else 2
            if not (stores or _computes_tensors(partition_body)):
                warps = 1
            workers.append((min(df.config.num_warps, warps), stores))
            partitions.append(_function(f"{name}_{suffix}", body.args, partition_body))
        body.body = _without_waits(_without_mma(body.body, mma), df, frozenset())
        functions = [body, *partitions]
        for fn in functions:
            dead_assignment_elimination(cast("list[ast.AST]", fn.body), df.dce_vars)
            _first_waits(fn.body)
        if launch_state is not None:
            launch, finish, launch_names = launch_state
            # Each partition reads the launch state in place of its grid wait.
            read = f"{', '.join(launch_names)}, = {launch.name}({', '.join(arguments)})"
            waits = any(_is_gdc_wait(st) for st in launch.body)
            for fn in functions:
                at = next((i for i, st in enumerate(fn.body) if _is_gdc_wait(st)), None)
                assert (at is not None) == waits, "every partition keeps the grid wait"
                if at is None:
                    fn.body.insert(0, statement_from_string(read))
                else:
                    fn.body[at] = statement_from_string(read)
            # The body hands the state it read to the kernel's rewrite.
            body.body.append(
                statement_from_string(f"return {', '.join(launch_names)},")
            )
        used = _all_names([st for fn in functions for st in fn.body])
        paces = [p for c in ids if (p := _channel("pace", c)) in used]
        extra = [
            *(_channel("full", c) for c in ids),
            *(_channel("empty", c) for c in ids),
            *paces,
            *(buffer for buffer, _, _ in buffers),
            *(view for view, *_ in views),
            *(n for acc in accs for n in acc[:3]),
            *([_TASKS, _TASK_FULL, _TASK_EMPTY] if claims else []),
        ]
        shell_arguments = cast("ast.arguments", clone_ast(body.args))
        for fn in functions:
            fn.args.args.extend(create_arg(argument) for argument in extra)
        body.name = f"{name}_body"
        call = f"({', '.join([*arguments, *extra])},)"
        tasks = ", ".join(f"({fn.name}, {call})" for fn in functions)
        # A persistent grid of ``num_sm_multiplier`` CTAs a SM keeps them resident:
        # each takes its share of the register file (the launch's ``maxnreg``).
        ctas = df.config.num_sm_multiplier
        threads = 32 * ctas * _cta_warps(df.config.num_warps, workers)
        device = CompileEnvironment.current().device
        if (
            threads
            > torch.cuda.get_device_properties(device).max_threads_per_multi_processor
        ):
            raise RingFallback(
                f"the ring: {ctas} CTAs a SM of {threads // ctas} threads "
                "exceed the SM's thread slots"
            )
        regs = _worker_registers(df.config.num_warps, workers, ctas)
        if polling := [
            i for i, p in enumerate(partitions) if _ring_calls(p, "tagged_load")
        ]:
            regs = _poller_registers(
                df.config.num_warps,
                workers,
                regs,
                polling,
                [role.startswith(f"producer{_GATED}_") for role, _ in slices],
                ctas,
            )
        # Workers may hold more than the launch's maxnreg (setmaxnreg moves the
        # rest from the body), so long as the body keeps setmaxnreg's minimum.
        maxnreg = _maxnreg(df.config.num_warps, workers, ctas)
        body_regs = _body_registers(df.config.num_warps, workers, regs, ctas)
        if body_regs < MIN_REGISTERS:
            raise RingFallback(
                f"the ring: {ctas} CTAs a SM leave the body {body_regs} registers "
                f"a thread beside workers of {regs}"
            )
        roles = [role for role, _ in slices]
        _move_prefetch_code(
            body, partitions, workers, roles.index("mma") if "mma" in roles else None
        )
        setup = []
        for c, groups in ((channels[i], g) for i, g in channel_groups.items()):
            # Every group (and poller) of a channel arrives on each full phase.
            arrivals = len(groups) + (pollers - 1) * (_GATED in groups)
            setup.append(
                f"{_channel('full', c)} = {ALIAS}.ring_barriers({_DEPTH}, {arrivals})"
            )
        setup += [
            f"{_channel('empty', c)} = {ALIAS}.ring_barriers({_DEPTH})" for c in ids
        ]
        setup += [f"{b} = {ALIAS}.ring_barriers({pace})" for b in paces]
        setup += [
            f"{b} = {ALIAS}.ring_buffer({d}, {_DEPTH} * {k})" for b, d, k in buffers
        ]
        setup += [
            f"{view} = {ALIAS}.ring_view({b}, {d}, {_DEPTH} * {boxes})"
            for view, b, d, boxes in views
        ]
        if claims:
            # The body and every partition but the claiming one take each ticket.
            setup += [
                f"{_TASKS} = {ALIAS}.task_buffer({_TASK_DEPTH})",
                f"{_TASK_FULL} = {ALIAS}.ring_barriers({_TASK_DEPTH})",
                f"{_TASK_EMPTY} = {ALIAS}.ring_barriers({_TASK_DEPTH}, {len(partitions)})",
            ]
        for (shape, _), index in self.acc_rings.items():
            acc, *barriers, _ = _acc(index)
            dims = ", ".join(map(str, [*shape, _ACC_DEPTH]))
            setup.append(f"{acc} = {gluon_ops.ALIAS}.acc_buffers({dims})")
            setup += [f"{b} = {ALIAS}.ring_barriers({_ACC_DEPTH})" for b in barriers]
        warps = ", ".join(str(w) for w, _ in workers)
        specialize = (
            f"gl.warp_specialize([{tasks}], [{warps}], [{', '.join(map(str, regs))}])"
        )
        if launch_state is None:
            setup.append(specialize)
        else:
            # The launch state is rewritten once every partition has read it.
            setup.append(f"{', '.join(launch_names)}, = {specialize}")  # pyrefly: ignore [unbound-name]
            setup.append(f"{finish.name}({', '.join([*arguments, *launch_names])})")  # pyrefly: ignore [unbound-name]
        shell = _function(
            name, shell_arguments, list(map(statement_from_string, setup))
        )
        # The launched shell takes the kernel's decorators (its specialization
        # options) and, several CTAs a SM, their share of the register file.
        shell.decorator_list = cast("list[ast.expr]", clone_ast(body.decorator_list))
        if ctas > 1:
            native = shell.decorator_list[-1]
            if not isinstance(native, ast.Call):
                native = shell.decorator_list[-1] = create(
                    ast.Call, func=native, args=[], keywords=[]
                )
            native.keywords.append(
                create(ast.keyword, arg="maxnreg", value=ast.Constant(maxnreg))
            )
        df.codegen.module_statements.extend(
            statement_from_string(f"{constant} = tl.constexpr({value})")
            for constant, value in [
                (_DEPTH, self.depth),
                *([(_TASK_DEPTH, task_depth)] if claims else []),
                *((_link(*step, "bytes"), n) for step, n in expects.items()),
                *box.items(),
            ]
        )
        prologue = [] if launch_state is None else [launch, finish]  # pyrefly: ignore [unbound-name]
        return [*prologue, *functions, shell]

    @staticmethod
    def _lanes(
        channels: Iterable[int],
        steps: dict[tuple[int, int], list[_Site]],
        rename: dict[str, str],
    ) -> tuple[
        list[tuple[str, str, int]], list[tuple[str, str, str, int]], dict[str, int]
    ]:
        """A slot is lanes: each step's operands, widest first, fill a lane as wide
        as the widest operand left, at whole-box offsets; the widest allocates it,
        the others view it.  A TMA copy lands 128-byte aligned: an operand takes a
        whole number of 128 bytes (its boxes).  Returns the lanes (with their boxes
        per slot), views and boxes; renames sites."""
        buffers: list[tuple[str, str, int]] = []
        views: list[tuple[str, str, str, int]] = []
        box: dict[str, int] = {}

        def size(site: _Site) -> int:
            return math.lcm(site.nbytes, 128)

        for channel in channels:
            left = [
                sorted(sites, key=lambda s: -size(s))
                for (c, _), sites in steps.items()
                if c == channel
            ]
            while any(left):
                widest = max(
                    (sites[0] for sites in left if sites),
                    key=lambda s: (size(s), s.slot_type[2]),
                )
                buffer = f"_helion_ring_buf{len(buffers)}"
                buffers.append(
                    (buffer, widest.descriptor.name, size(widest) // widest.nbytes)
                )
                for sites in left:
                    used = 0
                    while (
                        sites
                        and size(widest) - used >= size(sites[0])
                        and size(widest) % size(sites[0]) == used % size(sites[0]) == 0
                    ):
                        site = sites.pop(0)
                        box[f"{site.name}_box"] = used // site.nbytes
                        used += size(site)
                        rename[site.name] = buffer
                        if site.slot_type != widest.slot_type:
                            rename[site.name] = f"_helion_ring_view{len(views)}"
                            boxes = size(widest) // site.nbytes
                            views.append(
                                (rename[site.name], buffer, site.descriptor.name, boxes)
                            )
        return buffers, views, box


def _dot_between(load: torch.fx.Node, dot: torch.fx.Node) -> bool:
    """Whether another dot comes between ``load`` and ``dot`` (its graph's order)."""
    node = load.next
    while node is not dot and node.op != "root":
        if gluon_ops.is_dot(node):
            return True
        node = node.next
    return False


def _slice(
    body: list[ast.stmt],
    names: set[str],
    keep: Callable[[ast.stmt], bool] = lambda _: False,
) -> list[ast.stmt]:
    """The statements of ``body`` the values of ``names`` depend on (and those
    ``keep`` takes, with theirs), in order."""
    kept: list[ast.stmt] = []
    for statement in reversed(body):
        stored = _stored_names([statement])
        if stored & names or keep(statement):
            kept.insert(0, statement)
            simple = isinstance(statement, ast.Assign | ast.Expr)
            names = (names - stored) | _names(
                cast("ast.Assign", statement).value if simple else statement
            )
    return kept


def _blocks(tree: ast.AST) -> list[list[ast.stmt]]:
    """The statement lists of ``tree``: its bodies and else branches."""
    return [
        block
        for node in ast.walk(tree)
        for field, block in ast.iter_fields(node)
        if field in ("body", "orelse") and isinstance(block, list)
    ]


def _rebuilt(
    statement: ast.stmt,
    block: Callable[[list[ast.stmt]], list[ast.stmt]],
    copy: bool = True,
) -> ast.stmt | None:
    """A compound statement (a copy, unless not ``copy``) with ``block`` applied to
    its body (a ``pass`` if that empties it) and else branch; None for others."""
    if not isinstance(statement, (ast.For, ast.While, ast.If)):
        return None
    out = cast(
        "ast.For | ast.While | ast.If", clone_ast(statement) if copy else statement
    )
    out.body = block(out.body) or [create(ast.Pass)]
    out.orelse = block(out.orelse)
    return out


def _first_waits(body: list[ast.stmt]) -> None:
    """Skip the wait of an acquire whose step an earlier acquire waited for."""
    waited: set[tuple[str, str]] = set()  # (full barrier, step counter)
    for statement in body:
        if isinstance(statement, (ast.For, ast.While, ast.If)):
            _first_waits(statement.body)
            _first_waits(statement.orelse)
            waited.clear()
            continue
        if (acquire := _ring_call(statement, "ring_acquire")) is not None:
            key = (
                ast.unparse(_arg(acquire, "full")),
                ast.unparse(_arg(acquire, "step")),
            )
            if key in waited:
                acquire.keywords.append(
                    ast.keyword(arg="WAIT", value=ast.Constant(False))
                )
            waited.add(key)
        stored = _stored_names([statement])
        waited = {key for key in waited if key[1] not in stored}


def _is_wait(statement: ast.stmt, waits: set[str]) -> bool:
    if isinstance(statement, (ast.If, ast.For)):
        nested = [*statement.body, *statement.orelse]
        return bool(nested) and all(_is_wait(s, waits) for s in nested)
    if isinstance(statement, ast.While):
        return bool(_all_names([statement.test]) & waits)
    return isinstance(statement, ast.Assign) and _stored_names([statement]) <= waits


def _orders(
    device_function: DeviceFunction,
    statement: ast.stmt,
    accesses: frozenset[int] | None,
    publishes: bool = False,
) -> bool:
    """Whether a dependency wait orders an access of ``accesses`` (None: any) or,
    for the body (``publishes``), a barrier publication."""
    roles = device_function.sync_roles
    return any(
        accesses is None
        or (role := roles[name]).covers is None
        or role.covers & accesses  # pyrefly: ignore [unsupported-operation]
        or (publishes and role.publication)
        for name in _all_names([statement]) & device_function.sync_names("wait")
    )


def _without_waits(
    body: list[ast.stmt], df: DeviceFunction, after: frozenset[int]
) -> list[ast.stmt]:
    """The body without the waits ordering none of the accesses that may follow
    (``after``: those past this block) nor a barrier publication."""
    allocations = _argument_allocations(df)
    waits = df.sync_names("wait")
    later = [after]  # later[i]: the allocations body[i:] (or ``after``) access
    for statement in reversed(body):
        later.append(later[-1] | _named_allocations(statement, allocations))
    later.reverse()
    kept: list[ast.stmt] = []
    for index, statement in enumerate(body):
        if _is_wait(statement, waits) and not isinstance(statement, ast.If):
            if _orders(df, statement, later[index + 1], publishes=True):
                kept.append(statement)
        else:
            # A loop's next iterations follow the waits in its body.
            inner = later[index + isinstance(statement, ast.If)]
            block = functools.partial(_without_waits, df=df, after=inner)
            kept.append(_rebuilt(statement, block, copy=False) or statement)
    return kept


def _roots() -> dict[int, tuple[set[int], set[int]]] | None:
    """The roots storing to and loading each allocation (``_allocation``), from
    the tile dependency graph (None without one)."""
    graph = HostFunction.current().device_ir.tile_dependency_graph
    if graph is None:
        return None
    roots: dict[int, tuple[set[int], set[int]]] = {}
    for access, allocation in _access_allocations().items():
        tile_access = graph.accesses[access]
        stores, loads = roots.setdefault(allocation, (set(), set()))
        (loads if tile_access.kind == "load" else stores).add(tile_access.root)
    return roots


def _ordered_tensors(device_function: DeviceFunction) -> set[str]:
    """Tensor arguments only roots that do not read them write."""
    roots = _roots() or {}
    return {
        name
        for name, allocation in _argument_allocations(device_function).items()
        if (stores := roots.get(allocation, (set(), set()))[0])
        and not stores & roots[allocation][1]
    }


def _launch_state(
    body: ast.FunctionDef, device_function: DeviceFunction
) -> tuple[ast.FunctionDef, ast.FunctionDef, list[str]] | None:
    """The body's per-launch state reads (a function every partition calls) and
    rewrites (one the kernel calls after the join): (launch, finish, names)."""
    launch = device_function.sync_names("launch")
    statements = [
        s
        for s in body.body
        if isinstance(s, ast.Assign)
        and (targets := _stored_names([s]))
        and targets <= launch
    ]
    if not statements:
        return None
    names = [n for s in statements for n in sorted(_stored_names([s]))]
    # The reads and the statements they depend on, in program order.
    prefix = body.body[: body.body.index(statements[-1]) + 1]
    prologue = [
        cast("ast.stmt", clone_ast(s))
        for s in _slice(prefix, set(), lambda s: s in statements or _is_gdc_wait(s))
    ]
    # Every partition runs the reads (they must not write anything); the rewrites
    # sit at the body's top level, one per read.
    writes = {id(write) for write in device_function.launch_writes}
    rewrites = [s for s in body.body if id(s) in writes]
    if (
        any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == "store" or node.func.attr.startswith("atomic"))
            for statement in prologue
            for node in ast.walk(statement)
        )
        or not len(rewrites) == len(writes) > 0
        or not set(names) <= {n for s in rewrites for n in _names(s)}
    ):
        raise RingFallback("the ring: launch state the partitions cannot share")
    body.body = [s for s in body.body if s not in statements and s not in rewrites]
    returned = statement_from_string(f"return {', '.join(names)},")
    launch_fn = _function(f"{body.name}_launch", body.args, [*prologue, returned])
    finish_args = cast("ast.arguments", clone_ast(body.args))
    finish_args.args.extend(create_arg(name) for name in names)
    finish = _function(f"{body.name}_finish", finish_args, rewrites)
    return launch_fn, finish, names


def _function(
    name: str, args: ast.arguments, body: list[ast.stmt], decorator: str = "gluon.jit"
) -> ast.FunctionDef:
    return create(
        ast.FunctionDef,
        name=name,
        args=clone_ast(args),
        body=body,
        decorator_list=[expr_from_string(decorator)],
        type_params=[],
    )


# ---- the partitions: slices of the body -----------------------------------------


def _names(node: ast.AST | None) -> set[str]:
    return set() if node is None else _all_names([node])


def _name(name: str) -> ast.Name:
    return create(ast.Name, id=name, ctx=ast.Load())


def _call(attribute: str, args: list[ast.expr]) -> ast.Call:
    func = expr_from_string(f"{ALIAS}.{attribute}")
    return create(ast.Call, func=func, args=args, keywords=[])


def _ring_calls(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        c for c in ast.walk(node) if _is_tl_call(c, frozenset([attribute]), _MODULES)
    ]


def _arg_index(call: ast.Call, name: str) -> int:
    """The position of parameter ``name`` of the ring (or ``gluon_ops``)
    function ``call`` calls."""
    attr = cast("ast.Attribute", call.func).attr
    callee = globals().get(attr) or getattr(gluon_ops, attr)
    fn = callee.fn if isinstance(callee, JITFunction) else callee
    return list(inspect.signature(fn).parameters).index(name)


def _arg(call: ast.Call, name: str) -> ast.expr:
    """Argument ``name`` of a call of a ring (or ``gluon_ops``) function."""
    return call.args[_arg_index(call, name)]


def _skip(acquire: ast.Call) -> ast.Call:
    """The ``ring_skip`` of an acquire's step (from its same-named arguments)."""
    return _call("ring_skip", [_arg(acquire, n) for n in ring_skip.arg_names])  # pyrefly: ignore [missing-attribute]


def _ring_call(node: ast.AST, attribute: str) -> ast.Call | None:
    return next(iter(_ring_calls(node, attribute)), None)


class _Slice:
    """Keeps the statements a partition's seeds depend on, plus their control flow.
    Seeds: a producer's acquires (``puts``), the MMA side (``puts`` None), the
    ticket takes (``claims``).  Only a gated partition keeps dependency waits; any
    other effect a seed depends on rejects the configuration."""

    def __init__(
        self,
        device_function: DeviceFunction,
        puts: set[str] | None,
        gated: bool,
        mma: set[str] | None = None,
        claims: dict[int, ast.expr] | None = None,
        tagged: tuple[
            dict[str, tuple[int, ...]], dict[str, list[tuple[str, str, bool]]]
        ]
        | None = None,
        owner: tuple[int, int] = (0, 1),
    ) -> None:
        self.device_function = device_function
        self.puts = puts
        self.gated = gated
        # Tagged operands' geometries and group steps (``RingLowering.emit``).
        self.tagged, self.tagged_steps = tagged or ({}, {})
        # (this poller, pollers): poller i fills tagged steps i mod pollers, only
        # poller 0 the others; each arrives on every step (no parity aliasing).
        self.owner = owner
        # The weight producer (even once its reads make it keep waits).
        self.weights = not gated and bool(puts)
        # The allocations whose reads' waits a gated producer keeps (None: all).
        self.reads: frozenset[int] | None = None
        self.mma = mma or set()
        self.claims = claims
        self.scheduled = set(device_function.sync_roles)
        self.waits = device_function.sync_names("wait")
        self.polls = device_function.sync_names("poll")
        self.helpers = {name for name, *_ in device_function.triton_outlined_helpers}

    def block(
        self, body: list[ast.stmt], live: set[str], follows: bool
    ) -> list[ast.stmt]:
        kept: list[ast.stmt] = []
        for statement in reversed(body):
            later = follows or bool(kept)
            producer = bool(self.puts)
            if later and producer and self.unordered(statement):
                raise RingFallback(
                    "the ring: a producer cannot order its loads after "
                    f"{ast.unparse(statement)}"
                )
            # It keeps the waits ordering its reads (guarded ones one by one).
            if (
                self.gated
                and later
                and not isinstance(statement, ast.If)
                and _is_wait(statement, self.waits)
                and _orders(self.device_function, statement, self.reads)
            ):
                live |= _names(statement) - _stored_names([statement])
                kept.append(_scalar_form(statement, self.device_function.sync_roles))
            elif (result := self.statement(statement, live, later)) is not None:
                kept.extend(reversed(result) if isinstance(result, list) else [result])
        kept.reverse()
        return kept

    def unordered(self, statement: ast.stmt) -> bool:
        """Whether a producer cannot reproduce ``statement``'s synchronization: a user
        call or impure inline asm that is not one of the scheduler's."""
        if isinstance(statement, (ast.For, ast.While, ast.If)) or (
            isinstance(statement, ast.Assign)
            and _stored_names([statement]) <= self.scheduled
        ):
            return False
        for call in (n for n in ast.walk(statement) if isinstance(n, ast.Call)):
            func = call.func
            asm = (
                isinstance(func, ast.Attribute)
                and func.attr == "inline_asm_elementwise"
            )
            user = isinstance(func, ast.Name) and func.id not in self.helpers
            if (asm or user) and _effect_kind(call) != "pure":
                return True
        return False

    def statement(
        self, statement: ast.stmt, live: set[str], follows: bool
    ) -> ast.stmt | list[ast.stmt] | None:
        # A while loop on a wait or poll is a spin, not a loop to replay.
        if isinstance(statement, ast.For) or (
            isinstance(statement, ast.While)
            and not _names(statement.test) & (self.waits | self.polls)
        ):
            return self.loop(statement, live, follows)
        if isinstance(statement, ast.If):
            body_live, else_live = set(live), set(live)
            body = self.block(statement.body, body_live, follows)
            orelse = self.block(statement.orelse, else_live, follows)
            if not body and not orelse:
                return None
            live |= body_live | else_live | _names(statement.test)
            return create(
                ast.If,
                test=statement.test,
                body=body or [create(ast.Pass)],
                orelse=orelse,
            )
        if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return self.assignment(statement, live)
        skip = _ring_call(statement, "ring_skip")
        # Every partition waits for the previous grid (its loop bounds may read
        # what that grid wrote).
        if isinstance(statement, ast.Expr) and (
            _is_gdc_wait(statement)
            or (
                (skip and _names(_arg(skip, "NBYTES")) & self.puts)
                if self.puts is not None
                else _mma_side(statement, self.mma)
            )
        ):
            live |= _names(statement)
            return statement
        if (
            _ring_call(statement, "ring_acquire") is not None
            or _stored_names([statement]) & live
        ):
            raise RingFallback(
                f"the ring: a ring partition depends on {ast.unparse(statement)}"
            )
        return None

    def loop(
        self, statement: ast.For | ast.While, live: set[str], follows: bool
    ) -> ast.stmt | None:
        if isinstance(statement, ast.For):
            defined, header = _names(statement.target), _names(statement.iter)
            body_live = set(live)
        else:
            # A kept while loop evaluates its test after every iteration.
            defined, header = set(), _names(statement.test)
            body_live = live | header
        while True:
            entry = set(body_live)
            body = self.block(statement.body, entry, follows)
            carried = entry - defined
            done = carried <= body_live and (follows or not body)
            body_live |= carried
            # A kept statement runs again in the next iteration.
            follows = follows or bool(body)
            if done:
                break
        if not body:
            return None
        live |= carried | header
        if isinstance(statement, ast.While):
            return create(ast.While, test=statement.test, body=body, orelse=[])
        return create(
            ast.For,
            target=statement.target,
            iter=statement.iter,
            body=body,
            orelse=[],
            type_comment=None,
        )

    def assignment(
        self, statement: ast.Assign | ast.AugAssign | ast.AnnAssign, live: set[str]
    ) -> ast.stmt | list[ast.stmt] | None:
        targets = _stored_names([statement])
        if (take := _ring_call(statement, "task_take")) is not None:
            # Every partition takes every ticket (the task channel counts them).
            statement = cast("ast.Assign", clone_ast(statement))
            if self.claims is not None:
                claim = cast("ast.expr", clone_ast(self.claims[id(take)]))
                statement.value = _call("task_put", [*take.args, claim])
            live -= targets
            live |= _names(statement.value)
            return statement
        if (acquire := _ring_call(statement, "ring_acquire")) is not None:
            if self.puts is None:
                # The consumer side needs the slot alone.
                if not targets & live:
                    return None
                statement = cast("ast.Assign", clone_ast(statement))
                consumer = _ring_call(statement, "ring_acquire")
                assert consumer is not None
                for name in ("desc", "coords"):
                    consumer.args[_arg_index(consumer, name)] = ast.Constant(None)
                del consumer.args[_arg_index(consumer, "TAGGED") :]
                live -= targets
                live |= _names(consumer)
                return statement
            if not _names(_arg(acquire, "NBYTES")) & self.puts:
                return None
            box, first, skip = (
                _arg(acquire, "BOX"),
                _arg(acquire, "FIRST"),
                _skip(acquire),
            )
            tagged = isinstance(box, ast.Name) and box.id in self.tagged
            if self.owner[0] and not tagged:
                # Poller 0's step: the others only take their turn of it.
                if not (isinstance(first, ast.Constant) and first.value):
                    return None
                live |= _names(skip)
                return create(ast.Expr, value=skip)
            live |= _names(acquire)
            if tagged:
                # A tagged operand, polled from its mirror: its loads issue here,
                # every tagged operand of the group step completes at the last.
                runtime = _arg(acquire, "TAGGED")
                assert isinstance(box, ast.Name) and isinstance(runtime, ast.Tuple)
                geometry: list[ast.expr] = [
                    ast.Constant(n) for n in self.tagged[box.id]
                ]
                load = statement_from_string(f"_helion_tagged_{box.id} = 0")
                assert isinstance(load, ast.Assign)
                load.value = _call("tagged_load", [*runtime.elts, *geometry])
                issued: list[ast.stmt] = [load]
                steps = self.tagged_steps.get(box.id, [])
                epoch = runtime.elts[2]
                for index, (other, ring, first) in enumerate(steps):
                    live |= {ring, f"_helion_tagged_{other}"}
                    put: list[ast.expr] = [
                        _name(ring),
                        *(_arg(acquire, a) for a in ring_put_tagged.arg_names[1:5]),  # pyrefly: ignore [missing-attribute]
                        _name(f"_helion_tagged_{other}"),
                        epoch,
                        ast.Constant(first),
                        ast.Constant(index == len(steps) - 1),
                        _name(other),
                        *(ast.Constant(n) for n in self.tagged[other]),
                    ]
                    call = _call("ring_put_tagged", put)
                    issued.append(create(ast.Expr, value=call))
                poller, pollers = self.owner
                if pollers == 1:
                    return issued
                # This poller's steps: on the others' it issues nothing (a zero
                # tuple keeps the name bound) and only takes its turn.
                mine = expr_from_string(
                    f"{ast.unparse(_arg(acquire, 'step'))} % {pollers} == {poller}"
                )
                none = cast("ast.Assign", clone_ast(load))
                none.value = _call("tagged_none", [runtime.elts[0], *geometry])
                result: list[ast.stmt] = [
                    create(ast.If, test=mine, body=[load], orelse=[none])
                ]
                if issued[1:]:
                    result.append(
                        create(
                            ast.If,
                            test=clone_ast(mine),
                            body=issued[1:],
                            orelse=[create(ast.Expr, value=skip)],
                        )
                    )
                return result
            return create(ast.Expr, value=_call("ring_put", acquire.args))
        if not targets & live:
            return None
        value = cast("ast.expr", statement.value)
        if self.puts is None and _mma_side(statement, self.mma):
            live -= targets
            live |= _names(value)
            return statement
        for call in (n for n in ast.walk(value) if isinstance(n, ast.Call)):
            if not _may_write(call):
                continue
            roles = self.device_function.sync_roles
            if isinstance(statement, ast.Assign) and all(
                n in roles and roles[n].zero for n in targets
            ):
                live -= targets
                if self.gated:
                    live |= _names(value)
                    return statement
                return statement_from_string(f"{next(iter(targets))} = 0")
            raise RingFallback(
                f"the ring: a ring partition depends on {ast.unparse(statement)}"
            )
        if not isinstance(statement, ast.AugAssign):
            live -= targets
        live |= _names(value)
        return statement


def _claims_lead(
    body: list[ast.stmt], claims: dict[int, ast.expr], puts: set[str] | None
) -> bool:
    """Whether the weight producer steps in every claimed task loop (so its ring
    paces the claims it would issue)."""
    if not puts:
        return False
    for loop in (n for statement in body for n in ast.walk(statement)):
        if not isinstance(loop, ast.While) or not any(
            id(take) in claims for take in _ring_calls(loop, "task_take")
        ):
            continue
        sizes = [
            _arg(c, "NBYTES")
            for c in [
                *_ring_calls(loop, "ring_acquire"),
                *_ring_calls(loop, "ring_skip"),
            ]
        ]
        if not any(_names(size) & puts for size in sizes):
            return False
    return True


def _take_claims(body: list[ast.stmt], names: set[str]) -> dict[int, ast.expr]:
    """Turn the body's ticket claims into task-channel takes; returns each take's
    claim (by id), which the claiming partition issues instead."""
    claims: dict[int, ast.expr] = {}
    for block in _blocks(create(ast.Module, body=body, type_ignores=[])):
        for index in reversed(range(len(block))):
            statement = block[index]
            if not (
                isinstance(statement, ast.Assign)
                and (targets := _stored_names([statement]))
                and targets <= names
            ):
                continue
            take = expr_from_string(
                f"{ALIAS}.task_take({_TASKS}, {_TASK_FULL}, {_TASK_EMPTY}, "
                f"{_TASK_STEP}, {_TASK_DEPTH})"
            )
            claims[id(take)] = statement.value
            statement.value = cast("ast.expr", take)
            block.insert(
                index + 1, statement_from_string(f"{_TASK_STEP} = {_TASK_STEP} + 1")
            )
    return claims


def _mma_side(statement: ast.stmt, mma: set[str]) -> bool:
    """Whether ``statement`` is the MMA side of a dot the MMA partition issues."""
    return any(
        _is_tl_call(call, _MMA_CALLS, _MODULES)
        and (cast("ast.Attribute", call.func).attr in _ACC_CALLS or _names(call) & mma)
        for call in ast.walk(statement)
    )


def _loading_ahead(
    body: list[ast.stmt], device_function: DeviceFunction
) -> list[ast.expr]:
    """The guards of the roots (tile loops) that load ahead of the ring's first
    ringed root (True: an unguarded one, or one guarded by the body's values).
    Launch-state scalar loads and the ringed root's own loads do not count:
    there the fill is the critical path."""
    helpers = {n: b for n, _, b, _ in device_function.triton_outlined_helpers}

    def kinds(node: ast.AST, seen: frozenset[str]) -> Iterator[str]:
        """``node``'s loads and ring steps."""
        if isinstance(node, ast.Call):
            func = node.func
            if _ring_call(node, "ring_acquire") is node:
                yield "step"
            elif _is_load(node):
                yield "load"
            elif isinstance(func, ast.Name) and func.id in helpers.keys() - seen:
                seen |= {func.id}
                for statement in helpers[func.id]:
                    yield from kinds(statement, seen)
        for child in ast.iter_child_nodes(node):
            yield from kinds(child, seen)

    guards: list[ast.expr] = []
    for statement in body:
        found = set(kinds(statement, frozenset()))
        if "step" in found:
            break
        if "load" in found and isinstance(statement, (ast.If, ast.For, ast.While)):
            guarded = isinstance(statement, ast.If) and _names(statement.test) <= {"tl"}
            guards.append(statement.test if guarded else ast.Constant(True))
    return guards


def _any(tests: list[ast.expr]) -> ast.expr:
    if len(tests) == 1:
        return tests[0]
    return (
        create(ast.BoolOp, op=ast.Or(), values=tests) if tests else ast.Constant(False)
    )


def _paced_fill(
    body: list[ast.stmt],
    channels: list[int],
    K: int,
    waits: set[str],
    busy: ast.expr,
) -> list[ast.stmt]:
    """The weight producer's paced first fill (``ring_put_paced``): its puts up to
    its first kept wait on the CTAs ``busy`` holds (owed arrivals flush after each
    put, before each skip and that wait)."""
    if isinstance(busy, ast.Constant) and not busy.value:
        return body  # no first fill to pace

    def flush(full: ast.expr) -> ast.stmt:
        run, paced_ = _beside(full, "run").id, _beside(full, "paced").id
        return statement_from_string(
            f"{run} = {ALIAS}.ring_pace_flush({ast.unparse(full)}, "
            f"{_beside(full, 'pace').id}, {_beside(full, 'step').id}, {_DEPTH}, "
            f"{paced_}, {run}, {K})"
        )

    flushes = [flush(_name(_channel("full", c))) for c in channels]

    def paced(block: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for st in block:
            if (nested := _rebuilt(st, paced)) is not None:
                st = nested
            elif put := _ring_call(st, "ring_put"):
                full = _arg(put, "full")
                state = [_beside(full, "paced"), _beside(full, "run")]
                st = statement_from_string(f"{state[0].id}, {state[1].id} = 0")
                cast("ast.Assign", st).value = _call(
                    "ring_put_paced",
                    [  # pyrefly: ignore [bad-argument-type]
                        *put.args[:2],
                        _beside(full, "pace"),
                        *put.args[2:],
                        *state,
                        clone_ast(busy),
                        ast.Constant(K),
                    ],
                )
            elif skip := _ring_call(st, "ring_skip"):
                out.append(flush(_arg(skip, "full")))
            out.append(st)
        return out

    kept: list[ast.stmt] = []
    issued = False
    for index, st in enumerate(body):
        steps = bool(_ring_calls(st, "ring_put") or _ring_calls(st, "ring_skip"))
        if (issued or steps) and _all_names([st]) & waits:
            kept += [cast("ast.stmt", clone_ast(f)) for f in flushes] if issued else []
            kept += body[index:]
            break
        kept += paced([st]) if steps else [st]
        if _ring_calls(st, "ring_put"):
            kept += [cast("ast.stmt", clone_ast(f)) for f in flushes]
        issued |= steps
    if not any(_ring_calls(st, "ring_put_paced") for st in kept):
        return body
    # A flush right after the same flush has nothing left to forward.
    kept = [
        st
        for i, st in enumerate(kept)
        if not (
            i
            and _ring_call(st, "ring_pace_flush")
            and ast.unparse(st) == ast.unparse(kept[i - 1])
        )
    ]
    counters = [_channel(n, c) for c in channels for n in ("paced", "run")]
    inits = [statement_from_string(f"{c} = tl.full([], 0, tl.int32)") for c in counters]
    return [*inits, *kept]


def _without_mma(body: list[ast.stmt], mma: set[str]) -> list[ast.stmt]:
    """The body without the MMA partition's side of its dots."""
    kept: list[ast.stmt] = []
    for statement in body:
        if nested := _rebuilt(statement, functools.partial(_without_mma, mma=mma)):
            kept.append(nested)
        elif not _mma_side(statement, mma) and not _ring_call(statement, "ring_skip"):
            kept.append(statement)
    return kept


def _with_skips(
    body: list[ast.stmt], skips: dict[str, list[ast.stmt]]
) -> list[ast.stmt]:
    """The body with each step's skips before its count advances."""
    kept: list[ast.stmt] = []
    for statement in body:
        _rebuilt(statement, functools.partial(_with_skips, skips=skips), copy=False)
        for name in (
            _stored_names([statement]) if isinstance(statement, ast.Assign) else ()
        ):
            kept += skips.get(name, [])
        kept.append(statement)
    return kept


def _written_tensors(body: list[ast.stmt], device_function: DeviceFunction) -> set[str]:
    """Tensor arguments the kernel writes (device IR) or hands to impure asm."""
    storage = _argument_allocations(device_function)
    written = _written_allocations()
    for call in (n for s in body for n in ast.walk(s) if isinstance(n, ast.Call)):
        # Outlined helpers' stores are in the device IR.
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr.startswith("gdc_") or _is_tl_call(call, None, _MODULES):
            continue
        if _may_write(call):
            written |= _named_allocations(call, storage)
    return {name for name, key in storage.items() if key in written}


def _scalar_form(statement: ast.stmt, roles: dict[str, SyncRole]) -> ast.stmt:
    """A wait as a one-warp partition runs it: its syncs' scalar forms."""
    if nested := _rebuilt(statement, lambda b: [_scalar_form(s, roles) for s in b]):
        return nested
    targets = _stored_names([statement])
    role = roles.get(next(iter(targets))) if len(targets) == 1 else None
    form = statement if role is None or role.scalar is None else role.scalar
    return cast("ast.stmt", clone_ast(form))


def _is_gdc_wait(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(call := statement.value, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "gdc_wait"
    )


def _partition(
    body: list[ast.stmt], body_slice: _Slice
) -> tuple[list[ast.stmt], set[str]]:
    """The partition's statements, and the tensors it reads that this kernel writes
    without the partition ordering the reads."""
    partition = cast("list[ast.stmt]", clone_ast(body_slice.block(body, set(), False)))
    module = ast.Module(body=partition, type_ignores=[])
    puts = _ring_calls(module, "ring_put")
    # A gated producer's reads of what other roots write follow the kept waits
    # (a tagged operand's mirror reads, their tags); the claims update the ticket.
    claims = _ring_calls(module, "task_put")
    sources = _names(
        ast.Tuple(
            [_arg(c, "desc") for c in [*puts, *_ring_calls(module, "ring_put_paced")]]
            + [_arg(c, "ticket") for c in claims]
            + [_arg(c, "words") for c in _ring_calls(module, "tagged_load")],
            ast.Load(),
        )
    )
    written = _written_tensors(body, body_slice.device_function)
    if body_slice.gated:
        written -= _ordered_tensors(body_slice.device_function)
    return partition, (_names(module) - sources) & written


def _computes_tensors(body: list[ast.stmt]) -> bool:
    """Whether the partition computes tensor values (not just scalars)."""
    for call in (n for s in body for n in ast.walk(s)):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        # (A pointer load is a tensor only through an arange; tagged words and
        # descriptor loads are tensors.)
        tensors = (
            "arange",
            "reshape",
            "broadcast_to",
            "tagged_load",
            "load_tensor_descriptor",
        )
        if call.func.attr in tensors:
            return True
        if call.func.attr in ("full", "zeros") and not (
            call.args and isinstance(call.args[0], ast.List) and not call.args[0].elts
        ):
            return True
    return False


def _move_prefetch_code(
    body: ast.FunctionDef,
    partitions: list[ast.FunctionDef],
    workers: list[tuple[int, bool]],
    mma: int | None,
) -> None:
    """Move the entry code prefetch into the MMA partition (``mma``), idle until operands land."""
    is_prefetch = functools.partial(
        _is_tl_call,
        names=frozenset(["prefetch_code"]),
        modules=("helion_prefetch_code",),
    )
    prefetch = [
        st for st in body.body if isinstance(st, ast.Expr) and is_prefetch(st.value)
    ]
    if mma is None or not prefetch:
        return
    body.body = [st for st in body.body if st not in prefetch]
    # Imported here: importing prefetch_code installs its compiler hook.
    from ...runtime.triton.prefetch_code import prefetch_code_here

    prefetch_here = prefetch_code_here(workers[mma][0])
    partitions[mma].body.insert(0, statement_from_string(prefetch_here))


# ---- regeneration and finalization ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GluonRegeneration(RetryPolicy):
    """Generate a gluon kernel again without what its codegen rejected.

    - Tagged peer transport (``cross_loop_codegen``) is picked from the
      dependency graph; when a precondition only codegen can check fails
      (``TaggedPeerFallback``), the cross-rank pushes keep keyed peer counters
      (``env.tagged_peer_disabled``).
    - A ring the lowering rejects (``RingFallback``) leaves the kernel's loads
      plain (``env.ring_disabled``).

    Each fires at most once per ``generate_ast`` call (its repair switches off
    what raised it).
    """

    signals: tuple[type[BaseException], ...] = (TaggedPeerFallback, RingFallback)

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        env = CompileEnvironment.current()
        saved = env.tagged_peer_disabled, env.ring_disabled
        try:
            yield
        finally:
            env.tagged_peer_disabled, env.ring_disabled = saved

    def repair(
        self, signal: BaseException, config: Config, *, caller_graphs: bool
    ) -> Config | None:
        env = CompileEnvironment.current()
        ring = isinstance(signal, RingFallback)
        flag = "ring_disabled" if ring else "tagged_peer_disabled"
        if getattr(env, flag):
            return None
        what = "gluon ring" if ring else "tagged peer transport"
        log.info("%s disabled: %s", what, signal)
        setattr(env, flag, True)
        return config


REGENERATION = GluonRegeneration()


def finalize_ring(fn_def: ast.FunctionDef, ctx: DeviceBodyContext) -> list[ast.stmt]:
    """With a ring, its partitions and the kernel launching them
    (``GluonBackend.finalize_kernel_def``), ``fn_def`` becoming the body
    partition; returns the statements after it."""
    df = ctx.device_function
    if df.ring is None:
        return []
    if not df.ring.sites:
        raise RingFallback("no TMA load of this kernel can stream through it")
    return [st for st in df.ring.emit(fn_def) if st is not fn_def]
