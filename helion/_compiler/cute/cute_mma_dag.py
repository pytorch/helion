"""The ``cute_mma_dag`` lowering: a warp-level MMA dataflow owner for root tiles.

``cute_mma_dag="warp"`` replaces the SIMT body of a root tile whose device
loop is a dataflow of ``hl.dot`` calls over fp32 ``Frag`` accumulators -- a
*serial state recurrence* ``carry = dot(A_i, B_i, acc=rescale(carry))`` with
per-iteration stores of the carry, or a K-tile loop accumulating several
fragments that an epilogue of further dots (fragments as operands, masks,
rescales) and stores consumes -- by one CTA of ``cute_mma_dag_warps`` warps
that keeps the fragments in ``mma.sync`` C layout, stages the per-iteration
operand tiles into XOR-swizzled shared memory (``cp.async`` for provably
aligned direct loads, cooperative packet passes otherwise) and runs
``ldmatrix`` + ``mma.sync.m16n8k16`` over the static K extents.  The delta-rule
state kernels add fragments as operands inside the loop (the carry cast to
bf16 as a B operand of the
same iteration that accumulates into it), *derived* fragments (pointwise
values over several fragments and tiles loaded at the fragment coordinates:
``vnew = u - dot``, ``dS = dS - dot``), mid-iteration stores of derived
values and the orientation rule (a no-acc dot consumed only transposed is
emitted transposed, so its mask relabels by coordinates and no exchange is
needed).  Later increments (Frag reductions, tf32) extend the typed IR below
rather than rewrite it.

The module is split in three layers:

* the *typed IR* (:class:`MmaDagProgram`): value kinds ``Scalar`` (uniform
  0-d), ``Vec`` (1-D along the M, N or K role), ``Smem`` operand tiles
  (:class:`OperandTile`) and the ``Frag`` carry, built from the FX graphs by
  :func:`match_mma_dag` without looking at any config (the autotuner heuristic
  runs it on ``device_ir`` to decide admission);
* the *planner* (:func:`plan_mma_dag`): per config, assigns the warp layouts
  (M-split-first or N-split-first per fragment shape, chosen to feed as many
  fragment operands from registers as possible), checks the register /
  shared-memory budgets and the operand routes and records a
  :class:`MmaDagPlan` on ``cute_state``;
* the *emitter* (``cute_mma_dag_codegen.py``): renders the device body as
  source text from the plan, like ``cute_warp_mma_gemm.py`` does.

Admission fails closed: anything the matcher does not understand is "not
mine" (the existing lowering runs unchanged and the knobs never appear); a
config that selects the mode on an admitted kernel but cannot run it raises
``BackendUnsupported`` / ``InvalidConfig``, never a silent SIMT fallback.
"""

from __future__ import annotations

import dataclasses
import itertools
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from ... import exc
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from .cute_warp_mma_gemm import warp_mma_warp_layout

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_ir import GraphInfo
    from .cute_mma_dag_tcgen05 import Tcgen05Facts
    from .cute_mma_dag_tcgen05 import Tcgen05Plan

    Layouts = dict[tuple[int, int], tuple[int, int]]

MMA_DAG_KEY = "cute_mma_dag"
MMA_DAG_OFF = "off"
MMA_DAG_WARP = "warp"
# The TMEM-resident mode of the same owner (cute_mma_dag_tcgen05.py); it
# joins the knob's choices only where its facts admit it.
MMA_DAG_TCGEN05 = "tcgen05"
MMA_DAG_MODES: tuple[str, ...] = (MMA_DAG_OFF, MMA_DAG_WARP)
MMA_DAG_WARPS_KEY = "cute_mma_dag_warps"
MMA_DAG_WARP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
MMA_DAG_DEFAULT_WARPS = 4
MMA_DAG_STAGES_KEY = "cute_mma_dag_stages"
MMA_DAG_STAGE_CHOICES: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
# Four stages hide the per-iteration load latency of the serial chain (s2
# fwd_h 64x64 x 8 warps: 4 stages 128 us, 2 stages 227 us, 1 stage 273 us);
# with one CTA per SM eight keep the ring full (fwd_h 132 us, bwd_dh 258
# against 263), so the seeds try eight first where the shared memory allows.
MMA_DAG_DEFAULT_STAGES = 4
MMA_DAG_SEED_STAGES: tuple[int, ...] = (8, 4, 2)
# Operand routes: ``direct`` (cp.async packets straight into the swizzled
# tile), ``inplace`` (cp.async of the raw tile, then each thread rewrites its
# own packets through the pointwise DAG after its wait) and ``sync`` (a
# cooperative pass of synchronous global loads for unaligned sources).
ROUTE_DIRECT = "direct"
ROUTE_INPLACE = "inplace"
# ``convert``: cp.async of a wider (fp32) source tile into a raw ring slot,
# then each thread rewrites its own packets into the 16-bit operand tile.
ROUTE_CONVERT = "convert"
ROUTE_SYNC = "sync"
MMA_DAG_STORE_KEY = "cute_mma_dag_store"
MMA_DAG_STORE_STAGED = "staged"
MMA_DAG_STORE_DIRECT = "direct"
# ``staged`` writes the carry tile through a double-buffered shared tile as
# 16-byte packets (the mma.sync C layout alone stores eight scattered 16-byte
# row segments per warp instruction); ``direct`` stores the fragment pairs.
MMA_DAG_STORE_CHOICES: tuple[str, ...] = (MMA_DAG_STORE_STAGED, MMA_DAG_STORE_DIRECT)
MMA_DAG_EXCHANGE_KEY = "cute_mma_dag_exchange"
MMA_DAG_EXCHANGE_AUTO = "auto"
MMA_DAG_EXCHANGE_SMEM = "smem"
# ``auto`` feeds a fragment to a dot through registers where the warp layouts
# allow (the identity repack of a same-orientation A operand) and through a
# shared bf16 tile otherwise; ``smem`` always exchanges through shared memory
# (same numerics; the A/B lever for the register routes).
MMA_DAG_EXCHANGE_CHOICES: tuple[str, ...] = (
    MMA_DAG_EXCHANGE_AUTO,
    MMA_DAG_EXCHANGE_SMEM,
)
MMA_DAG_PASS_WARPS_KEY = "cute_mma_dag_pass_warps"
MMA_DAG_PASS_WARPS_OFF = "off"
MMA_DAG_PASS_WARPS_ON = "on"
# tcgen05 at eight warps with loop operand passes: ``on`` has warps 4-7
# rewrite the operand tiles one tile ahead of the epilogue warps' chain
# (handing each over through a per-stage mbarrier); ``off`` runs the passes
# on the chain, the extra warps draining the stores.
MMA_DAG_PASS_WARPS_CHOICES: tuple[str, ...] = (
    MMA_DAG_PASS_WARPS_OFF,
    MMA_DAG_PASS_WARPS_ON,
)
# Ring depth the pass-warp seed asks for: the next tile must land before
# its pass is due (at two slots the TMA of tile i + 1 goes out with
# iteration i and the pass warps wait on it: gated fwd_h_delta at s2 305 us
# against 268 at three).
MMA_DAG_PASS_WARP_STAGES = 3
# Pass work per pass-warp thread (16-byte packets per iteration) from which
# the pass-warp seed leads for a one-wave grid: measured at s2 on a B200,
# gated fwd_h_delta's k pass of 8 packets 338 -> 270 us (s1 70 -> 62, s0
# 31 -> 24; at bdv 64 451 -> 411), gated dstate's two 2-packet passes 548 ->
# 619, bwd_dh's 4-packet q*scale 180 -> 198 (the handoff costs ~90 cycles an
# iteration); the CTA-rich K-loop kernels (dqk, dv: 8 packets, grids far
# beyond the SM count) lose a CTA per SM to the pass seed's ring depth
# (318 -> 430, 186 -> 289) and stay with the autotuner.
MMA_DAG_PASS_WARP_PACKETS = 8
MMA_DAG_LOADS_KEY = "cute_mma_dag_loads"
MMA_DAG_LOADS_TMA = "tma"
MMA_DAG_LOADS_LANE = "lane"
# The tensor-memory region programs' derived values read 2-D tiles at the
# fragment coordinates (a fresh accumulator's seed, a reduction's operand):
# ``tma`` fetches them into the loop ring beside the operand tiles (one
# 16-byte shared load per chunk; the tiles that do not fit the shared memory
# stay lane loads), ``lane`` keeps every lane's global pair loads (one sector
# per lane per load) and the shared memory for occupancy.  Inert elsewhere.
MMA_DAG_LOADS_CHOICES: tuple[str, ...] = (MMA_DAG_LOADS_TMA, MMA_DAG_LOADS_LANE)
# The driver reserves this much shared memory per resident CTA: an SM's
# ``capacity + reserved`` bytes hold ``(capacity + reserved) // (smem +
# reserved)`` CTAs of ``smem`` bytes (B200: 233472 B per SM, 232448 per CTA).
MMA_DAG_SMEM_CTA_RESERVED = 1024
# Tensor memory columns of one SM (the tcgen05 mode's second occupancy limit).
MMA_DAG_TMEM_COLUMNS = 512
MMA_DAG_KEYS: tuple[str, ...] = (
    MMA_DAG_KEY,
    MMA_DAG_WARPS_KEY,
    MMA_DAG_STAGES_KEY,
    MMA_DAG_STORE_KEY,
    MMA_DAG_EXCHANGE_KEY,
    MMA_DAG_PASS_WARPS_KEY,
    MMA_DAG_LOADS_KEY,
)
# Live fp32 fragment words per thread the carry may occupy.
MMA_DAG_MAX_FRAG_REGS = 128
# Fragments plus one dot's operand words (repacked A, one K step of B).
MMA_DAG_MAX_LIVE_REGS = 192
# Tile ladder of the seed heuristic for the tunable M / N axes.
MMA_DAG_TILE_LADDER: tuple[int, ...] = (16, 32, 64, 128)
MMA_DAG_MIN_BM = 16
MMA_DAG_MIN_BN = 8
MMA_DAG_K_STEP = 16
_OPERAND_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
# fp32 operands run as tf32 (``mma.sync.m16n8k8.tf32``, inputs truncated to
# tf32 like Triton's ``input_precision='tf32'``) only under that setting.
_TF32_DTYPE = torch.float32


def _operand_dtypes() -> tuple[torch.dtype, ...]:
    from .mma_support import cute_fp32_dot_uses_tf32

    if cute_fp32_dot_uses_tf32():
        return (*_OPERAND_DTYPES, _TF32_DTYPE)
    return _OPERAND_DTYPES


# Scalar / vector side loads (decays, masks, scales) the per-lane loads read.
_AUX_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)
# Side values of at most one tile dim may also be int32 (window bounds a
# mask compares against).
_SIDE_DTYPES: tuple[torch.dtype, ...] = (*_AUX_DTYPES, torch.int32)
_STORE_DTYPES: tuple[torch.dtype, ...] = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
)
AXIS_ROOT = "root"
AXIS_RDIM = "rdim"
AXIS_LOOP = "loop"
AXIS_LEAD = "lead"


# ---------------------------------------------------------------------------
# Fragment coordinate model (pure functions; the CPU tests prove them)
# ---------------------------------------------------------------------------


def mma_dag_frag_regs(bm: int, bn: int, warps: int) -> int:
    """fp32 words per thread of one ``bm x bn`` fragment over ``warps`` warps."""
    return (bm * bn) // (32 * warps)


def accumulator_coordinates(
    *, lane: int, am: int, an: int, j: int, wmi: int, wni: int, wtm: int, wtn: int
) -> tuple[int, int]:
    """Tile-local ``(row, col)`` of register ``j`` of atom ``(am, an)`` in
    lane ``lane`` of the warp at ``(wmi, wni)`` of the warp grid.

    ``mma.sync.m16n8k16`` C layout: lane ``(g, c) = (lane // 4, lane % 4)``
    holds rows ``g`` / ``g + 8`` (``j < 2`` / ``j >= 2``) and columns ``2c``
    / ``2c + 1`` (``j`` even / odd) of each atom.
    """
    g, c = lane // 4, lane % 4
    row = wmi * wtm + 16 * am + g + 8 * (j >= 2)
    col = wni * wtn + 8 * an + 2 * c + (j & 1)
    return row, col


def ldmatrix_lane_matrix_halves(role: str, lane: int) -> tuple[int, int]:
    """``(mn_half, k_half)`` of the 8x8 matrix lane ``lane`` supplies a row
    address for in one ``ldmatrix.x4``.

    The A fragment orders its four words (rows 0-7, k 0-7), (rows 8-15, k
    0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15); two adjacent B atoms order
    theirs (atom, k 0-7), (atom, k 8-15), (atom + 1, k 0-7), (atom + 1, k
    8-15).  Lanes ``8q .. 8q + 7`` address matrix ``q``.
    """
    q = lane // 8
    if role == "a":
        return q % 2, q // 2
    return q // 2, q % 2


def ldmatrix_lane_address(
    *,
    role: str,
    trans: bool,
    lane: int,
    mn_base: int,
    k_step: int,
    elem_bytes: int = 2,
) -> tuple[int, int]:
    """Tile-local ``(row, chunk)`` of the 16-byte packet lane ``lane``
    addresses for one ``ldmatrix.x4`` of operand ``role`` ("a" / "b").

    ``trans`` says the shared tile stores the operand with K along its rows
    (its columns run along M / N) and is read with ``ldmatrix.trans``;
    otherwise K runs along the columns.  ``mn_base`` is the first M / N
    index of the atom (pair) and ``k_step`` the 16-wide K step.
    """
    l8 = lane % 8
    mn_half, k_half = ldmatrix_lane_matrix_halves(role, lane)
    per_chunk = 16 // elem_bytes
    if trans:
        row = 16 * k_step + l8 + 8 * k_half
        chunk = (mn_base + 8 * mn_half) // per_chunk
    else:
        row = mn_base + l8 + 8 * mn_half
        chunk = (16 * k_step + 8 * k_half) // per_chunk
    return row, chunk


def emulate_ldmatrix(
    tile: Sequence[Sequence[int]],
    *,
    trans: bool,
    addresses: Sequence[tuple[int, int]],
    elem_bytes: int = 2,
) -> list[list[list[int]]]:
    """Model ``ldmatrix.x4[.trans]`` on an index tile.

    ``tile[row][col]`` holds the element identities of a shared tile whose
    columns are contiguous; ``addresses[lane]`` is the ``(row, chunk)`` each
    lane supplies.  Returns ``[lane][word][half]`` element identities: lane
    ``(g, c)`` receives row ``g``, columns ``2c``, ``2c + 1`` of each 8x8
    matrix (rows ``2c``, ``2c + 1`` of column ``g`` with ``.trans``).
    """
    per_chunk = 16 // elem_bytes
    result: list[list[list[int]]] = []
    for lane in range(32):
        g, c = lane // 4, lane % 4
        words: list[list[int]] = []
        for matrix in range(4):
            base_row, base_chunk = addresses[8 * matrix]
            base_col = base_chunk * per_chunk
            for row_lane in range(8):
                row, chunk = addresses[8 * matrix + row_lane]
                assert (row, chunk) == (base_row + row_lane, base_chunk), (
                    "the eight lanes of one matrix must address consecutive rows"
                )
            if trans:
                words.append(
                    [tile[base_row + 2 * c + h][base_col + g] for h in range(2)]
                )
            else:
                words.append(
                    [tile[base_row + g][base_col + 2 * c + h] for h in range(2)]
                )
        result.append(words)
    return result


def frag_word_blocks(
    role: str, route: str, atom: int, k_step: int
) -> list[tuple[int, int, int]]:
    """The producer register blocks ``(am, an, half)`` packed into the words
    of a register-routed operand fragment (the mirror of the emitter's
    ``emit_frag_words``): for A (``atom`` = the consumer's atom row) the
    four words of ``k_step``; for B (``atom`` = the consumer's atom column)
    the two words, K block ``kb`` being the producer's row block ``(kb // 2,
    kb % 2)`` of n-atom ``atom`` when the ``[K, N]`` fragment is transposed
    by ``movmatrix``, or n-atom ``kb`` of row block ``(atom // 2, atom % 2)``
    when an ``[N, K]`` fragment is repacked as is.
    """
    if role == "a":
        return [
            (atom, 2 * k_step, 0),
            (atom, 2 * k_step, 1),
            (atom, 2 * k_step + 1, 0),
            (atom, 2 * k_step + 1, 1),
        ]
    blocks = []
    for kb in (2 * k_step, 2 * k_step + 1):
        if route == ROUTE_MOVMATRIX:
            blocks.append((kb // 2, atom, kb % 2))
        else:
            blocks.append((atom // 2, kb, atom % 2))
    return blocks


def swizzled_chunk(row: int, chunk: int, chunks_per_row: int) -> int:
    """Python model of ``cute_warp_mma_gemm._swizzled_chunk``."""
    if chunks_per_row >= 8:
        return chunk ^ (row % 8)
    if chunks_per_row == 1:
        return chunk
    rows_per_group = 8 // chunks_per_row
    return chunk ^ ((row // rows_per_group) % chunks_per_row)


def transposed_swizzled_chunk(row: int, chunk: int, chunks_per_row: int) -> int:
    """Python model of ``cute_mma_dag_codegen._transposed_swizzle``."""
    if chunks_per_row >= 8:
        return chunk ^ (row % 8) ^ ((row // 8) % 8)
    return swizzled_chunk(row, chunk, chunks_per_row)


# ---------------------------------------------------------------------------
# Typed IR
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Axis:
    """One tiled dimension of the kernel.

    ``root`` axes are the root grid's tile dims (block size from the config),
    ``loop`` a device loop's tile (a dot K extent per iteration, or an inner
    tile the loop's fragments are sized by), ``rdim`` a static ``:`` extent
    and ``lead`` a root block of size one (a batch coordinate, never a
    fragment dim).  ``extent`` is the full dim; ``loop`` the index (into
    ``MmaDagProgram.loops``) of the device loop a ``loop`` axis belongs to.
    """

    index: int
    block_id: int
    kind: str
    extent: int
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class Subscript:
    """One subscript position of a tensor access: a whole tile of ``axis``
    (``root`` / ``loop`` / ``lead`` tiles and ``:`` rdims) or a uniform
    integer expression ``expr`` over grid loop indices and the leading tile
    ids (``loops`` lists the loops whose index occurs in it;
    ``loop_dependent`` says there is at least one).  An ``axis`` subscript
    may carry a loop-invariant uniform ``offset`` over the leading tile ids
    (``x[tile_k.index + tile_c.begin * C]``): the tile's elements sit at
    ``offset`` plus the axis position, proven in bounds by the matcher."""

    kind: str  # "axis" | "scalar"
    axis: Axis | None = None
    expr: sympy.Expr | None = None
    loop_dependent: bool = False
    loops: tuple[int, ...] = ()
    offset: sympy.Expr | None = None

    @property
    def loop_set(self) -> frozenset[int]:
        """The loops this subscript varies with."""
        if self.kind == "axis":
            assert self.axis is not None
            return frozenset({self.axis.loop} if self.axis.loop >= 0 else ())
        return frozenset(self.loops)


@dataclasses.dataclass(frozen=True)
class Access:
    """A load or store of a tensor argument at admitted subscripts.

    ``dims`` are the axes of the accessed value's dims in positional order
    (one per non-scalar subscript, lead axes included); ``tile_dims`` the
    fragment-sized ones (lead axes dropped).
    """

    node: Node
    fake: torch.Tensor
    name: str
    strides: tuple[int, ...]
    subscripts: tuple[Subscript, ...]
    dims: tuple[Axis, ...]
    # Element stride of the tensor dim behind each value dim (positional:
    # the two ``:`` dims of ``w[idx, i, :, :]`` share one axis when D == C).
    dim_strides: tuple[int, ...] = ()

    @property
    def dtype(self) -> torch.dtype:
        return self.fake.dtype

    @property
    def tile_dims(self) -> tuple[Axis, ...]:
        return tuple(axis for axis in self.dims if axis.kind != AXIS_LEAD)

    @property
    def tile_dim_strides(self) -> tuple[int, ...]:
        """Element strides of ``tile_dims`` (lead dims dropped)."""
        return tuple(
            stride
            for axis, stride in zip(self.dims, self.dim_strides, strict=True)
            if axis.kind != AXIS_LEAD
        )

    @property
    def repeated_axes(self) -> bool:
        """Whether one axis indexes two dims (a ``[C, C]`` tile over one rdim)."""
        return len(set(self.dims)) != len(self.dims)

    def stride_of(self, axis: Axis) -> int:
        """Element stride of the tensor dim ``axis`` indexes (the axis must
        index exactly one dim; positional lookups use ``stride_at``)."""
        strides = [
            stride
            for subscript, stride in zip(self.subscripts, self.strides, strict=True)
            if subscript.kind == "axis" and subscript.axis is axis
        ]
        if len(strides) != 1:
            raise KeyError(axis)
        return strides[0]

    def subscript_index(self, position: int) -> int:
        """The subscript (tensor dim) behind value dim ``position``."""
        axis_positions = [
            index
            for index, subscript in enumerate(self.subscripts)
            if subscript.kind == "axis"
        ]
        return axis_positions[position]

    def stride_at(self, position: int) -> int:
        """Element stride of the tensor dim behind value dim ``position``."""
        return self.dim_strides[position]

    @property
    def tile_positions(self) -> tuple[int, ...]:
        """Value dim positions of the fragment-sized dims."""
        return tuple(
            index for index, axis in enumerate(self.dims) if axis.kind != AXIS_LEAD
        )

    @property
    def loops(self) -> frozenset[int]:
        """The device loops the access varies with (empty: loop-invariant)."""
        result: frozenset[int] = frozenset()
        for subscript in self.subscripts:
            result |= subscript.loop_set
        return result


@dataclasses.dataclass(frozen=True)
class Operand:
    """One dot operand: the shared tile it is read from and how.

    The tile is either a 2-D load (``load``; ``pointwise`` says a pointwise
    DAG sits between the load and the operand value) or a fragment
    (``frag``: a carry, a finished accumulator, an earlier dot result or a
    derived value).  ``swap`` says the operand *value* is the source read
    transposed (for a fragment: relative to the registers, so a transposed
    producer is folded in); ``flip`` says the dot is emitted transposed and
    the operand's logical ``(M, K)`` / ``(K, N)`` dims are the value's dims
    swapped; ``trans`` says the operand's K dim runs along the shared tile's
    rows, so it is read with ``ldmatrix.trans``.  ``aux_loads`` are the
    vector / scalar loads the pointwise DAG reads.
    """

    role: str  # "a" | "b"
    kind: str  # "load" | "frag"
    value: Node
    load: Access | None
    frag: Node | None
    pointwise: bool
    swap: bool
    rows: Axis
    cols: Axis
    trans: bool
    aux_loads: tuple[Access, ...]
    # Exchange tiles are shared between operands with the same source and
    # the same pointwise chain (``dA`` read straight and transposed).
    chain_key: tuple[object, ...]
    flip: bool = False
    # Position of the shared tile's rows among the source's two tile dims
    # (the columns are the other one); the dims may share one axis.
    rows_pos: int = 0
    # The shared tile holds the operand transposed relative to its source
    # (an fp32 B operand whose K would run along the rows: stored [N, K] so
    # ``ldmatrix`` serves the tf32 fragments; the pass / exchange writes
    # scatter).  ``rows`` / ``cols`` and ``trans`` describe the stored tile.
    transposed_tile: bool = False

    @property
    def cols_pos(self) -> int:
        return 1 - self.rows_pos

    @property
    def logical_swap(self) -> bool:
        """Whether the logical operand is the registers / tile transposed:
        a fragment ``[K, M]`` feeding A or ``[N, K]`` feeding B."""
        return self.swap != self.flip

    @property
    def dtype(self) -> torch.dtype:
        return cast("torch.Tensor", self.value.meta["val"]).dtype


@dataclasses.dataclass(frozen=True)
class Dot:
    """``result = dot(a, b, acc)`` with its fragment slot assignment.

    ``acc`` is None, a fragment leaf, or a pointwise DAG over one fragment
    leaf (``acc_frag``); ``slot`` is the register slot of the result (the
    accumulator's own slot when the dot accumulates in place).
    ``transposed`` says the orientation rule applies: the registers hold the
    transposed product (``rows`` / ``cols`` are the emitted dims and ``a`` /
    ``b`` the FX operands with their roles swapped), because every consumer
    reads the result transposed.
    """

    node: Node
    a: Operand
    b: Operand
    acc: Node | None
    acc_frag: Node | None
    acc_slot: int | None
    rows: Axis
    cols: Axis
    k: Axis
    region: str  # "loop" | "epilogue"
    slot: int
    transposed: bool = False
    # The device loop whose body holds the dot (-1: the root region).
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class Derived:
    """A fragment defined by a pointwise DAG over several fragment leaves
    (``frags``, all in the slot's layout) and 2-D tiles loaded at the
    fragment coordinates (``loads``): ``vnew = u - dot(w, h)``, ``dS = dS -
    dot(w^T, dv)``, or over tiles alone (``dk = dk_as + d_ba * beta``, a
    fresh accumulator seed).  ``slot`` is the register slot it is
    materialized in (an input's slot when that input dies here)."""

    node: Node
    slot: int
    frags: tuple[Node, ...]
    loads: tuple[Access, ...]
    aux_loads: tuple[Access, ...]
    rows: Axis
    cols: Axis
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class Store:
    """A store of a fragment-derived value; ``frag`` is the fragment leaf,
    ``swap`` says the store's tile dims are the fragment's swapped."""

    access: Access
    value: Node
    frag: Node
    swap: bool
    region: str
    # Loop stores: "carry_in" (the carry's value at loop entry, hoisted to
    # the iteration top) or "body" (a dot / derived value at its program
    # point); "epilogue" after the loop.
    source: str
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class Carry:
    """One loop-carried fragment: its init (``init``: a 2-D load; ``init_frag``:
    a fragment of the enclosing region, the accumulator of an earlier loop
    carried on, or with ``init_value`` a pointwise value over it evaluated
    per register; neither: zeros), the body placeholder, the body output
    (copied into the slot at the iteration end when computed elsewhere) and
    the phi after the loop."""

    slot: int
    init: Access | None
    placeholder: Node
    output: Node
    phi: Node
    rows: Axis
    cols: Axis
    init_frag: Node | None = None
    init_value: Node | None = None
    # The scalar / vector loads the init value reads.
    init_aux: tuple[Access, ...] = ()


@dataclasses.dataclass(frozen=True)
class VecCarry:
    """A loop-carried fp32 vector (``[1, C]``, varying along one axis) in
    vector slot ``slot``: zeros at entry, or an earlier loop's vector
    (``init_vec``) carried on.  An ``identity`` carry passes through the
    loop unchanged (a gradient accumulator of a decay mode the kernel does
    not run) and needs no registers (``slot`` is -1)."""

    slot: int
    placeholder: Node
    init: Node
    output: Node
    phi: Node
    identity: bool
    init_vec: Node | None = None


@dataclasses.dataclass(frozen=True)
class VecSlot:
    """A vector register set: ``axis`` is the dim it varies along,
    ``orient`` whether that is the rows (``sum(-1)``) or the columns
    (``sum(-2)``) of the fragments reduced into it; a ``scalar`` slot (no
    axis) is one value every lane holds."""

    axis: Axis | None
    orient: str  # "rows" | "cols" | "scalar"


@dataclasses.dataclass(frozen=True)
class Reduce:
    """``sum(value, -1)`` / ``sum(value, -2)`` of a pointwise value over one
    fragment leaf (``frag``, in the registers; aux loads beside it) into
    vector slot ``slot``: each lane sums its registers along the reduced
    dim, the quad / octet shuffles complete the warp's part and the warp
    grid's other split goes through shared memory.  With ``vec`` set the
    value is a vector reduced to a scalar."""

    node: Node
    value: Node
    frag: Node | None
    orient: str
    slot: int
    aux_loads: tuple[Access, ...]
    loop: int = -1
    vec: Node | None = None


@dataclasses.dataclass(frozen=True)
class VecOp:
    """A pointwise vector value over vector leaves (``vecs``: carries,
    reductions, earlier ops) and aux loads, materialized in vector slot
    ``slot`` (an input's when that input dies here)."""

    node: Node
    slot: int
    vecs: tuple[Node, ...]
    aux_loads: tuple[Access, ...]
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class VecStore:
    """A store of a vector value (``[1, C]`` fp32): the lanes holding each
    element once write it (``c == 0`` of the first warp column for a row
    vector); thread 0 stores a scalar."""

    access: Access
    value: Node
    vec: Node
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class Loop:
    """One device loop of the program and its body region.

    ``axes`` are the loop's tile axes (empty for ``hl.grid`` / fixed-step
    loops, whose index is ``symbol``), ``carries`` its fragment carries,
    ``steps`` the body in program order (dots, derived values, stores and
    nested loops).  An ``innermost`` loop stages its operand tiles through
    the cp.async ring; an outer loop issues its tiles once per iteration.
    """

    index: int
    parent: int
    depth: int
    node: Node
    graph_id: int
    axes: tuple[Axis, ...]
    symbol: sympy.Symbol | None
    begin: object
    end: object
    carries: tuple[Carry, ...]
    vec_carries: tuple[VecCarry, ...]
    steps: tuple[Step, ...]
    innermost: bool

    @property
    def carry_placeholders(self) -> frozenset[Node]:
        return frozenset(carry.placeholder for carry in self.carries)

    @property
    def small_carries(self) -> tuple[VecCarry, ...]:
        """The vector carries (the name the single-loop gate uses)."""
        return self.vec_carries

    @property
    def nested(self) -> tuple[Loop, ...]:
        return tuple(step for step in self.steps if isinstance(step, Loop))

    @property
    def static_trip(self) -> int | None:
        if isinstance(self.begin, int) and isinstance(self.end, int):
            return self.end - self.begin
        return None


Step = Dot | Derived | Store | Reduce | VecOp | VecStore | Loop


@dataclasses.dataclass(frozen=True)
class MmaDagProgram:
    """The matched MMA dataflow program (config-free).

    The root region ``body`` holds the root graph's steps in program order
    (loops included); ``loops`` lists every device loop in pre-order.  The
    single-loop fields (``loop_node`` .. ``epilogue_order``) describe the
    single-loop shape (one loop, an epilogue) for the tcgen05 mode, which
    admits only ``simple`` programs.
    """

    root_graph_id: int
    loop_graph_id: int
    loop_node: Node
    loop_axis: Axis | None
    loop_block_id: int
    loop_symbol: sympy.Symbol | None
    begin: object
    end: object
    axes: tuple[Axis, ...]
    carries: tuple[Carry, ...]
    dots: tuple[Dot, ...]
    stores: tuple[Store, ...]
    # Fragment slot -> (rows, cols) axes, carries first.
    slots: tuple[tuple[Axis, Axis], ...]
    # Fragment leaf node -> slot (placeholders, phis, loop outputs, dots,
    # derived values).
    frag_slots: dict[Node, int]
    # The (first) loop body in program order: dots, derived values and body
    # stores (carry snapshots are hoisted to the iteration top by the
    # emitter).
    steps: tuple[Dot | Derived | Store, ...] = ()
    derived: tuple[Derived, ...] = ()
    loops: tuple[Loop, ...] = ()
    body: tuple[Step, ...] = ()
    # Loop placeholders that alias a value of the enclosing region (read-only
    # loop inputs) -> that value.
    aliases: dict[Node, Node] = dataclasses.field(default_factory=dict)
    # Fragment leaf (dot, derived value, carry placeholder, phi) -> the
    # region defining it (loop index; -1: the root region).
    frag_regions: dict[Node, int] = dataclasses.field(default_factory=dict)
    # Vector slots and the vector leaves (carry placeholders, phis,
    # reductions, vector ops) in them.
    vec_slots: tuple[VecSlot, ...] = ()
    vec_leaves: dict[Node, int] = dataclasses.field(default_factory=dict)
    reduces: tuple[Reduce, ...] = ()
    vec_ops: tuple[VecOp, ...] = ()
    vec_stores: tuple[VecStore, ...] = ()
    # Vector slot -> the fragment slots whose warp split it follows (the
    # component of carries / ops / reductions it belongs to).
    vec_frag_slots: dict[int, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    # ``tile.index`` values (``tile.index[None, :]``) -> their root axis.
    tile_index_loads: dict[Node, Axis] = dataclasses.field(default_factory=dict)

    @property
    def dtype(self) -> torch.dtype:
        return self.dots[0].a.dtype

    @property
    def elem_bytes(self) -> int:
        return self.dtype.itemsize

    @property
    def simple(self) -> bool:
        """The single-loop shape: one innermost loop over at most one tile axis
        carrying fragments, every loop operand staged per iteration, derived
        values over fragments, then an epilogue."""
        if len(self.loops) != 1:
            return False
        loop = self.loops[0]
        return (
            loop.innermost
            and len(loop.axes) <= 1
            and not loop.small_carries
            and bool(loop.carries)
            and all(
                self.is_ring(dot, operand)
                for dot in self.dots
                if dot.loop == 0
                for operand in (dot.a, dot.b)
                if operand.load is not None
            )
            and all(derived.frags for derived in self.derived)
        )

    @property
    def lead_axes(self) -> tuple[Axis, ...]:
        return tuple(axis for axis in self.axes if axis.kind == AXIS_LEAD)

    @property
    def tunable_axes(self) -> tuple[Axis, ...]:
        return tuple(axis for axis in self.axes if axis.kind in (AXIS_ROOT, AXIS_LOOP))

    def dots_in(self, region: str) -> tuple[Dot, ...]:
        return tuple(dot for dot in self.dots if dot.region == region)

    def stores_in(self, region: str) -> tuple[Store, ...]:
        return tuple(store for store in self.stores if store.region == region)

    @property
    def carry_placeholders(self) -> frozenset[Node]:
        return frozenset(carry.placeholder for carry in self.carries)

    def frag_operands(self, region: str) -> tuple[tuple[Dot, Operand], ...]:
        return tuple(
            (dot, operand)
            for dot in self.dots_in(region)
            for operand in (dot.a, dot.b)
            if operand.kind == "frag"
        )

    def early_exchange(self, dot: Dot, operand: Operand) -> bool:
        """Whether a fragment operand of a loop dot reads that loop's carry
        at loop entry: its exchange tile is written before the iteration
        barrier (double-buffered by iteration parity) instead of behind its
        own."""
        return (
            dot.loop >= 0
            and operand.kind == "frag"
            and operand.frag in self.loops[dot.loop].carry_placeholders
        )

    def level(self, access: Access) -> int:
        """The innermost loop ``access`` varies with (-1: loop-invariant)."""
        loops = access.loops
        if not loops:
            return -1
        return max(loops, key=lambda index: self.loops[index].depth)

    def is_ring(self, dot: Dot, operand: Operand) -> bool:
        """Whether a load operand rides its dot's cp.async ring: the dot sits
        in an innermost loop and the tile varies with that loop.  Other
        loads are staged once per iteration of the loop they vary with (or
        once per CTA)."""
        return (
            operand.load is not None
            and dot.loop >= 0
            and self.loops[dot.loop].innermost
            and self.level(operand.load) == dot.loop
        )

    def ring_derived_load(self, derived: Derived, access: Access) -> bool:
        return (
            derived.loop >= 0
            and self.loops[derived.loop].innermost
            and self.level(access) == derived.loop
        )

    def loop_has_ring(self, loop: int) -> bool:
        """Whether loop ``loop`` stages tiles through the cp.async ring (its
        exchange tiles of the carry at entry are then double-buffered by
        iteration parity; without a ring the trailing barrier of every
        iteration orders them)."""
        return any(
            self.is_ring(dot, operand)
            for dot in self.dots
            if dot.loop == loop
            for operand in (dot.a, dot.b)
        ) or any(
            self.ring_derived_load(derived, access)
            for derived in self.derived
            if derived.loop == loop
            for access in derived.loads
        )

    def staged_ahead_of_loops(self, key: tuple[object, ...]) -> bool:
        """Whether the once-per-CTA staged tile with sharing key ``key`` is
        first read -- so its per-element pass runs -- ahead of the first
        loop's ring prologue: by a root dot before the first loop or by a
        dot inside that loop (the emitter settles a segment's tiles before
        running it).  The raw source tile of such a converted operand is
        dead once the loops start and may overlay their frames."""
        first_loop = next(
            (step.index for step in self.body if isinstance(step, Loop)), None
        )
        for step in self.body:
            if isinstance(step, Dot):
                if any(operand.chain_key == key for operand in (step.a, step.b)):
                    return True
            elif isinstance(step, Loop):
                return any(
                    operand.chain_key == key
                    for dot in self.dots
                    if dot.loop >= 0 and first_loop in self.enclosing_loops(dot.loop)
                    for operand in (dot.a, dot.b)
                )
        return False

    def read_within_one_root_segment(self, key: tuple[object, ...]) -> bool:
        """Whether every dot reading the operand tile with sharing key ``key``
        is a root dot, all in one run of root steps between two loops: a
        tile written for them is dead once the next loop starts and may
        overlay the loops' frames."""
        readers = [
            dot
            for dot in self.dots
            if any(operand.chain_key == key for operand in (dot.a, dot.b))
        ]
        if not readers or any(dot.loop >= 0 for dot in readers):
            return False
        positions = [
            index
            for index, step in enumerate(self.body)
            if isinstance(step, Dot) and any(step.node is dot.node for dot in readers)
        ]
        if len(positions) != len(readers):
            return False
        return not any(
            isinstance(step, Loop)
            for step in self.body[min(positions) : max(positions)]
        )

    def enclosing_loops(self, loop: int) -> list[int]:
        """Loop ``loop`` and the loops it is nested in, innermost first."""
        result = []
        while loop >= 0:
            result.append(loop)
            loop = self.loops[loop].parent
        return result

    def operand_frame(self, dot: Dot, operand: Operand) -> int:
        """The shared-memory frame of the operand's tile: a ring tile's loop,
        a staged load's level, an exchange tile's writing region (the dot's
        loop for its carry at entry, else the producing fragment's region)."""
        if operand.load is not None:
            return dot.loop if self.is_ring(dot, operand) else self.level(operand.load)
        assert operand.frag is not None
        if self.early_exchange(dot, operand):
            return dot.loop
        return self.frag_regions.get(self.resolve(operand.frag), -1)

    def resolve(self, node: Node) -> Node:
        """``node`` through its ``_new_var`` copies, ``_mask_to`` (an identity
        on whole tiles) and loop placeholders aliasing a value of the
        enclosing region."""
        return _resolve(node, self.aliases)

    @property
    def accesses(self) -> tuple[Access, ...]:
        result: list[Access] = []
        for carry in self.carries:
            if carry.init is not None:
                result.append(carry.init)
            result.extend(carry.init_aux)
        for dot in self.dots:
            for operand in (dot.a, dot.b):
                if operand.load is not None:
                    result.append(operand.load)
                result.extend(operand.aux_loads)
            result.extend(self.acc_loads.get(dot.node, ()))
        for derived in self.derived:
            result.extend(derived.loads)
            result.extend(derived.aux_loads)
        result.extend(store.access for store in self.stores)
        result.extend(self.store_aux_loads)
        for reduce in self.reduces:
            result.extend(reduce.aux_loads)
        for op in self.vec_ops:
            result.extend(op.aux_loads)
        result.extend(store.access for store in self.vec_stores)
        seen: dict[Node, Access] = {}
        for access in result:
            seen.setdefault(access.node, access)
        return tuple(seen.values())

    # Vector / scalar loads of the accumulator rescales and the store values
    # are attached after construction (they need the whole program).
    acc_loads: dict[Node, tuple[Access, ...]] = dataclasses.field(default_factory=dict)
    store_aux_loads: tuple[Access, ...] = ()
    # The epilogue's dots and stores in program order (their FX nodes).
    epilogue_order: tuple[Node, ...] = ()


@dataclasses.dataclass(frozen=True)
class AxisFacts:
    block_id: int
    kind: str
    extent: int
    # The device loop (index into ``CuteMmaDagFacts.loops``) a loop axis
    # tiles (-1 otherwise).
    loop: int = -1


@dataclasses.dataclass(frozen=True)
class TileFacts:
    """A shared tile: ``(rows, cols)`` axis indices, element size, and
    whether it lives in the operand ring (``stages`` slots) or once."""

    rows: int
    cols: int
    elem_bytes: int
    ring: bool
    # Element size of the raw source tile staged beside it (0: none).
    source_bytes: int = 0
    # Buffers of a non-ring tile (an exchange tile written before the
    # iteration barrier is double-buffered by iteration parity).
    copies: int = 1
    # The exchange-tile key of a fragment operand's tile (``FragEdgeFacts
    # .key``): allocated only when the layout solver exchanges that edge.
    edge_key: int | None = None
    # The loop whose shared-memory frame holds the tile (-1: the root
    # frame, live for the whole kernel); sibling loops' frames overlap.
    frame: int = -1
    # The raw source tile of a root-frame converted operand consumed ahead
    # of the first loop overlays the loops' frames (dead once they start).
    source_overlays: bool = False
    # A root-frame exchange tile read within one root segment overlays the
    # loops' frames as well (written after the loops before it, dead before
    # the loop after it starts).
    overlays: bool = False

    @property
    def bytes_per_element(self) -> int:
        return self.elem_bytes + self.source_bytes


@dataclasses.dataclass(frozen=True)
class LoopFacts:
    """A device loop's place in the nest: its parent (-1: root) and whether
    it is innermost (owns a cp.async ring of ``stages`` slots)."""

    parent: int
    innermost: bool


@dataclasses.dataclass(frozen=True)
class VecFacts:
    """A vector slot: the axis index it varies along (-1: a scalar),
    whether that is the rows or the columns of the fragments reduced into
    its family (``frags``: fragment slot indices), whose warp split it
    follows."""

    axis: int
    orient: str
    frags: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FragEdgeFacts:
    """A fragment operand of a dot: the producer and consumer slots, whether
    the logical operand is the producer's registers transposed, the K axis
    index, the element size, the exchange-tile sharing key, the region
    (loop edges are exchanged every iteration), whether the consumer dot
    runs inside a device loop, and whether such an operand is a hoisted
    root exchange: a fragment defined at the root and read by a loop's dot
    has one value for every iteration, so the region emitter writes its
    tile once ahead of the loop (``hoisted_exchange_loop``) -- an epilogue
    edge feeding a tensor-core operand in every iteration."""

    role: str
    logical_swap: bool
    producer: int
    consumer: int
    k: int
    elem_bytes: int
    key: int
    region: str
    in_loop: bool = False
    hoisted: bool = False


@dataclasses.dataclass(frozen=True)
class DotFacts:
    """One dot's transient registers: its result slot, its K axis index,
    whether its A operand is a fragment read straight (it may be repacked
    from registers for the whole K extent), whether B is a fragment and the
    operand element size (a tf32 K step is eight K: twice the words)."""

    slot: int
    k: int
    a_identity: bool
    b_frag: bool
    elem_bytes: int = 2


@dataclasses.dataclass(frozen=True)
class CuteMmaDagFacts:
    """What the config spec needs to validate the knobs without the graphs."""

    axes: tuple[AxisFacts, ...]
    # Fragment slots as (rows, cols) axis indices.
    frags: tuple[tuple[int, int], ...]
    tiles: tuple[TileFacts, ...]
    # Loop carry snapshots staged through shared memory: (rows, cols, bytes).
    staging: tuple[tuple[int, int, int], ...]
    # Axis indices that serve as a dot K extent (multiples of 16).
    k_axes: tuple[int, ...]
    smem_capacity: int
    grid_blocks: int = 1
    # The loop nest (``TileFacts.frame`` indexes it); one innermost loop
    # with no nesting is the single-loop shape.
    loops: tuple[LoopFacts, ...] = (LoopFacts(-1, True),)
    # The frame of each staging entry (empty: all in loop 0's frame).
    staging_frames: tuple[int, ...] = ()
    # Vector slots (row / column reductions of fragments and their carries).
    vecs: tuple[VecFacts, ...] = ()
    # The tcgen05 mode's admission facts (None: not admitted).
    tcgen05: Tcgen05Facts | None = None
    # Slot indices live together at some program point (the register budget
    # is the largest group); empty: every slot is live throughout.
    live: tuple[tuple[int, ...], ...] = ()
    # Fragment operands of the dots, in program order (the layout solver's
    # input; the seeds prefer warp counts that need no exchange tile).
    edges: tuple[FragEdgeFacts, ...] = ()
    # Streaming multiprocessors of the device (0: unknown): a second CTA per
    # SM only matters once there are more CTAs than SMs.
    sm_count: int = 0
    # Loop operand tiles rewritten by a per-element pass (a pointwise chain
    # or a dtype conversion): work that splits across more threads.
    loop_passes: int = 0
    dots: tuple[DotFacts, ...] = ()
    # Whether some fragment operand can take a register route (an A read
    # straight, or any B): only then can the exchange knob choose between
    # the registers and the shared tile.
    identity_candidates: bool = False

    def block_size(self, axis: int, sizes: Mapping[int, object]) -> int | None:
        facts = self.axes[axis]
        if facts.kind == AXIS_LEAD:
            return 1
        if facts.kind == AXIS_RDIM:
            return facts.extent
        value = sizes.get(facts.block_id)
        return value if type(value) is int else None

    @property
    def live_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.live or (tuple(range(len(self.frags))),)


@dataclasses.dataclass(frozen=True)
class SmemTileSpec:
    """One shared tile of the plan's layout: ``slots`` ring copies of
    ``rows x cols`` elements of ``elem_bytes`` at ``offset``; a converted
    operand stages its raw source tile beside it (``raw_offset``)."""

    name: str
    offset: int
    rows: int
    cols: int
    elem_bytes: int
    slots: int
    raw_offset: int = -1
    raw_elem_bytes: int = 0
    # The raw source tile of a transposed operand tile is ``cols x rows``.
    raw_swapped: bool = False
    # Ring slots of the raw source tile: the pass consumes it ahead of the
    # iteration barrier the refill follows, so one fewer than the operand
    # ring's (one at two stages).
    raw_slots: int = 1
    # The raw tile overlays the nested loops' frames (see
    # ``MmaDagProgram.staged_ahead_of_loops``).
    raw_overlays: bool = False
    # The tile itself does (a root exchange tile read within one root
    # segment, see ``MmaDagProgram.read_within_one_root_segment``).
    overlays: bool = False

    @property
    def bytes(self) -> int:
        return self.rows * self.cols * self.elem_bytes

    @property
    def raw_bytes(self) -> int:
        return self.rows * self.cols * self.raw_elem_bytes


@dataclasses.dataclass(frozen=True)
class MmaDagPlan:
    """The program plus the config-selected tiles, warps and routes."""

    program: MmaDagProgram
    # Axis index -> block size.
    sizes: tuple[int, ...]
    warps: int
    stages: int
    # Operand routes keyed by (dot node, role): direct / inplace / convert /
    # sync / identity / movmatrix / exchange.
    routes: dict[tuple[Node, str], str]
    staged_stores: tuple[Store, ...]
    exchange_mode: str
    # Fragment shape (rows, cols) -> warp grid (wm, wn).
    layouts: Layouts = dataclasses.field(default_factory=dict)
    # Derived values' 2-D tile loads (by load node): ``shared`` (cp.async
    # into a ring slot, read per lane from shared memory) or ``global``
    # (per-lane global pair loads at the fragment coordinates).
    derived_routes: dict[Node, str] = dataclasses.field(default_factory=dict)
    # ``warp`` (mma.sync, this module's emitter) or ``tcgen05``.
    compute: str = MMA_DAG_WARP
    tcgen05: Tcgen05Plan | None = None

    @property
    def threads(self) -> int:
        return 32 * self.warps

    def ring_derived(self, loop: int = -2) -> tuple[tuple[Derived, Access], ...]:
        """Derived tile loads staged through the ring of loop ``loop`` (every
        loop when omitted), in program order, one entry per load node."""
        seen: set[Node] = set()
        result = []
        for derived in self.program.derived:
            if loop != -2 and derived.loop != loop:
                continue
            for access in derived.loads:
                if (
                    self.derived_routes.get(access.node) == DERIVED_SHARED
                    and access.node not in seen
                ):
                    seen.add(access.node)
                    result.append((derived, access))
        return tuple(result)

    def size(self, axis: Axis) -> int:
        return self.sizes[axis.index]

    def slot_shape(self, slot: int) -> tuple[int, int]:
        rows, cols = self.program.slots[slot]
        return self.size(rows), self.size(cols)

    def slot_layout(self, slot: int) -> tuple[int, int]:
        shape = self.slot_shape(slot)
        layout = self.layouts.get(shape) or warp_mma_warp_layout(*shape, self.warps)
        assert layout is not None
        return layout

    def route(self, dot: Dot, operand: Operand) -> str:
        return self.routes[(dot.node, operand.role)]

    def exchange_copies(self, dot: Dot, operand: Operand) -> int:
        program = self.program
        return (
            2
            if program.early_exchange(dot, operand) and program.loop_has_ring(dot.loop)
            else 1
        )

    def tile_bytes(self, operand: Operand) -> int:
        return (
            self.size(operand.rows) * self.size(operand.cols) * operand.dtype.itemsize
        )

    def raw_tile_bytes(self, operand: Operand) -> int:
        """The raw source tile staged beside a converted operand (0: none)."""
        if operand.load is None or operand.load.dtype == operand.dtype:
            return 0
        return (
            self.size(operand.rows)
            * self.size(operand.cols)
            * operand.load.dtype.itemsize
        )

    def unique_operands(
        self, region: str, routes: tuple[str, ...]
    ) -> tuple[tuple[Dot, Operand], ...]:
        """The region's operands on ``routes`` with one entry per shared
        tile (operands with the same source and pointwise chain share it)."""
        return self.unique(
            (
                (dot, operand)
                for dot in self.program.dots_in(region)
                for operand in (dot.a, dot.b)
            ),
            routes,
        )

    def unique(
        self,
        pairs: Iterable[tuple[Dot, Operand]],
        routes: tuple[str, ...] | None = None,
    ) -> tuple[tuple[Dot, Operand], ...]:
        """``pairs`` filtered to ``routes`` with one entry per shared tile."""
        seen: set[tuple[object, ...]] = set()
        result = []
        for dot, operand in pairs:
            if routes is not None and self.route(dot, operand) not in routes:
                continue
            if operand.chain_key in seen:
                continue
            seen.add(operand.chain_key)
            result.append((dot, operand))
        return tuple(result)

    def ring_pairs(
        self, loop: int, routes: tuple[str, ...] | None = None
    ) -> tuple[tuple[Dot, Operand], ...]:
        """The per-iteration load operands of innermost loop ``loop`` (the
        ring candidates, synchronous ones included) on ``routes``."""
        program = self.program
        return self.unique(
            (
                (dot, operand)
                for dot in program.dots
                if loop == -2 or dot.loop == loop
                for operand in (dot.a, dot.b)
                if program.is_ring(dot, operand)
            ),
            routes,
        )

    def ring_operands(self, loop: int = -2) -> tuple[tuple[Dot, Operand], ...]:
        """Operands staged through the cp.async ring of loop ``loop`` (every
        loop when omitted), in program order."""
        return self.ring_pairs(loop, (ROUTE_DIRECT, ROUTE_INPLACE, ROUTE_CONVERT))

    def staged_operands(
        self, region: int, routes: tuple[str, ...] | None = None
    ) -> tuple[tuple[Dot, Operand], ...]:
        """Load operands staged once per execution of region ``region``
        (-1: once per CTA): tiles invariant in the dots' own loops that vary
        with ``region``, read by the dots of ``region`` itself or of loops
        nested in it."""
        program = self.program
        return self.unique(
            (
                (dot, operand)
                for dot in program.dots
                for operand in (dot.a, dot.b)
                if operand.load is not None
                and not program.is_ring(dot, operand)
                and program.level(operand.load) == region
            ),
            routes,
        )

    def early_prologues(self) -> dict[int, int]:
        """Loop -> the nested loop whose ring prologue the loop issues ahead
        of each iteration: during iteration ``i``'s epilogue (the steps after
        the nested loop) the ring tiles of iteration ``i + 1``'s first
        ``stages - 1`` nested iterations go out, overlapping the epilogue
        instead of being waited on right after they are issued.  A loop
        qualifies when it has exactly one nested loop, an innermost one
        with a ring at more than one stage, followed by at least one step,
        whose bounds name no loop's index (the prologue evaluates them
        ahead of the iteration), and no shared tile its own steps read
        (operands, derived tiles, exchange tiles, staging) overlaps the
        nested loop's frame: the prologue rewrites that frame while the
        epilogue runs."""
        program = self.program
        if self.stages < 2:
            return {}
        operand_specs, derived_specs, staging_specs, _total, _scratch = (
            self.smem_layout()
        )
        loop_symbols = {
            loop.symbol for loop in program.loops if loop.symbol is not None
        }
        result: dict[int, int] = {}
        for loop in program.loops:
            nested = [step for step in loop.steps if isinstance(step, Loop)]
            if len(nested) != 1 or not nested[0].innermost:
                continue
            inner = nested[0]
            if not program.loop_has_ring(inner.index):
                continue
            if any(
                _bound_symbols(bound) & loop_symbols
                for bound in (inner.begin, inner.end)
            ):
                continue
            position = loop.steps.index(inner)
            if not any(
                not isinstance(step, Loop) for step in loop.steps[position + 1 :]
            ):
                continue
            # The nested frame's extent: every tile of the nested loop.
            inner_ranges = [
                (spec.offset, spec.offset + spec.bytes * spec.slots)
                for dot in program.dots
                if dot.loop == inner.index
                for operand in (dot.a, dot.b)
                if self.route(dot, operand) not in REGISTER_ROUTES
                for spec in [operand_specs[operand.chain_key]]
            ]
            inner_ranges += [
                (spec.offset, spec.offset + spec.bytes * spec.slots)
                for derived, access in self.ring_derived(inner.index)
                for spec in [derived_specs[access.node]]
            ]
            inner_ranges += [
                (spec.raw_offset, spec.raw_offset + spec.raw_bytes * spec.raw_slots)
                for dot in program.dots
                if dot.loop == inner.index
                for operand in (dot.a, dot.b)
                if self.route(dot, operand) not in REGISTER_ROUTES
                for spec in [operand_specs[operand.chain_key]]
                if spec.raw_elem_bytes
            ]
            # The tiles the loop's own steps read.
            own_ranges = [
                (spec.offset, spec.offset + spec.bytes * spec.slots)
                for dot in program.dots
                if dot.loop == loop.index
                for operand in (dot.a, dot.b)
                if self.route(dot, operand) not in REGISTER_ROUTES
                for spec in [operand_specs[operand.chain_key]]
            ]
            own_ranges += [
                (spec.offset, spec.offset + spec.bytes * spec.slots)
                for derived in program.derived
                if derived.loop == loop.index
                for access in derived.loads
                if access.node in derived_specs
                for spec in [derived_specs[access.node]]
            ]
            own_ranges += [
                (spec.offset, spec.offset + spec.bytes * spec.slots)
                for store in self.staged_stores
                if store.loop == loop.index
                for spec in [staging_specs[store.access.node]]
            ]
            if any(
                a_lo < b_hi and b_lo < a_hi
                for a_lo, a_hi in inner_ranges
                for b_lo, b_hi in own_ranges
            ):
                continue
            result[loop.index] = inner.index
        return result

    def smem_layout(
        self,
    ) -> tuple[
        dict[tuple[object, ...], SmemTileSpec],
        dict[Node, SmemTileSpec],
        dict[Node, SmemTileSpec],
        int,
        SmemTileSpec | None,
    ]:
        """The shared-memory layout the emitter renders: operand tiles by
        sharing key (ring, staged / exchange tiles, in program order), the
        derived values' ring tiles by load node, the carry staging tiles by
        store node, and the total.  Operands with the same sharing key share
        one tile; every ring tile has a slot per stage (a single-slot tile
        rewritten at the top of the next iteration would race with lanes
        still reading it: there is no trailing barrier).  Tiles live in the
        frame of the loop that writes them; sibling loops' frames overlap
        (``mma_dag_frame_bytes``), nested ones follow their parent's own
        tiles."""
        program = self.program
        # (frame, kind, key) -> spec without offsets, in program order.
        items: list[tuple[int, str, object, SmemTileSpec]] = []
        seen: set[tuple[object, ...]] = set()
        count = 0
        for dot in program.dots:
            for operand in (dot.a, dot.b):
                route = self.route(dot, operand)
                if route in REGISTER_ROUTES or operand.chain_key in seen:
                    continue
                seen.add(operand.chain_key)
                exchange = route == ROUTE_EXCHANGE
                # An exchange tile of the carry at loop entry is written
                # before the iteration barrier: two copies by parity.
                if exchange:
                    slots = self.exchange_copies(dot, operand)
                elif program.is_ring(dot, operand):
                    slots = self.stages
                else:
                    slots = 1
                frame = program.operand_frame(dot, operand)
                spec = SmemTileSpec(
                    f"{'md_x' if exchange else 'md_t'}{count}",
                    0,
                    self.size(operand.rows),
                    self.size(operand.cols),
                    operand.dtype.itemsize,
                    slots,
                    overlays=exchange
                    and frame == -1
                    and program.read_within_one_root_segment(operand.chain_key),
                )
                count += 1
                if route == ROUTE_CONVERT:
                    load = operand.load
                    assert load is not None
                    spec = dataclasses.replace(
                        spec,
                        raw_offset=0,
                        raw_elem_bytes=load.dtype.itemsize,
                        raw_swapped=operand.transposed_tile,
                        raw_slots=mma_dag_raw_slots(slots),
                        raw_overlays=frame == -1
                        and program.staged_ahead_of_loops(operand.chain_key),
                    )
                items.append((frame, "operand", operand.chain_key, spec))
        # Derived values' tiles: one ring slot per stage, read per lane.
        for index, (owner, access) in enumerate(self.ring_derived()):
            rows, cols = access.tile_dims
            spec = SmemTileSpec(
                f"md_d{index}",
                0,
                self.size(rows),
                self.size(cols),
                access.dtype.itemsize,
                self.stages,
            )
            items.append((owner.loop, "derived", access.node, spec))
        for index, store in enumerate(self.staged_stores):
            rows, cols = self.slot_shape(program.frag_slots[store.frag])
            spec = SmemTileSpec(
                f"md_sg{index}",
                0,
                rows,
                cols,
                store.access.dtype.itemsize,
                2,
            )
            items.append((store.loop, "staging", store.access.node, spec))
        # The vector operations' shared scratch (cross-warp reductions,
        # relayouts, scalar reductions), used one at a time behind barriers.
        scratch_bytes = self.vec_scratch_bytes()
        if scratch_bytes:
            spec = SmemTileSpec("md_rv", 0, 1, scratch_bytes // 4, 4, 1)
            items.append((-1, "scratch", (), spec))
        # Frame layout: a loop's tiles follow the frames of the loops nested
        # in it (so the innermost ring sits at the start); sibling frames
        # share one extent.
        children: dict[int, list[int]] = {}
        for loop in program.loops:
            children.setdefault(loop.parent, []).append(loop.index)
        placed: dict[int, list[tuple[str, object, SmemTileSpec]]] = {}

        def layout(frame: int, base: int) -> int:
            nested = [layout(child, base) for child in children.get(frame, ())]
            frame_items = [item for item in items if item[0] == frame]
            # Raw source tiles dead once the nested loops start overlay
            # their frames, from the frame's base.
            overlay = base
            overlays: dict[object, int] = {}
            for _frame, _kind, key, spec in frame_items:
                if spec.raw_elem_bytes and spec.raw_overlays:
                    overlays[key] = overlay
                    overlay += spec.raw_bytes * spec.raw_slots
            for _frame, _kind, key, spec in frame_items:
                if spec.overlays:
                    overlays[key] = overlay
                    overlay += spec.bytes * spec.slots
            offset = max(max(nested, default=base), overlay)
            own: list[tuple[str, object, SmemTileSpec]] = []
            for _frame, kind, key, spec in frame_items:
                if spec.overlays:
                    own.append(
                        (kind, key, dataclasses.replace(spec, offset=overlays[key]))
                    )
                    continue
                spec = dataclasses.replace(spec, offset=offset)
                offset += spec.bytes * spec.slots
                if spec.raw_elem_bytes and key in overlays:
                    spec = dataclasses.replace(spec, raw_offset=overlays[key])
                elif spec.raw_elem_bytes:
                    spec = dataclasses.replace(spec, raw_offset=offset)
                    offset += spec.raw_bytes * spec.raw_slots
                own.append((kind, key, spec))
            placed[frame] = own
            return offset

        total = layout(-1, 0)
        tiles: dict[tuple[object, ...], SmemTileSpec] = {}
        derived: dict[Node, SmemTileSpec] = {}
        staging: dict[Node, SmemTileSpec] = {}
        scratch: SmemTileSpec | None = None
        for own in placed.values():
            for kind, key, spec in own:
                if kind == "operand":
                    tiles[cast("tuple[object, ...]", key)] = spec
                elif kind == "derived":
                    derived[cast("Node", key)] = spec
                elif kind == "staging":
                    staging[cast("Node", key)] = spec
                else:
                    scratch = spec
        return tiles, derived, staging, total, scratch

    def vec_layout(self, slot: int) -> tuple[int, int] | None:
        """The warp grid a vector slot's family follows (its fragments share
        the split along the vector's dim); None for a scalar."""
        family = self.program.vec_frag_slots[slot]
        if not family:
            return None
        return self.slot_layout(family[0])

    def vec_scratch_bytes(self) -> int:
        """The vector operations' shared scratch: the largest cross-warp
        reduction (one vector per warp along the other split), relayout
        (one vector) or scalar reduction (one value per warp)."""
        program = self.program
        scratch = 0
        for index, vec in enumerate(program.vec_slots):
            layout = self.vec_layout(index)
            if layout is None or vec.axis is None:
                scratch = max(scratch, self.warps * 4)
                continue
            wm, wn = layout
            other = wn if vec.orient == "rows" else wm
            scratch = max(scratch, max(other, 1) * self.size(vec.axis) * 4)
        return scratch

    @property
    def smem_bytes(self) -> int:
        return self.smem_layout()[3]


# Fragment operand routes: ``identity`` repacks the producer's registers into
# the operand words (an ``[M, K]`` fragment feeding A, an ``[N, K]`` one
# feeding B), ``movmatrix`` transposes each 8x8 block in registers (a ``[K,
# N]`` fragment feeding B), ``shuffle`` picks each fp32 (tf32) operand word
# from the producer's registers with two lane shuffles and a select (the
# same three shapes, one K step at a time) and ``exchange`` goes through a
# shared tile.
ROUTE_IDENTITY = "identity"
ROUTE_MOVMATRIX = "movmatrix"
ROUTE_SHUFFLE = "shuffle"
ROUTE_EXCHANGE = "exchange"
REGISTER_ROUTES: tuple[str, ...] = (ROUTE_IDENTITY, ROUTE_MOVMATRIX, ROUTE_SHUFFLE)
# Derived values' tile loads: staged through the ring and read per lane from
# shared memory, or loaded per lane from global memory.
DERIVED_SHARED = "shared"
DERIVED_GLOBAL = "global"


# ---------------------------------------------------------------------------
# Config legality (shared by the planner, ``normalize`` and the seeds)
# ---------------------------------------------------------------------------


def mma_dag_smem_estimate(
    facts: CuteMmaDagFacts,
    resolved: Sequence[int],
    stages: int,
    warps: int | None = None,
) -> int:
    """Shared memory the planner allocates for ``resolved`` block sizes (by
    axis index) at ``stages``: every operand tile (ringed ones ``stages``
    times, converted ones beside their raw source), the exchange tiles the
    layout solver needs at ``warps`` (all of them when ``warps`` is None)
    plus the double-buffered staging tile of every loop snapshot.  The
    planner's ``smem_bytes`` is this number for its selected routes."""
    exchanged: set[int] | None = None
    layouts: Layouts = {}
    if warps is not None and (facts.edges or facts.vecs):
        layouts, routes, _bytes = mma_dag_layout_solution(facts, resolved, warps)
        exchanged = {
            edge.key
            for edge, route in zip(facts.edges, routes, strict=True)
            if route == ROUTE_EXCHANGE
        }
    items: list[tuple[int, int]] = []
    overlays: list[tuple[int, int]] = []
    if warps is not None and facts.vecs:
        items.append((-1, mma_dag_vec_scratch(facts, resolved, layouts, warps)))
    for tile in facts.tiles:
        if (
            tile.edge_key is not None
            and exchanged is not None
            and tile.edge_key not in exchanged
        ):
            continue
        slots = stages if tile.ring else tile.copies
        elements = resolved[tile.rows] * resolved[tile.cols]
        source = elements * mma_dag_raw_slots(slots) * tile.source_bytes
        if tile.source_overlays:
            overlays.append((tile.frame, source))
            source = 0
        if tile.overlays:
            overlays.append((tile.frame, elements * slots * tile.elem_bytes))
            continue
        items.append((tile.frame, elements * slots * tile.elem_bytes + source))
    for index, (rows, cols, elem_bytes) in enumerate(facts.staging):
        frame = facts.staging_frames[index] if facts.staging_frames else 0
        items.append((frame, 2 * resolved[rows] * resolved[cols] * elem_bytes))
    return mma_dag_frame_bytes(facts.loops, items, overlays)


def mma_dag_raw_slots(slots: int) -> int:
    """Ring slots of a converted operand's raw source tile beside an operand
    ring of ``slots``: iteration ``i``'s pass consumes its raw tile before
    the iteration barrier, and the refill past that barrier (for iteration
    ``i + slots - 1``) may take the slot over, so ``slots - 1`` raw tiles are
    ever live (one at two stages); a single-slot tile keeps one."""
    return max(1, slots - 1)


def mma_dag_frame_bytes(
    loops: Sequence[LoopFacts],
    items: Sequence[tuple[int, int]],
    overlays: Sequence[tuple[int, int]] = (),
) -> int:
    """Shared memory of ``items`` (``(frame, bytes)``) laid out by frame: a
    loop's frame holds its own tiles and, past them, the frames of its
    nested loops; sibling loops never run together, so their frames share
    one extent (the largest).  The root frame (-1) is live throughout.
    ``overlays`` (``(frame, bytes)``) are dead once the frame's nested loops
    start and share their extent."""
    own: dict[int, int] = {}
    for frame, nbytes in items:
        own[frame] = own.get(frame, 0) + nbytes
    overlaid: dict[int, int] = {}
    for frame, nbytes in overlays:
        overlaid[frame] = overlaid.get(frame, 0) + nbytes
    children: dict[int, list[int]] = {}
    for index, loop in enumerate(loops):
        children.setdefault(loop.parent, []).append(index)

    def size(frame: int) -> int:
        nested = [size(child) for child in children.get(frame, ())]
        return own.get(frame, 0) + max(max(nested, default=0), overlaid.get(frame, 0))

    return size(-1)


def mma_dag_warp_layout(
    bm: int, bn: int, warps: int, n_first: bool = False
) -> tuple[int, int] | None:
    """``(wm, wn)`` warps along M and N of a ``bm x bn`` fragment: M split
    first (``warp_mma_warp_layout``) or N split first.

    A fragment that feeds a B operand from registers needs every warp to own
    whole K rows (``wm == 1``), the N-first grid; one that feeds A needs whole
    M rows (``wn == 1``), the M-first grid.
    """
    if not n_first:
        return warp_mma_warp_layout(bm, bn, warps)
    if warps < 1 or bm % 16 or bn % 8:
        return None
    wn = min(warps, bn // 8)
    if warps % wn:
        return None
    wm = warps // wn
    if (bm // 16) % wm:
        return None
    return wm, wn


def mma_dag_vec_split(
    vec: VecFacts,
    facts: CuteMmaDagFacts,
    resolved: Sequence[int],
    layouts: Layouts,
    warps: int,
) -> int | None:
    """The warp split a vector family follows (``wm`` of its fragments for a
    row vector, ``wn`` for a column vector), None when the fragments
    disagree (the layouts cannot carry the vector)."""
    if vec.orient == "scalar":
        return 1
    splits: set[int] = set()
    for slot in vec.frags:
        rows, cols = facts.frags[slot]
        shape = (resolved[rows], resolved[cols])
        layout = layouts.get(shape) or warp_mma_warp_layout(*shape, warps)
        if layout is None:
            return None
        splits.add(layout[0] if vec.orient == "rows" else layout[1])
    if len(splits) != 1:
        return None
    return splits.pop()


def mma_dag_vec_violations(
    facts: CuteMmaDagFacts, resolved: Sequence[int], layouts: Layouts, warps: int
) -> int:
    """How many vector families the layouts cannot carry."""
    return sum(
        1
        for vec in facts.vecs
        if mma_dag_vec_split(vec, facts, resolved, layouts, warps) is None
    )


def mma_dag_vec_regs(
    facts: CuteMmaDagFacts, resolved: Sequence[int], layouts: Layouts, warps: int
) -> int:
    """fp32 registers per lane the vector slots hold: a row vector over an
    extent ``E`` split ``wm`` ways is ``E / wm / 8`` words (rows ``g`` and
    ``g + 8`` of each 16-row atom), a column vector ``E / wn / 8``."""
    regs = 0
    for vec in facts.vecs:
        split = mma_dag_vec_split(vec, facts, resolved, layouts, warps)
        if split is None:
            continue
        if vec.orient == "scalar":
            regs += 1
        else:
            regs += max(1, resolved[vec.axis] // split // 8)
    return regs


def mma_dag_vec_scratch(
    facts: CuteMmaDagFacts, resolved: Sequence[int], layouts: Layouts, warps: int
) -> int:
    """The shared scratch the vector operations share (one write, barrier,
    read, barrier at a time): the largest of the cross-warp halves of the
    reductions (one fp32 vector per warp along the other split), the
    row / column relayouts (one vector) and the scalar reductions (one
    value per warp)."""
    scratch = 0
    for vec in facts.vecs:
        if vec.orient == "scalar":
            scratch = max(scratch, warps * 4)
            continue
        rows, cols = facts.frags[vec.frags[0]]
        shape = (resolved[rows], resolved[cols])
        layout = layouts.get(shape) or warp_mma_warp_layout(*shape, warps)
        if layout is None:
            continue
        other = layout[1] if vec.orient == "rows" else layout[0]
        scratch = max(scratch, max(other, 1) * resolved[vec.axis] * 4)
    return scratch


def mma_dag_transient_regs(
    facts: CuteMmaDagFacts, resolved: Sequence[int], warps: int
) -> int:
    """Registers one dot holds beside the fragments: the A words repacked
    from a fragment for the whole K extent (identity route; one K step of
    ``ldmatrix`` A fragments otherwise) plus one K step of B fragments for
    every n atom.  The dots run one after another, so the largest counts."""
    # The solver's warp grids and routes (M-first grids, fragment operands
    # repacked, without edge facts).
    layouts: Layouts = {}
    routes: dict[tuple[int, str], str] = {}
    if facts.edges:
        layouts, edge_routes, _bytes = mma_dag_layout_solution(facts, resolved, warps)
        for edge, route in zip(facts.edges, edge_routes, strict=True):
            key = (edge.consumer, edge.role)
            if route in REGISTER_ROUTES or key not in routes:
                routes[key] = route
    most = 0
    for facts_dot in facts.dots:
        rows, cols = facts.frags[facts_dot.slot]
        bm, bn = resolved[rows], resolved[cols]
        layout = layouts.get((bm, bn)) or warp_mma_warp_layout(bm, bn, warps)
        if layout is None:
            continue
        wm, wn = layout
        wtm, wtn = bm // wm, bn // wn
        k = resolved[facts_dot.k]
        a_route = routes.get((facts_dot.slot, "a"), ROUTE_IDENTITY)
        b_route = routes.get((facts_dot.slot, "b"), ROUTE_MOVMATRIX)
        # Words per K step and atom: a 16-bit K step is 16 K (four A words
        # per m atom, two B words per n atom), a tf32 one eight K.
        k_per_word = 64 if facts_dot.elem_bytes == 2 else 32
        a_words = (
            wtm * k // k_per_word
            if facts_dot.a_identity and wn == 1 and a_route in REGISTER_ROUTES
            else wtm // 4
        )
        # A register-routed B fragment binds its words for every K step.
        b_words = (
            k * wtn // k_per_word
            if facts_dot.b_frag and b_route in REGISTER_ROUTES
            else wtn // 4
        )
        most = max(most, a_words + b_words)
    return most


def mma_dag_config_reason(
    facts: CuteMmaDagFacts,
    sizes: Mapping[int, object],
    warps: object,
    stages: object,
    mode: str = MMA_DAG_WARP,
) -> str | None:
    """Why the block sizes / warps / stages cannot run the program in
    ``mode`` (None: they can).

    ``sizes`` maps block ids to the config's block sizes.
    """
    if mode == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import tcgen05_warps_reason

        warps_reason = tcgen05_warps_reason(warps)
        if warps_reason is not None:
            return warps_reason
    elif type(warps) is not int or warps not in MMA_DAG_WARP_CHOICES:
        return f"{MMA_DAG_WARPS_KEY} {warps!r} must be one of {MMA_DAG_WARP_CHOICES}"
    if type(stages) is not int or stages not in MMA_DAG_STAGE_CHOICES:
        return f"{MMA_DAG_STAGES_KEY} {stages!r} must be one of {MMA_DAG_STAGE_CHOICES}"
    resolved: list[int] = []
    for index, axis in enumerate(facts.axes):
        size = facts.block_size(index, sizes)
        if size is None:
            return f"the block size of tile axis {axis.block_id} must be an integer"
        if size <= 0 or size & (size - 1):
            return (
                f"block size {size} of tile axis {axis.block_id} must be a power of two"
            )
        if axis.kind in (AXIS_ROOT, AXIS_LOOP) and axis.extent % size:
            return f"block size {size} must divide the extent {axis.extent} of tile axis {axis.block_id}"
        resolved.append(size)
    for index in facts.k_axes:
        if resolved[index] % MMA_DAG_K_STEP:
            return f"the dot K extent {resolved[index]} must be a multiple of {MMA_DAG_K_STEP}"
    assert type(warps) is int
    if mode == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import tcgen05_config_reason

        return tcgen05_config_reason(facts, resolved, warps, stages)
    frag_words: list[int] = []
    for rows, cols in facts.frags:
        bm, bn = resolved[rows], resolved[cols]
        if bm % MMA_DAG_MIN_BM or bn % MMA_DAG_MIN_BN:
            return (
                f"a {bm}x{bn} fragment needs rows in multiples of 16 and columns of 8"
            )
        if warp_mma_warp_layout(bm, bn, warps) is None:
            return f"{warps} warps cannot tile a {bm}x{bn} fragment into 16x8 atoms"
        frag_words.append(bm * bn)
    # The budget is the largest set of fragments live at one program point
    # (an attention tile consumed before the next no-acc dot's result exists
    # shares its registers with it).
    words = max(sum(frag_words[slot] for slot in group) for group in facts.live_groups)
    regs = words // (32 * warps)
    if facts.vecs:
        layouts, _routes, _bytes = mma_dag_layout_solution(facts, resolved, warps)
        if mma_dag_vec_violations(facts, resolved, layouts, warps):
            return (
                f"{warps} warps cannot split the fragments of a vector reduction "
                "the same way"
            )
        regs += mma_dag_vec_regs(facts, resolved, layouts, warps)
    if regs > MMA_DAG_MAX_FRAG_REGS:
        return (
            f"the live fragments need {regs} registers per thread over {warps} "
            f"warps (limit {MMA_DAG_MAX_FRAG_REGS})"
        )
    live = regs + mma_dag_transient_regs(facts, resolved, warps)
    if live > MMA_DAG_MAX_LIVE_REGS:
        return (
            f"the fragments and one dot's operand words need {live} registers "
            f"per thread over {warps} warps (limit {MMA_DAG_MAX_LIVE_REGS})"
        )
    smem = mma_dag_smem_estimate(facts, resolved, stages, warps)
    if facts.smem_capacity and smem > facts.smem_capacity:
        return (
            f"{stages} stages of operand tiles need {smem} B of shared memory "
            f"(capacity {facts.smem_capacity})"
        )
    return None


def mma_dag_repair(
    facts: CuteMmaDagFacts,
    sizes: Mapping[int, object],
    warps: object,
    stages: object,
    mode: str = MMA_DAG_WARP,
) -> tuple[dict[int, int], int, int] | None:
    """A legal ``(sizes, warps, stages)`` near the request: fewer stages
    first, then fewer warps, then the larger tunable tile halved (None: no
    legal config)."""
    if mode == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import tcgen05_repair

        return tcgen05_repair(facts, sizes, warps, stages)
    tunable = [
        axis.block_id for axis in facts.axes if axis.kind in (AXIS_ROOT, AXIS_LOOP)
    ]
    fixed = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
        if axis.kind not in (AXIS_ROOT, AXIS_LOOP)
    }
    current: dict[int, int] = dict(fixed)
    for block_id in tunable:
        value = sizes.get(block_id)
        current[block_id] = value if type(value) is int else 16
    warps_now = warps if type(warps) is int and warps in MMA_DAG_WARP_CHOICES else 8
    stages_now = (
        stages if type(stages) is int and stages in MMA_DAG_STAGE_CHOICES else 4
    )
    for _ in range(64):
        if mma_dag_config_reason(facts, current, warps_now, stages_now) is None:
            return current, warps_now, stages_now
        if stages_now > 1 and (
            mma_dag_config_reason(facts, current, warps_now, 1) is None
            or "shared memory"
            in str(mma_dag_config_reason(facts, current, warps_now, stages_now))
        ):
            stages_now = max(s for s in MMA_DAG_STAGE_CHOICES if s < stages_now)
            continue
        legal_warps = [
            w
            for w in MMA_DAG_WARP_CHOICES
            if mma_dag_config_reason(facts, current, w, stages_now) is None
        ]
        if legal_warps:
            warps_now = max(legal_warps)
            continue
        candidates = [b for b in tunable if current[b] > 1]
        if not candidates:
            return None
        largest = max(candidates, key=lambda b: current[b])
        current[largest] //= 2
        # Powers of two below the fragment floors are hopeless for that axis.
        if current[largest] < 8:
            return None
    return None


def mma_dag_pass_packets(facts: CuteMmaDagFacts, resolved: Sequence[int]) -> int:
    """16-byte packets of the loop's passed operand tiles per pass-warp
    thread and iteration at the resolved sizes."""
    tc = facts.tcgen05
    if tc is None:
        return 0
    packets = sum(
        resolved[tile.rows] * resolved[tile.cols] * tile.source_bytes // 16
        for tile in tc.tiles
        if tile.region == "loop" and tile.passed
    )
    return packets // (32 * 4)


def mma_dag_cta_count(facts: CuteMmaDagFacts, resolved: Sequence[int]) -> int:
    """CTAs of the grid at the resolved sizes: the leading tiles times the
    root tiles' counts (one CTA per root tile)."""
    ctas = facts.grid_blocks
    for index, axis in enumerate(facts.axes):
        if axis.kind == AXIS_ROOT:
            ctas *= axis.extent // resolved[index]
    return ctas


def mma_dag_pass_warps_lead(facts: CuteMmaDagFacts, resolved: Sequence[int]) -> bool:
    """Whether the pass-warp seed leads (becomes the default): a grid of
    one wave -- the serial state chains, one CTA per SM, where the extra
    ring depth costs no occupancy -- whose passes are worth at least
    ``MMA_DAG_PASS_WARP_PACKETS`` per thread and iteration."""
    return (
        facts.sm_count > 0
        and mma_dag_cta_count(facts, resolved) <= facts.sm_count
        and mma_dag_pass_packets(facts, resolved) >= MMA_DAG_PASS_WARP_PACKETS
    )


def mma_dag_pass_warps_reason(
    facts: CuteMmaDagFacts, warps: int, stages: int, mode: str
) -> str | None:
    """Why ``cute_mma_dag_pass_warps='on'`` has no subject for this config
    (None: it has): the tcgen05 single-loop emitter at eight warps with loop
    operand passes and a ring of at least two slots (the pass warps rewrite
    the next tile while the chain's MMA reads the current one)."""
    if mode != MMA_DAG_TCGEN05:
        return "the warp mode runs its passes on every warp"
    if facts.tcgen05 is None or facts.tcgen05.regions:
        return "the region renderer has no pass warps"
    if facts.loop_passes == 0:
        return "no loop operand needs a pass"
    if warps != max(MMA_DAG_WARP_CHOICES):
        return f"the pass warps are warps 4-7 of {max(MMA_DAG_WARP_CHOICES)}"
    if stages < 2:
        return "the next tile needs its own ring slot (stages >= 2)"
    return None


def mma_dag_seed_warps(bm: int, bn: int) -> int:
    """The seed heuristic's warp count: the most warps (at most eight) that
    tile ``bm x bn`` within the register budget.

    The recurrence is a latency-bound serial chain with one CTA per SM, so
    more warps mean more memory parallelism and shorter per-warp instruction
    streams (64x64 at s2: 8 warps 117 us, 4 warps 132 us, 2 warps 200 us).
    """
    for warps in reversed(MMA_DAG_WARP_CHOICES):
        if (
            warp_mma_warp_layout(bm, bn, warps) is not None
            and mma_dag_frag_regs(bm, bn, warps) <= MMA_DAG_MAX_FRAG_REGS
        ):
            return warps
    return 1


def mma_dag_ctas_per_sm(
    smem: int, smem_capacity: int, tmem_cols: int = 0, limit: int = 32
) -> int:
    """The CTAs one SM holds at ``smem`` bytes of shared memory each (0 or
    an unknown capacity: ``limit``) and, for the tcgen05 mode, ``tmem_cols``
    tensor-memory columns each -- the occupancy prediction the seeds rank
    by (the register count is ptxas's and follows this occupancy: it sizes
    registers to what the static shared memory allows)."""
    ctas = limit
    if smem_capacity:
        ctas = min(
            ctas,
            (smem_capacity + MMA_DAG_SMEM_CTA_RESERVED)
            // (smem + MMA_DAG_SMEM_CTA_RESERVED),
        )
    if tmem_cols:
        ctas = min(ctas, MMA_DAG_TMEM_COLUMNS // tmem_cols)
    return ctas


def mma_dag_seed_configs(
    facts: CuteMmaDagFacts,
) -> list[tuple[dict[int, int], int, int]]:
    """Ranked ``(sizes, warps, stages)`` seeds.

    Tunable axes walk the ladder {16, 32, 64, 128} (divisors of the extent);
    warps are the most that tile every fragment within the register budget
    and, when fewer warps let every fragment operand of the loop take a
    register route (no exchange tile, no extra barrier per iteration), the
    most such warps as well, ranked first unless a loop operand needs a
    per-element pass (fwd_h_delta at s2: four warps 296 us, eight 335 us;
    gated, with the key rescaled in place, eight 522 against four 598;
    dstate, whose attention tile is exchanged at any warp count, keeps
    eight: 612 us against 679 at four); stages prefer eight.  Ranking: CTAs
    >= 128 first, then the largest (and squarest) root tile, the
    exchange-free warps, two CTAs per SM of shared memory (when the grid
    exceeds the SM count), the largest loop tile, stages, warps.
    """
    tunable = [axis for axis in facts.axes if axis.kind in (AXIS_ROOT, AXIS_LOOP)]
    fixed = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
        if axis.kind not in (AXIS_ROOT, AXIS_LOOP)
    }
    ladders = [
        [tile for tile in MMA_DAG_TILE_LADDER if axis.extent % tile == 0]
        for axis in tunable
    ]
    cta_rich = _cta_rich(facts)
    ranked: list[tuple[tuple[int, ...], tuple[dict[int, int], int, int]]] = []
    for choice in itertools.product(*ladders):
        sizes: dict[int, int] = dict(fixed)
        for axis, tile in zip(tunable, choice, strict=True):
            sizes[axis.block_id] = tile
        resolved = [facts.block_size(i, sizes) for i in range(len(facts.axes))]
        if any(size is None for size in resolved):
            continue
        resolved_sizes = cast("list[int]", resolved)
        legal = [
            w
            for w in MMA_DAG_WARP_CHOICES
            if mma_dag_config_reason(facts, sizes, w, 1) is None
        ]
        if not legal:
            continue
        # (warps, preferred): the budget warps, and the most exchange-free
        # warps when fewer suffice.  Per-element operand passes split across
        # the threads, so they favour the budget warps (gated fwd_h_delta:
        # eight warps 522 us against four 598); without one the
        # exchange-free count leads.  A kernel whose fragments are sized by
        # rdims and loop tiles alone, with a grid of leading tiles far
        # beyond the SM count (the per-chunk delta kernels: 8192 CTAs),
        # draws its parallelism from co-resident CTAs: four warps lead there
        # (at s2: dqkw 639 us against 656 at eight, wy_dL 450 against 477,
        # wy 794 against 1034, dk level) and tiles over budget at four warps
        # rank behind.
        candidates = [(max(legal), 1)]
        if cta_rich:
            if MMA_DAG_DEFAULT_WARPS in legal:
                candidates = [(MMA_DAG_DEFAULT_WARPS, 1)]
                if max(legal) != MMA_DAG_DEFAULT_WARPS:
                    candidates.append((max(legal), 0))
            else:
                candidates = [(max(legal), 0)]
        elif any(edge.region == "loop" for edge in facts.edges):
            exchange_free = [
                w
                for w in legal
                if mma_dag_layout_solution(facts, resolved_sizes, w)[2] == 0
            ]
            if exchange_free and max(exchange_free) != max(legal):
                if facts.loop_passes == 0:
                    candidates = [(max(exchange_free), 1), (max(legal), 0)]
                else:
                    candidates.append((max(exchange_free), 0))
        ctas = facts.grid_blocks
        root_area = 1
        root_perimeter = 0
        loop_area = 1
        # Loops of one iteration cannot overlap their loads with the
        # previous tile's MMAs: among CTA-rich kernels every loop keeps at
        # least two trips (wy_dL at s2: kk tiles of 64 run 420 us, 128
        # run 450).
        pipelined = 1
        for axis, tile in zip(tunable, choice, strict=True):
            if axis.kind == AXIS_ROOT:
                ctas *= axis.extent // tile
                root_area *= tile
                root_perimeter += tile
            else:
                loop_area *= tile
                if cta_rich and axis.extent // tile < 2:
                    pipelined = 0
        for warps, preferred in candidates:
            for stages in MMA_DAG_SEED_STAGES:
                if mma_dag_config_reason(facts, sizes, warps, stages) is not None:
                    continue
                # Shared memory for two CTAs per SM counts once the grid has
                # more CTAs than SMs (bdv 16 at s2: 256 CTAs, two stages 342
                # us against 444 at three); otherwise the deeper ring wins.
                two_per_sm = int(
                    _seed_two_per_sm(facts, ctas, resolved_sizes, warps, stages)
                )
                # Among equal root areas the squarer tile stages fewer
                # operand rows.
                rank = (
                    int(ctas >= 128),
                    root_area,
                    -root_perimeter,
                    preferred,
                    two_per_sm,
                    pipelined,
                    loop_area,
                    stages,
                    warps,
                )
                ranked.append((rank, (sizes, warps, stages)))
    ranked.sort(key=operator.itemgetter(0), reverse=True)
    return [seed for _rank, seed in ranked]


def _cta_rich(facts: CuteMmaDagFacts) -> bool:
    """A grid of leading tiles far beyond the SM count with no root tile
    (the per-chunk delta kernels): parallelism comes from co-resident CTAs."""
    return (
        facts.sm_count > 0
        and facts.grid_blocks >= 2 * facts.sm_count
        and not any(axis.kind == AXIS_ROOT for axis in facts.axes)
    )


def _seed_ctas(facts: CuteMmaDagFacts, sizes: Mapping[int, int]) -> int:
    ctas = facts.grid_blocks
    for axis in facts.axes:
        if axis.kind == AXIS_ROOT:
            ctas *= axis.extent // sizes[axis.block_id]
    return ctas


def _seed_two_per_sm(
    facts: CuteMmaDagFacts,
    ctas: int,
    resolved: Sequence[int],
    warps: int,
    stages: int,
) -> bool:
    """Whether the config's shared memory leaves room for two CTAs per SM
    (moot for a grid within the SM count or an unknown capacity)."""
    return (facts.sm_count > 0 and ctas <= facts.sm_count) or mma_dag_ctas_per_sm(
        mma_dag_smem_estimate(facts, resolved, stages, warps), facts.smem_capacity
    ) >= 2


def mma_dag_seed_variants(
    facts: CuteMmaDagFacts,
    ranked: Sequence[tuple[dict[int, int], int, int]],
) -> list[tuple[dict[int, int], int, int, str]]:
    """Measured neighbours of the top warp seed of a CTA-rich program,
    ``(sizes, warps, stages, exchange)``, to follow the ranked seeds
    ``ranked`` (the first seed stays).  The cold autotune of the gated delta
    rule at s1 left the ranked ladder for each of them; measured against the
    default at s1 / s2 (B200, us):

    - the ring one step deeper while two CTAs per SM still fit (the ladder
      walks 8 / 4 / 2: dqkw 193 -> 182 / 757 -> 713, wy_dL 117 -> 117 /
      446 -> 437), and the same with every fragment operand exchanged
      through shared memory (dqkw 182 / 704);
    - the fragment operands exchanged through shared memory where the
      automatic routes keep some in registers (dqkw 192 / 742), for 16-bit
      operands: the fp32 tiles of a tf32 chain cost wy 139 -> 155;
    - a loop axis a dot contracts over at its full extent -- one trip -- when
      that is a ladder tile and the shared memory keeps its CTAs per SM
      (wy_dL [64, 64, 128] 115 / 442; dqkw [128, 32, 64, 64, 64] 192 / 753;
      the nested K loop of dqkw at 128 halves the CTAs and runs 258), with
      its shared-memory exchange twin (wy_dL 110 / 414, dqkw 189 / 723);
    - a single-loop program with the ring at one stage (dk [64] 52 -> 49 /
      161 -> 156) and at the 32 tile with two warps and one stage, the most
      co-resident CTAs (dk 47 / 149); a multi-loop program loses the
      overlap (dqkw 205, wy_dL 126).
    """
    if not ranked or not _cta_rich(facts):
        return []
    sizes, warps, stages = ranked[0]
    seen = {
        (tuple(sorted(s.items())), w, st, MMA_DAG_EXCHANGE_AUTO) for s, w, st in ranked
    }
    variants: list[tuple[dict[int, int], int, int, str]] = []

    def resolve(candidate: Mapping[int, int]) -> list[int] | None:
        resolved = [facts.block_size(i, candidate) for i in range(len(facts.axes))]
        if any(size is None for size in resolved):
            return None
        return cast("list[int]", resolved)

    def ctas_per_sm(candidate: Mapping[int, int], w: int, st: int) -> int:
        resolved = resolve(candidate)
        if resolved is None or not facts.smem_capacity:
            return 0
        return facts.smem_capacity // max(
            1, mma_dag_smem_estimate(facts, resolved, st, w)
        )

    def exchanges(candidate: Mapping[int, int], w: int) -> bool:
        """Whether forcing the shared-memory exchange changes the routes
        (of 16-bit fragment operands)."""
        resolved = resolve(candidate)
        return bool(
            facts.edges
            and all(dot.elem_bytes == 2 for dot in facts.dots)
            and resolved is not None
            and any(
                route in REGISTER_ROUTES
                for route in mma_dag_layout_solution(facts, resolved, w)[1]
            )
        )

    def add(candidate: Mapping[int, int], w: int, st: int, exchange: str) -> None:
        key = (tuple(sorted(candidate.items())), w, st, exchange)
        if key in seen or mma_dag_config_reason(facts, candidate, w, st) is not None:
            return
        seen.add(key)
        variants.append((dict(candidate), w, st, exchange))

    def add_with_twin(candidate: Mapping[int, int], w: int, st: int) -> None:
        add(candidate, w, st, MMA_DAG_EXCHANGE_AUTO)
        if exchanges(candidate, w):
            add(candidate, w, st, MMA_DAG_EXCHANGE_SMEM)

    resolved = resolve(sizes)
    if resolved is None:
        return []
    ctas = _seed_ctas(facts, sizes)
    deeper = next((s for s in MMA_DAG_STAGE_CHOICES if s > stages), None)
    if deeper is not None and _seed_two_per_sm(facts, ctas, resolved, warps, deeper):
        add_with_twin(sizes, warps, deeper)
    if exchanges(sizes, warps):
        add(sizes, warps, stages, MMA_DAG_EXCHANGE_SMEM)
    occupancy = ctas_per_sm(sizes, warps, stages)
    for index in facts.k_axes:
        axis = facts.axes[index]
        if (
            axis.kind == AXIS_LOOP
            and axis.extent in MMA_DAG_TILE_LADDER
            and sizes.get(axis.block_id, axis.extent) < axis.extent
        ):
            full = {**sizes, axis.block_id: axis.extent}
            if ctas_per_sm(full, warps, stages) >= occupancy:
                add_with_twin(full, warps, stages)
    if len(facts.loops) == 1:
        if stages > 1:
            add(sizes, warps, 1, MMA_DAG_EXCHANGE_AUTO)
        small = {
            axis.block_id: 32
            if axis.kind == AXIS_LOOP and axis.extent % 32 == 0
            else size
            for axis in facts.axes
            for size in [sizes.get(axis.block_id)]
            if size is not None
        }
        if 2 in MMA_DAG_WARP_CHOICES:
            add({**sizes, **small}, 2, 1, MMA_DAG_EXCHANGE_AUTO)
    return variants


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def _static_int(value: object) -> int | None:
    from ...language.matmul_ops import _static_dim_value

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        return _static_dim_value(CompileEnvironment.current(), value)
    if isinstance(value, sympy.Integer):
        return int(value)
    return None


def _is_call(node: object, target: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
    )


def rounded_dot_dtype(node: object) -> torch.dtype | None:
    """The 16-bit dtype of a ``torch.matmul`` of two 16-bit tiles
    (``aten.mm``): a dot whose fp32 product is rounded once to the operands'
    dtype -- its fragment holds the fp32 product and every reader evaluates
    the rounded value.  None for anything else."""
    if not _is_call(node, torch.ops.aten.mm.default):
        return None
    assert isinstance(node, torch.fx.Node)
    if len(node.args) != 2 or node.kwargs:
        return None
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor) or val.dtype not in _OPERAND_DTYPES:
        return None
    for arg in node.args:
        arg_val = arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None
        if not isinstance(arg_val, torch.Tensor) or arg_val.dtype != val.dtype:
            return None
    return val.dtype


def _is_dot(node: object) -> bool:
    """An ``hl.dot`` or a rounded 16-bit ``aten.mm`` (``rounded_dot_dtype``)."""
    from ...language.matmul_ops import dot

    return _is_call(node, dot) or rounded_dot_dtype(node) is not None


def _dot_acc(node: Node) -> object:
    """The accumulator argument of a dot (``aten.mm`` has none)."""
    return node.args[2] if rounded_dot_dtype(node) is None else None


def _tile_index_position(node: Node) -> int | None:
    """``tile.index`` broadcast by ``None`` subscripts (``tile.index[None, :]``
    is a ``load`` of the ``tile_index`` value): the position of its one tile
    dim, None for any other node."""
    from ...language import memory_ops
    from ...language import tile_ops

    if not _is_call(node, memory_ops.load) or len(node.args) < 4:
        return None
    source, indices = node.args[0], node.args[1]
    if (
        not _is_call(source, tile_ops.tile_index)
        or node.args[2] is not None
        or node.args[3] is not None
        or not isinstance(indices, (list, tuple))
    ):
        return None
    positions = [i for i, index in enumerate(indices) if index is not None]
    if len(positions) != 1 or indices[positions[0]] != slice(None):
        return None
    return positions[0]


def _tile_signature(access: Access) -> tuple[object, ...] | None:
    """What makes two loads of a region the same tile: the tensor, its
    dtype and strides, and whole-tile subscripts of the same axes (loads
    with scalar subscripts are not deduplicated)."""
    if any(subscript.kind != "axis" for subscript in access.subscripts):
        return None
    return (
        access.name,
        access.dtype,
        access.strides,
        tuple(
            (cast("Axis", subscript.axis).block_id, subscript.offset)
            for subscript in access.subscripts
        ),
    )


def _storage(access: Access) -> StorageWeakRef:
    """The storage an access reads or writes: views of one tensor (a host
    slice passed as another argument) share it, their names do not."""
    return StorageWeakRef(access.fake.untyped_storage())


def _resolve(node: Node, aliases: Mapping[Node, Node]) -> Node:
    """``node`` through ``_new_var`` copies, ``_mask_to`` (an identity on
    whole tiles: every tile here divides its extent) and the placeholder
    ``aliases``."""
    from ...language import _tracing_ops

    while True:
        if _is_call(node, _tracing_ops._new_var) or _is_call(
            node, _tracing_ops._mask_to
        ):
            node = cast("Node", node.args[0])
        elif node in aliases:
            node = aliases[node]
        else:
            return node


def _is_swap_permute(node: object) -> bool:
    """A permute / transpose that swaps the two trailing dims."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if node.target is torch.ops.aten.permute.default:
        perm = list(node.args[1])  # type: ignore[arg-type]
        return perm == [*range(len(perm) - 2), len(perm) - 1, len(perm) - 2]
    if node.target is torch.ops.aten.transpose.int:
        ndim = cast("torch.Tensor", node.meta["val"]).ndim
        dims = {
            int(cast("int", node.args[1])) % ndim,
            int(cast("int", node.args[2])) % ndim,
        }
        return dims == {ndim - 2, ndim - 1}
    return False


def _is_pointwise(node: Node) -> bool:
    from ..inductor_lowering import PointwiseLowering

    if isinstance(node.meta.get("lowering"), PointwiseLowering):
        return True
    return node.op == "call_function" and node.target in (
        torch.ops.aten.where.self,
        torch.ops.aten.scalar_tensor.default,
    )


def _uniform_input(node: Node) -> bool:
    """Whether a chain input is uniform over the tile (0-d / size-1 dims)."""
    val = node.meta.get("val")
    return not isinstance(val, torch.Tensor) or all(
        isinstance(size, int) and size == 1 for size in val.shape
    )


_LOSSLESS_WIDTH = {
    torch.float16: 1,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


def _exact_round_trip(chain: tuple[object, ...], dtype: torch.dtype) -> bool:
    """Whether a pointwise chain over a load of ``dtype`` (a chain key, listed
    from the value down to the source) only widens the elements losslessly
    (16-bit -> fp32 / fp64, fp32 -> fp64) and narrows them back to ``dtype``:
    the identity on the bits.  Anything else (another op, a narrowing on
    the way, a different 16-bit type) is a real pass."""
    if dtype not in _LOSSLESS_WIDTH or not chain:
        return False
    targets: list[torch.dtype] = []
    for entry in chain:
        if not isinstance(entry, tuple) or len(entry) < 4:
            return False
        target, args, kwargs, inputs = entry[0], entry[1], entry[2], entry[3]
        if (
            target is not torch.ops.prims.convert_element_type.default
            or kwargs
            or inputs
            or not isinstance(args, tuple)
            or len(args) != 2
            or args[0] != "source"
            or not isinstance(args[1], torch.dtype)
            or args[1] not in _LOSSLESS_WIDTH
        ):
            return False
        targets.append(args[1])
    if targets[0] != dtype:
        return False
    current = dtype
    for target in reversed(targets[1:]):
        if target != current and _LOSSLESS_WIDTH[target] <= _LOSSLESS_WIDTH[current]:
            return False
        current = target
    return True


def _chain_key(chain: Sequence[tuple[object, ...]]) -> tuple[object, ...]:
    """The sharing key of a pointwise chain (listed from the operand value
    down to the source).

    A transposed read of the same values shares the tile -- the transpose
    is a property of the read, not of the tile -- so permute markers are
    dropped.  An op with a vector / tile input depends on which way the
    source is turned under it, so such an entry carries the parity of the
    permutes between it and the source (``(attn - col).T`` vs ``attn.T -
    col`` differ; ``where(mask, dA, 0)`` read straight and transposed do not).
    """
    permutes_below = sum(1 for entry in chain if len(entry) == 1)
    key: list[object] = []
    for entry in chain:
        if len(entry) == 1:
            permutes_below -= 1
            continue
        inputs = cast("tuple[Node, ...]", entry[3])
        orientation = None if all(map(_uniform_input, inputs)) else permutes_below % 2
        key.append((*entry, orientation))
    return tuple(key)


def _chain_arg(arg: object, source: Node) -> object:
    """An argument of a pointwise op for the sharing key: the traced source,
    another node, or the literal itself (lists element-wise)."""
    if isinstance(arg, torch.fx.Node):
        return "source" if arg is source else "node"
    if isinstance(arg, (list, tuple)):
        return tuple(_chain_arg(item, source) for item in arg)
    return arg


def _node_inputs(node: Node) -> list[Node]:
    from torch.fx.node import map_arg

    inputs: list[Node] = []

    def visit(value: Node) -> Node:
        inputs.append(value)
        return value

    map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
    return inputs


def _bound_symbols(bound: object) -> set[sympy.Symbol]:
    """The free symbols of a loop bound (a Python int, or an FX node whose
    value is an int / SymInt)."""
    if isinstance(bound, torch.fx.Node):
        val = bound.meta.get("val")
        if isinstance(val, torch.SymInt):
            return {cast("sympy.Symbol", s) for s in val._sympy_().free_symbols}
    return set()


class _NotMine(Exception):
    """Raised inside the matcher for any shape it does not own."""


@dataclasses.dataclass
class _LoopBuild:
    """A device loop under construction (registered before its body is
    walked, so the walkers can cross its boundary)."""

    index: int
    parent: int
    depth: int
    node: Node
    graph: GraphInfo
    placeholders: list[Node]
    args: list[Node]
    carry_placeholders: set[Node] = dataclasses.field(default_factory=set)
    vec_carries: list[VecCarry] = dataclasses.field(default_factory=list)


class _Matcher:
    def __init__(self, graphs: Sequence[GraphInfo]) -> None:
        self.graphs = graphs
        self.axes: dict[int, Axis] = {}
        self.axis_list: list[Axis] = []
        self.root_block_ids: tuple[int, ...] = ()
        # Canonical block id -> loop index of the tile loops and of the
        # grid-like (fixed step one) loops; the latter's index symbols.
        self.loop_blocks: dict[int, int] = {}
        self.grid_blocks: dict[int, int] = {}
        self.loop_symbols: dict[int, sympy.Symbol] = {}
        self.frag_slots: dict[Node, int] = {}
        self.slots: list[tuple[Axis, Axis]] = []
        self.carry_placeholders: set[Node] = set()
        # Fragment leaves whose registers hold the FX value transposed.
        self.transposed: set[Node] = set()
        # Program order of every node, loop bodies numbered at their call.
        self.order: dict[Node, int] = {}
        # Loop placeholders aliasing a value of the enclosing region.
        self.aliases: dict[Node, Node] = {}
        self.loop_builds: list[_LoopBuild] = []
        self.loops: list[Loop | None] = []
        self.loop_calls: dict[Node, int] = {}
        self.graph_by_id: dict[int, GraphInfo] = {}
        self.dots: list[Dot] = []
        self.stores: list[Store] = []
        self.derived: list[Derived] = []
        self.acc_loads: dict[Node, tuple[Access, ...]] = {}
        # Fragment leaf -> the region (loop index, -1 root) defining it.
        self.region_of: dict[Node, int] = {}
        # Vector slots, their leaves, and the steps producing / storing them.
        self.vec_slot_list: list[VecSlot] = []
        self.vec_leaves: dict[Node, int] = {}
        self.reduces: list[Reduce] = []
        self.vec_ops: list[VecOp] = []
        self.vec_stores: list[VecStore] = []
        # Union-find over vector slots (carries, ops and reductions link
        # them) and the fragment slots reduced into each.
        self.vec_parent: list[int] = []
        self.vec_frags: dict[int, list[int]] = {}
        # ``tile.index`` values (``_tile_index_position``) -> their root axis.
        self.tile_index_loads: dict[Node, Axis] = {}

    # -- axes ------------------------------------------------------------------

    def canonical(self, block_id: int) -> int:
        return self.env.canonical_block_id(block_id)

    def axis_of_block(
        self, block_id: int, kind: str | None = None, loop: int = -1
    ) -> Axis:
        block_id = self.canonical(block_id)
        axis = self.axes.get(block_id)
        if axis is None:
            info = self.env.block_sizes[block_id]
            if kind is None:
                if block_id in self.root_block_ids:
                    kind = AXIS_ROOT
                elif info.reduction:
                    kind = AXIS_RDIM
                elif block_id in self.loop_blocks:
                    kind = AXIS_LOOP
                    loop = self.loop_blocks[block_id]
                else:
                    raise _NotMine
            extent = _static_int(info.size)
            if extent is None or extent <= 0:
                raise _NotMine
            if kind == AXIS_RDIM and (extent & (extent - 1) or extent % MMA_DAG_K_STEP):
                raise _NotMine
            axis = Axis(len(self.axis_list), block_id, kind, extent, loop)
            self.axes[block_id] = axis
            self.axis_list.append(axis)
        return axis

    def axis_of_size(self, size: object) -> Axis | None:
        """The axis behind a value dim size (None for a size-1 dim)."""
        static = _static_int(size)
        if static == 1:
            return None
        block_id = self.env.resolve_block_id(size)
        if block_id is None:
            raise _NotMine
        return self.axis_of_block(block_id)

    def shape_axes(self, node: Node) -> tuple[Axis | None, ...]:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            raise _NotMine
        return tuple(self.axis_of_size(size) for size in val.shape)

    def tile_dims(self, node: Node) -> tuple[Axis, ...]:
        """The fragment-sized dims of ``node`` (size-1 and lead dims dropped)."""
        return tuple(
            axis
            for axis in self.shape_axes(node)
            if axis is not None and axis.kind != AXIS_LEAD
        )

    # -- helpers ------------------------------------------------------------------

    def is_frag_leaf(self, node: object) -> bool:
        return isinstance(node, torch.fx.Node) and self.resolve(node) in self.frag_slots

    def is_tile_id(self, subscript: object) -> bool:
        """A ``tile.id`` symbol (or the ``tile.begin`` of a leading tile,
        whose block size is one): a scalar coordinate, not a tile."""
        from ..host_function import HostFunction
        from ..variable_origin import TileBeginOrigin
        from ..variable_origin import TileIdOrigin

        if not isinstance(subscript, torch.fx.Node):
            return False
        val = subscript.meta.get("val")
        if not isinstance(val, torch.SymInt):
            return False
        expr = val._sympy_()
        info = HostFunction.current().expr_to_origin.get(expr)
        if info is None:
            return False
        if isinstance(info.origin, TileIdOrigin):
            return True
        return isinstance(info.origin, TileBeginOrigin) and self.is_lead_block(
            info.origin.block_id
        )

    def is_lead_block(self, block_id: int) -> bool:
        axis = self.axes.get(self.canonical(block_id))
        return axis is not None and axis.kind == AXIS_LEAD

    def lead_coordinate(self, origin: object) -> bool:
        """Whether a symbol of ``origin`` is a leading tile's coordinate (its
        id, or its begin: lead tiles have block size one); ``_NotMine`` for
        the id or begin of any other tile."""
        from ..variable_origin import TileBeginOrigin
        from ..variable_origin import TileIdOrigin

        if not isinstance(origin, (TileIdOrigin, TileBeginOrigin)):
            return False
        if self.axis_of_block(origin.block_id).kind != AXIS_LEAD:
            raise _NotMine
        return True

    def uniform_expr(self, subscript: object) -> tuple[sympy.Expr, tuple[int, ...]]:
        """A uniform integer subscript and the grid loops whose index occurs
        in it."""

        if isinstance(subscript, bool):
            raise _NotMine
        if isinstance(subscript, int):
            return sympy.Integer(subscript), ()
        if not isinstance(subscript, torch.fx.Node):
            raise _NotMine
        val = subscript.meta.get("val")
        if isinstance(val, int) and not isinstance(val, bool):
            return sympy.Integer(val), ()
        if not isinstance(val, torch.SymInt):
            raise _NotMine
        return self.uniform_sym(val._sympy_())

    def uniform_sym(self, expr: object) -> tuple[sympy.Expr, tuple[int, ...]]:
        """``uniform_expr`` of a sympy integer expression."""
        from ..host_function import HostFunction
        from ..variable_origin import GridOrigin
        from ..variable_origin import TensorSizeOrigin

        if not isinstance(expr, sympy.Expr):
            raise _NotMine
        expr_to_origin = HostFunction.current().expr_to_origin
        loops: list[int] = []
        for symbol in expr.free_symbols:
            info = expr_to_origin.get(cast("sympy.Symbol", symbol))
            if info is None:
                raise _NotMine
            origin = info.origin
            if type(origin) is GridOrigin:
                loop = self.grid_blocks.get(self.canonical(origin.block_id))
                if loop is None:
                    raise _NotMine
                known = self.loop_symbols.setdefault(loop, cast("sympy.Symbol", symbol))
                if known != symbol:
                    raise _NotMine
                if loop not in loops:
                    loops.append(loop)
            elif self.lead_coordinate(origin):
                pass
            elif not isinstance(origin, TensorSizeOrigin):
                raise _NotMine
        return expr, tuple(sorted(loops))

    def tile_offset(
        self, offset: object, axis: Axis, extent: int | None, stride: int, itemsize: int
    ) -> sympy.Expr:
        """The uniform offset of an ``axis`` subscript ``tile.index + offset``
        (``_NotMine`` unless it is a nonnegative affine form of the leading
        tile coordinates that keeps every tile of ``axis`` inside the
        ``extent``-long tensor dim, so the access needs no mask, and keeps
        16-byte alignment of the dim's byte offsets)."""
        from ..host_function import HostFunction
        from ..variable_origin import GridOrigin

        if isinstance(offset, torch.SymInt):
            offset = offset._sympy_()
        elif isinstance(offset, int) and not isinstance(offset, bool):
            offset = sympy.Integer(offset)
        expr, loops = self.uniform_sym(offset)
        if loops or extent is None:
            raise _NotMine
        expr_to_origin = HostFunction.current().expr_to_origin
        symbols = sorted(expr.free_symbols, key=str)
        highest: dict[sympy.Symbol, int] = {}
        for symbol in symbols:
            info = expr_to_origin.get(cast("sympy.Symbol", symbol))
            assert info is not None
            origin = info.origin
            if not self.lead_coordinate(origin):
                raise _NotMine
            assert isinstance(origin, GridOrigin)
            highest[cast("sympy.Symbol", symbol)] = (
                self.axis_of_block(origin.block_id).extent - 1
            )
        try:
            poly = sympy.Poly(expr, *symbols) if symbols else None
        except sympy.PolynomialError:
            raise _NotMine from None
        coefficients = (
            [sympy.Integer(expr)] if poly is None else [*poly.coeffs(), poly.TC()]
        )
        if poly is not None and poly.total_degree() > 1:
            raise _NotMine
        for coefficient in coefficients:
            if not coefficient.is_integer or coefficient < 0:
                raise _NotMine
            if int(coefficient) * stride * itemsize % 16:
                raise _NotMine
        largest = int(expr.xreplace(highest))
        if largest + axis.extent > extent:
            raise _NotMine
        return expr

    def bound_varies(self, val: torch.SymInt) -> bool:
        """Whether a grid loop bound varies within the kernel: any symbol
        that is not a host value (a tensor size, an int argument) or a lead
        tile's id -- an enclosing grid loop's index, a device tile's
        position, a value minted in device code (``int or int`` on a loop
        index), or a symbol of unknown origin.  The emitter renders bounds
        with the host's symbols, and an enclosing loop evaluates a nested
        loop's bounds ahead of its iteration (the early ring prologue)."""
        from ..host_function import HostFunction
        from ..variable_origin import TileBeginOrigin
        from ..variable_origin import TileIdOrigin

        expr_to_origin = HostFunction.current().expr_to_origin
        for symbol in val._sympy_().free_symbols:
            info = expr_to_origin.get(cast("sympy.Symbol", symbol))
            if info is None:
                return True
            origin = info.origin
            if isinstance(origin, (TileIdOrigin, TileBeginOrigin)):
                if self.axis_of_block(origin.block_id).kind != AXIS_LEAD:
                    return True
            elif not origin.is_host():
                return True
        return False

    def access(self, node: Node) -> Access:
        """Decode a load / store site; ``_NotMine`` for unsupported indexing."""
        from ...language import memory_ops
        from ..indexing_strategy import subscript_tile_info
        from .fx_matcher import _host_tensor_ref

        is_load = _is_call(node, memory_ops.load)
        if not is_load and not _is_call(node, memory_ops.store):
            raise _NotMine
        if is_load:
            if (
                len(node.args) < 4
                or node.args[2] is not None
                or node.args[3] is not None
            ):
                raise _NotMine
        elif len(node.args) < 4 or node.args[3] is not None:
            raise _NotMine
        ref = _host_tensor_ref(node.args[0])
        indices = node.args[1]
        if ref is None or not isinstance(indices, (list, tuple)):
            raise _NotMine
        fake = ref.fake
        if fake.ndim != len(indices):
            raise _NotMine
        strides: list[int] = []
        for stride in fake.stride():
            static = _static_int(stride)
            if static is None:
                raise _NotMine
            strides.append(static)
        value = node.meta.get("val") if is_load else node.args[2]
        value_fake = (
            value.meta.get("val") if isinstance(value, torch.fx.Node) else value
        )
        if not isinstance(value_fake, torch.Tensor):
            raise _NotMine
        value_sizes = list(value_fake.shape)
        subscripts: list[Subscript] = []
        dims: list[Axis] = []
        dim_strides: list[int] = []
        for dim, index in enumerate(indices):
            extent = _static_int(fake.shape[dim])
            if index == slice(None):
                if not value_sizes:
                    raise _NotMine
                size = value_sizes.pop(0)
                block_id = self.env.resolve_block_id(size)
                if block_id is None:
                    raise _NotMine
                axis = self.axis_of_block(block_id)
                if axis.kind != AXIS_RDIM or axis.extent != extent:
                    raise _NotMine
                subscripts.append(Subscript("axis", axis))
                dims.append(axis)
                dim_strides.append(strides[dim])
                continue
            if isinstance(index, slice):
                raise _NotMine
            tile = (
                None if self.is_tile_id(index) else subscript_tile_info(self.env, index)
            )
            tile_blocks = [*self.root_block_ids, *self.loop_blocks]
            if tile is not None and self.canonical(tile.block_id) in tile_blocks:
                if not value_sizes:
                    raise _NotMine
                axis = self.axis_of_block(tile.block_id)
                size = value_sizes.pop(0)
                size_block = self.env.resolve_block_id(size)
                if size_block is None or self.canonical(size_block) != axis.block_id:
                    raise _NotMine
                offset = None
                if not self.env.known_equal(tile.offset, 0):
                    offset = self.tile_offset(
                        tile.offset, axis, extent, strides[dim], fake.dtype.itemsize
                    )
                elif extent is None or extent != axis.extent:
                    raise _NotMine
                subscripts.append(Subscript("axis", axis, offset=offset))
                dims.append(axis)
                dim_strides.append(strides[dim])
                continue
            expr, loops = self.uniform_expr(index)
            subscripts.append(Subscript("scalar", None, expr, bool(loops), loops))
        if value_sizes:
            raise _NotMine
        return Access(
            node=node,
            fake=fake,
            name=ref.name,
            strides=tuple(strides),
            subscripts=tuple(subscripts),
            dims=tuple(dims),
            dim_strides=tuple(dim_strides),
        )

    def resolve(self, node: Node) -> Node:
        """``node`` through its ``_new_var`` copies, ``_mask_to`` and the
        loop placeholders aliasing a value of the enclosing region."""
        return _resolve(node, self.aliases)

    def vec_sources(self, node: Node) -> list[Node]:
        """The vector leaves under ``node`` (through pointwise ops, views,
        ``_new_var`` and aliases), in input order, each once."""
        from ...language import memory_ops
        from ...language import view_ops

        result: list[Node] = []
        seen: set[Node] = set()

        def visit(current: Node) -> None:
            current = self.resolve(current)
            if current in seen:
                return
            seen.add(current)
            if current in self.vec_leaves:
                result.append(current)
                return
            if current.op != "call_function" or _is_call(current, memory_ops.load):
                return
            if (
                _is_call(current, view_ops.subscript)
                or _is_call(current, torch.ops.aten.expand.default)
                or _is_pointwise(current)
            ):
                for inp in _node_inputs(current):
                    visit(inp)

        visit(node)
        return result

    def users_of(self, node: Node) -> list[Node]:
        """``node.users`` with loop boundaries crossed: a value passed to a
        ``_for_loop`` continues at the body placeholder(s) it binds."""
        from ...language import _tracing_ops

        result: list[Node] = []
        for user in node.users:
            if _is_call(user, _tracing_ops._for_loop) and user in self.loop_calls:
                build = self.loop_builds[self.loop_calls[user]]
                result.extend(
                    placeholder
                    for placeholder, arg in zip(
                        build.placeholders, build.args, strict=True
                    )
                    if arg is node
                )
            else:
                result.append(user)
        return result

    def has_source(self, node: Node, memo: dict[Node, bool]) -> bool:
        """Whether a 2-D tile load or a fragment lies under ``node``."""
        from ...language import memory_ops

        if node in memo:
            return memo[node]
        resolved = self.resolve(node)
        if self.is_frag_leaf(resolved):
            result = True
        elif _is_call(resolved, memory_ops.load):
            result = len(self.tile_dims(resolved)) == 2
        elif resolved.op != "call_function":
            result = False
        else:
            result = any(self.has_source(inp, memo) for inp in _node_inputs(resolved))
        memo[node] = result
        return result

    def aux_dag(self, node: Node, aux: list[Access], seen: set[Node]) -> None:
        """Admit a value with no 2-D source: loads of at most one tile dim,
        symnodes, iota, scalar tensors, broadcasts, permutes, pointwise."""
        from ...language import _tracing_ops
        from ...language import memory_ops
        from ...language import view_ops

        if node in seen:
            return
        seen.add(node)
        if node in self.aliases:
            self.aux_dag(self.aliases[node], aux, seen)
            return
        if node.op != "call_function":
            raise _NotMine
        if _is_call(node, torch.ops.aten.expand.default):
            # ``broadcast_to``: the output dims are the input's or size one.
            self.shape_axes(node)
            self.aux_dag(cast("Node", node.args[0]), aux, seen)
            return
        position = _tile_index_position(node)
        if position is not None:
            # A root tile's global indices: the CTA's base plus the
            # coordinate (``hl.arange`` relabels the same way).
            axis = self.shape_axes(node)[position]
            val = node.meta.get("val")
            if (
                axis is None
                or axis.kind != AXIS_ROOT
                or self.shape_axes(cast("Node", node.args[0])) != (axis,)
                or not isinstance(val, torch.Tensor)
                or val.dtype != torch.int32
            ):
                raise _NotMine
            self.tile_index_loads[node] = axis
            return
        if _is_call(node, memory_ops.load):
            access = self.access(node)
            if len(access.tile_dims) > 1 or access.dtype not in _SIDE_DTYPES:
                raise _NotMine
            aux.append(access)
            return
        if _is_call(node, _tracing_ops._get_symnode):
            if not isinstance(node.meta.get("val"), (torch.SymInt, torch.SymFloat)):
                raise _NotMine
            return
        if _is_call(node, torch.ops.prims.iota.default):
            if (
                len(node.args) != 1
                or node.kwargs.get("start", 0) != 0
                or node.kwargs.get("step", 1) != 1
            ):
                raise _NotMine
            self.shape_axes(node)
            return
        if _is_call(node, torch.ops.aten.scalar_tensor.default):
            if not isinstance(node.args[0], (int, float)):
                raise _NotMine
            return
        if _is_call(node, _tracing_ops._new_var) or _is_call(
            node, _tracing_ops._mask_to
        ):
            self.aux_dag(cast("Node", node.args[0]), aux, seen)
            return
        if _is_call(node, view_ops.subscript):
            index = node.args[1] if len(node.args) > 1 else None
            if not isinstance(index, (list, tuple)) or any(
                item is not None and item != slice(None) for item in index
            ):
                raise _NotMine
            self.shape_axes(node)
            self.aux_dag(cast("Node", node.args[0]), aux, seen)
            return
        if _is_swap_permute(node):
            self.shape_axes(node)
            self.aux_dag(cast("Node", node.args[0]), aux, seen)
            return
        if _is_pointwise(node):
            self.shape_axes(node)
            for inp in _node_inputs(node):
                self.aux_dag(inp, aux, seen)
            return
        raise _NotMine

    def trace_source(
        self, node: Node
    ) -> tuple[Node, bool, list[Access], tuple[object, ...], bool]:
        """Follow the single 2-D source path under ``node``.

        Returns the source (a 2-D load or a fragment leaf), whether the
        trailing dims were swapped on the way, the aux loads, the chain key
        of the pointwise ops between (for exchange sharing) and whether any
        pointwise op sits between.
        """
        from ...language import _tracing_ops
        from ...language import memory_ops

        memo: dict[Node, bool] = {}
        aux: list[Access] = []
        seen: set[Node] = set()
        chain: list[tuple[object, ...]] = []
        swap = False
        pointwise = False
        current = node
        for _ in range(64):
            if _is_call(current, _tracing_ops._new_var) or _is_call(
                current, _tracing_ops._mask_to
            ):
                current = cast("Node", current.args[0])
                continue
            if current in self.aliases:
                current = self.aliases[current]
                continue
            if self.is_frag_leaf(current):
                return current, swap, aux, _chain_key(chain), pointwise
            if _is_call(current, memory_ops.load):
                if len(self.tile_dims(current)) != 2:
                    raise _NotMine
                return current, swap, aux, _chain_key(chain), pointwise
            if _is_swap_permute(current):
                swap = not swap
                chain.append(("permute",))
                current = cast("Node", current.args[0])
                continue
            if not _is_pointwise(current):
                raise _NotMine
            if len(self.tile_dims(current)) != 2:
                raise _NotMine
            pointwise = True
            inputs = _node_inputs(current)
            sources = [inp for inp in inputs if self.has_source(inp, memo)]
            if len(sources) != 1:
                raise _NotMine
            for inp in inputs:
                if inp is not sources[0]:
                    self.aux_dag(inp, aux, seen)
            # The chain key names the op, where the source sits among its
            # arguments, its non-node arguments and its other inputs.
            chain.append(
                (
                    current.target,
                    tuple(_chain_arg(arg, sources[0]) for arg in current.args),
                    tuple(sorted((k, repr(v)) for k, v in current.kwargs.items())),
                    tuple(inp for inp in inputs if inp is not sources[0]),
                )
            )
            current = sources[0]
        raise _NotMine

    def slot_dims(self, leaf: Node) -> tuple[Axis, Axis]:
        """The (rows, cols) axes of the registers holding fragment ``leaf``
        (a transposed producer holds its FX value's dims swapped)."""
        return self.slots[self.frag_slots[leaf]]

    def register_swap(self, leaf: Node, swap: bool) -> bool:
        """``swap`` (FX permute parity from ``leaf`` to a value) relative to
        the registers of ``leaf``."""
        return swap != (leaf in self.transposed)

    def operand(
        self,
        role: str,
        node: object,
        dot_dims: tuple[Axis, Axis],
        *,
        flip: bool = False,
    ) -> Operand:
        """The operand of role ``role`` read from value ``node``; ``flip``
        says the dot is emitted transposed, so the value's dims are the
        logical operand dims swapped (``dot_dims`` are the logical ones)."""
        if not isinstance(node, torch.fx.Node):
            raise _NotMine
        value = node
        val = value.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype not in _operand_dtypes():
            raise _NotMine
        operand_dims = self.tile_dims(value)
        if len(operand_dims) != 2:
            raise _NotMine
        if flip:
            operand_dims = (operand_dims[1], operand_dims[0])
        # The operand's logical (M, K) / (K, N) must be the dot's axes.
        m_or_n, k = dot_dims
        expected = (m_or_n, k) if role == "a" else (k, m_or_n)
        if operand_dims != expected:
            raise _NotMine
        source, swap, aux, chain, pointwise = self.trace_source(value)
        if self.is_frag_leaf(source):
            source_dims = self.slot_dims(source)
            swap = self.register_swap(source, swap)
            frame = self.region_of.get(source, -1)
        else:
            source_dims = self.tile_dims(source)
            frame = self.innermost_loop(self.access(source).loops)
        if len(source_dims) != 2:
            raise _NotMine
        # The operand tile is written once per execution of its frame (the
        # exchange of a fragment's region, a staged or ring tile's loop):
        # the pointwise chain's aux loads must not vary below it.
        if not self.aux_within(aux, frame):
            raise _NotMine
        logical_swap = swap != flip
        if (
            tuple(reversed(source_dims)) if logical_swap else source_dims
        ) != operand_dims:
            raise _NotMine
        k_source_pos = (
            (0 if logical_swap else 1) if role == "a" else (1 if logical_swap else 0)
        )
        # A tf32 B operand whose K runs along the natural tile's rows is
        # stored transposed ([N, K]): ``ldmatrix.trans`` cannot read fp32,
        # and per-lane loads cost 16 per atom and K step.
        transposed_tile = role == "b" and val.dtype == torch.float32
        if self.is_frag_leaf(source):
            rows, cols = source_dims
            trans = k_source_pos == 0
            transposed_tile = transposed_tile and trans
            if transposed_tile:
                rows, cols = cols, rows
                trans = False
                chain = (*chain, "transposed")
            return Operand(
                role=role,
                kind="frag",
                value=value,
                load=None,
                frag=source,
                pointwise=pointwise,
                swap=swap,
                rows=rows,
                cols=cols,
                trans=trans,
                aux_loads=tuple(aux),
                chain_key=(source, chain),
                flip=flip,
                transposed_tile=transposed_tile,
            )
        access = self.access(source)
        if access.tile_dims != source_dims:
            raise _NotMine
        if pointwise and _exact_round_trip(chain, access.dtype):
            # ``x.float().to(x.dtype)`` leaves every element's bits as
            # loaded: the tile is staged as loaded, no per-element pass.
            pointwise = False
            chain = ()
        contiguous = [
            position
            for position, stride in enumerate(access.tile_dim_strides)
            if stride == 1
        ]
        if len(contiguous) != 1:
            raise _NotMine
        cols_pos = contiguous[0]
        rows_pos = 1 - cols_pos
        rows, cols = source_dims[rows_pos], source_dims[cols_pos]
        if (access.tile_dim_strides[rows_pos] * access.dtype.itemsize) % 4:
            raise _NotMine
        # The tile is staged as loaded, or rewritten per element (a 16-bit
        # tile in place or from an fp32 source, an fp32 operand from a 16-bit
        # source; the in-place pass rewrites 16-bit packets only).
        if not pointwise and access.dtype != val.dtype:
            raise _NotMine
        if pointwise and access.dtype not in (*_OPERAND_DTYPES, torch.float32):
            raise _NotMine
        if pointwise and access.dtype == val.dtype == torch.float32:
            raise _NotMine
        trans = k_source_pos == rows_pos
        # (Only the widen pass scatters into the transposed tile: an
        # unaligned source keeps the natural tile and per-lane B reads.)
        transposed_tile = (
            transposed_tile
            and trans
            and access.dtype != val.dtype
            and async_tile_reason(CompileEnvironment.current(), access, cols_pos)
            is None
        )
        if transposed_tile:
            rows, cols = cols, rows
            trans = False
            chain = (*chain, "transposed")
        return Operand(
            role=role,
            kind="load",
            value=value,
            load=access,
            frag=None,
            pointwise=pointwise,
            swap=swap,
            rows=rows,
            cols=cols,
            trans=trans,
            aux_loads=tuple(aux),
            chain_key=(source, chain),
            flip=flip,
            rows_pos=rows_pos,
            transposed_tile=transposed_tile,
        )

    def leaf_consumers(self, leaf: Node) -> list[Node]:
        """The nodes consuming ``leaf``'s value, looked through the
        transparent nodes (``_new_var``, permutes, views, pointwise ops)."""
        from ...language import _tracing_ops
        from ...language import view_ops

        consumers: list[Node] = []
        seen: set[Node] = set()
        stack = [leaf]
        while stack:
            for user in self.users_of(stack.pop()):
                if user in seen:
                    continue
                seen.add(user)
                if _is_call(user, torch.ops.aten.sym_size.int) or _is_call(
                    user, torch.ops.aten.sym_stride.int
                ):
                    # Shape queries read no data.
                    continue
                if (
                    _is_call(user, _tracing_ops._new_var)
                    or _is_call(user, _tracing_ops._mask_to)
                    or user in self.aliases
                    or _is_swap_permute(user)
                    or _is_call(user, view_ops.subscript)
                    or _is_pointwise(user)
                ):
                    stack.append(user)
                else:
                    consumers.append(user)
        return consumers

    def accumulates_in_place(self, node: Node, leaf: Node, loop: int) -> bool:
        """Whether dot ``node`` may accumulate into ``leaf``'s own registers.

        The dots and stores of a region are emitted in program order and a
        dot's accumulator is materialized right before its MMAs, so every
        other consumer of the leaf must come earlier in program order (it
        has read the registers by then) -- or be a loop-entry snapshot of
        the carry, which the loop body writes before any dot.  A later
        consumer would read the accumulated value: it forces a fresh slot.
        So does a leaf of an enclosing region (or of an earlier loop): the
        next iteration reads its registers again.
        """
        from ...language import memory_ops

        if self.region_of.get(leaf) != loop:
            return False
        order = self.order
        for consumer in self.leaf_consumers(leaf):
            if consumer is node or order[consumer] < order[node]:
                continue
            if (
                loop >= 0
                and _is_call(consumer, memory_ops.store)
                and leaf in self.loop_builds[loop].carry_placeholders
            ):
                continue
            return False
        return True

    def frag_value(self, node: object) -> tuple[Node, bool, list[Access]]:
        """A pointwise DAG over one fragment leaf: the leaf, whether the
        value is the leaf's registers transposed and the aux loads."""
        if not isinstance(node, torch.fx.Node):
            raise _NotMine
        source, swap, aux, _chain, _pointwise = self.trace_source(node)
        if not self.is_frag_leaf(source):
            raise _NotMine
        return source, self.register_swap(source, swap), aux

    def sources(self, node: Node, leaf: Node | None = None) -> list[Node]:
        """The fragment leaves and 2-D tile loads under ``node`` (through
        pointwise ops, permutes, subscripts and ``_new_var``), in input
        order, each once; ``leaf`` counts as a fragment leaf (a dot not yet
        registered)."""
        from ...language import _tracing_ops
        from ...language import memory_ops
        from ...language import view_ops

        result: list[Node] = []
        seen: set[Node] = set()

        def visit(current: Node) -> None:
            current = self.resolve(current)
            if current in seen:
                return
            seen.add(current)
            if current is leaf or self.is_frag_leaf(current):
                result.append(current)
                return
            if current.op != "call_function":
                return
            if _is_call(current, memory_ops.load):
                if len(self.tile_dims(current)) == 2:
                    result.append(current)
                return
            if (
                _is_call(current, view_ops.subscript)
                or _is_swap_permute(current)
                or _is_pointwise(current)
                or _is_call(current, _tracing_ops._new_var)
                or _is_call(current, _tracing_ops._mask_to)
            ):
                for inp in _node_inputs(current):
                    visit(inp)

        visit(node)
        return result

    def consumed_only_transposed(self, node: Node) -> bool:
        """Whether every use of fragment ``node`` is a dot operand read
        through an odd number of permutes (the orientation rule: emitting the
        product transposed makes those reads straight)."""
        from ...language import _tracing_ops
        from ...language import view_ops

        parities: list[bool] = []

        def visit(current: Node, parity: bool) -> bool:
            for user in self.users_of(current):
                if _is_dot(user):
                    if _dot_acc(user) is current or current not in user.args[:2]:
                        return False
                    parities.append(parity)
                elif _is_swap_permute(user):
                    if not visit(user, not parity):
                        return False
                elif user in self.aliases:
                    if not visit(user, parity):
                        return False
                elif (
                    _is_call(user, _tracing_ops._new_var)
                    or _is_call(user, _tracing_ops._mask_to)
                    or _is_call(user, view_ops.subscript)
                    or _is_pointwise(user)
                ):
                    if len(self.sources(user, node)) != 1 or not visit(user, parity):
                        return False
                else:
                    return False
            return True

        return visit(node, False) and bool(parities) and all(parities)

    def dies_at(self, leaf: Node, node: Node, *, skip: Collection[Node] = ()) -> bool:
        """Whether ``node`` is the last reader of ``leaf``'s registers (a
        fragment's or a vector's), so it may take them over: every other
        consumer precedes ``node`` in program order and nothing carries the
        value out of its graph.  Copies, aliases, views and pointwise values
        that are not fragments / vectors of their own are evaluated where
        they are consumed, so the walk looks through them to their readers.
        ``skip`` names users that read nothing (a loop's own carry
        placeholder and phi when ``node`` is the loop)."""
        from ...language import _tracing_ops
        from ...language import view_ops

        stack = [leaf]
        seen: set[Node] = set()
        while stack:
            for user in self.users_of(stack.pop()):
                if user in seen or user is node or user in skip:
                    continue
                seen.add(user)
                if user.op == "output":
                    return False
                if _is_call(user, torch.ops.aten.sym_size.int) or _is_call(
                    user, torch.ops.aten.sym_stride.int
                ):
                    continue
                if (
                    _is_call(user, _tracing_ops._new_var)
                    or _is_call(user, _tracing_ops._mask_to)
                    or user in self.aliases
                ):
                    stack.append(user)
                    continue
                if self.order.get(user, -1) >= self.order[node]:
                    return False
                if user in self.frag_slots or user in self.vec_leaves:
                    continue
                if (
                    _is_swap_permute(user)
                    or _is_call(user, view_ops.subscript)
                    or _is_call(user, torch.ops.aten.expand.default)
                    or _is_pointwise(user)
                ):
                    stack.append(user)
        return True

    # -- the match ----------------------------------------------------------

    def match(self) -> MmaDagProgram | None:
        try:
            return self._match()
        except _NotMine:
            return None

    def _number(self, graph: GraphInfo, counter: itertools.count[int]) -> None:
        """Program order of every node: a loop body is numbered at its
        ``_for_loop`` call, so the order compares across graphs."""
        from ...language import _tracing_ops

        for node in graph.graph.nodes:
            self.order[node] = next(counter)
            if _is_call(node, _tracing_ops._for_loop):
                body = self.graph_by_id.get(cast("int", node.args[0]))
                if body is not None:
                    self._number(body, counter)

    def _match(self) -> MmaDagProgram:
        from ...language import _tracing_ops
        from ...language import memory_ops
        from ..device_ir import ForLoopGraphInfo
        from ..device_ir import ReductionLoopGraphInfo
        from ..device_ir import RootGraphInfo
        from ..host_function import HostFunction
        from .cute_mma import analyze_cute_mma_node

        graphs = self.graphs
        roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
        loops = [
            graph
            for graph in graphs
            if isinstance(graph, ForLoopGraphInfo)
            and not isinstance(graph, ReductionLoopGraphInfo)
        ]
        if len(roots) != 1 or not loops or len(graphs) != 1 + len(loops):
            raise _NotMine
        root = roots[0]
        self.graph_by_id = {graph.graph_id: graph for graph in loops}
        self._number(root, itertools.count())
        if len(self.order) != sum(len(graph.graph.nodes) for graph in graphs):
            # A loop graph no ``_for_loop`` call reaches.
            raise _NotMine
        # The structural gate above needs no compile state; everything from
        # here on resolves block ids and symbol origins through the current
        # environment and host function.
        self.env = CompileEnvironment.current()
        device_ir = HostFunction.current().device_ir
        # Contractions the tensor-core families already claim (tcgen05
        # collective / warp-MMA search, or a dot their operand analyzer
        # accepts) keep their search surface: this owner exists for the
        # kernels that would otherwise run SIMT.
        if self.env.config_spec.cute_tcgen05_search_enabled:
            raise _NotMine
        for graph in graphs:
            for node in graph.graph.nodes:
                if _is_dot(node) and (
                    analyze_cute_mma_node(node, graphs=graphs) is not None
                ):
                    raise _NotMine
        if len(device_ir.grid_block_ids) != 1:
            raise _NotMine
        self.root_block_ids = tuple(
            self.canonical(b) for b in device_ir.grid_block_ids[0]
        )
        if len(set(self.root_block_ids)) != len(self.root_block_ids):
            raise _NotMine
        (root_output,) = root.graph.find_nodes(op="output")
        if root_output.args != (None,):
            raise _NotMine
        # Loop axes first (in loop order), then the root axes: the loops'
        # blocks are known from their calls.
        calls = sorted(
            (
                node
                for graph in graphs
                for node in graph.graph.nodes
                if _is_call(node, _tracing_ops._for_loop)
            ),
            key=lambda node: self.order[node],
        )
        for index, call in enumerate(calls):
            graph = self.graph_by_id.get(cast("int", call.args[0]))
            if not isinstance(graph, ForLoopGraphInfo):
                raise _NotMine
            for block_id in graph.block_ids:
                block_id = self.canonical(block_id)
                if block_id in self.loop_blocks or block_id in self.grid_blocks:
                    raise _NotMine
                source = self.env.block_sizes[block_id].block_size_source
                if isinstance(source, FixedBlockSizeSource):
                    self.grid_blocks[block_id] = index
                else:
                    self.loop_blocks[block_id] = index
                    self.axis_of_block(block_id, AXIS_LOOP, loop=index)

        # The dots tell which root blocks tile fragments; the rest are leads.
        dot_nodes = [
            node for graph in graphs for node in graph.graph.nodes if _is_dot(node)
        ]
        if not dot_nodes:
            raise _NotMine
        tiled_blocks: set[int] = set()
        for node in dot_nodes:
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor) or (
                val.dtype != torch.float32 and rounded_dot_dtype(node) is None
            ):
                raise _NotMine
            for size in val.shape[-2:]:
                block_id = self.env.resolve_block_id(size)
                if block_id is not None:
                    tiled_blocks.add(self.canonical(block_id))
            for operand in node.args[:2]:
                if isinstance(operand, torch.fx.Node):
                    operand_val = operand.meta.get("val")
                    if isinstance(operand_val, torch.Tensor):
                        for size in operand_val.shape[-2:]:
                            block_id = self.env.resolve_block_id(size)
                            if block_id is not None:
                                tiled_blocks.add(self.canonical(block_id))
        for block_id in self.root_block_ids:
            source = self.env.block_sizes[block_id].block_size_source
            if block_id in tiled_blocks:
                if isinstance(source, FixedBlockSizeSource):
                    raise _NotMine
                self.axis_of_block(block_id, AXIS_ROOT)
            else:
                if isinstance(source, FixedBlockSizeSource) and source.value != 1:
                    raise _NotMine
                self.axis_of_block(block_id, AXIS_LEAD)

        # -- the root region (loops recursively) in program order.
        body = self._region(root, -1)
        if len(self.loop_calls) != len(loops):
            raise _NotMine
        if not self.dots or not (self.stores or self.vec_stores):
            raise _NotMine
        store_aux: list[Access] = []
        for store in self.stores:
            _leaf, _swap, aux = self.frag_value(store.value)
            store_aux.extend(aux)
        for vec_store in self.vec_stores:
            store_aux.extend(self.aux_of_vec(vec_store.value))
        # Every vector family needs a fragment to take its warp split from.
        vec_frag_slots: dict[int, tuple[int, ...]] = {}
        for slot in range(len(self.vec_slot_list)):
            root_slot = self.vec_root(slot)
            frags = sorted(
                {
                    frag
                    for other in range(len(self.vec_slot_list))
                    if self.vec_root(other) == root_slot
                    for frag in self.vec_frags.get(other, ())
                }
            )
            if not frags and self.vec_slot_list[slot].axis is not None:
                raise _NotMine
            vec_frag_slots[slot] = tuple(frags)
        completed = [cast("Loop", loop) for loop in self.loops]
        first = completed[0]
        program = MmaDagProgram(
            root_graph_id=root.graph_id,
            loop_graph_id=first.graph_id,
            loop_node=first.node,
            loop_axis=first.axes[0] if first.axes else None,
            loop_block_id=self.canonical(
                cast("ForLoopGraphInfo", self.graph_by_id[first.graph_id]).block_ids[0]
            ),
            loop_symbol=first.symbol,
            begin=first.begin,
            end=first.end,
            axes=tuple(self.axis_list),
            carries=tuple(carry for loop in completed for carry in loop.carries),
            dots=tuple(self.dots),
            stores=tuple(self.stores),
            slots=tuple(self.slots),
            frag_slots=dict(self.frag_slots),
            steps=tuple(
                cast("Dot | Derived | Store", step)
                for step in first.steps
                if not isinstance(step, Loop)
            ),
            derived=tuple(self.derived),
            loops=tuple(completed),
            body=body,
            aliases=dict(self.aliases),
            frag_regions=dict(self.region_of),
            vec_slots=tuple(self.vec_slot_list),
            vec_leaves=dict(self.vec_leaves),
            reduces=tuple(self.reduces),
            vec_ops=tuple(self.vec_ops),
            vec_stores=tuple(self.vec_stores),
            vec_frag_slots=vec_frag_slots,
            tile_index_loads=dict(self.tile_index_loads),
            acc_loads=self.acc_loads,
            store_aux_loads=tuple(store_aux),
            epilogue_order=tuple(
                step.node if isinstance(step, Dot) else step.access.node
                for step in body
                if isinstance(step, (Dot, Store))
            ),
        )
        for d in program.dots:
            if d.a.dtype != d.b.dtype:
                raise _NotMine
        # A tensor the program both loads and stores (through any name or
        # view: by storage) is not staged: its tiles and side loads would
        # be read ahead of the stores.
        stored = {
            _storage(store.access) for store in (*program.stores, *program.vec_stores)
        }
        if any(
            not _is_call(access.node, memory_ops.store) and _storage(access) in stored
            for access in program.accesses
        ):
            raise _NotMine
        return program

    def _region(self, graph: GraphInfo, loop: int) -> tuple[Step, ...]:
        """The steps of ``graph`` (the body of loop ``loop``; -1: the root)
        in program order: dots, derived fragments, stores and nested loops.
        Every load is decoded (unsupported indexing is not mine) and every
        other node must feed one of the steps."""
        from ...language import _tracing_ops
        from ...language import memory_ops

        steps: list[Step] = []
        loaded: dict[tuple[object, ...], Node] = {}
        for node in graph.graph.nodes:
            if _is_call(node, memory_ops.load) and _tile_index_position(node) is None:
                access = self.access(node)
                signature = _tile_signature(access)
                if signature is None:
                    continue
                first = loaded.setdefault(signature, node)
                if first is not node:
                    # The same whole tile loaded again in this region (dk
                    # reads dL straight and transposed): one tile, every
                    # consumer reads the first load.
                    self.aliases[node] = first
        for node in graph.graph.nodes:
            step: Step
            if _is_call(node, _tracing_ops._for_loop):
                step = self._loop(node, loop)
            elif _is_dot(node):
                step = self._dot(node, loop)
                self.dots.append(step)
            elif _is_call(node, memory_ops.store):
                step = self._store(node, loop)
                if isinstance(step, VecStore):
                    self.vec_stores.append(step)
                else:
                    self.stores.append(step)
            elif _is_call(node, torch.ops.aten.sum.dim_IntList):
                step = self._reduce(node, loop)
                self.reduces.append(step)
            elif (
                _is_pointwise(node)
                and len(self.tile_dims(node)) == 2
                and len(self.sources(node)) > 1
            ):
                step = self._derived(node, loop)
                self.derived.append(step)
            elif (
                _is_pointwise(node)
                and len(self.tile_dims(node)) <= 1
                and self.vec_sources(node)
            ):
                step = self._vec_op(node, loop)
                self.vec_ops.append(step)
            else:
                continue
            steps.append(step)
        self._check_closure(graph, steps)
        return tuple(steps)

    def _check_closure(self, graph: GraphInfo, steps: Sequence[Step]) -> None:
        """Every call in ``graph`` feeds a step (a loop's arguments and
        outputs included); the ``getitem`` / ``_phi`` nodes of its loops
        are the carries' plumbing."""
        from ...language import _tracing_ops

        anchors: list[Node] = []
        for step in steps:
            if isinstance(step, Loop):
                anchors.append(step.node)
            elif isinstance(step, (Store, VecStore)):
                anchors.append(step.access.node)
            else:
                anchors.append(step.node)
        (output,) = graph.graph.find_nodes(op="output")
        outputs = output.args[0]
        if isinstance(outputs, (list, tuple)):
            anchors.extend(node for node in outputs if isinstance(node, torch.fx.Node))
        closure = self._closure(anchors)
        for node in graph.graph.nodes:
            if node.op != "call_function" or node in closure:
                continue
            if node.target is operator.getitem and node.args[0] in self.loop_calls:
                continue
            if node.target is _tracing_ops._phi:
                continue
            raise _NotMine

    def _loop(self, node: Node, parent: int) -> Loop:
        """A ``_for_loop`` call: its tile / grid axes, carries (fragments,
        and identity vectors / scalars), read-only inputs, then the body."""
        from ...language import _tracing_ops
        from ..device_ir import ForLoopGraphInfo

        if (
            len(node.args) != 4
            or not isinstance(node.args[1], (list, tuple))
            or not isinstance(node.args[2], (list, tuple))
            or not isinstance(node.args[3], (list, tuple))
        ):
            raise _NotMine
        graph = self.graph_by_id.get(cast("int", node.args[0]))
        if not isinstance(graph, ForLoopGraphInfo) or node in self.loop_calls:
            raise _NotMine
        if any(
            build.graph is graph for build in self.loop_builds
        ):  # pragma: no cover - two calls of one graph
            raise _NotMine
        begins, ends, args = (
            list(node.args[1]),
            list(node.args[2]),
            list(node.args[3]),
        )
        block_ids = [self.canonical(b) for b in graph.block_ids]
        if not (len(block_ids) == len(begins) == len(ends)) or not block_ids:
            raise _NotMine
        index = len(self.loop_builds)
        depth = 1 if parent < 0 else self.loop_builds[parent].depth + 1
        axes: list[Axis] = []
        begin: object = begins[0]
        end: object = ends[0]
        for block_id, lo, hi in zip(block_ids, begins, ends, strict=True):
            if block_id in self.root_block_ids:
                raise _NotMine
            source = self.env.block_sizes[block_id].block_size_source
            if isinstance(source, FixedBlockSizeSource):
                # ``hl.grid`` / a Python ``range``: one index, step one.
                if source.value != 1 or len(block_ids) != 1:
                    raise _NotMine
                for bound in (lo, hi):
                    if isinstance(bound, torch.fx.Node):
                        val = bound.meta.get("val")
                        if not isinstance(val, (int, torch.SymInt)):
                            raise _NotMine
                        if isinstance(val, torch.SymInt) and self.bound_varies(val):
                            raise _NotMine
                    elif type(bound) is not int:
                        raise _NotMine
                if self.grid_blocks.get(block_id) != index:
                    raise _NotMine
            else:
                # A tile loop walks its axis' whole extent in block steps.
                if self.loop_blocks.get(block_id) != index:
                    raise _NotMine
                axis = self.axis_of_block(block_id, AXIS_LOOP, loop=index)
                axes.append(axis)
                if (
                    lo != 0
                    or _static_int(
                        hi if not isinstance(hi, torch.fx.Node) else hi.meta.get("val")
                    )
                    != axis.extent
                ):
                    raise _NotMine
        if len(axes) > 1:
            # A multi-dim tile loop has no single bound; its iteration
            # space is the product of its axes' trips.
            begin, end = 0, 0
        placeholders = list(graph.graph.find_nodes(op="placeholder"))
        (output,) = graph.graph.find_nodes(op="output")
        outputs = output.args[0]
        if not isinstance(outputs, (list, tuple)):
            raise _NotMine
        if len(placeholders) != len(args) or not all(
            isinstance(arg, torch.fx.Node) for arg in args
        ):
            raise _NotMine
        getitems: dict[int, Node] = {}
        for user in node.users:
            if user.target is not operator.getitem or type(user.args[1]) is not int:
                raise _NotMine
            getitems[user.args[1]] = user
        if sorted(getitems) != list(range(len(outputs))):
            raise _NotMine
        carried: dict[Node, tuple[Node, Node, Node]] = {}
        for position, out in enumerate(outputs):
            getitem = getitems[position]
            if getitem.args != (node, position):
                raise _NotMine
            phis = list(getitem.users)
            if len(phis) != 1 or not _is_call(phis[0], _tracing_ops._phi):
                raise _NotMine
            phi = phis[0]
            init = phi.args[0]
            if phi.args != (init, getitem) or not isinstance(init, torch.fx.Node):
                raise _NotMine
            matches = [i for i, arg in enumerate(args) if arg is init]
            if len(matches) != 1 or not isinstance(out, torch.fx.Node):
                raise _NotMine
            placeholder = placeholders[matches[0]]
            if placeholder in carried:
                raise _NotMine
            carried[placeholder] = (init, out, phi)
        build = _LoopBuild(
            index,
            parent,
            depth,
            node,
            graph,
            placeholders,
            cast("list[Node]", args),
        )
        self.loop_builds.append(build)
        self.loops.append(None)
        self.loop_calls[node] = index
        carries: list[Carry] = []
        small: list[VecCarry] = []
        for placeholder, arg in zip(placeholders, args, strict=True):
            if placeholder not in carried:
                # A read-only input: the body reads the enclosing region's
                # value.
                self.aliases[placeholder] = cast("Node", arg)
                continue
            init, out, phi = carried[placeholder]
            val = placeholder.meta.get("val")
            if not isinstance(val, torch.Tensor) or val.dtype != torch.float32:
                raise _NotMine
            dims = self.tile_dims(placeholder)
            if len(dims) == 2:
                if self.tile_dims(init) != dims:
                    raise _NotMine
                carry_init, init_frag, slot = self._carry_init(
                    init, node, placeholder, phi
                )
                init_value = (
                    init
                    if init_frag is not None and self.resolve(init) is not init_frag
                    else None
                )
                init_aux = (
                    tuple(self.frag_value(init_value)[2])
                    if init_value is not None
                    else ()
                )
                if slot < 0:
                    slot = len(self.slots)
                    self.slots.append((dims[0], dims[1]))
                self.frag_slots[placeholder] = slot
                self.frag_slots[phi] = slot
                self.carry_placeholders.add(placeholder)
                build.carry_placeholders.add(placeholder)
                self.region_of[placeholder] = index
                self.region_of[phi] = parent
                carries.append(
                    Carry(
                        slot=slot,
                        init=carry_init,
                        placeholder=placeholder,
                        output=out,
                        phi=phi,
                        rows=dims[0],
                        cols=dims[1],
                        init_frag=init_frag,
                        init_value=init_value,
                        init_aux=init_aux,
                    )
                )
            elif len(dims) <= 1:
                if self.resolve(out) is placeholder:
                    # The identity pass-through (the gradient accumulator of
                    # a decay mode the kernel does not run): nothing may read
                    # it afterwards.
                    if phi.users:
                        raise _NotMine
                    small.append(VecCarry(-1, placeholder, init, out, phi, True))
                    continue
                init_vec, vslot = self._vec_carry_init(
                    init, node, placeholder, phi, dims[0] if dims else None
                )
                self.vec_leaves[placeholder] = vslot
                self.vec_leaves[phi] = vslot
                self.region_of[placeholder] = index
                self.region_of[phi] = parent
                small.append(
                    VecCarry(vslot, placeholder, init, out, phi, False, init_vec)
                )
                build.vec_carries.append(small[-1])
            else:
                raise _NotMine
        # ``_new_var`` copies of the placeholders are the same fragments.
        for body_node in graph.graph.nodes:
            if _is_call(body_node, _tracing_ops._new_var) and self.is_frag_leaf(
                body_node.args[0]
            ):
                self.frag_slots[body_node] = self.frag_slots[
                    self.resolve(cast("Node", body_node.args[0]))
                ]
        steps = self._region(graph, index)
        produced = {step.node for step in steps if isinstance(step, (Dot, Derived))}
        for carry in carries:
            # The loop output is a dot or derived value of this loop (copied
            # into the carry's slot at the iteration end when computed in
            # another one), or the carry itself, unchanged.
            out = self.resolve(carry.output)
            if out is not carry.placeholder and (
                out not in produced
                or self.slots[self.frag_slots[out]] != (carry.rows, carry.cols)
            ):
                raise _NotMine
        vec_produced = {
            step.node for step in steps if isinstance(step, (Reduce, VecOp))
        }
        for vec_carry in small:
            if vec_carry.identity:
                continue
            out = self.resolve(vec_carry.output)
            if out not in vec_produced or self.vec_leaves.get(out) != vec_carry.slot:
                raise _NotMine
        loop = Loop(
            index=index,
            parent=parent,
            depth=depth,
            node=node,
            graph_id=graph.graph_id,
            axes=tuple(axes),
            symbol=self.loop_symbols.get(index),
            begin=begin,
            end=end,
            carries=tuple(carries),
            vec_carries=tuple(small),
            steps=steps,
            innermost=not any(isinstance(step, Loop) for step in steps),
        )
        self.loops[index] = loop
        return loop

    # -- vectors -----------------------------------------------------------

    def new_vec_slot(self, axis: Axis | None, orient: str) -> int:
        self.vec_slot_list.append(VecSlot(axis, orient))
        self.vec_parent.append(len(self.vec_parent))
        return len(self.vec_slot_list) - 1

    def vec_root(self, slot: int) -> int:
        while self.vec_parent[slot] != slot:
            self.vec_parent[slot] = self.vec_parent[self.vec_parent[slot]]
            slot = self.vec_parent[slot]
        return slot

    def link_vecs(self, a: int, b: int) -> None:
        """Slots ``a`` and ``b`` of one orientation share one warp split (a
        value flows from one into the other without a relayout)."""
        ra, rb = self.vec_root(a), self.vec_root(b)
        if ra != rb and self.vec_slot_list[ra].orient == self.vec_slot_list[rb].orient:
            self.vec_parent[rb] = ra

    def _vec_carry_init(
        self,
        init: Node,
        loop_node: Node,
        placeholder: Node,
        phi: Node,
        axis: Axis | None,
    ) -> tuple[Node | None, int]:
        """A vector carry's init: zeros (a fresh slot) or a vector of the
        enclosing region (its slot when nothing else reads it after the
        loop starts, else a fresh slot it is copied into)."""
        from ...language import creation_ops

        if _is_call(init, creation_ops.full):
            if len(init.args) not in (3, 4) or (
                len(init.args) == 4 and init.args[3] is not None
            ):
                raise _NotMine
            fill = init.args[1]
            if type(fill) not in (int, float) or fill != 0:
                raise _NotMine
            if init.args[2] is not torch.float32:
                raise _NotMine
            return None, self.new_vec_slot(
                axis, "rows" if axis is not None else "scalar"
            )
        source = self.resolve(init)
        if source not in self.vec_leaves:
            raise _NotMine
        source_slot = self.vec_leaves[source]
        if self.vec_slot_list[source_slot].axis is not axis:
            raise _NotMine
        # The carry takes over the vector's registers when nothing else reads
        # them once the loop starts (its own placeholder and phi read nothing).
        if self.dies_at(source, loop_node, skip=(placeholder, phi)):
            return source, source_slot
        slot = self.new_vec_slot(axis, self.vec_slot_list[source_slot].orient)
        self.link_vecs(source_slot, slot)
        return source, slot

    def _reduce(self, node: Node, loop: int) -> Reduce:
        """``sum(value, [-1])`` of a pointwise value over one fragment leaf
        (aux loads beside it; a value over several sources is a derived
        fragment first) into a vector along the fragment's rows."""
        if len(node.args) < 2 or node.kwargs or len(node.args) > 3:
            raise _NotMine
        dims = node.args[1]
        if not isinstance(dims, (list, tuple)) or len(dims) != 1:
            raise _NotMine
        if len(node.args) == 3 and node.args[2]:
            raise _NotMine
        value = node.args[0]
        if not isinstance(value, torch.fx.Node):
            raise _NotMine
        val = value.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype != torch.float32:
            raise _NotMine
        ndim = val.ndim
        dim = int(cast("int", dims[0])) % ndim
        if dim == ndim - 1:
            orient = "rows"
        elif dim == ndim - 2:
            orient = "cols"
        else:
            raise _NotMine
        source = self.resolve(value)
        if source in self.vec_leaves:
            # A vector summed over its axis: a scalar.
            vec_slot = self.vec_leaves[source]
            vec_axis = self.vec_slot_list[vec_slot].axis
            if (
                orient != "rows"
                or vec_axis is None
                or self.tile_dims(value) != (vec_axis,)
                or self.tile_dims(node)
            ):
                raise _NotMine
            slot = self.new_vec_slot(None, "scalar")
            self.vec_leaves[node] = slot
            self.region_of[node] = loop
            return Reduce(
                node=node,
                value=value,
                frag=None,
                orient="scalar",
                slot=slot,
                aux_loads=(),
                loop=loop,
                vec=source,
            )
        leaf, swap, aux = self.frag_value(value)
        if swap or len(self.tile_dims(value)) != 2:
            raise _NotMine
        if self.region_of.get(leaf, -1) not in (loop, *self.enclosing_regions(loop)):
            raise _NotMine
        rows, cols = self.slot_dims(leaf)
        axis = rows if orient == "rows" else cols
        if self.tile_dims(node) != (axis,):
            raise _NotMine
        slot = self.new_vec_slot(axis, orient)
        self.vec_frags.setdefault(slot, []).append(self.frag_slots[leaf])
        self.vec_leaves[node] = slot
        self.region_of[node] = loop
        return Reduce(
            node=node,
            value=value,
            frag=leaf,
            orient=orient,
            slot=slot,
            aux_loads=tuple(aux),
            loop=loop,
        )

    def enclosing_regions(self, loop: int) -> list[int]:
        """The loops enclosing region ``loop`` and the root."""
        result: list[int] = []
        while loop >= 0:
            loop = self.loop_builds[loop].parent
            result.append(loop)
        return result

    def innermost_loop(self, loops: Iterable[int]) -> int:
        """The deepest of ``loops`` (-1 when there is none)."""
        return max(loops, key=lambda index: self.loop_builds[index].depth, default=-1)

    def aux_within(self, aux: Iterable[Access], frame: int) -> bool:
        """Whether every aux load varies only with region ``frame`` or a loop
        enclosing it, so a value evaluated once per execution of ``frame``
        sees the right elements."""
        allowed = {frame, *self.enclosing_regions(frame)}
        return all(access.loops <= allowed for access in aux)

    def _vec_op(self, node: Node, loop: int) -> VecOp:
        """A pointwise vector value: its vector inputs share one slot family;
        it takes over the slot of a dying input (a carry first), else a
        fresh one."""
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype != torch.float32:
            raise _NotMine
        dims = self.tile_dims(node)
        axis = dims[0] if dims else None
        vecs = self.vec_sources(node)
        for leaf in vecs:
            # Vector inputs vary along the value's axis; scalars broadcast.
            leaf_axis = self.vec_slot_list[self.vec_leaves[leaf]].axis
            if leaf_axis is not None and leaf_axis is not axis:
                raise _NotMine
        aux = self.aux_of_vec(node)
        slot = -1
        carries = self.carry_vec_placeholders
        ordered = sorted(vecs, key=lambda leaf: 0 if leaf in carries else 1)
        for leaf in ordered:
            # A dying input of the value's own kind, defined in this region
            # (an enclosing region's vector is read again by every iteration),
            # hands over its slot.
            if self.vec_slot_list[self.vec_leaves[leaf]].axis is not axis:
                continue
            if self.region_of.get(leaf) == loop and self.dies_at(leaf, node):
                slot = self.vec_leaves[leaf]
                break
        if slot < 0:
            if axis is None:
                slot = self.new_vec_slot(None, "scalar")
            else:
                orient = next(
                    self.vec_slot_list[self.vec_leaves[leaf]].orient
                    for leaf in vecs
                    if self.vec_slot_list[self.vec_leaves[leaf]].axis is axis
                )
                slot = self.new_vec_slot(axis, orient)
        for leaf in vecs:
            self.link_vecs(slot, self.vec_leaves[leaf])
        self.vec_leaves[node] = slot
        self.region_of[node] = loop
        return VecOp(
            node=node,
            slot=slot,
            vecs=tuple(vecs),
            aux_loads=tuple(aux),
            loop=loop,
        )

    @property
    def carry_vec_placeholders(self) -> set[Node]:
        return {
            carry.placeholder
            for build in self.loop_builds
            for carry in build.vec_carries
        }

    def aux_of_vec(self, node: Node) -> list[Access]:
        """The scalar / vector loads under a vector value beside its vector
        leaves."""
        from ...language import memory_ops
        from ...language import view_ops

        aux: list[Access] = []
        seen: set[Node] = set()

        def visit(current: Node) -> None:
            current = self.resolve(current)
            if current in seen or current in self.vec_leaves:
                return
            seen.add(current)
            if _is_call(current, memory_ops.load):
                return
            if not (
                _is_call(current, view_ops.subscript)
                or _is_call(current, torch.ops.aten.expand.default)
                or _is_pointwise(current)
            ):
                raise _NotMine
            for inp in _node_inputs(current):
                if self.vec_sources(inp):
                    visit(inp)
                else:
                    self.aux_dag(inp, aux, seen)

        visit(node)
        return aux

    def _closure(self, anchors: Sequence[Node]) -> set[Node]:
        seen: set[Node] = set()
        stack = list(anchors)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(node.all_input_nodes)
        return seen

    def _carry_init(
        self, init: Node, loop_node: Node, placeholder: Node, phi: Node
    ) -> tuple[Access | None, Node | None, int]:
        """A carry's init: ``(load access, fragment, slot)`` -- zeros (no
        access, no fragment), a 2-D load (seeded per lane), or a fragment
        of the enclosing region (an earlier loop's accumulator carried on:
        its own slot when nothing else reads it after this loop starts,
        else a copy into a fresh one).  ``slot`` < 0 asks for a fresh slot."""
        from ...language import creation_ops
        from ...language import memory_ops

        if _is_call(init, creation_ops.full):
            if len(init.args) not in (3, 4) or (
                len(init.args) == 4 and init.args[3] is not None
            ):
                raise _NotMine
            fill = init.args[1]
            if type(fill) not in (int, float) or fill != 0:
                raise _NotMine
            if init.args[2] is not torch.float32:
                raise _NotMine
            return None, None, -1
        source = self.resolve(init)
        if not self.is_frag_leaf(source) and any(
            self.is_frag_leaf(leaf) for leaf in self.sources(init)
        ):
            # A pointwise value over one fragment of the enclosing region
            # (``T = eye + A``): evaluated per register into a fresh slot.
            leaf, swap, _aux = self.frag_value(init)
            if swap or self.slot_dims(leaf) != self.tile_dims(init):
                raise _NotMine
            return None, leaf, -1
        if self.is_frag_leaf(source):
            if source in self.transposed:
                raise _NotMine
            # Every other reader of the fragment must precede the loop for
            # the carry to take over its registers (its own placeholder and
            # phi read nothing).
            exclusive = self.dies_at(source, loop_node, skip=(placeholder, phi))
            slot = self.frag_slots[source] if exclusive else -1
            return None, source, slot
        if _is_call(source, torch.ops.prims.convert_element_type.default):
            if source.args[1] is not torch.float32:
                raise _NotMine
            source = cast("Node", source.args[0])
        if not _is_call(source, memory_ops.load):
            raise _NotMine
        access = self.access(source)
        if len(access.tile_dims) != 2 or access.dtype not in _STORE_DTYPES:
            raise _NotMine
        return access, None, -1

    def encloses(self, loop: int, axis: Axis) -> bool:
        """Whether a loop axis belongs to region ``loop`` or an enclosing loop
        (any non-loop axis qualifies)."""
        if axis.kind != AXIS_LOOP:
            return True
        while loop >= 0:
            if loop == axis.loop:
                return True
            loop = self.loop_builds[loop].parent
        return False

    def _dot(self, node: Node, loop: int) -> Dot:
        region = "loop" if loop >= 0 else "epilogue"
        if rounded_dot_dtype(node) is not None:
            # The evaluator rounds the fragment where a pointwise reader
            # evaluates it; no reader takes its registers as they are.
            if not all(_is_pointwise(user) for user in self.users_of(node)):
                raise _NotMine
        elif len(node.args) not in (3, 4) or (
            len(node.args) == 4 and node.args[3] not in (None, torch.float32)
        ):
            raise _NotMine
        result_dims = self.tile_dims(node)
        if len(result_dims) != 2:
            raise _NotMine
        rows, cols = result_dims
        a_node, b_node = node.args[0], node.args[1]
        if not isinstance(a_node, torch.fx.Node) or not isinstance(
            b_node, torch.fx.Node
        ):
            raise _NotMine
        a_dims = self.tile_dims(a_node)
        if len(a_dims) != 2 or a_dims[0] is not rows:
            raise _NotMine
        k = a_dims[1]
        if k.kind == AXIS_LEAD:
            raise _NotMine
        # A loop axis sizes the dot only inside that loop (or a nested one).
        if not all(self.encloses(loop, axis) for axis in (rows, cols, k)):
            raise _NotMine
        acc = _dot_acc(node)
        # The orientation rule: a no-acc dot read only transposed is emitted
        # transposed (the FX B source feeds A and vice versa), so the reads
        # are straight and an ``hl.arange`` mask relabels by coordinates.
        transposed = acc is None and self.consumed_only_transposed(node)
        if transposed:
            a = self.operand("a", b_node, (cols, k), flip=True)
            b = self.operand("b", a_node, (rows, k), flip=True)
            rows, cols = cols, rows
            self.transposed.add(node)
        else:
            a = self.operand("a", a_node, (rows, k))
            b = self.operand("b", b_node, (cols, k))
        acc_frag: Node | None = None
        acc_slot: int | None = None
        slot: int
        if acc is None:
            slot = len(self.slots)
            self.slots.append((rows, cols))
        else:
            if not isinstance(acc, torch.fx.Node):
                raise _NotMine
            if self.tile_dims(acc) != result_dims:
                raise _NotMine
            acc_frag, swap, aux = self.frag_value(acc)
            if swap:
                raise _NotMine
            acc_slot = self.frag_slots[acc_frag]
            if self.slots[acc_slot] != (rows, cols):
                raise _NotMine
            self.acc_loads[node] = tuple(aux)
            # A pointwise acc (``T * 0.5``) is materialized into its slot
            # before the MMAs (hoisted to the loop top when nothing earlier
            # reads it); an operand of the same dot reading that leaf
            # (``dot(Apow, T, acc=T * 0.5)``) would then see the rescaled
            # registers through its exchange, repacked words or shuffles.
            rescaled = self.resolve(acc) is not acc_frag
            reads_leaf = any(
                operand.kind == "frag" and self.resolve(operand.frag) is acc_frag
                for operand in (a, b)
                if operand.frag is not None
            )
            if self.accumulates_in_place(node, acc_frag, loop) and not (
                rescaled and reads_leaf
            ):
                slot = acc_slot
            else:
                slot = len(self.slots)
                self.slots.append((rows, cols))
        self.frag_slots[node] = slot
        self.region_of[node] = loop
        return Dot(
            node=node,
            a=a,
            b=b,
            acc=acc if isinstance(acc, torch.fx.Node) else None,
            acc_frag=acc_frag,
            acc_slot=acc_slot,
            rows=rows,
            cols=cols,
            k=k,
            region=region,
            slot=slot,
            transposed=transposed,
            loop=loop,
        )

    def _derived(self, node: Node, loop: int) -> Derived:
        """A pointwise value over several fragments / 2-D tiles becomes a
        fragment of its own, computed register by register (tiles loaded at
        the fragment coordinates); over tiles alone it is a fresh
        accumulator seed (``dk = dk_as + d_ba * beta``)."""
        from ...language import memory_ops

        dims = self.tile_dims(node)
        if not all(self.encloses(loop, axis) for axis in dims):
            raise _NotMine
        val = node.meta.get("val")
        if (
            len(dims) != 2
            or not isinstance(val, torch.Tensor)
            or val.dtype != torch.float32
        ):
            raise _NotMine
        frags: list[Node] = []
        loads: list[Access] = []
        for source in self.sources(node):
            # Every source must sit at the value's own coordinates: a
            # fragment in the same slot shape, read straight; a load whose
            # tile dims are the value's.
            _leaf, swap, _aux = self.trace_source_to(node, source)
            if self.is_frag_leaf(source):
                if self.register_swap(source, swap) or self.slot_dims(source) != dims:
                    raise _NotMine
                frags.append(source)
            else:
                assert _is_call(source, memory_ops.load)
                access = self.access(source)
                if swap or access.tile_dims != dims or access.dtype not in _AUX_DTYPES:
                    raise _NotMine
                loads.append(access)
        if not frags and not loads:
            raise _NotMine
        aux = self.aux_of(node)
        # Reuse the slot of a fragment input defined in this region that dies
        # here (its only uses are inside this value's DAG), else a fresh one;
        # an enclosing region's fragment is read again by every iteration.
        slot = -1
        for leaf in frags:
            if self.region_of.get(leaf) == loop and self.dies_at(leaf, node):
                slot = self.frag_slots[leaf]
                break
        if slot < 0:
            slot = len(self.slots)
            self.slots.append((dims[0], dims[1]))
        self.frag_slots[node] = slot
        self.region_of[node] = loop
        return Derived(
            node=node,
            slot=slot,
            frags=tuple(frags),
            loads=tuple(loads),
            aux_loads=tuple(aux),
            rows=dims[0],
            cols=dims[1],
            loop=loop,
        )

    def trace_source_to(
        self, node: Node, source: Node
    ) -> tuple[Node, bool, list[Access]]:
        """The permute parity between ``node`` and one of its sources (the
        path is unique: pointwise DAGs are trees over their sources)."""
        from ...language import _tracing_ops
        from ...language import view_ops

        def visit(current: Node, parity: bool) -> bool | None:
            if current is source:
                return parity
            if _is_call(current, _tracing_ops._new_var) or _is_call(
                current, _tracing_ops._mask_to
            ):
                return visit(cast("Node", current.args[0]), parity)
            if current in self.aliases:
                return visit(self.aliases[current], parity)
            if current.op != "call_function" or self.is_frag_leaf(current):
                return None
            if _is_swap_permute(current):
                parity = not parity
            elif not (_is_call(current, view_ops.subscript) or _is_pointwise(current)):
                return None
            for inp in _node_inputs(current):
                result = visit(inp, parity)
                if result is not None:
                    return result
            return None

        parity = visit(node, False)
        if parity is None:
            raise _NotMine
        return source, parity, []

    def aux_of(self, node: Node) -> list[Access]:
        """The scalar / vector loads under ``node`` beside its sources (the
        inputs with no source go through ``aux_dag``)."""
        from ...language import memory_ops
        from ...language import view_ops

        aux: list[Access] = []
        seen: set[Node] = set()

        def visit(current: Node) -> None:
            current = self.resolve(current)
            if current in seen or self.is_frag_leaf(current):
                return
            seen.add(current)
            if _is_call(current, memory_ops.load):
                return
            if not (
                _is_call(current, view_ops.subscript)
                or _is_swap_permute(current)
                or _is_pointwise(current)
            ):
                raise _NotMine
            for inp in _node_inputs(current):
                if self.sources(inp):
                    visit(inp)
                else:
                    self.aux_dag(inp, aux, seen)

        visit(node)
        return aux

    def _store(self, node: Node, loop: int) -> Store | VecStore:
        from .matmul_utils import strip_new_vars

        region = "loop" if loop >= 0 else "epilogue"
        access = self.access(node)
        value = node.args[2]
        if not isinstance(value, torch.fx.Node):
            raise _NotMine
        value_fake = value.meta.get("val")
        if not isinstance(value_fake, torch.Tensor) or value_fake.dtype != access.dtype:
            raise _NotMine
        if len(access.tile_dims) <= 1:
            # A vector / scalar store: a vector value along the stored axis
            # (scalars broadcast), or a scalar value.
            if (
                access.dtype != torch.float32
                or self.tile_dims(value) != access.tile_dims
            ):
                raise _NotMine
            axis = access.tile_dims[0] if access.tile_dims else None
            vecs = self.vec_sources(value)
            if not vecs or any(
                self.vec_slot_list[self.vec_leaves[leaf]].axis not in (axis, None)
                for leaf in vecs
            ):
                raise _NotMine
            self.aux_of_vec(value)
            # The stored registers follow a vector input of the stored kind
            # (scalars alone broadcast into a vector store have no lane map).
            vec = next(
                (
                    leaf
                    for leaf in vecs
                    if self.vec_slot_list[self.vec_leaves[leaf]].axis is axis
                ),
                None,
            )
            if vec is None:
                raise _NotMine
            return VecStore(access=access, value=value, vec=vec, loop=loop)
        if len(access.tile_dims) != 2 or access.dtype not in _STORE_DTYPES:
            raise _NotMine
        leaf, swap, _aux = self.frag_value(value)
        if self.tile_dims(value) != access.tile_dims:
            raise _NotMine
        rows, cols = self.slots[self.frag_slots[leaf]]
        expected = (cols, rows) if swap else (rows, cols)
        if access.tile_dims != expected:
            raise _NotMine
        if region == "loop":
            # A loop store snapshots this loop's carry at loop entry (hoisted
            # to the iteration top; the value itself or its cast) or stores a
            # value computed in this loop at its program point.
            if leaf in self.loop_builds[loop].carry_placeholders:
                source = "carry_in"
                stored = strip_new_vars(value)
                if stored is not leaf and not (
                    _is_call(stored, torch.ops.prims.convert_element_type.default)
                    and strip_new_vars(cast("Node", stored.args[0])) is leaf
                ):
                    raise _NotMine
            elif self.region_of.get(leaf) == loop:
                source = "body"
            else:
                raise _NotMine
        else:
            source = "epilogue"
        return Store(
            access=access,
            value=value,
            frag=leaf,
            swap=swap,
            region=region,
            source=source,
            loop=loop,
        )


def match_mma_dag(graphs: Sequence[GraphInfo]) -> MmaDagProgram | None:
    """Match the MMA dataflow program in ``graphs`` (config-free).

    Needs the compile environment and host function to be current (the
    autotuner heuristic enters them; pre-codegen already runs inside them).
    """
    return _Matcher(graphs).match()


def _whole_k_words(program: MmaDagProgram, dot: Dot, operand: Operand) -> bool:
    """Whether a register-routed ``operand`` binds its words for the whole K
    extent before the MMAs: 16-bit fragments always (repacked once); fp32
    ones when the producer is the dot's own accumulator (the shuffles must
    read it before the MMAs rewrite it), else one K step at a time."""
    assert operand.frag is not None
    return operand.dtype.itemsize == 2 or program.frag_slots[operand.frag] == dot.slot


def mma_dag_facts(program: MmaDagProgram) -> CuteMmaDagFacts:
    from .cute_warp_mma_gemm import _device_sm_count
    from .tcgen05_config import CuteTcgen05Config

    env = CompileEnvironment.current()
    grid_blocks = 1
    for axis in program.lead_axes:
        grid_blocks *= axis.extent
    tiles: list[TileFacts] = []
    seen: set[tuple[object, ...]] = set()
    # Exchange-tile keys numbered as ``mma_dag_frag_edges`` numbers them.
    edge_keys: dict[tuple[object, ...], int] = {}
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.chain_key in seen:
                continue
            seen.add(operand.chain_key)
            source_bytes = 0
            if operand.load is not None and operand.load.dtype != operand.dtype:
                source_bytes = operand.load.dtype.itemsize
            frame = program.operand_frame(dot, operand)
            tiles.append(
                TileFacts(
                    operand.rows.index,
                    operand.cols.index,
                    operand.dtype.itemsize,
                    program.is_ring(dot, operand),
                    source_bytes,
                    2
                    if program.early_exchange(dot, operand)
                    and program.loop_has_ring(dot.loop)
                    else 1,
                    edge_keys.setdefault(operand.chain_key, len(edge_keys))
                    if operand.kind == "frag"
                    else None,
                    frame,
                    source_bytes > 0
                    and frame == -1
                    and program.staged_ahead_of_loops(operand.chain_key),
                    operand.kind == "frag"
                    and frame == -1
                    and program.read_within_one_root_segment(operand.chain_key),
                )
            )
    # Derived values' tiles ride the ring too (read per lane from the slot);
    # a tile of another level is loaded per lane from global memory.
    derived_loads: dict[Node, tuple[Derived, Access]] = {}
    for derived in program.derived:
        for access in derived.loads:
            if program.ring_derived_load(derived, access):
                derived_loads.setdefault(access.node, (derived, access))
    for derived, access in derived_loads.values():
        rows, cols = access.tile_dims
        tiles.append(
            TileFacts(
                rows.index,
                cols.index,
                access.dtype.itemsize,
                True,
                frame=derived.loop,
            )
        )
    snapshots = [
        store
        for store in program.stores
        if store.region == "loop" and store.source == "carry_in"
    ]
    staging = tuple(
        (
            program.slots[program.frag_slots[store.frag]][0].index,
            program.slots[program.frag_slots[store.frag]][1].index,
            store.access.dtype.itemsize,
        )
        for store in snapshots
    )
    from .cute_mma_dag_tcgen05 import tcgen05_facts

    return CuteMmaDagFacts(
        axes=tuple(
            AxisFacts(a.block_id, a.kind, a.extent, a.loop) for a in program.axes
        ),
        frags=tuple((rows.index, cols.index) for rows, cols in program.slots),
        tiles=tuple(tiles),
        staging=staging,
        k_axes=tuple(sorted({dot.k.index for dot in program.dots})),
        smem_capacity=CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device),
        grid_blocks=grid_blocks,
        tcgen05=tcgen05_facts(program),
        live=mma_dag_live_slots(program),
        edges=mma_dag_frag_edges(program),
        sm_count=_device_sm_count(env.device),
        loop_passes=sum(
            1
            for dot in program.dots
            for operand in (dot.a, dot.b)
            if operand.load is not None
            and program.is_ring(dot, operand)
            and (operand.pointwise or operand.load.dtype != operand.dtype)
        ),
        dots=tuple(
            DotFacts(
                dot.slot,
                dot.k.index,
                dot.a.kind == "frag"
                and not dot.a.logical_swap
                and _whole_k_words(program, dot, dot.a),
                dot.b.kind == "frag" and _whole_k_words(program, dot, dot.b),
                dot.a.dtype.itemsize,
            )
            for dot in program.dots
        ),
        identity_candidates=any(
            ((dot.a.kind == "frag" and not dot.a.logical_swap) or dot.b.kind == "frag")
            and (
                dot.a.dtype.itemsize == 2
                or any(
                    operand.kind == "frag" and program.operand_frame(dot, operand) >= 0
                    for operand in (dot.a, dot.b)
                )
            )
            for dot in program.dots
        ),
        loops=tuple(LoopFacts(loop.parent, loop.innermost) for loop in program.loops),
        staging_frames=tuple(store.loop for store in snapshots),
        vecs=tuple(
            VecFacts(
                slot.axis.index if slot.axis is not None else -1,
                slot.orient,
                program.vec_frag_slots[index],
            )
            for index, slot in enumerate(program.vec_slots)
        ),
    )


def mma_dag_live_slots(program: MmaDagProgram) -> tuple[tuple[int, ...], ...]:
    """The slot sets live at each step (loop bodies and root region alike,
    in program order).

    A loop's carries are live over its whole body (and on to their phis'
    last readers); a dot / derived result is live from its step to its last
    reader (operand, accumulator, derived input or store).
    """
    flat: list[Step] = []
    spans: dict[int, tuple[int, int]] = {}

    def walk(steps: Sequence[Step]) -> None:
        for step in steps:
            if isinstance(step, Loop):
                start = len(flat)
                walk(step.steps)
                spans[step.index] = (start, max(start, len(flat) - 1))
            else:
                flat.append(step)

    walk(program.body)
    defined: dict[int, int] = {}
    last_use: dict[int, int] = {}

    def use(slot: int, index: int) -> None:
        last_use[slot] = max(last_use.get(slot, index), index)

    def define(slot: int, index: int) -> None:
        defined[slot] = min(defined.get(slot, index), index)
        use(slot, index)

    for index, step in enumerate(flat):
        if isinstance(step, Dot):
            for operand in (step.a, step.b):
                if operand.frag is not None:
                    use(program.frag_slots[program.resolve(operand.frag)], index)
            if step.acc_frag is not None:
                use(program.frag_slots[step.acc_frag], index)
            define(step.slot, index)
        elif isinstance(step, Derived):
            for leaf in step.frags:
                use(program.frag_slots[leaf], index)
            define(step.slot, index)
        elif isinstance(step, (Store, Reduce)) and step.frag is not None:
            use(program.frag_slots[step.frag], index)
    for loop in program.loops:
        start, end = spans[loop.index]
        for carry in loop.carries:
            define(carry.slot, start)
            use(carry.slot, end)
    groups: list[tuple[int, ...]] = []
    for index in range(len(flat)):
        live = {
            slot for slot, start in defined.items() if start <= index <= last_use[slot]
        }
        groups.append(tuple(sorted(live)))
    return tuple(groups)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def _decline(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"cute_mma_dag: {reason}")


def async_route_reason(env: CompileEnvironment, operand: Operand) -> str | None:
    """Why the operand's source tile cannot be staged with 16-byte
    ``cp.async`` packets straight from global memory (None: it can).  A
    pointwise operand's source is rewritten packet by packet: in place when
    it has the operand's dtype, from a raw fp32 tile into the 16-bit tile
    otherwise."""
    load = operand.load
    assert load is not None
    if load.dtype != operand.dtype and torch.float32 not in (load.dtype, operand.dtype):
        return "the source tile dtype differs from the operand dtype"
    return async_tile_reason(env, load, operand.cols_pos)


def async_tile_reason(
    env: CompileEnvironment, load: Access, cols_pos: int
) -> str | None:
    """Why a 2-D tile whose tile dim ``cols_pos`` is the contiguous one
    cannot be staged with 16-byte ``cp.async`` packets (None: it can)."""
    from .memory_ops import cute_tensor_base_is_aligned

    es = load.dtype.itemsize
    if load.tile_dim_strides[cols_pos] != 1:
        return "the tile columns are not contiguous"
    if not cute_tensor_base_is_aligned(env, load.fake, 16):
        return "the tensor base is not provably 16-byte aligned"
    for stride in load.strides:
        if stride != 1 and (stride * es) % 16:
            return "a non-contiguous stride is not a multiple of 16 bytes"
    return None


def staged_store_reason(env: CompileEnvironment, store: Store, bn: int) -> str | None:
    """Why a loop carry snapshot (``bn`` columns) cannot be written as
    16-byte packets from the shared staging tile (None: it can)."""
    from .memory_ops import cute_tensor_base_is_aligned

    if store.region != "loop" or store.source != "carry_in" or store.swap:
        return "only loop-entry snapshots are staged"
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


def frag_register_route(
    *,
    role: str,
    logical_swap: bool,
    producer: tuple[int, int],
    consumer: tuple[int, int],
    k: int,
    layout_p: tuple[int, int],
    layout_c: tuple[int, int],
    elem_bytes: int = 2,
    per_iteration: bool = True,
) -> str | None:
    """The register route of a fragment operand (``producer`` / ``consumer``
    are the fragment shapes, ``layout_*`` their warp grids), or None when it
    must go through shared memory.

    Every lane of the consumer warp needs its operand words for the whole K
    extent, so the producer warp must own whole K rows of the fragment and
    the two warp grids must split the shared M / N extent identically:

    * A from ``[M, K]`` (``identity``): producer and consumer ``wn == 1`` with
      the same ``wm``;
    * B from ``[K, N]`` (``movmatrix``): producer and consumer ``wm == 1``
      with the same ``wn`` (each 8x8 block transposed in registers);
    * B from ``[N, K]`` (``identity``): producer ``wn == 1``, consumer ``wm ==
      1``, ``wm_p == wn_c``.

    An fp32 (tf32) operand takes the ``shuffle`` route in the same three
    shapes, but only when its exchange would be rewritten every iteration
    (``per_iteration``): a tile written once per CTA and read with ldmatrix
    costs less than shuffling the words again in every iteration.
    """
    rows_c, cols_c = consumer
    wm_p, wn_p = layout_p
    wm_c, wn_c = layout_c
    route: str | None = None
    if role == "a":
        if logical_swap:
            return None
        if producer == (rows_c, k) and wn_p == wn_c == 1 and wm_p == wm_c:
            route = ROUTE_IDENTITY
    elif not logical_swap:
        if producer == (k, cols_c) and wm_p == wm_c == 1 and wn_p == wn_c:
            route = ROUTE_MOVMATRIX
    elif producer == (cols_c, k) and wn_p == 1 and wm_c == 1 and wm_p == wn_c:
        route = ROUTE_IDENTITY
    if route is None or elem_bytes == 2:
        return route
    return ROUTE_SHUFFLE if per_iteration else None


def mma_dag_frag_edges(program: MmaDagProgram) -> tuple[FragEdgeFacts, ...]:
    """The fragment operands of the dots as facts, in program order (the
    planner maps the solver's routes back to the operands by this order)."""
    keys: dict[tuple[object, ...], int] = {}
    edges = []
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.kind != "frag":
                continue
            assert operand.frag is not None
            edges.append(
                FragEdgeFacts(
                    role=operand.role,
                    logical_swap=operand.logical_swap,
                    producer=program.frag_slots[operand.frag],
                    consumer=dot.slot,
                    k=dot.k.index,
                    elem_bytes=operand.dtype.itemsize,
                    key=keys.setdefault(operand.chain_key, len(keys)),
                    # Written per iteration of some loop, or once per CTA.
                    region="loop"
                    if program.operand_frame(dot, operand) >= 0
                    else "epilogue",
                    in_loop=dot.loop >= 0,
                    hoisted=dot.loop >= 0 and program.operand_frame(dot, operand) == -1,
                )
            )
    return tuple(edges)


def mma_dag_layout_solution(
    facts: CuteMmaDagFacts, resolved: Sequence[int], warps: int
) -> tuple[Layouts, tuple[str, ...], int]:
    """The warp grid of every fragment shape, the route of every fragment
    operand (by ``facts.edges`` order) and the bytes exchanged through shared
    memory per loop iteration: among the M-first / N-first choices per
    shape, the combination that carries every vector reduction (its
    fragments split the vector's dim the same way) and exchanges the fewest
    bytes per iteration (ties: fewer exchanged tiles, then M-first)."""
    slot_shapes = [(resolved[rows], resolved[cols]) for rows, cols in facts.frags]
    shapes = sorted(set(slot_shapes))
    best: tuple[tuple[int, int, int, int], Layouts, tuple[str, ...], int] | None = None
    for n_first in itertools.product((False, True), repeat=len(shapes)):
        layouts: Layouts = {}
        for shape, flag in zip(shapes, n_first, strict=True):
            layout = mma_dag_warp_layout(*shape, warps, n_first=flag)
            if layout is None:
                break
            layouts[shape] = layout
        if len(layouts) != len(shapes):
            continue
        violations = mma_dag_vec_violations(facts, resolved, layouts, warps)
        routes: list[str] = []
        exchanged: set[int] = set()
        loop_bytes = 0
        for edge in facts.edges:
            producer = slot_shapes[edge.producer]
            consumer = slot_shapes[edge.consumer]
            route = frag_register_route(
                role=edge.role,
                logical_swap=edge.logical_swap,
                producer=producer,
                consumer=consumer,
                k=resolved[edge.k],
                layout_p=layouts[producer],
                layout_c=layouts[consumer],
                elem_bytes=edge.elem_bytes,
                per_iteration=edge.region == "loop",
            )
            if route is None:
                route = ROUTE_EXCHANGE
                if edge.key not in exchanged:
                    exchanged.add(edge.key)
                    if edge.region == "loop":
                        loop_bytes += producer[0] * producer[1] * edge.elem_bytes
            routes.append(route)
        cost = (violations, loop_bytes, len(exchanged), sum(n_first))
        if best is None or cost < best[0]:
            best = (cost, layouts, tuple(routes), loop_bytes)
    assert best is not None
    return best[1], best[2], best[3]


def solve_mma_dag_layouts(
    program: MmaDagProgram,
    facts: CuteMmaDagFacts,
    plan_sizes: tuple[int, ...],
    warps: int,
) -> tuple[Layouts, dict[tuple[Node, str], str]]:
    """``mma_dag_layout_solution`` keyed by the program's operands."""
    layouts, routes, _bytes = mma_dag_layout_solution(facts, plan_sizes, warps)
    keyed: dict[tuple[Node, str], str] = {}
    index = 0
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.kind == "frag":
                keyed[(dot.node, operand.role)] = routes[index]
                index += 1
    assert index == len(routes)
    return layouts, keyed


def plan_mma_dag(graphs: Sequence[GraphInfo], config: object) -> MmaDagPlan | None:
    """The plan for ``config`` (None when the knob is off); raises
    ``BackendUnsupported`` when the knob is on but the kernel or the config
    cannot run the lowering."""
    from ..device_function import DeviceFunction

    get = cast("dict[str, object]", config).get  # pyrefly: ignore[missing-attribute]
    mode = get(MMA_DAG_KEY, MMA_DAG_OFF)
    if mode == MMA_DAG_OFF or mode is None:
        return None
    if mode not in (MMA_DAG_WARP, MMA_DAG_TCGEN05):
        raise _decline(f"unknown mode {mode!r}")
    program = match_mma_dag(graphs)
    if program is None:
        raise _decline("the kernel is not an MMA dataflow program the owner lowers")
    df = DeviceFunction.current()
    env = CompileEnvironment.current()
    sizes_by_block: dict[int, object] = {}
    for axis in program.axes:
        if axis.kind in (AXIS_ROOT, AXIS_LOOP, AXIS_LEAD):
            sizes_by_block[axis.block_id] = df.resolved_block_size(axis.block_id)
    for axis in program.lead_axes:
        if sizes_by_block[axis.block_id] != 1:
            raise _decline("leading root tiles must have block size 1")
    warps = get(MMA_DAG_WARPS_KEY, MMA_DAG_DEFAULT_WARPS)
    stages = get(MMA_DAG_STAGES_KEY, MMA_DAG_DEFAULT_STAGES)
    store_mode = get(MMA_DAG_STORE_KEY, MMA_DAG_STORE_STAGED)
    exchange_mode = get(MMA_DAG_EXCHANGE_KEY, MMA_DAG_EXCHANGE_AUTO)
    pass_warps = get(MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PASS_WARPS_OFF)
    loads_mode = get(MMA_DAG_LOADS_KEY, MMA_DAG_LOADS_TMA)
    facts = mma_dag_facts(program)
    reason = mma_dag_config_reason(
        facts, sizes_by_block, warps, stages, cast("str", mode)
    )
    if reason is not None:
        raise _decline(reason)
    if store_mode not in MMA_DAG_STORE_CHOICES:
        raise _decline(
            f"{MMA_DAG_STORE_KEY} {store_mode!r} must be one of {MMA_DAG_STORE_CHOICES}"
        )
    if exchange_mode not in MMA_DAG_EXCHANGE_CHOICES:
        raise _decline(
            f"{MMA_DAG_EXCHANGE_KEY} {exchange_mode!r} must be one of {MMA_DAG_EXCHANGE_CHOICES}"
        )
    if pass_warps not in MMA_DAG_PASS_WARPS_CHOICES:
        raise _decline(
            f"{MMA_DAG_PASS_WARPS_KEY} {pass_warps!r} must be one of {MMA_DAG_PASS_WARPS_CHOICES}"
        )
    if loads_mode not in MMA_DAG_LOADS_CHOICES:
        raise _decline(
            f"{MMA_DAG_LOADS_KEY} {loads_mode!r} must be one of {MMA_DAG_LOADS_CHOICES}"
        )
    assert type(warps) is int and type(stages) is int
    if pass_warps == MMA_DAG_PASS_WARPS_ON:
        reason = mma_dag_pass_warps_reason(facts, warps, stages, cast("str", mode))
        if reason:
            raise _decline(f"{MMA_DAG_PASS_WARPS_KEY}: {reason}")
    if get("pid_type", "flat") != "flat":
        raise _decline("the lowering needs the flat program-id grid")
    sizes = tuple(
        cast("int", facts.block_size(axis.index, sizes_by_block))
        for axis in program.axes
    )
    layouts, routes = solve_mma_dag_layouts(program, facts, sizes, warps)
    if mma_dag_vec_violations(facts, sizes, layouts, warps):
        raise _decline("the warp layouts cannot carry a vector reduction")
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.kind == "frag":
                if exchange_mode != MMA_DAG_EXCHANGE_AUTO:
                    routes[(dot.node, operand.role)] = ROUTE_EXCHANGE
                continue
            if async_route_reason(env, operand) is not None:
                route = ROUTE_SYNC
            elif operand.load is not None and operand.load.dtype != operand.dtype:
                route = ROUTE_CONVERT
            elif operand.pointwise:
                route = ROUTE_INPLACE
            else:
                route = ROUTE_DIRECT
            routes[(dot.node, operand.role)] = route
    derived_routes: dict[Node, str] = {}
    for derived in program.derived:
        for access in derived.loads:
            # The derived value's tile rides the ring when it varies with
            # the value's own innermost loop and cp.async can stage it;
            # otherwise each lane loads its pairs from global memory at the
            # fragment coordinates.
            derived_routes[access.node] = (
                DERIVED_SHARED
                if program.ring_derived_load(derived, access)
                and async_tile_reason(env, access, 1) is None
                else DERIVED_GLOBAL
            )
    staged = tuple(
        store
        for store in program.stores
        if store_mode == MMA_DAG_STORE_STAGED
        and staged_store_reason(
            env, store, sizes[program.slots[program.frag_slots[store.frag]][1].index]
        )
        is None
    )
    if mode == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import plan_tcgen05

        tc = plan_tcgen05(
            program,
            facts,
            sizes,
            routes,
            warps,
            stages,
            cast("str", store_mode),
            pass_warps == MMA_DAG_PASS_WARPS_ON,
            cast("str", loads_mode),
        )
        return MmaDagPlan(
            program=program,
            sizes=sizes,
            warps=warps,
            stages=stages,
            routes=routes,
            staged_stores=tc.staged_stores,
            exchange_mode=cast("str", exchange_mode),
            compute=MMA_DAG_TCGEN05,
            tcgen05=tc,
        )
    plan = MmaDagPlan(
        program=program,
        sizes=sizes,
        warps=warps,
        stages=stages,
        routes=routes,
        staged_stores=staged,
        exchange_mode=cast("str", exchange_mode),
        layouts=layouts,
        derived_routes=derived_routes,
    )
    if facts.smem_capacity and plan.smem_bytes > facts.smem_capacity:
        raise _decline(
            f"{plan.smem_bytes} B of shared memory exceed the capacity {facts.smem_capacity}"
        )
    return plan


__all__ = [
    "AXIS_LEAD",
    "AXIS_LOOP",
    "AXIS_RDIM",
    "AXIS_ROOT",
    "DERIVED_GLOBAL",
    "DERIVED_SHARED",
    "MMA_DAG_DEFAULT_STAGES",
    "MMA_DAG_DEFAULT_WARPS",
    "MMA_DAG_EXCHANGE_CHOICES",
    "MMA_DAG_EXCHANGE_KEY",
    "MMA_DAG_KEY",
    "MMA_DAG_KEYS",
    "MMA_DAG_LOADS_CHOICES",
    "MMA_DAG_LOADS_KEY",
    "MMA_DAG_LOADS_LANE",
    "MMA_DAG_LOADS_TMA",
    "MMA_DAG_MODES",
    "MMA_DAG_OFF",
    "MMA_DAG_PASS_WARPS_CHOICES",
    "MMA_DAG_PASS_WARPS_KEY",
    "MMA_DAG_PASS_WARPS_OFF",
    "MMA_DAG_PASS_WARPS_ON",
    "MMA_DAG_PASS_WARP_STAGES",
    "MMA_DAG_SEED_STAGES",
    "MMA_DAG_SMEM_CTA_RESERVED",
    "MMA_DAG_STAGES_KEY",
    "MMA_DAG_STAGE_CHOICES",
    "MMA_DAG_STORE_CHOICES",
    "MMA_DAG_STORE_DIRECT",
    "MMA_DAG_STORE_KEY",
    "MMA_DAG_STORE_STAGED",
    "MMA_DAG_TCGEN05",
    "MMA_DAG_TMEM_COLUMNS",
    "MMA_DAG_WARP",
    "MMA_DAG_WARPS_KEY",
    "MMA_DAG_WARP_CHOICES",
    "REGISTER_ROUTES",
    "ROUTE_CONVERT",
    "ROUTE_DIRECT",
    "ROUTE_EXCHANGE",
    "ROUTE_IDENTITY",
    "ROUTE_INPLACE",
    "ROUTE_MOVMATRIX",
    "ROUTE_SHUFFLE",
    "ROUTE_SYNC",
    "Access",
    "Axis",
    "Carry",
    "CuteMmaDagFacts",
    "Derived",
    "Dot",
    "FragEdgeFacts",
    "MmaDagPlan",
    "MmaDagProgram",
    "Operand",
    "Store",
    "Subscript",
    "accumulator_coordinates",
    "async_route_reason",
    "async_tile_reason",
    "emulate_ldmatrix",
    "frag_register_route",
    "ldmatrix_lane_address",
    "match_mma_dag",
    "mma_dag_config_reason",
    "mma_dag_ctas_per_sm",
    "mma_dag_facts",
    "mma_dag_frag_edges",
    "mma_dag_frag_regs",
    "mma_dag_layout_solution",
    "mma_dag_live_slots",
    "mma_dag_repair",
    "mma_dag_seed_configs",
    "mma_dag_seed_warps",
    "mma_dag_smem_estimate",
    "mma_dag_transient_regs",
    "mma_dag_warp_layout",
    "plan_mma_dag",
    "solve_mma_dag_layouts",
    "staged_store_reason",
    "swizzled_chunk",
    "transposed_swizzled_chunk",
]
