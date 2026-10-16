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

from ... import exc
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from .cute_warp_mma_gemm import warp_mma_warp_layout

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_ir import GraphInfo

    Layouts = dict[tuple[int, int], tuple[int, int]]

MMA_DAG_KEY = "cute_mma_dag"
MMA_DAG_OFF = "off"
MMA_DAG_WARP = "warp"
# ``tcgen05`` is reserved for a TMEM-resident mode of the same owner.
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
MMA_DAG_KEYS: tuple[str, ...] = (
    MMA_DAG_KEY,
    MMA_DAG_WARPS_KEY,
    MMA_DAG_STAGES_KEY,
    MMA_DAG_STORE_KEY,
    MMA_DAG_EXCHANGE_KEY,
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
# Scalar / vector side loads (decays, masks, scales) the per-lane loads read.
_AUX_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)
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


# ---------------------------------------------------------------------------
# Typed IR
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Axis:
    """One tiled dimension of the kernel.

    ``root`` axes are the root grid's tile dims (block size from the config),
    ``loop`` the inner device loop's tile (the dot K extent per iteration),
    ``rdim`` a static ``:`` extent and ``lead`` a root block of size one (a
    batch coordinate, never a fragment dim).  ``extent`` is the full dim.
    """

    index: int
    block_id: int
    kind: str
    extent: int


@dataclasses.dataclass(frozen=True)
class Subscript:
    """One subscript position of a tensor access: a whole tile of ``axis``
    (``root`` / ``loop`` / ``lead`` tiles and ``:`` rdims) or a uniform
    integer expression ``expr`` over the loop index and the leading tile
    ids (``loop_dependent`` says the loop index occurs in it)."""

    kind: str  # "axis" | "scalar"
    axis: Axis | None = None
    expr: sympy.Expr | None = None
    loop_dependent: bool = False


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


@dataclasses.dataclass(frozen=True)
class Derived:
    """A fragment defined by a pointwise DAG over several fragment leaves
    (``frags``, all in the slot's layout) and 2-D tiles loaded at the
    fragment coordinates (``loads``): ``vnew = u - dot(w, h)``, ``dS = dS -
    dot(w^T, dv)``.  ``slot`` is the register slot it is materialized in (an
    input's slot when that input dies here)."""

    node: Node
    slot: int
    frags: tuple[Node, ...]
    loads: tuple[Access, ...]
    aux_loads: tuple[Access, ...]
    rows: Axis
    cols: Axis


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


@dataclasses.dataclass(frozen=True)
class Carry:
    """One loop-carried fragment: its root init (None: zeros), the body
    placeholder, the body output and the phi after the loop."""

    slot: int
    init: Access | None
    placeholder: Node
    output: Node
    phi: Node
    rows: Axis
    cols: Axis


@dataclasses.dataclass(frozen=True)
class MmaDagProgram:
    """The matched MMA dataflow program (config-free)."""

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
    # The loop body in program order: dots, derived values and body stores
    # (carry snapshots are hoisted to the iteration top by the emitter).
    steps: tuple[Dot | Derived | Store, ...] = ()
    derived: tuple[Derived, ...] = ()

    @property
    def dtype(self) -> torch.dtype:
        return self.dots[0].a.dtype

    @property
    def elem_bytes(self) -> int:
        return self.dtype.itemsize

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
        """Whether a fragment operand of the loop reads the carry at loop
        entry: its exchange tile is written before the iteration barrier
        (double-buffered by iteration parity) instead of behind its own."""
        return (
            dot.region == "loop"
            and operand.kind == "frag"
            and operand.frag in self.carry_placeholders
        )

    @property
    def accesses(self) -> tuple[Access, ...]:
        result: list[Access] = []
        for carry in self.carries:
            if carry.init is not None:
                result.append(carry.init)
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

    @property
    def bytes_per_element(self) -> int:
        return self.elem_bytes + self.source_bytes


@dataclasses.dataclass(frozen=True)
class FragEdgeFacts:
    """A fragment operand of a dot: the producer and consumer slots, whether
    the logical operand is the producer's registers transposed, the K axis
    index, the element size, the exchange-tile sharing key and the region
    (loop edges are exchanged every iteration)."""

    role: str
    logical_swap: bool
    producer: int
    consumer: int
    k: int
    elem_bytes: int
    key: int
    region: str


@dataclasses.dataclass(frozen=True)
class DotFacts:
    """One dot's transient registers: its result slot, its K axis index,
    whether its A operand is a fragment read straight (it may be repacked
    from registers for the whole K extent) and whether B is a fragment."""

    slot: int
    k: int
    a_identity: bool
    b_frag: bool


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

    @property
    def threads(self) -> int:
        return 32 * self.warps

    def ring_derived(self) -> tuple[tuple[Derived, Access], ...]:
        """Derived tile loads staged through the ring, in program order,
        one entry per load node."""
        seen: set[Node] = set()
        result = []
        for derived in self.program.derived:
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
        return 2 if self.program.early_exchange(dot, operand) else 1

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
        seen: set[tuple[object, ...]] = set()
        result = []
        for dot in self.program.dots_in(region):
            for operand in (dot.a, dot.b):
                if self.route(dot, operand) not in routes or operand.chain_key in seen:
                    continue
                seen.add(operand.chain_key)
                result.append((dot, operand))
        return tuple(result)

    def ring_operands(self) -> tuple[tuple[Dot, Operand], ...]:
        """Loop operands staged through the cp.async ring, in program order."""
        return self.unique_operands(
            "loop", (ROUTE_DIRECT, ROUTE_INPLACE, ROUTE_CONVERT)
        )

    def smem_layout(
        self,
    ) -> tuple[
        dict[tuple[object, ...], SmemTileSpec],
        dict[Node, SmemTileSpec],
        dict[Node, SmemTileSpec],
        int,
    ]:
        """The shared-memory layout the emitter renders: operand tiles by
        sharing key (ring, synchronous / epilogue and exchange tiles, in
        program order), the derived values' ring tiles by load node, the
        carry staging tiles by store node, and the total.  Operands with
        the same sharing key share one tile; every loop tile has a ring
        slot per stage (a single-slot tile rewritten at the top of the next
        iteration would race with lanes still reading it: there is no
        trailing barrier)."""
        tiles: dict[tuple[object, ...], SmemTileSpec] = {}
        offset = 0
        for dot in self.program.dots:
            for operand in (dot.a, dot.b):
                route = self.route(dot, operand)
                if route in REGISTER_ROUTES or operand.chain_key in tiles:
                    continue
                exchange = route == ROUTE_EXCHANGE
                # An exchange tile of the carry at loop entry is written
                # before the iteration barrier: two copies by parity.
                if exchange:
                    slots = self.exchange_copies(dot, operand)
                else:
                    slots = self.stages if dot.region == "loop" else 1
                spec = SmemTileSpec(
                    f"{'md_x' if exchange else 'md_t'}{len(tiles)}",
                    offset,
                    self.size(operand.rows),
                    self.size(operand.cols),
                    operand.dtype.itemsize,
                    slots,
                )
                offset += spec.bytes * spec.slots
                if route == ROUTE_CONVERT:
                    load = operand.load
                    assert load is not None
                    spec = dataclasses.replace(
                        spec, raw_offset=offset, raw_elem_bytes=load.dtype.itemsize
                    )
                    offset += spec.raw_bytes * spec.slots
                tiles[operand.chain_key] = spec
        # Derived values' tiles: one ring slot per stage, read per lane.
        derived: dict[Node, SmemTileSpec] = {}
        for _derived, access in self.ring_derived():
            rows, cols = access.tile_dims
            spec = SmemTileSpec(
                f"md_d{len(derived)}",
                offset,
                self.size(rows),
                self.size(cols),
                access.dtype.itemsize,
                self.stages,
            )
            derived[access.node] = spec
            offset += spec.bytes * spec.slots
        staging: dict[Node, SmemTileSpec] = {}
        for store in self.staged_stores:
            rows, cols = self.slot_shape(self.program.frag_slots[store.frag])
            spec = SmemTileSpec(
                f"md_sg{len(staging)}",
                offset,
                rows,
                cols,
                store.access.dtype.itemsize,
                2,
            )
            staging[store.access.node] = spec
            offset += 2 * spec.bytes
        return tiles, derived, staging, offset

    @property
    def smem_bytes(self) -> int:
        return self.smem_layout()[3]


# Fragment operand routes: ``identity`` repacks the producer's registers into
# the operand words (an ``[M, K]`` fragment feeding A, an ``[N, K]`` one
# feeding B), ``movmatrix`` transposes each 8x8 block in registers (a ``[K,
# N]`` fragment feeding B) and ``exchange`` goes through a shared bf16 tile.
ROUTE_IDENTITY = "identity"
ROUTE_MOVMATRIX = "movmatrix"
ROUTE_EXCHANGE = "exchange"
REGISTER_ROUTES: tuple[str, ...] = (ROUTE_IDENTITY, ROUTE_MOVMATRIX)
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
    if warps is not None and facts.edges:
        _layouts, routes, _bytes = mma_dag_layout_solution(facts, resolved, warps)
        exchanged = {
            edge.key
            for edge, route in zip(facts.edges, routes, strict=True)
            if route == ROUTE_EXCHANGE
        }
    smem = 0
    for tile in facts.tiles:
        if (
            tile.edge_key is not None
            and exchanged is not None
            and tile.edge_key not in exchanged
        ):
            continue
        slots = stages if tile.ring else tile.copies
        smem += (
            slots * resolved[tile.rows] * resolved[tile.cols] * tile.bytes_per_element
        )
    for rows, cols, elem_bytes in facts.staging:
        smem += 2 * resolved[rows] * resolved[cols] * elem_bytes
    return smem


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
        a_words = (
            wtm * k // 64
            if facts_dot.a_identity and wn == 1 and a_route in REGISTER_ROUTES
            else wtm // 4
        )
        # A register-routed B fragment binds its words for every K step.
        b_words = (
            k * wtn // 64
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
) -> str | None:
    """Why the block sizes / warps / stages cannot run the program (None: they can).

    ``sizes`` maps block ids to the config's block sizes.
    """
    if type(warps) is not int or warps not in MMA_DAG_WARP_CHOICES:
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
    facts: CuteMmaDagFacts, sizes: Mapping[int, object], warps: object, stages: object
) -> tuple[dict[int, int], int, int] | None:
    """A legal ``(sizes, warps, stages)`` near the request: fewer stages
    first, then fewer warps, then the larger tunable tile halved (None: no
    legal config)."""
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
        # exchange-free count leads.
        candidates = [(max(legal), 1)]
        if any(edge.region == "loop" for edge in facts.edges):
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
        for axis, tile in zip(tunable, choice, strict=True):
            if axis.kind == AXIS_ROOT:
                ctas *= axis.extent // tile
                root_area *= tile
                root_perimeter += tile
            else:
                loop_area *= tile
        for warps, preferred in candidates:
            for stages in MMA_DAG_SEED_STAGES:
                if mma_dag_config_reason(facts, sizes, warps, stages) is not None:
                    continue
                smem = mma_dag_smem_estimate(facts, resolved_sizes, stages, warps)
                # Shared memory for two CTAs per SM counts once the grid has
                # more CTAs than SMs (bdv 16 at s2: 256 CTAs, two stages 342
                # us against 444 at three); otherwise the deeper ring wins.
                two_per_sm = int(
                    (facts.sm_count > 0 and ctas <= facts.sm_count)
                    or not facts.smem_capacity
                    or 2 * smem <= facts.smem_capacity
                )
                # Among equal root areas the squarer tile stages fewer
                # operand rows.
                rank = (
                    int(ctas >= 128),
                    root_area,
                    -root_perimeter,
                    preferred,
                    two_per_sm,
                    loop_area,
                    stages,
                    warps,
                )
                ranked.append((rank, (sizes, warps, stages)))
    ranked.sort(key=operator.itemgetter(0), reverse=True)
    return [seed for _rank, seed in ranked]


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


class _NotMine(Exception):
    """Raised inside the matcher for any shape it does not own."""


class _Matcher:
    def __init__(self, graphs: Sequence[GraphInfo]) -> None:
        self.graphs = graphs
        self.axes: dict[int, Axis] = {}
        self.axis_list: list[Axis] = []
        self.root_block_ids: tuple[int, ...] = ()
        self.loop_block_id: int = -1
        self.loop_axis: Axis | None = None
        self.loop_symbol: sympy.Symbol | None = None
        self.frag_slots: dict[Node, int] = {}
        self.slots: list[tuple[Axis, Axis]] = []
        self.carry_placeholders: set[Node] = set()
        # Fragment leaves whose registers hold the FX value transposed.
        self.transposed: set[Node] = set()
        # Program order of the loop body (for the in-place register checks).
        self.order: dict[Node, int] = {}

    # -- axes ------------------------------------------------------------------

    def canonical(self, block_id: int) -> int:
        return self.env.canonical_block_id(block_id)

    def axis_of_block(self, block_id: int, kind: str | None = None) -> Axis:
        block_id = self.canonical(block_id)
        axis = self.axes.get(block_id)
        if axis is None:
            info = self.env.block_sizes[block_id]
            if kind is None:
                if block_id in self.root_block_ids:
                    kind = AXIS_ROOT
                elif info.reduction:
                    kind = AXIS_RDIM
                elif block_id == self.loop_block_id:
                    kind = AXIS_LOOP
                else:
                    raise _NotMine
            extent = _static_int(info.size)
            if extent is None or extent <= 0:
                raise _NotMine
            if kind == AXIS_RDIM and (extent & (extent - 1) or extent % MMA_DAG_K_STEP):
                raise _NotMine
            axis = Axis(len(self.axis_list), block_id, kind, extent)
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
        return isinstance(node, torch.fx.Node) and node in self.frag_slots

    def is_tile_id(self, subscript: object) -> bool:
        """A ``tile.id`` symbol: a scalar coordinate, not a tile."""
        from ..host_function import HostFunction
        from ..variable_origin import TileIdOrigin

        if not isinstance(subscript, torch.fx.Node):
            return False
        val = subscript.meta.get("val")
        if not isinstance(val, torch.SymInt):
            return False
        expr = val._sympy_()
        info = HostFunction.current().expr_to_origin.get(expr)
        return info is not None and isinstance(info.origin, TileIdOrigin)

    def uniform_expr(self, subscript: object) -> tuple[sympy.Expr, bool]:
        from ..host_function import HostFunction
        from ..variable_origin import GridOrigin
        from ..variable_origin import TensorSizeOrigin
        from ..variable_origin import TileIdOrigin

        if isinstance(subscript, bool):
            raise _NotMine
        if isinstance(subscript, int):
            return sympy.Integer(subscript), False
        if not isinstance(subscript, torch.fx.Node):
            raise _NotMine
        val = subscript.meta.get("val")
        if isinstance(val, int) and not isinstance(val, bool):
            return sympy.Integer(val), False
        if not isinstance(val, torch.SymInt):
            raise _NotMine
        expr = val._sympy_()
        if not isinstance(expr, sympy.Expr):
            raise _NotMine
        expr_to_origin = HostFunction.current().expr_to_origin
        loop_dependent = False
        for symbol in expr.free_symbols:
            info = expr_to_origin.get(cast("sympy.Symbol", symbol))
            if info is None:
                raise _NotMine
            origin = info.origin
            if type(origin) is GridOrigin:
                if (
                    self.canonical(origin.block_id) != self.loop_block_id
                    or self.loop_axis is not None
                ):
                    raise _NotMine
                if self.loop_symbol is None:
                    self.loop_symbol = cast("sympy.Symbol", symbol)
                elif self.loop_symbol != symbol:
                    raise _NotMine
                loop_dependent = True
            elif isinstance(origin, TileIdOrigin):
                if self.axis_of_block(origin.block_id).kind != AXIS_LEAD:
                    raise _NotMine
            elif not isinstance(origin, TensorSizeOrigin):
                raise _NotMine
        return expr, loop_dependent

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
            tile_blocks = [*self.root_block_ids]
            if self.loop_axis is not None:
                tile_blocks.append(self.loop_block_id)
            if tile is not None and self.canonical(tile.block_id) in tile_blocks:
                if not self.env.known_equal(tile.offset, 0) or not value_sizes:
                    raise _NotMine
                axis = self.axis_of_block(tile.block_id)
                size = value_sizes.pop(0)
                size_block = self.env.resolve_block_id(size)
                if size_block is None or self.canonical(size_block) != axis.block_id:
                    raise _NotMine
                if extent is None or extent != axis.extent:
                    raise _NotMine
                subscripts.append(Subscript("axis", axis))
                dims.append(axis)
                dim_strides.append(strides[dim])
                continue
            expr, loop_dependent = self.uniform_expr(index)
            subscripts.append(Subscript("scalar", None, expr, loop_dependent))
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

    def has_source(self, node: Node, memo: dict[Node, bool]) -> bool:
        """Whether a 2-D tile load or a fragment lies under ``node``."""
        from ...language import memory_ops

        if node in memo:
            return memo[node]
        if self.is_frag_leaf(node):
            result = True
        elif _is_call(node, memory_ops.load):
            result = len(self.tile_dims(node)) == 2
        elif node.op != "call_function":
            result = False
        else:
            result = any(self.has_source(inp, memo) for inp in _node_inputs(node))
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
        if node.op != "call_function":
            raise _NotMine
        if _is_call(node, memory_ops.load):
            access = self.access(node)
            if len(access.tile_dims) > 1 or access.dtype not in _AUX_DTYPES:
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
        if not isinstance(val, torch.Tensor) or val.dtype not in _OPERAND_DTYPES:
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
        else:
            source_dims = self.tile_dims(source)
        if len(source_dims) != 2:
            raise _NotMine
        logical_swap = swap != flip
        if (
            tuple(reversed(source_dims)) if logical_swap else source_dims
        ) != operand_dims:
            raise _NotMine
        k_source_pos = (
            (0 if logical_swap else 1) if role == "a" else (1 if logical_swap else 0)
        )
        if self.is_frag_leaf(source):
            rows, cols = source_dims
            trans = k_source_pos == 0
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
            )
        access = self.access(source)
        if access.tile_dims != source_dims:
            raise _NotMine
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
        if not pointwise and access.dtype not in _OPERAND_DTYPES:
            raise _NotMine
        if pointwise and access.dtype not in (*_OPERAND_DTYPES, torch.float32):
            raise _NotMine
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
            trans=k_source_pos == rows_pos,
            aux_loads=tuple(aux),
            chain_key=(source, chain),
            flip=flip,
            rows_pos=rows_pos,
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
            for user in stack.pop().users:
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
                    or _is_swap_permute(user)
                    or _is_call(user, view_ops.subscript)
                    or _is_pointwise(user)
                ):
                    stack.append(user)
                else:
                    consumers.append(user)
        return consumers

    def accumulates_in_place(self, node: Node, leaf: Node, region: str) -> bool:
        """Whether dot ``node`` may accumulate into ``leaf``'s own registers.

        The dots and stores of a region are emitted in program order and a
        dot's accumulator is materialized right before its MMAs, so every
        other consumer of the leaf must come earlier in program order (it
        has read the registers by then) -- or be a loop-entry snapshot of
        the carry, which the loop body writes before any dot.  A later
        consumer would read the accumulated value: it forces a fresh slot.
        """
        from ...language import memory_ops

        order = self.node_order
        for consumer in self.leaf_consumers(leaf):
            if consumer is node or order[consumer] < order[node]:
                continue
            if (
                region == "loop"
                and _is_call(consumer, memory_ops.store)
                and leaf in self.carry_placeholders
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
            while _is_call(current, _tracing_ops._new_var) or _is_call(
                current, _tracing_ops._mask_to
            ):
                current = cast("Node", current.args[0])
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
        from ...language.matmul_ops import dot

        parities: list[bool] = []

        def visit(current: Node, parity: bool) -> bool:
            for user in current.users:
                if _is_call(user, dot):
                    if user.args[2] is current or current not in user.args[:2]:
                        return False
                    parities.append(parity)
                elif _is_swap_permute(user):
                    if not visit(user, not parity):
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

    def users_before(self, leaf: Node, limit: Node) -> bool:
        """Whether every use of fragment ``leaf`` (and of its ``_new_var``
        aliases) other than ``limit`` itself precedes ``limit`` in program
        order, so ``limit`` may overwrite the leaf's registers."""
        from ...language import _tracing_ops

        stack = [leaf]
        seen: set[Node] = set()
        while stack:
            current = stack.pop()
            for user in current.users:
                if user in seen or user is limit:
                    continue
                seen.add(user)
                if _is_call(user, _tracing_ops._new_var) or _is_call(
                    user, _tracing_ops._mask_to
                ):
                    stack.append(user)
                    continue
                if user.op == "output":
                    return False
                if self.order.get(user, -1) >= self.order[limit]:
                    return False
        return True

    # -- the match ----------------------------------------------------------

    def match(self) -> MmaDagProgram | None:
        try:
            return self._match()
        except _NotMine:
            return None

    def _match(self) -> MmaDagProgram:
        from ...language import _tracing_ops
        from ...language import memory_ops
        from ...language.matmul_ops import dot
        from ..device_ir import ForLoopGraphInfo
        from ..device_ir import RootGraphInfo
        from ..host_function import HostFunction
        from .cute_mma import analyze_cute_mma_node

        graphs = self.graphs
        roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
        loops = [graph for graph in graphs if isinstance(graph, ForLoopGraphInfo)]
        if len(graphs) != 2 or len(roots) != 1 or len(loops) != 1:
            raise _NotMine
        root, loop = roots[0], loops[0]
        self.node_order = {
            node: index
            for graph in graphs
            for index, node in enumerate(graph.graph.nodes)
        }
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
                if _is_call(node, dot) and (
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
        if len(loop.block_ids) != 1:
            raise _NotMine
        self.loop_block_id = self.canonical(loop.block_ids[0])
        if self.loop_block_id in self.root_block_ids:
            raise _NotMine
        loop_source = self.env.block_sizes[self.loop_block_id].block_size_source
        if isinstance(loop_source, FixedBlockSizeSource):
            if loop_source.value != 1:
                raise _NotMine
        else:
            self.loop_axis = self.axis_of_block(self.loop_block_id, AXIS_LOOP)

        # -- root: carry inits, the loop call, the phis, then the epilogue.
        loop_calls = [
            node for node in root.graph.nodes if _is_call(node, _tracing_ops._for_loop)
        ]
        if len(loop_calls) != 1:
            raise _NotMine
        loop_node = loop_calls[0]
        if (
            len(loop_node.args) != 4
            or loop_node.args[0] != loop.graph_id
            or not isinstance(loop_node.args[1], (list, tuple))
            or not isinstance(loop_node.args[2], (list, tuple))
            or not isinstance(loop_node.args[3], (list, tuple))
            or len(loop_node.args[1]) != 1
            or len(loop_node.args[2]) != 1
        ):
            raise _NotMine
        begin, end = loop_node.args[1][0], loop_node.args[2][0]
        for bound in (begin, end):
            if isinstance(bound, torch.fx.Node):
                if not isinstance(bound.meta.get("val"), (int, torch.SymInt)):
                    raise _NotMine
            elif type(bound) is not int:
                raise _NotMine
        if self.loop_axis is not None:
            # A tile loop walks its axis' whole extent in block-size steps.
            if (
                begin != 0
                or _static_int(
                    end if not isinstance(end, torch.fx.Node) else end.meta.get("val")
                )
                != self.loop_axis.extent
            ):
                raise _NotMine
        inits = list(loop_node.args[3])
        placeholders = list(loop.graph.find_nodes(op="placeholder"))
        (output,) = loop.graph.find_nodes(op="output")
        outputs = output.args[0]
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(inits):
            raise _NotMine
        if len(placeholders) != len(inits) or not inits:
            raise _NotMine
        getitems: dict[int, Node] = {}
        for user in loop_node.users:
            if user.target is not operator.getitem or type(user.args[1]) is not int:
                raise _NotMine
            getitems[user.args[1]] = user
        if sorted(getitems) != list(range(len(inits))):
            raise _NotMine
        (root_output,) = root.graph.find_nodes(op="output")
        if root_output.args != (None,):
            raise _NotMine

        # The dots tell which root blocks tile fragments; the rest are leads.
        dot_nodes = [
            node
            for graph in (loop, root)
            for node in graph.graph.nodes
            if _is_call(node, dot)
        ]
        if not dot_nodes:
            raise _NotMine
        tiled_blocks: set[int] = set()
        for node in dot_nodes:
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor) or val.dtype != torch.float32:
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
        if not any(axis.kind == AXIS_ROOT for axis in self.axis_list):
            raise _NotMine

        # -- carries
        carries: list[Carry] = []
        for index, (init, placeholder) in enumerate(
            zip(inits, placeholders, strict=True)
        ):
            if not isinstance(init, torch.fx.Node):
                raise _NotMine
            getitem = getitems[index]
            if getitem.args != (loop_node, index):
                raise _NotMine
            phis = list(getitem.users)
            if len(phis) != 1 or not _is_call(phis[0], _tracing_ops._phi):
                raise _NotMine
            phi = phis[0]
            if phi.args != (init, getitem):
                raise _NotMine
            carry_dims = self.tile_dims(placeholder)
            val = placeholder.meta.get("val")
            if (
                len(carry_dims) != 2
                or not isinstance(val, torch.Tensor)
                or val.dtype != torch.float32
                or self.tile_dims(init) != carry_dims
            ):
                raise _NotMine
            carry_init = self._carry_init(init)
            slot = len(self.slots)
            self.slots.append((carry_dims[0], carry_dims[1]))
            self.frag_slots[placeholder] = slot
            self.frag_slots[phi] = slot
            self.carry_placeholders.add(placeholder)
            carries.append(
                Carry(
                    slot=slot,
                    init=carry_init,
                    placeholder=placeholder,
                    output=cast("Node", outputs[index]),
                    phi=phi,
                    rows=carry_dims[0],
                    cols=carry_dims[1],
                )
            )
        # ``_new_var`` copies of the placeholders are the same fragments.
        for node in loop.graph.nodes:
            if (
                _is_call(node, _tracing_ops._new_var)
                or _is_call(node, _tracing_ops._mask_to)
            ) and node.args[0] in self.frag_slots:
                self.frag_slots[node] = self.frag_slots[cast("Node", node.args[0])]

        # -- loop body in program order: dots, derived fragments (pointwise
        # values over several fragments / tiles) and stores.
        dots: list[Dot] = []
        derived: list[Derived] = []
        steps: list[Dot | Derived | Store] = []
        stores: list[Store] = []
        acc_loads: dict[Node, tuple[Access, ...]] = {}
        self.order = {node: index for index, node in enumerate(loop.graph.nodes)}
        for node in loop.graph.nodes:
            if _is_call(node, memory_ops.load):
                self.access(node)
        for node in loop.graph.nodes:
            if _is_call(node, dot):
                step: Dot | Derived | Store = self._dot(node, "loop", acc_loads)
                dots.append(cast("Dot", step))
            elif _is_call(node, memory_ops.store):
                step = self._store(node, "loop")
                stores.append(cast("Store", step))
            elif (
                _is_pointwise(node)
                and len(self.tile_dims(node)) == 2
                and len(self.sources(node)) > 1
            ):
                step = self._derived(node)
                derived.append(cast("Derived", step))
            else:
                continue
            steps.append(step)
        produced = {d.node for d in dots} | {d.node for d in derived}
        for carry in carries:
            # The loop output is a dot or derived value computed into the
            # carry's own slot.
            if (
                carry.output not in produced
                or self.frag_slots.get(carry.output) != carry.slot
            ):
                raise _NotMine
        closure = self._closure(
            [
                *(d.node for d in dots),
                *(d.node for d in derived),
                *(s.access.node for s in stores),
            ]
        )
        for node in loop.graph.nodes:
            if node.op == "call_function" and node not in closure:
                raise _NotMine

        # -- epilogue: dots and stores after the loop in program order.
        after_loop = False
        epilogue_nodes: list[Node] = []
        for node in root.graph.nodes:
            if node is loop_node:
                after_loop = True
                continue
            if after_loop and node.op == "call_function":
                epilogue_nodes.append(node)
        epilogue_order: list[Node] = []
        for node in epilogue_nodes:
            if _is_call(node, dot):
                dots.append(self._dot(node, "epilogue", acc_loads))
                epilogue_order.append(node)
            elif _is_call(node, memory_ops.store):
                stores.append(self._store(node, "epilogue"))
                epilogue_order.append(node)
        store_aux: list[Access] = []
        for store in stores:
            _leaf, _swap, aux = self.frag_value(store.value)
            store_aux.extend(aux)
        anchors = [
            *(d.node for d in dots),
            *(s.access.node for s in stores),
            loop_node,
            *inits,
        ]
        closure = self._closure(anchors)
        for node in root.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node in closure
                or node in getitems.values()
                or node.target is _tracing_ops._phi
            ):
                continue
            raise _NotMine
        if not stores:
            raise _NotMine
        program = MmaDagProgram(
            root_graph_id=root.graph_id,
            loop_graph_id=loop.graph_id,
            loop_node=loop_node,
            loop_axis=self.loop_axis,
            loop_block_id=self.loop_block_id,
            loop_symbol=self.loop_symbol,
            begin=begin,
            end=end,
            axes=tuple(self.axis_list),
            carries=tuple(carries),
            dots=tuple(dots),
            stores=tuple(stores),
            slots=tuple(self.slots),
            frag_slots=dict(self.frag_slots),
            steps=tuple(steps),
            derived=tuple(derived),
            acc_loads=acc_loads,
            store_aux_loads=tuple(store_aux),
            epilogue_order=tuple(epilogue_order),
        )
        dtype = program.dots[0].a.dtype
        for d in program.dots:
            if d.a.dtype != dtype or d.b.dtype != dtype:
                raise _NotMine
        return program

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

    def _carry_init(self, init: Node) -> Access | None:
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
            return None
        source = init
        if _is_call(source, torch.ops.prims.convert_element_type.default):
            if source.args[1] is not torch.float32:
                raise _NotMine
            source = cast("Node", source.args[0])
        if not _is_call(source, memory_ops.load):
            raise _NotMine
        access = self.access(source)
        if len(access.tile_dims) != 2 or access.dtype not in _STORE_DTYPES:
            raise _NotMine
        if any(subscript.loop_dependent for subscript in access.subscripts):
            raise _NotMine
        return access

    def _dot(
        self, node: Node, region: str, acc_loads: dict[Node, tuple[Access, ...]]
    ) -> Dot:
        if len(node.args) not in (3, 4) or (
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
        if region == "epilogue" and k.kind == AXIS_LOOP:
            raise _NotMine
        acc = node.args[2]
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
            acc_loads[node] = tuple(aux)
            if self.accumulates_in_place(node, acc_frag, region):
                slot = acc_slot
            else:
                slot = len(self.slots)
                self.slots.append((rows, cols))
        self.frag_slots[node] = slot
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
        )

    def _derived(self, node: Node) -> Derived:
        """A pointwise value over several fragments / 2-D tiles becomes a
        fragment of its own, computed register by register (tiles loaded at
        the fragment coordinates)."""
        from ...language import memory_ops

        dims = self.tile_dims(node)
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
        if not frags:
            raise _NotMine
        aux = self.aux_of(node)
        # Reuse the slot of a fragment input that dies here (its only uses
        # are inside this value's DAG), else a fresh one.
        slot = -1
        for leaf in frags:
            if self.users_before(leaf, node) and self.last_use_is(leaf, node):
                slot = self.frag_slots[leaf]
                break
        if slot < 0:
            slot = len(self.slots)
            self.slots.append((dims[0], dims[1]))
        self.frag_slots[node] = slot
        return Derived(
            node=node,
            slot=slot,
            frags=tuple(frags),
            loads=tuple(loads),
            aux_loads=tuple(aux),
            rows=dims[0],
            cols=dims[1],
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
        from ...language import _tracing_ops
        from ...language import memory_ops
        from ...language import view_ops

        aux: list[Access] = []
        seen: set[Node] = set()

        def visit(current: Node) -> None:
            while _is_call(current, _tracing_ops._new_var) or _is_call(
                current, _tracing_ops._mask_to
            ):
                current = cast("Node", current.args[0])
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

    def last_use_is(self, leaf: Node, node: Node) -> bool:
        """Whether ``node`` is the last reader of fragment ``leaf`` in the
        loop body (so the derived value may take over its registers)."""
        from ...language import _tracing_ops

        stack = [leaf]
        seen: set[Node] = set()
        while stack:
            current = stack.pop()
            for user in current.users:
                if user in seen or user is node:
                    continue
                seen.add(user)
                if _is_call(user, _tracing_ops._new_var) or _is_call(
                    user, _tracing_ops._mask_to
                ):
                    stack.append(user)
                elif self.order.get(user, -1) > self.order[node]:
                    return False
        return True

    def _store(self, node: Node, region: str) -> Store:
        from .matmul_utils import strip_new_vars

        access = self.access(node)
        if len(access.tile_dims) != 2 or access.dtype not in _STORE_DTYPES:
            raise _NotMine
        value = node.args[2]
        if not isinstance(value, torch.fx.Node):
            raise _NotMine
        value_fake = value.meta.get("val")
        if not isinstance(value_fake, torch.Tensor) or value_fake.dtype != access.dtype:
            raise _NotMine
        leaf, swap, _aux = self.frag_value(value)
        if self.tile_dims(value) != access.tile_dims:
            raise _NotMine
        rows, cols = self.slots[self.frag_slots[leaf]]
        expected = (cols, rows) if swap else (rows, cols)
        if access.tile_dims != expected:
            raise _NotMine
        if region == "loop":
            # A loop store snapshots a carry at loop entry (hoisted to the
            # iteration top) or stores a value of the iteration in place.
            source = "carry_in" if leaf in self.carry_placeholders else "body"
            stored = strip_new_vars(value)
            if stored is not leaf and not (
                _is_call(stored, torch.ops.prims.convert_element_type.default)
                and strip_new_vars(cast("Node", stored.args[0])) is leaf
            ):
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
        )


def match_mma_dag(graphs: Sequence[GraphInfo]) -> MmaDagProgram | None:
    """Match the MMA dataflow program in ``graphs`` (config-free).

    Needs the compile environment and host function to be current (the
    autotuner heuristic enters them; pre-codegen already runs inside them).
    """
    return _Matcher(graphs).match()


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
            tiles.append(
                TileFacts(
                    operand.rows.index,
                    operand.cols.index,
                    operand.dtype.itemsize,
                    operand.kind == "load" and dot.region == "loop",
                    source_bytes,
                    2 if program.early_exchange(dot, operand) else 1,
                    edge_keys.setdefault(operand.chain_key, len(edge_keys))
                    if operand.kind == "frag"
                    else None,
                )
            )
    # Derived values' tiles ride the ring too (read per lane from the slot).
    derived_loads: dict[Node, Access] = {}
    for derived in program.derived:
        for access in derived.loads:
            derived_loads.setdefault(access.node, access)
    for access in derived_loads.values():
        rows, cols = access.tile_dims
        tiles.append(TileFacts(rows.index, cols.index, access.dtype.itemsize, True))
    staging = tuple(
        (
            program.slots[program.frag_slots[store.frag]][0].index,
            program.slots[program.frag_slots[store.frag]][1].index,
            store.access.dtype.itemsize,
        )
        for store in program.stores
        if store.region == "loop" and store.source == "carry_in"
    )
    return CuteMmaDagFacts(
        axes=tuple(AxisFacts(a.block_id, a.kind, a.extent) for a in program.axes),
        frags=tuple((rows.index, cols.index) for rows, cols in program.slots),
        tiles=tuple(tiles),
        staging=staging,
        k_axes=tuple(sorted({dot.k.index for dot in program.dots})),
        smem_capacity=CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device),
        grid_blocks=grid_blocks,
        live=mma_dag_live_slots(program),
        edges=mma_dag_frag_edges(program),
        sm_count=_device_sm_count(env.device),
        loop_passes=sum(
            1
            for dot in program.dots_in("loop")
            for operand in (dot.a, dot.b)
            if operand.load is not None
            and (operand.pointwise or operand.load.dtype != operand.dtype)
        ),
        dots=tuple(
            DotFacts(
                dot.slot,
                dot.k.index,
                dot.a.kind == "frag" and not dot.a.logical_swap,
                dot.b.kind == "frag",
            )
            for dot in program.dots
        ),
        identity_candidates=any(
            (dot.a.kind == "frag" and not dot.a.logical_swap) or dot.b.kind == "frag"
            for dot in program.dots
        ),
    )


def mma_dag_live_slots(program: MmaDagProgram) -> tuple[tuple[int, ...], ...]:
    """The slot sets live at each loop step and in the epilogue.

    Carries are live throughout; a dot / derived result is live from its
    step to its last reader (operand, accumulator, derived input or store).
    """
    carry_slots = {carry.slot for carry in program.carries}
    defined: dict[int, int] = {}
    last_use: dict[int, int] = {}

    def use(slot: int, index: int) -> None:
        last_use[slot] = max(last_use.get(slot, index), index)

    for index, step in enumerate(program.steps):
        if isinstance(step, Dot):
            for operand in (step.a, step.b):
                if operand.frag is not None:
                    use(program.frag_slots[operand.frag], index)
            if step.acc_frag is not None:
                use(program.frag_slots[step.acc_frag], index)
            defined.setdefault(step.slot, index)
            use(step.slot, index)
        elif isinstance(step, Derived):
            for leaf in step.frags:
                use(program.frag_slots[leaf], index)
            defined.setdefault(step.slot, index)
            use(step.slot, index)
        else:
            use(program.frag_slots[step.frag], index)
    groups: list[tuple[int, ...]] = []
    for index in range(len(program.steps)):
        live = set(carry_slots)
        for slot, start in defined.items():
            if start <= index <= last_use[slot]:
                live.add(slot)
        groups.append(tuple(sorted(live)))
    epilogue = set(carry_slots)
    epilogue.update(dot.slot for dot in program.dots_in("epilogue"))
    groups.append(tuple(sorted(epilogue)))
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
    if load.dtype != operand.dtype and load.dtype != torch.float32:
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
    """
    rows_c, cols_c = consumer
    wm_p, wn_p = layout_p
    wm_c, wn_c = layout_c
    if role == "a":
        if logical_swap:
            return None
        if producer == (rows_c, k) and wn_p == wn_c == 1 and wm_p == wm_c:
            return ROUTE_IDENTITY
        return None
    if not logical_swap:
        if producer == (k, cols_c) and wm_p == wm_c == 1 and wn_p == wn_c:
            return ROUTE_MOVMATRIX
        return None
    if producer == (cols_c, k) and wn_p == 1 and wm_c == 1 and wm_p == wn_c:
        return ROUTE_IDENTITY
    return None


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
                    region=dot.region,
                )
            )
    return tuple(edges)


def mma_dag_layout_solution(
    facts: CuteMmaDagFacts, resolved: Sequence[int], warps: int
) -> tuple[Layouts, tuple[str, ...], int]:
    """The warp grid of every fragment shape, the route of every fragment
    operand (by ``facts.edges`` order) and the bytes exchanged through shared
    memory per loop iteration: among the M-first / N-first choices per
    shape, the combination exchanging the fewest bytes per iteration (ties:
    fewer exchanged tiles, then M-first)."""
    slot_shapes = [(resolved[rows], resolved[cols]) for rows, cols in facts.frags]
    shapes = sorted(set(slot_shapes))
    best: tuple[tuple[int, int, int], Layouts, tuple[str, ...], int] | None = None
    for n_first in itertools.product((False, True), repeat=len(shapes)):
        layouts: Layouts = {}
        for shape, flag in zip(shapes, n_first, strict=True):
            layout = mma_dag_warp_layout(*shape, warps, n_first=flag)
            if layout is None:
                break
            layouts[shape] = layout
        if len(layouts) != len(shapes):
            continue
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
            )
            if route is None:
                route = ROUTE_EXCHANGE
                if edge.key not in exchanged:
                    exchanged.add(edge.key)
                    if edge.region == "loop":
                        loop_bytes += producer[0] * producer[1] * edge.elem_bytes
            routes.append(route)
        cost = (loop_bytes, len(exchanged), sum(n_first))
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
    if mode != MMA_DAG_WARP:
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
    facts = mma_dag_facts(program)
    reason = mma_dag_config_reason(facts, sizes_by_block, warps, stages)
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
    assert type(warps) is int and type(stages) is int
    if get("pid_type", "flat") != "flat":
        raise _decline("the lowering needs the flat program-id grid")
    sizes = tuple(
        cast("int", facts.block_size(axis.index, sizes_by_block))
        for axis in program.axes
    )
    layouts, routes = solve_mma_dag_layouts(program, facts, sizes, warps)
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
            # The derived value's tile rides the ring when cp.async can
            # stage it; otherwise each lane loads its pairs from global
            # memory at the fragment coordinates.
            derived_routes[access.node] = (
                DERIVED_GLOBAL
                if async_tile_reason(env, access, 1) is not None
                else DERIVED_SHARED
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
    "MMA_DAG_MODES",
    "MMA_DAG_OFF",
    "MMA_DAG_SEED_STAGES",
    "MMA_DAG_STAGES_KEY",
    "MMA_DAG_STAGE_CHOICES",
    "MMA_DAG_STORE_CHOICES",
    "MMA_DAG_STORE_DIRECT",
    "MMA_DAG_STORE_KEY",
    "MMA_DAG_STORE_STAGED",
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
]
