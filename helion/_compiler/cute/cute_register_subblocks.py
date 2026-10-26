"""The ``cute_register_subblocks`` family: register-resident static sub-block
programs on ``mma.sync`` tf32.

The anchored chunk grams of the linear-attention engine (``chunk_bwd_gram2_kda``,
``chunk_fwd_A_diag_anchored`` and its varlen variant) are many small static
programs: one CTA per (batch row, channel tile) runs a few 16-row sub-blocks
through pointwise fp32 gating chains and ``hl.dot`` of 16x16 fp32 blocks, in
static Python loops.  The computed-fragment lowering materializes every value
of such a body in shared memory behind a CTA barrier (44 barriers and 51 KB
of staging per CTA on the backward gram); this family renders the whole root
body as ONE straight-line DSL program per warp instead: every operand is
loaded from global memory straight into the ``mma.sync.m16n8k8`` fragment
layouts, the pointwise chains are evaluated per lane on its own fragment
elements in whichever layout the consumer needs, the dots run on tensor cores
at tf32 precision and the outputs are stored from the accumulator layout.  No
shared memory, no barriers.

Measured on B200 (device span of the standalone kernel, same tf32 numerics as
the fragment route and Triton): ``chunk_bwd_gram2_kda`` at (2048 x 64 x 128)
168 us against the fragment route's 500 us and Triton's 338 us for the
hand-written program this emitter reproduces (one warp per CTA, 16-channel
tiles, one item of prefetch).

Admission is a config-free fact of the device IR (``match_register_subblocks``):
one root tile loop with one ``block_size=1`` batch axis and at most one tiled
axis; static windows of literal extents (``start + hl.arange(E)`` or a literal
slice); pointwise chains in fp32, or in bf16 with every result rounded as
torch rounds it (casts, ``*``, ``+``, ``-``, ``neg``, ``exp2``, ``rsqrt``,
``where`` with static masks of ``hl.arange`` origin; a repeated operand such
as ``x * x``, which the lowering spells ``mul(x, None)``); the per-row
``sum(dim=-1, keepdim=True)`` of an fp32 tile over a static extent (the
fused row L2 norm ``x * rsqrt((x * x).sum(-1, keepdim=True) + eps)``);
``hl.dot`` with fp32 operands under tf32 ``dot_precision`` on sm_80+; plain
``=`` / ``+=`` stores; static Python loops and ``if`` guards flattened at
plan time.  Every integer the flattening needs (loop bounds, ``if`` tests,
window starts, subscripts) must be a literal, a specialized size or an
expression over the root axes: int kernel arguments and dynamic shapes are
runtime values and fail closed.  Anything else fails closed with the reason,
which the facts carry so the tests can assert refusals.

An output the program stores is read back only as it was written: by the
lane that stored it, at the coordinates it stored (a ``+=`` of the same
window, or any in-place epilogue of it); a window read in another layout, a
partially overlapping window, a window read before a write of the same item
and used after it, and a carried value reading a window its trip writes all
fail closed (there is no fence between the lanes of the warp).  Inside one
item, a store that writes a window another store reads or writes follows
every run of that store (``_store_waves``).

Lane layouts (lane ``(g, c) = (lane // 4, lane % 4)``), for a dot of
``M x K`` by ``K x N`` (``NT = N // 8`` n-tiles, ``Kq = K // 4``):

* A operand: rows ``g + 8h`` (+ 16 per m-tile), K columns ``[Kq*c, Kq*(c+1))``;
* B operand: K rows ``[Kq*c, Kq*(c+1))``, N columns ``[NT*g, NT*(g+1))``
  (``permuted`` N) or ``8*nt + g`` (``tiled`` N);
* C (accumulators, epilogues, stores): rows ``g + 8h``, N columns
  ``[2*NT*c, 2*NT*(c+1))`` (permuted) or ``8*nt + 2c + e`` (tiled);
* a row reduction over ``E`` columns: each lane sums its A-operand quarter
  (``k_aff`` over ``E / 4`` values, the registers a dot over ``K = E`` reads)
  and the four lanes of the row exchange partials by two butterfly shuffles,
  so every lane holds the row's total.

The K index is permuted as ``k(s, kk) = Kq*(kk % 4) + 2*s + kk // 4`` and the
N index (permuted mode) as ``n(nt, j) = NT*j + nt``: both are bijections, so
every dot is the same sum in another order and every lane's operand run is
contiguous in memory (16-byte packets at the usual tile widths).  The N mode
is ``tiled`` when some dot's B operand is K-contiguous (the forward gram's
transposed column operand, streamed one n-tile at a time) and ``permuted``
otherwise (the backward gram's channel-tile operands, loaded for every n-tile
at once).

Knobs (``register_subblocks_config_reason`` validates them against the facts):

* ``cute_register_subblocks``: ``off`` | ``on`` (the family switch);
* the tiled axis' block size is the channel tile (16 | 32 | 64);
* ``cute_register_subblocks_warps``: warps per CTA (consecutive work items);
* ``cute_register_subblocks_prefetch``: items of input loads issued ahead;
* ``cute_register_subblocks_mask``: static masks by ``selp`` on loaded values
  or by predicated zero-filling loads;
* ``cute_register_subblocks_colside``: ``+=`` windows written more than once
  read-modify-write memory or accumulate in registers until their last write.

A ``+=`` into a compiler-allocated zero-filled output the CTA owns (every
store's batch index is the CTA's own row and, under a tiled axis, its own
tile) reads zeros before its first write: that load is the constant zero and
the first write is a plain store; later writes read memory (or registers).
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import operator
import re
import struct
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import exc
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..compile_environment import LoopSpecBlockSizeSource
from ..compile_environment import shape_env_size_hint
from ..compile_environment import shape_env_var_hints
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import GridOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST

KEY = "cute_register_subblocks"
MODE_OFF = "off"
MODE_ON = "on"
MODES: tuple[str, ...] = (MODE_OFF, MODE_ON)
WARPS_KEY = "cute_register_subblocks_warps"
WARP_CHOICES: tuple[int, ...] = (1, 2, 4)
DEFAULT_WARPS = 1
PREFETCH_KEY = "cute_register_subblocks_prefetch"
PREFETCH_CHOICES: tuple[int, ...] = (0, 1)
DEFAULT_PREFETCH = 0
MASK_KEY = "cute_register_subblocks_mask"
MASK_SELECT = "select"
MASK_PREDICATE = "predicate"
MASK_CHOICES: tuple[str, ...] = (MASK_SELECT, MASK_PREDICATE)
COLSIDE_KEY = "cute_register_subblocks_colside"
COLSIDE_RMW = "rmw"
COLSIDE_REGS = "regs"
COLSIDE_CHOICES: tuple[str, ...] = (COLSIDE_RMW, COLSIDE_REGS)
ROLL_KEY = "cute_register_subblocks_roll"
ROLL_UNROLLED = "unrolled"
ROLL_LOOP = "rolled"
ROLL_CHOICES: tuple[str, ...] = (ROLL_UNROLLED, ROLL_LOOP)
KEYS: tuple[str, ...] = (KEY, WARPS_KEY, PREFETCH_KEY, MASK_KEY, COLSIDE_KEY, ROLL_KEY)
# The symbolic n-tile of a rolled item (``_DotLayout.tile_of``).
SYM_TILE = -1
# Channel tiles the register program may hold (one warp's fragments of a
# 16 x tile sub-block; 128 columns are 16 n-tiles of accumulators and spill).
TILE_CHOICES: tuple[int, ...] = (16, 32, 64)
# Widest row a lane-local reduction unrolls (``E / 4`` terms per lane).
MAX_REDUCE_EXTENT = 512
# fp32 values live per thread the liveness estimate may reach.  The estimate
# over-counts what ptxas keeps (it ignores rematerialization and the compiler's
# own reordering); calibrated on B200 against the backward gram's cubins:
# estimate 281 -> 255 registers and an 8-byte stack, 323 -> 64 B, 384 -> 520 B
# (still the fastest corner, 203 us), 549 -> 1368 B (285 us), 589 -> 363 us and
# 1089 -> 658 us of spill-bound time.  The cut keeps the corners whose spills
# the autotuner should still measure (549: 285 us) and drops the spill-bound
# ones (597: 363 us).  (600 would admit the 32-channel prefetch corner.)
REGISTER_BUDGET = 580
# A root whose flattened dots total this many multiply-adds seeds the family
# as the promoted default (the anchored grams: 20 pairs x 4 dots x 16x16x128).
SEED_MACS = 64**3
# ``mma.sync`` with tf32 operands exists from sm_80.
MIN_CAPABILITY = 8
PLAN_KIND = "helion_register_subblocks"
TILE = "tile"  # the symbolic extent of the tiled axis in value shapes
_RT = "_helion_rsb"
_RT_DAG = "_helion_rsb_dag"
_PREAMBLE = (
    f"import helion._compiler.cute._warp_mma_runtime as {_RT}\n"
    f"import helion._compiler.cute._mma_dag_runtime as {_RT_DAG}\n"
)
# (tile, prefetch, warps, colside) of the measured winners (B200,
# chunk_bwd_gram2_kda at 2048 x 64 x 128, device span): 16-channel tiles with
# register accumulation on four warps (201.5 us) and on one warp with a
# prefetched item (203.6 us; the fragment route runs 500 us, Triton 338); the
# read-modify-write corner of the same tile (280 us) and the 32-channel
# two-warp structure of the Triton winner (287 us) follow for the search.
SEEDS: tuple[tuple[int | None, int, int, str, str], ...] = (
    (16, 0, 4, COLSIDE_REGS, ROLL_UNROLLED),
    (16, 1, 1, COLSIDE_REGS, ROLL_UNROLLED),
    (16, 1, 2, COLSIDE_REGS, ROLL_UNROLLED),
    (16, 0, 1, COLSIDE_RMW, ROLL_UNROLLED),
    (32, 0, 2, COLSIDE_RMW, ROLL_UNROLLED),
)


class _NotMine(Exception):
    """The device IR is not a register sub-block program (the reason)."""


def _decline(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"register sub-block family: {reason}")


# --------------------------------------------------------------------------
# Lane coordinates


@dataclasses.dataclass(frozen=True)
class Affine:
    """``g * cg + c * cc + t * ct + k`` over the lane coordinates ``g = lane // 4``
    and ``c = lane % 4`` and the n-tile index ``t`` of a rolled loop (``ct`` is
    zero outside one)."""

    cg: int = 0
    cc: int = 0
    k: int = 0
    ct: int = 0
    # the group index ``rsb_grp`` of a parametric body (isomorphic groups
    # rendered once: their constants are affine in the group index)
    cgp: int = 0

    def __add__(self, other: Affine | int) -> Affine:
        if isinstance(other, int):
            return Affine(self.cg, self.cc, self.k + other, self.ct, self.cgp)
        return Affine(
            self.cg + other.cg,
            self.cc + other.cc,
            self.k + other.k,
            self.ct + other.ct,
            self.cgp + other.cgp,
        )

    def scale(self, factor: int) -> Affine:
        return Affine(
            self.cg * factor,
            self.cc * factor,
            self.k * factor,
            self.ct * factor,
            self.cgp * factor,
        )

    def __floordiv__(self, divisor: int) -> Affine:
        assert all(
            v % divisor == 0 for v in (self.cg, self.cc, self.k, self.ct, self.cgp)
        )
        return Affine(
            self.cg // divisor,
            self.cc // divisor,
            self.k // divisor,
            self.ct // divisor,
            self.cgp // divisor,
        )

    @property
    def is_const(self) -> bool:
        return self.cg == 0 and self.cc == 0 and self.ct == 0 and self.cgp == 0

    @property
    def lane_terms(self) -> tuple[int, int, int, int]:
        return (self.cg, self.cc, self.ct, self.cgp)

    def at_group(self, group: int) -> Affine:
        """The affine with the group term folded in for group ``group``."""
        return Affine(self.cg, self.cc, self.k + self.cgp * group, self.ct)

    def render(self, g: str, c: str) -> str:
        wide = g.endswith("64")
        t = "rsb_nt64" if wide else "rsb_nt"
        grp = "rsb_grp64" if wide else "rsb_grp"
        terms = []
        for coef, name in ((self.cg, g), (self.cc, c), (self.ct, t), (self.cgp, grp)):
            if coef == 1:
                terms.append(name)
            elif coef:
                terms.append(f"{coef} * {name}")
        if self.k or not terms:
            terms.append(str(self.k))
        return "(" + " + ".join(terms) + ")" if len(terms) > 1 else terms[0]


ZERO = Affine()
Coord = tuple[Affine, ...]


def _coord(lead: Coord, *parts: Affine) -> Coord:
    return (*lead, *parts)


# --------------------------------------------------------------------------
# Program IR


@dataclasses.dataclass(frozen=True)
class TensorRef:
    name: str
    fake: torch.Tensor
    strides: tuple[int, ...]
    shape: tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        return self.fake.dtype


# Index specs of one tensor dim of a load / store.
@dataclasses.dataclass(frozen=True)
class IdxBatch:
    """The CTA's batch row (the ``block_size=1`` tile)."""


@dataclasses.dataclass(frozen=True)
class IdxTile:
    """The CTA's channel tile (the tiled root axis)."""


@dataclasses.dataclass(frozen=True)
class IdxConst:
    value: int | Affine


@dataclasses.dataclass(frozen=True)
class IdxWindow:
    """``start + hl.arange(extent)``; the value dim may be padded to ``size``
    rows (the frontend rounds an arithmetic arange's block up), of which
    only the first ``extent`` are real."""

    start: int | Affine
    extent: int | Affine
    size: int | Affine = 0
    # A parametric window whose real extent varies by group: only the rolled
    # loop over its real columns may read it.
    padded: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.size, int) and not self.size:
            object.__setattr__(self, "size", self.extent)


@dataclasses.dataclass(frozen=True)
class IdxFull:
    extent: int


@dataclasses.dataclass(frozen=True)
class IdxScalar:
    """A lane-uniform integer expression over the batch index (and host ints)."""

    expr: sympy.Expr


@dataclasses.dataclass(frozen=True)
class IdxWindowBase:
    """``base + hl.arange(extent)`` at a uniform scalar load ``base`` (padded
    to ``size`` like ``IdxWindow``)."""

    base: Value
    start: int | Affine
    extent: int | Affine
    size: int | Affine = 0
    padded: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.size, int) and not self.size:
            object.__setattr__(self, "size", self.extent)


Idx = IdxBatch | IdxTile | IdxConst | IdxWindow | IdxFull | IdxScalar | IdxWindowBase
Shape = tuple[int | str | Affine, ...]


class Value:
    """A tile value of the flattened program (lazily rendered per lane)."""

    shape: Shape
    dtype: torch.dtype
    key: tuple[object, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __post_init__(self) -> None:
        """Recompute ``key`` from the children's keys (``_rekey``)."""


@dataclasses.dataclass(eq=False)
class Load(Value):
    ref: TensorRef
    dims: tuple[Idx, ...]
    # value dim of each tensor dim (None: a scalar subscript)
    value_dims: tuple[int | None, ...]
    pred: Value | None
    shape: Shape
    dtype: torch.dtype
    # The schedule item whose graph instance built the load (the prefetch
    # issues an item's own loads ahead, not every load its sinks pull in),
    # and how many stores of that item preceded the load in program order
    # (a load of an output observes exactly those writes).
    origin: object = None
    position: int = 0
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = (
            "load",
            self.ref.name,
            tuple(_idx_key(d) for d in self.dims),
            self.pred.key if self.pred is not None else None,
        )


@dataclasses.dataclass(eq=False)
class ScalarLoad(Value):
    """A lane-uniform 0-d load (a varlen window base or limit)."""

    ref: TensorRef
    dims: tuple[Idx, ...]
    dtype: torch.dtype
    shape: Shape = ()
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("sload", self.ref.name, tuple(_idx_key(d) for d in self.dims))


@dataclasses.dataclass(eq=False)
class Iota(Value):
    """``start + hl.arange(extent)`` (1-D)."""

    extent: int | Affine
    start: int | Affine
    shape: Shape = ()
    dtype: torch.dtype = torch.int32
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.shape = (self.extent,)
        self.key = ("iota", self.extent, self.start)


@dataclasses.dataclass(eq=False)
class WindowIota(Value):
    """``base + start + hl.arange(extent)`` at a uniform scalar load."""

    base: Value
    extent: int | Affine
    start: int | Affine
    shape: Shape = ()
    dtype: torch.dtype = torch.int32
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.shape = (self.extent,)
        self.key = ("wiota", self.base.key, self.extent, self.start)


@dataclasses.dataclass(eq=False)
class Const(Value):
    value: float
    shape: Shape = ()
    dtype: torch.dtype = torch.float32
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("const", self.value)


@dataclasses.dataclass(eq=False)
class ScalarSym(Value):
    """A host scalar (a float or int kernel argument / module constant)."""

    expr: sympy.Expr
    dtype: torch.dtype
    shape: Shape = ()
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("sym", str(self.expr))


@dataclasses.dataclass(eq=False)
class Zeros(Value):
    shape: Shape
    dtype: torch.dtype = torch.float32
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("zeros", self.shape)


@dataclasses.dataclass(eq=False)
class Pointwise(Value):
    op: str
    args: tuple[Value, ...]
    shape: Shape
    dtype: torch.dtype
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("pw", self.op, tuple(a.key for a in self.args))


@dataclasses.dataclass(eq=False)
class Compare(Value):
    op: str  # ge gt le lt eq ne
    lhs: Value
    rhs: Value
    shape: Shape
    dtype: torch.dtype = torch.bool
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("cmp", self.op, self.lhs.key, self.rhs.key)


@dataclasses.dataclass(eq=False)
class View(Value):
    """A ``subscript`` view: ``src_dims[d]`` is the source dim of output dim
    ``d`` (None for an inserted size-1 dim)."""

    x: Value
    src_dims: tuple[int | None, ...]
    shape: Shape
    dtype: torch.dtype
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("view", self.x.key, self.src_dims)


@dataclasses.dataclass(eq=False)
class Permute(Value):
    """The last two dims swapped."""

    x: Value
    shape: Shape
    dtype: torch.dtype
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("permute", self.x.key)


@dataclasses.dataclass(eq=False)
class Cast(Value):
    x: Value
    dtype: torch.dtype
    shape: Shape
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("cast", str(self.dtype), self.x.key)


@dataclasses.dataclass(eq=False)
class Dot(Value):
    a: Value
    b: Value
    shape: Shape
    dtype: torch.dtype = torch.float32
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("dot", self.a.key, self.b.key)


@dataclasses.dataclass(eq=False)
class RowReduce(Value):
    """``x.sum(dim=-1, keepdim=True)`` of an fp32 tile: each lane sums its
    quarter of the row in the A-operand lane layout of a dot over ``K =
    extent`` and the four lanes of the row exchange partials by butterfly
    shuffles (the result is read at the row's coordinate, last dim 0)."""

    op: str  # "sum"
    x: Value
    shape: Shape
    dtype: torch.dtype = torch.float32
    key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.key = ("rowreduce", self.op, self.x.key)


@dataclasses.dataclass(frozen=True)
class StoreOp:
    ref: TensorRef
    dims: tuple[Idx, ...]
    value_dims: tuple[int | None, ...]
    value: Value


@dataclasses.dataclass
class Item:
    """A maximal run of one graph instance between control-flow nodes."""

    index: int
    name: str
    stores: list[StoreOp] = dataclasses.field(default_factory=list)
    forced: list[Value] = dataclasses.field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.stores and not self.forced


@dataclasses.dataclass
class Program:
    root_graph_id: int
    batch_block_id: int
    batch_extent: int
    tile_block_id: int | None
    tile_extent: int | None
    items: list[Item]
    groups: list[list[int]]  # item indices per independent group, program order
    tensors: dict[str, TensorRef]
    stored_names: frozenset[str]
    # Outputs the first write of a window may store plainly (zero-filled,
    # compiler-allocated, every store of the CTA's own row and tile).
    zero_owned: frozenset[str]
    macs: int
    n_mode: str  # "permuted" | "tiled"
    has_static_mask: bool
    batch_symbol: sympy.Symbol | None
    scalar_exprs: tuple[sympy.Expr, ...]
    # Runs of consecutive groups whose items are isomorphic up to integer
    # constants affine in the group index, with those constants as
    # ``Affine`` group terms: one code path for every group of the run.
    merged: tuple[tuple[tuple[int, ...], list[Item]], ...] = ()

    @property
    def tile_choices(self) -> tuple[int, ...]:
        if self.tile_extent is None:
            return ()
        return tuple(t for t in TILE_CHOICES if self.tile_extent % t == 0)

    def work_items(self, bk: int | None) -> int:
        tiles = 1 if self.tile_extent is None else self.tile_extent // cast("int", bk)
        return self.batch_extent * tiles * len(self.groups)


def _idx_key(idx: Idx) -> tuple[object, ...]:
    if isinstance(idx, IdxBatch):
        return ("batch",)
    if isinstance(idx, IdxTile):
        return ("tile",)
    if isinstance(idx, IdxConst):
        return ("const", idx.value)
    if isinstance(idx, IdxWindow):
        return ("window", idx.start, idx.extent)
    if isinstance(idx, IdxFull):
        return ("full", idx.extent)
    if isinstance(idx, IdxScalar):
        return ("scalar", str(idx.expr))
    return ("wbase", idx.base.key, idx.start, idx.extent)


def _window_key(ref: TensorRef, dims: tuple[Idx, ...]) -> tuple[object, ...]:
    return (ref.name, tuple(_idx_key(d) for d in dims))


def _windows_overlap(a: tuple[object, ...], b: tuple[object, ...]) -> bool:
    if a[0] != b[0]:
        return False
    for da, db in zip(cast("tuple", a[1]), cast("tuple", b[1]), strict=True):
        da = cast("tuple", da)
        db = cast("tuple", db)
        ra = _static_range(da)
        rb = _static_range(db)
        if ra is None or rb is None:
            if da[0] in ("scalar", "wbase") or db[0] in ("scalar", "wbase"):
                if da != db:
                    # Two uniform offsets of unknown relation: assume overlap.
                    return True
            continue
        if ra[1] <= rb[0] or rb[1] <= ra[0]:
            return False
    return True


def _static_range(d: tuple[object, ...]) -> tuple[int, int] | None:
    """The static index range of a window dim key (None when it is not
    static: a scalar subscript, a runtime window base, or the group-affine
    bounds of a merged unit's window, which ``_windows_overlap`` then treats
    as overlapping)."""
    if d[0] == "const" and isinstance(d[1], int):
        return d[1], d[1] + 1
    if d[0] == "window" and isinstance(d[1], int) and isinstance(d[2], int):
        return d[1], d[1] + d[2]
    if d[0] == "full":
        return 0, cast("int", d[1])
    return None


# --------------------------------------------------------------------------
# Matching / flattening


def _is_call(node: object, target: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
    )


class _Matcher:
    """Flatten the device IR of a root into a ``Program`` (``_NotMine`` on
    anything the family does not lower)."""

    def __init__(self, graphs: Sequence[GraphInfo]) -> None:
        from ..device_ir import RootGraphInfo
        from ..host_function import HostFunction

        self.graphs = graphs
        self.env = CompileEnvironment.current()
        self.host = HostFunction.current()
        roots = [g for g in graphs if isinstance(g, RootGraphInfo)]
        if len(roots) != 1:
            raise _NotMine("the kernel must have exactly one root tile loop")
        self.root = roots[0]
        self.tensors: dict[str, TensorRef] = {}
        self.items: list[Item] = []
        self.loop_env: dict[int, int] = {}
        self.macs = 0
        self.has_static_mask = False
        self.scalar_exprs: list[sympy.Expr] = []
        self.batch_block_id = -1
        self.tile_block_id: int | None = None
        self.batch_symbol: sympy.Symbol | None = None
        self.b_k_contiguous = False
        self.current: Item | None = None

    # ---- root axes

    def analyze_root(self) -> None:
        env = self.env
        device_ir = self.host.device_ir
        if len(device_ir.grid_block_ids) != 1:
            raise _NotMine("the kernel must have exactly one root tile grid")
        batch: list[int] = []
        tiled: list[int] = []
        for block_id in device_ir.grid_block_ids[0]:
            info = env.block_sizes[block_id]
            source = info.block_size_source
            if isinstance(source, FixedBlockSizeSource) and source.value == 1:
                batch.append(block_id)
            elif isinstance(source, LoopSpecBlockSizeSource):
                tiled.append(block_id)
            else:
                raise _NotMine(
                    "a root axis must be a block_size=1 batch axis or one tiled axis"
                )
        if len(batch) != 1 or len(tiled) > 1:
            raise _NotMine(
                "the root must tile one block_size=1 batch axis and at most one channel axis"
            )
        self.batch_block_id = batch[0]
        extent = self.static_size(env.block_sizes[batch[0]].size)
        if extent is None:
            raise _NotMine("the batch extent must be static")
        self.batch_extent = extent
        self.tile_extent: int | None = None
        if tiled:
            self.tile_block_id = tiled[0]
            tile_extent = self.static_size(env.block_sizes[tiled[0]].size)
            if tile_extent is None or tile_extent % 16:
                raise _NotMine("the channel extent must be a static multiple of 16")
            self.tile_extent = tile_extent
            if not any(tile_extent % t == 0 for t in TILE_CHOICES):
                raise _NotMine(
                    f"no register tile of {TILE_CHOICES} divides {tile_extent}"
                )

    def block_of_symbol(self, sym: sympy.Symbol) -> tuple[int, str] | None:
        origin_info = self.host.expr_to_origin.get(sym)
        if origin_info is None:
            return None
        origin = origin_info.origin
        if isinstance(origin, BlockSizeOrigin):
            return origin.block_id, "size"
        if isinstance(origin, GridOrigin):
            return origin.block_id, "index"
        return None

    # ---- integer / symbolic evaluation

    def host_constant(self, sym: sympy.Symbol) -> int | None:
        """The value of a module global the kernel reads (Helion reads it
        when the kernel runs and specializes once, on the first call's
        value: kernel.py), or None for any other symbol -- an int argument
        or a closure cell (runtime inputs the bound kernel is keyed on by
        type), a dynamic shape -- whose size hint is one call's value."""
        info = self.host.expr_to_origin.get(sym)
        if info is None or not info.origin.is_global():
            return None
        hints = shape_env_var_hints(self.env.shape_env)
        if sym not in hints:
            return None
        return int(hints[sym])

    def static_size(self, value: object) -> int | None:
        """A size, stride or extent as an int: a literal, a specialized
        size or an expression over module globals; None otherwise."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        expr: sympy.Expr | None = None
        if isinstance(value, torch.SymInt):
            expr = cast("sympy.Expr", value._sympy_())
        elif isinstance(value, sympy.Expr):
            expr = value
        if expr is None:
            return None
        expr = self.env.specialize_expr(expr)
        subs: dict[sympy.Symbol, sympy.Integer] = {}
        for s in expr.free_symbols:
            assert isinstance(s, sympy.Symbol)
            constant = self.host_constant(s)
            if constant is None:
                return None
            subs[s] = sympy.Integer(constant)
        try:
            folded = sympy.sympify(expr.xreplace(subs))
        except Exception:
            return None
        if isinstance(folded, sympy.Integer):
            return int(folded)
        return None

    def sym(self, value: object) -> int | sympy.Expr:
        """An int-valued node or literal under the loop environment: an int,
        or a sympy expression over the batch index symbol."""
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        if isinstance(value, bool):
            raise _NotMine("a bool where an int was expected")
        if isinstance(value, int):
            return value
        if isinstance(value, (torch.SymInt, torch.SymBool)):
            expr = cast("sympy.Expr", value._sympy_())
        elif isinstance(value, sympy.Basic):
            expr = value
        else:
            raise _NotMine(f"not an integer value: {value!r}")
        subs: dict[sympy.Symbol, object] = {}
        kept: set[sympy.Symbol] = set()
        for s in expr.free_symbols:
            assert isinstance(s, sympy.Symbol)
            block = self.block_of_symbol(s)
            if block is not None:
                block_id, kind = block
                canonical = self.env.canonical_block_id(block_id)
                if canonical in self.loop_env and kind == "index":
                    subs[s] = sympy.Integer(self.loop_env[canonical])
                elif canonical == self.batch_block_id and kind == "index":
                    if self.batch_symbol is None:
                        self.batch_symbol = s
                    kept.add(s)
                    continue
                elif kind == "size" and canonical in (
                    self.batch_block_id,
                    self.tile_block_id,
                ):
                    # The tile object of a root axis: a subscript, not a value.
                    kept.add(s)
                    continue
                else:
                    raise _NotMine(f"symbol {s} of block {block_id} is not static")
            elif s in self.env.specialized_vars:
                subs[s] = sympy.Integer(shape_env_size_hint(self.env.shape_env, s))
            elif (constant := self.host_constant(s)) is not None:
                subs[s] = sympy.Integer(constant)
            else:
                # An int kernel argument or a dynamic shape: Helion keys the
                # bound kernel by its type, so its size hint is one call's
                # value, not a constant of the program.
                raise _NotMine(f"symbol {s} is a runtime value, not a static one")
        expr = cast("sympy.Expr", sympy.sympify(expr.xreplace(subs)))
        if isinstance(expr, sympy.logic.boolalg.BooleanAtom):
            return int(bool(expr))
        if isinstance(expr, sympy.Integer):
            return int(expr)
        if not expr.free_symbols <= kept:
            raise _NotMine(f"{expr} is not static")
        return expr

    def static_int(self, value: object) -> int:
        result = self.sym(value)
        if not isinstance(result, int):
            raise _NotMine(f"{result} must be static")
        return result

    def value_shape(self, fake: torch.Tensor) -> Shape:
        shape: list[int | str] = []
        for size in fake.shape:
            if isinstance(size, int):
                shape.append(size)
                continue
            block_id = self.env.get_block_id(size)
            if block_id is not None:
                canonical = self.env.canonical_block_id(block_id)
                if canonical == self.batch_block_id:
                    shape.append(1)
                    continue
                if canonical == self.tile_block_id:
                    shape.append(TILE)
                    continue
                info = self.env.block_sizes[canonical]
                if not info.reduction:
                    raise _NotMine(f"a value is sized by block {block_id}")
                # A full ``:`` slice contracted by a dot is a reduction block
                # of the dim's static extent.
                static = self.static_size(info.size)
                if static is None:
                    raise _NotMine(f"reduction block {block_id} has no static extent")
                shape.append(static)
                continue
            static = self.static_size(size)
            if static is None:
                raise _NotMine("every value extent must be static")
            shape.append(static)
        return tuple(shape)

    # ---- walking

    def run(self) -> Program:
        self.analyze_root()
        self.new_item(self.root.name)
        self.walk(self.root.graph, {}, self.root.name)
        self.flush_item()
        items = [item for item in self.items if not item.empty]
        if not any(item.stores for item in items):
            raise _NotMine("the root stores nothing")
        for index, item in enumerate(items):
            item.index = index
        # Loads built in a dropped (storeless) item belong, for the prefetch,
        # to the next kept item in program order (the loop prefix's row
        # operands are first used by the first pair of its row block).
        successor: dict[int, Item] = {}
        following: Item | None = None
        for item in reversed(self.items):
            if not item.empty:
                following = item
            elif following is not None:
                successor[id(item)] = following
        groups = self.group_items(items)
        stored = frozenset(s.ref.name for item in items for s in item.stores)
        self.check_item_hazards(items, stored)
        roots = [store.value for item in items for store in item.stores]
        roots.extend(value for item in items for value in item.forced)
        for value in _values_under(roots):
            if isinstance(value, Load) and id(value.origin) in successor:
                value.origin = successor[id(value.origin)]
        _rekey(roots, stored)
        zero_owned = self.prove_zero_owned(items, stored)
        return Program(
            root_graph_id=self.root.graph_id,
            batch_block_id=self.batch_block_id,
            batch_extent=self.batch_extent,
            tile_block_id=self.tile_block_id,
            tile_extent=self.tile_extent,
            items=items,
            groups=groups,
            tensors=self.tensors,
            stored_names=stored,
            zero_owned=zero_owned,
            macs=self.macs,
            n_mode="tiled" if self.b_k_contiguous else "permuted",
            has_static_mask=self.has_static_mask,
            batch_symbol=self.batch_symbol,
            scalar_exprs=tuple(self.scalar_exprs),
            merged=_merge_isomorphic_groups(items, groups, stored),
        )

    def new_item(self, name: str) -> None:
        self.current = Item(len(self.items), name)
        self.items.append(self.current)

    def flush_item(self) -> None:
        self.current = None

    def group_items(self, items: list[Item]) -> list[list[int]]:
        """Independent groups: items whose stores touch overlapping windows,
        or that read a window another item writes, share a group; items
        without stores join the next storing item (their forced values feed
        it)."""
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        windows = [[_window_key(s.ref, s.dims) for s in item.stores] for item in items]
        reads = [self.output_reads(item) for item in items]
        for i, j in itertools.combinations(range(len(items)), 2):
            conflict = any(
                _windows_overlap(a, b)
                for a in windows[i]
                for b in [*windows[j], *reads[j]]
            ) or any(_windows_overlap(a, b) for a in reads[i] for b in windows[j])
            if conflict:
                union(i, j)
        # Storeless items, and items whose loop trip forces carried values,
        # attach to the next storing item: the carried values feed the item
        # after the loop, and a pure value is rendered where it is consumed.
        last_store = None
        for i in reversed(range(len(items))):
            if items[i].forced and last_store is not None:
                union(i, last_store)
            if items[i].stores:
                last_store = i
            elif last_store is not None:
                union(i, last_store)
            else:
                # trailing storeless items: the previous storing item
                for j in reversed(range(i)):
                    if items[j].stores:
                        union(i, j)
                        break
        groups: dict[int, list[int]] = {}
        for i in range(len(items)):
            groups.setdefault(find(i), []).append(i)
        return [sorted(g) for g in sorted(groups.values(), key=lambda g: min(g))]

    def output_reads(self, item: Item) -> list[tuple[object, ...]]:
        """Windows of stored tensors the item's values load."""
        found: list[tuple[object, ...]] = []
        seen: set[int] = set()
        stored = {s.ref.name for it in self.items for s in it.stores}

        def visit(value: Value) -> None:
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Load):
                if value.ref.name in stored:
                    found.append(_window_key(value.ref, value.dims))
                if value.pred is not None:
                    visit(value.pred)
            for child in _children(value):
                visit(child)

        for store in item.stores:
            visit(store.value)
        for value in item.forced:
            visit(value)
        return found

    def prove_zero_owned(
        self, items: list[Item], stored: frozenset[str]
    ) -> frozenset[str]:
        owned: set[str] = set()
        for name in stored:
            ref = self.tensors[name]
            if not self.env.tensor_storage_is_zero_filled_allocation(ref.fake):
                continue
            if not self.env.tensor_storage_is_compiler_allocated(ref.fake):
                continue
            ok = True
            for item in items:
                for store in item.stores:
                    if store.ref.name != name:
                        continue
                    has_batch = any(
                        isinstance(d, IdxBatch)
                        or (
                            isinstance(d, IdxScalar)
                            and self.batch_symbol is not None
                            and d.expr == self.batch_symbol
                        )
                        for d in store.dims
                    )
                    has_tile = self.tile_block_id is None or any(
                        isinstance(d, IdxTile) for d in store.dims
                    )
                    static_rest = all(
                        isinstance(d, (IdxBatch, IdxTile, IdxConst, IdxWindow, IdxFull))
                        or (
                            isinstance(d, IdxScalar)
                            and self.batch_symbol is not None
                            and d.expr == self.batch_symbol
                        )
                        for d in store.dims
                    )
                    if not (has_batch and has_tile and static_rest):
                        ok = False
            if ok:
                owned.add(name)
        return frozenset(owned)

    @staticmethod
    def check_item_hazards(items: list[Item], stored: frozenset[str]) -> None:
        """Refuse the programs whose reads of stored windows the lazy,
        run-interleaved render cannot order: a load of an output observes
        the stores before its program position (``Load.position``), so a
        store (or carried value) using a load across a later write of an
        overlapping window of the same item, and a store reading its own
        window other than in place (the same window, coordinate for
        coordinate), fail closed.  Reads across items are ordered by the
        item schedule and checked at render time (``output_state``)."""
        for item in items:
            writes = [_window_key(s.ref, s.dims) for s in item.stores]
            for j, store in enumerate(item.stores):
                for load in _stored_loads(store.value, stored):
                    key = _window_key(load.ref, load.dims)
                    position = load.position if load.origin is item else 0
                    if any(
                        _windows_overlap(key, writes[t]) for t in range(position, j)
                    ):
                        raise _NotMine(
                            "a window read before a write of the same item is used after it"
                        )
                    if _windows_overlap(key, writes[j]) and not (
                        key == writes[j] and _reads_in_place(store.value, load)
                    ):
                        raise _NotMine("a store reads its own window in another layout")
            for value in item.forced:
                for load in _stored_loads(value, stored):
                    key = _window_key(load.ref, load.dims)
                    position = load.position if load.origin is item else 0
                    if any(
                        _windows_overlap(key, writes[t])
                        for t in range(position, len(writes))
                    ):
                        raise _NotMine(
                            "a carried value reads a window its trip writes after it"
                        )

    def walk(
        self, graph: torch.fx.Graph, placeholders: dict[Node, object], name: str
    ) -> list[object]:
        # (The language modules import the config spec, which imports this
        # module for its knob constants: resolve them at match time.)
        from ...language import _tracing_ops
        from ...language import memory_ops

        values: dict[Node, object] = {}
        outputs: list[object] = []
        for node in graph.nodes:
            if node.op == "placeholder":
                if node not in placeholders:
                    raise _NotMine("an unbound graph placeholder")
                values[node] = placeholders[node]
                continue
            if node.op == "output":
                out = node.args[0]
                if isinstance(out, (list, tuple)):
                    outputs = [self.resolve(a, values) for a in out]
                elif out is not None:
                    outputs = [self.resolve(out, values)]
                continue
            if node.op != "call_function":
                raise _NotMine(f"unsupported node {node.op}")
            if _is_call(node, _tracing_ops._for_loop):
                values[node] = self.flatten_loop(node, values, name)
                continue
            if _is_call(node, _tracing_ops._for_loop_step):
                raise _NotMine("stepped loops are not supported")
            if _is_call(node, _tracing_ops._if):
                self.flatten_if(node, values, name)
                values[node] = _Undefined()
                continue
            if _is_call(node, memory_ops.store):
                self.build_store(node, values)
                values[node] = None
                continue
            values[node] = self.build_value(node, values)
        return outputs

    def resolve(self, arg: object, values: dict[Node, object]) -> object:
        if isinstance(arg, torch.fx.Node):
            if arg not in values:
                raise _NotMine(f"use before definition of {arg.name}")
            return values[arg]
        return arg

    def flatten_loop(
        self, node: Node, values: dict[Node, object], name: str
    ) -> list[object]:
        from ..device_ir import ForLoopGraphInfo

        graph_id = cast("int", node.args[0])
        info = self.graphs[graph_id]
        if not isinstance(info, ForLoopGraphInfo) or len(info.block_ids) != 1:
            raise _NotMine("a device loop must be a one-axis for loop")
        begins = cast("Sequence[object]", node.args[1])
        ends = cast("Sequence[object]", node.args[2])
        if len(begins) != 1 or len(ends) != 1:
            raise _NotMine("a device loop must be a one-axis for loop")
        begin = self.static_int(begins[0])
        end = self.static_int(ends[0])
        args = [self.resolve(a, values) for a in cast("Sequence[object]", node.args[3])]
        block_id = self.env.canonical_block_id(info.block_ids[0])
        source = self.env.block_sizes[info.block_ids[0]].block_size_source
        if not (isinstance(source, FixedBlockSizeSource) and source.value == 1):
            raise _NotMine("a device loop must advance one index per trip")
        if end - begin > 64:
            raise _NotMine("a device loop of more than 64 trips is not flattened")
        placeholders_nodes = list(info.graph.find_nodes(op="placeholder"))
        if len(placeholders_nodes) != len(args):
            raise _NotMine("loop arguments do not match its placeholders")
        self.flush_item()
        # The loop returns its written values; the outer ``_phi(init,
        # getitem(loop, k))`` nodes say which carried argument each one
        # replaces on the next trip.
        positions = self.loop_output_positions(node)
        carried: list[object] = list(args)
        results: list[object] | None = None
        saved = self.loop_env.get(block_id)
        for trip in range(begin, end):
            self.loop_env[block_id] = trip
            self.new_item(f"{info.name}[{trip}]")
            bound = dict(zip(placeholders_nodes, carried, strict=True))
            outputs = self.walk(info.graph, bound, info.name)
            # Force the carried tiles at the end of each trip so each trip's
            # dots and epilogue stay in its own schedule item.
            item = self.current
            assert item is not None
            for out in outputs:
                if isinstance(out, Value) and _is_tile_value(out):
                    item.forced.append(out)
            self.flush_item()
            results = outputs
            for k, value in enumerate(outputs):
                position = positions.get(k)
                if position is not None:
                    carried[position] = value
        if saved is None:
            self.loop_env.pop(block_id, None)
        else:
            self.loop_env[block_id] = saved
        self.new_item(f"{name}+")
        if results is None:
            return [_Undefined() for _ in range(max(1, len(args)))]
        return results

    @staticmethod
    def loop_output_positions(node: Node) -> dict[int, int]:
        """Output index -> position among the loop's arguments of the value
        it replaces (read off the outer ``getitem`` / ``_phi`` users)."""
        from ...language import _tracing_ops

        inputs = list(cast("Sequence[object]", node.args[3]))
        if len(set(inputs)) != len(inputs):
            # ``inputs.index`` would rebind the first of two carries
            # initialized from one node.
            raise _NotMine(
                "a loop carrying one value in two positions is not flattened"
            )
        positions: dict[int, int] = {}
        for user in node.users:
            if user.target is not operator.getitem:
                continue
            k = cast("int", user.args[1])
            for phi in user.users:
                if _is_call(phi, _tracing_ops._phi) and phi.args[0] in inputs:
                    positions[k] = inputs.index(phi.args[0])
        return positions

    def flatten_if(self, node: Node, values: dict[Node, object], name: str) -> None:
        if len(node.users):
            raise _NotMine("a conditional whose results are used is not flattened")
        test = self.sym(node.args[0])
        if not isinstance(test, int):
            raise _NotMine("a conditional's test must be static")
        graph_id = cast("int", node.args[1] if test else node.args[2])
        args = cast("Sequence[object]", node.args[3] if test else node.args[4])
        info = self.graphs[graph_id]
        placeholders_nodes = list(info.graph.find_nodes(op="placeholder"))
        resolved = [self.resolve(a, values) for a in args]
        if len(placeholders_nodes) != len(resolved):
            raise _NotMine("branch arguments do not match its placeholders")
        self.flush_item()
        self.new_item(info.name)
        self.walk(
            info.graph, dict(zip(placeholders_nodes, resolved, strict=True)), info.name
        )
        self.flush_item()
        self.new_item(f"{name}+")

    # ---- values

    def tensor_ref(self, node: object) -> TensorRef:
        from .fx_matcher import _host_tensor_ref

        ref = _host_tensor_ref(node)
        if ref is None:
            raise _NotMine("loads and stores must address host tensors")
        fake = ref.fake
        if ref.name in self.tensors:
            return self.tensors[ref.name]
        strides = []
        shape = []
        for stride, size in zip(fake.stride(), fake.shape, strict=True):
            s = self.static_size(stride)
            n = self.static_size(size)
            if s is None or n is None:
                raise _NotMine("tensor shapes and strides must be static")
            strides.append(s)
            shape.append(n)
        result = TensorRef(ref.name, fake, tuple(strides), tuple(shape))
        self.tensors[ref.name] = result
        return result

    def index_specs(
        self,
        ref: TensorRef,
        indices: object,
        values: dict[Node, object],
        value_shape: Shape,
    ) -> tuple[tuple[Idx, ...], tuple[int | None, ...]]:
        if not isinstance(indices, (list, tuple)) or len(indices) != len(ref.shape):
            raise _NotMine("every tensor dim must be subscripted")
        dims: list[Idx] = []
        value_dims: list[int | None] = []
        next_dim = 0

        def padded(extent: int) -> int:
            if next_dim >= len(value_shape):
                raise _NotMine("subscript rank exceeds the value rank")
            size = value_shape[next_dim]
            if size == TILE:
                raise _NotMine("an arange window sized by the channel tile")
            size = cast("int", size)
            if size < extent:
                raise _NotMine("a window wider than its value")
            return size

        for position, entry in enumerate(indices):
            extent = ref.shape[position]
            if entry == slice(None):
                dims.append(IdxFull(extent))
                value_dims.append(next_dim)
                next_dim += 1
                continue
            if isinstance(entry, slice):
                # A literal ``start:stop`` (a slice over a loop index or an
                # ``hl.arange`` traces as an iota and takes the branch below).
                if entry.step not in (None, 1):
                    raise _NotMine("a stepped slice is not supported")
                start = 0 if entry.start is None else self.static_int(entry.start)
                stop = extent if entry.stop is None else self.static_int(entry.stop)
                if start < 0 or stop > extent or stop <= start:
                    raise _NotMine("a slice must lie inside its dim")
                dims.append(IdxWindow(start, stop - start, padded(stop - start)))
                value_dims.append(next_dim)
                next_dim += 1
                continue
            if isinstance(entry, bool):
                raise _NotMine("bool subscript")
            if isinstance(entry, int):
                dims.append(IdxConst(entry))
                value_dims.append(None)
                continue
            if not isinstance(entry, torch.fx.Node):
                raise _NotMine(f"unsupported subscript {entry!r}")
            val = entry.meta.get("val")
            if isinstance(val, torch.SymInt):
                block = None
                expr = cast("sympy.Expr", val._sympy_())
                if isinstance(expr, sympy.Symbol):
                    block = self.block_of_symbol(expr)
                if block is not None and block[1] == "size":
                    canonical = self.env.canonical_block_id(block[0])
                    if canonical == self.batch_block_id:
                        dims.append(IdxBatch())
                    elif canonical == self.tile_block_id:
                        dims.append(IdxTile())
                    else:
                        raise _NotMine(f"subscript by the tile of block {block[0]}")
                    value_dims.append(next_dim)
                    next_dim += 1
                    continue
                scalar = self.sym(val)
                if isinstance(scalar, int):
                    dims.append(IdxConst(scalar))
                else:
                    dims.append(IdxScalar(scalar))
                value_dims.append(None)
                continue
            if isinstance(val, int):
                dims.append(IdxConst(val))
                value_dims.append(None)
                continue
            value = self.resolve(entry, values)
            if isinstance(value, Iota):
                start = cast("int", value.start)
                iota_extent = cast("int", value.extent)
                if start < 0 or start + iota_extent > extent:
                    raise _NotMine("a window must lie inside its dim")
                dims.append(IdxWindow(start, iota_extent, padded(iota_extent)))
            elif isinstance(value, WindowIota):
                start = cast("int", value.start)
                iota_extent = cast("int", value.extent)
                dims.append(
                    IdxWindowBase(value.base, start, iota_extent, padded(iota_extent))
                )
            else:
                raise _NotMine(
                    "subscripts must be tiles, static scalars or arange windows"
                )
            value_dims.append(next_dim)
            next_dim += 1
        return tuple(dims), tuple(value_dims)

    def build_store(self, node: Node, values: dict[Node, object]) -> None:
        ref = self.tensor_ref(node.args[0])
        value = self.resolve(node.args[2], values)
        if not isinstance(value, Value):
            raise _NotMine("a store must store a tile value")
        dims, value_dims = self.index_specs(ref, node.args[1], values, value.shape)
        if len(node.args) > 3 and node.args[3] is not None:
            raise _NotMine("masked stores are not supported")
        if any(isinstance(d, IdxWindowBase) for d in dims):
            raise _NotMine("stores at a runtime window base are not supported")
        if value.ndim != sum(d is not None for d in value_dims):
            raise _NotMine("store value rank does not match its subscripts")
        if not _is_tile_value(value):
            raise _NotMine("a store must write a 16-row by 8-column aligned tile")
        if value.dtype != ref.dtype and not (
            isinstance(value, Cast) and value.dtype == ref.dtype
        ):
            raise _NotMine("store dtype mismatch")
        item = self.current
        assert item is not None
        item.stores.append(StoreOp(ref, dims, value_dims, value))

    def build_value(self, node: Node, values: dict[Node, object]) -> object:
        from ...language import _tracing_ops
        from ...language import creation_ops
        from ...language import matmul_ops
        from ...language import memory_ops
        from ...language import tile_ops
        from ...language import view_ops

        target = node.target
        args = node.args
        if _is_call(node, _tracing_ops._host_tensor):
            return self.tensor_ref(node)
        if (
            _is_call(node, _tracing_ops._get_symnode)
            or _is_call(node, torch.ops.aten.sym_size.int)
            or _is_call(node, tile_ops.tile_begin)
        ):
            val = node.meta.get("val")
            if isinstance(val, torch.SymFloat):
                # A float kernel argument / module constant: lane-uniform.
                expr = cast("sympy.Expr", val._sympy_())
                self.scalar_exprs.append(expr)
                return ScalarSym(expr, torch.float32)
            return self.sym(node)
        if target in (
            operator.mul,
            operator.add,
            operator.sub,
            operator.floordiv,
            operator.mod,
            operator.neg,
            operator.le,
            operator.lt,
            operator.ge,
            operator.gt,
            operator.eq,
            operator.ne,
            operator.truediv,
        ):
            val = node.meta.get("val")
            if isinstance(val, (int, torch.SymInt, torch.SymBool)):
                return self.sym(node)
            raise _NotMine(f"scalar operator {target} on a non-integer")
        if _is_call(node, _tracing_ops._new_var) or _is_call(
            node, _tracing_ops._mask_to
        ):
            if _is_call(node, _tracing_ops._mask_to) and args[1] not in (0, 0.0, False):
                raise _NotMine("_mask_to with a nonzero fill")
            return self.resolve(args[0], values)
        if _is_call(node, _tracing_ops._phi):
            rhs = self.resolve(args[1], values)
            if isinstance(rhs, _Undefined):
                return self.resolve(args[0], values)
            return rhs
        if target is operator.getitem:
            seq = self.resolve(args[0], values)
            if isinstance(seq, (list, tuple)):
                index = cast("int", args[1])
                if index >= len(seq):
                    return _Undefined()
                return seq[index]
            raise _NotMine("getitem of a non-sequence")
        if _is_call(node, torch.ops.prims.iota.default):
            extent = args[0]
            if type(extent) is not int or node.kwargs.get("step", 1) != 1:
                raise _NotMine("hl.arange must have a literal extent and unit step")
            start = self.static_int(node.kwargs.get("start", 0))
            return Iota(extent, start)
        if _is_call(node, memory_ops.load):
            return self.build_load(node, values)
        if _is_call(node, torch.ops.aten.scalar_tensor.default):
            return Const(float(cast("float", args[0])))
        if _is_call(node, creation_ops.full):
            fill = args[1]
            if fill not in (0, 0.0):
                raise _NotMine("only zero-filled hl.zeros tiles are supported")
            fake = node.meta.get("val")
            if not isinstance(fake, torch.Tensor) or fake.dtype != torch.float32:
                raise _NotMine("hl.zeros must be fp32")
            return Zeros(self.value_shape(fake))
        if _is_call(node, torch.ops.prims.convert_element_type.default):
            x = self.value(args[0], values)
            dtype = cast("torch.dtype", args[1])
            if dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise _NotMine(f"cast to {dtype} is not supported")
            return Cast(x, dtype, x.shape)
        if _is_call(node, view_ops.subscript):
            return self.build_view(node, values)
        if _is_call(node, torch.ops.aten.permute.default):
            x = self.value(args[0], values)
            perm = list(cast("Sequence[int]", args[1]))
            expected = [*range(x.ndim - 2), x.ndim - 1, x.ndim - 2]
            if perm != expected:
                raise _NotMine("only a transpose of the last two dims is supported")
            shape = (*x.shape[:-2], x.shape[-1], x.shape[-2])
            return Permute(x, shape, x.dtype)
        if _is_call(node, matmul_ops.dot):
            return self.build_dot(node, values)
        if _is_call(node, torch.ops.aten.sum.dim_IntList):
            return self.build_row_reduce(node, values)
        pointwise = {
            torch.ops.aten.add.Tensor: "add",
            torch.ops.aten.sub.Tensor: "sub",
            torch.ops.aten.mul.Tensor: "mul",
            torch.ops.aten.neg.default: "neg",
            torch.ops.aten.exp2.default: "exp2",
            torch.ops.aten.rsqrt.default: "rsqrt",
            torch.ops.aten.where.self: "where",
        }
        compares = {
            torch.ops.aten.ge.Tensor: "ge",
            torch.ops.aten.gt.Tensor: "gt",
            torch.ops.aten.le.Tensor: "le",
            torch.ops.aten.lt.Tensor: "lt",
            torch.ops.aten.eq.Tensor: "eq",
            torch.ops.aten.ne.Tensor: "ne",
        }
        if target in compares:
            args = self.lowered_args(node)
            lhs = self.value(args[0], values)
            rhs = self.value(args[1], values)
            fake = node.meta.get("val")
            assert isinstance(fake, torch.Tensor)
            if not (_coordinate_like(lhs) and _coordinate_like(rhs)):
                raise _NotMine("comparisons must be of arange coordinates and scalars")
            return Compare(compares[target], lhs, rhs, self.value_shape(fake))
        if target in pointwise:
            op = pointwise[target]
            fake = node.meta.get("val")
            if not isinstance(fake, torch.Tensor):
                raise _NotMine("pointwise op without a tensor result")
            operands = [self.value(a, values) for a in self.lowered_args(node)]
            if op == "add" and len(operands) == 2:
                # ``start + hl.arange(E)`` / ``base + hl.arange(E)`` windows
                for first, second in (operands, operands[::-1]):
                    if isinstance(first, Iota) and isinstance(second, Const):
                        if float(second.value) != int(second.value):
                            raise _NotMine("a non-integer arange offset")
                        return Iota(first.extent, first.start + int(second.value))
                    if isinstance(first, Iota) and isinstance(second, ScalarLoad):
                        return WindowIota(second, first.extent, first.start)
            if fake.dtype == torch.bool:
                raise _NotMine("boolean pointwise ops are not supported")
            if fake.dtype not in (torch.float32, torch.bfloat16):
                # The lanes compute in fp32; a bf16 result is rounded after
                # every op as torch rounds it (``_Emitter.rounded``), fp16
                # has no rounding helper.
                raise _NotMine(f"pointwise results of {fake.dtype} are not supported")
            if op == "where":
                self.has_static_mask = self.has_static_mask or isinstance(
                    operands[0], (Compare, View)
                )
            return Pointwise(op, tuple(operands), self.value_shape(fake), fake.dtype)
        raise _NotMine(f"unsupported op {target}")

    def value(self, arg: object, values: dict[Node, object]) -> Value:
        resolved = self.resolve(arg, values)
        if isinstance(resolved, Value):
            return resolved
        if isinstance(resolved, bool):
            raise _NotMine("bool operand")
        if isinstance(resolved, (int, float)):
            return Const(float(resolved))
        if isinstance(resolved, sympy.Expr):
            # An int expression over the batch index (a varlen head index)
            # or a lifted host int: lane-uniform Int32.
            return ScalarSym(resolved, torch.int32)
        if isinstance(arg, torch.fx.Node):
            val = arg.meta.get("val")
            if isinstance(val, torch.SymFloat):
                expr = cast("sympy.Expr", val._sympy_())
                self.scalar_exprs.append(expr)
                return ScalarSym(expr, torch.float32)
            if isinstance(val, torch.SymInt):
                return ScalarSym(cast("sympy.Expr", val._sympy_()), torch.int32)
        raise _NotMine(f"unsupported operand {resolved!r}")

    def build_load(self, node: Node, values: dict[Node, object]) -> Value:
        ref = self.tensor_ref(node.args[0])
        fake = node.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            raise _NotMine("load without a tensor result")
        dims, value_dims = self.index_specs(
            ref, node.args[1], values, self.value_shape(fake)
        )
        if len(node.args) > 3 and node.args[3] is not None:
            # An eviction policy is a hint; the register program ignores it.
            pass
        mask_node = node.args[2] if len(node.args) > 2 else None
        pred: Value | None = None
        if mask_node is not None:
            mask = self.resolve(mask_node, values)
            if not isinstance(mask, Value) or mask.dtype != torch.bool:
                raise _NotMine("a load mask must be a boolean tile")
            pred = mask
        if fake.ndim == 0:
            if pred is not None:
                raise _NotMine("masked scalar loads are not supported")
            if ref.dtype not in (torch.int32, torch.int64):
                raise _NotMine("scalar loads must be int32 / int64")
            return ScalarLoad(ref, dims, ref.dtype)
        if ref.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise _NotMine(f"loads of {ref.dtype} are not supported")
        shape = self.value_shape(fake)
        if len(shape) != sum(d is not None for d in value_dims):
            raise _NotMine("load rank does not match its subscripts")
        item = self.current
        return Load(
            ref,
            dims,
            value_dims,
            pred,
            shape,
            ref.dtype,
            origin=item,
            position=0 if item is None else len(item.stores),
        )

    @staticmethod
    def padding(
        dims: tuple[Idx, ...],
        value_dims: tuple[int | None, ...],
        coord: Coord,
        trips: int | Affine = 1,
    ) -> str:
        """Whether a fragment coordinate lies in the padded rows of a window
        (``masked``), in its real rows (``valid``), or straddles (``mixed``);
        ``trips`` is the trip count of the rolled loop its ``t`` term runs."""
        result = "valid"
        for d, vd in zip(dims, value_dims, strict=True):
            if vd is None or not isinstance(d, (IdxWindow, IdxWindowBase)):
                continue
            if d.padded and coord[vd].ct == 0:
                # a group-parametric (or padded) window read outside the
                # rolled loop that covers exactly its real columns
                return "mixed"
            if d.size == d.extent:
                continue
            if not isinstance(trips, int) or not isinstance(d.extent, int):
                return "mixed"
            aff = coord[vd]
            span = aff.ct * (trips - 1)
            lo = aff.k + min(0, 7 * aff.cg) + min(0, 3 * aff.cc) + min(0, span)
            hi = aff.k + max(0, 7 * aff.cg) + max(0, 3 * aff.cc) + max(0, span)
            if lo >= d.extent:
                return "masked"
            if hi >= d.extent:
                result = "mixed"
        return result

    def build_view(self, node: Node, values: dict[Node, object]) -> Value:
        x = self.value(node.args[0], values)
        index = node.args[1]
        if not isinstance(index, (list, tuple)):
            raise _NotMine("subscript view index")
        src_dims: list[int | None] = []
        src = 0
        for entry in index:
            if entry is None:
                src_dims.append(None)
            elif entry == slice(None):
                if src >= x.ndim:
                    raise _NotMine("subscript view rank")
                src_dims.append(src)
                src += 1
            else:
                raise _NotMine("only None / full-slice subscript views are supported")
        while src < x.ndim:
            src_dims.append(src)
            src += 1
        fake = node.meta.get("val")
        assert isinstance(fake, torch.Tensor)
        return View(x, tuple(src_dims), self.value_shape(fake), x.dtype)

    def build_dot(self, node: Node, values: dict[Node, object]) -> Value:
        from .mma_support import cute_fp32_dot_uses_tf32

        if len(node.args) > 2 and node.args[2] is not None:
            raise _NotMine("hl.dot with an accumulator is not supported")
        if len(node.args) > 3 and node.args[3] not in (None, torch.float32):
            raise _NotMine("hl.dot must accumulate in fp32")
        a = self.value(node.args[0], values)
        b = self.value(node.args[1], values)
        fake = node.meta.get("val")
        if not isinstance(fake, torch.Tensor) or fake.dtype != torch.float32:
            raise _NotMine("hl.dot must produce fp32")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise _NotMine("hl.dot operands must be fp32")
        if not cute_fp32_dot_uses_tf32():
            raise _NotMine("the dot precision must map to tf32")
        capability = self.env.config_spec.target_device_capability
        if capability is None or capability[0] < MIN_CAPABILITY:
            raise _NotMine("mma.sync tf32 needs sm_80 or newer")
        if a.ndim < 2 or b.ndim < 2 or a.ndim != b.ndim:
            raise _NotMine("hl.dot operands must be matrices")
        m, k = a.shape[-2], a.shape[-1]
        kb, n = b.shape[-2], b.shape[-1]
        if k != kb:
            raise _NotMine("hl.dot contraction mismatch")
        if not isinstance(m, int) or m % 16:
            raise _NotMine("hl.dot rows must be a static multiple of 16")
        if not isinstance(k, int) or k % 8:
            raise _NotMine("hl.dot contraction must be a static multiple of 8")
        if isinstance(n, int) and n % 8:
            raise _NotMine("hl.dot columns must be a multiple of 8")
        for lead in (*a.shape[:-2], *b.shape[:-2]):
            if lead != 1:
                raise _NotMine("hl.dot batch dims must be the CTA's row")
        n_extent = self.tile_extent if n == TILE else cast("int", n)
        assert n_extent is not None
        self.macs += m * k * n_extent
        if _contiguous_dim(b) == 0:
            self.b_k_contiguous = True
        return Dot(a, b, (*a.shape[:-2], m, n))

    @staticmethod
    def lowered_args(node: Node) -> tuple[object, ...]:
        """The operands of a lowered pointwise node.  The inductor lowering
        (``strip_unused_inputs``) replaces a repeated input by None after
        its first occurrence (``x * x`` is ``mul(x, None)``): that None is the
        one Node the other args name; with several candidates (a ``where``
        of two tensors) the operand is ambiguous and the program declines."""
        args = node.args
        if not any(a is None for a in args):
            return args
        if "lowering" not in node.meta:
            raise _NotMine(f"a None operand of {node.target}")
        nodes = {a for a in args if isinstance(a, torch.fx.Node)}
        if len(nodes) != 1:
            raise _NotMine(f"ambiguous repeated operand of {node.target}")
        (only,) = nodes
        return tuple(only if a is None else a for a in args)

    def build_row_reduce(self, node: Node, values: dict[Node, object]) -> Value:
        """``x.sum(dim=-1, keepdim=True)`` of an fp32 tile value over a static
        extent: a ``RowReduce`` (every other reduction declines)."""
        args = node.args
        x = self.value(args[0], values)
        dims = args[1] if len(args) > 1 else node.kwargs.get("dim")
        keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
        if node.kwargs.get("dtype") is not None:
            raise _NotMine("a sum with a dtype is not supported")
        fake = node.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            raise _NotMine("sum without a tensor result")
        if isinstance(dims, int):
            dims = [dims]
        if (
            not isinstance(dims, (list, tuple))
            or len(dims) != 1
            or x.ndim == 0
            or cast("int", dims[0]) % x.ndim != x.ndim - 1
        ):
            raise _NotMine("only a sum over the last dim is supported")
        if not keepdim:
            raise _NotMine("a sum must keep its reduced dim (keepdim=True)")
        if x.dtype != torch.float32 or fake.dtype != torch.float32:
            raise _NotMine("a sum must be of fp32 values")
        if x.ndim < 2 or any(lead != 1 for lead in x.shape[:-2]):
            raise _NotMine("a sum must be of a 2-D tile value")
        extent = x.shape[-1]
        if not isinstance(extent, int):
            raise _NotMine("a reduced extent must be static (not the channel tile)")
        if extent % 4 or extent > MAX_REDUCE_EXTENT:
            raise _NotMine(
                f"a reduced extent of {extent} (a multiple of 4 up to "
                f"{MAX_REDUCE_EXTENT} is supported)"
            )
        return RowReduce("sum", x, self.value_shape(fake), torch.float32)


class _Undefined:
    """A loop result with no trips (``_phi`` takes its other operand)."""


def _coordinate_like(value: Value) -> bool:
    """An operand a static comparison may have: an ``hl.arange`` coordinate
    (possibly broadcast by a view), a uniform scalar or a constant."""
    while isinstance(value, View):
        value = value.x
    return isinstance(value, (Iota, WindowIota, ScalarLoad, ScalarSym, Const))


def _children(value: Value) -> tuple[Value, ...]:
    if isinstance(value, Pointwise):
        return value.args
    if isinstance(value, Compare):
        return (value.lhs, value.rhs)
    if isinstance(value, (View, Permute, Cast)):
        return (value.x,)
    if isinstance(value, Dot):
        return (value.a, value.b)
    if isinstance(value, RowReduce):
        return (value.x,)
    if isinstance(value, WindowIota):
        return (value.base,)
    if isinstance(value, Load):
        bases = tuple(d.base for d in value.dims if isinstance(d, IdxWindowBase))
        return bases + ((value.pred,) if value.pred is not None else ())
    return ()


def _values_under(roots: Sequence[Value]) -> list[Value]:
    """Every value reachable from ``roots``, children before parents."""
    order: list[Value] = []
    seen: set[int] = set()

    def visit(value: Value) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        for child in _children(value):
            visit(child)
        order.append(value)

    for root in roots:
        visit(root)
    return order


def _rekey(roots: Sequence[Value], stored: frozenset[str]) -> None:
    """Recompute every structural key bottom-up.  A load of a stored tensor
    keys on its own instance: its value is what the program wrote so far, a
    point in program order rather than a structural identity, and every
    value built on it inherits that distinction (the render memo of one
    item's ``out + x`` must not serve the next item's)."""
    for value in _values_under(roots):
        value.__post_init__()
        if isinstance(value, Load) and value.ref.name in stored:
            value.key = (*value.key, id(value))


def _stored_loads(value: Value, stored: frozenset[str]) -> list[Load]:
    """The loads of stored tensors under ``value``."""
    return [
        v
        for v in _values_under([value])
        if isinstance(v, Load) and v.ref.name in stored
    ]


def _reads_in_place(value: Value, load: Load) -> bool:
    """Whether every use of ``load`` under ``value`` keeps its coordinates
    (same-shape pointwise operands, casts, identity views), so a store of
    ``value`` reads, at each coordinate, exactly the element it writes
    there -- by the lane that writes it.  A permute, a dot operand, a
    broadcast or a predicate reads other lanes' elements."""
    holds: dict[int, bool] = {}

    def contains(v: Value) -> bool:
        if v is load:
            return True
        hit = holds.get(id(v))
        if hit is None:
            hit = any(contains(c) for c in _children(v))
            holds[id(v)] = hit
        return hit

    in_place: dict[int, bool] = {}

    def visit(v: Value) -> bool:
        if v is load:
            return True
        hit = in_place.get(id(v))
        if hit is None:
            if isinstance(v, Pointwise):
                hit = all(
                    not contains(a) or (a.shape == v.shape and visit(a)) for a in v.args
                )
            elif isinstance(v, Cast):
                hit = visit(v.x)
            elif isinstance(v, View):
                hit = (
                    v.src_dims == tuple(range(v.x.ndim))
                    and v.shape == v.x.shape
                    and visit(v.x)
                )
            else:
                hit = False
            in_place[id(v)] = hit
        return hit

    return not contains(value) or visit(value)


def _store_waves(item: Item, stored: frozenset[str]) -> list[list[tuple[int, StoreOp]]]:
    """The item's stores in waves: a store that writes a window an earlier
    store reads or writes, or reads one it writes, follows every run of
    that store; the stores of one wave are independent and their runs may
    interleave."""
    writes = [_window_key(s.ref, s.dims) for s in item.stores]
    reads = [
        [_window_key(load.ref, load.dims) for load in _stored_loads(s.value, stored)]
        for s in item.stores
    ]
    wave_of: list[int] = []
    for j in range(len(item.stores)):
        wave = 0
        for i in range(j):
            if (
                _windows_overlap(writes[i], writes[j])
                or any(_windows_overlap(r, writes[j]) for r in reads[i])
                or any(_windows_overlap(r, writes[i]) for r in reads[j])
            ):
                wave = max(wave, wave_of[i] + 1)
        wave_of.append(wave)
    waves: list[list[tuple[int, StoreOp]]] = [
        [] for _ in range(max(wave_of, default=-1) + 1)
    ]
    for j, store in enumerate(item.stores):
        waves[wave_of[j]].append((j, store))
    return waves


def _round_bf16(value: float) -> float:
    """``float(bfloat16(value))``: round to nearest even on the fp32 bits."""
    if not math.isfinite(value):
        return value
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _is_tile_value(value: Value) -> bool:
    if value.ndim < 2:
        return False
    m, n = value.shape[-2], value.shape[-1]
    if not isinstance(m, int) or m % 16:
        return False
    if isinstance(n, int) and n % 8:
        return False
    return all(lead == 1 for lead in value.shape[:-2])


def _contiguous_dim(value: Value) -> int | None:
    """Which of the last two dims is memory-contiguous (0 rows, 1 cols)."""
    if isinstance(value, Load):
        if value.ndim < 2:
            return None
        row_dim, col_dim = value.ndim - 2, value.ndim - 1
        for tensor_dim, vd in enumerate(value.value_dims):
            if vd == col_dim and value.ref.strides[tensor_dim] == 1:
                return 1
            if vd == row_dim and value.ref.strides[tensor_dim] == 1:
                return 0
        return None
    if isinstance(value, Permute):
        inner = _contiguous_dim(value.x)
        return None if inner is None else 1 - inner
    if isinstance(value, (View, Cast)):
        return _contiguous_dim(value.x) if value.x.ndim >= 2 else None
    if isinstance(value, Pointwise):
        for arg in value.args:
            if arg.ndim >= 2:
                found = _contiguous_dim(arg)
                if found is not None:
                    return found
        return None
    if isinstance(value, Dot):
        return 1
    return None


class _Constants:
    """Walk an item's value DAG with every integer constant abstracted:
    collects the constants (in a canonical order) and the structural
    signature, or rebuilds the DAG with the constants substituted."""

    def __init__(self, subs: list[int | Affine] | None = None) -> None:
        # (kind, value): ``"hard"`` constants must be affine in the group
        # index; ``"shape"`` extents (a value's padded dims) may instead take
        # the widest group's value, since the rolled render walks the real
        # columns of the windows and never reads a shape past them.
        self.consts: list[tuple[str, int]] = []
        self.subs = subs
        self.memo: dict[int, Value] = {}
        # shared values are referred to by their visit ordinal (object ids
        # differ between the groups being compared)
        self.ordinals: dict[int, int] = {}
        self.position = 0

    def const(self, value: int, kind: str = "hard") -> int | Affine:
        self.consts.append((kind, value))
        if self.subs is None:
            return value
        result = self.subs[self.position]
        self.position += 1
        return result

    def shape(self, shape: Shape) -> Shape:
        return tuple(self.const(d, "shape") if isinstance(d, int) else d for d in shape)

    def idx(self, d: Idx) -> tuple[object, Idx]:
        if isinstance(d, IdxConst):
            assert isinstance(d.value, int)
            return "C", IdxConst(self.const(d.value))
        if isinstance(d, IdxWindow):
            assert isinstance(d.start, int) and isinstance(d.extent, int)
            start, extent = self.const(d.start), self.const(d.extent)
            # The padded size is not a constant of the walk: a rolled loop
            # over the real columns never reaches the padding (``padded``
            # refuses any other read of the window).
            padded = d.size != d.extent or isinstance(extent, Affine)
            return ("W",), IdxWindow(start, extent, extent, padded)
        if isinstance(d, IdxWindowBase):
            assert isinstance(d.start, int) and isinstance(d.extent, int)
            sig, base = self.value(d.base)
            start, extent = self.const(d.start), self.const(d.extent)
            padded = d.size != d.extent or isinstance(extent, Affine)
            return ("WB", sig), IdxWindowBase(base, start, extent, extent, padded)
        if isinstance(d, IdxFull):
            return ("F", d.extent), d
        if isinstance(d, IdxScalar):
            return ("S", str(d.expr)), d
        return (type(d).__name__,), d

    def value(self, v: Value) -> tuple[object, Value]:
        hit = self.memo.get(id(v))
        if hit is not None:
            return ("ref", self.ordinals[id(v)]), hit
        self.ordinals[id(v)] = len(self.ordinals)
        if isinstance(v, Load):
            dims = [self.idx(d) for d in v.dims]
            pred = self.value(v.pred) if v.pred is not None else None
            shape = self.shape(v.shape)
            out: Value = Load(
                v.ref,
                tuple(d for _s, d in dims),
                v.value_dims,
                pred[1] if pred else None,
                shape,
                v.dtype,
                origin=v.origin,
                position=v.position,
            )
            sig: object = (
                "load",
                v.ref.name,
                tuple(s for s, _d in dims),
                pred[0] if pred else None,
                v.value_dims,
            )
        elif isinstance(v, ScalarLoad):
            dims = [self.idx(d) for d in v.dims]
            out = ScalarLoad(v.ref, tuple(d for _s, d in dims), v.dtype)
            sig = ("sload", v.ref.name, tuple(s for s, _d in dims))
        elif isinstance(v, Iota):
            assert isinstance(v.extent, int) and isinstance(v.start, int)
            out = Iota(self.const(v.extent), self.const(v.start))
            sig = ("iota",)
        elif isinstance(v, WindowIota):
            assert isinstance(v.extent, int) and isinstance(v.start, int)
            bsig, base = self.value(v.base)
            out = WindowIota(base, self.const(v.extent), self.const(v.start))
            sig = ("wiota", bsig)
        elif isinstance(v, Const):
            out, sig = v, ("const", v.value)
        elif isinstance(v, ScalarSym):
            out, sig = v, ("sym", str(v.expr), str(v.dtype))
        elif isinstance(v, Zeros):
            out = Zeros(self.shape(v.shape), v.dtype)
            sig = ("zeros", str(v.dtype))
        elif isinstance(v, Pointwise):
            parts = [self.value(a) for a in v.args]
            out = Pointwise(
                v.op, tuple(a for _s, a in parts), self.shape(v.shape), v.dtype
            )
            sig = ("pw", v.op, tuple(s for s, _a in parts), str(v.dtype))
        elif isinstance(v, Compare):
            lsig, lhs = self.value(v.lhs)
            rsig, rhs = self.value(v.rhs)
            out = Compare(v.op, lhs, rhs, self.shape(v.shape))
            sig = ("cmp", v.op, lsig, rsig)
        elif isinstance(v, View):
            xsig, x = self.value(v.x)
            out = View(x, v.src_dims, self.shape(v.shape), v.dtype)
            sig = ("view", xsig, v.src_dims)
        elif isinstance(v, Permute):
            xsig, x = self.value(v.x)
            out = Permute(x, self.shape(v.shape), v.dtype)
            sig = ("permute", xsig)
        elif isinstance(v, Cast):
            xsig, x = self.value(v.x)
            out = Cast(x, v.dtype, self.shape(v.shape))
            sig = ("cast", str(v.dtype), xsig)
        elif isinstance(v, Dot):
            asig, a = self.value(v.a)
            bsig, b = self.value(v.b)
            out = Dot(a, b, self.shape(v.shape))
            sig = ("dot", asig, bsig)
        elif isinstance(v, RowReduce):
            xsig, x = self.value(v.x)
            out = RowReduce(v.op, x, self.shape(v.shape), v.dtype)
            sig = ("rowreduce", v.op, xsig)
        else:
            raise _NotMine(f"cannot abstract {type(v).__name__}")
        self.memo[id(v)] = out
        return sig, out

    def item(self, item: Item) -> tuple[object, Item]:
        stores = []
        sigs = []
        for store in item.stores:
            dims = [self.idx(d) for d in store.dims]
            vsig, value = self.value(store.value)
            stores.append(
                StoreOp(store.ref, tuple(d for _s, d in dims), store.value_dims, value)
            )
            sigs.append(
                (store.ref.name, tuple(s for s, _d in dims), store.value_dims, vsig)
            )
        forced = [self.value(v) for v in item.forced]
        out = Item(item.index, item.name, stores, [v for _s, v in forced])
        return (tuple(sigs), tuple(s for s, _v in forced)), out


def _merge_isomorphic_groups(
    items: list[Item], groups: list[list[int]], stored: frozenset[str]
) -> tuple[tuple[tuple[int, ...], list[Item]], ...]:
    """Runs of consecutive groups whose items are the same program up to
    integer constants affine in the group index, as parametric items."""
    if len(groups) < 2:
        return ()
    signatures: list[tuple[object, list[tuple[str, int]]]] = []
    for group in groups:
        walker = _Constants()
        try:
            sig = tuple(walker.item(items[i])[0] for i in group)
        except _NotMine:
            return ()
        signatures.append((sig, walker.consts))
    merged: list[tuple[tuple[int, ...], list[Item]]] = []
    start = 0
    while start < len(groups):
        end = start + 1
        while end < len(groups) and signatures[end][0] == signatures[start][0]:
            end += 1
        run = list(range(start, end))
        if len(run) >= 2:
            columns = list(zip(*(signatures[g][1] for g in run), strict=True))
            subs: list[int | Affine] = []
            ok = True
            for column in columns:
                kinds = {kind for kind, _v in column}
                values = [v for _k, v in column]
                a = values[0]
                b = values[1] - values[0]
                if any(values[j] != a + b * j for j in range(len(run))):
                    if kinds == {"shape"}:
                        subs.append(max(values))
                        continue
                    ok = False
                    break
                # constants are affine in the group index itself (the run
                # starts at group ``start``): v(grp) = a - b*start + b*grp
                subs.append(Affine(k=a - b * start, cgp=b) if b else a)
            if ok:
                walker = _Constants(subs)
                rebuilt = [walker.item(items[i])[1] for i in groups[start]]
                # (the rebuilt loads of stored tensors key on their own
                # instances like the originals)
                roots = [store.value for item in rebuilt for store in item.stores]
                roots.extend(value for item in rebuilt for value in item.forced)
                _rekey(roots, stored)
                merged.append((tuple(run), rebuilt))
        start = end
    return tuple(merged)


def match_register_subblocks(
    graphs: Sequence[GraphInfo],
) -> tuple[Program | None, str | None]:
    """The flattened program of the device IR, or the reason it is not one."""
    try:
        return _Matcher(graphs).run(), None
    except _NotMine as error:
        return None, str(error)


# --------------------------------------------------------------------------
# Facts (config-free, picklable) and config validation


@dataclasses.dataclass(frozen=True)
class RegisterSubblockFacts:
    root_graph_id: int
    tile_block_id: int | None
    tile_extent: int | None
    tile_choices: tuple[int, ...]
    batch_extent: int
    groups: int
    items: int
    macs: int
    has_static_mask: bool
    multi_write_windows: bool
    prefetch_subject: bool
    # Some item's n-tile walk can be a rolled loop (tiled N mode).
    roll_subject: bool
    # (tile or 0, prefetch, colside, roll) -> estimated fp32 values live per
    # thread, and -> why the render fails (None: renders)
    estimates: dict[tuple[int, int, str, str], int]
    failures: dict[tuple[int, int, str, str], str | None]

    @property
    def seed_worthy(self) -> bool:
        return self.macs >= SEED_MACS

    def work_items(self, bk: int | None) -> int:
        tiles = 1 if self.tile_extent is None else self.tile_extent // cast("int", bk)
        return self.batch_extent * tiles * self.groups

    def tile_key(self, bk: int | None) -> int:
        return 0 if self.tile_extent is None else cast("int", bk)


def register_subblock_facts(program: Program) -> RegisterSubblockFacts:
    """The config-free facts of a matched program, including the register
    estimate of every (tile, prefetch, colside) corner (a dry render)."""
    tiles: tuple[int | None, ...] = program.tile_choices or (None,)
    estimates: dict[tuple[int, int, str, str], int] = {}
    failures: dict[tuple[int, int, str, str], str | None] = {}
    windows: dict[tuple[object, ...], int] = {}
    for item in program.items:
        for store in item.stores:
            key = _window_key(store.ref, store.dims)
            windows[key] = windows.get(key, 0) + 1
    multi = any(count > 1 for count in windows.values())
    items_with_loads = sum(1 for item in program.items if item.stores or item.forced)
    roll_subject = program.n_mode == "tiled" and any(
        _Emitter.rolled_trips_of(program, item, windows) is not None
        for item in program.items
    )
    rolls = ROLL_CHOICES if roll_subject else (ROLL_UNROLLED,)
    for bk in tiles:
        for prefetch in PREFETCH_CHOICES:
            for colside in COLSIDE_CHOICES:
                if colside == COLSIDE_REGS and not multi:
                    continue
                for roll in rolls:
                    key = (0 if bk is None else bk, prefetch, colside, roll)
                    try:
                        emitter = _Emitter(
                            program,
                            bk=bk,
                            warps=1,
                            prefetch=prefetch,
                            mask=MASK_SELECT,
                            colside=colside,
                            roll=roll,
                            names=_DryNames(),
                        )
                        emitter.render()
                    except exc.BackendUnsupported as error:
                        failures[key] = str(error)
                        estimates[key] = 1 << 30
                    else:
                        failures[key] = None
                        estimates[key] = emitter.max_live
    return RegisterSubblockFacts(
        root_graph_id=program.root_graph_id,
        tile_block_id=program.tile_block_id,
        tile_extent=program.tile_extent,
        tile_choices=program.tile_choices,
        batch_extent=program.batch_extent,
        groups=len(program.groups),
        items=len(program.items),
        macs=program.macs,
        has_static_mask=program.has_static_mask,
        multi_write_windows=multi,
        prefetch_subject=items_with_loads > 1,
        roll_subject=roll_subject,
        estimates=estimates,
        failures=failures,
    )


def register_subblocks_config_reason(
    facts: RegisterSubblockFacts,
    *,
    bk: object,
    warps: object,
    prefetch: object,
    mask: object,
    colside: object,
    roll: object = ROLL_UNROLLED,
) -> str | None:
    """Why the knobs cannot run the family (None: they can)."""
    if facts.tile_extent is not None:
        if (
            not isinstance(bk, int)
            or isinstance(bk, bool)
            or bk not in facts.tile_choices
        ):
            return f"the channel tile must be one of {facts.tile_choices}, got {bk!r}"
    else:
        bk = None
    if warps not in WARP_CHOICES or type(warps) is not int:
        return f"{WARPS_KEY} must be one of {WARP_CHOICES}, got {warps!r}"
    if facts.work_items(bk) % warps:
        return f"{warps} warps do not divide the {facts.work_items(bk)} work items"
    if prefetch not in PREFETCH_CHOICES or type(prefetch) is not int:
        return f"{PREFETCH_KEY} must be one of {PREFETCH_CHOICES}, got {prefetch!r}"
    if mask not in MASK_CHOICES:
        return f"{MASK_KEY} must be one of {MASK_CHOICES}, got {mask!r}"
    if colside not in COLSIDE_CHOICES:
        return f"{COLSIDE_KEY} must be one of {COLSIDE_CHOICES}, got {colside!r}"
    if colside == COLSIDE_REGS and not facts.multi_write_windows:
        return "register accumulation has no window written more than once"
    if roll not in ROLL_CHOICES:
        return f"{ROLL_KEY} must be one of {ROLL_CHOICES}, got {roll!r}"
    if roll == ROLL_LOOP and not facts.roll_subject:
        return "no n-tile walk of the program can be rolled"
    key = (
        facts.tile_key(bk),
        cast("int", prefetch),
        cast("str", colside),
        cast("str", roll),
    )
    failure = facts.failures.get(key)
    if failure is not None:
        return failure
    estimate = facts.estimates.get(key)
    if estimate is None:
        return f"no register estimate for {key}"
    if estimate > REGISTER_BUDGET:
        return (
            f"tile {bk} with prefetch {prefetch} and {colside} accumulation holds "
            f"~{estimate} live values per thread (budget {REGISTER_BUDGET})"
        )
    return None


def register_subblocks_repair(
    facts: RegisterSubblockFacts,
    *,
    bk: object,
    warps: object,
    prefetch: object,
    mask: object,
    colside: object,
    roll: object = ROLL_UNROLLED,
) -> dict[str, object] | None:
    """The nearest legal knobs (prefetch first, then register accumulation,
    then the tile, then the warps), or None when no corner is legal."""
    tiles: tuple[int | None, ...] = facts.tile_choices or (None,)
    tile_order: list[int | None] = []
    if facts.tile_extent is not None and bk in tiles:
        tile_order.append(cast("int", bk))
    tile_order.extend(
        t for t in sorted(tiles, key=lambda t: -(t or 0)) if t not in tile_order
    )
    warp_order = [warps] if warps in WARP_CHOICES else []
    warp_order.extend(w for w in WARP_CHOICES if w not in warp_order)
    mask_value = mask if mask in MASK_CHOICES else MASK_SELECT
    pf_order = [prefetch] if prefetch in PREFETCH_CHOICES else []
    pf_order.extend(p for p in PREFETCH_CHOICES if p not in pf_order)
    cs_order = [colside] if colside in COLSIDE_CHOICES else []
    cs_order.extend(c for c in COLSIDE_CHOICES if c not in cs_order)
    roll_order = [roll] if roll in ROLL_CHOICES else []
    roll_order.extend(r for r in ROLL_CHOICES if r not in roll_order)
    for tile in tile_order:
        for w in warp_order:
            for cs in cs_order:
                for pf in pf_order:
                    for rl in roll_order:
                        if (
                            register_subblocks_config_reason(
                                facts,
                                bk=tile,
                                warps=w,
                                prefetch=pf,
                                mask=mask_value,
                                colside=cs,
                                roll=rl,
                            )
                            is None
                        ):
                            return {
                                "bk": tile,
                                WARPS_KEY: w,
                                PREFETCH_KEY: pf,
                                MASK_KEY: mask_value,
                                COLSIDE_KEY: cs,
                                ROLL_KEY: rl,
                            }
    return None


def register_subblocks_seed_knobs(
    facts: RegisterSubblockFacts,
) -> list[dict[str, object]]:
    """Ranked seed knobs (``bk`` is the channel tile or None): the measured
    winners where legal, then the legal corners by register estimate."""
    seeds: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    def add(
        tile: int | None,
        prefetch: int,
        warps: int,
        colside: str = COLSIDE_RMW,
        roll: str = ROLL_UNROLLED,
    ) -> None:
        if facts.tile_extent is None:
            tile = None
        reason = register_subblocks_config_reason(
            facts,
            bk=tile,
            warps=warps,
            prefetch=prefetch,
            mask=MASK_SELECT,
            colside=colside,
            roll=roll,
        )
        key = (tile, prefetch, warps, colside, roll)
        if reason is None and key not in seen:
            seen.add(key)
            seeds.append(
                {
                    "bk": tile,
                    WARPS_KEY: warps,
                    PREFETCH_KEY: prefetch,
                    MASK_KEY: MASK_SELECT,
                    COLSIDE_KEY: colside,
                    ROLL_KEY: roll,
                }
            )

    if facts.roll_subject:
        # A rolled n-tile walk (the forward gram: one short code path per
        # warp, 58 us against the unrolled form's 135-160 and the fragment
        # route's 105 on B200): one warp per work item first, then the CTA
        # of all the independent groups of one row (the four tiers).
        add(None, 0, 1, COLSIDE_RMW, ROLL_LOOP)
        if facts.groups in WARP_CHOICES:
            add(None, 0, facts.groups, COLSIDE_RMW, ROLL_LOOP)
        add(None, 0, 2, COLSIDE_RMW, ROLL_LOOP)
    for tile, prefetch, warps, colside, roll in SEEDS:
        add(tile, prefetch, warps, colside, roll)
        if colside == COLSIDE_REGS and not facts.multi_write_windows:
            add(tile, prefetch, warps, COLSIDE_RMW, roll)
    if facts.tile_extent is None:
        add(None, 0, 1)
        if facts.groups in WARP_CHOICES:
            add(None, 0, facts.groups)
        add(None, 0, 2)
    tiles: tuple[int | None, ...] = facts.tile_choices or (None,)
    rolls = ROLL_CHOICES if facts.roll_subject else (ROLL_UNROLLED,)
    corners = sorted(
        (
            (facts.estimates.get((facts.tile_key(t), p, c, r), 1 << 30), t, p, c, r)
            for t in tiles
            for p in PREFETCH_CHOICES
            for c in COLSIDE_CHOICES
            for r in rolls
        ),
        key=lambda x: (x[0], x[3] != COLSIDE_RMW, x[4] != ROLL_UNROLLED),
    )
    for _estimate, tile, prefetch, colside, roll in corners:
        add(tile, prefetch, 1, colside, roll)
        if len(seeds) >= 8:
            break
    return seeds


# --------------------------------------------------------------------------
# Plan


@dataclasses.dataclass
class RegisterSubblockPlan:
    program: Program
    bk: int | None
    warps: int
    prefetch: int
    mask: str
    colside: str
    roll: str = ROLL_UNROLLED

    @property
    def threads(self) -> int:
        return 32 * self.warps

    @property
    def total_ctas(self) -> int:
        return self.program.work_items(self.bk) // self.warps


def plan_register_subblocks(
    graphs: Sequence[GraphInfo], config: object
) -> RegisterSubblockPlan | None:
    """The plan of the root when the config selects the family (None: off);
    a config that selects it on a kernel it does not lower fails closed."""
    cfg = cast("dict[str, object]", config)
    mode = cfg.get(KEY, MODE_OFF)  # pyrefly: ignore[missing-attribute]
    if mode != MODE_ON:
        return None
    program, reason = match_register_subblocks(graphs)
    if program is None:
        raise _decline(cast("str", reason))
    env = CompileEnvironment.current()
    bk: int | None = None
    if program.tile_block_id is not None:
        value = env.block_sizes[program.tile_block_id].from_config(config)  # pyrefly: ignore[bad-argument-type]
        if not isinstance(value, int):
            raise _decline("the channel tile must be static")
        bk = value
    facts = env.config_spec.cute_register_subblocks_facts
    if facts is None:
        facts = register_subblock_facts(program)
    knobs = {
        "bk": bk,
        "warps": cfg.get(WARPS_KEY, DEFAULT_WARPS),  # pyrefly: ignore[missing-attribute]
        "prefetch": cfg.get(PREFETCH_KEY, DEFAULT_PREFETCH),  # pyrefly: ignore[missing-attribute]
        "mask": cfg.get(MASK_KEY, MASK_SELECT),  # pyrefly: ignore[missing-attribute]
        "colside": cfg.get(COLSIDE_KEY, COLSIDE_RMW),  # pyrefly: ignore[missing-attribute]
        "roll": cfg.get(ROLL_KEY, ROLL_UNROLLED),  # pyrefly: ignore[missing-attribute]
    }
    reason = register_subblocks_config_reason(facts, **knobs)  # pyrefly: ignore[bad-argument-type]
    if reason is not None:
        raise _decline(reason)
    return RegisterSubblockPlan(
        program,
        bk,
        cast("int", knobs["warps"]),
        cast("int", knobs["prefetch"]),
        cast("str", knobs["mask"]),
        cast("str", knobs["colside"]),
        cast("str", knobs["roll"]),
    )


# --------------------------------------------------------------------------
# Emitter


class _Names:
    """Device names of the program's tensors and host scalars."""

    def tensor(self, ref: TensorRef) -> str:
        raise NotImplementedError

    def scalar(self, sym: sympy.Symbol) -> str:
        raise NotImplementedError


class _DryNames(_Names):
    def tensor(self, ref: TensorRef) -> str:
        return f"T_{ref.name}"

    def scalar(self, sym: sympy.Symbol) -> str:
        return f"S_{sym.name}"


class _DeviceNames(_Names):
    def __init__(self, cg: GenerateAST) -> None:
        self.df = cg.device_function

    def tensor(self, ref: TensorRef) -> str:
        name = self.df.tensor_arg(ref.fake, prefer_name=ref.name).name
        self.df.placeholder_args.add(name)
        return name

    def scalar(self, sym: sympy.Symbol) -> str:
        return self.df.sympy_expr(sym)


@dataclasses.dataclass(frozen=True)
class Reg:
    name: str


@dataclasses.dataclass(frozen=True)
class ConstF:
    value: float


@dataclasses.dataclass(frozen=True)
class IntAff:
    aff: Affine


@dataclasses.dataclass(frozen=True)
class IntReg:
    """A lane-uniform integer register (a scalar load)."""

    name: str


@dataclasses.dataclass(frozen=True)
class Pred:
    op: str
    lhs: str  # Int32 expressions
    rhs: str


@dataclasses.dataclass(frozen=True)
class ConstB:
    value: bool


Operand = Reg | ConstF | IntAff | IntReg | Pred | ConstB

_SEL_OP = {"ge": "ge", "gt": "gt", "le": "le", "lt": "lt", "eq": "eq", "ne": "ne"}
_PY_CMP = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="}
_VAR_RE = re.compile(r"\bv\d+\b")


@dataclasses.dataclass
class _WindowState:
    writes_total: int
    writes_done: int = 0
    regs: dict[Coord, Operand] | None = None
    # the C-layout coordinates its stores wrote (a read-back must stay
    # within them: the lane reads what it stored)
    coords: set[Coord] = dataclasses.field(default_factory=set)

    def copy(self) -> _WindowState:
        return _WindowState(
            self.writes_total,
            self.writes_done,
            None if self.regs is None else dict(self.regs),
            set(self.coords),
        )


@dataclasses.dataclass
class _DotLayout:
    m_tiles: int
    k: int
    n_tiles: int
    c_coords: dict[tuple[int, int, int], Coord]
    tile_of: dict[Coord, tuple[int, int]]
    done: set[tuple[int, int]] = dataclasses.field(default_factory=set)
    operands_issued: bool = False


class _Emitter:
    """Render one group's items as straight-line DSL text.

    Two passes: the first records, per item, the input loads its sinks pull
    in (and the writes of every output window); the second replays item
    ``t + prefetch``'s input loads at the start of item ``t`` and knows which
    write of a window is its last.  ``max_live`` is the liveness estimate of
    the rendered values (the register-budget heuristic of the knobs).
    """

    def __init__(
        self,
        program: Program,
        *,
        bk: int | None,
        warps: int,
        prefetch: int,
        mask: str,
        colside: str,
        names: _Names,
        roll: str = ROLL_UNROLLED,
    ) -> None:
        self.program = program
        self.bk = bk
        self.warps = warps
        self.prefetch = prefetch
        self.mask = mask
        self.colside = colside
        self.roll = roll
        self.names = names
        # The rolled loop being rendered: its trip count, and (first pass)
        # the loop-invariant requests to hoist ahead of it.
        self.loop_trips: int | Affine | None = None
        self.hoist_record: list[tuple[object, ...]] | None = None
        self.unit_groups: tuple[int, ...] = (0,)
        self.fast_math = (
            CompileEnvironment.has_current()
            and CompileEnvironment.current().settings.fast_math
        )
        self.max_live = 0
        self.lines: list[str] = []
        self.counter = 0
        self.indent = ""
        # per-group state (reset by ``render_group``)
        self.memo: dict[tuple[tuple[object, ...], Coord], Operand] = {}
        self.bits_memo: dict[str, str] = {}
        self.base_vars: dict[tuple[object, ...], str] = {}
        self.scalar_regs: dict[str, str] = {}
        self.dots: dict[tuple[object, ...], _DotLayout] = {}
        self.windows: dict[tuple[object, ...], _WindowState] = {}
        self.recording: list[
            list[tuple[Load, tuple[Coord, ...], tuple[Operand | None, ...] | None]]
        ] = []
        self.item_pos = 0
        self.replay: (
            list[
                list[tuple[Load, tuple[Coord, ...], tuple[Operand | None, ...] | None]]
            ]
            | None
        ) = None
        self.write_totals: dict[tuple[object, ...], int] = {}
        self.pass_index = 0

    # ---- text

    def new_var(self) -> str:
        self.counter += 1
        return f"v{self.counter}"

    def emit(self, text: str) -> None:
        self.lines.append(self.indent + text)

    def size(self, dim: int | str | Affine) -> int | Affine:
        if dim == TILE:
            assert self.bk is not None
            return self.bk
        return cast("int | Affine", dim)

    # ---- rendering entry

    def render(self) -> str:
        program = self.program
        self.lines = []
        self.counter = 0
        head: list[str] = [
            "rsb_tidx, _, _ = cute.arch.thread_idx()",
            "rsb_warp = rsb_tidx // 32",
            "rsb_lane = rsb_tidx % 32",
            "rsb_g = rsb_lane // 4",
            "rsb_c = rsb_lane % 4",
            "rsb_g64 = cutlass.Int64(rsb_g)",
            "rsb_c64 = cutlass.Int64(rsb_c)",
            "rsb_bid = cutlass.Int32(cute.arch.block_idx()[0])",
            f"rsb_gw = rsb_bid * {self.warps} + rsb_warp",
        ]
        groups = len(program.groups)
        if groups > 1:
            head.extend([f"rsb_grp = rsb_gw % {groups}", f"rsb_t = rsb_gw // {groups}"])
        else:
            head.append("rsb_t = rsb_gw")
        if program.tile_extent is not None:
            assert self.bk is not None
            kt = program.tile_extent // self.bk
            head.extend(
                [
                    f"rsb_kt = rsb_t % {kt}",
                    f"rsb_bhn = rsb_t // {kt}",
                    f"rsb_col0 = rsb_kt * {self.bk}",
                ]
            )
        else:
            head.append("rsb_bhn = rsb_t")
        if groups > 1:
            head.append("rsb_grp64 = cutlass.Int64(rsb_grp)")
        self.lines.extend(head)
        # host scalars once per CTA
        for expr in dict.fromkeys(program.scalar_exprs):
            self.scalar_reg(expr, torch.float32)
        max_live = 0
        units = self.render_units()
        for index, (unit_groups, items) in enumerate(units):
            self.unit_groups = unit_groups
            if len(units) > 1:
                if index == 0:
                    self.emit(f"if rsb_grp <= {unit_groups[-1]}:")
                elif index < len(units) - 1:
                    self.emit(f"elif rsb_grp <= {unit_groups[-1]}:")
                else:
                    self.emit("else:")
                self.indent = "    "
            start = len(self.lines)
            self.render_group(items)
            max_live = max(max_live, self.liveness(self.lines[start:]))
            self.indent = ""
        self.max_live = max_live
        return "\n".join(self.lines) + "\n"

    def render_units(self) -> list[tuple[tuple[int, ...], list[Item]]]:
        """The code paths: each group its own, except the runs of isomorphic
        groups a rolled render covers with one parametric body."""
        program = self.program
        merged = dict(program.merged) if self.roll == ROLL_LOOP else {}
        covered = {g for run in merged for g in run}
        units: list[tuple[tuple[int, ...], list[Item]]] = []
        for index, group in enumerate(program.groups):
            if index in covered:
                run = next(run for run in merged if index in run)
                if index == run[0]:
                    units.append((run, merged[run]))
                continue
            units.append(((index,), [program.items[i] for i in group]))
        return units

    def max_int(self, value: int | str | Affine) -> int:
        """A static extent, or a group-affine one at the unit's last group."""
        if isinstance(value, Affine):
            folded = value.at_group(max(self.unit_groups))
            assert folded.is_const
            return folded.k
        assert isinstance(value, int)
        return value

    def liveness(self, lines: list[str]) -> int:
        live: set[str] = set()
        best = 0
        for line in reversed(lines):
            text = line.strip()
            defs: set[str] = set()
            uses: set[str] = set()
            if " = " in text and not text.startswith(("if ", "elif ")):
                lhs, rhs = text.split(" = ", 1)
                defs = set(_VAR_RE.findall(lhs))
                uses = set(_VAR_RE.findall(rhs))
            else:
                uses = set(_VAR_RE.findall(text))
            live -= defs
            live |= uses
            best = max(best, len(live))
        return best

    def render_group(self, items: list[Item]) -> None:
        # pass 1: record
        self.reset_group_state()
        self.pass_index = 1
        self.recording = [[] for _ in items]
        self.replay = None
        saved_lines = list(self.lines)
        saved_counter = self.counter
        saved_scalars = dict(self.scalar_regs)
        for position, item in enumerate(items):
            self.item_pos = position
            self.render_item(item)
        totals = {key: state.writes_done for key, state in self.windows.items()}
        recorded = self.recording
        # pass 2: replay with prefetch and last-write knowledge
        self.lines = saved_lines
        self.counter = saved_counter
        self.scalar_regs = saved_scalars
        self.reset_group_state()
        self.pass_index = 2
        self.write_totals = totals
        self.replay = recorded
        self.recording = [[] for _ in items]
        for position, item in enumerate(items):
            self.item_pos = position
            if self.prefetch:
                for ahead in range(1, self.prefetch + 1):
                    if position + ahead < len(items):
                        target = items[position + ahead]
                        for load, coords, preds in recorded[position + ahead]:
                            if (
                                load.ref.name in self.program.stored_names
                                or load.origin is not target
                                or any(a.ct for c in coords for a in c)
                            ):
                                continue
                            self.emit_load(load, list(coords), preds)
            self.render_item(item)
        for state in self.windows.values():
            if state.regs is not None and state.writes_done != state.writes_total:
                raise _decline("an accumulated window was never flushed")

    def reset_group_state(self) -> None:
        self.memo = {}
        self.bits_memo = {}
        self.base_vars = {}
        self.dots = {}
        self.windows = {}

    def render_item(self, item: Item) -> None:
        trips = self.rolled_trips(item)
        if trips is not None:
            self.render_rolled_item(item, trips)
            return
        # Stores first, in waves (a store that writes a window another store
        # of the item reads or writes follows every run of that store), the
        # runs of a wave interleaved by column so the dots feeding several
        # stores stream one n-tile at a time; then the carried values the
        # loop forces at the end of its trip.
        for wave in _store_waves(item, self.program.stored_names):
            runs: list[tuple[tuple[int, int, int], StoreOp, list[Coord], bool]] = []
            for si, store in wave:
                store_runs = self.store_runs(store)
                for position, (sort_key, coords) in enumerate(store_runs):
                    last = position == len(store_runs) - 1
                    runs.append(((sort_key[0], sort_key[1], si), store, coords, last))
            runs.sort(key=operator.itemgetter(0))
            for _key, store, coords, last in runs:
                self.emit_store_run(store, coords, last)
        for value in item.forced:
            coords = self.c_layout_coords(value)
            self.eval_many(value, coords)

    # ---- rolled n-tile loops

    @staticmethod
    def rolled_trips_of(
        program: Program, item: Item, write_counts: dict[tuple[object, ...], int]
    ) -> int | Affine | None:
        """The trip count of the item's n-tile loop when its stores can be
        rendered as one rolled walk: tiled N mode, every store a tile of the
        same width written once in the program, nothing else to force."""
        if program.n_mode != "tiled" or not item.stores or item.forced:
            return None
        trips: int | Affine | None = None
        for store in item.stores:
            n = store.value.shape[-1]
            if isinstance(n, int) and n % 8:
                return None
            if not isinstance(n, (int, Affine)):
                return None
            valid = _valid_extent(store.value, store.value.ndim - 1)
            if valid is None:
                for d, vd in zip(store.dims, store.value_dims, strict=True):
                    if vd == store.value.ndim - 1 and isinstance(
                        d, (IdxWindow, IdxWindowBase)
                    ):
                        valid = d.extent
            extent = valid if valid is not None else n
            count = extent // 8
            if isinstance(count, int) and count < 2:
                return None
            if trips is not None and count != trips:
                return None
            trips = count
            if write_counts.get(_window_key(store.ref, store.dims), 1) > 1:
                return None
        return trips

    def rolled_trips(self, item: Item) -> int | Affine | None:
        if self.roll != ROLL_LOOP:
            return None
        counts = self.write_totals if self.pass_index == 2 else {}
        trips = self.rolled_trips_of(self.program, item, counts)
        if (
            trips is None
            and self.pass_index == 2
            and self.rolled_trips_of(self.program, item, {}) is not None
        ):
            raise _decline("a rolled n-tile walk needs windows written once")
        return trips

    def render_rolled_item(self, item: Item, trips: int | Affine) -> None:
        """One rolled walk over the item's n-tiles: the loop-invariant values
        (operands, anchors, address bases) are rendered ahead of a
        ``cutlass.range`` loop whose body computes one n-tile of every
        store (a first pass over the body records what to hoist)."""
        self.loop_trips = trips
        snapshot = (
            list(self.lines),
            self.counter,
            dict(self.memo),
            dict(self.bits_memo),
            dict(self.base_vars),
            {k: dataclasses.replace(v, done=set(v.done)) for k, v in self.dots.items()},
            {k: v.copy() for k, v in self.windows.items()},
            dict(self.scalar_regs),
        )
        self.hoist_record = []
        self.render_rolled_body(item)
        record = self.hoist_record
        self.hoist_record = None
        (
            self.lines,
            self.counter,
            self.memo,
            self.bits_memo,
            self.base_vars,
            self.dots,
            self.windows,
            self.scalar_regs,
        ) = snapshot
        for entry in record:
            kind = entry[0]
            if kind == "eval":
                self.eval_many(
                    cast("Value", entry[1]), list(cast("tuple[Coord, ...]", entry[2]))
                )
            elif kind == "abits":
                self.issue_a_operand(cast("Dot", entry[1]))
            elif kind == "base":
                self.base_var(
                    cast("TensorRef", entry[1]),
                    cast("tuple[Idx, ...]", entry[2]),
                    cast("tuple[int | None, ...]", entry[3]),
                    None,
                )
        outer = self.indent
        bound = trips if isinstance(trips, int) else trips.render("rsb_g", "rsb_c")
        self.emit(f"for rsb_nt in cutlass.range({bound}):")
        self.indent = outer + "    "
        self.emit("rsb_nt64 = cutlass.Int64(rsb_nt)")
        self.render_rolled_body(item)
        self.indent = outer
        # the loop's values are per trip: forget them past the loop
        self.memo = {
            key: op for key, op in self.memo.items() if not any(a.ct for a in key[1])
        }
        for layout in self.dots.values():
            layout.done = {t for t in layout.done if t[1] != SYM_TILE}
        self.loop_trips = None

    def render_rolled_body(self, item: Item) -> None:
        for store in item.stores:
            value = store.value
            m, _n = self.dims_mn(value)
            lead = (ZERO,) * (value.ndim - 2)
            for mi in range(m // 16):
                for h in range(2):
                    row = Affine(cg=1, k=16 * mi + 8 * h)
                    coords = [
                        _coord(lead, row, Affine(cc=2, k=e, ct=8)) for e in range(2)
                    ]
                    last = mi == m // 16 - 1 and h == 1
                    self.emit_store_run(store, coords, last)

    # ---- layouts

    def dims_mn(self, value: Value) -> tuple[int, int]:
        """Rows and (at the unit's widest group) columns of a tile value."""
        m = self.max_int(self.size(value.shape[-2]))
        n = self.max_int(self.size(value.shape[-1]))
        return m, n

    def c_layout_coords(self, value: Value) -> list[Coord]:
        m, n = self.dims_mn(value)
        lead = (ZERO,) * (value.ndim - 2)
        nt = n // 8
        coords: list[Coord] = []
        for mi in range(m // 16):
            for h in range(2):
                row = Affine(cg=1, k=16 * mi + 8 * h)
                for mm in range(2 * nt):
                    coords.append(_coord(lead, row, self.c_col(nt, mm)))
        return coords

    def c_col(self, nt: int, mm: int) -> Affine:
        if self.program.n_mode == "permuted":
            return Affine(cc=2 * nt, k=mm)
        tile, e = divmod(mm, 2)
        return Affine(cc=2, k=8 * tile + e)

    def c_m_index(self, nt: int, tile: int, e: int) -> int:
        """The C-layout column slot ``mm`` of accumulator element ``e`` of n-tile ``tile``."""
        if self.program.n_mode == "permuted":
            return nt * e + tile
        return 2 * tile + e

    def b_col(self, nt: int, tile: int) -> Affine:
        if self.program.n_mode == "permuted":
            return Affine(cg=nt, k=tile)
        return Affine(cg=1, k=8 * tile)

    # ---- stores

    def store_runs(self, store: StoreOp) -> list[tuple[tuple[int, int], list[Coord]]]:
        """C-layout coordinate runs of the store, keyed for interleaving by
        (column slot, row half)."""
        value = store.value
        m, n = self.dims_mn(value)
        nt = n // 8
        lead = (ZERO,) * (value.ndim - 2)
        runs: list[tuple[tuple[int, int], list[Coord]]] = []
        for mi in range(m // 16):
            for h in range(2):
                row = Affine(cg=1, k=16 * mi + 8 * h)
                if self.program.n_mode == "permuted":
                    coords = [
                        _coord(lead, row, self.c_col(nt, mm)) for mm in range(2 * nt)
                    ]
                    runs.append(((0, 16 * mi + 8 * h), coords))
                else:
                    for tile in range(nt):
                        coords = [
                            _coord(lead, row, self.c_col(nt, 2 * tile + e))
                            for e in range(2)
                        ]
                        runs.append(((8 * tile, 16 * mi + 8 * h), coords))
        return runs

    def emit_store_run(
        self, store: StoreOp, coords: list[Coord], last_run: bool
    ) -> None:
        value = store.value
        if isinstance(value, Cast) and value.dtype == store.ref.dtype:
            value = value.x
        ops = self.eval_many(value, coords)
        key = _window_key(store.ref, store.dims)
        state = self.windows.get(key)
        if state is None:
            total = (
                self.write_totals.get(key, 1 << 20) if self.pass_index == 2 else 1 << 20
            )
            state = _WindowState(total)
            self.windows[key] = state
        state.coords.update(coords)
        # A store's writes are counted once, at its last run (its own runs
        # read the window as it was before the store).
        if self.colside == COLSIDE_REGS:
            # Register accumulation: the window lives in registers from its
            # first write to its last (the recorded count), then flushes.
            if state.regs is None:
                state.regs = {}
            for coord, op in zip(coords, ops, strict=True):
                state.regs[coord] = op
            if last_run:
                state.writes_done += 1
                if self.pass_index == 2 and state.writes_done == state.writes_total:
                    all_coords = list(state.regs)
                    self.emit_memory_store(
                        store, all_coords, [state.regs[c] for c in all_coords]
                    )
            return
        self.emit_memory_store(store, coords, ops)
        if last_run:
            state.writes_done += 1

    def emit_memory_store(
        self, store: StoreOp, coords: list[Coord], ops: list[Operand]
    ) -> None:
        ref = store.ref
        es = ref.dtype.itemsize
        entries: list[tuple[Affine, Operand]] = []
        for coord, op in zip(coords, ops, strict=True):
            state = _Matcher.padding(
                store.dims, store.value_dims, coord, self.loop_trips or 1
            )
            if state == "masked":
                continue  # the padded columns of the window are not written
            if state == "mixed":
                raise _decline("a padded window must end on a fragment boundary")
            aff = self.byte_affine(ref, store.dims, store.value_dims, coord)
            entries.append((aff, op))
        if not entries:
            return
        base, align = self.base_var(ref, store.dims, store.value_dims, None)
        for group in self.group_runs(entries, es):
            self.emit_store_vectors(ref, base, align, group, es)

    def emit_store_vectors(
        self,
        ref: TensorRef,
        base: str,
        align: int,
        run: list[tuple[Affine, Operand]],
        es: int,
    ) -> None:
        dtype = ref.dtype
        position = 0
        while position < len(run):
            aff0 = run[position][0]
            remaining = len(run) - position
            width = self.vector_width(aff0, es, align, remaining, max_bytes=16)
            chunk = run[position : position + width]
            address = f"{base} + {aff0.render('rsb_g64', 'rsb_c64')}"
            values = [self.f32(op) for _aff, op in chunk]
            if dtype == torch.float32:
                if width == 4:
                    self.emit(f"{_RT}.stg_v4_f32({address}, {', '.join(values)})")
                elif width == 2:
                    self.emit(f"{_RT}.stg_v2_f32({address}, {', '.join(values)})")
                else:
                    self.emit(f"{_RT}.stg_f32({address}, {values[0]})")
            else:
                pack = "pack_bf16x2" if dtype == torch.bfloat16 else "pack_f16x2"
                scalar = "stg_bf16" if dtype == torch.bfloat16 else "stg_f16"
                if width == 1:
                    self.emit(f"{_RT}.{scalar}({address}, {values[0]})")
                else:
                    words = []
                    for i in range(0, width, 2):
                        w = self.new_var()
                        self.emit(f"{w} = {_RT}.{pack}({values[i]}, {values[i + 1]})")
                        words.append(w)
                    if len(words) == 1:
                        self.emit(f"{_RT}.stg_b32({address}, {words[0]})")
                    elif len(words) == 2:
                        self.emit(f"{_RT}.stg_v2_b32({address}, {', '.join(words)})")
                    else:
                        self.emit(f"{_RT}.stg_v4_b32({address}, {', '.join(words)})")
            position += width

    # ---- addresses

    def base_var(
        self,
        ref: TensorRef,
        dims: tuple[Idx, ...],
        value_dims: tuple[int | None, ...],
        pred: object,
    ) -> tuple[str, int]:
        """The lane-uniform byte base of an access (tensor pointer, batch row,
        channel tile, scalar subscripts) and its proven alignment."""
        signature: list[object] = []
        for d in dims:
            if isinstance(d, (IdxBatch, IdxTile, IdxScalar)):
                signature.append(_idx_key(d))
            elif isinstance(d, IdxWindowBase):
                signature.append(("wbase", d.base.key))
            else:
                signature.append(None)
        key = (ref.name, tuple(signature))
        if self.hoist_record is not None and key not in self.base_vars:
            self.hoist_record.append(("base", ref, dims, value_dims))
        align = 16 if self.base_aligned(ref) else ref.dtype.itemsize
        es = ref.dtype.itemsize
        terms = [f"{self.names.tensor(ref)}.iterator.toint()"]
        for d, stride in zip(dims, ref.strides, strict=True):
            sb = stride * es
            if isinstance(d, IdxBatch):
                terms.append(f"cutlass.Int64(rsb_bhn) * {sb}")
                align = math.gcd(align, sb)
            elif isinstance(d, IdxTile):
                assert self.bk is not None
                terms.append(f"cutlass.Int64(rsb_col0) * {sb}")
                align = math.gcd(align, sb * self.bk)
            elif isinstance(d, IdxScalar):
                terms.append(
                    f"cutlass.Int64({self.render_scalar_index(d.expr)}) * {sb}"
                )
                align = math.gcd(align, sb)
            elif isinstance(d, IdxWindowBase):
                reg = self.scalar_load(cast("ScalarLoad", d.base))
                terms.append(f"cutlass.Int64({reg}) * {sb}")
                align = math.gcd(align, sb)
        if key in self.base_vars:
            return self.base_vars[key], align
        name = self.new_var()
        self.emit(f"{name} = " + " + ".join(terms))
        self.base_vars[key] = name
        return name, align

    def base_aligned(self, ref: TensorRef) -> bool:
        if not CompileEnvironment.has_current():
            return True
        from .memory_ops import cute_tensor_base_is_aligned

        return cute_tensor_base_is_aligned(CompileEnvironment.current(), ref.fake, 16)

    def byte_affine(
        self,
        ref: TensorRef,
        dims: tuple[Idx, ...],
        value_dims: tuple[int | None, ...],
        coord: Coord,
    ) -> Affine:
        es = ref.dtype.itemsize
        total = ZERO
        for d, stride, vd in zip(dims, ref.strides, value_dims, strict=True):
            sb = stride * es
            if isinstance(d, IdxConst):
                total = total + (
                    d.value.scale(sb) if isinstance(d.value, Affine) else sb * d.value
                )
            elif isinstance(d, (IdxWindow, IdxWindowBase)):
                assert vd is not None
                total = total + (coord[vd] + d.start).scale(sb)
            elif isinstance(d, (IdxFull, IdxTile)):
                assert vd is not None
                total = total + coord[vd].scale(sb)
        return total

    def render_scalar_index(self, expr: sympy.Expr) -> str:
        from .printer import cute_texpr

        subs: dict[sympy.Symbol, sympy.Symbol] = {}
        for s in expr.free_symbols:
            assert isinstance(s, sympy.Symbol)
            if s == self.program.batch_symbol:
                subs[s] = sympy.Symbol("rsb_bhn", integer=True, nonnegative=True)
            else:
                subs[s] = sympy.Symbol(self.names.scalar(s), integer=True)
        return cute_texpr(cast("sympy.Expr", expr.xreplace(subs)))

    def scalar_reg(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        key = str(expr)
        if key in self.scalar_regs:
            return self.scalar_regs[key]
        if isinstance(expr, sympy.Symbol):
            source = self.names.scalar(expr)
        else:
            source = self.render_scalar_index(expr)
        name = self.new_var()
        if dtype == torch.float32:
            self.emit(f"{name} = cutlass.Float32({source})")
        else:
            self.emit(f"{name} = cutlass.Int32({source})")
        self.scalar_regs[key] = name
        return name

    def scalar_load(self, value: ScalarLoad) -> str:
        key = (value.key, ())
        hit = self.memo.get(key)
        if hit is not None:
            return cast("IntReg", hit).name
        base, _align = self.base_var(
            value.ref, value.dims, (None,) * len(value.dims), None
        )
        offset = 0
        es = value.ref.dtype.itemsize
        for d, stride in zip(value.dims, value.ref.strides, strict=True):
            if isinstance(d, IdxConst):
                assert isinstance(d.value, int)
                offset += d.value * stride * es
        name = self.new_var()
        helper = "ldg_s32" if value.ref.dtype == torch.int32 else "ldg_s64"
        self.emit(f"{name} = {_RT_DAG}.{helper}({base} + {offset})")
        self.memo[key] = IntReg(name)
        return name

    # ---- vectorization

    def group_runs(
        self, entries: list[tuple[Affine, Operand]], es: int
    ) -> list[list[tuple[Affine, Operand]]]:
        """Maximal runs of consecutive addresses sharing their lane terms."""
        buckets: dict[tuple[int, int, int, int], list[tuple[Affine, Operand]]] = {}
        for aff, op in entries:
            buckets.setdefault(aff.lane_terms, []).append((aff, op))
        runs: list[list[tuple[Affine, Operand]]] = []
        for bucket in buckets.values():
            bucket.sort(key=lambda e: e[0].k)
            current: list[tuple[Affine, Operand]] = []
            for entry in bucket:
                if current and entry[0].k != current[-1][0].k + es:
                    runs.append(current)
                    current = []
                if current and entry[0].k == current[-1][0].k:
                    continue
                current.append(entry)
            if current:
                runs.append(current)
        return runs

    @staticmethod
    def vector_width(
        aff: Affine, es: int, align: int, remaining: int, *, max_bytes: int
    ) -> int:
        width = max_bytes // es
        while width > 1:
            bytes_ = width * es
            if (
                width <= remaining
                and align % bytes_ == 0
                and aff.cg % bytes_ == 0
                and aff.cc % bytes_ == 0
                and aff.ct % bytes_ == 0
                and aff.k % bytes_ == 0
            ):
                return width
            width //= 2
        return 1

    # ---- loads

    def emit_load(
        self,
        load: Load,
        coords: list[Coord],
        preds: tuple[Operand | None, ...] | None = None,
    ) -> None:
        """Load ``coords`` of ``load`` into the memo (vectorized runs).
        ``preds`` overrides the per-coordinate predicate (a fused ``where``)."""
        pending = []
        seen_coords: set[Coord] = set()
        for i, c in enumerate(coords):
            # (a broadcast request repeats the coordinates of its size-1 dims)
            if (load.key, c) in self.memo or c in seen_coords:
                continue
            seen_coords.add(c)
            pending.append((i, c))
        if not pending:
            return
        # The padded rows of a window hold zeros (the frontend masks them).
        real: list[tuple[int, Coord]] = []
        for i, c in pending:
            state = _Matcher.padding(
                load.dims, load.value_dims, c, self.loop_trips or 1
            )
            if state == "masked":
                self.memo[(load.key, c)] = ConstF(0.0)
            elif state == "mixed":
                raise _decline("a padded window must end on a fragment boundary")
            else:
                real.append((i, c))
        pending = real
        if not pending:
            return
        if preds is None and load.pred is None:
            # outputs read before their first write are the zeros they hold
            state = self.output_state(load, [c for _i, c in pending])
            if state == "zero":
                for _i, c in pending:
                    self.memo[(load.key, c)] = ConstF(0.0)
                return
            if isinstance(state, dict):
                for _i, c in pending:
                    self.memo[(load.key, c)] = state[c]
                return
        if self.pass_index == 1 and self.recording:
            self.recording[self.item_pos].append((load, tuple(coords), preds))
        ref = load.ref
        es = ref.dtype.itemsize
        base, align = self.base_var(ref, load.dims, load.value_dims, None)
        ordered = ref.name in self.program.stored_names
        entries: list[tuple[Affine, Coord, str | None]] = []
        for i, c in pending:
            aff = self.byte_affine(ref, load.dims, load.value_dims, c)
            pred_op: Operand | None = preds[i] if preds is not None else None
            if pred_op is None and load.pred is not None:
                pred_op = self.eval(
                    load.pred, self.broadcast(c, load.shape, load.pred.shape)
                )
            pred_text = self.pred_int(pred_op) if pred_op is not None else None
            entries.append((aff, c, pred_text))
        # group by lane terms and predicate
        buckets: dict[
            tuple[int, int, int, int, str | None], list[tuple[Affine, Coord]]
        ] = {}
        for aff, c, pred_text in entries:
            buckets.setdefault((*aff.lane_terms, pred_text), []).append((aff, c))
        for (_cg, _cc, _ct, _cgp, pred_text), bucket in buckets.items():
            bucket.sort(key=lambda e: e[0].k)
            position = 0
            while position < len(bucket):
                run = [bucket[position]]
                while (
                    position + len(run) < len(bucket)
                    and bucket[position + len(run)][0].k == run[-1][0].k + es
                ):
                    run.append(bucket[position + len(run)])
                self.emit_load_run(load, base, align, run, pred_text, ordered)
                position += len(run)

    def emit_load_run(
        self,
        load: Load,
        base: str,
        align: int,
        run: list[tuple[Affine, Coord]],
        pred: str | None,
        ordered: bool,
    ) -> None:
        es = load.ref.dtype.itemsize
        dtype = load.ref.dtype
        position = 0
        while position < len(run):
            aff0 = run[position][0]
            remaining = len(run) - position
            width = self.vector_width(aff0, es, align, remaining, max_bytes=16)
            address = f"{base} + {aff0.render('rsb_g64', 'rsb_c64')}"
            chunk = run[position : position + width]
            names = [self.new_var() for _ in range(width)]
            if dtype == torch.float32:
                if pred is not None:
                    helper = {
                        4: f"{_RT}.ldg_v4_f32_pred",
                        2: f"{_RT_DAG}.ldg_v2_f32_pred",
                        1: f"{_RT_DAG}.ldg_f32_pred",
                    }[width]
                    args = f"{address}, {pred}"
                elif ordered:
                    helper = {
                        4: f"{_RT}.ldg_v4_f32",
                        2: f"{_RT}.ldg_v2_f32",
                        1: f"{_RT}.ldg_f32",
                    }[width]
                    args = address
                else:
                    helper = {
                        4: f"{_RT}.ldg_v4_f32_ro",
                        2: f"{_RT}.ldg_v2_f32_ro",
                        1: f"{_RT}.ldg_f32_ro",
                    }[width]
                    args = address
                self.emit(f"{', '.join(names)} = {helper}({args})")
            else:
                unpack = "unpack_bf16x2" if dtype == torch.bfloat16 else "unpack_f16x2"
                if width == 1:
                    scalar = (
                        "ldg_bf16_as_f32"
                        if dtype == torch.bfloat16
                        else "ldg_f16_as_f32"
                    )
                    if pred is not None:
                        self.emit(
                            f"{names[0]} = {_RT_DAG}.{scalar}_pred({address}, {pred})"
                        )
                    else:
                        self.emit(f"{names[0]} = {_RT}.{scalar}({address})")
                else:
                    words = [self.new_var() for _ in range(width // 2)]
                    if pred is not None:
                        helper = {
                            4: f"{_RT}.ldg_v4_b32_pred",
                            2: f"{_RT}.ldg_v2_b32_pred",
                            1: f"{_RT_DAG}.ldg_b32_pred",
                        }[width // 2]
                        args = f"{address}, {pred}"
                    elif ordered:
                        helper = {
                            4: f"{_RT}.ldg_v4_b32_ord",
                            2: f"{_RT}.ldg_v2_b32",
                            1: f"{_RT}.ldg_b32",
                        }[width // 2]
                        args = address
                    else:
                        helper = {
                            4: f"{_RT}.ldg_v4_b32_ro",
                            2: f"{_RT}.ldg_v2_b32_ro",
                            1: f"{_RT}.ldg_b32_ro",
                        }[width // 2]
                        args = address
                    self.emit(f"{', '.join(words)} = {helper}({args})")
                    for i, w in enumerate(words):
                        self.emit(
                            f"{names[2 * i]}, {names[2 * i + 1]} = {_RT}.{unpack}({w})"
                        )
            for (_aff, coord), name in zip(chunk, names, strict=True):
                self.memo[(load.key, coord)] = Reg(name)
            position += width

    def output_state(
        self, load: Load, coords: list[Coord]
    ) -> str | dict[Coord, Operand]:
        """What a load of an output window at ``coords`` returns: ``"zero"``
        before any overlapping write of a zero-filled owned output, the
        register copy under register accumulation, else ``"memory"``.

        A window this warp wrote is read back only as it was written: the
        same window, at coordinates the lane stored (the C layout of its
        stores).  Another layout or a partially overlapping window would
        read other lanes' stores through memory without a fence between
        the lanes: declined (every knob corner fails, the family stays off).
        """
        if load.ref.name not in self.program.stored_names:
            return "memory"
        key = _window_key(load.ref, load.dims)
        written = [
            state
            for k, state in self.windows.items()
            if state.writes_done and _windows_overlap(k, key)
        ]
        if not written:
            if load.ref.name in self.program.zero_owned:
                return "zero"
            return "memory"
        exact = self.windows.get(key)
        if (
            len(written) != 1
            or exact is None
            or written[0] is not exact
            or any(c not in exact.coords for c in coords)
        ):
            raise _decline(
                "a stored window is read back outside the layout it was written in"
            )
        if self.colside == COLSIDE_REGS:
            assert exact.regs is not None
            return exact.regs
        return "memory"

    # ---- evaluation

    @staticmethod
    def broadcast(coord: Coord, shape: Shape, arg_shape: Shape) -> Coord:
        offset = len(shape) - len(arg_shape)
        result = []
        for j, size in enumerate(arg_shape):
            c = coord[j + offset]
            result.append(ZERO if size == 1 else c)
        return tuple(result)

    def eval(self, value: Value, coord: Coord) -> Operand:
        return self.eval_many(value, [coord])[0]

    def eval_many(self, value: Value, coords: list[Coord]) -> list[Operand]:
        if self.hoist_record is not None and not any(a.ct for c in coords for a in c):
            # loop-invariant: rendered ahead of the rolled loop
            self.hoist_record.append(("eval", value, tuple(coords)))
        result: list[Operand | None] = [self.memo.get((value.key, c)) for c in coords]
        # (a broadcast request repeats the coordinates of its size-1 dims:
        # every distinct coordinate is rendered once)
        missing = list(
            dict.fromkeys(c for c, r in zip(coords, result, strict=True) if r is None)
        )
        if not missing:
            return cast("list[Operand]", result)
        if isinstance(value, Load):
            self.emit_load(value, missing)
        elif isinstance(value, ScalarLoad):
            self.scalar_load(value)
            for c in missing:
                self.memo[(value.key, c)] = self.memo[(value.key, ())]
        elif isinstance(value, Const):
            for c in missing:
                self.memo[(value.key, c)] = ConstF(value.value)
        elif isinstance(value, Zeros):
            for c in missing:
                self.memo[(value.key, c)] = ConstF(0.0)
        elif isinstance(value, ScalarSym):
            name = self.scalar_reg(value.expr, value.dtype)
            for c in missing:
                self.memo[(value.key, c)] = (
                    Reg(name) if value.dtype == torch.float32 else IntReg(name)
                )
        elif isinstance(value, Iota):
            for c in missing:
                self.memo[(value.key, c)] = IntAff(c[0] + value.start)
        elif isinstance(value, WindowIota):
            raise _decline("a runtime window base may only subscript tensors")
        elif isinstance(value, View):
            inner = [self.view_coord(value, c) for c in missing]
            ops = self.eval_many(value.x, inner)
            for c, op in zip(missing, ops, strict=True):
                self.memo[(value.key, c)] = op
        elif isinstance(value, Permute):
            inner = [_coord(c[:-2], c[-1], c[-2]) for c in missing]
            ops = self.eval_many(value.x, inner)
            for c, op in zip(missing, ops, strict=True):
                self.memo[(value.key, c)] = op
        elif isinstance(value, Cast):
            ops = self.eval_many(
                value.x,
                [self.broadcast(c, value.shape, value.x.shape) for c in missing],
            )
            for c, op in zip(missing, ops, strict=True):
                self.memo[(value.key, c)] = self.cast(op, value.x.dtype, value.dtype)
        elif isinstance(value, Compare):
            lhs = self.eval_many(
                value.lhs,
                [self.broadcast(c, value.shape, value.lhs.shape) for c in missing],
            )
            rhs = self.eval_many(
                value.rhs,
                [self.broadcast(c, value.shape, value.rhs.shape) for c in missing],
            )
            for c, lo, ro in zip(missing, lhs, rhs, strict=True):
                self.memo[(value.key, c)] = self.compare(value.op, lo, ro)
        elif isinstance(value, Pointwise):
            self.eval_pointwise(value, missing)
        elif isinstance(value, Dot):
            self.eval_dot(value, missing)
        elif isinstance(value, RowReduce):
            self.eval_row_reduce(value, missing)
        else:
            raise _decline(f"cannot render {type(value).__name__}")
        return [self.memo[(value.key, c)] for c in coords]

    @staticmethod
    def view_coord(view: View, coord: Coord) -> Coord:
        inner: list[Affine] = [ZERO] * view.x.ndim
        for d, src in enumerate(view.src_dims):
            if src is not None:
                inner[src] = ZERO if view.x.shape[src] == 1 else coord[d]
        return tuple(inner)

    def cast(self, op: Operand, source: torch.dtype, target: torch.dtype) -> Operand:
        if source == torch.bool:
            if isinstance(op, ConstB):
                return ConstF(1.0 if op.value else 0.0)
            if isinstance(op, Pred):
                name = self.new_var()
                self.emit(
                    f"{name} = {_RT}.sel_{_SEL_OP[op.op]}_f32({op.lhs}, {op.rhs}, "
                    "cutlass.Float32(1.0), cutlass.Float32(0.0))"
                )
                return Reg(name)
            raise _decline("cast of a non-static predicate")
        if target == torch.float32:
            if isinstance(op, IntAff):
                if op.aff.is_const:
                    return ConstF(float(op.aff.k))
                name = self.new_var()
                self.emit(
                    f"{name} = cutlass.Float32({op.aff.render('rsb_g', 'rsb_c')})"
                )
                return Reg(name)
            return op  # 16-bit values are held widened already
        if target == torch.bfloat16:
            return self.rounded(op)
        if target == torch.float16:
            raise _decline("fp16 rounding outside a store is not supported")
        raise _decline(f"cast to {target}")

    def rounded(self, op: Operand) -> Operand:
        """``op`` rounded to bfloat16 (held widened), as torch rounds the
        result of every bf16 op."""
        if isinstance(op, ConstF):
            return ConstF(_round_bf16(op.value))
        if isinstance(op, IntAff) and op.aff.is_const:
            return ConstF(_round_bf16(float(op.aff.k)))
        name = self.new_var()
        self.emit(f"{name} = {_RT}.round_bf16_f32({self.f32(op)})")
        return Reg(name)

    def int_text(self, op: Operand) -> str:
        if isinstance(op, IntAff):
            return op.aff.render("rsb_g", "rsb_c")
        if isinstance(op, IntReg):
            return op.name
        if isinstance(op, ConstF) and float(op.value) == int(op.value):
            return str(int(op.value))
        raise _decline("an integer comparison of a non-integer value")

    def compare(self, op: str, lhs: Operand, rhs: Operand) -> Operand:
        if (
            isinstance(lhs, IntAff)
            and isinstance(rhs, IntAff)
            and lhs.aff.is_const
            and rhs.aff.is_const
        ):
            a, b = lhs.aff.k, rhs.aff.k
            return ConstB(
                {
                    "ge": a >= b,
                    "gt": a > b,
                    "le": a <= b,
                    "lt": a < b,
                    "eq": a == b,
                    "ne": a != b,
                }[op]
            )
        return Pred(op, self.int_text(lhs), self.int_text(rhs))

    def pred_int(self, op: Operand) -> str:
        if isinstance(op, ConstB):
            return f"cutlass.Int32({1 if op.value else 0})"
        if isinstance(op, Pred):
            return f"(cutlass.Int32(1) if ({op.lhs}) {_PY_CMP[op.op]} ({op.rhs}) else cutlass.Int32(0))"
        raise _decline("a load predicate must be a comparison")

    def f32(self, op: Operand) -> str:
        if isinstance(op, Reg):
            return op.name
        if isinstance(op, ConstF):
            return f"cutlass.Float32({op.value!r})"
        if isinstance(op, IntAff) and op.aff.is_const:
            return f"cutlass.Float32({float(op.aff.k)!r})"
        raise _decline(f"{type(op).__name__} is not an fp32 value")

    def eval_pointwise(self, value: Pointwise, coords: list[Coord]) -> None:
        op = value.op
        if op == "where":
            self.eval_where(value, coords)
            return
        arg_ops = [
            self.eval_many(
                arg, [self.broadcast(c, value.shape, arg.shape) for c in coords]
            )
            for arg in value.args
        ]
        bf16 = value.dtype == torch.bfloat16
        for i, c in enumerate(coords):
            ops = [col[i] for col in arg_ops]
            result = self.arith(op, ops)
            self.memo[(value.key, c)] = self.rounded(result) if bf16 else result

    def arith(self, op: str, ops: list[Operand]) -> Operand:
        consts = [o.value for o in ops if isinstance(o, ConstF)]
        if len(consts) == len(ops):
            a = consts[0]
            if op == "neg":
                return ConstF(-a)
            if op == "exp2":
                return ConstF(2.0**a)
            if op == "rsqrt":
                if a > 0:
                    return ConstF(a**-0.5)
                # torch: rsqrt(+0.0) = inf, rsqrt(-0.0) = -inf
                return ConstF(math.copysign(math.inf, a) if a == 0 else math.nan)
            b = consts[1]
            return ConstF({"add": a + b, "sub": a - b, "mul": a * b}[op])
        texts = [self.f32(o) for o in ops]
        name = self.new_var()
        if op == "neg":
            self.emit(f"{name} = -{texts[0]}")
        elif op in ("exp2", "rsqrt"):
            fast = ", fastmath=True" if self.fast_math else ""
            self.emit(f"{name} = cute.math.{op}({texts[0]}{fast})")
        else:
            symbol = {"add": "+", "sub": "-", "mul": "*"}[op]
            self.emit(f"{name} = {texts[0]} {symbol} {texts[1]}")
        return Reg(name)

    def eval_where(self, value: Pointwise, coords: list[Coord]) -> None:
        cond, x, y = value.args
        cond_ops = self.eval_many(
            cond, [self.broadcast(c, value.shape, cond.shape) for c in coords]
        )
        y_is_zero = isinstance(y, Const) and y.value == 0.0 or isinstance(y, Zeros)
        load = _loaded_value(x)
        # ``where(mask, load, 0)``: a predicated load zero-fills under the
        # mask already; otherwise under the predicate knob the load itself
        # takes the mask as its predicate.
        if y_is_zero and load is not None:
            x_coords = [self.broadcast(c, value.shape, x.shape) for c in coords]
            if load.pred is not None and load.pred.key == cond.key:
                ops = self.eval_many(x, x_coords)
                for c, op in zip(coords, ops, strict=True):
                    self.memo[(value.key, c)] = op
                return
            if (
                self.mask == MASK_PREDICATE
                and load.pred is None
                and all(isinstance(o, (Pred, ConstB)) for o in cond_ops)
                and load.ref.name not in self.program.stored_names
                and x is load
            ):
                preds = tuple(
                    None if isinstance(o, ConstB) and o.value else o for o in cond_ops
                )
                fused = [
                    c for c, p in zip(x_coords, preds, strict=True) if p is not None
                ]
                fused_preds = tuple(p for p in preds if p is not None)
                if fused:
                    self.emit_load(load, fused, fused_preds)
                plain = [c for c, p in zip(x_coords, preds, strict=True) if p is None]
                if plain:
                    self.emit_load(load, plain)
                for c, xc, p in zip(coords, x_coords, preds, strict=True):
                    op = self.memo[(load.key, xc)]
                    if isinstance(p, ConstB) and not p.value:
                        op = ConstF(0.0)
                    self.memo[(value.key, c)] = op
                return
        x_ops = self.eval_many(
            x, [self.broadcast(c, value.shape, x.shape) for c in coords]
        )
        y_ops = self.eval_many(
            y, [self.broadcast(c, value.shape, y.shape) for c in coords]
        )
        if value.dtype == torch.bfloat16:
            # (a Python scalar operand of a bf16 ``where`` is a bf16 value)
            x_ops = [self.rounded(o) if isinstance(o, ConstF) else o for o in x_ops]
            y_ops = [self.rounded(o) if isinstance(o, ConstF) else o for o in y_ops]
        for c, co, xo, yo in zip(coords, cond_ops, x_ops, y_ops, strict=True):
            if isinstance(co, ConstB):
                self.memo[(value.key, c)] = xo if co.value else yo
                continue
            if not isinstance(co, Pred):
                raise _decline("where on a non-static predicate")
            name = self.new_var()
            self.emit(
                f"{name} = {_RT}.sel_{_SEL_OP[co.op]}_f32({co.lhs}, {co.rhs}, {self.f32(xo)}, {self.f32(yo)})"
            )
            self.memo[(value.key, c)] = Reg(name)

    # ---- row reductions

    def eval_row_reduce(self, value: RowReduce, coords: list[Coord]) -> None:
        """The row total at each requested ``(lead, row, 0)`` coordinate: the
        lane's ``E / 4`` values of the row in the A-operand layout (``k_aff``,
        so a dot over ``K = E`` of the same value reads the same registers),
        summed lane-locally in up to four partial chains and a balanced add,
        then exchanged across the row's four lanes by two butterfly shuffles.
        Lane-uniform control flow only (the family renders none other), so
        every lane of the warp reaches every shuffle."""
        extent = self.size(value.x.shape[-1])
        if not isinstance(extent, int):
            raise _decline("a reduced extent must be the same in every merged group")
        kq = extent // 4
        lead_dims = value.ndim - 2
        for c in coords:
            if c[-1] != ZERO:
                raise _decline("a row reduction is read outside its row")
            lead, row = c[:lead_dims], c[lead_dims]
            if row.cc:
                # the butterfly partners (lanes xor 1, xor 2) differ in c:
                # they must hold the same row (a B-operand K row does not)
                raise _decline(
                    "a row reduction's row must not depend on the lane quarter"
                )
            x_coords = [_coord(lead, row, self.k_aff(kq, j)) for j in range(kq)]
            ops = self.eval_many(value.x, x_coords)
            constants = [o.value for o in ops if isinstance(o, ConstF)]
            if len(constants) == len(ops):
                # lane-invariant terms: the four lanes hold the same quarter
                self.memo[(value.key, c)] = ConstF(4.0 * math.fsum(constants))
                continue
            texts = [self.f32(o) for o in ops]
            chains = min(4, len(texts))
            partials: list[str] = []
            for p in range(chains):
                chunk = texts[p * len(texts) // chains : (p + 1) * len(texts) // chains]
                name = self.new_var()
                self.emit(f"{name} = {' + '.join(chunk)}")
                partials.append(name)
            while len(partials) > 1:
                merged: list[str] = []
                for i in range(0, len(partials) - 1, 2):
                    name = self.new_var()
                    self.emit(f"{name} = {partials[i]} + {partials[i + 1]}")
                    merged.append(name)
                if len(partials) % 2:
                    merged.append(partials[-1])
                partials = merged
            total = partials[0]
            for offset in (1, 2):
                name = self.new_var()
                self.emit(
                    f"{name} = {total} + cute.arch.shuffle_sync_bfly({total}, "
                    f"offset={offset}, mask=0xffffffff, mask_and_clamp=31)"
                )
                total = name
            self.memo[(value.key, c)] = Reg(total)

    # ---- dots

    def dot_layout(self, dot: Dot) -> _DotLayout:
        layout = self.dots.get(dot.key)
        if layout is not None:
            return layout
        m, n = self.dims_mn(dot)
        k = self.max_int(self.size(dot.a.shape[-1]))
        m_tiles, nt = m // 16, n // 8
        lead = (ZERO,) * (dot.ndim - 2)
        c_coords: dict[tuple[int, int, int], Coord] = {}
        tile_of: dict[Coord, tuple[int, int]] = {}
        for mi in range(m_tiles):
            for h in range(2):
                row = Affine(cg=1, k=16 * mi + 8 * h)
                for mm in range(2 * nt):
                    coord = _coord(lead, row, self.c_col(nt, mm))
                    c_coords[(mi, h, mm)] = coord
                    tile = mm % nt if self.program.n_mode == "permuted" else mm // 2
                    tile_of[coord] = (mi, tile)
                if self.program.n_mode == "tiled":
                    # the symbolic n-tile of a rolled loop
                    for e in range(2):
                        coord = _coord(lead, row, Affine(cc=2, k=e, ct=8))
                        c_coords[(mi, h, SYM_TILE - e)] = coord
                        tile_of[coord] = (mi, SYM_TILE)
        layout = _DotLayout(m_tiles, k, nt, c_coords, tile_of)
        self.dots[dot.key] = layout
        return layout

    @staticmethod
    def k_aff(kq: int, v: int) -> Affine:
        """The K coordinate of lane value ``v`` (of the lane's ``kq``): the
        lane's values come in packets of eight (a 16-byte packet of 16-bit
        elements) interleaved across the four lanes of a row, so the four
        lanes of one row read 64 contiguous bytes per packet (full 32-byte
        sectors); a K extent of 16 keeps one packet of four per lane."""
        packet = 8 if kq % 8 == 0 else kq
        return Affine(cc=packet, k=(v // packet) * 4 * packet + v % packet)

    def issue_a_operand(self, dot: Dot) -> None:
        """Render the A operand of ``dot`` (every lane value, as tf32 bits)."""
        layout = self.dot_layout(dot)
        kq = layout.k // 4
        lead = (ZERO,) * (dot.ndim - 2)
        a_coords = [
            _coord(lead, Affine(cg=1, k=16 * mi + 8 * h), self.k_aff(kq, j))
            for mi in range(layout.m_tiles)
            for h in range(2)
            for j in range(kq)
        ]
        for op in self.eval_many(dot.a, a_coords):
            self.bits(op)

    def eval_dot(self, dot: Dot, coords: list[Coord]) -> None:
        layout = self.dot_layout(dot)
        k, nt = layout.k, layout.n_tiles
        kq = k // 4
        lead = (ZERO,) * (dot.ndim - 2)
        tiles: list[tuple[int, int]] = []
        for c in coords:
            tile = layout.tile_of.get(c)
            if tile is None:
                raise _decline("a dot result is read outside its accumulator layout")
            if tile not in tiles and tile not in layout.done:
                tiles.append(tile)
        if not tiles:
            return
        if not layout.operands_issued:
            layout.operands_issued = True
            if self.hoist_record is not None:
                self.hoist_record.append(("abits", dot))
            self.issue_a_operand(dot)
            if _contiguous_dim(dot.b) != 0:
                b_coords = [
                    _coord(lead, self.k_aff(kq, i), self.b_col(nt, tile))
                    for tile in range(nt)
                    for i in range(kq)
                ]
                self.eval_many(dot.b, b_coords)
        valid_n = _valid_extent(dot.b, dot.b.ndim - 1)
        for mi, tile in tiles:
            symbolic = tile == SYM_TILE
            if not symbolic and isinstance(valid_n, int) and 8 * tile >= valid_n:
                # An n-tile in the padded columns of the operand (the
                # frontend's ``_mask_to`` zeroes them): zeros.
                for h in range(2):
                    for e in range(2):
                        coord = layout.c_coords[(mi, h, self.c_m_index(nt, tile, e))]
                        self.memo[(dot.key, coord)] = ConstF(0.0)
                layout.done.add((mi, tile))
                continue
            b_col = Affine(cg=1, ct=8) if symbolic else self.b_col(nt, tile)
            b_coords = [_coord(lead, self.k_aff(kq, i), b_col) for i in range(kq)]
            b_ops = self.eval_many(dot.b, b_coords)
            acc = ["cutlass.Float32(0.0)"] * 4
            for s in range(k // 8):
                a_ops = self.eval_many(
                    dot.a,
                    [
                        _coord(lead, Affine(cg=1, k=16 * mi), self.k_aff(kq, 2 * s)),
                        _coord(
                            lead, Affine(cg=1, k=16 * mi + 8), self.k_aff(kq, 2 * s)
                        ),
                        _coord(
                            lead, Affine(cg=1, k=16 * mi), self.k_aff(kq, 2 * s + 1)
                        ),
                        _coord(
                            lead, Affine(cg=1, k=16 * mi + 8), self.k_aff(kq, 2 * s + 1)
                        ),
                    ],
                )
                a_bits = [self.bits(o) for o in a_ops]
                b_bits = [self.bits(b_ops[2 * s]), self.bits(b_ops[2 * s + 1])]
                outs = [self.new_var() for _ in range(4)]
                self.emit(
                    f"{', '.join(outs)} = {_RT}.mma_tf32({', '.join(a_bits)}, {', '.join(b_bits)}, {', '.join(acc)})"
                )
                acc = outs
            for h in range(2):
                for e in range(2):
                    mm = SYM_TILE - e if symbolic else self.c_m_index(nt, tile, e)
                    coord = layout.c_coords[(mi, h, mm)]
                    self.memo[(dot.key, coord)] = Reg(acc[2 * h + e])
            layout.done.add((mi, tile))

    def bits(self, op: Operand) -> str:
        if isinstance(op, ConstF):
            word = struct.unpack("<I", struct.pack("<f", op.value))[0]
            return f"cutlass.Uint32({word})"
        text = self.f32(op)
        hit = self.bits_memo.get(text)
        if hit is not None:
            return hit
        name = self.new_var()
        self.emit(f"{name} = {_RT_DAG}.f32_bits({text})")
        self.bits_memo[text] = name
        return name


def _valid_extent(value: Value, dim: int) -> int | Affine | None:
    """The real extent of ``value``'s dim ``dim`` when it is a padded arange
    window of some load under it (None: the whole dim is real)."""
    if isinstance(value, Load):
        for d, vd in zip(value.dims, value.value_dims, strict=True):
            if (
                vd == dim
                and isinstance(d, (IdxWindow, IdxWindowBase))
                and d.size != d.extent
            ):
                return d.extent
        return None
    if isinstance(value, Permute):
        if dim >= value.ndim - 2:
            dim = value.ndim - 1 if dim == value.ndim - 2 else value.ndim - 2
        return _valid_extent(value.x, dim)
    if isinstance(value, View):
        src = value.src_dims[dim]
        return None if src is None else _valid_extent(value.x, src)
    if isinstance(value, Cast):
        return _valid_extent(value.x, dim)
    if isinstance(value, Pointwise):
        found: int | Affine | None = None
        for arg in value.args:
            offset = value.ndim - arg.ndim
            if dim - offset >= 0 and arg.shape[dim - offset] == value.shape[dim]:
                extent = _valid_extent(arg, dim - offset)
                if extent is None:
                    continue
                if found is None:
                    found = extent
                elif isinstance(found, int) and isinstance(extent, int):
                    found = min(found, extent)
        return found
    return None


def _loaded_value(value: Value) -> Load | None:
    while isinstance(value, (Cast, View)):
        value = value.x
    return value if isinstance(value, Load) else None


# --------------------------------------------------------------------------
# Codegen entry


def emit_register_subblocks_module_statements(cg: GenerateAST) -> None:
    if getattr(cg, "_helion_register_subblocks_module_emitted", False):
        return
    cg._helion_register_subblocks_module_emitted = True  # type: ignore[attr-defined]
    for stmt in ast.parse(_PREAMBLE).body:
        cg.module_statements.append(stmt)


def codegen_register_subblocks(cg: GenerateAST) -> bool:
    """Replace the device body by the register sub-block program of the plan."""
    df = cg.device_function
    plan = df.cute_state.register_subblocks_plan
    if plan is None:
        return False
    root = cg.current_root_graph_info
    if root is None or root.graph_id != plan.program.root_graph_id:
        return False
    emit_register_subblocks_module_statements(cg)
    emitter = _Emitter(
        plan.program,
        bk=plan.bk,
        warps=plan.warps,
        prefetch=plan.prefetch,
        mask=plan.mask,
        colside=plan.colside,
        roll=plan.roll,
        names=_DeviceNames(cg),
    )
    source = emitter.render()
    cg.cute_wrapper_plans.append(
        {
            "kind": PLAN_KIND,
            "total_ctas": plan.total_ctas,
            "threads": plan.threads,
            "warps": plan.warps,
            "live_estimate": emitter.max_live,
        }
    )
    df.body = list(ast.parse(textwrap.dedent(source)).body)
    df.preamble = []
    df.cute_state.owned_root_block_dims = (plan.threads, 1, 1)
    return True
