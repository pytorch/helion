"""Fused tcgen05 body for gated (softmax-free) attention over jagged rows.

This generalizes the CuTe flash-attention family in three directions:

* **Score plan.** Instead of the online-softmax chain, the QK score feeds an
  arbitrary elementwise *gate program* captured from the FX graph (for HSTU:
  ``where(causal & mask_q & mask_kv, silu(qk * alpha) * scale, 0)``).  There is
  no running max or sum, so the O rescale stage is absent and the P tile is the
  gate output cast to the MMA dtype.  The program is replayed on 32-column
  vectors of score elements with the frontend's dtype rounding points
  preserved (a bf16 node is rounded through bf16 -- with the paired
  ``cvt.rn.bf16x2`` -- before it is consumed; a rounding of a value already
  exact in that dtype is elided).  Boolean terms that compare the KV column
  index with a column-independent scalar are per-chunk 32-bit masks, ANDed
  and materialized once at the select that consumes them; they also bound the
  KV range of a query tile (columns every row masks are skipped).  Masked
  scores are replaced by 1.0 before the gate runs (their gate values are
  discarded) so TMA zero fill never sends the IEEE division down its slow
  path.  ``exp`` feeding ``c + e`` uses the flush-to-zero ex2 (bit-identical
  there); the division stays IEEE unless the ``fast_math`` setting is on.
* **Row loads.** Two forms.  *Root form*: Q/K/V/O rows are
  ``x[tile.index + offset, lane, :]`` on a rank-3 host tensor whose leading
  coordinate is a singleton root axis (the head), with a loop-invariant scalar
  row offset (``seq_offsets[b]``); the query tile is the last root axis and the
  KV loop bound is a runtime scalar program (``tile_q.end``).  *Sequence form*:
  a grid over sequences with data-dependent ``hl.tile(start, end)`` query and
  KV loops sharing the sequence bounds, ``x[tile, :, :]`` rows batched over
  the lane dim (``bmm``); one CTA per (sequence, lane) walks the sequence's
  query tiles (heaviest first).  TMA descriptors cover the whole
  ``[rows, D, lanes]`` tensor; the per-CTA tiles are addressed with
  ``cute.domain_offset`` so partial tiles past the end of the tensor are
  zero-filled by TMA.  fp32 rows run the MMA as tf32 (Helion's default
  ``dot_precision``).
* **Roles.** The body is warp-specialized: one TMA/MMA producer warp and
  ``cute_flash_gate_warpgroups`` 128-thread gate warpgroups that split every
  KV tile chunk-wise (the K/V ring depth ``cute_flash_kv_stage`` and the KV
  tile width are the other knobs).  Warpgroup 0 drains O with one vectorized
  row store per thread.  Warp 1 owns the TMEM allocation (overlapping warp
  0's mbarrier setup and first loads) and frees it after a named barrier the
  gate warps arrive on right after their last TMEM read, so the deallocation
  overlaps the output store instead of following it.

Intentional semantic deviations from the FX program:

* The output store: rows whose gate is forced to zero by a row-only mask term
  (``mask_q``) are not stored.  The frontend would store zeros there, which
  for jagged layouts alias the next sequence's rows (a write race the
  reference does not have); skipping them yields the reference values.  This
  is only done when the accumulator starts at zero and the masked value is
  the constant zero.
* Non-finite V rows only (finite data is bit-identical): KV tiles that the P
  mask provably zeroes are skipped, so a NaN/inf V row inside such a tile no
  longer poisons the output through ``0 * V`` as it does in the frontend; and
  in the sequence form the last partial KV tile of a sequence loads the next
  sequence's rows unmasked (the frontend masks them to zero), so a NaN/inf
  there reaches the output as ``0 * inf``.
* The 64-row query tile follows the program's own block-size dependence: a
  KV loop bounded by the query tile (``hl.tile(0, tile_q.end)``) ends at
  that tile's end, so rows 128..191 of a sequence visit KV columns below 192
  at block size 64 but below 256 at block size 128 (the Triton backend does
  the same).  Finite data is identical either way (the extra columns are
  masked to P = 0); only ``0 * NaN/inf`` poisoning from V rows in those
  columns moves with the tile height.
"""

from __future__ import annotations

import ast
import contextvars
import dataclasses
import operator
import re
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from ..device_ir import ElseGraphInfo
from ..device_ir import ForLoopGraphInfo
from ..device_ir import GraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import NodeArgsGraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from .cute_flash import FLASH_KV_STAGE_KEY
from .cute_flash import _flash_graph_host_tensors
from .cute_flash import _flash_ws_qk_ahead
from .cute_flash import emit_flash_module_statements

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import sympy

    from ...runtime.config import Config
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST

GATED_WRAPPER_KIND = "helion_flash_gated"
GATED_Q_TILE = 128
# Query tile heights (rows per CTA work item): the M=128 tcgen05 tile, or the
# M=64 tile whose ``16x64b`` TMEM shapes hand each gate thread half a row (the
# two column parities of a row go to lanes ``t`` and ``t ^ 2``), so a tile has
# twice the CTAs and each gate thread half the columns.
GATED_Q_TILE_CHOICES = (128, 64)
GATED_KV_TILE_CHOICES = (64, 128)
GATED_HEAD_DIMS = (32, 64, 128)
GATED_KV_STAGE_CHOICES = (2, 3, 4)
# 128-thread gate warpgroups per CTA.  They split the S columns of one KV
# tile chunk-wise (chunk ``ci`` belongs to warpgroup ``ci % G``): the gate is
# issue-bound on one warp per SM sub-partition, so more warpgroups multiply the
# issue slots.  Legal when the chunk count divides evenly.
GATED_GATE_WARPGROUP_CHOICES = (1, 2, 4)
GATED_GATE_WARPGROUP_DEFAULT = 2
FLASH_GATE_WARPGROUPS_KEY = "cute_flash_gate_warpgroups"
GATED_IO_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Reserve for mbarriers, the TMEM holding slot and allocator padding.
_GATED_SMEM_RESERVE_BYTES = 1024
# sm_100 opt-in per-CTA shared memory; used when the device is unknown (CPU
# codegen) so the search surface is identical on and off the GPU.
_GATED_DEFAULT_SMEM_CAPACITY = 232448
_LOG2_E = 1.4426950408889634
# Score elements per thread per TMEM load chunk (one lane of the gate vector
# each): 32 (``Ld32x32bOp`` / ``Ld16x64bOp`` with 32 repeats), or fewer at the
# 64-row tile when more gate warpgroups split a thread's half row.
GATED_CHUNK_COLS = 32
GATED_MIN_CHUNK_COLS = 16
# The otherwise idle warp 1 owns the TMEM allocation: it allocates while warp 0
# initializes the mbarriers and issues the first loads, and frees after the
# named teardown barrier the TMEM-reading warps arrive on (id 1 is the
# allocation barrier every thread joins).
GATED_TMEM_WARP = 1
GATED_TEARDOWN_BARRIER_ID = 2

# Kernel param order MUST match the ``call_args`` appended by the
# ``helion_flash_gated`` wrapper plan in ``runtime/cute/launcher.py``.
GATED_KERNEL_PARAMS = [
    "_flash_qk_mma",
    "_flash_pv_mma",
    "_flash_tma_q",
    "_flash_mQt",
    "_flash_tma_k",
    "_flash_mKt",
    "_flash_tma_v",
    "_flash_mVt",
    "_flash_mOt",
    "_flash_qsl",
    "_flash_ksl",
    "_flash_vsl",
    "_flash_ptl",
]

# Program domains: which tile coordinates a captured node depends on.
_Q_ROW = "q_row"
_KV_COL = "kv_col"
_KV_SCALAR = "kv_scalar"
# Query-tile scalars (``tile_q.begin/end/id``): one value per work item of a
# CTA (the sequence form walks a sequence's query tiles inside the CTA).
_Q_TILE = "q_tile"

_AND_TARGETS: frozenset[object] = frozenset(
    {
        operator.and_,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.logical_and.default,
    }
)
_OR_TARGETS: frozenset[object] = frozenset(
    {
        operator.or_,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.logical_or.default,
    }
)
_ADD_TARGETS: frozenset[object] = frozenset(
    {operator.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar}
)
_SUB_TARGETS: frozenset[object] = frozenset(
    {operator.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar}
)
_MUL_TARGETS: frozenset[object] = frozenset(
    {operator.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar}
)
_DIV_TARGETS: frozenset[object] = frozenset(
    {operator.truediv, torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar}
)
_FLOORDIV_TARGETS: frozenset[object] = frozenset({operator.floordiv})
_MOD_TARGETS: frozenset[object] = frozenset(
    {operator.mod, torch.ops.aten.remainder.Tensor, torch.ops.aten.remainder.Scalar}
)
_COMPARE_TARGETS: dict[object, str] = {
    operator.gt: ">",
    operator.ge: ">=",
    operator.lt: "<",
    operator.le: "<=",
    operator.eq: "==",
    operator.ne: "!=",
    torch.ops.aten.gt.Tensor: ">",
    torch.ops.aten.gt.Scalar: ">",
    torch.ops.aten.ge.Tensor: ">=",
    torch.ops.aten.ge.Scalar: ">=",
    torch.ops.aten.lt.Tensor: "<",
    torch.ops.aten.lt.Scalar: "<",
    torch.ops.aten.le.Tensor: "<=",
    torch.ops.aten.le.Scalar: "<=",
    torch.ops.aten.eq.Tensor: "==",
    torch.ops.aten.eq.Scalar: "==",
    torch.ops.aten.ne.Tensor: "!=",
    torch.ops.aten.ne.Scalar: "!=",
}
_UNARY_MATH_TARGETS: dict[object, str] = {
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.exp2.default: "exp2",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.silu.default: "silu",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.rsqrt.default: "rsqrt",
    torch.ops.aten.log.default: "log",
    torch.ops.aten.reciprocal.default: "reciprocal",
    torch.ops.aten.neg.default: "neg",
    operator.neg: "neg",
}
_MINMAX_TARGETS: dict[object, str] = {
    torch.ops.aten.maximum.default: ">",
    torch.ops.aten.minimum.default: "<",
}
_VIEW_TARGETS: frozenset[object] = frozenset(
    {
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        view_ops.subscript,
    }
)
_CONVERT_TARGETS: frozenset[object] = frozenset(
    {torch.ops.prims.convert_element_type.default, torch.ops.aten._to_copy.default}
)
_LT_TARGETS: frozenset[object] = frozenset(
    {operator.lt, torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar}
)
_GT_TARGETS: frozenset[object] = frozenset(
    {operator.gt, torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Scalar}
)
# Calls without effects that may appear anywhere in the matched graphs (dead
# or feeding matched nodes) without being part of a checked program.
_PURE_LEAF_TARGETS: frozenset[object] = frozenset(
    {
        _tracing_ops._new_var,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        tile_ops.tile_index,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_id,
        tile_ops.tile_block_size,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.sym_stride.int,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.scalar_tensor.default,
        creation_ops.full,
    }
)


def _gated_io_dtype_str(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "cutlass.Float16"
    if dtype is torch.bfloat16:
        return "cutlass.BFloat16"
    assert dtype is torch.float32, dtype
    return "cutlass.Float32"


def _gated_mma_dtype_str(dtype: torch.dtype) -> str:
    """MMA operand type: fp32 tensors run the tcgen05 MMA as tf32 (Helion's
    default ``dot_precision``, the same policy as Triton's ``tl.dot``)."""
    if dtype is torch.float32:
        return "cutlass.TFloat32"
    return _gated_io_dtype_str(dtype)


def _is_full_slice(value: object) -> bool:
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


def _node_val(node: torch.fx.Node) -> object:
    return node.meta.get("val")


def _symnode_block_id(node: object) -> int | None:
    """Block id named by a ``_get_symnode('block_size_N')`` node."""
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is _tracing_ops._get_symnode
        and node.args
        and isinstance(node.args[0], str)
        and node.args[0].startswith("block_size_")
    ):
        suffix = node.args[0].removeprefix("block_size_")
        return int(suffix) if suffix.isdigit() else None
    return None


def _tile_index_block_id(env: CompileEnvironment, node: torch.fx.Node) -> int | None:
    val = _node_val(node)
    if not isinstance(val, torch.Tensor) or val.ndim != 1:
        return None
    return env.get_block_id(val.shape[0])


def _tile_op_block_id(env: CompileEnvironment, node: torch.fx.Node) -> int | None:
    """Block id addressed by ``tile_begin``/``tile_end``/``tile_id``."""
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    block_id = _symnode_block_id(node.args[0])
    if block_id is not None:
        return block_id
    val = _node_val(node.args[0])
    if isinstance(val, torch.SymInt):
        return env.get_block_id(val)
    return None


class _GraphScope:
    """Resolve placeholders across nested Helion graphs to their outer nodes."""

    def __init__(self, graphs: Iterable[GraphInfo]) -> None:
        self.infos = tuple(graphs)
        self.info_by_graph: dict[torch.fx.Graph, GraphInfo] = {
            info.graph: info for info in self.infos
        }
        self._placeholders: dict[torch.fx.Graph, list[torch.fx.Node]] = {}

    def outer_node(self, placeholder: torch.fx.Node) -> torch.fx.Node | None:
        info = self.info_by_graph.get(placeholder.graph)
        if not isinstance(info, NodeArgsGraphInfo):
            return None
        placeholders = self._placeholders.get(placeholder.graph)
        if placeholders is None:
            placeholders = [
                node for node in placeholder.graph.nodes if node.op == "placeholder"
            ]
            self._placeholders[placeholder.graph] = placeholders
        index = placeholders.index(placeholder)
        if index >= len(info.node_args):
            return None
        outer = info.node_args[index]
        return outer if isinstance(outer, torch.fx.Node) else None

    def resolve(self, value: object) -> object:
        """Strip ``_new_var`` wrappers and cross graph boundaries."""
        while isinstance(value, torch.fx.Node):
            if value.op == "placeholder":
                outer = self.outer_node(value)
                if outer is None:
                    return value
                value = outer
                continue
            if (
                value.op == "call_function"
                and value.target is _tracing_ops._new_var
                and value.args
            ):
                value = value.args[0]
                continue
            return value
        return value


@dataclasses.dataclass(frozen=True)
class GatedRowAccess:
    """One ``x[tile.index + offset, lane, :]`` access on a rank-3 host tensor."""

    name: str
    node: torch.fx.Node
    row_dim: int
    lane_dim: int
    lane_block_id: int | None
    lane_constant: int | None
    offset: torch.fx.Node | None
    # Index nodes this access consumes (for the whole-graph audit).
    index_nodes: tuple[torch.fx.Node, ...]
    # ``x[tile, :, :]``: every lane of the lane dim (the kernel batches the
    # matmuls over it); the fused body handles one lane per CTA.
    lane_all: bool = False

    @property
    def lane_key(self) -> tuple[object, ...]:
        return (self.lane_block_id, self.lane_constant, self.lane_all)


@dataclasses.dataclass(frozen=True, eq=False)
class GatedAttentionMatch:
    """Config-independent facts about a gated attention kernel."""

    root_block_ids: tuple[int, ...]
    q_block_id: int
    kv_block_id: int
    head_dim: int
    io_dtype: torch.dtype
    score_dtype: torch.dtype
    loop_graph: ForLoopGraphInfo
    if_node: torch.fx.Node | None
    if_predicate: object
    loop_node: torch.fx.Node
    kv_end: object
    q: GatedRowAccess
    k: GatedRowAccess
    v: GatedRowAccess
    o: GatedRowAccess
    score_node: torch.fx.Node
    p_node: torch.fx.Node
    # ``where`` conjuncts of the exact jagged-length form
    # ``tile_q.index < off[b + 1] - off[b]``: rows they mask belong to the next
    # sequence (or padding), so the fused body skips their store.
    store_skip_terms: tuple[torch.fx.Node, ...]
    scope: _GraphScope
    # ``root``: the query tile is the last root axis (jagged_hstu_attn).
    # ``sequence``: the root is a grid over sequences and the query tiles are
    # a device loop over ``[begin, end)`` whose bounds are scalar programs
    # (jagged_hstu_attn_2); the KV loop shares the begin.  Rows and columns
    # are absolute, the lane dim is batched (``lane_extent`` lanes).
    form: str = "root"
    q_loop_node: torch.fx.Node | None = None
    q_begin: object = 0
    q_end: object = None
    lane_extent: int | None = None

    @property
    def leading_block_ids(self) -> tuple[int, ...]:
        if self.form == "sequence":
            return self.root_block_ids
        return self.root_block_ids[:-1]

    @property
    def row_offset(self) -> torch.fx.Node | None:
        return self.q.offset


class GatedAttentionPlan(NamedTuple):
    """Config-dependent selection of the fused gated body."""

    match: GatedAttentionMatch
    kv_tile: int
    kv_stage: int
    gate_warpgroups: int = 1
    q_tile: int = GATED_Q_TILE

    @property
    def threads(self) -> int:
        """One producer warpgroup plus the gate warpgroups."""
        return 128 * (1 + self.gate_warpgroups)

    @property
    def geometry(self) -> GatedChunkGeometry:
        geometry = gated_chunk_geometry(self.q_tile, self.kv_tile, self.gate_warpgroups)
        assert geometry is not None
        return geometry


class GatedChunkGeometry(NamedTuple):
    """How a gate thread walks its share of a KV tile's score columns.

    ``chunk_cols`` elements per TMEM load chunk (the gate vector width), each
    element ``col_stride`` columns apart (1 at the 128-row tile, where a thread
    owns a whole row; 2 at the 64-row tile, where lanes ``t`` and ``t ^ 2``
    hold the even and odd columns of a row), ``chunk_iters`` chunks per KV tile
    per thread.
    """

    chunk_cols: int
    col_stride: int
    chunk_iters: int

    @property
    def chunk_span(self) -> int:
        """Score columns covered by one chunk."""
        return self.chunk_cols * self.col_stride


def gated_chunk_geometry(
    q_tile: int, kv_tile: int, gate_warpgroups: int
) -> GatedChunkGeometry | None:
    """The chunk geometry of a tile shape and gate warpgroup count, or None
    when the count does not split the thread's columns into whole chunks of at
    least ``GATED_MIN_CHUNK_COLS`` elements."""
    if q_tile not in GATED_Q_TILE_CHOICES:
        return None
    col_stride = 1 if q_tile == GATED_Q_TILE else 2
    thread_cols = kv_tile // col_stride
    chunk_cols = min(GATED_CHUNK_COLS, thread_cols // gate_warpgroups)
    if chunk_cols < GATED_MIN_CHUNK_COLS or chunk_cols % 2:
        return None
    if thread_cols % (chunk_cols * gate_warpgroups):
        return None
    return GatedChunkGeometry(
        chunk_cols, col_stride, thread_cols // (chunk_cols * gate_warpgroups)
    )


class GatedSearchSurface(NamedTuple):
    head_dim: int
    io_dtype: torch.dtype
    q_block_id: int
    kv_block_id: int
    kv_tile_choices: tuple[int, ...]
    kv_stage_choices: tuple[int, ...]
    kv_stage_default: int
    gate_warpgroup_choices: tuple[int, ...] = (1,)
    gate_warpgroup_default: int = 1
    # Ring depths that fit shared memory for each fused tile shape ((query
    # rows, KV tile), depths): a wide KV tile of fp32 rows fits fewer stages
    # than a narrow one, and the 64-row Q tile fits one more than the 128-row.
    kv_stage_choices_by_tile: tuple[tuple[tuple[int, int], tuple[int, ...]], ...] = ()
    q_tile_choices: tuple[int, ...] = (GATED_Q_TILE,)

    @property
    def block_size_targets(self) -> dict[int, int]:
        return {self.q_block_id: GATED_Q_TILE, self.kv_block_id: GATED_Q_TILE}


# ---------------------------------------------------------------------------
# Program validation (shared by the matcher and the emitter)
# ---------------------------------------------------------------------------


class _ProgramChecker:
    """Validate that a sub-graph is a supported elementwise/scalar program.

    Leaves are the score node, tile coordinates, host scalars, constants and
    scalar loads of host tensors.  Everything else must be an op from the
    supported elementwise table.  Domains record which tile coordinates each
    node depends on so the emitter can hoist row-only and CTA-only values.
    """

    def __init__(
        self,
        env: CompileEnvironment,
        scope: _GraphScope,
        *,
        score_node: torch.fx.Node | None,
        q_block_id: int,
        kv_block_id: int,
        leading_block_ids: tuple[int, ...],
    ) -> None:
        self.env = env
        self.scope = scope
        self.score_node = score_node
        self.q_block_id = q_block_id
        self.kv_block_id = kv_block_id
        self.leading_block_ids = leading_block_ids
        self.domains: dict[torch.fx.Node, frozenset[str]] = {}
        self.order: list[torch.fx.Node] = []
        self._active: set[torch.fx.Node] = set()

    def tile_index_domain(self, node: torch.fx.Node) -> frozenset[str] | None:
        block_id = _tile_index_block_id(self.env, node)
        if block_id == self.q_block_id:
            return frozenset({_Q_ROW})
        if block_id == self.kv_block_id:
            return frozenset({_KV_COL})
        return None

    def tile_scalar_domain(self, node: torch.fx.Node) -> frozenset[str] | None:
        block_id = _tile_op_block_id(self.env, node)
        if block_id == self.kv_block_id:
            return frozenset({_KV_SCALAR})
        if block_id == self.q_block_id:
            return frozenset({_Q_TILE})
        if block_id in self.leading_block_ids:
            return frozenset()
        return None

    def check(self, value: object) -> frozenset[str] | None:
        """Return the domain of ``value`` or None when unsupported."""
        value = self.scope.resolve(value)
        if isinstance(value, (bool, int, float)):
            return frozenset()
        if not isinstance(value, torch.fx.Node):
            return None
        node = value
        cached = self.domains.get(node)
        if cached is not None:
            return cached
        if node in self._active:
            return None
        self._active.add(node)
        domain = self._check_node(node)
        self._active.discard(node)
        if domain is None:
            return None
        self.domains[node] = domain
        self.order.append(node)
        return domain

    def _check_args(self, args: Iterable[object]) -> frozenset[str] | None:
        domain: frozenset[str] = frozenset()
        for arg in args:
            arg_domain = self.check(arg)
            if arg_domain is None:
                return None
            domain |= arg_domain
        return domain

    def _check_node(self, node: torch.fx.Node) -> frozenset[str] | None:
        if node.op == "placeholder":
            return None
        if node.op != "call_function":
            return None
        target = node.target
        if node is self.score_node:
            return frozenset({_Q_ROW, _KV_COL})
        if target is tile_ops.tile_index:
            return self.tile_index_domain(node)
        if target in (tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id):
            return self.tile_scalar_domain(node)
        if target is _tracing_ops._get_symnode:
            block_id = _symnode_block_id(node)
            if block_id is not None and block_id not in (
                self.q_block_id,
                self.kv_block_id,
                *self.leading_block_ids,
            ):
                return None
            val = _node_val(node)
            if not isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                return None
            return frozenset()
        if target is torch.ops.aten.scalar_tensor.default:
            return frozenset() if isinstance(node.args[0], (int, float, bool)) else None
        if target is creation_ops.full:
            fill = node.args[1] if len(node.args) > 1 else None
            return frozenset() if isinstance(fill, (int, float, bool)) else None
        if target is memory_ops.load:
            return self._check_scalar_load(node)
        if target is _tracing_ops._mask_to:
            if len(node.args) != 2 or not isinstance(node.args[1], (int, float)):
                return None
            domain = self.check(node.args[0])
            if domain is None:
                return None
            return domain | frozenset({_Q_ROW, _KV_COL})
        if target in _VIEW_TARGETS:
            if target is view_ops.subscript:
                index = node.args[1] if len(node.args) > 1 else None
                if not isinstance(index, (list, tuple)) or not all(
                    item is None or _is_full_slice(item) for item in index
                ):
                    return None
            return self.check(node.args[0])
        if target in _CONVERT_TARGETS:
            if node.kwargs and set(node.kwargs) - {"dtype"}:
                return None
            return self.check(node.args[0])
        if target is torch.ops.aten.where.self:
            if len(node.args) != 3:
                return None
            return self._check_args(node.args)
        if (
            target in _ADD_TARGETS
            or target in _SUB_TARGETS
            or target in _MUL_TARGETS
            or target in _DIV_TARGETS
            or target in _FLOORDIV_TARGETS
            or target in _MOD_TARGETS
            or target in _COMPARE_TARGETS
            or target in _AND_TARGETS
            or target in _OR_TARGETS
            or target in _MINMAX_TARGETS
        ):
            if len(node.args) != 2:
                return None
            if (
                target in _ADD_TARGETS | _SUB_TARGETS
                and node.kwargs.get("alpha", 1) != 1
            ):
                return None
            if target in _DIV_TARGETS and node.kwargs.get("rounding_mode") is not None:
                return None
            if node.kwargs and set(node.kwargs) - {"alpha", "rounding_mode"}:
                return None
            return self._check_args(node.args)
        if target in _UNARY_MATH_TARGETS:
            if len(node.args) != 1 or node.kwargs:
                return None
            return self.check(node.args[0])
        return None

    def _check_scalar_load(self, node: torch.fx.Node) -> frozenset[str] | None:
        val = _node_val(node)
        if not isinstance(val, torch.Tensor) or val.ndim != 0:
            return None
        if val.dtype not in (torch.int32, torch.int64, torch.float32):
            return None
        if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
            return None
        tensor = node.args[0]
        if (
            not isinstance(tensor, torch.fx.Node)
            or tensor.op != "call_function"
            or tensor.target is not _tracing_ops._host_tensor
        ):
            return None
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)) or not indices:
            return None
        domain = self._check_args(indices)
        if domain is None or domain:
            return None
        return frozenset()


def _flatten_bool_terms(node: object, targets: frozenset[object]) -> list[object]:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
        and len(node.args) == 2
    ):
        return _flatten_bool_terms(node.args[0], targets) + _flatten_bool_terms(
            node.args[1], targets
        )
    return [node]


def _is_zero_constant(value: object, scope: _GraphScope) -> bool:
    value = scope.resolve(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) == 0.0
    if isinstance(value, torch.fx.Node) and value.op == "call_function":
        if value.target is torch.ops.aten.scalar_tensor.default:
            return (
                isinstance(value.args[0], (int, float)) and float(value.args[0]) == 0.0
            )
        if value.target is creation_ops.full and len(value.args) > 1:
            fill = value.args[1]
            return isinstance(fill, (int, float)) and float(fill) == 0.0
    return False


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _host_tensor_name(node: object) -> str | None:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is _tracing_ops._host_tensor
        and node.args
        and isinstance(node.args[0], str)
    ):
        return node.args[0]
    return None


def _parse_row_access(
    env: CompileEnvironment,
    scope: _GraphScope,
    node: torch.fx.Node,
    *,
    row_block_id: int,
    leading_block_ids: tuple[int, ...],
) -> GatedRowAccess | None:
    """Match ``load/store(x, [tile.index + off, lane, :])`` on a rank-3 tensor."""
    if node.op != "call_function" or node.target not in (
        memory_ops.load,
        memory_ops.store,
    ):
        return None
    name = _host_tensor_name(node.args[0])
    if name is None:
        return None
    indices = node.args[1] if len(node.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None
    if node.target is memory_ops.load:
        if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
            return None
    elif len(node.args) > 3 and node.args[3] is not None:
        return None
    if not _is_full_slice(indices[2]):
        return None
    if _is_full_slice(indices[1]):
        # ``x[tile, :, :]`` on the row block: the block-size symnode indexes
        # the whole tile (rows are absolute), every lane is loaded.
        first = scope.resolve(indices[0])
        if not isinstance(first, torch.fx.Node):
            return None
        if _symnode_block_id(first) != row_block_id:
            return None
        consumed_all: list[torch.fx.Node] = [first]
        if isinstance(indices[0], torch.fx.Node) and indices[0] is not first:
            consumed_all.append(indices[0])
        return GatedRowAccess(
            name, node, 0, 1, None, None, None, tuple(consumed_all), lane_all=True
        )
    row_dim: int | None = None
    lane_dim: int | None = None
    lane_block_id: int | None = None
    lane_constant: int | None = None
    offset: torch.fx.Node | None = None
    consumed: list[torch.fx.Node] = []
    for dim in (0, 1):
        index = indices[dim]
        if isinstance(index, int) and not isinstance(index, bool):
            if lane_dim is not None:
                return None
            lane_dim = dim
            lane_constant = index
            continue
        if not isinstance(index, torch.fx.Node):
            return None
        consumed.append(index)
        resolved = scope.resolve(index)
        if not isinstance(resolved, torch.fx.Node) or resolved.op != "call_function":
            return None
        consumed.append(resolved)
        if resolved.target is tile_ops.tile_begin:
            block_id = _tile_op_block_id(env, resolved)
            if block_id is None or block_id not in leading_block_ids:
                return None
            if lane_dim is not None:
                return None
            lane_dim = dim
            lane_block_id = block_id
            continue
        row_index = resolved
        row_offset: torch.fx.Node | None = None
        if resolved.target in _ADD_TARGETS and len(resolved.args) == 2:
            lhs, rhs = (scope.resolve(arg) for arg in resolved.args)
            candidates = [(lhs, rhs), (rhs, lhs)]
            row_index = None
            for tile_part, offset_part in candidates:
                if (
                    isinstance(tile_part, torch.fx.Node)
                    and tile_part.op == "call_function"
                    and tile_part.target is tile_ops.tile_index
                    and isinstance(offset_part, torch.fx.Node)
                ):
                    row_index = tile_part
                    row_offset = offset_part
                    break
            if row_index is None:
                return None
        if (
            not isinstance(row_index, torch.fx.Node)
            or row_index.target is not tile_ops.tile_index
            or _tile_index_block_id(env, row_index) != row_block_id
        ):
            return None
        if row_dim is not None:
            return None
        row_dim = dim
        offset = row_offset
        consumed.append(row_index)
    if row_dim is None or lane_dim is None:
        return None
    return GatedRowAccess(
        name,
        node,
        row_dim,
        lane_dim,
        lane_block_id,
        lane_constant,
        offset,
        tuple(consumed),
    )


def _gated_call_nodes(graphs: Iterable[GraphInfo]) -> list[torch.fx.Node]:
    """Every call in the matched graphs; the matcher must account for each."""
    return [
        node
        for info in graphs
        for node in info.graph.nodes
        if node.op == "call_function"
    ]


def _strip_views(value: object, scope: _GraphScope) -> object:
    value = scope.resolve(value)
    while (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target in _VIEW_TARGETS
    ):
        value = scope.resolve(value.args[0])
    return value


def _is_one(value: object, scope: _GraphScope) -> bool:
    value = scope.resolve(value)
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def _leading_axis_coordinate(
    env: CompileEnvironment,
    scope: _GraphScope,
    value: object,
    leading_block_ids: tuple[int, ...],
) -> int | None:
    """Block id when ``value`` is ``tile.begin`` / ``tile.id`` of a leading axis."""
    value = scope.resolve(value)
    if (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target in (tile_ops.tile_begin, tile_ops.tile_id)
    ):
        block_id = _tile_op_block_id(env, value)
        if block_id is not None and block_id in leading_block_ids:
            return block_id
    return None


def _is_successor(
    env: CompileEnvironment,
    scope: _GraphScope,
    a: object,
    b: object,
    leading_block_ids: tuple[int, ...],
) -> bool:
    """``a == b + 1`` for a leading-axis coordinate ``b`` (``tile.begin`` or
    ``tile.id`` of a block-1 axis): an explicit ``+ 1`` or that axis's
    ``tile.end``. Any other pair (a start/end table ``off[2b]``, ``off[2b + 1]``
    for one) is not the jagged length: its gap rows must store zeros."""
    block_id = _leading_axis_coordinate(env, scope, b, leading_block_ids)
    if block_id is None:
        return False
    a = scope.resolve(a)
    if not isinstance(a, torch.fx.Node) or a.op != "call_function":
        return False
    if a.target in _ADD_TARGETS and len(a.args) == 2 and a.kwargs.get("alpha", 1) == 1:
        lhs, rhs = a.args
        return (
            _is_one(lhs, scope)
            and _leading_axis_coordinate(env, scope, rhs, leading_block_ids) == block_id
        ) or (
            _is_one(rhs, scope)
            and _leading_axis_coordinate(env, scope, lhs, leading_block_ids) == block_id
        )
    if a.target is tile_ops.tile_end:
        return _tile_op_block_id(env, a) == block_id
    return False


def _scalar_load_parts(node: object) -> tuple[str, object] | None:
    """``(tensor name, index)`` of a rank-1 scalar load ``x[i]``."""
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is memory_ops.load
    ):
        name = _host_tensor_name(node.args[0])
        indices = node.args[1] if len(node.args) > 1 else None
        if name is not None and isinstance(indices, (list, tuple)):
            if len(indices) == 1:
                return name, indices[0]
    return None


def _is_jagged_length_term(
    env: CompileEnvironment,
    scope: _GraphScope,
    term: torch.fx.Node,
    *,
    row_offset: torch.fx.Node | None,
    q_block_id: int,
    leading_block_ids: tuple[int, ...],
) -> bool:
    """``tile_q.index < off[b + 1] - off[b]`` where ``off[b]`` is the row offset."""
    if row_offset is None:
        return False
    if term.target in _LT_TARGETS:
        index_side, length_side = term.args
    elif term.target in _GT_TARGETS:
        length_side, index_side = term.args
    else:
        return False
    index_node = scope.resolve(index_side)
    if (
        not isinstance(index_node, torch.fx.Node)
        or index_node.op != "call_function"
        or index_node.target is not tile_ops.tile_index
        or _tile_index_block_id(env, index_node) != q_block_id
    ):
        return False
    length = scope.resolve(length_side)
    if (
        not isinstance(length, torch.fx.Node)
        or length.op != "call_function"
        or length.target not in _SUB_TARGETS
        or len(length.args) != 2
        or length.kwargs.get("alpha", 1) != 1
    ):
        return False
    ends, starts = (scope.resolve(arg) for arg in length.args)
    if starts is not row_offset:
        return False
    starts_parts = _scalar_load_parts(starts)
    ends_parts = _scalar_load_parts(ends)
    if starts_parts is None or ends_parts is None or starts_parts[0] != ends_parts[0]:
        return False
    return _is_successor(env, scope, ends_parts[1], starts_parts[1], leading_block_ids)


def _unwrap_permute(node: object) -> torch.fx.Node | None:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.permute.default
        and len(node.args) >= 2
        and list(node.args[1]) == [1, 0]  # pyrefly: ignore [bad-argument-type]
        and isinstance(node.args[0], torch.fx.Node)
    ):
        return node.args[0]
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.t.default
        and isinstance(node.args[0], torch.fx.Node)
    ):
        return node.args[0]
    return None


def match_gated_attention(device_ir: DeviceIR) -> GatedAttentionMatch | None:
    """Config-independent detector for the fused gated attention body (either
    form; see ``GatedAttentionMatch.form``)."""
    match = _match_root_form(device_ir)
    if match is None:
        match = _match_sequence_form(device_ir)
    return match


def _permute_source(node: object, order: list[int], scope: _GraphScope) -> object:
    """``permute(x, order)``'s ``x`` (resolved), else None."""
    node = scope.resolve(node)
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.permute.default
        and len(node.args) >= 2
        and list(node.args[1]) == order  # pyrefly: ignore [bad-argument-type]
    ):
        return scope.resolve(node.args[0])
    return None


def _match_sequence_form(device_ir: DeviceIR) -> GatedAttentionMatch | None:
    """``for s in hl.grid(B): start, end = off[s], off[s + 1]; for tile_q in
    hl.tile(start, end): ...; for tile_kv in hl.tile(start, end): bmm over the
    lane (head) dim ...; out[tile_q, :, :] = acc.transpose(0, 1)``."""
    env = CompileEnvironment.current()
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 1:
        return None
    seq_block_id = device_ir.grid_block_ids[0][0]
    graphs = tuple(device_ir.graphs)
    scope = _GraphScope(graphs)
    root_graphs = [info for info in graphs if isinstance(info, RootGraphInfo)]
    if len(root_graphs) != 1:
        return None
    root_graph = root_graphs[0]
    loop_graphs = [
        info
        for info in graphs
        if isinstance(info, ForLoopGraphInfo)
        and not isinstance(info, ReductionLoopGraphInfo)
    ]
    if len(loop_graphs) != 2 or any(len(info.block_ids) != 1 for info in loop_graphs):
        return None
    call_nodes = _gated_call_nodes(graphs)
    loop_nodes = {
        node.args[0]: node
        for node in call_nodes
        if node.target is _tracing_ops._for_loop
    }
    if len(loop_nodes) != 2:
        return None
    q_graph = next(
        (
            info
            for info in loop_graphs
            if loop_nodes.get(info.graph_id) is not None
            and loop_nodes[info.graph_id].graph is root_graph.graph
        ),
        None,
    )
    if q_graph is None:
        return None
    kv_graph = next(info for info in loop_graphs if info is not q_graph)
    q_loop_node = loop_nodes[q_graph.graph_id]
    kv_loop_node = loop_nodes.get(kv_graph.graph_id)
    if kv_loop_node is None or kv_loop_node.graph is not q_graph.graph:
        return None
    q_block_id = q_graph.block_ids[0]
    kv_block_id = kv_graph.block_ids[0]
    if len({seq_block_id, q_block_id, kv_block_id}) != 3:
        return None
    # Loop bounds: both loops run over the same [begin, end) scalars.
    bounds = []
    for loop in (q_loop_node, kv_loop_node):
        begins, ends = loop.args[1], loop.args[2]
        if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
            return None
        if len(begins) != 1 or len(ends) != 1:
            return None
        bounds.append((scope.resolve(begins[0]), scope.resolve(ends[0])))
    (q_begin, q_end), (kv_begin, kv_end) = bounds
    if q_begin is not kv_begin or q_end is not kv_end:
        return None
    if not isinstance(q_begin, torch.fx.Node) or not isinstance(q_end, torch.fx.Node):
        return None
    stores = [node for node in call_nodes if node.target is memory_ops.store]
    if len(stores) != 1 or stores[0].graph is not q_graph.graph:
        return None
    store_node = stores[0]

    # The two batched contractions: ``bmm(q_blk, k_blk^T)`` and
    # ``baddbmm(acc, P, v_blk)`` (or ``bmm`` plus ``add``).
    mm_nodes = [
        node
        for node in kv_graph.graph.nodes
        if node.op == "call_function"
        and node.target in (torch.ops.aten.bmm.default, torch.ops.aten.baddbmm.default)
    ]
    if len(mm_nodes) != 2:
        return None
    score_node: torch.fx.Node | None = None
    pv_node: torch.fx.Node | None = None
    k_load: torch.fx.Node | None = None
    v_load: torch.fx.Node | None = None
    view_nodes: list[torch.fx.Node] = []
    for node in mm_nodes:
        if node.kwargs:
            return None
        if node.target is torch.ops.aten.bmm.default:
            kt = _permute_source(node.args[1], [0, 2, 1], scope)
            if kt is not None:
                k_src = _permute_source(kt, [1, 0, 2], scope)
                if isinstance(k_src, torch.fx.Node) and score_node is None:
                    score_node, k_load = node, k_src
                    view_nodes.extend(
                        n
                        for n in (scope.resolve(node.args[1]), kt)
                        if isinstance(n, torch.fx.Node)
                    )
                    continue
        rhs = (
            node.args[2]
            if node.target is torch.ops.aten.baddbmm.default
            else node.args[1]
        )
        v_src = _permute_source(rhs, [1, 0, 2], scope)
        if not isinstance(v_src, torch.fx.Node) or pv_node is not None:
            return None
        pv_node, v_load = node, v_src
        resolved_rhs = scope.resolve(rhs)
        if isinstance(resolved_rhs, torch.fx.Node):
            view_nodes.append(resolved_rhs)
    if score_node is None or pv_node is None or k_load is None or v_load is None:
        return None
    q_src = _permute_source(score_node.args[0], [1, 0, 2], scope)
    if not isinstance(q_src, torch.fx.Node):
        return None
    q_view = scope.resolve(score_node.args[0])
    if isinstance(q_view, torch.fx.Node):
        view_nodes.append(q_view)
    q_access = _parse_row_access(
        env, scope, q_src, row_block_id=q_block_id, leading_block_ids=()
    )
    k_access = _parse_row_access(
        env, scope, k_load, row_block_id=kv_block_id, leading_block_ids=()
    )
    v_access = _parse_row_access(
        env, scope, v_load, row_block_id=kv_block_id, leading_block_ids=()
    )
    o_access = _parse_row_access(
        env, scope, store_node, row_block_id=q_block_id, leading_block_ids=()
    )
    accesses = (q_access, k_access, v_access, o_access)
    if any(access is None or not access.lane_all for access in accesses):
        return None
    accesses = cast("tuple[GatedRowAccess, ...]", accesses)
    if q_src.graph is not q_graph.graph:
        return None
    if k_load.graph is not kv_graph.graph or v_load.graph is not kv_graph.graph:
        return None
    if len({access.name for access in accesses}) != 4:
        return None
    host_tensors = _flash_graph_host_tensors(graphs)
    fakes = [host_tensors.get(access.name) for access in accesses]
    if any(fake is None for fake in fakes):
        return None
    q_fake = fakes[0]
    assert q_fake is not None
    if q_fake.ndim != 3 or q_fake.dtype not in GATED_IO_DTYPES:
        return None
    head_dim = q_fake.shape[2]
    lane_extent = q_fake.shape[1]
    if not isinstance(head_dim, int) or head_dim not in GATED_HEAD_DIMS:
        return None
    if not isinstance(lane_extent, int) or lane_extent < 1:
        return None
    for fake in fakes:
        assert fake is not None
        if (
            fake.ndim != 3
            or fake.dtype != q_fake.dtype
            or not fake.is_contiguous()
            or fake.shape[1] != lane_extent
            or fake.shape[2] != head_dim
        ):
            return None
        rows = fake.shape[0]
        if isinstance(rows, int) and rows >= 2**31:
            return None
    io_dtype = q_fake.dtype
    score_val = _node_val(score_node)
    pv_val = _node_val(pv_node)
    if not isinstance(score_val, torch.Tensor) or not isinstance(pv_val, torch.Tensor):
        return None
    if score_val.ndim != 3 or pv_val.ndim != 3:
        return None
    if score_val.shape[0] != lane_extent or pv_val.shape[0] != lane_extent:
        return None
    if env.get_block_id(score_val.shape[1]) != q_block_id:
        return None
    if env.get_block_id(score_val.shape[2]) != kv_block_id:
        return None
    if env.get_block_id(pv_val.shape[1]) != q_block_id or pv_val.shape[2] != head_dim:
        return None

    # Accumulator carried by the KV loop, seeded with zeros, permuted back and
    # stored once.
    outputs = next(node for node in kv_graph.graph.nodes if node.op == "output")
    output_values = outputs.args[0]
    if not isinstance(output_values, (list, tuple)) or len(output_values) != 1:
        return None
    acc_out = output_values[0]
    if not isinstance(acc_out, torch.fx.Node):
        return None
    if pv_node.target is torch.ops.aten.baddbmm.default:
        if acc_out is not pv_node:
            return None
        acc_carry = scope.resolve(pv_node.args[0])
        p_node = pv_node.args[1]
    else:
        if (
            acc_out.op != "call_function"
            or acc_out.target is not torch.ops.aten.add.Tensor
            or len(acc_out.args) != 2
            or acc_out.kwargs.get("alpha", 1) != 1
        ):
            return None
        if acc_out.args[0] is pv_node:
            carry_arg = acc_out.args[1]
        elif acc_out.args[1] is pv_node:
            carry_arg = acc_out.args[0]
        else:
            return None
        acc_carry = scope.resolve(carry_arg)
        p_node = pv_node.args[0]
    acc_val = _node_val(acc_out)
    if not isinstance(acc_val, torch.Tensor) or acc_val.dtype != torch.float32:
        return None
    if not isinstance(p_node, torch.fx.Node):
        return None
    if (
        not isinstance(acc_carry, torch.fx.Node)
        or acc_carry.op != "call_function"
        or acc_carry.target is not creation_ops.full
        or not _is_zero_constant(acc_carry, scope)
    ):
        return None
    loop_inputs = kv_loop_node.args[3]
    if not isinstance(loop_inputs, (list, tuple)) or acc_carry not in loop_inputs:
        return None
    store_value = store_node.args[2]
    value = store_value
    if isinstance(value, torch.fx.Node) and value.op == "call_function":
        if value.target in _CONVERT_TARGETS:
            target_dtype = (
                value.args[1] if len(value.args) > 1 else value.kwargs.get("dtype")
            )
            if target_dtype != io_dtype:
                return None
            value = value.args[0]
    permuted = _permute_source(value, [1, 0, 2], scope)
    if not isinstance(permuted, torch.fx.Node):
        return None
    store_view = scope.resolve(value)
    value = permuted
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not _tracing_ops._phi
        or len(value.args) != 2
    ):
        return None
    phi_sources = {scope.resolve(arg) for arg in value.args}
    loop_results = [
        node
        for node in q_graph.graph.nodes
        if node.op == "call_function"
        and node.target is operator.getitem
        and node.args[0] is kv_loop_node
        and node.args[1] == 0
    ]
    if len(loop_results) != 1 or phi_sources != {acc_carry, loop_results[0]}:
        return None
    value_val = _node_val(value)
    if not isinstance(value_val, torch.Tensor) or value_val.dtype != torch.float32:
        return None

    checker = _ProgramChecker(
        env,
        scope,
        score_node=score_node,
        q_block_id=q_block_id,
        kv_block_id=kv_block_id,
        leading_block_ids=(seq_block_id,),
    )
    p_domain = checker.check(p_node)
    if p_domain is None or _Q_ROW not in p_domain or _KV_COL not in p_domain:
        return None
    if score_node not in checker.domains:
        return None
    for scalar in (q_begin, q_end):
        domain = checker.check(scalar)
        if domain is None or domain:
            return None

    accounted: set[torch.fx.Node] = set(checker.domains)
    accounted.update(
        (
            q_loop_node,
            kv_loop_node,
            store_node,
            value,
            loop_results[0],
            score_node,
            pv_node,
            q_src,
            k_load,
            v_load,
            acc_out,
            acc_carry,
        )
    )
    accounted.update(view_nodes)
    if isinstance(store_value, torch.fx.Node):
        accounted.add(store_value)
    if isinstance(store_view, torch.fx.Node):
        accounted.add(store_view)
    for access in accesses:
        accounted.update(access.index_nodes)
    for node in call_nodes:
        if node in accounted or node.target in _PURE_LEAF_TARGETS:
            continue
        return None
    return GatedAttentionMatch(
        root_block_ids=(seq_block_id,),
        q_block_id=q_block_id,
        kv_block_id=kv_block_id,
        head_dim=head_dim,
        io_dtype=io_dtype,
        score_dtype=score_val.dtype,
        loop_graph=kv_graph,
        if_node=None,
        if_predicate=True,
        loop_node=kv_loop_node,
        kv_end=q_end,
        q=accesses[0],
        k=accesses[1],
        v=accesses[2],
        o=accesses[3],
        score_node=score_node,
        p_node=p_node,
        store_skip_terms=(),
        scope=scope,
        form="sequence",
        q_loop_node=q_loop_node,
        q_begin=q_begin,
        q_end=q_end,
        lane_extent=lane_extent,
    )


def _match_root_form(device_ir: DeviceIR) -> GatedAttentionMatch | None:
    """Config-independent detector for the fused gated attention body.

    Shape: a single root grid whose last axis is the query tile and whose
    leading axes are singleton (batch, head, ...) coordinates; an optional
    root-level ``_if`` guarding the body; one KV ``_for_loop`` starting at 0;
    two ``mm`` nodes (``q @ k.T`` and ``P @ v``) with an fp32 accumulator carried
    through the loop and stored once as the output.
    """
    env = CompileEnvironment.current()
    if len(device_ir.grid_block_ids) != 1:
        return None
    root_block_ids = tuple(device_ir.grid_block_ids[0])
    if len(root_block_ids) < 2:
        return None
    q_block_id = root_block_ids[-1]
    leading_block_ids = root_block_ids[:-1]
    graphs = tuple(device_ir.graphs)
    scope = _GraphScope(graphs)
    root_graphs = [info for info in graphs if isinstance(info, RootGraphInfo)]
    if len(root_graphs) != 1:
        return None
    root_graph = root_graphs[0]
    loop_graphs = [
        info
        for info in graphs
        if isinstance(info, ForLoopGraphInfo)
        and not isinstance(info, ReductionLoopGraphInfo)
    ]
    if len(loop_graphs) != 1 or len(loop_graphs[0].block_ids) != 1:
        return None
    loop_graph = loop_graphs[0]
    kv_block_id = loop_graph.block_ids[0]
    if kv_block_id in root_block_ids:
        return None

    call_nodes = _gated_call_nodes(graphs)
    loop_nodes = [
        node
        for node in call_nodes
        if node.target is _tracing_ops._for_loop and node.args[0] == loop_graph.graph_id
    ]
    if len(loop_nodes) != 1:
        return None
    loop_node = loop_nodes[0]
    body_info = scope.info_by_graph.get(loop_node.graph)
    if body_info is None:
        return None
    if_node: torch.fx.Node | None = None
    if_predicate: object = True
    if isinstance(body_info, IfGraphInfo):
        if_nodes = [
            node
            for node in call_nodes
            if node.target is _tracing_ops._if and node.args[1] == body_info.graph_id
        ]
        if len(if_nodes) != 1 or if_nodes[0].graph is not root_graph.graph:
            return None
        if_node = if_nodes[0]
        else_info = next(
            (info for info in graphs if info.graph_id == if_node.args[2]), None
        )
        if not isinstance(else_info, ElseGraphInfo) or any(
            node.op == "call_function" for node in else_info.graph.nodes
        ):
            return None
        if if_node.args[4]:
            return None
        if_predicate = if_node.args[0]
    elif body_info is not root_graph:
        return None
    stores = [node for node in call_nodes if node.target is memory_ops.store]
    if len(stores) != 1 or stores[0].graph is not body_info.graph:
        return None
    store_node = stores[0]
    # Every other call (atomics, extra loops, reductions, ...) is rejected by
    # the whole-graph audit at the end of the match.

    # KV loop bounds: begin must be the literal zero so kv tile ``i`` covers
    # columns ``[i * bn, (i + 1) * bn)``.
    begins = loop_node.args[1]
    ends = loop_node.args[2]
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    if len(begins) != 1 or len(ends) != 1:
        return None
    if not (isinstance(begins[0], int) and begins[0] == 0):
        return None
    kv_end = ends[0]

    # The two contractions.
    mm_nodes = [
        node
        for node in loop_graph.graph.nodes
        if node.op == "call_function"
        and node.target in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)
    ]
    if len(mm_nodes) != 2:
        return None
    score_node: torch.fx.Node | None = None
    pv_node: torch.fx.Node | None = None
    k_load: torch.fx.Node | None = None
    kt_node: torch.fx.Node | None = None
    v_load: torch.fx.Node | None = None
    for node in mm_nodes:
        if node.target is torch.ops.aten.mm.default:
            operand = node.args[1]
            kt = _unwrap_permute(operand)
            if kt is not None:
                assert isinstance(operand, torch.fx.Node)
                resolved = scope.resolve(kt)
                if not isinstance(resolved, torch.fx.Node):
                    return None
                if score_node is not None:
                    return None
                score_node, k_load, kt_node = node, resolved, operand
                continue
        rhs = (
            node.args[2]
            if node.target is torch.ops.aten.addmm.default
            else node.args[1]
        )
        resolved = scope.resolve(rhs)
        if not isinstance(resolved, torch.fx.Node):
            return None
        if pv_node is not None:
            return None
        pv_node, v_load = node, resolved
    if score_node is None or pv_node is None or kt_node is None:
        return None
    if k_load is None or v_load is None:
        return None
    if score_node.kwargs or pv_node.kwargs:
        return None
    q_load = scope.resolve(score_node.args[0])
    if not isinstance(q_load, torch.fx.Node):
        return None
    q_access = _parse_row_access(
        env, scope, q_load, row_block_id=q_block_id, leading_block_ids=leading_block_ids
    )
    k_access = _parse_row_access(
        env,
        scope,
        k_load,
        row_block_id=kv_block_id,
        leading_block_ids=leading_block_ids,
    )
    v_access = _parse_row_access(
        env,
        scope,
        v_load,
        row_block_id=kv_block_id,
        leading_block_ids=leading_block_ids,
    )
    o_access = _parse_row_access(
        env,
        scope,
        store_node,
        row_block_id=q_block_id,
        leading_block_ids=leading_block_ids,
    )
    if q_access is None or k_access is None or v_access is None or o_access is None:
        return None
    if q_load.graph is not body_info.graph:
        return None
    if k_load.graph is not loop_graph.graph or v_load.graph is not loop_graph.graph:
        return None
    accesses = (q_access, k_access, v_access, o_access)
    if len({access.name for access in accesses}) != 4:
        return None
    if len({access.lane_key for access in accesses}) != 1:
        return None
    if len({access.offset for access in accesses}) != 1:
        return None
    host_tensors = _flash_graph_host_tensors(graphs)
    fakes = [host_tensors.get(access.name) for access in accesses]
    if any(fake is None for fake in fakes):
        return None
    q_fake = fakes[0]
    assert q_fake is not None
    if q_fake.ndim != 3 or q_fake.dtype not in GATED_IO_DTYPES:
        return None
    head_dim = q_fake.shape[2]
    if not isinstance(head_dim, int) or head_dim not in GATED_HEAD_DIMS:
        return None
    for access, fake in zip(accesses, fakes, strict=True):
        assert fake is not None
        if (
            fake.ndim != 3
            or fake.dtype != q_fake.dtype
            or not fake.is_contiguous()
            or fake.shape[2] != head_dim
        ):
            return None
        # Row coordinates are 32-bit (TMA); element offsets are 64-bit.
        rows = fake.shape[access.row_dim]
        if isinstance(rows, int) and rows >= 2**31:
            return None
    io_dtype = q_fake.dtype
    score_val = _node_val(score_node)
    pv_val = _node_val(pv_node)
    if not isinstance(score_val, torch.Tensor) or not isinstance(pv_val, torch.Tensor):
        return None
    if score_val.ndim != 2 or pv_val.ndim != 2:
        return None
    if score_val.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if env.get_block_id(score_val.shape[0]) != q_block_id:
        return None
    if env.get_block_id(score_val.shape[1]) != kv_block_id:
        return None
    if env.get_block_id(pv_val.shape[0]) != q_block_id:
        return None
    if pv_val.shape[1] != head_dim:
        return None

    # Accumulator: ``acc = acc + pv`` (or ``addmm(acc, P, v)``) carried by the
    # loop, seeded with zeros, converted once and stored.
    outputs = next(node for node in loop_graph.graph.nodes if node.op == "output")
    output_values = outputs.args[0]
    if not isinstance(output_values, (list, tuple)) or len(output_values) != 1:
        return None
    acc_out = output_values[0]
    if not isinstance(acc_out, torch.fx.Node):
        return None
    if pv_node.target is torch.ops.aten.addmm.default:
        if acc_out is not pv_node:
            return None
        acc_carry = scope.resolve(pv_node.args[0])
        p_node = pv_node.args[1]
    else:
        if (
            acc_out.op != "call_function"
            or acc_out.target is not torch.ops.aten.add.Tensor
            or len(acc_out.args) != 2
            or acc_out.kwargs.get("alpha", 1) != 1
        ):
            return None
        if acc_out.args[0] is pv_node:
            carry_arg = acc_out.args[1]
        elif acc_out.args[1] is pv_node:
            carry_arg = acc_out.args[0]
        else:
            return None
        acc_carry = scope.resolve(carry_arg)
        p_node = pv_node.args[0]
    acc_val = _node_val(acc_out)
    if not isinstance(acc_val, torch.Tensor) or acc_val.dtype != torch.float32:
        return None
    if not isinstance(p_node, torch.fx.Node):
        return None
    if (
        not isinstance(acc_carry, torch.fx.Node)
        or acc_carry.op != "call_function"
        or acc_carry.target is not creation_ops.full
        or not _is_zero_constant(acc_carry, scope)
    ):
        return None
    # The zero accumulator must be the loop input feeding the carry.
    loop_inputs = loop_node.args[3]
    if not isinstance(loop_inputs, (list, tuple)) or acc_carry not in loop_inputs:
        return None

    # Store value: convert(_phi(zeros, loop result)) or the phi itself.
    store_value = store_node.args[2]
    value = store_value
    if isinstance(value, torch.fx.Node) and value.op == "call_function":
        if value.target in _CONVERT_TARGETS:
            target_dtype = (
                value.args[1] if len(value.args) > 1 else value.kwargs.get("dtype")
            )
            if target_dtype != io_dtype:
                return None
            value = value.args[0]
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not _tracing_ops._phi
        or len(value.args) != 2
    ):
        return None
    phi_sources = {scope.resolve(arg) for arg in value.args}
    loop_results = [
        node
        for node in body_info.graph.nodes
        if node.op == "call_function"
        and node.target is operator.getitem
        and node.args[0] is loop_node
        and node.args[1] == 0
    ]
    if len(loop_results) != 1 or phi_sources != {acc_carry, loop_results[0]}:
        return None
    value_val = _node_val(value)
    if not isinstance(value_val, torch.Tensor) or value_val.dtype != torch.float32:
        return None

    # The gate program between the score and P plus every scalar program.
    checker = _ProgramChecker(
        env,
        scope,
        score_node=score_node,
        q_block_id=q_block_id,
        kv_block_id=kv_block_id,
        leading_block_ids=leading_block_ids,
    )
    p_domain = checker.check(p_node)
    if p_domain is None:
        return None
    if _Q_ROW not in p_domain or _KV_COL not in p_domain:
        return None
    if score_node not in checker.domains:
        return None
    # The KV bound and the guard may depend on the query tile (per work item);
    # the row offset must be a CTA-level scalar.
    for scalar, allowed in (
        (kv_end, frozenset({_Q_TILE})),
        (if_predicate, frozenset({_Q_TILE})),
        (q_access.offset, frozenset()),
    ):
        if scalar is None:
            continue
        domain = checker.check(scalar)
        if domain is None or not domain <= allowed:
            return None
    # Store skipping. A ``where`` conjunct of the exact jagged-length form
    # ``tile_q.index < off[b + 1] - off[b]`` (``off[b]`` being the row offset
    # and ``b`` a leading-axis coordinate, so consecutive entries bound the
    # sequence) masks rows that belong to the next sequence or to padding; the
    # frontend's zero store there races with that sequence's own store, so the
    # fused body skips it. Every other row-only term keeps the frontend
    # semantics: the gate zeroes P, so the zero accumulator is stored.
    store_skip_terms: list[torch.fx.Node] = []
    gate = p_node
    while isinstance(gate, torch.fx.Node) and gate.op == "call_function":
        if gate.target is _tracing_ops._mask_to or gate.target in _CONVERT_TARGETS:
            gate = scope.resolve(gate.args[0])
            continue
        break
    if (
        isinstance(gate, torch.fx.Node)
        and gate.op == "call_function"
        and gate.target is torch.ops.aten.where.self
        and _is_zero_constant(gate.args[2], scope)
    ):
        for term in _flatten_bool_terms(gate.args[0], _AND_TARGETS):
            resolved = _strip_views(term, scope)
            if not isinstance(resolved, torch.fx.Node):
                continue
            if checker.domains.get(resolved) != frozenset({_Q_ROW}):
                continue
            if _is_jagged_length_term(
                env,
                scope,
                resolved,
                row_offset=q_access.offset,
                q_block_id=q_block_id,
                leading_block_ids=leading_block_ids,
            ):
                store_skip_terms.append(resolved)

    # Whole-graph audit: every call in the root, ``if`` and loop graphs must be
    # a matched structural node, part of a checked program, or a pure leaf.
    # Anything else (atomics, other stores or loops, reductions) is rejected.
    accounted: set[torch.fx.Node] = set(checker.domains)
    accounted.update(
        (
            loop_node,
            store_node,
            value,
            loop_results[0],
            score_node,
            pv_node,
            kt_node,
            q_load,
            k_load,
            v_load,
            acc_out,
            acc_carry,
        )
    )
    if isinstance(store_value, torch.fx.Node):
        accounted.add(store_value)
    if if_node is not None:
        accounted.add(if_node)
    for access in accesses:
        accounted.update(access.index_nodes)
    for node in call_nodes:
        if node in accounted or node.target in _PURE_LEAF_TARGETS:
            continue
        return None
    return GatedAttentionMatch(
        root_block_ids=root_block_ids,
        q_block_id=q_block_id,
        kv_block_id=kv_block_id,
        head_dim=head_dim,
        io_dtype=io_dtype,
        score_dtype=score_val.dtype,
        loop_graph=loop_graph,
        if_node=if_node,
        if_predicate=if_predicate,
        loop_node=loop_node,
        kv_end=kv_end,
        q=q_access,
        k=k_access,
        v=v_access,
        o=o_access,
        score_node=score_node,
        p_node=p_node,
        store_skip_terms=tuple(store_skip_terms),
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Resource legality (shared by the search surface and the detector)
# ---------------------------------------------------------------------------


def gated_smem_bytes(
    head_dim: int,
    kv_tile: int,
    kv_stage: int,
    dtype: torch.dtype,
    q_tile: int = GATED_Q_TILE,
) -> int:
    """Shared memory of the fused body: the ``q_tile``-row Q tile, the
    ``kv_stage``-deep K and V rings of ``kv_tile`` rows, and the reserve."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return (
        q_tile * head_dim * elem
        + 2 * kv_stage * kv_tile * head_dim * elem
        + _GATED_SMEM_RESERVE_BYTES
    )


def _gated_smem_capacity(env: CompileEnvironment) -> int:
    from .tcgen05_config import CuteTcgen05Config

    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
    return capacity if capacity > 0 else _GATED_DEFAULT_SMEM_CAPACITY


def gated_kv_stage_choices(
    head_dim: int,
    kv_tile: int,
    dtype: torch.dtype,
    *,
    smem_capacity: int,
    q_tile: int = GATED_Q_TILE,
) -> tuple[int, ...]:
    """K/V ring depths whose shared memory fits at this tile shape (the
    64-row Q tile leaves room for a depth the 128-row tile does not)."""
    return tuple(
        stage
        for stage in GATED_KV_STAGE_CHOICES
        if gated_smem_bytes(head_dim, kv_tile, stage, dtype, q_tile) <= smem_capacity
    )


def gated_kv_stage_default(choices: Sequence[int]) -> int:
    return 3 if 3 in choices else choices[0]


def gated_gate_warpgroup_choices(q_tile: int, kv_tile: int) -> tuple[int, ...]:
    """Gate warpgroup counts that split the tile's chunks evenly."""
    return tuple(
        g
        for g in GATED_GATE_WARPGROUP_CHOICES
        if gated_chunk_geometry(q_tile, kv_tile, g) is not None
    )


def gated_gate_warpgroup_default(choices: Sequence[int]) -> int:
    return (
        GATED_GATE_WARPGROUP_DEFAULT
        if GATED_GATE_WARPGROUP_DEFAULT in choices
        else choices[0]
    )


def _config_envelope_ok(config: Config) -> bool:
    if config.pid_type != "flat":
        return False
    if any(grouping != 1 for grouping in config.l2_groupings):
        return False
    if any(order != [*range(len(order))] for order in config.loop_orders):
        return False
    if any(thread_count != 0 for thread_count in config.num_threads):
        return False
    cute_vector_widths = config.config.get("cute_vector_widths", [])
    return not (
        isinstance(cute_vector_widths, list)
        and any(width != 1 for width in cute_vector_widths)
    )


def detect_flash_gated_search_surface(device_ir: DeviceIR) -> GatedSearchSurface | None:
    """Config-independent search-surface detector for the gated body."""
    from ..backend import _attention_flash_gate_enabled
    from ..backend import _attention_flash_supported
    from ..backend import _flash_block_sizes_reachable

    if not _attention_flash_gate_enabled() or not _attention_flash_supported():
        return None
    match = match_gated_attention(device_ir)
    if match is None:
        return None
    env = CompileEnvironment.current()
    for block_id in match.leading_block_ids:
        if not isinstance(env.block_sizes[block_id].size, int):
            return None
    targets = {match.q_block_id: GATED_Q_TILE, match.kv_block_id: GATED_Q_TILE}
    if not _flash_block_sizes_reachable(env, targets):
        return None
    q_tile_choices = tuple(
        q_tile
        for q_tile in GATED_Q_TILE_CHOICES
        if _flash_block_sizes_reachable(
            env, {match.q_block_id: q_tile, match.kv_block_id: GATED_Q_TILE}
        )
    )
    # Every KV tile width with at least one ring depth that fits shared memory
    # at some query tile height is a choice; the depth choices are validated
    # per (query rows, KV tile) shape.
    capacity = _gated_smem_capacity(env)
    by_tile = {
        (q_tile, kv_tile): gated_kv_stage_choices(
            match.head_dim,
            kv_tile,
            match.io_dtype,
            smem_capacity=capacity,
            q_tile=q_tile,
        )
        for q_tile in q_tile_choices
        for kv_tile in GATED_KV_TILE_CHOICES
    }
    by_tile = {tile: stages for tile, stages in by_tile.items() if stages}
    if not by_tile:
        return None
    union = tuple(sorted({stage for stages in by_tile.values() for stage in stages}))
    return GatedSearchSurface(
        head_dim=match.head_dim,
        io_dtype=match.io_dtype,
        q_block_id=match.q_block_id,
        kv_block_id=match.kv_block_id,
        kv_tile_choices=tuple(sorted({kv_tile for _, kv_tile in by_tile})),
        kv_stage_choices=union,
        kv_stage_default=gated_kv_stage_default(union),
        gate_warpgroup_choices=GATED_GATE_WARPGROUP_CHOICES,
        gate_warpgroup_default=GATED_GATE_WARPGROUP_DEFAULT,
        kv_stage_choices_by_tile=tuple(sorted(by_tile.items())),
        q_tile_choices=q_tile_choices,
    )


def detect_gated_attention_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> GatedAttentionPlan | None:
    """Config-dependent detector: the loop is the gated body at a fused tile shape."""
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    if not env.config_spec.cute_flash_gated_search_enabled:
        return None
    state = fn.cute_state
    if not state.attention_flash_gated_probed:
        state.attention_flash_gated_probe = match_gated_attention(
            HostFunction.current().device_ir
        )
        state.attention_flash_gated_probed = True
    match = state.attention_flash_gated_probe
    if match is None:
        return None
    trigger = match.q_block_id if match.form == "sequence" else match.kv_block_id
    if block_ids != [trigger]:
        return None
    if not _config_envelope_ok(config):
        return None
    for block_id in match.leading_block_ids:
        if env.block_sizes[block_id].from_config(config) != 1:
            return None
        if not isinstance(env.block_sizes[block_id].size, int):
            return None
    bm = env.block_sizes[match.q_block_id].from_config(config)
    bn = env.block_sizes[match.kv_block_id].from_config(config)
    if (
        not isinstance(bm, int)
        or bm not in GATED_Q_TILE_CHOICES
        or not isinstance(bn, int)
        or bn not in GATED_KV_TILE_CHOICES
    ):
        return None
    choices = gated_kv_stage_choices(
        match.head_dim,
        bn,
        match.io_dtype,
        smem_capacity=_gated_smem_capacity(env),
        q_tile=bm,
    )
    if not choices:
        return None
    requested = config.config.get(FLASH_KV_STAGE_KEY)
    if requested is None:
        kv_stage = gated_kv_stage_default(choices)
    elif isinstance(requested, int) and requested in choices:
        kv_stage = requested
    else:
        return None
    wg_choices = gated_gate_warpgroup_choices(bm, bn)
    requested_wgs = config.config.get(FLASH_GATE_WARPGROUPS_KEY)
    if requested_wgs is None:
        gate_warpgroups = gated_gate_warpgroup_default(wg_choices)
    elif isinstance(requested_wgs, int) and requested_wgs in wg_choices:
        gate_warpgroups = requested_wgs
    else:
        return None
    return GatedAttentionPlan(match, bn, kv_stage, gate_warpgroups, bm)


# ---------------------------------------------------------------------------
# Program emission
# ---------------------------------------------------------------------------


class _Value(NamedTuple):
    expr: str
    # "f" fp32, "i" int32, "b" bool, "m" column bitmask: a per-thread Int32
    # whose bit ``j`` is the truth value of a boolean per-element term for
    # column ``j`` of the current chunk (a compare between the column index
    # and a column-independent scalar, or a combination of such).
    kind: str
    # A ``GATED_CHUNK_COLS``-wide TensorSSA (one score column per lane of the
    # vector) rather than a scalar.  Per-element sections are emitted in this
    # form so conversions pair up (``cvt.rn.bf16x2.f32``) and the DSL keeps
    # the chunk as straight-line vector arithmetic.
    vec: bool = False
    # Narrow float dtypes this fp32 value is known to be exactly representable
    # in (it was rounded to them, or it is a constant that is).  Rounding to
    # one of these dtypes is the identity and is elided.
    exact: frozenset[torch.dtype] = frozenset()
    # For Int32 vectors equal to ``column index + col_offset``: the scalar
    # Int32 expression ``col_offset``.  Compares against a scalar then become
    # bitmask range terms instead of 32 per-element compares.
    col_offset: str | None = None
    # For Int32 scalars equal to ``the thread's row index + row_offset``.
    # Column compares against such values bound the active KV columns of the
    # whole query tile through the tile's last row.
    row_offset: str | None = None
    # Provenance of a float ``cond ? a : b`` whose condition is a bitmask:
    # ``(mask_terms, a, b_expr)``.  Nested selects with the same fill merge
    # into one select on the AND of their (deduplicated) mask terms.
    sel: tuple[tuple[str, ...], _Value, str] | None = None
    # For a vector rounded to a narrow float dtype: ``(dtype, the fp32
    # expression before the rounding)``.  When P is stored in that dtype the
    # store's own conversion performs the rounding, so the fp32 round trip
    # ahead of it is skipped (rounding twice to one dtype is rounding once).
    pre_round: tuple[torch.dtype, str] | None = None


# The chunk geometry of the body being emitted (set by
# ``emit_gated_flash_device_body``): the gate vectors' width and the column
# stride between a thread's consecutive elements.
_DEFAULT_GEOMETRY = GatedChunkGeometry(GATED_CHUNK_COLS, 1, 1)
_GEOMETRY: contextvars.ContextVar[GatedChunkGeometry] = contextvars.ContextVar(
    "gated_chunk_geometry", default=_DEFAULT_GEOMETRY
)


def _vshape() -> str:
    return f"({_GEOMETRY.get().chunk_cols},)"


_KIND_CUTLASS = {"f": "cutlass.Float32", "i": "cutlass.Int32", "b": "cutlass.Boolean"}
_NARROW_FLOAT_DTYPES = (torch.bfloat16, torch.float16)


def _dtype_kind(dtype: torch.dtype) -> str:
    if dtype is torch.bool:
        return "b"
    if dtype.is_floating_point:
        return "f"
    return "i"


def _exact_float_dtypes(value: float) -> frozenset[torch.dtype]:
    """Narrow dtypes in which the fp32 constant ``value`` is exactly representable."""
    value = float(torch.tensor(value, dtype=torch.float32).item())
    return frozenset(
        dtype
        for dtype in _NARROW_FLOAT_DTYPES
        if float(torch.tensor(value, dtype=dtype).float().item()) == value
    )


def _approx_rcp_flags(vec: bool) -> str:
    """``cute.math.rcp`` flags for the ``fast_math`` approximate reciprocal.

    The vector form (``nvgpu.rcp``) only exists as approx+ftz; the scalar form
    keeps the plain approximate reciprocal.
    """
    return "approx=True, ftz=True" if vec else "approx=True"


def _bcast(value: _Value) -> str:
    """Vector expression of ``value`` (scalars are broadcast to the chunk width)."""
    if value.vec:
        return value.expr
    return f"cute.full({_vshape()}, {value.expr}, {_KIND_CUTLASS[value.kind]})"


def _mask_to_bool_vec(mask_expr: str) -> str:
    """Materialize a column bitmask as a Boolean vector (bit ``j`` -> lane ``j``)."""
    return (
        f"((cute.full({_vshape()}, {mask_expr}, cutlass.Int32) & flash_bit_iota)"
        " != cutlass.Int32(0))"
    )


def _to_f(value: _Value) -> str:
    if value.kind == "f":
        return value.expr
    if value.kind == "m":
        value = _Value(_mask_to_bool_vec(value.expr), "b", True)
    if value.vec:
        if value.kind == "i":
            return f"{value.expr}.to(cutlass.Float32)"
        return (
            f"cute.where({value.expr}, cute.full({_vshape()}, 1.0, cutlass.Float32), "
            "cutlass.Float32(0.0))"
        )
    if value.kind == "i":
        return f"cutlass.Float32({value.expr})"
    return f"cutlass.Float32(cutlass.Int32({value.expr}))"


def _to_i(value: _Value) -> str:
    if value.kind == "i":
        return value.expr
    if value.kind == "m":
        value = _Value(_mask_to_bool_vec(value.expr), "b", True)
    if value.vec:
        if value.kind == "f":
            return f"{value.expr}.to(cutlass.Int32)"
        return (
            f"cute.where({value.expr}, cute.full({_vshape()}, 1, cutlass.Int32), "
            "cutlass.Int32(0))"
        )
    return f"cutlass.Int32({value.expr})"


def _to_b(value: _Value) -> str:
    if value.kind == "b":
        return value.expr
    if value.kind == "m":
        return _mask_to_bool_vec(value.expr)
    if value.vec:
        zero = "cutlass.Float32(0.0)" if value.kind == "f" else "cutlass.Int32(0)"
        return f"({value.expr} != {zero})"
    return f"({value.expr} != 0)"


def _as_kind(value: _Value, kind: str) -> _Value:
    """``value`` converted to ``kind`` (same vector-ness; a bitmask becomes a vector)."""
    if value.kind == kind:
        return value
    conv = {"f": _to_f, "i": _to_i, "b": _to_b}[kind]
    return _Value(conv(value), kind, value.vec or value.kind == "m")


_COL_COMPARE_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def _col_mask(op: str, n_expr: str) -> str:
    """Bitmask of the chunk elements ``j`` (0 <= j < chunk_cols) whose column
    ``stride * j`` (relative to the thread's chunk base) satisfies ``<op> n``.

    ``n`` is a per-thread Int32; the shifts run in 64 bits so an amount of
    ``chunk_cols`` is well defined.  With a column stride of 2 (64-row tile)
    ``2j < n`` is ``j < ceil(n / 2)``, an arithmetic shift of ``n + 1``.
    """
    geometry = _GEOMETRY.get()
    width = geometry.chunk_cols
    stride = geometry.col_stride
    assert stride in (1, 2)

    def below(bound: str) -> str:  # elements j with stride * j < bound
        if stride == 2:
            bound = f"(({bound} + cutlass.Int32(1)) >> cutlass.Int32(1))"
        clamped = f"cutlass.min(cutlass.max({bound}, cutlass.Int32(0)), cutlass.Int32({width}))"
        return (
            f"cutlass.Int32((cutlass.Int64(1) << cutlass.Int64({clamped}))"
            " - cutlass.Int64(1))"
        )

    if op == "<":
        return below(n_expr)
    if op == "<=":
        return below(f"({n_expr} + cutlass.Int32(1))")
    if op == ">":
        return f"(~{below(f'({n_expr} + cutlass.Int32(1))')})"
    if op == ">=":
        return f"(~{below(n_expr)})"
    if stride == 1:
        in_range = (
            f"({n_expr} >= cutlass.Int32(0)) & ({n_expr} < cutlass.Int32({width}))"
        )
        index = n_expr
    else:
        in_range = (
            f"({n_expr} >= cutlass.Int32(0)) & ({n_expr} < cutlass.Int32({2 * width}))"
            f" & (({n_expr} & cutlass.Int32(1)) == cutlass.Int32(0))"
        )
        index = f"({n_expr} >> cutlass.Int32(1))"
    bit = (
        f"cutlass.select_({in_range}, "
        f"cutlass.Int32(cutlass.Int64(1) << cutlass.Int64(cutlass.min(cutlass.max({index}, "
        f"cutlass.Int32(0)), cutlass.Int32({width - 1})))), cutlass.Int32(0))"
    )
    if op == "==":
        return bit
    assert op == "!="
    return f"(~{bit})"


def _and_or(a: _Value, b: _Value, symbol: str) -> _Value:
    """Boolean ``a & b`` / ``a | b`` keeping bitmasks as bitmasks where possible."""
    if a.kind == "m" and b.kind == "m":
        return _Value(f"({a.expr} {symbol} {b.expr})", "m")
    for mask, other in ((a, b), (b, a)):
        if mask.kind == "m" and other.kind == "b" and not other.vec:
            if symbol == "&":
                return _Value(
                    f"cutlass.select_({other.expr}, {mask.expr}, cutlass.Int32(0))", "m"
                )
            return _Value(
                f"cutlass.select_({other.expr}, cutlass.Int32(-1), {mask.expr})", "m"
            )
    a = _as_kind(a, "b")
    b = _as_kind(b, "b")
    if a.vec or b.vec:
        return _Value(f"({_bcast(a)} {symbol} {_bcast(b)})", "b", True)
    return _Value(f"({a.expr} {symbol} {b.expr})", "b")


def _round_to_dtype(value: _Value, dtype: torch.dtype) -> _Value:
    """Round ``value`` to the frontend dtype of the node.

    Rounding to a narrow float dtype the value is already exact in (a value
    produced by that same rounding, a select between two such values, or a
    representable constant) is the identity and is not emitted.
    """
    if dtype in _NARROW_FLOAT_DTYPES:
        if dtype in value.exact and value.kind == "f":
            return value
        cls = "cutlass.BFloat16" if dtype is torch.bfloat16 else "cutlass.Float16"
        fexpr = _to_f(value)
        if value.vec or value.kind == "m":
            return _Value(
                f"_helion_flash_rt.gated_round_vec({fexpr}, {cls})",
                "f",
                True,
                frozenset({dtype}),
                pre_round=(dtype, fexpr),
            )
        return _Value(
            f"cutlass.Float32({cls}({fexpr}))", "f", False, frozenset({dtype})
        )
    if dtype in (torch.float32, torch.float64):
        if value.kind == "f":
            return value
        return _Value(_to_f(value), "f", value.vec)
    if dtype is torch.bool:
        return _as_kind(value, "b")
    return _as_kind(value, "i")


def _select(cond: _Value, a: _Value, b: _Value, kind: str) -> _Value:
    """``cond ? a : b`` in ``kind``; exactness is the intersection of the branches.

    A bitmask condition is materialized once per select; ``m1 ? (m2 ? x : c) : c``
    folds into ``(m1 & m2) ? x : c`` so a chain of masks costs one materialization.
    """
    a = _as_kind(a, kind)
    b = _as_kind(b, kind)
    exact = a.exact & b.exact if kind == "f" else frozenset()
    if cond.kind == "m":
        terms = [cond.expr]
        while kind == "f" and a.sel is not None and a.sel[2] == b.expr and not b.vec:
            terms.extend(term for term in a.sel[0] if term not in terms)
            a = a.sel[1]
        mask = " & ".join(terms) if len(terms) == 1 else f"({' & '.join(terms)})"
        return _Value(
            f"cute.where({_mask_to_bool_vec(mask)}, {_bcast(a)}, {b.expr})",
            kind,
            True,
            exact,
            sel=(tuple(terms), a, b.expr) if kind == "f" and not b.vec else None,
        )
    if cond.vec or a.vec or b.vec:
        # ``cute.where`` broadcasts a scalar branch itself; the first branch
        # is made a vector so at least one branch is one.
        return _Value(
            f"cute.where({_bcast(_as_kind(cond, 'b'))}, {_bcast(a)}, {b.expr})",
            kind,
            True,
            exact,
        )
    return _Value(
        f"cutlass.select_({_to_b(cond)}, {a.expr}, {b.expr})", kind, False, exact
    )


class _GatedEmitter:
    """Replay the captured FX programs as CuTe scalar statements.

    Statements are collected into four sections ordered by dependency:
    ``cta`` (per CTA, before the roles), ``row`` (per gate thread, before the
    KV loop), ``kviter`` (per KV tile) and ``elem`` (per score element).
    """

    def __init__(self, df: DeviceFunction, plan: GatedAttentionPlan) -> None:
        self.df = df
        self.plan = plan
        self.match = plan.match
        self.env = CompileEnvironment.current()
        self.scope = plan.match.scope
        self.checker = _ProgramChecker(
            self.env,
            self.scope,
            score_node=plan.match.score_node,
            q_block_id=plan.match.q_block_id,
            kv_block_id=plan.match.kv_block_id,
            leading_block_ids=plan.match.leading_block_ids,
        )
        self.sections: dict[str, list[str]] = {
            "cta": [],
            "item": [],
            "row": [],
            "kviter": [],
            "elem": [],
        }
        # Structure of the bound column bitmasks, for the KV range analysis:
        # ("cmp", op, other, col_offset) | ("and", a, b) | ("gate", inner).
        self.mask_info: dict[str, tuple[object, ...]] = {}
        # Section each bound name was emitted in (``cta``/``item`` names are
        # uniform over the work item; ``row``/``kviter``/``elem`` are not).
        self.value_section: dict[str, str] = {}
        self.final_mask_terms: tuple[str, ...] = ()
        self.values: dict[torch.fx.Node, _Value] = {}
        self.counter = 0
        self.tensor_names: set[str] = set()
        # Under the existing ``fast_math`` setting the gate uses the hardware
        # ex2 without the denormal fixup and an approximate reciprocal for
        # divisions (Triton's div.full contract); the default is IEEE.
        self.fast_math = bool(self.env.settings.fast_math)

    # -- helpers -----------------------------------------------------------

    def _section_for(self, domain: frozenset[str]) -> str:
        if _KV_COL in domain or (_Q_ROW in domain and _KV_SCALAR in domain):
            return "elem"
        if _KV_SCALAR in domain:
            return "kviter"
        if _Q_ROW in domain:
            return "row"
        if _Q_TILE in domain:
            return "item"
        return "cta"

    @property
    def sequence_form(self) -> bool:
        return self.match.form == "sequence"

    def _abs(self, relative: str) -> str:
        """Absolute row/column coordinate of a tile-relative one."""
        if self.sequence_form:
            return f"(flash_row_base + {relative})"
        return relative

    def _bind(self, section: str, value: _Value) -> _Value:
        name = f"flash_g{self.counter}"
        self.counter += 1
        self.value_section[name] = section
        # Per-element values are ``GATED_CHUNK_COLS``-wide TensorSSA vectors:
        # one statement per node keeps the chunk a straight-line block of
        # vector arithmetic (the IEEE division stays per element inside it).
        self.sections[section].append(f"{name} = {value.expr}")
        return _Value(
            name,
            value.kind,
            value.vec,
            value.exact,
            value.col_offset,
            value.row_offset,
            value.sel,
            value.pre_round,
        )

    def item_uniform(self, value: _Value) -> bool:
        """Whether a scalar is uniform over the work item (a CTA/item-level
        program), as opposed to per gate thread (row) or per KV tile/chunk."""
        if value.vec or "flash_gate_row" in value.expr or "flash_kv" in value.expr:
            return False
        return all(
            self.value_section.get(name, "cta") in ("cta", "item")
            for name in _GATED_NODE_NAME_RE.findall(value.expr)
        )

    def block_size_literal(self, block_id: int) -> int:
        if block_id == self.match.q_block_id:
            return self.plan.q_tile
        if block_id == self.match.kv_block_id:
            return self.plan.kv_tile
        return 1

    def extent_expr(self, block_id: int) -> str:
        size = self.env.block_sizes[block_id].size
        if isinstance(size, int):
            return str(size)
        assert size is not None
        return self.df.sympy_expr(size._sympy_())  # pyrefly: ignore [missing-attribute]

    def q_extent_expr(self) -> str:
        return f"cutlass.Int32({self.extent_expr(self.match.q_block_id)})"

    def tile_scalar_expr(self, target: object, block_id: int) -> _Value:
        if block_id == self.match.kv_block_id:
            if target is tile_ops.tile_begin:
                return _Value(self._abs("flash_kv_col0"), "i")
            if target is tile_ops.tile_end:
                return _Value(self._abs("flash_kv_col_end"), "i")
            return _Value("flash_kv", "i")
        if block_id == self.match.q_block_id:
            if target is tile_ops.tile_begin:
                return _Value(self._abs("flash_q_row0"), "i")
            if target is tile_ops.tile_end:
                return _Value(self._abs("flash_q_row_end"), "i")
            return _Value("flash_m_tile", "i")
        axis = self.match.leading_block_ids.index(block_id)
        if target is tile_ops.tile_end:
            # Leading axes run at block size 1, so ``end == begin + 1``.
            return _Value(f"(flash_axis{axis} + cutlass.Int32(1))", "i")
        return _Value(f"flash_axis{axis}", "i")

    # -- emission ----------------------------------------------------------

    def emit(self, value: object) -> _Value:
        value = self.scope.resolve(value)
        if isinstance(value, bool):
            return _Value(f"cutlass.Boolean({value})", "b")
        if isinstance(value, int):
            return _Value(f"cutlass.Int32({value})", "i")
        if isinstance(value, float):
            return _Value(
                f"cutlass.Float32({value!r})", "f", False, _exact_float_dtypes(value)
            )
        assert isinstance(value, torch.fx.Node), value
        node = value
        cached = self.values.get(node)
        if cached is not None:
            return cached
        domain = self.checker.check(node)
        assert domain is not None, f"unsupported gate node {node.format_node()}"
        result = self._emit_node(node, domain)
        self.values[node] = result
        return result

    def _const(self, node: torch.fx.Node, value: object) -> _Value:
        val = _node_val(node)
        dtype = val.dtype if isinstance(val, torch.Tensor) else None
        if dtype is None:
            return self.emit(value)
        kind = _dtype_kind(dtype)
        if kind == "b":
            return _Value(f"cutlass.Boolean({bool(value)})", "b")
        if kind == "i":
            return _Value(f"cutlass.Int32({int(value)})", "i")  # pyrefly: ignore [bad-argument-type]
        # Constants are rounded to the node dtype at codegen time (round to
        # nearest even, like the device conversion would).
        rounded = float(torch.tensor(float(value), dtype=dtype).float().item())  # pyrefly: ignore [bad-argument-type]
        return _Value(
            f"cutlass.Float32({rounded!r})", "f", False, _exact_float_dtypes(rounded)
        )

    def _node_result_dtype(self, node: torch.fx.Node) -> torch.dtype | None:
        val = _node_val(node)
        if isinstance(val, torch.Tensor):
            return val.dtype
        if isinstance(val, torch.SymBool):
            return torch.bool
        if isinstance(val, torch.SymFloat):
            return torch.float32
        if isinstance(val, torch.SymInt):
            return torch.int32
        if isinstance(val, bool):
            return torch.bool
        if isinstance(val, int):
            return torch.int32
        if isinstance(val, float):
            return torch.float32
        return None

    def _emit_node(self, node: torch.fx.Node, domain: frozenset[str]) -> _Value:
        target = node.target
        match = self.match
        section = self._section_for(domain)
        if node is match.score_node:
            base = _Value("flash_s_v", "f", True)
            return self._bind("elem", _round_to_dtype(base, match.score_dtype))
        if target is tile_ops.tile_index:
            block_id = _tile_index_block_id(self.env, node)
            if block_id == match.q_block_id:
                if self.sequence_form:
                    return _Value(
                        "(flash_row_base + flash_gate_row)",
                        "i",
                        row_offset="flash_row_base",
                    )
                return _Value("flash_gate_row", "i", row_offset="cutlass.Int32(0)")
            if self.sequence_form:
                return _Value(
                    "(flash_gate_col_v + flash_row_base)",
                    "i",
                    True,
                    col_offset="flash_row_base",
                )
            return _Value("flash_gate_col_v", "i", True, col_offset="cutlass.Int32(0)")
        if target in (tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id):
            block_id = _tile_op_block_id(self.env, node)
            assert block_id is not None
            return self.tile_scalar_expr(target, block_id)
        if target is _tracing_ops._get_symnode:
            block_id = _symnode_block_id(node)
            if block_id is not None:
                return _Value(
                    f"cutlass.Int32({self.block_size_literal(block_id)})", "i"
                )
            val = _node_val(node)
            assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool))
            if isinstance(val, torch.SymInt):
                axis_block = self.env.get_block_id(val)
                if axis_block in match.leading_block_ids:
                    axis = match.leading_block_ids.index(axis_block)
                    return _Value(f"flash_axis{axis}", "i")
            expr = self.df.sympy_expr(cast("sympy.Expr", val.node.expr))
            if isinstance(val, torch.SymFloat):
                return self._bind("cta", _Value(f"cutlass.Float32({expr})", "f"))
            if isinstance(val, torch.SymBool):
                return self._bind("cta", _Value(f"cutlass.Boolean({expr})", "b"))
            return self._bind("cta", _Value(f"cutlass.Int32({expr})", "i"))
        if target is torch.ops.aten.scalar_tensor.default:
            return self._const(node, node.args[0])
        if target is creation_ops.full:
            return self._const(node, node.args[1])
        if target is memory_ops.load:
            return self._emit_scalar_load(node)
        if target is _tracing_ops._mask_to:
            inner = self.emit(node.args[0])
            other = node.args[1]
            dtype = self._node_result_dtype(node) or torch.float32
            in_tile = _Value("flash_elem_in_tile_m", "m")
            if inner.kind == "f":
                fill = _Value(
                    f"cutlass.Float32({float(other)!r})",  # pyrefly: ignore [bad-argument-type]
                    "f",
                    False,
                    _exact_float_dtypes(float(other)),  # pyrefly: ignore [bad-argument-type]
                )
                return self._bind(
                    "elem", _round_to_dtype(_select(in_tile, inner, fill, "f"), dtype)
                )
            fill = _Value(f"cutlass.Int32({int(other)})", "i")  # pyrefly: ignore [bad-argument-type]
            return self._bind("elem", _select(in_tile, inner, fill, "i"))
        if target in _VIEW_TARGETS:
            return self.emit(node.args[0])
        if target in _CONVERT_TARGETS:
            inner = self.emit(node.args[0])
            dtype = self._node_result_dtype(node)
            assert dtype is not None
            return self._bind(section, _round_to_dtype(inner, dtype))
        if target is torch.ops.aten.where.self:
            cond = self.emit(node.args[0])
            a = self.emit(node.args[1])
            b = self.emit(node.args[2])
            dtype = self._node_result_dtype(node) or torch.float32
            kind = _dtype_kind(dtype)
            selected = _select(cond, a, b, kind)
            if kind == "f":
                return self._bind(section, _round_to_dtype(selected, dtype))
            return self._bind(section, selected)
        if target in _COMPARE_TARGETS:
            op = _COMPARE_TARGETS[target]
            a = self.emit(node.args[0])
            b = self.emit(node.args[1])
            for col, other, cop in ((a, b, op), (b, a, _COL_COMPARE_FLIP[op])):
                if (
                    col.col_offset is not None
                    and other.kind in ("i", "b")
                    and not other.vec
                ):
                    # ``col + off <op> s``  <=>  ``stride * j <op> s - off - chunk_base``
                    n_expr = f"({_to_i(other)} - {col.col_offset} - flash_chunk_base)"
                    bound = self._bind(section, _Value(_col_mask(cop, n_expr), "m"))
                    self.mask_info[bound.expr] = ("cmp", cop, other, col.col_offset)
                    return bound
            kind = "f" if (a.kind == "f" or b.kind == "f") else "i"
            return self._bind(section, self._binary_expr(a, b, op, kind, "b"))
        if target in _AND_TARGETS or target in _OR_TARGETS:
            a = self.emit(node.args[0])
            b = self.emit(node.args[1])
            symbol = "&" if target in _AND_TARGETS else "|"
            if a.kind in ("b", "m") and b.kind in ("b", "m"):
                bound = self._bind(section, _and_or(a, b, symbol))
                if bound.kind == "m" and symbol == "&":
                    if a.kind == "m" and b.kind == "m":
                        self.mask_info[bound.expr] = ("and", a.expr, b.expr)
                    else:
                        self.mask_info[bound.expr] = (
                            "gate",
                            a.expr if a.kind == "m" else b.expr,
                        )
                return bound
            return self._bind(section, self._binary_expr(a, b, symbol, "i", "i"))
        if target in _MINMAX_TARGETS:
            op = _MINMAX_TARGETS[target]
            a = self.emit(node.args[0])
            b = self.emit(node.args[1])
            dtype = self._node_result_dtype(node) or torch.float32
            kind = "f" if _dtype_kind(dtype) == "f" else "i"
            cond = self._binary_expr(a, b, op, kind, "b")
            selected = _select(cond, a, b, kind)
            if kind == "f":
                return self._bind(section, _round_to_dtype(selected, dtype))
            return self._bind(section, selected)
        if (
            target in _ADD_TARGETS
            or target in _SUB_TARGETS
            or target in _MUL_TARGETS
            or target in _DIV_TARGETS
            or target in _FLOORDIV_TARGETS
            or target in _MOD_TARGETS
        ):
            return self._emit_binary(node, section)
        if target in _UNARY_MATH_TARGETS:
            return self._emit_unary(node, _UNARY_MATH_TARGETS[target], section)
        raise AssertionError(f"unsupported gate node {node.format_node()}")

    @staticmethod
    def _binary_expr(
        a: _Value, b: _Value, symbol: str, kind: str, out_kind: str
    ) -> _Value:
        """``a <symbol> b`` with both operands converted to ``kind``."""
        a = _as_kind(a, kind)
        b = _as_kind(b, kind)
        vec = a.vec or b.vec
        if vec:
            # Broadcast the scalar side explicitly: TensorSSA operators accept
            # scalar operands, but a scalar on the left would dispatch to the
            # scalar type's operator first.
            return _Value(f"({_bcast(a)} {symbol} {_bcast(b)})", out_kind, True)
        return _Value(f"({a.expr} {symbol} {b.expr})", out_kind, False)

    def _emit_binary(self, node: torch.fx.Node, section: str) -> _Value:
        target = node.target
        a = self.emit(node.args[0])
        b = self.emit(node.args[1])
        dtype = self._node_result_dtype(node)
        assert dtype is not None
        kind = _dtype_kind(dtype)
        if target in _ADD_TARGETS:
            symbol = "+"
        elif target in _SUB_TARGETS:
            symbol = "-"
        elif target in _MUL_TARGETS:
            symbol = "*"
        elif target in _DIV_TARGETS:
            symbol = "/"
        elif target in _FLOORDIV_TARGETS:
            symbol = "//"
        else:
            symbol = "%"
        if target in _MOD_TARGETS:
            return self._emit_floor_mod(a, b, kind, dtype, section)
        if target in _DIV_TARGETS and self.fast_math:
            fa, fb = _as_kind(a, "f"), _as_kind(b, "f")
            rcp = _Value(
                f"cute.math.rcp({fb.expr}, {_approx_rcp_flags(fb.vec)})", "f", fb.vec
            )
            expr = self._binary_expr(fa, rcp, "*", "f", "f")
            return self._bind(section, _round_to_dtype(expr, dtype))
        if kind == "f" or target in _DIV_TARGETS:
            expr = self._binary_expr(a, b, symbol, "f", "f")
            return self._bind(section, _round_to_dtype(expr, dtype))
        if kind == "b":
            expr = self._binary_expr(a, b, symbol, "i", "i")
            return self._bind(section, _as_kind(expr, "b"))
        expr = self._binary_expr(a, b, symbol, "i", "i")
        if symbol in ("+", "-"):
            # ``(col + off) + s`` / ``(col + off) - s`` / ``s + (col + off)`` stay
            # affine in the column index; likewise for the row index.
            if a.col_offset is not None and not b.vec and b.kind in ("i", "b"):
                expr = expr._replace(col_offset=f"({a.col_offset} {symbol} {_to_i(b)})")
            elif (
                symbol == "+"
                and b.col_offset is not None
                and not a.vec
                and a.kind in ("i", "b")
            ):
                expr = expr._replace(col_offset=f"({b.col_offset} + {_to_i(a)})")
            # ``row + s`` is affine in the row only for an item-uniform ``s``
            # (the offset is read at item level for the KV bound).
            if (
                a.row_offset is not None
                and b.kind in ("i", "b")
                and self.item_uniform(b)
            ):
                expr = expr._replace(row_offset=f"({a.row_offset} {symbol} {_to_i(b)})")
            elif (
                symbol == "+"
                and b.row_offset is not None
                and a.kind in ("i", "b")
                and self.item_uniform(a)
            ):
                expr = expr._replace(row_offset=f"({b.row_offset} + {_to_i(a)})")
        return self._bind(section, expr)

    def kv_end_bounds(self) -> list[str]:
        """Tile-relative exclusive column bounds implied by the final P mask.

        A term ``col + off < s`` (or ``<=``) zeroes every column at or past
        ``s - off`` (``+ 1``) for the row it belongs to; when ``s`` is a scalar,
        or affine in the thread's row (``row + r``: the tile's last row gives
        the loosest bound), the KV tiles past that column contribute nothing
        to any row of the tile and are skipped.  Row-only gates and ANDs only
        tighten a mask, so their inner terms count; ORs do not.
        """
        bounds: list[str] = []
        todo = list(self.final_mask_terms)
        seen: set[str] = set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            info = self.mask_info.get(name)
            if info is None:
                continue
            if info[0] == "and":
                todo.extend(cast("tuple[str, str]", info[1:3]))
            elif info[0] == "gate":
                todo.append(cast("str", info[1]))
            else:
                _, op, other, col_offset = info
                other = cast("_Value", other)
                if op not in ("<", "<="):
                    continue
                if other.row_offset is not None:
                    s_expr = (
                        f"((flash_q_row_end - cutlass.Int32(1)) + {other.row_offset})"
                    )
                elif not self.item_uniform(other):
                    # A per-row (or per-tile) bound that is not ``row +/- c``
                    # cannot be read at item level: no KV bound from this term.
                    continue
                else:
                    s_expr = _to_i(other)
                plus = " + cutlass.Int32(1)" if op == "<=" else ""
                bounds.append(f"({s_expr}{plus} - {col_offset})")
        return bounds

    def finish_elem_section(self, p_value: _Value) -> str:
        """Close the per-element program; returns the P expression for the chunk.

        P is the gate value where the element is inside the tile (rows past the
        tile and columns past kv_end contribute nothing, whatever the gate
        computes), else 0.  When that final select's condition is a column
        bitmask (a function of tile coordinates only, never of the scores),
        the masked elements' scores are replaced by 1.0 before the gate
        program runs: their gate values are discarded by the select, so the
        result is unchanged, and the substitution keeps zero-filled rows (TMA
        fill past the tensor end gives S = 0, hence a zero dividend) off the
        IEEE division's slow path, which otherwise serializes the whole warp.
        The bitmask statements the select needs are hoisted ahead of the
        first use of the scores; they depend on scalars only.
        """
        zero = _Value("cutlass.Float32(0.0)", "f", False, _exact_float_dtypes(0.0))
        in_tile = _Value("flash_elem_in_tile_m", "m")
        selected = _select(in_tile, _as_kind(p_value, "f"), zero, "f")
        elem = self.sections["elem"]
        if selected.sel is None or not self.values.get(self.match.score_node):
            elem.insert(0, "flash_s_v = flash_s_raw_v")
            return _bcast(selected)
        mask_terms = selected.sel[0]
        self.final_mask_terms = tuple(mask_terms)
        # Hoist the mask statements (and what they reference) above the score.
        defs = {line.split(" = ", 1)[0].strip(): line for line in elem}
        needed: list[str] = []
        todo = [term for term in mask_terms if term in defs]
        while todo:
            name = todo.pop()
            if name in needed:
                continue
            needed.append(name)
            todo.extend(
                ref
                for ref in re.findall(r"flash_g\d+", defs[name].split(" = ", 1)[1])
                if ref in defs and ref not in needed
            )
        hoisted = [line for line in elem if line.split(" = ", 1)[0].strip() in needed]
        rest = [line for line in elem if line not in hoisted]
        mask = (
            " & ".join(mask_terms)
            if len(mask_terms) == 1
            else f"({' & '.join(mask_terms)})"
        )
        # The mask as an Int32 vector of all-ones / zero lanes: both selects
        # are then one 3-input bit select per element (``LOP3``) on the fp32
        # patterns, instead of a Boolean vector that the compiler keeps live as
        # a bitmask (two inserts per element) and re-tests at the second select.
        vshape = _vshape()
        ones = f"cute.full({vshape}, cutlass.Int32(-1), cutlass.Int32)"
        zeros = f"cute.full({vshape}, cutlass.Int32(0), cutlass.Int32)"
        one_f = f"cute.full({vshape}, cutlass.Int32({0x3F800000}), cutlass.Int32)"
        self.sections["elem"] = [
            *hoisted,
            f"flash_p_keep_v = cute.where({_mask_to_bool_vec(mask)}, {ones}, {zeros})",
            (
                "flash_s_v = ((flash_s_raw_v.bitcast(cutlass.Int32) & flash_p_keep_v)"
                f" | ((flash_p_keep_v ^ {ones}) & {one_f})).bitcast(cutlass.Float32)"
            ),
            *rest,
        ]
        a = selected.sel[1]
        # A final bf16/fp16 rounding is performed by the P store's conversion
        # to the io dtype; the fp32 round trip ahead of it is skipped.
        a_expr = _bcast(a)
        if a.pre_round is not None and a.pre_round[0] == self.match.io_dtype:
            a_expr = a.pre_round[1]
        return f"({a_expr}.bitcast(cutlass.Int32) & flash_p_keep_v).bitcast(cutlass.Float32)"

    def _emit_floor_mod(
        self, a: _Value, b: _Value, kind: str, dtype: torch.dtype, section: str
    ) -> _Value:
        """torch/Python remainder (sign of the divisor) from the DSL's truncating ``%``."""
        op_kind = "f" if kind == "f" else "i"
        zero = _Value(
            "cutlass.Float32(0.0)" if op_kind == "f" else "cutlass.Int32(0)", op_kind
        )
        a = _as_kind(a, op_kind)
        b = _as_kind(b, op_kind)
        rem = self._bind(section, self._binary_expr(a, b, "%", op_kind, op_kind))
        neg_rem = self._binary_expr(rem, zero, "<", op_kind, "b")
        pos_rem = self._binary_expr(rem, zero, ">", op_kind, "b")
        pos_b = self._binary_expr(b, zero, ">", op_kind, "b")
        neg_b = self._binary_expr(b, zero, "<", op_kind, "b")
        fix = self._binary_expr(
            self._binary_expr(neg_rem, pos_b, "&", "b", "b"),
            self._binary_expr(pos_rem, neg_b, "&", "b", "b"),
            "|",
            "b",
            "b",
        )
        expr = _select(
            fix, self._binary_expr(rem, b, "+", op_kind, op_kind), rem, op_kind
        )
        if kind == "f":
            return self._bind(section, _round_to_dtype(expr, dtype))
        if kind == "b":
            return self._bind(section, _as_kind(expr, "b"))
        return self._bind(section, expr)

    def _exp_feeds_add_one(self, node: torch.fx.Node) -> bool:
        """Whether ``exp(...)`` is consumed only by ``e + c`` with ``|c| >= 2**-100``.

        At such a site the ex2 denormal handling is dead: a flushed result
        (``e < 2**-126``) and the unflushed one both round ``c + e`` to ``c``
        (``ulp(c) / 2 > 2**-125``), and a denormal argument gives exactly 1.0
        either way.  The result is therefore bit-identical with the
        ``ex2.approx.ftz`` form, which saves the range check and two scalings.
        """
        users = list(node.users)
        if len(users) != 1:
            return False
        user = users[0]
        if user.target not in _ADD_TARGETS or len(user.args) != 2:
            return False
        other = user.args[1] if user.args[0] is node else user.args[0]
        other = self.scope.resolve(other)
        if isinstance(other, torch.fx.Node):
            if other.target is torch.ops.aten.scalar_tensor.default:
                other = other.args[0]
            elif other.target is creation_ops.full:
                other = other.args[1]
            else:
                return False
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            return False
        return abs(float(other)) >= 2.0**-100

    def _exp2(self, x: _Value, *, ftz_exact: bool) -> _Value:
        """``2**x`` in fp32; ``ftz_exact`` marks a site where the flush-to-zero
        form is provably identical (see ``_exp_feeds_add_one``)."""
        fm = ", fastmath=True" if (self.fast_math or ftz_exact) else ""
        return _Value(f"cute.math.exp2({x.expr}{fm})", "f", x.vec)

    def _emit_unary(self, node: torch.fx.Node, op: str, section: str) -> _Value:
        inner = self.emit(node.args[0])
        dtype = self._node_result_dtype(node) or torch.float32
        if op == "neg":
            if _dtype_kind(dtype) == "i":
                iv = _as_kind(inner, "i")
                return self._bind(section, _Value(f"(-{iv.expr})", "i", iv.vec))
            fv = _as_kind(inner, "f")
            return self._bind(
                section, _round_to_dtype(_Value(f"(-{fv.expr})", "f", fv.vec), dtype)
            )
        x = _as_kind(inner, "f")
        vec = x.vec
        one = _Value("cutlass.Float32(1.0)", "f")

        def sigmoid() -> _Value:
            # The exp2 feeds ``1 + e`` directly, so the ftz form is exact here.
            e = self._exp2(
                _Value(f"((-({x.expr})) * {_LOG2_E!r})", "f", vec), ftz_exact=True
            )
            denom = self._binary_expr(e, one, "+", "f", "f")
            if self.fast_math:
                return _Value(
                    f"cute.math.rcp({denom.expr}, {_approx_rcp_flags(vec)})", "f", vec
                )
            return _Value(f"cute.math.rcp({denom.expr})", "f", vec)

        if op == "exp":
            expr = self._exp2(
                _Value(f"(({x.expr}) * {_LOG2_E!r})", "f", vec),
                ftz_exact=self._exp_feeds_add_one(node),
            )
        elif op == "exp2":
            expr = self._exp2(x, ftz_exact=self._exp_feeds_add_one(node))
        elif op == "sigmoid":
            expr = sigmoid()
        elif op == "silu":
            expr = self._binary_expr(x, sigmoid(), "*", "f", "f")
        elif op == "tanh":
            expr = _Value(f"cute.math.tanh({x.expr})", "f", vec)
        elif op == "relu":
            zero = _Value("cutlass.Float32(0.0)", "f", False, _exact_float_dtypes(0.0))
            expr = _select(self._binary_expr(x, zero, ">", "f", "b"), x, zero, "f")
        elif op == "sqrt":
            expr = _Value(f"cute.math.sqrt({x.expr})", "f", vec)
        elif op == "rsqrt":
            expr = _Value(f"cute.math.rsqrt({x.expr})", "f", vec)
        elif op == "log":
            expr = _Value(f"cute.math.log({x.expr})", "f", vec)
        else:
            assert op == "reciprocal"
            if self.fast_math:
                expr = _Value(
                    f"cute.math.rcp({x.expr}, {_approx_rcp_flags(vec)})", "f", vec
                )
            else:
                expr = _Value(f"cute.math.rcp({x.expr})", "f", vec)
        return self._bind(section, _round_to_dtype(expr, dtype))

    def _emit_scalar_load(self, node: torch.fx.Node) -> _Value:
        host_tensors = _flash_graph_host_tensors(self.scope.infos)
        name = _host_tensor_name(node.args[0])
        assert name is not None
        fake = host_tensors[name]
        arg = self.df.tensor_arg(fake, prefer_name=name)
        self.tensor_names.add(arg.name)
        indices = node.args[1]
        assert isinstance(indices, (list, tuple))
        # 64-bit element offsets: the tensor may exceed 2**31 elements.
        terms = [
            f"cutlass.Int64({_to_i(self.emit(index))}) * cutlass.Int64({arg.name}.layout.stride[{dim}])"
            for dim, index in enumerate(indices)
        ]
        pointer = f"({arg.name}.iterator + {' + '.join(terms)})"
        val = _node_val(node)
        assert isinstance(val, torch.Tensor)
        if val.dtype.is_floating_point:
            return self._bind("cta", _Value(f"cutlass.Float32({pointer}.load())", "f"))
        return self._bind("cta", _Value(f"cutlass.Int32({pointer}.load())", "i"))


# ---------------------------------------------------------------------------
# Device body
# ---------------------------------------------------------------------------


def _indent(src: str, prefix: str) -> str:
    return textwrap.indent(textwrap.dedent(src).strip("\n"), prefix)


def _gated_prologue_loads(kv_stage: int, *, indent: str, q_tile: str) -> str:
    """Warp-0 block issuing a work item's Q TMA load and its first ``kv_stage``
    K/V ring loads (``indent`` is the block's indentation)."""
    lines = [
        "flash_q_empty = flash_q_prod.acquire_and_advance()",
        f"cute.copy(_flash_tma_q, tQgQ[None, {q_tile}], tQsQ[None, flash_q_empty.index],",
        "          tma_bar_ptr=flash_q_empty.barrier)",
    ]
    for pf in range(kv_stage):
        lines.extend(
            (
                f"if cutlass.Int32({pf}) < flash_num_kv_active:",
                "    flash_k_empty = flash_k_prod.acquire_and_advance()",
                f"    cute.copy(_flash_tma_k, tKgK[None, cutlass.Int32({pf})],",
                "              tKsK[None, flash_k_empty.index], tma_bar_ptr=flash_k_empty.barrier)",
                "    flash_v_empty = flash_v_prod.acquire_and_advance()",
                f"    cute.copy(_flash_tma_v, tVgV[None, cutlass.Int32({pf})],",
                "              tVsV[None, flash_v_empty.index], tma_bar_ptr=flash_v_empty.barrier)",
            )
        )
    return "\n".join(indent + line for line in lines)


def _gated_producer_body(kv_stage: int, item_section: str) -> str:
    """Warp 0: per work item the tcgen05 MMAs and the steady-state K/V ring over
    a runtime KV count (item 0's Q load and first ring loads were issued during
    setup; later items issue their own at the head of the item)."""
    qk0_pf = f"""
            if cutlass.Int32({kv_stage}) < flash_num_kv_active:
                flash_k_empty = flash_k_prod.acquire_and_advance()
                cute.copy(_flash_tma_k, tKgK[None, cutlass.Int32({kv_stage})],
                          tKsK[None, flash_k_empty.index], tma_bar_ptr=flash_k_empty.barrier)"""
    qk_next = _flash_ws_qk_ahead(
        kv_stage,
        kpf=True,
        qk_condition="flash_kv + cutlass.Int32(1) < flash_num_kv_active",
        kpf_condition=f"flash_kv + cutlass.Int32({kv_stage + 1}) < flash_num_kv_active",
    )
    # PV(kv) reads P[kv % 2] from TMEM and accumulates into the single O
    # buffer. Unlike the softmax body there is no per-tile O rescale, so the O
    # pipeline stage is acquired ONCE per item before the KV loop and committed
    # ONCE after the last PV (``tcgen05.commit`` fires when every prior MMA has
    # completed); a per-tile acquire would wait for a consumer release that
    # only happens in the epilogue.
    pv_current = f"""
                flash_p_full = flash_p_ready_cons.wait_and_advance()
                flash_v_full = flash_v_cons.wait_and_advance()
                flash_first_acc = flash_o_started
                if (flash_p_idx % 2) == 0:
                    for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
                        _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_first_acc | (flash_kp != 0))
                        cute.gemm(_flash_pv_mma, tOtO, tOrP0[None, None, flash_kp, 0],
                                  tOrV[None, None, flash_kp, flash_v_full.index], tOtO)
                else:
                    for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
                        _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_first_acc | (flash_kp != 0))
                        cute.gemm(_flash_pv_mma, tOtO, tOrP1[None, None, flash_kp, 0],
                                  tOrV[None, None, flash_kp, flash_v_full.index], tOtO)
                flash_v_full.release()
                flash_p_full.release()
                flash_p_idx = (flash_p_idx + 1) % 2
                flash_o_started = cutlass.Boolean(True)
                if flash_kv + cutlass.Int32({kv_stage}) < flash_num_kv_active:
                    flash_v_empty = flash_v_prod.acquire_and_advance()
                    cute.copy(_flash_tma_v, tVgV[None, flash_kv + cutlass.Int32({kv_stage})],
                              tVsV[None, flash_v_empty.index], tma_bar_ptr=flash_v_empty.barrier)"""
    # The steady-state helper is written for a 12-space loop body; the item
    # loop adds one level.
    qk_next = qk_next.replace("\n", "\n    ")
    later_loads = _gated_prologue_loads(
        kv_stage, indent="                ", q_tile="flash_m_tile"
    )
    return f"""
if warp_idx == 0:
        flash_nk = cute.size(tSrQ, mode=[2])
        flash_nk2 = cute.size(tOrP0, mode=[2])
        flash_p_idx = cutlass.Int32(0)
        flash_qk_idx = cutlass.Int32(0)
        for flash_item in cutlass.range(flash_n_items, unroll=1):
{item_section}
            if flash_item > cutlass.Int32(0):
{later_loads}
            flash_o_started = cutlass.Boolean(False)
            flash_q_full = flash_q_cons.wait_and_advance()
            if cutlass.Int32(0) < flash_num_kv_active:
                flash_k_full = flash_k_cons.wait_and_advance()
                flash_s_handle = flash_mma_s_prod.acquire_and_advance()
                if (flash_qk_idx % 2) == 0:
                    for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                        _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                        cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q_full.index],
                                  tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
                else:
                    for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                        _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                        cute.gemm(_flash_qk_mma, tStS1, tSrQ[None, None, flash_kp, flash_q_full.index],
                                  tSrK[None, None, flash_kp, flash_k_full.index], tStS1)
                flash_s_handle.commit()
                flash_k_full.release()
                flash_qk_idx = (flash_qk_idx + 1) % 2{qk0_pf}
            flash_o_handle = flash_mma_o_prod.acquire_and_advance()
            for flash_active_kv in cutlass.range(flash_num_kv_active, unroll=1):
                flash_kv = flash_active_kv{qk_next}{pv_current}
            flash_o_handle.commit()
            flash_q_full.release()"""


def _gated_consumer_body(
    *,
    hd: int,
    bn: int,
    q_tile: int,
    geometry: GatedChunkGeometry,
    gate_warpgroups: int,
    io_dtype: str,
    item_section: str,
    row_section: str,
    kviter_section: str,
    elem_section: str,
    p_expr: str,
    store_predicate: str,
    lane_expr: str,
) -> str:
    chunk_cols = geometry.chunk_cols
    half_rows = q_tile != GATED_Q_TILE
    # Chunk ``ci`` of every KV tile belongs to gate warpgroup ``ci % G``.
    chunk_index = (
        f"flash_cq * cutlass.Int32({gate_warpgroups}) + flash_gate_wg"
        if gate_warpgroups > 1
        else "flash_cq"
    )
    epilogue_guard = (
        "if flash_gate_wg == 0:" if gate_warpgroups > 1 else "if cutlass.Boolean(True):"
    )
    teardown_arrive_epilogue = _gated_teardown_arrive(gate_warpgroups, 20)
    teardown_arrive_loop_end = _gated_teardown_arrive(gate_warpgroups, 12)
    # ``1 << j`` as an Int32 literal (bit 31 is the sign bit).
    bit_literal = "(cutlass.Int32(1) << cutlass.Int32(flash_j))"
    in_tile_mask = _col_mask("<", "(flash_kv_end - flash_chunk_base)")
    if half_rows:
        # ``16x64b`` TMEM shapes at the 64-row tile: thread ``t`` of a gate
        # warpgroup holds one column parity (``(t // 2) % 2``) of row
        # ``16 * (t // 32) + (t % 32) // 4 + 8 * (t % 2)`` (rows 16q..16q+15 sit
        # in lanes 16q..16q+15 of quadrant q); the partner thread ``t ^ 2`` holds
        # the other parity.  The thread's O half row is its parity's half.
        row_lines = """flash_col_parity = (flash_local_tidx // cutlass.Int32(2)) % cutlass.Int32(2)
            flash_gate_row = flash_q_row0 + cutlass.Int32(16) * (flash_local_tidx // cutlass.Int32(32)) + (flash_local_tidx % cutlass.Int32(32)) // cutlass.Int32(4) + cutlass.Int32(8) * (flash_local_tidx % cutlass.Int32(2))"""
        chunk_base = "flash_chunk_col0 + flash_col_parity"
        o_cols = hd // 2
        o_ld_atom = (
            f"cute_tcgen05_flash.Ld16x64bOp(cute_tcgen05_flash.Repetition({o_cols}))"
        )
        o_offset = f"\n                    + cutlass.Int64(flash_col_parity) * cutlass.Int64({o_cols})"
        o_convert = f"""flash_rego_row.store(_helion_flash_rt.gated_m64_o_half_row(
                    flash_reg_row.load(), {io_dtype}, flash_col_parity))"""
    else:
        row_lines = "flash_gate_row = flash_q_row0 + flash_local_tidx"
        chunk_base = "flash_chunk_col0"
        o_cols = hd
        o_ld_atom = f"cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition({min(hd, 64)}))"
        o_offset = ""
        o_convert = f"flash_rego.store(flash_reg.load().to({io_dtype}))"
    # 16-bit P is packed two per 32-bit TMEM word (at the 64-row tile each word
    # takes one value from each thread of a lane pair); 32-bit P overwrites S in
    # place (the thread's own columns at either tile).
    if io_dtype == "cutlass.Float32":
        p_store = (
            "tSTrS.store(flash_p_v)"
            if not half_rows
            else "cute.make_tensor(tSTrS.iterator, flash_chunk_layout).store(flash_p_v)"
        )
    elif half_rows:
        p_store = f"""cute.make_tensor(tSTrS.iterator, flash_word_layout).store(
                        _helion_flash_rt.gated_m64_p_words(flash_p_v, {io_dtype}, flash_col_parity)
                        .bitcast(cutlass.Float32))"""
    else:
        p_store = f"""tSTrS_e = cute.make_tensor(
                        cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), flash_chunk_layout)
                    tSTrS_e.store(flash_p_v.to({io_dtype}))"""
    return f"""
if warp_idx >= 4:
        flash_gate_wg = warp_idx // cutlass.Int32(4) - cutlass.Int32(1)
        flash_s_idx = cutlass.Int32(0)
        flash_ld_chunk_shape = tLDcS[None, 0, None, None].shape
        flash_st_chunk_shape = tSTcS[None, 0, 0].shape
        # Flat views of one thread's chunk: the gate runs on ``chunk_cols``-wide
        # vectors (one lane per score element; the elements are ``col_stride``
        # columns apart).
        flash_chunk_layout = cute.make_layout(({chunk_cols},), stride=(1,))
        flash_word_layout = cute.make_layout(({chunk_cols // 2},), stride=(1,))
        flash_iota_frag = cute.make_rmem_tensor(({chunk_cols},), cutlass.Int32)
        flash_bit_frag = cute.make_rmem_tensor(({chunk_cols},), cutlass.Int32)
        for flash_j in cutlass.range_constexpr({chunk_cols}):
            flash_iota_frag[flash_j] = cutlass.Int32({geometry.col_stride} * flash_j)
            flash_bit_frag[flash_j] = cutlass.Int32({bit_literal})
        flash_col_iota = flash_iota_frag.load()
        flash_bit_iota = flash_bit_frag.load()
        # O epilogue partitions (item independent): a thread owns one output
        # row (Ld32x32b at the 128-row tile) or one half row (Ld16x64b at the
        # 64-row tile).
        flash_epi_tiler = ((cute.size(tOtO, mode=[0, 0]), cute.size(tOtO, mode=[0, 1])),)
        tOtO_epi = cute.zipped_divide(tOtO, flash_epi_tiler)
        cO = cute.make_identity_tensor(({q_tile}, {hd}))
        tOcO_epi = cute.zipped_divide(flash_pvt.partition_C(cO), flash_epi_tiler)
        flash_o_ld_atom = cute.make_copy_atom({o_ld_atom}, cutlass.Float32)
        flash_tiled_o_ld = cute_tcgen05_flash.make_tmem_copy(flash_o_ld_atom, tOtO_epi[None, 0])
        flash_thr_o_ld = flash_tiled_o_ld.get_slice(flash_local_tidx)
        tDtO = flash_thr_o_ld.partition_S(tOtO_epi)
        tDcO = flash_thr_o_ld.partition_D(tOcO_epi)
        flash_reg = cute.make_rmem_tensor(tDcO[None, None, 0].shape, cutlass.Float32)
        flash_reg_row = cute.make_tensor(
            flash_reg.iterator, cute.make_layout(({o_cols},), stride=(1,)))
        flash_rego = cute.make_rmem_tensor(tDcO[None, None, 0].shape, {io_dtype})
        flash_rego_row = cute.make_tensor(
            flash_rego.iterator, cute.make_layout(({o_cols},), stride=(1,)))
        for flash_item in cutlass.range(flash_n_items, unroll=1):
{item_section}
            {row_lines}
            flash_row_in_tile = flash_gate_row < flash_q_extent
{row_section}
            for flash_kv in cutlass.range(flash_num_kv_active, unroll=1):
                flash_kv_col0 = flash_kv * cutlass.Int32({bn})
                flash_kv_col_end = cutlass.min(flash_kv_col0 + cutlass.Int32({bn}), flash_kv_end)
{kviter_section}
                flash_s_full = flash_mma_s_cons.wait_and_advance()
                # Chunked t2r -> gate -> r2t: one 32-column fragment is live at a
                # time so the scheduler can interleave independent elements instead
                # of serializing a whole 128-column row under register pressure.
                for flash_cq in cutlass.range({geometry.chunk_iters}, unroll=1):
                    flash_ci = {chunk_index}
                    flash_chunk_col0 = flash_kv_col0 + flash_ci * cutlass.Int32({geometry.chunk_span})
                    flash_chunk_base = {chunk_base}
                    tLDrS = cute.make_rmem_tensor(flash_ld_chunk_shape, cutlass.Float32)
                    if (flash_s_idx % 2) == 0:
                        cute.copy(flash_tiled_ld0, tLDtS0[None, flash_ci, None, None], tLDrS)
                    else:
                        cute.copy(flash_tiled_ld1, tLDtS1[None, flash_ci, None, None], tLDrS)
                    cute.arch.fence_view_async_tmem_load()
                    flash_s_raw_v = cute.make_tensor(tLDrS.iterator, flash_chunk_layout).load()
                    flash_gate_col_v = flash_col_iota + flash_chunk_base
                    # Bit j: column j of the chunk is inside the tile (row and kv_end).
                    flash_elem_in_tile_m = cutlass.select_(
                        flash_row_in_tile, {in_tile_mask}, cutlass.Int32(0))
{elem_section}
                    flash_p_v = {p_expr}
                    tSTrS = cute.make_rmem_tensor(flash_st_chunk_shape, cutlass.Float32)
                    {p_store}
                    if (flash_s_idx % 2) == 0:
                        cute.copy(flash_tiled_st0, tSTrS, tSTtS0[None, 0, flash_ci])
                    else:
                        cute.copy(flash_tiled_st1, tSTrS, tSTtS1[None, 0, flash_ci])
                cute.arch.fence_view_async_tmem_store()
                flash_p_handle = flash_p_ready_prod.acquire_and_advance()
                flash_p_handle.commit()
                flash_s_full.release()
                flash_s_idx = (flash_s_idx + 1) % 2
            # Only gate warpgroup 0 drains O (the store is ~1% of the tile time).
            {epilogue_guard}
                flash_o_row = flash_row_base + flash_gate_row
                # 64-bit element offset (O may exceed 2**31 elements) asserted
                # divisible by 8 elements: every row and lane stride is
                # ``head_dim`` times a power of two, so the offset keeps the base
                # pointer's 16-byte alignment and the row store vectorizes.
                flash_o_row_ptr = _flash_mOt.iterator + cute.assume(
                    cutlass.Int64(flash_o_row) * cutlass.Int64(_flash_mOt.layout.stride[0])
                    + cutlass.Int64({lane_expr}) * cutlass.Int64(_flash_mOt.layout.stride[2]){o_offset},
                    divby=8)
                gO_row = cute.make_tensor(flash_o_row_ptr, cute.make_layout(({o_cols},), stride=(1,)))
                flash_store_row_ok = {store_predicate}
                # The O stage is committed once by warp 0 after the item's last PV
                # (or with no MMAs at all when the KV range is empty, in which case
                # the frontend's zero accumulator is stored).
                flash_o_full = flash_mma_o_cons.wait_and_advance()
                if flash_num_kv_active > 0:
                    cute.copy(flash_tiled_o_ld, tDtO[None, None, 0], flash_reg)
                    cute.arch.fence_view_async_tmem_load()
                else:
                    flash_reg.fill(0.0)
                flash_o_full.release()
                if flash_item == flash_n_items - cutlass.Int32(1):
                    # Last TMEM read of the item loop: let warp {GATED_TMEM_WARP} free TMEM
                    # while the output rows are converted and stored.
                    {teardown_arrive_epilogue}
                {o_convert}
                if flash_store_row_ok:
                    cute.autovec_copy(flash_rego_row, gO_row)
        # Warpgroups without an epilogue (and an epilogue warpgroup with no work
        # item) are done with TMEM once the item loop ends.
        if (flash_gate_wg != cutlass.Int32(0)) | (flash_n_items == cutlass.Int32(0)):
            {teardown_arrive_loop_end}"""


def gated_tmem_columns(head_dim: int, kv_tile: int, packed_p: bool = False) -> int:
    """TMEM columns for S0, S1, O and -- for packed 16-bit P -- the two P
    buffers (``kv_tile // 2`` columns each), rounded up to the allocation
    granularity."""
    needed = 2 * kv_tile + head_dim + (kv_tile if packed_p else 0)
    columns = 32
    while columns < needed:
        columns *= 2
    return columns


def _gated_setup_body(
    *,
    hd: int,
    bn: int,
    q_tile: int,
    chunk_cols: int,
    kv_stage: int,
    io_dtype: str,
    mma_dtype: str,
    gate_warpgroups: int,
) -> tuple[str, str, str, str]:
    """Shared smem/TMEM/pipeline/TMA setup (all threads), in four parts.

    ``a1`` (shared memory, the TMEM allocation, the mbarriers and their fence)
    depends on nothing the kernel loads; it is emitted *before* the CTA's
    scalar loads (the jagged row offsets) because ptxas converts a loaded
    warp-uniform value to a uniform register right behind the load, stalling
    every warp there: work placed after the load never overlaps it.  ``a2``
    (the TMEM allocation barrier, TMEM tensors and copy partitions) follows
    item 0's loads.  ``slices`` addresses the global
    tensors at the CTA's row base (CTA level; harmless for an inactive CTA)
    and ``loads`` issues item 0's TMA loads under the CTA predicate.
    """
    packed_p = io_dtype != "cutlass.Float32"
    tmem_cols = gated_tmem_columns(hd, bn, packed_p)
    threads = 128 * (1 + gate_warpgroups)
    gate_threads = 128 * gate_warpgroups
    # P is stored back one chunk at a time: ``chunk_cols // 2`` fp32 words of
    # packed 16-bit P, or the ``chunk_cols`` words of 32-bit P.
    p_store_rep = chunk_cols if io_dtype == "cutlass.Float32" else chunk_cols // 2
    # TMEM copy atoms: one row per thread at the 128-row tile (32x32b), one
    # column parity of a row per thread at the 64-row tile (16x64b).
    # Packed P lives in its own two TMEM buffers after O; 32-bit P aliases S.
    p_base_offset = 2 * bn + hd if packed_p else 0
    p1_offset = bn // 2 if packed_p else bn
    if q_tile == GATED_Q_TILE:
        ld_atom = f"cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition({chunk_cols}))"
        st_atom = f"cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({p_store_rep}))"
    else:
        ld_atom = f"cute_tcgen05_flash.Ld16x64bOp(cute_tcgen05_flash.Repetition({chunk_cols}))"
        st_atom = f"cute_tcgen05_flash.St16x64bOp(cute_tcgen05_flash.Repetition({p_store_rep}))"
    # Item 0's loads are issued as soon as the row base is known (the mbarriers
    # were published by the TMEM allocation barrier of the first part).
    prologue_loads = (
        "if (warp_idx == 0) & (flash_n_items > cutlass.Int32(0)):\n"
        + _gated_prologue_loads(kv_stage, indent="    ", q_tile="flash_m_tile")
    )
    part_a1 = f"""
flash_local_tidx = tidx % 128
# Start the TMA descriptor fetches first: they overlap the smem/mbarrier setup
# instead of sitting in front of the first Q/K/V load.
if warp_idx == 0:
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_q)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_k)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_v)
_flash_storage_cls = _helion_flash_rt.flash_gated_shared_storage({hd}, {bn}, {kv_stage}, {mma_dtype}, {q_tile})
smem = cutlass_utils_flash.SmemAllocator()
storage = smem.allocate(_flash_storage_cls)
sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)
sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)
sV = storage.sV.get_tensor(_flash_vsl.outer, swizzle=_flash_vsl.inner)

flash_tmem_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=1, num_threads={threads})
# Warp {GATED_TMEM_WARP} allocates (and later frees) TMEM while warp 0 initializes the
# mbarriers and issues item 0's loads.
flash_tmem = cutlass_utils_flash.TmemAllocator(
    storage.tmem_holding_buf.ptr, barrier_for_retrieve=flash_tmem_bar,
    allocator_warp_id={GATED_TMEM_WARP})
flash_tmem.allocate({tmem_cols})
if warp_idx == {GATED_TMEM_WARP}:
    # One allocation per CTA: hand the permit back right away so co-resident
    # CTAs can allocate, instead of at the very end.
    flash_tmem.relinquish_alloc_permit()
    # The allocation writes the TMEM address to shared memory as a tcgen05
    # operation: publish it before the allocation barrier (a co-resident CTA
    # gets a nonzero address that an unfenced reader can miss).
    _helion_flash_rt.tcgen05_fence_before_thread_sync()

flash_q_bytes = cute.size_in_bytes({mma_dtype}, cute.select(_flash_qsl, mode=[0, 1, 2]))
flash_k_bytes = cute.size_in_bytes({mma_dtype}, cute.select(_flash_ksl, mode=[0, 1, 2]))
flash_v_bytes = cute.size_in_bytes({mma_dtype}, cute.select(_flash_vsl, mode=[0, 1, 2]))
flash_q_prod, flash_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr(),
    defer_sync=True).make_participants()
flash_k_prod, flash_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr(),
    defer_sync=True).make_participants()
flash_v_prod, flash_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_v_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr(),
    defer_sync=True).make_participants()
flash_mma_s_prod, flash_mma_s_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=2,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, {gate_threads}),
    barrier_storage=storage.mma_s_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
flash_mma_o_prod, flash_mma_o_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_o_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
flash_p_ready_prod, flash_p_ready_cons = cutlass_pipeline_flash.PipelineAsync.create(
    num_stages=2,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, {gate_threads}),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 32),
    barrier_storage=storage.p_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
# Warp 0 initialized every mbarrier above (``defer_sync``); one fence publishes
# them to the async proxy and the TMEM ``wait_for_alloc`` barrier below is the
# single CTA-wide sync of the prologue (it also publishes them to the other
# threads before item 0's loads arrive on them).
cute.arch.mbarrier_init_fence()

flash_qkt = _flash_qk_mma.get_slice(0)
flash_pvt = _flash_pv_mma.get_slice(0)
tSrQ = flash_qkt.make_fragment_A(sQ)
tSrK = flash_qkt.make_fragment_B(sK)
tOrV = flash_pvt.make_fragment_B(sV)
flash_qk_acc_shape = flash_qkt.partition_shape_C(({q_tile}, {bn}))
tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
flash_pv_acc_shape = flash_pvt.partition_shape_C(({q_tile}, {hd}))
tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)
"""
    part_a2 = f"""
flash_tmem.wait_for_alloc()
_helion_flash_rt.tcgen05_fence_after_thread_sync()
flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
flash_s_layout = tStS.layout
# Double-buffered S: S0 @ col 0, S1 @ col bn, O @ col 2 * bn.  Packed 16-bit
# P has two buffers of its own @ 2 * bn + hd (a chunk's P words would
# otherwise alias S columns another gate warpgroup has yet to load); 32-bit
# P overwrites the thread's own S columns in place.
tStS0 = cute.make_tensor(flash_tmem_ptr, flash_s_layout)
tStS1 = cute.make_tensor(flash_tmem_ptr + {bn}, flash_s_layout)
tOtO = cute.make_tensor(flash_tmem_ptr + {2 * bn}, tOtO.layout)
flash_p_base = flash_tmem_ptr + {p_base_offset}
# ``make_fragment_A`` of a TMEM tensor does not carry the tensor's column base
# (neither a static offset nor the CTA's allocation base, which is nonzero for
# the second gated CTA on an SM).  The fragment is built once and moved to
# each P buffer through its own iterator, in ``mma_dtype`` units, by the
# allocation base column plus the buffer's offset.
tP = cute.make_tensor(flash_tmem_ptr, _flash_ptl.outer)
tOrP_base = flash_pvt.make_fragment_A(tP)
flash_tmem_col0 = cutlass.Int32(flash_tmem_ptr.toint()) & cutlass.Int32(65535)
tOrP0 = cute.make_tensor(
    tOrP_base.iterator + (cutlass.Float32.width // {mma_dtype}.width)
    * (flash_tmem_col0 + cutlass.Int32({p_base_offset})),
    tOrP_base.layout)
tOrP1 = cute.make_tensor(
    tOrP_base.iterator + (cutlass.Float32.width // {mma_dtype}.width)
    * (flash_tmem_col0 + cutlass.Int32({p_base_offset + p1_offset})),
    tOrP_base.layout)

cS = cute.make_identity_tensor(({q_tile}, {bn}))
tScS = flash_qkt.partition_C(cS)
flash_ld_atom = cute.make_copy_atom({ld_atom}, cutlass.Float32)
flash_tiled_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS0)
flash_tiled_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS1)
flash_thr_ld0 = flash_tiled_ld0.get_slice(flash_local_tidx)
flash_thr_ld1 = flash_tiled_ld1.get_slice(flash_local_tidx)
tLDtS0 = flash_thr_ld0.partition_S(tStS0)
tLDtS1 = flash_thr_ld1.partition_S(tStS1)
tLDcS = flash_thr_ld0.partition_D(tScS)

flash_tilePlikeFP32 = {bn} // cutlass.Float32.width * {mma_dtype}.width
flash_P_layout = cute.composition(flash_s_layout, cute.make_layout(({q_tile}, flash_tilePlikeFP32)))
tStS_P0 = cute.make_tensor(flash_p_base, flash_P_layout)
tStS_P1 = cute.make_tensor(flash_p_base + {p1_offset}, flash_P_layout)
flash_tScS_P_layout = cute.composition(tScS.layout, cute.make_layout(({q_tile}, flash_tilePlikeFP32)))
tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
flash_st_atom = cute.make_copy_atom({st_atom}, cutlass.Float32)
flash_tiled_st0 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P0)
flash_tiled_st1 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P1)
flash_thr_st0 = flash_tiled_st0.get_slice(flash_local_tidx)
flash_thr_st1 = flash_tiled_st1.get_slice(flash_local_tidx)
tSTtS0 = flash_thr_st0.partition_D(tStS_P0)
tSTtS1 = flash_thr_st1.partition_D(tStS_P1)
tSTcS = flash_thr_st0.partition_S(tScS_P)
"""
    slices = f"""
# Jagged rows: shift the TMA coordinate tensors by the per-CTA row base so
# tile 0 of Q and tile ``kv`` of K/V address rows ``base + kv * bn``.  Rows past
# the tensor extent are zero-filled by TMA.
flash_mQ_rows = cute.domain_offset((flash_row_base, 0, 0), _flash_mQt)
flash_mK_rows = cute.domain_offset((flash_row_base, 0, 0), _flash_mKt)
flash_mV_rows = cute.domain_offset((0, flash_row_base, 0), _flash_mVt)
gQ = cute.flat_divide(flash_mQ_rows, cute.select(({q_tile}, {bn}, {hd}), mode=[0, 2]))
gK = cute.flat_divide(flash_mK_rows, cute.select(({q_tile}, {bn}, {hd}), mode=[1, 2]))
gV = cute.flat_divide(flash_mV_rows, cute.select(({q_tile}, {hd}, {bn}), mode=[1, 2]))
tSgQ = flash_qkt.partition_A(gQ)
tSgK = flash_qkt.partition_B(gK)
tOgV = flash_pvt.partition_B(gV)
tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(
    _flash_tma_q, 0, cute.make_layout(1),
    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(
    _flash_tma_k, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(
    _flash_tma_v, 0, cute.make_layout(1),
    cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))
tQgQ = tQgQ_qdl[None, None, 0, flash_lane]
tKgK = tKgK_kdl[None, None, 0, flash_lane]
tVgV = tVgV_dkl[None, 0, None, flash_lane]
"""
    return part_a1, part_a2, slices, prologue_loads


def _gated_teardown_threads(gate_warpgroups: int) -> int:
    """Threads on the teardown barrier: the TMEM-owning warp and the gate warps."""
    return 32 + 128 * gate_warpgroups


def _gated_teardown_arrive(gate_warpgroups: int, indent: int) -> str:
    """Order the warp's TMEM accesses before the teardown barrier, then arrive."""
    return (
        "_helion_flash_rt.tcgen05_fence_before_thread_sync()\n"
        + " " * indent
        + f"cute.arch.barrier_arrive(barrier_id={GATED_TEARDOWN_BARRIER_ID}, "
        f"number_of_threads={_gated_teardown_threads(gate_warpgroups)})"
    )


def _gated_teardown(gate_warpgroups: int) -> str:
    """Warp 1 frees TMEM once every gate warp has arrived (each right after its
    last TMEM read); the producer warp and the other idle warps just exit."""
    return f"""
if warp_idx == {GATED_TMEM_WARP}:
    cute.arch.barrier(barrier_id={GATED_TEARDOWN_BARRIER_ID}, number_of_threads={_gated_teardown_threads(gate_warpgroups)})
    _helion_flash_rt.tcgen05_fence_after_thread_sync()
    flash_tmem.free(flash_tmem_ptr)
"""


_GATED_NODE_NAME_RE = re.compile(r"\bflash_g\d+\b")
_GATED_CAST_LOAD_RE = re.compile(r"^(cutlass\.\w+)\((.*\.load\(\))\)$")


def _split_load_dependent(statements: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split the CTA-level ``name = expr`` statements into those that do not
    depend on a global load's value (the loads themselves included) and those
    that do, preserving order within each group.

    A load wrapped in a dtype cast (``cutlass.Int32((...).load())``) is split
    into the bare load and a late cast so the loads are issued together at the
    head of the load section and their uses follow (the setup that has to
    precede the loads is emitted ahead of both groups; see the emission order
    in ``_emit_gated_flash_device_body``).
    """
    dependent: set[str] = set()
    early: list[str] = []
    late: list[str] = []
    for stmt in statements:
        name, _, rhs = stmt.partition(" = ")
        name = name.strip()
        if set(_GATED_NODE_NAME_RE.findall(rhs)) & dependent:
            dependent.add(name)
            late.append(stmt)
            continue
        cast_load = _GATED_CAST_LOAD_RE.match(rhs)
        if cast_load is not None:
            cast, load = cast_load.groups()
            dependent.add(name)
            early.append(f"{name}_raw = {load}")
            late.append(f"{name} = {cast}({name}_raw)")
            continue
        if ".load()" in rhs:
            dependent.add(name)
        early.append(stmt)
    return early, late


def emit_gated_flash_device_body(
    df: DeviceFunction,
    plan: GatedAttentionPlan,
) -> tuple[list[ast.AST], set[str]]:
    """Build the fused gated body; returns the statements and the tensor arg names it reads."""
    match = plan.match
    hd = match.head_dim
    bn = plan.kv_tile
    q_tile = plan.q_tile
    io_dtype = _gated_io_dtype_str(match.io_dtype)
    mma_dtype = _gated_mma_dtype_str(match.io_dtype)
    token = _GEOMETRY.set(plan.geometry)
    try:
        return _emit_gated_flash_device_body(
            df,
            plan,
            hd=hd,
            bn=bn,
            q_tile=q_tile,
            io_dtype=io_dtype,
            mma_dtype=mma_dtype,
        )
    finally:
        _GEOMETRY.reset(token)


def _emit_gated_flash_device_body(
    df: DeviceFunction,
    plan: GatedAttentionPlan,
    *,
    hd: int,
    bn: int,
    q_tile: int,
    io_dtype: str,
    mma_dtype: str,
) -> tuple[list[ast.AST], set[str]]:
    match = plan.match
    emitter = _GatedEmitter(df, plan)
    sequence = match.form == "sequence"
    root_extents = [
        emitter.extent_expr(block_id) for block_id in match.leading_block_ids
    ]

    decode_lines = ["flash_pid = cutlass.Int32(cute.arch.block_idx()[0])"]
    divisor = "cutlass.Int32(1)"
    if sequence:
        # One CTA per (sequence, lane): the launch grid is the sequence grid
        # times the lane extent (``launch_grid_multiplier``); the lane is the
        # fastest coordinate.
        lanes = match.lane_extent
        assert lanes is not None
        decode_lines.extend(
            (
                f"flash_lane = flash_pid % cutlass.Int32({lanes})",
                f"flash_axis0 = flash_pid // cutlass.Int32({lanes})",
            )
        )
    else:
        # Leading-axis decode mirrors Helion's default flat pid order (axis 0 fastest).
        for axis, extent in enumerate(root_extents):
            decode_lines.append(
                f"flash_axis{axis} = (flash_pid // {divisor}) % cutlass.Int32({extent})"
            )
            divisor = f"({divisor} * cutlass.Int32({extent}))"

    # Scalar programs: row base, KV loop end, CTA predicate, lane coordinate.
    if sequence:
        row_base_expr = _to_i(emitter.emit(match.q_begin))
        q_extent_expr = f"cutlass.max({_to_i(emitter.emit(match.q_end))} - flash_row_base, cutlass.Int32(0))"
        n_items_expr = (
            f"(flash_q_extent + cutlass.Int32({q_tile - 1})) // cutlass.Int32({q_tile})"
        )
        m_tile_expr = "flash_n_items - cutlass.Int32(1) - flash_item"
        kv_end_expr = "flash_q_extent"
        lane_expr = "flash_lane"
    else:
        if match.row_offset is not None:
            row_base_expr = _to_i(emitter.emit(match.row_offset))
        else:
            row_base_expr = "cutlass.Int32(0)"
        q_extent_expr = emitter.q_extent_expr()
        n_items_expr = "cutlass.Int32(1)"
        # Query tiles are walked from the last to the first: under a
        # causal-style KV bound the last tile has the most KV tiles, so the
        # heaviest CTAs start in the first wave instead of forming the tail.
        m_tile_expr = (
            f"(flash_q_extent + cutlass.Int32({q_tile - 1})) // cutlass.Int32({q_tile})"
            f" - cutlass.Int32(1) - flash_pid // {divisor}"
        )
        kv_end_value = emitter.emit(match.kv_end)
        kv_end_expr = _to_i(kv_end_value)
        if match.q.lane_block_id is not None:
            lane_expr = emitter.tile_scalar_expr(
                tile_ops.tile_begin, match.q.lane_block_id
            ).expr
        else:
            assert match.q.lane_constant is not None
            lane_expr = f"cutlass.Int32({match.q.lane_constant})"
    predicate_value = emitter.emit(match.if_predicate)
    row_terms = [_to_b(emitter.emit(term)) for term in match.store_skip_terms]
    p_value = emitter.emit(match.p_node)
    p_expr = emitter.finish_elem_section(p_value)
    # KV tiles past the columns the P mask provably zeroes are skipped.
    kv_end_terms = [kv_end_expr, *emitter.kv_end_bounds()]
    kv_end_min = kv_end_terms[0]
    for term in kv_end_terms[1:]:
        kv_end_min = f"cutlass.min({kv_end_min}, {term})"

    # Per work item (query tile): tile coordinates, the emitter's item-level
    # scalars and the KV range.  Evaluated once at CTA level for item 0 (the
    # CTA predicate and the setup-time loads read it) and per item in the roles.
    def item_lines(item: str) -> list[str]:
        return [
            *([f"flash_item = {item}"] if item != "flash_item" else []),
            f"flash_m_tile = {m_tile_expr}",
            f"flash_q_row0 = flash_m_tile * cutlass.Int32({q_tile})",
            (
                "flash_q_row_end = cutlass.min("
                f"flash_q_row0 + cutlass.Int32({q_tile}), flash_q_extent)"
            ),
            *emitter.sections["item"],
            f"flash_kv_end = {kv_end_min}",
            (
                "flash_num_kv_active = cutlass.max((flash_kv_end + cutlass.Int32("
                f"{bn - 1})) // cutlass.Int32({bn}), cutlass.Int32(0))"
            ),
        ]

    early_cta, late_cta = _split_load_dependent(emitter.sections["cta"])
    cta_lines = [
        *late_cta,
        f"flash_row_base = {row_base_expr}",
        f"flash_q_extent = {q_extent_expr}",
        f"flash_n_items = {n_items_expr}",
        *item_lines("cutlass.Int32(0)"),
        f"flash_lane = {lane_expr}",
        f"flash_body_active = {_to_b(predicate_value)}",
    ]
    store_terms = [
        "flash_row_in_tile",
        *row_terms,
        "(flash_o_row < cutlass.Int32(_flash_mOt.layout.shape[0]))",
    ]
    item_section = _indent("\n".join(item_lines("flash_item")), " " * 12)
    consumer = _gated_consumer_body(
        hd=hd,
        bn=bn,
        q_tile=q_tile,
        geometry=plan.geometry,
        gate_warpgroups=plan.gate_warpgroups,
        io_dtype=io_dtype,
        item_section=item_section,
        row_section=_indent("\n".join(emitter.sections["row"]) or "pass", " " * 12),
        kviter_section=_indent(
            "\n".join(emitter.sections["kviter"]) or "pass", " " * 16
        ),
        elem_section=_indent("\n".join(emitter.sections["elem"]) or "pass", " " * 20),
        p_expr=p_expr,
        store_predicate=" & ".join(store_terms),
        lane_expr="flash_lane",
    )
    setup_a1, setup_a2, slices, item0_loads = _gated_setup_body(
        hd=hd,
        bn=bn,
        q_tile=q_tile,
        chunk_cols=plan.geometry.chunk_cols,
        kv_stage=plan.kv_stage,
        io_dtype=io_dtype,
        mma_dtype=mma_dtype,
        gate_warpgroups=plan.gate_warpgroups,
    )
    core = _gated_producer_body(plan.kv_stage, item_section) + "\n" + consumer
    # An inactive CTA still allocated TMEM: its gate warps arrive on the
    # teardown barrier so warp 1 can free it.
    inactive = (
        f"if warp_idx >= 4:\n    {_gated_teardown_arrive(plan.gate_warpgroups, 4)}"
    )
    # Order: decode, the load-independent setup (shared memory, TMEM
    # allocation, mbarriers), then the scalar loads and their uses -- item 0's
    # TMA loads go out right behind the row base -- and only then the
    # allocation barrier and the TMEM partitions.  ptxas hoists a global load
    # above the (warp-0-only) mbarrier initializations and converts the loaded
    # warp-uniform value to a uniform register right behind it, which stalls
    # every warp for the load's latency before the setup; the CTA-scope
    # acquire-release fence pins the loads behind the setup (~200 ns of device
    # time at the two-tile shapes).
    src = "\n".join(
        [
            "tidx, _, _ = cute.arch.thread_idx()",
            "warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
            *decode_lines,
            setup_a1,
            "cute.arch.fence_acq_rel_cta()",
            *early_cta,
            *cta_lines,
            slices,
            "if flash_body_active:",
            textwrap.indent(item0_loads, "    "),
            setup_a2,
            "if flash_body_active:",
            textwrap.indent(core, "    "),
            "else:",
            textwrap.indent(inactive, "    "),
            _gated_teardown(plan.gate_warpgroups),
        ]
    )
    statements: list[ast.AST] = list(ast.parse(src).body)
    return statements, set(emitter.tensor_names)


# ---------------------------------------------------------------------------
# Codegen entry
# ---------------------------------------------------------------------------


def codegen_gated_attention_flash(cg: GenerateAST) -> bool:
    """Replace the device body with the fused gated attention kernel.

    Mirrors ``cute_flash.codegen_attention_flash``: registers the Q/K/V/O
    tensor args, records the ``helion_flash_gated`` wrapper plan (the host
    builds the tcgen05 tiled MMAs, TMA atoms and smem layouts), appends the
    wrapper-only kernel params and emits the whole device body.
    """
    df = cg.device_function
    plan = df.cute_state.attention_flash_gated_match
    if plan is None:
        return False
    match = plan.match
    host_tensors = _flash_graph_host_tensors(df.codegen.codegen_graphs)
    accesses = {"q": match.q, "k": match.k, "v": match.v, "o": match.o}
    args = {}
    for key, access in accesses.items():
        fake = host_tensors.get(access.name)
        if fake is None:
            return False
        arg = df.tensor_arg(fake, prefer_name=access.name)
        if (
            arg.fake_value.ndim != 3
            or arg.fake_value.dtype != match.io_dtype
            or not arg.fake_value.is_contiguous()
            or int(arg.fake_value.shape[2]) != match.head_dim
        ):
            return False
        args[key] = arg
    body, scalar_tensor_names = emit_gated_flash_device_body(df, plan)
    emit_flash_module_statements(cg)
    wrapper_plan: dict[str, object] = {
        "kind": GATED_WRAPPER_KIND,
        "q_name": args["q"].name,
        "k_name": args["k"].name,
        "v_name": args["v"].name,
        "o_name": args["o"].name,
        "head_dim": match.head_dim,
        "kv_tile": plan.kv_tile,
        "q_tile": plan.q_tile,
        "kv_stage": plan.kv_stage,
        "gate_warpgroups": plan.gate_warpgroups,
        "dtype": _gated_io_dtype_str(match.io_dtype),
        "mma_dtype": _gated_mma_dtype_str(match.io_dtype),
    }
    for key, access in accesses.items():
        wrapper_plan[f"{key}_row_dim"] = access.row_dim
        wrapper_plan[f"{key}_lane_dim"] = access.lane_dim
    cg.cute_wrapper_plans.append(wrapper_plan)
    df.wrapper_only_params.extend(GATED_KERNEL_PARAMS)
    df.placeholder_args.update(arg.name for arg in args.values())
    df.placeholder_args.update(scalar_tensor_names)
    cg.cute_uses_matmul = True
    df.cute_state.attention_flash_threads = plan.threads
    if match.form == "sequence":
        assert match.lane_extent is not None
        df.cute_state.launch_grid_multiplier = match.lane_extent
    df.body = body
    df.preamble = []
    return True


def gated_seed_configs(
    block_size_lists: Sequence[Sequence[int]],
    kv_stage_choices: Sequence[int],
    kv_stage_default: int,
    tiles: Sequence[tuple[int, int]] | None = None,
    kv_stage_choices_by_tile: Mapping[tuple[int, int], Sequence[int]] | None = None,
) -> list[Config]:
    """Generation-zero seeds: every fused tile shape (``tiles`` gives each
    block-size list's (query rows, KV tile)) at every KV depth legal for it and
    every gate warpgroup count above one (default depth and count first).

    A single gate warpgroup stays a search choice but is not seeded: it never
    won a cold autotune of either example at any shape (the gate is issue
    bound, so two or four warpgroups always beat it), and dropping it cuts the
    seed count -- and the cold-autotune wall time -- by about a third.
    """
    from ...runtime.config import Config

    ordered = [
        kv_stage_default,
        *(s for s in kv_stage_choices if s != kv_stage_default),
    ]
    if tiles is None:
        tiles = [(GATED_Q_TILE, max(GATED_KV_TILE_CHOICES))] * len(block_size_lists)
    seeds: list[Config] = []
    for stage in ordered:
        for block_sizes, (q_tile, kv_tile) in zip(block_size_lists, tiles, strict=True):
            if (
                kv_stage_choices_by_tile is not None
                and stage
                not in kv_stage_choices_by_tile.get((q_tile, kv_tile), kv_stage_choices)
            ):
                continue
            wg_choices = gated_gate_warpgroup_choices(q_tile, kv_tile)
            seeded = tuple(g for g in wg_choices if g > 1) or wg_choices
            wg_default = gated_gate_warpgroup_default(seeded)
            for wgs in (wg_default, *(g for g in seeded if g != wg_default)):
                seeds.append(
                    Config.from_dict(
                        {
                            "block_sizes": list(block_sizes),
                            FLASH_KV_STAGE_KEY: stage,
                            FLASH_GATE_WARPGROUPS_KEY: wgs,
                        }
                    )
                )
    return seeds


__all__ = [
    "FLASH_GATE_WARPGROUPS_KEY",
    "GATED_GATE_WARPGROUP_CHOICES",
    "GATED_GATE_WARPGROUP_DEFAULT",
    "GATED_HEAD_DIMS",
    "GATED_KERNEL_PARAMS",
    "GATED_KV_STAGE_CHOICES",
    "GATED_KV_TILE_CHOICES",
    "GATED_Q_TILE",
    "GATED_Q_TILE_CHOICES",
    "GATED_WRAPPER_KIND",
    "GatedAttentionMatch",
    "GatedAttentionPlan",
    "GatedChunkGeometry",
    "GatedSearchSurface",
    "codegen_gated_attention_flash",
    "detect_flash_gated_search_surface",
    "detect_gated_attention_loop",
    "emit_gated_flash_device_body",
    "gated_chunk_geometry",
    "gated_gate_warpgroup_choices",
    "gated_gate_warpgroup_default",
    "gated_kv_stage_choices",
    "gated_kv_stage_default",
    "gated_seed_configs",
    "gated_smem_bytes",
    "match_gated_attention",
]
