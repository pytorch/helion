"""Matcher for the FlexAttention envelope of the fused CuTe flash BACKWARD path.

The target form is the attention-gym ``attn_flex_bwd`` kernel (the backward
twin of ``examples/flex_attention.py``): a 3-axis root grid
``hl.tile([B, Hk, N], block_size=[1, 1, block_n])`` loading one K and one V
tile per work item and carrying fp32 dK/dV accumulators through a unit
``hl.tile(hk * g, hk * g + g, block_size=1)`` loop over the q heads of the
GQA group; inside it two BlockMask-driven Q walks (an outer counted loop
over a 0-d count load ``q_num_blocks[b', h', n_begin // KB]`` with an inner
data-dependent ``hl.tile(start_m, end_m)`` loop per listed Q block: the first
walk the full list, the second the partial list); and a Q-tile body with the
five backward dots

    S^T = K Q^T (fp32) ; dP^T = V dO^T (fp32) ; dV += P^T dO ; dK += dS Q ;
    dQ[b, h, tile_m, :] += (dS^T K) * scale   (``hl.atomic_add`` on fp32)

around the softmax recompute ``P = exp2(score * log2e - (lse_safe * log2e)
[None, :])`` with ``score = score_mod(S^T * scale, b, h, q_idx, kv_idx)``
(``where(mask_mod(...), score, -inf)`` on the partial walk), ``lse_safe =
where(lse == -inf, 0, lse)``, ``dS = P (dP^T - delta[None, :])`` and ``dS =
score_mod_bwd(dS, S^T * scale, b, h, q_idx, kv_idx)`` before the io-dtype
convert.

Like the forward matcher (``flex_match.py``) this one canonicalizes instead
of classifying: coordinates, loop-carried roles, the leading ``* scale``,
the ``* log2e`` pair, the ``-inf`` select and the ``lse_safe`` guard are
recognized structurally and whatever is left between the scaled score and
the exp2 entry becomes the plan's ``score_region`` / ``mask_region``;
whatever is left between ``dS`` and its convert becomes the ``grad_region``
(the Jacobian ``J = score_mod_bwd(1, score, ...)``: a VJP is linear in its
gradient argument, so the emitter multiplies ``dS`` by ``J`` rendered as an
ordinary score region; an identity ``score_mod_bwd`` gives no region).

The regions are built in the hook renderer's frame **transposed**: the
backward tile is S^T (thread row = KV row, chunk columns = Q columns), so the
renderer's row/item coordinate (its ``q_idx`` slot) carries the KV index and
its lane/tile coordinate (the ``kv_idx`` slot) carries the Q index. The
``_Ctx`` of the forward matcher is reused with the grid's KV axis as its
``m_id`` (row) and the inner Q-tile loop as its ``kv_id`` (lane), so the
coordinate leaves land in the swapped slots without any extra mapping; the
emitter passes the renderer the KV tile as ``m_tile_expr`` and the Q tile as
``kv_tile_expr``.
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .attention_plan import FlexAuxBuffer
from .attention_plan import FlexRegion
from .flex_match import _COORD_PLACEHOLDER_NAMES
from .flex_match import _COORD_PLACEHOLDER_SHAPES
from .flex_match import FLEX_KV_BLOCK
from .flex_match import FLEX_Q_BLOCKS
from .flex_match import _const_true
from .flex_match import _Ctx
from .flex_match import _Dep
from .flex_match import _host_tensor_name
from .flex_match import _host_tensor_value
from .flex_match import _int_literal
from .flex_match import _is_call
from .flex_match import _is_full_slice
from .flex_match import _is_neg_inf
from .flex_match import _Leaf
from .flex_match import _load_parts
from .flex_match import _loop_targets
from .flex_match import _region_key
from .flex_match import _RegionBuilder
from .flex_match import _scalar_value
from .flex_match import _strip_identity
from .flex_match import _strip_new_var
from .flex_match import _table_lead_index_ok
from .flex_match import _unit_fixed_block
from .flex_match import _UnmappedRegionValue
from .flex_match import contiguous_strides
from .flex_match import flex_operand_strides

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from torch.fx.node import Argument

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction

_aten = torch.ops.aten
_prims = torch.ops.prims

# Head-dim pairs the fused backward body hosts (the FA4 1-CTA structure).
# Head dims (D of Q / K / dK / dQ, Dv of V / dO / dV) the fused backward
# supports: multiples of 16 whose accumulators fit the 512 TMEM columns (see
# :func:`flex_bwd_tmem_plan`); the shared-memory tiles pad to 64-column
# atoms and the MMAs run the exact extents.
FLEX_BWD_HEAD_DIM_STEP = 16


@dataclasses.dataclass(frozen=True)
class FlexBwdTmemPlan:
    """The TMEM column plan of a flex backward kernel (512 columns x 128
    lanes). S and dP are 128 columns each, P (bf16 pairs, 64 columns) is
    packed into S's columns, dV is Dv wide, dK and dQ D wide.

    ``plain``: S@0, dP@128, then dV, dK, dQ at 32-aligned offsets.
    ``alias_dp``: dQ aliases dP's columns (the dQ MMA runs after the dS pass
    read dP; the next dP MMA waits for the reduce warps to drain dQ);
    D <= 128.
    ``pair`` (D == Dv == 256): the 2-CTA pair kernel's two cluster roles
    (the pair's M=128 MMAs pack a CTA's 64 rows as 128 lanes x N/2 columns).
    dK/dV clusters (64 KV rows per CTA): dK@``dk_off`` 0 and dV@``dv_off``
    128 (128 columns each), S@``s_off`` 256 and dP@``dp_off`` 320 (64 each)
    with a second step parity 128 columns above. dQ clusters (64 Q rows per
    CTA): S^T@0 and dP^T@64 with their parity at 128 / 192, the dQ
    accumulator@``dq_off`` 256 (128 columns, the whole Q tile's walk).
    ``alias_s`` (D == 192 only): dP@64 overlaps S's upper half (the dP MMA
    waits on the ``s_read`` barrier until both softmax halves read S out; P
    packs into S's lower 64 columns, the upper half's store after ``s_read``
    too), dQ runs as three 64-column chunk MMAs through the dP slots after
    the dS pass (``dq_chunk_cols``: the next S waits for the first two
    drains, dP for the third), dV and dK follow at 192. The MMA warp issues
    dK before the next S there, so a single Q stage suffices.
    """

    mode: str
    dp_off: int
    dv_off: int
    dk_off: int
    dq_off: int
    s_off: int = 0

    @property
    def dq_alias(self) -> bool:
        return self.mode not in ("plain", "pair")

    @property
    def p_cols(self) -> tuple[int, int]:
        """TMEM column of each softmax half's packed P (32 columns each)."""
        return (0, 32) if self.mode == "alias_s" else (0, 64)

    @property
    def dq_chunk_cols(self) -> tuple[int, int, int]:
        """alias_s: the TMEM column of each 64-wide dQ chunk -- the dP slot
        above S's upper half, the one on it, and the first again (reused
        once its chunk drained)."""
        assert self.mode == "alias_s"
        return (self.dp_off + 64, self.dp_off, self.dp_off + 64)

    @property
    def p_offsets(self) -> tuple[int, ...]:
        """TMEM column of each 16-wide K block of P for the dV MMA."""
        if self.mode == "alias_s":
            return tuple(k * 8 for k in range(8))
        return tuple((k // 4) * 64 + (k % 4) * 8 for k in range(8))


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def flex_bwd_tmem_plan(head_dim: int, head_dim_v: int) -> FlexBwdTmemPlan | None:
    """See :class:`FlexBwdTmemPlan` (None: the accumulators do not fit)."""
    if (head_dim, head_dim_v) == (256, 256):
        # dK/dV clusters: dK 0, dV 128, S 256 / 384, dP 320 / 448 (two step
        # parities); dQ clusters: S^T 0 / 128, dP^T 64 / 192, dQ 256.
        return FlexBwdTmemPlan("pair", 320, 128, 0, 256, s_off=256)
    dv_off = 256
    dk_off = dv_off + _round_up(head_dim_v, 32)
    dq_off = dk_off + _round_up(head_dim, 32)
    if dq_off + head_dim <= 512:
        return FlexBwdTmemPlan("plain", 128, dv_off, dk_off, dq_off)
    if head_dim <= 128 and dk_off + head_dim <= 512:
        return FlexBwdTmemPlan("alias_dp", 128, dv_off, dk_off, 128)
    # alias_s is the D == 192 plan (its dQ runs as three 64-column chunks
    # through the dP slots, see the emitter); Dv is bounded by dK's columns.
    dv_off = 192
    dk_off = dv_off + _round_up(head_dim_v, 32)
    if head_dim == 192 and dk_off + head_dim <= 512:
        return FlexBwdTmemPlan("alias_s", 64, dv_off, dk_off, 0)
    return None


def flex_bwd_head_dims_supported(head_dim: int, head_dim_v: int) -> bool:
    """Whether the fused backward handles the (D, Dv) pair: multiples of 16
    with both at most 128 (every such pair has a plain or alias_dp TMEM plan
    and fits shared memory), the MLA pair (192, 128) (the alias_s plan) or
    (256, 256) (the 2-CTA pair kernel).
    Other pairs with a TMEM plan on paper (D > 128 with a small Dv, Dv up to
    256) are not emitted: their tiles do not fit shared memory."""
    step = FLEX_BWD_HEAD_DIM_STEP
    if head_dim % step or head_dim_v % step or head_dim < step or head_dim_v < step:
        return False
    if head_dim <= 128 and head_dim_v <= 128:
        return flex_bwd_tmem_plan(head_dim, head_dim_v) is not None
    return (head_dim, head_dim_v) in ((192, 128), (256, 256))


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FlexBwdWalk:
    """The BlockMask Q-side tables as the kernel receives them: counts
    ``[Bm, Hm, C]`` and Q block ids ``[Bm, Hm, C, L]`` (int32) for the
    partial (masked) and full lists, one row per ``kv_block`` KV rows."""

    part_cnt_name: str
    part_idx_name: str
    full_cnt_name: str
    full_idx_name: str
    q_block: int
    kv_block: int
    table_batch: int
    table_heads: int
    table_cols: int


@dataclasses.dataclass(frozen=True)
class FlexBwdPlan:
    """Config-independent facts of a matched flex backward kernel."""

    head_dim: int
    head_dim_v: int
    io_dtype: torch.dtype
    batch: int
    heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    walk: FlexBwdWalk
    # Regions in the renderer's (transposed) frame; see the module docstring.
    score_region: FlexRegion | None
    mask_region: FlexRegion | None
    grad_region: FlexRegion | None
    aux_buffers: tuple[FlexAuxBuffer, ...]
    host_scalars: tuple[str, ...]
    # The QK scale: a literal, or the name of the host scalar argument.
    scale_value: float | None
    scale_arg: str | None
    # ``exp2(score * c - (lse_safe * c)[None, :])``: the factor ``c`` applied
    # to the natural-unit score and LSE (log2(e) for a natural-log LSE).
    log2_factor: float
    # The strides of the six ``[B, H, S, D]`` operands in ``(Q, dO, K, V, dK,
    # dV)`` order (``flex_match.flex_operand_strides`` form; row-major for a
    # contiguous tensor). ``operand_strides`` is ``None`` when all six are
    # contiguous (the rank-3 ``(S, D, B*H)`` TMA maps) and this tuple
    # otherwise (rank-4 ``(S, D, H, B)`` maps built from the strides).
    strides: tuple[tuple[int, ...], ...]
    # The LSE tensor is already in the exp2 base with no-visible-key rows
    # mapped to 0 (the preprocess kernel's ``lse2``): the body stages it as is.
    lse_prescaled: bool = False
    # The regions in the forward frame (row = Q, lanes = KV), rendered by the
    # 2-CTA pair kernel's dQ clusters (``cute_flash_bwd_flex_pair``); None
    # when the frame did not capture (only the pair plan needs them).
    score_region_q: FlexRegion | None = None
    mask_region_q: FlexRegion | None = None
    grad_region_q: FlexRegion | None = None
    regions_q_captured: bool = False

    @property
    def operand_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        q = (self.batch, self.heads, self.q_seq, self.head_dim)
        do = (self.batch, self.heads, self.q_seq, self.head_dim_v)
        k = (self.batch, self.kv_heads, self.kv_seq, self.head_dim)
        v = (self.batch, self.kv_heads, self.kv_seq, self.head_dim_v)
        return (q, do, k, v, k, v)

    @property
    def operand_strides(self) -> tuple[tuple[int, ...], ...] | None:
        if all(
            strides == contiguous_strides(shape)
            for strides, shape in zip(self.strides, self.operand_shapes, strict=True)
        ):
            return None
        return self.strides

    @property
    def group(self) -> int:
        return self.heads // self.kv_heads

    @property
    def num_bh(self) -> int:
        return self.batch * self.heads

    @property
    def num_q_tiles(self) -> int:
        return -(-self.q_seq // 128)

    @property
    def num_kv_tiles(self) -> int:
        return -(-self.kv_seq // 128)

    @property
    def ragged_m(self) -> bool:
        return self.q_seq % 128 != 0

    @property
    def ragged_n(self) -> bool:
        return self.kv_seq % 128 != 0

    @property
    def num_q_tiles_padded(self) -> int:
        """128-row Q tiles the list walk can visit: every listed Q block of
        ``q_block`` rows expands to ``q_block // 128`` tiles, so a ragged
        last block reaches past ``ceil(q_seq / 128)`` (those tiles hold no
        valid rows: the TMA unit zero-fills them and the stagings clamp)."""
        q_block = self.walk.q_block
        return -(-self.q_seq // q_block) * (q_block // 128)

    @property
    def max_steps(self) -> int:
        """Upper bound of the Q-tile steps of one KV tile: every Q block of
        every q head of the group, in 128-row tiles."""
        return self.group * self.num_q_tiles_padded


@dataclasses.dataclass(frozen=True)
class FlexBwdNest:
    outer_graph_id: int
    inner_graph_id: int
    outer_block_id: int
    inner_block_id: int
    masked: bool


@dataclasses.dataclass(frozen=True)
class FlexBwdMatch:
    """Everything the surface and the emitter need from a flex backward kernel."""

    plan: FlexBwdPlan
    b_block_id: int
    hk_block_id: int
    kv_block_id: int
    h_block_id: int
    q_block_id: int
    full_nest: FlexBwdNest
    part_nest: FlexBwdNest
    q_name: str
    k_name: str
    v_name: str
    do_name: str
    lse_name: str
    delta_name: str
    dq_name: str
    dk_name: str
    dv_name: str

    @property
    def loop_block_ids(self) -> tuple[int, ...]:
        ids: list[int] = [self.h_block_id]
        for nest in (self.full_nest, self.part_nest):
            for block_id in (nest.outer_block_id, nest.inner_block_id):
                if block_id not in ids:
                    ids.append(block_id)
        return tuple(ids)

    @property
    def table_names(self) -> tuple[str, str, str, str]:
        walk = self.plan.walk
        return (
            walk.part_cnt_name,
            walk.part_idx_name,
            walk.full_cnt_name,
            walk.full_idx_name,
        )

    @property
    def aux_names(self) -> tuple[str, ...]:
        return tuple(buffer.name for buffer in self.plan.aux_buffers)


class _Outer(NamedTuple):
    loop: torch.fx.Node
    graph_id: int
    block_id: int
    cnt_name: str
    cnt_value: torch.Tensor
    cnt_index: list[object]


class _Inner(NamedTuple):
    loop: torch.fx.Node
    graph_id: int
    block_id: int
    idx_name: str
    idx_value: torch.Tensor
    idx_index: list[object]
    q_block: int
    q_len: int


class _Regions(NamedTuple):
    score: FlexRegion | None
    mask: FlexRegion | None
    grad: FlexRegion | None
    aux: tuple[FlexAuxBuffer, ...]
    host_scalars: tuple[str, ...]
    score_key: tuple[object, ...] | None
    grad_key: tuple[object, ...] | None
    load_nodes: frozenset[torch.fx.Node]


class _Body(NamedTuple):
    q_name: str
    do_name: str
    lse_name: str
    delta_name: str
    dq_name: str
    k_root: torch.fx.Node
    v_root: torch.fx.Node
    io_dtype: torch.dtype
    scale_literal: float | None
    scale_symbol: str | None
    log2_factor: float
    lse_prescaled: bool
    regions: _Regions
    # Placeholder indices (inner body graph) of the dV and dK accumulators.
    acc_slots: tuple[int, int]
    # The same regions in the forward frame (row = Q, lanes = KV): what the
    # 2-CTA pair kernel's dQ clusters render; None when they did not capture.
    regions_q: _Regions | None = None


class _BwdCtx(_Ctx):
    """The forward matcher's context with the backward's axis roles: the
    grid is ``(b, hk, kv)``; ``m_id`` (the renderer's row coordinate) is the
    KV axis, ``kv_id`` (the lane coordinate) the inner Q-tile loop, and
    ``h_id`` the unit q-head loop inside the item (the grid's own head axis
    is the KV head, ``hk_id``)."""

    def __init__(
        self,
        graphs: Sequence[GraphInfo],
        root_grid_ids: Sequence[int],
        env: CompileEnvironment,
        host: HostFunction,
        reasons: list[str] | None,
    ) -> None:
        super().__init__(graphs, root_grid_ids, env, host, reasons)
        self.hk_id = self.h_id
        self.h_id = -1
        self.group = 1


class _BwdRegionBuilder(_RegionBuilder):
    """The forward region builder with a second value leaf: the gradient
    argument of ``score_mod_bwd`` (``grad``), rendered as the literal 1.0 so
    the region evaluates to the Jacobian."""

    def __init__(self, ctx: _Ctx, graph_id: int, score_root: torch.fx.Node) -> None:
        inner = typing.cast("object", _GraphRef(graph_id))
        super().__init__(ctx, inner, score_root)  # pyrefly: ignore[bad-argument-type]
        self.grad_root: torch.fx.Node | None = None

    def _classify_leaf(self, node: torch.fx.Node) -> _Leaf | None:
        if self.grad_root is not None:
            stripped = _strip_new_var(node)
            if node is self.grad_root or stripped is self.grad_root:
                return _Leaf("grad", "grad")
        return super()._classify_leaf(node)

    def _dep(self, node: torch.fx.Node):  # noqa: ANN202
        leaf = self.leaf_of(node)
        if leaf is not None and leaf.kind == "grad":
            return _Dep(frozenset({"score"}), None)
        return super()._dep(node)

    def build(  # type: ignore[override]
        self,
        nodes: list[torch.fx.Node],
        output: torch.fx.Node,
        aux_order: list[str],
        host_scalars: list[str],
        coord_meta: dict[str, object],
    ) -> tuple[torch.fx.Graph, torch.fx.Node]:
        if self.grad_root is None:
            return super().build(nodes, output, aux_order, host_scalars, coord_meta)
        # Temporarily make the gradient leaf render as the literal 1.0: the
        # parent's ``transform`` maps unknown leaf kinds to host scalars, so
        # route it through a fresh ``full([], 1.0)`` node instead.
        graph, result = _build_with_grad_literal(
            self, nodes, output, aux_order, host_scalars, coord_meta
        )
        return graph, result


class _GraphRef(NamedTuple):
    graph_id: int


def _build_with_grad_literal(
    builder: _BwdRegionBuilder,
    nodes: list[torch.fx.Node],
    output: torch.fx.Node,
    aux_order: list[str],
    host_scalars: list[str],
    coord_meta: dict[str, object],
) -> tuple[torch.fx.Graph, torch.fx.Node]:
    """``_RegionBuilder.build`` with the gradient leaf mapped to ``full([],
    1.0)`` (a typed literal the renderer folds)."""
    from ...language import creation_ops
    from ...language import memory_ops
    from .flex_match import _FULL_TARGETS
    from .flex_match import _is_identity_view

    graph = torch.fx.Graph()
    placeholders: dict[str, torch.fx.Node] = {}
    for key in ("score", "b", "h", "q", "kv"):
        placeholder = graph.placeholder(_COORD_PLACEHOLDER_NAMES[key])
        placeholder.meta["val"] = coord_meta[key]
        placeholders[key] = placeholder
    for index, name in enumerate(aux_order):
        placeholder = graph.placeholder(f"aux_{index}")
        placeholder.meta["val"] = builder.aux_nodes[name].meta.get("val")
        placeholder.meta["flex_source_name"] = name
        placeholders[f"aux:{name}"] = placeholder
    for index, name in enumerate(host_scalars):
        placeholder = graph.placeholder(f"host_scalar_{index}")
        placeholder.meta["val"] = builder.host_scalar_nodes[name].meta.get("val")
        placeholder.meta["flex_source_name"] = name
        placeholders[f"hs:{name}"] = placeholder
    mapping: dict[torch.fx.Node, Argument] = {}

    def transform(node: torch.fx.Node) -> Argument:
        if node in mapping:
            return mapping[node]
        leaf = builder.leaf_of(node)
        if leaf is None:
            raise _UnmappedRegionValue(node.name)
        result: Argument
        if leaf.kind == "score":
            result = placeholders["score"]
        elif leaf.kind == "grad":
            # The gradient argument as the Python literal 1.0: the region then
            # evaluates to the Jacobian (a VJP is linear in its gradient), and
            # the pointwise lowerings take a number in any operand position.
            result = 1.0
        elif leaf.kind == "coord":
            result = placeholders[leaf.key]
        else:
            result = placeholders[f"hs:{leaf.key}"]
        mapping[node] = result
        return result

    def is_grad(arg: object) -> bool:
        if not isinstance(arg, torch.fx.Node):
            return False
        leaf = builder.leaf_of(arg)
        return leaf is not None and leaf.kind == "grad"

    for node in nodes:
        if _strip_new_var(node) is not node or _is_identity_view(node):
            source = node.args[0]
            assert isinstance(source, torch.fx.Node)
            mapping[node] = transform(source)
            continue
        if node.target is _aten.sym_size.int:
            continue
        if node.target in (_aten.mul.Tensor, _aten.mul.Scalar) and len(node.args) == 2:
            # ``grad * J`` / ``J * grad``: the factor itself (a VJP is linear in
            # its gradient, so the product with the unit gradient is ``J``).
            lhs, rhs = node.args
            if is_grad(lhs) and not is_grad(rhs):
                mapping[node] = (
                    transform(rhs) if isinstance(rhs, torch.fx.Node) else rhs
                )
                continue
            if is_grad(rhs) and not is_grad(lhs):
                mapping[node] = (
                    transform(lhs) if isinstance(lhs, torch.fx.Node) else lhs
                )
                continue
        if node.target in _FULL_TARGETS or node.target is creation_ops.full:
            target = node.target
            assert callable(target)
            rest = torch.fx.node.map_arg(tuple(node.args[1:]), transform)
            kwargs = torch.fx.node.map_arg(dict(node.kwargs), transform)
            copied = graph.call_function(target, ([], *rest), kwargs)
            copied.meta = dict(node.meta)
            mapping[node] = copied
            continue
        if node.target is memory_ops.load:
            parts = _load_parts(node)
            assert parts is not None
            tensor, indices = parts
            name = _host_tensor_name(tensor)
            assert name is not None
            new_indices: list[object] = []
            for index in indices:
                if isinstance(index, torch.fx.Node):
                    new_indices.append(transform(index))
                else:
                    new_indices.append(index)
            copied = graph.call_function(
                memory_ops.load,
                (placeholders[f"aux:{name}"], new_indices, None, None),  # pyrefly: ignore[bad-argument-type]
            )
            copied.meta = dict(node.meta)
            copied.meta["flex_source_name"] = name
            copied.meta["flex_index_class"] = builder.classify_load(node)
            mapping[node] = copied
            continue
        reads_grad = any(is_grad(arg) for arg in node.args)
        copied = graph.node_copy(node, transform)
        if reads_grad:
            if (
                copied.target in (_aten.mul.Tensor, _aten.add.Tensor)
                and len(copied.args) == 2
                and not isinstance(copied.args[0], torch.fx.Node)
                and isinstance(copied.args[1], torch.fx.Node)
            ):
                # The lowering preparation binds tensor inputs positionally:
                # keep the tensor operand first now that the gradient is a
                # literal.
                copied.args = (copied.args[1], copied.args[0])
            # The copy keeps the source node's prepared lowering, built for
            # two tensor operands; with the gradient now a literal it is
            # stale, so drop it and let the renderer re-prepare it. Nodes
            # whose operands are unchanged keep theirs (the tracer encodes a
            # repeated operand as None next to its prepared lowering, which
            # a fresh preparation could not rebuild).
            copied.meta.pop("lowering", None)
        mapping[node] = copied
    result = mapping[output] if output in mapping else transform(output)
    assert isinstance(result, torch.fx.Node)
    graph.output(result)
    graph.lint()
    return graph, result


# -- entry ---------------------------------------------------------------------


def _host_value_named(ctx: _Ctx, name: str) -> torch.Tensor | None:
    """The fake value of the host tensor ``name`` (any graph)."""
    for info in ctx.graphs.values():
        for node in info.graph.nodes:
            if _host_tensor_name(node) == name:
                return _host_tensor_value(node)
    return None


def match_flex_attention_bwd(
    graphs: Iterable[GraphInfo],
    root_grid_ids: Sequence[int],
    *,
    env: CompileEnvironment,
    host_function: HostFunction | None = None,
    reasons: list[str] | None = None,
) -> FlexBwdMatch | None:
    """Match the flex backward envelope over a kernel's whole graph set.

    ``graphs`` are ``device_ir.graphs`` (surface time) or the codegen copies;
    ``root_grid_ids`` is the kernel's single root grid ``(b, hk, kv)``.
    ``reasons`` collects the first failing condition for diagnostics; without
    it the first decline is logged at DEBUG level
    (``HELION_LOGS=+helion._compiler.cute.flex_bwd_match``).
    """
    logged = reasons is None and log.isEnabledFor(logging.DEBUG)
    if logged:
        reasons = []
    match = _match_flex_attention_bwd(
        graphs, root_grid_ids, env=env, host_function=host_function, reasons=reasons
    )
    if match is None and logged and reasons:
        log.debug("flex backward envelope declined: %s", reasons[0])
    return match


def _match_flex_attention_bwd(
    graphs: Iterable[GraphInfo],
    root_grid_ids: Sequence[int],
    *,
    env: CompileEnvironment,
    host_function: HostFunction | None,
    reasons: list[str] | None,
) -> FlexBwdMatch | None:
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    if host_function is None:
        host_function = HostFunction.current()
    graphs = tuple(graphs)
    if len(root_grid_ids) != 3:
        if reasons is not None:
            reasons.append("flex_bwd: the root grid must have three axes (b, hk, n)")
        return None
    ctx = _BwdCtx(graphs, root_grid_ids, env, host_function, reasons)
    roots = [info for info in graphs if isinstance(info, RootGraphInfo)]
    if len(roots) != 1:
        ctx.decline("flex_bwd: exactly one root graph")
        return None
    root = roots[0]
    for block_id in (ctx.b_id, ctx.hk_id):
        if not _unit_fixed_block(env, block_id):
            ctx.decline(f"flex_bwd: root axis {block_id} is not pinned to block size 1")
            return None
    for info in graphs:
        if isinstance(info, ForLoopGraphInfo) and len(info.block_ids) != 1:
            ctx.decline("flex_bwd: multi-dimensional device loop")
            return None
        if not isinstance(info, (ForLoopGraphInfo, RootGraphInfo)):
            ctx.decline(f"flex_bwd: unsupported graph {type(info).__name__}")
            return None

    root_facts = _match_root(ctx, root)
    if root_facts is None:
        return None
    group_loop, k_root, v_root, dv_full, dk_full, dk_factor = root_facts
    outers = _match_group_graph(ctx, group_loop)
    if outers is None:
        return None
    nests: list[tuple[_Outer, _Inner, _Body]] = []
    for outer in outers:
        inner = _match_outer_graph(ctx, outer)
        if inner is None:
            return None
        body = _match_inner_body(ctx, inner, k_root, v_root)
        if body is None:
            return None
        nests.append((outer, inner, body))
    return _assemble(
        ctx, root, group_loop, k_root, v_root, dv_full, dk_full, dk_factor, nests
    )


# -- root --------------------------------------------------------------------------


def _match_root(
    ctx: _BwdCtx, root: GraphInfo
) -> (
    tuple[
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        object,
    ]
    | None
):
    """The root: ``k_j = key[b, hk, tile_n, :]``, ``v_j = value[...]``, two fp32
    zero accumulators, one unit loop over the q heads of the group carrying
    them, and the dV / dK stores of its results."""
    from ...language import creation_ops
    from ...language import memory_ops
    from ...language._tracing_ops import _for_loop
    from ..device_ir import ForLoopGraphInfo

    loops: list[torch.fx.Node] = []
    loads: list[torch.fx.Node] = []
    fulls: list[torch.fx.Node] = []
    stores: list[torch.fx.Node] = []
    for node in root.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is _for_loop:
            loops.append(node)
        elif node.target in _loop_targets():
            ctx.decline("flex_bwd: unsupported control flow in the root graph")
            return None
        elif node.target is memory_ops.load:
            loads.append(node)
        elif node.target is creation_ops.full or node.target is _aten.full.default:
            fulls.append(node)
        elif node.target is memory_ops.store:
            stores.append(node)
    if len(loops) != 1:
        ctx.decline(
            f"flex_bwd: the root must hold one q-head loop (found {len(loops)})"
        )
        return None
    if len(loads) != 2 or len(stores) != 2 or len(fulls) != 2:
        ctx.decline("flex_bwd: the root must load K and V, zero dK/dV and store them")
        return None
    k_root = v_root = None
    for load in loads:
        parts = _load_parts(load)
        if parts is None:
            ctx.decline("flex_bwd: K/V load with a mask or eviction policy")
            return None
        tensor, indices = parts
        value = _host_tensor_value(tensor)
        if (
            value is None
            or value.ndim != 4
            or len(indices) != 4
            or not ctx.unit_axis_index(indices[0], ctx.b_id)
            or not ctx.unit_axis_index(indices[1], ctx.hk_id)
            or ctx.block_symnode_id(indices[2]) != ctx.m_id
            or not _is_full_slice(indices[3])
        ):
            ctx.decline("flex_bwd: K/V load is not key[b, hk, tile_n, :]")
            return None
    k_root, v_root = loads  # order resolved by the dots below
    for acc in fulls:
        if len(acc.args) < 3 or acc.args[1] != 0.0 or acc.args[2] is not torch.float32:
            ctx.decline("flex_bwd: the dK/dV accumulators must be fp32 zero tiles")
            return None
    loop = loops[0]
    graph_id, begins, ends, inputs = (
        loop.args[0],
        loop.args[1],
        loop.args[2],
        loop.args[3],
    )
    if not (
        isinstance(graph_id, int)
        and isinstance(ctx.graphs.get(graph_id), ForLoopGraphInfo)
        and isinstance(begins, (list, tuple))
        and isinstance(ends, (list, tuple))
        and len(begins) == 1
        and len(ends) == 1
        and isinstance(inputs, (list, tuple))
    ):
        ctx.decline("flex_bwd: malformed q-head loop")
        return None
    info = ctx.graphs[graph_id]
    assert isinstance(info, ForLoopGraphInfo)
    h_block_id = info.block_ids[0]
    if not _unit_fixed_block(ctx.env, h_block_id):
        ctx.decline("flex_bwd: the q-head loop is not a unit tile loop")
        return None
    ctx.h_id = h_block_id
    # begin = hk_begin * g (or hk_begin for g == 1); end = begin + g.
    group = _group_of_bounds(ctx, begins[0], ends[0])
    if group is None:
        ctx.decline("flex_bwd: the q-head loop bounds are not hk * g .. hk * g + g")
        return None
    ctx.group = group
    # The accumulators are loop inputs whose results the stores consume.
    dv_full = dk_full = None
    dk_factor: object = None
    for store in stores:
        value = store.args[2]
        index = store.args[1]
        if not (
            isinstance(value, torch.fx.Node)
            and value.target is _prims.convert_element_type.default
            and isinstance(index, (list, tuple))
            and len(index) == 4
            and ctx.block_symnode_id(index[0]) == ctx.b_id
            and ctx.block_symnode_id(index[1]) == ctx.hk_id
            and ctx.block_symnode_id(index[2]) == ctx.m_id
            and _is_full_slice(index[3])
        ):
            ctx.decline("flex_bwd: dK/dV store is not out[tile_b, tile_hk, tile_n, :]")
            return None
        inner: object = _strip_identity(value.args[0])
        scaled = False
        factor: object = None
        if _is_call(inner, _aten.mul.Tensor):
            assert isinstance(inner, torch.fx.Node)
            scaled = True
            factor = inner.args[1]
            inner = _strip_identity(inner.args[0])
        phi_input = _phi_source(inner)
        if phi_input is None or phi_input not in inputs:
            ctx.decline("flex_bwd: dK/dV store does not consume the loop's accumulator")
            return None
        if scaled:
            dk_full = phi_input
            dk_factor = factor
        else:
            dv_full = phi_input
    if dv_full is None or dk_full is None or dv_full is dk_full:
        ctx.decline("flex_bwd: expected one scaled (dK) and one plain (dV) store")
        return None
    if dv_full not in fulls or dk_full not in fulls:
        ctx.decline("flex_bwd: the stored accumulators are not the root's zero tiles")
        return None
    return loop, k_root, v_root, dv_full, dk_full, dk_factor


def _phi_source(node: object) -> torch.fx.Node | None:
    """The loop input a ``_phi(init, getitem(loop, i))`` result started from."""
    from ...language._tracing_ops import _phi

    node = _strip_new_var(node)
    if not _is_call(node, _phi):
        return None
    assert isinstance(node, torch.fx.Node)
    init = node.args[0]
    return init if isinstance(init, torch.fx.Node) else None


def _group_of_bounds(ctx: _BwdCtx, begin: object, end: object) -> int | None:
    """``g`` of ``begin = hk_begin * g`` and ``end = begin + g``."""
    import operator

    from ...language.tile_ops import tile_begin

    def is_hk_begin(node: object) -> bool:
        return _is_call(node, tile_begin) and ctx.unit_axis_index(node, ctx.hk_id)

    g: int | None = None
    if is_hk_begin(begin):
        g = 1
    elif _is_call(begin, operator.mul) and isinstance(begin, torch.fx.Node):
        lhs, rhs = begin.args
        if is_hk_begin(lhs) and isinstance(rhs, int):
            g = rhs
        elif is_hk_begin(rhs) and isinstance(lhs, int):
            g = lhs
    if g is None:
        return None
    if not (_is_call(end, operator.add) and isinstance(end, torch.fx.Node)):
        return None
    lhs, rhs = end.args
    if lhs is begin and rhs == g:
        return g
    if rhs is begin and lhs == g:
        return g
    return None


# -- group loop graph ----------------------------------------------------------


def _match_group_graph(ctx: _BwdCtx, loop: torch.fx.Node) -> list[_Outer] | None:
    """The q-head body: ``bcast_h``, two counted list walks (count =
    load(cnt[b', h', n_begin // KB]), nothing else."""
    import operator

    from ...language import memory_ops
    from ...language import tile_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _new_var
    from ...language._tracing_ops import _phi
    from ..device_ir import ForLoopGraphInfo

    graph_id = loop.args[0]
    assert isinstance(graph_id, int)
    graph = ctx.graphs[graph_id].graph
    allowed = (
        _new_var,
        _get_symnode,
        _host_tensor,
        tile_ops.tile_begin,
        tile_ops.tile_id,
        tile_ops.tile_index,
        memory_ops.load,
        _prims.iota.default,
        _aten.add.Tensor,
        _for_loop,
        operator.getitem,
        operator.mod,
        operator.add,
        _phi,
    )
    from ...language import view_ops

    loops: list[torch.fx.Node] = []
    loads: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is view_ops.subscript or node.target in (
            _aten.unsqueeze.default,
            _aten.view.default,
            _aten.reshape.default,
        ):
            continue
        if node.target not in allowed:
            ctx.decline(f"flex_bwd: q-head body has {node.target}")
            return None
        if node.target is _for_loop:
            loops.append(node)
        elif node.target is memory_ops.load:
            loads.append(node)
    if len(loops) != 2 or len(loads) != 2:
        ctx.decline(
            "flex_bwd: the q-head body must hold two list walks and their counts"
        )
        return None
    outers: list[_Outer] = []
    for node in loops:
        inner_graph_id, begins, ends = node.args[0], node.args[1], node.args[2]
        if not isinstance(inner_graph_id, int) or begins != [0]:
            ctx.decline("flex_bwd: the outer list loop must start at 0")
            return None
        if not isinstance(ends, (list, tuple)) or len(ends) != 1:
            ctx.decline("flex_bwd: the outer list loop needs one bound")
            return None
        parts = _load_parts(ends[0])
        if parts is None or ends[0] not in loads:
            ctx.decline("flex_bwd: the outer list bound is not a table load")
            return None
        tensor, indices = parts
        value = _host_tensor_value(tensor)
        name = _host_tensor_name(tensor)
        if (
            value is None
            or name is None
            or value.ndim != 3
            or value.dtype != torch.int32
            or len(indices) != 3
        ):
            ctx.decline("flex_bwd: the count table must be a 3-D int32 tensor")
            return None
        info = ctx.graphs.get(inner_graph_id)
        if not isinstance(info, ForLoopGraphInfo):
            ctx.decline("flex_bwd: outer loop graph missing")
            return None
        if _col_index_block(ctx, indices[2]) is None:
            ctx.decline("flex_bwd: table column index is not n_begin // KB")
            return None
        outers.append(
            _Outer(node, inner_graph_id, info.block_ids[0], name, value, indices)
        )
    return outers


def _col_index_block(ctx: _BwdCtx, node: object) -> int | None:
    """``KB`` of a table column index ``n_begin // KB`` (0 for a literal 0 on
    a one-column table)."""
    literal = _int_literal(node)
    if literal is not None:
        return 0 if literal == 0 else None
    sym = ctx.scalar_sym(node)
    if (
        sym is None
        or sym.block_id != ctx.m_id
        or sym.role != "begin"
        or sym.mod is not None
    ):
        return None
    return sym.div


# -- outer list graph ----------------------------------------------------------


def _match_outer_graph(ctx: _BwdCtx, outer: _Outer) -> _Inner | None:
    """``start_m = load(idx[b', h', col, tile_id]) * QB``; ``end_m =
    minimum(start_m + QB, full([], M))``; one inner ``_for_loop([start], [end])``."""
    import operator

    from ...language import creation_ops
    from ...language import memory_ops
    from ...language import tile_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _new_var
    from ...language._tracing_ops import _phi
    from ..device_ir import ForLoopGraphInfo

    allowed = (
        _new_var,
        _get_symnode,
        _host_tensor,
        tile_ops.tile_id,
        tile_ops.tile_begin,
        memory_ops.load,
        _aten.mul.Tensor,
        _aten.add.Tensor,
        _aten.full.default,
        creation_ops.full,
        _aten.minimum.default,
        _for_loop,
        operator.getitem,
        _phi,
    )
    graph = ctx.graphs[outer.graph_id].graph
    inner_loops: list[torch.fx.Node] = []
    loads: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in allowed:
            ctx.decline(f"flex_bwd: list body has {node.target}")
            return None
        if node.target is _for_loop:
            inner_loops.append(node)
        elif node.target is memory_ops.load:
            loads.append(node)
    if len(inner_loops) != 1 or len(loads) != 1:
        ctx.decline("flex_bwd: the list body must hold one Q loop and one index load")
        return None
    loop = inner_loops[0]
    graph_id, begins, ends = loop.args[0], loop.args[1], loop.args[2]
    if (
        not isinstance(graph_id, int)
        or not isinstance(begins, (list, tuple))
        or not isinstance(ends, (list, tuple))
        or len(begins) != 1
        or len(ends) != 1
    ):
        ctx.decline("flex_bwd: the Q loop must have one begin and one end")
        return None
    start, end = begins[0], ends[0]
    if not (isinstance(start, torch.fx.Node) and _is_call(start, _aten.mul.Tensor)):
        ctx.decline("flex_bwd: Q loop begin is not idx * QB")
        return None
    if len(start.args) != 2:
        ctx.decline("flex_bwd: Q loop begin is not idx * QB")
        return None
    load_arg, qb_arg = start.args
    if _is_call(qb_arg, memory_ops.load):
        load_arg, qb_arg = qb_arg, load_arg
    if load_arg is not loads[0] or not isinstance(qb_arg, int) or qb_arg <= 0:
        ctx.decline("flex_bwd: QB must be a specialized positive int")
        return None
    q_block = qb_arg
    parts = _load_parts(load_arg)
    assert parts is not None
    tensor, indices = parts
    value = _host_tensor_value(tensor)
    name = _host_tensor_name(tensor)
    if (
        value is None
        or name is None
        or value.ndim != 4
        or value.dtype != torch.int32
        or len(indices) != 4
    ):
        ctx.decline("flex_bwd: the index table must be a 4-D int32 tensor")
        return None
    if not ctx.unit_axis_index(indices[3], outer.block_id):
        ctx.decline("flex_bwd: the index table's last index is not the list position")
        return None
    if not (isinstance(end, torch.fx.Node) and _is_call(end, _aten.minimum.default)):
        ctx.decline("flex_bwd: Q loop end is not minimum(start + QB, M)")
        return None
    if len(end.args) != 2:
        ctx.decline("flex_bwd: Q loop end is not minimum(start + QB, M)")
        return None
    sum_node, m_node = end.args
    if _scalar_value(sum_node) is not None:
        sum_node, m_node = m_node, sum_node
    q_len_value = _scalar_value(m_node)
    if (
        not _is_call(sum_node, _aten.add.Tensor)
        or q_len_value is None
        or q_len_value != int(q_len_value)
    ):
        ctx.decline("flex_bwd: Q loop end is not minimum(start + QB, M)")
        return None
    assert isinstance(sum_node, torch.fx.Node)
    add_args = list(sum_node.args)
    if start not in add_args:
        ctx.decline("flex_bwd: Q loop end does not add QB to the begin")
        return None
    add_args.remove(start)
    if add_args != [q_block]:
        ctx.decline("flex_bwd: Q loop end adds a different block size")
        return None
    info = ctx.graphs.get(graph_id)
    if not isinstance(info, ForLoopGraphInfo):
        ctx.decline("flex_bwd: Q loop graph missing")
        return None
    block_id = info.block_ids[0]
    if ctx.kv_id is None:
        ctx.kv_id = block_id
    elif ctx.kv_id != block_id:
        ctx.decline("flex_bwd: the two Q loops use different block sizes")
        return None
    return _Inner(
        loop, graph_id, block_id, name, value, indices, q_block, int(q_len_value)
    )


# -- inner body ----------------------------------------------------------------


def _match_inner_body(
    ctx: _BwdCtx, inner: _Inner, k_root: torch.fx.Node, v_root: torch.fx.Node
) -> _Body | None:
    from ...language import matmul_ops
    from ...language import memory_ops
    from ...language._tracing_ops import _mask_to
    from ...language.atomic_ops import atomic_add

    graph_id = inner.graph_id
    graph = ctx.graphs[graph_id].graph
    dots: list[torch.fx.Node] = []
    loads: list[torch.fx.Node] = []
    atomics: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is matmul_ops.dot:
            dots.append(node)
        elif node.target is memory_ops.load:
            loads.append(node)
        elif node.target is atomic_add:
            atomics.append(node)
        elif node.target is memory_ops.store or node.target in _loop_targets():
            ctx.decline(f"flex_bwd: Q body has {node.target}")
            return None
        elif node.target is _mask_to:
            ctx.decline("flex_bwd: Q body has a masked value")
            return None
    if len(dots) != 5 or len(atomics) != 1:
        ctx.decline("flex_bwd: the Q body needs five hl.dot calls and one atomic add")
        return None

    # Resolve K / V placeholders to the root loads.
    def resolves_to(node: object, root_node: torch.fx.Node) -> bool:
        gid, resolved = ctx.resolve(graph_id, node)
        return resolved is root_node and gid not in ctx.parents

    # S^T = dot(K, permute(Q)) and dP^T = dot(V, permute(dO)): fp32 out, no acc.
    s_dot = dp_dot = dv_dot = dk_dot = dq_dot = None
    for dot in dots:
        a, b = dot.args[0], dot.args[1]
        acc = dot.args[2] if len(dot.args) > 2 else None
        out_dtype = dot.args[3] if len(dot.args) > 3 else None
        if out_dtype not in (None, torch.float32):
            ctx.decline("flex_bwd: hl.dot out_dtype must be fp32")
            return None
        b_perm = _is_call(b, _aten.permute.default, _aten.transpose.int)
        a_perm = _is_call(a, _aten.permute.default, _aten.transpose.int)
        if acc is None and b_perm and resolves_to(a, k_root):
            s_dot = dot
        elif acc is None and b_perm and resolves_to(a, v_root):
            dp_dot = dot
        elif acc is None and a_perm and resolves_to(b, k_root):
            dq_dot = dot
        elif acc is not None and _is_call(a, _prims.convert_element_type.default):
            # dV accumulates P^T (the converted exp2) with dO; dK accumulates
            # the converted dS with Q.
            assert isinstance(a, torch.fx.Node)
            if _is_call(_strip_new_var(a.args[0]), _aten.exp2.default):
                dv_dot = dot
            else:
                dk_dot = dot
    if any(d is None for d in (s_dot, dp_dot, dv_dot, dk_dot, dq_dot)):
        # The K / V loads may have been matched in the other order.
        if (
            s_dot is None
            and dp_dot is None
            and any(resolves_to(d.args[0], v_root) for d in dots)
        ):
            return _match_inner_body(ctx, inner, v_root, k_root)
        ctx.decline(
            "flex_bwd: cannot identify the five backward dots "
            + str(
                {
                    "s": s_dot is not None,
                    "dp": dp_dot is not None,
                    "dv": dv_dot is not None,
                    "dk": dk_dot is not None,
                    "dq": dq_dot is not None,
                }
            )
        )
        return None
    assert s_dot and dp_dot and dv_dot and dk_dot and dq_dot
    if any(not _permute_is_transpose(d.args[1]) for d in (s_dot, dp_dot)) or (
        not _permute_is_transpose(dq_dot.args[0])
    ):
        ctx.decline("flex_bwd: operands must be transposed with permute([1, 0])")
        return None

    # Q / dO / LSE / delta loads: [b, h, tile_m, (:)].
    def row_load(node: object, ndim: int) -> tuple[str, torch.Tensor] | None:
        parts = _load_parts(node)
        if parts is None:
            return None
        tensor, indices = parts
        value = _host_tensor_value(tensor)
        name = _host_tensor_name(tensor)
        if (
            value is None
            or name is None
            or value.ndim != ndim
            or len(indices) != ndim
            or not ctx.unit_axis_index(indices[0], ctx.b_id)
            or not ctx.unit_axis_index(indices[1], ctx.h_id)
            or ctx.block_symnode_id(indices[2]) != ctx.kv_id
            or (ndim == 4 and not _is_full_slice(indices[3]))
        ):
            return None
        return name, value

    q_perm = s_dot.args[1]
    do_perm = dp_dot.args[1]
    assert isinstance(q_perm, torch.fx.Node) and isinstance(do_perm, torch.fx.Node)
    q_load = q_perm.args[0]
    do_load = do_perm.args[0]
    q_facts = row_load(q_load, 4)
    do_facts = row_load(do_load, 4)
    if q_facts is None or do_facts is None:
        ctx.decline("flex_bwd: Q / dO load is not query[b, h, tile_m, :]")
        return None
    if dv_dot.args[1] is not do_load or dk_dot.args[1] is not q_load:
        ctx.decline("flex_bwd: dV must consume dO and dK must consume Q")
        return None
    q_value = q_facts[1]
    io_dtype = q_value.dtype

    # Scaled score and its consumers.
    s_users = list(s_dot.users)
    if len(s_users) != 1:
        ctx.decline("flex_bwd: the S^T score must have one consumer")
        return None
    scale_literal: float | None = None
    scale_symbol: str | None = None
    score_root = s_users[0]
    if not (_is_call(score_root, _aten.mul.Tensor) and len(score_root.args) == 2):
        ctx.decline("flex_bwd: S^T must be scaled (s * scale) before the mods")
        return None
    other = score_root.args[1] if score_root.args[0] is s_dot else score_root.args[0]
    if isinstance(other, (int, float)) and not isinstance(other, bool):
        scale_literal = float(other)
    else:
        scale_symbol = ctx.host_scalar_name(other)
        if scale_symbol is None:
            ctx.decline("flex_bwd: the scale is neither a literal nor a host scalar")
            return None

    # P = exp2(score * c - (lse_safe * c)[None, :]).
    exp2_nodes = [
        n for n in graph.nodes if _is_call(n, _aten.exp2.default) and _consumes_sub(n)
    ]
    p_node = None
    for candidate in exp2_nodes:
        if any(
            _is_call(u, _prims.convert_element_type.default) and u is dv_dot.args[0]
            for u in candidate.users
        ):
            p_node = candidate
            break
    if p_node is None:
        ctx.decline("flex_bwd: cannot find P = exp2(...) feeding dV")
        return None
    sub_node = p_node.args[0]
    assert isinstance(sub_node, torch.fx.Node)
    lhs, rhs = sub_node.args[:2]
    log2_factor, score_src = _peel_factor(lhs)
    if log2_factor is None or score_src is None:
        ctx.decline("flex_bwd: the exp2 argument is not score * c - lse term")
        return None
    lse_name = _match_lse_term(ctx, rhs, log2_factor)
    if lse_name is None:
        ctx.decline(
            "flex_bwd: the LSE term is neither lse2[None, :] nor "
            "(where(lse == -inf, 0, lse) * c)[None, :]"
        )
        return None
    lse_prescaled = _lse_prescaled(rhs)
    score_src = _strip_new_var(score_src)
    mask_pred: object = None
    if _is_call(score_src, _aten.where.self, _aten.where.ScalarOther):
        assert isinstance(score_src, torch.fx.Node)
        if len(score_src.args) == 3 and _is_neg_inf(score_src.args[2]):
            mask_pred, score_src = score_src.args[0], score_src.args[1]
            if _const_true(mask_pred):
                mask_pred = None
            score_src = _strip_new_var(score_src)

    # dS = P * (dP^T - delta[None, :]); grad region; convert.
    ds_c = dk_dot.args[0]
    if not (
        isinstance(ds_c, torch.fx.Node)
        and ds_c.target is _prims.convert_element_type.default
        and ds_c.args[1] == io_dtype
    ):
        ctx.decline("flex_bwd: dK must consume dS converted to the io dtype")
        return None
    dq_perm = dq_dot.args[0]
    assert isinstance(dq_perm, torch.fx.Node)
    if dq_perm.args[0] is not ds_c:
        ctx.decline("flex_bwd: dQ must consume the transposed converted dS")
        return None
    ds_src = _strip_new_var(ds_c.args[0])
    ds_node = _find_ds(graph, p_node, dp_dot)
    if ds_node is None:
        ctx.decline("flex_bwd: cannot find dS = P * (dP - delta[None, :])")
        return None
    delta_facts = row_load(_delta_load_of(ds_node, dp_dot), 3)
    if delta_facts is None:
        ctx.decline("flex_bwd: the delta term is not delta[b, h, tile_m][None, :]")
        return None
    lse_facts = row_load(_lse_load_of(ctx, rhs), 3)
    if lse_facts is None or lse_facts[0] != lse_name:
        ctx.decline("flex_bwd: the LSE load is not lse[b, h, tile_m]")
        return None
    grad_src: object = None if ds_src is ds_node else ds_src

    # atomic_add(dq, [b, h, tile_m, :], dq_dot * scale)
    atomic = atomics[0]
    dq_base = atomic.args[0]
    dq_name = _host_tensor_name(dq_base)
    dq_value = _host_tensor_value(dq_base)
    index = atomic.args[1]
    value = atomic.args[2]
    if (
        dq_name is None
        or dq_value is None
        or dq_value.ndim != 4
        or dq_value.dtype != torch.float32
        or not isinstance(index, (list, tuple))
        or len(index) != 4
        or not ctx.unit_axis_index(index[0], ctx.b_id)
        or not ctx.unit_axis_index(index[1], ctx.h_id)
        or ctx.block_symnode_id(index[2]) != ctx.kv_id
        or not _is_full_slice(index[3])
    ):
        ctx.decline(
            "flex_bwd: the dQ atomic is not atomic_add(dq, [b, h, tile_m, :], ...)"
        )
        return None
    if not (
        isinstance(value, torch.fx.Node)
        and _is_call(value, _aten.mul.Tensor)
        and value.args[0] is dq_dot
        and _same_scale(ctx, value.args[1], scale_literal, scale_symbol)
    ):
        ctx.decline("flex_bwd: the dQ atomic value is not (dS^T K) * scale")
        return None

    # Regions.
    regions = _capture_bwd_regions(
        ctx, graph_id, s_dot, score_root, score_src, mask_pred, ds_node, grad_src
    )
    if regions is None:
        return None
    # The forward-frame regions (``coord_of_block`` with the axis roles
    # swapped: the Q axis in the renderer's row slot); their failure to
    # capture only matters to the pair kernel, which checks for them.
    m_id, kv_id = ctx.m_id, ctx.kv_id
    assert kv_id is not None
    ctx.m_id, ctx.kv_id = kv_id, m_id
    try:
        regions_q = _capture_bwd_regions(
            ctx,
            graph_id,
            s_dot,
            score_root,
            score_src,
            mask_pred,
            ds_node,
            grad_src,
        )
    finally:
        ctx.m_id, ctx.kv_id = m_id, kv_id
    # Closed world of the body's loads.
    allowed_loads = {q_load, do_load}
    for n in loads:
        parts = _load_parts(n)
        if parts is not None:
            name = _host_tensor_name(parts[0])
            if name in (lse_facts[0], delta_facts[0]):
                allowed_loads.add(n)
    if any(n not in allowed_loads and n not in regions.load_nodes for n in loads):
        ctx.decline("flex_bwd: the Q body loads a tensor outside the regions")
        return None
    # Accumulator placeholders: dv_dot / dk_dot accumulate loop placeholders.
    dv_slot = ctx.placeholder_index(graph_id, dv_dot.args[2])
    dk_slot = ctx.placeholder_index(graph_id, dk_dot.args[2])
    if dv_slot is None or dk_slot is None:
        ctx.decline("flex_bwd: dV / dK must accumulate the loop-carried tiles")
        return None
    return _Body(
        q_facts[0],
        do_facts[0],
        lse_facts[0],
        delta_facts[0],
        dq_name,
        k_root,
        v_root,
        io_dtype,
        scale_literal,
        scale_symbol,
        log2_factor,
        lse_prescaled,
        regions,
        (dv_slot, dk_slot),
        regions_q=regions_q,
    )


def _permute_is_transpose(node: object) -> bool:
    if _is_call(node, _aten.permute.default):
        assert isinstance(node, torch.fx.Node)
        return node.args[1] == [1, 0]
    if _is_call(node, _aten.transpose.int):
        assert isinstance(node, torch.fx.Node)
        return tuple(node.args[1:3]) in ((0, 1), (1, 0))
    return False


def _consumes_sub(node: torch.fx.Node) -> bool:
    return len(node.args) == 1 and _is_call(node.args[0], _aten.sub.Tensor)


def _peel_factor(node: object) -> tuple[float | None, object]:
    """``x * c`` -> ``(c, x)`` for a literal ``c``."""
    node = _strip_new_var(node)
    if not (_is_call(node, _aten.mul.Tensor) and isinstance(node, torch.fx.Node)):
        return None, None
    lhs, rhs = node.args[:2]
    if isinstance(rhs, (int, float)) and not isinstance(rhs, bool):
        return float(rhs), lhs
    if isinstance(lhs, (int, float)) and not isinstance(lhs, bool):
        return float(lhs), rhs
    return None, None


def _match_lse_term(ctx: _BwdCtx, node: object, factor: float) -> str | None:
    """The LSE name of ``lse2[None, :]`` (the preprocess kernel's exp2-base row
    statistic, ``lse_prescaled``) or of ``(where(lse == -inf, 0.0, lse) *
    factor)[None, :]`` (converted in the body)."""
    from ...language import view_ops

    node = _strip_new_var(node)
    if not (_is_call(node, view_ops.subscript) and isinstance(node, torch.fx.Node)):
        return None
    if node.args[1] != [None, slice(None)]:
        return None
    plain = _load_parts(node.args[0])
    if plain is not None:
        return _host_tensor_name(plain[0])
    inner_factor, lse_safe = _peel_factor(node.args[0])
    if inner_factor != factor:
        return None
    lse_safe = _strip_new_var(lse_safe)
    if not (
        _is_call(lse_safe, _aten.where.self) and isinstance(lse_safe, torch.fx.Node)
    ):
        return None
    cond, zero, lse_load = lse_safe.args[:3]
    if _scalar_value(zero) != 0.0:
        return None
    if not (_is_call(cond, _aten.eq.Scalar) and isinstance(cond, torch.fx.Node)):
        return None
    if cond.args[0] is not lse_load or not _is_neg_inf(cond.args[1]):
        return None
    parts = _load_parts(lse_load)
    if parts is None:
        return None
    return _host_tensor_name(parts[0])


def _lse_load_of(ctx: _BwdCtx, node: object) -> object:
    node = _strip_new_var(node)
    assert isinstance(node, torch.fx.Node)
    if _load_parts(node.args[0]) is not None:
        return node.args[0]
    _factor, lse_safe = _peel_factor(node.args[0])
    lse_safe = _strip_new_var(lse_safe)
    assert isinstance(lse_safe, torch.fx.Node)
    return lse_safe.args[2]


def _lse_prescaled(node: object) -> bool:
    """True when the exp2's LSE term is the plain ``lse2[None, :]`` load."""
    node = _strip_new_var(node)
    assert isinstance(node, torch.fx.Node)
    return _load_parts(node.args[0]) is not None


def _find_ds(
    graph: torch.fx.Graph, p_node: torch.fx.Node, dp_dot: torch.fx.Node
) -> torch.fx.Node | None:
    """``mul(p, sub(dp, subscript(delta, [None, :])))`` (either operand order)."""
    for user in p_node.users:
        if not _is_call(user, _aten.mul.Tensor):
            continue
        other = user.args[1] if user.args[0] is p_node else user.args[0]
        other = _strip_new_var(other)
        if _is_call(other, _aten.sub.Tensor) and isinstance(other, torch.fx.Node):
            if other.args[0] is dp_dot:
                return user
    return None


def _delta_load_of(ds_node: torch.fx.Node, dp_dot: torch.fx.Node) -> object:
    """The ``delta`` load of ``mul(p, sub(dp, subscript(delta, [None, :])))``."""
    from ...language import view_ops

    for arg in ds_node.args[:2]:
        sub = _strip_new_var(arg)
        if _is_call(sub, _aten.sub.Tensor):
            assert isinstance(sub, torch.fx.Node)
            if sub.args[0] is not dp_dot:
                return None
            bcast = _strip_new_var(sub.args[1])
            if not (
                _is_call(bcast, view_ops.subscript) and isinstance(bcast, torch.fx.Node)
            ):
                return None
            if bcast.args[1] != [None, slice(None)]:
                return None
            return bcast.args[0]
    return None


def _same_scale(
    ctx: _BwdCtx, node: object, literal: float | None, symbol: str | None
) -> bool:
    if literal is not None:
        return isinstance(node, (int, float)) and float(node) == literal
    return ctx.host_scalar_name(node) == symbol


# -- regions -----------------------------------------------------------------


def _capture_bwd_regions(
    ctx: _BwdCtx,
    graph_id: int,
    s_dot: torch.fx.Node,
    score_root: torch.fx.Node,
    score_src: object,
    mask_pred: object,
    ds_node: torch.fx.Node,
    grad_src: object,
) -> _Regions | None:
    builder = _BwdRegionBuilder(ctx, graph_id, score_root)
    score_nodes = mask_nodes = grad_nodes = None
    if score_src is not score_root:
        score_nodes = builder.collect(score_src)
        if score_nodes is None:
            ctx.decline(builder.failed or "flex_bwd: score region rejected")
            return None
        if not any(leaf.kind == "score" for leaf in builder.leaves.values()):
            ctx.decline("flex_bwd: the score region does not read the score")
            return None
    if mask_pred is not None:
        mask_nodes = builder.collect(mask_pred)
        if mask_nodes is None:
            ctx.decline(builder.failed or "flex_bwd: mask region rejected")
            return None
    if grad_src is not None:
        builder.grad_root = ds_node
        grad_nodes = builder.collect(grad_src)
        if grad_nodes is None:
            ctx.decline(builder.failed or "flex_bwd: score_mod_bwd region rejected")
            return None
        if not any(leaf.kind == "grad" for leaf in builder.leaves.values()):
            ctx.decline("flex_bwd: score_mod_bwd does not read its gradient argument")
            return None
    coord_meta: dict[str, object] = {"score": s_dot.meta.get("val")}
    for node, leaf in builder.leaves.items():
        if leaf.kind == "coord" and leaf.key not in coord_meta:
            coord_meta[leaf.key] = node.meta.get("val")
    for key, shape in _COORD_PLACEHOLDER_SHAPES.items():
        if key not in coord_meta:
            coord_meta[key] = torch.empty(
                shape, dtype=torch.int32, device=ctx.env.device
            )
    aux_order = list(builder.aux_order)
    host_scalars = list(builder.host_scalars)
    score_graph = mask_graph = grad_graph = None
    try:
        if score_nodes is not None:
            assert isinstance(score_src, torch.fx.Node)
            score_graph = builder.build(
                score_nodes, score_src, aux_order, host_scalars, coord_meta
            )
        if mask_nodes is not None:
            assert isinstance(mask_pred, torch.fx.Node)
            mask_graph = builder.build(
                mask_nodes, mask_pred, aux_order, host_scalars, coord_meta
            )
        if grad_nodes is not None:
            assert isinstance(grad_src, torch.fx.Node)
            grad_graph = builder.build(
                grad_nodes, grad_src, aux_order, host_scalars, coord_meta
            )
    except _UnmappedRegionValue as error:
        ctx.decline(
            f"flex_bwd: region value {error} is not a score, coordinate or scalar"
        )
        return None
    aux_buffers: list[FlexAuxBuffer] = []
    for name in aux_order:
        value = _host_tensor_value(builder.aux_nodes[name])
        assert value is not None
        classes = {builder.classify_load(load) for load in builder.aux_loads[name]}
        index_class = classes.pop() if len(classes) == 1 else "general"
        aux_buffers.append(
            FlexAuxBuffer(
                name=name,
                dtype=value.dtype,
                shape=tuple(int(size) for size in value.shape),
                strides=tuple(int(stride) for stride in value.stride()),
                index_class=index_class,
            )
        )

    def region(
        graph_output: tuple[torch.fx.Graph, torch.fx.Node] | None,
    ) -> FlexRegion | None:
        if graph_output is None:
            return None
        return FlexRegion(graph_output[0], graph_output[1])

    return _Regions(
        region(score_graph),
        region(mask_graph),
        region(grad_graph),
        tuple(aux_buffers),
        tuple(host_scalars),
        None if score_graph is None else _region_key(score_graph[0]),
        None if grad_graph is None else _region_key(grad_graph[0]),
        frozenset(load for loads in builder.aux_loads.values() for load in loads),
    )


# -- assembly ----------------------------------------------------------------


def _assemble(
    ctx: _BwdCtx,
    root: GraphInfo,
    group_loop: torch.fx.Node,
    k_root: torch.fx.Node,
    v_root: torch.fx.Node,
    dv_full: torch.fx.Node,
    dk_full: torch.fx.Node,
    dk_factor: object,
    nests: list[tuple[_Outer, _Inner, _Body]],
) -> FlexBwdMatch | None:
    (full_outer, full_inner, full_body), (part_outer, part_inner, part_body) = nests
    if full_body.regions.mask is not None:
        ctx.decline("flex_bwd: the first (full) list body must not carry a mask")
        return None
    # The emitter scales dK by the softmax scale of the scores: the store's
    # factor must be that scale (literal or host scalar), nothing else.
    if not _same_scale(ctx, dk_factor, part_body.scale_literal, part_body.scale_symbol):
        ctx.decline("flex_bwd: the dK store factor is not the softmax scale")
        return None
    same = (
        full_body.q_name == part_body.q_name
        and full_body.do_name == part_body.do_name
        and full_body.lse_name == part_body.lse_name
        and full_body.delta_name == part_body.delta_name
        and full_body.dq_name == part_body.dq_name
        and full_body.k_root is part_body.k_root
        and full_body.v_root is part_body.v_root
        and full_body.io_dtype == part_body.io_dtype
        and full_body.scale_literal == part_body.scale_literal
        and full_body.scale_symbol == part_body.scale_symbol
        and full_body.log2_factor == part_body.log2_factor
        and full_body.lse_prescaled == part_body.lse_prescaled
        and full_body.acc_slots == part_body.acc_slots
    )
    if not same:
        ctx.decline("flex_bwd: the two bodies differ in operands, scale or exp base")
        return None
    if full_body.regions.score_key != part_body.regions.score_key:
        ctx.decline(
            "flex_bwd: the full body's score region differs from the partial one"
        )
        return None
    if full_body.regions.grad_key != part_body.regions.grad_key:
        ctx.decline(
            "flex_bwd: the full body's score_mod_bwd differs from the partial one"
        )
        return None
    full_regions, part_regions = full_body.regions, part_body.regions
    if (
        full_regions.host_scalars
        != part_regions.host_scalars[: len(full_regions.host_scalars)]
        or full_regions.aux != part_regions.aux[: len(full_regions.aux)]
    ):
        ctx.decline("flex_bwd: the two bodies disagree on captured tables or scalars")
        return None
    if full_inner.q_block != part_inner.q_block or full_inner.q_len != part_inner.q_len:
        ctx.decline("flex_bwd: the two walks disagree on QB or M")
        return None
    k_parts = _load_parts(full_body.k_root)
    v_parts = _load_parts(full_body.v_root)
    assert k_parts is not None and v_parts is not None
    k_name = _host_tensor_name(k_parts[0])
    v_name = _host_tensor_name(v_parts[0])
    k_value = _host_tensor_value(k_parts[0])
    v_value = _host_tensor_value(v_parts[0])
    assert k_name and v_name and k_value is not None and v_value is not None
    q_value = None
    for info in ctx.graphs.values():
        for node in info.graph.nodes:
            if _host_tensor_name(node) == full_body.q_name:
                q_value = _host_tensor_value(node)
                break
    assert q_value is not None
    batch, heads, q_seq, head_dim = (int(s) for s in q_value.shape)
    kv_batch, kv_heads, kv_seq, k_dim = (int(s) for s in k_value.shape)
    head_dim_v = int(v_value.shape[3])
    group = ctx.group
    if kv_batch != batch or k_dim != head_dim or kv_heads * group != heads:
        ctx.decline("flex_bwd: Q/K/V shapes disagree with the q-head group")
        return None
    if tuple(int(s) for s in v_value.shape) != (batch, kv_heads, kv_seq, head_dim_v):
        ctx.decline("flex_bwd: V shape mismatch")
        return None
    io_dtype = full_body.io_dtype
    if io_dtype not in (torch.float16, torch.bfloat16):
        ctx.decline("flex_bwd: Q/K/V must be fp16 or bf16")
        return None
    if k_value.dtype != io_dtype or v_value.dtype != io_dtype:
        ctx.decline("flex_bwd: K/V dtype mismatch")
        return None
    do_value = _host_value_named(ctx, full_body.do_name)
    if do_value is None or tuple(int(s) for s in do_value.shape) != (
        batch,
        heads,
        q_seq,
        head_dim_v,
    ):
        ctx.decline("flex_bwd: dO shape mismatch")
        return None
    operand_strides: list[tuple[int, ...]] = []
    for label, value in (
        ("Q", q_value),
        ("dO", do_value),
        ("K", k_value),
        ("V", v_value),
    ):
        strides = flex_operand_strides(value)
        if strides is None:
            ctx.decline(
                f"flex_bwd: {label} strides {tuple(value.stride())} are not a "
                "unit inner stride with non-zero 16-byte aligned outer strides"
            )
            return None
        operand_strides.append(strides)
    if not flex_bwd_head_dims_supported(head_dim, head_dim_v):
        ctx.decline(f"flex_bwd: unsupported head dims ({head_dim}, {head_dim_v})")
        return None
    if full_inner.q_len != q_seq:
        ctx.decline("flex_bwd: the Q loop clamps to a length other than M")
        return None
    q_block = full_inner.q_block
    if q_block not in FLEX_Q_BLOCKS:
        ctx.decline(f"flex_bwd: QB must be in {FLEX_Q_BLOCKS} (got {q_block})")
        return None
    kv_block = _col_index_block(ctx, part_outer.cnt_index[2])
    if kv_block is None:
        return None
    if kv_block == 0:
        kv_block = FLEX_KV_BLOCK
    if kv_block != FLEX_KV_BLOCK:
        ctx.decline(f"flex_bwd: KB must be {FLEX_KV_BLOCK} (got {kv_block})")
        return None
    tables = (
        (part_outer.cnt_name, part_outer.cnt_value, part_outer.cnt_index),
        (part_inner.idx_name, part_inner.idx_value, part_inner.idx_index),
        (full_outer.cnt_name, full_outer.cnt_value, full_outer.cnt_index),
        (full_inner.idx_name, full_inner.idx_value, full_inner.idx_index),
    )
    table_batch = int(part_outer.cnt_value.shape[0])
    table_heads = int(part_outer.cnt_value.shape[1])
    cols = int(part_outer.cnt_value.shape[2])
    if cols * kv_block < kv_seq:
        ctx.decline("flex_bwd: the tables have too few columns for N")
        return None
    for name, value, indices in tables:
        if (
            int(value.shape[0]) != table_batch
            or int(value.shape[1]) != table_heads
            or int(value.shape[2]) != cols
        ):
            ctx.decline(f"flex_bwd: table {name} geometry differs")
            return None
        if not _table_lead_index_ok(ctx, indices[0], ctx.b_id, table_batch, batch):
            ctx.decline(f"flex_bwd: table {name} batch index is not b % Bm")
            return None
        if not _table_lead_index_ok(ctx, indices[1], ctx.h_id, table_heads, heads):
            ctx.decline(f"flex_bwd: table {name} head index is not h % Hm")
            return None
        col_block = _col_index_block(ctx, indices[2])
        if col_block is None or (col_block != 0 and col_block != kv_block):
            ctx.decline(f"flex_bwd: table {name} column index is not n_begin // KB")
            return None
    if len({name for name, _, _ in tables}) != 4:
        ctx.decline("flex_bwd: the four tables must be distinct tensors")
        return None
    if not _accumulators_threaded(ctx, root, group_loop, dv_full, dk_full, nests):
        return None
    dk_name = dv_name = None
    for node in root.graph.nodes:
        from ...language import memory_ops

        if _is_call(node, memory_ops.store):
            value = node.args[2]
            assert isinstance(value, torch.fx.Node)
            inner: object = _strip_identity(value.args[0])
            name = _host_tensor_name(node.args[0])
            if _is_call(inner, _aten.mul.Tensor):
                dk_name = name
            else:
                dv_name = name
    assert dk_name is not None and dv_name is not None
    for label, name, shape in (
        ("dK", dk_name, (batch, kv_heads, kv_seq, head_dim)),
        ("dV", dv_name, (batch, kv_heads, kv_seq, head_dim_v)),
    ):
        value = _host_value_named(ctx, name)
        if value is None or tuple(int(s) for s in value.shape) != shape:
            ctx.decline(f"flex_bwd: {label} shape mismatch")
            return None
        strides = flex_operand_strides(value)
        if strides is None:
            ctx.decline(
                f"flex_bwd: {label} strides {tuple(value.stride())} are not a "
                "unit inner stride with non-zero 16-byte aligned outer strides"
            )
            return None
        operand_strides.append(strides)
    # Closed world of host tensors.
    aux_names = tuple(buffer.name for buffer in part_body.regions.aux)
    allowed = {
        full_body.q_name,
        k_name,
        v_name,
        full_body.do_name,
        full_body.lse_name,
        full_body.delta_name,
        full_body.dq_name,
        dk_name,
        dv_name,
        *aux_names,
        *(name for name, _, _ in tables),
    }
    present: set[str] = set()
    for info in ctx.graphs.values():
        for node in info.graph.nodes:
            name = _host_tensor_name(node)
            if name is not None:
                present.add(name)
    if present != allowed:
        ctx.decline(f"flex_bwd: unexpected host tensors {sorted(present - allowed)}")
        return None
    core = {
        full_body.q_name,
        k_name,
        v_name,
        full_body.do_name,
        full_body.lse_name,
        full_body.delta_name,
        full_body.dq_name,
        dk_name,
        dv_name,
    }
    if len(core) != 9:
        ctx.decline("flex_bwd: the nine core tensors must be distinct")
        return None
    regions = part_body.regions
    regions_q = part_body.regions_q
    plan = FlexBwdPlan(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        io_dtype=io_dtype,
        batch=batch,
        heads=heads,
        kv_heads=kv_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        walk=FlexBwdWalk(
            part_cnt_name=part_outer.cnt_name,
            part_idx_name=part_inner.idx_name,
            full_cnt_name=full_outer.cnt_name,
            full_idx_name=full_inner.idx_name,
            q_block=q_block,
            kv_block=kv_block,
            table_batch=table_batch,
            table_heads=table_heads,
            table_cols=cols,
        ),
        score_region=regions.score,
        mask_region=regions.mask,
        grad_region=regions.grad,
        aux_buffers=regions.aux,
        host_scalars=regions.host_scalars,
        scale_value=part_body.scale_literal,
        scale_arg=part_body.scale_symbol,
        log2_factor=part_body.log2_factor,
        lse_prescaled=part_body.lse_prescaled,
        strides=tuple(operand_strides),
        score_region_q=None if regions_q is None else regions_q.score,
        mask_region_q=None if regions_q is None else regions_q.mask,
        grad_region_q=None if regions_q is None else regions_q.grad,
        regions_q_captured=regions_q is not None,
    )
    # The storage must fit this device's shared memory in the emitter's most
    # frugal configuration (plain rings, 4 dQ boxes, the smallest step table),
    # else the kernel has no fused path and the generic one cannot take it.
    from .cute_flash_bwd_flex import flex_bwd_smem_fits

    tmem = flex_bwd_tmem_plan(head_dim, head_dim_v)
    if tmem is not None and tmem.mode == "pair":
        if not plan.regions_q_captured:
            ctx.decline("flex_bwd: the pair kernel needs the forward-frame regions")
            return None
        if plan.operand_strides is not None:
            ctx.decline("flex_bwd: the pair kernel takes contiguous operands only")
            return None
        # QB in FLEX_Q_BLOCKS and KB == FLEX_KV_BLOCK (checked above) are the
        # 128-row multiples the pair kernel's step tiles need.
        assert q_block % 128 == 0 and kv_block % 128 == 0, (q_block, kv_block)
    if not flex_bwd_smem_fits(plan, ctx.env.device):
        if tmem is not None and tmem.mode == "pair":
            from .cute_flash_bwd_flex import flex_bwd_pair_smem_decline

            ctx.decline(flex_bwd_pair_smem_decline(plan, ctx.env.device))
        else:
            ctx.decline(
                f"flex_bwd: head dims ({head_dim}, {head_dim_v}) do not fit shared memory"
            )
        return None
    assert ctx.kv_id is not None
    return FlexBwdMatch(
        plan=plan,
        b_block_id=ctx.b_id,
        hk_block_id=ctx.hk_id,
        kv_block_id=ctx.m_id,
        h_block_id=ctx.h_id,
        q_block_id=ctx.kv_id,
        full_nest=FlexBwdNest(
            full_outer.graph_id,
            full_inner.graph_id,
            full_outer.block_id,
            full_inner.block_id,
            False,
        ),
        part_nest=FlexBwdNest(
            part_outer.graph_id,
            part_inner.graph_id,
            part_outer.block_id,
            part_inner.block_id,
            regions.mask is not None,
        ),
        q_name=full_body.q_name,
        k_name=k_name,
        v_name=v_name,
        do_name=full_body.do_name,
        lse_name=full_body.lse_name,
        delta_name=full_body.delta_name,
        dq_name=full_body.dq_name,
        dk_name=dk_name,
        dv_name=dv_name,
    )


def _accumulators_threaded(
    ctx: _BwdCtx,
    root: GraphInfo,
    group_loop: torch.fx.Node,
    dv_full: torch.fx.Node,
    dk_full: torch.fx.Node,
    nests: list[tuple[_Outer, _Inner, _Body]],
) -> bool:
    """The dV / dK accumulators flow root -> q-head loop -> full walk -> Q
    body -> partial walk -> Q body: the full body's accumulator placeholders
    resolve to the root's zero tiles, the partial body's to ``_phi`` results
    of the full walk whose initial values resolve there."""
    from ...language._tracing_ops import _phi

    (_full_outer, full_inner, full_body), (_part_outer, part_inner, part_body) = nests
    for nest_inner, body, through_phi in (
        (full_inner, full_body, False),
        (part_inner, part_body, True),
    ):
        for slot, acc in zip(body.acc_slots, (dv_full, dk_full), strict=True):
            placeholder = ctx.placeholders(nest_inner.graph_id)[slot]
            graph_id, node = ctx.resolve(nest_inner.graph_id, placeholder)
            if through_phi:
                node = _strip_new_var(node)
                if not _is_call(node, _phi):
                    ctx.decline(
                        "flex_bwd: the partial walk does not continue the full walk's accumulator"
                    )
                    return False
                assert isinstance(node, torch.fx.Node)
                graph_id, node = ctx.resolve(graph_id, node.args[0])
            if node is not acc:
                ctx.decline(
                    "flex_bwd: accumulator is not threaded from the root zero tile"
                )
                return False
    return True


__all__ = [
    "FLEX_BWD_HEAD_DIM_STEP",
    "FlexBwdMatch",
    "FlexBwdPlan",
    "FlexBwdTmemPlan",
    "FlexBwdWalk",
    "flex_bwd_head_dims_supported",
    "flex_bwd_tmem_plan",
    "match_flex_attention_bwd",
]
