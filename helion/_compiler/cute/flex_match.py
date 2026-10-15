"""Matcher for the FlexAttention envelope of the fused CuTe flash path.

``examples/flex_attention.py::helion_flex_attention_kernel`` is the target
form: a 3-axis root grid ``hl.tile([B, H, M], block_size=[1, 1, block_m])``,
a Q tile loaded once per work item, one or two BlockMask-driven KV walks
(an outer counted loop over a 0-d count load with an inner data-dependent
``hl.tile(start_n, end_n)`` loop per listed block), and an online-softmax
step whose score transform is the user's ``score_mod`` / ``mask_mod``
captured as pointwise FX regions.

The matcher canonicalizes instead of classifying: coordinates (``b``, ``h``,
``q_idx``, ``kv_idx``), loop-carried roles (by dataflow through a two-level
cross-graph placeholder resolver), the leading ``qk * scale`` multiply, the
trailing ``* log2(e)`` multiply, the final ``where(mask, score, -inf)`` and
the ``m_safe`` / ``l_safe`` guards are all recognized structurally; whatever
is left between the QK dot and the softmax entry becomes the plan's
``score_region`` / ``mask_region`` (``FlexRegion``), admitted only when it is
pure pointwise over the canonical coordinates, lifted host tensors
(``hl.load`` inside the stage) and host scalars.

``match_flex_envelope`` is the single entry for both the autotune surface
(``detect_flash_search_surface`` on ``device_ir.graphs``) and codegen
(``_detect_attention_mma_loop`` on ``codegen_graphs``), so the two never
disagree on what a flex kernel is. ``FLEX_PLAN_ENABLED`` gates the
integration points until the list-walk emitter lands; the matcher itself is
always callable.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch
import torch.fx
from torch.utils._sympy.functions import FloorDiv

from .attention_plan import AttentionScorePlan
from .attention_plan import FlexAuxBuffer
from .attention_plan import FlexKvWalk
from .attention_plan import FlexRegion
from .attention_plan import flash_head_dims_supported

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from torch.fx.node import Argument

    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..variable_origin import Origin

# Flipped by the emitter workstream once the fa4 list-walk body exists. The
# integration points (surface detector, codegen hook) consult it; tests patch
# it to exercise the plumbing before the emitter lands.
FLEX_PLAN_ENABLED = True
# v1 restrictions of the walk: 128-wide KV tiles (one BlockMask KV block per
# kernel tile) and Q blocks of one or two 128-row tiles.
FLEX_KV_BLOCK = 128
FLEX_Q_BLOCKS = (128, 256)
LOG2E = 1.4426950408889634
# Coordinates in region order: the placeholder names and the plan contract.
FLEX_COORDS = ("b", "h", "q", "kv")
_COORD_PLACEHOLDER_NAMES = {
    "score": "score",
    "b": "b_idx",
    "h": "h_idx",
    "q": "q_idx",
    "kv": "kv_idx",
}
# Shapes of the synthesized coordinate metadata (the kernel's broadcast
# vectors are ``[1, 1]`` for b/h, ``[tile_m, 1]`` for q, ``[1, tile_n]`` for kv).
_COORD_PLACEHOLDER_SHAPES = {
    "b": (1, 1),
    "h": (1, 1),
    "q": (FLEX_KV_BLOCK, 1),
    "kv": (1, FLEX_KV_BLOCK),
}

_aten = torch.ops.aten
_prims = torch.ops.prims

# Pointwise ``aten`` ops a region may contain (per element, IEEE order).
_REGION_ATEN_TARGETS: frozenset[object] = frozenset(
    {
        _aten.add.Tensor,
        _aten.add.Scalar,
        _aten.sub.Tensor,
        _aten.sub.Scalar,
        _aten.mul.Tensor,
        _aten.mul.Scalar,
        _aten.div.Tensor,
        _aten.div.Scalar,
        _aten.div.Tensor_mode,
        _aten.neg.default,
        _aten.abs.default,
        _aten.minimum.default,
        _aten.maximum.default,
        _aten.clamp.default,
        _aten.clamp.Tensor,
        _aten.clamp_min.default,
        _aten.clamp_max.default,
        _aten.floor_divide.default,
        _aten.floor_divide.Scalar,
        _aten.remainder.Tensor,
        _aten.remainder.Scalar,
        _aten.fmod.Tensor,
        _aten.fmod.Scalar,
        _aten.eq.Tensor,
        _aten.eq.Scalar,
        _aten.ne.Tensor,
        _aten.ne.Scalar,
        _aten.lt.Tensor,
        _aten.lt.Scalar,
        _aten.le.Tensor,
        _aten.le.Scalar,
        _aten.gt.Tensor,
        _aten.gt.Scalar,
        _aten.ge.Tensor,
        _aten.ge.Scalar,
        _aten.bitwise_and.Tensor,
        _aten.bitwise_and.Scalar,
        _aten.bitwise_or.Tensor,
        _aten.bitwise_or.Scalar,
        _aten.bitwise_xor.Tensor,
        _aten.bitwise_xor.Scalar,
        _aten.bitwise_not.default,
        _aten.logical_and.default,
        _aten.logical_or.default,
        _aten.logical_xor.default,
        _aten.logical_not.default,
        _aten.where.self,
        _aten.where.ScalarOther,
        _aten.where.ScalarSelf,
        _aten.where.Scalar,
        _aten.scalar_tensor.default,
        _aten.full.default,
        _aten.sigmoid.default,
        _aten.log.default,
        _aten.log1p.default,
        _aten.log2.default,
        _aten.tanh.default,
        _aten.exp.default,
        _aten.exp2.default,
        _aten.sqrt.default,
        _aten.rsqrt.default,
        _aten.pow.Tensor_Scalar,
        _aten.square.default,
        _aten.sign.default,
        _aten.floor.default,
        _aten.ceil.default,
        _aten.trunc.default,
        _aten.round.default,
        _aten.reciprocal.default,
        _prims.convert_element_type.default,
    }
)
# Shape-only ops that are the identity per element.
_IDENTITY_VIEW_TARGETS: frozenset[object] = frozenset(
    {
        _aten.unsqueeze.default,
        _aten.expand.default,
        _aten.view.default,
        _aten.reshape.default,
        _aten.alias.default,
        _aten.clone.default,
    }
)
# Ops whose ``shape`` argument is metadata (replaced by ``[]`` in the region).
_FULL_TARGETS: frozenset[object] = frozenset({_aten.full.default})


class _ScalarSym(NamedTuple):
    """A host scalar index resolved to its tile origin.

    ``role`` is ``"begin"`` (``tile.begin``), ``"id"`` (``tile.id``) or
    ``"block"`` (the block-size symnode used as an index: the tile itself,
    which for a unit block is its single index). ``div`` / ``mod`` record a
    folded ``// c`` / ``% c`` on the symbol.
    """

    block_id: int
    role: str
    div: int
    mod: int | None


@dataclasses.dataclass(frozen=True)
class FlexNest:
    """One BlockMask list walk: the outer counted loop and its inner KV loop."""

    outer_graph_id: int
    inner_graph_id: int
    outer_block_id: int
    inner_block_id: int
    cnt_name: str
    idx_name: str
    masked: bool


@dataclasses.dataclass(frozen=True)
class FlexMatch:
    """Everything the surface and the emitter need from a flex kernel."""

    plan: AttentionScorePlan
    b_block_id: int
    h_block_id: int
    q_block_id: int
    kv_block_id: int
    # The first nest in program order walks the full list, the second the
    # partial (masked) list.
    full_nest: FlexNest
    part_nest: FlexNest
    q_name: str
    k_name: str
    v_name: str
    o_name: str
    lse_name: str | None
    lse_log_base: str | None
    batch: int
    heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    head_dim: int
    head_dim_v: int
    io_dtype: torch.dtype
    aux_names: tuple[str, ...]
    # ``None`` when the scale is a runtime symbol (``plan.scale_arg``).
    scale_value: float | None
    exp_base: str
    trailing_scale: float

    @property
    def loop_block_ids(self) -> tuple[int, ...]:
        ids: list[int] = []
        for nest in (self.full_nest, self.part_nest):
            for block_id in (nest.outer_block_id, nest.inner_block_id):
                if block_id not in ids:
                    ids.append(block_id)
        return tuple(ids)

    @property
    def graph_ids(self) -> tuple[int, int, int, int]:
        return (
            self.full_nest.outer_graph_id,
            self.full_nest.inner_graph_id,
            self.part_nest.outer_graph_id,
            self.part_nest.inner_graph_id,
        )

    @property
    def num_bh(self) -> int:
        return self.batch * self.heads

    @property
    def num_kv(self) -> int:
        return (self.kv_seq + FLEX_KV_BLOCK - 1) // FLEX_KV_BLOCK

    @property
    def table_names(self) -> tuple[str, str, str, str]:
        walk = self.plan.kv_walk
        assert walk is not None
        return (
            walk.part_cnt_name,
            walk.part_idx_name,
            walk.full_cnt_name,
            walk.full_idx_name,
        )


class _Leaf(NamedTuple):
    """A region leaf: ``kind`` in score/coord/aux/host_scalar/const."""

    kind: str
    key: str


class _Dep(NamedTuple):
    """Dependence facts of an index expression: which canonical coordinates
    (and loaded values) it reads, and its affine ``(q, kv)`` coefficients
    when it is affine in them (``None`` otherwise)."""

    deps: frozenset[str]
    affine: tuple[float, float] | None


_CONST_DEP = _Dep(frozenset(), (0.0, 0.0))


def _is_call(node: object, *targets: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
    )


def _strip_new_var(node: object) -> object:
    from ...language._tracing_ops import _new_var

    while _is_call(node, _new_var):
        assert isinstance(node, torch.fx.Node)
        node = node.args[0]
    return node


def _is_full_slice(value: object) -> bool:
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


def _is_identity_view(node: object) -> bool:
    from ...language import view_ops

    if _is_call(node, *_IDENTITY_VIEW_TARGETS):
        return True
    if not _is_call(node, view_ops.subscript):
        return False
    assert isinstance(node, torch.fx.Node)
    indices = node.args[1] if len(node.args) > 1 else None
    return isinstance(indices, (list, tuple)) and all(
        index is None or _is_full_slice(index) for index in indices
    )


def _strip_identity(node: object) -> object:
    """Strip ``_new_var`` wrappers and per-element identity views."""
    while True:
        node = _strip_new_var(node)
        if _is_identity_view(node):
            assert isinstance(node, torch.fx.Node)
            node = node.args[0]
            continue
        return node


def _host_tensor_name(node: object) -> str | None:
    from ...language._tracing_ops import _host_tensor

    if _is_call(node, _host_tensor):
        assert isinstance(node, torch.fx.Node)
        name = node.args[0] if node.args else None
        return name if isinstance(name, str) else None
    return None


def _host_tensor_value(node: object) -> torch.Tensor | None:
    if _host_tensor_name(node) is None:
        return None
    assert isinstance(node, torch.fx.Node)
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _load_parts(node: object) -> tuple[torch.fx.Node, list[object]] | None:
    """``(host_tensor_node, indices)`` of a plain ``hl.load`` (no mask/policy)."""
    from ...language import memory_ops

    if not _is_call(node, memory_ops.load):
        return None
    assert isinstance(node, torch.fx.Node)
    if len(node.args) < 2:
        return None
    if len(node.args) > 2 and node.args[2] is not None:
        return None
    if len(node.args) > 3 and node.args[3] is not None:
        return None
    if node.kwargs:
        return None
    tensor, indices = node.args[0], node.args[1]
    if _host_tensor_name(tensor) is None or not isinstance(indices, (list, tuple)):
        return None
    assert isinstance(tensor, torch.fx.Node)
    return tensor, list(indices)


def _node_value(node: object) -> object:
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else node


def _sympy_of(node: object) -> sympy.Expr | None:
    """The sympy expression behind a symnode-valued FX node (or a python int)."""
    if isinstance(node, bool):
        return None
    if isinstance(node, int):
        return sympy.Integer(node)
    value = _node_value(node)
    if isinstance(value, (torch.SymInt, torch.SymFloat)):
        expr = value._sympy_()
        return expr if isinstance(expr, sympy.Expr) else None
    return None


def _int_literal(node: object) -> int | None:
    """A python int or a constant SymInt (``Mod(b, 1)`` folds to one)."""
    if isinstance(node, bool):
        return None
    if isinstance(node, int):
        return node
    expr = _sympy_of(node)
    if isinstance(expr, sympy.Integer):
        return int(expr)
    return None


def _scalar_value(node: object) -> float | None:
    """A python number or a 0-d constant tensor node (``scalar_tensor`` /
    ``full([], c)``)."""
    from ...language import creation_ops

    if isinstance(node, bool):
        return float(node)
    if isinstance(node, (int, float)):
        return float(node)
    if _is_call(node, _aten.scalar_tensor.default):
        assert isinstance(node, torch.fx.Node)
        value = node.args[0] if node.args else None
        return float(value) if isinstance(value, (int, float)) else None
    if _is_call(node, _aten.full.default, creation_ops.full):
        assert isinstance(node, torch.fx.Node)
        if len(node.args) < 2 or node.args[0] not in ([], ()):
            return None
        value = node.args[1]
        return float(value) if isinstance(value, (int, float)) else None
    return None


def _is_neg_inf(node: object) -> bool:
    value = _scalar_value(node)
    return value is not None and value == float("-inf")


def _const_true(node: object) -> bool:
    if node is True:
        return True
    if _is_call(node, _aten.scalar_tensor.default, _aten.full.default):
        assert isinstance(node, torch.fx.Node)
        value = node.args[1] if node.target is _aten.full.default else node.args[0]
        shape_ok = node.target is not _aten.full.default or node.args[0] in ([], ())
        return shape_ok and value is True
    from ...language import creation_ops

    if _is_call(node, creation_ops.full):
        assert isinstance(node, torch.fx.Node)
        return node.args[0] in ([], ()) and node.args[1] is True
    return False


def _graph_outputs(graph: torch.fx.Graph) -> list[object] | None:
    for node in graph.nodes:
        if node.op == "output":
            if len(node.args) != 1 or not isinstance(node.args[0], (list, tuple)):
                return None
            return list(node.args[0])
    return None


def _phi_result(node: object, loop: torch.fx.Node, index: int) -> object:
    """``lhs`` of ``_phi(lhs, getitem(loop, index))``; ``None`` otherwise."""
    from ...language._tracing_ops import _phi

    if not _is_call(node, _phi):
        return None
    assert isinstance(node, torch.fx.Node)
    if len(node.args) < 2:
        return None
    getitem = node.args[1]
    if not _is_call(getitem, operator.getitem):
        return None
    assert isinstance(getitem, torch.fx.Node)
    if getitem.args[0] is not loop or getitem.args[1] != index:
        return None
    return node.args[0]


def flex_host_scalar_symbol(
    host_function: HostFunction, name: str
) -> sympy.Symbol | None:
    """The registered host symbol named ``name`` (``plan.host_scalars`` /
    ``plan.scale_arg`` entries), with its original assumptions."""
    for expr in host_function.expr_to_origin:
        if isinstance(expr, sympy.Symbol) and expr.name == name:
            return expr
    return None


def flex_host_scalar_device_name(
    df: DeviceFunction, host_function: HostFunction, name: str
) -> str | None:
    """The kernel-argument expression of the host scalar ``name`` at codegen
    (``"scale"`` for the scale argument, ``"score_mod_closure_0"`` for a
    closure int); registers the argument on ``df`` like any scalar read."""
    symbol = flex_host_scalar_symbol(host_function, name)
    return None if symbol is None else df.sympy_expr(symbol)


class _UnmappedRegionValue(Exception):
    """A region node that is neither an admitted op nor a leaf (score,
    coordinate, host scalar) was used as a value while building a region."""


class _Ctx:
    """Per-match state: the graph set, the parent map and the decline log."""

    def __init__(
        self,
        graphs: Sequence[GraphInfo],
        root_grid_ids: Sequence[int],
        env: CompileEnvironment,
        host: HostFunction,
        reasons: list[str] | None,
    ) -> None:
        from ...language._tracing_ops import _for_loop

        self.env = env
        self.host = host
        self.reasons = reasons
        self.graphs = {info.graph_id: info for info in graphs}
        self.b_id, self.h_id, self.m_id = (int(bid) for bid in root_grid_ids)
        self.kv_id: int | None = None
        self.parents: dict[int, tuple[int, torch.fx.Node]] = {}
        self._placeholders: dict[int, list[torch.fx.Node]] = {}
        for info in graphs:
            self._placeholders[info.graph_id] = [
                node for node in info.graph.nodes if node.op == "placeholder"
            ]
            for node in info.graph.nodes:
                if _is_call(node, _for_loop) and isinstance(node.args[0], int):
                    self.parents[node.args[0]] = (info.graph_id, node)

    def decline(self, reason: str) -> None:
        if self.reasons is not None:
            self.reasons.append(reason)

    def placeholders(self, graph_id: int) -> list[torch.fx.Node]:
        return self._placeholders[graph_id]

    def coord_of_block(self, block_id: int) -> str | None:
        if block_id == self.b_id:
            return "b"
        if block_id == self.h_id:
            return "h"
        if block_id == self.m_id:
            return "q"
        if self.kv_id is not None and block_id == self.kv_id:
            return "kv"
        return None

    # -- symbol resolution -------------------------------------------------

    def origin_of(self, expr: sympy.Expr) -> Origin | None:
        info = self.host.expr_to_origin.get(expr)
        return None if info is None else info.origin

    def scalar_sym(self, node: object) -> _ScalarSym | None:
        """Resolve an index node (symnode, ``tile_begin``, ``tile_id``) to
        its tile origin, folding one ``// c`` and one ``% c``."""
        from ..variable_origin import BlockSizeOrigin
        from ..variable_origin import TileBeginOrigin
        from ..variable_origin import TileIdOrigin

        expr = _sympy_of(node)
        if expr is None:
            return None
        div, mod = 1, None
        if type(expr).__name__ in ("Mod", "PythonMod") and len(expr.args) == 2:
            base, modulus = expr.args
            if not isinstance(modulus, sympy.Integer):
                return None
            mod = int(modulus)
            expr = base
        if isinstance(expr, FloorDiv) and len(expr.args) == 2:
            base, divisor = expr.args
            if not isinstance(divisor, sympy.Integer):
                return None
            div = int(divisor)
            expr = base
        if not isinstance(expr, sympy.Symbol):
            return None
        origin = self.origin_of(expr)
        if isinstance(origin, TileBeginOrigin):
            role = "begin"
        elif isinstance(origin, TileIdOrigin):
            role = "id"
        elif isinstance(origin, BlockSizeOrigin):
            role = "block"
        else:
            return None
        return _ScalarSym(origin.block_id, role, div, mod)

    def unit_axis_index(self, node: object, block_id: int) -> bool:
        """``node`` is the scalar index of unit-tile axis ``block_id`` (its
        block symnode, ``tile.begin`` or ``tile.id``)."""
        sym = self.scalar_sym(node)
        return (
            sym is not None
            and sym.block_id == block_id
            and sym.div == 1
            and sym.mod is None
        )

    def host_scalar_name(self, node: object) -> str | None:
        """The symbol name when ``node`` is a plain host scalar symnode
        (kernel argument, closure cell, module global, tensor size)."""
        expr = _sympy_of(node)
        if not isinstance(expr, sympy.Symbol):
            return None
        origin = self.origin_of(expr)
        if origin is None or not origin.is_host():
            return None
        return expr.name

    # -- cross-graph resolution -------------------------------------------

    def resolve(self, graph_id: int, node: object) -> tuple[int, object]:
        """Follow a loop placeholder to the value the enclosing ``_for_loop``
        passes in, recursively (inner body -> outer body -> root)."""
        while True:
            node = _strip_new_var(node)
            if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
                return graph_id, node
            parent = self.parents.get(graph_id)
            if parent is None:
                return graph_id, node
            parent_graph_id, loop = parent
            index = self.placeholders(graph_id).index(node)
            inputs = loop.args[3] if len(loop.args) > 3 else None
            if not isinstance(inputs, (list, tuple)) or index >= len(inputs):
                return graph_id, node
            node = inputs[index]
            graph_id = parent_graph_id

    def placeholder_index(self, graph_id: int, node: object) -> int | None:
        node = _strip_new_var(node)
        if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
            return None
        return self.placeholders(graph_id).index(node)

    def block_symnode_id(self, node: object) -> int | None:
        from ...language._tracing_ops import _get_symnode

        if not _is_call(node, _get_symnode):
            return None
        assert isinstance(node, torch.fx.Node)
        name = node.args[0] if node.args else None
        if not isinstance(name, str) or not name.startswith("block_size_"):
            return None
        return int(name[len("block_size_") :])

    def coord_vector(self, graph_id: int, node: object) -> str | None:
        """The canonical coordinate a per-element index vector stands for:
        ``tile.index`` or ``tile.begin + hl.arange(tile.block_size)`` of one
        of the four axes (``tile.begin + hl.arange(1)`` for the one-element
        batch / head tiles), under any number of broadcast views, possibly
        threaded through loop placeholders."""
        from ...language import tile_ops

        for _ in range(8):
            graph_id, node = self.resolve(graph_id, node)
            stripped = _strip_identity(node)
            if stripped is not node:
                node = stripped
                continue
            break
        if _is_call(node, tile_ops.tile_index):
            assert isinstance(node, torch.fx.Node)
            block_id = self.block_symnode_id(node.args[0] if node.args else None)
            return None if block_id is None else self.coord_of_block(block_id)
        if not _is_call(node, _aten.add.Tensor):
            return None
        assert isinstance(node, torch.fx.Node)
        if len(node.args) != 2:
            return None
        lhs, rhs = node.args
        for iota, begin in ((lhs, rhs), (rhs, lhs)):
            if not _is_call(iota, _prims.iota.default):
                continue
            assert isinstance(iota, torch.fx.Node)
            start = iota.kwargs.get("start", iota.args[1] if len(iota.args) > 1 else 0)
            step = iota.kwargs.get("step", iota.args[2] if len(iota.args) > 2 else 1)
            if start != 0 or step != 1:
                return None
            begin_sym = self.scalar_sym(begin)
            if (
                begin_sym is None
                or begin_sym.role != "begin"
                or begin_sym.div != 1
                or begin_sym.mod is not None
            ):
                return None
            length = iota.args[0] if iota.args else None
            block_id = self.block_symnode_id(length)
            if block_id is None:
                # ``tile.begin + hl.arange(1)``: the one-element batch / head
                # tiles keep a literal extent so the vector is a concrete
                # ``[1, 1]`` for the broadcast against the scores (a fixed
                # ``block_size=1`` symbol is not narrowed to one).
                if length != 1 or self.coord_of_block(begin_sym.block_id) not in (
                    "b",
                    "h",
                ):
                    return None
                block_id = begin_sym.block_id
            elif begin_sym.block_id != block_id:
                return None
            return self.coord_of_block(block_id)
        return None


class _OuterLoop(NamedTuple):
    loop: torch.fx.Node
    graph_id: int
    block_id: int
    cnt_name: str
    cnt_value: torch.Tensor
    cnt_index: list[object]
    q_block: int | None


class _InnerLoop(NamedTuple):
    loop: torch.fx.Node
    graph_id: int
    block_id: int
    idx_name: str
    idx_value: torch.Tensor
    idx_index: list[object]
    kv_block: int
    kv_len: int


class _Regions(NamedTuple):
    score: torch.fx.Graph | None
    score_output: torch.fx.Node | None
    mask: torch.fx.Graph | None
    mask_output: torch.fx.Node | None
    aux: tuple[FlexAuxBuffer, ...]
    host_scalars: tuple[str, ...]
    score_key: tuple[object, ...] | None
    # The body's original ``hl.load`` nodes the regions captured.
    load_nodes: frozenset[torch.fx.Node]


class _Body(NamedTuple):
    """Facts of one inner KV body."""

    q_root: torch.fx.Node
    k_name: str
    v_name: str
    k_value: torch.Tensor
    v_value: torch.Tensor
    kv_group: int
    scale_literal: float | None
    scale_symbol: str | None
    scale_peeled: bool
    trailing_scale: float
    exp_base: str
    regions: _Regions
    role_placeholders: tuple[int, int, int]


def match_flex_envelope(
    graphs: Iterable[GraphInfo],
    root_grid_ids: Sequence[int],
    *,
    env: CompileEnvironment,
    host_function: HostFunction | None = None,
    reasons: list[str] | None = None,
) -> FlexMatch | None:
    """Match the FlexAttention envelope over a kernel's whole graph set.

    ``graphs`` are ``device_ir.graphs`` (surface time) or the codegen copies
    (``fn.codegen.codegen_graphs``); ``root_grid_ids`` is the kernel's single
    root grid. ``reasons`` collects the first failing condition for
    diagnostics. The result's node identities belong to ``graphs``; callers
    must not mix the two graph sets.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    if host_function is None:
        host_function = HostFunction.current()
    graphs = tuple(graphs)
    if len(root_grid_ids) != 3:
        if reasons is not None:
            reasons.append("flex: the root grid must have three axes (b, h, m)")
        return None
    ctx = _Ctx(graphs, root_grid_ids, env, host_function, reasons)
    roots = [info for info in graphs if isinstance(info, RootGraphInfo)]
    if len(roots) != 1:
        ctx.decline("flex: exactly one root graph")
        return None
    root = roots[0]
    for block_id in (ctx.b_id, ctx.h_id):
        if not _unit_fixed_block(env, block_id):
            ctx.decline(f"flex: root axis {block_id} is not pinned to block size 1")
            return None
    for info in graphs:
        if isinstance(info, ForLoopGraphInfo) and len(info.block_ids) != 1:
            ctx.decline("flex: multi-dimensional device loop")
            return None
        if not isinstance(info, (ForLoopGraphInfo, RootGraphInfo)):
            ctx.decline(f"flex: unsupported graph {type(info).__name__}")
            return None

    outer_loops = _root_outer_loops(ctx, root)
    if outer_loops is None:
        return None
    nests: list[tuple[_OuterLoop, _InnerLoop, _Body]] = []
    for outer in outer_loops:
        inner = _match_outer_graph(ctx, outer)
        if inner is None:
            return None
        body = _match_inner_body(ctx, inner)
        if body is None:
            return None
        nests.append((outer, inner, body))
    return _assemble(ctx, root, nests)


def _unit_fixed_block(env: CompileEnvironment, block_id: int) -> bool:
    from ..compile_environment import FixedBlockSizeSource

    source = env.block_sizes[block_id].block_size_source
    return (
        isinstance(source, FixedBlockSizeSource)
        and isinstance(source.value, int)
        and source.value == 1
    )


def _root_outer_loops(ctx: _Ctx, root: GraphInfo) -> list[_OuterLoop] | None:
    """The root graph's list walks: ``_for_loop(g, [0], [count], inputs)``
    with ``count = load(cnt[b', h', m_begin // QB])`` (0-d int32)."""
    from ...language._tracing_ops import _for_loop
    from ..device_ir import ForLoopGraphInfo

    loops: list[_OuterLoop] = []
    for node in root.graph.nodes:
        if not _is_call(node, _for_loop):
            if _is_call(node, *_loop_targets()):
                ctx.decline("flex: unsupported control flow in the root graph")
                return None
            continue
        graph_id, begins, ends = node.args[0], node.args[1], node.args[2]
        if not isinstance(graph_id, int) or begins != [0]:
            ctx.decline("flex: the outer list loop must start at 0")
            return None
        if not isinstance(ends, (list, tuple)) or len(ends) != 1:
            ctx.decline("flex: the outer list loop needs one bound")
            return None
        parts = _load_parts(ends[0])
        if parts is None:
            ctx.decline("flex: the outer list bound is not a table load")
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
            ctx.decline("flex: the count table must be a 3-D int32 tensor")
            return None
        info = ctx.graphs.get(graph_id)
        if not isinstance(info, ForLoopGraphInfo):
            ctx.decline("flex: outer loop graph missing")
            return None
        q_block = _row_index_block(ctx, indices[2], value)
        if q_block is None:
            ctx.decline("flex: table row index is not m_begin // QB")
            return None
        loops.append(
            _OuterLoop(
                node,
                graph_id,
                info.block_ids[0],
                name,
                value,
                indices,
                q_block if q_block > 0 else None,
            )
        )
    if len(loops) != 2:
        ctx.decline(
            f"flex: expected two list walks (full, partial), found {len(loops)}"
        )
        return None
    return loops


def _row_index_block(ctx: _Ctx, node: object, table: torch.Tensor) -> int | None:
    """``QB`` of a table row index ``m_begin // QB`` (0 for a literal 0 row
    on a one-row table, which stands for any ``QB >= M``)."""
    literal = _int_literal(node)
    if literal is not None:
        return 0 if literal == 0 and int(table.shape[2]) == 1 else None
    sym = ctx.scalar_sym(node)
    if (
        sym is None
        or sym.block_id != ctx.m_id
        or sym.role != "begin"
        or sym.mod is not None
    ):
        return None
    return sym.div


def _table_lead_index_ok(
    ctx: _Ctx, node: object, block_id: int, table_dim: int, full_dim: int
) -> bool:
    """``node`` indexes a table's leading (batch / head) dim of size
    ``table_dim`` from the axis ``block_id`` whose extent is ``full_dim``:
    the axis scalar itself (``table_dim == full_dim``), ``axis % table_dim``,
    or the literal 0 of a broadcast (size-1) dim."""
    literal = _int_literal(node)
    if literal is not None:
        return literal == 0 and table_dim == 1
    sym = ctx.scalar_sym(node)
    if sym is None or sym.block_id != block_id or sym.div != 1:
        return False
    if sym.mod is None:
        return table_dim == full_dim
    return sym.mod == table_dim and full_dim % table_dim == 0


def _loop_targets() -> tuple[object, ...]:
    from ...language import _tracing_ops

    return (
        _tracing_ops._for_loop,
        _tracing_ops._for_loop_step,
        _tracing_ops._while_loop,
        _tracing_ops._if,
    )


def _match_outer_graph(ctx: _Ctx, outer: _OuterLoop) -> _InnerLoop | None:
    """The outer list body: one inner ``_for_loop([start], [end])`` with
    ``start = load(idx[b', h', row, tile_id]) * KB`` and
    ``end = minimum(start + KB, full([], N))``; nothing else."""
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
            ctx.decline(f"flex: list body has {node.target}")
            return None
        if node.target is _for_loop:
            inner_loops.append(node)
        elif node.target is memory_ops.load:
            loads.append(node)
    if len(inner_loops) != 1 or len(loads) != 1:
        ctx.decline("flex: the list body must hold one KV loop and one index load")
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
        ctx.decline("flex: the KV loop must have one begin and one end")
        return None
    start, end = begins[0], ends[0]
    # start = load(idx[...]) * KB
    if not isinstance(start, torch.fx.Node) or not _is_call(start, _aten.mul.Tensor):
        ctx.decline("flex: KV loop begin is not idx * KB")
        return None
    if len(start.args) != 2:
        ctx.decline("flex: KV loop begin is not idx * KB")
        return None
    load_arg, kb_arg = start.args
    if _is_call(kb_arg, memory_ops.load):
        load_arg, kb_arg = kb_arg, load_arg
    if load_arg is not loads[0] or not isinstance(kb_arg, int) or kb_arg <= 0:
        ctx.decline("flex: KB must be a specialized positive int")
        return None
    kv_block = kb_arg
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
        ctx.decline("flex: the index table must be a 4-D int32 tensor")
        return None
    if not ctx.unit_axis_index(indices[3], outer.block_id):
        ctx.decline("flex: the index table's last index is not the list position")
        return None
    # end = minimum(start + KB, full([], N))
    if not isinstance(end, torch.fx.Node) or not _is_call(end, _aten.minimum.default):
        ctx.decline("flex: KV loop end is not minimum(start + KB, N)")
        return None
    if len(end.args) != 2:
        ctx.decline("flex: KV loop end is not minimum(start + KB, N)")
        return None
    sum_node, n_node = end.args
    if _scalar_value(sum_node) is not None:
        sum_node, n_node = n_node, sum_node
    kv_len_value = _scalar_value(n_node)
    if (
        not _is_call(sum_node, _aten.add.Tensor)
        or kv_len_value is None
        or kv_len_value != int(kv_len_value)
    ):
        ctx.decline("flex: KV loop end is not minimum(start + KB, N)")
        return None
    assert isinstance(sum_node, torch.fx.Node)
    add_args = list(sum_node.args)
    if start not in add_args:
        ctx.decline("flex: KV loop end does not add KB to the begin")
        return None
    add_args.remove(start)
    if add_args != [kv_block]:
        ctx.decline("flex: KV loop end adds a different block size")
        return None
    info = ctx.graphs.get(graph_id)
    if not isinstance(info, ForLoopGraphInfo):
        ctx.decline("flex: KV loop graph missing")
        return None
    block_id = info.block_ids[0]
    if ctx.kv_id is None:
        ctx.kv_id = block_id
    elif ctx.kv_id != block_id:
        ctx.decline("flex: the two KV loops use different block sizes")
        return None
    return _InnerLoop(
        loop,
        graph_id,
        block_id,
        name,
        value,
        indices,
        kv_block,
        int(kv_len_value),
    )


def _match_inner_body(ctx: _Ctx, inner: _InnerLoop) -> _Body | None:
    from ...language import creation_ops
    from ...language import matmul_ops
    from ...language import memory_ops
    from ...language._tracing_ops import _mask_to

    graph = ctx.graphs[inner.graph_id].graph
    graph_id = inner.graph_id
    dots: list[torch.fx.Node] = []
    loads: list[torch.fx.Node] = []
    mask_tos: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is matmul_ops.dot:
            dots.append(node)
        elif node.target is memory_ops.load:
            loads.append(node)
        elif node.target is _mask_to:
            mask_tos.append(node)
        elif node.target is memory_ops.store or node.target in _loop_targets():
            ctx.decline(f"flex: KV body has {node.target}")
            return None
    if len(dots) != 2:
        ctx.decline("flex: the KV body needs exactly two hl.dot calls")
        return None
    qk_dot = pv_dot = None
    for dot in dots:
        if _is_call(dot.args[1], _aten.permute.default, _aten.transpose.int):
            qk_dot = dot
        else:
            pv_dot = dot
    if qk_dot is None or pv_dot is None:
        ctx.decline("flex: cannot tell the QK dot from the PV dot")
        return None
    if len(qk_dot.args) < 3 or len(pv_dot.args) < 3:
        ctx.decline("flex: hl.dot without an accumulator")
        return None
    for dot in (qk_dot, pv_dot):
        out_dtype = dot.args[3] if len(dot.args) > 3 else None
        if out_dtype not in (None, torch.float32):
            ctx.decline("flex: hl.dot out_dtype must be fp32")
            return None

    # Q: a loop placeholder that resolves to the root Q load.
    q_graph_id, q_root = ctx.resolve(graph_id, qk_dot.args[0])
    q_parts = _load_parts(q_root)
    if q_graph_id in ctx.parents or q_parts is None:
        ctx.decline("flex: the QK operand is not the root Q load")
        return None
    assert isinstance(q_root, torch.fx.Node)
    q_tensor, q_indices = q_parts
    q_value = _host_tensor_value(q_tensor)
    if (
        q_value is None
        or q_value.ndim != 4
        or len(q_indices) != 4
        or not ctx.unit_axis_index(q_indices[0], ctx.b_id)
        or not ctx.unit_axis_index(q_indices[1], ctx.h_id)
        or ctx.block_symnode_id(q_indices[2]) != ctx.m_id
        or not _is_full_slice(q_indices[3])
    ):
        ctx.decline("flex: the Q load is not query[b, h, tile_m, :]")
        return None
    # QK accumulator: a fresh zero tile.
    qk_acc = qk_dot.args[2]
    if not (
        _is_call(qk_acc, creation_ops.full, _aten.full.default)
        and isinstance(qk_acc, torch.fx.Node)
        and len(qk_acc.args) > 1
        and qk_acc.args[1] == 0.0
    ):
        ctx.decline("flex: the QK accumulator is not a zero tile")
        return None
    # K^T and V loads.
    k_perm = qk_dot.args[1]
    assert isinstance(k_perm, torch.fx.Node)
    if k_perm.target is _aten.permute.default:
        if k_perm.args[1] != [1, 0]:
            ctx.decline("flex: K must be transposed with permute([1, 0])")
            return None
    elif tuple(k_perm.args[1:3]) not in ((0, 1), (1, 0)):
        ctx.decline("flex: K must be transposed with transpose(0, 1)")
        return None
    k_load = k_perm.args[0]
    v_load = pv_dot.args[1]
    kv_facts = []
    for load in (k_load, v_load):
        facts = _kv_load_facts(ctx, load)
        if facts is None:
            ctx.decline("flex: K/V load is not key[b, h // g, tile_n, :]")
            return None
        kv_facts.append(facts)
    (k_name, k_value, k_group), (v_name, v_value, v_group) = kv_facts
    if k_group != v_group:
        ctx.decline("flex: K and V disagree on the GQA divisor")
        return None
    assert isinstance(k_load, torch.fx.Node)
    assert isinstance(v_load, torch.fx.Node)

    # Scale, trailing multiply, mask split and the softmax entry.
    qk_users = list(qk_dot.users)
    if len(qk_users) != 1:
        ctx.decline("flex: the QK score must have one consumer")
        return None
    scale_literal: float | None = 1.0
    scale_symbol: str | None = None
    score_root: torch.fx.Node = qk_dot
    first = qk_users[0]
    if _is_call(first, _aten.mul.Tensor) and len(first.args) == 2:
        other = first.args[1] if first.args[0] is qk_dot else first.args[0]
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            scale_literal = float(other)
            score_root = first
        else:
            name = ctx.host_scalar_name(other)
            if name is not None:
                scale_literal = None
                scale_symbol = name
                score_root = first
    entries = []
    for mask_to in mask_tos:
        if len(mask_to.args) < 2 or not _is_neg_inf(mask_to.args[1]):
            continue
        source = mask_to.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        users = list(source.users)
        subs = [user for user in users if _is_call(user, _aten.sub.Tensor)]
        feeds_running_max = any(
            _is_call(amax, _aten.amax.default)
            and any(_is_call(user, _aten.maximum.default) for user in amax.users)
            for amax in mask_to.users
        )
        if (
            len(users) == 2
            and len(subs) == 1
            and mask_to in users
            and feeds_running_max
        ):
            entries.append((source, mask_to, subs[0]))
    if len(entries) != 1:
        ctx.decline("flex: cannot find the online-softmax entry")
        return None
    entry, entry_mask_to, centered = entries[0]
    trailing_scale = 1.0
    pre_entry: object = _strip_new_var(entry)
    if _is_call(pre_entry, _aten.mul.Tensor):
        assert isinstance(pre_entry, torch.fx.Node)
        lhs, rhs = pre_entry.args[:2]
        factor = rhs if isinstance(lhs, torch.fx.Node) else lhs
        tensor = lhs if isinstance(lhs, torch.fx.Node) else rhs
        if isinstance(factor, (int, float)) and not isinstance(factor, bool):
            trailing_scale = float(factor)
            pre_entry = _strip_new_var(tensor)
    mask_pred: object = None
    score_src: object = pre_entry
    if _is_call(pre_entry, _aten.where.self, _aten.where.ScalarOther):
        assert isinstance(pre_entry, torch.fx.Node)
        if len(pre_entry.args) == 3 and _is_neg_inf(pre_entry.args[2]):
            mask_pred, score_src = pre_entry.args[0], pre_entry.args[1]
            if _const_true(mask_pred):
                mask_pred = None
    score_src = _strip_new_var(score_src)
    scale_peeled = score_src is score_root
    regions = _capture_regions(
        ctx,
        inner,
        qk_dot,
        score_root if scale_peeled else qk_dot,
        None if scale_peeled else score_src,
        mask_pred,
    )
    if regions is None:
        return None

    # The online-softmax chain (rank-2 form with the -inf-safe max).
    chain = _match_softmax_chain(
        ctx, graph_id, entry, entry_mask_to, centered, pv_dot, v_load
    )
    if chain is None:
        return None
    exp_base, roles, p_node = chain
    pv_p = pv_dot.args[0]
    pv_dtype = k_value.dtype
    if _is_call(pv_p, _prims.convert_element_type.default):
        assert isinstance(pv_p, torch.fx.Node)
        if pv_p.args[1] != pv_dtype:
            ctx.decline("flex: P must be converted to the K/V dtype")
            return None
        pv_p = pv_p.args[0]
    if pv_p is not p_node:
        ctx.decline("flex: the PV operand is not the softmax numerator")
        return None

    # Closed world of the body's loads: K, V and the region loads.
    if any(
        load not in (k_load, v_load) and load not in regions.load_nodes
        for load in loads
    ):
        ctx.decline("flex: the KV body loads a tensor outside the regions")
        return None
    return _Body(
        q_root,
        k_name,
        v_name,
        k_value,
        v_value,
        k_group,
        scale_literal,
        scale_symbol,
        scale_peeled,
        trailing_scale,
        exp_base,
        regions,
        roles,
    )


def _kv_load_facts(ctx: _Ctx, load: object) -> tuple[str, torch.Tensor, int] | None:
    parts = _load_parts(load)
    if parts is None:
        return None
    tensor, indices = parts
    value = _host_tensor_value(tensor)
    name = _host_tensor_name(tensor)
    if value is None or name is None or value.ndim != 4 or len(indices) != 4:
        return None
    if not ctx.unit_axis_index(indices[0], ctx.b_id):
        return None
    head = ctx.scalar_sym(indices[1])
    if head is None or head.block_id != ctx.h_id or head.mod is not None:
        return None
    if ctx.kv_id is None or ctx.block_symnode_id(indices[2]) != ctx.kv_id:
        return None
    if not _is_full_slice(indices[3]):
        return None
    return name, value, head.div


def _match_softmax_chain(
    ctx: _Ctx,
    graph_id: int,
    entry: torch.fx.Node,
    entry_mask_to: torch.fx.Node,
    centered: torch.fx.Node,
    pv_dot: torch.fx.Node,
    v_load: torch.fx.Node,
) -> tuple[str, tuple[int, int, int], torch.fx.Node] | None:
    """Verify the rank-2 online softmax from the entry node and return the
    exp base, the ``(m, l, acc)`` placeholder indices and the P node."""
    from ...language._tracing_ops import _mask_to

    graph = entry.graph
    amax_users = [u for u in entry_mask_to.users if _is_call(u, _aten.amax.default)]
    if len(amax_users) != 1 or len(entry_mask_to.users) != 1:
        ctx.decline("flex: the masked score must feed one amax")
        return None
    amax = amax_users[0]
    if (
        len(amax.args) < 2
        or amax.args[1] != [-1]
        or (len(amax.args) > 2 and amax.args[2])
    ):
        ctx.decline("flex: amax must reduce the last dim without keepdim")
        return None
    max_users = [u for u in amax.users if _is_call(u, _aten.maximum.default)]
    if len(max_users) != 1 or len(amax.users) != 1:
        ctx.decline("flex: amax must feed one maximum")
        return None
    m_ij = max_users[0]
    prev_m = m_ij.args[1] if m_ij.args[0] is amax else m_ij.args[0]
    m_index = ctx.placeholder_index(graph_id, prev_m)
    if m_index is None:
        ctx.decline("flex: the running max is not loop-carried")
        return None
    # m_safe = where(m_ij == -inf, 0, m_ij) (optional).
    m_safe: torch.fx.Node = m_ij
    for user in m_ij.users:
        if not _is_call(user, _aten.where.self, _aten.where.ScalarSelf):
            continue
        if len(user.args) != 3 or user.args[2] is not m_ij:
            continue
        cond = user.args[0]
        if not (
            _is_call(cond, _aten.eq.Scalar)
            and isinstance(cond, torch.fx.Node)
            and cond.args[0] is m_ij
            and _is_neg_inf(cond.args[1])
        ):
            continue
        if _scalar_value(user.args[1]) == 0.0:
            m_safe = user
            break
    # centered = entry - m_safe[:, None]
    if centered.args[0] is not entry or _strip_identity(centered.args[1]) is not m_safe:
        ctx.decline("flex: the centered score does not subtract the safe max")
        return None
    p_users = [
        u for u in centered.users if _is_call(u, _aten.exp.default, _aten.exp2.default)
    ]
    if len(p_users) != 1 or len(centered.users) != 1:
        ctx.decline("flex: the centered score must feed one exp/exp2")
        return None
    p_node = p_users[0]
    exp_base = "exp2" if p_node.target is _aten.exp2.default else "exp"
    l_ij = None
    for user in p_node.users:
        if not (
            _is_call(user, _mask_to) and user.args[0] is p_node and user.args[1] == 0
        ):
            continue
        for sum_user in user.users:
            if _is_call(sum_user, _aten.sum.dim_IntList) and sum_user.args[1] == [-1]:
                l_ij = sum_user
    if l_ij is None:
        ctx.decline("flex: no row sum of P")
        return None
    # alpha = exp*(prev_m - m_safe)
    alpha = None
    for user in m_safe.users:
        if not (
            _is_call(user, _aten.sub.Tensor)
            and len(user.args) == 2
            and user.args[1] is m_safe
            and ctx.placeholder_index(graph_id, user.args[0]) == m_index
        ):
            continue
        exp_users = [u for u in user.users if _is_call(u, p_node.target)]
        if len(exp_users) == 1 and len(user.users) == 1:
            alpha = exp_users[0]
    if alpha is None:
        ctx.decline("flex: no alpha = exp(prev_m - m_safe)")
        return None
    # l = prev_l * alpha + l_ij
    l_update = None
    l_index = None
    for user in alpha.users:
        if not (_is_call(user, _aten.mul.Tensor) and len(user.args) == 2):
            continue
        other = user.args[1] if user.args[0] is alpha else user.args[0]
        index = ctx.placeholder_index(graph_id, other)
        if index is None:
            continue
        for add in user.users:
            if (
                _is_call(add, _aten.add.Tensor)
                and l_ij in add.args
                and user in add.args
            ):
                l_update = add
                l_index = index
    if l_update is None or l_index is None:
        ctx.decline("flex: no l = prev_l * alpha + l_ij")
        return None
    # acc = prev_acc * alpha[:, None]; pv = dot(p, v, acc)
    acc_rescaled = pv_dot.args[2]
    if not (
        _is_call(acc_rescaled, _aten.mul.Tensor)
        and isinstance(acc_rescaled, torch.fx.Node)
    ):
        ctx.decline("flex: the PV accumulator is not the rescaled acc")
        return None
    acc_index = None
    for lhs, rhs in (
        (acc_rescaled.args[0], acc_rescaled.args[1]),
        (acc_rescaled.args[1], acc_rescaled.args[0]),
    ):
        if _strip_identity(rhs) is alpha:
            acc_index = ctx.placeholder_index(graph_id, lhs)
    if acc_index is None:
        ctx.decline("flex: the acc rescale does not multiply the carried acc by alpha")
        return None
    if pv_dot.args[1] is not v_load:
        ctx.decline("flex: the PV dot does not read the V load")
        return None
    outputs = _graph_outputs(graph)
    if outputs is None or len(outputs) != 3:
        ctx.decline("flex: the KV body must return (m, l, acc)")
        return None
    if (
        _strip_new_var(outputs[0]) is not m_ij
        or _strip_new_var(outputs[1]) is not l_update
        or _strip_new_var(outputs[2]) is not pv_dot
    ):
        ctx.decline("flex: the KV body outputs are not (m_ij, l, acc) in order")
        return None
    if len({m_index, l_index, acc_index}) != 3:
        ctx.decline("flex: the (m, l, acc) roles collide")
        return None
    return exp_base, (m_index, l_index, acc_index), p_node


# -- region capture -----------------------------------------------------------


class _RegionBuilder:
    """Collects the pointwise nodes between the leaves and a region output,
    classifies the leaves and copies the nodes into a fresh FX graph with
    the canonical placeholder order."""

    def __init__(self, ctx: _Ctx, inner: _InnerLoop, score_root: torch.fx.Node) -> None:
        self.ctx = ctx
        self.inner = inner
        self.graph_id = inner.graph_id
        self.score_root = score_root
        self.leaves: dict[torch.fx.Node, _Leaf] = {}
        self.aux_order: list[str] = []
        self.aux_nodes: dict[str, torch.fx.Node] = {}
        self.aux_loads: dict[str, list[torch.fx.Node]] = {}
        self.host_scalars: list[str] = []
        self.host_scalar_nodes: dict[str, torch.fx.Node] = {}
        self.deps: dict[object, _Dep] = {}
        self.failed: str | None = None

    def fail(self, reason: str) -> None:
        if self.failed is None:
            self.failed = reason

    # -- classification ----------------------------------------------------

    def leaf_of(self, node: torch.fx.Node) -> _Leaf | None:
        if node in self.leaves:
            return self.leaves[node]
        leaf = self._classify_leaf(node)
        if leaf is not None:
            self.leaves[node] = leaf
        return leaf

    def _classify_leaf(self, node: torch.fx.Node) -> _Leaf | None:
        ctx = self.ctx
        if node is self.score_root:
            return _Leaf("score", "score")
        stripped = _strip_new_var(node)
        if isinstance(stripped, torch.fx.Node) and stripped is self.score_root:
            return _Leaf("score", "score")
        coord = ctx.coord_vector(self.graph_id, node)
        if coord is not None:
            return _Leaf("coord", coord)
        if isinstance(stripped, torch.fx.Node) and stripped.op == "placeholder":
            return None
        name = ctx.host_scalar_name(node)
        if name is not None:
            if name not in self.host_scalars:
                self.host_scalars.append(name)
                self.host_scalar_nodes[name] = node
            return _Leaf("host_scalar", name)
        return None

    def collect(self, output: object) -> list[torch.fx.Node] | None:
        """Region nodes (original graph order) feeding ``output``."""
        if not isinstance(output, torch.fx.Node):
            self.fail("flex: region output is a constant")
            return None
        seen: set[torch.fx.Node] = set()
        order: list[torch.fx.Node] = []
        stack = [output]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if self.leaf_of(node) is not None:
                continue
            if not self._admit(node):
                return None
            order.append(node)
            for arg in self._region_inputs(node):
                if isinstance(arg, torch.fx.Node):
                    stack.append(arg)
        if self.failed is not None:
            return None
        positions = {n: i for i, n in enumerate(output.graph.nodes)}
        order.sort(key=lambda n: positions[n])
        return order

    def _region_inputs(self, node: torch.fx.Node) -> list[object]:
        """The node's inputs that belong to the region (shape metadata and
        the host tensor of a load are not region values)."""
        from ...language import creation_ops
        from ...language import memory_ops

        if node.target in _FULL_TARGETS or node.target is creation_ops.full:
            return []
        if node.target is memory_ops.load:
            parts = _load_parts(node)
            assert parts is not None
            return list(parts[1])
        inputs: list[object] = []
        torch.fx.node.map_arg((node.args, node.kwargs), inputs.append)
        return inputs

    def _admit(self, node: torch.fx.Node) -> bool:
        from ...language import creation_ops
        from ...language import memory_ops
        from ...language._tracing_ops import _new_var

        if node.op == "placeholder":
            self.fail(
                "flex: region reads a loop-carried value that is not a coordinate"
            )
            return False
        if node.op != "call_function":
            self.fail(f"flex: region has a {node.op} node")
            return False
        target = node.target
        if target is _new_var or _is_identity_view(node):
            return True
        if target in _REGION_ATEN_TARGETS or target is creation_ops.full:
            if target is creation_ops.full and len(node.args) < 2:
                self.fail("flex: malformed hl.full in a region")
                return False
            if node.kwargs and target not in (
                _aten.scalar_tensor.default,
                _aten.full.default,
                _aten.div.Tensor_mode,
                _aten.clamp.default,
                _aten.clamp.Tensor,
            ):
                self.fail(f"flex: region op {target} with keyword arguments")
                return False
            return True
        if target is memory_ops.load:
            parts = _load_parts(node)
            if parts is None:
                self.fail("flex: region load with a mask or eviction policy")
                return False
            tensor, indices = parts
            value = _host_tensor_value(tensor)
            name = _host_tensor_name(tensor)
            if value is None or name is None:
                self.fail("flex: region load of a non-host tensor")
                return False
            if len(indices) != value.ndim or any(
                not isinstance(index, (torch.fx.Node, int)) or isinstance(index, bool)
                for index in indices
            ):
                self.fail("flex: region load must index every dim with an int tensor")
                return False
            if name not in self.aux_order:
                self.aux_order.append(name)
                self.aux_nodes[name] = tensor
            self.aux_loads.setdefault(name, []).append(node)
            return True
        self.fail(f"flex: region has non-pointwise op {target}")
        return False

    # -- dependence analysis ----------------------------------------------

    def dep(self, node: object) -> _Dep:
        if not isinstance(node, torch.fx.Node):
            return _CONST_DEP
        if node in self.deps:
            return self.deps[node]
        result = self._dep(node)
        self.deps[node] = result
        return result

    def _dep(self, node: torch.fx.Node) -> _Dep:
        from ...language import creation_ops
        from ...language import memory_ops

        leaf = self.leaf_of(node)
        if leaf is not None:
            if leaf.kind == "score":
                return _Dep(frozenset({"score"}), None)
            if leaf.kind == "coord":
                affine = (
                    (1.0, 0.0)
                    if leaf.key == "q"
                    else (0.0, 1.0)
                    if leaf.key == "kv"
                    else (0.0, 0.0)
                )
                return _Dep(frozenset({leaf.key}), affine)
            return _Dep(frozenset({"host_scalar"}), (0.0, 0.0))
        target = node.target
        if _is_identity_view(node) or _strip_new_var(node) is not node:
            return self.dep(node.args[0])
        if target is memory_ops.load:
            parts = _load_parts(node)
            assert parts is not None
            load_deps: set[str] = {"value"}
            for index in parts[1]:
                load_deps |= self.dep(index).deps
            return _Dep(frozenset(load_deps), None)
        if (
            target in _FULL_TARGETS
            or target is creation_ops.full
            or target is _aten.scalar_tensor.default
        ):
            return _CONST_DEP
        inputs = [self.dep(arg) for arg in self._region_inputs(node)]
        deps = frozenset().union(*(d.deps for d in inputs))
        affine: tuple[float, float] | None = None
        if target in (
            _aten.add.Tensor,
            _aten.sub.Tensor,
            _aten.add.Scalar,
            _aten.sub.Scalar,
        ):
            lhs, rhs = (self.dep(arg) for arg in node.args[:2])
            if lhs.affine is not None and rhs.affine is not None:
                sign = -1.0 if target in (_aten.sub.Tensor, _aten.sub.Scalar) else 1.0
                affine = (
                    lhs.affine[0] + sign * rhs.affine[0],
                    lhs.affine[1] + sign * rhs.affine[1],
                )
        elif target is _aten.neg.default and inputs and inputs[0].affine is not None:
            affine = (-inputs[0].affine[0], -inputs[0].affine[1])
        elif target in (_aten.mul.Tensor, _aten.mul.Scalar):
            lhs_arg, rhs_arg = node.args[:2]
            for factor_arg, other_arg in ((lhs_arg, rhs_arg), (rhs_arg, lhs_arg)):
                if isinstance(factor_arg, (int, float)) and not isinstance(
                    factor_arg, bool
                ):
                    other = self.dep(other_arg)
                    if other.affine is not None:
                        affine = (
                            other.affine[0] * factor_arg,
                            other.affine[1] * factor_arg,
                        )
                    break
        elif target is _prims.convert_element_type.default:
            affine = inputs[0].affine if inputs else None
        return _Dep(deps, affine)

    def classify_load(self, load: torch.fx.Node) -> str:
        parts = _load_parts(load)
        assert parts is not None
        indices = parts[1]
        index_deps = [self.dep(index) for index in indices]
        all_deps = frozenset().union(*(d.deps for d in index_deps))
        if "value" in all_deps or "score" in all_deps:
            return "value"
        last = index_deps[-1]
        if "kv" not in all_deps:
            return "row_uniform" if "q" in all_deps else "head"
        if "kv" not in last.deps or last.affine is None:
            return "general"
        if last.affine[1] == 1.0:
            leading_q = any("q" in d.deps for d in index_deps[:-1])
            return "row_col" if leading_q else "column_affine"
        if last.affine[1] == -1.0:
            return "diagonal"
        return "general"

    # -- graph construction -------------------------------------------------

    def build(
        self,
        nodes: list[torch.fx.Node],
        output: torch.fx.Node,
        aux_order: list[str],
        host_scalars: list[str],
        coord_meta: dict[str, object],
    ) -> tuple[torch.fx.Graph, torch.fx.Node]:
        from ...language import creation_ops
        from ...language import memory_ops

        graph = torch.fx.Graph()
        placeholders: dict[str, torch.fx.Node] = {}
        for key in ("score", "b", "h", "q", "kv"):
            placeholder = graph.placeholder(_COORD_PLACEHOLDER_NAMES[key])
            placeholder.meta["val"] = coord_meta[key]
            placeholders[key] = placeholder
        for index, name in enumerate(aux_order):
            placeholder = graph.placeholder(f"aux_{index}")
            placeholder.meta["val"] = self.aux_nodes[name].meta.get("val")
            placeholder.meta["flex_source_name"] = name
            placeholders[f"aux:{name}"] = placeholder
        for index, name in enumerate(host_scalars):
            placeholder = graph.placeholder(f"host_scalar_{index}")
            placeholder.meta["val"] = self.host_scalar_nodes[name].meta.get("val")
            placeholder.meta["flex_source_name"] = name
            placeholders[f"hs:{name}"] = placeholder
        mapping: dict[torch.fx.Node, torch.fx.Node] = {}

        def transform(node: torch.fx.Node) -> torch.fx.Node:
            if node in mapping:
                return mapping[node]
            leaf = self.leaf_of(node)
            if leaf is None:
                # Not a region value (a shape query the admission let through
                # as metadata, say) used as one: decline, never render it.
                raise _UnmappedRegionValue(node.name)
            if leaf.kind == "score":
                result = placeholders["score"]
            elif leaf.kind == "coord":
                result = placeholders[leaf.key]
            else:
                result = placeholders[f"hs:{leaf.key}"]
            mapping[node] = result
            return result

        for node in nodes:
            if _strip_new_var(node) is not node or _is_identity_view(node):
                source = node.args[0]
                assert isinstance(source, torch.fx.Node)
                mapping[node] = transform(source)
                continue
            if node.target is _aten.sym_size.int:
                # Shape metadata (``zeros_like(score)`` sizes its ``full`` by
                # ``sym_size_int(score, d)``): never a region value.
                continue
            if node.target in _FULL_TARGETS or node.target is creation_ops.full:
                # The shape argument is metadata (``zeros_like(score)`` traces
                # to ``full([sym_size(q, 0), block_size_kv], 0.0)``): drop it
                # instead of mapping its symbolic size nodes into the region.
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
                new_indices: list[Argument] = []
                for index in indices:
                    if isinstance(index, torch.fx.Node):
                        new_indices.append(transform(index))
                    else:
                        new_indices.append(index)  # pyrefly: ignore[bad-argument-type]
                copied = graph.call_function(
                    memory_ops.load,
                    (placeholders[f"aux:{name}"], new_indices, None, None),
                )
                copied.meta = dict(node.meta)
                copied.meta["flex_source_name"] = name
                copied.meta["flex_index_class"] = self.classify_load(node)
                mapping[node] = copied
                continue
            mapping[node] = graph.node_copy(node, transform)
        result = mapping[output] if output in mapping else transform(output)
        graph.output(result)
        graph.lint()
        return graph, result


def _region_key(graph: torch.fx.Graph) -> tuple[object, ...]:
    """A structural signature of a region graph: node names are ignored and
    placeholders are referenced by their canonical names, so two regions
    that differ only in the placeholders they leave unused compare equal."""
    body = [node for node in graph.nodes if node.op != "placeholder"]
    index = {node: i for i, node in enumerate(body)}

    def key_of(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            if arg.op == "placeholder":
                return ("placeholder", str(arg.target))
            return ("node", index[arg])
        if isinstance(arg, (list, tuple)):
            return (type(arg).__name__, tuple(key_of(a) for a in arg))
        if isinstance(arg, slice):
            return ("slice", arg.start, arg.stop, arg.step)
        return (type(arg).__name__, repr(arg))

    return tuple(
        (
            node.op,
            str(node.target),
            key_of(node.args),
            tuple(sorted((k, key_of(v)) for k, v in node.kwargs.items())),
        )
        for node in body
    )


def _capture_regions(
    ctx: _Ctx,
    inner: _InnerLoop,
    qk_dot: torch.fx.Node,
    score_root: torch.fx.Node,
    score_src: object,
    mask_pred: object,
) -> _Regions | None:
    """Capture the score region (``score_src`` back to ``score_root``) and
    the mask region (``mask_pred`` back to the leaves) of one KV body."""
    builder = _RegionBuilder(ctx, inner, score_root)
    score_nodes: list[torch.fx.Node] | None = None
    mask_nodes: list[torch.fx.Node] | None = None
    if score_src is not None:
        score_nodes = builder.collect(score_src)
        if score_nodes is None:
            ctx.decline(builder.failed or "flex: score region rejected")
            return None
        if not any(leaf.kind == "score" for leaf in builder.leaves.values()):
            ctx.decline("flex: the score region does not read the QK score")
            return None
    if mask_pred is not None:
        mask_nodes = builder.collect(mask_pred)
        if mask_nodes is None:
            ctx.decline(builder.failed or "flex: mask region rejected")
            return None
    # Any leaf that is the QK dot itself while the scale was peeled means a
    # mask that reads the score; the peel would change its units.
    if score_src is None and any(
        leaf.kind == "score" for leaf in builder.leaves.values()
    ):
        ctx.decline("flex: the mask predicate reads the score")
        return None
    coord_meta: dict[str, object] = {"score": qk_dot.meta.get("val")}
    for node, leaf in builder.leaves.items():
        if leaf.kind == "coord" and leaf.key not in coord_meta:
            coord_meta[leaf.key] = node.meta.get("val")
    for key, shape in _COORD_PLACEHOLDER_SHAPES.items():
        if key not in coord_meta:
            # A coordinate no region reads still gets a typed placeholder
            # (int32, tile-shaped like the kernel's broadcast vectors).
            coord_meta[key] = torch.empty(
                shape, dtype=torch.int32, device=ctx.env.device
            )
    aux_order = list(builder.aux_order)
    host_scalars = list(builder.host_scalars)
    score_graph = score_output = mask_graph = mask_output = None
    try:
        if score_nodes is not None:
            assert isinstance(score_src, torch.fx.Node)
            score_graph, score_output = builder.build(
                score_nodes, score_src, aux_order, host_scalars, coord_meta
            )
        if mask_nodes is not None:
            assert isinstance(mask_pred, torch.fx.Node)
            mask_graph, mask_output = builder.build(
                mask_nodes, mask_pred, aux_order, host_scalars, coord_meta
            )
    except _UnmappedRegionValue as error:
        ctx.decline(f"flex: region value {error} is not a score, coordinate or scalar")
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
    return _Regions(
        score_graph,
        score_output,
        mask_graph,
        mask_output,
        tuple(aux_buffers),
        tuple(host_scalars),
        None if score_graph is None else _region_key(score_graph),
        frozenset(load for loads in builder.aux_loads.values() for load in loads),
    )


# -- assembly -----------------------------------------------------------------


def _assemble(
    ctx: _Ctx,
    root: GraphInfo,
    nests: list[tuple[_OuterLoop, _InnerLoop, _Body]],
) -> FlexMatch | None:
    (full_outer, full_inner, full_body), (part_outer, part_inner, part_body) = nests
    if full_body.regions.mask is not None:
        ctx.decline("flex: the first (full) list body must not carry a mask")
        return None
    if full_body.q_root is not part_body.q_root:
        ctx.decline("flex: the two bodies read different Q tiles")
        return None
    same_step = (
        full_body.k_name == part_body.k_name
        and full_body.v_name == part_body.v_name
        and full_body.kv_group == part_body.kv_group
        and full_body.scale_literal == part_body.scale_literal
        and full_body.scale_symbol == part_body.scale_symbol
        and full_body.scale_peeled == part_body.scale_peeled
        and full_body.trailing_scale == part_body.trailing_scale
        and full_body.exp_base == part_body.exp_base
    )
    if not same_step:
        ctx.decline("flex: the two bodies differ in operands, scale or exp base")
        return None
    if full_body.regions.score_key != part_body.regions.score_key:
        ctx.decline("flex: the full body's score region differs from the partial one")
        return None
    full_regions, part_regions = full_body.regions, part_body.regions
    if (
        full_regions.host_scalars
        != part_regions.host_scalars[: len(full_regions.host_scalars)]
        or full_regions.aux != part_regions.aux[: len(full_regions.aux)]
    ):
        ctx.decline("flex: the two bodies disagree on captured tables or scalars")
        return None
    if (
        full_inner.kv_block != part_inner.kv_block
        or full_inner.kv_len != part_inner.kv_len
    ):
        ctx.decline("flex: the two walks disagree on KB or N")
        return None
    q_block = (
        part_outer.q_block if part_outer.q_block is not None else full_outer.q_block
    )
    if (
        full_outer.q_block is not None
        and part_outer.q_block is not None
        and full_outer.q_block != part_outer.q_block
    ):
        ctx.decline("flex: the two walks disagree on QB")
        return None

    # Host tensors and geometry.
    q_parts = _load_parts(full_body.q_root)
    assert q_parts is not None
    q_tensor = q_parts[0]
    q_name = _host_tensor_name(q_tensor)
    q_value = _host_tensor_value(q_tensor)
    assert q_name is not None and q_value is not None
    batch, heads, q_seq, head_dim = (int(size) for size in q_value.shape)
    kv_group = full_body.kv_group
    k_value, v_value = full_body.k_value, full_body.v_value
    kv_seq = int(k_value.shape[2])
    head_dim_v = int(v_value.shape[3])
    io_dtype = q_value.dtype
    if kv_group <= 0 or heads % kv_group != 0:
        ctx.decline("flex: the GQA divisor does not divide the head count")
        return None
    kv_heads = heads // kv_group
    if io_dtype not in (torch.float16, torch.bfloat16):
        ctx.decline("flex: Q/K/V must be fp16 or bf16")
        return None
    for value, expected in (
        (k_value, (batch, kv_heads, kv_seq, head_dim)),
        (v_value, (batch, kv_heads, kv_seq, head_dim_v)),
    ):
        if (
            tuple(int(size) for size in value.shape) != expected
            or value.dtype != io_dtype
        ):
            ctx.decline("flex: K/V shape or dtype mismatch")
            return None
    for value in (q_value, k_value, v_value):
        if not value.is_contiguous():
            ctx.decline("flex: Q/K/V must be contiguous")
            return None
    if not flash_head_dims_supported(head_dim, head_dim_v):
        ctx.decline(f"flex: unsupported head dims ({head_dim}, {head_dim_v})")
        return None
    if batch <= 0 or heads <= 0 or q_seq <= 0 or kv_seq <= 0:
        ctx.decline("flex: empty tensors")
        return None
    if full_inner.kv_len != kv_seq:
        ctx.decline("flex: the KV loop clamps to a length other than N")
        return None
    kv_block = full_inner.kv_block
    trivial = kv_block >= kv_seq
    if q_block is None:
        if not trivial:
            ctx.decline("flex: cannot determine the Q block of the tables")
            return None
        q_block = q_seq
    if not trivial:
        if kv_block != FLEX_KV_BLOCK:
            ctx.decline(f"flex: KB must be {FLEX_KV_BLOCK} (got {kv_block})")
            return None
        if q_block not in FLEX_Q_BLOCKS:
            ctx.decline(f"flex: QB must be in {FLEX_Q_BLOCKS} (got {q_block})")
            return None
    # Tables: counts [Bm, Hm, R], indices [Bm, Hm, R, L] with one geometry.
    tables = (
        (part_outer.cnt_name, part_outer.cnt_value, part_outer.cnt_index),
        (part_inner.idx_name, part_inner.idx_value, part_inner.idx_index),
        (full_outer.cnt_name, full_outer.cnt_value, full_outer.cnt_index),
        (full_inner.idx_name, full_inner.idx_value, full_inner.idx_index),
    )
    table_batch = int(part_outer.cnt_value.shape[0])
    table_heads = int(part_outer.cnt_value.shape[1])
    rows = int(part_outer.cnt_value.shape[2])
    if rows * q_block < q_seq:
        ctx.decline("flex: the tables have too few rows for M")
        return None
    for name, value, indices in tables:
        if (
            int(value.shape[0]) != table_batch
            or int(value.shape[1]) != table_heads
            or int(value.shape[2]) != rows
        ):
            ctx.decline(f"flex: table {name} geometry differs")
            return None
        if not _table_lead_index_ok(ctx, indices[0], ctx.b_id, table_batch, batch):
            ctx.decline(f"flex: table {name} batch index is not b % Bm")
            return None
        if not _table_lead_index_ok(ctx, indices[1], ctx.h_id, table_heads, heads):
            ctx.decline(f"flex: table {name} head index is not h % Hm")
            return None
        row_block = _row_index_block(ctx, indices[2], value)
        if row_block is None or (row_block != 0 and row_block != q_block):
            ctx.decline(f"flex: table {name} row index is not m_begin // QB")
            return None
    if len({name for name, _, _ in tables}) != 4:
        ctx.decline("flex: the four tables must be distinct tensors")
        return None

    # Loop-carried state through both nests and the epilogue.
    loops = _phi_chain(ctx, root, nests)
    if loops is None:
        return None
    final_m, final_l, final_acc = loops
    epilogue = _match_epilogue(
        ctx,
        root,
        final_m,
        final_l,
        final_acc,
        (batch, heads, q_seq),
        head_dim_v,
        io_dtype,
    )
    if epilogue is None:
        return None
    o_name, lse_name, lse_log_base = epilogue

    # Closed world of host tensors.
    aux_names = tuple(buffer.name for buffer in part_body.regions.aux)
    allowed = {q_name, full_body.k_name, full_body.v_name, o_name, *aux_names}
    allowed.update(name for name, _, _ in tables)
    if lse_name is not None:
        allowed.add(lse_name)
    present: set[str] = set()
    for info in ctx.graphs.values():
        for node in info.graph.nodes:
            name = _host_tensor_name(node)
            if name is not None:
                present.add(name)
    if present != allowed:
        ctx.decline(f"flex: unexpected host tensors {sorted(present - allowed)}")
        return None
    if len({q_name, full_body.k_name, full_body.v_name, o_name}) != 4:
        ctx.decline("flex: Q/K/V/O must be distinct tensors")
        return None
    if not _root_graph_admitted(ctx, root):
        return None

    # The plan.
    base = LOG2E if part_body.exp_base == "exp" else 1.0
    multiplier = part_body.trailing_scale * base
    scale_literal = part_body.scale_literal
    if part_body.scale_peeled and scale_literal is not None:
        qk_scale_log2 = scale_literal * multiplier
    else:
        qk_scale_log2 = multiplier
    lse_scale = math.log(2.0) if lse_log_base == "log" else 1.0
    regions = part_body.regions
    score_region = (
        None
        if regions.score is None
        else FlexRegion(regions.score, regions.score_output)  # pyrefly: ignore[bad-argument-type]
    )
    mask_region = (
        None if regions.mask is None else FlexRegion(regions.mask, regions.mask_output)  # pyrefly: ignore[bad-argument-type]
    )
    walk = FlexKvWalk(
        part_cnt_name=part_outer.cnt_name,
        part_idx_name=part_inner.idx_name,
        full_cnt_name=full_outer.cnt_name,
        full_idx_name=full_inner.idx_name,
        q_block=q_block,
        kv_block=kv_block,
        table_batch=table_batch,
        table_heads=table_heads,
        trivial=trivial,
    )
    plan = AttentionScorePlan(
        head_dim=head_dim,
        qk_scale_log2=qk_scale_log2,
        lse_scale=lse_scale,
        modifiers=(),
        kv_group=kv_group,
        head_dim_v=None if head_dim_v == head_dim else head_dim_v,
        q_seq=q_seq,
        kv_seq=kv_seq,
        kv_walk=walk,
        score_region=score_region,
        mask_region=mask_region,
        aux_buffers=regions.aux,
        host_scalars=regions.host_scalars,
        scale_arg=part_body.scale_symbol,
        scale_peeled=part_body.scale_peeled,
    )
    return FlexMatch(
        plan=plan,
        b_block_id=ctx.b_id,
        h_block_id=ctx.h_id,
        q_block_id=ctx.m_id,
        kv_block_id=full_inner.block_id,
        full_nest=FlexNest(
            full_outer.graph_id,
            full_inner.graph_id,
            full_outer.block_id,
            full_inner.block_id,
            full_outer.cnt_name,
            full_inner.idx_name,
            False,
        ),
        part_nest=FlexNest(
            part_outer.graph_id,
            part_inner.graph_id,
            part_outer.block_id,
            part_inner.block_id,
            part_outer.cnt_name,
            part_inner.idx_name,
            mask_region is not None,
        ),
        q_name=q_name,
        k_name=full_body.k_name,
        v_name=full_body.v_name,
        o_name=o_name,
        lse_name=lse_name,
        lse_log_base=lse_log_base,
        batch=batch,
        heads=heads,
        kv_heads=kv_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        io_dtype=io_dtype,
        aux_names=aux_names,
        scale_value=scale_literal,
        exp_base=part_body.exp_base,
        trailing_scale=part_body.trailing_scale,
    )


def _phi_chain(
    ctx: _Ctx,
    root: GraphInfo,
    nests: list[tuple[_OuterLoop, _InnerLoop, _Body]],
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node] | None:
    """Prove the (m, l, acc) state flows root inits -> nest 1 -> ``_phi`` ->
    nest 2 -> ``_phi`` in order; return the final three ``_phi`` nodes."""
    from ...language._tracing_ops import _phi

    carried: list[object] | None = None
    final: list[torch.fx.Node] = []
    for outer, inner, body in nests:
        inner_inputs = inner.loop.args[3]
        outer_inputs = outer.loop.args[3]
        assert isinstance(inner_inputs, (list, tuple))
        assert isinstance(outer_inputs, (list, tuple))
        outer_graph = ctx.graphs[outer.graph_id].graph
        outer_outputs = _graph_outputs(outer_graph)
        if outer_outputs is None or len(outer_outputs) != 3:
            ctx.decline("flex: the list body must return (m, l, acc)")
            return None
        root_values: list[object] = []
        for role, inner_index in enumerate(body.role_placeholders):
            if inner_index >= len(inner_inputs):
                ctx.decline("flex: loop-carried role outside the KV loop inputs")
                return None
            outer_index = ctx.placeholder_index(
                outer.graph_id, inner_inputs[inner_index]
            )
            if outer_index is None or outer_index >= len(outer_inputs):
                ctx.decline(
                    "flex: the KV loop state is not threaded through the list loop"
                )
                return None
            lhs = _phi_result(outer_outputs[role], inner.loop, role)
            if ctx.placeholder_index(outer.graph_id, lhs) != outer_index:
                ctx.decline("flex: the list body's outputs do not merge its own state")
                return None
            root_values.append(outer_inputs[outer_index])
        if carried is None:
            for value in root_values:
                if _is_call(value, _phi) or _load_parts(value) is not None:
                    ctx.decline("flex: the first walk does not start from fresh state")
                    return None
        elif any(
            value is not carried_value
            for value, carried_value in zip(root_values, carried, strict=True)
        ):
            ctx.decline(
                "flex: the second walk does not continue the first walk's state"
            )
            return None
        final = []
        for role, value in enumerate(root_values):
            phis = [
                node
                for node in root.graph.nodes
                if _phi_result(node, outer.loop, role) is value
            ]
            if len(phis) != 1:
                ctx.decline("flex: missing _phi after a list walk")
                return None
            final.append(phis[0])
        carried = list(final)
    assert len(final) == 3
    return final[0], final[1], final[2]


def _match_epilogue(
    ctx: _Ctx,
    root: GraphInfo,
    final_m: torch.fx.Node,
    final_l: torch.fx.Node,
    final_acc: torch.fx.Node,
    bhm: tuple[int, int, int],
    head_dim_v: int,
    io_dtype: torch.dtype,
) -> tuple[str, str | None, str | None] | None:
    """``l_safe`` guard, ``acc / l_safe[:, None]`` -> O store and the
    optional LSE store; returns ``(o_name, lse_name, lse_log_base)``."""
    from ...language import memory_ops

    l_safe: torch.fx.Node = final_l
    for user in final_l.users:
        if not _is_call(user, _aten.where.self, _aten.where.ScalarSelf):
            continue
        if len(user.args) != 3 or user.args[2] is not final_l:
            continue
        cond = user.args[0]
        if (
            _is_call(cond, _aten.eq.Scalar)
            and isinstance(cond, torch.fx.Node)
            and cond.args[0] is final_l
            and _scalar_value(cond.args[1]) == 0.0
            and _scalar_value(user.args[1]) == 1.0
        ):
            l_safe = user
            break
    stores = [node for node in root.graph.nodes if _is_call(node, memory_ops.store)]
    for info in ctx.graphs.values():
        if info is root:
            continue
        if any(_is_call(node, memory_ops.store) for node in info.graph.nodes):
            ctx.decline("flex: store inside a loop body")
            return None
    o_name: str | None = None
    lse_name: str | None = None
    lse_log_base: str | None = None
    batch, heads, q_seq = bhm
    for store in stores:
        if len(store.args) < 3 or (len(store.args) > 3 and store.args[3] is not None):
            ctx.decline("flex: masked store")
            return None
        tensor, indices, value = store.args[:3]
        name = _host_tensor_name(tensor)
        host = _host_tensor_value(tensor)
        if name is None or host is None or not isinstance(indices, (list, tuple)):
            ctx.decline("flex: store to a non-host tensor")
            return None
        if (
            len(indices) < 3
            or not ctx.unit_axis_index(indices[0], ctx.b_id)
            or not ctx.unit_axis_index(indices[1], ctx.h_id)
            or ctx.block_symnode_id(indices[2]) != ctx.m_id
        ):
            ctx.decline(f"flex: store to {name} is not indexed [b, h, tile_m, ...]")
            return None
        stored = _strip_identity(value)
        if host.ndim == 4:
            if len(indices) != 4 or not _is_full_slice(indices[3]):
                ctx.decline("flex: the O store needs a trailing full slice")
                return None
            if _is_call(stored, _prims.convert_element_type.default):
                assert isinstance(stored, torch.fx.Node)
                if stored.args[1] != host.dtype:
                    ctx.decline("flex: the O store converts to the wrong dtype")
                    return None
                stored = _strip_identity(stored.args[0])
            if not (
                _is_call(stored, _aten.div.Tensor)
                and isinstance(stored, torch.fx.Node)
                and stored.args[0] is final_acc
                and _strip_identity(stored.args[1]) is l_safe
            ):
                ctx.decline("flex: the O store is not acc / l_safe")
                return None
            if (
                tuple(int(size) for size in host.shape)
                != (batch, heads, q_seq, head_dim_v)
                or host.dtype != io_dtype
                or not host.is_contiguous()
                or o_name is not None
            ):
                ctx.decline("flex: the O tensor geometry is wrong or duplicated")
                return None
            o_name = name
            continue
        if host.ndim == 3:
            if len(indices) != 3:
                ctx.decline("flex: the LSE store must index [b, h, tile_m]")
                return None
            base = _lse_log_base(stored, final_m, l_safe)
            if base is None:
                ctx.decline(
                    "flex: the LSE value is not m + log2(l) (or its natural-log form)"
                )
                return None
            if (
                tuple(int(size) for size in host.shape) != (batch, heads, q_seq)
                or host.dtype != torch.float32
                or not host.is_contiguous()
                or lse_name is not None
            ):
                ctx.decline("flex: the LSE tensor geometry is wrong or duplicated")
                return None
            lse_name, lse_log_base = name, base
            continue
        ctx.decline(f"flex: unexpected store to {name}")
        return None
    if o_name is None:
        ctx.decline("flex: no O store")
        return None
    return o_name, lse_name, lse_log_base


def _lse_log_base(
    value: object, final_m: torch.fx.Node, l_safe: torch.fx.Node
) -> str | None:
    """``"log2"`` for ``m + log2(l)``, ``"log"`` for ``(m + log2(l)) / log2e``,
    ``(m + log2(l)) * ln2`` or ``m + log(l)``."""
    natural = False
    value = _strip_identity(value)
    if _is_call(value, _aten.div.Tensor) and isinstance(value, torch.fx.Node):
        factor = value.args[1]
        if not (
            isinstance(factor, (int, float))
            and math.isclose(factor, LOG2E, rel_tol=1e-6)
        ):
            return None
        natural = True
        value = _strip_identity(value.args[0])
    elif _is_call(value, _aten.mul.Tensor) and isinstance(value, torch.fx.Node):
        lhs, rhs = value.args[:2]
        factor = rhs if isinstance(lhs, torch.fx.Node) else lhs
        tensor = lhs if isinstance(lhs, torch.fx.Node) else rhs
        if not (
            isinstance(factor, (int, float))
            and math.isclose(factor, math.log(2.0), rel_tol=1e-6)
        ):
            return None
        natural = True
        value = _strip_identity(tensor)
    if not (_is_call(value, _aten.add.Tensor) and isinstance(value, torch.fx.Node)):
        return None
    lhs, rhs = value.args[:2]
    log_node = rhs if lhs is final_m else lhs if rhs is final_m else None
    if log_node is None or not _is_call(
        log_node, _aten.log2.default, _aten.log.default
    ):
        return None
    assert isinstance(log_node, torch.fx.Node)
    if _strip_identity(log_node.args[0]) is not l_safe:
        return None
    if log_node.target is _aten.log.default:
        return None if natural else "log"
    return "log" if natural else "log2"


def _root_graph_admitted(ctx: _Ctx, root: GraphInfo) -> bool:
    """Only the prologue / epilogue ops of the envelope may appear in the root."""
    from ...language import creation_ops
    from ...language import memory_ops
    from ...language import tile_ops
    from ...language import view_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _new_var
    from ...language._tracing_ops import _phi

    allowed = {
        _new_var,
        _get_symnode,
        _host_tensor,
        tile_ops.tile_begin,
        tile_ops.tile_index,
        tile_ops.tile_id,
        memory_ops.load,
        memory_ops.store,
        creation_ops.full,
        _aten.full.default,
        _prims.iota.default,
        _aten.add.Tensor,
        _aten.sub.Tensor,
        _aten.mul.Tensor,
        _aten.div.Tensor,
        view_ops.subscript,
        _for_loop,
        operator.getitem,
        _phi,
        _aten.where.self,
        _aten.where.ScalarSelf,
        _aten.eq.Scalar,
        _aten.scalar_tensor.default,
        _aten.log2.default,
        _aten.log.default,
        _prims.convert_element_type.default,
        _aten.sym_size.int,
        *_IDENTITY_VIEW_TARGETS,
    }
    loads = 0
    for node in root.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is memory_ops.load:
            loads += 1
        if node.target in allowed:
            continue
        value = node.meta.get("val")
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            continue
        ctx.decline(f"flex: root graph has {node.target}")
        return False
    if loads != 3:
        ctx.decline("flex: the root graph may load only Q and the two counts")
        return False
    return True
