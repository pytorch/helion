"""Per-element renderer of the flex envelope's score_mod / mask_mod regions.

A flex plan (``AttentionScorePlan.is_flex``) carries the user's ``score_mod``
and ``mask_mod`` as captured pointwise FX regions (``FlexRegion``: placeholders
``score, b, h, q_idx, kv_idx, *aux buffers, *host scalars``).  This module
turns a region into CuTe DSL scalar statements for the flash softmax warps:
one row per thread, the thread's 32-column (or 16-column) score chunk in a
rank-1 rmem fragment, coordinates from the t2r partition.

Rendering rules (``artifacts/attn-sparse-2026-10-10/spec_interfaces.md`` §2):

* Pointwise ``aten`` nodes go through ``PointwiseLowering.codegen_from_input_asts``
  with substituted scalar inputs, so they use the same CuTe op overrides as the
  SIMT path (``sigmoid`` template, ``cute.math.log``, ``exp2``, the
  ``fast_math`` variants); ``where`` is a typed ``cutlass.select_``;
  ``scalar_tensor`` / ``full`` are typed literals; shape ops are the identity;
  integer ``%`` is Python-mod (``_cute_python_mod``); ``tanh`` follows the
  softcap policy (exp2 identity, ``tanh.approx`` under ``fast_math``).
* Every node is placed by its dependence on the coordinates: item-invariant
  (``b``/``h``/``q`` only) values render once per work item, tile-invariant
  (``kv_tile`` only, including ``kv_idx // c`` with ``c % 128 == 0``) once per
  KV step, chunk-invariant once per chunk, the rest in the unrolled element
  loop.  Item-invariant values live across the KV loop in registers, so they
  are capped (``ITEM_LIVE_CAP``) and the cheapest are demoted to the tile
  prologue.
* Captured loads are classified against the KV lane: lane-free indices give a
  single guarded load; a unit-stride last index ``kv_idx + c`` with a 16-byte
  aligned chunk base gives one vector load per group of lanes; anything else
  is a per-element gather.  Gathers clamp the index, load unconditionally and
  ``select_`` zero for out-of-range coordinates (no per-element ``IfExp``).
* Comparisons of ``kv_idx + c`` whose half-open threshold is a multiple of
  the tile / chunk width are constant over it (``kvcmp``), and
  ``(kv_idx + c) % m`` with ``m`` and ``c`` multiples of the width is the
  first lane's remainder plus the lane offset (``kvmod``).
* In the ``"bitmask"`` mask form a mask predicate that is, per chunk, a
  boolean combination of lane intervals (comparisons of lane-linear
  integers, ``abs``, ``x // c`` block bounds, ``x % m`` residue patterns)
  is built as one keep word per chunk from lane-uniform values and applied
  through the constexpr bit-test select chain that ptxas turns into R2P +
  predicated selects; operands outside that language stay per element and
  read the lane bits of the foldable ones.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import operator
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import cast
from unittest.mock import patch

import torch
from torch._inductor.virtualized import V
from torch.fx import Node
from torch.fx import map_arg
from torch.fx.experimental import proxy_tensor
from torch.fx.traceback import preserve_node_meta

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..source_location import SourceLocation
from .attention_plan import FlexAuxBuffer
from .attention_plan import FlexRegion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .attention_plan import AttentionScorePlan


class FlexHookCode(NamedTuple):
    """The three source blocks of one rendered hook (see ``render_flex_hook``)."""

    # Every field is zero or more complete newline-terminated lines at a
    # consistent indentation (the emitter dedents and re-indents each block).
    # Once per work item, before the KV loop.
    item_prelude: str
    # Once per KV step, before the ``s_full`` wait.
    tile_prologue: str
    # Inside the PASS1 chunk loop after the t2r copy (nested loops inside).
    chunk_body: str


# Item-invariant values held in registers across the KV loop.
ITEM_LIVE_CAP = 8

# Placement classes, ordered by nesting depth.
_ITEM = 0
_TILE = 1
_CHUNK = 2
_GROUP = 3
_ELEMENT = 4

# Dependence tokens.
_T_B = "b"
_T_H = "h"
_T_Q = "q"
_T_KVT = "kvt"
_T_CHUNK = "chunk"
_T_LANE = "lane"
_ITEM_TOKENS = frozenset({_T_B, _T_H, _T_Q})
_TILE_TOKENS = _ITEM_TOKENS | {_T_KVT}
_CHUNK_TOKENS = _TILE_TOKENS | {_T_CHUNK}

# Loop variables of the rendered element loop.
_ELEM = "flash_flex_i"
_GROUP_VAR = "flash_flex_g"
_LANE_VAR = "flash_flex_j"

_SHAPE_OP_TARGETS = frozenset(
    {
        view_ops.subscript,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.expand.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.prims.broadcast_in_dim.default,
    }
)
_LITERAL_TARGETS = frozenset(
    {torch.ops.aten.scalar_tensor.default, torch.ops.aten.full.default}
)
_WHERE_TARGETS = frozenset(
    {
        torch.ops.aten.where.self,
        torch.ops.aten.where.ScalarOther,
        torch.ops.aten.where.ScalarSelf,
        torch.ops.aten.where.Scalar,
    }
)
_REMAINDER_TARGETS = frozenset(
    {torch.ops.aten.remainder.Scalar, torch.ops.aten.remainder.Tensor}
)
_FMOD_TARGETS = frozenset({torch.ops.aten.fmod.Scalar, torch.ops.aten.fmod.Tensor})
# fp32 unary ops torch's CuteDSL overrides do not render; the DSL has them.
# ``round`` is torch's round-half-to-even.
_UNARY_MATH_TARGETS = {
    torch.ops.aten.ceil.default: "ceil",
    torch.ops.aten.trunc.default: "trunc",
    torch.ops.aten.round.default: "roundeven",
}
_FLOORDIV_TARGETS = frozenset(
    {
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.floor_divide.Scalar,
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.div.Scalar_mode,
    }
)
_ADD_TARGETS = frozenset({torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar})
_SUB_TARGETS = frozenset({torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar})
_MUL_TARGETS = frozenset({torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar})
# Boolean connectives of a mask predicate (the keep-word peephole combines
# their operands' lane words bitwise).
_BOOL_AND_TARGETS = frozenset(
    {
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.logical_and.default,
    }
)
_BOOL_OR_TARGETS = frozenset(
    {
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.logical_or.default,
    }
)
_BOOL_XOR_TARGETS = frozenset(
    {
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.bitwise_xor.Scalar,
        torch.ops.aten.logical_xor.default,
    }
)
_BOOL_NOT_TARGETS = frozenset(
    {torch.ops.aten.bitwise_not.default, torch.ops.aten.logical_not.default}
)
_ABS_TARGET = torch.ops.aten.abs.default
# ``a cmp b`` as ``b cmp' a`` and as ``not (a cmp b)``.
_FLIP_COMPARE = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
_NEGATE_COMPARE = {
    "lt": "ge",
    "le": "gt",
    "gt": "le",
    "ge": "lt",
    "eq": "ne",
    "ne": "eq",
}
# Mask forms of ``render_flex_hook``: the per-chunk keep word applied through
# the R2P select chain, or the per-element predicate.
MASK_FORMS = ("bitmask", "element")
_COMPARE_OPERATORS = {
    torch.ops.aten.gt.Tensor: "gt",
    torch.ops.aten.gt.Scalar: "gt",
    torch.ops.aten.ge.Tensor: "ge",
    torch.ops.aten.ge.Scalar: "ge",
    torch.ops.aten.lt.Tensor: "lt",
    torch.ops.aten.lt.Scalar: "lt",
    torch.ops.aten.le.Tensor: "le",
    torch.ops.aten.le.Scalar: "le",
    torch.ops.aten.eq.Tensor: "eq",
    torch.ops.aten.eq.Scalar: "eq",
    torch.ops.aten.ne.Tensor: "ne",
    torch.ops.aten.ne.Scalar: "ne",
}
_INT_DTYPES = frozenset(
    {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
)

# Natural 16-byte vector width per element size.
_VECTOR_WIDTH = {1: 16, 2: 8, 4: 4}
_VECTOR_BYTES = 16
# Lanes of one window vector (``load_window``): a lane-affine gather
# ``table[.., a*kv_idx + rest]`` with ``a = +-1`` reads ``chunk_cols``
# consecutive entries per thread and chunk; the thread's first entry is not
# vector-aligned (it depends on the row), so the window is loaded from the
# aligned-down base in ``_WINDOW_LANES``-wide vectors and each element picks
# its lane with ``_WINDOW_LANES - 1`` selects.  Two lanes (one select per
# element, ``chunk_cols // 2 + 1`` loads per chunk) minimise the issue count
# for every element width.
_WINDOW_LANES = 2


def _unsupported(message: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"flex score_mod renderer: {message}")


def _py_literal(value: object) -> str:
    """A Python source literal for a constant (``inf`` needs ``float(...)``)."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        if math.isnan(value):
            return "float('nan')"
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    raise _unsupported(f"unsupported literal {value!r}")


def _is_int_dtype(dtype: torch.dtype) -> bool:
    return dtype in _INT_DTYPES


def _node_dtype(node: Node) -> torch.dtype | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype
    if isinstance(value, torch.SymInt):
        return torch.int32
    if isinstance(value, torch.SymFloat):
        return torch.float32
    if isinstance(value, torch.SymBool):
        return torch.bool
    return None


@dataclasses.dataclass(frozen=True)
class _Affine:
    """An integer affine form over the coordinate roots ``b, h, q, kv``."""

    coeffs: tuple[tuple[str, int], ...]
    const: int

    @staticmethod
    def constant(value: int) -> _Affine:
        return _Affine((), value)

    @staticmethod
    def root(name: str) -> _Affine:
        return _Affine(((name, 1),), 0)

    def as_dict(self) -> dict[str, int]:
        return dict(self.coeffs)

    @staticmethod
    def _from_dict(coeffs: dict[str, int], const: int) -> _Affine:
        return _Affine(
            tuple(sorted((k, v) for k, v in coeffs.items() if v != 0)), const
        )

    def add(self, other: _Affine, sign: int = 1) -> _Affine:
        coeffs = self.as_dict()
        for key, value in other.coeffs:
            coeffs[key] = coeffs.get(key, 0) + sign * value
        return _Affine._from_dict(coeffs, self.const + sign * other.const)

    def scale(self, factor: int) -> _Affine:
        return _Affine._from_dict(
            {k: v * factor for k, v in self.coeffs}, self.const * factor
        )

    def is_kv_plus_const(self) -> bool:
        return self.coeffs == (("kv", 1),)

    def kv_coefficient(self) -> int:
        return dict(self.coeffs).get("kv", 0)

    def interval(self, bounds: dict[str, int]) -> tuple[int, int] | None:
        """``[lo, hi]`` of the form over roots in ``[0, bounds[root])``;
        ``None`` when a root has no bound."""
        lo = hi = self.const
        for root, coef in self.coeffs:
            bound = bounds.get(root)
            if bound is None or bound <= 0:
                return None
            if coef > 0:
                hi += coef * (bound - 1)
            else:
                lo += coef * (bound - 1)
        return lo, hi


@dataclasses.dataclass
class _NodeInfo:
    role: str
    deps: frozenset[str]
    affine: _Affine | None
    # Placement class after demotion.
    cls: int = _ITEM
    demoted: bool = False
    # Per-role payload: aux index for loads, placeholder slot for coordinates
    # and host scalars, the constant divisor of ``kv_idx // c`` forms.
    aux_index: int = -1
    scalar_index: int = -1
    divisor: int = 0
    vector_width: int = 0
    # Loads: every index proven inside the table from the coordinate bounds
    # (no guard, no clamp); the window form's lane count and the sign of the
    # last index's ``kv_idx`` coefficient.
    in_bounds: bool = False
    window: int = 0
    window_sign: int = 1
    # ``kvcmp`` (a comparison of ``kv_idx + c`` constant over a tile / chunk)
    # and ``kvmod`` (``(kv_idx + c) % m`` whose remainder is lane-affine over
    # a tile / chunk): the class of the hoisted part, the offset ``c`` and,
    # for ``kvcmp``, the comparison of ``base + c`` against zero.
    uniform_cls: int = _ELEMENT
    offset: int = 0
    cmp_op: str = ""


@dataclasses.dataclass
class _RegionInfo:
    tag: str
    region: FlexRegion
    nodes: dict[Node, _NodeInfo]
    placeholders: list[Node]
    output: Node


@dataclasses.dataclass(frozen=True)
class _Lin:
    """A lane-linear integer of one chunk: ``coef * j + sum(k * term) + const``
    over the lane index ``j`` (``kv_idx = cbase + j``); ``terms`` are rendered
    lane-uniform ``Int32`` expressions, ``cls`` the placement class of the
    uniform part."""

    coef: int
    # ``(expression, multiplier, placement class)`` per uniform term.
    terms: tuple[tuple[str, int, int], ...]
    const: int

    @property
    def cls(self) -> int:
        return max([_ITEM, *(cls for _expr, _value, cls in self.terms)])

    @staticmethod
    def constant(value: int) -> _Lin:
        return _Lin(0, (), value)

    @staticmethod
    def term(expr: str, cls: int, coef: int = 0) -> _Lin:
        return _Lin(coef, ((expr, 1, cls),), 0)

    def add(self, other: _Lin, sign: int = 1) -> _Lin:
        terms = {key: (value, cls) for key, value, cls in self.terms}
        for key, value, cls in other.terms:
            terms[key] = (terms.get(key, (0, cls))[0] + sign * value, cls)
        return _Lin(
            self.coef + sign * other.coef,
            tuple(sorted((k, v, c) for k, (v, c) in terms.items() if v != 0)),
            self.const + sign * other.const,
        )

    def scale(self, factor: int) -> _Lin:
        if factor == 0:
            return _Lin.constant(0)
        return _Lin(
            self.coef * factor,
            tuple((k, v * factor, c) for k, v, c in self.terms),
            self.const * factor,
        )

    def uniform_expr(self) -> str:
        """The uniform part (without ``coef * j``) as an ``Int32`` expression."""
        parts: list[str] = []
        for key, value, _cls in self.terms:
            if value == 1:
                parts.append(key)
            elif value == -1:
                parts.append(f"-{key}")
            else:
                parts.append(f"{key} * cutlass.Int32({value})")
        if self.const or not parts:
            parts.append(
                f"-cutlass.Int32({-self.const})"
                if self.const < 0
                else f"cutlass.Int32({self.const})"
            )
        expr = parts[0]
        for part in parts[1:]:
            expr = (
                f"{expr} - {part[1:]}" if part.startswith("-") else f"{expr} + {part}"
            )
        return expr


@dataclasses.dataclass(frozen=True)
class _Word:
    """A chunk keep word: ``expr`` names an ``Int32`` whose bit ``j`` is the
    predicate at lane ``j``, or ``None`` for a lane-uniform predicate whose
    rendered Boolean is ``uniform``; ``cls`` is the placement class."""

    expr: str | None
    uniform: str | None
    cls: int


def _class_of(deps: frozenset[str]) -> int:
    if deps <= _ITEM_TOKENS:
        return _ITEM
    if deps <= _TILE_TOKENS:
        return _TILE
    if deps <= _CHUNK_TOKENS:
        return _CHUNK
    return _ELEMENT


def _int_constant(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Node) and value.target in _LITERAL_TARGETS:
        literal = (
            value.args[-1]
            if value.target is torch.ops.aten.full.default
            else (value.args[0])
        )
        dtype = _node_dtype(value)
        if isinstance(literal, int) and not isinstance(literal, bool):
            return literal
        if (
            isinstance(literal, float)
            and dtype is not None
            and _is_int_dtype(dtype)
            and float(literal).is_integer()
        ):
            return int(literal)
    return None


def _floordiv_operands(node: Node) -> tuple[Node, object] | None:
    """``(x, divisor)`` for a Python floor division node, else ``None``."""
    if node.target not in _FLOORDIV_TARGETS:
        return None
    if node.target in (torch.ops.aten.div.Tensor_mode, torch.ops.aten.div.Scalar_mode):
        mode = node.kwargs.get("rounding_mode")
        if mode is None and len(node.args) > 2:
            mode = node.args[2]
        if mode != "floor":
            return None
    x = node.args[0]
    if not isinstance(x, Node):
        return None
    return x, node.args[1]


def _lowering_inputs(node: Node) -> list[Node]:
    """The node's lowering inputs in ``InductorLowering.input_asts`` order."""
    inputs: list[Node] = []

    def visit(value: Node) -> Node:
        if value not in inputs:
            inputs.append(value)
        return value

    map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
    return inputs


class _Analyzer:
    """Dependence / affine analysis of one region."""

    def __init__(
        self,
        plan: AttentionScorePlan,
        chunk_cols: int,
        coord_bounds: Sequence[int | None] | None = None,
    ) -> None:
        self.plan = plan
        self.chunk_cols = chunk_cols
        self.num_aux = len(plan.aux_buffers)
        self.num_scalars = len(plan.host_scalars)
        # Exclusive upper bounds of the coordinate roots (``b``, ``h``, ``q``,
        # ``kv``) as the kernel evaluates the hook: the rows and columns of
        # every tile the walk can visit, live or not.
        self.bounds: dict[str, int] = {}
        if coord_bounds is not None:
            for root, bound in zip(("b", "h", "q", "kv"), coord_bounds, strict=True):
                if bound is not None:
                    self.bounds[root] = int(bound)

    def analyze(self, region: FlexRegion, tag: str) -> _RegionInfo:
        nodes: dict[Node, _NodeInfo] = {}
        placeholders = [n for n in region.graph.nodes if n.op == "placeholder"]
        expected = 5 + self.num_aux + self.num_scalars
        if len(placeholders) != expected:
            raise _unsupported(
                f"{tag} region has {len(placeholders)} placeholders, expected "
                f"{expected} (score, b, h, q_idx, kv_idx, {self.num_aux} aux, "
                f"{self.num_scalars} host scalars)"
            )
        for slot, node in enumerate(placeholders):
            nodes[node] = self._placeholder(slot)
        for node in region.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                continue
            if node.op != "call_function":
                raise _unsupported(f"unexpected {node.op} node {node.name}")
            nodes[node] = self._call(node, nodes)
        for info in nodes.values():
            info.cls = _class_of(info.deps)
        return _RegionInfo(tag, region, nodes, placeholders, region.output)

    def _placeholder(self, slot: int) -> _NodeInfo:
        if slot == 0:
            return _NodeInfo("score", frozenset({_T_LANE}), None)
        if slot == 1:
            return _NodeInfo("b", frozenset({_T_B}), _Affine.root("b"))
        if slot == 2:
            return _NodeInfo("h", frozenset({_T_H}), _Affine.root("h"))
        if slot == 3:
            return _NodeInfo("q", frozenset({_T_Q}), _Affine.root("q"))
        if slot == 4:
            return _NodeInfo("kv", frozenset({_T_KVT, _T_LANE}), _Affine.root("kv"))
        if slot < 5 + self.num_aux:
            return _NodeInfo("aux", frozenset(), None, aux_index=slot - 5)
        return _NodeInfo("hs", frozenset(), None, scalar_index=slot - 5 - self.num_aux)

    def _input_deps(self, node: Node, nodes: dict[Node, _NodeInfo]) -> frozenset[str]:
        deps: set[str] = set()
        for value in _lowering_inputs(node):
            info = nodes.get(value)
            if info is None:
                raise _unsupported(
                    f"{node.name} reads {value.name} from outside the region"
                )
            if info.role == "aux":
                raise _unsupported(
                    f"{node.name} uses aux buffer {value.name} outside of hl.load"
                )
            deps |= info.deps
        return frozenset(deps)

    def _affine_of(self, value: object, nodes: dict[Node, _NodeInfo]) -> _Affine | None:
        if isinstance(value, Node):
            return nodes[value].affine
        constant = _int_constant(value)
        if constant is not None:
            return _Affine.constant(constant)
        return None

    def _call(self, node: Node, nodes: dict[Node, _NodeInfo]) -> _NodeInfo:
        target = node.target
        dtype = _node_dtype(node)
        if target is _tracing_ops._host_tensor:
            # A matcher-copied load may keep its ``_host_tensor`` source node.
            for aux_index, buffer in enumerate(self.plan.aux_buffers):
                if buffer.name == node.args[0]:
                    return _NodeInfo("aux", frozenset(), None, aux_index=aux_index)
            raise _unsupported(
                f"{node.name}: host tensor {node.args[0]!r} is not an aux buffer"
            )
        if target in _SHAPE_OP_TARGETS:
            source = node.args[0]
            assert isinstance(source, Node)
            info = nodes[source]
            if info.role == "aux":
                raise _unsupported(f"{node.name} reshapes aux buffer {source.name}")
            return _NodeInfo("alias", info.deps, info.affine)
        if target in _LITERAL_TARGETS:
            constant = _int_constant(node)
            affine = _Affine.constant(constant) if constant is not None else None
            return _NodeInfo("literal", frozenset(), affine)
        if target is memory_ops.load:
            return self._load(node, nodes)
        deps = self._input_deps(node, nodes)
        is_int = dtype is not None and _is_int_dtype(dtype)
        if (operands := _floordiv_operands(node)) is not None and is_int:
            x, divisor = operands
            c = _int_constant(divisor)
            x_affine = nodes[x].affine
            if c is not None and c > 0 and x_affine is not None:
                if x_affine.is_kv_plus_const():
                    if c % 128 == 0 and x_affine.const % 128 == 0:
                        return _NodeInfo("kvdiv", frozenset({_T_KVT}), None, divisor=c)
                    if (
                        c % self.chunk_cols == 0
                        and x_affine.const % self.chunk_cols == 0
                    ):
                        return _NodeInfo(
                            "kvdiv",
                            frozenset({_T_KVT, _T_CHUNK}),
                            None,
                            divisor=c,
                        )
                if not x_affine.coeffs:
                    return _NodeInfo(
                        "pointwise", deps, _Affine.constant(x_affine.const // c)
                    )
            return _NodeInfo("pointwise", deps, None)
        if target in _REMAINDER_TARGETS:
            if is_int and (kvmod := self._kvmod(node, nodes)) is not None:
                return kvmod
            return _NodeInfo("pymod" if is_int else "remainder_f", deps, None)
        if target in _COMPARE_OPERATORS and (kvcmp := self._kvcmp(node, nodes)):
            return kvcmp
        if target in _FMOD_TARGETS:
            return _NodeInfo("fmod", deps, None)
        if target is torch.ops.aten.tanh.default:
            return _NodeInfo("tanh", deps, None)
        if target is torch.ops.aten.sign.default:
            return _NodeInfo("sign", deps, None)
        if target in _UNARY_MATH_TARGETS and dtype is torch.float32:
            return _NodeInfo("unary_math", deps, None)
        if target in _WHERE_TARGETS:
            return _NodeInfo("where", deps, None)
        if target is torch.ops.prims.convert_element_type.default:
            source = node.args[0]
            affine = (
                nodes[source].affine if isinstance(source, Node) and is_int else None
            )
            return _NodeInfo("convert", deps, affine)
        affine: _Affine | None = None
        if is_int and len(node.args) >= 2:
            left = self._affine_of(node.args[0], nodes)
            right = self._affine_of(node.args[1], nodes)
            if left is not None and right is not None:
                if target in _ADD_TARGETS and node.kwargs.get("alpha", 1) == 1:
                    affine = left.add(right)
                elif target in _SUB_TARGETS and node.kwargs.get("alpha", 1) == 1:
                    affine = left.add(right, -1)
                elif target in _MUL_TARGETS:
                    if not right.coeffs:
                        affine = left.scale(right.const)
                    elif not left.coeffs:
                        affine = right.scale(left.const)
        elif is_int and target is torch.ops.aten.neg.default:
            inner = self._affine_of(node.args[0], nodes)
            affine = inner.scale(-1) if inner is not None else None
        return _NodeInfo("pointwise", deps, affine)

    def _spans(self) -> tuple[tuple[int, int, frozenset[str]], ...]:
        """``(span, class, deps)`` of the lane spans a ``kv_idx + c`` form can
        be hoisted over: the 128-column tile, then the chunk."""
        return (
            (128, _TILE, frozenset({_T_KVT})),
            (self.chunk_cols, _CHUNK, frozenset({_T_KVT, _T_CHUNK})),
        )

    def _kvcmp(self, node: Node, nodes: dict[Node, _NodeInfo]) -> _NodeInfo | None:
        """A comparison ``kv_idx + c <op> 0`` (either operand order, ``kv``
        coefficient +-1, no other coordinate) that is constant over a tile or
        a chunk: in half-open form the threshold is a multiple of the span."""
        op = _COMPARE_OPERATORS[cast("Any", node.target)]
        if op in ("eq", "ne") or len(node.args) < 2:
            return None
        left = self._affine_of(node.args[0], nodes)
        right = self._affine_of(node.args[1], nodes)
        if left is None or right is None:
            return None
        d = left.add(right, -1)
        if d.coeffs != (("kv", 1),) and d.coeffs != (("kv", -1),):
            return None
        c = d.const
        if d.coeffs[0][1] == -1:
            c, op = -c, _FLIP_COMPARE[op]
        # ``base + j + c <op> 0`` over ``j in [0, span)``: ``lt`` / ``ge``
        # split at ``j = -c - base``, ``le`` / ``gt`` at ``j = 1 - c - base``;
        # with ``base`` a multiple of the span the lanes agree when the
        # threshold is one too, and the value is that of ``base + c``.
        threshold = -c if op in ("lt", "ge") else 1 - c
        for span, cls, deps in self._spans():
            if threshold % span == 0:
                return _NodeInfo(
                    "kvcmp", deps, None, uniform_cls=cls, offset=c, cmp_op=op
                )
        return None

    def _kvmod(self, node: Node, nodes: dict[Node, _NodeInfo]) -> _NodeInfo | None:
        """``(kv_idx + c) % m`` with ``m`` and ``c`` multiples of a span: the
        remainder is ``(base + c) % m`` at the span's first lane plus the lane
        offset (no wrap inside the span)."""
        x = node.args[0]
        m = _int_constant(node.args[1])
        if not isinstance(x, Node) or m is None or m <= 0:
            return None
        x_affine = nodes[x].affine
        if x_affine is None or not x_affine.is_kv_plus_const():
            return None
        c = x_affine.const
        for span, cls, deps in self._spans():
            if m % span == 0 and c % span == 0:
                return _NodeInfo(
                    "kvmod",
                    deps | {_T_LANE},
                    None,
                    divisor=m,
                    uniform_cls=cls,
                    offset=c,
                )
        return None

    def _load(self, node: Node, nodes: dict[Node, _NodeInfo]) -> _NodeInfo:
        source, indices = node.args[0], node.args[1]
        if isinstance(source, Node) and nodes[source].role == "aux":
            aux_index = nodes[source].aux_index
        else:
            source_name = node.meta.get("flex_source_name")
            names = [buffer.name for buffer in self.plan.aux_buffers]
            if not isinstance(source_name, str) or source_name not in names:
                raise _unsupported(f"{node.name} loads from a non-aux tensor")
            aux_index = names.index(source_name)
        if node.args[2:] and any(arg is not None for arg in node.args[2:]):
            raise _unsupported(f"{node.name} uses extra_mask / eviction_policy")
        if not isinstance(indices, (list, tuple)):
            raise _unsupported(f"{node.name} has a non-list index")
        buffer = self.plan.aux_buffers[aux_index]
        if len(indices) != len(buffer.shape):
            raise _unsupported(
                f"{node.name} indexes {buffer.name} with {len(indices)} indices "
                f"for rank {len(buffer.shape)}"
            )
        deps: set[str] = set()
        for index in indices:
            if isinstance(index, Node):
                info = nodes[index]
                if info.role == "aux":
                    raise _unsupported(f"{node.name} indexes with aux buffer")
                deps |= info.deps
            elif _int_constant(index) is None:
                raise _unsupported(
                    f"{node.name}: index {index!r} must be a tensor or int literal"
                )
        info = _NodeInfo("load", frozenset(deps), None, aux_index=aux_index)
        info.in_bounds = self._in_bounds(indices, buffer, nodes)
        info.vector_width = self._vector_width(node, buffer, nodes)
        if info.in_bounds and not info.vector_width:
            info.window, info.window_sign = self._window(indices, buffer, nodes)
        return info

    def _in_bounds(
        self,
        indices: Sequence[object],
        buffer: FlexAuxBuffer,
        nodes: dict[Node, _NodeInfo],
    ) -> bool:
        """Every index is an affine form of the coordinates whose range over
        the kernel's coordinate bounds lies inside the table."""
        for dim, index in enumerate(indices):
            size = buffer.shape[dim]
            if isinstance(index, Node):
                affine = nodes[index].affine
                interval = None if affine is None else affine.interval(self.bounds)
                if interval is None:
                    return False
                lo, hi = interval
            else:
                constant = _int_constant(index)
                assert constant is not None
                lo = hi = constant
            if lo < 0 or hi >= size:
                return False
        return True

    def _window(
        self,
        indices: Sequence[object],
        buffer: FlexAuxBuffer,
        nodes: dict[Node, _NodeInfo],
    ) -> tuple[int, int]:
        """``(lanes, sign)`` of the window form: the unit-stride last index
        is ``+-kv_idx + <lane-free>``, every other index is lane-free and
        keeps the row base vector-aligned.  ``(0, 1)`` otherwise."""
        if not indices or buffer.strides[-1] != 1:
            return 0, 1
        last = indices[-1]
        if not isinstance(last, Node):
            return 0, 1
        affine = nodes[last].affine
        if affine is None or affine.kv_coefficient() not in (1, -1):
            return 0, 1
        lanes = _WINDOW_LANES
        if buffer.shape[-1] % lanes != 0 or self.chunk_cols % lanes != 0:
            return 0, 1
        for dim, index in enumerate(indices[:-1]):
            if isinstance(index, Node) and _T_LANE in nodes[index].deps:
                return 0, 1
            if buffer.strides[dim] % lanes != 0:
                return 0, 1
        return lanes, affine.kv_coefficient()

    def _vector_width(
        self, node: Node, buffer: FlexAuxBuffer, nodes: dict[Node, _NodeInfo]
    ) -> int:
        """Natural lane-contiguous vector width of a load, or 0 (gather)."""
        indices = cast("Sequence[object]", node.args[1])
        if not indices or buffer.strides[-1] != 1:
            return 0
        last = indices[-1]
        if not isinstance(last, Node):
            return 0
        affine = nodes[last].affine
        if affine is None or not affine.is_kv_plus_const():
            return 0
        itemsize = buffer.dtype.itemsize
        width = _VECTOR_WIDTH.get(itemsize, 0)
        if width == 0:
            return 0
        if buffer.shape[-1] % width != 0 or affine.const % width != 0:
            return 0
        for dim, index in enumerate(indices[:-1]):
            if isinstance(index, Node) and _T_LANE in nodes[index].deps:
                return 0
            if (buffer.strides[dim] * itemsize) % _VECTOR_BYTES != 0:
                return 0
        return width


def _demote_item_values(infos: list[_RegionInfo], cap: int) -> None:
    """Keep at most ``cap`` item-invariant values live across the KV loop."""
    candidates: list[tuple[int, Node, _NodeInfo]] = []
    order = 0
    for info in infos:
        for node, meta in info.nodes.items():
            order += 1
            if meta.cls == _ITEM and meta.role not in (
                "b",
                "h",
                "q",
                "kv",
                "aux",
                "hs",
                "literal",
                "alias",
                "score",
            ):
                candidates.append((order, node, meta))
    lookup = {node: meta for info in infos for node, meta in info.nodes.items()}

    def effective_class(node: Node) -> int:
        meta = lookup.get(node)
        if meta is None:
            return _ELEMENT
        return _TILE if meta.demoted else meta.cls

    def live() -> list[tuple[int, Node, _NodeInfo]]:
        result = []
        for order, node, meta in candidates:
            if meta.demoted:
                continue
            is_output = any(node is info.output for info in infos)
            if is_output or any(effective_class(u) != _ITEM for u in node.users):
                result.append((order, node, meta))
        return result

    def demote(node: Node) -> None:
        meta = lookup[node]
        if meta.demoted or meta.cls != _ITEM:
            return
        meta.demoted = True
        meta.cls = _TILE
        for user in node.users:
            if user in lookup and lookup[user].cls == _ITEM:
                demote(user)

    while len(current := live()) > cap:
        victim = min(
            current, key=lambda entry: (8 if entry[2].role == "load" else 1, -entry[0])
        )
        demote(victim[1])


class _HookEmitter:
    def __init__(
        self,
        plan: AttentionScorePlan,
        *,
        stage: int,
        score_tensor: str,
        coord_tensor: str,
        m_tile_expr: str,
        kv_tile_expr: str,
        chunk_expr: str,
        chunk_cols: int,
        row_coord_expr: str,
        device_function: DeviceFunction,
    ) -> None:
        self.plan = plan
        self.stage = stage
        self.score_tensor = score_tensor
        self.coord_tensor = coord_tensor
        self.m_tile_expr = m_tile_expr
        self.kv_tile_expr = kv_tile_expr
        self.chunk_expr = chunk_expr
        self.chunk_cols = chunk_cols
        self.row_coord_expr = row_coord_expr
        self.device_function = device_function
        self.codegen = device_function.codegen
        self.env = CompileEnvironment.current()
        self.blocks: dict[int, list[str]] = {
            _ITEM: [],
            _TILE: [],
            _CHUNK: [],
            _GROUP: [],
            _ELEMENT: [],
        }
        self.roots_emitted: set[str] = set()
        self.values: dict[tuple[str, Node], str] = {}
        self.vector_width = 0
        # Keep words of the mask region's predicate nodes (``None``: the node
        # has no word form) and the per-region word counter.
        self.words: dict[tuple[str, Node], _Word | None] = {}
        self.word_count: dict[str, int] = {}
        self.word_names: dict[tuple[str, str], str] = {}
        # ``kvmod`` nodes: the remainder at the chunk's first lane.
        self.kvmod_lane0: dict[tuple[str, Node], str] = {}

    # -- naming / placement -------------------------------------------------

    def prefix(self) -> str:
        return f"flash_flex{self.stage}"

    def name_for(self, info: _RegionInfo, node: Node) -> str:
        return f"{self.prefix()}{info.tag}_{node.name}"

    def emit(self, cls: int, line: str) -> None:
        self.blocks[cls].append(line)

    def dtype_str(self, dtype: torch.dtype) -> str:
        return self.env.backend.dtype_str(dtype)

    def root(self, name: str) -> str:
        """The expression of a coordinate root, emitting its definition once."""
        var = f"{self.prefix()}_{name}"
        if name in self.roots_emitted:
            return var
        self.roots_emitted.add(name)
        if name == "b":
            self.emit(
                _ITEM, f"{var} = cutlass.Int32(flash_bh) // cutlass.Int32(_flash_heads)"
            )
        elif name == "h":
            self.emit(
                _ITEM, f"{var} = cutlass.Int32(flash_bh) % cutlass.Int32(_flash_heads)"
            )
        elif name == "q":
            self.emit(
                _ITEM,
                f"{var} = cutlass.Int32({self.row_coord_expr}) + "
                f"cutlass.Int32({self.m_tile_expr}) * cutlass.Int32(128)",
            )
        elif name == "kvbase":
            self.emit(
                _TILE,
                f"{var} = cutlass.Int32({self.kv_tile_expr}) * cutlass.Int32(128)",
            )
        elif name == "cbase":
            self.emit(
                _CHUNK,
                f"{var} = {self.root('kvbase')} + "
                f"cutlass.Int32({self.coord_tensor}[0][1])",
            )
        elif name == "gbase":
            self.emit(
                _GROUP,
                f"{var} = {self.root('kvbase')} + cutlass.Int32("
                f"{self.coord_tensor}[{_GROUP_VAR} * {self.vector_width}][1])",
            )
        elif name == "kv":
            self.emit(
                _ELEMENT,
                f"{var} = {self.root('kvbase')} + "
                f"cutlass.Int32({self.coord_tensor}[{_ELEM}][1])",
            )
        else:
            raise AssertionError(name)
        return var

    # -- values -------------------------------------------------------------

    def value(self, info: _RegionInfo, node: Node) -> str:
        key = (info.tag, node)
        if key in self.values:
            return self.values[key]
        meta = info.nodes[node]
        word = self.words.get(key)
        if word is not None and word.expr is not None and _T_LANE in meta.deps:
            # A lane-dependent predicate with a keep word, read per element by
            # an operand the peephole could not fold: its lane bit.
            result = self.name_for(info, node)
            self.emit(_ELEMENT, f"{result} = {self.lane_bit(word.expr)}")
        else:
            result = self._render(info, node, meta)
        self.values[key] = result
        return result

    def scalar_value(self, info: _RegionInfo, value: object) -> str:
        if isinstance(value, Node):
            return self.value(info, value)
        return _py_literal(value)

    def _render(self, info: _RegionInfo, node: Node, meta: _NodeInfo) -> str:
        role = meta.role
        if role == "score":
            return f"cutlass.Float32({self.score_tensor}[{_ELEM}])"
        if role in ("b", "h", "q", "kv"):
            return self.root(role)
        if role == "hs":
            # Kernel params arrive uncast (Python / DSL scalars); type them here.
            dtype = _node_dtype(node)
            t = self.dtype_str(dtype if dtype is not None else torch.int32)
            scalar_name = self.plan.host_scalars[meta.scalar_index]
            if scalar_name == self.plan.scale_arg:
                return f"{t}(_flash_scale)"
            return f"{t}(_flash_hs{meta.scalar_index})"
        if role == "aux":
            raise _unsupported(f"aux buffer {node.name} used as a value")
        if role == "alias":
            source = node.args[0]
            assert isinstance(source, Node)
            return self.value(info, source)
        if role == "literal":
            return self._literal(node)
        name = self.name_for(info, node)
        if role == "load":
            self._load(info, node, meta, name)
        elif role == "kvdiv":
            self._kvdiv(info, node, meta, name)
        elif role == "kvcmp":
            self._kvcmp(info, node, meta, name)
        elif role == "kvmod":
            self._kvmod(info, node, meta, name)
        elif role == "pymod":
            self._pymod(info, node, meta, name)
        elif role in ("remainder_f", "fmod"):
            self._helper_binary(info, node, meta, name)
        elif role == "sign":
            self._sign(info, node, meta, name)
        elif role == "unary_math":
            self._unary_math(info, node, meta, name)
        elif role == "tanh":
            self._tanh(info, node, meta, name)
        elif role == "where":
            self._where(info, node, meta, name)
        elif role == "convert":
            self._convert(info, node, meta, name)
        else:
            self._pointwise(info, node, meta, name)
        return name

    def _literal(self, node: Node) -> str:
        dtype = _node_dtype(node)
        assert dtype is not None
        value = (
            node.args[1] if node.target is torch.ops.aten.full.default else node.args[0]
        )
        if not isinstance(value, (bool, int, float)):
            raise _unsupported(f"{node.name}: non-constant fill value {value!r}")
        return f"{self.dtype_str(dtype)}({_py_literal(value)})"

    def _kvdiv(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        operands = _floordiv_operands(node)
        assert operands is not None
        x_affine = info.nodes[operands[0]].affine
        assert x_affine is not None
        base = self.root("kvbase") if meta.cls == _TILE else self.root("cbase")
        offset = f" + cutlass.Int32({x_affine.const})" if x_affine.const else ""
        self.emit(
            meta.cls,
            f"{name} = ({base}{offset}) // cutlass.Int32({meta.divisor})",
        )

    def _pymod(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        dtype = _node_dtype(node)
        assert dtype is not None
        t = self.dtype_str(dtype)
        left = self.scalar_value(info, node.args[0])
        right = self.scalar_value(info, node.args[1])
        self.emit(meta.cls, f"{name} = _cute_python_mod({t}({left}), {t}({right}))")

    def _helper_binary(
        self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str
    ) -> None:
        dtype = _node_dtype(node)
        assert dtype is not None
        t = self.dtype_str(dtype)
        helper = "_cute_flex_fmod" if meta.role == "fmod" else "_cute_flex_remainder"
        left = self.scalar_value(info, node.args[0])
        right = self.scalar_value(info, node.args[1])
        self.emit(meta.cls, f"{name} = {t}({helper}({t}({left}), {t}({right})))")

    def _sign(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        # ``select_`` chain; a NaN input gives 0 where torch gives NaN.
        dtype = _node_dtype(node)
        assert dtype is not None
        t = self.dtype_str(dtype)
        x = self.scalar_value(info, node.args[0])
        self.emit(
            meta.cls,
            f"{name} = {t}(cutlass.select_({x} > {t}(0), {t}(1), "
            f"cutlass.select_({x} < {t}(0), {t}(-1), {t}(0))))",
        )

    def _unary_math(
        self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str
    ) -> None:
        function = _UNARY_MATH_TARGETS[cast("Any", node.target)]
        x = self.scalar_value(info, node.args[0])
        self.emit(meta.cls, f"{name} = cute.math.{function}(cutlass.Float32({x}))")

    def _tanh(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        dtype = _node_dtype(node)
        if dtype is not torch.float32:
            self._pointwise(info, node, meta, name)
            return
        approx = "True" if self.env.settings.fast_math else "False"
        x = self.scalar_value(info, node.args[0])
        self.emit(meta.cls, f"{name} = _cute_flex_tanh(cutlass.Float32({x}), {approx})")

    def _where(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        dtype = _node_dtype(node)
        assert dtype is not None
        t = self.dtype_str(dtype)
        condition, true_value, false_value = (
            self.scalar_value(info, arg) for arg in node.args[:3]
        )
        self.emit(
            meta.cls,
            f"{name} = {t}(cutlass.select_({condition}, {t}({true_value}), "
            f"{t}({false_value})))",
        )

    def _convert(
        self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str
    ) -> None:
        source = node.args[0]
        assert isinstance(source, Node)
        source_dtype = _node_dtype(source)
        dtype = _node_dtype(node)
        assert dtype is not None and source_dtype is not None
        t = self.dtype_str(dtype)
        x = self.value(info, source)
        if source_dtype is torch.bool and dtype is not torch.bool:
            # A predicate has no numeric constructor; select the 0/1 value.
            expr = f"{t}(cutlass.select_({x}, {t}(1), {t}(0)))"
        elif dtype is torch.bool and source_dtype is not torch.bool:
            expr = f"({x} != {self.dtype_str(source_dtype)}(0))"
        else:
            expr = f"{t}({x})"
        self.emit(meta.cls, f"{name} = {expr}")

    def _span_base(self, meta: _NodeInfo) -> str:
        base = self.root("kvbase") if meta.uniform_cls == _TILE else self.root("cbase")
        if meta.offset:
            return f"({base} + cutlass.Int32({meta.offset}))"
        return base

    def _kvcmp(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        # Constant over the span: the comparison of its first lane's
        # ``kv_idx + c`` against zero (analyzer ``_kvcmp``).
        self.emit(
            meta.cls,
            f"{name} = operator.{meta.cmp_op}({self._span_base(meta)}, cutlass.Int32(0))",
        )

    def _kvmod_lane0(self, info: _RegionInfo, node: Node) -> str:
        """``(kv_idx + c) % m`` at the chunk's first lane: the span's
        first-lane remainder (a power-of-two ``m`` is a mask; two's complement
        gives the non-negative Python remainder) plus the chunk's column."""
        key = (info.tag, node)
        if key in self.kvmod_lane0:
            return self.kvmod_lane0[key]
        meta = info.nodes[node]
        name = self.name_for(info, node)
        m = meta.divisor
        base = self._span_base(meta)
        if m & (m - 1) == 0:
            remainder = f"{base} & cutlass.Int32({m - 1})"
        else:
            remainder = f"_cute_python_mod({base}, cutlass.Int32({m}))"
        self.emit(meta.uniform_cls, f"{name}_u = {remainder}")
        lane0 = f"{name}_u"
        if meta.uniform_cls == _TILE:
            lane0 = f"{name}_c"
            self.emit(
                _CHUNK,
                f"{lane0} = {name}_u + cutlass.Int32({self.coord_tensor}[0][1])",
            )
        self.kvmod_lane0[key] = lane0
        return lane0

    def _kvmod(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        lane0 = self._kvmod_lane0(info, node)
        self.emit(_ELEMENT, f"{name} = {lane0} + cutlass.Int32({_ELEM})")

    # -- keep words -----------------------------------------------------------
    #
    # A mask predicate restricted to one chunk is, lane by lane, bit ``j`` of
    # a ``chunk_cols``-bit word.  Comparisons of lane-linear integers
    # (``+-j`` plus lane-uniform terms) give half-open lane intervals from
    # ``_cute_flex_lane_below``; block equalities on ``x // c`` are the same
    # intervals scaled; ``x % m`` against a residue set is a rotated,
    # replicated period pattern; ``abs`` splits into two sides; the
    # connectives are bitwise.  The word is built once per chunk from
    # lane-uniform values and applied through the constexpr bit-test select
    # chain (``lane_bit``), which ptxas turns into R2P + predicated selects.
    # Anything else keeps its per-element value; a connective with one word
    # operand and one per-element operand reads the word's lane bit.

    def full_word(self) -> str:
        width = self.chunk_cols
        return "-1" if width == 32 else str((1 << width) - 1)

    def lane_bit(self, word: str) -> str:
        return f"cutlass.Boolean({word} & (cutlass.Int32(1) << cutlass.Int32({_ELEM})))"

    def new_word(self, tag: str, cls: int, expr: str) -> str:
        """Name an ``Int32`` word expression (once per distinct text)."""
        key = (tag, expr)
        if key in self.word_names:
            return self.word_names[key]
        index = self.word_count.get(tag, 0)
        self.word_count[tag] = index + 1
        name = f"{self.prefix()}{tag}_w{index}"
        self.emit(cls, f"{name} = cutlass.Int32({expr})")
        self.word_names[key] = name
        return name

    def word_expr(self, tag: str, word: _Word) -> str:
        if word.expr is not None:
            return word.expr
        assert word.uniform is not None
        return self.new_word(
            tag,
            word.cls,
            f"cutlass.select_({word.uniform}, cutlass.Int32({self.full_word()}), "
            "cutlass.Int32(0))",
        )

    def word_not(self, tag: str, word: _Word) -> _Word:
        if word.expr is None:
            assert word.uniform is not None
            return _Word(None, f"(~{word.uniform})", word.cls)
        expr = f"~{word.expr}"
        if self.chunk_cols < 32:
            expr = f"{expr} & cutlass.Int32({self.full_word()})"
        return _Word(self.new_word(tag, word.cls, expr), None, word.cls)

    def word_binary(self, tag: str, op: str, a: _Word, b: _Word) -> _Word:
        cls = max(a.cls, b.cls)
        if a.expr is None and b.expr is None:
            return _Word(None, f"({a.uniform} {op} {b.uniform})", cls)
        expr = f"{self.word_expr(tag, a)} {op} {self.word_expr(tag, b)}"
        return _Word(self.new_word(tag, cls, expr), None, cls)

    def below_word(self, tag: str, limit: _Lin) -> _Word:
        """Lanes ``j < limit`` (``limit`` lane-uniform)."""
        expr = f"_cute_flex_lane_below({limit.uniform_expr()}, {self.chunk_cols})"
        return _Word(self.new_word(tag, limit.cls, expr), None, limit.cls)

    def interval_word(self, tag: str, op: str, limit: _Lin) -> _Word:
        """Lanes ``j <op> limit`` (``limit`` lane-uniform)."""
        if op == "lt":
            return self.below_word(tag, limit)
        if op == "le":
            return self.below_word(tag, limit.add(_Lin.constant(1)))
        if op == "ge":
            return self.word_not(tag, self.below_word(tag, limit))
        if op == "gt":
            return self.word_not(tag, self.below_word(tag, limit.add(_Lin.constant(1))))
        equal = self.word_binary(
            tag,
            "&",
            self.below_word(tag, limit.add(_Lin.constant(1))),
            self.word_not(tag, self.below_word(tag, limit)),
        )
        if op == "eq":
            return equal
        assert op == "ne"
        return self.word_not(tag, equal)

    def compare_word(self, tag: str, left: _Lin, right: _Lin, op: str) -> _Word | None:
        """Word of ``left <op> right`` for lane-linear operands."""
        d = left.add(right, -1)
        if d.coef == 0:
            return _Word(
                None, f"operator.{op}({d.uniform_expr()}, cutlass.Int32(0))", d.cls
            )
        if d.coef not in (1, -1):
            return None
        if d.coef == -1:
            d, op = d.scale(-1), _FLIP_COMPARE[op]
        # ``j + u <op> 0``  <=>  ``j <op> -u``
        return self.interval_word(tag, op, dataclasses.replace(d, coef=0).scale(-1))

    def lane_linear(self, info: _RegionInfo, value: object) -> _Lin | None:
        if not isinstance(value, Node):
            constant = _int_constant(value)
            return None if constant is None else _Lin.constant(constant)
        meta = info.nodes[value]
        constant = _int_constant(value)
        if constant is not None and meta.role == "literal":
            return _Lin.constant(constant)
        dtype = _node_dtype(value)
        if dtype is None or not _is_int_dtype(dtype):
            return None
        if _T_LANE not in meta.deps:
            return _Lin.term(f"cutlass.Int32({self.value(info, value)})", meta.cls)
        if meta.role == "kv":
            return _Lin.term(self.root("cbase"), _CHUNK, coef=1)
        if meta.role == "alias":
            return self.lane_linear(info, value.args[0])
        if meta.role == "kvmod":
            return _Lin.term(self._kvmod_lane0(info, value), _CHUNK, coef=1)
        if meta.role == "convert":
            source = value.args[0]
            assert isinstance(source, Node)
            source_dtype = _node_dtype(source)
            if source_dtype is None or not _is_int_dtype(source_dtype):
                return None
            return self.lane_linear(info, source)
        target = value.target
        if target in _ADD_TARGETS or target in _SUB_TARGETS:
            if value.kwargs.get("alpha", 1) != 1:
                return None
            left = self.lane_linear(info, value.args[0])
            right = self.lane_linear(info, value.args[1])
            if left is None or right is None:
                return None
            return left.add(right, 1 if target in _ADD_TARGETS else -1)
        if target in _MUL_TARGETS:
            left = self.lane_linear(info, value.args[0])
            right = self.lane_linear(info, value.args[1])
            if left is None or right is None:
                return None
            if right.coef == 0 and not right.terms:
                return left.scale(right.const)
            if left.coef == 0 and not left.terms:
                return right.scale(left.const)
            return None
        if target is torch.ops.aten.neg.default:
            inner = self.lane_linear(info, value.args[0])
            return None if inner is None else inner.scale(-1)
        return None

    def abs_word(
        self, info: _RegionInfo, tag: str, inner: object, other: object, op: str
    ) -> _Word | None:
        """``abs(inner) <op> other`` with ``other`` lane-uniform: two sides."""
        x = self.lane_linear(info, inner)
        y = self.lane_linear(info, other)
        if x is None or y is None or y.coef != 0:
            return None
        if op in ("lt", "le"):
            # |x| < y  <=>  x < y and x > -y
            a = self.compare_word(tag, x, y, op)
            b = self.compare_word(tag, x, y.scale(-1), _FLIP_COMPARE[op])
            return None if a is None or b is None else self.word_binary(tag, "&", a, b)
        if op in ("gt", "ge"):
            a = self.compare_word(tag, x, y, op)
            b = self.compare_word(tag, x, y.scale(-1), _FLIP_COMPARE[op])
            return None if a is None or b is None else self.word_binary(tag, "|", a, b)
        a = self.compare_word(tag, x, y, "eq")
        b = self.compare_word(tag, x, y.scale(-1), "eq")
        if a is None or b is None:
            return None
        equal = self.word_binary(tag, "|", a, b)
        return equal if op == "eq" else self.word_not(tag, equal)

    def floordiv_word(
        self, info: _RegionInfo, tag: str, x: object, c: object, other: object, op: str
    ) -> _Word | None:
        """``x // c <op> other`` (``c > 0``, ``other`` lane-uniform) as
        half-open bounds on ``x``: ``x // c >= y  <=>  x >= y * c`` and
        ``x // c < y  <=>  x < y * c``."""
        divisor = _int_constant(c)
        if divisor is None or divisor <= 0:
            return None
        lx = self.lane_linear(info, x)
        ly = self.lane_linear(info, other)
        if lx is None or ly is None or ly.coef != 0:
            return None
        low = ly.scale(divisor)
        high = ly.add(_Lin.constant(1)).scale(divisor)
        if op == "ge":
            return self.compare_word(tag, lx, low, "ge")
        if op == "lt":
            return self.compare_word(tag, lx, low, "lt")
        if op == "le":
            return self.compare_word(tag, lx, high, "lt")
        if op == "gt":
            return self.compare_word(tag, lx, high, "ge")
        a = self.compare_word(tag, lx, low, "ge")
        b = self.compare_word(tag, lx, high, "lt")
        if a is None or b is None:
            return None
        equal = self.word_binary(tag, "&", a, b)
        return equal if op == "eq" else self.word_not(tag, equal)

    def mod_word(
        self, info: _RegionInfo, tag: str, node: Node, other: object, op: str
    ) -> _Word | None:
        """``x % m <op> other`` with ``m < chunk_cols``: the residue pattern of
        one period, rotated by the first lane's residue and replicated over
        the chunk.  ``abs(x) % m`` against zero is ``x % m`` against zero."""
        x, m_arg = node.args[0], node.args[1]
        m = _int_constant(m_arg)
        width = self.chunk_cols
        if m is None or m <= 0 or m >= width or not isinstance(x, Node):
            return None
        y = self.lane_linear(info, other)
        if y is None or y.coef != 0:
            return None
        y_static = y.const if not y.terms else None
        if x.target is _ABS_TARGET:
            if op not in ("eq", "ne") or y_static != 0:
                return None
            x = x.args[0]
        lx = self.lane_linear(info, x)
        if lx is None or lx.coef not in (1, -1):
            return None
        period_mask = (1 << m) - 1
        # Residues ``r in [0, m)`` with ``r <op> y``: a static word when ``y``
        # is a constant, else built from the period's ``below`` words.
        pattern: int | str
        if y_static is not None:
            pattern = sum(
                1 << r for r in range(m) if getattr(operator, op)(r, y_static)
            )
        elif lx.coef == -1:
            return None
        else:
            below = f"_cute_flex_lane_below({{}}, {m})"
            y_expr = y.uniform_expr()
            y1_expr = y.add(_Lin.constant(1)).uniform_expr()
            if op == "lt":
                pattern = below.format(y_expr)
            elif op == "le":
                pattern = below.format(y1_expr)
            elif op == "ge":
                pattern = f"~{below.format(y_expr)} & cutlass.Int32({period_mask})"
            elif op == "gt":
                pattern = f"~{below.format(y1_expr)} & cutlass.Int32({period_mask})"
            else:
                equal = f"{below.format(y1_expr)} & ~{below.format(y_expr)}"
                pattern = (
                    equal
                    if op == "eq"
                    else f"~({equal}) & cutlass.Int32({period_mask})"
                )
        # The first lane's residue ``u0 = u % m``; terms that are multiples
        # of the chunk width (the chunk base) vanish when ``m`` divides it.
        uniform = dataclasses.replace(lx, coef=0)
        if width % m == 0:
            cbase = self.root("cbase")
            uniform = _Lin(
                0, tuple(t for t in uniform.terms if t[0] != cbase), uniform.const
            )
        if lx.coef == -1:
            # Lanes hold ``u - j``: residue ``r`` sits at ``j = u0 - r``, the
            # reflected pattern rotated left.
            assert isinstance(pattern, int)
            pattern = sum(1 << ((-r) % m) for r in range(m) if pattern & (1 << r))
        cls = uniform.cls
        if not uniform.terms:
            u0: int | str = uniform.const % m
        elif m & (m - 1) == 0:
            u0 = f"({uniform.uniform_expr()}) & cutlass.Int32({m - 1})"
        else:
            u0 = f"_cute_python_mod({uniform.uniform_expr()}, cutlass.Int32({m}))"
        rotate_left = lx.coef == -1
        if isinstance(pattern, int) and isinstance(u0, int):
            shift = u0 if rotate_left else (m - u0) % m
            rotated: int | str = (
                (pattern << shift) | (pattern >> (m - shift))
            ) & period_mask
        elif u0 == 0:
            assert isinstance(pattern, str)
            rotated = self.new_word(tag, cls, pattern)
        else:
            if isinstance(u0, int):
                u0_expr = f"cutlass.Int32({u0})"
                back_expr = f"cutlass.Int32({m - u0})"
            else:
                u0_name = self.new_word(tag, cls, u0)
                u0_expr = u0_name
                back_expr = f"(cutlass.Int32({m}) - {u0_name})"
            p = (
                f"cutlass.Int32({pattern})"
                if isinstance(pattern, int)
                else self.new_word(tag, cls, pattern)
            )
            if rotate_left:
                rotated = f"(({p} << {u0_expr}) | ({p} >> {back_expr})) & cutlass.Int32({period_mask})"
            else:
                rotated = f"(({p} >> {u0_expr}) | ({p} << {back_expr})) & cutlass.Int32({period_mask})"
        replicate = sum(1 << (k * m) for k in range(-(-width // m)))
        if isinstance(rotated, int):
            word_value = (rotated * replicate) & ((1 << width) - 1)
            if width == 32 and word_value >= 1 << 31:
                word_value -= 1 << 32
            return _Word(self.new_word(tag, _ITEM, str(word_value)), None, _ITEM)
        if replicate >= 1 << 31:
            replicate -= 1 << 32
        expr = f"({rotated}) * cutlass.Int32({replicate})"
        if width < 32:
            expr = f"({expr}) & cutlass.Int32({self.full_word()})"
        return _Word(self.new_word(tag, cls, expr), None, cls)

    def predicate_word(self, info: _RegionInfo, tag: str, node: Node) -> _Word | None:
        op = _COMPARE_OPERATORS[cast("Any", node.target)]
        x, y = node.args[0], node.args[1]
        x_dtype = _node_dtype(x) if isinstance(x, Node) else None
        y_dtype = _node_dtype(y) if isinstance(y, Node) else None
        if x_dtype is torch.bool and y_dtype is torch.bool and op in ("eq", "ne"):
            assert isinstance(x, Node) and isinstance(y, Node)
            a = self.mask_word(info, x)
            b = self.mask_word(info, y)
            if a is None or b is None:
                return None
            different = self.word_binary(tag, "^", a, b)
            return different if op == "ne" else self.word_not(tag, different)
        if x_dtype is torch.bool or y_dtype is torch.bool:
            return None
        for lhs, rhs, side_op in ((x, y, op), (y, x, _FLIP_COMPARE[op])):
            if not isinstance(lhs, Node) or _T_LANE not in info.nodes[lhs].deps:
                continue
            if lhs.target is _ABS_TARGET:
                return self.abs_word(info, tag, lhs.args[0], rhs, side_op)
            if (operands := _floordiv_operands(lhs)) is not None:
                return self.floordiv_word(
                    info, tag, operands[0], operands[1], rhs, side_op
                )
            if info.nodes[lhs].role == "pymod":
                return self.mod_word(info, tag, lhs, rhs, side_op)
        lx = self.lane_linear(info, x)
        ly = self.lane_linear(info, y)
        if lx is None or ly is None:
            return None
        return self.compare_word(tag, lx, ly, op)

    def mask_word(self, info: _RegionInfo, node: Node) -> _Word | None:
        """The keep word of a predicate node, or ``None``."""
        key = (info.tag, node)
        if key in self.words:
            return self.words[key]
        self.words[key] = None
        word = self._mask_word(info, node)
        self.words[key] = word
        return word

    def _mask_word(self, info: _RegionInfo, node: Node) -> _Word | None:
        meta = info.nodes[node]
        tag = info.tag
        if _node_dtype(node) is not torch.bool:
            return None
        if _T_LANE not in meta.deps:
            return _Word(None, self.value(info, node), meta.cls)
        if meta.role == "alias":
            return self.mask_word(info, cast("Node", node.args[0]))
        if meta.role == "convert":
            source = node.args[0]
            assert isinstance(source, Node)
            if _node_dtype(source) is not torch.bool:
                return None
            return self.mask_word(info, source)
        target = node.target
        if target in _BOOL_NOT_TARGETS:
            inner = self.mask_word(info, cast("Node", node.args[0]))
            return None if inner is None else self.word_not(tag, inner)
        for targets, op in (
            (_BOOL_AND_TARGETS, "&"),
            (_BOOL_OR_TARGETS, "|"),
            (_BOOL_XOR_TARGETS, "^"),
        ):
            if target in targets:
                words = [
                    self.mask_word(info, arg) if isinstance(arg, Node) else None
                    for arg in node.args[:2]
                ]
                if words[0] is None or words[1] is None:
                    return None
                return self.word_binary(tag, op, words[0], words[1])
        if target in _WHERE_TARGETS:
            if not all(isinstance(arg, Node) for arg in node.args[:3]):
                return None
            c, a, b = (self.mask_word(info, cast("Node", arg)) for arg in node.args[:3])
            if c is None or a is None or b is None:
                return None
            return self.word_binary(
                tag,
                "|",
                self.word_binary(tag, "&", c, a),
                self.word_binary(tag, "&", self.word_not(tag, c), b),
            )
        if target in _COMPARE_OPERATORS:
            return self.predicate_word(info, tag, node)
        return None

    def _pointwise(
        self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str
    ) -> None:
        from ..aten_lowering import LoweringContext
        from ..inductor_lowering import PointwiseLowering

        lowering = node.meta.get("lowering")
        if not isinstance(lowering, PointwiseLowering):
            operator_name = _COMPARE_OPERATORS.get(cast("Any", node.target))
            if operator_name is not None:
                left = self.scalar_value(info, node.args[0])
                right = self.scalar_value(info, node.args[1])
                self.emit(
                    meta.cls, f"{name} = operator.{operator_name}({left}, {right})"
                )
                return
            raise _unsupported(
                f"{node.name} ({node.target}) has no pointwise lowering "
                f"({type(lowering).__name__})"
            )
        inputs = _lowering_inputs(node)
        if len(inputs) != len(lowering.input_names):
            raise _unsupported(
                f"{node.name}: {len(inputs)} inputs for lowering names "
                f"{lowering.input_names}"
            )
        input_asts = [expr_from_string(self.value(info, value)) for value in inputs]
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = self.codegen
        ctx.env = {}
        statements: list[ast.AST] = []
        with self.codegen.set_statements(statements), V.set_current_node(node):
            result = lowering.codegen_from_input_asts(ctx, node, input_asts)
        if not isinstance(result, ast.AST):
            raise _unsupported(f"{node.name}: invalid pointwise result {result!r}")
        statements, result = _rename_temporaries(statements, result, name)
        last = statements[-1] if statements else None
        if (
            isinstance(result, ast.Name)
            and isinstance(last, ast.Assign)
            and len(last.targets) == 1
            and isinstance(last.targets[0], ast.Name)
            and last.targets[0].id == result.id
        ):
            # Bind the node's name directly instead of ``v_N`` then an alias.
            for statement in statements[:-1]:
                self.emit(meta.cls, ast.unparse(statement))
            self.emit(meta.cls, f"{name} = {ast.unparse(last.value)}")
            return
        for statement in statements:
            self.emit(meta.cls, ast.unparse(statement))
        self.emit(meta.cls, f"{name} = {ast.unparse(result)}")

    # -- loads --------------------------------------------------------------

    def _load(self, info: _RegionInfo, node: Node, meta: _NodeInfo, name: str) -> None:
        buffer = self.plan.aux_buffers[meta.aux_index]
        tensor = f"_flash_aux{meta.aux_index}"
        t = self.dtype_str(buffer.dtype)
        zero = f"{t}(False)" if buffer.dtype is torch.bool else f"{t}(0)"
        indices = cast("Sequence[object]", node.args[1])
        proven = meta.in_bounds
        width = self.vector_width if meta.vector_width and meta.cls == _ELEMENT else 0
        window = meta.window if meta.cls == _ELEMENT and not width else 0
        if width:
            load_cls = _GROUP
        elif window:
            load_cls = _CHUNK
        else:
            load_cls = meta.cls
        last_dim = len(indices) - 1
        guards: list[str] = []
        offsets: list[str] = []
        for dim, index in enumerate(indices):
            if (width or window) and dim == last_dim:
                break
            size, stride = buffer.shape[dim], buffer.strides[dim]
            index_var = f"{name}_i{dim}"
            if isinstance(index, Node):
                # The cast lives where the index value does (an item-invariant
                # row index is cast once per item, not once per lane).
                dim_cls = min(info.nodes[index].cls, load_cls)
                self.emit(
                    dim_cls, f"{index_var} = cutlass.Int32({self.value(info, index)})"
                )
            else:
                constant = _int_constant(index)
                assert constant is not None
                self.emit(_ITEM, f"{index_var} = cutlass.Int32({constant})")
            if not proven:
                guards.append(
                    f"({index_var} >= cutlass.Int32(0)) & ({index_var} < cutlass.Int32({size}))"
                )
            if stride == 0:
                continue
            if proven:
                # Every index is inside the table: no clamp.
                offsets.append(f"{index_var} * cutlass.Int32({stride})")
                continue
            clamped = (
                f"cutlass.min(cutlass.max({index_var}, cutlass.Int32(0)), "
                f"cutlass.Int32({size - 1}))"
            )
            offsets.append(f"{clamped} * cutlass.Int32({stride})")
        if not width and not window:
            pointer = f"{tensor}.iterator" + "".join(
                f" + {offset}" for offset in offsets
            )
            load = f"{t}(({pointer}).load())"
            if proven:
                self.emit(load_cls, f"{name} = {load}")
                return
            guard = " & ".join(guards)
            self.emit(load_cls, f"{name}_ok = {guard}")
            self.emit(
                load_cls, f"{name} = {t}(cutlass.select_({name}_ok, {load}, {zero}))"
            )
            return
        if buffer.dtype is torch.bool:
            # A bool table is loaded as bytes: ``cutlass.Boolean``'s MLIR type
            # is ``i1`` and a ``vector<16xi1>`` load traps the DSL's lowering
            # (the host process dies with SIGILL); each lane is ``!= 0``.
            vec_t = "cutlass.Int8"
        else:
            vec_t = t

        def lane_value(value: str) -> str:
            if buffer.dtype is torch.bool:
                return f"{vec_t}({value}) != {vec_t}(0)"
            return f"{t}({value})"

        last = cast("Node", indices[last_dim])
        affine = info.nodes[last].affine
        assert affine is not None
        size = buffer.shape[last_dim]
        pointer = f"{tensor}.iterator" + "".join(f" + {o}" for o in offsets)
        if window:
            self._window_load(
                name,
                pointer,
                affine,
                sign=meta.window_sign,
                lanes=window,
                size=size,
                vec_t=vec_t,
                lane_value=lane_value,
            )
            return
        # Lane-contiguous last index ``kv_idx + c``: one vector load per group
        # of ``width`` lanes, selected per lane.  ``gbase`` is the group's
        # first KV index; the group never straddles the table end because
        # ``shape[-1] % width == 0`` and the base is ``width``-aligned.
        lo = f"{name}_lo"
        offset = f" + cutlass.Int32({affine.const})" if affine.const else ""
        self.emit(_GROUP, f"{lo} = {self.root('gbase')}{offset}")
        lane = lane_value(f"{name}_vec[{_LANE_VAR}]")
        if proven:
            self.emit(
                _GROUP,
                f"{name}_vec = _cute_flex_load_vec({pointer} + {lo}, {width}, {vec_t})",
            )
            self.emit(_ELEMENT, f"{name} = {t}({lane})")
            return
        guards.append(f"({lo} >= cutlass.Int32(0)) & ({lo} < cutlass.Int32({size}))")
        clamped = f"cutlass.min(cutlass.max({lo}, cutlass.Int32(0)), cutlass.Int32({size - width}))"
        self.emit(_GROUP, f"{name}_ok = {' & '.join(guards)}")
        self.emit(
            _GROUP,
            f"{name}_vec = _cute_flex_load_vec({pointer} + {clamped}, {width}, {vec_t})",
        )
        self.emit(_ELEMENT, f"{name} = {t}(cutlass.select_({name}_ok, {lane}, {zero}))")

    def _window_load(
        self,
        name: str,
        pointer: str,
        affine: _Affine,
        *,
        sign: int,
        lanes: int,
        size: int,
        vec_t: str,
        lane_value: Callable[[str], str],
    ) -> None:
        """Lane-affine last index ``+-kv_idx + rest`` proven inside the table.

        The thread's ``chunk_cols`` entries of a chunk are consecutive (rising
        with the column for ``+``, falling for ``-``) but start at a
        row-dependent, unaligned entry.  Per chunk: load them from the
        ``lanes``-aligned base at or below the first entry as
        ``chunk_cols // lanes + 1`` vectors; the last vector is only read
        when the misalignment ``m`` is nonzero, in which case it lies inside
        the table, so its address is clamped to the table end instead of
        guarded.  Per element: pick the lane ``m + slot`` with ``lanes - 1``
        selects.  The same entries as the gather, bitwise the same values.
        """
        positive: list[str] = []
        negative: list[str] = []
        for root, coef in affine.coeffs:
            if root == "kv":
                base = self.root("cbase")
                if sign < 0:
                    base = f"({base} + cutlass.Int32({self.chunk_cols - 1}))"
                (negative if coef < 0 else positive).append(base)
            elif coef in (1, -1):
                (negative if coef < 0 else positive).append(self.root(root))
            else:
                positive.append(f"{self.root(root)} * cutlass.Int32({coef})")
        constant = f"cutlass.Int32({affine.const})"
        terms = [*positive, constant] if positive else [constant]
        first = " + ".join(terms) + "".join(f" - {term}" for term in negative)
        lo = f"{name}_lo"
        base = f"{name}_wb"
        self.emit(_CHUNK, f"{lo} = {first}")
        self.emit(_CHUNK, f"{base} = {lo} & cutlass.Int32({-lanes})")
        self.emit(_CHUNK, f"{name}_m = {lo} - {base}")
        bits = lanes.bit_length() - 1
        for bit in range(bits):
            self.emit(
                _CHUNK,
                f"{name}_m{bit} = ({name}_m & cutlass.Int32({1 << bit})) != cutlass.Int32(0)",
            )
        count = self.chunk_cols // lanes + 1
        last = f"cutlass.min({base} + cutlass.Int32({self.chunk_cols}), cutlass.Int32({size - lanes}))"
        self.emit(
            _CHUNK,
            f"{name}_win = _cute_flex_load_window({pointer} + {base}, "
            f"{pointer} + {last}, {count}, {lanes}, {vec_t})",
        )
        slot = _ELEM if sign > 0 else f"{self.chunk_cols - 1} - {_ELEM}"

        def tree(low: int, bit: int) -> str:
            if bit < 0:
                return f"{name}_win[{slot} + {low}]"
            return (
                f"cutlass.select_({name}_m{bit}, {tree(low + (1 << bit), bit - 1)}, "
                f"{tree(low, bit - 1)})"
            )

        self.emit(_ELEMENT, f"{name} = {lane_value(tree(0, bits - 1))}")

    # -- assembly -----------------------------------------------------------

    def render_region(self, info: _RegionInfo, *, keep_word: bool = False) -> str:
        """Render the region; its output's expression.

        With ``keep_word`` (a mask region in the bitmask form) the output is
        rendered through the keep-word peephole: a lane word (returned as the
        ``lane_bit`` Boolean of the element) when the whole predicate folds,
        else the per-element value whose foldable operands read lane bits.
        """
        if keep_word:
            word = self.mask_word(info, info.output)
            if word is not None and word.expr is not None:
                return self.lane_bit(word.expr)
            return self.value(info, info.output)
        for node in info.region.graph.nodes:
            if node.op != "call_function":
                continue
            if node is info.output:
                continue
            meta = info.nodes[node]
            if meta.role in ("alias", "literal", "aux"):
                continue
            self.value(info, node)
        return self.value(info, info.output)


def _indent_lines(lines: list[str], indent: int) -> list[str]:
    pad = " " * indent
    return [pad + line for line in lines]


def _join_lines(lines: list[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


class _RenameNames(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__()
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        renamed = self.mapping.get(node.id)
        if renamed is None:
            return node
        return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)


def _rename_temporaries(
    statements: list[ast.AST], result: ast.AST, prefix: str
) -> tuple[list[ast.AST], ast.AST]:
    """Rename the lifted ``v_N`` temporaries to ``{prefix}_t{k}``.

    The hook text then depends only on (stage, region, node), so the emitter
    can deduplicate the prelude / prologue lines of a stage's two hooks by
    exact text instead of computing shared values twice.
    """
    mapping: dict[str, str] = {}
    for statement in statements:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id not in mapping:
                    mapping[target.id] = f"{prefix}_t{len(mapping)}"
    if not mapping:
        return statements, result
    transformer = _RenameNames(mapping)
    return [transformer.visit(statement) for statement in statements], (
        transformer.visit(result)
    )


def render_flex_hook(
    plan: AttentionScorePlan,
    *,
    stage: int,
    score_tensor: str,
    coord_tensor: str,
    m_tile_expr: str,
    kv_tile_expr: str,
    chunk_expr: str,
    chunk_cols: int,
    indent: int,
    with_score: bool,
    with_mask: bool,
    tail_mask: bool,
    row_coord_expr: str | None = None,
    item_indent: int | None = None,
    tile_indent: int | None = None,
    device_function: DeviceFunction | None = None,
    coord_bounds: Sequence[int | None] | None = None,
    mask_form: str = "bitmask",
) -> FlexHookCode:
    """Render the flex hook of ``stage`` for one PASS1 chunk.

    ``mask_form`` selects how the mask region's predicate reaches the
    ``-inf`` select: ``"bitmask"`` folds it into a per-chunk keep word where
    the peephole applies (``_HookEmitter.mask_word``) and tests the lane bit
    per element; ``"element"`` evaluates it per element.

    Every ``FlexHookCode`` field is complete newline-terminated lines; the
    emitter dedents and re-indents them (``indent`` is the chunk body's
    indentation, the prelude / prologue default to ``indent - 8`` /
    ``indent - 4``).  ``row_coord_expr`` is the thread's row within the
    128-row tile as an expression valid outside the chunk loop; by default
    the chunk index inside ``coord_tensor`` is replaced by ``0``
    (``tLDcS[None, flash_ci, None, None]`` -> ``tLDcS[None, 0, None, None][0][0]``).

    The whole hook text is deterministic per (stage, region, node) -- bound
    names and temporaries alike -- and placement (item / tile / chunk /
    element) is decided over all of the plan's regions regardless of the
    flags, so the two hooks of a stage (masked and unmasked) emit identical
    prelude / prologue lines for shared values and the emitter can
    deduplicate them by exact text.
    """
    if device_function is None:
        device_function = DeviceFunction.current()
    regions: list[tuple[str, FlexRegion]] = []
    if plan.score_region is not None:
        regions.append(("s", plan.score_region))
    if plan.mask_region is not None:
        regions.append(("m", plan.mask_region))
    emit_tags = {tag for tag, flag in (("s", with_score), ("m", with_mask)) if flag}
    for _tag, region in regions:
        _ensure_lowerings(region.graph)
    analyzer = _Analyzer(plan, chunk_cols, coord_bounds)
    infos = [analyzer.analyze(region, tag) for tag, region in regions]
    _demote_item_values(infos, ITEM_LIVE_CAP)
    infos = [info for info in infos if info.tag in emit_tags]
    if row_coord_expr is None:
        base = re.sub(rf"\b{re.escape(chunk_expr)}\b", "0", coord_tensor)
        row_coord_expr = f"{base}[0][0]"
    emitter = _HookEmitter(
        plan,
        stage=stage,
        score_tensor=score_tensor,
        coord_tensor=coord_tensor,
        m_tile_expr=m_tile_expr,
        kv_tile_expr=kv_tile_expr,
        chunk_expr=chunk_expr,
        chunk_cols=chunk_cols,
        row_coord_expr=row_coord_expr,
        device_function=device_function,
    )
    widths = [
        meta.vector_width
        for info in infos
        for meta in info.nodes.values()
        if meta.role == "load" and meta.vector_width and meta.cls == _ELEMENT
    ]
    if widths:
        emitter.vector_width = min(widths)
        if chunk_cols % emitter.vector_width != 0:
            emitter.vector_width = 0
    if mask_form not in MASK_FORMS:
        raise _unsupported(f"unknown mask form {mask_form!r}")
    outputs = {
        info.tag: emitter.render_region(
            info, keep_word=info.tag == "m" and mask_form == "bitmask"
        )
        for info in infos
    }

    store: str | None = None
    score_out = outputs.get("s")
    keep = outputs.get("m")
    current = f"cutlass.Float32({score_tensor}[{_ELEM}])"
    if score_out is not None and keep is not None:
        store = (
            f"{score_tensor}[{_ELEM}] = cutlass.Float32(cutlass.select_({keep}, "
            f"cutlass.Float32({score_out}), -cutlass.Float32.inf))"
        )
    elif score_out is not None:
        store = f"{score_tensor}[{_ELEM}] = cutlass.Float32({score_out})"
    elif keep is not None:
        store = (
            f"{score_tensor}[{_ELEM}] = cutlass.Float32(cutlass.select_({keep}, "
            f"{current}, -cutlass.Float32.inf))"
        )

    chunk_lines: list[str] = list(emitter.blocks[_CHUNK])
    element_lines = list(emitter.blocks[_ELEMENT])
    if store is not None:
        element_lines.append(store)
    if element_lines or emitter.blocks[_GROUP]:
        if emitter.vector_width:
            width = emitter.vector_width
            chunk_lines.append(
                f"for {_GROUP_VAR} in cutlass.range_constexpr("
                f"cute.size({score_tensor}) // {width}):"
            )
            chunk_lines.extend(_indent_lines(emitter.blocks[_GROUP], 4))
            chunk_lines.extend(
                (
                    f"    for {_LANE_VAR} in cutlass.range_constexpr({width}):",
                    f"        {_ELEM} = {_GROUP_VAR} * {width} + {_LANE_VAR}",
                )
            )
            chunk_lines.extend(_indent_lines(element_lines, 8))
        else:
            chunk_lines.append(
                f"for {_ELEM} in cutlass.range_constexpr(cute.size({score_tensor})):"
            )
            chunk_lines.extend(_indent_lines(element_lines, 4))
    if tail_mask:
        # After the score / mask selects: a score_mod may map the zero-filled
        # tail columns to finite values and a mask_mod may keep them.
        chunk_lines.append(
            f"_helion_flash_rt.mask_r2p_sm100_rank1({score_tensor}, "
            f"_flash_kv_tail_cols - {chunk_expr} * {chunk_cols})"
        )

    item_pad = indent - 8 if item_indent is None else item_indent
    tile_pad = indent - 4 if tile_indent is None else tile_indent
    return FlexHookCode(
        _join_lines(_indent_lines(emitter.blocks[_ITEM], max(item_pad, 0))),
        _join_lines(_indent_lines(emitter.blocks[_TILE], max(tile_pad, 0))),
        _join_lines(_indent_lines(chunk_lines, indent)),
    )


def _scalar_kind(value: object) -> str:
    if isinstance(value, torch.SymBool):
        return "bool"
    if isinstance(value, torch.SymFloat):
        return "float32"
    return "int32"


def flex_hook_params(plan: AttentionScorePlan) -> list[tuple[str, str]]:
    """``(kernel parameter name, kind)`` of the hook's aux tensors and scalars.

    Aux tensors are ``_flash_aux{i}`` (``"tensor"``) in plan order; host
    scalars are ``_flash_hs{j}`` (``"int32"`` / ``"float32"`` / ``"bool"``)
    in plan order, except the runtime scale (``plan.scale_arg``), which the
    emitter already provides as ``_flash_scale``.
    """
    params: list[tuple[str, str]] = [
        (f"_flash_aux{i}", "tensor") for i in range(len(plan.aux_buffers))
    ]
    regions = [r for r in (plan.score_region, plan.mask_region) if r is not None]
    for j, name in enumerate(plan.host_scalars):
        if name == plan.scale_arg:
            continue
        slot = 5 + len(plan.aux_buffers) + j
        kind = "int32"
        for region in regions:
            placeholders = [n for n in region.graph.nodes if n.op == "placeholder"]
            if slot < len(placeholders):
                kind = _scalar_kind(placeholders[slot].meta.get("val"))
                break
        params.append((f"_flash_hs{j}", kind))
    return params


def _ensure_lowerings(graph: torch.fx.Graph) -> None:
    from ..inductor_lowering import prepare_graph_lowerings

    if all(
        "lowering" in node.meta for node in graph.nodes if node.op == "call_function"
    ):
        return
    prepare_graph_lowerings(graph)


# -- region construction (tests, harness) -------------------------------------


def classify_aux_index(
    plan_regions: Sequence[FlexRegion],
    aux_buffers: Sequence[FlexAuxBuffer],
    host_scalars: Sequence[str],
    chunk_cols: int = 32,
) -> list[str]:
    """``FlexAuxBuffer.index_class`` of every aux buffer from its loads."""
    from .attention_plan import AttentionScorePlan

    plan = AttentionScorePlan(
        head_dim=64,
        qk_scale_log2=1.0,
        aux_buffers=tuple(aux_buffers),
        host_scalars=tuple(host_scalars),
    )
    analyzer = _Analyzer(plan, chunk_cols)
    classes: list[set[str]] = [set() for _ in aux_buffers]
    for region in plan_regions:
        info = analyzer.analyze(region, "x")
        for node, meta in info.nodes.items():
            if meta.role != "load":
                continue
            classes[meta.aux_index].add(_load_index_class(info, node, meta))
    return [next(iter(found)) if len(found) == 1 else "general" for found in classes]


def _load_index_class(info: _RegionInfo, node: Node, meta: _NodeInfo) -> str:
    indices = cast("Sequence[object]", node.args[1])
    index_nodes = [index for index in indices if isinstance(index, Node)]
    if any(info.nodes[index].role == "load" for index in index_nodes) or any(
        _ancestor_load(info, index) for index in index_nodes
    ):
        return "value"
    if not (_T_LANE in meta.deps or _T_KVT in meta.deps):
        return "row_uniform" if _T_Q in meta.deps else "head"
    last = indices[-1] if indices else None
    affine = info.nodes[last].affine if isinstance(last, Node) else None
    others_lane_free = all(
        _T_LANE not in info.nodes[index].deps for index in index_nodes[:-1]
    )
    if affine is not None and others_lane_free:
        if affine.is_kv_plus_const():
            return "column_affine"
        if affine.coeffs == (("kv", -1), ("q", 1)):
            return "diagonal"
    if _T_Q in meta.deps and others_lane_free:
        return "row_col"
    return "general"


def _ancestor_load(info: _RegionInfo, node: Node) -> bool:
    seen: set[Node] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in seen or current not in info.nodes:
            continue
        seen.add(current)
        if info.nodes[current].role == "load":
            return True
        stack.extend(arg for arg in current.all_input_nodes)
    return False


def build_flex_region(
    fn: Callable[..., object],
    *,
    aux_specs: Sequence[FlexAuxBuffer | tuple[str, torch.dtype, Sequence[int]]] = (),
    host_scalar_specs: Sequence[tuple[str, type]] = (),
    chunk_cols: int = 32,
) -> tuple[FlexRegion, tuple[FlexAuxBuffer, ...], tuple[str, ...]]:
    """Trace ``fn(score, b, h, q_idx, kv_idx, *aux, *scalars)`` into a region.

    The arguments are fake tensors with the FlexAttention broadcast shapes
    (``[128, 128]`` fp32 score, ``[1, 1]`` ``b`` / ``h``, ``[128, 1]`` q,
    ``[1, 128]`` kv, all int32), the aux tensors (``FlexAuxBuffer`` or
    ``(name, dtype, shape)``; contiguous strides) and unbacked
    ``SymInt`` / ``SymFloat`` host scalars.  Table reads must use ``hl.load``.
    Requires an active ``CompileEnvironment``; the returned graph has the
    device IR's node targets, ``meta["val"]``, ``meta["location"]`` and
    lowerings assigned.  The aux buffers come back with their ``index_class``.
    """
    from ..device_ir import _get_custom_decomp_table

    env = CompileEnvironment.current()
    buffers: list[FlexAuxBuffer] = []
    aux_tensors: list[torch.Tensor] = []
    for spec in aux_specs:
        if isinstance(spec, FlexAuxBuffer):
            tensor = torch.empty_strided(spec.shape, spec.strides, dtype=spec.dtype)
            buffers.append(spec)
        else:
            name, dtype, shape = spec
            tensor = torch.empty(tuple(shape), dtype=dtype)
            buffers.append(
                FlexAuxBuffer(name, dtype, tuple(tensor.shape), tuple(tensor.stride()))
            )
        aux_tensors.append(tensor)
    scalars: list[object] = []
    scalar_names: list[str] = []
    for name, kind in host_scalar_specs:
        scalar_names.append(name)
        if kind is float:
            scalars.append(env.shape_env.create_unbacked_symfloat())
        elif kind is int:
            scalars.append(env.shape_env.create_unbacked_symint())
        else:
            raise _unsupported(f"host scalar {name}: unsupported kind {kind!r}")
    score = torch.empty(128, 128, dtype=torch.float32)
    b = torch.empty(1, 1, dtype=torch.int32)
    h = torch.empty(1, 1, dtype=torch.int32)
    q_idx = torch.empty(128, 1, dtype=torch.int32)
    kv_idx = torch.empty(1, 128, dtype=torch.int32)
    code = fn.__code__
    location = SourceLocation(
        code.co_firstlineno, 0, code.co_firstlineno, 0, fn.__name__, code.co_filename
    )
    with (
        preserve_node_meta(),
        patch.object(
            torch.fx.proxy,
            "_COPY_META_FIELDS",
            [*torch.fx.proxy._COPY_META_FIELDS, "location"],
        ),
        location,
    ):
        module = proxy_tensor.make_fx(
            fn, decomposition_table=_get_custom_decomp_table()
        )(score, b, h, q_idx, kv_idx, *aux_tensors, *scalars)
    graph = module.graph
    output_node = next(node for node in graph.nodes if node.op == "output")
    result = output_node.args[0]
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise _unsupported("a region must return one value")
        result = result[0]
    if not isinstance(result, Node):
        raise _unsupported(f"a region must return a tensor, got {result!r}")
    for node in graph.nodes:
        if node.op == "call_function" and "location" not in node.meta:
            node.meta["location"] = location
    _ensure_lowerings(graph)
    region = FlexRegion(graph, result)
    classes = classify_aux_index([region], buffers, scalar_names, chunk_cols)
    classified = tuple(
        dataclasses.replace(buffer, index_class=index_class)
        for buffer, index_class in zip(buffers, classes, strict=True)
    )
    return region, classified, tuple(scalar_names)


def region_signature(fn: Callable[..., object]) -> list[str]:
    """Parameter names of a region function (for harness plumbing)."""
    return list(inspect.signature(fn).parameters)
