"""Pair the exponentials, exact divisions and fp32 arithmetic of adjacent
V-lanes into Blackwell's packed fp32 instructions, bit for bit.

A vectorized lane loop (``for v in cutlass.range_constexpr(V)``) computes
every element of its packet with the same statements.  Blackwell issues two
fp32 FMAs, adds or multiplies as one instruction (``FFMA2``, ``FADD2``,
``FMUL2``), but NVVM leaves the scalar code the backend generates scalar.  An
fp32 ``exp`` is eight instructions and the exact-division fast path three;
two lanes at once take twelve and three.

Every lane's result stays bit for bit what the scalar code computes:

- ``cute.math.exp(x)`` lowers to libdevice's ``__nv_expf``;
  ``_cute_exp_f32x2_parts`` (``device.ptx``) runs that sequence on both lanes
  with its FMAs and its subtraction packed, each lane rounded as the scalar
  instruction rounds it, and returns each lane's ex2 result ``p`` and scale
  factor ``s``.  The product ``p * s`` stays a plain multiply in the lane's
  code, so the compiler contracts it into a following add (a running sum's
  ``FFMA``) exactly as it contracts the product inside the scalar exp.
- ``_cute_exact_div_fast_f32(a, b, y)`` (``exact_division``) becomes
  ``_cute_exact_div_fast_f32x2`` when ``b`` and ``y`` are loop invariants.
- An fp32 product, sum or difference and an explicit ``cute.math.fma``
  become ``cute.arch.{mul,add,sub,fma}_packed_f32x2``, which round each lane
  once to nearest as the scalar instruction does.  What could differ is FMA
  contraction: NVVM fuses a scalar product into the sum that reads it, never
  a packed ``.rn`` operation.  So a product pairs only when no sum, no
  difference and no call that could inline one reads it (also through
  ``* 1``, ``/ 1`` or their -1 forms, which NVVM folds away), and a sum or a
  difference only when no operand may be a product (a ``*``, a call other
  than a load, a conversion, an extremum, an fma or a sum, a quotient by a
  constant, a select of either, or a value of unknown type or origin): the
  scalar forms of the paired operations had nothing to contract with, and
  the operations left scalar keep their partners.  An element of a register
  fragment or list may be a product when the kernel puts one into it (read
  in the store's own block) or uses it in any way that could write it
  unseen, through an alias or a view included; a loaded packet's element
  is none.  A sum or difference of
  a widened bf16 lane stays scalar too, for speed: ptxas reads the bf16 half
  of a register straight into a scalar add (``FHADD.BF16``), while a packed
  add needs both lanes widened first.

How the lanes interleave: the rewritten loop runs ``V // 2`` trips, each the
body for lane ``2p`` (``a``), then for lane ``2p + 1`` (``b``) with ``b``'s
local names renamed.  At each pair, lane ``b``'s paired call and the
statements that compute its operands move up to lane ``a``'s call.  They must
be pure register computations of loop invariants, the lane index and ``b``'s
own earlier locals that index only fragments, packets and lists the body
cannot write (no store, append or call reaches them, under their own name or
another), so running them before the rest of lane ``a`` changes nothing.  A
name read before it is assigned in the body (a running sum or extremum) or
read after the loop keeps its name in both lanes, and the statements that
touch it keep their order: lane ``a``'s, then lane ``b``'s; so do stores and
call statements.  Lane ``b``'s exp product is emitted where ``b``'s exp was,
or right after the pair when a later pair of lane ``b`` reads it (NVVM
contracts a product into the sums that read it wherever it sits in the
unrolled body); a pair whose lane ``b`` operands cannot be computed that
early keeps both lanes scalar.  A body with anything but assignments,
register stores and call statements (a branch, a nested loop) stays scalar.

Kernels with ``fast_math`` keep their scalar exps here: fast math already
pairs whole loops (``factor_affine_reductions.pack_fp32_constexpr_loops``)
and lowers ``exp`` through ``exp2``.  Targets before sm_100 have no packed
fp32 instructions.
"""

from __future__ import annotations

import ast
import dataclasses
from itertools import starmap
import math
import re
import struct
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ._ast_pass_utils import _PURE_OPERATOR_CALLS
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import is_constexpr_range
from ._ast_pass_utils import live_in
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import loop_live_end
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from ._ast_pass_utils import substitute_names
from .half_extremum_folds import HELPER as HALF_FOLD_HELPER
from .half_extremum_folds import loaded_packets
from .half_extremum_folds import loads_packet
from .hoist_loop_invariant_recip import EXACT_DIV_FAST_HELPER

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from ..device_ir import DeviceIR

EXP_PAIR_HELPER = "_cute_exp_f32x2_parts"
DIV_PAIR_HELPER = "_cute_exact_div_fast_f32x2"
_EXP_CALL = "cute.math.exp"
_FMA_CALL = "cute.math.fma"
_FP32_CTOR = "cutlass.Float32"
_FP64_CTOR = "cutlass.Float64"
_BF16_CTOR = "cutlass.BFloat16"
# The packed fp32 operation of each paired arithmetic kind.
_PACKED_ARITHMETIC = {
    "mul": "cute.arch.mul_packed_f32x2",
    "add": "cute.arch.add_packed_f32x2",
    "sub": "cute.arch.sub_packed_f32x2",
    "fma": "cute.arch.fma_packed_f32x2",
}
_ARITHMETIC_KINDS: tuple[tuple[type[ast.operator], str], ...] = (
    (ast.Mult, "mul"),
    (ast.Add, "add"),
    (ast.Sub, "sub"),
)
# The calls whose exps the pairs take (``has_pairable_exps_or_divisions``).
_TRACED_EXPS = frozenset({torch.ops.aten.exp.default, torch.ops.aten.sigmoid.default})
# The traced operations the generated code computes as fp32 products, sums,
# differences or fmas.
_TRACED_ARITHMETIC = frozenset(
    {
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.addcmul.default,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.rsub.Scalar,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
    }
)

# The calls a moved statement may make: conversions and pure arithmetic.  A
# statement of lane ``b`` that makes any other call (a load, a shuffle, a store
# helper) keeps its place, and the pairs that need it keep both lanes scalar.
_PURE_CALL_PREFIXES = ("cute.math.",)
_PURE_CALLS = frozenset(
    {"abs", "cute.arch.fmax", "cute.arch.fmin", EXACT_DIV_FAST_HELPER}
)
_CUTLASS_TYPE = re.compile(r"cutlass\.[A-Z]\w*")
# The largest stored value ``_definitions`` reads through its block's
# assignments, in AST nodes.
_INLINE_LIMIT = 64


def has_pairable_exps_or_divisions(device_ir: DeviceIR) -> bool:
    """Whether the kernel traces an exp or a sigmoid (which lowers through
    one) below float64, or a true division: with its fp32 arithmetic, what
    offers ``cute_pair_lane_math`` to the autotuner.  Structure only, like
    ``has_fp32_arithmetic``."""
    from ..inductor_lowering import _is_plain_true_division

    for info in device_ir.graphs:
        for node in info.graph.nodes:
            value = node.meta.get("val")
            if (
                node.op == "call_function"
                and isinstance(value, torch.Tensor)
                and value.dtype.is_floating_point
                and (
                    (node.target in _TRACED_EXPS and value.dtype != torch.float64)
                    or _is_plain_true_division(node)
                )
            ):
                return True
    return False


def has_fp32_arithmetic(device_ir: DeviceIR) -> bool:
    """Whether the kernel traces an fp32 product, sum or difference: with its
    exps and divisions, what offers ``cute_pair_lane_math`` to the
    autotuner.  Structure only, no extents; the pass proves each pair on the
    generated code, and a kernel with none renders the same either way."""
    return any(
        node.op == "call_function"
        and node.target in _TRACED_ARITHMETIC
        and isinstance(value := node.meta.get("val"), torch.Tensor)
        and value.dtype is torch.float32
        for info in device_ir.graphs
        for node in info.graph.nodes
    )


def pair_lane_math(
    body: list[ast.stmt],
    *,
    fast_math: bool,
    target_device_capability: tuple[int, int] | None,
    tensor_dtypes: Mapping[str, str] | None = None,
    constants: Mapping[str, int] | None = None,
    parameters: frozenset[str] | None = None,
) -> list[ast.stmt]:
    """Pair the exps, exact divisions and fp32 arithmetic of the V-loops in
    ``body``; ``tensor_dtypes`` maps tensor arguments to their element types
    (``cutlass.Float32``), the types of their scalar loads
    (``(t.iterator + ...).load()``).  ``constants`` gives the values of the
    module's constants ``body`` reads (``_BLOCK_SIZE_0 = 1``), Python
    numbers when the kernel is traced; given the kernel's ``parameters``,
    which may alias each other, any other name ``body`` reads but never
    assigns is such a constant, of unknown value."""
    if (
        fast_math
        or target_device_capability is None
        or target_device_capability < (10, 0)
    ):
        return body
    occupied = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    escaped, shared = _escaped_containers(body, loaded_packets(body))
    values = _definitions(body)
    for name, value in (constants or {}).items():
        if name in occupied and name not in values:
            values[name] = [ast.Constant(value=value)]
    unknown: set[str] = set()
    if parameters is not None:
        unknown = occupied - values.keys() - parameters
        shared |= parameters
    return _rewrite_block(
        body,
        set(),
        occupied,
        _Definitions(
            values,
            dict(tensor_dtypes or {}),
            escaped,
            shared,
            _constexpr_indices(body, values) | unknown,
        ),
    )


def _constexpr_indices(
    body: list[ast.stmt], values: dict[str, list[ast.expr]]
) -> frozenset[str]:
    """The targets of the ``cutlass.range_constexpr`` loops in ``body`` that
    nothing else assigns: Python ints when the kernel is traced."""
    targets = {
        node.target.id
        for statement in body
        for node in ast.walk(statement)
        if is_constexpr_range(node)
        and isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
    }
    return frozenset(
        name
        for name in targets
        if all(
            isinstance(value, ast.Name) and value.id == name
            for value in values.get(name, ())
        )
    )


@dataclasses.dataclass(frozen=True)
class _Definitions:
    """What the kernel assigns (``_definitions``), the element types of its
    tensor arguments, and the containers it uses in ways the analysis does
    not follow or whose storage another name writes
    (``_escaped_containers``)."""

    values: dict[str, list[ast.expr]]
    tensor_dtypes: dict[str, str]
    escaped: frozenset[str] = frozenset()
    shared: frozenset[str] = frozenset()
    # The names only ``cutlass.range_constexpr`` loops bind: Python ints at
    # trace time (``_constexpr_indices``); and the module's constants of
    # unknown value (``pair_lane_math``).
    constexpr_indices: frozenset[str] = frozenset()


def _rewrite_block(
    body: list[ast.stmt],
    live_out: set[str],
    occupied: set[str],
    definitions: _Definitions,
) -> list[ast.stmt]:
    live_after: list[set[str]] = [set() for _ in body]
    live = set(live_out)
    for index in reversed(range(len(body))):
        live_after[index] = set(live)
        live = live_in([body[index]], live)
    result: list[ast.stmt] = []
    for index, statement in enumerate(body):
        if isinstance(statement, ast.For):
            statement.body = _rewrite_block(
                statement.body,
                loop_live_end(statement, live_after[index])
                if isinstance(statement.target, ast.Name)
                # ``live_in`` reads such a loop whole: all its reads stay live.
                else live_after[index] | names_read(statement),
                occupied,
                definitions,
            )
            statement.orelse = _rewrite_block(
                statement.orelse, live_after[index], occupied, definitions
            )
            paired = _pair_lanes(statement, live_after[index], occupied, definitions)
            result.append(statement if paired is None else paired)
            continue
        if isinstance(statement, ast.If):
            statement.body = _rewrite_block(
                statement.body, live_after[index], occupied, definitions
            )
            statement.orelse = _rewrite_block(
                statement.orelse, live_after[index], occupied, definitions
            )
        result.append(statement)
    return result


@dataclasses.dataclass
class _Pair:
    """A pairable statement: ``kind`` ``"exp"`` (operands: the exp argument),
    ``"div"`` (the dividend, divisor and reciprocal) or a ``_PACKED_ARITHMETIC``
    kind (its operands) assigning ``target``."""

    kind: str
    target: str
    operands: list[ast.expr]


def _match_pair(statement: ast.stmt, assigned: set[str]) -> _Pair | None:
    """``x = [cutlass.Float32](cute.math.exp(cutlass.Float32(e)))`` or ``x =
    [cutlass.Float32](_cute_exact_div_fast_f32(a, b, y))`` with ``b`` and
    ``y`` loop invariants (not in ``assigned``); both calls return fp32."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return None
    target = statement.targets[0].id
    value = statement.value
    if _is_call(value, _FP32_CTOR, 1):
        assert isinstance(value, ast.Call)
        value = value.args[0]
    if _is_exp_call(value):
        assert isinstance(value, ast.Call)
        return _Pair("exp", target, [value.args[0]])
    if _is_call(value, EXACT_DIV_FAST_HELPER, 3):
        assert isinstance(value, ast.Call)
        invariant = all(
            isinstance(operand, ast.Constant)
            or (isinstance(operand, ast.Name) and operand.id not in assigned)
            for operand in value.args[1:]
        )
        if invariant:
            return _Pair("div", target, list(value.args))
    return None


def _is_call(node: ast.AST, path: str, arity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and call_path_str(node.func) == path
        and len(node.args) == arity
        and not node.keywords
    )


def _is_exp_call(node: ast.AST) -> bool:
    """``cute.math.exp`` of an explicitly fp32 operand: the form whose
    lowering ``_cute_exp_f32x2_parts`` reproduces."""
    return (
        _is_call(node, _EXP_CALL, 1)
        and isinstance(node, ast.Call)
        and _is_call(node.args[0], _FP32_CTOR, 1)
    )


def _is_pairable_call(node: ast.AST) -> bool:
    return _is_exp_call(node) or _is_call(node, EXACT_DIV_FAST_HELPER, 3)


def _fresh(occupied: set[str], base: str) -> str:
    name = base
    suffix = 0
    while name in occupied:
        suffix += 1
        name = f"{base}_{suffix}"
    occupied.add(name)
    return name


def _simple_statement(statement: ast.stmt) -> bool:
    """``name = value``, ``tensor[index] = value`` or a call statement (a
    packet's ``append``, a store): the statements a lane copy keeps
    verbatim."""
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Call)
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if isinstance(target, ast.Subscript):
        return isinstance(target.value, ast.Name)
    return isinstance(target, ast.Name)


def _written(statement: ast.stmt) -> str | None:
    """The name ``statement`` assigns or stores into; None for a call
    statement."""
    if isinstance(statement, ast.Expr):
        return None
    assert isinstance(statement, ast.Assign)
    target = statement.targets[0]
    if isinstance(target, ast.Subscript):
        target = target.value
    assert isinstance(target, ast.Name)
    return target.id


def _touched(statement: ast.stmt) -> set[str]:
    """The names whose values ``statement`` may change: its target, or every
    name a call statement passes (a call may mutate any object it sees)."""
    if isinstance(statement, ast.Expr):
        return names_read(statement)
    written = _written(statement)
    assert written is not None
    return {written}


class _Exposed(ast.NodeVisitor):
    """The pairable calls of an expression outside the arms of conditional
    expressions and boolean operators (where hoisting would evaluate them
    unconditionally) and outside nested scopes, innermost first."""

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.visit(node.values[0])

    def visit(self, node: ast.AST) -> None:
        if not isinstance(
            node,
            (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            super().visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if _is_pairable_call(node):
            self.calls.append(node)


class _ReplaceCall(ast.NodeTransformer):
    def __init__(self, call: ast.Call, name: str) -> None:
        self.call = call
        self.name = name

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if node is self.call:
            return ast.Name(id=self.name, ctx=ast.Load())
        self.generic_visit(node)
        return node


def _hoist_pairable_calls(
    body: list[ast.stmt], occupied: set[str], definitions: _Definitions
) -> list[ast.stmt]:
    """Give every pairable call nested in a larger expression its own
    assignment, reusing an earlier assignment of the same call when nothing
    it reads changed in between (the reduction folds keep a dead ``v =
    exp(d)`` beside the ``acc + exp(d)`` that uses the same exp)."""
    result: list[ast.stmt] = []
    for statement in body:
        assert isinstance(statement, (ast.Assign, ast.Expr))
        top = statement.value
        finder = _Exposed()
        finder.visit(top)
        for call in finder.calls:
            if isinstance(statement, ast.Assign) and (
                call is top
                or (
                    _is_call(top, _FP32_CTOR, 1)
                    and isinstance(top, ast.Call)
                    and call is top.args[0]
                )
            ):
                continue
            name = _reusable(result, call, definitions)
            if name is None:
                name = _fresh(occupied, "_helion_paired_call")
                result.append(
                    _at(statement, f"{name} = {{call}}", call=deep_clone(call))
                )
            statement.value = _ReplaceCall(call, name).visit(statement.value)
        result.append(statement)
    return result


def _reusable(
    earlier: list[ast.stmt], call: ast.Call, definitions: _Definitions
) -> str | None:
    """The name an earlier statement assigned ``call`` to, when neither it
    nor anything ``call`` reads was assigned since, by that statement
    either (``y = exp(y)`` holds the exp of the earlier ``y``).  None when
    ``call`` indexes a container another name may write
    (``_escaped_containers``) or one that is no plain name."""
    if _indexes_unnamed(call) or _subscripted(call) & (
        definitions.escaped | definitions.shared
    ):
        return None
    key = ast.dump(call)
    reads = names_read(call)
    for index in reversed(range(len(earlier))):
        statement = earlier[index]
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id not in reads
            and ast.dump(statement.value) == key
        ):
            written = statement.targets[0].id
            if not any(
                _touched(later) & (reads | {written}) for later in earlier[index + 1 :]
            ):
                return written
        if _touched(statement) & reads:
            return None
    return None


class _LaneCopy(ast.NodeTransformer):
    def __init__(self, lane: str, index: str, renames: dict[str, str]) -> None:
        self.lane = lane
        self.index = index
        self.renames = renames

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        # A lambda's parameters shadow the lane index and the renamed names
        # in its body (``lambda a, b: cute.arch.fmax(a, b)``).
        arguments = node.args
        params = {
            arg.arg
            for arg in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *filter(None, (arguments.vararg, arguments.kwarg)),
            )
        }
        outer = self.lane, self.renames
        if self.lane in params:
            self.lane = ""
        self.renames = {k: v for k, v in self.renames.items() if k not in params}
        try:
            return self.generic_visit(node)
        finally:
            self.lane, self.renames = outer

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.lane and isinstance(node.ctx, ast.Load):
            return expr_from_string(self.index)
        if node.id in self.renames:
            return ast.copy_location(
                ast.Name(id=self.renames[node.id], ctx=node.ctx), node
            )
        return node


def _lane_copy(
    statement: ast.stmt, lane: str, index: str, renames: dict[str, str]
) -> ast.stmt:
    return _LaneCopy(lane, index, renames).visit(deep_clone(statement))


def _pure_value(node: ast.expr) -> bool:
    """Only conversions and pure arithmetic (no loads, stores or calls with
    effects) under ``node``."""
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.Await,
                ast.Lambda,
                ast.NamedExpr,
                ast.Yield,
                ast.YieldFrom,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            return False
        if not isinstance(child, ast.Call):
            continue
        path = call_path_str(child.func)
        if path is None:
            if not (
                isinstance(child.func, ast.Attribute) and child.func.attr == "bitcast"
            ):
                return False
        elif not (
            path in _PURE_CALLS
            or path.startswith(_PURE_CALL_PREFIXES)
            or (
                path.startswith("operator.")
                and path.rsplit(".", 1)[-1] in _PURE_OPERATOR_CALLS
            )
            or _CUTLASS_TYPE.fullmatch(path)
        ):
            return False
    return True


def _movable(statement: ast.stmt, carried: set[str], readable: set[str]) -> bool:
    """A lane ``b`` statement that may run before the rest of lane ``a``: a
    pure register computation into a local that reads none of the
    ``carried`` names (running values) and indexes only ``readable``
    containers (``_readable_containers``)."""
    return (
        isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and _pure_value(statement.value)
        and not names_read(statement) & carried
        and not _indexes_unnamed(statement.value)
        and _subscripted(statement.value) <= readable
    )


def _indexed(node: ast.AST) -> ast.expr | None:
    """The container ``node`` indexes: ``c`` of ``c[i]`` or
    ``operator.getitem(c, i)``."""
    if isinstance(node, ast.Subscript):
        return node.value
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) == "operator.getitem"
        and node.args
    ):
        return node.args[0]
    return None


def _index_root(node: ast.expr) -> ast.expr:
    """The container at the root of an index chain (``c`` of ``c[i][j]``)."""
    while (inner := _indexed(node)) is not None:
        node = inner
    return node


def _subscripted(node: ast.AST) -> set[str]:
    """The names ``node`` indexes (``c[i]``, ``c[i][j]``, ``operator.getitem(c,
    i)``), by the root of each index chain."""
    return {
        name
        for child in ast.walk(node)
        if (container := _indexed(child)) is not None
        for name in (_root_names(_index_root(container)) or ())
    }


def _root_names(root: ast.expr) -> list[str] | None:
    """The names an index chain's root ``root`` stands for: a plain name, or
    the names of a list or tuple display of names (``(v0, v1)[i][j]``
    indexes ``v0`` or ``v1``); None for any other container."""
    if isinstance(root, ast.Name):
        return [root.id]
    if isinstance(root, (ast.List, ast.Tuple)) and all(
        isinstance(item, ast.Name) for item in root.elts
    ):
        return [cast("ast.Name", item).id for item in root.elts]
    return None


def _indexes_unnamed(node: ast.AST) -> bool:
    """Whether ``node`` indexes a container that is no plain name nor a
    display of names (``holder.frag[v]``), whose writes the analysis cannot
    see."""
    return any(
        (container := _indexed(child)) is not None
        and _root_names(_index_root(container)) is None
        for child in ast.walk(node)
    )


def _readable_containers(
    body: list[ast.stmt], stored: set[str], definitions: _Definitions
) -> set[str]:
    """The containers ``body`` indexes whose elements a moved statement may
    read early: ``body`` does not store into, append to or pass to a call
    any of them (``stored``), and no write elsewhere in ``body`` can reach
    one under another name.  So none is used in a way the analysis does not
    follow, none shares its storage with a name that writes it (an alias
    ``d = c``, a view ``d = c[None]``, a list it is in or appended to,
    ``_escaped_containers``), and the kernel gives one no value that might
    share another's storage (``_storage_followed``)."""
    return {
        name
        for statement in body
        for name in _subscripted(statement)
        if name not in stored
        and name not in definitions.escaped
        and name not in definitions.shared
        and _storage_followed(name, definitions, frozenset())
    }


def _storage_followed(
    name: str, definitions: _Definitions, seen: frozenset[str]
) -> bool:
    """Whether every value the kernel gives ``name`` is storage the analysis
    follows: an allocation, a list display, a loaded packet, or an alias or
    a view of a name whose values are such (``seen``: the names being
    decided).  A name standing for itself (a loop variable) is not."""
    values = definitions.values.get(name, ())
    for value in values:
        if (
            _fills_container(value)
            or isinstance(value, (ast.List, ast.Tuple))
            or _packet_valued(value, definitions, frozenset())
        ):
            continue
        if isinstance(value, ast.Subscript) and _is_view(value):
            value = _index_root(value)
        if not isinstance(value, ast.Name) or value.id == name:
            return False
        if value.id not in seen and not _storage_followed(
            value.id, definitions, seen | {name}
        ):
            return False
    return True


def _packet_valued(
    value: ast.expr, definitions: _Definitions, seen: frozenset[str]
) -> bool:
    """Whether ``value`` is a loaded packet: a packet load, a name every value
    of which is one, or an element of a list that holds only them and goes
    nowhere the analysis does not follow (``seen`` holds the names and lists
    whose values are being decided: a cycle passes values around and adds
    none)."""
    if loads_packet(value):
        return True
    if isinstance(value, ast.Name):
        key = value.id
    elif (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and not _is_view(value)
        and value.value.id not in definitions.escaped
    ):
        key = _stored_key(value.value.id)
    else:
        return False
    if key in seen:
        return True
    values = definitions.values.get(key)
    return bool(values) and all(
        _packet_valued(item, definitions, seen | {key}) for item in values
    )


# --------------------------------------------------------------------------
# Plain fp32 arithmetic
# --------------------------------------------------------------------------

# Calls whose result is a DSL fp32 value when an argument is one.
_FP32_RESULT_CALLS = frozenset(
    {
        "cute.math.abs",
        "cute.math.absf",
        "cute.math.atan2",
        "cute.math.cos",
        "cute.math.div",
        "cute.math.erf",
        "cute.math.exp",
        "cute.math.exp2",
        _FMA_CALL,
        "cute.math.log",
        "cute.math.log2",
        "cute.math.rsqrt",
        "cute.math.sin",
        "cute.math.sqrt",
        "cute.math.tanh",
        "cute.arch.fmax",
        "cute.arch.fmin",
    }
)
# Calls that read an fp32 argument without contracting it into a sum: value
# conversions and the explicit fma (whose operands are already unfused).
_NON_CONTRACTING_CONSUMERS = frozenset(
    {
        "cutlass.BFloat16",
        "cutlass.Float16",
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Uint16",
        "cutlass.Uint32",
        _FMA_CALL,
        "cute.arch.fmax",
        "cute.arch.fmin",
    }
)
# Calls whose result is no product: the fma and the exact division's fast
# path (which ends in one) round once, min and max select an operand, the
# shuffle reductions by max, min or sum end in their combine, and the rest
# load.  Any other call, a math function or an inlined ``@cute.jit`` helper
# alike, may end in an ``fmul`` (``_ends_without_product``).
_NON_PRODUCT_CALLS = frozenset(
    {
        _FMA_CALL,
        "cute.arch.fmax",
        "cute.arch.fmin",
        "cute.math.max",
        "cute.math.min",
        EXACT_DIV_FAST_HELPER,
        HALF_FOLD_HELPER,
        "cute.arch.warp_reduction_max",
        "cute.arch.warp_reduction_min",
        "cute.arch.warp_reduction_sum",
        "_cute_scalar_policy_load",
        "cute.arch.load",
    }
)
# The reductions of the ``_cute_grouped_reduce_*`` helpers that end in no
# product (``reduce_helpers._OP_TABLE``).
_NON_PRODUCT_REDUCTIONS = frozenset({"sum", "max", "min", "any"})
_RMEM_TENSOR = "cute.make_rmem_tensor"
# Conversions between integer types of the same width as a bitcast's, which
# keep its bits (``_Products._bitcast_source``).
_INTEGER_BIT_CASTS = frozenset({"cutlass.Int32", "cutlass.Uint32"})
# What an expression's value is: a DSL fp32 value, or a Python float or int
# the trace folds.
_F32 = "f32"
_PY_FLOAT = "float"
_PY_INT = "int"


def _definitions(body: list[ast.stmt]) -> dict[str, list[ast.expr]]:
    """Each name the kernel assigns -> the values it assigns, a loop variable
    or an augmented or unpacked name standing for itself (no one value); and
    ``_stored_key(name)`` -> the values the kernel puts into the container
    ``name`` (``name[i] = v``, ``name[i] op= v`` as ``name[i] op v``,
    ``name.append(v)``, the items of a list display), read in the store's
    own block: the names that block assigned
    before the store are replaced by their values there, since a temporary
    such as ``load`` holds something else in every loop that assigns it."""
    definitions: dict[str, list[ast.expr]] = {}

    def stand_alone(target: ast.expr) -> None:
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                definitions.setdefault(child.id, []).append(
                    ast.Name(id=child.id, ctx=ast.Load())
                )

    def visit(block: list[ast.stmt]) -> None:
        local: dict[str, ast.expr] = {}
        for statement in block:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    definitions.setdefault(target.id, []).append(statement.value)
                    local[target.id] = _inline(statement.value, local)
                    if isinstance(statement.value, (ast.List, ast.Tuple)):
                        definitions.setdefault(_stored_key(target.id), []).extend(
                            _inline(item, local) for item in statement.value.elts
                        )
                    continue
                if isinstance(target, ast.Subscript) and isinstance(
                    target.value, ast.Name
                ):
                    definitions.setdefault(_stored_key(target.value.id), []).append(
                        _inline(statement.value, local)
                    )
                    continue
            appended = _appended(statement)
            if appended is not None:
                container, value = appended
                definitions.setdefault(_stored_key(container), []).append(
                    _inline(value, local)
                )
            if (
                isinstance(statement, ast.AugAssign)
                and isinstance(statement.target, ast.Subscript)
                and isinstance(statement.target.value, ast.Name)
            ):
                # ``c[i] *= w`` stores ``c[i] * w``.
                element = deep_clone(statement.target)
                element.ctx = ast.Load()
                definitions.setdefault(
                    _stored_key(statement.target.value.id), []
                ).append(
                    _inline(
                        ast.BinOp(left=element, op=statement.op, right=statement.value),
                        local,
                    )
                )
            if isinstance(statement, (ast.For, ast.AugAssign, ast.AnnAssign)):
                stand_alone(statement.target)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    stand_alone(target)
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if isinstance(nested, list):
                    visit(nested)
            if isinstance(statement, ast.Try):
                for handler in statement.handlers:
                    visit(handler.body)
            for name in names_written(statement):
                local.pop(name, None)

    visit(body)
    return definitions


def _appended(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    """``(container, value)`` of a statement ``container.append(value)``."""
    if (
        isinstance(statement, ast.Expr)
        and isinstance(call := statement.value, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and len(call.args) == 1
        and not call.keywords
    ):
        return call.func.value.id, call.args[0]
    return None


def _escaped_containers(
    body: list[ast.stmt], loaded: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """The names the kernel uses other than as containers whose elements
    ``_definitions`` sees, and the names whose storage another name writes.
    The first are the names used in any way but ``c[i]``, ``c.append(v)``,
    an assignment to ``c``, an alias ``d = c`` or ``d = c[None]`` (a view,
    an index with ``None``, a slice or an ellipsis), an item of a list
    display assigned to a name or a value appended to a container, or ``c``
    folded by ``HALF_FOLD_HELPER``: a call taking ``c``, a method or
    attribute of it may write elements the analysis never sees, and so may
    a write through an alias.  The second are the names of an alias group
    (aliases, views, and the containers a name is listed in or appended to,
    unless it is a loaded packet, ``loaded``, an SSA value; transitively)
    that one of them writes or uses in such a way; a store through an index
    chain (``c[i][j] = v``) writes its root."""
    parents = _parents(body)
    # The names indexed somewhere: storage, where a scalar value is not.
    containers = {name for statement in body for name in _subscripted(statement)}
    escaped: set[str] = set()
    aliases: list[tuple[str, str]] = []
    written: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            parent = parents.get(id(node))
            if isinstance(parent, ast.Subscript) and parent.value is node:
                chain = [parent]
                while (
                    isinstance(outer := parents.get(id(chain[-1])), ast.Subscript)
                    and outer.value is chain[-1]
                ):
                    chain.append(outer)
                if isinstance(chain[-1].ctx, ast.Store):
                    written.add(node.id)
                elif any(_is_view(index) for index in chain):
                    assign = parents.get(id(chain[-1]))
                    if (
                        isinstance(assign, ast.Assign)
                        and assign.value is chain[-1]
                        and len(assign.targets) == 1
                        and isinstance(assign.targets[0], ast.Name)
                    ):
                        aliases.append((assign.targets[0].id, node.id))
                    else:
                        escaped.add(node.id)
                continue
            if (
                isinstance(parent, ast.Attribute)
                and parent.attr == "append"
                and isinstance(call := parents.get(id(parent)), ast.Call)
                and call.func is parent
            ):
                written.add(node.id)
                continue
            if isinstance(parent, ast.Assign) and parent.value is node:
                if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
                    aliases.append((parent.targets[0].id, node.id))
                    continue
            if (
                isinstance(parent, (ast.List, ast.Tuple))
                and isinstance(display_parent := parents.get(id(parent)), ast.Subscript)
                and display_parent.value is parent
            ):
                # An element of a name the display indexes (``(v0, v1)[i]``),
                # read or, through a store, written.
                chain = [display_parent]
                while (
                    isinstance(outer := parents.get(id(chain[-1])), ast.Subscript)
                    and outer.value is chain[-1]
                ):
                    chain.append(outer)
                if isinstance(chain[-1].ctx, ast.Store):
                    written.add(node.id)
                elif any(_is_view(index) for index in chain):
                    escaped.add(node.id)
                continue
            if (
                isinstance(parent, (ast.List, ast.Tuple))
                and isinstance(assign := parents.get(id(parent)), ast.Assign)
                and assign.value is parent
                and len(assign.targets) == 1
                and isinstance(assign.targets[0], ast.Name)
            ):
                # A container listed shares its storage with the list
                # (``frags = [frag]``), unless a loaded packet.
                if node.id in containers and node.id not in loaded:
                    aliases.append((assign.targets[0].id, node.id))
                continue
            if isinstance(parent, ast.Call) and node in parent.args:
                if call_path_str(parent.func) == HALF_FOLD_HELPER:
                    continue
                if (
                    isinstance(parent.func, ast.Attribute)
                    and parent.func.attr == "append"
                    and isinstance(parent.func.value, ast.Name)
                ):
                    # So does a container appended to one.
                    if node.id in containers and node.id not in loaded:
                        aliases.append((parent.func.value.id, node.id))
                    continue
            escaped.add(node.id)
    changed = True
    while changed:
        changed = False
        for alias, original in aliases:
            if original not in escaped and (alias in escaped or alias in written):
                escaped.add(original)
                changed = True
    groups: dict[str, set[str]] = {}
    for alias, original in aliases:
        group = groups.get(alias, {alias}) | groups.get(original, {original})
        for name in group:
            groups[name] = group
    shared = {name for name, group in groups.items() if group & (written | escaped)}
    return frozenset(escaped), frozenset(shared)


def _is_view(subscript: ast.Subscript) -> bool:
    """Whether ``subscript`` takes a view of its container (an index with
    ``None``, a slice or an ellipsis) rather than one element."""
    index = subscript.slice
    return any(
        isinstance(part, ast.Slice)
        or (isinstance(part, ast.Constant) and part.value in (None, Ellipsis))
        for part in (index.elts if isinstance(index, ast.Tuple) else [index])
    )


class _Abstract(ast.NodeTransformer):
    """Drop what the type and product analyses never read: a scalar load's
    address offsets (``(t.iterator + i).load()`` becomes
    ``t.iterator.load()``), subscript indices and select tests."""

    def visit_Call(self, node: ast.Call) -> ast.expr:
        tensor = _scalar_load_tensor(node)
        if tensor is None:
            self.generic_visit(node)
            return node
        load = expr_from_string(f"{tensor}.iterator.load()")
        assert isinstance(load, ast.expr)
        return load

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        node.value = self.visit(node.value)
        node.slice = ast.Constant(value=0)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        node.body = self.visit(node.body)
        node.orelse = self.visit(node.orelse)
        node.test = ast.Constant(value=True)
        return node


def _inline(value: ast.expr, local: dict[str, ast.expr]) -> ast.expr:
    """``value``, abstracted (``_Abstract``), with the names of ``local``
    replaced by their values, or without the replacement when that grows past
    ``_INLINE_LIMIT`` nodes (its names then stand for every value the kernel
    gives them)."""
    abstract = _Abstract().visit(deep_clone(value))
    assert isinstance(abstract, ast.expr)
    if not any(
        isinstance(node, ast.Name) and node.id in local for node in ast.walk(abstract)
    ):
        return abstract
    inlined = substitute_names(deep_clone(abstract), local, clone=deep_clone)
    if sum(1 for _ in ast.walk(inlined)) > _INLINE_LIMIT:
        return abstract
    return inlined


def _stored_key(name: str) -> str:
    """The ``_definitions`` key of what the kernel stores into ``name[...]``."""
    return f"{name}[]"


class _Kinds:
    """Which expressions of a loop body are DSL fp32 values: ``local``
    assignments of the body first, then the kernel's ``definitions``."""

    def __init__(
        self,
        local: dict[str, ast.expr],
        definitions: dict[str, list[ast.expr]],
        lane: str,
        tensor_dtypes: Mapping[str, str] | None = None,
        escaped: frozenset[str] = frozenset(),
        constexpr_indices: frozenset[str] = frozenset(),
    ) -> None:
        self.local = local
        self.definitions = definitions
        self.lane = lane
        self.tensor_dtypes = tensor_dtypes or {}
        self.escaped = escaped
        self.constexpr_indices = constexpr_indices
        self.cache: dict[str, str | None] = {}
        self.resolving: set[str] = set()
        self._kernel_wide: _Kinds | None = None

    def kernel_wide(self) -> _Kinds:
        """These kinds without the loop body's own assignments, for the
        values stored into register fragments (``_definitions`` reads them
        in the stores' blocks)."""
        if not self.local:
            return self
        if self._kernel_wide is None:
            self._kernel_wide = _Kinds(
                {},
                self.definitions,
                "",
                self.tensor_dtypes,
                self.escaped,
                self.constexpr_indices,
            )
        return self._kernel_wide

    def load_dtype(self, node: ast.Call) -> str | None:
        """The element type of a scalar load's tensor argument."""
        tensor = _scalar_load_tensor(node)
        return None if tensor is None else self.tensor_dtypes.get(tensor)

    def bf16_lane(self, node: ast.expr) -> bool:
        """Whether ``node`` is a bf16 value the loop body computes (a lane,
        not a loop invariant)."""
        if isinstance(node, ast.Name):
            value = self.local.get(node.id)
            return value is not None and self._resolve(
                f"bf16 {node.id}", lambda: self.bf16_lane(value)
            )
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return self.fragment_type(node.value.id) == _BF16_CTOR
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "bitcast"
                and len(node.args) == 1
            ):
                return call_path_str(node.args[0]) == _BF16_CTOR
            return (
                call_path_str(node.func) == _BF16_CTOR
                or self.load_dtype(node) == _BF16_CTOR
            )
        return False

    def widened_bf16_lane(self, node: ast.expr) -> bool:
        """Whether ``node`` is a bf16 lane widened to fp32, directly, through
        the body's names or through an fp32 fragment that holds only such
        values.  ptxas reads the bf16 half of a register straight into a
        scalar add (``FHADD.BF16``); a packed add needs both lanes widened
        first, two more instructions per pair."""
        if isinstance(node, ast.Name):
            value = self.local.get(node.id)
            return value is not None and self._resolve(
                f"widened {node.id}", lambda: self.widened_bf16_lane(value)
            )
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            tensor = node.value.id
            if self.fragment_type(tensor) != _FP32_CTOR:
                return False
            kinds = self.kernel_wide()
            stored = kinds.definitions.get(_stored_key(tensor), [])
            return bool(stored) and kinds._resolve(
                f"widened {_stored_key(tensor)}",
                lambda: all(kinds.widened_bf16_lane(value) for value in stored),
            )
        if (
            isinstance(node, ast.Call)
            and call_path_str(node.func) == _FP32_CTOR
            and len(node.args) == 1
        ):
            (operand,) = node.args
            return self.bf16_lane(operand) or (
                self.kind(operand) == _F32 and self.widened_bf16_lane(operand)
            )
        return False

    def fragment_type(self, tensor: str) -> str | None:
        """The element type of the register fragment ``tensor`` when every
        allocation of it gives the same one."""
        types = {
            call_path_str(value.args[1])
            if isinstance(value, ast.Call)
            and call_path_str(value.func) == _RMEM_TENSOR
            and len(value.args) == 2
            else None
            for value in self.definitions.get(tensor, [])
        }
        return types.pop() if len(types) == 1 else None

    def _resolve(self, key: str, compute: Callable[[], bool]) -> bool:
        """``compute()``, or False when ``key`` is already being resolved
        (a cycle proves nothing)."""
        if key in self.resolving:
            return False
        self.resolving.add(key)
        try:
            return compute()
        finally:
            self.resolving.discard(key)

    def values(self, name: str) -> list[ast.expr]:
        if name in self.local:
            return [self.local[name]]
        return self.definitions.get(name, [])

    def name(self, name: str) -> str | None:
        if name == self.lane:
            return _PY_INT
        if name in self.cache:
            return self.cache[name]
        if name in self.resolving:
            return None
        values = self.values(name)
        if not values:
            return None
        self.resolving.add(name)
        kinds = {self.kind(value) for value in values}
        self.resolving.discard(name)
        result = kinds.pop() if len(kinds) == 1 else None
        self.cache[name] = result
        return result

    def kind(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return None
            if isinstance(node.value, float):
                return _PY_FLOAT
            return _PY_INT if isinstance(node.value, int) else None
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return self.kind(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            kinds = {self.kind(node.left), self.kind(node.right)}
            if None in kinds:
                return None
            if _F32 in kinds:
                return _F32
            if isinstance(node.op, ast.Div) or _PY_FLOAT in kinds:
                return _PY_FLOAT
            return _PY_INT
        if isinstance(node, ast.IfExp):
            kinds = {self.kind(node.body), self.kind(node.orelse)}
            return _F32 if _F32 in kinds and kinds <= {_F32, _PY_FLOAT} else None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # An element of an fp32 register fragment.
            return _F32 if self.fragment_type(node.value.id) == _FP32_CTOR else None
        if isinstance(node, ast.Call):
            path = call_path_str(node.func)
            if path == _FP32_CTOR:
                return _F32
            if path == "_cute_scalar_policy_load":
                return (
                    _F32
                    if len(node.args) > 1 and call_path_str(node.args[1]) == _FP32_CTOR
                    else None
                )
            if path in _FP32_RESULT_CALLS:
                return (
                    _F32
                    if any(self.kind(argument) == _F32 for argument in node.args)
                    else None
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "bitcast"
                and len(node.args) == 1
                and call_path_str(node.args[0]) == _FP32_CTOR
            ):
                return _F32
            if self.load_dtype(node) == _FP32_CTOR:
                return _F32
        return None


def _trace_time(
    node: ast.expr, kinds: _Kinds, seen: frozenset[str] = frozenset()
) -> bool:
    """Whether ``node`` is a constant when the kernel is traced: a Python
    number (the lane index of a constexpr loop, the index of any other such
    loop and the module's constants included), a ``cutlass.Float32`` of
    one, which LLVM sees as a literal, or arithmetic of them."""
    if kinds.kind(node) in (_PY_INT, _PY_FLOAT) or (
        isinstance(node, ast.Name) and node.id in kinds.constexpr_indices
    ):
        return True
    if _fp32_bits(node) is not None:
        return True
    if _is_call(node, _FP32_CTOR, 1):
        assert isinstance(node, ast.Call)
        return _trace_time(node.args[0], kinds, seen)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _trace_time(node.operand, kinds, seen)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        return _trace_time(node.left, kinds, seen) and _trace_time(
            node.right, kinds, seen
        )
    if isinstance(node, ast.Name) and node.id not in seen:
        values = kinds.values(node.id)
        return bool(values) and all(
            _trace_time(value, kinds, seen | {node.id}) for value in values
        )
    return False


def _fp32_bits(node: ast.expr) -> int | None:
    """The bits of ``cutlass.Int32(n).bitcast(cutlass.Float32)`` (or of a
    ``cutlass.Uint32``), as the printer spells an fp32 constant whose sign
    a literal would lose (negative zero); None for anything else."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "bitcast"
        and len(node.args) == 1
        and not node.keywords
        and call_path_str(node.args[0]) == _FP32_CTOR
    ):
        return None
    source = node.func.value
    if not (
        isinstance(source, ast.Call)
        and call_path_str(source.func) in ("cutlass.Int32", "cutlass.Uint32")
        and len(source.args) == 1
        and not source.keywords
    ):
        return None
    bits = source.args[0]
    negative = isinstance(bits, ast.UnaryOp) and isinstance(bits.op, ast.USub)
    if isinstance(bits, ast.UnaryOp) and negative:
        bits = bits.operand
    if not (isinstance(bits, ast.Constant) and type(bits.value) is int):
        return None
    return (-bits.value if negative else bits.value) & 0xFFFFFFFF


def _trace_value(
    node: ast.expr, kinds: _Kinds, seen: frozenset[str] = frozenset()
) -> int | float | None:
    """The value of the trace-time constant ``node`` when it is plain to
    see (a literal, a sign, a ``cutlass.Float32`` of one, the bits of one,
    a name with that one value), else None; a ``cutlass.Float32`` rounds to
    fp32."""
    bits = _fp32_bits(node)
    if bits is not None:
        return struct.unpack("f", struct.pack("I", bits))[0]
    if isinstance(node, ast.Constant):
        value = node.value
        return (
            value
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _trace_value(node.operand, kinds, seen)
        if value is None or isinstance(node.op, ast.UAdd):
            return value
        return -value
    if _is_call(node, _FP32_CTOR, 1):
        assert isinstance(node, ast.Call)
        value = _trace_value(node.args[0], kinds, seen)
        return None if value is None else _as_f32(value)
    if isinstance(node, ast.Name) and node.id not in seen:
        values = kinds.values(node.id)
        if len(values) == 1:
            return _trace_value(values[0], kinds, seen | {node.id})
    return None


def _exact_reciprocal(value: float) -> bool:
    """Whether the fp32 ``value`` has an exact fp32 reciprocal (a power of
    two, either sign): the divisors LLVM turns into a multiply."""
    value = _as_f32(value)
    if value == 0 or not math.isfinite(value):
        return False
    mantissa, _ = math.frexp(abs(value))
    return (
        mantissa == 0.5
        and math.isfinite(_as_f32(1.0 / value))
        and (_as_f32(1.0 / value) != 0)
    )


def _as_f32(value: float) -> float:
    """``value`` rounded to fp32, as a trace-time constant becomes one."""
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _fold_operand(node: ast.BinOp, kinds: _Kinds) -> tuple[ast.expr, bool] | None:
    """``(x, certain)`` when NVVM may fold ``node`` to ``x`` or ``-x``:
    ``x * 1``, ``1 * x``, ``x / 1`` and their ``-1`` forms, ``x + -0.0``,
    ``-0.0 + x``, ``x - 0.0`` and ``-0.0 - x``, whose constant (as fp32)
    LLVM folds away whatever ``x`` holds, so a product reaching a sum
    through one still fuses into it.  ``certain`` is False for a trace-time
    operand whose value ``_trace_value`` cannot see, which may be such a
    constant; the lane index of the constexpr loop is one, a different
    constant in each lane."""
    if isinstance(node.op, (ast.Mult, ast.Div)):
        # The unit: on either side of a product, on the right of a quotient.
        sides = [(node.right, node.left)]
        if isinstance(node.op, ast.Mult):
            sides.append((node.left, node.right))
        for constant, operand in sides:
            if not _trace_time(constant, kinds):
                continue
            value = _trace_value(constant, kinds)
            if value is None:
                return operand, False
            if _as_f32(value) in (1.0, -1.0):
                return operand, True
        return None
    if not isinstance(node.op, (ast.Add, ast.Sub)):
        return None
    # The zero, and whether it must be -0.0: added on either side, -0.0 (an
    # added +0.0 turns -0.0 into +0.0); subtracted, +0.0, the int 0 too; a
    # -0.0 minus ``x`` is ``-x``.
    if isinstance(node.op, ast.Add):
        sides = [(node.right, node.left, True), (node.left, node.right, True)]
    else:
        sides = [(node.right, node.left, False), (node.left, node.right, True)]
    for constant, operand, negative in sides:
        if not _trace_time(constant, kinds):
            continue
        value = _trace_value(constant, kinds)
        if value is None:
            return operand, False
        zero = _as_f32(value)
        if zero == 0 and (math.copysign(1.0, zero) < 0) == negative:
            return operand, True
    return None


def _fma_folds(node: ast.Call, kinds: _Kinds) -> frozenset[str]:
    """What LLVM may fold the fma ``node`` into: ``"add"`` for a
    multiplicand of 1 or -1 (``fma(x, 1.0, z)`` is ``x + z``), ``"mul"``
    for an addend of -0.0 (``fma(x, y, -0.0)`` is ``x * y``); a trace-time
    operand whose value ``_trace_value`` cannot see may be either."""
    folds: set[str] = set()
    *multiplicands, addend = node.args
    for constant in multiplicands:
        if _trace_time(constant, kinds):
            value = _trace_value(constant, kinds)
            if value is None or _as_f32(value) in (1.0, -1.0):
                folds.add("add")
    if _trace_time(addend, kinds):
        value = _trace_value(addend, kinds)
        zero = None if value is None else _as_f32(value)
        if zero is None or (zero == 0 and math.copysign(1.0, zero) < 0):
            folds.add("mul")
    return frozenset(folds)


class _Products:
    """Whether an expression's value may be a product NVVM can contract."""

    def __init__(self, kinds: _Kinds) -> None:
        self.kinds = kinds
        self.resolving: set[str] = set()
        self._kernel_wide: _Products | None = None

    def kernel_wide(self) -> _Products:
        """The analysis without the loop body's own assignments, for the
        values stored into fragments: they are read in the stores' blocks."""
        if not self.kinds.local:
            return self
        if self._kernel_wide is None:
            self._kernel_wide = _Products(self.kinds.kernel_wide())
        return self._kernel_wide

    def holds_product(self, container: str) -> bool:
        """Whether an element of ``container`` may be a product: what the
        kernel puts into it (``_definitions``), or what the packet or
        container it is loaded, allocated or taken from holds.  Anything
        else defining it, or a use that may write it unseen
        (``_escaped_containers``), may give it any element."""
        kinds = self.kinds
        key = f"holds {container}"
        if container in kinds.escaped or key in self.resolving:
            return True
        self.resolving.add(key)
        try:
            for value in kinds.definitions.get(container, []):
                if isinstance(value, (ast.List, ast.Tuple)) or _fills_container(value):
                    continue
                if isinstance(value, ast.Name):
                    if self.holds_product(value.id):
                        return True
                    continue
                if isinstance(value, ast.Subscript) and isinstance(
                    value.value, ast.Name
                ):
                    # A packet taken from a container of packets.
                    if any(
                        self.packet_holds_product(packet)
                        for packet in kinds.definitions.get(
                            _stored_key(value.value.id), []
                        )
                    ) or self.holds_product(value.value.id):
                        return True
                    continue
                return True
            return any(
                self.may_be_product(value)
                for value in kinds.definitions.get(_stored_key(container), [])
            )
        finally:
            self.resolving.discard(key)

    def packet_holds_product(self, packet: ast.expr) -> bool:
        """Whether an element of the container value ``packet`` may be a
        product."""
        if _fills_container(packet):
            return False
        if isinstance(packet, ast.Name):
            return self.holds_product(packet.id)
        return True

    def may_be_product(self, node: ast.expr) -> bool:
        kinds = self.kinds
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Subscript):
            # An element of a container: of a loaded packet no product, of a
            # register fragment or list what the kernel puts there.
            if not isinstance(node.value, ast.Name):
                return True
            return self.kernel_wide().holds_product(node.value.id)
        if isinstance(node, ast.Name):
            if node.id == kinds.lane:
                return False
            if node.id in self.resolving:
                return True
            self.resolving.add(node.id)
            try:
                return any(
                    self.may_be_product(value) for value in kinds.values(node.id)
                )
            finally:
                self.resolving.discard(node.id)
        if isinstance(node, ast.UnaryOp):
            return self.may_be_product(node.operand)
        if isinstance(node, ast.BinOp):
            folded = _fold_operand(node, kinds)
            if folded is not None:
                # NVVM folds it to ``operand`` or its negation, or may.
                operand, certain = folded
                if certain:
                    return self.may_be_product(operand)
                if self.may_be_product(operand):
                    return True
            if kinds.kind(node) in (_PY_INT, _PY_FLOAT):
                # Computed at trace time.
                return False
            if isinstance(node.op, ast.Div):
                # LLVM turns a division by a power of two into the multiply
                # by its exact reciprocal, which a following add then fuses:
                # a quotient by a trace-time constant may be a product,
                # unless its value shows that it is no power of two.  A
                # value of unknown type counts as fp32.
                if not _trace_time(node.right, kinds):
                    return False
                divisor = _trace_value(node.right, kinds)
                return divisor is None or _exact_reciprocal(divisor)
            return isinstance(node.op, ast.Mult)
        if isinstance(node, ast.IfExp):
            return self.may_be_product(node.body) or self.may_be_product(node.orelse)
        if isinstance(node, ast.Call):
            path = call_path_str(node.func)
            if path in (_FP32_CTOR, _FP64_CTOR):
                # Widening to fp64 is exact, and LLVM folds the round trip
                # ``Float32(Float64(x))`` back to ``x``.
                return bool(node.args) and self.may_be_product(node.args[0])
            if path == "cute.arch.warp_reduction":
                # A shuffle reduction ends in its combine, a product only
                # when that multiplies (or is unknown).
                return len(node.args) != 2 or not _combines_without_product(
                    node.args[1]
                )
            if (source := self._bitcast_source(node)) is not None:
                # LLVM folds ``bitcast(bitcast(x))`` back to ``x``.
                return self.may_be_product(source)
            if _is_call(node, _FMA_CALL, 3) and "mul" in _fma_folds(node, kinds):
                # Folded into the product of its multiplicands.
                return True
            return not _ends_without_product(node, path)
        return True

    def _bitcast_source(self, node: ast.Call) -> ast.expr | None:
        """For a bitcast ``node`` of a value that is itself a bitcast (through
        integer conversions, which keep its bits, and names of one value),
        what that inner bitcast reinterprets, else None."""
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast"):
            return None
        value: ast.expr = node.func.value
        seen: set[str] = set()
        while True:
            if (
                isinstance(value, ast.Call)
                and call_path_str(value.func) in _INTEGER_BIT_CASTS
                and len(value.args) == 1
            ):
                value = value.args[0]
            elif isinstance(value, ast.Name) and value.id not in seen:
                values = self.kinds.values(value.id)
                if len(values) != 1:
                    return None
                seen.add(value.id)
                value = values[0]
            else:
                break
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "bitcast"
        ):
            return value.func.value
        return None


def _ends_without_product(node: ast.Call, path: str | None) -> bool:
    """Whether the call ``node`` (at ``path``) is known to end in no product:
    a load, an allocation or a bitcast, a conversion to a type other than
    fp32, a ``_NON_PRODUCT_CALLS`` call, or a grouped reduction by sum,
    max, min or any."""
    if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
        return True
    if _scalar_load_tensor(node) is not None or _fills_container(node):
        return True
    if path is None:
        return False
    if path in _NON_PRODUCT_CALLS or (
        path != _FP32_CTOR and _CUTLASS_TYPE.fullmatch(path) is not None
    ):
        return True
    return (
        path.startswith("_cute_grouped_reduce_")
        and len(node.args) > 1
        and isinstance(reduction := node.args[1], ast.Constant)
        and reduction.value in _NON_PRODUCT_REDUCTIONS
    )


def _combines_without_product(combine: ast.expr) -> bool:
    """Whether a warp reduction's combine is a known one that multiplies
    nothing: a lambda adding, or-ing or taking a max or min, or a
    NaN-propagating extremum helper."""
    if isinstance(combine, ast.Name):
        return combine.id in ("_cute_nan_max", "_cute_nan_min")
    if not isinstance(combine, ast.Lambda):
        return False
    body = combine.body
    if isinstance(body, ast.BinOp):
        return isinstance(body.op, (ast.Add, ast.BitOr, ast.BitAnd, ast.BitXor))
    return isinstance(body, ast.Call) and call_path_str(body.func) in (
        "cute.arch.fmax",
        "cute.arch.fmin",
        "max",
        "min",
    )


def _fills_container(value: ast.expr) -> bool:
    """Whether ``value`` allocates a register fragment or loads a packet:
    its elements are what the kernel stores there, or memory reads."""
    return loads_packet(value) or (
        isinstance(value, ast.Call) and call_path_str(value.func) == _RMEM_TENSOR
    )


def _scalar_load_tensor(node: ast.Call) -> str | None:
    """The tensor of a scalar load ``(t.iterator + offset).load()``, a read
    of one element of ``t``."""
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and not node.args
        and not node.keywords
    ):
        return None
    tensors = {
        sub.value.id
        for sub in ast.walk(node.func.value)
        if isinstance(sub, ast.Attribute)
        and sub.attr == "iterator"
        and isinstance(sub.value, ast.Name)
    }
    return next(iter(tensors)) if len(tensors) == 1 else None


def _parents(statements: list[ast.stmt]) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for statement in statements:
        for parent in ast.walk(statement):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
    return parents


def _feeds_a_sum(
    name: str,
    statements: list[ast.stmt],
    parents: dict[int, ast.AST],
    kinds: _Kinds,
    seen: set[str] | None = None,
) -> bool:
    """Whether ``name`` (or a value that only forwards it) is an operand of a
    sum or a difference, or reaches a consumer that could contract it."""
    seen = set() if seen is None else seen
    if name in seen:
        return False
    seen.add(name)
    return any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        and _use_feeds_a_sum(node, statements, parents, kinds, seen)
        for statement in statements
        for node in ast.walk(statement)
    )


def _use_feeds_a_sum(
    node: ast.expr,
    statements: list[ast.stmt],
    parents: dict[int, ast.AST],
    kinds: _Kinds,
    seen: set[str],
) -> bool:
    parent = parents.get(id(node))
    if isinstance(parent, ast.Assign):
        # A plain forward ``u = t``: follow ``u``.
        target = parent.targets[0] if len(parent.targets) == 1 else None
        if isinstance(target, ast.Name) and parent.value is node:
            return _feeds_a_sum(target.id, statements, parents, kinds, seen)
        return True
    if isinstance(parent, ast.BinOp):
        if isinstance(parent.op, (ast.Add, ast.Sub)):
            return True
        if (
            isinstance(parent.op, (ast.Mult, ast.Div))
            and (folded := _fold_operand(parent, kinds)) is not None
            and folded[0] is node
        ):
            return _use_feeds_a_sum(parent, statements, parents, kinds, seen)
        return not isinstance(parent.op, (ast.Mult, ast.Div))
    if isinstance(parent, ast.UnaryOp):
        return _use_feeds_a_sum(parent, statements, parents, kinds, seen)
    if isinstance(parent, ast.IfExp):
        if parent.test is node:
            return False
        return _use_feeds_a_sum(parent, statements, parents, kinds, seen)
    if isinstance(parent, ast.Compare):
        return False
    if isinstance(parent, ast.Call):
        if node not in parent.args:
            return True
        path = call_path_str(parent.func)
        if path == _FP32_CTOR:
            return _use_feeds_a_sum(parent, statements, parents, kinds, seen)
        if _is_call(parent, _FMA_CALL, 3) and "add" in _fma_folds(parent, kinds):
            # Folded into the sum of its other operands.
            return True
        # Any other call may add it, a container it is appended to included
        # (a later loop may read the element into a sum).
        return path not in _NON_CONTRACTING_CONSUMERS
    return True


def _arithmetic_pairs(
    body: list[ast.stmt],
    lane: str,
    carried: set[str],
    definitions: _Definitions,
) -> dict[int, str]:
    """The ``_PACKED_ARITHMETIC`` kind of each statement of ``body`` that
    pairs (module docstring: fp32, and nothing to contract), by index."""
    local = {
        statement.targets[0].id: statement.value
        for statement in body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id not in carried
    }
    kinds = _Kinds(
        local,
        definitions.values,
        lane,
        definitions.tensor_dtypes,
        definitions.escaped,
        definitions.constexpr_indices,
    )
    products = _Products(kinds)
    parents = _parents(body)
    result: dict[int, str] = {}
    for index, statement in enumerate(body):
        if not (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
        ):
            continue
        name = statement.targets[0].id
        value = statement.value
        if name in carried or kinds.kind(value) != _F32:
            continue
        if _is_call(value, _FMA_CALL, 3):
            # Unless LLVM may fold it into a sum or a product, which NVVM
            # then contracts with its neighbours (``_fma_folds``).
            assert isinstance(value, ast.Call)
            if not _fma_folds(value, kinds):
                result[index] = "fma"
            continue
        if not isinstance(value, ast.BinOp):
            continue
        kind = next(
            (kind for op, kind in _ARITHMETIC_KINDS if isinstance(value.op, op)), None
        )
        if kind is None or _F32 not in (
            kinds.kind(value.left),
            kinds.kind(value.right),
        ):
            continue
        if _fold_operand(value, kinds) is not None:
            # NVVM folds ``x * 1``, ``x - 0.0`` and their kin away, keeping
            # ``x`` bit for bit (a NaN's payload too), where a packed
            # operation would compute.
            continue
        if kind == "mul":
            if _feeds_a_sum(name, body, parents, kinds):
                continue
        elif any(
            products.may_be_product(operand) or kinds.widened_bf16_lane(operand)
            for operand in (value.left, value.right)
        ):
            continue
        result[index] = kind
    return result


def _match_arithmetic(statement: ast.stmt, kind: str) -> _Pair:
    """``statement``'s pair of ``kind`` (``_arithmetic_pairs``)."""
    assert isinstance(statement, ast.Assign)
    target = statement.targets[0]
    value = statement.value
    assert isinstance(target, ast.Name)
    if kind == "fma":
        assert isinstance(value, ast.Call)
        return _Pair(kind, target.id, list(value.args))
    assert isinstance(value, ast.BinOp)
    return _Pair(kind, target.id, [value.left, value.right])


def _as_fp32(operand: ast.expr) -> ast.expr:
    """``operand`` as a DSL fp32 value: ``cutlass.Float32(operand)``, unless
    it is one already."""
    if _is_call(operand, _FP32_CTOR, 1):
        return deep_clone(operand)
    value = expr_from_string(f"{_FP32_CTOR}({{operand}})", operand=deep_clone(operand))
    assert isinstance(value, ast.expr)
    return value


def _fp32_pair(operand_a: ast.expr, operand_b: ast.expr) -> ast.expr:
    """``(cutlass.Float32(a), cutlass.Float32(b))``, a packed operand."""
    value = expr_from_string("({a}, {b})", a=_as_fp32(operand_a), b=_as_fp32(operand_b))
    assert isinstance(value, ast.expr)
    return value


def _operand_closure(
    statement: ast.stmt, lane_b: list[ast.stmt], defs: dict[str, int], stop: set[int]
) -> set[int]:
    """The indices of lane ``b``'s statements whose values ``statement``'s
    operands depend on, not looking through the statements in ``stop``."""
    assert isinstance(statement, ast.Assign)
    result: set[int] = set()
    todo = list(names_read(statement.value))
    while todo:
        index = defs.get(todo.pop())
        if index is None or index in result:
            continue
        result.add(index)
        if index not in stop:
            body_statement = lane_b[index]
            assert isinstance(body_statement, ast.Assign)
            todo.extend(names_read(body_statement.value))
    return result


def _pair_lanes(
    loop: ast.For,
    live_after: set[str],
    occupied: set[str],
    definitions: _Definitions,
) -> ast.For | None:
    """``loop`` with its exps, exact divisions and fp32 arithmetic paired
    over adjacent lanes, or None when nothing pairs."""
    extent = loop_extent(loop, kinds=("constexpr",), minimum=2)
    if extent is None or extent % 2:
        return None
    assert isinstance(loop.target, ast.Name)
    lane = loop.target.id
    if lane in live_after:
        # A read after the loop sees its last lane; the paired loop binds
        # another name.
        return None
    if not all(_simple_statement(statement) for statement in loop.body):
        return None
    if not any(
        _is_pairable_call(node)
        or isinstance(node, ast.BinOp)
        or _is_call(node, _FMA_CALL, 3)
        for statement in loop.body
        for node in ast.walk(statement)
    ):
        return None
    body = _hoist_pairable_calls(
        [deep_clone(statement) for statement in loop.body], occupied, definitions
    )
    assigned = {
        statement.targets[0].id
        for statement in body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
    }
    if lane in assigned:
        return None
    defs: dict[str, int] = {}
    carried: set[str] = set()
    stored: set[str] = set()
    for index, statement in enumerate(body):
        carried |= (names_read(statement) & assigned) - set(defs)
        if isinstance(statement, ast.Assign) and isinstance(
            statement.targets[0], ast.Name
        ):
            name = statement.targets[0].id
            if name in defs and name not in carried:
                return None
            # A running value, read before its first assignment, may be
            # updated more than once per lane: its statements keep their
            # order and none of them moves.
            defs.setdefault(name, index)
        else:
            stored |= _touched(statement)
    carried |= assigned & live_after
    local = assigned - carried
    arithmetic = _arithmetic_pairs(body, lane, carried, definitions)

    def match(index: int, statement: ast.stmt, assigned: set[str]) -> _Pair | None:
        kind = arithmetic.get(index)
        if kind is not None:
            return _match_arithmetic(statement, kind)
        return _match_pair(statement, assigned)

    # The lane index varies per lane like the names the body assigns: no
    # invariant divisor (its lane copies would not match either).
    varying = assigned | {lane}
    pairs = [
        index
        for index, statement in enumerate(body)
        if (pair := match(index, statement, varying)) is not None
        and pair.target in local
    ]
    if not pairs:
        return None

    pair_lane = _fresh(occupied, f"{lane}_pair")
    renames = {name: _fresh(occupied, f"{name}_b") for name in sorted(local)}
    lane_a = [_lane_copy(statement, lane, f"2 * {pair_lane}", {}) for statement in body]
    lane_b = [
        _lane_copy(statement, lane, f"2 * {pair_lane} + 1", renames)
        for statement in body
    ]
    defs_b = {renames[name]: index for name, index in defs.items() if name in renames}
    assigned_b = {renames.get(name, name) for name in assigned}
    readable_b = {
        renames.get(name, name)
        for name in _readable_containers(body, stored, definitions)
    }

    result: list[ast.stmt] = []
    next_a = 0
    done_b: set[int] = set()
    # Lane ``b``'s exp products not yet emitted: index -> (name, statement).
    deferred_b: dict[int, tuple[str, ast.stmt]] = {}
    formed = False
    for index in pairs:
        # Each lane's own copy: an operand may read the lane index.
        pair = match(index, lane_a[index], assigned)
        pair_b = match(index, lane_b[index], assigned_b)
        assert pair is not None and pair_b is not None and pair_b.kind == pair.kind
        closure = _operand_closure(
            lane_b[index], lane_b, defs_b, done_b | set(deferred_b)
        )
        products = {i for i in closure if i in deferred_b}
        moved = closure - done_b - products
        # The paired call itself runs early too: it must not read a running
        # value either.
        if not all(_movable(lane_b[i], carried, readable_b) for i in moved | {index}):
            continue
        formed = True
        result.extend(lane_a[next_a:index])
        next_a = index + 1
        # In body order: a moved statement may read a deferred product.
        result.extend(
            deferred_b.pop(i)[1] if i in products else lane_b[i]
            for i in sorted(moved | products)
        )
        done_b |= moved | products
        if pair.kind == "exp":
            parts = [
                _fresh(occupied, f"{pair.target}_{part}")
                for part in ("pa", "pb", "sa", "sb")
            ]
            result.extend(
                (
                    _call_statement(
                        lane_a[index],
                        parts,
                        EXP_PAIR_HELPER,
                        [pair.operands[0], pair_b.operands[0]],
                    ),
                    _at(lane_a[index], f"{pair.target} = {parts[0]} * {parts[2]}"),
                )
            )
            deferred_b[index] = (
                pair_b.target,
                _at(lane_b[index], f"{pair_b.target} = {parts[1]} * {parts[3]}"),
            )
        elif pair.kind == "div":
            dividend, divisor, reciprocal = pair.operands
            result.append(
                _call_statement(
                    lane_a[index],
                    [pair.target, pair_b.target],
                    DIV_PAIR_HELPER,
                    [dividend, pair_b.operands[0], divisor, reciprocal],
                )
            )
            done_b.add(index)
        else:
            result.append(
                _call_statement(
                    lane_a[index],
                    [pair.target, pair_b.target],
                    _PACKED_ARITHMETIC[pair.kind],
                    list(
                        starmap(
                            _fp32_pair, zip(pair.operands, pair_b.operands, strict=True)
                        )
                    ),
                )
            )
            done_b.add(index)
    if not formed:
        return None
    result.extend(lane_a[next_a:])
    for index, statement in enumerate(lane_b):
        if index in deferred_b:
            result.append(deferred_b[index][1])
        elif index not in done_b:
            result.append(statement)
    target = ast.Name(id=pair_lane, ctx=ast.Store())
    iterator = expr_from_string(f"cutlass.range_constexpr({extent // 2})")
    assert isinstance(iterator, ast.expr)
    if isinstance(loop, ExtendedAST):
        return cast("ast.For", loop.copy(target=target, iter=iterator, body=result))
    return ast.copy_location(
        ast.For(target=target, iter=iterator, body=result, orelse=[]), loop
    )


def _at(origin: ast.stmt, template: str, **placeholders: ast.AST) -> ast.stmt:
    """``statement_from_string`` at ``origin``'s source location (the
    location comments the generated code shows)."""
    if isinstance(origin, ExtendedAST):
        with origin:
            return statement_from_string(template, **placeholders)
    return statement_from_string(template, **placeholders)


def _call_statement(
    origin: ast.stmt, targets: list[str], helper: str, operands: list[ast.expr]
) -> ast.stmt:
    """``targets = helper(*operands)`` at ``origin``'s location."""
    arguments = {
        f"operand{i}": deep_clone(operand) for i, operand in enumerate(operands)
    }
    call = ", ".join("{" + name + "}" for name in arguments)
    return _at(origin, f"{', '.join(targets)} = {helper}({call})", **arguments)
