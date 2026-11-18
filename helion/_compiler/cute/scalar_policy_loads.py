"""Lower hinted scalar loads after all CuTe memory scheduling passes.

``lower_scalar_policy_loads`` turns the hinted ``cute.arch.load(ptr,
cutlass.T, <hint>)`` placeholders of 4- and 8-byte elements into
exact-policy PTX.  ``hint_element_loads`` runs after every other pass, under
the ``cute_scalar_load_hints`` config knob: it gives the element loads that
are still plain the L2 policy their tensor's load slots name, and makes a
hinted element load under a mask one predicated load.  Under
``cute_narrow_load_hints`` it does the same for every policy of a 2-byte
tensor.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import names_stored_or_deleted
from .cache_policy_loads import LOAD_POLICY_KEYWORDS
from .cache_policy_loads import CachePolicy
from .dtypes import CUTLASS_INTEGER_TYPE_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

_SCALAR_TYPES = frozenset(("Float32", "Float64", "Int32", "Int64", "Uint32", "Uint64"))


def _is_arch_load(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "arch"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "cute"
    )


def scalar_load_policy(node: ast.Call) -> str | None:
    """The ``_cute_scalar_policy_load`` policy a hinted scalar
    ``cute.arch.load(ptr, cutlass.<T>, <hint>)`` lowers to, or None when
    ``lower_scalar_policy_loads`` leaves the call as it is."""
    if not _is_arch_load(node) or len(node.args) != 2 or len(node.keywords) != 1:
        return None
    dtype = node.args[1]
    if (
        not isinstance(dtype, ast.Attribute)
        or not isinstance(dtype.value, ast.Name)
        or dtype.value.id != "cutlass"
        or dtype.attr not in _SCALAR_TYPES
    ):
        return None
    cache_policy = CachePolicy.from_load_keywords(node.keywords)
    return cache_policy.scalar_policy if cache_policy is not None else None


def is_extended_scalar_load(node: ast.Call) -> bool:
    """Whether a hinted scalar ``cute.arch.load`` reaches NVVM as an extended
    load: one with cache keywords and a scalar dtype (not a vector type) that
    ``lower_scalar_policy_loads`` does not turn into exact-policy PTX."""
    if not _is_arch_load(node) or not node.keywords:
        return False
    if len(node.args) >= 2 and (
        isinstance(node.args[1], ast.Call)
        and ast.unparse(node.args[1].func) == "ir.VectorType.get"
    ):
        return False
    return scalar_load_policy(node) is None


class _LowerScalarPolicyLoads(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        policy = scalar_load_policy(node)
        if policy is None:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="_cute_scalar_policy_load", ctx=ast.Load()),
                args=[*node.args, ast.Constant(value=policy)],
                keywords=[],
            ),
            node,
        )


def lower_scalar_policy_loads(body: list[ast.stmt]) -> list[ast.stmt]:
    """Preserve load recognition until the final device emission boundary.

    Mixing scalar NVVM extended loads with ordinary LLVM loads can abort the
    SDK load/store vectorizer. Exact-policy PTX avoids that grouping without
    changing the selected cache hints. Earlier scheduling and alias passes
    must still see their original ``cute.arch.load`` nodes.
    """
    if not any(
        isinstance(node, ast.keyword) and node.arg in LOAD_POLICY_KEYWORDS
        for statement in body
        for node in ast.walk(statement)
    ):
        # Every lowered load carries a policy keyword; without one the
        # transformer would return each statement unchanged.
        return list(body)
    lower = _LowerScalarPolicyLoads()
    return [cast("ast.stmt", lower.visit(statement)) for statement in body]


# -- the element hints of ``cute_scalar_load_hints`` -------------------------

_SCALAR_POLICY_LOAD = "_cute_scalar_policy_load"
_ELEMENT_POLICY_LOAD = "_cute_element_policy_load"
_ELEMENT_POLICY_LOAD_IF = "_cute_element_policy_load_if"
# Calls whose value depends on no memory: the launch coordinates.
_COORDINATE_CALLS = frozenset(
    f"cute.arch.{name}" for name in ("thread_idx", "block_idx", "block_dim", "grid_dim")
)
# The attributes of a tensor whose constant subscripts are launch-invariant
# integers: ``x.layout.stride[0]``, ``x.shape[1]``.
_LAYOUT_ATTRIBUTES = frozenset(("stride", "shape"))


def _is_zero_of(node: ast.expr, dtype: str) -> bool:
    """Whether ``node`` is ``<dtype>(0)``, the masked-off value of a scalar
    load of that type (``_cute_masked_value_expr``)."""
    return (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and type(node.args[0].value) is int
        and node.args[0].value == 0
        and ast.unparse(node.func) == dtype
    )


def _is_constant_index(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def _is_plain_address(node: ast.expr) -> bool:
    """Whether evaluating the pointer ``node`` reads no memory and has no
    effect, so it may be evaluated for an element its mask excludes: names
    and their attributes (``x.iterator``), integer constants, arithmetic,
    integer casts, the launch coordinates (``cute.arch.thread_idx()[0]``)
    and constant subscripts of a layout attribute (``x.layout.stride[0]``,
    ``x.shape[1]``).  Any other subscript may be an element of a tensor
    (``table[i]``, the form a gather through a loaded index takes), and a
    helper call or a conditional may only run where the mask holds.
    """
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _is_plain_address(node.value)
    if isinstance(node, ast.BinOp):
        return _is_plain_address(node.left) and _is_plain_address(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_plain_address(node.operand)
    if isinstance(node, ast.Subscript):
        if not _is_constant_index(node.slice):
            return False
        value = node.value
        if isinstance(value, ast.Call):
            return (
                call_path_str(value.func) in _COORDINATE_CALLS
                and not value.args
                and not value.keywords
            )
        return (
            isinstance(value, ast.Attribute)
            and value.attr in _LAYOUT_ATTRIBUTES
            and _is_plain_address(value.value)
        )
    if isinstance(node, ast.Call):
        return (
            call_path_str(node.func) in CUTLASS_INTEGER_TYPE_NAMES
            and not node.keywords
            and all(_is_plain_address(argument) for argument in node.args)
        )
    return False


def _pointer_tensor(pointer: ast.expr) -> str | None:
    """The tensor whose elements ``pointer`` addresses: ``X.iterator`` plus
    or minus offsets that name no iterator, so the pointer keeps ``X``'s
    element type.  None for any other pointer (a named or selected pointer,
    a cast one)."""
    node = pointer
    while isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        if any(
            isinstance(child, ast.Attribute) and child.attr == "iterator"
            for child in ast.walk(node.right)
        ):
            return None
        node = node.left
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    ):
        return node.value.id
    return None


def _helper_call(name: str, arguments: list[ast.expr], like: ast.AST) -> ast.Call:
    return ast.copy_location(
        ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=arguments, keywords=[]),
        like,
    )


class _HintElementLoads(ast.NodeTransformer):
    def __init__(
        self, hints: Mapping[str, tuple[str, str]], predicate_lowered: bool
    ) -> None:
        self.hints = hints
        self.predicate_lowered = predicate_lowered
        # Inside a branch of a conditional expression a hinted load would be
        # inline PTX under a branch: nothing is rewritten there.  The
        # masked-zero form is taken at the top of its expression only.
        self.conditional_depth = 0

    def plain_load(self, node: ast.expr) -> tuple[ast.expr, str, str] | None:
        """``(pointer, dtype, policy)`` of a plain ``(PTR).load()`` of a
        hinted tensor's element."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and not node.args
            and not node.keywords
        ):
            return None
        pointer = node.func.value
        hint = self.hints.get(_pointer_tensor(pointer) or "")
        return None if hint is None else (pointer, *hint)

    def hinted_load(self, node: ast.expr) -> tuple[ast.expr, str, str] | None:
        """``(pointer, dtype, policy)`` of a lowered hinted scalar load,
        ``_cute_scalar_policy_load(PTR, cutlass.T, 'policy')``."""
        if not (
            self.predicate_lowered
            and isinstance(node, ast.Call)
            and call_path_str(node.func) == _SCALAR_POLICY_LOAD
            and len(node.args) == 3
            and not node.keywords
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, str)
        ):
            return None
        return node.args[0], ast.unparse(node.args[1]), node.args[2].value

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        # ``load if mask else T(0)``: the masked element load.  The
        # predicated helper loads under the mask in one instruction and
        # yields the same zero where the mask is false.  It evaluates the
        # address for a masked-off element too, so the address must be
        # plain arithmetic; any other load keeps its conditional, and a
        # plain load its plain form.
        load = self.plain_load(node.body) or self.hinted_load(node.body)
        if (
            load is not None
            and not self.conditional_depth
            and _is_zero_of(node.orelse, load[1])
            and _is_plain_address(load[0])
        ):
            pointer, dtype, policy = load
            return _helper_call(
                _ELEMENT_POLICY_LOAD_IF,
                [
                    pointer,
                    cast("ast.expr", self.visit(node.test)),
                    ast.parse(dtype, mode="eval").body,
                    ast.Constant(value=policy),
                ],
                node,
            )
        node.test = cast("ast.expr", self.visit(node.test))
        self.conditional_depth += 1
        node.body = cast("ast.expr", self.visit(node.body))
        node.orelse = cast("ast.expr", self.visit(node.orelse))
        self.conditional_depth -= 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        load = None if self.conditional_depth else self.plain_load(node)
        if load is None:
            return self.generic_visit(node)
        pointer, dtype, policy = load
        return _helper_call(
            _ELEMENT_POLICY_LOAD,
            [pointer, ast.parse(dtype, mode="eval").body, ast.Constant(value=policy)],
            node,
        )


def hint_element_loads(
    definition: ast.FunctionDef,
    hints: Mapping[str, tuple[str, str] | None],
    *,
    predicate_lowered: bool = True,
) -> list[ast.stmt]:
    """The body of the kernel ``definition`` with the element-load forms of
    ``cute_scalar_load_hints`` and ``cute_narrow_load_hints``.

    ``hints`` maps a tensor parameter to the cutlass type of its elements
    and the cache policy its load slots name (``element_load_hint``, recorded
    by the load lowering), or to None where the tensor takes no hint (one of
    its 2-byte loads spells out its own policy).  Every ``(X.iterator + offset).load()`` of such a
    tensor that is still a plain element load takes it:
    ``_cute_element_policy_load(ptr, T, policy)``.  This is the last pass, so
    the loads the vectorizers merged into packets, the loads a register
    serves and the loads other passes rewrote are gone, and no pass saw a
    hinted form it does not read.  A hint changes no value, so which loads
    remain is a matter of speed only.

    A hinted load under a mask, ``load if mask else T(0)`` with a plain
    hinted tensor load or (``predicate_lowered``: the scalar knob, not the
    narrow one alone) a lowered ``_cute_scalar_policy_load`` as its load,
    becomes ``_cute_element_policy_load_if(ptr, mask, T, policy)``:
    one predicated load, where inline PTX inside the conditional compiles to
    a branch around it.  That needs the masked-off value to be ``T(0)`` of
    the load's type and the address to be plain arithmetic
    (``_is_plain_address``), since the predicated form evaluates the address
    where the mask is false too.  Otherwise a lowered load keeps its
    conditional and a plain load anywhere inside a conditional expression
    stays plain; so does a masked load nested in a branch of another
    conditional.

    A tensor is taken by name: only parameters of ``definition`` that the
    body never rebinds count.
    """
    parameters = {argument.arg for argument in definition.args.args}
    rebound: set[str] = set()
    for statement in definition.body:
        rebound |= names_stored_or_deleted(statement)
        for node in ast.walk(statement):
            if isinstance(node, ast.For):
                # A generated loop target may carry no Store context.
                rebound |= {
                    child.id
                    for child in ast.walk(node.target)
                    if isinstance(child, ast.Name)
                }
    hint = _HintElementLoads(
        {
            name: value
            for name, value in hints.items()
            if value is not None and name in parameters and name not in rebound
        },
        predicate_lowered,
    )
    return [cast("ast.stmt", hint.visit(statement)) for statement in definition.body]
