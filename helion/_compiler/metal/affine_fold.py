"""Fold integer division and remainder out of index arithmetic whose
operands codegen knows: constants, names it knows to be multiples of some
integer, and names it bounds.

An element of a fragment run (``fragment_mma``) re-runs the Scalar
statements that compute it with its position bound to constants and to the
lane's row and column (``_fm``, ``_fn``).  Their index arithmetic divides
and takes remainders (``k // 8`` picks the word of a packed code, ``k % 8``
its nibble), which the Metal compiler can't simplify without knowing that
the K tile's offset is a multiple of 8 and that ``_fn`` is even.  With
those facts, element ``e`` of step ``s`` at K ``offset + 8 * (_fn + e) + s``
reads word ``offset / 8 + _fn + e`` and nibble ``s``: the same word
expression at every step, whose loads the compiler then shares.

An integer is affine, ``c + sum(a_i * x_i)``, over names ``x_i`` with a
:class:`Fact` each.  ``(D + R) // d`` is ``D / d + q`` when every term of
``D`` is a multiple of ``d`` and ``R``'s range lies within one multiple of
``d``, ``[q * d, q * d + d)``; ``(D + R) % d`` is then ``R - q * d``.
"""

from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction
import math
from typing import TYPE_CHECKING

from .mpp_matmul import BIND

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class Fact:
    """What codegen knows of an integer name: a multiple of ``multiple``,
    at least ``lo`` and at most ``hi`` (None: unbounded).  A floor term's
    ``depends`` holds the names it is computed from, its own floor terms'
    included."""

    multiple: int = 1
    lo: int | None = None
    hi: int | None = None
    depends: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class Affine:
    """``const + sum(coeff * name)``, each ``coeff * multiple`` an integer."""

    const: Fraction
    terms: tuple[tuple[str, Fraction], ...] = ()

    @staticmethod
    def of(value: int) -> Affine:
        return Affine(Fraction(value))

    @staticmethod
    def term(name: str, coeff: int = 1) -> Affine:
        return Affine(Fraction(0), ((name, Fraction(coeff)),))

    def __add__(self, other: Affine) -> Affine:
        terms = dict(self.terms)
        for name, coeff in other.terms:
            terms[name] = terms.get(name, Fraction(0)) + coeff
        return Affine(
            self.const + other.const,
            tuple(sorted((n, c) for n, c in terms.items() if c)),
        )

    def scale(self, factor: Fraction) -> Affine:
        if not factor:
            return Affine(Fraction(0))
        return Affine(
            self.const * factor, tuple((n, c * factor) for n, c in self.terms)
        )

    @property
    def constant(self) -> int | None:
        if self.terms or self.const.denominator != 1:
            return None
        return int(self.const)


def _range(
    value: Affine, facts: Mapping[str, Fact]
) -> tuple[Fraction | None, Fraction | None]:
    """The least and greatest ``value`` takes, exactly (None: unbounded)."""
    lo: Fraction | None = value.const
    hi: Fraction | None = value.const
    for name, coeff in value.terms:
        fact = facts[name]
        least, most = (fact.lo, fact.hi) if coeff > 0 else (fact.hi, fact.lo)
        lo = None if lo is None or least is None else lo + coeff * least
        hi = None if hi is None or most is None else hi + coeff * most
    return lo, hi


def floor_divide(value: Affine, divisor: int, facts: dict[str, Fact]) -> Affine | None:
    """``value // divisor`` as an affine value, or None.  ``value`` is ``D +
    T + c``: ``D`` the terms that are multiples of ``divisor``, ``T`` the
    others, ``c`` the constant.  When ``T + c`` stays within one multiple of
    ``divisor``, the quotient is ``D / divisor`` plus that multiple's index.
    Otherwise, if ``T``'s terms are multiples of some ``g`` dividing
    ``divisor``, ``c``'s remainder mod ``g`` can't carry ``T + c`` across a
    multiple of ``divisor``: the quotient is ``D / divisor + q + F``, ``F =
    (T + c') / divisor`` with ``c = q * divisor + c' + c mod g``, a new name
    (added to ``facts``) that two values differing in ``c mod g`` share."""
    whole: list[tuple[str, Fraction]] = []
    rest: list[tuple[str, Fraction]] = []
    for name, coeff in value.terms:
        step = coeff * facts[name].multiple
        (whole if step.denominator == 1 and step % divisor == 0 else rest).append(
            (name, coeff)
        )
    base = Affine(Fraction(0), tuple(whole)).scale(Fraction(1, divisor))
    lo, hi = _range(Affine(value.const, tuple(rest)), facts)
    if lo is not None and hi is not None:
        quotient = math.floor(lo / divisor)
        if math.floor(hi / divisor) == quotient:
            return base + Affine.of(quotient)
    steps = [coeff * facts[name].multiple for name, coeff in rest]
    if value.const.denominator != 1 or any(st.denominator != 1 for st in steps):
        return None
    g = math.gcd(*(int(st) for st in steps))
    const = int(value.const)
    quotient, kept = divmod(const - const % g, divisor)
    if divisor % g:
        return None
    inner = Affine(Fraction(kept), tuple(rest))
    lo, hi = _range(inner, facts)
    if lo is None or lo < 0:
        return None
    quotient_of = ast.BinOp(render(inner), ast.Div(), ast.Constant(divisor))
    name = f"({ast.unparse(quotient_of)})"
    facts[name] = Fact(
        1,
        math.floor(lo / divisor),
        None if hi is None else math.floor(hi / divisor),
        frozenset().union(*({n} | facts[n].depends for n, _ in rest)),
    )
    return base + Affine.of(quotient) + Affine.term(name)


def remainder(value: Affine, divisor: int, facts: dict[str, Fact]) -> Affine | None:
    """``value % divisor``: ``value - divisor * (value // divisor)``."""
    quotient = floor_divide(value, divisor, facts)
    if quotient is None:
        return None
    return value + quotient.scale(Fraction(-divisor))


#: Calls the walker prints as floor division and remainder.
_FLOOR_DIVIDE = "c10.metal.floor_divide"
_REMAINDER = "c10.metal.remainder"


def _callee(node: ast.AST) -> str | None:
    """A call's callee as written (``c10.metal.floor_divide``)."""
    if isinstance(node, ast.Call):
        return ast.unparse(node.func)
    return None


class _Evaluator:
    def __init__(self, facts: dict[str, Fact], env: dict[str, object]) -> None:
        self.facts = facts
        self.env = env
        #: Whether the value evaluated last read a signed ``int``: a
        #: ``static_cast<int>`` or a name assigned from one.
        self.signed = False
        #: The names assigned a signed value.
        self.signed_names: set[str] = set()

    def nonnegative(self, value: Affine) -> bool:
        lo = _range(value, self.facts)[0]
        return lo is not None and lo >= 0

    def __call__(self, node: ast.AST) -> Affine | None:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Affine.of(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.signed_names:
                self.signed = True
            value = self.env.get(node.id)
            if isinstance(value, Affine):
                return value
            if value is None and node.id in self.facts:
                return Affine.term(node.id)
            return None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            vector = self.env.get(node.value.id)
            index = self(node.slice)
            if isinstance(vector, tuple) and index is not None:
                at = index.constant
                if at is not None and 0 <= at < len(vector):
                    return vector[at]
            return None
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        name = _callee(node)
        if (
            name == "static_cast"
            and isinstance(node, ast.Call)
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "int"
        ):
            # A tile index read as the signed index type: the names the fold
            # models (lane coordinates, element positions, tile offsets) are
            # int index values, which the cast leaves as they are.  Like every
            # + and * the fold rewrites, this assumes the index arithmetic
            # doesn't overflow 32 bits (a view that may reach 2**31 computes
            # its offsets in long, which the fold leaves alone).  The value
            # stays signed: :func:`fold` renders it as an int.
            self.signed = True
            return self(node.args[1])
        if name in (_FLOOR_DIVIDE, _REMAINDER):
            assert isinstance(node, ast.Call)
            value, divisor = (self(a) for a in node.args)
            d = None if divisor is None else divisor.constant
            if value is None or d is None or d <= 0:
                return None
            fold = floor_divide if name == _FLOOR_DIVIDE else remainder
            return fold(value, d, self.facts)
        return None

    def _binop(self, node: ast.BinOp) -> Affine | None:
        left, right = self(node.left), self(node.right)
        if left is None or right is None:
            return None
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left + right.scale(Fraction(-1))
        if isinstance(op, ast.Mult):
            if left.constant is not None:
                return right.scale(Fraction(left.constant))
            if right.constant is not None:
                return left.scale(Fraction(right.constant))
            return None
        d = right.constant
        if d is None or d <= 0:
            return None
        # C's integer / and % are floor division and remainder at a
        # dividend that can't be negative.
        if isinstance(op, (ast.Div, ast.FloorDiv)) and (
            isinstance(op, ast.FloorDiv) or self.nonnegative(left)
        ):
            return floor_divide(left, d, self.facts)
        if isinstance(op, ast.Mod) and self.nonnegative(left):
            return remainder(left, d, self.facts)
        if isinstance(op, ast.BitAnd) and left.constant is not None:
            return Affine.of(left.constant & d)
        if isinstance(op, ast.RShift) and left.constant is not None:
            return Affine.of(left.constant >> d)
        return None


def _divides(node: ast.AST) -> bool:
    """Whether ``node`` divides or takes a remainder."""
    return any(
        isinstance(n, ast.BinOp)
        and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod))
        or _callee(n) in (_FLOOR_DIVIDE, _REMAINDER)
        for n in ast.walk(node)
    )


def render(value: Affine) -> ast.expr:
    """``value`` as an expression: each term ``name * a`` or, with ``a =
    p / q``, ``name / q * p`` (an exact division: ``q`` divides the name's
    multiple), then the constant."""
    parts: list[str] = []
    # Positive terms first, so a leading term needs no sign.
    for name, coeff in sorted(value.terms, key=lambda t: t[1] < 0):
        term = name if coeff.denominator == 1 else f"({name} / {coeff.denominator})"
        if abs(coeff.numerator) != 1:
            term = f"{term} * {abs(coeff.numerator)}"
        parts.append(("- " if coeff < 0 else "+ ") + term)
    const = int(value.const)
    if const or not parts:
        parts.append(("- " if const < 0 else "+ ") + str(abs(const)))
    text = " ".join(parts)
    text = text[2:] if text.startswith("+ ") else f"0 {text}"
    return ast.parse(f"({text})", mode="eval").body


def evaluate(
    node: ast.AST, env: Mapping[str, object], facts: dict[str, Fact]
) -> Affine | None:
    """``node``'s affine value over ``env`` and ``facts``, or None;
    ``facts`` gains the facts of the floor terms in it."""
    return _Evaluator(facts, dict(env))(node)


def fold(
    stmts: list[ast.stmt], env: dict[str, object], facts: dict[str, Fact]
) -> list[ast.stmt]:
    """``stmts`` with each assignment or element binding whose value divides
    or takes a remainder, and is affine over ``env`` (names' affine values,
    or tuples of them for a ``uint3``) and ``facts``, rewritten to its
    folded form.  ``env`` gains the statements' affine values, and ``facts``
    the facts of the floor terms in them (``floor_divide``)."""
    evaluate = _Evaluator(facts, env)
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.Expr) and _callee(stmt.value) == BIND:
            call = stmt.value
            assert isinstance(call, ast.Call)
            name = ast.literal_eval(call.args[0])
            value = call.args[1]
            if _callee(value) == "uint3":
                assert isinstance(value, ast.Call)
                parts = tuple(evaluate(a) for a in value.args)
                env[name] = parts
                evaluate.signed_names.discard(name)
                if not any(_divides(a) for a in value.args):
                    out.append(stmt)
                    continue
                args = [
                    render(v) if v is not None and _divides(a) else a
                    for a, v in zip(value.args, parts, strict=True)
                ]
                out.append(_rebind(stmt, name, ast.Call(value.func, args, [])))
                continue
            folded = _value(evaluate, name, value)
            env[name] = folded
            if folded is not None and _divides(value):
                out.append(_rebind(stmt, name, _render(folded, evaluate.signed)))
                continue
        elif (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            name = stmt.targets[0].id
            folded = _value(evaluate, name, stmt.value)
            env[name] = folded
            if folded is not None and _divides(stmt.value):
                value = _render(folded, evaluate.signed)
                new = ast.Assign([ast.Name(name, ast.Store())], value)
                out.append(ast.fix_missing_locations(ast.copy_location(new, stmt)))
                continue
        out.append(stmt)
    return out


def _value(evaluate: _Evaluator, name: str, node: ast.expr) -> Affine | None:
    """``node``'s affine value, assigned to ``name``; ``name`` is signed
    when the value is."""
    evaluate.signed = False
    value = evaluate(node)
    if evaluate.signed:
        evaluate.signed_names.add(name)
    else:
        evaluate.signed_names.discard(name)
    return value


def _render(value: Affine, signed: bool) -> ast.expr:
    """``value`` rendered, as an ``int`` when it was signed: its terms may be
    unsigned (a tile offset is a ``uint``), over which a negative value, an
    index before the tile's start, would wrap."""
    rendered = render(value)
    if not signed or value.constant is not None:
        return rendered
    return ast.Call(
        ast.Name("static_cast", ast.Load()), [ast.Name("int", ast.Load()), rendered], []
    )


def _rebind(stmt: ast.stmt, name: str, value: ast.expr) -> ast.stmt:
    call = ast.Call(ast.Name(BIND, ast.Load()), [ast.Constant(name), value], [])
    return ast.fix_missing_locations(ast.copy_location(ast.Expr(call), stmt))
