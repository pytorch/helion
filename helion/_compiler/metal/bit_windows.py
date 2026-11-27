"""Read a lane's run of packed codes from one window of words.

A kernel that packs codes of ``b`` bits little-endian into a tensor of
``W``-bit words (``W`` = 8 or 16) reads the code at bit ``X`` of a row from
the word that holds its first bit and the next one, which the code straddles
where ``X % W + b > W`` (a 5-bit code 4 times in 16, a 6-bit code 2 in 8)::

    lo = t[row + X // W]
    hi = t[row + min(X // W + 1, C)]  # or t[row + X // W + 1]
    q = ((zext(lo) | (zext(hi) << W)) >> (X % W)) & (2**b - 1)

``(zext(lo) | (zext(hi) << W)) >> (X % W)`` holds the row's bits from ``X`` on,
so ``q`` is the row's bits ``[X, X + b)``.  The elements a lane holds of a
register operand (``mpp_registers``) read codes at adjacent K positions, at
bits ``X_0 + c_j`` for constants ``c_j``.  Where the fold can't reduce ``X_0 //
W`` and ``X_0 % W`` (an M5 lane's 4 adjacent K positions start at a multiple
of 4, 20 or 24 bits apart for 5- or 6-bit codes), every element pays a
division, a remainder, two loads and a variable shift.  The run's codes lie
within the row's bits ``[X_0, X_0 + c_max + b)``, which the ``n = ceil((s +
c_max + b) / W)`` words from word ``X_0 // W`` hold, ``s`` the most ``X_0 % W``
takes over the facts: the run loads them once into a window shifted right by
``X_0 % W`` (:func:`window_lines`), and each element's code is ``(window >>
c_j) & (2**b - 1)`` (:func:`window_code`).

The window loads no word before the first code's ``lo`` or past the last
code's ``hi``, and those two as the codes load them: its first is the first
code's ``lo``; a word past the last code's ``lo`` (one at most, or the run
keeps its loads) is that code's ``hi``, clamped as it is.  A word between
codes that aren't adjacent, which no code reads, lies between words they
read, inside the row.  Where
the codes clamp ``hi``, a word may be one code's ``hi`` and another lane's
code's ``lo``, read clamped and unclamped: the facts prove that no code's
``lo`` lies past ``C``, so both read the same word.
"""

from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction
import math
from typing import TYPE_CHECKING
from typing import cast

from .affine_fold import FLOOR_DIVIDE
from .affine_fold import REMAINDER
from .affine_fold import Affine
from .affine_fold import evaluate
from .affine_fold import render
from .affine_fold import value_range
from .lane_fold import MASKED_LOAD
from .lane_fold import load_mask_other
from .lane_fold import tl_load
from .mpp_matmul import BIND
from .mpp_matmul import call_name
from .mpp_matmul import names_read

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from .affine_fold import Fact

#: A word type's bits, whether its integer conversion is signed, and its
#: unsigned type, by the MSL type of the tensor's elements.
_WORDS = {
    "short": (16, True, "ushort"),
    "ushort": (16, False, "ushort"),
    "char": (8, True, "uchar"),
    "uchar": (8, False, "uchar"),
}
#: The widest window: a ``ulong``.
_WINDOW_BITS = 64


@dataclasses.dataclass(frozen=True)
class _Floor:
    """The index ``base + bit // width``."""

    base: Affine
    bit: Affine
    width: int


@dataclasses.dataclass(frozen=True)
class _Next:
    """The index ``base + min(bit // width + 1, clamp)``, or ``base + bit //
    width + 1`` without a clamp."""

    base: Affine
    bit: Affine
    width: int
    clamp: Affine | None


@dataclasses.dataclass(frozen=True)
class _Rem:
    """``bit % width``."""

    bit: Affine
    width: int


@dataclasses.dataclass(frozen=True)
class _Load:
    """A word of ``tensor`` read at ``index``, as the tensor's type."""

    tensor: str
    index: _Floor | _Next


@dataclasses.dataclass(frozen=True)
class _Word:
    """A loaded word converted to an integer: zero-extended, or
    sign-extended from a signed type."""

    load: _Load
    zext: bool


@dataclasses.dataclass(frozen=True)
class _High:
    """A zero-extended word shifted left by its width."""

    load: _Load


@dataclasses.dataclass(frozen=True)
class _Pair:
    """``zext(lo) | (zext(hi) << W)`` of a word and the next."""

    lo: _Load
    hi: _Load


@dataclasses.dataclass(frozen=True)
class _Stream:
    """A pair shifted right by its first word's bit: the row's bits from
    ``bit`` on."""

    pair: _Pair


@dataclasses.dataclass(frozen=True)
class Code:
    """A packed code an element's statement ``name = ...`` computes: bits
    ``[bit, bit + bits)`` of the row of ``tensor`` (words of ``width`` bits,
    ``word`` their unsigned type) whose first word is ``row``, read from the
    word holding its first bit and the next, that one's index clamped to
    ``clamp`` (row-relative; None: unclamped)."""

    name: str
    tensor: str
    word: str
    width: int
    row: Affine
    bit: Affine
    bits: int
    clamp: Affine | None


def _uncast(node: ast.expr) -> ast.expr:
    """``x`` of ``static_cast(T, x)`` where ``T`` is ``int`` or the type of
    ``c10.metal.min``'s operands' sum (``decltype(a + b)``), which it takes
    them through."""
    while (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "static_cast"
        and len(node.args) == 2
        and (
            ast.unparse(node.args[0]) == "int"
            or ast.unparse(node.args[0]).startswith("decltype(")
        )
    ):
        node = node.args[1]
    return node


def _broadcast(node: ast.Subscript) -> bool:
    """Whether a subscript only reshapes (``v[None, :]``)."""
    index = node.slice
    elts = index.elts if isinstance(index, ast.Tuple) else [index]
    return bool(elts) and all(
        (isinstance(e, ast.Slice) and e.lower is None and e.upper is None)
        or (isinstance(e, ast.Constant) and e.value is None)
        for e in elts
    )


class _Finder:
    """The symbolic values of an element's statements (:func:`find_codes`)."""

    def __init__(
        self,
        env: Mapping[str, object],
        facts: dict[str, Fact],
        types: Mapping[str, str],
    ) -> None:
        # The fold's affine values, and those of the names the fold leaves
        # but the finder resolves (a select on a mask that always holds).
        self.env = dict(env)
        # Gains the facts of the floor terms its values hold.
        self.facts = facts
        self.types = types
        self.values: dict[str, object] = {}

    def affine(self, node: ast.expr) -> Affine | None:
        return evaluate(node, self.env, self.facts)

    def __call__(self, node: ast.expr) -> object:
        value = self.affine(node)
        if value is not None:
            return value
        if isinstance(node, ast.Name):
            return self.values.get(node.id)
        if isinstance(node, ast.Subscript) and _broadcast(node):
            return self(node.value)
        if isinstance(node, ast.IfExp):
            test = self.truth(node.test)
            return None if test is None else self(node.body if test else node.orelse)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return self.truth(node)
        return None

    def truth(self, node: ast.expr) -> bool | None:
        """Whether a condition holds at every position the facts allow
        (True), at none (False), or None where that isn't known; a tile
        mask over whole tiles holds at every element (True)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            value = self.values.get(node.id)
            return value if isinstance(value, bool) else None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            parts = [self.truth(v) for v in node.values]
            return False if False in parts else None if None in parts else True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            parts = [self.truth(node.left), self.truth(node.right)]
            return False if False in parts else None if None in parts else True
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = self.affine(node.left), self.affine(node.comparators[0])
            if a is None or b is None:
                return None
            op = node.ops[0]
            # b - a against 0: a < b is 0 < b - a.
            lo, hi = value_range(b + a.scale(Fraction(-1)), self.facts)
            if isinstance(op, (ast.Gt, ast.GtE)):
                lo, hi = (None if hi is None else -hi), (None if lo is None else -lo)
            strict = isinstance(op, (ast.Lt, ast.Gt))
            if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                return None
            if lo is not None and (lo > 0 if strict else lo >= 0):
                return True
            if hi is not None and (hi <= 0 if strict else hi < 0):
                return False
        return None

    def _call(self, node: ast.Call) -> object:
        callee = ast.unparse(node.func)
        if callee in (FLOOR_DIVIDE, REMAINDER):
            bit = self.affine(node.args[0])
            width = self.affine(node.args[1])
            if bit is None or width is None or width.constant not in (8, 16):
                return None
            if callee == REMAINDER:
                return _Rem(bit, width.constant)
            return _Floor(Affine.of(0), bit, width.constant)
        if callee == "c10.metal.min" and len(node.args) == 2:
            a, b = (self(_uncast(arg)) for arg in node.args)
            for index, clamp in ((a, b), (b, a)):
                if (
                    isinstance(index, _Floor)
                    and index.base == Affine.of(1)
                    and isinstance(clamp, Affine)
                ):
                    return _Next(Affine.of(0), index.bit, index.width, clamp)
            return None
        if callee == MASKED_LOAD and len(node.args) == 3:
            # A load made unconditional, then selected: its own value where
            # its mask holds everywhere.
            return self(node.args[0]) if self.truth(node.args[1]) else None
        if callee == "static_cast" and len(node.args) == 2:
            target = ast.unparse(node.args[0])
            value = self(node.args[1])
            if isinstance(value, _Load) and target == "int":
                return _Word(value, not _WORDS[self.types[value.tensor]][1])
            if isinstance(value, (_Floor, _Next, _Rem)) and target == "int":
                return value
            return None
        if (load := tl_load(node)) is not None and (
            (masked := load_mask_other(load)) is None or self.truth(masked[0])
        ):
            ptr = load.args[0]
            if not (
                isinstance(ptr, ast.BinOp)
                and isinstance(ptr.op, ast.Add)
                and isinstance(ptr.left, ast.Name)
                and self.types.get(ptr.left.id) in _WORDS
            ):
                return None
            index = self(ptr.right)
            if isinstance(index, (_Floor, _Next)):
                tensor = ptr.left.id
                if index.width != _WORDS[self.types[tensor]][0]:
                    return None
                return _Load(tensor, index)
        return None

    def _binop(self, node: ast.BinOp) -> object:
        left, right = self(node.left), self(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            for index, offset in ((left, right), (right, left)):
                if isinstance(offset, Affine) and isinstance(index, _Floor):
                    return dataclasses.replace(index, base=index.base + offset)
                if isinstance(offset, Affine) and isinstance(index, _Next):
                    return dataclasses.replace(index, base=index.base + offset)
            return None
        if isinstance(op, ast.Mult):
            for index, factor in ((left, right), (right, left)):
                if isinstance(factor, Affine) and factor.constant == 1:
                    return index if isinstance(index, (_Floor, _Next)) else None
            return None
        if isinstance(op, ast.BitAnd):
            for value, mask in ((left, right), (right, left)):
                bits = _mask_bits(mask)
                if bits is None:
                    continue
                if isinstance(value, _Word):
                    width = _WORDS[self.types[value.load.tensor]][0]
                    return _Word(value.load, True) if bits == width else None
                if isinstance(value, _Stream):
                    return self._code(value, bits)
            return None
        if isinstance(op, ast.LShift):
            if (
                isinstance(left, _Word)
                and left.zext
                and isinstance(right, Affine)
                and right.constant == _WORDS[self.types[left.load.tensor]][0]
            ):
                return _High(left.load)
            return None
        if isinstance(op, ast.BitOr):
            for lo, hi in ((left, right), (right, left)):
                if isinstance(lo, _Word) and lo.zext and isinstance(hi, _High):
                    pair = _pair(lo.load, hi.load)
                    if pair is not None:
                        return pair
            return None
        if isinstance(op, ast.RShift):
            if (
                isinstance(left, _Pair)
                and isinstance(right, _Rem)
                and isinstance(left.lo.index, _Floor)
                and right.bit == left.lo.index.bit
                and right.width == left.lo.index.width
            ):
                return _Stream(left)
            return None
        return None

    def _code(self, stream: _Stream, bits: int) -> Code | None:
        lo, hi = stream.pair.lo, stream.pair.hi
        assert isinstance(lo.index, _Floor) and isinstance(hi.index, _Next)
        width, _, word = _WORDS[self.types[lo.tensor]]
        # The pair holds at least W + 1 bits from the code's first bit on.
        if bits > width + 1:
            return None
        return Code(
            "",
            lo.tensor,
            word,
            width,
            _normal(lo.index.base),
            _normal(lo.index.bit),
            bits,
            None if hi.index.clamp is None else _normal(hi.index.clamp),
        )


def _normal(value: Affine) -> Affine:
    """``value`` with its terms in order, so equal values compare equal."""
    return Affine.of(0) + value


def _mask_bits(value: object) -> int | None:
    """``b`` of a constant mask ``2**b - 1``, ``b`` at least 1."""
    if not isinstance(value, Affine) or value.constant is None:
        return None
    mask = value.constant
    if mask <= 0 or mask & (mask + 1):
        return None
    return mask.bit_length()


def _pair(lo: _Load, hi: _Load) -> _Pair | None:
    """The pair of words ``lo`` and ``hi`` when ``hi`` is the word after
    ``lo``'s, clamped or not, in the same row of the same tensor."""
    if lo.tensor != hi.tensor or not isinstance(lo.index, _Floor):
        return None
    index = hi.index
    if isinstance(index, _Floor):
        # Unclamped: base + bit // W + 1.
        index = _Next(index.base + Affine.of(-1), index.bit, index.width, None)
    if (
        index.base != lo.index.base
        or index.bit != lo.index.bit
        or index.width != lo.index.width
    ):
        return None
    return _Pair(lo, dataclasses.replace(hi, index=index))


def find_codes(
    body: list[ast.stmt],
    env: Mapping[str, object],
    facts: dict[str, Fact],
    types: Mapping[str, str],
) -> list[Code]:
    """The packed codes an element's folded statements ``body`` compute:
    ``env`` holds their names' affine values and ``facts`` the facts those
    fold over (``operand_elements.element_body``), which gains the facts of
    the floor terms in the codes; ``types`` holds the tensors' MSL element
    types.  A code whose bit position the fold reduced to a constant shift
    isn't one: its loads are already shared."""
    finder = _Finder(env, facts, types)
    codes: list[Code] = []
    for stmt in body:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name = stmt.targets[0].id
        value = finder(stmt.value)
        if isinstance(value, Code):
            codes.append(dataclasses.replace(value, name=name))
            value = None
        if isinstance(value, Affine):
            finder.env[name] = value
        finder.values[name] = value
    return codes


@dataclasses.dataclass(frozen=True)
class Window:
    """The words a run of codes reads (:func:`plan_windows`): ``count``
    words from the word holding bit ``first.bit`` of ``first``'s row, each
    clamped as the codes' next word is (``clamped``), held in ``name`` shifted
    right by that bit's offset in its word; each member, ``(element, code
    name, offset)``, is its code's bits ``[offset, offset + bits)`` of it."""

    name: str
    first: Code
    count: int
    clamped: tuple[bool, ...]
    members: tuple[tuple[int, str, int], ...]


def _residues(bit: Affine, width: int, facts: Mapping[str, Fact]) -> list[int] | None:
    """The values ``bit % width`` may take: each term's step (its
    coefficient times its name's multiple) moves it by a multiple of the
    steps' gcd with ``width``."""
    steps: list[int] = []
    for name, coeff in bit.terms:
        step = coeff * facts[name].multiple
        if step.denominator != 1:
            return None
        steps.append(int(step))
    if bit.const.denominator != 1:
        return None
    g = math.gcd(width, *steps)
    return list(range(int(bit.const) % g, width, g))


def plan_windows(
    codes: list[list[Code]], facts: Mapping[str, Fact], name: Callable[[], str]
) -> list[Window]:
    """The windows of a step's elements, ``codes[e]`` element ``e``'s codes
    (:func:`find_codes`), over ``facts``: one for each run of codes in one
    row of one tensor whose bits differ by constants, ``name()`` naming
    each.  A run is left to its elements where its window wouldn't load
    fewer words than they do (two a code) or fit a ``ulong``, or would load
    a word past the last code's ``hi``, or clamp a word a code reads
    unclamped."""
    runs: dict[tuple[object, ...], list[tuple[int, Code]]] = {}
    for e, element in enumerate(codes):
        for code in element:
            key = (
                code.tensor,
                code.width,
                code.bits,
                code.row,
                code.clamp,
                code.bit.terms,
            )
            runs.setdefault(key, []).append((e, code))
    windows: list[Window] = []
    for run in runs.values():
        if any(c.bit.const.denominator != 1 for _, c in run):
            continue
        first = min((c for _, c in run), key=lambda c: c.bit.const)
        offsets = [int(c.bit.const - first.bit.const) for _, c in run]
        width, last = first.width, max(offsets)
        lo, hi = value_range(first.bit, facts)
        residues = _residues(first.bit, width, facts)
        if lo is None or lo < 0 or not residues:
            continue
        count = math.ceil((max(residues) + last + first.bits) / width)
        # Fewer loads than the codes' own two each, in a ulong at most.
        if count * width > _WINDOW_BITS or count >= 2 * len(run):
            continue
        # Each residue's last code's lo is word (s + last) // W of the window;
        # its hi the next.  Words past every lane's last lo are some lane's
        # last hi, which the window reads as that code does.
        if any(count - 1 > (s + last) // width + 1 for s in residues):
            continue
        within = min((s + last) // width for s in residues)
        clamped = tuple(first.clamp is not None and i > within for i in range(count))
        if first.clamp is not None and count > 1:
            # A word the window reads unclamped may be some code's next word,
            # and one it clamps another lane's code's first word, which the
            # codes read clamped and unclamped: both read the same word where
            # no code's first word lies past the clamp.
            least = value_range(first.clamp, facts)[0]
            if hi is None or least is None or (hi + last) // width > least:
                continue
        windows.append(
            Window(
                name(),
                first,
                count,
                clamped,
                tuple(
                    (e, c.name, off) for (e, c), off in zip(run, offsets, strict=True)
                ),
            )
        )
    return windows


def window_lines(window: Window, to_msl: Callable[[ast.expr], str]) -> list[str]:
    """The MSL that loads ``window``'s words into its name, shifted right by
    its first code's bit offset in its first word."""
    first, name = window.first, window.name
    width = first.width
    utype = "uint" if window.count * width <= 32 else "ulong"
    x, w, p = f"{name}_x", f"{name}_w", f"{name}_p"
    words = []
    for i, clamped in enumerate(window.clamped):
        index = w if i == 0 else f"{w} + {i}"
        if clamped:
            assert first.clamp is not None
            index = f"metal::min({index}, {to_msl(render(first.clamp))})"
        word = f"{utype}({p}[{index}])"
        words.append(word if i == 0 else f"({word} << {i * width})")
    shift = width.bit_length() - 1
    pointer = f"const device {first.word}*"
    row = to_msl(render(first.row))
    return [
        f"const int {x} = {to_msl(render(first.bit))};",
        f"const int {w} = {x} >> {shift};",
        f"{pointer} {p} = ({pointer})({first.tensor}) + ({row});",
        f"const {utype} {name} = ({' | '.join(words)}) >> ({x} & {width - 1});",
    ]


def window_code(window: Window, offset: int) -> ast.expr:
    """A member's code at ``offset`` bits of its window, as an ``int``."""
    mask = (1 << window.first.bits) - 1
    return ast.parse(
        f"static_cast(int, ({window.name} >> {offset}) & {mask})", mode="eval"
    ).body


def rewrite(body: list[ast.stmt], codes: Mapping[str, ast.expr]) -> list[ast.stmt]:
    """An element's statements ``body`` with each assignment to a name in
    ``codes`` assigning its window's code instead, and the assignments and
    bindings nothing after reads dropped: the codes' loads, their indices
    and words.  An element's statements recompute an operand's value
    (``operand_elements.OperandRecomputable``): every assignment is pure."""
    out: list[ast.stmt] = []
    for stmt in body:
        name = _target(stmt)
        if isinstance(stmt, ast.Assign) and name in codes:
            new = ast.Assign([ast.Name(name, ast.Store())], codes[name])
            stmt = ast.fix_missing_locations(ast.copy_location(new, stmt))
        out.append(stmt)
    live: set[str] = set()
    kept: list[ast.stmt] = []
    for stmt in reversed(out):
        name = _target(stmt)
        if name is not None and name not in live:
            continue
        kept.append(stmt)
        live |= names_read(stmt)
    return kept[::-1]


def _target(stmt: ast.stmt) -> str | None:
    """The name a single-name assignment or an element binding assigns."""
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and call_name(stmt.value) == BIND
    ):
        return cast("str", ast.literal_eval(stmt.value.args[0]))
    return None
