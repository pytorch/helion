"""Replace a value computed from one small integer by a table lookup.

An element of a fragment run (``fragment_mma``) decodes its operand from a
packed code: ``q = (w >> s) & 0xF`` picks a nibble, and a chain of integer
and float ops turns it into the code's value (half bits, a bitcast, a
conversion).  Every op of that chain runs per element, though ``q`` takes
only 16 values.  When a statement masks an integer with a constant ``m =
2**b - 1`` (``q = x & m``, at most the config's
``metal_tabulate_max_entries`` values), every value
computed from ``q`` and constants alone is a function of ``q``'s ``m + 1``
values: codegen evaluates it at each of them and reads the result from a
``constant`` table indexed by ``q``.

The evaluation follows MSL's semantics for the ops it knows: integer
arithmetic on values in ``[0, 2**31)`` (anything outside, where an int and a
uint would differ, keeps the chain), integral casts wrapping to their width,
bit casts between equal widths, float conversions rounding to nearest even,
and float arithmetic in the operands' type.  A chain with any other op is
kept as written.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
from fractions import Fraction
import functools
import math
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch

from .mpp_matmul import assign_target
from .mpp_matmul import assigned_names
from .search_surface import METAL_TABULATE_CHOICES

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

#: Most entries a table may have (``metal_tabulate_max_entries``).
MAX_TABLE = METAL_TABULATE_CHOICES[-1]

#: Integral types: (bits, signed).
_INTEGRAL = {
    "char": (8, True),
    "uchar": (8, False),
    "short": (16, True),
    "ushort": (16, False),
    "int": (32, True),
    "uint": (32, False),
}
_FLOATS = {"half": 16, "float": 32, "bfloat": 16}
#: Calls whose first argument is a type.
_CASTS = ("static_cast", "as_type")


@dataclasses.dataclass(frozen=True)
class _Value:
    """An evaluated value: a Python int with its integral type (``None``: an
    integer whose type the chain doesn't fix, in ``[0, 2**31)``), or a float
    with its float type."""

    value: int | float
    ctype: str | None


def _bf16(x: float) -> float:
    """``x`` rounded to the nearest bfloat, ties to even."""
    if not math.isfinite(x):
        # The rounding below would carry a NaN's mantissa into its sign.
        raise _Unknown
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _to_float(x: float, ctype: str) -> float:
    """``x`` rounded to ``ctype``; past its range, inf, as the GPU gives."""
    with np.errstate(over="ignore"):
        if ctype == "float":
            return float(np.float32(x))
        if ctype == "half":
            return float(np.float16(x))
        return _bf16(float(np.float32(x)))


def _bits(v: _Value) -> int:
    """The bits of a float value."""
    assert isinstance(v.value, float)
    if v.ctype == "float":
        return struct.unpack("<I", struct.pack("<f", v.value))[0]
    if v.ctype == "half":
        return int(np.array([v.value], dtype=np.float16).view(np.uint16)[0])
    return struct.unpack("<I", struct.pack("<f", v.value))[0] >> 16


def _from_bits(bits: int, ctype: str) -> float:
    if ctype == "float":
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    if ctype == "half":
        return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _wrap(x: int, ctype: str) -> int:
    bits, signed = _INTEGRAL[ctype]
    x &= (1 << bits) - 1
    return x - (1 << bits) if signed and x >> (bits - 1) else x


def _round(x: Fraction, ctype: str) -> float:
    """``x`` rounded once to ``ctype``, to nearest, ties to even."""
    dtype = {"float": np.float32, "half": np.float16}[ctype]
    with np.errstate(over="ignore"):
        near = dtype(float(x)) if abs(x) < 2**1024 else dtype(np.inf)
        if not np.isfinite(near):
            # Past the type's range: the chain keeps computing its inf.
            raise _Unknown
        candidates = [
            near,
            np.nextafter(near, dtype(-np.inf)),
            np.nextafter(near, dtype(np.inf)),
        ]
    finite = [c for c in candidates if np.isfinite(c)]

    def key(c: np.floating) -> tuple[Fraction, int]:
        bits = int(np.array([c]).view(np.uint32 if ctype == "float" else np.uint16)[0])
        return abs(Fraction(float(c)) - x), bits & 1

    return float(min(finite, key=key))


def _type_name(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


class _Unknown(Exception):
    """The chain has an op or a value the evaluation doesn't model."""


class _Evaluate:
    def __init__(self, values: Mapping[str, _Value]) -> None:
        self.values = values

    def __call__(self, node: ast.expr) -> _Value:
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return self._int(node.value, None)
            if type(node.value) is float:
                return _Value(_to_float(node.value, "float"), "float")
            raise _Unknown
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.values[node.id]
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("metal.fma", "fma")
            and len(node.args) == 3
        ):
            return self._fma(*(self(a) for a in node.args))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and len(node.args) == 2
            and (ctype := _type_name(node.args[0])) is not None
        ):
            if node.func.id == "static_cast":
                return self._cast(self(node.args[1]), ctype)
            if node.func.id == "as_type":
                return self._bitcast(self(node.args[1]), ctype)
        raise _Unknown

    @staticmethod
    def _int(x: int, ctype: str | None) -> _Value:
        if ctype is None and not 0 <= x < 2**31:
            raise _Unknown
        return _Value(x, ctype)

    def _binop(self, node: ast.BinOp) -> _Value:
        left, right = self(node.left), self(node.right)
        op = type(node.op)
        if isinstance(left.value, int) and isinstance(right.value, int):
            # Nonnegative operands and a result in [0, 2**31): the same in an
            # int and a uint, whatever the operands' types promote to.
            a, b = left.value, right.value
            if a < 0 or b < 0:
                raise _Unknown
            ints: dict[type[ast.operator], Callable[[], int]] = {
                ast.BitAnd: lambda: a & b,
                ast.BitOr: lambda: a | b,
                ast.BitXor: lambda: a ^ b,
                ast.Add: lambda: a + b,
                ast.Sub: lambda: a - b,
                ast.Mult: lambda: a * b,
                ast.LShift: lambda: a << b if b < 32 else -1,
                ast.RShift: lambda: a >> b if b < 32 else -1,
            }
            if op not in ints:
                raise _Unknown
            return self._int(ints[op](), None)
        if not (isinstance(left.value, float) and isinstance(right.value, float)):
            raise _Unknown
        if "bfloat" in (left.ctype, right.ctype):
            raise _Unknown
        ctype = "half" if left.ctype == right.ctype == "half" else "float"
        a, b = (_to_float(v.value, ctype) for v in (left, right))
        if op is ast.Add:
            result = a + b
        elif op is ast.Sub:
            result = a - b
        elif op is ast.Mult:
            result = a * b
        elif op is ast.Div and b != 0:
            result = a / b
        else:
            raise _Unknown
        return _Value(_to_float(result, ctype), ctype)

    @staticmethod
    def _fma(a: _Value, b: _Value, c: _Value) -> _Value:
        """``a * b + c`` with one rounding."""
        if not all(
            isinstance(v.value, float)
            and math.isfinite(v.value)
            and v.ctype in ("float", "half")
            for v in (a, b, c)
        ):
            raise _Unknown
        ctype = "half" if a.ctype == b.ctype == c.ctype == "half" else "float"
        exact = Fraction(a.value) * Fraction(b.value) + Fraction(c.value)
        if exact == 0:
            # Fractions have no -0: an exact zero is -0 only as the sum of
            # two -0s (a zero product's sign is its factors' signs').
            negative = (
                a.value * b.value == 0
                and c.value == 0
                and math.copysign(1.0, a.value) * math.copysign(1.0, b.value) < 0
                and math.copysign(1.0, c.value) < 0
            )
            return _Value(-0.0 if negative else 0.0, ctype)
        return _Value(_round(exact, ctype), ctype)

    def _cast(self, v: _Value, ctype: str) -> _Value:
        if ctype in _INTEGRAL:
            if not isinstance(v.value, int):
                raise _Unknown
            return _Value(_wrap(v.value, ctype), ctype)
        if ctype not in _FLOATS:
            raise _Unknown
        return _Value(_to_float(float(v.value), ctype), ctype)

    def _bitcast(self, v: _Value, ctype: str) -> _Value:
        if ctype in _FLOATS and isinstance(v.value, int):
            if v.ctype is None or _INTEGRAL[v.ctype][0] != _FLOATS[ctype]:
                raise _Unknown
            bits = v.value & ((1 << _FLOATS[ctype]) - 1)
            value = _from_bits(bits, ctype)
            if math.isnan(value):
                # A host float may not keep a NaN's payload bits, which a
                # bitcast back to an integer would read.
                raise _Unknown
            return _Value(value, ctype)
        if ctype in _INTEGRAL and isinstance(v.value, float):
            assert v.ctype is not None
            if _FLOATS[v.ctype] != _INTEGRAL[ctype][0] or math.isnan(v.value):
                # A NaN's bits are the host's, not necessarily the GPU's.
                raise _Unknown
            return _Value(_wrap(_bits(v), ctype), ctype)
        raise _Unknown


def _literal(value: _Value) -> str:
    if isinstance(value.value, int):
        return str(value.value)
    if not math.isfinite(value.value):
        raise _Unknown
    return repr(value.value) + {"float": "f", "half": "h"}[str(value.ctype)]


def _reads(node: ast.AST) -> set[str]:
    """The names ``node`` reads as values: not a called function, nor a
    cast's type."""
    names: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Call):
            cast = isinstance(n.func, ast.Name) and n.func.id in _CASTS
            stack += n.args[1:] if cast else n.args
            stack += [k.value for k in n.keywords]
        else:
            stack.extend(ast.iter_child_nodes(n))
    return names


def _assignments(stmts: list[ast.stmt]) -> collections.Counter[str]:
    """How many times ``stmts`` assign each name (``mpp_matmul.assigned_names``,
    as ``fragment_mma.kernel_constants`` counts them).  A name assigned once
    is assigned only by its one statement."""
    return collections.Counter(
        name for stmt in stmts for n in ast.walk(stmt) for name in assigned_names(n)
    )


def _field_mask(m: object, max_entries: int) -> bool:
    """Whether ``m`` masks a low-bit field of at most ``max_entries``
    values: ``2**b - 1``."""
    return type(m) is int and 0 < m < max_entries and not m & (m + 1)


def _mask_size(
    stmt: ast.stmt, consts: Mapping[str, _Value], max_entries: int
) -> int | None:
    """``m + 1`` when ``stmt`` is ``q = x & m`` with ``m`` a constant
    ``2**b - 1`` below ``max_entries``: ``q`` picks a ``b``-bit field."""
    value = stmt.value if isinstance(stmt, ast.Assign) else None
    if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitAnd)):
        return None
    for side in (value.left, value.right):
        try:
            m = _Evaluate(consts)(side)
        except _Unknown:
            continue
        if _field_mask(m.value, max_entries):
            assert isinstance(m.value, int)
            return m.value + 1
    return None


def tabulate(
    stmts: list[ast.stmt],
    constants: Mapping[str, int | float],
    tables: dict[tuple[str, tuple[str, ...]], str],
    max_entries: int,
) -> list[ast.stmt]:
    """``stmts`` (straight-line) with each value computed from one integer
    masked to at most ``max_entries`` values and ``constants`` alone, and
    read by a statement that is not, read from a table instead.  ``tables``
    maps each table's element type and entries to its name, shared across
    calls; the caller declares them."""
    if max_entries <= 0:
        # No tables (the default): skip the scan, so a config without
        # ``metal_tabulate_max_entries`` emits the statements as computed.
        return stmts
    counts = _assignments(stmts)
    assigned = set(counts)
    once = {name for name, count in counts.items() if count == 1}
    # The names constant throughout ``stmts``: the kernel's, and those
    # assigned once from constants alone.
    consts = {
        name: _Value(v, None)
        if isinstance(v, int)
        else _Value(_to_float(v, "float"), "float")
        for name, v in constants.items()
        if name not in assigned
    }
    for stmt in stmts:
        name = assign_target(stmt)
        if name is None or name not in once:
            continue
        assert isinstance(stmt, ast.Assign)
        if not (_reads(stmt.value) - set(consts)):
            with contextlib.suppress(_Unknown):
                consts[name] = _Evaluate(consts)(stmt.value)
    out = list(stmts)
    i = 0
    while i < len(out):
        size = _mask_size(out[i], consts, max_entries)
        source = assign_target(out[i])
        if size is not None and source is not None and source in once:
            out = _tabulate_from(out, i, source, size, consts, once, tables)
        i += 1
    return out


def _tabulate_from(
    stmts: list[ast.stmt],
    at: int,
    source: str,
    size: int,
    consts: Mapping[str, _Value],
    once: set[str],
    tables: dict[tuple[str, tuple[str, ...]], str],
) -> list[ast.stmt]:
    """``stmts`` with the values computed from ``source`` (assigned at
    ``at``, an integer below ``size``) and ``consts`` alone read from
    tables."""
    # The statements computed from ``source`` and constants alone, by ops
    # the evaluation models (tried at ``source = 0``).
    cone: dict[str, int] = {}
    trial: dict[str, _Value] = {**consts, source: _Value(0, None)}
    for j in range(at + 1, len(stmts)):
        name = assign_target(stmts[j])
        if name is None or name not in once or name in consts:
            continue
        value = stmts[j].value  # type: ignore[attr-defined]
        reads = _reads(value) - set(consts)
        if not (reads and reads <= {source, *cone}):
            continue
        try:
            trial[name] = _Evaluate(trial)(value)
        except _Unknown:
            continue
        cone[name] = j
    outputs = [
        name
        for name in cone
        if any(
            name in _reads(s) and assign_target(s) not in cone
            for s in stmts[cone[name] + 1 :]
        )
    ]
    out = list(stmts)
    tabulated = []
    for name in outputs:
        key = _table(stmts, cone, name, source, size, consts)
        if key is None:
            continue
        table = tables.setdefault(key, f"_helion_table_{len(tables)}")
        lookup = ast.Subscript(
            ast.Name(table, ast.Load()), ast.Name(source, ast.Load()), ast.Load()
        )
        new = ast.Assign([ast.Name(name, ast.Store())], lookup)
        out[cone[name]] = ast.fix_missing_locations(
            ast.copy_location(new, stmts[cone[name]])
        )
        tabulated.append(name)
    if not tabulated:
        return stmts
    # Drop the chain's statements nothing reads any more.
    live = set().union(*(_reads(s) for s in out if assign_target(s) not in cone))
    for name, j in sorted(cone.items(), key=lambda t: -t[1]):
        if name in tabulated or name in live:
            live |= _reads(out[j])
        else:
            out[j] = ast.Pass()
    return [s for s in out if not isinstance(s, ast.Pass)]


def _table(
    stmts: list[ast.stmt],
    cone: Mapping[str, int],
    output: str,
    source: str,
    size: int,
    consts: Mapping[str, _Value],
) -> tuple[str, tuple[str, ...]] | None:
    """The element type and entries of ``output``'s table: its value at
    each ``source`` below ``size``.  None when its chain only copies
    ``source``, an entry isn't modeled or finite, or its type isn't fixed
    (an integer's promoted type, or a bfloat's literals).  Any chain of one
    op or more qualifies: whether a lookup beats the ops it replaces depends
    on the chain and the GPU, which the search's
    ``metal_tabulate_max_entries`` (0: no tables) decides by timing."""
    needed = {output}
    for name, j in reversed(list(cone.items())):
        if name in needed:
            needed |= _reads(stmts[j].value) & set(cone)  # type: ignore[attr-defined]
    chain = [(name, j) for name, j in cone.items() if name in needed]
    ops = sum(
        isinstance(n, (ast.BinOp, ast.Call))
        for _, j in chain
        for n in ast.walk(stmts[j].value)  # type: ignore[attr-defined]
    )
    if not ops:
        return None
    reads = set().union(*(_reads(stmts[j].value) for _, j in chain))  # type: ignore[attr-defined]
    return _entries(
        tuple((name, ast.unparse(stmts[j].value)) for name, j in chain),  # type: ignore[attr-defined]
        output,
        source,
        size,
        tuple(
            sorted(
                (name, consts[name], _key(consts[name]))
                for name in reads
                if name in consts
            )
        ),
    )


def _key(value: _Value) -> tuple[str | None, int]:
    """``value`` as a cache key: a float by its bits and type, as ``==``
    takes 0.0 for -0.0, whose tables differ (a sign bit cast to an integer)."""
    if isinstance(value.value, float):
        return value.ctype, _bits(value)
    return value.ctype, value.value


@functools.lru_cache(maxsize=1024)
def _entries(
    chain: tuple[tuple[str, str], ...],
    output: str,
    source: str,
    size: int,
    consts: tuple[tuple[str, _Value, tuple[str | None, int]], ...],
) -> tuple[str, tuple[str, ...]] | None:
    """:func:`_table`'s result for the chain ``(name, expression)``,
    evaluated over the constants it reads (each with its :func:`_key`);
    cached, as every element of every step of a fragment run decodes through
    the same chain."""
    exprs = [(name, ast.parse(src, mode="eval").body) for name, src in chain]
    entries: list[_Value] = []
    known = {name: value for name, value, _ in consts}
    try:
        for q in range(size):
            values: dict[str, _Value] = {**known, source: _Value(q, None)}
            for name, expr in exprs:
                values[name] = _Evaluate(values)(expr)
            entries.append(values[output])
        ctypes = {v.ctype for v in entries}
        if len(ctypes) != 1 or None in ctypes or "bfloat" in ctypes:
            return None
        return str(entries[0].ctype), tuple(_literal(v) for v in entries)
    except _Unknown:
        return None


#: Ops of a chain in a kernel's FX graphs (:func:`table_sizes`): the
#: element-wise integer and float ops and casts ``_Evaluate`` models.
_FX_CHAIN_OPS = frozenset(
    {
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Scalar,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.bitwise_left_shift.Tensor,
        torch.ops.aten.__lshift__.Scalar,
        torch.ops.aten.__lshift__.Tensor,
        torch.ops.aten.__rshift__.Scalar,
        torch.ops.aten.__rshift__.Tensor,
        torch.ops.aten.bitwise_right_shift.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.view.dtype,
        torch.ops.aten.alias.default,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
    }
)


#: The chain ops that fix a value's type: a table's entries need one
#: (``_entries`` keeps an integer of promoted type computed).
_FX_TYPE_FIXING_OPS = frozenset(
    {
        torch.ops.aten.view.dtype,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
    }
)


def table_sizes(nodes: Iterable[torch.fx.Node]) -> set[int]:
    """The table sizes chains among ``nodes`` would take: for each node that
    masks a value to a low-bit field of at most :data:`MAX_TABLE` values,
    the field's size when a chain of element-wise ops computes from it and
    constants alone up to a cast, which fixes the entries' type.  An FX-level estimate of what
    :func:`tabulate` finds in a fragment element's statements, which counts
    statements and models fewer ops: a size it offers may build no table,
    which costs the search a duplicate config."""
    sizes: set[int] = set()
    for node in nodes:
        if node.target not in (
            torch.ops.aten.bitwise_and.Scalar,
            torch.ops.aten.bitwise_and.Tensor,
        ):
            continue
        mask = node.args[1]
        if not _field_mask(mask, MAX_TABLE):
            continue
        cone = {node}
        todo = list(node.users)
        while todo:
            user = todo.pop()
            if user in cone or user.target not in _FX_CHAIN_OPS:
                continue
            if all(a in cone for a in user.all_input_nodes):
                cone.add(user)
                todo += user.users
        # Only a chain that reaches a cast has a table: an integer of
        # promoted type has none (``_entries``).
        if any(n.target in _FX_TYPE_FIXING_OPS for n in cone):
            assert isinstance(mask, int)
            sizes.add(mask + 1)
    return sizes
