"""A structural view of printed TritonGPU IR.

Passes installed through ``triton.knobs.runtime.add_stages_inspection_hook``
receive the module in Triton's own MLIR build, whose C++ API is not available
to Helion and whose Python bindings cannot move, clone or erase operations.
This module parses the module's printed text into operations, regions, blocks
and SSA values, lets a pass edit that tree, and prints it back.  Triton's
parser then reads and verifies the result.

Only the structure every op shares is read: result and operand SSA names,
region braces, block labels.  Each op keeps its exact text, so types,
attributes and custom syntax stay opaque and an unedited module prints back
byte for byte.  The printed form it relies on:

* an op starts a line; it continues onto further lines only through regions
  (a line whose last character outside strings is ``{`` opens one, a line
  starting with ``}`` closes it and may open the next, as in ``} else {``) and
  through lines that start with a bare word, such as ``default {`` or
  ``partition0(...) num_warps(1) {`` of ``ttg.warp_specialize``;
* region arguments are printed in the op's text before the region, as
  ``%x: type`` (``tt.func``, partitions) or ``%x = %init`` (``scf.for``,
  ``scf.while``), or in a block label ``^bb0(%x: type):``; op results are
  ``%a, %b = ...`` or ``%a:2 = ...`` and their uses ``%a#1``;
* names are scoped by region: a use resolves to the definition in the
  innermost enclosing region that has one, so sibling regions may reuse names.

:func:`cross_check` matches the tree against Triton's own parse of the same
text (``ir.module.walk``): every op, its results, its block's arguments and
every use.  A pass reads operand order and attributes through that
:class:`Alignment`, never from the printed syntax, and leaves a module whose
tree disagrees with Triton's unchanged.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import os
import re
import tempfile
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# An SSA id outside strings: ``%name``, ``%name#k`` (a use of result k) or
# ``%name:N`` (N results, only in an op's result list).
_SSA = re.compile(r"%([0-9]+|[A-Za-z_$.\-][\w$.\-]*)(?:#([0-9]+))?")
_BARE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\b")
_OP_NAME = re.compile(r'"([^"]+)"|([A-Za-z_][\w$]*(?:\.[\w$]+)+|module)\b')


class ParseError(ValueError):
    """The text does not have the structure this module reads."""


@dataclasses.dataclass(frozen=True)
class SsaToken:
    """One SSA id in an op's text (outside strings)."""

    name: str
    index: int | None
    start: int
    end: int
    # ``%x:`` (a typed definition or a result group) or ``%x =``.
    colon: bool
    equals: bool


_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')
_EQUALS = re.compile(r"\s*=(?!=)")


@functools.lru_cache(maxsize=1 << 16)
def _strings_and_ssa(line: str) -> tuple[tuple[SsaToken, ...], str]:
    """SSA tokens of ``line`` outside string literals, and ``line`` with every
    string literal's contents blanked (same length)."""
    code = _STRING.sub(lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', line)
    if code.count('"') % 2:
        raise ParseError(f"unterminated string: {line.strip()[:80]!r}")
    tokens = tuple(
        SsaToken(
            name=m.group(1),
            index=int(m.group(2)) if m.group(2) else None,
            start=m.start(),
            end=m.end(),
            colon=code.startswith(":", m.end()),
            equals=_EQUALS.match(code, m.end()) is not None,
        )
        for m in _SSA.finditer(code)
    )
    return tokens, code


def _opens_region(line: str) -> bool:
    _, code = _strings_and_ssa(line)
    return code.rstrip().endswith("{")


def _closes_region(line: str) -> bool:
    return line.lstrip().startswith("}")


def _is_continuation(line: str) -> bool:
    s = line.lstrip()
    m = _BARE_WORD.match(s)
    return m is not None and m.group(0) != "module" and not s[m.end() :].startswith(".")


def _is_trivia(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("//")


@dataclasses.dataclass(eq=False, repr=False)
class Value:
    """An SSA value: an op result or a block argument (a region argument
    printed in the op's text is one of the region's entry block)."""

    name: str
    # Results of one op printed as ``%a:N`` share a name; ``count`` is N.
    count: int = 1
    # The op it is a result of, or the block it is an argument of.
    owner_op: Op | None = None
    owner_block: Block | None = None
    users: list[Op] = dataclasses.field(default_factory=list)

    @property
    def is_block_argument(self) -> bool:
        return self.owner_block is not None

    def __repr__(self) -> str:
        return f"Value(%{self.name}, count={self.count})"


@dataclasses.dataclass(eq=False, repr=False)
class Use:
    value: Value
    index: int | None

    def __repr__(self) -> str:
        suffix = "" if self.index is None else f"#{self.index}"
        return f"Use(%{self.value.name}{suffix})"


@dataclasses.dataclass(eq=False, repr=False)
class Text:
    """Lines of an op outside its regions (with their line endings)."""

    lines: list[str]

    def __repr__(self) -> str:
        return f"Text({len(self.lines)} lines)"


@dataclasses.dataclass(eq=False, repr=False)
class Block:
    label: str | None = None
    items: list[Op | str] = dataclasses.field(default_factory=list)
    parent: Region | None = None
    arguments: list[Value] = dataclasses.field(default_factory=list)

    @property
    def ops(self) -> list[Op]:
        return [x for x in self.items if isinstance(x, Op)]

    def __repr__(self) -> str:
        return f"Block({len(self.ops)} ops)"

    def text(self) -> str:
        return (self.label or "") + "".join(
            x if isinstance(x, str) else x.text() for x in self.items
        )


@dataclasses.dataclass(eq=False, repr=False)
class Region:
    blocks: list[Block] = dataclasses.field(default_factory=list)
    parent: Op | None = None
    # Arguments printed in the op's text before the region.
    header_arguments: list[Value] = dataclasses.field(default_factory=list)
    defs: dict[str, Value] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Region({len(self.blocks)} blocks)"

    def text(self) -> str:
        return "".join(b.text() for b in self.blocks)


@dataclasses.dataclass(eq=False, repr=False)
class Op:
    name: str
    parts: list[Text | Region]
    parent: Block | None = None
    results: list[Value] = dataclasses.field(default_factory=list)
    operands: list[Use] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        return f"Op({self.name}: {self.first_line.strip()[:60]!r})"

    @property
    def regions(self) -> list[Region]:
        return [p for p in self.parts if isinstance(p, Region)]

    @property
    def parent_op(self) -> Op | None:
        r = self.parent.parent if self.parent is not None else None
        return r.parent if r is not None else None

    @property
    def first_line(self) -> str:
        first = self.parts[0]
        assert isinstance(first, Text)
        return first.lines[0]

    def text(self) -> str:
        return "".join(
            "".join(p.lines) if isinstance(p, Text) else p.text() for p in self.parts
        )

    def walk(self) -> Iterator[Op]:
        """This op and every op nested in it, in printed order."""
        yield self
        for r in self.regions:
            for b in r.blocks:
                for op in b.ops:
                    yield from op.walk()

    def post_order(self) -> Iterator[Op]:
        """Every op nested in this op, then this op (``ir.module.walk``'s
        order)."""
        for r in self.regions:
            for b in r.blocks:
                for op in b.ops:
                    yield from op.post_order()
        yield self

    def ancestors(self) -> Iterator[Op]:
        op = self.parent_op
        while op is not None:
            yield op
            op = op.parent_op

    # -- edits ---------------------------------------------------------------
    def erase(self) -> None:
        assert self.parent is not None
        self.parent.items.remove(self)
        self.parent = None

    def move_before(self, other: Op, *, reindent: bool = True) -> None:
        """Move this op (with its regions) right before ``other``.  Names are
        kept; the caller keeps them in scope (or calls
        :meth:`Module.analyze` and checks :attr:`Module.unresolved`)."""
        target = other.parent
        assert target is not None and self is not other
        self.erase()
        if reindent:
            self._reindent(_indent_of(other.first_line))
        target.items.insert(target.items.index(other), self)
        self.parent = target

    def _reindent(self, indent: str) -> None:
        _shift_op(self, len(indent) - len(_indent_of(self.first_line)))


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" "))]


def _shift(line: str, delta: int) -> str:
    if delta == 0 or not line.strip():
        return line
    if delta > 0:
        return " " * delta + line
    pad = len(line) - len(line.lstrip(" "))
    return line[min(pad, -delta) :]


def _shift_op(op: Op, delta: int) -> None:
    """Indent every line of ``op`` (its text, block labels, nested ops and
    trivia) by ``delta`` columns."""
    if delta == 0:
        return
    for p in op.parts:
        if isinstance(p, Text):
            p.lines = [_shift(line, delta) for line in p.lines]
            continue
        for b in p.blocks:
            if b.label is not None:
                b.label = _shift(b.label, delta)
            for k, x in enumerate(b.items):
                if isinstance(x, str):
                    b.items[k] = _shift(x, delta)
                else:
                    _shift_op(x, delta)


class Module:
    """A printed TTGIR module: text before the top-level op (attribute
    aliases), the ``module`` op, and text after it (location aliases)."""

    def __init__(self, text: str) -> None:
        self.source = text
        lines = text.splitlines(keepends=True)
        i, n = 0, len(lines)
        while i < n and not lines[i].startswith("module"):
            i += 1
        if i == n:
            raise ParseError("no top-level module op")
        self.prologue = "".join(lines[:i])
        self.top, i = _parse_op(lines, i, None)
        self.epilogue = "".join(lines[i:])
        self.unresolved: list[tuple[Op, str]] = []
        self.analyze()

    def text(self) -> str:
        return self.prologue + self.top.text() + self.epilogue

    def walk(self) -> Iterator[Op]:
        return self.top.walk()

    def functions(self) -> list[Op]:
        return [op for op in self.walk() if op.name == "tt.func"]

    def analyze(self) -> None:
        """(Re)compute results, region arguments, operands and users."""
        self.unresolved = []
        for op in self.walk():
            op.results, op.operands = [], []
            for r in op.regions:
                r.defs, r.header_arguments = {}, []
                for b in r.blocks:
                    b.arguments = []
        _define(self.top)
        _resolve(self.top, self.unresolved)


def parse(text: str) -> Module:
    return Module(text)


# -- cross-check with Triton's own parse -------------------------------------
class Mismatch(ParseError):
    """The tree disagrees with Triton's parse of the same text."""


_Key = tuple[Value, "int | None"]


@dataclasses.dataclass(eq=False, repr=False)
class Alignment:
    """The tree's ops and values matched to Triton's (see :func:`cross_check`).
    Edits that move ops keep it valid; it describes no op added later."""

    # Triton's module: its ops and values live only as long as it does.
    module: Any
    _ops: dict[Op, Any]
    _operands: dict[Op, list[Use]]
    _values: dict[_Key, Any]
    _types: dict[_Key, str] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Alignment({len(self._ops)} ops)"

    def triton(self, op: Op) -> Any:  # noqa: ANN401 - an untyped Triton binding
        """Triton's ``ir.operation`` for ``op`` (its attribute getters)."""
        return self._ops[op]

    def operands(self, op: Op) -> list[Use]:
        """``op``'s uses in Triton's operand order (the op definition's, which
        the printed syntax need not follow)."""
        return self._operands[op]

    def value(self, use: Use | Value) -> Any:  # noqa: ANN401
        """Triton's ``ir.value`` for a use's value (or for a single value)."""
        return self._values[_key(use)]

    def type(self, use: Use | Value) -> str:
        """The printed MLIR type of a use's value (or of a single value)."""
        key = _key(use)
        found = self._types.get(key)
        if found is None:
            found = self._types[key] = str(self._values[key].get_type())
        return found


def _key(use: Use | Value) -> _Key:
    if isinstance(use, Value):
        return (use, None)
    return (use.value, use.index if use.value.count > 1 else None)


def _triton_name(op: Op) -> str:
    return "builtin.module" if op.name == "module" else op.name


def cross_check(m: Module, mod: Any) -> Alignment:  # noqa: ANN401
    """Match ``m`` to ``mod``, Triton's ``ir.module`` parsed from the same
    text; raise :class:`Mismatch` where they differ.

    The ops must be the same, in the same order (Triton's walk visits ops in
    post-order), with as many results; each of our blocks must be one of
    Triton's with as many arguments; each op's uses must name the values
    Triton's operands are.  Ops the printer leaves implicit are skipped: an
    ``scf.yield`` without operands, and ``ttg.warp_specialize.partitions``,
    whose operands belong to the ``ttg.warp_specialize`` around it.
    """
    if m.unresolved:
        op, name = m.unresolved[0]
        raise Mismatch(f"{op.name}: %{name} is not defined")
    theirs: list[Any] = []
    mod.walk(theirs.append)
    ours = list(m.top.post_order())
    pairs: list[tuple[Op, Any, list[Any]]] = []
    i, captures = 0, None
    for t in theirs:
        name = t.get_name()
        operands = [t.get_operand(k) for k in range(t.get_num_operands())]
        if name == "ttg.warp_specialize.partitions":
            captures = operands
            continue
        a = ours[i] if i < len(ours) else None
        if (
            name == "scf.yield"
            and not operands
            and (a is None or a.name != name or a.operands)
        ):
            continue
        if captures is not None:
            if name != "ttg.warp_specialize":
                raise Mismatch(f"partitions outside a warp_specialize ({name})")
            operands, captures = operands + captures, None
        if a is None or _triton_name(a) != name:
            got = "nothing" if a is None else a.name
            raise Mismatch(f"op {i}: {got} where Triton has {name}")
        pairs.append((a, t, operands))
        i += 1
    if i != len(ours):
        raise Mismatch(f"{len(ours) - i} ops Triton does not have ({ours[i].name})")
    key_of: dict[int, _Key] = {}
    values: dict[_Key, Any] = {}

    def bind(tv: Any, key: _Key) -> None:  # noqa: ANN401
        if key in values:
            raise Mismatch(f"%{key[0].name} is two of Triton's values")
        key_of[tv.id()] = key
        values[key] = tv

    blocks: dict[Block, Any] = {}
    owner: dict[int, Block] = {}
    for a, t, _ in pairs:
        if sum(v.count for v in a.results) != t.get_num_results():
            raise Mismatch(f"{a.name}: {t.get_num_results()} results in Triton")
        k = 0
        for v in a.results:
            for j in range(v.count):
                bind(t.get_result(k), (v, j if v.count > 1 else None))
                k += 1
        if a.parent is None:
            continue
        tb = t.get_block()
        seen = blocks.get(a.parent)
        if seen is None:
            blocks[a.parent] = seen = tb
        if seen.id() != tb.id() or owner.setdefault(tb.id(), a.parent) is not a.parent:
            raise Mismatch(f"{a.name}: in another block than in Triton")
    for b, tb in blocks.items():
        assert b.parent is not None
        args = (b.parent.header_arguments if b is b.parent.blocks[0] else []) + list(
            b.arguments
        )
        if len(args) != tb.get_num_arguments():
            raise Mismatch(
                f"a block of {b.parent.parent.name if b.parent.parent else '?'}: "
                f"{len(args)} arguments, {tb.get_num_arguments()} in Triton"
            )
        for k, v in enumerate(args):
            bind(tb.arg(k), (v, None))
    in_order: dict[Op, list[Use]] = {}
    for a, _, operands in pairs:
        uses: dict[_Key, list[Use]] = collections.defaultdict(list)
        for u in a.operands:
            uses[_key(u)].append(u)
        ordered = []
        for tv in operands:
            key = key_of.get(tv.id())
            if key is None or not uses[key]:
                raise Mismatch(f"{a.name}: uses differ from Triton's operands")
            ordered.append(uses[key].pop(0))
        if len(ordered) != len(a.operands):
            raise Mismatch(f"{a.name}: uses differ from Triton's operands")
        in_order[a] = ordered
    return Alignment(mod, {a: t for a, t, _ in pairs}, in_order, values)


def triton_parse(text: str) -> Any:  # noqa: ANN401
    """``text`` parsed and verified by Triton in a new context; raises
    :class:`Mismatch` if Triton rejects it."""
    from triton._C.libtriton import ir  # pyrefly: ignore[missing-module-attribute]

    context = ir.context()
    ir.load_dialects(context)
    fd, path = tempfile.mkstemp(suffix=".ttgir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        try:
            mod = ir.parse_mlir_module(path, context)
        except RuntimeError as e:
            raise Mismatch(f"Triton cannot parse the module ({e})") from e
    finally:
        os.unlink(path)
    mod.context = context  # keeps the context alive with the module
    if not mod.verify():
        raise Mismatch("Triton's verifier rejects the module")
    return mod


# -- parsing ------------------------------------------------------------------
def _parse_op(lines: list[str], i: int, parent: Block | None) -> tuple[Op, int]:
    m = re.match(r"\s*(?:%[^=]*?=\s*)?", lines[i])
    name_m = _OP_NAME.match(lines[i], m.end() if m else 0)
    if name_m is None:
        raise ParseError(f"line {i + 1}: not an op: {lines[i].strip()[:80]!r}")
    op = Op(name=name_m.group(1) or name_m.group(2), parts=[], parent=parent)
    text = [lines[i]]
    cur = lines[i]
    i += 1
    n = len(lines)
    while True:
        if _opens_region(cur):
            op.parts.append(Text(text))
            region = Region(parent=op)
            i = _parse_region(lines, i, region)
            if i >= n:
                raise ParseError(f"unclosed region of {op.name}")
            op.parts.append(region)
            text, cur = [lines[i]], lines[i]
            i += 1
            continue
        if i < n and _is_continuation(lines[i]):
            text.append(lines[i])
            cur = lines[i]
            i += 1
            continue
        op.parts.append(Text(text))
        return op, i


def _parse_region(lines: list[str], i: int, region: Region) -> int:
    """Parse blocks up to the line that closes ``region``; return its index."""
    block = Block(parent=region)
    region.blocks.append(block)
    n = len(lines)
    while i < n:
        line = lines[i]
        if _closes_region(line):
            return i
        if _is_trivia(line):
            block.items.append(line)
            i += 1
            continue
        if line.lstrip().startswith("^"):
            if block.items or block.label is not None:
                block = Block(parent=region)
                region.blocks.append(block)
            block.label = line
            i += 1
            continue
        op, i = _parse_op(lines, i, block)
        block.items.append(op)
    return i


# -- SSA analysis -----------------------------------------------------------
def _result_prefix(line: str) -> tuple[list[tuple[str, int]], int]:
    """``(name, count)`` of each result in ``line``'s result list and the
    offset where the op name starts (0 without results)."""
    tokens, code = _strings_and_ssa(line)
    m = re.match(r"\s*((?:%[^=\s,]+\s*,\s*)*%[^=\s,]+)\s*=(?!=)\s*", code)
    if m is None:
        return [], 0
    out = []
    for part in m.group(1).split(","):
        pm = re.fullmatch(r"\s*%([^:\s]+)(?::([0-9]+))?\s*", part)
        if pm is None:
            return [], 0
        out.append((pm.group(1), int(pm.group(2) or 1)))
    return out, m.end()


def _segments(op: Op) -> Iterator[tuple[Text, Region | None]]:
    """Each text part of ``op`` with the region it introduces (if any)."""
    for k, p in enumerate(op.parts):
        if isinstance(p, Text):
            nxt = op.parts[k + 1] if k + 1 < len(op.parts) else None
            yield p, nxt if isinstance(nxt, Region) else None


def _define(op: Op) -> None:
    """Record ``op``'s results in its parent region and every definition
    inside ``op`` (region arguments, nested results)."""
    names, _ = _result_prefix(op.first_line)
    op.results = [Value(name=nm, count=c, owner_op=op) for nm, c in names]
    region = op.parent.parent if op.parent is not None else None
    if region is not None:
        for v in op.results:
            region.defs[v.name] = v
    first = True
    for text, introduces in _segments(op):
        for li, line in enumerate(text.lines):
            tokens, _ = _strings_and_ssa(line)
            skip = 0
            if first and li == 0:
                skip = _result_prefix(line)[1]
            for t in tokens:
                if t.start < skip or introduces is None:
                    continue
                if t.colon or t.equals:
                    v = Value(name=t.name, owner_block=introduces.blocks[0])
                    introduces.header_arguments.append(v)
                    introduces.defs[t.name] = v
        first = False
    for r in op.regions:
        for b in r.blocks:
            if b.label is not None:
                for t in _strings_and_ssa(b.label)[0]:
                    if t.colon:
                        v = Value(name=t.name, owner_block=b)
                        b.arguments.append(v)
                        r.defs[t.name] = v
            for x in b.ops:
                _define(x)


def _lookup(region: Region | None, name: str) -> Value | None:
    while region is not None:
        v = region.defs.get(name)
        if v is not None:
            return v
        owner = region.parent
        block = owner.parent if owner is not None else None
        region = block.parent if block is not None else None
    return None


def _resolve(op: Op, unresolved: list[tuple[Op, str]]) -> None:
    outer = op.parent.parent if op.parent is not None else None
    first = True
    for text, introduces in _segments(op):
        for li, line in enumerate(text.lines):
            tokens, _ = _strings_and_ssa(line)
            skip = _result_prefix(line)[1] if first and li == 0 else 0
            for t in tokens:
                if t.start < skip:
                    continue
                if introduces is not None and (t.colon or t.equals):
                    continue
                v = _lookup(outer, t.name)
                if v is None:
                    unresolved.append((op, t.name))
                    continue
                op.operands.append(Use(v, t.index))
                v.users.append(op)
        first = False
    for r in op.regions:
        for b in r.blocks:
            for x in b.ops:
                _resolve(x, unresolved)
