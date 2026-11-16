"""Parse the MSL expressions Inductor's MetalOverrides emits into Python AST.

Helion's device code is Python AST: it is printed as Python source and
``metal_jit`` reads it back and prints MSL (``msl_ast_walker``).  Inductor's
``MetalOverrides`` writes each op as MSL text -- ``a && b``, ``c ? a : b``,
``::metal::precise::atan2(y, x)``, ``static_cast<float>(x)`` -- so this parses
that text by C++'s expression grammar and builds the AST forms the walker
prints back, each of which survives a round trip through Python source:

========================  ==================================================
MSL                       Python AST (as Python source)
========================  ==================================================
``a && b``, ``a || b``    ``a and b``, ``a or b``
``!a``                    ``not a``
``c ? a : b``             ``a if c else b``
``ns::f``                 ``ns.f`` (the walker prints ``::`` for a chain
                          rooted at a C++ namespace)
``::ns::f``               ``_msl_global.ns.f`` (see ``GLOBAL_SCOPE``)
``static_cast<T>(x)``     ``static_cast(T, x)``: the template argument
                          first; also ``as_type`` and the other C++ casts
                          (``TEMPLATE_CALLS``)
``decltype(e)``           ``decltype(e)``
``true``, ``false``       ``True``, ``False``
========================  ==================================================

A template argument is parsed only after a name in ``TEMPLATE_CALLS``:
``a < b`` elsewhere is a comparison, as C++ resolves it for non-templates.  A
cast is a plain call (a subscript ``static_cast[T]`` would read as indexing a
tensor), which the backend declares pure (``PURE_CALLS``) so lane-loop
placement still moves it like the arithmetic around it.  Text outside this
grammar raises ``BackendUnsupported``.
"""

from __future__ import annotations

import ast
import keyword
import re
from typing import Callable

from ... import exc

#: Root of an attribute chain that MSL spells with a leading ``::``.
GLOBAL_SCOPE = "_msl_global"

#: The C++ casts and MSL template functions whose ``<...>`` is a type, not a
#: comparison: ``static_cast<float>(x)``, ``as_type<uint>(x)``.
TEMPLATE_CALLS = frozenset(
    {"static_cast", "reinterpret_cast", "const_cast", "bit_cast", "as_type"}
)

#: The calls ``parse_msl_expression`` builds that compute a register from
#: registers: the casts and ``decltype``.  (``MetalBackend.pure_device_calls``)
PURE_CALLS = TEMPLATE_CALLS | {"decltype"}

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>::|&&|\|\||\+\+|--|->|==|!=|<=|>=|<<|>>|[-+*/%&|^!~<>?:(),.\[\]])
    """,
    re.VERBOSE,
)

#: Binary operators by precedence, loosest first (C++'s order).
_BINARY_LEVELS: tuple[dict[str, ast.operator | ast.cmpop], ...] = (
    {"|": ast.BitOr()},
    {"^": ast.BitXor()},
    {"&": ast.BitAnd()},
    {"==": ast.Eq(), "!=": ast.NotEq()},
    {"<": ast.Lt(), "<=": ast.LtE(), ">": ast.Gt(), ">=": ast.GtE()},
    {"<<": ast.LShift(), ">>": ast.RShift()},
    {"+": ast.Add(), "-": ast.Sub()},
    {"*": ast.Mult(), "/": ast.Div(), "%": ast.Mod()},
)

_UNARY: dict[str, Callable[[], ast.unaryop]] = {
    "!": ast.Not,
    "~": ast.Invert,
    "-": ast.USub,
    "+": ast.UAdd,
}

_CONSTANT_NAMES = {"true": True, "false": False}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise exc.BackendUnsupported(
                "metal", f"MSL expression {text!r}: unexpected {text[pos:]!r}"
            )
        pos = match.end()
        kind = match.lastgroup
        assert kind is not None
        if kind != "space":
            tokens.append((kind, match.group()))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    def fail(self, what: str) -> exc.BackendUnsupported:
        return exc.BackendUnsupported("metal", f"MSL expression {self.text!r}: {what}")

    def peek(self) -> str | None:
        return self.tokens[self.pos][1] if self.pos < len(self.tokens) else None

    def take(self, expected: str | None = None) -> tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise self.fail(f"expected {expected!r}, got the end")
        token = self.tokens[self.pos]
        if expected is not None and token[1] != expected:
            raise self.fail(f"expected {expected!r}, got {token[1]!r}")
        self.pos += 1
        return token

    def name(self) -> str:
        """An identifier, which must also be one in the device code's Python."""
        kind, text = self.take()
        if kind != "name" or keyword.iskeyword(text):
            raise self.fail(f"expected a name, got {text!r}")
        return text

    def parse(self) -> ast.expr:
        node = self.conditional()
        if self.peek() is not None:
            raise self.fail(f"unexpected {self.peek()!r}")
        return node

    def conditional(self) -> ast.expr:
        test = self.logical_or()
        if self.peek() != "?":
            return test
        self.take("?")
        body = self.conditional()
        self.take(":")
        orelse = self.conditional()
        return ast.IfExp(test=test, body=body, orelse=orelse)

    def logical_or(self) -> ast.expr:
        return self.boolean("||", ast.Or, self.logical_and)

    def logical_and(self) -> ast.expr:
        return self.boolean("&&", ast.And, lambda: self.binary(0))

    def boolean(
        self, token: str, op: type[ast.boolop], operand: Callable[[], ast.expr]
    ) -> ast.expr:
        values = [operand()]
        while self.peek() == token:
            self.take(token)
            values.append(operand())
        return values[0] if len(values) == 1 else ast.BoolOp(op=op(), values=values)

    def binary(self, level: int) -> ast.expr:
        if level == len(_BINARY_LEVELS):
            return self.unary()
        ops = _BINARY_LEVELS[level]
        left = self.binary(level + 1)
        while (token := self.peek()) in ops:
            self.take()
            right = self.binary(level + 1)
            op = ops[token]
            # C++ comparisons are left-associative binary operators, not
            # Python's chains: ``a < b < c`` is ``(a < b) < c``.
            left = (
                ast.Compare(left=left, ops=[type(op)()], comparators=[right])
                if isinstance(op, ast.cmpop)
                else ast.BinOp(left=left, op=type(op)(), right=right)
            )
        return left

    def unary(self) -> ast.expr:
        token = self.peek()
        if token in _UNARY:
            self.take()
            return ast.UnaryOp(op=_UNARY[token](), operand=self.unary())
        return self.postfix(self.primary())

    def postfix(self, node: ast.expr) -> ast.expr:
        while True:
            token = self.peek()
            if token == "(":
                node = ast.Call(func=node, args=self.arguments(), keywords=[])
            elif token == "[":
                self.take("[")
                index = self.conditional()
                self.take("]")
                node = ast.Subscript(value=node, slice=index, ctx=ast.Load())
            elif token == ".":
                self.take(".")
                node = ast.Attribute(value=node, attr=self.name(), ctx=ast.Load())
            else:
                return node

    def arguments(self) -> list[ast.expr]:
        self.take("(")
        args: list[ast.expr] = []
        if self.peek() != ")":
            args.append(self.conditional())
            while self.peek() == ",":
                self.take(",")
                args.append(self.conditional())
        self.take(")")
        return args

    def primary(self) -> ast.expr:
        if self.peek() == "(":
            self.take("(")
            node = self.conditional()
            self.take(")")
            return node
        kind, text = self.take()
        if kind == "number":
            value = float(text) if any(c in text for c in ".eE") else int(text)
            return ast.Constant(value=value)
        if kind == "string":
            return ast.Constant(value=ast.literal_eval(text))
        if kind == "name" and text in _CONSTANT_NAMES:
            return ast.Constant(value=_CONSTANT_NAMES[text])
        if kind == "name" or text == "::":
            name = self.qualified(text)
            if self.peek() == "<" and _base_name(name) in TEMPLATE_CALLS:
                self.take("<")
                template_arg = self.type_()
                self.take(">")
                if self.peek() != "(":
                    raise self.fail("a template function must be called")
                return ast.Call(
                    func=name, args=[template_arg, *self.arguments()], keywords=[]
                )
            return name
        raise self.fail(f"unexpected {text!r}")

    def qualified(self, first: str) -> ast.expr:
        """``a::b::c`` (or ``::a::b``) from its first token."""
        if first == "::":
            node: ast.expr = ast.Name(id=GLOBAL_SCOPE, ctx=ast.Load())
            node = ast.Attribute(value=node, attr=self.name(), ctx=ast.Load())
        else:
            if keyword.iskeyword(first):
                raise self.fail(f"expected a name, got {first!r}")
            node = ast.Name(id=first, ctx=ast.Load())
        while self.peek() == "::":
            self.take("::")
            node = ast.Attribute(value=node, attr=self.name(), ctx=ast.Load())
        return node

    def type_(self) -> ast.expr:
        """A template argument: a (qualified) type name or ``decltype(e)``."""
        kind, text = self.take()
        if text == "decltype":
            return ast.Call(
                func=ast.Name(id="decltype", ctx=ast.Load()),
                args=self.arguments(),
                keywords=[],
            )
        if kind != "name" and text != "::":
            raise self.fail(f"expected a type, got {text!r}")
        return self.qualified(text)


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def cast_source(metal_type: str, expr: str) -> str:
    """Device-code source of the MSL ``static_cast<metal_type>(expr)``, in the
    form ``parse_msl_expression`` builds: ``static_cast(metal_type, expr)``."""
    return f"static_cast({metal_type}, {expr})"


def parse_msl_expression(text: str) -> ast.expr:
    """The Python AST of the MSL expression *text* (see the module docstring)."""
    return ast.fix_missing_locations(ast.Expression(_Parser(text).parse())).body
