"""Occurrence-independent thread-axis references in emitted CuTe syntax."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re

_THREAD_AXIS = re.compile(r"cute\.arch\.thread_idx\(\)\[(\d+)\]")
_AXIS_PREFIX_SIZE = len("cute.arch.thread_idx()[")


def _thread_axis(node: ast.Subscript) -> int | None:
    index = node.slice
    call = node.value
    if not (
        isinstance(index, ast.Constant)
        and type(index.value) is int
        and index.value >= 0
        and isinstance(call, ast.Call)
        and not call.args
        and not call.keywords
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "thread_idx"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "arch"
    ):
        return None
    namespace = call.func.value.value
    # The previous regex had no identifier boundary: qualified names and names
    # ending in "cute" also matched. Preserve that conservative behavior.
    if (
        isinstance(namespace, ast.Name)
        and namespace.id.endswith("cute")
        or isinstance(namespace, ast.Attribute)
        and namespace.attr.endswith("cute")
    ):
        return index.value
    return None


def _prefix(text: str) -> str:
    end = min(_AXIS_PREFIX_SIZE, len(text))
    while end < len(text) and text[end].isdecimal():
        end += 1
    return text[: end + 1]


def _suffix(text: str) -> str:
    start = len(text)
    while start and text[start - 1].isdecimal():
        start -= 1
    return text[max(0, start - _AXIS_PREFIX_SIZE) :]


@dataclass(frozen=True)
class _LiteralBoundary:
    length: int
    prefix: str
    suffix: str

    @staticmethod
    def from_text(text: str) -> _LiteralBoundary:
        return _LiteralBoundary(len(text), _prefix(text), _suffix(text))

    def then(self, other: _LiteralBoundary, axes: set[int]) -> _LiteralBoundary:
        axes.update(
            int(axis) for axis in _THREAD_AXIS.findall(self.suffix + other.prefix)
        )
        return _LiteralBoundary(
            self.length + other.length,
            _prefix(self.prefix + other.prefix)
            if len(self.prefix) == self.length
            else self.prefix,
            _suffix(self.suffix + other.suffix)
            if len(other.suffix) == other.length
            else other.suffix,
        )


def _joined_literal_axes(
    node: ast.JoinedStr, summaries: dict[int, _LiteralBoundary], axes: set[int]
) -> None:
    # Nested JoinedStr nodes concatenate their literal pieces. Memoize only
    # boundary context, so a shared literal DAG does not expand its full text.
    # A cross-boundary match needs at most the fixed prefix plus its decimal
    # run and closing bracket. Decimal runs retain the regex's unbounded width.
    pending: list[tuple[ast.AST, bool]] = [(node, False)]
    while pending:
        current, ready = pending.pop()
        if id(current) in summaries:
            continue
        if isinstance(current, ast.JoinedStr):
            if not ready:
                pending.append((current, True))
                pending.extend((value, False) for value in reversed(current.values))
                continue
            summary = _LiteralBoundary.from_text("")
            for value in current.values:
                summary = summary.then(summaries[id(value)], axes)
        elif isinstance(current, ast.Constant) and isinstance(current.value, str):
            summary = _LiteralBoundary.from_text(current.value)
        else:
            # An interpolation emits braces and cannot join literal fragments
            # into this regex. Its expression is inspected by the main walk.
            summary = _LiteralBoundary.from_text("\0")
        summaries[id(current)] = summary


def referenced_thread_axes(statement: ast.AST) -> set[int]:
    """Collect the old textual query's axes without expanding shared syntax.

    This is a presence query only. Repeated syntax cannot change its set; this
    traversal must not be used for lexical memory ordering or execution counts.
    Keep the old conservative matches in literal text, using only leaf unparse.
    """
    axes: set[int] = set()
    pending = [statement]
    seen: set[int] = set()
    summaries: dict[int, _LiteralBoundary] = {}
    type_ignores = (
        {ignore.lineno: f"ignore{ignore.tag}" for ignore in statement.type_ignores}
        if isinstance(statement, ast.Module)
        else {}
    )
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ast.Subscript):
            axis = _thread_axis(node)
            if axis is not None:
                axes.add(axis)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            axes.update(int(axis) for axis in _THREAD_AXIS.findall(ast.unparse(node)))
        elif isinstance(node, ast.JoinedStr):
            _joined_literal_axes(node, summaries, axes)
        elif isinstance(
            node,
            (
                ast.Assign,
                ast.For,
                ast.AsyncFor,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.With,
                ast.AsyncWith,
            ),
        ):
            # These are exactly the statement kinds whose type_comment is
            # emitted by ast.unparse; arg.type_comment is not emitted.
            comment = type_ignores.get(node.lineno) or node.type_comment
            if comment is not None:
                axes.update(int(axis) for axis in _THREAD_AXIS.findall(comment))
        pending.extend(ast.iter_child_nodes(node))
    return axes
