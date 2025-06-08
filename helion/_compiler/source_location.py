from __future__ import annotations

import re
import sys
import threading
import traceback
import typing
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

from torch.fx.traceback import get_current_meta
from torch.fx.traceback import has_preserved_node_meta

from .traceback_compat import format_frame_summary

if TYPE_CHECKING:
    import ast
    from typing_extensions import Self

    from .ast_extension import ExtendedAST

    class _TLS(Protocol):
        locations: list[SourceLocation]

    _T = TypeVar("_T", ast.AST, ExtendedAST)


# pyre-ignore-all-errors[16, 28]: lineno/colno/etc are not defined
tls: _TLS = typing.cast("_TLS", threading.local())


class SourceLocation(traceback.FrameSummary):
    """Represents a location in the source code a node came from."""

    def __init__(
        self,
        lineno: int,
        colno: int,
        end_lineno: int,
        end_colno: int,
        name: str,
        filename: str,
    ) -> None:
        if sys.version_info >= (3, 11):
            super().__init__(
                lookup_line=False,
                lineno=lineno,
                end_lineno=end_lineno,
                colno=colno,
                end_colno=end_colno,
                name=name,
                filename=filename,
            )
        else:
            super().__init__(
                lookup_line=False,
                lineno=lineno,
                name=name,
                filename=filename,
            )
            self.end_lineno = end_lineno
            self.colno = colno
            self.end_colno = end_colno

    @staticmethod
    def from_ast(node: ast.AST) -> SourceLocation:
        from .host_function import HostFunction

        host_function = HostFunction.current()
        code = host_function.fn.__code__
        offset = code.co_firstlineno - 1
        return SourceLocation(
            node.lineno + offset,
            node.col_offset + host_function.column_offset,
            node.end_lineno + offset,
            node.end_col_offset + host_function.column_offset,
            filename=code.co_filename,
            name=code.co_name,
        )

    def to_ast(self, node: _T) -> _T:
        if "lineno" in node._attributes:
            node.lineno = self.lineno
            node.col_offset = self.colno
            node.end_lineno = self.end_lineno
            node.end_col_offset = self.end_colno
        return node

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}"

    def __repr__(self) -> str:
        return f"<SourceLocation {re.sub(r'^.*/', '', self.filename)}:{self.lineno}>"

    def format(self) -> str:
        return format_frame_summary(self)

    def _key(self) -> tuple[str, int | None, int, int, int]:
        return (self.filename, self.lineno, self.colno, self.end_lineno, self.end_colno)

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SourceLocation):
            return False
        return self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Self) -> bool:
        return self._key() < other._key()

    def __le__(self, other: Self) -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: Self) -> bool:
        return self._key() > other._key()

    def __ge__(self, other: Self) -> bool:
        return self._key() >= other._key()

    def __bool__(self) -> bool:
        return True

    def __enter__(self) -> None:
        try:
            tls.locations.append(self)
        except AttributeError:
            tls.locations = [self]
        self.set_fx_location()

    def __exit__(self, *args: object) -> None:
        locations = tls.locations
        locations.pop()
        if locations:
            locations[-1].set_fx_location()
        else:
            UnknownLocation().set_fx_location()

    def set_fx_location(self) -> None:
        if has_preserved_node_meta():
            meta = get_current_meta()
            meta["location"] = self
            meta["stack_trace"] = re.sub(
                r"^((?:.*\n){2})(.|\n)*$", r"\1", self.format()
            )


class UnknownLocation(SourceLocation):
    def __init__(self) -> None:
        super().__init__(0, 0, 0, 0, "<unknown>", "<unknown>")

    def __str__(self) -> str:
        return "unknown location"

    def __repr__(self) -> str:
        return "<unknown location>"

    def format(self) -> str:
        return "unknown location\n"

    def __bool__(self) -> bool:
        return False

    def set_fx_location(self) -> None:
        if has_preserved_node_meta():
            meta = get_current_meta()
            meta.pop("location", None)
            meta.pop("stack_trace", None)


def current_location() -> SourceLocation:
    try:
        return tls.locations[-1]
    except (AttributeError, IndexError):
        return UnknownLocation()
