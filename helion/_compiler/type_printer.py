from __future__ import annotations

from typing import TYPE_CHECKING

from .ast_extension import ExtendedAST
from .ast_extension import _TupleParensRemovedUnparser

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterator


class OutputLines:
    def __init__(self, parent: ASTPrinter) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.last_newline = 0
        self.parent = parent

    def extend(self, lines: list[str]) -> None:
        """Keep track of the index right after the last newline so insert_annotation can insert in the correct spot."""
        lines = "".join(lines).splitlines(keepends=True)
        if not lines:
            return
        self.lines.extend(lines)
        if lines[-1].endswith("\n"):
            self.last_newline = len(self.lines)
        elif len(lines) > 1:
            assert lines[-2].endswith("\n")
            self.last_newline = len(self.lines) - 1

    def __bool__(self) -> bool:
        return bool(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[str]:
        return iter(self.lines)

    def insert_annotation(self, annotation: str) -> None:
        assert "\n" not in annotation
        indent = "    " * self.parent._indent
        self.lines.insert(self.last_newline, f"{indent}# {annotation}\n")
        self.last_newline += 1

    def append(self, text: str) -> None:
        self.extend([text])


class ASTPrinter(_TupleParensRemovedUnparser):
    # pyre-ignore[13]
    _indent: int

    def __init__(self, *args, **kwargs) -> None:  # pyre-ignore[2]
        super().__init__(*args, **kwargs)
        assert self._source == []
        self._source = self.output = OutputLines(self)

    def traverse(self, node: ast.AST | list[ast.AST]) -> None:
        if isinstance(node, ExtendedAST):
            for annotation in node.debug_annotations():
                if annotation:
                    self.output.insert_annotation(
                        f"{type(node).__name__}: {annotation}"
                    )
        # pyre-ignore[16]
        super().traverse(node)


def print_ast(node: ast.AST) -> str:
    printer = ASTPrinter()
    printer.traverse(node)
    result = "".join(printer.output)
    del printer.output  # break reference cycle
    return result
