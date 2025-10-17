from __future__ import annotations

from typing import TYPE_CHECKING

from .ast_extension import ExtendedAST
from .ast_extension import _TupleParensRemovedUnparser
from .output_lines import OutputLines

if TYPE_CHECKING:
    import ast


class ASTPrinter(_TupleParensRemovedUnparser):
    _indent: int

    def __init__(self, *args: object, **kwargs: object) -> None:
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

        super().traverse(node)


def print_ast(node: ast.AST) -> str:
    printer = ASTPrinter()
    printer.traverse(node)
    result = "".join(printer.output)
    del printer.output  # break reference cycle
    return result
