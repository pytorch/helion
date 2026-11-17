"""Change only scalar range indexing in the shared attention source."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING

from .cute_paged_attention import paged_online_attention
from .cute_source import transform_kernel

if TYPE_CHECKING:
    import helion


class _BoundsIndex(ast.NodeTransformer):
    def __init__(self, index: str) -> None:
        self.index = index
        self.changed = 0

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "bounds":
            assert isinstance(node.slice, ast.Tuple)
            node.slice.elts[0] = ast.parse(self.index, mode="eval").body
            self.changed += 1
        return node


@functools.cache
def paired_bounds_attention(index: str) -> helion.Kernel:
    assert index in ("request", "item // 2", "item", "head")
    transform = _BoundsIndex(index)
    result = transform_kernel(
        paged_online_attention,
        transform,
        name="paired_bounds_attention",
        filename=f"<helion_testing_paired_bounds_{index}>",
    )
    assert transform.changed == 2
    return result
