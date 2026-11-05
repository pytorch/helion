"""Metal-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Importing this module runs the ``@<op>_lowering.register_codegen("metal")``
registrations; ``_codegen_modules`` imports it.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch.fx.node import map_arg

from ... import exc
from ..aten_lowering import _env_arg
from ..aten_lowering import permute_lowering
from ..aten_lowering import reshape_lowering
from ..aten_lowering import squeeze_lowering
from ..aten_lowering import stack_lowering
from ..aten_lowering import view_lowering
from .factored_dims import StackedValue
from .factored_dims import merged_stack
from .factored_dims import stacked

if TYPE_CHECKING:
    from torch.fx.node import Node

    from ..aten_lowering import LoweringContext


def _input_ast(ctx: LoweringContext, node: Node) -> ast.AST:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    return tensor


@squeeze_lowering.register_codegen("metal")
def codegen_squeeze_metal(ctx: LoweringContext, node: Node) -> object:
    # A squeezed dim has a literal size of 1, which never owns a block id or
    # a thread axis, so each thread keeps its element.
    return _input_ast(ctx, node)


@permute_lowering.register_codegen("metal")
def codegen_permute_metal(ctx: LoweringContext, node: Node) -> object:
    # Values are held by block id, so permuting relabels dims without moving
    # an element between threads.  ``positional_rebind`` rejects the
    # consumers that would read a dim by position.
    return _input_ast(ctx, node)


@stack_lowering.register_codegen("metal")
def codegen_stack_metal(ctx: LoweringContext, node: Node) -> object:
    return stacked(ctx, node)


@view_lowering.register_codegen("metal")
@reshape_lowering.register_codegen("metal")
def codegen_view_metal(ctx: LoweringContext, node: Node) -> object:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    if isinstance(tensor, StackedValue):
        return merged_stack(node, tensor)
    raise exc.BackendUnsupported(
        "metal",
        f"{node.target}: Metal lowers a view only where it merges a stacked dim "
        "into the block dim before it",
    )
