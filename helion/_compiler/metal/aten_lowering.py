"""Metal-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Importing this module runs the ``@<op>_lowering.register_codegen("metal")``
registrations; ``_codegen_modules`` imports it.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch.fx.node import map_arg

from ..aten_lowering import _env_arg
from ..aten_lowering import permute_lowering
from ..aten_lowering import squeeze_lowering

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
