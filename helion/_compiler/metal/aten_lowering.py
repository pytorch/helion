"""Metal-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Importing this module runs the ``@<op>_lowering.register_codegen("metal")``
registrations; ``_codegen_modules`` imports it.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node
from torch.fx.node import map_arg

from ... import exc
from ..ast_extension import expr_from_string
from ..aten_lowering import _env_arg
from ..aten_lowering import permute_lowering
from ..aten_lowering import reshape_lowering
from ..aten_lowering import squeeze_lowering
from ..aten_lowering import stack_lowering
from ..aten_lowering import view_dtype_lowering
from ..aten_lowering import view_lowering
from ..compile_environment import CompileEnvironment
from .factored_dims import StackedValue
from .factored_dims import merged_stack
from .factored_dims import stacked
from .msl_expr import cast_source

if TYPE_CHECKING:
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


@view_dtype_lowering.register_codegen("metal")
def codegen_view_dtype_metal(ctx: LoweringContext, node: Node) -> object:
    """Reinterpret equal-width values within their thread: ``as_type<T>``,
    after rounding to the declared dtype, since a lowering can hold a narrow
    value in a wider register."""
    source_node, target = node.args[:2]
    assert isinstance(source_node, Node) and isinstance(target, torch.dtype)
    source = source_node.meta["val"].dtype
    if source.itemsize != target.itemsize or torch.bool in (source, target):
        raise exc.BackendUnsupported(
            "metal",
            f"view.dtype from {source} to {target}: Metal reinterprets only "
            "between numeric dtypes of one width",
        )
    backend = CompileEnvironment.current().backend
    return expr_from_string(
        f"as_type({backend.dtype_str(target)}, "
        f"{cast_source(backend.dtype_str(source), '{x}')})",
        x=_input_ast(ctx, node),
    )
