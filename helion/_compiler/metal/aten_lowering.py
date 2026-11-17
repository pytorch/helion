"""Metal-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Importing this module runs the ``@<op>_lowering.register_codegen("metal")``
registrations; ``_codegen_modules`` imports it.
"""

from __future__ import annotations

import ast
from itertools import starmap
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node
from torch.fx.node import map_arg

from ... import exc
from ..ast_extension import expr_from_string
from ..aten_lowering import _env_arg
from ..aten_lowering import expand_lowering
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


def _only_unit_dims_change(node: Node) -> bool:
    """Whether a view only inserts or drops dims of size 1, keeping the
    other dims, in order, at their sizes."""
    source = node.args[0]
    assert isinstance(source, Node)
    env = CompileEnvironment.current()

    def kept(shape: torch.Size) -> list[int | torch.SymInt]:
        return [s for s in shape if not env.known_equal(s, 1)]

    before = kept(source.meta["val"].shape)
    after = kept(node.meta["val"].shape)
    return len(before) == len(after) and all(
        starmap(env.known_equal, zip(before, after, strict=True))
    )


@expand_lowering.register_codegen("metal")
def codegen_expand_metal(ctx: LoweringContext, node: Node) -> object:
    """Broadcast dims of size 1, or new leading dims, onto tile blocks.

    A value is held by the block ids of its dims and a dim of size 1 owns
    none, so the thread at any position of the new block already holds the
    value for it, as in any broadcasting op.  A dim expanded to a size that
    is not a tile block (a literal) has no block id to hold it by, and
    raises.
    """
    env = CompileEnvironment.current()
    source = node.args[0]
    assert isinstance(source, Node)
    before = [*source.meta["val"].shape]
    after = [*node.meta["val"].shape]
    before = [1] * (len(after) - len(before)) + before
    for old, new in zip(before, after, strict=True):
        if not env.known_equal(old, new) and env.get_block_id(new) is None:
            raise exc.BackendUnsupported(
                "metal",
                f"{node.target} to a dim of size {new}: Metal broadcasts only "
                "onto a tile's block",
            )
    return _input_ast(ctx, node)


@view_lowering.register_codegen("metal")
@reshape_lowering.register_codegen("metal")
def codegen_view_metal(ctx: LoweringContext, node: Node) -> object:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    if isinstance(tensor, StackedValue):
        return merged_stack(node, tensor)
    if _only_unit_dims_change(node):
        # Like squeeze: each thread keeps its element, held by the same
        # block ids, since the dims that carry them are unchanged.
        assert isinstance(tensor, ast.AST)
        return tensor
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
