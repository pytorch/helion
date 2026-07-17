"""Metal-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "metal")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.memory_ops import load
from ...language.memory_ops import store

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(store, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton.
    # This produces tl.store(ptr + offset, val, mask) in the AST;
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_store_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_store(
            state, tensor, [*subscript], value, extra_mask, None
        )
    raise exc.BackendUnsupported("metal", f"store target type: {type(tensor)}")


@_decorators.codegen(load, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton.
    # This produces tl.load(ptr + offset, mask, other=0) in the AST;
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None
    assert isinstance(eviction_policy, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_load_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, None
        )
    raise exc.BackendUnsupported("metal", f"load tensor type: {type(tensor)}")
