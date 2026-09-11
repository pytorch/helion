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


def _reject_aliased_slice_dims(
    tensor: torch.Tensor, subscript: list[object] | tuple[object, ...]
) -> None:
    """Reject an access whose full slices share one reduction index.

    ``CompileEnvironment.allocate_reduction_dimension`` caches by size, so two
    ``:`` axes of equal length get the same block id and therefore the same
    ``tid`` component.  A tile-level backend keeps them apart by broadcasting;
    Metal, where each value is a scalar per thread, would collapse them and
    walk the diagonal -- ``out[tile, :, :] = a[tile, :, None] * b[None, None, :]``
    over ``[m, n, n]`` writes only ``out[m, i, i]``.

    ``MetalBackend.validate_reduction_input`` catches the same mistake once a
    reduction consumes the axes; this covers accesses that never reach a
    reduction at all.

    Sizes are compared pairwise with ``known_equal`` rather than looked up in a
    dict keyed by the size itself.  Under dynamic shapes a size is a ``SymInt``,
    which is unhashable and whose ``__eq__`` compares symbols; the sizes that
    actually alias are the ones ``allocate_reduction_dimension`` unifies, and
    ``known_equal`` is the predicate it unifies them with.
    """
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    full_slices: list[tuple[int, int | torch.SymInt]] = []
    dim = -1
    for index in subscript:
        if index is None:
            continue  # inserts a new axis; consumes no tensor dimension
        dim += 1
        if not (isinstance(index, slice) and index == slice(None)):
            continue
        size = tensor.size(dim)
        # A length-1 slice never reaches allocate_reduction_dimension (see
        # ``indexing_strategy``, which indexes it with a constant 0), so it
        # claims no thread axis and cannot collapse onto another dimension.
        if isinstance(size, int) and size == 1:
            continue
        full_slices.append((dim, size))

    for position, (first, first_size) in enumerate(full_slices):
        for other, other_size in full_slices[position + 1 :]:
            if env.known_equal(first_size, other_size):
                raise exc.BackendUnsupported(
                    "metal",
                    f"dimensions {first} and {other} are both indexed by a "
                    f"full slice of length {first_size}; equal-length slices "
                    "share one reduction index, and Metal maps one index to "
                    "one thread axis, so the two axes would collapse onto the "
                    "same thread",
                )


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
        _reject_aliased_slice_dims(tensor, subscript)
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
        _reject_aliased_slice_dims(tensor, subscript)
        device_fn = state.device_function
        device_fn.device_load_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, None
        )
    raise exc.BackendUnsupported("metal", f"load tensor type: {type(tensor)}")
