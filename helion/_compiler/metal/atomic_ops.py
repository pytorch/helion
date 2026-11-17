"""Metal-backend codegen for the atomic ops of ``helion.language.atomic_ops``.

An atomic lowers like a store, through the pointer indexing strategy, to
``tl.atomic_<op>(ptr + offset, value, mask=..., sem=...)``, which the MSL
walker emits as ``atomic_fetch_<op>_explicit`` on the element.  Metal's
atomics are relaxed only, on 32-bit ints and (``add``, ``xchg``) floats.

Unlike a store, an atomic isn't idempotent: every thread and lane holding
the same element would apply it.  A thread holds one index per block, and
the threads of a block the atomic's index doesn't vary over (or the lanes
of a lane loop over one) hold the same element, so only the first of them
applies it (:func:`_one_writer_guard`).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.atomic_ops import _to_ast_values
from ...language.atomic_ops import atomic_add
from ...language.atomic_ops import atomic_and
from ...language.atomic_ops import atomic_max
from ...language.atomic_ops import atomic_min
from ...language.atomic_ops import atomic_or
from ...language.atomic_ops import atomic_xchg
from ...language.atomic_ops import atomic_xor
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..tile_strategy import DeviceGridState
from ..tile_strategy import DeviceLoopState

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..inductor_lowering import CodegenState

#: The atomic ops by the element dtypes Metal runs them on.
_DTYPES = {
    "atomic_add": (torch.float32, torch.int32),
    "atomic_xchg": (torch.float32, torch.int32),
    "atomic_and": (torch.int32,),
    "atomic_or": (torch.int32,),
    "atomic_xor": (torch.int32,),
    "atomic_max": (torch.int32,),
    "atomic_min": (torch.int32,),
}


def _unsupported(message: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("metal", message)


def _one_writer_guard(
    state: CodegenState, target: torch.Tensor, index: list[object]
) -> str | None:
    """The condition under which a thread applies the atomic, so that one
    thread and lane applies it per element: for each block spread over
    several threads, or walked by a lane loop, around the atomic that isn't
    a dim of its index, only the thread and lane at the tile's first
    position.  Every thread and lane along such a block holds the same
    value (it has no dim of the block), and the first position is never
    masked off.  None when every such block is indexed."""
    env = CompileEnvironment.current()
    shape = SubscriptIndexing.compute_shape(target, index, state)
    indexed = {env.get_block_id(size) for size in shape} - {None}
    strategy = state.device_function.tile_strategy
    parallel: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, (DeviceLoopState, DeviceGridState)):
                parallel |= loop.lane_loop_blocks
            parallel |= {
                block
                for block in loop.block_ids
                if (strategy.thread_extent_for_block_id(block) or 1) > 1
            }
    cg = state.codegen
    terms = [
        f"({cg.index_var(block)} == {cg.tile_begin_var(block)})"
        for block in sorted(parallel - indexed)
    ]
    return " & ".join(terms) or None


def _codegen_atomic(op: str, state: CodegenState) -> ast.AST:
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = state.proxy_arg(len(state.ast_args) - 1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)
    if target not in HostFunction.current().tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(op)
    if target.dtype not in _DTYPES[op]:
        raise _unsupported(f"hl.{op} on {target.dtype}")
    if sem != "relaxed":
        raise _unsupported(f"hl.{op} with sem={sem!r}: Metal's atomics are relaxed")
    if len(HostFunction.current().device_ir.phases) > 1:
        # The phases' coherent buffers (msl_phases) are picked from the
        # stores and loads each phase makes, which don't count atomics.
        raise _unsupported(f"hl.{op} in a kernel with hl.barrier() phases")
    guard = _one_writer_guard(state, target, index)
    (value,) = _to_ast_values([state.ast_args[2]])
    device_fn = state.device_function
    indexing_idx = device_fn.atomic_op_index
    device_fn.atomic_op_index += 1
    strategy = device_fn.get_atomic_indexing_strategy(indexing_idx)
    atomic = strategy.codegen_atomic(
        op, state, target, index, value, expr_from_string(repr(sem))
    )
    if guard is not None:
        assert isinstance(atomic, ast.Call)
        (mask,) = [k for k in atomic.keywords if k.arg == "mask"]
        guarded = (
            expr_from_string(guard)
            if isinstance(mask.value, ast.Constant) and mask.value.value is None
            else expr_from_string(f"({{mask}}) & {guard}", mask=mask.value)
        )
        assert isinstance(guarded, ast.expr)
        mask.value = guarded
    return atomic


def _register(op_fn: Callable[..., object], op: str) -> None:
    @_decorators.codegen(op_fn, "metal")
    def _(state: CodegenState) -> ast.AST:
        return _codegen_atomic(op, state)


for _fn, _op in (
    (atomic_add, "atomic_add"),
    (atomic_xchg, "atomic_xchg"),
    (atomic_and, "atomic_and"),
    (atomic_or, "atomic_or"),
    (atomic_xor, "atomic_xor"),
    (atomic_max, "atomic_max"),
    (atomic_min, "atomic_min"),
):
    _register(_fn, _op)
