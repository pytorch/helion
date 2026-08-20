"""FlyDSL-backend codegen for ops defined in ``helion.language._tracing_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "flydsl")``
registrations; ``_codegen_modules`` imports it so registration keeps the same
eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.simd import constant_repr

from ...language import _decorators
from ...language._tracing_ops import _mask_to
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_mask_to, "flydsl")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))

    from .memory_ops import _flydsl_col_tail_pred

    env = CompileEnvironment.current()
    df = state.device_function
    # Bind the masked value to a var so the mask/select can reference it several
    # times. ``expr`` is a per-thread vector (length V).
    e_var = df.new_var("_maskin", dce=True)
    state.add_statement(
        statement_from_string(f"{e_var} = {{expr}}", expr=state.ast_arg(0))
    )

    # Build a per-element bool VECTOR mask matching ``expr``. For the looped
    # reduction column with a tail, use the exact per-element predicate
    # (``roffset + lane*V + iota < N``) -- the strategy's scalar ``mask_var``
    # (``rindex < N``) is only correct at V=1. Other dims (e.g. a bm>1 row mask)
    # contribute a scalar ``mask_var`` broadcast to the vector shape.
    # The vectorized column dim (flydsl tiles V=4 contiguous elems/thread over it)
    # is the last tensor dim with a block id; its mask must be per-element, while
    # row dims broadcast one scalar mask across the thread's vector.
    _col_index = None
    for size in tensor.size():
        _bi = env.resolve_block_id(size)
        if _bi is not None:
            _col_index = _bi

    mask_terms: list[str] = []
    for size in tensor.size():
        index = env.resolve_block_id(size)
        if index is None:
            continue
        strategy = df.tile_strategy.block_id_to_strategy.get((index,))
        _tc = getattr(strategy, "_thread_count", 0) or 0
        _lb = getattr(strategy, "_loop_block_size", 0) or 0
        mask_var = state.codegen.mask_var(index)
        if (
            env.block_sizes[index].reduction
            and strategy is not None
            and _tc > 0
            and _lb >= _tc
            and not env.known_multiple(env.block_sizes[index].numel, _lb)
        ):
            _v = max(1, _lb // _tc)
            _pred = _flydsl_col_tail_pred(
                state,
                strategy.offset_var(index),
                _v,
                state.sympy_expr(env.block_sizes[index].numel),
                lane_mod=_tc,
            )
            mask_terms.append(_pred)
        elif index == _col_index and _tc == 0 and mask_var is not None:
            # Explicit ``hl.tile(n)`` reduction column (user-tiled, so NOT a
            # ``reduction=True`` looped strategy): this dim is vectorized V=4
            # elems/thread, element j at column ``index_var*4 + j``. The scalar
            # ``mask_var`` (``idx < N``) is only correct at V=1, so build the
            # per-element predicate matching the copy's column addressing.
            _ci = state.codegen.index_var(index)
            _v = 4
            _elems = ", ".join(str(j) for j in range(_v))
            _iota = f"fx.Vector.from_elements([{_elems}], fx.Int32)"
            mask_terms.append(
                f"((({_ci}) * {_v}) + {_iota}) "
                f"< ({state.sympy_expr(env.block_sizes[index].numel)})"
            )
        elif mask_var is not None:
            # Broadcast the runtime scalar mask to a bool vector: flydsl's
            # ``filled`` only accepts a compile-time constant fill, so AND the
            # scalar into an all-true bool vector (``&`` promotes scalar->vector).
            mask_terms.append(
                f"(fx.Vector.filled_like({e_var}, True, dtype=fx.Boolean) "
                f"& ({mask_var}))"
            )

    if not mask_terms:
        return expr_from_string(e_var)

    mask_expr = (
        mask_terms[0]
        if len(mask_terms) == 1
        else " & ".join(f"({m})" for m in mask_terms)
    )
    dtype_str = env.backend.dtype_str(tensor.dtype)
    other_typed = f"{dtype_str}({constant_repr(other)})"
    # Broadcast identity to expr's shape via a shape-only zero vector (NaN-safe
    # for +/-inf, unlike ``(e)-(e)``).
    return expr_from_string(
        f"({mask_expr}).select({e_var}, (fx.Vector.filled_like({e_var}, 0) + ({other_typed})))"
    )
