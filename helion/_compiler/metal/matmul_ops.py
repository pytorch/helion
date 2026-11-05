"""Metal-backend codegen for ``hl.dot`` outside an MPP region.

``dot(a, b)[m, n] = sum_k f32(a[m, k]) * f32(b[k, n])``.  Each thread forms
the product at its own ``(m, k, n)``, and K's own reduction strategy combines
it, the way the scalar path lowers ``(a[:, :, None] * b[None]).sum(1)``.
Helion masks every ``hl.dot`` operand to 0 past each block's extent
(``_mask_to``), so a K tail or a surplus K thread adds nothing.

A canonical K-loop ``hl.dot`` never gets here (``rewrite_mpp_graphs``
replaces it with an MPP region before codegen), and one the MPP planner
admits (a K held as per-thread parts) is forwarded to
``mpp_matmul.codegen_dot_metal``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.matmul_ops import dot
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from .padded_twin import reduction_dims_padding_to

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _unsupported(message: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("metal", message)


def _dim_block(size: int | torch.SymInt) -> int | None:
    """The block that indexes one dimension of a dot operand, or ``None`` for
    a literal 1.

    A tile or symbolic slice carries its block and a grid index its grid
    block.  A static size resolves to the reduction dimension of that length,
    which must be the only one that could carry it: a constant-shape
    ``hl.zeros``/``hl.full`` pads every dim to a power of two
    (``tensor_utils._pad_dim``), so ``hl.zeros([4, 12])`` shows K as 16 while
    K's reduction dimension is 12 long, and an unrelated 16-long ``:`` slice
    would claim it.  A Metal thread holds one index per block, so the block
    picked decides which elements are summed.
    """
    if isinstance(size, int) and size == 1:
        return None
    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(size)
    if block_id is None:
        raise _unsupported(f"hl.dot dimension {size} is not a tile or a `:` slice")
    if isinstance(size, int):
        candidates = reduction_dims_padding_to(size)
        if candidates != [block_id]:
            raise _unsupported(
                f"hl.dot dimension {size} matches reduction dimensions "
                f"{candidates}: one of length {size} and one that a "
                f"constant-shape hl.zeros/hl.full pads to {size}, so Metal can't "
                "tell which one indexes the operand; give the `:` slices lengths "
                "that stay distinct when padded to a power of two, or tile this "
                "dimension"
            )
    return block_id


@_decorators.codegen(dot, "metal")
def _(state: CodegenState) -> ast.AST:
    # Imported here: reduction_strategy imports device_ir, which needs the
    # helion package that is still initializing when this module registers.
    from ..reduction_strategy import LoopedReductionStrategy
    from .mpp_matmul import codegen_dot_metal
    from .mpp_matmul import planned
    from .reduction_strategy import MetalPersistentReductionStrategy

    if planned(state.fx_node):
        return codegen_dot_metal(state)
    env = CompileEnvironment.current()
    backend = env.backend
    lhs, rhs, acc, out_dtype = state.proxy_args
    assert isinstance(lhs, torch.Tensor)
    assert isinstance(rhs, torch.Tensor)
    assert acc is None or isinstance(acc, torch.Tensor)
    assert out_dtype is None or isinstance(out_dtype, torch.dtype)
    assert state.fx_node is not None
    out = state.fx_node.meta["val"]
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise _unsupported("batched (3-D) hl.dot outside an MPP region")
    if any(t.dtype not in _FLOAT_DTYPES for t in (lhs, rhs, out)):
        raise _unsupported(
            f"hl.dot of {lhs.dtype} x {rhs.dtype} -> {out.dtype} outside an MPP "
            "region; only float16, bfloat16 and float32 are supported"
        )
    if acc is not None and out_dtype is not None and out_dtype != acc.dtype:
        # Triton rounds the product sum to out_dtype before adding acc.
        raise _unsupported(
            f"hl.dot with out_dtype={out_dtype} and a {acc.dtype} acc outside an "
            "MPP region; pass an out_dtype equal to the acc's dtype"
        )

    shapes = f"{list(lhs.shape)} x {list(rhs.shape)}"
    m_block, k_block, rhs_k_block, n_block = map(
        _dim_block, (lhs.size(0), lhs.size(1), rhs.size(0), rhs.size(1))
    )
    # Raw ids, not canonical ones: a tile aliased to another block's size is
    # still its own loop with its own index.
    if rhs_k_block != k_block:
        raise _unsupported(
            f"hl.dot of {shapes} contracts lhs and rhs over different blocks "
            "(e.g. a `:` slice against a tile of the same extent); index both "
            "operands' K with the same tile or slice"
        )
    blocks = [b for b in (m_block, k_block, n_block) if b is not None]
    if len(set(blocks)) != len(blocks):
        raise _unsupported(
            f"hl.dot of {shapes} indexes two of M, K, N with one block "
            "(equal-sized `:` slices share a block); a Metal thread holds one "
            "index per block, so it would only cover their diagonal"
        )

    f32 = torch.float32
    result = expr_from_string(
        "{lhs} * {rhs}",
        lhs=backend.cast_ast(state.ast_arg(0), f32),
        rhs=backend.cast_ast(state.ast_arg(1), f32),
    )
    if k_block is not None:
        if env.block_sizes[k_block].reduction:
            strategy = state.device_function.tile_strategy.get_reduction_strategy(
                k_block
            )
        else:
            strategy = backend.create_block_reduction_strategy(state, k_block)
        # A dot's reduction dimension is never rollable
        # (``ReductionRoller.has_matmul_with_rdim``).
        assert not isinstance(strategy, LoopedReductionStrategy)
        if (
            isinstance(strategy, MetalPersistentReductionStrategy)
            and strategy.serializes_lanes
        ):
            # Strategy creation folds only a dim a reduction contracts, so a
            # dot's K is laned without a fold, and a sum across threads alone
            # would keep one lane's partial.
            strategy.fold_lanes()
        product = state.codegen.lift(result, dce=True, prefix="dot_product").id
        fake_product = lhs.to(f32).unsqueeze(-1) * rhs.to(f32).unsqueeze(-3)
        result = strategy.codegen_reduction(
            state, product, "sum", 1, fake_product, fake_product.sum(1)
        )
    if acc is not None:
        result = expr_from_string(
            "{acc} + {result}",
            acc=backend.cast_ast(state.ast_arg(2), f32),
            result=result,
        )
    # The node's dtype is out_dtype, or the acc's dtype when there is an acc.
    return backend.cast_ast(result, out.dtype)
