"""Metal-backend codegen for a matmul on the scalar contraction.

``dot(a, b)[m, n] = sum_k f32(a[m, k]) * f32(b[k, n])``, or the same sum in
int32 for int8 operands, wrapping like ``tl.dot``'s int32 accumulator.  Each
thread forms the product at its own ``(m, k, n)`` (and batch index), and K's
own reduction strategy combines it, the way the scalar path lowers
``(a[:, :, None] * b[None]).sum(1)``.  A K held as per-thread parts
(``factored_dims``) sums each thread's parts' products first.
Helion masks every matmul operand to 0 past each block's extent
(``_mask_to``), so a K tail or a surplus K thread adds nothing.

Every matmul of a ``metal_matmul="scalar"`` config lowers here: ``hl.dot``
and the aten matmuls (``mpp_matmul._codegen_aten_matmul``).  Under the other
choices, a matmul the MPP planner admits (``mpp_matmul.plan_mpp_matmuls``) is
forwarded to ``mpp_matmul.codegen_dot_metal``, and one it leaves lowers here.
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
    from .mpp_matmul import codegen_dot_metal
    from .mpp_matmul import planned

    if planned(state.fx_node):
        return codegen_dot_metal(state)
    return scalar_contraction(state)


def scalar_contraction(state: CodegenState) -> ast.AST:
    """The scalar contraction of ``state``'s matmul, whose args are
    ``hl.dot``'s: lhs, rhs, acc (None for none) and out_dtype (None for
    none).  An aten matmul the MPP planner doesn't take lowers here too
    (``mpp_matmul._codegen_aten_matmul``)."""
    # Imported here: reduction_strategy imports device_ir, which needs the
    # helion package that is still initializing when this module registers.
    from ..reduction_strategy import LoopedReductionStrategy
    from .factored_dims import FactoredValue
    from .reduction_strategy import MetalPersistentReductionStrategy

    env = CompileEnvironment.current()
    backend = env.backend
    lhs, rhs, acc, out_dtype = state.proxy_args
    assert isinstance(lhs, torch.Tensor)
    assert isinstance(rhs, torch.Tensor)
    assert acc is None or isinstance(acc, torch.Tensor)
    assert out_dtype is None or isinstance(out_dtype, torch.dtype)
    assert state.fx_node is not None
    out = state.fx_node.meta["val"]
    if lhs.ndim != rhs.ndim or lhs.ndim not in (2, 3):
        raise _unsupported(
            f"a matmul of a {lhs.ndim}-D lhs and a {rhs.ndim}-D rhs on the scalar "
            "contraction; both must be 2-D, or both 3-D with one batch block"
        )
    if all(t.dtype in _FLOAT_DTYPES for t in (lhs, rhs, out)):
        compute = torch.float32
    elif lhs.dtype == rhs.dtype == torch.int8 and out.dtype == torch.int32:
        # An int8 product fits int32 exactly; the sums wrap past K ~ 2**17,
        # as tl.dot's int32 accumulator does.
        compute = torch.int32
    else:
        raise _unsupported(
            f"a matmul of {lhs.dtype} x {rhs.dtype} -> {out.dtype} on the scalar "
            "contraction; only float16, bfloat16 and float32, or int8 to int32, "
            "are supported"
        )

    shapes = f"{list(lhs.shape)} x {list(rhs.shape)}"
    lhs_ast, rhs_ast = state.ast_args[0], state.ast_args[1]
    factored = [isinstance(a, FactoredValue) for a in (lhs_ast, rhs_ast)]
    if any(factored):
        # A K of length c * block, held as c per-thread parts
        # (``factored_dims``): each thread sums its parts' products, and K's
        # reduction combines the threads.
        for side, value, k_dim in (
            ("lhs", lhs_ast, lhs.ndim - 1),
            ("rhs", rhs_ast, rhs.ndim - 2),
        ):
            if isinstance(value, FactoredValue) and value.dim != k_dim:
                raise _unsupported(
                    f"a matmul of {shapes} whose {side} holds its dim {value.dim}, "
                    "not K, as per-thread parts; only a K held as parts lowers "
                    "on the scalar contraction"
                )
        if not (
            isinstance(lhs_ast, FactoredValue)
            and isinstance(rhs_ast, FactoredValue)
            and (lhs_ast.block_id, lhs_ast.factor) == (rhs_ast.block_id, rhs_ast.factor)
        ):
            raise _unsupported(
                f"a matmul of {shapes} whose K only one operand holds as "
                "per-thread parts, or the two hold differently"
            )
        k_block = rhs_k_block = lhs_ast.block_id
        lhs_fake, rhs_fake = lhs_ast.part_fake, rhs_ast.part_fake
        terms = [
            expr_from_string(
                "{lhs} * {rhs}",
                lhs=backend.cast_ast(lp, compute),
                rhs=backend.cast_ast(rp, compute),
            )
            for lp, rp in zip(lhs_ast.parts, rhs_ast.parts, strict=True)
        ]
        result = terms[0]
        for term in terms[1:]:
            result = expr_from_string("{a} + {b}", a=result, b=term)
    else:
        k_block, rhs_k_block = map(_dim_block, (lhs.size(-1), rhs.size(-2)))
        lhs_fake, rhs_fake = lhs, rhs
        result = expr_from_string(
            "{lhs} * {rhs}",
            lhs=backend.cast_ast(state.ast_arg(0), compute),
            rhs=backend.cast_ast(state.ast_arg(1), compute),
        )
    m_block, n_block = map(_dim_block, (lhs.size(-2), rhs.size(-1)))
    batch: list[int | None] = []
    if lhs.ndim == 3:
        lhs_b, rhs_b = map(_dim_block, (lhs.size(0), rhs.size(0)))
        if lhs_b != rhs_b:
            raise _unsupported(
                f"a matmul of {shapes} whose operands' batch dims are different "
                "blocks; index both with the same tile or slice"
            )
        batch.append(lhs_b)
    # Raw ids, not canonical ones: a tile aliased to another block's size is
    # still its own loop with its own index.
    if rhs_k_block != k_block:
        raise _unsupported(
            f"a matmul of {shapes} contracts lhs and rhs over different blocks "
            "(e.g. a `:` slice against a tile of the same extent); index both "
            "operands' K with the same tile or slice"
        )
    blocks = [b for b in (*batch, m_block, k_block, n_block) if b is not None]
    if len(set(blocks)) != len(blocks):
        raise _unsupported(
            f"a matmul of {shapes} indexes two of its dims with one block "
            "(equal-sized `:` slices share a block); a Metal thread holds one "
            "index per block, so it would only cover their diagonal"
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
        fake_product = lhs_fake.to(compute).unsqueeze(-1) * rhs_fake.to(
            compute
        ).unsqueeze(-3)
        k_dim = fake_product.ndim - 2
        result = strategy.codegen_reduction(
            state,
            product,
            "sum",
            k_dim,
            fake_product,
            fake_product.sum(k_dim, dtype=compute),
        )
    if acc is not None:
        if out_dtype is not None and out_dtype != acc.dtype:
            # Triton rounds the product sum to out_dtype before adding acc.
            result = backend.cast_ast(backend.cast_ast(result, out_dtype), compute)
        result = expr_from_string(
            "{acc} + {result}",
            acc=backend.cast_ast(state.ast_arg(2), compute),
            result=result,
        )
    # The node's dtype is out_dtype, or the acc's dtype when there is an acc.
    return backend.cast_ast(result, out.dtype)
