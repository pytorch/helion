# ruff: noqa: ANN001
# Runtime support for the ``cute_mma_dag`` tcgen05 compute mode (see
# ``cute_mma_dag_tcgen05.py``).  The generated kernel imports this module as
# ``_helion_mma_dag_tc`` and reaches the CUTLASS primitives (``prims``), the
# TMA descriptor type (``cuda``) and the few helpers below through it; the
# host wrapper builds the tensor maps with :func:`make_tensor_map`.
from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import _ASM_KW
from ._flash_row_mma_runtime import lds_v4_f32  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401

__all__ = [
    "cuda",
    "make_tensor_map",
    "mbar_wait",
    "prims",
    "sts_v4_f32",
    "tma_swizzle",
]

TMEM_SPACE = 6


@cute.jit
def mbar_wait(mbar, parity) -> None:
    """Spin until the mbarrier phase with ``parity`` has completed."""
    while not prims.mbarrier_wait_parity(mbar, parity, prims.MBarrierWait.TRY):
        pass


@dsl_user_op
def sts_v4_f32(
    smem_address: object,
    x0: object,
    x1: object,
    x2: object,
    x3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v4.f32``: one 16-byte packet of four fp32 values."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


def tma_swizzle(name: str) -> cuda.TensorMapSwizzle:
    return {
        "none": cuda.TensorMapSwizzle.none,
        "s32b": cuda.TensorMapSwizzle.s32b,
        "s64b": cuda.TensorMapSwizzle.s64b,
        "s128b": cuda.TensorMapSwizzle.s128b,
    }[name]


@cute.jit
def make_tensor_map(
    tensor: cute.Tensor,
    dims: cutlass.Constexpr,
    box: cutlass.Constexpr,
    swizzle: cutlass.Constexpr,
) -> cuda.TensorMap:
    """A tiled TMA descriptor over the tensor dims ``dims`` (innermost first)
    with the box ``box`` (same order) and the named swizzle.

    Runs in the host wrapper: ``tensor`` is the wrapper's view of the kernel
    argument, ``dims`` the positions of the tensor's modes in TMA order.
    """
    shape = tuple(tensor.shape[d] for d in dims)  # pyrefly: ignore[bad-index]
    # The innermost dim is the operand's contiguous dim (proven at compile
    # time); the descriptor needs that unit stride as a static value.
    stride = (1, *(tensor.stride[d] for d in dims[1:]))  # pyrefly: ignore[bad-index]
    view = cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))
    return cuda.create_tensor_map_tiled_from_view(
        view,
        box_dims=tuple(box),
        stride_order=tuple(range(len(dims))),
        swizzle=tma_swizzle(swizzle),
    )
