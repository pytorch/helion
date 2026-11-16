# ruff: noqa: ANN001
# Runtime support for the ``cute_mma_dag`` tcgen05 compute mode (see
# ``cute_mma_dag_tcgen05.py``).  The generated kernel imports this module as
# ``_helion_mma_dag_tc`` and reaches the CUTLASS primitives (``prims``), the
# TMA descriptor type (``cuda``), the ``device/ptx.py`` rows and ``device/sync``
# waits re-exported here and the few helpers below through it; the host
# wrapper builds the tensor maps with :func:`make_tensor_map`.
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401
from .device.ptx import atom_add_release_gpu_u32
from .device.ptx import fence_acq_rel_gpu
from .device.ptx import ld_acquire_gpu_u32
from .device.ptx import ldcg_f32
from .device.ptx import lds_v4_f32  # noqa: F401
from .device.ptx import sts_v4_f32
from .device.sync import mbar_wait
from .device.sync import mbar_wait_cluster
from .device.sync import mbar_wait_sleep
from .device.sync import st_async_remote_f32

__all__ = [
    "atom_add_release_gpu_u32",
    "cuda",
    "fence_acq_rel_gpu",
    "ld_acquire_gpu_u32",
    "ldcg_f32",
    "make_tensor_map",
    "mbar_wait",
    "mbar_wait_cluster",
    "mbar_wait_sleep",
    "prims",
    "spin_until_reached_acquire_gpu_u32",
    "st_async_remote_f32",
    "sts_v4_f32",
    "tma_swizzle",
]

TMEM_SPACE = 6


def tma_swizzle(name: str) -> cuda.TensorMapSwizzle:
    return {
        "none": cuda.TensorMapSwizzle.none,
        "s32b": cuda.TensorMapSwizzle.s32b,
        "s64b": cuda.TensorMapSwizzle.s64b,
        "s128b": cuda.TensorMapSwizzle.s128b,
        "s128b_atom_32b": cuda.TensorMapSwizzle.s128b_atom_32b,
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


# -- the fixup split's global exchange (``emit_fixup_allreduce``) ---------------


@cute.jit
def spin_until_reached_acquire_gpu_u32(gmem_address, target) -> None:
    """Spin until the u32 counter at ``gmem_address`` reached ``target`` in
    modular arithmetic: the counter only grows and wraps, so "reached" is
    ``counter - target`` (mod 2^32) below 2^31."""
    while (ld_acquire_gpu_u32(gmem_address) - target) >= cutlass.Uint32(2147483648):
        pass
