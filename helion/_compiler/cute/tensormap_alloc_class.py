"""Small allocation class for the TensorMaps a kernel rewrites on the device.

A tiled TensorMap carries an *allocation class* (PTX ISA 9.5, "Out of Boundary
Access" and ``tensormap.replace``; ``cuTensorMapSetAllocationClass`` in the
CUDA 13.5 driver API). A TMA load whose box leaves the tensor still issues
memory requests for the elements outside it, at addresses the unit generates
from the descriptor's global address: anywhere in the 128 KiB that follow it
in the large class, inside a 4 KiB window that is always mapped in the small
one. The large class is therefore only valid while ``[global_address,
global_address + 128 KiB)`` is accessible. Otherwise the behavior is
undefined, and on sm_100 the load reads up to 128 KiB past a shorter tensor:
an illegal memory access when that memory is unmapped. Measured there, a
small-class load touches nothing outside its tensor, and a store issues no
such request in either class.

``cuTensorMapEncodeTiled`` picks the class from the encoded geometry: large
once the tensor holds 128 KiB of elements, so that the window is the tensor's
own memory. A grouped kernel encodes one template for a whole operand and then
rewrites its address, extents and strides per group
(``TensorMapManager.update_tensormap``), so a group shorter than 128 KiB
inherited the large class of the operand it is cut from. PTX ISA 9.5 defines
``tensormap.replace`` of ``global_address`` to reset the class to small for
this reason. The CuTe DSL's toolchain is at PTX ISA 9.4, and the rewrite the
kernels use (``cpasync.update_tma_descriptor``, under ``update_tensormap``)
leaves the selector as encoded: measured with CuTe DSL 4.8.1, where a
template that is rewritten and not cleared still loads past a short group.

``select_small_alloc_class`` gives the kernel's shared-memory template the
class that newer ``tensormap.replace`` leaves behind, once, right after
``init_tensormap_from_atom``: every descriptor published from the template is
small, whatever group it describes. A template needs it unless every group it
describes has the template's own extent. The clear has two forms: one elected
lane of the owning warp (the general grouped path) or every lane of it (the
flat grouped kernel, ``select_small_alloc_class_every_lane``). Either way the
clear is ordered ahead of the template's first rewrite by a barrier: the
elected form ends in the warp's, the other leaves it to its caller.

The selector is bit 21 of the second quadword of the descriptor
(``ALLOC_CLASS_QWORD``, ``ALLOC_CLASS_LARGE``). NVIDIA's documentation does
not give that position. The code of its CuTe DSL does, from 4.8.1, the
release this backend requires: the experimental ``tensormap_replace``
(``cutlass.experimental.primitives.nvvm_wrapper``) resets a descriptor copy
to "the always-safe out-of-bounds fetch mode" by clearing bit 21 of the
descriptor's 32-bit word 2, the same bit, when it has replaced
``global_dim`` or ``global_stride``: on every device and without asking the
driver, unless its caller passes ``keep_oob_fetch_window=True`` (legal, it
says, only while 128 KiB of mapped memory follow the address). The kernels
do not rewrite through that primitive, so the helper clears the bit as the
primitive does: on every device, in every template a kernel rewrites to
another extent. A test holds the two constants to the DSL's
(``test_cute_grouped_descriptors.py``), so a release that moves the bit
fails there instead of diverging from this module.

Besides the DSL's code, one device stands behind the clear. On a B200 with
driver 580.82.07 the bit is the one the driver's encoder changes, next to
the extent, between a tensor just below and one at 128 KiB, and a
descriptor with the bit cleared by hand loads like one the driver encoded
small; the kernels' page-touch tests (``helion._testing.cuda_page_touch``)
see no page touched past a short last group with the clear, and pages
touched without it. No device is known whose encoder puts the class
elsewhere.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

# The selector inside the 16-quadword descriptor. CuTe DSL 4.8.1 clears the
# same bit as bit 21 of 32-bit word 2 (``_TENSORMAP_FIXUP_WORD`` and
# ``_TENSORMAP_FIXUP_MASK`` of ``cutlass.experimental.primitives.nvvm_wrapper``).
ALLOC_CLASS_QWORD = 1
ALLOC_CLASS_LARGE = 1 << 21


@cute.jit
def select_small_alloc_class(
    tensormap_smem: cute.Pointer,
    warp_id: cutlass.Constexpr[int],
) -> None:
    """Give the template at ``tensormap_smem`` the small allocation class.

    Call once, after the ``init_tensormap_from_atom`` of the same pointer and
    ``warp_id`` and before the template's first ``update_tensormap``.

    One lane of the owning warp clears it and the warp reconverges behind
    that lane, as ``init_tensormap_from_atom`` does around the copy, so no
    lane rewrites or publishes the template before the clear.
    """
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == warp_id:
        with cute.arch.elect_one():
            selector = tensormap_smem + ALLOC_CLASS_QWORD
            encoded = selector.load()  # pyrefly: ignore[missing-attribute]
            selector.store(  # pyrefly: ignore[missing-attribute]
                encoded & cutlass.Int64(~ALLOC_CLASS_LARGE)
            )
    cute.arch.sync_warp()


@cute.jit
def select_small_alloc_class_every_lane(
    tensormap_smem: cute.Pointer,
    warp_id: cutlass.Constexpr[int],
) -> None:
    """``select_small_alloc_class`` without the election.

    Every lane of the owning warp loads the quadword and stores it with the
    selector cleared. The lanes store one value, whichever of them a lane's
    load follows, so the warp needs no election and no reconvergence. The
    stores of different lanes are not ordered among each other or ahead of
    what one lane does next, though: the caller puts a barrier of the warp
    (``cute.arch.sync_warp``) or of the block between the call and the
    template's first rewrite, where a single lane replaces fields of the
    template and a store that landed later would undo what it changed in
    this quadword.
    """
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == warp_id:
        selector = tensormap_smem + ALLOC_CLASS_QWORD
        encoded = selector.load()  # pyrefly: ignore[missing-attribute]
        selector.store(  # pyrefly: ignore[missing-attribute]
            encoded & cutlass.Int64(~ALLOC_CLASS_LARGE)
        )
