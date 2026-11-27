"""Bounded store-only rank-3 TensorMaps for live grouped row intervals."""

from __future__ import annotations

# Whether the autotuner proposes ``tcgen05_grouped_d_shift``: compiler seeds
# that carry it and the knob as a search coordinate.  Off, because the shift
# stores through a descriptor whose out-of-bounds behaviour PTX leaves
# undefined (``d_shift_searched``).  An explicit config and a cached result
# select the shift whatever this says.
D_SHIFT_SEARCHED = False


def d_shift_searched() -> bool:
    """Whether a search may propose the bounded D shift on its own.

    The shift's store descriptor is static, has the extents (M, N, M + 1), or
    (N, M, M + 1) for an N,M worklist, and is based M rows below the output.
    ``cuTensorMapEncodeTiled`` selects the large allocation class for it once
    those extents hold 128 KiB, which is nearly every shape (N = 64 in bf16
    from M = 32, an output of 4 KiB), and PTX ISA 9.5 ("Out of Boundary
    Access") leaves a large-class operation whose box leaves the tensor
    undefined unless the 128 KiB from the descriptor's address are
    accessible.  Here that is memory below the output, which the kernel does
    not own, and every group whose rows do not fill its last tile stores such
    a box.

    Measured, on a B200 with driver 580.82.07 only: a TMA store through a
    descriptor of this geometry writes the in-bounds bytes and touches no
    other page, while a load through it does read below the output.  The
    shift only stores, so nothing went wrong there; no other device was
    measured, and the documentation promises nothing.

    So no candidate of a search stores through the shift.  Its seeds leave
    the compiler's palette: the shifted copies of the full-allocation seeds
    and the paired prefix recipes, which are legal only on the shift.  The
    M,N worklist tiles stay, seeded without the shift: such a tile stores
    through its group's own descriptor, which the kernel rebuilds per group
    at the group's first row inside the output, as the N,M seeds do.  The
    knob and the recipes' knob are no coordinates of a search
    (``tcgen05_config.surface.flat_fields``), so no draw or mutation names
    them, and a seed, or a cached config that seeds a search, comes to the
    search on its flat fields, without them.  A cache hit starts no search
    and is compiled as saved, with the shift where its entry names it.  An
    LLM's reply is merged and normalized like any caller's config and can
    name them unprompted; the search takes them out of such a proposal and
    benchmarks what is left, the same worklist without the shift
    (``tcgen05_config.surface.repair_search_proposal``).  A config that
    names the shift is validated, normalized and compiled as before.

    The gate is about the shift's descriptor and vouches for no other.  The
    group's own descriptor is the kernel's copy of the output's host-encoded
    one, and the per-group rewrite replaces the copy's address, extents and
    strides and leaves its allocation class as it is, so a base inside the
    output does not settle the same rule for it: its tail stores are defined
    where the copy is small-class or 128 KiB of the output remain from the
    group's first row.  ``KNOWN_GAPS.md`` lists the descriptors that do not
    meet this.

    The gate can go once the shift's descriptor is small-class: a CuTe DSL
    that emits PTX ISA 9.5's ``.alloc_small``,
    ``cuTensorMapSetAllocationClass`` (CUDA 13.5) on a descriptor Helion
    encodes itself, or a copy that the kernel makes and clears on the device
    as CuTe DSL 4.8.1's experimental ``tensormap_replace`` clears a
    descriptor whose ``global_dim`` or ``global_stride`` it replaces (bit 21
    of 32-bit word 2); or once the tail tiles are addressed through a
    descriptor that is based inside the output and is itself small-class or
    keeps 128 KiB of the output from its base.
    """
    return D_SHIFT_SEARCHED


def bounded_d_shift_index_domain(m: int, n: int, row_tile: int) -> bool:
    """Bound element offsets and every padded signed TMA coordinate.

    A descriptor based at D-M*stride_m has row extent M and an extra extent
    M+1. For a frontend-clipped interval [start, start+length), coordinates
    (M-length+row, column, start+length) address D+(start+row)*stride_m
    while the row extent rejects every padded row. The shifted pointer is
    never used for a load.
    """
    limit = (1 << 31) - 1
    return (
        min(m, n, row_tile) > 0
        and m + 1 <= limit
        and m * n <= limit
        and m + row_tile - 1 <= limit
    )
