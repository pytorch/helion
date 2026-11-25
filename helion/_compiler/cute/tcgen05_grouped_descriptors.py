"""Static admission shared by grouped device and wrapper descriptor codegen.

The source-width projection remains responsible for checking every dynamic
interval. This proof only permits converting its *proved element bases* to row
coordinates. It never reads offsets or substitutes narrowed offset arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ... import exc

DESCRIPTOR_KEY = "cute_grouped_descriptor_policy"
DYNAMIC = "dynamic"
WRAPPED = "wrapped"
WRAP_EXTENT = 1 << 30
WRAP_STRIDE = 1 << 34
EXTRA_EXTENT = (1 << 31) - 1
# The flat grouped kernel's row-packed buffers hold fp32.
ELEMENT_BYTES = 4
# Bytes of elements from which ``cuTensorMapEncodeTiled`` selects the large
# allocation class, and the window a large-class descriptor needs accessible
# from its global address (PTX ISA 9.5, "Out of Boundary Access";
# ``tensormap_alloc_class`` describes the class).
LARGE_ALLOC_MIN_BYTES = 1 << 17


def wrapped_grouped_descriptor_supported(
    groups: int, rows: int, reduction: int, columns: int, offset_bits: int
) -> bool:
    """Contiguous FP32 metadata, after the typed flat-provider admission.

    A power-of-two row stride divides both source moduli (2**32 and 2**64).
    Thus a wrapped source product remains divisible by the stride. Input and
    output bases can differ and must each be divided independently. Existing
    projection proves nonnegative bases and complete spans before narrowing.

    Each buffer has <2**31 elements and strides >=4, hence admitted coordinates
    (base/stride + length) are <2**29. B-length is positive for B=2**30; B and
    every extra coordinate fit their descriptor extent. Outer byte strides are
    positive, 16-byte aligned and <2**40. At a live row i the element address is
    (B-length+i)*stride + B*(2**34-stride) + (base/stride+length)*stride,
    equal to 2**64+base+i*stride. At i>=length the first dimension is OOB.

    Those OOB rows are why both row-packed buffers must span the large-class
    window (``LARGE_ALLOC_MIN_BYTES``; ``cuTensorMapEncodeTiled`` gives a
    wrapped descriptor that class for its virtual extents, whatever the buffer
    holds): the descriptor's global address is the buffer's first element, and
    a load that leaves a group issues its out-of-bounds requests anywhere in
    the 128 KiB after it. A shorter input was read up to 128 KiB past its end
    on every launch. The output is only stored to, which issues no such
    request on sm_100, but its descriptor carries the same class. Bindings
    with a shorter buffer keep the dynamic policy.
    """
    return (
        all(type(v) is int for v in (groups, rows, reduction, columns, offset_bits))
        and offset_bits in (32, 64)
        and 0 < groups < EXTRA_EXTENT
        and 0 < rows < WRAP_EXTENT
        and all(
            4 <= v < EXTRA_EXTENT and v & (v - 1) == 0 for v in (reduction, columns)
        )
        and max(rows * reduction, rows * columns, groups * reduction * columns)
        <= EXTRA_EXTENT
        and ELEMENT_BYTES * rows * min(reduction, columns) >= LARGE_ALLOC_MIN_BYTES
    )


@dataclass(frozen=True)
class WrappedGroupedDescriptorPlan:
    groups: int
    rows: int
    reduction: int
    columns: int
    offset_bits: int

    def __post_init__(self) -> None:
        if not wrapped_grouped_descriptor_supported(
            self.groups, self.rows, self.reduction, self.columns, self.offset_bits
        ):
            raise ValueError("unsupported wrapped grouped descriptor metadata")

    def cache_identity(self) -> tuple[object, ...]:
        return (
            "wrapped_flat_grouped_descriptors_v1",
            self.groups,
            self.rows,
            self.reduction,
            self.columns,
            self.offset_bits,
            WRAP_EXTENT,
            WRAP_STRIDE,
            EXTRA_EXTENT,
        )

    @classmethod
    def from_identity(cls, identity: object) -> WrappedGroupedDescriptorPlan:
        """The plan that generated source names in its wrapper plans, as the
        launcher reads it (``host_wrappers.wrapped_grouped_descriptors``).

        Source that left the process (``BoundKernel.to_code``) can name the
        wrapped policy for metadata this module does not admit: source
        generated without the large-class window, for a row-packed buffer
        below it. The launch is refused with what to do about it; generated
        again, such a binding keeps the dynamic policy.
        """
        if not isinstance(identity, tuple) or len(identity) != 9:
            raise ValueError("invalid wrapped descriptor identity")
        metadata = identity[1:6]
        if not wrapped_grouped_descriptor_supported(*metadata):
            raise exc.BackendUnsupported(
                "cute",
                "wrapped grouped descriptors for (groups, rows, reduction, "
                f"columns, offset bits) = {metadata}, which the wrapped policy "
                "does not admit. Among its conditions, both row-packed buffers "
                f"span {LARGE_ALLOC_MIN_BYTES} bytes; source generated without "
                "that one has to be generated again (BoundKernel.to_code)",
            )
        plan = cls(*metadata)
        if plan.cache_identity() != identity:
            raise ValueError("inconsistent wrapped descriptor identity")
        return plan

    def layout(self, arg: str, *, output: bool = False) -> str:
        """The one rank-four recipe used by both A/B and D wrapper emission."""
        shape = (
            f"{arg}_shape1, {WRAP_EXTENT}" if output else f"{WRAP_EXTENT}, {arg}_shape1"
        )
        stride = (
            f"{arg}_stride1, {arg}_stride0"
            if output
            else f"{arg}_stride0, {arg}_stride1"
        )
        return (
            f"({shape}, {EXTRA_EXTENT}, {EXTRA_EXTENT}), "
            f"stride=({stride}, cutlass.Int64({WRAP_STRIDE}) - {arg}_stride0, "
            f"{arg}_stride0)"
        )
