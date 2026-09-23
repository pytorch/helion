"""Static admission shared by grouped device and wrapper descriptor codegen.

The source-width projection remains responsible for checking every dynamic
interval. This proof only permits converting its *proved element bases* to row
coordinates. It never reads offsets or substitutes narrowed offset arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

DESCRIPTOR_KEY = "cute_grouped_descriptor_policy"
DYNAMIC = "dynamic"
WRAPPED = "wrapped"
WRAP_EXTENT = 1 << 30
WRAP_STRIDE = 1 << 34
EXTRA_EXTENT = (1 << 31) - 1


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
        if not isinstance(identity, tuple) or len(identity) != 9:
            raise ValueError("invalid wrapped descriptor identity")
        plan = cls(*identity[1:6])
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
