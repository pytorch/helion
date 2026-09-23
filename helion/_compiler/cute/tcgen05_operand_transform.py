"""Dormant plans for a raw-TMA operand conversion before native MMA.

No lowering selects these plans yet. A caller must separately prove its input,
role, output and allocation contracts before using the pipeline helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from operator import itemgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cutlass.cute as cute


class Tcgen05OperandType(str, Enum):
    FLOAT32 = "cutlass.Float32"
    TFLOAT32 = "cutlass.TFloat32"


class Tcgen05OperandTransformKind(str, Enum):
    TF32_RNA = "tf32_rna"


@dataclass(frozen=True)
class Tcgen05OperandRolePlan:
    """One-CTA roles for the proved producer/converter/MMA/output schedule.

    Four epilogue warps, one MMA warp and one raw-TMA warp precede conversion;
    one scheduler warp follows it. Padding completes the physical warpgroups
    but never consumes scheduler work. Helion's mailbox producer does not
    consume its own work; an explicitly self-consuming scheduler adds one
    arrival. The caller must use these role guards and mailbox participants.
    """

    converter_warps: int = 1
    scheduler_self_consumer: bool = False

    def __post_init__(self) -> None:
        if type(self.converter_warps) is not int or self.converter_warps not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("the proved converter warp choices are 1, 2, 4 and 8")
        if type(self.scheduler_self_consumer) is not bool:
            raise ValueError("scheduler self-consumption requires an explicit bool")

    @property
    def epilogue_warp_ids(self) -> tuple[int, ...]:
        return (0, 1, 2, 3)

    @property
    def mma_warp(self) -> int:
        return 4

    @property
    def raw_tma_warp(self) -> int:
        return 5

    @property
    def converter_first_warp(self) -> int:
        return 6

    @property
    def converter_warp_ids(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.converter_first_warp,
                self.converter_first_warp + self.converter_warps,
            )
        )

    @property
    def scheduler_warp(self) -> int:
        return self.converter_first_warp + self.converter_warps

    @property
    def active_warps(self) -> int:
        return self.scheduler_warp + 1

    @property
    def threads_per_cta(self) -> int:
        return ((self.active_warps + 3) // 4) * 128

    @property
    def thread_block_dims(self) -> tuple[int, int, int]:
        return (self.threads_per_cta, 1, 1)

    @property
    def padding_warp_ids(self) -> tuple[int, ...]:
        return tuple(range(self.active_warps, self.threads_per_cta // 32))

    @property
    def converted_full_lane_arrivals(self) -> int:
        return self.converter_warps * 32

    @property
    def scheduler_empty_warp_arrivals(self) -> int:
        return len(self.scheduler_consumer_warp_ids)

    @property
    def scheduler_consumer_warp_ids(self) -> tuple[int, ...]:
        return tuple(range(self.scheduler_warp + int(self.scheduler_self_consumer)))

    def converter_rank(self, physical_warp: int) -> int:
        """Host proof of the relative rank passed under the conversion guard."""
        if type(physical_warp) is not int or physical_warp not in (
            self.converter_warp_ids
        ):
            raise ValueError("the physical warp does not own a conversion role")
        return physical_warp - self.converter_first_warp

    def cache_identity(self) -> tuple[object, ...]:
        return (
            "tcgen05_operand_roles_v1",
            ("epilogue_warps", self.epilogue_warp_ids),
            ("mma_warp", self.mma_warp),
            ("raw_tma_warp", self.raw_tma_warp),
            ("converter_warps", self.converter_warp_ids),
            ("scheduler_warp", self.scheduler_warp),
            ("padding_warps", self.padding_warp_ids),
            ("threads_per_cta", self.threads_per_cta),
            ("thread_block_dims", self.thread_block_dims),
            ("scheduler_self_consumer", self.scheduler_self_consumer),
            ("scheduler_consumer_warps", self.scheduler_consumer_warp_ids),
            ("converted_full_lane_arrivals", self.converted_full_lane_arrivals),
            ("scheduler_empty_warp_arrivals", self.scheduler_empty_warp_arrivals),
        )


@dataclass(frozen=True)
class Tcgen05OperandResourceUsage:
    """Account actual whole-kernel usage, including inactive physical warps.

    Shared sizes must come from the full compiled object and launch, not from
    summing the operand spans: allocator padding, metadata and epilogues also
    consume storage. Register allocation is uniform; a caller with runtime
    register redistribution needs a separate proof. This dormant record is not
    a runtime admission decision.
    """

    roles: Tcgen05OperandRolePlan
    registers_per_thread: int
    static_shared_bytes: int
    dynamic_shared_bytes: int
    register_redistribution: bool = False

    def __post_init__(self) -> None:
        if self.register_redistribution is not False:
            raise ValueError(
                "runtime register redistribution needs separate accounting"
            )
        if (
            type(self.registers_per_thread) is not int
            or not 1 <= self.registers_per_thread <= 255
        ):
            raise ValueError("invalid compiled SM100 register count")
        if any(
            type(size) is not int or size < 0
            for size in (self.static_shared_bytes, self.dynamic_shared_bytes)
        ):
            raise ValueError("shared storage requires nonnegative byte counts")

    @property
    def allocated_registers_per_cta(self) -> int:
        # SM100 allocates registers in 256-register units per physical warp.
        registers_per_warp = ((self.registers_per_thread * 32 + 255) // 256) * 256
        return registers_per_warp * (self.roles.threads_per_cta // 32)

    @property
    def shared_bytes(self) -> int:
        return self.static_shared_bytes + self.dynamic_shared_bytes

    def fits(self, *, shared_capacity_bytes: int) -> bool:
        """Use the existing device/launch shared limit; never shrink stages."""
        if type(shared_capacity_bytes) is not int or shared_capacity_bytes <= 0:
            raise ValueError("shared capacity must be an actual positive byte limit")
        return (
            self.roles.threads_per_cta <= 1024
            and self.allocated_registers_per_cta <= 65536
            and self.shared_bytes <= shared_capacity_bytes
        )


@dataclass(frozen=True)
class Tcgen05OperandDescriptorPlan:
    source_type: Tcgen05OperandType
    tma_internal_type: Tcgen05OperandType
    mma_type: Tcgen05OperandType

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, Tcgen05OperandType)
            for value in (self.source_type, self.tma_internal_type, self.mma_type)
        ):
            raise ValueError(
                "native operand descriptor types must be typed enum values"
            )

    def cache_identity(self) -> tuple[str, str, str]:
        return (
            self.source_type.value,
            self.tma_internal_type.value,
            self.mma_type.value,
        )


@dataclass(frozen=True)
class Tcgen05OperandStageSpan:
    """Physical Float32 words belonging to one stage of a proved SDK layout.

    Construct through ``prove_operand_stage_span``. ``alignment_bytes`` is an
    existing allocation fact, never an alignment assumption for a new pointer.
    ``layout_identity`` retains logical layout information for descriptor keys.
    """

    words: int
    stride_words: int
    stages: int
    alignment_bytes: int
    layout_identity: tuple[object, ...]

    @property
    def allocation_bytes(self) -> int:
        return ((self.stages - 1) * self.stride_words + self.words) * 4

    def cache_identity(self) -> tuple[object, ...]:
        return (
            self.words,
            self.stride_words,
            self.stages,
            self.alignment_bytes,
            self.layout_identity,
        )


@dataclass(frozen=True)
class Tcgen05OperandTransformPlan:
    kind: Tcgen05OperandTransformKind
    descriptor: Tcgen05OperandDescriptorPlan
    a: Tcgen05OperandStageSpan
    b: Tcgen05OperandStageSpan
    converter_warps: int = 1
    scheduler_self_consumer: bool = False

    def __post_init__(self) -> None:
        Tcgen05OperandRolePlan(self.converter_warps, self.scheduler_self_consumer)
        if self.kind is not Tcgen05OperandTransformKind.TF32_RNA:
            raise ValueError("unsupported native operand transform")
        if self.descriptor != Tcgen05OperandDescriptorPlan(
            Tcgen05OperandType.FLOAT32,
            Tcgen05OperandType.FLOAT32,
            Tcgen05OperandType.TFLOAT32,
        ):
            raise ValueError("TF32 RNA requires raw Float32 TMA and TFloat32 MMA")
        if self.a.stages != self.b.stages:
            raise ValueError("raw A/B stages must share one pipeline depth")
        for span in (self.a, self.b):
            if (
                type(span.stages) is not int
                or span.stages <= 0
                or type(span.words) is not int
                or span.words <= 0
                or span.words % (128 * self.converter_warps)
                or type(span.stride_words) is not int
                or span.stride_words < span.words
                or span.stride_words % 4
                or type(span.alignment_bytes) is not int
                or span.alignment_bytes < 16
                or span.alignment_bytes & (span.alignment_bytes - 1)
            ):
                raise ValueError("invalid proved Float32 stage span")
            # Stage and lane arithmetic in the helper is Int32. Every packet's
            # first word and all four addressed words must remain nonnegative.
            if span.allocation_bytes > (1 << 31) * 4:
                raise ValueError("the operand stage offset exceeds the Int32 domain")

    @property
    def stages(self) -> int:
        return self.a.stages

    @property
    def roles(self) -> Tcgen05OperandRolePlan:
        return Tcgen05OperandRolePlan(
            self.converter_warps, self.scheduler_self_consumer
        )

    @property
    def converted_full_arrivals(self) -> int:
        # One publication from every lane after that lane's stores and fence.
        return 32 * self.converter_warps

    @property
    def scheduler_consumer_warps(self) -> int:
        # A different channel: one scheduler release per conversion warp.
        return self.converter_warps

    @property
    def converted_full_barrier_bytes(self) -> int:
        # The original UMMA-empty barriers are shared, never duplicated here.
        return 8 * self.stages

    def cache_identity(self) -> tuple[object, ...]:
        """Fields that a future wrapper must include in its existing cache key.

        Compiled helper source hashes remain an additional required dependency;
        this dormant plan does not register a new runtime wrapper family.
        """
        return (
            "tcgen05_operand_transform_v2",
            self.kind.value,
            self.descriptor.cache_identity(),
            self.a.cache_identity(),
            self.b.cache_identity(),
            ("converter_warps", self.converter_warps),
            ("converted_full_thread_arrivals", self.converted_full_arrivals),
            ("scheduler_release_warps", self.scheduler_consumer_warps),
            ("roles", self.roles.cache_identity()),
            "cta_group_one",
        )


def _static_modes(shape: object, stride: object) -> list[tuple[int, int]] | None:
    if type(shape) is int and type(stride) is int:
        return [(shape, stride)] if shape > 0 and stride >= 0 else None
    if not isinstance(shape, tuple) or not isinstance(stride, tuple):
        return None
    if len(shape) != len(stride):
        return None
    result = []
    for child_shape, child_stride in zip(shape, stride, strict=True):
        child = _static_modes(child_shape, child_stride)
        if child is None:
            return None
        result.extend(child)
    return result


def prove_operand_stage_span(
    layout: cute.Layout | cute.ComposedLayout,
    *,
    stages: int,
    alignment_bytes: int,
) -> Tcgen05OperandStageSpan | None:
    """Prove disjoint dense physical spans for a static last-mode stage ring.

    Compact permutations of the inner modes are allowed. A swizzle must preserve
    each aligned span: all XOR destination bits lie within its power-of-two word
    extent, and the base/stride keep those bits within the allocation. The XOR
    source bits may include stage/base bits; they only permute the low positions.
    No logical-coordinate enumeration or runtime shape hint is used as a proof.
    """
    import cutlass.cute as cute

    if (
        type(stages) is not int
        or stages <= 0
        or type(alignment_bytes) is not int
        or alignment_bytes < 16
        or alignment_bytes & (alignment_bytes - 1)
    ):
        return None
    swizzle: tuple[int, int, int] | None = None
    if isinstance(layout, cute.ComposedLayout):
        inner = layout.inner
        offset = layout.offset
        if (
            not isinstance(inner, cute.Swizzle)
            or type(offset) is not int
            or offset != 0
        ):
            return None
        swizzle = (inner.num_bits, inner.num_base, inner.num_shift)
        outer = layout.outer
    else:
        outer = layout
    shape, stride = outer.shape, outer.stride
    if (
        not isinstance(shape, tuple)
        or not isinstance(stride, tuple)
        or len(shape) != len(stride)
        or len(shape) < 2
        or type(shape[-1]) is not int
        or shape[-1] != stages
        or type(stride[-1]) is not int
    ):
        return None
    modes = _static_modes(shape[:-1], stride[:-1])
    if modes is None:
        return None
    words = 1
    ordered_modes = sorted(
        ((size, step) for size, step in modes if size != 1), key=itemgetter(1)
    )
    for extent, step in ordered_modes:
        if step != words:
            return None
        words *= extent
    # CuTe canonicalizes the sole stage's stride to zero. Its only reachable
    # coordinate is zero, so a physical stride of ``words`` is equivalent and
    # keeps the converter's arithmetic common to all ring depths.
    stage_stride = words if stages == 1 and stride[-1] == 0 else stride[-1]
    if words % 128 or stage_stride < words or stage_stride % 4:
        return None
    if swizzle is not None:
        bits, base, shift = swizzle
        if bits < 0 or base < 0 or abs(shift) < bits:
            return None
        if bits:
            destination_end = base + max(0, -shift) + bits
            span_alignment = 1 << destination_end
            if (
                words & (words - 1)
                or span_alignment > words
                or stage_stride % words
                or alignment_bytes < 4 * span_alignment
            ):
                return None
    return Tcgen05OperandStageSpan(
        words=words,
        stride_words=stage_stride,
        stages=stages,
        alignment_bytes=alignment_bytes,
        layout_identity=(shape, stride, swizzle),
    )
