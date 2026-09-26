"""Schedule admission for the proved flattened grouped program.

These records do not register a runtime implementation or autotuner choice.
They connect original static CUDA IR, binding-key-backed alignment, the typed
provider and the complete simultaneous storage plan. Rejected bindings retain
their own ordinary lowering; a compiled fallback is never borrowed from a
different tensor shape, stride, dtype or pointer-alignment specialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from .memory_ops import tensor_has_specialized_base_alignment
from .memory_ops import tensor_has_specialized_tma_alignment
from .mma_support import cute_fp32_dot_uses_tf32
from .tcgen05_flattened_prefix import FlatGroupedBlockPrefixPlan
from .tcgen05_flattened_prefix import FlatGroupedProviderPlan
from .tcgen05_flattened_prefix import FlatRowLayout
from .tcgen05_grouped_descriptors import DYNAMIC
from .tcgen05_grouped_descriptors import WRAPPED
from .tcgen05_grouped_descriptors import WrappedGroupedDescriptorPlan
from .tcgen05_grouped_descriptors import wrapped_grouped_descriptor_supported
from .tcgen05_operand_transform import Tcgen05OperandRolePlan
from .tcgen05_storage import NativeSharedAllocation
from .tcgen05_storage import NativeSharedStoragePlan
from .tcgen05_tma_rn import Tcgen05TmaRnRoles

if TYPE_CHECKING:
    import torch

    from ..compile_environment import CompileEnvironment
    from .tcgen05_flat_grouped_ir import FlatGroupedIRProof


@dataclass(frozen=True)
class FlatGroupedRnaSchedule:
    provider: FlatGroupedProviderPlan
    roles: Tcgen05OperandRolePlan | Tcgen05TmaRnRoles
    block_k: int
    ab_stages: int
    storage: NativeSharedStoragePlan
    shared_capacity: int
    reserved_shared: int = 1024
    descriptors: WrappedGroupedDescriptorPlan | None = None
    tma_rn: bool = False
    resident_ctas: int = 1
    block_prefix: FlatGroupedBlockPrefixPlan | None = None

    def __post_init__(self) -> None:
        if (
            self.descriptors is not None
            and self.descriptors
            != WrappedGroupedDescriptorPlan(
                self.provider.groups,
                self.rows,
                self.reduction,
                self.columns,
                self.provider.offset_bits,
            )
        ):
            raise ValueError("descriptor policy disagrees with typed provider")
        if (
            type(self.tma_rn) is not bool
            or type(self.roles)
            is not (Tcgen05TmaRnRoles if self.tma_rn else Tcgen05OperandRolePlan)
            or self.roles.scheduler_self_consumer
            or (self.provider.tile_m, self.provider.tile_n) != (128, 128)
            or type(self.block_k) is not int
            or type(self.ab_stages) is not int
            or not _pipeline_supported(
                self.tma_rn, self.block_k, self.ab_stages, self.resident_ctas
            )
            or self.reduction % self.block_k
            or self.columns % 4
            or self.rows <= 0
            or self.reserved_shared != 1024
            or type(self.shared_capacity) is not int
            or not 1024 < self.shared_capacity <= 232448
            or self.storage
            != flat_grouped_storage(
                self.provider.groups,
                self.block_k,
                self.ab_stages,
                tma_rn=self.tma_rn,
                block_prefix=self.block_prefix,
            )
            or (
                self.block_prefix is not None
                and (
                    not self.tma_rn
                    or self.block_prefix.groups != self.provider.groups
                    or self.block_prefix.warps * 32 != self.roles.threads_per_cta
                )
            )
            or not self.storage.fits(
                capacity=self.shared_capacity, reserved=self.reserved_shared
            )
            or self.resident_ctas * (self.storage.bytes + self.reserved_shared)
            > self.shared_capacity
        ):
            raise ValueError("unsupported complete grouped RNA schedule")

    @property
    def rows(self) -> int:
        return self.provider.max_rows

    @property
    def columns(self) -> int:
        return self.provider.output.width

    @property
    def reduction(self) -> int:
        return self.provider.a.width

    @property
    def thread_block(self) -> tuple[int, int, int]:
        # This emitter retains the native template's exact x/y ABI. The role
        # proof is linear-warp based and charges the same full physical block.
        return 32, self.roles.threads_per_cta // 32, 1

    def cache_identity(self) -> tuple[object, ...]:
        return (
            "flat_grouped_tma_rn_schedule_v1"
            if self.tma_rn
            else "flat_grouped_rna_schedule_v1",
            self.provider.cache_identity(),
            self.roles.cache_identity(),
            self.thread_block,
            self.block_k,
            self.ab_stages,
            ("accumulator_slots", 2),
            ("output_stages", 2),
            ("output_subtile", (128, 32)),
            ("tma_descriptor", "tf32_rn")
            if self.tma_rn
            else ("raw_descriptor", "float32"),
            ("mma_operand", "tf32_tma_rn" if self.tma_rn else "tf32_rna"),
            ("accumulator_and_output", "float32"),
            ("bias", "one_fp32_add_after_complete_k"),
            ("orientation", "nm"),
            self.storage.cache_identity,
            self.reserved_shared,
            self.shared_capacity,
            *((self.descriptors.cache_identity(),) if self.descriptors else ()),
            *(
                ((("resident_ctas_per_sm", self.resident_ctas),))
                if self.resident_ctas != 1
                else ()
            ),
            *((self.block_prefix.cache_identity(),) if self.block_prefix else ()),
        )


def _pipeline_supported(
    tma_rn: bool, block_k: int, stages: int, resident_ctas: int
) -> bool:
    return (
        type(resident_ctas) is int
        and resident_ctas in (1, 2)
        and (
            (block_k, stages) in ((32, 6), (64, 3))
            or (tma_rn and (block_k, stages) == (32, 2))
        )
        and (resident_ctas == 1 or (tma_rn and (block_k, stages) == (32, 2)))
    )


def flat_grouped_storage(
    groups: int,
    block_k: int,
    ab_stages: int,
    *,
    tma_rn: bool = False,
    block_prefix: FlatGroupedBlockPrefixPlan | None = None,
) -> NativeSharedStoragePlan:
    """The complete kernel's thirteen disjoint simultaneously-live objects."""
    storage = NativeSharedStoragePlan.pack(
        (
            NativeSharedAllocation("tcgen05_work_tile_smem_ptr", 9 * 4, 16),
            NativeSharedAllocation("tcgen05_tmem_holding_buf", 4, 4),
            NativeSharedAllocation("tcgen05_tmem_dealloc_mbar_ptr", 8, 8),
            NativeSharedAllocation("tcgen05_acc_pipeline_barriers", 2 * 2 * 8, 8),
            NativeSharedAllocation("tcgen05_sched_pipeline_mbars", 2 * 8, 8),
            NativeSharedAllocation("smem_a", ab_stages * 128 * block_k * 4, 128),
            NativeSharedAllocation("smem_b", ab_stages * 128 * block_k * 4, 128),
            NativeSharedAllocation("tcgen05_grouped_tensormap_smem_ptr", 256, 128),
            NativeSharedAllocation("tcgen05_ab_pipeline_mbars", ab_stages * 2 * 8, 8),
            NativeSharedAllocation("tcgen05_grouped_d_tensormap_smem_ptr", 128, 128),
            NativeSharedAllocation("tcgen05_sD_ptr", 2 * 128 * 32 * 4, 1024),
            NativeSharedAllocation("tcgen05_prefix_ptr", groups * 4, 16),
            *(
                (
                    NativeSharedAllocation(
                        "tcgen05_converted_full_ptr", ab_stages * 8, 8
                    ),
                )
                if not tma_rn
                else ()
            ),
        )
    )
    if block_prefix is not None:
        if not tma_rn or block_prefix.groups != groups:
            raise ValueError("block prefix storage requires the matching TMA RN plan")
        storage = storage.append(
            NativeSharedAllocation(
                "tcgen05_prefix_chunk_summaries_ptr", block_prefix.summary_bytes, 16
            )
        )
    return storage


def _static_metadata(tensor: torch.Tensor) -> bool:
    return all(type(value) is int for value in (*tensor.shape, *tensor.stride()))


def flat_grouped_rna_schedule(
    env: CompileEnvironment,
    proof: FlatGroupedIRProof,
    *,
    converter_warps: int,
    block_k: int,
    ab_stages: int,
    shared_capacity: int,
    descriptor_policy: str = DYNAMIC,
    tma_rn: bool = False,
    resident_ctas: int = 1,
    prefix_scan: str = "warp",
) -> FlatGroupedRnaSchedule | None:
    """Admit static binding facts, never symbolic size hints or tensor values."""
    if descriptor_policy not in (DYNAMIC, WRAPPED):
        raise ValueError("invalid grouped descriptor policy")
    tensors = (proof.offsets, proof.a, proof.b, proof.bias, proof.output)
    if (
        type(tma_rn) is not bool
        or not env.settings.static_shapes
        or not cute_fp32_dot_uses_tf32()
        or any(tensor.device != env.device for tensor in tensors)
        or not all(_static_metadata(tensor) for tensor in tensors)
        or type(converter_warps) is not int
        or converter_warps not in ((0,) if tma_rn else (1, 2, 4, 8))
        or type(block_k) is not int
        or type(ab_stages) is not int
        or not _pipeline_supported(tma_rn, block_k, ab_stages, resident_ctas)
        or type(shared_capacity) is not int
        or not 1024 < shared_capacity <= 232448
        or prefix_scan not in ("warp", "block")
    ):
        return None
    m, k = proof.a.shape
    g, b_k, n = proof.b.shape
    limit = (1 << 31) - 1
    if (
        not (0 < m < limit and 0 < g < limit and 0 < k < limit and 0 < n < limit)
        or b_k != k
        or k % block_k
        or n % 4
        or tuple(proof.a.stride()) != (k, 1)
        or tuple(proof.b.stride()) != (k * n, n, 1)
        or tuple(proof.bias.shape) != (g, n)
        or tuple(proof.bias.stride()) != (n, 1)
        or tuple(proof.offsets.shape) != (g + 1,)
        or tuple(proof.offsets.stride()) != (1,)
        or tuple(proof.output.shape) != (m * n,)
        or tuple(proof.output.stride()) != (1,)
        or max(m * k, m * n, g * k * n) > limit
        or max(m, n) > limit - 128 + 1
        or g * ((m + 127) // 128) * ((n + 127) // 128) > limit
        or 4 * g > shared_capacity
        or not all(
            tensor_has_specialized_tma_alignment(env, tensor)
            for tensor in (proof.a, proof.b)
        )
        or not tensor_has_specialized_base_alignment(env, proof.bias, 16)
    ):
        return None
    roles = (
        Tcgen05TmaRnRoles()
        if tma_rn
        else Tcgen05OperandRolePlan(converter_warps, scheduler_self_consumer=False)
    )
    block_prefix = None
    if prefix_scan == "block":
        warps = roles.threads_per_cta // 32
        if not tma_rn or roles.threads_per_cta % 32 or not 0 < g <= 32 * warps <= 1024:
            return None
        block_prefix = FlatGroupedBlockPrefixPlan(g, warps)
    storage = flat_grouped_storage(
        g, block_k, ab_stages, tma_rn=tma_rn, block_prefix=block_prefix
    )
    if (
        not storage.fits(capacity=shared_capacity, reserved=1024)
        or resident_ctas * (storage.bytes + 1024) > shared_capacity
    ):
        return None
    provider = FlatGroupedProviderPlan(
        g,
        FlatRowLayout(m * k, k, k, 16),
        FlatRowLayout(m * n, n, n, 16),
        128,
        128,
        cast("Literal[32, 64]", proof.offset_bits),
    )
    return FlatGroupedRnaSchedule(
        provider,
        roles,
        block_k,
        ab_stages,
        storage,
        shared_capacity,
        descriptors=(
            WrappedGroupedDescriptorPlan(g, m, k, n, proof.offset_bits)
            if descriptor_policy == WRAPPED
            and wrapped_grouped_descriptor_supported(g, m, k, n, proof.offset_bits)
            else None
        ),
        tma_rn=tma_rn,
        resident_ctas=resident_ctas,
        block_prefix=block_prefix,
    )
