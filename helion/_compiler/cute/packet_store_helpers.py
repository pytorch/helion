"""L2 policy adapters for integer packets after their values are packed."""

from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import vector
import cutlass.cute as cute
from cutlass.cutlass_dsl import get_current_env_manager

from ._mlir_compat import ir
from .l2_policy import fixed_l2_evict_last_store_policy_supported
from .l2_policy import store_u32x4_l2_evict_last
from .vec_utils import _store_fp8_vec
from .vec_utils import store_u8_vec


def _store_policy_supported() -> bool:
    manager = get_current_env_manager()
    if manager is None:
        return False
    architecture = cutlass.Arch.from_string(manager.arch)
    return fixed_l2_evict_last_store_policy_supported(
        (architecture.major, architecture.minor)
    )


def store_v16b_l2_evict_last(ptr: object, packed: object) -> None:
    """Store an existing aligned 16-byte integer packet with the fixed hint.

    Compiler packet-store proofs supply alignment, global address space and
    full coverage. Scalar/narrow values and unsupported targets use the
    existing store. Bitcasting the already-packed carrier preserves all bits.
    """
    (raw,) = cutlass.extract_mlir_values(packed)
    packet_type = raw.type
    if (
        not isinstance(packet_type, ir.VectorType)
        or packet_type.scalable
        or not isinstance(packet_type.element_type, ir.IntegerType)
        or len(packet_type.shape) != 1
        or packet_type.shape[0] * packet_type.element_type.width != 128
        or not _store_policy_supported()
    ):
        cute.arch.store(ptr, packed)
        return
    words = vector.bitcast(ir.VectorType.get([4], cutlass.Uint32.mlir_type), raw)
    store_u32x4_l2_evict_last(
        ptr,
        [cutlass.Uint32(vector.extract(words, [], [index])) for index in range(4)],
    )


def store_u8_vec_l2_evict_last(ptr: object, values: list) -> None:
    store_u8_vec(ptr, values, packet_store=store_v16b_l2_evict_last)


def store_fp8e4m3fn_vec_l2_evict_last(ptr: object, values: list) -> None:
    _store_fp8_vec(
        ptr, values, cutlass.Float8E4M3FN, packet_store=store_v16b_l2_evict_last
    )


def store_fp8e5m2_vec_l2_evict_last(ptr: object, values: list) -> None:
    _store_fp8_vec(
        ptr, values, cutlass.Float8E5M2, packet_store=store_v16b_l2_evict_last
    )
