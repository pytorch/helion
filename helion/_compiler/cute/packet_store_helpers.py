"""L2 evict-last forms of the FP8 packet flushes (``cute_store_policy``)."""

from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import vector

from ._mlir_compat import ir
from .l2_policy import store_u32x4_l2_evict_last
from .vec_utils import _store_fp8_vec


def store_v16b_l2_evict_last(ptr: object, packed: object) -> None:
    """Store an already packed 16-byte integer packet with the L2 hint.

    ``lower_packet_store_policy`` selects these helpers only for full 16-byte
    packets on the validated target; bitcasting the carrier keeps every bit.
    """
    (raw,) = cutlass.extract_mlir_values(packed)
    packet_type = ir.VectorType(raw.type)
    assert packet_type.shape[0] * packet_type.element_type.width == 128
    words = vector.bitcast(ir.VectorType.get([4], cutlass.Uint32.mlir_type), raw)
    store_u32x4_l2_evict_last(
        ptr,
        [cutlass.Uint32(vector.extract(words, [], [index])) for index in range(4)],
    )


def store_fp8e4m3fn_vec_l2_evict_last(ptr: object, values: list) -> None:
    _store_fp8_vec(
        ptr, values, cutlass.Float8E4M3FN, packet_store=store_v16b_l2_evict_last
    )


def store_fp8e5m2_vec_l2_evict_last(ptr: object, values: list) -> None:
    _store_fp8_vec(
        ptr, values, cutlass.Float8E5M2, packet_store=store_v16b_l2_evict_last
    )
