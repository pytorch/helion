"""SM100+ 32/16-byte noncoherent loads with an L2 evict-first policy.

``memory_ops`` emits these only for a full, naturally aligned packet of a
source that no store or atomic in the kernel can alias, on SM100+ with
CUDA 13+; every other ``nc_l2_first`` site keeps its ordinary load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .l2_policy import _load_vector_with_policy

if TYPE_CHECKING:
    from ._mlir_compat import ir

_ASM_NC_L2_FIRST_32B = (
    "ld.global.nc.L2::evict_first.v8.b32 {$0,$1,$2,$3,$4,$5,$6,$7}, [$8];"
)
# The static ``L2::evict_first`` qualifier is only legal on 32-byte loads, so
# the 16-byte form builds the same policy and passes it as a cache hint.
_ASM_NC_L2_FIRST_16B = (
    "{ .reg .b64 policy; "
    "createpolicy.fractional.L2::evict_first.b64 policy, 1.0; "
    "ld.global.nc.L2::cache_hint.v4.b32 {$0,$1,$2,$3}, [$4], policy; }"
)


def load_nc_l2_evict_first_32b(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Load a 32-byte read-only packet, reinterpreted as ``vector_type``."""
    return _load_vector_with_policy(pointer, vector_type, _ASM_NC_L2_FIRST_32B, words=8)


def load_nc_l2_evict_first_16b(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Load a 16-byte read-only packet, reinterpreted as ``vector_type``."""
    return _load_vector_with_policy(pointer, vector_type, _ASM_NC_L2_FIRST_16B, words=4)
