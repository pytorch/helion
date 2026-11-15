"""SM100+ 32/16-byte noncoherent loads with an L2 evict-first policy.

``memory_ops`` emits these for a full, naturally aligned packet of a source
that no store or atomic in the kernel can alias, on SM100+ with CUDA 13+.
The predicated packets of a masked row strip (``masked_packet_load``) take
the same noncoherent evict-first load at an ``nc_l2_first`` site, at every
width from 4 to 64 bytes, as they carry every L2 policy family, at 32 and 64
bytes too without ``cute_wide_packet_hints`` (the unpredicated 32-byte
packet stays plain there); a scalar load keeps its ordinary form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .l2_policy import _load_vector_with_policy
from .l2_policy import _policy_assembly

if TYPE_CHECKING:
    from ._mlir_compat import ir

_ASM_NC_L2_FIRST_32B = (
    "ld.global.nc.L2::evict_first.v8.b32 {$0,$1,$2,$3,$4,$5,$6,$7}, [$8];"
)
# The static ``L2::evict_first`` qualifier is only legal on 32-byte loads, so
# the 16-byte form builds the same policy and passes it as a cache hint.
_ASM_NC_L2_FIRST_16B = _policy_assembly(4, "nc_l2_first")


def load_nc_l2_evict_first_32b(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Load a 32-byte read-only packet, reinterpreted as ``vector_type``."""
    return _load_vector_with_policy(pointer, vector_type, _ASM_NC_L2_FIRST_32B, words=8)


def load_nc_l2_evict_first_16b(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Load a 16-byte read-only packet, reinterpreted as ``vector_type``."""
    return _load_vector_with_policy(pointer, vector_type, _ASM_NC_L2_FIRST_16B, words=4)
