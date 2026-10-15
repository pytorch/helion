"""SM100+ 16/32-byte noncoherent loads with explicit L2 evict-first policy.

The compiler proves whole-kernel read-only storage and aliases before it
selects the native helper. The candidate preserves ordinary load semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass.cute as cute

from .l2_policy import _load_vector_with_policy

if TYPE_CHECKING:
    from ._mlir_compat import ir

_ASM_NC_L2_FIRST_32B = (
    "ld.global.nc.L2::evict_first.v8.b32 {$0,$1,$2,$3,$4,$5,$6,$7}, [$8];"
)
_ASM_NC_L2_FIRST_16B = (
    "{ .reg .b64 policy; "
    "createpolicy.fractional.L2::evict_first.b64 policy, 1.0; "
    "ld.global.nc.L2::cache_hint.v4.b32 {$0,$1,$2,$3}, [$4], policy; }"
)


def load_nc_l2_first_candidate(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Ordinary load fallback until the late compiler proof resolves it."""
    return cute.arch.load(pointer, vector_type)


def load_nc_l2_evict_first_32b(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Load an initialized, naturally aligned 32-byte read-only packet.

    The caller proves SM100+/CUDA13+, complete packet safety, and no writes
    through the source or an alias anywhere in the kernel. Reinterpretation
    from raw UInt32 words preserves every source bit and requested dtype.
    """
    return _load_vector_with_policy(pointer, vector_type, _ASM_NC_L2_FIRST_32B, words=8)


def load_nc_l2_evict_first_16b(pointer: object, vector_type: ir.VectorType) -> ir.Value:
    """Read-only 16-byte packet using the legal descriptor-based form.

    The static L2::evict_first shortcut is restricted to 32 bytes. This
    constructor plus cache_hint instruction preserves the same policy for
    a naturally aligned 16-byte packet under the compiler's alias proof.
    """
    return _load_vector_with_policy(pointer, vector_type, _ASM_NC_L2_FIRST_16B, words=4)
