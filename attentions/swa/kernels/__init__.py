# SWA (Sliding Window Attention) Kernels - Standalone package

# Non-paged (dense) kernels
from .flash_attention import FlashAttention
from .forward import _attn_fwd
from .backward import _attn_bwd_preprocess, _attn_bwd_dk_dv, _attn_bwd_dq

# Paged kernels with sliding window support
from .paged_swa_decode import (
    paged_swa_decode_fwd,
    paged_swa_decode_blocktable_fwd,
)
from .paged_swa_prefill import (
    paged_swa_prefill_fwd,
    paged_swa_extend_fwd,
)

__all__ = [
    # Non-paged (dense)
    "FlashAttention",
    "_attn_fwd",
    "_attn_bwd_preprocess",
    "_attn_bwd_dk_dv",
    "_attn_bwd_dq",
    # Paged - CSR style
    "paged_swa_decode_fwd",
    "paged_swa_prefill_fwd",
    "paged_swa_extend_fwd",
    # Paged - Block Table style
    "paged_swa_decode_blocktable_fwd",
]
