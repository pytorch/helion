# GQA Attention Kernels - Standalone package
#
# Non-paged (dense) kernels:
from .triton_prefill_attention import context_attention_fwd
from .triton_decode_attention import (
    triton_decode_attention_fwd,
    triton_grouped_decode_attention_fwd,
)

# Paged attention - CSR style (from SGLang)
# Uses kv_indptr/kv_indices for compact paged KV cache indexing
from .paged_decode_csr import decode_attention_fwd as paged_decode_csr_fwd
from .paged_prefill_csr import context_attention_fwd as paged_prefill_csr_fwd
from .extend_attention_csr import (
    extend_attention_fwd as extend_attention_csr_fwd,
    extend_attention_fwd_unified as extend_attention_csr_unified_fwd,
)

# Paged attention - Block Table style (from vLLM)
# Uses 2D block_table [num_seqs, max_blocks] for O(1) lookup
from .paged_decode_blocktable import unified_attention_blocktable as paged_decode_blocktable_fwd
from .paged_prefill_blocktable import chunked_prefill_paged_decode_blocktable

__all__ = [
    # Non-paged (dense)
    "context_attention_fwd",
    "triton_decode_attention_fwd",
    "triton_grouped_decode_attention_fwd",
    # Paged - CSR style
    "paged_decode_csr_fwd",
    "paged_prefill_csr_fwd",
    "extend_attention_csr_fwd",
    "extend_attention_csr_unified_fwd",
    # Paged - Block Table style
    "paged_decode_blocktable_fwd",
    "chunked_prefill_paged_decode_blocktable",
]
