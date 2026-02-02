# KDA (Key-Differential Attention) Kernels - Standalone implementations
# Note: The full chunk_kda and fused_recurrent_kda require many additional fla dependencies.
# For standalone use, the following are available:
from .naive import naive_recurrent_kda, naive_chunk_kda
from .gate import (
    naive_kda_gate,
    naive_kda_lowerbound_gate,
    fused_kda_gate,
    kda_gate_fwd,
    kda_gate_bwd,
    kda_gate_chunk_cumsum,
)

__all__ = [
    # Naive/reference implementations (standalone)
    "naive_recurrent_kda",
    "naive_chunk_kda",
    # Gate functions (standalone)
    "naive_kda_gate",
    "naive_kda_lowerbound_gate",
    "fused_kda_gate",
    "kda_gate_fwd",
    "kda_gate_bwd",
    "kda_gate_chunk_cumsum",
]
