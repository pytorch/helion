# RWKV6 Kernels - Standalone implementations
# Note: chunk_rwkv6 has complex fla dependencies, so it's not exported here.
# For the full chunk implementation, use the fla package directly.
from .fused_recurrent import fused_recurrent_rwkv6

__all__ = [
    'fused_recurrent_rwkv6',
]
