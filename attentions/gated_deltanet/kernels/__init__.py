# GatedDeltaNet (Gated Delta Rule) Kernels - Standalone implementations
from .naive import naive_recurrent_gated_delta_rule
from .fused_recurrent import fused_recurrent_gated_delta_rule

__all__ = [
    # Naive/reference implementations (standalone)
    "naive_recurrent_gated_delta_rule",
    # Fused recurrent Triton implementation (standalone)
    "fused_recurrent_gated_delta_rule",
]
