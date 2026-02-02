# Mamba SSM Kernels - Standalone implementations
# State Space Model kernels from the Mamba project

from .mamba_utils import (
    softplus,
    autotune_configs,
    use_deterministic_mode,
    alloc_tile_workspace,
    finalize_tile_workspace,
    custom_fwd,
    custom_bwd,
)
from .ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
from .ssd_state_passing import (
    _state_passing_fwd,
    _state_passing_bwd,
    state_passing,
    state_passing_ref,
)
from .selective_state_update import (
    selective_state_update,
    selective_state_update_ref,
)
from .k_activations import _swiglu_fwd, _swiglu_bwd, swiglu
from .layernorm_gated import (
    rmsnorm_fn,
    layernorm_fn,
    _layer_norm_fwd,
    _layer_norm_bwd,
    RMSNorm,
    LayerNorm,
)

__all__ = [
    # Utilities
    'softplus',
    'autotune_configs',
    'use_deterministic_mode',
    'alloc_tile_workspace',
    'finalize_tile_workspace',
    'custom_fwd',
    'custom_bwd',
    # SSD BMM
    '_bmm_chunk_fwd',
    '_bmm_chunk_bwd',
    # State passing
    '_state_passing_fwd',
    '_state_passing_bwd',
    'state_passing',
    'state_passing_ref',
    # Selective state update
    'selective_state_update',
    'selective_state_update_ref',
    # Activations
    '_swiglu_fwd',
    '_swiglu_bwd',
    'swiglu',
    # LayerNorm
    'rmsnorm_fn',
    'layernorm_fn',
    '_layer_norm_fwd',
    '_layer_norm_bwd',
    'RMSNorm',
    'LayerNorm',
]
