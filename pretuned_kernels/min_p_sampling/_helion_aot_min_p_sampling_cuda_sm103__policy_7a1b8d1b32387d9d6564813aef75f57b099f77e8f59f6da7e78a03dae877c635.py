"""Measured SM103 configs, keyed by exact public kernel signatures. No runtime I/O."""
from __future__ import annotations
import helion
import torch

STRUCTURAL_POLICY = helion.CuteStructuralPolicy(cute_region_fission=True, cute_full_slice_matmul_tiling=True, cute_segmented_matmul_tiling=True, cute_flatten_nested_reductions=True, cute_materialize_transformed_operands=True)

def _signature(args):
    def part(value):
        if isinstance(value, torch.Tensor):
            return str(value.dtype), tuple(value.shape), tuple(value.stride())
        if isinstance(value, torch.dtype):
            return str(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"Unsupported argument: {type(value).__name__}")
    return tuple(part(value) for value in args)

_MIN_P_CONFIGS = {(('torch.float32', (2056192,), (1,)), ('torch.int64', (1,), (1,)), 0.05, 128512, 'torch.int32'): {'block_sizes': [1,
                                                                                                                   1024,
                                                                                                                   1],
                                                                                                   'cute_cluster_n': 16,
                                                                                                   'cute_fragment_packet_loads': True,
                                                                                                   'cute_fragment_published_scalars': True,
                                                                                                   'cute_fragment_pure_producer_regions': True,
                                                                                                   'cute_fragment_reduction': 'warp',
                                                                                                   'cute_fragment_scan': 'cooperative',
                                                                                                   'cute_fragment_threads': 256,
                                                                                                   'cute_fragment_warp_scan': True,
                                                                                                   'cute_independent_reduction': False,
                                                                                                   'cute_lane_layouts': ['blocked',
                                                                                                                         'blocked',
                                                                                                                         'blocked',
                                                                                                                         'blocked'],
                                                                                                   'cute_min_blocks_per_mp': 0,
                                                                                                   'cute_reduction_reloads': ['auto'],
                                                                                                   'cute_replicated_reduction': False,
                                                                                                   'cute_vector_packet_unroll': False,
                                                                                                   'cute_vector_widths': [1,
                                                                                                                          1,
                                                                                                                          1,
                                                                                                                          1],
                                                                                                   'load_eviction_policies': ['',
                                                                                                                              'last',
                                                                                                                              '',
                                                                                                                              'first',
                                                                                                                              'first',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              'last',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              'last',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              '',
                                                                                                                              ''],
                                                                                                   'loop_orders': [[0,
                                                                                                                    1]],
                                                                                                   'num_threads': [1,
                                                                                                                   128,
                                                                                                                   0,
                                                                                                                   0]}}

def key_min_p(*args):
    key = _signature(args)
    if key not in _MIN_P_CONFIGS:
        raise ValueError(f"No measured min_p config for {key}")
    return key

def autotune_min_p(*args):
    return helion.CuteStructuralConfig(_MIN_P_CONFIGS[key_min_p(*args)], STRUCTURAL_POLICY)

CONFIGS = [helion.CuteStructuralConfig(config, STRUCTURAL_POLICY) for configs in (_MIN_P_CONFIGS.values(),) for config in configs]

_HELION_AOT_STRUCTURAL_MANIFEST = {"schema": "helion.aot.structural_model", "version": 1,"policy": STRUCTURAL_POLICY.to_dict(), "policy_id": STRUCTURAL_POLICY.identity(), "configs": {"min_p": [helion.CuteStructuralConfig(config, STRUCTURAL_POLICY).to_dict() for config in _MIN_P_CONFIGS.values()]}}
