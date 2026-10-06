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

_CHAIN_CONFIGS = {(('torch.float32', (99, 1, 32000), (32000, 32000, 1)), ('torch.int32', (99, 1), (1, 1)), ('torch.float32', (99, 2, 32000), (64000, 32000, 1)), ('torch.int64', (1,), (1,)), None, None): {'block_sizes': [4,
                                                                                                                                                                                                           64,
                                                                                                                                                                                                           4],
                                                                                                                                                                                           'cute_cluster_n': 8,
                                                                                                                                                                                           'cute_fragment_producer_cache': True,
                                                                                                                                                                                           'cute_fragment_pure_producer_regions': True,
                                                                                                                                                                                           'cute_fragment_reduction': 'warp',
                                                                                                                                                                                           'cute_fragment_threads': 512,
                                                                                                                                                                                           'cute_fragment_warp_scan': True,
                                                                                                                                                                                           'cute_independent_reduction': False,
                                                                                                                                                                                           'cute_lane_layouts': ['blocked',
                                                                                                                                                                                                                 'blocked',
                                                                                                                                                                                                                 'strided',
                                                                                                                                                                                                                 'blocked',
                                                                                                                                                                                                                 'blocked'],
                                                                                                                                                                                           'cute_min_blocks_per_mp': 0,
                                                                                                                                                                                           'cute_reduction_reloads': ['register',
                                                                                                                                                                                                                      'register'],
                                                                                                                                                                                           'cute_replicated_reduction': False,
                                                                                                                                                                                           'cute_vector_packet_unroll': False,
                                                                                                                                                                                           'cute_vector_widths': [1,
                                                                                                                                                                                                                  1,
                                                                                                                                                                                                                  2,
                                                                                                                                                                                                                  4,
                                                                                                                                                                                                                  2],
                                                                                                                                                                                           'load_eviction_policies': ['',
                                                                                                                                                                                                                      'l1_l2_first',
                                                                                                                                                                                                                      '',
                                                                                                                                                                                                                      'l2_last',
                                                                                                                                                                                                                      'l2_last',
                                                                                                                                                                                                                      '',
                                                                                                                                                                                                                      'l1_l2_first',
                                                                                                                                                                                                                      'last',
                                                                                                                                                                                                                      'l1_l2_first',
                                                                                                                                                                                                                      'l2_last',
                                                                                                                                                                                                                      'l1_l2_first',
                                                                                                                                                                                                                      '',
                                                                                                                                                                                                                      'l1_l2_last',
                                                                                                                                                                                                                      '',
                                                                                                                                                                                                                      'first',
                                                                                                                                                                                                                      'first',
                                                                                                                                                                                                                      'last',
                                                                                                                                                                                                                      'l1_l2_last',
                                                                                                                                                                                                                      'l1_l2_first'],
                                                                                                                                                                                           'num_threads': [4,
                                                                                                                                                                                                           64,
                                                                                                                                                                                                           0,
                                                                                                                                                                                                           0,
                                                                                                                                                                                                           0]}}

def _normalize(args):
    if len(args) == 4:
        return (*args, None, None)
    if len(args) == 5:
        return (*args, None)
    return args

def key_chain(*args):
    key = _signature(_normalize(args))
    if key not in _CHAIN_CONFIGS:
        raise ValueError(f"No measured chain config for {key}")
    return key

def autotune_chain(*args):
    return helion.CuteStructuralConfig(_CHAIN_CONFIGS[key_chain(*args)], STRUCTURAL_POLICY)

CONFIGS = [helion.CuteStructuralConfig(config, STRUCTURAL_POLICY) for configs in (_CHAIN_CONFIGS.values(),) for config in configs]

_HELION_AOT_STRUCTURAL_MANIFEST = {"schema": "helion.aot.structural_model", "version": 1,"policy": STRUCTURAL_POLICY.to_dict(), "policy_id": STRUCTURAL_POLICY.identity(), "configs": {"chain": [helion.CuteStructuralConfig(config, STRUCTURAL_POLICY).to_dict() for config in _CHAIN_CONFIGS.values()]}}
