"""Fixed batch-one Gemma 4 CuTe configuration for the 148-SM B200.

The public preset reads CONFIG directly. The benchmark runner uses the SM100
filename to select the target architecture.
"""

from __future__ import annotations

# Full-search return on the published batch-one model. Keep the exact knob
# values, including disabled alternatives, so a changed default cannot
# silently select a different program.
CONFIG = {
    "block_sizes": [1, 16, 4096, 1, 64, 64, 256],
    "num_threads": [1, 4, 32, 1, 4, 2, 4, 1, 1, 1],
    "loop_orders": [[0, 1], [0, 1], [1, 0, 2], [0, 1], [0, 1, 2], [0, 1]],
    "reduction_loops": [256],
    "load_eviction_policies": [
        "l2_last" if i in (5, 11, 14, 17) else "" for i in range(32)
    ],
    "cute_store_policy": "default",
    "cute_compiler_pipeline": "opt2",
    "cute_independent_reduction": False,
    "cute_replicated_reduction": False,
    "cute_vector_packet_unroll": False,
    "cute_vloop_sink": False,
    "cute_grid_lane_jam": False,
    "cute_scratch_forwarding": False,
    "cute_pair_lane_math": False,
    "cute_lane_unroll": 1,
    "cute_unpadded_rows": False,
    "cute_uniform_lane_compares": False,
    "cute_tail_peel": False,
    "cute_defer_tile_reduce": True,
    "cute_scalar_load_hints": True,
    "cute_pdl": False,
    "pid_type": "persistent_interleaved",
    "cross_loop_pipeline": "barrier",
    "cute_single_row_publication": True,
    "cute_cooperative_selection_merge": True,
    "cute_spare_warp_row_prefetch": True,
    "cute_ordered_column_pairs": True,
    "cute_parallel_row_sum": True,
    "cute_warp_row_reductions": True,
    "cute_unroll_warp_row_sweeps": True,
    "cute_fragment_topk_network": True,
    "cute_staged_projection_grid": 3,
    "cute_packed_row_publication": True,
    "cute_cooperative_row_norm": True,
    "cute_staged_indexed_rows": True,
    "cute_direct_indexed_packets": True,
    "cute_reuse_indexed_row_input": True,
    "cute_early_indexed_input": True,
    "cute_indexed_row_epilogue": True,
    "cute_vector_widths": [4 if i in (8, 18) else 1 for i in range(21)],
    "cute_lane_layouts": ["strided" if i in (13, 15) else "blocked" for i in range(21)],
    "cute_reduction_reloads": ["gmem", "auto", "auto"],
    "cute_warp_fp32_redux": False,
    "cute_cluster_warp_online": False,
    "cute_register_row": False,
    "cute_register_row_tensor_copy": False,
    "cute_scalar_epilogue": False,
    "cute_scalar_epilogue_keep_inputs": False,
    "cute_scalar_epilogue_owner_fold": False,
    "cute_register_tile_warp_fold": False,
    "cute_register_tile_cooperative_fold": False,
    "cute_register_tile_store_first": False,
    "cute_cluster_n": 1,
    "cute_min_blocks_per_mp": 0,
}
