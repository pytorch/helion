"""Pretuned B200 heuristic for BF16 residual statistics."""


def key_residual(*args) -> int:
    return 0


def autotune_residual(*args) -> dict:
    return {
        "block_sizes": [1],
        "cute_vector_widths": [4],
        "pid_type": "flat",
        "reduction_loops": [2048],
        "cute_packed_bf16x2_reduction": True,
        "cute_packed_bf16x2_threads_per_row": 128,
        "cute_packed_bf16x2_warp0_epilogue": True,
    }
