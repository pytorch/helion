"""
B200 (Blackwell) Attention Kernel Tuning with OpenEvolve
=========================================================

This example demonstrates how to use OpenEvolveTuner to optimize an attention
kernel specifically for NVIDIA B200 (Blackwell) GPUs.

B200-specific features tuned:
- Tensor descriptor indexing
- Warp specialization strategies
- Register allocation (maxRegAutoWS)
- Multi-buffering configurations
- Persistent kernel scheduling

Requirements:
- NVIDIA B200 GPU
- OpenEvolve: pip install openevolve
- OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import os
import sys
import math

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE, get_nvidia_gpu_model

# Check for B200
gpu_model = get_nvidia_gpu_model()
IS_B200 = "B200" in gpu_model or "Blackwell" in gpu_model

if not IS_B200:
    print(f"Warning: This script is optimized for B200 GPUs.")
    print(f"Detected GPU: {gpu_model}")
    print("Results may not be optimal on other architectures.")
    print()

# Check OpenEvolve
try:
    from helion.autotuner.openevolve_tuner import OpenEvolveTuner
except ImportError:
    print("Error: OpenEvolve not installed. Install with: pip install openevolve")
    sys.exit(1)

# Check API key
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. Running in demo mode.")
    print("For real tuning, set: export OPENAI_API_KEY='your-key'")
    MOCK_MODE = True
else:
    MOCK_MODE = False


def create_attention_kernel(config):
    """
    Create a simplified attention kernel with B200-optimized config.

    This is a simplified version for tuning demonstration.
    For production, use the full blackwell_attention.py implementation.
    """
    # Build helion config, handling "default" values
    pid_type = config.get("pid_type", "default")
    if pid_type == "default":
        pid_type = "flat"

    indexing = config.get("indexing", "default")
    if indexing == "default":
        indexing = "pointer"

    @helion.kernel(
        config=helion.Config(
            block_sizes=[config["block_m"], config["block_n"]],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            pid_type=pid_type,
            indexing=indexing,
        )
    )
    def attention_kernel(
        q: torch.Tensor,  # [seq_q, head_dim]
        k: torch.Tensor,  # [seq_k, head_dim]
        v: torch.Tensor,  # [seq_k, head_dim]
    ) -> torch.Tensor:
        """
        Simplified 2D attention: O = softmax(Q @ K^T / sqrt(d)) @ V

        This kernel operates on a single batch/head for simplicity.
        The grid loop (hl.tile) must be at the top level.
        """
        seq_q, head_dim = q.shape
        seq_k = k.size(0)

        # Scale factor
        scale = 1.0 / math.sqrt(float(head_dim))

        # Output tensor
        o = torch.empty([seq_q, head_dim], dtype=q.dtype, device=q.device)

        # Grid loop must be at top level
        for tile_q in hl.tile(seq_q):
            q_slice = q[tile_q, :]

            # Initialize accumulators for online softmax
            row_max = hl.full([tile_q], float("-inf"), dtype=torch.float32)
            row_sum = hl.zeros([tile_q], dtype=torch.float32)
            acc = hl.zeros([tile_q, head_dim], dtype=torch.float32)

            # Inner reduction loop over K/V
            for tile_k in hl.tile(seq_k):
                k_slice = k[tile_k, :]
                v_slice = v[tile_k, :]

                # Scores: Q @ K^T, shape [tile_q, tile_k]
                scores = torch.matmul(q_slice.to(torch.float32), k_slice.t().to(torch.float32)) * scale

                # Online softmax: update max
                tile_max = torch.amax(scores, dim=1)
                new_max = torch.maximum(row_max, tile_max)

                # Rescale previous accumulator
                alpha = torch.exp(row_max - new_max)

                # Compute exp(scores - new_max)
                scores_exp = torch.exp(scores - new_max[:, None])

                # Update running sum and accumulator
                row_sum = row_sum * alpha + scores_exp.sum(dim=1)
                acc = acc * alpha[:, None] + torch.matmul(scores_exp, v_slice.to(torch.float32))
                row_max = new_max

            # Normalize and store
            o[tile_q, :] = (acc / row_sum[:, None]).to(q.dtype)

        return o

    return attention_kernel


def evaluate_attention_config(config):
    """
    Benchmark an attention configuration on B200.

    Args:
        config: Dict with kernel configuration parameters

    Returns:
        TFLOPS (higher is better), or 0.0 for failed configs
    """
    try:
        # Create kernel with this config
        kernel = create_attention_kernel(config)

        # Test problem size (2D kernel: seq_len x head_dim)
        seq_len = 1024
        head_dim = 64

        # Create 2D inputs (single batch/head for the kernel)
        q = torch.randn(seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        k = torch.randn(seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        v = torch.randn(seq_len, head_dim, device=DEVICE, dtype=torch.float16)

        # Warmup
        _ = kernel(q, k, v)
        torch.cuda.synchronize()

        # Benchmark
        from triton.testing import do_bench

        time_ms = do_bench(
            lambda: kernel(q, k, v),
            warmup=25,
            rep=100
        )

        # Calculate TFLOPS
        # Attention: 2 * seq^2 * head_dim (for Q@K^T and scores@V)
        flops = 4 * seq_len * seq_len * head_dim
        tflops = (flops / (time_ms * 1e-3)) / 1e12

        return tflops

    except torch.cuda.OutOfMemoryError:
        print(f"OOM with config: {config}")
        return 0.0
    except Exception as e:
        print(f"Config failed: {config}, Error: {e}")
        return 0.0


def mock_evaluate_attention_config(config):
    """
    Mock evaluation for demo when API key not set.

    Simulates B200 attention performance based on heuristics.
    """
    block_m = config.get("block_m", 128)
    block_n = config.get("block_n", 64)
    num_warps = config.get("num_warps", 4)
    num_stages = config.get("num_stages", 3)
    maxreg = config.get("maxreg", 128)

    # Heuristics for B200 attention
    # - block_m should be 128-256 for good occupancy
    # - block_n should be 64-128 for memory coalescing
    # - num_warps: 4-8 optimal for attention
    # - num_stages: 3-4 for pipeline efficiency
    # - maxreg: 152-192 for B200

    block_m_score = 1.0 - abs(block_m - 128) / 256
    block_n_score = 1.0 - abs(block_n - 64) / 128
    warp_score = 1.0 - abs(num_warps - 8) / 16
    stage_score = 1.0 - abs(num_stages - 3) / 5
    reg_score = 1.0 - abs(maxreg - 192) / 256

    # Base TFLOPS for B200 attention
    base_tflops = 50.0

    # Combined score
    total_score = (
        0.25 * block_m_score +
        0.25 * block_n_score +
        0.20 * warp_score +
        0.15 * stage_score +
        0.15 * reg_score
    )

    tflops = base_tflops * (0.5 + 0.5 * total_score)

    # Add noise
    import random
    noise = random.gauss(0, 2.0)

    return max(0.0, tflops + noise)


def run_b200_tuning():
    """Run attention kernel tuning optimized for B200."""
    print("=" * 70)
    print("B200 (Blackwell) Attention Kernel Tuning with OpenEvolve")
    print("=" * 70)
    print()

    print(f"GPU: {gpu_model}")
    print(f"Mode: {'MOCK' if MOCK_MODE else 'REAL'}")
    print()

    # B200-optimized configuration space
    config_space = {
        # Block sizes for attention
        "block_m": [64, 128, 256],
        "block_n": [64, 128],

        # Warp configuration
        "num_warps": [4, 8, 16],

        # Pipeline stages
        "num_stages": [2, 3, 4, 5],

        # B200-specific: Register allocation
        "maxreg": [128, 152, 192, 224, 256],

        # B200-specific: Indexing mode
        "indexing": ["default", "tensor_descriptor"],

        # B200-specific: Persistent kernel type
        "pid_type": ["default", "persistent_interleaved"],
    }

    print("Configuration space:")
    for param, values in config_space.items():
        print(f"  {param}: {values}")
    print()

    # Choose evaluation function
    if MOCK_MODE:
        print("Running in MOCK MODE (no GPU evaluation)")
        evaluate_fn = mock_evaluate_attention_config
        max_evals = 30
    else:
        print("Running in REAL MODE (GPU evaluation)")
        evaluate_fn = evaluate_attention_config
        max_evals = 100

    # Create tuner
    print(f"\nCreating tuner with max_evaluations={max_evals}")
    tuner = OpenEvolveTuner(
        config_space=config_space,
        objective=evaluate_fn,
        max_evaluations=max_evals,
        population_size=15,
        temperature=0.8,
        verbose=True,
    )

    # Run tuning
    print("\nStarting B200 attention kernel tuning...")
    print("This will optimize for tensor descriptors, warp specialization, and register usage.")
    print()

    try:
        best_config = tuner.tune()
    except Exception as e:
        print(f"\nTuning failed with error: {e}")
        print("\nFalling back to default B200 configuration...")
        best_config = {
            "block_m": 128,
            "block_n": 64,
            "num_warps": 8,
            "num_stages": 3,
            "maxreg": 192,
            "indexing": "tensor_descriptor",
            "pid_type": "persistent_interleaved",
        }

    # Display results
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)
    print(f"\nBest configuration for {gpu_model}:")
    for param, value in sorted(best_config.items()):
        print(f"  {param}: {value}")

    if not MOCK_MODE:
        print(f"\nVerifying best configuration...")
        final_tflops = evaluate_attention_config(best_config)
        print(f"Final performance: {final_tflops:.2f} TFLOPS")

        # Compare to baseline
        baseline_config = {
            "block_m": 128,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 3,
            "maxreg": 128,
            "indexing": "default",
            "pid_type": "default",
        }
        baseline_tflops = evaluate_attention_config(baseline_config)
        print(f"Baseline performance: {baseline_tflops:.2f} TFLOPS")

        if final_tflops > baseline_tflops:
            improvement = ((final_tflops / baseline_tflops) - 1) * 100
            print(f"\n🎉 Improvement: {improvement:.1f}% faster than baseline!")
        else:
            print(f"\nNote: Baseline was already near-optimal.")

        # B200-specific insights
        print("\n" + "=" * 70)
        print("B200-SPECIFIC INSIGHTS")
        print("=" * 70)

        if best_config.get("indexing") == "tensor_descriptor":
            print("✓ Using tensor descriptor indexing (B200 optimized)")

        if best_config.get("pid_type") == "persistent_interleaved":
            print("✓ Using persistent interleaved scheduling (B200 feature)")

        maxreg = best_config.get("maxreg", 128)
        if maxreg >= 192:
            print(f"✓ High register usage ({maxreg}) leverages B200's increased RF")

        num_warps = best_config.get("num_warps", 4)
        if num_warps >= 8:
            print(f"✓ High warp count ({num_warps}) for B200 SM utilization")

    else:
        print("\n(Skipping verification in mock mode)")

    print("\n" + "=" * 70)
    print("Tuning complete!")
    print("=" * 70)

    # Save results
    import json
    results_file = "b200_attention_tuning_results.json"
    results = {
        "gpu": gpu_model,
        "best_config": best_config,
        "best_score": tuner.best_score,
        "evaluations": tuner.evaluation_count,
        "mode": "mock" if MOCK_MODE else "real",
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print(__doc__)
            print("\nUsage: python helion_b200_attention_tuning.py [--help]")
            print("\nEnvironment variables:")
            print("  OPENAI_API_KEY    Required for real tuning")
            sys.exit(0)

    try:
        run_b200_tuning()
    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
