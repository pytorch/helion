"""
Helion Vector Add Tuning Example with OpenEvolve
=================================================

This example demonstrates how to use the OpenEvolveTuner to automatically
find optimal kernel configurations for a simple vector addition kernel.

The tuner uses OpenEvolve's evolutionary algorithm to search through the
configuration space and find settings that maximize throughput.

Requirements:
- OpenEvolve installed: pip install openevolve
- OPENAI_API_KEY environment variable set
"""

from __future__ import annotations

import os
import sys

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE

# Check if OpenEvolve is available
try:
    from helion.autotuner.openevolve_tuner import OpenEvolveTuner
except ImportError:
    print("Error: OpenEvolve not installed. Install with: pip install openevolve")
    sys.exit(1)

# Check if OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. Using mock mode for demonstration.")
    print("For real tuning, set your API key: export OPENAI_API_KEY='your-key-here'")
    MOCK_MODE = True
else:
    MOCK_MODE = False


def create_vector_add_kernel(config):
    """
    Create a vector addition kernel with the given configuration.

    Args:
        config: Dictionary with 'block_size' and 'num_warps' keys

    Returns:
        Compiled Helion kernel function
    """

    @helion.kernel(
        config=helion.Config(
            block_sizes=[config["block_size"]], num_warps=config["num_warps"]
        )
    )
    def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two vectors element-wise."""
        n = x.size(0)
        out = torch.empty_like(x)
        for tile_n in hl.tile([n]):
            out[tile_n] = x[tile_n] + y[tile_n]
        return out

    return vector_add


def evaluate_config(config):
    """
    Evaluate a kernel configuration by measuring its throughput.

    Args:
        config: Dictionary with kernel configuration parameters

    Returns:
        Throughput in GB/s (higher is better), or 0.0 if config fails
    """
    try:
        # Create kernel with this config
        kernel = create_vector_add_kernel(config)

        # Create test inputs
        n = 1024 * 1024  # 1M elements
        x = torch.randn(n, device=DEVICE, dtype=torch.float32)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)

        # Warmup
        _ = kernel(x, y)
        torch.cuda.synchronize()

        # Benchmark
        from triton.testing import do_bench

        time_ms = do_bench(lambda: kernel(x, y), warmup=25, rep=100)

        # Calculate throughput
        # Vector add reads 2 arrays and writes 1 array
        bytes_accessed = x.numel() * x.element_size() * 3
        throughput_gbs = (bytes_accessed / (time_ms * 1e-3)) / 1e9

        return throughput_gbs

    except Exception as e:
        print(f"Config failed: {config}, Error: {e}")
        return 0.0


def mock_evaluate_config(config):
    """
    Mock evaluation for demonstration when OPENAI_API_KEY is not set.

    This simulates performance based on heuristics:
    - Larger block sizes are generally better up to a point
    - 4 warps tends to be optimal for simple kernels
    """
    block_size = config.get("block_size", 128)
    num_warps = config.get("num_warps", 4)

    # Simulate performance with a simple heuristic
    # Peak performance around block_size=256 and num_warps=4
    block_score = 1.0 - abs(block_size - 256) / 512
    warp_score = 1.0 - abs(num_warps - 4) / 8

    # Combine scores
    base_throughput = 400.0  # GB/s
    throughput = base_throughput * (0.5 + 0.5 * block_score) * (0.5 + 0.5 * warp_score)

    # Add some noise
    import random
    noise = random.gauss(0, 10)

    return max(0.0, throughput + noise)


def run_tuning_example():
    """Run the vector add tuning example."""
    print("=" * 70)
    print("Helion Vector Add Kernel Tuning with OpenEvolve")
    print("=" * 70)

    # Define the search space
    config_space = {
        "block_size": [32, 64, 128, 256, 512, 1024],
        "num_warps": [1, 2, 4, 8],
    }

    print("\nConfiguration space:")
    for param, values in config_space.items():
        print(f"  {param}: {values}")

    # Choose evaluation function
    if MOCK_MODE:
        print("\nRunning in MOCK MODE (no GPU evaluation)")
        evaluate_fn = mock_evaluate_config
        max_evals = 20  # Fewer evaluations in mock mode
    else:
        print("\nRunning in REAL MODE (GPU evaluation)")
        evaluate_fn = evaluate_config
        max_evals = 50  # More evaluations for real tuning

    # Create tuner
    tuner = OpenEvolveTuner(
        config_space=config_space,
        objective=evaluate_fn,
        max_evaluations=max_evals,
        population_size=10,
        temperature=0.8,
        verbose=True,
    )

    # Run tuning
    print(f"\nStarting tuning with max_evaluations={max_evals}...")
    print("This may take a few minutes depending on the number of evaluations.\n")

    try:
        best_config = tuner.tune()
    except Exception as e:
        print(f"\nTuning failed with error: {e}")
        print("\nFalling back to default configuration...")
        best_config = {"block_size": 128, "num_warps": 4}

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBest configuration found:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")

    # Verify the best config works
    if not MOCK_MODE:
        print(f"\nVerifying best configuration...")
        final_perf = evaluate_config(best_config)
        print(f"Final performance: {final_perf:.2f} GB/s")

        # Compare to baseline
        baseline_config = {"block_size": 128, "num_warps": 4}
        baseline_perf = evaluate_config(baseline_config)
        print(f"Baseline performance (block_size=128, num_warps=4): {baseline_perf:.2f} GB/s")

        if final_perf > baseline_perf:
            improvement = ((final_perf / baseline_perf) - 1) * 100
            print(f"\nImprovement: {improvement:.1f}% faster than baseline!")
        else:
            print(f"\nNote: Baseline was already near-optimal for this kernel.")
    else:
        print("\n(Skipping verification in mock mode)")

    print("\n" + "=" * 70)
    print("Tuning complete!")
    print("=" * 70)


def run_simple_test():
    """Run a simple test without OpenEvolve to verify kernel works."""
    print("\nRunning simple vector add test (no tuning)...")

    config = {"block_size": 128, "num_warps": 4}
    kernel = create_vector_add_kernel(config)

    # Small test
    x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
    y = torch.randn(1024, device=DEVICE, dtype=torch.float32)

    result = kernel(x, y)
    expected = x + y

    # Check correctness
    if torch.allclose(result, expected, rtol=1e-5, atol=1e-5):
        print("✓ Vector add kernel is working correctly!")
        return True
    else:
        print("✗ Vector add kernel output is incorrect!")
        print(f"  Max error: {(result - expected).abs().max().item()}")
        return False


if __name__ == "__main__":
    # First verify the kernel works
    if not run_simple_test():
        print("\nKernel verification failed. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 70)

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python helion_vector_add_tuning.py [--simple|--help]")
            print("\nOptions:")
            print("  --simple    Run simple test only (no tuning)")
            print("  --help      Show this help message")
            print("\nEnvironment variables:")
            print("  OPENAI_API_KEY    Required for real tuning with OpenEvolve")
            sys.exit(0)
        elif sys.argv[1] == "--simple":
            print("Simple test completed successfully!")
            sys.exit(0)

    # Run the full tuning example
    try:
        run_tuning_example()
    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
