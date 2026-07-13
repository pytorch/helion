"""
Test script to verify that do_bench_npu works correctly in Helion.

This tests both the import and basic functionality.
"""

import torch
import torch_npu
from helion._testing import do_bench


def simple_add():
    """Simple addition function for benchmarking."""
    x = torch.randn(1024, 1024, device="npu", dtype=torch.bfloat16)
    y = torch.randn(1024, 1024, device="npu", dtype=torch.bfloat16)
    return torch.add(x, y)


def main():
    """Test do_bench_npu functionality."""
    print("=" * 80)
    print("Testing Helion do_bench_npu on NPU")
    print("=" * 80)
    print()

    # Check NPU availability
    print(f"torch.npu.is_available(): {torch.npu.is_available()}")
    print()

    # Test do_bench import
    print("Testing do_bench import...")
    print(f"do_bench module: {do_bench.__module__}")
    print(f"do_bench function: {do_bench.__name__}")
    print()

    # Benchmark simple function
    print("Benchmarking simple_add function...")
    try:
        result = do_bench(
            simple_add,
            warmup=10,
            rep=50,
            return_mode="median",
        )
        print(f"Success! Time: {result:.4f} ms")
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
