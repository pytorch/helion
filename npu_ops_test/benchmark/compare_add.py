"""
Performance Comparison: Triton Ascend vs PyTorch torch.add on NPU
================================================================

This script compares the performance of Triton-Ascend kernel implementation
against PyTorch's native torch.add using triton.testing.do_bench_npu for
accurate NPU benchmarking.
"""

import torch
import torch_npu
import triton

# Import the Triton implementation
from triton_add import triton_add_2d


def benchmark_functions(fns_dict, warmup=10, active=30):
    """
    Benchmark multiple functions using triton.testing.do_bench_npu.

    Args:
        fns_dict: Dictionary of {name: function} to benchmark
        warmup: Number of warmup iterations (default: 5 in do_bench_npu)
        active: Number of benchmark iterations (default: 30 in do_bench_npu)

    Returns:
        Dictionary of {name: timing_result}
    """
    from triton.testing import do_bench_npu

    result_dict = {}
    for name, fn in fns_dict.items():
        try:
            # Call do_bench_npu with a single function
            # It will automatically convert to list: if not isinstance(funcs, list): funcs = [funcs]
            result = do_bench_npu(
                fn,
                warmup=warmup,
                active=active,
            )

            # Handle different return types from do_bench_npu
            # It may return a list (when passed multiple functions) or a single value
            if isinstance(result, list):
                # Extract single value from list
                if len(result) > 0:
                    result = result[0]
                else:
                    result = 0.0

            # Store the result
            result_dict[name] = result
            print(f"{name:30s}: {result:.4f} ms")

        except Exception as e:
            print(f"{name:30s}: Error - {e}")
            result_dict[name] = None

    return result_dict


def run_comparison(M=1024, N=1024, dtype=torch.bfloat16, warmup=10, rep=100):
    """
    Run performance comparison between Triton and PyTorch.

    Args:
        M: First dimension size
        N: Second dimension size
        dtype: Data type for tensors
        warmup: Number of warmup iterations
        rep: Number of benchmark iterations
    """
    device = "npu"

    print("=" * 80)
    print("Performance Comparison: Triton Ascend vs PyTorch torch.add")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Tensor size: ({M}, {N})")
    print(f"  Data type: {dtype}")
    print(f"  Device: {device}")
    print(f"  Warmup iterations: {warmup}")
    print(f"  Benchmark iterations: {rep}")
    print()

    # Create test data
    x = torch.randn(M, N, device=device, dtype=dtype)
    y = torch.randn(M, N, device=device, dtype=dtype)

    # Verify correctness first
    print("=" * 80)
    print("Correctness Verification")
    print("=" * 80)

    output_triton = triton_add_2d(x, y)
    output_torch = torch.add(x, y)

    max_error = torch.max(torch.abs(output_triton - output_torch))
    is_correct = torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)

    print(f"Maximum absolute error: {max_error:.6e}")
    print(f"Correctness check: {'✓ PASSED' if is_correct else '✗ FAILED'}")
    print()

    # Prepare benchmark functions
    print("=" * 80)
    print("Performance Benchmarking")
    print("=" * 80)

    # Warm up NPU
    torch_npu.npu.synchronize()
    _ = torch.add(x, y)
    torch_npu.npu.synchronize()

    # Define functions to benchmark
    # Note: We use closures to capture x and y
    fns_dict = {
        "PyTorch torch.add": lambda: torch.add(x, y),
        "Triton (block=128x128)": lambda: triton_add_2d(x, y, block_m=128, block_n=128),
        "Triton (block=64x64)": lambda: triton_add_2d(x, y, block_m=64, block_n=64),
        "Triton (block=256x64)": lambda: triton_add_2d(x, y, block_m=256, block_n=64),
        "Triton (block=64x256)": lambda: triton_add_2d(x, y, block_m=64, block_n=256),
    }

    # Benchmark
    results = benchmark_functions(fns_dict, warmup=warmup, active=rep)

    # Calculate speedup
    print()
    print("=" * 80)
    print("Performance Summary")
    print("=" * 80)

    torch_time = results.get("PyTorch torch.add")
    if torch_time is None:
        print("Error: PyTorch torch.add benchmark failed!")
        return

    print(f"\nBaseline (PyTorch torch.add): {torch_time:.4f} ms\n")
    print(f"{'Implementation':<30s} {'Time (ms)':<12s} {'Speedup':<12s} {'Status'}")
    print("-" * 80)

    for name, time_ms in results.items():
        if time_ms is None:
            print(f"{name:<30s} {'N/A':<12s} {'N/A':<12s} {'ERROR'}")
        else:
            speedup = torch_time / time_ms
            status = "✓" if speedup >= 0.95 else "✗"
            if speedup > 1:
                print(f"{name:<30s} {time_ms:<12.4f} {speedup:<12.2f}x {status} (FASTER)")
            else:
                print(f"{name:<30s} {time_ms:<12.4f} {speedup:<12.2f}x {status} (SLOWER)")

    print()

    # Find best Triton configuration
    triton_results = [(name, time_ms) for name, time_ms in results.items()
                      if name.startswith("Triton") and time_ms is not None]
    if triton_results:
        best_name, best_time = min(triton_results, key=lambda x: x[1])
        best_speedup = torch_time / best_time
        print(f"Best Triton configuration: {best_name}")
        print(f"  Time: {best_time:.4f} ms")
        print(f"  Speedup vs PyTorch: {best_speedup:.2f}x")
        print()

    # Compute theoretical throughput
    num_elements = M * N
    if torch_time > 0:
        torch_throughput = (num_elements / torch_time) / 1e6  # Mega elements per second
        print(f"PyTorch throughput: {torch_throughput:.2f} M elements/s")

    if triton_results and best_time is not None and best_time > 0:
        triton_throughput = (num_elements / best_time) / 1e6
        print(f"Triton throughput: {triton_throughput:.2f} M elements/s")
        print(f"Throughput ratio: {triton_throughput / torch_throughput:.2f}x")


def main():
    """
    Main entry point for performance comparison.
    """
    print("\n" + "=" * 80)
    print("NPU Performance Comparison: Triton Ascend vs PyTorch")
    print("=" * 80)
    print()

    # Run comparison with different sizes
    test_cases = [
        (1024, 1024),
        (2048, 2048),
        (512, 2048),
    ]

    for i, (M, N) in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}/{len(test_cases)}: ({M}, {N})")
        print(f"{'=' * 80}")
        run_comparison(M=M, N=N, dtype=torch.bfloat16, warmup=10, rep=100)
        print()


if __name__ == "__main__":
    main()
