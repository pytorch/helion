# Kernelkit utilities - local implementation
# Provides utility functions that replace the kernelkit external package

import time
import math
from typing import Optional, List, Dict, Any
import torch


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


class Counter:
    """Simple counter for generating unique IDs."""
    def __init__(self, start: int = 0):
        self._value = start

    def next(self) -> int:
        val = self._value
        self._value += 1
        return val


# Terminal colors
colors = {
    'RED_BG': '\033[41m',
    'GREEN_BG': '\033[42m',
    'CYAN_BG': '\033[46m',
    'CLEAR': '\033[0m',
}


def check_is_allclose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-5,
    cos_diff_tol: Optional[float] = None,
) -> bool:
    """
    Check if two tensors are close within tolerance.

    Args:
        name: Name of the tensor being checked
        actual: Actual tensor
        expected: Expected tensor
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance
        cos_diff_tol: Cosine difference tolerance (optional)

    Returns:
        True if tensors are close, False otherwise
    """
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()

    # Handle NaN values
    actual_nan_mask = torch.isnan(actual_flat)
    expected_nan_mask = torch.isnan(expected_flat)

    # Remove NaN values for comparison
    if actual_nan_mask.any() or expected_nan_mask.any():
        # Check that NaN patterns match
        if not torch.equal(actual_nan_mask, expected_nan_mask):
            print(f"[{name}] NaN pattern mismatch")
            return False
        # Compare only non-NaN values
        actual_flat = actual_flat[~actual_nan_mask]
        expected_flat = expected_flat[~expected_nan_mask]

    if actual_flat.numel() == 0:
        return True

    # Compute absolute difference
    abs_diff = (actual_flat - expected_flat).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Compute relative difference
    rel_diff = abs_diff / (expected_flat.abs() + 1e-10)
    max_rel_diff = rel_diff.max().item()

    # Check tolerance
    within_abs_tol = max_abs_diff <= abs_tol
    within_rel_tol = max_rel_diff <= rel_tol
    is_close = within_abs_tol or within_rel_tol

    # Optional: cosine similarity check
    if cos_diff_tol is not None and actual_flat.numel() > 0:
        cos_sim = torch.nn.functional.cosine_similarity(
            actual_flat.unsqueeze(0), expected_flat.unsqueeze(0)
        ).item()
        cos_diff = 1.0 - cos_sim
        within_cos_tol = cos_diff <= cos_diff_tol
        is_close = is_close or within_cos_tol

    if not is_close:
        print(f"[{name}] FAILED: max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}")
    else:
        print(f"[{name}] PASSED: max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}")

    return is_close


class BenchResult:
    """Result from benchmark runs."""

    def __init__(self, times: List[float], kernel_times: Dict[str, float] = None):
        self._times = times
        self._kernel_times = kernel_times or {}
        self._kernel_names = list(self._kernel_times.keys())

    def get_kernel_names(self) -> List[str]:
        """Get names of profiled kernels."""
        return self._kernel_names

    def get_kernel_time(self, name: str) -> float:
        """Get time for a specific kernel (in seconds)."""
        for k, v in self._kernel_times.items():
            if name in k:
                return v
        return sum(self._times) / len(self._times) if self._times else 0.0

    def get_e2e_time(self, start_kernel: str, end_kernel: str) -> float:
        """Get end-to-end time between two kernels."""
        # Simplified: just return total time
        return sum(self._times) / len(self._times) if self._times else 0.0

    def mean_time(self) -> float:
        """Get mean execution time."""
        return sum(self._times) / len(self._times) if self._times else 0.0


def is_using_profiling_tools() -> bool:
    """Check if profiling tools are active."""
    return False


def bench_kineto(fn, num_tests: int = 10, warmup: int = 3) -> BenchResult:
    """
    Benchmark a function using PyTorch's profiler.

    Args:
        fn: Function to benchmark
        num_tests: Number of test iterations
        warmup: Number of warmup iterations

    Returns:
        BenchResult with timing information
    """
    # Warmup
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    # Benchmark
    times = []
    kernel_times: Dict[str, float] = {}

    try:
        # Try to use PyTorch profiler for detailed kernel timing
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(num_tests):
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        # Extract kernel times from profiler
        for event in prof.key_averages():
            if event.key and event.cuda_time_total > 0:
                kernel_times[event.key] = event.cuda_time_total / 1e6 / num_tests  # Convert to seconds

    except Exception:
        # Fallback: simple timing without profiler
        for _ in range(num_tests):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return BenchResult(times, kernel_times)
