"""
Deterministic vs Non-Deterministic Reductions
==============================================

This example demonstrates the difference between deterministic and non-deterministic reduction patterns.
Deterministic reductions guarantee bit-exact reproducibility across runs, while non-deterministic reductions
using atomics may produce slightly different results due to floating-point operation ordering.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
# Deterministic Row Sum
# -------------------
@helion.kernel()
def row_sum_deterministic(x: torch.Tensor) -> torch.Tensor:
    """
    Deterministic row-wise sum using tree reduction.

    Each row is processed by a single block, which performs a tree-based reduction
    in a fixed order. This guarantees the same result across multiple runs.

    Args:
        x: Input tensor of shape [m, n]

    Returns:
        Output tensor of shape [m] containing the sum of each row
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(dim=1)

    return out


# %%
# Non-Deterministic Row Sum
# ------------------------
@helion.kernel()
def row_sum_nondeterministic(x: torch.Tensor) -> torch.Tensor:
    """
    Non-deterministic row-wise sum using atomic operations.

    Each row is split across multiple blocks along the column dimension, with
    atomic_add used to accumulate partial sums. The order of atomic operations
    varies between runs, causing different floating-point rounding and thus
    slightly different results.

    Args:
        x: Input tensor of shape [m, n]

    Returns:
        Output tensor of shape [m] containing the sum of each row
    """
    m, n = x.shape
    out = torch.zeros([m], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        hl.atomic_add(out, [tile_m], x[tile_m, tile_n].sum(dim=1))

    return out


# %%
# Reproducibility Check
# -------------------
def check_reproducibility(kernel: object, x: torch.Tensor, num_runs: int = 10) -> None:
    """
    Check if a kernel produces deterministic results across multiple runs.

    Args:
        kernel: Kernel function to test
        x: Input tensor
        num_runs: Number of times to run the kernel
    """
    results = [kernel(x) for _ in range(num_runs)]
    all_equal = all(torch.equal(results[0], r) for r in results[1:])

    if all_equal:
        print(f"✓ {kernel.name}: DETERMINISTIC")
    else:
        variance = torch.stack(results).var().item()
        print(f"✗ {kernel.name}: NON-DETERMINISTIC (variance={variance:.2e})")


# %%
# Testing Function
# -------------
def test(m: int, n: int, dtype: torch.dtype = torch.float32) -> None:
    """
    Test the reduction kernels and compare their reproducibility and performance.

    Args:
        m: Number of rows
        n: Number of columns
        dtype: Data type for the tensors
    """
    x = torch.randn([m, n], device=DEVICE, dtype=dtype)

    print("Reproducibility check:")
    check_reproducibility(row_sum_deterministic, x)
    check_reproducibility(row_sum_nondeterministic, x)

    print("\nPerformance:")
    run_example(
        row_sum_deterministic,
        {
            "nondeterministic": row_sum_nondeterministic,
            "pytorch": lambda x: x.sum(dim=1),
        },
        (x,),
    )


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the reduction kernel test.
    Tests with 1000 rows and 2048 columns using float32.
    """
    test(1000, 2048)


if __name__ == "__main__":
    main()
