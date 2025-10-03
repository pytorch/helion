"""
Element-wise Addition Example
===========================

This example demonstrates how to implement an element-wise addition kernel using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import datetime
import getpass
import inspect
import os

import torch

import helion
from helion._testing import run_example
import helion.language as hl
from helion.autotuner import LocalAutotuneCache
from helion.autotuner import search_algorithms

# %%
# Addition Kernel
# --------------
def _one_generation_autotuner(bound_kernel, args, **kwargs):
    autotuner_name = os.environ.get("HELION_AUTOTUNER", "PatternSearch")
    autotuner_cls = search_algorithms.get(autotuner_name)
    if autotuner_cls is None:
        raise ValueError(
            f"Unknown HELION_AUTOTUNER value: {autotuner_name}, valid options are: "
            f"{', '.join(search_algorithms.keys())}"
        )

    tuner_kwargs = dict(kwargs)
    init_params = inspect.signature(autotuner_cls.__init__).parameters
    if "max_generations" in init_params:
        tuner_kwargs.setdefault("max_generations", 1)
    if "num_generations" in init_params:
        tuner_kwargs.setdefault("num_generations", 1)
    if "initial_population" in init_params:
        tuner_kwargs.setdefault("initial_population", 10)

    autotuner = autotuner_cls(bound_kernel, args, **tuner_kwargs)
    return LocalAutotuneCache(autotuner)


REBENCH_THRESHOLD = 1.0005


@helion.kernel(
    autotuner_fn=_one_generation_autotuner,
    autotune_rebenchmark_threshold=REBENCH_THRESHOLD,
)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise with broadcasting support.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        A new tensor containing the element-wise sum of x and y
    """
    # match pytorch broadcasting rules
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        # match type promotion of torch.add
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    # tile will be a tuple of blocks
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the add kernel implementation against PyTorch's native add function.

    Args:
        m: First dimension of the test tensors
        n: Second dimension of the test tensors
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    y = torch.randn([m, n], device="cuda", dtype=torch.float16)
    run_example(add, torch.add, (x, y))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the add kernel verification with 1024x1024 tensors.
    """
    check(1024, 1024)


if __name__ == "__main__":
    main()
