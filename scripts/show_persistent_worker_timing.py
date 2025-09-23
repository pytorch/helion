#!/usr/bin/env python3
"""Run a minimal autotune run and print persistent-worker timing stats."""

from __future__ import annotations

import importlib
import logging
import os

import torch

from test.test_autotuner import trigger_cuda_unrecoverable_error

import helion.autotuner.base_search as base_search_module
import helion.autotuner.finite_search as finite_search_module


def main() -> None:
    os.environ.setdefault("HELION_AUTOTUNE_PERSISTENT_WORKER", "1")
    os.environ.setdefault("HELION_AUTOTUNE_TIMING", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to demonstrate the persistent worker timing output")

    base_search = importlib.reload(base_search_module)
    finite_search = importlib.reload(finite_search_module)

    bound_kernel, args, _, good_config = trigger_cuda_unrecoverable_error()
    bound_kernel.settings.autotune_log_level = logging.DEBUG

    search = finite_search.FiniteSearch(
        bound_kernel,
        args,
        configs=[good_config] * 100,
    )

    try:
        best = search.autotune()
        print(f"\nBest config selected: {best}")
    finally:
        search._close_benchmark_worker()
        torch.cuda.synchronize()

    print(
        "\nTiming summary is emitted on stderr; rerun with "
        "'python scripts/show_persistent_worker_timing.py 2>&1 | tee timing.log'"
        " if you want to capture it."
    )


if __name__ == "__main__":
    main()
