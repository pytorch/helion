from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from examples.aot_example import row_softmax
import torch

from test._cute_binding import _mock_cuda_unavailable

from .cute_population_contracts import _target
import helion


@contextmanager
def _cpu_target():
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bound(
    *, dtype=torch.bfloat16, static=False, shape=(5, 513), layout="dense", effort="none"
):
    if layout == "transpose":
        x = torch.empty(tuple(reversed(shape)), dtype=dtype).t()
    elif layout == "slice":
        x = torch.empty((shape[0], shape[1] * 2), dtype=dtype)[:, ::2]
    elif layout == "unaligned":
        x = torch.empty(shape[0] * shape[1] + 1, dtype=dtype).as_strided(
            shape, (shape[1], 1), 1
        )
    else:
        x = torch.empty(shape, dtype=dtype)
    kernel = helion.kernel(
        row_softmax.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort=effort,
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        cute_materialize_transformed_operands=True,
    )
    return kernel._bind_isolated((x,))


def _config(*, width=4096, threads=128, rows=1, vector=8):
    return helion.Config(
        block_sizes=[rows, width],
        num_threads=[rows if rows > 1 else 0, threads],
        cute_vector_widths=[1, vector],
        cute_lane_layouts=["blocked", "strided"],
    )
