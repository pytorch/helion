"""Shared kernels and CPU code generation for TCgen05 AUX regressions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import torch

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import patch_cute_mma_support
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_aux(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batches, m, k = lhs.shape
    n = rhs.shape[2]
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        out[bi, mi, ni] = (acc * scale[bi, mi, ni] + bias[bi, mi, ni]).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rank_two_aux(
    lhs: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = lhs.shape
    n = rhs.shape[1]
    out = torch.empty_like(bias)
    for mi, ni in hl.tile([m, n]):
        acc = hl.zeros([mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.addmm(acc, lhs[mi, ki], rhs[ki, ni])
        out[mi, ni] = (acc + bias[mi, ni]).to(out.dtype)
    return out


def _config(**overrides: Any) -> helion.Config:
    values: dict[str, Any] = {
        "block_sizes": [64, 256, 64],
        "num_warps": 8,
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
        "tcgen05_warp_spec_c_input_warps": 1,
        "tcgen05_aux_load_mode": "tma",
    }
    return helion.Config(**(values | overrides))


def _inputs(
    *,
    batches: int = 3,
    m: int = 128,
    n: int = 512,
    k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((batches, m, k), (batches, k, n), *((batches, m, n),) * 3)
    )


def _rank_two_inputs() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((128, 128), (128, 512), (128, 512))
    )


@contextmanager
def _cpu_codegen() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _rank_two_code(args: tuple[torch.Tensor, ...], config: helion.Config) -> str:
    with _cpu_codegen():
        return _rank_two_aux._bind_isolated(args).to_code(config)
