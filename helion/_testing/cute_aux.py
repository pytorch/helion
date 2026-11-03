"""Shared kernels and CPU code generation for tcgen05 AUX regressions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import torch

from . import patch_cute_mma_support
from . import patch_target_facts
from .cute_codegen import mock_cuda_unavailable
import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Generator


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def batched_aux(
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
def rank_two_aux(
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


# The role-local tcgen05 GEMM configuration the AUX tests start from; a
# sparse base, independent of the host.
AUX_CONFIG_BASE: dict[str, Any] = {
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


def aux_config(**overrides: object) -> helion.Config:
    """:data:`AUX_CONFIG_BASE` with ``overrides`` applied."""
    values: dict[str, Any] = dict(AUX_CONFIG_BASE)
    values.update(overrides)
    return helion.Config(**values)


def aux_inputs(
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


def rank_two_inputs() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((128, 128), (128, 512), (128, 512))
    )


@contextmanager
def cpu_codegen() -> Generator[None, None, None]:
    """Generate tcgen05 AUX source on CPU for a 148-SM sm103 target.

    These tests inspect generated source without compiling or launching CuTe.
    ``mock_cuda_unavailable`` supplies the CPU-only device probes; the target
    facts below pin what the schedule heuristics read.  Only the compile
    environment sees the GB300 capability (the bind key keeps the host's); a
    two-sided model is a separate change.
    """
    with (
        patch_cute_mma_support(),
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch_target_facts(num_sm=148, smem_capacity_bytes=232448),
    ):
        yield


def rank_two_code(args: tuple[torch.Tensor, ...], config: helion.Config) -> str:
    with cpu_codegen():
        return rank_two_aux._bind_isolated(args).to_code(config)


@contextmanager
def as_sm100() -> Generator[None, None, None]:
    """CPU codegen with the flash gates forced on (no CUDA, no CuTe compile)."""
    with (
        cpu_codegen(),
        patch(
            "helion._compiler.cute.analysis.attention_match._attention_flash_supported",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.analysis.attention_match._attention_flash_gate_enabled",
            return_value=True,
        ),
    ):
        yield
