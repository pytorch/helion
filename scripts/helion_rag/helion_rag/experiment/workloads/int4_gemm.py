"""Corpus workload: bfloat16 x int4 packed GEMM."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE

    from examples.int4_gemm import _pack_int4_matrix
    from examples.int4_gemm import matmul_bf16_int4
    from examples.int4_gemm import reference_matmul_bf16_int4

    m, k, n = 512, 1024, 512
    a = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
    b_unpacked = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=DEVICE)
    b_packed = _pack_int4_matrix(b_unpacked)
    return matmul_bf16_int4, reference_matmul_bf16_int4, (a, b_packed)


register("int4_gemm-512x1024x512", Workload("matmul_bf16_int4", 2e-1, 1.0, _build))
