from __future__ import annotations

import pytest
import torch

from helion.runtime import _torch_dtype_to_cutlass

cutlass = pytest.importorskip("cutlass")


def test_fp16_bf16_fp32_unchanged() -> None:
    assert _torch_dtype_to_cutlass(torch.float16) is cutlass.Float16
    assert _torch_dtype_to_cutlass(torch.bfloat16) is cutlass.BFloat16
    assert _torch_dtype_to_cutlass(torch.float32) is cutlass.Float32


def test_fp8_e4m3fn_maps_to_cutlass() -> None:
    assert _torch_dtype_to_cutlass(torch.float8_e4m3fn) is cutlass.Float8E4M3FN


def test_fp8_e5m2_maps_to_cutlass() -> None:
    assert _torch_dtype_to_cutlass(torch.float8_e5m2) is cutlass.Float8E5M2
