"""Additional codegen-only coverage for pretuned Triton and CuTe kernels.

This module is collected explicitly by ``compare_pretuned_codegen.py``. It is
kept outside ``test/`` so the comparatively expensive B200 coverage does not
become part of the normal test suite.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Protocol
from unittest import mock

import torch

if TYPE_CHECKING:
    from types import ModuleType

    from helion.runtime.kernel import BoundKernel


class _BindableKernel(Protocol):
    def bind(self, args: tuple[object, ...]) -> BoundKernel[object]: ...


def _load_pretuned_module(name: str) -> ModuleType:
    path = Path.cwd() / "pretuned_kernels" / name / f"{name}.py"
    module_name = f"_helion_codegen_compare_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        raise RuntimeError("pretuned CuTe codegen comparison requires an SM100 GPU")


def _lower_pretuned(kernel: _BindableKernel, args: tuple[object, ...]) -> None:
    bound = kernel.bind(args)
    bound.autotune(args, force=False)


def test_nvfp4_gemv_triton() -> None:
    _require_b200()
    module = _load_pretuned_module("nvfp4_gemv")
    n, k = 4096, 4096

    weight, x, weight_scale, x_scale = module._make_fp4in_inputs(n, k)
    weight_bytes = weight.view(torch.uint8)
    fp4_args = (
        weight_bytes,
        x.view(torch.uint8),
        weight_scale.reshape(-1).view(torch.int8),
        x_scale.reshape(-1).view(torch.int8),
        torch.empty(n, dtype=torch.bfloat16, device="cuda"),
        1.0,
    )
    _lower_pretuned(module.nvfp4_gemv_fp4in_kernel, fp4_args)

    weight, x, weight_scale = module._make_bf16in_inputs(n, k)
    weight_bytes = weight.view(torch.uint8)
    bf16_args = (
        weight_bytes,
        x.view(weight_bytes.shape[1] // 8, 16),
        weight_scale.reshape(-1).view(torch.int8),
        torch.empty(n, dtype=torch.bfloat16, device="cuda"),
        1.0,
    )
    _lower_pretuned(module.nvfp4_gemv_bf16in_kernel, bf16_args)


def test_scale_mm_cute() -> None:
    _require_b200()
    module = _load_pretuned_module("scale_mm_cute")
    cases = (
        (module.scale_mm_cute_skinny_m, (1, 4096, 256), False),
        (module.scale_mm_cute_swap_ab, (16, 4096, 256), True),
        (module.scale_mm_cute, (64, 2048, 2048), False),
    )
    for kernel, shape, swap_scale in cases:
        x, y, scale_a, scale_b = module._make_inputs(*shape)
        args = (x, y, scale_a[:, 0] if swap_scale else scale_a, scale_b)
        _lower_pretuned(kernel, args)


def test_nvfp4_gemv_cute() -> None:
    _require_b200()
    module = _load_pretuned_module("nvfp4_gemv_cute")
    for n, k, rows in ((4096, 4096, 2), (10240, 8192, 4)):
        weight, x, weight_scale, x_scale = module._make_fp4in_inputs(n, k)
        weight_bytes = weight.view(torch.uint8)
        fp4_args = (
            weight_bytes,
            x.view(torch.uint8),
            weight_scale.reshape(-1),
            x_scale.reshape(-1),
            torch.empty(n, dtype=torch.bfloat16, device="cuda"),
            1.0,
        )
        _lower_pretuned(
            getattr(module, f"nvfp4_gemv_fp4in_rows{rows}_kernel"), fp4_args
        )

        weight, x, weight_scale = module._make_bf16in_inputs(n, k)
        weight_bytes = weight.view(torch.uint8)
        bf16_args = (
            weight_bytes,
            x.view(weight_bytes.shape[1] // 8, 16),
            weight_scale.reshape(-1),
            torch.empty(n, dtype=torch.bfloat16, device="cuda"),
            1.0,
        )
        _lower_pretuned(
            getattr(module, f"nvfp4_gemv_bf16in_rows{rows}_kernel"), bf16_args
        )


def test_cute_fragment_epilogues() -> None:
    _require_b200()
    for name in ("projection_rotary", "interleaved_swiglu"):
        module = _load_pretuned_module(name)
        _lower_pretuned(getattr(module, name), module._make_inputs(*module.SHAPES[0]))


def test_cute_grouped_gemm() -> None:
    _require_b200()
    with mock.patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}):
        module = _load_pretuned_module("grouped_gemm")
        for case in module.CASES[:2]:
            device = torch.device("cuda", torch.cuda.current_device())
            group_a, group_b, expected = module._COMPARE.make_inputs(
                case.problems, device
            )
            outputs = module._COMPARE.make_outputs(group_a, group_b)
            module._prepare_helion(case, group_a, group_b, outputs, expected)()
