"""Kernel/shape registry for autotuner study campaigns.

Each entry names an example kernel and builds concrete arguments for one
shape. Shapes are chosen to be memory- or compute-bound (not overhead-bound)
so autotuning quality differences are measurable.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Callable

import torch


def identity_epilogue(
    acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Picklable stand-in for the examples' default epilogue lambda (the
    benchmark worker torch.save()s kernel args, and lambdas don't pickle)."""
    return acc


@dataclasses.dataclass(frozen=True)
class KernelCase:
    name: str  # unique "<kernel>-<shape>" identifier
    module: str  # examples module name (no .py)
    attr: str  # kernel attribute inside the module
    make_args: Callable[[torch.device], tuple[Any, ...]]


def _matmul_4096(device: torch.device) -> tuple[Any, ...]:
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    y = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    return (x, y, identity_epilogue)


def _matmul_wide(device: torch.device) -> tuple[Any, ...]:
    x = torch.randn(8192, 1024, device=device, dtype=torch.float16)
    y = torch.randn(1024, 8192, device=device, dtype=torch.float16)
    return (x, y, identity_epilogue)


def _attention_2k(device: torch.device) -> tuple[Any, ...]:
    q, k, v = (
        torch.randn(4, 32, 2048, 64, device=device, dtype=torch.float16)
        for _ in range(3)
    )
    return (q, k, v)


def _attention_4k128(device: torch.device) -> tuple[Any, ...]:
    q, k, v = (
        torch.randn(2, 16, 4096, 128, device=device, dtype=torch.float16)
        for _ in range(3)
    )
    return (q, k, v)


def _softmax_two_pass(device: torch.device) -> tuple[Any, ...]:
    return (torch.randn(8192, 8192, device=device, dtype=torch.float32),)


def _rms_norm(device: torch.device) -> tuple[Any, ...]:
    x = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    weight = torch.randn(8192, device=device, dtype=torch.float16)
    return (x, weight)


def _layer_norm(device: torch.device) -> tuple[Any, ...]:
    x = torch.randn(4096, 10240, device=device, dtype=torch.float16)
    weight = torch.randn(10240, device=device, dtype=torch.float16)
    bias = torch.randn(10240, device=device, dtype=torch.float16)
    return (x, [10240], weight, bias)


def _cross_entropy(device: torch.device) -> tuple[Any, ...]:
    n, v = 16384, 131072
    logits = torch.randn(n, v, device=device, dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device=device, dtype=torch.int64)
    return (logits, labels)


def _bmm(device: torch.device) -> tuple[Any, ...]:
    a = torch.randn(16, 512, 768, device=device, dtype=torch.float16)
    b = torch.randn(16, 768, 1024, device=device, dtype=torch.float16)
    return (a, b)


def _matmul_split_k(device: torch.device) -> tuple[Any, ...]:
    x = torch.randn(64, 32768, device=device, dtype=torch.float16)
    y = torch.randn(32768, 64, device=device, dtype=torch.float16)
    return (x, y, identity_epilogue)


def _longsum(device: torch.device) -> tuple[Any, ...]:
    return (torch.randn(256, 262144, device=device, dtype=torch.float32),)


def _fp8_gemm(device: torch.device) -> tuple[Any, ...]:
    x = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    y = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    x_fp8 = x.to(torch.float8_e4m3fn)
    y_fp8 = y.T.contiguous().T.to(torch.float8_e4m3fn)
    return (x_fp8, y_fp8)


def _gather_gemv(device: torch.device) -> tuple[Any, ...]:
    b, s, n = 8, 8192, 4
    w = torch.randn(b, s, s, device=device, dtype=torch.float16)
    idx = torch.randint(0, b, (n,), device=device, dtype=torch.int32)
    x = torch.randn(s, device=device, dtype=torch.float16)
    return (w, idx, x)


def _welford(device: torch.device) -> tuple[Any, ...]:
    s, d = 262144, 1024
    weight = torch.rand(d, device=device, dtype=torch.float32)
    bias = torch.rand(d, device=device, dtype=torch.float32)
    x = torch.randn(s, d, device=device, dtype=torch.float32)
    return (weight, bias, x)


KERNEL_CASES: dict[str, KernelCase] = {
    case.name: case
    for case in [
        KernelCase("matmul-4096", "matmul", "matmul", _matmul_4096),
        KernelCase("matmul-wide", "matmul", "matmul", _matmul_wide),
        KernelCase("attention-2k64", "attention", "attention", _attention_2k),
        KernelCase("attention-4k128", "attention", "attention", _attention_4k128),
        KernelCase("softmax2p-8192", "softmax", "softmax_two_pass", _softmax_two_pass),
        KernelCase("rmsnorm-8192", "rms_norm", "rms_norm_fwd", _rms_norm),
        KernelCase("layernorm-10240", "layer_norm", "layer_norm_fwd", _layer_norm),
        KernelCase(
            "crossentropy-131k", "cross_entropy", "cross_entropy", _cross_entropy
        ),
        KernelCase("bmm-16x512", "bmm", "bmm", _bmm),
        KernelCase("splitk-32768", "matmul_split_k", "matmul_split_k", _matmul_split_k),
        KernelCase("longsum-256x262k", "long_sum", "longsum", _longsum),
        KernelCase("fp8gemm-4096", "fp8_gemm", "fp8_gemm", _fp8_gemm),
        KernelCase("gathergemv-8192", "gather_gemv", "gather_gemv", _gather_gemv),
        KernelCase("welford-262kx1k", "welford", "welford", _welford),
    ]
}


def load_kernel(case: KernelCase) -> object:
    import importlib

    module = importlib.import_module(f"examples.{case.module}")
    return getattr(module, case.attr)


def main() -> None:
    for name in KERNEL_CASES:
        print(name)


if __name__ == "__main__":
    main()
