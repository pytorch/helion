from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Generator
import sys

import torch

import helion
import helion.language as hl

SHAPES = (
    (64, 128, 64),
    (128, 128, 128),
    (128, 256, 256),
    (256, 256, 256),
    (512, 1024, 512),
    (1024, 1024, 1024),
    (2048, 2048, 1024),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 8192, 8192),
    (16384, 16384, 16384),
    (32768, 8192, 8192),
    (32768, 16384, 8192),
)

try:
    from tritonbench.utils.triton_op import BenchmarkOperator
    from tritonbench.utils.triton_op import register_benchmark
except ImportError:
    BenchmarkOperator = object  # type: ignore[assignment,misc]
    register_benchmark = None  # type: ignore[assignment]


@helion.kernel(backend="cute")
def scaled_mm_cute(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        acc = acc * scale_a[tile_m, :].to(torch.float32)
        acc = acc * scale_b[:, tile_n].to(torch.float32)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="triton")
def scaled_mm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        acc = acc * scale_a[tile_m, :].to(torch.float32)
        acc = acc * scale_b[:, tile_n].to(torch.float32)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def _direct_config(m: int, k: int, n: int) -> helion.Config:
    if (m, k, n) == (64, 128, 64):
        return helion.Config(block_sizes=[64, 64, 64])
    if (m, k, n) == (128, 128, 128):
        return helion.Config(block_sizes=[128, 128, 64])
    if (m, k, n) == (128, 256, 256):
        return helion.Config(block_sizes=[64, 256, 64])
    if (m, k, n) == (256, 256, 256):
        return helion.Config(block_sizes=[64, 128, 64])
    if (m, k, n) == (512, 1024, 512):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (1024, 1024, 1024):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (2048, 2048, 1024):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (4096, 4096, 4096):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (8192, 8192, 8192):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (16384, 8192, 8192):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (16384, 16384, 16384):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (32768, 8192, 8192):
        return helion.Config(block_sizes=[128, 128, 128])
    if (m, k, n) == (32768, 16384, 8192):
        return helion.Config(block_sizes=[128, 128, 128])
    return helion.Config(block_sizes=[64, 64, 64])


def scaled_mm_tritonbench(
    tb_op: object,
    a: torch.Tensor,
    b_helion: torch.Tensor,
    b_torch: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    m, k = a.size()
    _, n = b_helion.size()
    bound = scaled_mm_cute.bind((a, b_helion, scale_a, scale_b))
    compiled = bound.compile_config(_direct_config(m, k, n))
    return lambda: compiled(a, b_helion, scale_a, scale_b)


def scaled_mm_triton_backend_tritonbench(
    tb_op: object,
    a: torch.Tensor,
    b_helion: torch.Tensor,
    b_torch: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    bound = scaled_mm_triton.bind((a, b_helion, scale_a, scale_b))
    config = bound.autotune((a, b_helion, scale_a, scale_b), force=False)
    print(f"Helion Triton scaled_mm autotune config: {config}", file=sys.stderr)
    compiled = bound.compile_config(config)
    return lambda: compiled(a, b_helion, scale_a, scale_b)


if register_benchmark is not None:

    class Operator(BenchmarkOperator):  # type: ignore[misc,valid-type]
        DEFAULT_METRICS = ["latency", "speedup", "accuracy"]
        DEFAULT_PRECISION = "bypass"
        FWD_ONLY = True

        @register_benchmark(operator_name="scaled_mm", baseline=True)
        def torch_scaled_mm(
            self,
            a: torch.Tensor,
            b_helion: torch.Tensor,
            b_torch: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> Callable[[], torch.Tensor]:
            return lambda: torch._scaled_mm(
                a,
                b_torch,
                scale_a,
                scale_b,
                out_dtype=torch.bfloat16,
            )

        def get_input_iter(self) -> Generator[tuple[torch.Tensor, ...], None, None]:
            for m, k, n in SHAPES:
                torch.manual_seed(0)
                a = torch.randn(m, k, device=self.device, dtype=torch.float32).to(
                    torch.float8_e4m3fn
                )
                b_values = torch.randn(k, n, device=self.device, dtype=torch.float32)
                b_helion = b_values.to(torch.float8_e4m3fn)
                b_torch = b_values.to(torch.float8_e4m3fn).t().contiguous().t()
                scale_a = (
                    torch.rand(m, 1, device=self.device, dtype=torch.float32) * 0.25
                    + 0.875
                )
                scale_b = (
                    torch.rand(1, n, device=self.device, dtype=torch.float32) * 0.25
                    + 0.875
                )
                yield a, b_helion, b_torch, scale_a, scale_b

        def get_x_val(self, example_inputs: Any) -> tuple[int, int, int]:
            a = example_inputs[0]
            b = example_inputs[1]
            m, k = a.size()
            _, n = b.size()
            return m, k, n
