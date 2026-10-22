"""BF16 RMSNorm with thirteen pretuned GB300 CuTe configurations.

The checked-in SM103 heuristic selects exact contiguous BF16 shapes with
``eps=1e-5``. Other inputs use the backend's default configuration. Shape
specialization is explicit, matching the program used to measure the presets.
The kernel accumulates in FP32 and returns the input dtype.

Run ``python -m pretuned_kernels.rms_norm_cute.rms_norm_cute`` to benchmark the
included shapes with CUDA graphs and cold L2.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


SHAPES = [
    (2048, 1024),
    (2048, 4096),
    (2048, 8192),
    (2048, 16384),
    (2048, 32768),
    (4096, 3584),
    (4096, 7168),
    (16384, 8192),
    (32768, 256),
    (32768, 4096),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
]


def _make_inputs(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, float]:
    m, n = shape
    x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, dtype=x.dtype, device=x.device)
    return x, weight, 1e-5


def _collect_inputs() -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
    for shape in SHAPES:
        yield _make_inputs(shape)


def _rms_norm_shape_key(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[int, int, str, str, float, bool, bool]:
    m, n = x.size()
    return (
        m,
        n,
        str(x.dtype),
        str(weight.dtype),
        eps,
        x.is_contiguous(),
        weight.is_contiguous(),
    )


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    key=_rms_norm_shape_key,
    collect_fn=_collect_inputs,
    measure_fn=_collect_inputs,
)
def rms_norm_cute(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


def _rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return F.rms_norm(x, (x.size(1),), weight, eps=eps)


def use_cudagraph() -> bool:
    """Measure the presets with CUDA graphs and cold L2."""
    return True


def main(verbose: bool = True) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _bench import run_sweep

    baselines = [
        ("torch", _rms_norm_torch),
        ("torch_compile", torch.compile(_rms_norm_torch)),
    ]

    def make_calls(shape: tuple[int, int]) -> tuple:
        inputs = _make_inputs(shape)
        expected = _rms_norm_torch(*inputs)
        torch.testing.assert_close(
            rms_norm_cute(*inputs), expected, rtol=0.02, atol=0.02
        )

        def helion_call() -> torch.Tensor:
            return rms_norm_cute(*inputs)

        baseline_calls = [(name, (lambda fn=fn: fn(*inputs))) for name, fn in baselines]
        m, n = shape
        return helion_call, baseline_calls, f"{m:>8d}  {n:>8d}"

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>8s}  {'N':>8s}",
    )


if __name__ == "__main__":
    main()
