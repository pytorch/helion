"""TPU/Pallas benchmark runner for Helion examples.

Runs selected Helion examples with autotuning on TPU and reports results
in the same JSON format as the GPU benchmark runner (benchmarks/run.py).

Usage:
    # Run all default kernels
    HELION_BACKEND=pallas python benchmarks/run_tpu.py

    # Run specific kernels
    HELION_BACKEND=pallas python benchmarks/run_tpu.py --kernel exp,add

    # Output results to JSON (compatible with pytorch benchmark hub)
    HELION_BACKEND=pallas python benchmarks/run_tpu.py --output results.json

    # List available kernels
    HELION_BACKEND=pallas python benchmarks/run_tpu.py --list-kernels
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
import functools
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch
from torch import nn

from helion._testing import DEVICE
from helion._testing import run_example

if TYPE_CHECKING:
    import types

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# Shape generators for multi-shape benchmarking.
# Each returns a list of (label, args_tuple) pairs.
def _exp_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/exp.py main() (10240*10240 = ~105M elems).
    sizes = [10240 * 10240, 16384 * 16384, 1048576, 262144, 65536, 16384, 4096, 1024]
    return [
        (
            f"[{n}]",
            (torch.randn(n, device=DEVICE, dtype=torch.float32, requires_grad=True),),
        )
        for n in sizes
    ]


def _add_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/add.py main() (1024x1024).
    # 1st = canonical small (matches main()); 2nd = very large (escape the
    # ~180 µs torch_tpu sync overhead floor and exercise memory-bandwidth).
    sizes = [(1024, 1024), (16384, 16384), (2048, 2048), (512, 512), (128, 128)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in sizes
    ]


def _softmax_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # 1st small/canonical; 2nd very large to be memory-bound.
    shapes = [(1024, 256), (8192, 8192), (1024, 512), (1024, 2048), (1024, 4096)]
    return [
        (
            f"[{m},{n}]",
            (torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),),
        )
        for m, n in shapes
    ]


def _welford_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # 1st canonical; 2nd is 2x the rows (same D). welford's autotune is
    # expensive (~16 min/shape at full effort), and (524288, 4096) blew
    # through the 60-min cap last run — back off to (524288, 1024).
    configs = [(262144, 1024), (524288, 1024), (262144, 1536), (262144, 2048)]
    return [
        (
            f"[{s},{d}]",
            (
                torch.rand(d, device=DEVICE, dtype=torch.float32),
                torch.rand(d, device=DEVICE, dtype=torch.float32),
                torch.rand(s, d, device=DEVICE, dtype=torch.float32),
            ),
        )
        for s, d in configs
    ]


def _welford_baseline(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


def _attention_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/attention.py main() so --num-shapes 1 gives
    # the canonical example config.
    # First entry mirrors examples/attention.py main(); second is the
    # reviewer's flagship shape — much larger seq_len + head_dim so the
    # bench exercises real LM-scale attention.
    # 2nd shape doubles seq_len. Tried (4,32,4096,128) and (8,32,8192,128)
    # but the autotuner default OOMs at d=128 + larger seq before the search
    # can compute its baseline. (2,32,2048,64) is the largest that fits.
    configs = [
        (2, 32, 1024, 64),
        (2, 32, 2048, 64),
        (1, 4, 512, 64),
        (1, 4, 1024, 64),
        (2, 8, 512, 64),
    ]
    return [
        (
            f"[{z},{h},{n},{d}]",
            tuple(
                torch.randn(z, h, n, d, device=DEVICE, dtype=torch.bfloat16)
                for _ in range(3)
            ),
        )
        for z, h, n, d in configs
    ]


def _bmm_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [
        (16, 512, 768, 1024),
        (64, 2048, 2048, 2048),
        (8, 256, 512, 256),
        (4, 1024, 512, 512),
    ]
    return [
        (
            f"[{b},{m},{k},{n}]",
            (
                torch.randn(b, m, k, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(b, k, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for b, m, k, n in configs
    ]


def _matmul_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/matmul.py main()'s check(1024, 1024, 1024).
    configs = [(1024, 1024, 1024), (8192, 8192, 8192), (1024, 2048, 2048)]
    return [
        (
            f"[{m},{k},{n}]",
            (
                torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, k, n in configs
    ]


def _matmul_layernorm_baseline(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    matmul_out = torch.matmul(x, y)
    return torch.nn.functional.layer_norm(
        matmul_out.to(torch.float32),
        normalized_shape=[matmul_out.shape[-1]],
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
    ).to(matmul_out.dtype)


def _matmul_layernorm_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # Use larger, regular shapes than examples/matmul_layernorm.py main()
    # (which uses small/odd n=200,400 to dodge an unrelated power-of-2 bug).
    # (4096,4096,4096) hit "Default config failed while computing baseline"
    # — autotuner default OOMs before search starts.
    configs = [(1024, 1024, 1024), (2048, 2048, 2048)]
    return [
        (
            f"[{m},{k},{n}]",
            (
                torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, k, n in configs
    ]


def _broadcast_matmul_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [
        (16, 512, 768, 1024),
        (64, 2048, 2048, 2048),
        (8, 256, 512, 256),
        (4, 1024, 512, 512),
    ]
    return [
        (
            f"[{b},{m},{k},{n}]",
            (
                torch.randn(b, m, k, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for b, m, k, n in configs
    ]


def _geglu_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/geglu.py main()'s first kernel_test_shape.
    shapes = [
        (8, 2048, 4096),
        (16, 8192, 8192),
        (4096, 2048),
        (2048, 1024),
        (1024, 512),
    ]
    return [
        (
            "[" + ",".join(str(d) for d in s) + "]",
            (
                torch.randn(*s, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(*s, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for s in shapes
    ]


def _geglu_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.functional.gelu(a, approximate="tanh").to(b.dtype) * b


def _swiglu_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/swiglu.py main()'s first kernel_test_shape.
    shapes = [
        (4, 8192, 4096),
        (16, 16384, 4096),
        (8, 8192, 4096),
        (4096, 2048),
        (1024, 512),
    ]
    return [
        (
            "[" + ",".join(str(d) for d in s) + "]",
            (
                torch.randn(*s, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(*s, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for s in shapes
    ]


def _low_mem_dropout_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/low_mem_dropout.py main()'s first check call.
    sizes = [8192, 33554432, 32768, 262144, 65536, 16384, 4096]
    return [
        (
            f"[{n}]",
            (0.25, torch.randn(n, device=DEVICE, dtype=torch.float32), 123),
        )
        for n in sizes
    ]


def _low_mem_dropout_baseline(p: float, x: torch.Tensor, seed: int) -> torch.Tensor:
    return nn.functional.dropout(x, p=p, training=True)


def _swiglu_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.functional.silu(a).to(b.dtype) * b


def _se_block_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # 1st matches examples/se_block.py main() (1024, 1024); 2nd very large.
    configs = [(1024, 1024), (8192, 8192)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(n, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in configs
    ]


def _se_block_baseline(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return 2 * x * torch.sigmoid(x @ w)


def _squeeze_and_excitation_net_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # 1st matches examples/squeeze_and_excitation_net.py main() (1024,1024,1024);
    # 2nd very large.
    configs = [(1024, 1024, 1024), (4096, 4096, 4096)]
    return [
        (
            f"[{m},{n},{k}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.float32),
                torch.randn(n, k, device=DEVICE, dtype=torch.float32),
                torch.randn(k, n, device=DEVICE, dtype=torch.float32),
            ),
        )
        for m, n, k in configs
    ]


def _squeeze_and_excitation_net_baseline(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b))


def _rms_norm_bwd_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # 1st small canonical; 2nd very large to exercise scaling.
    # rsqrt is computed from x via the fwd kernel.
    configs = [(2048, 4096), (8192, 8192)]
    out: list[tuple[str, tuple[Any, ...]]] = []
    eps = 1e-5
    for m, n in configs:
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        grad_out = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        # rsqrt is [m, 1]: 1 / sqrt(mean(x^2) + eps)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(var + eps)
        out.append((f"[{m},{n}]", (grad_out, x, weight, rsqrt)))
    return out


def _rms_norm_bwd_baseline(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirror the kernel's signature: (dx, dw).
    x_norm = x * rsqrt
    dx_norm = grad_out * weight
    dw = (grad_out * x_norm).sum(dim=0)
    # dx = (dx_norm - x_norm * mean(dx_norm * x_norm)) * rsqrt
    mean_term = (dx_norm * x_norm).mean(dim=-1, keepdim=True)
    dx = (dx_norm - x_norm * mean_term) * rsqrt
    return dx, dw


def _cross_entropy_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # Pallas constrains shapes here for two reasons:
    # 1. PR #2054's gather strategy caps the logits table at 16 MiB VMEM.
    # 2. The kernel reads a full [tile_n, V] row into VMEM, so V drives the
    #    window size; with tile_n=128 fp32, V <= ~2048 to stay under the
    #    64 MiB VMEM cap when combined with the gather one-hot intermediate.
    # Pick small but representative shapes accordingly.
    # Only one shape fits comfortably under TPU VMEM at fp32; (256, 2048)
    # already OOMs (75 MB > 64 MB cap), so we don't add a "larger" 2nd shape.
    configs = [(128, 2048)]
    return [
        (
            f"[{n},{v}]",
            (
                torch.randn(n, v, device=DEVICE, dtype=torch.float32),
                torch.randint(0, v, (n,), device=DEVICE, dtype=torch.int32),
            ),
        )
        for n, v in configs
    ]


def _cross_entropy_baseline(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # torch.nn.functional.cross_entropy requires Long labels; Pallas requires
    # int32. Cast at the baseline boundary.
    return nn.functional.cross_entropy(logits, labels.long())


def _embedding_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry mirrors examples/embedding.py main() (16x64 weights, [256,32]
    # indices). PR #2054's gather strategy keeps the weight table in VMEM
    # (16 MiB threshold), so we cap the larger shapes accordingly.
    # 2nd shape kept modestly larger than the 1st; (1024,128,4096,256) made
    # autotune take >30 min, exceeding the per-kernel budget.
    configs = [(256, 32, 16, 64), (512, 64, 32, 64), (1024, 128, 4096, 256)]
    return [
        (
            f"[{a},{b},{ne},{ed}]",
            (
                torch.randint(0, ne, (a, b), device=DEVICE, dtype=torch.int32),
                torch.randn(ne, ed, device=DEVICE, dtype=torch.float32),
            ),
        )
        for a, b, ne, ed in configs
    ]


def _batch_softmax_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [(16, 512, 1024), (64, 2048, 4096), (32, 512, 1024), (4, 1024, 512)]
    return [
        (
            f"[{b},{m},{n}]",
            (torch.randn(b, m, n, device=DEVICE, dtype=torch.bfloat16),),
        )
        for b, m, n in configs
    ]


def _rms_norm_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # (8192,16384) hit "No working config found" — search space exhausted at
    # 16384 trailing dim. Step back to (8192,8192).
    configs = [(2048, 4096), (8192, 8192), (2048, 8192), (4096, 4096)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in configs
    ]


def _rms_norm_baseline(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    return (weight * (hidden * torch.rsqrt(variance + 1e-5))).to(x.dtype)


def _layer_norm_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/layer_norm.py main() (4096, 10240).
    configs = [(4096, 10240), (16384, 16384), (2048, 4096), (2048, 8192), (4096, 4096)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                [n],
                torch.randn(n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(n, device=DEVICE, dtype=torch.bfloat16),
                1e-5,
            ),
        )
        for m, n in configs
    ]


def _softmax_shapes_basic() -> list[tuple[str, tuple[Any, ...]]]:
    # The kernel specializes on the trailing dim, so changing it across the
    # 2nd shape produces a shape-mismatch failure. Scale M instead for the 2nd.
    # Kernel specializes on the trailing dim, so keep N=2560 across shapes.
    # Scale M to grow memory traffic for the 2nd shape.
    shapes = [(4096, 2560), (65536, 2560), (8192, 2560), (1024, 8192)]
    return [
        (
            f"[{m},{n}]",
            (torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),),
        )
        for m, n in shapes
    ]


def _sum_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # First entry matches examples/sum.py main() so --num-shapes 1 gives the
    # canonical example config (fp32).
    shapes = [(5120, 2560), (16384, 16384), (2048, 8192)]
    return [
        (
            f"[{m},{n}]",
            (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        )
        for m, n in shapes
    ]


def _long_sum_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    # Long reduction dim: 131072 = 4x the 32768 block size used by the
    # looped variants, so they actually loop.
    # (64, 524288) fp32 is 128 MB just for the input — VMEM OOM (cap is 64 MB).
    # Keep the reduction dim 524288 to test long reductions, but smaller batch.
    shapes = [(4, 131072), (8, 524288)]
    return [
        (
            f"[{m},{n}]",
            (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        )
        for m, n in shapes
    ]


# Kernel mappings for TPU/Pallas benchmarks.
# Format: kernel_name -> (module_file, kernel_fn_name, baseline_fn, shapes_fn,
#                         max_mismatch_pct)
#   module_file: filename in examples/ (without .py)
#   kernel_fn_name: attribute name of the helion kernel in the module
#   baseline_fn: callable that produces reference output (None = call main())
#   shapes_fn: callable returning list of (label, args) pairs (None = call main())
#   max_mismatch_pct: fraction of elements allowed to mismatch (default None = strict)
#
# This list contains only kernels that reliably pass on Pallas/TPU.
# Each value is (module_file, kernel_fn_name, baseline_fn, shapes_fn,
# max_mismatch_pct[, torch_compile_compatible]). The optional 6th element
# defaults to True and should be set to False for kernels whose baseline
# cannot be compiled with torch.compile(backend="tpu") (e.g. dropout RNG).
KernelMapping = (
    tuple[
        str,
        str,
        Callable[..., Any] | None,
        Callable[[], list[tuple[str, tuple[Any, ...]]]] | None,
        float | None,
    ]
    | tuple[
        str,
        str,
        Callable[..., Any] | None,
        Callable[[], list[tuple[str, tuple[Any, ...]]]] | None,
        float | None,
        bool,
    ]
)
KERNEL_MAPPINGS: dict[str, KernelMapping] = {
    "exp": ("exp", "exp", torch.exp, _exp_shapes, None),
    "add": ("add", "add", torch.add, _add_shapes, None),
    "softmax_two_pass": (
        "softmax",
        "softmax_two_pass",
        functools.partial(torch.softmax, dim=-1),
        _softmax_shapes,
        None,
    ),
    "welford": ("welford", "welford", _welford_baseline, _welford_shapes, None),
    "attention": (
        "attention",
        "attention",
        torch.nn.functional.scaled_dot_product_attention,
        _attention_shapes,
        None,
    ),
    "bmm": ("bmm", "bmm", torch.bmm, _bmm_shapes, None),
    "matmul": ("matmul", "matmul", torch.matmul, _matmul_shapes, None),
    "matmul_layernorm": (
        "matmul_layernorm",
        "matmul_layernorm",
        _matmul_layernorm_baseline,
        _matmul_layernorm_shapes,
        None,
    ),
    "cross_entropy": (
        "cross_entropy",
        "cross_entropy",
        _cross_entropy_baseline,
        _cross_entropy_shapes,
        None,
    ),
    "embedding": (
        "embedding",
        "embedding",
        torch.nn.functional.embedding,
        _embedding_shapes,
        None,
    ),
    "broadcast_matmul": (
        "broadcast_matmul",
        "broadcast_matmul",
        torch.matmul,
        _broadcast_matmul_shapes,
        None,
    ),
    "geglu": ("geglu", "geglu", _geglu_baseline, _geglu_shapes, None),
    # low_mem_dropout: helion uses a deterministic seed-based mask while
    # torch.nn.functional.dropout uses a random mask, so outputs always differ.
    # Allow 100% mismatch to skip correctness but still benchmark both.
    # torch.compile(backend="tpu") can't capture dropout RNG into a graph
    # (BackendCompilerFailed), so disable the torch.compile baseline here.
    "low_mem_dropout": (
        "low_mem_dropout",
        "low_mem_dropout",
        _low_mem_dropout_baseline,
        _low_mem_dropout_shapes,
        1.0,
        False,  # torch_compile_compatible
    ),
    "swiglu": ("swiglu", "swiglu_fwd", _swiglu_baseline, _swiglu_shapes, None),
    "batch_softmax": (
        "batch_softmax",
        "batch_softmax",
        functools.partial(torch.softmax, dim=-1),
        _batch_softmax_shapes,
        None,
    ),
    "sum": (
        "sum",
        "sum_kernel",
        functools.partial(torch.sum, dim=-1),
        _sum_shapes,
        None,
    ),
    "long_sum_naive": (
        "long_sum",
        "longsum",
        functools.partial(torch.sum, dim=-1),
        _long_sum_shapes,
        None,
    ),
    "long_sum_loop": (
        "long_sum",
        "longsum_w_red_loop",
        functools.partial(torch.sum, dim=-1),
        _long_sum_shapes,
        None,
    ),
    "long_sum_manual": (
        "long_sum",
        "longsum_manual",
        functools.partial(torch.sum, dim=-1),
        _long_sum_shapes,
        None,
    ),
    "layer_norm": (
        "layer_norm",
        "layer_norm",
        torch.nn.functional.layer_norm,
        _layer_norm_shapes,
        None,
    ),
    "softmax": (
        "softmax",
        "softmax",
        functools.partial(torch.softmax, dim=-1),
        _softmax_shapes_basic,
        None,
    ),
    # rms_norm_fwd returns (output, inv_rms) but _rms_norm_baseline returns
    # only the primary tensor, so run_kernel_inner unwraps the kernel output.
    # rms_norm_bwd's baseline returns a tuple too, so no unwrap there.
    "rms_norm": (
        "rms_norm",
        "rms_norm_fwd",
        _rms_norm_baseline,
        _rms_norm_shapes,
        None,
    ),
    "rms_norm_bwd": (
        "rms_norm",
        "rms_norm_bwd",
        _rms_norm_bwd_baseline,
        _rms_norm_bwd_shapes,
        None,
    ),
    "se_block": (
        "se_block",
        "se_block_fwd",
        _se_block_baseline,
        _se_block_shapes,
        None,
    ),
    "squeeze_and_excitation_net": (
        "squeeze_and_excitation_net",
        "squeeze_and_excitation_net_fwd",
        _squeeze_and_excitation_net_baseline,
        _squeeze_and_excitation_net_shapes,
        None,
    ),
}


@dataclass
class ShapeResult:
    shape: str
    passed: bool
    kernel_time_ms: float = 0.0
    baseline_time_ms: float = 0.0
    compile_baseline_time_ms: float = 0.0
    speedup: float = 0.0  # Helion vs default torch (kDefault).
    compile_vs_default: float = 0.0  # default-torch / torch.compile (full-graph win).
    error: str | None = None


@dataclass
class KernelResult:
    name: str
    passed: bool
    kernel_time_ms: float = 0.0
    error: str | None = None
    shape_results: list[ShapeResult] = field(default_factory=list)


def import_example(module_file: str) -> types.ModuleType:
    """Import an example module by filename."""
    module_path = EXAMPLES_DIR / f"{module_file}.py"
    spec = importlib.util.spec_from_file_location(
        f"examples.{module_file}", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KERNEL_TIMEOUT = int(os.environ.get("HELION_BENCHMARK_KERNEL_TIMEOUT", "1200"))
NUM_SHAPES: int | None = None  # Set from CLI; None means all shapes


class _KernelTimeout(Exception):
    """Raised by SIGALRM when a kernel exceeds its timeout."""


def _alarm_handler(signum: int, frame: object) -> None:
    raise _KernelTimeout


def run_kernel(name: str) -> KernelResult:
    """Run a single kernel benchmark with a signal-based timeout.

    Uses SIGALRM instead of multiprocessing to avoid fork-after-TPU-init
    deadlocks on Linux.
    """
    # Reset torch._dynamo state between kernels: each kernel calls
    # torch.compile(baseline_fn, backend="tpu", fullgraph=True) and the
    # global recompile_limit (default 8) is shared across kernels. Without
    # the reset, kernels late in the run hit "Hard failure due to
    # fullgraph=True" once dynamo gives up recompiling.
    torch._dynamo.reset()
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(KERNEL_TIMEOUT)
    try:
        return run_kernel_inner(name)
    except _KernelTimeout:
        return KernelResult(
            name=name,
            passed=False,
            error=f"Timed out after {KERNEL_TIMEOUT}s",
        )
    except Exception as e:
        return KernelResult(name=name, passed=False, error=str(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_kernel_inner(name: str) -> KernelResult:
    """Run a single kernel benchmark: accuracy check + timing vs baseline."""
    if name not in KERNEL_MAPPINGS:
        return KernelResult(name=name, passed=False, error=f"Unknown kernel: {name}")

    mapping = KERNEL_MAPPINGS[name]
    module_file, kernel_fn_name, baseline_fn, shapes_fn, max_mismatch_pct = mapping[:5]
    torch_compile_compatible = mapping[5] if len(mapping) > 5 else True

    try:
        mod = import_example(module_file)
        kernel_fn = getattr(mod, kernel_fn_name)

        # For kernels with None baseline/shapes, call main() directly
        # (they have complex setup that's hard to replicate here)
        if baseline_fn is None or shapes_fn is None:
            start = time.perf_counter()
            mod.main()
            elapsed = time.perf_counter() - start
            return KernelResult(
                name=name,
                passed=True,
                kernel_time_ms=elapsed * 1000,
            )

        shapes = shapes_fn()
        if NUM_SHAPES is not None:
            shapes = shapes[:NUM_SHAPES]
        all_passed = True
        shape_results: list[ShapeResult] = []

        # Some kernels (e.g. rms_norm_fwd) return (primary, aux) but their
        # baseline returns only the primary tensor; in that case unwrap the
        # kernel output so run_example's len-equality check passes. When the
        # baseline ALSO returns a tuple (e.g. rms_norm_bwd → (grad_x, grad_w)),
        # leave both sides as tuples so they compare element-wise.
        def _unwrap_first(fn: Callable[..., Any]) -> Callable[..., torch.Tensor]:
            @functools.wraps(fn)
            def wrapper(*a: object) -> torch.Tensor:
                result = fn(*a)
                return result[0] if isinstance(result, tuple) else result

            return wrapper

        if not isinstance(baseline_fn(*shapes[0][1]), tuple):
            kernel_fn = _unwrap_first(kernel_fn)

        for label, args in shapes:
            print(f"  Shape {label}:", file=sys.stderr)
            try:
                # Build baselines dict. "torch" is the default kDefault path
                # (lazy-eager); "torch_compile" runs the same baseline through
                # torch.compile(backend="tpu") which uses kDeferAll / full graph.
                # Some baselines can't be compiled (e.g., dropout RNG); for
                # those, the kernel mapping sets torch_compile_compatible=False.
                baselines: dict[str, Callable[..., Any]] = {"torch": baseline_fn}
                if torch_compile_compatible:
                    try:
                        baselines["torch_compile"] = torch.compile(
                            baseline_fn,
                            backend="tpu",
                            dynamic=False,
                            fullgraph=True,
                        )
                    except Exception as e:
                        print(
                            f"    torch.compile setup failed; skipping torch.compile column: {e}",
                            file=sys.stderr,
                        )

                timings = run_example(
                    kernel_fn, baselines, args, max_mismatch_pct=max_mismatch_pct
                )
                kernel_ms = timings.get("helion", 0.0)
                baseline_ms = timings.get("torch", 0.0)
                compile_baseline_ms = timings.get("torch_compile", 0.0)
                speedup = baseline_ms / kernel_ms if kernel_ms > 0 else 0.0
                compile_vs_default = (
                    baseline_ms / compile_baseline_ms
                    if compile_baseline_ms > 0 and baseline_ms > 0
                    else 0.0
                )
                shape_results.append(
                    ShapeResult(
                        shape=label,
                        passed=True,
                        kernel_time_ms=kernel_ms,
                        baseline_time_ms=baseline_ms,
                        compile_baseline_time_ms=compile_baseline_ms,
                        speedup=speedup,
                        compile_vs_default=compile_vs_default,
                    )
                )
            except Exception as e:
                print(f"    FAIL: {e}", file=sys.stderr)
                shape_results.append(
                    ShapeResult(shape=label, passed=False, error=str(e))
                )
                all_passed = False

        return KernelResult(name=name, passed=all_passed, shape_results=shape_results)

    except Exception as e:
        return KernelResult(name=name, passed=False, error=str(e))


def write_results_json(output: str, results: list[KernelResult]) -> None:
    """Write results in the same JSON format as benchmarks/run.py for pytorch benchmark hub."""
    device = os.environ.get("HELION_BACKEND", "pallas")
    records: list[dict[str, Any]] = []
    for result in results:
        if result.shape_results:
            for sr in result.shape_results:
                records.append(
                    {
                        "benchmark": {
                            "name": "Helion TPU Benchmark",
                            "extra_info": {"device": device},
                        },
                        "model": {"name": result.name},
                        "metric": {
                            "name": "accuracy",
                            "benchmark_values": [1.0 if sr.passed else 0.0],
                        },
                        "shape": [sr.shape],
                    }
                )
        else:
            records.append(
                {
                    "benchmark": {
                        "name": "Helion TPU Benchmark",
                        "extra_info": {"device": device},
                    },
                    "model": {"name": result.name},
                    "metric": {
                        "name": "accuracy",
                        "benchmark_values": [1.0 if result.passed else 0.0],
                    },
                    "shape": [],
                }
            )
        if result.kernel_time_ms > 0:
            records.append(
                {
                    "benchmark": {
                        "name": "Helion TPU Benchmark",
                        "extra_info": {"device": device},
                    },
                    "model": {"name": result.name},
                    "metric": {
                        "name": "kernel_time_ms",
                        "benchmark_values": [result.kernel_time_ms],
                    },
                    "shape": [],
                }
            )

    if os.path.exists(output):
        try:
            with open(output) as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    records = existing + records
        except (OSError, json.JSONDecodeError):
            pass

    with open(output, "w") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPU/Pallas benchmark runner for Helion examples",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--kernel",
        "--op",
        type=str,
        dest="kernel",
        help="Comma-separated list of kernels to run. If not specified, runs all kernels.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (compatible with pytorch benchmark hub)",
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=None,
        help="Max number of shapes to benchmark per kernel (default: all)",
    )
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="List available kernel names and exit",
    )
    args = parser.parse_args()

    global NUM_SHAPES
    NUM_SHAPES = args.num_shapes

    if args.list_kernels:
        for name in KERNEL_MAPPINGS:
            print(name)
        return

    if args.kernel:
        kernel_names = [k.strip() for k in args.kernel.split(",") if k.strip()]
        # Validate
        for name in kernel_names:
            if name not in KERNEL_MAPPINGS:
                print(f"Error: Unknown kernel '{name}'", file=sys.stderr)
                print(
                    f"Available kernels: {', '.join(KERNEL_MAPPINGS.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        kernel_names = list(KERNEL_MAPPINGS.keys())

    print(
        f"Running {len(kernel_names)} TPU kernels: {', '.join(kernel_names)}",
        file=sys.stderr,
    )
    print(
        f"HELION_BACKEND={os.environ.get('HELION_BACKEND', '(not set)')}",
        file=sys.stderr,
    )
    print("=" * 65, file=sys.stderr)

    results: list[KernelResult] = []
    for name in kernel_names:
        print(f"\n{'=' * 65}", file=sys.stderr)
        print(f"Kernel: {name}", file=sys.stderr)
        print(f"{'=' * 65}", file=sys.stderr)
        result = run_kernel(name)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  Status: {status}", file=sys.stderr)
        if result.error:
            print(f"  Error: {result.error}", file=sys.stderr)
        if result.shape_results:
            for sr in result.shape_results:
                sr_status = "PASS" if sr.passed else "FAIL"
                print(f"    {sr.shape}: {sr_status}", file=sys.stderr)

    # Summary table. "Speedup" = Helion vs default torch (kDefault).
    # "vs default" next to torch.compile = how much faster compile is than
    # default torch (1.0x means no win, 2.0x means compile is 2x faster).
    width = 100
    print(f"\n{'=' * width}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * width}", file=sys.stderr)
    print(
        f"{'Kernel':<22} {'Shape':<16} {'Status':<8} "
        f"{'Helion (ms)':<14} {'Torch (ms)':<14} {'Speedup':<10} "
        f"{'Torch.compile (ms)':<20} {'vs default':<10}",
        file=sys.stderr,
    )
    print(f"{'-' * width}", file=sys.stderr)
    for result in results:
        if result.shape_results:
            for sr in result.shape_results:
                status = "PASS" if sr.passed else "FAIL"
                kernel_str = (
                    f"{sr.kernel_time_ms:.4f}" if sr.kernel_time_ms > 0 else "-"
                )
                baseline_str = (
                    f"{sr.baseline_time_ms:.4f}" if sr.baseline_time_ms > 0 else "-"
                )
                speedup_str = f"{sr.speedup:.2f}x" if sr.speedup > 0 else "-"
                compile_str = (
                    f"{sr.compile_baseline_time_ms:.4f}"
                    if sr.compile_baseline_time_ms > 0
                    else "-"
                )
                compile_vs_str = (
                    f"{sr.compile_vs_default:.2f}x"
                    if sr.compile_vs_default > 0
                    else "-"
                )
                print(
                    f"{result.name:<22} {sr.shape:<16} {status:<8} "
                    f"{kernel_str:<14} {baseline_str:<14} {speedup_str:<10} "
                    f"{compile_str:<20} {compile_vs_str:<10}",
                    file=sys.stderr,
                )
        else:
            status = "PASS" if result.passed else "FAIL"
            time_str = (
                f"{result.kernel_time_ms:.1f}" if result.kernel_time_ms > 0 else "-"
            )
            print(
                f"{result.name:<22} {'main()':<16} {status:<8} "
                f"{time_str:<14} {'-':<14} {'-':<10} "
                f"{'-':<20} {'-':<10}",
                file=sys.stderr,
            )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{'-' * width}", file=sys.stderr)
    print(f"Total: {passed}/{total} passed", file=sys.stderr)
    print(f"{'=' * width}\n", file=sys.stderr)

    if args.output:
        write_results_json(args.output, results)
        print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
