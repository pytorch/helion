"""CuteDSL backend benchmark: compare codegen output between Triton and CuteDSL backends.

This benchmark generates kernel code using both backends and reports:
- Code generation time
- Generated code size (lines)
- Whether the generated code is valid Python

Usage:
    python benchmarks/cutedsl_benchmark.py [--kernel <name>] [--all]

Example:
    python benchmarks/cutedsl_benchmark.py --kernel matmul
    python benchmarks/cutedsl_benchmark.py --all
"""

from __future__ import annotations

import argparse
import ast as py_ast
import sys
import time

import torch

import helion
import helion.language as hl
from helion.runtime.config import Config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

def add_combine_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def max_combine_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


# Element-wise add
@helion.kernel(backend="triton")
def _triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


@helion.kernel(backend="cutedsl")
def _cutedsl_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


# Softmax
@helion.kernel(backend="triton")
def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        row_max = hl.reduce(max_combine_fn, row, dim=1, keep_dims=True)
        row_exp = torch.exp(row - row_max)
        row_sum = hl.reduce(add_combine_fn, row_exp, dim=1, keep_dims=True)
        result[i, :] = row_exp / row_sum
    return result


@helion.kernel(backend="cutedsl")
def _cutedsl_softmax(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        row_max = hl.reduce(max_combine_fn, row, dim=1, keep_dims=True)
        row_exp = torch.exp(row - row_max)
        row_sum = hl.reduce(add_combine_fn, row_exp, dim=1, keep_dims=True)
        result[i, :] = row_exp / row_sum
    return result


# Matmul
@helion.kernel(backend="triton")
def _triton_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    result = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        result[tile_m, tile_n] = acc
    return result


@helion.kernel(backend="cutedsl")
def _cutedsl_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    result = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        result[tile_m, tile_n] = acc
    return result


# Attention
@helion.kernel(backend="triton", static_shapes=True)
def _triton_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    qk_scale = head_dim ** -0.5 / 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cutedsl", static_shapes=True)
def _cutedsl_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    qk_scale = head_dim ** -0.5 / 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "add": {
        "triton": _triton_add,
        "cutedsl": _cutedsl_add,
        "args_fn": lambda size: (
            torch.randn(size, device=DEVICE, dtype=torch.float32),
            torch.randn(size, device=DEVICE, dtype=torch.float32),
        ),
        "sizes": [(1024,), (65536,), (1048576,), (4194304,)],
        "config_kwargs": {},
    },
    "softmax": {
        "triton": _triton_softmax,
        "cutedsl": _cutedsl_softmax,
        "args_fn": lambda size: (
            torch.randn(size, device=DEVICE, dtype=torch.float32),
        ),
        "sizes": [(32, 64), (128, 256), (512, 1024), (1024, 4096)],
        "config_kwargs": {},
    },
    "matmul": {
        "triton": _triton_matmul,
        "cutedsl": _cutedsl_matmul,
        "args_fn": lambda size: (
            torch.randn(size, device=DEVICE, dtype=torch.float16),
            torch.randn(size[1], size[0], device=DEVICE, dtype=torch.float16),
        ),
        "sizes": [(128, 128), (256, 256), (512, 512), (1024, 1024)],
        "config_kwargs": {},
    },
    "attention": {
        "triton": _triton_attention,
        "cutedsl": _cutedsl_attention,
        "args_fn": lambda size: (
            torch.randn(*size, device=DEVICE, dtype=torch.float32),
            torch.randn(*size, device=DEVICE, dtype=torch.float32),
            torch.randn(*size, device=DEVICE, dtype=torch.float32),
        ),
        "sizes": [
            (1, 4, 64, 32),
            (1, 4, 128, 64),
            (2, 8, 64, 32),
        ],
        "config_kwargs": {"block_sizes": [1, 16, 16]},
    },
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_codegen(
    kernel_fn: object,
    args: tuple[torch.Tensor, ...],
    config_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Benchmark code generation for a kernel.

    Returns dict with:
        code: generated code string
        codegen_time_ms: time to generate code
        num_lines: number of lines
        parseable: whether code parses as valid Python
    """
    bound = kernel_fn.bind(args)
    if config_kwargs:
        config = Config(**config_kwargs)
    else:
        config = bound.config_spec.default_config()

    t0 = time.perf_counter()
    code = bound.to_triton_code(config)
    t1 = time.perf_counter()

    parseable = True
    try:
        py_ast.parse(code)
    except SyntaxError:
        parseable = False

    return {
        "code": code,
        "codegen_time_ms": (t1 - t0) * 1000,
        "num_lines": len(code.splitlines()),
        "parseable": parseable,
    }


def run_benchmark(kernel_name: str) -> None:
    """Run codegen benchmark for a single kernel type."""
    if kernel_name not in BENCHMARKS:
        print(f"Unknown kernel: {kernel_name}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)

    bench = BENCHMARKS[kernel_name]
    triton_fn = bench["triton"]
    cutedsl_fn = bench["cutedsl"]
    args_fn = bench["args_fn"]
    config_kwargs = bench["config_kwargs"]

    print(f"\n{'=' * 70}")
    print(f"  Benchmark: {kernel_name}")
    print(f"{'=' * 70}")

    for size in bench["sizes"]:
        args = args_fn(size)
        size_str = "x".join(str(s) for s in size)

        # Triton codegen
        triton_result = benchmark_codegen(triton_fn, args, config_kwargs)

        # CuteDSL codegen
        cutedsl_result = benchmark_codegen(cutedsl_fn, args, config_kwargs)

        print(f"\n  Size: {size_str}")
        print(f"  {'Backend':<12} {'Codegen (ms)':>14} {'Lines':>8} {'Parseable':>10}")
        print(f"  {'-' * 48}")
        print(
            f"  {'Triton':<12} {triton_result['codegen_time_ms']:>14.2f} "
            f"{triton_result['num_lines']:>8} "
            f"{'yes' if triton_result['parseable'] else 'NO':>10}"
        )
        print(
            f"  {'CuteDSL':<12} {cutedsl_result['codegen_time_ms']:>14.2f} "
            f"{cutedsl_result['num_lines']:>8} "
            f"{'yes' if cutedsl_result['parseable'] else 'NO':>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CuteDSL vs Triton codegen benchmark")
    parser.add_argument(
        "--kernel", type=str, default=None,
        help="Kernel to benchmark (add, softmax, matmul, attention). Default: all.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all benchmarks.",
    )
    args = parser.parse_args()

    print("CuteDSL Backend Codegen Benchmark")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"Compute Capability: SM{cap[0]}{cap[1]}")

    if args.kernel:
        kernels = [k.strip() for k in args.kernel.split(",")]
    else:
        kernels = list(BENCHMARKS.keys())

    for kernel_name in kernels:
        run_benchmark(kernel_name)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
