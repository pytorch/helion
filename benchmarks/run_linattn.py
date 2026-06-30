"""Linear-attention benchmark runner: Helion vs FLA.

Times the chunked linear-attention kernels in KERNEL_MAPPINGS against FLA and
reports results in the same JSON format as the GPU benchmark runner
(benchmarks/run.py). These kernels have no TritonBench operator, so FLA is the
baseline and helion_speedup is Helion vs FLA. Accuracy gates Helion's drift
from the fp32 ref_* recurrence to within ACC_FACTOR of FLA's own drift.

Usage:
    python benchmarks/run_linattn.py --output helionbench.json
    python benchmarks/run_linattn.py --kernel simple_gla,gla --num-shapes 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
import importlib
import json
import math
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast

import torch

if TYPE_CHECKING:
    import types

from helion._testing import DEVICE
from helion._testing import do_bench

# Helion passes a shape if its drift from the fp32 reference is within
# ACC_FACTOR of FLA's, so the gate tracks FLA's own bf16 accuracy. The floor
# keeps the check from being absurdly strict where FLA is near-exact.
# TODO: 2.0 is a guess; set ACC_FACTOR from measured Helion/FLA drift.
ACC_FACTOR = 2.0
ACC_FLOOR = 1e-3

# bf16 matches FLA's production baseline.
DTYPE = torch.bfloat16

# Six production shapes from flash-linear-attention/benchmarks/ops/registry.py.
# (name, B, T, H, D); D is both the query/key and the value dim (D = DV).
SHAPES: list[tuple[str, int, int, int, int]] = [
    ("B1_T8192_H96_D128", 1, 8192, 96, 128),
    ("B2_T16384_H16_D128", 2, 16384, 16, 128),
    ("B4_T2048_H16_D128", 4, 2048, 16, 128),
    ("B4_T4096_H64_D128", 4, 4096, 64, 128),
    ("B8_T2048_H32_D256", 8, 2048, 32, 256),
    ("B8_T1024_H8_D64", 8, 1024, 8, 64),
]

# kernel -> (helion entry fn, fp32 reference fn, native FLA fn, gate type).
# gate type selects the decay input the kernel takes: None (no gate), "scalar"
# (per-head log-decay, token-first [B, T, H]), or "vector" (per-channel
# log-decay, head-first [B, H, T, D]).
KERNEL_MAPPINGS: dict[str, tuple[str, str, str, str | None]] = {
    "linear_attn": (
        "chunk_linear_attn",
        "ref_linear_attn",
        "fla_chunk_linear_attn_native",
        None,
    ),
    "retention": (
        "chunk_retention",
        "ref_retention",
        "fla_chunk_retention_native",
        None,
    ),
    "simple_gla": (
        "chunk_simple_gla",
        "ref_simple_gla",
        "fla_chunk_simple_gla_native",
        "scalar",
    ),
    "gla": ("chunk_gla", "ref_gla", "fla_chunk_gla_native", "vector"),
}


@dataclass
class ShapeResult:
    shape: str
    passed: bool
    kernel_time_ms: float = 0.0
    fla_time_ms: float = 0.0
    speedup: float = 0.0  # Helion vs FLA (fla_ms / helion_ms).
    error: str | None = None


@dataclass
class KernelResult:
    name: str
    shape_results: list[ShapeResult] = field(default_factory=list)


def _make_gate(b: int, t: int, h: int, d: int, gate: str | None) -> torch.Tensor | None:
    if gate == "scalar":  # per-head log-decay, token-first [B, T, H].
        return torch.nn.functional.logsigmoid(
            torch.randn(b, t, h, dtype=torch.float32, device=DEVICE)
        )
    if gate == "vector":  # per-channel log-decay, head-first [B, H, T, D].
        return (
            torch.nn.functional.logsigmoid(torch.randn(b, h, t, d, device=DEVICE))
            .clamp_min(-5.0)
            .to(DTYPE)
        )
    return None


def _to_fla_layout(t: torch.Tensor, gate: str | None, is_gate: bool) -> torch.Tensor:
    """FLA is token-first [B, T, H, D]; Helion is head-first [B, H, T, D].

    q/k/v always transpose. The scalar gate is already token-first [B, T, H]
    (no transpose); the vector gate is head-first like q/k/v.
    """
    if is_gate and gate == "scalar":
        return t.contiguous()
    return t.transpose(1, 2).contiguous()


def _accuracy(
    ref_fn: Callable[..., torch.Tensor],
    helion_fn: Callable[..., torch.Tensor],
    fla_fn: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    gate: str | None,
    scale: float,
) -> bool:
    """Pass if Helion's forward drift vs the fp32 reference is within
    ACC_FACTOR of FLA's."""
    from fla.utils import get_err_ratio  # pyrefly: ignore

    ref_args = [q.float(), k.float(), v.float()]
    if g is not None:
        ref_args.append(g.float())
    with torch.no_grad():
        ref = ref_fn(*ref_args, scale=scale)

    h_args = [q, k, v] + ([g] if g is not None else [])
    h_out = helion_fn(*h_args, scale=scale)

    f_args = [
        _to_fla_layout(q, None, False),
        _to_fla_layout(k, None, False),
        _to_fla_layout(v, None, False),
    ]
    if g is not None:
        f_args.append(_to_fla_layout(g, gate, True))
    f_out = fla_fn(*f_args, scale=scale).transpose(1, 2)

    h_ratio = get_err_ratio(h_out, ref)
    f_ratio = get_err_ratio(f_out, ref)
    return h_ratio <= max(ACC_FACTOR * f_ratio, ACC_FLOOR)


def run_shape(
    mod: types.ModuleType,
    entry_name: str,
    ref_name: str,
    fla_name: str,
    gate: str | None,
    b: int,
    t: int,
    h: int,
    d: int,
) -> ShapeResult:
    label = next(name for (name, *dims) in SHAPES if tuple(dims) == (b, t, h, d))
    scale = 1.0 / math.sqrt(d)

    entry = getattr(mod, entry_name)
    ref = getattr(mod, ref_name)
    fla = getattr(mod, fla_name)

    q = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE)
    k = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE)
    v = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE)
    g = _make_gate(b, t, h, d, gate)

    passed = _accuracy(ref, entry, fla, q, k, v, g, gate, scale)

    # Transpose to FLA's layout off the timer so FLA isn't charged a tax the
    # head-first Helion kernel does not pay.
    f_q = _to_fla_layout(q, None, False)
    f_k = _to_fla_layout(k, None, False)
    f_v = _to_fla_layout(v, None, False)
    f_g = _to_fla_layout(g, gate, True) if g is not None else None

    h_args = [q, k, v] + ([g] if g is not None else [])
    f_args = [f_q, f_k, f_v] + ([f_g] if f_g is not None else [])

    def helion_fwd() -> torch.Tensor:
        return entry(*h_args, scale=scale)

    def fla_fwd() -> torch.Tensor:
        return fla(*f_args, scale=scale)

    # Warm up (autotune / compile) off the timed region.
    helion_fwd()
    fla_fwd()
    torch.cuda.empty_cache()

    h_ms = cast("float", do_bench(helion_fwd))
    f_ms = cast("float", do_bench(fla_fwd))
    speedup = f_ms / h_ms if h_ms > 0 else 0.0
    return ShapeResult(label, passed, h_ms, f_ms, speedup)


def run_kernel(name: str, num_shapes: int | None) -> KernelResult:
    entry_name, ref_name, fla_name, gate = KERNEL_MAPPINGS[name]
    mod = importlib.import_module(f"examples.{name}.chunk")
    result = KernelResult(name=name)
    shapes = SHAPES if num_shapes is None else SHAPES[:num_shapes]
    for label, b, t, h, d in shapes:
        try:
            sr = run_shape(mod, entry_name, ref_name, fla_name, gate, b, t, h, d)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            sr = ShapeResult(label, passed=False, error="OOM")
        except Exception as exc:
            sr = ShapeResult(label, passed=False, error=str(exc))
        result.shape_results.append(sr)
        status = "ok" if sr.passed else f"FAIL ({sr.error})" if sr.error else "FAIL"
        print(
            f"  {name:<12} {sr.shape:<22} {status:<14} "
            f"helion {sr.kernel_time_ms:>8.3f}ms  fla {sr.fla_time_ms:>8.3f}ms  "
            f"{sr.speedup * 100:>7.1f}% FLA"
        )
    return result


def write_results_json(output: str, results: list[KernelResult]) -> None:
    """Emit the dashboard's helionbench.json schema: one record per
    (kernel, metric) with parallel shape and benchmark_values arrays."""
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    records: list[dict[str, Any]] = []

    def add_metric(
        kernel: str, metric: str, shapes: list[str], values: list[float]
    ) -> None:
        if not shapes or not values:
            return
        records.append(
            {
                "benchmark": {
                    "name": "Helion Linear-Attention Benchmark",
                    "extra_info": {"device": device, "backend": "triton"},
                },
                "model": {"name": kernel},
                "metric": {"name": metric, "benchmark_values": values},
                "shape": shapes,
            }
        )

    for result in results:
        runnable = [sr for sr in result.shape_results if sr.error is None]
        if not runnable:
            continue
        shapes = [sr.shape for sr in runnable]
        add_metric(
            result.name,
            "helion_accuracy",
            shapes,
            [1.0 if sr.passed else 0.0 for sr in runnable],
        )
        add_metric(
            result.name,
            "helion_latency_ms",
            shapes,
            [sr.kernel_time_ms for sr in runnable],
        )
        add_metric(
            result.name, "helion_speedup", shapes, [sr.speedup for sr in runnable]
        )
        # FLA is the Triton reference; report it at 1.0 so the dashboard's
        # Triton column reflects the comparison baseline.
        add_metric(result.name, "triton_speedup", shapes, [1.0 for _ in runnable])

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        default=",".join(KERNEL_MAPPINGS),
        help="comma-separated kernel names (default: all)",
    )
    parser.add_argument(
        "--num-shapes", type=int, default=None, help="cap number of shapes"
    )
    parser.add_argument(
        "--output", default=None, help="write helionbench.json to this path"
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.kernel.split(",") if n.strip()]
    unknown = [n for n in names if n not in KERNEL_MAPPINGS]
    if unknown:
        raise SystemExit(
            f"unknown kernels: {unknown}; choose from {list(KERNEL_MAPPINGS)}"
        )

    results = []
    for name in names:
        print(f"=== {name} ===")
        results.append(run_kernel(name, args.num_shapes))

    if args.output:
        write_results_json(args.output, results)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
