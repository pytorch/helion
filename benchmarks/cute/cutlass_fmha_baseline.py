"""CUTLASS CuTe-DSL Blackwell FMHA forward baseline wrapper.

This wraps NVIDIA's CUTLASS CuTe-DSL Blackwell SM100 fused multi-head attention
forward reference kernel so it can serve as the strongest "hand-written CuTe"
attention baseline (quack ships no attention kernel).

The reference lives outside this repo at::

    <CUTLASS>/examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py
    class BlackwellFusedMultiHeadAttentionForward

and is imported by file path (its ``__main__`` sys.path hack is bypassed, so the
example root must be on ``PYTHONPATH`` for ``from helpers import fmha_helpers``).

Important reference constraints (from fmha.py ``run()`` validation):
  * Supported head dimensions: 32, 64, 128.
  * ``in_dtype`` / ``out_dtype`` must be Float8E4M3FN or Float16 -- **bfloat16 is
    NOT supported** by this kernel. For bf16 shapes we substitute fp16 (identical
    FLOPs, near-identical perf on Blackwell) and the caller should note this.
  * ``mma_tiler_mn`` must be (128, 128).
  * Accumulation dtypes (qk/pv) must be Float32.

Layout note: the reference uses ``(B, S, H, D)`` tensors. Helion's
compare_attention_backends.py uses ``(z, h, seq, head_dim)``. The latency is
layout-independent for a square problem, so ``run_cutlass_fmha_latency`` takes
the Helion ``(z, h, seq, head_dim)`` convention and internally maps to the
reference's ``(B, S, H, D) = (z, seq, h, head_dim)`` shape.

Usage (standalone, prints one JSON line)::

    PYTHONPATH=<CUTLASS>/examples/python/CuTeDSL CUDA_VISIBLE_DEVICES=6 \
      python benchmarks/cute/cutlass_fmha_baseline.py \
        --z 2 --h 32 --seq-len 1024 --head-dim 64 --dtype float16 --causal 0

Because the reference JIT-compiles per shape (~7s) and mutates global import
state, each shape is best run in a fresh subprocess (which ``main()`` does when
invoked per-shape).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]

_CUTLASS_EXAMPLES_ENV = "CUTLASS_CUTEDSL_EXAMPLES"
_CUTLASS_ROOT_ENV = "CUTLASS_ROOT"
_DEFAULT_CUTLASS_EXAMPLES = REPO_ROOT / "cutlass" / "examples" / "python" / "CuTeDSL"

_FMHA_REL = "cute/blackwell/kernel/attention/fmha/fmha.py"


def _cutlass_examples_root() -> str:
    examples_root = os.environ.get(_CUTLASS_EXAMPLES_ENV)
    if examples_root:
        return examples_root
    cutlass_root = os.environ.get(_CUTLASS_ROOT_ENV)
    if cutlass_root:
        return str(Path(cutlass_root).expanduser() / "examples" / "python" / "CuTeDSL")
    return str(_DEFAULT_CUTLASS_EXAMPLES)


def _load_reference_module() -> ModuleType:
    """Import the reference fmha.py by file path.

    The example root must be importable so ``from helpers import fmha_helpers``
    resolves; we add it to sys.path (the reference's own ``__main__`` sys.path
    insert is skipped because we import it as a module, not as ``__main__``).
    """
    root = _cutlass_examples_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    fmha_path = os.path.join(root, _FMHA_REL)
    if not os.path.exists(fmha_path):
        raise FileNotFoundError(
            f"CUTLASS FMHA reference not found at {fmha_path!r}. "
            f"Set {_CUTLASS_EXAMPLES_ENV} to the CuTeDSL examples root, "
            f"set {_CUTLASS_ROOT_ENV} to a CUTLASS checkout, or clone CUTLASS "
            f"under {REPO_ROOT / 'cutlass'}."
        )
    spec = importlib.util.spec_from_file_location("cutlass_fmha_ref", fmha_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cutlass_dtype(dtype: str) -> tuple[Any, bool]:
    import cutlass

    key = dtype.lower()
    if key in ("float16", "fp16", "half", "bfloat16", "bf16"):
        # bfloat16 is unsupported by the reference; substitute fp16.
        return cutlass.Float16, (key in ("bfloat16", "bf16"))
    if key in ("float8e4m3fn", "fp8", "float8"):
        return cutlass.Float8E4M3FN, False
    raise ValueError(f"unsupported dtype {dtype!r} for cutlass fmha reference")


def run_cutlass_fmha_latency(
    z: int,
    h: int,
    seq_len: int,
    head_dim: int,
    dtype: str = "float16",
    causal: bool = False,
    *,
    warmup_iterations: int = 5,
    iterations: int = 20,
    is_persistent: bool = True,
) -> dict[str, Any]:
    """Benchmark the CUTLASS CuTe-DSL Blackwell FMHA forward for one shape.

    Args use Helion's ``(z, h, seq_len, head_dim)`` convention. Returns a dict
    with latency (ms), TFLOP/s, and bookkeeping. Raises on unsupported shapes.
    """
    if head_dim not in (32, 64, 128):
        raise ValueError(f"head_dim {head_dim} unsupported (must be 32/64/128)")

    in_dtype, dtype_substituted = _cutlass_dtype(dtype)
    out_dtype = in_dtype

    import cutlass

    module = _load_reference_module()

    # Reference layout is (B, S, H, D).
    q_shape = (z, seq_len, h, head_dim)
    k_shape = (z, seq_len, h, head_dim)

    # The CuTe DSL's cute.compile() parses sys.argv for its own ``-diagnostic``
    # flag with a strict parser, which chokes on our CLI flags. Reduce argv to
    # just the program name while the kernel compiles, then restore it.
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        # run() returns execution time in MICROSECONDS.
        exec_time_us = module.run(
            q_shape=q_shape,
            k_shape=k_shape,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            mma_tiler_mn=(128, 128),
            is_persistent=is_persistent,
            is_causal=bool(causal),
            bottom_right_align=False,
            lse_calculation=False,
            window_size=(-1, -1),
            scale_q=1.0,
            scale_k=1.0,
            scale_v=1.0,
            inv_scale_o=1.0,
            scale_softmax=0.0,
            tolerance=1e-01,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            skip_ref_check=False,
            use_cold_l2=False,
        )
    finally:
        sys.argv = saved_argv

    latency_ms = float(exec_time_us) / 1000.0
    flops = 4.0 * z * h * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5
    tflops = flops / (latency_ms * 1e9) if latency_ms > 0 else float("nan")

    return {
        "impl": "cutlass-fmha",
        "z": z,
        "h": h,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": dtype,
        "dtype_used": "float16" if dtype_substituted else dtype,
        "dtype_substituted": dtype_substituted,
        "causal": int(bool(causal)),
        "latency_ms": latency_ms,
        "tflops": tflops,
        "mma_tiler_mn": [128, 128],
        "is_persistent": is_persistent,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CUTLASS CuTe-DSL Blackwell FMHA forward baseline."
    )
    parser.add_argument("--z", type=int, default=2)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--causal", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit one JSON line.")
    args = parser.parse_args(argv)

    result = run_cutlass_fmha_latency(
        z=args.z,
        h=args.h,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        dtype=args.dtype,
        causal=bool(args.causal),
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )

    if args.json:
        print("RESULT_JSON " + json.dumps(result))
    else:
        print(
            f"cutlass-fmha z={result['z']} h={result['h']} "
            f"seq={result['seq_len']} d={result['head_dim']} "
            f"dtype={result['dtype']}(used {result['dtype_used']}) "
            f"causal={result['causal']}: "
            f"{result['latency_ms']:.4f} ms, {result['tflops']:.1f} TFLOP/s"
        )


if __name__ == "__main__":
    main()
