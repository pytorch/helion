"""Benchmark fixed Helion and Tokamax TPU attention configurations.

The parent process launches each implementation and shape in a fresh child
process. TorchTPU and native JAX therefore never compete for the same PJRT
runtime. No code path in this file invokes either project's autotuner.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any

from .configs import DENSE_HELION_CONFIGS
from .configs import HEAD_DIMS
from .configs import SEQUENCE_LENGTHS
from .configs import causal_helion_config

if TYPE_CHECKING:
    from collections.abc import Callable

DENSE_BATCH = 8
DENSE_HEADS = 32
CAUSAL_BATCH = 1
CAUSAL_HEADS = 8
IMPLEMENTATIONS = ("helion", "tokamax")
MODES = ("dense", "causal")
REQUIRED_LIBTPU_FLAGS = (
    "--xla_tpu_dvfs_p_state=7",
    "--xla_tpu_scoped_vmem_limit_kib=65536",
)


def _configure_libtpu() -> None:
    existing = os.environ.get("LIBTPU_INIT_ARGS", "").split()
    prefixes = tuple(flag.split("=", 1)[0] for flag in REQUIRED_LIBTPU_FLAGS)
    extras = [flag for flag in existing if not flag.startswith(prefixes)]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join((*REQUIRED_LIBTPU_FLAGS, *extras))


def _shape(mode: str, sequence_length: int, head_dim: int) -> tuple[int, ...]:
    if mode == "dense":
        return DENSE_BATCH, DENSE_HEADS, sequence_length, head_dim
    return CAUSAL_BATCH, CAUSAL_HEADS, sequence_length, head_dim


def _flops(mode: str, sequence_length: int, head_dim: int) -> int:
    if mode == "dense":
        pairs = sequence_length**2
        batch_heads = DENSE_BATCH * DENSE_HEADS
    else:
        pairs = sequence_length * (sequence_length + 1) // 2
        batch_heads = CAUSAL_BATCH * CAUSAL_HEADS
    # QK and probability/value each perform one multiply and one add per D.
    return 4 * batch_heads * pairs * head_dim


def _median_runtime(
    function: Callable[[], Any],
    synchronize: Callable[[Any], None],
    *,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> float:
    for _ in range(warmups):
        synchronize(function())

    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        output = None
        for _ in range(calls_per_sample):
            output = function()
        synchronize(output)
        timings.append((time.perf_counter() - started) / calls_per_sample)
    return statistics.median(timings)


def _run_helion_worker(
    mode: str,
    sequence_length: int,
    head_dim: int,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    if mode == "causal":
        return _run_causal_helion_jax_worker(
            sequence_length,
            head_dim,
            warmups,
            samples,
            calls_per_sample,
        )
    import torch

    from .attention import attention_kernel
    import helion
    from helion._testing import DEVICE
    from helion.autotuner.benchmarking import synchronize_device

    shape = _shape(mode, sequence_length, head_dim)
    query = torch.ones(shape, dtype=torch.bfloat16, device=DEVICE)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    config = DENSE_HELION_CONFIGS[head_dim, sequence_length]
    kernel = helion.kernel(
        attention_kernel,
        backend="pallas",
        static_shapes=True,
        autotune_effort="none",
        config=helion.Config.from_dict(config),
    )

    output = kernel(query, key, value)
    repeated = kernel(query, key, value)
    synchronize_device()
    torch.testing.assert_close(output, torch.ones_like(output), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output, repeated, atol=0, rtol=0)

    runtime = _median_runtime(
        lambda: kernel(query, key, value),
        lambda _output: synchronize_device(),
        warmups=warmups,
        samples=samples,
        calls_per_sample=calls_per_sample,
    )
    return {
        "config": config,
        "median_ms": runtime * 1e3,
        "tflops": _flops(mode, sequence_length, head_dim) / runtime / 1e12,
    }


def _run_causal_helion_jax_worker(
    sequence_length: int,
    head_dim: int,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from .causal_attention import configured_causal_attention

    shape = (CAUSAL_HEADS, sequence_length, head_dim)
    query = jnp.ones(shape, dtype=jnp.bfloat16)
    key = jnp.ones_like(query)
    value = jnp.ones_like(query)
    query_panel_size = min(sequence_length, 2048)
    query_blocks = sequence_length // query_panel_size
    kv_group_width = 1 if sequence_length <= 4096 else 2
    query_block_ids = jnp.asarray(
        [
            query_block
            for query_block in range(query_blocks)
            for _ in range(query_block // kv_group_width + 1)
        ],
        dtype=jnp.int32,
    )
    kv_group_ids = jnp.asarray(
        [
            kv_group
            for query_block in range(query_blocks)
            for kv_group in range(query_block // kv_group_width + 1)
        ],
        dtype=jnp.int32,
    )
    kernel = configured_causal_attention(sequence_length, head_dim)

    @jax.jit
    def compiled_attention(
        q: object,
        k: object,
        v: object,
        query_ids: object,
        kv_ids: object,
    ) -> object:
        return kernel.jax_fn(q, k, v, query_ids, kv_ids)

    def attention() -> object:
        return compiled_attention(
            query,
            key,
            value,
            query_block_ids,
            kv_group_ids,
        )

    output = attention()
    repeated = attention()
    jax.block_until_ready(repeated)
    if not bool(jnp.allclose(output, jnp.ones_like(output), atol=2e-2, rtol=2e-2)):
        raise AssertionError("Helion causal attention failed the correctness check")
    if not bool(jnp.array_equal(output, repeated)):
        raise AssertionError("Helion causal attention was not deterministic")

    runtime = _median_runtime(
        attention,
        jax.block_until_ready,
        warmups=warmups,
        samples=samples,
        calls_per_sample=calls_per_sample,
    )
    return {
        "config": causal_helion_config(head_dim, sequence_length),
        "median_ms": runtime * 1e3,
        "tflops": _flops("causal", sequence_length, head_dim) / runtime / 1e12,
    }


def _run_tokamax_worker(
    mode: str,
    sequence_length: int,
    head_dim: int,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from .tokamax_attention import make_tokamax_attention

    if mode == "dense":
        shape = _shape(mode, sequence_length, head_dim)
    else:
        shape = (CAUSAL_HEADS, sequence_length, head_dim)
    query = jnp.ones(shape, dtype=jnp.bfloat16)
    key = jnp.ones_like(query)
    value = jnp.ones_like(query)
    attention, config = make_tokamax_attention(mode, head_dim, sequence_length)

    output = attention(query, key, value)
    repeated = attention(query, key, value)
    jax.block_until_ready(repeated)
    if not bool(jnp.allclose(output, jnp.ones_like(output), atol=2e-2, rtol=2e-2)):
        raise AssertionError("Tokamax attention failed the all-ones correctness check")
    if not bool(jnp.array_equal(output, repeated)):
        raise AssertionError("Tokamax attention was not deterministic")

    runtime = _median_runtime(
        lambda: attention(query, key, value),
        jax.block_until_ready,
        warmups=warmups,
        samples=samples,
        calls_per_sample=calls_per_sample,
    )
    return {
        "config": config,
        "median_ms": runtime * 1e3,
        "tflops": _flops(mode, sequence_length, head_dim) / runtime / 1e12,
    }


def _worker(arguments: argparse.Namespace) -> None:
    calls_per_sample = 1 if arguments.mode == "dense" else arguments.causal_calls
    if arguments.implementation == "helion":
        result = _run_helion_worker(
            arguments.mode,
            arguments.sequence_length,
            arguments.head_dim,
            arguments.warmups,
            arguments.samples,
            calls_per_sample,
        )
    else:
        result = _run_tokamax_worker(
            arguments.mode,
            arguments.sequence_length,
            arguments.head_dim,
            arguments.warmups,
            arguments.samples,
            calls_per_sample,
        )
    result.update(
        mode=arguments.mode,
        implementation=arguments.implementation,
        sequence_length=arguments.sequence_length,
        head_dim=arguments.head_dim,
        samples=arguments.samples,
        calls_per_sample=calls_per_sample,
    )
    print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}", flush=True)


def _result_key(
    mode: str,
    head_dim: int,
    sequence_length: int,
    implementation: str,
) -> str:
    return f"{mode}/d{head_dim}/s{sequence_length}/{implementation}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_child(
    arguments: argparse.Namespace,
    mode: str,
    head_dim: int,
    sequence_length: int,
    implementation: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "tpu_attention_results.benchmark",
        "--worker",
        "--mode",
        mode,
        "--implementation",
        implementation,
        "--head-dim",
        str(head_dim),
        "--sequence-length",
        str(sequence_length),
        "--warmups",
        str(arguments.warmups),
        "--samples",
        str(arguments.samples),
        "--causal-calls",
        str(arguments.causal_calls),
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    result = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT_JSON="):
            result = json.loads(line.removeprefix("RESULT_JSON="))
    return_code = process.wait()
    if return_code or result is None:
        raise RuntimeError(
            f"benchmark child failed with exit code {return_code}: {command}"
        )
    return result


def _display_sequence(sequence_length: int) -> str:
    return f"{sequence_length // 1024}K"


def _write_markdown(
    path: Path,
    results: dict[str, Any],
    modes: list[str],
    head_dims: list[int],
    sequence_lengths: list[int],
) -> None:
    lines = [
        "# TPU v7 attention results",
        "",
        (
            "Fixed-config BF16 measurements. Dense attention uses B=8/H=32; "
            "causal attention uses B=1/H=8. Speedup is Tokamax latency divided "
            "by Helion latency."
        ),
        "",
    ]
    for mode in modes:
        lines.extend(
            [
                f"## {mode.capitalize()} attention",
                "",
                "| D | Sequence | Helion ms | Tokamax ms | Helion TFLOP/s | Tokamax TFLOP/s | Speedup |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for head_dim in head_dims:
            for sequence_length in sequence_lengths:
                helion_result = results[
                    _result_key(mode, head_dim, sequence_length, "helion")
                ]
                tokamax_result = results[
                    _result_key(mode, head_dim, sequence_length, "tokamax")
                ]
                speedup = tokamax_result["median_ms"] / helion_result["median_ms"]
                lines.append(
                    f"| {head_dim} | {_display_sequence(sequence_length)} "
                    f"| {helion_result['median_ms']:.4f} "
                    f"| {tokamax_result['median_ms']:.4f} "
                    f"| {helion_result['tflops']:.2f} "
                    f"| {tokamax_result['tflops']:.2f} | {speedup:.3f}x |"
                )
        lines.extend(
            ("", f"![{mode.capitalize()} attention](./{mode}_attention.png)", "")
        )
    path.write_text("\n".join(lines))


def _plot(
    output_dir: Path,
    mode: str,
    results: dict[str, Any],
    head_dims: list[int],
    sequence_lengths: list[int],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(head_dims),
        figsize=(6 * len(head_dims), 5),
        sharey=False,
        squeeze=False,
    )
    for axis, head_dim in zip(axes[0], head_dims, strict=True):
        for implementation, color, marker in (
            ("helion", "#4285F4", "o"),
            ("tokamax", "#F9AB00", "^"),
        ):
            throughputs = [
                results[_result_key(mode, head_dim, sequence_length, implementation)][
                    "tflops"
                ]
                for sequence_length in sequence_lengths
            ]
            axis.plot(
                range(len(sequence_lengths)),
                throughputs,
                color=color,
                marker=marker,
                linewidth=2,
                label=implementation.capitalize(),
            )
        axis.set_title(f"D={head_dim}")
        axis.set_xticks(
            range(len(sequence_lengths)),
            [_display_sequence(length) for length in sequence_lengths],
            rotation=45,
        )
        axis.set_xlabel("Sequence length")
        axis.set_ylabel("Effective TFLOP/s")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(
        f"TPU v7 BF16 {mode.capitalize()} Attention: Helion vs Tokamax",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"{mode}_attention.png", dpi=180)
    plt.close(figure)


def _parent(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    if arguments.resume and results_path.exists():
        payload = json.loads(results_path.read_text())
    else:
        payload = {
            "metadata": {
                "libtpu_init_args": os.environ["LIBTPU_INIT_ARGS"],
                "warmups": arguments.warmups,
                "samples": arguments.samples,
                "causal_calls_per_sample": arguments.causal_calls,
                "dense_shape": [DENSE_BATCH, DENSE_HEADS, "S", "D"],
                "causal_shape": [CAUSAL_BATCH, CAUSAL_HEADS, "S", "D"],
            },
            "results": {},
        }
    results = payload["results"]

    for mode in arguments.modes:
        for head_dim in arguments.head_dims:
            for sequence_length in arguments.sequence_lengths:
                for implementation in IMPLEMENTATIONS:
                    key = _result_key(
                        mode,
                        head_dim,
                        sequence_length,
                        implementation,
                    )
                    if arguments.resume and key in results:
                        print(f"[resume] {key}", flush=True)
                        continue
                    print(f"[run] {key}", flush=True)
                    results[key] = _run_child(
                        arguments,
                        mode,
                        head_dim,
                        sequence_length,
                        implementation,
                    )
                    _write_json(results_path, payload)

    expected = {
        _result_key(mode, head_dim, sequence_length, implementation)
        for mode in arguments.modes
        for head_dim in arguments.head_dims
        for sequence_length in arguments.sequence_lengths
        for implementation in IMPLEMENTATIONS
    }
    missing = expected - results.keys()
    if missing:
        raise RuntimeError(f"missing benchmark results: {sorted(missing)}")
    for mode in arguments.modes:
        _plot(
            output_dir,
            mode,
            results,
            arguments.head_dims,
            arguments.sequence_lengths,
        )
    _write_markdown(
        output_dir / "results.md",
        results,
        arguments.modes,
        arguments.head_dims,
        arguments.sequence_lengths,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--head-dims", nargs="+", type=int, default=list(HEAD_DIMS))
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=list(SEQUENCE_LENGTHS),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--causal-calls", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> None:
    _configure_libtpu()
    arguments = _parse_arguments()
    if arguments.worker:
        required = (
            arguments.implementation,
            arguments.mode,
            arguments.head_dim,
            arguments.sequence_length,
        )
        if any(value is None for value in required):
            raise ValueError("worker mode requires implementation, mode, D, and S")
        _worker(arguments)
    else:
        _parent(arguments)


if __name__ == "__main__":
    main()
