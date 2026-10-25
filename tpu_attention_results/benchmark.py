"""Benchmark fixed Helion and Tokamax TPU attention configurations.

The parent process launches each implementation and shape in a fresh child
process. TorchTPU and native JAX therefore never compete for the same PJRT
runtime. No code path in this file invokes either project's autotuner.
"""

from __future__ import annotations

import argparse
import json
import math
from operator import itemgetter
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

BATCH = 8
HEADS = 32
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


def _shape(sequence_length: int, head_dim: int) -> tuple[int, ...]:
    return BATCH, HEADS, sequence_length, head_dim


def _flops(mode: str, sequence_length: int, head_dim: int) -> int:
    if mode == "dense":
        pairs = sequence_length**2
    else:
        pairs = sequence_length * (sequence_length + 1) // 2
    # QK and probability/value each perform one multiply and one add per D.
    return 4 * BATCH * HEADS * pairs * head_dim


def _random_reference_check(
    attention: Callable[..., Any],
    make_inputs: Callable[[], tuple[Any, ...]],
    to_numpy: Callable[[Any], Any],
    mode: str,
    sequence_length: int,
    head_dim: int,
) -> dict[str, float]:
    """Run full attention, checking selected rows against an FP32 reference."""
    import numpy as np

    query, key, value = make_inputs()
    output = attention(query, key, value)
    positions = np.asarray(
        [
            0,
            1,
            sequence_length // 4,
            sequence_length // 2,
            sequence_length - 2,
            sequence_length - 1,
        ]
    )
    max_abs = 0.0
    max_relative_l2 = 0.0
    for batch, head in ((0, 0), (BATCH // 2, HEADS // 2), (BATCH - 1, HEADS - 1)):
        q = to_numpy(query[batch, head])[positions]
        k = to_numpy(key[batch, head])
        v = to_numpy(value[batch, head])
        scores = (q @ k.T) * np.float32(1.0 / math.sqrt(head_dim))
        if mode == "causal":
            scores = np.where(
                positions[:, None] >= np.arange(sequence_length)[None, :],
                scores,
                -np.inf,
            )
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected = probabilities @ v
        actual = to_numpy(output[batch, head])[positions]
        np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.01)
        relative_l2 = float(
            np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-8)
        )
        if relative_l2 > 0.02:
            raise AssertionError(f"sampled relative L2 {relative_l2} exceeds 0.02")
        max_abs = max(max_abs, float(np.max(np.abs(actual - expected))))
        max_relative_l2 = max(max_relative_l2, relative_l2)
    return {"sampled_max_abs": max_abs, "sampled_max_relative_l2": max_relative_l2}


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
        return _run_causal_helion_torchtpu_worker(
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

    shape = _shape(sequence_length, head_dim)
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


def _run_causal_helion_torchtpu_worker(
    sequence_length: int,
    head_dim: int,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    import torch

    from .causal_attention import causal_attention
    from helion._testing import DEVICE
    from helion.autotuner.benchmarking import synchronize_device

    shape = _shape(sequence_length, head_dim)
    torch.manual_seed(0)
    numerical = _random_reference_check(
        causal_attention,
        lambda: tuple(
            torch.randn(shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
        ),
        lambda tensor: tensor.cpu().float().numpy(),
        "causal",
        sequence_length,
        head_dim,
    )
    query = torch.ones(shape, dtype=torch.bfloat16, device=DEVICE)
    key = torch.ones_like(query)
    value = torch.ones_like(query)

    output = causal_attention(query, key, value)
    repeated = causal_attention(query, key, value)
    synchronize_device()
    # Check every element without full-size comparison temporaries in HBM.
    for batch in range(BATCH):
        for head in range(HEADS):
            actual_head = output[batch, head].cpu()
            repeated_head = repeated[batch, head].cpu()
            torch.testing.assert_close(
                actual_head, torch.ones_like(actual_head), atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(actual_head, repeated_head, atol=0, rtol=0)

    runtime = _median_runtime(
        lambda: causal_attention(query, key, value),
        lambda _output: synchronize_device(),
        warmups=warmups,
        samples=samples,
        calls_per_sample=calls_per_sample,
    )

    return {
        "config": causal_helion_config(head_dim, sequence_length),
        "batching": "flatten_batch_heads",
        "launcher": "torchtpu",
        "median_ms": runtime * 1e3,
        "tflops": _flops("causal", sequence_length, head_dim) / runtime / 1e12,
        **numerical,
    }


def _measure_jax_attention(
    attention: Callable[..., Any],
    mode: str,
    sequence_length: int,
    head_dim: int,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    shape = _shape(sequence_length, head_dim)
    numerical = {}
    if mode == "causal":
        numerical = _random_reference_check(
            attention,
            lambda: tuple(
                jax.random.normal(key, shape, dtype=jnp.bfloat16)
                for key in jax.random.split(jax.random.key(0), 3)
            ),
            lambda array: np.asarray(array).astype(np.float32),
            mode,
            sequence_length,
            head_dim,
        )
    query = jnp.ones(shape, dtype=jnp.bfloat16)
    key = jnp.ones_like(query)
    value = jnp.ones_like(query)

    output = attention(query, key, value)
    repeated = attention(query, key, value)
    jax.block_until_ready(repeated)
    if not bool(jnp.allclose(output, jnp.ones_like(output), atol=2e-2, rtol=2e-2)):
        raise AssertionError("all-ones correctness failed")
    if not bool(jnp.array_equal(output, repeated)):
        raise AssertionError("exact repeated-call determinism failed")

    runtime = _median_runtime(
        lambda: attention(query, key, value),
        jax.block_until_ready,
        warmups=warmups,
        samples=samples,
        calls_per_sample=calls_per_sample,
    )
    return {
        "median_ms": runtime * 1e3,
        "tflops": _flops(mode, sequence_length, head_dim) / runtime / 1e12,
        **numerical,
    }


def _run_tokamax_worker(
    mode: str,
    sequence_length: int,
    head_dim: int,
    warmups: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    from .tokamax_attention import make_tokamax_attention

    if mode == "dense":
        attention, config = make_tokamax_attention(mode, head_dim, sequence_length)
        return {
            "config": config,
            **_measure_jax_attention(
                attention,
                mode,
                sequence_length,
                head_dim,
                warmups,
                samples,
                calls_per_sample,
            ),
        }

    candidates = []
    for batching in ("vmap_batch", "flatten_batch_heads"):
        attention, config = make_tokamax_attention(
            mode, head_dim, sequence_length, batching=batching
        )
        measurement = _measure_jax_attention(
            attention,
            mode,
            sequence_length,
            head_dim,
            warmups,
            samples,
            calls_per_sample,
        )
        candidates.append({"batching": batching, **measurement})
    best = min(candidates, key=itemgetter("median_ms"))
    return {"config": config, "candidates": candidates, **best}


def _worker(arguments: argparse.Namespace) -> None:
    calls_per_sample = arguments.calls_per_sample
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
        shape=_shape(arguments.sequence_length, arguments.head_dim),
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
        "--calls-per-sample",
        str(arguments.calls_per_sample),
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
            "Fixed-config BF16 measurements. Both dense and causal attention use B=8/H=32. "
            "Speedup is Tokamax latency divided "
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
        f"TPU v7 BF16 {mode.capitalize()} Attention: B={BATCH}, H={HEADS}",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"{mode}_attention.png", dpi=180)
    plt.close(figure)


def _parent(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    metadata = {
        "libtpu_init_args": os.environ["LIBTPU_INIT_ARGS"],
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "calls_per_sample": arguments.calls_per_sample,
        "shape": [BATCH, HEADS, "S", "D"],
        "causal_q_scale": "1/sqrt(D)",
        "helion_causal_launcher": "torchtpu",
    }
    if arguments.resume and results_path.exists():
        payload = json.loads(results_path.read_text())
        for field, value in metadata.items():
            if payload["metadata"][field] != value:
                raise ValueError(f"checkpoint has a different {field}")
    else:
        payload = {
            "metadata": metadata,
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
    parser.add_argument("--calls-per-sample", type=int, default=1)
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
