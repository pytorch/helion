"""CuTe-backend autotune and benchmark sweep harness.

Drives a curated subset of ``test/test_examples.py`` through normal
Helion autotune under ``HELION_BACKEND=cute``. Each test runs in a
fresh subprocess with a wall-clock timeout so a hang in one example
cannot poison the next. Results are appended to a JSONL file for
later grep / diff.

Also drives the active matmul target sweep from ``cute_plan.md``:
32 matmul-family targets spanning bf16/fp16/fp32 and 8 epilogues, each
measured for ATen, Quack-direct, and Helion-CuTe by default. The
Quack-direct row uses the benchmark harness's brief per-target tuning
mode and records the selected config. Each measurement runs via
``benchmarks/cute/compare_matmul_backends.py`` in a fresh subprocess.

Usage::

    python benchmarks/cute/cute_autotune_sweep.py \\
        --output cute_sweep.jsonl

    # Run the smaller non-fp32 coverage list with a longer budget:
    python benchmarks/cute/cute_autotune_sweep.py \\
        --list-name nonfp32 \\
        --output cute_sweep_nonfp32.jsonl \\
        --timeout 1800

    # Dry-run the test list (no GPU work):
    python benchmarks/cute/cute_autotune_sweep.py --list

    # Show the active 32 x 3 matmul benchmark worklist (no GPU work):
    python benchmarks/cute/cute_autotune_sweep.py --matmul-target-sweep --list

    # Show exact benchmark commands without launching GPU work:
    python benchmarks/cute/cute_autotune_sweep.py --matmul-target-sweep --dry-run

    # Show only target 4's frequent active-target worklist:
    python benchmarks/cute/cute_autotune_sweep.py --matmul-target-sweep --matmul-targets 4 --list

    # Run a smaller subset by substring match:
    python benchmarks/cute/cute_autotune_sweep.py --filter add --filter softmax

The plugin in ``benchmarks/cute/cute_autotune_sweep_plugin.py`` patches
``helion._testing.code_and_output`` / ``output_only`` so each invocation
runs ``bound.autotune(args, force=True)``. Tests that build their own
``helion.Config(...)`` directly (e.g. ``test_split_k_barrier_accuracy``,
``test_matmul_bwd``, ``test_addmm_bwd``) bypass the plugin and run with
their forced config rather than autotune. The curated node-ID list
below intentionally omits those tests; if you add new ones, drop tests
that build their own ``Config`` and call the kernel directly.

``@skipIfCute`` tests are skipped at pytest collection. ``@xfailIfCute``
tests still *run* — pytest records them as ``xfail`` on failure (or
``xpassed`` on success), but the test body executes and can still
exercise / crash the GPU. The curated list keeps ``@xfailIfCute`` cases
out so the sweep does not spend wall-clock on paths the plan already
classifies as unsupported.

GPU policy: use physical GPUs 6 or 7 only. The pytest-node sweep inherits
``CUDA_VISIBLE_DEVICES`` from the caller. The matmul target sweep accepts
``--gpus`` as physical GPU IDs, defaults to ``6,7``, rejects GPUs 0-5,
and sets each child subprocess's ``CUDA_VISIBLE_DEVICES`` to exactly one
allowed physical GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLUGIN_MODULE = "benchmarks.cute.cute_autotune_sweep_plugin"

MATMUL_PERFORMANCE_CSV_FIELDS = (
    "timestamp",
    "run_id",
    "run_started_at",
    "run_finished_at",
    "target",
    "shape",
    "m",
    "n",
    "k",
    "dtype",
    "epilogue",
    "backend",
    "backend_label",
    "gpu",
    "tflops",
    "mom_median_ms",
    "best_tflops",
    "best_ms",
    "quack_tflops",
    "gap_vs_quack_pct",
    "passes_5pct",
    "returncode",
    "timed_out",
    "json_parse_failed",
    "wall_clock_seconds",
    "output_path",
)


@dataclass(frozen=True)
class MatmulTarget:
    target_id: int
    m: int
    n: int
    k: int
    dtype: str
    epilogue: str

    @property
    def shape(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"

    @property
    def require_tcgen05(self) -> bool:
        # fp32 inputs fall back to the universal MMA atom on B200; tcgen05
        # f16/bf16 cores are only engaged for half-precision GEMMs.
        return self.dtype in ("bfloat16", "float16")


@dataclass(frozen=True)
class MatmulBackend:
    impl: str
    label: str


@dataclass(frozen=True)
class MatmulWorkItem:
    target: MatmulTarget
    backend: MatmulBackend


MATMUL_TARGETS: tuple[MatmulTarget, ...] = (
    # bf16 (T1-T20): primary tcgen05 path.
    MatmulTarget(1, 1024, 1024, 1024, "bfloat16", "none"),
    MatmulTarget(2, 1024, 4096, 1024, "bfloat16", "bias"),
    MatmulTarget(3, 2048, 2048, 2048, "bfloat16", "none"),
    MatmulTarget(4, 2048, 4096, 2048, "bfloat16", "gelu"),
    MatmulTarget(5, 4096, 2048, 2048, "bfloat16", "bias"),
    MatmulTarget(6, 4096, 4096, 4096, "bfloat16", "none"),
    MatmulTarget(7, 8192, 1024, 1024, "bfloat16", "relu"),
    MatmulTarget(8, 1024, 8192, 1024, "bfloat16", "none"),
    MatmulTarget(9, 8192, 2048, 2048, "bfloat16", "bias_relu"),
    MatmulTarget(10, 2048, 8192, 2048, "bfloat16", "none"),
    MatmulTarget(11, 8192, 8192, 8192, "bfloat16", "bias_relu"),
    MatmulTarget(12, 5000, 5000, 5000, "bfloat16", "bias_residual_gelu"),
    MatmulTarget(13, 16384, 1024, 4096, "bfloat16", "silu"),
    MatmulTarget(14, 1024, 16384, 4096, "bfloat16", "residual_add"),
    MatmulTarget(15, 4096, 4096, 16384, "bfloat16", "none"),
    MatmulTarget(16, 4096, 4096, 512, "bfloat16", "bias"),
    MatmulTarget(17, 4096, 11008, 4096, "bfloat16", "silu"),
    MatmulTarget(18, 4096, 4096, 11008, "bfloat16", "none"),
    MatmulTarget(19, 3072, 3072, 3072, "bfloat16", "gelu"),
    MatmulTarget(20, 6144, 6144, 6144, "bfloat16", "residual_add"),
    # fp16 (T21-T28): secondary tcgen05 path.
    MatmulTarget(21, 1024, 1024, 1024, "float16", "none"),
    MatmulTarget(22, 2048, 2048, 2048, "float16", "bias"),
    MatmulTarget(23, 4096, 4096, 4096, "float16", "relu"),
    MatmulTarget(24, 8192, 4096, 2048, "float16", "bias_relu"),
    MatmulTarget(25, 2048, 4096, 2048, "float16", "residual_add"),
    MatmulTarget(26, 4096, 11008, 4096, "float16", "silu"),
    MatmulTarget(27, 1536, 6144, 1536, "float16", "gelu"),
    MatmulTarget(28, 6144, 6144, 6144, "float16", "bias_residual_gelu"),
    # fp32 (T29-T32): universal MMA fallback (no tcgen05).
    MatmulTarget(29, 1024, 1024, 1024, "float32", "none"),
    MatmulTarget(30, 2048, 2048, 2048, "float32", "bias"),
    MatmulTarget(31, 4096, 4096, 2048, "float32", "relu"),
    MatmulTarget(32, 4096, 4096, 4096, "float32", "bias_residual_gelu"),
)

MATMUL_BACKENDS: tuple[MatmulBackend, ...] = (
    MatmulBackend("aten", "ATen"),
    MatmulBackend("quack-direct", "Quack-direct"),
    MatmulBackend("helion-cute", "Helion-CuTe"),
)

# Curated list of CuTe-capable example tests. The prompt says "Start
# from ``test/test_examples.py`` and ``benchmarks/run.py``'s example
# map; target at least 20 CuTe-capable examples". Selection rules:
#   * Use only tests that go through ``check_example`` / ``code_and_output``
#     / ``output_only`` — the plugin can only patch those entry points,
#     so tests that build a ``helion.Config(...)`` and call the kernel
#     directly (``test_matmul_bwd``, ``test_addmm_bwd``,
#     ``test_split_k_barrier_accuracy``) are intentionally excluded;
#     they would pass with their forced config rather than autotune.
#   * Skip ``@xfailIfCute``-marked tests: ``xfailIfCute`` does *not*
#     skip execution, so those tests would still run autotune and spend
#     wall-clock on paths the plan already classifies as unsupported.
#   * Prefer matmul variants (bmm, broadcast, layernorm-fused, split-k),
#     fused-epilogue paths, stable non-matmul kernels (softmax,
#     reductions, norm, cross-entropy, embedding, attention), and a
#     handful of harder tests (jagged, grouped GEMM) that stress CuTe
#     lowering corners.
SWEEP_NODE_IDS: tuple[str, ...] = (
    # Pure matmul variants — the load-bearing acceptance row.
    "test/test_examples.py::TestExamples::test_matmul_default",
    "test/test_examples.py::TestExamples::test_matmul",
    "test/test_examples.py::TestExamples::test_matmul_split_k",
    "test/test_examples.py::TestExamples::test_bmm",
    "test/test_examples.py::TestExamples::test_bmm_non_divisible_k",
    "test/test_examples.py::TestExamples::test_broadcast_matmul",
    "test/test_examples.py::TestExamples::test_matmul_layernorm_static_shapes",
    "test/test_examples.py::TestExamples::test_matmul_layernorm_half_dtype_multi_k_tile",
    # Element-wise / reduction kernels — stable CuTe targets.
    "test/test_examples.py::TestExamples::test_add",
    "test/test_examples.py::TestExamples::test_add_loop_order",
    "test/test_examples.py::TestExamples::test_softmax",
    "test/test_examples.py::TestExamples::test_softmax_two_pass",
    "test/test_examples.py::TestExamples::test_cross_entropy",
    "test/test_examples.py::TestExamples::test_rms_norm_fwd",
    "test/test_examples.py::TestExamples::test_layernorm_with_bias",
    "test/test_examples.py::TestExamples::test_layernorm_no_bias",
    "test/test_examples.py::TestExamples::test_sum",
    "test/test_examples.py::TestExamples::test_long_sum",
    "test/test_examples.py::TestExamples::test_exp_fwd",
    "test/test_examples.py::TestExamples::test_concat",
    "test/test_examples.py::TestExamples::test_embedding_pointers",
    # Activations / fused-epilogue chains.
    "test/test_examples.py::TestExamples::test_geglu",
    "test/test_examples.py::TestExamples::test_swiglu",
    "test/test_examples.py::TestExamples::test_jsd",
    "test/test_examples.py::TestExamples::test_kl_div",
    # Harder / corner-case kernels.
    "test/test_examples.py::TestExamples::test_attention_pointer",
    "test/test_examples.py::TestExamples::test_jagged_dense_add",
    "test/test_examples.py::TestExamples::test_jagged_mean",
    "test/test_examples.py::TestExamples::test_grouped_gemm_jagged",
)


# Non-fp32 coverage sub-sweep. Curated list of CuTe-capable nodes whose
# matmul-shaped or attention-shaped inputs are bf16 / fp16. Distinct
# from ``SWEEP_NODE_IDS`` (which is fp32-dominated) so future cycles
# can ask "does the non-fp32 autotune path still pass cleanly across
# the example suite?" without re-running the full default list.
#
# Selection rules:
#   * Tests must use ``check_example`` / ``code_and_output`` /
#     ``output_only`` so the plugin's autotune patch fires — same
#     constraint as ``SWEEP_NODE_IDS`` above.
#   * Tests must not be ``@xfailIfCute`` / ``@skipIfCute`` /
#     ``@skipIfFn(... cute)`` for the same reason as the default
#     list.
#
# ``test_matmul_bf16_tcgen05`` is the bf16 fixture added so this
# list actually fires the ``uses_tcgen05`` codegen marker. The other
# fixtures in the list use small shapes (e.g.
# ``test_bmm_non_divisible_k`` = 4x128x384x128) that do not pass
# ``matmul_ops.enforce_dot_requirements`` (M divisible by 64 and
# M >= 64 after ``update_min_block``, etc.), so they still record
# zero tcgen05 marker hits.
SWEEP_NODE_IDS_NONFP32: tuple[str, ...] = (
    # 256^3 bf16: sized just above the tcgen05 admission floor;
    # reliably fires ``uses_tcgen05`` under autotune.
    "test/test_examples.py::TestExamples::test_matmul_bf16_tcgen05",
    # Half-precision matmul / GEMM variants. ``test_bmm`` and
    # ``test_bmm_non_divisible_k`` appear in the default list too;
    # listing them here documents intent.
    "test/test_examples.py::TestExamples::test_bmm",
    "test/test_examples.py::TestExamples::test_bmm_non_divisible_k",
    "test/test_examples.py::TestExamples::test_moe_matmul_ogs",
    "test/test_examples.py::TestExamples::test_grouped_gemm_jagged",
    # Half-precision attention.
    "test/test_examples.py::TestExamples::test_jagged_hstu_attn",
)

NODE_LIST_BY_NAME: dict[str, tuple[str, ...]] = {
    "default": SWEEP_NODE_IDS,
    "nonfp32": SWEEP_NODE_IDS_NONFP32,
}


def _parse_matmul_gpus(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("expected at least one GPU ID")
    gpus: list[int] = []
    for token in tokens:
        try:
            gpu = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated GPU IDs, got {value!r}"
            ) from exc
        if gpu not in (6, 7):
            raise argparse.ArgumentTypeError(
                "matmul target sweep may only use physical GPUs 6 and 7"
            )
        if gpu in gpus:
            raise argparse.ArgumentTypeError(f"duplicate GPU ID {gpu}")
        gpus.append(gpu)
    return tuple(gpus)


def _parse_matmul_target_ids(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("expected at least one target ID")
    valid_ids = {target.target_id for target in MATMUL_TARGETS}
    target_ids: list[int] = []
    for token in tokens:
        try:
            target_id = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated target IDs, got {value!r}"
            ) from exc
        if target_id not in valid_ids:
            raise argparse.ArgumentTypeError(
                f"matmul target ID must be one of {sorted(valid_ids)}"
            )
        if target_id in target_ids:
            raise argparse.ArgumentTypeError(f"duplicate target ID {target_id}")
        target_ids.append(target_id)
    return tuple(target_ids)


def _iter_matmul_work_items(
    target_ids: tuple[int, ...] | None = None,
) -> tuple[MatmulWorkItem, ...]:
    target_filter = set(target_ids) if target_ids is not None else None
    return tuple(
        MatmulWorkItem(target, backend)
        for target in MATMUL_TARGETS
        if target_filter is None or target.target_id in target_filter
        for backend in MATMUL_BACKENDS
    )


def _format_matmul_work_item(item: MatmulWorkItem) -> str:
    return (
        f"target={item.target.target_id} shape={item.target.shape} "
        f"dtype={item.target.dtype} epilogue={item.target.epilogue} "
        f"backend={item.backend.label} impl={item.backend.impl}"
    )


def _print_matmul_worklist(work_items: tuple[MatmulWorkItem, ...]) -> None:
    for item in work_items:
        print(_format_matmul_work_item(item))
    target_count = len({item.target.target_id for item in work_items})
    print(f"# {target_count} targets x {len(MATMUL_BACKENDS)} backends")
    print(f"# {len(work_items)} matmul target/backend work items")


def _build_matmul_compare_cmd(
    item: MatmulWorkItem,
    *,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    seed: int,
    skip_correctness: int,
    json_output: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "cute" / "compare_matmul_backends.py"),
        "--impl",
        item.backend.impl,
        "--m",
        str(item.target.m),
        "--n",
        str(item.target.n),
        "--k",
        str(item.target.k),
        "--epilogue",
        item.target.epilogue,
        "--dtype",
        item.target.dtype,
        "--num-runs",
        str(num_runs),
        "--warmup-ms",
        str(warmup_ms),
        "--rep-ms",
        str(rep_ms),
        "--seed",
        str(seed),
        "--skip-correctness",
        str(skip_correctness),
    ]
    if item.backend.impl == "quack-direct":
        cmd.extend(["--quack-max-swizzle-size", "8", "--quack-tune", "brief"])
    if item.backend.impl == "helion-cute":
        cmd.extend(
            ["--helion-require-tcgen05", "1" if item.target.require_tcgen05 else "0"]
        )
    elif item.backend.impl == "helion-triton":
        cmd.extend(["--helion-require-tcgen05", "0"])
    if json_output:
        cmd.append("--json")
    return cmd


def _format_shell_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def _print_matmul_dry_run(
    work_items: tuple[MatmulWorkItem, ...],
    *,
    gpus: tuple[int, ...],
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    seed: int,
    skip_correctness: int,
) -> None:
    for index, item in enumerate(work_items):
        gpu = gpus[index % len(gpus)]
        cmd = _build_matmul_compare_cmd(
            item,
            num_runs=num_runs,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            seed=seed,
            skip_correctness=skip_correctness,
            json_output=True,
        )
        print(f"CUDA_VISIBLE_DEVICES={gpu} {_format_shell_command(cmd)}")
    print(f"# {len(work_items)} commands")


def _matmul_result_cell(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    if record.get("timed_out"):
        return "TIMEOUT"
    returncode = record.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        return "FAIL"
    if record.get("json_parse_failed"):
        return "NOJSON"
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return "-"
    value = payload.get("mom_median_tflops", payload.get("median_tflops"))
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.1f}"


def _matmul_hard_failure(record: dict[str, Any]) -> bool:
    return bool(
        record.get("timed_out")
        or record.get("json_parse_failed")
        or record.get("returncode") != 0
    )


def _matmul_targets_for_results(
    results: list[dict[str, Any]],
) -> tuple[MatmulTarget, ...]:
    target_ids = {
        record["target_id"]
        for record in results
        if isinstance(record.get("target_id"), int)
    }
    if not target_ids:
        return MATMUL_TARGETS
    return tuple(target for target in MATMUL_TARGETS if target.target_id in target_ids)


def _format_matmul_markdown_table(results: list[dict[str, Any]]) -> str:
    records_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for record in results:
        target_id = record.get("target_id")
        backend = record.get("backend")
        if isinstance(target_id, int) and isinstance(backend, str):
            records_by_key[(target_id, backend)] = record

    backend_labels = [backend.label for backend in MATMUL_BACKENDS]
    lines = [
        "| Target | Shape | Dtype | Epilogue | " + " | ".join(backend_labels) + " |",
        "|---:|---|---|---|" + "|".join("---:" for _ in MATMUL_BACKENDS) + "|",
    ]
    for target in _matmul_targets_for_results(results):
        cells = [
            _matmul_result_cell(records_by_key.get((target.target_id, backend.impl)))
            for backend in MATMUL_BACKENDS
        ]
        lines.append(
            f"| {target.target_id} | {target.shape} | {target.dtype} | "
            f"{target.epilogue} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def _format_matmul_quack_configs(results: list[dict[str, Any]]) -> str:
    records_by_target: dict[int, dict[str, Any]] = {}
    for record in results:
        if record.get("backend") != "quack-direct":
            continue
        target_id = record.get("target_id")
        payload = record.get("payload")
        if not isinstance(target_id, int) or not isinstance(payload, dict):
            continue
        config = payload.get("config")
        if isinstance(config, dict):
            records_by_target[target_id] = config
    if not records_by_target:
        return ""
    lines = ["", "## Quack Configs", "", "| Target | Config |", "|---:|---|"]
    for target in MATMUL_TARGETS:
        config = records_by_target.get(target.target_id)
        if config is None:
            continue
        lines.append(f"| {target.target_id} | `{json.dumps(config, sort_keys=True)}` |")
    return "\n".join(lines) + "\n"


def _format_matmul_quack_tuning(results: list[dict[str, Any]]) -> str:
    records_by_target: dict[int, list[Any]] = {}
    for record in results:
        if record.get("backend") != "quack-direct":
            continue
        target_id = record.get("target_id")
        payload = record.get("payload")
        if not isinstance(target_id, int) or not isinstance(payload, dict):
            continue
        tuning = payload.get("quack_tuning")
        if isinstance(tuning, list):
            records_by_target[target_id] = tuning
    if not records_by_target:
        return ""

    lines = [
        "",
        "## Quack Tuning",
        "",
        "| Target | Candidate | mom-median TFLOP/s | Config |",
        "|---:|---:|---:|---|",
    ]
    for target in MATMUL_TARGETS:
        tuning = records_by_target.get(target.target_id)
        if tuning is None:
            continue
        for index, entry in enumerate(tuning, start=1):
            if not isinstance(entry, dict):
                continue
            config = entry.get("config")
            tflops = entry.get("mom_median_tflops")
            if not isinstance(config, dict) or not isinstance(tflops, (int, float)):
                continue
            lines.append(
                f"| {target.target_id} | {index} | {tflops:.1f} | "
                f"`{json.dumps(config, sort_keys=True)}` |"
            )
    if len(lines) == 5:
        return ""
    return "\n".join(lines) + "\n"


def _format_matmul_markdown_summary(results: list[dict[str, Any]]) -> str:
    return (
        _format_matmul_markdown_table(results)
        + _format_matmul_quack_configs(results)
        + _format_matmul_quack_tuning(results)
    )


def _parse_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


_PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def _now_iso() -> str:
    """ISO timestamp in Pacific Time (auto-DST). All sweep logs use this."""
    return datetime.now(_PACIFIC_TZ).isoformat(timespec="seconds")


def _run_id_from_timestamp(timestamp: str) -> str:
    return timestamp.replace("+00:00", "Z").replace(":", "").replace("-", "")


def _run_matmul_work_item(
    item: MatmulWorkItem,
    *,
    gpu: int,
    timeout_seconds: float,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    seed: int,
    skip_correctness: int,
) -> dict[str, Any]:
    cmd = _build_matmul_compare_cmd(
        item,
        num_runs=num_runs,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        seed=seed,
        skip_correctness=skip_correctness,
        json_output=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started_at = _now_iso()
    started = time.monotonic()
    timed_out = False
    returncode: int | None
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = (
            (exc.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            (exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
    elapsed = time.monotonic() - started
    finished_at = _now_iso()
    payload = None if timed_out or returncode != 0 else _parse_json_from_stdout(stdout)
    json_parse_failed = not timed_out and returncode == 0 and payload is None
    return {
        "target_id": item.target.target_id,
        "shape": item.target.shape,
        "dtype": item.target.dtype,
        "epilogue": item.target.epilogue,
        "backend": item.backend.impl,
        "backend_label": item.backend.label,
        "gpu": gpu,
        "command": cmd,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "json_parse_failed": json_parse_failed,
        "payload": payload,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def _ordered_matmul_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_key = {
        (record.get("target_id"), record.get("backend")): record for record in results
    }
    ordered: list[dict[str, Any]] = []
    for target in MATMUL_TARGETS:
        for backend in MATMUL_BACKENDS:
            record = records_by_key.get((target.target_id, backend.impl))
            if record is not None:
                ordered.append(record)
    return ordered


def _payload_float(record: dict[str, Any], key: str) -> float | None:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _matmul_quack_tflops_by_target(
    results: list[dict[str, Any]],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for record in results:
        if record.get("backend") != "quack-direct":
            continue
        tflops = _payload_float(record, "mom_median_tflops")
        if tflops is not None:
            out[int(record["target_id"])] = tflops
    return out


def _append_matmul_performance_csv(
    results: list[dict[str, Any]],
    *,
    csv_path: Path,
    run_id: str,
    run_started_at: str,
    run_finished_at: str,
    output_path: Path | None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    quack_tflops_by_target = _matmul_quack_tflops_by_target(results)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATMUL_PERFORMANCE_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for record in _ordered_matmul_results(results):
            target_id = int(record["target_id"])
            tflops = _payload_float(record, "mom_median_tflops")
            mom_ms = _payload_float(record, "mom_median_ms")
            best_tflops = _payload_float(record, "best_tflops")
            best_ms = _payload_float(record, "best_ms")
            quack_tflops = quack_tflops_by_target.get(target_id)
            gap_vs_quack: float | None = None
            passes_5pct = ""
            if (
                record.get("backend") == "helion-cute"
                and tflops is not None
                and quack_tflops is not None
            ):
                gap_vs_quack = (quack_tflops - tflops) / quack_tflops * 100.0
                passes_5pct = str(gap_vs_quack <= 5.0)
            writer.writerow(
                {
                    "timestamp": record.get("finished_at", run_finished_at),
                    "run_id": run_id,
                    "run_started_at": run_started_at,
                    "run_finished_at": run_finished_at,
                    "target": target_id,
                    "shape": record.get("shape", ""),
                    "m": record.get("shape", "").split("x")[0],
                    "n": record.get("shape", "").split("x")[1],
                    "k": record.get("shape", "").split("x")[2],
                    "dtype": record.get("dtype", ""),
                    "epilogue": record.get("epilogue", ""),
                    "backend": record.get("backend", ""),
                    "backend_label": record.get("backend_label", ""),
                    "gpu": record.get("gpu", ""),
                    "tflops": "" if tflops is None else f"{tflops:.6f}",
                    "mom_median_ms": "" if mom_ms is None else f"{mom_ms:.9f}",
                    "best_tflops": "" if best_tflops is None else f"{best_tflops:.6f}",
                    "best_ms": "" if best_ms is None else f"{best_ms:.9f}",
                    "quack_tflops": ""
                    if quack_tflops is None
                    else f"{quack_tflops:.6f}",
                    "gap_vs_quack_pct": ""
                    if gap_vs_quack is None
                    else f"{gap_vs_quack:.6f}",
                    "passes_5pct": passes_5pct,
                    "returncode": ""
                    if record.get("returncode") is None
                    else record["returncode"],
                    "timed_out": record.get("timed_out", ""),
                    "json_parse_failed": record.get("json_parse_failed", ""),
                    "wall_clock_seconds": f"{record.get('wall_clock_seconds', 0):.3f}",
                    "output_path": "" if output_path is None else str(output_path),
                }
            )


def _compact_matmul_result(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    compact: dict[str, Any] = {
        "target_id": record.get("target_id"),
        "shape": record.get("shape"),
        "dtype": record.get("dtype"),
        "epilogue": record.get("epilogue"),
        "backend": record.get("backend"),
        "backend_label": record.get("backend_label"),
        "gpu": record.get("gpu"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "wall_clock_seconds": record.get("wall_clock_seconds"),
        "returncode": record.get("returncode"),
        "timed_out": record.get("timed_out"),
        "json_parse_failed": record.get("json_parse_failed"),
        "payload": payload,
    }
    if _matmul_hard_failure(record):
        compact["stdout_tail"] = record.get("stdout_tail")
        compact["stderr_tail"] = record.get("stderr_tail")
    return compact


def _append_matmul_run_jsonl(
    results: list[dict[str, Any]],
    *,
    jsonl_path: Path,
    run_id: str,
    run_started_at: str,
    run_finished_at: str,
    run_wall_clock_seconds: float,
    output_path: Path | None,
    args: argparse.Namespace,
    exit_code: int,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "run_wall_clock_seconds": run_wall_clock_seconds,
        "exit_code": exit_code,
        "output_path": None if output_path is None else str(output_path),
        "performance_csv_path": str(args.append_performance_csv)
        if args.append_performance_csv
        else None,
        "gpus": list(args.gpus),
        "timeout_seconds": args.timeout,
        "num_runs": args.num_runs,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "seed": args.seed,
        "skip_correctness": args.skip_correctness,
        "results": [
            _compact_matmul_result(r) for r in _ordered_matmul_results(results)
        ],
    }
    with jsonl_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _run_matmul_target_sweep(args: argparse.Namespace) -> int:
    work_items = _iter_matmul_work_items(args.matmul_targets)
    if args.list:
        _print_matmul_worklist(work_items)
        return 0
    if args.summary_only:
        print(
            "--summary-only is only implemented for the pytest-node sweep.",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        _print_matmul_dry_run(
            work_items,
            gpus=args.gpus,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            seed=args.seed,
            skip_correctness=args.skip_correctness,
        )
        return 0

    output_path: Path | None = None
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    run_started_at = _now_iso()
    run_started = time.monotonic()
    run_id = args.run_id or _run_id_from_timestamp(run_started_at)
    print(
        f"[cute_autotune_sweep] running {len(work_items)} matmul measurements; "
        f"gpus={','.join(str(gpu) for gpu in args.gpus)}; "
        f"timeout={args.timeout}s; output={output_path}; run_id={run_id}",
        flush=True,
    )

    next_index = 0
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        in_flight: dict[Any, int] = {}

        def submit_next(gpu: int) -> None:
            nonlocal next_index
            if next_index >= len(work_items):
                return
            item = work_items[next_index]
            next_index += 1
            print(
                f"[{next_index}/{len(work_items)}] gpu={gpu} "
                f"{_format_matmul_work_item(item)}",
                flush=True,
            )
            future = executor.submit(
                _run_matmul_work_item,
                item,
                gpu=gpu,
                timeout_seconds=args.timeout,
                num_runs=args.num_runs,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                seed=args.seed,
                skip_correctness=args.skip_correctness,
            )
            in_flight[future] = gpu

        for gpu in args.gpus:
            submit_next(gpu)

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                gpu = in_flight.pop(future)
                result = future.result()
                results.append(result)
                status = (
                    "timeout" if result["timed_out"] else f"rc={result['returncode']}"
                )
                if result["json_parse_failed"]:
                    status = "missing-json"
                print(
                    f"[done] gpu={gpu} target={result['target_id']} "
                    f"backend={result['backend']} {status}",
                    flush=True,
                )
                submit_next(gpu)

    if output_path is not None:
        output_path.write_text(_format_matmul_markdown_summary(results))
    run_finished_at = _now_iso()
    run_wall_clock_seconds = time.monotonic() - run_started
    exit_code = 2 if any(_matmul_hard_failure(result) for result in results) else 0
    if args.append_performance_csv:
        csv_path = Path(args.append_performance_csv).resolve()
        _append_matmul_performance_csv(
            results,
            csv_path=csv_path,
            run_id=run_id,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            output_path=output_path,
        )
        print(f"[cute_autotune_sweep] appended {csv_path}", flush=True)
    if args.append_run_jsonl:
        jsonl_path = Path(args.append_run_jsonl).resolve()
        _append_matmul_run_jsonl(
            results,
            jsonl_path=jsonl_path,
            run_id=run_id,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            run_wall_clock_seconds=run_wall_clock_seconds,
            output_path=output_path,
            args=args,
            exit_code=exit_code,
        )
        print(f"[cute_autotune_sweep] appended {jsonl_path}", flush=True)
    if output_path is not None:
        print(f"[cute_autotune_sweep] wrote {output_path}", flush=True)
    return exit_code


def _filter_node_ids(
    node_ids: tuple[str, ...], substrings: list[str]
) -> tuple[str, ...]:
    if not substrings:
        return node_ids
    out: list[str] = []
    for nid in node_ids:
        if any(s in nid for s in substrings):
            out.append(nid)
    return tuple(out)


def _run_one(
    node_id: str,
    output_path: Path,
    timeout_seconds: float,
    backend: str,
    extra_pytest_args: list[str],
    autotune_budget_seconds: int | None,
    autotune_max_generations: int | None,
) -> dict[str, object]:
    """Run a single test in a fresh subprocess. Returns a result-summary dict."""
    env = os.environ.copy()
    env["HELION_BACKEND"] = backend
    env["HELION_AUTOTUNE_SWEEP_RESULT_JSON"] = str(output_path)
    # Budget caps the generation loop but not the initial-population
    # benchmark; see the active G1 autotuner-reliability notes in
    # ``cute_plan.md``.
    if autotune_budget_seconds is not None:
        env["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(autotune_budget_seconds)
    if autotune_max_generations is not None:
        env["HELION_AUTOTUNE_MAX_GENERATIONS"] = str(autotune_max_generations)
    # ``CUDA_VISIBLE_DEVICES`` is intentionally inherited from the caller
    # per the GPU policy — the harness must not unset or broaden it.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        PLUGIN_MODULE,
        "--no-header",
        "-q",
        "-rN",
        "--tb=line",
        node_id,
        *extra_pytest_args,
    ]
    started = time.monotonic()
    timed_out = False
    returncode: int | None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # ``None`` sentinel (not ``-1``) keeps the summary's "died on
        # signal / abort" tally mutually exclusive from the "timeouts"
        # tally — see ``_print_summary``.
        returncode = None
        stdout = (
            (exc.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            (exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
    elapsed = time.monotonic() - started
    return {
        "nodeid": node_id,
        "wall_clock_seconds": elapsed,
        "subprocess_returncode": returncode,
        "subprocess_timed_out": timed_out,
        "subprocess_stdout_tail": stdout[-2000:],
        "subprocess_stderr_tail": stderr[-2000:],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path. Pytest-node sweeps require this and write JSONL. "
            "Matmul target sweeps may omit this; if provided, an optional "
            "Markdown summary table is written. The canonical matmul sweep "
            "writes only to --append-performance-csv and --append-run-jsonl."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help=(
            "Per-test wall-clock timeout in seconds (default: 900). The "
            "matmul-shaped autotune paths can run >10 min per node on "
            "B200 when many candidates compile cleanly — for an "
            "exhaustive sweep that disambiguates 'budget-bound' from "
            "'real hang' on the matmul-shaped paths, pass "
            "``--timeout 1800``. The active matmul target sweep in "
            "``cute_plan.md`` uses its own 3600 s target budget."
        ),
    )
    parser.add_argument(
        "--backend",
        default="cute",
        help="HELION_BACKEND value (default: cute).",
    )
    parser.add_argument(
        "--matmul-target-sweep",
        action="store_true",
        help=(
            "Run/list the active matmul benchmark sweep instead of the "
            "pytest-node autotune sweep."
        ),
    )
    parser.add_argument(
        "--matmul-targets",
        type=_parse_matmul_target_ids,
        help=(
            "Comma-separated target IDs for --matmul-target-sweep. Omit for "
            "the full 8-target sweep; pass a single ID for frequent targeted "
            "active-target rebenchmarks."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=_parse_matmul_gpus,
        default=_parse_matmul_gpus("6,7"),
        help=(
            "Comma-separated physical GPU IDs allowed for --matmul-target-sweep. "
            "Only 6 and 7 are accepted; default: 6,7."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "With --matmul-target-sweep, print the exact benchmark commands "
            "without launching GPU work."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Matmul target sweep do_bench invocation count (default: 5).",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=1000,
        help="Matmul target sweep do_bench warmup in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=500,
        help="Matmul target sweep do_bench rep window in milliseconds (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Matmul target sweep random seed (default: 0).",
    )
    parser.add_argument(
        "--skip-correctness",
        type=int,
        choices=(0, 1),
        default=0,
        help="Forwarded to compare_matmul_backends.py for matmul target sweeps.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional identifier for a matmul target sweep run. Defaults to the "
            "Pacific-Time run-start timestamp. Included in structured append logs."
        ),
    )
    parser.add_argument(
        "--append-performance-csv",
        default=".logs/performance_by_target.csv",
        help=(
            "With --matmul-target-sweep, append one structured row per "
            "target/backend measurement to this CSV path "
            "(default: .logs/performance_by_target.csv)."
        ),
    )
    parser.add_argument(
        "--append-run-jsonl",
        default=".logs/benchmark_runs.jsonl",
        help=(
            "With --matmul-target-sweep, append one JSON object per full sweep "
            "run to this JSONL path, including run metadata and compact results "
            "(default: .logs/benchmark_runs.jsonl)."
        ),
    )
    parser.add_argument(
        "--list-name",
        default="default",
        choices=sorted(NODE_LIST_BY_NAME.keys()),
        help=(
            "Curated node-ID list to run. ``default`` is the broad "
            "29-node CuTe-capable list; ``nonfp32`` is a smaller "
            "bf16/fp16 coverage list that now includes "
            "``test_matmul_bf16_tcgen05`` (256^3 bf16) as the first "
            "fixture above the tcgen05 admission floor; the other "
            "nodes in the list use small shapes and still record zero "
            "tcgen05 marker hits — see the list's inline comment."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=(
            "Run only node IDs that contain this substring. Repeatable; "
            "if any substring matches, the node ID is kept."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the (filtered) node ID list and exit.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Read the JSONL file pointed at by ``--output`` and re-print the "
            "summary without running anything. The file is consumed read-only "
            "in this mode; nothing is appended or overwritten."
        ),
    )
    parser.add_argument(
        "--extra-pytest-arg",
        action="append",
        default=[],
        help="Extra args appended to each pytest invocation (repeatable).",
    )
    parser.add_argument(
        "--autotune-budget-seconds",
        type=int,
        default=None,
        help=(
            "If set, exports ``HELION_AUTOTUNE_BUDGET_SECONDS=<value>`` to "
            "each test subprocess. Caps the autotune generation loop's "
            "wall-clock. Combined with ``--autotune-max-generations 0``, "
            "this rescues ``test_bmm`` (non-fp32 list). "
            "``test_matmul_default`` and ``test_broadcast_matmul`` still "
            "time out because the initial-population benchmark phase is "
            "not budget-aware — see the active G1 autotuner-reliability "
            "notes in ``cute_plan.md`` for the deeper fix."
        ),
    )
    parser.add_argument(
        "--autotune-max-generations",
        type=int,
        default=None,
        help=(
            "If set, exports ``HELION_AUTOTUNE_MAX_GENERATIONS=<value>`` to "
            "each test subprocess. ``0`` runs only the initial population + "
            "rebenchmark + finishing phase. Combined with "
            "``--autotune-budget-seconds``, this rescues ``test_bmm`` only; "
            "``test_matmul_default`` and ``test_broadcast_matmul`` still "
            "time out — see the active G1 autotuner-reliability notes "
            "in ``cute_plan.md``."
        ),
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.matmul_target_sweep:
        return _run_matmul_target_sweep(args)

    if not args.output:
        print(
            "--output is required for the pytest-node sweep.",
            file=sys.stderr,
        )
        return 1
    output_path = Path(args.output).resolve()
    selected_list = NODE_LIST_BY_NAME[args.list_name]
    node_ids = _filter_node_ids(selected_list, args.filter)

    if args.list:
        for nid in node_ids:
            print(nid)
        print(f"# {len(node_ids)} test node IDs")
        return 0

    if args.summary_only:
        return _print_summary(output_path)

    if not node_ids:
        print("No test node IDs match the requested filters.", file=sys.stderr)
        return 1

    # Fresh JSONL output — both the per-test plugin records and the
    # per-test subprocess wrapper records below are appended.
    if output_path.exists():
        output_path.unlink()
    output_path.touch()

    print(
        f"[cute_autotune_sweep] running {len(node_ids)} tests from "
        f"list={args.list_name!r}; timeout={args.timeout}s; "
        f"backend={args.backend}; output={output_path}",
        flush=True,
    )

    summary_records: list[dict[str, object]] = []
    for i, node_id in enumerate(node_ids, start=1):
        print(f"[{i}/{len(node_ids)}] {node_id}", flush=True)
        result = _run_one(
            node_id,
            output_path,
            timeout_seconds=args.timeout,
            backend=args.backend,
            extra_pytest_args=list(args.extra_pytest_arg),
            autotune_budget_seconds=args.autotune_budget_seconds,
            autotune_max_generations=args.autotune_max_generations,
        )
        summary_records.append(result)
        # Write a wrapper record so the JSONL can correlate
        # subprocess-level outcomes (timeouts, segfaults) with the
        # plugin's per-test record (which only fires for tests that
        # actually reach pytest's ``call`` phase).
        with output_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "kind": "subprocess_summary",
                        **result,
                        # Trim the embedded stdout / stderr a bit for
                        # the inline summary; the per-test record above
                        # carries its own tail.
                        # pyrefly: ignore [bad-index]
                        "subprocess_stdout_tail": result["subprocess_stdout_tail"][
                            -512:
                        ],
                        # pyrefly: ignore [bad-index]
                        "subprocess_stderr_tail": result["subprocess_stderr_tail"][
                            -512:
                        ],
                    }
                )
                + "\n"
            )

    return _print_summary(output_path)


def _print_summary(output_path: Path) -> int:
    if not output_path.exists():
        print(f"No sweep output at {output_path}", file=sys.stderr)
        return 1
    print(f"\n[cute_autotune_sweep] summary from {output_path}\n")
    test_records: list[dict[str, object]] = []
    subprocess_records: list[dict[str, object]] = []
    for raw_line in output_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        rec = json.loads(raw_line)
        if rec.get("kind") == "subprocess_summary":
            subprocess_records.append(rec)
        else:
            test_records.append(rec)

    outcomes: Counter[str] = Counter()
    timeouts = 0
    seg_or_signal = 0
    nonzero_rc_no_call_record: list[dict[str, object]] = []
    test_record_nodeids = {str(r.get("nodeid")) for r in test_records}
    for sub in subprocess_records:
        if sub.get("subprocess_timed_out"):
            timeouts += 1
            continue
        rc = sub.get("subprocess_returncode")
        if isinstance(rc, int) and rc < 0:
            seg_or_signal += 1
            continue
        # Pytest exits non-zero on collection failures, import errors,
        # or fixture setup failures that never reach a per-test call
        # record. Treat those as harness failures so the sweep does not
        # silently exit 0 when nothing actually ran.
        if (
            isinstance(rc, int)
            and rc != 0
            and str(sub.get("nodeid")) not in test_record_nodeids
        ):
            nonzero_rc_no_call_record.append(sub)
    for rec in test_records:
        outcomes[str(rec.get("outcome", "unknown"))] += 1

    print(f"  tests reaching pytest call phase: {len(test_records)}")
    for k, v in sorted(outcomes.items()):
        print(f"    {k}: {v}")
    print(f"  subprocess timeouts: {timeouts}")
    print(f"  subprocess died on signal / abort: {seg_or_signal}")
    print(
        f"  subprocess nonzero exit without call record: "
        f"{len(nonzero_rc_no_call_record)}"
    )
    print()

    failures = [
        r for r in test_records if r.get("outcome") not in ("passed", "skipped")
    ]
    if failures:
        print("Failures (nodeid -> outcome / autotune_seconds):")
        for rec in failures:
            print(
                f"  {rec.get('nodeid')} :: {rec.get('outcome')} "
                f"(autotune_seconds={rec.get('autotune_seconds')})"
            )
        print()
    timed_out_subs = [r for r in subprocess_records if r.get("subprocess_timed_out")]
    if timed_out_subs:
        print("Subprocess timeouts (nodeid -> wall_clock_seconds):")
        for rec in timed_out_subs:
            print(f"  {rec.get('nodeid')} :: {rec.get('wall_clock_seconds'):.1f}s")
        print()
    if nonzero_rc_no_call_record:
        print(
            "Subprocess nonzero exit without per-test call record "
            "(collection / import / setup failure):"
        )
        for rec in nonzero_rc_no_call_record:
            print(
                f"  {rec.get('nodeid')} :: rc={rec.get('subprocess_returncode')} "
                f"wall={rec.get('wall_clock_seconds'):.1f}s"
            )
        print()

    # Tests where autotune ran and the canonical CuTe codegen markers
    # fired — useful sanity that the sweep actually exercised the
    # tcgen05 path on matmul-shaped examples.
    tcgen05_hits = [
        r
        for r in test_records
        if isinstance(r.get("codegen_markers"), dict)
        and bool(r["codegen_markers"].get("uses_tcgen05"))  # type: ignore[union-attr]
    ]
    print(f"  tcgen05 codegen marker hits: {len(tcgen05_hits)}")
    if failures or timeouts or seg_or_signal or nonzero_rc_no_call_record:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
