from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib
import json
import math
import operator
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from typing import cast

from benchmarks.cute import compare_attention_backends
import pytest
import torch

from helion._testing import skipUnlessCuteAvailable
import helion.autotuner.benchmarking as benchmarking
import helion.autotuner.metrics as autotune_metrics
from helion.runtime.settings import default_autotuner_fn


@pytest.fixture(autouse=True)
def _clear_process_local_cute_library_override(monkeypatch):
    # CuTe initialization may populate this process-local locator. Strict benchmark
    # tests must begin without an ambient codegen override, including under xdist.
    monkeypatch.delenv("CUTE_DSL_LIBS", raising=False)


def test_mirrored_bench_generic_rotates_balanced_pairs(monkeypatch) -> None:
    calls: list[int] = []
    cleanups: list[int] = []
    clock = [0.0]

    def perf_counter() -> float:
        return clock[0]

    def call(index: int) -> None:
        calls.append(index)
        clock[0] += 0.001

    def after_call(index: int) -> None:
        cleanups.append(index)
        clock[0] += 1.0

    monkeypatch.setattr(benchmarking, "_make_l2_cache_clearer", lambda: lambda: None)
    monkeypatch.setattr(benchmarking, "synchronize_device", lambda: None)
    monkeypatch.setattr(benchmarking.time, "perf_counter", perf_counter)
    trace = benchmarking.mirrored_bench_generic(
        [lambda index=index: call(index) for index in range(3)],
        repeat=4,
        after_call=after_call,
    )

    assert trace.orders == [
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
    ]
    assert calls == [0, 1, 2, *[index for order in trace.orders for index in order]]
    assert cleanups == calls
    assert all(times == pytest.approx([1.0] * 3) for times in trace.elapsed_ms)
    assert trace.medians_ms == pytest.approx([1.0] * 3)
    assert trace.sweep_count == 4
    assert trace.calls_per_sample == 1
    assert trace.total_calls == 4
    assert trace.target_ms is None
    assert trace.repeat_reference_perf_ms is None


def test_mirrored_bench_generic_bounds_fast_kernel_trace(monkeypatch) -> None:
    calls = 0
    clock = [0.0]

    def fn() -> None:
        nonlocal calls
        calls += 1
        clock[0] += 0.000_001

    monkeypatch.setattr(benchmarking, "_make_l2_cache_clearer", lambda: lambda: None)
    monkeypatch.setattr(benchmarking, "synchronize_device", lambda: None)
    monkeypatch.setattr(benchmarking.time, "perf_counter", lambda: clock[0])

    trace = benchmarking.mirrored_bench_generic([fn], repeat=20_000)

    assert trace.sweep_count == 64
    assert trace.calls_per_sample == 313
    assert trace.total_calls == 20_032
    assert len(trace.orders) == len(trace.elapsed_ms) == trace.sweep_count
    assert calls == 1 + trace.total_calls
    assert trace.medians_ms == pytest.approx([0.001])


class _FakeStream:
    def wait_stream(self, stream):
        self.waited_stream = stream


class _FakeStreamContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGraphContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeCuteGraphContext:
    def __init__(self, cuda):
        self.cuda = cuda

    def __enter__(self):
        return self.cuda.CUDAGraph()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


class _FakeCuda:
    def __init__(self, *, available=True, capturing=False):
        self.available = available
        self.capturing = capturing
        self.current_stream_obj = _FakeStream()
        self.graph_obj = None
        self.synchronize_count = 0

    def is_available(self):
        return self.available

    def is_current_stream_capturing(self):
        return self.capturing

    def Stream(self):
        return _FakeStream()

    def stream(self, stream):
        return _FakeStreamContext()

    def current_stream(self):
        return self.current_stream_obj

    def synchronize(self):
        self.synchronize_count += 1

    def CUDAGraph(self):
        self.graph_obj = _FakeGraph()
        return self.graph_obj

    def graph(self, graph):
        return _FakeGraphContext()


def _fake_torch(cuda):
    return SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip=None))


_FAKE_COMPILER_SEED = {
    "block_sizes": [1, 128, 128],
    "cute_flash_topology": "fa4",
    "cute_flash_causal_lpt_swizzle": 4,
}

_REBENCHMARK_OVERRIDE_KEYS = (
    "HELION_AUTOTUNE_FINAL_PICK_TOP_K",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K",
    "HELION_AUTOTUNE_SUSPICIOUS_REBENCHMARK_RATIO",
    "HELION_CAP_REBENCHMARK_REPEAT",
    "HELION_REBENCHMARK_THRESHOLD",
)


def _clear_strict_autotune_overrides(monkeypatch):
    # Strict autotune validation fails closed on ambient rebenchmark and codegen
    # overrides; CI matrices export some (e.g. HELION_INTERPRET=1,
    # HELION_DEBUG_DTYPE_ASSERTS=1), so scrub them before exercising it.
    for key in _REBENCHMARK_OVERRIDE_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in compare_attention_backends._codegen_override_env_keys(set()):
        monkeypatch.delenv(key, raising=False)


def _attention_subprocess_args(**overrides):
    args = SimpleNamespace(
        z=1,
        h=2,
        seq_len=128,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=1,
        epilogue="none",
        num_runs=5,
        warmup_ms=25,
        rep_ms=100,
        seed=123,
        power_cap_w=750,
        skip_correctness=0,
        helion_force_flash_config=1,
        helion_force_autotune=0,
        helion_require_full_autotune=0,
        helion_return_lse=0,
        helion_cute_benchmark_timer="wall",
        helion_env=[],
        helion_autotune_effort=None,
        helion_autotune_budget_seconds=None,
        helion_autotune_max_generations=None,
        helion_autotune_best_of_k=None,
        helion_autotune_benchmark_timeout=None,
        helion_autotune_accuracy_check=None,
        helion_autotuner_initial_population=None,
        helion_config=[],
        helion_seed_config=[],
        impls=[],
        stream_subprocesses=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_attention_tilegym_tileir_registry():
    assert "tilegym-tileir" in compare_attention_backends.ALL_IMPLS
    assert "tilegym-tileir" not in compare_attention_backends.DEFAULT_IMPLS
    assert "tilegym-tileir" in compare_attention_backends._DISPLAY_IMPLS
    assert compare_attention_backends._IMPL_LABELS["tilegym-tileir"] == "TileGym+TileIR"
    assert compare_attention_backends._IMPL_KEYS["tilegym-tileir"] == "tilegym_tileir"


def test_attention_tilegym_tileir_dispatch(monkeypatch):
    monkeypatch.setattr(
        compare_attention_backends,
        "_benchmark_tilegym_tileir",
        lambda args: {"impl": args.impl},
    )

    assert compare_attention_backends._run_impl(
        SimpleNamespace(impl="tilegym-tileir")
    ) == {"impl": "tilegym-tileir"}


def test_attention_tilegym_kwargs_set_dense_and_matrix_bias():
    dense_args = SimpleNamespace(causal=0)
    dense_kwargs = compare_attention_backends._tilegym_attention_kwargs(
        dense_args, None
    )
    bias = torch.ones(1)
    biased_kwargs = compare_attention_backends._tilegym_attention_kwargs(
        dense_args, bias
    )

    assert dense_kwargs["is_causal"] is False
    assert dense_kwargs["layout"] == "bnsd"
    assert dense_kwargs["bias_type"] is None
    assert biased_kwargs["bias_type"] == "matrix"
    assert biased_kwargs["bias"] is bias


def test_attention_tilegym_unavailable_skips(monkeypatch):
    def unavailable():
        raise RuntimeError("TileIR unavailable")

    monkeypatch.setattr(
        compare_attention_backends,
        "_import_tilegym_fmha",
        unavailable,
    )

    result = compare_attention_backends._benchmark_tilegym_tileir(
        _attention_subprocess_args(causal=0, biased=0)
    )

    assert result["impl"] == "tilegym-tileir"
    assert result["accuracy"] == "SKIP"
    assert result["skipped_reason"] == "TileIR unavailable"


def test_attention_tilegym_result_records_best_config(monkeypatch):
    class FakeConfig:
        def all_kwargs(self):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "num_warps": 4,
                "num_stages": 3,
            }

    tensor = torch.ones(1)
    monkeypatch.setattr(
        compare_attention_backends,
        "_import_tilegym_fmha",
        lambda: (
            lambda *args, **kwargs: tensor,
            lambda: FakeConfig(),
        ),
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_make_inputs",
        lambda args, dtype: (tensor, tensor, tensor),
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_make_bias",
        lambda args, dtype: None,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_sdpa_reference",
        lambda *args, **kwargs: tensor,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_check_close",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_bench_steady",
        lambda *args, **kwargs: {
            "best_ms": 1.0,
            "median_ms": 1.0,
            "mean_ms": 1.0,
            "std_ms": 0.0,
            "runs_ms": [1.0],
        },
    )

    result = compare_attention_backends._benchmark_tilegym_tileir(
        _attention_subprocess_args(causal=0, biased=0)
    )

    assert result["accuracy"] == "PASS"
    assert result["config"] == {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "num_warps": 4,
        "num_stages": 3,
    }


def test_bench_steady_scores_every_timer_on_median(monkeypatch) -> None:
    """Backends must share one per-run statistic: the default do_bench path
    must request the median, matching the CuTe backend-timer path."""
    modes: list[str] = []

    def fake_bench(fn, *, warmup, rep, return_mode):
        modes.append(return_mode)
        return 1.0

    triton_testing = pytest.importorskip(
        "triton.testing", reason="default do_bench path requires triton"
    )

    monkeypatch.setattr(compare_attention_backends, "_gpu_warmup", lambda ms: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(triton_testing, "do_bench", fake_bench)

    default_stats = compare_attention_backends._bench_steady(
        lambda: None,
        num_runs=3,
        warmup_ms=1,
        rep_ms=1,
        cache_warmup_calls=1,
        thermal_warmup_ms=0,
    )
    assert modes == ["median"] * 3
    assert default_stats["median_ms"] == 1.0

    modes.clear()
    backend_stats = compare_attention_backends._bench_steady(
        lambda: None,
        num_runs=2,
        warmup_ms=1,
        rep_ms=1,
        do_bench_fn=fake_bench,
        cache_warmup_calls=1,
        thermal_warmup_ms=0,
    )
    assert modes == ["median"] * 2
    assert backend_stats["runs_ms"] == [1.0, 1.0]


def test_wait_for_gpu_cooldown_reasons(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(
        compare_attention_backends.time,
        "sleep",
        lambda s: clock.__setitem__("now", clock["now"] + s),
    )
    monkeypatch.setattr(
        compare_attention_backends.time, "perf_counter", lambda: clock["now"]
    )

    def temps(sequence):
        it = iter(sequence)
        return lambda: next(it)

    monkeypatch.setattr(compare_attention_backends, "_gpu_temperature_c", temps([45.0]))
    result = compare_attention_backends._wait_for_gpu_cooldown(55.0)
    assert result["reason"] == "already_cool"
    assert result["waited_s"] == 0.0
    assert result["start_temp_c"] == 45.0

    monkeypatch.setattr(
        compare_attention_backends, "_gpu_temperature_c", temps([70.0, 60.0, 54.0])
    )
    result = compare_attention_backends._wait_for_gpu_cooldown(55.0)
    assert result["reason"] == "reached"
    assert result["end_temp_c"] == 54.0

    # Temperature stuck just above the threshold plateaus out instead of
    # spinning until the timeout (ambient may make the threshold unreachable).
    monkeypatch.setattr(
        compare_attention_backends,
        "_gpu_temperature_c",
        temps([70.0] + [69.9] * 100),
    )
    result = compare_attention_backends._wait_for_gpu_cooldown(55.0)
    assert result["reason"] == "plateau"
    assert result["end_temp_c"] == 69.9

    monkeypatch.setattr(
        compare_attention_backends,
        "_gpu_temperature_c",
        temps([90.0, 85.0, 80.0, 75.0]),
    )
    result = compare_attention_backends._wait_for_gpu_cooldown(55.0, timeout_s=10.0)
    assert result["reason"] == "timeout"

    # Hard cap: a temperature that rises forever (never reaches the target,
    # never plateaus) must still terminate at the default timeout.
    state = {"temp": 60.0}

    def rising():
        state["temp"] += 1.0
        return state["temp"]

    monkeypatch.setattr(compare_attention_backends, "_gpu_temperature_c", rising)
    result = compare_attention_backends._wait_for_gpu_cooldown(55.0)
    assert result["reason"] == "timeout"
    assert result["waited_s"] <= compare_attention_backends._COOLDOWN_MAX_WAIT_S + 5.0

    monkeypatch.setattr(compare_attention_backends, "_gpu_temperature_c", temps([None]))
    assert compare_attention_backends._wait_for_gpu_cooldown(55.0) is None


def test_bench_steady_reports_cooldown_provenance(monkeypatch) -> None:
    marker = {"reason": "reached", "waited_s": 42.0}
    calls = []

    def fake_cooldown(max_temp_c):
        calls.append(max_temp_c)
        return dict(marker)

    def fake_bench(fn, *, warmup, rep, return_mode):
        return 1.0

    triton_testing = pytest.importorskip(
        "triton.testing", reason="default do_bench path requires triton"
    )
    monkeypatch.setattr(compare_attention_backends, "_gpu_warmup", lambda ms: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(triton_testing, "do_bench", fake_bench)
    monkeypatch.setattr(
        compare_attention_backends, "_wait_for_gpu_cooldown", fake_cooldown
    )

    stats = compare_attention_backends._bench_steady(
        lambda: None,
        num_runs=1,
        warmup_ms=1,
        rep_ms=1,
        cache_warmup_calls=1,
        thermal_warmup_ms=0,
        cooldown_max_temp_c=55.0,
    )
    assert calls == [55.0]
    assert stats["thermal_cooldown"] == marker

    stats = compare_attention_backends._bench_steady(
        lambda: None,
        num_runs=1,
        warmup_ms=1,
        rep_ms=1,
        cache_warmup_calls=1,
        thermal_warmup_ms=0,
    )
    assert calls == [55.0]
    assert "thermal_cooldown" not in stats


def test_cooldown_target_is_startup_reference_plus_margin(monkeypatch) -> None:
    args = SimpleNamespace(measure_cooldown_margin_c=3.0)

    # No startup reference captured (or unreadable) -> cooldown disabled.
    monkeypatch.setattr(compare_attention_backends, "_STARTUP_GPU_TEMP_C", None)
    assert compare_attention_backends._cooldown_target_temp_c(args) is None

    monkeypatch.setattr(compare_attention_backends, "_STARTUP_GPU_TEMP_C", 41.0)
    assert compare_attention_backends._cooldown_target_temp_c(args) == pytest.approx(
        44.0
    )
    # Negative margin disables; a missing attribute (bare namespace) disables.
    assert (
        compare_attention_backends._cooldown_target_temp_c(
            SimpleNamespace(measure_cooldown_margin_c=-1.0)
        )
        is None
    )
    assert compare_attention_backends._cooldown_target_temp_c(SimpleNamespace()) is None


def test_capture_startup_gpu_temperature(monkeypatch) -> None:
    monkeypatch.setattr(compare_attention_backends, "_STARTUP_GPU_TEMP_C", None)
    monkeypatch.setattr(compare_attention_backends, "_gpu_temperature_c", lambda: 39.0)
    compare_attention_backends._capture_startup_gpu_temperature()
    assert compare_attention_backends._STARTUP_GPU_TEMP_C == 39.0

    monkeypatch.setattr(compare_attention_backends, "_gpu_temperature_c", lambda: None)
    compare_attention_backends._capture_startup_gpu_temperature()
    assert compare_attention_backends._STARTUP_GPU_TEMP_C is None


def test_attention_epilogue_shape_metadata_preserves_identity_schema():
    args = _attention_subprocess_args(biased=0)

    identity_shape = compare_attention_backends._shape_dict(args)
    assert identity_shape == {
        "z": 1,
        "h": 2,
        "seq_len": 128,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 0,
        "biased": 0,
    }

    args.epilogue = "relu"
    relu_shape = compare_attention_backends._shape_dict(args)
    assert relu_shape == {**identity_shape, "epilogue": "relu"}
    assert compare_attention_backends._report_shape_key(
        identity_shape, context="identity"
    ) != compare_attention_backends._report_shape_key(relu_shape, context="relu")
    assert compare_attention_backends._shape_label(identity_shape).endswith(
        "_causal0_biased0"
    )
    assert compare_attention_backends._shape_label(relu_shape).endswith(
        "_causal0_biased0_epiloguerelu"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"dtype": "float16"}, "requires --dtype bfloat16"),
        ({"dtype": "bfloat16", "biased": 1}, "not compatible with --biased"),
        (
            {"dtype": "bfloat16", "biased": 0, "helion_return_lse": 1},
            "requires --helion-return-lse 0",
        ),
    ),
)
def test_attention_relu_epilogue_rejects_unsupported_workloads(overrides, message):
    args = _attention_subprocess_args(**{"epilogue": "relu", "biased": 0, **overrides})

    with pytest.raises(SystemExit, match=message):
        compare_attention_backends._validate_epilogue_workload(args)


def test_attention_relu_epilogue_accepts_bfloat16_output_only():
    args = _attention_subprocess_args(
        dtype="bfloat16", biased=0, epilogue="relu", helion_return_lse=0
    )

    compare_attention_backends._validate_epilogue_workload(args)


def test_attention_cli_parses_relu_epilogue(monkeypatch):
    monkeypatch.setattr(
        compare_attention_backends.sys,
        "argv",
        ["compare_attention_backends.py", "--epilogue", "relu"],
    )

    assert compare_attention_backends.parse_args().epilogue == "relu"


def test_attention_force_flash_config_uses_compiler_default_seed():
    args = SimpleNamespace(
        helion_config=[],
        helion_force_flash_config=1,
        helion_backend="cute",
    )

    config, overrides = compare_attention_backends._make_helion_config(
        args, _FAKE_COMPILER_SEED
    )

    assert overrides == {}
    assert config == {
        "block_sizes": [1, 128, 128],
        "cute_flash_topology": "fa4",
        "cute_flash_causal_lpt_swizzle": 4,
    }


def test_attention_force_flash_config_applies_manual_overrides_to_seed():
    args = SimpleNamespace(
        helion_config=[("cute_flash_causal_lpt_swizzle", 0)],
        helion_force_flash_config=1,
        helion_backend="cute",
    )

    config, overrides = compare_attention_backends._make_helion_config(
        args, _FAKE_COMPILER_SEED
    )

    assert overrides == {"cute_flash_causal_lpt_swizzle": 0}
    assert config["cute_flash_topology"] == "fa4"
    assert config["cute_flash_causal_lpt_swizzle"] == 0


def test_attention_force_flash_config_falls_back_without_compiler_seed():
    args = SimpleNamespace(
        helion_config=[],
        helion_force_flash_config=1,
        helion_backend="cute",
    )
    config, overrides = compare_attention_backends._make_helion_config(args, None)

    assert overrides == {}
    assert config == {"block_sizes": [1, 128, 128]}


def test_attention_compiler_flash_seed_config_uses_promoted_default():
    bound = SimpleNamespace(
        config_spec=SimpleNamespace(
            compiler_default_config=object(),
            compiler_seed_configs=[
                SimpleNamespace(config={"block_sizes": [1, 128, 128]})
            ],
            default_config=lambda: SimpleNamespace(
                config={
                    "block_sizes": [1, 128, 128],
                    "cute_flash_topology": "fa4",
                }
            ),
        )
    )

    config = compare_attention_backends._compiler_flash_seed_config(bound, "cute")

    assert config == {
        "block_sizes": [1, 128, 128],
        "cute_flash_topology": "fa4",
    }


def test_attention_compiler_flash_seed_config_falls_back_to_seed_list():
    bound = SimpleNamespace(
        config_spec=SimpleNamespace(
            compiler_default_config=None,
            compiler_seed_configs=[
                SimpleNamespace(
                    config={
                        "block_sizes": [64, 64],
                        "num_warps": 8,
                    }
                ),
                SimpleNamespace(
                    config={
                        "block_sizes": [1, 128, 128],
                        "cute_flash_kv_order": "ascending",
                    }
                ),
            ],
        )
    )

    config = compare_attention_backends._compiler_flash_seed_config(bound, "cute")

    assert config == {
        "block_sizes": [1, 128, 128],
        "cute_flash_kv_order": "ascending",
    }


def test_attention_compiler_flash_seed_config_skips_nonflash_default():
    bound = SimpleNamespace(
        config_spec=SimpleNamespace(
            compiler_default_config=object(),
            compiler_seed_configs=[
                SimpleNamespace(
                    config={
                        "block_sizes": [1, 128, 128],
                        "cute_flash_topology": "fa4",
                    }
                )
            ],
            default_config=lambda: SimpleNamespace(
                config={"block_sizes": [64, 64], "num_warps": 8}
            ),
        )
    )

    config = compare_attention_backends._compiler_flash_seed_config(bound, "cute")

    assert config == {
        "block_sizes": [1, 128, 128],
        "cute_flash_topology": "fa4",
    }


def test_attention_subprocess_forwards_helion_cute_timer(monkeypatch):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(
        helion_cute_benchmark_timer="event",
        helion_require_full_autotune=0,
        helion_seed_config=[("block_sizes", [1, 64, 128])],
        dtype="bfloat16",
        biased=0,
        epilogue="relu",
    )

    cmd = compare_attention_backends._build_subprocess_cmd(args, "helion-cute")

    flag_index = cmd.index("--helion-cute-benchmark-timer")
    assert cmd[flag_index + 1] == "event"
    cooldown_index = cmd.index("--measure-cooldown-margin-c")
    assert cmd[cooldown_index + 1] == "3.0"
    power_index = cmd.index("--power-cap-w")
    assert cmd[power_index + 1] == "750"
    seed_index = cmd.index("--helion-seed-config")
    assert cmd[seed_index + 1] == "block_sizes=[1, 64, 128]"
    epilogue_index = cmd.index("--epilogue")
    assert cmd[epilogue_index + 1] == "relu"
    truth_index = cmd.index("--helion-require-full-autotune")
    assert cmd[truth_index + 1] == "0"


def test_attention_helion_cute_timer_defaults_to_event(monkeypatch):
    # Event timing is the default so helion-cute samples use the exact same
    # CUDA-event do_bench path as the SDPA/FlexAttention/FA4 baselines; the
    # wall timer stays available for cross-checks only.
    monkeypatch.setattr(
        compare_attention_backends.sys,
        "argv",
        ["compare_attention_backends.py", "--impl", "helion-cute"],
    )
    parser_args = compare_attention_backends.parse_args()
    assert parser_args.helion_cute_benchmark_timer == "event"
    assert (
        compare_attention_backends._helion_benchmark_timer(parser_args, "cute")
        == "event"
    )


def test_attention_subprocess_limits_strict_autotune_to_helion_cute(monkeypatch):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(helion_require_full_autotune=1)

    cmd = compare_attention_backends._build_subprocess_cmd(args, "helion-triton")

    truth_index = cmd.index("--helion-require-full-autotune")
    assert cmd[truth_index + 1] == "0"
    assert "HELION_DISABLE_AUTOTUNER_HEURISTICS=1" not in cmd
    assert "HELION_AUTOTUNER_INITIAL_POPULATION=from_random" not in cmd


@pytest.mark.parametrize(
    ("impl", "backend", "expected_require_full_autotune"),
    (
        ("helion-cute", "cute", 1),
        ("helion-triton", "triton", 0),
        ("helion-tileir", "tileir", 0),
    ),
)
def test_attention_direct_impl_limits_strict_autotune_to_helion_cute(
    monkeypatch, impl, backend, expected_require_full_autotune
):
    args = SimpleNamespace(impl=impl, helion_require_full_autotune=1)
    seen = []

    def benchmark_helion(args):
        seen.append((args.helion_backend, args.helion_require_full_autotune))
        return {"impl": impl}

    monkeypatch.setattr(
        compare_attention_backends, "_benchmark_helion", benchmark_helion
    )

    assert compare_attention_backends._run_impl(args) == {"impl": impl}
    assert seen == [(backend, expected_require_full_autotune)]


def test_attention_shape_subprocess_forwards_helion_cute_timer(monkeypatch):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(
        helion_cute_benchmark_timer="event", helion_require_full_autotune=1
    )
    seen_cmds = []

    def run_json_subprocess(cmd, args):
        seen_cmds.append(cmd)
        return 0, {"shape": {}, "results": []}, "", ""

    monkeypatch.setattr(
        compare_attention_backends, "_run_json_subprocess", run_json_subprocess
    )

    compare_attention_backends._run_shape_subprocess(
        args, (1, 2, 128, 64, "float16", 0, 1)
    )

    flag_index = seen_cmds[0].index("--helion-cute-benchmark-timer")
    assert seen_cmds[0][flag_index + 1] == "event"
    power_index = seen_cmds[0].index("--power-cap-w")
    assert seen_cmds[0][power_index + 1] == "750"
    truth_index = seen_cmds[0].index("--helion-require-full-autotune")
    assert seen_cmds[0][truth_index + 1] == "1"
    epilogue_index = seen_cmds[0].index("--epilogue")
    assert seen_cmds[0][epilogue_index + 1] == "none"
    assert "HELION_DISABLE_AUTOTUNER_HEURISTICS=1" not in seen_cmds[0]
    assert "HELION_AUTOTUNER_INITIAL_POPULATION=from_random" not in seen_cmds[0]


def test_attention_shape_subprocess_skips_strict_mode_without_helion_cute(
    monkeypatch,
):
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        impls=["sdpa"],
        skip_correctness=1,
    )
    seen_cmds = []

    def run_json_subprocess(cmd, args):
        seen_cmds.append(cmd)
        return 0, {"shape": {}, "results": []}, "", ""

    monkeypatch.setattr(
        compare_attention_backends, "_run_json_subprocess", run_json_subprocess
    )

    compare_attention_backends._run_shape_subprocess(
        args, (1, 2, 128, 64, "float16", 0, 0)
    )

    truth_index = seen_cmds[0].index("--helion-require-full-autotune")
    assert seen_cmds[0][truth_index + 1] == "0"


@pytest.mark.parametrize(
    ("returncode", "payload", "message"),
    (
        (1, None, "subprocess failed"),
        (0, None, "produced no JSON output"),
    ),
)
def test_attention_run_all_strict_autotune_fails_closed(
    monkeypatch, returncode, payload, message
):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        impls=["helion-cute"],
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_run_json_subprocess",
        lambda cmd, args: (returncode, payload, "stdout", "stderr"),
    )

    with pytest.raises(SystemExit, match=message):
        compare_attention_backends._run_all(args)


@pytest.mark.parametrize(
    ("returncode", "payload", "message"),
    (
        (1, None, "shape subprocess failed"),
        (0, None, "shape subprocess produced no JSON output"),
    ),
)
def test_attention_shape_subprocess_strict_autotune_fails_closed(
    monkeypatch, returncode, payload, message
):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        impls=["helion-cute"],
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_run_json_subprocess",
        lambda cmd, args: (returncode, payload, "stdout", "stderr"),
    )

    with pytest.raises(SystemExit, match=message):
        compare_attention_backends._run_shape_subprocess(
            args, (1, 2, 128, 64, "float16", 0, 0)
        )


def test_attention_merge_json_rejects_unvalidated_strict_autotune():
    args = SimpleNamespace(helion_require_full_autotune=1)

    with pytest.raises(SystemExit, match="cannot verify"):
        compare_attention_backends._run_merge_json(args)


def _full_autotune_provenance(**overrides):
    compiler_seed_policy = copy.deepcopy(_full_autotune_compiler_seed_policy())
    fragment_default = {
        "block_sizes": [1, 128, 128],
        "cute_flash_exp2_packet": "1x1",
        "cute_flash_pipeline_family": "fa4_2cta",
        "cute_flash_topology": "fa4",
    }
    fragment_default_json = json.dumps(
        fragment_default, sort_keys=True, separators=(",", ":")
    )
    coverage_config_sha256 = hashlib.sha256(
        fragment_default_json.encode("utf-8")
    ).hexdigest()
    coverage_configs_sha256 = hashlib.sha256(
        json.dumps([fragment_default], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    terminal_policy = {
        "schema_version": 2,
        "policy_version": 2,
        "lane_policy_version": 14,
        "coordinate_policy": "same_leaf_full_surface_normalized_coordinate_v2",
        "measurement_policy": "mirrored_rotating_batched_wall_v2",
        "rounds": 2,
        "beam_width": 4,
        "radius": 2,
        "minimum_improvement_fraction": 0.001,
        "round_target_ms": 200.0,
        "confirmation_target_ms": 5000.0,
    }
    terminal_surface = {
        "schema_version": 1,
        "radius": 2,
        "leaves": [
            {
                "leaf": {
                    "family": "fa4_2cta",
                    "compound_packet": None,
                    "softmax_disc": True,
                },
                "coordinates": [
                    {
                        "flat_index": 0,
                        "key": "block_sizes",
                        "sequence_index": 0,
                        "fragment_type": "BlockSizeFragment",
                        "overridden": False,
                        "active_values": [1],
                        "neighbors_by_value": [{"from_value": 1, "to_values": []}],
                    }
                ],
            }
        ],
    }
    provenance = {
        "helion_import_path": str(
            compare_attention_backends.REPO_ROOT / "helion" / "__init__.py"
        ),
        "helion_expected_package_path": str(
            compare_attention_backends.REPO_ROOT / "helion"
        ),
        "helion_import_root_matches_repo": True,
        "attention_example_import_path": str(
            compare_attention_backends.REPO_ROOT / "examples" / "attention.py"
        ),
        "attention_example_expected_module_path": str(
            compare_attention_backends.REPO_ROOT / "examples" / "attention.py"
        ),
        "attention_example_import_matches_repo": True,
        "helion_checkout_git_commit": "a" * 40,
        "helion_checkout_git_describe": "a" * 12,
        "helion_source_tree_sha256": "b" * 64,
        "helion_source_tree_file_count": 1,
        "helion_source_tree_dirty": False,
        "physical_gpu_selection": "6",
        "strict_runtime_environment": {
            "cuda_device_order": "PCI_BUS_ID",
            "forbidden_overrides": {},
            "startup_pythonpath": None,
            "worker_pythonpath": str(compare_attention_backends.REPO_ROOT),
        },
        "require_full_autotune": True,
        "effort": "full",
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_best_of_k": 1,
        "autotune_random_seed": 123,
        "autotune_config_overrides": {},
        "user_seed_configs": False,
        "disable_autotuner_heuristics": False,
        "compiler_seed_config_count": compiler_seed_policy["raw_config_count"],
        "compiler_seed_policy": compiler_seed_policy,
        "compiler_default_config": False,
        "kernel_declared_config_count": 0,
        "autotune_initial_population_strategy_override": None,
        "autotune_initial_population_size": 100,
        "flash_exact_effective_search_space_size": None,
        "flash_exact_effective_search_space_config_ids": None,
        "flash_exact_effective_search_space_sha256": None,
        "autotune_lfbo_max_generations": 20,
        "autotuner_initial_population_env": "from_random",
        "autotuner_env": "",
        "autotune_num_neighbors_cap_env": "-1",
        "autotuner_fn": "helion.runtime.settings.default_autotuner_fn",
        "autotuner_fn_is_default": True,
        "autotune_baseline_fn": (
            "benchmarks.cute.compare_attention_backends._sdpa_reference"
        ),
        "autotune_baseline_fn_is_expected": True,
        "autotune_baseline_atol": 5e-2,
        "autotune_baseline_rtol": 2e-2,
        "autotune_baseline_accuracy_check_fn": False,
        "autotune_benchmark_fn": False,
        "autotune_rebenchmark_threshold": None,
        "autotune_suspicious_rebenchmark_ratio": None,
        "autotune_accuracy_check": True,
        "autotune_compile_timeout": 60,
        "autotune_benchmark_subprocess": True,
        "autotune_benchmark_subprocess_env": "",
        "autotune_benchmark_timeout": 30,
        "autotune_adaptive_timeout": True,
        "autotune_force_persistent": False,
        "autotune_finishing_rounds_env": "",
        "autotune_ignore_errors": False,
        "autotune_search_acf": [],
        "autotune_config_filter": False,
        "active_value_prior_keys": [],
        "flash_value_prior_keys": [],
        "flash_structural_coverage_design": [
            {"config": fragment_default, "config_sha256": coverage_config_sha256}
        ],
        "flash_structural_coverage_design_count": 1,
        "flash_structural_coverage_design_sha256": coverage_configs_sha256,
        "flash_structural_coverage_design_source": (
            "normalized active ConfigSpec fragments"
        ),
        "flash_structural_coverage_active_values": [
            {"key": "cute_flash_topology", "value": "fa4"}
        ],
        "flash_structural_coverage_uncovered_values": [],
        "flash_structural_coverage_underqualified_values": [],
        "flash_structural_leaf_catalog": [
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": True,
            }
        ],
        "flash_pipeline_lane_catalog": [
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": True,
                "pipeline_lanes": [],
            }
        ],
        "flash_clc_lane_catalog": [],
        "flash_clc_lane_catalog_sha256": hashlib.sha256(b"[]").hexdigest(),
        "flash_structural_coverage_underqualified_leaves": [],
        "flash_structural_coverage_interaction_key_groups": [
            [
                "cute_flash_epi_tma",
                "cute_flash_epi_stg",
                "cute_flash_epi_stg_store",
                "cute_flash_epi_stg_gmem",
            ]
        ],
        "flash_structural_coverage_active_interactions": [],
        "flash_structural_coverage_uncovered_interactions": [],
        "flash_structural_qualification_values": [],
        "flash_structural_parent_coverage_prefix_count": 1,
        "flash_structural_qualification_prefix_count": 1,
        "flash_structural_population_budget": 50,
        "flash_structural_injected_design_count": 1,
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_family_probe_generations": 1,
        "flash_structural_family_probe_candidates_per_path": 20,
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": 4,
        "flash_structural_retained_family_limit": 1,
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_starting_path_limit": 14,
        "flash_structural_family_probe_path_limit": 0,
        "flash_structural_maximum_path_capacity": 14,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        "flash_terminal_coordinate_refinement_policy": terminal_policy,
        "flash_terminal_coordinate_refinement_policy_sha256": hashlib.sha256(
            json.dumps(terminal_policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "flash_terminal_coordinate_surface_catalog": terminal_surface,
        "flash_terminal_coordinate_surface_catalog_sha256": hashlib.sha256(
            json.dumps(terminal_surface, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "flash_fragment_default_config": fragment_default,
        "flash_fragment_default_sha256": hashlib.sha256(
            fragment_default_json.encode("utf-8")
        ).hexdigest(),
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": 64,
        "final_repeatability_passed": True,
        "autotune_cache": "LocalAutotuneCache",
        "rebenchmark_env_overrides": {},
        "selected_source_sha256": "a" * 64,
        "selected_config": {"block_sizes": [1, 128, 128]},
        "selected_config_is_structural_coverage_design_member": False,
        "selected_config_nearest_structural_coverage_design_field_distance": 1,
        "selected_config_nearest_structural_coverage_design_config_sha256": [
            coverage_config_sha256
        ],
    }
    provenance.update(overrides)
    return provenance


def test_attention_required_full_autotune_rejects_foreign_helion_checkout():
    with pytest.raises(SystemExit, match="not imported from the checkout"):
        compare_attention_backends._validate_required_full_autotune(
            _full_autotune_provenance(helion_import_root_matches_repo=False)
        )


def test_attention_required_full_autotune_rejects_foreign_attention_example():
    with pytest.raises(SystemExit, match="attention example was not imported"):
        compare_attention_backends._validate_required_full_autotune(
            _full_autotune_provenance(attention_example_import_matches_repo=False)
        )


def test_attention_required_full_autotune_rejects_foreign_source_before_setup(
    monkeypatch,
):
    setup_called = False

    def apply_env(_args):
        nonlocal setup_called
        setup_called = True
        return {}

    monkeypatch.setattr(
        compare_attention_backends,
        "_helion_source_provenance",
        lambda: {
            "helion_import_root_matches_repo": True,
            "attention_example_import_matches_repo": False,
        },
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(compare_attention_backends, "_apply_helion_env", apply_env)

    with pytest.raises(SystemExit, match="source modules were not imported"):
        compare_attention_backends._benchmark_helion(
            SimpleNamespace(helion_require_full_autotune=1)
        )

    assert setup_called is False


def test_attention_required_full_autotune_rejects_dirty_source_before_setup(
    monkeypatch,
):
    setup_called = False

    def apply_env(_args):
        nonlocal setup_called
        setup_called = True
        return {}

    monkeypatch.setattr(
        compare_attention_backends,
        "_helion_source_provenance",
        lambda: {
            "helion_import_root_matches_repo": True,
            "attention_example_import_matches_repo": True,
            "helion_source_tree_dirty": True,
        },
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(compare_attention_backends, "_apply_helion_env", apply_env)

    with pytest.raises(SystemExit, match="source checkout has tracked or untracked"):
        compare_attention_backends._benchmark_helion(
            SimpleNamespace(helion_require_full_autotune=1)
        )

    assert setup_called is False


def test_attention_required_full_autotune_accepts_unchanged_source():
    source = _full_autotune_provenance()

    compare_attention_backends._validate_helion_source_unchanged(source, source.copy())


@pytest.mark.parametrize(
    "override",
    (
        {"helion_checkout_git_commit": "c" * 40},
        {"helion_source_tree_sha256": "d" * 64},
        {"helion_source_tree_file_count": 2},
        {"helion_source_tree_dirty": True},
    ),
)
def test_attention_required_full_autotune_rejects_source_change(override):
    initial = _full_autotune_provenance()
    current = {**initial, **override}

    with pytest.raises(SystemExit, match="source checkout"):
        compare_attention_backends._validate_helion_source_unchanged(initial, current)


def test_attention_required_full_autotune_records_post_measurement_source(monkeypatch):
    provenance = _full_autotune_provenance()
    final_source = {
        key: provenance[key]
        for key in compare_attention_backends._SOURCE_STABILITY_KEYS
    }
    monkeypatch.setattr(
        compare_attention_backends,
        "_helion_source_provenance",
        lambda: final_source.copy(),
    )

    compare_attention_backends._validate_post_measurement_source(provenance)

    assert provenance["post_measurement_source_verified"] is True
    assert provenance["post_measurement_source"] == final_source


def _clear_strict_runtime_overrides(monkeypatch):
    for key in list(os.environ):
        if (
            key in compare_attention_backends._STRICT_FORBIDDEN_RUNTIME_ENV
            or key.startswith(
                compare_attention_backends._STRICT_FORBIDDEN_RUNTIME_ENV_PREFIXES
            )
        ):
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    "name",
    (
        "PYTHONPATH",
        "CUDA_MODULE_LOADING",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "NVIDIA_TF32_OVERRIDE",
        "PYTORCH_CUDA_ALLOC_CONF",
    ),
)
def test_attention_required_full_autotune_rejects_runtime_override(monkeypatch, name):
    _clear_strict_runtime_overrides(monkeypatch)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv(name, "test-value")

    with (
        pytest.raises(SystemExit, match="ambient runtime overrides"),
        compare_attention_backends._strict_helion_runtime_environment(True),
    ):
        pass


def test_attention_required_full_autotune_controls_worker_pythonpath(monkeypatch):
    _clear_strict_runtime_overrides(monkeypatch)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    with compare_attention_backends._strict_helion_runtime_environment(
        True
    ) as evidence:
        assert os.environ["PYTHONPATH"] == str(compare_attention_backends.REPO_ROOT)
        assert evidence == {
            "cuda_device_order": "PCI_BUS_ID",
            "forbidden_overrides": {},
            "startup_pythonpath": None,
            "worker_pythonpath": str(compare_attention_backends.REPO_ROOT),
        }

    assert "PYTHONPATH" not in os.environ


def test_attention_required_full_autotune_requires_pci_bus_order(monkeypatch):
    _clear_strict_runtime_overrides(monkeypatch)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)

    with (
        pytest.raises(SystemExit, match="CUDA_DEVICE_ORDER=PCI_BUS_ID"),
        compare_attention_backends._strict_helion_runtime_environment(True),
    ):
        pass


def test_attention_helion_source_provenance_matches_benchmark_checkout():
    provenance = compare_attention_backends._helion_source_provenance()

    assert provenance["helion_import_root_matches_repo"] is True
    assert provenance["helion_expected_package_path"] == str(
        compare_attention_backends.REPO_ROOT / "helion"
    )
    assert provenance["attention_example_import_matches_repo"] is True
    assert provenance["attention_example_expected_module_path"] == str(
        compare_attention_backends.REPO_ROOT / "examples" / "attention.py"
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=compare_attention_backends.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if git_head.returncode != 0:
        # CI containers cannot always read the mounted checkout's git metadata
        # (e.g. dubious-ownership); the script degrades provenance to None then.
        assert provenance["helion_checkout_git_commit"] is None
        assert provenance["helion_source_tree_sha256"] is None
        assert provenance["helion_source_tree_file_count"] is None
    else:
        assert provenance["helion_checkout_git_commit"] == git_head.stdout.strip()
        assert len(provenance["helion_source_tree_sha256"]) == 64
        assert provenance["helion_source_tree_file_count"] > 0


def test_attention_direct_script_pins_helion_to_benchmark_checkout(tmp_path):
    foreign_package = tmp_path / "helion"
    foreign_package.mkdir()
    (foreign_package / "__init__.py").write_text(
        '__version__ = "foreign"\n', encoding="utf-8"
    )
    foreign_examples = tmp_path / "examples"
    foreign_examples.mkdir()
    (foreign_examples / "__init__.py").write_text("", encoding="utf-8")
    (foreign_examples / "attention.py").write_text("", encoding="utf-8")
    script = Path(compare_attention_backends.__file__).resolve()
    # Hand the result back through a file: runtime imports (e.g. TPU libraries)
    # can write startup noise to the child's stdout.
    result_path = tmp_path / "child_result.json"
    child_code = (
        "import json, pathlib, runpy, sys; "
        f"ns = runpy.run_path({str(script)!r}); "
        "before = 'examples.attention' in sys.modules; "
        "provenance = ns['_helion_source_provenance'](); "
        "after = 'examples.attention' in sys.modules; "
        "payload = json.dumps({'provenance': provenance, 'before': before, "
        "'after': after}); "
        f"pathlib.Path({str(result_path)!r}).write_text(payload, encoding='utf-8')"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    child_result = json.loads(result_path.read_text(encoding="utf-8"))
    provenance = child_result["provenance"]

    assert child_result["before"] is False
    assert child_result["after"] is False
    assert provenance["helion_import_root_matches_repo"] is True
    assert provenance["helion_import_path"] == str(
        compare_attention_backends.REPO_ROOT / "helion" / "__init__.py"
    )
    assert provenance["attention_example_import_path"] == str(
        compare_attention_backends.REPO_ROOT / "examples" / "attention.py"
    )


@functools.lru_cache(maxsize=1)
def _full_autotune_trial_configs():
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(
        _full_autotune_config_spec(), _flash_pipeline_family_override="fa4_2cta"
    )
    state = random.getstate()
    random.seed(12345)
    try:
        raw = generation.random_population(140)
    finally:
        random.setstate(state)
    configs_by_depth = {2: [], 3: []}
    seen: set[str] = set()
    witness_generation = ConfigGeneration(_full_autotune_config_spec())
    for (
        leaf,
        key,
        value,
    ), witness in witness_generation.flash_pipeline_lane_witnesses().items():
        if (
            leaf.pipeline_family != "fa4_2cta"
            or leaf.compound_exp2_packet is not None
            or leaf.softmax_disc is not False
            or key != "cute_flash_kv_stage"
            or value not in configs_by_depth
        ):
            continue
        config = dict(witness.config)
        config_id = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        if config_id not in seen:
            seen.add(config_id)
            configs_by_depth[value].append(config)
    for index, candidate in enumerate(raw):
        projected = compare_attention_backends._canonical_flash_projection(
            generation,
            dict(candidate.config),
            {
                "cute_flash_kv_stage": 2 if index % 2 == 0 else 3,
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_pipeline_family": "fa4_2cta",
                "cute_flash_softmax_disc": False,
            },
        )
        if compare_attention_backends._flash_structural_leaf_dict(projected) != {
            "family": "fa4_2cta",
            "compound_packet": None,
            "softmax_disc": False,
        }:
            continue
        config_id = hashlib.sha256(
            json.dumps(projected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        depth = cast("int", projected["cute_flash_kv_stage"])
        if config_id not in seen and len(configs_by_depth[depth]) < 51:
            seen.add(config_id)
            configs_by_depth[depth].append(projected)
        if all(len(configs) == 51 for configs in configs_by_depth.values()):
            break
    assert all(len(configs) == 51 for configs in configs_by_depth.values())
    return tuple(
        config
        for pair in zip(configs_by_depth[2], configs_by_depth[3], strict=True)
        for config in pair
    )


@functools.lru_cache(maxsize=1)
def _full_autotune_terminal_fixture():
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.runtime.config import Config

    configs = [Config.from_dict(config) for config in _full_autotune_trial_configs()]
    initial = min(configs[:100], key=canonical_config_id)
    initial_id = canonical_config_id(initial)
    leaf = flash_structural_leaf_from_config(initial.config)
    assert leaf is not None
    config_spec = _full_autotune_config_spec()
    root_generation = ConfigGeneration(config_spec)
    overrides = dict(root_generation._override_values)
    overrides[FLASH_PIPELINE_FAMILY_KEY] = leaf.pipeline_family
    overrides[FLASH_SOFTMAX_DISC_KEY] = leaf.softmax_disc
    if leaf.compound_exp2_packet is not None:
        overrides[FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
    leaf_generation = config_spec.create_config_generation(
        overrides=overrides,
        advanced_controls_files=root_generation._advanced_controls_files,
        process_group_name=root_generation.process_group_name,
    )
    projections = root_generation.canonicalize_coordinate_projections(
        leaf_generation.coordinate_neighbor_projections(
            leaf_generation.flatten(initial), radius=2
        ),
        base_config=initial,
    )
    manifest = {initial_id: {"config": copy.deepcopy(initial.config)}}
    requests = []
    candidate_ids = []
    for projection in projections:
        outcome = projection.outcome
        projected = projection.config
        if (
            outcome == "candidate"
            and projected is not None
            and flash_structural_leaf_from_config(projected.config) != leaf
        ):
            outcome = "different_leaf"
        projected_id = canonical_config_id(projected) if projected is not None else None
        if projected is not None:
            assert projected_id is not None
            manifest[projected_id] = {"config": copy.deepcopy(projected.config)}
        if outcome == "candidate":
            assert projected_id is not None
            candidate_ids.append(projected_id)
        requests.append(
            {
                "flat_index": projection.flat_index,
                "key": projection.key,
                "sequence_index": projection.sequence_index,
                "from_value": copy.deepcopy(projection.from_value),
                "to_value": copy.deepcopy(projection.to_value),
                "outcome": outcome,
                "config_id": projected_id,
            }
        )
    manifest = dict(sorted(manifest.items()))
    transcript = {
        "schema_version": 2,
        "policy_version": 2,
        "lane_policy_version": 14,
        "coordinate_policy": "same_leaf_full_surface_normalized_coordinate_v2",
        "measurement_policy": "mirrored_rotating_batched_wall_v2",
        "rounds_planned": 2,
        "beam_width": 4,
        "maximum_projection_parent_count": 5,
        "projection_parent_count": 1,
        "rounds_started": 1,
        "rounds_completed": 1,
        "completed": True,
        "budget_exhausted": False,
        "termination_reason": "no_candidates",
        "search_generation": 20,
        "preterminal_num_configs_tested": 102,
        "preterminal_registry_config_count": 1,
        "preterminal_registry_config_ids_hash_policy": (
            "sorted_compact_json_sha256_v1"
        ),
        "preterminal_registry_config_ids_sha256": hashlib.sha256(
            json.dumps([initial_id], separators=(",", ":")).encode()
        ).hexdigest(),
        "radius": 2,
        "minimum_improvement_fraction": 0.001,
        "initial_incumbent_config_id": initial_id,
        "refined_config_id": initial_id,
        "final_config_id": initial_id,
        "projection_attempt_count": len(requests),
        "unique_candidate_count": len(set(candidate_ids)),
        "new_candidate_count": 0,
        "reused_candidate_count": 0,
        "intra_terminal_reused_candidate_count": 0,
        "prior_failed_candidate_count": len(set(candidate_ids)),
        "accepted_config_ids": [],
        "config_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "config_manifest": manifest,
        "rounds": [
            {
                "round_index": 1,
                "incumbent_config_id": initial_id,
                "leaf": {
                    "family": leaf.pipeline_family,
                    "compound_packet": leaf.compound_exp2_packet,
                    "softmax_disc": leaf.softmax_disc,
                },
                "parent_config_ids": [initial_id],
                "parent_projections": [
                    {
                        "parent_config_id": initial_id,
                        "coordinate_requests": requests,
                    }
                ],
                "candidate_config_ids": candidate_ids,
                "new_candidate_ids": [],
                "reused_candidate_ids": [],
                "intra_terminal_reused_candidate_ids": [],
                "prior_failed_candidate_ids": candidate_ids,
                "candidate_results": [],
                "comparison_config_ids": [],
                "measurement": None,
                "round_best_config_id": initial_id,
                "selected_config_id": initial_id,
                "accepted": False,
                "improvement_fraction": 0.0,
                "beam_config_ids": [initial_id],
            }
        ],
        "confirmation": {
            "candidate_config_ids": [initial_id],
            "measurement": None,
            "best_config_id": initial_id,
            "selected_config_id": initial_id,
            "accepted": False,
            "improvement_fraction": 0.0,
            "skipped_reason": "single_candidate",
        },
    }
    return transcript, copy.deepcopy(initial.config)


@functools.lru_cache(maxsize=1)
def _full_autotune_terminal_surface_catalog():
    from helion.autotuner.config_generation import ConfigGeneration

    return ConfigGeneration(
        _full_autotune_config_spec()
    ).flash_terminal_coordinate_surface_catalog(radius=2)


def _synchronize_full_autotune_terminal_boundary(trial):
    terminal = trial["search_phase_metrics"]["terminal_coordinate_refinement"]
    terminal["search_generation"] = trial["num_generations"]
    terminal["preterminal_num_configs_tested"] = (
        trial["num_configs_tested"] - terminal["new_candidate_count"]
    )


def _restore_full_autotune_terminal_fixture(trial, provenance):
    terminal, selected_config = copy.deepcopy(_full_autotune_terminal_fixture())
    terminal_surface = copy.deepcopy(_full_autotune_terminal_surface_catalog())
    trial["search_phase_metrics"]["terminal_coordinate_refinement"] = terminal
    trial["selected_config"] = selected_config
    provenance["selected_config"] = copy.deepcopy(selected_config)
    provenance["flash_terminal_coordinate_surface_catalog"] = terminal_surface
    provenance["flash_terminal_coordinate_surface_catalog_sha256"] = hashlib.sha256(
        json.dumps(terminal_surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _synchronize_full_autotune_terminal_boundary(trial)


def _terminal_measurement(config_ids, values, *, target_ms, repeat_reference_perf_ms):
    desired_calls = min(20_000, max(3, int(target_ms / repeat_reference_perf_ms)))
    desired_calls = max(2, desired_calls + desired_calls % 2)
    calls_per_sample = max(1, math.ceil(desired_calls / 64))
    sweep_count = math.ceil(desired_calls / calls_per_sample)
    if sweep_count % 2:
        sweep_count += 1
    total_calls = sweep_count * calls_per_sample
    indices = list(range(len(config_ids)))
    orders = []
    for sweep in range(sweep_count):
        offset = (sweep // 2) % len(indices)
        rotated = indices[offset:] + indices[:offset]
        orders.append(rotated if sweep % 2 == 0 else list(reversed(rotated)))
    return {
        "base_order": list(config_ids),
        "target_ms": target_ms,
        "repeat_reference_perf_ms": repeat_reference_perf_ms,
        "sweep_count": sweep_count,
        "calls_per_sample": calls_per_sample,
        "total_calls": total_calls,
        "elapsed_ms": [[values[index] for index in order] for order in orders],
        "median_ms": [
            {"config_id": config_id, "value": values[index]}
            for index, config_id in enumerate(config_ids)
        ],
    }


@functools.lru_cache(maxsize=1)
def _measured_full_autotune_terminal_fixture():
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.runtime.config import Config

    _base_terminal, initial_config = _full_autotune_terminal_fixture()
    initial = Config.from_dict(initial_config)
    initial_id = canonical_config_id(initial)
    leaf = flash_structural_leaf_from_config(initial.config)
    assert leaf is not None
    config_spec = _full_autotune_config_spec()
    root_generation = ConfigGeneration(config_spec)
    overrides = dict(root_generation._override_values)
    overrides[FLASH_PIPELINE_FAMILY_KEY] = leaf.pipeline_family
    overrides[FLASH_SOFTMAX_DISC_KEY] = leaf.softmax_disc
    if leaf.compound_exp2_packet is not None:
        overrides[FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
    leaf_generation = config_spec.create_config_generation(
        overrides=overrides,
        advanced_controls_files=root_generation._advanced_controls_files,
        process_group_name=root_generation.process_group_name,
    )
    manifest = {initial_id: {"config": copy.deepcopy(initial.config)}}

    def project(parent_ids):
        parents = {
            config_id: Config.from_dict(manifest[config_id]["config"])
            for config_id in parent_ids
        }
        parent_configs = set(parents.values())
        round_seen = set()
        parent_projections = []
        candidate_ids = []
        for parent_id in parent_ids:
            requests = []
            for projection in root_generation.canonicalize_coordinate_projections(
                leaf_generation.coordinate_neighbor_projections(
                    leaf_generation.flatten(parents[parent_id]), radius=2
                ),
                base_config=parents[parent_id],
            ):
                outcome = projection.outcome
                projected = projection.config
                if (
                    outcome == "candidate"
                    and projected is not None
                    and flash_structural_leaf_from_config(projected.config) != leaf
                ):
                    outcome = "different_leaf"
                elif outcome == "candidate" and projected in parent_configs:
                    outcome = "beam_alias"
                elif outcome == "candidate" and projected in round_seen:
                    outcome = "round_candidate_alias"
                projected_id = (
                    canonical_config_id(projected) if projected is not None else None
                )
                if projected is not None:
                    assert projected_id is not None
                    manifest[projected_id] = {"config": copy.deepcopy(projected.config)}
                if outcome == "candidate":
                    assert projected is not None and projected_id is not None
                    round_seen.add(projected)
                    candidate_ids.append(projected_id)
                requests.append(
                    {
                        "flat_index": projection.flat_index,
                        "key": projection.key,
                        "sequence_index": projection.sequence_index,
                        "from_value": copy.deepcopy(projection.from_value),
                        "to_value": copy.deepcopy(projection.to_value),
                        "outcome": outcome,
                        "config_id": projected_id,
                    }
                )
            parent_projections.append(
                {
                    "parent_config_id": parent_id,
                    "coordinate_requests": requests,
                }
            )
        return parent_projections, candidate_ids

    first_projections, first_candidates = project([initial_id])
    selected_id = first_candidates[0]
    selected = Config.from_dict(manifest[selected_id]["config"])
    selected_source = _test_measurement_source_hash(selected_id)
    first_other_candidates = [
        config_id for config_id in first_candidates if config_id != selected_id
    ]
    second_projections, second_candidates = project([selected_id, initial_id])
    all_prior_failed = set(first_other_candidates) | set(second_candidates)
    round_leaf = {
        "family": leaf.pipeline_family,
        "compound_packet": leaf.compound_exp2_packet,
        "softmax_disc": leaf.softmax_disc,
    }
    first_measurement = _terminal_measurement(
        [initial_id, selected_id],
        [60.0, 54.0],
        target_ms=200.0,
        repeat_reference_perf_ms=60.0,
    )
    second_measurement = _terminal_measurement(
        [selected_id, initial_id],
        [54.0, 60.0],
        target_ms=200.0,
        repeat_reference_perf_ms=60.0,
    )
    confirmation_measurement = _terminal_measurement(
        [initial_id, selected_id],
        [60.0, 54.0],
        target_ms=5000.0,
        repeat_reference_perf_ms=60.0,
    )
    rounds: list[dict[str, Any]] = [
        {
            "round_index": 1,
            "incumbent_config_id": initial_id,
            "leaf": round_leaf,
            "parent_config_ids": [initial_id],
            "parent_projections": first_projections,
            "candidate_config_ids": first_candidates,
            "new_candidate_ids": [],
            "reused_candidate_ids": [selected_id],
            "intra_terminal_reused_candidate_ids": [],
            "prior_failed_candidate_ids": first_other_candidates,
            "candidate_results": [
                {
                    "config_id": selected_id,
                    "attempt_perf": 60.0,
                    "selection_perf": 54.0,
                    "status": "ok",
                    "source_hash": selected_source,
                }
            ],
            "comparison_config_ids": [initial_id, selected_id],
            "measurement": first_measurement,
            "round_best_config_id": selected_id,
            "selected_config_id": selected_id,
            "accepted": True,
            "improvement_fraction": 0.1,
            "beam_config_ids": [selected_id, initial_id],
        },
        {
            "round_index": 2,
            "incumbent_config_id": selected_id,
            "leaf": round_leaf,
            "parent_config_ids": [selected_id, initial_id],
            "parent_projections": second_projections,
            "candidate_config_ids": second_candidates,
            "new_candidate_ids": [],
            "reused_candidate_ids": [],
            "intra_terminal_reused_candidate_ids": [],
            "prior_failed_candidate_ids": second_candidates,
            "candidate_results": [],
            "comparison_config_ids": [selected_id, initial_id],
            "measurement": second_measurement,
            "round_best_config_id": selected_id,
            "selected_config_id": selected_id,
            "accepted": False,
            "improvement_fraction": 0.0,
            "beam_config_ids": [selected_id, initial_id],
        },
    ]
    manifest = dict(sorted(manifest.items()))
    transcript = {
        "schema_version": 2,
        "policy_version": 2,
        "lane_policy_version": 14,
        "coordinate_policy": "same_leaf_full_surface_normalized_coordinate_v2",
        "measurement_policy": "mirrored_rotating_batched_wall_v2",
        "rounds_planned": 2,
        "beam_width": 4,
        "maximum_projection_parent_count": 5,
        "projection_parent_count": 3,
        "rounds_started": 2,
        "rounds_completed": 2,
        "completed": True,
        "budget_exhausted": False,
        "termination_reason": "round_limit",
        "search_generation": 20,
        "preterminal_num_configs_tested": 102,
        "preterminal_registry_config_count": 2,
        "preterminal_registry_config_ids_hash_policy": (
            "sorted_compact_json_sha256_v1"
        ),
        "preterminal_registry_config_ids_sha256": hashlib.sha256(
            json.dumps(
                sorted([initial_id, selected_id]), separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "radius": 2,
        "minimum_improvement_fraction": 0.001,
        "initial_incumbent_config_id": initial_id,
        "refined_config_id": selected_id,
        "final_config_id": selected_id,
        "projection_attempt_count": sum(
            len(parent["coordinate_requests"])
            for round_metric in rounds
            for parent in round_metric["parent_projections"]
        ),
        "unique_candidate_count": len(set(first_candidates + second_candidates)),
        "new_candidate_count": 0,
        "reused_candidate_count": 1,
        "intra_terminal_reused_candidate_count": 0,
        "prior_failed_candidate_count": len(all_prior_failed),
        "accepted_config_ids": [selected_id],
        "config_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "config_manifest": manifest,
        "rounds": rounds,
        "confirmation": {
            "candidate_config_ids": [initial_id, selected_id],
            "measurement": confirmation_measurement,
            "best_config_id": selected_id,
            "selected_config_id": selected_id,
            "accepted": True,
            "improvement_fraction": 0.1,
            "skipped_reason": None,
        },
    }
    return transcript, copy.deepcopy(selected.config), selected_source


def _test_measurement_source_hash(config_id: str) -> str:
    return hashlib.sha256(f"source:{config_id}".encode()).hexdigest()


def _populate_measurement_source_hashes(value, *, overwrite: bool = False) -> None:
    """Attach stable generated-source identities to fabricated measurements."""
    if isinstance(value, dict):
        if {
            "attempt_perf",
            "selection_perf",
            "status",
        } <= value.keys() and (
            "config_id" in value or "transferred_config_id" in value
        ):
            config_id = value.get("config_id", value.get("transferred_config_id"))
            source_hash = (
                _test_measurement_source_hash(config_id)
                if isinstance(config_id, str)
                and value.get("status") not in {"unknown", "projection_rejected"}
                else None
            )
            if overwrite or "source_hash" not in value:
                value["source_hash"] = source_hash
        for child in value.values():
            _populate_measurement_source_hashes(child, overwrite=overwrite)
    elif isinstance(value, list):
        for child in value:
            _populate_measurement_source_hashes(child, overwrite=overwrite)


def _full_autotune_trial(**overrides):
    configs = [copy.deepcopy(config) for config in _full_autotune_trial_configs()]
    terminal_refinement, selected_config = copy.deepcopy(
        _full_autotune_terminal_fixture()
    )
    config_ids = [
        hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        for config in configs
    ]
    initial_config_ids = config_ids[:100]
    conditional_config_ids = config_ids[100:]
    kv2_initial_config_ids = initial_config_ids[::2]
    kv3_initial_config_ids = initial_config_ids[1::2]
    best_config_id = min(initial_config_ids)
    alternate_lane_ids = (
        kv3_initial_config_ids
        if best_config_id in kv2_initial_config_ids
        else kv2_initial_config_ids
    )
    retained_config_ids = [best_config_id, min(alternate_lane_ids)]
    retained_lane_values = [
        2 if config_id in kv2_initial_config_ids else 3
        for config_id in retained_config_ids
    ]
    lane_specs = [
        (2, kv2_initial_config_ids, conditional_config_ids[0]),
        (3, kv3_initial_config_ids, conditional_config_ids[1]),
    ]

    def decision_result(config_id, measurement_pass_index):
        return {
            "config_id": config_id,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "measurement_pass_index": measurement_pass_index,
        }

    def measurement_update(config_id):
        return {
            "config_id": config_id,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
        }

    trial = {
        "input_shapes": repr([(2, 32, 65536, 64)] * 3),
        "dtypes": repr(["torch.float16"] * 3),
        "hardware": "NVIDIA B200",
        "random_seed": 123,
        "num_configs_tested": 102,
        "num_compile_failures": 0,
        "num_worker_failures": 0,
        "num_isolated_rebenchmark_timeouts": 0,
        "num_accuracy_failures": 0,
        "num_successful_candidate_measurements": 102,
        "num_unique_sources": 102,
        "num_source_deduplications": 0,
        "num_generations": 20,
        "search_algorithm": "LFBOTreeSearch",
        "selected_source_hash": "a" * 64,
        "selected_config": selected_config,
        "selected_source_was_measured": True,
        "search_phase_metrics": {
            "phase": "cute_flash_structural_qualification_v22",
            "cute_flash_lane_policy_version": 14,
            "completed": True,
            "initial_config_count": 100,
            "initial_config_ids": initial_config_ids,
            "config_manifest": {
                config_id: {
                    "config": config,
                }
                for config_id, config in zip(config_ids, configs, strict=True)
            },
            "initial_results": [
                {
                    "config_id": config_id,
                    "family": "fa4_2cta",
                    "compound_packet": None,
                    "softmax_disc": False,
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "measurement_pass_index": 0,
                    "pipeline_lanes": [
                        {
                            "key": "cute_flash_kv_stage",
                            "value": 2 if index % 2 == 0 else 3,
                        }
                    ],
                }
                for index, config_id in enumerate(initial_config_ids)
            ],
            "exact_space_enumerated": False,
            "exact_space_exhausted": False,
            "exact_space_raw_budget": 100,
            "exact_space_config_ids": [],
            "leaf_count": 1,
            "ordinary_leaf_count": 1,
            "compound_leaf_count": 0,
            "pipeline_qualification_keys": [
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ],
            "leaf_results": [
                {
                    "family": "fa4_2cta",
                    "compound_packet": None,
                    "softmax_disc": False,
                    "initial_config_ids": initial_config_ids,
                    "space_exhausted": False,
                    "space_config_count": None,
                    "ordinary_search_required": False,
                    "rounds": [
                        {
                            "candidate_config_ids": [],
                            "neighbor_generation_limit": 0,
                            "ordinary_neighbor_generation_limit": 0,
                            "parent_decisions": [
                                {
                                    "job_index": job_index,
                                    "kind": "witness",
                                    "pipeline_lane": {
                                        "key": "cute_flash_kv_stage",
                                        "value": value,
                                    },
                                    "selection_kind": "ranked_existing",
                                    "candidate_results": [
                                        decision_result(config_id, 0)
                                        for config_id in sorted(lane_initial_ids)
                                    ],
                                    "selected_config_id": min(lane_initial_ids),
                                    "generated_config_ids": [],
                                }
                                for job_index, (
                                    value,
                                    lane_initial_ids,
                                    _conditional_config_id,
                                ) in enumerate(lane_specs)
                            ],
                        },
                        {
                            "candidate_config_ids": conditional_config_ids,
                            "neighbor_generation_limit": 200,
                            "ordinary_neighbor_generation_limit": 0,
                            "parent_decisions": [
                                {
                                    "job_index": job_index,
                                    "kind": "conditional",
                                    "pipeline_lane": {
                                        "key": "cute_flash_kv_stage",
                                        "value": value,
                                    },
                                    "selection_kind": "ranked_parent",
                                    "candidate_results": [
                                        decision_result(config_id, 1)
                                        for config_id in sorted(lane_initial_ids)
                                    ],
                                    "selected_config_id": min(lane_initial_ids),
                                    "generated_config_ids": [conditional_config_id],
                                }
                                for job_index, (
                                    value,
                                    lane_initial_ids,
                                    conditional_config_id,
                                ) in enumerate(lane_specs)
                            ],
                        },
                    ],
                    "pipeline_lanes": [
                        {
                            "key": "cute_flash_kv_stage",
                            "value": value,
                            "initial_config_ids": lane_initial_ids,
                            "space_exhausted": False,
                            "space_config_count": None,
                            "conditional_required": True,
                            "rounds": [
                                {
                                    "candidate_config_ids": [min(lane_initial_ids)],
                                    "neighbor_generation_limit": 0,
                                },
                                {
                                    "candidate_config_ids": [conditional_config_id],
                                    "neighbor_generation_limit": 100,
                                },
                            ],
                            "witness_attempted": True,
                            "witness_config_id": min(lane_initial_ids),
                            "witness_succeeded": True,
                            "conditional_candidate_ids": [conditional_config_id],
                            "successful_conditional_candidate_ids": [
                                conditional_config_id
                            ],
                            "repair_candidate_ids": [],
                            "successful_repair_candidate_ids": [],
                            "repair_parent_decisions": [],
                            "terminal_failure_exhausted": False,
                            "complete": True,
                        }
                        for value, lane_initial_ids, conditional_config_id in lane_specs
                    ],
                    "qualified_results": [
                        {
                            "config_id": config_id,
                            "attempt_perf": 1.0,
                            "selection_perf": 1.0,
                            "status": "ok",
                            "measurement_pass_index": 2,
                            "pipeline_lanes": [
                                {
                                    "key": "cute_flash_kv_stage",
                                    "value": 2 if index % 2 == 0 else 3,
                                }
                            ],
                        }
                        for index, config_id in enumerate(initial_config_ids)
                    ]
                    + [
                        {
                            "config_id": conditional_config_id,
                            "attempt_perf": 1.0,
                            "selection_perf": 1.0,
                            "status": "ok",
                            "measurement_pass_index": 2,
                            "pipeline_lanes": [
                                {
                                    "key": "cute_flash_kv_stage",
                                    "value": value,
                                }
                            ],
                        }
                        for value, _lane_initial_ids, conditional_config_id in lane_specs
                    ],
                    "retained_config_ids": retained_config_ids,
                    "complete": True,
                }
            ],
            "qualification_rounds": 2,
            "qualification_rounds_started": 2,
            "qualification_rounds_completed": 2,
            "qualification_passes_planned": 2,
            "qualification_passes_started": 2,
            "qualification_passes_completed": 2,
            "measurement_timeline": [
                {
                    "pass_index": 0,
                    "updates": [
                        measurement_update(config_id)
                        for config_id in sorted(initial_config_ids)
                    ],
                },
                {"pass_index": 1, "updates": []},
                {
                    "pass_index": 2,
                    "updates": [
                        measurement_update(config_id)
                        for config_id in sorted(conditional_config_ids)
                    ],
                },
            ],
            "budget_exhausted": False,
            "schedule_anchor_design_source": (
                "live family x ordinary packet x softmax protocol from fragment defaults"
            ),
            "schedule_anchor_pass_planned": False,
            "schedule_anchor_pass_started": False,
            "schedule_anchor_count": 0,
            "schedule_anchor_complete": True,
            "schedule_anchor_results": [],
            "pipeline_candidate_limit_per_leaf_per_round": 4,
            "conditional_candidates_per_pipeline_lane": 1,
            "qualification_failure_retries": 1,
            "family_probe_generations": 1,
            "family_probe_generations_started": 0,
            "family_probe_generations_completed": 0,
            "family_probe_candidates_per_path": 20,
            "family_probe_required": False,
            "family_probe_complete": True,
            "family_probe_path_limit": 0,
            "family_probe_paths": [],
            "neighbor_generation_limit_per_leaf_per_round": 200,
            "candidate_count": 2,
            "leaves_with_candidates": 1,
            "retained_candidates_per_leaf": 2,
            "retained_family_cap": 4,
            "retained_family_limit": 1,
            "retained_family_slowdown_limit": 2.0,
            "clc_families": [],
            "compound_catalog_complete": True,
            "compound_catalog_errors": [],
            "compound_transfers": [],
            "starting_path_limit": 14,
            "maximum_path_capacity": 14,
            "unrestricted_path_exhausts_generation_budget": True,
            "retained_families": [
                {
                    "family": "fa4_2cta",
                    "score": 1.0,
                    "score_compound_packet": None,
                    "score_softmax_disc": False,
                    "parent_promoted": True,
                    "starting_paths": [
                        {
                            "family": "fa4_2cta",
                            "compound_packet": None,
                            "softmax_disc": False,
                            "config_id": config_id,
                            "unrestricted": index == 2,
                            "pipeline_lane": (
                                None
                                if index != 1
                                else {
                                    "key": "cute_flash_kv_stage",
                                    "value": retained_lane_values[1],
                                }
                            ),
                        }
                        for index, config_id in enumerate(
                            [*retained_config_ids, best_config_id]
                        )
                    ],
                }
            ],
            "retained_path_count": len(retained_config_ids) + 1,
            "terminal_coordinate_refinement": terminal_refinement,
        },
    }
    trial.update(overrides)
    if "search_phase_metrics" not in overrides:
        _synchronize_full_autotune_terminal_boundary(trial)
    _populate_measurement_source_hashes(trial.get("search_phase_metrics"))
    return trial


def _full_autotune_trial_with_isolated_rebenchmark_invalidation(
    status, invalidated_id=None
):
    trial = _full_autotune_trial(
        num_isolated_rebenchmark_timeouts=int(status == "timeout")
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    retained_ids = set(cast("list[str]", leaf["retained_config_ids"]))
    invalidated = next(
        result
        for result in cast("list[dict[str, Any]]", leaf["qualified_results"])
        if result["config_id"] not in retained_ids
        and (invalidated_id is None or result["config_id"] == invalidated_id)
    )
    invalidated.update(
        attempt_perf=None,
        selection_perf=None,
        status=status,
    )
    cast("list[dict[str, Any]]", phase["measurement_timeline"][2]["updates"]).append(
        {
            key: invalidated[key]
            for key in (
                "config_id",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
            )
        }
    )
    phase["measurement_timeline"][2]["updates"].sort(
        key=operator.itemgetter("config_id")
    )
    return trial, invalidated["config_id"]


def _replace_measurement_source_hash(value, config_id, source_hash):
    if isinstance(value, dict):
        if value.get("config_id") == config_id and "source_hash" in value:
            value["source_hash"] = source_hash
        for child in value.values():
            _replace_measurement_source_hash(child, config_id, source_hash)
    elif isinstance(value, list):
        for child in value:
            _replace_measurement_source_hash(child, config_id, source_hash)


def _add_isolated_rebenchmark_alias_invalidation(trial, invalidated_id, *, complete):
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    retained_ids = set(cast("list[str]", leaf["retained_config_ids"]))
    qualified = cast("list[dict[str, Any]]", leaf["qualified_results"])
    alias = next(
        result
        for result in qualified
        if result["config_id"] != invalidated_id
        and result["config_id"] not in retained_ids
    )
    invalidated_update = next(
        item
        for item in phase["measurement_timeline"][2]["updates"]
        if item["config_id"] == invalidated_id
    )
    source_hash = invalidated_update["source_hash"]
    _replace_measurement_source_hash(phase, alias["config_id"], source_hash)
    if complete:
        alias.update(
            attempt_perf=None,
            selection_perf=None,
            status=invalidated_update["status"],
        )
        phase["measurement_timeline"][2]["updates"].append(
            {
                key: alias[key]
                for key in (
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                )
            }
        )
        phase["measurement_timeline"][2]["updates"].sort(
            key=operator.itemgetter("config_id")
        )
    return alias["config_id"]


def _full_autotune_trial_provenance(**overrides):
    design_configs = [
        copy.deepcopy(config) for config in _full_autotune_trial_configs()[:2]
    ]
    design = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for config in design_configs
    ]
    terminal_surface = copy.deepcopy(_full_autotune_terminal_surface_catalog())
    _terminal_refinement, selected_config = _full_autotune_terminal_fixture()
    provenance = _full_autotune_provenance(
        flash_structural_coverage_design=design,
        flash_structural_coverage_design_count=len(design),
        flash_structural_coverage_design_sha256=hashlib.sha256(
            json.dumps(design_configs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        flash_structural_injected_design_count=len(design),
        flash_structural_coverage_active_values=[
            {"key": "cute_flash_topology", "value": "fa4"},
            {"key": "cute_flash_kv_stage", "value": 2},
            {"key": "cute_flash_kv_stage", "value": 3},
        ],
        flash_structural_leaf_catalog=[
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            }
        ],
        flash_pipeline_lane_catalog=[
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
                "pipeline_lanes": [
                    {"key": "cute_flash_kv_stage", "value": 2},
                    {"key": "cute_flash_kv_stage", "value": 3},
                ],
            }
        ],
        flash_terminal_coordinate_surface_catalog=terminal_surface,
        flash_terminal_coordinate_surface_catalog_sha256=hashlib.sha256(
            json.dumps(terminal_surface, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        selected_config=copy.deepcopy(selected_config),
    )
    provenance.update(overrides)
    return provenance


def _full_autotune_trial_with_complete_schedule_anchors():
    from helion.autotuner.config_generation import ConfigGeneration

    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    generation = ConfigGeneration(_full_autotune_config_spec())
    anchors = generation.flash_low_confound_schedule_anchor_configs()
    assert len(anchors) == 131

    def config_id(config):
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    anchor_configs = [dict(anchor.config) for anchor in anchors]
    anchor_ids = [config_id(config) for config in anchor_configs]
    perf_by_id = {
        anchor_id: 1.0 + index / 1000 for index, anchor_id in enumerate(anchor_ids)
    }
    leaf_by_id = {
        anchor_id: compare_attention_backends._flash_structural_leaf_dict(config)
        for anchor_id, config in zip(anchor_ids, anchor_configs, strict=True)
    }
    leaf_catalog: list[dict[str, object]] = []
    anchor_ids_by_leaf: dict[str, list[str]] = {}
    for anchor_id in anchor_ids:
        leaf = leaf_by_id[anchor_id]
        assert leaf is not None
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        if leaf not in leaf_catalog:
            leaf_catalog.append(leaf)
        anchor_ids_by_leaf.setdefault(leaf_key, []).append(anchor_id)
    assert len(leaf_catalog) == 27
    ordinary_widths: dict[str, int] = {}
    for leaf in leaf_catalog:
        if leaf["compound_packet"] is None:
            family = cast("str", leaf["family"])
            ordinary_widths[family] = ordinary_widths.get(family, 0) + 1
    retained_family_cap = 4
    retained_family_limit = min(retained_family_cap, len(ordinary_widths))
    promoted_protocol_count = sum(
        sorted(ordinary_widths.values(), reverse=True)[:retained_family_limit]
    )
    starting_path_limit = max(
        14,
        1 + promoted_protocol_count + retained_family_limit,
    )
    family_probe_path_limit = len(ordinary_widths) + 1
    maximum_path_capacity = max(starting_path_limit, family_probe_path_limit)

    design_ids: list[str] = []
    for leaf in leaf_catalog:
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        design_ids.extend(anchor_ids_by_leaf[leaf_key][:2])
    design_configs = [anchor_configs[anchor_ids.index(item)] for item in design_ids]
    design = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for config in design_configs
    ]
    initial_ids = [
        *design_ids,
        *(anchor_id for anchor_id in anchor_ids if anchor_id not in design_ids),
    ][:100]
    initial_id_set = set(initial_ids)
    new_anchor_ids = [
        anchor_id for anchor_id in anchor_ids if anchor_id not in initial_id_set
    ]
    assert len(new_anchor_ids) == 31

    def measurement_state(config_id_value: str) -> dict[str, object]:
        perf = perf_by_id[config_id_value]
        return {
            "config_id": config_id_value,
            "attempt_perf": perf,
            "selection_perf": perf,
            "status": "ok",
        }

    def measurement(config_id_value: str, pass_index: int) -> dict[str, object]:
        return {
            **measurement_state(config_id_value),
            "measurement_pass_index": pass_index,
        }

    leaf_results = []
    qualified_for_retention = []
    for leaf in leaf_catalog:
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        leaf_anchor_ids = anchor_ids_by_leaf[leaf_key]
        ranked_ids = sorted(leaf_anchor_ids, key=lambda item: (perf_by_id[item], item))
        rounds = []
        for pass_index in (1, 2):
            rounds.append(
                {
                    "candidate_config_ids": [],
                    "neighbor_generation_limit": 200,
                    "ordinary_neighbor_generation_limit": 200,
                    "parent_decisions": [
                        {
                            "job_index": 0,
                            "kind": "ordinary",
                            "pipeline_lane": None,
                            "selection_kind": "ranked_parent",
                            "candidate_results": [
                                measurement(anchor_id, pass_index)
                                for anchor_id in ranked_ids
                            ],
                            "selected_config_id": ranked_ids[0],
                            "generated_config_ids": [],
                        }
                    ],
                }
            )
        leaf_results.append(
            {
                **leaf,
                "initial_config_ids": [
                    anchor_id
                    for anchor_id in initial_ids
                    if leaf_by_id[anchor_id] == leaf
                ],
                "space_exhausted": False,
                "space_config_count": None,
                "ordinary_search_required": True,
                "rounds": rounds,
                "pipeline_lanes": [],
                "qualified_results": [
                    {
                        **measurement(anchor_id, 4),
                        "pipeline_lanes": [],
                    }
                    for anchor_id in leaf_anchor_ids
                ],
                "retained_config_ids": ranked_ids[:2],
                "complete": True,
            }
        )
        qualified_for_retention.append(
            (
                leaf["family"],
                None,
                leaf["softmax_disc"],
                [
                    (anchor_id, perf_by_id[anchor_id], frozenset())
                    for anchor_id in leaf_anchor_ids
                ],
                (),
            )
        )

    active_values = [
        {
            "key": "cute_flash_pipeline_family",
            "value": leaf["family"],
        }
        for leaf in leaf_catalog
        if leaf["softmax_disc"] is True
    ]
    family_probe_starts = sorted(
        (
            min(
                (
                    anchor_id
                    for anchor_id in anchor_ids
                    if leaf_by_id[anchor_id]["family"] == family
                ),
                key=lambda item: (perf_by_id[item], item),
            )
            for family in ordinary_widths
        ),
        key=lambda item: (
            perf_by_id[item],
            item,
            leaf_by_id[item]["family"],
        ),
    )
    global_probe_start = min(anchor_ids, key=lambda item: (perf_by_id[item], item))
    family_probe_paths = [
        {
            **leaf_by_id[anchor_id],
            "starting_config_id": anchor_id,
            "unrestricted": False,
            "rounds": [
                {
                    "probe_generation": 1,
                    "measurement_pass_index": 4,
                    "candidate_ids": [],
                    "results": [],
                }
            ],
        }
        for anchor_id in family_probe_starts
    ]
    family_probe_paths.append(
        {
            **leaf_by_id[global_probe_start],
            "starting_config_id": global_probe_start,
            "unrestricted": True,
            "rounds": [
                {
                    "probe_generation": 1,
                    "measurement_pass_index": 4,
                    "candidate_ids": [],
                    "results": [],
                }
            ],
        }
    )
    assert len(family_probe_paths) == family_probe_path_limit
    provenance.update(
        flash_structural_coverage_design=design,
        flash_structural_coverage_design_count=len(design),
        flash_structural_coverage_design_sha256=hashlib.sha256(
            json.dumps(design_configs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        flash_structural_injected_design_count=min(50, len(design)),
        flash_structural_coverage_active_values=active_values,
        flash_structural_leaf_catalog=leaf_catalog,
        flash_pipeline_lane_catalog=[
            {**leaf, "pipeline_lanes": []} for leaf in leaf_catalog
        ],
        flash_clc_lane_catalog=[],
        flash_clc_lane_catalog_sha256=hashlib.sha256(b"[]").hexdigest(),
        flash_structural_qualification_values=active_values,
        flash_structural_parent_coverage_prefix_count=len(leaf_catalog),
        flash_structural_qualification_prefix_count=len(design),
        flash_structural_population_budget=50,
        flash_structural_retained_family_cap=retained_family_cap,
        flash_structural_retained_family_limit=retained_family_limit,
        flash_structural_starting_path_limit=starting_path_limit,
        flash_structural_family_probe_path_limit=family_probe_path_limit,
        flash_structural_maximum_path_capacity=maximum_path_capacity,
    )
    phase.update(
        schedule_anchor_pass_planned=True,
        schedule_anchor_pass_started=True,
        schedule_anchor_count=len(anchor_ids),
        schedule_anchor_complete=True,
        schedule_anchor_results=[
            {
                **leaf_by_id[anchor_id],
                **measurement(anchor_id, 1),
            }
            for anchor_id in anchor_ids
        ],
        initial_config_count=len(initial_ids),
        initial_config_ids=initial_ids,
        config_manifest={
            anchor_id: {"config": config}
            for anchor_id, config in zip(anchor_ids, anchor_configs, strict=True)
        },
        initial_results=[
            {
                **leaf_by_id[anchor_id],
                **measurement(anchor_id, 0),
                "pipeline_lanes": [],
            }
            for anchor_id in initial_ids
        ],
        leaf_count=len(leaf_catalog),
        ordinary_leaf_count=len(leaf_catalog),
        compound_leaf_count=0,
        leaf_results=leaf_results,
        qualification_rounds_started=4,
        qualification_rounds_completed=4,
        qualification_passes_planned=4,
        qualification_passes_started=4,
        qualification_passes_completed=4,
        measurement_timeline=[
            {
                "pass_index": 0,
                "updates": [
                    measurement_state(anchor_id) for anchor_id in sorted(initial_ids)
                ],
            },
            {
                "pass_index": 1,
                "updates": [
                    measurement_state(anchor_id) for anchor_id in sorted(new_anchor_ids)
                ],
            },
            {"pass_index": 2, "updates": []},
            {"pass_index": 3, "updates": []},
            {"pass_index": 4, "updates": []},
        ],
        candidate_count=len(new_anchor_ids),
        leaves_with_candidates=len(
            {
                json.dumps(leaf_by_id[anchor_id], sort_keys=True, separators=(",", ":"))
                for anchor_id in new_anchor_ids
            }
        ),
        clc_families=[],
        compound_transfers=[],
        family_probe_generations_started=1,
        family_probe_generations_completed=1,
        family_probe_required=True,
        family_probe_complete=True,
        family_probe_path_limit=family_probe_path_limit,
        family_probe_paths=family_probe_paths,
        retained_family_cap=retained_family_cap,
        retained_family_limit=retained_family_limit,
        starting_path_limit=starting_path_limit,
        maximum_path_capacity=maximum_path_capacity,
        retained_families=(
            compare_attention_backends._expected_flash_structural_retention(
                qualified_for_retention,
                retained_per_leaf=2,
                retained_family_cap=retained_family_cap,
                retained_family_limit=retained_family_limit,
                retained_family_slowdown_limit=2.0,
                starting_path_limit=starting_path_limit,
                pipeline_qualification_keys=(
                    "cute_flash_kv_stage",
                    "cute_flash_s_stage",
                ),
            )
        ),
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    trial.update(
        num_configs_tested=len(anchor_ids),
        num_successful_candidate_measurements=len(anchor_ids),
        num_unique_sources=len(anchor_ids),
    )
    _restore_full_autotune_terminal_fixture(trial, provenance)
    _populate_measurement_source_hashes(phase, overwrite=True)
    return trial, provenance, anchor_ids


def _full_autotune_trial_with_family_probe_candidate():
    from helion.autotuner.config_generation import ConfigGeneration

    trial, provenance, anchor_ids = (
        _full_autotune_trial_with_complete_schedule_anchors()
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    probe_path = cast("dict[str, Any]", phase["family_probe_paths"][0])
    probe_leaf = {
        key: probe_path[key] for key in ("family", "compound_packet", "softmax_disc")
    }
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])
    generation = ConfigGeneration(_full_autotune_config_spec())
    state = random.getstate()
    random.seed(24680)
    try:
        candidates = generation.random_population(500)
    finally:
        random.setstate(state)
    candidate_config = next(
        dict(candidate.config)
        for candidate in candidates
        if compare_attention_backends._flash_structural_leaf_dict(candidate.config)
        == probe_leaf
        and hashlib.sha256(
            json.dumps(candidate.config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        not in manifest
    )
    candidate_id = hashlib.sha256(
        json.dumps(candidate_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    candidate_perf = 100.0
    measurement = {
        "config_id": candidate_id,
        "attempt_perf": candidate_perf,
        "selection_perf": candidate_perf,
        "status": "ok",
        "measurement_pass_index": 4,
    }
    manifest[candidate_id] = {"config": candidate_config}
    probe_round = cast("dict[str, Any]", probe_path["rounds"][0])
    probe_round["candidate_ids"] = [candidate_id]
    probe_round["results"] = [measurement.copy()]
    timeline_updates = cast(
        "list[dict[str, Any]]", phase["measurement_timeline"][4]["updates"]
    )
    timeline_updates.append(
        {
            key: measurement[key]
            for key in measurement
            if key != "measurement_pass_index"
        }
    )
    timeline_updates.sort(key=operator.itemgetter("config_id"))
    leaf_result = next(
        result
        for result in cast("list[dict[str, Any]]", phase["leaf_results"])
        if {key: result[key] for key in probe_leaf} == probe_leaf
    )
    cast("list[dict[str, Any]]", leaf_result["qualified_results"]).append(
        {**measurement, "pipeline_lanes": []}
    )
    phase["candidate_count"] += 1
    candidate_leaf_key = json.dumps(probe_leaf, sort_keys=True, separators=(",", ":"))
    existing_candidate_leaf_keys = {
        json.dumps(
            compare_attention_backends._flash_structural_leaf_dict(
                manifest[cast("str", update["config_id"])]["config"]
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        for update in cast(
            "list[dict[str, object]]", phase["measurement_timeline"][1]["updates"]
        )
    }
    if candidate_leaf_key not in existing_candidate_leaf_keys:
        phase["leaves_with_candidates"] += 1
    trial["num_configs_tested"] += 1
    trial["num_successful_candidate_measurements"] += 1
    trial["num_unique_sources"] += 1
    _synchronize_full_autotune_terminal_boundary(trial)
    _populate_measurement_source_hashes(phase, overwrite=True)
    return trial, provenance, [*anchor_ids, candidate_id]


@functools.lru_cache(maxsize=1)
def _cached_full_autotune_trial_with_pipeline_repair():
    from helion.autotuner.config_generation import ConfigGeneration

    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    for timeline_pass in phase["measurement_timeline"]:
        timeline_pass["updates"].sort(key=operator.itemgetter("config_id"))
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    lane = cast("dict[str, Any]", leaf["pipeline_lanes"][0])
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])

    generation = ConfigGeneration(
        _full_autotune_config_spec(), _flash_pipeline_family_override="fa4_2cta"
    )
    state = random.getstate()
    random.seed(67890)
    try:
        candidates = generation.random_population(300)
    finally:
        random.setstate(state)
    repair_config = None
    repair_id = None
    for candidate in candidates:
        projected = compare_attention_backends._canonical_flash_projection(
            generation,
            dict(candidate.config),
            {
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": lane["value"],
                "cute_flash_pipeline_family": "fa4_2cta",
                "cute_flash_softmax_disc": False,
            },
        )
        projected_id = hashlib.sha256(
            json.dumps(projected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        if (
            projected_id not in manifest
            and compare_attention_backends._flash_structural_leaf_dict(projected)
            == {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            }
        ):
            repair_config = projected
            repair_id = projected_id
            break
    assert repair_config is not None and repair_id is not None
    manifest[repair_id] = {"config": repair_config}

    witness_generation = ConfigGeneration(_full_autotune_config_spec())
    witness_config = next(
        config
        for (witness_leaf, key, value), config in (
            witness_generation.flash_pipeline_lane_witnesses().items()
        )
        if witness_leaf.pipeline_family == "fa4_2cta"
        and witness_leaf.compound_exp2_packet is None
        and witness_leaf.softmax_disc is False
        and key == lane["key"]
        and value == lane["value"]
    )
    witness_config_dict = dict(witness_config.config)
    witness_id = hashlib.sha256(
        json.dumps(witness_config_dict, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert manifest[witness_id]["config"] == witness_config_dict
    conditional_id = cast("str", lane["conditional_candidate_ids"][0])
    failed_ids = set(cast("list[str]", lane["initial_config_ids"])) | {
        witness_id,
        conditional_id,
    }
    lane["witness_config_id"] = witness_id
    lane["rounds"][0]["candidate_config_ids"] = [witness_id]
    witness_decision = next(
        decision
        for decision in leaf["rounds"][0]["parent_decisions"]
        if decision["pipeline_lane"]["value"] == lane["value"]
    )
    witness_decision.update(
        selection_kind="catalog_witness",
        candidate_results=[
            {
                "config_id": witness_id,
                "attempt_perf": None,
                "selection_perf": None,
                "status": "error",
                "measurement_pass_index": 0,
            }
        ],
        selected_config_id=witness_id,
        generated_config_ids=[],
    )
    conditional_decision = next(
        decision
        for decision in leaf["rounds"][1]["parent_decisions"]
        if decision["pipeline_lane"]["value"] == lane["value"]
    )
    conditional_decision["candidate_results"].sort(key=operator.itemgetter("config_id"))
    conditional_decision["selected_config_id"] = conditional_decision[
        "candidate_results"
    ][0]["config_id"]

    def mark_failures(value):
        if isinstance(value, dict):
            if value.get("config_id") in failed_ids and value.get("status") not in {
                None,
                "unknown",
            }:
                value.update(
                    attempt_perf=None,
                    selection_perf=None,
                    status="error",
                )
            for item in value.values():
                mark_failures(item)
        elif isinstance(value, list):
            for item in value:
                mark_failures(item)

    mark_failures(phase)
    failed_attempts = [
        {
            "config_id": config_id,
            "attempt_perf": None,
            "selection_perf": None,
            "status": "error",
            "measurement_pass_index": 2,
        }
        for config_id in sorted((witness_id, conditional_id))
    ]
    lane_decision = {
        "repair_index": 0,
        "candidate_results": failed_attempts,
        "selected_config_id": failed_attempts[0]["config_id"],
        "generated_config_ids": [repair_id],
    }
    lane["witness_succeeded"] = False
    lane["successful_conditional_candidate_ids"] = []
    lane["repair_candidate_ids"] = [repair_id]
    lane["successful_repair_candidate_ids"] = [repair_id]
    lane["repair_parent_decisions"] = [lane_decision]
    for current_lane in cast("list[dict[str, Any]]", leaf["pipeline_lanes"]):
        current_lane["rounds"].append(
            {
                "candidate_config_ids": ([repair_id] if current_lane is lane else []),
                "neighbor_generation_limit": 200 if current_lane is lane else 0,
            }
        )
    leaf["rounds"].append(
        {
            "candidate_config_ids": [repair_id],
            "neighbor_generation_limit": 200,
            "ordinary_neighbor_generation_limit": 0,
            "parent_decisions": [
                {
                    "job_index": 0,
                    "kind": "failure_repair",
                    "pipeline_lane": {"key": lane["key"], "value": lane["value"]},
                    "selection_kind": "ranked_failed_parent",
                    **lane_decision,
                }
            ],
        }
    )
    leaf["qualified_results"].append(
        {
            "config_id": repair_id,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "measurement_pass_index": 3,
            "pipeline_lanes": [{"key": lane["key"], "value": lane["value"]}],
        }
    )
    witness_result = next(
        result
        for result in cast("list[dict[str, Any]]", leaf["qualified_results"])
        if result["config_id"] == witness_id
    )
    witness_result.update(
        attempt_perf=1.0,
        selection_perf=1.0,
        status="deduplicated",
    )
    lane["witness_succeeded"] = True

    pipeline_lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    successful = [
        (
            result["config_id"],
            result["selection_perf"],
            frozenset(
                (membership["key"], membership["value"])
                for membership in result["pipeline_lanes"]
            ),
        )
        for result in leaf["qualified_results"]
        if result["status"] in {"ok", "deduplicated"}
    ]
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        pipeline_lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [("fa4_2cta", None, False, successful, pipeline_lanes)],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    phase["candidate_count"] = 3
    for key in (
        "qualification_rounds_started",
        "qualification_rounds_completed",
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
    ):
        phase[key] = 3
    for qualified in leaf["qualified_results"]:
        qualified["measurement_pass_index"] = 3
    phase["measurement_timeline"].append(
        {
            "pass_index": 3,
            "updates": [
                {
                    "config_id": repair_id,
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                },
                {
                    "config_id": witness_id,
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "deduplicated",
                },
            ],
        }
    )
    phase["measurement_timeline"][-1]["updates"].sort(
        key=operator.itemgetter("config_id")
    )
    # The structural phase has 52 successes and 51 failures. Model 48 later
    # LFBO successes so the full non-exhaustive trial meets its 100-success gate.
    trial.update(
        num_configs_tested=151,
        num_worker_failures=len(failed_ids),
        num_successful_candidate_measurements=100,
        num_unique_sources=150,
        num_source_deduplications=1,
    )
    _synchronize_full_autotune_terminal_boundary(trial)
    _populate_measurement_source_hashes(phase)
    shared_source_hash = "f" * 64

    def share_repaired_source_hash(value) -> None:
        if isinstance(value, dict):
            if value.get("config_id") in {witness_id, repair_id} and value.get(
                "status"
            ) not in {None, "unknown"}:
                value["source_hash"] = shared_source_hash
            for child in value.values():
                share_repaired_source_hash(child)
        elif isinstance(value, list):
            for child in value:
                share_repaired_source_hash(child)

    share_repaired_source_hash(phase)
    return trial, provenance, witness_id


def _full_autotune_trial_with_pipeline_repair():
    return copy.deepcopy(_cached_full_autotune_trial_with_pipeline_repair())


def _full_autotune_trial_with_terminal_pipeline_failure():
    trial, provenance, witness_id = _full_autotune_trial_with_pipeline_repair()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    lane = cast("dict[str, Any]", leaf["pipeline_lanes"][0])
    conditional_id = cast("str", lane["conditional_candidate_ids"][0])
    repair_id = cast("str", lane["repair_candidate_ids"][0])
    terminal_failure_ids = {witness_id, conditional_id, repair_id}

    def mark_terminal_failures(value) -> None:
        if isinstance(value, dict):
            if value.get("config_id") in terminal_failure_ids and "status" in value:
                value.update(
                    attempt_perf=None,
                    selection_perf=None,
                    status="error",
                )
            for child in value.values():
                mark_terminal_failures(child)
        elif isinstance(value, list):
            for child in value:
                mark_terminal_failures(child)

    mark_terminal_failures(phase)
    final_updates = cast(
        "list[dict[str, Any]]", phase["measurement_timeline"][-1]["updates"]
    )
    phase["measurement_timeline"][-1]["updates"] = [
        update for update in final_updates if update["config_id"] != witness_id
    ]
    lane["witness_succeeded"] = False
    lane["successful_conditional_candidate_ids"] = []
    lane["successful_repair_candidate_ids"] = []
    lane["terminal_failure_exhausted"] = True

    pipeline_lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    successful = [
        (
            result["config_id"],
            result["selection_perf"],
            frozenset(
                (membership["key"], membership["value"])
                for membership in result["pipeline_lanes"]
            ),
        )
        for result in leaf["qualified_results"]
        if result["status"] in {"ok", "deduplicated"}
    ]
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        pipeline_lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [("fa4_2cta", None, False, successful, pipeline_lanes)],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    trial.update(
        num_worker_failures=trial["num_worker_failures"] + 1,
        num_unique_sources=trial["num_unique_sources"] + 1,
        num_source_deduplications=0,
    )
    _populate_measurement_source_hashes(phase, overwrite=True)
    return trial, provenance, witness_id, repair_id


def _full_autotune_trial_with_compound_transfer():
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.exc import InvalidConfig

    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    ordinary = cast("dict[str, Any]", phase["leaf_results"][0])
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])
    packet = "deg2_16x6"
    compound_leaf = {
        "family": "fa4_2cta",
        "compound_packet": packet,
        "softmax_disc": False,
    }
    provenance["flash_structural_coverage_active_values"].append(
        {"key": "cute_flash_exp2_packet", "value": packet}
    )
    provenance["flash_structural_qualification_values"] = [
        {"key": "cute_flash_exp2_packet", "value": packet}
    ]
    provenance["flash_structural_leaf_catalog"].append(compound_leaf)
    provenance["flash_pipeline_lane_catalog"].append(
        {**compound_leaf, "pipeline_lanes": []}
    )
    provenance["flash_structural_qualification_prefix_count"] = 2

    source_candidates = sorted(
        [
            {
                key: qualified[key]
                for key in (
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "measurement_pass_index",
                )
            }
            for qualified in ordinary["qualified_results"]
        ],
        key=operator.itemgetter("selection_perf", "config_id"),
    )
    generation = ConfigGeneration(_full_autotune_config_spec())
    attempted_ids: list[str] = []
    selected_ids: list[str] = []
    transfers: list[dict[str, Any]] = []
    projected_keys: set[str] = set()
    for candidate in source_candidates:
        source_id = candidate["config_id"]
        source_config = manifest[source_id]["config"]
        attempted_ids.append(source_id)
        try:
            projected = compare_attention_backends._canonical_flash_projection(
                generation,
                source_config,
                {"cute_flash_exp2_packet": packet},
            )
        except InvalidConfig:
            continue
        projected_key = json.dumps(projected, sort_keys=True, separators=(",", ":"))
        if (
            projected_key in projected_keys
            or compare_attention_backends._flash_structural_leaf_dict(projected)
            != compound_leaf
            or any(
                key in source_config and projected.get(key) != source_config[key]
                for key in ("cute_flash_kv_stage", "cute_flash_s_stage")
            )
        ):
            continue
        projected_keys.add(projected_key)
        target_id = hashlib.sha256(projected_key.encode()).hexdigest()[:16]
        manifest[target_id] = {"config": projected}
        selected_ids.append(source_id)
        transfers.append(
            {
                "source_config_id": source_id,
                "source_config": copy.deepcopy(source_config),
                "transferred_config_id": target_id,
                "projected_config": projected,
                "attempt_perf": 1.2,
                "selection_perf": 1.2,
                "status": "ok",
                "measurement_pass_index": 3,
                "projection_overrides": {"cute_flash_exp2_packet": packet},
                "projected_config_id": target_id,
                "preserved_pipeline_values": {
                    key: source_config[key]
                    for key in ("cute_flash_kv_stage", "cute_flash_s_stage")
                    if key in source_config
                },
            }
        )
        if len(transfers) == 2:
            break
    assert len(transfers) == 2
    phase["compound_transfers"] = [
        {
            **compound_leaf,
            "limit": 2,
            "transfer_target_count": len(transfers),
            "transfer_count": len(transfers),
            "primary_transfer_config_ids": [
                transfer["transferred_config_id"] for transfer in transfers
            ],
            "backfill_rounds": [],
            "successful_transfer_config_ids": [
                transfer["transferred_config_id"] for transfer in transfers
            ],
            "qualified_transfer_config_ids": [
                transfer["transferred_config_id"] for transfer in transfers
            ],
            "failure_statuses_allowed": True,
            "source_selection": {
                "candidate_results": source_candidates,
                "combination_prefix_count": 0,
                "attempted_config_ids": attempted_ids,
                "selected_config_ids": selected_ids,
            },
            "transfers": transfers,
            "complete": True,
        }
    ]
    phase["leaf_count"] = 2
    phase["compound_leaf_count"] = 1
    for field in (
        "qualification_rounds_started",
        "qualification_rounds_completed",
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
    ):
        phase[field] = 3
    for qualified in ordinary["qualified_results"]:
        qualified["measurement_pass_index"] = 3
    phase["measurement_timeline"].append(
        {
            "pass_index": 3,
            "updates": sorted(
                [
                    {
                        "config_id": transfer["transferred_config_id"],
                        "attempt_perf": transfer["attempt_perf"],
                        "selection_perf": transfer["selection_perf"],
                        "status": transfer["status"],
                    }
                    for transfer in transfers
                ],
                key=operator.itemgetter("config_id"),
            ),
        }
    )
    phase["candidate_count"] += len(transfers)
    phase["leaves_with_candidates"] += 1

    ordinary_members = [
        (
            qualified["config_id"],
            qualified["selection_perf"],
            frozenset(
                (lane["key"], lane["value"]) for lane in qualified["pipeline_lanes"]
            ),
        )
        for qualified in ordinary["qualified_results"]
    ]
    compound_members = [
        (transfer["transferred_config_id"], 1.2, frozenset()) for transfer in transfers
    ]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [
                (
                    "fa4_2cta",
                    None,
                    False,
                    ordinary_members,
                    (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3)),
                ),
                ("fa4_2cta", packet, False, compound_members, ()),
            ],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    for field in (
        "num_configs_tested",
        "num_successful_candidate_measurements",
        "num_unique_sources",
    ):
        trial[field] += len(transfers)
    _synchronize_full_autotune_terminal_boundary(trial)
    _populate_measurement_source_hashes(phase)
    return trial, provenance


def _full_autotune_config_spec():
    from helion._compiler.backend import CuteBackend
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec

    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=64,
        num_kv=512,
        num_bh=2,
        tensor_4d_heads=1,
        dtype=torch.float16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        standard_dense_output=True,
    )
    seeds = spec.autotune_seed_configs()
    assert seeds
    spec.compiler_seed_configs = list(seeds)
    spec.compiler_seed_timeout_retry_repetitions = 3
    spec.autotuner_heuristics = ["cute_flash_attention"]
    return spec


@functools.lru_cache(maxsize=1)
def _full_autotune_compiler_seed_policy():
    from helion.autotuner.config_generation import ConfigGeneration

    spec = _full_autotune_config_spec()
    return compare_attention_backends._compiler_seed_policy(
        spec, ConfigGeneration(spec)
    )


def test_attention_strict_schedule_anchor_enumerator_matches_live_product():
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(_full_autotune_config_spec())
    independent = compare_attention_backends._independent_flash_schedule_anchor_configs(
        generation
    )
    audited = compare_attention_backends._strict_flash_schedule_anchor_configs(
        generation,
        trial_index=0,
    )

    assert independent
    assert [config.config for config in audited] == [
        config.config for config in independent
    ]


def test_attention_strict_schedule_anchor_enumerator_rejects_producer_omission(
    monkeypatch,
):
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(_full_autotune_config_spec())
    produced = generation.flash_low_confound_schedule_anchor_configs()
    assert produced
    monkeypatch.setattr(
        generation,
        "flash_low_confound_schedule_anchor_configs",
        lambda: produced[:-1],
    )

    with pytest.raises(RuntimeError, match="differs from independent"):
        compare_attention_backends._strict_flash_schedule_anchor_configs(
            generation,
            trial_index=0,
        )


class _FixtureConfigGeneration:
    def __init__(self, delegate, provenance, trial):
        from helion._compiler.cute.cute_flash import FlashStructuralLeaf
        from helion.runtime.config import Config

        self.delegate = delegate
        self.provenance = provenance
        self.trial = trial
        self.Config = Config
        self._key_to_flat_indices = dict(delegate._key_to_flat_indices)
        self.leaves = [
            FlashStructuralLeaf(
                leaf["family"], leaf["compound_packet"], leaf["softmax_disc"]
            )
            for leaf in provenance.get("flash_structural_leaf_catalog", [])
        ]
        phase = trial["search_phase_metrics"]
        manifest = phase["config_manifest"]
        self.initial_population_configs = tuple(
            self.Config.from_dict(copy.deepcopy(manifest[config_id]["config"]))
            for config_id in phase["initial_config_ids"]
        )
        anchor_results = phase["schedule_anchor_results"]
        self.schedule_anchor_configs = [
            self.Config.from_dict(
                copy.deepcopy(manifest[result["config_id"]]["config"])
            )
            for result in anchor_results
        ]
        if not self.schedule_anchor_configs:
            # Compact legacy fixtures deliberately model a surface without the
            # required protocol axis. Production receives the real layout.
            self._key_to_flat_indices.pop("cute_flash_softmax_disc", None)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def random_population_flat(
        self, initial_population_size, *, user_seed_configs=(), log_func=None
    ):
        assert not user_seed_configs
        return [
            self.delegate.flatten(copy.deepcopy(config))
            for config in self.initial_population_configs
        ]

    def flash_deterministic_population_configs(self):
        return [
            self.Config.from_dict(item["config"])
            for item in self.provenance["flash_structural_coverage_design"]
        ]

    def flash_low_confound_schedule_anchor_configs(self):
        return list(self.schedule_anchor_configs)

    def flash_structural_leaf_catalog(self):
        return list(self.leaves)

    def flash_pipeline_lane_catalog(self):
        entries = self.provenance["flash_pipeline_lane_catalog"]
        return {
            leaf: tuple(
                (lane["key"], lane["value"])
                for lane in next(
                    entry
                    for entry in entries
                    if entry["family"] == leaf.pipeline_family
                    and entry["compound_packet"] == leaf.compound_exp2_packet
                    and entry["softmax_disc"] == leaf.softmax_disc
                )["pipeline_lanes"]
            )
            for leaf in self.leaves
        }

    def flash_structural_coverage_active_values(self):
        return [
            (item["key"], item["value"])
            for item in self.provenance["flash_structural_coverage_active_values"]
        ]

    def flash_structural_coverage_uncovered_values(self):
        return [
            (item["key"], item["value"])
            for item in self.provenance["flash_structural_coverage_uncovered_values"]
        ]

    def flash_structural_coverage_underqualified_values(self):
        return [
            (item["key"], item["value"], item["witness_count"])
            for item in self.provenance[
                "flash_structural_coverage_underqualified_values"
            ]
        ]

    def flash_structural_coverage_underqualified_leaves(self):
        by_key = {
            (
                leaf.pipeline_family,
                leaf.compound_exp2_packet,
                leaf.softmax_disc,
            ): leaf
            for leaf in self.leaves
        }
        return [
            (
                by_key[
                    (
                        item["family"],
                        item["compound_packet"],
                        item["softmax_disc"],
                    )
                ],
                item["witness_count"],
            )
            for item in self.provenance[
                "flash_structural_coverage_underqualified_leaves"
            ]
        ]

    def flash_structural_coverage_active_interactions(self):
        return [
            (tuple(item["keys"]), tuple(item["values"]))
            for item in self.provenance["flash_structural_coverage_active_interactions"]
        ]

    def flash_structural_coverage_uncovered_interactions(self):
        return [
            (tuple(item["keys"]), tuple(item["values"]))
            for item in self.provenance[
                "flash_structural_coverage_uncovered_interactions"
            ]
        ]

    def flash_structural_parent_coverage_prefix_count(self):
        return self.provenance["flash_structural_parent_coverage_prefix_count"]

    def flash_structural_qualification_prefix_count(self):
        return self.provenance["flash_structural_qualification_prefix_count"]

    def flash_structural_population_budget(self, initial_population_size):
        return self.provenance["flash_structural_population_budget"]

    def flash_clc_lane_catalog(self):
        by_key = {
            (
                leaf.pipeline_family,
                leaf.compound_exp2_packet,
                leaf.softmax_disc,
            ): leaf
            for leaf in self.leaves
        }
        return {
            by_key[(item["family"], item["compound_packet"], item["softmax_disc"])]: {
                "legal_values": tuple(item["legal_values"]),
                "search_values": tuple(item["search_values"]),
                "anchor_values": tuple(item["anchor_values"]),
                "refinement_values": tuple(item["refinement_values"]),
                "attempted_values": tuple(item["planned_values"]),
            }
            for item in self.provenance["flash_clc_lane_catalog"]
        }

    def flash_clc_lane_witnesses(self):
        by_key = {
            (
                leaf.pipeline_family,
                leaf.compound_exp2_packet,
                leaf.softmax_disc,
            ): leaf
            for leaf in self.leaves
        }
        manifest = self.trial["search_phase_metrics"]["config_manifest"]
        return {
            (
                by_key[
                    (
                        item["family"],
                        item["compound_packet"],
                        item["softmax_disc"],
                    )
                ],
                int(value),
            ): (self.Config.from_dict(manifest[config_id]["config"]))
            for item in self.provenance["flash_clc_lane_catalog"]
            for value, config_id in item["witness_config_ids"].items()
        }

    def flash_exact_effective_search_space_configs(self, initial_population_size):
        phase = self.trial["search_phase_metrics"]
        if not phase["exact_space_enumerated"]:
            return None
        manifest = phase["config_manifest"]
        return [
            self.Config.from_dict(manifest[config_id]["config"])
            for config_id in phase["exact_space_config_ids"]
        ]


@functools.lru_cache(maxsize=1)
def _cached_full_autotune_config_generation():
    from helion.autotuner.config_generation import ConfigGeneration

    return ConfigGeneration(_full_autotune_config_spec())


def _validate_full_autotune_trials(provenance, trials, *, expected_fixture_trial=None):
    from helion.runtime.config import Config

    phase = (trials[0].get("search_phase_metrics") or {}) if trials else {}
    leaf_catalog = provenance.get("flash_structural_leaf_catalog", [])
    clc_fixture = bool(
        leaf_catalog
        and isinstance(leaf_catalog[0], dict)
        and leaf_catalog[0].get("family") == "fa4_clc"
    )
    compound_fixture = any(
        isinstance(leaf, dict) and leaf.get("compound_packet") is not None
        for leaf in leaf_catalog
    )
    if clc_fixture and phase.get("exact_space_enumerated") is not True:
        expected_trial, expected_provenance, _reused_id = (
            _full_autotune_trial_with_reused_clc_combination()
        )
    elif clc_fixture:
        expected_trial, expected_provenance = (
            _exact_small_space_trial_with_exhausted_clc()
        )
    elif compound_fixture:
        expected_trial, expected_provenance = (
            _full_autotune_trial_with_compound_transfer()
        )
    elif phase.get("exact_space_enumerated") is True:
        expected_trial, expected_provenance, _ids = (
            _exact_small_space_trial_provenance()
        )
    else:
        expected_trial = _full_autotune_trial()
        expected_provenance = _full_autotune_trial_provenance()
    if expected_fixture_trial is not None:
        expected_trial = expected_fixture_trial
        expected_provenance = provenance
    config_spec = _full_autotune_config_spec()
    seed_phase = (
        phase
        if isinstance(phase.get("initial_results"), list)
        else expected_trial["search_phase_metrics"]
    )
    fixture_seed_ids = [
        result["config_id"]
        for result in reversed(seed_phase["initial_results"])
        if result["status"] in {"ok", "deduplicated"}
    ][:2]
    fixture_seeds = [
        Config.from_dict(seed_phase["config_manifest"][config_id]["config"])
        for config_id in fixture_seed_ids
    ]
    config_spec.compiler_seed_configs = fixture_seeds
    config_spec.compiler_seed_timeout_retry_repetitions = 3
    config_spec.autotuner_heuristics = ["cute_flash_attention"]
    config_spec.autotune_seed_configs = lambda: list(fixture_seeds)
    normalization_context = compare_attention_backends._flash_normalization_context(
        config_spec
    )
    provenance.setdefault("flash_normalization_context", normalization_context)
    provenance.setdefault(
        "flash_normalization_context_sha256",
        hashlib.sha256(
            json.dumps(
                normalization_context, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    )
    fixture_generation = _FixtureConfigGeneration(
        _cached_full_autotune_config_generation(), expected_provenance, expected_trial
    )
    compiler_seed_policy = compare_attention_backends._compiler_seed_policy(
        config_spec, fixture_generation
    )
    provenance["compiler_seed_config_count"] = compiler_seed_policy["raw_config_count"]
    provenance["compiler_seed_policy"] = compiler_seed_policy
    return compare_attention_backends._validate_required_full_autotune_trials(
        provenance,
        trials,
        config_spec=config_spec,
        expected_input_shapes=repr([(2, 32, 65536, 64)] * 3),
        expected_dtypes=repr(["torch.float16"] * 3),
        expected_hardware="NVIDIA B200",
        config_generation=fixture_generation,
    )


@pytest.mark.parametrize(
    "override",
    (
        {"effort": "quick"},
        {"physical_gpu_selection": "6,7"},
        {"strict_runtime_environment": None},
        {"helion_source_tree_sha256": None},
        {"helion_source_tree_file_count": 0},
        {"helion_source_tree_dirty": True},
        {"helion_checkout_git_commit": None},
        {"effective_force_autotune": False},
        {"fixed_config": True},
        {"autotune_budget_seconds": 60},
        {"autotune_max_generations": 10},
        {"autotune_best_of_k": 2},
        {"autotune_config_overrides": {"num_warps": 4}},
        {"user_seed_configs": True},
        {"disable_autotuner_heuristics": True},
        {"compiler_seed_config_count": 1},
        {"compiler_default_config": True},
        {"kernel_declared_config_count": 1},
        {"flash_structural_coverage_design": []},
        {
            "flash_structural_coverage_uncovered_values": [
                {"key": "cute_flash_wait_hint", "value": 0}
            ]
        },
        {
            "flash_structural_coverage_underqualified_values": [
                {
                    "key": "cute_flash_pipeline_family",
                    "value": "fa4_2cta",
                    "witness_count": 1,
                }
            ]
        },
        {
            "flash_structural_coverage_underqualified_leaves": [
                {
                    "family": "fa4_2cta",
                    "compound_packet": None,
                    "softmax_disc": False,
                    "witness_count": 1,
                }
            ]
        },
        {"flash_structural_leaf_catalog": []},
        {"flash_structural_qualification_rounds": 1},
        {
            "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 3
        },
        {"flash_structural_retained_candidates_per_leaf": 1},
        {"flash_structural_retained_family_cap": 3},
        {"flash_structural_retained_family_limit": 5},
        {"flash_structural_retained_family_slowdown_limit": 1.5},
        {"flash_structural_starting_path_limit": 4},
        {"flash_structural_unrestricted_path_exhausts_generation_budget": False},
        {"flash_terminal_coordinate_refinement_policy_sha256": "0" * 64},
        {"flash_terminal_coordinate_surface_catalog": {"schema_version": 1}},
        {"flash_terminal_coordinate_surface_catalog_sha256": "0" * 64},
        {
            "flash_structural_coverage_uncovered_interactions": [
                {
                    "keys": ["cute_flash_epi_stg_store"],
                    "values": ["whole"],
                }
            ]
        },
        {"flash_structural_coverage_uncovered_interactions": None},
        {"flash_structural_coverage_interaction_key_groups": []},
        {
            "flash_structural_coverage_active_interactions": [
                {"keys": ["cute_flash_topology"], "values": ["fa4"]}
            ]
        },
        {"flash_structural_parent_coverage_prefix_count": 2},
        {"flash_structural_qualification_prefix_count": 51},
        {"flash_structural_population_budget": 0},
        {"flash_structural_injected_design_count": 2},
        {"autotune_initial_population_size": 0},
        {"autotune_lfbo_max_generations": 19},
        {
            "flash_structural_coverage_active_values": [
                {"key": "cute_flash_wait_hint", "value": 0}
            ]
        },
        {"active_value_prior_keys": ["num_warps"]},
        {"flash_value_prior_keys": ["cute_flash_wait_hint"]},
        {"dense_d64_2cta_performance_anchor_present": True},
        {"cute_flash_env_overrides": {"HELION_CUTE_FLASH_WAIT_HINT": "0"}},
        {"autotune_initial_population_strategy_override": "from_best_available"},
        {"autotuner_initial_population_env": "from_best_available"},
        {"autotuner_env": "RandomSearch"},
        {"autotune_num_neighbors_cap_env": "1"},
        {"autotuner_fn_is_default": False},
        {"autotune_baseline_fn_is_expected": False},
        {"autotune_baseline_atol": 0.1},
        {"autotune_baseline_rtol": 0.1},
        {"autotune_baseline_accuracy_check_fn": True},
        {"autotune_benchmark_fn": True},
        {"autotune_rebenchmark_threshold": 0.1},
        {"autotune_suspicious_rebenchmark_ratio": 1.5},
        {"autotune_accuracy_check": False},
        {"autotune_compile_timeout": 1},
        {"autotune_benchmark_subprocess": False},
        {"autotune_benchmark_timeout": 1},
        {"autotune_adaptive_timeout": False},
        {"autotune_force_persistent": True},
        {"autotune_finishing_rounds_env": "2"},
        {"autotune_ignore_errors": True},
        {"autotune_search_acf": ["shape-specific.bin"]},
        {"autotune_config_filter": True},
        {"final_correctness_enabled": False},
        {"autotune_cache": "StrictLocalAutotuneCache"},
        {"rebenchmark_env_overrides": {"HELION_REBENCHMARK_THRESHOLD": "0"}},
    ),
)
def test_attention_required_full_autotune_rejects_partial_search(override):
    with pytest.raises(SystemExit, match="rejected this run"):
        compare_attention_backends._validate_required_full_autotune(
            _full_autotune_provenance(**override)
        )


def test_attention_required_full_autotune_requires_structural_design_count():
    provenance = _full_autotune_provenance()
    provenance.pop("flash_structural_coverage_design_count")

    with pytest.raises(SystemExit, match="rejected this run"):
        compare_attention_backends._validate_required_full_autotune(provenance)


def test_attention_required_full_autotune_accepts_unrestricted_search():
    compare_attention_backends._validate_required_full_autotune(
        _full_autotune_provenance()
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda policy: policy.update(schema_version=True),
        lambda policy: policy.update(schema_version=2),
        lambda policy: policy.update(kind="noncanonical"),
        lambda policy: policy.update(heuristic_names=["cute_flash_attention", "x"]),
        lambda policy: policy.update(raw_config_count=0),
        lambda policy: policy.update(effective_config_ids=[]),
        lambda policy: policy.update(effective_config_ids=[{}]),
        lambda policy: policy.update(effective_config_ids_sha256="0" * 64),
        lambda policy: policy.update(timeout_retry_repetitions=None),
    ),
)
def test_attention_required_full_autotune_rejects_noncanonical_seed_policy(mutate):
    provenance = _full_autotune_provenance()
    mutate(provenance["compiler_seed_policy"])

    with pytest.raises(SystemExit, match="compiler-seed policy"):
        compare_attention_backends._validate_required_full_autotune(provenance)


def test_attention_compiler_seed_policy_requires_exact_live_seed_order():
    from helion.autotuner.config_generation import ConfigGeneration

    spec = _full_autotune_config_spec()
    generation = ConfigGeneration(spec)
    policy = compare_attention_backends._compiler_seed_policy(spec, generation)

    assert policy["kind"] == "canonical_cute_flash"
    assert policy["heuristic_names"] == ["cute_flash_attention"]
    assert policy["raw_config_count"] > len(policy["effective_config_ids"])
    assert policy["timeout_retry_repetitions"] == 3

    spec.compiler_seed_configs.reverse()
    reordered = compare_attention_backends._compiler_seed_policy(spec, generation)
    assert reordered["kind"] == "noncanonical"


def test_attention_compiler_seed_policy_rejects_extra_alias_and_invalid_seed():
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.runtime.config import Config

    spec = _full_autotune_config_spec()
    generation = ConfigGeneration(spec)
    canonical_ids = compare_attention_backends._compiler_seed_policy(spec, generation)[
        "effective_config_ids"
    ]

    spec.compiler_seed_configs.append(copy.deepcopy(spec.compiler_seed_configs[0]))
    aliased = compare_attention_backends._compiler_seed_policy(spec, generation)
    assert aliased["kind"] == "noncanonical"
    assert aliased["effective_config_ids"] == canonical_ids

    ids, invalid_count = compare_attention_backends._ordered_effective_config_ids(
        generation,
        [Config.from_dict({"block_sizes": "invalid"})],
    )
    assert ids == []
    assert invalid_count == 1


@pytest.mark.parametrize(
    ("causal", "seq_len"),
    (
        *((False, seq_len) for seq_len in (32768, 65536, 131072, 262144)),
        *((True, seq_len) for seq_len in (65536, 131072, 262144, 524288)),
    ),
)
def test_attention_canonical_compiler_seeds_fit_all_eight_b200_populations(
    causal, seq_len
):
    from helion._compiler.backend import CuteBackend
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec

    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 0),
        num_sm=148,
    )
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=64,
        num_kv=seq_len // 128,
        num_bh=64,
        tensor_4d_heads=32,
        dtype=torch.float16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
    )
    spec.compiler_seed_configs = spec.autotune_seed_configs()
    spec.compiler_seed_timeout_retry_repetitions = 3
    spec.autotuner_heuristics = ["cute_flash_attention"]
    generation = ConfigGeneration(spec)

    policy = compare_attention_backends._compiler_seed_policy(spec, generation)
    generation_zero_ids = compare_attention_backends._replay_strict_attention_initial_population_config_ids(
        generation,
        random_seed=123,
        initial_population_size=100,
    )

    assert policy["kind"] == "canonical_cute_flash"
    assert policy["raw_config_count"] == (9 if causal else 26)
    assert len(policy["effective_config_ids"]) == (9 if causal else 25)
    assert len(generation_zero_ids) == 100
    assert set(policy["effective_config_ids"]) <= set(generation_zero_ids)


@pytest.mark.parametrize(
    ("kernel_name", "shape", "expected_raw_count", "expected_effective_count"),
    (
        ("attention_output", (2, 32, 262144, 128), 7, 7),
        ("causal_attention_output", (2, 32, 524288, 128), 2, 2),
        ("attention_output", (1, 32, 524288, 64), 15, 14),
        ("causal_attention_output", (1, 32, 1048576, 64), 7, 7),
        ("attention_output", (8, 32, 524288, 64), 1, 1),
        ("causal_attention_output", (8, 32, 786432, 64), 6, 6),
        ("attention_relu_output", (2, 32, 524288, 64), 15, 14),
        ("causal_attention_relu_output", (2, 32, 1048576, 64), 6, 6),
        ("causal_attention_output", (2, 32, 65536, 64), 7, 7),
    ),
)
@skipUnlessCuteAvailable("binding a cute-backend kernel requires the CuTe DSL")
def test_attention_bound_varied_b200_compiler_seed_order_and_population(
    monkeypatch,
    kernel_name,
    shape,
    expected_raw_count,
    expected_effective_count,
):
    from torch._inductor.runtime.hints import DeviceProperties
    from torch._subclasses.fake_tensor import FakeTensorMode

    import helion
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config

    attention_example = importlib.import_module("examples.attention")
    compile_environment = importlib.import_module(
        "helion._compiler.compile_environment"
    )
    kernel_module = importlib.import_module("helion.runtime.kernel")
    runtime_module = importlib.import_module("helion.runtime")
    source_kernel = getattr(attention_example, kernel_name)
    kernel = helion.kernel(source_kernel.fn, static_shapes=True, backend="cute")
    fake_device_properties = DeviceProperties(
        type="cuda",
        index=0,
        multi_processor_count=148,
        cc=100,
        major=10,
        regs_per_multiprocessor=65536,
        max_threads_per_multi_processor=2048,
        max_threads_per_block=1024,
        warp_size=32,
    )
    fake_cuda_properties = SimpleNamespace(
        name="NVIDIA B200",
        major=10,
        minor=0,
        multi_processor_count=148,
        regs_per_multiprocessor=65536,
        max_threads_per_multi_processor=2048,
        max_threads_per_block=1024,
        warp_size=32,
        total_memory=192 * 1024**3,
    )
    monkeypatch.setattr(
        DeviceProperties,
        "create",
        classmethod(lambda _cls, _device: fake_device_properties),
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device=None: (10, 0)
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device=None: fake_cuda_properties
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        kernel_module, "target_device_capability", lambda _device: (10, 0)
    )
    monkeypatch.setattr(
        compile_environment,
        "target_device_capability",
        lambda _device: (10, 0),
    )
    monkeypatch.setattr(
        runtime_module, "get_num_sm", lambda _device, reserved_sms=0: 148
    )

    with FakeTensorMode():
        q = torch.empty(
            shape,
            device="cuda",  # @ignore-device-lint
            dtype=torch.bfloat16,
        )
        bound = kernel.bind((q, q, q))

    spec = bound.config_spec
    regenerated_seeds = spec.autotune_seed_configs()
    assert [dict(config.config) for config in spec.compiler_seed_configs] == [
        dict(config.config) for config in regenerated_seeds
    ]

    generation = spec.create_config_generation()
    policy = compare_attention_backends._compiler_seed_policy(spec, generation)
    generation_zero_ids = compare_attention_backends._replay_strict_attention_initial_population_config_ids(
        generation,
        random_seed=123,
        initial_population_size=100,
    )

    assert policy["kind"] == "canonical_cute_flash"
    assert policy["raw_config_count"] == expected_raw_count
    assert len(policy["effective_config_ids"]) == expected_effective_count
    assert len(generation_zero_ids) == 100
    assert set(policy["effective_config_ids"]) <= set(generation_zero_ids)

    base = generation.flash_deterministic_population_configs()[0]
    leaf = flash_structural_leaf_from_config(base.config)
    assert leaf is not None
    overrides = {
        FLASH_PIPELINE_FAMILY_KEY: leaf.pipeline_family,
        FLASH_SOFTMAX_DISC_KEY: leaf.softmax_disc,
    }
    if leaf.compound_exp2_packet is not None:
        overrides[FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
    leaf_generation = spec.create_config_generation(overrides=overrides)
    raw_projections = leaf_generation.coordinate_neighbor_projections(
        leaf_generation.flatten(base), radius=2
    )
    projections = generation.canonicalize_coordinate_projections(
        raw_projections,
        base_config=base,
    )
    for projection in projections:
        if projection.config is not None:
            _, canonical = generation.canonicalize_flat(
                generation.flatten(projection.config)
            )
            assert projection.config == canonical
    if kernel_name == "causal_attention_output" and shape == (2, 32, 65536, 64):
        assert any(
            raw.config is not None
            and raw.config.config.get("cute_vector_widths") == [1, 1, 1]
            and normalized.config is not None
            and "cute_vector_widths" not in normalized.config.config
            for raw, normalized in zip(raw_projections, projections, strict=True)
        )


def test_attention_compiler_seed_generation_zero_requires_terminal_evidence():
    seed_ids = ["a" * 16, "b" * 16]
    policy = {"effective_config_ids": seed_ids}
    records = [
        {
            "config_id": seed_ids[0],
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "source_hash": "c" * 64,
            "measurement_pass_index": 0,
        },
        {
            "config_id": seed_ids[1],
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "deduplicated",
            "source_hash": "d" * 64,
            "measurement_pass_index": 0,
        },
    ]
    compare_attention_backends._validate_compiler_seed_generation_zero(
        policy, records, trial_index=1
    )

    with pytest.raises(RuntimeError, match="compiler seed in pass 0"):
        compare_attention_backends._validate_compiler_seed_generation_zero(
            policy,
            records,
            trial_index=1,
            invalidated_config_ids={seed_ids[0]},
        )

    for key, value in (
        ("status", "error"),
        ("attempt_perf", None),
        ("selection_perf", float("inf")),
        ("source_hash", None),
        ("measurement_pass_index", 1),
    ):
        invalid = copy.deepcopy(records)
        invalid[0][key] = value
        with pytest.raises(RuntimeError, match="compiler seed in pass 0"):
            compare_attention_backends._validate_compiler_seed_generation_zero(
                policy, invalid, trial_index=1
            )


def test_attention_required_full_autotune_rejects_missing_unrestricted_path_policy():
    provenance = _full_autotune_provenance()
    provenance.pop("flash_structural_unrestricted_path_exhausts_generation_budget")

    with pytest.raises(SystemExit, match="rejected this run"):
        compare_attention_backends._validate_required_full_autotune(provenance)


def test_attention_required_full_autotune_rejects_missing_lfbo_generation_budget():
    provenance = _full_autotune_provenance()
    provenance.pop("autotune_lfbo_max_generations")

    with pytest.raises(SystemExit, match="rejected this run"):
        compare_attention_backends._validate_required_full_autotune(provenance)


def test_attention_required_full_autotune_requires_injected_prefix_coverage():
    configs = [
        {"cute_flash_wait_hint": 1},
        {"cute_flash_wait_hint": 0},
    ]
    design = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        for config in configs
    ]
    provenance = _full_autotune_provenance(
        flash_structural_coverage_design=design,
        flash_structural_coverage_design_count=len(design),
        flash_structural_coverage_design_sha256=hashlib.sha256(
            json.dumps(configs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        flash_structural_coverage_active_values=[
            {"key": "cute_flash_wait_hint", "value": 0}
        ],
        flash_structural_population_budget=1,
        flash_structural_injected_design_count=1,
    )

    with pytest.raises(SystemExit, match="injected structural design does not cover"):
        compare_attention_backends._validate_required_full_autotune(provenance)


def test_attention_required_full_autotune_allows_anchor_qualification_after_injection():
    provenance = _full_autotune_provenance()
    first = provenance["flash_structural_coverage_design"][0]
    second_config = {
        **first["config"],
        "cute_flash_wait_hint": 0,
    }
    second = {
        "config": second_config,
        "config_sha256": hashlib.sha256(
            json.dumps(second_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    design = [first, second]
    configs = [candidate["config"] for candidate in design]
    provenance.update(
        {
            "flash_structural_coverage_design": design,
            "flash_structural_coverage_design_count": len(design),
            "flash_structural_coverage_design_sha256": hashlib.sha256(
                json.dumps(configs, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "flash_structural_qualification_prefix_count": 2,
            "flash_structural_population_budget": 1,
            "flash_structural_injected_design_count": 1,
        }
    )

    compare_attention_backends._validate_required_full_autotune(provenance)


def test_attention_required_full_autotune_rejects_default_name_spoof():
    with pytest.raises(SystemExit, match="custom autotuner function"):
        compare_attention_backends._validate_required_full_autotune(
            _full_autotune_provenance(
                autotuner_fn="test.default_autotuner_fn",
                autotuner_fn_is_default=False,
            )
        )


def test_attention_autotune_provenance_records_effective_search(monkeypatch):
    from helion._compiler.backend import CuteBackend
    from helion._compiler.cute import cute_flash
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec

    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    monkeypatch.setenv("HELION_AUTOTUNER_INITIAL_POPULATION", "from_random")
    monkeypatch.setenv("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", "-1")
    monkeypatch.delenv("HELION_AUTOTUNER", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_FINISHING_ROUNDS", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS", raising=False)
    args = _attention_subprocess_args(
        helion_force_autotune=1, helion_require_full_autotune=1
    )

    def make_spec(*, is_causal: bool) -> ConfigSpec:
        spec = ConfigSpec(backend=CuteBackend())
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=256,
            num_bh=64,
            dtype=torch.float16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
        )
        seeds = spec.autotune_seed_configs()
        assert seeds
        spec.compiler_seed_configs = list(seeds)
        spec.compiler_seed_timeout_retry_repetitions = 3
        spec.autotuner_heuristics = ["cute_flash_attention"]
        return spec

    dense_spec = make_spec(is_causal=False)
    fragment_default = dict(dense_spec.default_config().config)
    bound = SimpleNamespace(
        settings=SimpleNamespace(
            force_autotune=False,
            autotune_effort="full",
            autotune_budget_seconds=None,
            autotune_max_generations=None,
            autotune_best_of_k=2,
            autotune_accuracy_check=True,
            autotune_compile_timeout=60,
            autotune_benchmark_subprocess=True,
            autotune_benchmark_timeout=30,
            autotune_adaptive_timeout=True,
            autotune_force_persistent=False,
            autotune_ignore_errors=False,
            autotune_random_seed=123,
            autotune_cache="LocalAutotuneCache",
            disable_autotuner_heuristics=False,
            autotune_initial_population_strategy=None,
            autotuner_fn=default_autotuner_fn,
            autotune_baseline_fn=compare_attention_backends._sdpa_reference,
            autotune_baseline_atol=5e-2,
            autotune_baseline_rtol=2e-2,
            autotune_baseline_accuracy_check_fn=None,
            autotune_benchmark_fn=None,
            autotune_rebenchmark_threshold=None,
            autotune_suspicious_rebenchmark_ratio=None,
            autotune_config_overrides={},
            autotune_seed_configs=None,
            autotune_search_acf=[],
            autotune_config_filter=None,
        ),
        config_spec=dense_spec,
        kernel=SimpleNamespace(configs=[]),
    )

    provenance = compare_attention_backends._helion_autotune_provenance(
        args,
        bound,
        fixed_config=None,
        expected_baseline_fn=compare_attention_backends._sdpa_reference,
    )

    assert provenance["require_full_autotune"] is True
    assert provenance["effective_force_autotune"] is True
    assert provenance["cache_read_policy"] == "bypass"
    assert provenance["cache_write_policy"] == "write"
    compiler_seed_policy = provenance["compiler_seed_policy"]
    assert compiler_seed_policy["schema_version"] == 1
    assert compiler_seed_policy["kind"] == "canonical_cute_flash"
    assert compiler_seed_policy["heuristic_names"] == ["cute_flash_attention"]
    assert compiler_seed_policy["raw_config_count"] == len(
        dense_spec.compiler_seed_configs
    )
    assert compiler_seed_policy["effective_config_ids"]
    assert (
        compiler_seed_policy["effective_config_ids_sha256"]
        == hashlib.sha256(
            json.dumps(
                compiler_seed_policy["effective_config_ids"], separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    )
    assert compiler_seed_policy["timeout_retry_repetitions"] == 3
    assert provenance["compiler_seed_config_count"] == len(
        dense_spec.compiler_seed_configs
    )
    assert provenance["compiler_default_config"] is False
    assert provenance["kernel_declared_config_count"] == 0
    assert provenance["autotuner_fn_is_default"] is True
    assert provenance["autotune_baseline_fn"].endswith("._sdpa_reference")
    assert provenance["autotune_baseline_fn_is_expected"] is True
    assert provenance["autotune_benchmark_fn"] is False
    assert provenance["autotune_benchmark_subprocess"] is True
    assert provenance["autotune_benchmark_subprocess_env"] == ""
    assert (
        cute_flash.FLASH_PIPELINE_FAMILY_KEY
        not in provenance["active_value_prior_keys"]
    )
    assert provenance["flash_value_prior_keys"] == []
    assert provenance["flash_fragment_default_config"] == fragment_default
    assert (
        provenance["flash_fragment_default_sha256"]
        == hashlib.sha256(
            json.dumps(fragment_default, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    normalization_context = compare_attention_backends._flash_normalization_context(
        dense_spec
    )
    assert provenance["flash_normalization_context"] == normalization_context
    assert (
        provenance["flash_normalization_context_sha256"]
        == hashlib.sha256(
            json.dumps(
                normalization_context, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    )
    assert provenance["flash_structural_coverage_design_source"] == (
        "normalized active ConfigSpec fragments"
    )
    assert provenance["flash_structural_coverage_active_values"]
    assert provenance["flash_structural_coverage_uncovered_values"] == []
    assert provenance["flash_structural_coverage_underqualified_values"] == []
    assert provenance["flash_structural_coverage_underqualified_leaves"] == []
    assert provenance["flash_structural_leaf_catalog"]
    config_generation = ConfigGeneration(dense_spec)
    terminal_policy = provenance["flash_terminal_coordinate_refinement_policy"]
    assert terminal_policy == {
        "schema_version": 2,
        "policy_version": 2,
        "lane_policy_version": 14,
        "coordinate_policy": "same_leaf_full_surface_normalized_coordinate_v2",
        "measurement_policy": "mirrored_rotating_batched_wall_v2",
        "rounds": 2,
        "beam_width": 4,
        "radius": 2,
        "minimum_improvement_fraction": 0.001,
        "round_target_ms": 200.0,
        "confirmation_target_ms": 5000.0,
    }
    assert provenance["flash_terminal_coordinate_refinement_policy_sha256"] == (
        hashlib.sha256(
            json.dumps(terminal_policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    terminal_surface = config_generation.flash_terminal_coordinate_surface_catalog(
        radius=2
    )
    assert provenance["flash_terminal_coordinate_surface_catalog"] == terminal_surface
    assert len(terminal_surface["leaves"]) == len(
        config_generation.flash_structural_leaf_catalog()
    )
    assert provenance["flash_terminal_coordinate_surface_catalog_sha256"] == (
        hashlib.sha256(
            json.dumps(terminal_surface, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    lane_catalog = config_generation.flash_pipeline_lane_catalog()
    assert provenance["flash_pipeline_lane_catalog"] == [
        {
            "family": leaf.pipeline_family,
            "compound_packet": leaf.compound_exp2_packet,
            "softmax_disc": leaf.softmax_disc,
            "pipeline_lanes": [
                {"key": key, "value": value} for key, value in lane_catalog[leaf]
            ],
        }
        for leaf in config_generation.flash_structural_leaf_catalog()
    ]
    assert provenance["flash_structural_coverage_active_interactions"]
    assert provenance["flash_structural_coverage_uncovered_interactions"] == []
    assert provenance["flash_structural_qualification_values"]
    assert provenance["flash_structural_qualification_rounds"] == 2
    assert (
        provenance[
            "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round"
        ]
        == 4
    )
    assert provenance["flash_structural_retained_candidates_per_leaf"] == 2
    assert provenance["flash_structural_family_probe_generations"] == 1
    assert provenance["flash_structural_family_probe_candidates_per_path"] == 20
    assert provenance["flash_structural_retained_family_cap"] == 4
    expected_retained_family_limit = (
        config_generation.flash_structural_effective_family_limit(4)
    )
    assert (
        provenance["flash_structural_retained_family_limit"]
        == expected_retained_family_limit
    )
    assert provenance["flash_structural_retained_family_slowdown_limit"] == 2.0
    expected_starting_path_limit = (
        config_generation.flash_structural_starting_path_limit(
            minimum=14,
            retained_families=4,
            retained_candidates_per_leaf=2,
        )
    )
    assert expected_starting_path_limit > 14
    assert (
        provenance["flash_structural_starting_path_limit"]
        == expected_starting_path_limit
    )
    expected_probe_path_limit = (
        config_generation.flash_structural_family_probe_path_limit(4, 1)
    )
    assert (
        provenance["flash_structural_family_probe_path_limit"]
        == expected_probe_path_limit
    )
    assert provenance["flash_structural_maximum_path_capacity"] == max(
        expected_starting_path_limit, expected_probe_path_limit
    )
    assert provenance["autotune_lfbo_max_generations"] == 20
    assert (
        provenance["flash_structural_unrestricted_path_exhausts_generation_budget"]
        is True
    )

    design = provenance["flash_structural_coverage_design"]
    assert provenance["flash_structural_coverage_design_count"] == len(design)
    assert provenance["flash_structural_parent_coverage_prefix_count"] > 0
    assert (
        provenance["flash_structural_parent_coverage_prefix_count"]
        <= provenance["flash_structural_qualification_prefix_count"]
        <= len(design)
    )
    assert (
        provenance["flash_structural_population_budget"]
        <= provenance["autotune_initial_population_size"]
    )
    assert provenance["flash_structural_injected_design_count"] == min(
        provenance["flash_structural_population_budget"], len(design)
    )
    if len(design) <= provenance["autotune_initial_population_size"]:
        assert provenance["flash_structural_injected_design_count"] == len(design)
    assert len(design) > 1
    for candidate in design:
        expected = hashlib.sha256(
            json.dumps(
                candidate["config"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        assert candidate["config_sha256"] == expected
    assert len(provenance["flash_structural_coverage_design_sha256"]) == 64

    causal_spec = make_spec(is_causal=True)
    bound.config_spec = causal_spec
    causal_provenance = compare_attention_backends._helion_autotune_provenance(
        args,
        bound,
        fixed_config=None,
        expected_baseline_fn=compare_attention_backends._sdpa_reference,
    )
    assert causal_provenance["flash_value_prior_keys"] == []
    assert causal_provenance["flash_structural_coverage_uncovered_values"] == []
    assert causal_provenance["flash_structural_coverage_underqualified_values"] == []
    assert causal_provenance["flash_structural_coverage_underqualified_leaves"] == []
    assert causal_provenance["flash_structural_leaf_catalog"]
    assert causal_provenance["flash_pipeline_lane_catalog"]
    assert causal_provenance["flash_structural_coverage_active_interactions"]
    assert causal_provenance["flash_structural_coverage_uncovered_interactions"] == []
    assert causal_provenance["flash_structural_qualification_values"]
    assert causal_provenance["flash_structural_coverage_design"]


def test_attention_autotune_provenance_supports_triton_config_spec():
    import helion
    from helion._compiler.backend import TritonBackend
    from helion.autotuner.config_spec import ConfigSpec

    baseline = lambda *_args: None  # noqa: E731
    config_spec = ConfigSpec(backend=TritonBackend())
    bound = SimpleNamespace(
        settings=helion.Settings(
            backend="triton",
            autotune_effort="quick",
            autotune_baseline_fn=baseline,
        ),
        config_spec=config_spec,
        kernel=SimpleNamespace(configs=[]),
    )
    args = SimpleNamespace(
        helion_force_autotune=True,
        helion_require_full_autotune=0,
        skip_correctness=0,
    )

    provenance = compare_attention_backends._helion_autotune_provenance(
        args, bound, None, baseline
    )

    assert config_spec.cute_flash_search_enabled is False
    assert provenance["flash_normalization_context"] == {
        "schema_version": 1,
        "backend": "triton",
        "enabled": False,
    }
    assert provenance["flash_structural_coverage_design_source"] == "disabled"
    for field in (
        "flash_structural_coverage_active_values",
        "flash_structural_coverage_design",
        "flash_structural_coverage_uncovered_values",
        "flash_structural_coverage_underqualified_values",
        "flash_structural_leaf_catalog",
        "flash_pipeline_lane_catalog",
        "flash_clc_lane_catalog",
        "flash_structural_coverage_underqualified_leaves",
        "flash_structural_coverage_active_interactions",
        "flash_structural_coverage_uncovered_interactions",
        "flash_structural_qualification_values",
    ):
        assert provenance[field] == []


@pytest.mark.parametrize(
    "causal,packet,expected_fields",
    (
        (
            False,
            "deg2_16x6",
            {
                "cute_flash_pipeline_family": "fa4_2cta",
                "cute_flash_softmax_disc": True,
            },
        ),
        (
            True,
            "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
            {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_softmax_disc": True,
                "cute_flash_disc_pipe": 2,
                "cute_flash_split_p_arrive": True,
            },
        ),
    ),
)
def test_attention_compound_projection_uses_shape_config_generation(
    causal, packet, expected_fields
):
    from helion._compiler.backend import CuteBackend
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec

    spec = ConfigSpec(backend=CuteBackend())
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=128,
        num_kv=256,
        num_bh=64,
        tensor_4d_heads=32,
        dtype=torch.bfloat16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
    )
    generation = ConfigGeneration(spec)
    source_family = "fa4" if causal else "fa4_2cta"
    source = next(
        config.config
        for config in generation.flash_deterministic_population_configs()
        if (
            (leaf := flash_structural_leaf_from_config(config.config)) is not None
            and leaf.pipeline_family == source_family
            and leaf.compound_exp2_packet is None
        )
    )

    projected = compare_attention_backends._canonical_flash_projection(
        generation,
        source,
        {"cute_flash_exp2_packet": packet},
    )

    assert projected["cute_flash_exp2_packet"] == packet
    assert projected["cute_flash_e2e_schedule"] == "16/6"
    assert all(projected[key] == value for key, value in expected_fields.items())
    wrong = dict(projected)
    wrong["cute_flash_softmax_disc" if not causal else "cute_flash_disc_pipe"] = (
        False if not causal else 5
    )
    assert wrong != compare_attention_backends._canonical_flash_projection(
        generation,
        source,
        {"cute_flash_exp2_packet": packet},
    )


def test_attention_autotune_provenance_compares_winner_to_coverage_design() -> None:
    design = [
        {"config": {"a": 1, "b": 2}, "config_sha256": "a" * 64},
        {"config": {"a": 1, "b": 3, "c": 4}, "config_sha256": "b" * 64},
    ]
    provenance: dict[str, object] = {
        "flash_structural_coverage_design": design,
        "selected_config": {"a": 1, "b": 3},
    }
    compare_attention_backends._record_selected_structural_coverage_design_provenance(
        provenance
    )
    assert provenance["selected_config_is_structural_coverage_design_member"] is False
    assert (
        provenance["selected_config_nearest_structural_coverage_design_field_distance"]
        == 1
    )
    assert provenance[
        "selected_config_nearest_structural_coverage_design_config_sha256"
    ] == [
        "a" * 64,
        "b" * 64,
    ]

    provenance["selected_config"] = {"a": 1, "b": 2}
    compare_attention_backends._record_selected_structural_coverage_design_provenance(
        provenance
    )
    assert provenance["selected_config_is_structural_coverage_design_member"] is True
    assert (
        provenance["selected_config_nearest_structural_coverage_design_field_distance"]
        == 0
    )
    assert provenance[
        "selected_config_nearest_structural_coverage_design_config_sha256"
    ] == ["a" * 64]


def test_attention_autotune_provenance_handles_no_coverage_design() -> None:
    provenance: dict[str, object] = {
        "flash_structural_coverage_design": [],
        "selected_config": {"block_sizes": [128]},
    }
    compare_attention_backends._record_selected_structural_coverage_design_provenance(
        provenance
    )
    assert provenance["selected_config_is_structural_coverage_design_member"] is False
    assert (
        provenance["selected_config_nearest_structural_coverage_design_field_distance"]
        is None
    )
    assert (
        provenance["selected_config_nearest_structural_coverage_design_config_sha256"]
        == []
    )


def test_attention_required_full_autotune_requires_winner_guidance_provenance() -> None:
    provenance = _full_autotune_trial_provenance()
    provenance.pop("selected_config_nearest_structural_coverage_design_field_distance")
    with pytest.raises(
        RuntimeError, match="winner-to-structural-coverage-design provenance"
    ):
        _validate_full_autotune_trials(
            provenance,
            [_full_autotune_trial()],
        )


def test_attention_required_full_autotune_rejects_unexpected_algorithm():
    provenance = _full_autotune_trial_provenance()
    with pytest.raises(RuntimeError, match="unexpected search algorithms"):
        _validate_full_autotune_trials(
            provenance,
            [_full_autotune_trial(search_algorithm="RandomSearch")],
        )


def test_attention_required_full_autotune_accepts_standard_algorithm():
    trial, provenance, _anchor_ids = (
        _full_autotune_trial_with_complete_schedule_anchors()
    )
    _validate_full_autotune_trials(
        provenance,
        [trial],
        expected_fixture_trial=trial,
    )


def test_attention_required_full_autotune_accepts_unlimited_family_cap():
    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    provenance["flash_structural_retained_family_cap"] = None
    phase["retained_family_cap"] = None

    _validate_full_autotune_trials(provenance, [trial])


# Building the trial fixture and validating it both walk the full flash config
# surface; the first test to warm the lru_cache fixtures can exceed the 60s
# CI per-test timeout on slower runners.
@pytest.mark.timeout(300)
def test_attention_required_full_autotune_accepts_measured_family_probe_candidate():
    trial, provenance, _config_ids = _full_autotune_trial_with_family_probe_candidate()

    _validate_full_autotune_trials(
        provenance,
        [trial],
        expected_fixture_trial=trial,
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda phase: phase["family_probe_paths"].pop(),
        lambda phase: phase["family_probe_paths"][-1].update(unrestricted=False),
        lambda phase: phase["family_probe_paths"][0]["rounds"][0].update(
            measurement_pass_index=3
        ),
    ),
)
def test_attention_required_full_autotune_rejects_malformed_family_probe(mutate):
    trial, provenance, _anchor_ids = (
        _full_autotune_trial_with_complete_schedule_anchors()
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    mutate(phase)

    with pytest.raises(RuntimeError, match="family-probe"):
        _validate_full_autotune_trials(
            provenance,
            [trial],
            expected_fixture_trial=trial,
        )


def test_attention_required_full_autotune_accepts_shifted_schedule_anchor_pass():
    trial, provenance, anchor_ids = (
        _full_autotune_trial_with_complete_schedule_anchors()
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])

    _validate_full_autotune_trials(
        provenance,
        [trial],
        expected_fixture_trial=trial,
    )

    assert [item["pass_index"] for item in phase["measurement_timeline"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [result["config_id"] for result in phase["schedule_anchor_results"]] == (
        anchor_ids
    )
    assert {
        result["measurement_pass_index"] for result in phase["schedule_anchor_results"]
    } == {1}
    assert (
        phase["leaf_results"][0]["rounds"][0]["parent_decisions"][0][
            "candidate_results"
        ][0]["measurement_pass_index"]
        == 1
    )


def test_attention_required_full_autotune_rejects_schedule_anchor_producer_omission(
    monkeypatch,
):
    trial, provenance, _anchor_ids = (
        _full_autotune_trial_with_complete_schedule_anchors()
    )
    monkeypatch.setattr(
        _FixtureConfigGeneration,
        "flash_low_confound_schedule_anchor_configs",
        lambda _self: [],
    )

    with pytest.raises(RuntimeError, match="differs from independent"):
        _validate_full_autotune_trials(
            provenance,
            [trial],
            expected_fixture_trial=trial,
        )


def test_attention_required_full_autotune_accepts_compound_transfer_decision():
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    _validate_full_autotune_trials(provenance, [trial])


def test_attention_compound_transfer_snapshot_precedes_family_probe():
    config_id = "a" * 16
    source_hash = "b" * 64
    transfer_state = {
        "attempt_perf": 1.0,
        "selection_perf": 1.1,
        "status": "ok",
        "source_hash": source_hash,
    }
    post_probe_state = {**transfer_state, "selection_perf": 1.2}
    measurement_states = [{} for _ in range(13)]
    measurement_states[11][config_id] = transfer_state
    measurement_states[12][config_id] = post_probe_state
    transfer = {**transfer_state, "measurement_pass_index": 11}

    pre_probe_pass = compare_attention_backends._flash_pre_probe_pass_index(12, 1, True)

    assert pre_probe_pass == 11
    assert compare_attention_backends._measurement_snapshot_matches(
        transfer,
        measurement_states,
        config_id=config_id,
        expected_pass_index=pre_probe_pass,
    )
    assert not compare_attention_backends._measurement_snapshot_matches(
        transfer,
        measurement_states,
        config_id=config_id,
        expected_pass_index=12,
    )
    assert compare_attention_backends._flash_qualified_member_from_measurement_state(
        config_id, post_probe_state
    ) == (config_id, 1.2, frozenset())


def test_attention_compound_retention_uses_final_measurement(monkeypatch):
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    compound = cast("list[dict[str, Any]]", phase["compound_transfers"])[0]
    transfers = cast("list[dict[str, Any]]", compound["transfers"])
    refreshed_id = cast("str", transfers[0]["transferred_config_id"])
    final_updates = cast(
        "list[dict[str, Any]]", phase["measurement_timeline"][-1]["updates"]
    )
    refreshed_update = next(
        update for update in final_updates if update["config_id"] == refreshed_id
    )
    refreshed_update["selection_perf"] = 0.8

    ordinary = cast("dict[str, Any]", phase["leaf_results"][0])
    ordinary_members = [
        (
            qualified["config_id"],
            qualified["selection_perf"],
            frozenset(
                (lane["key"], lane["value"]) for lane in qualified["pipeline_lanes"]
            ),
        )
        for qualified in ordinary["qualified_results"]
    ]
    final_perf_by_id = {
        update["config_id"]: update["selection_perf"] for update in final_updates
    }
    compound_members = [
        (
            transfer["transferred_config_id"],
            final_perf_by_id[transfer["transferred_config_id"]],
            frozenset(),
        )
        for transfer in transfers
    ]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [
                (
                    "fa4_2cta",
                    None,
                    False,
                    ordinary_members,
                    (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3)),
                ),
                ("fa4_2cta", "deg2_16x6", False, compound_members, ()),
            ],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )

    original_matches = compare_attention_backends._measurement_snapshot_matches

    def accept_immutable_transfer(snapshot, measurement_states, **kwargs):
        if snapshot in transfers and kwargs["config_id"] == refreshed_id:
            return True
        return original_matches(snapshot, measurement_states, **kwargs)

    monkeypatch.setattr(
        compare_attention_backends,
        "_measurement_snapshot_matches",
        accept_immutable_transfer,
    )

    _validate_full_autotune_trials(provenance, [trial])


def test_attention_compound_transfer_validation_uses_pre_probe_pass(monkeypatch):
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    compound = cast("list[dict[str, Any]]", phase["compound_transfers"])[0]
    transfers = cast("list[dict[str, Any]]", compound["transfers"])
    transfer_config_ids = {
        cast("str", transfer["transferred_config_id"]) for transfer in transfers
    }
    final_pass = cast("int", phase["qualification_passes_completed"])
    pre_probe_pass = final_pass - 1
    observed_passes: list[int] = []
    original_matches = compare_attention_backends._measurement_snapshot_matches

    monkeypatch.setattr(
        compare_attention_backends,
        "_flash_pre_probe_pass_index",
        lambda _pass_count, _generations, _required: pre_probe_pass,
    )

    def record_transfer_pass(snapshot, measurement_states, **kwargs):
        if kwargs["config_id"] in transfer_config_ids:
            observed_passes.append(kwargs["expected_pass_index"])
            return True
        return original_matches(snapshot, measurement_states, **kwargs)

    monkeypatch.setattr(
        compare_attention_backends,
        "_measurement_snapshot_matches",
        record_transfer_pass,
    )

    _validate_full_autotune_trials(provenance, [trial])

    assert pre_probe_pass in observed_passes


def test_attention_strict_prevalidation_output_is_atomic(tmp_path):
    result_path = tmp_path / "result.json"
    args = argparse.Namespace(json_output=str(result_path))
    provenance = {
        "require_full_autotune": True,
        "selected_config": {"block_sizes": [1, 128, 128]},
        "trials": [{"num_configs_tested": 10}],
    }

    compare_attention_backends._write_strict_prevalidation_output(args, provenance)

    output_path = tmp_path / "result.strict-prevalidation.json"
    assert json.loads(output_path.read_text()) == {
        "schema_version": 1,
        "status": "autotune_complete_prevalidation",
        "autotune_provenance": provenance,
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_attention_streamed_failure_preserves_strict_prevalidation(
    tmp_path, monkeypatch
):
    final_path = tmp_path / "campaign.json"
    evidence = {
        "schema_version": 1,
        "status": "autotune_complete_prevalidation",
        "autotune_provenance": {"require_full_autotune": True},
    }

    def run(command, **_kwargs):
        json_path = Path(command[command.index("--json-output") + 1])
        json_path.with_name("result.strict-prevalidation.json").write_text(
            json.dumps(evidence) + "\n"
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(compare_attention_backends.subprocess, "run", run)
    args = SimpleNamespace(stream_subprocesses=True, json_output=str(final_path))

    returncode, payload, _, _ = compare_attention_backends._run_json_subprocess(
        ["python", "benchmark.py", "--impl", "helion-cute"], args
    )

    assert returncode == 1
    assert payload is None
    assert (
        json.loads((tmp_path / "campaign.strict-prevalidation.json").read_text())
        == evidence
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("compound_catalog_complete", False),
        (
            "compound_catalog_errors",
            [
                {
                    "family": "fa4_2cta",
                    "compound_packet": "deg2_16x6",
                    "softmax_disc": False,
                    "error": "missing_ordinary_protocol_leaf",
                }
            ],
        ),
    ),
)
def test_attention_required_full_autotune_rejects_incomplete_compound_catalog(
    field: str,
    value: object,
) -> None:
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    phase[field] = value

    with pytest.raises(RuntimeError, match="incomplete compound structural catalog"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_counts_novel_compound_leaf():
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    phase["leaves_with_candidates"] -= 1

    with pytest.raises(
        RuntimeError, match="inconsistent exact structural qualification candidates"
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_compound_attempt_after_limit():
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    transfer = cast("dict[str, Any]", phase["compound_transfers"][0])
    selection = cast("dict[str, Any]", transfer["source_selection"])
    selection["attempted_config_ids"].append(
        selection["candidate_results"][len(selection["attempted_config_ids"])][
            "config_id"
        ]
    )

    with pytest.raises(RuntimeError, match="immutable compound source decision"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_compound_projection_override():
    trial, provenance = _full_autotune_trial_with_compound_transfer()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    transfer = cast("dict[str, Any]", phase["compound_transfers"][0])
    transfer["transfers"][0]["projection_overrides"] = {"cute_flash_exp2_packet": "1x1"}

    with pytest.raises(RuntimeError, match="malformed v22 compound transfer"):
        _validate_full_autotune_trials(provenance, [trial])


@pytest.mark.parametrize(
    "field",
    (
        "flash_structural_leaf_catalog",
        "flash_pipeline_lane_catalog",
    ),
)
def test_attention_required_full_autotune_recomputes_structural_catalogs(field):
    provenance = _full_autotune_trial_provenance()
    provenance[field] = []

    with pytest.raises(RuntimeError, match="live ConfigGeneration"):
        _validate_full_autotune_trials(provenance, [_full_autotune_trial()])


def test_attention_required_full_autotune_rejects_normalization_context_mismatch():
    provenance = _full_autotune_trial_provenance(
        flash_normalization_context={"schema_version": 1},
        flash_normalization_context_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="normalization context"):
        _validate_full_autotune_trials(provenance, [_full_autotune_trial()])


def test_attention_required_full_autotune_rejects_empty_qualification():
    with pytest.raises(RuntimeError, match="live ConfigGeneration"):
        _validate_full_autotune_trials(
            _full_autotune_provenance(flash_structural_leaf_catalog=[]),
            [_full_autotune_trial()],
        )


@pytest.mark.parametrize(
    "override, field",
    (
        ({"input_shapes": "[(1,)]"}, "input_shapes"),
        ({"dtypes": "['torch.bfloat16']"}, "dtypes"),
        ({"hardware": "NVIDIA H100"}, "hardware"),
    ),
)
def test_attention_required_full_autotune_rejects_mismatched_identity(override, field):
    with pytest.raises(RuntimeError, match=field):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(**override)],
        )


def _validate_terminal_refinement_fixture(
    terminal, selected_config, selected_source_hash
):
    provenance = _full_autotune_trial_provenance()
    compare_attention_backends._validate_flash_terminal_coordinate_refinement(
        provenance,
        {"terminal_coordinate_refinement": terminal},
        {
            "num_generations": 20,
            "num_configs_tested": 102,
            "selected_config": selected_config,
            "selected_source_hash": selected_source_hash,
        },
        trial_index=1,
        config_generation=_cached_full_autotune_config_generation(),
    )


def test_attention_terminal_refinement_replays_measured_beam():
    terminal, selected_config, selected_source_hash = copy.deepcopy(
        _measured_full_autotune_terminal_fixture()
    )

    _validate_terminal_refinement_fixture(
        terminal, selected_config, selected_source_hash
    )
    confirmation = terminal["confirmation"]["measurement"]
    assert confirmation["sweep_count"] == 42
    assert confirmation["calls_per_sample"] == 2
    assert confirmation["total_calls"] == 84


@pytest.mark.parametrize("status", ("accuracy_error", "source_rejected"))
def test_attention_terminal_refinement_accepts_nonselectable_failure(status):
    terminal, selected_config, selected_source_hash = copy.deepcopy(
        _measured_full_autotune_terminal_fixture()
    )
    first_round = terminal["rounds"][0]
    failure_id = first_round["prior_failed_candidate_ids"][0]
    first_round["candidate_results"].append(
        {
            "config_id": failure_id,
            "attempt_perf": None,
            "selection_perf": None,
            "status": status,
            "source_hash": _test_measurement_source_hash(failure_id),
        }
    )

    _validate_terminal_refinement_fixture(
        terminal, selected_config, selected_source_hash
    )


def test_attention_terminal_refinement_rejects_omitted_projection():
    terminal, selected_config, selected_source_hash = copy.deepcopy(
        _measured_full_autotune_terminal_fixture()
    )
    terminal["rounds"][0]["parent_projections"][0]["coordinate_requests"].pop()

    with pytest.raises(RuntimeError, match="projection enumeration"):
        _validate_terminal_refinement_fixture(
            terminal, selected_config, selected_source_hash
        )


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda terminal: terminal["rounds"][0]["measurement"]["elapsed_ms"][
                0
            ].pop(),
            "malformed terminal coordinate measurement elapsed row",
        ),
        (
            lambda terminal: terminal["rounds"][0]["measurement"]["median_ms"][
                0
            ].update(value=1.1),
            "inconsistent median",
        ),
        (
            lambda terminal: terminal["rounds"][0]["measurement"].update(
                target_ms=201.0
            ),
            "invalid repeat provenance",
        ),
        (
            lambda terminal: terminal["rounds"][0]["measurement"].update(
                repeat_reference_perf_ms=1e6
            ),
            "repeat reference is inconsistent",
        ),
        (
            lambda terminal: terminal["rounds"][0]["measurement"].update(sweep_count=6),
            "inconsistent batched call sizing",
        ),
        (
            lambda terminal: terminal["confirmation"]["measurement"].update(
                total_calls=83
            ),
            "inconsistent batched call sizing",
        ),
        (
            lambda terminal: terminal["rounds"][0].update(accepted=False),
            "improvement decision",
        ),
        (
            lambda terminal: terminal["rounds"][0].update(
                beam_config_ids=list(reversed(terminal["rounds"][0]["beam_config_ids"]))
            ),
            "beam that does not match",
        ),
    ),
)
def test_attention_terminal_refinement_rejects_statistical_or_beam_tampering(
    mutate, message
):
    terminal, selected_config, selected_source_hash = copy.deepcopy(
        _measured_full_autotune_terminal_fixture()
    )
    mutate(terminal)

    with pytest.raises(RuntimeError, match=message):
        _validate_terminal_refinement_fixture(
            terminal, selected_config, selected_source_hash
        )


def test_attention_terminal_refinement_rejects_insufficient_raw_timing_work():
    terminal, selected_config, selected_source_hash = copy.deepcopy(
        _measured_full_autotune_terminal_fixture()
    )
    first_round = terminal["rounds"][0]
    first_round["measurement"] = _terminal_measurement(
        first_round["comparison_config_ids"],
        [11.25, 10.0],
        target_ms=200.0,
        repeat_reference_perf_ms=45.0,
    )

    with pytest.raises(RuntimeError, match="insufficient raw timing work"):
        _validate_terminal_refinement_fixture(
            terminal, selected_config, selected_source_hash
        )


def test_attention_terminal_refinement_rejects_unlinked_selected_source():
    terminal, selected_config, _selected_source_hash = copy.deepcopy(
        _measured_full_autotune_terminal_fixture()
    )

    with pytest.raises(RuntimeError, match="final config/source"):
        _validate_terminal_refinement_fixture(terminal, selected_config, "f" * 64)


def test_attention_required_full_autotune_validates_structural_qualification():
    provenance = _full_autotune_trial_provenance()
    phase = _full_autotune_trial()["search_phase_metrics"]
    _validate_full_autotune_trials(
        provenance,
        [_full_autotune_trial(search_phase_metrics=phase)],
    )

    leaf = phase["leaf_results"][0]
    leaf["rounds"][1]["candidate_config_ids"][0] = "not-a-config-id"
    with pytest.raises(RuntimeError, match="qualification round"):
        _validate_full_autotune_trials(
            provenance,
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_accepts_pipeline_failure_repair():
    trial, provenance, _witness_id = _full_autotune_trial_with_pipeline_repair()

    _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_accepts_terminal_pipeline_failure():
    trial, provenance, _witness_id, _repair_id = (
        _full_autotune_trial_with_terminal_pipeline_failure()
    )

    _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_unexhausted_pipeline_failure():
    trial, provenance, _witness_id, _repair_id = (
        _full_autotune_trial_with_terminal_pipeline_failure()
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    lane = cast("dict[str, Any]", phase["leaf_results"][0]["pipeline_lanes"][0])
    lane["repair_parent_decisions"] = []

    with pytest.raises(
        RuntimeError,
        match="qualification pass accounting|incomplete v22 pipeline lane",
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_success_marked_exhausted():
    trial, provenance, _witness_id = _full_autotune_trial_with_pipeline_repair()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    lane = cast("dict[str, Any]", phase["leaf_results"][0]["pipeline_lanes"][0])
    lane["terminal_failure_exhausted"] = True

    with pytest.raises(RuntimeError, match="incomplete v22 pipeline lane"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_nonretryable_terminal_failure():
    trial, provenance, _witness_id, repair_id = (
        _full_autotune_trial_with_terminal_pipeline_failure()
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])

    def mark_accuracy_error(value) -> None:
        if isinstance(value, dict):
            if value.get("config_id") == repair_id and "status" in value:
                value["status"] = "accuracy_error"
            for child in value.values():
                mark_accuracy_error(child)
        elif isinstance(value, list):
            for child in value:
                mark_accuracy_error(child)

    mark_accuracy_error(phase)
    trial["num_accuracy_failures"] = 1
    with pytest.raises(
        RuntimeError,
        match="pipeline lane|terminal|successful measured witness",
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_unproven_source_repair():
    trial, provenance, _witness_id = _full_autotune_trial_with_pipeline_repair()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    lane = cast("dict[str, Any]", phase["leaf_results"][0]["pipeline_lanes"][0])
    repair_id = cast("str", lane["repair_candidate_ids"][0])

    def replace_repair_source_hash(value) -> None:
        if isinstance(value, dict):
            if value.get("config_id") == repair_id and "source_hash" in value:
                value["source_hash"] = "e" * 64
            for child in value.values():
                replace_repair_source_hash(child)
        elif isinstance(value, list):
            for child in value:
                replace_repair_source_hash(child)

    replace_repair_source_hash(phase)
    with pytest.raises(RuntimeError, match="effective-source repair"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_malformed_pipeline_repair():
    trial, provenance, _witness_id = _full_autotune_trial_with_pipeline_repair()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    lane = cast("dict[str, Any]", phase["leaf_results"][0]["pipeline_lanes"][0])
    lane["repair_parent_decisions"][0]["generated_config_ids"] = []

    with pytest.raises(RuntimeError, match="pipeline repair decision"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_illegitimate_pipeline_repair():
    trial, provenance, witness_id = _full_autotune_trial_with_pipeline_repair()

    def mark_accuracy_error(value):
        if isinstance(value, dict):
            if value.get("config_id") == witness_id and "status" in value:
                value["status"] = "accuracy_error"
            for item in value.values():
                mark_accuracy_error(item)
        elif isinstance(value, list):
            for item in value:
                mark_accuracy_error(item)

    mark_accuracy_error(trial["search_phase_metrics"])
    trial["num_accuracy_failures"] = 1

    with pytest.raises(
        RuntimeError,
        match=(
            "measurement timeline|pipeline failure repair|pipeline repair decision|"
            "successful measured witness"
        ),
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_requires_pipeline_lane_catalog():
    provenance = _full_autotune_trial_provenance()
    provenance.pop("flash_pipeline_lane_catalog")

    with pytest.raises(RuntimeError, match="live ConfigGeneration"):
        _validate_full_autotune_trials(provenance, [_full_autotune_trial()])


@pytest.mark.parametrize("mutate_provenance", (False, True))
def test_attention_required_full_autotune_rejects_fabricated_pipeline_lane_catalog(
    mutate_provenance,
):
    provenance = _full_autotune_trial_provenance()
    phase = _full_autotune_trial()["search_phase_metrics"]
    catalog = provenance["flash_pipeline_lane_catalog"][0]["pipeline_lanes"]
    phase_lanes = phase["leaf_results"][0]["pipeline_lanes"]
    target = catalog if mutate_provenance else phase_lanes
    target.reverse()

    with pytest.raises(
        RuntimeError,
        match="live ConfigGeneration|seed-dependent or fabricated pipeline lane catalog",
    ):
        _validate_full_autotune_trials(
            provenance,
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_unwitnessed_pipeline_lane():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    initial_ids = leaf["initial_config_ids"]
    kv2_lane, kv3_lane = leaf["pipeline_lanes"]
    kv2_lane["initial_config_ids"] = list(initial_ids)
    kv3_lane["initial_config_ids"] = []
    for result in leaf["qualified_results"]:
        result["pipeline_lanes"] = [{"key": "cute_flash_kv_stage", "value": 2}]

    with pytest.raises(
        RuntimeError,
        match=(
            "omits a generation-zero measurement|nonmember candidate|"
            "lacks a successful measured witness"
        ),
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_unassigned_lane_round_id():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    candidate_id = "a" * 16
    leaf["rounds"][0]["candidate_config_ids"] = [candidate_id]
    leaf["qualified_results"].append(
        {
            "config_id": candidate_id,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "measurement_pass_index": 2,
            "pipeline_lanes": [{"key": "cute_flash_kv_stage", "value": 2}],
        }
    )
    phase["candidate_count"] = 1
    phase["leaves_with_candidates"] = 1

    with pytest.raises(RuntimeError, match="per-lane qualification round IDs"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_omitted_generated_lane_witness():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    lane = leaf["pipeline_lanes"][0]
    witness_id = "b" * 16
    lane["rounds"][0]["candidate_config_ids"] = [witness_id]
    lane["witness_config_id"] = witness_id
    leaf["qualified_results"].append(
        {
            "config_id": witness_id,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "pipeline_lanes": [{"key": "cute_flash_kv_stage", "value": 2}],
        }
    )
    phase["candidate_count"] = 3

    with pytest.raises(RuntimeError, match="per-lane qualification round IDs"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("candidate_count", 3), ("leaves_with_candidates", 0)),
)
def test_attention_required_full_autotune_recomputes_candidate_accounting(
    field: str, value: int
):
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase[field] = value

    with pytest.raises(RuntimeError, match="exact structural qualification candidates"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_wrong_leaf_neighbor_limit():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["leaf_results"][0]["rounds"][0]["neighbor_generation_limit"] = 199

    with pytest.raises(
        RuntimeError,
        match="structural qualification round|per-lane qualification round",
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_wrong_lane_neighbor_limit():
    phase = _full_autotune_trial()["search_phase_metrics"]
    lane_round = phase["leaf_results"][0]["pipeline_lanes"][0]["rounds"][0]
    lane_round["neighbor_generation_limit"] = 99

    with pytest.raises(
        RuntimeError, match="qualification pass accounting|per-lane qualification round"
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_false_lane_membership():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["leaf_results"][0]["qualified_results"][0]["pipeline_lanes"] = [
        {"key": "cute_flash_kv_stage", "value": 4}
    ]

    with pytest.raises(RuntimeError, match="pipeline lane membership"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_lane_absent_from_active_values():
    provenance = _full_autotune_trial_provenance()
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    leaf["pipeline_lanes"][1]["value"] = 7
    provenance["flash_pipeline_lane_catalog"][0]["pipeline_lanes"][1]["value"] = 7
    for result in leaf["qualified_results"]:
        if result["pipeline_lanes"][0]["value"] == 3:
            result["pipeline_lanes"][0]["value"] = 7

    with pytest.raises(
        RuntimeError,
        match="live ConfigGeneration|absent from the active-value manifest",
    ):
        _validate_full_autotune_trials(
            provenance,
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_boolean_zero_lane_neighbor_limit():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["pipeline_candidate_limit_per_leaf_per_round"] = 1
    kv2_lane, kv3_lane = phase["leaf_results"][0]["pipeline_lanes"]
    kv2_lane["rounds"][0]["neighbor_generation_limit"] = 200
    kv2_lane["rounds"][1]["neighbor_generation_limit"] = False
    kv3_lane["rounds"][0]["neighbor_generation_limit"] = 0
    kv3_lane["rounds"][1]["neighbor_generation_limit"] = 200

    with pytest.raises(
        RuntimeError, match="qualification pass accounting|per-lane qualification round"
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(
                flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round=1
            ),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_recomputes_lane_diverse_retention():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    kv3_ids = leaf["pipeline_lanes"][1]["initial_config_ids"]
    leaf["retained_config_ids"] = sorted(kv3_ids)[:2]

    with pytest.raises(RuntimeError, match="incorrect retained candidates"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_skipped_conditional_without_exact_proof():
    phase = _full_autotune_trial()["search_phase_metrics"]
    lane = phase["leaf_results"][0]["pipeline_lanes"][0]
    lane["space_exhausted"] = True
    lane["conditional_required"] = False

    with pytest.raises(RuntimeError, match="incomplete v22 pipeline lane"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_non_novel_conditional_candidate():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    lane = leaf["pipeline_lanes"][0]
    old_id = lane["conditional_candidate_ids"][0]
    initial_id = lane["initial_config_ids"][0]
    lane["conditional_candidate_ids"] = [initial_id]
    lane["successful_conditional_candidate_ids"] = [initial_id]
    for lane_round in lane["rounds"]:
        lane_round["candidate_config_ids"] = [
            initial_id if config_id == old_id else config_id
            for config_id in lane_round["candidate_config_ids"]
        ]

    with pytest.raises(RuntimeError, match="incomplete v22 pipeline lane"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_conditional_missing_from_leaf_round():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf = phase["leaf_results"][0]
    conditional_id = leaf["pipeline_lanes"][0]["conditional_candidate_ids"][0]
    leaf["rounds"][1]["candidate_config_ids"].remove(conditional_id)

    with pytest.raises(RuntimeError, match="per-lane qualification round IDs"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_recomputes_starting_path_lane():
    phase = _full_autotune_trial()["search_phase_metrics"]
    paths = phase["retained_families"][0]["starting_paths"]
    paths[1]["pipeline_lane"] = {
        "key": "cute_flash_kv_stage",
        "value": 2 if paths[1]["pipeline_lane"]["value"] == 3 else 3,
    }

    with pytest.raises(RuntimeError, match="retained structural family ranking"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


@pytest.mark.parametrize(
    ("phase_update", "match"),
    (
        (
            {"phase": "cute_flash_structural_qualification_v12"},
            "incomplete or non-v22",
        ),
        ({"cute_flash_lane_policy_version": 2}, "incomplete or non-v22"),
        (
            {"pipeline_qualification_keys": ["cute_flash_s_stage"]},
            "qualification bounds",
        ),
        ({"qualification_rounds": 1}, "qualification bounds"),
        ({"qualification_rounds_started": 1}, "qualification pass accounting"),
        ({"qualification_rounds_completed": 1}, "qualification pass accounting"),
        ({"qualification_passes_started": 1}, "qualification pass accounting"),
        (
            {"conditional_candidates_per_pipeline_lane": 0},
            "qualification pass accounting",
        ),
        ({"qualification_failure_retries": 0}, "qualification pass accounting"),
        ({"budget_exhausted": True}, "incomplete or non-v22"),
        ({"pipeline_candidate_limit_per_leaf_per_round": 3}, "qualification bounds"),
        ({"neighbor_generation_limit_per_leaf_per_round": 0}, "qualification bounds"),
        (
            {"neighbor_generation_limit_per_leaf_per_round": 199},
            "qualification bounds",
        ),
        ({"retained_family_cap": 3}, "qualification bounds"),
        ({"retained_family_limit": 5}, "qualification bounds"),
        ({"starting_path_limit": 4}, "qualification bounds"),
        (
            {"unrestricted_path_exhausts_generation_budget": False},
            "qualification bounds",
        ),
        ({"leaf_results": []}, "ordinary/compound leaf counts"),
        ({"retained_families": []}, "retained structural family ranking"),
    ),
)
def test_attention_required_full_autotune_rejects_incomplete_leaf_qualification(
    phase_update, match
):
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase.update(phase_update)
    with pytest.raises(RuntimeError, match=match):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_missing_phase_exhaustion_policy():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase.pop("unrestricted_path_exhausts_generation_budget")

    with pytest.raises(RuntimeError, match="qualification bounds"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_unmeasured_starting_path():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["retained_families"][0]["starting_paths"][0]["config_id"] = "b" * 16
    with pytest.raises(RuntimeError, match="retained structural family ranking"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_wrong_unrestricted_path():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["retained_families"][0]["starting_paths"][-1]["unrestricted"] = False
    with pytest.raises(RuntimeError, match="retained structural family ranking"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_requires_exact_qualified_membership():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["leaf_results"][0]["qualified_results"].append(
        {
            "config_id": "e" * 16,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "measurement_pass_index": 2,
            "pipeline_lanes": [{"key": "cute_flash_kv_stage", "value": 2}],
        }
    )
    with pytest.raises(
        RuntimeError,
        match="inconsistent result status/performance pair|inconsistent measured IDs",
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_failed_retained_candidate():
    phase = _full_autotune_trial()["search_phase_metrics"]
    retained_id = phase["leaf_results"][0]["retained_config_ids"][0]
    qualified = next(
        item
        for item in phase["leaf_results"][0]["qualified_results"]
        if item["config_id"] == retained_id
    )
    qualified.update(status="error", attempt_perf=None, selection_perf=None)
    with pytest.raises(
        RuntimeError,
        match=(
            "incorrect retained candidates|lacks a successful measured witness|"
            "fabricated measurements|inconsistent measured result timeline snapshot"
        ),
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def test_attention_required_full_autotune_rejects_status_perf_mismatch():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["leaf_results"][0]["qualified_results"][0]["status"] = "error"
    with pytest.raises(RuntimeError, match="status/performance"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


@pytest.mark.parametrize("status", ("error", "timeout"))
def test_attention_required_full_autotune_accepts_isolated_rebenchmark_invalidation(
    status,
):
    trial, _invalidated_id = (
        _full_autotune_trial_with_isolated_rebenchmark_invalidation(status)
    )

    _validate_full_autotune_trials(
        _full_autotune_trial_provenance(),
        [trial],
    )


def test_attention_required_full_autotune_rejects_invalidated_compiler_seed():
    seed_id = _full_autotune_trial()["search_phase_metrics"]["initial_results"][-1][
        "config_id"
    ]
    trial, invalidated_id = _full_autotune_trial_with_isolated_rebenchmark_invalidation(
        "timeout", invalidated_id=seed_id
    )
    assert invalidated_id == seed_id

    with pytest.raises(RuntimeError, match="compiler seed in pass 0"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [trial],
        )


def test_attention_required_full_autotune_accepts_source_wide_rebenchmark_invalidation():
    trial, invalidated_id = _full_autotune_trial_with_isolated_rebenchmark_invalidation(
        "timeout"
    )
    _add_isolated_rebenchmark_alias_invalidation(trial, invalidated_id, complete=True)

    _validate_full_autotune_trials(
        _full_autotune_trial_provenance(),
        [trial],
    )


def test_attention_required_full_autotune_rejects_partial_source_invalidation():
    trial, invalidated_id = _full_autotune_trial_with_isolated_rebenchmark_invalidation(
        "timeout"
    )
    _add_isolated_rebenchmark_alias_invalidation(trial, invalidated_id, complete=False)

    with pytest.raises(RuntimeError, match="incomplete v22 effective-source"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [trial],
        )


def test_attention_required_full_autotune_rejects_unaccounted_timeout_source():
    trial, _invalidated_id = (
        _full_autotune_trial_with_isolated_rebenchmark_invalidation("timeout")
    )
    trial["num_isolated_rebenchmark_timeouts"] = 0

    with pytest.raises(RuntimeError, match="fewer isolated rebenchmark timeouts"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [trial],
        )


def test_attention_required_full_autotune_rejects_rebenchmark_invalidation_source_change():
    trial, invalidated_id = _full_autotune_trial_with_isolated_rebenchmark_invalidation(
        "timeout"
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    update = next(
        item
        for item in phase["measurement_timeline"][2]["updates"]
        if item["config_id"] == invalidated_id
    )
    update["source_hash"] = "b" * 64

    with pytest.raises(RuntimeError, match="measurement state transition"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [trial],
        )


@pytest.mark.parametrize("perf_key", ("attempt_perf", "selection_perf"))
def test_attention_required_full_autotune_rejects_rebenchmark_invalidation_perf(
    perf_key,
):
    trial, invalidated_id = _full_autotune_trial_with_isolated_rebenchmark_invalidation(
        "error"
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    update = next(
        item
        for item in phase["measurement_timeline"][2]["updates"]
        if item["config_id"] == invalidated_id
    )
    update[perf_key] = 1.0

    with pytest.raises(RuntimeError, match="measurement timeline update"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [trial],
        )


def test_attention_required_full_autotune_recomputes_retained_tie_order():
    phase = _full_autotune_trial()["search_phase_metrics"]
    phase["leaf_results"][0]["retained_config_ids"].reverse()
    with pytest.raises(RuntimeError, match="incorrect retained candidates"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


@pytest.mark.parametrize("field", ("score", "starting_paths"))
def test_attention_required_full_autotune_recomputes_parent_promotion(field):
    phase = _full_autotune_trial()["search_phase_metrics"]
    retained = phase["retained_families"][0]
    if field == "score":
        retained["score"] = 2.0
    else:
        retained["starting_paths"].reverse()
    with pytest.raises(RuntimeError, match="retained structural family ranking"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


def _structural_leaf_qualification(
    family, compound_packet, members, *, softmax_disc=True
):
    return (
        family,
        compound_packet,
        softmax_disc,
        [(config_id, perf, frozenset()) for config_id, perf in members],
        (),
    )


def test_attention_structural_retention_mirror_matches_live_capacity_selector():
    import helion
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
    from helion.autotuner.base_search import PopulationMember
    from helion.autotuner.effort_profile import get_effort_profile
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

    stage_key = "cute_flash_kv_stage"

    def member(
        perf: float,
        family: str,
        packet: str,
        softmax_disc: bool,
        stage: int,
        wait_hint: int,
    ) -> PopulationMember:
        return PopulationMember(
            fn=lambda: None,
            perfs=[perf],
            flat_values=[],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family=family,
                cute_flash_exp2_packet=packet,
                cute_flash_softmax_disc=softmax_disc,
                cute_flash_kv_stage=stage,
                cute_flash_wait_hint=wait_hint,
            ),
            status="ok",
        )

    global_winner = member(1.0, "fa4_2cta", "1x1", True, 2, 0)
    lane_alternate = member(1.2, "fa4_2cta", "1x1", True, 3, 1)
    compounds = [
        member(1.001, "fa4_2cta", "deg2_16x6", True, 2, 2),
        member(1.002, "fa4_2cta", "deg1_16x8", True, 2, 3),
        member(1.003, "fa4_2cta", "deg1_8x2_corr10", True, 2, 4),
    ]
    ordinary_members = [global_winner]
    wait_hint = 5
    for family, perf in (
        ("fa4_2cta", 1.01),
        ("fa4", 1.05),
        ("ws_overlap", 1.06),
        ("fa4_clc", 1.07),
    ):
        for softmax_disc in (True, False):
            if family == "fa4_2cta" and softmax_disc:
                continue
            ordinary_members.append(
                member(
                    perf + (0.001 if not softmax_disc else 0.0),
                    family,
                    "1x1",
                    softmax_disc,
                    2,
                    wait_hint,
                )
            )
            wait_hint += 1
    ordinary_secondaries = [
        member(1.3 + index * 0.01, family, "1x1", True, 3, wait_hint + index)
        for index, family in enumerate(("fa4", "ws_overlap", "fa4_clc"))
    ]
    population = [
        global_winner,
        lane_alternate,
        *compounds,
        *ordinary_members[1:],
        *ordinary_secondaries,
    ]
    global_leaf = flash_structural_leaf_from_config(global_winner.config.config)
    assert global_leaf is not None
    lanes = ((stage_key, 2), (stage_key, 3))

    policy = get_effort_profile("full").flash_structural_search
    assert policy is not None
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
    search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
    search.flash_structural_search = policy
    search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
    search.population = population
    search._flash_qualified_pipeline_lanes = {global_leaf: lanes}

    qualified_by_leaf = []
    for leaf in dict.fromkeys(
        flash_structural_leaf_from_config(item.config.config) for item in population
    ):
        assert leaf is not None
        leaf_lanes = lanes if leaf == global_leaf else ()
        members = [
            (
                canonical_config_id(item.config),
                item.perf,
                frozenset(
                    lane
                    for lane in leaf_lanes
                    if item.config.config.get(lane[0]) == lane[1]
                ),
            )
            for item in population
            if flash_structural_leaf_from_config(item.config.config) == leaf
        ]
        qualified_by_leaf.append(
            (
                leaf.pipeline_family,
                leaf.compound_exp2_packet,
                leaf.softmax_disc,
                members,
                leaf_lanes,
            )
        )

    ordinary_widths: dict[str, int] = {}
    compound_count = 0
    for family, compound_packet, _softmax_disc, _members, _lanes in qualified_by_leaf:
        if compound_packet is None:
            ordinary_widths[family] = ordinary_widths.get(family, 0) + 1
        else:
            compound_count += 1
    retained_family_limit = (
        len(ordinary_widths)
        if policy.retained_families is None
        else min(policy.retained_families, len(ordinary_widths))
    )
    promoted_count = min(retained_family_limit, len(ordinary_widths))
    promoted_protocol_count = sum(
        sorted(ordinary_widths.values(), reverse=True)[:promoted_count]
    )
    search.copies = max(
        policy.starting_paths,
        1
        + promoted_protocol_count
        + (promoted_count if policy.retained_candidates_per_leaf > 1 else 0)
        + compound_count,
    )

    search._select_starting_paths()
    actual = search._autotune_metrics.search_phase_metrics["retained_families"]

    expected = compare_attention_backends._expected_flash_structural_retention(
        qualified_by_leaf,
        retained_per_leaf=policy.retained_candidates_per_leaf,
        retained_family_cap=policy.retained_families,
        retained_family_limit=retained_family_limit,
        retained_family_slowdown_limit=policy.retained_family_slowdown_limit,
        starting_path_limit=search.copies,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert expected == actual
    selected_paths = [path for family in expected for path in family["starting_paths"]]
    assert search.copies == 16
    assert len(selected_paths) == 16
    assert sum(path["compound_packet"] is not None for path in selected_paths) == 3
    selected_ids = {path["config_id"] for path in selected_paths}
    assert {
        canonical_config_id(member.config) for member in ordinary_secondaries
    } <= selected_ids
    assert {canonical_config_id(item.config) for item in compounds} <= selected_ids
    assert {
        family["family"]: sum(
            path["compound_packet"] is None for path in family["starting_paths"]
        )
        for family in expected
    } == {
        "fa4_2cta": 4,
        "fa4": 3,
        "ws_overlap": 3,
        "fa4_clc": 3,
    }
    assert {
        path["softmax_disc"]
        for path in selected_paths
        if path["compound_packet"] is None and path["pipeline_lane"] is None
    } == {False, True}
    assert any(
        path["config_id"] == canonical_config_id(lane_alternate.config)
        and path["pipeline_lane"] == {"key": stage_key, "value": 3}
        for path in selected_paths
    )


def test_attention_structural_retention_counts_ordinary_lane_alternate_as_parent():
    kv2 = ("cute_flash_kv_stage", 2)
    kv3 = ("cute_flash_kv_stage", 3)
    lanes = (kv2, kv3)
    config_ids = [f"{index:016x}" for index in range(6)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            (
                "compound_rich",
                "deg2_16x6",
                True,
                [(config_ids[0], 0.5, frozenset((kv2,)))],
                lanes,
            ),
            (
                "compound_rich",
                None,
                True,
                [
                    (config_ids[1], 1.0, frozenset((kv2,))),
                    (config_ids[2], 1.1, frozenset((kv3,))),
                ],
                lanes,
            ),
            _structural_leaf_qualification("alpha", None, [(config_ids[3], 0.8)]),
            _structural_leaf_qualification("beta", None, [(config_ids[4], 0.9)]),
            _structural_leaf_qualification("gamma", None, [(config_ids[5], 1.2)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=3,
        retained_family_limit=3,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=4,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained] == [
        "compound_rich",
        "alpha",
        "beta",
    ]
    assert retained[0]["parent_promoted"] is True
    assert retained[0]["starting_paths"] == [
        {
            "family": "compound_rich",
            "compound_packet": None,
            "softmax_disc": True,
            "config_id": config_ids[1],
            "unrestricted": False,
            "pipeline_lane": None,
        },
        {
            "family": "compound_rich",
            "compound_packet": "deg2_16x6",
            "softmax_disc": True,
            "config_id": config_ids[0],
            "unrestricted": True,
            "pipeline_lane": None,
        },
    ]


def test_attention_structural_retention_selects_top_four_and_distinct_leaves():
    config_ids = [f"{index:016x}" for index in range(8)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "alpha", None, [(config_ids[0], 1.0), (config_ids[2], 1.0)]
            ),
            _structural_leaf_qualification(
                "alpha", "deg2_16x6", [(config_ids[1], 1.0)]
            ),
            _structural_leaf_qualification("beta", None, [(config_ids[3], 1.0)]),
            _structural_leaf_qualification("gamma", None, [(config_ids[4], 2.0)]),
            _structural_leaf_qualification("delta", None, [(config_ids[5], 3.0)]),
            _structural_leaf_qualification("epsilon", None, [(config_ids[6], 4.0)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=4,
        retained_family_limit=4,
        retained_family_slowdown_limit=10.0,
        starting_path_limit=7,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]
    assert [path["config_id"] for path in retained[0]["starting_paths"]] == [
        config_ids[0],
        config_ids[2],
        config_ids[1],
        config_ids[0],
    ]
    assert [path["unrestricted"] for path in retained[0]["starting_paths"]] == [
        False,
        False,
        False,
        True,
    ]
    assert sum(len(family["starting_paths"]) for family in retained) == 7


def test_attention_structural_retention_unlimited_keeps_slow_live_family():
    config_ids = [f"{index:016x}" for index in range(5)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(family, None, [(config_id, perf)])
            for family, config_id, perf in zip(
                ("alpha", "beta", "gamma", "delta", "epsilon"),
                config_ids,
                (1.0, 1.1, 1.2, 1.3, 5.0),
                strict=True,
            )
        ],
        retained_per_leaf=2,
        retained_family_cap=None,
        retained_family_limit=5,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=6,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
    ]


def test_attention_structural_retention_does_not_prune_close_sibling_leaf():
    config_ids = [f"{index:016x}" for index in range(8)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "alpha", None, [(config_ids[0], 1.0), (config_ids[1], 1.001)]
            ),
            _structural_leaf_qualification("beta", None, [(config_ids[2], 1.015)]),
            _structural_leaf_qualification(
                "beta", "deg2_16x6", [(config_ids[3], 1.01)]
            ),
            _structural_leaf_qualification(
                "beta", "deg1_16x8", [(config_ids[4], 1.011)]
            ),
            _structural_leaf_qualification("gamma", None, [(config_ids[5], 3.0)]),
            _structural_leaf_qualification("delta", None, [(config_ids[6], 3.1)]),
            _structural_leaf_qualification("epsilon", None, [(config_ids[7], 3.2)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=4,
        retained_family_limit=4,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=6,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    by_family = {family["family"]: family["starting_paths"] for family in retained}
    assert [path["config_id"] for path in by_family["alpha"]] == [
        config_ids[0],
        config_ids[1],
        config_ids[0],
    ]
    assert [path["config_id"] for path in by_family["beta"]] == config_ids[2:5]
    assert [
        path["config_id"]
        for paths in by_family.values()
        for path in paths
        if path["unrestricted"]
    ] == [config_ids[0]]


def test_attention_structural_retention_does_not_parent_promote_compound_only():
    config_ids = [f"{index:016x}" for index in range(5)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "compound_only", "deg2_16x6", [(config_ids[0], 0.5)]
            ),
            _structural_leaf_qualification("alpha", None, [(config_ids[1], 1.0)]),
            _structural_leaf_qualification("beta", None, [(config_ids[2], 1.1)]),
            _structural_leaf_qualification("gamma", None, [(config_ids[3], 1.2)]),
            _structural_leaf_qualification("delta", None, [(config_ids[4], 1.3)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=4,
        retained_family_limit=4,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=5,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained] == [
        "compound_only",
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]
    assert retained[0]["parent_promoted"] is False
    assert len(retained[0]["starting_paths"]) == 1


def test_attention_structural_retention_scores_families_by_ordinary_leaf():
    config_ids = [f"{index:016x}" for index in range(7)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "compound_rich", None, [(config_ids[0], 1.5)]
            ),
            _structural_leaf_qualification(
                "compound_rich", "deg2_16x6", [(config_ids[1], 0.9)]
            ),
            _structural_leaf_qualification("alpha", None, [(config_ids[2], 0.8)]),
            _structural_leaf_qualification("beta", None, [(config_ids[3], 1.0)]),
            _structural_leaf_qualification("gamma", None, [(config_ids[4], 1.1)]),
            _structural_leaf_qualification("delta", None, [(config_ids[5], 1.2)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=4,
        retained_family_limit=4,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=5,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained if family["parent_promoted"]] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]
    assert all(
        family["score_compound_packet"] is None
        for family in retained
        if family["parent_promoted"]
    )


def test_attention_structural_retention_keeps_global_compound_winner():
    config_ids = [f"{index:016x}" for index in range(6)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "compound_rich", None, [(config_ids[0], 1.5)]
            ),
            _structural_leaf_qualification(
                "compound_rich", "deg2_16x6", [(config_ids[1], 0.5)]
            ),
            _structural_leaf_qualification("alpha", None, [(config_ids[2], 1.0)]),
            _structural_leaf_qualification("beta", None, [(config_ids[3], 1.1)]),
            _structural_leaf_qualification("gamma", None, [(config_ids[4], 1.2)]),
            _structural_leaf_qualification("delta", None, [(config_ids[5], 1.3)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=4,
        retained_family_limit=4,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=4,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    unrestricted = [
        path
        for family in retained
        for path in family["starting_paths"]
        if path["unrestricted"]
    ]
    assert unrestricted == [
        {
            "family": "compound_rich",
            "compound_packet": "deg2_16x6",
            "softmax_disc": True,
            "config_id": config_ids[1],
            "unrestricted": True,
            "pipeline_lane": None,
        }
    ]
    assert [family["family"] for family in retained] == [
        "compound_rich",
        "alpha",
        "beta",
        "gamma",
    ]
    assert retained[0]["parent_promoted"] is False


def test_attention_structural_retention_gives_dominated_compound_winner_one_path():
    config_ids = [f"{index:016x}" for index in range(6)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "compound_rich", None, [(config_ids[0], 3.0)]
            ),
            _structural_leaf_qualification(
                "compound_rich", "deg2_16x6", [(config_ids[1], 0.5)]
            ),
            _structural_leaf_qualification("alpha", None, [(config_ids[2], 1.0)]),
            _structural_leaf_qualification("beta", None, [(config_ids[3], 1.1)]),
            _structural_leaf_qualification("gamma", None, [(config_ids[4], 1.2)]),
            _structural_leaf_qualification("delta", None, [(config_ids[5], 1.3)]),
        ],
        retained_per_leaf=2,
        retained_family_cap=4,
        retained_family_limit=4,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=4,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained] == [
        "compound_rich",
        "alpha",
        "beta",
        "gamma",
    ]
    assert retained[0]["parent_promoted"] is False
    assert retained[0]["starting_paths"] == [
        {
            "family": "compound_rich",
            "compound_packet": "deg2_16x6",
            "softmax_disc": True,
            "config_id": config_ids[1],
            "unrestricted": True,
            "pipeline_lane": None,
        }
    ]
    assert all(family["parent_promoted"] for family in retained[1:])


def test_attention_structural_retention_replaces_dominated_family():
    config_ids = [f"{index:016x}" for index in range(6)]
    retained = compare_attention_backends._expected_flash_structural_retention(
        [
            _structural_leaf_qualification(
                "alpha",
                None,
                [
                    (config_ids[0], 1.0),
                    (config_ids[1], 1.1),
                    (config_ids[2], 1.2),
                ],
            ),
            _structural_leaf_qualification(
                "alpha", "deg2_16x6", [(config_ids[3], 1.3)]
            ),
            _structural_leaf_qualification("beta", None, [(config_ids[4], 1.4)]),
            _structural_leaf_qualification("dominated", None, [(config_ids[5], 2.01)]),
        ],
        retained_per_leaf=3,
        retained_family_cap=4,
        retained_family_limit=3,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=5,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )

    assert [family["family"] for family in retained] == ["alpha", "beta"]
    assert sum(len(family["starting_paths"]) for family in retained) == 5
    assert config_ids[5] not in {
        path["config_id"] for family in retained for path in family["starting_paths"]
    }


def test_attention_required_full_autotune_rejects_packet_owner_mismatch():
    phase = _full_autotune_trial()["search_phase_metrics"]
    leaf_result = cast("dict[str, Any]", phase["leaf_results"][0])
    leaf_result["compound_packet"] = "deg2_16x6"
    with pytest.raises(
        RuntimeError,
        match="invalid exact structural leaf result|malformed v22 ordinary",
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(search_phase_metrics=phase)],
        )


@pytest.mark.parametrize(
    "override",
    (
        {"num_configs_tested": 99},
        {"num_unique_sources": 0},
        {"num_generations": 0},
        {"num_generations": 19},
        {"num_generations": 21},
        {"search_phase_metrics": None},
    ),
)
def test_attention_required_full_autotune_rejects_incomplete_trial(override):
    with pytest.raises(RuntimeError):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(), [_full_autotune_trial(**override)]
        )


@pytest.mark.parametrize(
    "value",
    (None, True, -1, 1.0),
)
def test_attention_required_full_autotune_rejects_invalid_isolated_timeout_count(
    value,
):
    with pytest.raises(
        RuntimeError, match="invalid isolated rebenchmark timeout count"
    ):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(num_isolated_rebenchmark_timeouts=value)],
        )


def test_attention_required_full_autotune_allows_bounded_candidate_timeouts():
    _validate_full_autotune_trials(
        _full_autotune_trial_provenance(),
        [
            _full_autotune_trial(
                num_configs_tested=107,
                num_worker_failures=7,
                num_successful_candidate_measurements=100,
                num_unique_sources=107,
            )
        ],
    )


def test_attention_required_full_autotune_rejects_too_few_successes():
    with pytest.raises(RuntimeError, match="1 successful candidate measurements"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [
                _full_autotune_trial(
                    num_worker_failures=99,
                    num_successful_candidate_measurements=1,
                )
            ],
        )


def test_attention_required_full_autotune_does_not_count_failed_source_aliases():
    with pytest.raises(RuntimeError, match="99 successful candidate measurements"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [
                _full_autotune_trial(
                    num_successful_candidate_measurements=99,
                    num_source_deduplications=20,
                )
            ],
        )


def test_attention_required_full_autotune_checks_every_trial():
    provenance = _full_autotune_trial_provenance(autotune_best_of_k=2)

    with pytest.raises(RuntimeError, match="trial 2 covered only"):
        _validate_full_autotune_trials(
            provenance,
            [
                _full_autotune_trial(),
                _full_autotune_trial(
                    random_seed=124,
                    num_configs_tested=99,
                    selected_source_hash="b" * 64,
                ),
            ],
        )


def _exact_small_space_trial_provenance():
    trial = _full_autotune_trial(
        num_configs_tested=4,
        num_successful_candidate_measurements=4,
        num_unique_sources=4,
        num_generations=0,
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    exact_ids = cast("list[str]", phase["initial_config_ids"])[:4]
    phase["initial_config_ids"] = exact_ids
    phase["initial_config_count"] = len(exact_ids)
    phase["initial_results"] = [
        result
        for result in cast("list[dict[str, Any]]", phase["initial_results"])
        if result["config_id"] in exact_ids
    ]
    phase["config_manifest"] = {
        config_id: record
        for config_id, record in cast(
            "dict[str, dict[str, Any]]", phase["config_manifest"]
        ).items()
        if config_id in exact_ids
    }
    phase["exact_space_enumerated"] = True
    phase["exact_space_exhausted"] = True
    phase["exact_space_config_ids"] = exact_ids
    leaf["initial_config_ids"] = exact_ids
    leaf["space_exhausted"] = True
    leaf["space_config_count"] = len(exact_ids)
    leaf["ordinary_search_required"] = False
    leaf["qualified_results"] = [
        result
        for result in cast("list[dict[str, Any]]", leaf["qualified_results"])
        if result["config_id"] in exact_ids
    ]
    for lane in cast("list[dict[str, Any]]", leaf["pipeline_lanes"]):
        lane_record = {"key": lane["key"], "value": lane["value"]}
        lane["initial_config_ids"] = [
            result["config_id"]
            for result in cast("list[dict[str, Any]]", leaf["qualified_results"])
            if lane_record in result["pipeline_lanes"]
        ]
        lane["space_exhausted"] = True
        lane["space_config_count"] = len(lane["initial_config_ids"])
        lane["conditional_required"] = False
        lane["conditional_candidate_ids"] = []
        lane["successful_conditional_candidate_ids"] = []
        lane["witness_config_id"] = min(lane["initial_config_ids"])
        lane["rounds"][0]["candidate_config_ids"] = [lane["witness_config_id"]]
        lane["rounds"][1] = {
            "candidate_config_ids": [],
            "neighbor_generation_limit": 0,
        }
    for round_result in cast("list[dict[str, Any]]", leaf["rounds"]):
        round_result["candidate_config_ids"] = []
        round_result["neighbor_generation_limit"] = 0
        round_result["ordinary_neighbor_generation_limit"] = 0
    leaf["rounds"][0]["parent_decisions"] = [
        {
            "job_index": job_index,
            "kind": "witness",
            "pipeline_lane": {"key": lane["key"], "value": lane["value"]},
            "selection_kind": "ranked_existing",
            "candidate_results": [
                {
                    "config_id": config_id,
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "measurement_pass_index": 0,
                }
                for config_id in sorted(lane["initial_config_ids"])
            ],
            "selected_config_id": min(lane["initial_config_ids"]),
            "generated_config_ids": [],
        }
        for job_index, lane in enumerate(leaf["pipeline_lanes"])
    ]
    leaf["rounds"][1]["parent_decisions"] = []
    phase["measurement_timeline"] = [
        {
            "pass_index": 0,
            "updates": [
                {
                    "config_id": config_id,
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                }
                for config_id in sorted(exact_ids)
            ],
        },
        {"pass_index": 1, "updates": []},
        {"pass_index": 2, "updates": []},
    ]
    phase["candidate_count"] = 0
    phase["leaves_with_candidates"] = 0
    successful = [
        (
            cast("str", result["config_id"]),
            cast("float", result["selection_perf"]),
            frozenset(
                (cast("str", lane["key"]), cast("int", lane["value"]))
                for lane in cast("list[dict[str, Any]]", result["pipeline_lanes"])
            ),
        )
        for result in cast("list[dict[str, Any]]", leaf["qualified_results"])
    ]
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3)),
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [
                (
                    "fa4_2cta",
                    None,
                    False,
                    successful,
                    (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3)),
                )
            ],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"])
        for family in cast("list[dict[str, Any]]", phase["retained_families"])
    )
    provenance = _full_autotune_trial_provenance(
        flash_exact_effective_search_space_size=len(exact_ids),
        flash_exact_effective_search_space_config_ids=exact_ids,
        flash_exact_effective_search_space_sha256=hashlib.sha256(
            json.dumps(exact_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    _populate_measurement_source_hashes(phase)
    return trial, provenance, exact_ids


def test_attention_required_full_autotune_accepts_exact_small_space_exhaustion():
    trial, provenance, _exact_ids = _exact_small_space_trial_provenance()

    _validate_full_autotune_trials(provenance, [trial])


def _exact_small_space_trial_with_exhausted_clc():
    trial, provenance, exact_ids = _exact_small_space_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(_full_autotune_config_spec())
    expected_catalog = compare_attention_backends._flash_clc_lane_provenance(
        generation,
        leaf_catalog=[
            {
                "family": "fa4_clc",
                "compound_packet": None,
                "softmax_disc": False,
            }
        ],
    )
    assert len(expected_catalog) == 1
    expected_clc = expected_catalog[0]
    witnesses = list(
        cast("dict[str, str]", expected_clc["witness_config_ids"]).values()
    )
    witness_by_id = {
        hashlib.sha256(
            json.dumps(config.config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]: dict(config.config)
        for (leaf, _value), config in generation.flash_clc_lane_witnesses().items()
        if leaf.pipeline_family == "fa4_clc"
    }
    clc_configs = [witness_by_id[witness] for witness in witnesses]
    clc_configs.extend(
        compare_attention_backends._canonical_flash_projection(
            generation, config, {"cute_flash_kv_stage": 2}
        )
        for config in clc_configs[:2]
    )
    clc_ids = [
        hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        for config in clc_configs
    ]
    replacements = dict(zip(exact_ids, clc_ids, strict=True))

    def replace_ids(value):
        if isinstance(value, str):
            return replacements.get(value, value).replace("fa4_2cta", "fa4_clc")
        if isinstance(value, list):
            return [replace_ids(item) for item in value]
        if isinstance(value, dict):
            return {replace_ids(key): replace_ids(item) for key, item in value.items()}
        return value

    trial = replace_ids(trial)
    provenance = replace_ids(provenance)
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    phase["measurement_timeline"][0]["updates"].sort(
        key=operator.itemgetter("config_id")
    )
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])
    for config_id, config in zip(clc_ids, clc_configs, strict=True):
        manifest[config_id] = {"config": config}
    initial_results = cast("list[dict[str, Any]]", phase["initial_results"])
    qualified_results = cast(
        "list[dict[str, Any]]", phase["leaf_results"][0]["qualified_results"]
    )

    def memberships(config_id):
        value = manifest[config_id]["config"]["cute_flash_kv_stage"]
        return [{"key": "cute_flash_kv_stage", "value": value}]

    for result in [*initial_results, *qualified_results]:
        result["pipeline_lanes"] = memberships(result["config_id"])
    leaf_result = cast("dict[str, Any]", phase["leaf_results"][0])
    for lane in cast("list[dict[str, Any]]", leaf_result["pipeline_lanes"]):
        lane["initial_config_ids"] = [
            result["config_id"]
            for result in initial_results
            if {"key": lane["key"], "value": lane["value"]} in result["pipeline_lanes"]
        ]
        lane["witness_config_id"] = min(lane["initial_config_ids"])
        lane["rounds"][0]["candidate_config_ids"] = [lane["witness_config_id"]]
    leaf_result["rounds"][0]["parent_decisions"] = [
        {
            "job_index": job_index,
            "kind": "witness",
            "pipeline_lane": {"key": lane["key"], "value": lane["value"]},
            "selection_kind": "ranked_existing",
            "candidate_results": [
                {
                    "config_id": config_id,
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "measurement_pass_index": 0,
                }
                for config_id in sorted(lane["initial_config_ids"])
            ],
            "selected_config_id": min(lane["initial_config_ids"]),
            "generated_config_ids": [],
        }
        for job_index, lane in enumerate(leaf_result["pipeline_lanes"])
    ]
    exact_ids = cast("list[str]", phase["initial_config_ids"])
    provenance["flash_exact_effective_search_space_config_ids"] = exact_ids
    provenance["flash_exact_effective_search_space_sha256"] = hashlib.sha256(
        json.dumps(exact_ids, separators=(",", ":")).encode()
    ).hexdigest()
    provenance["flash_clc_lane_catalog"] = expected_catalog
    design_configs = clc_configs[:2]
    provenance["flash_structural_coverage_design"] = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for config in design_configs
    ]
    provenance["flash_structural_coverage_design_count"] = len(design_configs)
    provenance["flash_structural_coverage_design_sha256"] = hashlib.sha256(
        json.dumps(design_configs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance["flash_structural_injected_design_count"] = len(design_configs)
    provenance["flash_clc_lane_catalog_sha256"] = hashlib.sha256(
        json.dumps(expected_catalog, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    witness_candidate_results = [
        {
            "value": value,
            "config_id": expected_clc["witness_config_ids"][str(value)],
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "measurement_pass_index": 3,
        }
        for value in expected_clc["planned_values"]
    ]
    witness_selection_results = sorted(
        witness_candidate_results,
        key=operator.itemgetter("selection_perf", "config_id", "value"),
    )
    selected_values = [result["value"] for result in witness_selection_results]
    selected_config_ids = [result["config_id"] for result in witness_selection_results]
    phase["clc_families"] = [
        {
            "family": "fa4_clc",
            "softmax_disc": False,
            "space_exhausted": True,
            "legal_values": list(expected_clc["legal_values"]),
            "search_values": list(expected_clc["search_values"]),
            "anchor_values": list(expected_clc["anchor_values"]),
            "refinement_values": list(expected_clc["refinement_values"]),
            "planned_values": list(expected_clc["planned_values"]),
            "attempted_values": list(expected_clc["planned_values"]),
            "witness_config_ids": dict(expected_clc["witness_config_ids"]),
            "witness_repair_candidate_ids": {},
            "witness_repair_parent_decisions": [],
            "value_space_exhausted": {"1": True, "2": True},
            "witness_candidate_results": witness_candidate_results,
            "witness_selection_results": witness_selection_results,
            "selected_values": selected_values,
            "selected_config_ids": selected_config_ids,
            "conditional_values": [],
            "conditional_neighbor_generation_limit": 0,
            "conditional_parent_decisions": [],
            "conditional_repair_candidate_ids": {},
            "conditional_repair_parent_decisions": [],
            "retained_values": selected_values,
            "retained_config_ids": selected_config_ids,
            "retained_value_decisions": [
                {
                    "value": result["value"],
                    "candidate_results": [
                        {
                            key: result[key]
                            for key in (
                                "config_id",
                                "attempt_perf",
                                "selection_perf",
                                "status",
                                "measurement_pass_index",
                            )
                        }
                    ],
                    "selected_config_id": result["config_id"],
                }
                for result in witness_selection_results
            ],
            "retained_ranking_results": witness_selection_results,
            "conditional_candidate_ids": {},
            "combination_required": False,
            "depth_selection": {
                "candidate_results": [],
                "selected_representatives": [],
            },
            "combination_candidate_ids": [],
            "combination_depth_config_ids": [],
            "combination_divisor_values": [],
            "combination_cells": [],
            "combination_projection_complete": True,
            "successful_combination_depth_config_ids": [],
            "successful_combination_divisor_values": [],
            "combination_row_coverage_complete": True,
            "combination_column_coverage_complete": True,
            "combination_failure_statuses_allowed": True,
            "complete": True,
        }
    ]
    for key in (
        "qualification_rounds_started",
        "qualification_rounds_completed",
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
    ):
        phase[key] = 3
    for qualified in qualified_results:
        qualified["measurement_pass_index"] = 3
    phase["measurement_timeline"].append({"pass_index": 3, "updates": []})
    provenance["flash_structural_coverage_active_values"].append(
        {"key": "cute_flash_clc_heads_per_batch", "value": 1}
    )
    successful = [
        (
            cast("str", result["config_id"]),
            cast("float", result["selection_perf"]),
            frozenset(
                (cast("str", lane["key"]), cast("int", lane["value"]))
                for lane in cast("list[dict[str, Any]]", result["pipeline_lanes"])
            ),
        )
        for result in qualified_results
    ]
    pipeline_lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        pipeline_lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf_result["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [("fa4_clc", None, False, successful, pipeline_lanes)],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    _populate_measurement_source_hashes(phase, overwrite=True)
    _restore_full_autotune_terminal_fixture(trial, provenance)
    return trial, provenance


@functools.lru_cache(maxsize=1)
def _cached_full_autotune_trial_with_reused_clc_combination():
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(_full_autotune_config_spec())
    leaf = {
        "family": "fa4_clc",
        "compound_packet": None,
        "softmax_disc": False,
    }

    def config_id(config):
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    configs_by_depth: dict[int, list[dict[str, Any]]] = {2: [], 3: []}
    for depth in configs_by_depth:
        seen: set[str] = set()
        for source in _full_autotune_trial_configs():
            projected = compare_attention_backends._canonical_flash_projection(
                generation,
                source,
                {
                    "cute_flash_pipeline_family": "fa4_clc",
                    "cute_flash_clc_heads_per_batch": 1,
                    "cute_flash_kv_stage": depth,
                    "cute_flash_softmax_disc": False,
                },
            )
            projected_id = config_id(projected)
            if (
                compare_attention_backends._flash_structural_leaf_dict(projected)
                == leaf
                and projected_id not in seen
            ):
                seen.add(projected_id)
                configs_by_depth[depth].append(projected)
        for source in list(configs_by_depth[depth]):
            if len(configs_by_depth[depth]) >= 52:
                break
            first_load_order = cast("int", source["cute_flash_first_load_order"])
            for value in range(4):
                if value == first_load_order:
                    continue
                projected = compare_attention_backends._canonical_flash_projection(
                    generation,
                    source,
                    {"cute_flash_first_load_order": value},
                )
                projected_id = config_id(projected)
                if projected_id not in seen:
                    seen.add(projected_id)
                    configs_by_depth[depth].append(projected)
                    break
        assert len(configs_by_depth[depth]) >= 52

    configs = [
        config
        for pair in zip(configs_by_depth[2][:51], configs_by_depth[3][:51], strict=True)
        for config in pair
    ]
    ids = [config_id(config) for config in configs]
    assert len(ids) == len(set(ids)) == 102

    base_trial = _full_autotune_trial()
    base_ids = [config_id(config) for config in _full_autotune_trial_configs()]
    replacements = dict(zip(base_ids, ids, strict=True))

    def replace_base(value):
        if isinstance(value, str):
            return replacements.get(value, value).replace("fa4_2cta", "fa4_clc")
        if isinstance(value, list):
            return [replace_base(item) for item in value]
        if isinstance(value, dict):
            return {
                replace_base(key): replace_base(item) for key, item in value.items()
            }
        return value

    trial = replace_base(base_trial)
    provenance = replace_base(_full_autotune_trial_provenance())
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    for timeline_pass in phase["measurement_timeline"]:
        timeline_pass["updates"].sort(key=operator.itemgetter("config_id"))
    leaf_result = cast("dict[str, Any]", phase["leaf_results"][0])
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])
    manifest.clear()
    manifest.update(
        {
            candidate_id: {"config": config}
            for candidate_id, config in zip(ids, configs, strict=True)
        }
    )
    lanes = cast("list[dict[str, Any]]", leaf_result["pipeline_lanes"])
    for lane in lanes:
        lane["witness_config_id"] = min(lane["initial_config_ids"])
        lane["rounds"][0]["candidate_config_ids"] = [lane["witness_config_id"]]
    for round_index, round_result in enumerate(
        cast("list[dict[str, Any]]", leaf_result["rounds"])
    ):
        for decision_record, _lane in zip(
            cast("list[dict[str, Any]]", round_result["parent_decisions"]),
            lanes,
            strict=True,
        ):
            decision_record["candidate_results"] = sorted(
                decision_record["candidate_results"],
                key=operator.itemgetter("selection_perf", "config_id"),
            )
            decision_record["selected_config_id"] = decision_record[
                "candidate_results"
            ][0]["config_id"]
            if round_index == 0:
                decision_record["generated_config_ids"] = []

    expected_catalog = compare_attention_backends._flash_clc_lane_provenance(
        generation, leaf_catalog=[leaf]
    )
    assert len(expected_catalog) == 1
    expected_clc = expected_catalog[0]
    witness_configs = {
        config_id(dict(config.config)): dict(config.config)
        for (witness_leaf, _value), config in (
            generation.flash_clc_lane_witnesses().items()
        )
        if witness_leaf.pipeline_family == "fa4_clc"
    }
    witness_ids = cast("dict[str, str]", expected_clc["witness_config_ids"])
    assert set(witness_ids.values()) <= set(witness_configs)

    pre_combination_configs = dict(zip(ids, configs, strict=True))
    pre_combination_configs.update(
        (witness_id, witness_configs[witness_id]) for witness_id in witness_ids.values()
    )
    conditional_configs: dict[int, tuple[str, dict[str, Any]]] = {}
    for value in cast("list[int]", expected_clc["planned_values"]):
        for source in [
            *configs_by_depth[2],
            *configs_by_depth[3],
        ]:
            projected = compare_attention_backends._canonical_flash_projection(
                generation,
                source,
                {"cute_flash_clc_heads_per_batch": value},
            )
            projected_id = config_id(projected)
            if (
                compare_attention_backends._flash_structural_leaf_dict(projected)
                == leaf
                and projected_id not in pre_combination_configs
            ):
                conditional_configs[value] = (projected_id, projected)
                pre_combination_configs[projected_id] = projected
                break
        assert value in conditional_configs

    def decision(candidate_id, measurement_pass_index):
        return {
            "config_id": candidate_id,
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "measurement_pass_index": measurement_pass_index,
        }

    qualified_by_id = {
        result["config_id"]: result
        for result in cast("list[dict[str, Any]]", leaf_result["qualified_results"])
    }

    def add_qualified(candidate_id, config):
        manifest[candidate_id] = {"config": config}
        qualified_by_id.setdefault(
            candidate_id,
            {
                **decision(candidate_id, 5),
                "pipeline_lanes": [
                    {
                        "key": "cute_flash_kv_stage",
                        "value": config["cute_flash_kv_stage"],
                    }
                ],
            },
        )

    for candidate_id, config in pre_combination_configs.items():
        add_qualified(candidate_id, config)

    witness_candidate_results = [
        {"value": value, **decision(witness_ids[str(value)], 3)}
        for value in cast("list[int]", expected_clc["planned_values"])
    ]
    witness_selection_results = sorted(
        witness_candidate_results,
        key=operator.itemgetter("selection_perf", "config_id", "value"),
    )
    selected_values = [result["value"] for result in witness_selection_results]
    selected_config_ids = [result["config_id"] for result in witness_selection_results]
    conditional_ids = {
        str(value): [conditional_configs[value][0]] for value in selected_values
    }
    value_decisions = []
    chosen_by_value = {}
    for value in selected_values:
        candidates = sorted(
            [
                decision(conditional_configs[value][0], 4),
                decision(witness_ids[str(value)], 4),
            ],
            key=operator.itemgetter("selection_perf", "config_id"),
        )
        chosen_by_value[value] = candidates[0]
        value_decisions.append(
            {
                "value": value,
                "candidate_results": candidates,
                "selected_config_id": candidates[0]["config_id"],
            }
        )
    retained_ranking = sorted(
        [{"value": value, **snapshot} for value, snapshot in chosen_by_value.items()],
        key=operator.itemgetter("selection_perf", "config_id", "value"),
    )
    retained_values = [result["value"] for result in retained_ranking]
    retained_config_ids = [result["config_id"] for result in retained_ranking]

    depth_candidate_ids = sorted(pre_combination_configs)
    depth_candidates = [
        decision(candidate_id, 4) for candidate_id in depth_candidate_ids
    ]
    pipeline_lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    depth_members = [
        (
            candidate_id,
            1.0,
            frozenset(
                lane
                for lane in pipeline_lanes
                if pre_combination_configs[candidate_id][lane[0]] == lane[1]
            ),
        )
        for candidate_id in depth_candidate_ids
    ]
    depth_representatives = (
        compare_attention_backends._expected_flash_lane_diverse_members(
            depth_members,
            pipeline_lanes,
            limit=2,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    depth_selection = {
        "candidate_results": depth_candidates,
        "selected_representatives": [
            {
                "config_id": member[0],
                "assigned_pipeline_lane": (
                    None if lane is None else {"key": lane[0], "value": lane[1]}
                ),
            }
            for member, lane in depth_representatives
        ],
    }

    combination_depth_ids = [member[0] for member, _lane in depth_representatives]
    combination_ids = []
    combination_cells = []
    seen_combination_configs: set[str] = set()
    for member, _lane in depth_representatives:
        source = pre_combination_configs[member[0]]
        for value in retained_values:
            projected = compare_attention_backends._canonical_flash_projection(
                generation,
                source,
                {"cute_flash_clc_heads_per_batch": value},
            )
            projected_key = json.dumps(projected, sort_keys=True, separators=(",", ":"))
            projected_id = config_id(projected)
            combination_cells.append(
                {
                    "depth_config_id": member[0],
                    "divisor_value": value,
                    "projected_config_id": projected_id,
                    **decision(projected_id, 5),
                }
            )
            if projected_key in seen_combination_configs:
                continue
            seen_combination_configs.add(projected_key)
            combination_ids.append(projected_id)
            add_qualified(projected_id, projected)
    reused_ids = set(combination_ids) & set(depth_candidate_ids)
    assert reused_ids

    phase["clc_families"] = [
        {
            "family": "fa4_clc",
            "softmax_disc": False,
            "space_exhausted": False,
            "legal_values": list(expected_clc["legal_values"]),
            "search_values": list(expected_clc["search_values"]),
            "anchor_values": list(expected_clc["anchor_values"]),
            "refinement_values": list(expected_clc["refinement_values"]),
            "planned_values": list(expected_clc["planned_values"]),
            "attempted_values": list(expected_clc["planned_values"]),
            "witness_config_ids": dict(witness_ids),
            "witness_repair_candidate_ids": {},
            "witness_repair_parent_decisions": [],
            "value_space_exhausted": {
                str(value): False
                for value in cast("list[int]", expected_clc["planned_values"])
            },
            "witness_candidate_results": witness_candidate_results,
            "witness_selection_results": witness_selection_results,
            "selected_values": selected_values,
            "selected_config_ids": selected_config_ids,
            "conditional_values": selected_values,
            "conditional_neighbor_generation_limit": max(200, len(selected_values)),
            "conditional_parent_decisions": [
                {
                    "value": value,
                    "candidate_results": [decision(witness_ids[str(value)], 3)],
                    "selected_config_id": witness_ids[str(value)],
                    "generated_config_ids": conditional_ids[str(value)],
                    "neighbor_generation_limit": (
                        (index + 1) * 200 // len(selected_values)
                        - index * 200 // len(selected_values)
                    ),
                }
                for index, value in enumerate(selected_values)
            ],
            "conditional_repair_candidate_ids": {},
            "conditional_repair_parent_decisions": [],
            "retained_values": retained_values,
            "retained_config_ids": retained_config_ids,
            "retained_value_decisions": value_decisions,
            "retained_ranking_results": retained_ranking,
            "conditional_candidate_ids": conditional_ids,
            "combination_required": True,
            "depth_selection": depth_selection,
            "combination_candidate_ids": combination_ids,
            "combination_depth_config_ids": combination_depth_ids,
            "combination_divisor_values": retained_values,
            "combination_cells": combination_cells,
            "combination_projection_complete": True,
            "successful_combination_depth_config_ids": combination_depth_ids,
            "successful_combination_divisor_values": retained_values,
            "combination_row_coverage_complete": True,
            "combination_column_coverage_complete": True,
            "combination_failure_statuses_allowed": True,
            "complete": True,
        }
    ]
    leaf_result["qualified_results"] = list(qualified_by_id.values())
    successful = [
        (
            result["config_id"],
            result["selection_perf"],
            frozenset(
                (lane["key"], lane["value"]) for lane in result["pipeline_lanes"]
            ),
        )
        for result in leaf_result["qualified_results"]
    ]
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        pipeline_lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf_result["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [("fa4_clc", None, False, successful, pipeline_lanes)],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    phase["candidate_count"] = len(qualified_by_id) - phase["initial_config_count"]
    phase["leaves_with_candidates"] = 1
    for key in (
        "qualification_rounds_started",
        "qualification_rounds_completed",
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
    ):
        phase[key] = 5
    for qualified in leaf_result["qualified_results"]:
        qualified["measurement_pass_index"] = 5

    measured_before_witness = set(ids)
    witness_updates = sorted(set(witness_ids.values()) - measured_before_witness)
    measured_before_conditional = measured_before_witness | set(witness_ids.values())
    conditional_updates = sorted(
        {
            config_id
            for config_ids_for_value in conditional_ids.values()
            for config_id in config_ids_for_value
        }
        - measured_before_conditional
    )
    measured_before_combination = measured_before_conditional | set(conditional_updates)
    combination_updates = sorted(set(combination_ids) - measured_before_combination)

    def timeline_updates(config_ids_for_pass):
        return [
            {
                "config_id": candidate_id,
                "attempt_perf": 1.0,
                "selection_perf": 1.0,
                "status": "ok",
            }
            for candidate_id in config_ids_for_pass
        ]

    phase["measurement_timeline"].extend(
        [
            {"pass_index": 3, "updates": timeline_updates(witness_updates)},
            {"pass_index": 4, "updates": timeline_updates(conditional_updates)},
            {"pass_index": 5, "updates": timeline_updates(combination_updates)},
        ]
    )

    for key in (
        "num_configs_tested",
        "num_successful_candidate_measurements",
        "num_unique_sources",
    ):
        trial[key] = len(qualified_by_id)
    provenance["flash_clc_lane_catalog"] = expected_catalog
    provenance["flash_clc_lane_catalog_sha256"] = hashlib.sha256(
        json.dumps(expected_catalog, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    design_configs = configs[:2]
    provenance["flash_structural_coverage_design"] = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for config in design_configs
    ]
    provenance["flash_structural_coverage_design_count"] = len(design_configs)
    provenance["flash_structural_coverage_design_sha256"] = hashlib.sha256(
        json.dumps(design_configs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance["flash_structural_injected_design_count"] = len(design_configs)
    provenance["flash_structural_coverage_active_values"].append(
        {"key": "cute_flash_clc_heads_per_batch", "value": 1}
    )
    _restore_full_autotune_terminal_fixture(trial, provenance)
    _populate_measurement_source_hashes(phase, overwrite=True)
    return trial, provenance, min(reused_ids)


def _full_autotune_trial_with_reused_clc_combination():
    return copy.deepcopy(_cached_full_autotune_trial_with_reused_clc_combination())


def _full_autotune_trial_with_retried_empty_clc_conditional():
    trial, provenance, reused_id = _full_autotune_trial_with_reused_clc_combination()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    clc = cast("dict[str, Any]", phase["clc_families"][0])
    value = cast("int", clc["conditional_values"][0])
    value_key = str(value)
    candidate_id = cast("str", clc["conditional_candidate_ids"][value_key][0])
    witness_id = cast("str", clc["witness_config_ids"][value_key])

    clc["conditional_candidate_ids"][value_key] = []
    primary_decision = next(
        decision
        for decision in cast(
            "list[dict[str, Any]]", clc["conditional_parent_decisions"]
        )
        if decision["value"] == value
    )
    primary_decision["generated_config_ids"] = []
    repair_parent = copy.deepcopy(primary_decision["candidate_results"][0])
    repair_parent["measurement_pass_index"] = 4
    clc["conditional_repair_candidate_ids"] = {value_key: [candidate_id]}
    clc["conditional_repair_parent_decisions"] = [
        {
            "kind": "conditional_failure_repair",
            "value": value,
            "repair_index": 0,
            "candidate_results": [repair_parent],
            "selected_config_id": witness_id,
            "generated_config_ids": [candidate_id],
            "neighbor_generation_limit": 200,
        }
    ]

    for decision in cast("list[dict[str, Any]]", clc["retained_value_decisions"]):
        for candidate in decision["candidate_results"]:
            candidate["measurement_pass_index"] = 5
    for result in cast("list[dict[str, Any]]", clc["retained_ranking_results"]):
        result["measurement_pass_index"] = 5
    depth_selection = cast("dict[str, Any]", clc["depth_selection"])
    for result in cast("list[dict[str, Any]]", depth_selection["candidate_results"]):
        result["measurement_pass_index"] = 5
    for cell in cast("list[dict[str, Any]]", clc["combination_cells"]):
        cell["measurement_pass_index"] = 6

    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    for result in cast("list[dict[str, Any]]", leaf["qualified_results"]):
        result["measurement_pass_index"] = 6
    for key in (
        "qualification_rounds_started",
        "qualification_rounds_completed",
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
    ):
        phase[key] = 6

    timeline = cast("list[dict[str, Any]]", phase["measurement_timeline"])
    conditional_pass = next(item for item in timeline if item["pass_index"] == 4)
    moved_update = next(
        update
        for update in cast("list[dict[str, Any]]", conditional_pass["updates"])
        if update["config_id"] == candidate_id
    )
    conditional_pass["updates"].remove(moved_update)
    combination_pass = next(item for item in timeline if item["pass_index"] == 5)
    combination_pass["pass_index"] = 6
    timeline.insert(
        timeline.index(combination_pass),
        {"pass_index": 5, "updates": [moved_update]},
    )
    return trial, provenance, reused_id


def _full_autotune_trial_with_failed_clc_combination():
    trial, provenance, reused_id = _full_autotune_trial_with_reused_clc_combination()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    clc = cast("dict[str, Any]", phase["clc_families"][0])
    depth_ids = set(cast("list[str]", clc["combination_depth_config_ids"]))
    failed_cell = next(
        cell
        for cell in cast("list[dict[str, Any]]", clc["combination_cells"])
        if cell["config_id"] not in depth_ids
    )
    failed_id = cast("str", failed_cell["config_id"])
    failed_cell.update(
        attempt_perf=None,
        selection_perf=None,
        status="error",
    )
    qualified = cast("list[dict[str, Any]]", leaf["qualified_results"])
    next(result for result in qualified if result["config_id"] == failed_id).update(
        attempt_perf=None,
        selection_perf=None,
        status="error",
    )
    timeline_update = next(
        update
        for timeline_pass in cast("list[dict[str, Any]]", phase["measurement_timeline"])
        for update in cast("list[dict[str, Any]]", timeline_pass["updates"])
        if update["config_id"] == failed_id
    )
    timeline_update.update(
        attempt_perf=None,
        selection_perf=None,
        status="error",
    )

    pipeline_lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    successful = [
        (
            result["config_id"],
            result["selection_perf"],
            frozenset(
                (membership["key"], membership["value"])
                for membership in result["pipeline_lanes"]
            ),
        )
        for result in qualified
        if result["status"] in {"ok", "deduplicated"}
    ]
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        pipeline_lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [("fa4_clc", None, False, successful, pipeline_lanes)],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )
    trial["num_worker_failures"] += 1
    trial["num_successful_candidate_measurements"] -= 1
    return trial, provenance, reused_id, failed_cell


def test_attention_required_full_autotune_accepts_exhausted_clc_without_children():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()

    _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_fabricated_clc_legal_value():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    phase["clc_families"][0]["legal_values"].append(999)

    with pytest.raises(RuntimeError, match="CLC family record"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_narrowed_clc_search_values():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    phase["clc_families"][0]["search_values"].pop()

    with pytest.raises(RuntimeError, match="CLC family record"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_recomputes_clc_catalog():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()
    provenance["flash_clc_lane_catalog"] = []
    provenance["flash_clc_lane_catalog_sha256"] = hashlib.sha256(b"[]").hexdigest()

    with pytest.raises(RuntimeError, match="CLC provenance.*live ConfigGeneration"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_reordered_clc_snapshot():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()
    clc = trial["search_phase_metrics"]["clc_families"][0]
    clc["witness_selection_results"].reverse()

    with pytest.raises(RuntimeError, match="immutable CLC witness decision"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_clc_repair_passes_must_use_compact_batches():
    assert compare_attention_backends._flash_repair_passes_are_compact(
        [7, 7, 7, 7, 8], per_pass_limit=4
    )
    assert not compare_attention_backends._flash_repair_passes_are_compact(
        [7, 7, 7, 8, 8], per_pass_limit=4
    )


def test_attention_clc_depth_candidates_keep_reused_combinations():
    leaf = {
        "family": "fa4_clc",
        "compound_packet": None,
        "softmax_disc": False,
    }
    reused_combination_id = "a" * 16
    novel_combination_id = "b" * 16
    other_depth_id = "c" * 16
    successful_ids = {
        reused_combination_id,
        novel_combination_id,
        other_depth_id,
    }
    manifest_leaves = dict.fromkeys(successful_ids, leaf)

    expected = compare_attention_backends._expected_flash_clc_depth_candidate_ids(
        successful_ids,
        manifest_leaves,
        leaf,
        {reused_combination_id, other_depth_id},
    )

    assert expected == {reused_combination_id, other_depth_id}


def test_attention_required_full_autotune_accepts_reused_clc_combination():
    trial, provenance, reused_id = _full_autotune_trial_with_reused_clc_combination()
    clc = cast("dict[str, Any]", trial["search_phase_metrics"]["clc_families"][0])

    assert reused_id in clc["combination_candidate_ids"]
    assert reused_id in {
        result["config_id"] for result in clc["depth_selection"]["candidate_results"]
    }
    _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_accepts_retried_empty_clc_conditional():
    trial, provenance, _reused_id = (
        _full_autotune_trial_with_retried_empty_clc_conditional()
    )
    clc = cast("dict[str, Any]", trial["search_phase_metrics"]["clc_families"][0])

    repaired_value = cast("int", clc["conditional_repair_parent_decisions"][0]["value"])
    assert clc["conditional_candidate_ids"][str(repaired_value)] == []
    assert clc["conditional_repair_candidate_ids"][str(repaired_value)]
    _validate_full_autotune_trials(provenance, [trial])


@pytest.mark.parametrize("neighbor_limit", (99, 100.0))
def test_attention_required_full_autotune_rejects_wrong_clc_neighbor_allocation(
    neighbor_limit,
):
    trial, provenance, _reused_id = _full_autotune_trial_with_reused_clc_combination()
    clc = cast("dict[str, Any]", trial["search_phase_metrics"]["clc_families"][0])
    clc["conditional_parent_decisions"][0]["neighbor_generation_limit"] = neighbor_limit

    with pytest.raises(RuntimeError, match="CLC conditional-parent decision"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_accepts_failed_clc_combination_with_coverage():
    trial, provenance, _reused_id, failed_cell = (
        _full_autotune_trial_with_failed_clc_combination()
    )
    clc = cast("dict[str, Any]", trial["search_phase_metrics"]["clc_families"][0])

    assert len(clc["combination_cells"]) == 4
    assert failed_cell["status"] == "error"
    assert clc["combination_row_coverage_complete"] is True
    assert clc["combination_column_coverage_complete"] is True
    _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_uncovered_clc_combination_axis():
    trial, provenance, _reused_id, failed_cell = (
        _full_autotune_trial_with_failed_clc_combination()
    )
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    clc = cast("dict[str, Any]", phase["clc_families"][0])
    failed_column = failed_cell["divisor_value"]
    second_cell = next(
        cell
        for cell in cast("list[dict[str, Any]]", clc["combination_cells"])
        if cell["divisor_value"] == failed_column
        and cell["config_id"] != failed_cell["config_id"]
    )
    clc["combination_cells"].remove(second_cell)

    with pytest.raises(
        RuntimeError, match="CLC.*coverage|CLC depth/divisor combinations"
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_malformed_clc_combination_cell():
    trial, provenance, _reused_id = _full_autotune_trial_with_reused_clc_combination()
    clc = cast("dict[str, Any]", trial["search_phase_metrics"]["clc_families"][0])
    clc["combination_cells"][0].pop("selection_perf")

    with pytest.raises(RuntimeError, match="CLC combination cell"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_dropped_reused_clc_depth():
    trial, provenance, reused_id = _full_autotune_trial_with_reused_clc_combination()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    clc = cast("dict[str, Any]", phase["clc_families"][0])
    depth_selection = cast("dict[str, Any]", clc["depth_selection"])
    depth_selection["candidate_results"] = [
        result
        for result in depth_selection["candidate_results"]
        if result["config_id"] != reused_id
    ]
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])
    lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    members = [
        (
            result["config_id"],
            result["selection_perf"],
            frozenset(
                lane
                for lane in lanes
                if manifest[result["config_id"]]["config"][lane[0]] == lane[1]
            ),
        )
        for result in depth_selection["candidate_results"]
    ]
    representatives = compare_attention_backends._expected_flash_lane_diverse_members(
        members,
        lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    depth_selection["selected_representatives"] = [
        {
            "config_id": member[0],
            "assigned_pipeline_lane": (
                None if lane is None else {"key": lane[0], "value": lane[1]}
            ),
        }
        for member, lane in representatives
    ]

    with pytest.raises(RuntimeError, match="incomplete immutable CLC depth decision"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_uses_stage_clc_measurements():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    qualified = cast("list[dict[str, Any]]", leaf["qualified_results"])
    selection_perf_by_id = {
        config_id: float(index + 1)
        for index, config_id in enumerate(
            sorted(cast("str", result["config_id"]) for result in qualified)
        )
    }

    def update_measurement_scores(value):
        if isinstance(value, dict):
            config_id = value.get("config_id")
            if config_id in selection_perf_by_id and "selection_perf" in value:
                value["selection_perf"] = selection_perf_by_id[config_id]
            for child in value.values():
                update_measurement_scores(child)
        elif isinstance(value, list):
            for child in value:
                update_measurement_scores(child)

    update_measurement_scores(phase)
    successful = [
        (
            cast("str", result["config_id"]),
            cast("float", result["selection_perf"]),
            frozenset(
                (cast("str", lane["key"]), cast("int", lane["value"]))
                for lane in result["pipeline_lanes"]
            ),
        )
        for result in qualified
    ]
    lanes = (("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3))
    retained = compare_attention_backends._expected_flash_lane_diverse_members(
        successful,
        lanes,
        limit=2,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
    )
    leaf["retained_config_ids"] = [member[0] for member, _lane in retained]
    phase["retained_families"] = (
        compare_attention_backends._expected_flash_structural_retention(
            [("fa4_clc", None, False, successful, lanes)],
            retained_per_leaf=2,
            retained_family_cap=4,
            retained_family_limit=4,
            retained_family_slowdown_limit=2.0,
            starting_path_limit=14,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
    )
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in phase["retained_families"]
    )

    _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_dropped_initial_leaf_member():
    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    victim = cast("str", phase["initial_config_ids"][-1])
    leaf["initial_config_ids"].remove(victim)
    leaf["qualified_results"] = [
        item for item in leaf["qualified_results"] if item["config_id"] != victim
    ]
    for lane in leaf["pipeline_lanes"]:
        if victim in lane["initial_config_ids"]:
            lane["initial_config_ids"].remove(victim)

    with pytest.raises(
        RuntimeError,
        match="inconsistent v22 initial population|omits a generation-zero measurement",
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_tampered_random_fill():
    from helion.autotuner.config_generation import ConfigGeneration

    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    protected_ids = set(cast("list[str]", leaf["retained_config_ids"]))
    for lane in cast("list[dict[str, Any]]", leaf["pipeline_lanes"]):
        protected_ids.add(cast("str", lane["witness_config_id"]))
        protected_ids.update(cast("list[str]", lane["conditional_candidate_ids"]))
        protected_ids.update(cast("list[str]", lane["repair_candidate_ids"]))
    for round_result in cast("list[dict[str, Any]]", leaf["rounds"]):
        for decision in cast("list[dict[str, Any]]", round_result["parent_decisions"]):
            selected_id = decision["selected_config_id"]
            if selected_id is not None:
                protected_ids.add(cast("str", selected_id))
    initial_ids = cast("list[str]", phase["initial_config_ids"])
    victim_id = next(
        config_id
        for config_id in reversed(initial_ids)
        if config_id not in protected_ids
    )
    manifest = cast("dict[str, dict[str, Any]]", phase["config_manifest"])
    victim_config = cast("dict[str, object]", manifest[victim_id]["config"])
    generation = ConfigGeneration(_full_autotune_config_spec())
    replacement_config = compare_attention_backends._canonical_flash_projection(
        generation,
        victim_config,
        {
            "cute_flash_first_load_order": (
                cast("int", victim_config["cute_flash_first_load_order"]) + 1
            )
            % 4
        },
    )
    replacement_id = hashlib.sha256(
        json.dumps(replacement_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert replacement_id not in manifest

    def replace_config_id(value):
        if isinstance(value, dict):
            for key, child in value.items():
                value[key] = replace_config_id(child)
            updates = value.get("updates")
            if isinstance(updates, list):
                updates.sort(key=operator.itemgetter("config_id"))
            candidates = value.get("candidate_results")
            if isinstance(candidates, list):
                candidates.sort(
                    key=lambda item: (
                        item["selection_perf"]
                        if item["selection_perf"] is not None
                        else float("inf"),
                        item["config_id"],
                    )
                )
            return value
        if isinstance(value, list):
            return [replace_config_id(child) for child in value]
        return replacement_id if value == victim_id else value

    manifest.pop(victim_id)
    replace_config_id(phase)
    manifest[replacement_id] = {"config": replacement_config}

    with pytest.raises(RuntimeError, match="inconsistent v22 initial population"):
        _validate_full_autotune_trials(provenance, [trial])

    _validate_full_autotune_trials(
        provenance,
        [trial],
        expected_fixture_trial=trial,
    )


def test_attention_strict_initial_population_replay_uses_seed_without_rng_leak():
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(_full_autotune_config_spec())
    original_state = random.getstate()
    try:
        random.seed(987654321)
        saved_state = random.getstate()

        first_ids = compare_attention_backends._replay_strict_attention_initial_population_config_ids(
            generation,
            random_seed=123,
            initial_population_size=100,
        )
        assert random.getstate() == saved_state
        second_ids = compare_attention_backends._replay_strict_attention_initial_population_config_ids(
            generation,
            random_seed=124,
            initial_population_size=100,
        )

        assert random.getstate() == saved_state
        assert isinstance(first_ids, list)
        assert len(first_ids) == len(second_ids) == 100
        assert first_ids != second_ids
    finally:
        random.setstate(original_state)


def test_attention_strict_initial_population_replay_requires_exact_order():
    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    initial_ids = cast("list[str]", phase["initial_config_ids"])
    initial_ids[0], initial_ids[1] = initial_ids[1], initial_ids[0]

    with pytest.raises(RuntimeError, match="inconsistent v22 initial population"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_derives_initial_lane_from_manifest():
    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    first, second = cast("list[dict[str, Any]]", phase["initial_results"])[:2]
    first["pipeline_lanes"], second["pipeline_lanes"] = (
        second["pipeline_lanes"],
        first["pipeline_lanes"],
    )
    qualified = {item["config_id"]: item for item in leaf["qualified_results"]}
    qualified[first["config_id"]]["pipeline_lanes"] = first["pipeline_lanes"]
    qualified[second["config_id"]]["pipeline_lanes"] = second["pipeline_lanes"]
    for lane in leaf["pipeline_lanes"]:
        lane["initial_config_ids"] = [
            item["config_id"]
            for item in phase["initial_results"]
            if {"key": lane["key"], "value": lane["value"]} in item["pipeline_lanes"]
        ]

    with pytest.raises(RuntimeError, match="pipeline lane membership"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_reused_pipeline_conditional():
    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    lanes = cast("list[dict[str, Any]]", leaf["pipeline_lanes"])
    reused_id = lanes[0]["conditional_candidate_ids"][0]
    lanes[1]["conditional_candidate_ids"] = [reused_id]
    lanes[1]["successful_conditional_candidate_ids"] = [reused_id]
    lanes[1]["rounds"][1]["candidate_config_ids"] = [reused_id]

    with pytest.raises(RuntimeError, match="incomplete v22 pipeline lane"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_incomplete_parent_snapshot():
    trial = _full_autotune_trial()
    provenance = _full_autotune_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    decision = leaf["rounds"][0]["parent_decisions"][0]
    decision["candidate_results"].pop()

    with pytest.raises(RuntimeError, match="incomplete immutable pipeline parent"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_recomputes_exact_space():
    trial, provenance, _exact_ids = _exact_small_space_trial_provenance()
    provenance["flash_exact_effective_search_space_config_ids"] = provenance[
        "flash_exact_effective_search_space_config_ids"
    ][:-1]
    provenance["flash_exact_effective_search_space_size"] -= 1
    provenance["flash_exact_effective_search_space_sha256"] = hashlib.sha256(
        json.dumps(
            provenance["flash_exact_effective_search_space_config_ids"],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    with pytest.raises(RuntimeError, match="live ConfigGeneration"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_exhausted_clc_combination():
    trial, provenance = _exact_small_space_trial_with_exhausted_clc()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    clc = cast("dict[str, Any]", phase["clc_families"][0])
    clc["combination_candidate_ids"] = [phase["initial_config_ids"][0]]

    with pytest.raises(RuntimeError, match="CLC family record"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_invalid_exact_space_digest():
    trial, provenance, _exact_ids = _exact_small_space_trial_provenance()
    provenance["flash_exact_effective_search_space_sha256"] = "0" * 64

    with pytest.raises(
        RuntimeError,
        match="effective-space provenance|invalid exact effective search space",
    ):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_uncovered_exact_config():
    trial, provenance, exact_ids = _exact_small_space_trial_provenance()
    phase = cast("dict[str, Any]", trial["search_phase_metrics"])
    leaf = cast("dict[str, Any]", phase["leaf_results"][0])
    leaf["qualified_results"] = [
        result
        for result in cast("list[dict[str, Any]]", leaf["qualified_results"])
        if result["config_id"] != exact_ids[-1]
    ]

    with pytest.raises(RuntimeError):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_bounds_exact_space_generations():
    trial, provenance, _exact_ids = _exact_small_space_trial_provenance()
    trial["num_generations"] = 21

    with pytest.raises(RuntimeError, match="at most 20"):
        _validate_full_autotune_trials(provenance, [trial])


def test_attention_required_full_autotune_rejects_missing_trial():
    provenance = _full_autotune_trial_provenance(autotune_best_of_k=2)

    with pytest.raises(RuntimeError, match="recorded 1 trials, expected 2"):
        _validate_full_autotune_trials(provenance, [_full_autotune_trial()])


@pytest.mark.parametrize("second_seed", (123, 124.0, 125, None, True))
def test_attention_required_full_autotune_rejects_wrong_trial_seed(second_seed):
    provenance = _full_autotune_trial_provenance(autotune_best_of_k=2)

    with pytest.raises(RuntimeError, match="trial 2 recorded random seed"):
        _validate_full_autotune_trials(
            provenance,
            [
                _full_autotune_trial(),
                _full_autotune_trial(
                    random_seed=second_seed,
                    selected_source_hash="b" * 64,
                ),
            ],
        )


def test_attention_required_full_autotune_accepts_consecutive_trial_seeds():
    provenance = _full_autotune_trial_provenance(autotune_best_of_k=2)

    _validate_full_autotune_trials(
        provenance,
        [
            _full_autotune_trial(),
            _full_autotune_trial(
                random_seed=124,
                selected_source_hash="b" * 64,
            ),
        ],
    )


def test_attention_required_full_autotune_rejects_unmeasured_trial_winner():
    with pytest.raises(RuntimeError, match="did not link its winner"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(),
            [_full_autotune_trial(selected_source_was_measured=False)],
        )


@pytest.mark.parametrize(
    "provenance_override",
    (
        {"selected_source_sha256": "b" * 64},
        {"selected_config": {"block_sizes": [1, 64, 128]}},
    ),
)
def test_attention_required_full_autotune_rejects_unlinked_final_winner(
    provenance_override,
):
    with pytest.raises(RuntimeError, match="not the measured winner"):
        _validate_full_autotune_trials(
            _full_autotune_trial_provenance(**provenance_override),
            [_full_autotune_trial()],
        )


def test_attention_required_full_autotune_rejects_incorrect_winner():
    with pytest.raises(RuntimeError, match="failed final correctness"):
        compare_attention_backends._validate_required_full_autotune_correctness(
            _full_autotune_provenance(), "FAIL"
        )


@pytest.mark.parametrize(
    "provenance_override, message",
    (
        ({"final_correctness_launches": 1}, "repeated final correctness"),
        ({"final_repeatability_passed": False}, "exact repeatability"),
    ),
)
def test_attention_required_full_autotune_requires_repeatability(
    provenance_override, message
):
    with pytest.raises(RuntimeError, match=message):
        compare_attention_backends._validate_required_full_autotune_correctness(
            _full_autotune_provenance(**provenance_override), "PASS"
        )


def test_attention_result_correctness_checks_output_and_lse() -> None:
    output = torch.ones((2, 3), dtype=torch.float16)
    lse = torch.ones((2,), dtype=torch.float32)
    assert compare_attention_backends._check_attention_result_close(
        (output, lse), (output.clone(), lse.clone()), torch.float16
    )
    assert not compare_attention_backends._check_attention_result_close(
        (output, lse), (output.clone(), lse + 1), torch.float16
    )
    assert not compare_attention_backends._check_attention_result_close(
        output.float(), output, torch.float16
    )
    assert not compare_attention_backends._check_attention_result_close(
        output[:1], output, torch.float16
    )


def test_attention_chunked_correctness_handles_noncontiguous_full_output() -> None:
    expected = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5).transpose(1, 2)
    actual = expected.clone()

    assert not actual.is_contiguous()
    assert compare_attention_backends._check_tensor_close_in_chunks(
        actual,
        expected,
        max_temp_bytes=2 * 4 * 20,
    )

    actual[:, -1, -1] += 1.0
    assert not compare_attention_backends._check_tensor_close_in_chunks(
        actual,
        expected,
        max_temp_bytes=2 * 4 * 20,
    )


def test_attention_chunked_correctness_rejects_nonfinite_values() -> None:
    expected = torch.tensor([[float("inf")]], dtype=torch.float32)

    assert not compare_attention_backends._check_close(
        expected.clone(), expected, torch.float32
    )


def test_attention_huge_shape_correctness_uses_bounded_row_chunks() -> None:
    output = torch.empty(
        (8, 32, 786432, 64),
        dtype=torch.bfloat16,
        device="meta",  # @ignore-device-lint
    )

    chunk_rows = compare_attention_backends._comparison_chunk_rows(output)
    elements_per_row = output.numel() // output.shape[-2]

    assert 1 <= chunk_rows < output.shape[-2]
    assert (
        chunk_rows
        * elements_per_row
        * compare_attention_backends._CORRECTNESS_TEMP_BYTES_PER_ELEMENT
        <= compare_attention_backends._CORRECTNESS_MAX_TEMP_BYTES
    )


def test_attention_repeated_correctness_requires_exact_repeatability() -> None:
    expected = torch.zeros((2, 3), dtype=torch.float16)
    outputs = iter((expected.clone(), expected.clone(), expected.clone()))
    assert compare_attention_backends._check_attention_result_repeatedly(
        lambda: next(outputs), expected, torch.float16, launches=3
    )

    outputs = iter((expected.clone(), expected.clone().add_(0.001)))
    assert not compare_attention_backends._check_attention_result_repeatedly(
        lambda: next(outputs), expected, torch.float16, launches=2
    )

    shared = expected.clone()
    launch = 0

    def aliased_output():
        nonlocal launch
        launch += 1
        return shared.fill_(0.01 if launch == 1 else 0.0)

    assert not compare_attention_backends._check_attention_result_repeatedly(
        aliased_output, expected, torch.float16, launches=2
    )


def test_attention_capture_autotune_metrics_records_search_counts():
    metrics = autotune_metrics.AutotuneMetrics(
        input_shapes="[(2, 32, 65536, 64)]",
        dtypes="['torch.float16']",
        hardware="NVIDIA B200",
        random_seed=123,
        search_algorithm="TestSearch",
        num_configs_tested=17,
        num_compile_failures=1,
        num_worker_failures=2,
        num_isolated_rebenchmark_timeouts=4,
        num_accuracy_failures=3,
        num_successful_candidate_measurements=9,
        num_unique_sources=11,
        num_source_deduplications=6,
        selected_config={"block_sizes": [1, 128, 128]},
        selected_source_hash="a" * 64,
        selected_source_was_measured=True,
    )
    with compare_attention_backends._capture_helion_autotune_metrics() as captured:
        autotune_metrics._run_post_autotune_hooks(metrics)

    assert captured == [
        {
            "input_shapes": "[(2, 32, 65536, 64)]",
            "dtypes": "['torch.float16']",
            "hardware": "NVIDIA B200",
            "random_seed": 123,
            "search_algorithm": "TestSearch",
            "num_configs_tested": 17,
            "num_compile_failures": 1,
            "num_worker_failures": 2,
            "num_isolated_rebenchmark_timeouts": 4,
            "num_accuracy_failures": 3,
            "num_successful_candidate_measurements": 9,
            "num_unique_sources": 11,
            "num_source_deduplications": 6,
            "num_generations": 0,
            "autotune_time": 0.0,
            "best_perf_ms": 0.0,
            "selected_config": {"block_sizes": [1, 128, 128]},
            "selected_source_hash": "a" * 64,
            "selected_source_was_measured": True,
        }
    ]


@pytest.mark.parametrize("print_output_code", (False, True))
def test_attention_selected_source_code_preserves_repro_setting(print_output_code):
    config = {"block_sizes": [1, 128, 128]}
    calls = []

    def to_triton_code(active_config, *, emit_repro_caller):
        calls.append((active_config, emit_repro_caller))
        return "generated source"

    bound = SimpleNamespace(
        settings=SimpleNamespace(print_output_code=print_output_code),
        to_triton_code=to_triton_code,
    )

    assert (
        compare_attention_backends._helion_selected_source_code(bound, config)
        == "generated source"
    )
    assert calls == [(config, print_output_code)]


def test_attention_helion_cute_timer_selects_bench_fn():
    from helion.autotuner.benchmarking import do_bench_generic

    wall_args = SimpleNamespace(helion_cute_benchmark_timer="wall")
    assert (
        compare_attention_backends._helion_do_bench_fn(wall_args, "cute")
        is do_bench_generic
    )

    event_args = SimpleNamespace(helion_cute_benchmark_timer="event")
    assert compare_attention_backends._helion_do_bench_fn(event_args, "cute") is None
    # Non-cute backends always use the default CUDA-event do_bench.
    assert compare_attention_backends._helion_do_bench_fn(wall_args, "triton") is None


@pytest.mark.parametrize(
    "two_cta_marker",
    (
        "cute_tcgen05_flash.CtaGroup.TWO",
        "is_two_cta=True",
        "'use_2cta_instrs': True",
    ),
)
def test_attention_codegen_markers_accept_generated_tcgen05_alias(
    two_cta_marker: str,
):
    code = f"""
from cutlass.cute.nvgpu import tcgen05 as cute_tcgen05_flash
cute_tcgen05_flash.commit(ptr, mask, {two_cta_marker})
PipelineTmaUmma.create()
"""

    assert compare_attention_backends._helion_codegen_markers(code) == {
        "uses_tcgen05": True,
        "uses_tcgen05_two_cta": True,
        "uses_tma_umma_pipeline": True,
        "uses_relu_epilogue": False,
    }


def test_attention_codegen_markers_detect_fused_relu_epilogue() -> None:
    code = "_helion_flash_rt.relu_fragment_inplace(flash_reg)"

    assert compare_attention_backends._helion_codegen_markers(code)[
        "uses_relu_epilogue"
    ]


def test_attention_output_epilogue_semantics() -> None:
    output = torch.tensor([-2.0, 3.0])

    assert compare_attention_backends._apply_output_epilogue(output, "none") is output
    torch.testing.assert_close(
        compare_attention_backends._apply_output_epilogue(output, "relu"),
        torch.tensor([0.0, 3.0]),
    )


def test_attention_relu_example_baselines_match_relu_sdpa() -> None:
    attention_example = importlib.import_module("examples.attention")
    q = torch.randn(1, 1, 4, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    torch.testing.assert_close(
        attention_example._attention_relu_output_baseline(q, k, v),
        torch.relu(attention_example._attention_output_baseline(q, k, v)),
    )
    torch.testing.assert_close(
        attention_example._causal_attention_relu_output_baseline(q, k, v),
        torch.relu(attention_example._causal_attention_output_baseline(q, k, v)),
    )


def _attention_test_stats() -> dict[str, object]:
    return {
        "best_ms": 1.0,
        "median_ms": 1.0,
        "mean_ms": 1.0,
        "std_ms": 0.0,
        "runs_ms": [1.0],
    }


def _patch_attention_result_metadata(monkeypatch) -> None:
    monkeypatch.setattr(compare_attention_backends, "_gpu_name", lambda: "B200")
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_implementation_version",
        lambda impl, **kwargs: {"version": impl, "version_label": impl},
    )


def test_attention_sdpa_times_relu_epilogue(monkeypatch) -> None:
    args = _attention_subprocess_args(
        impl="sdpa", dtype="bfloat16", biased=0, epilogue="relu"
    )
    q = torch.tensor([-2.0, 3.0])
    monkeypatch.setattr(
        compare_attention_backends, "_make_inputs", lambda args, dtype: (q, q, q)
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_sdpa_reference",
        lambda q, k, v, *, causal, bias=None: q,
    )
    timed_outputs = []

    def bench(fn, **kwargs):
        timed_outputs.append(fn())
        return _attention_test_stats()

    monkeypatch.setattr(compare_attention_backends, "_bench_steady", bench)
    _patch_attention_result_metadata(monkeypatch)

    result = compare_attention_backends._benchmark_sdpa(args)

    torch.testing.assert_close(timed_outputs[0], torch.tensor([0.0, 3.0]))
    assert result["shape"]["epilogue"] == "relu"
    assert result["epilogue_flops_included"] is False


def test_attention_fa4_times_relu_epilogue(monkeypatch) -> None:
    args = _attention_subprocess_args(
        impl="fa4", dtype="bfloat16", biased=0, epilogue="relu"
    )
    q = torch.zeros((1, 2, 3, 1), dtype=torch.bfloat16)
    native_output = torch.tensor(
        [[[[-2.0], [3.0]], [[-4.0], [5.0]], [[-6.0], [7.0]]]],
        dtype=torch.bfloat16,
    )
    expected = torch.relu(native_output).transpose(1, 2)
    monkeypatch.setattr(
        compare_attention_backends, "_make_inputs", lambda args, dtype: (q, q, q)
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_attention_output_reference",
        lambda args, q, k, v: expected,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_import_fa4",
        lambda: SimpleNamespace(
            flash_attn_func=lambda q, k, v, **kwargs: (native_output.clone(), None)
        ),
    )
    timed_outputs = []

    def bench(fn, **kwargs):
        timed_outputs.append(fn())
        return _attention_test_stats()

    monkeypatch.setattr(compare_attention_backends, "_bench_steady", bench)
    _patch_attention_result_metadata(monkeypatch)

    result = compare_attention_backends._benchmark_fa4(args)

    torch.testing.assert_close(timed_outputs[0], torch.relu(native_output))
    assert result["accuracy"] == "PASS"


def test_attention_flexattention_compiles_relu_with_attention(monkeypatch) -> None:
    flex_module = importlib.import_module("torch.nn.attention.flex_attention")
    args = _attention_subprocess_args(
        impl="flexattention", dtype="bfloat16", biased=0, epilogue="relu"
    )
    q = torch.tensor([-2.0, 3.0])
    monkeypatch.setattr(
        compare_attention_backends, "_make_inputs", lambda args, dtype: (q, q, q)
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_attention_output_reference",
        lambda args, q, k, v, *, bias=None: torch.relu(q),
    )
    monkeypatch.setattr(
        flex_module,
        "flex_attention",
        lambda q, k, v, **kwargs: q,
    )
    compile_targets = []

    def compile_fn(fn, *, fullgraph):
        assert fullgraph is True
        compile_targets.append(fn)
        return fn

    monkeypatch.setattr(compare_attention_backends.torch, "compile", compile_fn)
    timed_outputs = []

    def bench(fn, **kwargs):
        timed_outputs.append(fn())
        return _attention_test_stats()

    monkeypatch.setattr(compare_attention_backends, "_bench_steady", bench)
    _patch_attention_result_metadata(monkeypatch)

    result = compare_attention_backends._benchmark_flexattention(args)

    assert len(compile_targets) == 1
    torch.testing.assert_close(timed_outputs[0], torch.tensor([0.0, 3.0]))
    assert result["accuracy"] == "PASS"


def test_attention_markdown_and_wide_csv_include_timer(tmp_path):
    payload = {
        "shape": {
            "z": 1,
            "h": 2,
            "seq_len": 128,
            "head_dim": 64,
            "dtype": "float16",
            "causal": 0,
            "biased": 1,
        },
        "results": [
            {
                "impl": "helion-cute",
                "version": "Helion test-version",
                "version_label": "test-version",
                "flop_model": "softmax_attention_forward",
                "gpu": "NVIDIA B200",
                "physical_gpu": "7",
                "power_cap_w": 750,
                "accuracy": "PASS",
                "benchmark_timer": "event",
                "notes": ["test note"],
                "helion_overrides": {"autotuned": True},
                "best_ms": 0.1,
                "median_ms": 0.1,
                "mom_median_ms": 0.1,
                "best_tflops": 1.0,
                "median_tflops": 1.0,
                "mom_median_tflops": 1.0,
            }
        ],
    }

    markdown_rows = compare_attention_backends._markdown_rows(payload)
    wide_rows = compare_attention_backends._wide_rows([payload])

    assert markdown_rows[0]["timer"] == "event"
    assert markdown_rows[0]["version"] == "Helion test-version"
    assert wide_rows[0]["helion_cute_timer"] == "event"
    assert wide_rows[0]["helion_cute_version"] == "Helion test-version"
    assert wide_rows[0]["helion_cute_flop_model"] == "softmax_attention_forward"
    assert json.loads(wide_rows[0]["helion_cute_notes"]) == ["test note"]
    assert json.loads(wide_rows[0]["helion_cute_helion_overrides"]) == {
        "autotuned": True
    }
    assert wide_rows[0]["gpu"] == "NVIDIA B200"
    assert wide_rows[0]["physical_gpu"] == "7"
    assert wide_rows[0]["power_cap_w"] == 750

    csv_path = tmp_path / "attention.csv"
    compare_attention_backends._write_wide_csv(csv_path, wide_rows)
    assert b"\r\n" not in csv_path.read_bytes()


def test_attention_versioned_plot_label():
    payloads = [
        {
            "results": [
                {
                    "impl": "sdpa",
                    "version": "PyTorch test; cuDNN 9.20.0",
                    "version_label": "cuDNN 9.20.0",
                }
            ]
        }
    ]

    assert compare_attention_backends._versioned_impl_label("sdpa", payloads) == (
        "torch SDPA\ncuDNN 9.20.0"
    )


def test_attention_versioned_plot_label_supports_generic_override():
    payloads = [
        {
            "results": [
                {
                    "impl": "kernelagent-1x",
                    "accuracy": "PASS",
                    "version_label": "KernelAgent test version",
                }
            ]
        }
    ]

    assert (
        compare_attention_backends._versioned_impl_label(
            "kernelagent-1x",
            payloads,
            {"kernelagent-1x": "Archived campaign label ($123 tokens)"},
        )
        == "Archived campaign label ($123 tokens)\nKernelAgent test version"
    )


def test_attention_plot_impl_label_parser_is_generic():
    assert compare_attention_backends._parse_plot_impl_label(
        "sdpa=Reference implementation"
    ) == ("sdpa", "Reference implementation")

    with pytest.raises(
        compare_attention_backends.argparse.ArgumentTypeError,
        match="unknown implementation",
    ):
        compare_attention_backends._parse_plot_impl_label("unknown=label")
    with pytest.raises(
        compare_attention_backends.argparse.ArgumentTypeError,
        match="must not be empty",
    ):
        compare_attention_backends._parse_plot_impl_label("sdpa=")


@pytest.mark.parametrize(
    ("impl", "version_label"),
    [
        (
            "kernelagent-1x",
            "KernelAgent v2+archived / Opus-5.0 / Triton 3.7.0",
        ),
        (
            "kernelagent-closed-1x",
            "KernelAgent v3-archived / GPT-5.6 / CuTe 4.5.1",
        ),
    ],
)
def test_attention_kernelagent_plot_uses_archived_version_label(
    impl, version_label, monkeypatch
):
    payloads = [{"results": [{"impl": impl, "version_label": version_label}]}]
    monkeypatch.setattr(
        compare_attention_backends,
        "_implementation_version",
        lambda impl: pytest.fail(f"unexpected live version lookup for {impl}"),
    )

    assert compare_attention_backends._versioned_impl_label(impl, payloads) == (
        f"{compare_attention_backends._IMPL_LABELS[impl]}\n{version_label}"
    )


def test_attention_backend_plot_labels_are_consistent():
    assert compare_attention_backends._IMPL_LABELS["helion-triton"] == (
        "Helion (backend=Triton)"
    )
    assert compare_attention_backends._IMPL_LABELS["helion-cute"] == (
        "Helion (backend=CuTe)"
    )
    assert compare_attention_backends._IMPL_LABELS["helion-tileir"] == (
        "Helion (backend=TileIR)"
    )
    assert compare_attention_backends._IMPL_LABELS["flexattention"] == (
        "FlexAttention (backend=Triton)"
    )
    assert compare_attention_backends._IMPL_LABELS["flexattention-cute"] == (
        "FlexAttention (backend=CuTe)"
    )
    assert compare_attention_backends._IMPL_LABELS["tlx"] == "TLX attention"
    assert compare_attention_backends._KERNELAGENT_BUDGET_LABELS == {
        "kernelagent-1x": "1x",
        "kernelagent-2x": "2x",
        "kernelagent-10x": "10x",
        "kernelagent-closed-1x": "1x",
        "kernelagent-closed-2x": "2x",
    }
    assert (
        "KernelAgent Public"
        in compare_attention_backends._IMPL_LABELS["kernelagent-1x"]
    )
    assert compare_attention_backends._IMPL_LABELS["kernelagent-closed-1x"] == (
        "KernelAgent Closed (1x Helion tuning time)"
    )
    assert compare_attention_backends._IMPL_LABELS["kernelagent-1x"] == (
        "KernelAgent Public (1x Helion tuning time)"
    )


def test_attention_kernelagent_version_labels_come_from_manifests():
    public = {
        "kernelagent_commit": "abcdef0123456789",
        "kernelagent_display_version": "v2+abcdef01",
        "model": "claude-opus-next",
        "model_display_name": "Opus-5.0",
        "triton_version": "3.7.0+selection",
    }
    closed = {
        "kernelagent_version": "v4-test",
        "kernelagent_display_version": "v4-test",
        "model": "gpt-test",
        "model_display_name": "GPT-5.6",
        "cutlass_dsl_version": "4.5.1",
    }

    assert compare_attention_backends._kernelagent_version_info(
        "kernelagent-1x", public, evaluation_backend_version="3.8.0+evaluation"
    ) == {
        "version": (
            "KernelAgent commit abcdef01; model claude-opus-next; "
            "Triton 3.8.0+evaluation; selected with Triton 3.7.0+selection"
        ),
        "version_label": "KernelAgent v2+abcdef01 / Opus-5.0 / Triton 3.8.0",
    }
    assert compare_attention_backends._kernelagent_version_info(
        "kernelagent-closed-1x", closed, evaluation_backend_version="4.6.1"
    ) == {
        "version": (
            "KernelAgent v4-test; model gpt-test; CuTe 4.6.1; selected with CuTe 4.5.1"
        ),
        "version_label": "KernelAgent v4-test / GPT-5.6 / CuTe 4.6.1",
    }


def test_attention_kernelagent_version_without_manifest_is_generic():
    assert compare_attention_backends._implementation_version("kernelagent-1x") == {
        "version": "KernelAgent metadata is supplied by the run manifest",
        "version_label": "run manifest metadata",
    }


def test_attention_kernelagent_manifest_validates_complete_campaign_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    args = SimpleNamespace(
        impl="kernelagent-1x",
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        seed=123,
    )
    manifest = {
        "budget_label": "1x",
        "shape": compare_attention_backends._shape_dict(args),
        "physical_gpu": 7,
        "power_cap_w": 750,
        "seed": 123,
        "kernelagent_display_version": "v2+abcdef01",
        "model_display_name": "Opus-5.0",
    }

    assert (
        compare_attention_backends._validate_kernelagent_manifest(
            args.impl, manifest, args, tmp_path
        )
        is manifest
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        (
            "shape",
            {
                "z": 2,
                "h": 32,
                "seq_len": 32768,
                "head_dim": 64,
                "dtype": "float16",
                "causal": False,
                "biased": 0,
            },
        ),
        ("physical_gpu", 6),
        ("power_cap_w", 700),
        ("seed", 456),
    ),
)
def test_attention_kernelagent_manifest_rejects_campaign_mismatch(
    tmp_path, monkeypatch, field, bad_value
):
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    args = SimpleNamespace(
        impl="kernelagent-1x",
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        seed=123,
    )
    manifest = {
        "budget_label": "1x",
        "shape": compare_attention_backends._shape_dict(args),
        "physical_gpu": 7,
        "power_cap_w": 750,
        "seed": 123,
        "kernelagent_display_version": "v2+abcdef01",
        "model_display_name": "Opus-5.0",
    }
    manifest[field] = bad_value

    with pytest.raises(SystemExit, match="manifest mismatch"):
        compare_attention_backends._validate_kernelagent_manifest(
            args.impl, manifest, args, tmp_path
        )


def test_attention_plot_version_labels_are_concise(tmp_path, monkeypatch):
    monkeypatch.setenv("HELION_BENCHMARK_HELION_VERSION", "1.4.0.dev38+g016ad645")
    monkeypatch.setattr(
        compare_attention_backends,
        "_package_version",
        lambda package: {
            "triton": "3.7.0+git88b227e2",
            "nvidia-cutlass-dsl": "4.5.1",
        }[package],
    )
    monkeypatch.setattr(
        compare_attention_backends.torch,
        "__version__",
        "2.13.0.dev20260506+cu130",
    )
    monkeypatch.setattr(
        compare_attention_backends, "_resolve_fa4_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_git_describe",
        lambda root: "fa4-v4.0.0.beta23",
    )

    assert (
        compare_attention_backends._implementation_version("helion-triton")[
            "version_label"
        ]
        == "Helion 1.4.0.dev38+g016ad645 / Triton 3.7.0"
    )
    assert (
        compare_attention_backends._implementation_version("helion-cute")[
            "version_label"
        ]
        == "Helion 1.4.0.dev38+g016ad645 / CuTe 4.5.1"
    )
    assert (
        compare_attention_backends._implementation_version(
            "gluon", resolve_external_sources=False
        )["version_label"]
        == "Triton 3.7.0"
    )
    assert (
        compare_attention_backends._implementation_version("flexattention")[
            "version_label"
        ]
        == "PyTorch 2.13.0.dev20260506; Triton 3.7.0"
    )
    assert compare_attention_backends._implementation_version("flexattention-cute")[
        "version_label"
    ] == ("PyTorch 2.13.0.dev20260506; FA4 fa4-v4.0.0.beta23; CuTe 4.5.1")
    assert (
        compare_attention_backends._implementation_version("fa4")["version_label"]
        == "fa4-v4.0.0.beta23; CuTe 4.5.1"
    )


def test_attention_closed_kernelagent_failure_does_not_require_source(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "dense_32768_1x"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernelagent_family": "closed_binary",
                "kernelagent_version": "v3-20260730",
                "kernelagent_display_version": "v3-20260730",
                "model_display_name": "GPT-5.6",
                "shape": {
                    "z": 2,
                    "h": 32,
                    "seq_len": 32768,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": 0,
                    "biased": 0,
                },
                "seq_len": 32768,
                "causal": False,
                "physical_gpu": 7,
                "power_cap_w": 750,
                "seed": 123,
                "budget_label": "1x",
                "budget_seconds": 708.6,
                "elapsed_seconds": 708.6,
                "model": "gpt-5.6-sol",
                "cutlass_dsl_version": "4.5.1",
                "status": "FAIL",
                "failure_reason": "No verified candidate.",
            }
        )
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_implementation_version",
        lambda impl: {"version": impl, "version_label": impl},
    )
    monkeypatch.setattr(
        compare_attention_backends, "_package_version", lambda package: "evaluation"
    )
    monkeypatch.setattr(compare_attention_backends, "_gpu_name", lambda: "B200")
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    args = SimpleNamespace(
        impl="kernelagent-closed-1x",
        kernelagent_closed_results_root=str(tmp_path),
        kernelagent_results_root=None,
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        seed=123,
    )

    result = compare_attention_backends._benchmark_kernelagent(args)

    assert result["accuracy"] == "FAIL"
    assert result["error"] == "No verified candidate."
    assert result["config"]["selection_cute_version"] == "4.5.1"
    assert result["config"]["evaluation_cute_version"] is None
    assert "best_ms" not in result


def test_attention_successful_kernelagent_requires_declared_source_hash(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "dense_32768_1x"
    run_dir.mkdir()
    (run_dir / "selected_kernel.py.txt").write_text(
        "def kernel_function(q, k, v):\n    return q\n"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernelagent_commit": "abcdef0123456789",
                "kernelagent_display_version": "v2+abcdef01",
                "model": "claude-opus-next",
                "model_display_name": "Opus-5.0",
                "triton_version": "3.7.0+selection",
                "shape": {
                    "z": 2,
                    "h": 32,
                    "seq_len": 32768,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": 0,
                    "biased": 0,
                },
                "physical_gpu": 7,
                "power_cap_w": 750,
                "seed": 123,
                "budget_label": "1x",
                "budget_seconds": 1.0,
                "elapsed_seconds": 1.0,
            }
        )
    )
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    args = SimpleNamespace(
        impl="kernelagent-1x",
        kernelagent_closed_results_root=None,
        kernelagent_results_root=str(tmp_path),
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        seed=123,
    )

    with pytest.raises(SystemExit, match="no declared source hash"):
        compare_attention_backends._benchmark_kernelagent(args)


def test_attention_kernelagent_rejects_manifest_source_hash_mismatch(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "dense_32768_1x"
    run_dir.mkdir()
    (run_dir / "selected_kernel.py.txt").write_text(
        "def kernel_function(q, k, v):\n    return q\n"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernelagent_family": "closed_binary",
                "kernelagent_version": "v4-test",
                "kernelagent_display_version": "v4-test",
                "model_display_name": "GPT-test",
                "shape": {
                    "z": 2,
                    "h": 32,
                    "seq_len": 32768,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": 0,
                    "biased": 0,
                },
                "seq_len": 32768,
                "causal": False,
                "physical_gpu": 7,
                "power_cap_w": 750,
                "seed": 123,
                "budget_label": "1x",
                "budget_seconds": 1.0,
                "elapsed_seconds": 1.0,
                "model": "gpt-test",
                "cutlass_dsl_version": "4.6.1",
                "selection": {"source_sha256": "0" * 64},
            }
        )
    )
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    args = SimpleNamespace(
        impl="kernelagent-closed-1x",
        kernelagent_closed_results_root=str(tmp_path),
        kernelagent_results_root=None,
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        seed=123,
    )

    with pytest.raises(SystemExit, match="source hash mismatch"):
        compare_attention_backends._benchmark_kernelagent(args)


def test_attention_public_kernelagent_rejects_invalid_output_contract(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "dense_32768_1x"
    run_dir.mkdir()
    source = "def kernel_function(q, k, v):\n    return None\n"
    (run_dir / "selected_kernel.py.txt").write_text(source)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernelagent_commit": "abcdef0123456789",
                "kernelagent_display_version": "v2+abcdef01",
                "model_display_name": "Opus-test",
                "shape": {
                    "z": 2,
                    "h": 32,
                    "seq_len": 32768,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": 0,
                    "biased": 0,
                },
                "seq_len": 32768,
                "causal": False,
                "physical_gpu": 7,
                "power_cap_w": 750,
                "seed": 123,
                "budget_label": "1x",
                "budget_seconds": 1.0,
                "elapsed_seconds": 1.0,
                "model": "claude-opus-next",
                "triton_version": "3.7.0+selection",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "selection": {},
            }
        )
    )
    monkeypatch.setattr(
        compare_attention_backends, "_make_inputs", lambda args, dtype: (None,) * 3
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_sdpa_reference",
        lambda q, k, v, *, causal: "expected",
    )
    monkeypatch.setattr(
        compare_attention_backends, "_package_version", lambda package: "evaluation"
    )
    monkeypatch.setattr(compare_attention_backends, "_gpu_name", lambda: "B200")
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    args = SimpleNamespace(
        impl="kernelagent-1x",
        kernelagent_closed_results_root=None,
        kernelagent_results_root=str(tmp_path),
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        skip_correctness=False,
        num_runs=1,
        warmup_ms=0,
        rep_ms=0,
        seed=123,
    )

    result = compare_attention_backends._benchmark_kernelagent(args)

    assert result["accuracy"] == "FAIL"
    assert result["error"] == (
        "Selected KernelAgent source failed final-harness correctness."
    )
    assert "best_ms" not in result
    assert "abcdef01" in result["version"]


@pytest.mark.parametrize("stress_passes", (True, False))
def test_attention_kernelagent_execution_scrubs_cli_argv(
    tmp_path, monkeypatch, stress_passes
):
    run_dir = tmp_path / "causal_65536_1x"
    run_dir.mkdir()
    source = "import sys\ndef kernel_function(q, k, v):\n    return tuple(sys.argv)\n"
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernelagent_family": "closed_binary",
                "kernelagent_version": "v3-20260730",
                "kernelagent_display_version": "v3-20260730",
                "model_display_name": "GPT-5.6",
                "shape": {
                    "z": 2,
                    "h": 32,
                    "seq_len": 65536,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": 1,
                    "biased": 0,
                },
                "seq_len": 65536,
                "causal": True,
                "physical_gpu": 6,
                "power_cap_w": 750,
                "seed": 123,
                "budget_label": "1x",
                "budget_seconds": 3732.2,
                "elapsed_seconds": 3732.2,
                "model": "gpt-5.6-sol",
                "cutlass_dsl_version": "4.5.1",
                "status": "PASS",
                "selection": {
                    "candidate_id": 1,
                    "median_ms": 1.0,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                },
            }
        )
    )
    (run_dir / "selected_kernel.py.txt").write_text(source)
    monkeypatch.setattr(
        compare_attention_backends,
        "_implementation_version",
        lambda impl: {"version": impl, "version_label": impl},
    )
    monkeypatch.setattr(
        compare_attention_backends, "_package_version", lambda package: "evaluation"
    )
    monkeypatch.setattr(compare_attention_backends, "_gpu_name", lambda: "B200")
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "6"
    )
    monkeypatch.setattr(
        compare_attention_backends, "_make_inputs", lambda args, dtype: (None,) * 3
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_sdpa_reference",
        lambda q, k, v, *, causal: "expected",
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_check_kernelagent_output",
        lambda actual, expected: True,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_check_kernelagent_repeat",
        lambda first, repeated: True,
    )
    stress_checks = []

    def check_stress(run, args, dtype):
        assert compare_attention_backends.sys.argv == ["attention-benchmark"]
        stress_checks.append((run, args, dtype))
        return stress_passes

    monkeypatch.setattr(
        compare_attention_backends, "_check_kernelagent_stress_case", check_stress
    )

    benchmark_calls = []

    def bench(fn, **kwargs):
        benchmark_calls.append((fn, kwargs))
        assert fn() == ("attention-benchmark",)
        return {
            "best_ms": 1.0,
            "median_ms": 1.0,
            "mean_ms": 1.0,
            "std_ms": 0.0,
            "runs_ms": [1.0],
        }

    monkeypatch.setattr(compare_attention_backends, "_bench_steady", bench)
    monkeypatch.setattr(
        compare_attention_backends.sys,
        "argv",
        ["attention-benchmark", "--h", "32"],
    )
    args = SimpleNamespace(
        impl="kernelagent-closed-1x",
        kernelagent_closed_results_root=str(tmp_path),
        kernelagent_results_root=None,
        z=2,
        h=32,
        seq_len=65536,
        head_dim=64,
        dtype="float16",
        causal=1,
        biased=0,
        power_cap_w=750,
        skip_correctness=False,
        num_runs=1,
        warmup_ms=0,
        rep_ms=0,
        seed=123,
    )

    result = compare_attention_backends._benchmark_kernelagent(args)

    assert result["accuracy"] == ("PASS" if stress_passes else "FAIL")
    assert result["config"]["selection_cute_version"] == "4.5.1"
    assert result["config"]["evaluation_cute_version"] == "evaluation"
    assert result["config"]["standard_correctness_executed"] is True
    assert result["config"]["repeat_determinism_executed"] is True
    assert result["config"]["stress_correctness_executed"] is True
    assert len(stress_checks) == 1
    assert len(benchmark_calls) == int(stress_passes)
    assert ("best_ms" in result) is stress_passes
    assert compare_attention_backends.sys.argv == [
        "attention-benchmark",
        "--h",
        "32",
    ]


def test_attention_public_kernelagent_runs_repeat_and_stress_checks(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "dense_32768_1x"
    run_dir.mkdir()
    source = "def kernel_function(q, k, v):\n    return 'output'\n"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    (run_dir / "selected_kernel.py.txt").write_text(source)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kernelagent_commit": "abcdef0123456789",
                "kernelagent_display_version": "v2+abcdef01",
                "model": "claude-opus-next",
                "model_display_name": "Opus-5.0",
                "triton_version": "3.7.0+selection",
                "shape": {
                    "z": 2,
                    "h": 32,
                    "seq_len": 32768,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": 0,
                    "biased": 0,
                },
                "physical_gpu": 7,
                "power_cap_w": 750,
                "seed": 123,
                "budget_label": "1x",
                "budget_seconds": 1.0,
                "elapsed_seconds": 1.0,
                "source_sha256": source_hash,
                "selection": {},
            }
        )
    )
    monkeypatch.setattr(
        compare_attention_backends, "_physical_gpu_selection", lambda: "7"
    )
    monkeypatch.setattr(
        compare_attention_backends, "_make_inputs", lambda args, dtype: (None,) * 3
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_sdpa_reference",
        lambda q, k, v, *, causal: "expected",
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_check_kernelagent_output",
        lambda actual, expected: True,
    )
    repeat_calls = []
    monkeypatch.setattr(
        compare_attention_backends,
        "_check_kernelagent_repeat",
        lambda first, repeated: repeat_calls.append((first, repeated)) or True,
    )
    stress_calls = []
    monkeypatch.setattr(
        compare_attention_backends,
        "_check_kernelagent_stress_case",
        lambda run, args, dtype: stress_calls.append((run, args, dtype)) or True,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_bench_steady",
        lambda fn, **kwargs: {
            "best_ms": 1.0,
            "median_ms": 1.0,
            "mean_ms": 1.0,
            "std_ms": 0.0,
            "runs_ms": [1.0],
        },
    )
    monkeypatch.setattr(
        compare_attention_backends, "_package_version", lambda package: "evaluation"
    )
    monkeypatch.setattr(compare_attention_backends, "_gpu_name", lambda: "B200")
    args = SimpleNamespace(
        impl="kernelagent-1x",
        kernelagent_closed_results_root=None,
        kernelagent_results_root=str(tmp_path),
        z=2,
        h=32,
        seq_len=32768,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=0,
        power_cap_w=750,
        seed=123,
        skip_correctness=False,
        num_runs=1,
        warmup_ms=0,
        rep_ms=0,
    )

    result = compare_attention_backends._benchmark_kernelagent(args)

    assert result["accuracy"] == "PASS"
    assert len(repeat_calls) == 1
    assert len(stress_calls) == 1
    assert result["config"]["repeat_determinism_executed"] is True
    assert result["config"]["stress_correctness_executed"] is True
    assert result["config"]["selection_triton_version"] == "3.7.0+selection"


def test_attention_kernelagent_evaluation_notes_match_executed_checks():
    note = compare_attention_backends._kernelagent_evaluation_note

    assert "correctness checks were skipped" in note(
        "CuTe",
        "4.5.1",
        "4.6.1",
        standard_executed=False,
        repeat_executed=False,
        stress_executed=False,
        passed=False,
        measured=True,
    )
    assert "repeat and stress were not run" in note(
        "CuTe",
        "4.5.1",
        "4.6.1",
        standard_executed=True,
        repeat_executed=False,
        stress_executed=False,
        passed=False,
        measured=False,
    )
    assert "exact repeatability failed" in note(
        "CuTe",
        "4.5.1",
        "4.6.1",
        standard_executed=True,
        repeat_executed=True,
        stress_executed=False,
        passed=False,
        measured=False,
    )
    assert "stress failed" in note(
        "CuTe",
        "4.5.1",
        "4.6.1",
        standard_executed=True,
        repeat_executed=True,
        stress_executed=True,
        passed=False,
        measured=False,
    )
    assert "exact repeatability" in note(
        "CuTe",
        "4.5.1",
        "4.6.1",
        standard_executed=True,
        repeat_executed=True,
        stress_executed=True,
        passed=True,
        measured=True,
    )


def test_attention_kernelagent_output_contract_rejects_non_cuda_outputs():
    expected = torch.empty((1, 1, 2, 2), dtype=torch.float16)

    assert not compare_attention_backends._check_kernelagent_output(None, expected)
    assert not compare_attention_backends._check_kernelagent_output(expected, expected)


def test_attention_kernelagent_skips_relu_workload_before_loading_results(
    monkeypatch,
):
    monkeypatch.setattr(
        compare_attention_backends,
        "_kernelagent_run_dir",
        lambda args: pytest.fail("unsupported ReLU workload resolved a run directory"),
    )
    args = _attention_subprocess_args(
        impl="kernelagent-1x", dtype="bfloat16", biased=0, epilogue="relu"
    )

    result = compare_attention_backends._benchmark_kernelagent(args)

    assert result["accuracy"] == "SKIP"
    assert "identity-epilogue" in result["skipped_reason"]


def test_attention_tileir_version_includes_each_toolchain_component(monkeypatch):
    monkeypatch.setattr(
        compare_attention_backends,
        "_git_describe",
        lambda root: "v1.4.0-38-g016ad645",
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_package_version",
        lambda package: {"triton": "3.6.0", "nvtriton": "3.6.0"}[package],
    )
    monkeypatch.setattr(
        compare_attention_backends, "_tileir_toolchain_version", lambda: "13.3"
    )

    version = compare_attention_backends._implementation_version("helion-tileir")

    assert version == {
        "version": ("Helion 1.4.0.dev38+g016ad645; nvtriton 3.6.0; TileIR 13.3"),
        "version_label": (
            "Helion 1.4.0.dev38+g016ad645 / nvtriton 3.6.0 / TileIR 13.3"
        ),
    }


def test_attention_helion_version_can_be_supplied_for_isolated_runtime(monkeypatch):
    monkeypatch.setenv("HELION_BENCHMARK_HELION_VERSION", "1.4.0.dev38+g016ad645")
    monkeypatch.setattr(
        compare_attention_backends,
        "_package_version",
        lambda package: {"triton": "3.6.0", "nvtriton": "3.6.0"}[package],
    )
    monkeypatch.setattr(
        compare_attention_backends, "_tileir_toolchain_version", lambda: "13.3"
    )

    version = compare_attention_backends._implementation_version("helion-tileir")

    assert version["version"].startswith("Helion 1.4.0.dev38+g016ad645;")


@pytest.mark.parametrize(
    ("git_describe", "expected"),
    (
        ("v1.4.0-38-g016ad645", "1.4.0.dev38+g016ad645"),
        ("v1.4.0", "1.4.0"),
        ("016ad645", "016ad645"),
        ("v1.4.0-38-g016ad645-dirty", "1.4.0.dev38+g016ad645.dirty"),
        ("v1.4.0-dirty", "1.4.0+dirty"),
    ),
)
def test_attention_helion_git_version_is_explicitly_development(
    git_describe: str, expected: str
):
    assert (
        compare_attention_backends._format_git_development_version(git_describe)
        == expected
    )


def test_attention_git_version_marks_dirty_worktrees(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="v1.4.0-1-gabcdef01-dirty\n")

    monkeypatch.setattr(compare_attention_backends.subprocess, "run", run)

    assert compare_attention_backends._git_describe(tmp_path) == (
        "v1.4.0-1-gabcdef01-dirty"
    )
    assert calls[0][0][-1] == "--dirty"


def test_attention_power_cap_is_verified(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="750.00\n")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(compare_attention_backends.subprocess, "run", run)

    assert compare_attention_backends._verify_power_cap_w(750) == 750
    assert calls[0][0][1:3] == ["-i", "6"]
    assert calls[0][1] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 30,
    }


def test_attention_power_cap_mismatch_is_rejected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(
        compare_attention_backends.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="850.00\n"),
    )

    with pytest.raises(SystemExit, match="requested benchmark label is 750 W"):
        compare_attention_backends._verify_power_cap_w(750)


def test_attention_report_rejects_mixed_power_caps():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    payloads = [
        {
            "shape": shape,
            "results": [
                {
                    "impl": "sdpa",
                    "shape": shape,
                    "accuracy": "FAIL",
                    "gpu": "NVIDIA B200",
                    "physical_gpu": str(6 + index),
                    "power_cap_w": power_cap_w,
                }
            ],
        }
        for index, power_cap_w in enumerate((750, 850))
    ]

    with pytest.raises(ValueError, match="mixes GPU power limits"):
        compare_attention_backends._benchmark_setup_label(payloads)


def test_attention_report_rejects_mismatched_result_shape():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result_shape = {**shape, "seq_len": 131072}
    payloads = [
        {
            "shape": shape,
            "results": [
                {
                    "impl": "sdpa",
                    "shape": result_shape,
                    "accuracy": "PASS",
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="does not match payload shape"):
        compare_attention_backends._validate_report_payloads(payloads)


def test_attention_report_rejects_mixed_input_seeds():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    results = []
    for impl, input_seed in (("helion-cute", 123), ("sdpa", 456)):
        results.append(
            {
                "impl": impl,
                "shape": shape,
                "accuracy": "PASS",
                "version": "test-version",
                "benchmark_timer": "event",
                "flop_model": "softmax_attention_forward",
                "gpu": "NVIDIA B200",
                "physical_gpu": "6",
                "power_cap_w": 750,
                "input_seed": input_seed,
            }
        )

    with pytest.raises(ValueError, match="mixes input seeds"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": results}]
        )


def test_attention_report_rejects_mixed_recorded_and_missing_input_seeds():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = {
        "shape": shape,
        "accuracy": "PASS",
        "version": "test-version",
        "benchmark_timer": "event",
        "flop_model": "softmax_attention_forward",
        "gpu": "NVIDIA B200",
        "physical_gpu": "6",
        "power_cap_w": 750,
    }
    results = [
        {**result, "impl": "helion-cute", "input_seed": 123},
        {**result, "impl": "sdpa"},
    ]

    with pytest.raises(ValueError, match="recorded and missing input seeds"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": results}]
        )


def test_attention_report_accepts_legacy_missing_input_seeds():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = {
        "impl": "sdpa",
        "shape": shape,
        "accuracy": "PASS",
        "version": "test-version",
        "benchmark_timer": "event",
        "flop_model": "softmax_attention_forward",
        "gpu": "NVIDIA B200",
        "physical_gpu": "6",
        "power_cap_w": 750,
    }

    compare_attention_backends._validate_report_payloads(
        [{"shape": shape, "results": [result]}]
    )


@pytest.mark.parametrize("field", ("version", "benchmark_timer", "flop_model"))
def test_attention_report_rejects_mixed_successful_metadata(field):
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    payloads = []
    for value in ("first", "second"):
        result = {
            "impl": "helion-cute",
            "shape": shape,
            "accuracy": "PASS",
            "version": "same-version",
            "benchmark_timer": "event",
            "flop_model": "softmax_attention_forward",
            "gpu": "NVIDIA B200",
            "physical_gpu": "6",
            "power_cap_w": 750,
        }
        result[field] = value
        payloads.append({"shape": shape, "results": [result]})

    with pytest.raises(ValueError, match=f"mixes {field} metadata"):
        compare_attention_backends._validate_report_payloads(payloads)


@pytest.mark.parametrize("field", ("version", "benchmark_timer", "flop_model"))
def test_attention_report_requires_successful_metadata(field):
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = {
        "impl": "helion-cute",
        "shape": shape,
        "accuracy": "PASS",
        "version": "1.4.0.dev1+gabcdef01",
        "benchmark_timer": "event",
        "flop_model": "softmax_attention_forward",
        "gpu": "NVIDIA B200",
        "physical_gpu": "6",
        "power_cap_w": 750,
    }
    del result[field]

    with pytest.raises(ValueError, match=f"has no {field} metadata"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": [result]}]
        )


@pytest.mark.parametrize("field", ("gpu", "physical_gpu", "power_cap_w"))
def test_attention_report_requires_successful_environment_metadata(field):
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = {
        "impl": "helion-cute",
        "shape": shape,
        "accuracy": "PASS",
        "version": "1.4.0.dev1+gabcdef01",
        "benchmark_timer": "event",
        "flop_model": "softmax_attention_forward",
        "gpu": "NVIDIA B200",
        "physical_gpu": "6",
        "power_cap_w": 750,
    }
    del result[field]

    with pytest.raises(ValueError, match=f"has no {field} metadata"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": [result]}]
        )


def _strict_report_result(shape, source_hash):
    fixture_provenance = _full_autotune_provenance()
    terminal_policy = copy.deepcopy(
        fixture_provenance["flash_terminal_coordinate_refinement_policy"]
    )
    terminal_surface = copy.deepcopy(
        fixture_provenance["flash_terminal_coordinate_surface_catalog"]
    )
    return {
        "impl": "helion-cute",
        "shape": shape,
        "accuracy": "PASS",
        "version": "same-dirty-version",
        "benchmark_timer": "event",
        "flop_model": "softmax_attention_forward",
        "gpu": "NVIDIA B200",
        "physical_gpu": "6",
        "power_cap_w": 750,
        "input_seed": 0,
        "helion_overrides": {
            "autotune_provenance": {
                "require_full_autotune": True,
                "helion_source_tree_sha256": source_hash,
                "helion_checkout_git_commit": "a" * 40,
                "post_measurement_source_verified": True,
                "post_measurement_source": {
                    "helion_source_tree_sha256": source_hash,
                    "helion_checkout_git_commit": "a" * 40,
                    "helion_source_tree_dirty": False,
                },
                "flash_normalization_context": {
                    "schema_version": 1,
                    "dtype": f"torch.{shape['dtype']}",
                    "head_dim": shape["head_dim"],
                    "num_kv": (shape["seq_len"] + 127) // 128,
                    "is_causal": bool(shape["causal"]),
                    "biased": bool(shape.get("biased", 0)),
                },
                "flash_terminal_coordinate_refinement_policy": terminal_policy,
                "flash_terminal_coordinate_refinement_policy_sha256": hashlib.sha256(
                    json.dumps(
                        terminal_policy, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "flash_terminal_coordinate_surface_catalog": terminal_surface,
                "flash_terminal_coordinate_surface_catalog_sha256": hashlib.sha256(
                    json.dumps(
                        terminal_surface, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "trials": [
                    {
                        "search_phase_metrics": {
                            "phase": "cute_flash_structural_qualification_v22",
                            "cute_flash_lane_policy_version": 14,
                            "terminal_coordinate_refinement": {
                                "schema_version": 2,
                                "policy_version": 2,
                                "lane_policy_version": 14,
                                "coordinate_policy": (
                                    "same_leaf_full_surface_normalized_coordinate_v2"
                                ),
                                "measurement_policy": (
                                    "mirrored_rotating_batched_wall_v2"
                                ),
                                "radius": 2,
                            },
                        }
                    }
                ],
            }
        },
    }


def test_attention_report_requires_source_postcheck():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = _strict_report_result(shape, "a" * 64)
    del result["helion_overrides"]["autotune_provenance"][
        "post_measurement_source_verified"
    ]

    with pytest.raises(ValueError, match="incomplete strict full-autotune provenance"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": [result]}]
        )


@pytest.mark.parametrize(
    ("seq_len", "causal", "expected_num_kv"),
    ((32768, 0, 256), (524288, 1, 4096)),
)
def test_attention_report_accepts_published_all8_normalization(
    seq_len, causal, expected_num_kv
):
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": seq_len,
        "head_dim": 64,
        "dtype": "float16",
        "causal": causal,
        "biased": 0,
    }
    result = _strict_report_result(shape, "a" * 64)
    normalization = result["helion_overrides"]["autotune_provenance"][
        "flash_normalization_context"
    ]
    assert normalization["num_kv"] == expected_num_kv

    compare_attention_backends._validate_report_payloads(
        [{"shape": shape, "results": [result]}]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dtype", "torch.bfloat16"),
        ("head_dim", 128),
        ("num_kv", 32768),
        ("is_causal", False),
    ),
)
def test_attention_report_rejects_shape_mismatched_normalization_context(field, value):
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = _strict_report_result(shape, "a" * 64)
    result["helion_overrides"]["autotune_provenance"]["flash_normalization_context"][
        field
    ] = value

    with pytest.raises(ValueError, match="normalization context does not match"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": [result]}]
        )


def test_attention_report_rejects_mixed_strict_source_fingerprints():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    payloads = [
        {"shape": shape, "results": [_strict_report_result(shape, value * 64)]}
        for value in ("a", "b")
    ]

    with pytest.raises(ValueError, match="source or structural-schema"):
        compare_attention_backends._validate_report_payloads(payloads)


def test_attention_report_rejects_mixed_terminal_coordinate_surfaces():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    first = _strict_report_result(shape, "a" * 64)
    second = _strict_report_result(shape, "a" * 64)
    provenance = second["helion_overrides"]["autotune_provenance"]
    surface = provenance["flash_terminal_coordinate_surface_catalog"]
    surface["leaves"][0]["coordinates"][0]["fragment_type"] = "OtherFragment"
    provenance["flash_terminal_coordinate_surface_catalog_sha256"] = hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="within one validated report shape"):
        compare_attention_backends._validate_report_payloads(
            [
                {"shape": shape, "results": [first]},
                {"shape": shape, "results": [second]},
            ]
        )


def test_attention_report_rejects_terminal_surface_nonce_grouping_bypass():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    first = _strict_report_result(shape, "a" * 64)
    second = _strict_report_result(shape, "a" * 64)
    provenance = second["helion_overrides"]["autotune_provenance"]
    provenance["flash_normalization_context"]["nonce"] = "different-group"
    surface = provenance["flash_terminal_coordinate_surface_catalog"]
    surface["leaves"][0]["coordinates"][0]["fragment_type"] = "OtherFragment"
    provenance["flash_terminal_coordinate_surface_catalog_sha256"] = hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="within one validated report shape"):
        compare_attention_backends._validate_report_payloads(
            [
                {"shape": shape, "results": [first]},
                {"shape": shape, "results": [second]},
            ]
        )


def test_attention_report_allows_shape_specific_terminal_coordinate_surfaces():
    dense_shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 0,
        "biased": 0,
    }
    causal_shape = {**dense_shape, "causal": 1}
    dense = _strict_report_result(dense_shape, "a" * 64)
    causal = _strict_report_result(causal_shape, "a" * 64)
    provenance = causal["helion_overrides"]["autotune_provenance"]
    surface = provenance["flash_terminal_coordinate_surface_catalog"]
    surface["leaves"][0]["coordinates"][0]["fragment_type"] = "CausalFragment"
    provenance["flash_terminal_coordinate_surface_catalog_sha256"] = hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    compare_attention_backends._validate_report_payloads(
        [
            {"shape": dense_shape, "results": [dense]},
            {"shape": causal_shape, "results": [causal]},
        ]
    )


def test_attention_report_rejects_duplicate_implementation_rows():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 65536,
        "head_dim": 64,
        "dtype": "float16",
        "causal": 1,
        "biased": 0,
    }
    result = _strict_report_result(shape, "a" * 64)

    with pytest.raises(ValueError, match="duplicate implementation"):
        compare_attention_backends._validate_report_payloads(
            [{"shape": shape, "results": [result, copy.deepcopy(result)]}]
        )


def test_attention_cli_rejects_duplicate_implementations():
    with pytest.raises(SystemExit, match="duplicate implementations"):
        compare_attention_backends._validate_requested_impls(["sdpa", "sdpa"])


def test_attention_gluon_path_uses_explicit_file(tmp_path, monkeypatch):
    source = tmp_path / "attention_forward.py"
    source.write_text("# test\n")
    monkeypatch.setenv("HELION_GLUON_ATTENTION_PATH", str(source))

    assert compare_attention_backends._resolve_gluon_attention_path() == source


def test_attention_tlx_path_uses_isolated_runtime(tmp_path, monkeypatch):
    source = (
        tmp_path
        / "triton"
        / "language"
        / "extra"
        / "tlx"
        / "tutorials"
        / "blackwell_fa_ws_pipelined_persistent.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("# test\n")
    monkeypatch.setenv("HELION_TLX_RUNTIME_ROOT", str(tmp_path))

    assert compare_attention_backends._resolve_tlx_attention_path() == source


def test_attention_tlx_version_identifies_meta_triton(tmp_path, monkeypatch):
    source = tmp_path / "attention.py"
    source.write_text("# test\n")
    monkeypatch.setenv("HELION_BENCHMARK_HELION_VERSION", "test")
    monkeypatch.setenv("HELION_TLX_ATTENTION_PATH", str(source))
    monkeypatch.setenv("HELION_TLX_REVISION", "abc123")
    monkeypatch.setattr(
        compare_attention_backends,
        "_package_version",
        lambda package: "3.7.4",
    )
    monkeypatch.setattr(
        compare_attention_backends.importlib,
        "import_module",
        lambda module: SimpleNamespace(__version__="3.7.4+fb"),
    )

    version = compare_attention_backends._implementation_version("tlx")

    assert version["version_label"] == "Meta Triton 3.7.4+fb"
    assert version["version"].startswith(
        "Meta Triton 3.7.4+fb; integrated TLX; package 3.7.4; revision abc123;"
    )


def test_attention_tlx_subprocess_uses_isolated_runtime(tmp_path, monkeypatch):
    seen_env = None

    def run(cmd, **kwargs):
        nonlocal seen_env
        seen_env = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout='{"impl": "tlx"}\n', stderr="")

    monkeypatch.setenv("HELION_TLX_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "existing-pythonpath")
    monkeypatch.setattr(compare_attention_backends.subprocess, "run", run)
    args = SimpleNamespace(stream_subprocesses=False)

    returncode, payload, _, _ = compare_attention_backends._run_json_subprocess(
        ["python", "benchmark.py", "--impl", "tlx"], args
    )

    assert returncode == 0
    assert payload == {"impl": "tlx"}
    assert seen_env is not None
    assert seen_env["PYTHONPATH"] == f"{tmp_path}:existing-pythonpath"


@pytest.mark.parametrize("impl", ("fa4", "gluon", "tlx"))
def test_attention_skipped_optional_impl_does_not_resolve_source(monkeypatch, impl):
    def unexpected_resolve():
        pytest.fail("skipped implementation resolved its optional source tree")

    monkeypatch.setattr(
        compare_attention_backends, "_resolve_fa4_root", unexpected_resolve
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_resolve_gluon_attention_path",
        unexpected_resolve,
    )
    monkeypatch.setattr(
        compare_attention_backends,
        "_resolve_tlx_attention_path",
        unexpected_resolve,
    )
    args = _attention_subprocess_args(impl=impl, biased=1)

    result = getattr(compare_attention_backends, f"_benchmark_{impl}")(args)

    assert result["accuracy"] == "SKIP"
    assert "implementation skipped" in result["version"]


@pytest.mark.parametrize("impl", ("gluon", "tlx"))
def test_attention_optional_impl_skips_relu_before_resolving_source(monkeypatch, impl):
    def unexpected_resolve():
        pytest.fail("unsupported ReLU implementation resolved its source tree")

    monkeypatch.setattr(
        compare_attention_backends,
        f"_resolve_{impl}_attention_path",
        unexpected_resolve,
    )
    args = _attention_subprocess_args(
        impl=impl, dtype="bfloat16", biased=0, epilogue="relu"
    )

    result = getattr(compare_attention_backends, f"_benchmark_{impl}")(args)

    assert result["accuracy"] == "SKIP"
    assert "output epilogues" in result["skipped_reason"]


def test_attention_flexattention_backends_are_explicit():
    assert compare_attention_backends._FLEXATTENTION_BACKENDS == {
        "flexattention": "TRITON",
        "flexattention-cute": "FLASH",
    }


@pytest.mark.parametrize("shape_suite", ("representative", "variants"))
def test_attention_strict_all_shapes_rejects_biased_suite_before_gpu_check(
    monkeypatch, shape_suite
):
    args = SimpleNamespace(
        merge_json=None,
        all_shapes=True,
        shape_suite=shape_suite,
        helion_require_full_autotune=1,
        impls=[],
    )

    def unexpected_gpu_check():
        pytest.fail("GPU policy was checked before strict suite preflight")

    monkeypatch.setattr(compare_attention_backends, "parse_args", lambda: args)
    monkeypatch.setattr(
        compare_attention_backends, "_check_gpu_policy", unexpected_gpu_check
    )

    with pytest.raises(SystemExit, match="includes biased attention"):
        compare_attention_backends.main()


@pytest.mark.parametrize(
    ("shape_suite", "impls"),
    (("dense_causal8", []), ("representative", ["sdpa"])),
)
def test_attention_strict_all_shapes_accepts_searchable_or_non_helion_suite(
    shape_suite, impls
):
    args = SimpleNamespace(
        shape_suite=shape_suite,
        helion_require_full_autotune=1,
        impls=impls,
    )

    compare_attention_backends._validate_all_shapes_full_autotune(args)


def test_attention_plot_version_labels_ignore_failed_results():
    payloads = [
        {
            "results": [
                {
                    "impl": "kernelagent-closed-1x",
                    "accuracy": "FAIL",
                    "version_label": "CuTe 4.5.1",
                },
                {
                    "impl": "kernelagent-closed-1x",
                    "accuracy": "PASS",
                    "version_label": "CuTe 4.6.1",
                },
            ]
        }
    ]

    assert compare_attention_backends._versioned_impl_label(
        "kernelagent-closed-1x", payloads
    ).endswith("\nCuTe 4.6.1")


def test_attention_plot_impls_are_ordered_by_increasing_average():
    values = {
        "fast": [8.0, 10.0],
        "partial": [float("nan"), 5.0],
        "slow": [1.0, 3.0],
        "missing": [float("nan"), float("nan")],
    }

    assert compare_attention_backends._impls_by_average_performance(values) == [
        "slow",
        "partial",
        "fast",
    ]


def test_attention_plot_geomean_requires_a_complete_positive_series():
    values = {
        "complete": [4.0, 16.0],
        "partial": [float("nan"), 5.0],
        "invalid": [1.0, 0.0],
        "missing": [],
    }

    assert compare_attention_backends._geomean_performance_by_impl(values) == {
        "complete": pytest.approx(8.0)
    }


def test_attention_plot_shape_label_compacts_sequence_length():
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": 131072,
        "head_dim": 64,
        "causal": 1,
    }
    assert (
        compare_attention_backends._shape_plot_label(shape) == "causal\n2x32\n128Kx64"
    )

    shape["seq_len"] = 1536
    assert (
        compare_attention_backends._shape_plot_label(shape) == "causal\n2x32\n1536x64"
    )


def test_attention_plot_dtype_label():
    payloads = [{"shape": {"dtype": "float16"}}]
    assert compare_attention_backends._benchmark_dtype_label(payloads) == "FP16"

    payloads.append({"shape": {"dtype": "bfloat16"}})
    assert compare_attention_backends._benchmark_dtype_label(payloads) == "mixed dtypes"


def test_attention_dense_causal8_suite_uses_larger_shapes():
    shapes = compare_attention_backends._SHAPE_SUITES["dense_causal8"]
    dense_seq_lens = [shape[2] for shape in shapes if shape[5] == 0]
    causal_seq_lens = [shape[2] for shape in shapes if shape[5] == 1]

    assert dense_seq_lens == [32768, 65536, 131072, 262144]
    assert causal_seq_lens == [65536, 131072, 262144, 524288]


def test_attention_autotune_timeout_env_overrides():
    args = SimpleNamespace(
        helion_env=[],
        helion_autotune_effort=None,
        helion_autotune_budget_seconds=None,
        helion_autotune_max_generations=None,
        helion_autotune_best_of_k=None,
        helion_autotune_benchmark_timeout=180,
        helion_autotune_accuracy_check=0,
        helion_autotuner_initial_population=None,
    )

    assert compare_attention_backends._helion_env_overrides(args) == {
        "HELION_AUTOTUNE_BENCHMARK_TIMEOUT": "180",
        "HELION_AUTOTUNE_ACCURACY_CHECK": "0",
    }


def test_attention_required_full_autotune_enables_canonical_compiler_seeds(
    monkeypatch,
):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(helion_require_full_autotune=1)

    overrides = compare_attention_backends._helion_env_overrides(args)
    assert overrides["HELION_DISABLE_AUTOTUNER_HEURISTICS"] == "0"
    assert overrides["HELION_AUTOTUNER_INITIAL_POPULATION"] == "from_random"
    assert overrides["HELION_AUTOTUNER"] == ""
    assert overrides["HELION_CAP_AUTOTUNE_NUM_NEIGHBORS"] == "-1"


def test_attention_required_full_autotune_rejects_disabled_heuristics():
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        helion_env=[("HELION_DISABLE_AUTOTUNER_HEURISTICS", "1")],
    )

    with pytest.raises(SystemExit, match="conflicts"):
        compare_attention_backends._helion_env_overrides(args)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("helion_seed_config", [("block_sizes", [1, 128, 128])], "seed-config"),
        ("helion_config", [("block_sizes", [1, 128, 128])], "fixed"),
    ),
)
def test_attention_required_full_autotune_rejects_cli_configs(argument, value, message):
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        **{argument: value},
    )

    with pytest.raises(SystemExit, match=message):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_required_full_autotune_rejects_skipped_correctness():
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        skip_correctness=1,
    )

    with pytest.raises(SystemExit, match="correctness"):
        compare_attention_backends._helion_env_overrides(args)


@pytest.mark.parametrize("source", ("argument", "environment"))
def test_attention_required_full_autotune_rejects_disabled_benchmark_subprocess(
    monkeypatch, source
):
    _clear_strict_autotune_overrides(monkeypatch)
    key = "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS"
    monkeypatch.delenv(key, raising=False)
    args = _attention_subprocess_args(helion_require_full_autotune=1)
    if source == "argument":
        args.helion_env = [(key, "0")]
    else:
        monkeypatch.setenv(key, "false")

    with pytest.raises(SystemExit, match="isolated autotune benchmark subprocess"):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_non_strict_allows_disabled_benchmark_subprocess(monkeypatch):
    key = "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS"
    monkeypatch.delenv(key, raising=False)
    args = _attention_subprocess_args(
        helion_require_full_autotune=0,
        helion_env=[(key, "0")],
    )

    assert compare_attention_backends._helion_env_overrides(args)[key] == "0"


@pytest.mark.parametrize("source", ("argument", "environment"))
def test_attention_required_full_autotune_rejects_cute_flash_override(
    monkeypatch, source
):
    override = ("HELION_CUTE_FLASH_WAIT_HINT", "0")
    args = _attention_subprocess_args(helion_require_full_autotune=1)
    if source == "argument":
        args.helion_env = [override]
    else:
        monkeypatch.setenv(*override)

    with pytest.raises(SystemExit, match="CuTe flash"):
        compare_attention_backends._helion_env_overrides(args)


@pytest.mark.parametrize(
    "key,value",
    (
        ("HELION_CUTE_MMA_IMPL", "universal"),
        ("CUTE_DSL_ENABLE_ASSERTIONS", "1"),
        ("CUDA_LAUNCH_BLOCKING", "1"),
        ("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    ),
)
@pytest.mark.parametrize("source", ("argument", "environment"))
def test_attention_required_full_autotune_rejects_codegen_environment(
    monkeypatch, key, value, source
):
    monkeypatch.delenv(key, raising=False)
    args = _attention_subprocess_args(helion_require_full_autotune=1)
    if source == "argument":
        args.helion_env = [(key, value)]
    else:
        monkeypatch.setenv(key, value)

    with pytest.raises(SystemExit, match="codegen overrides"):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_required_full_autotune_rejects_ambient_helion_codegen(
    monkeypatch,
):
    monkeypatch.setenv("HELION_FAST_MATH", "1")
    args = _attention_subprocess_args(helion_require_full_autotune=1)

    with pytest.raises(SystemExit, match="codegen overrides"):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_required_full_autotune_allows_auto_cute_arch(monkeypatch):
    _clear_strict_autotune_overrides(monkeypatch)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    args = _attention_subprocess_args(helion_require_full_autotune=1)

    overrides = compare_attention_backends._helion_env_overrides(args)

    assert "CUTE_DSL_ARCH" not in overrides


def test_attention_required_full_autotune_rejects_explicit_cute_arch(monkeypatch):
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        helion_env=[("CUTE_DSL_ARCH", "sm_100a")],
    )

    with pytest.raises(SystemExit, match="codegen overrides"):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_required_full_autotune_rejects_unknown_explicit_env(monkeypatch):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        helion_env=[("HELION_UNKNOWN_EXPERIMENT", "1")],
    )

    with pytest.raises(SystemExit, match="unknown --helion-env"):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_required_full_autotune_allows_seed_and_cache_dirs(monkeypatch):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        helion_env=[
            ("HELION_AUTOTUNE_RANDOM_SEED", "123"),
            ("HELION_CACHE_DIR", "/tmp/helion-cache"),
            ("CUTE_DSL_CACHE_DIR", "/tmp/cute-cache"),
        ],
    )

    overrides = compare_attention_backends._helion_env_overrides(args)

    assert overrides["HELION_AUTOTUNE_RANDOM_SEED"] == "123"
    assert overrides["HELION_CACHE_DIR"] == "/tmp/helion-cache"
    assert overrides["CUTE_DSL_CACHE_DIR"] == "/tmp/cute-cache"


@pytest.mark.parametrize(
    "key",
    _REBENCHMARK_OVERRIDE_KEYS,
)
@pytest.mark.parametrize("source", ("argument", "environment"))
def test_attention_required_full_autotune_rejects_rebenchmark_override(
    monkeypatch, key, source
):
    _clear_strict_autotune_overrides(monkeypatch)
    args = _attention_subprocess_args(helion_require_full_autotune=1)
    if source == "argument":
        args.helion_env = [(key, "1")]
    else:
        monkeypatch.setenv(key, "1")

    with pytest.raises(SystemExit, match="rebenchmark overrides"):
        compare_attention_backends._helion_env_overrides(args)


@pytest.mark.parametrize(
    "helion_env, initial_population",
    (
        ([("HELION_AUTOTUNER", "RandomSearch")], None),
        ([("HELION_AUTOTUNER_INITIAL_POPULATION", "from_best_available")], None),
        ([("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", "1")], None),
        ([], "from_best_available"),
    ),
)
def test_attention_required_full_autotune_rejects_warm_start(
    helion_env, initial_population
):
    args = _attention_subprocess_args(
        helion_require_full_autotune=1,
        helion_env=helion_env,
        helion_autotuner_initial_population=initial_population,
    )

    with pytest.raises(SystemExit):
        compare_attention_backends._helion_env_overrides(args)


def test_attention_gpu_policy_is_opt_in(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS", raising=False)

    compare_attention_backends._check_gpu_policy()


def test_attention_gpu_policy_restricts_when_configured(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS", "6,7")

    with pytest.raises(SystemExit):
        compare_attention_backends._check_gpu_policy()


@pytest.mark.parametrize("visible", (None, "", "6,7", "6,"))
def test_attention_required_full_autotune_requires_one_visible_gpu(
    monkeypatch, visible
):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    with pytest.raises(SystemExit, match="exactly one physical GPU"):
        compare_attention_backends._validate_strict_gpu_selection(True)


def test_attention_required_full_autotune_accepts_one_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    compare_attention_backends._validate_strict_gpu_selection(True)


def test_cudagraph_defaults_off(monkeypatch):
    fake_cuda = _FakeCuda()
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)

    def fn():
        return "plain"

    assert benchmarking._maybe_cudagraph_replay(fn) is fn


def test_cudagraph_replay_wraps_callable(monkeypatch):
    import helion.runtime as helion_runtime

    fake_cuda = _FakeCuda()
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.setattr(
        helion_runtime,
        "cute_cuda_graph",
        lambda: _FakeCuteGraphContext(fake_cuda),
    )
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    calls = []

    def fn():
        calls.append("call")
        return len(calls)

    replay = benchmarking._maybe_cudagraph_replay(fn)

    assert replay() == 2
    assert calls == ["call", "call"]
    assert fake_cuda.graph_obj.replay_count == 1


def test_run_example_enables_cudagraph_only_for_final_benchmark(monkeypatch):
    import helion._testing as testing

    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)
    seen = []

    def compute_repeat(fn, *, default_cudagraph=False):
        seen.append(("compute_repeat", default_cudagraph))
        return 1

    def interleaved_bench(fns, *, repeat, desc=None, default_cudagraph=False):
        seen.append(("interleaved_bench", default_cudagraph))
        return [1.0, 2.0]

    monkeypatch.setattr(testing, "compute_repeat", compute_repeat)
    monkeypatch.setattr(testing, "interleaved_bench", interleaved_bench)

    testing.run_example(lambda x: x + 1, lambda x: x + 1, (torch.ones(1),))

    assert seen == [("compute_repeat", True), ("interleaved_bench", True)]
    assert "HELION_BENCHMARK_CUDAGRAPH" not in os.environ


def test_cudagraph_auto_falls_back_when_unavailable(monkeypatch):
    fake_cuda = _FakeCuda(available=False)
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")

    def fn():
        return "fallback"

    assert benchmarking._maybe_cudagraph_replay(fn) is fn


def test_cudagraph_auto_skips_nested_capture(monkeypatch):
    fake_cuda = _FakeCuda(capturing=True)
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")

    def fn():
        return "nested"

    assert benchmarking._maybe_cudagraph_replay(fn) is fn
