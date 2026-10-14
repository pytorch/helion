from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from itertools import starmap
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any

from benchmarks.cute import attngym_campaign as campaign
from benchmarks.cute import attngym_figure as figure
from benchmarks.cute import compare_attngym_backends as harness
from benchmarks.cute.attngym_report import aggregate
import pytest
import torch


@pytest.fixture
def args(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    monkeypatch.setattr(harness, "_git_commit", lambda: "test-commit")
    return argparse.Namespace(
        suite="figure",
        variant=None,
        shape_id=None,
        impl="flex-triton",
        impls=None,
        json=None,
        num_runs=2,
        warmup_ms=10,
        rep_ms=20,
        cooldown_temp=0.0,
        seed=37,
        force_autotune=0,
        skip_correctness=0,
        config=None,
        save_config=None,
        out=None,
        effort="full",
        budget=60,
    )


def _small_case(name: str = "noop"):
    original = next(case for case in figure.CASES.values() if case.name == name)
    return replace(original, b=2, hq=4, hkv=2, m=5, n=5)


def _cpu_qkv(variant):
    return (
        torch.zeros((variant.b, variant.hq, variant.m, variant.d), dtype=variant.dtype),
        torch.zeros(
            (variant.b, variant.hkv, variant.n, variant.d), dtype=variant.dtype
        ),
        torch.zeros(
            (variant.b, variant.hkv, variant.n, variant.value_dim), dtype=variant.dtype
        ),
    )


def test_case_selection_preserves_each_variant_shape():
    cases = harness._suite_cases("figure")
    assert len(cases) == 87
    assert len({case.name for case in cases.values()}) == 29
    alibi = harness._select_cases("figure", ["alibi"])
    assert len(alibi) == 3
    assert len({case.shape_id for case in alibi}) == 3
    assert harness._select_cases("figure", ["alibi"], alibi[1].shape_id) == [alibi[1]]
    assert harness._select_cases("figure", [harness._case_id(alibi[1])]) == [alibi[1]]
    with pytest.raises(ValueError, match="no matching"):
        harness._select_cases("figure", ["causal"], alibi[1].shape_id)
    with pytest.raises(ValueError, match="unknown variants"):
        harness._select_cases("figure", ["missing-variant"])
    assert len(harness._select_cases("legacy")) == 16


def test_skip_preserves_shape_parameters_and_benchmark_settings(args):
    variant = _small_case("mla-causal")
    result = harness._skipped(args, variant, "unsupported")
    assert result["accuracy"] == "SKIP"
    assert result["suite"] == "figure"
    assert result["shape_id"] == variant.shape_id
    assert result["variant"] == variant.name
    assert result["parameters"] == variant.extras
    assert result["shape"] == {
        "b": 2,
        "hq": 4,
        "hkv": 2,
        "m": 5,
        "n": 5,
        "d": 192,
        "dv": 128,
        "dtype": "bfloat16",
    }
    assert result["input_seed"] == 37
    assert result["num_runs"] == 2
    assert result["warmup_ms"] == 10
    assert result["rep_ms"] == 20
    assert result["skip_correctness"] is False
    json.dumps(result, allow_nan=False)


def test_accuracy_checks_geometry_dtype_finiteness_and_relative_error():
    variant = _small_case("mla-causal")
    shape = (variant.b, variant.hq, variant.m, variant.value_dim)
    ref = torch.full(shape, 10.0, dtype=variant.dtype)
    assert harness._check_accuracy(variant, ref + 0.125, ref)[0] == "PASS"
    assert harness._check_accuracy(variant, ref + 1.0, ref)[0] == "FAIL"
    assert harness._check_accuracy(variant, ref.float(), ref)[0] == "FAIL"
    assert harness._check_accuracy(variant, ref[..., :64], ref)[0] == "FAIL"
    for value in (float("nan"), float("inf")):
        out = ref.clone()
        out[0, 0, 0, 0] = value
        assert harness._check_accuracy(variant, out, ref) == ("FAIL", None)


def test_input_seed_changes_inputs_and_is_reproducible():
    variant = _small_case()
    first = harness._make_flex_inputs(variant, torch.device("cpu"), seed=37)[0]
    repeated = harness._make_flex_inputs(variant, torch.device("cpu"), seed=37)[0]
    changed = harness._make_flex_inputs(variant, torch.device("cpu"), seed=38)[0]
    assert all(starmap(torch.equal, zip(first, repeated, strict=True)))
    assert all(not torch.equal(a, b) for a, b in zip(first, changed, strict=True))


@pytest.mark.parametrize(
    "dtype,tolerance", ((torch.bfloat16, 0.03), (torch.float16, 0.02))
)
def test_legacy_accuracy_preserves_absolute_tolerance(dtype, tolerance):
    variant = replace(_small_case("gqa-causal"), suite="legacy", dtype=dtype)
    ref = torch.zeros(variant.b, variant.hq, variant.m, variant.value_dim, dtype=dtype)
    assert harness._check_accuracy(variant, ref + tolerance / 2, ref)[0] == "PASS"
    assert harness._check_accuracy(variant, ref + tolerance * 2, ref)[0] == "FAIL"


@pytest.mark.parametrize("impl", harness.FIGURE_IMPLS)
def test_subprocess_propagates_case_seed_and_original_options(
    args, impl, tmp_path, monkeypatch
):
    variant = harness._select_cases("figure", ["mla-causal"])[1]
    args.json = str(tmp_path / "results.jsonl")
    calls = []

    def fake_run(cmd, *, env, check):
        calls.append((cmd, env))
        child_args = argparse.Namespace(**(vars(args) | {"impl": impl}))
        output = Path(cmd[cmd.index("--json") + 1])
        output.write_text(json.dumps(harness._skipped(child_args, variant, "test")))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("HELION_BACKEND", "triton")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = harness._spawn(args, harness._case_id(variant), impl)
    cmd, env = calls[0]
    for flag, value in (
        ("--suite", "figure"),
        ("--variant", variant.shape_id),
        ("--seed", "37"),
        ("--num-runs", "2"),
        ("--warmup-ms", "10"),
        ("--rep-ms", "20"),
        ("--impl", impl),
    ):
        assert cmd[cmd.index(flag) + 1] == value
    if impl == "helion-cute":
        assert env["HELION_BACKEND"] == "cute"
    else:
        assert "HELION_BACKEND" not in env
    assert result["shape_id"] == variant.shape_id
    assert result["shape"]["dv"] == 128
    assert not Path(cmd[cmd.index("--json") + 1]).exists()


def test_subprocess_crash_retains_complete_case_identity(args, tmp_path, monkeypatch):
    variant = harness._select_cases("figure", ["mla-causal"])[1]
    args.json = str(tmp_path / "results.jsonl")
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 9)
    )
    result = harness._spawn(args, harness._case_id(variant), "fa4")
    assert result["accuracy"] == "ERROR"
    assert result["impl"] == "fa4"
    assert result["shape_id"] == variant.shape_id
    assert result["shape"]["dv"] == 128
    assert result["input_seed"] == 37
    assert "exit 9" in result["skip_reason"]


def test_single_dispatch_resolves_full_case_key(args, monkeypatch):
    variant = harness._select_cases("figure", ["alibi"])[1]
    args.variant = variant.shape_id
    captured = []

    def benchmark(options, case):
        captured.append(case)
        return harness._skipped(options, case, "test")

    monkeypatch.setattr(harness, "_benchmark_flex", benchmark)
    result = harness._run_single(args)
    assert captured == [variant]
    assert result["shape_id"] == variant.shape_id


@pytest.mark.parametrize(
    "suite,names,count",
    (("figure", "alibi", 3), ("legacy", "alibi,softcap", 2)),
)
def test_cli_explicit_selection_measures_every_case(
    suite, names, count, tmp_path, monkeypatch
):
    output = tmp_path / "results.jsonl"
    calls = []

    def spawn(options, key, impl):
        calls.append((key, impl))
        case = harness._suite_cases(options.suite)[key]
        return harness._skipped(options, case, "test")

    monkeypatch.setattr(harness, "_spawn", spawn)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_attngym_backends.py",
            "--suite",
            suite,
            "--variant",
            names,
            "--impl",
            "flex-triton",
            "--json",
            str(output),
        ],
    )
    harness.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    expected = {
        harness._case_id(case)
        for case in harness._select_cases(suite, names.split(","))
    }
    assert len(records) == count
    assert {key for key, impl in calls} == expected
    assert {row["shape_id"] for row in records} == expected


def test_cli_compile_error_retains_exact_case_identity(tmp_path, monkeypatch):
    variant = harness._select_cases("figure", ["mla-causal"])[1]
    output = tmp_path / "result.jsonl"

    def fail(options):
        raise RuntimeError("test compile failure")

    monkeypatch.setattr(harness, "_run_single", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_attngym_backends.py",
            "--suite",
            "figure",
            "--shape-id",
            variant.shape_id,
            "--impl",
            "helion-cute",
            "--json",
            str(output),
        ],
    )
    harness.main()
    result = json.loads(output.read_text())
    assert result["accuracy"] == "ERROR"
    assert result["shape_id"] == variant.shape_id
    assert result["shape"]["dv"] == 128
    assert "test compile failure" in result["skip_reason"]


def test_cudnn_is_forced_and_dense_mask_is_prepared_once(
    args, monkeypatch: pytest.MonkeyPatch
):
    variant = _small_case("alibi")
    args.impl = "sdpa"
    qkv = _cpu_qkv(variant)
    bias = torch.zeros((1, variant.hq, variant.m, variant.n), dtype=torch.float32)
    active = False
    calls: dict[str, Any] = {"bias_calls": 0}

    def dense_bias(case, device):
        assert case is variant
        calls["bias_calls"] += 1
        return bias

    def make_qkv(case, device, seed):
        calls["seed"] = seed
        return qkv

    @contextmanager
    def sdpa_kernel(backends):
        nonlocal active
        calls["backends"] = backends
        active = True
        yield
        active = False

    def sdpa(q, k, v, **kwargs):
        assert active
        calls["sdpa_kwargs"] = kwargs
        return torch.zeros(
            (variant.b, variant.hq, variant.m, variant.value_dim), dtype=variant.dtype
        )

    def run_impl(args, case, fn, qkv, **kwargs):
        fn()
        fn()
        return {"accuracy": "PASS"}

    monkeypatch.setattr(figure, "dense_bias", dense_bias)
    monkeypatch.setattr(harness.agv, "make_qkv", make_qkv)
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    monkeypatch.setattr(harness, "_run_impl", run_impl)
    assert harness._benchmark_sdpa(args, variant)["accuracy"] == "PASS"
    assert calls["bias_calls"] == 1
    assert calls["seed"] == 37
    assert calls["backends"] == [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    actual_bias = calls["sdpa_kwargs"]["attn_mask"]
    assert actual_bias.dtype == variant.dtype
    torch.testing.assert_close(actual_bias, bias.to(variant.dtype))
    assert calls["sdpa_kwargs"]["is_causal"] is False
    assert calls["sdpa_kwargs"]["enable_gqa"] is True


def test_cudnn_bias_centering_preserves_large_alibi_probabilities():
    bias = torch.arange(2048, dtype=torch.float32)[None, :] * (2**-0.5)
    value = torch.randn(
        2048, 64, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
    ).float()
    expected = torch.softmax(bias, dim=-1) @ value
    prepared = harness._prepare_cudnn_bias(bias, torch.bfloat16)
    actual = torch.softmax(prepared.float(), dim=-1) @ value
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    # The reference is sensitive to the loss of bias differences near the
    # row maximum; casting the uncentered input gives a materially wrong mix.
    uncentered = torch.softmax(bias.bfloat16().float(), dim=-1) @ value
    assert (uncentered - expected).abs().max() > 0.5


def test_cudnn_bias_centering_preserves_fully_masked_rows():
    bias = torch.tensor([[4.0, -float("inf")], [-float("inf"), -float("inf")]])
    actual = harness._prepare_cudnn_bias(bias, torch.bfloat16)
    expected = torch.tensor(
        [[0.0, -float("inf")], [-float("inf"), -float("inf")]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.isnan(actual).any()


def test_nonlinear_sdpa_skip_allocates_no_inputs_or_dense_mask(
    args, monkeypatch: pytest.MonkeyPatch
):
    args.impl = "sdpa"

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported variants must not allocate inputs or bias")

    monkeypatch.setattr(figure, "dense_bias", unexpected)
    monkeypatch.setattr(harness.agv, "make_qkv", unexpected)
    result = harness._benchmark_sdpa(args, _small_case("softcap"))
    assert result["accuracy"] == "SKIP"
    assert result["shape_id"].startswith("softcap.")
    assert "not expressible" in result["skip_reason"]


def test_fa4_unsupported_variant_skips_before_allocating_inputs(args, monkeypatch):
    args.impl = "fa4"

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported FA4 variant must not allocate inputs")

    monkeypatch.setattr(harness.agv, "make_qkv", unexpected)
    result = harness._benchmark_fa4(args, _small_case("alibi"))
    assert result["accuracy"] == "SKIP"
    assert result["shape_id"].startswith("alibi.")


def test_pure_causal_sdpa_uses_native_cudnn_mask(args, monkeypatch: pytest.MonkeyPatch):
    args.impl = "sdpa"
    variant = _small_case("mla-causal")
    qkv = _cpu_qkv(variant)
    captured = {}

    def unexpected_bias(*args, **kwargs):
        raise AssertionError("pure causal attention should use the native cuDNN mask")

    def sdpa(q, k, v, **kwargs):
        captured.update(kwargs)
        return q

    def run_impl(args, case, fn, qkv, **kwargs):
        fn()
        return {"accuracy": "PASS"}

    monkeypatch.setattr(figure, "dense_bias", unexpected_bias)
    monkeypatch.setattr(harness.agv, "make_qkv", lambda case, device, seed: qkv)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    monkeypatch.setattr(harness, "_run_impl", run_impl)
    harness._benchmark_sdpa(args, variant)
    assert captured["is_causal"] is True
    assert captured["attn_mask"] is None
    assert captured["enable_gqa"] is True


@pytest.mark.parametrize(
    "name,causal,window,softcap",
    (
        ("noop", False, (None, None), 0.0),
        ("causal", True, (None, None), 0.0),
        ("gqa-causal", True, (None, None), 0.0),
        ("mla-causal", True, (None, None), 0.0),
        ("softcap", False, (None, None), 30.0),
        ("sliding-window", True, (1024, 0), 0.0),
        ("gemma2", True, (1024, 0), 50.0),
    ),
)
def test_fa4_entry_receives_mask_score_options_and_asymmetric_dv(
    args, name, causal, window, softcap, monkeypatch
):
    args.impl = "fa4"
    variant = _small_case(name)
    captured = {}

    def make_qkv(case, device, seed):
        captured["seed"] = seed
        return _cpu_qkv(case)

    def flash_attn_func(q, k, v, **kwargs):
        captured["shapes"] = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        captured["kwargs"] = kwargs
        return torch.zeros(
            (variant.b, variant.m, variant.hq, variant.value_dim), dtype=variant.dtype
        ), None

    def run_impl(options, case, fn, inputs, **kwargs):
        output = fn()
        assert output.shape == (variant.b, variant.hq, variant.m, variant.value_dim)
        return {"accuracy": "PASS"}

    cab = ModuleType("compare_attention_backends")
    cab._import_fa4 = lambda: SimpleNamespace(flash_attn_func=flash_attn_func)
    monkeypatch.setitem(sys.modules, "compare_attention_backends", cab)
    monkeypatch.setattr(harness.agv, "make_qkv", make_qkv)
    monkeypatch.setattr(harness, "_run_impl", run_impl)
    assert harness._benchmark_fa4(args, variant)["accuracy"] == "PASS"
    assert captured["seed"] == 37
    assert captured["kwargs"] == {
        "softmax_scale": None,
        "causal": causal,
        "window_size": window,
        "softcap": softcap,
        "return_lse": False,
    }
    assert captured["shapes"][2][-1] == variant.value_dim


def test_fa4_document_adapter_flattens_and_restores_the_layout(
    args, monkeypatch: pytest.MonkeyPatch
):
    args.impl = "fa4"
    variant = replace(
        _small_case("document"),
        extras={"causal": True, "document_lengths": [2, 3]},
    )
    captured = {}
    qkv = _cpu_qkv(variant)
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    def flash_attn_varlen_func(q, k, v, **kwargs):
        captured["shapes"] = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        captured["kwargs"] = kwargs
        return (
            torch.zeros(
                (variant.b * variant.m, variant.hq, variant.value_dim),
                dtype=variant.dtype,
            ),
            None,
        )

    fc = SimpleNamespace(
        __file__="/test/fa4/flash_attn/cute/__init__.py",
        flash_attn_varlen_func=flash_attn_varlen_func,
    )
    cab = ModuleType("compare_attention_backends")
    cab._import_fa4 = lambda: fc
    monkeypatch.setitem(sys.modules, "compare_attention_backends", cab)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(harness.agv, "make_qkv", lambda case, device, seed: qkv)

    def run_impl(args, case, fn, qkv, **kwargs):
        output = fn()
        assert output.shape == (variant.b, variant.hq, variant.m, variant.value_dim)
        assert output.dtype == variant.dtype
        return {"accuracy": "PASS"}

    monkeypatch.setattr(harness, "_run_impl", run_impl)
    result = harness._benchmark_fa4(args, variant)
    assert result["accuracy"] == "PASS"
    assert captured["shapes"] == ((10, 4, 64), (10, 2, 64), (10, 2, 64))
    kwargs = captured["kwargs"]
    assert kwargs["cu_seqlens_q"].tolist() == [0, 2, 5, 7, 10]
    assert kwargs["cu_seqlens_k"] is kwargs["cu_seqlens_q"]
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3
    assert kwargs["causal"] is True
    assert kwargs["return_lse"] is False


@pytest.mark.parametrize("impl", ("flex-triton", "flex-triton-ma", "flex-cute"))
def test_flex_backend_is_explicit_and_compile_shapes_are_static(
    args, impl, monkeypatch: pytest.MonkeyPatch
):
    args.impl = impl
    variant = _small_case()
    qkv = _cpu_qkv(variant)
    captured = {}

    def make_inputs(case, device, seed):
        captured["seed"] = seed
        return qkv, None, None

    def fake_compile(fn, **options):
        captured["compile"] = options

        def compiled(q, k, v, **kwargs):
            captured["call"] = kwargs
            return q

        return compiled

    cab = ModuleType("compare_attention_backends")
    cab._import_fa4 = lambda: None
    monkeypatch.setitem(sys.modules, "compare_attention_backends", cab)
    monkeypatch.setattr(harness, "_make_flex_inputs", make_inputs)
    monkeypatch.setattr(torch, "compile", fake_compile)

    def run_impl(args, case, fn, qkv, **kwargs):
        fn()
        return {"accuracy": "PASS"}

    monkeypatch.setattr(harness, "_run_impl", run_impl)
    harness._benchmark_flex(args, variant)
    assert captured["seed"] == 37
    assert captured["compile"]["dynamic"] is False
    assert captured["compile"]["fullgraph"] is True
    assert captured["compile"]["mode"] == (
        "max-autotune-no-cudagraphs" if impl == "flex-triton-ma" else None
    )
    assert captured["call"]["kernel_options"] == {
        "BACKEND": "FLASH" if impl == "flex-cute" else "TRITON"
    }


def test_generic_helion_route_saves_distinct_per_shape_configs(
    args, tmp_path, monkeypatch
):
    variants = [
        replace(case, b=2, hq=4, hkv=2, m=5, n=5)
        for case in harness._select_cases("figure", ["alibi"])
    ]
    calls = []

    class Config:
        def to_json(self):
            return '{"block_sizes": [128]}'

    class Bound:
        def autotune(self, call_args, force):
            calls.append(force)
            return Config()

        def set_config(self, config):
            assert config.block_sizes == [128]

        def __call__(self, q, k, v):
            return torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=q.dtype)

    def kernel_call_args(case, q, k, v, device):
        return SimpleNamespace(bind=lambda call_args: Bound()), (q, k, v)

    def unexpected_legacy(*args, **kwargs):
        raise AssertionError("figure cases use the generic Helion kernel")

    def run_impl(options, case, fn, inputs, **kwargs):
        output = fn()
        assert output.shape == (case.b, case.hq, case.m, case.value_dim)
        return {"accuracy": "PASS"}

    generic = ModuleType("benchmarks.cute.attngym_flex")
    generic.kernel_call_args = kernel_call_args
    legacy = ModuleType("attngym_kernels")
    legacy.kernel_call_args = unexpected_legacy
    monkeypatch.setitem(sys.modules, "benchmarks.cute.attngym_flex", generic)
    monkeypatch.setitem(sys.modules, "attngym_kernels", legacy)
    monkeypatch.setattr(
        harness.agv, "make_qkv", lambda case, device, seed: _cpu_qkv(case)
    )
    monkeypatch.setattr(harness, "_run_impl", run_impl)
    args.save_config = str(tmp_path)
    args.force_autotune = 1
    for case in variants:
        for impl in harness.HELION_IMPLS:
            args.impl = impl
            monkeypatch.setenv(
                "HELION_BACKEND", "cute" if impl == "helion-cute" else "triton"
            )
            harness._benchmark_helion(args, case)
    configs = list(tmp_path.glob("*.json"))
    assert len(configs) == 6
    assert {path.name for path in configs} == {
        f"{case.shape_id}.{impl}.json"
        for case in variants
        for impl in harness.HELION_IMPLS
    }
    assert calls == [True] * 6
    args.config = str(configs[0])
    args.save_config = None
    harness._benchmark_helion(args, variants[0])
    assert len(calls) == 6


def test_failed_accuracy_retains_original_timing_settings(
    args, monkeypatch: pytest.MonkeyPatch
):
    variant = _small_case()
    qkv = _cpu_qkv(variant)
    output = torch.ones(
        (variant.b, variant.hq, variant.m, variant.value_dim), dtype=variant.dtype
    )
    captured = {}
    stats = {
        "best_ms": 1.0,
        "median_ms": 1.0,
        "mean_ms": 1.0,
        "std_ms": 0.0,
        "runs_ms": [1.0, 1.0],
    }

    def bench(fn, **kwargs):
        captured.update(kwargs)
        return stats

    monkeypatch.setattr(harness, "_bench_helpers", lambda: (bench, lambda: None))
    monkeypatch.setattr(
        harness.agv, "reference_output", lambda *args: torch.zeros_like(output)
    )
    monkeypatch.setattr(harness.agv, "variant_flops", lambda *args: 1e9)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA GB300")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 3))
    result = harness._run_impl(args, variant, lambda: output, qkv)
    assert result["accuracy"] == "FAIL"
    assert result["median_ms"] == 1.0
    assert captured["num_runs"] == 2
    assert captured["warmup_ms"] == 10
    assert captured["rep_ms"] == 20
    assert result["shape_id"] == variant.shape_id
    report = aggregate([result], expected_shapes=1)
    row = next(row for row in report["variants"] if row["variant"] == variant.name)
    assert row["impls"][args.impl]["mean_tflops"] is None


def test_campaign_shards_every_case_once():
    cases = harness._select_cases("figure")
    shards = campaign._shard_cases(cases, [0, 2, 3])
    assert sorted(len(shard) for shard in shards.values()) == [29, 29, 29]
    assert {case.shape_id for shard in shards.values() for case in shard} == set(
        figure.CASES
    )
    with pytest.raises(ValueError, match="without duplicates"):
        campaign._shard_cases(cases, [0, 0])


def test_campaign_keeps_case_backends_on_one_gpu_and_records_child_failures(
    args, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    args.out = str(tmp_path / "campaign")
    args.seed = 0
    cases = harness._select_cases("figure", ["mla-causal"])[:2]
    impls = ["fa4", "helion-cute"]
    calls = []

    def fake_run(cmd, *, env, stdout, stderr, check):
        calls.append((cmd, env))
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 9)
        shape_id = cmd[cmd.index("--shape-id") + 1]
        impl = cmd[cmd.index("--impl") + 1]
        variant = figure.CASES[shape_id]
        result_args = argparse.Namespace(**(vars(args) | {"impl": impl}))
        result = harness._skipped(result_args, variant, "test")
        result["physical_gpu"] = env["CUDA_VISIBLE_DEVICES"]
        with Path(cmd[cmd.index("--json") + 1]).open("a") as results:
            results.write(json.dumps(result) + "\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert campaign.run_gpu(3, cases, impls, args) == 1
    records = [
        json.loads(line)
        for line in Path(f"{args.out}_gpu3.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    assert {record["physical_gpu"] for record in records} == {"3"}
    assert records[0]["accuracy"] == "ERROR"
    assert all(record["shape"]["dv"] == 128 for record in records)
    assert all(record["suite"] == "figure" for record in records)
    assert all(record["input_seed"] == 0 for record in records)
    assert len(list(Path(f"{args.out}_logs").glob("*.log"))) == 4
    for cmd, env in calls:
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        if cmd[cmd.index("--impl") + 1] == "helion-cute":
            assert env["HELION_BACKEND"] == "cute"
            assert env["HELION_AUTOTUNE_EFFORT"] == "full"
            assert env["HELION_SKIP_CACHE"] == "1"
            assert cmd[cmd.index("--force-autotune") + 1] == "1"


def test_campaign_dry_run_writes_all_case_assignments_without_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    out = tmp_path / "campaign"

    def unexpected_process(*args, **kwargs):
        raise AssertionError("dry run must not start processes")

    monkeypatch.setattr(subprocess, "run", unexpected_process)
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_process)
    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_process)
    monkeypatch.setattr(
        harness, "_git_commit", lambda root=harness.REPO_ROOT: "test-commit"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attngym_campaign.py",
            "--out",
            str(out),
            "--suite",
            "figure",
            "--gpus",
            "0,3",
            "--dry-run",
        ],
    )
    campaign.main()
    manifest = json.loads(Path(f"{out}_manifest.json").read_text())
    assert len(manifest["campaign"]["case_gpus"]) == 87
    assert manifest["campaign"]["impls"] == list(campaign.FIGURE_IMPLS)
    assert manifest["campaign"]["input_seed"] == 0
    assert manifest["campaign"]["skip_correctness"] is False
