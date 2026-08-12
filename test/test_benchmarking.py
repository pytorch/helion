from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

from benchmarks.cute import compare_attention_backends
import pytest
import torch

import helion.autotuner.benchmarking as benchmarking


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


def _attention_subprocess_args(**overrides):
    args = SimpleNamespace(
        z=1,
        h=2,
        seq_len=128,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=1,
        num_runs=5,
        warmup_ms=25,
        rep_ms=100,
        seed=123,
        power_cap_w=750,
        skip_correctness=0,
        helion_force_flash_config=1,
        helion_force_autotune=0,
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


def test_attention_subprocess_forwards_helion_cute_timer():
    args = _attention_subprocess_args(
        helion_cute_benchmark_timer="event",
        helion_seed_config=[("block_sizes", [1, 64, 128])],
    )

    cmd = compare_attention_backends._build_subprocess_cmd(args, "helion-cute")

    flag_index = cmd.index("--helion-cute-benchmark-timer")
    assert cmd[flag_index + 1] == "event"
    power_index = cmd.index("--power-cap-w")
    assert cmd[power_index + 1] == "750"
    seed_index = cmd.index("--helion-seed-config")
    assert cmd[seed_index + 1] == "block_sizes=[1, 64, 128]"


def test_attention_shape_subprocess_forwards_helion_cute_timer(monkeypatch):
    args = _attention_subprocess_args(helion_cute_benchmark_timer="event")
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


def test_attention_helion_cute_timer_selects_bench_fn():
    calls = []

    def wall_timer(*args, **kwargs):
        return 1.0

    backend = SimpleNamespace(
        get_do_bench=lambda: calls.append("get_do_bench") or wall_timer
    )
    bound = SimpleNamespace(env=SimpleNamespace(backend=backend))

    wall_args = SimpleNamespace(helion_cute_benchmark_timer="wall")
    assert (
        compare_attention_backends._helion_do_bench_fn(bound, wall_args, "cute")
        is wall_timer
    )
    assert calls == ["get_do_bench"]

    event_args = SimpleNamespace(helion_cute_benchmark_timer="event")
    assert (
        compare_attention_backends._helion_do_bench_fn(bound, event_args, "cute")
        is None
    )
    assert (
        compare_attention_backends._helion_do_bench_fn(bound, wall_args, "triton")
        is None
    )
    assert calls == ["get_do_bench"]


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
    }


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


def test_attention_flexattention_backends_are_explicit():
    assert compare_attention_backends._FLEXATTENTION_BACKENDS == {
        "flexattention": "TRITON",
        "flexattention-cute": "FLASH",
    }


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


def test_attention_gpu_policy_is_opt_in(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS", raising=False)

    compare_attention_backends._check_gpu_policy()


def test_attention_gpu_policy_restricts_when_configured(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS", "6,7")

    with pytest.raises(SystemExit):
        compare_attention_backends._check_gpu_policy()


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
