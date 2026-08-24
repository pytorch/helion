from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import Mock

from benchmarks.cute import compare_grouped_gemm_defaults as runner
from benchmarks.cute import cublaslt_grouped_gemm
from benchmarks.cute import cudnn_grouped_gemm
from benchmarks.cute import cutlass_contiguous_grouped_gemm
from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute import quack_grouped_gemm
from pretuned_kernels import _bench
from pretuned_kernels.grouped_gemm_deepgemm import grouped_gemm_deepgemm
from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles
import pytest
import torch

from helion._compiler.cute.grouped_worklist_policy import (
    get_grouped_worklist_target_policy,
)


@dataclass(frozen=True)
class _QuackConfig:
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int | None = None
    num_warps: int | None = None
    pingpong: bool = False
    is_dynamic_persistent: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    cluster_k: int = 1
    split_k: int = 1
    swap_ab: bool = False
    max_swizzle_size: int = 8
    device_capacity: int = 10
    use_tma_gather: bool = False


class _Inputs:
    def __init__(self, *, n: int = 3, k: int = 4) -> None:
        self.case = SimpleNamespace(
            total_m=3,
            n=n,
            k=k,
            groups=2,
            actual_ms=(1, 2),
        )
        self.compact_a = torch.arange(3 * k, dtype=torch.bfloat16).reshape(3, k)
        self.b = torch.arange(2 * n * k, dtype=torch.bfloat16).reshape(2, n, k)
        self.b_n_major = self.b.transpose(1, 2).contiguous().transpose(1, 2)
        self.offsets = torch.tensor((0, 1, 3), dtype=torch.int32)

    def b_for_layout(self, layout: str) -> torch.Tensor:
        return self.b if layout == "k_major" else self.b_n_major

    def compact_a_slices(self) -> tuple[torch.Tensor, ...]:
        return self.compact_a[:1], self.compact_a[1:]

    def compact_output_slices(self, output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return output[:1], output[1:]


def _prepared(provider: str, b_layout: str) -> common.PreparedImplementation:
    output = torch.empty(1)
    return common.PreparedImplementation(
        name=provider,
        call=lambda: output,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: (cast("torch.Tensor", result),),
        config={"provider": provider, "b_layout": b_layout},
    )


def test_official_cases_match_reviewed_manifest() -> None:
    cases = common.official_cases()
    assert len(cases) == len(reviewed_profiles.OFFICIAL_SHAPES) == 8
    assert reviewed_profiles.official_actual_ms() == tuple(
        case.actual_ms for case in cases
    )


def test_common_input_factory_preserves_logical_values_and_layouts() -> None:
    case = common.GroupedGemmCase("test", 0, 2, 2, 3, 4, (1, 2))
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state()

    inputs = common.make_inputs(case, torch.device("cpu"), seed=0)

    assert inputs.compact_a.shape == (3, 4)
    assert inputs.offsets.tolist() == [0, 1, 3]
    assert inputs.b.stride() == (12, 4, 1)
    assert "b_n_major" not in inputs.__dict__
    assert inputs.b_n_major.stride() == (12, 1, 3)
    assert "b_n_major" in inputs.__dict__
    assert torch.equal(inputs.b, inputs.b_n_major)
    for group, (a, expected) in enumerate(
        zip(inputs.compact_a_slices(), inputs.oracle, strict=True)
    ):
        torch.testing.assert_close(expected, a.float() @ inputs.b[group].float().T)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_canonical_layout_gb300_workloads_resolve_only_measured_target_tunings() -> (
    None
):
    policy = get_grouped_worklist_target_policy(("cuda", "NVIDIA GB300", "sm103"))
    tuned_rows = set()
    for case, shape in zip(
        common.official_cases(),
        reviewed_profiles.OFFICIAL_SHAPES,
        strict=True,
    ):
        profile = reviewed_profiles.exact_reviewed_worklist_profile(
            shape.groups,
            shape.expected_m_per_group,
            shape.n,
            shape.k,
        )
        source_tiles = (
            sum(
                common.align(actual_m, profile.source_m_tile)
                for actual_m in case.actual_ms
            )
            // profile.source_m_tile
        )
        tuning = policy.tuning_for(
            groups=case.groups,
            n=case.n,
            k=case.k,
            b_major="k",
            source_m_tile=profile.source_m_tile,
            source_tiles=source_tiles,
            num_sm=152,
        )
        if tuning is not None:
            tuned_rows.add(case.row_index)

    assert tuned_rows == {1, 2, 3, 5, 6, 7}


@pytest.mark.parametrize(
    ("name", "capability", "expected"),
    (
        ("NVIDIA B200", (10, 0), True),
        ("NVIDIA GB300", (10, 3), True),
        ("NVIDIA GB300", (10, 0), False),
        ("NVIDIA H100", (9, 0), False),
    ),
)
def test_supported_grouped_gemm_device_identity(
    name: str,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    assert common.is_supported_grouped_gemm_device("cuda", name, capability) is expected


def test_reviewed_aot_is_rejected_outside_b200() -> None:
    runner._validate_helion_selection_for_device(
        "compiler_heuristic",
        {"capability": [10, 3]},
    )
    runner._validate_helion_selection_for_device(
        "final_reviewed_aot",
        {"capability": [10, 0]},
    )
    with pytest.raises(RuntimeError, match="only on B200/SM100"):
        runner._validate_helion_selection_for_device(
            "final_reviewed_aot",
            {"capability": [10, 3]},
        )


@pytest.mark.parametrize(
    ("capability", "target"),
    (((10, 0), "100a"), ((10, 3), "103a"), ((9, 0), None)),
)
def test_cutlass_target_matches_blackwell_device(
    capability: tuple[int, int], target: str | None
) -> None:
    if target is None:
        with pytest.raises(RuntimeError, match="B200/SM100 or GB300/SM103"):
            cutlass_contiguous_grouped_gemm.cutlass_target_sm(capability)
    else:
        assert cutlass_contiguous_grouped_gemm.cutlass_target_sm(capability) == target


@pytest.mark.parametrize(
    ("timings", "valid"),
    (
        ((1.0, 2.0), True),
        ((0.0, 1.0), False),
        ((1.0, float("inf")), False),
    ),
)
def test_run_case_keeps_setup_outside_timing_and_rng_isolated(
    monkeypatch: Any,
    timings: tuple[float, float],
    valid: bool,
) -> None:
    depth = 0
    events: list[tuple[str, int]] = []

    @contextmanager
    def fork_rng(*, devices: list[int]):
        nonlocal depth
        assert devices == [0]
        events.append(("fork_enter", depth))
        depth += 1
        try:
            yield
        finally:
            depth -= 1
            events.append(("fork_exit", depth))

    inputs = SimpleNamespace(oracle=(object(),))
    prepared = {
        name: SimpleNamespace(name=name, config={"kind": name})
        for name in ("helion", "provider")
    }
    captured = {
        name: SimpleNamespace(replay=lambda name=name: events.append((name, depth)))
        for name in ("helion", "provider")
    }

    def capture(implementation: object, _oracle: object) -> tuple[object, object]:
        name = cast("Any", implementation).name
        events.append((f"capture_{name}", depth))
        return captured[name], {"ok": True}

    def make_inputs(*_args: object, seed: int) -> tuple[object, object]:
        assert seed == 0
        events.append(("make_inputs", depth))
        return inputs, object()

    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(runner, "make_exact_common_inputs", make_inputs)
    monkeypatch.setattr(
        runner,
        "prepare_helion",
        lambda *_args: events.append(("prepare_helion", depth)) or prepared["helion"],
    )
    monkeypatch.setattr(
        runner,
        "prepare_provider_default",
        lambda *_args, **_kwargs: (
            events.append(("prepare_provider", depth)) or prepared["provider"]
        ),
    )
    monkeypatch.setattr(runner, "_validated_capture", capture)
    monkeypatch.setattr(
        _bench,
        "thermal_warmup",
        lambda duration: events.append((f"warmup_{duration}", depth)),
    )

    def benchmark(functions: list[object], *, rep: int) -> tuple[float, float]:
        assert functions == [captured["helion"].replay, captured["provider"].replay]
        assert rep == 102
        events.append(("benchmark", depth))
        return timings

    monkeypatch.setattr(_bench, "bench_pre_captured_cudagraphs", benchmark)

    def invoke() -> dict[str, object]:
        return runner.run_case(
            "quack",
            cast("Any", SimpleNamespace(as_dict=lambda: {"id": "case"})),
            cast("Any", SimpleNamespace(row_index=0)),
            cast("Any", SimpleNamespace(index=0)),
            helion_selection="final_reviewed_aot",
            cutlass_root=None,
            deepgemm_root=None,
        )

    if valid:
        result = invoke()
        assert cast("Any", result["timings"])["helion_speedup"] == 2.0
    else:
        with pytest.raises(RuntimeError, match="finite and positive"):
            invoke()

    assert events == [
        ("make_inputs", 0),
        ("fork_enter", 0),
        ("prepare_helion", 1),
        ("prepare_provider", 1),
        ("capture_helion", 1),
        ("capture_provider", 1),
        ("fork_exit", 0),
        ("warmup_10000", 0),
        ("fork_enter", 0),
        ("benchmark", 1),
        ("fork_exit", 0),
    ]


def test_validated_capture_uses_warmups_and_rejects_failed_validation(
    monkeypatch: Any,
) -> None:
    prepared = SimpleNamespace(name="implementation")
    captured = object()
    calls: list[tuple[object, object]] = []
    validation = {"ok": True, "poisoned_replay_rewrote_output": True}

    def capture(value: object, *, warmups: int) -> object:
        calls.append((value, warmups))
        return captured

    monkeypatch.setattr(common, "capture_implementation", capture)
    monkeypatch.setattr(common, "validate_capture", lambda *_args: validation)
    assert runner._validated_capture(cast("Any", prepared), ()) == (
        captured,
        validation,
    )
    assert calls == [(prepared, 2)]

    validation["ok"] = False
    validation["poisoned_replay_rewrote_output"] = False
    with pytest.raises(RuntimeError, match="failed correctness"):
        runner._validated_capture(cast("Any", prepared), ())


def test_validate_capture_fails_when_replay_leaves_poisoned_output(
    monkeypatch: Any,
) -> None:
    output = torch.ones(1, dtype=torch.bfloat16)
    prepared = common.PreparedImplementation(
        name="does-not-rewrite",
        call=lambda: output,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: (cast("torch.Tensor", result),),
        config={},
    )
    captured = common.CapturedImplementation(
        prepared=prepared,
        graph=cast("Any", SimpleNamespace(replay=lambda: None)),
        result=output,
        owners=(),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    validation = common.validate_capture(captured, (torch.ones(1),))

    assert validation["poisoned_replay_rewrote_output"] is False
    assert validation["repeat_replay_exact"] is False
    assert validation["ok"] is False


def test_validate_capture_rejects_nonrepeatable_replay(monkeypatch: Any) -> None:
    output = torch.ones(1, dtype=torch.bfloat16)
    replay_count = 0

    def replay() -> None:
        nonlocal replay_count
        replay_count += 1
        output.fill_(2 if replay_count == 1 else 1)

    prepared = common.PreparedImplementation(
        name="nonrepeatable",
        call=lambda: output,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: (cast("torch.Tensor", result),),
        config={},
    )
    captured = common.CapturedImplementation(
        prepared=prepared,
        graph=cast("Any", SimpleNamespace(replay=replay)),
        result=output,
        owners=(),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    validation = common.validate_capture(captured, (torch.ones(1),))

    assert validation["poisoned_replay_rewrote_output"] is True
    assert validation["repeat_replay_exact"] is False
    assert validation["mismatch_count"] == 0
    assert validation["ok"] is False


def test_correctness_rejects_sparse_large_error() -> None:
    expected = torch.full((100,), 64.0)
    output = expected.to(torch.bfloat16)
    output[0] += 2
    prepared = common.PreparedImplementation(
        name="sparse-error",
        call=lambda: output,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: (cast("torch.Tensor", result),),
        config={},
    )
    captured = common.CapturedImplementation(
        prepared=prepared,
        graph=cast("Any", SimpleNamespace(replay=lambda: None)),
        result=output,
        owners=(),
    )

    correctness = common.check_correctness(captured, (expected,))

    assert correctness["max_normalized_diff"] < common.CORRECTNESS_MAX_NORMALIZED_DIFF
    assert correctness["mismatch_count"] == 1
    assert correctness["ok"] is False


def test_all_helion_selections_use_canonical_k_layout_and_distinct_apis(
    monkeypatch: Any,
) -> None:
    policy = runner.LIVE_AUTOTUNE_SEARCH_POLICY
    base_config = {
        "tcgen05_grouped_mode": "worklist_nm",
        "tcgen05_grouped_worklist_source_m_tile": 256,
    }
    actions: dict[str, list[str]] = {
        "compiler_heuristic": [],
        "live_autotune": [],
        "final_reviewed_aot": [],
    }
    bound_caches: dict[str, str] = {}
    current_mode = ""

    class ConfigSpec:
        compiler_default_config = SimpleNamespace(config=base_config)
        compiler_seed_configs = (compiler_default_config,)
        autotuner_heuristics = ("cute_tcgen05_grouped_worklist",)
        cute_tcgen05_search_enabled = False

        def default_config(self) -> object:
            return self.compiler_default_config

        def normalized_config(self, config: object) -> object:
            return config

    class Bound:
        def __init__(self, settings: object) -> None:
            self.settings = settings
            self.config_spec = ConfigSpec()
            self.env = SimpleNamespace(config_spec=self.config_spec)

        def set_config(self, _config: object) -> None:
            actions[current_mode].append("set_config")

        def autotune(self, _args: object, *, force: bool) -> object:
            assert force
            actions[current_mode].append("autotune")
            return SimpleNamespace(config={**base_config, "num_stages": 3})

        def __call__(self, *_args: object) -> torch.Tensor:
            return torch.empty(1, dtype=torch.bfloat16)

    class Kernel:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(
                autotune_cache="AOTAutotuneCache",
                autotune_effort=policy["autotune_effort"],
                autotune_random_seed=policy["random_seed"],
                autotune_budget_seconds=policy["budget_seconds"],
                autotune_max_generations=policy["max_generations_override"],
                autotune_best_of_k=policy["best_of_k"],
                autotune_accuracy_check=policy["accuracy_check"],
                autotune_benchmark_subprocess=policy["benchmark_subprocess"],
                autotune_benchmark_timeout=policy["benchmark_timeout_seconds"],
                autotune_compile_timeout=policy["configured_compile_timeout_seconds"],
                autotune_precompile="unset",
                autotune_adaptive_timeout=policy["adaptive_timeout"],
                autotune_initial_population_strategy=None,
                autotune_config_overrides=policy["config_overrides"],
                autotune_search_acf=policy["advanced_controls_files"],
                disable_autotuner_heuristics=False,
            )

        def bind(self, _args: object) -> Bound:
            bound_caches[current_mode] = self.settings.autotune_cache
            return Bound(self.settings)

    packed = SimpleNamespace(
        a=object(),
        worklist=object(),
        output_padding_is_zero=lambda _output: True,
        output_slices=lambda _output: (),
    )
    selected_layouts: list[str] = []
    inputs = SimpleNamespace(
        case=SimpleNamespace(expected_m_per_group=128, id="case"),
        b_for_layout=lambda layout: selected_layouts.append(layout) or object(),
    )
    profile = SimpleNamespace(
        source_m_tile=256,
        b_major="n",
        config_name="test",
    )
    lfbo = SimpleNamespace(
        max_generations=policy["effective_max_generations"],
        initial_population_strategy=policy["initial_population_strategy"],
        initial_population=policy["initial_population"],
        copies=policy["copies"],
        best_available_pad_random=policy["best_available_pad_random"],
    )
    monkeypatch.setenv("HELION_AUTOTUNER", str(policy["algorithm"]))
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    monkeypatch.setattr(common, "pack_compact_rows", lambda *_args: packed)
    monkeypatch.setattr(
        grouped_gemm_deepgemm,
        "create_grouped_gemm_deepgemm_kernel",
        Kernel,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        "helion.autotuner.effort_profile.get_effort_profile",
        lambda _effort: SimpleNamespace(
            lfbo_pattern_search=lfbo,
            finishing_rounds=policy["finishing_rounds"],
        ),
    )
    monkeypatch.setattr(
        "pretuned_kernels.grouped_gemm_deepgemm.reviewed_runtime.effective_reviewed_config",
        lambda _bound, _profile: {"effective": base_config},
    )

    selections = {}
    for current_mode in actions:
        selections[current_mode] = runner.prepare_helion(
            cast("Any", inputs),
            cast("Any", profile),
            current_mode,
        ).config

    assert actions == {
        "compiler_heuristic": ["set_config"],
        "live_autotune": ["autotune"],
        "final_reviewed_aot": [],
    }
    assert bound_caches == {
        "compiler_heuristic": "AOTAutotuneCache",
        "live_autotune": runner.LIVE_AUTOTUNE_CACHE,
        "final_reviewed_aot": "AOTAutotuneCache",
    }
    assert selections["compiler_heuristic"]["selection_api"].startswith(
        "BoundKernel.set_config"
    )
    assert selections["live_autotune"]["selection_api"] == (
        "BoundKernel.autotune(force=True)"
    )
    assert selections["compiler_heuristic"]["live_search"] is None
    assert selections["live_autotune"]["live_search"] == policy
    assert selected_layouts == ["k_major", "k_major", "k_major"]
    assert all(
        selection["benchmark_b_layout"] == "k_major"
        and not selection["benchmark_b_layout_matches_reviewed_profile"]
        for selection in selections.values()
    )


def test_cutlass_uses_tvm_ffi_before_compiling_registry_first(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class OperatorClass:
        pass

    class GlobalOptions:
        enabled = False

        @property
        def use_tvm_ffi(self) -> bool:
            return type(self).enabled

        @use_tvm_ffi.setter
        def use_tvm_ffi(self, value: bool) -> None:
            type(self).enabled = value
            events.append("use_tvm_ffi")

    def operator(name: str) -> SimpleNamespace:
        metadata = SimpleNamespace(
            design=SimpleNamespace(
                use_2cta_mma=False,
                tile_shape=(128, 128, 64),
                cluster_shape=(1, 1, 1),
            ),
            operator_class=OperatorClass,
            operator_name=name,
        )

        def compile_operator(_args: object, *, target_sm: str) -> object:
            events.append(f"compile:{name}")
            return SimpleNamespace(compiled_for=target_sm)

        return SimpleNamespace(
            metadata=metadata,
            compile=Mock(side_effect=compile_operator),
        )

    first, second = operator("first"), operator("second")

    def get_operators(
        _args: object,
        *,
        metadata_filter: Any,
        target_sm: str,
        providers: list[object],
    ) -> list[SimpleNamespace]:
        events.append("get_operators")
        assert GlobalOptions.enabled
        assert metadata_filter(first.metadata)
        assert target_sm == cutlass_contiguous_grouped_gemm.CUTLASS_TARGET_SMS[(10, 0)]
        assert providers == [ops.CuTeDSLProvider]
        return [first, second]

    ops = SimpleNamespace(
        __version__=cutlass_contiguous_grouped_gemm.CUTLASS_OPERATOR_API_VERSION,
        GlobalOptions=GlobalOptions,
        GroupedGemmArguments=lambda *_args, **_kwargs: object(),
        get_operators=get_operators,
        CuTeDSLProvider=object(),
    )
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm,
        "verify_cutlass_checkout",
        lambda root: {"head": "test", "path": str(root)},
    )
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm,
        "require_cutlass_dependencies",
        lambda: None,
    )
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm,
        "_load_operator_api",
        lambda _root: (
            ops,
            OperatorClass,
            {
                "operator_api": "operators/cutlass/operators/__init__.py",
                "operator_module": "operators/cutlass/operators/provider.py",
            },
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())

    prepared = cutlass_contiguous_grouped_gemm.prepare_cutlass_default(
        cast("Any", _Inputs(n=8, k=8)),
        cutlass_root=tmp_path,
    )

    assert events == ["use_tvm_ffi", "get_operators", "compile:first"]
    first.compile.assert_called_once()
    second.compile.assert_not_called()
    assert prepared.config["operator_name"] == "first"
    assert prepared.config["global_options"] == {"use_tvm_ffi": True}
    assert prepared.config["release_tag"] == "v4.7.0"
    assert prepared.config["module_origins"]["operator_api"].startswith(
        "operators/cutlass/"
    )


def test_cutlass_operator_modules_must_come_from_validated_checkout(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "operators" / "cutlass"
    operator_package = package_root / "operators"
    provider_file = operator_package / "provider.py"
    operator_package.mkdir(parents=True)
    (operator_package / "__init__.py").write_text("")
    provider_file.write_text("")

    cutlass = ModuleType("cutlass")
    cutlass.__path__ = []  # type: ignore[attr-defined]
    ops = ModuleType("cutlass.operators")
    ops.__file__ = str(operator_package / "__init__.py")
    provider = ModuleType(cutlass_contiguous_grouped_gemm._OPERATOR_MODULE)
    provider.__file__ = str(provider_file)

    class OperatorClass:
        pass

    OperatorClass.__module__ = cutlass_contiguous_grouped_gemm._OPERATOR_MODULE
    provider.ContiguousOffset2D3DGemmDenseOperator = OperatorClass  # type: ignore[attr-defined]
    modules = {
        "cutlass": cutlass,
        "cutlass.operators": ops,
        cutlass_contiguous_grouped_gemm._OPERATOR_MODULE: provider,
    }
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm.importlib,
        "import_module",
        lambda name: modules[name],
    )

    loaded_ops, loaded_class, origins = (
        cutlass_contiguous_grouped_gemm._load_operator_api(tmp_path)
    )

    assert loaded_ops is ops
    assert loaded_class is OperatorClass
    assert origins == {
        "operator_api": {
            "path": "operators/cutlass/operators/__init__.py",
            "sha256": common.file_sha256(operator_package / "__init__.py"),
        },
        "operator_module": {
            "path": "operators/cutlass/operators/provider.py",
            "sha256": common.file_sha256(provider_file),
        },
    }

    ops.__file__ = str(tmp_path.parent / "polluted.py")
    (tmp_path.parent / "polluted.py").write_text("")
    with pytest.raises(RuntimeError, match="outside the validated checkout"):
        cutlass_contiguous_grouped_gemm._load_operator_api(tmp_path)


def test_cudnn_uses_public_a_fallback_build_and_execute(monkeypatch: Any) -> None:
    events: list[str] = []
    float_type = object()
    mode_a, mode_fallback = object(), object()
    descriptor = Mock()
    descriptor.set_uid.return_value = descriptor
    descriptor.set_output.return_value = descriptor
    descriptor.set_data_type.return_value = descriptor
    graph = Mock()
    graph.tensor.return_value = descriptor
    graph.moe_grouped_matmul.return_value = descriptor
    graph.get_workspace_size.return_value = 0
    graph._plan_index = 1
    graph.get_execution_plan_count.return_value = 3
    graph.get_plan_name_at_index.return_value = "engine-42"
    graph.get_engine_and_knobs_at_index.return_value = (
        42,
        {"CUDNN_KNOB_TYPE_SPLIT_K": 2},
    )
    graph.build_plans.side_effect = lambda: events.append("build_plans")
    graph.execute.side_effect = lambda *_args, **_kwargs: events.append("execute")
    handle = object()
    cudnn = SimpleNamespace(
        data_type=SimpleNamespace(
            FLOAT=float_type,
            BFLOAT16=object(),
            INT32=object(),
        ),
        moe_grouped_matmul_mode=SimpleNamespace(NONE=object()),
        heur_mode=SimpleNamespace(A=mode_a, FALLBACK=mode_fallback),
        create_handle=Mock(return_value=handle),
        destroy_handle=Mock(),
        set_stream=Mock(),
        pygraph=Mock(return_value=graph),
    )
    monkeypatch.setattr(
        cudnn_grouped_gemm,
        "_validated_cudnn_runtime",
        lambda: (
            cudnn,
            cudnn_grouped_gemm.CUDNN_FRONTEND_VERSION,
            cudnn_grouped_gemm.CUDNN_BACKEND_VERSION,
            {
                "frontend": {
                    "module": {"path": "/cudnn/__init__.py", "sha256": "front"}
                },
                "requested_cuda_runtime": {
                    "path": "/cuda/libcudart.so.13",
                    "sha256": "runtime",
                },
            },
        ),
    )
    monkeypatch.setattr(
        cudnn_grouped_gemm,
        "_backend_library_identities",
        lambda: {"libraries": [{"path": "/cudnn/libcudnn.so.9"}]},
    )
    monkeypatch.setattr(
        cudnn_grouped_gemm,
        "_loaded_cuda_runtime_identity",
        lambda: {"path": "/cuda/libcudart.so.13", "sha256": "runtime"},
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=1),
    )

    prepared = cudnn_grouped_gemm.prepare_cudnn_default(
        cast("Any", _Inputs()),
        b_layout="k_major",
    )
    prepared.call()

    cudnn.pygraph.assert_called_once_with(
        intermediate_data_type=float_type,
        compute_data_type=float_type,
        handle=handle,
    )
    assert graph.moe_grouped_matmul.call_args.kwargs["compute_data_type"] is float_type
    graph.create_execution_plans.assert_called_once_with([mode_a, mode_fallback])
    assert events == ["build_plans", "execute"]
    assert prepared.config["plan"]["heuristic_modes"] == ["A", "FALLBACK"]
    assert prepared.config["plan"] == {
        "selection": "graph_build_default",
        "heuristic_modes": ["A", "FALLBACK"],
        "candidate_count": 3,
        "selected_index": 1,
        "name": "engine-42",
        "engine_id": 42,
        "knobs": [{"type": "CUDNN_KNOB_TYPE_SPLIT_K", "value": 2}],
        "workspace_bytes": 0,
    }
    runtime = cast("dict[str, object]", prepared.config["runtime"])
    assert runtime["backend_libraries"] == {
        "libraries": [{"path": "/cudnn/libcudnn.so.9"}]
    }
    assert runtime["loaded_cuda_runtime"] == {
        "path": "/cuda/libcudart.so.13",
        "sha256": "runtime",
    }
    config_hash = common.config_sha256(prepared.config)
    cast("Any", prepared.config["plan"])["engine_id"] = 43
    assert common.config_sha256(prepared.config) != config_hash


def test_cudnn_loaded_libraries_must_come_from_pinned_distribution(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "site"
    relative = (
        Path("nvidia/cudnn/lib/libcudnn.so.9"),
        Path("nvidia/cudnn/lib/libcudnn_graph.so.9"),
    )
    expected = tuple(package_root / path for path in relative)
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    distribution = SimpleNamespace(
        version=cudnn_grouped_gemm.CUDNN_BACKEND_DISTRIBUTION_VERSION,
        files=relative,
        locate_file=lambda path: package_root / path,
    )
    monkeypatch.setattr(
        cudnn_grouped_gemm,
        "_distribution",
        lambda *_args: distribution,
    )
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda _prefix: expected,
    )

    identity = cudnn_grouped_gemm._backend_library_identities()

    libraries = cast("list[dict[str, str]]", identity["libraries"])
    assert [Path(item["path"]).name for item in libraries] == [
        "libcudnn.so.9",
        "libcudnn_graph.so.9",
    ]

    foreign = tmp_path / "foreign" / "libcudnn_graph.so.9"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign")
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda _prefix: (expected[0], foreign),
    )
    with pytest.raises(RuntimeError, match="not all from the pinned distribution"):
        cudnn_grouped_gemm._backend_library_identities()


def test_cudnn_loaded_cuda_runtime_must_match_requested_library(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    expected = tmp_path / "pinned" / "libcudart.so.13"
    expected.parent.mkdir()
    expected.write_bytes(b"pinned")
    identity = {"path": str(expected), "sha256": common.file_sha256(expected)}
    monkeypatch.setattr(
        cudnn_grouped_gemm,
        "configure_cudnn_cudart_library",
        lambda: identity,
    )
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda _prefix: (expected,),
    )
    assert cudnn_grouped_gemm._loaded_cuda_runtime_identity() == identity

    foreign = tmp_path / "foreign" / "libcudart.so.13"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign")
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda _prefix: (expected, foreign),
    )
    with pytest.raises(RuntimeError, match="loaded CUDA runtimes"):
        cudnn_grouped_gemm._loaded_cuda_runtime_identity()


@pytest.mark.parametrize(
    ("b_layout", "expected_transa", "expected_a"),
    (
        (
            "k_major",
            cublaslt_grouped_gemm._CUBLAS_OP_T,
            {
                "a_rows": [4, 7],
                "a_columns": [3, 6],
                "a_leading_dimensions": [4, 7],
            },
        ),
        (
            "n_major",
            cublaslt_grouped_gemm._CUBLAS_OP_N,
            {
                "a_rows": [3, 6],
                "a_columns": [4, 7],
                "a_leading_dimensions": [3, 6],
            },
        ),
    ),
)
def test_cublaslt_descriptor_mapping_covers_both_b_layouts(
    b_layout: str,
    expected_transa: int,
    expected_a: dict[str, list[int]],
) -> None:
    transa, dimensions = cublaslt_grouped_gemm.cublaslt_layout_values(
        ((2, 3, 4, 1), (5, 6, 7, 1)),
        b_layout,
    )

    assert transa == expected_transa
    assert {name: dimensions[name] for name in expected_a} == expected_a
    assert dimensions["b_rows"] == [4, 7]
    assert dimensions["b_columns"] == [2, 5]
    assert dimensions["b_leading_dimensions"] == [4, 7]
    assert dimensions["output_rows"] == [3, 6]
    assert dimensions["output_columns"] == [2, 5]
    assert dimensions["output_leading_dimensions"] == [3, 6]


def test_cublaslt_library_identity_includes_loaded_binary_hash(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "libcublasLt.so.13"
    library_path.write_bytes(b"cublaslt-test")
    distribution = SimpleNamespace(
        version=cublaslt_grouped_gemm.CUBLASLT_DISTRIBUTION_VERSION,
        locate_file=lambda _path: library_path,
    )
    library = SimpleNamespace(
        _name=str(library_path),
        cublasLtGetVersion=lambda: cublaslt_grouped_gemm.CUBLASLT_LIBRARY_VERSION,
    )
    monkeypatch.setattr(cublaslt_grouped_gemm, "_distribution", lambda: distribution)
    monkeypatch.setattr(cublaslt_grouped_gemm.ctypes, "CDLL", lambda _path: library)
    monkeypatch.setattr(cublaslt_grouped_gemm, "_configure_library", lambda _lib: None)
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda _prefix: (library_path,),
    )
    cublaslt_grouped_gemm._validated_cublaslt_library.cache_clear()
    try:
        loaded, identity = cublaslt_grouped_gemm._validated_cublaslt_library()
    finally:
        cublaslt_grouped_gemm._validated_cublaslt_library.cache_clear()

    assert loaded is library
    assert identity == {
        "distribution": cublaslt_grouped_gemm.CUBLASLT_DISTRIBUTION,
        "package_version": cublaslt_grouped_gemm.CUBLASLT_DISTRIBUTION_VERSION,
        "library_path": str(library_path),
        "library_sha256": (
            "fc10e2d45104ebb4a1a361d83239527b892fa148561a6085535b184ce60fbf7b"
        ),
        "library_version": cublaslt_grouped_gemm.CUBLASLT_LIBRARY_VERSION,
    }

    foreign = tmp_path / "foreign" / "libcublasLt.so.13"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign")
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda _prefix: (foreign,),
    )
    cublaslt_grouped_gemm._validated_cublaslt_library.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="mapped cuBLASLt libraries"):
            cublaslt_grouped_gemm._validated_cublaslt_library()
    finally:
        cublaslt_grouped_gemm._validated_cublaslt_library.cache_clear()


def test_cublaslt_requests_one_heuristic_and_uses_rank_zero(monkeypatch: Any) -> None:
    capacities: list[int] = []

    def heuristic(*args: object) -> int:
        capacities.append(cast("int", args[7]))
        result = cast("Any", args[8])._obj
        result.workspace_size = 0
        result.state = 0
        result.waves_count = 1.0
        cast("Any", args[9])._obj.value = 1
        return 0

    library = SimpleNamespace(
        **{
            name: Mock(return_value=0)
            for name in (
                "cublasLtCreate",
                "cublasLtDestroy",
                "cublasLtMatmul",
                "cublasLtMatmulDescCreate",
                "cublasLtMatmulDescDestroy",
                "cublasLtMatmulDescSetAttribute",
                "cublasLtMatmulPreferenceCreate",
                "cublasLtMatmulPreferenceDestroy",
                "cublasLtMatmulPreferenceSetAttribute",
                "cublasLtMatrixLayoutDestroy",
            )
        },
        cublasLtMatmulAlgoGetHeuristicForStream=heuristic,
    )
    monkeypatch.setattr(
        cublaslt_grouped_gemm,
        "_validated_cublaslt_library",
        lambda: (
            library,
            {
                "library_path": "/cuda/libcublasLt.so.13",
                "library_sha256": "test-sha256",
                "library_version": 1,
            },
        ),
    )
    monkeypatch.setattr(
        cublaslt_grouped_gemm,
        "_create_grouped_matrix_layouts",
        lambda *_args: (
            ctypes.c_void_p(1),
            ctypes.c_void_p(2),
            ctypes.c_void_p(3),
        ),
    )
    monkeypatch.setattr(
        cublaslt_grouped_gemm,
        "_grouped_algorithm_supported",
        lambda *_args: True,
    )
    monkeypatch.setattr(cublaslt_grouped_gemm, "_DEFAULT_WORKSPACE_BYTES", 16)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=1, synchronize=lambda: None),
    )

    prepared = cublaslt_grouped_gemm.prepare_cublaslt_default(
        cast("Any", _Inputs()),
        b_layout="k_major",
    )
    prepared.call()

    selected = cast("Any", prepared.config["selected_algorithm"])
    assert capacities == [cublaslt_grouped_gemm.CUBLASLT_HEURISTIC_QUERY_CAPACITY]
    assert selected["heuristic_rank"] == 0
    assert prepared.config["heuristic_query_capacity"] == 1
    assert prepared.config["library"]["library_sha256"] == "test-sha256"
    library.cublasLtMatmul.assert_called_once()


def test_quack_public_default_call_contract(monkeypatch: Any) -> None:
    inputs = _Inputs()
    public_calls: list[dict[str, object]] = []
    replay_calls: list[dict[str, object]] = []
    interface = ModuleType("quack.gemm_interface")
    selected_config = _QuackConfig()
    dispatch_plan = SimpleNamespace(
        _asdict=lambda: {
            "compiled_fn": object(),
            "tile_M": 256,
            "tile_N": 256,
            "cluster_M": 2,
            "cluster_N": 1,
        }
    )

    def gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        public_calls.append({"a": a, "b": b, **kwargs})
        return cast("torch.Tensor", kwargs["out"])

    def resolved_gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        **kwargs: object,
    ) -> tuple[_QuackConfig, int, bool, object]:
        replay_calls.append({"a": a, "b": b, "output": output, **kwargs})
        return selected_config, 1, True, dispatch_plan

    interface.gemm = gemm  # type: ignore[attr-defined]
    interface.gemm_tuned = SimpleNamespace(  # type: ignore[attr-defined]
        best_config=SimpleNamespace(kwargs={"config": selected_config}),
        fn=resolved_gemm,
    )
    monkeypatch.setattr(
        quack_grouped_gemm,
        "_package_identity",
        lambda quack_root=None: {
            "module_version": "test",
            "source_root": "/tmp/quack",
            "source_provenance": quack_grouped_gemm._source_provenance(),
        },
    )
    monkeypatch.setattr(
        quack_grouped_gemm,
        "_import_quack_from_root",
        lambda _root, _name: interface,
    )
    monkeypatch.setattr(
        quack_grouped_gemm,
        "_loaded_quack_modules",
        lambda _root: {
            "quack": {"path": "quack/__init__.py", "sha256": "test"},
            "quack.gemm_interface": {
                "path": "quack/gemm_interface.py",
                "sha256": "test",
            },
        },
    )

    prepared = quack_grouped_gemm.prepare_quack_default(
        cast("Any", inputs),
        b_layout="n_major",
    )
    prepared.call()
    assert len(public_calls) == 1
    assert public_calls[0]["a"] is inputs.compact_a
    assert public_calls[0]["cu_seqlens_m"] is inputs.offsets
    assert "dynamic_scheduler" not in public_calls[0]
    assert "tuned" not in public_calls[0]
    assert "split_k" not in public_calls[0]
    assert len(replay_calls) == 2
    assert all(call["config"] is selected_config for call in replay_calls)
    assert prepared.config["requested_config"] is None
    assert prepared.config["benchmark_label"] == (
        "quack-main@c8ec3170 (post-v0.6.4, non-release)"
    )
    assert prepared.config["b_layout"] == "n_major"
    assert prepared.config["resolved_dynamic_scheduler"] is True
    assert prepared.config["selection_mode"] == "public_api_default_tuned"
    assert prepared.config["selection_timed"] is False
    assert prepared.config["tuned"] is True
    assert prepared.config["resolved_split_k"] == 1
    assert prepared.config["dispatch_plan"]["fields"]["tile_M"] == 256
    config_hash = common.config_sha256(prepared.config)
    cast("Any", prepared.config["dispatch_plan"])["fields"]["tile_M"] = 128
    assert common.config_sha256(prepared.config) != config_hash


def test_quack_source_override_rejects_native_distribution(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    distribution = SimpleNamespace(
        version=quack_grouped_gemm.QUACK_PACKAGE_METADATA_VERSION,
        files=(Path("quack/_native.so"), Path("quack/api.py")),
    )
    monkeypatch.setattr(
        quack_grouped_gemm.importlib.metadata,
        "distribution",
        lambda _name: distribution,
    )

    with pytest.raises(RuntimeError, match="native artifacts"):
        quack_grouped_gemm.verify_quack_installation(tmp_path)


def test_quack_source_override_requires_child_import_path(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    distribution = SimpleNamespace(
        version=quack_grouped_gemm.QUACK_PACKAGE_METADATA_VERSION,
        files=(Path("quack/api.py"),),
    )
    monkeypatch.setattr(
        quack_grouped_gemm.importlib.metadata,
        "distribution",
        lambda _name: distribution,
    )

    with pytest.raises(RuntimeError, match="worker import path"):
        quack_grouped_gemm.verify_quack_installation(tmp_path)


def test_quack_editable_root_requires_pep610_editable_flag(tmp_path: Path) -> None:
    distribution = SimpleNamespace(
        read_text=lambda _name: json.dumps(
            {"url": tmp_path.as_uri(), "dir_info": {"editable": False}}
        )
    )

    with pytest.raises(RuntimeError, match="not installed from an editable checkout"):
        quack_grouped_gemm._editable_root(cast("Any", distribution))


def test_quack_source_provenance_is_explicitly_not_a_release() -> None:
    provenance = quack_grouped_gemm._source_provenance()

    assert quack_grouped_gemm.QUACK_PACKAGE_METADATA_VERSION == "0.6.4"
    assert provenance == {
        "kind": "upstream_main_snapshot",
        "repository": "https://github.com/Dao-AILab/quack",
        "commit": "c8ec3170057987da0ec99883736f381ea1937cf3",
        "base_release_tag": "v0.6.4",
        "is_formal_release": False,
        "benchmark_label": "quack-main@c8ec3170 (post-v0.6.4, non-release)",
    }


def test_quack_requires_pinned_source_dependencies(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        quack_grouped_gemm.importlib.metadata,
        "version",
        lambda name: (
            "4.6.2"
            if name == "nvidia-cutlass-dsl"
            else quack_grouped_gemm.QUACK_DEPENDENCY_VERSIONS[name]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="nvidia-cutlass-dsl is 4.6.2, expected 4.7.0",
    ):
        quack_grouped_gemm._dependency_versions()


@pytest.mark.parametrize("polluted_name", ("quack", "quack.gemm_interface"))
def test_quack_rejects_preloaded_modules_outside_validated_checkout(
    monkeypatch: Any,
    tmp_path: Path,
    polluted_name: str,
) -> None:
    for name in tuple(sys.modules):
        if name == "quack" or name.startswith("quack."):
            monkeypatch.delitem(sys.modules, name)

    root = tmp_path / "checkout"
    package_dir = root / "quack"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("")
    if polluted_name != "quack":
        package = ModuleType("quack")
        package.__file__ = str(package_file)
        package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "quack", package)

    polluted_file = tmp_path / "polluted.py"
    polluted_file.write_text("")
    polluted = ModuleType(polluted_name)
    polluted.__file__ = str(polluted_file)
    monkeypatch.setitem(sys.modules, polluted_name, polluted)

    with pytest.raises(RuntimeError, match="outside the validated checkout"):
        quack_grouped_gemm._loaded_quack_modules(root)


def test_provider_dispatch_uses_one_default_preparer(
    monkeypatch: Any,
) -> None:
    inputs = cast("Any", _Inputs())
    calls: list[tuple[str, str]] = []
    quack_root = Path("/tmp/pinned-quack")

    def layout_preparer(provider: str) -> Any:
        def prepare(
            _inputs: object,
            *,
            b_layout: str,
            quack_root: Path | None = None,
        ) -> object:
            if provider == "quack":
                assert quack_root == Path("/tmp/pinned-quack")
            calls.append((provider, b_layout))
            return _prepared(provider, b_layout)

        return prepare

    modules = {
        "benchmarks.cute.cutlass_contiguous_grouped_gemm": (
            "prepare_cutlass_default",
            lambda _inputs, *, cutlass_root: (
                calls.append(("cutlass", str(cutlass_root)))
                or _prepared("cutlass", "k_major")
            ),
        ),
        "benchmarks.cute.quack_grouped_gemm": (
            "prepare_quack_default",
            layout_preparer("quack"),
        ),
        "benchmarks.cute.cudnn_grouped_gemm": (
            "prepare_cudnn_default",
            layout_preparer("cudnn"),
        ),
        "benchmarks.cute.cublaslt_grouped_gemm": (
            "prepare_cublaslt_default",
            layout_preparer("cublaslt"),
        ),
    }
    for module_name, (function_name, function) in modules.items():
        module = ModuleType(module_name)
        setattr(module, function_name, function)
        monkeypatch.setitem(sys.modules, module_name, module)

    for provider in tuple(
        provider for provider in runner.PROVIDERS if provider != "deepgemm"
    ):
        prepared = runner.prepare_provider_default(
            provider,
            inputs,
            cutlass_root=Path("/tmp/cutlass") if provider == "cutlass" else None,
            deepgemm_root=None,
            quack_root=quack_root,
        )
        assert isinstance(prepared, common.PreparedImplementation)
        assert prepared.config["benchmark_b_layout"] == "k_major"
    assert calls == [
        ("quack", "k_major"),
        ("cudnn", "k_major"),
        ("cublaslt", "k_major"),
        ("cutlass", "/tmp/cutlass"),
    ]


def test_deepgemm_uses_final_public_api_without_candidate_search(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from benchmarks.cute import grouped_gemm_deepgemm_support as support

    inputs = cast("Any", _Inputs())
    launch = Mock()
    module = SimpleNamespace(m_grouped_bf16_gemm_nt_contiguous=launch)
    monkeypatch.setattr(
        support,
        "import_deepgemm",
        lambda root, alignment: (
            module,
            {"root": str(root), "m_alignment": alignment},
        ),
    )
    prepared = runner.prepare_provider_default(
        "deepgemm",
        inputs,
        cutlass_root=None,
        deepgemm_root=tmp_path,
    )
    result = prepared.call()
    a, b, output, layout = launch.call_args.args
    assert result is output
    assert prepared.owners[0] is module
    assert b is inputs.b
    launch.assert_called_once_with(
        a,
        b,
        output,
        layout,
        compiled_dims="nk",
        use_psum_layout=False,
        ensure_zero_padding=False,
    )
    assert prepared.config["api"] == {
        "function": "m_grouped_bf16_gemm_nt_contiguous",
        "compiled_dims": "nk",
        "use_psum_layout": False,
        "ensure_zero_padding": False,
    }
    assert prepared.config["a_layout"]["alignment"] == 224
    assert prepared.config["benchmark_b_layout"] == "k_major"
