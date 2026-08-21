from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import ctypes
from dataclasses import dataclass
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


@pytest.mark.parametrize(
    "foreign_module",
    (
        None,
        "cutlass.operators",
        cutlass_contiguous_grouped_gemm._OPERATOR_MODULE,
    ),
)
def test_cutlass_loader_rejects_cached_operator_outside_checkout(
    monkeypatch: Any,
    tmp_path: Path,
    foreign_module: str | None,
) -> None:
    package_root = tmp_path / "operators" / "cutlass"
    operators_file = package_root / "operators" / "__init__.py"
    operators_file.parent.mkdir(parents=True)
    operators_file.write_text("")
    operator_file = (
        package_root
        / "operators"
        / "providers"
        / "cutedsl"
        / "gemm"
        / "sm100_contiguous_offset_2d3d_dense_gemm.py"
    )
    operator_file.parent.mkdir(parents=True)
    operator_file.write_text("")
    outside_file = tmp_path / "installed" / "operator.py"
    outside_file.parent.mkdir()
    outside_file.write_text("")

    cutlass = ModuleType("cutlass")
    cutlass.__path__ = []
    operators = ModuleType("cutlass.operators")
    operators.__file__ = str(
        outside_file if foreign_module == "cutlass.operators" else operators_file
    )
    operator_module = ModuleType(cutlass_contiguous_grouped_gemm._OPERATOR_MODULE)
    operator_module.__file__ = str(
        outside_file
        if foreign_module == cutlass_contiguous_grouped_gemm._OPERATOR_MODULE
        else operator_file
    )
    operator_class = type("Operator", (), {})
    operator_module.ContiguousOffset2D3DGemmDenseOperator = operator_class
    modules = {
        "cutlass": cutlass,
        "cutlass.operators": operators,
        cutlass_contiguous_grouped_gemm._OPERATOR_MODULE: operator_module,
    }
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm.importlib,
        "import_module",
        modules.__getitem__,
    )

    if foreign_module is None:
        assert cutlass_contiguous_grouped_gemm._load_operator_api(tmp_path) == (
            operators,
            operator_class,
        )
    else:
        with pytest.raises(RuntimeError, match="outside the pinned CUTLASS checkout"):
            cutlass_contiguous_grouped_gemm._load_operator_api(tmp_path)
    assert cutlass.__path__[0] == str(package_root)


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

    def make_inputs(*_args: object) -> tuple[object, object]:
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
    assert validation["ok"] is False


def test_compiler_and_live_selection_use_distinct_kernel_apis(
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
    inputs = SimpleNamespace(
        case=SimpleNamespace(expected_m_per_group=128, id="case"),
        b_for_layout=lambda _layout: object(),
    )
    profile = SimpleNamespace(source_m_tile=256, b_major="k", config_name="test")
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

    selections = {}
    for current_mode in actions:
        selections[current_mode] = runner.prepare_helion(
            cast("Any", inputs), cast("Any", profile), current_mode
        ).config

    assert actions == {
        "compiler_heuristic": ["set_config"],
        "live_autotune": ["autotune"],
    }
    assert bound_caches == {
        "compiler_heuristic": "AOTAutotuneCache",
        "live_autotune": runner.LIVE_AUTOTUNE_CACHE,
    }
    assert selections["compiler_heuristic"]["selection_api"].startswith(
        "BoundKernel.set_config"
    )
    assert selections["live_autotune"]["selection_api"] == (
        "BoundKernel.autotune(force=True)"
    )
    assert selections["compiler_heuristic"]["live_search"] is None
    assert selections["live_autotune"]["live_search"] == policy


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
        assert target_sm == cutlass_contiguous_grouped_gemm.CUTLASS_TARGET_SM
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
        lambda _root: {"head": "test"},
    )
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm,
        "require_cutlass_dependencies",
        lambda: None,
    )
    monkeypatch.setattr(
        cutlass_contiguous_grouped_gemm,
        "_load_operator_api",
        lambda _root: (ops, OperatorClass),
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
            Path("/cuda/libcudart.so.13"),
            cudnn,
            cudnn_grouped_gemm.CUDNN_FRONTEND_VERSION,
            cudnn_grouped_gemm.CUDNN_BACKEND_VERSION,
        ),
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
    monkeypatch.setattr(cublaslt_grouped_gemm, "_distribution", lambda: object())
    monkeypatch.setattr(
        cublaslt_grouped_gemm,
        "_validated_cublaslt_library",
        lambda: (library, {"library_version": 1}),
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
    library.cublasLtMatmul.assert_called_once()


def test_quack_public_default_call_contract(monkeypatch: Any) -> None:
    inputs = _Inputs()
    calls: list[dict[str, object]] = []
    interface = ModuleType("quack.gemm_interface")

    def default_config(device: torch.device) -> _QuackConfig:
        assert device == inputs.compact_a.device
        return _QuackConfig()

    def gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append({"a": a, "b": b, **kwargs})
        return cast("torch.Tensor", kwargs["out"])

    interface.default_config = default_config  # type: ignore[attr-defined]
    interface.gemm = gemm  # type: ignore[attr-defined]
    quack = ModuleType("quack")
    quack.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "quack", quack)
    monkeypatch.setitem(sys.modules, "quack.gemm_interface", interface)
    monkeypatch.setattr(
        quack_grouped_gemm,
        "_package_identity",
        lambda: {"module_version": "test"},
    )

    prepared = quack_grouped_gemm.prepare_quack_default(
        cast("Any", inputs),
        b_layout="n_major",
    )
    prepared.call()
    assert len(calls) == 1
    assert calls[0]["a"] is inputs.compact_a
    assert calls[0]["cu_seqlens_m"] is inputs.offsets
    assert calls[0]["dynamic_scheduler"] is False
    assert calls[0]["tuned"] is False
    assert calls[0]["split_k"] == 1
    assert prepared.config["requested_config"] is None
    assert prepared.config["b_layout"] == "n_major"
    assert prepared.config["resolved_dynamic_scheduler"] is True


@pytest.mark.parametrize(
    ("profile_major", "reviewed_layout"), (("k", "k_major"), ("n", "n_major"))
)
def test_provider_dispatch_uses_one_default_preparer(
    monkeypatch: Any,
    profile_major: str,
    reviewed_layout: str,
) -> None:
    inputs = cast("Any", _Inputs())
    calls: list[tuple[str, str]] = []

    def layout_preparer(provider: str) -> Any:
        def prepare(_inputs: object, *, b_layout: str) -> object:
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
            cast("Any", SimpleNamespace(b_major=profile_major)),
            cutlass_root=Path("/tmp/cutlass") if provider == "cutlass" else None,
            deepgemm_root=None,
        )
        assert isinstance(prepared, common.PreparedImplementation)
        assert prepared.config["physical_b_layout_matches_reviewed"] is (
            provider != "cutlass" or reviewed_layout == "k_major"
        )
    assert calls == [
        ("quack", reviewed_layout),
        ("cudnn", reviewed_layout),
        ("cublaslt", reviewed_layout),
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
        cast("Any", SimpleNamespace(b_major="n")),
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
    assert prepared.config["physical_b_layout_matches_reviewed"] is False
