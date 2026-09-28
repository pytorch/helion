from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import Mock

import pytest
import torch

import helion
from helion import exc
from helion.autotuner import base_search
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner import kernel_args
from helion.autotuner import precompile_future
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmarking import MirroredBenchmarkTrace
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from pathlib import Path


@pytest.fixture(autouse=True)
def _cpu_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (benchmark_job, benchmark_provider, precompile_future):
        monkeypatch.setattr(module, "synchronize_device", lambda: None)


def _pure_reference(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    result = out.clone()
    result[: x.numel()] = x + 1
    return result


def _good(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out[: x.numel()] = x + 1
    return out


def _bad(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.fill_(19)
    x.fill_(23)
    return out


def _provider(
    args: Sequence[object],
    baseline: Callable[..., object] = _pure_reference,
) -> LocalBenchmarkProvider:
    settings = Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=0,
        autotune_baseline_rtol=0,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
        autotune_progress_bar=False,
        autotune_log_level=logging.CRITICAL,
    )
    kernel = Mock()
    kernel.env.process_group_name = None
    config_spec = Mock()
    config_spec.cute_flash_search_enabled = False
    config_spec.compiler_seed_timeout_retry_repetitions = None
    config_spec.backend.get_do_bench.return_value = None
    return LocalBenchmarkProvider(
        kernel,
        settings,
        config_spec,
        args,
        AutotuningLogger(settings),
        AutotuneMetrics(),
    )


@pytest.mark.parametrize("wall_clock", [False, True])
def test_worker_cached_template_survives_bad_then_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wall_clock: bool
) -> None:
    args = (torch.arange(4.0), torch.zeros(8))
    args_path = str(tmp_path / "args.pt")
    baseline_path = str(tmp_path / "baseline.pt")
    torch.save(args, args_path)
    torch.save(_pure_reference(*args), baseline_path)
    template = kernel_args.load_trusted_kernel_args(args_path)
    assert kernel_args.load_trusted_kernel_args(args_path) is template
    functions = {"bad": _bad, "good": _good}
    monkeypatch.setattr(
        benchmark_job, "_load_compiled_fn_for_worker", functions.__getitem__
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    timed = False
    clone_calls = 0
    clone = kernel_args._clone_args

    def private(*args: Any, **kwargs: Any) -> Sequence[object]:
        nonlocal clone_calls
        assert not timed
        clone_calls += 1
        return clone(*args, **kwargs)

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        nonlocal timed
        assert kwargs == {
            "return_mode": "median",
            "warmup": 1,
            "rep": 50,
            "fixed_repetitions": None,
            "probe_long_kernel": False,
        }
        timed = True
        fn()
        fn()
        timed = False
        return 1.0

    monkeypatch.setattr(benchmark_job, "_clone_args", private)
    monkeypatch.setattr(benchmark_job, "do_bench", bench)
    monkeypatch.setattr(benchmark_job, "do_bench_generic", bench)
    for name in ("bad", "good", "good"):
        assert (
            benchmark_job.BenchmarkJob(
                cast("Any", name), args_path, use_wall_clock=wall_clock
            )()
            == 1.0
        )
        result = benchmark_job.AccuracyCheckJob(
            cast("Any", name), args_path, baseline_path, atol=0, rtol=0
        )()
        assert result.ok is (name == "good")
        torch.testing.assert_close(template, args, rtol=0, atol=0)
    assert clone_calls == 6  # one per job, not per timed invocation
    kernel_args.load_trusted_kernel_args.cache_clear()
    benchmark_job._load_trusted_baseline_output.cache_clear()


@pytest.mark.parametrize("accuracy", [False, True])
def test_in_process_accuracy_and_timing_have_separate_private_storage(
    monkeypatch: pytest.MonkeyPatch, accuracy: bool
) -> None:
    args = (torch.arange(4.0), torch.zeros(8))
    provider = _provider(args)
    provider.settings.autotune_accuracy_check = accuracy
    assert provider.mutated_arg_indices == []
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    first_timing_call = True
    timed = False
    clone_calls = 0
    clone = kernel_args._clone_args

    def private(*args: Any, **kwargs: Any) -> Sequence[object]:
        nonlocal clone_calls
        assert not timed
        clone_calls += 1
        return clone(*args, **kwargs)

    def checked(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        seen.append((x, out))
        return _good(x, out)

    def benchmarked(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        nonlocal first_timing_call
        if first_timing_call:
            assert torch.equal(out, torch.zeros_like(out))
            first_timing_call = False
        seen.append((x, out))
        return _good(x, out)

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        nonlocal timed
        assert kwargs == {
            "return_mode": "median",
            "warmup": 1,
            "rep": 50,
            "process_group_name": None,
        }
        timed = True
        fn()
        fn()
        timed = False
        return 1.0

    monkeypatch.setattr(benchmark_provider, "_clone_args", private)
    monkeypatch.setattr(benchmark_provider, "do_bench", bench)
    cast("Any", provider.kernel).bench_compile_config.return_value = benchmarked
    if accuracy:
        assert provider._benchmark_function(
            helion.Config(), cast("Any", _bad)
        ) == float("inf")
        assert provider._autotune_metrics.num_accuracy_failures == 1
    assert provider._benchmark_function(helion.Config(), cast("Any", checked)) == 1.0
    assert clone_calls == 2 + int(accuracy)
    assert seen[0][0].data_ptr() != seen[1][0].data_ptr()
    assert all(item[0] is not args[0] and item[1] is not args[1] for item in seen)
    assert torch.equal(args[0], torch.arange(4.0))
    assert torch.equal(args[1], torch.zeros(8))


@pytest.mark.parametrize("mutating_reference", [False, True])
def test_reference_mutation_comparison_semantics_are_unchanged(
    mutating_reference: bool,
) -> None:
    args = (torch.arange(4.0), torch.zeros(8))
    provider = _provider(args, _good if mutating_reference else _pure_reference)
    assert provider.mutated_arg_indices == ([1] if mutating_reference else [])
    working = kernel_args._clone_args(args, None)
    assert provider._validate_against_baseline(
        helion.Config(), _good(*cast("Any", working)), working
    )
    # A pure reference must not require candidate output arguments to remain
    # zero. Conversely, a mutating reference still validates post-arguments.
    assert provider._validate_against_baseline(
        helion.Config(), provider._baseline_output, args
    ) is (not mutating_reference)
    torch.testing.assert_close(args, (torch.arange(4.0), torch.zeros(8)))


def test_subprocess_accuracy_fallback_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = (torch.arange(4.0), torch.zeros(8))
    provider = _provider(args)
    monkeypatch.setattr(provider, "_run_subprocess_benchmark_job", lambda *a, **kw: 1.0)
    monkeypatch.setattr(provider, "_run_subprocess_accuracy_check_job", lambda fn: None)
    assert provider._benchmark_function_subprocess(
        helion.Config(), cast("Any", _bad)
    ) == float("inf")
    assert (
        provider._benchmark_function_subprocess(helion.Config(), cast("Any", _good))
        == 1.0
    )
    torch.testing.assert_close(args, (torch.arange(4.0), torch.zeros(8)))


@pytest.mark.parametrize("mode", ["fork", "spawn"])
def test_precompile_parent_never_passes_template(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["fork", "spawn"]
) -> None:
    args = (torch.arange(4.0), torch.zeros(8))
    provider = _provider(args)
    provider.settings.autotune_precompile = mode
    monkeypatch.setattr(provider, "_precompile_context", lambda: None)
    monkeypatch.setattr(provider, "_next_precompile_result_path", lambda: "unused")

    def create(**kwargs: Any) -> None:
        _bad(*kwargs["args"])
        assert kwargs["args"][0] is not args[0]

    monkeypatch.setattr(precompile_future.PrecompileFuture, "create", create)
    provider._create_precompile_future(helion.Config(), cast("Any", _bad))
    torch.testing.assert_close(args, (torch.arange(4.0), torch.zeros(8)))


def test_spawn_precompile_clones_cached_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = (torch.arange(4.0), torch.zeros(8))
    path = str(tmp_path / "args.pt")
    torch.save(args, path)
    template = kernel_args.load_trusted_kernel_args(path)
    monkeypatch.setattr(precompile_future, "start_isolated_process_group", lambda: None)
    monkeypatch.setattr(precompile_future, "_load_compiled_fn", lambda spec: _bad)
    written = Mock()
    monkeypatch.setattr(precompile_future, "_write_result_file", written)
    exited = Mock()
    monkeypatch.setattr(precompile_future.os, "_exit", exited)
    for _ in range(2):
        precompile_future._run_kernel_in_subprocess_spawn(
            cast("Any", "bad"), path, "unused", "decorator"
        )
        torch.testing.assert_close(template, args)
    assert written.call_args.args == ("unused", {"status": "ok"})
    assert exited.call_args.args == (0,)
    kernel_args.load_trusted_kernel_args.cache_clear()


def _search(args: Sequence[object]) -> PopulationBasedSearch:
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.settings = Settings(
        autotune_progress_bar=False, autotune_log_level=logging.CRITICAL
    )
    search.args = args
    search.log = AutotuningLogger(search.settings)
    search.best_perf_so_far = 1.0
    search.kernel = cast(
        "Any", SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    )
    search.benchmark_provider = cast(
        "Any",
        SimpleNamespace(
            mutated_arg_indices=[],
            benchmark_isolated=lambda *a, **kw: None,
        ),
    )
    search.config_spec = cast("Any", SimpleNamespace(backend=None))
    return search


@pytest.mark.parametrize(
    "mode",
    ["sequential", "interleaved", "custom", "mirrored", "device", "sacrificial", "oom"],
)
def test_finalists_never_share_storage(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    original = torch.zeros(4)
    search = _search((original,))
    seen: list[torch.Tensor] = []

    def candidate(value: torch.Tensor) -> None:
        assert torch.equal(value, torch.zeros_like(value))
        seen.append(value)
        value.fill_(7)

    members = [
        PopulationMember(cast("Any", candidate), [1.0], [], helion.Config(num_warps=n))
        for n in (4, 8)
    ]

    def multiple(fns: Sequence[Callable[[], object]], **kwargs: object) -> list[float]:
        for fn in fns:
            fn()
        return [1.0] * len(fns)

    def single(fn: Callable[[], object], **kwargs: object) -> float:
        fn()
        return 1.0

    def mirrored(
        fns: Sequence[Callable[[], object]], **kwargs: object
    ) -> MirroredBenchmarkTrace:
        return MirroredBenchmarkTrace([[0, 1]], [[1.0, 1.0]], multiple(fns))

    def device(
        fns: Sequence[Callable[[], object]],
        reference: Callable[[], object],
        **kwargs: object,
    ) -> list[tuple[float, float]]:
        multiple(fns)
        reference()
        return [(1.0, 0.0), (1.0, 0.0)]

    monkeypatch.setattr(base_search, "do_bench", single)
    monkeypatch.setattr(base_search, "interleaved_bench", multiple)
    monkeypatch.setattr(base_search, "mirrored_bench_generic", mirrored)
    if mode == "custom":
        search.settings.autotune_benchmark_fn = multiple
    if mode == "oom":
        monkeypatch.setattr(
            base_search,
            "_clone_args",
            Mock(side_effect=torch.OutOfMemoryError("private allocation")),
        )
        with pytest.raises(exc.AutotuneError, match="candidate-private"):
            search.rebenchmark(members, use_isolated=False)
        assert seen == []
    elif mode == "mirrored":
        search.mirrored_rebenchmark(members, desc="test", target_ms=1)
    elif mode == "device":
        search._run_final_pick_verification_device_micros(
            members[0], members, device_micros_bench=device
        )
    else:
        search.rebenchmark(
            members,
            use_isolated=mode == "sacrificial",
            candidate_private_args=mode == "sacrificial",
            use_interleaved=mode == "interleaved",
            confirm_suspicious=False,
            target_ms=1,
        )
    assert len({value.data_ptr() for value in seen}) == len(seen)
    assert all(value is not original for value in seen)
    assert torch.equal(original, torch.zeros_like(original))


def test_multishape_reference_does_not_write_child_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi = MultiShapeBenchmarkProvider.__new__(MultiShapeBenchmarkProvider)
    multi.args = cast("Any", SimpleNamespace(relative_to="baseline"))
    children = [_provider((torch.arange(float(n)), torch.zeros(n * 2))) for n in (2, 4)]
    for child in children:
        # Simulate a reference whose writes were equal to the initial values
        # during discovery: this must not suppress storage isolation.
        child.settings.autotune_baseline_fn = _bad

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        fn()
        return 1.0

    monkeypatch.setattr(benchmark_provider, "do_bench", bench)
    for i, child in enumerate(children):
        assert multi._measure_reference(child, i) == 1.0
        assert torch.equal(
            cast("torch.Tensor", child.args[0]), torch.arange(float((i + 1) * 2))
        )
        assert torch.count_nonzero(cast("torch.Tensor", child.args[1])) == 0


def test_clone_compatibility_and_nonleaf_aliases() -> None:
    assert benchmark_provider._clone_args is kernel_args._clone_args
    leaf = torch.arange(12.0, requires_grad=True)
    nonleaf = leaf * 2
    view = nonleaf[1::2]
    args = (nonleaf, view, view)
    cloned = cast("tuple[torch.Tensor, ...]", kernel_args._clone_args(args, None))
    assert cloned[1] is cloned[2]
    assert cloned[0].untyped_storage()._cdata == cloned[1].untyped_storage()._cdata
    assert cloned[0].untyped_storage()._cdata != nonleaf.untyped_storage()._cdata
    assert all(value.requires_grad and value.is_leaf for value in cloned)
    assert cloned[1].stride() == view.stride()
    assert cloned[1].storage_offset() == view.storage_offset()


def test_symmetric_signal_pad_replacement_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.ones(4)
    clone = tensor.clone()
    monkeypatch.setattr(
        kernel_args, "is_symm_mem_tensor", lambda arg, group: arg is tensor
    )
    make_clone = Mock(return_value=clone)
    monkeypatch.setattr(kernel_args, "_clone_symm_mem_tensor", make_clone)
    monkeypatch.setattr(
        kernel_args,
        "get_signal_pad_ptrs_dev",
        lambda arg, group: 11 if arg is tensor else 22,
    )
    result = kernel_args._clone_args((tensor, tensor, 11), "group")
    assert result[0] is result[1] is clone
    assert result[2] == 22
    make_clone.assert_called_once_with(tensor, "group")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_equal_zero_scratch_writes_still_use_private_storage(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    value = torch.zeros(8, device=device)
    observed: list[torch.Tensor] = []

    def writes_equal_values(out: torch.Tensor) -> torch.Tensor:
        out.zero_()
        observed.append(out)
        return out

    provider = _provider((value,), writes_equal_values)
    assert provider.mutated_arg_indices == []
    observed.clear()
    cast("Any", provider.kernel).bench_compile_config.return_value = writes_equal_values

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        fn()
        return 1.0

    monkeypatch.setattr(benchmark_provider, "do_bench", bench)
    for _ in range(2):
        assert (
            provider._benchmark_function(
                helion.Config(), cast("Any", writes_equal_values)
            )
            == 1.0
        )
    assert all(out is not value for out in observed)
    # Accuracy and warmup snapshots are distinct for both candidates.
    assert len({out.data_ptr() for out in observed}) == 4
    assert torch.count_nonzero(value) == 0


def test_worker_preserves_duplicate_and_strided_alias_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = torch.arange(128, dtype=torch.float32)
    view = storage.as_strided((3, 4), (6, 1), 2)
    floating = storage[8:72:2]
    broadcast = storage[20:21].expand(3, 4)
    args = (view, floating, view, broadcast)
    path = str(tmp_path / "aliases.pt")
    torch.save(args, path)
    template = cast(
        "tuple[torch.Tensor, ...]", kernel_args.load_trusted_kernel_args(path)
    )
    groups: list[tuple[torch.Tensor, ...]] = []

    def candidate(*values: torch.Tensor) -> None:
        groups.append(values)
        assert values[0] is values[2]
        assert len({v.untyped_storage()._cdata for v in values}) == 1
        assert (
            values[0].untyped_storage()._cdata != template[0].untyped_storage()._cdata
        )
        for expected, actual in zip(template, values, strict=True):
            assert (
                actual.shape,
                actual.stride(),
                actual.storage_offset(),
                actual.dtype,
            ) == (
                expected.shape,
                expected.stride(),
                expected.storage_offset(),
                expected.dtype,
            )
        values[0].zero_()

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        fn()
        return 1.0

    monkeypatch.setattr(
        benchmark_job, "_load_compiled_fn_for_worker", lambda spec: candidate
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "do_bench", bench)
    for _ in range(2):
        benchmark_job.BenchmarkJob(cast("Any", "aliases"), path)()
        for original, untouched in zip(args, template, strict=True):
            assert torch.equal(original, untouched)
    assert (
        groups[0][0].untyped_storage()._cdata != groups[1][0].untyped_storage()._cdata
    )
    kernel_args.load_trusted_kernel_args.cache_clear()


def test_spawn_memory_admission_counts_template_and_private_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider((torch.zeros(8),), lambda x: x.clone())
    provider.settings.autotune_precompile = "spawn"
    provider.settings.autotune_precompile_jobs = 8
    cast("Any", provider.kernel).env.device = torch.device("cuda")
    # (32 template + 32 private + 32 output) * existing 2x safety = 192/job.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (450, 450))
    assert provider._decide_num_jobs() == 2


def test_in_process_mixed_dtype_alias_groups_are_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = torch.arange(128, dtype=torch.uint8)
    view = storage.view(torch.int16).as_strided((3, 4), (6, 1), 2)
    other = storage[8:72].view(torch.float32)[::2]
    args = (view, other, view)
    provider = _provider(args, lambda *values: torch.zeros(1))
    seen: list[tuple[torch.Tensor, ...]] = []

    def candidate(*values: torch.Tensor) -> torch.Tensor:
        assert values[0] is values[2]
        assert values[0].untyped_storage()._cdata == values[1].untyped_storage()._cdata
        assert values[0].untyped_storage()._cdata != storage.untyped_storage()._cdata
        seen.append(values)
        values[0].zero_()
        return torch.zeros(1)

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        fn()
        return 1.0

    monkeypatch.setattr(benchmark_provider, "do_bench", bench)
    cast("Any", provider.kernel).bench_compile_config.return_value = candidate
    for _ in range(2):
        assert (
            provider._benchmark_function(helion.Config(), cast("Any", candidate)) == 1.0
        )
    assert len({values[0].untyped_storage()._cdata for values in seen}) == 4
    assert torch.equal(storage, torch.arange(128, dtype=torch.uint8))


def test_multishape_candidate_and_finalist_calls_use_private_child_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children = [_provider((torch.arange(float(n)), torch.zeros(n * 2))) for n in (2, 4)]
    multi = MultiShapeBenchmarkProvider.__new__(MultiShapeBenchmarkProvider)
    multi.log = Mock()

    def bench(fn: Callable[[], object], **kwargs: object) -> float:
        fn()
        return 1.0

    monkeypatch.setattr(benchmark_provider, "do_bench", bench)
    for i, child in enumerate(children):
        cast("Any", child.kernel).bench_compile_config.return_value = _good

        def benchmark(
            configs: list[helion.Config],
            *,
            desc: str,
            selected_child: LocalBenchmarkProvider = child,
        ) -> list[benchmark_provider.BenchmarkResult]:
            return [
                benchmark_provider.BenchmarkResult(
                    config,
                    cast("Any", _good),
                    selected_child._benchmark_function(config, cast("Any", _good)),
                    "ok",
                    None,
                )
                for config in configs
            ]

        monkeypatch.setattr(child, "benchmark", benchmark)
        for _ in range(2):
            results = multi._benchmark_child(
                child, [helion.Config()], desc="candidate or finalist", case_index=i
            )
            assert results[0].perf == 1.0
        assert torch.equal(
            cast("torch.Tensor", child.args[0]), torch.arange(float((i + 1) * 2))
        )
        assert torch.count_nonzero(cast("torch.Tensor", child.args[1])) == 0
