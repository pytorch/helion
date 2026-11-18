from __future__ import annotations

import contextlib
import gc
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch
import weakref

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import is_cuda
from helion._testing import skipIfNotCUDA
from helion._testing.cute_zero_fill import COLUMN_SUM_CONFIG
from helion._testing.cute_zero_fill import atomic_column_sum
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn
import helion.language as hl
from helion.runtime.cute import compile_cache
from helion.runtime.cute import launch_caches
from helion.runtime.cute import launcher

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path

    from helion.autotuner.precompile_future import SerializedCompiledFunction


def _launch_entry(storage: torch.Tensor) -> launcher._CuteLaunchArgCacheEntry:
    return launcher._CuteLaunchArgCacheEntry(
        schema=(),
        launch_args=(),
        grouped_static_metadata=(),
        owned_tensors=(storage,),
    )


def test_managed_capture_retains_ungrouped_descriptor_through_lru_eviction() -> None:
    cute_kernel = SimpleNamespace()
    resources = launch_caches.CudaGraphResources({}, {}, set(), set())
    graph = SimpleNamespace(_helion_resources=resources)
    first_ref: weakref.ReferenceType[torch.Tensor] | None = None
    build_count = 0

    def build(
        _kernel: object,
        _args: tuple[object, ...],
        _grid: tuple[int, int, int],
        **_kwargs: object,
    ) -> launcher._CuteLaunchArgCacheEntry:
        nonlocal build_count, first_ref
        build_count += 1
        storage = torch.empty(128, dtype=torch.uint8)
        if build_count == 1:
            first_ref = weakref.ref(storage)
        return _launch_entry(storage)

    token = launch_caches.ACTIVE_CUDA_GRAPH.set(graph)  # pyrefly: ignore
    try:
        with (
            patch.object(
                launcher,
                "_cuda_stream_capture_context",
                return_value=(101, 202),
            ),
            patch.object(launcher, "_cute_kernel_param_is_constexpr", return_value=()),
            patch.object(launcher, "_build_cute_schema_and_args", side_effect=build),
            patch.object(torch.cuda, "current_stream", return_value="capture-stream"),
            patch.object(torch.Tensor, "record_stream", autospec=True),
        ):
            first = launcher._build_cached_cute_schema_and_args(
                cute_kernel, (), (1, 1, 1)
            )
    finally:
        launch_caches.ACTIVE_CUDA_GRAPH.reset(token)

    capture_cache = cute_kernel._helion_cute_capture_owned_launch_tensors
    assert len(capture_cache) == 1
    assert first_ref is not None
    assert first_ref() is first.owned_tensors[0]
    del first

    with (
        patch.object(
            launcher,
            "_cuda_stream_capture_context",
            return_value=(101, None),
        ),
        patch.object(launcher, "_cute_kernel_param_is_constexpr", return_value=()),
        patch.object(launcher, "_build_cute_schema_and_args", side_effect=build),
    ):
        for grid_x in range(2, 10):
            launcher._build_cached_cute_schema_and_args(cute_kernel, (), (grid_x, 1, 1))

    assert len(cute_kernel._helion_cute_launch_arg_cache) == 8
    gc.collect()
    assert first_ref() is not None

    resources.release()
    gc.collect()
    assert capture_cache == {}
    assert first_ref() is None


def test_raw_capture_conservatively_retains_ungrouped_descriptor() -> None:
    cute_kernel = SimpleNamespace()
    storage = torch.empty(128, dtype=torch.uint8)
    storage_ref = weakref.ref(storage)

    with patch.object(
        launcher,
        "_cuda_stream_capture_context",
        return_value=(303, 404),
    ):
        launcher._retain_cute_capture_owned_launch_tensors(
            cute_kernel,
            grouped_launch_contexts=(),
            owned_tensors=(storage,),
        )

    capture_cache = cute_kernel._helion_cute_capture_owned_launch_tensors
    del storage
    gc.collect()
    assert storage_ref() is not None
    capture_cache.clear()
    gc.collect()
    assert storage_ref() is None


def test_last_launch_fast_hit_promotes_owned_tensors_to_capture() -> None:
    cute_kernel = SimpleNamespace()
    storage = torch.empty(128, dtype=torch.uint8)
    launch = _launch_entry(storage)
    last_launch = SimpleNamespace(
        arg_guard=SimpleNamespace(grouped_launch_contexts=()),
        launch=launch,
        compiled=lambda *args: args,
    )

    with (
        patch.object(
            launcher, "_cute_last_launch_cache_entry", return_value=last_launch
        ),
        patch.object(launcher, "_retain_cute_capture_owned_launch_tensors") as retain,
        patch.object(launcher, "_record_cute_owned_launch_tensors") as record,
        patch.object(launcher, "_cute_current_stream", return_value="stream"),
    ):
        result = launcher.default_cute_launcher(cute_kernel, (1,), 7)

    assert result == ("stream",)
    retain.assert_called_once_with(
        cute_kernel,
        grouped_launch_contexts=(),
        owned_tensors=(storage,),
    )
    record.assert_called_once_with((storage,))


@skipIfNotCUDA()
def test_managed_graph_reset_releases_resources_before_recapture() -> None:
    graph = launch_caches.CuteCUDAGraph.__new__(launch_caches.CuteCUDAGraph)
    resources = launch_caches.CudaGraphResources({}, {}, set(), set())
    graph._helion_resources = resources
    owner = SimpleNamespace(cache={})

    def retain(key: str) -> weakref.ReferenceType[torch.Tensor]:
        value = object()
        owner.cache[key] = value
        resources.cache_entries[(id(owner), "cache", key)] = (
            owner,
            "cache",
            key,
            value,
        )
        tensor = torch.empty(1)
        resources.tensors[id(tensor)] = tensor
        return weakref.ref(tensor)

    def parent_reset(_graph: object) -> None:
        assert resources.cache_entries
        assert resources.tensors

    with patch.object(
        torch.cuda.CUDAGraph,
        "reset",
        autospec=True,
        side_effect=parent_reset,
    ) as reset:
        first_ref = retain("first")
        graph.reset()
        assert first_ref() is None
        assert owner.cache == {}
        assert resources == launch_caches.CudaGraphResources({}, {}, set(), set())

        second_ref = retain("second")
        graph.reset()
        assert second_ref() is None
        assert owner.cache == {}
        assert resources == launch_caches.CudaGraphResources({}, {}, set(), set())

    assert reset.call_count == 2


@helion.kernel(static_shapes=True, autotune_effort="none")
def _captured_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


_Candidate = tuple["SerializedCompiledFunction", torch.Tensor, torch.Tensor]


@pytest.fixture(scope="module")
def compile_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """A fresh on-disk compile cache for the module's candidates: every copy
    loaded from a candidate's wrapper is then a reload (the DSL keeps a
    fresh compile alive for the life of the process)."""
    if not is_cuda():
        pytest.skip("CUDA graph capture requires CUDA")
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv(
            "CUTE_DSL_CACHE_DIR", str(tmp_path_factory.mktemp("cute-dsl-cache"))
        )
        yield


@pytest.fixture(scope="module")
def candidate(compile_cache_dir: None) -> _Candidate:
    """A kernel as the benchmark worker gets a candidate: its serialized
    wrapper, with the compiled module in the on-disk compile cache, and its
    arguments."""
    x = torch.randn(64, 64, device=DEVICE)
    y = torch.randn(64, 64, device=DEVICE)
    bound = _captured_add.bind((x, y))
    fn = bound.compile_config(bound.config_spec.default_config())
    fn(x, y)
    torch.cuda.synchronize()
    return _serialize_compiled_fn(fn), x, y


@pytest.fixture(scope="module")
def zero_fill_candidate(
    compile_cache_dir: None,
) -> tuple[SerializedCompiledFunction, torch.Tensor]:
    """A ``cute_pdl`` candidate whose host zero fill launches a second
    kernel of the same generated module."""
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEVICE)
    bound = atomic_column_sum.bind((x,))
    fn = bound.compile_config(
        bound.config_spec.normalized_config(
            helion.Config.from_dict({**COLUMN_SUM_CONFIG, "cute_pdl": True})
        )
    )
    fn(x)
    torch.cuda.synchronize()
    return _serialize_compiled_fn(fn), x


@contextlib.contextmanager
def _collector_off() -> Iterator[None]:
    """Nothing is collected until the test collects: a dropped candidate
    waits, as in the worker, for a collection inside the next capture."""
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _library_modules(kernel: object) -> list[weakref.ReferenceType[object]]:
    """The DSL modules that hold the CUDA libraries ``kernel``'s compiled
    launchers loaded."""
    return [
        weakref.ref(compiled.jit_module)
        for compiled_launcher in getattr(
            kernel, "_helion_cute_compiled_launchers", {}
        ).values()
        for compiled in (compiled_launcher._compiled, compiled_launcher._capi_compiled)
        if getattr(compiled, "jit_module", None) is not None
    ]


def _launched_copy(
    spec: SerializedCompiledFunction, *args: torch.Tensor
) -> tuple[Callable[..., torch.Tensor], list[weakref.ReferenceType[object]]]:
    """A copy of the candidate loaded and launched once, as a worker job
    does, and the DSL modules that hold the CUDA libraries it loaded, for
    every kernel of the generated module."""
    fn = _load_compiled_fn(spec)
    fn(*args)
    torch.cuda.synchronize()
    kernel = fn.__globals__[f"_helion_{fn.__name__}"]
    for compiled_launcher in kernel._helion_cute_compiled_launchers.values():
        # Reloaded from the on-disk cache, not compiled: a fresh compile is a
        # TVM-FFI function, which the DSL keeps alive, with a C-API twin.
        assert not compile_cache.is_tvm_ffi_compiled_function(
            compiled_launcher._compiled
        )
        assert compiled_launcher._capi_compiled is None
        assert compiled_launcher._launch_devices == {torch.cuda.current_device()}
    modules = [
        module
        for name, value in fn.__globals__.items()
        if name.startswith("_helion_")
        for module in _library_modules(value)
    ]
    assert _library_modules(kernel)
    assert all(type(module()).__name__ == "CudaDialectJitModule" for module in modules)
    assert _holding_libraries(modules) == [True] * len(modules)
    return fn, modules


def _holding_libraries(modules: list[weakref.ReferenceType[object]]) -> list[bool]:
    return [
        bool(module.cuda_library) and not module.is_unloaded()  # type: ignore[attr-defined]
        for module in [reference() for reference in modules]
    ]


@contextlib.contextmanager
def _capture(kind: str) -> Iterator[torch.cuda.CUDAGraph]:
    if kind == "cute_cuda_graph":
        with launch_caches.cute_cuda_graph() as managed:
            yield managed
    else:
        raw = torch.cuda.CUDAGraph()
        with torch.cuda.graph(raw):
            yield raw


def _capture_over_collection(
    candidate: _Candidate, kind: str, garbage: list[weakref.ReferenceType[object]]
) -> list[bool]:
    """The worker's next job: load the candidate again, warm it and capture
    two launches with a collection between them, which finalizes ``garbage``
    (the modules of a dropped copy) inside the capture; then replay. Returns
    which of ``garbage`` held a loaded library when the collection ran."""
    spec, x, y = candidate
    live, _modules = _launched_copy(spec, x, y)
    graph = None
    try:
        with _capture(kind) as graph:
            first = live(x, y)
            held = _holding_libraries(garbage)
            gc.collect()
            assert [module() for module in garbage] == [None] * len(garbage)
            second = live(x, y)
        x.add_(1.0)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(first, x + y)
        torch.testing.assert_close(second, x + y)
    finally:
        if graph is not None:
            graph.reset()
        _unload_compiled_fn(live)
    return held


_CAPTURES = ("cute_cuda_graph", "torch.cuda.graph")


@skipIfNotCUDA()
@pytest.mark.parametrize("capture", _CAPTURES)
def test_capture_survives_collecting_a_candidate_unloaded_at_job_end(
    candidate: _Candidate, capture: str
) -> None:
    spec, x, y = candidate
    with _collector_off():
        dead, modules = _launched_copy(spec, x, y)
        # The end of a benchmark-worker job: the libraries are unloaded here,
        # and the modules are garbage only the collector frees.
        _unload_compiled_fn(dead)
        del dead
        held = _capture_over_collection(candidate, capture, modules)
    assert held == [False] * len(modules)


@skipIfNotCUDA()
@pytest.mark.parametrize("capture", _CAPTURES)
def test_capture_survives_finalizing_a_candidate_that_holds_its_library(
    candidate: _Candidate, capture: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy dropped without the worker's teardown still holds its library
    when the collector finalizes it inside the capture: the finalizer leaves
    the library loaded instead of asking the driver to unload it there."""
    spec, x, y = candidate
    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    with _collector_off():
        dead, modules = _launched_copy(spec, x, y)
        sys.modules.pop(dead.__module__)
        del dead
        held = _capture_over_collection(candidate, capture, modules)
    assert held == [True] * len(modules)
    assert compile_cache.skipped_capture_unloads() == len(modules)


@skipIfNotCUDA()
def test_job_end_unload_reaches_the_zero_fill_kernel(
    zero_fill_candidate: tuple[SerializedCompiledFunction, torch.Tensor],
) -> None:
    """The host zero fill of a ``cute_pdl`` candidate launches a second
    kernel of its generated module, which loads a library of its own: the
    end of the job unloads it with the main kernel's, so that neither waits
    for the collector."""
    spec, x = zero_fill_candidate
    with _collector_off():
        fn, modules = _launched_copy(spec, x)
        fill = fn.__globals__[f"_helion_zero_fill_0_helion_{fn.__name__}"]
        assert _library_modules(fill)
        _unload_compiled_fn(fn)
        assert _holding_libraries(modules) == [False] * len(modules)


def _modules_holding_libraries() -> int:
    """How many of the DSL's modules in this process hold a loaded library."""
    return sum(
        1
        for item in gc.get_objects()
        if type(item).__name__ == "CudaDialectJitModule"
        and item.cuda_library
        and not item.is_unloaded()
    )


@skipIfNotCUDA()
def test_benchmark_job_leaves_no_loaded_library_behind(
    candidate: _Candidate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark worker's job, run here: the library its candidate loaded
    is unloaded when the job ends, so none waits for the collector to unload
    it inside the next job's capture."""
    spec, x, y = candidate
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    torch.save((x, y), tmp_path / "args.pt")
    job = BenchmarkJob(
        fn_spec=spec,
        args_path=str(tmp_path / "args.pt"),
        fixed_repetitions=2,
        l2_flush_mode="off",
        input_banks=2,
    )
    with _collector_off():
        loaded = _modules_holding_libraries()
        for _ in range(2):
            assert job() > 0
            assert _modules_holding_libraries() == loaded
