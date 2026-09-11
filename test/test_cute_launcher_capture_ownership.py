from __future__ import annotations

import gc
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import torch

from helion._testing import skipIfNotCUDA
from helion.runtime.cute import launcher


def _launch_entry(storage: torch.Tensor) -> launcher._CuteLaunchArgCacheEntry:
    return launcher._CuteLaunchArgCacheEntry(
        schema=(),
        launch_args=(),
        grouped_static_metadata=(),
        owned_tensors=(storage,),
    )


def test_managed_capture_retains_ungrouped_descriptor_through_lru_eviction() -> None:
    cute_kernel = SimpleNamespace()
    resources = launcher._CuteCudaGraphResources({}, {}, set(), set())
    graph = SimpleNamespace(_helion_resources=resources)
    first_ref: weakref.ReferenceType[torch.Tensor] | None = None
    build_count = 0

    def build(
        _kernel: object,
        _args: tuple[object, ...],
        _grid: tuple[int, int, int],
    ) -> launcher._CuteLaunchArgCacheEntry:
        nonlocal build_count, first_ref
        build_count += 1
        storage = torch.empty(128, dtype=torch.uint8)
        if build_count == 1:
            first_ref = weakref.ref(storage)
        return _launch_entry(storage)

    token = launcher._CUTE_ACTIVE_CUDA_GRAPH.set(graph)  # pyrefly: ignore
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
        launcher._CUTE_ACTIVE_CUDA_GRAPH.reset(token)

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
    graph = launcher._CuteCUDAGraph.__new__(launcher._CuteCUDAGraph)
    resources = launcher._CuteCudaGraphResources({}, {}, set(), set())
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
        assert resources == launcher._CuteCudaGraphResources({}, {}, set(), set())

        second_ref = retain("second")
        graph.reset()
        assert second_ref() is None
        assert owner.cache == {}
        assert resources == launcher._CuteCudaGraphResources({}, {}, set(), set())

    assert reset.call_count == 2
