"""Exchange temporary ownership: real CPU buffers, mocked enqueue/graph context only."""

from __future__ import annotations

from contextlib import contextmanager
import gc
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest
import torch

from helion._testing.cute_codegen import forbid_cuda_work
from helion.runtime.cute import launcher

KINDS = ("helion_tcgen05_split_k", "helion_mma_dag_fixup")


def _kernel(kind):
    plan = {"kind": kind, "param": "exchange"}
    if kind == "helion_tcgen05_split_k":
        plan.update(tiles=2, splits=4, tile_elems=8)
    else:
        plan.update(tiles=2, rounds=4, slabs=2, rows=8)
    return SimpleNamespace(_helion_cute_wrapper_plans=(plan,))


@contextmanager
def _cpu_launch(context):
    with (
        forbid_cuda_work(),
        patch.object(
            launcher,
            "_cuda_stream_capture_context",
            side_effect=lambda device: context[0],
        ),
        patch.object(launcher, "_validate_cute_launcher_tensor"),
        patch.object(
            launcher,
            "_get_cute_launcher_imports",
            return_value=(
                "gmem",
                lambda dtype, pointer, space, **kwargs: pointer,
                lambda: "stream",
            ),
        ),
        patch.object(
            torch.cuda,
            "current_stream",
            side_effect=lambda device=None: "stream-" + str(context[0][0]),
        ),
        patch.object(
            torch.cuda,
            "is_current_stream_capturing",
            side_effect=lambda: context[0][1] is not None,
        ),
        patch.object(torch.Tensor, "record_stream", autospec=True) as recorded,
    ):
        yield recorded


@contextmanager
def _owned_graph():
    # A CPU analogue of the existing private capture-owner scope/resources;
    # production ownership is supplied only by cute_cuda_graph itself.
    resources = launcher._CuteCudaGraphResources({}, {}, set(), set())
    graph = SimpleNamespace(_helion_resources=resources)
    token = launcher._CUTE_ACTIVE_CUDA_GRAPH.set(graph)
    try:
        yield resources
    finally:
        launcher._CUTE_ACTIVE_CUDA_GRAPH.reset(token)


def _workspace(kernel, kind):
    if kind == "helion_tcgen05_split_k":
        return launcher._tcgen05_split_k_workspace(
            kernel, device=torch.device("cpu"), tiles=2, splits=4, tile_elems=8
        )
    return launcher._mma_dag_fixup_workspace(
        kernel,
        device=torch.device("cpu"),
        row_tiles=4,
        tiles=2,
        rounds=4,
        slabs=2,
        rows=8,
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("capture_id", (None, 111))
def test_unowned_exchange_is_per_call_and_not_kept_in_any_pointer_cache(
    kind, capture_id
):
    kernel = _kernel(kind)
    arg = torch.empty((2, 4), dtype=torch.float32)
    context = [(31, capture_id)]
    with _cpu_launch(context):
        first = launcher._build_cached_cute_schema_and_args(kernel, (arg,), (8, 1, 1))
        first_tensor = first.owned_tensors[0]
        ref = weakref.ref(first_tensor)
        second = launcher._build_cached_cute_schema_and_args(kernel, (arg,), (8, 1, 1))
        assert second.owned_tensors[0] is not first_tensor
        assert second.launch_args[-4] != first.launch_args[-4]
        assert not hasattr(kernel, "_helion_tcgen05_split_k_workspace_cache")
        assert not hasattr(kernel, "_helion_mma_dag_fixup_workspace_cache")
        assert not hasattr(kernel, "_helion_cute_launch_arg_cache")
        assert not hasattr(kernel, "_helion_cute_capture_owned_launch_tensors")
        launcher._set_cute_last_launch_cache_entry(
            kernel, (arg,), (8, 1, 1), (128, 1, 1), None, first, object()
        )
        assert not hasattr(kernel, "_helion_cute_last_launch_cache")
        assert (
            launcher._cute_last_launch_cache_entry(
                kernel, (arg,), (8, 1, 1), (128, 1, 1), None
            )
            is None
        )
        del first, first_tensor
        gc.collect()
        assert ref() is None


@pytest.mark.parametrize("kind", KINDS)
def test_owned_scope_without_active_capture_still_uses_temporary(kind):
    kernel = _kernel(kind)
    arg = torch.empty(4)
    with _cpu_launch([(31, None)]), _owned_graph() as resources:
        launch = launcher._build_cached_cute_schema_and_args(kernel, (arg,), (8, 1, 1))
        assert not launcher._cute_owned_exchange_capture(kernel, (arg,))
        assert not resources.tensors
        assert not hasattr(kernel, "_helion_cute_launch_arg_cache")
        assert launch.owned_tensors


@pytest.mark.parametrize("kind", KINDS)
def test_owned_capture_tracks_every_strong_workspace_owner_and_release(kind):
    kernel = _kernel(kind)
    arg = torch.empty(4)
    context = [(31, 101)]
    with _cpu_launch(context), _owned_graph() as resources:
        first = launcher._build_cached_cute_schema_and_args(kernel, (arg,), (8, 1, 1))
        ref = weakref.ref(first.owned_tensors[0])
        assert (
            launcher._build_cached_cute_schema_and_args(kernel, (arg,), (8, 1, 1))
            is first
        )
        assert any(
            key[1] == "_helion_cute_launch_arg_cache" for key in resources.cache_entries
        )
        assert any("workspace_cache" in key[1] for key in resources.cache_entries)
        assert any(
            key[1] == "_helion_cute_capture_owned_launch_tensors"
            for key in resources.cache_entries
        )
        del first
        gc.collect()
        assert ref() is not None
        resources.release()
        gc.collect()
        assert ref() is None
        assert kernel._helion_cute_launch_arg_cache == {}
        assert kernel._helion_cute_capture_owned_launch_tensors == {}


@pytest.mark.parametrize("kind", KINDS)
def test_graph_retains_evicted_workspace_without_releasing_another_graph(kind):
    kernel = _kernel(kind)
    arg = torch.empty(4)
    context = [(31, 101)]
    with (
        _cpu_launch(context),
        patch.object(launcher, "_CUTE_LAUNCH_ARG_CACHE_LIMIT", 1),
        patch.object(launcher, "_TCGEN05_SPLIT_K_WORKSPACE_CACHE_LIMIT", 1),
        patch.object(launcher, "_MMA_DAG_FIXUP_WORKSPACE_CACHE_LIMIT", 1),
    ):
        with _owned_graph() as first_resources:
            first = launcher._build_cached_cute_schema_and_args(
                kernel, (arg,), (8, 1, 1)
            )
            first_ref = weakref.ref(first.owned_tensors[0])
        context[0] = (32, 102)
        with _owned_graph() as second_resources:
            second = launcher._build_cached_cute_schema_and_args(
                kernel, (arg,), (8, 1, 1)
            )
            second_ref = weakref.ref(second.owned_tensors[0])
            assert second.owned_tensors[0] is not first.owned_tensors[0]
        del first, second
        gc.collect()
        assert first_ref() is not None and second_ref() is not None
        first_resources.release()
        gc.collect()
        assert first_ref() is None
        assert second_ref() is not None
        assert kernel._helion_cute_launch_arg_cache
        second_resources.release()
        gc.collect()
        assert second_ref() is None


@pytest.mark.parametrize("kind", KINDS)
def test_both_exchange_cache_keys_follow_stream_and_capture_changes(kind):
    kernel = _kernel(kind)
    arg = torch.empty(4)
    context = [(31, 101)]
    with _cpu_launch(context):
        first = launcher._cute_launch_arg_cache_key(kernel, (arg,), (8, 1, 1))
        first_context = launcher._cute_grouped_launch_contexts(kernel, (arg,))
        context[0] = (32, 101)
        second = launcher._cute_launch_arg_cache_key(kernel, (arg,), (8, 1, 1))
        assert second != first
        assert launcher._cute_grouped_launch_contexts(kernel, (arg,)) != first_context
        context[0] = (32, 102)
        assert launcher._cute_launch_arg_cache_key(kernel, (arg,), (8, 1, 1)) != second


@pytest.mark.parametrize("kind", KINDS)
def test_temporary_counter_initialization_is_fresh_but_owned_capture_reuses(kind):
    kernel = _kernel(kind)
    context = [(31, None)]
    with _cpu_launch(context):
        first = _workspace(kernel, kind)
        first.fill_(29)
        second = _workspace(kernel, kind)
        assert second is not first
        assert second[-256:].count_nonzero().item() == 0
        context[0] = (31, 101)
        with _owned_graph() as resources:
            owned = _workspace(kernel, kind)
            owned.fill_(37)
            assert _workspace(kernel, kind) is owned
            assert owned[-256:].eq(37).all().item()
            resources.release()


@pytest.mark.parametrize("kind", KINDS)
def test_local_keepalive_covers_stream_record_and_full_producer_reducer_enqueue(kind):
    kernel = _kernel(kind)
    arg = torch.empty(4)
    context = [(31, None)]
    pointers, refs, stages = [], [], []
    build = launcher._build_cute_schema_and_args

    def capture_build(*args):
        result = build(*args)
        refs.append(weakref.ref(result.owned_tensors[0]))
        return result

    def compiled(*args):
        assert refs[-1]() is not None
        pointers.append(args[-5])
        stages.append("producer")
        assert refs[-1]() is not None
        stages.append("reducer")

    with (
        _cpu_launch(context) as recorded,
        patch.object(
            launcher, "_build_cute_schema_and_args", side_effect=capture_build
        ),
        patch.object(launcher, "_get_compiled_cute_launcher", return_value=compiled),
    ):
        for stream in (31, 32):
            context[0] = (stream, None)
            launcher._launch_cute_marshalled(
                kernel, (arg,), (8, 1, 1), (128, 1, 1), None
            )
            assert recorded.call_args.args[-1] == "stream-" + str(stream)
            gc.collect()
            assert refs[-1]() is None
        assert stages == ["producer", "reducer", "producer", "reducer"]
        assert not hasattr(kernel, "_helion_cute_last_launch_cache")
        assert getattr(kernel, "_helion_cute_fastpath", False) is False
        assert not hasattr(kernel, "_helion_cute_launch_arg_cache")
