from __future__ import annotations

from types import SimpleNamespace
import weakref

import pytest
import torch

from helion import exc
from helion.runtime.cute import launcher
from helion.runtime.cute.chunk_prefill import prefill_resources
from helion.runtime.cute.chunk_prefill import validate_args


def _plan() -> dict[str, object]:
    return {
        "kind": "chunk_prefill_sm100",
        "device_abi": 1,
        "threads": 512,
        "heads": 1,
        "sequences": 2,
        "total_tokens": 16,
        "task_order": "identity",
        "schedule": "prefix_tail_2",
        "prefix_count": 2,
        "sequence_groups": 2,
        **{
            f"{name}_idx": index
            for index, name in enumerate(
                (
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "a_log",
                    "dt",
                    "initial_state",
                    "cu_seqlens",
                    "out",
                    "final_state",
                )
            )
        },
    }


def _args() -> tuple[torch.Tensor, ...]:
    q = torch.empty((16, 128), dtype=torch.bfloat16)
    state = torch.empty((2, 1, 128, 128), dtype=torch.float32)
    return (
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty((16,), dtype=torch.bfloat16),
        torch.empty((1,), dtype=torch.float32),
        torch.empty((1, 128), dtype=torch.float32),
        state,
        torch.tensor([0, 0, 16], dtype=torch.int64),
        torch.empty_like(q),
        torch.empty_like(state),
    )


def test_prefill_segmented_abi_rejects_state_rounding_and_aliases() -> None:
    args = _args()
    validate_args(_plan(), args)
    for index, replacement in ((7, args[7].bfloat16()), (10, args[7]), (9, args[0])):
        invalid = list(args)
        invalid[index] = replacement
        with pytest.raises(exc.BackendUnsupported):
            validate_args(_plan(), invalid)


def test_prefill_resources_isolate_streams_and_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, args = _plan(), _args()
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[plan])
    context = [17, None]
    retained = []
    monkeypatch.setattr(
        launcher, "_cuda_stream_capture_context", lambda device: tuple(context)
    )
    monkeypatch.setattr(launcher, "_validate_cute_launcher_tensor", lambda tensor: None)
    monkeypatch.setattr(
        launcher,
        "_track_cute_cuda_graph_cache_entry",
        lambda *args: retained.append(args),
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "Event", object)

    first = prefill_resources(kernel, plan, args)
    assert prefill_resources(kernel, plan, args) is first
    first_key = launcher._cute_launch_arg_cache_key(kernel, args, (1, 2, 1))
    guard = launcher._cute_last_launch_arg_guard(kernel, args, (1, 2, 1))
    assert guard.matches(kernel, args, (1, 2, 1))
    for stream, capture in ((18, None), (17, 31), (17, 32)):
        context[:] = stream, capture
        other = prefill_resources(kernel, plan, args)
        assert other is not first
        assert all(
            a.data_ptr() != b.data_ptr()
            for a, b in zip(first.states, other.states, strict=True)
        )
        assert other.streams != first.streams
        assert launcher._cute_launch_arg_cache_key(kernel, args, (1, 2, 1)) != first_key
        assert not guard.matches(kernel, args, (1, 2, 1))
    capture_resource = retained[-1][3]
    assert capture_resource is other
    assert retained[-1][4] == other.states
    # Eager cache churn cannot discard a captured stream/event owner.
    for stream in range(100, 112):
        context[:] = stream, None
        prefill_resources(kernel, plan, args)
    assert any(
        resource is capture_resource
        for resource in kernel._helion_cute_prefill_resources.values()
    )
    eager = [key for key in kernel._helion_cute_prefill_resources if key[2] is None]
    assert len(eager) == 8


def test_prefill_stream_fork_join_samples_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion.runtime.cute.chunk_prefill import PrefillResources

    events = []

    class Event:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream: Stream) -> None:
            events.append(("record", self.name, stream.name))

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event: Event) -> None:
            events.append(("wait", self.name, event.name))

    origin = Stream("origin-a")
    workers = (Stream("worker0"), Stream("worker1"))
    resources = PrefillResources(
        (torch.empty(1), torch.empty(1)),
        workers,
        Event("ready"),
        (Event("done0"), Event("done1")),
    )
    entry = launcher._CuteLaunchArgCacheEntry((), (), (), resources.states, resources)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: origin)
    monkeypatch.setattr(launcher, "_cute_current_stream", lambda: origin.name)

    def compiled(stream: str) -> None:
        events.append(("launch", stream))

    launcher._launch_cute_entry(compiled, entry)
    assert events == [
        ("record", "ready", "origin-a"),
        ("wait", "worker0", "ready"),
        ("wait", "worker1", "ready"),
        ("launch", "origin-a"),
        ("record", "done0", "worker0"),
        ("wait", "origin-a", "done0"),
        ("record", "done1", "worker1"),
        ("wait", "origin-a", "done1"),
    ]
    origin = Stream("origin-b")
    events.clear()
    launcher._launch_cute_entry(compiled, entry)
    assert ("launch", "origin-b") in events
    assert ("wait", "origin-b", "done1") in events


def test_prefill_managed_capture_retains_stream_owners_until_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion.runtime.cute.chunk_prefill import PrefillResources

    class StreamOrEvent:
        pass

    origin = StreamOrEvent()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: origin)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    retained = launcher._CuteCudaGraphResources({}, {}, set(), set())
    kernel = SimpleNamespace(
        _helion_cute_prefill_resources={},
        _helion_cute_launch_arg_cache={},
        _helion_cute_last_launch_cache=None,
    )

    def populate():
        resource = PrefillResources(
            (torch.empty(1), torch.empty(1)),
            (StreamOrEvent(),),
            StreamOrEvent(),
            (StreamOrEvent(),),
            order=torch.empty(2, dtype=torch.int32),
        )
        kernel._helion_cute_prefill_resources["capture"] = resource
        entry = launcher._CuteLaunchArgCacheEntry((), (), (), resource.states, resource)
        kernel._helion_cute_launch_arg_cache["launch"] = entry
        kernel._helion_cute_last_launch_cache = entry
        retained.retain(
            kernel,
            "_helion_cute_prefill_resources",
            "capture",
            resource,
            resource.tensors,
        )
        return (
            weakref.ref(resource),
            weakref.ref(resource.streams[0]),
            weakref.ref(resource.ready),
            weakref.ref(resource.order),
        )

    refs = populate()
    kernel._helion_cute_prefill_resources.clear()
    kernel._helion_cute_launch_arg_cache.clear()
    kernel._helion_cute_last_launch_cache = None
    assert all(reference() is not None for reference in refs)
    retained.release()
    assert all(reference() is None for reference in refs)


def test_prefill_precompute_single_owns_only_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute")
    plan, args = _plan(), _args()
    plan.update(
        schedule="single", task_order="longest_first_precompute", prefix_count=0
    )
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[plan])
    context = [17, None]
    monkeypatch.setattr(
        launcher, "_cuda_stream_capture_context", lambda device: tuple(context)
    )
    monkeypatch.setattr(launcher, "_validate_cute_launcher_tensor", lambda tensor: None)
    resource = prefill_resources(kernel, plan, args)
    assert resource.states == () and resource.streams == ()
    assert resource.ready is None and resource.done == ()
    assert resource.order is not None and resource.order.dtype is torch.int32
    assert resource.order.shape == (2,)
    assert resource.tensors == (resource.order,)
    assert resource.order_kernel is not None
    key = launcher._cute_launch_arg_cache_key(kernel, args, (1, 2, 1))
    guard = launcher._cute_last_launch_arg_guard(kernel, args, (1, 2, 1))
    context[0] = 18
    other = prefill_resources(kernel, plan, args)
    assert other.order.data_ptr() != resource.order.data_ptr()
    assert launcher._cute_launch_arg_cache_key(kernel, args, (1, 2, 1)) != key
    assert not guard.matches(kernel, args, (1, 2, 1))


def test_prefill_sort_is_submitted_before_worker_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion.runtime.cute.chunk_prefill import PrefillResources

    pytest.importorskip("cutlass.cute")
    calls = []
    order = torch.empty(2, dtype=torch.int32)
    resource = PrefillResources(
        (torch.empty(1), torch.empty(1)),
        (),
        None,
        (),
        order=order,
        order_kernel=object(),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "origin")
    monkeypatch.setattr(launcher, "_cute_current_stream", lambda: "origin")
    monkeypatch.setattr(
        launcher, "default_cute_launcher", lambda *args, **kwargs: calls.append("sort")
    )
    monkeypatch.setattr(resource, "fork", lambda origin: calls.append("fork"))
    monkeypatch.setattr(resource, "join", lambda origin: calls.append("join"))
    entry = launcher._CuteLaunchArgCacheEntry(
        (),
        (),
        (),
        resource.tensors,
        resource,
        (torch.empty(3, dtype=torch.int64), order),
    )
    launcher._launch_cute_entry(lambda stream: calls.append("prefill"), entry)
    assert calls == ["sort", "fork", "prefill", "join"]
