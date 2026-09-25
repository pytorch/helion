from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
import weakref

import pytest
import torch

from helion import exc
from helion.runtime.cute import launcher
from helion.runtime.cute.chunk_prefill import append_host_call
from helion.runtime.cute.chunk_prefill import get_host
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


def _bt32_plan(task_order: str = "identity", *, heads: int = 8) -> dict[str, object]:
    return {
        "kind": "chunk_prefill_sm100",
        "device_abi": 2,
        "threads": 1024,
        "chunk_size": 32,
        "numerical_policy": "centered_bt32_fp32_rhs_v2",
        "heads": heads,
        "sequences": 2,
        "total_tokens": 16,
        "task_order": task_order,
        "schedule": "single",
        "prefix_count": 0,
        "sequence_groups": 1,
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
                    "scale",
                    "gate_scale",
                )
            )
        },
    }


def _bt32_args(*, heads: int = 8) -> tuple[object, ...]:
    tokens, sequences = 16, 2
    rows = tokens * heads
    activation = torch.empty((rows, 128), dtype=torch.bfloat16)
    state = torch.empty((sequences, heads, 128, 128), dtype=torch.float32)
    return (
        activation,
        torch.empty_like(activation),
        torch.empty_like(activation),
        torch.empty_like(activation),
        torch.empty((rows,), dtype=torch.bfloat16),
        torch.empty((heads,), dtype=torch.float32),
        torch.empty((heads, 128), dtype=torch.float32),
        state,
        torch.tensor([0, 0, tokens], dtype=torch.int64),
        torch.empty_like(activation),
        torch.empty_like(state),
        128**-0.5,
        -5 * 1.4426950408889634,
    )


def test_prefill_segmented_abi_rejects_state_rounding_and_aliases() -> None:
    args = _args()
    validate_args(_plan(), args)
    for index, replacement in ((7, args[7].bfloat16()), (10, args[7]), (9, args[0])):
        invalid = list(args)
        invalid[index] = replacement
        with pytest.raises(exc.BackendUnsupported):
            validate_args(_plan(), invalid)


@pytest.mark.parametrize("task_order", ["identity", "longest_first_precompute"])
def test_bt32_plan_and_host_resolution(task_order: str) -> None:
    pytest.importorskip("cutlass.cute")
    plan = _bt32_plan(task_order)
    validate_args(plan, _bt32_args())
    assert (
        get_host(plan).__module__ == "helion._compiler.cute.chunk_prefill_bt32.device"
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("device_abi", 1),
        ("device_abi", 3),
        ("threads", 512),
        ("chunk_size", 16),
        ("numerical_policy", "native_bt16_bf16_rhs_v1"),
        ("task_order", "longest_first"),
        ("schedule", "prefix_tail_2"),
    ],
)
def test_bt32_rejects_forged_plan(key: str, value: object) -> None:
    plan = _bt32_plan() | {key: value}
    with pytest.raises(exc.BackendUnsupported):
        append_host_call([], plan)


@pytest.mark.parametrize(
    "invalid",
    ["int32_cu", "heads", "noncontiguous", "misaligned", "output_alias"],
)
def test_bt32_argument_guards(invalid: str) -> None:
    validate_args(_bt32_plan(), _bt32_args())
    heads = 7 if invalid == "heads" else 8
    plan = _bt32_plan(heads=heads)
    args = list(_bt32_args(heads=heads))
    q = args[0]
    assert isinstance(q, torch.Tensor)
    if invalid == "int32_cu":
        cu = args[8]
        assert isinstance(cu, torch.Tensor)
        args[8] = cu.to(torch.int32)
    elif invalid == "noncontiguous":
        args[0] = torch.empty((q.shape[0], 256), dtype=q.dtype)[:, ::2]
    elif invalid == "misaligned":
        args[0] = torch.empty(q.numel() + 1, dtype=q.dtype)[1:].view_as(q)
    elif invalid == "output_alias":
        args[9] = q
    with pytest.raises(exc.BackendUnsupported):
        validate_args(plan, args)


@pytest.mark.parametrize("task_order", ["identity", "longest_first_precompute"])
def test_bt32_host_call_arguments(task_order: str) -> None:
    body: list[str] = []
    append_host_call(body, _bt32_plan(task_order))
    tree = ast.parse("def wrapper():\n" + "\n".join(body))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_helion_chunk_prefill_host"
    )
    assert len(call.args) == 24
    if task_order == "longest_first_precompute":
        assert ast.unparse(call.args[8]) == "_prefill_order"
    else:
        assert ast.literal_eval(call.args[8]) is None
    assert all(ast.literal_eval(call.args[index]) is None for index in (9, 15, 16, 17))
    assert ast.unparse(call.args[10]) == "arg7"
    assert ast.unparse(call.args[12]) == "arg10"
    assert isinstance(call.args[18], ast.Constant)
    assert ast.literal_eval(call.args[18]) == 0
    assert ast.literal_eval(call.args[19]) is True
    assert ast.literal_eval(call.args[21]) == 1024
    assert ast.unparse(call.args[22]) == "cutlass.BFloat16"
    assert ast.literal_eval(call.args[23]) == "identity"
    code = "\n".join(body)
    assert "cute.make_layout((1, 16, 8, 128), stride=(128, 1024, 128, 1))" in code
    assert "cute.make_layout((1, 16, 8), stride=(8, 8, 1))" in code


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

    events: list[tuple[str, ...]] = []

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
        cast("tuple[torch.cuda.Stream, ...]", workers),
        cast("torch.cuda.Event", Event("ready")),
        cast("tuple[torch.cuda.Event, ...]", (Event("done0"), Event("done1"))),
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
            cast("tuple[torch.cuda.Stream, ...]", (StreamOrEvent(),)),
            cast("torch.cuda.Event", StreamOrEvent()),
            cast("tuple[torch.cuda.Event, ...]", (StreamOrEvent(),)),
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
    assert other.order is not None
    assert resource.order is not None
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
