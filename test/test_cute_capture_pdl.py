from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from helion.runtime.cute import capture_pdl_launch
from helion.runtime.cute.capture_pdl import CaptureGuard
from helion.runtime.cute.capture_pdl import CaptureKey
from helion.runtime.cute.capture_pdl import Dependency
from helion.runtime.cute.capture_pdl import LaunchEffects
from helion.runtime.cute.capture_pdl import StorageRange
from helion.runtime.cute.capture_pdl import WaitMode
from helion.runtime.cute.capture_pdl import storage_union
from helion.runtime.cute.capture_pdl_cuda import CaptureSnapshot
from helion.runtime.cute.capture_pdl_cuda import CudaCaptureDriver
from helion.runtime.cute.capture_pdl_cuda import _KernelParams
from helion.runtime.cute.capture_pdl_cuda import finish_capture_launch

pytestmark = pytest.mark.cpu_only
KEY = CaptureKey(context=1, thread=2, stream=3, capture_id=4, graph=5)
X = StorageRange(0, 100, 200)
Y = StorageRange(0, 300, 400)
Z = StorageRange(0, 500, 600)


def launch(guard, node, predecessor, reads=(), writes=(), key=KEY):
    decision = guard.decide(
        key,
        node,
        () if predecessor is None else (Dependency(predecessor, 1, 0, 1),),
        LaunchEffects(reads, writes),
    )
    guard.record(key, node, decision)
    return decision


def test_pending_union_catches_non_immediate_producer_and_resets_at_anchor():
    guard = CaptureGuard()
    assert launch(guard, 1, None, writes=(X,)).mode == WaitMode.EARLY
    assert launch(guard, 2, 1, writes=(Y,)).mode == WaitMode.BEFORE_WRITES
    assert launch(guard, 3, 2, reads=(X,), writes=(Z,)).mode == WaitMode.EARLY
    # Node 3 waited for both older writers before triggering its successors.
    assert launch(guard, 4, 3, reads=(X, Y)).mode == WaitMode.BEFORE_WRITES


def test_unknown_intervening_node_and_multiple_dependencies_create_early_anchor():
    guard = CaptureGuard()
    launch(guard, 1, None, writes=(X,))
    for node, dependencies in (
        (3, (Dependency(2),)),
        (4, (Dependency(1), Dependency(2))),
    ):
        decision = guard.decide(KEY, node, dependencies, LaunchEffects((), (Y,)))
        assert decision.mode == WaitMode.EARLY
        guard.record(KEY, node, decision)
        assert launch(guard, node + 10, node, reads=(X,)).mode == WaitMode.BEFORE_WRITES


@pytest.mark.parametrize(
    "dependency",
    (
        Dependency(1, 2, 0, 1),
        Dependency(1, 1, 0, 0),
        Dependency(1, 1, 1, 1),
        Dependency(1, reserved=b"\1\0\0\0\0"),
    ),
)
def test_unknown_and_launch_order_edges_are_untrackable(dependency):
    guard = CaptureGuard()
    launch(guard, 1, None)
    decision = guard.decide(KEY, 2, (dependency,), LaunchEffects((), ()))
    assert decision.mode == WaitMode.EARLY
    assert not decision.trackable


@pytest.mark.parametrize(
    "field", ("context", "thread", "stream", "capture_id", "graph")
)
def test_capture_identity_components_never_share_pending_state(field):
    guard = CaptureGuard()
    launch(guard, 1, None)
    other = replace(KEY, **{field: getattr(KEY, field) + 10})
    assert launch(guard, 2, 1, key=other).mode == WaitMode.EARLY


def test_aliases_use_allocation_intervals_not_tensor_identity_or_data_pointer():
    guard = CaptureGuard()
    launch(guard, 1, None, writes=(X,))
    assert (
        launch(guard, 2, 1, reads=(StorageRange(0, 199, 205),)).mode == WaitMode.EARLY
    )
    # Adjacent ranges and a distinct device cannot alias.
    assert (
        launch(guard, 3, 1, reads=(StorageRange(0, 200, 210),)).mode
        == WaitMode.BEFORE_WRITES
    )
    assert (
        launch(guard, 4, 1, reads=(StorageRange(1, 100, 200),)).mode
        == WaitMode.BEFORE_WRITES
    )


def test_unknown_effects_and_noncapture_never_create_a_proof():
    guard = CaptureGuard()
    for key, effects in ((None, LaunchEffects((), ())), (KEY, None)):
        decision = guard.decide(key, 1, (), effects)
        assert decision.mode == WaitMode.EARLY
        assert not decision.trackable


def test_failed_mode_patch_is_not_recorded_as_a_late_node():
    guard = CaptureGuard()
    launch(guard, 1, None, writes=(X,))
    selected = guard.decide(KEY, 2, (Dependency(1),), LaunchEffects((), (Y,)))
    assert selected.mode == WaitMode.BEFORE_WRITES
    # The adapter did not successfully install this choice: no record call.
    assert launch(guard, 3, 2, reads=(X,)).mode == WaitMode.EARLY


def test_bounded_eviction_and_capture_release_only_lose_overlap():
    guard = CaptureGuard(max_captures=1, max_nodes=2, max_ranges=1)
    launch(guard, 1, None, writes=(X,))
    # Disjoint writes exceed the bounded pending set and force a fresh anchor.
    assert launch(guard, 2, 1, writes=(Y,)).mode == WaitMode.EARLY
    launch(guard, 3, 2)
    assert launch(guard, 4, 1).mode == WaitMode.EARLY
    other = replace(KEY, capture_id=44)
    launch(guard, 1, None, key=other)
    assert KEY not in guard.captures
    guard.release(other)
    assert not guard.captures


def test_union_preserves_device_and_empty_range_semantics():
    assert storage_union(
        (
            StorageRange(0, 150, 300),
            X,
            StorageRange(0, 300, 300),
            StorageRange(1, 100, 200),
        )
    ) == (StorageRange(0, 100, 300), StorageRange(1, 100, 200))
    assert not StorageRange(0, 150, 150).overlaps(X)


class FakeCaptureDriver:
    def __init__(self, after, dependencies=(), *, patch_succeeds=True):
        self.after = after
        self.dependencies = dependencies
        self.patch_succeeds = patch_succeeds
        self.patches = []

    def snapshot(self, stream):
        assert stream == KEY.stream
        return self.after

    def kernel_dependencies(self, node):
        return self.dependencies

    def set_wait_mode(self, node, mode, parameter_sizes):
        assert parameter_sizes[-1] == 4
        self.patches.append((node, mode))
        return self.patch_succeeds


def test_post_capture_proof_patches_the_unique_actual_node():
    guard = CaptureGuard()
    launch(guard, 1, None, writes=(X,))
    driver = FakeCaptureDriver(
        CaptureSnapshot(KEY, frozenset((1, 2))), (Dependency(1, 1, 0, 1),)
    )
    assert (
        finish_capture_launch(
            driver,
            guard,
            CaptureSnapshot(KEY, frozenset((1,))),
            LaunchEffects((), (Y,)),
            (8, 4),
        )
        == WaitMode.BEFORE_WRITES
    )
    assert driver.patches == [(2, WaitMode.BEFORE_WRITES)]
    assert guard.captures[KEY][2] == (X, Y)


@pytest.mark.parametrize(
    "after",
    (
        None,
        CaptureSnapshot(replace(KEY, capture_id=100), frozenset((1, 2))),
        CaptureSnapshot(KEY, frozenset((2,))),  # Removed old node: graph surgery.
        CaptureSnapshot(KEY, frozenset((1,))),  # No enqueue.
        CaptureSnapshot(KEY, frozenset((1, 2, 3))),  # Extra/interleaved enqueue.
    ),
)
def test_changed_or_ambiguous_capture_keeps_original_safe_mode(after):
    guard = CaptureGuard()
    launch(guard, 1, None)
    driver = FakeCaptureDriver(after)
    assert (
        finish_capture_launch(
            driver,
            guard,
            CaptureSnapshot(KEY, frozenset((1,))),
            LaunchEffects((), ()),
            (8, 4),
        )
        == WaitMode.EARLY
    )
    assert not driver.patches
    assert list(guard.captures[KEY]) == [1]


def test_actual_unknown_incoming_edge_cannot_use_stale_pre_enqueue_tail():
    guard = CaptureGuard()
    launch(guard, 1, None)
    driver = FakeCaptureDriver(
        CaptureSnapshot(KEY, frozenset((1, 2, 3))), (Dependency(2, 1, 0, 1),)
    )
    # Node 2 was present before our enqueue but is an unknown external producer.
    assert (
        finish_capture_launch(
            driver,
            guard,
            CaptureSnapshot(KEY, frozenset((1, 2))),
            LaunchEffects((X,), ()),
            (8, 4),
        )
        == WaitMode.EARLY
    )
    assert not driver.patches


def test_failed_native_argument_patch_does_not_publish_pending_state():
    guard = CaptureGuard()
    launch(guard, 1, None)
    driver = FakeCaptureDriver(
        CaptureSnapshot(KEY, frozenset((1, 2))), (Dependency(1),), patch_succeeds=False
    )
    assert (
        finish_capture_launch(
            driver,
            guard,
            CaptureSnapshot(KEY, frozenset((1,))),
            LaunchEffects((), (X,)),
            (8, 4),
        )
        == WaitMode.EARLY
    )
    assert 2 not in guard.captures[KEY]


def test_empty_capture_and_root_dependencies_do_not_issue_zero_capacity_queries():
    def context(out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 11
        return 0

    def capture(stream, status, capture_id, graph, dependencies, edges, count):
        ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 1
        ctypes.cast(capture_id, ctypes.POINTER(ctypes.c_uint64))[0] = 12
        ctypes.cast(graph, ctypes.POINTER(ctypes.c_void_p))[0] = 13
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    def nodes(graph, output, count):
        assert output is None, "a zero-capacity output array is not a count query"
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    def node_type(node, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    def dependencies(node, output, edges, count):
        assert output is None and edges is None
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    driver = object.__new__(CudaCaptureDriver)
    driver.library = SimpleNamespace(
        cuCtxGetCurrent=context,
        cuStreamGetCaptureInfo_v3=capture,
        cuGraphGetNodes=nodes,
        cuGraphNodeGetType=node_type,
        cuGraphNodeGetDependencies_v2=dependencies,
    )
    snapshot = driver.snapshot(10)
    assert snapshot is not None and not snapshot.nodes
    assert driver.kernel_dependencies(14) == ()


@pytest.mark.parametrize(
    "maps,expected_maps,fail_update",
    [
        (4, 4, False),
        (5, 5, False),
        (4, 4, True),
        (5, 5, True),
        (4, 5, False),
        (5, 4, False),
    ],
)
def test_native_mode_patch_copies_argument_values_and_changes_only_the_mode(
    maps,
    expected_maps,
    fail_update,
):
    sizes = (8,) * 9 + (128,) * maps + (4,)
    expected_sizes = (8,) * 9 + (128,) * expected_maps + (4,)
    original = (
        *(
            bytes((index + byte) % 256 for byte in range(size))
            for index, size in enumerate(sizes[:-1])
        ),
        bytes(4),
    )
    values = [ctypes.create_string_buffer(value, len(value)) for value in original]
    pointers = (ctypes.c_void_p * len(values))(
        *(ctypes.addressof(value) for value in values)
    )
    captured = _KernelParams(func=55, parameters=pointers)
    updated = []

    def get_params(node, output):
        ctypes.memmove(output, ctypes.byref(captured), ctypes.sizeof(captured))
        return 0

    def parameter(function, index, offset, size):
        assert index < len(sizes), "never query an intentionally out-of-range parameter"
        ctypes.cast(offset, ctypes.POINTER(ctypes.c_size_t))[0] = index * 128
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = sizes[index]
        return 0

    def set_params(node, address):
        params = ctypes.cast(address, ctypes.POINTER(_KernelParams)).contents
        # Emulate invalidation of the node-owned argument bytes on replacement.
        for value in values:
            ctypes.memset(value, 255, ctypes.sizeof(value))
        updated.append(
            tuple(
                ctypes.string_at(params.parameters[i], size)
                for i, size in enumerate(sizes)
            )
        )
        return 1 if fail_update else 0

    driver = object.__new__(CudaCaptureDriver)
    driver.library = SimpleNamespace(
        cuGraphKernelNodeGetParams_v2=get_params,
        cuFuncGetParamInfo=parameter,
        cuGraphKernelNodeSetParams_v2=set_params,
    )
    if expected_maps != maps:
        assert not driver.set_wait_mode(7, WaitMode.BEFORE_WRITES, expected_sizes)
        assert updated == []
        assert tuple(value.raw for value in values) == original
    else:
        if fail_update:
            with pytest.raises(RuntimeError, match="wait-mode update failed"):
                driver.set_wait_mode(7, WaitMode.BEFORE_WRITES, expected_sizes)
        else:
            assert driver.set_wait_mode(7, WaitMode.BEFORE_WRITES, expected_sizes)
        assert updated == [(*original[:-1], b"\1\0\0\0")]


@pytest.mark.parametrize("grid", [(0,), (1, 0, 1), (-1,), (), (1, 1, 1, 1)])
def test_empty_or_unproved_grid_never_enters_capture_tracking(grid):
    from helion.runtime.cute.capture_pdl_launch import prepare_capture_launch

    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=(
            {
                "kind": "helion_paged_attention",
                "capture_pdl_contract": "paired-output-stores-v1",
                "capture_pdl_parameter_sizes": (128, 128, 128, 128, 4),
            },
            {"kind": "helion_mma_dag_alias"},
        ),
        _helion_cute_use_pdl=True,
        _helion_cute_last_launch_cache=object(),
    )
    # No argument/storage/driver access is needed: there will be no tracked
    # enqueue even if a concurrent thread adds one compatible graph node.
    assert prepare_capture_launch(kernel, (), grid) is None


def test_capture_suffix_follows_final_wrapper_signature_and_rejects_unknown_abi():
    from helion._compiler.cute.host_metadata import _capture_pdl_wrapper_parameter_sizes

    annotation = "cutlass.GridConstant[_helion_mma_dag_tc.cuda.TensorMap]"
    for count in (4, 5):
        params = [f"map{index}" for index in range(count)] + ["mode"]
        annotations = dict.fromkeys(params[:-1], annotation) | {"mode": "cutlass.Int32"}
        assert _capture_pdl_wrapper_parameter_sizes(params, annotations, "mode") == (
            (128,) * count + (4,)
        )
        for invalid_params, invalid_annotations in (
            (params[:-1], annotations),
            (["mode", *params[:-1]], annotations),
            (
                [*params[:-1], "scalar", "mode"],
                annotations | {"scalar": "cutlass.Int32"},
            ),
            (params, annotations | {"map0": "unknown.TensorMap"}),
            (params, annotations | {"mode": "cutlass.Int64"}),
            ([params[0], *params], annotations),
        ):
            assert (
                _capture_pdl_wrapper_parameter_sizes(
                    invalid_params, invalid_annotations, "mode"
                )
                is None
            )


@pytest.mark.parametrize(
    "suffix", [None, (), (4,), (128, 8), (128, 4, 128), (128.0, 4)]
)
def test_missing_or_inconsistent_wrapper_suffix_stays_early(suffix):
    from helion.runtime.cute.capture_pdl_launch import prepare_capture_launch

    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=(
            {
                "kind": "helion_paged_attention",
                "capture_pdl_contract": "paired-output-stores-v1",
                "capture_pdl_parameter_sizes": suffix,
            },
            {"kind": "helion_mma_dag_alias"},
        ),
        _helion_cute_use_pdl=True,
        _helion_cute_last_launch_cache=object(),
    )
    # Decline before inspecting arguments, storages, devices, or graph state.
    assert prepare_capture_launch(kernel, (), (1,)) is None


def test_pending_launch_restores_its_device_for_snapshot_and_mode_update(monkeypatch):
    current = [torch.device("cuda:1")]
    launch_device = torch.device("cuda:0")
    events = []

    @contextmanager
    def device_context(device):
        previous = current[0]
        current[0] = device
        try:
            yield
        finally:
            current[0] = previous

    def finish(driver, guard, before, effects, sizes):
        assert current[0] == launch_device
        events.append((driver, guard, before, effects, sizes))

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(capture_pdl_launch, "finish_capture_launch", finish)
    guard = SimpleNamespace(captures={})
    before = SimpleNamespace(key="launch-on-device-zero")
    pending = capture_pdl_launch.PendingCaptureLaunch(
        driver=object(),
        guard=guard,
        before=before,
        effects=object(),
        parameter_sizes=(8, 128, 4),
        device=launch_device,
    )
    pending.finish()
    assert len(events) == 1
    assert current[0] == torch.device("cuda:1")
