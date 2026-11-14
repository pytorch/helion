from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple
from typing import cast
import weakref

import pytest
import torch
from torch.utils._pytree import register_pytree_node

import helion
from helion._testing import cpu_render_target
from helion._testing.binding_observation import assert_scalar_addition
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner.accuracy import assert_close
from helion.autotuner.accuracy import relative_l2_disagreement
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import _clone_args
from helion.autotuner.input_banks import clone_input_banks
from helion.autotuner.kernel_args import _clone_tensor_leaves
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@dataclass(frozen=True, slots=True)
class _MetadataMarker:
    value: int = 1


_METADATA_ANCHOR = _MetadataMarker()


@helion.kernel(autotune_effort="none")
def _metadata_identity(x, marker):
    out = torch.empty_like(x)
    scale = 1
    if marker is not _METADATA_ANCHOR:
        scale = -1
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * scale
    return out


@pytest.mark.parametrize("private_bank", [False, True])
def test_tensor_free_record_preserves_generated_host_identity(private_bank):
    original = (torch.ones(8), _METADATA_ANCHOR)
    cloned = (
        _clone_tensor_leaves(original) if private_bank else _clone_args(original, None)
    )
    with cpu_render_target():
        bound = cpu_bind(_metadata_identity, original)
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda *args, **kwargs: launches.append(args),
            scalar_values=True,
        )
        for args in (original, cloned, (original[0], _MetadataMarker())):
            host(*args)
    assert [
        [arg for arg in launch if type(arg) in (int, float, bool)]
        for launch in launches
    ] == [[1], [1], [-1]]
    assert cloned[1] is _METADATA_ANCHOR
    assert cloned[0] is not original[0]


def test_clone_args_preserves_records_outside_selected_storage_paths():
    @dataclass
    class Record:
        value: object
        metadata: object = None

    shared = []
    marker = Record(shared)
    shared.append(marker)
    tensor = torch.arange(4)
    affected = Record(tensor, marker)
    untouched = Record(torch.ones(3), marker)
    args = (tensor, affected, marker, shared, untouched)
    cloned = _clone_args(args, None, idx_to_clone=(0,))
    assert cloned[0] is cloned[1].value
    assert cloned[0] is not tensor and cloned[1] is not affected
    assert cloned[1].metadata is cloned[2] is marker
    assert cloned[3] is shared and cloned[3][0] is marker
    assert cloned[4] is untouched
    cloned[0].add_(5)
    assert torch.equal(tensor, torch.arange(4))


def test_tensor_free_record_and_function_cycles_keep_original_identity():
    @dataclass
    class Record:
        value: object = None

    record = Record()

    def inner():
        return record

    def outer():
        return inner()

    shared = [record, outer]
    record.value = shared
    args = (torch.arange(4), record, shared, inner, outer)
    for cloned in (_clone_args(args, None), _clone_tensor_leaves(args)):
        assert cloned[0] is not args[0]
        assert cloned[1] is record and cloned[2] is shared
        assert cloned[3] is inner and cloned[4] is outer
        assert cloned[4]() is record


def _container_identity_reference(x, left, right):
    return x + (2 if left is right else 5)


@helion.kernel(
    autotune_baseline_fn=_container_identity_reference, autotune_effort="none"
)
def _container_identity(x, left, right):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if left is right:
            out[tile] = x[tile] + 2
        else:
            out[tile] = x[tile] + 5
    return out


@pytest.mark.parametrize("alias", [False, True])
def test_container_identity_baseline_matches_compiled_branch(monkeypatch, alias):
    x = torch.arange(4, dtype=torch.float32)
    left = [x]
    right = left if alias else [x]
    args = (x, left, right)
    with cpu_render_target():
        bound = cpu_bind(_container_identity, args)
        assert_scalar_addition(bound, 2 if alias else 5)
    provider = object.__new__(LocalBenchmarkProvider)
    provider.kernel = bound
    provider.settings = bound.settings
    provider.args = args
    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    output, mutated, post_args = provider._compute_baseline()
    assert torch.equal(output, x + (2 if alias else 5))
    assert mutated == []
    assert (post_args[1] is post_args[2]) is alias
    assert post_args[0] is post_args[1][0] is post_args[2][0]
    assert torch.equal(x, torch.arange(4, dtype=torch.float32))


@dataclass
class _MutableRecord:
    tensor: torch.Tensor
    untouched: torch.Tensor | None = None


def _mutate_record_reference(marker, direct, record, tail):
    record.tensor.add_(marker)
    return record.tensor.clone()


@helion.kernel(autotune_baseline_fn=_mutate_record_reference, autotune_effort="none")
def _mutate_record(marker, direct, record, tail):
    values = record.tensor
    for tile in hl.tile(values.numel()):
        values[tile] = values[tile] + marker
    return values


@pytest.mark.parametrize("alias", [False, True])
def test_dataclass_baseline_is_private_and_reports_original_leaf_indices(
    monkeypatch, alias
):
    original = torch.arange(8, dtype=torch.float32)
    direct = original if alias else torch.zeros(8)
    tail = torch.ones(8)
    untouched = torch.arange(3)
    args = (3, direct, _MutableRecord(original, untouched), tail)
    with cpu_render_target():
        bound = cpu_bind(_mutate_record, args)
        code = bound.to_code(helion.Config(block_sizes=[16]))
    assert "record.tensor" in code
    provider = object.__new__(LocalBenchmarkProvider)
    provider.kernel = bound
    provider.settings = bound.settings
    provider.args = args
    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    output, mutated, post_args = provider._compute_baseline()
    assert torch.equal(original, torch.arange(8, dtype=torch.float32))
    assert torch.equal(output, original + 3)
    assert mutated == ([1, 2] if alias else [2])
    assert torch.equal(post_args[2].tensor, original + 3)
    assert torch.equal(post_args[2].untouched, untouched)
    working = _clone_args(args, None, idx_to_clone=mutated)
    assert working[3] is tail
    if alias:
        assert working[1] is working[2].tensor
    _mutate_record_reference(*working)
    assert torch.equal(original, torch.arange(8, dtype=torch.float32))


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("corrupt_mutation", [False, True])
def test_dataclass_worker_accuracy_checks_post_call_fields(
    monkeypatch, alias, corrupt_mutation
):
    tensor = torch.arange(8, dtype=torch.float32)
    direct = tensor if alias else torch.zeros_like(tensor)
    args = (3, direct, _MutableRecord(tensor, torch.ones(3)), torch.zeros(4))
    expected_args = _clone_args(args, None)
    output = _mutate_record_reference(*expected_args)

    def candidate(marker, direct, record, tail):
        assert (direct is record.tensor) is alias
        result = _mutate_record_reference(marker, direct, record, tail)
        if corrupt_mutation:
            record.tensor.add_(1)
        return result

    monkeypatch.setattr(
        benchmark_job, "_load_compiled_fn_for_worker", lambda spec: candidate
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: args)
    monkeypatch.setattr(
        benchmark_job,
        "_load_trusted_baseline_output",
        lambda path: output if path == "output" else expected_args,
    )
    monkeypatch.setattr(benchmark_job, "synchronize_device", lambda: None)
    job = benchmark_job.AccuracyCheckJob(
        None,
        "args",
        "output",
        atol=0,
        rtol=0,
        mutated_arg_indices=(1, 2) if alias else (2,),
        baseline_post_args_path="post-args",
    )
    result = job()
    assert result.ok is not corrupt_mutation
    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))
    assert torch.equal(args[2].untouched, torch.ones(3))


def test_record_post_call_comparison_checks_fields_and_types():
    original = _MutableRecord(torch.arange(8, dtype=torch.float32))
    copy = _clone_args((original,), None)[0]
    assert_close(copy, original, atol=0, rtol=0)
    assert relative_l2_disagreement(copy, original, 0.0) is None
    copy.tensor.add_(1)
    with pytest.raises(AssertionError):
        assert_close(copy, original, atol=0, rtol=0)
    assert relative_l2_disagreement(copy, original, 0.0) is not None
    with pytest.raises(AssertionError, match="Dataclass type mismatch"):
        assert_close(original, original.tensor, atol=0, rtol=0)
    assert relative_l2_disagreement(original, original.tensor, 0.0) is not None


def _record_tolerances(output, *, post_args=None, settings=None):
    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = settings or helion.Settings(autotune_fp8_accuracy="bitwise")
    provider._baseline_output = output
    provider._baseline_post_args = post_args
    provider.mutated_arg_indices = [] if post_args is None else [0]
    provider.log = lambda message: None
    return provider._compute_effective_tolerances()


@pytest.mark.parametrize("mutated", [False, True])
def test_dataclass_fp8_bitwise_accuracy_rejects_changed_bytes(mutated):
    @dataclass(frozen=True, slots=True)
    class Record:
        values: object

    expected = torch.tensor([100], dtype=torch.uint8).view(torch.float8_e4m3fn)
    actual = torch.tensor([101], dtype=torch.uint8).view(torch.float8_e4m3fn)
    direct_tolerances = _record_tolerances(expected)
    assert direct_tolerances == (0.0, 0.0)
    with pytest.raises(AssertionError):
        assert_close(actual, expected, *direct_tolerances, fp8_accuracy="bitwise")

    reference, candidate = Record({"tensor": expected}), Record({"tensor": actual})
    tolerances = (
        _record_tolerances(None, post_args=(reference,))
        if mutated
        else _record_tolerances(reference)
    )
    assert_close(reference, reference, *tolerances, fp8_accuracy="bitwise")
    with pytest.raises(AssertionError):
        assert_close(candidate, reference, *tolerances, fp8_accuracy="bitwise")
    assert tolerances == direct_tolerances


def test_record_tolerances_follow_compared_fields_and_handle_cycles():
    @dataclass
    class Record:
        values: object
        link: object = None

    @dataclass
    class CallableRecord:
        ignored: torch.Tensor

        def __call__(self):
            raise AssertionError("dtype discovery must not call inputs")

    fp8 = torch.ones(1).to(torch.float8_e4m3fn)
    record = Record(fp8)
    record.link = record
    # Non-field metadata and callable fields are not accuracy operands.
    vars(record)["metadata"] = torch.ones(1)
    output = (record, CallableRecord(torch.ones(1)))
    assert _record_tolerances(output) == (0.0, 0.0)
    record.link = Record(torch.ones(1))
    assert _record_tolerances(output) == (0.01, 0.01)


def test_record_tolerances_preserve_registered_pytree_contract():
    @dataclass
    class Record:
        values: torch.Tensor
        ignored: torch.Tensor

    rebuilt = []

    def unflatten(children, context):
        rebuilt.append(True)
        return Record(children[0], context)

    register_pytree_node(
        Record,
        lambda record: ([record.values], record.ignored),
        unflatten,
    )
    record = Record(torch.ones(1).to(torch.float8_e4m3fn), torch.ones(1))
    assert _record_tolerances(record) == (0.0, 0.0)
    assert rebuilt == []


@pytest.mark.parametrize(
    ("mode", "explicit_atol", "expected"),
    [("ulp", None, (0.01, 0.01)), ("bitwise", 0.25, (0.25, 0.01))],
)
def test_record_tolerances_preserve_mode_and_explicit_values(
    mode, explicit_atol, expected
):
    record = _MutableRecord(torch.ones(1).to(torch.float8_e4m3fn))
    settings = helion.Settings(
        autotune_fp8_accuracy=mode, autotune_baseline_atol=explicit_atol
    )
    assert _record_tolerances(record, settings=settings) == expected


def test_record_accuracy_cycles_compare_every_record_pair():
    @dataclass
    class Record:
        link: object
        tensor: torch.Tensor

    first = Record(None, torch.zeros(4))
    second = Record(first, torch.zeros(4))
    first.link = (first, second)
    clone = _clone_args((first,), None)[0]
    assert clone.link[0] is clone and clone.link[1].link is clone
    assert_close(first, clone, atol=0, rtol=0)
    assert relative_l2_disagreement(first, clone, 0.0) is None
    clone.link[1].tensor.add_(1)
    with pytest.raises(AssertionError):
        assert_close(first, clone, atol=0, rtol=0)
    assert relative_l2_disagreement(first, clone, 0.0) is not None

    # Visiting one actual record must not suppress comparison against a
    # different expected record at a later occurrence.
    first.link = first
    equal, different = _clone_args((first,), None)[0], _clone_args((first,), None)[0]
    different.tensor.add_(1)
    with pytest.raises(AssertionError):
        assert_close((first, first), (equal, different), atol=0, rtol=0)
    assert relative_l2_disagreement((first, first), (equal, different), 0.0) is not None


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


def test_clone_args_preserves_shared_storage_views_and_duplicate_refs() -> None:
    storage = torch.arange(128, dtype=torch.uint8)
    int16_view = storage.view(torch.int16).as_strided((3, 4), (6, 1), 2)
    float32_view = storage[8:72].view(torch.float32)[::2]
    broadcast_view = storage[20:21].expand(3, 4)
    unrelated = torch.ones(4)
    args = ([int16_view, float32_view, int16_view, broadcast_view], unrelated)

    cloned = cast(
        "tuple[list[torch.Tensor], torch.Tensor]",
        _clone_args(args, process_group_name=None, idx_to_clone=(0,)),
    )
    cloned_views, cloned_unrelated = cloned

    assert cloned_views[0] is cloned_views[2]
    assert len({_storage_id(tensor) for tensor in cloned_views}) == 1
    assert _storage_id(cloned_views[0]) != _storage_id(int16_view)
    assert cloned_unrelated is unrelated
    for original, copy_ in zip(
        (int16_view, float32_view, int16_view, broadcast_view),
        cloned_views,
        strict=True,
    ):
        assert copy_.dtype is original.dtype
        assert copy_.shape == original.shape
        assert copy_.stride() == original.stride()
        assert copy_.storage_offset() == original.storage_offset()
        assert copy_.untyped_storage().nbytes() == original.untyped_storage().nbytes()

    original_bytes = storage.clone()
    cloned_views[0].fill_(0)
    assert torch.equal(storage, original_bytes)


def test_clone_args_creates_candidate_private_alias_groups() -> None:
    storage = torch.arange(64, dtype=torch.float32)
    even = storage[3:35:2]
    overlapping = storage[7:23]
    args = (even, overlapping)

    first = cast(
        "tuple[torch.Tensor, torch.Tensor]",
        _clone_args(args, process_group_name=None, idx_to_clone=(0,)),
    )
    second = cast(
        "tuple[torch.Tensor, torch.Tensor]",
        _clone_args(args, process_group_name=None, idx_to_clone=(0,)),
    )

    assert _storage_id(first[0]) == _storage_id(first[1])
    assert _storage_id(second[0]) == _storage_id(second[1])
    assert _storage_id(first[0]) != _storage_id(second[0])
    assert _storage_id(first[0]) != _storage_id(storage)
    first[1].zero_()
    assert not torch.equal(first[0], second[0])
    assert torch.equal(storage, torch.arange(64, dtype=torch.float32))


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("selection", [None, (1,), ()])
def test_clone_args_retains_registered_temporary_views(wrapped, selection):
    @dataclass
    class ViewBox:
        tensor: torch.Tensor
        metadata: object

    @dataclass
    class Holder:
        value: object

    class Pair(NamedTuple):
        value: object

    references = []

    def flatten(value):
        assert all(reference() is not None for reference in references)
        tensor = value.tensor.view_as(value.tensor)
        references.append(weakref.ref(tensor))
        return [tensor], value.metadata

    register_pytree_node(
        ViewBox,
        flatten,
        lambda children, metadata: ViewBox(children[0], metadata),
    )
    original = torch.arange(8, dtype=torch.float32)
    unrelated = torch.ones(3)
    metadata = object()
    box = ViewBox(original[1::2], metadata)
    if wrapped:
        pair = Pair(box)
        values = {"record": Holder(box), "pairs": OrderedDict(a=pair, b=pair)}
    else:
        values = box
    cloned = _clone_args((17, values, original, unrelated), None, selection)
    if wrapped:
        result = cloned[1]["record"].value
        pairs = cloned[1]["pairs"]
        assert type(pairs) is OrderedDict and list(pairs) == ["a", "b"]
        assert type(pairs["a"]) is Pair and pairs["a"] is pairs["b"]
        assert result is pairs["a"].value
    else:
        result = cloned[1]
    assert result.metadata is metadata
    assert result.tensor.stride() == box.tensor.stride()
    assert result.tensor.storage_offset() == box.tensor.storage_offset()
    assert _storage_id(result.tensor) == _storage_id(cloned[2])
    assert (_storage_id(result.tensor) != _storage_id(original)) is (selection != ())
    assert (cloned[3] is unrelated) is (selection is not None)
    if selection != ():
        result.tensor.add_(10)
        assert torch.equal(cloned[2][1::2], original[1::2] + 10)
        assert torch.equal(original, torch.arange(8, dtype=torch.float32))
    assert len(references) == 1


def _callback_alias_reference(x, callback):
    x.add_(3)
    return callback().clone()


@helion.kernel(autotune_baseline_fn=_callback_alias_reference, autotune_effort="none")
def _callback_alias_kernel(x, callback):
    value = callback()
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 3
    return value


@pytest.mark.parametrize("capture", ["closure", "default", "keyword_default"])
def test_mutated_argument_keeps_callback_alias_in_baseline_and_banks(
    monkeypatch, capture
):
    x = torch.arange(8, dtype=torch.float32)
    captured = x[1::2]
    if capture == "closure":

        def callback():
            return captured
    elif capture == "default":

        def callback(value=captured):
            return value
    else:

        def callback(*, value=captured):
            return value

    args = (x, callback)
    with cpu_render_target():
        bound = cpu_bind(_callback_alias_kernel, args)
    provider = object.__new__(LocalBenchmarkProvider)
    provider.kernel = bound
    provider.settings = bound.settings
    provider.args = args
    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    output, mutated, post_args = provider._compute_baseline()
    torch.testing.assert_close(output, captured + 3)
    assert mutated == [0, 1]
    assert _storage_id(post_args[0]) == _storage_id(post_args[1]())
    torch.testing.assert_close(post_args[1](), captured + 3)

    unrelated = torch.ones(3)

    def unchanged():
        return unrelated

    args_for_benchmark = _clone_args(
        (x, callback, callback, unchanged), None, idx_to_clone=mutated
    )
    assert args_for_benchmark[1] is args_for_benchmark[2]
    assert args_for_benchmark[1] is not callback
    assert args_for_benchmark[3] is unchanged
    banks = clone_input_banks(args_for_benchmark, 3)
    assert len({_storage_id(bank[0]) for bank in banks} | {_storage_id(x)}) == 4
    for direct, first, second, other in banks:
        assert first is second
        torch.testing.assert_close(other(), unrelated)
        view = first()
        assert _storage_id(direct) == _storage_id(view)
        assert view.stride() == captured.stride()
        assert view.storage_offset() == captured.storage_offset()
        torch.testing.assert_close(
            _callback_alias_reference(direct, first), captured + 3
        )
    torch.testing.assert_close(x, torch.arange(8, dtype=torch.float32))


def test_selected_callback_preserves_record_cycle_and_ordinary_leaf_positions():
    @dataclass
    class Record:
        tensor: torch.Tensor
        callback: object = None

    x = torch.arange(4)
    record = Record(x)
    empty = object()

    def callback(read_empty=False):
        if read_empty:
            return empty  # noqa: F821 - deliberately empty closure cell
        return record

    record.callback = callback
    del empty
    unrelated = torch.ones(2)
    # A function remains one ordinary pytree slot, even when its operands
    # contain a cycle and an empty closure cell.
    direct, fn, tail = _clone_args((x, callback, unrelated), None, idx_to_clone=(1,))
    assert direct is not x
    assert fn is not callback and fn().callback is fn
    assert fn().tensor is direct
    assert tail is unrelated
    with pytest.raises(NameError, match="empty"):
        fn(True)
    direct.add_(5)
    torch.testing.assert_close(x, torch.arange(4))
