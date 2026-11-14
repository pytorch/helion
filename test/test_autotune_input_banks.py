from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import pickle
from types import SimpleNamespace
from typing import ClassVar
from typing import NamedTuple
from unittest.mock import Mock
import weakref

import pytest
import torch
from torch.utils._pytree import register_pytree_node

from helion import exc
from helion._compiler import source_location
from helion.autotuner import base_search
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner import input_banks
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.input_banks import InputBankCycle
from helion.autotuner.input_banks import InputBankError
from helion.autotuner.input_banks import InputBankMemoryError
from helion.autotuner.input_banks import _tensor_operands
from helion.autotuner.input_banks import check_input_bank_mode
from helion.autotuner.input_banks import clone_input_banks
from helion.autotuner.input_banks import tensor_storage_bytes
from helion.autotuner.local_cache import _search_policy_hash
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.random_search import RandomSearch
from helion.runtime.config import Config
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("context", ["distributed", "mtia", "unsupported"])
def test_bank_provider_rejects_known_context_before_baseline(monkeypatch, context):
    for name in (
        "HELION_AUTOTUNE_L2_FLUSH",
        "HELION_AUTOTUNE_FINAL_L2_FLUSH",
        "HELION_AUTOTUNE_TARGET_REGIME",
    ):
        monkeypatch.setenv(name, "replay")
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setattr(
        benchmark_provider.dist, "is_initialized", lambda: context == "distributed"
    )
    baseline = Mock(return_value=(None, [], ()))
    monkeypatch.setattr(LocalBenchmarkProvider, "_compute_baseline", baseline)
    monkeypatch.setattr(
        LocalBenchmarkProvider, "_compute_effective_tolerances", lambda self: (0, 0)
    )
    monkeypatch.setattr(LocalBenchmarkProvider, "_decide_num_jobs", lambda self: 1)
    monkeypatch.setattr(
        LocalBenchmarkProvider,
        "_subprocess_benchmark_uses_wall_clock",
        lambda self: False,
    )
    kernel = SimpleNamespace(
        env=SimpleNamespace(
            device=torch.device("mtia" if context == "mtia" else "cuda")
        ),
        supports_subprocess_benchmark=lambda: context != "unsupported",
    )
    config_spec = SimpleNamespace(backend=SimpleNamespace(get_do_bench=lambda: None))
    with pytest.raises(exc.InvalidAPIUsage, match="isolated|distributed|MTIA"):
        LocalBenchmarkProvider(
            kernel,
            Settings(autotune_benchmark_input_banks=3),
            config_spec,
            (),
            Mock(),
            Mock(),
        )
    baseline.assert_not_called()


def test_bank_setup_refuses_unserializable_args_before_search(tmp_path):
    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(autotune_precompile=False)
    provider._input_bank_count = 3
    provider.args = ()
    provider.log = Mock()
    provider._precompile_tmpdir = SimpleNamespace(name=str(tmp_path))
    provider._precompile_args_path = None
    provider._subprocess_benchmark_enabled = lambda: True
    provider._subprocess_accuracy_check_enabled = lambda: False
    provider._save_kernel_args = Mock(return_value=ValueError("opaque argument"))
    with pytest.raises(InputBankError, match="arguments|pickl"):
        provider.setup()
    assert not provider._args_unpicklable
    assert provider._precompile_args_path is None


@pytest.mark.parametrize("failure", ["serialize", "load"])
@pytest.mark.parametrize("phase", ["search", "final"])
def test_bank_wrapper_failure_does_not_disable_later_candidates(
    monkeypatch, failure, phase
):
    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(autotune_accuracy_check=False)
    provider._input_bank_count = 3
    provider.log = Mock()
    provider._autotune_metrics = SimpleNamespace(num_configs_tested=0)
    provider._precompile_args_path = "args.pt"
    provider.mutated_arg_indices = ()
    provider._subprocess_benchmark_enabled = lambda: True
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    provider._timeout_retry_repetitions = lambda *args: None
    provider._probe_long_kernel = lambda: False
    provider._worker_compile_timeout = lambda: 10
    provider._record_compile_failure = Mock()
    worker = Mock()
    worker.run.side_effect = (
        [benchmark_job.CompiledFunctionLoadError("missing candidate source"), 0.75]
        if failure == "load"
        else [0.75]
    )
    provider._benchmark_worker = worker
    provider._get_benchmark_worker = lambda: worker
    monkeypatch.setattr(
        benchmark_provider,
        "_serialize_compiled_fn",
        Mock(side_effect=[RuntimeError("unserializable candidate"), object()])
        if failure == "serialize"
        else lambda fn: object(),
    )
    fns = [lambda: None, lambda: None]
    if phase == "search":
        assert provider._benchmark_function(Config(), fns[0]) == float("inf")
        assert provider._benchmark_function(Config(), fns[1]) == 0.75
        provider._record_compile_failure.assert_called_once()
    else:
        assert provider.benchmark_isolated(fns, warmup=1, rep=1) == [None, 0.75]
    assert not provider._subprocess_wrapper_unloadable
    worker.shutdown.assert_not_called()


@pytest.mark.parametrize("failure", ["serialize", "load"])
def test_bank_accuracy_wrapper_failure_keeps_worker_available(monkeypatch, failure):
    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings()
    provider._input_bank_count = 3
    provider._precompile_args_path = "args.pt"
    provider._precompile_baseline_path = "baseline.pt"
    provider._precompile_baseline_post_args_path = None
    provider._subprocess_accuracy_check_enabled = lambda: True
    provider.mutated_arg_indices = ()
    provider._effective_atol = 0.01
    provider._effective_rtol = 0.01
    provider._scale_atol = False
    provider.log = Mock()
    expected = benchmark_job.AccuracyCheckResult(ok=True)
    worker = Mock()
    worker.run.side_effect = (
        [benchmark_job.CompiledFunctionLoadError("missing candidate source"), expected]
        if failure == "load"
        else [expected]
    )
    provider._benchmark_worker = worker
    provider._get_benchmark_worker = lambda: worker
    monkeypatch.setattr(
        benchmark_provider,
        "_serialize_compiled_fn",
        Mock(side_effect=[RuntimeError("unserializable candidate"), object()])
        if failure == "serialize"
        else lambda fn: object(),
    )
    error_type = (
        RuntimeError
        if failure == "serialize"
        else benchmark_job.CompiledFunctionLoadError
    )
    with pytest.raises(error_type):
        provider._run_subprocess_accuracy_check_job(lambda: None)
    assert provider._run_subprocess_accuracy_check_job(lambda: None) is expected
    assert not provider._subprocess_wrapper_unloadable
    worker.shutdown.assert_not_called()


@pytest.mark.parametrize("error_type", [InputBankError, InputBankMemoryError])
def test_bank_error_pickle_preserves_message_and_state(monkeypatch, error_type):
    worker_location = source_location.SourceLocation(1, 0, 1, 4, "worker", __file__)
    parent_location = source_location.SourceLocation(2, 0, 2, 4, "parent", __file__)
    monkeypatch.setattr(source_location, "current_location", lambda: worker_location)
    error = error_type("bank context failed")
    error.worker_job = 17
    original_message = str(error)
    monkeypatch.setattr(source_location, "current_location", lambda: parent_location)
    for _ in range(2):
        error = pickle.loads(pickle.dumps(error))
        assert type(error) is error_type
        assert isinstance(error, exc.InvalidAPIUsage)
        assert str(error) == original_message
        assert error.args == (original_message,)
        assert error.worker_job == 17
        assert error.location.name == "worker"
    assert ("Invalid usage of Helion API" in original_message) == (
        error_type is InputBankError
    )


def test_default_preserves_original_arguments(monkeypatch):
    monkeypatch.delenv("HELION_AUTOTUNE_BENCHMARK_INPUT_BANKS", raising=False)
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "0")
    check_input_bank_mode(1, wall_clock=True, l2_flush_mode="call")
    args = (torch.arange(8), 17)
    assert clone_input_banks(args, 1)[0] is args
    assert Settings().autotune_benchmark_input_banks == 1


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_bank_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        Settings(autotune_benchmark_input_banks=count)


def test_explicit_and_environment_count(monkeypatch):
    monkeypatch.setenv("HELION_AUTOTUNE_BENCHMARK_INPUT_BANKS", "3")
    assert Settings().autotune_benchmark_input_banks == 3
    assert (
        Settings(autotune_benchmark_input_banks=2).autotune_benchmark_input_banks == 2
    )
    assert Settings().copy().autotune_benchmark_input_banks == 3


def test_banks_preserve_mixed_aliases_offsets_and_closures():
    storage = torch.arange(128).to(torch.int8)
    values = storage.as_strided((3, 4), (16, 2), 5)
    scales = storage.view(torch.uint8).as_strided((3, 2), (16, 1), 12)

    def epilogue():
        return values

    args = (values, scales, values, epilogue, 7, None)
    banks = clone_input_banks(args, 3)
    all_storage = []
    for bank in banks:
        for original, cloned in zip(args[:3], bank[:3], strict=True):
            assert original.dtype == cloned.dtype
            assert original.shape == cloned.shape
            assert original.stride() == cloned.stride()
            assert original.storage_offset() == cloned.storage_offset()
            assert torch.equal(original, cloned)
        pointers = {tensor.untyped_storage().data_ptr() for tensor in bank[:3]}
        assert len(pointers) == 1
        assert bank[3]().untyped_storage().data_ptr() in pointers
        assert bank[4:] == args[4:]
        all_storage.extend(pointers)
    assert len(set(all_storage)) == 3
    banks[1][0].fill_(42)
    assert torch.equal(banks[0][0], banks[2][0])
    assert not torch.equal(banks[0][0], banks[1][0])
    assert len(_tensor_operands(args)) == 4  # three explicit views plus closure


def test_empty_cell_callback_keeps_other_captures_and_defaults():
    tensor = torch.arange(8)
    empty = object()

    def callback(value=tensor, *, keyword=tensor, read_empty=False):
        if read_empty:
            return empty  # noqa: F821 - deliberately empty closure cell
        return tensor, value, keyword

    del empty
    assert tensor_storage_bytes(_tensor_operands((callback,))) == (
        tensor.untyped_storage().nbytes()
    )
    banks = clone_input_banks((tensor, callback, callback), 3)
    for index, (direct, first, second) in enumerate(banks):
        assert first is second
        assert (first is callback) is (index == 0)
        assert all(value is direct for value in first())
        cells = dict(zip(first.__code__.co_freevars, first.__closure__, strict=True))
        pytest.raises(ValueError, getattr, cells["empty"], "cell_contents")
        with pytest.raises(NameError, match="empty"):
            first(read_empty=True)
    banks[1][0].add_(10)
    assert torch.equal(banks[0][0], banks[2][0])
    assert not torch.equal(banks[1][0], banks[0][0])


def test_callback_record_cycle_clones_shared_cells_and_tensor_aliases():
    @dataclasses.dataclass
    class Record:
        tensor: torch.Tensor
        callback: object = None

    record = Record(torch.arange(4))
    empty = object()

    def first(read_empty=False):
        if read_empty:
            return empty  # noqa: F821 - deliberately empty closure cell
        return record

    def second():
        return record

    record.callback = first
    del empty
    banks = clone_input_banks((record.tensor, record, first, second), 3)
    for tensor, copied, a, b in banks:
        assert copied.tensor is tensor
        assert a() is b() is copied
        assert copied.callback is a
        a_cells = dict(zip(a.__code__.co_freevars, a.__closure__, strict=True))
        b_cells = dict(zip(b.__code__.co_freevars, b.__closure__, strict=True))
        assert a_cells["record"] is b_cells["record"]
        with pytest.raises(NameError, match="empty"):
            a(True)
    banks[1][0].add_(7)
    assert torch.equal(banks[0][0], banks[2][0])
    assert not torch.equal(banks[1][0], banks[0][0])


def test_cloned_callback_keeps_empty_kwdefaults_alias():
    tensor = torch.arange(4)

    def callback():
        return tensor

    callback.__kwdefaults__ = {}
    for direct, fn, metadata in clone_input_banks(
        (tensor, callback, callback.__kwdefaults__), 3
    ):
        assert fn() is direct
        assert fn.__kwdefaults__ is metadata is callback.__kwdefaults__


def test_bank_storage_accounting_handles_internal_container_cycles():
    # This checks the cloning utility's graph contract, not frontend admission.
    tensor = torch.arange(4)
    sequence = [tensor]
    mapping = {"cycle": sequence}
    sequence.append(mapping)
    args = (tensor, sequence, mapping)
    assert tensor_storage_bytes(_tensor_operands(args)) == (
        tensor.untyped_storage().nbytes()
    )
    for direct, values, lookup in clone_input_banks(args, 3):
        assert values[0] is direct
        assert values[1] is lookup and lookup["cycle"] is values
        assert tensor_storage_bytes(_tensor_operands((values, lookup))) == (
            tensor.untyped_storage().nbytes()
        )


def test_dataclass_banks_preserve_direct_aliases_and_isolate_mutations():
    @dataclasses.dataclass
    class Holder:
        tensor: torch.Tensor

    tensor = torch.arange(8)
    holder = Holder(tensor)
    banks = clone_input_banks((tensor, holder, holder), 3)
    for direct, record, duplicate in banks:
        assert record is duplicate
        assert direct is record.tensor
        assert torch.equal(direct, record.tensor)
    assert len({bank[0].untyped_storage().data_ptr() for bank in banks}) == 3
    banks[1][1].tensor.fill_(42)
    assert torch.equal(banks[1][0], torch.full_like(tensor, 42))
    assert torch.equal(banks[0][0], banks[2][0])
    assert torch.equal(tensor, torch.arange(8))


def test_banks_preserve_container_aliases_through_records_and_functions():
    class Pair(NamedTuple):
        values: object

    @dataclasses.dataclass
    class Holder:
        values: object

    tensor = torch.arange(4)
    shared = [tensor]
    pair = Pair(shared)
    mapping = {"values": shared, "pair": pair}

    def captured():
        return shared

    args = (
        tensor,
        shared,
        shared,
        pair,
        pair,
        mapping,
        mapping,
        Holder(shared),
        captured,
    )
    banks = clone_input_banks(args, 3)
    for direct, left, right, first, second, a, b, record, fn in banks:
        assert left is right is first.values is a["values"] is record.values is fn()
        assert first is second is a["pair"]
        assert a is b
        assert type(first) is Pair
        assert direct is left[0]
    banks[1][1].append(torch.ones(1))
    banks[1][0].add_(10)
    assert len(banks[0][1]) == len(banks[2][1]) == 1
    assert torch.equal(banks[0][0], banks[2][0])
    assert not torch.equal(banks[0][0], banks[1][0])


@pytest.mark.parametrize("complex_values", [False, True])
def test_banks_preserve_negative_and_conjugate_storage_views(complex_values):
    storage = torch.arange(8, dtype=torch.float32)
    if complex_values:
        storage = torch.complex(storage, storage + 1)
    view = storage[1:7:2]
    negative = torch._neg_view(view)
    views = [view, negative]
    if complex_values:
        views.extend((view.conj(), negative.conj()))
    args = (views, negative, negative)
    banks = clone_input_banks(args, 3)
    for cloned_views, first, second in banks:
        assert first is second is cloned_views[1]
        assert len({v.untyped_storage()._cdata for v in cloned_views}) == 1
        for original, cloned in zip(views, cloned_views, strict=True):
            assert torch.equal(original, cloned)
            assert cloned.is_neg() == original.is_neg()
            assert cloned.is_conj() == original.is_conj()
            assert cloned.stride() == original.stride()
            assert cloned.storage_offset() == original.storage_offset()
    assert len({bank[1].untyped_storage()._cdata for bank in banks}) == 3
    banks[1][0][0].add_(10)
    assert torch.equal(banks[1][1], -banks[1][0][0])
    assert torch.equal(banks[0][1], banks[2][1])
    assert not torch.equal(banks[0][1], banks[1][1])


def test_dataclass_fields_are_counted_as_input_storage():
    @dataclasses.dataclass
    class Holder:
        tensors: object

    storage = torch.arange(64, dtype=torch.int8)
    record = Holder((storage[4:20:2], Holder(storage.view(torch.int16))))
    tensors = _tensor_operands((record,))
    assert len(tensors) == 2
    assert (
        tensor_storage_bytes(_tensor_operands((record,)))
        == storage.untyped_storage().nbytes()
    )


@pytest.mark.parametrize("slotted", [False, True])
def test_dataclass_bank_copies_skip_hooks_and_preserve_stored_state(slotted):
    calls = []

    class Metadata:
        __slots__ = ("metadata",)

    @dataclasses.dataclass(frozen=True, slots=slotted)
    class Holder(Metadata):
        tensor: torch.Tensor
        required: dataclasses.InitVar[int]
        view: torch.Tensor = dataclasses.field(init=False)
        shared: ClassVar[torch.Tensor] = torch.ones(3)

        def __post_init__(self, required):
            calls.append(required)
            object.__setattr__(self, "view", self.tensor[1::2])
            object.__setattr__(self, "metadata", object())

        def __copy__(self):
            raise AssertionError("copy hook must not run")

        def __deepcopy__(self, memo):
            raise AssertionError("deepcopy hook must not run")

    record = Holder(torch.arange(8), 19)
    if not slotted:
        object.__setattr__(record, "extra", object())
    banks = clone_input_banks((record,), 3)
    assert calls == [19]
    assert (
        tensor_storage_bytes(_tensor_operands((record,)))
        == record.tensor.untyped_storage().nbytes()
    )
    for (bank,) in banks:
        assert type(bank) is Holder
        assert bank.metadata is record.metadata
        assert bank.shared is record.shared
        assert bank.view.stride() == record.view.stride()
        assert bank.view.storage_offset() == 1
        assert (
            bank.view.untyped_storage().data_ptr()
            == bank.tensor.untyped_storage().data_ptr()
        )
        if not slotted:
            assert bank.extra is record.extra
    banks[1][0].view.fill_(42)
    assert torch.equal(banks[1][0].tensor[1::2], torch.full((4,), 42))
    assert torch.equal(banks[0][0].tensor, banks[2][0].tensor)


def test_dataclass_banks_include_function_captures_defaults_and_cycles():
    @dataclasses.dataclass
    class Holder:
        tensor: torch.Tensor
        link: object = None

    tensor = torch.arange(4)
    record = Holder(tensor)
    record.link = record

    def captured():
        return record

    def default(value=record):
        return value

    banks = clone_input_banks((tensor, captured, default), 3)
    for direct, captured_fn, default_fn in banks:
        copy = captured_fn()
        assert copy is default_fn()
        assert copy.link is copy
        assert copy.tensor is direct
    assert (
        tensor_storage_bytes(_tensor_operands((captured, default)))
        == tensor.untyped_storage().nbytes()
    )
    banks[1][1]().tensor.add_(100)
    assert torch.equal(banks[0][0], banks[2][0])
    assert not torch.equal(banks[1][0], banks[0][0])


def test_registered_dataclass_keeps_its_pytree_contract():
    @dataclasses.dataclass
    class Registered:
        tensor: torch.Tensor
        metadata: object

    register_pytree_node(
        Registered,
        lambda value: ([value.tensor], value.metadata),
        lambda children, metadata: Registered(*children, metadata),
    )
    tensor = torch.arange(4)
    metadata = torch.ones(8)
    original = Registered(tensor, metadata)
    banks = clone_input_banks((tensor, original, original), 3)
    for direct, record, duplicate in banks:
        assert record is duplicate
        assert record.tensor is direct
        assert record.metadata is metadata
    assert (
        tensor_storage_bytes(_tensor_operands((banks[0][1],)))
        == tensor.untyped_storage().nbytes()
    )


def test_registered_nodes_keep_temporary_leaves_alive_during_bank_clone():
    class Generated:
        def __init__(self, value, tensor=None):
            self.value = value
            self.tensor = tensor

    references = []

    def flatten(node):
        # The first node never owns its generated leaf. Its identity must
        # remain reserved while the later sibling's flatten function runs.
        assert all(reference() is not None for reference in references)
        tensor = torch.tensor([node.value])
        references.append(weakref.ref(tensor))
        return [tensor], node.value

    register_pytree_node(
        Generated,
        flatten,
        lambda children, value: Generated(value, children[0]),
    )
    first, second = clone_input_banks((Generated(3), Generated(7)), 2)[1]
    assert len(references) == 2
    assert torch.equal(first.tensor, torch.tensor([3]))
    assert torch.equal(second.tensor, torch.tensor([7]))
    assert first.tensor is not second.tensor
    assert (
        first.tensor.untyped_storage()._cdata != second.tensor.untyped_storage()._cdata
    )


def test_nonrecord_opaque_objects_remain_unchanged():
    tensor = torch.arange(4)
    opaque = SimpleNamespace(tensor=tensor)
    banks = clone_input_banks((tensor, opaque), 2)
    assert banks[1][1] is opaque
    assert _tensor_operands((opaque,)) == []


def test_cycle_order_outputs_and_mutation_are_independent():
    banks = clone_input_banks((torch.tensor([0]),), 3)
    seen = []

    def fn(value):
        seen.append(value.data_ptr())
        value.add_(1)
        return value.clone()

    cycle = InputBankCycle(fn, banks)
    first = cycle()
    second = cycle()
    assert seen == [bank[0].data_ptr() for bank in banks] * 2
    assert all(torch.equal(value, torch.tensor([1])) for value in first)
    assert all(torch.equal(value, torch.tensor([2])) for value in second)


def test_partial_cycle_raises_without_returning_a_short_result():
    seen = []

    def fn(index):
        seen.append(index)
        if index == 1:
            raise RuntimeError("capture failed")
        return index

    with pytest.raises(RuntimeError, match="capture failed"):
        InputBankCycle(fn, ((0,), (1,), (2,)))()
    assert seen == [0, 1]


@pytest.mark.parametrize(
    "graph,mode,wall_clock",
    [
        ("0", "replay", False),
        ("1", "call", False),
        ("0", "off", False),
        ("1", "replay", True),
        ("1", "off", True),
    ],
)
def test_unsupported_timer_modes_refuse(monkeypatch, graph, mode, wall_clock):
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", graph)
    with pytest.raises(exc.InvalidAPIUsage):
        check_input_bank_mode(3, wall_clock=wall_clock, l2_flush_mode=mode)


@pytest.mark.parametrize(
    "environment,override",
    [("replay", None), ("off", None), ("off", "replay"), ("replay", "off")],
)
def test_worker_uses_whole_cycle_and_normalizes_each_call(
    monkeypatch, environment, override
):
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", environment)
    regime = override or environment
    seen = []
    args = (torch.arange(8)[3:5],)

    def fn(value):
        assert value.storage_offset() == 3
        seen.append(value.data_ptr())
        return value + 1

    monkeypatch.setattr(benchmark_job, "_load_compiled_fn_for_worker", lambda spec: fn)
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: args)
    monkeypatch.setattr(benchmark_job, "steady_graph_dispatch", lambda **kwargs: None)
    monkeypatch.setattr(benchmark_job, "synchronize_device", lambda: None)
    monkeypatch.setattr(
        benchmark_job,
        "make_input_bank_cycle",
        lambda fn, args, count: InputBankCycle(fn, clone_input_banks(args, count)),
    )

    def bench(call, **kwargs):
        assert kwargs["fixed_repetitions"] == 7
        assert kwargs["l2_flush_mode"] == regime
        if regime == "off":
            assert kwargs["pre_timing"] is call
            assert len(kwargs["pre_timing"]()) == 3
        else:
            assert kwargs["pre_timing"] is None
        result = call()
        assert len(result) == 3
        return 12.0

    monkeypatch.setattr(benchmark_job, "do_bench_cuda_graph", bench)
    job = benchmark_job.BenchmarkJob(
        None,
        "unused",
        input_banks=3,
        fixed_repetitions=7,
        mutated_arg_indices=(0,),
        l2_flush_mode=override,
    )
    assert job() == 4.0
    assert len(seen) == (12 if regime == "off" else 6)
    assert all(seen[index : index + 3] == seen[:3] for index in range(0, len(seen), 3))
    assert len(set(seen)) == 3


def test_cache_context_versions_cloning_and_distinguishes_explicit_banks():

    def identity(count):
        search = SimpleNamespace(
            settings=Settings(autotune_benchmark_input_banks=count),
            _benchmark_provider_cls=LocalBenchmarkProvider,
        )
        return _search_policy_hash(search, cute_flash_search_enabled=False)

    assert identity(1) != ""
    assert identity(1) == identity(1)
    assert identity(2) != identity(3)
    assert identity(3) == identity(3)


@pytest.mark.parametrize("regime", ["replay", "off"])
def test_provider_forwards_bank_count_to_worker(monkeypatch, regime):

    captured = []
    provider = object.__new__(benchmark_provider.LocalBenchmarkProvider)
    provider._input_bank_count = 3
    provider._precompile_args_path = "unused"
    provider._subprocess_wrapper_unloadable = False
    provider.mutated_arg_indices = [2]
    provider.settings = Settings(autotune_benchmark_input_banks=3)
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    provider._probe_long_kernel = lambda: False
    provider._worker_compile_timeout = lambda: 30
    provider._get_benchmark_worker = lambda: SimpleNamespace(
        run=lambda job, **kwargs: captured.append(job) or 4.0
    )
    monkeypatch.setattr(benchmark_provider, "_serialize_compiled_fn", lambda fn: None)
    assert (
        provider._run_subprocess_benchmark_job(
            lambda: None, warmup=1, rep=50, args_slot=7, l2_flush_mode=regime
        )
        == 4.0
    )
    assert captured[0].input_banks == 3
    assert captured[0].args_slot == 7
    assert captured[0].mutated_arg_indices == (2,)
    assert captured[0].l2_flush_mode == regime


@pytest.mark.parametrize("flash", [False, True])
def test_resident_bank_cache_does_not_reuse_replay_winner(monkeypatch, flash):

    search = SimpleNamespace(
        settings=Settings(autotune_benchmark_input_banks=3),
        cache_policy=lambda: {"unchanged_policy": True},
        _benchmark_provider_cls=LocalBenchmarkProvider,
    )

    def key(search_mode, final_mode):
        monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", search_mode)
        monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", final_mode)
        return _search_policy_hash(search, cute_flash_search_enabled=flash)

    replay = key("replay", "replay")
    legacy = (
        json.dumps({"unchanged_policy": True}, sort_keys=True, separators=(",", ":"))
        if flash
        else "helion.benchmark_input_banks.v1:3"
    )
    assert replay != hashlib.sha256(legacy.encode()).hexdigest()
    off_keys = (key("off", "replay"), key("replay", "off"), key("off", "off"))
    assert len(set(off_keys)) == 3
    assert replay not in off_keys
    assert key("replay", "replay") == replay
    search.settings = Settings(autotune_benchmark_input_banks=1)
    assert key("replay", "replay") == key("off", "off")


def test_search_keeps_explicit_bank_context():

    settings = Settings(autotune_benchmark_input_banks=3)
    kernel = SimpleNamespace(
        settings=settings,
        config_spec=SimpleNamespace(
            backend=SimpleNamespace(supports_precompile=lambda: True)
        ),
    )
    search = object.__new__(RandomSearch)
    BaseSearch.__init__(search, kernel, ())
    assert search._input_bank_count == 3


@pytest.mark.parametrize("worker_enabled", [False, True])
def test_provider_refuses_any_in_process_fallback(worker_enabled):

    provider = object.__new__(LocalBenchmarkProvider)
    provider._input_bank_count = 3
    provider.log = SimpleNamespace(debug=lambda message: None)
    provider._autotune_metrics = SimpleNamespace(num_configs_tested=0)
    provider._subprocess_benchmark_enabled = lambda: worker_enabled
    provider._benchmark_function_subprocess = lambda *args, **kwargs: None
    with pytest.raises(exc.InvalidAPIUsage, match="in-process benchmarking"):
        provider._benchmark_function(None, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"autotune_benchmark_subprocess": False},
        {"autotune_benchmark_fn": lambda fns: []},
        {"autotune_handoff": True},
    ],
)
def test_provider_rejects_paths_that_hide_or_bypass_banks(kwargs):

    settings = Settings(autotune_benchmark_input_banks=3, **kwargs)
    with pytest.raises(exc.InvalidAPIUsage, match="isolated CUDA"):
        LocalBenchmarkProvider(None, settings, None, (), None, None)


@pytest.mark.parametrize("final_mode", ["replay", "off"])
@pytest.mark.parametrize("target", ["auto", "replay", "0"])
def test_provider_requires_an_explicit_matching_final_target(
    monkeypatch, final_mode, target
):

    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", final_mode)
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", final_mode)
    monkeypatch.setenv("HELION_AUTOTUNE_TARGET_REGIME", target)
    baselines = []

    def baseline(provider):
        baselines.append(provider)
        return None, [], ()

    monkeypatch.setattr(LocalBenchmarkProvider, "_compute_baseline", baseline)
    monkeypatch.setattr(
        LocalBenchmarkProvider, "_compute_effective_tolerances", lambda self: (0, 0)
    )
    monkeypatch.setattr(LocalBenchmarkProvider, "_decide_num_jobs", lambda self: 1)
    monkeypatch.setattr(
        LocalBenchmarkProvider,
        "_subprocess_benchmark_uses_wall_clock",
        lambda self: False,
    )
    settings = Settings(
        autotune_benchmark_input_banks=3, autotune_benchmark_subprocess=True
    )
    spec = SimpleNamespace(backend=SimpleNamespace(get_do_bench=lambda: None))
    kernel = SimpleNamespace(
        env=SimpleNamespace(device=torch.device("cuda")),
        supports_subprocess_benchmark=lambda: True,
    )
    if target == ("0" if final_mode == "off" else "replay"):
        provider = LocalBenchmarkProvider(kernel, settings, spec, (), None, None)
        assert baselines == [provider]
        assert provider._input_bank_count == 3
    else:
        with pytest.raises(exc.InvalidAPIUsage, match="match the final L2 regime"):
            LocalBenchmarkProvider(kernel, settings, spec, (), None, None)
        assert not baselines


def test_memory_budget_refuses_before_copying(monkeypatch):

    storage = SimpleNamespace(data_ptr=lambda: 17, nbytes=lambda: 100)
    tensor = SimpleNamespace(
        device=torch.device("cuda:0"), untyped_storage=lambda: storage
    )
    monkeypatch.setattr(input_banks, "_tensor_operands", lambda args: [tensor, tensor])
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (800, 800))
    copied = []
    monkeypatch.setattr(
        input_banks,
        "clone_input_banks",
        lambda args, count: copied.append(count) or tuple(args for _ in range(count)),
    )
    cycle = input_banks.make_input_bank_cycle(lambda: None, (), 3)
    assert len(cycle.banks) == 3 and copied == [3]
    with pytest.raises(exc.InvalidAPIUsage, match="memory budget"):
        input_banks.make_input_bank_cycle(lambda: None, (), 4)
    assert copied == [3]


def test_empty_function_cells_match_the_clone_contract():
    tensor = torch.ones(1)

    def empty():
        return tensor

    assert empty.__closure__ is not None
    del empty.__closure__[0].cell_contents
    args = (torch.ones(2), empty)
    assert len(_tensor_operands(args)) == 1
    assert clone_input_banks(args, 2)[1][1] is empty


@pytest.mark.parametrize("retained", [40, 60])
def test_cycle_output_budget_counts_live_outputs(monkeypatch, retained):

    allocations = iter((100, 100 + retained))
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda device: next(allocations)
    )
    cycle = InputBankCycle(
        lambda: torch.ones(2),
        ((), (), ()),
        extra_input_bytes=10,
        memory_budget_bytes=60,
    )
    if retained == 60:
        with pytest.raises(InputBankError, match="retained outputs"):
            cycle.warm()
    else:
        assert len(cycle.warm()) == 3


def test_resident_budget_keeps_both_output_cycles_live(monkeypatch):

    outputs = []

    def run():
        output = torch.ones(1)
        outputs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: 10 * sum(output() is not None for output in outputs),
    )
    cycle = InputBankCycle(
        run, ((), (), ()), extra_input_bytes=10, memory_budget_bytes=50
    )
    assert len(cycle.warm()) == 3
    assert all(output() is None for output in outputs)
    with pytest.raises(InputBankMemoryError, match="retained outputs"):
        cycle.warm(retained_cycles=2)
    assert len(outputs) == 9


@pytest.mark.parametrize("isolated", [False, True])
def test_bank_configuration_failure_is_not_a_candidate_score(isolated):

    def fail(*args, **kwargs):
        raise InputBankError("bank memory budget")

    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(autotune_benchmark_input_banks=3)
    provider._input_bank_count = 3
    provider._timeout_retry_repetitions = lambda *args: None
    provider._run_candidate_benchmark_job = fail
    provider._run_subprocess_benchmark_job = fail
    provider._subprocess_benchmark_enabled = lambda: True
    with pytest.raises(InputBankError, match="bank memory budget"):
        if isolated:
            provider.benchmark_isolated([lambda: None], warmup=0, rep=1)
        else:
            provider._benchmark_function_subprocess(None, None)


@pytest.mark.parametrize("isolated", [False, True])
def test_candidate_bank_oom_does_not_discard_other_candidates(isolated):
    failed, good = (lambda: None), (lambda: None)

    def benchmark(fn, **kwargs):
        if fn is failed:
            raise InputBankMemoryError("candidate retained outputs exceed budget")
        return 2.0

    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(
        autotune_benchmark_input_banks=3, autotune_accuracy_check=False
    )
    provider.log = Mock()
    provider._input_bank_count = 3
    provider._timeout_retry_repetitions = lambda *args: None
    provider._run_candidate_benchmark_job = lambda config, fn: benchmark(fn)
    provider._run_subprocess_benchmark_job = benchmark
    provider._subprocess_benchmark_enabled = lambda: True
    provider._subprocess_wrapper_unloadable = False
    provider._last_benchmark_failure_status = None
    provider._worker_failure_config_ids = []
    provider._isolated_candidate_configs = {}
    provider._autotune_metrics = SimpleNamespace(num_worker_failures=0)
    if isolated:
        assert provider.benchmark_isolated([failed, good], warmup=0, rep=1) == [
            IsolatedBenchmarkFailure("error"),
            2.0,
        ]
        # Optional private copies still require the existing whole-phase retry.
        with pytest.raises(InputBankMemoryError, match="retained outputs"):
            provider.benchmark_isolated([failed], warmup=0, rep=1, args_slots=[0])
    else:
        config = Config(block_sizes=[16])
        assert provider._benchmark_function_subprocess(config, failed) == float("inf")
        assert provider._last_benchmark_failure_status == "error"
        assert provider._worker_failure_config_ids == [id(config)]
        assert provider._autotune_metrics.num_worker_failures == 1
        assert provider._benchmark_function_subprocess(config, good) == 2.0


def test_bank_verification_uses_worker_on_an_interleaving_backend():

    search = object.__new__(PatternSearch)
    search._input_bank_count = 3
    search.settings = Settings(autotune_benchmark_input_banks=3)
    search._final_rebenchmark_backend = lambda: SimpleNamespace(
        final_rebenchmark_isolated=lambda: False
    )
    assert search._final_rebenchmark_use_isolated()


@pytest.mark.parametrize(
    "outcome,flash",
    # Every worker outcome is distinct; flash additionally controls whether
    # conclusive failures quarantine earlier accepted configurations.
    [
        ("unavailable", False),
        ("missing_one", False),
        ("missing_one", True),
        ("context", False),
        ("ok", False),
        ("timeout", False),
        ("error", False),
        ("timeout", True),
        ("error", True),
        ("unkillable", False),
    ],
)
def test_mirrored_refinement_uses_complete_worker_cycles(monkeypatch, outcome, flash):

    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "replay")
    members = [
        PopulationMember(lambda: None, [2.0], [], Config(block_sizes=[size]))
        for size in [8, 16]
    ]
    fns = [member.fn for member in members]
    samples = []

    def measure(functions, **kwargs):
        assert kwargs["fixed_repetitions"] == 1
        assert kwargs["l2_flush_mode"] == "replay"
        index = fns.index(functions[0])
        samples.append(index)
        if outcome == "unavailable":
            return None
        if index == 1:
            if outcome == "missing_one":
                return [None]
            if outcome == "context":
                raise InputBankError("unsupported bank context")
            if outcome == "unkillable":
                error = BenchmarkWorkerUnkillable("worker survived kill")
                error.fn_index = 0  # The provider receives a one-function batch.
                raise error
            if outcome in ("timeout", "error"):
                return [IsolatedBenchmarkFailure(outcome)]
        return [2.0 - index]

    provider = object.__new__(LocalBenchmarkProvider)
    provider.register_isolated_candidates = Mock()
    provider.private_argument_copies_affordable = lambda count: False
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    provider.benchmark_isolated = measure
    search = object.__new__(PatternSearch)
    search._input_bank_count = 3
    search.config_spec = SimpleNamespace(cute_flash_search_enabled=flash)
    search.settings = Settings(autotune_benchmark_input_banks=3)
    search.benchmark_provider = provider
    search._repeat_for_target_ms = lambda *args: 10
    search._apply_rebenchmark_timings = Mock()
    search._fallback_excluded_configs = set()
    monkeypatch.setattr(
        base_search, "mirrored_bench_generic", Mock(side_effect=AssertionError)
    )
    if outcome == "context":
        with pytest.raises(InputBankError, match="unsupported bank context"):
            search.mirrored_rebenchmark(members, desc="test")
        search._apply_rebenchmark_timings.assert_not_called()
        return
    if outcome == "unkillable":
        with pytest.raises(BenchmarkWorkerUnkillable) as caught:
            search.mirrored_rebenchmark(members, desc="test")
        assert caught.value.fn_index == 1
        assert search._fallback_excluded_configs == {members[1].config}
        search._apply_rebenchmark_timings.assert_not_called()
        return
    trace = search.mirrored_rebenchmark(members, desc="test")
    assert trace.orders == [[0, 1], [1, 0], [1, 0], [0, 1]]
    if outcome in ("unavailable", "missing_one"):
        all_missing = outcome == "unavailable"
        assert samples == ([0, 1] if all_missing else [0, 1, 0, 0, 0])
        assert trace.medians_ms == [float("inf") if all_missing else 2.0, float("inf")]
        assert trace.statuses == ["incomplete" if all_missing else "ok", "incomplete"]
        applied = search._apply_rebenchmark_timings.call_args
        assert applied.args == (members, [2.0, 2.0])
        assert applied.kwargs["trace_results"] == [None if all_missing else 2.0, None]
        assert applied.kwargs["failure_statuses"] == [None, None]
        assert not search._fallback_excluded_configs
        return
    if outcome in ("timeout", "error"):
        assert samples == [0, 1, 0, 0, 0]
        assert trace.medians_ms == [2.0, float("inf")]
        applied = search._apply_rebenchmark_timings.call_args
        expected = 2.0 if outcome == "timeout" and not flash else float("inf")
        assert applied.args == (members, [2.0, expected])
        assert applied.kwargs["failure_statuses"] == [None, outcome if flash else None]
        if not flash:
            assert applied.kwargs["trace_results"][1] == IsolatedBenchmarkFailure(
                outcome
            )
        return
    assert samples == [index for order in trace.orders for index in order]
    assert trace.medians_ms == [2.0, 1.0]
    assert trace.calls_per_sample == 3
    assert trace.total_calls == 12
    applied = search._apply_rebenchmark_timings.call_args
    assert applied.args == (members, [2.0, 1.0])
    assert applied.kwargs["failure_statuses"] == [None, None]


@pytest.mark.parametrize(
    "banks,free_kib,expected", [(1, 4, True), (3, 9, False), (3, 10, True)]
)
def test_private_copies_leave_the_bank_budget(monkeypatch, banks, free_kib, expected):

    tensor = Mock(spec=torch.Tensor)
    tensor.device = torch.device("cuda")
    tensor.is_cuda = True
    tensor.untyped_storage.return_value.data_ptr.return_value = 1234
    tensor.untyped_storage.return_value.nbytes.return_value = 1024
    provider = object.__new__(LocalBenchmarkProvider)
    provider.args = (tensor, tensor)
    provider._input_bank_count = banks
    provider._baseline_output = None
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_kib * 1024, 0))
    assert provider.private_argument_copies_affordable(2) == expected


def test_missing_rebenchmark_preserves_observations(monkeypatch):
    member = PopulationMember(
        lambda: None, [2.0], [], Config(block_sizes=[8]), status="ok"
    )
    search = object.__new__(PatternSearch)
    search.log = SimpleNamespace(trace_enabled=False)
    search.best_perf_so_far = 2.0
    search._invalidate_cute_flash_rebenchmark_failures = Mock(return_value=[])
    search._refresh_benchmarked_members_after_rebenchmark = Mock()
    monkeypatch.setattr(base_search, "_refresh_handoff", Mock())
    search._apply_rebenchmark_timings([member], [2.0], trace_results=[None])
    assert member.perfs == [2.0]
    assert member.status == "ok"
    assert search.best_perf_so_far == 2.0
    search._invalidate_cute_flash_rebenchmark_failures.assert_called_once_with(
        [member], [None]
    )


@pytest.mark.parametrize("regime", ["replay", "off"])
def test_private_copies_budget_retained_outputs_and_captured_aliases(
    monkeypatch, regime
):

    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", regime)
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", regime)

    def tensor(pointer, size):
        value = Mock(spec=torch.Tensor)
        value.device = torch.device("cuda")
        value.is_cuda = True
        value.untyped_storage.return_value.data_ptr.return_value = pointer
        value.untyped_storage.return_value.nbytes.return_value = size
        return value

    values = tensor(1, 1024)
    alias = tensor(1, 1024)
    output = tensor(2, 512)

    def epilogue():
        return values, alias

    provider = object.__new__(LocalBenchmarkProvider)
    provider.args = (epilogue,)
    provider._input_bank_count = 3
    provider._baseline_output = (output, output)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * 1024, 0))
    # Eight input copies leave 12 KiB free. The 3.5 KiB bank cycle
    # exceeds both the total headroom and its 3 KiB bank budget.
    assert not provider.private_argument_copies_affordable(8)
    # Resident capture additionally retains the primer's outputs: 5 KiB rather
    # than 3.5 KiB, exceeding the 4.5 KiB budget even with two private copies.
    assert provider.private_argument_copies_affordable(2) == (regime == "replay")
    # With two private copies, total headroom fits, but retaining larger
    # outputs needs 5 KiB (replay) or 8 KiB (off), above the 4.5 KiB budget.
    output.untyped_storage.return_value.nbytes.return_value = 1024
    assert not provider.private_argument_copies_affordable(2)
    provider._baseline_output = None
    assert provider.private_argument_copies_affordable(8)


@pytest.mark.parametrize("phase", ["mirrored", "isolated"])
@pytest.mark.parametrize(
    "failure", ["budget", "second_budget", "unsupported", "kernel"]
)
def test_private_bank_failure_restarts_the_whole_phase(monkeypatch, phase, failure):
    # Both consumers must discard an already accepted private sample; the
    # independent failure axis pins budget-only, single-retry behavior.

    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "replay")
    members = [
        PopulationMember(lambda: None, [2.0], [], Config(block_sizes=[size]))
        for size in (8, 16)
    ]
    fns = [member.fn for member in members]
    provider = object.__new__(LocalBenchmarkProvider)
    worker = Mock()
    provider._benchmark_worker = worker
    provider.log = Mock()
    provider.register_isolated_candidates = Mock()
    provider.private_argument_copies_affordable = lambda count: (
        not provider._private_argument_copies_disabled
    )
    provider.steady_window_benchmark_available = lambda: True
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    samples = []
    error = {
        "budget": InputBankMemoryError("outputs exceed budget"),
        "second_budget": InputBankMemoryError("outputs exceed budget"),
        "unsupported": InputBankError("unsupported context"),
        "kernel": RuntimeError("kernel failed"),
    }[failure]

    def benchmark(functions, **kwargs):
        slots = kwargs.get("args_slots", [None] * len(functions))
        results = []
        for fn, slot in zip(functions, slots, strict=True):
            index = fns.index(fn)
            samples.append((index, slot))
            if index == 1 and (slot is not None or failure == "second_budget"):
                raise error
            results.append(100.0 if slot is not None else 2.0 - index)
        return results

    provider.benchmark_isolated = benchmark
    search = object.__new__(PatternSearch)
    search._input_bank_count = 3
    search.settings = Settings(autotune_benchmark_input_banks=3)
    search.benchmark_provider = provider
    search._apply_rebenchmark_timings = Mock()
    search._fallback_excluded_configs = set()

    def run():
        if phase == "mirrored":
            return search._mirrored_bank_rebenchmark(members, 10, "test", 1.0, 2.0)
        return search._isolated_rebenchmark_results(
            members, rep=1, desc="test", candidate_private_args=True
        )

    if failure != "budget":
        with pytest.raises(type(error)) as caught:
            run()
        assert caught.value is error
        search._apply_rebenchmark_timings.assert_not_called()
    else:
        result = run()
        if phase == "mirrored":
            assert result.medians_ms == [2.0, 1.0]
            assert all(value < 100 for row in result.elapsed_ms for value in row)
        else:
            assert result == [2.0, 1.0]
    assert samples[:2] == [(0, 0), (1, 1)]
    if failure in ("budget", "second_budget"):
        worker.shutdown.assert_called_once_with()
        assert provider._benchmark_worker is None
        assert provider._private_argument_copies_disabled
        provider.log.warning.assert_called_once()
        assert str(error) in provider.log.warning.call_args.args[0]
        assert (
            "Discarding this phase's readings" in provider.log.warning.call_args.args[0]
        )
        assert samples[2:4] == [(0, None), (1, None)]
        assert all(slot is None for _, slot in samples[2:])
    else:
        worker.shutdown.assert_not_called()
        assert len(samples) == 2


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, RuntimeError])
def test_optional_copy_failure_is_classified_before_any_timing(
    monkeypatch, count, error_type
):

    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "replay")
    fn = Mock()
    unload = Mock()
    timer = Mock()
    error = error_type("optional copy failed")
    monkeypatch.setattr(benchmark_job, "_load_compiled_fn_for_worker", lambda spec: fn)
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", unload)
    monkeypatch.setattr(
        benchmark_job, "load_private_kernel_args", Mock(side_effect=error)
    )
    monkeypatch.setattr(benchmark_job, "do_bench_cuda_graph", timer)
    expected = (
        InputBankMemoryError
        if count > 1 and error_type is torch.OutOfMemoryError
        else error_type
    )
    with pytest.raises(expected) as caught:
        benchmark_job.BenchmarkJob(None, "unused", input_banks=count, args_slot=0)()
    if expected is error_type:
        assert caught.value is error
    fn.assert_not_called()
    timer.assert_not_called()
    unload.assert_called_once_with(fn)


@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, RuntimeError])
def test_retained_output_oom_is_a_retryable_budget_failure(monkeypatch, error_type):

    error = error_type("output allocation failed")
    fn = Mock(side_effect=error)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    cycle = InputBankCycle(fn, ((), (), ()), memory_budget_bytes=1024)
    expected = (
        InputBankMemoryError if error_type is torch.OutOfMemoryError else error_type
    )
    with pytest.raises(expected) as caught:
        cycle.warm()
    if expected is error_type:
        assert caught.value is error
    fn.assert_called_once_with()
