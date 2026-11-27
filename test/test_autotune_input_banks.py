from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import json
import logging
import pickle
import statistics
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NamedTuple
from typing import cast
from unittest.mock import Mock
import weakref

import pytest
import torch
from torch.utils._pytree import register_pytree_node

from helion import exc
from helion._compiler import source_location
from helion._testing.search_harness import mock_kernel
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import base_search
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner import input_banks
from helion.autotuner import local_cache
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.benchmarking import central_mean
from helion.autotuner.input_banks import InputBankCycle
from helion.autotuner.input_banks import InputBankError
from helion.autotuner.input_banks import InputBankMemoryError
from helion.autotuner.input_banks import _tensor_operands
from helion.autotuner.input_banks import check_input_bank_mode
from helion.autotuner.input_banks import clone_input_banks
from helion.autotuner.input_banks import tensor_storage_bytes
from helion.autotuner.local_cache import _search_policy_hash
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.search_space_logger import canonical_config_id
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from helion.autotuner.base_search import _AutotunableKernel

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
    provider._autotune_metrics = SimpleNamespace(
        num_configs_tested=0,
        num_worker_job_exceptions=0,
        num_isolated_rebenchmark_exceptions=0,
    )
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


@pytest.mark.parametrize("regime", ["0", "replay"])
def test_bank_search_keeps_the_event_timer_below_the_short_kernel_bound(
    monkeypatch, regime
):
    # A search whose initial population reads below its backend's
    # short-kernel bound moves its candidates to the CUDA activity trace
    # (``PopulationBasedSearch._settle_candidate_timer``). A bank search
    # never does, however short its kernel: a bank cycle is timed between
    # two event nodes of one captured graph, which exists only under
    # HELION_BENCHMARK_CUDAGRAPH=1 in the replay or the off regime
    # (``check_input_bank_mode``), and a search in either keeps the events.
    # Its readings stay central means of whole cycles.

    def settled(banks: int) -> tuple[PopulationBasedSearch, list[PopulationMember]]:
        """A search over ``banks`` banks, on a backend whose bound is 64 us
        and a worker that can trace, after an initial population whose best
        config reads 23.8 us."""
        kernel = mock_kernel(Settings(autotune_log_level=logging.CRITICAL))
        kernel.config_spec.backend.autotune_short_kernel_below_us = lambda spec: 64.0
        search = PopulationBasedSearch(cast("_AutotunableKernel", kernel), ())
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider._benchmark_worker = Mock()
        provider.config_spec = kernel.config_spec
        provider._effective_source_results = {}
        provider._input_bank_count = banks
        search._input_bank_count = banks
        search.benchmark_provider = provider
        search._kernel_timer_usable = Mock(return_value=True)

        def read_again(members: list[PopulationMember], **kwargs: object) -> None:
            for timed in members:
                timed.perfs.append(timed.perf - 0.0046)

        search.rebenchmark = Mock(side_effect=read_again)
        search.log = Mock()
        search.best_perf_so_far = 0.0238
        population = [PopulationMember(lambda: None, [0.0238], (0,), Config())]
        search._settle_candidate_timer(population)
        return search, population

    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", regime)
    check_input_bank_mode(3)
    search, population = settled(3)
    assert search.short_kernel_below_us == 64.0
    assert not search.traces_candidates
    assert not search.benchmark_provider.trace_search_candidates
    # No profiling session in the worker, and nothing is read again.
    search._kernel_timer_usable.assert_not_called()
    search.rebenchmark.assert_not_called()
    assert population[0].perfs == [0.0238]
    # The bank regime keeps the events, not the reading or the bound: the
    # same search with one bank in the default regime settles on the trace.
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH")
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH")
    search, population = settled(1)
    assert search.traces_candidates
    assert search.benchmark_provider.trace_search_candidates
    search._kernel_timer_usable.assert_called_once_with()
    search.rebenchmark.assert_called_once()
    assert population[0].perfs == [pytest.approx(0.0192)]


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
        # The job reads every replay's event time and averages them itself.
        assert kwargs["return_mode"] == "all"
        if regime == "off":
            assert kwargs["pre_timing"] is call
            assert len(kwargs["pre_timing"]()) == 3
        else:
            assert kwargs["pre_timing"] is None
        result = call()
        assert len(result) == 3
        # Values of the timer grid, a step within the stall margin of the
        # median: the median alone would read 12.0, 4.0 per call.
        return [*([12.0] * 6), *([15.0] * 4)]

    monkeypatch.setattr(benchmark_job, "do_bench_cuda_graph", bench)
    job = benchmark_job.BenchmarkJob(
        None,
        "unused",
        input_banks=3,
        fixed_repetitions=7,
        mutated_arg_indices=(0,),
        l2_flush_mode=override,
    )
    assert job() == pytest.approx(4.375)
    assert len(seen) == (12 if regime == "off" else 6)
    assert all(seen[index : index + 3] == seen[:3] for index in range(0, len(seen), 3))
    assert len(set(seen)) == 3


def test_central_mean_resolves_the_timer_grid_and_drops_stalled_replays():
    # Event times of a captured cycle sit on a grid; a hundred replays of
    # three candidates whose share of samples on the slower step differs.
    step = 2.048
    low, high = 40 * step, 41 * step
    shares = (30, 40, 60)
    windows = [[low] * (100 - share) + [high] * share for share in shares]
    medians = [statistics.median(window) for window in windows]
    # The median ties the first two and puts a whole step between the last
    # two, whose times differ by a fifth of one.
    assert medians == [low, low, high]
    means = [central_mean(window) for window in windows]
    assert means == sorted(means) and len(set(means)) == 3
    assert means[2] - means[1] == pytest.approx(0.25 * step)
    # Stalled replays are left out before a tenth of the rest is dropped at
    # each end: nine of them leave the reading of the other 91 samples,
    # where one moves the plain mean past the next candidate.
    stalled = [*windows[0][:91], *([100 * high] * 9)]
    assert central_mean(stalled) == pytest.approx(central_mean(windows[0][:91]))
    assert central_mean(windows[0][:91]) == pytest.approx(
        statistics.fmean(windows[0][9:82])
    )
    assert statistics.fmean([*windows[0][:99], 100 * high]) > means[2]
    # A window of fewer than ten samples has no tenth to drop and is averaged
    # whole, so a few replays still read the share of samples on each step:
    # four values from three replays, where their median reads two.
    short = [central_mean([low] * (3 - share) + [high] * share) for share in range(4)]
    assert short == pytest.approx([low + share * step / 3 for share in range(4)])
    assert central_mean([low, high]) == pytest.approx((low + high) / 2)
    assert central_mean([low]) == low
    # A replay more than a quarter of the window's median above that median
    # stalled and is left out, however few samples remain: one of a long
    # cycle, two of nine, and one of two, which a median could not drop.
    assert central_mean([6.0, 6.5, 12.0]) == 6.25
    assert central_mean([low, high, high, 100 * high]) == pytest.approx(
        (low + 2 * high) / 3
    )
    assert central_mean([low] * 7 + [2 * high, 100 * high]) == low
    assert central_mean([low, 2 * low]) == low
    # The margin itself is not a stall.
    assert central_mean([4.0, 4.0, 5.0]) == pytest.approx(13 / 3)
    assert central_mean([4.0, 4.0, 5.5]) == 4.0
    # From ten samples that did not stall up, the lowest and the highest
    # tenth are left out.
    assert central_mean([low] * 9 + [100 * high]) == low
    assert central_mean([*range(70, 80)]) == pytest.approx(74.5)
    assert central_mean([60.0, *([75.0] * 8), 90.0]) == 75.0
    assert central_mean([60.0, *([75.0] * 8)]) == pytest.approx(660 / 9)


@pytest.mark.parametrize("count", [9, 10, 11, 12, 20, 21])
def test_central_mean_drops_stalled_replays_at_every_sample_count(count):
    # A window with more stalls than a tenth of its samples: a rule that
    # drops by rank from ten samples up would average one of them in (two
    # stalls of ten read 200 where the eight others read 75).
    assert central_mean([75.0] * (count - 2) + [1075.0] * 2) == 75.0
    # ... and on the timer's grid, where the other samples take two values:
    # the reading is the one of the window without its stalls.
    step = 2.048
    low, high = 40 * step, 41 * step
    quiet = [low] * (count - 5) + [high] * 3
    assert central_mean([*quiet, 100 * high, 100 * high]) == pytest.approx(
        central_mean(quiet)
    )
    assert low < central_mean(quiet) < high


def test_central_mean_reads_low_where_a_timer_step_exceeds_the_margin():
    # A cycle of less than four steps per replay: a sample one step above
    # the median is more than a quarter above it and counts as stalled, so
    # the reading is the mean of the samples at or below the median.
    step = 2.048
    window = [2 * step] * 3 + [3 * step] * 10 + [4 * step] * 7
    assert statistics.median(window) == 3 * step
    assert central_mean(window) == pytest.approx(
        statistics.fmean([2 * step] * 2 + [3 * step] * 9)
    )
    assert central_mean(window) < statistics.median(window)
    # From four steps up the neighbouring step is inside the margin.
    window = [4 * step] * 12 + [5 * step] * 8
    assert central_mean(window) == pytest.approx(
        statistics.fmean([4 * step] * 10 + [5 * step] * 6)
    )


def test_cache_context_versions_cloning_and_distinguishes_explicit_banks():

    def identity(count):
        search = SimpleNamespace(
            settings=Settings(autotune_benchmark_input_banks=count),
            _benchmark_provider_cls=LocalBenchmarkProvider,
            _benchmark_prelude=None,
        )
        return _search_policy_hash(search, cute_flash_search_enabled=False)

    assert identity(1) != ""
    assert identity(1) == identity(1)
    assert identity(2) != identity(3)
    assert identity(3) == identity(3)


@pytest.mark.parametrize("flash", [False, True])
def test_bank_timing_version_rekeys_every_bank_search_and_no_other(monkeypatch, flash):
    # A bank cycle's statistic, the final selection's tie-break and the
    # re-timing rule decide which configs a bank search visits and returns,
    # so their version is part of that search's cache identity: on every
    # backend (the keys of a CuTe flash kernel and of every other kernel
    # alike) and in the replay and the resident regime. Results cached under
    # another version are not served. A one-bank search's identity carries
    # no such version.

    def flash_policy(count):
        # A flash search's own policy records its bank count.
        return {"unchanged_policy": True, "banks": count}

    def identity(count, regime="replay"):
        monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", regime)
        monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", regime)
        search = SimpleNamespace(
            settings=Settings(autotune_benchmark_input_banks=count),
            cache_policy=lambda: flash_policy(count),
            _benchmark_provider_cls=LocalBenchmarkProvider,
            _benchmark_prelude=None,
        )
        return _search_policy_hash(search, cute_flash_search_enabled=flash)

    def keyed(count, version=None):
        # The identity of a replay search: before the timing version
        # existed, or under ``version`` of the bank timing policy.
        payload = (
            json.dumps(flash_policy(count), sort_keys=True, separators=(",", ":"))
            if flash
            else ""
            if count == 1
            else f"helion.benchmark_input_banks.v1:{count}"
        )
        stamp = b"" if version is None else b"helion.bank_timing.v%d\0" % version
        return hashlib.sha256(
            b"helion.input_clone.v6\0" + stamp + payload.encode()
        ).hexdigest()

    current = local_cache._BANK_TIMING_VERSION
    assert identity(1) == keyed(1)
    assert identity(3) != keyed(3)
    assert identity(3) == keyed(3, current)
    assert identity(3) not in {keyed(3, current + step) for step in (-1, 1, 2)}
    before = {
        (count, regime): identity(count, regime)
        for count in (1, 2, 3)
        for regime in ("replay", "off")
    }
    assert len({key for (count, _), key in before.items() if count > 1}) == 4
    monkeypatch.setattr(
        local_cache, "_BANK_TIMING_VERSION", local_cache._BANK_TIMING_VERSION + 1
    )
    for (count, regime), key in before.items():
        assert (identity(count, regime) == key) is (count == 1)


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
        _benchmark_prelude=None,
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
    search.population = list(members)
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


# Re-timings of a search that times bank cycles: two windows of the
# candidates' length per member instead of one long window
# (``PopulationBasedSearch._bank_cycle_retimings``).


def _retiming_search(banks, benchmark_isolated, search_type=PatternSearch):
    """A bare search with what a re-timing reads, on a stand-in worker that
    times what ``benchmark_isolated`` says."""
    settings = Settings(
        autotune_benchmark_input_banks=banks, autotune_log_level=logging.CRITICAL
    )
    search = object.__new__(search_type)
    search.settings = settings
    search._input_bank_count = banks
    search.log = Mock(trace_enabled=False)
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.config_spec = mock_kernel(settings).config_spec
    search.benchmark_provider = SimpleNamespace(
        benchmark_isolated=benchmark_isolated,
        mutated_arg_indices=[],
        settles_source_aliases=lambda: False,
    )
    search.population = []
    search._kernels = {}
    search._benchmarked_members = {}
    search._pinned_finalist_configs = set()
    search._pinned_finalist_members = {}
    search._fallback_excluded_configs = set()
    search.best_perf_so_far = 2.0
    return search


def _retimed_members(*firsts):
    """Members with one reading each, and the letter that names each."""
    members = [
        PopulationMember(lambda: None, [first], [], Config(block_sizes=[index + 1]))
        for index, first in enumerate(firsts)
    ]
    names = {id(member.fn): "abcdefgh"[index] for index, member in enumerate(members)}
    return members, names


def _retiming_debug_lines(search):
    """What a re-timing logged at debug level: the sweeps' summary line (not
    there when it was logged at info level) and the line of raw windows."""
    lines = [
        call.args[0]() if callable(call.args[0]) else call.args[0]
        for call in search.log.debug.call_args_list
    ]
    return [line for line in lines if "re-timed in" in line or "windows in ms" in line]


@pytest.mark.parametrize("banks", [3, 1])
def test_bank_retiming_reads_two_candidate_windows_and_their_median(monkeypatch, banks):
    # A one-bank search re-times a member in one window of the re-timing
    # target and reads that window (the control). A search that times bank
    # cycles takes two windows of the candidates' length, the second sweep
    # in reverse order, and a member reads the median of the reading it had
    # and the two. A window that sits apart does not become the member's
    # reading, on whichever side of the reading the other window lies: above
    # it (a reads 2.02, not the windows' mean 2.16; e reads 2.28 with a
    # window far below) or across it (b and h keep theirs). Neither does an
    # earlier reading that both windows contradict (d). A reading equal to
    # its lower window (f) or its upper one (g) stays.

    monkeypatch.delenv("HELION_CAP_REBENCHMARK_REPEAT", raising=False)
    members, names = _retimed_members(2.00, 2.10, 2.20, 1.90, 2.30, 2.50, 2.40, 2.60)
    script = {
        "a": [2.30, 2.02],
        "b": [2.05, 2.50],
        "c": [2.21, 2.19],
        "d": [2.41, 2.39],
        "e": [1.70, 2.28],
        "f": [2.50, 2.56],
        "g": [2.34, 2.40],
        "h": [2.90, 2.58],
    }
    first_windows = [[windows[0]] for windows in script.values()]
    calls = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        order = "".join(names[id(fn)] for fn in fns)
        calls.append((order, warmup, rep, fresh_process))
        return [script[name].pop(0) for name in order]

    search = _retiming_search(banks, benchmark_isolated)
    search.best_perf_so_far = 1.90
    search.rebenchmark(members, desc="verifying")
    if banks == 1:
        assert calls == [("abcdefgh", 1, 200, False)]
        assert [m.perfs[1:] for m in members] == first_windows
        return
    window = benchmark_provider.CANDIDATE_WINDOW_MS
    assert window == 50
    assert calls == [("abcdefgh", 1, window, False), ("hgfedcba", 1, window, False)]
    assert [m.perfs[1:] for m in members] == [
        [2.02],
        [2.10],
        [2.20],
        [2.39],
        [2.28],
        [2.50],
        [2.40],
        [2.60],
    ]
    # The reading a member had stays its first; the lowest stays the search's.
    assert [m.perfs[0] for m in members] == [
        2.00,
        2.10,
        2.20,
        1.90,
        2.30,
        2.50,
        2.40,
        2.60,
    ]
    assert search.best_perf_so_far == 1.90


def test_bank_retiming_under_a_drift_reads_each_config_in_its_earlier_window():
    # What the rule gives on a device that drifts while the two sweeps run:
    # the k-th of their twelve windows reads k - 1 twelfths of 2% above the
    # six members' common reading. Both windows of a member then lie above
    # its reading and it takes the lower one, the first sweep's: six
    # identical configs read in the order of that sweep, the last 0.83%
    # above the first, which keeps its reading. The median does not cancel
    # a drift (the mean of a member's two windows would read all six
    # equal); it is there for the single window that sits apart, which is
    # what the device shows (``_BANK_CYCLE_RETIMING_ROUNDS``).

    members, names = _retimed_members(*[2.00] * 6)
    taken = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        readings = []
        for fn in fns:
            readings.append(2.00 * (1.0 + 0.02 * len(taken) / 12))
            taken.append(names[id(fn)])
        return readings

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="verifying")
    assert "".join(taken) == "abcdef" + "fedcba"
    readings = [member.perfs[-1] for member in members]
    assert readings == [2.00 * (1.0 + 0.02 * place / 12) for place in range(6)]
    assert readings[0] == 2.00
    assert readings[-1] / readings[0] - 1.0 == pytest.approx(0.02 * 5 / 12)


def test_bank_retiming_reads_a_member_without_a_reading_from_its_windows():
    # A member whose reading is not finite (a) or that has none (d) has
    # nothing for its windows to confirm: it reads their mean. Neither it
    # nor a member whose reading is zero (e) or whose windows do not read a
    # finite time (f) is among the readings the sweep's shift is taken
    # from, which leaves b (+0.0%) and c (+1.0%).

    members, names = _retimed_members(float("inf"), 2.00, 2.10, 9.99, 0.0, 2.00)
    members[3].perfs.clear()
    script = {
        "a": [2.20, 2.40],
        "b": [1.995, 2.005],
        "c": [2.111, 2.131],
        "d": [2.50, 2.70],
        "e": [1.00, 3.00],
        "f": [float("inf"), 2.02],
    }

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        return [script[names[id(fn)]].pop(0) for fn in fns]

    search = _retiming_search(3, benchmark_isolated)
    results = search._isolated_rebenchmark_results(
        members, rep=200, desc="test", candidate_private_args=False
    )
    assert results == [
        pytest.approx(2.30),
        2.00,
        2.111,
        pytest.approx(2.60),
        1.00,
        2.02,
    ]
    assert not search.log.call_args_list
    summary, raw = _retiming_debug_lines(search)
    # (the windows of a, d, e and f sit more than 1% apart; c's 0.95%)
    assert summary == (
        "test: 2 configs re-timed in 2 sweeps of 50 ms windows, which read +0.5% "
        "(median) from the readings they repeat; the two windows of 4 of 6 "
        "configs (67%) sit more than 1% apart"
    )
    assert f"{canonical_config_id(members[0].config)}: none | 2.2 2.4 | " in raw
    assert f"{canonical_config_id(members[3].config)}: none | 2.5 2.7 | 2.6" in raw
    assert f"{canonical_config_id(members[4].config)}: 0.0 | 1.0 3.0 | 1.0" in raw
    assert f"{canonical_config_id(members[5].config)}: 2.0 | inf 2.02 | 2.02" in raw


# First readings (50 ms windows) and verification re-timings (one 200 ms
# window each) of twelve configs in the initial population of a cold search
# of a paged attention kernel on a B200, in us per call. The first is the
# seed that read best and went on to win the search's final selection; the
# re-timing ranked it seventh, so none of the five paths started from it.
_LOGGED_VERIFICATION_US = (
    (260.90, 279.86),
    (261.81, 281.91),
    (263.48, 282.03),
    (264.10, 267.98),
    (264.15, 276.32),
    (264.37, 288.43),
    (264.63, 277.87),
    (264.89, 289.27),
    (265.05, 280.27),
    (267.31, 279.69),
    (267.93, 276.29),
    (269.84, 277.63),
)
# What the windows of that search's final selection moved by from one to the
# next (us): a window inside the device's boost period reads the kernel to a
# timer step.
_BOOST_WINDOW_STEPS_US = (0.2, 1.2, -0.8, -0.8, 0.3, -0.5)


@pytest.mark.parametrize("banks", [3, 1])
@pytest.mark.parametrize("search_type", [PatternSearch, LFBOTreeSearch])
def test_bank_search_starts_its_paths_from_the_best_candidate_windows(
    monkeypatch, search_type, banks
):
    # The stand-in worker encodes what sweeps of re-timing windows measured
    # on that device, on the parent of this change: over 46 configs of this
    # kernel, windows of 50 ms read the configs' first readings to a timer
    # step (1.002 times them in the median) and windows of 200 ms, which
    # run into the driver's power cap, 1.083 times them. So a window of the
    # candidates' length reads a config's first reading to a timer step and
    # a longer one the logged 200 ms re-timing. It tests the rule's
    # arithmetic through the real ``_autotune``; that a device reads so is
    # what those sweeps showed, not this test. With one bank the search
    # re-times in the long window
    # and starts no path from the seed that read best (the control, and what
    # the logged search did); with bank cycles its windows stay the
    # candidates' and the paths start from the best of them, the seed first.

    monkeypatch.delenv("HELION_CAP_REBENCHMARK_REPEAT", raising=False)
    members = [
        PopulationMember(
            lambda: None,
            [first / 1000.0],
            (index,),
            Config(block_sizes=[index + 1]),
            status="ok",
        )
        for index, (first, _) in enumerate(_LOGGED_VERIFICATION_US)
    ]
    logged = {
        id(member.fn): pair
        for member, pair in zip(members, _LOGGED_VERIFICATION_US, strict=True)
    }
    steps = itertools.cycle(_BOOST_WINDOW_STEPS_US)
    windows = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        windows.append(rep)
        return [
            (logged[id(fn)][1] if rep > 100 else logged[id(fn)][0] + next(steps))
            / 1000.0
            for fn in fns
        ]

    search = _retiming_search(banks, benchmark_isolated, search_type)
    search.best_perf_so_far = members[0].perf
    search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
    search.copies = 5
    search.max_generations = 1
    search.similarity_penalty = 1.0
    search.compile_timeout_lower_bound = 0.0
    search.compile_timeout_quantile = 0.0
    search.log.autotune_tracing.return_value = contextlib.nullcontext()
    search.config_gen = Mock(encode_config=lambda flat: flat)
    search._autotune_metrics = AutotuneMetrics()
    by_flat = {member.flat_values: member for member in members}
    starts = []

    def path(*args, **kwargs):
        starts.append(next(arg for arg in args if isinstance(arg, PopulationMember)))
        return iter(())

    stubs = {
        "_generate_initial_population_flat": Mock(return_value=[*by_flat]),
        "make_unbenchmarked": Mock(side_effect=by_flat.get),
        "benchmark_initial_population": Mock(),
        "_random_fallback_population_target": Mock(return_value=None),
        "set_adaptive_compile_timeout": Mock(),
        "capture_compiler_seed_members": Mock(),
        "_pattern_search_from": Mock(side_effect=path),
        "_pruned_pattern_search_from": Mock(side_effect=path),
        "_budgeted_range": lambda *args: iter(range(*args)),
        "_finalize": Mock(return_value=members[0].config),
        # LFBO only:
        "_append_training_sample": Mock(),
        "_fit_surrogate": Mock(),
        "_run_flash_structural_qualification": Mock(return_value=0),
        "_autotune_budget_exceeded_across_ranks": Mock(return_value=False),
        "_polish_descent": Mock(),
    }
    for name, stub in stubs.items():
        setattr(search, name, stub)
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.check_population_consistency", Mock()
    )

    search._autotune()

    seed = members[0]
    assert len(starts) == 5
    if banks == 1:
        assert windows == [200]
        # the five lowest 200 ms windows: first-reading ranks 4, 11, 5, 12, 7
        assert starts == [members[i] for i in (3, 10, 4, 11, 6)]
        assert seed not in starts
        return
    assert windows == [50, 50]
    assert starts[0] is seed
    # its first reading sits between its two windows (261.10 and 260.40 us)
    assert seed.perfs == [seed.perfs[0], seed.perfs[0]]
    # The four best first readings start a path, in their order. The next
    # five first readings sit within one timer step of each other (264.15
    # to 265.05 us), so which of them is fifth is that step's choice.
    assert starts[:4] == members[:4]
    assert any(starts[4] is member for member in members[4:9])


@pytest.mark.parametrize(
    "outcome", ["error", "timeout", "missing", "unavailable", "unkillable"]
)
def test_bank_retiming_settles_a_member_whose_window_gave_no_reading(
    monkeypatch, outcome
):
    # Each way a window can end without a reading. A member that failed or
    # was not timed keeps what one long window would have left it and is
    # not timed again (a timeout would cost the benchmark timeout twice);
    # the others keep their two windows. A worker that cannot time the
    # batch, or that survives its kill, ends the re-timing as before.

    monkeypatch.delenv("HELION_CAP_REBENCHMARK_REPEAT", raising=False)
    members, names = _retimed_members(2.00, 2.10, 2.20)
    script = {"a": [2.04, 2.02], "b": [2.14, 2.12], "c": [2.24, 2.22]}
    calls = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        order = "".join(names[id(fn)] for fn in fns)
        calls.append(order)
        if len(calls) == 2:
            if outcome == "unavailable":
                return None
            if outcome == "unkillable":
                error = BenchmarkWorkerUnkillable("worker survived kill")
                error.fn_index = order.index("a")
                raise error
        readings = []
        for name in order:
            reading = script[name].pop(0)
            if name == "b" and len(calls) == 1:
                if outcome in ("error", "timeout"):
                    reading = IsolatedBenchmarkFailure(outcome)
                elif outcome == "missing":
                    reading = None
            readings.append(reading)
        return readings

    search = _retiming_search(3, benchmark_isolated)
    a, b, c = members
    if outcome == "unavailable":
        with pytest.raises(exc.InvalidAPIUsage, match="isolated rebenchmarking"):
            search.rebenchmark(members)
        assert calls == ["abc", "cba"]
        assert [m.perfs for m in members] == [[2.00], [2.10], [2.20]]
        return
    if outcome == "unkillable":
        with pytest.raises(BenchmarkWorkerUnkillable) as caught:
            search.rebenchmark(members)
        # The second sweep runs in reverse: its third candidate is a.
        assert calls == ["abc", "cba"]
        assert caught.value.fn_index == 0
        assert search._fallback_excluded_configs == {a.config}
        return
    search.rebenchmark(members)
    assert calls == ["abc", "ca"]
    assert a.perfs == [2.00, 2.02]
    assert c.perfs == [2.20, 2.22]
    assert b.perfs == ([2.10, float("inf")] if outcome == "error" else [2.10])


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_bank_retiming_keeps_a_sweeps_failures_when_a_later_sweep_is_abandoned(
    failure,
):
    # In the first sweep a fails and b gets no reading. The second sweep, d
    # then c, ends at c in a worker that survives its kill, and the
    # re-timing returns no results. What the first sweep found is not lost
    # with them: when the search then keeps the best config it measured, that
    # is neither a, whose reading from before is the lowest, nor c, but b,
    # which did not fail.

    members, names = _retimed_members(2.00, 2.10, 2.20, 2.30)
    a, b, c, d = members
    calls = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        order = "".join(names[id(fn)] for fn in fns)
        calls.append(order)
        if len(calls) == 2:
            error = BenchmarkWorkerUnkillable("worker survived kill")
            error.fn_index = order.index("c")
            raise error
        first = {"a": IsolatedBenchmarkFailure(failure), "b": None}
        return [first.get(name, 2.5) for name in order]

    search = _retiming_search(3, benchmark_isolated)
    # (a copy: the setter of ``best`` writes into the population)
    search.population = [*members]
    search.finishing_rounds = 1
    search._final_selection_enabled = lambda: True
    search.run_finishing_phase = lambda best, rounds: search.rebenchmark(members)
    assert search._finalize() == b.config
    assert calls == ["abcd", "dc"]
    assert search._fallback_excluded_configs == {a.config, c.config}
    # No member took a reading from the abandoned re-timing.
    assert [m.perfs for m in members] == [[2.00], [2.10], [2.20], [2.30]]


@pytest.mark.parametrize("flash", [False, True])
def test_abandoned_bank_retiming_applies_the_failures_of_every_sweep(flash):
    # The batches are the provider's own, over a stand-in for the worker's
    # job. In the first sweep a hits a sticky CUDA error. The second sweep
    # runs the others in reverse: e times out, and d is under test when the
    # worker survives its kill, so the provider's error names the second
    # candidate of that sweep and holds one result before it. The re-timing
    # hands the search both failures and d by their places among the
    # members, and the search treats them as it does when a one-bank batch
    # ends that way. ``alias`` is another config of a's generated code and
    # is not re-timed. Of the readings from before, e's is the lowest and
    # the alias's the next.

    members, names = _retimed_members(2.00, 2.10, 2.20, 2.30, 1.90, 1.95)
    a, b, c, d, e, alias = members
    windows = dict.fromkeys(names.values(), 0)
    calls = []

    def job(fn, **kwargs):
        name = names[id(fn)]
        calls.append(name)
        windows[name] += 1
        if name == "a":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        if windows[name] == 2 and name == "e":
            raise BenchmarkTimeout("timed out")
        if windows[name] == 2 and name == "d":
            raise BenchmarkWorkerUnkillable("worker survived kill")
        return 2.5

    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(autotune_log_level=logging.CRITICAL)
    provider.log = Mock()
    provider._autotune_metrics = AutotuneMetrics()
    provider._subprocess_benchmark_enabled = lambda: True
    provider._subprocess_wrapper_unloadable = False
    provider._isolated_candidate_configs = {}
    provider._benchmark_worker = None
    provider._run_subprocess_benchmark_job = job
    search = _retiming_search(3, provider.benchmark_isolated)
    search.config_spec.cute_flash_search_enabled = flash
    search.config_spec.backend.generated_source_hash = lambda fn: (
        "shared" if fn in (a.fn, alias.fn) else None
    )
    invalidate = search.benchmark_provider.invalidate_effective_source_hash = Mock()
    search.population = [*members]
    search.finishing_rounds = 1
    search._final_selection_enabled = lambda: True
    caught = []

    def retime(best, rounds):
        try:
            search.rebenchmark(members[:5])
        except BenchmarkWorkerUnkillable as error:
            caught.append(error)
            raise

    search.run_finishing_phase = retime
    selected = search._finalize()

    assert calls == list("abcdeed")
    (error,) = caught
    assert error.fn_index == members.index(d)
    assert error.batch_results == (
        IsolatedBenchmarkFailure("error"),
        None,
        None,
        None,
        IsolatedBenchmarkFailure("timeout"),
    )
    assert search._fallback_excluded_configs == {a.config, d.config, e.config}
    infinite = [float("inf")]
    if flash:
        # as a completed re-timing leaves a failed kernel: every config of it
        assert selected == b.config
        assert [m.perfs for m in members] == [
            infinite,
            [2.10],
            [2.20],
            infinite,
            infinite,
            infinite,
        ]
        assert [m.status for m in (a, d, e, alias)] == [
            "error",
            "timeout",
            "timeout",
            "error",
        ]
        invalidate.assert_called_once_with("shared")
    else:
        # the readings stay; the alias is a config like any other
        assert selected == alias.config
        assert [m.perfs for m in members] == [
            [2.00],
            [2.10],
            [2.20],
            [2.30],
            [1.90],
            [1.95],
        ]
        invalidate.assert_not_called()


@pytest.mark.parametrize("outcome", ["error", "missing"])
def test_bank_retiming_asks_nothing_of_the_worker_once_every_member_is_settled(
    outcome,
):
    # Every member fails, or gets no reading, in its first window: there is
    # nobody left for the second sweep, and the worker is not asked to time
    # an empty batch.

    members, names = _retimed_members(2.00, 2.10, 2.20)
    calls = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        calls.append("".join(names[id(fn)] for fn in fns))
        return [
            IsolatedBenchmarkFailure("error") if outcome == "error" else None
            for _ in fns
        ]

    search = _retiming_search(3, benchmark_isolated)
    results = search._isolated_rebenchmark_results(
        members, rep=200, desc="test", candidate_private_args=False
    )
    assert calls == ["abc"]
    if outcome == "error":
        assert results == [IsolatedBenchmarkFailure("error")] * 3
    else:
        assert results == [None] * 3
    assert not _retiming_debug_lines(search)
    assert not search.log.call_args_list


@pytest.mark.parametrize(
    "cap,target_ms,window",
    [(None, 200.0, 50), (None, 5000.0, 50), (None, 20.0, 20), ("10", 200.0, 10)],
)
def test_bank_retiming_window_is_the_candidates_unless_a_shorter_one_is_asked(
    monkeypatch, cap, target_ms, window
):

    if cap is None:
        monkeypatch.delenv("HELION_CAP_REBENCHMARK_REPEAT", raising=False)
    else:
        monkeypatch.setenv("HELION_CAP_REBENCHMARK_REPEAT", cap)
    members, _ = _retimed_members(2.00, 2.10)
    reps = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        reps.append(rep)
        return [2.0 for _ in fns]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, target_ms=target_ms)
    assert reps == [window, window]


@pytest.mark.parametrize("final_verification", [True, False])
def test_bank_final_verification_keeps_its_one_steady_window(final_verification):
    # The single-pass final verification names its own window (the final
    # rebenchmark target) and times every finalist once, in a process of
    # its own where private copies do not fit; only the re-timings of the
    # search phases take the candidates' window (the control).

    members, names = _retimed_members(2.00, 2.10)
    calls = []

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        calls.append(("".join(names[id(fn)] for fn in fns), rep, fresh_process))
        return [2.5, 2.6][: len(fns)]

    search = _retiming_search(3, benchmark_isolated)
    results = search._isolated_rebenchmark_results(
        members, rep=5000, desc="test", candidate_private_args=final_verification
    )
    if final_verification:
        assert calls == [("ab", 5000, True)]
        assert results == [2.5, 2.6]
    else:
        assert calls == [("ab", 50, False), ("ba", 50, False)]
        assert results == [2.5, 2.5]


@pytest.mark.parametrize("shift", [0.08, 0.01, -0.08])
def test_bank_retiming_says_when_its_windows_read_another_device_state(shift):
    # Windows that sit 8% above the readings they repeat (what the 200 ms
    # window did) are another device state than the candidates were timed
    # in, and so are windows 8% below them: the search says so. Windows a
    # timer step away are not (the control), and are logged at debug only.

    members, _ = _retimed_members(2.00, 2.10, 2.20)
    firsts = {id(member.fn): member.perf for member in members}

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        return [firsts[id(fn)] * (1.0 + shift) for fn in fns]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="Verifying initial results")
    notes = [call.args[0] for call in search.log.call_args_list]
    details = _retiming_debug_lines(search)
    said = f"3 configs re-timed in 2 sweeps of 50 ms windows, which read {shift:+.1%}"
    if abs(shift) == 0.08:
        assert len(notes) == 1 and len(details) == 1
        assert notes[0].startswith("Verifying initial results: " + said)
        assert "do not read one device state" in notes[0]
    else:
        assert not notes
        assert len(details) == 2 and said in details[0]
    assert "re-timing windows in ms" in details[-1]


@pytest.mark.parametrize(
    "window,said,level",
    [
        (2.038, "+1.9%", "debug"),
        (2.0398, "+2.0%", "debug"),
        (2.04, "+2.0%", "info"),
        (1.962, "-1.9%", "debug"),
        (1.96, "-2.0%", "info"),
    ],
)
def test_bank_retiming_reports_a_shift_from_two_percent_on(window, said, level):
    # The sweep's line moves to info level where its windows read 2% from
    # the readings they repeat, in either direction, and stays at debug
    # level under that: at 1.9% and at 1.99%, which the line rounds to the
    # same +2.0% as the 2% that is reported. (2.04 against 2.00 is 2% as
    # written; in doubles the ratio less one is a hair above the threshold.)

    members, _ = _retimed_members(2.00, 2.00, 2.00)

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        return [window for _ in fns]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="Generation 2")
    notes = [call.args[0] for call in search.log.call_args_list]
    details = _retiming_debug_lines(search)
    line = (
        "Generation 2: 3 configs re-timed in 2 sweeps of 50 ms windows, which "
        f"read {said} (median) from the readings they repeat; the two windows "
        "of 0 of 3 configs (0%) sit more than 1% apart"
    )
    if level == "info":
        assert notes == [
            line + ": the windows of this search do not read one device state"
        ]
        assert len(details) == 1
    else:
        assert not notes
        assert details[0] == line and len(details) == 2


def test_bank_retiming_logs_the_shift_of_both_windows_and_every_raw_window():
    # The shift a sweep logs is the median over its members of what the
    # mean of a member's two windows reads against the reading it had:
    # +1%, +4% and +9% here, so +4.0% (the first windows alone give +2%,
    # the largest member +9%). Beside it the sweep logs every raw window:
    # a member's trace row holds the reading it takes, which for a (2.00,
    # between its windows) is the reading it had and for b and c the nearer
    # window.

    members, names = _retimed_members(2.00, 2.00, 2.00)
    script = {"a": [1.98, 2.06], "b": [2.04, 2.12], "c": [2.17, 2.19]}

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        return [script[names[id(fn)]].pop(0) for fn in fns]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="Generation 3")
    assert [member.perfs[-1] for member in members] == [2.00, 2.04, 2.17]
    (note,) = [call.args[0] for call in search.log.call_args_list]
    assert note == (
        "Generation 3: 3 configs re-timed in 2 sweeps of 50 ms windows, which "
        "read +4.0% (median) from the readings they repeat; the two windows of "
        "2 of 3 configs (67%) sit more than 1% apart: the windows of this "
        "search do not read one device state"
    )
    a, b, c = (canonical_config_id(member.config) for member in members)
    (raw,) = _retiming_debug_lines(search)
    assert raw == (
        "Generation 3: re-timing windows in ms (config_id: the reading it had "
        "| its windows | the reading it takes): "
        f"{a}: 2.0 | 1.98 2.06 | 2.0; "
        f"{b}: 2.0 | 2.04 2.12 | 2.04; "
        f"{c}: 2.0 | 2.17 2.19 | 2.17"
    )


def test_bank_retiming_counts_the_members_whose_windows_sit_apart():
    # Two windows count as apart when they differ by more than 1% of the
    # lower one: a's 0.995% and c's 1% exactly do not, b's 1.005% does
    # (0.995% of its upper window). The sweep's median shift (+0.5%) sees
    # none of it.

    members, names = _retimed_members(2.00, 2.00, 100.0)
    script = {"a": [2.00, 2.0199], "b": [2.0201, 2.00], "c": [100.0, 101.0]}

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        return [script[names[id(fn)]].pop(0) for fn in fns]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="Polish round 1")
    assert not search.log.call_args_list
    summary, _ = _retiming_debug_lines(search)
    assert summary == (
        "Polish round 1: 3 configs re-timed in 2 sweeps of 50 ms windows, which "
        "read +0.5% (median) from the readings they repeat; the two windows of "
        "1 of 3 configs (33%) sit more than 1% apart"
    )


def test_bank_retiming_counts_the_share_among_the_members_it_re_timed():
    # A member whose first window fails is settled there and has no pair of
    # windows: the share is of the members the sweeps re-timed, not of the
    # members they were asked for. b fails; of a and c, c's windows sit 2%
    # apart and a's half a percent.

    members, names = _retimed_members(2.00, 2.00, 2.00)
    script = {"a": [2.00, 2.01], "b": [None], "c": [2.00, 2.04]}

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        readings = [script[names[id(fn)]].pop(0) for fn in fns]
        return [
            IsolatedBenchmarkFailure("error") if reading is None else reading
            for reading in readings
        ]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="Polish round 2")
    assert not search.log.call_args_list
    assert script == {"a": [], "b": [], "c": []}
    summary, _ = _retiming_debug_lines(search)
    assert summary == (
        "Polish round 2: 2 configs re-timed in 2 sweeps of 50 ms windows, which "
        "read +0.6% (median) from the readings they repeat; the two windows of "
        "1 of 2 configs (50%) sit more than 1% apart"
    )


def test_bank_retiming_line_of_raw_windows_tells_a_tie_from_a_near_tie():
    # The line prints every value with the digits that tell it from any
    # other, so it shows which of the three a member took. a's reading
    # equals its second window (what one search logged: 0.023211 with
    # windows 0.022840 and 0.023211); b's second window is 1e-10 ms above
    # its reading and its first further above, so b takes that window, a
    # value six decimals would print as the reading.

    members, names = _retimed_members(0.023211, 0.025944)
    script = {"a": [0.022840, 0.023211], "b": [0.026166, 0.0259440001]}

    def benchmark_isolated(fns, *, warmup, rep, desc, fresh_process=False):
        return [script[names[id(fn)]].pop(0) for fn in fns]

    search = _retiming_search(3, benchmark_isolated)
    search.rebenchmark(members, desc="Polish round 3")
    assert [member.perfs[-1] for member in members] == [0.023211, 0.0259440001]
    a, b = (canonical_config_id(member.config) for member in members)
    _, raw = _retiming_debug_lines(search)
    assert raw.endswith(
        f"{a}: 0.023211 | 0.02284 0.023211 | 0.023211; "
        f"{b}: 0.025944 | 0.026166 0.0259440001 | 0.0259440001"
    )
