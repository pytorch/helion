from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import pytest
import torch

from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb
from helion.runtime.kernel import BoundKernel
from helion.runtime.settings import Settings


def _bound() -> tuple[BoundKernel, torch.Tensor]:
    """Real BoundKernel cache methods with a CPU-only source/backend fixture."""
    bound = object.__new__(BoundKernel)
    tensor = torch.arange(8)
    backend = SimpleNamespace(
        setup_compile_cache_dir=lambda index: None,
        requires_shape_specialized_module=False,
        annotate_compiled_module=lambda module, source, name: setattr(
            module, "source_digest", hashlib.sha256(source.encode()).hexdigest()
        ),
    )
    spec = SimpleNamespace(
        num_sm=152,
        target_device_capability=(10, 3),
        cute_chained_matmul_search_enabled=False,
        normalized_config=lambda config: Config(**copy.deepcopy(config.config)),
    )
    bound.kernel = SimpleNamespace(
        name="source_batch_fixture",
        settings=Settings(
            backend="cute", print_output_code=False, autotune_cute_source_workers=2
        ),
        cute_structural_policy=None,
        _validate_structural_policy=lambda: None,
        _reset_generation=0,
        _specialization_generation=0,
        _cute_grouped_static_tail_extra_descriptors=(),
        _specialize_extra=(),
        _compiler_seed_specialize_extra=(),
    )
    bound._env = SimpleNamespace(
        backend=backend,
        device=torch.device("cpu"),
        process_group_name=None,
        fake_mode=SimpleNamespace(avoid_device_init=True),
        config_spec=spec,
        runtime_input_specializations={},
        bound_runtime_input_specialization_results={},
        compiler_fact_specialization_facts={},
        specialized_vars=set(),
        cute_resolved_wrapper_plans=[],
        cute_fission_plan=None,
        cute_register_tile_disabled=False,
    )
    bound._base_spec_key = ("fixture",)
    bound._reset_generation = 0
    bound._compiler_seed_specialization_extractors = ()
    bound._runtime_tensor_refs_by_name = {"x": weakref.ref(tensor)}
    bound._cute_eviction_policy_receipts = {}
    bound._compile_cache = {}
    bound._cache_path_map = {}
    bound.host_function = SimpleNamespace(
        body=ast.parse("pass").body,
        device_ir=SimpleNamespace(graphs=[]),
        global_imports={},
        rng_seed_slot_count=0,
    )
    bound.fake_args = ()
    bound.format_kernel_decorator = lambda config, settings: "fixture"
    bound.maybe_log_repro = lambda *args, **kwargs: None
    bound.to_triton_code = lambda config, **kwargs: (
        f"def source_batch_fixture():\n    return {config.config['block_sizes'][0]}\n"
    )
    return bound, tensor


def _prepared(
    bound: BoundKernel, tensor: torch.Tensor, config: Config
) -> sb.PreparedSource:
    source = bound.to_triton_code(config)
    return sb.PreparedSource(
        bound,
        sb._config_key(config),
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        sb._snapshot(bound, (tensor,)),
        (tensor,),
        (tensor.untyped_storage(),),
        "",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("autotune_cute_source_workers", -1),
        ("autotune_cute_source_workers", 1.5),
        ("autotune_cute_source_workers", True),
    ],
)
def test_invalid_worker_count(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="source_workers"):
        Settings(**{field: value})


def test_default_and_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HELION_AUTOTUNE_CUTE_SOURCE_WORKERS", raising=False)
    assert Settings().autotune_cute_source_workers == 0
    monkeypatch.setenv("HELION_AUTOTUNE_CUTE_SOURCE_WORKERS", "2")
    assert Settings().autotune_cute_source_workers == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "tensor_data",
        "tensor_owner",
        "tensor_shape",
        "host_ast",
        "specialization",
        "settings",
        "target",
        "wrapper",
        "reset",
    ],
)
def test_prepared_source_rejects_live_mutation(mutation: str) -> None:
    bound, tensor = _bound()
    config = Config(block_sizes=[8])
    prepared = _prepared(bound, tensor, config)
    assert prepared.consume(bound, config) is not None
    if mutation == "tensor_data":
        tensor.add_(1)
    elif mutation == "tensor_owner":
        tensor.set_(torch.zeros(8, dtype=tensor.dtype))
    elif mutation == "tensor_shape":
        tensor.resize_(4)
    elif mutation == "host_ast":
        bound.host_function.body.append(ast.Pass())
    elif mutation == "specialization":
        bound.env.bound_runtime_input_specialization_results["new"] = 1
    elif mutation == "settings":
        bound.settings.fast_math = not bound.settings.fast_math
    elif mutation == "target":
        bound.config_spec.num_sm += 1
    elif mutation == "wrapper":
        bound.env.cute_resolved_wrapper_plans.append({"kind": "changed"})
    elif mutation == "reset":
        bound._reset_generation += 1
    assert prepared.consume(bound, config) is None


def test_receipt_identity_config_and_source_integrity() -> None:
    bound, tensor = _bound()
    config = Config(block_sizes=[8])
    prepared = _prepared(bound, tensor, config)
    other, _tensor = _bound()
    assert prepared.consume(other, config) is None
    assert prepared.consume(bound, Config(block_sizes=[4])) is None
    assert (
        dataclasses.replace(prepared, source="changed").consume(bound, config) is None
    )


def test_real_pycodecache_matches_serial_and_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    serial, serial_tensor = _bound()
    parallel, parallel_tensor = _bound()
    config = Config(block_sizes=[8])
    normal = serial.compile_config(config, allow_print=False)
    prepared = _prepared(parallel, parallel_tensor, config)
    parallel.to_triton_code = lambda *args, **kwargs: pytest.fail(
        "prepared source regenerated"
    )
    imported = parallel._compile_config(
        config, allow_print=False, prepared_source=prepared
    )
    assert imported is normal
    assert parallel._cache_path_map == serial._cache_path_map
    assert imported.__code__.co_filename == parallel._cache_path_map[config]
    assert Path(imported.__code__.co_filename).read_text() == prepared.source
    assert parallel.compile_config(config, allow_print=False) is imported
    assert serial_tensor.shape == (8,)


def test_stale_receipt_uses_original_error_path() -> None:
    bound, tensor = _bound()
    config = Config(block_sizes=[8])
    prepared = _prepared(bound, tensor, config)
    tensor.add_(1)
    error = ValueError("original source failure")

    def fail(*args: object, **kwargs: object) -> str:
        raise error

    bound.to_triton_code = fail
    with pytest.raises(ValueError) as caught:
        bound._compile_config(config, allow_print=False, prepared_source=prepared)
    assert caught.value is error
    assert not bound._compile_cache


@pytest.mark.parametrize(
    "reason",
    [
        "foreign_kernel",
        "distributed",
        "chained",
        "fission",
        "unknown_target",
        "print_code",
        "dead_owner",
        "missing_owner",
        "content_fact",
        "threads",
        "sdk_guard_missing",
    ],
)
def test_unsupported_routes_decline(
    reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, tensor = _bound()
    args = [tensor]
    monkeypatch.setattr(sb, "_guard_targets", list)
    if reason == "foreign_kernel":
        bound = SimpleNamespace()
    elif reason == "distributed":
        bound.env.process_group_name = "group"
    elif reason == "chained":
        bound.config_spec.cute_chained_matmul_search_enabled = True
    elif reason == "fission":
        bound.env.cute_fission_plan = object()
    elif reason == "unknown_target":
        bound.config_spec.num_sm = 0
    elif reason == "print_code":
        bound.settings.print_output_code = True
    elif reason == "dead_owner":
        temporary = torch.empty(1)
        bound._runtime_tensor_refs_by_name["dead"] = weakref.ref(temporary)
        temporary = None
    elif reason == "missing_owner":
        args = []
    elif reason == "content_fact":
        bound.env.runtime_input_specializations["content"] = SimpleNamespace(
            reusable_tensor_properties=frozenset()
        )
    elif reason == "threads":
        monkeypatch.setattr(sb, "source_threads_supported", lambda: False)
    elif reason == "sdk_guard_missing":

        def missing() -> list[tuple[object, str]]:
            raise AttributeError("missing SDK guard")

        monkeypatch.setattr(sb, "_guard_targets", missing)
    assert sb.SourceBatch.create(bound, args) is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
def test_batch_exact_order_and_no_child_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, tensor = _bound()
    monkeypatch.setattr(
        sb,
        "_guard_targets",
        lambda: [(BoundKernel, "compile_config"), (BoundKernel, "_compile_config")],
    )
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    configs = [Config(block_sizes=[8]), Config(block_sizes=[4])]
    prepared = batch.prepare(configs, lambda: False)
    assert list(prepared) == [0, 1] or list(prepared) == [1, 0]
    assert not bound._compile_cache and not bound._cache_path_map
    assert [prepared[i].source for i in range(2)] == [
        bound.to_triton_code(c) for c in configs
    ]
    assert [prepared[i].config_key for i in range(2)] == [
        sb._config_key(c) for c in configs
    ]
    assert torch.equal(tensor, torch.arange(8))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
def test_budget_stops_dispatch_without_changing_config_order(tmp_path: Path) -> None:
    checks = 0

    def budget() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    rows = sb._run_workers(
        [0, 1, 2],
        lambda index: 0,
        output=tmp_path,
        worker_timeout=1,
        budget_exceeded=budget,
    )
    assert [row["index"] for row in rows] == [0]


@pytest.mark.parametrize(
    "mode", ["success", "error", "timeout", "fork_failure", "interrupt"]
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
def test_owned_process_cleanup(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def child(index: int) -> int:
        if mode == "error" and index == 0:
            return 7
        if mode in ("timeout", "fork_failure", "interrupt"):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(10)
        return 0

    if mode in ("fork_failure", "interrupt"):
        fork = os.fork
        count = 0

        def fail_second() -> int:
            nonlocal count
            count += 1
            if count == 2:
                if mode == "interrupt":
                    raise KeyboardInterrupt
                raise OSError("second fork failed")
            return fork()

        monkeypatch.setattr(sb.os, "fork", fail_second)
    expected = {
        "error": RuntimeError,
        "timeout": TimeoutError,
        "fork_failure": OSError,
        "interrupt": KeyboardInterrupt,
    }
    if mode == "success":
        rows = sb._run_workers([0, 1], child, output=tmp_path, worker_timeout=1)
        assert len(rows) == 2 and all(row["exitcode"] == 0 for row in rows)
    else:
        with pytest.raises(expected[mode]):
            sb._run_workers([0, 1], child, output=tmp_path, worker_timeout=0.1)
    rows = json.loads((tmp_path / "reaps.json").read_text())
    for row in rows:
        with pytest.raises(ChildProcessError):
            os.waitpid(row["pid"], os.WNOHANG)


def test_metadata_function_defaults_and_unknown_objects() -> None:
    bound, tensor = _bound()
    config = Config(block_sizes=[8])

    def classifier(values: object, offset: int = 1) -> object:
        return values

    bound.env.bound_runtime_input_specialization_results["callback"] = classifier
    prepared = _prepared(bound, tensor, config)
    classifier.__defaults__ = (2,)
    assert prepared.consume(bound, config) is None

    class Hidden:
        def __repr__(self) -> str:
            return "unchanging"

    with pytest.raises(TypeError, match="Unsupported source metadata"):
        sb._freeze_metadata(Hidden(), bound)


@pytest.mark.parametrize("failure", ["snapshot", "missing_file", "invalid_json"])
def test_post_worker_validation_falls_back(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, tensor = _bound()
    batch = sb.SourceBatch(bound, (tensor,), [], 1)
    original_snapshot = sb._snapshot
    completed = False

    def workers(*args: object, **kwargs: object) -> list[dict[str, int]]:
        nonlocal completed
        completed = True
        output = kwargs["output"]
        if failure == "invalid_json":
            (output / "0.py").write_text("source")
            (output / "0.json").write_text("invalid JSON")
        return [{"index": 0, "pid": 1, "exitcode": 0}]

    def snapshot(*args: object) -> str:
        if completed and failure == "snapshot":
            raise ValueError("post-worker proof failed")
        return original_snapshot(*args)

    monkeypatch.setattr(sb, "_run_workers", workers)
    monkeypatch.setattr(sb, "_snapshot", snapshot)
    assert batch.prepare([Config(block_sizes=[8])], lambda: False) == {}
    assert not bound._compile_cache


def _compile_prefix() -> object:
    # Execute the actual provider method only through its compile loop. Native,
    # correctness and benchmark phases are deliberately outside this CPU test.
    import inspect
    import textwrap

    from helion.autotuner import benchmark_provider

    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(benchmark_provider.LocalBenchmarkProvider.benchmark)
        )
    )
    method = tree.body[0]
    stop = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "deduplicated_indices"
    )
    method.body = method.body[:stop] + ast.parse("return compiled, results").body
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
    )
    namespace = dict(vars(benchmark_provider))
    exec(compile(module, "<original-provider-compile-prefix>", "exec"), namespace)
    return namespace["benchmark"]


@pytest.mark.parametrize("workers", [0, 1, 2, 3])
def test_provider_original_order_cache_and_caller_worker_bound(
    workers: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    bound, tensor = _bound()
    bound.settings.autotune_cute_source_workers = workers
    observed = []
    normal_annotate = bound.env.backend.annotate_compiled_module

    def annotate(module: object, source: str, name: str) -> None:
        observed.append(source)
        normal_annotate(module, source, name)

    bound.env.backend.annotate_compiled_module = annotate
    monkeypatch.setattr(sb, "_guard_targets", list)
    if workers <= 1:

        def forbidden(*args: object) -> None:
            pytest.fail("default/serial route attempted source batching")

        monkeypatch.setattr(sb.SourceBatch, "create", forbidden)
    provider = SimpleNamespace(
        kernel=bound,
        args=[tensor],
        settings=bound.settings,
        config_spec=bound.config_spec,
        _validate_timing_policy=lambda: None,
        _effective_source_dedup_enabled=lambda: False,
        _effective_source_hashes=set(),
        _budget_exceeded_synced=lambda: False,
        _autotune_metrics=SimpleNamespace(num_successful_candidate_measurements=1),
    )
    configs = [Config(block_sizes=[n]) for n in (8, 4, 8, 2)]
    compiled, results = _compile_prefix()(provider, configs)
    assert list(compiled) == list(range(4))
    assert compiled[0] is compiled[2]
    assert [fn() for fn in compiled.values()] == [8, 4, 8, 2]
    assert [r.config for r in results] == configs
    # Duplicate cache hit keeps normal annotation/publication behavior.
    assert observed == [
        bound.to_triton_code(Config(block_sizes=[n])) for n in (8, 4, 2)
    ]


def test_provider_budget_tail_after_started_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, tensor = _bound()
    expired = False
    original_batch = sb.SourceBatch(bound, (tensor,), [], 1)

    def prepare(configs: object, budget: object) -> dict[int, sb.PreparedSource]:
        nonlocal expired
        result = {
            i: _prepared(bound, tensor, config) for i, config in enumerate(configs)
        }
        expired = True
        return result

    original_batch.prepare = prepare
    monkeypatch.setattr(sb.SourceBatch, "create", lambda *args: original_batch)
    provider = SimpleNamespace(
        kernel=bound,
        args=[tensor],
        settings=bound.settings,
        config_spec=bound.config_spec,
        _validate_timing_policy=lambda: None,
        _effective_source_dedup_enabled=lambda: False,
        _effective_source_hashes=set(),
        _budget_exceeded_synced=lambda: expired,
        _autotune_metrics=SimpleNamespace(num_successful_candidate_measurements=1),
    )
    configs = [
        Config(block_sizes=[8]),
        Config(block_sizes=[4]),
        Config(block_sizes=[2]),
    ]
    compiled, results = _compile_prefix()(provider, configs)
    assert list(compiled) == [0]
    assert [r.config for r in results] == configs
    assert results[1].perf == results[2].perf == float("inf")


def test_parent_source_wall_category(monkeypatch: pytest.MonkeyPatch) -> None:
    from helion import _compile_time

    bound, tensor = _bound()
    batch = sb.SourceBatch(bound, (tensor,), [], 1)

    def work(*args: object) -> dict[int, sb.PreparedSource]:
        time.sleep(0.01)
        return {}

    monkeypatch.setattr(batch, "_prepare", work)
    monkeypatch.setattr(_compile_time, "_enabled", True)
    tracker = _compile_time.get_tracker()
    before = tracker._timings["CuTe.source_batch_wall"]
    batch.prepare([], lambda: False)
    assert tracker._timings["CuTe.source_batch_wall"] - before >= 0.01


@pytest.mark.skipif(sys.platform != "linux", reason="Linux source batch eligibility")
def test_observed_autotuner_live_closure_is_not_source_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, tensor = _bound()
    searches: list[object] = []

    def observed_autotuner(*args: object, **kwargs: object) -> object:
        search = object()
        searches.append(search)
        return search

    bound.settings.autotuner_fn = observed_autotuner
    observed_autotuner()
    monkeypatch.setattr(sb, "_guard_targets", list)
    assert sb.SourceBatch.create(bound, [tensor]) is not None
    config = Config(block_sizes=[8])
    prepared = _prepared(bound, tensor, config)
    searches.append(object())
    assert prepared.consume(bound, config) is not None
    bound.settings.autotuner_fn = lambda *args, **kwargs: None
    assert prepared.consume(bound, config) is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
@pytest.mark.parametrize("handler", [signal.SIG_IGN, lambda *args: None])
def test_custom_child_reaping_declines_before_fork(
    handler: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bound, tensor = _bound()
    original = signal.getsignal
    monkeypatch.setattr(
        signal,
        "getsignal",
        lambda sig: handler if sig == signal.SIGCHLD else original(sig),
    )
    monkeypatch.setattr(
        os, "fork", lambda: pytest.fail("fork under foreign child reaper")
    )
    assert sb.SourceBatch.create(bound, [tensor]) is None
    with pytest.raises(RuntimeError, match="exclusive child reaping"):
        sb._run_workers([0], lambda i: 0, output=tmp_path, worker_timeout=1)


def test_target_metadata_is_parent_resolved_and_device_bound() -> None:
    from contextlib import ExitStack

    from helion._compiler.cute.mma_support import CuteMmaSupport
    from helion.runtime.cute.source_target import TargetMetadata

    properties = SimpleNamespace(
        major=10, minor=3, name="fixture", multi_processor_count=152
    )
    support = CuteMmaSupport(True, True, False, True, False, False)
    cuda = SimpleNamespace(
        current_device=lambda: 0,
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_capability=lambda i: (10, 3),
        get_device_name=lambda i: "fixture",
        get_device_properties=lambda i: properties,
        is_initialized=lambda: False,
        _is_in_bad_fork=lambda: True,
    )
    original = vars(cuda).copy()

    class Interface:
        current_device = staticmethod(cuda.current_device)
        device_count = staticmethod(cuda.device_count)
        get_device_properties = staticmethod(cuda.get_device_properties)

        class Worker:
            current_device = staticmethod(cuda.current_device)
            get_device_properties = staticmethod(cuda.get_device_properties)

    modules = tuple(
        SimpleNamespace(get_cute_mma_support=lambda: support) for _ in range(3)
    )
    metadata = TargetMetadata(
        SimpleNamespace(cuda=cuda, device=torch.device),
        Interface,
        properties,
        modules,
        support,
    )
    with ExitStack() as stack:
        metadata.install(stack)
        for device in (None, 0, "cuda", torch.device("cuda:0")):
            assert cuda.get_device_properties(device).major == properties.major
            assert cuda.get_device_properties(device) is not properties
            assert (
                Interface.Worker.get_device_properties(device).major == properties.major
            )
            assert cuda.get_device_capability(device) == (10, 3)
        for bad in (1, -1, True, "cpu", "cuda:1"):
            with pytest.raises((ValueError, TypeError)):
                cuda.get_device_properties(bad)
        assert all(m.get_cute_mma_support() is support for m in modules)
        assert cuda.is_initialized is original["is_initialized"]
        assert cuda._is_in_bad_fork is original["_is_in_bad_fork"]
        metadata.check()
        properties.major = 9
        with pytest.raises(AssertionError):
            metadata.check()
    assert vars(cuda) == original


@pytest.mark.skipif(sys.platform != "linux", reason="Linux source batch eligibility")
def test_metadata_snapshot_failure_retries_original_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, tensor = _bound()
    bound.env.device = torch.device("cuda:0")
    monkeypatch.setattr(sb, "_guard_targets", list)
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None

    def unavailable(*args: object) -> None:
        raise RuntimeError("unsupported target metadata")

    monkeypatch.setattr(sb.TargetMetadata, "capture", unavailable)
    assert batch.prepare([Config(block_sizes=[8])], lambda: False) == {}
    assert not bound._compile_cache


def _metadata_fixture(properties: object) -> object:
    from helion._compiler.cute.mma_support import CuteMmaSupport
    from helion.runtime.cute.source_target import TargetMetadata

    support = CuteMmaSupport(True, True, False, True, False, False)
    cuda = SimpleNamespace(
        current_device=lambda: 0,
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_capability=lambda i: (10, 3),
        get_device_name=lambda i: "fixture",
    )
    return TargetMetadata(
        SimpleNamespace(cuda=cuda, device=torch.device),
        object,
        properties,
        (),
        support,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
def test_child_properties_never_touch_runtime_backed_owner(tmp_path: Path) -> None:
    parent_pid = os.getpid()

    class RuntimeProperties:
        major = 10
        minor = 3
        name = "fixture"
        multi_processor_count = 152

        def __getattribute__(self, name: str) -> object:
            assert os.getpid() == parent_pid, "runtime-backed getter in child"
            return object.__getattribute__(self, name)

        def __dir__(self) -> list[str]:
            assert os.getpid() == parent_pid, "runtime-backed dir in child"
            return ["major", "minor", "name", "multi_processor_count"]

    props = RuntimeProperties()
    metadata = _metadata_fixture(props)

    def child(index: int) -> int:
        view = metadata.get_properties(0)
        assert view.major == 10 and view.multi_processor_count == 152
        metadata.check()
        with pytest.raises(AssertionError):
            metadata._property_values()
        with pytest.raises(AttributeError):
            assert view.unknown_property is not None
        with pytest.raises(AttributeError):
            view.major = 9
        return 0

    assert (
        sb._run_workers([0], child, output=tmp_path, worker_timeout=2)[0]["exitcode"]
        == 0
    )
    metadata.check()
    props.major = 9
    with pytest.raises(AssertionError):
        metadata.check()


def test_property_view_mutation_and_owner_replacement_reject() -> None:
    from helion.runtime.cute.source_target import _DeviceProperties

    props = SimpleNamespace(major=10, minor=3, name="fixture")
    metadata = _metadata_fixture(props)
    metadata.frozen_properties = _DeviceProperties((("major", 9),))
    with pytest.raises(AssertionError):
        metadata.check()
    metadata = _metadata_fixture(props)
    metadata.properties = SimpleNamespace(**vars(props))
    with pytest.raises(AssertionError):
        metadata.check()


@pytest.fixture
def idle_tqdm() -> object:
    pytest.importorskip("tqdm")
    from tqdm.std import tqdm

    from helion.runtime.cute import source_threads as st

    assert st.prepare_source_fork()
    interval = tqdm.monitor_interval
    # Disable output, not the monitor: disable=True would skip instance tracking.
    import io

    bar = tqdm(total=1, file=io.StringIO(), mininterval=1000)
    monitor = tqdm.monitor
    assert monitor is not None and monitor.is_alive()
    bar.close()
    assert not tqdm._instances
    try:
        yield monitor
    finally:
        monitor.was_killed.set()
        monitor.join(timeout=2)
        assert not monitor.is_alive()
        if tqdm.monitor is monitor:
            tqdm.monitor = None
        import atexit

        atexit.unregister(monitor._atexit_signal)
        assert tqdm.monitor_interval == interval


def test_idle_monitor_retirement_preserves_normal_recreation(idle_tqdm: object) -> None:
    import io

    from tqdm.std import tqdm

    from helion.runtime.cute import source_threads as st

    monitor = idle_tqdm
    interval = tqdm.monitor_interval
    assert st.source_threads_supported()
    assert st.prepare_source_fork()
    assert not monitor.is_alive() and monitor.was_killed.is_set()
    assert tqdm.monitor is None and tqdm.monitor_interval == interval
    bar = tqdm(total=1, file=io.StringIO())
    replacement = tqdm.monitor
    assert replacement is not monitor and replacement.is_alive()
    assert not st.source_threads_supported()
    assert not st.prepare_source_fork()
    assert not replacement.was_killed.is_set()
    bar.close()
    assert st.prepare_source_fork()


def test_foreign_live_thread_declines_without_touching_monitor(
    idle_tqdm: object,
) -> None:
    from helion.runtime.cute import source_threads as st

    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()
    try:
        assert not st.source_threads_supported()
        assert not st.prepare_source_fork()
        assert not idle_tqdm.was_killed.is_set()
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "event",
        "run",
        "callback",
        "rebound_callback",
        "rebound_event",
        "class_join",
        "class_is_alive",
    ],
)
def test_monitor_identity_or_event_changes_decline(
    idle_tqdm: object, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from tqdm.std import tqdm

    from helion.runtime.cute import source_threads as st

    original_event = idle_tqdm.was_killed
    # Restore before fixture cleanup, which must use the real thread/event.
    with monkeypatch.context() as local:
        if change == "owner":
            local.setattr(tqdm, "monitor", object())
        elif change == "event":
            local.setattr(idle_tqdm, "was_killed", SimpleNamespace())
        elif change == "run":
            local.setattr(idle_tqdm, "run", lambda: None)
        elif change == "callback":
            local.setattr(idle_tqdm, "_atexit_signal", lambda: None)
        elif change in ("class_join", "class_is_alive"):
            local.setattr(
                type(idle_tqdm), change.removeprefix("class_"), lambda *a, **k: None
            )
        elif change == "rebound_callback":
            from types import MethodType

            local.setattr(
                idle_tqdm,
                "_atexit_signal",
                MethodType(
                    type(idle_tqdm)._atexit_signal,
                    SimpleNamespace(was_killed=threading.Event()),
                ),
            )
        else:
            from types import MethodType

            local.setattr(
                original_event,
                "set",
                MethodType(st._EVENT_METHODS["set"], threading.Event()),
            )
        assert not st.prepare_source_fork()
        assert not original_event.is_set()


def test_monitor_failed_join_never_forks_or_unhooks(
    idle_tqdm: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tqdm.std import tqdm

    from helion.runtime.cute import source_threads as st

    joined = []
    # Exercise post-join failure independently of OS scheduling; no fake thread
    # is admitted by the production identity predicate.
    monkeypatch.setattr(st, "_idle_monitor", lambda: idle_tqdm)
    monkeypatch.setattr(idle_tqdm, "join", lambda *, timeout: joined.append(timeout))
    monkeypatch.setattr(idle_tqdm, "is_alive", lambda: True)
    monkeypatch.setattr(os, "fork", lambda: pytest.fail("fork before successful drain"))
    with pytest.raises(RuntimeError, match="quiescent"):
        sb._run_workers([0], lambda i: 0, output=tmp_path, worker_timeout=1)
    assert joined == [1.0]
    assert tqdm.monitor is idle_tqdm
    monkeypatch.undo()


def test_monitor_owner_changes_after_join_decline(
    idle_tqdm: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tqdm.std import tqdm

    from helion.runtime.cute import source_threads as st

    replacement = object()
    real_join = idle_tqdm.join

    def changed_join(*, timeout: float) -> None:
        real_join(timeout=timeout)
        tqdm.monitor = replacement

    with monkeypatch.context() as local:
        local.setattr(st, "_idle_monitor", lambda: idle_tqdm)
        local.setattr(idle_tqdm, "join", changed_join)
        assert not st.prepare_source_fork()
        assert tqdm.monitor is replacement
    tqdm.monitor = None


def test_lone_main_thread_needs_no_tqdm_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion.runtime.cute import source_threads as st

    monkeypatch.delitem(sys.modules, "tqdm._monitor", raising=False)
    monkeypatch.delitem(sys.modules, "tqdm.std", raising=False)
    assert st.source_threads_supported()
    assert st.prepare_source_fork()
    assert "tqdm._monitor" not in sys.modules and "tqdm.std" not in sys.modules


def test_missing_optional_monitor_api_declines(
    idle_tqdm: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from helion.runtime.cute import source_threads as st

    with monkeypatch.context() as local:
        local.delattr(type(idle_tqdm), "_atexit_signal")
        assert not st.source_threads_supported()
        assert not st.prepare_source_fork()
        assert not idle_tqdm.was_killed.is_set()


@pytest.mark.parametrize("alias", ["absent", "callable", "invalid"])
def test_installed_guard_api_and_optional_compile_alias(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass import cute

    if alias == "absent":
        monkeypatch.delattr(cutlass, "compile", raising=False)
    else:
        value = (
            (lambda: pytest.fail("compile guard discovery executed an alias"))
            if alias == "callable"
            else object()
        )
        monkeypatch.setattr(cutlass, "compile", value, raising=False)
    if alias == "invalid":
        with pytest.raises(TypeError, match="Missing source worker guard: compile"):
            sb._guard_targets()
        return
    targets = sb._guard_targets()
    assert ((cutlass, "compile") in targets) == (alias == "callable")
    assert (cute, "compile") in targets
    assert (type(cute.compile), "_compile") in targets
    assert (type(cute.compile), "to_precompiled_mlir") in targets
