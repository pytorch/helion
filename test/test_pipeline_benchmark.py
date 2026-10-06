from __future__ import annotations

import contextlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from helion.autotuner import pipeline_benchmark as module
from helion.runtime.config import Config
from helion.runtime.pipeline import PipelineConfig


@pytest.fixture
def fake_graph_runtime(monkeypatch):
    state = SimpleNamespace(
        scope=None,
        phase="prepare",
        operations=0,
        captures=0,
        replays=0,
        capture_error=None,
        broken_replay=False,
        benchmark_calls=0,
    )

    class Scope:
        def __init__(self, bundle, **kwargs):
            self.configs = dict(bundle.configs)
            self.stages = {}
            self.trace = []
            self.frozen = False

        @property
        def bundle(self):
            return PipelineConfig(self.configs)

        @contextlib.contextmanager
        def activate(self):
            assert state.scope is None
            state.scope = self
            try:
                yield self
            finally:
                state.scope = None

        def freeze(self):
            self.frozen = True

    def capture(call):
        assert state.scope.frozen
        state.captures += 1
        if state.capture_error:
            raise RuntimeError(state.capture_error)
        state.phase = "capture"
        output = call()
        expected = output.clone()
        state.phase = "prepared"

        def replay():
            state.replays += 1
            if not state.broken_replay:
                output.copy_(expected)

        return SimpleNamespace(replay=replay), output

    monkeypatch.setattr(module, "PipelineScope", Scope)
    monkeypatch.setattr(module, "_require_gpu_inputs", lambda args: None)
    monkeypatch.setattr(module, "_capture_graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return state


def _topk_operation(state):
    def operation(x, k):
        state.operations += 1
        scope = state.scope
        width = scope.bundle.configs["tile"]["block_sizes"][0]
        current = x
        while True:
            key = f"topk:{current.shape[1]}"
            if scope.frozen and key not in scope.stages:
                raise ValueError("New binding after freeze")
            scope.trace.append(key)
            scope.configs[key] = Config(block_sizes=[width])
            scope.stages.setdefault(
                key,
                SimpleNamespace(
                    source_hash="test-topk-source",
                    argument_metadata={"shape": list(current.shape)},
                    config=scope.configs[key],
                    config_source="bundle",
                    config_space_identity=("triton", "test"),
                ),
            )
            chunks = []
            for chunk in current.split(width, dim=1):
                values = chunk.topk(min(k, chunk.shape[1]), dim=1).values
                if values.shape[1] != k:
                    values = torch.nn.functional.pad(
                        values, (0, k - values.shape[1]), value=-math.inf
                    )
                chunks.append(values)
            current = torch.cat(chunks, dim=1)
            if current.shape[1] == k:
                return current

    return operation


def test_candidate_topology_and_whole_graph_geomean(fake_graph_runtime):
    state = fake_graph_runtime
    values = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    cases = [(values, 2), (-values, 2)]
    samples = iter([2.0, 8.0, 6.0, 24.0])

    def benchmark(replay):
        before = state.operations
        assert state.scope.frozen
        replay()
        assert state.operations == before  # Only recorded GPU work is measured.
        state.benchmark_calls += 1
        return next(samples)

    evaluator = module.PipelineEvaluator(
        _topk_operation(state),
        cases,
        reference=lambda x, k: x.topk(k, dim=1).values,
        check=torch.testing.assert_close,
        benchmark=benchmark,
    )
    short = evaluator.evaluate(PipelineConfig({"tile": Config(block_sizes=[16])}))
    long = evaluator.evaluate(PipelineConfig({"tile": Config(block_sizes=[4])}))
    assert short.status == long.status == "ok"
    assert short.traces == (("topk:128", "topk:16"),) * 2
    assert (
        long.traces
        == (("topk:128", "topk:64", "topk:32", "topk:16", "topk:8", "topk:4"),) * 2
    )
    assert short.timings_ms == (2.0, 8.0) and short.aggregate_ms == pytest.approx(4.0)
    assert long.timings_ms == (6.0, 24.0) and long.aggregate_ms == pytest.approx(12.0)
    assert state.benchmark_calls == 4
    assert state.scope is None
    assert "topk:64" in long.bundle.configs
    assert "topk:64" not in short.bundle.configs
    json.dumps(long.to_dict())  # No tensors or bound kernels leak into receipts.


@pytest.mark.parametrize(
    "failure", ["capture", "poison", "invalid_timing", "wrong_peer"]
)
def test_failed_candidates_never_get_a_valid_score(fake_graph_runtime, failure):
    state = fake_graph_runtime
    data = torch.arange(128, dtype=torch.float32).reshape(1, 128)
    if failure == "capture":
        state.capture_error = "graph capture unavailable"
    state.broken_replay = failure == "poison"

    def benchmark(replay):
        state.benchmark_calls += 1
        replay()
        return math.nan if failure == "invalid_timing" else 1.0

    def reference(x, k):
        value = x.topk(k, dim=1).values
        return value + 10 if failure == "wrong_peer" and x[0, 0] < 0 else value

    evaluator = module.PipelineEvaluator(
        _topk_operation(state),
        [(data, 2), (-data - 1, 2)],
        reference=reference,
        check=torch.testing.assert_close,
        benchmark=benchmark,
    )
    result = evaluator.evaluate(PipelineConfig({"tile": Config(block_sizes=[16])}))
    assert result.status == "error" and math.isinf(result.aggregate_ms)
    assert result.error
    assert state.scope is None
    if failure != "invalid_timing":
        assert state.benchmark_calls == 0
    if failure == "capture":
        assert "graph capture unavailable" in result.error


def test_input_aliases_preserved_and_mutation_rejected_without_changing_caller(
    fake_graph_runtime,
):
    storage = torch.arange(8, dtype=torch.float32)
    before = storage.clone()
    views = (storage[:4], storage[::2])

    def operation(x, y):
        assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
        x.add_(1)
        return x + y

    evaluator = module.PipelineEvaluator(
        operation,
        [views],
        reference=lambda x, y: x + y + 1,
        check=lambda actual, expected: None,
    )
    result = evaluator.evaluate(PipelineConfig({}))
    assert result.status == "error" and "external input" in result.error
    torch.testing.assert_close(storage, before)
    assert fake_graph_runtime.scope is None


def test_fatal_gpu_error_propagates_and_scope_is_restored(fake_graph_runtime):
    def operation(x):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    evaluator = module.PipelineEvaluator(
        operation,
        [(torch.ones(1),)],
        reference=lambda x: x,
        check=torch.testing.assert_close,
    )
    with pytest.raises(RuntimeError, match="illegal memory access"):
        evaluator.evaluate(PipelineConfig({}))
    assert fake_graph_runtime.scope is None


def test_input_reset_cannot_silently_enter_measured_replay():
    with pytest.raises(ValueError, match="Input-reset"):
        module.PipelineEvaluator(
            lambda x: x,
            [(torch.ones(1),)],
            reference=lambda x: x,
            check=torch.testing.assert_close,
            reset=lambda: None,
        )


def test_input_clone_preserves_contiguous_view_offset_and_alias_metadata():
    storage = torch.arange(32, dtype=torch.float32, requires_grad=True)
    contiguous = storage[4:12]
    strided = storage[3:19:2]
    clone, other, repeated = module._clone_inputs((contiguous, strided, contiguous))
    assert clone is repeated
    assert clone.storage_offset() == 4 and other.storage_offset() == 3
    assert clone.stride() == (1,) and other.stride() == (2,)
    assert clone.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
    assert clone.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
    assert not clone.requires_grad and not other.requires_grad
    torch.testing.assert_close(clone, contiguous)
    torch.testing.assert_close(other, strided)
    (single,) = module._clone_inputs((contiguous,))
    assert single.storage_offset() == 4


def test_input_metadata_mutation_is_not_hidden_by_unchanged_bytes(fake_graph_runtime):
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def operation(x):
        x.resize_(4, 2)
        return x.clone()

    result = module.PipelineEvaluator(
        operation,
        [(source,)],
        reference=lambda x: x.reshape(4, 2),
        check=torch.testing.assert_close,
    ).evaluate(PipelineConfig())
    assert result.status == "error" and "external input" in result.error
    assert source.shape == (2, 4)


def test_default_timer_times_only_already_captured_replay(monkeypatch):
    replays = []
    recordings = []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            recordings.append(len(replays))

        def elapsed_time(self, other):
            return 0.125

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    result = module.benchmark_graph_replay(
        lambda: replays.append(True), warmup=2, repeat=3
    )
    assert result == 0.125
    assert len(replays) == 5
    assert recordings == [2, 3, 3, 4, 4, 5]


def test_distributed_pipeline_tuning_is_rejected_upfront(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="Distributed"):
        module.PipelineEvaluator(
            lambda x: x,
            [(torch.ones(1),)],
            reference=lambda x: x,
            check=torch.testing.assert_close,
        )
