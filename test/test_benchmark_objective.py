from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkObjectiveSession
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.benchmark_objective import WorkloadMeasurement
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _corrupt(x: torch.Tensor) -> torch.Tensor:
    x.fill_(99)
    return x


class FakePipeline:
    def __init__(self) -> None:
        self.storage = torch.arange(8, dtype=torch.float32)
        self.raw_args = (self.storage, self.storage[2:6])
        self.policy = {"partner": "v1", "reset": "pristine", "oracle": "identity"}
        self.events: list[tuple[str, int]] = []
        self.private: list[torch.Tensor] = []

    def cache_record(self) -> dict[str, object]:
        return dict(self.policy)

    def bind(
        self, request: BenchmarkRequest, private_args: Sequence[object]
    ) -> BenchmarkWorkload:
        owner, alias = (cast("torch.Tensor", value) for value in private_args)
        assert alias.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
        assert alias.storage_offset() == 2
        self.private.append(owner)
        original = owner.clone()
        ordinal = len(self.private)

        def reset() -> None:
            self.events.append(("reset", ordinal))
            owner.copy_(original)

        def run() -> object:
            self.events.append(("prepare", ordinal))
            prepared = owner + 1
            self.events.append(("candidate", ordinal))
            output = request.compiled(prepared)
            self.events.append(("finish", ordinal))
            alias.add_(1)
            return output

        def validate(output: object) -> None:
            self.events.append(("validate", ordinal))
            torch.testing.assert_close(cast("torch.Tensor", output), original + 1)
            expected = original.clone()
            expected[2:6].add_(1)
            torch.testing.assert_close(owner, expected)

        return BenchmarkWorkload(
            run, reset, validate, lambda: self.events.append(("close", ordinal))
        )


def _request(value: int = 1) -> BenchmarkRequest:
    return BenchmarkRequest.create(Config(block_sizes=[value]), _identity, "stage-v1")


def _timer(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
    samples = []
    for workload in workloads:
        for _ in range(3):
            workload.reset()
            output = workload.run()
            # Every timing invocation starts from the same private raw data.
            workload.validate(output)
        samples.append([1.0, 1.1, 1.2])
    return samples


def test_repeated_batches_receive_private_full_pipeline_workloads() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    request = _request()
    results = []
    for _batch in range(4):
        results.extend(session.measure([request], _timer))
    assert all(result.measurement is not None for result in results)
    assert {result.measurement.epoch for result in results if result.measurement} == {
        session.epoch
    }
    assert len({tensor.data_ptr() for tensor in objective.private}) == 4
    assert all(
        tensor.data_ptr() != objective.storage.data_ptr()
        for tensor in objective.private
    )
    torch.testing.assert_close(objective.storage, torch.arange(8, dtype=torch.float32))
    assert [name for name, ordinal in objective.events if ordinal == 1] == [
        *(["reset", "prepare", "candidate", "finish", "validate"] * 4),
        "validate",
        "close",
    ]


def test_session_keeps_its_original_raw_input_snapshot() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    objective.storage.fill_(100)
    session.measure([_request()], _timer)
    expected = torch.arange(8, dtype=torch.float32)
    expected[2:6].add_(1)
    torch.testing.assert_close(objective.private[0], expected)
    assert objective.storage.eq(100).all()


def test_full_boundary_failure_rejects_only_that_candidate() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    bad = BenchmarkRequest.create(Config(block_sizes=[2]), _corrupt, "bad-stage")
    results = session.measure([bad, _request()], _timer)
    assert results[0].measurement is None and results[0].accuracy_error
    assert results[1].measurement is not None
    assert [name for name, ordinal in objective.events if ordinal == 1].count(
        "prepare"
    ) == 1
    assert [name for name, _ in objective.events].count("close") == 2


def test_changed_epoch_rejects_retained_scores_and_candidate_binding() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    request = _request()
    result = session.measure([request], _timer)[0]
    assert result.measurement is not None
    objective.policy["partner"] = "v2"
    with pytest.raises(BenchmarkObjectiveError, match="fresh search"):
        session.measure([request], _timer)
    new_session = BenchmarkObjectiveSession(objective)
    with pytest.raises(BenchmarkObjectiveError, match="another workload"):
        result.measurement.validate(new_session.epoch, request)
    assert session.epoch != new_session.epoch


def test_cache_identity_separates_partner_and_candidate_without_mutating_config() -> (
    None
):
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    config = Config(block_sizes=[1])
    request = BenchmarkRequest.create(config, _identity, "stage-v1")
    old_key = session.measurement_key(request)
    config.config["block_sizes"] = [2]
    assert request.config.config["block_sizes"] == [1]
    assert session.measurement_key(_request(2)) != old_key
    default_key = "old-kernel-key"
    cache_key = session.extra_cache_key(default_key)
    objective.policy["partner"] = "v2"
    assert (
        BenchmarkObjectiveSession(objective).extra_cache_key(default_key) != cache_key
    )


@pytest.mark.parametrize("samples", [[], [float("nan")], [0.0], [-1.0], [True]])
def test_invalid_event_samples_never_become_scores(samples: list[float]) -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    with pytest.raises(BenchmarkObjectiveError, match="finite and positive"):
        session.measure([_request()], lambda workloads: [samples])
    assert objective.events[-1][0] == "close"


def test_timer_failure_closes_every_bound_candidate() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)

    def fail(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        session.measure([_request(), _request(2)], fail)
    assert [name for name, _ in objective.events].count("close") == 2


def test_epoch_change_during_timer_fails_after_releasing_workloads() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)

    def change(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
        objective.policy["reset"] = "changed"
        return [[1.0]]

    with pytest.raises(BenchmarkObjectiveError, match="fresh search"):
        session.measure([_request()], change)
    assert objective.events[-1][0] == "close"


def test_a_measurement_cannot_be_reused_for_another_candidate() -> None:
    request = _request()
    result = WorkloadMeasurement("epoch", request.digest, (1.0,))
    with pytest.raises(BenchmarkObjectiveError, match="another workload"):
        result.validate("epoch", _request(2))


def test_session_epoch_is_readonly() -> None:
    session = BenchmarkObjectiveSession(FakePipeline())
    original = session.epoch
    attribute = "epoch"
    with pytest.raises(AttributeError):
        setattr(session, attribute, "unrelated-epoch")
    assert session.epoch == original
    result = session.measure([_request()], _timer)[0]
    assert result.measurement is not None
    assert result.measurement.epoch == original


def test_session_rejects_digest_inconsistent_with_declared_policy() -> None:
    session = BenchmarkObjectiveSession(FakePipeline())
    session._epoch = "wrong-digest"
    with pytest.raises(BenchmarkObjectiveError, match="fresh search epoch"):
        session.extra_cache_key("kernel-key")
    with pytest.raises(BenchmarkObjectiveError, match="fresh search epoch"):
        session.measure([_request()], _timer)


def test_final_timed_failure_discards_only_that_candidates_measurement() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    calls = 0

    def later_corruption(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return value if calls == 1 else value + 100

    bad = BenchmarkRequest.create(Config(block_sizes=[2]), later_corruption, "later")

    def timing(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
        for workload in workloads:
            for _ in range(2):
                workload.reset()
                workload.run()
        return [[1.0, 1.1] for workload in workloads]

    results = session.measure([bad, _request()], timing)
    assert results[0].measurement is None and results[0].accuracy_error
    assert results[1].measurement is not None
    assert calls == 3  # Initial validation plus the two timed calls; no rerun.
    for ordinal in (1, 2):
        actions = [name for name, candidate in objective.events if candidate == ordinal]
        assert actions.count("reset") == 3
        assert actions.count("validate") == 2
        assert actions[-2:] == ["validate", "close"]


def test_postcheck_observes_replayed_mutation_of_capture_output() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)

    def capture_then_replay(
        workloads: Sequence[BenchmarkWorkload],
    ) -> list[list[float]]:
        workload = workloads[0]
        workload.reset()
        captured_output = cast("torch.Tensor", workload.run())
        # CPU model of replay updating stable captured output buffers without
        # invoking the Python run callback again.
        captured_output.fill_(99)
        return [[1.0, 1.0]]

    result = session.measure([_request()], capture_then_replay)[0]
    assert result.measurement is None and result.accuracy_error
    actions = [name for name, _ in objective.events]
    assert actions.count("reset") == 2
    assert actions.count("candidate") == 2
    assert actions[-2:] == ["validate", "close"]


def test_timer_cannot_return_a_score_without_invoking_the_workload() -> None:
    objective = FakePipeline()
    session = BenchmarkObjectiveSession(objective)
    with pytest.raises(BenchmarkObjectiveError, match="did not invoke"):
        session.measure([_request()], lambda workloads: [[1.0]])
    assert objective.events[-1][0] == "close"


def test_postcheck_accepts_a_legitimate_none_output() -> None:
    class NoneOutput(FakePipeline):
        def bind(
            self, request: BenchmarkRequest, private_args: Sequence[object]
        ) -> BenchmarkWorkload:
            def validate(output: object) -> None:
                assert output is None

            return BenchmarkWorkload(
                lambda: None,
                lambda: None,
                validate,
                lambda: self.events.append(("close", 1)),
            )

    def timing(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
        for workload in workloads:
            workload.reset()
            assert workload.run() is None
        return [[1.0]]

    objective = NoneOutput()
    result = BenchmarkObjectiveSession(objective).measure([_request()], timing)[0]
    assert result.measurement is not None
    assert objective.events == [("close", 1)]


def test_postcheck_resource_failure_is_not_an_accuracy_rejection() -> None:
    class FailingPostcheck(FakePipeline):
        def bind(
            self, request: BenchmarkRequest, private_args: Sequence[object]
        ) -> BenchmarkWorkload:
            workload = super().bind(request, private_args)
            checks = 0

            def validate(output: object) -> None:
                nonlocal checks
                checks += 1
                if checks == 2:
                    raise RuntimeError("postcheck failed")
                workload.validate(output)

            return BenchmarkWorkload(
                workload.run, workload.reset, validate, workload.close
            )

    def timing(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
        workloads[0].reset()
        workloads[0].run()
        return [[1.0]]

    objective = FailingPostcheck()
    with pytest.raises(RuntimeError, match="postcheck failed"):
        BenchmarkObjectiveSession(objective).measure([_request()], timing)
    assert objective.events[-1][0] == "close"


def test_postcheck_policy_mutation_invalidates_the_batch() -> None:
    class MutatingPostcheck(FakePipeline):
        def bind(
            self, request: BenchmarkRequest, private_args: Sequence[object]
        ) -> BenchmarkWorkload:
            workload = super().bind(request, private_args)
            checks = 0

            def validate(output: object) -> None:
                nonlocal checks
                checks += 1
                workload.validate(output)
                if checks == 2:
                    self.policy["partner"] = "changed after timing"

            return BenchmarkWorkload(
                workload.run, workload.reset, validate, workload.close
            )

    def timing(workloads: Sequence[BenchmarkWorkload]) -> list[list[float]]:
        workloads[0].reset()
        workloads[0].run()
        return [[1.0]]

    objective = MutatingPostcheck()
    with pytest.raises(BenchmarkObjectiveError, match="fresh search"):
        BenchmarkObjectiveSession(objective).measure([_request()], timing)
    assert objective.events[-1][0] == "close"
