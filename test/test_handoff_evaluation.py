from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from helion.autotuner import benchmarking
from helion.autotuner.benchmark_provider import _clone_args
from helion.autotuner.benchmark_worker import BenchmarkTimeout
import helion.autotuner.handoff_evaluation as evaluation_module
from helion.autotuner.handoff_evaluation import _CaseOutcome
from helion.autotuner.handoff_evaluation import _device_context
from helion.autotuner.handoff_evaluation import _EvaluateCase
from helion.autotuner.handoff_evaluation import evaluate_handoff

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(evaluation_module, "synchronize_device", lambda: None)
    monkeypatch.setattr(benchmarking, "synchronize_device", lambda: None)
    monkeypatch.setattr(benchmarking, "_make_l2_cache_clearer", lambda: lambda: None)


def _bundle(
    directory: Path,
    source: str = "def kernel(x):\n    return x + 1\n",
    *,
    count: int = 1,
    reference: Callable[[torch.Tensor], object] = lambda x: x + 1,
) -> dict[str, Any]:
    cases = []
    for index in range(count):
        case_dir = directory / f"case_{index}"
        case_dir.mkdir()
        (case_dir / "kernel.py").write_text(source)
        args = (torch.arange(16, dtype=torch.float32),)
        reference_args = _clone_args(args, None, preserve_storage=True)
        output = reference(cast("torch.Tensor", reference_args[0]))
        torch.save(args, case_dir / "inputs.pt")
        torch.save((output, reference_args), case_dir / "reference.pt")
        cases.append(
            {
                "source": f"case_{index}/kernel.py",
                "entrypoint": "kernel",
                "inputs": f"case_{index}/inputs.pt",
                "reference": f"case_{index}/reference.pt",
                "timing": "wall_clock",
                "atol": 0.0,
                "rtol": 0.0,
                "scale_atol": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "objective": {
            "aggregation": "geomean",
            "unit": "ms",
            "reference_latencies": None,
        },
        "cases": cases,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _job(directory: Path, repetitions: int = 3) -> _EvaluateCase:
    manifest = json.loads((directory / "manifest.json").read_text())
    case = manifest["cases"][0]
    source = directory / case["source"]
    return _EvaluateCase(
        str(source),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        case["entrypoint"],
        str(directory / case["inputs"]),
        str(directory / case["reference"]),
        case["timing"],
        case["atol"],
        case["rtol"],
        case["scale_atol"],
        repetitions,
        case.get("alias_signature"),
    )


def test_worker_snapshot_survives_source_edits(tmp_path: Path) -> None:
    _bundle(tmp_path)
    first = evaluate_handoff(tmp_path, repetitions=2, timeout=30)
    assert first.ok and first.perf is not None and first.perf > 0
    case = first.cases[0]
    assert len(case.samples) == 2
    assert case.perf is not None and case.noise is not None
    assert case.source_path is not None
    frozen = Path(case.source_path).read_text()
    assert hashlib.sha256(frozen.encode()).hexdigest() == case.source_hash
    saved = json.loads((Path(first.directory) / "evaluation.json").read_text())
    assert saved["created_at"] == first.created_at
    assert saved["cases"][0]["samples"] == list(case.samples)

    edited = "def kernel(x):\n    return x + 2\n"
    (tmp_path / "case_0/kernel.py").write_text(edited)
    second = evaluate_handoff(tmp_path, repetitions=1, timeout=30)
    assert not second.ok and second.perf is None
    assert second.cases[0].status == "accuracy_error"
    assert second.cases[0].source_hash != case.source_hash
    assert Path(case.source_path).read_text() == frozen
    assert (tmp_path / "case_0/kernel.py").read_text() == edited
    assert Path(second.directory, "evaluation.json").is_file()


def test_mutation_is_reset_before_every_measured_call(tmp_path: Path) -> None:
    _bundle(
        tmp_path,
        "def kernel(x):\n    return x.add_(1)\n",
        reference=lambda x: x.add_(1),
    )
    pristine = torch.load(tmp_path / "case_0/inputs.pt", weights_only=False)
    outcome = _job(tmp_path)()
    assert outcome.status == "ok", outcome.error
    assert len(outcome.samples) == 3
    assert all(sample > 0 for sample in outcome.samples)
    saved = torch.load(tmp_path / "case_0/inputs.pt", weights_only=False)
    torch.testing.assert_close(saved, pristine)


def test_undeclared_mutation_fails_despite_correct_output(tmp_path: Path) -> None:
    _bundle(
        tmp_path,
        "def kernel(x):\n    x.add_(1)\n    return x - 1\n",
        reference=lambda x: x.clone(),
    )
    outcome = _job(tmp_path)()
    assert outcome.status == "accuracy_error"
    assert outcome.samples == ()


def test_small_undeclared_mutation_ignores_output_tolerances(tmp_path: Path) -> None:
    manifest = _bundle(
        tmp_path,
        "def kernel(x):\n    x.add_(0.0001)\n    return x + 1\n",
    )
    manifest["cases"][0].update(atol=0.1, rtol=0.1)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    outcome = _job(tmp_path)()
    assert outcome.status == "accuracy_error"
    assert outcome.samples == ()


def test_output_alias_loss_is_rejected(tmp_path: Path) -> None:
    _bundle(tmp_path, "def kernel(x):\n    return x.clone()\n", reference=lambda x: x)
    outcome = _job(tmp_path)()
    assert outcome.status == "accuracy_error"
    assert "aliases" in (outcome.error or "")


def test_original_alias_contract_can_differ_from_numerical_reference(
    tmp_path: Path,
) -> None:
    manifest = _bundle(
        tmp_path, "def kernel(x):\n    return x.clone()\n", reference=lambda x: x
    )
    manifest["cases"][0]["alias_signature"] = [0, 1]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    outcome = _job(tmp_path)()
    assert outcome.status == "ok", outcome.error


def test_unchanged_nan_inputs_compare_exactly(tmp_path: Path) -> None:
    _bundle(tmp_path, "def kernel(x):\n    return x.new_zeros(x.shape)\n")
    args = (torch.tensor([float("nan"), 1.0]),)
    torch.save(args, tmp_path / "case_0/inputs.pt")
    torch.save((torch.zeros(2), args), tmp_path / "case_0/reference.pt")
    outcome = _job(tmp_path)()
    assert outcome.status == "ok", outcome.error


def test_nondefault_device_context_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []

    @contextmanager
    def device_context(device: torch.device):
        observed.append(("enter", device))
        yield
        observed.append(("exit", device))

    monkeypatch.setattr(torch.cuda, "device", device_context)
    tensor = Mock(spec=torch.Tensor, device=torch.device("cuda:1"))
    with _device_context((tensor,)):
        assert observed == [("enter", torch.device("cuda:1"))]
    assert observed[-1] == ("exit", torch.device("cuda:1"))
    second = Mock(spec=torch.Tensor, device=torch.device("cuda:0"))
    with pytest.raises(ValueError, match="single accelerator"):
        _device_context((tensor, second))


@pytest.mark.parametrize("timing", ["wall_clock", "cuda_graph"])
def test_measured_invocations_are_also_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing: str
) -> None:
    _bundle(
        tmp_path,
        "calls = 0\ndef kernel(x):\n    global calls\n    calls += 1\n    return x.clone() if calls == 1 else x + 1\n",
        reference=lambda x: x.clone(),
    )
    if timing == "cuda_graph":

        def graph_bench(
            fn: Callable[[], object],
            *,
            reset: Callable[[], None],
            **kwargs: Any,
        ) -> float:
            reset()
            fn()
            return 1.0

        monkeypatch.setattr(evaluation_module, "do_bench_cuda_graph", graph_bench)
    outcome = dataclasses.replace(_job(tmp_path), timing=timing)()
    assert outcome.status == "accuracy_error"
    assert outcome.phase == "validation"
    assert len(outcome.samples) == (0 if timing == "cuda_graph" else 1)


@pytest.mark.parametrize(
    "source, message",
    [
        ("def kernel(:\n", "SyntaxError"),
        ("import helion\ndef kernel(x):\n    return x\n", "without importing Helion"),
        (
            "def kernel(x):\n    raise RuntimeError('broken candidate')\n",
            "broken candidate",
        ),
    ],
)
def test_invalid_source_is_recorded(tmp_path: Path, source: str, message: str) -> None:
    _bundle(tmp_path, source)
    outcome = _job(tmp_path)()
    assert outcome.status in ("load_error", "error")
    assert message in (outcome.error or "")


def test_timeout_keeps_the_failed_source(tmp_path: Path) -> None:
    _bundle(
        tmp_path,
        "import time\ndef kernel(x):\n    time.sleep(60)\n    return x + 1\n",
    )
    result = evaluate_handoff(tmp_path, repetitions=1, timeout=8)
    assert not result.ok and result.perf is None
    assert result.cases[0].status == "timeout"
    assert result.cases[0].source_path is not None
    assert Path(result.cases[0].source_path).is_file()
    assert Path(result.directory, "evaluation.json").is_file()


@pytest.mark.parametrize("aggregation", ["geomean", "max"])
@pytest.mark.parametrize("references", [None, [2.0, 2.0]])
def test_aggregation_and_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregation: str,
    references: list[float] | None,
) -> None:
    manifest = _bundle(tmp_path, count=2)
    manifest["objective"] = {
        "aggregation": aggregation,
        "unit": "ratio" if references else "ms",
        "reference_latencies": references,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    class Worker:
        def __init__(self, device: int | None) -> None:
            self.index = 0

        def run(self, job: _EvaluateCase, timeout: float) -> _CaseOutcome:
            self.index += 1
            value = 2.0 * self.index
            return _CaseOutcome("ok", (value, value + 2.0, value + 4.0))

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(evaluation_module, "BenchmarkWorker", Worker)
    result = evaluate_handoff(tmp_path)
    expected = math.sqrt(4.0 * 6.0) if aggregation == "geomean" else 6.0
    if references:
        expected /= 2.0
    assert result.ok
    assert result.perf == pytest.approx(expected)
    assert [case.noise for case in result.cases] == [2.0, 2.0]
    assert [case.perf for case in result.cases] == [4.0, 6.0]
    assert json.loads(Path(result.directory, "evaluation.json").read_text())["ok"]


def test_all_sources_are_frozen_before_first_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _bundle(tmp_path, count=2)
    original = (tmp_path / manifest["cases"][1]["source"]).read_text()

    class Worker:
        def __init__(self, device: int | None) -> None:
            self.index = 0

        def run(self, job: _EvaluateCase, timeout: float) -> _CaseOutcome:
            if self.index == 0:
                (tmp_path / manifest["cases"][1]["source"]).write_text(
                    "invalid replacement"
                )
            self.index += 1
            assert Path(job.source_path).read_text() == original
            return _CaseOutcome("ok", (1.0,))

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(evaluation_module, "BenchmarkWorker", Worker)
    assert evaluate_handoff(tmp_path).ok


def test_failed_case_disables_aggregate_and_records_partial_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundle(tmp_path, count=2)

    class Worker:
        def __init__(self, device: int | None) -> None:
            self.index = 0

        def run(self, job: _EvaluateCase, timeout: float) -> _CaseOutcome:
            self.index += 1
            if self.index == 1:
                return _CaseOutcome("ok", (1.0,))
            return _CaseOutcome("accuracy_error", (0.1,), "wrong result")

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(evaluation_module, "BenchmarkWorker", Worker)
    result = evaluate_handoff(tmp_path)
    assert not result.ok and result.perf is None
    assert result.cases[1].samples == (0.1,)
    assert result.cases[1].perf is None


def test_clone_preserves_offsets_aliases_and_default_behavior() -> None:
    storage = torch.arange(33, dtype=torch.float32)
    offset = storage[1:17]
    default = cast("torch.Tensor", _clone_args((offset,), None)[0])
    preserved = cast(
        "torch.Tensor", _clone_args((offset,), None, preserve_storage=True)[0]
    )
    assert default.storage_offset() == 0
    assert preserved.storage_offset() == 1
    assert preserved.untyped_storage().nbytes() == storage.untyped_storage().nbytes()
    assert preserved.data_ptr() % 16 == offset.data_ptr() % 16
    overlapping = storage[9:25]
    copied = cast(
        "tuple[torch.Tensor, ...]",
        _clone_args((offset, overlapping, offset), None, preserve_storage=True),
    )
    assert copied[0] is copied[2]
    assert copied[0].untyped_storage() == copied[1].untyped_storage()
    copied[0][8] = -1
    assert copied[1][0] == -1
    assert storage[9] == 9


def test_offset_view_survives_serialization_and_evaluation(tmp_path: Path) -> None:
    _bundle(tmp_path)
    args = (torch.arange(17, dtype=torch.float32)[1:],)
    torch.save(args, tmp_path / "case_0/inputs.pt")
    torch.save((args[0] + 1, args), tmp_path / "case_0/reference.pt")
    (tmp_path / "case_0/kernel.py").write_text(
        "def kernel(x):\n    assert x.storage_offset() == 1\n    assert x.data_ptr() % 16 == 4\n    return x + 1\n"
    )
    outcome = _job(tmp_path)()
    assert outcome.status == "ok", dataclasses.asdict(outcome)


def test_deadline_prevents_dispatch_and_caps_each_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundle(tmp_path, count=2)
    clock = [100.0]
    timeouts = []

    class Worker:
        def __init__(self, device: int | None) -> None:
            pass

        def run(self, job: _EvaluateCase, timeout: float) -> _CaseOutcome:
            timeouts.append(timeout)
            clock[0] += 3
            return _CaseOutcome("ok", (1.0,))

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(evaluation_module, "BenchmarkWorker", Worker)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    result = evaluate_handoff(tmp_path, timeout=120, deadline=102.0)
    assert timeouts == [2.0]
    assert not result.ok and result.perf is None
    assert result.cases[1].status == "timeout"
    assert "before case started" in (result.cases[1].error or "")


def test_timeout_retains_last_complete_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundle(tmp_path)

    class Worker:
        def __init__(self, device: int | None) -> None:
            pass

        def run(self, job: _EvaluateCase, timeout: float) -> _CaseOutcome:
            assert job.progress_path is not None
            Path(job.progress_path).write_text(
                '{"phase": "load", "timestamp": 1}\n'
                '{"phase": "warmup", "timestamp": 2, "samples": [1.0, 1.1]}\n'
                '{"phase": "val'
            )
            raise BenchmarkTimeout("worker timed out")

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(evaluation_module, "BenchmarkWorker", Worker)
    result = evaluate_handoff(tmp_path)
    assert result.cases[0].phase == "warmup"
    assert result.cases[0].status == "timeout"
    assert result.cases[0].samples == (1.0, 1.1)
    assert result.cases[0].perf is None and result.perf is None
    assert result.completed_at is not None
    assert result.cases[0].progress_path is not None


def test_final_journal_error_does_not_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundle(tmp_path)
    dumps = json.dumps

    def fail_final_record(value: object, **kwargs: Any) -> str:
        if isinstance(value, dict) and value.get("phase") == "complete":
            raise OSError("journal unavailable")
        return dumps(value, **kwargs)

    monkeypatch.setattr(json, "dumps", fail_final_record)
    job = dataclasses.replace(
        _job(tmp_path), progress_path=str(tmp_path / "progress.jsonl")
    )
    outcome = job()
    assert outcome.status == "error"
    assert "journal unavailable" in (outcome.error or "")
