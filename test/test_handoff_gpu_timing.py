from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from helion._testing import DEVICE
from helion.autotuner.benchmark_provider import _clone_args
from helion.autotuner.benchmarking import do_bench_cuda_graph
from helion.autotuner.handoff_evaluation import _alias_signature
from helion.autotuner.handoff_evaluation import _EvaluateCase
from helion.autotuner.handoff_evaluation import _input_reset
import helion.runtime as helion_runtime

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def test_graph_timer_excludes_host_dispatch() -> None:
    x = torch.ones(16, device=DEVICE)
    calls = []

    def run() -> torch.Tensor:
        calls.append(1)
        time.sleep(0.04)
        return x + 1

    milliseconds = do_bench_cuda_graph(run, fixed_repetitions=3, return_mode="median")
    # A 40 ms host delay runs during warmup/capture, never inside timed replays.
    assert calls == [1, 1]
    assert isinstance(milliseconds, float)
    assert 0 < milliseconds < 20


def test_graph_timer_propagates_capture_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.ones(16, device=DEVICE)
    monkeypatch.setattr(
        helion_runtime,
        "cute_cuda_graph",
        Mock(side_effect=RuntimeError("capture failed")),
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        do_bench_cuda_graph(lambda: x + 1, fixed_repetitions=1)


def test_graph_reset_preserves_overlapping_offset_views() -> None:
    storage = torch.arange(33, dtype=torch.float32, device=DEVICE)
    inputs = (storage[1:17], storage[9:25])
    args = cast(
        "tuple[torch.Tensor, torch.Tensor]",
        _clone_args(inputs, None, preserve_storage=True),
    )
    output = None

    def run() -> torch.Tensor:
        nonlocal output
        args[0].add_(1)
        output = args[1]
        return output

    do_bench_cuda_graph(run, fixed_repetitions=7, reset=_input_reset(inputs, args))
    expected = storage.clone()
    expected[1:17] += 1
    torch.testing.assert_close(output, expected[9:25])
    assert args[0].storage_offset() == 1
    assert args[1].storage_offset() == 9
    assert args[0].untyped_storage() == args[1].untyped_storage()
    torch.testing.assert_close(storage, torch.arange(33, device=DEVICE).float())


@pytest.mark.parametrize("error", [None, "visible", "outside"])
def test_native_graph_evaluation_checks_replayed_mutation(
    tmp_path: Path, error: str | None
) -> None:
    source = (
        "calls = 0\n"
        "def kernel(x):\n"
        "    global calls\n"
        "    calls += 1\n"
        f"    x.add_(2 if {error == 'visible'} and calls > 1 else 1)\n"
        f"    if {error == 'outside'} and calls > 1:\n"
        "        x.as_strided((17,), (1,), storage_offset=0)[:1].fill_(999)\n"
        "    return x\n"
    )
    source_path = tmp_path / "kernel.py"
    source_path.write_text(source)
    args = (torch.arange(17, dtype=torch.float32, device=DEVICE)[1:],)
    expected = _clone_args(args, None, preserve_storage=True)
    output = cast("torch.Tensor", expected[0]).add_(1)
    torch.save(args, tmp_path / "inputs.pt")
    torch.save((output, expected), tmp_path / "reference.pt")
    outcome = _EvaluateCase(
        str(source_path),
        hashlib.sha256(source.encode()).hexdigest(),
        "kernel",
        str(tmp_path / "inputs.pt"),
        str(tmp_path / "reference.pt"),
        "cuda_graph",
        0.0,
        0.0,
        False,
        2,
        _alias_signature((output, expected)),
        str(tmp_path / "progress.jsonl"),
    )()
    assert outcome.status == ("accuracy_error" if error else "ok"), outcome.error
    assert outcome.phase == ("validation" if error else "complete")
    assert len(outcome.samples) == (0 if error else 2)
    if error == "outside":
        assert "outside input views" in (outcome.error or "")
