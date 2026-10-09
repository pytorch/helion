from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion.autotuner import FiniteSearch
from helion.autotuner import HandoffPolicy
from helion.autotuner import find_handoff
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("multi_shape", [False, True])
def test_handoff_validates_generated_kernels(
    backend: str, multi_shape: bool, tmp_path: Path
) -> None:
    if backend == "cute":
        pytest.importorskip("cutlass.cute")

    @helion.kernel(
        backend=backend,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
        autotune_accuracy_check=False,
        autotune_log=str(tmp_path / "run"),
        autotune_log_details=True,
    )
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(a)
        for tile in hl.tile(a.size()):
            out[tile] = a[tile] + b[tile]
        return out

    args = (torch.randn(256, device=DEVICE), torch.randn(256, device=DEVICE))
    bound = add.bind(args)
    cases = ((bound, args),)
    search_args: Sequence[object] = args
    if multi_shape:
        second = (torch.randn(2048, device=DEVICE), torch.randn(2048, device=DEVICE))
        cases = (*cases, (add.bind(second), second))
        search_args = cast(
            "Sequence[object]",
            _MultiShapeAutotuneArgs(
                cases=cases,
                aggregation="max",
                relative_to="default",
                cache_tag=None,
                workload_key=("handoff_test",),
            ),
        )
    search = FiniteSearch(
        bound,
        search_args,
        [helion.Config(block_sizes=[64]), helion.Config(block_sizes=[128])],
    )
    point = find_handoff(search, HandoffPolicy(after_trials=1))

    assert point.reason == "trials"
    assert point.progress.trials == 2  # The full batch completes before stopping.
    assert point.progress.unique_sources == 2
    assert point.objective_unit == ("ratio" if multi_shape else "ms")
    assert all(len(candidate.samples) == 3 for candidate in point.finalists)
    assert len(point.measurements) == 8
    assert add.settings.autotune_accuracy_check is False
    torch.testing.assert_close(point.fn(*args), args[0] + args[1])
    for case, inputs in cases:
        torch.testing.assert_close(
            case.compile_config(point.config)(*inputs), inputs[0] + inputs[1]
        )
    if multi_shape:
        assert point.reference_latencies is not None
        assert all(len(m.per_shape) == 2 for m in point.measurements)
    trace = [
        json.loads(line)
        for line in (tmp_path / "run.trace.jsonl").read_text().splitlines()
    ]
    assert len({row["run_id"] for row in trace}) == 1
    assert len([row for row in trace if row["event"] == "trial"]) == 2
    assert len([row for row in trace if row["event"] == "handoff_confirmation"]) == 6
    assert trace[-1]["event"] == "run_end"
    assert trace[-1]["status"] == "ok"
