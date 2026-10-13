from __future__ import annotations

import gc
import hashlib
from pathlib import Path

from benchmarks.cute.recurrent_decode_inputs import build
from benchmarks.cute.recurrent_decode_objective import ProjectedObjective
from benchmarks.cute.recurrent_decode_objective import make_objective
from benchmarks.cute.recurrent_decode_objective import selected_guards
import pytest
import torch

from test.test_recurrent_decode_programs import eager_kernel
from test.test_recurrent_decode_programs import small_case

from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.kernel_args import _clone_args


@pytest.mark.parametrize(
    "family,tokens",
    [
        ("gdn-decode", 1),
        ("kda-decode", 1),
        ("gdn-mtp", 2),
        ("gdn-mtp", 4),
        ("mamba2-decode", 1),
        ("retention-decode", 1),
    ],
)
def test_full_objective_guards_and_second_update(family: str, tokens: int) -> None:
    case = small_case(family, tokens)
    bundle = build(case, "helion")
    objective = make_objective(bundle, case, {"test": "cpu"})
    eager = eager_kernel(next(iter(bundle["kernels"].values())))
    observed = []

    def invoke(*args: object) -> object:
        observed.append(tuple(id(x) for x in args))
        return eager(*args)

    request = BenchmarkRequest(
        "{}", invoke, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    rows = selected_guards(objective, request, bundle, case, {"test": "cpu"})
    assert len(rows) == 5
    private = tuple(_clone_args(objective.raw_args, None))
    work = objective.bind(request, private)
    try:
        work.reset()
        work.validate(work.run())
        first = observed[-1]
        gc.collect()
        work.reset()
        work.validate(work.run())
        assert observed[-1] == first  # Temporary squeeze views survived between calls.
        private[0][0].fill_(1)
        with pytest.raises(AssertionError):
            work.validate(work.run())
    finally:
        work.close()


def test_inactive_checkpoint_pool_and_output_corruptions() -> None:
    case = small_case("gdn-decode")
    bundle = build(case, "helion")
    objective = make_objective(bundle, case, {"test": "cpu"})
    assert isinstance(objective, ProjectedObjective)
    eager = eager_kernel(next(iter(bundle["kernels"].values())))

    def invoke(*args: object) -> object:
        return eager(*args)

    request = BenchmarkRequest(
        "{}", invoke, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    for target in ("checkpoint", "pool", "output"):
        private = tuple(_clone_args(objective.raw_args, None))
        work = objective.bind(request, private)
        try:
            work.reset()
            output = work.run()
            if target == "checkpoint":
                private[0][10].fill_(1)
            elif target == "pool":
                private[3][:1].add_(1)
            else:
                private[0][9].fill_(float("nan"))
            with pytest.raises(AssertionError):
                work.validate(output)
        finally:
            work.close()


@pytest.mark.parametrize(
    "route",
    [
        "fla-recurrent-prepared",
        "fla-chunk-prepared",
        "fla-chunk-prepared-gamma",
        "fla-recurrent-prepared-gamma",
    ],
)
def test_retention_baseline_adapters_preserve_boundary(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    import importlib
    from types import SimpleNamespace

    from benchmarks.cute.recurrent_decode_inputs import _retention_reference
    from benchmarks.cute.recurrent_decode_objective import baseline_workload

    calls = []

    def fla(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        state = kwargs["initial_state"]
        chunk = "chunk_size" in kwargs
        if chunk:
            assert state.is_contiguous()
            state = state.transpose(-2, -1).contiguous()
        output = torch.empty_like(kwargs["v"])
        final = torch.empty_like(state)
        _retention_reference(
            kwargs["q"], kwargs["k"], kwargs["v"], state, final, output, kwargs["scale"]
        )
        return output, final.transpose(-2, -1).contiguous() if chunk else final

    module = SimpleNamespace(
        fused_recurrent_retention=fla,
        chunk_simple_gla=fla,
        fused_recurrent_simple_gla=fla,
    )
    original_import = importlib.import_module

    def load(name: str, package: str | None = None) -> object:
        if name.startswith("fla.ops.retention"):
            return module
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", load)
    case = small_case("retention-decode")
    bundle = build(case, route)
    work = baseline_workload(bundle)
    work.reset()
    work.validate(work.run())
    first_gamma = calls[-1]["g_gamma"]
    work.reset()
    work.validate(work.run())
    assert torch.equal(first_gamma, calls[-1]["g_gamma"])
    if route.endswith("-gamma"):
        assert first_gamma is calls[-1]["g_gamma"]
    assert ("chunk_size" in calls[-1]) == ("chunk" in route)
    before = bundle["inputs"]["state"].clone()
    last_output = work.run()
    bundle["inputs"]["q"].fill_(1)
    with pytest.raises(AssertionError, match="readonly"):
        work.validate(last_output)
    assert torch.equal(bundle["inputs"]["state"], before)
    work.close()
