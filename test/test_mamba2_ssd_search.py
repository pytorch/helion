from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.cute.mamba2_ssd_bootstrap import ReadonlyArguments
from benchmarks.cute.mamba2_ssd_inputs import sha
from benchmarks.cute.mamba2_ssd_inputs import stages
from benchmarks.cute.mamba2_ssd_search import compile_defaults
from benchmarks.cute.mamba2_ssd_search import run_sweep
import pytest
import torch

from test.test_mamba2_ssd_bootstrap import Bound
from test.test_mamba2_ssd_programs import small as small

from helion import exc
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkRequest


class Timing:
    def cache_record(self):
        return {"CPU_TEST_ONLY": True}


def test_real_mamba_role_callbacks_bootstrap_and_complete_tuple(small, tmp_path):
    binding, _ = small
    bundle = binding["bundle"]
    order = stages(binding["case"])
    bounds = []

    class EagerBound(Bound):
        def __init__(self, role, original):
            super().__init__(seeds=(2,))
            self.role = role
            self.original = original

        def compile_config(self, config):
            self.calls.append(("compile", config.block_sizes[0]))
            if self.role == 0 and len(order) == 4 and config.block_sizes == [1]:
                raise exc.BackendUnsupported(
                    "cute", "CPU stand-in for inherited state default"
                )

            def compiled(*args):
                return self.original(*args)

            return compiled

    kernels = {}
    for role, name in enumerate(order):
        bound = EagerBound(role, binding["requests"][role].compiled)
        bounds.append(bound)
        kernels[name] = SimpleNamespace(bind=lambda args, bound=bound: bound)
    with (
        patch.dict(bundle["objective_binding"], {"kernels": kernels}),
        patch("benchmarks.cute.mamba2_ssd_search.configure", lambda k: None),
    ):
        result = compile_defaults(binding["case"], bundle, tmp_path, lambda: None)
    assert len(result["bootstrap_members"]) == len(order)
    assert all(
        m.receipt["native_validation"] == "pending_cpu_standin_only"
        for m in result["bootstrap_members"]
    )
    assert all(
        m.result is r
        for m, r in zip(
            result["bootstrap_members"], result["representatives"], strict=True
        )
    )
    assert [b.seed_queries for b in bounds] == (
        [1, 0, 0, 0] if len(order) == 4 else [0]
    )
    seen = []

    def epoch(bound, args, *, objective, timing, output):
        seen.append(objective.selected)
        return result["requests"][objective.selected]

    final = run_sweep(
        result,
        provenance={"CPU": True},
        timing=Timing(),
        output=tmp_path / "epochs",
        synchronize=lambda: None,
        run_epoch=epoch,
    )
    assert seen == list(range(len(order))) and len(final["requests"]) == len(order)
    # Bootstrap authority remains live across the API handoff.
    bundle["inputs"]["x"].reshape(-1)[0].add_(1)
    with pytest.raises(BenchmarkObjectiveError, match="bytes"):
        run_sweep(
            result,
            provenance={"CPU": True},
            timing=Timing(),
            output=tmp_path / "bad",
            synchronize=lambda: None,
            run_epoch=epoch,
        )


def test_bad_complete_tuple_never_enters_full107(small, tmp_path):
    binding, _ = small
    last = binding["requests"][-1]

    def corrupt(*args):
        return last.compiled(*args) + 1

    binding["requests"][-1] = BenchmarkRequest.create(
        last.config, corrupt, sha(__file__)
    )

    def forbidden(*args, **kwargs):
        raise RuntimeError("entered full107")

    with pytest.raises(AssertionError):
        run_sweep(
            binding,
            provenance={},
            timing=Timing(),
            output=tmp_path,
            synchronize=lambda: None,
            run_epoch=forbidden,
        )


def test_raw_mutation_at_epoch_boundary_aborts_before_recapture(small, tmp_path):
    binding, _ = small
    seen = []

    def epoch(bound, args, *, objective, timing, output):
        seen.append(objective.selected)
        binding["bundle"]["inputs"]["x"].reshape(-1)[0].add_(0.03125)
        return binding["requests"][objective.selected]

    with pytest.raises(BenchmarkObjectiveError, match="bytes"):
        run_sweep(
            binding,
            provenance={},
            timing=Timing(),
            output=tmp_path,
            synchronize=lambda: None,
            run_epoch=epoch,
        )
    assert seen == [0]


def test_generic_readonly_guard_tracks_current_owner_and_nested_aliases():
    raw = torch.arange(8.0)
    args = ({"pair": (raw[:4], raw[4:])},)
    guard = ReadonlyArguments(args)
    guard.validate(args)
    args[0]["pair"][1].set_(raw.clone()[4:])
    with pytest.raises(BenchmarkObjectiveError, match="owner"):
        guard.validate(args)
