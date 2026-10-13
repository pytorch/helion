from __future__ import annotations

from itertools import starmap
from pathlib import Path
import sys
from types import ModuleType

from benchmarks.cute.retention_prefill_inputs import BASELINES
from benchmarks.cute.retention_prefill_inputs import STAGES
from benchmarks.cute.retention_prefill_inputs import _retention_reference
from benchmarks.cute.retention_prefill_inputs import arguments
from benchmarks.cute.retention_prefill_inputs import build
from benchmarks.cute.retention_prefill_inputs import cases
from benchmarks.cute.retention_prefill_inputs import make_kernel
from benchmarks.cute.retention_prefill_inputs import prepare_fixed_prefix
from benchmarks.cute.retention_prefill_inputs import sha
from benchmarks.cute.retention_prefill_objective import RAW_NAMES
from benchmarks.cute.retention_prefill_objective import baseline_workload
from benchmarks.cute.retention_prefill_objective import make_objective
from benchmarks.cute.retention_prefill_objective import selected_guards
from examples.linear.retention_prefill import retention_prefill_fixed_decay
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import helion
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.kernel_args import _clone_args
from helion.runtime.ref_mode import RefMode


class CpuDot(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.dtype, torch.ops.aten.bmm.dtype):
            return args[0].float() @ args[1].float()
        return func(*args, **(kwargs or {}))


@pytest.fixture
def case():
    return dict(cases()[0], heads=1, value_heads=1, device="cpu")


@pytest.fixture
def bundle(case):
    return build(case)


def oracle_call(q, k, v, prefix, initial, final, scale, chunk):
    assert chunk == 64 and prefix.shape[1:] == (2, 64)
    output, expected = _retention_reference(q, k, v, initial, scale)
    final.copy_(expected)
    return output, final


def request(compiled=oracle_call):
    return BenchmarkRequest.create(
        helion.Config(block_sizes=[16]), compiled, sha(__file__)
    )


def test_program_settings_and_original_prefix(case, bundle):
    kernel = make_kernel(effort="full", source_workers=8, native_workers=32)
    assert kernel.fn is retention_prefill_fixed_decay.fn
    assert kernel.settings.static_shapes and kernel.settings.backend == "cute"
    assert not kernel.settings.fast_math and not kernel.configs
    assert not kernel.settings.autotune_seed_configs
    assert not kernel.settings.autotune_config_overrides
    assert kernel.settings.autotune_compile_timeout == 120
    assert kernel.settings.autotune_budget_seconds is None
    assert kernel.settings.autotune_max_generations is None
    assert kernel.settings.autotune_best_of_k == 1
    assert tuple(bundle["kernels"]) == STAGES
    log_gamma, prefix = prepare_fixed_prefix(1, 128, 64, torch.device("cpu"))
    assert torch.equal(log_gamma, bundle["inputs"][RAW_NAMES[-2]])
    assert torch.equal(prefix, bundle["inputs"][RAW_NAMES[-1]])
    assert prefix[:, 0].equal(prefix[:, 1])
    args = arguments(bundle["inputs"], case, bundle["metadata"]["scale"])[0]
    assert len(args) == 8 and args[3] is bundle["inputs"][RAW_NAMES[-1]]
    assert (
        args[4] is bundle["inputs"]["state"]
        and args[5] is bundle["inputs"]["final_state"]
    )


def test_public_eager_full_oracle(case, bundle):
    eager = helion.kernel(
        retention_prefill_fixed_decay.fn, ref_mode=RefMode.EAGER, fast_math=False
    )
    args = arguments(bundle["inputs"], case, bundle["metadata"]["scale"])[0]
    pristine = tuple(x.clone() for x in args[:5])
    with CpuDot():
        result = eager(*args)
    assert result[1] is args[5]
    assert result[0].untyped_storage()._cdata not in {
        x.untyped_storage()._cdata for x in args[:6]
    }
    for actual, expected in zip(result, bundle["reference"](), strict=True):
        torch.testing.assert_close(
            actual, expected, atol=0.02, rtol=0.01, equal_nan=False
        )
    assert all(starmap(torch.equal, zip(args[:5], pristine, strict=True)))


def test_original_guards_and_physical_canary(case, bundle):
    shape = bundle["inputs"]["final_state"].shape
    backing = torch.full((bundle["inputs"]["final_state"].numel() + 2,), 77.0)
    bundle["inputs"]["final_state"] = backing[1:-1].view(shape)
    obj = make_objective(bundle, case, {"test": "cpu"})
    assert selected_guards(obj, request()) == [
        "original",
        "output_nan",
        "output_finite",
        "repeat_reset",
        "repeat_without_reset",
    ]
    private = tuple(_clone_args(obj.raw_args, None))
    work = obj.bind(request(), private)
    result = work.run()
    work.validate(result)
    final = private[RAW_NAMES.index("final_state")]
    physical = torch.empty(0).set_(
        final.untyped_storage(), 0, (final.untyped_storage().nbytes() // 4,), (1,)
    )
    physical[0] = 0
    with pytest.raises(AssertionError):
        work.validate(result)
    work.close()


@pytest.mark.parametrize(
    "name", [n for n in RAW_NAMES if n not in ("output", "final_state")]
)
def test_readonly_inputs(case, bundle, name):
    obj = make_objective(bundle, case, {"test": "cpu"})
    private = tuple(_clone_args(obj.raw_args, None))
    work = obj.bind(request(), private)
    result = work.run()
    private[RAW_NAMES.index(name)].flatten()[0].add_(1)
    with pytest.raises(AssertionError):
        work.validate(result)
    work.close()


def test_factory_and_result_owner_rejection(case, bundle):
    obj = make_objective(bundle, case, {"test": "cpu"})
    with pytest.raises(BenchmarkObjectiveError, match="alias factory"):
        obj.bind(request(), obj.raw_args)
    private = tuple(_clone_args(obj.raw_args, None))
    work = obj.bind(request(), private)
    out = work.run()
    with pytest.raises(AssertionError):
        work.validate((out[0].clone(), out[1]))
    out[0].flatten()[0] = float("nan")
    with pytest.raises(AssertionError):
        work.validate(out)
    work.close()
    with pytest.raises(AssertionError):
        work.run()
    obj.raw_args[0].flatten()[0].add_(1)
    with pytest.raises(BenchmarkObjectiveError, match="raw values"):
        obj.cache_record()


@pytest.mark.parametrize("route", BASELINES)
def test_baseline_boundaries(case, bundle, route, monkeypatch):
    suffix, callee, prepared = BASELINES[route]
    calls = []
    expected = bundle["reference"]()

    def operation(**kwargs):
        calls.append(kwargs)
        return tuple(x.clone() for x in expected)

    module = ModuleType(f"fla.ops.retention.{suffix}")
    module.__file__ = __file__
    setattr(module, callee, operation)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    baseline = build(case, route)
    for name in ("q", "k", "v", "state"):
        assert torch.equal(bundle["inputs"][name], baseline["inputs"][name])
    work = baseline_workload(baseline)
    for _ in range(2):
        work.reset()
        work.validate(work.run())
    assert calls[0]["initial_state"] is baseline["inputs"]["state"]
    assert calls[0]["output_final_state"] is True and calls[0]["cu_seqlens"] is None
    if suffix == "chunk":
        assert calls[0]["chunk_size"] is None
    else:
        assert calls[0]["state_v_first"] is False and calls[0]["reverse"] is False
    assert (calls[0]["g_gamma"] is calls[1]["g_gamma"]) is prepared
    baseline["inputs"]["retention_head_index"].add_(1)
    with pytest.raises(AssertionError):
        work.validate(expected)
    work.close()


def test_explicit_config_cannot_seed_search(tmp_path):
    from benchmarks.cute.compare_retention_prefill_backends import run

    with pytest.raises(ValueError, match="cannot seed"):
        run(cases()[0], "helion", output=tmp_path / "unused", config={}, effort="full")
    assert not (tmp_path / "unused").exists()


def test_public_main_and_no_artifact_imports():
    import ast

    root = Path(__file__).resolve().parents[1]
    for relative in (
        "examples/linear/retention_prefill.py",
        "benchmarks/cute/retention_prefill_inputs.py",
        "benchmarks/cute/retention_prefill_objective.py",
        "benchmarks/cute/compare_retention_prefill_backends.py",
    ):
        text = (root / relative).read_text()
        assert "artifacts/" not in text and "sys.path" not in text
        tree = ast.parse(text)
        if relative.endswith(
            ("retention_prefill.py", "compare_retention_prefill_backends.py")
        ):
            assert any(
                isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body
            )
