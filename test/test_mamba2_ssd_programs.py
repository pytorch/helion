from __future__ import annotations

from contextlib import contextmanager
import sys
from types import ModuleType
from unittest.mock import patch

from benchmarks.cute.mamba2_ssd_inputs import RAW_NAMES
from benchmarks.cute.mamba2_ssd_inputs import build
from benchmarks.cute.mamba2_ssd_inputs import cases
from benchmarks.cute.mamba2_ssd_inputs import sha
from benchmarks.cute.mamba2_ssd_inputs import stages
from benchmarks.cute.mamba2_ssd_objective import MambaObjective
from benchmarks.cute.mamba2_ssd_objective import baseline_workload
from benchmarks.cute.mamba2_ssd_objective import check_result
from benchmarks.cute.mamba2_ssd_objective import selected_guards
from benchmarks.cute.mamba2_ssd_objective import stage_arguments
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import helion
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.kernel_args import _clone_args
import helion.language as hl
from helion.runtime.ref_mode import RefMode


class CpuDot(TorchDispatchMode):
    """Emulate the original FP32-accumulating MMA in CPU eager evaluation."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.dtype, torch.ops.aten.bmm.dtype):
            return args[0].float() @ args[1].float()
        return func(*args, **(kwargs or {}))


@contextmanager
def scalar_mask_reference():
    original = hl.store._ref_fn

    def ref(tensor, index, value, extra_mask=None):
        if type(extra_mask) is bool:
            assert tensor.device.type == "cpu"
            extra_mask = torch.tensor(extra_mask, dtype=torch.bool)
        return original(tensor, index, value, extra_mask)

    with patch.object(hl.store, "_ref_fn", ref):
        yield


@pytest.fixture(params=[1, 2])
def small(request):
    case = dict(
        cases()[request.param - 1],
        tokens=request.param * 8,
        chunk_size=8,
        heads=4,
        groups=2,
        value_dim=16,
        state_dim=8,
        device="cpu",
    )
    bundle = build(case)
    values, arguments, requests, calls = [], [], [], []

    def wrap(kernel, name):
        def compiled(*args):
            calls.append(name)
            return kernel(*args)

        return compiled

    with CpuDot(), scalar_mask_reference():
        for index, (name, kernel) in enumerate(
            bundle["objective_binding"]["kernels"].items()
        ):
            eager = helion.kernel(kernel.fn, ref_mode=RefMode.EAGER, fast_math=False)
            compiled = wrap(eager, name)
            args = stage_arguments(bundle["inputs"], case, index, values)
            arguments.append(args)
            values.append(compiled(*args))
            requests.append(
                BenchmarkRequest.create(
                    helion.Config(block_sizes=[1]), compiled, sha(__file__)
                )
            )
        calls.clear()
        yield (
            {
                "case": case,
                "bundle": bundle,
                "bounds": [None] * len(requests),
                "args": arguments,
                "requests": requests,
                "representatives": values,
            },
            calls,
        )


def objective(binding, selected=0):
    return MambaObjective(
        binding["bundle"],
        case=binding["case"],
        selected=selected,
        partners={i: r for i, r in enumerate(binding["requests"]) if i != selected},
        provenance={"CPU_EAGER": True},
        timing_policy={"CPU_ONLY": True},
    )


def test_all_roles_full_oracle_and_repeated_call(small):
    binding, calls = small
    for selected in range(len(binding["requests"])):
        obj = objective(binding, selected)
        assert selected_guards(obj, binding["requests"][selected]) == [
            "original",
            "repeat_reset",
            "repeat_without_reset",
        ]
    assert calls == list(stages(binding["case"])) * (3 * len(binding["requests"]))


@pytest.mark.parametrize("name", RAW_NAMES)
def test_every_readonly_input_is_protected(small, name):
    binding, _ = small
    obj = objective(binding)
    private = tuple(_clone_args(obj.raw_args, None))
    work = obj.bind(binding["requests"][0], private)
    out = work.run()
    work.validate(out)
    private[RAW_NAMES.index(name)].flatten()[0].add_(1)
    with pytest.raises(AssertionError):
        work.validate(out)
    work.close()


def test_result_identity_corruption_and_closed_workload(small):
    binding, _ = small
    obj = objective(binding)
    work = obj.bind(binding["requests"][0], tuple(_clone_args(obj.raw_args, None)))
    out = work.run()
    with pytest.raises(AssertionError):
        work.validate((out[0].clone(),))
    out[0].flatten()[0].add_(1)
    with pytest.raises(AssertionError):
        work.validate(out)
    work.close()
    with pytest.raises(AssertionError):
        work.run()


def test_stage_alias_and_dtype_rejected(small):
    binding, _ = small
    output = binding["representatives"][-1]
    used = {output.untyped_storage()._cdata}
    with pytest.raises(AssertionError, match="aliases"):
        check_result(
            output, binding["case"], len(binding["requests"]) - 1, used, output.device
        )
    with pytest.raises(AssertionError):
        check_result(
            output.float(),
            binding["case"],
            len(binding["requests"]) - 1,
            set(),
            output.device,
        )


def test_settings_and_callback_migration(small):
    binding, _ = small
    for name, original in binding["bundle"]["kernels"].items():
        copied = binding["bundle"]["objective_binding"]["kernels"][name]
        assert original.settings.autotune_baseline_fn is not None
        assert copied.settings.autotune_baseline_fn is None
        expected = original.settings.to_dict()
        expected["autotune_baseline_fn"] = None
        assert copied.settings.to_dict() == expected
        assert copied.settings.static_shapes and copied.settings.backend == "cute"
        assert copied.settings.fast_math is False
        assert not copied.configs and not copied.settings.autotune_seed_configs
        assert not copied.settings.autotune_config_overrides


@pytest.mark.parametrize(
    "route", ["flashinfer-cute", "flashinfer-cake", "mamba", "vllm", "sglang"]
)
def test_all_baseline_call_boundaries_and_readonly(route, monkeypatch):
    case = dict(
        cases()[0],
        tokens=8,
        chunk_size=8,
        heads=4,
        groups=2,
        value_dim=16,
        state_dim=8,
        device="cpu",
    )
    expected = build(case)["reference"]()[0]
    calls = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        if "out" in kwargs:
            kwargs["out"].copy_(expected.reshape_as(kwargs["out"]))
            return expected
        if route.startswith("flashinfer"):
            return (expected.clone(),)
        return expected.clone()

    if route.startswith("flashinfer"):
        module = ModuleType("flashinfer.mamba.ssd_combined")

        class SSDCombined:
            def __init__(self, *args, **kwargs):
                assert args == (8, 4, 16, 8, 2)
                assert kwargs["state_dtype"] == torch.bfloat16 and kwargs["has_d"]
                assert kwargs["backend"] == (
                    "cake" if route.endswith("cake") else "cute"
                )

            run = staticmethod(operation)

        module.SSDCombined = SSDCombined
    else:
        path = {
            "mamba": "mamba_ssm.ops.triton.ssd_combined",
            "vllm": "vllm.model_executor.layers.mamba.ops.ssd_combined",
            "sglang": "sglang.kernels.ops.mamba.triton_ops.ssd_combined",
        }[route]
        module = ModuleType(path)
        setattr(
            module,
            "mamba_chunk_scan_combined_varlen"
            if route == "vllm"
            else "mamba_chunk_scan_combined",
            operation,
        )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    bundle = build(case, route)
    work = baseline_workload(bundle)
    work.reset()
    result = work.run()
    work.validate(result)
    args, kwargs = calls[-1]
    if route == "vllm":
        assert kwargs["x"].shape == (8, 4, 16)
        assert (
            kwargs["x"].untyped_storage()._cdata
            == bundle["inputs"]["x"].untyped_storage()._cdata
        )
        assert kwargs["cu_seqlens"].tolist() == [0, 8]
        assert kwargs["cu_chunk_seqlens"].tolist() == [0, 8]
        assert kwargs["seq_idx"].tolist() == [0]
        assert kwargs["last_chunk_indices"].tolist() == [0]
    elif route == "sglang":
        assert args[0] is bundle["inputs"]["x"]
        assert kwargs["return_final_states"] is False
        assert kwargs["dt_softplus"] is False and kwargs["initial_states"] is None
    bundle["inputs"]["dt"].flatten()[0].add_(1)
    with pytest.raises(AssertionError):
        work.validate(result)
    work.close()
