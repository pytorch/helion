from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner.accuracy import assert_close
from helion.autotuner.accuracy import relative_l2_disagreement
from helion.autotuner.accuracy import tensor_dtypes
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.kernel_args import _clone_args
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _callback(capture, tensor):
    if capture == "closure":

        def callback():
            return tensor

    elif capture == "default":

        def callback(value=tensor):
            return value

    else:

        def callback(*, value=tensor):
            return value

    return callback


def _mutate_callback_reference(x, callback, tail):
    callback().add_(3)
    # The return cannot reveal an incorrect write to the independent operand.
    return x.clone()


@helion.kernel(
    autotune_baseline_fn=_mutate_callback_reference,
    autotune_baseline_atol=0,
    autotune_baseline_rtol=0,
    autotune_effort="none",
)
def _mutate_callback(x, callback, tail):
    value = callback()
    for tile in hl.tile(x.numel()):
        value[tile] = value[tile] + 3
    return x


@pytest.mark.parametrize("capture", ["closure", "default", "keyword_default"])
def test_callback_baseline_detects_original_pytree_slot(monkeypatch, capture):
    x = torch.zeros(8)
    tensor = torch.arange(8, dtype=torch.float32)
    tail = torch.ones(4)
    args = (x, _callback(capture, tensor), tail)
    # A closure is the actual frontend-admitted kernel witness. Default-only
    # tensor callbacks currently fail earlier FakeTensor conversion; test their
    # provider operand contract using the same real kernel/settings, without
    # claiming that separate frontend path is admitted.
    bind_args = (x, _callback("closure", tensor), tail)
    with cpu_render_target():
        bound = cpu_bind(_mutate_callback, bind_args)
        bound.to_code(bound.config_spec.default_config())
    provider = object.__new__(LocalBenchmarkProvider)
    provider.kernel = bound
    provider.settings = bound.settings
    provider.args = args
    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    output, mutated, post_args = provider._compute_baseline()
    assert mutated == [1]
    assert torch.equal(output, x)
    assert torch.equal(post_args[1](), tensor + 3)
    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))

    provider._baseline_output = output
    provider._baseline_post_args = post_args
    provider.mutated_arg_indices = mutated
    provider._effective_atol = provider._effective_rtol = 0
    provider._scale_atol = False
    provider.log = SimpleNamespace(warning=lambda message: None)
    for corrupt in (False, True):
        candidate_args = _clone_args(args, None, idx_to_clone=mutated)
        assert candidate_args[0] is x and candidate_args[2] is tail
        assert candidate_args[1]().untyped_storage() is not tensor.untyped_storage()
        candidate_output = _mutate_callback_reference(*candidate_args)
        if corrupt:
            candidate_args[1]().add_(1)
        assert (
            provider._validate_against_baseline(
                helion.Config(), candidate_output, candidate_args
            )
            is not corrupt
        )
        assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))


@pytest.mark.parametrize("capture", ["closure", "default", "keyword_default"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_worker_checks_callback_post_call_tensors(monkeypatch, capture, corrupt):
    tensor = torch.arange(8, dtype=torch.float32)
    args = (torch.zeros(8), _callback(capture, tensor), torch.ones(4))
    expected_args = _clone_args(args, None)
    output = _mutate_callback_reference(*expected_args)

    def candidate(x, callback, tail):
        result = _mutate_callback_reference(x, callback, tail)
        if corrupt:
            callback().add_(1)
        return result

    monkeypatch.setattr(
        benchmark_job, "_load_compiled_fn_for_worker", lambda spec: candidate
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: args)
    monkeypatch.setattr(
        benchmark_job,
        "_load_trusted_baseline_output",
        lambda path: output if path == "output" else expected_args,
    )
    monkeypatch.setattr(benchmark_job, "synchronize_device", lambda: None)
    result = benchmark_job.AccuracyCheckJob(
        None,
        "args",
        "output",
        atol=0,
        rtol=0,
        mutated_arg_indices=(1,),
        baseline_post_args_path="post-args",
    )()
    assert result.ok is not corrupt
    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))


def test_callback_cycles_and_aliases_keep_comparison_and_dtype_coverage():
    @dataclass
    class Record:
        tensor: torch.Tensor
        backref: object = None

    tensor = torch.arange(8, dtype=torch.float32)
    record = Record(tensor)

    def callback():
        return record.tensor

    record.backref = callback
    original = (tensor, callback)
    copied = _clone_args(original, None, idx_to_clone=(1,))
    assert copied[0] is copied[1]()
    assert copied[0] is not tensor
    assert_close(copied, original, atol=0, rtol=0)
    assert relative_l2_disagreement(copied, original, 0.0) is None
    copied[1]().add_(1)
    # Compare only callbacks: the direct tensor must not mask missing traversal.
    with pytest.raises(AssertionError):
        assert_close(copied[1], callback, atol=0, rtol=0)
    assert relative_l2_disagreement(copied[1], callback, 0.0) is not None
    fp8 = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
    assert tensor_dtypes(_callback("closure", fp8)) == {torch.float8_e4m3fn}
