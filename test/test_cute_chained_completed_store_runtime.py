from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_completed_store import _completed_sequence
from .test_cute_chained_completed_store_search import _config
from .test_cute_chained_leaf_set_runtime import _assert_bits
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_leaves import _source
from helion import exc
from helion._compiler.cute import chained_completed_members as members
from helion._compiler.cute import chained_completed_store as completed
from helion._compiler.cute import chained_preparation_pipeline as pipeline
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

CASES = tuple(
    (dtype, output_dtype, steps, masked)
    for dtype in (torch.bfloat16, torch.float16)
    for output_dtype in (torch.bfloat16, torch.float16, torch.float32)
    for steps in (0, 1, 7)
    for masked in (False, True)
)


def _inputs(dtype, output_dtype, steps, masked, *, device="cpu", seed=20260926):
    generator = torch.Generator().manual_seed(seed)

    def values(shape, kind, scale):
        return (torch.randint(-4, 5, shape, generator=generator).float() * scale).to(
            device=device, dtype=kind
        )

    indices = torch.stack(
        [torch.randperm(32, generator=generator) for _ in range(max(1, steps))]
    )[:steps].to(device=device, dtype=torch.int32)
    mask = torch.randint(0, 2, (steps, 32), generator=generator).bool().to(device)
    return (
        values((steps, 32, 16), dtype, 1 / 128),
        values((steps, 16, 16), dtype, 1 / 128),
        values((steps, 32, 16), dtype, 1 / 128),
        values((steps, 16, 16), dtype, 1 / 128),
        values((128, 16), torch.float32, 1 / 64),
        indices,
        mask,
        output_dtype,
        masked,
    )


def _reference(args):
    a, b, c, d, state, indices, mask = [value.cpu() for value in args[:7]]
    output_dtype, masked = args[7:]
    history = torch.full((a.shape[0], 32, 128), float("nan"), dtype=output_dtype)
    defined = torch.zeros(history.shape, dtype=torch.bool)
    for step in range(a.shape[0]):
        left = (a[step].float() + c[step].float()).to(a.dtype)
        prepared = (left.double() @ b[step].double()).float().to(a.dtype)
        snapshot = state.to(a.dtype)
        result = (prepared.double() @ snapshot.double().T).float()
        state = (snapshot.double() @ d[step].double() + state.double()).float()
        value = torch.tanh(result + 0.125).to(output_dtype)
        valid = mask[step] if masked else torch.ones(32, dtype=torch.bool)
        rows = indices[step][valid].long()
        history[step, rows] = value[valid]
        defined[step, rows] = True
    return (history, state), defined


def _runtime_config(enabled, masked):
    config = _config(enabled)
    if masked:
        config.config["cute_chained_output_lease_snapshot"] = True
    return config


def _defined_values(outputs, defined):
    return outputs[0][defined.to(outputs[0].device)], outputs[1]


@pytest.mark.parametrize("dtype,output_dtype,steps,masked", CASES)
def test_completed_store_public_preflight(dtype, output_dtype, steps, masked):
    args = _inputs(dtype, output_dtype, steps, masked)
    reference, defined = _reference(args)
    assert all(
        torch.isfinite(value).all() for value in _defined_values(reference, defined)
    )
    if not steps:
        _assert_bits(reference[1], args[4])
    observed = []
    emit = pipeline.emit_preparation_pipeline

    def record(*args, **kwargs):
        observed.append(kwargs.get("completed_member_store"))
        return emit(*args, **kwargs)

    with patch.object(pipeline, "emit_preparation_pipeline", record):
        old = _source(_completed_sequence, args, _runtime_config(False, masked))
        new = _source(_completed_sequence, args, _runtime_config(True, masked))
    assert observed == [None, True]
    assert "chain_store_0_register" not in old
    assert "chain_store_0_register" in new
    # The masked output lease keeps its staged carry update. Otherwise the
    # completed store makes the carry resident and removes the extra join
    # that separated the staged carry reads from stores.
    assert "chain_loop_carry_0_next" in old
    assert ("chain_loop_carry_0_next" in new) is masked
    join = "chain_recurrence_barrier.arrive_and_wait()"
    assert old.count(join) - new.count(join) == int(not masked)
    for operation in (
        "cute.arch.mbarrier_wait(",
        "cute.arch.fence_view_async_tmem_load()",
        "chain_sync.arrive_mbarrier(",
        "cute.arch.sync_threads()",
    ):
        assert old.count(operation) == new.count(operation)


def test_false_public_option_bypasses_discovery_and_true_cannot_be_ineffective():
    args = _inputs(torch.bfloat16, torch.float32, 1, False)
    with patch.object(completed, "prepare_completed_store", side_effect=AssertionError):
        _source(_completed_sequence, args, _runtime_config(False, False))
    with (
        patch.object(members, "plan_completed_member_store", return_value=None),
        pytest.raises(exc.InternalError, match="lacks one exclusive result"),
    ):
        _source(_completed_sequence, args, _runtime_config(True, False))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype,output_dtype,steps,masked", CASES)
def test_completed_store_gpu_original_math_bits_replay(
    dtype, output_dtype, steps, masked
):
    generations = [
        _inputs(dtype, output_dtype, steps, masked, device=DEVICE, seed=seed)
        for seed in (20260926, 20260927)
    ]
    compiled = []
    for enabled in (False, True):
        bound = _completed_sequence._bind_isolated(generations[0])
        with bound.env.use_runtime_arg_values(
            _runtime_values(_completed_sequence, generations[0])
        ):
            compiled.append(bound.compile_config(_runtime_config(enabled, masked)))
    for values in generations:
        saved = tuple(value.clone() for value in values[:7])
        reference, defined = _reference(values)
        control, actual = (kernel(*values) for kernel in compiled)
        torch.cuda.synchronize()
        snapshots = tuple(value.clone() for value in actual)
        for value, old, expected in zip(
            _defined_values(actual, defined),
            _defined_values(control, defined),
            _defined_values(reference, defined),
            strict=True,
        ):
            _assert_bits(value, old)
            torch.testing.assert_close(value.cpu(), expected, atol=1e-5, rtol=1e-3)
        for _ in range(3):
            repeated = compiled[1](*values)
            for value, old in zip(
                _defined_values(repeated, defined),
                _defined_values(snapshots, defined),
                strict=True,
            ):
                _assert_bits(value, old)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled[1](*values)
        for _ in range(3):
            for value in captured:
                value.fill_(float("nan"))
            graph.replay()
            for value, old in zip(
                _defined_values(captured, defined),
                _defined_values(snapshots, defined),
                strict=True,
            ):
                _assert_bits(value, old)
        if not steps:
            _assert_bits(actual[1], saved[4])
            _assert_bits(captured[1], saved[4])
        for value, old in zip(values[:7], saved, strict=True):
            _assert_bits(value, old)
