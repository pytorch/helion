from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_vector_group_integration import _config
from .test_cute_chained_vector_group_integration import _root_distinct_leaves
from .test_cute_chained_vector_group_integration import _root_shared_gram
from helion import exc
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

_CASES = (
    ("gram", 0),
    ("gram", 1),
    ("gram", 2),
    ("distinct", 16),
    ("distinct", 64),
    ("distinct", 128),
)


def _fixture(device, dtype, family, variant, alternate, *, rows=256, columns=256):
    generator = torch.Generator(device=device).manual_seed(65713)
    reduction = 32 if family == "gram" else variant
    shapes = [(rows, reduction)]
    if family == "distinct":
        shapes.append((128, reduction))
    shapes.append((128, columns))
    values = []
    for index, shape in enumerate(shapes):
        if alternate and index == 0:
            value = torch.empty((shape[0], shape[1] * 2), dtype=dtype, device=device)[
                :, ::2
            ]
        else:
            value = torch.empty(shape, dtype=dtype, device=device)
        value.copy_(torch.randn(shape, device=device, generator=generator) * 0.0625)
        values.append(value)
    if family == "gram":
        values.append(variant)
    config = _config("root", False)
    config.config["cute_chained_tmem_early_release"] = alternate
    return (
        _root_shared_gram if family == "gram" else _root_distinct_leaves,
        tuple(values),
        config,
    )


def _reference(values, family):
    x, weight = values[0], values[1] if family == "gram" else values[2]
    output = torch.empty(
        (x.shape[0], weight.shape[1]), dtype=torch.float32, device=x.device
    )
    for begin in range(0, x.shape[0], 128):
        raw = x[begin : begin + 128].float()
        if family == "gram":
            if values[2] == 1:
                mask = (torch.arange(raw.shape[0], device=x.device) + begin) % 2 == 0
                raw = torch.where(mask[:, None], raw, 0.0)
            elif values[2] == 2:
                mask = torch.arange(raw.shape[1], device=x.device) % 2 == 0
                raw = torch.where(mask[None, :], raw, 0.0)
            left = (torch.exp(raw * 0.125) * 0.0625).to(x.dtype)
            right = torch.zeros((128, raw.shape[1]), dtype=x.dtype, device=x.device)
            right[: raw.shape[0]] = left
        else:
            left = torch.exp(raw * 0.125).to(x.dtype)
            right = torch.exp(values[1].float() * 0.125).to(x.dtype)
        # Preserve the graph's FP32 result followed by its explicit low-precision
        # bridge. FP64 contractions are independent of the shared MMA emitter.
        first = (left.double() @ right.double().T).float().to(x.dtype)
        output[begin : begin + 128] = (first.double() @ weight.double()).float()
    return output


@pytest.mark.parametrize("family,variant", _CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("alternate", (False, True))
def test_weighted_pair_original_shapes_preserve_original_source(
    family, variant, dtype, alternate
):
    initialized = torch.cuda.is_initialized()
    kernel, values, config = _fixture("cpu", dtype, family, variant, alternate)
    with _cpu_codegen(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = kernel._bind_isolated(values).to_code(config)
    with (
        _cpu_codegen(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = kernel._bind_isolated(values).to_code(config)
    assert after == before
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    sequence = emitted.call_args.kwargs["root_actions"].sequence
    assert sequence.pair_completed == sequence.next_stage == 2
    assert sequence.pair_bridged and sequence.pair_selection is sequence.pair_inputs
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("family,variant", (("gram", 0), ("distinct", 32)))
def test_partial_root_shape_keeps_original_admission_rejection(family, variant):
    # The original root matcher rejects these partial launch extents before
    # either old or shared physical lowering. Keep that boundary explicit.
    kernel, values, config = _fixture(
        "cpu", torch.bfloat16, family, variant, False, rows=259, columns=192
    )
    for old in (True, False):
        with (
            _cpu_codegen(),
            patch.object(roots, "codegen_plain_root", return_value=False)
            if old
            else patch.object(
                roots, "codegen_plain_root", wraps=roots.codegen_plain_root
            ),
            pytest.raises(exc.InvalidConfig, match="invalid chained MMA schedule"),
        ):
            kernel._bind_isolated(values).to_code(config)


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("family,variant", _CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("alternate", (False, True))
def test_weighted_pair_original_bits_fp64_replay_and_immutable_inputs_gpu(
    family, variant, dtype, alternate
):
    kernel, canonical, config = _fixture(DEVICE, dtype, family, variant, alternate)
    with patch.object(roots, "codegen_plain_root", return_value=False):
        original = kernel._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = kernel._bind_isolated(canonical).compile_config(config)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    for generation in range(2):
        values, tensors = [], []
        for value in canonical:
            if isinstance(value, torch.Tensor):
                tensor = torch.empty_strided(
                    value.shape, value.stride(), dtype=value.dtype, device=value.device
                )
                tensor.copy_(value)
                if generation:
                    tensor.mul_(0.5)
                tensors.append(tensor)
                values.append(tensor)
            else:
                values.append(value)
        saved = tuple(value.clone() for value in tensors)
        actual = shared(*values)
        _bits_equal(actual, original(*values))
        # Keep the original signed inputs for exact migration/replay checks.
        # Near an FP16 bridge midpoint, legitimate FP32 accumulation orders can
        # round to adjacent halves. Signed cancellation in the second dot can
        # amplify that difference, so use a well-conditioned second operand for
        # the independent FP64 check (with the same tolerance and first dot).
        reference_values = list(values)
        if family == "distinct":
            reference_values[2] = values[2].abs() + 0.125
            reference_actual = shared(*reference_values)
            _bits_equal(reference_actual, original(*reference_values))
        else:
            reference_actual = actual
        torch.testing.assert_close(
            reference_actual,
            _reference(reference_values, family),
            rtol=2e-3,
            atol=2e-3,
        )
        for _ in range(3):
            _bits_equal(actual, shared(*values))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = shared(*values)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            _bits_equal(captured, actual)
        for value, before in zip(tensors, saved, strict=True):
            _bits_equal(value, before)
