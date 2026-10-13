from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.linear import linear_attention_preprocessing as preparation
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.nn.functional as F

from ._cute_aux import _cpu_codegen
import helion
from helion.runtime.ref_mode import RefMode

if TYPE_CHECKING:
    from collections.abc import Iterator


def _inputs(
    batch: int,
    tokens: int,
    heads: int,
    value_heads: int,
    key_dim: int = 12,
    value_dim: int = 20,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20261005)

    def random(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=dtype) * 0.2

    return (
        random((batch, tokens, heads, key_dim), torch.bfloat16),
        random((batch, tokens, heads, key_dim), torch.bfloat16),
        random((batch, tokens, value_heads, value_dim), torch.bfloat16),
        random((batch, tokens, value_heads), torch.bfloat16),
        random((batch, tokens, value_heads), torch.bfloat16),
        random((value_heads,), torch.float32),
        random((value_heads,), torch.float32),
        random((batch, value_heads, value_dim, key_dim), torch.float32),
    )


@pytest.fixture
def reference_kernels() -> Iterator[None]:
    kernels = {
        name: helion.kernel(
            getattr(preparation, name).fn,
            backend="cute",
            ref_mode=RefMode.EAGER,
        )
        for name in (
            "pack_gated_delta_inputs_helion",
            "prepare_gated_delta_gates_helion",
        )
    }
    with _cpu_codegen(), patch.multiple(preparation, **kernels):
        yield


def _expected(
    args: tuple[torch.Tensor, ...], chunk_size: int
) -> tuple[torch.Tensor, ...]:
    q, k, v, a, b, a_log, bias, initial = args
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.size(2), v.size(3)
    copies = value_heads // heads
    chunks = (tokens + chunk_size - 1) // chunk_size
    pad = chunks * chunk_size - tokens
    rows = batch * value_heads * chunks
    q_hf = q.transpose(1, 2).repeat_interleave(copies, dim=1)
    k_hf = k.transpose(1, 2).repeat_interleave(copies, dim=1)
    v_hf = v.transpose(1, 2).contiguous()
    log_decay = -a_log.exp()[None, None, :] * F.softplus(a.float() + bias)
    beta = b.float().sigmoid()
    return (
        F.pad(q_hf, (0, 0, 0, pad)).reshape(rows, chunk_size, key_dim),
        F.pad(k_hf, (0, 0, 0, pad)).reshape(rows, chunk_size, key_dim),
        F.pad(v_hf, (0, 0, 0, pad)).reshape(rows, chunk_size, value_dim),
        F.pad(beta.transpose(1, 2).contiguous(), (0, pad)).reshape(rows, chunk_size),
        F.pad(log_decay.transpose(1, 2).contiguous(), (0, pad))
        .reshape(batch * value_heads, chunks, chunk_size)
        .cumsum(-1),
        initial.transpose(-2, -1)
        .contiguous()
        .reshape(batch * value_heads, key_dim, value_dim)
        .bfloat16(),
    )


@pytest.mark.parametrize(
    ("batch", "tokens", "heads", "value_heads", "view_v"),
    [
        (1, 32, 2, 6, False),
        (2, 19, 2, 4, False),
        (1, 16, 2, 4, True),
        (2, 32, 1, 1, True),
        (2, 16, 2, 4, False),
    ],
)
def test_native_layout_grouping_tails_and_state_rounding(
    reference_kernels: None,
    batch: int,
    tokens: int,
    heads: int,
    value_heads: int,
    view_v: bool,
) -> None:
    args = _inputs(batch, tokens, heads, value_heads)
    originals = tuple(value.clone() for value in args)
    actual = preparation.prepare_gated_delta_prefill(*args, chunk_size=16)
    expected = _expected(args, 16)
    for index, (observed, reference) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(
            observed,
            reference,
            atol=4e-6 if index == 4 else 1e-7 if index == 3 else 0,
            rtol=2e-7 if index == 4 else 0,
        )
    assert (
        actual.v.untyped_storage().data_ptr() == args[2].untyped_storage().data_ptr()
    ) == view_v
    assert actual.initial_state.dtype == torch.bfloat16
    assert actual.beta.dtype == actual.cumulative_log_decay.dtype == torch.float32
    assert (
        actual.decay_last.untyped_storage().data_ptr()
        == actual.cumulative_log_decay.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(actual.decay_last, actual.cumulative_log_decay[:, :, -1])
    for original, value in zip(originals, args, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    if tokens % 16:
        tail = tokens % 16
        cumulative = actual.cumulative_log_decay[:, -1]
        torch.testing.assert_close(
            cumulative[:, tail:],
            cumulative[:, tail - 1 : tail].expand(-1, 16 - tail),
            atol=0,
            rtol=0,
        )


def test_unbounded_softplus_extremes_preserve_fp32_gate_order(
    reference_kernels: None,
) -> None:
    args = list(_inputs(1, 16, 1, 2))
    values = torch.tensor([-1000.0, -100.0, -21.0, -1.0, 0.0, 1.0, 20.0, 21.0, 1000.0])
    args[3] = values.repeat(4)[:32].view(1, 16, 2).bfloat16()
    args[4] = -args[3]
    actual = preparation.prepare_gated_delta_prefill(*args, chunk_size=16)
    expected = _expected(tuple(args), 16)
    assert torch.isfinite(actual.cumulative_log_decay).all()
    torch.testing.assert_close(
        actual.cumulative_log_decay, expected[4], atol=1e-5, rtol=2e-7
    )
    torch.testing.assert_close(actual.beta, expected[3], atol=1e-7, rtol=0)
    assert actual.cumulative_log_decay.min() < -1000


@pytest.mark.parametrize("tokens", [128, 2048])
def test_trace_shapes_fake_allocation_and_stage_views(tokens: int) -> None:
    calls: list[tuple[object, ...]] = []

    def capture(*args: object) -> None:
        calls.append(args)

    with (
        FakeTensorMode(),
        patch.multiple(
            preparation,
            pack_gated_delta_inputs_helion=capture,
            prepare_gated_delta_gates_helion=capture,
        ),
    ):
        args = _inputs(1, tokens, 16, 32, 128, 128)
        prepared = preparation.prepare_gated_delta_prefill(*args)
        chunks = tokens // 64
        assert (
            prepared.q.shape
            == prepared.k.shape
            == prepared.v.shape
            == (32 * chunks, 64, 128)
        )
        assert prepared.beta.shape == (32 * chunks, 64)
        assert prepared.cumulative_log_decay.shape == (32, chunks, 64)
        assert prepared.initial_state.shape == (32, 128, 128)
        assert (
            prepared.k.view(32, chunks, 64, 128).untyped_storage()
            is prepared.k.untyped_storage()
        )
        assert (
            prepared.cumulative_log_decay.view(32 * chunks, 64).untyped_storage()
            is prepared.cumulative_log_decay.untyped_storage()
        )
        assert len(calls) == 2 and calls[0][-1] is True


@pytest.mark.parametrize("tokens", [19, 64, 128, 2048])
def test_preparation_cpu_codegen(tokens: int) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def pack(*args: object) -> None:
        calls.append(("pack_gated_delta_inputs_helion", args))

    def gates(*args: object) -> None:
        calls.append(("prepare_gated_delta_gates_helion", args))

    args = _inputs(1, tokens, 16, 32, 128, 128)
    with patch.multiple(
        preparation,
        pack_gated_delta_inputs_helion=pack,
        prepare_gated_delta_gates_helion=gates,
    ):
        preparation.prepare_gated_delta_prefill(*args)
    with _cpu_codegen():
        for name, kernel_args in calls:
            original = getattr(preparation, name)
            kernel = helion.kernel(
                original.fn, backend="cute", static_shapes=True, autotune_effort="none"
            )
            bound = kernel._bind_isolated(kernel_args)
            source = bound.to_code(bound.config_spec.default_config())
            assert "@cute.kernel" in source
            assert "cute.gemm" not in source
            if name == "prepare_gated_delta_gates_helion":
                assert "prefix" in source and "range(" in source
                assert "Float32" in source
            else:
                assert "BFloat16" in source


def test_gva_and_state_contract_errors() -> None:
    args = list(_inputs(1, 16, 2, 3))
    with pytest.raises(ValueError, match="Q/K/V"):
        preparation.prepare_gated_delta_prefill(*args, chunk_size=16)
    args = list(_inputs(1, 16, 2, 4))
    args[-1] = args[-1].bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        preparation.prepare_gated_delta_prefill(*args, chunk_size=16)
