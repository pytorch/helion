from __future__ import annotations

import pytest
import torch

from test.test_cute_batched_aux_tma import _config
from test.test_cute_mapped_aux_codegen import _inputs
from test.test_cute_mapped_aux_codegen import _mapped_batched_aux

TensorInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]


def _random_inputs(dtype: torch.dtype, group_size: int, device: str) -> TensorInputs:
    templates = _inputs(dtype, group_size)

    def random_tensor(template: object) -> torch.Tensor:
        assert isinstance(template, torch.Tensor)
        return torch.randn(template.shape, dtype=template.dtype, device=device) * 0.05

    return (
        random_tensor(templates[0]),
        random_tensor(templates[1]),
        random_tensor(templates[2]),
        random_tensor(templates[3]),
        random_tensor(templates[4]),
        random_tensor(templates[5]),
        group_size,
    )


def _reference_parts(
    args: TensorInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lhs, rhs, x, d, weights, output, group_size = args
    batches, rows, columns = output.shape
    groups = columns // group_size
    decay = weights.view(batches, groups, rows).transpose(1, 2)
    scale = (decay.double() - decay[:, -1:, :].double() * 0.5).exp().to(output.dtype)
    scale = scale.repeat_interleave(group_size, dim=2)
    bias = (x.float() * d.repeat_interleave(group_size).float()).to(output.dtype)
    return torch.bmm(lhs.double(), rhs.double()), scale, bias


def _mapping_inputs(dtype: torch.dtype, group_size: int, device: str) -> TensorInputs:
    """Exactly unit accumulator and distinguishable work, row and head scales."""
    batches, rows, columns, inner = 3, 128, 768, 64
    groups = columns // group_size
    lhs = torch.zeros((batches, rows, inner), dtype=dtype, device=device)
    rhs = torch.zeros((batches, inner, columns), dtype=dtype, device=device)
    lhs[:, :, 0] = 1
    rhs[:, 0, :] = 1
    x = torch.full((batches, rows, columns), 0.25, dtype=dtype, device=device)
    head = torch.arange(groups, device=device)
    d = ((head % 4).float() * 0.25 - 0.5).to(dtype)
    # Binary fractions keep input rounding out of the index-ownership test.
    decay = (
        torch.arange(batches, device=device)[:, None, None].float() * 0.5
        + (head % 8)[None, :, None].float() * 0.125
        + (torch.arange(rows, device=device) % 4)[None, None, :].float() * 0.25
    )
    out = torch.full_like(x, float("nan"))
    return lhs, rhs, x, d, decay.contiguous().view(-1), out, group_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires TCgen05 GPU")
@pytest.mark.parametrize("group_size", [3, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mapped_aux_gpu_five_seeds(dtype: torch.dtype, group_size: int) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires TCgen05 GPU")
    fn = None
    for seed in range(5):
        torch.manual_seed(seed)
        args = _random_inputs(dtype, group_size, "cuda")
        lhs, rhs, x, d, weights, output, _ = args
        output.fill_(float("nan"))
        before = [value.clone() for value in args[:5]]
        acc, scale, bias = _reference_parts(args)
        expected = acc * scale.double() + bias.double()
        if fn is None:
            bound = _mapped_batched_aux._bind_isolated(tuple(args))
            fn = bound.compile_config(
                _config(
                    tcgen05_consumer_regs=128, tcgen05_aux_role_local_scheduler=True
                )
            )
        actual = fn(*args)
        assert actual is output and actual.dtype == dtype
        torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.02)
        saved = actual.clone()
        output.fill_(float("nan"))
        fn(*args)
        torch.testing.assert_close(output, saved, atol=0, rtol=0)
        for value, original in zip(args[:5], before, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
        next_args = (*args[:5], torch.empty_like(output), group_size)
        fn(*next_args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*next_args)
        next_args[5].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(next_args[5], saved, atol=0, rtol=0)
        torch.testing.assert_close(output, saved, atol=0, rtol=0)
        for value, original in zip(args[:5], before, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)


@pytest.mark.parametrize("group_size", [3, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mutation",
    ["omitted_scale", "wrong_work", "wrong_row", "wrong_head", "row_as_last"],
)
def test_mapping_fixture_rejects_index_mutations(
    dtype: torch.dtype, group_size: int, mutation: str
) -> None:
    args = _mapping_inputs(dtype, group_size, "cpu")
    acc, scale, bias = _reference_parts(args)
    torch.testing.assert_close(acc, torch.ones_like(acc), atol=0, rtol=0)
    expected = acc * scale.double() + bias.double()
    if mutation == "omitted_scale":
        wrong_scale = torch.ones_like(scale)
    elif mutation == "wrong_work":
        wrong_scale = scale.roll(1, dims=0)
    elif mutation == "wrong_row":
        wrong_scale = scale.roll(1, dims=1)
    elif mutation == "wrong_head":
        wrong_scale = scale.roll(group_size, dims=2)
    else:
        weights = args[4].view(3, 768 // group_size, 128).transpose(1, 2)
        wrong_scale = (weights.double() * 0.5).exp().to(dtype)
        wrong_scale = wrong_scale.repeat_interleave(group_size, dim=2)
    wrong = (acc * wrong_scale.double() + bias.double()).to(dtype)
    # Keep exactly the runtime tolerance; this proves fixture sensitivity.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong.double(), expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires TCgen05 GPU")
@pytest.mark.parametrize("group_size", [3, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mapped_aux_gpu_mapping_sensitive(dtype: torch.dtype, group_size: int) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires TCgen05 GPU")
    args = _mapping_inputs(dtype, group_size, "cuda")
    before = [value.clone() for value in args[:5]]
    acc, scale, bias = _reference_parts(args)
    expected = acc * scale.double() + bias.double()
    bound = _mapped_batched_aux._bind_isolated(args)
    fn = bound.compile_config(
        _config(tcgen05_consumer_regs=128, tcgen05_aux_role_local_scheduler=True)
    )
    actual = fn(*args)
    assert actual is args[5] and actual.dtype == dtype
    torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.02)
    for value, original in zip(args[:5], before, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
