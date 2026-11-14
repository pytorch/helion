from __future__ import annotations

import math

from benchmarks.cute.sdg_attention_kernel import headwise
from benchmarks.cute.sdg_attention_kernel import recurrence
from benchmarks.cute.sdg_attention_kernel import rowwise
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA


def _inputs():
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    q = torch.randn(
        2, 4, 16, 256, generator=generator, device=DEVICE, dtype=torch.bfloat16
    )[..., ::2]
    caches = []
    for _ in range(2):
        storage = torch.randint(
            -8,
            9,
            (5, 128, 2, 132),
            generator=generator,
            device=DEVICE,
            dtype=torch.int8,
        )
        data = storage[..., :128]
        scales = storage[..., 128:].view(torch.uint8)
        scales.copy_(
            torch.randint(
                123,
                128,
                scales.shape,
                generator=generator,
                device=DEVICE,
                dtype=torch.uint8,
            )
        )
        caches.append((data, scales))
    (k, ks), (v, vs) = caches
    pages = torch.tensor([[2, 0, 4], [1, 3, 4]], device=DEVICE, dtype=torch.int32)
    lengths = torch.tensor([129, 253], device=DEVICE, dtype=torch.int32)
    out = torch.full_like(q, float("nan"), memory_format=torch.preserve_format)
    return q, k, v, ks, vs, pages, lengths, out


def _dense_reference(q, k, v, ks, vs, pages, lengths, window):
    # Independent dense attention: gather pages in logical order, decode each
    # 32-channel scale group, then form one masked softmax over all tokens.
    decoded = [
        (data.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, -1))
        .to(q.dtype)
        .float()
        for data, scale in ((k, ks), (v, vs))
    ]
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    for batch in range(q.shape[0]):
        length = int(lengths[batch])
        key, value = [
            data[pages[batch].long()]
            .flatten(0, 1)[:length]
            .repeat_interleave(q.shape[2] // data.shape[2], 1)
            for data in decoded
        ]
        scores = torch.einsum("qhd,khd->hqk", q[batch].float(), key)
        scores *= 1 / math.sqrt(q.shape[-1])
        key_position = torch.arange(length, device=q.device)
        query_position = length - q.shape[1] + torch.arange(q.shape[1], device=q.device)
        visible = key_position[None, :] <= query_position[:, None]
        if window is not None:
            visible &= key_position[None, :] >= query_position[:, None] - window
        scores.masked_fill_(~visible[None, :, :], -float("inf"))
        output[batch] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value)
        lse[batch] = scores.logsumexp(-1).T
    return output, lse


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "family", [recurrence, rowwise, headwise], ids=["recurrence", "rowwise", "headwise"]
)
@pytest.mark.parametrize("window,return_lse", [(None, True), (31, False)])
def test_authored_sdg_program_matches_dense_attention(family, window, return_lse):
    q, k, v, ks, vs, pages, lengths, out = values = _inputs()
    args = (*values[:7], window, 128, return_lse, out)
    kernel = helion.kernel(
        family.paged_query_first_direct_attention.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
    )
    bound = kernel.bind(args)
    config = bound.config_spec.default_config()
    if family is rowwise:
        config.config["rescale_threshold"] = 2
    compiled = bound.compile_config(config)
    # A single compilation also checks that data and page metadata remain live.
    for changed in (False, True):
        if changed:
            q.neg_()
            pages.copy_(pages.flip(1))
            lengths.copy_(lengths.new_tensor([257, 131]))
        expected, expected_lse = _dense_reference(*values[:7], window)
        out.fill_(float("nan"))
        actual, actual_lse = compiled(*args)
        assert actual is out
        # The authored programs round the PV probabilities to BF16; the dense
        # reference keeps them in FP32. This permits that specified rounding.
        torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)
        if return_lse:
            torch.testing.assert_close(actual_lse, expected_lse, rtol=0.002, atol=0.002)
        else:
            assert actual_lse is None
