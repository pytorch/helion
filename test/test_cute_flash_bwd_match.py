"""Scale forms accepted by the fused flash-attention backward matcher.

``examples/attention.py`` pre-scales K by ``sm_scale * log2(e)`` (so the
softmax recompute has no exp2-domain multiply), rescales dQ by the constant
``ln(2)`` and dK by ``sm_scale``. The matcher used to require one host scalar
for both dK and dQ, so that kernel fell back to the generic SIMT lowering
(seconds per call). The store roles and dot operands it accepts are checked
on CPU binds in test_cute_flash_bwd_match_variants.py.
"""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfRefEager

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._testing import import_example

attention_backward = import_example("attention").attention_backward

RCP_LN2 = 1.4426950408889634


def _inputs(
    shape: tuple[int, int, int, int],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """(kernel args with K pre-scaled, float64 reference dq/dk/dv)."""
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(4)
    )
    sm_scale = 1.0 / math.sqrt(shape[-1])
    q64, k64, v64 = (t.double().requires_grad_() for t in (q, k, v))
    scores = q64 @ k64.transpose(-1, -2) * sm_scale
    out = scores.softmax(-1) @ v64
    out.backward(do.double())
    o = out.detach().to(q.dtype)
    lse = (torch.logsumexp(scores.detach(), -1) * RCP_LN2).float()
    delta = (o.float() * do.float()).sum(-1)
    args = (q, k * (sm_scale * RCP_LN2), v, o, lse, do, delta, sm_scale)
    grads = tuple(t.grad for t in (q64, k64, v64))
    return args, grads  # pyrefly: ignore [bad-return]


@skipIfRefEager("checks the generated CuTe kernel kind")
@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 256, 128)])
def test_example_backward_takes_the_flash_path(shape: tuple[int, ...]) -> None:
    kernel = helion.kernel(
        attention_backward.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )
    args, expected = _inputs(shape)  # pyrefly: ignore [bad-argument-type]
    code, grads = code_and_output(kernel, args, block_sizes=[128, 128])
    assert "'kind': 'helion_flash_bwd'" in code
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )


@skipIfRefEager("checks the generated CuTe kernel kind")
@pytest.mark.parametrize(
    ("head_dim", "knobs", "loop_heads"),
    [
        (64, {"cute_flash_bwd_persistent": 1}, 7),
        (128, {"cute_flash_bwd_persistent": 1}, 7),
        (128, {"cute_flash_bwd_two_cta": 1}, 0),
        (128, {"cute_flash_bwd_two_cta": 1, "cute_flash_bwd_persistent": 1}, 9),
    ],
    ids=[
        "persistent_d64",
        "persistent_d128",
        "two_cta_d128",
        "two_cta_persistent_d128",
    ],
)
def test_persistent_and_two_cta_bodies_match_reference(
    head_dim: int, knobs: dict[str, int], loop_heads: int
) -> None:
    """The dense 1-CTA / 2-CTA bodies under the CLC persistent walk (each
    warp role's per-tile region inside a ``while fbwd_work.is_valid_tile``
    loop) and the 2-CTA cluster body: the small shape against the float64
    reference, then a 160-tile problem (more tiles than SMs, so CTAs steal
    second tiles) against the one-tile-per-CTA kernel on the same inputs."""
    kernel = helion.kernel(
        attention_backward.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )
    args, expected = _inputs((1, 2, 256, head_dim))  # pyrefly: ignore [bad-argument-type]
    code, grads = code_and_output(kernel, args, block_sizes=[128, 128], **knobs)
    assert "'kind': 'helion_flash_bwd'" in code
    assert code.count("fbwd_work = fbwd_clc_hw.initial_work_tile_info()") == loop_heads
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )
    # 160 KV tiles: the forward statistics computed in fp32 on the device
    # (random LSE / delta would overflow the exp2 recompute to NaN).
    torch.manual_seed(1)
    long_shape = (1, 2, 128 * 80, head_dim)
    q, k, v, do = (
        torch.randn(long_shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(4)
    )
    sm_scale = 1.0 / math.sqrt(head_dim)
    scores = (q.float() @ k.float().transpose(-1, -2)) * sm_scale
    lse = (torch.logsumexp(scores, -1) * RCP_LN2).float()
    o = (scores.softmax(-1) @ v.float()).to(torch.bfloat16)
    del scores
    delta = (o.float() * do.float()).sum(-1)
    long_args = (q, k * (sm_scale * RCP_LN2), v, o, lse, do, delta, sm_scale)
    plain = helion.kernel(
        attention_backward.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )
    _, reference = code_and_output(plain, long_args, block_sizes=[128, 128])
    _, walked = code_and_output(kernel, long_args, block_sizes=[128, 128], **knobs)
    # dQ is reduce-added in tile order (fp32, order-sensitive at the ulp
    # level). The persistent walk runs the one-tile body per tile, so its dK /
    # dV are bitwise; the 2-CTA body accumulates a 256-row tile's products in
    # the cluster MMA's order and is compared within bf16 rounding.
    torch.testing.assert_close(walked[0], reference[0], atol=1e-3, rtol=1e-3)
    tol = (
        {"atol": 2e-2, "rtol": 2e-2}
        if "cute_flash_bwd_two_cta" in knobs
        else {"atol": 0, "rtol": 0}
    )
    for actual, ref in zip(walked[1:], reference[1:], strict=True):
        torch.testing.assert_close(actual, ref, **tol)


@skipIfRefEager("checks the generated CuTe kernel kind")
def test_example_decorator_config_normalizes_onto_the_pin() -> None:
    """``examples/attention.py`` pins a Triton-shaped ``block_sizes=[64, 64]``
    config on ``attention_backward``. The CuTe fused backward surface pins both
    block sizes to 128 (``BlockSizeSpec.pin_structurally``), so the decorator
    config, honoured without autotuning, normalizes onto the pin and emits the
    fused kernel instead of the scalar SIMT lowering (117 ms vs 0.03 ms)."""
    kernel = helion.kernel(
        attention_backward.fn,
        config=attention_backward.configs[0],
        backend="cute",
        static_shapes=True,
        autotune_accuracy_check=False,
        force_autotune=False,
    )
    assert kernel.configs[0].block_sizes == [64, 64]
    args, expected = _inputs((1, 2, 256, 64))  # pyrefly: ignore [bad-argument-type]
    bound = kernel.bind(args)
    config = bound._implicit_config()
    assert config is not None and config.block_sizes == [64, 64]
    normalized = bound._normalized_config_copy(config)
    assert normalized.block_sizes == [128, 128]
    code = bound.to_code(config)
    assert "'kind': 'helion_flash_bwd'" in code
    grads = kernel(*args)
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )
