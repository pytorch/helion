"""GPU tests: ``cute_grid_lane_jam`` leaves every result bitwise unchanged.

The jam (``cute/jam_grid_lanes.py``) only reorders the work of independent
lanes, loads the lanes' consecutive elements as one vector and reduces the
lanes' values in one fragment reduce of the scalar reduces' combine order, so
each kernel matches its knob-off twin bit for bit at the same config.  Adds
into a privatized atomic target, whose order is unspecified, are summed per
cell; exact integer sums keep those bitwise too.  The codegen side is
``test_cute_jam_grid_lanes_codegen.py``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cute_grid_lane_jam_kernels import HEAD_ROWS_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import ROW_NORM_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import SCALAR_ROWS_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import SCATTER_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import bias_softmax
from helion._testing.cute_grid_lane_jam_kernels import bucket_grad_scatter
from helion._testing.cute_grid_lane_jam_kernels import doubled_rows_plus_one
from helion._testing.cute_grid_lane_jam_kernels import head_bias
from helion._testing.cute_grid_lane_jam_kernels import head_bias_softmax_bwd
from helion._testing.cute_grid_lane_jam_kernels import head_quotients
from helion._testing.cute_grid_lane_jam_kernels import heads_config
from helion._testing.cute_grid_lane_jam_kernels import heads_innermost_bias
from helion._testing.cute_grid_lane_jam_kernels import invariant_rows_over_heads
from helion._testing.cute_grid_lane_jam_kernels import kept_row_norm
from helion._testing.cute_grid_lane_jam_kernels import row_norm
from helion._testing.cute_grid_lane_jam_kernels import scale_rows_in_place


def _run(keys: int, jam: bool, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    kernel = helion.kernel(
        bias_softmax,
        backend="cute",
        static_shapes=True,
        config=helion.Config(**heads_config(2, keys), cute_grid_lane_jam=jam),
    )
    return kernel(*args)


@pytest.mark.parametrize("keys", [256, 512])
def test_jam_is_bitwise_the_rolled_head_loop(keys: int) -> None:
    # 256 keys: warp reduces; 512: two-warp shared-memory reduces, batched
    # into fragment reduces under the knob.
    torch.manual_seed(0)
    scores = torch.randn(2, 8, 4, keys, device=DEVICE, dtype=torch.bfloat16) * 4
    bias = heads_innermost_bias(2, 8, 4, keys, DEVICE)
    expected = _run(keys, False, (scores, bias))
    actual = _run(keys, True, (scores, bias))
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    reference = torch.softmax(scores.float() + bias.float(), dim=-1)
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=1e-3)


def test_jammed_quotients_are_bitwise_the_rolled_head_loop() -> None:
    # Zeros and infinities among the inputs.
    torch.manual_seed(0)
    scores = torch.randn(2, 8, 4, 512, device=DEVICE, dtype=torch.bfloat16)
    bias = heads_innermost_bias(2, 8, 4, 512, DEVICE)
    scores.view(-1)[::97] = 0
    # The bias is a heads-innermost view, so it is marked by position.
    positions = torch.arange(bias.numel(), device=DEVICE).view(bias.shape)
    bias.masked_fill_(positions % 89 == 0, float("inf"))
    results = [
        helion.kernel(
            head_quotients,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**heads_config(1, 512), cute_grid_lane_jam=jam),
        )(scores, bias)
        for jam in (False, True)
    ]
    assert torch.equal(results[1].view(torch.int16), results[0].view(torch.int16))
    x, y = scores.float(), bias.float()
    torch.testing.assert_close(
        results[1].float(), (x / y + y / (x + 1.0)).bfloat16().float(), rtol=0, atol=0
    )


def test_a_head_invariant_store_is_bitwise_the_rolled_head_loop() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 256, device=DEVICE, dtype=torch.bfloat16)
    results = []
    for jam in (False, True):
        out = torch.empty(2, 8, 4, 256, device=DEVICE, dtype=torch.bfloat16)
        helion.kernel(
            invariant_rows_over_heads,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**heads_config(1), cute_grid_lane_jam=jam),
        )(x, out)
        results.append(out)
    assert torch.equal(results[1].view(torch.int16), results[0].view(torch.int16))
    expected = (x * 2)[:, None].expand(2, 8, 4, 256)
    torch.testing.assert_close(results[1], expected, rtol=0, atol=0)


def test_forwarded_head_rows_are_bitwise_the_rolled_head_loop() -> None:
    # Register reloads and a forwarded workspace: each lane's copy of the
    # caches holds what its own store wrote, read back through the move.
    torch.manual_seed(0)
    scores = torch.randn(16, 4, 512, device=DEVICE, dtype=torch.bfloat16) * 4
    grad = torch.randn(16, 4, 512, device=DEVICE, dtype=torch.bfloat16)
    bias = head_bias(8, 4, 512, DEVICE)
    logits = scores.float() + bias.repeat(2, 1, 1)
    row_max = logits.amax(-1)
    row_sum = torch.exp(logits - row_max[..., None]).sum(-1)
    args = (grad, scores, bias, row_max, row_sum)
    kernels = [
        helion.kernel(
            head_bias_softmax_bwd,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                **{
                    **HEAD_ROWS_CONFIG,
                    "cute_reduction_reloads": ["register"],
                    "cute_scratch_forwarding": True,
                    "cute_grid_lane_jam": jam,
                }
            ),
        )
        for jam in (False, True)
    ]
    rolled, jammed = (kernel.bind(args).to_code() for kernel in kernels)
    assert jammed != rolled
    assert "mov.b32" in jammed
    assert not [
        line
        for line in jammed.splitlines()
        if "probs_ws.iterator" in line and ".load()" in line
    ]
    results = [kernel(*args) for kernel in kernels]
    assert torch.equal(results[1].view(torch.int16), results[0].view(torch.int16))
    probs = torch.exp(logits - row_max[..., None]) / row_sum[..., None]
    total = (grad.float() * probs).sum(-1, keepdim=True)
    reference = probs * (grad.float() - total)
    torch.testing.assert_close(results[1].float(), reference, rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("kept", [False, True])
def test_jammed_rows_are_bitwise_the_rolled_row_loop(kept: bool) -> None:
    # ``kept``: the rows also load a boolean mask per element.
    torch.manual_seed(0)
    x = torch.randn(64, 2048, device=DEVICE, dtype=torch.bfloat16)
    keep = torch.rand(64, 2048, device=DEVICE) > 0.25
    fn, args = (kept_row_norm, (x, keep)) if kept else (row_norm, (x,))
    results = [
        helion.kernel(
            fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**ROW_NORM_CONFIG, cute_grid_lane_jam=jam),
        )(*args)
        for jam in (False, True)
    ]
    assert torch.equal(results[1].view(torch.int16), results[0].view(torch.int16))
    v = x.float() * keep.float() if kept else x.float()
    reference = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(results[1].float(), reference, rtol=2e-2, atol=2e-2)


def test_rows_updated_in_place_are_bitwise_the_rolled_head_loop() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4, 256, device=DEVICE, dtype=torch.bfloat16)
    results = [
        helion.kernel(
            scale_rows_in_place,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**heads_config(1), cute_grid_lane_jam=jam),
        )(x.clone())
        for jam in (False, True)
    ]
    assert torch.equal(results[1].view(torch.int16), results[0].view(torch.int16))
    torch.testing.assert_close(results[1], x * 2, rtol=0, atol=0)


def _head_rows_inputs(
    keys: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """``head_bias_softmax_bwd``'s arguments over 16 rows of four queries
    and ``keys`` keys, and the eager result."""
    torch.manual_seed(0)
    scores = torch.randn(16, 4, keys, device=DEVICE, dtype=torch.bfloat16) * 4
    grad = torch.randn(16, 4, keys, device=DEVICE, dtype=torch.bfloat16)
    bias = head_bias(8, 4, keys, DEVICE)
    logits = scores.float() + bias.repeat(2, 1, 1)
    row_max = logits.amax(-1)
    row_sum = torch.exp(logits - row_max[..., None]).sum(-1)
    probs = torch.exp(logits - row_max[..., None]) / row_sum[..., None]
    total = (grad.float() * probs).sum(-1, keepdim=True)
    reference = probs * (grad.float() - total)
    return (grad, scores, bias, row_max, row_sum), reference


def _assert_head_rows_jam_is_bitwise(
    config: dict[str, object], args: tuple[torch.Tensor, ...], reference: torch.Tensor
) -> None:
    """``config`` with and without ``cute_grid_lane_jam`` stores the same
    bits, and matches eager."""
    results = [
        helion.kernel(
            head_bias_softmax_bwd,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**{**config, "cute_grid_lane_jam": jam}),
        )(*args)
        for jam in (False, True)
    ]
    assert torch.equal(results[1].view(torch.int16), results[0].view(torch.int16))
    torch.testing.assert_close(results[1].float(), reference, rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"block_sizes": [8, 1]},
        {"cute_reduction_reloads": ["register"]},
        {"cute_reduction_reloads": ["register"], "cute_vector_widths": [4, 1, 1]},
    ],
    ids=["four_heads", "eight_heads", "register_reloads", "four_keys_per_thread"],
)
def test_head_rows_are_bitwise_the_rolled_head_loop(knobs: dict[str, object]) -> None:
    # Eight heads: two packets of four fp32 bias elements per key; register
    # reloads: a register cache per lane; four keys per thread: the jam
    # seed's alternate, each lane's keys loaded in packets.
    args, reference = _head_rows_inputs(512)
    _assert_head_rows_jam_is_bitwise({**HEAD_ROWS_CONFIG, **knobs}, args, reference)


@pytest.mark.parametrize("keys_per_thread", [4, 8])
def test_the_wide_seeds_are_bitwise_the_rolled_head_loop(keys_per_thread: int) -> None:
    # The jam seed's alternates as the heuristic plants them for T5's
    # 1024-key rows: 128 threads of four keys (two packets each in the
    # chunk) or eight fill a 1024-key chunk.  Before sm_100 a lane loads at
    # most 16 bytes whole, and the eight-key seed is not planted.
    if keys_per_thread == 8 and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("the eight-key seed needs 32-byte packets (sm_100)")
    args, reference = _head_rows_inputs(1024)
    bound = helion.kernel(
        head_bias_softmax_bwd, backend="cute", static_shapes=True
    ).bind(args)
    (seed,) = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_grid_lane_jam")
        and seed.config.get("cute_vector_widths") == [keys_per_thread, 1, 1]
        and not seed.config.get("cute_scratch_forwarding")
    ]
    assert seed["num_threads"] == [1, 1, 128]
    assert seed["reduction_loops"] == [1024]
    _assert_head_rows_jam_is_bitwise(seed, args, reference)


def test_rows_read_back_after_an_in_place_store_match_eager() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 512, device=DEVICE, dtype=torch.bfloat16)
    results = []
    for jam in (False, True):
        updated = x.clone()
        out = helion.kernel(
            doubled_rows_plus_one,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**SCALAR_ROWS_CONFIG, cute_grid_lane_jam=jam),
        )(updated)
        results.append((updated, out))
    (off_x, off_out), (on_x, on_out) = results
    assert torch.equal(on_x.view(torch.int16), off_x.view(torch.int16))
    assert torch.equal(on_out.view(torch.int16), off_out.view(torch.int16))
    torch.testing.assert_close(on_out, x * 2 + 1, rtol=0, atol=0)


def test_jammed_privatized_adds_match_eager() -> None:
    # Integer gradients sum exactly in any order, so combining the lanes'
    # adds per cell (a reordering of the copy's adds) must match the rolled
    # loop and eager bit for bit.  Four buckets give each thread's eight
    # lanes cells in common.
    torch.manual_seed(0)
    grads = torch.randint(-4, 5, (16, 2, 8, 256), device=DEVICE).to(torch.bfloat16)
    buckets = torch.randint(0, 4, (8, 256), device=DEVICE) * 7
    results = []
    for jam in (False, True):
        out = torch.zeros(32, 2, device=DEVICE)
        helion.kernel(
            bucket_grad_scatter,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**SCATTER_CONFIG, cute_grid_lane_jam=jam),
        )(grads, buckets, out)
        results.append(out)
    total = grads.double().sum(0)
    expected = torch.zeros(32, 2, device=DEVICE, dtype=torch.float64)
    for head in range(2):
        expected[:, head].index_add_(0, buckets.flatten(), total[head].flatten())
    assert torch.equal(results[0], expected.float())
    assert torch.equal(results[1], expected.float())
