"""Native replay checks for full-slice consumer axis liveness."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cute_phase_liveness import explicit_barrier_row_topk
from helion._testing.cute_phase_liveness import explicit_direct_selection
from helion._testing.cute_phase_liveness import grouped_selection_after_producer
from helion._testing.cute_phase_liveness import routed_after_producer
from helion._testing.cute_phase_liveness import routed_phase_config
from helion._testing.cute_phase_liveness import slice_sum_phases


@pytest.mark.parametrize("tensor_indices", [False, True])
def test_projected_static_memory_consumer_matches_reference(tensor_indices):
    x = (torch.arange(2 * 3 * 7 * 8, device=DEVICE).reshape(2, 3, 7, 8) % 7 - 3).to(
        torch.bfloat16
    )
    y = (torch.arange(2 * 3 * 8 * 11, device=DEVICE).reshape(2, 3, 8, 11) % 5 - 2).to(
        torch.bfloat16
    )
    bound = slice_sum_phases.bind((x, y, tensor_indices))
    config = dict(bound.config_spec.default_config()) | {
        "pid_type": "flat",
    }
    compiled = bound.compile_config(helion.Config.from_dict(config))
    for multiplier in (1.0, -1.0):
        x.mul_(multiplier)
        value = x.float() @ y.float()
        expected = (
            value + value.sum(dim=2, keepdim=True) + value.sum(dim=3, keepdim=True)
        ).sum(dim=1)
        assert not torch.equal(expected[0], expected[1])
        assert not torch.equal(expected[:, 0], expected[:, 1])
        assert not torch.equal(expected[:, :, 0], expected[:, :, 1])
        torch.testing.assert_close(
            compiled(x, y, tensor_indices), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("pid_type", ["persistent_blocked", "persistent_interleaved"])
def test_warp_routing_consumes_a_preceding_persistent_phase(pid_type):
    rows = 307
    x = torch.arange(rows * 16, dtype=torch.int32, device=DEVICE).reshape(rows, 16)
    payload = torch.arange(16, dtype=x.dtype, device=DEVICE) * 7
    bound = routed_after_producer.bind((x, payload, 4))
    config = routed_phase_config(8, warp=True)
    config.config["pid_type"] = pid_type
    compiled = bound.compile_config(config)
    compiled(x, payload, 4)
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        result = compiled(x, payload, 4)
    for shift in (0, 3):
        x.copy_(x.roll(shift, dims=1))
        values, indices = (x * 3 - 1).topk(4, dim=-1)
        graph.replay()
        torch.testing.assert_close(result, values + payload[indices], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["direct", "negative_gather", "wrapped_gather"])
def test_grouped_selection_preserves_the_unpadded_index_domain(mode):
    rows, groups, width = 307, 2, 31
    # Strictly negative unique scores make a zero-padded 32nd lane observable.
    x = -torch.arange(
        1, 1 + rows * groups * width, dtype=torch.int32, device=DEVICE
    ).view(rows, -1)
    payload = torch.arange(groups * width, dtype=x.dtype, device=DEVICE).square()
    out = torch.empty((rows, groups, 4), dtype=x.dtype, device=DEVICE)
    bound = grouped_selection_after_producer.bind((x, payload, out, mode))
    fn = bound.compile_config(routed_phase_config(8, warp=False))
    offsets = width * torch.arange(groups, device=DEVICE)[None, :, None]
    for shift in (5, 14):
        x.copy_(x.roll(shift, dims=-1))
        values, indices = (x - 2).view(rows, groups, width).topk(4)
        expected = values + payload[indices + offsets]
        if mode == "negative_gather":
            # This computed-fragment backend guards negative tensor lanes in
            # both scores and payload. Only explicit normalization may wrap.
            expected = torch.zeros_like(expected)
        torch.testing.assert_close(
            fn(x, payload, out, mode),
            expected,
            rtol=0,
            atol=0,
        )


def test_explicit_barrier_shared_warp_selection() -> None:
    x = torch.arange(2 * 32, device=DEVICE).reshape(2, 32).roll(7, 1)
    config = dict(routed_phase_config(32, warp=True))
    config.pop("cross_loop_pipeline")
    fn = explicit_barrier_row_topk.bind((x,)).compile_config(
        helion.Config.from_dict(config)
    )
    previous = (x * 3 - 1).topk(4).values
    torch.testing.assert_close(fn(x), previous, rtol=0, atol=0)
    for shift in (3, 11):
        x.copy_(x.roll(shift, 1) + shift)
        expected = (x * 3 - 1).topk(4).values
        assert not torch.equal(expected, previous)
        torch.testing.assert_close(fn(x), expected, rtol=0, atol=0)
        previous = expected


@pytest.mark.parametrize(
    ("warp", "companion", "reduction_tile"),
    [
        (False, "direct", None),
        (True, "direct", None),
        (False, "reduce", None),
        (False, "reduce", 32),
        (True, "loop", None),
    ],
    ids=["False", "True", "native-reduction", "rolled-reduction", "nested-loop"],
)
def test_flat_topk_gathers_values_and_payload_on_changed_rows(
    warp, companion, reduction_tile
):
    x = torch.arange(7 * 62, device=DEVICE, dtype=torch.int32).reshape(7, 62)
    payload = torch.arange(62, device=DEVICE, dtype=torch.int32) * 7 - 13
    out = torch.empty((7, 2, 4), device=DEVICE, dtype=torch.int32)
    config = helion.Config(
        block_sizes=[32], pid_type="flat", cute_fragment_topk_network=warp
    )
    if companion == "reduce":
        config.config["reduction_loops"] = [reduction_tile]
    args = (x, payload, out, companion)
    fn = explicit_direct_selection.bind(args).compile_config(config)
    for change in (False, True):
        if change:
            x.copy_(-x.roll(3, 1))
        v, idx = (x.reshape(7, 2, 31) - 2).topk(4)
        expected = (
            v + payload[(idx + torch.tensor([0, 31], device=DEVICE).view(1, 2, 1))]
        )
        if companion == "reduce":
            expected = expected + payload[torch.arange(64, device=DEVICE) % 62].sum()
        elif companion == "loop":
            expected = expected + 3
        out.fill_(-55555)
        torch.testing.assert_close(fn(*args), expected, rtol=0, atol=0)
