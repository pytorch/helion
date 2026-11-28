"""Native replay checks for full-slice consumer axis liveness."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cute_phase_liveness import explicit_barrier_row_topk
from helion._testing.cute_phase_liveness import explicit_direct_selection
from helion._testing.cute_phase_liveness import grouped_selection_after_producer
from helion._testing.cute_phase_liveness import ordered_column_pool
from helion._testing.cute_phase_liveness import routed_after_producer
from helion._testing.cute_phase_liveness import routed_phase_config
from helion._testing.cute_phase_liveness import row_sum_after_producer
from helion._testing.cute_phase_liveness import row_sum_phase_config
from helion._testing.cute_phase_liveness import row_sum_two_outputs
from helion._testing.cute_phase_liveness import selected_rows_after_producer
from helion._testing.cute_phase_liveness import selected_rows_config
from helion._testing.cute_phase_liveness import slice_sum_phases
from helion._testing.cute_staged_indexed_rows import selected_projection_pair
from helion._testing.cute_staged_indexed_rows import selected_projection_with_epilogue


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


@pytest.mark.parametrize(
    "mode", ["direct", "negative_gather", "wrapped_gather", "explicit_network"]
)
def test_grouped_selection_preserves_the_unpadded_index_domain(mode):
    rows, groups, width = 307, 2, 31
    # Strictly negative unique scores make a zero-padded 32nd lane observable.
    x = -torch.arange(
        1, 1 + rows * groups * width, dtype=torch.int32, device=DEVICE
    ).view(rows, -1)
    payload = torch.arange(groups * width, dtype=x.dtype, device=DEVICE).square()
    out = torch.empty((rows, groups, 4), dtype=x.dtype, device=DEVICE)
    config = routed_phase_config(8, warp=False)
    if mode == "explicit_network":
        del config.config["cross_loop_pipeline"]
        config.config["cute_fragment_topk_network"] = True
        args = (x, payload, out, "direct")
        bound = explicit_direct_selection.bind(args)
    else:
        args = (x, payload, out, mode)
        bound = grouped_selection_after_producer.bind(args)
    fn = bound.compile_config(config)
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
            fn(*args),
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


@pytest.mark.parametrize("warp", [False, True])
def test_row_carry_reads_the_previous_phases_published_activation(warp: bool):
    x = (torch.arange(128, device=DEVICE) % 13 - 6).bfloat16()
    w = (torch.arange(17 * 128, device=DEVICE).reshape(17, 128) % 11 - 5).bfloat16()
    out = torch.empty(17, dtype=torch.float32, device=DEVICE)
    config = row_sum_phase_config()
    if warp:
        config = helion.Config.from_dict(
            dict(config)
            | {"cute_fragment_reduction": "warp", "block_sizes": [64, 32, 8]}
        )
    compiled = row_sum_after_producer.bind((x, w, out, False)).compile_config(config)
    for _ in range(2):
        torch.testing.assert_close(
            compiled(x, w, out, False), w.float() @ (x * 2).float(), rtol=0, atol=0
        )
        x.neg_()


def test_checked_row_carry_matches_signed_inputs_and_preserves_output_canaries() -> (
    None
):
    rows, k = 8, 16
    x = (torch.arange(k, device=DEVICE) % 7 - 3).float()
    w = (torch.arange(rows * k, device=DEVICE).reshape(rows, k) % 11 - 5).float()
    w[:, 0] = 1
    sentinels = [torch.full((rows + 2,), -313.0, device=DEVICE) for _ in range(2)]
    a, b = (storage[1:-1] for storage in sentinels)
    args = (x, w, a, b, True)
    bound = row_sum_two_outputs.bind(args)
    for enabled in (False, True):
        fn = bound.compile_config(
            helion.Config(block_sizes=[2, 4], cute_fragment_row_carry=enabled)
        )
        for _ in range(2):
            expected = w @ x
            fn(*args)
            torch.testing.assert_close(a, expected, rtol=0, atol=0)
            torch.testing.assert_close(b, 2 * expected, rtol=0, atol=0)
            for storage in sentinels:
                assert storage[0].item() == storage[-1].item() == -313
                storage.fill_(-313)
            x.neg_()


def test_warp_carry_packets_handle_sliced_weights_and_partial_rows():
    # Shape and strides match, but a bound kernel may specialize the input
    # pointer residue. Dispatch each view through the real binding key.
    rows, k = 17, 512
    x = (torch.arange(k, device=DEVICE) % 13 - 6).bfloat16()
    storage = (torch.arange(rows * k + 1, device=DEVICE) % 11 - 5).bfloat16()
    output = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    aligned = storage[:-1].view(rows, k)
    unaligned = storage[1:].view(rows, k)
    assert aligned.data_ptr() % 16 == 0 and unaligned.data_ptr() % 16 == 2
    config = helion.Config.from_dict(
        dict(row_sum_phase_config())
        | {
            "block_sizes": [256, 32, 8],
            "cute_fragment_reduction": "warp",
            "cute_fragment_vector_io": True,
        }
    )
    for weight in (aligned, unaligned):
        call = row_sum_after_producer.bind((x, weight, output, False)).compile_config(
            config
        )
        torch.testing.assert_close(
            call(x, weight, output, False),
            weight.float() @ (x * 2).float(),
            rtol=0,
            atol=0,
        )
        x.neg_()


@pytest.mark.parametrize(
    ("activated", "staged_grid", "active_joins"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_staged_bf16_projections_preserve_changed_router_and_rounded_pair(
    activated, staged_grid, active_joins
):
    if torch.cuda.get_device_capability(DEVICE) < (8, 0):
        pytest.skip("asynchronous copies and BF16 warp MMA require SM80+")
    rng = torch.Generator(device=DEVICE).manual_seed(78)

    def signed(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randint(0, 2, shape, device=DEVICE, generator=rng) * 2 - 1
        ).bfloat16()

    x = signed((2, 256))
    # The valid -1 lookup has real backing before the logical view as well.
    # A broken raw signed offset therefore reads a distinctive protected row.
    pair_base = signed((6, 64, 256))
    pair_base[0].fill_(9)
    pair = pair_base[1:]
    down_base = signed((6, 64, 32 if activated else 64))
    if activated:
        # Two copies of the identity expose every activation at the end of the
        # joined down phase; poison the prefix to reveal incorrect -1 indexing.
        down_base[1:].copy_(torch.eye(32, device=DEVICE).repeat(2, 1))
    down_base[0].fill_(-9)
    down = down_base[1:]
    scale = signed((256,))
    route = torch.tensor([[0, 2, 3], [4, 1, 0]], device=DEVICE, dtype=torch.int32)
    scores = torch.tensor([[0.25, -0.5, 1.0], [2.0, 0.5, -1.0]], device=DEVICE)
    if staged_grid:
        if torch.cuda.get_device_capability(DEVICE) != (10, 0):
            pytest.skip("two resident waves are validated on SM100")
        # Twelve producer tiles per copy; exercise two waves and a final
        # partial column wave on the selected device, including its joins.
        num_sm = helion.runtime.get_num_sm(torch.device(DEVICE))
        repeats = (2 * num_sm) // 12 + 2
        route = route.repeat(1, repeats)
        scores = scores.repeat(1, repeats)
    else:
        repeats = 1
    args = (x, pair, down, route, scores, scale, 3.0)
    config = helion.Config(
        block_sizes=[16, 256, 1, 1, 64, 32] if activated else [16, 256, 64, 64],
        num_threads=[4, 32, 1, 1, 4, 2, 1] if activated else [4, 32, 4, 2, 1],
        reduction_loops=[64],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_defer_tile_reduce=True,
        cute_staged_indexed_rows=True,
    )
    if staged_grid:
        config.config["cute_staged_projection_grid"] = 2
    if active_joins:
        config.config["cute_phase_participation"] = True
    if activated:
        config.config["cute_indexed_row_epilogue"] = True
    op = selected_projection_with_epilogue if activated else selected_projection_pair
    call = op.bind(args).compile_config(config)
    call(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mid, ids, weights, result = call(*args)
    for changed in (False, True):
        if changed:
            route.copy_(
                torch.tensor(
                    [[3, -1, 1], [2, 0, 4]], device=DEVICE, dtype=route.dtype
                ).repeat(1, repeats)
            )
            x.neg_()
            scores.neg_()
        graph.replay()
        # x consists of signed ones, so rsqrt(mean(x^2) + 3) is exactly 1/2.
        # Dyadic operands give exact FP32 sums in either reduction grouping.
        v = (x * 0.5 * scale).float()
        a = (pair[route.long()].float() * v[:, None, None, :]).sum(-1).bfloat16()
        if activated:
            lo, hi = a.float().chunk(2, dim=-1)
            activation = (lo * 0.25 - hi * 0.5).bfloat16()
        else:
            activation = a
        y = (
            (down[route.long()].float() * activation.float()[:, :, None, :]).sum(-1)
            * scores[:, :, None]
        ).bfloat16()
        for actual, expected in (
            (mid, a.flatten(0, 1)),
            (ids, route),
            (weights, scores),
            (result, y),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_warp_row_dot_replays_selected_rows_with_independent_base_alignment():
    if torch.cuda.get_device_capability(DEVICE) < (8, 0):
        pytest.skip("BF16 warp MMA requires SM80+")
    rows, k, banks = 64, 768, 3
    rng = torch.Generator(device=DEVICE).manual_seed(42)
    dtype = torch.bfloat16
    # Small signed integers make every BF16 product and FP32 sum exact,
    # including both halves of each stripe and all four complete stripes.
    weights = torch.randint(
        -6, 7, (banks * rows * k + 1,), device=DEVICE, generator=rng
    ).to(dtype)
    x = torch.randint(-7, 8, (k,), device=DEVICE, generator=rng).to(dtype)
    activation = torch.empty(k + 1, dtype=dtype, device=DEVICE)
    bank = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    sentinel = 1234567.0
    storage = torch.full((rows + 2,), sentinel, device=DEVICE)
    out = storage[1:-1]
    for w_offset, v_offset, unroll in (
        (0, 0, False),
        (0, 0, True),
        (1, 0, False),
        (0, 1, True),
    ):
        weight = weights.narrow(0, w_offset, banks * rows * k).view(banks * rows, k)
        published = activation.narrow(0, v_offset, k)
        assert weight.data_ptr() % 16 == 2 * w_offset
        assert published.data_ptr() % 16 == 2 * v_offset
        args = (x, weight, bank, published, out)
        config = selected_rows_config(unroll=unroll)
        call = selected_rows_after_producer.bind(args).compile_config(config)
        call(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(*args)
        if (w_offset, v_offset, unroll) == (0, 0, False):
            primary = graph, weight, args
        for index in (0, banks - 1, -1, banks):
            bank.fill_(index)
            x.neg_()
            published.fill_(float("nan"))
            out.fill_(float("nan"))
            graph.replay()
            expected_x = x.float() * 2
            if 0 <= index < banks:
                selected = weight.view(banks, rows, k)[index].float()
                expected = (selected * expected_x).sum(-1)
            else:
                # The DSL indexes the flattened matrix with a tensor of
                # physical row numbers. Its fragment bounds guard rejects
                # both negative and past-the-end rows in either half stripe.
                expected = torch.zeros_like(out)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(published.float(), expected_x, rtol=0, atol=0)
            assert storage[0].item() == storage[-1].item() == sentinel

    # Two nonzero terms enter distinct MMA instructions. FP32 rounds their
    # sum to 2**24 + 4: merely grouping exact small integers wouldn't detect
    # a different instruction-level rounding policy.
    graph, weight, aligned_args = primary
    bank.zero_()
    x.fill_(0.5)
    weight.zero_()
    weight[:, 0] = 2**24
    weight[:, 32] = 3
    graph.replay()
    expected = torch.full_like(out, 2**24 + 4)
    torch.testing.assert_close(out, expected, rtol=0, atol=2)
    strict = helion.kernel(
        selected_rows_after_producer.fn,
        backend="cute",
        static_shapes=True,
        fast_math=False,
        autotune_effort="none",
    ).bind(aligned_args)
    assert not strict.config_spec.cute_warp_row_dot_roots
    strict_floor = helion.Config(
        block_sizes=[256, 32, 32],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
    )
    strict.compile_config(strict_floor)(*aligned_args)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_ordered_columns_cover_each_full_pair_and_scalar_tail() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("packed FP32 column sums require SM10x")
    slots, columns = 4, 384
    cfg = helion.Config(
        block_sizes=[128, 256],
        num_threads=[128, 4, 1],
        reduction_loops=[None],
        cute_ordered_column_pairs=True,
    )
    # The second placement makes both real BF16 device arguments two-byte
    # aligned. Its first tile must use the same scalar fallback as its tail.
    for offset in (0, 1):
        inputs = torch.empty(slots * columns + 4, device=DEVICE, dtype=torch.bfloat16)
        storage = torch.full((columns + 8,), 123.0, device=DEVICE, dtype=torch.bfloat16)
        x = inputs[offset : offset + slots * columns].reshape(1, slots, columns)
        out = storage[2 + offset : 2 + offset + columns].view(1, columns)
        p = torch.arange(128, device=DEVICE, dtype=torch.float32)
        assert x.data_ptr() % 4 == out.data_ptr() % 4 == 2 * offset
        args = (x, p, out)
        call = ordered_column_pool.bind(args).compile_config(cfg)
        x.zero_()
        call(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y, peer = call(*args)
        for sign in (1, -1):
            for ix, number in enumerate((2.0**26, 1, -(2.0**26), 2)):
                x[0, ix].fill_(sign * number)
            # Includes a BF16 halfway tie at 1, cancellation and both signs.
            x[0, 0, ::3] = 1
            x[0, 1, ::3] = 2.0**-8
            x[0, 2:, ::3] = 0
            out.fill_(float("nan"))
            p.add_(1)
            graph.replay()
            reference = torch.zeros_like(out, dtype=torch.float32)
            for slot in range(slots):
                reference.add_(x[:, slot, :].float())
            assert torch.equal(
                y.view(torch.int16), reference.bfloat16().view(torch.int16)
            )
            assert torch.equal(peer, p + 1)
            assert bool((storage[: 2 + offset] == 123).all()) and bool(
                (storage[2 + offset + columns :] == 123).all()
            )
